import os
import base64
import tempfile
from datetime import date, timedelta, datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


def write_service_account_to_temp(env_var_name: str) -> str:
    b64_json = os.getenv(env_var_name)
    if not b64_json:
        raise ValueError(f"Missing base64 service account in env: {env_var_name}")
    decoded = base64.b64decode(b64_json)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(decoded)
    tmp.flush()
    return tmp.name

SERVICE_ACCOUNT_ENV = "ACCESS_TOKEN"
SERVICE_ACCOUNT_FILE = write_service_account_to_temp(SERVICE_ACCOUNT_ENV)


def get_bigquery_client() -> bigquery.Client:
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    client = bigquery.Client(credentials=creds, project=creds.project_id)
    return client


PROJECT_ID = os.getenv("PROJECT_ID")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET")
TICKER_DETAILS_TABLE = os.getenv("TICKER_DETAILS_TABLE", "TickerDetails")
TICKER_DETAILS_FQN = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{TICKER_DETAILS_TABLE}"

# Use the same table as analytics_data for price data
BIGQUERY_TABLE = os.getenv("BIGQUERY_TABLE")
TABLE_FQN = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"

DAYS_IN_YEAR = 365.0


def get_unique_indices() -> List[Dict[str, str]]:
    """
    Fetches unique Index values from the TickerDetails table in BigQuery.
    Returns a list of dictionaries with 'value' and 'label' keys for the frontend.
    """
    client = get_bigquery_client()
    
    query = f"""
        SELECT DISTINCT Index
        FROM `{TICKER_DETAILS_FQN}`
        WHERE Index IS NOT NULL
        ORDER BY Index
    """
    
    try:
        df = client.query(query).to_dataframe()
        
        if df.empty:
            # Return default options if no data found
            return [
                {"value": "sp500", "label": "S&P 500"},
                {"value": "nasdaq100", "label": "Nasdaq 100"},
                {"value": "dowjones", "label": "Dow Jones"},
            ]
        
        # Map the Index values to frontend-friendly format
        # Create a mapping for known index patterns
        index_mapping = {}
        
        for index_value in df['Index'].unique():
            if pd.isna(index_value):
                continue
            
            index_str = str(index_value).strip()
            
            # Normalize index names to match frontend expectations
            if "S&P" in index_str or "SP" in index_str or ("500" in index_str and "S&P" not in index_str):
                if "sp500" not in index_mapping:
                    index_mapping["sp500"] = "S&P 500"
            elif "Nasdaq" in index_str and "100" in index_str:
                if "nasdaq100" not in index_mapping:
                    index_mapping["nasdaq100"] = "Nasdaq 100"
            elif "Dow" in index_str and ("30" in index_str or "Jones" in index_str):
                if "dowjones" not in index_mapping:
                    index_mapping["dowjones"] = "Dow Jones"
            else:
                # For other indices, use the original value
                value_key = index_str.lower().replace(" ", "-").replace(".", "")
                if value_key not in index_mapping:
                    index_mapping[value_key] = index_str
        
        # Convert mapping to list format
        unique_indices = [{"value": k, "label": v} for k, v in index_mapping.items()]
        
        # Sort to ensure consistent order (defaults first, then others)
        default_order = {"sp500": 1, "nasdaq100": 2, "dowjones": 3}
        unique_indices.sort(key=lambda x: (default_order.get(x["value"], 999), x["label"]))
        
        # Ensure we have at least the default options if no data found
        if len(unique_indices) == 0:
            return [
                {"value": "sp500", "label": "S&P 500"},
                {"value": "nasdaq100", "label": "Nasdaq 100"},
                {"value": "dowjones", "label": "Dow Jones"},
                {"value": "others", "label": "Others"},
            ]
        
        # Add "Others" if not present and we don't have all defaults
        default_values = {"sp500", "nasdaq100", "dowjones"}
        has_all_defaults = all(any(idx["value"] == dv for idx in unique_indices) for dv in default_values)
        
        if not has_all_defaults and not any(idx["value"] == "others" for idx in unique_indices):
            unique_indices.append({"value": "others", "label": "Others"})
        
        return unique_indices
        
    except Exception as e:
        print(f"Error fetching indices from TickerDetails: {e}")
        # Return default options on error
        return [
            {"value": "sp500", "label": "S&P 500"},
            {"value": "nasdaq100", "label": "Nasdaq 100"},
            {"value": "dowjones", "label": "Dow Jones"},
            {"value": "others", "label": "Others"},
        ]


def get_period_options() -> List[Dict[str, str]]:
    """
    Returns period options for the Performance Return page.
    Based on the DYNAMIC_PERIOD_DEFS structure from analytics_data.py
    """
    period_defs = [
        {"name": "Last date", "value": "last-date"},
        {"name": "Week", "value": "week"},
        {"name": "Last Month", "value": "last-month"},
        {"name": "Last 3 months", "value": "last-3-months"},
        {"name": "Last 6 months", "value": "last-6-months"},
        {"name": "Year to Date (YTD)", "value": "ytd"},
        {"name": "Last 1 year", "value": "last-1-year"},
        {"name": "Last 2 years", "value": "last-2-years"},
        {"name": "Last 3 years", "value": "last-3-years"},
        {"name": "Last 5 years", "value": "last-5-years"},
        {"name": "Last 10 years", "value": "last-10-years"},
    ]
    
    return [{"value": p["value"], "label": p["name"]} for p in period_defs]


def map_index_value_to_db_index(index_value: str) -> str:
    """
    Maps frontend index value (e.g., 'dowjones') back to database Index value (e.g., 'Dow Jones 30').
    This is the reverse mapping of what we do in get_unique_indices.
    """
    # Get all unique indices from the database
    client = get_bigquery_client()
    query = f"""
        SELECT DISTINCT Index
        FROM `{TICKER_DETAILS_FQN}`
        WHERE Index IS NOT NULL
        ORDER BY Index
    """
    
    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            return index_value
        
        # Find matching index in database based on the frontend value
        for db_index in df['Index'].unique():
            if pd.isna(db_index):
                continue
            
            db_index_str = str(db_index).strip()
            
            # Match based on the same logic used in get_unique_indices
            if index_value == "sp500":
                if "S&P" in db_index_str or "SP" in db_index_str or ("500" in db_index_str and "S&P" not in db_index_str):
                    return db_index_str
            elif index_value == "nasdaq100":
                if "Nasdaq" in db_index_str and "100" in db_index_str:
                    return db_index_str
            elif index_value == "dowjones":
                if "Dow" in db_index_str and ("30" in db_index_str or "Jones" in db_index_str):
                    return db_index_str
            else:
                # For other indices, check if the value matches
                value_key = db_index_str.lower().replace(" ", "-").replace(".", "")
                if value_key == index_value:
                    return db_index_str
        
        return index_value
    except Exception as e:
        print(f"Error mapping index value: {e}")
        return index_value


def calculate_period_dates(period_value: str) -> Tuple[date, date]:
    """
    Calculates start_date and end_date based on the period value.
    Returns (start_date, end_date) tuple.
    """
    # Get the maximum date from the table as end_date
    client = get_bigquery_client()
    query = f"SELECT MAX(Date) AS max_date FROM `{TABLE_FQN}`"
    df = client.query(query).to_dataframe()
    if df.empty or pd.isna(df.loc[0, "max_date"]):
        end_date = date.today()
    else:
        max_date = df.loc[0, "max_date"]
        if isinstance(max_date, pd.Timestamp):
            end_date = max_date.date()
        elif isinstance(max_date, datetime):
            end_date = max_date.date()
        else:
            end_date = max_date
    
    # Calculate start_date based on period
    period_map = {
        "last-date": timedelta(days=1),
        "week": timedelta(days=7),
        "last-month": timedelta(days=30),
        "last-3-months": timedelta(days=91),
        "last-6-months": timedelta(days=183),
        "ytd": None,  # Special case - year to date
        "last-1-year": timedelta(days=int(1 * DAYS_IN_YEAR)),
        "last-2-years": timedelta(days=int(2 * DAYS_IN_YEAR)),
        "last-3-years": timedelta(days=int(3 * DAYS_IN_YEAR)),
        "last-5-years": timedelta(days=int(5 * DAYS_IN_YEAR)),
        "last-10-years": timedelta(days=int(10 * DAYS_IN_YEAR)),
    }
    
    if period_value == "ytd":
        start_date = date(end_date.year, 1, 1)
    elif period_value in period_map:
        start_date = end_date - period_map[period_value]
    else:
        # Default to 1 year
        start_date = end_date - timedelta(days=int(1 * DAYS_IN_YEAR))
    
    return start_date, end_date


def fetch_all_ticker_prices(client: bigquery.Client, ticker_symbols: List[str], extended_start: date, end_date: date) -> pd.DataFrame:
    """
    Fetches price data for all tickers in a single batch query (optimized).
    Returns a DataFrame with columns: Date, Ticker, Adj_Close
    """
    if not ticker_symbols:
        return pd.DataFrame()
    
    last_error = None
    df = None
    
    for adj_col in ["`Adj Close`", "Adj_close", None]:
        col_expr = f"{adj_col} as Adj_Close_raw" if adj_col else "NULL as Adj_Close_raw"
        sql = f"""
        SELECT Date, Ticker, {col_expr}, Close as Close_raw
        FROM `{TABLE_FQN}`
        WHERE Ticker IN UNNEST(@tickers)
          AND Date BETWEEN @extended_start AND @end_date
        ORDER BY Ticker, Date
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("tickers", "STRING", ticker_symbols),
                bigquery.ScalarQueryParameter("extended_start", "DATE", extended_start),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            ]
        )
        try:
            df = client.query(sql, job_config=job_config).to_dataframe()
            break
        except Exception as e:
            last_error = e
            print(f"Error fetching batch price data: {e}")
            continue
    
    if df is None or df.empty:
        print(f"No price data found for tickers between {extended_start} and {end_date}")
        return pd.DataFrame()
    
    # Convert Date column to date type
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    # Prefer adjusted close if present, otherwise use close
    if "Adj_Close_raw" in df.columns:
        df["Adj_Close"] = df["Adj_Close_raw"].combine_first(df["Close_raw"]) if "Close_raw" in df.columns else df["Adj_Close_raw"]
    else:
        df["Adj_Close"] = df["Close_raw"] if "Close_raw" in df.columns else None
    
    # Sort by ticker and date
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    
    return df


def get_prices_from_batch_data(price_data: pd.DataFrame, ticker: str, start_date: date, end_date: date) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts start and end prices for a specific ticker from batch-fetched price data.
    Returns (start_price, end_price) tuple.
    """
    if price_data.empty:
        return (None, None)
    
    # Filter data for this specific ticker
    ticker_data = price_data[price_data["Ticker"].str.upper() == ticker.upper()].copy()
    
    if ticker_data.empty:
        return (None, None)
    
    # Get start price (latest available price on or before start_date)
    start_prices = ticker_data[ticker_data["Date"] <= start_date]
    if not start_prices.empty:
        start_price = float(start_prices.iloc[-1]["Adj_Close"]) if pd.notna(start_prices.iloc[-1]["Adj_Close"]) else None
    else:
        # If no price before start_date, use first available price
        if not ticker_data.empty:
            start_price = float(ticker_data.iloc[0]["Adj_Close"]) if pd.notna(ticker_data.iloc[0]["Adj_Close"]) else None
        else:
            start_price = None
    
    # Get end price (latest available price on or before end_date)
    end_prices = ticker_data[ticker_data["Date"] <= end_date]
    if not end_prices.empty:
        end_price = float(end_prices.iloc[-1]["Adj_Close"]) if pd.notna(end_prices.iloc[-1]["Adj_Close"]) else None
    else:
        # If no price before end_date, use last available price
        if not ticker_data.empty:
            end_price = float(ticker_data.iloc[-1]["Adj_Close"]) if pd.notna(ticker_data.iloc[-1]["Adj_Close"]) else None
        else:
            end_price = None
    
    return (start_price, end_price)


def fetch_price_series_for_ticker(client: bigquery.Client, ticker: str, start_date: date, end_date: date) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetches start and end prices for a ticker in the given date range.
    Returns (start_price, end_price) tuple.
    Uses the same logic as analytics_data.py to prefer Adj Close.
    Fetches a wider date range to ensure we can find prices on or before the requested dates.
    """
    last_error = None
    df = None
    
    # Extend the date range by a few days before start_date to ensure we can find prices
    # This is similar to how analytics_data.py handles it
    extended_start = start_date - timedelta(days=30)  # Look back 30 days to find nearest price
    
    for adj_col in ["`Adj Close`", "Adj_close", None]:
        col_expr = f"{adj_col} as Adj_Close_raw" if adj_col else "NULL as Adj_Close_raw"
        sql = f"""
        SELECT Date, {col_expr}, Close as Close_raw
        FROM `{TABLE_FQN}`
        WHERE Ticker = @ticker
          AND Date BETWEEN @extended_start AND @end_date
        ORDER BY Date
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper()),
                bigquery.ScalarQueryParameter("extended_start", "DATE", extended_start),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            ]
        )
        try:
            df = client.query(sql, job_config=job_config).to_dataframe()
            break
        except Exception as e:
            last_error = e
            print(f"Error fetching price data for {ticker}: {e}")
            continue
    
    if df is None or df.empty:
        print(f"No price data found for ticker {ticker} between {extended_start} and {end_date}")
        return (None, None)
    
    # Convert Date column to date type
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    # Prefer adjusted close if present, otherwise use close
    if "Adj_Close_raw" in df.columns:
        df["Adj_Close"] = df["Adj_Close_raw"].combine_first(df["Close_raw"]) if "Close_raw" in df.columns else df["Adj_Close_raw"]
    else:
        df["Adj_Close"] = df["Close_raw"] if "Close_raw" in df.columns else None
    
    # Sort by date to ensure proper ordering
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Get start price (latest available price on or before start_date)
    # This matches the price_on_or_before logic from analytics_data.py
    start_prices = df[df["Date"] <= start_date]
    if not start_prices.empty:
        start_price = float(start_prices.iloc[-1]["Adj_Close"]) if pd.notna(start_prices.iloc[-1]["Adj_Close"]) else None
    else:
        # If no price before start_date, use first available price
        if not df.empty:
            start_price = float(df.iloc[0]["Adj_Close"]) if pd.notna(df.iloc[0]["Adj_Close"]) else None
        else:
            start_price = None
    
    # Get end price (latest available price on or before end_date)
    end_prices = df[df["Date"] <= end_date]
    if not end_prices.empty:
        end_price = float(end_prices.iloc[-1]["Adj_Close"]) if pd.notna(end_prices.iloc[-1]["Adj_Close"]) else None
    else:
        # If no price before end_date, use last available price
        if not df.empty:
            end_price = float(df.iloc[-1]["Adj_Close"]) if pd.notna(df.iloc[-1]["Adj_Close"]) else None
        else:
            end_price = None
    
    return (start_price, end_price)


def calculate_total_return_percentage(start_price: Optional[float], end_price: Optional[float]) -> Optional[float]:
    """
    Calculates total return percentage.
    Returns None if start_price is None or 0.
    """
    if start_price is None or start_price == 0:
        return None
    if end_price is None:
        return None
    return round(((end_price - start_price) / start_price) * 100, 2)


def get_ticker_details_by_index(index_value: str, period_value: str) -> List[Dict[str, Any]]:
    """
    Fetches ticker details from TickerDetails table filtered by Index.
    Also calculates total return percentage for each ticker based on the period.
    Maps the frontend index value to the database Index value.
    """
    client = get_bigquery_client()
    
    # Map frontend index value to database Index value
    db_index = map_index_value_to_db_index(index_value)
    
    # Calculate date range based on period
    start_date, end_date = calculate_period_dates(period_value)
    
    query = f"""
        SELECT 
            Symbol,
            Security,
            Sector,
            Industry,
            Index
        FROM `{TICKER_DETAILS_FQN}`
        WHERE Index = @index_value
        ORDER BY Symbol
    """
    
    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("index_value", "STRING", db_index),
            ]
        )
        
        df = client.query(query, job_config=job_config).to_dataframe()
        
        if df.empty:
            return []
        
        # Get all ticker symbols
        ticker_symbols = [str(row["Symbol"]).upper() for _, row in df.iterrows() if pd.notna(row["Symbol"])]
        
        if not ticker_symbols:
            return []
        
        print(f"Fetching ticker details for index: {db_index}, period: {period_value}, date range: {start_date} to {end_date}")
        print(f"Processing {len(ticker_symbols)} tickers")
        
        # Fetch all price data for all tickers in a single query (optimized)
        extended_start = start_date - timedelta(days=30)
        price_data = fetch_all_ticker_prices(client, ticker_symbols, extended_start, end_date)
        
        # Convert DataFrame to list of dictionaries with row numbers and calculate total return
        result = []
        for idx, row in df.iterrows():
            symbol = str(row["Symbol"]).upper() if pd.notna(row["Symbol"]) else ""
            
            if not symbol:
                continue
            
            # Calculate total return percentage from the batch-fetched price data
            try:
                start_price, end_price = get_prices_from_batch_data(price_data, symbol, start_date, end_date)
                total_return_pct = calculate_total_return_percentage(start_price, end_price)
                
                if start_price is None or end_price is None:
                    print(f"Warning: Missing price data for {symbol} - start_price: {start_price}, end_price: {end_price}")
            except Exception as e:
                print(f"Error calculating return for {symbol}: {e}")
                total_return_pct = None
            
            result.append({
                "row": idx + 1,
                "symbol": symbol,
                "security": str(row["Security"]) if pd.notna(row["Security"]) else "",
                "sector": str(row["Sector"]) if pd.notna(row["Sector"]) else "",
                "industry": str(row["Industry"]) if pd.notna(row["Industry"]) else "",
                "index": str(row["Index"]) if pd.notna(row["Index"]) else "",
                "totalReturnPercentage": total_return_pct,
                "price": end_price,
            })
        
        print(f"Returning {len(result)} ticker details with total return calculations")
        return result
        
    except Exception as e:
        print(f"Error fetching ticker details by index: {e}")
        return []