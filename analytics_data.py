import os
import base64
import tempfile
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
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
BIGQUERY_TABLE = os.getenv("BIGQUERY_TABLE")
TABLE_FQN = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"


DAYS_IN_YEAR = 365.0
DAYS_MONTH = 30.4
DAYS_QUARTER = 91.3

DYNAMIC_PERIOD_DEFS = [
    {"name": "Last date", "type": "days", "days": 1},
    {"name": "Week", "type": "days", "days": 7},
    {"name": "Last Month", "type": "days", "days": 30},
    {"name": "Last 3 months", "type": "days", "days": 91},
    {"name": "Last 6 months", "type": "days", "days": 183},
    {"name": "Year to Date (YTD)", "type": "ytd"},
    {"name": "Last 1 year", "type": "years", "years": 1},
    {"name": "Last 2 years", "type": "years", "years": 2},
    {"name": "Last 3 years", "type": "years", "years": 3},
    {"name": "Last 5 years", "type": "years", "years": 5},
    {"name": "Last 10 years", "type": "years", "years": 10},
    {"name": "Last 15 years", "type": "years", "years": 15},
    {"name": "Last 20 years", "type": "years", "years": 20},
    {"name": "Last 25 years", "type": "years", "years": 25},
    {"name": "Last 50 years", "type": "years", "years": 50},
]

# Predefined start years as you gave earlier
PREDEFINED_START_YEARS = [2024, 2023, 2022, 2020, 2015, 2010, 2005, 2000, 1975]

# ------------------- BigQuery helpers -------------------

def get_max_date(client: bigquery.Client) -> date:
    q = f"SELECT MAX(Date) AS max_date FROM `{TABLE_FQN}`"
    df = client.query(q).to_dataframe()
    if df.empty or pd.isna(df.loc[0, "max_date"]):
        raise ValueError("No dates found in table.")
    md = df.loc[0, "max_date"]
    if isinstance(md, pd.Timestamp):
        return md.date()
    if isinstance(md, datetime):
        return md.date()
    return md

def fetch_price_series(client: bigquery.Client, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Return DataFrame with columns ['Date', 'Adj_Close'] for ticker between start_date and end_date inclusive.
    Prefers adjusted close. Tries columns in order: `Adj Close`, Adj_close, then falls back to Close.
    """
    last_error = None
    df = None

    for adj_col in ["`Adj Close`", "Adj_close", None]:
        col_expr = f"{adj_col} as Adj_Close_raw" if adj_col else "NULL as Adj_Close_raw"
        sql = f"""
        SELECT Date, {col_expr}, Close as Close_raw
        FROM `{TABLE_FQN}`
        WHERE Ticker = @ticker
          AND Date BETWEEN @start_date AND @end_date
        ORDER BY Date
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            ]
        )
        try:
            df = client.query(sql, job_config=job_config).to_dataframe()
            break
        except Exception as e:
            last_error = e
            continue

    if df is None:
        # Surface the last query error if all attempts failed
        raise last_error or RuntimeError("Failed to fetch price series")

    if df.empty:
        return pd.DataFrame(columns=["Date", "Adj_Close"])

    # Prefer adjusted close if present, otherwise use close
    if "Adj_Close_raw" in df.columns:
        df["Adj_Close"] = df["Adj_Close_raw"].combine_first(df["Close_raw"]) if "Close_raw" in df.columns else df["Adj_Close_raw"]
    else:
        df["Adj_Close"] = df["Close_raw"] if "Close_raw" in df.columns else None

    df = df[["Date", "Adj_Close"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def price_on_or_before(df: pd.DataFrame, target: date) -> Tuple[Optional[date], Optional[float]]:
    """
    Return (date_found, price) for latest date <= target in df.
    If none exists, returns (None, None).
    """
    if df.empty:
        return (None, None)
    df_before = df[df["Date"] <= target]
    if df_before.empty:
        return (None, None)
    row = df_before.iloc[-1]
    return (row["Date"], float(row["Adj_Close"]) if pd.notna(row["Adj_Close"]) else None)

# ------------------- Calculation utilities -------------------

def years_between(start: date, end: date) -> float:
    return round(((end - start).days) / DAYS_IN_YEAR, 3)

def pct_return(start_price: float, end_price: float) -> Optional[float]:
    if start_price is None or start_price == 0:
        return None
    return round(((end_price - start_price) / start_price) * 100, 2)

# ------------------- Calculators -------------------

def calculate_dynamic_periods(client: bigquery.Client, ticker: str, end_date: date) -> List[Dict[str, Any]]:
    # Fetch extended history to cover longest needed period (50 years)
    longest_years = max([p.get("years", 0) for p in DYNAMIC_PERIOD_DEFS if p.get("type") == "years"] + [0])
    earliest_needed = end_date - timedelta(days=int((longest_years + 1) * DAYS_IN_YEAR))
    prices = fetch_price_series(client, ticker, earliest_needed, end_date)

    results = []
    for p in DYNAMIC_PERIOD_DEFS:
        name = p["name"]
        if p["type"] == "point":
            start_req = end_date
        elif p["type"] == "days":
            start_req = end_date - timedelta(days=p["days"])
        elif p["type"] == "years":
            start_req = end_date - timedelta(days=int(p["years"] * DAYS_IN_YEAR))
        elif p["type"] == "ytd":
            start_req = date(end_date.year, 1, 1)
        else:
            start_req = end_date

        start_found, start_price = price_on_or_before(prices, start_req)
        end_found, end_price = price_on_or_before(prices, end_date)

        yrs = years_between(start_req, end_date)
        price_diff = None
        total_return = None
        if start_price is not None and end_price is not None:
            price_diff = round(end_price - start_price, 4)
            total_return = pct_return(start_price, end_price)

        results.append({
            "period": name,
            "start_date_requested": start_req.isoformat(),
            "start_date_found": start_found.isoformat() if start_found else None,
            "end_date_requested": end_date.isoformat(),
            "end_date_found": end_found.isoformat() if end_found else None,
            "years": yrs,
            "start_price": None if start_price is None else round(start_price, 4),
            "end_price": None if end_price is None else round(end_price, 4),
            "price_difference": price_diff,
            "total_return_pct": total_return,
         
        })
    return results

def calculate_predefined_periods(client: bigquery.Client, ticker: str, end_date: date) -> List[Dict[str, Any]]:
    # Predefined periods use Jan 1 of the start year up to end_date (per your instruction)
    earliest = date(min(PREDEFINED_START_YEARS) - 1, 1, 1)
    prices = fetch_price_series(client, ticker, earliest, end_date)
    results = []
    for s_year in PREDEFINED_START_YEARS:
        start_req = date(s_year, 1, 1)
        end_req = end_date
        start_found, start_price = price_on_or_before(prices, start_req)
        end_found, end_price = price_on_or_before(prices, end_req)
        yrs = years_between(start_req, end_req)
        price_diff = None
        total_return = None
        if start_price is not None and end_price is not None:
            price_diff = round(end_price - start_price, 4)
            total_return = pct_return(start_price, end_price)
        results.append({
            "period": f"{s_year} - {end_req.year}",
            "start_date_requested": start_req.isoformat(),
            "start_date_found": start_found.isoformat() if start_found else None,
            "end_date_requested": end_req.isoformat(),
            "end_date_found": end_found.isoformat() if end_found else None,
            "years": round(yrs),
            "start_price": None if start_price is None else round(start_price, 4),
            "end_price": None if end_price is None else round(end_price, 4),
            "price_difference": price_diff,
            "total_return_pct": total_return,
        })
    return results

def calculate_annual_returns(client: bigquery.Client, ticker: str, end_date: date, from_year: int = 1970) -> List[Dict[str, Any]]:
    prices = fetch_price_series(client, ticker, date(from_year, 1, 1), end_date)
    results = []
    for yr in range(from_year, end_date.year + 1):
        start_req = date(yr, 1, 1)
        # for the active year, use end_date; for past years use Dec 31
        if yr == end_date.year:
            end_req = end_date
        else:
            end_req = date(yr, 12, 31)
        start_found, start_price = price_on_or_before(prices, start_req)
        end_found, end_price = price_on_or_before(prices, end_req)
        yrs = years_between(start_req, end_req)
        price_diff = None
        total_return = None
        if start_price is not None and end_price is not None:
            price_diff = round(end_price - start_price, 4)
            total_return = pct_return(start_price, end_price)
        results.append({
            "year": yr,
            "start_date_requested": start_req.isoformat(),
            "start_date_found": start_found.isoformat() if start_found else None,
            "end_date_requested": end_req.isoformat(),
            "end_date_found": end_found.isoformat() if end_found else None,
            "years": yrs,
            "start_price": None if start_price is None else round(start_price, 4),
            "end_price": None if end_price is None else round(end_price, 4),
            "price_difference": price_diff,
            "total_return_pct": total_return,
        })
    return results

def calculate_custom_range(client: bigquery.Client, ticker: str, start_date: date, end_date: date) -> Dict[str, Any]:
    prices = fetch_price_series(client, ticker, start_date - timedelta(days=5), end_date)
    start_found, start_price = price_on_or_before(prices, start_date)
    end_found, end_price = price_on_or_before(prices, end_date)
    yrs = years_between(start_date, end_date)
    price_diff = None
    total_return = None
    if start_price is not None and end_price is not None:
        price_diff = round(end_price - start_price, 4)
        total_return = pct_return(start_price, end_price)
    return {
        "start_date_requested": start_date.isoformat(),
        "start_date_found": start_found.isoformat() if start_found else None,
        "end_date_requested": end_date.isoformat(),
        "end_date_found": end_found.isoformat() if end_found else None,
        "years": yrs,
        "start_price": None if start_price is None else round(start_price, 4),
        "end_price": None if end_price is None else round(end_price, 4),
        "price_difference": price_diff,
        "total_return_pct": total_return,
    }

# ---------- ms ------------------

def calculate_monthly_returns(client: bigquery.Client, ticker: str, end_date: date, from_year: int = 1970) -> List[Dict[str, Any]]:
    prices = fetch_price_series(client, ticker, date(from_year, 1, 1), end_date)
    if prices.empty:
        return []

    prices['Date'] = pd.to_datetime(prices['Date'])
    prices.set_index('Date', inplace=True)
    monthly = prices['Adj_Close'].resample('M').last()

    results = []
    for i in range(1, len(monthly)):
        start_date = monthly.index[i - 1].date()
        end_date_ = monthly.index[i].date()
        start_price = monthly.iloc[i - 1]
        end_price = monthly.iloc[i]
        years = years_between(start_date, end_date_)
        price_diff = round(end_price - start_price, 4)
        total_return = pct_return(start_price, end_price)
        results.append({
            "month": end_date_.strftime("%Y-%m"),
            "start_date_requested": start_date.isoformat(),
            "start_date_found": start_date.isoformat(),
            "end_date_requested": end_date_.isoformat(),
            "end_date_found": end_date_.isoformat(),
            "years": years,
            "start_price": round(start_price, 4),
            "end_price": round(end_price, 4),
            "price_difference": price_diff,
            "total_return_pct": total_return,
        })
    return results


def calculate_quarterly_returns(client: bigquery.Client, ticker: str, end_date: date, from_year: int = 1970) -> List[Dict[str, Any]]:
    prices = fetch_price_series(client, ticker, date(from_year, 1, 1), end_date)
    if prices.empty:
        return []

    prices['Date'] = pd.to_datetime(prices['Date'])
    prices.set_index('Date', inplace=True)
    quarterly = prices['Adj_Close'].resample('Q').last()

    results = []
    for i in range(1, len(quarterly)):
        start_date = quarterly.index[i - 1].date()
        end_date_ = quarterly.index[i].date()
        start_price = quarterly.iloc[i - 1]
        end_price = quarterly.iloc[i]
        years = years_between(start_date, end_date_)
        price_diff = round(end_price - start_price, 4)
        total_return = pct_return(start_price, end_price)
        results.append({
            "quarter": f"{end_date_.year}-Q{((end_date_.month - 1)//3) + 1}",
            "start_date_requested": start_date.isoformat(),
            "start_date_found": start_date.isoformat(),
            "end_date_requested": end_date_.isoformat(),
            "end_date_found": end_date_.isoformat(),
            "years": years,
            "start_price": round(start_price, 4),
            "end_price": round(end_price, 4),
            "price_difference": price_diff,
            "total_return_pct": total_return,
        })
    return results

# ---------- ms ------------------



# ------------------- Main public function -------------------

def calculate_all_returns(ticker: str,
                          include_predefined: bool = True,
                          include_annual: bool = True,
                          custom_range: Optional[Tuple[date, date]] = None,
                          annual_from_year: int = 2010) -> Dict[str, Any]:
    client = get_bigquery_client()
    end_date = get_max_date(client)
    t = ticker.upper()

    dynamic = calculate_dynamic_periods(client, t, end_date)
    predefined = calculate_predefined_periods(client, t, end_date) if include_predefined else []
    annual = calculate_annual_returns(client, t, end_date, from_year=annual_from_year) if include_annual else []
    monthly = calculate_monthly_returns(client, t, end_date, from_year=annual_from_year)
    quarterly = calculate_quarterly_returns(client, t, end_date, from_year=annual_from_year)
    custom = None
    if custom_range:
        custom = calculate_custom_range(client, t, custom_range[0], custom_range[1])

    return {
        "ticker": t,
        "as_of_date": end_date.isoformat(),
        "dynamic_periods": dynamic,
        "predefined_periods": predefined,
        "annual_returns": annual,
        "quarterly_returns": quarterly,
        "monthly_returns": monthly,
        "custom_range": custom
    }





# ------------------- Example runner -------------------
if __name__ == "__main__":
    # Quick example: ensure env vars set (ACCESS_TOKEN base64, PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE)
    # Replace "AAPL" with the frontend-provided ticker.
    out = calculate_all_returns("AAPL", custom_range=None, annual_from_year=2018)
    import json
    print(json.dumps(out, indent=2, default=str))
