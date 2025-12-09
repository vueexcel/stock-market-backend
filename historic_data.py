import os
import base64
import tempfile
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from calendar import monthrange


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

def get_monthly_ohlc(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetches OHLC data for a ticker from BigQuery and aggregates it by month.
    If start_date or end_date are not provided, they are automatically fetched from BigQuery.
    """
    client = get_bigquery_client()

    # --- Step 1: Auto-fetch start and end dates if missing ---
    if not start_date or not end_date:
        date_query = f"""
            SELECT 
                MIN(Date) AS min_date, 
                MAX(Date) AS max_date
            FROM `{TABLE_FQN}`
            WHERE Ticker = @ticker
        """
        date_job = client.query(
            date_query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper())
                ]
            )
        )
        date_df = date_job.to_dataframe()
        if date_df.empty or pd.isna(date_df.loc[0, "min_date"]) or pd.isna(date_df.loc[0, "max_date"]):
            raise ValueError(f"No date range found for ticker '{ticker}' in table {TABLE_FQN}")

        # Fill missing arguments
        if not start_date:
            start_date = date_df.loc[0, "min_date"].strftime("%Y-%m-%d")
        if not end_date:
            end_date = date_df.loc[0, "max_date"].strftime("%Y-%m-%d")

    # --- Step 2: Main monthly OHLC query ---
    query = f"""
    WITH stock_data AS (
        SELECT
            Ticker,
            Date,
            Open,
            High,
            Low,
            Close,
            `Adj Close`,
            EXTRACT(YEAR FROM Date) AS year,
            EXTRACT(MONTH FROM Date) AS month
        FROM `{TABLE_FQN}`
        WHERE Ticker = @ticker
          AND Date BETWEEN @start_date AND @end_date
    ),
    monthly_summary AS (
        SELECT
            Ticker,
            year,
            month,
            ARRAY_AGG(Open ORDER BY Date ASC LIMIT 1)[OFFSET(0)] AS Open,
            MAX(High) AS High,
            MIN(Low) AS Low,
            ARRAY_AGG(Close ORDER BY Date DESC LIMIT 1)[OFFSET(0)] AS Close,
            ARRAY_AGG(`Adj Close` ORDER BY Date DESC LIMIT 1)[OFFSET(0)] AS `Adj Close`
        FROM stock_data
        GROUP BY Ticker, year, month
    )
    SELECT *
    FROM monthly_summary
    ORDER BY year, month
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper()),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )

    results = client.query(query, job_config=job_config).result()
    data = []
    for row in results:
        year = int(row["year"])
        month = int(row["month"])
        last_day = monthrange(year, month)[1]
        data.append({
            "ticker": row["Ticker"],
            "year": year,
            "month": month,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "adj_close": float(row["Adj Close"]),
            "start_date": f"{year}-{month:02d}-01",
            "end_date": f"{year}-{month:02d}-{last_day:02d}"
        })

    # data = [
    #     {
    #         "ticker": row["Ticker"],
    #         "year": int(row["year"]),
    #         "month": int(row["month"]),
    #         "open": float(row["Open"]),
    #         "high": float(row["High"]),
    #         "low": float(row["Low"]),
    #         "close": float(row["Close"]),
    #         "start_date": start_date,
    #         "end_date": end_date
    #     }
    #     for row in results
    # ]

    print(f"Fetched monthly OHLC for {ticker} from {start_date} to {end_date}")
    return data


    """
    Fetches OHLC data for a ticker from BigQuery and aggregates it by month.
    """
    client = get_bigquery_client()
    
    query = f"""
    WITH stock_data AS (
        SELECT
            Ticker,
            Date,
            Open,
            High,
            Low,
            Close,
            Adj Close,
            EXTRACT(YEAR FROM Date) AS year,
            EXTRACT(MONTH FROM Date) AS month
        FROM `{TABLE_FQN}`
        WHERE Ticker = @ticker
        {f"AND Date >= '{start_date}'" if start_date else ""}
        {f"AND Date <= '{end_date}'" if end_date else ""}
    ),
    monthly_summary AS (
        SELECT
            Ticker,
            year,
            month,
            ARRAY_AGG(Open ORDER BY Date ASC LIMIT 1)[OFFSET(0)] AS Open,
            MAX(High) AS High,
            MIN(Low) AS Low,
            ARRAY_AGG(Close ORDER BY Date DESC LIMIT 1)[OFFSET(0)] AS Close
        FROM stock_data
        GROUP BY Ticker, year, month
    )
    SELECT *
    FROM monthly_summary
    ORDER BY year, month
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper())
        ]
    )
    
    query_job = client.query(query, job_config=job_config)
    results = query_job.result()
    
    return [
        {
            "ticker": row["Ticker"],
            "year": int(row["year"]),
            "month": int(row["month"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "adj_close": float(row["Adj Close"]),
        }
        for row in results
    ]