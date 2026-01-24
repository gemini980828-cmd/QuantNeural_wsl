"""
Stooq Price Loader for QUANT-NEURAL.

Provides:
- load_stooq_daily_prices: Load daily prices from Stooq bulk CSV with PIT cutoff.
- resample_to_monthly: Aggregate daily bars to month-end OHLCV.

Point-in-Time (PIT) Rules:
- Only data with date <= as_of_date is included.
- No "now" or system time logic.
- No network calls; local file read only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_stooq_daily_prices(
    path: str,
    *,
    as_of_date: str,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Load daily prices from a Stooq bulk CSV with PIT cutoff.
    
    Parameters
    ----------
    path : str
        Path to the Stooq bulk CSV file.
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format. Only rows with date <= as_of_date
        are included.
    ticker : str | None
        If provided, filter to only this ticker.
    
    Returns
    -------
    pd.DataFrame
        Columns: ["date", "ticker", "open", "high", "low", "close", "volume"]
        Sorted by (ticker, date) ascending.
    
    Raises
    ------
    ValueError
        If result is empty, dates cannot be parsed, OHLC values are invalid,
        or data fails sanity checks.
    """
    # Parse as_of_date to datetime for comparison
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    
    # Read CSV with flexible header handling
    df = pd.read_csv(path)
    
    # Normalize column names to uppercase for matching
    df.columns = [c.strip().upper().replace("<", "").replace(">", "") for c in df.columns]
    
    # Required columns
    required_cols = ["TICKER", "PER", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Filter to PER == "D" (daily rows only)
    df = df[df["PER"].astype(str).str.strip().str.upper() == "D"].copy()
    
    if df.empty:
        raise ValueError("No daily (PER=D) rows found in input.")
    
    # Parse DATE column (YYYYMMDD format)
    df["date"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")
    
    # Validate no NaT dates
    if df["date"].isna().any():
        raise ValueError("Some dates could not be parsed (NaT detected).")
    
    # Standardize columns
    df["ticker"] = df["TICKER"].astype(str).str.strip()
    df["open"] = pd.to_numeric(df["OPEN"], errors="coerce").astype(float)
    df["high"] = pd.to_numeric(df["HIGH"], errors="coerce").astype(float)
    df["low"] = pd.to_numeric(df["LOW"], errors="coerce").astype(float)
    df["close"] = pd.to_numeric(df["CLOSE"], errors="coerce").astype(float)
    df["volume"] = pd.to_numeric(df["VOL"], errors="coerce").astype(float)
    
    # Filter by ticker if specified
    if ticker is not None:
        df = df[df["ticker"] == ticker].copy()
    
    # Apply PIT cutoff
    df = df[df["date"] <= as_of_dt].copy()
    
    # Validate non-empty after filtering
    if df.empty:
        raise ValueError(
            f"No data remaining after ticker filter and PIT cutoff (as_of_date={as_of_date})."
        )
    
    # Select and order columns
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
    
    # Validate OHLC values are finite
    ohlc_cols = ["open", "high", "low", "close"]
    for col in ohlc_cols:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values detected in {col} column.")
    
    # OHLC sanity checks per row
    # high must be >= max(open, close, low)
    max_vals = df[["open", "close", "low"]].max(axis=1)
    if (df["high"] < max_vals).any():
        raise ValueError("OHLC sanity violation: high < max(open, close, low)")
    
    # low must be <= min(open, close, high)
    min_vals = df[["open", "close", "high"]].min(axis=1)
    if (df["low"] > min_vals).any():
        raise ValueError("OHLC sanity violation: low > min(open, close, high)")
    
    # Sort by (ticker, date) ascending with stable sort for determinism
    df = df.sort_values(by=["ticker", "date"], kind="mergesort").reset_index(drop=True)
    
    # Remove duplicates, keeping last (after sorting)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    
    # Final sort (deterministic)
    df = df.sort_values(by=["ticker", "date"], kind="mergesort").reset_index(drop=True)
    
    # Validate monotonic dates within each ticker
    for t in df["ticker"].unique():
        t_dates = df.loc[df["ticker"] == t, "date"]
        if not t_dates.is_monotonic_increasing:
            raise ValueError(f"Dates are not monotonic increasing for ticker {t}")
    
    return df


def resample_to_monthly(
    daily: pd.DataFrame,
    *,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Resample daily OHLCV to month-end bars.
    
    Parameters
    ----------
    daily : pd.DataFrame
        Daily prices with columns: ["date", "ticker", "open", "high", "low", "close", "volume"]
    price_col : str
        Column to use for close (default "close"). Currently only "close" is meaningful
        but parameter maintained for API flexibility.
    
    Returns
    -------
    pd.DataFrame
        Monthly OHLCV with columns: ["date", "ticker", "open", "high", "low", "close", "volume"]
        where "date" is the month-end timestamp.
        Sorted by (ticker, date) ascending.
    
    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in daily.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Work with a copy
    df = daily.copy()
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Add month-end period for grouping
    df["month_end"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    
    # Aggregate per (ticker, month_end)
    result_rows = []
    
    for (ticker, month_end), group in df.groupby(["ticker", "month_end"], sort=True):
        # Sort group by date to ensure first/last are correct
        group = group.sort_values("date")
        
        row = {
            "date": month_end,
            "ticker": ticker,
            "open": group["open"].iloc[0],       # first open
            "high": group["high"].max(),          # max high
            "low": group["low"].min(),            # min low
            "close": group["close"].iloc[-1],     # last close
            "volume": group["volume"].sum(),      # sum volume
        }
        result_rows.append(row)
    
    result = pd.DataFrame(result_rows)
    
    # Handle empty case
    if result.empty:
        return pd.DataFrame(columns=required_cols)
    
    # Ensure correct column order
    result = result[["date", "ticker", "open", "high", "low", "close", "volume"]]
    
    # Enforce float dtype for all numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    result[numeric_cols] = result[numeric_cols].astype(float)
    
    # Sort by (ticker, date) ascending
    result = result.sort_values(by=["ticker", "date"], kind="mergesort").reset_index(drop=True)
    
    return result
