"""
Backtest Artifacts IO.

Provides deterministic IO helpers that write prices and scores DataFrames
to CSV files in formats compatible with run_scores_backtest_from_csv().

Design Principles:
- No external dependencies beyond numpy/pandas (standard project deps)
- Deterministic: stable sorting and deterministic CSV layout
- Fail-fast: raise clear exceptions on invalid inputs
"""

import numpy as np
import pandas as pd


def write_scores_csv(
    scores: pd.DataFrame,
    *,
    path: str,
    date_col: str = "date",
) -> None:
    """
    Write scores DataFrame to CSV in wide format.
    
    Parameters
    ----------
    scores : pd.DataFrame
        Score matrix with DatetimeIndex and asset columns.
    path : str
        Output CSV path.
    date_col : str
        Column name for date in output CSV (default "date").
    
    Raises
    ------
    ValueError
        If scores fail validation.
    
    Examples
    --------
    >>> write_scores_csv(scores, path="scores.csv")
    """
    if not isinstance(scores, pd.DataFrame):
        raise ValueError("scores must be a DataFrame")
    
    df = scores.copy()
    
    # Validate columns
    if not df.columns.is_unique:
        raise ValueError("scores columns must be unique")
    
    k = len(df.columns)
    if k < 2:
        raise ValueError(f"scores must have k_assets >= 2, got {k}")
    
    # Convert index to datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        raise ValueError(f"scores index must be convertible to datetime: {e}")
    
    # Sort by date
    df = df.sort_index()
    
    # Validate unique dates
    if not df.index.is_unique:
        raise ValueError("scores must have unique dates (duplicate index values found)")
    
    # Validate monotonic after sort
    if not df.index.is_monotonic_increasing:
        raise ValueError("scores index is not monotonic increasing after sort")
    
    # Validate finite values
    if not np.all(np.isfinite(df.values)):
        raise ValueError("scores must be finite (no NaN/inf)")
    
    # Reset index to make date a column
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: date_col})
    
    # Write CSV
    df.to_csv(path, index=False)


def write_prices_csv(
    prices: pd.DataFrame,
    *,
    path: str,
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: str = "ticker",
    format: str = "auto",
) -> None:
    """
    Write prices DataFrame to CSV in wide or long format.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex.
        Wide format: ticker columns with price values.
        Long format: columns [ticker_col, price_col].
    path : str
        Output CSV path.
    price_col : str
        Column name for price (default "close").
    date_col : str
        Column name for date in output CSV (default "date").
    ticker_col : str
        Column name for ticker (default "ticker").
    format : str
        Output format: "auto", "wide", or "long".
    
    Raises
    ------
    ValueError
        If prices fail validation.
    
    Examples
    --------
    >>> write_prices_csv(prices, path="prices.csv", format="wide")
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a DataFrame")
    
    if format not in ("auto", "wide", "long"):
        raise ValueError(f"format must be 'auto', 'wide', or 'long', got '{format}'")
    
    df = prices.copy()
    
    # Detect format if auto
    if format == "auto":
        if ticker_col in df.columns and price_col in df.columns:
            format = "long"
        else:
            format = "wide"
    
    # Convert index to datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        raise ValueError(f"prices index must be convertible to datetime: {e}")
    
    if format == "wide":
        _write_prices_wide(df, path, date_col)
    else:  # long
        _write_prices_long(df, path, date_col, ticker_col, price_col)


def _write_prices_wide(
    df: pd.DataFrame,
    path: str,
    date_col: str,
) -> None:
    """Write wide-format prices to CSV."""
    # Validate columns
    if not df.columns.is_unique:
        raise ValueError("wide prices columns must be unique")
    
    k = len(df.columns)
    if k < 2:
        raise ValueError(f"wide prices must have k_assets >= 2, got {k}")
    
    # Sort by date
    df = df.sort_index()
    
    # Validate unique dates
    if not df.index.is_unique:
        raise ValueError("wide prices must have unique dates (duplicate index values found)")
    
    # Validate all values finite and positive
    values = df.values
    if not np.all(np.isfinite(values)):
        raise ValueError("wide prices must be finite (no NaN/inf)")
    
    if not np.all(values > 0):
        raise ValueError("wide prices must be positive (all values > 0)")
    
    # Reset index to make date a column
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: date_col})
    
    # Write CSV
    df.to_csv(path, index=False)


def _write_prices_long(
    df: pd.DataFrame,
    path: str,
    date_col: str,
    ticker_col: str,
    price_col: str,
) -> None:
    """Write long-format prices to CSV."""
    # Validate required columns
    missing = []
    for col in [ticker_col, price_col]:
        if col not in df.columns:
            missing.append(col)
    
    if missing:
        raise ValueError(
            f"long prices missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Extract only needed columns
    df = df[[ticker_col, price_col]].copy()
    
    # Validate all values finite and positive
    price_values = df[price_col].values
    if not np.all(np.isfinite(price_values)):
        raise ValueError("long prices must have finite price values (no NaN/inf)")
    
    if not np.all(price_values > 0):
        raise ValueError("long prices must have positive price values (all > 0)")
    
    # Reset index to make date a column
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: date_col})
    
    # Sort by (date, ticker)
    df = df.sort_values([date_col, ticker_col])
    
    # Validate unique (date, ticker) pairs
    duplicates = df.duplicated(subset=[date_col, ticker_col], keep=False)
    if duplicates.any():
        raise ValueError("long prices must have unique (date, ticker) pairs")
    
    # Reorder columns: date, ticker, price
    df = df[[date_col, ticker_col, price_col]]
    
    # Write CSV
    df.to_csv(path, index=False)
