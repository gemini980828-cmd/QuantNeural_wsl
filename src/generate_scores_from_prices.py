"""
Score Generator from Prices.

Generates monthly momentum scores from price data with PIT safety and integrity gates.

Design Principles:
- Deterministic: no randomness, no system clock, no network
- PIT-safe: scores use only past prices (no look-ahead)
- Fail-fast: clear ValueError on invalid inputs

Usage:
    from src.generate_scores_from_prices import (
        load_prices_csv,
        compute_monthly_momentum_scores,
        write_scores_csv,
    )
    
    prices = load_prices_csv("prices.csv")
    scores = compute_monthly_momentum_scores(prices, lookback_days=252)
    write_scores_csv(scores, out_scores_csv_path="scores.csv")
"""

from typing import Optional

import numpy as np
import pandas as pd


def load_prices_csv(
    prices_csv_path: str,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Load prices CSV into DataFrame.
    
    Supports both wide format (date + ticker columns) and long format
    (date, ticker, close).
    
    Parameters
    ----------
    prices_csv_path : str
        Path to prices CSV.
    date_col : str
        Column name for date (default "date").
    ticker_col : str
        Column name for ticker in long format (default "ticker").
    price_col : str
        Column name for price (default "close").
    
    Returns
    -------
    pd.DataFrame
        For wide format: DatetimeIndex, columns=tickers, values=prices.
        For long format: DatetimeIndex, columns=["ticker", price_col].
    
    Raises
    ------
    ValueError
        If required columns are missing or dates cannot be parsed.
    """
    df = pd.read_csv(prices_csv_path)
    
    if date_col not in df.columns:
        raise ValueError(
            f"prices CSV missing required date column '{date_col}'. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Detect format: long if ticker_col exists
    is_long = ticker_col in df.columns
    
    if is_long:
        # Long format
        if price_col not in df.columns:
            raise ValueError(
                f"Long-format prices CSV missing required price column '{price_col}'. "
                f"Available columns: {list(df.columns)}"
            )
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.sort_index()
        
        # Ensure ticker column is named "ticker"
        if ticker_col != "ticker":
            df = df.rename(columns={ticker_col: "ticker"})
        
        # Keep only ticker and price columns
        df = df[["ticker", price_col]].copy()
        
    else:
        # Wide format
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.sort_index()
    
    return df


def _detect_leading_plateau(
    prices_wide: pd.DataFrame,
    leading_plateau_days: int,
) -> list:
    """
    Detect tickers with suspicious leading constant prices.
    
    Returns list of tickers where the first `leading_plateau_days` non-NaN
    prices have only 1 unique value (padded history artifact).
    """
    bad_tickers = []
    
    for ticker in prices_wide.columns:
        series = prices_wide[ticker].dropna()
        
        if len(series) < leading_plateau_days:
            continue
        
        first_values = series.iloc[:leading_plateau_days]
        unique_values = first_values.nunique()
        
        if unique_values == 1:
            bad_tickers.append(ticker)
    
    return bad_tickers


def compute_monthly_momentum_scores(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 252,
    rebalance: str = "M",
    min_coverage: float = 1.0,
    enforce_no_leading_plateau: bool = True,
    leading_plateau_days: int = 252,
) -> pd.DataFrame:
    """
    Compute monthly momentum scores from prices.
    
    PIT-safe by construction: momentum at date t uses only prices up to t.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Either wide format (DatetimeIndex, columns=tickers) or long format
        (DatetimeIndex, columns=["ticker", price_col]).
    lookback_days : int
        Number of trading days for momentum calculation (default 252 = 1 year).
    rebalance : str
        Rebalance frequency: "M" (monthly) or "Q" (quarterly).
    min_coverage : float
        Minimum fraction of non-NaN prices required (0.0 to 1.0, default 1.0).
    enforce_no_leading_plateau : bool
        If True, detect and reject tickers with constant leading prices.
    leading_plateau_days : int
        Number of leading days to check for plateau (default 252).
    
    Returns
    -------
    pd.DataFrame
        Wide scores DataFrame with DatetimeIndex (actual last trading day of
        each period) and ticker columns. No NaN/inf values.
    
    Raises
    ------
    ValueError
        If scores contain NaN/inf, min_coverage not met, or leading plateau detected.
    """
    if rebalance not in ("M", "Q"):
        raise ValueError(f"rebalance must be 'M' or 'Q', got '{rebalance}'")
    
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    
    # Convert long to wide if needed
    if "ticker" in prices.columns:
        # Long format -> pivot to wide
        price_col = [c for c in prices.columns if c != "ticker"][0]
        prices_wide = prices.pivot_table(
            values=price_col,
            index=prices.index,
            columns="ticker",
            aggfunc="last"  # In case of duplicates, take last
        )
    else:
        prices_wide = prices.copy()
    
    if prices_wide.empty:
        raise ValueError("No price data after loading")
    
    if len(prices_wide.columns) < 2:
        raise ValueError(
            f"Need at least 2 tickers for scores, got {len(prices_wide.columns)}"
        )
    
    # Check leading plateau integrity gate
    if enforce_no_leading_plateau:
        bad_tickers = _detect_leading_plateau(prices_wide, leading_plateau_days)
        if bad_tickers:
            raise ValueError(
                f"Leading plateau detected (first {leading_plateau_days} days have constant price): "
                f"{bad_tickers}. This may indicate pre-IPO padded history. "
                "Use enforce_no_leading_plateau=False to skip this check."
            )
    
    # Filter tickers by coverage
    total_days = len(prices_wide)
    coverage = prices_wide.notna().sum() / total_days
    valid_tickers = coverage[coverage >= min_coverage].index.tolist()
    
    dropped_tickers = [t for t in prices_wide.columns if t not in valid_tickers]
    if dropped_tickers:
        print(f"Dropped {len(dropped_tickers)} tickers due to coverage < {min_coverage}")
    
    if len(valid_tickers) < 2:
        raise ValueError(
            f"Only {len(valid_tickers)} tickers meet coverage threshold {min_coverage}. "
            f"Need at least 2 tickers."
        )
    
    prices_wide = prices_wide[valid_tickers]
    
    # Compute momentum (price / price_shifted - 1)
    # Using shift for PIT safety: at date t, we use price[t] / price[t - lookback_days] - 1
    momentum = prices_wide / prices_wide.shift(lookback_days) - 1
    
    # Resample to get actual last trading day of each period
    if rebalance == "M":
        period_freq = "ME"  # Month-end
    else:  # "Q"
        period_freq = "QE"  # Quarter-end
    
    # Get the actual last trading day in each period by taking last non-NaN value
    scores = momentum.resample(period_freq).last()
    
    # Drop rows before we have enough lookback
    # First valid row is at index >= lookback_days from start
    scores = scores.iloc[1:]  # Drop first row which is typically NaN from shift
    
    # Drop rows with any NaN (incomplete periods)
    scores = scores.dropna()
    
    if scores.empty:
        raise ValueError(
            "No valid scores after applying lookback and dropping NaNs. "
            f"lookback_days={lookback_days}, total_days={total_days}"
        )
    
    # Final validation: no NaN or inf
    if scores.isna().any().any():
        nan_counts = scores.isna().sum()
        bad_cols = nan_counts[nan_counts > 0].index.tolist()
        raise ValueError(f"Scores contain NaN values in columns: {bad_cols}")
    
    if not np.isfinite(scores.values).all():
        raise ValueError("Scores contain infinite values")
    
    return scores


def write_scores_csv(
    scores: pd.DataFrame,
    *,
    out_scores_csv_path: str,
    date_col: str = "date",
) -> None:
    """
    Write scores DataFrame to CSV.
    
    Parameters
    ----------
    scores : pd.DataFrame
        Wide scores DataFrame with DatetimeIndex and ticker columns.
    out_scores_csv_path : str
        Output CSV path.
    date_col : str
        Column name for date in output (default "date").
    """
    if not isinstance(scores, pd.DataFrame):
        raise ValueError("scores must be a DataFrame")
    
    if scores.empty:
        raise ValueError("scores DataFrame is empty")
    
    df = scores.copy()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: date_col})
    df = df.sort_values(date_col)
    
    # Validate no NaN/inf
    numeric_cols = [c for c in df.columns if c != date_col]
    for col in numeric_cols:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Column {col} contains NaN or inf values")
    
    df.to_csv(out_scores_csv_path, index=False)
