"""
Real-Data Dataset Builder for QUANT-NEURAL.

Constructs aligned X (features) and Y (labels) datasets for model training.

Point-in-Time (PIT) Rules:
- Prices: only dates <= as_of_date
- Labels: next-month returns computed from monthly closes
- No future data visible at any point
- No "now" or system time logic

Label Construction:
- y_t = (close[t+1] / close[t]) - 1
- Last available month has NaN (no t+1 available)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.stooq_prices import load_stooq_daily_prices, resample_to_monthly


def build_monthly_next_returns_from_stooq(
    stooq_csv_by_ticker: dict[str, str],
    *,
    tickers_in_order: list[str],
    as_of_date: str,
) -> pd.DataFrame:
    """
    Build monthly next-return labels from Stooq price data.
    
    Parameters
    ----------
    stooq_csv_by_ticker : dict[str, str]
        Mapping of ticker -> local CSV file path.
    tickers_in_order : list[str]
        Ordered list of exactly 10 tickers for S0_Y..S9_Y columns.
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format.
    
    Returns
    -------
    pd.DataFrame
        Index: month-end DatetimeIndex (monotonic increasing)
        Columns: exactly ["S0_Y", "S1_Y", ..., "S9_Y"]
        Values: next-month returns (NaN for last available month)
        dtype: float for all columns
    
    Raises
    ------
    ValueError
        If tickers_in_order length != 10, fewer than 2 months after alignment,
        non-finite close values, or index.max() > as_of_date.
    """
    # Validate ticker count
    if len(tickers_in_order) != 10:
        raise ValueError(
            f"tickers_in_order must have exactly 10 tickers, got {len(tickers_in_order)}"
        )
    
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    
    # Load monthly closes for each ticker
    monthly_closes: dict[str, pd.Series] = {}
    
    for ticker in tickers_in_order:
        if ticker not in stooq_csv_by_ticker:
            raise ValueError(f"Ticker {ticker} not found in stooq_csv_by_ticker")
        
        csv_path = stooq_csv_by_ticker[ticker]
        
        # Load daily prices with PIT cutoff
        daily = load_stooq_daily_prices(csv_path, as_of_date=as_of_date, ticker=ticker)
        
        # Resample to monthly
        monthly = resample_to_monthly(daily)
        
        # Extract close series indexed by date
        close_series = monthly.set_index("date")["close"]
        
        # Validate non-finite values
        if not np.isfinite(close_series).all():
            raise ValueError(f"Non-finite close values detected for ticker {ticker}")
        
        monthly_closes[ticker] = close_series
    
    # Align all tickers to common month-end index (INNER JOIN / intersection)
    common_index = None
    for ticker in tickers_in_order:
        series = monthly_closes[ticker]
        if common_index is None:
            common_index = series.index
        else:
            common_index = common_index.intersection(series.index)
    
    if common_index is None or len(common_index) < 2:
        raise ValueError(
            "Fewer than 2 months available after aligning all tickers"
        )
    
    # Sort common index
    common_index = common_index.sort_values()
    
    # Validate as_of_date cutoff
    if common_index.max() > as_of_dt:
        raise ValueError(
            f"index.max()={common_index.max()} > as_of_date={as_of_date}"
        )
    
    # Build aligned close DataFrame
    aligned_closes = pd.DataFrame(index=common_index)
    for i, ticker in enumerate(tickers_in_order):
        aligned_closes[f"S{i}_close"] = monthly_closes[ticker].loc[common_index]
    
    # Compute next-month returns: y_t = (close[t+1] / close[t]) - 1
    returns = pd.DataFrame(index=common_index)
    for i in range(10):
        close_col = f"S{i}_close"
        # Shift by -1 to get next month's close
        next_close = aligned_closes[close_col].shift(-1)
        returns[f"S{i}_Y"] = (next_close / aligned_closes[close_col]) - 1
    
    # Ensure float dtype
    returns = returns.astype(float)
    
    return returns


def build_real_data_xy_dataset(
    *,
    feature_builder: Callable[..., pd.DataFrame],
    feature_builder_kwargs: dict,
    stooq_csv_by_ticker_for_labels: dict[str, str],
    label_tickers_in_order: list[str],
    as_of_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build aligned X (features) and Y (labels) datasets for model training.
    
    Parameters
    ----------
    feature_builder : Callable
        Function that builds the X feature frame. Must accept as_of_date kwarg.
    feature_builder_kwargs : dict
        Keyword arguments to pass to feature_builder (except as_of_date).
    stooq_csv_by_ticker_for_labels : dict[str, str]
        Mapping of ticker -> local CSV file path for label construction.
    label_tickers_in_order : list[str]
        Ordered list of exactly 10 tickers for label columns.
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (X_aligned, Y_aligned) where:
        - X_aligned: 20-column feature frame (S0_H1..S9_H1, S0_H2..S9_H2)
        - Y_aligned: 10-column label frame (S0_Y..S9_Y)
        Both have identical, monotonic DatetimeIndex.
    
    Raises
    ------
    ValueError
        If X doesn't have 20 columns, Y doesn't have 10 columns,
        or alignment produces empty intersection.
    """
    # Build features (X)
    X = feature_builder(**feature_builder_kwargs, as_of_date=as_of_date)
    
    # Build labels (Y)
    Y = build_monthly_next_returns_from_stooq(
        stooq_csv_by_ticker_for_labels,
        tickers_in_order=label_tickers_in_order,
        as_of_date=as_of_date,
    )
    
    # Find index intersection
    idx = X.index.intersection(Y.index)
    
    if len(idx) == 0:
        raise ValueError("No overlapping months between X and Y")
    
    # Sort the intersection
    idx = idx.sort_values()
    
    # Align both frames
    X_aligned = X.loc[idx].copy()
    Y_aligned = Y.loc[idx].copy()
    
    # Validate X shape
    if len(X_aligned.columns) != 20:
        raise ValueError(
            f"X must have exactly 20 columns (H1/H2 features), got {len(X_aligned.columns)}"
        )
    
    # Validate Y shape
    if len(Y_aligned.columns) != 10:
        raise ValueError(
            f"Y must have exactly 10 columns (S0_Y..S9_Y), got {len(Y_aligned.columns)}"
        )
    
    # Validate index alignment
    if not X_aligned.index.equals(Y_aligned.index):
        raise ValueError("X and Y indexes are not identical after alignment")
    
    # Validate monotonic
    if not X_aligned.index.is_monotonic_increasing:
        raise ValueError("Aligned index is not monotonic increasing")
    
    return X_aligned, Y_aligned
