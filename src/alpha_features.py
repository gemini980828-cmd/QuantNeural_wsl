"""
Alpha feature engineering for ranking model.

All features are calculated using vectorized pandas/numpy operations.
No ta-lib or external dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical alpha features to a single-ticker OHLCV DataFrame.
    
    Input: DataFrame with columns [date, open, high, low, close, volume],
           sorted by date ascending.
    Output: DataFrame with added feature columns (all float32).
    
    Features implemented (pure pandas/numpy; vectorized):
    - vol_20d: 20-day annualized volatility
    - mom_5d, mom_21d, mom_63d: Rate of change
    - rsi_14d: Relative Strength Index (Wilder smoothing)
    - bbands_20d: Bollinger Bands percent_b
    - atr_14d_norm: Normalized Average True Range
    
    NaN policy: NaNs are preserved for XGBoost to handle as missing values.
    """
    df = df.copy()
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    
    # 1) vol_20d: 20-day annualized volatility
    daily_ret = close.pct_change()
    vol_20d = daily_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252)
    df["vol_20d"] = vol_20d.astype(np.float32)
    
    # 2) mom_5d, mom_21d, mom_63d: Rate of change
    df["mom_5d"] = (close / close.shift(5) - 1.0).astype(np.float32)
    df["mom_21d"] = (close / close.shift(21) - 1.0).astype(np.float32)
    df["mom_63d"] = (close / close.shift(63) - 1.0).astype(np.float32)
    
    # 3) rsi_14d: Relative Strength Index (Wilder-style EWM smoothing)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing: alpha = 1/14, equivalent to span = 2*14 - 1 = 27
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Clip to [0, 100] for safety
    rsi = rsi.clip(0.0, 100.0)
    df["rsi_14d"] = rsi.astype(np.float32)
    
    # 4) bbands_20d: Bollinger Bands percent_b
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std(ddof=0)
    upper = sma_20 + 2.0 * std_20
    lower = sma_20 - 2.0 * std_20
    band_width = upper - lower
    # Handle division by zero: set to NaN where band_width is 0
    percent_b = (close - lower) / band_width.replace(0, np.nan)
    df["bbands_20d"] = percent_b.astype(np.float32)
    
    # 5) atr_14d_norm: Normalized Average True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder-style smoothing for ATR
    atr_14 = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    # Normalize by close, handle division by zero
    atr_14d_norm = atr_14 / close.replace(0, np.nan)
    df["atr_14d_norm"] = atr_14d_norm.astype(np.float32)
    
    return df


def add_alpha_targets(
    df: pd.DataFrame,
    horizon_days: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add forward return targets to a single-ticker DataFrame.
    
    Input: DataFrame with 'close' and 'date', sorted by date ascending.
    Output: DataFrame with added target columns (all float32):
      - fwd_ret_5d, fwd_ret_10d, fwd_ret_21d, fwd_ret_63d (by default)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'close' column.
    horizon_days : list[int], optional
        Forward return horizons in days. Default: [5, 10, 21, 63].
        63d aligns with quarterly (Q) rebalance schedules.
    
    Logic:
      fwd_ret_Nd[t] = (close[t+N] / close[t]) - 1.0
    The last N rows for each horizon will be NaN.
    """
    if horizon_days is None:
        horizon_days = [5, 10, 21, 63]
    
    df = df.copy()
    close = df["close"].astype(np.float64)
    
    for n in horizon_days:
        # Forward return: shift close backward (negative shift gives future values)
        future_close = close.shift(-n)
        fwd_ret = (future_close / close) - 1.0
        col_name = f"fwd_ret_{n}d"
        df[col_name] = fwd_ret.astype(np.float32)
    
    return df
