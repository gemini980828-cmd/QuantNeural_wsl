"""
Backtest Runner from CSV Artifacts.

Provides a minimal, deterministic entrypoint that loads CSV files
and runs the E2E backtest wrapper:
  prices.csv + scores.csv → target_weights → backtest → metrics

Design Principles:
- No external dependencies beyond numpy/pandas (standard project deps)
- Deterministic: no randomness, no system clock, no network
- Fail-fast: raise clear exceptions on invalid inputs
"""

from typing import Optional

import pandas as pd

from src.e2e_backtest import run_scores_backtest


def _load_scores_csv(
    csv_path: str,
    date_col: str,
) -> pd.DataFrame:
    """
    Load scores CSV into DataFrame.
    
    Expected format: wide with date column + asset columns.
    """
    df = pd.read_csv(csv_path)
    
    # Validate date column exists
    if date_col not in df.columns:
        raise ValueError(
            f"scores CSV missing required date column '{date_col}'. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Parse date and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()
    
    # Validate unique index
    if not df.index.is_unique:
        raise ValueError("scores CSV has duplicate dates")
    
    return df


def _load_prices_csv(
    csv_path: str,
    date_col: str,
    ticker_col: str,
    price_col: str,
) -> pd.DataFrame:
    """
    Load prices CSV into DataFrame.
    
    Supports two formats:
    A) Long format: columns [date_col, ticker_col, price_col]
    B) Wide format: date_col + ticker columns
    """
    df = pd.read_csv(csv_path)
    
    # Detect format by presence of ticker_col
    is_long = ticker_col in df.columns
    
    if is_long:
        # Long format
        # Validate required columns
        missing = []
        for col in [date_col, ticker_col, price_col]:
            if col not in df.columns:
                missing.append(col)
        
        if missing:
            raise ValueError(
                f"Long-format prices CSV missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Parse date and set as index
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        # Keep only ticker and price columns
        df = df[[ticker_col, price_col]]
        
        # Sort by index then ticker
        df = df.sort_values([df.index.name, ticker_col])
        df = df.sort_index(kind="stable")
        
    else:
        # Wide format
        # Validate date column exists
        if date_col not in df.columns:
            raise ValueError(
                f"Wide-format prices CSV missing required date column '{date_col}'. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Parse date and set as index
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.sort_index()
        
        # Validate unique index
        if not df.index.is_unique:
            raise ValueError("Wide-format prices CSV has duplicate dates")
    
    return df


def run_scores_backtest_from_csv(
    *,
    prices_csv_path: str,
    scores_csv_path: str,
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: str = "ticker",
    rebalance: str = "M",
    execution_lag_days: int = 1,
    method: str = "softmax",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    max_weight: Optional[float] = None,
    score_transform: str = "none",
    winsorize_q_low: float = 0.01,
    winsorize_q_high: float = 0.99,
    zscore_eps: float = 1e-12,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_equity: float = 1.0,
    max_gross_leverage: Optional[float] = None,
) -> dict:
    """
    Load CSVs and run E2E backtest.
    
    Parameters
    ----------
    prices_csv_path : str
        Path to prices CSV (wide or long format).
    scores_csv_path : str
        Path to scores CSV (wide format: date + asset columns).
    price_col : str
        Column name for close prices (default "close").
    date_col : str
        Column name for date (default "date").
    ticker_col : str
        Column name for ticker in long format (default "ticker").
    rebalance : str
        Rebalance frequency: "M" (monthly) or "Q" (quarterly).
    execution_lag_days : int
        Days to lag execution after signal (>= 0).
    method : str
        Weight construction method: "softmax", "rank", or "topk".
    temperature : float
        Temperature for softmax method (> 0).
    top_k : int, optional
        Number of top assets for topk method.
    max_weight : float, optional
        Maximum weight per asset in (0, 1].
    cost_bps : float
        Transaction cost in basis points.
    slippage_bps : float
        Slippage in basis points.
    initial_equity : float
        Starting equity (default 1.0).
    max_gross_leverage : float, optional
        Maximum gross leverage cap.
    
    Returns
    -------
    dict
        Backtest result containing:
        - equity_curve, daily_returns, rebalance_dates, weights_used,
          turnover, costs, trades, metrics (8 harness keys)
        - target_weights (added by E2E wrapper)
    
    Raises
    ------
    ValueError
        If CSVs fail validation or backtest fails.
    
    Examples
    --------
    >>> result = run_scores_backtest_from_csv(
    ...     prices_csv_path="data/prices.csv",
    ...     scores_csv_path="data/scores.csv",
    ...     method="topk",
    ...     top_k=5
    ... )
    >>> print(result["metrics"]["sharpe"])
    """
    # Load CSVs
    prices = _load_prices_csv(
        prices_csv_path,
        date_col=date_col,
        ticker_col=ticker_col,
        price_col=price_col,
    )
    
    scores = _load_scores_csv(
        scores_csv_path,
        date_col=date_col,
    )
    
    # Run E2E backtest (fail-fast pass-through)
    result = run_scores_backtest(
        prices,
        scores,
        price_col=price_col,
        rebalance=rebalance,
        execution_lag_days=execution_lag_days,
        method=method,
        temperature=temperature,
        top_k=top_k,
        max_weight=max_weight,
        score_transform=score_transform,
        winsorize_q_low=winsorize_q_low,
        winsorize_q_high=winsorize_q_high,
        zscore_eps=zscore_eps,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity,
        max_gross_leverage=max_gross_leverage,
    )
    
    return result
