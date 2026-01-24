"""
End-to-End Backtest Wiring.

Provides a convenience wrapper that wires:
- scores_to_target_weights (weights adapter)
- run_backtest (backtest harness)

This is NOT a performance backtest; it is a correctness & integration gate.

Design Principles:
- No external dependencies beyond numpy/pandas (standard project deps)
- Deterministic: no randomness, no system clock, no network
- Fail-fast: let underlying validators raise as appropriate
"""

from typing import Optional, Union

import pandas as pd

from src.weights_adapter import scores_to_target_weights
from src.backtest_harness import run_backtest


def run_scores_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    price_col: str = "close",
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
    End-to-end convenience wrapper for scores -> weights -> backtest.
    
    Converts model scores to target weights and runs the deterministic
    backtest harness with alignment, lag, and fail-fast validation.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data in wide format (DatetimeIndex, ticker columns) or
        long format (DatetimeIndex, columns ["ticker", price_col]).
    scores : pd.DataFrame
        Score matrix with shape (n_dates, k_assets).
        Index = dates, columns = tickers.
    price_col : str
        Column name for close prices (default "close").
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
    score_transform : str
        Optional pre-softmax per-date transform applied across assets
        (see src.weights_adapter.scores_to_target_weights).
    winsorize_q_low, winsorize_q_high : float
        Winsorization quantiles when score_transform includes "winsorize".
    zscore_eps : float
        Z-score std guard epsilon when score_transform includes "zscore".
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
        - equity_curve : pd.Series
        - daily_returns : pd.Series
        - rebalance_dates : pd.DatetimeIndex
        - weights_used : pd.DataFrame
        - turnover : pd.Series
        - costs : pd.Series
        - trades : pd.DataFrame
        - metrics : dict
        - target_weights : pd.DataFrame (added by this wrapper)
    
    Raises
    ------
    ValueError
        If inputs fail validation (propagated from underlying modules).
    
    Examples
    --------
    >>> result = run_scores_backtest(prices, scores, method="softmax")
    >>> print(result["metrics"]["sharpe"])
    """
    # Step 1: Convert scores to target weights
    target_weights = scores_to_target_weights(
        scores,
        method=method,
        temperature=temperature,
        top_k=top_k,
        max_weight=max_weight,
        score_transform=score_transform,
        winsorize_q_low=winsorize_q_low,
        winsorize_q_high=winsorize_q_high,
        zscore_eps=zscore_eps,
    )
    
    # Step 2: Run the backtest harness
    result = run_backtest(
        prices,
        target_weights,
        price_col=price_col,
        rebalance=rebalance,
        execution_lag_days=execution_lag_days,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity,
        max_gross_leverage=max_gross_leverage,
    )
    
    # Step 3: Add target_weights to result
    result["target_weights"] = target_weights
    
    return result
