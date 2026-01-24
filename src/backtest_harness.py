"""
Minimal deterministic portfolio backtest harness.

Provides:
- validate_prices_frame: Validate price DataFrame structure
- validate_target_weights: Validate target weights DataFrame
- resample_rebalance_dates: Select rebalance dates (monthly/quarterly)
- align_and_lag_weights: Align signals to trading calendar with execution lag
- run_backtest: Run complete backtest with equity curve and metrics

Design Principles:
- No network, no system clock, no "today/now" logic
- Deterministic: same inputs => identical outputs
- Fail-fast: raise clear exceptions on invalid inputs
- Wiring-neutral: accepts weights, agnostic to model internals
"""

from typing import Optional

import numpy as np
import pandas as pd


def validate_prices_frame(
    prices: pd.DataFrame,
    *,
    price_col: str = "close"
) -> None:
    """
    Validate price DataFrame structure.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data. Can be:
        - Long format: columns ["ticker", price_col] with DatetimeIndex
          (duplicate dates allowed, but (date, ticker) pairs must be unique)
        - Wide format: DatetimeIndex, columns are tickers, values are prices
          (dates must be unique)
    price_col : str
        Column name for close prices (used in long format).
    
    Raises
    ------
    ValueError
        If prices fail validation.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pd.DataFrame")
    
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices.index must be a DatetimeIndex")
    
    if len(prices) == 0:
        raise ValueError("prices must not be empty")
    
    # Determine format: long or wide
    is_long_format = "ticker" in prices.columns and price_col in prices.columns
    
    if is_long_format:
        # Long format validation
        # - Allow duplicate dates (normal for multiple tickers per date)
        # - Require monotonic non-decreasing index
        # - Require unique (date, ticker) pairs
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices.index must be sorted (monotonic non-decreasing)")
        
        # Check unique (date, ticker) pairs
        date_ticker_pairs = list(zip(prices.index, prices["ticker"]))
        if len(date_ticker_pairs) != len(set(date_ticker_pairs)):
            raise ValueError(
                "Duplicate (date, ticker) pairs found. Each date+ticker combination "
                "must appear at most once."
            )
        
        price_values = prices[price_col].values
    else:
        # Wide format validation
        # - Require strictly increasing index with unique dates
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices.index must be strictly increasing")
        
        if len(prices.index) != len(prices.index.unique()):
            raise ValueError("prices.index must have unique dates (no duplicates)")
        
        price_values = prices.values.flatten()
    
    # Check finite and positive (both formats)
    if not np.all(np.isfinite(price_values)):
        raise ValueError("All prices must be finite (no NaN, inf, -inf)")
    
    if not np.all(price_values > 0):
        raise ValueError("All prices must be > 0")


def validate_target_weights(
    weights: pd.DataFrame
) -> None:
    """
    Validate target weights DataFrame.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Target weights with DatetimeIndex, columns are tickers.
    
    Raises
    ------
    ValueError
        If weights fail validation.
    """
    if not isinstance(weights, pd.DataFrame):
        raise ValueError("weights must be a pd.DataFrame")
    
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise ValueError("weights.index must be a DatetimeIndex")
    
    if len(weights) == 0:
        raise ValueError("weights must not be empty")
    
    # Check strictly increasing index
    if not weights.index.is_monotonic_increasing:
        raise ValueError("weights.index must be strictly increasing")
    
    if len(weights.index) != len(weights.index.unique()):
        raise ValueError("weights.index must have unique values (no duplicates)")
    
    # Check finite values (allow negatives for shorting)
    if not np.all(np.isfinite(weights.values)):
        raise ValueError("All weights must be finite (no NaN, inf, -inf)")


def resample_rebalance_dates(
    signal_dates: pd.DatetimeIndex,
    *,
    rebalance: str
) -> pd.DatetimeIndex:
    """
    Select rebalance dates based on frequency.
    
    Parameters
    ----------
    signal_dates : pd.DatetimeIndex
        Original signal dates (e.g., monthly).
    rebalance : str
        "M" for monthly (all dates), "Q" for quarterly (every 3rd date).
    
    Returns
    -------
    pd.DatetimeIndex
        Selected rebalance dates.
    
    Raises
    ------
    ValueError
        If rebalance is not "M" or "Q".
    """
    if rebalance not in ("M", "Q"):
        raise ValueError(f"rebalance must be 'M' or 'Q', got '{rebalance}'")
    
    if len(signal_dates) == 0:
        return pd.DatetimeIndex([])
    
    if rebalance == "M":
        return signal_dates
    
    # Quarterly: every 3rd date (0, 3, 6, ...)
    indices = list(range(0, len(signal_dates), 3))
    return signal_dates[indices]


def align_and_lag_weights(
    weights: pd.DataFrame,
    prices_index: pd.DatetimeIndex,
    *,
    execution_lag_days: int = 1
) -> pd.DataFrame:
    """
    Align signal dates to trading calendar with execution lag.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Target weights with signal dates as index.
    prices_index : pd.DatetimeIndex
        Trading calendar (trading days from price data).
    execution_lag_days : int
        Number of trading days to delay execution (default 1).
    
    Returns
    -------
    pd.DataFrame
        Weights indexed by effective trade dates (subset of prices_index).
    
    Notes
    -----
    For each weight row dated t:
    1. Find first trading day >= t in prices_index
    2. Shift forward by execution_lag_days within prices_index
    3. If beyond last price date, drop the signal
    """
    if len(weights) == 0:
        return pd.DataFrame(columns=weights.columns)
    
    if execution_lag_days < 0:
        raise ValueError("execution_lag_days must be >= 0")
    
    prices_index = prices_index.sort_values()
    
    effective_dates = []
    effective_weights = []
    
    for signal_date in weights.index:
        # Find first trading day >= signal_date
        candidates = prices_index[prices_index >= signal_date]
        
        if len(candidates) == 0:
            # Signal date beyond last trading day
            continue
        
        first_trade_idx = prices_index.get_loc(candidates[0])
        
        # Apply execution lag
        lagged_idx = first_trade_idx + execution_lag_days
        
        if lagged_idx >= len(prices_index):
            # Lagged date beyond last trading day
            continue
        
        effective_date = prices_index[lagged_idx]
        effective_dates.append(effective_date)
        effective_weights.append(weights.loc[signal_date].values)
    
    if len(effective_dates) == 0:
        return pd.DataFrame(columns=weights.columns)
    
    return pd.DataFrame(
        effective_weights,
        index=pd.DatetimeIndex(effective_dates),
        columns=weights.columns
    )


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    price_col: str = "close",
    rebalance: str = "M",
    execution_lag_days: int = 1,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_equity: float = 1.0,
    max_gross_leverage: Optional[float] = None
) -> dict:
    """
    Run complete portfolio backtest.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices (wide format: index=dates, columns=tickers).
    target_weights : pd.DataFrame
        Target weights (index=signal dates, columns=tickers).
    price_col : str
        Price column name (for long format compatibility, not used in wide).
    rebalance : str
        "M" for monthly rebalancing, "Q" for quarterly.
    execution_lag_days : int
        Trading days to delay execution (default 1).
    cost_bps : float
        Transaction cost in basis points (applied on turnover).
    slippage_bps : float
        Slippage in basis points (applied on turnover).
    initial_equity : float
        Starting equity value (default 1.0).
    max_gross_leverage : float or None
        If set, scale weights so sum(abs(w)) <= cap at each rebalance.
    
    Returns
    -------
    dict
        Backtest results with keys:
        - equity_curve: pd.Series
        - daily_returns: pd.Series
        - rebalance_dates: pd.DatetimeIndex
        - weights_used: pd.DataFrame
        - turnover: pd.Series (sum of |delta_w| at each rebalance)
        - costs: pd.Series
        - trades: pd.DataFrame
        - metrics: dict
    
    Notes
    -----
    - Turnover is computed as sum(|delta_w|) at each rebalance.
    - Costs = equity * turnover * (cost_bps + slippage_bps) / 10000.
    - Between rebalances, weights are held constant.
    - Fail-fast on missing prices.
    """
    # Validate inputs
    validate_prices_frame(prices, price_col=price_col)
    validate_target_weights(target_weights)
    
    # Convert prices to wide format if needed
    if "ticker" in prices.columns:
        # Long format -> wide
        price_wide = prices.pivot(columns="ticker", values=price_col)
    else:
        price_wide = prices.copy()
    
    price_wide = price_wide.sort_index()
    trading_days = price_wide.index
    tickers = list(price_wide.columns)
    
    # Ensure target_weights columns are subset of tickers
    missing_tickers = set(target_weights.columns) - set(tickers)
    if missing_tickers:
        raise ValueError(f"Tickers in weights not found in prices: {missing_tickers}")
    
    # Align weights to price tickers (fill missing with 0)
    aligned_weights = target_weights.reindex(columns=tickers, fill_value=0.0)
    
    # Resample to rebalance frequency
    rebal_signal_dates = resample_rebalance_dates(aligned_weights.index, rebalance=rebalance)
    rebal_weights = aligned_weights.loc[rebal_signal_dates]
    
    # Align and lag weights to trading calendar
    effective_weights = align_and_lag_weights(
        rebal_weights, trading_days, execution_lag_days=execution_lag_days
    )
    
    if len(effective_weights) == 0:
        raise ValueError("No valid rebalance dates after alignment and lag")
    
    # Apply max gross leverage scaling
    if max_gross_leverage is not None:
        for dt in effective_weights.index:
            gross = np.abs(effective_weights.loc[dt].values).sum()
            if gross > max_gross_leverage:
                scale = max_gross_leverage / gross
                effective_weights.loc[dt] = effective_weights.loc[dt] * scale
    
    rebalance_dates = effective_weights.index
    
    # Determine backtest period: from first rebalance date to end
    first_rebal = rebalance_dates[0]
    backtest_days = trading_days[trading_days >= first_rebal]
    
    # FAIL-FAST: Check for missing prices (NaN) in backtest panel
    # Only check tickers that are actually used in effective_weights (non-zero at least once)
    # Use effective_weights (already aligned to tickers) to avoid KeyError
    used_tickers = [t for t in tickers if (effective_weights[t].abs() > 0).any()]
    if len(used_tickers) == 0:
        used_tickers = tickers  # Fallback to all tickers if weights are all zero
    
    price_panel = price_wide.loc[backtest_days, used_tickers]
    nan_mask = price_panel.isna()
    if nan_mask.any().any():
        n_missing = nan_mask.sum().sum()
        # Get example dates and tickers with missing values
        example_locs = np.argwhere(nan_mask.values)[:5]  # Up to 5 examples
        example_dates = [str(backtest_days[i]) for i, _ in example_locs]
        example_tickers = [used_tickers[j] for _, j in example_locs]
        raise ValueError(
            f"Missing prices in backtest panel: n_missing={n_missing}, "
            f"example_dates={example_dates[:3]}, example_tickers={example_tickers[:3]}"
        )
    
    # Compute FORWARD returns: return from day t to t+1
    # This is critical: weights effective on day t earn the return from t to t+1
    # forward_return[t] = (price[t+1] - price[t]) / price[t]
    # Use fill_method=None to avoid FutureWarning and explicitly handle NaN
    forward_returns = price_wide.pct_change(fill_method=None).shift(-1)
    
    # Set last row to 0.0 explicitly (no t+1 price available)
    forward_returns.iloc[-1] = 0.0
    
    # FAIL-FAST: Check for any remaining NaN in forward returns over backtest period
    # Exclude last row from check (it's intentionally 0.0)
    returns_panel = forward_returns.loc[backtest_days[:-1], used_tickers] if len(backtest_days) > 1 else pd.DataFrame()
    if len(returns_panel) > 0:
        nan_returns = returns_panel.isna()
        if nan_returns.any().any():
            n_missing = nan_returns.sum().sum()
            raise ValueError(
                f"Missing forward returns in backtest panel: n_missing={n_missing}. "
                f"This indicates gaps in price data."
            )
    
    # Initialize
    equity_values = []
    daily_ret_values = []
    current_weights = np.zeros(len(tickers), dtype=np.float64)
    equity = initial_equity
    
    # Turnover, costs, trades tracking
    turnover_list = []
    costs_list = []
    trades_records = []
    
    rebal_idx = 0
    
    # OPTIMIZATION: Precompute ticker index mapping ONCE
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    # OPTIMIZATION: Precompute used_tickers indices ONCE (avoid per-day list.index())
    if len(used_tickers) > 0:
        used_indices = np.array([ticker_to_idx[t] for t in used_tickers], dtype=np.intp)
        # OPTIMIZATION: Pre-slice forward returns into NumPy matrix ONCE
        forward_returns_used = forward_returns.loc[backtest_days, used_tickers].to_numpy(
            dtype=np.float64, copy=False
        )
    else:
        used_indices = np.array([], dtype=np.intp)
        forward_returns_used = np.zeros((len(backtest_days), 0), dtype=np.float64)
    
    # OPTIMIZATION: Pre-extract effective_weights as numpy for faster rebalance lookup
    effective_weights_arr = effective_weights.to_numpy(dtype=np.float64)
    
    # OPTIMIZATION: Use row index instead of per-day .loc
    for row_idx, day in enumerate(backtest_days):
        # Check for rebalance
        if rebal_idx < len(rebalance_dates) and day == rebalance_dates[rebal_idx]:
            new_weights = effective_weights_arr[rebal_idx].astype(np.float64).flatten()
            
            # Compute turnover: sum of |delta_w|
            delta_w = new_weights - current_weights
            turnover = np.abs(delta_w).sum()
            turnover_list.append(turnover)
            
            # Compute and apply costs
            cost = equity * turnover * (cost_bps + slippage_bps) / 10000.0
            costs_list.append(cost)
            equity -= cost
            
            # Record trades
            for i, ticker in enumerate(tickers):
                dw = float(delta_w[i])
                if dw != 0.0:
                    trades_records.append({
                        "date": day,
                        "ticker": ticker,
                        "delta_weight": dw
                    })
            
            current_weights = new_weights
            rebal_idx += 1
        
        # Compute daily portfolio return using FORWARD returns
        # Weights on day t are applied to forward_return[t] = return from t to t+1
        # OPTIMIZATION: Use precomputed used_indices and forward_returns_used array
        if len(used_tickers) > 0:
            used_weights = current_weights[used_indices]
            used_returns = forward_returns_used[row_idx]
            port_return = np.dot(used_weights, used_returns)
        else:
            port_return = 0.0
        
        # Update equity
        equity *= (1.0 + port_return)
        
        equity_values.append(equity)
        daily_ret_values.append(port_return)
    
    # Build output series
    equity_curve = pd.Series(equity_values, index=backtest_days, name="equity")
    daily_returns = pd.Series(daily_ret_values, index=backtest_days, name="daily_return")
    turnover_series = pd.Series(turnover_list, index=rebalance_dates, name="turnover")
    costs_series = pd.Series(costs_list, index=rebalance_dates, name="costs")
    
    trades_df = pd.DataFrame(trades_records)
    if len(trades_df) == 0:
        trades_df = pd.DataFrame(columns=["date", "ticker", "delta_weight"])
    
    # Compute metrics
    n_days = len(backtest_days)
    ann_factor = 252  # Trading days per year
    
    total_return = equity_curve.iloc[-1] / initial_equity - 1.0
    years = n_days / ann_factor
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    ann_vol = daily_returns.std() * np.sqrt(ann_factor)
    
    # Maximum drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    total_turnover = turnover_series.sum()
    total_cost = costs_series.sum()
    
    # Compute CAGR / AnnVol (formerly labeled "sharpe" but not standard Sharpe ratio)
    cagr_over_vol = cagr / ann_vol if ann_vol > 0 else 0.0
    
    metrics = {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "cagr_over_vol": cagr_over_vol,  # New canonical name
        "sharpe": cagr_over_vol,  # DEPRECATED: kept for backward compatibility
        "max_drawdown": max_drawdown,
        "total_turnover": total_turnover,
        "total_cost": total_cost
    }
    
    # Deprecation warning for "sharpe" key
    warnings = [
        "METRIC_DEPRECATED:sharpe:use cagr_over_vol (sharpe here is CAGR/AnnVol, not standard Sharpe)."
    ]
    
    return {
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "rebalance_dates": rebalance_dates,
        "weights_used": effective_weights,
        "turnover": turnover_series,
        "costs": costs_series,
        "trades": trades_df,
        "metrics": metrics,
        "warnings": warnings
    }
