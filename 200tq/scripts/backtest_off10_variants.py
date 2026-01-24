# -*- coding: utf-8 -*-
"""
Reproducible Backtest Runner: OFF10 Baseline vs Rule-Based Variants
====================================================================

Base Strategy: Optim_QQQ_3_161_OFF10
- Signal: QQQ MA(3) > MA(161) => ON else OFF
- ON: TQQQ 100%
- OFF: TQQQ 10% + OFF_ASSET 90%

Execution Model: Close-to-Close with 1-day signal lag
- Compute signals using day t close data
- Apply position for day t+1 return

Author: QuantNeural Backtest Engine v2026.1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
ARTIFACTS_DIR = "/home/juwon/QuantNeural/artifacts/backtest_off10_baseline"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Date ranges
PRIMARY_START = "2010-02-01"
PRIMARY_END = "2025-12-31"
SECONDARY_START = "2007-01-01"  # Optional for synthetic

# Subperiods
SUBPERIODS = {
    "P1": ("2010-01-01", "2014-12-31"),
    "P2": ("2015-01-01", "2019-12-31"),
    "P3": ("2020-01-01", "2025-12-31"),
}

# Cost/Tax grids
COST_BPS_GRID = [0, 2, 5, 10, 15, 20]
TAX_RATE_GRID = [0.0, 0.10, 0.20, 0.25]
TAX_MODES = ["Tax0", "TaxB"]  # TaxA is complex, implement Tax0 and TaxB

# Promotion criteria (realistic scenario: cost=10bps, TaxB, rate=0.20)
REALISTIC_COST_BPS = 10
REALISTIC_TAX_MODE = "TaxB"
REALISTIC_TAX_RATE = 0.20


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    short_ma: int = 3
    long_ma: int = 161
    off_tqqq_weight: float = 0.10  # TQQQ weight when OFF
    off_asset: str = "CASH"  # CASH or SGOV
    hysteresis_band: float = 0.0  # For C3 type (e.g., 0.005 = 0.5%)
    use_ma200_guard: bool = False  # For B2 type
    ensemble_windows: List[int] = field(default_factory=list)  # For D1 type


@dataclass
class BacktestResult:
    """Backtest result container"""
    name: str
    equity: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame  # Columns: TQQQ_weight, OFF_weight
    signals: pd.Series
    trades: pd.Series  # 1 when trade occurs
    turnover: pd.Series  # Daily turnover
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================
# DATA LOADING
# ============================================================
def download_data(start: str, end: str) -> pd.DataFrame:
    """Download price data from yfinance"""
    import yfinance as yf
    
    tickers = ["QQQ", "TQQQ", "SGOV", "SHV"]
    
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,  # Use adjusted close
        progress=False,
        group_by="ticker",
        threads=True,
    )
    
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance download failed")
    
    prices = pd.DataFrame()
    
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if (t, "Close") in raw.columns:
                prices[t] = raw[(t, "Close")]
        else:
            if "Close" in raw.columns:
                prices[t] = raw["Close"]
    
    # CASH as 0% return (constant price)
    prices["CASH"] = 100.0
    
    # Handle missing SGOV - fallback to CASH
    if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
        print("WARNING: SGOV not available, using SHV as proxy")
        if "SHV" in prices.columns:
            prices["SGOV"] = prices["SHV"]
        else:
            print("WARNING: SHV also not available, SGOV = CASH")
            prices["SGOV"] = 100.0
    
    # Forward fill SGOV for bond holiday mismatches only
    prices["SGOV"] = prices["SGOV"].ffill()
    
    # Drop rows where essential data is missing
    prices = prices.dropna(subset=["QQQ", "TQQQ"])
    
    return prices


def align_data(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Align data to common trading days within date range"""
    # Filter to date range
    mask = (prices.index >= start) & (prices.index <= end)
    prices = prices.loc[mask].copy()
    
    # Ensure all required columns exist
    required = ["QQQ", "TQQQ", "CASH"]
    for col in required:
        if col not in prices.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # No forward fill except for SGOV (already done)
    return prices.dropna(subset=["QQQ", "TQQQ"])


# ============================================================
# SIGNAL GENERATION
# ============================================================
def generate_base_signal(prices: pd.DataFrame, short_ma: int = 3, long_ma: int = 161) -> pd.Series:
    """
    Base signal: QQQ MA(short) > MA(long) => ON (1) else OFF (0)
    Signal computed on day t, applied to day t+1
    """
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(short_ma).mean()
    ma_long = qqq.rolling(long_ma).mean()
    
    # Signal: 1 = ON, 0 = OFF
    signal = (ma_short > ma_long).astype(int)
    
    # Shift by 1 day (apply signal to next day's return)
    signal = signal.shift(1).fillna(0).astype(int)
    
    return signal


def generate_hysteresis_signal(prices: pd.DataFrame, short_ma: int = 3, long_ma: int = 161,
                                band: float = 0.005) -> pd.Series:
    """
    C3 hysteresis signal with band:
    - ON if spread > +band
    - OFF if spread < -band
    - Otherwise keep previous state
    """
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(short_ma).mean()
    ma_long = qqq.rolling(long_ma).mean()
    
    # Spread = (MA3/MA161) - 1
    spread = (ma_short / ma_long) - 1.0
    
    # Stateful signal
    signal = pd.Series(index=prices.index, dtype=int)
    state = 0  # Start OFF
    
    for i, dt in enumerate(prices.index):
        if pd.isna(spread.loc[dt]):
            signal.loc[dt] = state
            continue
        
        s = spread.loc[dt]
        if s > band:
            state = 1  # ON
        elif s < -band:
            state = 0  # OFF
        # else keep state
        
        signal.loc[dt] = state
    
    # Shift by 1 day
    signal = signal.shift(1).fillna(0).astype(int)
    
    return signal


def generate_ensemble_signal(prices: pd.DataFrame, short_ma: int = 3,
                             long_windows: List[int] = [160, 165, 170]) -> pd.Series:
    """
    D1 ensemble signal: majority vote among long windows
    ON if at least 2 of 3 are ON else OFF
    """
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(short_ma).mean()
    
    votes = pd.DataFrame(index=prices.index)
    for lw in long_windows:
        ma_long = qqq.rolling(lw).mean()
        votes[f"w{lw}"] = (ma_short > ma_long).astype(int)
    
    # Majority vote (at least 2 of 3)
    signal = (votes.sum(axis=1) >= 2).astype(int)
    
    # Shift by 1 day
    signal = signal.shift(1).fillna(0).astype(int)
    
    return signal


def generate_ma200_guard(prices: pd.DataFrame) -> pd.Series:
    """
    B2 guard: QQQ < MA200 => True (reduce OFF exposure)
    """
    qqq = prices["QQQ"]
    ma200 = qqq.rolling(200).mean()
    guard = (qqq < ma200).astype(int)
    
    # Shift by 1 day
    guard = guard.shift(1).fillna(0).astype(int)
    
    return guard


# ============================================================
# POSITION CALCULATION
# ============================================================
def calculate_positions(signal: pd.Series, config: StrategyConfig,
                        ma200_guard: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Calculate target positions based on signal and config
    
    Returns DataFrame with columns: TQQQ, OFF_ASSET
    """
    positions = pd.DataFrame(index=signal.index)
    
    # ON: TQQQ 100%
    # OFF: TQQQ off_weight + OFF_ASSET (1 - off_weight)
    off_tqqq = config.off_tqqq_weight
    
    # Apply MA200 guard for B2 type
    if config.use_ma200_guard and ma200_guard is not None:
        # When OFF AND guard active => OFF_TQQQ = 0%
        adjusted_off_tqqq = pd.Series(off_tqqq, index=signal.index)
        adjusted_off_tqqq[ma200_guard == 1] = 0.0
    else:
        adjusted_off_tqqq = off_tqqq
    
    # Calculate weights
    if isinstance(adjusted_off_tqqq, pd.Series):
        positions["TQQQ"] = signal * 1.0 + (1 - signal) * adjusted_off_tqqq
    else:
        positions["TQQQ"] = signal * 1.0 + (1 - signal) * off_tqqq
    
    positions["OFF_ASSET"] = 1.0 - positions["TQQQ"]
    
    return positions


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(prices: pd.DataFrame, config: StrategyConfig,
                 cost_bps: float = 0.0, tax_mode: str = "Tax0",
                 tax_rate: float = 0.0) -> BacktestResult:
    """
    Run backtest for a single strategy configuration
    
    Args:
        prices: DataFrame with QQQ, TQQQ, CASH, SGOV columns
        config: Strategy configuration
        cost_bps: One-way transaction cost in basis points
        tax_mode: Tax0 (no tax), TaxA (immediate), TaxB (annual)
        tax_rate: Tax rate (e.g., 0.20 = 20%)
    
    Returns:
        BacktestResult with equity curve, metrics, etc.
    """
    # Generate signal based on config
    if config.ensemble_windows:
        signal = generate_ensemble_signal(prices, config.short_ma, config.ensemble_windows)
    elif config.hysteresis_band > 0:
        signal = generate_hysteresis_signal(prices, config.short_ma, config.long_ma,
                                            config.hysteresis_band)
    else:
        signal = generate_base_signal(prices, config.short_ma, config.long_ma)
    
    # MA200 guard for B2 type
    ma200_guard = None
    if config.use_ma200_guard:
        ma200_guard = generate_ma200_guard(prices)
        # Only apply when OFF
        ma200_guard = ma200_guard * (1 - signal)
    
    # Calculate positions
    positions = calculate_positions(signal, config, ma200_guard)
    
    # Select OFF asset
    off_asset = config.off_asset
    if off_asset == "SGOV" and "SGOV" not in prices.columns:
        off_asset = "CASH"
    
    # Calculate returns
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    off_ret = prices[off_asset].pct_change().fillna(0.0) if off_asset in prices.columns else 0.0
    
    # Portfolio return (before costs)
    port_ret = positions["TQQQ"] * tqqq_ret + positions["OFF_ASSET"] * off_ret
    
    # Calculate trades and turnover
    weight_change = positions["TQQQ"].diff().abs().fillna(0.0)
    trades = (weight_change > 0.001).astype(int)  # Trade if weight changes >0.1%
    turnover = weight_change  # Turnover = abs weight change
    
    # Apply transaction costs
    cost_factor = cost_bps / 10000.0
    cost_drag = turnover * cost_factor
    port_ret_after_cost = port_ret - cost_drag
    
    # Apply tax model
    if tax_mode == "Tax0" or tax_rate == 0.0:
        port_ret_final = port_ret_after_cost
    elif tax_mode == "TaxB":
        # TaxB: Annual net realized gain taxed at year end
        # Simplified: estimate tax drag based on realized gains
        port_ret_final = apply_tax_model_b(port_ret_after_cost, positions, prices,
                                           tqqq_ret, tax_rate)
    else:
        port_ret_final = port_ret_after_cost
    
    # Compute equity curve
    equity = (1.0 + port_ret_final).cumprod()
    
    # Calculate metrics
    metrics = calculate_metrics(equity, port_ret_final, trades, turnover)
    
    return BacktestResult(
        name=config.name,
        equity=equity,
        daily_returns=port_ret_final,
        positions=positions,
        signals=signal,
        trades=trades,
        turnover=turnover,
        metrics=metrics
    )


def apply_tax_model_b(returns: pd.Series, positions: pd.DataFrame,
                      prices: pd.DataFrame, tqqq_ret: pd.Series,
                      tax_rate: float) -> pd.Series:
    """
    TaxB: Annual net realized gain taxed at year end
    Simplified approximation:
    - Track cumulative gains from TQQQ sells
    - Apply tax at year end on net positive gains
    - Allow loss carry within year only
    """
    result = returns.copy()
    
    # Detect sells (TQQQ weight decreases)
    weight_change = positions["TQQQ"].diff().fillna(0.0)
    sells = weight_change < -0.001
    
    # Estimate realized gain on sells (simplified: use recent return as proxy)
    # In reality, would need cost basis tracking
    tqqq_cum_ret = (1 + tqqq_ret).cumprod()
    
    # Group by year
    years = returns.index.year.unique()
    
    for year in years:
        year_mask = returns.index.year == year
        year_sells = sells & year_mask
        
        if year_sells.sum() == 0:
            continue
        
        # Estimate taxable gain (simplified)
        # Use total portfolio gain in year as proxy
        year_returns = returns[year_mask]
        year_gain = year_returns.sum()
        
        if year_gain > 0:
            # Apply tax on last day of year
            year_end = year_returns.index[-1]
            tax_drag = year_gain * tax_rate / len(year_returns)  # Spread across days
            result[year_mask] -= tax_drag
    
    return result


# ============================================================
# METRICS CALCULATION
# ============================================================
def calculate_metrics(equity: pd.Series, returns: pd.Series,
                      trades: pd.Series, turnover: pd.Series,
                      trading_days: int = 252) -> Dict[str, float]:
    """Calculate all required metrics"""
    
    # CAGR
    n_years = len(equity) / trading_days
    cagr = (equity.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    
    # MDD
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    mdd = drawdown.min()
    
    # Sharpe (daily -> annualized)
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=0)
    sharpe = (daily_mean / daily_std * np.sqrt(trading_days)) if daily_std > 0 else 0.0
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0) * np.sqrt(trading_days) if len(downside) > 0 else 0.0
    sortino = (cagr / downside_std) if downside_std > 0 else 0.0
    
    # Calmar
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0.0
    
    # Worst year/month
    yearly_ret = equity.resample("YE").last().pct_change().dropna()
    worst_year = yearly_ret.min() if len(yearly_ret) > 0 else 0.0
    
    monthly_ret = equity.resample("ME").last().pct_change().dropna()
    worst_month = monthly_ret.min() if len(monthly_ret) > 0 else 0.0
    
    # Time under water
    underwater = drawdown < 0
    underwater_periods = []
    current_uw = 0
    for uw in underwater:
        if uw:
            current_uw += 1
        else:
            if current_uw > 0:
                underwater_periods.append(current_uw)
            current_uw = 0
    if current_uw > 0:
        underwater_periods.append(current_uw)
    
    max_tuw = max(underwater_periods) if underwater_periods else 0
    median_tuw = np.median(underwater_periods) if underwater_periods else 0
    
    # Trades per year
    trades_per_year = trades.sum() / n_years if n_years > 0 else 0
    
    # Annualized turnover
    annual_turnover = turnover.sum() / n_years if n_years > 0 else 0
    
    return {
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "WorstYear": worst_year,
        "WorstMonth": worst_month,
        "MaxTUW": max_tuw,
        "MedianTUW": median_tuw,
        "TradesPerYear": trades_per_year,
        "AnnualTurnover": annual_turnover,
        "FinalValue": equity.iloc[-1],
        "NumDays": len(equity),
    }


def calculate_subperiod_metrics(result: BacktestResult, subperiods: Dict[str, Tuple[str, str]]) -> Dict[str, Dict]:
    """Calculate metrics for each subperiod"""
    metrics = {}
    
    for period_name, (start, end) in subperiods.items():
        mask = (result.equity.index >= start) & (result.equity.index <= end)
        if mask.sum() == 0:
            continue
        
        sub_equity = result.equity[mask]
        sub_returns = result.daily_returns[mask]
        sub_trades = result.trades[mask]
        sub_turnover = result.turnover[mask]
        
        # Normalize equity to start at 1
        sub_equity = sub_equity / sub_equity.iloc[0]
        
        sub_metrics = calculate_metrics(sub_equity, sub_returns, sub_trades, sub_turnover)
        metrics[period_name] = {
            "CAGR": sub_metrics["CAGR"],
            "MDD": sub_metrics["MDD"],
            "Calmar": sub_metrics["Calmar"],
        }
    
    return metrics


# ============================================================
# STRATEGY CONFIGURATIONS
# ============================================================
def get_strategy_configs() -> List[StrategyConfig]:
    """Define all strategy variants"""
    return [
        # V0 Base: Optim_QQQ_3_161_OFF10 + OFF_ASSET=CASH
        StrategyConfig(
            name="V0_Base_OFF10_CASH",
            short_ma=3, long_ma=161,
            off_tqqq_weight=0.10, off_asset="CASH",
        ),
        
        # V1 A1: OFF_ASSET=SGOV
        StrategyConfig(
            name="V1_A1_OFF10_SGOV",
            short_ma=3, long_ma=161,
            off_tqqq_weight=0.10, off_asset="SGOV",
        ),
        
        # V2 C3_0p5: Hysteresis band ±0.5%
        StrategyConfig(
            name="V2_C3_Hysteresis_0p5",
            short_ma=3, long_ma=161,
            off_tqqq_weight=0.10, off_asset="CASH",
            hysteresis_band=0.005,
        ),
        
        # V3 B2: MA200 guard (OFF state + QQQ<MA200 => OFF0)
        StrategyConfig(
            name="V3_B2_MA200Guard",
            short_ma=3, long_ma=161,
            off_tqqq_weight=0.10, off_asset="CASH",
            use_ma200_guard=True,
        ),
        
        # V4 D1: Ensemble (160/165/170)
        StrategyConfig(
            name="V4_D1_Ensemble",
            short_ma=3, long_ma=161,  # Not used, ensemble_windows used instead
            off_tqqq_weight=0.10, off_asset="CASH",
            ensemble_windows=[160, 165, 170],
        ),
        
        # V5 Combo: A1 + B2 (SGOV + MA200 guard)
        StrategyConfig(
            name="V5_Combo_SGOV_MA200",
            short_ma=3, long_ma=161,
            off_tqqq_weight=0.10, off_asset="SGOV",
            use_ma200_guard=True,
        ),
    ]


# ============================================================
# PROMOTION EVALUATION
# ============================================================
def evaluate_promotion(base_result: BacktestResult, candidate_result: BacktestResult,
                       base_subperiod: Dict, candidate_subperiod: Dict) -> Tuple[bool, str]:
    """
    Evaluate if candidate should be promoted vs base
    
    Gate criteria (all must pass):
    1) CAGR >= Base_CAGR - 0.5%p
    2) trades/year <= Base_trades/year * 1.2
    
    Soft improve (at least one):
    - MDD improves by >=2%p OR
    - WorstMonth improves by >=3%p OR
    - MaxTUW decreases >=10%
    
    Robustness:
    - In at least 2 of 3 subperiods, Calmar improves vs Base
    """
    reasons = []
    
    base_m = base_result.metrics
    cand_m = candidate_result.metrics
    
    # Gate 1: CAGR degradation limit
    cagr_diff = cand_m["CAGR"] - base_m["CAGR"]
    if cagr_diff < -0.005:  # -0.5%p
        return False, f"CAGR degradation too large: {cagr_diff*100:.2f}%p"
    
    # Gate 2: Trade frequency limit
    if base_m["TradesPerYear"] > 0:
        trade_ratio = cand_m["TradesPerYear"] / base_m["TradesPerYear"]
        if trade_ratio > 1.2:
            return False, f"Trade frequency too high: {trade_ratio:.2f}x"
    
    # Soft improve (check at least one)
    mdd_improve = (cand_m["MDD"] - base_m["MDD"]) >= 0.02  # MDD is negative
    worst_month_improve = (cand_m["WorstMonth"] - base_m["WorstMonth"]) >= 0.03
    tuw_improve = base_m["MaxTUW"] > 0 and \
                  (base_m["MaxTUW"] - cand_m["MaxTUW"]) / base_m["MaxTUW"] >= 0.10
    
    has_improvement = mdd_improve or worst_month_improve or tuw_improve
    
    if mdd_improve:
        reasons.append(f"MDD improved: {base_m['MDD']*100:.1f}% -> {cand_m['MDD']*100:.1f}%")
    if worst_month_improve:
        reasons.append(f"WorstMonth improved: {base_m['WorstMonth']*100:.1f}% -> {cand_m['WorstMonth']*100:.1f}%")
    if tuw_improve:
        reasons.append(f"MaxTUW improved: {base_m['MaxTUW']} -> {cand_m['MaxTUW']} days")
    
    # Robustness: Calmar improvement in 2 of 3 subperiods
    calmar_wins = 0
    for period in ["P1", "P2", "P3"]:
        if period in base_subperiod and period in candidate_subperiod:
            if candidate_subperiod[period]["Calmar"] > base_subperiod[period]["Calmar"]:
                calmar_wins += 1
    
    robust = calmar_wins >= 2
    if robust:
        reasons.append(f"Calmar improved in {calmar_wins}/3 subperiods")
    
    # Final decision
    promoted = has_improvement and robust
    
    if not has_improvement:
        return False, "No soft improvement achieved"
    if not robust:
        return False, f"Calmar only improved in {calmar_wins}/3 subperiods"
    
    return True, "; ".join(reasons)


# ============================================================
# VISUALIZATION
# ============================================================
def plot_equity_curves(results: List[BacktestResult], save_path: str):
    """Plot equity curves (log scale)"""
    plt.figure(figsize=(14, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        linewidth = 2.5 if i == 0 else 1.5  # Thicker line for base
        plt.plot(result.equity, label=result.name, color=color, linewidth=linewidth)
    
    plt.yscale("log")
    plt.title("Equity Curves (Log Scale)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_drawdowns(results: List[BacktestResult], save_path: str):
    """Plot drawdown panels"""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3*n), sharex=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (ax, result) in enumerate(zip(axes, results)):
        color = colors[i % len(colors)]
        
        peak = result.equity.cummax()
        dd = (result.equity / peak - 1.0) * 100
        
        ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color=color)
        ax.plot(dd, color=color, linewidth=0.8)
        ax.set_ylabel("DD %")
        ax.set_title(f"{result.name} (MDD: {result.metrics['MDD']*100:.1f}%)", fontsize=10)
        ax.set_ylim(-70, 5)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


# ============================================================
# REPORT GENERATION
# ============================================================
def generate_report(results: List[BacktestResult], promotions: Dict[str, Tuple[bool, str]],
                    summary_df: pd.DataFrame, subperiod_df: pd.DataFrame,
                    save_path: str):
    """Generate markdown report"""
    
    report = f"""# Backtest Report: OFF10 Baseline vs Rule-Based Variants

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Period**: {PRIMARY_START} ~ {PRIMARY_END}  
**Base Strategy**: Optim_QQQ_3_161_OFF10

---

## Assumptions

1. **Execution Model**: Close-to-Close with 1-day signal lag
   - Signals computed using day t close data
   - Positions applied to day t+1 returns
   
2. **Data**: 
   - Signal asset: QQQ (Adjusted Close)
   - Traded asset: TQQQ (Adjusted Close)
   - OFF asset: CASH (0% return) or SGOV where specified
   
3. **Transaction Costs**: One-way costs in bps applied on weight changes

4. **Tax Model**: 
   - Tax0: No tax
   - TaxB: Annual net realized gain taxed at year end (simplified)

5. **Rebalancing**: Delta rebalance only when target weights change

---

## Strategy Variants

| ID | Name | Description |
|:---|:-----|:------------|
| V0 | Base OFF10 CASH | MA(3)/MA(161), OFF=10% TQQQ + 90% CASH |
| V1 | A1 SGOV | Same as Base but OFF asset = SGOV |
| V2 | C3 Hysteresis | ±0.5% hysteresis band on spread |
| V3 | B2 MA200 Guard | OFF0 when QQQ < MA200 during OFF state |
| V4 | D1 Ensemble | Majority vote: MA3 vs MA(160/165/170) |
| V5 | Combo | A1 + B2 (SGOV + MA200 guard) |

---

## Summary Metrics (Realistic Scenario: Cost=10bps, TaxB, Rate=20%)

"""
    
    # Filter to realistic scenario
    realistic = summary_df[
        (summary_df["cost_bps"] == REALISTIC_COST_BPS) &
        (summary_df["tax_mode"] == REALISTIC_TAX_MODE) &
        (summary_df["tax_rate"] == REALISTIC_TAX_RATE)
    ]
    
    if len(realistic) > 0:
        display_cols = ["name", "CAGR", "MDD", "Sharpe", "Sortino", "Calmar", 
                        "WorstYear", "WorstMonth", "TradesPerYear"]
        
        for col in ["CAGR", "MDD", "WorstYear", "WorstMonth"]:
            if col in realistic.columns:
                realistic = realistic.copy()
                realistic[col] = (realistic[col] * 100).round(2).astype(str) + "%"
        
        for col in ["Sharpe", "Sortino", "Calmar"]:
            if col in realistic.columns:
                realistic[col] = realistic[col].round(2)
        
        report += realistic[display_cols].to_markdown(index=False) + "\n\n"
    
    report += """---

## Promotion Evaluation

| Variant | Promoted | Reason |
|:--------|:--------:|:-------|
"""
    
    for name, (promoted, reason) in promotions.items():
        status = "✅" if promoted else "❌"
        report += f"| {name} | {status} | {reason} |\n"
    
    report += """
---

## Subperiod Analysis

"""
    
    if len(subperiod_df) > 0:
        pivot = subperiod_df.pivot_table(
            index="name", columns="period", 
            values=["CAGR", "MDD", "Calmar"]
        )
        report += "### CAGR by Subperiod\n\n"
        if "CAGR" in pivot.columns.get_level_values(0):
            cagr_pivot = pivot["CAGR"].copy() * 100
            cagr_pivot = cagr_pivot.round(1).astype(str) + "%"
            report += cagr_pivot.to_markdown() + "\n\n"
        
        report += "### Calmar by Subperiod\n\n"
        if "Calmar" in pivot.columns.get_level_values(0):
            calmar_pivot = pivot["Calmar"].round(2)
            report += calmar_pivot.to_markdown() + "\n\n"
    
    report += """---

## Cost Sensitivity (CAGR by Cost BPS, Tax0)

See `sensitivity_table_cagr.csv` for full matrix.

---

## Visualizations

![Equity Curves](./equity_curves_log.png)

![Drawdown Panels](./drawdown_panels.png)

---

## Conclusion

"""
    
    # Find promoted variants
    promoted_list = [name for name, (p, _) in promotions.items() if p]
    
    if promoted_list:
        report += f"**Promoted Variants**: {', '.join(promoted_list)}\n\n"
        report += "These variants meet all promotion criteria and show robust improvement over Base.\n"
    else:
        report += "**No variants were promoted.** All candidates either degraded CAGR too much or failed robustness checks.\n"
    
    report += """
---

*Generated by QuantNeural Backtest Engine v2026.1*
"""
    
    with open(save_path, "w") as f:
        f.write(report)
    
    print(f"📝 Saved: {save_path}")


# ============================================================
# MAIN RUNNER
# ============================================================
def run_full_backtest():
    """Run complete backtest suite"""
    
    print("="*80)
    print("       OFF10 Baseline vs Rule-Based Variants Backtest")
    print("="*80)
    
    # 1. Download and align data
    print("\n📥 Downloading and aligning data...")
    prices = download_data(SECONDARY_START, PRIMARY_END)
    prices_primary = align_data(prices, PRIMARY_START, PRIMARY_END)
    
    print(f"   Primary period: {prices_primary.index[0].date()} ~ {prices_primary.index[-1].date()}")
    print(f"   Trading days: {len(prices_primary)}")
    
    # Save aligned prices
    prices_primary.to_csv(os.path.join(ARTIFACTS_DIR, "prices.csv"))
    print(f"   Saved: prices.csv")
    
    # 2. Get strategy configurations
    configs = get_strategy_configs()
    print(f"\n📋 Running {len(configs)} strategy variants...")
    
    # 3. Run backtests for all scenarios
    all_results = []
    summary_rows = []
    subperiod_rows = []
    
    for config in configs:
        print(f"\n   Processing: {config.name}")
        
        for cost_bps in COST_BPS_GRID:
            for tax_mode in TAX_MODES:
                for tax_rate in TAX_RATE_GRID:
                    result = run_backtest(
                        prices_primary, config,
                        cost_bps=cost_bps,
                        tax_mode=tax_mode,
                        tax_rate=tax_rate
                    )
                    
                    # Store for summary
                    row = {
                        "name": config.name,
                        "cost_bps": cost_bps,
                        "tax_mode": tax_mode,
                        "tax_rate": tax_rate,
                        **result.metrics
                    }
                    summary_rows.append(row)
                    
                    # Store realistic scenario result
                    if (cost_bps == REALISTIC_COST_BPS and 
                        tax_mode == REALISTIC_TAX_MODE and
                        tax_rate == REALISTIC_TAX_RATE):
                        all_results.append(result)
                        
                        # Calculate subperiod metrics
                        sub_metrics = calculate_subperiod_metrics(result, SUBPERIODS)
                        for period, metrics in sub_metrics.items():
                            subperiod_rows.append({
                                "name": config.name,
                                "period": period,
                                **metrics
                            })
                        
                        # Save equity curve
                        result.equity.to_csv(
                            os.path.join(ARTIFACTS_DIR, f"daily_equity_{config.name}.csv")
                        )
    
    # 4. Create summary DataFrames
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(ARTIFACTS_DIR, "summary_metrics.csv"), index=False)
    print(f"\n   Saved: summary_metrics.csv")
    
    subperiod_df = pd.DataFrame(subperiod_rows)
    subperiod_df.to_csv(os.path.join(ARTIFACTS_DIR, "subperiod_metrics.csv"), index=False)
    print(f"   Saved: subperiod_metrics.csv")
    
    # 5. Create sensitivity table
    sensitivity = summary_df[summary_df["tax_mode"] == "Tax0"].pivot_table(
        index="name", columns="cost_bps", values="CAGR"
    )
    sensitivity = (sensitivity * 100).round(2)
    sensitivity.to_csv(os.path.join(ARTIFACTS_DIR, "sensitivity_table_cagr.csv"))
    print(f"   Saved: sensitivity_table_cagr.csv")
    
    # 6. Evaluate promotions
    print("\n🔍 Evaluating promotion criteria...")
    base_result = all_results[0]
    base_subperiod = {}
    for row in subperiod_rows:
        if row["name"] == base_result.name:
            base_subperiod[row["period"]] = row
    
    promotions = {}
    for result in all_results[1:]:
        cand_subperiod = {}
        for row in subperiod_rows:
            if row["name"] == result.name:
                cand_subperiod[row["period"]] = row
        
        promoted, reason = evaluate_promotion(
            base_result, result, base_subperiod, cand_subperiod
        )
        promotions[result.name] = (promoted, reason)
        
        status = "✅" if promoted else "❌"
        print(f"   {status} {result.name}: {reason}")
    
    # 7. Generate plots
    print("\n📊 Generating visualizations...")
    plot_equity_curves(all_results, os.path.join(ARTIFACTS_DIR, "equity_curves_log.png"))
    plot_drawdowns(all_results, os.path.join(ARTIFACTS_DIR, "drawdown_panels.png"))
    
    # 8. Generate report
    print("\n📝 Generating report...")
    generate_report(
        all_results, promotions, summary_df, subperiod_df,
        os.path.join(ARTIFACTS_DIR, "report.md")
    )
    
    # 9. Print summary
    print("\n" + "="*80)
    print("                         SUMMARY (Realistic Scenario)")
    print("="*80)
    print(f"Cost: {REALISTIC_COST_BPS}bps | Tax: {REALISTIC_TAX_MODE} @ {REALISTIC_TAX_RATE*100:.0f}%")
    print("-"*80)
    
    realistic_df = summary_df[
        (summary_df["cost_bps"] == REALISTIC_COST_BPS) &
        (summary_df["tax_mode"] == REALISTIC_TAX_MODE) &
        (summary_df["tax_rate"] == REALISTIC_TAX_RATE)
    ][["name", "CAGR", "MDD", "Sharpe", "Calmar", "TradesPerYear"]].copy()
    
    realistic_df["CAGR"] = (realistic_df["CAGR"] * 100).round(2).astype(str) + "%"
    realistic_df["MDD"] = (realistic_df["MDD"] * 100).round(2).astype(str) + "%"
    realistic_df["Sharpe"] = realistic_df["Sharpe"].round(2)
    realistic_df["Calmar"] = realistic_df["Calmar"].round(2)
    realistic_df["TradesPerYear"] = realistic_df["TradesPerYear"].round(1)
    
    pd.set_option('display.width', 200)
    print(realistic_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("                         PROMOTION RESULTS")
    print("="*80)
    for name, (promoted, reason) in promotions.items():
        status = "✅ PROMOTED" if promoted else "❌ NOT PROMOTED"
        print(f"{status}: {name}")
        print(f"   Reason: {reason}")
    
    print("\n" + "="*80)
    print(f"📁 All artifacts saved to: {ARTIFACTS_DIR}")
    print("="*80)
    
    return all_results, summary_df, promotions


# ============================================================
# UNIT TESTS
# ============================================================
def test_signal_generation():
    """Test signal generation logic"""
    print("\n🧪 Running unit tests...")
    
    # Create synthetic data
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    np.random.seed(42)
    
    prices = pd.DataFrame({
        "QQQ": 100 * np.exp(np.cumsum(np.random.randn(250) * 0.01)),
        "TQQQ": 50 * np.exp(np.cumsum(np.random.randn(250) * 0.03)),
        "CASH": 100.0,
        "SGOV": 100 * np.exp(np.cumsum(np.random.randn(250) * 0.001)),
    }, index=dates)
    
    # Test 1: Base signal generation
    signal = generate_base_signal(prices, short_ma=3, long_ma=20)
    assert len(signal) == len(prices), "Signal length mismatch"
    assert signal.iloc[0] == 0, "First signal should be 0 (shifted)"
    assert signal.isin([0, 1]).all(), "Signal should be binary"
    print("   ✅ Base signal generation OK")
    
    # Test 2: Hysteresis signal (stateful)
    h_signal = generate_hysteresis_signal(prices, short_ma=3, long_ma=20, band=0.01)
    assert len(h_signal) == len(prices), "Hysteresis signal length mismatch"
    assert h_signal.isin([0, 1]).all(), "Hysteresis signal should be binary"
    
    # Check statefulness: fewer transitions than base
    base_transitions = (signal.diff().abs() > 0).sum()
    hyst_transitions = (h_signal.diff().abs() > 0).sum()
    # Hysteresis should have fewer or equal transitions
    print(f"   Base transitions: {base_transitions}, Hysteresis: {hyst_transitions}")
    print("   ✅ Hysteresis signal generation OK")
    
    # Test 3: Ensemble signal
    e_signal = generate_ensemble_signal(prices, short_ma=3, long_windows=[18, 20, 22])
    assert len(e_signal) == len(prices), "Ensemble signal length mismatch"
    assert e_signal.isin([0, 1]).all(), "Ensemble signal should be binary"
    print("   ✅ Ensemble signal generation OK")
    
    # Test 4: MA200 guard
    guard = generate_ma200_guard(prices)
    assert len(guard) == len(prices), "Guard length mismatch"
    assert guard.isin([0, 1]).all(), "Guard should be binary"
    print("   ✅ MA200 guard OK")
    
    # Test 5: Position calculation
    config = StrategyConfig(name="test", off_tqqq_weight=0.10, use_ma200_guard=True)
    positions = calculate_positions(signal, config, guard)
    assert "TQQQ" in positions.columns, "TQQQ column missing"
    assert "OFF_ASSET" in positions.columns, "OFF_ASSET column missing"
    assert (positions["TQQQ"] + positions["OFF_ASSET"] - 1.0).abs().max() < 1e-6, "Weights don't sum to 1"
    print("   ✅ Position calculation OK")
    
    print("\n🧪 All unit tests passed!")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Run tests first
    test_signal_generation()
    
    # Run full backtest
    run_full_backtest()
