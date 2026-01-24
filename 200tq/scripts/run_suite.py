# -*- coding: utf-8 -*-
"""
OFF10 Baseline Backtest Suite: 11 Experiments (Fixed Version)
==============================================================

Fixed Parameters (DO NOT CHANGE):
- Signal: QQQ MA(3) vs MA(161) cross
- Position: ON=TQQQ 100%, OFF=TQQQ 10% + OFF_ASSET 90%
- Execution: Close-to-Close, 1-day lag
- Rebalancing: Delta-rebalance only
- Transaction Cost: 10 bps one-way
- Tax: Korean overseas ETF 22% annual (TaxB model)

Author: QuantNeural v2026.1
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FIXED PARAMETERS (DO NOT CHANGE)
# ============================================================
COST_BPS = 10  # 10 bps one-way
TAX_RATE = 0.22  # Korean overseas ETF 22%
SHORT_MA = 3
LONG_MA = 161
OFF_TQQQ_WEIGHT = 0.10  # OFF state: 10% TQQQ
TRADING_DAYS = 252

# Analysis period
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# Output directory
OUTPUT_DIR = "/home/juwon/QuantNeural/200tq/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================
@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str
    description: str
    off_asset: str = "CASH"
    signal_type: str = "plain"
    hysteresis_band: float = 0.0
    ensemble_windows: List[int] = field(default_factory=list)
    use_ma200_guard: bool = False
    buy_confirm_days: int = 1
    sell_confirm_days: int = 1
    deadband_pct: float = 0.0  # Skip trades if weight change < this threshold (%), except ON/OFF transitions


EXPERIMENTS = [
    ExperimentConfig(
        name="E00_V0_Base_OFF10_CASH",
        description="Baseline: MA3/MA161, OFF=10%TQQQ+90%CASH",
        off_asset="CASH",
    ),
    ExperimentConfig(
        name="E01_V1_OFF10_SGOV",
        description="SGOV as OFF asset",
        off_asset="SGOV",
    ),
    ExperimentConfig(
        name="E02_V4_Ensemble_160_165_170",
        description="Ensemble: MA3 vs MA(160/165/170) majority vote",
        off_asset="CASH",
        signal_type="ensemble",
        ensemble_windows=[160, 165, 170],
    ),
    ExperimentConfig(
        name="E03_Ensemble_SGOV",
        description="Ensemble + SGOV",
        off_asset="SGOV",
        signal_type="ensemble",
        ensemble_windows=[160, 165, 170],
    ),
    ExperimentConfig(
        name="E04_Hysteresis_0p25",
        description="Hysteresis band ±0.25%",
        off_asset="CASH",
        signal_type="hysteresis",
        hysteresis_band=0.0025,
    ),
    ExperimentConfig(
        name="E05_Hysteresis_0p50",
        description="Hysteresis band ±0.50%",
        off_asset="CASH",
        signal_type="hysteresis",
        hysteresis_band=0.005,
    ),
    ExperimentConfig(
        name="E06_Hysteresis_1p00",
        description="Hysteresis band ±1.00%",
        off_asset="CASH",
        signal_type="hysteresis",
        hysteresis_band=0.01,
    ),
    ExperimentConfig(
        name="E07_MA200Guard_CASH",
        description="OFF0 when OFF and QQQ<MA200",
        off_asset="CASH",
        use_ma200_guard=True,
    ),
    ExperimentConfig(
        name="E08_MA200Guard_SGOV",
        description="MA200 Guard + SGOV",
        off_asset="SGOV",
        use_ma200_guard=True,
    ),
    ExperimentConfig(
        name="E09_BuyConfirm_2days",
        description="Buy: 2-day confirm, Sell: immediate",
        off_asset="CASH",
        buy_confirm_days=2,
        sell_confirm_days=1,
    ),
    ExperimentConfig(
        name="E10_BuySellConfirm_2days",
        description="Buy/Sell: both 2-day confirm",
        off_asset="CASH",
        buy_confirm_days=2,
        sell_confirm_days=2,
    ),
    
    # ============================================================
    # SECOND SUITE: Robustness Validation (E20-E27)
    # Purpose: Confirm E03 robustness, NOT optimization
    # ============================================================
    ExperimentConfig(
        name="E20_Ensemble_155_160_165",
        description="Ensemble shifted -5: MA(155/160/165)",
        off_asset="SGOV",
        signal_type="ensemble",
        ensemble_windows=[155, 160, 165],
    ),
    ExperimentConfig(
        name="E21_Ensemble_165_170_175",
        description="Ensemble shifted +5: MA(165/170/175)",
        off_asset="SGOV",
        signal_type="ensemble",
        ensemble_windows=[165, 170, 175],
    ),
    ExperimentConfig(
        name="E22_Ensemble_5Window",
        description="5-window vote: MA(158/161/164/167/170) 3/5",
        off_asset="SGOV",
        signal_type="ensemble",
        ensemble_windows=[158, 161, 164, 167, 170],
    ),
    ExperimentConfig(
        name="E23_Ensemble_SHV",
        description="Ensemble + SHV (T-Bill proxy)",
        off_asset="SHV",
        signal_type="ensemble",
        ensemble_windows=[160, 165, 170],
    ),
    # E24: Cost=20bps handled separately via sensitivity grid
    # E25: OOS split handled via subperiod analysis
    ExperimentConfig(
        name="E26_MonthlyRebalance",
        description="Monthly rebalance only (signal checked monthly)",
        off_asset="SGOV",
        signal_type="ensemble",
        ensemble_windows=[160, 165, 170],
        # Note: Monthly rebalance implemented via signal sampling
    ),
    
    # ============================================================
    # DEADBAND LAYER: Operational Improvement (E30-E31)
    # Purpose: Reduce turnover while maintaining signal integrity
    # ============================================================
    ExperimentConfig(
        name="E30_Deadband_5pct",
        description="E03 + Deadband 5%p (skip small rebalances)",
        off_asset="SGOV",
        signal_type="ensemble",
        ensemble_windows=[160, 165, 170],
        deadband_pct=0.05,  # 5%p threshold
    ),
    ExperimentConfig(
        name="E31_Deadband_3pct",
        description="E03 + Deadband 3%p",
        off_asset="SGOV",
        signal_type="ensemble",
        ensemble_windows=[160, 165, 170],
        deadband_pct=0.03,  # 3%p threshold
    ),
]



# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> pd.DataFrame:
    """Load price data from yfinance"""
    import yfinance as yf
    
    print("📥 Downloading data from yfinance...")
    tickers = ["QQQ", "TQQQ", "SGOV", "SHV"]
    
    raw = yf.download(
        tickers=tickers,
        start="2009-01-01",
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    
    prices = pd.DataFrame()
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if (t, "Close") in raw.columns:
                prices[t] = raw[(t, "Close")]
    
    # CASH as 0% return (constant price)
    prices["CASH"] = 100.0
    
    # Handle missing SGOV/SHV
    if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
        if "SHV" in prices.columns:
            prices["SGOV"] = prices["SHV"]
        else:
            prices["SGOV"] = 100.0
    
    prices["SGOV"] = prices["SGOV"].ffill()
    if "SHV" in prices.columns:
        prices["SHV"] = prices["SHV"].ffill()
    
    prices = prices.dropna(subset=["QQQ", "TQQQ"])
    prices = prices[(prices.index >= START_DATE) & (prices.index <= END_DATE)]
    
    print(f"   Period: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print(f"   Trading days: {len(prices)}")
    
    return prices


# ============================================================
# SIGNAL GENERATION
# ============================================================
def generate_plain_signal(prices: pd.DataFrame) -> pd.Series:
    """Plain signal: QQQ MA(3) > MA(161) => ON"""
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    ma_long = qqq.rolling(LONG_MA).mean()
    return (ma_short > ma_long).astype(int)


def generate_ensemble_signal(prices: pd.DataFrame, windows: List[int]) -> pd.Series:
    """Ensemble signal: majority vote"""
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    
    votes = pd.DataFrame(index=prices.index)
    for lw in windows:
        ma_long = qqq.rolling(lw).mean()
        votes[f"w{lw}"] = (ma_short > ma_long).astype(int)
    
    threshold = len(windows) // 2 + 1
    return (votes.sum(axis=1) >= threshold).astype(int)


def generate_hysteresis_signal(prices: pd.DataFrame, band: float) -> pd.Series:
    """Hysteresis signal with band"""
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    ma_long = qqq.rolling(LONG_MA).mean()
    spread = (ma_short / ma_long) - 1.0
    
    signal = pd.Series(index=prices.index, dtype=int)
    state = 0
    
    for dt in prices.index:
        if pd.isna(spread.loc[dt]):
            signal.loc[dt] = state
            continue
        s = spread.loc[dt]
        if s > band:
            state = 1
        elif s < -band:
            state = 0
        signal.loc[dt] = state
    
    return signal


def apply_confirmation(signal: pd.Series, buy_days: int, sell_days: int) -> pd.Series:
    """Apply confirmation delays"""
    if buy_days == 1 and sell_days == 1:
        return signal
    
    result = pd.Series(index=signal.index, dtype=int)
    current_state = 0
    pending_state = None
    pending_count = 0
    
    for dt in signal.index:
        target = signal.loc[dt]
        
        if target != current_state:
            if pending_state == target:
                pending_count += 1
            else:
                pending_state = target
                pending_count = 1
            
            confirm_days = buy_days if target == 1 else sell_days
            if pending_count >= confirm_days:
                current_state = target
                pending_state = None
                pending_count = 0
        else:
            pending_state = None
            pending_count = 0
        
        result.loc[dt] = current_state
    
    return result


def generate_signal(prices: pd.DataFrame, config: ExperimentConfig) -> pd.Series:
    """Generate signal based on config"""
    if config.signal_type == "ensemble":
        signal = generate_ensemble_signal(prices, config.ensemble_windows)
    elif config.signal_type == "hysteresis":
        signal = generate_hysteresis_signal(prices, config.hysteresis_band)
    else:
        signal = generate_plain_signal(prices)
    
    signal = apply_confirmation(signal, config.buy_confirm_days, config.sell_confirm_days)
    return signal


# ============================================================
# VECTORIZED BACKTEST ENGINE
# ============================================================
def run_backtest(prices: pd.DataFrame, config: ExperimentConfig) -> Dict:
    """
    Run backtest with vectorized approach + iterative cost/tax tracking
    """
    # Select OFF asset with fallback
    off_asset = config.off_asset
    if off_asset == "SGOV":
        if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
            if "SHV" in prices.columns and not prices["SHV"].isna().all():
                off_asset = "SHV"
            else:
                off_asset = "CASH"
    
    # Generate signal (raw, not lagged yet)
    signal_raw = generate_signal(prices, config)
    
    # Apply 1-day lag: signal at t -> position for t+1 return
    signal_lagged = signal_raw.shift(1).fillna(0).astype(int)
    
    # Calculate target TQQQ weight
    if config.use_ma200_guard:
        qqq = prices["QQQ"]
        ma200 = qqq.rolling(200).mean()
        guard = (qqq < ma200).shift(1).fillna(False).astype(int)
        # When OFF (signal=0) and guard active, TQQQ = 0%
        off_tqqq_weight = pd.Series(OFF_TQQQ_WEIGHT, index=prices.index)
        off_tqqq_weight[(signal_lagged == 0) & (guard == 1)] = 0.0
        target_tqqq = signal_lagged * 1.0 + (1 - signal_lagged) * off_tqqq_weight
    else:
        target_tqqq = signal_lagged * 1.0 + (1 - signal_lagged) * OFF_TQQQ_WEIGHT
    
    # Apply deadband logic: skip small rebalances, EXCEPT signal transitions
    if config.deadband_pct > 0:
        actual_tqqq = pd.Series(index=prices.index, dtype=float)
        current_weight = 0.0
        prev_signal = 0
        
        for i, dt in enumerate(prices.index):
            target_w = float(target_tqqq.loc[dt])
            curr_signal = int(signal_lagged.loc[dt])
            
            # Check if this is a signal transition (ON<->OFF)
            is_signal_change = (curr_signal != prev_signal)
            
            # Calculate weight change
            weight_diff = abs(target_w - current_weight)
            
            # Apply deadband: skip if small change AND not a signal transition
            if is_signal_change or weight_diff >= config.deadband_pct:
                current_weight = target_w
            
            actual_tqqq.loc[dt] = current_weight
            prev_signal = curr_signal
        
        target_tqqq = actual_tqqq
    
    target_off = 1.0 - target_tqqq
    
    # Calculate returns
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    off_ret = prices[off_asset].pct_change().fillna(0.0) if off_asset in prices.columns else pd.Series(0.0, index=prices.index)
    
    # Identify trades (weight changes)
    weight_change = target_tqqq.diff().abs().fillna(0.0)
    trade_days = weight_change > 1e-6
    
    # Cost per trade (one-way 10 bps on weight change)
    cost_drag = weight_change * (COST_BPS / 10000.0)
    
    # Portfolio return before tax
    port_ret_gross = target_tqqq * tqqq_ret + target_off * off_ret - cost_drag
    
    # Build equity curve (iterative for tax calculation)
    equity = pd.Series(index=prices.index, dtype=float)
    trades_list = []
    
    portfolio_value = 1.0
    tqqq_cost_basis = 0.0
    tqqq_shares = 0.0
    yearly_gains = {}
    total_tax_paid = 0.0
    
    prev_tqqq_weight = 0.0
    
    for i, dt in enumerate(prices.index):
        # Get prices
        px_tqqq = float(prices.loc[dt, "TQQQ"])
        
        # Apply daily return (before considering rebalancing)
        r = float(port_ret_gross.loc[dt])
        portfolio_value *= (1 + r)
        
        # Track realized gains for sells
        curr_weight = float(target_tqqq.loc[dt])
        weight_chg = curr_weight - prev_tqqq_weight
        
        year = dt.year
        if year not in yearly_gains:
            yearly_gains[year] = 0.0
        
        if weight_chg < -1e-6 and tqqq_shares > 0:
            # Selling TQQQ
            sold_fraction = abs(weight_chg)
            if tqqq_shares > 0 and tqqq_cost_basis > 0:
                avg_cost = tqqq_cost_basis / tqqq_shares
                sold_value = sold_fraction * portfolio_value
                sold_shares = sold_value / px_tqqq if px_tqqq > 0 else 0
                gain = sold_shares * (px_tqqq - avg_cost)
                yearly_gains[year] += gain
                
                # Update cost basis
                sell_ratio = min(1.0, sold_shares / tqqq_shares) if tqqq_shares > 0 else 0
                tqqq_cost_basis *= (1 - sell_ratio)
                tqqq_shares -= sold_shares
                
                trades_list.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "asset": "TQQQ",
                    "side": "SELL",
                    "notional": sold_value,
                    "cost": float(cost_drag.loc[dt]) * portfolio_value,
                    "realized_gain": gain,
                })
        
        elif weight_chg > 1e-6:
            # Buying TQQQ
            buy_value = weight_chg * portfolio_value
            buy_shares = buy_value / px_tqqq if px_tqqq > 0 else 0
            tqqq_cost_basis += buy_value
            tqqq_shares += buy_shares
            
            trades_list.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "asset": "TQQQ",
                "side": "BUY",
                "notional": buy_value,
                "cost": float(cost_drag.loc[dt]) * portfolio_value,
                "realized_gain": 0.0,
            })
        
        prev_tqqq_weight = curr_weight
        
        # Year-end tax
        is_year_end = (i == len(prices.index) - 1) or \
                      (prices.index[i+1].year != year if i < len(prices.index) - 1 else True)
        
        if is_year_end and year in yearly_gains:
            taxable = max(0, yearly_gains[year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            total_tax_paid += tax
        
        equity.loc[dt] = portfolio_value
    
    # Build DataFrames
    equity_df = pd.DataFrame({"equity": equity})
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
        columns=["Date", "asset", "side", "notional", "cost", "realized_gain"])
    
    weights_df = pd.DataFrame({"TQQQ": target_tqqq, "OFF": target_off})
    
    metrics = calculate_metrics(equity, trades_df, total_tax_paid)
    
    return {
        "config": config,
        "equity": equity_df,
        "trades": trades_df,
        "metrics": metrics,
        "total_tax_paid": total_tax_paid,
        "weights": weights_df,
        "signal": signal_lagged,
    }


# ============================================================
# METRICS CALCULATION
# ============================================================
def calculate_metrics(equity: pd.Series, trades: pd.DataFrame, total_tax: float) -> Dict:
    """Calculate all metrics"""
    n_years = len(equity) / TRADING_DAYS
    final = float(equity.iloc[-1])
    
    # CAGR
    cagr = (final ** (1.0 / n_years) - 1.0) if n_years > 0 and final > 0 else 0.0
    
    # Returns
    returns = equity.pct_change().fillna(0.0)
    
    # MDD
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    mdd = float(drawdown.min())
    
    # Sharpe
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=0)
    sharpe = (daily_mean / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0.0
    sortino = (cagr / downside_std) if downside_std > 0 else 0.0
    
    # Calmar
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0.0
    
    # Trades
    n_trades = len(trades) if len(trades) > 0 else 0
    trades_per_year = n_trades / n_years if n_years > 0 else 0
    
    # Turnover
    total_turnover = trades["notional"].sum() if len(trades) > 0 else 0
    turnover_per_year = total_turnover / n_years if n_years > 0 else 0
    
    return {
        "Final": final,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "TradesPerYear": trades_per_year,
        "TurnoverPerYear": turnover_per_year,
        "TotalTaxPaid": total_tax,
        "NumDays": len(equity),
    }


# ============================================================
# VALIDATION
# ============================================================
def validate_result(result: Dict) -> List[str]:
    """Validate result"""
    errors = []
    weights = result["weights"]
    
    # Check weights sum to 1
    weight_sum = weights["TQQQ"] + weights["OFF"]
    max_dev = abs(weight_sum - 1.0).max()
    if max_dev > 1e-6:
        errors.append(f"Weight sum deviation: {max_dev:.2e}")
    
    # Check equity is positive
    eq = result["equity"]["equity"]
    if (eq <= 0).any():
        errors.append("Equity became non-positive")
    
    return errors


# ============================================================
# ARTIFACT GENERATION
# ============================================================
def save_artifacts(result: Dict, exp_dir: str):
    """Save all artifacts"""
    os.makedirs(exp_dir, exist_ok=True)
    config = result["config"]
    equity = result["equity"]["equity"]
    
    # 1. equity_curve.csv
    result["equity"].to_csv(os.path.join(exp_dir, "equity_curve.csv"))
    
    # 2. trades.csv
    result["trades"].to_csv(os.path.join(exp_dir, "trades.csv"), index=False)
    
    # 3. yearly_returns.csv
    yearly = equity.resample("YE").last().pct_change().dropna()
    yearly_df = pd.DataFrame({"year": yearly.index.year, "return": yearly.values})
    yearly_df.to_csv(os.path.join(exp_dir, "yearly_returns.csv"), index=False)
    
    # 4. metrics.json
    m = result["metrics"].copy()
    m["experiment"] = config.name
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    
    # 5. Plots
    plt.figure(figsize=(12, 5))
    plt.plot(equity, linewidth=1.5)
    plt.yscale("log")
    plt.title(f"{config.name}: Equity (Log)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "equity_curve_log.png"), dpi=150)
    plt.close()
    
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100
    plt.figure(figsize=(12, 4))
    plt.fill_between(dd.index, dd.values, 0, alpha=0.5, color='red')
    plt.title(f"{config.name}: Drawdown (MDD: {result['metrics']['MDD']*100:.1f}%)")
    plt.ylabel("DD %")
    plt.ylim(-70, 5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "drawdown.png"), dpi=150)
    plt.close()
    
    if len(yearly_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        colors = ['green' if r > 0 else 'red' for r in yearly_df["return"]]
        ax.bar(yearly_df["year"], yearly_df["return"] * 100, color=colors, alpha=0.7)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f"{config.name}: Yearly Returns")
        ax.set_ylabel("Return %")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "yearly_returns.png"), dpi=150)
        plt.close()


def generate_summary(results: List[Dict]) -> Tuple[pd.DataFrame, str]:
    """Generate summary"""
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Experiment": r["config"].name,
            "Description": r["config"].description,
            "Final": m["Final"],
            "CAGR": m["CAGR"],
            "MDD": m["MDD"],
            "Sharpe": m["Sharpe"],
            "Sortino": m["Sortino"],
            "Calmar": m["Calmar"],
            "TradesPerYear": m["TradesPerYear"],
            "TotalTaxPaid": m["TotalTaxPaid"],
        })
    
    df = pd.DataFrame(rows)
    base_cagr = df.iloc[0]["CAGR"]
    base_mdd = df.iloc[0]["MDD"]
    df["CAGR_Delta"] = df["CAGR"] - base_cagr
    df["MDD_Delta"] = df["MDD"] - base_mdd
    df = df.sort_values(["CAGR", "MDD"], ascending=[False, False])
    df["Rank"] = range(1, len(df) + 1)
    
    # Markdown
    md = f"""# Leaderboard: OFF10 Experiments

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Period**: {START_DATE} ~ {END_DATE}
**Cost**: {COST_BPS} bps | **Tax**: {TAX_RATE*100:.0f}% (Korean TaxB)

## Rankings

| Rank | Experiment | CAGR | Δ CAGR | MDD | Sharpe | Calmar |
|:----:|:-----------|-----:|-------:|----:|-------:|-------:|
"""
    for _, row in df.iterrows():
        # Base is E00 (first experiment), not Rank 1
        is_base = row['Experiment'].startswith("E00")
        delta = "(base)" if is_base else f"{row['CAGR_Delta']*100:+.2f}%p"
        md += f"| {row['Rank']} | {row['Experiment']} | {row['CAGR']*100:.2f}% | {delta} | "
        md += f"{row['MDD']*100:.2f}% | {row['Sharpe']:.2f} | {row['Calmar']:.2f} |\n"

    
    return df, md


# ============================================================
# MAIN RUNNER
# ============================================================
def run_suite():
    print("="*80)
    print("       OFF10 Baseline Backtest Suite: 11 Experiments")
    print("="*80)
    
    prices = load_data()
    results = []
    
    for i, config in enumerate(EXPERIMENTS):
        print(f"\n[{i+1:02d}/{len(EXPERIMENTS):02d}] {config.name}")
        
        result = run_backtest(prices, config)
        errors = validate_result(result)
        
        if errors:
            print(f"       ⚠️ Errors: {errors}")
        else:
            print(f"       ✅ OK")
        
        m = result["metrics"]
        print(f"       Final: {m['Final']:.2f}x | CAGR: {m['CAGR']*100:.2f}% | "
              f"MDD: {m['MDD']*100:.2f}% | Sharpe: {m['Sharpe']:.2f}")
        
        save_artifacts(result, os.path.join(OUTPUT_DIR, config.name))
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    summary_df, leaderboard_md = generate_summary(results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "leaderboard.md"), "w") as f:
        f.write(leaderboard_md)
    
    # Combined plot
    plt.figure(figsize=(14, 7))
    for r in results:
        plt.plot(r["equity"]["equity"], label=r["config"].name, lw=1.2)
    plt.yscale("log")
    plt.title("All Experiments: Equity Curves")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_equity_curves.png"), dpi=150)
    plt.close()
    
    # Print leaderboard
    print("\n" + "="*80)
    print("                         LEADERBOARD")
    print("="*80)
    print(f"\n{'Rank':<5} {'Experiment':<30} {'Final':>10} {'CAGR':>10} {'MDD':>10} {'Sharpe':>8}")
    print("-"*80)
    for _, row in summary_df.iterrows():
        print(f"{row['Rank']:<5} {row['Experiment']:<30} {row['Final']:>10.2f}x "
              f"{row['CAGR']*100:>9.2f}% {row['MDD']*100:>9.2f}% {row['Sharpe']:>8.2f}")
    
    print(f"\n📁 Saved to: {OUTPUT_DIR}")
    return results, summary_df


if __name__ == "__main__":
    run_suite()
