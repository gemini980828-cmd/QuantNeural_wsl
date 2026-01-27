# -*- coding: utf-8 -*-
"""
E03 Flash Crash Protocol Impact Analysis
=========================================

Compares E03_Ensemble_SGOV with and without Flash Crash emergency exit rules.

SSOT Rules (Part 4.2):
1. QQQ daily close drops >= 7% -> Force OFF10
2. TQQQ drops >= 20% from entry price -> Force OFF10

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
# FIXED PARAMETERS (Same as run_suite.py)
# ============================================================
COST_BPS = 10
TAX_RATE = 0.22
SHORT_MA = 3
OFF_TQQQ_WEIGHT = 0.10
TRADING_DAYS = 252

# Flash Crash thresholds (from SSOT Part 4.2)
QQQ_CRASH_THRESHOLD = -0.07   # -7% daily
TQQQ_STOP_THRESHOLD = -0.20   # -20% from entry

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/experiments/flash_crash_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    
    prices["CASH"] = 100.0
    
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
# SIGNAL GENERATION (E03 Ensemble)
# ============================================================
def generate_ensemble_signal(prices: pd.DataFrame, windows: List[int] = [160, 165, 170]) -> pd.Series:
    """E03 Ensemble signal: majority vote"""
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    
    votes = pd.DataFrame(index=prices.index)
    for lw in windows:
        ma_long = qqq.rolling(lw).mean()
        votes[f"w{lw}"] = (ma_short > ma_long).astype(int)
    
    threshold = len(windows) // 2 + 1
    return (votes.sum(axis=1) >= threshold).astype(int)


# ============================================================
# FLASH CRASH DETECTION
# ============================================================
def detect_flash_crash_events(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Flash Crash events based on SSOT rules:
    1. QQQ daily return <= -7%
    2. TQQQ drawdown from recent high >= -20% (simplified as daily return)
    
    Returns DataFrame with crash events
    """
    qqq_ret = prices["QQQ"].pct_change()
    tqqq_ret = prices["TQQQ"].pct_change()
    
    # Rule 1: QQQ crashes -7% or more in a day
    qqq_crash = qqq_ret <= QQQ_CRASH_THRESHOLD
    
    # Rule 2: TQQQ drops -20% or more (we track from entry, but simplified here)
    # For accurate tracking, we need entry price which requires iterative logic
    tqqq_crash = tqqq_ret <= TQQQ_STOP_THRESHOLD
    
    crash_events = pd.DataFrame({
        "qqq_ret": qqq_ret,
        "tqqq_ret": tqqq_ret,
        "qqq_crash": qqq_crash,
        "tqqq_crash": tqqq_crash,
        "any_crash": qqq_crash | tqqq_crash,
    }, index=prices.index)
    
    return crash_events


# ============================================================
# BACKTEST ENGINE WITH FLASH CRASH
# ============================================================
def run_backtest(
    prices: pd.DataFrame, 
    use_flash_crash: bool = False,
    name: str = "E03"
) -> Dict:
    """
    Run E03 backtest with optional Flash Crash protocol
    """
    off_asset = "SGOV"
    if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
        off_asset = "SHV" if "SHV" in prices.columns else "CASH"
    
    # Generate E03 ensemble signal
    signal_raw = generate_ensemble_signal(prices)
    signal_lagged = signal_raw.shift(1).fillna(0).astype(int)
    
    # Flash crash events
    crash_events = detect_flash_crash_events(prices)
    
    # Calculate returns
    qqq_ret = prices["QQQ"].pct_change().fillna(0.0)
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    off_ret = prices[off_asset].pct_change().fillna(0.0)
    
    # Iterative backtest with Flash Crash logic
    equity = []
    trades_list = []
    flash_crash_triggers = []
    
    portfolio_value = 1.0
    tqqq_cost_basis = 0.0
    tqqq_shares = 0.0
    tqqq_entry_price = None  # Track entry price for -20% rule
    yearly_gains = {}
    total_tax_paid = 0.0
    
    current_weight = 0.0
    prev_signal = 0
    flash_crash_active = False
    flash_crash_cooldown = 0  # Days to wait after flash crash
    
    for i, dt in enumerate(prices.index):
        px_tqqq = float(prices.loc[dt, "TQQQ"])
        px_qqq = float(prices.loc[dt, "QQQ"])
        
        # Get base signal
        base_signal = int(signal_lagged.loc[dt]) if not pd.isna(signal_lagged.loc[dt]) else 0
        
        # ============================================================
        # FLASH CRASH PROTOCOL (SSOT Part 4.2)
        # ============================================================
        flash_trigger = None
        if use_flash_crash and i > 0:
            # Rule 1: QQQ daily return <= -7%
            qqq_daily_ret = float(qqq_ret.loc[dt])
            if qqq_daily_ret <= QQQ_CRASH_THRESHOLD:
                flash_trigger = f"QQQ_CRASH ({qqq_daily_ret*100:.1f}%)"
            
            # Rule 2: TQQQ from entry price >= -20%
            if tqqq_entry_price is not None and current_weight > 0.5:  # ON state
                tqqq_drawdown = (px_tqqq / tqqq_entry_price) - 1.0
                if tqqq_drawdown <= TQQQ_STOP_THRESHOLD:
                    flash_trigger = f"TQQQ_STOP ({tqqq_drawdown*100:.1f}% from entry)"
        
        # Determine target weight
        if flash_trigger:
            # Force OFF10
            target_weight = OFF_TQQQ_WEIGHT
            flash_crash_triggers.append({
                "date": dt.strftime("%Y-%m-%d"),
                "trigger": flash_trigger,
                "qqq_price": px_qqq,
                "tqqq_price": px_tqqq,
                "tqqq_entry": tqqq_entry_price,
                "portfolio_value": portfolio_value,
            })
            flash_crash_active = True
            flash_crash_cooldown = 1  # Wait 1 day before normal signal resumes
        elif flash_crash_cooldown > 0:
            # Still in cooldown, maintain OFF10
            target_weight = OFF_TQQQ_WEIGHT
            flash_crash_cooldown -= 1
        else:
            # Normal signal
            target_weight = 1.0 if base_signal == 1 else OFF_TQQQ_WEIGHT
            flash_crash_active = False
        
        # Track entry price when going ON
        if target_weight > 0.5 and current_weight <= 0.5:
            tqqq_entry_price = px_tqqq
        elif target_weight <= 0.5:
            tqqq_entry_price = None
        
        # Calculate daily return using TARGET weight (same as run_suite.py)
        weight_change = abs(target_weight - current_weight)
        cost = weight_change * (COST_BPS / 10000.0) if weight_change > 1e-6 else 0.0
        
        # CRITICAL: Use target_weight for return calculation (not current_weight)
        # This matches run_suite.py line 423: port_ret_gross = target_tqqq * tqqq_ret + ...
        port_ret = target_weight * float(tqqq_ret.loc[dt]) + (1 - target_weight) * float(off_ret.loc[dt]) - cost
        portfolio_value *= (1 + port_ret)
        
        # Track trades
        if weight_change > 1e-6:
            side = "BUY" if target_weight > current_weight else "SELL"
            year = dt.year
            if year not in yearly_gains:
                yearly_gains[year] = 0.0
            
            if side == "SELL" and tqqq_shares > 0 and tqqq_cost_basis > 0:
                avg_cost = tqqq_cost_basis / tqqq_shares
                sold_fraction = abs(target_weight - current_weight)
                sold_value = sold_fraction * portfolio_value
                sold_shares = sold_value / px_tqqq if px_tqqq > 0 else 0
                gain = sold_shares * (px_tqqq - avg_cost)
                yearly_gains[year] += gain
                
                sell_ratio = min(1.0, sold_shares / tqqq_shares) if tqqq_shares > 0 else 0
                tqqq_cost_basis *= (1 - sell_ratio)
                tqqq_shares -= sold_shares
            elif side == "BUY":
                buy_value = (target_weight - current_weight) * portfolio_value
                buy_shares = buy_value / px_tqqq if px_tqqq > 0 else 0
                tqqq_cost_basis += buy_value
                tqqq_shares += buy_shares
            
            trades_list.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "side": side,
                "from_weight": current_weight,
                "to_weight": target_weight,
                "cost": cost * portfolio_value,
                "flash_crash": flash_trigger is not None,
            })
        
        current_weight = target_weight
        prev_signal = base_signal
        
        # Year-end tax
        is_year_end = (i == len(prices.index) - 1) or \
                      (prices.index[i+1].year != dt.year if i < len(prices.index) - 1 else True)
        
        if is_year_end and dt.year in yearly_gains:
            taxable = max(0, yearly_gains[dt.year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            total_tax_paid += tax
        
        equity.append({"date": dt, "equity": portfolio_value})
    
    # Build result
    equity_df = pd.DataFrame(equity).set_index("date")
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
    flash_df = pd.DataFrame(flash_crash_triggers) if flash_crash_triggers else pd.DataFrame()
    
    # Calculate metrics
    equity_series = equity_df["equity"]
    n_years = len(equity_series) / TRADING_DAYS
    final = float(equity_series.iloc[-1])
    
    cagr = (final ** (1.0 / n_years) - 1.0) if n_years > 0 and final > 0 else 0.0
    
    returns = equity_series.pct_change().fillna(0.0)
    peak = equity_series.cummax()
    drawdown = equity_series / peak - 1.0
    mdd = float(drawdown.min())
    
    daily_std = returns.std(ddof=0)
    sharpe = (returns.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0
    
    metrics = {
        "name": name,
        "use_flash_crash": use_flash_crash,
        "Final": final,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "TotalTax": total_tax_paid,
        "NumTrades": len(trades_df),
        "FlashCrashTriggers": len(flash_df),
    }
    
    return {
        "equity": equity_df,
        "trades": trades_df,
        "flash_crashes": flash_df,
        "metrics": metrics,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================
def run_comparison():
    print("=" * 80)
    print("       E03 Flash Crash Protocol Impact Analysis")
    print("=" * 80)
    
    prices = load_data()
    
    # Run both variants
    print("\n[1/2] Running E03 Baseline (no Flash Crash)...")
    baseline = run_backtest(prices, use_flash_crash=False, name="E03_Baseline")
    print(f"       CAGR: {baseline['metrics']['CAGR']*100:.2f}% | MDD: {baseline['metrics']['MDD']*100:.2f}%")
    
    print("\n[2/2] Running E03 + Flash Crash Protocol...")
    with_fc = run_backtest(prices, use_flash_crash=True, name="E03_FlashCrash")
    print(f"       CAGR: {with_fc['metrics']['CAGR']*100:.2f}% | MDD: {with_fc['metrics']['MDD']*100:.2f}%")
    print(f"       Flash Crash triggers: {with_fc['metrics']['FlashCrashTriggers']}")
    
    # Save results
    print("\n" + "=" * 80)
    print("                         COMPARISON")
    print("=" * 80)
    
    comparison = pd.DataFrame([
        baseline["metrics"],
        with_fc["metrics"],
    ])
    
    comparison["CAGR_pct"] = comparison["CAGR"] * 100
    comparison["MDD_pct"] = comparison["MDD"] * 100
    
    print(f"\n{'Variant':<25} {'Final':>10} {'CAGR':>10} {'MDD':>10} {'Sharpe':>8} {'FC Triggers':>12}")
    print("-" * 80)
    for _, row in comparison.iterrows():
        print(f"{row['name']:<25} {row['Final']:>10.2f}x {row['CAGR_pct']:>9.2f}% "
              f"{row['MDD_pct']:>9.2f}% {row['Sharpe']:>8.2f} {row['FlashCrashTriggers']:>12}")
    
    # Delta analysis
    delta_cagr = (with_fc["metrics"]["CAGR"] - baseline["metrics"]["CAGR"]) * 100
    delta_mdd = (with_fc["metrics"]["MDD"] - baseline["metrics"]["MDD"]) * 100
    
    print("\n" + "-" * 80)
    print(f"DELTA (FlashCrash - Baseline):")
    print(f"  CAGR: {delta_cagr:+.2f}%p")
    print(f"  MDD:  {delta_mdd:+.2f}%p (positive = less drawdown)")
    
    # Flash crash events detail
    if len(with_fc["flash_crashes"]) > 0:
        print(f"\n📊 Flash Crash Events ({len(with_fc['flash_crashes'])} triggers):")
        print("-" * 80)
        for _, event in with_fc["flash_crashes"].iterrows():
            print(f"  {event['date']}: {event['trigger']}")
    
    # Save artifacts
    comparison.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False)
    baseline["equity"].to_csv(os.path.join(OUTPUT_DIR, "equity_baseline.csv"))
    with_fc["equity"].to_csv(os.path.join(OUTPUT_DIR, "equity_flash_crash.csv"))
    
    if len(with_fc["flash_crashes"]) > 0:
        with_fc["flash_crashes"].to_csv(os.path.join(OUTPUT_DIR, "flash_crash_events.csv"), index=False)
    
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "baseline": baseline["metrics"],
            "flash_crash": with_fc["metrics"],
            "delta_cagr_pct": delta_cagr,
            "delta_mdd_pct": delta_mdd,
        }, f, indent=2)
    
    # Plot comparison
    plt.figure(figsize=(14, 6))
    plt.plot(baseline["equity"]["equity"], label="E03 Baseline", lw=1.5)
    plt.plot(with_fc["equity"]["equity"], label="E03 + Flash Crash", lw=1.5, linestyle="--")
    
    # Mark flash crash events
    if len(with_fc["flash_crashes"]) > 0:
        for _, event in with_fc["flash_crashes"].iterrows():
            plt.axvline(pd.to_datetime(event["date"]), color="red", alpha=0.3, linestyle=":")
    
    plt.yscale("log")
    plt.title("E03 vs E03 + Flash Crash Protocol")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (log)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_chart.png"), dpi=150)
    plt.close()
    
    print(f"\n📁 Saved to: {OUTPUT_DIR}")
    
    return baseline, with_fc, comparison


if __name__ == "__main__":
    run_comparison()
