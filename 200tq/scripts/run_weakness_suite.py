# -*- coding: utf-8 -*-
"""
E03 Weakness Suite: 7 Experiments (E40-E46)
============================================

Addressing identified weaknesses in E03 SSOT v2026.1:

  E40  SameDay_Close      Remove 1-day lag (trade at same-day close)
  E41  WalkForward_OOS    IS 2010-2017 → OOS 2018-2025
  E42  Benchmark_Package  E03 vs QQQ B&H, TQQQ B&H, QQQ+SMA200, TQQQ+SMA200
  E43  Emergency_5pct     Emergency exit: QQQ -5% / TQQQ -15%
  E44  Emergency_7pct     Emergency exit: QQQ -7% / TQQQ -20% (current SSOT)
  E45  Emergency_10pct    Emergency exit: QQQ -10% / TQQQ -30%
  E46  VolDrag_Analysis   Quantify TQQQ 3x volatility drag

Constraint: No changes to existing scripts — standalone file only.

Author: QuantNeural Weakness Suite v2026.2
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONSTANTS (same as run_suite.py)
# ============================================================
COST_BPS = 10
TAX_RATE = 0.22
SHORT_MA = 3
OFF_TQQQ_WEIGHT = 0.10
TRADING_DAYS = 252

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
ENSEMBLE_WINDOWS = [160, 165, 170]

OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/experiments/weakness_suite"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING (identical to run_suite.py)
# ============================================================
def load_data() -> pd.DataFrame:
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
# SIGNAL GENERATION
# ============================================================
def generate_ensemble_signal(
    prices: pd.DataFrame, windows: Optional[List[int]] = None
) -> pd.Series:
    if windows is None:
        windows = ENSEMBLE_WINDOWS
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    votes = pd.DataFrame(index=prices.index)
    for lw in windows:
        ma_long = qqq.rolling(lw).mean()
        votes[f"w{lw}"] = (ma_short > ma_long).astype(int)
    threshold = len(windows) // 2 + 1
    return (votes.sum(axis=1) >= threshold).astype(int)


# ============================================================
# METRICS CALCULATION
# ============================================================
def calculate_metrics(
    equity_series: pd.Series, n_trades: int = 0, total_tax: float = 0.0
) -> Dict:
    n_years = len(equity_series) / TRADING_DAYS
    final = float(equity_series.iloc[-1])
    cagr = (final ** (1.0 / n_years) - 1.0) if n_years > 0 and final > 0 else 0.0

    returns = equity_series.pct_change().fillna(0.0)

    peak = equity_series.cummax()
    drawdown = equity_series / peak - 1.0
    mdd = float(drawdown.min())

    daily_std = returns.std(ddof=0)
    sharpe = (
        (returns.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0
    )

    downside = returns[returns < 0]
    downside_std = (
        downside.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0.0
    )
    sortino = (cagr / downside_std) if downside_std > 0 else 0.0
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0.0

    return {
        "Final": final,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "NumTrades": n_trades,
        "TotalTaxPaid": total_tax,
        "NumDays": len(equity_series),
    }


# ============================================================
# CORE E03 BACKTEST ENGINE (parameterised)
# ============================================================
def run_e03_backtest(
    prices: pd.DataFrame,
    lag: int = 1,
    windows: Optional[List[int]] = None,
    emergency_qqq: Optional[float] = None,
    emergency_tqqq: Optional[float] = None,
) -> Dict:
    """
    Parameterised E03 engine.

    - lag: 0 → same-day close execution; 1 → next-day (default).
    - windows: ensemble MA windows (default [160,165,170]).
    - emergency_qqq/tqqq: optional flash-crash thresholds.
    """
    if windows is None:
        windows = ENSEMBLE_WINDOWS

    off_asset = "SGOV"
    if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
        off_asset = "SHV" if "SHV" in prices.columns else "CASH"

    signal_raw = generate_ensemble_signal(prices, windows)
    signal_lagged = signal_raw.shift(lag).fillna(0).astype(int)

    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    qqq_ret = prices["QQQ"].pct_change().fillna(0.0)
    off_ret = prices[off_asset].pct_change().fillna(0.0)

    use_emergency = emergency_qqq is not None or emergency_tqqq is not None

    equity_values = []
    trades_count = 0
    emergency_triggers: List[Dict] = []

    portfolio_value = 1.0
    tqqq_cost_basis = 0.0
    tqqq_shares = 0.0
    tqqq_entry_price: Optional[float] = None
    yearly_gains: Dict[int, float] = {}
    total_tax_paid = 0.0
    current_weight = 0.0
    emergency_cooldown = 0

    for i, dt in enumerate(prices.index):
        px_tqqq = float(prices.loc[dt, "TQQQ"])
        base_signal = (
            int(signal_lagged.loc[dt])
            if not pd.isna(signal_lagged.loc[dt])
            else 0
        )

        # --- emergency exit check ---
        em_trigger = None
        if use_emergency and i > 0:
            if emergency_qqq is not None:
                qr = float(qqq_ret.loc[dt])
                if qr <= emergency_qqq:
                    em_trigger = f"QQQ_CRASH ({qr*100:.1f}%)"
            if (
                emergency_tqqq is not None
                and tqqq_entry_price is not None
                and current_weight > 0.5
            ):
                tqqq_dd = (px_tqqq / tqqq_entry_price) - 1.0
                if tqqq_dd <= emergency_tqqq:
                    em_trigger = f"TQQQ_STOP ({tqqq_dd*100:.1f}%)"

        # --- target weight ---
        if em_trigger:
            target_weight = OFF_TQQQ_WEIGHT
            emergency_triggers.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "trigger": em_trigger,
                    "portfolio_value": portfolio_value,
                }
            )
            emergency_cooldown = 1
        elif emergency_cooldown > 0:
            target_weight = OFF_TQQQ_WEIGHT
            emergency_cooldown -= 1
        else:
            target_weight = 1.0 if base_signal == 1 else OFF_TQQQ_WEIGHT

        # track entry price
        if target_weight > 0.5 and current_weight <= 0.5:
            tqqq_entry_price = px_tqqq
        elif target_weight <= 0.5:
            tqqq_entry_price = None

        # --- daily PnL ---
        weight_change = abs(target_weight - current_weight)
        cost = weight_change * (COST_BPS / 10000.0) if weight_change > 1e-6 else 0.0
        port_ret = (
            target_weight * float(tqqq_ret.loc[dt])
            + (1 - target_weight) * float(off_ret.loc[dt])
            - cost
        )
        portfolio_value *= 1 + port_ret

        # --- tax bookkeeping ---
        year = dt.year
        if year not in yearly_gains:
            yearly_gains[year] = 0.0

        if weight_change > 1e-6:
            trades_count += 1
            if (
                target_weight < current_weight
                and tqqq_shares > 0
                and tqqq_cost_basis > 0
            ):
                avg_cost = tqqq_cost_basis / tqqq_shares
                sold_value = abs(target_weight - current_weight) * portfolio_value
                sold_shares = sold_value / px_tqqq if px_tqqq > 0 else 0
                gain = sold_shares * (px_tqqq - avg_cost)
                yearly_gains[year] += gain
                sell_ratio = (
                    min(1.0, sold_shares / tqqq_shares) if tqqq_shares > 0 else 0
                )
                tqqq_cost_basis *= 1 - sell_ratio
                tqqq_shares -= sold_shares
            elif target_weight > current_weight:
                buy_value = (target_weight - current_weight) * portfolio_value
                buy_shares = buy_value / px_tqqq if px_tqqq > 0 else 0
                tqqq_cost_basis += buy_value
                tqqq_shares += buy_shares

        current_weight = target_weight

        # year-end tax
        is_year_end = (i == len(prices.index) - 1) or (
            prices.index[i + 1].year != year if i < len(prices.index) - 1 else True
        )
        if is_year_end and year in yearly_gains:
            taxable = max(0, yearly_gains[year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            total_tax_paid += tax

        equity_values.append(portfolio_value)

    equity = pd.Series(equity_values, index=prices.index, name="equity")
    metrics = calculate_metrics(equity, n_trades=trades_count, total_tax=total_tax_paid)

    return {
        "equity": equity,
        "metrics": metrics,
        "emergency_triggers": emergency_triggers,
        "signal": signal_lagged,
    }


# ============================================================
# BENCHMARK ENGINES (for E42)
# ============================================================
def run_buyandhold(prices: pd.DataFrame, asset: str) -> Dict:
    """Buy-and-hold. No trades → no realised gains → no tax."""
    px = prices[asset].copy()
    equity = px / px.iloc[0]
    return {"equity": equity, "metrics": calculate_metrics(equity)}


def run_sma_switch(
    prices: pd.DataFrame,
    on_asset: str,
    off_asset: str,
    sma_window: int = 200,
) -> Dict:
    """QQQ > SMA(window) → on_asset, else off_asset. 1-day lag, costs, tax."""
    qqq = prices["QQQ"]
    sma = qqq.rolling(sma_window).mean()
    signal = (qqq > sma).astype(int)
    signal_lagged = signal.shift(1).fillna(0).astype(int)

    on_ret = prices[on_asset].pct_change().fillna(0.0)
    off_ret_s = prices[off_asset].pct_change().fillna(0.0)

    equity_values = []
    portfolio_value = 1.0
    current_weight = 0.0
    yearly_gains: Dict[int, float] = {}
    total_tax_paid = 0.0
    on_cost_basis = 0.0
    on_shares = 0.0
    n_trades = 0

    for i, dt in enumerate(prices.index):
        px_on = float(prices.loc[dt, on_asset])
        target_weight = float(signal_lagged.loc[dt])

        weight_change = abs(target_weight - current_weight)
        cost = weight_change * (COST_BPS / 10000.0) if weight_change > 1e-6 else 0.0
        port_ret = (
            target_weight * float(on_ret.loc[dt])
            + (1 - target_weight) * float(off_ret_s.loc[dt])
            - cost
        )
        portfolio_value *= 1 + port_ret

        year = dt.year
        if year not in yearly_gains:
            yearly_gains[year] = 0.0

        if weight_change > 1e-6:
            n_trades += 1
            if (
                target_weight < current_weight
                and on_shares > 0
                and on_cost_basis > 0
            ):
                avg_cost = on_cost_basis / on_shares
                sold_value = abs(target_weight - current_weight) * portfolio_value
                sold_shares = sold_value / px_on if px_on > 0 else 0
                gain = sold_shares * (px_on - avg_cost)
                yearly_gains[year] += gain
                sell_ratio = (
                    min(1.0, sold_shares / on_shares) if on_shares > 0 else 0
                )
                on_cost_basis *= 1 - sell_ratio
                on_shares -= sold_shares
            elif target_weight > current_weight:
                buy_value = (target_weight - current_weight) * portfolio_value
                buy_shares = buy_value / px_on if px_on > 0 else 0
                on_cost_basis += buy_value
                on_shares += buy_shares

        current_weight = target_weight

        is_year_end = (i == len(prices.index) - 1) or (
            prices.index[i + 1].year != year if i < len(prices.index) - 1 else True
        )
        if is_year_end and year in yearly_gains:
            taxable = max(0, yearly_gains[year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            total_tax_paid += tax

        equity_values.append(portfolio_value)

    equity = pd.Series(equity_values, index=prices.index, name="equity")
    return {
        "equity": equity,
        "metrics": calculate_metrics(equity, n_trades=n_trades, total_tax=total_tax_paid),
    }


# ============================================================
# HELPER: sub-period metrics from an equity curve
# ============================================================
def subperiod_metrics(equity: pd.Series, start: str, end: str) -> Dict:
    """Extract and normalise equity slice, then compute metrics."""
    sub = equity[(equity.index >= start) & (equity.index <= end)].copy()
    if len(sub) < 2:
        return calculate_metrics(sub)
    sub = sub / sub.iloc[0]
    return calculate_metrics(sub)


# ============================================================
# E40 — Same-Day Close Execution
# ============================================================
def run_E40(prices: pd.DataFrame) -> Dict:
    print("\n[E40] SameDay_Close: lag=0 vs lag=1 ...")
    baseline = run_e03_backtest(prices, lag=1)
    sameday = run_e03_backtest(prices, lag=0)

    exp_dir = os.path.join(OUTPUT_DIR, "E40_SameDay_Close")
    os.makedirs(exp_dir, exist_ok=True)

    pd.DataFrame(
        {"E03_Baseline": baseline["equity"], "E40_SameDay": sameday["equity"]}
    ).to_csv(os.path.join(exp_dir, "equity_curves.csv"))

    delta_cagr = (sameday["metrics"]["CAGR"] - baseline["metrics"]["CAGR"]) * 100
    delta_mdd = (sameday["metrics"]["MDD"] - baseline["metrics"]["MDD"]) * 100

    comp = {
        "E03_Baseline_lag1": baseline["metrics"],
        "E40_SameDay_lag0": sameday["metrics"],
        "delta_CAGR_pct": delta_cagr,
        "delta_MDD_pct": delta_mdd,
    }
    with open(os.path.join(exp_dir, "comparison.json"), "w") as f:
        json.dump(comp, f, indent=2)

    plt.figure(figsize=(14, 6))
    plt.plot(
        baseline["equity"],
        label=f"E03 Baseline (lag=1) CAGR={baseline['metrics']['CAGR']*100:.1f}%",
        lw=1.5,
    )
    plt.plot(
        sameday["equity"],
        label=f"E40 SameDay (lag=0) CAGR={sameday['metrics']['CAGR']*100:.1f}%",
        lw=1.5,
        ls="--",
    )
    plt.yscale("log")
    plt.title("E40: Same-Day Close vs Next-Day Execution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "comparison.png"), dpi=150)
    plt.close()

    bm, sm = baseline["metrics"], sameday["metrics"]
    print(f"   Baseline: CAGR={bm['CAGR']*100:.2f}%  MDD={bm['MDD']*100:.2f}%")
    print(f"   SameDay:  CAGR={sm['CAGR']*100:.2f}%  MDD={sm['MDD']*100:.2f}%")
    print(f"   ΔCAGR: {delta_cagr:+.2f}%p   ΔMDD: {delta_mdd:+.2f}%p")

    return {"E03_Baseline": bm, "E40_SameDay": sm}


# ============================================================
# E41 — Walk-Forward Out-of-Sample
# ============================================================
def run_E41(prices: pd.DataFrame) -> Dict:
    print("\n[E41] WalkForward_OOS: IS 2010-2017 → OOS 2018-2025 ...")

    split = "2018-01-01"
    is_prices = prices[prices.index < split].copy()
    oos_prices = prices[prices.index >= split].copy()

    print(
        f"   IS:  {is_prices.index[0].date()} ~ {is_prices.index[-1].date()} "
        f"({len(is_prices)} days)"
    )
    print(
        f"   OOS: {oos_prices.index[0].date()} ~ {oos_prices.index[-1].date()} "
        f"({len(oos_prices)} days)"
    )

    # --- Grid search on IS ---
    candidates = list(range(150, 181, 5))  # 150,155,...,180
    best_cagr = -999.0
    best_combo: List[int] = []
    grid_rows: List[Dict] = []

    for combo in combinations(candidates, 3):
        r = run_e03_backtest(is_prices, lag=1, windows=list(combo))
        c = r["metrics"]["CAGR"]
        grid_rows.append(
            {"windows": str(list(combo)), "CAGR": c, "MDD": r["metrics"]["MDD"]}
        )
        if c > best_cagr:
            best_cagr = c
            best_combo = list(combo)

    print(f"   IS-Best windows: {best_combo}  CAGR={best_cagr*100:.2f}%")

    # --- Full-period runs for fair OOS comparison ---
    full_e03 = run_e03_backtest(prices, lag=1, windows=ENSEMBLE_WINDOWS)
    full_best = run_e03_backtest(prices, lag=1, windows=best_combo)

    is_e03 = subperiod_metrics(full_e03["equity"], START_DATE, "2017-12-31")
    oos_e03 = subperiod_metrics(full_e03["equity"], split, END_DATE)
    is_best = subperiod_metrics(full_best["equity"], START_DATE, "2017-12-31")
    oos_best = subperiod_metrics(full_best["equity"], split, END_DATE)

    deg_default = oos_e03["CAGR"] / is_e03["CAGR"] if is_e03["CAGR"] != 0 else 0
    deg_best = oos_best["CAGR"] / is_best["CAGR"] if is_best["CAGR"] != 0 else 0

    exp_dir = os.path.join(OUTPUT_DIR, "E41_WalkForward_OOS")
    os.makedirs(exp_dir, exist_ok=True)

    pd.DataFrame(grid_rows).to_csv(
        os.path.join(exp_dir, "is_grid_search.csv"), index=False
    )

    result = {
        "IS_best_windows": best_combo,
        "E03_default_windows": ENSEMBLE_WINDOWS,
        "IS_E03_default": is_e03,
        "OOS_E03_default": oos_e03,
        "IS_best_combo": is_best,
        "OOS_best_combo": oos_best,
        "degradation_ratio_default": deg_default,
        "degradation_ratio_best": deg_best,
    }
    with open(os.path.join(exp_dir, "comparison.json"), "w") as f:
        json.dump(result, f, indent=2)

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    eq_is_e03 = full_e03["equity"][full_e03["equity"].index < split]
    eq_is_best = full_best["equity"][full_best["equity"].index < split]
    eq_is_e03_n = eq_is_e03 / eq_is_e03.iloc[0]
    eq_is_best_n = eq_is_best / eq_is_best.iloc[0]
    axes[0].plot(eq_is_e03_n, label=f"E03 Default CAGR={is_e03['CAGR']*100:.1f}%")
    axes[0].plot(
        eq_is_best_n,
        label=f"IS-Best {best_combo} CAGR={is_best['CAGR']*100:.1f}%",
        ls="--",
    )
    axes[0].set_title("In-Sample (2010-2017)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    eq_oos_e03 = full_e03["equity"][full_e03["equity"].index >= split]
    eq_oos_best = full_best["equity"][full_best["equity"].index >= split]
    eq_oos_e03_n = eq_oos_e03 / eq_oos_e03.iloc[0]
    eq_oos_best_n = eq_oos_best / eq_oos_best.iloc[0]
    axes[1].plot(eq_oos_e03_n, label=f"E03 Default CAGR={oos_e03['CAGR']*100:.1f}%")
    axes[1].plot(
        eq_oos_best_n,
        label=f"IS-Best {best_combo} CAGR={oos_best['CAGR']*100:.1f}%",
        ls="--",
    )
    axes[1].set_title("Out-of-Sample (2018-2025)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "walkforward.png"), dpi=150)
    plt.close()

    print(f"   OOS E03 Default:  CAGR={oos_e03['CAGR']*100:.2f}%  MDD={oos_e03['MDD']*100:.2f}%")
    print(f"   OOS IS-Best:      CAGR={oos_best['CAGR']*100:.2f}%  MDD={oos_best['MDD']*100:.2f}%")
    print(f"   Degradation (default): {deg_default:.3f}")
    print(f"   Degradation (best):    {deg_best:.3f}")

    return result


# ============================================================
# E42 — Benchmark Package
# ============================================================
def run_E42(prices: pd.DataFrame) -> Dict:
    print("\n[E42] Benchmark_Package: E03 vs 4 benchmarks ...")

    results = {}

    print("   BM1: QQQ Buy & Hold ...")
    results["BM1_QQQ_BH"] = run_buyandhold(prices, "QQQ")

    print("   BM2: TQQQ Buy & Hold ...")
    results["BM2_TQQQ_BH"] = run_buyandhold(prices, "TQQQ")

    print("   BM3: QQQ + SMA200 Timing ...")
    results["BM3_QQQ_SMA200"] = run_sma_switch(prices, "QQQ", "SGOV", 200)

    print("   BM4: TQQQ + SMA200 Timing ...")
    results["BM4_TQQQ_SMA200"] = run_sma_switch(prices, "TQQQ", "SGOV", 200)

    print("   E03: Ensemble + SGOV ...")
    e03 = run_e03_backtest(prices, lag=1)
    results["E03_Ensemble_SGOV"] = {"equity": e03["equity"], "metrics": e03["metrics"]}

    exp_dir = os.path.join(OUTPUT_DIR, "E42_Benchmark_Package")
    os.makedirs(exp_dir, exist_ok=True)

    rows = [{"Strategy": n, **r["metrics"]} for n, r in results.items()]
    comp_df = pd.DataFrame(rows).sort_values("CAGR", ascending=False)
    comp_df["Rank"] = range(1, len(comp_df) + 1)
    comp_df.to_csv(os.path.join(exp_dir, "comparison.csv"), index=False)

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump({n: r["metrics"] for n, r in results.items()}, f, indent=2)

    plt.figure(figsize=(14, 7))
    for name, r in results.items():
        plt.plot(
            r["equity"],
            label=f"{name} (CAGR={r['metrics']['CAGR']*100:.1f}%)",
            lw=1.5,
        )
    plt.yscale("log")
    plt.title("E42: Benchmark Comparison")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "benchmark_chart.png"), dpi=150)
    plt.close()

    print(f"\n   {'Rank':<5} {'Strategy':<25} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Final':>10}")
    print("   " + "-" * 70)
    for _, row in comp_df.iterrows():
        print(
            f"   {int(row['Rank']):<5} {row['Strategy']:<25} "
            f"{row['CAGR']*100:>7.2f}% {row['MDD']*100:>7.2f}% "
            f"{row['Sharpe']:>8.2f} {row['Final']:>10.2f}x"
        )

    return {n: r["metrics"] for n, r in results.items()}


# ============================================================
# E43 / E44 / E45 — Emergency Exit Calibration
# ============================================================
def run_E43_44_45(prices: pd.DataFrame) -> Dict:
    print("\n[E43-E45] Emergency Exit Calibration ...")

    configs = {
        "E03_NoEmergency": dict(emergency_qqq=None, emergency_tqqq=None),
        "E43_Emg_5pct":    dict(emergency_qqq=-0.05, emergency_tqqq=-0.15),
        "E44_Emg_7pct":    dict(emergency_qqq=-0.07, emergency_tqqq=-0.20),
        "E45_Emg_10pct":   dict(emergency_qqq=-0.10, emergency_tqqq=-0.30),
    }

    results = {}
    for name, cfg in configs.items():
        print(f"   {name} ...")
        r = run_e03_backtest(prices, lag=1, **cfg)
        results[name] = r
        m = r["metrics"]
        print(
            f"      CAGR={m['CAGR']*100:.2f}%  MDD={m['MDD']*100:.2f}%  "
            f"Triggers={len(r['emergency_triggers'])}"
        )

    # per-experiment artefacts
    for eid, suffix in [("E43", "5pct"), ("E44", "7pct"), ("E45", "10pct")]:
        ed = os.path.join(OUTPUT_DIR, f"{eid}_Emergency_{suffix}")
        os.makedirs(ed, exist_ok=True)
        key = f"{eid}_Emg_{suffix}"
        r = results[key]
        r["equity"].to_frame().to_csv(os.path.join(ed, "equity_curve.csv"))
        with open(os.path.join(ed, "metrics.json"), "w") as f:
            json.dump(r["metrics"], f, indent=2)
        if r["emergency_triggers"]:
            pd.DataFrame(r["emergency_triggers"]).to_csv(
                os.path.join(ed, "emergency_events.csv"), index=False
            )

    # combined comparison
    comp_dir = os.path.join(OUTPUT_DIR, "E43_44_45_Emergency_Comparison")
    os.makedirs(comp_dir, exist_ok=True)

    rows = []
    for name, r in results.items():
        row = r["metrics"].copy()
        row["Name"] = name
        row["EmergencyTriggers"] = len(r["emergency_triggers"])
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(comp_dir, "comparison.csv"), index=False)

    plt.figure(figsize=(14, 7))
    for name, r in results.items():
        plt.plot(
            r["equity"],
            label=f"{name} CAGR={r['metrics']['CAGR']*100:.1f}%",
            lw=1.5,
        )
    plt.yscale("log")
    plt.title("Emergency Exit Calibration: E03 vs E43 / E44 / E45")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "comparison.png"), dpi=150)
    plt.close()

    print(f"\n   {'Name':<20} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Triggers':>10}")
    print("   " + "-" * 60)
    for _, row in comp_df.iterrows():
        print(
            f"   {row['Name']:<20} {row['CAGR']*100:>7.2f}% "
            f"{row['MDD']*100:>7.2f}% {row['Sharpe']:>8.2f} "
            f"{int(row['EmergencyTriggers']):>10}"
        )

    return {n: r["metrics"] for n, r in results.items()}


# ============================================================
# E46 — Volatility Drag Analysis
# ============================================================
def run_E46(prices: pd.DataFrame) -> Dict:
    print("\n[E46] VolDrag_Analysis: quantifying TQQQ 3× volatility drag ...")

    signal_raw = generate_ensemble_signal(prices)
    signal_lagged = signal_raw.shift(1).fillna(0).astype(int)

    qqq_ret = prices["QQQ"].pct_change().fillna(0.0)
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)

    # --- daily drag ---
    daily_theoretical_3x = qqq_ret * 3
    daily_drag = tqqq_ret - daily_theoretical_3x

    # --- identify ON/OFF blocks ---
    block_id = (signal_lagged != signal_lagged.shift(1)).cumsum()
    periods: List[Dict] = []

    for bid in sorted(block_id.unique()):
        mask = block_id == bid
        p = prices[mask]
        if len(p) < 2:
            continue

        state = "ON" if int(signal_lagged[mask].iloc[0]) == 1 else "OFF"

        qqq_s = float(p["QQQ"].iloc[0])
        qqq_e = float(p["QQQ"].iloc[-1])
        tqqq_s = float(p["TQQQ"].iloc[0])
        tqqq_e = float(p["TQQQ"].iloc[-1])

        qqq_period_ret = qqq_e / qqq_s - 1.0
        tqqq_period_ret = tqqq_e / tqqq_s - 1.0

        # theoretical 3× via daily compounding
        qqq_daily = p["QQQ"].pct_change().fillna(0.0)
        theo_cum = float((1 + qqq_daily * 3).cumprod().iloc[-1]) - 1.0

        drag = tqqq_period_ret - theo_cum
        n_days = len(p)
        drag_ann = drag * (TRADING_DAYS / n_days) if n_days > 0 else 0.0
        qqq_vol = float(qqq_daily.std() * np.sqrt(TRADING_DAYS))

        periods.append(
            {
                "start": p.index[0].strftime("%Y-%m-%d"),
                "end": p.index[-1].strftime("%Y-%m-%d"),
                "days": n_days,
                "state": state,
                "qqq_return": qqq_period_ret,
                "tqqq_return": tqqq_period_ret,
                "theo_3x_daily": theo_cum,
                "drag": drag,
                "drag_annualized": drag_ann,
                "qqq_annualized_vol": qqq_vol,
            }
        )

    periods_df = pd.DataFrame(periods)
    on_df = periods_df[periods_df["state"] == "ON"]
    off_df = periods_df[periods_df["state"] == "OFF"]

    # --- full-period summary ---
    full_qqq_ret = float(prices["QQQ"].iloc[-1] / prices["QQQ"].iloc[0]) - 1
    full_tqqq_ret = float(prices["TQQQ"].iloc[-1] / prices["TQQQ"].iloc[0]) - 1
    theo_full = float((1 + qqq_ret * 3).cumprod().iloc[-1]) - 1
    full_drag = full_tqqq_ret - theo_full
    n_years = len(prices) / TRADING_DAYS

    def _block_stats(df: pd.DataFrame) -> Dict:
        if len(df) == 0:
            return {"count": 0}
        return {
            "count": len(df),
            "avg_days": float(df["days"].mean()),
            "avg_drag": float(df["drag"].mean()),
            "median_drag": float(df["drag"].median()),
            "total_drag": float(df["drag"].sum()),
            "avg_qqq_vol": float(df["qqq_annualized_vol"].mean()),
        }

    stats: Dict = {
        "full_period": {
            "qqq_total_return": full_qqq_ret,
            "tqqq_total_return": full_tqqq_ret,
            "theo_3x_daily": theo_full,
            "total_drag": full_drag,
            "drag_annualized": full_drag / n_years if n_years > 0 else 0,
        },
        "on_periods": _block_stats(on_df),
        "off_periods": _block_stats(off_df),
    }

    # --- sideways analysis (|QQQ 50d return| < 5%) ---
    qqq_50d = prices["QQQ"].pct_change(50)
    sw_mask = qqq_50d.abs() < 0.05
    sw_days = int(sw_mask.sum())
    if sw_days > 10:
        sw_tqqq = tqqq_ret[sw_mask]
        sw_qqq = qqq_ret[sw_mask]
        drag_per_day = float(sw_tqqq.mean() - 3 * sw_qqq.mean())
        stats["sideways"] = {
            "days": sw_days,
            "pct_of_total": sw_days / len(prices),
            "drag_per_day": drag_per_day,
            "drag_annualized": drag_per_day * TRADING_DAYS,
        }

    # --- save artefacts ---
    exp_dir = os.path.join(OUTPUT_DIR, "E46_VolDrag_Analysis")
    os.makedirs(exp_dir, exist_ok=True)

    periods_df.to_csv(os.path.join(exp_dir, "period_details.csv"), index=False)
    with open(os.path.join(exp_dir, "drag_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # --- plots ---
    if len(periods_df) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        colors = ["green" if s == "ON" else "gray" for s in periods_df["state"]]
        axes[0].bar(range(len(periods_df)), periods_df["drag"] * 100, color=colors, alpha=0.7)
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].set_title("E46: Volatility Drag per Period (green=ON, gray=OFF)")
        axes[0].set_ylabel("Drag (%)")
        axes[0].set_xlabel("Period #")

        axes[1].scatter(
            periods_df["qqq_annualized_vol"] * 100,
            periods_df["drag"] * 100,
            c=colors,
            alpha=0.7,
            s=periods_df["days"].clip(upper=200) * 2,
        )
        axes[1].set_title("Drag vs QQQ Annualized Volatility (size ∝ period length)")
        axes[1].set_xlabel("QQQ Annualized Vol (%)")
        axes[1].set_ylabel("Drag (%)")
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "drag_analysis.png"), dpi=150)
        plt.close()

    # --- daily drag histogram ---
    plt.figure(figsize=(10, 5))
    plt.hist(daily_drag.dropna() * 100, bins=100, alpha=0.7, edgecolor="black", lw=0.3)
    plt.axvline(0, color="red", lw=1)
    plt.title(
        f"TQQQ Daily Tracking Error (mean={daily_drag.mean()*100:.4f}%, "
        f"std={daily_drag.std()*100:.4f}%)"
    )
    plt.xlabel("Daily Drag (%)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "daily_drag_histogram.png"), dpi=150)
    plt.close()

    print(f"   Full-period TQQQ drag: {full_drag*100:.2f}%")
    print(f"   ON  periods: {stats['on_periods'].get('count',0)}, avg drag: {stats['on_periods'].get('avg_drag',0)*100:.3f}%")
    print(f"   OFF periods: {stats['off_periods'].get('count',0)}, avg drag: {stats['off_periods'].get('avg_drag',0)*100:.3f}%")
    if "sideways" in stats:
        sw = stats["sideways"]
        print(f"   Sideways days: {sw['days']} ({sw['pct_of_total']*100:.1f}%), ann drag: {sw['drag_annualized']*100:.2f}%")

    return stats


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("       E03 Weakness Suite: 7 Experiments (E40-E46)")
    print("       Addressing identified weaknesses in E03 SSOT")
    print("=" * 80)

    prices = load_data()
    all_results: Dict = {}

    all_results["E40"] = run_E40(prices)
    all_results["E41"] = run_E41(prices)
    all_results["E42"] = run_E42(prices)
    all_results["E43_44_45"] = run_E43_44_45(prices)
    all_results["E46"] = run_E46(prices)

    # ---- master JSON ----
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- final summary ----
    print("\n" + "=" * 80)
    print("                    WEAKNESS SUITE — FINAL SUMMARY")
    print("=" * 80)

    e40 = all_results["E40"]
    print(
        f"\n📊 E40 (SameDay Close):"
        f"\n   lag=0 CAGR: {e40['E40_SameDay']['CAGR']*100:.2f}%"
        f"  vs  lag=1: {e40['E03_Baseline']['CAGR']*100:.2f}%"
    )

    e41 = all_results["E41"]
    print(
        f"\n📊 E41 (Walk-Forward OOS):"
        f"\n   IS→OOS degradation (default [160,165,170]): {e41['degradation_ratio_default']:.3f}"
        f"\n   IS→OOS degradation (IS-best {e41['IS_best_windows']}):  {e41['degradation_ratio_best']:.3f}"
    )

    print("\n📊 E42 (Benchmark Package):")
    for name, m in all_results["E42"].items():
        print(f"   {name:<25} CAGR={m['CAGR']*100:>7.2f}%  MDD={m['MDD']*100:>7.2f}%")

    print("\n📊 E43-E45 (Emergency Calibration):")
    for name, m in all_results["E43_44_45"].items():
        print(f"   {name:<20} CAGR={m['CAGR']*100:>7.2f}%  MDD={m['MDD']*100:>7.2f}%")

    e46 = all_results["E46"]
    print(
        f"\n📊 E46 (Volatility Drag):"
        f"\n   Full-period drag: {e46['full_period']['total_drag']*100:.2f}%"
        f"\n   ON avg drag:  {e46['on_periods'].get('avg_drag',0)*100:.3f}%"
        f"\n   OFF avg drag: {e46['off_periods'].get('avg_drag',0)*100:.3f}%"
    )

    print(f"\n📁 All artefacts saved to: {OUTPUT_DIR}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
