# -*- coding: utf-8 -*-
"""
Paper-Based E03 Improvement Suite
===================================

Based on recent academic papers (2023-2026):

  F1  Signal Stability Filter     (Declerck & Vy, 2024 SSRN:5032806)
      → When E03 signal flips frequently → reduce position
  F2  Autocorrelation Filter      (Hsieh et al., 2025 arXiv:2504.20116)
      → When QQQ return autocorrelation is negative → reduce position
  F3  Combined (F1 + F2)
      → Both filters active simultaneously

Engine: Reuses flexible backtest from MDD reduction suite.
        Cost 10bps, Korean 22% tax, 2010-2025.

Author: QuantNeural Paper Improvement Suite v2026.2
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS (identical to E03 / MDD reduction suite)
# ============================================================
COST_BPS = 10
TAX_RATE = 0.22
SHORT_MA = 3
OFF_TQQQ_WEIGHT = 0.10
TRADING_DAYS = 252
ENSEMBLE_WINDOWS = [160, 165, 170]

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/experiments/paper_improvements"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> pd.DataFrame:
    import yfinance as yf
    print("📥 Downloading data...")
    tickers = ["QQQ", "TQQQ", "SGOV", "SHV"]
    raw = yf.download(
        tickers=tickers, start="2009-01-01", end=END_DATE,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    prices = pd.DataFrame()
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if (t, "Close") in raw.columns:
                prices[t] = raw[(t, "Close")]
    prices["CASH"] = 100.0
    if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
        prices["SGOV"] = prices.get("SHV", pd.Series(100.0, index=prices.index))
    for c in ["SGOV", "SHV"]:
        if c in prices.columns:
            prices[c] = prices[c].ffill()
    prices = prices.dropna(subset=["QQQ", "TQQQ"])
    prices = prices[(prices.index >= START_DATE) & (prices.index <= END_DATE)]
    print(f"   Period: {prices.index[0].date()} ~ {prices.index[-1].date()} ({len(prices)} days)")
    return prices


# ============================================================
# SIGNAL
# ============================================================
def generate_ensemble_signal(prices: pd.DataFrame, windows=None) -> pd.Series:
    if windows is None:
        windows = ENSEMBLE_WINDOWS
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    votes = pd.DataFrame(index=prices.index)
    for lw in windows:
        votes[f"w{lw}"] = (ma_short > qqq.rolling(lw).mean()).astype(int)
    threshold = len(windows) // 2 + 1
    return (votes.sum(axis=1) >= threshold).astype(int)


# ============================================================
# METRICS
# ============================================================
def calc_metrics(equity: pd.Series, n_trades: int = 0, total_tax: float = 0.0) -> Dict:
    n_years = len(equity) / TRADING_DAYS
    final = float(equity.iloc[-1])
    cagr = (final ** (1.0 / n_years) - 1.0) if n_years > 0 and final > 0 else 0.0
    rets = equity.pct_change().fillna(0.0)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    daily_std = rets.std(ddof=0)
    sharpe = (rets.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0
    ds = rets[rets < 0]
    ds_std = ds.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(ds) > 0 else 0.0
    sortino = cagr / ds_std if ds_std > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return {"Final": final, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
            "Sortino": sortino, "Calmar": calmar, "Trades": n_trades,
            "TotalTax": total_tax}


# ============================================================
# GENERAL BACKTEST ENGINE
# ============================================================
def run_flexible_backtest(
    prices: pd.DataFrame,
    target_weight_series: pd.Series,
    on_asset: str = "TQQQ",
    off_asset: str = "SGOV",
    no_trade_band: float = 0.05,
) -> Dict:
    on_ret = prices[on_asset].pct_change().fillna(0.0)
    oa = off_asset if off_asset in prices.columns else "CASH"
    off_ret = prices[oa].pct_change().fillna(0.0)

    portfolio_value = 1.0
    actual_weight = 0.0
    on_cost_basis = 0.0
    on_shares = 0.0
    yearly_gains: Dict[int, float] = {}
    total_tax = 0.0
    n_trades = 0
    equity_vals = []

    for i, dt in enumerate(prices.index):
        px_on = float(prices.loc[dt, on_asset])
        target = float(target_weight_series.loc[dt]) if dt in target_weight_series.index else actual_weight

        diff = abs(target - actual_weight)
        if diff > no_trade_band:
            cost = diff * (COST_BPS / 10000.0)
            year = dt.year
            if year not in yearly_gains:
                yearly_gains[year] = 0.0

            if target < actual_weight and on_shares > 0 and on_cost_basis > 0:
                avg_cost = on_cost_basis / on_shares
                sold_value = (actual_weight - target) * portfolio_value
                sold_shares = sold_value / px_on if px_on > 0 else 0
                gain = sold_shares * (px_on - avg_cost)
                yearly_gains[year] += gain
                sr = min(1.0, sold_shares / on_shares) if on_shares > 0 else 0
                on_cost_basis *= (1 - sr)
                on_shares -= sold_shares
            elif target > actual_weight:
                buy_value = (target - actual_weight) * portfolio_value
                buy_shares = buy_value / px_on if px_on > 0 else 0
                on_cost_basis += buy_value
                on_shares += buy_shares

            actual_weight = target
            n_trades += 1
        else:
            cost = 0.0

        r_on = float(on_ret.loc[dt])
        r_off = float(off_ret.loc[dt])
        port_ret = actual_weight * r_on + (1 - actual_weight) * r_off - cost
        portfolio_value *= (1 + port_ret)

        if portfolio_value > 0 and actual_weight > 0:
            on_portion = actual_weight * (1 + r_on)
            off_portion = (1 - actual_weight) * (1 + r_off)
            total = on_portion + off_portion
            actual_weight = on_portion / total if total > 0 else actual_weight

        year = dt.year
        if year not in yearly_gains:
            yearly_gains[year] = 0.0
        is_year_end = (i == len(prices.index) - 1) or \
                      (prices.index[i+1].year != year if i < len(prices.index) - 1 else True)
        if is_year_end and year in yearly_gains:
            taxable = max(0, yearly_gains[year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            total_tax += tax

        equity_vals.append(portfolio_value)

    equity = pd.Series(equity_vals, index=prices.index, name="equity")
    return {"equity": equity, "metrics": calc_metrics(equity, n_trades, total_tax)}


# ============================================================
# HELPER: rolling autocorrelation
# ============================================================
def rolling_autocorrelation(returns: pd.Series, lookback: int, lag: int) -> pd.Series:
    """Calculate rolling autocorrelation of returns at given lag."""
    result = pd.Series(index=returns.index, dtype=float)
    vals = returns.values
    for i in range(lookback + lag, len(vals)):
        window = vals[i - lookback:i]
        x = window[:-lag]
        y = window[lag:]
        if len(x) < 10:
            result.iloc[i] = 0.0
            continue
        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=0), np.std(y, ddof=0)
        if sx < 1e-12 or sy < 1e-12:
            result.iloc[i] = 0.0
        else:
            result.iloc[i] = np.mean((x - mx) * (y - my)) / (sx * sy)
    return result.fillna(0.0)


# ============================================================
# HELPER: signal flip count
# ============================================================
def rolling_signal_flips(signal: pd.Series, window: int) -> pd.Series:
    """Count how many times the signal changes over a rolling window."""
    changes = signal.diff().abs()  # 1 when signal flips, 0 otherwise
    return changes.rolling(window, min_periods=1).sum()


# ============================================================
# E03 BASELINE
# ============================================================
def run_e03_baseline(prices: pd.DataFrame) -> Dict:
    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    w = signal_lag * 1.0 + (1 - signal_lag) * OFF_TQQQ_WEIGHT
    return run_flexible_backtest(prices, w, no_trade_band=0.0)


# ============================================================
# EXPERIMENT F1: Signal Stability Filter
# (Declerck & Vy, 2024)
# ============================================================
def run_exp_F1(prices: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("[F1] Signal Stability Filter (Declerck & Vy, 2024)")
    print("    When signal flips frequently → reduce TQQQ position")
    print("=" * 70)

    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    w_base = signal_lag * 1.0 + (1 - signal_lag) * OFF_TQQQ_WEIGHT

    # Grid: flip_window × flip_threshold × reduced_weight
    flip_windows = [20, 30, 40]
    flip_thresholds = [3, 5, 7, 10]
    reduced_weights = [0.30, 0.50, 0.70]

    results = {}

    for fw in flip_windows:
        flips = rolling_signal_flips(signal_lag, fw).shift(1).fillna(0)
        for ft in flip_thresholds:
            for rw in reduced_weights:
                choppy = flips >= ft
                w = w_base.copy()
                # In choppy periods: cap ON-weight at reduced_weight
                w[choppy & (signal_lag == 1)] = rw
                # OFF periods stay at 10%

                r = run_flexible_backtest(prices, w, no_trade_band=0.03)
                m = r["metrics"]
                key = f"FW{fw}_FT{ft}_RW{int(rw*100)}"
                results[key] = {**m, "FlipWindow": fw, "FlipThreshold": ft,
                                "ReducedWeight": rw, "equity": r["equity"]}
                print(f"   {key}: CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% "
                      f"Calmar={m['Calmar']:.2f} Trades={m['Trades']}")

    # Save grid
    ed = os.path.join(OUTPUT_DIR, "F1_SignalStability")
    os.makedirs(ed, exist_ok=True)
    rows = [{k: v for k, v in val.items() if k != "equity"}
            for key, val in results.items()]
    for i, key in enumerate(results.keys()):
        rows[i]["Config"] = key
    pd.DataFrame(rows).to_csv(os.path.join(ed, "grid.csv"), index=False)

    best_k = max(results, key=lambda k: results[k]["Calmar"])
    bm = results[best_k]
    print(f"\n   ★ Best Calmar: {best_k} → CAGR={bm['CAGR']*100:.2f}% "
          f"MDD={bm['MDD']*100:.2f}% Calmar={bm['Calmar']:.2f}")

    return {"grid": {k: {kk: vv for kk, vv in v.items() if kk != "equity"}
                     for k, v in results.items()},
            "best": best_k, "best_metrics": {k: v for k, v in bm.items() if k != "equity"},
            "best_equity": results[best_k]["equity"]}


# ============================================================
# EXPERIMENT F2: Autocorrelation Filter
# (Hsieh, Chang, Chen, 2025)
# ============================================================
def run_exp_F2(prices: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("[F2] Autocorrelation Filter (Hsieh et al., 2025)")
    print("    When QQQ return autocorrelation < threshold → reduce position")
    print("=" * 70)

    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    w_base = signal_lag * 1.0 + (1 - signal_lag) * OFF_TQQQ_WEIGHT

    qqq_ret = prices["QQQ"].pct_change().fillna(0.0)

    # Grid: lookback × lag × threshold × reduced_weight
    lookbacks = [20, 40, 60]
    lags = [1, 3, 5]
    thresholds = [0.0, -0.05, -0.10]
    reduced_weights = [0.30, 0.50]

    results = {}

    # Pre-compute autocorrelations
    autocorr_cache = {}
    for lb in lookbacks:
        for lag in lags:
            key_ac = (lb, lag)
            print(f"   Computing autocorrelation LB={lb}, Lag={lag}...")
            autocorr_cache[key_ac] = rolling_autocorrelation(qqq_ret, lb, lag)

    for lb in lookbacks:
        for lag in lags:
            ac = autocorr_cache[(lb, lag)].shift(1).fillna(0)
            for thr in thresholds:
                for rw in reduced_weights:
                    mean_revert = ac < thr
                    w = w_base.copy()
                    w[mean_revert & (signal_lag == 1)] = rw

                    r = run_flexible_backtest(prices, w, no_trade_band=0.03)
                    m = r["metrics"]
                    key = f"LB{lb}_L{lag}_T{int(thr*100)}_RW{int(rw*100)}"
                    results[key] = {**m, "Lookback": lb, "Lag": lag,
                                    "Threshold": thr, "ReducedWeight": rw,
                                    "equity": r["equity"]}
                    print(f"   {key}: CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% "
                          f"Calmar={m['Calmar']:.2f} Trades={m['Trades']}")

    ed = os.path.join(OUTPUT_DIR, "F2_Autocorrelation")
    os.makedirs(ed, exist_ok=True)
    rows = [{k: v for k, v in val.items() if k != "equity"}
            for key, val in results.items()]
    for i, key in enumerate(results.keys()):
        rows[i]["Config"] = key
    pd.DataFrame(rows).to_csv(os.path.join(ed, "grid.csv"), index=False)

    best_k = max(results, key=lambda k: results[k]["Calmar"])
    bm = results[best_k]
    print(f"\n   ★ Best Calmar: {best_k} → CAGR={bm['CAGR']*100:.2f}% "
          f"MDD={bm['MDD']*100:.2f}% Calmar={bm['Calmar']:.2f}")

    return {"grid": {k: {kk: vv for kk, vv in v.items() if kk != "equity"}
                     for k, v in results.items()},
            "best": best_k, "best_metrics": {k: v for k, v in bm.items() if k != "equity"},
            "best_equity": results[best_k]["equity"]}


# ============================================================
# EXPERIMENT F3: Combined (F1 + F2)
# ============================================================
def run_exp_F3(prices: pd.DataFrame,
               best_f1_params: Dict, best_f2_params: Dict) -> Dict:
    print("\n" + "=" * 70)
    print("[F3] Combined: Signal Stability + Autocorrelation")
    print(f"    F1 best: FW={best_f1_params['FlipWindow']}, "
          f"FT={best_f1_params['FlipThreshold']}, RW={best_f1_params['ReducedWeight']}")
    print(f"    F2 best: LB={best_f2_params['Lookback']}, "
          f"L={best_f2_params['Lag']}, T={best_f2_params['Threshold']}, "
          f"RW={best_f2_params['ReducedWeight']}")
    print("=" * 70)

    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    w_base = signal_lag * 1.0 + (1 - signal_lag) * OFF_TQQQ_WEIGHT

    # F1 filter
    fw = best_f1_params["FlipWindow"]
    ft = best_f1_params["FlipThreshold"]
    rw1 = best_f1_params["ReducedWeight"]
    flips = rolling_signal_flips(signal_lag, fw).shift(1).fillna(0)
    choppy = flips >= ft

    # F2 filter
    lb = best_f2_params["Lookback"]
    lag = best_f2_params["Lag"]
    thr = best_f2_params["Threshold"]
    rw2 = best_f2_params["ReducedWeight"]
    qqq_ret = prices["QQQ"].pct_change().fillna(0.0)
    ac = rolling_autocorrelation(qqq_ret, lb, lag).shift(1).fillna(0)
    mean_revert = ac < thr

    results = {}

    # Variant 1: OR (either filter triggers → reduce)
    w_or = w_base.copy()
    either = choppy | mean_revert
    rw_avg = (rw1 + rw2) / 2
    w_or[either & (signal_lag == 1)] = rw_avg
    r = run_flexible_backtest(prices, w_or, no_trade_band=0.03)
    m = r["metrics"]
    results["F3_OR"] = {**m, "equity": r["equity"]}
    print(f"   F3_OR (either → {rw_avg:.0%}): CAGR={m['CAGR']*100:.2f}% "
          f"MDD={m['MDD']*100:.2f}% Calmar={m['Calmar']:.2f}")

    # Variant 2: AND (both must trigger → reduce more aggressively)
    w_and = w_base.copy()
    both = choppy & mean_revert
    w_and[both & (signal_lag == 1)] = min(rw1, rw2)
    # Single filter → moderate reduction
    w_and[choppy & ~mean_revert & (signal_lag == 1)] = rw1
    w_and[~choppy & mean_revert & (signal_lag == 1)] = rw2
    r = run_flexible_backtest(prices, w_and, no_trade_band=0.03)
    m = r["metrics"]
    results["F3_LAYERED"] = {**m, "equity": r["equity"]}
    print(f"   F3_LAYERED (each reduces independently): "
          f"CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% Calmar={m['Calmar']:.2f}")

    # Variant 3: Multiplicative (both reduce → multiply weights)
    w_mult = w_base.copy()
    scale = pd.Series(1.0, index=prices.index)
    scale[choppy] *= rw1
    scale[mean_revert] *= rw2
    w_mult_on = (signal_lag == 1)
    w_mult[w_mult_on] = scale[w_mult_on]
    r = run_flexible_backtest(prices, w_mult, no_trade_band=0.03)
    m = r["metrics"]
    results["F3_MULT"] = {**m, "equity": r["equity"]}
    print(f"   F3_MULT (multiplicative scaling): "
          f"CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% Calmar={m['Calmar']:.2f}")

    # Variant 4: Graduated - use number of active filters to set weight
    w_grad = w_base.copy()
    n_active = choppy.astype(int) + mean_revert.astype(int)
    # 0 filters → 100%, 1 filter → 60%, 2 filters → 30%
    graduated_w = pd.Series(1.0, index=prices.index)
    graduated_w[n_active == 1] = 0.60
    graduated_w[n_active == 2] = 0.30
    w_grad[signal_lag == 1] = graduated_w[signal_lag == 1]
    r = run_flexible_backtest(prices, w_grad, no_trade_band=0.03)
    m = r["metrics"]
    results["F3_GRADUATED"] = {**m, "equity": r["equity"]}
    print(f"   F3_GRADUATED (0→100%, 1→60%, 2→30%): "
          f"CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% Calmar={m['Calmar']:.2f}")

    ed = os.path.join(OUTPUT_DIR, "F3_Combined")
    os.makedirs(ed, exist_ok=True)
    rows = [{"Variant": k, **{kk: vv for kk, vv in v.items() if kk != "equity"}}
            for k, v in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ed, "variants.csv"), index=False)

    best_k = max(results, key=lambda k: results[k]["Calmar"])
    bm = results[best_k]
    print(f"\n   ★ Best: {best_k} → CAGR={bm['CAGR']*100:.2f}% "
          f"MDD={bm['MDD']*100:.2f}% Calmar={bm['Calmar']:.2f}")

    return {"variants": {k: {kk: vv for kk, vv in v.items() if kk != "equity"}
                         for k, v in results.items()},
            "best": best_k,
            "best_metrics": {k: v for k, v in bm.items() if k != "equity"},
            "best_equity": results[best_k]["equity"]}


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  Paper-Based E03 Improvement Suite")
    print("  F1: Signal Stability Filter (Declerck & Vy, 2024)")
    print("  F2: Autocorrelation Filter (Hsieh et al., 2025)")
    print("  F3: Combined (F1 + F2)")
    print("=" * 70)

    prices = load_data()

    # --- Baseline ---
    print("\n[Baseline] E03 Ensemble + SGOV ...")
    baseline = run_e03_baseline(prices)
    bm = baseline["metrics"]
    eq_baseline = baseline["equity"]
    print(f"   CAGR={bm['CAGR']*100:.2f}% MDD={bm['MDD']*100:.2f}% "
          f"Sharpe={bm['Sharpe']:.2f} Calmar={bm['Calmar']:.2f}")

    # --- Experiments ---
    res_F1 = run_exp_F1(prices)
    res_F2 = run_exp_F2(prices)

    # Extract best params for F3
    f1_best_params = res_F1["grid"][res_F1["best"]]
    f2_best_params = res_F2["grid"][res_F2["best"]]
    res_F3 = run_exp_F3(prices, f1_best_params, f2_best_params)

    # ============================================================
    # MASTER COMPARISON
    # ============================================================
    print("\n" + "=" * 70)
    print("                    MASTER COMPARISON")
    print("=" * 70)

    comparison = {
        "E03_Baseline": bm,
        f"F1_Best_{res_F1['best']}": res_F1["best_metrics"],
        f"F2_Best_{res_F2['best']}": res_F2["best_metrics"],
        f"F3_Best_{res_F3['best']}": res_F3["best_metrics"],
    }

    print(f"\n   {'Strategy':<40} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Trades':>8}")
    print("   " + "-" * 85)
    for name, m in comparison.items():
        print(f"   {name:<40} {m['CAGR']*100:>7.2f}% {m['MDD']*100:>7.2f}% "
              f"{m['Sharpe']:>8.2f} {m['Calmar']:>8.2f} {m['Trades']:>8}")

    # Deltas
    print(f"\n   {'Strategy':<40} {'ΔCAGR':>8} {'ΔMDD':>8} {'ΔCalmar':>8}")
    print("   " + "-" * 65)
    for name, m in comparison.items():
        if name == "E03_Baseline":
            continue
        dc = (m["CAGR"] - bm["CAGR"]) * 100
        dm = (m["MDD"] - bm["MDD"]) * 100
        dcal = m["Calmar"] - bm["Calmar"]
        print(f"   {name:<40} {dc:>+7.2f}% {dm:>+7.2f}% {dcal:>+8.2f}")

    # Save comparison
    with open(os.path.join(OUTPUT_DIR, "master_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    rows = [{"Strategy": k, **v} for k, v in comparison.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "master_comparison.csv"), index=False)

    # --- Equity curves ---
    print("\n   Generating equity curve comparison plot ...")
    plt.figure(figsize=(16, 8))
    plt.plot(eq_baseline, label=f"E03 Baseline (CAGR={bm['CAGR']*100:.1f}%, MDD={bm['MDD']*100:.1f}%)", lw=2, color="black")

    if res_F1.get("best_equity") is not None:
        m1 = res_F1["best_metrics"]
        plt.plot(res_F1["best_equity"],
                 label=f"F1 Best (CAGR={m1['CAGR']*100:.1f}%, MDD={m1['MDD']*100:.1f}%)",
                 lw=1.5, ls="--", color="blue")

    if res_F2.get("best_equity") is not None:
        m2 = res_F2["best_metrics"]
        plt.plot(res_F2["best_equity"],
                 label=f"F2 Best (CAGR={m2['CAGR']*100:.1f}%, MDD={m2['MDD']*100:.1f}%)",
                 lw=1.5, ls="-.", color="red")

    if res_F3.get("best_equity") is not None:
        m3 = res_F3["best_metrics"]
        plt.plot(res_F3["best_equity"],
                 label=f"F3 Best (CAGR={m3['CAGR']*100:.1f}%, MDD={m3['MDD']*100:.1f}%)",
                 lw=1.5, ls=":", color="green")

    plt.yscale("log")
    plt.title("Paper-Based Improvement Suite: Equity Curves (Log Scale)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "equity_comparison.png"), dpi=150)
    plt.close()

    # --- Filter activation stats ---
    print("\n   Computing filter activation statistics ...")
    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)

    # F1 activation
    fw = f1_best_params["FlipWindow"]
    ft = f1_best_params["FlipThreshold"]
    flips = rolling_signal_flips(signal_lag, fw).shift(1).fillna(0)
    f1_active = (flips >= ft) & (signal_lag == 1)
    on_days = (signal_lag == 1).sum()
    print(f"   F1 filter active: {f1_active.sum()} days out of {on_days} ON days "
          f"({f1_active.sum()/on_days*100:.1f}%)")

    # F2 activation
    lb = f2_best_params["Lookback"]
    lag = f2_best_params["Lag"]
    thr = f2_best_params["Threshold"]
    qqq_ret = prices["QQQ"].pct_change().fillna(0.0)
    ac = rolling_autocorrelation(qqq_ret, lb, lag).shift(1).fillna(0)
    f2_active = (ac < thr) & (signal_lag == 1)
    print(f"   F2 filter active: {f2_active.sum()} days out of {on_days} ON days "
          f"({f2_active.sum()/on_days*100:.1f}%)")

    both_active = f1_active & f2_active
    print(f"   Both active:      {both_active.sum()} days "
          f"({both_active.sum()/on_days*100:.1f}%)")

    print(f"\n📁 All results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    main()
