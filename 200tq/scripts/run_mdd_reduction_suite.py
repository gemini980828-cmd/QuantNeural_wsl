# -*- coding: utf-8 -*-
"""
MDD Reduction Suite: 5 Experiments (A-E)
=========================================

Goal: Reduce E03's MDD (-54%) while maintaining CAGR (~33%)
via volatility-targeting, fast exits, and multi-asset rotation.

  A  Vol-Target Only       Pure vol-target (no E03 signal)
  B  Vol-Target + E03      E03 ON/OFF gate + vol-target in ON
  C  B + Fast Exit         B + QQQ<MA(20) cap or DD-based cap
  D  Discrete Regime       Vol buckets → discrete weights
  E  Multi-Asset Rotation  {TQQQ, UPRO, TLT, SGOV} momentum + MA200

Engine: Iterative with weight drift tracking, no-trade bands,
        cost 10bps, Korean 22% tax on realized gains.

Author: QuantNeural MDD Reduction Suite v2026.2
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================
COST_BPS = 10
TAX_RATE = 0.22
SHORT_MA = 3
OFF_TQQQ_WEIGHT = 0.10
TRADING_DAYS = 252
ENSEMBLE_WINDOWS = [160, 165, 170]

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/experiments/mdd_reduction"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> pd.DataFrame:
    import yfinance as yf
    print("📥 Downloading data...")
    tickers = ["QQQ", "TQQQ", "SGOV", "SHV", "UPRO", "TLT"]
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
    for c in ["SGOV", "SHV", "TLT", "UPRO"]:
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
# GENERAL BACKTEST ENGINE — supports continuous weights
# ============================================================
def run_flexible_backtest(
    prices: pd.DataFrame,
    target_weight_series: pd.Series,
    on_asset: str = "TQQQ",
    off_asset: str = "SGOV",
    no_trade_band: float = 0.05,
) -> Dict:
    """
    Backtest with arbitrary daily target weights for on_asset.
    Remainder goes to off_asset.
    Includes: weight drift tracking, no-trade bands, cost, Korean tax.
    """
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

        # --- decide whether to trade ---
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

        # --- daily return using ACTUAL weight ---
        r_on = float(on_ret.loc[dt])
        r_off = float(off_ret.loc[dt])
        port_ret = actual_weight * r_on + (1 - actual_weight) * r_off - cost
        portfolio_value *= (1 + port_ret)

        # --- weight drift ---
        if portfolio_value > 0 and actual_weight > 0:
            on_portion = actual_weight * (1 + r_on)
            off_portion = (1 - actual_weight) * (1 + r_off)
            total = on_portion + off_portion
            actual_weight = on_portion / total if total > 0 else actual_weight

        # --- year-end tax ---
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
# MULTI-ASSET BACKTEST (for Experiment E)
# ============================================================
def run_multi_asset_backtest(
    prices: pd.DataFrame,
    daily_allocations: pd.DataFrame,  # columns = asset names, values = weights
    no_trade_band: float = 0.05,
) -> Dict:
    """Simple multi-asset backtest. Tax simplified: on total portfolio gains."""
    assets = daily_allocations.columns.tolist()
    rets = {a: prices[a].pct_change().fillna(0.0) for a in assets if a in prices.columns}

    portfolio_value = 1.0
    actual_weights = {a: 0.0 for a in assets}
    yearly_gains: Dict[int, float] = {}
    total_tax = 0.0
    n_trades = 0
    equity_vals = []
    prev_value = 1.0

    for i, dt in enumerate(prices.index):
        # target weights
        targets = {}
        for a in assets:
            targets[a] = float(daily_allocations.loc[dt, a]) if dt in daily_allocations.index else actual_weights.get(a, 0.0)

        # total absolute weight change
        total_change = sum(abs(targets.get(a, 0) - actual_weights.get(a, 0)) for a in assets)
        if total_change > no_trade_band:
            cost = total_change * (COST_BPS / 10000.0)
            actual_weights = dict(targets)
            n_trades += 1
        else:
            cost = 0.0

        # daily return
        port_ret = sum(actual_weights.get(a, 0) * float(rets[a].loc[dt]) for a in assets if a in rets) - cost
        portfolio_value *= (1 + port_ret)

        # weight drift
        if portfolio_value > 0:
            drifted = {}
            for a in assets:
                w = actual_weights.get(a, 0)
                r = float(rets[a].loc[dt]) if a in rets else 0
                drifted[a] = w * (1 + r)
            total = sum(drifted.values())
            if total > 0:
                actual_weights = {a: drifted[a] / total for a in assets}

        # year-end tax (simplified: on portfolio growth)
        year = dt.year
        if year not in yearly_gains:
            yearly_gains[year] = 0.0
        is_year_end = (i == len(prices.index) - 1) or \
                      (prices.index[i+1].year != year if i < len(prices.index) - 1 else True)
        if is_year_end:
            gain = portfolio_value - prev_value
            if gain > 0:
                tax = gain * TAX_RATE
                portfolio_value -= tax
                total_tax += tax
            prev_value = portfolio_value

        equity_vals.append(portfolio_value)

    equity = pd.Series(equity_vals, index=prices.index, name="equity")
    return {"equity": equity, "metrics": calc_metrics(equity, n_trades, total_tax)}


# ============================================================
# HELPER: rolling vol
# ============================================================
def rolling_vol(prices: pd.DataFrame, lookback: int) -> pd.Series:
    return prices["QQQ"].pct_change().rolling(lookback).std() * np.sqrt(TRADING_DAYS)


# ============================================================
# EXPERIMENT A: Pure Vol-Target (no E03 signal)
# ============================================================
def run_exp_A(prices: pd.DataFrame) -> Dict:
    print("\n" + "="*70)
    print("[A] Pure Vol-Target (no trend filter)")
    print("="*70)

    v_targets = [10, 12, 15, 18]
    lookbacks = [10, 20, 40, 60]
    results = {}

    for vt in v_targets:
        for lb in lookbacks:
            vol = rolling_vol(prices, lb)
            w = ((vt / 100.0) / (3 * vol)).clip(0, 1).fillna(0)
            r = run_flexible_backtest(prices, w, no_trade_band=0.05)
            m = r["metrics"]
            results[(vt, lb)] = m
            print(f"   V={vt}% LB={lb}d: CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% "
                  f"Calmar={m['Calmar']:.2f} Trades={m['Trades']}")

    # save grid
    ed = os.path.join(OUTPUT_DIR, "A_VolTarget_Only")
    os.makedirs(ed, exist_ok=True)
    rows = [{"V_target": vt, "Lookback": lb, **m} for (vt, lb), m in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ed, "grid.csv"), index=False)

    # find best Calmar
    best_k = max(results, key=lambda k: results[k]["Calmar"])
    print(f"\n   ★ Best Calmar: V={best_k[0]}% LB={best_k[1]}d → "
          f"CAGR={results[best_k]['CAGR']*100:.2f}% MDD={results[best_k]['MDD']*100:.2f}% "
          f"Calmar={results[best_k]['Calmar']:.2f}")

    return {"grid": results, "best": best_k, "best_metrics": results[best_k]}


# ============================================================
# EXPERIMENT B: E03 + Vol-Target
# ============================================================
def run_exp_B(prices: pd.DataFrame) -> Dict:
    print("\n" + "="*70)
    print("[B] E03 + Vol-Target (trend-gated)")
    print("="*70)

    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)

    v_targets = [10, 12, 15, 18]
    lookbacks = [10, 20, 40, 60]
    results = {}

    for vt in v_targets:
        for lb in lookbacks:
            vol = rolling_vol(prices, lb)
            w_vol = ((vt / 100.0) / (3 * vol)).clip(0, 1).fillna(0)
            # ON: vol-target weight, OFF: 10%
            w = signal_lag * w_vol + (1 - signal_lag) * OFF_TQQQ_WEIGHT
            r = run_flexible_backtest(prices, w, no_trade_band=0.05)
            m = r["metrics"]
            results[(vt, lb)] = m
            print(f"   V={vt}% LB={lb}d: CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% "
                  f"Calmar={m['Calmar']:.2f} Trades={m['Trades']}")

    ed = os.path.join(OUTPUT_DIR, "B_E03_VolTarget")
    os.makedirs(ed, exist_ok=True)
    rows = [{"V_target": vt, "Lookback": lb, **m} for (vt, lb), m in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ed, "grid.csv"), index=False)

    best_k = max(results, key=lambda k: results[k]["Calmar"])
    print(f"\n   ★ Best Calmar: V={best_k[0]}% LB={best_k[1]}d → "
          f"CAGR={results[best_k]['CAGR']*100:.2f}% MDD={results[best_k]['MDD']*100:.2f}% "
          f"Calmar={results[best_k]['Calmar']:.2f}")

    return {"grid": results, "best": best_k, "best_metrics": results[best_k]}


# ============================================================
# EXPERIMENT C: E03 + Vol-Target + Fast Exit
# ============================================================
def run_exp_C(prices: pd.DataFrame, best_vt: int, best_lb: int) -> Dict:
    print("\n" + "="*70)
    print(f"[C] E03 + Vol-Target (V={best_vt}%, LB={best_lb}d) + Fast Exit")
    print("="*70)

    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    vol = rolling_vol(prices, best_lb)
    w_vol = ((best_vt / 100.0) / (3 * vol)).clip(0, 1).fillna(0)
    w_base = signal_lag * w_vol + (1 - signal_lag) * OFF_TQQQ_WEIGHT

    qqq = prices["QQQ"]
    results = {}

    # C1: QQQ < MA(10) → cap 20%
    ma10 = qqq.rolling(10).mean()
    below_ma10 = (qqq < ma10).shift(1).fillna(False)
    w_c1 = w_base.copy()
    w_c1[below_ma10] = w_c1[below_ma10].clip(upper=0.20)
    r = run_flexible_backtest(prices, w_c1, no_trade_band=0.05)
    results["C1_MA10_cap20"] = r["metrics"]
    print(f"   C1 QQQ<MA10→cap20%: CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    # C2: QQQ < MA(20) → cap 20%
    ma20 = qqq.rolling(20).mean()
    below_ma20 = (qqq < ma20).shift(1).fillna(False)
    w_c2 = w_base.copy()
    w_c2[below_ma20] = w_c2[below_ma20].clip(upper=0.20)
    r = run_flexible_backtest(prices, w_c2, no_trade_band=0.05)
    results["C2_MA20_cap20"] = r["metrics"]
    print(f"   C2 QQQ<MA20→cap20%: CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    # C3: QQQ drawdown from 60d high >= 10% → cap 20%
    high60 = qqq.rolling(60).max()
    dd60 = (qqq / high60 - 1)
    dd_10 = (dd60 <= -0.10).shift(1).fillna(False)
    w_c3 = w_base.copy()
    w_c3[dd_10] = w_c3[dd_10].clip(upper=0.20)
    r = run_flexible_backtest(prices, w_c3, no_trade_band=0.05)
    results["C3_DD10pct_cap20"] = r["metrics"]
    print(f"   C3 QQQ DD≥10%→cap20%: CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    # C4: QQQ drawdown from 60d high >= 15% → cap 20%
    dd_15 = (dd60 <= -0.15).shift(1).fillna(False)
    w_c4 = w_base.copy()
    w_c4[dd_15] = w_c4[dd_15].clip(upper=0.20)
    r = run_flexible_backtest(prices, w_c4, no_trade_band=0.05)
    results["C4_DD15pct_cap20"] = r["metrics"]
    print(f"   C4 QQQ DD≥15%→cap20%: CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    # C5: Combined: QQQ < MA(20) AND vol>25% → cap 10%, else use vol-target
    high_vol = (vol > 0.25).shift(1).fillna(False)
    w_c5 = w_base.copy()
    combined_mask = below_ma20 & high_vol
    w_c5[combined_mask] = w_c5[combined_mask].clip(upper=0.10)
    r = run_flexible_backtest(prices, w_c5, no_trade_band=0.05)
    results["C5_MA20_HighVol_cap10"] = r["metrics"]
    print(f"   C5 MA20+HighVol→cap10%: CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    ed = os.path.join(OUTPUT_DIR, "C_FastExit")
    os.makedirs(ed, exist_ok=True)
    rows = [{"Variant": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ed, "variants.csv"), index=False)

    best_k = max(results, key=lambda k: results[k]["Calmar"])
    print(f"\n   ★ Best: {best_k} → Calmar={results[best_k]['Calmar']:.2f}")

    return {"variants": results, "best": best_k, "best_metrics": results[best_k]}


# ============================================================
# EXPERIMENT D: Discrete Regime
# ============================================================
def run_exp_D(prices: pd.DataFrame) -> Dict:
    print("\n" + "="*70)
    print("[D] Discrete Regime (vol buckets)")
    print("="*70)

    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    vol20 = rolling_vol(prices, 20).shift(1).fillna(0.15)

    configs = {
        "D1_15_25": {"low": 0.15, "high": 0.25, "w_low": 1.0, "w_mid": 0.60, "w_high": 0.20},
        "D2_18_28": {"low": 0.18, "high": 0.28, "w_low": 1.0, "w_mid": 0.50, "w_high": 0.10},
        "D3_12_20": {"low": 0.12, "high": 0.20, "w_low": 1.0, "w_mid": 0.70, "w_high": 0.30},
        "D4_15_25_noOff": {"low": 0.15, "high": 0.25, "w_low": 1.0, "w_mid": 0.60, "w_high": 0.20, "off_w": 0.0},
    }

    results = {}
    for name, cfg in configs.items():
        w = pd.Series(index=prices.index, dtype=float)
        low_mask = vol20 < cfg["low"]
        high_mask = vol20 > cfg["high"]
        mid_mask = ~low_mask & ~high_mask

        w[low_mask] = cfg["w_low"]
        w[mid_mask] = cfg["w_mid"]
        w[high_mask] = cfg["w_high"]

        off_w = cfg.get("off_w", OFF_TQQQ_WEIGHT)
        w_final = signal_lag * w + (1 - signal_lag) * off_w

        r = run_flexible_backtest(prices, w_final, no_trade_band=0.05)
        m = r["metrics"]
        results[name] = m
        print(f"   {name}: CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% "
              f"Calmar={m['Calmar']:.2f} Trades={m['Trades']}")

    ed = os.path.join(OUTPUT_DIR, "D_DiscreteRegime")
    os.makedirs(ed, exist_ok=True)
    rows = [{"Config": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ed, "configs.csv"), index=False)

    best_k = max(results, key=lambda k: results[k]["Calmar"])
    print(f"\n   ★ Best: {best_k} → Calmar={results[best_k]['Calmar']:.2f}")

    return {"configs": results, "best": best_k, "best_metrics": results[best_k]}


# ============================================================
# EXPERIMENT E: Multi-Asset Rotation
# ============================================================
def run_exp_E(prices: pd.DataFrame) -> Dict:
    print("\n" + "="*70)
    print("[E] Multi-Asset Rotation")
    print("="*70)

    risky_assets = []
    for a in ["TQQQ", "UPRO", "TLT"]:
        if a in prices.columns and not prices[a].isna().all():
            risky_assets.append(a)
    safe_asset = "SGOV"
    all_assets = risky_assets + [safe_asset]

    results = {}

    def _build_rotation_weights(mom_window: int, top_n: int, use_ma200: bool) -> pd.DataFrame:
        alloc = pd.DataFrame(0.0, index=prices.index, columns=all_assets)
        mom = pd.DataFrame(index=prices.index)
        for a in risky_assets:
            mom[a] = prices[a].pct_change(mom_window)

        ma200 = {}
        if use_ma200:
            for a in risky_assets:
                ma200[a] = prices[a].rolling(200).mean()

        # monthly rebalance (first trading day of each month)
        months = prices.index.to_period("M")
        rebal_dates = []
        prev_m = None
        for dt in prices.index:
            m = dt.to_period("M")
            if m != prev_m:
                rebal_dates.append(dt)
                prev_m = m

        current_alloc = {a: 0.0 for a in all_assets}
        current_alloc[safe_asset] = 1.0

        for i, dt in enumerate(prices.index):
            if dt in rebal_dates and not mom.loc[dt].isna().all():
                # rank by momentum
                scores = {}
                for a in risky_assets:
                    if pd.isna(mom.loc[dt, a]):
                        continue
                    if use_ma200 and a in ma200:
                        if pd.isna(ma200[a].loc[dt]):
                            continue
                        if float(prices.loc[dt, a]) < float(ma200[a].loc[dt]):
                            continue  # fails MA200 filter
                    scores[a] = float(mom.loc[dt, a])

                new_alloc = {a: 0.0 for a in all_assets}
                if scores:
                    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
                    selected = ranked[:top_n]
                    w_each = 1.0 / len(selected)
                    for a in selected:
                        new_alloc[a] = w_each
                else:
                    new_alloc[safe_asset] = 1.0

                # ensure weights sum to 1
                total_risky = sum(new_alloc[a] for a in risky_assets)
                new_alloc[safe_asset] = 1.0 - total_risky
                current_alloc = new_alloc

            for a in all_assets:
                alloc.loc[dt, a] = current_alloc.get(a, 0.0)

        return alloc

    # E1: Top-1 by 6m momentum + MA200
    print("   E1: Top-1, 6m momentum, MA200 filter ...")
    alloc = _build_rotation_weights(126, 1, True)
    r = run_multi_asset_backtest(prices, alloc, no_trade_band=0.05)
    results["E1_Top1_6m_MA200"] = r["metrics"]
    print(f"      CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    # E2: Top-1 by 12m momentum + MA200
    print("   E2: Top-1, 12m momentum, MA200 filter ...")
    alloc = _build_rotation_weights(252, 1, True)
    r = run_multi_asset_backtest(prices, alloc, no_trade_band=0.05)
    results["E2_Top1_12m_MA200"] = r["metrics"]
    print(f"      CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    # E3: Top-2 by 6m momentum + MA200, 50/50
    print("   E3: Top-2, 6m momentum, MA200 filter, 50/50 ...")
    alloc = _build_rotation_weights(126, 2, True)
    r = run_multi_asset_backtest(prices, alloc, no_trade_band=0.05)
    results["E3_Top2_6m_MA200"] = r["metrics"]
    print(f"      CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    # E4: Top-1 by 6m, no MA200 filter
    print("   E4: Top-1, 6m momentum, NO MA200 filter ...")
    alloc = _build_rotation_weights(126, 1, False)
    r = run_multi_asset_backtest(prices, alloc, no_trade_band=0.05)
    results["E4_Top1_6m_noMA200"] = r["metrics"]
    print(f"      CAGR={r['metrics']['CAGR']*100:.2f}% MDD={r['metrics']['MDD']*100:.2f}% Calmar={r['metrics']['Calmar']:.2f}")

    ed = os.path.join(OUTPUT_DIR, "E_MultiAsset")
    os.makedirs(ed, exist_ok=True)
    rows = [{"Variant": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ed, "variants.csv"), index=False)

    best_k = max(results, key=lambda k: results[k]["Calmar"])
    print(f"\n   ★ Best: {best_k} → Calmar={results[best_k]['Calmar']:.2f}")

    return {"variants": results, "best": best_k, "best_metrics": results[best_k]}


# ============================================================
# E03 BASELINE (for comparison)
# ============================================================
def run_e03_baseline(prices: pd.DataFrame) -> Dict:
    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    w = signal_lag * 1.0 + (1 - signal_lag) * OFF_TQQQ_WEIGHT
    return run_flexible_backtest(prices, w, no_trade_band=0.0)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("       MDD Reduction Suite: 5 Experiments (A-E)")
    print("       Goal: Reduce MDD while maintaining CAGR")
    print("=" * 70)

    prices = load_data()

    # --- Baseline ---
    print("\n[Baseline] E03 Ensemble + SGOV ...")
    baseline = run_e03_baseline(prices)
    bm = baseline["metrics"]
    print(f"   CAGR={bm['CAGR']*100:.2f}% MDD={bm['MDD']*100:.2f}% "
          f"Sharpe={bm['Sharpe']:.2f} Calmar={bm['Calmar']:.2f}")

    # --- Experiments ---
    res_A = run_exp_A(prices)
    res_B = run_exp_B(prices)
    res_C = run_exp_C(prices, res_B["best"][0], res_B["best"][1])
    res_D = run_exp_D(prices)
    res_E = run_exp_E(prices)

    # ============================================================
    # MASTER COMPARISON
    # ============================================================
    print("\n" + "=" * 70)
    print("                    MASTER COMPARISON")
    print("=" * 70)

    comparison = {
        "E03_Baseline": bm,
        f"A_Best_V{res_A['best'][0]}_LB{res_A['best'][1]}": res_A["best_metrics"],
        f"B_Best_V{res_B['best'][0]}_LB{res_B['best'][1]}": res_B["best_metrics"],
        f"C_Best_{res_C['best']}": res_C["best_metrics"],
        f"D_Best_{res_D['best']}": res_D["best_metrics"],
        f"E_Best_{res_E['best']}": res_E["best_metrics"],
    }

    print(f"\n   {'Strategy':<35} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Trades':>8}")
    print("   " + "-" * 80)
    for name, m in comparison.items():
        print(f"   {name:<35} {m['CAGR']*100:>7.2f}% {m['MDD']*100:>7.2f}% "
              f"{m['Sharpe']:>8.2f} {m['Calmar']:>8.2f} {m['Trades']:>8}")

    # deltas vs baseline
    print(f"\n   {'Strategy':<35} {'ΔCAGR':>8} {'ΔMDD':>8} {'ΔCalmar':>8}")
    print("   " + "-" * 60)
    for name, m in comparison.items():
        if name == "E03_Baseline":
            continue
        dc = (m["CAGR"] - bm["CAGR"]) * 100
        dm = (m["MDD"] - bm["MDD"]) * 100
        dcal = m["Calmar"] - bm["Calmar"]
        print(f"   {name:<35} {dc:>+7.2f}% {dm:>+7.2f}% {dcal:>+8.2f}")

    # save all
    with open(os.path.join(OUTPUT_DIR, "master_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    rows = [{"Strategy": k, **v} for k, v in comparison.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "master_comparison.csv"), index=False)

    # --- Run best candidates through equity curve comparison ---
    print("\n   Generating equity curve comparison plot ...")
    best_configs = {
        "E03_Baseline": lambda: run_e03_baseline(prices),
    }

    # Re-run bests for equity curves
    bv, bl = res_B["best"]
    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)
    vol = rolling_vol(prices, bl)
    w_vol = ((bv / 100.0) / (3 * vol)).clip(0, 1).fillna(0)
    w_b = signal_lag * w_vol + (1 - signal_lag) * OFF_TQQQ_WEIGHT

    eq_baseline = baseline["equity"]
    eq_b = run_flexible_backtest(prices, w_b, no_trade_band=0.05)["equity"]

    # Best C variant
    qqq = prices["QQQ"]
    best_c_name = res_C["best"]
    w_c = w_b.copy()
    if "MA10" in best_c_name:
        ma = qqq.rolling(10).mean()
        mask = (qqq < ma).shift(1).fillna(False)
        w_c[mask] = w_c[mask].clip(upper=0.20)
    elif "MA20" in best_c_name:
        ma = qqq.rolling(20).mean()
        mask = (qqq < ma).shift(1).fillna(False)
        w_c[mask] = w_c[mask].clip(upper=0.20)
    elif "DD10" in best_c_name:
        h60 = qqq.rolling(60).max()
        mask = ((qqq / h60 - 1) <= -0.10).shift(1).fillna(False)
        w_c[mask] = w_c[mask].clip(upper=0.20)
    elif "DD15" in best_c_name:
        h60 = qqq.rolling(60).max()
        mask = ((qqq / h60 - 1) <= -0.15).shift(1).fillna(False)
        w_c[mask] = w_c[mask].clip(upper=0.20)
    elif "MA20_HighVol" in best_c_name:
        ma20 = qqq.rolling(20).mean()
        mask = ((qqq < ma20) & (vol > 0.25)).shift(1).fillna(False)
        w_c[mask] = w_c[mask].clip(upper=0.10)
    eq_c = run_flexible_backtest(prices, w_c, no_trade_band=0.05)["equity"]

    plt.figure(figsize=(16, 8))
    plt.plot(eq_baseline, label=f"E03 Baseline (CAGR={bm['CAGR']*100:.1f}%, MDD={bm['MDD']*100:.1f}%)", lw=2)
    plt.plot(eq_b, label=f"B Best V{bv}/LB{bl} (CAGR={res_B['best_metrics']['CAGR']*100:.1f}%, MDD={res_B['best_metrics']['MDD']*100:.1f}%)", lw=1.5, ls="--")
    plt.plot(eq_c, label=f"C Best {best_c_name} (CAGR={res_C['best_metrics']['CAGR']*100:.1f}%, MDD={res_C['best_metrics']['MDD']*100:.1f}%)", lw=1.5, ls="-.")
    plt.yscale("log")
    plt.title("MDD Reduction Suite: Equity Curves (Log Scale)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "equity_comparison.png"), dpi=150)
    plt.close()

    print(f"\n📁 All results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    main()
