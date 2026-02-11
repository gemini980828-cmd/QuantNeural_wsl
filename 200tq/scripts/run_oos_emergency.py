# -*- coding: utf-8 -*-
"""
E03 + F1 Signal Stability Filter + Emergency Exit (-5%/-15%)
OOS Verification Suite
=======================================================

Tests:
  1) E03 Baseline (full period)
  2) E03 + Emergency Exit -5%/-15% (full period)
  3) E03 + F1 Signal Stability (full period)
  4) E03 + F1 + Emergency Exit -5%/-15% (full period)
  5) OOS Verification: IS 2010-2017, OOS 2018-2025 for ALL above
  6) F3 Graduated (with/without emergency) for comparison

Author: QuantNeural OOS+Emergency Suite v2026.2
"""

import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
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
OOS_SPLIT = "2018-01-01"

EMERGENCY_QQQ = -0.05
EMERGENCY_TQQQ = -0.15

# F1 best params from previous run
F1_FLIP_WINDOW = 40
F1_FLIP_THRESHOLD = 3
F1_REDUCED_WEIGHT = 0.70

# F3 Graduated params
F3_AC_LOOKBACK = 20
F3_AC_LAG = 1
F3_AC_THRESHOLD = -0.05
F3_AC_REDUCED_WEIGHT = 0.50

OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/experiments/oos_emergency"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> pd.DataFrame:
    import yfinance as yf
    print("📥 Downloading data...")
    tickers = ["QQQ", "TQQQ", "SGOV", "SHV"]
    raw = yf.download(tickers=tickers, start="2009-01-01", end=END_DATE,
                      auto_adjust=True, progress=False, group_by="ticker", threads=True)
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
# SIGNAL & HELPERS
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


def rolling_signal_flips(signal: pd.Series, window: int) -> pd.Series:
    changes = signal.diff().abs()
    return changes.rolling(window, min_periods=1).sum()


def rolling_autocorrelation(returns: pd.Series, lookback: int, lag: int) -> pd.Series:
    result = pd.Series(index=returns.index, dtype=float)
    vals = returns.values
    for i in range(lookback + lag, len(vals)):
        window = vals[i - lookback:i]
        x, y = window[:-lag], window[lag:]
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
    return {"Final": round(final, 4), "CAGR": round(cagr, 6), "MDD": round(mdd, 6),
            "Sharpe": round(sharpe, 4), "Sortino": round(sortino, 4),
            "Calmar": round(calmar, 4), "Trades": n_trades, "TotalTax": round(total_tax, 4)}


def subperiod_metrics(equity: pd.Series, start: str, end: str) -> Dict:
    sub = equity[(equity.index >= start) & (equity.index <= end)]
    if len(sub) < 2:
        return {"CAGR": 0, "MDD": 0, "Sharpe": 0, "Calmar": 0, "Final": 0}
    norm = sub / sub.iloc[0]
    return calc_metrics(norm)


# ============================================================
# UNIFIED BACKTEST ENGINE
# ============================================================
def run_backtest(
    prices: pd.DataFrame,
    use_f1: bool = False,
    use_f3_graduated: bool = False,
    use_emergency: bool = False,
    label: str = "E03",
) -> Dict:
    """
    Unified engine supporting all strategy variants:
    - E03 baseline
    - +F1 signal stability filter
    - +F3 graduated (F1 + autocorrelation)
    - +Emergency exit (-5%/-15%)
    Any combination allowed.
    """
    signal = generate_ensemble_signal(prices)
    signal_lag = signal.shift(1).fillna(0).astype(int)

    # Compute F1 filter
    f1_choppy = pd.Series(False, index=prices.index)
    if use_f1 or use_f3_graduated:
        flips = rolling_signal_flips(signal_lag, F1_FLIP_WINDOW).shift(1).fillna(0)
        f1_choppy = flips >= F1_FLIP_THRESHOLD

    # Compute F2 autocorrelation filter (for F3 only)
    f2_mean_revert = pd.Series(False, index=prices.index)
    if use_f3_graduated:
        qqq_ret = prices["QQQ"].pct_change().fillna(0.0)
        ac = rolling_autocorrelation(qqq_ret, F3_AC_LOOKBACK, F3_AC_LAG).shift(1).fillna(0)
        f2_mean_revert = ac < F3_AC_THRESHOLD

    # Build base weight series (before emergency)
    base_weight = pd.Series(0.0, index=prices.index)
    for i, dt in enumerate(prices.index):
        sig = int(signal_lag.iloc[i])
        if sig == 1:
            w = 1.0
            if use_f3_graduated:
                n_active = int(f1_choppy.iloc[i]) + int(f2_mean_revert.iloc[i])
                if n_active == 1:
                    w = 0.60
                elif n_active >= 2:
                    w = 0.30
            elif use_f1:
                if f1_choppy.iloc[i]:
                    w = F1_REDUCED_WEIGHT
            base_weight.iloc[i] = w
        else:
            base_weight.iloc[i] = OFF_TQQQ_WEIGHT

    # Run the backtest with emergency exit
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    qqq_ret = prices["QQQ"].pct_change().fillna(0.0)
    off_asset = "SGOV" if "SGOV" in prices.columns and not prices["SGOV"].isna().all() else "CASH"
    off_ret = prices[off_asset].pct_change().fillna(0.0)

    portfolio_value = 1.0
    current_weight = 0.0
    tqqq_entry_price: Optional[float] = None
    tqqq_cost_basis = 0.0
    tqqq_shares = 0.0
    yearly_gains: Dict[int, float] = {}
    total_tax = 0.0
    n_trades = 0
    emergency_triggers: List[Dict] = []
    emergency_cooldown = 0
    equity_vals = []

    for i, dt in enumerate(prices.index):
        px_tqqq = float(prices.loc[dt, "TQQQ"])
        target = float(base_weight.iloc[i])

        # --- Emergency exit check ---
        em_trigger = None
        if use_emergency and i > 0:
            qr = float(qqq_ret.iloc[i])
            if qr <= EMERGENCY_QQQ:
                em_trigger = f"QQQ_CRASH ({qr*100:.1f}%)"
            if (tqqq_entry_price is not None and current_weight > 0.5):
                tqqq_dd = (px_tqqq / tqqq_entry_price) - 1.0
                if tqqq_dd <= EMERGENCY_TQQQ:
                    em_trigger = f"TQQQ_STOP ({tqqq_dd*100:.1f}%)"

        if em_trigger:
            target = OFF_TQQQ_WEIGHT
            emergency_triggers.append({
                "date": dt.strftime("%Y-%m-%d"), "trigger": em_trigger,
                "portfolio_value": round(portfolio_value, 4)
            })
            emergency_cooldown = 1
        elif emergency_cooldown > 0:
            target = OFF_TQQQ_WEIGHT
            emergency_cooldown -= 1

        # Track entry price
        if target > 0.5 and current_weight <= 0.5:
            tqqq_entry_price = px_tqqq
        elif target <= 0.5:
            tqqq_entry_price = None

        # Weight change & cost
        weight_change = abs(target - current_weight)
        cost = weight_change * (COST_BPS / 10000.0) if weight_change > 1e-6 else 0.0

        # Tax bookkeeping
        year = dt.year
        if year not in yearly_gains:
            yearly_gains[year] = 0.0

        if weight_change > 1e-6:
            n_trades += 1
            if target < current_weight and tqqq_shares > 0 and tqqq_cost_basis > 0:
                avg_cost = tqqq_cost_basis / tqqq_shares
                sold_value = (current_weight - target) * portfolio_value
                sold_shares = sold_value / px_tqqq if px_tqqq > 0 else 0
                gain = sold_shares * (px_tqqq - avg_cost)
                yearly_gains[year] += gain
                sr = min(1.0, sold_shares / tqqq_shares) if tqqq_shares > 0 else 0
                tqqq_cost_basis *= (1 - sr)
                tqqq_shares -= sold_shares
            elif target > current_weight:
                buy_value = (target - current_weight) * portfolio_value
                buy_shares = buy_value / px_tqqq if px_tqqq > 0 else 0
                tqqq_cost_basis += buy_value
                tqqq_shares += buy_shares

        # Daily PnL
        port_ret = target * float(tqqq_ret.iloc[i]) + (1 - target) * float(off_ret.iloc[i]) - cost
        portfolio_value *= (1 + port_ret)
        current_weight = target

        # Year-end tax
        is_year_end = (i == len(prices.index) - 1) or \
                      (prices.index[i+1].year != year if i < len(prices.index) - 1 else True)
        if is_year_end and year in yearly_gains:
            taxable = max(0, yearly_gains[year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            total_tax += tax

        equity_vals.append(portfolio_value)

    equity = pd.Series(equity_vals, index=prices.index, name="equity")
    metrics = calc_metrics(equity, n_trades, total_tax)

    return {
        "equity": equity, "metrics": metrics,
        "emergency_triggers": emergency_triggers,
        "label": label,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  E03 + F1 + Emergency Exit (-5%/-15%) — OOS Verification Suite")
    print("=" * 70)

    prices = load_data()

    # ============================================================
    # FULL PERIOD RUNS
    # ============================================================
    strategies = {
        "A_E03_Baseline": dict(use_f1=False, use_f3_graduated=False, use_emergency=False),
        "B_E03_Emergency": dict(use_f1=False, use_f3_graduated=False, use_emergency=True),
        "C_E03_F1": dict(use_f1=True, use_f3_graduated=False, use_emergency=False),
        "D_E03_F1_Emergency": dict(use_f1=True, use_f3_graduated=False, use_emergency=True),
        "E_E03_F3_Graduated": dict(use_f1=False, use_f3_graduated=True, use_emergency=False),
        "F_E03_F3_Graduated_Emergency": dict(use_f1=False, use_f3_graduated=True, use_emergency=True),
    }

    full_results = {}
    for name, params in strategies.items():
        print(f"\n[Full] {name} ...")
        r = run_backtest(prices, label=name, **params)
        m = r["metrics"]
        full_results[name] = r
        em_cnt = len(r["emergency_triggers"])
        print(f"   CAGR={m['CAGR']*100:.2f}% MDD={m['MDD']*100:.2f}% "
              f"Sharpe={m['Sharpe']:.2f} Calmar={m['Calmar']:.2f} "
              f"Trades={m['Trades']} Emergency={em_cnt}")

    # ============================================================
    # OOS VERIFICATION: IS 2010-2017 / OOS 2018-2025
    # ============================================================
    print("\n" + "=" * 70)
    print("  OOS Verification: IS 2010-2017 → OOS 2018-2025")
    print("=" * 70)

    oos_results = {}
    for name, r in full_results.items():
        eq = r["equity"]
        is_m = subperiod_metrics(eq, START_DATE, "2017-12-31")
        oos_m = subperiod_metrics(eq, OOS_SPLIT, END_DATE)
        deg_cagr = oos_m["CAGR"] / is_m["CAGR"] if is_m["CAGR"] != 0 else 0
        deg_calmar = oos_m["Calmar"] / is_m["Calmar"] if is_m["Calmar"] != 0 else 0
        oos_results[name] = {
            "IS": is_m, "OOS": oos_m,
            "DegradationRatio_CAGR": round(deg_cagr, 4),
            "DegradationRatio_Calmar": round(deg_calmar, 4),
        }
        print(f"\n   {name}:")
        print(f"     IS:  CAGR={is_m['CAGR']*100:.2f}% MDD={is_m['MDD']*100:.2f}% "
              f"Calmar={is_m['Calmar']:.2f}")
        print(f"     OOS: CAGR={oos_m['CAGR']*100:.2f}% MDD={oos_m['MDD']*100:.2f}% "
              f"Calmar={oos_m['Calmar']:.2f}")
        print(f"     Degradation CAGR: {deg_cagr:.3f}  Calmar: {deg_calmar:.3f}")

    # ============================================================
    # MASTER COMPARISON TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("                    MASTER COMPARISON (FULL PERIOD)")
    print("=" * 70)

    baseline_m = full_results["A_E03_Baseline"]["metrics"]

    header = f"   {'Strategy':<35} {'CAGR':>7} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Trades':>7} {'Emrg':>5}"
    print(header)
    print("   " + "-" * 80)
    for name, r in full_results.items():
        m = r["metrics"]
        em = len(r["emergency_triggers"])
        print(f"   {name:<35} {m['CAGR']*100:>6.2f}% {m['MDD']*100:>7.2f}% "
              f"{m['Sharpe']:>7.2f} {m['Calmar']:>7.2f} {m['Trades']:>7} {em:>5}")

    print(f"\n   {'Strategy':<35} {'ΔCAGR':>7} {'ΔMDD':>8} {'ΔCalmar':>8}")
    print("   " + "-" * 60)
    for name, r in full_results.items():
        if name == "A_E03_Baseline":
            continue
        m = r["metrics"]
        dc = (m["CAGR"] - baseline_m["CAGR"]) * 100
        dm = (m["MDD"] - baseline_m["MDD"]) * 100
        dcal = m["Calmar"] - baseline_m["Calmar"]
        print(f"   {name:<35} {dc:>+6.2f}% {dm:>+7.2f}% {dcal:>+8.2f}")

    # ============================================================
    # OOS SUMMARY TABLE
    # ============================================================
    print(f"\n   {'Strategy':<35} {'IS CAGR':>8} {'OOS CAGR':>9} {'IS Cal':>7} {'OOS Cal':>8} {'Deg CAGR':>9} {'Deg Cal':>8}")
    print("   " + "-" * 90)
    for name, oos in oos_results.items():
        print(f"   {name:<35} {oos['IS']['CAGR']*100:>7.2f}% {oos['OOS']['CAGR']*100:>8.2f}% "
              f"{oos['IS']['Calmar']:>7.2f} {oos['OOS']['Calmar']:>8.2f} "
              f"{oos['DegradationRatio_CAGR']:>9.3f} {oos['DegradationRatio_Calmar']:>8.3f}")

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    # Master comparison CSV
    rows = []
    for name, r in full_results.items():
        m = r["metrics"]
        oos = oos_results[name]
        rows.append({
            "Strategy": name,
            **m,
            "EmergencyTriggers": len(r["emergency_triggers"]),
            "IS_CAGR": oos["IS"]["CAGR"], "IS_MDD": oos["IS"]["MDD"],
            "IS_Calmar": oos["IS"]["Calmar"],
            "OOS_CAGR": oos["OOS"]["CAGR"], "OOS_MDD": oos["OOS"]["MDD"],
            "OOS_Calmar": oos["OOS"]["Calmar"],
            "DegradationRatio_CAGR": oos["DegradationRatio_CAGR"],
            "DegradationRatio_Calmar": oos["DegradationRatio_Calmar"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "master_comparison.csv"), index=False)

    # Emergency triggers
    for name, r in full_results.items():
        if r["emergency_triggers"]:
            ed = os.path.join(OUTPUT_DIR, name)
            os.makedirs(ed, exist_ok=True)
            pd.DataFrame(r["emergency_triggers"]).to_csv(
                os.path.join(ed, "emergency_events.csv"), index=False)

    # JSON
    json_out = {}
    for name, r in full_results.items():
        json_out[name] = {
            "full_period": r["metrics"],
            "emergency_count": len(r["emergency_triggers"]),
            "oos": oos_results[name],
        }
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(json_out, f, indent=2, default=str)

    # ============================================================
    # EQUITY CURVES
    # ============================================================
    print("\n   Generating plots ...")

    # Full period
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = {"A_E03_Baseline": "black", "B_E03_Emergency": "gray",
              "C_E03_F1": "blue", "D_E03_F1_Emergency": "red",
              "E_E03_F3_Graduated": "green", "F_E03_F3_Graduated_Emergency": "orange"}
    styles = {"A_E03_Baseline": "-", "B_E03_Emergency": "--",
              "C_E03_F1": "-", "D_E03_F1_Emergency": "-",
              "E_E03_F3_Graduated": "-.", "F_E03_F3_Graduated_Emergency": "-."}
    for name, r in full_results.items():
        m = r["metrics"]
        ax.plot(r["equity"], label=f"{name} (CAGR={m['CAGR']*100:.1f}%, MDD={m['MDD']*100:.1f}%)",
                color=colors.get(name, "purple"), ls=styles.get(name, "-"), lw=1.5)
    ax.set_yscale("log")
    ax.set_title("Full Period Equity Curves (Log Scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "equity_full_period.png"), dpi=150)
    plt.close()

    # IS vs OOS split
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for name, r in full_results.items():
        eq = r["equity"]
        # IS
        eq_is = eq[eq.index < OOS_SPLIT]
        eq_is_n = eq_is / eq_is.iloc[0]
        axes[0].plot(eq_is_n, label=name.split("_", 1)[1][:20],
                     color=colors.get(name, "purple"), ls=styles.get(name, "-"), lw=1.3)
        # OOS
        eq_oos = eq[eq.index >= OOS_SPLIT]
        eq_oos_n = eq_oos / eq_oos.iloc[0]
        axes[1].plot(eq_oos_n, label=name.split("_", 1)[1][:20],
                     color=colors.get(name, "purple"), ls=styles.get(name, "-"), lw=1.3)
    axes[0].set_yscale("log"); axes[0].set_title("IS Period (2010-2017)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_yscale("log"); axes[1].set_title("OOS Period (2018-2025)")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "equity_is_vs_oos.png"), dpi=150)
    plt.close()

    print(f"\n📁 All results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
