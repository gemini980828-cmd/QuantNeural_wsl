# -*- coding: utf-8 -*-
"""
54% CAGR Reproduction Experiments
=================================

Testing various assumptions to reproduce the claimed 54% CAGR:
- E11: t+0 execution + no cost + pre-tax (baseline reproduction attempt)
- E12: E11 + overheat 1.51 (151% instead of 251%)
- E13: E11 + annualized volatility interpretation
- E14: E11 + overheat 1.51 + annualized vol
- E15: E11 + no QLD step + overheat 1.51
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
TRADING_DAYS = 252

SHORT_MA = 3

OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/experiments/reproduce_54_cagr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSETS = ["TQQQ", "QLD", "SGOV"]


@dataclass
class ExperimentConfig:
    name: str
    description: str
    
    # Strategy type: "user" or "e03"
    strategy_type: str = "user"
    
    # E03 specific
    ensemble_windows: list = None  # type: ignore
    
    # Execution mode
    use_t0_execution: bool = False  # True = look-ahead, False = t+1
    cost_bps: float = 10.0  # 0 = no cost
    apply_tax: bool = True  # False = pre-tax only
    tax_rate: float = 0.22
    
    # Strategy parameters (for user strategy)
    ma_long_qqq: int = 161
    ma_long_tqqq: int = 200
    use_vol_lock: bool = True
    vol_threshold: float = 0.062
    vol_annualized: bool = False  # True = interpret as annualized vol
    use_overheat: bool = True
    overheat_ratio: float = 2.51  # 1.51 = 151%, 2.51 = 251%
    reentry_ratio: float = 2.18   # 1.18 = 118%, 2.18 = 218%
    use_qld_step: bool = True
    use_entry_gate: bool = False
    
    def __post_init__(self):
        if self.ensemble_windows is None:
            self.ensemble_windows = [160, 165, 170]


# Define experiments to reproduce 54%
EXPERIMENTS = [
    # =========================================================================
    # Baselines
    # =========================================================================
    
    # User strategy: t+1 (our original implementation)
    ExperimentConfig(
        name="User_t1_baseline",
        description="User: t+1, 10bps, 22% tax",
        strategy_type="user",
        use_t0_execution=False,
        cost_bps=10.0,
        apply_tax=True,
    ),
    
    # E03: t+1 (our original implementation)
    ExperimentConfig(
        name="E03_t1_baseline",
        description="E03: t+1, 10bps, 22% tax",
        strategy_type="e03",
        use_t0_execution=False,
        cost_bps=10.0,
        apply_tax=True,
    ),
    
    # =========================================================================
    # User Strategy: AH variants
    # =========================================================================
    
    ExperimentConfig(
        name="User_t0_ideal",
        description="User: t+0, no cost, no tax",
        strategy_type="user",
        use_t0_execution=True,
        cost_bps=0.0,
        apply_tax=False,
    ),
    
    ExperimentConfig(
        name="User_AH_20bps",
        description="User: AH 0.2% slip, no tax",
        strategy_type="user",
        use_t0_execution=True,
        cost_bps=20.0,
        apply_tax=False,
    ),
    
    ExperimentConfig(
        name="User_AH_20bps_tax",
        description="User: AH 0.2% slip, 22% tax",
        strategy_type="user",
        use_t0_execution=True,
        cost_bps=20.0,
        apply_tax=True,
    ),
    
    ExperimentConfig(
        name="User_AH_50bps_tax",
        description="User: AH 0.5% slip, 22% tax",
        strategy_type="user",
        use_t0_execution=True,
        cost_bps=50.0,
        apply_tax=True,
    ),
    
    # =========================================================================
    # E03 Strategy: AH variants
    # =========================================================================
    
    ExperimentConfig(
        name="E03_t0_ideal",
        description="E03: t+0, no cost, no tax",
        strategy_type="e03",
        use_t0_execution=True,
        cost_bps=0.0,
        apply_tax=False,
    ),
    
    ExperimentConfig(
        name="E03_AH_20bps",
        description="E03: AH 0.2% slip, no tax",
        strategy_type="e03",
        use_t0_execution=True,
        cost_bps=20.0,
        apply_tax=False,
    ),
    
    ExperimentConfig(
        name="E03_AH_20bps_tax",
        description="E03: AH 0.2% slip, 22% tax",
        strategy_type="e03",
        use_t0_execution=True,
        cost_bps=20.0,
        apply_tax=True,
    ),
    
    ExperimentConfig(
        name="E03_AH_50bps",
        description="E03: AH 0.5% slip, no tax",
        strategy_type="e03",
        use_t0_execution=True,
        cost_bps=50.0,
        apply_tax=False,
    ),
    
    ExperimentConfig(
        name="E03_AH_50bps_tax",
        description="E03: AH 0.5% slip, 22% tax",
        strategy_type="e03",
        use_t0_execution=True,
        cost_bps=50.0,
        apply_tax=True,
    ),
    
    ExperimentConfig(
        name="E03_AH_100bps_tax",
        description="E03: AH 1.0% slip, 22% tax",
        strategy_type="e03",
        use_t0_execution=True,
        cost_bps=100.0,
        apply_tax=True,
    ),
]


# =============================================================================
# Data Loading (same as original)
# =============================================================================

def build_synthetic_leverage(base, leverage):
    r = base.pct_change().fillna(0.0)
    r_lev = leverage * r
    return 100.0 * (1.0 + r_lev).cumprod()


def splice_series(synth, actual):
    if actual is None or actual.isna().all():
        return synth
    actual = actual.reindex(synth.index)
    first = actual.first_valid_index()
    if first is None:
        return synth
    if first > synth.index[0]:
        denom = float(synth.loc[first])
        scale = float(actual.loc[first]) / denom if denom != 0 else 1.0
        combined = synth * scale
        combined.loc[first:] = actual.loc[first:].ffill()
        return combined
    return actual.ffill()


def build_off_asset(base_idx, sgov, shv):
    if sgov is None or sgov.isna().all():
        if shv is None or shv.isna().all():
            return pd.Series(100.0, index=base_idx)
        sgov = shv
    sgov = sgov.reindex(base_idx).ffill()
    if sgov.isna().any():
        if shv is not None and not shv.isna().all():
            shv = shv.reindex(base_idx).ffill()
            sgov = sgov.fillna(shv)
    if sgov.isna().any():
        sgov = sgov.fillna(100.0)
    return sgov


def load_data() -> pd.DataFrame:
    import yfinance as yf

    buffer_start = pd.to_datetime(START_DATE) - pd.Timedelta(days=500)
    tickers = ["QQQ", "TQQQ", "QLD", "SGOV", "SHV"]

    raw = yf.download(
        tickers=tickers,
        start=buffer_start.strftime("%Y-%m-%d"),
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    prices = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in raw.columns:
                prices[t] = raw[(t, "Close")]
    else:
        if "Close" in raw.columns:
            prices["QQQ"] = raw["Close"]

    if "QQQ" not in prices.columns:
        raise ValueError("QQQ data not available from yfinance")

    qqq = prices["QQQ"].ffill()
    base_idx = pd.DatetimeIndex(qqq.index)

    synth_2x = build_synthetic_leverage(qqq, 2.0)
    synth_3x = build_synthetic_leverage(qqq, 3.0)

    tqqq_actual = prices["TQQQ"] if "TQQQ" in prices.columns else None
    qld_actual = prices["QLD"] if "QLD" in prices.columns else None

    tqqq = splice_series(synth_3x, tqqq_actual)
    qld = splice_series(synth_2x, qld_actual)

    sgov = prices["SGOV"] if "SGOV" in prices.columns else None
    shv = prices["SHV"] if "SHV" in prices.columns else None
    sgov_series = build_off_asset(base_idx, sgov, shv)

    df = pd.DataFrame(index=base_idx)
    df["QQQ"] = qqq
    df["TQQQ"] = tqqq
    df["QLD"] = qld
    df["SGOV"] = sgov_series

    df = df.dropna(subset=["QQQ", "TQQQ", "QLD", "SGOV"])
    return df


# =============================================================================
# Signal Generation
# =============================================================================

def generate_ensemble_signal(prices: pd.DataFrame, windows: list) -> pd.Series:
    """E03 ensemble voting signal"""
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    votes = pd.DataFrame(index=prices.index)
    for window in windows:
        ma_long = qqq.rolling(window).mean()
        votes[f"ma{window}"] = (ma_short > ma_long).astype(int)
    threshold = len(windows) // 2 + 1
    return (votes.sum(axis=1) >= threshold).astype(int)


def build_e03_weights(prices: pd.DataFrame, windows: list):
    """Build weights for E03 ensemble strategy"""
    signal = generate_ensemble_signal(prices, windows)
    weights = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    weights.loc[signal == 1, "TQQQ"] = 1.0
    weights.loc[signal == 0, "SGOV"] = 1.0
    return {"weights": weights, "signal": signal}


def compute_trend_score(prices: pd.DataFrame, ma_long_qqq: int, ma_long_tqqq: int) -> pd.Series:
    qqq = prices["QQQ"]
    tqqq = prices["TQQQ"]
    qqq_ma3 = qqq.rolling(SHORT_MA).mean()
    qqq_long = qqq.rolling(ma_long_qqq).mean()
    tqqq_long = tqqq.rolling(ma_long_tqqq).mean()
    
    score = pd.Series(0, index=prices.index, dtype=int)
    score += (qqq_ma3 > qqq_long).astype(int)
    score += (tqqq > tqqq_long).astype(int)
    return score


def compute_overheat_state(tqqq, ma_long, overheat_ratio, reentry_ratio):
    ma = tqqq.rolling(ma_long).mean()
    ratio = tqqq / ma
    state = False
    series = pd.Series(index=tqqq.index, dtype=bool)
    
    for dt in tqqq.index:
        r = ratio.loc[dt]
        if pd.isna(r):
            series.loc[dt] = state
            continue
        if not state and r >= overheat_ratio:
            state = True
        elif state and r <= reentry_ratio:
            state = False
        series.loc[dt] = state
    
    return series, ratio


def map_score_to_asset(score: int, use_qld_step: bool) -> str:
    if score <= 0:
        return "SGOV"
    if score == 1 and use_qld_step:
        return "QLD"
    return "TQQQ"


def build_user_weights(prices: pd.DataFrame, config: ExperimentConfig):
    score = compute_trend_score(prices, config.ma_long_qqq, config.ma_long_tqqq).fillna(0).astype(int)

    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    vol20_daily = tqqq_ret.rolling(20).std()
    
    # Handle annualized vs daily volatility interpretation
    if config.vol_annualized:
        # If threshold is interpreted as annualized, convert daily vol to annualized
        vol20 = vol20_daily * np.sqrt(TRADING_DAYS)
    else:
        vol20 = vol20_daily
    
    vol_lock = (vol20 >= config.vol_threshold) if config.use_vol_lock else pd.Series(False, index=prices.index)

    if config.use_overheat:
        overheat_state, overheat_ratio = compute_overheat_state(
            prices["TQQQ"],
            config.ma_long_tqqq,
            config.overheat_ratio,
            config.reentry_ratio,
        )
    else:
        overheat_state = pd.Series(False, index=prices.index)
        overheat_ratio = prices["TQQQ"] * 0.0

    forced_off = vol_lock | overheat_state

    weights = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    state_asset = "SGOV"
    prev_score = 0

    for dt in prices.index:
        s = int(score.loc[dt])
        if forced_off.loc[dt]:
            state_asset = "SGOV"
        else:
            if config.use_entry_gate:
                if state_asset == "SGOV":
                    if prev_score == 0 and s > 0:
                        state_asset = map_score_to_asset(s, config.use_qld_step)
                    else:
                        state_asset = "SGOV"
                else:
                    if s == 0:
                        state_asset = "SGOV"
                    else:
                        state_asset = map_score_to_asset(s, config.use_qld_step)
            else:
                state_asset = map_score_to_asset(s, config.use_qld_step)

        weights.loc[dt, state_asset] = 1.0
        prev_score = s

    return {
        "weights": weights,
        "score": score,
        "vol_lock": vol_lock,
        "overheat_state": overheat_state,
    }


# =============================================================================
# Backtest Engine
# =============================================================================

def calculate_metrics(equity: pd.Series, trades_df: pd.DataFrame, total_tax: float) -> dict:
    n_years = len(equity) / TRADING_DAYS
    final_val = float(equity.iloc[-1])

    cagr = (final_val ** (1.0 / n_years) - 1.0) if n_years > 0 and final_val > 0 else 0.0

    returns = equity.pct_change().fillna(0.0)
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    mdd = float(drawdown.min())

    daily_std = returns.std(ddof=0)
    sharpe = (returns.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0

    n_trades = len(trades_df) if len(trades_df) > 0 else 0

    return {
        "Final": final_val,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Trades": n_trades,
        "TotalTax": total_tax,
    }


def run_backtest(prices: pd.DataFrame, weights_df: pd.DataFrame, config: ExperimentConfig) -> dict:
    # Apply execution lag based on config
    if config.use_t0_execution:
        weights_exec = weights_df.copy()  # t+0: same day execution (look-ahead)
    else:
        weights_exec = weights_df.shift(1).fillna(0.0)  # t+1: next day execution

    mask = (prices.index >= START_DATE) & (prices.index <= END_DATE)
    prices = prices.loc[mask]
    weights_exec = weights_exec.reindex(prices.index).fillna(0.0)

    returns = prices[ASSETS].pct_change().fillna(0.0)
    turnover = weights_exec.diff().abs().sum(axis=1).fillna(0.0) * 0.5
    
    cost_rate = config.cost_bps / 10000.0
    cost_drag = turnover * cost_rate

    port_ret_gross = (weights_exec * returns).sum(axis=1) - cost_drag
    equity_pre = (1.0 + port_ret_gross).cumprod()

    # Tax calculation (if applicable)
    if config.apply_tax:
        equity_post = pd.Series(index=prices.index, dtype=float)
        portfolio_value = 1.0
        cost_basis = {asset: 0.0 for asset in ASSETS}
        shares = {asset: 0.0 for asset in ASSETS}
        yearly_gains: dict[int, float] = {}
        total_tax = 0.0
        trades: list[dict] = []
        prev_weights = pd.Series(0.0, index=ASSETS)

        for i, dt in enumerate(prices.index):
            portfolio_value *= (1.0 + float(port_ret_gross.loc[dt]))
            curr_weights = weights_exec.loc[dt]
            weight_delta = curr_weights - prev_weights
            year = dt.year
            
            if year not in yearly_gains:
                yearly_gains[year] = 0.0

            for asset in ASSETS:
                delta = float(weight_delta.loc[asset])
                if abs(delta) < 1e-12:
                    continue

                price = float(prices.loc[dt, asset])

                if delta < 0:
                    sold_value = abs(delta) * portfolio_value
                    sold_shares = sold_value / price if price > 0 else 0.0
                    if shares[asset] > 0 and cost_basis[asset] > 0:
                        avg_cost = cost_basis[asset] / shares[asset]
                        gain = sold_shares * (price - avg_cost)
                        yearly_gains[year] += gain
                        sell_ratio = min(1.0, sold_shares / shares[asset]) if shares[asset] > 0 else 0.0
                        cost_basis[asset] *= (1.0 - sell_ratio)
                        shares[asset] -= sold_shares

                    trades.append({"Date": dt, "asset": asset, "side": "SELL"})
                else:
                    buy_value = delta * portfolio_value
                    buy_shares = buy_value / price if price > 0 else 0.0
                    cost_basis[asset] += buy_value
                    shares[asset] += buy_shares
                    trades.append({"Date": dt, "asset": asset, "side": "BUY"})

            prev_weights = curr_weights

            next_year = int(pd.Timestamp(prices.index[i + 1]).year) if i < len(prices.index) - 1 else year
            is_year_end = (i == len(prices.index) - 1) or (next_year != year)
            if is_year_end:
                taxable = max(0.0, yearly_gains[year])
                tax = taxable * config.tax_rate
                portfolio_value -= tax
                total_tax += tax

            equity_post.loc[dt] = portfolio_value

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = calculate_metrics(equity_post, trades_df, total_tax)
        equity = equity_post
    else:
        # No tax: just use pre-tax equity
        equity = equity_pre
        trades_df = pd.DataFrame()
        metrics = calculate_metrics(equity, trades_df, 0.0)

    return {
        "config": config,
        "equity": equity,
        "metrics": metrics,
    }


# =============================================================================
# Reporting
# =============================================================================

def generate_leaderboard(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        c = r["config"]
        rows.append({
            "Experiment": c.name,
            "Description": c.description,
            "t+0": "Yes" if c.use_t0_execution else "No",
            "Cost": f"{c.cost_bps}bps",
            "Tax": "Yes" if c.apply_tax else "No",
            "Overheat": c.overheat_ratio,
            "AnnVol": "Yes" if c.vol_annualized else "No",
            "QLD": "Yes" if c.use_qld_step else "No",
            "CAGR": m["CAGR"],
            "MDD": m["MDD"],
            "Sharpe": m["Sharpe"],
            "Trades": m["Trades"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("CAGR", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


def write_leaderboard_md(df: pd.DataFrame) -> str:
    header = (
        "# 54% CAGR Reproduction Experiments\n\n"
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"**Period**: {START_DATE} ~ {END_DATE}\n"
        f"**Target**: 54.23% CAGR (claimed)\n\n"
    )

    table = (
        "| Rank | Experiment | t+0 | Cost | Tax | Overheat | AnnVol | QLD | CAGR | MDD | Sharpe |\n"
        "|:----:|:-----------|:---:|:----:|:---:|:--------:|:------:|:---:|-----:|----:|-------:|\n"
    )

    lines = []
    for _, row in df.iterrows():
        rank = int(row["Rank"])
        cagr_pct = row["CAGR"] * 100
        mdd_pct = row["MDD"] * 100
        
        # Highlight if close to 54%
        cagr_str = f"**{cagr_pct:.2f}%**" if cagr_pct > 50 else f"{cagr_pct:.2f}%"
        
        lines.append(
            f"| {rank} | {row['Experiment']} | {row['t+0']} | {row['Cost']} | "
            f"{row['Tax']} | {row['Overheat']} | {row['AnnVol']} | {row['QLD']} | "
            f"{cagr_str} | {mdd_pct:.2f}% | {row['Sharpe']:.2f} |"
        )

    return header + table + "\n".join(lines) + "\n"


def plot_equity_curves(results: list[dict], output_path: str) -> None:
    plt.figure(figsize=(14, 7))
    for r in results:
        plt.plot(r["equity"], label=r["config"].name, lw=1.2)
    plt.yscale("log")
    plt.title("Equity Curves - 54% CAGR Reproduction Experiments")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend(loc="upper left", fontsize=7, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def run_suite() -> tuple[list[dict], pd.DataFrame]:
    print("=" * 80)
    print("54% CAGR Reproduction Experiment Suite")
    print("=" * 80)
    print(f"Target: 54.23% CAGR (claimed by original strategy)")
    print()

    prices = load_data()
    print(f"Data loaded: {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}")
    print()
    
    results: list[dict] = []

    for i, config in enumerate(EXPERIMENTS):
        print(f"[{i + 1:02d}/{len(EXPERIMENTS):02d}] {config.name}")

        # Choose weight generation based on strategy type
        if config.strategy_type == "e03":
            data = build_e03_weights(prices, config.ensemble_windows)
        else:
            data = build_user_weights(prices, config)
        
        weights = data["weights"]
        result = run_backtest(prices, weights, config)
        results.append(result)

        m = result["metrics"]
        cagr_pct = m["CAGR"] * 100
        match_indicator = " <-- CLOSE TO 54%!" if cagr_pct > 50 else ""
        print(
            f"    CAGR: {cagr_pct:.2f}% | MDD: {m['MDD']*100:.2f}% | "
            f"Sharpe: {m['Sharpe']:.2f} | Trades: {m['Trades']}{match_indicator}"
        )

    # Generate outputs
    leaderboard = generate_leaderboard(results)
    leaderboard.to_csv(os.path.join(OUTPUT_DIR, "leaderboard.csv"), index=False)
    
    leaderboard_md = write_leaderboard_md(leaderboard)
    with open(os.path.join(OUTPUT_DIR, "leaderboard.md"), "w") as f:
        f.write(leaderboard_md)

    plot_equity_curves(results, os.path.join(OUTPUT_DIR, "equity_curves.png"))

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(leaderboard[["Rank", "Experiment", "CAGR", "MDD", "Sharpe"]].to_string(index=False))
    print()
    print(f"Saved outputs to: {OUTPUT_DIR}")
    
    # Check if any experiment matches 54%
    best_cagr = leaderboard["CAGR"].max() * 100
    print()
    print(f"Best CAGR achieved: {best_cagr:.2f}%")
    print(f"Target CAGR: 54.23%")
    print(f"Gap: {54.23 - best_cagr:.2f}%p")
    
    return results, leaderboard


if __name__ == "__main__":
    run_suite()
