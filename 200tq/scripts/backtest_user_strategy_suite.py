# -*- coding: utf-8 -*-
"""
User Strategy vs E03 Backtest Suite
===================================

Requirements:
- Signal at t -> Position at t+1 (1-day lag)
- 10 bps transaction cost on turnover
- 22% tax on annual realized gains (year-end settlement)
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
import warnings

warnings.filterwarnings("ignore")


START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
TRADING_DAYS = 252

COST_BPS = 10
COST_RATE = COST_BPS / 10000.0
TAX_RATE = 0.22

SHORT_MA = 3

OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/experiments/user_strategy_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSETS = ["TQQQ", "QLD", "SGOV"]


@dataclass
class ExperimentConfig:
    name: str
    description: str
    strategy_type: str
    ensemble_windows: list[int] = field(default_factory=list)
    ma_long_qqq: int = 161
    ma_long_tqqq: int = 200
    use_entry_gate: bool = False
    use_vol_lock: bool = True
    vol_threshold: float = 0.062
    use_overheat: bool = True
    overheat_ratio: float = 2.51
    reentry_ratio: float = 2.18
    use_qld_step: bool = True


EXPERIMENTS = [
    ExperimentConfig(
        name="E00_E03_Ensemble_Baseline",
        description="E03 ensemble, OFF=100% SGOV",
        strategy_type="e03",
        ensemble_windows=[160, 165, 170],
    ),
    ExperimentConfig(
        name="E01_User_Relaxed_Gate",
        description="User strategy, relaxed entry gate",
        strategy_type="user",
        use_entry_gate=False,
    ),
    ExperimentConfig(
        name="E02_User_Strict_Gate",
        description="User strategy, strict entry gate",
        strategy_type="user",
        use_entry_gate=True,
    ),
    ExperimentConfig(
        name="E03_User_No_VolLock",
        description="User strategy without volatility lock",
        strategy_type="user",
        use_vol_lock=False,
    ),
    ExperimentConfig(
        name="E04_User_No_Overheat",
        description="User strategy without overheat exit",
        strategy_type="user",
        use_overheat=False,
    ),
    ExperimentConfig(
        name="E05_User_No_QLD_Step",
        description="User strategy, score 1 maps to TQQQ",
        strategy_type="user",
        use_qld_step=False,
    ),
    ExperimentConfig(
        name="E06a_Vol_Sensitivity_5.0",
        description="Vol lock threshold 0.050",
        strategy_type="user",
        vol_threshold=0.050,
    ),
    ExperimentConfig(
        name="E06b_Vol_Sensitivity_6.2",
        description="Vol lock threshold 0.062 (baseline)",
        strategy_type="user",
        vol_threshold=0.062,
    ),
    ExperimentConfig(
        name="E06c_Vol_Sensitivity_7.5",
        description="Vol lock threshold 0.075",
        strategy_type="user",
        vol_threshold=0.075,
    ),
    ExperimentConfig(
        name="E07a_Overheat_2.30",
        description="Overheat 2.30, reentry 2.00",
        strategy_type="user",
        overheat_ratio=2.30,
        reentry_ratio=2.00,
    ),
    ExperimentConfig(
        name="E07b_Overheat_2.51",
        description="Overheat 2.51, reentry 2.18 (baseline)",
        strategy_type="user",
        overheat_ratio=2.51,
        reentry_ratio=2.18,
    ),
    ExperimentConfig(
        name="E07c_Overheat_2.70",
        description="Overheat 2.70, reentry 2.35",
        strategy_type="user",
        overheat_ratio=2.70,
        reentry_ratio=2.35,
    ),
    ExperimentConfig(
        name="E08a_MA161_Sensitivity_160",
        description="QQQ MA long 160",
        strategy_type="user",
        ma_long_qqq=160,
    ),
    ExperimentConfig(
        name="E08b_MA161_Sensitivity_161",
        description="QQQ MA long 161 (baseline)",
        strategy_type="user",
        ma_long_qqq=161,
    ),
    ExperimentConfig(
        name="E08c_MA161_Sensitivity_165",
        description="QQQ MA long 165",
        strategy_type="user",
        ma_long_qqq=165,
    ),
    ExperimentConfig(
        name="E09a_TQQQ_MA200_Sens_180",
        description="TQQQ MA long 180",
        strategy_type="user",
        ma_long_tqqq=180,
    ),
    ExperimentConfig(
        name="E09b_TQQQ_MA200_Sens_200",
        description="TQQQ MA long 200 (baseline)",
        strategy_type="user",
        ma_long_tqqq=200,
    ),
    ExperimentConfig(
        name="E09c_TQQQ_MA200_Sens_220",
        description="TQQQ MA long 220",
        strategy_type="user",
        ma_long_tqqq=220,
    ),
    ExperimentConfig(
        name="E10a_E03_Ensemble_155_160_165",
        description="E03 ensemble windows 155/160/165",
        strategy_type="e03",
        ensemble_windows=[155, 160, 165],
    ),
    ExperimentConfig(
        name="E10b_E03_Ensemble_160_165_170",
        description="E03 ensemble windows 160/165/170",
        strategy_type="e03",
        ensemble_windows=[160, 165, 170],
    ),
]


def build_synthetic_leverage(base, leverage):
    r = base.pct_change().fillna(0.0)
    r_lev = leverage * r
    return 100.0 * (1.0 + r_lev).cumprod()  # type: ignore[reportAttributeAccessIssue]


def splice_series(synth, actual):
    if actual is None or actual.isna().all():
        return synth
    actual = actual.reindex(synth.index)
    first = actual.first_valid_index()
    if first is None:
        return synth
    if first > synth.index[0]:  # type: ignore[reportOperatorIssue]
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
    import yfinance as yf  # type: ignore[reportMissingImports]

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
    if shv is not None:
        df["SHV"] = shv.reindex(base_idx).ffill()

    df = df.dropna(subset=["QQQ", "TQQQ", "QLD", "SGOV"])
    return df


def generate_ensemble_signal(prices: pd.DataFrame, windows: list[int]) -> pd.Series:
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    votes = pd.DataFrame(index=prices.index)
    for window in windows:
        ma_long = qqq.rolling(window).mean()
        votes[f"ma{window}"] = (ma_short > ma_long).astype(int)
    threshold = len(windows) // 2 + 1
    return (votes.sum(axis=1) >= threshold).astype(int)


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


def compute_overheat_state(
    tqqq,
    ma_long,
    overheat_ratio,
    reentry_ratio,
):
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
    vol20 = tqqq_ret.rolling(20).std()
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
        "overheat_ratio": overheat_ratio,
    }


def build_e03_weights(prices: pd.DataFrame, windows: list[int]):
    signal = generate_ensemble_signal(prices, windows)
    weights = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    weights.loc[signal == 1, "TQQQ"] = 1.0
    weights.loc[signal == 0, "SGOV"] = 1.0
    return {"weights": weights, "signal": signal}


def calculate_metrics(
    equity_post: pd.Series,
    equity_pre: pd.Series,
    trades: pd.DataFrame,
    total_tax_paid: float,
) -> dict:
    n_years = len(equity_post) / TRADING_DAYS
    final_post = float(equity_post.iloc[-1])
    final_pre = float(equity_pre.iloc[-1])

    cagr_post = (final_post ** (1.0 / n_years) - 1.0) if n_years > 0 and final_post > 0 else 0.0
    cagr_pre = (final_pre ** (1.0 / n_years) - 1.0) if n_years > 0 and final_pre > 0 else 0.0

    returns = equity_post.pct_change().fillna(0.0)
    peak = equity_post.cummax()
    drawdown = equity_post / peak - 1.0
    mdd = float(drawdown.min())

    daily_std = returns.std(ddof=0)
    sharpe = (returns.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0.0
    sortino = (cagr_post / downside_std) if downside_std > 0 else 0.0

    calmar = (cagr_post / abs(mdd)) if mdd != 0 else 0.0

    n_trades = len(trades) if len(trades) > 0 else 0

    turnover_per_year = 0.0
    if n_years > 0 and len(trades) > 0:
        turnover_per_year = trades["notional"].sum() / n_years

    return {
        "FinalPostTax": final_post,
        "FinalPreTax": final_pre,
        "CAGR_PostTax": cagr_post,
        "CAGR_PreTax": cagr_pre,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Trades": n_trades,
        "TurnoverPerYear": turnover_per_year,
        "TotalTaxPaid": total_tax_paid,
        "NumDays": len(equity_post),
    }


def run_backtest(prices: pd.DataFrame, weights_df: pd.DataFrame, config: ExperimentConfig) -> dict:
    weights_exec = weights_df.shift(1).fillna(0.0)

    mask = (prices.index >= START_DATE) & (prices.index <= END_DATE)
    prices = prices.loc[mask]
    weights_exec = weights_exec.reindex(prices.index).fillna(0.0)

    returns = prices[ASSETS].pct_change().fillna(0.0)
    turnover = weights_exec.diff().abs().sum(axis=1).fillna(0.0) * 0.5
    cost_drag = turnover * COST_RATE

    port_ret_gross = (weights_exec * returns).sum(axis=1) - cost_drag
    equity_pre = (1.0 + port_ret_gross).cumprod()

    equity_post = pd.Series(index=prices.index, dtype=float)
    portfolio_value = 1.0
    cost_basis = {asset: 0.0 for asset in ASSETS}
    shares = {asset: 0.0 for asset in ASSETS}
    yearly_gains: dict[int, float] = {}
    tax_by_year: dict[int, float] = {}
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
            notional = abs(delta) * portfolio_value
            trade_cost = abs(delta) * portfolio_value * COST_RATE * 0.5

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
                else:
                    gain = 0.0

                trades.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "asset": asset,
                    "side": "SELL",
                    "notional": sold_value,
                    "cost": trade_cost,
                    "realized_gain": gain,
                })
            else:
                buy_value = delta * portfolio_value
                buy_shares = buy_value / price if price > 0 else 0.0
                cost_basis[asset] += buy_value
                shares[asset] += buy_shares
                trades.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "asset": asset,
                    "side": "BUY",
                    "notional": buy_value,
                    "cost": trade_cost,
                    "realized_gain": 0.0,
                })

        prev_weights = curr_weights

        next_year = int(pd.Timestamp(prices.index[i + 1]).year) if i < len(prices.index) - 1 else year
        is_year_end = (i == len(prices.index) - 1) or (next_year != year)
        if is_year_end:
            taxable = max(0.0, yearly_gains[year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            tax_by_year[year] = tax

        equity_post.loc[dt] = portfolio_value

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Date", "asset", "side", "notional", "cost", "realized_gain"]
    )

    metrics = calculate_metrics(equity_post, equity_pre, trades_df, sum(tax_by_year.values()))

    return {
        "config": config,
        "equity": pd.DataFrame({"equity": equity_post, "equity_pre_tax": equity_pre}),
        "trades": trades_df,
        "metrics": metrics,
        "tax_by_year": tax_by_year,
        "weights": weights_exec,
    }


def save_experiment_artifacts(result: dict, exp_dir: str) -> None:
    os.makedirs(exp_dir, exist_ok=True)

    result["equity"].to_csv(os.path.join(exp_dir, "equity_curve.csv"))
    result["trades"].to_csv(os.path.join(exp_dir, "trades.csv"), index=False)

    metrics = result["metrics"].copy()
    metrics["experiment"] = result["config"].name
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def generate_leaderboard(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Experiment": r["config"].name,
            "Description": r["config"].description,
            "CAGR_PostTax": m["CAGR_PostTax"],
            "CAGR_PreTax": m["CAGR_PreTax"],
            "MDD": m["MDD"],
            "Sharpe": m["Sharpe"],
            "Sortino": m["Sortino"],
            "Calmar": m["Calmar"],
            "FinalPostTax": m["FinalPostTax"],
            "Trades": m["Trades"],
            "TurnoverPerYear": m["TurnoverPerYear"],
            "TotalTaxPaid": m["TotalTaxPaid"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("CAGR_PostTax", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


def write_leaderboard_md(df: pd.DataFrame) -> str:
    header = (
        "# Leaderboard: User Strategy vs E03\n\n"
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"**Period**: {START_DATE} ~ {END_DATE}\n"
        f"**Cost**: {COST_BPS} bps | **Tax**: {TAX_RATE*100:.0f}%\n\n"
    )

    table = (
        "| Rank | Experiment | CAGR (Post) | CAGR (Pre) | MDD | Sharpe | Calmar | Total Tax |\n"
        "|:----:|:-----------|-----------:|----------:|----:|-------:|-------:|---------:|\n"
    )

    lines = []
    for _, row in df.iterrows():
        rank = int(row["Rank"])  # type: ignore[reportArgumentType]
        lines.append(
            f"| {rank} | {row['Experiment']} | {row['CAGR_PostTax']*100:.2f}% | "
            f"{row['CAGR_PreTax']*100:.2f}% | {row['MDD']*100:.2f}% | {row['Sharpe']:.2f} | "
            f"{row['Calmar']:.2f} | {row['TotalTaxPaid']:.2f} |"
        )

    return header + table + "\n".join(lines) + "\n"


def plot_equity_curves(results: list[dict], output_path: str) -> None:
    plt.figure(figsize=(14, 7))
    for r in results:
        plt.plot(r["equity"]["equity"], label=r["config"].name, lw=1.2)
    plt.yscale("log")
    plt.title("Equity Curves (Post-Tax, Log Scale)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdowns(results: list[dict], output_path: str) -> None:
    plt.figure(figsize=(14, 6))
    for r in results:
        equity = r["equity"]["equity"]
        peak = equity.cummax()
        dd = (equity / peak - 1.0) * 100
        plt.plot(dd, label=r["config"].name, lw=1.1)
    plt.title("Drawdowns (Post-Tax)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown %")
    plt.legend(loc="lower left", fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_yearly_heatmap(yearly_df: pd.DataFrame, output_path: str) -> None:
    if yearly_df.empty:
        return

    data = yearly_df.T
    years = list(data.columns)
    experiments = list(data.index)
    values = data.values.astype(float)
    values = np.nan_to_num(values, nan=0.0)
    max_abs = float(np.nanmax(np.abs(values))) if values.size > 0 else 0.1
    vmax = max(0.1, max_abs)

    fig, ax = plt.subplots(figsize=(max(10, len(years) * 0.6), max(6, len(experiments) * 0.35)))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments)
    ax.set_title("Yearly Returns Heatmap (Post-Tax)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Return")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_sensitivity_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        cfg = r["config"]
        m = r["metrics"]
        rows.append({
            "Experiment": cfg.name,
            "Strategy": cfg.strategy_type,
            "EnsembleWindows": "-".join(str(x) for x in cfg.ensemble_windows) if cfg.ensemble_windows else "",
            "MA_Long_QQQ": cfg.ma_long_qqq,
            "MA_Long_TQQQ": cfg.ma_long_tqqq,
            "UseEntryGate": cfg.use_entry_gate,
            "UseVolLock": cfg.use_vol_lock,
            "VolThreshold": cfg.vol_threshold,
            "UseOverheat": cfg.use_overheat,
            "OverheatRatio": cfg.overheat_ratio,
            "ReentryRatio": cfg.reentry_ratio,
            "UseQLDStep": cfg.use_qld_step,
            "CAGR_PostTax": m["CAGR_PostTax"],
            "MDD": m["MDD"],
            "Sharpe": m["Sharpe"],
        })
    return pd.DataFrame(rows)


def run_suite() -> tuple[list[dict], pd.DataFrame]:
    print("=" * 80)
    print("User Strategy vs E03 Backtest Suite")
    print("=" * 80)

    prices = load_data()
    results: list[dict] = []
    yearly_returns = {}
    tax_rows = []

    for i, config in enumerate(EXPERIMENTS):
        print(f"[{i + 1:02d}/{len(EXPERIMENTS):02d}] {config.name}")

        if config.strategy_type == "e03":
            data = build_e03_weights(prices, config.ensemble_windows)
            weights = data["weights"]
        else:
            data = build_user_weights(prices, config)
            weights = data["weights"]

        result = run_backtest(prices, weights, config)
        results.append(result)

        equity = result["equity"]["equity"]
        yearly = equity.resample("YE").last().pct_change().dropna()
        yearly.index = yearly.index.year
        yearly_returns[config.name] = yearly

        for year, tax in result["tax_by_year"].items():
            tax_rows.append({"Experiment": config.name, "Year": year, "Tax": tax})

        exp_dir = os.path.join(OUTPUT_DIR, config.name)
        save_experiment_artifacts(result, exp_dir)

        m = result["metrics"]
        print(
            f"    Final: {m['FinalPostTax']:.2f}x | CAGR: {m['CAGR_PostTax']*100:.2f}% | "
            f"MDD: {m['MDD']*100:.2f}% | Sharpe: {m['Sharpe']:.2f}"
        )

    leaderboard = generate_leaderboard(results)
    leaderboard.to_csv(os.path.join(OUTPUT_DIR, "leaderboard.csv"), index=False)
    leaderboard_md = write_leaderboard_md(leaderboard)
    with open(os.path.join(OUTPUT_DIR, "leaderboard.md"), "w") as f:
        f.write(leaderboard_md)

    plot_equity_curves(results, os.path.join(OUTPUT_DIR, "equity_curves.png"))
    plot_drawdowns(results, os.path.join(OUTPUT_DIR, "drawdowns.png"))

    yearly_df = pd.DataFrame(yearly_returns)
    plot_yearly_heatmap(yearly_df, os.path.join(OUTPUT_DIR, "yearly_returns_heatmap.png"))

    sensitivity_df = build_sensitivity_table(results)
    sensitivity_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_table.csv"), index=False)

    tax_df = pd.DataFrame(tax_rows)
    tax_df.to_csv(os.path.join(OUTPUT_DIR, "tax_by_year.csv"), index=False)

    print("=" * 80)
    print(f"Saved outputs to: {OUTPUT_DIR}")
    return results, leaderboard


if __name__ == "__main__":
    run_suite()
