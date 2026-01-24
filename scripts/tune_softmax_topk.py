"""
Grid search tuner for softmax_topk sizing.

Purpose:
- Keep your alpha (scores) fixed
- Tune the score->weights conversion layer (temperature/max_weight/transform)
  to reduce concentration/turnover while preserving performance.

This script loads prices.csv + scores.csv ONCE, then runs multiple backtests
in-process for faster iteration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.e2e_backtest import run_scores_backtest


def _load_wide_panel(path: str, *, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df


def _weights_stats(weights: pd.DataFrame) -> dict:
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return {}

    arr = weights.to_numpy(dtype=np.float64, copy=False)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return {}

    tol = 1e-12
    row_max = arr.max(axis=1)
    n_holdings = (arr > tol).sum(axis=1)

    denom = np.sum(arr * arr, axis=1)
    eff_n = np.where(denom > 0, 1.0 / denom, 0.0)

    top_n = min(10, arr.shape[1])
    topn = np.partition(arr, -top_n, axis=1)[:, -top_n:] if top_n > 0 else arr[:, :0]
    topn_share = topn.sum(axis=1) if top_n > 0 else np.zeros(arr.shape[0], dtype=np.float64)

    return {
        "max_weight_max": float(round(float(row_max.max()), 10)),
        "max_weight_p95": float(round(float(np.quantile(row_max, 0.95)), 10)),
        "max_weight_median": float(round(float(np.median(row_max)), 10)),
        "n_holdings_median": int(np.median(n_holdings)),
        "eff_n_median": float(round(float(np.median(eff_n)), 10)),
        "top10_share_median": float(round(float(np.median(topn_share)), 10)),
    }


@dataclass(frozen=True)
class GridConfig:
    temperatures: list[float]
    max_weights: list[float | None]


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    vals = []
    for part in _parse_csv_list(s):
        vals.append(float(part))
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune softmax_topk sizing via grid search.")

    p.add_argument("--prices_csv_path", type=str, required=True)
    p.add_argument("--scores_csv_path", type=str, required=True)
    p.add_argument("--date_col", type=str, default="date")

    p.add_argument("--rebalance", type=str, default="Q", choices=["M", "Q"])
    p.add_argument("--execution_lag_days", type=int, default=1)
    p.add_argument("--cost_bps", type=float, default=10.0)
    p.add_argument("--slippage_bps", type=float, default=5.0)

    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--score_transform", type=str, default="winsorize_zscore",
                   choices=["none", "winsorize", "zscore", "winsorize_zscore"])
    p.add_argument("--winsorize_q_low", type=float, default=0.01)
    p.add_argument("--winsorize_q_high", type=float, default=0.99)
    p.add_argument("--zscore_eps", type=float, default=1e-12)

    p.add_argument(
        "--temperatures",
        type=str,
        default="0.25,0.5,1,2,4",
        help="Comma-separated temperature grid (default: 0.25,0.5,1,2,4)",
    )
    p.add_argument(
        "--max_weights",
        type=str,
        default="0.02,0.01,0.005",
        help="Comma-separated max_weight grid; use 'none' to include no cap (default: 0.02,0.01,0.005)",
    )
    p.add_argument("--out_csv", type=str, default=None, help="Optional output CSV path")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    prices = _load_wide_panel(args.prices_csv_path, date_col=args.date_col)
    scores = _load_wide_panel(args.scores_csv_path, date_col=args.date_col)

    temps = _parse_float_list(args.temperatures)

    max_weights = []
    for part in _parse_csv_list(args.max_weights):
        if part.lower() in {"none", "null"}:
            max_weights.append(None)
        else:
            max_weights.append(float(part))

    grid = GridConfig(temperatures=temps, max_weights=max_weights)

    rows: list[dict] = []

    for t in grid.temperatures:
        for mw in grid.max_weights:
            res = run_scores_backtest(
                prices,
                scores,
                rebalance=args.rebalance,
                execution_lag_days=args.execution_lag_days,
                method="softmax_topk",
                temperature=float(t),
                top_k=int(args.top_k),
                max_weight=mw,
                score_transform=args.score_transform,
                winsorize_q_low=args.winsorize_q_low,
                winsorize_q_high=args.winsorize_q_high,
                zscore_eps=args.zscore_eps,
                cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
            )

            m = res["metrics"]
            ws = _weights_stats(res["target_weights"])
            rows.append({
                "temperature": float(t),
                "max_weight": (None if mw is None else float(mw)),
                "cagr": float(m.get("cagr", np.nan)),
                "ann_vol": float(m.get("ann_vol", np.nan)),
                "cagr_over_vol": float(m.get("cagr_over_vol", m.get("sharpe", np.nan))),
                "max_drawdown": float(m.get("max_drawdown", np.nan)),
                "total_turnover": float(m.get("total_turnover", np.nan)),
                "total_cost": float(m.get("total_cost", np.nan)),
                "n_trades": int(len(res.get("trades", []))),
                **ws,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["cagr_over_vol", "cagr"], ascending=[False, False]).reset_index(drop=True)

    with pd.option_context("display.max_columns", 50, "display.width", 140):
        print(df.head(30).to_string(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

