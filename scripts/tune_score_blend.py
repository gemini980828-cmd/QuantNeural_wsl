"""
Grid-search blend weights between two score panels and backtest each blend.

This is a research utility to answer:
  "Does an ML score add incremental value when blended with a robust baseline?"

Blend definition (per score date t)
---------------------------------
1) Cross-sectional z-score each panel across tickers (finite-only):
     z_base(t, i), z_var(t, i)
2) Blend:
     s_blend(t, i) = (1-w) * z_base(t, i) + w * z_var(t, i)

Then run the standard E2E backtest harness on the blended scores.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow running this script directly via `python scripts/...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.e2e_backtest import run_scores_backtest  # noqa: E402


def _normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.endswith(".US"):
        t = t[: -len(".US")]
    return t


def _read_wide_panel_csv(path: str, *, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"CSV missing date column '{date_col}': {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    if not df.index.is_unique:
        raise ValueError(f"CSV has duplicate dates: {path}")
    df.columns = [_normalize_ticker(c) for c in df.columns]
    df = df.loc[:, df.columns.notna()]
    return df


def _cs_zscore(df: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    out = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        m = np.isfinite(row)
        if int(m.sum()) < 2:
            out[i] = row
            continue
        mu = float(np.mean(row[m]))
        sd = float(np.std(row[m]))
        if not np.isfinite(mu) or not np.isfinite(sd) or sd < eps:
            out[i] = row - mu
            continue
        out[i] = (row - mu) / sd
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid-search blended score weights and backtest.")
    p.add_argument("--prices_csv_path", required=True)
    p.add_argument("--base_scores_csv_path", required=True)
    p.add_argument("--variant_scores_csv_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--date_col", default="date")
    p.add_argument("--price_col", default="close")

    p.add_argument("--rebalance", choices=["M", "Q"], default="Q")
    p.add_argument("--execution_lag_days", type=int, default=1)
    p.add_argument("--method", choices=["topk", "rank", "softmax", "softmax_topk"], default="topk")
    p.add_argument("--top_k", type=int, default=400)
    p.add_argument("--cost_bps", type=float, default=0.0)
    p.add_argument("--slippage_bps", type=float, default=0.0)
    p.add_argument("--zscore_eps", type=float, default=1e-12)

    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Blend weights w in [0,1] (w=0 => base only, w=1 => variant only).",
    )
    p.add_argument("--export_best_scores", action="store_true", help="Write best blended scores.csv to out_dir.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    prices = _read_wide_panel_csv(args.prices_csv_path, date_col=args.date_col)
    base = _read_wide_panel_csv(args.base_scores_csv_path, date_col=args.date_col)
    var = _read_wide_panel_csv(args.variant_scores_csv_path, date_col=args.date_col)

    # Strict intersection for fair comparison
    common_dates = base.index.intersection(var.index)
    if len(common_dates) < 2:
        raise SystemExit("No common score dates between base and variant panels.")

    common_tickers = sorted(set(base.columns) & set(var.columns) & set(prices.columns))
    if len(common_tickers) < 10:
        raise SystemExit(f"Too few common tickers after intersection: {len(common_tickers)}")

    base = base.loc[common_dates, common_tickers].astype(float)
    var = var.loc[common_dates, common_tickers].astype(float)
    prices = prices[common_tickers].astype(float)

    if base.isna().any().any() or var.isna().any().any():
        raise SystemExit("Scores contain NaN after intersection (expected finite, gated scores).")

    base_z = _cs_zscore(base, eps=float(args.zscore_eps))
    var_z = _cs_zscore(var, eps=float(args.zscore_eps))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    best_w: float | None = None
    best_sharpe: float | None = None
    best_scores: pd.DataFrame | None = None

    for w in args.weights:
        w = float(w)
        if not (0.0 <= w <= 1.0):
            raise SystemExit(f"Invalid weight (must be in [0,1]): {w}")
        scores_blend = (1.0 - w) * base_z + w * var_z

        result = run_scores_backtest(
            prices,
            scores_blend,
            price_col=args.price_col,
            rebalance=args.rebalance,
            execution_lag_days=int(args.execution_lag_days),
            method=args.method,
            top_k=int(args.top_k),
            cost_bps=float(args.cost_bps),
            slippage_bps=float(args.slippage_bps),
        )
        m = result["metrics"]
        sharpe = float(m.get("sharpe", np.nan))
        rows.append(
            {
                "w": w,
                "sharpe": sharpe,
                "cagr": float(m.get("cagr", np.nan)),
                "ann_vol": float(m.get("ann_vol", np.nan)),
                "max_drawdown": float(m.get("max_drawdown", np.nan)),
                "total_turnover": float(m.get("total_turnover", np.nan)),
            }
        )

        if np.isfinite(sharpe) and (best_sharpe is None or sharpe > best_sharpe):
            best_sharpe = sharpe
            best_w = w
            best_scores = scores_blend

    summary = pd.DataFrame(rows).sort_values("w", kind="mergesort").reset_index(drop=True)
    summary.to_csv(out_dir / "blend_grid_summary.csv", index=False)

    if args.export_best_scores and best_scores is not None and best_w is not None:
        out = best_scores.reset_index().rename(columns={"index": args.date_col})
        out.to_csv(out_dir / "scores_blend_best.csv", index=False)

    print(f"Wrote: {out_dir / 'blend_grid_summary.csv'}")
    if best_w is not None and best_sharpe is not None:
        print(f"Best w={best_w:.3f} sharpe={best_sharpe:.6f}")
        if args.export_best_scores:
            print(f"Wrote: {out_dir / 'scores_blend_best.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

