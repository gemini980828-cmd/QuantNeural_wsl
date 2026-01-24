"""
CLI Runner for Backtest from CSV Artifacts.

Provides a single "run button" CLI that:
- Loads prices.csv + scores.csv using existing runner
- Runs the backtest deterministically
- Writes standard output artifacts to an output directory

Usage:
    python src/run_scores_backtest_cli.py \
        --prices_csv_path data/prices.csv \
        --scores_csv_path data/scores.csv \
        --out_dir results/

Design Principles:
- Deterministic: no randomness, no system clock, no network
- Fail-fast: propagate validation errors clearly
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

# Allow running via `python src/run_scores_backtest_cli.py` from repo root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.run_scores_backtest_from_csv import run_scores_backtest_from_csv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest from CSV artifacts and write output files."
    )
    
    # Required arguments
    parser.add_argument(
        "--prices_csv_path",
        type=str,
        required=True,
        help="Path to prices CSV (wide or long format)"
    )
    parser.add_argument(
        "--scores_csv_path",
        type=str,
        required=True,
        help="Path to scores CSV (wide format)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for artifacts"
    )
    
    # Optional arguments (pass-through to runner)
    parser.add_argument(
        "--price_col",
        type=str,
        default="close",
        help="Price column name (default: close)"
    )
    parser.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Date column name (default: date)"
    )
    parser.add_argument(
        "--ticker_col",
        type=str,
        default="ticker",
        help="Ticker column name for long format (default: ticker)"
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default="Q",
        choices=["M", "Q"],
        help="Rebalance frequency: M (monthly) or Q (quarterly) [default: Q]"
    )
    parser.add_argument(
        "--execution_lag_days",
        type=int,
        default=1,
        help="Execution lag in days (default: 1)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="topk",
        choices=["softmax", "softmax_topk", "rank", "topk"],
        help="Weight construction method [default: topk]"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax (default: 1.0)"
    )
    parser.add_argument(
        "--score_transform",
        type=str,
        default="none",
        choices=["none", "winsorize", "zscore", "winsorize_zscore"],
        help="Optional per-date transform before softmax to reduce outliers [default: none]"
    )
    parser.add_argument(
        "--winsorize_q_low",
        type=float,
        default=0.01,
        help="Winsorize lower quantile for score_transform (default: 0.01)"
    )
    parser.add_argument(
        "--winsorize_q_high",
        type=float,
        default=0.99,
        help="Winsorize upper quantile for score_transform (default: 0.99)"
    )
    parser.add_argument(
        "--zscore_eps",
        type=float,
        default=1e-12,
        help="Z-score std guard epsilon for score_transform (default: 1e-12)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=400,
        help="Top K assets for topk/rank/softmax_topk method [default: 400]"
    )
    parser.add_argument(
        "--max_weight",
        type=float,
        default=None,
        help="Maximum weight per asset (0, 1]"
    )
    parser.add_argument(
        "--cost_bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points [default: 10]"
    )
    parser.add_argument(
        "--slippage_bps",
        type=float,
        default=5.0,
        help="Slippage in basis points [default: 5]"
    )
    parser.add_argument(
        "--initial_equity",
        type=float,
        default=1.0,
        help="Initial equity (default: 1.0)"
    )
    parser.add_argument(
        "--max_gross_leverage",
        type=float,
        default=None,
        help="Maximum gross leverage"
    )
    
    return parser.parse_args(argv)


def write_artifacts(result: dict, args: argparse.Namespace) -> None:
    """Write backtest result artifacts to output directory."""
    out_dir = args.out_dir
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # 1) equity_curve.csv
    equity_curve = result["equity_curve"]
    equity_df = equity_curve.reset_index()
    equity_df.columns = ["date", "equity"]
    equity_df.to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)
    
    # 2) trades.csv
    trades = result["trades"]
    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    
    # 3) summary_metrics.json
    weights_stats = {}
    try:
        target_weights_df = result.get("target_weights")
        if isinstance(target_weights_df, pd.DataFrame) and not target_weights_df.empty:
            arr = target_weights_df.to_numpy(dtype=np.float64, copy=False)
            if arr.ndim == 2 and arr.shape[1] > 0:
                tol = 1e-12
                row_max = arr.max(axis=1)
                n_holdings = (arr > tol).sum(axis=1)

                denom = np.sum(arr * arr, axis=1)
                eff_n = np.where(denom > 0, 1.0 / denom, 0.0)

                top_n = min(10, arr.shape[1])
                topn = np.partition(arr, -top_n, axis=1)[:, -top_n:] if top_n > 0 else arr[:, :0]
                topn_share = topn.sum(axis=1) if top_n > 0 else np.zeros(arr.shape[0], dtype=np.float64)

                weights_stats = {
                    "max_weight_max": float(round(float(row_max.max()), 10)),
                    "max_weight_p95": float(round(float(np.quantile(row_max, 0.95)), 10)),
                    "max_weight_median": float(round(float(np.median(row_max)), 10)),
                    "n_holdings_median": int(np.median(n_holdings)),
                    "eff_n_median": float(round(float(np.median(eff_n)), 10)),
                    "top10_share_median": float(round(float(np.median(topn_share)), 10)),
                }
    except Exception:
        weights_stats = {}

    summary = {
        "metrics": result["metrics"],
        "weights_stats": weights_stats,
        "params": {
            "prices_csv_path": args.prices_csv_path,
            "scores_csv_path": args.scores_csv_path,
            "price_col": args.price_col,
            "date_col": args.date_col,
            "ticker_col": args.ticker_col,
            "rebalance": args.rebalance,
            "execution_lag_days": args.execution_lag_days,
            "method": args.method,
            "temperature": args.temperature,
            "score_transform": args.score_transform,
            "winsorize_q_low": args.winsorize_q_low,
            "winsorize_q_high": args.winsorize_q_high,
            "zscore_eps": args.zscore_eps,
            "top_k": args.top_k,
            "max_weight": args.max_weight,
            "cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
            "initial_equity": args.initial_equity,
            "max_gross_leverage": args.max_gross_leverage,
        },
        "n_rebalances": len(result["rebalance_dates"]),
        "n_trades": len(result["trades"]),
        "warnings": result.get("warnings", []),
    }
    with open(os.path.join(out_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    # 4) target_weights.csv
    target_weights = result["target_weights"]
    tw_df = target_weights.reset_index()
    tw_df = tw_df.rename(columns={tw_df.columns[0]: "date"})
    tw_df.to_csv(os.path.join(out_dir, "target_weights.csv"), index=False)
    
    # 5) weights_used.csv
    weights_used = result["weights_used"]
    wu_df = weights_used.reset_index()
    wu_df = wu_df.rename(columns={wu_df.columns[0]: "date"})
    wu_df.to_csv(os.path.join(out_dir, "weights_used.csv"), index=False)
    
    # 6) turnover.csv
    turnover = result["turnover"]
    to_df = turnover.reset_index()
    to_df.columns = ["date", "turnover"]
    to_df.to_csv(os.path.join(out_dir, "turnover.csv"), index=False)
    
    # 7) costs.csv
    costs = result["costs"]
    costs_df = costs.reset_index()
    costs_df.columns = ["date", "cost"]
    costs_df.to_csv(os.path.join(out_dir, "costs.csv"), index=False)
    
    # 8) returns.csv
    returns = result["daily_returns"]
    ret_df = returns.reset_index()
    ret_df.columns = ["date", "return"]
    ret_df.to_csv(os.path.join(out_dir, "returns.csv"), index=False)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.
    
    Parameters
    ----------
    argv : list, optional
        Command line arguments. If None, uses sys.argv[1:].
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)
    
    # Run backtest
    result = run_scores_backtest_from_csv(
        prices_csv_path=args.prices_csv_path,
        scores_csv_path=args.scores_csv_path,
        price_col=args.price_col,
        date_col=args.date_col,
        ticker_col=args.ticker_col,
        rebalance=args.rebalance,
        execution_lag_days=args.execution_lag_days,
        method=args.method,
        temperature=args.temperature,
        score_transform=args.score_transform,
        winsorize_q_low=args.winsorize_q_low,
        winsorize_q_high=args.winsorize_q_high,
        zscore_eps=args.zscore_eps,
        top_k=args.top_k,
        max_weight=args.max_weight,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        initial_equity=args.initial_equity,
        max_gross_leverage=args.max_gross_leverage,
    )
    
    # Write artifacts
    write_artifacts(result, args)
    
    # Print summary
    print(f"Backtest completed successfully.")
    print(f"Output directory: {args.out_dir}")
    print(f"Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.6f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
