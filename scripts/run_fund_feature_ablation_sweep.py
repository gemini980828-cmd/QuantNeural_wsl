"""
Task 10.2.11: Fund Feature Ablation Stability Sweep

Runs the ablation suite across multiple random seeds and time windows
to assess statistical significance and stability of any observed lift.

Produces:
- sweep_summary.json (schema_version: "10.2.11")
- sweep_summary.csv (flat table)
- REPORT_10_2_11.md (human-readable verdict)
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_fund_feature_ablation_suite import run_fund_feature_ablation_suite

# Feature modes from ablation suite
FEATURE_MODES = ["fund_full", "fund_zeroed", "fund_shuffled", "tech_only"]

# Default windows for sweep
DEFAULT_WINDOWS = [
    {"id": "W1_long", "train_end": "2014-12-31", "val_end": "2016-12-31"},
    {"id": "W2_mid", "train_end": "2017-12-31", "val_end": "2019-12-31"},
    {"id": "W3_recent", "train_end": "2019-12-31", "val_end": "2021-12-31"},
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def run_fund_feature_ablation_sweep(
    *,
    fund_alpha_dataset_path: str,
    prices_csv_path: str,
    baseline_scores_csv_path: str,
    out_dir: str,
    as_of_date: str,
    windows: list[dict] | None = None,
    seeds: list[int] | None = None,
    subset_mode: Literal["all", "sec_covered", "sec_missing", "split"] = "split",
    rebalance: str = "Q",
    target_col: str = "fwd_ret_21d",
    top_k: int = 400,
    invert_scores: bool = False,
    min_train_samples: int = 500,
    fund_dataset_path: str | None = None,
) -> dict:
    """
    Runs Task 10.2.10 ablation suite across multiple (train_end, val_end) windows and seeds.
    
    Reuses run_fund_feature_ablation_suite(...) for each run.
    
    Produces:
      - sweep_summary.json (schema_version: "10.2.11")
      - sweep_summary.csv  (flat table)
      - REPORT_10_2_11.md  (human-readable)
    
    Returns a dict with paths to these artifacts.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if fund_dataset_path is None:
        fund_dataset_path = fund_alpha_dataset_path
    
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Task 10.2.11: Fund Feature Ablation Stability Sweep")
    print("=" * 70)
    print(f"Windows: {len(windows)}, Seeds: {len(seeds)}")
    print(f"Total runs: {len(windows) * len(seeds)}")
    print("=" * 70)
    
    # Collect results
    rows: list[dict] = []
    warnings_list: list[str] = []
    
    for w_idx, window in enumerate(windows):
        window_id = window.get("id", f"W{w_idx}")
        train_end = window["train_end"]
        val_end = window["val_end"]
        
        for seed in seeds:
            run_id = f"{window_id}_seed{seed}"
            run_dir = out_path / run_id
            
            print(f"\n[{run_id}] Running ablation suite...")
            
            try:
                result = run_fund_feature_ablation_suite(
                    fund_alpha_dataset_path=fund_alpha_dataset_path,
                    prices_csv_path=prices_csv_path,
                    baseline_scores_csv_path=baseline_scores_csv_path,
                    out_dir=str(run_dir),
                    as_of_date=as_of_date,
                    train_end=train_end,
                    val_end=val_end,
                    rebalance=rebalance,
                    target_col=target_col,
                    top_k=top_k,
                    seed=seed,
                    invert_scores=invert_scores,
                    subset_mode=subset_mode,
                    fund_dataset_path=fund_dataset_path,
                    min_train_samples=min_train_samples,
                )
                
                # Extract per-mode metrics
                modes_data = result.get("modes", {})
                for mode in FEATURE_MODES:
                    mode_data = modes_data.get(mode, {})
                    row = {
                        "window_id": window_id,
                        "train_end": train_end,
                        "val_end": val_end,
                        "seed": seed,
                        "mode": mode,
                        "ic_mean": mode_data.get("ic_mean"),
                        "delta_all": mode_data.get("delta_cagr_vol_all"),
                        "delta_sec_covered": mode_data.get("delta_cagr_vol_sec_covered"),
                        "delta_sec_missing": mode_data.get("delta_cagr_vol_sec_missing"),
                        "run_ok": mode_data.get("training_success", False),
                        "warnings": "; ".join(mode_data.get("warnings", [])[:2])[:50],
                    }
                    rows.append(row)
                
                print(f"[{run_id}] Completed successfully")
                
            except Exception as e:
                error_msg = f"SWEEP_RUN_FAILED:{run_id}:{str(e)[:50]}"
                warnings_list.append(error_msg)
                print(f"[{run_id}] FAILED: {e}")
                
                # Record failed rows
                for mode in FEATURE_MODES:
                    rows.append({
                        "window_id": window_id,
                        "train_end": train_end,
                        "val_end": val_end,
                        "seed": seed,
                        "mode": mode,
                        "ic_mean": None,
                        "delta_all": None,
                        "delta_sec_covered": None,
                        "delta_sec_missing": None,
                        "run_ok": False,
                        "warnings": error_msg[:50],
                    })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Compute aggregated metrics
    aggregated = _compute_aggregated_metrics(df)
    
    # Write sweep_summary.csv
    csv_path = out_path / "sweep_summary.csv"
    df.to_csv(csv_path, index=False, lineterminator="\n")
    
    # Write sweep_summary.json
    json_path = out_path / "sweep_summary.json"
    summary = {
        "schema_version": "10.2.11",
        "config": {
            "fund_alpha_dataset_path": fund_alpha_dataset_path,
            "prices_csv_path": prices_csv_path,
            "baseline_scores_csv_path": baseline_scores_csv_path,
            "as_of_date": as_of_date,
            "windows": windows,
            "seeds": seeds,
            "subset_mode": subset_mode,
            "rebalance": rebalance,
            "target_col": target_col,
            "top_k": top_k,
        },
        "rows": rows,
        "aggregated_metrics": aggregated,
        "warnings": warnings_list,
    }
    
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    # Generate report
    report_path = out_path / "REPORT_10_2_11.md"
    _generate_sweep_report(df, aggregated, report_path, len(windows), len(seeds))
    
    print(f"\n{'=' * 70}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Report: {report_path}")
    print(f"{'=' * 70}")
    
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "report_path": str(report_path),
        "aggregated_metrics": aggregated,
    }


def _compute_aggregated_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregated metrics from sweep results."""
    aggregated: dict = {
        "per_mode": {},
        "pairwise_diffs": {},
        "win_rates": {},
    }
    
    # Per-mode aggregates
    for mode in FEATURE_MODES:
        mode_df = df[(df["mode"] == mode) & (df["run_ok"] == True)]
        
        ic_vals = mode_df["ic_mean"].dropna()
        delta_vals = mode_df["delta_all"].dropna()
        
        aggregated["per_mode"][mode] = {
            "n_runs": len(mode_df),
            "ic_mean_mean": float(ic_vals.mean()) if len(ic_vals) > 0 else None,
            "ic_mean_std": float(ic_vals.std()) if len(ic_vals) > 0 else None,
            "delta_all_mean": float(delta_vals.mean()) if len(delta_vals) > 0 else None,
            "delta_all_std": float(delta_vals.std()) if len(delta_vals) > 0 else None,
        }
    
    # Pairwise comparisons (fund_full vs others)
    for compare_mode in ["tech_only", "fund_zeroed", "fund_shuffled"]:
        diff_key = f"fund_full_minus_{compare_mode}"
        
        # Get paired observations (same window + seed)
        fund_full_df = df[(df["mode"] == "fund_full") & (df["run_ok"] == True)].set_index(["window_id", "seed"])
        compare_df = df[(df["mode"] == compare_mode) & (df["run_ok"] == True)].set_index(["window_id", "seed"])
        
        common_idx = fund_full_df.index.intersection(compare_df.index)
        
        if len(common_idx) > 0:
            fund_delta = fund_full_df.loc[common_idx, "delta_all"].dropna()
            comp_delta = compare_df.loc[common_idx, "delta_all"].dropna()
            
            # Align indices
            common = fund_delta.index.intersection(comp_delta.index)
            if len(common) > 0:
                diff = fund_delta.loc[common] - comp_delta.loc[common]
                aggregated["pairwise_diffs"][diff_key] = {
                    "n_pairs": len(diff),
                    "delta_diff_mean": float(diff.mean()),
                    "delta_diff_std": float(diff.std()) if len(diff) > 1 else 0.0,
                }
                
                # Win rate
                wins = (diff > 0).sum()
                aggregated["win_rates"][diff_key] = {
                    "wins": int(wins),
                    "total": len(diff),
                    "rate": float(wins / len(diff)) if len(diff) > 0 else 0.0,
                }
            else:
                aggregated["pairwise_diffs"][diff_key] = {"n_pairs": 0, "delta_diff_mean": None, "delta_diff_std": None}
                aggregated["win_rates"][diff_key] = {"wins": 0, "total": 0, "rate": None}
        else:
            aggregated["pairwise_diffs"][diff_key] = {"n_pairs": 0, "delta_diff_mean": None, "delta_diff_std": None}
            aggregated["win_rates"][diff_key] = {"wins": 0, "total": 0, "rate": None}
    
    return aggregated


def _generate_sweep_report(
    df: pd.DataFrame,
    aggregated: dict,
    report_path: Path,
    n_windows: int,
    n_seeds: int,
) -> None:
    """Generate REPORT_10_2_11.md from sweep results."""
    lines = [
        "# Task 10.2.11 — Fund Feature Ablation Stability Sweep",
        "",
        "## Run Matrix",
        "",
        f"- Windows: {n_windows}",
        f"- Seeds: {n_seeds}",
        f"- Total runs: {n_windows * n_seeds}",
        f"- Successful runs per mode:",
        "",
    ]
    
    # Success counts per mode
    for mode in FEATURE_MODES:
        n_ok = aggregated["per_mode"].get(mode, {}).get("n_runs", 0)
        lines.append(f"  - {mode}: {n_ok}")
    
    # Per-mode aggregated table
    lines.extend([
        "",
        "## Aggregated Results by Mode",
        "",
        "| Mode | IC Mean (μ±σ) | Delta All (μ±σ) |",
        "|------|---------------|-----------------|",
    ])
    
    for mode in FEATURE_MODES:
        mode_stats = aggregated["per_mode"].get(mode, {})
        ic_mean = mode_stats.get("ic_mean_mean")
        ic_std = mode_stats.get("ic_mean_std")
        delta_mean = mode_stats.get("delta_all_mean")
        delta_std = mode_stats.get("delta_all_std")
        
        ic_str = f"{ic_mean:.4f}±{ic_std:.4f}" if ic_mean is not None else "N/A"
        delta_str = f"{delta_mean:+.4f}±{delta_std:.4f}" if delta_mean is not None else "N/A"
        
        lines.append(f"| {mode} | {ic_str} | {delta_str} |")
    
    # Pairwise diff table
    lines.extend([
        "",
        "## Pairwise Differences (FUND_FULL − Other)",
        "",
        "| Comparison | ΔDelta (μ±σ) | Win Rate |",
        "|------------|--------------|----------|",
    ])
    
    for compare_mode in ["tech_only", "fund_zeroed", "fund_shuffled"]:
        diff_key = f"fund_full_minus_{compare_mode}"
        diff_stats = aggregated["pairwise_diffs"].get(diff_key, {})
        win_stats = aggregated["win_rates"].get(diff_key, {})
        
        diff_mean = diff_stats.get("delta_diff_mean")
        diff_std = diff_stats.get("delta_diff_std")
        win_rate = win_stats.get("rate")
        wins = win_stats.get("wins", 0)
        total = win_stats.get("total", 0)
        
        diff_str = f"{diff_mean:+.4f}±{diff_std:.4f}" if diff_mean is not None else "N/A"
        win_str = f"{win_rate:.1%} ({wins}/{total})" if win_rate is not None else "N/A"
        
        lines.append(f"| FUND_FULL − {compare_mode} | {diff_str} | {win_str} |")
    
    # Verdict
    lines.extend(["", "## Verdict", ""])
    
    win_rate_vs_tech = aggregated["win_rates"].get("fund_full_minus_tech_only", {}).get("rate", 0)
    diff_vs_tech = aggregated["pairwise_diffs"].get("fund_full_minus_tech_only", {}).get("delta_diff_mean", 0)
    diff_vs_shuffle = aggregated["pairwise_diffs"].get("fund_full_minus_fund_shuffled", {}).get("delta_diff_mean", 0)
    
    # Determine verdict
    promote = False
    reasons = []
    
    if win_rate_vs_tech is not None and win_rate_vs_tech >= 0.7:
        reasons.append(f"✓ Win rate vs TECH_ONLY: {win_rate_vs_tech:.1%} ≥ 70%")
    else:
        reasons.append(f"✗ Win rate vs TECH_ONLY: {win_rate_vs_tech:.1%} < 70%" if win_rate_vs_tech else "✗ Win rate: N/A")
    
    if diff_vs_tech is not None and diff_vs_tech > 0.005:
        reasons.append(f"✓ Mean ΔDelta vs TECH_ONLY: {diff_vs_tech:+.4f} > 0.005")
    else:
        reasons.append(f"✗ Mean ΔDelta vs TECH_ONLY: {diff_vs_tech:+.4f}" if diff_vs_tech else "✗ Mean ΔDelta: N/A")
    
    if diff_vs_shuffle is not None and diff_vs_shuffle > 0:
        reasons.append(f"✓ FUND_SHUFFLED underperforms FUND_FULL: {diff_vs_shuffle:+.4f}")
    else:
        reasons.append(f"✗ FUND_SHUFFLED does not underperform: {diff_vs_shuffle}" if diff_vs_shuffle else "✗ Shuffle comparison: N/A")
    
    # Compute promote
    if (win_rate_vs_tech is not None and win_rate_vs_tech >= 0.7 and
        diff_vs_tech is not None and diff_vs_tech > 0.005 and
        diff_vs_shuffle is not None and diff_vs_shuffle > 0):
        promote = True
    
    for reason in reasons:
        lines.append(f"- {reason}")
    
    lines.append("")
    if promote:
        lines.append("**VERDICT: PROMOTE_FUND** — Fundamental features provide consistent, statistically meaningful lift.")
    else:
        lines.append("**VERDICT: NO_PROMOTION** — Insufficient evidence for consistent fundamental feature value.")
    
    lines.append("")
    
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Task 10.2.11: Fund Feature Ablation Stability Sweep")
    parser.add_argument("--fund-alpha-dataset-path", required=True, help="Path to FUND alpha dataset CSV.gz")
    parser.add_argument("--prices-csv-path", required=True, help="Path to prices CSV")
    parser.add_argument("--baseline-scores-csv-path", required=True, help="Path to baseline scores CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--as-of-date", default="2024-10-01", help="PIT cutoff date")
    parser.add_argument("--subset-mode", choices=["all", "sec_covered", "sec_missing", "split"],
                        default="split", help="Subset mode for backtest")
    parser.add_argument("--rebalance", choices=["M", "Q"], default="Q", help="Rebalance frequency")
    parser.add_argument("--target-col", default="fwd_ret_21d", help="Target column")
    parser.add_argument("--top-k", type=int, default=400, help="Top K selection")
    parser.add_argument("--invert-scores", choices=["on", "off"], default="off", help="Invert scores")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated list of seeds")
    parser.add_argument("--min-train-samples", type=int, default=500, help="Minimum training samples")
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    try:
        run_fund_feature_ablation_sweep(
            fund_alpha_dataset_path=args.fund_alpha_dataset_path,
            prices_csv_path=args.prices_csv_path,
            baseline_scores_csv_path=args.baseline_scores_csv_path,
            out_dir=args.out_dir,
            as_of_date=args.as_of_date,
            windows=DEFAULT_WINDOWS,
            seeds=seeds,
            subset_mode=args.subset_mode,
            rebalance=args.rebalance,
            target_col=args.target_col,
            top_k=args.top_k,
            invert_scores=(args.invert_scores == "on"),
            min_train_samples=args.min_train_samples,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
