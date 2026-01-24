"""
Task 10.2.10: Fund Feature Ablation Suite

Runs controlled ablations/permutations of the FUND feature block:
1. fund_full - Use all features (baseline)
2. tech_only - Exclude FUND columns (control)
3. fund_zeroed - Set FUND to 0, _is_missing to 1 (remove signal)
4. fund_shuffled - Shuffle FUND within each date (break association)

Generates REPORT_10_2_10.md and ablation_summary.json.
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

from scripts.train_xgb_alpha_from_dataset import train_xgb_alpha_from_dataset, FEATURE_MODES
from scripts.run_xgb_alpha_ab_backtest import run_xgb_alpha_ab_backtest


def run_fund_feature_ablation_suite(
    *,
    fund_alpha_dataset_path: str,
    prices_csv_path: str,
    baseline_scores_csv_path: str,
    out_dir: str,
    as_of_date: str = "2024-10-01",
    train_end: str = "2014-12-31",
    val_end: str = "2016-12-31",
    rebalance: str = "Q",
    target_col: str = "fwd_ret_21d",
    top_k: int = 400,
    seed: int = 42,
    invert_scores: bool = False,
    subset_mode: Literal["all", "sec_covered", "sec_missing", "split"] = "all",
    fund_dataset_path: str | None = None,
    min_train_samples: int = 500,
) -> dict:
    """
    Run ablation suite for all 4 feature modes.
    
    Returns dict with paths and results.
    """
    np.random.seed(seed)
    
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    # For subset modes, fund_dataset_path defaults to fund_alpha_dataset_path
    if subset_mode != "all" and fund_dataset_path is None:
        fund_dataset_path = fund_alpha_dataset_path
    
    results: dict[str, dict] = {}
    warnings_list: list[str] = []
    
    print("=" * 60)
    print("Task 10.2.10: Fund Feature Ablation Suite")
    print("=" * 60)
    print(f"Dataset: {fund_alpha_dataset_path}")
    print(f"Out dir: {out_dir}")
    print(f"Modes:   {FEATURE_MODES}")
    print("=" * 60)
    
    for mode in FEATURE_MODES:
        print(f"\n[{mode}] Training...")
        mode_dir = out_path / mode
        model_dir = mode_dir / "model"
        ab_dir = mode_dir / "ab"
        
        mode_result = {
            "feature_mode": mode,
            "model_dir": str(model_dir),
            "ab_dir": str(ab_dir),
            "training_success": False,
            "backtest_success": False,
            "ic_mean": None,
            "delta_cagr_vol_all": None,
            "delta_cagr_vol_sec_covered": None,
            "delta_cagr_vol_sec_missing": None,
            "n_features_used": None,
            "warnings": [],
        }
        
        # Step 1: Train
        try:
            train_result = train_xgb_alpha_from_dataset(
                alpha_dataset_path=fund_alpha_dataset_path,
                as_of_date=as_of_date,
                train_end=train_end,
                val_end=val_end,
                out_dir=str(model_dir),
                rebalance=rebalance,
                target_col=target_col,
                seed=seed,
                top_k=top_k,
                feature_mode=mode,
                min_train_samples=min_train_samples,
            )
            
            mode_result["training_success"] = True
            mode_result["scores_csv"] = train_result["scores_csv"]
            mode_result["ic_csv"] = train_result["ic_csv"]
            mode_result["summary_json"] = train_result["summary_json"]
            
            # Read summary for metrics
            with open(train_result["summary_json"]) as f:
                summary = json.load(f)
            mode_result["ic_mean"] = summary["results"].get("ic_spearman_mean")
            mode_result["n_features_used"] = summary.get("n_features_used")
            mode_result["warnings"].extend(summary.get("warnings", [])[:3])
            
            print(f"[{mode}] Training success. IC={mode_result['ic_mean']:.4f if mode_result['ic_mean'] else 'N/A'}")
            
        except Exception as e:
            error_msg = f"ABLATION_MODE_FAILED:{mode}:training:{str(e)[:50]}"
            mode_result["warnings"].append(error_msg)
            warnings_list.append(error_msg)
            print(f"[{mode}] Training FAILED: {e}")
            results[mode] = mode_result
            continue
        
        # Step 2: A/B Backtest
        print(f"[{mode}] Running A/B backtest...")
        try:
            ab_result = run_xgb_alpha_ab_backtest(
                prices_csv_path=prices_csv_path,
                baseline_scores_csv_path=baseline_scores_csv_path,
                xgb_scores_csv_path=train_result["scores_csv"],
                out_dir=str(ab_dir),
                seed=seed,
                invert_scores=invert_scores,
                subset_mode=subset_mode,
                fund_dataset_path=fund_dataset_path,
            )
            
            mode_result["backtest_success"] = True
            mode_result["diagnostics_json"] = ab_result.get("diagnostics_json")
            
            # Extract delta metrics
            if subset_mode == "split":
                # Get from each subset
                for subset_name in ["all", "sec_covered", "sec_missing"]:
                    subset_res = ab_result.get(subset_name, {})
                    xgb_res = subset_res.get("xgb", {})
                    if "delta" in xgb_res:
                        delta_val = xgb_res["delta"].get("delta_cagr_over_vol")
                        if subset_name == "all":
                            mode_result["delta_cagr_vol_all"] = delta_val
                        elif subset_name == "sec_covered":
                            mode_result["delta_cagr_vol_sec_covered"] = delta_val
                        else:
                            mode_result["delta_cagr_vol_sec_missing"] = delta_val
            else:
                # Single mode
                xgb_res = ab_result.get("xgb", {})
                if "delta" in xgb_res:
                    mode_result["delta_cagr_vol_all"] = xgb_res["delta"].get("delta_cagr_over_vol")
            
            # Read diagnostics for warnings
            if mode_result["diagnostics_json"] and Path(mode_result["diagnostics_json"]).exists():
                with open(mode_result["diagnostics_json"]) as f:
                    diag = json.load(f)
                mode_result["warnings"].extend(diag.get("warnings", [])[:3])
            
            print(f"[{mode}] Backtest success. Delta CAGR/Vol={mode_result['delta_cagr_vol_all']:.4f if mode_result['delta_cagr_vol_all'] else 'N/A'}")
            
        except Exception as e:
            error_msg = f"ABLATION_MODE_FAILED:{mode}:backtest:{str(e)[:50]}"
            mode_result["warnings"].append(error_msg)
            warnings_list.append(error_msg)
            print(f"[{mode}] Backtest FAILED: {e}")
        
        results[mode] = mode_result
    
    # Generate report using helper function
    report_path = out_path / "REPORT_10_2_10.md"
    _generate_ablation_report(results, report_path, subset_mode)
    
    # Generate ablation_summary.json
    summary_path = out_path / "ablation_summary.json"
    ablation_summary = {
        "schema_version": "10.2.10",
        "config": {
            "fund_alpha_dataset_path": fund_alpha_dataset_path,
            "prices_csv_path": prices_csv_path,
            "baseline_scores_csv_path": baseline_scores_csv_path,
            "as_of_date": as_of_date,
            "train_end": train_end,
            "val_end": val_end,
            "rebalance": rebalance,
            "target_col": target_col,
            "top_k": top_k,
            "seed": seed,
            "subset_mode": subset_mode,
        },
        "modes": results,
        "warnings": warnings_list,
    }
    
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ablation_summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    print(f"\n{'=' * 60}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")
    
    return {
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "modes": results,
    }


def _generate_ablation_report(
    results: dict[str, dict],
    report_path: Path,
    subset_mode: str,
) -> None:
    """Generate REPORT_10_2_10.md from ablation results."""
    lines = [
        "# Task 10.2.10 — Fund Feature Ablation Report",
        "",
        "## Ablation Results",
        "",
    ]
    
    # Build table header based on subset_mode
    if subset_mode == "split":
        lines.append("| Mode | IC Mean | N Features | Delta (all) | Delta (sec_covered) | Delta (sec_missing) | Notes |")
        lines.append("|------|---------|------------|-------------|---------------------|---------------------|-------|")
    else:
        lines.append("| Mode | IC Mean | N Features | Delta CAGR/Vol | Notes |")
        lines.append("|------|---------|------------|----------------|-------|")
    
    for mode in FEATURE_MODES:
        res = results.get(mode, {})
        ic_val = res.get("ic_mean")
        n_feat = res.get("n_features_used")
        delta_all = res.get("delta_cagr_vol_all")
        
        ic_str = f"{ic_val:.4f}" if ic_val is not None else "N/A"
        n_feat_str = str(n_feat) if n_feat is not None else "N/A"
        delta_all_str = f"+{delta_all:.4f}" if delta_all is not None else "N/A"
        
        # Truncate notes
        notes_list = res.get("warnings", [])
        notes_str = "; ".join(notes_list[:2])[:50] if notes_list else ""
        
        if subset_mode == "split":
            delta_covered = res.get("delta_cagr_vol_sec_covered")
            delta_missing = res.get("delta_cagr_vol_sec_missing")
            delta_cov_str = f"+{delta_covered:.4f}" if delta_covered is not None else "N/A"
            delta_miss_str = f"+{delta_missing:.4f}" if delta_missing is not None else "N/A"
            lines.append(f"| {mode} | {ic_str} | {n_feat_str} | {delta_all_str} | {delta_cov_str} | {delta_miss_str} | {notes_str} |")
        else:
            lines.append(f"| {mode} | {ic_str} | {n_feat_str} | {delta_all_str} | {notes_str} |")
    
    # Add verdict
    lines.extend(["", "## Verdict", ""])
    
    fund_full = results.get("fund_full", {})
    tech_only = results.get("tech_only", {})
    fund_shuffled = results.get("fund_shuffled", {})
    fund_zeroed = results.get("fund_zeroed", {})
    
    # Compare IC
    ic_fund = fund_full.get("ic_mean")
    ic_tech = tech_only.get("ic_mean")
    ic_shuffle = fund_shuffled.get("ic_mean")
    
    if ic_fund is not None and ic_tech is not None:
        if ic_fund > ic_tech:
            lines.append(f"- **FUND_FULL IC ({ic_fund:.4f}) > TECH_ONLY IC ({ic_tech:.4f})**: Fundamentals add predictive value.")
        else:
            lines.append(f"- **FUND_FULL IC ({ic_fund:.4f}) <= TECH_ONLY IC ({ic_tech:.4f})**: Fundamentals do NOT add IC lift.")
    
    if ic_fund is not None and ic_shuffle is not None:
        if ic_shuffle < ic_fund:
            lines.append(f"- **FUND_SHUFFLED IC ({ic_shuffle:.4f}) < FUND_FULL IC ({ic_fund:.4f})**: Shuffling degrades IC, confirming signal.")
        else:
            lines.append(f"- **FUND_SHUFFLED IC ({ic_shuffle:.4f}) >= FUND_FULL IC ({ic_fund:.4f})**: Shuffling did NOT degrade IC (unexpected).")
    
    # Compare Delta CAGR/Vol
    delta_fund = fund_full.get("delta_cagr_vol_all")
    delta_tech = tech_only.get("delta_cagr_vol_all")
    
    if delta_fund is not None and delta_tech is not None:
        if delta_fund > delta_tech:
            lines.append(f"- **FUND_FULL Delta ({delta_fund:.4f}) > TECH_ONLY Delta ({delta_tech:.4f})**: Backtest confirms fundamental value.")
        else:
            lines.append(f"- **FUND_FULL Delta ({delta_fund:.4f}) <= TECH_ONLY Delta ({delta_tech:.4f})**: Fundamentals do NOT improve backtest.")
    
    lines.append("")
    
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Task 10.2.10: Fund Feature Ablation Suite")
    parser.add_argument("--fund-alpha-dataset-path", required=True, help="Path to FUND alpha dataset CSV.gz")
    parser.add_argument("--prices-csv-path", required=True, help="Path to prices CSV")
    parser.add_argument("--baseline-scores-csv-path", required=True, help="Path to baseline scores CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--as-of-date", default="2024-10-01", help="PIT cutoff date")
    parser.add_argument("--train-end", default="2014-12-31", help="Training end date")
    parser.add_argument("--val-end", default="2016-12-31", help="Validation end date")
    parser.add_argument("--rebalance", choices=["M", "Q"], default="Q", help="Rebalance frequency")
    parser.add_argument("--target-col", default="fwd_ret_21d", help="Target column")
    parser.add_argument("--top-k", type=int, default=400, help="Top K selection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--invert-scores", choices=["on", "off"], default="off", help="Invert scores")
    parser.add_argument("--subset-mode", choices=["all", "sec_covered", "sec_missing", "split"], 
                        default="all", help="Subset mode for backtest")
    parser.add_argument("--fund-dataset-path", default=None, help="Path to FUND dataset for subset filtering")
    
    args = parser.parse_args()
    
    try:
        run_fund_feature_ablation_suite(
            fund_alpha_dataset_path=args.fund_alpha_dataset_path,
            prices_csv_path=args.prices_csv_path,
            baseline_scores_csv_path=args.baseline_scores_csv_path,
            out_dir=args.out_dir,
            as_of_date=args.as_of_date,
            train_end=args.train_end,
            val_end=args.val_end,
            rebalance=args.rebalance,
            target_col=args.target_col,
            top_k=args.top_k,
            seed=args.seed,
            invert_scores=(args.invert_scores == "on"),
            subset_mode=args.subset_mode,
            fund_dataset_path=args.fund_dataset_path,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
