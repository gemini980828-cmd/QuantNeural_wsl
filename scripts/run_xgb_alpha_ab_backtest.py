"""
A/B Backtest Wrapper: Baseline vs XGB Alpha Scores.

Calls the existing A/B harness with locked execution baseline:
- rebalance=Q
- method=topk
- top_k=400
- cost_bps=10
- slippage_bps=5

Supports:
- Standard A/B (baseline vs xgb)
- Inverse-score sanity check (baseline vs -1.0 * xgb)
- OOS diagnostics from ic_by_date.csv
- Subset-mode analysis (Task 10.2.8): SEC-covered vs SEC-missing attribution split

This is a thin wrapper that does NOT re-implement backtest logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ab_backtest import run_ab_backtest_from_score_csvs

# Task 10.2.8: SEC fundamental columns for coverage detection
SEC_FUNDAMENTAL_COLS = [
    "total_assets", "total_liabilities", "stockholders_equity",
    "revenues", "net_income", "operating_cash_flow", "shares_outstanding",
    "assets", "liabilities", "equity", "cash", "shares_out",
    "leverage", "cash_to_assets", "book_to_assets", "mktcap",
]

# Task 10.2.9: Penalty value for non-matching tickers in subset modes
# Must be lower than any realistic score to ensure they're never selected
SUBSET_PENALTY = -1e15


def _load_ic_diagnostics(xgb_scores_csv_path: str) -> dict | None:
    """Load IC diagnostics from ic_by_date.csv if it exists alongside scores.csv."""
    scores_path = Path(xgb_scores_csv_path)
    ic_path = scores_path.parent / "ic_by_date.csv"
    
    if not ic_path.exists():
        return None
    
    try:
        df = pd.read_csv(ic_path)
        if df.empty or "date" not in df.columns:
            return None
        
        return {
            "n_dates_scored": len(df),
            "date_min": str(df["date"].min()),
            "date_max": str(df["date"].max()),
            "ic_mean": float(df["ic_spearman"].mean()) if "ic_spearman" in df.columns else None,
        }
    except Exception:
        return None


def _invert_scores_csv(input_path: str, output_path: str) -> None:
    """Create an inverted scores CSV (multiply all ticker columns by -1.0)."""
    df = pd.read_csv(input_path)
    
    # Identify ticker columns (all except 'date')
    ticker_cols = [c for c in df.columns if c != "date"]
    
    # Multiply by -1.0
    for col in ticker_cols:
        df[col] = -1.0 * df[col].astype(np.float64)
    
    # Write with deterministic formatting
    df.to_csv(output_path, index=False, float_format="%.10f", lineterminator="\n")


def _load_coverage_map(
    fund_dataset_path: str,
) -> dict[tuple[str, str], tuple[bool, float | None]]:
    """
    Load coverage map from FUND dataset.
    
    Returns dict keyed by (date_str, ticker_upper) -> (is_covered: bool, mktcap: float | None).
    """
    if not fund_dataset_path or not Path(fund_dataset_path).exists():
        return {}
    
    # Load only needed columns for efficiency
    usecols = ["date", "ticker"]
    
    # Try to detect available columns
    try:
        if fund_dataset_path.endswith(".gz"):
            sample = pd.read_csv(fund_dataset_path, compression="gzip", nrows=5)
        else:
            sample = pd.read_csv(fund_dataset_path, nrows=5)
    except Exception:
        return {}
    
    cols_available = set(sample.columns)
    
    # Add coverage columns
    if "any_sec_present" in cols_available:
        usecols.append("any_sec_present")
    else:
        # Add SEC fundamental columns that exist
        for col in SEC_FUNDAMENTAL_COLS:
            if col in cols_available:
                usecols.append(col)
    
    # Add mktcap for tilt audit
    mktcap_col = None
    if "log_mktcap" in cols_available:
        usecols.append("log_mktcap")
        mktcap_col = "log_mktcap"
    elif "mktcap" in cols_available:
        usecols.append("mktcap")
        mktcap_col = "mktcap"
    
    # Load data
    try:
        if fund_dataset_path.endswith(".gz"):
            df = pd.read_csv(fund_dataset_path, compression="gzip", usecols=usecols)
        else:
            df = pd.read_csv(fund_dataset_path, usecols=usecols)
    except Exception:
        return {}
    
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".US", "", regex=False)
    
    # Compute coverage flag
    if "any_sec_present" in df.columns:
        df["_covered"] = df["any_sec_present"] == 1
    else:
        sec_cols = [c for c in SEC_FUNDAMENTAL_COLS if c in df.columns]
        if sec_cols:
            df["_covered"] = df[sec_cols].notna().any(axis=1)
        else:
            df["_covered"] = False
    
    # Get mktcap value
    if mktcap_col:
        df["_mktcap"] = df[mktcap_col].astype(float)
    else:
        df["_mktcap"] = np.nan
    
    # Build map
    result = {}
    for _, row in df.iterrows():
        key = (str(row["date"]), str(row["ticker"]))
        result[key] = (bool(row["_covered"]), float(row["_mktcap"]) if pd.notna(row["_mktcap"]) else None)
    
    return result


def _filter_scores_by_subset(
    scores_df: pd.DataFrame,
    coverage_map: dict[tuple[str, str], tuple[bool, float | None]],
    subset_mode: Literal["sec_covered", "sec_missing"],
    warnings_list: list[str],
) -> pd.DataFrame:
    """
    Filter scores DataFrame to only include tickers matching the subset mode.
    
    Returns a new DataFrame with NaN for non-matching tickers.
    """
    result = scores_df.copy()
    ticker_cols = [c for c in result.columns if c != "date"]
    
    for idx, row in result.iterrows():
        date_str = str(row["date"])
        for ticker in ticker_cols:
            key = (date_str, ticker.upper())
            if key in coverage_map:
                is_covered, _ = coverage_map[key]
            else:
                # Default to missing if not found in coverage map
                is_covered = False
            
            # Apply filter based on subset mode
            keep = (subset_mode == "sec_covered" and is_covered) or \
                   (subset_mode == "sec_missing" and not is_covered)
            
            if not keep:
                result.at[idx, ticker] = np.nan
    
    return result


def _compute_subset_audit(
    filtered_scores_df: pd.DataFrame,
    original_scores_df: pd.DataFrame,
    coverage_map: dict[tuple[str, str], tuple[bool, float | None]],
    subset_mode: str,
    top_k: int,
) -> dict:
    """
    Compute subset audit statistics.
    
    Task 10.2.9: Fixed to use original_scores_df for all-universe size calculation.
    """
    ticker_cols = [c for c in filtered_scores_df.columns if c != "date"]
    original_ticker_cols = [c for c in original_scores_df.columns if c != "date"]
    
    subset_universe_sizes = []
    all_universe_sizes = []
    dates_lt_topk = []
    
    for idx, row in filtered_scores_df.iterrows():
        date_str = str(row["date"])
        
        # Count subset universe: non-penalty values in filtered scores
        subset_count = 0
        for ticker in ticker_cols:
            val = row[ticker]
            # Consider as part of subset if finite and not penalty value
            if pd.notna(val) and val > SUBSET_PENALTY / 2:
                subset_count += 1
        subset_universe_sizes.append(subset_count)
        
        if subset_count < top_k:
            dates_lt_topk.append(date_str)
        
        # Count all universe from ORIGINAL unfiltered scores
        orig_row = original_scores_df.iloc[idx]
        all_count = sum(1 for t in original_ticker_cols if pd.notna(orig_row[t]))
        all_universe_sizes.append(all_count)
    
    mean_all = np.mean(all_universe_sizes) if all_universe_sizes else 0
    mean_subset = np.mean(subset_universe_sizes) if subset_universe_sizes else 0
    
    return {
        "subset_mode": subset_mode,
        "mean_universe_size_per_date": float(mean_subset),
        "min_universe_size_per_date": int(min(subset_universe_sizes)) if subset_universe_sizes else 0,
        "dates_with_universe_lt_topk": dates_lt_topk,
        "fraction_of_all_universe": float(mean_subset / mean_all) if mean_all > 0 else 0.0,
    }


def _compute_tilt_audit(
    baseline_scores_df: pd.DataFrame,
    variant_scores_df: pd.DataFrame,
    coverage_map: dict[tuple[str, str], tuple[bool, float | None]],
    top_k: int,
) -> dict:
    """
    Compute tilt audit comparing baseline vs variant TopK selection mktcap.
    
    Returns dict with mktcap comparison or available=false if not present.
    """
    # Check if mktcap is available
    has_mktcap = any(m is not None for _, (_, m) in coverage_map.items())
    
    if not has_mktcap:
        return {"available": False}
    
    ticker_cols = [c for c in baseline_scores_df.columns if c != "date"]
    
    baseline_mktcaps = []
    variant_mktcaps = []
    
    for idx, row in baseline_scores_df.iterrows():
        date_str = str(row["date"])
        
        # Get baseline TopK
        baseline_scores = [(t, float(row[t])) for t in ticker_cols if pd.notna(row[t])]
        baseline_scores.sort(key=lambda x: x[1], reverse=True)
        baseline_topk = [t for t, _ in baseline_scores[:top_k]]
        
        # Get variant TopK
        var_row = variant_scores_df.iloc[idx]
        variant_scores = [(t, float(var_row[t])) for t in ticker_cols if pd.notna(var_row[t])]
        variant_scores.sort(key=lambda x: x[1], reverse=True)
        variant_topk = [t for t, _ in variant_scores[:top_k]]
        
        # Compute mean mktcap for each
        base_mktcap_vals = []
        for t in baseline_topk:
            key = (date_str, t.upper())
            if key in coverage_map and coverage_map[key][1] is not None:
                base_mktcap_vals.append(coverage_map[key][1])
        
        var_mktcap_vals = []
        for t in variant_topk:
            key = (date_str, t.upper())
            if key in coverage_map and coverage_map[key][1] is not None:
                var_mktcap_vals.append(coverage_map[key][1])
        
        if base_mktcap_vals:
            baseline_mktcaps.append(np.mean(base_mktcap_vals))
        if var_mktcap_vals:
            variant_mktcaps.append(np.mean(var_mktcap_vals))
    
    if not baseline_mktcaps or not variant_mktcaps:
        return {"available": False}
    
    return {
        "available": True,
        "baseline_mean_mktcap_across_dates": float(np.mean(baseline_mktcaps)),
        "variant_mean_mktcap_across_dates": float(np.mean(variant_mktcaps)),
        "mktcap_diff_variant_minus_baseline": float(np.mean(variant_mktcaps) - np.mean(baseline_mktcaps)),
    }


def _run_subset_backtest(
    *,
    prices_csv_path: str,
    baseline_scores_csv_path: str,
    xgb_scores_csv_path: str,
    out_dir: str,
    seed: int,
    invert_scores: bool,
    subset_mode: Literal["all", "sec_covered", "sec_missing"],
    coverage_map: dict[tuple[str, str], tuple[bool, float | None]],
    warnings_list: list[str],
    top_k: int = 400,
) -> dict:
    """Run backtest for a single subset mode."""
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    # Load original scores (keep unfiltered copies for audit)
    original_baseline_df = pd.read_csv(baseline_scores_csv_path)
    original_variant_df = pd.read_csv(xgb_scores_csv_path)
    
    # Make working copies
    baseline_df = original_baseline_df.copy()
    variant_df = original_variant_df.copy()
    
    # Apply subset filtering if not "all"
    if subset_mode != "all":
        baseline_df = _filter_scores_by_subset(baseline_df, coverage_map, subset_mode, warnings_list)
        variant_df = _filter_scores_by_subset(variant_df, coverage_map, subset_mode, warnings_list)
    
    # Task 10.2.9: Apply penalty fill to NaN values before writing to CSV
    # This ensures the backtest harness never sees NaN values
    ticker_cols_base = [c for c in baseline_df.columns if c != "date"]
    ticker_cols_var = [c for c in variant_df.columns if c != "date"]
    
    for col in ticker_cols_base:
        baseline_df[col] = baseline_df[col].fillna(SUBSET_PENALTY)
    for col in ticker_cols_var:
        variant_df[col] = variant_df[col].fillna(SUBSET_PENALTY)
    
    # Write filtered scores to temp files (with penalty fill, no NaNs)
    filtered_baseline_path = out_path / "_filtered_baseline_scores.csv"
    filtered_variant_path = out_path / "_filtered_variant_scores.csv"
    
    baseline_df.to_csv(filtered_baseline_path, index=False, float_format="%.10f", lineterminator="\n")
    variant_df.to_csv(filtered_variant_path, index=False, float_format="%.10f", lineterminator="\n")
    
    # Compute subset audit using penalty-filled variant and original for full universe
    subset_audit = _compute_subset_audit(variant_df, original_variant_df, coverage_map, subset_mode, top_k)
    
    # Compute effective top_k for this subset (fail-safe)
    effective_top_k = top_k
    if subset_audit["min_universe_size_per_date"] < top_k:
        effective_top_k = max(1, subset_audit["min_universe_size_per_date"])
        for date_str in subset_audit["dates_with_universe_lt_topk"]:
            warn_msg = f"SUBSET_TOO_SMALL:{subset_mode}:{date_str}:{subset_audit['min_universe_size_per_date']}"
            warnings_list.append(warn_msg)
            print(f"WARNING: {warn_msg}", file=sys.stderr)
    
    results = {}
    
    # Run standard A/B
    xgb_out_dir = out_path / "xgb"
    xgb_out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        xgb_result = run_ab_backtest_from_score_csvs(
            prices_csv_path=prices_csv_path,
            baseline_scores_csv_path=str(filtered_baseline_path),
            variant_scores_csv_path=str(filtered_variant_path),
            output_dir=str(xgb_out_dir),
            rebalance="Q",
            method="topk",
            top_k=effective_top_k,
            cost_bps=10.0,
            slippage_bps=5.0,
            max_weight=None,
            seed=seed,
        )
        
        results["xgb"] = {
            "baseline_summary_json": str(xgb_out_dir / "baseline_summary.json"),
            "variant_summary_json": str(xgb_out_dir / "variant_summary.json"),
            "delta_summary_json": str(xgb_out_dir / "delta_summary.json"),
            "baseline_metrics": xgb_result["baseline_metrics"],
            "variant_metrics": xgb_result["variant_metrics"],
            "delta": xgb_result["delta"],
        }
    except Exception as e:
        results["xgb"] = {"error": str(e)}
        warnings_list.append(f"XGB_BACKTEST_FAILED:{subset_mode}:{e}")
    
    # Run inverse if requested
    if invert_scores:
        xgb_inverse_out_dir = out_path / "xgb_inverse"
        xgb_inverse_out_dir.mkdir(parents=True, exist_ok=True)
        
        inverted_csv_path = xgb_inverse_out_dir / "xgb_scores_inverted.csv"
        _invert_scores_csv(str(filtered_variant_path), str(inverted_csv_path))
        
        try:
            inverse_result = run_ab_backtest_from_score_csvs(
                prices_csv_path=prices_csv_path,
                baseline_scores_csv_path=str(filtered_baseline_path),
                variant_scores_csv_path=str(inverted_csv_path),
                output_dir=str(xgb_inverse_out_dir),
                rebalance="Q",
                method="topk",
                top_k=effective_top_k,
                cost_bps=10.0,
                slippage_bps=5.0,
                max_weight=None,
                seed=seed,
            )
            
            results["xgb_inverse"] = {
                "baseline_summary_json": str(xgb_inverse_out_dir / "baseline_summary.json"),
                "variant_summary_json": str(xgb_inverse_out_dir / "variant_summary.json"),
                "delta_summary_json": str(xgb_inverse_out_dir / "delta_summary.json"),
                "inverted_scores_csv": str(inverted_csv_path),
                "baseline_metrics": inverse_result["baseline_metrics"],
                "variant_metrics": inverse_result["variant_metrics"],
                "delta": inverse_result["delta"],
            }
        except Exception as e:
            results["xgb_inverse"] = {"error": str(e)}
            warnings_list.append(f"XGB_INVERSE_BACKTEST_FAILED:{subset_mode}:{e}")
    
    # Write subset_audit.json
    subset_audit["effective_top_k"] = effective_top_k
    subset_audit_path = out_path / "subset_audit.json"
    with open(subset_audit_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(subset_audit, f, ensure_ascii=False, indent=2, sort_keys=True)
    results["subset_audit"] = subset_audit
    
    # Compute and write tilt_audit.json
    tilt_audit = _compute_tilt_audit(baseline_df, variant_df, coverage_map, effective_top_k)
    tilt_audit_path = out_path / "tilt_audit.json"
    with open(tilt_audit_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(tilt_audit, f, ensure_ascii=False, indent=2, sort_keys=True)
    results["tilt_audit"] = tilt_audit
    
    # Cleanup temp files
    try:
        filtered_baseline_path.unlink()
        filtered_variant_path.unlink()
    except Exception:
        pass
    
    return results


def _generate_split_report(
    all_results: dict,
    sec_covered_results: dict,
    sec_missing_results: dict,
    ic_diagnostics: dict | None,
    out_dir: Path,
) -> str:
    """Generate REPORT_10_2_8.md for split mode."""
    
    def get_metric(results: dict, key: str, default: str = "N/A") -> str:
        try:
            if "xgb" in results and "delta" in results["xgb"]:
                val = results["xgb"]["delta"].get(key)
                if val is not None:
                    return f"{val:+.4f}"
        except Exception:
            pass
        return default
    
    def get_ic_mean(results: dict) -> str:
        try:
            if "subset_audit" in results:
                # IC is from the variant model, not from subset
                pass
        except Exception:
            pass
        return "N/A"
    
    lines = [
        "# Task 10.2.8 — Attribution Split Report",
        "",
        "## Subset Comparison",
        "",
        "| Subset | IC Mean | Delta CAGR/Vol (XGB) | Delta CAGR/Vol (XGB_INV) | Mean Universe | Fraction |",
        "|--------|---------|----------------------|--------------------------|---------------|----------|",
    ]
    
    for name, results in [("all", all_results), ("sec_covered", sec_covered_results), ("sec_missing", sec_missing_results)]:
        ic_mean = "N/A"
        if name == "all" and ic_diagnostics and ic_diagnostics.get("ic_mean") is not None:
            ic_mean = f"{ic_diagnostics['ic_mean']:.4f}"
        
        delta_xgb = get_metric(results, "cagr_over_vol")
        delta_inv = "N/A"
        if "xgb_inverse" in results and "delta" in results["xgb_inverse"]:
            try:
                val = results["xgb_inverse"]["delta"].get("cagr_over_vol")
                if val is not None:
                    delta_inv = f"{val:+.4f}"
            except Exception:
                pass
        
        mean_univ = "N/A"
        fraction = "N/A"
        if "subset_audit" in results:
            mean_univ = f"{results['subset_audit'].get('mean_universe_size_per_date', 0):.0f}"
            fraction = f"{results['subset_audit'].get('fraction_of_all_universe', 0):.1%}"
        
        lines.append(f"| {name} | {ic_mean} | {delta_xgb} | {delta_inv} | {mean_univ} | {fraction} |")
    
    # Verdict section
    lines.extend([
        "",
        "## Verdict",
        "",
    ])
    
    # Analyze where improvement comes from
    all_delta = None
    covered_delta = None
    missing_delta = None
    
    try:
        if "xgb" in all_results and "delta" in all_results["xgb"]:
            all_delta = all_results["xgb"]["delta"].get("cagr_over_vol")
        if "xgb" in sec_covered_results and "delta" in sec_covered_results["xgb"]:
            covered_delta = sec_covered_results["xgb"]["delta"].get("cagr_over_vol")
        if "xgb" in sec_missing_results and "delta" in sec_missing_results["xgb"]:
            missing_delta = sec_missing_results["xgb"]["delta"].get("cagr_over_vol")
    except Exception:
        pass
    
    if covered_delta is not None and missing_delta is not None:
        if covered_delta > 0 and (missing_delta <= 0 or covered_delta > missing_delta * 2):
            lines.append("- **Improvement concentrated in SEC-COVERED subset**: The positive delta is primarily from tickers with SEC fundamental data.")
        elif missing_delta > 0 and (covered_delta <= 0 or missing_delta > covered_delta * 2):
            lines.append("- **Improvement concentrated in SEC-MISSING subset**: The positive delta comes from non-SEC tickers, suggesting the model benefits from technical features.")
        elif covered_delta > 0 and missing_delta > 0:
            lines.append("- **Both subsets show improvement**: The model provides value across both SEC-covered and SEC-missing universes.")
        else:
            lines.append("- **Inconclusive**: Neither subset shows clear improvement.")
    else:
        lines.append("- **Unable to compute subset comparison** due to missing metrics.")
    
    # Check for stability warnings
    covered_lt = sec_covered_results.get("subset_audit", {}).get("dates_with_universe_lt_topk", [])
    missing_lt = sec_missing_results.get("subset_audit", {}).get("dates_with_universe_lt_topk", [])
    
    if covered_lt:
        lines.append(f"- **Warning**: SEC-COVERED subset had {len(covered_lt)} dates with universe < top_k (top_k adapted).")
    if missing_lt:
        lines.append(f"- **Warning**: SEC-MISSING subset had {len(missing_lt)} dates with universe < top_k (top_k adapted).")
    
    lines.append("")
    
    report_content = "\n".join(lines)
    report_path = out_dir / "REPORT_10_2_8.md"
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report_content)
    
    return str(report_path)


def run_xgb_alpha_ab_backtest(
    *,
    prices_csv_path: str,
    baseline_scores_csv_path: str,
    xgb_scores_csv_path: str,
    out_dir: str,
    seed: int = 42,
    invert_scores: bool = False,
    subset_mode: Literal["all", "sec_covered", "sec_missing", "split"] = "all",
    fund_dataset_path: Optional[str] = None,
) -> dict:
    """
    Run A/B backtest comparing baseline vs XGB alpha scores.
    
    Calls run_ab_backtest_from_score_csvs() using the repository's locked 
    execution baseline:
    - rebalance=Q
    - method=topk
    - top_k=400
    - cost_bps=10
    - slippage_bps=5
    
    Parameters
    ----------
    prices_csv_path : str
        Path to prices CSV (wide format: date + ticker columns).
    baseline_scores_csv_path : str
        Path to baseline scores CSV.
    xgb_scores_csv_path : str
        Path to XGB alpha scores CSV.
    out_dir : str
        Output directory for summary JSON files.
    seed : int
        Random seed for determinism (default 42).
    invert_scores : bool
        If True, also run baseline vs inverted XGB scores (default False).
    subset_mode : str
        One of: "all", "sec_covered", "sec_missing", "split" (default "all").
        - all: existing behavior
        - sec_covered: filter to SEC-covered tickers only
        - sec_missing: filter to SEC-missing tickers only
        - split: run all three and produce consolidated report
    fund_dataset_path : str, optional
        Path to FUND dataset CSV (for coverage flags). Required for subset modes.
    
    Returns
    -------
    dict
        Dictionary with result paths and metrics.
    """
    np.random.seed(seed)
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    warnings_list: list[str] = []
    results = {}
    
    # Load IC diagnostics if available
    ic_diagnostics = _load_ic_diagnostics(xgb_scores_csv_path)
    
    # Load coverage map if subset mode requires it
    coverage_map = {}
    if subset_mode in ("sec_covered", "sec_missing", "split"):
        if not fund_dataset_path:
            raise ValueError(f"fund_dataset_path is required for subset_mode='{subset_mode}'")
        coverage_map = _load_coverage_map(fund_dataset_path)
        if not coverage_map:
            raise ValueError(f"Could not load coverage data from {fund_dataset_path}")
    
    # Handle split mode
    if subset_mode == "split":
        all_out = out_path / "all"
        covered_out = out_path / "sec_covered"
        missing_out = out_path / "sec_missing"
        
        all_results = _run_subset_backtest(
            prices_csv_path=prices_csv_path,
            baseline_scores_csv_path=baseline_scores_csv_path,
            xgb_scores_csv_path=xgb_scores_csv_path,
            out_dir=str(all_out),
            seed=seed,
            invert_scores=invert_scores,
            subset_mode="all",
            coverage_map=coverage_map,
            warnings_list=warnings_list,
        )
        
        covered_results = _run_subset_backtest(
            prices_csv_path=prices_csv_path,
            baseline_scores_csv_path=baseline_scores_csv_path,
            xgb_scores_csv_path=xgb_scores_csv_path,
            out_dir=str(covered_out),
            seed=seed,
            invert_scores=invert_scores,
            subset_mode="sec_covered",
            coverage_map=coverage_map,
            warnings_list=warnings_list,
        )
        
        missing_results = _run_subset_backtest(
            prices_csv_path=prices_csv_path,
            baseline_scores_csv_path=baseline_scores_csv_path,
            xgb_scores_csv_path=xgb_scores_csv_path,
            out_dir=str(missing_out),
            seed=seed,
            invert_scores=invert_scores,
            subset_mode="sec_missing",
            coverage_map=coverage_map,
            warnings_list=warnings_list,
        )
        
        # Generate consolidated report
        report_path = _generate_split_report(
            all_results, covered_results, missing_results, ic_diagnostics, out_path
        )
        
        results["all"] = all_results
        results["sec_covered"] = covered_results
        results["sec_missing"] = missing_results
        results["report_path"] = report_path
        
    elif subset_mode in ("sec_covered", "sec_missing"):
        # Single subset mode
        subset_results = _run_subset_backtest(
            prices_csv_path=prices_csv_path,
            baseline_scores_csv_path=baseline_scores_csv_path,
            xgb_scores_csv_path=xgb_scores_csv_path,
            out_dir=str(out_path),
            seed=seed,
            invert_scores=invert_scores,
            subset_mode=subset_mode,
            coverage_map=coverage_map,
            warnings_list=warnings_list,
        )
        results = subset_results
        
    else:
        # Original "all" mode behavior (backward compatible)
        xgb_out_dir = out_path / "xgb"
        xgb_out_dir.mkdir(parents=True, exist_ok=True)
        
        xgb_result = run_ab_backtest_from_score_csvs(
            prices_csv_path=prices_csv_path,
            baseline_scores_csv_path=baseline_scores_csv_path,
            variant_scores_csv_path=xgb_scores_csv_path,
            output_dir=str(xgb_out_dir),
            rebalance="Q",
            method="topk",
            top_k=400,
            cost_bps=10.0,
            slippage_bps=5.0,
            max_weight=None,
            seed=seed,
        )
        
        results["xgb"] = {
            "baseline_summary_json": str(xgb_out_dir / "baseline_summary.json"),
            "variant_summary_json": str(xgb_out_dir / "variant_summary.json"),
            "delta_summary_json": str(xgb_out_dir / "delta_summary.json"),
            "baseline_metrics": xgb_result["baseline_metrics"],
            "variant_metrics": xgb_result["variant_metrics"],
            "delta": xgb_result["delta"],
            "dates_used": xgb_result["dates_used"],
            "tickers_used": xgb_result["tickers_used"],
        }
        
        if invert_scores:
            xgb_inverse_out_dir = out_path / "xgb_inverse"
            xgb_inverse_out_dir.mkdir(parents=True, exist_ok=True)
            
            inverted_csv_path = xgb_inverse_out_dir / "xgb_scores_inverted.csv"
            _invert_scores_csv(xgb_scores_csv_path, str(inverted_csv_path))
            
            inverse_result = run_ab_backtest_from_score_csvs(
                prices_csv_path=prices_csv_path,
                baseline_scores_csv_path=baseline_scores_csv_path,
                variant_scores_csv_path=str(inverted_csv_path),
                output_dir=str(xgb_inverse_out_dir),
                rebalance="Q",
                method="topk",
                top_k=400,
                cost_bps=10.0,
                slippage_bps=5.0,
                max_weight=None,
                seed=seed,
            )
            
            results["xgb_inverse"] = {
                "baseline_summary_json": str(xgb_inverse_out_dir / "baseline_summary.json"),
                "variant_summary_json": str(xgb_inverse_out_dir / "variant_summary.json"),
                "delta_summary_json": str(xgb_inverse_out_dir / "delta_summary.json"),
                "inverted_scores_csv": str(inverted_csv_path),
                "baseline_metrics": inverse_result["baseline_metrics"],
                "variant_metrics": inverse_result["variant_metrics"],
                "delta": inverse_result["delta"],
                "dates_used": inverse_result["dates_used"],
                "tickers_used": inverse_result["tickers_used"],
            }
    
    # Write diagnostics.json
    diagnostics = {
        "schema_version": "10.2.8",
        "seed": seed,
        "subset_mode": subset_mode,
        "paths": {
            "prices_csv": prices_csv_path,
            "baseline_scores_csv": baseline_scores_csv_path,
            "xgb_scores_csv": xgb_scores_csv_path,
            "fund_dataset": fund_dataset_path,
        },
        "invert_mode": invert_scores,
        "warnings": warnings_list,
    }
    
    if ic_diagnostics:
        diagnostics["oos_window"] = ic_diagnostics
    else:
        diagnostics["oos_window"] = {"warning": "ic_by_date.csv not found or invalid"}
    
    diagnostics_path = out_path / "diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    results["diagnostics_json"] = str(diagnostics_path)
    results["diagnostics"] = diagnostics
    
    return results


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="A/B Backtest: Baseline vs XGB Alpha Scores (Q400 execution baseline)"
    )
    parser.add_argument("--prices-csv-path", required=True, help="Path to prices CSV")
    parser.add_argument("--baseline-scores-csv-path", required=True, help="Path to baseline scores CSV")
    parser.add_argument("--xgb-scores-csv-path", required=True, help="Path to XGB alpha scores CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--xgb-invert-scores",
        choices=["on", "off"],
        default="off",
        help="Also run baseline vs inverted XGB scores (default: off)"
    )
    parser.add_argument(
        "--subset-mode",
        choices=["all", "sec_covered", "sec_missing", "split"],
        default="all",
        help="Subset mode: all (default), sec_covered, sec_missing, or split (run all three)"
    )
    parser.add_argument(
        "--fund-dataset-path",
        default=None,
        help="Path to FUND dataset CSV (required for subset modes)"
    )
    
    args = parser.parse_args()
    invert = args.xgb_invert_scores == "on"
    
    print("=" * 60)
    print("A/B Backtest: Baseline vs XGB Alpha (Q400 baseline)")
    print("=" * 60)
    print(f"prices_csv_path:          {args.prices_csv_path}")
    print(f"baseline_scores_csv_path: {args.baseline_scores_csv_path}")
    print(f"xgb_scores_csv_path:      {args.xgb_scores_csv_path}")
    print(f"out_dir:                  {args.out_dir}")
    print(f"seed:                     {args.seed}")
    print(f"invert_scores:            {invert}")
    print(f"subset_mode:              {args.subset_mode}")
    print(f"fund_dataset_path:        {args.fund_dataset_path}")
    print("=" * 60)
    print()
    
    try:
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=args.prices_csv_path,
            baseline_scores_csv_path=args.baseline_scores_csv_path,
            xgb_scores_csv_path=args.xgb_scores_csv_path,
            out_dir=args.out_dir,
            seed=args.seed,
            invert_scores=invert,
            subset_mode=args.subset_mode,
            fund_dataset_path=args.fund_dataset_path,
        )
        
        print()
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        if args.subset_mode == "split":
            print("\n[Split Mode Results]")
            for subset_name in ["all", "sec_covered", "sec_missing"]:
                if subset_name in result:
                    subset = result[subset_name]
                    if "xgb" in subset and "delta" in subset["xgb"]:
                        delta = subset["xgb"]["delta"].get("cagr_over_vol", "N/A")
                        print(f"  {subset_name}: Delta CAGR/Vol = {delta:+.4f}" if isinstance(delta, float) else f"  {subset_name}: Delta CAGR/Vol = {delta}")
            if "report_path" in result:
                print(f"\n  Report: {result['report_path']}")
        elif "xgb" in result:
            xgb = result["xgb"]
            if "delta" in xgb:
                print("\n[XGB vs Baseline]")
                print(f"  Baseline CAGR/Vol: {xgb['baseline_metrics'].get('cagr_over_vol', 'N/A'):.4f}")
                print(f"  XGB CAGR/Vol:      {xgb['variant_metrics'].get('cagr_over_vol', 'N/A'):.4f}")
                print(f"  Delta CAGR/Vol:    {xgb['delta'].get('cagr_over_vol', 'N/A'):.4f}")
            
            if "xgb_inverse" in result and "delta" in result["xgb_inverse"]:
                inv = result["xgb_inverse"]
                print("\n[XGB_INVERSE vs Baseline]")
                print(f"  Baseline CAGR/Vol: {inv['baseline_metrics'].get('cagr_over_vol', 'N/A'):.4f}")
                print(f"  XGB_INV CAGR/Vol:  {inv['variant_metrics'].get('cagr_over_vol', 'N/A'):.4f}")
                print(f"  Delta CAGR/Vol:    {inv['delta'].get('cagr_over_vol', 'N/A'):.4f}")
        
        # Diagnostics
        diag = result.get("diagnostics", {})
        print("\n[Diagnostics]")
        if "oos_window" in diag and "n_dates_scored" in diag["oos_window"]:
            oos = diag["oos_window"]
            print(f"  OOS dates scored:  {oos['n_dates_scored']}")
            print(f"  OOS date range:    {oos['date_min']} to {oos['date_max']}")
            if oos.get("ic_mean") is not None:
                print(f"  IC mean:           {oos['ic_mean']:.4f}")
        else:
            print("  OOS window:        (ic_by_date.csv not found)")
        
        if diag.get("warnings"):
            print(f"\n  Warnings: {len(diag['warnings'])}")
            for w in diag["warnings"][:5]:
                print(f"    - {w}")
        
        print()
        print("Artifacts written:")
        print(f"  - {result.get('diagnostics_json', 'N/A')}")
        if args.subset_mode == "split":
            for subset_name in ["all", "sec_covered", "sec_missing"]:
                if subset_name in result and "xgb" in result[subset_name] and "delta_summary_json" in result[subset_name]["xgb"]:
                    print(f"  - {result[subset_name]['xgb']['delta_summary_json']}")
        elif "xgb" in result and "delta_summary_json" in result["xgb"]:
            print(f"  - {result['xgb']['delta_summary_json']}")
        if "xgb_inverse" in result and "delta_summary_json" in result["xgb_inverse"]:
            print(f"  - {result['xgb_inverse']['delta_summary_json']}")
        print("=" * 60)
        
        return 0
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
