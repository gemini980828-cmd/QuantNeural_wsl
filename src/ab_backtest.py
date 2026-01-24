"""
A/B Backtest Harness for QUANT-NEURAL.

Compares two score panels (baseline vs variant) through identical backtest
pipelines, producing summary metrics and deltas.

Design Principles:
- Deterministic: same inputs + seed => identical outputs
- Fail-fast: clear exceptions on mismatched/invalid inputs
- No default CLI changes: locked baseline = Q/topk/400/10/5
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.run_scores_backtest_from_csv import (
    run_scores_backtest_from_csv,
    _load_scores_csv,
    _load_prices_csv,
)


def _validate_score_panel(
    scores: pd.DataFrame,
    name: str,
) -> None:
    """Validate a score panel for A/B comparison."""
    # Check monotonic increasing index
    if not scores.index.is_monotonic_increasing:
        raise ValueError(f"{name} scores: index is not monotonic increasing")
    
    # Check unique index
    if not scores.index.is_unique:
        raise ValueError(f"{name} scores: index has duplicate dates")
    
    # Check all finite values
    if not np.all(np.isfinite(scores.values)):
        nan_count = np.sum(np.isnan(scores.values))
        inf_count = np.sum(np.isinf(scores.values))
        raise ValueError(
            f"{name} scores contain non-finite values: "
            f"{nan_count} NaN, {inf_count} inf"
        )


def _compute_delta_summary(
    baseline_metrics: dict,
    variant_metrics: dict,
) -> dict:
    """Compute delta summary (variant - baseline) for key metrics."""
    delta_keys = [
        "cagr_over_vol",
        "cagr",
        "ann_vol",
        "max_drawdown",
        "total_turnover",
        "total_cost",
    ]
    
    delta = {}
    for key in delta_keys:
        baseline_val = baseline_metrics.get(key, 0.0)
        variant_val = variant_metrics.get(key, 0.0)
        delta[key] = variant_val - baseline_val
    
    # Add percentage improvement for key metrics
    if baseline_metrics.get("cagr_over_vol", 0) != 0:
        delta["cagr_over_vol_pct_change"] = (
            (variant_metrics.get("cagr_over_vol", 0) - baseline_metrics.get("cagr_over_vol", 0))
            / abs(baseline_metrics.get("cagr_over_vol", 0)) * 100
        )
    else:
        delta["cagr_over_vol_pct_change"] = 0.0
    
    return delta


def _write_scores_panel_csv(scores: pd.DataFrame, path: str) -> None:
    """
    Write a score panel to CSV with explicit "date" column.
    
    Ensures:
    - First column named "date"
    - Dates in deterministic order (already monotonic)
    - index=False (date is a column, not index)
    
    Parameters
    ----------
    scores : pd.DataFrame
        Score panel with DatetimeIndex and ticker columns.
    path : str
        Output file path.
    """
    # Reset index to make date a column
    out = scores.reset_index()
    
    # Rename index column to "date" if needed
    if out.columns[0] != "date":
        out = out.rename(columns={out.columns[0]: "date"})
    
    # Write with index=False
    out.to_csv(path, index=False)


def _cleanup_temp_file(path: str) -> None:
    """Remove a temp file if it exists; ignore errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass  # Ignore cleanup failures


def run_ab_backtest_from_score_csvs(
    *,
    prices_csv_path: str,
    baseline_scores_csv_path: str,
    variant_scores_csv_path: str,
    output_dir: str,
    rebalance: str = "Q",
    method: str = "topk",
    top_k: int = 400,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    max_weight: Optional[float] = None,
    seed: int = 42,
) -> dict:
    """
    Run A/B backtest comparing baseline vs variant scores.
    
    Parameters
    ----------
    prices_csv_path : str
        Path to prices CSV (wide format: date + ticker columns).
    baseline_scores_csv_path : str
        Path to baseline scores CSV.
    variant_scores_csv_path : str
        Path to variant scores CSV.
    output_dir : str
        Directory to write output artifacts.
    rebalance : str
        Rebalance frequency: "M" or "Q" (default "Q").
    method : str
        Weight construction method (default "topk").
    top_k : int
        Number of top assets for topk method (default 400).
    cost_bps : float
        Transaction cost in basis points (default 10.0).
    slippage_bps : float
        Slippage in basis points (default 5.0).
    max_weight : float, optional
        Maximum weight per asset.
    seed : int
        Random seed for determinism (default 42).
    
    Returns
    -------
    dict
        A/B comparison results with keys:
        - "baseline_metrics": dict
        - "variant_metrics": dict
        - "delta": dict
        - "dates_used": list of date strings
        - "tickers_used": list of ticker strings
    
    Raises
    ------
    ValueError
        If scores have mismatched dates/tickers or contain invalid values.
    """
    # Set deterministic seed (for any potential randomness in pipeline)
    np.random.seed(seed)
    
    # Load score panels
    baseline_scores = _load_scores_csv(baseline_scores_csv_path, date_col="date")
    variant_scores = _load_scores_csv(variant_scores_csv_path, date_col="date")
    
    # Validate each panel
    _validate_score_panel(baseline_scores, "baseline")
    _validate_score_panel(variant_scores, "variant")
    
    # Compute intersection of dates
    common_dates = baseline_scores.index.intersection(variant_scores.index)
    if len(common_dates) == 0:
        raise ValueError(
            "No common dates between baseline and variant score panels. "
            f"Baseline: {baseline_scores.index[0]} to {baseline_scores.index[-1]}, "
            f"Variant: {variant_scores.index[0]} to {variant_scores.index[-1]}"
        )
    
    # Compute intersection of tickers
    common_tickers = sorted(
        set(baseline_scores.columns).intersection(set(variant_scores.columns))
    )
    if len(common_tickers) == 0:
        raise ValueError(
            "No common tickers between baseline and variant score panels. "
            f"Baseline tickers: {list(baseline_scores.columns)[:5]}..., "
            f"Variant tickers: {list(variant_scores.columns)[:5]}..."
        )
    
    # Subset to common dates and tickers
    baseline_subset = baseline_scores.loc[common_dates, common_tickers]
    variant_subset = variant_scores.loc[common_dates, common_tickers]
    
    # Re-validate after subsetting
    _validate_score_panel(baseline_subset, "baseline (after intersection)")
    _validate_score_panel(variant_subset, "variant (after intersection)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write subsetted scores to temp CSVs for backtest
    baseline_temp_csv = os.path.join(output_dir, "_temp_baseline_scores.csv")
    variant_temp_csv = os.path.join(output_dir, "_temp_variant_scores.csv")
    
    _write_scores_panel_csv(baseline_subset, baseline_temp_csv)
    _write_scores_panel_csv(variant_subset, variant_temp_csv)
    
    # Common backtest params
    backtest_params = {
        "prices_csv_path": prices_csv_path,
        "rebalance": rebalance,
        "method": method,
        "top_k": top_k,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "max_weight": max_weight,
    }
    
    try:
        # Run baseline backtest
        baseline_result = run_scores_backtest_from_csv(
            **backtest_params,
            scores_csv_path=baseline_temp_csv,
        )
        
        # Run variant backtest
        variant_result = run_scores_backtest_from_csv(
            **backtest_params,
            scores_csv_path=variant_temp_csv,
        )
    finally:
        # Cleanup temp files even on failure
        _cleanup_temp_file(baseline_temp_csv)
        _cleanup_temp_file(variant_temp_csv)
    
    # Extract metrics
    baseline_metrics = baseline_result["metrics"]
    variant_metrics = variant_result["metrics"]
    
    # Compute delta
    delta = _compute_delta_summary(baseline_metrics, variant_metrics)
    
    # Build summary dicts (JSON-serializable)
    baseline_summary = {
        "metrics": baseline_metrics,
        "params": {
            "rebalance": rebalance,
            "method": method,
            "top_k": top_k,
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
        },
        "source": "baseline",
    }
    
    variant_summary = {
        "metrics": variant_metrics,
        "params": {
            "rebalance": rebalance,
            "method": method,
            "top_k": top_k,
            "cost_bps": cost_bps,
            "slippage_bps": slippage_bps,
        },
        "source": "variant",
    }
    
    delta_summary = {
        "delta": delta,
        "interpretation": {
            "positive_cagr_over_vol": "variant outperforms baseline" if delta["cagr_over_vol"] > 0 else "baseline outperforms variant",
        },
    }
    
    # Write artifacts
    with open(os.path.join(output_dir, "baseline_summary.json"), "w") as f:
        json.dump(baseline_summary, f, indent=2, sort_keys=True)
    
    with open(os.path.join(output_dir, "variant_summary.json"), "w") as f:
        json.dump(variant_summary, f, indent=2, sort_keys=True)
    
    with open(os.path.join(output_dir, "delta_summary.json"), "w") as f:
        json.dump(delta_summary, f, indent=2, sort_keys=True)
    
    return {
        "baseline_metrics": baseline_metrics,
        "variant_metrics": variant_metrics,
        "delta": delta,
        "dates_used": [str(d.date()) for d in common_dates],
        "tickers_used": common_tickers,
    }
