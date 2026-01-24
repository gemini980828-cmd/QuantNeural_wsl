"""
Rolling Temperature Calibration Sweep (v9.6.3).

Runs rolling windows of temperature calibration (VAL-fit) and evaluates on
subsequent TEST windows to assess whether calibration generalizes.

Key Properties:
- Shadow-Only: Does not change any trading/execution defaults.
- Deterministic: Same inputs + same seed => byte-identical outputs.
- Fail-Safe: Individual window failures do not crash the sweep.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd

from src.shadow_risk_calibration import run_shadow_risk_temperature_calibration


def _generate_windows(
    as_of_date: str,
    train_years: int,
    val_years: int,
    test_years: int,
    step_months: int,
    data_start_date: pd.Timestamp,
) -> list[dict]:
    """
    Generate rolling window boundaries.
    
    Returns list of dicts with train_end, val_end, test_end as date strings.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Calculate the earliest val_end that allows train_years + val_years history
    min_train_start = data_start_date
    min_val_end = min_train_start + pd.DateOffset(years=train_years + val_years)
    
    # Start from min_val_end and step backwards to find first window
    # whose test_end can reach as_of_date
    windows = []
    
    # Work forwards from min_val_end
    val_end = min_val_end
    window_id = 0
    
    while True:
        train_end = val_end - pd.DateOffset(years=val_years)
        test_end = val_end + pd.DateOffset(years=test_years)
        
        # Cap test_end at as_of_date
        if test_end > as_of_dt:
            test_end = as_of_dt
        
        # Check if this window makes sense
        if val_end > as_of_dt:
            break
        
        windows.append({
            "window_id": window_id,
            "train_end": train_end.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        
        window_id += 1
        val_end = val_end + pd.DateOffset(months=step_months)
    
    return windows


def _extract_metrics_from_json(json_path: str) -> dict:
    """Extract required metrics from calibration JSON output."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        best_temp = data.get("best_temperature")
        metrics = data.get("metrics", {})
        
        val = metrics.get("val", {})
        test = metrics.get("test", {})
        
        val_uncal = val.get("uncalibrated", {})
        val_cal = val.get("calibrated", {})
        test_uncal = test.get("uncalibrated", {})
        test_cal = test.get("calibrated", {})
        
        return {
            "best_temperature": best_temp,
            "n_obs_val": val_uncal.get("n_obs", 0),
            "n_obs_test": test_uncal.get("n_obs", 0),
            "val_ece_uncal": val_uncal.get("ece"),
            "val_ece_cal": val_cal.get("ece"),
            "test_ece_uncal": test_uncal.get("ece"),
            "test_ece_cal": test_cal.get("ece"),
            "val_log_loss_uncal": val_uncal.get("log_loss"),
            "val_log_loss_cal": val_cal.get("log_loss"),
            "test_log_loss_uncal": test_uncal.get("log_loss"),
            "test_log_loss_cal": test_cal.get("log_loss"),
        }
    except Exception:
        return {
            "best_temperature": None,
            "n_obs_val": 0,
            "n_obs_test": 0,
            "val_ece_uncal": None,
            "val_ece_cal": None,
            "test_ece_uncal": None,
            "test_ece_cal": None,
            "val_log_loss_uncal": None,
            "val_log_loss_cal": None,
            "test_log_loss_uncal": None,
            "test_log_loss_cal": None,
        }


def _compute_summary(rows: list[dict]) -> dict:
    """Compute aggregate summary statistics from window results."""
    n_total = len(rows)
    
    # Evaluated windows have non-null test_ece_cal
    evaluated = [r for r in rows if r["test_ece_cal"] is not None]
    n_evaluated = len(evaluated)
    n_skipped = n_total - n_evaluated
    
    # Extract arrays for stats
    test_ece_uncal = [r["test_ece_uncal"] for r in evaluated if r["test_ece_uncal"] is not None]
    test_ece_cal = [r["test_ece_cal"] for r in evaluated if r["test_ece_cal"] is not None]
    best_temps = [r["best_temperature"] for r in evaluated if r["best_temperature"] is not None]
    
    def safe_mean(arr):
        return float(round(np.mean(arr), 10)) if len(arr) > 0 else None
    
    def safe_median(arr):
        return float(round(np.median(arr), 10)) if len(arr) > 0 else None
    
    # Share improved: test_ece_cal < test_ece_uncal
    n_improved = sum(
        1 for r in evaluated
        if r["test_ece_cal"] is not None and r["test_ece_uncal"] is not None
        and r["test_ece_cal"] < r["test_ece_uncal"]
    )
    share_improved = float(round(n_improved / n_evaluated, 10)) if n_evaluated > 0 else None
    
    # Delta = cal - uncal (positive = worse, negative = better)
    deltas = [
        r["test_ece_cal"] - r["test_ece_uncal"]
        for r in evaluated
        if r["test_ece_cal"] is not None and r["test_ece_uncal"] is not None
    ]
    
    return {
        "n_windows_total": n_total,
        "n_windows_evaluated": n_evaluated,
        "n_windows_skipped": n_skipped,
        "test_ece_uncal_mean": safe_mean(test_ece_uncal),
        "test_ece_uncal_median": safe_median(test_ece_uncal),
        "test_ece_cal_mean": safe_mean(test_ece_cal),
        "test_ece_cal_median": safe_median(test_ece_cal),
        "delta_test_ece_mean": safe_mean(deltas),
        "share_improved_test_ece": share_improved,
        "best_temperature_mean": safe_mean(best_temps),
        "best_temperature_median": safe_median(best_temps),
    }


def run_shadow_risk_temperature_sweep(
    shadow_csv_path: str,
    overlay_csv_path: str,
    output_dir: str,
    *,
    as_of_date: str,
    horizon_days: int = 63,
    drawdown_threshold: float = -0.10,
    n_bins: int = 10,
    seed: int = 42,
    train_years: int = 3,
    val_years: int = 1,
    test_years: int = 1,
    step_months: int = 6,
    min_obs_val: int = 100,
    min_obs_test: int = 100,
) -> dict:
    """
    Run rolling temperature calibration sweep.
    
    For each window:
    - Fit temperature on VAL: (train_end, val_end]
    - Evaluate on TEST: (val_end, test_end]
    
    Writes:
    - shadow_risk_temp_sweep_results.csv (per-window results)
    - shadow_risk_temp_sweep_summary.json (aggregate summary)
    
    Returns dict with summary stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load shadow CSV to determine data start date
    shadow_df = pd.read_csv(shadow_csv_path)
    shadow_df["date"] = pd.to_datetime(shadow_df["date"])
    shadow_df = shadow_df.sort_values("date")
    data_start_date = shadow_df["date"].min()
    
    # Generate windows
    windows = _generate_windows(
        as_of_date=as_of_date,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        step_months=step_months,
        data_start_date=data_start_date,
    )
    
    results = []
    
    for window in windows:
        window_id = window["window_id"]
        train_end = window["train_end"]
        val_end = window["val_end"]
        test_end = window["test_end"]
        
        # Create per-window output directory
        window_dir = os.path.join(output_dir, f"window_{window_id:03d}")
        os.makedirs(window_dir, exist_ok=True)
        
        row = {
            "window_id": window_id,
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
            "best_temperature": None,
            "n_obs_val": 0,
            "n_obs_test": 0,
            "val_ece_uncal": None,
            "val_ece_cal": None,
            "test_ece_uncal": None,
            "test_ece_cal": None,
            "val_log_loss_uncal": None,
            "val_log_loss_cal": None,
            "test_log_loss_uncal": None,
            "test_log_loss_cal": None,
            "warning": "",
        }
        
        try:
            # Run calibration for this window
            run_shadow_risk_temperature_calibration(
                shadow_csv_path=shadow_csv_path,
                overlay_csv_path=overlay_csv_path,
                output_dir=window_dir,
                train_end=train_end,
                val_end=val_end,
                as_of_date=test_end,
                horizon_days=horizon_days,
                drawdown_threshold=drawdown_threshold,
                n_bins=n_bins,
                seed=seed,
            )
            
            # Extract metrics from produced JSON
            json_path = os.path.join(window_dir, "shadow_risk_metrics_calibrated_temp.json")
            metrics = _extract_metrics_from_json(json_path)
            row.update(metrics)
            
            # Check observation counts
            if row["n_obs_val"] < min_obs_val or row["n_obs_test"] < min_obs_test:
                row["warning"] = "SWEEP_SKIP:insufficient_obs"
                row["best_temperature"] = None
                row["val_ece_cal"] = None
                row["test_ece_cal"] = None
                row["val_log_loss_cal"] = None
                row["test_log_loss_cal"] = None
                
        except Exception:
            row["warning"] = "SWEEP_ERROR:calibration_failed"
        
        results.append(row)
    
    # Sort by val_end (should already be sorted, but explicit for safety)
    results.sort(key=lambda r: r["val_end"])
    
    # Write CSV
    _write_results_csv(results, output_dir)
    
    # Compute summary
    summary = _compute_summary(results)
    
    # Write JSON
    _write_summary_json(
        output_dir=output_dir,
        summary=summary,
        config={
            "as_of_date": as_of_date,
            "drawdown_threshold": drawdown_threshold,
            "horizon_days": horizon_days,
            "min_obs_test": min_obs_test,
            "min_obs_val": min_obs_val,
            "n_bins": n_bins,
            "seed": seed,
            "step_months": step_months,
            "test_years": test_years,
            "train_years": train_years,
            "val_years": val_years,
        },
    )
    
    return {
        "summary": summary,
        "n_windows": len(results),
    }


def _write_results_csv(results: list[dict], output_dir: str) -> None:
    """Write sweep results CSV with deterministic format."""
    columns = [
        "window_id",
        "train_end",
        "val_end",
        "test_end",
        "best_temperature",
        "n_obs_val",
        "n_obs_test",
        "val_ece_uncal",
        "val_ece_cal",
        "test_ece_uncal",
        "test_ece_cal",
        "val_log_loss_uncal",
        "val_log_loss_cal",
        "test_log_loss_uncal",
        "test_log_loss_cal",
        "warning",
    ]
    
    if len(results) == 0:
        # Create empty DataFrame with columns
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(results)
        df = df[columns]  # Enforce column order
    
    output_path = os.path.join(output_dir, "shadow_risk_temp_sweep_results.csv")
    df.to_csv(output_path, index=False, float_format="%.10f", lineterminator="\n")


def _write_summary_json(output_dir: str, summary: dict, config: dict) -> None:
    """Write sweep summary JSON with deterministic format."""
    output_json = {
        "schema_version": "9.6.3",
        "config": config,
        "summary": summary,
    }
    
    output_path = os.path.join(output_dir, "shadow_risk_temp_sweep_summary.json")
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(output_json, f, indent=2, sort_keys=True)
