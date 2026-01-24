"""
Shadow Risk Probability Calibration via Temperature Scaling (v9.6.2).

Implements post-hoc temperature scaling calibrated on VAL split only,
then evaluates on TEST split. This improves ECE for the decision gate.

Key Properties:
- Shadow-Only: Does not change any trading/execution defaults.
- Deterministic: Same inputs + same seed => byte-identical outputs.
- VAL-fit / TEST-eval: Temperature is fit only on VAL, never on TEST.
- Fail-Safe: Never crashes; always writes artifacts with stable warnings.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd


def _compute_forward_labels(
    spy_ret_1d: pd.Series,
    horizon_days: int,
    drawdown_threshold: float,
) -> pd.Series:
    """
    Compute forward-looking risk-off labels for each date.
    
    For date t, the horizon window is (t+1 ... t+horizon_days).
    y_risk_off(t) = 1 if:
      - forward_return < 0, OR
      - forward_max_drawdown <= drawdown_threshold
    else 0.
    
    Last horizon_days rows have NaN labels (no future data).
    """
    n = len(spy_ret_1d)
    labels = pd.Series(index=spy_ret_1d.index, dtype=float)
    labels[:] = np.nan
    
    for i in range(n - horizon_days):
        # Window is (t+1 ... t+horizon_days), i.e., next horizon_days after date i
        window = spy_ret_1d.iloc[i + 1:i + 1 + horizon_days]
        
        if len(window) < horizon_days:
            continue
        
        # Forward return: product(1 + r) - 1
        forward_return = (1 + window).prod() - 1
        
        # Forward max drawdown: compute cumulative equity from 1.0
        equity = (1 + window).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        forward_max_dd = drawdown.min()
        
        # Label
        if forward_return < 0 or forward_max_dd <= drawdown_threshold:
            labels.iloc[i] = 1.0
        else:
            labels.iloc[i] = 0.0
    
    return labels


def _compute_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to logits with clipping."""
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function with numerical stability."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


def _compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Compute binary cross-entropy log loss."""
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return float(loss)


def _compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score."""
    return float(np.mean((y_pred - y_true) ** 2))


def _compute_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Compute ROC AUC if both classes present."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return None
    
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None
    
    pos_preds = y_pred[pos_mask]
    neg_preds = y_pred[neg_mask]
    
    # Count concordant pairs
    n_concordant = 0
    n_tied = 0
    for p in pos_preds:
        n_concordant += np.sum(neg_preds < p)
        n_tied += np.sum(neg_preds == p)
    
    n_pairs = len(pos_preds) * len(neg_preds)
    if n_pairs == 0:
        return None
    
    auc = (n_concordant + 0.5 * n_tied) / n_pairs
    return float(round(auc, 10))


def _compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """
    Compute Expected Calibration Error with equal-width bins.
    
    ECE = sum(|frac_pos - mean_pred| * count) / n
    
    Returns (ece, calibration_bins).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_bins = []
    ece = 0.0
    n_total = len(y_true)
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        
        count = int(mask.sum())
        
        if count > 0:
            frac_pos = float(round(y_true[mask].mean(), 10))
            mean_pred = float(round(y_pred[mask].mean(), 10))
            ece += count * abs(frac_pos - mean_pred)
        else:
            frac_pos = None
            mean_pred = float(round((lo + hi) / 2, 10))
        
        calibration_bins.append({
            "bin_idx": i,
            "count": count,
            "frac_pos": frac_pos,
            "mean_pred": mean_pred,
        })
    
    if n_total > 0:
        ece = ece / n_total
    
    return float(round(ece, 10)), calibration_bins


def _compute_split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute all metrics for a split."""
    if len(y_true) == 0:
        return {
            "n_obs": 0,
            "base_rate": None,
            "brier": None,
            "log_loss": None,
            "roc_auc": None,
            "ece": None,
            "calibration_bins": [],
        }
    
    ece, calibration_bins = _compute_ece(y_true, y_pred, n_bins)
    roc_auc = _compute_roc_auc(y_true, y_pred)
    
    return {
        "n_obs": int(len(y_true)),
        "base_rate": float(round(y_true.mean(), 10)),
        "brier": float(round(_compute_brier_score(y_true, y_pred), 10)),
        "log_loss": float(round(_compute_log_loss(y_true, y_pred), 10)),
        "roc_auc": roc_auc,
        "ece": ece,
        "calibration_bins": calibration_bins,
    }


def _find_best_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    t_min: float = 0.25,
    t_max: float = 5.0,
    n_grid: int = 400,
) -> float | None:
    """
    Find best temperature via grid search minimizing log loss on VAL.
    
    Tie-break: smallest T.
    Returns None if y_true has single class (cannot fit).
    """
    # Check for single-class
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return None
    
    temperatures = np.linspace(t_min, t_max, n_grid)
    best_t = None
    best_loss = float("inf")
    
    for t in temperatures:
        p_cal = _sigmoid(logits / t)
        loss = _compute_log_loss(y_true, p_cal)
        
        if loss < best_loss:
            best_loss = loss
            best_t = t
    
    if best_t is not None:
        return float(round(best_t, 10))
    return None


def run_shadow_risk_temperature_calibration(
    shadow_csv_path: str,
    overlay_csv_path: str,
    output_dir: str,
    *,
    train_end: str,
    val_end: str,
    as_of_date: str,
    horizon_days: int = 63,
    drawdown_threshold: float = -0.10,
    n_bins: int = 10,
    seed: int = 42,
) -> dict:
    """
    Apply temperature scaling calibration to shadow risk probabilities.
    
    Reads:
      - shadow_csv_path: must contain columns ['date','p_risk_off'] at minimum
      - overlay_csv_path: must contain columns ['date','spy_ret_1d'] at minimum
    
    Writes into output_dir:
      - shadow_risk_calibrated_temp.csv (same rows as shadow, adds p_risk_off_cal)
      - shadow_risk_metrics_calibrated_temp.json (train/val/test metrics)
    
    Returns:
      - "best_temperature": float | None
      - "metrics": dict (train/val/test with uncalibrated and calibrated metrics)
      - "warnings": list[str]
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    result_warnings: list[str] = []
    
    # Parse dates
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    as_of_dt = pd.to_datetime(as_of_date)
    
    # =========================================================================
    # Load shadow CSV
    # =========================================================================
    shadow_df = pd.read_csv(shadow_csv_path)
    shadow_df["date"] = pd.to_datetime(shadow_df["date"])
    shadow_df = shadow_df.set_index("date").sort_index()
    
    if "p_risk_off" not in shadow_df.columns:
        result_warnings.append("MISSING_P_RISK_OFF_COLUMN")
        shadow_df["p_risk_off"] = 0.5
    
    # Store original column order for output
    original_columns = list(shadow_df.columns)
    
    # =========================================================================
    # Load overlay CSV for SPY returns
    # =========================================================================
    overlay_df = pd.read_csv(overlay_csv_path)
    overlay_df["date"] = pd.to_datetime(overlay_df["date"])
    overlay_df = overlay_df.set_index("date").sort_index()
    
    if "spy_ret_1d" not in overlay_df.columns:
        result_warnings.append("MISSING_SPY_RET_1D_COLUMN")
        overlay_df["spy_ret_1d"] = 0.0
    
    # =========================================================================
    # Compute forward labels from SPY returns
    # =========================================================================
    spy_ret_1d = overlay_df["spy_ret_1d"].dropna()
    labels = _compute_forward_labels(spy_ret_1d, horizon_days, drawdown_threshold)
    
    # =========================================================================
    # Merge shadow probabilities with labels
    # =========================================================================
    # Align by date index
    p_risk_off = shadow_df["p_risk_off"].copy()
    
    # Create combined DataFrame with aligned dates
    combined = pd.DataFrame({
        "p_risk_off": p_risk_off,
        "y_risk_off": labels,
    })
    
    # Drop rows with missing/non-finite values
    combined = combined.dropna()
    combined = combined[np.isfinite(combined["p_risk_off"]) & np.isfinite(combined["y_risk_off"])]
    
    if len(combined) == 0:
        result_warnings.append("NO_VALID_OBSERVATIONS")
        # Write outputs with identity transform
        shadow_df["p_risk_off_cal"] = shadow_df["p_risk_off"]
        _write_csv(shadow_df, original_columns, output_dir)
        empty_metrics = _empty_metrics_struct()
        _write_json(output_dir, None, empty_metrics, result_warnings,
                    train_end, val_end, as_of_date, horizon_days, drawdown_threshold, n_bins, seed)
        return {"best_temperature": None, "metrics": empty_metrics, "warnings": result_warnings}
    
    # =========================================================================
    # Split by date
    # =========================================================================
    train_mask = combined.index <= train_end_dt
    val_mask = (combined.index > train_end_dt) & (combined.index <= val_end_dt)
    test_mask = (combined.index > val_end_dt) & (combined.index <= as_of_dt)
    
    train_df = combined[train_mask]
    val_df = combined[val_mask]
    test_df = combined[test_mask]
    
    if len(train_df) == 0:
        result_warnings.append("TRAIN_EMPTY")
    
    # =========================================================================
    # Compute uncalibrated metrics first
    # =========================================================================
    uncalibrated_metrics = {
        "train": _compute_split_metrics(train_df["y_risk_off"].values, train_df["p_risk_off"].values, n_bins) if len(train_df) > 0 else _compute_split_metrics(np.array([]), np.array([]), n_bins),
        "val": _compute_split_metrics(val_df["y_risk_off"].values, val_df["p_risk_off"].values, n_bins) if len(val_df) > 0 else _compute_split_metrics(np.array([]), np.array([]), n_bins),
        "test": _compute_split_metrics(test_df["y_risk_off"].values, test_df["p_risk_off"].values, n_bins) if len(test_df) > 0 else _compute_split_metrics(np.array([]), np.array([]), n_bins),
    }
    
    # =========================================================================
    # Fit temperature on VAL only
    # =========================================================================
    best_temperature = None
    
    if len(val_df) == 0:
        result_warnings.append("VAL_EMPTY:cannot_calibrate")
    else:
        val_p = val_df["p_risk_off"].values
        val_y = val_df["y_risk_off"].values
        val_logits = _compute_logits(val_p)
        
        best_temperature = _find_best_temperature(val_logits, val_y)
        
        if best_temperature is None:
            result_warnings.append("VAL_SINGLE_CLASS:cannot_calibrate")
    
    # =========================================================================
    # Apply calibration to all dates
    # =========================================================================
    all_logits = _compute_logits(shadow_df["p_risk_off"].values)
    
    if best_temperature is not None:
        p_calibrated = _sigmoid(all_logits / best_temperature)
    else:
        # Identity transform when calibration not possible
        p_calibrated = shadow_df["p_risk_off"].values.copy()
    
    shadow_df["p_risk_off_cal"] = p_calibrated
    
    # =========================================================================
    # Compute calibrated metrics for each split
    # =========================================================================
    def get_calibrated_split_p(split_df: pd.DataFrame) -> np.ndarray:
        if len(split_df) == 0:
            return np.array([])
        return shadow_df.loc[split_df.index, "p_risk_off_cal"].values
    
    calibrated_metrics = {
        "train": _compute_split_metrics(train_df["y_risk_off"].values, get_calibrated_split_p(train_df), n_bins) if len(train_df) > 0 else _compute_split_metrics(np.array([]), np.array([]), n_bins),
        "val": _compute_split_metrics(val_df["y_risk_off"].values, get_calibrated_split_p(val_df), n_bins) if len(val_df) > 0 else _compute_split_metrics(np.array([]), np.array([]), n_bins),
        "test": _compute_split_metrics(test_df["y_risk_off"].values, get_calibrated_split_p(test_df), n_bins) if len(test_df) > 0 else _compute_split_metrics(np.array([]), np.array([]), n_bins),
    }
    
    if len(test_df) == 0:
        result_warnings.append("TEST_EMPTY")
    
    # =========================================================================
    # Combine metrics
    # =========================================================================
    metrics = {
        "train": {"uncalibrated": uncalibrated_metrics["train"], "calibrated": calibrated_metrics["train"]},
        "val": {"uncalibrated": uncalibrated_metrics["val"], "calibrated": calibrated_metrics["val"]},
        "test": {"uncalibrated": uncalibrated_metrics["test"], "calibrated": calibrated_metrics["test"]},
    }
    
    # =========================================================================
    # Write outputs
    # =========================================================================
    _write_csv(shadow_df, original_columns, output_dir)
    _write_json(output_dir, best_temperature, metrics, result_warnings,
                train_end, val_end, as_of_date, horizon_days, drawdown_threshold, n_bins, seed)
    
    return {
        "best_temperature": best_temperature,
        "metrics": metrics,
        "warnings": result_warnings,
    }


def _empty_metrics_struct() -> dict:
    """Return empty metrics structure for edge cases."""
    empty = {"n_obs": 0, "base_rate": None, "brier": None, "log_loss": None, "roc_auc": None, "ece": None, "calibration_bins": []}
    return {
        "train": {"uncalibrated": empty.copy(), "calibrated": empty.copy()},
        "val": {"uncalibrated": empty.copy(), "calibrated": empty.copy()},
        "test": {"uncalibrated": empty.copy(), "calibrated": empty.copy()},
    }


def _write_csv(shadow_df: pd.DataFrame, original_columns: list, output_dir: str) -> None:
    """Write calibrated CSV with deterministic format."""
    # Ensure p_risk_off_cal is at the end
    out_columns = original_columns + ["p_risk_off_cal"]
    out_df = shadow_df[out_columns].copy()
    out_df = out_df.reset_index()
    
    output_csv_path = os.path.join(output_dir, "shadow_risk_calibrated_temp.csv")
    out_df.to_csv(output_csv_path, index=False, float_format="%.10f", lineterminator="\n")


def _write_json(
    output_dir: str,
    best_temperature: float | None,
    metrics: dict,
    warnings: list[str],
    train_end: str,
    val_end: str,
    as_of_date: str,
    horizon_days: int,
    drawdown_threshold: float,
    n_bins: int,
    seed: int,
) -> None:
    """Write metrics JSON with deterministic format."""
    output_json = {
        "schema_version": "9.6.2",
        "config": {
            "as_of_date": as_of_date,
            "drawdown_threshold": drawdown_threshold,
            "horizon_days": horizon_days,
            "n_bins": n_bins,
            "seed": seed,
            "train_end": train_end,
            "val_end": val_end,
        },
        "best_temperature": best_temperature,
        "metrics": metrics,
        "warnings": warnings,
    }
    
    output_json_path = os.path.join(output_dir, "shadow_risk_metrics_calibrated_temp.json")
    with open(output_json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(output_json, f, indent=2, sort_keys=True)
