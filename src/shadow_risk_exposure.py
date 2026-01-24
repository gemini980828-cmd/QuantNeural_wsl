"""
Shadow Risk/Exposure Gating ML Module for QUANT-NEURAL.

Produces a per-date "risk_off probability" and suggested exposure scalar.
This is SHADOW-ONLY: NO trading impact; outputs are for logging/analysis only.

Key Properties:
- PIT-Safe: Features at date t use only information available at or before t.
- Train-Only Fit: Scaler/model fit uses TRAIN only; val/test are transform/predict only.
- Deterministic: Same inputs + same seed => byte-identical CSV output.
- Fail-Safe: Never crash; write valid CSV with safe fallback if data issues.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _build_risk_features(
    prices: pd.DataFrame,
    spy_ticker: str,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Build PIT-safe risk features from SPY price series.
    
    Features:
    - ret_21d: 21-day return (pct_change)
    - ret_63d: 63-day return
    - mom_252d: 252-day momentum
    - vol_63d: 63-day rolling volatility of daily returns
    - dd_126d: Drawdown from 126-day rolling max
    
    All features use ONLY past information.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Get SPY series
    spy = prices[spy_ticker].copy()
    spy = spy[spy.index <= as_of_dt]
    spy = spy.sort_index()
    
    # Compute features
    daily_ret = spy.pct_change(1)
    
    features = pd.DataFrame(index=spy.index)
    features["ret_21d"] = spy.pct_change(21)
    features["ret_63d"] = spy.pct_change(63)
    features["mom_252d"] = spy.pct_change(252)
    features["vol_63d"] = daily_ret.rolling(63, min_periods=63).std()
    
    # Drawdown from 126-day rolling max
    rolling_max_126 = spy.rolling(126, min_periods=126).max()
    features["dd_126d"] = (spy / rolling_max_126) - 1
    
    return features


def _build_labels(
    prices: pd.DataFrame,
    spy_ticker: str,
    horizon_days: int,
    as_of_date: str,
) -> pd.Series:
    """
    Build risk_off labels based on forward returns.
    
    Label:
    - risk_off = 1 if forward return over horizon_days < 0
    - risk_off = 0 otherwise
    
    The last horizon_days rows are dropped (no forward label available).
    """
    as_of_dt = pd.to_datetime(as_of_date)
    
    spy = prices[spy_ticker].copy()
    spy = spy[spy.index <= as_of_dt]
    spy = spy.sort_index()
    
    # Forward return
    forward_ret = spy.shift(-horizon_days) / spy - 1
    
    # risk_off = 1 if forward return < 0, NaN if no forward data
    risk_off = (forward_ret < 0).astype("float")
    risk_off[forward_ret.isna()] = np.nan
    risk_off.name = "risk_off"
    
    return risk_off


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str,
    val_end: str,
    as_of_date: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into TRAIN / VAL / TEST by date.
    
    TRAIN: <= train_end
    VAL: (train_end, val_end]
    TEST: (val_end, as_of_date]
    """
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    as_of_dt = pd.to_datetime(as_of_date)
    
    train_mask = X.index <= train_end_dt
    val_mask = (X.index > train_end_dt) & (X.index <= val_end_dt)
    test_mask = (X.index > val_end_dt) & (X.index <= as_of_dt)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_shadow_risk_exposure_logit(
    prices: pd.DataFrame,
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    spy_ticker: str = "SPY",
    horizon_days: int = 63,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run shadow risk exposure logistic regression.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide price panel with DatetimeIndex and ticker columns.
    as_of_date : str
        PIT cutoff date (YYYY-MM-DD).
    train_end : str
        End of training period.
    val_end : str
        End of validation period.
    output_csv_path : str
        Path to write output CSV.
    spy_ticker : str
        Ticker to use for features (default: "SPY").
    horizon_days : int
        Forward horizon for risk label (default: 63).
    seed : int
        Random seed for reproducibility (default: 42).
    
    Returns
    -------
    pd.DataFrame
        Output dataframe with columns:
        - p_risk_off: Predicted probability of risk-off
        - w_beta_suggested: Suggested w_beta (capped at 0.35)
        - exposure_suggested: Suggested exposure (1 - p_risk_off)
        - ret_21d, ret_63d, mom_252d, vol_63d, dd_126d: Features
    """
    feature_cols = ["ret_21d", "ret_63d", "mom_252d", "vol_63d", "dd_126d"]
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    
    # =========================================================================
    # Fail-Safe: Check SPY ticker exists
    # =========================================================================
    if spy_ticker not in prices.columns:
        logger.warning(f"SHADOW_RISK_ML:spy_ticker '{spy_ticker}' not in prices columns")
        # Use prices index as reference for fallback dates
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        return _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="missing_spy",
            reference_dates=ref_dates,
        )
    
    # =========================================================================
    # Build features and labels
    # =========================================================================
    try:
        features = _build_risk_features(prices, spy_ticker, as_of_date)
        labels = _build_labels(prices, spy_ticker, horizon_days, as_of_date)
    except Exception as e:
        logger.warning(f"SHADOW_RISK_ML:feature_build_failed - {e}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        return _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="feature_build_failed",
            reference_dates=ref_dates,
        )
    
    # Align indices
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    
    # Drop the last horizon_days rows (no valid forward label)
    # These are rows where label is NaN due to shift
    valid_mask = labels.notna()
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    # Drop rows with NaN features
    combined = pd.concat([features, labels], axis=1).dropna()
    features = combined[feature_cols]
    labels = combined["risk_off"]
    
    # =========================================================================
    # Fail-Safe: Check sufficient data
    # =========================================================================
    if len(features) < 100:
        logger.warning(f"SHADOW_RISK_ML:insufficient_data - only {len(features)} rows")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        return _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="insufficient_data",
            reference_dates=ref_dates,
        )
    
    # =========================================================================
    # Split data
    # =========================================================================
    X_train, y_train, X_val, y_val, X_test, y_test = _split_data(
        features, labels, train_end, val_end, as_of_date
    )
    
    # =========================================================================
    # Fail-Safe: Check y_train has at least 2 classes
    # =========================================================================
    if len(X_train) < 10:
        logger.warning(f"SHADOW_RISK_ML:train_too_small - only {len(X_train)} train rows")
        ref_dates = features.index
        return _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="train_too_small",
            reference_dates=ref_dates,
        )
    
    unique_classes = y_train.nunique()
    if unique_classes < 2:
        logger.warning(f"SHADOW_RISK_ML:degenerate_labels - only {unique_classes} class(es) in train")
        base_rate = y_train.mean() if len(y_train) > 0 else 0.5
        ref_dates = features.index
        return _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="degenerate_labels",
            fallback_p=base_rate,
            reference_dates=ref_dates,
        )
    
    # =========================================================================
    # Fit model (TRAIN-ONLY)
    # =========================================================================
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        )),
    ])
    
    pipeline.fit(X_train.values, y_train.values.astype(int))
    
    # =========================================================================
    # Predict on VAL + TEST
    # =========================================================================
    X_all = pd.concat([X_train, X_val, X_test])
    X_all = X_all.sort_index()
    
    # Only predict for VAL + TEST dates (but include train for completeness)
    proba = pipeline.predict_proba(X_all.values)[:, 1]
    
    # =========================================================================
    # Build output dataframe
    # =========================================================================
    output = X_all.copy()
    output["p_risk_off"] = proba
    output["w_beta_suggested"] = np.clip(proba, 0.0, 0.35)
    output["exposure_suggested"] = np.clip(1.0 - proba, 0.0, 1.0)
    
    # Filter to VAL + TEST only for the final output
    output_mask = output.index > train_end_dt
    output = output[output_mask].copy()
    
    # Ensure column order for determinism
    output_cols = ["p_risk_off", "w_beta_suggested", "exposure_suggested"] + feature_cols
    output = output[output_cols]
    
    # Ensure finite values
    output = output.fillna(0.0)
    output = output.replace([np.inf, -np.inf], 0.0)
    
    # Sort index for determinism
    output = output.sort_index()
    output.index.name = "date"
    
    # =========================================================================
    # Write CSV (deterministic format)
    # =========================================================================
    output.to_csv(output_csv_path, float_format="%.10f")
    
    return output


def _write_fallback_csv(
    output_csv_path: str,
    train_end: str,
    as_of_date: str,
    feature_cols: list[str],
    reason: str,
    fallback_p: float = 0.5,
    reference_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Write a fallback CSV with safe default values.
    
    Called when fail-safe conditions are triggered.
    Covers (train_end, as_of_date] dates (VAL+TEST region).
    """
    train_end_dt = pd.to_datetime(train_end)
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Use reference dates if provided, otherwise generate monthly
    if reference_dates is not None and len(reference_dates) > 0:
        # Ensure reference_dates is a DatetimeIndex
        ref_idx = pd.DatetimeIndex(reference_dates)
        # Filter to (train_end, as_of_date] range
        dates = ref_idx[(ref_idx > train_end_dt) & (ref_idx <= as_of_dt)]
    else:
        # Fallback to monthly dates from train_end to as_of_date
        dates = pd.date_range(start=train_end_dt, end=as_of_dt, freq="ME")
    
    if len(dates) == 0:
        dates = pd.DatetimeIndex([as_of_dt])
    
    output = pd.DataFrame(index=dates)
    output["p_risk_off"] = fallback_p
    output["w_beta_suggested"] = 0.0
    output["exposure_suggested"] = 1.0
    
    for col in feature_cols:
        output[col] = 0.0
    
    output.index.name = "date"
    output = output.sort_index()
    
    output.to_csv(output_csv_path, float_format="%.10f")
    
    logger.warning(f"SHADOW_RISK_ML:fallback_csv_written - reason={reason}")
    
    return output


def _compute_calibration_bins(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Compute calibration bins for predicted probabilities.
    
    Returns list of dicts with: count, mean_pred, frac_pos
    """
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= low) & (y_proba <= high)
        else:
            mask = (y_proba >= low) & (y_proba < high)
        
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(round(y_proba[mask].mean(), 10))
            frac_pos = float(round(y_true[mask].mean(), 10))
        else:
            mean_pred = float(round((low + high) / 2, 10))
            frac_pos = None
        
        bins.append({
            "bin_idx": i,
            "count": count,
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
        })
    
    return bins


def _compute_ece(calibration_bins: list[dict], n_total: int) -> float | None:
    """
    Compute Expected Calibration Error from calibration bins.
    
    ECE = sum(|frac_pos - mean_pred| * count) / n_total
    """
    if n_total == 0:
        return None
    
    ece = 0.0
    for b in calibration_bins:
        if b["frac_pos"] is not None and b["count"] > 0:
            ece += abs(b["frac_pos"] - b["mean_pred"]) * b["count"]
    
    return float(round(ece / n_total, 10))


def _compute_split_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    split_name: str,
) -> dict:
    """
    Compute metrics for a single split (TRAIN/VAL/TEST).
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
    
    n_obs = int(len(y_true))
    
    if n_obs == 0:
        return {
            "n_obs": 0,
            "base_rate": None,
            "brier": None,
            "roc_auc": None,
            "log_loss": None,
            "calibration_bins": [],
            "ece": None,
        }
    
    base_rate = float(round(y_true.mean(), 10))
    
    # Brier score
    try:
        brier = float(round(brier_score_loss(y_true, y_proba), 10))
    except Exception:
        brier = None
    
    # ROC AUC (requires at least 2 classes)
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(f"SHADOW_RISK_METRICS:single_class_{split_name} - cannot compute roc_auc")
        roc_auc = None
    else:
        try:
            roc_auc = float(round(roc_auc_score(y_true, y_proba), 10))
        except Exception:
            roc_auc = None
    
    # Log loss
    if len(unique_classes) < 2:
        logloss = None
    else:
        try:
            logloss = float(round(log_loss(y_true, y_proba), 10))
        except Exception:
            logloss = None
    
    # Calibration bins
    calibration_bins = _compute_calibration_bins(y_true, y_proba)
    
    # ECE
    ece = _compute_ece(calibration_bins, n_obs)
    
    return {
        "n_obs": n_obs,
        "base_rate": base_rate,
        "brier": brier,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "calibration_bins": calibration_bins,
        "ece": ece,
    }


def run_shadow_risk_exposure_logit_with_metrics(
    prices: pd.DataFrame,
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    output_metrics_json_path: str,
    spy_ticker: str = "SPY",
    horizon_days: int = 63,
    seed: int = 42,
) -> dict:
    """
    Run shadow risk exposure logistic regression with metrics export.
    
    Produces the SAME CSV as run_shadow_risk_exposure_logit() plus a JSON
    file with classification quality and calibration metrics per split.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide price panel with DatetimeIndex and ticker columns.
    as_of_date : str
        PIT cutoff date (YYYY-MM-DD).
    train_end : str
        End of training period.
    val_end : str
        End of validation period.
    output_csv_path : str
        Path to write output CSV.
    output_metrics_json_path : str
        Path to write metrics JSON.
    spy_ticker : str
        Ticker to use for features (default: "SPY").
    horizon_days : int
        Forward horizon for risk label (default: 63).
    seed : int
        Random seed for reproducibility (default: 42).
    
    Returns
    -------
    dict
        Dictionary with:
        - csv_df: The output DataFrame (same as CSV content)
        - metrics: The metrics dictionary (same as JSON content)
    """
    import json
    
    feature_cols = ["ret_21d", "ret_63d", "mom_252d", "vol_63d", "dd_126d"]
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_metrics_json_path) or ".", exist_ok=True)
    
    # Initialize metrics structure
    config = {
        "as_of_date": as_of_date,
        "train_end": train_end,
        "val_end": val_end,
        "spy_ticker": spy_ticker,
        "horizon_days": horizon_days,
        "seed": seed,
    }
    
    metrics = {
        "schema_version": "9.5.2",
        "config": config,
        "train": None,
        "val": None,
        "test": None,
    }
    
    # =========================================================================
    # Fail-Safe: Check SPY ticker exists
    # =========================================================================
    if spy_ticker not in prices.columns:
        logger.warning(f"SHADOW_RISK_ML:spy_ticker '{spy_ticker}' not in prices columns")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="missing_spy",
            reference_dates=ref_dates,
        )
        logger.warning("SHADOW_RISK_METRICS:fallback - missing_spy")
        _write_fallback_metrics_json(output_metrics_json_path, metrics, "missing_spy")
        return {"csv_df": csv_df, "metrics": metrics}
    
    # =========================================================================
    # Build features and labels
    # =========================================================================
    try:
        features = _build_risk_features(prices, spy_ticker, as_of_date)
        labels = _build_labels(prices, spy_ticker, horizon_days, as_of_date)
    except Exception as e:
        logger.warning(f"SHADOW_RISK_ML:feature_build_failed - {e}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="feature_build_failed",
            reference_dates=ref_dates,
        )
        logger.warning("SHADOW_RISK_METRICS:fallback - feature_build_failed")
        _write_fallback_metrics_json(output_metrics_json_path, metrics, "feature_build_failed")
        return {"csv_df": csv_df, "metrics": metrics}
    
    # Align indices
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    
    # Drop rows with NaN labels (last horizon_days)
    valid_mask = labels.notna()
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    # Drop rows with NaN features
    combined = pd.concat([features, labels], axis=1).dropna()
    features = combined[feature_cols]
    labels = combined["risk_off"]
    
    # =========================================================================
    # Fail-Safe: Check sufficient data
    # =========================================================================
    if len(features) < 100:
        logger.warning(f"SHADOW_RISK_ML:insufficient_data - only {len(features)} rows")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="insufficient_data",
            reference_dates=ref_dates,
        )
        logger.warning("SHADOW_RISK_METRICS:fallback - insufficient_data")
        _write_fallback_metrics_json(output_metrics_json_path, metrics, "insufficient_data")
        return {"csv_df": csv_df, "metrics": metrics}
    
    # =========================================================================
    # Split data
    # =========================================================================
    X_train, y_train, X_val, y_val, X_test, y_test = _split_data(
        features, labels, train_end, val_end, as_of_date
    )
    
    # =========================================================================
    # Fail-Safe: Check y_train has at least 2 classes
    # =========================================================================
    if len(X_train) < 10:
        logger.warning(f"SHADOW_RISK_ML:train_too_small - only {len(X_train)} train rows")
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="train_too_small",
            reference_dates=ref_dates,
        )
        logger.warning("SHADOW_RISK_METRICS:fallback - train_too_small")
        _write_fallback_metrics_json(output_metrics_json_path, metrics, "train_too_small")
        return {"csv_df": csv_df, "metrics": metrics}
    
    unique_classes = y_train.nunique()
    if unique_classes < 2:
        logger.warning(f"SHADOW_RISK_ML:degenerate_labels - only {unique_classes} class(es) in train")
        base_rate = y_train.mean() if len(y_train) > 0 else 0.5
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason="degenerate_labels",
            fallback_p=base_rate,
            reference_dates=ref_dates,
        )
        logger.warning("SHADOW_RISK_METRICS:fallback - degenerate_labels")
        _write_fallback_metrics_json(output_metrics_json_path, metrics, "degenerate_labels")
        return {"csv_df": csv_df, "metrics": metrics}
    
    # =========================================================================
    # Fit model (TRAIN-ONLY)
    # =========================================================================
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        )),
    ])
    
    pipeline.fit(X_train.values, y_train.values.astype(int))
    
    # =========================================================================
    # Predict on all splits
    # =========================================================================
    proba_train = pipeline.predict_proba(X_train.values)[:, 1] if len(X_train) > 0 else np.array([])
    proba_val = pipeline.predict_proba(X_val.values)[:, 1] if len(X_val) > 0 else np.array([])
    proba_test = pipeline.predict_proba(X_test.values)[:, 1] if len(X_test) > 0 else np.array([])
    
    # =========================================================================
    # Compute metrics per split
    # =========================================================================
    metrics["train"] = _compute_split_metrics(y_train.values.astype(int), proba_train, "train")
    metrics["val"] = _compute_split_metrics(y_val.values.astype(int), proba_val, "val")
    metrics["test"] = _compute_split_metrics(y_test.values.astype(int), proba_test, "test")
    
    # =========================================================================
    # Build output CSV (same as original function)
    # =========================================================================
    X_all = pd.concat([X_train, X_val, X_test])
    X_all = X_all.sort_index()
    
    proba = pipeline.predict_proba(X_all.values)[:, 1]
    
    output = X_all.copy()
    output["p_risk_off"] = proba
    output["w_beta_suggested"] = np.clip(proba, 0.0, 0.35)
    output["exposure_suggested"] = np.clip(1.0 - proba, 0.0, 1.0)
    
    # Filter to VAL + TEST only for the final output
    output_mask = output.index > train_end_dt
    output = output[output_mask].copy()
    
    # Ensure column order for determinism
    output_cols = ["p_risk_off", "w_beta_suggested", "exposure_suggested"] + feature_cols
    output = output[output_cols]
    
    # Ensure finite values
    output = output.fillna(0.0)
    output = output.replace([np.inf, -np.inf], 0.0)
    
    # Sort index for determinism
    output = output.sort_index()
    output.index.name = "date"
    
    # Write CSV
    output.to_csv(output_csv_path, float_format="%.10f")
    
    # Write metrics JSON (deterministic)
    _write_metrics_json(output_metrics_json_path, metrics)
    
    return {"csv_df": output, "metrics": metrics}


def run_shadow_risk_exposure_mlp_with_metrics(
    prices: pd.DataFrame,
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    output_metrics_json_path: str,
    spy_ticker: str = "SPY",
    horizon_days: int = 63,
    seed: int = 42,
    hidden_layer_sizes: tuple = (8, 4),
    alpha: float = 1e-2,
    max_iter: int = 1000,
    tol: float = 1e-2,
) -> dict:
    """
    Run shadow risk exposure MLP classifier with metrics export.
    
    Uses the SAME PIT-safe features/labels/splits as the logistic version,
    but trains an MLP (sklearn.neural_network.MLPClassifier) instead.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide price panel with DatetimeIndex and ticker columns.
    as_of_date : str
        PIT cutoff date (YYYY-MM-DD).
    train_end : str
        End of training period.
    val_end : str
        End of validation period.
    output_csv_path : str
        Path to write output CSV.
    output_metrics_json_path : str
        Path to write metrics JSON.
    spy_ticker : str
        Ticker to use for features (default: "SPY").
    horizon_days : int
        Forward horizon for risk label (default: 63).
    seed : int
        Random seed for reproducibility (default: 42).
    hidden_layer_sizes : tuple
        Hidden layer sizes for MLP (default: (8, 4)).
    alpha : float
        L2 regularization (default: 1e-2).
    max_iter : int
        Max iterations for MLP (default: 1000).
    tol : float
        Optimization tolerance for MLP (default: 1e-2).
    
    Returns
    -------
    dict
        Dictionary with csv_df, metrics, and warnings.
    """
    import json
    from sklearn.neural_network import MLPClassifier
    
    feature_cols = ["ret_21d", "ret_63d", "mom_252d", "vol_63d", "dd_126d"]
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_metrics_json_path) or ".", exist_ok=True)
    
    warnings_list = []
    
    # Initialize metrics structure
    config = {
        "as_of_date": as_of_date,
        "train_end": train_end,
        "val_end": val_end,
        "spy_ticker": spy_ticker,
        "horizon_days": horizon_days,
        "seed": seed,
    }
    model_params = {
        "model_type": "mlp",
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": seed,
    }
    
    metrics = {
        "schema_version": "9.6.12.1",
        "config": config,
        "model_params": model_params,
        "train": None,
        "val": None,
        "test": None,
    }

    
    # =========================================================================
    # Fail-Safe: Check SPY ticker exists
    # =========================================================================
    if spy_ticker not in prices.columns:
        reason = "missing_spy"
        logger.warning(f"SHADOW_RISK_MLP:{reason}")
        warnings_list.append(f"SHADOW_RISK_MLP:{reason}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        _write_fallback_metrics_json_mlp(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}
    
    # =========================================================================
    # Build features and labels
    # =========================================================================
    try:
        features = _build_risk_features(prices, spy_ticker, as_of_date)
        labels = _build_labels(prices, spy_ticker, horizon_days, as_of_date)
    except Exception as e:
        reason = "feature_build_failed"
        logger.warning(f"SHADOW_RISK_MLP:{reason}")
        warnings_list.append(f"SHADOW_RISK_MLP:{reason}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        _write_fallback_metrics_json_mlp(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}
    
    # Align indices
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    
    # Drop rows with NaN labels (last horizon_days)
    valid_mask = labels.notna()
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    # Drop rows with NaN features
    combined = pd.concat([features, labels], axis=1).dropna()
    features = combined[feature_cols]
    labels = combined["risk_off"]
    
    # =========================================================================
    # Fail-Safe: Check sufficient data
    # =========================================================================
    if len(features) < 100:
        reason = "insufficient_data"
        logger.warning(f"SHADOW_RISK_MLP:{reason} - only {len(features)} rows")
        warnings_list.append(f"SHADOW_RISK_MLP:{reason}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        _write_fallback_metrics_json_mlp(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}
    
    # =========================================================================
    # Split data
    # =========================================================================
    X_train, y_train, X_val, y_val, X_test, y_test = _split_data(
        features, labels, train_end, val_end, as_of_date
    )
    
    # =========================================================================
    # Fail-Safe: Check y_train has at least 2 classes
    # =========================================================================
    if len(X_train) < 10:
        reason = "train_too_small"
        logger.warning(f"SHADOW_RISK_MLP:{reason} - only {len(X_train)} train rows")
        warnings_list.append(f"SHADOW_RISK_MLP:{reason}")
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        _write_fallback_metrics_json_mlp(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}
    
    unique_classes = y_train.nunique()
    if unique_classes < 2:
        reason = "degenerate_labels"
        logger.warning(f"SHADOW_RISK_MLP:{reason} - only {unique_classes} class(es) in train")
        warnings_list.append(f"SHADOW_RISK_MLP:{reason}")
        base_rate = y_train.mean() if len(y_train) > 0 else 0.5
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            fallback_p=base_rate,
            reference_dates=ref_dates,
        )
        _write_fallback_metrics_json_mlp(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}
    
    # =========================================================================
    # Fit model (TRAIN-ONLY)
    # =========================================================================
    try:
        # StandardScaler fit on TRAIN only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_val_scaled = scaler.transform(X_val.values) if len(X_val) > 0 else np.array([]).reshape(0, len(feature_cols))
        X_test_scaled = scaler.transform(X_test.values) if len(X_test) > 0 else np.array([]).reshape(0, len(feature_cols))
        
        # MLP with deterministic settings
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            solver="lbfgs",  # Deterministic solver
        )
        mlp.fit(X_train_scaled, y_train.values.astype(int))
        
    except Exception as e:
        reason = "model_fit_failed"
        logger.warning(f"SHADOW_RISK_MLP:{reason}")
        warnings_list.append(f"SHADOW_RISK_MLP:{reason}")
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        _write_fallback_metrics_json_mlp(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}
    
    # =========================================================================
    # Predict on all splits
    # =========================================================================
    proba_train = mlp.predict_proba(X_train_scaled)[:, 1] if len(X_train) > 0 else np.array([])
    proba_val = mlp.predict_proba(X_val_scaled)[:, 1] if len(X_val) > 0 else np.array([])
    proba_test = mlp.predict_proba(X_test_scaled)[:, 1] if len(X_test) > 0 else np.array([])
    
    # =========================================================================
    # Compute metrics per split
    # =========================================================================
    metrics["train"] = _compute_split_metrics(y_train.values.astype(int), proba_train, "train")
    metrics["val"] = _compute_split_metrics(y_val.values.astype(int), proba_val, "val")
    metrics["test"] = _compute_split_metrics(y_test.values.astype(int), proba_test, "test")
    
    # Add warnings if single-class splits
    if metrics["val"].get("roc_auc") is None and len(y_val) > 0:
        warnings_list.append("SHADOW_RISK_MLP:val_single_class")
    if metrics["test"].get("roc_auc") is None and len(y_test) > 0:
        warnings_list.append("SHADOW_RISK_MLP:test_single_class")
    
    # =========================================================================
    # Build output CSV (same format as logit version)
    # =========================================================================
    X_all = pd.concat([X_train, X_val, X_test])
    X_all = X_all.sort_index()
    
    X_all_scaled = scaler.transform(X_all.values)
    proba = mlp.predict_proba(X_all_scaled)[:, 1]
    
    output = X_all.copy()
    output["p_risk_off"] = proba
    output["w_beta_suggested"] = np.clip(proba, 0.0, 0.35)
    output["exposure_suggested"] = np.clip(1.0 - proba, 0.0, 1.0)
    
    # Filter to VAL + TEST only for the final output
    output_mask = output.index > train_end_dt
    output = output[output_mask].copy()
    
    # Ensure column order for determinism
    output_cols = ["p_risk_off", "w_beta_suggested", "exposure_suggested"] + feature_cols
    output = output[output_cols]
    
    # Ensure finite values
    output = output.fillna(0.0)
    output = output.replace([np.inf, -np.inf], 0.0)
    
    # Sort index for determinism
    output = output.sort_index()
    output.index.name = "date"
    
    # Write CSV with deterministic format
    output.to_csv(output_csv_path, float_format="%.10f", lineterminator="\n")
    
    # Write metrics JSON (deterministic)
    _write_metrics_json_mlp(output_metrics_json_path, metrics)
    
    return {"csv_df": output, "metrics": metrics, "warnings": warnings_list}


def run_shadow_risk_exposure_xgb_with_metrics(
    *,
    prices: "pd.DataFrame",
    as_of_date: str,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    output_metrics_json_path: str,
    spy_ticker: str,
    horizon_days: int,
    seed: int = 42,
) -> dict:
    """
    Run shadow risk exposure using XGBClassifier (with fallback).

    Mirrors the logit/MLP contract: same features, PIT discipline, and artifacts.
    """
    feature_cols = ["ret_21d", "ret_63d", "mom_252d", "vol_63d", "dd_126d"]
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    as_of_dt = pd.to_datetime(as_of_date)

    # Ensure directories exist
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_metrics_json_path) or ".", exist_ok=True)

    warnings_list: list[str] = []

    # Model backend selection (optional import + deterministic fallback)
    backend = "sklearn_hist_gb"
    model = None

    xgb_params = {
        "random_state": seed,
        "n_jobs": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "verbosity": 0,
    }
    hist_params = {
        "loss": "log_loss",
        "max_depth": 3,
        "learning_rate": 0.05,
        "max_iter": 300,
        "random_state": seed,
        "early_stopping": False,
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        try:
            model = XGBClassifier(**xgb_params)
            backend = "xgboost"
        except Exception:
            warnings_list.append("SR_XGB_WARN:param_adjusted")
            backend = "sklearn_hist_gb"
            model = None
    except Exception:
        warnings_list.append("SR_XGB_FALLBACK:xgboost_unavailable")
        backend = "sklearn_hist_gb"
        model = None

    if model is None:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(**hist_params)
        except Exception:
            reason = "backend_init_failed"
            warnings_list.append(f"SR_XGB_FAIL:{reason}")
            metrics = _build_xgb_metrics_stub(
                as_of_date=as_of_date,
                train_end=train_end,
                val_end=val_end,
                spy_ticker=spy_ticker,
                horizon_days=horizon_days,
                seed=seed,
                backend=backend,
                xgb_params=xgb_params,
                hist_params=hist_params,
                warnings=warnings_list,
            )
            csv_df = _write_fallback_csv(
                output_csv_path=output_csv_path,
                train_end=train_end,
                as_of_date=as_of_date,
                feature_cols=feature_cols,
                reason=reason,
                reference_dates=pd.DatetimeIndex(prices.index),
            )
            _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
            return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    # Initialize metrics structure
    metrics = _build_xgb_metrics_stub(
        as_of_date=as_of_date,
        train_end=train_end,
        val_end=val_end,
        spy_ticker=spy_ticker,
        horizon_days=horizon_days,
        seed=seed,
        backend=backend,
        xgb_params=xgb_params,
        hist_params=hist_params,
        warnings=warnings_list,
    )

    # =========================================================================
    # Fail-Safe: Check SPY ticker exists
    # =========================================================================
    if spy_ticker not in prices.columns:
        reason = "missing_spy"
        warnings_list.append(f"SR_XGB_FAIL:{reason}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        metrics["warnings"] = warnings_list
        _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    # =========================================================================
    # Build features and labels
    # =========================================================================
    try:
        features = _build_risk_features(prices, spy_ticker, as_of_date)
        labels = _build_labels(prices, spy_ticker, horizon_days, as_of_date)
    except Exception:
        reason = "feature_build_failed"
        warnings_list.append(f"SR_XGB_FAIL:{reason}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        metrics["warnings"] = warnings_list
        _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    # Align indices
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Drop rows with NaN labels (last horizon_days)
    valid_mask = labels.notna()
    features = features[valid_mask]
    labels = labels[valid_mask]

    # Drop rows with NaN features
    combined = pd.concat([features, labels], axis=1).dropna()
    features = combined[feature_cols]
    labels = combined["risk_off"]

    # =========================================================================
    # Fail-Safe: Check sufficient data
    # =========================================================================
    if len(features) < 100:
        reason = "insufficient_data"
        warnings_list.append(f"SR_XGB_FAIL:{reason}")
        price_idx = pd.DatetimeIndex(prices.index)
        ref_dates = price_idx[price_idx <= as_of_dt]
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        metrics["warnings"] = warnings_list
        _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    # =========================================================================
    # Split data
    # =========================================================================
    X_train, y_train, X_val, y_val, X_test, y_test = _split_data(
        features, labels, train_end, val_end, as_of_date
    )

    # =========================================================================
    # Fail-Safe: Check y_train has at least 2 classes
    # =========================================================================
    if len(X_train) < 10:
        reason = "train_too_small"
        warnings_list.append(f"SR_XGB_FAIL:{reason}")
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        metrics["warnings"] = warnings_list
        _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    unique_classes = y_train.nunique()
    if unique_classes < 2:
        reason = "degenerate_labels"
        warnings_list.append(f"SR_XGB_FAIL:{reason}")
        base_rate = y_train.mean() if len(y_train) > 0 else 0.5
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            fallback_p=base_rate,
            reference_dates=ref_dates,
        )
        metrics["warnings"] = warnings_list
        _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    # =========================================================================
    # Fit model (TRAIN-ONLY)
    # =========================================================================
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_val_scaled = scaler.transform(X_val.values) if len(X_val) > 0 else np.array([]).reshape(0, len(feature_cols))
        X_test_scaled = scaler.transform(X_test.values) if len(X_test) > 0 else np.array([]).reshape(0, len(feature_cols))

        model.fit(X_train_scaled, y_train.values.astype(int))
    except Exception:
        reason = "model_fit_failed"
        warnings_list.append(f"SR_XGB_FAIL:{reason}")
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        metrics["warnings"] = warnings_list
        _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    # =========================================================================
    # Predict on all splits
    # =========================================================================
    try:
        proba_train = model.predict_proba(X_train_scaled)[:, 1] if len(X_train) > 0 else np.array([])
        proba_val = model.predict_proba(X_val_scaled)[:, 1] if len(X_val) > 0 else np.array([])
        proba_test = model.predict_proba(X_test_scaled)[:, 1] if len(X_test) > 0 else np.array([])
    except Exception:
        reason = "predict_failed"
        warnings_list.append(f"SR_XGB_FAIL:{reason}")
        ref_dates = features.index
        csv_df = _write_fallback_csv(
            output_csv_path=output_csv_path,
            train_end=train_end,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
            reason=reason,
            reference_dates=ref_dates,
        )
        metrics["warnings"] = warnings_list
        _write_fallback_metrics_json_xgb(output_metrics_json_path, metrics, reason)
        return {"csv_df": csv_df, "metrics": metrics, "warnings": warnings_list, "fallback_reason": reason}

    # =========================================================================
    # Compute metrics per split
    # =========================================================================
    metrics["train"] = _compute_split_metrics(y_train.values.astype(int), proba_train, "train")
    metrics["val"] = _compute_split_metrics(y_val.values.astype(int), proba_val, "val")
    metrics["test"] = _compute_split_metrics(y_test.values.astype(int), proba_test, "test")

    if metrics["val"].get("roc_auc") is None and len(y_val) > 0:
        warnings_list.append("SR_XGB_WARN:val_single_class")
    if metrics["test"].get("roc_auc") is None and len(y_test) > 0:
        warnings_list.append("SR_XGB_WARN:test_single_class")

    # =========================================================================
    # Build output CSV (same format as logit version)
    # =========================================================================
    X_all = pd.concat([X_train, X_val, X_test])
    X_all = X_all.sort_index()

    X_all_scaled = scaler.transform(X_all.values)
    proba = model.predict_proba(X_all_scaled)[:, 1]

    output = X_all.copy()
    output["p_risk_off"] = proba
    output["w_beta_suggested"] = np.clip(proba, 0.0, 0.35)
    output["exposure_suggested"] = np.clip(1.0 - proba, 0.0, 1.0)

    # Filter to VAL + TEST only for the final output
    output_mask = output.index > train_end_dt
    output = output[output_mask].copy()

    # Ensure column order for determinism
    output_cols = ["p_risk_off", "w_beta_suggested", "exposure_suggested"] + feature_cols
    output = output[output_cols]

    # Ensure finite values
    output = output.fillna(0.0)
    output = output.replace([np.inf, -np.inf], 0.0)

    # Sort index for determinism
    output = output.sort_index()
    output.index.name = "date"

    output.to_csv(output_csv_path, float_format="%.10f", lineterminator="\n")

    metrics["warnings"] = warnings_list
    _write_metrics_json_xgb(output_metrics_json_path, metrics)

    return {"csv_df": output, "metrics": metrics, "warnings": warnings_list}


def _write_fallback_metrics_json_mlp(output_path: str, metrics: dict, reason: str) -> None:
    """Write fallback metrics JSON for MLP with nulls."""
    import json
    
    empty_split = {
        "n_obs": 0,
        "base_rate": None,
        "brier": None,
        "log_loss": None,
        "roc_auc": None,
        "ece": None,
        "calibration_bins": [],
    }
    
    metrics["train"] = empty_split.copy()
    metrics["val"] = empty_split.copy()
    metrics["test"] = empty_split.copy()
    metrics["fallback_reason"] = reason
    
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def _write_metrics_json_mlp(output_path: str, metrics: dict) -> None:
    """Write metrics JSON for MLP with deterministic serialization."""
    import json
    
    def round_floats(obj, decimals=10):
        if isinstance(obj, float):
            return round(obj, decimals)
        elif isinstance(obj, dict):
            return {k: round_floats(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(x, decimals) for x in obj]
        return obj
    
    metrics_rounded = round_floats(metrics)

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics_rounded, f, indent=2, sort_keys=True)


def _build_xgb_metrics_stub(
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    spy_ticker: str,
    horizon_days: int,
    seed: int,
    backend: str,
    xgb_params: dict,
    hist_params: dict,
    warnings: list[str],
) -> dict:
    """Initialize metrics structure for XGB variant."""
    config = {
        "as_of_date": as_of_date,
        "train_end": train_end,
        "val_end": val_end,
        "spy_ticker": spy_ticker,
        "horizon_days": horizon_days,
        "seed": seed,
    }

    model_params = {
        "model_type": "xgb",
        "backend": backend,
        "seed": seed,
        "random_state": seed,
    }

    if backend == "xgboost":
        model_params.update({k: xgb_params[k] for k in sorted(xgb_params)})
    else:
        model_params.update({k: hist_params[k] for k in sorted(hist_params)})

    return {
        "schema_version": "9.6.13",
        "config": config,
        "model_params": model_params,
        "train": None,
        "val": None,
        "test": None,
        "warnings": list(warnings),
    }


def _write_fallback_metrics_json_xgb(output_path: str, metrics: dict, reason: str) -> None:
    """Write fallback metrics JSON for XGB with nulls."""
    empty_split = {
        "n_obs": 0,
        "base_rate": None,
        "brier": None,
        "log_loss": None,
        "roc_auc": None,
        "ece": None,
        "calibration_bins": [],
    }

    metrics["train"] = empty_split.copy()
    metrics["val"] = empty_split.copy()
    metrics["test"] = empty_split.copy()
    metrics["fallback_reason"] = reason

    _write_metrics_json_xgb(output_path, metrics)


def _write_metrics_json_xgb(output_path: str, metrics: dict) -> None:
    """Write metrics JSON with deterministic serialization."""
    import json

    json_str = json.dumps(
        metrics,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json_str)

def _write_fallback_metrics_json(
    output_path: str,
    metrics: dict,
    reason: str,
) -> None:
    """Write fallback metrics JSON with nulls."""
    import json
    
    fallback_split = {
        "n_obs": 0,
        "base_rate": None,
        "brier": None,
        "roc_auc": None,
        "log_loss": None,
        "calibration_bins": [],
        "ece": None,
    }
    
    metrics["train"] = fallback_split.copy()
    metrics["val"] = fallback_split.copy()
    metrics["test"] = fallback_split.copy()
    metrics["fallback_reason"] = reason
    
    _write_metrics_json(output_path, metrics)


def _write_metrics_json(output_path: str, metrics: dict) -> None:
    """Write metrics JSON with deterministic serialization."""
    import json
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


# =============================================================================
# Overlay Backtest Functions
# =============================================================================

def _compute_overlay_split_metrics(
    overlay_ret: pd.Series,
    exposure: pd.Series,
    split_name: str,
    ann_factor: float = 252,
) -> dict:
    """
    Compute overlay metrics for a single split.
    """
    n_obs = int(len(overlay_ret))
    
    if n_obs == 0:
        return {
            "n_obs": 0,
            "total_return": None,
            "cagr": None,
            "ann_vol": None,
            "cagr_over_vol": None,
            "max_drawdown": None,
            "avg_exposure": None,
            "turnover_exposure": None,
        }
    
    # Total return
    equity = (1 + overlay_ret).cumprod()
    total_return = float(round(equity.iloc[-1] - 1, 10)) if len(equity) > 0 else None
    
    # CAGR
    years = n_obs / ann_factor
    if years > 0 and equity.iloc[-1] > 0:
        cagr = float(round((equity.iloc[-1] ** (1 / years)) - 1, 10))
    else:
        cagr = None
    
    # Annualized volatility
    ann_vol = float(round(overlay_ret.std() * np.sqrt(ann_factor), 10)) if n_obs > 1 else None
    
    # CAGR / Vol
    if cagr is not None and ann_vol is not None and ann_vol > 0:
        cagr_over_vol = float(round(cagr / ann_vol, 10))
    else:
        cagr_over_vol = None
    
    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(round(drawdown.min(), 10)) if len(drawdown) > 0 else None
    
    # Avg exposure
    avg_exposure = float(round(exposure.mean(), 10)) if len(exposure) > 0 else None
    
    # Turnover (sum of absolute changes in exposure)
    if len(exposure) > 1:
        turnover_exposure = float(round(exposure.diff().abs().sum(), 10))
    else:
        turnover_exposure = 0.0
    
    return {
        "n_obs": n_obs,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "cagr_over_vol": cagr_over_vol,
        "max_drawdown": max_drawdown,
        "avg_exposure": avg_exposure,
        "turnover_exposure": turnover_exposure,
    }


def run_shadow_risk_overlay_spy_only(
    prices: pd.DataFrame,
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    shadow_csv_path: str,
    output_overlay_csv_path: str,
    output_overlay_metrics_json_path: str,
    spy_ticker: str = "SPY",
    cash_daily_return: float = 0.0,
) -> dict:
    """
    Run SPY-only overlay backtest using shadow risk exposure signal.
    
    Uses exposure_suggested from shadow CSV to form a simple SPY/cash overlay.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide price panel with DatetimeIndex and ticker columns.
    as_of_date : str
        PIT cutoff date (YYYY-MM-DD).
    train_end : str
        End of training period.
    val_end : str
        End of validation period.
    shadow_csv_path : str
        Path to the shadow risk CSV (must exist).
    output_overlay_csv_path : str
        Path to write overlay time series CSV.
    output_overlay_metrics_json_path : str
        Path to write overlay metrics JSON.
    spy_ticker : str
        Ticker for SPY returns (default: "SPY").
    cash_daily_return : float
        Daily return for cash allocation (default: 0.0).
    
    Returns
    -------
    dict
        Dictionary with:
        - overlay_df: The overlay DataFrame (same as CSV content)
        - overlay_metrics: The metrics dictionary (same as JSON content)
    """
    import json
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_overlay_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_overlay_metrics_json_path) or ".", exist_ok=True)
    
    # Initialize metrics structure
    config = {
        "as_of_date": as_of_date,
        "train_end": train_end,
        "val_end": val_end,
        "spy_ticker": spy_ticker,
        "cash_daily_return": cash_daily_return,
        "shadow_csv_path": shadow_csv_path,
    }
    
    overlay_metrics = {
        "schema_version": "9.5.3",
        "config": config,
        "train": None,
        "val": None,
        "test": None,
    }
    
    # =========================================================================
    # Fail-Safe: Check shadow CSV exists
    # =========================================================================
    if not os.path.exists(shadow_csv_path):
        logger.warning(f"SHADOW_RISK_OVERLAY:shadow_csv_missing - {shadow_csv_path}")
        return _write_fallback_overlay(
            output_overlay_csv_path=output_overlay_csv_path,
            output_overlay_metrics_json_path=output_overlay_metrics_json_path,
            overlay_metrics=overlay_metrics,
            reason="shadow_csv_missing",
            as_of_date=as_of_date,
        )
    
    # =========================================================================
    # Load shadow CSV
    # =========================================================================
    try:
        shadow_df = pd.read_csv(shadow_csv_path, index_col=0, parse_dates=True)
        if "exposure_suggested" not in shadow_df.columns:
            raise ValueError("exposure_suggested column missing")
    except Exception as e:
        logger.warning(f"SHADOW_RISK_OVERLAY:shadow_csv_invalid - {e}")
        return _write_fallback_overlay(
            output_overlay_csv_path=output_overlay_csv_path,
            output_overlay_metrics_json_path=output_overlay_metrics_json_path,
            overlay_metrics=overlay_metrics,
            reason="shadow_csv_invalid",
            as_of_date=as_of_date,
        )
    
    # =========================================================================
    # Fail-Safe: Check SPY exists in prices
    # =========================================================================
    if spy_ticker not in prices.columns:
        logger.warning(f"SHADOW_RISK_OVERLAY:spy_ticker_missing - {spy_ticker}")
        return _write_fallback_overlay(
            output_overlay_csv_path=output_overlay_csv_path,
            output_overlay_metrics_json_path=output_overlay_metrics_json_path,
            overlay_metrics=overlay_metrics,
            reason="spy_ticker_missing",
            as_of_date=as_of_date,
        )
    
    # =========================================================================
    # Compute SPY daily returns
    # =========================================================================
    spy_prices = prices[spy_ticker].dropna()
    spy_ret = spy_prices.pct_change(1).fillna(0.0)
    spy_ret.name = "spy_ret_1d"
    
    # =========================================================================
    # Align exposure to SPY returns (use intersection)
    # =========================================================================
    exposure = shadow_df["exposure_suggested"]
    common_idx = exposure.index.intersection(spy_ret.index)
    
    if len(common_idx) < 2:
        logger.warning(f"SHADOW_RISK_OVERLAY:insufficient_overlap - {len(common_idx)} dates")
        return _write_fallback_overlay(
            output_overlay_csv_path=output_overlay_csv_path,
            output_overlay_metrics_json_path=output_overlay_metrics_json_path,
            overlay_metrics=overlay_metrics,
            reason="insufficient_overlap",
            as_of_date=as_of_date,
        )
    
    common_idx = common_idx.sort_values()
    exposure = exposure.loc[common_idx]
    spy_ret_aligned = spy_ret.loc[common_idx]
    
    # =========================================================================
    # Apply "weight at t earns return t→t+1" semantics
    # overlay_ret[t] = exposure[t-1] * spy_ret[t] + (1 - exposure[t-1]) * cash_ret
    # =========================================================================
    exposure_shifted = exposure.shift(1).fillna(1.0)  # First day: full exposure
    overlay_ret = exposure_shifted * spy_ret_aligned + (1 - exposure_shifted) * cash_daily_return
    overlay_ret.iloc[0] = 0.0  # First day return is 0
    overlay_ret.name = "overlay_ret_1d"
    
    # Compute equity curve
    overlay_equity = (1 + overlay_ret).cumprod()
    overlay_equity.name = "overlay_equity"
    
    # =========================================================================
    # Build output dataframe
    # =========================================================================
    overlay_df = pd.DataFrame({
        "exposure_suggested": exposure,
        "spy_ret_1d": spy_ret_aligned,
        "overlay_ret_1d": overlay_ret,
        "overlay_equity": overlay_equity,
    })
    overlay_df = overlay_df.sort_index()
    overlay_df.index.name = "date"
    
    # =========================================================================
    # Compute metrics per split
    # =========================================================================
    train_mask = overlay_df.index <= train_end_dt
    val_mask = (overlay_df.index > train_end_dt) & (overlay_df.index <= val_end_dt)
    test_mask = (overlay_df.index > val_end_dt) & (overlay_df.index <= as_of_dt)
    
    overlay_metrics["train"] = _compute_overlay_split_metrics(
        overlay_df.loc[train_mask, "overlay_ret_1d"],
        overlay_df.loc[train_mask, "exposure_suggested"],
        "train",
    )
    overlay_metrics["val"] = _compute_overlay_split_metrics(
        overlay_df.loc[val_mask, "overlay_ret_1d"],
        overlay_df.loc[val_mask, "exposure_suggested"],
        "val",
    )
    overlay_metrics["test"] = _compute_overlay_split_metrics(
        overlay_df.loc[test_mask, "overlay_ret_1d"],
        overlay_df.loc[test_mask, "exposure_suggested"],
        "test",
    )
    
    # =========================================================================
    # Write artifacts
    # =========================================================================
    overlay_df.to_csv(output_overlay_csv_path, float_format="%.10f")
    _write_metrics_json(output_overlay_metrics_json_path, overlay_metrics)
    
    return {"overlay_df": overlay_df, "overlay_metrics": overlay_metrics}


def _write_fallback_overlay(
    output_overlay_csv_path: str,
    output_overlay_metrics_json_path: str,
    overlay_metrics: dict,
    reason: str,
    as_of_date: str,
) -> dict:
    """Write fallback overlay artifacts when computation fails."""
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Fallback overlay CSV with single row
    fallback_df = pd.DataFrame({
        "exposure_suggested": [1.0],
        "spy_ret_1d": [0.0],
        "overlay_ret_1d": [0.0],
        "overlay_equity": [1.0],
    }, index=pd.DatetimeIndex([as_of_dt]))
    fallback_df.index.name = "date"
    fallback_df.to_csv(output_overlay_csv_path, float_format="%.10f")
    
    # Fallback metrics
    fallback_split = {
        "n_obs": 0,
        "total_return": None,
        "cagr": None,
        "ann_vol": None,
        "cagr_over_vol": None,
        "max_drawdown": None,
        "avg_exposure": None,
        "turnover_exposure": None,
    }
    
    overlay_metrics["train"] = fallback_split.copy()
    overlay_metrics["val"] = fallback_split.copy()
    overlay_metrics["test"] = fallback_split.copy()
    overlay_metrics["fallback_reason"] = reason
    
    _write_metrics_json(output_overlay_metrics_json_path, overlay_metrics)
    
    logger.warning(f"SHADOW_RISK_OVERLAY:fallback_written - reason={reason}")
    
    return {"overlay_df": fallback_df, "overlay_metrics": overlay_metrics}
