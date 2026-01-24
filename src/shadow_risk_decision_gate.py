"""
Shadow Risk Decision Gate Evaluator for QUANT-NEURAL.

Evaluates whether to promote shadow risk gating from shadow-only to execution-control
based on overlay performance metrics and calibration quality.

Key Properties:
- Fail-Safe: Never crash; writes valid decision JSON even on error.
- Deterministic: Same inputs => byte-identical JSON output.
- Shadow-Only: This module does NOT change any trading outcomes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_spy_buy_hold_metrics(
    spy_ret_1d: pd.Series,
    ann_factor: float = 252,
) -> dict:
    """
    Compute SPY buy-and-hold metrics from daily returns.
    """
    n_obs = len(spy_ret_1d)
    
    if n_obs == 0:
        return {
            "n_obs": 0,
            "total_return": None,
            "cagr": None,
            "ann_vol": None,
            "cagr_over_vol": None,
            "max_drawdown": None,
        }
    
    # Build equity curve
    spy_equity = (1 + spy_ret_1d).cumprod()
    
    # Total return
    total_return = float(round(spy_equity.iloc[-1] - 1, 10))
    
    # CAGR
    years = n_obs / ann_factor
    if years > 0 and spy_equity.iloc[-1] > 0:
        cagr = float(round((spy_equity.iloc[-1] ** (1 / years)) - 1, 10))
    else:
        cagr = None
    
    # Annualized volatility
    if n_obs > 1:
        ann_vol = float(round(spy_ret_1d.std() * np.sqrt(ann_factor), 10))
    else:
        ann_vol = None
    
    # CAGR / Vol
    if cagr is not None and ann_vol is not None and ann_vol > 0:
        cagr_over_vol = float(round(cagr / ann_vol, 10))
    else:
        cagr_over_vol = None
    
    # Max drawdown
    running_max = spy_equity.cummax()
    drawdown = (spy_equity - running_max) / running_max
    max_drawdown = float(round(drawdown.min(), 10)) if len(drawdown) > 0 else None
    
    return {
        "n_obs": n_obs,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "cagr_over_vol": cagr_over_vol,
        "max_drawdown": max_drawdown,
    }


def _compute_dd_reduction(
    overlay_max_dd: float | None,
    spy_max_dd: float | None,
) -> float | None:
    """
    Compute drawdown reduction: (|spy_dd| - |overlay_dd|) / |spy_dd|
    """
    if overlay_max_dd is None or spy_max_dd is None:
        return None
    
    if not np.isfinite(overlay_max_dd) or not np.isfinite(spy_max_dd):
        return None
    
    dd_overlay = abs(overlay_max_dd)
    dd_spy = abs(spy_max_dd)
    
    if dd_spy <= 0:
        return None
    
    dd_reduction = (dd_spy - dd_overlay) / dd_spy
    return float(round(dd_reduction, 10))


def _write_decision_json(output_path: str, decision_data: dict) -> None:
    """Write decision JSON with deterministic serialization."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(decision_data, f, indent=2, sort_keys=True)


def evaluate_shadow_risk_promotion_decision(
    *,
    risk_metrics_json_path: str,
    overlay_csv_path: str,
    overlay_metrics_json_path: str,
    output_decision_json_path: str,
    dd_reduction_threshold: float = 0.20,
    cagr_over_vol_threshold: float = 1.0,
    ece_threshold: float = 0.05,
) -> dict:
    """
    Evaluate whether to promote shadow risk gating to execution-control.
    
    Reads metrics from 9.5.2/9.5.3 artifacts and evaluates against thresholds
    defined in PLANS 9.6.0.
    
    Parameters
    ----------
    risk_metrics_json_path : str
        Path to 9.5.2 risk metrics JSON (contains test.ece).
    overlay_csv_path : str
        Path to 9.5.3 overlay CSV (contains spy_ret_1d for buy-hold computation).
    overlay_metrics_json_path : str
        Path to 9.5.3 overlay metrics JSON (contains cagr_over_vol, max_drawdown).
    output_decision_json_path : str
        Path to write decision JSON.
    dd_reduction_threshold : float
        Minimum drawdown reduction required (default: 0.20 = 20%).
    cagr_over_vol_threshold : float
        Minimum CAGR/Vol ratio required (default: 1.0).
    ece_threshold : float
        Maximum ECE allowed (default: 0.05).
    
    Returns
    -------
    dict
        Decision data including decision, checks, and reasons.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_decision_json_path) or ".", exist_ok=True)
    
    # Initialize decision structure
    decision_data = {
        "schema_version": "9.6.0",
        "inputs": {
            "risk_metrics_json_path": risk_metrics_json_path,
            "overlay_csv_path": overlay_csv_path,
            "overlay_metrics_json_path": overlay_metrics_json_path,
        },
        "computed": {
            "overlay": {"cagr_over_vol": None, "max_drawdown": None},
            "spy_buy_hold": {"cagr_over_vol": None, "max_drawdown": None},
            "dd_reduction": None,
            "test_ece": None,
        },
        "thresholds": {
            "cagr_over_vol": cagr_over_vol_threshold,
            "dd_reduction": dd_reduction_threshold,
            "ece": ece_threshold,
        },
        "checks": {
            "pass_cagr_over_vol": False,
            "pass_dd_reduction": False,
            "pass_ece": False,
        },
        "decision": "RETAIN_SHADOW_ONLY",
        "reasons": [],
        "warnings": [],
    }
    
    # =========================================================================
    # Load risk metrics JSON (9.5.2)
    # =========================================================================
    try:
        if not os.path.exists(risk_metrics_json_path):
            raise FileNotFoundError(f"Risk metrics JSON not found: {risk_metrics_json_path}")
        
        with open(risk_metrics_json_path, "r", encoding="utf-8") as f:
            risk_metrics = json.load(f)
        
        test_ece = risk_metrics.get("test", {}).get("ece")
        decision_data["computed"]["test_ece"] = test_ece
        
    except Exception as e:
        logger.warning(f"SHADOW_RISK_GATE:risk_metrics_load_failed - {e}")
        decision_data["warnings"].append(f"Failed to load risk metrics: {e}")
        decision_data["reasons"].append("Cannot evaluate: risk metrics JSON unavailable")
        _write_decision_json(output_decision_json_path, decision_data)
        return decision_data
    
    # =========================================================================
    # Load overlay metrics JSON (9.5.3)
    # =========================================================================
    try:
        if not os.path.exists(overlay_metrics_json_path):
            raise FileNotFoundError(f"Overlay metrics JSON not found: {overlay_metrics_json_path}")
        
        with open(overlay_metrics_json_path, "r", encoding="utf-8") as f:
            overlay_metrics = json.load(f)
        
        # Extract overlay metrics (use test split if available, else val)
        test_metrics = overlay_metrics.get("test", {})
        overlay_cagr_over_vol = test_metrics.get("cagr_over_vol")
        overlay_max_dd = test_metrics.get("max_drawdown")
        
        decision_data["computed"]["overlay"]["cagr_over_vol"] = overlay_cagr_over_vol
        decision_data["computed"]["overlay"]["max_drawdown"] = overlay_max_dd
        
    except Exception as e:
        logger.warning(f"SHADOW_RISK_GATE:overlay_metrics_load_failed - {e}")
        decision_data["warnings"].append(f"Failed to load overlay metrics: {e}")
        decision_data["reasons"].append("Cannot evaluate: overlay metrics JSON unavailable")
        _write_decision_json(output_decision_json_path, decision_data)
        return decision_data
    
    # =========================================================================
    # Load overlay CSV and compute SPY buy-and-hold metrics (TEST window only)
    # =========================================================================
    # First, extract TEST window boundaries from config
    val_end_dt = None
    as_of_dt = None
    test_window_fallback = False
    
    # Try risk_metrics config first
    try:
        config = risk_metrics.get("config", {})
        val_end_str = config.get("val_end")
        as_of_str = config.get("as_of_date")
        
        if val_end_str and as_of_str:
            val_end_dt = pd.to_datetime(val_end_str)
            as_of_dt = pd.to_datetime(as_of_str)
    except Exception:
        pass
    
    # Fallback to overlay_metrics config if needed
    if val_end_dt is None or as_of_dt is None:
        try:
            config = overlay_metrics.get("config", {})
            val_end_str = config.get("val_end")
            as_of_str = config.get("as_of_date")
            
            if val_end_str and as_of_str:
                val_end_dt = pd.to_datetime(val_end_str)
                as_of_dt = pd.to_datetime(as_of_str)
        except Exception:
            pass
    
    # Record warning if fallback to full window
    if val_end_dt is None or as_of_dt is None:
        test_window_fallback = True
        decision_data["warnings"].append("TEST_WINDOW_FALLBACK:missing_config_dates")
    
    try:
        if not os.path.exists(overlay_csv_path):
            raise FileNotFoundError(f"Overlay CSV not found: {overlay_csv_path}")
        
        overlay_df = pd.read_csv(overlay_csv_path, index_col=0, parse_dates=True)
        
        if "spy_ret_1d" not in overlay_df.columns:
            raise ValueError("spy_ret_1d column missing from overlay CSV")
        
        spy_ret_1d = overlay_df["spy_ret_1d"].dropna()
        
        # Filter to TEST window: (val_end, as_of_date]
        if val_end_dt is not None and as_of_dt is not None:
            spy_ret_1d_test = spy_ret_1d[
                (spy_ret_1d.index > val_end_dt) & (spy_ret_1d.index <= as_of_dt)
            ]
        elif val_end_dt is not None:
            # as_of_dt missing, use index max
            as_of_dt = spy_ret_1d.index.max()
            spy_ret_1d_test = spy_ret_1d[spy_ret_1d.index > val_end_dt]
        else:
            # Full window fallback
            spy_ret_1d_test = spy_ret_1d
        
        spy_metrics = _compute_spy_buy_hold_metrics(spy_ret_1d_test)
        
        decision_data["computed"]["spy_buy_hold"]["cagr_over_vol"] = spy_metrics["cagr_over_vol"]
        decision_data["computed"]["spy_buy_hold"]["max_drawdown"] = spy_metrics["max_drawdown"]
        
    except Exception as e:
        logger.warning(f"SHADOW_RISK_GATE:overlay_csv_load_failed - {e}")
        decision_data["warnings"].append(f"Failed to load overlay CSV: {e}")
        decision_data["reasons"].append("Cannot evaluate: overlay CSV unavailable")
        _write_decision_json(output_decision_json_path, decision_data)
        return decision_data
    
    # =========================================================================
    # Compute drawdown reduction (TEST window: overlay vs SPY)
    # =========================================================================
    dd_reduction = _compute_dd_reduction(overlay_max_dd, spy_metrics["max_drawdown"])
    decision_data["computed"]["dd_reduction"] = dd_reduction
    
    # =========================================================================
    # Evaluate acceptance criteria
    # =========================================================================
    reasons = []
    
    # Check 1: CAGR/Vol
    if overlay_cagr_over_vol is not None and np.isfinite(overlay_cagr_over_vol):
        pass_cagr_over_vol = overlay_cagr_over_vol >= cagr_over_vol_threshold
        decision_data["checks"]["pass_cagr_over_vol"] = pass_cagr_over_vol
        if not pass_cagr_over_vol:
            reasons.append(
                f"CAGR/Vol {overlay_cagr_over_vol:.4f} < threshold {cagr_over_vol_threshold}"
            )
    else:
        decision_data["checks"]["pass_cagr_over_vol"] = False
        reasons.append("CAGR/Vol unavailable or non-finite")
    
    # Check 2: Drawdown reduction
    if dd_reduction is not None and np.isfinite(dd_reduction):
        pass_dd_reduction = dd_reduction >= dd_reduction_threshold
        decision_data["checks"]["pass_dd_reduction"] = pass_dd_reduction
        if not pass_dd_reduction:
            reasons.append(
                f"DD reduction {dd_reduction:.4f} < threshold {dd_reduction_threshold}"
            )
    else:
        decision_data["checks"]["pass_dd_reduction"] = False
        reasons.append("Drawdown reduction unavailable or non-finite")
    
    # Check 3: ECE
    if test_ece is not None and np.isfinite(test_ece):
        pass_ece = test_ece < ece_threshold
        decision_data["checks"]["pass_ece"] = pass_ece
        if not pass_ece:
            reasons.append(f"ECE {test_ece:.4f} >= threshold {ece_threshold}")
    else:
        decision_data["checks"]["pass_ece"] = False
        reasons.append("ECE unavailable or non-finite")
    
    # =========================================================================
    # Final decision
    # =========================================================================
    all_pass = all([
        decision_data["checks"]["pass_cagr_over_vol"],
        decision_data["checks"]["pass_dd_reduction"],
        decision_data["checks"]["pass_ece"],
    ])
    
    if all_pass:
        decision_data["decision"] = "PROMOTE_EXECUTION_CONTROL"
        decision_data["reasons"] = ["All acceptance criteria passed"]
    else:
        decision_data["decision"] = "RETAIN_SHADOW_ONLY"
        decision_data["reasons"] = reasons
    
    # =========================================================================
    # Write decision JSON
    # =========================================================================
    _write_decision_json(output_decision_json_path, decision_data)
    
    return decision_data
