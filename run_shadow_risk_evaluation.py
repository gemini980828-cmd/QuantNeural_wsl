"""
Shadow Risk SPY-Only Evaluation Runner

Compares Shadow Risk Logit vs MLP in a controlled SPY-only setting:
- BANS non-SPY proxies (fail-fast if SPY unavailable)
- Generates deterministic artifacts for both models side-by-side
- Runs full evaluation: risk exposure, overlay, decision gate, temperature sweep

This is SHADOW-ONLY diagnostics: NO trading impact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.shadow_risk_exposure import (
    run_shadow_risk_exposure_logit_with_metrics,
    run_shadow_risk_exposure_mlp_with_metrics,
    run_shadow_risk_exposure_xgb_with_metrics,
    run_shadow_risk_overlay_spy_only,
)
from src.shadow_risk_decision_gate import evaluate_shadow_risk_promotion_decision
from src.shadow_risk_calibration_sweep import run_shadow_risk_temperature_sweep
from src.shadow_risk_overlay_diagnostics import compute_overlay_exposure_diagnostics
from src.shadow_risk_overlay_policy_sweep import run_overlay_policy_sweep
from src.shadow_risk_overlay_policy_recommendation import build_overlay_policy_recommendation
from src.shadow_risk_overlay_policy_apply import apply_recommended_overlay_policy

logger = logging.getLogger(__name__)


def _normalize_overlay_metrics_path(metrics_json_path: str) -> None:
    """Normalize shadow_csv_path for deterministic overlay metrics artifacts."""
    try:
        with open(metrics_json_path, encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("config", {})
        if isinstance(config, dict):
            config["shadow_csv_path"] = "shadow_risk.csv"
            data["config"] = config
        with open(metrics_json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception:
        logger.warning(f"SHADOW_RISK_OVERLAY:normalize_path_failed path={metrics_json_path}")


def _generate_ops_report(
    result: dict,
    horizon_dir: Path,
    as_of_date: str,
    horizon_days: int,
    champion_variant: str = "xgb",
    overlay_mode: str = "policy_best",
    calibration_mode: str = "raw",
) -> str:
    """Generate deterministic ops summary report for champion lock."""
    report_path = str(horizon_dir / "ops_shadow_risk_report.json")
    warnings_list: list[str] = []
    
    try:
        models = result.get("models", {})
        champion_data = models.get(champion_variant, {})
        
        if not champion_data:
            warnings_list.append(f"OPS_WARN:champion_variant_missing:{champion_variant}")
        
        # Determine paths based on overlay_mode
        if overlay_mode == "policy_best":
            overlay_csv_key = "shadow_risk_overlay_policy_best_csv"
            overlay_metrics_key = "shadow_risk_overlay_policy_best_metrics_json"
            overlay_diag_key = "shadow_risk_overlay_policy_best_diagnostics_json"
        else:
            overlay_csv_key = "shadow_risk_overlay_csv"
            overlay_metrics_key = "shadow_risk_overlay_metrics_json"
            overlay_diag_key = "shadow_risk_overlay_diagnostics_json"
        
        selected_paths = {
            "shadow_risk_csv": champion_data.get("shadow_risk_csv"),
            "shadow_risk_metrics_json": champion_data.get("shadow_risk_metrics_json"),
            "overlay_csv": champion_data.get(overlay_csv_key) or champion_data.get("shadow_risk_overlay_csv"),
            "overlay_metrics_json": champion_data.get(overlay_metrics_key) or champion_data.get("shadow_risk_overlay_metrics_json"),
            "overlay_diagnostics_json": champion_data.get(overlay_diag_key) or champion_data.get("shadow_risk_overlay_diagnostics_json"),
            "policy_recommendation_json": champion_data.get("shadow_risk_overlay_policy_recommendation_json"),
            "policy_best_overlay_metrics_json": champion_data.get("shadow_risk_overlay_policy_best_metrics_json"),
        }
        
        # Extract risk metrics
        risk_metrics = {"test_ece": None, "test_brier": None, "test_logloss": None, "test_auc": None}
        risk_path = selected_paths.get("shadow_risk_metrics_json")
        if risk_path and Path(risk_path).exists():
            try:
                with open(risk_path, encoding="utf-8") as f:
                    rm = json.load(f)
                test_data = rm.get("test", {})
                risk_metrics["test_ece"] = test_data.get("ece")
                risk_metrics["test_brier"] = test_data.get("brier")
                risk_metrics["test_logloss"] = test_data.get("log_loss")
                risk_metrics["test_auc"] = test_data.get("roc_auc")
            except Exception as e:
                warnings_list.append(f"OPS_WARN:risk_parse_error:{str(e)[:40]}")
        else:
            warnings_list.append("OPS_WARN:risk_metrics_missing")
        
        # Extract overlay metrics
        overlay_metrics = {"test_cagr_over_vol": None, "test_max_dd": None, "avg_exposure": None, "turnover_proxy": None}
        ov_path = selected_paths.get("overlay_metrics_json")
        if ov_path and Path(ov_path).exists():
            try:
                with open(ov_path, encoding="utf-8") as f:
                    om = json.load(f)
                if "test" in om:
                    overlay_metrics["test_cagr_over_vol"] = om["test"].get("cagr_over_vol")
                    overlay_metrics["test_max_dd"] = om["test"].get("max_drawdown")
                elif "policy" in om:
                    overlay_metrics["test_cagr_over_vol"] = om["policy"].get("cagr_over_vol")
                    overlay_metrics["test_max_dd"] = om["policy"].get("max_dd")
                    overlay_metrics["avg_exposure"] = om["policy"].get("avg_exposure")
                    overlay_metrics["turnover_proxy"] = om["policy"].get("turnover_proxy")
            except Exception as e:
                warnings_list.append(f"OPS_WARN:overlay_parse_error:{str(e)[:40]}")
        else:
            warnings_list.append("OPS_WARN:overlay_metrics_missing")
        
        # Diagnostics for exposure/turnover
        if overlay_metrics["avg_exposure"] is None:
            diag_path = selected_paths.get("overlay_diagnostics_json")
            if diag_path and Path(diag_path).exists():
                try:
                    with open(diag_path, encoding="utf-8") as f:
                        diag = json.load(f)
                    diag_data = diag.get("diagnostics", {})
                    overlay_metrics["avg_exposure"] = diag_data.get("avg_exposure")
                    overlay_metrics["turnover_proxy"] = diag_data.get("turnover_proxy")
                except Exception:
                    pass
        
        # Collect metrics from all variants for comparison
        all_cagr_vol, all_max_dd, all_auc = [], [], []
        for var_name, var_data in models.items():
            try:
                vm_path = var_data.get(overlay_metrics_key) or var_data.get("shadow_risk_overlay_metrics_json")
                if vm_path and Path(vm_path).exists():
                    with open(vm_path, encoding="utf-8") as f:
                        vm = json.load(f)
                    cov = vm.get("test", vm.get("policy", {})).get("cagr_over_vol")
                    mdd = vm.get("test", vm.get("policy", {})).get("max_drawdown") or vm.get("test", vm.get("policy", {})).get("max_dd")
                    if cov is not None:
                        all_cagr_vol.append(cov)
                    if mdd is not None:
                        all_max_dd.append(mdd)
            except Exception:
                pass
            try:
                rm_path = var_data.get("shadow_risk_metrics_json")
                if rm_path and Path(rm_path).exists():
                    with open(rm_path, encoding="utf-8") as f:
                        rm = json.load(f)
                    auc = rm.get("test", {}).get("roc_auc")
                    if auc is not None:
                        all_auc.append(auc)
            except Exception:
                pass
        
        # Operational decision logic
        status = "OK"
        reasons: list[str] = []
        
        # Check for missing files
        for key, path in selected_paths.items():
            if path and not Path(path).exists():
                status = "FAIL"
                reasons.append("MISSING_ARTIFACTS")
                break
        
        cov = overlay_metrics.get("test_cagr_over_vol")
        mdd = overlay_metrics.get("test_max_dd")
        auc = risk_metrics.get("test_auc")
        
        if cov is None and mdd is None:
            if status != "FAIL":
                status = "FAIL"
            reasons.append("METRICS_NAN")
        
        if cov is not None and cov <= 0 and status == "OK":
            status = "WARN"
            reasons.append("CAGR_OVER_VOL_NEGATIVE")
        
        if mdd is not None and mdd <= -0.20 and status == "OK":
            status = "WARN"
            reasons.append("MAX_DD_SEVERE")
        
        if auc is not None and auc < 0.55 and status == "OK":
            status = "WARN"
            reasons.append("AUC_LOW")
        
        # Re-evaluation triggers
        if all_cagr_vol and cov is not None:
            best_cov = max(all_cagr_vol)
            if cov <= 0.75 * best_cov:
                reasons.append("OVERLAY_CAGR_OVER_VOL_DROP_GT_25PCT")
        
        if all_max_dd and mdd is not None:
            best_mdd = max(all_max_dd)
            if mdd < best_mdd - 0.05:
                reasons.append("MAX_DD_WORSE_BY_GT_0.05")
        
        if all_auc and auc is not None:
            best_auc = max(all_auc)
            if auc < best_auc - 0.10:
                reasons.append("AUC_DROP_GT_0.10")
        
        recommended_action = "HALT" if status == "FAIL" else ("REVIEW" if status == "WARN" else "KEEP")
        
        report = {
            "schema_version": "9.6.16",
            "as_of_date": as_of_date,
            "horizon_days": horizon_days,
            "champion": {
                "variant": champion_variant,
                "calibration_mode": calibration_mode,
                "overlay_mode": overlay_mode,
                "selected_paths": selected_paths,
            },
            "key_metrics": {"risk": risk_metrics, "overlay": overlay_metrics},
            "operational_decision": {"status": status, "reasons": sorted(set(reasons)), "recommended_action": recommended_action},
            "re_evaluation_policy": {
                "cadence": "monthly",
                "hard_triggers": ["MISSING_ARTIFACTS", "METRICS_NAN", "OVERLAY_CAGR_OVER_VOL_DROP_GT_25PCT", "MAX_DD_WORSE_BY_GT_0.05", "AUC_DROP_GT_0.10"],
                "notes": "Champion is XGB raw; temperature scaling is not applied by default. Policy-best overlay is the operational overlay by default.",
            },
            "warnings": warnings_list,
        }
    except Exception as e:
        warnings_list.append(f"OPS_FAIL:report_error:{str(e)[:60]}")
        report = {
            "schema_version": "9.6.16", "as_of_date": as_of_date, "horizon_days": horizon_days,
            "champion": {"variant": champion_variant, "calibration_mode": calibration_mode, "overlay_mode": overlay_mode, "selected_paths": {}},
            "key_metrics": {"risk": {}, "overlay": {}},
            "operational_decision": {"status": "FAIL", "reasons": ["REPORT_GENERATION_ERROR"], "recommended_action": "HALT"},
            "re_evaluation_policy": {"cadence": "monthly", "hard_triggers": [], "notes": ""},
            "warnings": warnings_list,
        }
    
    try:
        json_str = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        with open(report_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_str)
    except Exception as e:
        logger.warning(f"OPS_REPORT:write_failed:{str(e)[:50]}")
    
    return report_path


def _load_spy_prices(spy_csv_path: str, spy_ticker: str, as_of_date: str) -> pd.DataFrame:
    """
    Load SPY prices from CSV and return DataFrame with single SPY column.
    
    Supports two formats:
    1. Stooq format: <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    2. Simple format: date,close
    
    Raises ValueError if SPY price series cannot be constructed.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    
    try:
        df = pd.read_csv(spy_csv_path)
    except Exception as e:
        raise ValueError(f"SPY_REQUIRED:missing_spy_price_series - cannot read CSV: {e}")
    
    if len(df) == 0:
        raise ValueError("SPY_REQUIRED:missing_spy_price_series - empty CSV")
    
    # Detect format
    cols_lower = [c.lower() for c in df.columns]
    
    if "<close>" in cols_lower or "<CLOSE>" in df.columns:
        # Stooq format
        date_col = "<DATE>" if "<DATE>" in df.columns else "<date>"
        close_col = "<CLOSE>" if "<CLOSE>" in df.columns else "<close>"
        
        if date_col not in df.columns or close_col not in df.columns:
            raise ValueError("SPY_REQUIRED:missing_spy_price_series - missing required columns")
        
        df["date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
        df["close"] = df[close_col]
    elif "date" in cols_lower and "close" in cols_lower:
        # Simple format
        date_col = [c for c in df.columns if c.lower() == "date"][0]
        close_col = [c for c in df.columns if c.lower() == "close"][0]
        
        df["date"] = pd.to_datetime(df[date_col])
        df["close"] = df[close_col]
    else:
        raise ValueError("SPY_REQUIRED:missing_spy_price_series - unrecognized CSV format")
    
    # Filter and sort
    df = df[["date", "close"]].dropna()
    df = df[df["date"] <= as_of_dt]
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date")
    df = df.set_index("date")
    
    if len(df) < 100:
        raise ValueError(f"SPY_REQUIRED:missing_spy_price_series - insufficient data ({len(df)} rows)")
    
    # Create prices DataFrame with SPY column
    # Strip .US suffix if present for internal processing
    internal_ticker = spy_ticker.replace(".US", "")
    prices = pd.DataFrame({internal_ticker: df["close"]})
    
    return prices


def _validate_spy_ticker(spy_ticker: str) -> None:
    """
    Validate that spy_ticker is an SPY-like identifier.
    
    Raises ValueError if ticker appears to be a proxy (non-SPY).
    """
    if "SPY" not in spy_ticker.upper():
        raise ValueError(f"SPY_REQUIRED:proxy_not_allowed - ticker '{spy_ticker}' is not SPY")


def run_shadow_risk_spy_only_evaluation(
    *,
    spy_csv_path: str,
    spy_ticker: str,
    as_of_date: str,
    train_end: str,
    val_end: str,
    output_root_dir: str,
    seed: int = 42,
    horizon_days: int = 63,
    sweep_train_years: int = 1,
    sweep_val_years: int = 1,
    sweep_test_years: int = 1,
    sweep_step_months: int = 3,
    ops_mode: str = "off",
    ops_champion_variant: str = "xgb",
    ops_overlay_mode: str = "policy_best",
    ops_calibration_mode: str = "raw",
) -> dict:
    """
    Run SPY-only shadow risk evaluation for both logit and MLP models.
    
    Parameters
    ----------
    spy_csv_path : str
        Path to SPY price CSV (stooq or simple format).
    spy_ticker : str
        SPY ticker name (e.g., "SPY.US"). Non-SPY tickers are banned.
    as_of_date : str
        PIT cutoff date (YYYY-MM-DD).
    train_end : str
        End of training period.
    val_end : str
        End of validation period.
    output_root_dir : str
        Root directory for output artifacts.
    seed : int
        Random seed for reproducibility.
    horizon_days : int
        Forward horizon for risk labels.
    sweep_train_years, sweep_val_years, sweep_test_years, sweep_step_months : int
        Temperature calibration sweep parameters.
    
    Returns
    -------
    dict
        Dictionary with paths to key artifacts for each model variant.
    
    Raises
    ------
    ValueError
        If spy_ticker is not SPY-like or if SPY prices cannot be loaded.
    """
    # =========================================================================
    # Proxy Ban (Fail-Fast)
    # =========================================================================
    _validate_spy_ticker(spy_ticker)
    
    # =========================================================================
    # Load SPY Prices
    # =========================================================================
    internal_ticker = spy_ticker.replace(".US", "")
    prices = _load_spy_prices(spy_csv_path, spy_ticker, as_of_date)
    
    # =========================================================================
    # Create Output Directory Structure (includes horizon for ablation)
    # =========================================================================
    output_root = Path(output_root_dir)
    horizon_dir = output_root / f"horizon_{horizon_days}"
    
    variants = ["logit", "mlp", "xgb"]
    result = {"models": {}}
    
    for variant in variants:
        variant_dir = horizon_dir / variant
        
        # Clean slate: remove existing directory for determinism
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        sweep_dir = variant_dir / "temp_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Define artifact paths
        shadow_risk_csv = str(variant_dir / "shadow_risk.csv")
        shadow_risk_metrics_json = str(variant_dir / "shadow_risk_metrics.json")
        shadow_risk_overlay_csv = str(variant_dir / "shadow_risk_overlay.csv")
        shadow_risk_overlay_metrics_json = str(variant_dir / "shadow_risk_overlay_metrics.json")
        shadow_risk_decision_gate_json = str(variant_dir / "shadow_risk_decision_gate.json")
        
        # =====================================================================
        # Run Model-Specific Exposure
        # =====================================================================
        xgb_exported = None
        if variant == "logit":
            run_shadow_risk_exposure_logit_with_metrics(
                prices=prices,
                as_of_date=as_of_date,
                train_end=train_end,
                val_end=val_end,
                output_csv_path=shadow_risk_csv,
                output_metrics_json_path=shadow_risk_metrics_json,
                spy_ticker=internal_ticker,
                horizon_days=horizon_days,
                seed=seed,
            )
        elif variant == "mlp":
            run_shadow_risk_exposure_mlp_with_metrics(
                prices=prices,
                as_of_date=as_of_date,
                train_end=train_end,
                val_end=val_end,
                output_csv_path=shadow_risk_csv,
                output_metrics_json_path=shadow_risk_metrics_json,
                spy_ticker=internal_ticker,
                horizon_days=horizon_days,
                seed=seed,
            )
        else:  # xgb
            try:
                run_shadow_risk_exposure_xgb_with_metrics(
                    prices=prices,
                    as_of_date=as_of_date,
                    train_end=train_end,
                    val_end=val_end,
                    output_csv_path=shadow_risk_csv,
                    output_metrics_json_path=shadow_risk_metrics_json,
                    spy_ticker=internal_ticker,
                    horizon_days=horizon_days,
                    seed=seed,
                )
                xgb_exported = True
            except Exception as e:
                logger.warning(f"SHADOW_RISK_XGB:exception msg={str(e)[:100]}")
                xgb_exported = False
        
        # =====================================================================
        # Run Overlay Backtest
        # =====================================================================
        run_shadow_risk_overlay_spy_only(
            prices=prices,
            as_of_date=as_of_date,
            train_end=train_end,
            val_end=val_end,
            shadow_csv_path=shadow_risk_csv,
            output_overlay_csv_path=shadow_risk_overlay_csv,
            output_overlay_metrics_json_path=shadow_risk_overlay_metrics_json,
            spy_ticker=internal_ticker,
        )
        if variant == "xgb":
            _normalize_overlay_metrics_path(shadow_risk_overlay_metrics_json)
        
        # =====================================================================
        # Run Overlay Exposure Diagnostics (fail-safe)
        # =====================================================================
        overlay_diagnostics_json = str(variant_dir / "overlay_diagnostics.json")
        try:
            compute_overlay_exposure_diagnostics(
                shadow_risk_overlay_csv,
                output_json_path=overlay_diagnostics_json,
                exposure_column="exposure_suggested",
                seed=seed,
            )
            overlay_diagnostics_exported = True
        except Exception as e:
            # Should never happen (function is fail-safe), but guard anyway
            logger.warning(f"SHADOW_RISK_DIAG:exception variant={variant} msg={str(e)[:100]}")
            overlay_diagnostics_exported = False
        
        # =====================================================================
        # Run Decision Gate
        # =====================================================================
        evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=shadow_risk_metrics_json,
            overlay_csv_path=shadow_risk_overlay_csv,
            overlay_metrics_json_path=shadow_risk_overlay_metrics_json,
            output_decision_json_path=shadow_risk_decision_gate_json,
        )
        
        # =====================================================================
        # Run Temperature Calibration Sweep
        # =====================================================================
        run_shadow_risk_temperature_sweep(
            shadow_csv_path=shadow_risk_csv,
            overlay_csv_path=shadow_risk_overlay_csv,
            output_dir=str(sweep_dir),
            as_of_date=as_of_date,
            horizon_days=horizon_days,
            train_years=sweep_train_years,
            val_years=sweep_val_years,
            test_years=sweep_test_years,
            step_months=sweep_step_months,
            seed=seed,
        )
        
        # =====================================================================
        # Run Overlay Policy Sweep (fail-safe)
        # =====================================================================
        policy_sweep_results_csv = str(variant_dir / "overlay_policy_sweep_results.csv")
        policy_sweep_summary_json = str(variant_dir / "overlay_policy_sweep_summary.json")
        try:
            run_overlay_policy_sweep(
                shadow_csv_path=shadow_risk_csv,
                overlay_csv_path=shadow_risk_overlay_csv,
                output_results_csv_path=policy_sweep_results_csv,
                output_summary_json_path=policy_sweep_summary_json,
                seed=seed,
            )
            policy_sweep_exported = True
        except Exception as e:
            # Should never happen (function is fail-safe), but guard anyway
            logger.warning(f"SHADOW_RISK_POLICY_SWEEP:exception variant={variant} msg={str(e)[:100]}")
            policy_sweep_exported = False
        
        # =====================================================================
        # Build Policy Recommendation (fail-safe)
        # =====================================================================
        policy_recommendation_json = str(variant_dir / "overlay_policy_recommendation.json")
        try:
            build_overlay_policy_recommendation(
                overlay_metrics_json_path=shadow_risk_overlay_metrics_json,
                overlay_diagnostics_json_path=overlay_diagnostics_json,
                policy_sweep_results_csv_path=policy_sweep_results_csv,
                policy_sweep_summary_json_path=policy_sweep_summary_json,
                output_json_path=policy_recommendation_json,
                seed=seed,
            )
            policy_recommendation_exported = True
        except Exception as e:
            # Should never happen (function is fail-safe), but guard anyway
            logger.warning(f"SHADOW_RISK_POLICY_REC:exception variant={variant} msg={str(e)[:100]}")
            policy_recommendation_exported = False
        
        # =====================================================================
        # Apply Policy to Create Counterfactual Overlay (fail-safe)
        # =====================================================================
        policy_best_csv = str(variant_dir / "shadow_risk_overlay_policy_best.csv")
        policy_best_metrics_json = str(variant_dir / "shadow_risk_overlay_policy_best_metrics.json")
        policy_best_diagnostics_json = str(variant_dir / "overlay_policy_best_diagnostics.json")
        try:
            apply_recommended_overlay_policy(
                shadow_csv_path=shadow_risk_csv,
                overlay_csv_path=shadow_risk_overlay_csv,
                policy_recommendation_json_path=policy_recommendation_json,
                output_overlay_csv_path=policy_best_csv,
                output_metrics_json_path=policy_best_metrics_json,
                output_diagnostics_json_path=policy_best_diagnostics_json,
                seed=seed,
            )
            policy_best_exported = True
        except Exception as e:
            # Should never happen (function is fail-safe), but guard anyway
            logger.warning(f"SHADOW_RISK_POLICY_APPLY:exception variant={variant} msg={str(e)[:100]}")
            policy_best_exported = False
        
        # Record artifact paths
        result["models"][variant] = {
            "shadow_risk_csv": shadow_risk_csv,
            "shadow_risk_metrics_json": shadow_risk_metrics_json,
            "shadow_risk_overlay_csv": shadow_risk_overlay_csv,
            "shadow_risk_overlay_metrics_json": shadow_risk_overlay_metrics_json,
            "shadow_risk_overlay_diagnostics_json": overlay_diagnostics_json,
            "shadow_risk_overlay_diagnostics_json_exported": overlay_diagnostics_exported,
            "shadow_risk_overlay_policy_sweep_results_csv": policy_sweep_results_csv,
            "shadow_risk_overlay_policy_sweep_summary_json": policy_sweep_summary_json,
            "shadow_risk_overlay_policy_sweep_exported": policy_sweep_exported,
            "shadow_risk_overlay_policy_recommendation_json": policy_recommendation_json,
            "shadow_risk_overlay_policy_recommendation_exported": policy_recommendation_exported,
            "shadow_risk_overlay_policy_best_csv": policy_best_csv,
            "shadow_risk_overlay_policy_best_metrics_json": policy_best_metrics_json,
            "shadow_risk_overlay_policy_best_diagnostics_json": policy_best_diagnostics_json,
            "shadow_risk_overlay_policy_best_exported": policy_best_exported,
            "shadow_risk_decision_gate_json": shadow_risk_decision_gate_json,
            "temp_sweep_dir": str(sweep_dir),
            "temp_sweep_results_csv": str(sweep_dir / "shadow_risk_temp_sweep_results.csv"),
            "temp_sweep_summary_json": str(sweep_dir / "shadow_risk_temp_sweep_summary.json"),
        }
        if variant == "xgb":
            result["models"][variant]["shadow_risk_exposure_xgb_exported"] = xgb_exported
    
    result["output_root_dir"] = str(output_root)
    result["horizon_dir"] = str(horizon_dir)
    result["config"] = {
        "spy_ticker": spy_ticker,
        "as_of_date": as_of_date,
        "train_end": train_end,
        "val_end": val_end,
        "seed": seed,
        "horizon_days": horizon_days,
    }
    
    # Generate ops report if ops_mode is on
    if ops_mode == "on":
        ops_report_path = _generate_ops_report(
            result=result,
            horizon_dir=horizon_dir,
            as_of_date=as_of_date,
            horizon_days=horizon_days,
            champion_variant=ops_champion_variant,
            overlay_mode=ops_overlay_mode,
            calibration_mode=ops_calibration_mode,
        )
        result["ops_report_json"] = ops_report_path
    
    return result


def main():
    """CLI entrypoint for SPY-only shadow risk evaluation."""
    parser = argparse.ArgumentParser(
        description="Run SPY-only shadow risk evaluation comparing logit vs MLP models"
    )
    
    parser.add_argument("--spy-csv-path", required=True, help="Path to SPY price CSV")
    parser.add_argument("--spy-ticker", default="SPY.US", help="SPY ticker name (default: SPY.US)")
    parser.add_argument("--as-of-date", required=True, help="PIT cutoff date (YYYY-MM-DD)")
    parser.add_argument("--train-end", required=True, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--val-end", required=True, help="Validation end date (YYYY-MM-DD)")
    parser.add_argument("--output-root-dir", default="artifacts/shadow_risk/eval", 
                        help="Root directory for output artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--horizon-days", type=int, default=63, help="Forward horizon days")
    parser.add_argument("--ops-mode", default="off", choices=["off", "on"],
                        help="Enable ops mode to generate champion report (default: off)")
    parser.add_argument("--ops-champion-variant", default="xgb",
                        help="Champion variant for ops report (default: xgb)")
    parser.add_argument("--ops-overlay-mode", default="policy_best", choices=["baseline", "policy_best"],
                        help="Overlay mode for ops report (default: policy_best)")
    parser.add_argument("--ops-calibration-mode", default="raw", choices=["raw", "calibrated"],
                        help="Calibration mode for ops report (default: raw)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Shadow Risk SPY-Only Evaluation Runner")
    print("=" * 60)
    print(f"spy_csv_path:  {args.spy_csv_path}")
    print(f"spy_ticker:    {args.spy_ticker}")
    print(f"as_of_date:    {args.as_of_date}")
    print(f"train_end:     {args.train_end}")
    print(f"val_end:       {args.val_end}")
    print(f"output_root:   {args.output_root_dir}")
    print(f"seed:          {args.seed}")
    print(f"horizon_days:  {args.horizon_days}")
    print("=" * 60)
    print()
    
    try:
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=args.spy_csv_path,
            spy_ticker=args.spy_ticker,
            as_of_date=args.as_of_date,
            train_end=args.train_end,
            val_end=args.val_end,
            output_root_dir=args.output_root_dir,
            seed=args.seed,
            horizon_days=args.horizon_days,
            ops_mode=args.ops_mode,
            ops_champion_variant=args.ops_champion_variant,
            ops_overlay_mode=args.ops_overlay_mode,
            ops_calibration_mode=args.ops_calibration_mode,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print()
    
    for variant in ["logit", "mlp"]:
        print(f"[{variant.upper()}] Artifacts:")
        paths = result["models"][variant]
        for key, path in paths.items():
            print(f"  {key}: {path}")
        print()
    
    # Load and compare decision gates
    print("=" * 60)
    print(f"DECISION GATE COMPARISON (horizon={args.horizon_days}d)")
    print("=" * 60)
    
    for variant in ["logit", "mlp"]:
        decision_path = result["models"][variant]["shadow_risk_decision_gate_json"]
        with open(decision_path) as f:
            decision = json.load(f)
        
        computed = decision.get("computed", {})
        checks = decision.get("checks", {})
        
        # Load overlay metrics for additional reporting
        overlay_metrics_path = result["models"][variant]["shadow_risk_overlay_metrics_json"]
        overlay_cagr_vol = None
        overlay_mdd = None
        try:
            with open(overlay_metrics_path) as f:
                overlay_metrics = json.load(f)
            overlay_cagr_vol = overlay_metrics.get("test", {}).get("cagr_over_vol")
            overlay_mdd = overlay_metrics.get("test", {}).get("max_drawdown")
        except Exception:
            pass
        
        print(f"\n[{variant.upper()}] horizon={args.horizon_days}d")
        print(f"  decision:       {decision.get('decision')}")
        print(f"  test_ece:       {computed.get('test_ece')}")
        print(f"  test_cagr/vol:  {overlay_cagr_vol}")
        print(f"  test_max_dd:    {overlay_mdd}")
        print(f"  reasons:        {decision.get('reasons')}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
