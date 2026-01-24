"""
Shadow Risk Overlay Policy Apply

Applies the recommended overlay policy mapping and produces
counterfactual overlay artifacts for apples-to-apples comparison.

This is SHADOW-ONLY diagnostics: NO trading impact on main system.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.shadow_risk_overlay_diagnostics import compute_overlay_exposure_diagnostics

logger = logging.getLogger(__name__)

# Schema version for metrics JSON
SCHEMA_VERSION = "9.6.11"


def apply_recommended_overlay_policy(
    shadow_csv_path: str,
    overlay_csv_path: str,
    policy_recommendation_json_path: str,
    *,
    output_overlay_csv_path: str,
    output_metrics_json_path: str,
    output_diagnostics_json_path: str,
    p_column: str = "p_risk_off",
    spy_ret_column: str = "spy_ret_1d",
    baseline_exposure_column: str = "exposure_suggested",
    cash_daily_return: float = 0.0,
    eps: float = 1e-12,
    seed: int = 42,
) -> dict:
    """
    Apply recommended overlay policy and produce counterfactual artifacts.

    Returns metrics dict with schema_version="9.6.11".
    """
    # Initialize result structure
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recommended_policy": None,
        "baseline": None,
        "policy": None,
        "deltas_vs_baseline": None,
        "warnings": [],
    }

    # CSV columns for output
    csv_columns = [
        "date", "spy_ret_1d", "p_risk_off", "exposure_baseline",
        "exposure_suggested", "overlay_ret_1d", "overlay_equity",
        "policy_id", "beta_cap", "threshold",
    ]

    try:
        # =====================================================================
        # Load recommendation
        # =====================================================================
        policy = _load_recommendation(policy_recommendation_json_path)
        if policy is None:
            result["warnings"].append(
                f"SR_POLICY_APPLY_FAIL:recommendation_load_failed path={policy_recommendation_json_path}"
            )
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        result["recommended_policy"] = {
            "policy_id": policy.get("policy_id"),
            "beta_cap": policy.get("beta_cap"),
            "threshold": policy.get("threshold"),
        }

        beta_cap = policy.get("beta_cap")
        threshold = policy.get("threshold")

        if beta_cap is None or threshold is None:
            result["warnings"].append(
                "SR_POLICY_APPLY_FAIL:missing_policy_params beta_cap or threshold is null"
            )
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        # =====================================================================
        # Load shadow CSV
        # =====================================================================
        if not Path(shadow_csv_path).exists():
            result["warnings"].append(
                f"SR_POLICY_APPLY_FAIL:shadow_csv_not_found path={shadow_csv_path}"
            )
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        shadow_df = pd.read_csv(shadow_csv_path)
        if p_column not in shadow_df.columns:
            result["warnings"].append(
                f"SR_POLICY_APPLY_FAIL:missing_column column={p_column} in shadow CSV"
            )
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        # =====================================================================
        # Load overlay CSV
        # =====================================================================
        if not Path(overlay_csv_path).exists():
            result["warnings"].append(
                f"SR_POLICY_APPLY_FAIL:overlay_csv_not_found path={overlay_csv_path}"
            )
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        overlay_df = pd.read_csv(overlay_csv_path)
        if spy_ret_column not in overlay_df.columns:
            result["warnings"].append(
                f"SR_POLICY_APPLY_FAIL:missing_column column={spy_ret_column} in overlay CSV"
            )
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        # =====================================================================
        # Merge and align on date
        # =====================================================================
        if "date" not in shadow_df.columns or "date" not in overlay_df.columns:
            result["warnings"].append("SR_POLICY_APPLY_FAIL:missing_date_column")
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        shadow_df["date"] = pd.to_datetime(shadow_df["date"])
        overlay_df["date"] = pd.to_datetime(overlay_df["date"])

        # Select columns from each dataframe
        shadow_cols = ["date", p_column]
        overlay_cols = ["date", spy_ret_column]
        if baseline_exposure_column in overlay_df.columns:
            overlay_cols.append(baseline_exposure_column)

        merged = pd.merge(
            shadow_df[shadow_cols],
            overlay_df[overlay_cols],
            on="date",
            how="inner",
        ).sort_values("date").reset_index(drop=True)

        if len(merged) == 0:
            result["warnings"].append("SR_POLICY_APPLY_WARN:no_common_dates")
            _write_empty_outputs(
                result, csv_columns, output_overlay_csv_path,
                output_metrics_json_path, output_diagnostics_json_path
            )
            return result

        n = len(merged)

        # =====================================================================
        # Compute policy exposure
        # =====================================================================
        exposure_policy = []
        for p in merged[p_column].values:
            if p <= threshold:
                p_adj = 0.0
            else:
                p_adj = (p - threshold) / (1.0 - threshold) if threshold < 1.0 else 0.0
            w_beta = beta_cap * p_adj
            exp = max(0.0, min(1.0, 1.0 - w_beta))
            exposure_policy.append(exp)

        # =====================================================================
        # Compute overlay returns (shifted-weight semantics)
        # =====================================================================
        spy_ret = merged[spy_ret_column].values
        overlay_ret = [0.0]  # First return is 0
        for t in range(1, n):
            prev_exp = exposure_policy[t - 1]
            r = prev_exp * spy_ret[t] + (1 - prev_exp) * cash_daily_return
            overlay_ret.append(r)

        # =====================================================================
        # Compute equity curve
        # =====================================================================
        equity = [1.0]
        for r in overlay_ret[1:]:
            equity.append(equity[-1] * (1 + r))

        # =====================================================================
        # Build output DataFrame
        # =====================================================================
        out_df = pd.DataFrame({
            "date": merged["date"].dt.strftime("%Y-%m-%d"),
            "spy_ret_1d": merged[spy_ret_column].values,
            "p_risk_off": merged[p_column].values,
            "exposure_baseline": merged[baseline_exposure_column].values if baseline_exposure_column in merged.columns else [float("nan")] * n,
            "exposure_suggested": exposure_policy,
            "overlay_ret_1d": overlay_ret,
            "overlay_equity": equity,
            "policy_id": [policy.get("policy_id")] * n,
            "beta_cap": [beta_cap] * n,
            "threshold": [threshold] * n,
        })

        # Ensure column order
        out_df = out_df[csv_columns]

        # =====================================================================
        # Write CSV
        # =====================================================================
        _write_csv(out_df, output_overlay_csv_path)

        # =====================================================================
        # Compute metrics for policy
        # =====================================================================
        result["policy"] = _compute_metrics(
            exposure=exposure_policy,
            overlay_ret=overlay_ret,
            equity=equity,
            eps=eps,
        )

        # =====================================================================
        # Compute metrics for baseline (if available)
        # =====================================================================
        if baseline_exposure_column in merged.columns:
            baseline_exposure = merged[baseline_exposure_column].values.tolist()
            # Compute baseline overlay returns with same method
            baseline_overlay_ret = [0.0]
            for t in range(1, n):
                prev_exp = baseline_exposure[t - 1]
                r = prev_exp * spy_ret[t] + (1 - prev_exp) * cash_daily_return
                baseline_overlay_ret.append(r)
            baseline_equity = [1.0]
            for r in baseline_overlay_ret[1:]:
                baseline_equity.append(baseline_equity[-1] * (1 + r))

            result["baseline"] = _compute_metrics(
                exposure=baseline_exposure,
                overlay_ret=baseline_overlay_ret,
                equity=baseline_equity,
                eps=eps,
            )

            # Compute deltas
            result["deltas_vs_baseline"] = _compute_deltas(
                result["policy"], result["baseline"]
            )

        # =====================================================================
        # Write metrics JSON
        # =====================================================================
        _write_json(result, output_metrics_json_path)

        # =====================================================================
        # Compute diagnostics using existing module
        # =====================================================================
        try:
            compute_overlay_exposure_diagnostics(
                output_overlay_csv_path,
                output_json_path=output_diagnostics_json_path,
                exposure_column="exposure_suggested",
                seed=seed,
            )
        except Exception as e:
            result["warnings"].append(
                f"SR_POLICY_APPLY_WARN:diagnostics_failed msg={str(e)[:100]}"
            )
            _write_fallback_diagnostics(output_diagnostics_json_path)

        return result

    except Exception as e:
        error_msg = str(e)[:200]
        result["warnings"].append(f"SR_POLICY_APPLY_FAIL:exception msg={error_msg}")
        _write_empty_outputs(
            result, csv_columns, output_overlay_csv_path,
            output_metrics_json_path, output_diagnostics_json_path
        )
        return result


def _load_recommendation(path: str) -> dict | None:
    """Load recommended policy from recommendation JSON."""
    try:
        if not Path(path).exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return data.get("recommended_policy")
    except Exception:
        return None


def _compute_metrics(
    *,
    exposure: list,
    overlay_ret: list,
    equity: list,
    eps: float,
) -> dict:
    """Compute overlay metrics."""
    n = len(exposure)

    # Exposure statistics
    avg_exposure = sum(exposure) / n if n > 0 else None
    variance = sum((e - avg_exposure) ** 2 for e in exposure) / n if n > 0 and avg_exposure is not None else 0.0
    std_exposure = math.sqrt(variance) if variance >= 0 else None

    # Switches and turnover
    n_switches = 0
    turnover_proxy = 0.0
    if n > 1:
        for i in range(1, n):
            delta = abs(exposure[i] - exposure[i - 1])
            turnover_proxy += delta
            if delta > eps:
                n_switches += 1
    n_deltas = n - 1 if n > 1 else 1
    avg_abs_delta = turnover_proxy / n_deltas if n_deltas > 0 else None

    # Total return
    total_return = equity[-1] / equity[0] - 1 if len(equity) > 0 and equity[0] > 0 else None

    # CAGR (252 trading days)
    years = n / 252.0 if n > 0 else 1.0
    if equity[-1] > 0 and years > 0:
        cagr = (equity[-1] ** (1.0 / years)) - 1
    else:
        cagr = None

    # Volatility (population std * sqrt(252))
    if len(overlay_ret) > 1:
        mean_ret = sum(overlay_ret) / len(overlay_ret)
        var_ret = sum((r - mean_ret) ** 2 for r in overlay_ret) / len(overlay_ret)
        vol = math.sqrt(var_ret) * math.sqrt(252)
    else:
        vol = None

    # CAGR/Vol
    cagr_over_vol = cagr / vol if cagr is not None and vol is not None and vol > eps else None

    # Max drawdown
    max_dd = 0.0
    peak = equity[0]
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "n_obs": n,
        "total_return": _safe_float(total_return),
        "cagr": _safe_float(cagr),
        "vol": _safe_float(vol),
        "cagr_over_vol": _safe_float(cagr_over_vol),
        "max_dd": _safe_float(max_dd),
        "avg_exposure": _safe_float(avg_exposure),
        "std_exposure": _safe_float(std_exposure),
        "turnover_proxy": _safe_float(turnover_proxy),
        "n_switches": n_switches,
        "avg_abs_delta_exposure": _safe_float(avg_abs_delta),
    }


def _compute_deltas(policy: dict, baseline: dict) -> dict:
    """Compute deltas between policy and baseline."""
    deltas = {}
    for key in ["total_return", "cagr", "vol", "cagr_over_vol", "max_dd",
                "avg_exposure", "std_exposure", "turnover_proxy", "avg_abs_delta_exposure"]:
        p_val = policy.get(key)
        b_val = baseline.get(key)
        if p_val is not None and b_val is not None:
            deltas[f"delta_{key}"] = p_val - b_val
        else:
            deltas[f"delta_{key}"] = None

    # n_switches is int
    if policy.get("n_switches") is not None and baseline.get("n_switches") is not None:
        deltas["delta_n_switches"] = policy["n_switches"] - baseline["n_switches"]
    else:
        deltas["delta_n_switches"] = None

    return deltas


def _safe_float(val: Any) -> float | None:
    """Convert value to float, returning None for NaN/inf/invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _write_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV deterministically."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def _write_json(data: dict, path: str) -> None:
    """Write JSON deterministically."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)


def _json_default(obj: Any) -> Any:
    """Handle non-serializable types."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_empty_outputs(
    result: dict,
    csv_columns: list,
    csv_path: str,
    metrics_path: str,
    diagnostics_path: str,
) -> None:
    """Write empty outputs on failure."""
    # Write empty CSV
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False, lineterminator="\n")

    # Write metrics JSON
    _write_json(result, metrics_path)

    # Write fallback diagnostics
    _write_fallback_diagnostics(diagnostics_path)


def _write_fallback_diagnostics(path: str) -> None:
    """Write fallback diagnostics JSON with null values."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fallback = {
        "schema_version": "9.6.7",
        "diagnostics": {
            "n_obs": 0,
            "avg_exposure": None,
            "std_exposure": None,
            "frac_exposure_lt_1": None,
            "n_switches": 0,
            "turnover_proxy": None,
            "avg_abs_delta_exposure": None,
        },
        "warnings": ["SR_POLICY_APPLY_WARN:fallback_diagnostics"],
    }
    json_str = json.dumps(
        fallback,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)
