"""
Shadow Risk Overlay Policy Recommendation Builder

Compares baseline overlay performance against policy sweep candidates
and produces a deterministic recommendation artifact.

This is SHADOW-ONLY diagnostics: NO trading impact on main system.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Schema version for output JSON
SCHEMA_VERSION = "9.6.10"


def build_overlay_policy_recommendation(
    *,
    overlay_metrics_json_path: str,
    overlay_diagnostics_json_path: str,
    policy_sweep_results_csv_path: str,
    policy_sweep_summary_json_path: str,
    output_json_path: str,
    max_dd_tolerance: float = 0.02,
    turnover_multiplier_cap: float = 1.5,
    require_higher_avg_exposure: bool = True,
    top_n: int = 5,
    seed: int = 42,
) -> dict:
    """
    Build policy recommendation comparing baseline vs sweep candidates.

    Returns deterministic recommendation JSON with schema_version="9.6.10".
    """
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "cagr_over_vol": None,
            "cagr": None,
            "vol": None,
            "max_dd": None,
            "avg_exposure": None,
            "turnover_proxy": None,
        },
        "candidates": {
            "top_by_cagr_over_vol": [],
            "top_by_max_dd": [],
            "top_by_cagr": [],
        },
        "recommended_policy": None,
        "selection_rules": {
            "max_dd_tolerance": max_dd_tolerance,
            "turnover_multiplier_cap": turnover_multiplier_cap,
            "require_higher_avg_exposure": require_higher_avg_exposure,
            "top_n": top_n,
        },
        "warnings": [],
    }

    try:
        # =====================================================================
        # Load baseline metrics
        # =====================================================================
        baseline_metrics = _load_baseline_metrics(overlay_metrics_json_path)
        if baseline_metrics.get("_error"):
            result["warnings"].append(baseline_metrics["_error"])
        else:
            result["baseline"]["cagr_over_vol"] = baseline_metrics.get("cagr_over_vol")
            result["baseline"]["cagr"] = baseline_metrics.get("cagr")
            result["baseline"]["vol"] = baseline_metrics.get("vol")
            result["baseline"]["max_dd"] = baseline_metrics.get("max_dd")

        # =====================================================================
        # Load baseline diagnostics
        # =====================================================================
        baseline_diagnostics = _load_baseline_diagnostics(overlay_diagnostics_json_path)
        if baseline_diagnostics.get("_error"):
            result["warnings"].append(baseline_diagnostics["_error"])
        else:
            result["baseline"]["avg_exposure"] = baseline_diagnostics.get("avg_exposure")
            result["baseline"]["turnover_proxy"] = baseline_diagnostics.get("turnover_proxy")

        # =====================================================================
        # Load policy sweep results
        # =====================================================================
        sweep_df = _load_sweep_results(policy_sweep_results_csv_path)
        if sweep_df is None:
            result["warnings"].append(
                f"SR_POLICY_REC_FAIL:sweep_results_not_found path={policy_sweep_results_csv_path}"
            )
            _write_json(result, output_json_path)
            return result

        if len(sweep_df) == 0:
            result["warnings"].append("SR_POLICY_REC_WARN:sweep_results_empty")
            _write_json(result, output_json_path)
            return result

        # =====================================================================
        # Add deltas vs baseline to each policy
        # =====================================================================
        policies = _compute_policies_with_deltas(sweep_df, result["baseline"])

        # =====================================================================
        # Build candidate lists (sorted, deterministic)
        # =====================================================================
        result["candidates"]["top_by_cagr_over_vol"] = _get_top_n(
            policies, "cagr_over_vol", top_n, maximize=True
        )
        result["candidates"]["top_by_max_dd"] = _get_top_n(
            policies, "max_dd", top_n, maximize=True  # highest = least negative
        )
        result["candidates"]["top_by_cagr"] = _get_top_n(
            policies, "cagr", top_n, maximize=True
        )

        # =====================================================================
        # Select recommended policy
        # =====================================================================
        result["recommended_policy"] = _select_recommended_policy(
            policies=policies,
            baseline=result["baseline"],
            max_dd_tolerance=max_dd_tolerance,
            turnover_multiplier_cap=turnover_multiplier_cap,
            require_higher_avg_exposure=require_higher_avg_exposure,
            sweep_summary_path=policy_sweep_summary_json_path,
        )

        # =====================================================================
        # Write output
        # =====================================================================
        _write_json(result, output_json_path)
        return result

    except Exception as e:
        error_msg = str(e)[:200]
        result["warnings"].append(f"SR_POLICY_REC_FAIL:exception msg={error_msg}")
        _write_json(result, output_json_path)
        return result


def _load_baseline_metrics(path: str) -> dict:
    """Load baseline overlay metrics JSON."""
    try:
        if not Path(path).exists():
            return {"_error": f"SR_POLICY_REC_FAIL:metrics_not_found path={path}"}

        with open(path) as f:
            data = json.load(f)

        # Extract test metrics (may be at different levels)
        if "test" in data:
            test = data["test"]
        else:
            test = data

        return {
            "cagr_over_vol": _safe_float(test.get("cagr_over_vol")),
            "cagr": _safe_float(test.get("cagr")),
            "vol": _safe_float(test.get("ann_vol") or test.get("vol")),
            "max_dd": _safe_float(test.get("max_drawdown") or test.get("max_dd")),
        }
    except Exception as e:
        return {"_error": f"SR_POLICY_REC_FAIL:metrics_parse_error msg={str(e)[:100]}"}


def _load_baseline_diagnostics(path: str) -> dict:
    """Load baseline overlay diagnostics JSON."""
    try:
        if not Path(path).exists():
            return {"_error": f"SR_POLICY_REC_FAIL:diagnostics_not_found path={path}"}

        with open(path) as f:
            data = json.load(f)

        diagnostics = data.get("diagnostics", data)
        return {
            "avg_exposure": _safe_float(diagnostics.get("avg_exposure")),
            "turnover_proxy": _safe_float(diagnostics.get("turnover_proxy")),
        }
    except Exception as e:
        return {"_error": f"SR_POLICY_REC_FAIL:diagnostics_parse_error msg={str(e)[:100]}"}


def _load_sweep_results(path: str) -> pd.DataFrame | None:
    """Load policy sweep results CSV."""
    try:
        if not Path(path).exists():
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def _compute_policies_with_deltas(df: pd.DataFrame, baseline: dict) -> list[dict]:
    """Convert DataFrame rows to policy dicts with deltas vs baseline."""
    policies = []

    for _, row in df.iterrows():
        policy = {
            "policy_id": int(row.get("policy_id", 0)),
            "beta_cap": _safe_float(row.get("beta_cap")),
            "threshold": _safe_float(row.get("threshold")),
            "n_obs": int(row.get("n_obs", 0)),
            "avg_exposure": _safe_float(row.get("avg_exposure")),
            "std_exposure": _safe_float(row.get("std_exposure")),
            "turnover_proxy": _safe_float(row.get("turnover_proxy")),
            "n_switches": int(row.get("n_switches", 0)),
            "cagr_over_vol": _safe_float(row.get("cagr_over_vol")),
            "cagr": _safe_float(row.get("cagr")),
            "vol": _safe_float(row.get("vol")),
            "max_dd": _safe_float(row.get("max_dd")),
            "deltas_vs_baseline": {
                "delta_avg_exposure": _compute_delta(
                    _safe_float(row.get("avg_exposure")),
                    baseline.get("avg_exposure"),
                ),
                "delta_turnover_proxy": _compute_delta(
                    _safe_float(row.get("turnover_proxy")),
                    baseline.get("turnover_proxy"),
                ),
                "delta_cagr_over_vol": _compute_delta(
                    _safe_float(row.get("cagr_over_vol")),
                    baseline.get("cagr_over_vol"),
                ),
                "delta_max_dd": _compute_delta(
                    _safe_float(row.get("max_dd")),
                    baseline.get("max_dd"),
                ),
            },
        }
        policies.append(policy)

    return policies


def _get_top_n(policies: list[dict], metric: str, n: int, maximize: bool) -> list[dict]:
    """Get top N policies by a metric with deterministic tie-breaking."""
    # Filter to policies with valid metric
    valid = [p for p in policies if p.get(metric) is not None]

    # Sort: by metric (desc if maximize, asc otherwise), then by policy_id (asc)
    valid.sort(
        key=lambda p: (
            -p[metric] if maximize else p[metric],
            p["policy_id"],
        )
    )

    return valid[:n]


def _select_recommended_policy(
    *,
    policies: list[dict],
    baseline: dict,
    max_dd_tolerance: float,
    turnover_multiplier_cap: float,
    require_higher_avg_exposure: bool,
    sweep_summary_path: str,
) -> dict | None:
    """Select recommended policy using guardrails."""
    # Filter to valid policies
    candidates = [
        p for p in policies
        if p.get("cagr_over_vol") is not None and p.get("max_dd") is not None
    ]

    if len(candidates) == 0:
        return _fallback_to_sweep_best(sweep_summary_path, policies)

    baseline_max_dd = baseline.get("max_dd")
    baseline_turnover = baseline.get("turnover_proxy")
    baseline_avg_exposure = baseline.get("avg_exposure")

    # Apply guardrails
    if baseline_max_dd is not None:
        # Keep policies with max_dd >= baseline_max_dd - tolerance
        # (max_dd is negative, so more negative is worse)
        threshold = baseline_max_dd - max_dd_tolerance
        candidates = [p for p in candidates if p["max_dd"] >= threshold]

    if baseline_turnover is not None and baseline_turnover > 0:
        # Keep policies with turnover <= baseline * multiplier cap
        threshold = baseline_turnover * turnover_multiplier_cap
        candidates = [p for p in candidates if p["turnover_proxy"] <= threshold]

    if require_higher_avg_exposure and baseline_avg_exposure is not None:
        # Keep policies with avg_exposure > baseline
        candidates = [p for p in candidates if p["avg_exposure"] > baseline_avg_exposure]

    if len(candidates) == 0:
        return _fallback_to_sweep_best(sweep_summary_path, policies)

    # Choose best by cagr_over_vol with deterministic tie-breaking
    # Tie-break: higher max_dd, then lower turnover_proxy, then lower policy_id
    candidates.sort(
        key=lambda p: (
            -p["cagr_over_vol"],
            -p["max_dd"],  # higher max_dd is better
            p["turnover_proxy"],  # lower turnover is better
            p["policy_id"],
        )
    )

    return candidates[0]


def _fallback_to_sweep_best(sweep_summary_path: str, policies: list[dict]) -> dict | None:
    """Fall back to sweep best_by_cagr_over_vol if guardrails filter everything."""
    try:
        if not Path(sweep_summary_path).exists():
            return None

        with open(sweep_summary_path) as f:
            summary = json.load(f)

        best = summary.get("best_by_cagr_over_vol")
        if best is None:
            return None

        policy_id = best.get("policy_id")
        if policy_id is None:
            return None

        # Find in policies list
        for p in policies:
            if p["policy_id"] == policy_id:
                return p

        return None
    except Exception:
        return None


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


def _compute_delta(new_val: float | None, baseline_val: float | None) -> float | None:
    """Compute delta between new value and baseline."""
    if new_val is None or baseline_val is None:
        return None
    return new_val - baseline_val


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
