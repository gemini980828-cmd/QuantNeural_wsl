"""
Shadow Risk Overlay Policy Sweep

Sweeps a grid of thresholded p_risk_off → exposure mappings to identify
optimal policy configurations for the shadow risk overlay.

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
SCHEMA_VERSION = "9.6.9"

# Default sweep grid
DEFAULT_BETA_CAPS = [0.10, 0.20, 0.35]
DEFAULT_THRESHOLDS = [0.0, 0.4, 0.5, 0.6]


def run_overlay_policy_sweep(
    *,
    shadow_csv_path: str,
    overlay_csv_path: str,
    output_results_csv_path: str,
    output_summary_json_path: str,
    p_column: str = "p_risk_off",
    spy_ret_column: str = "spy_ret_1d",
    cash_daily_return: float = 0.0,
    beta_caps: list[float] | None = None,
    thresholds: list[float] | None = None,
    eps: float = 1e-12,
    seed: int = 42,
) -> dict:
    """
    Sweep thresholded p_risk_off → exposure mappings and compute metrics.

    For each (beta_cap, threshold):
        p_adj = 0 if p <= threshold else (p - threshold) / (1 - threshold)
        w_beta = beta_cap * p_adj
        exposure = clip(1 - w_beta, 0, 1)

    Uses shifted-weight semantics:
        overlay_ret[t] = exposure[t-1] * spy_ret[t] + (1 - exposure[t-1]) * cash
        overlay_ret[0] = 0.0

    Returns dict matching the summary JSON schema.
    """
    # Use defaults if not provided
    if beta_caps is None:
        beta_caps = DEFAULT_BETA_CAPS.copy()
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # Initialize result structure
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "beta_caps": beta_caps,
            "thresholds": thresholds,
            "p_column": p_column,
            "spy_ret_column": spy_ret_column,
            "cash_daily_return": cash_daily_return,
            "eps": eps,
            "seed": seed,
        },
        "best_by_cagr_over_vol": None,
        "best_by_max_dd": None,
        "best_by_cagr": None,
        "warnings": [],
    }

    # Column names for results CSV
    result_columns = [
        "policy_id", "beta_cap", "threshold",
        "n_obs", "avg_exposure", "std_exposure", "frac_full_exposure",
        "n_switches", "turnover_proxy", "avg_abs_delta_exposure",
        "total_return", "cagr", "vol", "cagr_over_vol", "max_dd",
    ]

    try:
        # =====================================================================
        # Load shadow CSV
        # =====================================================================
        shadow_path = Path(shadow_csv_path)
        if not shadow_path.exists():
            result["warnings"].append(
                f"SR_POLICY_SWEEP_FAIL:shadow_file_not_found path={shadow_csv_path}"
            )
            _write_empty_results(result_columns, output_results_csv_path)
            _write_json(result, output_summary_json_path)
            return result

        shadow_df = pd.read_csv(shadow_path)
        if p_column not in shadow_df.columns:
            result["warnings"].append(
                f"SR_POLICY_SWEEP_FAIL:missing_column column={p_column} in shadow CSV"
            )
            _write_empty_results(result_columns, output_results_csv_path)
            _write_json(result, output_summary_json_path)
            return result

        # =====================================================================
        # Load overlay CSV
        # =====================================================================
        overlay_path = Path(overlay_csv_path)
        if not overlay_path.exists():
            result["warnings"].append(
                f"SR_POLICY_SWEEP_FAIL:overlay_file_not_found path={overlay_csv_path}"
            )
            _write_empty_results(result_columns, output_results_csv_path)
            _write_json(result, output_summary_json_path)
            return result

        overlay_df = pd.read_csv(overlay_path)
        if spy_ret_column not in overlay_df.columns:
            result["warnings"].append(
                f"SR_POLICY_SWEEP_FAIL:missing_column column={spy_ret_column} in overlay CSV"
            )
            _write_empty_results(result_columns, output_results_csv_path)
            _write_json(result, output_summary_json_path)
            return result

        # =====================================================================
        # Align data on date
        # =====================================================================
        if "date" not in shadow_df.columns or "date" not in overlay_df.columns:
            result["warnings"].append("SR_POLICY_SWEEP_FAIL:missing_date_column")
            _write_empty_results(result_columns, output_results_csv_path)
            _write_json(result, output_summary_json_path)
            return result

        shadow_df["date"] = pd.to_datetime(shadow_df["date"])
        overlay_df["date"] = pd.to_datetime(overlay_df["date"])

        merged = pd.merge(
            shadow_df[["date", p_column]],
            overlay_df[["date", spy_ret_column]],
            on="date",
            how="inner",
        ).sort_values("date").reset_index(drop=True)

        if len(merged) == 0:
            result["warnings"].append("SR_POLICY_SWEEP_WARN:no_common_dates")
            _write_empty_results(result_columns, output_results_csv_path)
            _write_json(result, output_summary_json_path)
            return result

        p_values = merged[p_column].values
        spy_ret = merged[spy_ret_column].values
        n_obs = len(merged)

        # =====================================================================
        # Sweep grid
        # =====================================================================
        all_results = []
        policy_id = 0

        for beta_cap in sorted(beta_caps):
            for threshold in sorted(thresholds):
                policy_id += 1
                metrics = _compute_policy_metrics(
                    p_values=p_values,
                    spy_ret=spy_ret,
                    beta_cap=beta_cap,
                    threshold=threshold,
                    cash_daily_return=cash_daily_return,
                    eps=eps,
                )
                metrics["policy_id"] = policy_id
                metrics["beta_cap"] = beta_cap
                metrics["threshold"] = threshold
                metrics["n_obs"] = n_obs
                all_results.append(metrics)

        # =====================================================================
        # Build results DataFrame
        # =====================================================================
        results_df = pd.DataFrame(all_results)[result_columns]

        # =====================================================================
        # Find best policies
        # =====================================================================
        result["best_by_cagr_over_vol"] = _find_best(
            results_df, "cagr_over_vol", maximize=True
        )
        result["best_by_max_dd"] = _find_best(
            results_df, "max_dd", maximize=True  # highest = least negative
        )
        result["best_by_cagr"] = _find_best(
            results_df, "cagr", maximize=True
        )

        # =====================================================================
        # Write outputs
        # =====================================================================
        _write_results_csv(results_df, output_results_csv_path)
        _write_json(result, output_summary_json_path)

        return result

    except Exception as e:
        error_msg = str(e)[:200]
        result["warnings"].append(f"SR_POLICY_SWEEP_FAIL:exception msg={error_msg}")
        _write_empty_results(result_columns, output_results_csv_path)
        _write_json(result, output_summary_json_path)
        return result


def _compute_policy_metrics(
    *,
    p_values: list | Any,
    spy_ret: list | Any,
    beta_cap: float,
    threshold: float,
    cash_daily_return: float,
    eps: float,
) -> dict:
    """Compute metrics for a single policy."""
    n = len(p_values)

    # Compute exposure series
    exposure = []
    for p in p_values:
        if p <= threshold:
            p_adj = 0.0
        else:
            p_adj = (p - threshold) / (1.0 - threshold) if threshold < 1.0 else 0.0
        w_beta = beta_cap * p_adj
        exp = max(0.0, min(1.0, 1.0 - w_beta))
        exposure.append(exp)
    exposure = list(exposure)

    # Exposure statistics
    avg_exposure = sum(exposure) / n if n > 0 else 0.0
    variance = sum((e - avg_exposure) ** 2 for e in exposure) / n if n > 0 else 0.0
    std_exposure = math.sqrt(variance)
    frac_full_exposure = sum(1 for e in exposure if e >= 1.0 - eps) / n if n > 0 else 0.0

    # Switches and turnover (over deltas)
    n_switches = 0
    turnover_proxy = 0.0
    if n > 1:
        for i in range(1, n):
            delta = abs(exposure[i] - exposure[i - 1])
            turnover_proxy += delta
            if delta > eps:
                n_switches += 1
    n_deltas = n - 1 if n > 1 else 1
    avg_abs_delta = turnover_proxy / n_deltas

    # Shifted-weight overlay returns
    overlay_ret = [0.0]  # First return is 0 (no prior exposure)
    for t in range(1, n):
        prev_exp = exposure[t - 1]
        r = prev_exp * spy_ret[t] + (1 - prev_exp) * cash_daily_return
        overlay_ret.append(r)

    # Equity curve
    equity = [1.0]
    for r in overlay_ret[1:]:
        equity.append(equity[-1] * (1 + r))

    # Metrics
    total_return = equity[-1] / equity[0] - 1 if len(equity) > 0 else 0.0

    # CAGR (252 trading days)
    years = n / 252.0 if n > 0 else 1.0
    if equity[-1] > 0 and years > 0:
        cagr = (equity[-1] ** (1.0 / years)) - 1
    else:
        cagr = 0.0

    # Volatility (population std * sqrt(252))
    if len(overlay_ret) > 1:
        mean_ret = sum(overlay_ret) / len(overlay_ret)
        var_ret = sum((r - mean_ret) ** 2 for r in overlay_ret) / len(overlay_ret)
        vol = math.sqrt(var_ret) * math.sqrt(252)
    else:
        vol = 0.0

    # CAGR/Vol
    cagr_over_vol = cagr / vol if vol > eps else None

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
        "avg_exposure": avg_exposure,
        "std_exposure": std_exposure,
        "frac_full_exposure": frac_full_exposure,
        "n_switches": n_switches,
        "turnover_proxy": turnover_proxy,
        "avg_abs_delta_exposure": avg_abs_delta,
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "cagr_over_vol": cagr_over_vol,
        "max_dd": max_dd,
    }


def _find_best(df: pd.DataFrame, metric: str, maximize: bool) -> dict | None:
    """Find best policy by a metric."""
    valid = df[df[metric].notna()]
    if len(valid) == 0:
        return None

    if maximize:
        idx = valid[metric].idxmax()
    else:
        idx = valid[metric].idxmin()

    row = valid.loc[idx]
    return {
        "policy_id": int(row["policy_id"]),
        "beta_cap": float(row["beta_cap"]),
        "threshold": float(row["threshold"]),
        "cagr_over_vol": float(row["cagr_over_vol"]) if pd.notna(row["cagr_over_vol"]) else None,
        "cagr": float(row["cagr"]),
        "vol": float(row["vol"]),
        "max_dd": float(row["max_dd"]),
    }


def _write_results_csv(df: pd.DataFrame, path: str) -> None:
    """Write results CSV deterministically."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_empty_results(columns: list, path: str) -> None:
    """Write empty results CSV with headers only."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


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
