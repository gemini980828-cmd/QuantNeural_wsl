"""
Shadow Risk Overlay Diagnostics

Computes exposure-related diagnostic metrics from overlay CSV files to analyze
risk-gating behavior (exposure distribution, switching frequency, turnover).

This is SHADOW-ONLY diagnostics: NO trading impact.
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
SCHEMA_VERSION = "9.6.7"

# Threshold for detecting exposure changes
EPS = 1e-12


def compute_overlay_exposure_diagnostics(
    overlay_csv_path: str,
    *,
    output_json_path: str,
    exposure_column: str = "exposure_suggested",
    equity_column: str = "overlay_equity",
    seed: int = 42,
) -> dict:
    """
    Reads an overlay CSV and computes diagnostic stats for exposure analysis.

    Parameters
    ----------
    overlay_csv_path : str
        Path to overlay CSV file with exposure and equity columns.
    output_json_path : str
        Path to write output JSON diagnostics.
    exposure_column : str
        Column name for exposure values (default: "exposure_suggested").
    equity_column : str
        Column name for equity values (default: "overlay_equity").
    seed : int
        Random seed (unused but kept for API consistency).

    Returns
    -------
    dict
        Dictionary containing schema_version, diagnostics, and warnings.
        Same dict is written to output_json_path.

    Notes
    -----
    Fail-safe: Never raises to caller. On any error, writes valid JSON with
    null metrics and informative warnings.
    """
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostics": {
            "n_obs": None,
            "avg_exposure": None,
            "std_exposure": None,
            "frac_exposure_lt_1": None,
            "n_switches": None,
            "turnover_proxy": None,
            "avg_abs_delta_exposure": None,
        },
        "warnings": [],
    }

    try:
        # =====================================================================
        # Load CSV
        # =====================================================================
        csv_path = Path(overlay_csv_path)
        if not csv_path.exists():
            result["warnings"].append(
                f"SR_DIAG_FAIL:file_not_found path={overlay_csv_path}"
            )
            result["diagnostics"]["n_obs"] = 0
            _write_json(result, output_json_path)
            return result

        df = pd.read_csv(csv_path)

        if len(df) == 0:
            result["warnings"].append("SR_DIAG_FAIL:empty_csv")
            result["diagnostics"]["n_obs"] = 0
            _write_json(result, output_json_path)
            return result

        # =====================================================================
        # Validate required columns
        # =====================================================================
        if exposure_column not in df.columns:
            result["warnings"].append(
                f"SR_DIAG_FAIL:missing_column column={exposure_column}"
            )
            result["diagnostics"]["n_obs"] = 0
            _write_json(result, output_json_path)
            return result

        # =====================================================================
        # Extract usable rows (finite exposure values)
        # =====================================================================
        exposure_series = df[exposure_column]
        usable_mask = exposure_series.notna() & exposure_series.apply(
            lambda x: math.isfinite(x) if isinstance(x, (int, float)) else False
        )
        usable_exposure = exposure_series[usable_mask].astype(float)

        n_obs = len(usable_exposure)
        result["diagnostics"]["n_obs"] = n_obs

        if n_obs == 0:
            result["warnings"].append("SR_DIAG_WARN:no_usable_rows")
            _write_json(result, output_json_path)
            return result

        # =====================================================================
        # Compute exposure statistics
        # =====================================================================
        avg_exposure = float(usable_exposure.mean())
        # Use population std (ddof=0) for determinism and simplicity
        std_exposure = float(usable_exposure.std(ddof=0))
        frac_exposure_lt_1 = float((usable_exposure < 1.0).sum()) / n_obs

        result["diagnostics"]["avg_exposure"] = avg_exposure
        result["diagnostics"]["std_exposure"] = std_exposure
        result["diagnostics"]["frac_exposure_lt_1"] = frac_exposure_lt_1

        # =====================================================================
        # Compute switches and turnover (consecutive usable rows only)
        # =====================================================================
        if n_obs < 2:
            result["diagnostics"]["n_switches"] = 0
            result["diagnostics"]["turnover_proxy"] = 0.0
            result["diagnostics"]["avg_abs_delta_exposure"] = 0.0
        else:
            usable_values = usable_exposure.values
            deltas = []
            n_switches = 0

            for i in range(1, len(usable_values)):
                delta = usable_values[i] - usable_values[i - 1]
                deltas.append(abs(delta))
                if abs(delta) > EPS:
                    n_switches += 1

            turnover_proxy = float(sum(deltas))
            n_deltas = len(deltas)
            avg_abs_delta = turnover_proxy / max(n_deltas, 1)

            result["diagnostics"]["n_switches"] = n_switches
            result["diagnostics"]["turnover_proxy"] = turnover_proxy
            result["diagnostics"]["avg_abs_delta_exposure"] = avg_abs_delta

        # =====================================================================
        # Write output
        # =====================================================================
        _write_json(result, output_json_path)
        return result

    except Exception as e:
        # Fail-safe: catch all exceptions and still write valid JSON
        error_msg = str(e)[:200]  # Truncate long error messages
        result["warnings"].append(f"SR_DIAG_FAIL:exception msg={error_msg}")
        result["diagnostics"]["n_obs"] = 0
        _write_json(result, output_json_path)
        return result


def _write_json(data: dict, output_path: str) -> None:
    """Write JSON deterministically (sorted keys, no NaN, consistent formatting)."""
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert None to null in JSON output
    json_str = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)


def _json_default(obj: Any) -> Any:
    """Handle non-serializable types."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
