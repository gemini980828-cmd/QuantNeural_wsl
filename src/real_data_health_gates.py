"""
Real-Data Health Gates for QUANT-NEURAL.

Provides validation functions to ensure:
- Coverage: minimum months of data
- Missingness: acceptable NaN ratios
- PIT Invariance: no look-ahead bias

All functions are deterministic and do not use network or system time.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def expected_h1h2_columns() -> list[str]:
    """
    Return the exact expected column names for H1/H2 features.
    
    Returns
    -------
    list[str]
        Exactly 20 columns: S0_H1..S9_H1, S0_H2..S9_H2
    """
    return [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]


def compute_missing_ratios(frame: pd.DataFrame) -> pd.Series:
    """
    Compute missing (NaN) ratio per column.
    
    Parameters
    ----------
    frame : pd.DataFrame
        Feature frame to analyze.
    
    Returns
    -------
    pd.Series
        Missing ratio (0.0 to 1.0) per column, preserving column order.
    """
    if frame.empty:
        return pd.Series(dtype=float)
    
    n_rows = len(frame)
    missing_counts = frame.isna().sum()
    return missing_counts / n_rows


def run_real_data_health_gates(
    frame: pd.DataFrame,
    *,
    as_of_date: str,
    min_months: int = 18,
    max_feature_missing_ratio: float = 0.20,
    ignore_first_n_rows_for_missing: int = 12,
    min_sector_firms: int = 3,
    max_low_count_month_ratio: float = 1.0,  # Disabled by default (fail-safe)
) -> dict:
    """
    Run all real-data health gates and return a structured report.
    
    Parameters
    ----------
    frame : pd.DataFrame
        Feature frame to validate (index should be month-end timestamps).
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format.
    min_months : int
        Minimum required months of data.
    max_feature_missing_ratio : float
        Maximum allowed missing ratio per column (after ignoring first N rows).
    ignore_first_n_rows_for_missing : int
        Number of initial rows to ignore when computing missing ratios.
    min_sector_firms : int
        Minimum firms per sector for representativeness (default 3).
    max_low_count_month_ratio : float
        Maximum ratio of months with low firm counts (default 1.0, effectively disabled).
    
    Returns
    -------
    dict
        Structured report with keys:
        - passed: bool
        - failed_gates: list[str]
        - metrics: dict with detailed metrics
        - sector_counts_present: bool
        - sector_low_count_ratio: dict (if counts present)
    """
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    
    failed_gates: list[str] = []
    
    n_rows = len(frame)
    n_cols = len(frame.columns)
    
    # =========================================================================
    # Compute metrics
    # =========================================================================
    
    # Index properties
    index_is_datetimeindex = isinstance(frame.index, pd.DatetimeIndex)
    index_is_monotonic = frame.index.is_monotonic_increasing if n_rows > 0 else True
    index_is_unique = frame.index.is_unique if n_rows > 0 else True
    
    # Date range
    start_date = frame.index.min().isoformat() if n_rows > 0 and index_is_datetimeindex else None
    end_date = frame.index.max().isoformat() if n_rows > 0 and index_is_datetimeindex else None
    
    # As-of cutoff check
    max_index_le_as_of = True
    if n_rows > 0 and index_is_datetimeindex:
        max_index_le_as_of = frame.index.max() <= as_of_dt
    
    # Column matching
    expected_cols = expected_h1h2_columns()
    columns_match_expected = list(frame.columns) == expected_cols
    
    # Dtype check
    all_float_dtypes = all(pd.api.types.is_float_dtype(frame[c]) for c in frame.columns) if n_cols > 0 else True
    
    # Missing ratios - full
    missing_ratios_full = compute_missing_ratios(frame).to_dict()
    
    # Missing ratios - after ignoring first N rows
    if n_rows > ignore_first_n_rows_for_missing:
        eval_frame = frame.iloc[ignore_first_n_rows_for_missing:]
        missing_ratios_eval = compute_missing_ratios(eval_frame).to_dict()
        max_missing_ratio_eval = max(missing_ratios_eval.values()) if missing_ratios_eval else 0.0
    else:
        missing_ratios_eval = {}
        max_missing_ratio_eval = 0.0
    
    # =========================================================================
    # Apply gates
    # =========================================================================
    
    # GATE_MIN_MONTHS
    if n_rows < min_months:
        failed_gates.append("GATE_MIN_MONTHS")
    
    # GATE_INDEX_TYPE
    if not index_is_datetimeindex:
        failed_gates.append("GATE_INDEX_TYPE")
    
    # GATE_INDEX_MONOTONIC
    if not index_is_monotonic:
        failed_gates.append("GATE_INDEX_MONOTONIC")
    
    # GATE_INDEX_UNIQUE
    if not index_is_unique:
        failed_gates.append("GATE_INDEX_UNIQUE")
    
    # GATE_AS_OF_CUTOFF
    if not max_index_le_as_of:
        failed_gates.append("GATE_AS_OF_CUTOFF")
    
    # GATE_COLUMNS_EXACT
    if not columns_match_expected:
        failed_gates.append("GATE_COLUMNS_EXACT")
    
    # GATE_FLOAT_DTYPES
    if not all_float_dtypes:
        failed_gates.append("GATE_FLOAT_DTYPES")
    
    # GATE_MISSINGNESS
    if n_rows > ignore_first_n_rows_for_missing:
        for col, ratio in missing_ratios_eval.items():
            if ratio > max_feature_missing_ratio:
                failed_gates.append("GATE_MISSINGNESS")
                break
    
    # =========================================================================
    # Sector representativeness gate (fail-safe: only applies if attrs present)
    # =========================================================================
    sector_counts_present = False
    sector_low_count_ratio: dict[str, float] = {}
    
    try:
        counts_df = frame.attrs.get("sector_counts")
        
        if isinstance(counts_df, pd.DataFrame):
            expected_count_cols = [f"S{i}_n_firms" for i in range(10)]
            
            # Validate structure: same index, 10 columns with correct names
            if (counts_df.index.equals(frame.index) and 
                list(counts_df.columns) == expected_count_cols):
                
                sector_counts_present = True
                
                # Apply after ignoring first N rows
                if n_rows > ignore_first_n_rows_for_missing:
                    eval_counts = counts_df.iloc[ignore_first_n_rows_for_missing:]
                    n_eval_rows = len(eval_counts)
                    
                    if n_eval_rows > 0:
                        gate_failed = False
                        for col in expected_count_cols:
                            low_count_mask = eval_counts[col] < min_sector_firms
                            low_ratio = low_count_mask.sum() / n_eval_rows
                            sector_low_count_ratio[col] = float(low_ratio)
                            
                            if low_ratio > max_low_count_month_ratio:
                                gate_failed = True
                        
                        if gate_failed:
                            failed_gates.append("GATE_SECTOR_COUNTS_MIN")
    except Exception:
        # Fail-safe: do not crash, just report counts as missing
        sector_counts_present = False
    
    passed = len(failed_gates) == 0
    
    report = {
        "passed": passed,
        "failed_gates": failed_gates,
        "metrics": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "start": start_date,
            "end": end_date,
            "max_index_le_as_of": max_index_le_as_of,
            "index_is_monotonic": index_is_monotonic,
            "index_is_unique": index_is_unique,
            "columns_match_expected": columns_match_expected,
            "all_float_dtypes": all_float_dtypes,
            "missing_ratios_full": missing_ratios_full,
            "missing_ratios_eval": missing_ratios_eval,
            "max_missing_ratio_eval": max_missing_ratio_eval,
        },
        "sector_counts_present": sector_counts_present,
    }
    
    if sector_counts_present:
        report["sector_low_count_ratio"] = sector_low_count_ratio
        report["sector_min_firms_threshold"] = min_sector_firms
        report["sector_max_low_ratio_threshold"] = max_low_count_month_ratio
    
    return report


def check_no_lookahead_invariance(
    builder: Callable[..., pd.DataFrame],
    *,
    builder_kwargs: dict,
    cutoff_dates: list[str],
) -> dict:
    """
    Check that earlier cutoffs are invariant to later data additions.
    
    Parameters
    ----------
    builder : Callable
        Function that builds a feature frame. Must accept as_of_date as kwarg.
    builder_kwargs : dict
        Keyword arguments to pass to builder (except as_of_date).
    cutoff_dates : list[str]
        List of cutoff dates in "YYYY-MM-DD" format.
    
    Returns
    -------
    dict
        Result with keys:
        - passed: bool
        - checked_pairs: list[tuple[str, str]]
        - failed_pair: tuple[str, str] | None
        - error: str | None
    """
    try:
        # Sort cutoff dates ascending
        sorted_dates = sorted(cutoff_dates)
        
        if len(sorted_dates) < 2:
            return {
                "passed": True,
                "checked_pairs": [],
                "failed_pair": None,
                "error": None,
            }
        
        # Build frames for each cutoff
        frames = {}
        for cutoff in sorted_dates:
            frames[cutoff] = builder(**builder_kwargs, as_of_date=cutoff)
        
        checked_pairs = []
        
        # Compare adjacent pairs
        for i in range(len(sorted_dates) - 1):
            early = sorted_dates[i]
            late = sorted_dates[i + 1]
            
            early_frame = frames[early]
            late_frame = frames[late]
            
            # Restrict late frame to early frame's index
            overlapping_index = early_frame.index
            late_restricted = late_frame.loc[late_frame.index.isin(overlapping_index)]
            
            # Both should have the same rows for comparison
            early_aligned = early_frame.loc[early_frame.index.isin(late_restricted.index)]
            
            try:
                pd.testing.assert_frame_equal(
                    early_aligned,
                    late_restricted,
                    check_names=True,
                )
                checked_pairs.append((early, late))
            except AssertionError as e:
                return {
                    "passed": False,
                    "checked_pairs": checked_pairs,
                    "failed_pair": (early, late),
                    "error": str(e),
                }
        
        return {
            "passed": True,
            "checked_pairs": checked_pairs,
            "failed_pair": None,
            "error": None,
        }
    
    except Exception as e:
        return {
            "passed": False,
            "checked_pairs": [],
            "failed_pair": None,
            "error": str(e),
        }


def assert_real_data_health_gates(report: dict) -> None:
    """
    Assert that health gates passed, raising ValueError if not.
    
    Parameters
    ----------
    report : dict
        Report from run_real_data_health_gates().
    
    Raises
    ------
    ValueError
        If report["passed"] is False.
    """
    if not report.get("passed", False):
        failed_gates = report.get("failed_gates", [])
        raise ValueError(
            f"Health gates failed: {', '.join(failed_gates) if failed_gates else 'unknown'}"
        )
