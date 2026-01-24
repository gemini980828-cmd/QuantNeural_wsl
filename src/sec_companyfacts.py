"""
SEC Companyfacts Loader for QUANT-NEURAL.

Provides:
- load_companyfacts_json: Load SEC companyfacts JSON with PIT filed-date cutoff.
- select_latest_filed: Select latest-filed row per unique key within PIT cutoff.

Point-in-Time (PIT) Rules:
- Only data with filed <= as_of_date is included.
- The "filed" date (not "end") determines availability.
- When multiple filings exist for the same period, the latest-filed wins.
- No "now" or system time logic.
- No network calls; local file read only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_cik(cik: str) -> str:
    """
    Normalize CIK to 10-digit zero-padded string.
    
    Parameters
    ----------
    cik : str
        CIK value (may have leading zeros or not).
    
    Returns
    -------
    str
        10-digit zero-padded CIK string.
    """
    # Keep digits only
    digits_only = "".join(c for c in str(cik) if c.isdigit())
    # Left-pad with zeros to length 10
    return digits_only.zfill(10)


def load_companyfacts_json(
    path: str,
    *,
    as_of_date: str,
    cik: str | None = None,
) -> pd.DataFrame:
    """
    Load SEC companyfacts JSON with PIT filed-date cutoff.
    
    Parameters
    ----------
    path : str
        Path to the companyfacts JSON file.
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format. Only rows with filed <= as_of_date
        are included.
    cik : str | None
        If provided, validate that JSON cik matches after normalization.
    
    Returns
    -------
    pd.DataFrame
        Columns: cik, taxonomy, tag, unit, end, filed, val (plus optional: fy, fp, form, frame, accn)
        Sorted by (taxonomy, tag, unit, end, filed) ascending.
    
    Raises
    ------
    ValueError
        If CIK mismatch, date parsing fails, val is non-finite, or empty after cutoff.
    """
    # Parse as_of_date
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    
    # Read JSON file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract CIK from JSON
    json_cik_raw = str(data.get("cik", ""))
    json_cik = _normalize_cik(json_cik_raw)
    
    # Validate CIK if provided
    if cik is not None:
        provided_cik = _normalize_cik(cik)
        if provided_cik != json_cik:
            raise ValueError(
                f"CIK mismatch: provided '{provided_cik}' != JSON '{json_cik}'"
            )
    
    # Flatten facts into rows
    rows = []
    facts_section = data.get("facts", {})
    
    for taxonomy, tags in facts_section.items():
        if not isinstance(tags, dict):
            continue
        for tag, tag_data in tags.items():
            if not isinstance(tag_data, dict):
                continue
            units_section = tag_data.get("units", {})
            if not isinstance(units_section, dict):
                continue
            for unit_name, entries in units_section.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    row = {
                        "cik": json_cik,
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "unit": unit_name,
                        "end": entry.get("end"),
                        "filed": entry.get("filed"),
                        "val": entry.get("val"),
                    }
                    # Optional columns
                    for opt_col in ["fy", "fp", "form", "frame", "accn"]:
                        if opt_col in entry:
                            row[opt_col] = entry[opt_col]
                    rows.append(row)
    
    if not rows:
        raise ValueError("No facts found in JSON file.")
    
    df = pd.DataFrame(rows)
    
    # Parse dates
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    
    # Validate no NaT in required date columns
    if df["end"].isna().any():
        raise ValueError("Some 'end' dates could not be parsed (NaT detected).")
    if df["filed"].isna().any():
        raise ValueError("Some 'filed' dates could not be parsed (NaT detected).")
    
    # Parse val
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype(float)
    
    # Validate no NaN in val
    if df["val"].isna().any():
        raise ValueError("Some 'val' values are NaN (non-numeric values detected).")
    
    # PIT filed gate: keep only rows with filed <= as_of_date
    df = df[df["filed"] <= as_of_dt].copy()
    
    if df.empty:
        raise ValueError(
            f"No data remaining after PIT filed cutoff (as_of_date={as_of_date})."
        )
    
    # Deterministic sorting
    sort_cols = ["taxonomy", "tag", "unit", "end", "filed"]
    df = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    
    return df


def select_latest_filed(
    facts: pd.DataFrame,
    *,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Select latest-filed row per unique key within PIT cutoff.
    
    This implements the PIT "latest-filed wins" rule: for each unique
    (taxonomy, tag, unit, end), keep the row with the maximum filed date
    that is still <= as_of_date.
    
    Parameters
    ----------
    facts : pd.DataFrame
        DataFrame with columns: taxonomy, tag, unit, end, filed, val, etc.
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format.
    
    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with one row per (taxonomy, tag, unit, end),
        sorted deterministically.
    
    Raises
    ------
    ValueError
        If empty after cutoff.
    """
    # Parse as_of_date
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    
    # Defensive PIT gate again
    df = facts[facts["filed"] <= as_of_dt].copy()
    
    if df.empty:
        raise ValueError(
            f"No data remaining after PIT filed cutoff (as_of_date={as_of_date})."
        )
    
    # Key columns
    key_cols = ["taxonomy", "tag", "unit", "end"]
    
    # Sort by key + filed ascending, then take last per key (= max filed)
    df = df.sort_values(by=key_cols + ["filed"], kind="mergesort")
    
    # Keep last occurrence per key (latest filed within cutoff)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    
    # Final deterministic sorting
    sort_cols = ["taxonomy", "tag", "unit", "end", "filed"]
    df = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    
    return df
