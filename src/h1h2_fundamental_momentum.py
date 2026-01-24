"""
H1/H2 Trailing Fundamental Momentum for QUANT-NEURAL.

Builds 20 monthly features from SEC fundamentals:
- 10 sectors × 2 horizons (H1=short 3M, H2=long 12M)
- Relative to market momentum (sector - market)

Point-in-Time (PIT) Rules:
- A value is usable only after its filed date.
- Latest-filed wins for same (cik, tag, unit, end).
- Forward-fill: if nothing new is filed, latest known value persists.
- No "now" or system time logic.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _slog(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Signed log transform for handling negative values.
    
    slog(x) = sign(x) * log(|x| + eps)
    """
    return np.sign(x) * np.log(np.abs(x) + eps)


def _select_tag_for_cik(
    facts: pd.DataFrame,
    cik: str,
    tag_priority: Tuple[str, ...],
    unit: str,
) -> str | None:
    """
    Select first available tag from priority list for a given CIK.
    
    Returns None if no tag from priority list has data.
    """
    cik_facts = facts[(facts["cik"] == cik) & (facts["unit"] == unit)]
    for tag in tag_priority:
        if (cik_facts["tag"] == tag).any():
            return tag
    return None


def _get_pit_visible_facts(
    facts: pd.DataFrame,
    month_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Get facts visible as-of month_end (filed <= month_end).
    Apply latest-filed-wins for same (cik, tag, unit, end).
    """
    # Filter by filed <= month_end
    visible = facts[facts["filed"] <= month_end].copy()
    
    if visible.empty:
        return visible
    
    # Latest-filed wins: sort by key + filed, keep last
    key_cols = ["cik", "tag", "unit", "end"]
    visible = visible.sort_values(by=key_cols + ["filed"], kind="mergesort")
    visible = visible.drop_duplicates(subset=key_cols, keep="last")
    
    return visible


def _compute_ttm_for_cik_at_month(
    visible_facts: pd.DataFrame,
    cik: str,
    tag: str,
    unit: str,
    month_end: pd.Timestamp,
) -> float:
    """
    Compute TTM (Trailing Twelve Months) for a single CIK at a given month.
    
    TTM = sum of last 4 quarterly values with end <= month_end.
    Returns NaN if fewer than 4 periods available.
    """
    cik_facts = visible_facts[
        (visible_facts["cik"] == cik) &
        (visible_facts["tag"] == tag) &
        (visible_facts["unit"] == unit) &
        (visible_facts["end"] <= month_end)
    ].copy()
    
    if cik_facts.empty:
        return np.nan
    
    # Sort by end ascending, take last 4 distinct end dates
    cik_facts = cik_facts.sort_values("end", kind="mergesort")
    unique_ends = cik_facts["end"].unique()
    
    if len(unique_ends) < 4:
        return np.nan
    
    # Take last 4 periods
    last_4_ends = unique_ends[-4:]
    ttm_rows = cik_facts[cik_facts["end"].isin(last_4_ends)]
    
    return ttm_rows["val"].sum()


def build_h1h2_relative_fundamental_momentum(
    facts: pd.DataFrame,
    *,
    month_ends: pd.DatetimeIndex,
    cik_to_sector: dict[str, int],
    n_sectors: int = 10,
    tag_priority: tuple[str, ...] = ("OperatingIncomeLoss", "NetIncomeLoss"),
    unit: str = "USD",
    method: str = "logdiff",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Build H1/H2 relative fundamental momentum features.
    
    Parameters
    ----------
    facts : pd.DataFrame
        Tidy table with columns: cik, tag, unit, end, filed, val
    month_ends : pd.DatetimeIndex
        Monthly end dates for feature construction.
    cik_to_sector : dict[str, int]
        Mapping from CIK string to sector index (0 to n_sectors-1).
    n_sectors : int
        Number of sectors (default 10).
    tag_priority : tuple[str, ...]
        Priority list of fundamental tags to use per CIK.
    unit : str
        Unit to filter (default "USD").
    method : str
        "logdiff" or "pct" for momentum calculation.
    eps : float
        Small value to avoid log(0) or division by zero.
    
    Returns
    -------
    pd.DataFrame
        Index: month_ends
        Columns: S0_H1, S1_H1, ..., S9_H1, S0_H2, S1_H2, ..., S9_H2 (20 total)
        All float dtype.
    
    Notes
    -----
    PIT Rules:
    - Only rows with filed <= month_end are visible at each month.
    - Latest-filed wins for same (cik, tag, unit, end).
    - TTM = sum of last 4 quarterly values (requires 4 periods).
    - H1 = 3-month momentum (relative to market)
    - H2 = 12-month momentum (relative to market)
    """
    if method not in ("logdiff", "pct"):
        raise ValueError(f"method must be 'logdiff' or 'pct', got '{method}'")
    
    # Enforce fixed 20-feature shape (n_sectors must be 10)
    if n_sectors != 10:
        raise ValueError(
            "n_sectors must be 10 to preserve the fixed 20-feature shape. "
            f"Got n_sectors={n_sectors}"
        )
    
    # Validate required columns
    required_cols = ["cik", "tag", "unit", "end", "filed", "val"]
    for col in required_cols:
        if col not in facts.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Get unique CIKs that are in our sector mapping
    all_ciks = facts["cik"].unique()
    valid_ciks = [c for c in all_ciks if c in cik_to_sector]
    
    # Select tag for each CIK
    cik_tags = {}
    for cik in valid_ciks:
        tag = _select_tag_for_cik(facts, cik, tag_priority, unit)
        if tag is not None:
            cik_tags[cik] = tag
    
    # Initialize sector totals per month
    # We need extra months before for lag calculation
    n_months = len(month_ends)
    
    # Build sector and market totals for each month_end (with lookback)
    # We need up to 12 months of history for H2
    all_months = pd.date_range(
        start=month_ends.min() - pd.DateOffset(months=12),
        end=month_ends.max(),
        freq="ME"
    )
    
    # Compute sector totals for all months
    sector_totals = np.full((len(all_months), n_sectors), np.nan)
    market_totals = np.full(len(all_months), np.nan)
    
    # Track per-month per-sector firm counts for diagnostics
    sector_counts_all = np.zeros((len(all_months), n_sectors), dtype=int)
    
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    
    for month_idx, month_end in enumerate(all_months):
        # Get PIT-visible facts
        visible = _get_pit_visible_facts(facts, month_end)
        
        if visible.empty:
            continue
        
        # Compute TTM per CIK
        cik_ttms = {}
        for cik, tag in cik_tags.items():
            ttm = _compute_ttm_for_cik_at_month(visible, cik, tag, unit, month_end)
            if not np.isnan(ttm):
                cik_ttms[cik] = ttm
        
        # Aggregate to sectors
        sector_sums = np.zeros(n_sectors)
        sector_counts = np.zeros(n_sectors, dtype=int)
        
        for cik, ttm in cik_ttms.items():
            sector_idx = cik_to_sector.get(cik)
            if sector_idx is not None and 0 <= sector_idx < n_sectors:
                sector_sums[sector_idx] += ttm
                sector_counts[sector_idx] += 1
        
        # Store sector counts for diagnostics
        sector_counts_all[month_idx, :] = sector_counts
        
        # Store sector totals (NaN if no companies in sector)
        for s in range(n_sectors):
            if sector_counts[s] > 0:
                sector_totals[month_idx, s] = sector_sums[s]
        
        # Market total = sum of all CIK TTMs
        if cik_ttms:
            market_totals[month_idx] = sum(cik_ttms.values())
    
    # Compute momentum for output months only
    h1_cols = [f"S{i}_H1" for i in range(n_sectors)]
    h2_cols = [f"S{i}_H2" for i in range(n_sectors)]
    output_cols = h1_cols + h2_cols
    
    result_data = np.full((n_months, n_sectors * 2), np.nan)
    
    for out_idx, month_end in enumerate(month_ends):
        if month_end not in month_to_idx:
            continue
        
        t_idx = month_to_idx[month_end]
        
        # H1: 3-month lookback
        t_minus_3 = month_end - pd.DateOffset(months=3)
        t_minus_3_idx = None
        for m, idx in month_to_idx.items():
            if m <= t_minus_3 and (t_minus_3_idx is None or m > all_months[t_minus_3_idx]):
                t_minus_3_idx = idx
        
        # H2: 12-month lookback
        t_minus_12 = month_end - pd.DateOffset(months=12)
        t_minus_12_idx = None
        for m, idx in month_to_idx.items():
            if m <= t_minus_12 and (t_minus_12_idx is None or m > all_months[t_minus_12_idx]):
                t_minus_12_idx = idx
        
        # Current values
        F_sector_t = sector_totals[t_idx, :]
        F_mkt_t = market_totals[t_idx]
        
        # Lagged values for H1
        if t_minus_3_idx is not None:
            F_sector_t3 = sector_totals[t_minus_3_idx, :]
            F_mkt_t3 = market_totals[t_minus_3_idx]
        else:
            F_sector_t3 = np.full(n_sectors, np.nan)
            F_mkt_t3 = np.nan
        
        # Lagged values for H2
        if t_minus_12_idx is not None:
            F_sector_t12 = sector_totals[t_minus_12_idx, :]
            F_mkt_t12 = market_totals[t_minus_12_idx]
        else:
            F_sector_t12 = np.full(n_sectors, np.nan)
            F_mkt_t12 = np.nan
        
        # Compute momentum
        if method == "logdiff":
            # H1: short momentum
            delta_short_sector = _slog(F_sector_t, eps) - _slog(F_sector_t3, eps)
            delta_short_mkt = _slog(F_mkt_t, eps) - _slog(F_mkt_t3, eps)
            
            # H2: long momentum
            delta_long_sector = _slog(F_sector_t, eps) - _slog(F_sector_t12, eps)
            delta_long_mkt = _slog(F_mkt_t, eps) - _slog(F_mkt_t12, eps)
        else:  # pct
            # H1: short momentum
            delta_short_sector = (F_sector_t - F_sector_t3) / (np.abs(F_sector_t3) + eps)
            delta_short_mkt = (F_mkt_t - F_mkt_t3) / (np.abs(F_mkt_t3) + eps)
            
            # H2: long momentum
            delta_long_sector = (F_sector_t - F_sector_t12) / (np.abs(F_sector_t12) + eps)
            delta_long_mkt = (F_mkt_t - F_mkt_t12) / (np.abs(F_mkt_t12) + eps)
        
        # Relative features
        X_H1 = delta_short_sector - delta_short_mkt
        X_H2 = delta_long_sector - delta_long_mkt
        
        result_data[out_idx, :n_sectors] = X_H1
        result_data[out_idx, n_sectors:] = X_H2
    
    result = pd.DataFrame(
        result_data,
        index=month_ends,
        columns=output_cols,
    )
    
    # Ensure float dtype
    result = result.astype(float)
    
    # Build sector counts DataFrame for output months (diagnostics via attrs)
    sector_counts_output = np.zeros((n_months, n_sectors), dtype=int)
    for out_idx, month_end in enumerate(month_ends):
        if month_end in month_to_idx:
            t_idx = month_to_idx[month_end]
            sector_counts_output[out_idx, :] = sector_counts_all[t_idx, :]
    
    counts_cols = [f"S{i}_n_firms" for i in range(n_sectors)]
    counts_df = pd.DataFrame(
        sector_counts_output,
        index=month_ends,
        columns=counts_cols,
    )
    
    # Attach via attrs (does not change feature columns)
    result.attrs["sector_counts"] = counts_df
    
    return result
