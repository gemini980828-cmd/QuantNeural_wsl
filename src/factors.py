"""
Factors module for QUANT-NEURAL.

Provides:
- winsorize_series: per-date cross-sectional clipping
- zscore_cross_section: cross-sectional z-score per date bucket
- build_style_factors: construct 5 style factors (Value, Growth, Momentum, Quality, Size)
- build_relative_earnings_momentum: 20-dim relative earnings features for MLP input
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WinsorizeParams:
    """Parameters for cross-sectional winsorization.
    
    Attributes
    ----------
    lower_q : float
        Lower quantile threshold (default 0.01 = 1%).
    upper_q : float
        Upper quantile threshold (default 0.99 = 99%).
    """
    lower_q: float = 0.01
    upper_q: float = 0.99


def winsorize_series(s: pd.Series, p: WinsorizeParams) -> pd.Series:
    """
    Winsorize a series by clipping at specified quantiles.
    
    This is cross-sectional clipping, NOT a train-fitted transformer.
    Computes quantiles from the input series and clips to those bounds.
    
    Parameters
    ----------
    s : pd.Series
        Input series to winsorize.
    p : WinsorizeParams
        Winsorization parameters.
    
    Returns
    -------
    pd.Series
        Winsorized series with same index as input.
    """
    lo = s.quantile(p.lower_q)
    hi = s.quantile(p.upper_q)
    return s.clip(lower=lo, upper=hi)


def zscore_cross_section(
    df: pd.DataFrame,
    cols: List[str],
    group_col: Optional[str],
    date_col: str
) -> pd.DataFrame:
    """
    Z-score columns cross-sectionally within date (and optionally group) buckets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : List[str]
        Columns to z-score.
    group_col : Optional[str]
        Optional grouping column (e.g., sector). If None, only group by date.
    date_col : str
        Date column name.
    
    Returns
    -------
    pd.DataFrame
        Copy of df with transformed columns.
    
    Notes
    -----
    - For each group, if std == 0 or not finite, returns zeros for that group.
    - Uses ddof=0 for population standard deviation.
    """
    df = df.copy()
    
    # Build groupby keys
    keys = [date_col]
    if group_col is not None:
        keys.append(group_col)
    
    def _zscore_series(s: pd.Series) -> pd.Series:
        """Z-score a single series within its group."""
        x = s.values.astype(np.float64)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        
        if sd == 0 or not np.isfinite(sd) or not np.isfinite(mu):
            return pd.Series(np.zeros(len(x)), index=s.index)
        else:
            return pd.Series((x - mu) / sd, index=s.index)
    
    # Apply z-score to each column within each group
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby(keys, group_keys=False)[col].transform(_zscore_series)
    
    return df


def build_style_factors(
    df: pd.DataFrame,
    *,
    date_col: str,
    wins: WinsorizeParams
) -> pd.DataFrame:
    """
    Build 5 style factors from raw input columns.
    
    Steps:
    1. Winsorize raw inputs per-date.
    2. Create primitives (inverse ratios, log size).
    3. Z-score cross-sectionally per date.
    4. Combine into style factors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with required columns:
        PER, PBR, DivYield, SalesGrowth_3Y, NetIncomeGrowth_3Y, EPSGrowth_Fwd,
        R_1M, R_3M, R_12M, ROE, OPM, DebtRatio, EarningsVol, MarketCap
    date_col : str
        Date column name.
    wins : WinsorizeParams
        Winsorization parameters.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: Value, Growth, Momentum, Quality, Size.
        Intermediate columns may also be present.
    
    Notes
    -----
    - This module is cross-sectional per date. It does NOT use future dates.
    - Winsorization is per-date clipping (not train-fitted).
    """
    df = df.copy()
    
    # Raw input columns to winsorize
    raw_cols = [
        "PER", "PBR", "DivYield",
        "SalesGrowth_3Y", "NetIncomeGrowth_3Y", "EPSGrowth_Fwd",
        "R_1M", "R_3M", "R_12M",
        "ROE", "OPM", "DebtRatio", "EarningsVol",
        "MarketCap"
    ]
    
    # Step 1: Winsorize per-date
    for col in raw_cols:
        if col in df.columns:
            df[col] = df.groupby(date_col)[col].transform(
                lambda s: winsorize_series(s, wins)
            )
    
    # Step 2: Create primitives with inverse transformations
    # Replace 0 with NaN before inversion to avoid inf
    def safe_inv(s: pd.Series) -> pd.Series:
        """Compute 1/x, replacing 0 with NaN."""
        s = s.replace(0, np.nan)
        return 1.0 / s
    
    df["inv_PER"] = safe_inv(df["PER"])
    df["inv_PBR"] = safe_inv(df["PBR"])
    df["inv_Debt"] = safe_inv(df["DebtRatio"])
    df["inv_EarnVol"] = safe_inv(df["EarningsVol"])
    
    # Size: -log(MarketCap) so smaller = higher score
    # Replace 0 with NaN before log
    mcap = df["MarketCap"].replace(0, np.nan)
    df["inv_Size"] = -np.log(mcap)
    
    # Step 3: Z-score cross-sectionally per date
    z_cols = [
        "inv_PER", "inv_PBR", "DivYield",
        "SalesGrowth_3Y", "NetIncomeGrowth_3Y", "EPSGrowth_Fwd",
        "R_1M", "R_3M", "R_12M",
        "ROE", "OPM", "inv_Debt", "inv_EarnVol",
        "inv_Size"
    ]
    
    df = zscore_cross_section(df, cols=z_cols, group_col=None, date_col=date_col)
    
    # Step 4: Combine into style factors (mean of components)
    # Handle NaN gracefully with np.nanmean
    df["Value"] = df[["inv_PER", "inv_PBR", "DivYield"]].mean(axis=1, skipna=True)
    df["Growth"] = df[["SalesGrowth_3Y", "NetIncomeGrowth_3Y", "EPSGrowth_Fwd"]].mean(axis=1, skipna=True)
    df["Momentum"] = df[["R_1M", "R_3M", "R_12M"]].mean(axis=1, skipna=True)
    df["Quality"] = df[["ROE", "OPM", "inv_Debt", "inv_EarnVol"]].mean(axis=1, skipna=True)
    df["Size"] = df["inv_Size"]
    
    return df


def build_relative_earnings_momentum(
    sector_op_fy1: pd.DataFrame,
    sector_op_fy2: pd.DataFrame,
    market_op_fy1: pd.Series,
    market_op_fy2: pd.Series,
    method: str = "logdiff"
) -> pd.DataFrame:
    """
    Build 20-dim relative earnings momentum features (MLP inputs).
    
    For each sector s:
      rel_fy1[s] = delta(sector_op_fy1[s]) - delta(market_op_fy1)
      rel_fy2[s] = delta(sector_op_fy2[s]) - delta(market_op_fy2)
    
    Parameters
    ----------
    sector_op_fy1 : pd.DataFrame
        Sector operating profit FY1, index=date, columns=sectors.
    sector_op_fy2 : pd.DataFrame
        Sector operating profit FY2, index=date, columns=sectors.
    market_op_fy1 : pd.Series
        Market operating profit FY1, index=date.
    market_op_fy2 : pd.Series
        Market operating profit FY2, index=date.
    method : str
        "logdiff" (default): log(x).diff()
        "pct": x.pct_change()
    
    Returns
    -------
    pd.DataFrame
        Shape (T, 20) with columns:
        [sector_FY1 for each sector] + [sector_FY2 for each sector]
    
    Raises
    ------
    ValueError
        If method is not in {"logdiff", "pct"}.
        If sector_op_fy1 and sector_op_fy2 have different columns.
        If indices are mismatched or not unique.
    """
    # Validate method
    if method not in {"logdiff", "pct"}:
        raise ValueError(f"method must be 'logdiff' or 'pct', got '{method}'")
    
    # Validate sector columns match
    if list(sector_op_fy1.columns) != list(sector_op_fy2.columns):
        raise ValueError(
            f"sector_op_fy1 columns {list(sector_op_fy1.columns)} "
            f"!= sector_op_fy2 columns {list(sector_op_fy2.columns)}"
        )
    
    sectors = list(sector_op_fy1.columns)
    
    # Sort all by index
    sector_op_fy1 = sector_op_fy1.sort_index()
    sector_op_fy2 = sector_op_fy2.sort_index()
    market_op_fy1 = market_op_fy1.sort_index()
    market_op_fy2 = market_op_fy2.sort_index()
    
    # Validate indices match
    if not sector_op_fy1.index.equals(sector_op_fy2.index):
        raise ValueError("sector_op_fy1 and sector_op_fy2 must have identical indices")
    if not sector_op_fy1.index.equals(market_op_fy1.index):
        raise ValueError("sector_op_fy1 and market_op_fy1 must have identical indices")
    if not sector_op_fy1.index.equals(market_op_fy2.index):
        raise ValueError("sector_op_fy1 and market_op_fy2 must have identical indices")
    
    # Validate unique index
    if not sector_op_fy1.index.is_unique:
        raise ValueError("Index must be unique")
    
    index = sector_op_fy1.index
    
    # Define delta function based on method
    def compute_delta(x: pd.Series) -> pd.Series:
        """Compute delta using specified method."""
        x = x.astype(float)
        if method == "logdiff":
            # log(x), replacing non-positive with NaN
            x_log = np.log(x.where(x > 0))
            return x_log.diff()
        else:  # pct
            return x.pct_change()
    
    # Compute market deltas
    delta_market_fy1 = compute_delta(market_op_fy1)
    delta_market_fy2 = compute_delta(market_op_fy2)
    
    # Build result columns
    result_data = {}
    
    # FY1 columns: sector_FY1
    for sector in sectors:
        delta_sector = compute_delta(sector_op_fy1[sector])
        rel = delta_sector - delta_market_fy1
        result_data[f"{sector}_FY1"] = rel.values
    
    # FY2 columns: sector_FY2
    for sector in sectors:
        delta_sector = compute_delta(sector_op_fy2[sector])
        rel = delta_sector - delta_market_fy2
        result_data[f"{sector}_FY2"] = rel.values
    
    # Build DataFrame with correct column order
    columns = [f"{s}_FY1" for s in sectors] + [f"{s}_FY2" for s in sectors]
    result = pd.DataFrame(result_data, index=index, columns=columns)
    
    return result
