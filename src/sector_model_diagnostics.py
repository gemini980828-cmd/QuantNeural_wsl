"""
Sector Model Diagnostics for QUANT-NEURAL.

Provides analysis tools to diagnose WHY sector-broadcast MLP underperforms:
- Is the model's sector ranking wrong (predictive failure)?
- Or is the portfolio result dominated by tie-breaking / selection mechanics?

This is analysis-only: no portfolio construction changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


def load_sector_to_tickers_json(path: str) -> dict[str, list[str]]:
    """
    Load sector_to_tickers mapping from JSON file.
    
    Parameters
    ----------
    path : str
        Path to JSON file with structure {"S0": [...], "S1": [...], ...}.
    
    Returns
    -------
    dict[str, list[str]]
        Mapping with sorted sector keys and sorted tickers within each sector.
    
    Raises
    ------
    ValueError
        If file cannot be read or JSON is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load sector_to_tickers JSON: {e}") from e
    
    if not isinstance(data, dict):
        raise ValueError("JSON root must be a dict")
    
    # Sort sector keys and tickers within each sector
    result = {}
    for sector_key in sorted(data.keys()):
        tickers = data[sector_key]
        if not isinstance(tickers, list):
            raise ValueError(f"Value for sector '{sector_key}' must be a list")
        result[sector_key] = sorted(str(t) for t in tickers)
    
    return result


def compute_sector_realized_next_returns(
    prices_csv_path: str,
    sector_to_tickers: dict[str, list[str]],
    *,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Compute per-sector next-period realized returns at each date.
    
    Parameters
    ----------
    prices_csv_path : str
        Path to prices CSV with 'date' column and ticker columns.
    sector_to_tickers : dict[str, list[str]]
        Mapping of sector_id -> list of tickers.
    dates : list[pd.Timestamp]
        List of dates to compute returns for.
    
    Returns
    -------
    pd.DataFrame
        Index = dates (aligned), columns = sorted sector keys,
        values = realized next-period returns (NaN if insufficient data).
    """
    # Load prices
    prices = pd.read_csv(prices_csv_path)
    date_col = "date" if "date" in prices.columns else prices.columns[0]
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices = prices.sort_values(date_col).set_index(date_col)
    
    # Get all tickers from mapping
    all_tickers = set()
    for tickers in sector_to_tickers.values():
        all_tickers.update(tickers)
    
    # Filter to available tickers
    available_tickers = [t for t in all_tickers if t in prices.columns]
    prices = prices[available_tickers]
    
    # Sort dates
    dates_sorted = sorted(dates)
    sector_keys = sorted(sector_to_tickers.keys())
    
    # Build output
    result = pd.DataFrame(index=dates_sorted, columns=sector_keys, dtype=float)
    
    for i, d in enumerate(dates_sorted):
        if d not in prices.index:
            continue
        
        # Find next date
        next_dates = [nd for nd in prices.index if nd > d]
        if not next_dates:
            continue
        next_d = min(next_dates)
        
        for sector_key in sector_keys:
            sector_tickers = sector_to_tickers.get(sector_key, [])
            valid_tickers = [t for t in sector_tickers if t in prices.columns]
            
            if not valid_tickers:
                result.loc[d, sector_key] = np.nan
                continue
            
            # Get prices at d and next_d
            prices_d = prices.loc[d, valid_tickers]
            prices_next = prices.loc[next_d, valid_tickers]
            
            # Compute returns for tickers with both prices valid
            returns = []
            for t in valid_tickers:
                p_d = prices_d[t] if isinstance(prices_d, pd.Series) else prices_d
                p_next = prices_next[t] if isinstance(prices_next, pd.Series) else prices_next
                if pd.notna(p_d) and pd.notna(p_next) and p_d > 0:
                    returns.append((p_next / p_d) - 1)
            
            if returns:
                result.loc[d, sector_key] = np.mean(returns)
            else:
                result.loc[d, sector_key] = np.nan
    
    return result.astype(float)


def compute_sector_ic_timeseries(
    sector_scores: pd.DataFrame,
    sector_realized: pd.DataFrame,
    *,
    method: str = "spearman",
) -> pd.Series:
    """
    Compute cross-sectional IC between predicted sector scores and realized returns.
    
    Parameters
    ----------
    sector_scores : pd.DataFrame
        Index = dates, columns = sector_ids, values = predicted scores.
    sector_realized : pd.DataFrame
        Index = dates, columns = sector_ids, values = realized returns.
    method : str
        Correlation method: "spearman" or "pearson".
    
    Returns
    -------
    pd.Series
        IC value for each date.
    
    Raises
    ------
    ValueError
        If method is not "spearman" or "pearson".
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"method must be 'spearman' or 'pearson', got '{method}'")
    
    # Align dates and columns
    common_dates = sector_scores.index.intersection(sector_realized.index)
    common_cols = [c for c in sector_scores.columns if c in sector_realized.columns]
    
    scores = sector_scores.loc[common_dates, common_cols]
    realized = sector_realized.loc[common_dates, common_cols]
    
    ic_values = []
    for d in common_dates:
        s = scores.loc[d].values.astype(float)
        r = realized.loc[d].values.astype(float)
        
        # Filter out NaNs
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < 2:
            ic_values.append(np.nan)
            continue
        
        s_valid = s[mask]
        r_valid = r[mask]
        
        if method == "spearman":
            corr, _ = stats.spearmanr(s_valid, r_valid)
        else:
            corr, _ = stats.pearsonr(s_valid, r_valid)
        
        ic_values.append(corr)
    
    return pd.Series(ic_values, index=common_dates, name=f"IC_{method}")


def compute_sector_hit_rate(
    sector_scores: pd.DataFrame,
    sector_realized: pd.DataFrame,
    *,
    top_n: int = 1,
) -> float:
    """
    Compute hit rate: fraction of dates where top_n predicted sectors match top_n realized.
    
    Parameters
    ----------
    sector_scores : pd.DataFrame
        Index = dates, columns = sector_ids, values = predicted scores.
    sector_realized : pd.DataFrame
        Index = dates, columns = sector_ids, values = realized returns.
    top_n : int
        Number of top sectors to compare.
    
    Returns
    -------
    float
        Average hit rate in [0, 1].
    """
    common_dates = sector_scores.index.intersection(sector_realized.index)
    common_cols = [c for c in sector_scores.columns if c in sector_realized.columns]
    
    if len(common_cols) < top_n:
        return 0.0
    
    scores = sector_scores.loc[common_dates, common_cols]
    realized = sector_realized.loc[common_dates, common_cols]
    
    hits = 0
    valid_dates = 0
    
    for d in common_dates:
        s = scores.loc[d]
        r = realized.loc[d]
        
        # Filter out NaNs
        valid_cols = [c for c in common_cols if pd.notna(s[c]) and pd.notna(r[c])]
        if len(valid_cols) < top_n:
            continue
        
        top_by_score = set(s[valid_cols].nlargest(top_n).index)
        top_by_real = set(r[valid_cols].nlargest(top_n).index)
        
        if top_by_score == top_by_real:
            hits += 1
        valid_dates += 1
    
    return hits / valid_dates if valid_dates > 0 else 0.0


def compute_tie_stats(scores_panel: pd.DataFrame) -> dict:
    """
    Compute tie statistics for a ticker-level broadcast panel.
    
    Parameters
    ----------
    scores_panel : pd.DataFrame
        Index = dates, columns = tickers, values = scores.
    
    Returns
    -------
    dict
        Aggregated tie statistics:
        - n_unique_mean, n_unique_median, n_unique_max
        - max_tie_group_mean, max_tie_group_median, max_tie_group_max
        - max_tie_frac_mean, max_tie_frac_median, max_tie_frac_max
    """
    n_unique_list = []
    max_tie_group_list = []
    max_tie_frac_list = []
    
    for d in scores_panel.index:
        row = scores_panel.loc[d].dropna()
        if len(row) == 0:
            continue
        
        value_counts = row.value_counts()
        n_unique = len(value_counts)
        max_tie_group = value_counts.max()
        max_tie_frac = max_tie_group / len(row)
        
        n_unique_list.append(n_unique)
        max_tie_group_list.append(max_tie_group)
        max_tie_frac_list.append(max_tie_frac)
    
    if not n_unique_list:
        return {
            "n_unique_mean": 0.0,
            "n_unique_median": 0.0,
            "n_unique_max": 0,
            "max_tie_group_mean": 0.0,
            "max_tie_group_median": 0.0,
            "max_tie_group_max": 0,
            "max_tie_frac_mean": 0.0,
            "max_tie_frac_median": 0.0,
            "max_tie_frac_max": 0.0,
        }
    
    return {
        "n_unique_mean": float(np.mean(n_unique_list)),
        "n_unique_median": float(np.median(n_unique_list)),
        "n_unique_max": int(np.max(n_unique_list)),
        "max_tie_group_mean": float(np.mean(max_tie_group_list)),
        "max_tie_group_median": float(np.median(max_tie_group_list)),
        "max_tie_group_max": int(np.max(max_tie_group_list)),
        "max_tie_frac_mean": float(np.mean(max_tie_frac_list)),
        "max_tie_frac_median": float(np.median(max_tie_frac_list)),
        "max_tie_frac_max": float(np.max(max_tie_frac_list)),
    }


def run_sector_autopsy(
    prices_csv_path: str,
    sector_to_tickers_json_path: str,
    baseline_sector_broadcast_scores_csv_path: str,
    mlp_sector_broadcast_scores_csv_path: str,
    output_dir: str,
) -> dict:
    """
    Run full sector autopsy analysis.
    
    Parameters
    ----------
    prices_csv_path : str
        Path to prices CSV.
    sector_to_tickers_json_path : str
        Path to sector_to_tickers JSON.
    baseline_sector_broadcast_scores_csv_path : str
        Path to baseline sector-broadcast scores CSV.
    mlp_sector_broadcast_scores_csv_path : str
        Path to MLP sector-broadcast scores CSV.
    output_dir : str
        Directory to write output artifacts.
    
    Returns
    -------
    dict
        Summary with IC, hit_rate, and tie_stats for both models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load inputs
    sector_to_tickers = load_sector_to_tickers_json(sector_to_tickers_json_path)
    
    def load_scores(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        date_col = "date" if "date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        return df
    
    baseline_scores = load_scores(baseline_sector_broadcast_scores_csv_path)
    mlp_scores = load_scores(mlp_sector_broadcast_scores_csv_path)
    
    # Intersect dates and tickers
    common_dates = baseline_scores.index.intersection(mlp_scores.index)
    common_tickers = [c for c in baseline_scores.columns if c in mlp_scores.columns]
    
    baseline_scores = baseline_scores.loc[common_dates, common_tickers]
    mlp_scores = mlp_scores.loc[common_dates, common_tickers]
    
    # Build per-sector scores (take mean across tickers in sector for each date)
    def aggregate_to_sector(panel: pd.DataFrame, s2t: dict) -> pd.DataFrame:
        sector_keys = sorted(s2t.keys())
        result = pd.DataFrame(index=panel.index, columns=sector_keys, dtype=float)
        for sector_key in sector_keys:
            sector_tickers = [t for t in s2t.get(sector_key, []) if t in panel.columns]
            if sector_tickers:
                result[sector_key] = panel[sector_tickers].mean(axis=1)
            else:
                result[sector_key] = np.nan
        return result
    
    baseline_sector = aggregate_to_sector(baseline_scores, sector_to_tickers)
    mlp_sector = aggregate_to_sector(mlp_scores, sector_to_tickers)
    
    # Compute realized returns
    sector_realized = compute_sector_realized_next_returns(
        prices_csv_path,
        sector_to_tickers,
        dates=list(common_dates),
    )
    
    # IC timeseries
    baseline_ic_spearman = compute_sector_ic_timeseries(baseline_sector, sector_realized, method="spearman")
    baseline_ic_pearson = compute_sector_ic_timeseries(baseline_sector, sector_realized, method="pearson")
    mlp_ic_spearman = compute_sector_ic_timeseries(mlp_sector, sector_realized, method="spearman")
    mlp_ic_pearson = compute_sector_ic_timeseries(mlp_sector, sector_realized, method="pearson")
    
    # Hit rates
    baseline_hit_1 = compute_sector_hit_rate(baseline_sector, sector_realized, top_n=1)
    baseline_hit_2 = compute_sector_hit_rate(baseline_sector, sector_realized, top_n=2)
    mlp_hit_1 = compute_sector_hit_rate(mlp_sector, sector_realized, top_n=1)
    mlp_hit_2 = compute_sector_hit_rate(mlp_sector, sector_realized, top_n=2)
    
    # Tie stats
    baseline_tie = compute_tie_stats(baseline_scores)
    mlp_tie = compute_tie_stats(mlp_scores)
    
    # Build summary
    summary = {
        "baseline": {
            "ic_spearman_mean": float(np.nanmean(baseline_ic_spearman)),
            "ic_spearman_std": float(np.nanstd(baseline_ic_spearman)),
            "ic_pearson_mean": float(np.nanmean(baseline_ic_pearson)),
            "ic_pearson_std": float(np.nanstd(baseline_ic_pearson)),
            "hit_rate_top1": baseline_hit_1,
            "hit_rate_top2": baseline_hit_2,
            "tie_stats": baseline_tie,
        },
        "mlp": {
            "ic_spearman_mean": float(np.nanmean(mlp_ic_spearman)),
            "ic_spearman_std": float(np.nanstd(mlp_ic_spearman)),
            "ic_pearson_mean": float(np.nanmean(mlp_ic_pearson)),
            "ic_pearson_std": float(np.nanstd(mlp_ic_pearson)),
            "hit_rate_top1": mlp_hit_1,
            "hit_rate_top2": mlp_hit_2,
            "tie_stats": mlp_tie,
        },
        "meta": {
            "n_dates": len(common_dates),
            "n_tickers": len(common_tickers),
            "n_sectors": len(sector_to_tickers),
        },
    }
    
    # Write JSON (sorted keys for determinism)
    output_path = Path(output_dir) / "sector_autopsy_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    return summary
