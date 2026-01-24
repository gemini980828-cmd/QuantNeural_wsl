# -*- coding: utf-8 -*-
"""
Metrics Utilities for OFF10 Robustness Report
==============================================

Shared utilities for column mapping, CAGR/MDD calculation,
and markdown table generation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================================
# COLUMN MAPPING - Handle various column name conventions
# ============================================================
CAGR_CANDIDATES = ["CAGR", "cagr", "cagr_after_tax", "cagr_net", "CAGR_Net"]
MDD_CANDIDATES = ["MDD", "mdd", "max_drawdown", "MaxDrawdown"]
FINAL_CANDIDATES = ["Final", "final", "FinalValue", "final_value", "FinalMultiple"]
EXPERIMENT_CANDIDATES = ["Experiment", "experiment", "Name", "name", "Strategy", "strategy"]
PERIOD_CANDIDATES = ["period", "Period", "Subperiod", "subperiod"]


def find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """Find column from candidate list"""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Could not find column from candidates: {candidates}")
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to canonical form"""
    df = df.copy()
    
    # Map to standard names
    mappings = {
        find_column(df, EXPERIMENT_CANDIDATES, required=False): "Experiment",
        find_column(df, CAGR_CANDIDATES, required=False): "CAGR",
        find_column(df, MDD_CANDIDATES, required=False): "MDD",
        find_column(df, FINAL_CANDIDATES, required=False): "Final",
        find_column(df, PERIOD_CANDIDATES, required=False): "Period",
    }
    
    # Apply valid mappings
    rename_map = {k: v for k, v in mappings.items() if k is not None and k != v}
    df = df.rename(columns=rename_map)
    
    return df


# ============================================================
# METRICS CALCULATION
# ============================================================
def calculate_delta(df: pd.DataFrame, baseline_name: str, metric_col: str) -> pd.Series:
    """Calculate delta vs baseline for a metric"""
    baseline_row = df[df["Experiment"].str.contains(baseline_name, case=False, na=False)]
    if len(baseline_row) == 0:
        raise ValueError(f"Baseline '{baseline_name}' not found in data")
    
    baseline_value = baseline_row[metric_col].iloc[0]
    return df[metric_col] - baseline_value


def calculate_cagr_from_returns(returns: pd.Series, trading_days: int = 252) -> float:
    """Calculate CAGR from daily returns"""
    equity = (1 + returns).cumprod()
    n_years = len(returns) / trading_days
    if n_years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return equity.iloc[-1] ** (1.0 / n_years) - 1.0


def calculate_mdd_from_equity(equity: pd.Series) -> float:
    """Calculate MDD from equity curve"""
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


# ============================================================
# MARKDOWN TABLE GENERATION
# ============================================================
def df_to_markdown(df: pd.DataFrame, float_fmt: Dict[str, str] = None) -> str:
    """Convert DataFrame to markdown table"""
    if float_fmt is None:
        float_fmt = {}
    
    # Format columns
    df_fmt = df.copy()
    for col, fmt in float_fmt.items():
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "N/A")
    
    # Build markdown
    headers = "| " + " | ".join(df_fmt.columns) + " |"
    separator = "|" + "|".join(["---"] * len(df_fmt.columns)) + "|"
    
    rows = []
    for _, row in df_fmt.iterrows():
        row_str = "| " + " | ".join(str(v) for v in row.values) + " |"
        rows.append(row_str)
    
    return "\n".join([headers, separator] + rows)


def format_pct(x: float) -> str:
    """Format as percentage"""
    return f"{x*100:.2f}%"


def format_pct_delta(x: float) -> str:
    """Format as percentage point delta"""
    return f"{x*100:+.2f}%p"


def format_mult(x: float) -> str:
    """Format as multiple"""
    return f"{x:.2f}x"


# ============================================================
# DECISION LOGIC
# ============================================================
def check_full_period_pass(df: pd.DataFrame, baseline: str, candidate: str) -> Tuple[bool, str]:
    """Check if candidate has ΔCAGR >= 0 vs baseline in full period"""
    df = standardize_columns(df)
    
    base_row = df[df["Experiment"].str.contains(baseline, case=False, na=False)]
    cand_row = df[df["Experiment"].str.contains(candidate, case=False, na=False)]
    
    if len(base_row) == 0 or len(cand_row) == 0:
        return False, "Missing experiment data"
    
    base_cagr = base_row["CAGR"].iloc[0]
    cand_cagr = cand_row["CAGR"].iloc[0]
    delta = cand_cagr - base_cagr
    
    passed = delta >= 0
    msg = f"ΔCAGR = {delta*100:+.2f}%p {'≥' if passed else '<'} 0"
    return passed, msg


def check_subperiod_pass(df: pd.DataFrame, baseline: str, candidate: str, 
                          min_pass: int = 2) -> Tuple[bool, str, Dict]:
    """Check if candidate passes in at least min_pass subperiods"""
    df = standardize_columns(df)
    
    base_rows = df[df["Experiment"].str.contains(baseline, case=False, na=False)]
    cand_rows = df[df["Experiment"].str.contains(candidate, case=False, na=False)]
    
    if len(base_rows) == 0 or len(cand_rows) == 0:
        return False, "Missing experiment data", {}
    
    # Merge on period
    periods = base_rows["Period"].unique() if "Period" in df.columns else []
    
    results = {}
    passes = 0
    
    for period in periods:
        base_cagr = base_rows[base_rows["Period"] == period]["CAGR"].iloc[0]
        cand_cagr = cand_rows[cand_rows["Period"] == period]["CAGR"].iloc[0]
        delta = cand_cagr - base_cagr
        passed = delta >= 0
        results[period] = {"delta": delta, "passed": passed}
        if passed:
            passes += 1
    
    overall_pass = passes >= min_pass
    msg = f"{passes}/{len(periods)} subperiods with ΔCAGR ≥ 0 (need {min_pass})"
    
    return overall_pass, msg, results


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================
def block_bootstrap_cagr(returns1: pd.Series, returns2: pd.Series,
                         block_len: int = 10, n_samples: int = 5000,
                         seed: int = 42, trading_days: int = 252) -> Dict:
    """
    Block bootstrap to estimate ΔCAGR distribution
    
    Returns dict with: samples, p_positive, median, q5, q95
    """
    np.random.seed(seed)
    
    # Align indices
    common_idx = returns1.index.intersection(returns2.index)
    r1 = returns1.loc[common_idx].values
    r2 = returns2.loc[common_idx].values
    
    n = len(r1)
    n_blocks = n // block_len
    
    delta_cagrs = []
    
    for _ in range(n_samples):
        # Sample blocks with replacement
        block_starts = np.random.randint(0, n - block_len + 1, size=n_blocks)
        
        # Build bootstrap sample
        sample_r1 = []
        sample_r2 = []
        for start in block_starts:
            sample_r1.extend(r1[start:start+block_len])
            sample_r2.extend(r2[start:start+block_len])
        
        # Calculate CAGR
        sample_r1 = np.array(sample_r1)
        sample_r2 = np.array(sample_r2)
        
        eq1 = np.cumprod(1 + sample_r1)
        eq2 = np.cumprod(1 + sample_r2)
        
        n_years = len(sample_r1) / trading_days
        if n_years > 0 and eq1[-1] > 0 and eq2[-1] > 0:
            cagr1 = eq1[-1] ** (1/n_years) - 1
            cagr2 = eq2[-1] ** (1/n_years) - 1
            delta_cagrs.append(cagr2 - cagr1)
    
    delta_cagrs = np.array(delta_cagrs)
    
    return {
        "samples": delta_cagrs,
        "p_positive": (delta_cagrs > 0).mean(),
        "median": np.median(delta_cagrs),
        "q5": np.percentile(delta_cagrs, 5),
        "q95": np.percentile(delta_cagrs, 95),
        "mean": np.mean(delta_cagrs),
        "std": np.std(delta_cagrs),
    }
