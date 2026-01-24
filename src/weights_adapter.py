"""
Predictions to Target Weights Adapter.

Converts model scores/predictions into long-only target portfolio weights
suitable for backtest_harness.run_backtest().

Design Principles:
- No external dependencies beyond numpy/pandas (standard project deps)
- Deterministic: no randomness, no system clock, no network
- Fail-fast: raise clear exceptions on invalid inputs
"""

from typing import Optional

import numpy as np
import pandas as pd


def _quantile_1d(x: np.ndarray, q: float) -> float:
    """
    Compute quantile with linear interpolation (deterministic, NumPy-version agnostic).

    Parameters
    ----------
    x : np.ndarray
        1D array of finite values.
    q : float
        Quantile in [0, 1].
    """
    if not isinstance(q, (int, float)) or not np.isfinite(q):
        raise ValueError(f"q must be a finite float in [0,1], got {q}")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q}")

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty for quantile computation")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite for quantile computation")

    x_sorted = np.sort(x, kind="mergesort")
    n = x_sorted.size

    pos = float(q) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))

    if lo == hi:
        return float(x_sorted[lo])

    frac = pos - lo
    return float((1.0 - frac) * x_sorted[lo] + frac * x_sorted[hi])


def _winsorize_row(row: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """
    Winsorize a 1D row by clipping to [q_low, q_high] quantiles.

    Notes
    -----
    - Deterministic given fixed inputs.
    - Requires finite row values (validated upstream).
    """
    if not isinstance(q_low, (int, float)) or not np.isfinite(q_low):
        raise ValueError(f"winsorize_q_low must be a finite float, got {q_low}")
    if not isinstance(q_high, (int, float)) or not np.isfinite(q_high):
        raise ValueError(f"winsorize_q_high must be a finite float, got {q_high}")

    q_low_f = float(q_low)
    q_high_f = float(q_high)

    if not (0.0 <= q_low_f < q_high_f <= 1.0):
        raise ValueError(
            f"winsorize_q_low/winsorize_q_high must satisfy 0 <= low < high <= 1, "
            f"got low={q_low_f}, high={q_high_f}"
        )

    lo = _quantile_1d(row, q_low_f)
    hi = _quantile_1d(row, q_high_f)
    return np.clip(np.asarray(row, dtype=np.float64), lo, hi)


def _zscore_row(row: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Z-score a 1D row cross-sectionally: (x - mean) / std.

    If std is too small or non-finite, returns zeros.
    """
    if not isinstance(eps, (int, float)) or not np.isfinite(eps) or float(eps) <= 0:
        raise ValueError(f"eps must be a positive finite float, got {eps}")

    x = np.asarray(row, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))

    if (not np.isfinite(mu)) or (not np.isfinite(sd)) or (sd <= float(eps)):
        return np.zeros_like(x, dtype=np.float64)

    return (x - mu) / sd


def _transform_softmax_scores_row(
    row: np.ndarray,
    *,
    score_transform: str,
    winsorize_q_low: float,
    winsorize_q_high: float,
    zscore_eps: float,
) -> np.ndarray:
    """
    Transform scores row before softmax to reduce outlier-driven concentration.

    Supported transforms:
    - "none"
    - "winsorize"
    - "zscore"
    - "winsorize_zscore"
    """
    if not isinstance(score_transform, str):
        raise ValueError(f"score_transform must be a str, got {type(score_transform).__name__}")

    t = score_transform.strip().lower()
    if t not in {"none", "winsorize", "zscore", "winsorize_zscore"}:
        raise ValueError(
            "score_transform must be one of "
            "['none','winsorize','zscore','winsorize_zscore'], "
            f"got '{score_transform}'"
        )

    out = np.asarray(row, dtype=np.float64)

    if t in {"winsorize", "winsorize_zscore"}:
        out = _winsorize_row(out, winsorize_q_low, winsorize_q_high)

    if t in {"zscore", "winsorize_zscore"}:
        out = _zscore_row(out, eps=zscore_eps)

    return out


def _validate_scores(scores: pd.DataFrame) -> None:
    """
    Validate scores DataFrame.
    
    Raises
    ------
    ValueError
        If scores fail validation.
    """
    if not isinstance(scores, pd.DataFrame):
        raise ValueError("scores must be a DataFrame")
    
    n, k = scores.shape
    
    if k < 2:
        raise ValueError(f"scores must have k_assets >= 2, got {k}")
    
    # Check monotonic increasing index
    if not scores.index.is_monotonic_increasing:
        raise ValueError("scores.index must be monotonic increasing")
    
    # Check unique index
    if not scores.index.is_unique:
        raise ValueError("scores.index must have unique values (no duplicates)")
    
    # Check unique columns
    if not scores.columns.is_unique:
        raise ValueError("scores.columns must be unique")
    
    # Check finite values
    if not np.all(np.isfinite(scores.values)):
        raise ValueError("scores must be finite (no NaN/inf)")


def _apply_softmax(row: np.ndarray, temperature: float) -> np.ndarray:
    """Apply softmax to a row of scores."""
    s = row / temperature
    s = s - np.max(s)  # Numerical stability
    exp_s = np.exp(s)
    weights = exp_s / np.sum(exp_s)
    return weights


def _sorted_indices_by_score(row: np.ndarray, columns: pd.Index) -> list[int]:
    """
    Deterministically sort indices by:
    1) score descending
    2) column name ascending (tie-break)
    """
    k = len(row)
    sorting_keys = [(-float(row[i]), str(columns[i])) for i in range(k)]
    return sorted(range(k), key=lambda i: sorting_keys[i])


def _apply_softmax_topk(
    row: np.ndarray,
    columns: pd.Index,
    *,
    top_k: int,
    temperature: float,
    score_transform: str,
    winsorize_q_low: float,
    winsorize_q_high: float,
    zscore_eps: float,
) -> np.ndarray:
    """
    Apply top-k prefilter, then optional score transform, then softmax on active set.

    Assets outside top_k receive exactly 0 weight.
    """
    row = np.asarray(row, dtype=np.float64)
    k = len(row)

    if not isinstance(top_k, int) or not (1 <= top_k <= k):
        raise ValueError(f"top_k must be int in [1, {k}], got {top_k}")

    sorted_indices = _sorted_indices_by_score(row, columns)
    active = sorted_indices[:top_k]

    active_scores = row[active]
    active_scores = _transform_softmax_scores_row(
        active_scores,
        score_transform=score_transform,
        winsorize_q_low=winsorize_q_low,
        winsorize_q_high=winsorize_q_high,
        zscore_eps=zscore_eps,
    )

    active_w = _apply_softmax(active_scores, temperature)

    weights = np.zeros(k, dtype=np.float64)
    weights[active] = active_w
    return weights


def _apply_rank(row: np.ndarray, columns: pd.Index, top_k: Optional[int] = None) -> np.ndarray:
    """
    Convert row to rank-based weights with optional top_k sparsification.
    
    Deterministic tie-breaking: score descending, then column name ascending.
    
    Parameters
    ----------
    row : np.ndarray
        Score values for this row.
    columns : pd.Index
        Column names for deterministic tie-breaking.
    top_k : int, optional
        If provided, only top_k assets get non-zero weight.
    
    Returns
    -------
    np.ndarray
        Rank-based weights summing to 1.0.
    """
    k = len(row)
    
    # Deterministic sorting: score descending, then column name ascending for ties
    sorted_indices = _sorted_indices_by_score(row, columns)
    
    # Determine active set size
    active_count = top_k if top_k is not None else k
    
    # Assign ranks only to active (selected) assets
    # Highest score gets rank = active_count, lowest gets rank = 1
    weights = np.zeros(k, dtype=np.float64)
    
    for rank_pos, idx in enumerate(sorted_indices[:active_count]):
        # rank_pos=0 (highest) gets rank_value = active_count
        # rank_pos=active_count-1 (lowest selected) gets rank_value = 1
        rank_value = active_count - rank_pos
        weights[idx] = rank_value
    
    # Normalize to sum to 1.0
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    
    return weights


def _apply_topk(row: np.ndarray, top_k: int, columns: pd.Index) -> np.ndarray:
    """Select top-k assets by score with equal weights."""
    k = len(row)
    
    # For deterministic tie-breaking: sort by (negative score, column name)
    # We create an auxiliary array for stable sorting
    sorted_indices = _sorted_indices_by_score(row, columns)
    
    # Select top_k
    selected = sorted_indices[:top_k]
    
    weights = np.zeros(k, dtype=np.float64)
    weights[selected] = 1.0 / top_k
    
    return weights


def _enforce_max_weight(
    weights: np.ndarray, 
    max_weight: float,
    preserve_sparsity: bool = True,
) -> np.ndarray:
    """
    Enforce maximum weight constraint via iterative redistribution.
    
    Parameters
    ----------
    weights : np.ndarray
        Initial weights summing to 1.
    max_weight : float
        Maximum allowed weight per asset.
    preserve_sparsity : bool
        If True, redistribute only among active (weight > 0) assets.
    
    Returns
    -------
    np.ndarray
        Capped weights summing to 1.
    
    Raises
    ------
    ValueError
        If max_weight is infeasible for the active asset count.
    """
    weights = weights.copy()
    tol = 1e-12
    
    # Identify active assets (sparse support)
    if preserve_sparsity:
        active_mask = weights > tol
        active_count = np.sum(active_mask)
        
        # Feasibility check: can we allocate 100% with active assets?
        if active_count * max_weight < 1.0 - tol:
            raise ValueError(
                f"Infeasible max_weight={max_weight}: cannot allocate 100% "
                f"with {active_count} active assets "
                f"(max_weight * active_count = {max_weight * active_count:.4f} < 1.0)"
            )
    else:
        active_mask = np.ones(len(weights), dtype=bool)
    
    for _ in range(len(weights)):  # Max iterations = number of assets
        capped = (weights > max_weight + tol) & active_mask
        if not np.any(capped):
            break
        
        # Amount to redistribute
        excess = np.sum(weights[capped] - max_weight)
        
        # Cap the over-weighted assets
        weights[capped] = max_weight
        
        # Find active assets that can receive more weight
        uncapped = (weights < max_weight - tol) & active_mask
        if not np.any(uncapped):
            # All active at max_weight
            break
        
        # Redistribute proportionally to uncapped weights
        uncapped_sum = np.sum(weights[uncapped])
        if uncapped_sum > 0:
            weights[uncapped] += excess * (weights[uncapped] / uncapped_sum)
        else:
            # Equal redistribution among uncapped active
            n_uncapped = np.sum(uncapped)
            weights[uncapped] = excess / n_uncapped
    
    return weights


def scores_to_target_weights(
    scores: pd.DataFrame,
    *,
    method: str = "softmax",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    max_weight: Optional[float] = None,
    score_transform: str = "none",
    winsorize_q_low: float = 0.01,
    winsorize_q_high: float = 0.99,
    zscore_eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Convert model scores to target portfolio weights.
    
    Parameters
    ----------
    scores : pd.DataFrame
        Score matrix with shape (n_dates, k_assets).
        Index = dates, columns = tickers/labels.
    method : str
        Weight construction method:
        - "softmax": exp(s/T) normalized
        - "softmax_topk": top-k prefilter then softmax (sparse)
        - "rank": rank-based weights
        - "topk": equal weight on top-k assets
    temperature : float
        Temperature for softmax method (default 1.0). Must be > 0.
    top_k : int, optional
        Number of top assets to select for topk/softmax_topk methods.
        Required if method in {"topk", "softmax_topk"}.
    max_weight : float, optional
        Maximum weight per asset. If provided, must be in (0, 1].
        Excess weight is redistributed proportionally.
    score_transform : str
        Optional pre-softmax transform applied per-date across assets to reduce
        outlier-driven concentration. Supported:
        - "none" (default)
        - "winsorize"
        - "zscore"
        - "winsorize_zscore"
    winsorize_q_low, winsorize_q_high : float
        Winsorization quantiles when score_transform includes "winsorize".
    zscore_eps : float
        Small epsilon for z-score std guard when score_transform includes "zscore".
    
    Returns
    -------
    pd.DataFrame
        Target weights with same shape, index, and columns as scores.
        Each row sums to 1.0, all values >= 0.
    
    Raises
    ------
    ValueError
        If inputs fail validation.
    
    Examples
    --------
    >>> scores = pd.DataFrame({"A": [1, 2], "B": [3, 1]}, index=["2023-01", "2023-02"])
    >>> weights = scores_to_target_weights(scores, method="softmax")
    """
    # Validate scores
    _validate_scores(scores)
    
    n, k = scores.shape
    
    # Normalize method
    method_lower = method.lower()
    if method_lower not in ("softmax", "softmax_topk", "rank", "topk"):
        raise ValueError(
            f"method must be 'softmax', 'softmax_topk', 'rank', or 'topk', got '{method}'"
        )
    
    # Validate temperature
    if method_lower in ("softmax", "softmax_topk") and temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    
    # Validate score transform (softmax-only)
    if method_lower not in ("softmax", "softmax_topk") and str(score_transform).strip().lower() != "none":
        raise ValueError("score_transform is only supported when method='softmax' or 'softmax_topk'")
    
    # Validate top_k
    if method_lower in ("topk", "softmax_topk"):
        if top_k is None:
            raise ValueError(f"top_k is required when method='{method_lower}'")
        if not isinstance(top_k, int) or not (1 <= top_k <= k):
            raise ValueError(f"top_k must be int in [1, {k}], got {top_k}")
    
    # Validate max_weight (defer active-set feasibility to enforcement)
    if max_weight is not None:
        if not (0 < max_weight <= 1):
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
    
    # Validate top_k for rank and topk methods
    if method_lower in ("topk", "softmax_topk"):
        if top_k is None:
            raise ValueError(f"top_k is required when method='{method_lower}'")
        if not isinstance(top_k, int) or not (1 <= top_k <= k):
            raise ValueError(f"top_k must be int in [1, {k}], got {top_k}")
    
    if method_lower == "rank" and top_k is not None:
        if not isinstance(top_k, int) or not (1 <= top_k <= k):
            raise ValueError(f"top_k must be int in [1, {k}], got {top_k}")
    
    # Determine active asset count for max_weight feasibility pre-check
    if max_weight is not None:
        if method_lower in ("topk", "softmax_topk"):
            active_count = top_k
        elif method_lower == "rank" and top_k is not None:
            active_count = top_k
        else:
            active_count = k  # Dense methods use all assets
        
        if max_weight * active_count < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible max_weight={max_weight}: cannot allocate 100% "
                f"with {active_count} active assets "
                f"(max_weight * active_count = {max_weight * active_count:.4f} < 1.0)"
            )
    
    # Compute weights for each row
    weights_arr = np.zeros((n, k), dtype=np.float64)
    
    for i in range(n):
        row = scores.values[i]
        
        if method_lower == "softmax":
            row2 = _transform_softmax_scores_row(
                row,
                score_transform=score_transform,
                winsorize_q_low=winsorize_q_low,
                winsorize_q_high=winsorize_q_high,
                zscore_eps=zscore_eps,
            )
            weights_arr[i] = _apply_softmax(row2, temperature)
        elif method_lower == "softmax_topk":
            weights_arr[i] = _apply_softmax_topk(
                row,
                scores.columns,
                top_k=int(top_k),
                temperature=temperature,
                score_transform=score_transform,
                winsorize_q_low=winsorize_q_low,
                winsorize_q_high=winsorize_q_high,
                zscore_eps=zscore_eps,
            )
        elif method_lower == "rank":
            weights_arr[i] = _apply_rank(row, scores.columns, top_k)
        else:  # topk
            weights_arr[i] = _apply_topk(row, top_k, scores.columns)
        
        # Apply max_weight cap (preserve sparsity for sparse methods)
        if max_weight is not None:
            is_sparse = (method_lower in ("topk", "softmax_topk")) or (method_lower == "rank" and top_k is not None)
            weights_arr[i] = _enforce_max_weight(
                weights_arr[i], max_weight, preserve_sparsity=is_sparse
            )
    
    # Build output DataFrame
    result = pd.DataFrame(
        weights_arr,
        index=scores.index,
        columns=scores.columns
    )
    
    return result
