"""
SHAP-Style Feature Attribution via Shapley Sampling.

Provides:
- shapley_sampling_values: Compute per-feature Shapley values via permutation sampling
- global_feature_importance: Aggregate Shapley values into global importance

Design Principles:
- No external dependencies beyond numpy/pandas (no shap library)
- Deterministic: reproducible given fixed seed
- Fail-fast: raise clear exceptions on invalid inputs

References:
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
- Strumbelj & Kononenko, "Explaining prediction models and individual predictions with feature contributions" (2014)
"""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


def _coerce_to_2d_array(arr: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Coerce input to 2D numpy array."""
    if isinstance(arr, pd.DataFrame):
        result = arr.values
    else:
        result = np.asarray(arr)
    
    if result.ndim == 1:
        result = result.reshape(1, -1)
    
    if result.ndim != 2:
        raise ValueError(f"Input must be 1D or 2D, got {result.ndim}D")
    
    return result.astype(np.float64)


def _validate_inputs(
    x: np.ndarray,
    baseline: Optional[np.ndarray],
    n_permutations: int
) -> None:
    """Validate inputs and raise ValueError on issues."""
    n, d = x.shape
    
    if n < 1:
        raise ValueError(f"x must have at least 1 sample, got {n}")
    if d < 1:
        raise ValueError(f"x must have at least 1 feature, got {d}")
    
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or inf values")
    
    if baseline is not None:
        if baseline.shape != (d,):
            raise ValueError(
                f"baseline must have shape ({d},), got {baseline.shape}"
            )
        if not np.all(np.isfinite(baseline)):
            raise ValueError("baseline contains NaN or inf values")
    
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")


def shapley_sampling_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: Union[np.ndarray, pd.DataFrame],
    *,
    baseline: Optional[np.ndarray] = None,
    background: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    n_permutations: int = 128,
    seed: int = 0
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Compute per-feature Shapley value attributions via permutation sampling.
    
    Parameters
    ----------
    predict_fn : callable
        Model prediction function accepting (n, d) array, returning:
        - (n,) for single-output
        - (n, k) for multi-output
    x : np.ndarray or pd.DataFrame
        Input samples, shape (n, d).
    baseline : np.ndarray, optional
        Reference point, shape (d,). Features are "missing" when set to baseline values.
        If None and background provided, uses mean(background, axis=0).
        If both None, uses zeros.
    background : np.ndarray or pd.DataFrame, optional
        Background dataset, shape (m, d), used to derive baseline if baseline is None.
    n_permutations : int
        Number of random feature orderings to sample (default 128).
    seed : int
        Random seed for reproducibility (default 0).
    
    Returns
    -------
    shap_values : np.ndarray or pd.DataFrame
        - Single-output: shape (n, d), same type as x if x is DataFrame
        - Multi-output: shape (n, d, k), always numpy array
    
    Raises
    ------
    ValueError
        If inputs are invalid (NaN, shape mismatch, etc.).
    
    Notes
    -----
    Algorithm (Shapley sampling approximation):
    For each sample x_i and each random permutation of features:
    1. Start with x_curr = baseline
    2. For each feature j in permutation order:
       - Set x_curr[j] = x_i[j]
       - Compute marginal contribution = f(x_curr) - f(x_prev)
       - Add marginal to contribution for feature j
    3. Average contributions across permutations
    """
    # Remember input type for output
    input_is_df = isinstance(x, pd.DataFrame)
    input_index = None
    input_columns = None
    
    if input_is_df:
        input_index = x.index
        input_columns = x.columns
    
    # Coerce to numpy
    x_arr = _coerce_to_2d_array(x)
    n, d = x_arr.shape
    
    # Determine baseline
    if baseline is not None:
        baseline_arr = np.asarray(baseline).flatten().astype(np.float64)
    elif background is not None:
        bg_arr = _coerce_to_2d_array(background)
        if bg_arr.shape[1] != d:
            raise ValueError(
                f"background must have {d} features, got {bg_arr.shape[1]}"
            )
        if not np.all(np.isfinite(bg_arr)):
            raise ValueError("background contains NaN or inf values")
        baseline_arr = np.mean(bg_arr, axis=0)
    else:
        baseline_arr = np.zeros(d, dtype=np.float64)
    
    # Validate inputs
    _validate_inputs(x_arr, baseline_arr, n_permutations)
    
    # Probe model output shape
    probe_out = predict_fn(x_arr[:1])
    if probe_out.shape[0] != 1:
        raise ValueError(
            f"predict_fn output must have shape (n, ...) matching input n, "
            f"but for n=1 input got output shape {probe_out.shape}"
        )
    
    if probe_out.ndim == 1:
        is_multi_output = False
        k = 1
    elif probe_out.ndim == 2:
        is_multi_output = True
        k = probe_out.shape[1]
    else:
        raise ValueError(
            f"predict_fn must return (n,) or (n,k), got shape {probe_out.shape}"
        )
    
    # Initialize contributions
    if is_multi_output:
        contributions = np.zeros((n, d, k), dtype=np.float64)
    else:
        contributions = np.zeros((n, d), dtype=np.float64)
    
    # Random generator for reproducibility
    rng = np.random.default_rng(seed)
    
    # Shapley sampling
    for _ in range(n_permutations):
        # Random permutation of feature indices
        perm = rng.permutation(d)
        
        for i in range(n):
            x_i = x_arr[i]
            x_curr = baseline_arr.copy()
            
            # Compute f(baseline)
            f_prev = predict_fn(x_curr.reshape(1, -1))
            if f_prev.ndim == 1:
                f_prev = f_prev[0]
            else:
                f_prev = f_prev[0]  # (k,) for multi-output
            
            for j in perm:
                # Set feature j to actual value
                x_curr[j] = x_i[j]
                
                # Compute f(x_curr)
                f_curr = predict_fn(x_curr.reshape(1, -1))
                if f_curr.ndim == 1:
                    f_curr = f_curr[0]
                else:
                    f_curr = f_curr[0]  # (k,)
                
                # Marginal contribution
                marginal = f_curr - f_prev
                
                # Accumulate
                if is_multi_output:
                    contributions[i, j, :] += marginal
                else:
                    contributions[i, j] += marginal
                
                f_prev = f_curr
    
    # Average over permutations
    contributions /= n_permutations
    
    # Return appropriate type
    if is_multi_output:
        # Multi-output: always return numpy (n, d, k)
        return contributions
    else:
        # Single-output: return DataFrame if input was DataFrame
        if input_is_df:
            return pd.DataFrame(contributions, index=input_index, columns=input_columns)
        return contributions


def global_feature_importance(
    shap_values: np.ndarray,
    *,
    agg: str = "mean_abs"
) -> np.ndarray:
    """
    Aggregate SHAP values into global feature importance.
    
    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n, d) for single-output or (n, d, k) for multi-output.
    agg : str
        Aggregation method:
        - "mean_abs": mean of absolute values over samples (default)
        - "mean": mean over samples (signed)
    
    Returns
    -------
    importance : np.ndarray
        Shape (d,) for single-output or (d, k) for multi-output.
    
    Raises
    ------
    ValueError
        If agg is not recognized or shap_values has invalid shape.
    """
    if agg not in ("mean_abs", "mean"):
        raise ValueError(f"agg must be 'mean_abs' or 'mean', got '{agg}'")
    
    if isinstance(shap_values, pd.DataFrame):
        shap_arr = shap_values.values
    else:
        shap_arr = np.asarray(shap_values)
    
    if shap_arr.ndim not in (2, 3):
        raise ValueError(
            f"shap_values must be 2D (n,d) or 3D (n,d,k), got {shap_arr.ndim}D"
        )
    
    if agg == "mean_abs":
        return np.mean(np.abs(shap_arr), axis=0)
    else:  # mean
        return np.mean(shap_arr, axis=0)
