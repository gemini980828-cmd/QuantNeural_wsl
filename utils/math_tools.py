"""
Math tools module for QUANT-NEURAL.

Provides mathematical utilities for quantitative finance calculations.
"""

import numpy as np


def weighted_harmonic_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted harmonic mean with robust handling of invalid values.

    This function is designed for valuation aggregation where:
    - Values must be positive (e.g., P/E, P/B ratios)
    - Invalid observations (non-positive, NaN, Inf) are excluded
    - Weights are rescaled after exclusion

    Parameters
    ----------
    values : np.ndarray
        Array of values (e.g., valuation ratios). Only finite values > 0 are used.
    weights : np.ndarray
        Array of weights. Must be non-negative. Only finite weights > 0 are used.

    Returns
    -------
    float
        The weighted harmonic mean, or np.nan if no valid entries remain.

    Raises
    ------
    ValueError
        If shapes of values and weights do not match.
        If any weight is negative.

    Notes
    -----
    - Values <= 0 are excluded (invalid for harmonic mean).
    - Zero weights are excluded (contribute nothing).
    - Remaining weights are rescaled to sum to 1.0.
    - If all entries are invalid, returns np.nan (fail-safe, not fail-fast,
      since this is a valid outcome for degenerate inputs).

    Examples
    --------
    >>> weighted_harmonic_mean(np.array([2.0, 4.0]), np.array([0.5, 0.5]))
    2.6666666666666665

    >>> weighted_harmonic_mean(np.array([2.0, -1.0, 4.0]), np.array([0.2, 0.7, 0.1]))
    2.4  # -1.0 excluded, weights rescaled
    """
    # Convert to float numpy arrays
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Check shape match
    if values.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: values has shape {values.shape}, "
            f"weights has shape {weights.shape}"
        )

    # Check for negative weights (before any masking)
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    # Build valid mask:
    # - values must be finite and > 0
    # - weights must be finite and > 0
    valid_mask = (
        np.isfinite(values) & (values > 0) &
        np.isfinite(weights) & (weights > 0)
    )

    # If no valid entries, return NaN
    if not np.any(valid_mask):
        return float("nan")

    # Extract valid values and weights
    v = values[valid_mask]
    w = weights[valid_mask]

    # Rescale weights to sum to 1
    w_sum = np.sum(w)
    if w_sum <= 0 or not np.isfinite(w_sum):
        return float("nan")
    w = w / w_sum

    # Compute weighted harmonic mean: 1 / sum(w_i / v_i)
    denom = np.sum(w / v)

    if denom <= 0 or not np.isfinite(denom):
        return float("nan")

    return float(np.sum(w) / denom)  # sum(w) == 1 after rescale
