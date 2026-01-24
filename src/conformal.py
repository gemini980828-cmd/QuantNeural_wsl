"""
Split Conformal Prediction Intervals for Multi-Output Regression.

Provides:
- SplitConformalRegressor: Compute calibrated prediction intervals using split conformal

Design Principles:
- No external dependencies beyond numpy/pandas (standard project deps)
- Deterministic: no randomness, no system clock, no network
- Fail-fast: raise clear exceptions on invalid inputs
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def _coerce_to_2d_array(arr: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """
    Coerce input to 2D numpy array.
    
    Parameters
    ----------
    arr : np.ndarray, pd.DataFrame, or pd.Series
        Input array-like.
    
    Returns
    -------
    np.ndarray
        2D array with shape (n, k).
    """
    if isinstance(arr, pd.DataFrame):
        result = arr.values
    elif isinstance(arr, pd.Series):
        result = arr.values
    else:
        result = np.asarray(arr)
    
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    
    if result.ndim != 2:
        raise ValueError(f"Input must be 1D or 2D, got {result.ndim}D")
    
    return result


class SplitConformalRegressor:
    """
    Split Conformal Prediction for multi-output regression.
    
    Computes calibrated prediction intervals using residuals from a calibration set.
    
    Algorithm:
    1. fit(y_true, y_pred): compute residuals = |y_true - y_pred| on calibration set
    2. quantile(alpha): compute the (1-alpha) quantile of residuals per output
    3. predict_interval(y_pred_new, alpha): return (y_pred_new - q, y_pred_new + q)
    
    Attributes
    ----------
    residuals_ : np.ndarray
        Calibration residuals, shape (n, k).
    n_ : int
        Number of calibration samples.
    k_ : int
        Number of outputs.
    
    Examples
    --------
    >>> scr = SplitConformalRegressor()
    >>> scr.fit(y_cal_true, y_cal_pred)
    >>> lower, upper = scr.predict_interval(y_test_pred, alpha=0.1)
    """
    
    def __init__(self):
        self.residuals_: Optional[np.ndarray] = None
        self.n_: int = 0
        self.k_: int = 0
        self._fitted: bool = False
    
    def fit(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
        y_pred: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> "SplitConformalRegressor":
        """
        Compute calibration residuals from true and predicted values.
        
        Parameters
        ----------
        y_true : array-like
            True values on calibration set, shape (n,) or (n, k).
        y_pred : array-like
            Predicted values on calibration set, same shape as y_true.
        
        Returns
        -------
        self
            Fitted regressor.
        
        Raises
        ------
        ValueError
            If shapes don't match, inputs contain NaN/inf, or n < 2.
        """
        # Coerce to 2D numpy arrays
        y_true_arr = _coerce_to_2d_array(y_true)
        y_pred_arr = _coerce_to_2d_array(y_pred)
        
        # Shape validation
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true_arr.shape}, "
                f"y_pred has shape {y_pred_arr.shape}"
            )
        
        n, k = y_true_arr.shape
        
        # n >= 2 validation
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        
        # Finite validation
        if not np.all(np.isfinite(y_true_arr)):
            raise ValueError("y_true contains NaN or inf values")
        
        if not np.all(np.isfinite(y_pred_arr)):
            raise ValueError("y_pred contains NaN or inf values")
        
        # Compute residuals
        self.residuals_ = np.abs(y_true_arr - y_pred_arr)
        self.n_ = n
        self.k_ = k
        self._fitted = True
        
        return self
    
    def quantile(self, *, alpha: float) -> np.ndarray:
        """
        Compute the conformal quantile for each output.
        
        Parameters
        ----------
        alpha : float
            Miscoverage rate, must satisfy 0 < alpha < 1.
        
        Returns
        -------
        np.ndarray
            Quantile values, shape (k,).
        
        Raises
        ------
        RuntimeError
            If fit() has not been called.
        ValueError
            If alpha is invalid.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before quantile()")
        
        # Alpha validation
        if not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be a number, got {type(alpha).__name__}")
        
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}")
        
        n = self.n_
        k = self.k_
        
        # Compute k_index: ceil((n + 1) * (1 - alpha))
        k_index = int(np.ceil((n + 1) * (1 - alpha)))
        
        # Clamp to [1, n]
        k_index = max(1, min(k_index, n))
        
        # Compute qhat for each column using np.partition (O(n) vs O(n log n) for full sort)
        qhat = np.zeros(k, dtype=np.float64)
        for j in range(k):
            col = self.residuals_[:, j]
            # np.partition puts the k_index-1 smallest element at position k_index-1
            # (k_index-1 because 0-indexed, and we want the k_index-th smallest)
            partitioned = np.partition(col, k_index - 1)
            qhat[j] = partitioned[k_index - 1]
        
        return qhat
    
    def predict_interval(
        self,
        y_pred: Union[np.ndarray, pd.DataFrame, pd.Series],
        *,
        alpha: float
    ) -> Tuple[Union[np.ndarray, pd.DataFrame, pd.Series], 
               Union[np.ndarray, pd.DataFrame, pd.Series]]:
        """
        Compute prediction intervals for new predictions.
        
        Parameters
        ----------
        y_pred : array-like
            New predictions, shape (m,) or (m, k).
        alpha : float
            Miscoverage rate, must satisfy 0 < alpha < 1.
        
        Returns
        -------
        lower, upper : tuple
            Lower and upper bounds of prediction intervals.
            Returns same type as input (DataFrame, Series, or ndarray).
        
        Raises
        ------
        RuntimeError
            If fit() has not been called.
        ValueError
            If inputs are invalid (NaN/inf, shape mismatch, invalid alpha).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict_interval()")
        
        # Remember input type for output
        input_type = type(y_pred)
        input_index = None
        input_columns = None
        
        if isinstance(y_pred, pd.DataFrame):
            input_index = y_pred.index
            input_columns = y_pred.columns
        elif isinstance(y_pred, pd.Series):
            input_index = y_pred.index
        
        # Coerce to 2D numpy array
        y_pred_arr = _coerce_to_2d_array(y_pred)
        
        # Finite validation
        if not np.all(np.isfinite(y_pred_arr)):
            raise ValueError("y_pred contains NaN or inf values")
        
        # k dimension validation
        if y_pred_arr.shape[1] != self.k_:
            raise ValueError(
                f"k dimension mismatch: fitted with k={self.k_}, "
                f"got y_pred with k={y_pred_arr.shape[1]}"
            )
        
        # Compute quantile
        qhat = self.quantile(alpha=alpha)
        
        # Compute bounds
        lower_arr = y_pred_arr - qhat
        upper_arr = y_pred_arr + qhat
        
        # Convert back to input type
        if input_type == pd.DataFrame:
            lower = pd.DataFrame(lower_arr, index=input_index, columns=input_columns)
            upper = pd.DataFrame(upper_arr, index=input_index, columns=input_columns)
        elif input_type == pd.Series:
            # Series: squeeze to 1D if k=1
            if self.k_ == 1:
                lower = pd.Series(lower_arr.flatten(), index=input_index)
                upper = pd.Series(upper_arr.flatten(), index=input_index)
            else:
                # Multi-output from Series is ambiguous, return DataFrame
                lower = pd.DataFrame(lower_arr, index=input_index)
                upper = pd.DataFrame(upper_arr, index=input_index)
        else:
            # Squeeze if original was 1D and k=1
            if self.k_ == 1 and y_pred_arr.shape[0] == lower_arr.size:
                lower = lower_arr.flatten()
                upper = upper_arr.flatten()
            else:
                lower = lower_arr
                upper = upper_arr
        
        return lower, upper
