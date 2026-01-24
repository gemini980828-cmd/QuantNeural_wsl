"""
Temperature Scaling Calibration for Binary Classification.

Provides:
- TemperatureScalerBinary: Post-hoc probability calibration via temperature scaling

Design Principles:
- No external dependencies beyond numpy/pandas (standard project deps)
- Deterministic: no randomness, no system clock, no network
- Fail-fast: raise clear exceptions on invalid inputs

References:
- Guo et al., "On Calibration of Modern Neural Networks" (2017)
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    
    Uses the identity: sigmoid(z) = exp(z)/(1+exp(z)) for z>=0,
    sigmoid(z) = 1/(1+exp(-z)) for z<0 to avoid overflow.
    """
    result = np.empty_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    
    # For z >= 0: 1 / (1 + exp(-z))
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    
    # For z < 0: exp(z) / (1 + exp(z))
    exp_z = np.exp(z[neg_mask])
    result[neg_mask] = exp_z / (1.0 + exp_z)
    
    return result


def _coerce_scores_to_1d(arr: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """Coerce scores to 1D numpy array."""
    if isinstance(arr, pd.DataFrame):
        if arr.shape[1] == 1:
            result = arr.iloc[:, 0].values
        else:
            raise ValueError(f"DataFrame must have 1 column for binary scoring, got {arr.shape[1]}")
    elif isinstance(arr, pd.Series):
        result = arr.values
    else:
        result = np.asarray(arr).flatten()
    
    return result.astype(np.float64)


def _coerce_labels_to_1d(arr: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """Coerce labels to 1D numpy array."""
    if isinstance(arr, pd.DataFrame):
        if arr.shape[1] == 1:
            result = arr.iloc[:, 0].values
        else:
            result = arr.values.flatten()
    elif isinstance(arr, pd.Series):
        result = arr.values
    else:
        result = np.asarray(arr).flatten()
    
    return result


class TemperatureScalerBinary:
    """
    Temperature Scaling Calibrator for Binary Classification.
    
    Calibrates model confidence by learning a single temperature T > 0 that
    minimizes negative log-likelihood on a calibration set.
    
    For logits z:
        calibrated_prob = sigmoid(z / T)
    
    A higher T (T > 1) softens predictions (less confident).
    A lower T (T < 1) sharpens predictions (more confident).
    T = 1.0 is the baseline (no change).
    
    Parameters
    ----------
    grid_size : int
        Number of temperature values to search (default 400).
    t_min : float
        Minimum temperature to search (default 0.05).
    t_max : float
        Maximum temperature to search (default 10.0).
    eps : float
        Small value for numerical stability (default 1e-12).
    
    Attributes
    ----------
    temperature_ : float
        Best temperature found after fitting.
    fitted_ : bool
        Whether fit() has been called.
    
    Examples
    --------
    >>> ts = TemperatureScalerBinary()
    >>> ts.fit(logits_cal, y_cal, input_type="logits")
    >>> proba_calibrated = ts.transform(logits_test, input_type="logits")
    """
    
    def __init__(
        self,
        *,
        grid_size: int = 400,
        t_min: float = 0.05,
        t_max: float = 10.0,
        eps: float = 1e-12
    ):
        # Validate constructor args
        if grid_size < 10:
            raise ValueError(f"grid_size must be >= 10, got {grid_size}")
        if not (0 < t_min < t_max):
            raise ValueError(f"Must have 0 < t_min < t_max, got t_min={t_min}, t_max={t_max}")
        if not (0 < eps < 1e-3):
            raise ValueError(f"eps must be in (0, 1e-3), got {eps}")
        
        self.grid_size = grid_size
        self.t_min = t_min
        self.t_max = t_max
        self.eps = eps
        
        self.temperature_: Optional[float] = None
        self.fitted_: bool = False
    
    def _to_logits(
        self,
        scores: np.ndarray,
        input_type: str
    ) -> np.ndarray:
        """Convert scores to logits based on input_type."""
        if input_type == "logits":
            return scores
        elif input_type == "proba":
            # Validate probabilities in [0, 1]
            if np.any(scores < -1e-6) or np.any(scores > 1 + 1e-6):
                raise ValueError(
                    f"For input_type='proba', scores must be in [0,1]. "
                    f"Got min={scores.min():.4f}, max={scores.max():.4f}"
                )
            # Clip to [eps, 1-eps] and convert to logits
            p_clipped = np.clip(scores, self.eps, 1 - self.eps)
            return np.log(p_clipped / (1 - p_clipped))
        else:
            raise ValueError(f"input_type must be 'logits' or 'proba', got '{input_type}'")
    
    def _compute_nll(
        self,
        logits: np.ndarray,
        y: np.ndarray,
        temperature: float
    ) -> float:
        """Compute mean negative log-likelihood for a given temperature."""
        p = _stable_sigmoid(logits / temperature)
        p_clipped = np.clip(p, self.eps, 1 - self.eps)
        
        nll = -np.mean(
            y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)
        )
        return nll
    
    def fit(
        self,
        scores: Union[np.ndarray, pd.DataFrame, pd.Series],
        y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
        *,
        input_type: str = "logits"
    ) -> "TemperatureScalerBinary":
        """
        Fit the temperature scaler on a calibration set.
        
        Parameters
        ----------
        scores : array-like
            Model outputs (logits or probabilities), shape (n,) or (n, 1).
        y_true : array-like
            True binary labels {0, 1}, shape (n,) or (n, 1).
        input_type : str
            "logits" or "proba" (default "logits").
        
        Returns
        -------
        self
            Fitted scaler.
        
        Raises
        ------
        ValueError
            If inputs are invalid (NaN/inf, non-binary labels, n < 2).
        """
        # Coerce to 1D
        scores_arr = _coerce_scores_to_1d(scores)
        y_arr = _coerce_labels_to_1d(y_true)
        
        n = len(scores_arr)
        
        # Validate n >= 2
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        
        # Validate shapes match
        if len(y_arr) != n:
            raise ValueError(
                f"Shape mismatch: scores has {n} samples, y_true has {len(y_arr)}"
            )
        
        # Validate finite scores
        if not np.all(np.isfinite(scores_arr)):
            raise ValueError("scores contains NaN or inf values")
        
        # Validate binary labels
        unique_labels = np.unique(y_arr)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"y_true must be binary {{0, 1}}, got unique values: {unique_labels}"
            )
        
        # Convert to logits
        logits = self._to_logits(scores_arr, input_type)
        y = y_arr.astype(np.float64)
        
        # Grid search for best temperature
        t_grid = np.logspace(
            np.log10(self.t_min),
            np.log10(self.t_max),
            self.grid_size
        )
        
        best_t = 1.0
        best_nll = float('inf')
        
        for t in t_grid:
            nll = self._compute_nll(logits, y, t)
            if nll < best_nll:
                best_nll = nll
                best_t = t
        
        self.temperature_ = best_t
        self.fitted_ = True
        
        return self
    
    def transform(
        self,
        scores: Union[np.ndarray, pd.DataFrame, pd.Series],
        *,
        input_type: str = "logits"
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Apply temperature scaling to obtain calibrated probabilities.
        
        Parameters
        ----------
        scores : array-like
            Model outputs (logits or probabilities).
        input_type : str
            "logits" or "proba" (default "logits").
        
        Returns
        -------
        calibrated : same type as input
            Calibrated probabilities in (0, 1).
        
        Raises
        ------
        RuntimeError
            If fit() has not been called.
        ValueError
            If inputs are invalid.
        """
        if not self.fitted_:
            raise RuntimeError("Must call fit() before transform()")
        
        # Remember input type for output
        input_container = type(scores)
        input_index = None
        input_columns = None
        original_shape = None
        
        if isinstance(scores, pd.DataFrame):
            input_index = scores.index
            input_columns = scores.columns
            original_shape = scores.shape
        elif isinstance(scores, pd.Series):
            input_index = scores.index
        else:
            original_shape = np.asarray(scores).shape
        
        # Coerce to 1D
        scores_arr = _coerce_scores_to_1d(scores)
        
        # Validate finite
        if not np.all(np.isfinite(scores_arr)):
            raise ValueError("scores contains NaN or inf values")
        
        # Convert to logits
        logits = self._to_logits(scores_arr, input_type)
        
        # Apply temperature scaling
        calibrated = _stable_sigmoid(logits / self.temperature_)
        
        # Clip to ensure (0, 1)
        calibrated = np.clip(calibrated, self.eps, 1 - self.eps)
        
        # Convert back to input type
        if input_container == pd.DataFrame:
            return pd.DataFrame(
                calibrated.reshape(original_shape),
                index=input_index,
                columns=input_columns
            )
        elif input_container == pd.Series:
            return pd.Series(calibrated, index=input_index)
        else:
            # Return same shape as input
            if original_shape is not None:
                return calibrated.reshape(original_shape)
            return calibrated
    
    def predict_proba(
        self,
        scores: Union[np.ndarray, pd.DataFrame, pd.Series],
        *,
        input_type: str = "logits"
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Alias for transform(). Apply temperature scaling.
        
        Parameters
        ----------
        scores : array-like
            Model outputs (logits or probabilities).
        input_type : str
            "logits" or "proba" (default "logits").
        
        Returns
        -------
        calibrated : same type as input
            Calibrated probabilities in (0, 1).
        """
        return self.transform(scores, input_type=input_type)
