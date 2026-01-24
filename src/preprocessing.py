"""
Preprocessing module for QUANT-NEURAL.

Provides:
- QuantDataProcessor: RankGauss (QuantileTransformer) with train-only fit.
- HP filter: Hodrick-Prescott with lambda mapping for classic/ravn_uhlig/manual.
- Hamilton regression filter: residual series aligned to forward target timestamp.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from statsmodels.tsa.filters.hp_filter import hpfilter


# Type aliases
HPMode = Literal["classic", "ravn_uhlig", "manual"]
FilterType = Literal["hp", "hamilton"]


@dataclass(frozen=True)
class HPParams:
    """Parameters for Hodrick-Prescott filter.
    
    Attributes
    ----------
    mode : HPMode
        Lambda selection mode: "classic", "ravn_uhlig", or "manual".
    lamb_manual : Optional[float]
        Manual lambda value. Required if mode="manual".
    leakage_guard : bool
        If True, use PIT-safe rolling window approach (default).
        If False, use standard two-sided filter (leaks future data).
    lookback : Optional[int]
        Maximum window size for rolling HP filter when leakage_guard=True.
        None means use all available past data. Default 120 (months).
    """
    mode: HPMode = "classic"
    lamb_manual: Optional[float] = None
    leakage_guard: bool = True
    lookback: Optional[int] = 120


@dataclass(frozen=True)
class HamiltonParams:
    """Parameters for Hamilton regression filter.
    
    Attributes
    ----------
    h : int
        Forecast horizon (forward-looking periods). Default 24.
    p : int
        Number of lags to use as predictors. Default 12.
    """
    h: int = 24
    p: int = 12


class QuantDataProcessor:
    """
    Quantitative data processor with RankGauss and filtering capabilities.
    
    Implements train-only fit pattern for RankGauss transformation.
    
    Parameters
    ----------
    rankgauss : bool
        Whether to enable RankGauss (QuantileTransformer) normalization.
    n_quantiles : int
        Maximum number of quantiles for QuantileTransformer.
    random_state : int
        Random state for reproducibility.
    
    Examples
    --------
    >>> proc = QuantDataProcessor(rankgauss=True, random_state=42)
    >>> proc.fit_rankgauss(X_train)
    >>> X_train_rg = proc.transform_rankgauss(X_train)
    >>> X_val_rg = proc.transform_rankgauss(X_val)  # transform only
    """
    
    # HP lambda mappings
    _HP_CLASSIC = {"M": 14400.0, "Q": 1600.0, "Y": 100.0}
    _HP_RAVN_UHLIG = {"M": 129600.0, "Q": 1600.0, "Y": 6.25}
    
    def __init__(
        self,
        *,
        rankgauss: bool = False,
        n_quantiles: int = 2000,
        random_state: int = 42
    ):
        self.rankgauss = rankgauss
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self._qt: Optional[QuantileTransformer] = None
        self._is_fitted = False
    
    def fit_rankgauss(self, X_train: np.ndarray) -> None:
        """
        Fit RankGauss transformer on TRAINING data only.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix. Shape (n_samples, n_features).
        
        Notes
        -----
        - If rankgauss=False, this method does nothing.
        - Uses QuantileTransformer with output_distribution="normal".
        - n_quantiles is capped at X_train.shape[0].
        """
        if not self.rankgauss:
            return
        
        X_train = np.asarray(X_train, dtype=np.float64)
        n_quantiles_actual = min(self.n_quantiles, X_train.shape[0])
        
        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles_actual,
            subsample=int(1e9),
            random_state=self.random_state
        )
        self._qt.fit(X_train)
        self._is_fitted = True
    
    def transform_rankgauss(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted RankGauss transformer.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform. Shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Transformed feature matrix (Gaussian-distributed).
        
        Raises
        ------
        RuntimeError
            If rankgauss=True but fit_rankgauss was not called first.
        """
        if not self.rankgauss:
            return np.asarray(X, dtype=np.float64)
        
        if not self._is_fitted or self._qt is None:
            raise RuntimeError(
                "RankGauss is enabled but fit_rankgauss() was not called. "
                "Call fit_rankgauss(X_train) before transform_rankgauss()."
            )
        
        X = np.asarray(X, dtype=np.float64)
        return self._qt.transform(X)
    
    @staticmethod
    def _hp_lambda(freq: Literal["M", "Q", "Y"], hp: HPParams) -> float:
        """
        Get HP filter lambda value based on frequency and mode.
        
        Parameters
        ----------
        freq : Literal["M", "Q", "Y"]
            Data frequency: Monthly, Quarterly, or Yearly.
        hp : HPParams
            HP filter parameters.
        
        Returns
        -------
        float
            Lambda value for HP filter.
        
        Raises
        ------
        ValueError
            If mode="manual" but lamb_manual is not provided.
            If freq is not recognized.
        """
        if hp.mode == "manual":
            if hp.lamb_manual is None:
                raise ValueError(
                    "HPParams mode='manual' requires lamb_manual to be set."
                )
            return hp.lamb_manual
        
        if freq not in ("M", "Q", "Y"):
            raise ValueError(f"Unsupported frequency: {freq}. Must be M, Q, or Y.")
        
        if hp.mode == "classic":
            return QuantDataProcessor._HP_CLASSIC[freq]
        elif hp.mode == "ravn_uhlig":
            return QuantDataProcessor._HP_RAVN_UHLIG[freq]
        else:
            raise ValueError(f"Unknown HP mode: {hp.mode}")
    
    def apply_hp_filter(
        self,
        series: pd.Series,
        *,
        freq: Literal["M", "Q", "Y"],
        hp: HPParams
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply Hodrick-Prescott filter to a time series.
        
        Parameters
        ----------
        series : pd.Series
            Time series data with datetime-like index.
        freq : Literal["M", "Q", "Y"]
            Data frequency for lambda selection.
        hp : HPParams
            HP filter parameters.
        
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (cycle, trend) series with same index as input (after dropna).
        
        Notes
        -----
        If hp.leakage_guard=True (default), uses PIT-safe rolling window approach
        where output at time t depends only on data up to t.
        
        If hp.leakage_guard=False, uses standard two-sided filter which may leak
        future information (suitable for offline analysis only).
        """
        # Clean and sort
        series = series.dropna().astype(float)
        series = series.sort_index()
        
        if len(series) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        lamb = self._hp_lambda(freq, hp)
        
        if hp.leakage_guard:
            # PIT-safe rolling window approach
            n = len(series)
            cycle_vals = np.zeros(n, dtype=np.float64)
            trend_vals = np.zeros(n, dtype=np.float64)
            
            MIN_WINDOW_SIZE = 5  # Minimum points for stable HP filter
            
            for i in range(n):
                # Define window: data up to i (inclusive)
                if hp.lookback is None:
                    window_start = 0
                else:
                    window_start = max(0, i + 1 - hp.lookback)
                
                window = series.values[window_start:i+1]
                
                if len(window) < MIN_WINDOW_SIZE:
                    # Fallback: not enough data for HP filter
                    trend_vals[i] = series.values[i]
                    cycle_vals[i] = 0.0
                else:
                    try:
                        c, t = hpfilter(window, lamb=lamb)
                        # Use LAST element of filtered window (corresponds to time i)
                        cycle_vals[i] = c[-1]
                        trend_vals[i] = t[-1]
                    except Exception:
                        # hpfilter failed (numerical issues)
                        trend_vals[i] = series.values[i]
                        cycle_vals[i] = 0.0
        else:
            # Standard two-sided filter (may leak future data)
            cycle_vals, trend_vals = hpfilter(series.values, lamb=lamb)
        
        return (
            pd.Series(cycle_vals, index=series.index, name="cycle"),
            pd.Series(trend_vals, index=series.index, name="trend")
        )
    
    @staticmethod
    def apply_hamilton_filter(
        series: pd.Series,
        *,
        params: HamiltonParams
    ) -> pd.Series:
        """
        Apply Hamilton regression filter to extract cyclical component.
        
        The Hamilton filter uses an h-period ahead forecast based on p lags.
        The residual (actual - forecast) is the cyclical component.
        
        Parameters
        ----------
        series : pd.Series
            Time series data with datetime-like index.
        params : HamiltonParams
            Hamilton filter parameters (h=horizon, p=lags).
        
        Returns
        -------
        pd.Series
            Residual series indexed by TARGET timestamps (y_{t+h}).
            Empty series if insufficient data.
        
        Notes
        -----
        - For each observation i where we can form p lags and a forward target:
          predictors = [y[i], y[i-1], ..., y[i-p+1]]
          target = y[i+h]
        - The residual timestamp corresponds to the TARGET timestamp (index[i+h]).
        - This is inherently forward-looking; see DATA_CONTRACT.md for usage rules.
        """
        # Clean and sort
        series = series.dropna().astype(float)
        series = series.sort_index()
        
        y = series.values
        idx = series.index
        n = len(y)
        h = params.h
        p = params.p
        
        # Minimum length required: need at least p lags (indices 0..p-1) and forward target
        # First valid i is p-1 (so we have y[p-1], y[p-2], ..., y[0] as p lags)
        # For target: need i + h < n, so i < n - h
        # Valid range: i in [p-1, n-h-1]
        
        if n < p + h:
            return pd.Series(dtype=float, name="hamilton_residual")
        
        # Build design matrix and target vector
        # For i in [p-1, n-h-1]:
        #   X row: [1 (intercept), y[i], y[i-1], ..., y[i-p+1]]
        #   Y: y[i+h]
        
        valid_start = p - 1
        valid_end = n - h - 1  # inclusive
        n_obs = valid_end - valid_start + 1
        
        if n_obs <= 0:
            return pd.Series(dtype=float, name="hamilton_residual")
        
        # Design matrix: intercept + p lags
        X = np.ones((n_obs, p + 1), dtype=np.float64)
        Y = np.zeros(n_obs, dtype=np.float64)
        target_indices = []
        
        for j, i in enumerate(range(valid_start, valid_end + 1)):
            # Lags: y[i], y[i-1], ..., y[i-p+1]
            for lag_idx in range(p):
                X[j, lag_idx + 1] = y[i - lag_idx]
            Y[j] = y[i + h]
            target_indices.append(i + h)
        
        # OLS via least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        
        # Predictions and residuals
        Y_pred = X @ coeffs
        residuals_out = Y - Y_pred
        
        # Index by target timestamps
        result_index = idx[target_indices]
        
        return pd.Series(
            residuals_out,
            index=result_index,
            name="hamilton_residual"
        )
