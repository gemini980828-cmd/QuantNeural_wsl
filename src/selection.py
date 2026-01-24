"""
Selection module for QUANT-NEURAL.

Provides:
- LassoParams: Configuration for Lasso feature selection.
- ModelSelector: Sparse feature selection using LassoCV + TimeSeriesSplit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class LassoParams:
    """Parameters for Lasso feature selection.
    
    Attributes
    ----------
    n_splits : int
        Number of TimeSeriesSplit folds for cross-validation.
    random_state : int
        Random state for reproducibility.
    max_iter : int
        Maximum iterations for LassoCV solver.
    """
    n_splits: int = 5
    random_state: int = 42
    max_iter: int = 20000


class ModelSelector:
    """
    Sparse feature selection using LassoCV with TimeSeriesSplit.
    
    Parameters
    ----------
    params : LassoParams
        Configuration parameters.
    
    Attributes
    ----------
    model : LassoCV | None
        Trained LassoCV model after calling select_features_lasso.
    """
    
    def __init__(self, params: LassoParams):
        self.params = params
        self.model: LassoCV | None = None
    
    def select_features_lasso(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """
        Select features using LassoCV with TimeSeriesSplit cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Target vector, shape (n_samples,).
        feature_names : List[str]
            Names of features. Length must equal n_features.
        
        Returns
        -------
        List[str]
            Names of selected features (those with non-zero coefficients).
        
        Raises
        ------
        ValueError
            If len(feature_names) != X.shape[1].
        
        Notes
        -----
        - Data is assumed to be time-ordered. DO NOT shuffle before calling.
        - Uses TimeSeriesSplit for cross-validation (no data leakage).
        - Selected features have abs(coef_) > 0.
        """
        # Validate feature names length
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must equal "
                f"number of features in X ({X.shape[1]})"
            )
        
        # Create TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=self.params.n_splits)
        
        # Create and fit LassoCV
        self.model = LassoCV(
            cv=tscv,
            random_state=self.params.random_state,
            max_iter=self.params.max_iter,
            n_jobs=None
        )
        self.model.fit(X, y)
        
        # Select features with non-zero coefficients
        selected_indices = np.where(np.abs(self.model.coef_) > 0)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        return selected_features
