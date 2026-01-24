"""
Regime module for QUANT-NEURAL.

Provides:
- RegimeParams: Configuration for regime detection.
- RegimeDetector: LogisticRegression-based regime classifier with action rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class RegimeParams:
    """Parameters for regime detection.
    
    Attributes
    ----------
    threshold : float
        Probability threshold for action decision. Default 0.60.
    C : float
        Inverse regularization strength for LogisticRegression.
    max_iter : int
        Maximum iterations for solver.
    class_weight : Optional[str]
        Class weight strategy (e.g., "balanced" or None).
    solver : str
        Solver algorithm for LogisticRegression. Default "liblinear" for
        deterministic binary classification support.
    random_state : int
        Random seed for reproducibility. Default 42.
    """
    threshold: float = 0.60
    C: float = 1.0
    max_iter: int = 2000
    class_weight: Optional[str] = None
    solver: str = "liblinear"
    random_state: int = 42


class RegimeDetector:
    """
    Logistic Regression regime detector.
    
    Classifies market regimes and provides probability estimates.
    
    Parameters
    ----------
    params : RegimeParams
        Configuration parameters.
    
    Notes
    -----
    - Assumes binary labels {0, 1} where 1 = "up" regime.
    - Calibration is NOT implemented here (Phase 5 only).
    """
    
    def __init__(self, params: RegimeParams):
        self.params = params
        self.model = LogisticRegression(
            C=params.C,
            max_iter=params.max_iter,
            class_weight=params.class_weight,
            solver=params.solver,
            random_state=params.random_state
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit logistic regression model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features).
        y_train : np.ndarray
            Training labels, shape (n_samples,). Binary {0, 1}.
        """
        self.model.fit(X_train, y_train)
    
    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of "up" regime (y=1).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Probability of y=1 for each sample, shape (n_samples,).
        """
        return self.model.predict_proba(X)[:, 1]
    
    def brier(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Brier score (mean squared error of probability estimates).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_true : np.ndarray
            True binary labels.
        
        Returns
        -------
        float
            Brier score. Lower is better.
        """
        p_up = self.predict_proba_up(X)
        return float(brier_score_loss(y_true, p_up))
    
    def action(self, p_up: float) -> str:
        """
        Determine action based on probability threshold.
        
        Parameters
        ----------
        p_up : float
            Probability of "up" regime.
        
        Returns
        -------
        str
            "AGGRESSIVE" if p_up >= threshold, else "DEFENSIVE".
        """
        if p_up >= self.params.threshold:
            return "AGGRESSIVE"
        else:
            return "DEFENSIVE"
