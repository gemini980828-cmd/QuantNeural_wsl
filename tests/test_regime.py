"""
Tests for src/regime.py

Covers:
- predict_proba_up shape and range [0, 1]
- brier() returns finite float
- action() threshold behavior (0.60 boundary)
"""

import numpy as np
import pytest

from src.regime import RegimeParams, RegimeDetector


class TestRegimeDetector:
    """Test RegimeDetector class."""
    
    @pytest.fixture
    def fitted_detector(self):
        """Create and fit a RegimeDetector on synthetic data."""
        np.random.seed(42)
        
        # Create synthetic binary classification data
        n = 300
        X = np.random.randn(n, 3)
        
        # logits = 2*X[:,0] - 1*X[:,1]
        logits = 2.0 * X[:, 0] - 1.0 * X[:, 1]
        y = (logits > 0).astype(int)
        
        # Split by time order: first 200 train, last 100 test
        X_train, y_train = X[:200], y[:200]
        X_test, y_test = X[200:], y[200:]
        
        detector = RegimeDetector(RegimeParams())
        detector.fit(X_train, y_train)
        
        return detector, X_test, y_test
    
    def test_predict_proba_up_shape(self, fitted_detector):
        """Test that predict_proba_up returns correct shape."""
        detector, X_test, _ = fitted_detector
        
        p = detector.predict_proba_up(X_test)
        
        assert p.shape == (len(X_test),), f"Expected shape ({len(X_test)},), got {p.shape}"
    
    def test_predict_proba_up_range(self, fitted_detector):
        """Test that probabilities are within [0, 1] and finite."""
        detector, X_test, _ = fitted_detector
        
        p = detector.predict_proba_up(X_test)
        
        assert np.all(np.isfinite(p)), "Probabilities must be finite"
        assert np.all(p >= 0), "Probabilities must be >= 0"
        assert np.all(p <= 1), "Probabilities must be <= 1"
    
    def test_brier_is_finite_float(self, fitted_detector):
        """Test that brier() returns a finite float."""
        detector, X_test, y_test = fitted_detector
        
        b = detector.brier(X_test, y_test)
        
        assert isinstance(b, float), f"Expected float, got {type(b)}"
        assert np.isfinite(b), "Brier score must be finite"
    
    def test_brier_reasonable_range(self, fitted_detector):
        """Test that Brier score is in reasonable range [0, 1]."""
        detector, X_test, y_test = fitted_detector
        
        b = detector.brier(X_test, y_test)
        
        # Brier score is MSE of probabilities, should be in [0, 1]
        assert 0 <= b <= 1, f"Brier score {b} outside [0, 1]"
    
    def test_action_aggressive_at_threshold(self):
        """Test that action returns AGGRESSIVE at exactly threshold."""
        detector = RegimeDetector(RegimeParams(threshold=0.60))
        
        assert detector.action(0.60) == "AGGRESSIVE"
    
    def test_action_defensive_below_threshold(self):
        """Test that action returns DEFENSIVE below threshold."""
        detector = RegimeDetector(RegimeParams(threshold=0.60))
        
        assert detector.action(0.5999) == "DEFENSIVE"
    
    def test_action_aggressive_above_threshold(self):
        """Test that action returns AGGRESSIVE above threshold."""
        detector = RegimeDetector(RegimeParams(threshold=0.60))
        
        assert detector.action(0.75) == "AGGRESSIVE"
    
    def test_action_defensive_at_zero(self):
        """Test that action returns DEFENSIVE at 0."""
        detector = RegimeDetector(RegimeParams(threshold=0.60))
        
        assert detector.action(0.0) == "DEFENSIVE"
    
    def test_action_aggressive_at_one(self):
        """Test that action returns AGGRESSIVE at 1."""
        detector = RegimeDetector(RegimeParams(threshold=0.60))
        
        assert detector.action(1.0) == "AGGRESSIVE"
    
    def test_custom_threshold(self):
        """Test action with custom threshold."""
        detector = RegimeDetector(RegimeParams(threshold=0.50))
        
        assert detector.action(0.50) == "AGGRESSIVE"
        assert detector.action(0.499) == "DEFENSIVE"
    
    def test_regime_deterministic_fit_same_seed_same_outputs(self):
        """Test that same seed produces identical outputs across independent fits."""
        # Build deterministic synthetic dataset with fixed seed
        rng = np.random.RandomState(999)
        n = 200
        X = rng.randn(n, 4)
        logits = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2]
        y = (logits > 0).astype(int)
        
        # Split: first 150 train, last 50 test (time-ordered)
        X_train, y_train = X[:150], y[:150]
        X_test = X[150:]
        
        # Fit two independent RegimeDetector instances with same params
        params = RegimeParams(random_state=123, solver="liblinear")
        
        detector1 = RegimeDetector(params)
        detector1.fit(X_train, y_train)
        proba1 = detector1.predict_proba_up(X_test)
        
        detector2 = RegimeDetector(params)
        detector2.fit(X_train, y_train)
        proba2 = detector2.predict_proba_up(X_test)
        
        # Assert probabilities are exactly identical
        np.testing.assert_array_equal(
            proba1, proba2,
            err_msg="Probabilities must be identical with same random_state"
        )
        
        # Assert regime classifications at threshold=0.60 are identical
        threshold = params.threshold
        actions1 = [detector1.action(p) for p in proba1]
        actions2 = [detector2.action(p) for p in proba2]
        assert actions1 == actions2, "Actions must be identical with same seed"
    
    def test_regime_outputs_finite_and_valid_range(self):
        """Test that outputs are in [0,1] and finite after deterministic fit."""
        rng = np.random.RandomState(777)
        n = 100
        X = rng.randn(n, 3)
        y = (X[:, 0] > 0).astype(int)
        
        params = RegimeParams(random_state=42, solver="liblinear")
        detector = RegimeDetector(params)
        detector.fit(X[:80], y[:80])
        proba = detector.predict_proba_up(X[80:])
        
        # All outputs must be finite and in [0, 1]
        assert np.all(np.isfinite(proba)), "Probabilities must be finite"
        assert np.all((proba >= 0) & (proba <= 1)), "Probabilities must be in [0, 1]"
        
        # Brier score must be finite
        brier = detector.brier(X[80:], y[80:])
        assert np.isfinite(brier), "Brier score must be finite"

