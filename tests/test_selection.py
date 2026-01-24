"""
Tests for src/selection.py

Covers:
- Feature name length mismatch raises ValueError
- Synthetic selection chooses true signal features (f0, f2)
"""

import numpy as np
import pytest

from src.selection import LassoParams, ModelSelector


class TestModelSelector:
    """Test ModelSelector class."""
    
    def test_feature_name_mismatch_raises(self):
        """Test that feature_names length mismatch raises ValueError."""
        selector = ModelSelector(LassoParams())
        
        # X has 3 features, but feature_names has 2
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        feature_names = ["f0", "f1"]  # Missing one
        
        with pytest.raises(ValueError, match="feature_names length"):
            selector.select_features_lasso(X, y, feature_names)
    
    def test_synthetic_selection_finds_signal_features(self):
        """Test that LassoCV selects true signal features on synthetic data."""
        # Deterministic seed
        np.random.seed(0)
        
        # Create time-ordered synthetic data (do not shuffle)
        n = 200
        n_features = 5
        
        # Generate features
        X = np.random.randn(n, n_features)
        
        # y = 3*X[:,0] - 2*X[:,2] + small noise
        # Strong signal to ensure stable selection
        noise = 0.01 * np.random.randn(n)
        y = 3.0 * X[:, 0] - 2.0 * X[:, 2] + noise
        
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        
        # Run feature selection
        selector = ModelSelector(LassoParams(n_splits=5, random_state=42))
        selected = selector.select_features_lasso(X, y, feature_names)
        
        # Assert f0 and f2 are in selected (they have the signal)
        assert "f0" in selected, f"Expected f0 in selected, got {selected}"
        assert "f2" in selected, f"Expected f2 in selected, got {selected}"
        
        # Assert selected is subset of feature_names
        assert set(selected).issubset(set(feature_names))
        
        # Assert model is fitted
        assert selector.model is not None
    
    def test_model_stored_after_fit(self):
        """Test that model attribute is set after selection."""
        np.random.seed(42)
        
        X = np.random.randn(100, 3)
        y = X[:, 0] + 0.1 * np.random.randn(100)
        feature_names = ["a", "b", "c"]
        
        selector = ModelSelector(LassoParams())
        
        # Before fitting
        assert selector.model is None
        
        # After fitting
        selector.select_features_lasso(X, y, feature_names)
        assert selector.model is not None
    
    def test_returns_list_of_strings(self):
        """Test that return type is List[str]."""
        np.random.seed(123)
        
        X = np.random.randn(100, 4)
        y = 2.0 * X[:, 1] + 0.01 * np.random.randn(100)
        feature_names = ["x1", "x2", "x3", "x4"]
        
        selector = ModelSelector(LassoParams())
        selected = selector.select_features_lasso(X, y, feature_names)
        
        assert isinstance(selected, list)
        assert all(isinstance(s, str) for s in selected)
