"""
Tests for src/conformal.py

Covers:
- Fit rejects NaN and shape mismatch
- Quantile order statistic definition (multi-output)
- Predict interval shapes and types preservation
- Alpha validation and fit-required checks
- Determinism (repeat calls identical)
"""

import numpy as np
import pandas as pd
import pytest

from src.conformal import SplitConformalRegressor


class TestFitValidation:
    """Test fit() input validation."""
    
    def test_fit_rejects_nan(self):
        """Test fit() raises ValueError on NaN in y_true."""
        scr = SplitConformalRegressor()
        
        y_true = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
        y_pred = np.array([[1.1, 1.9], [2.1, 3.1], [4.1, 5.1]])
        
        with pytest.raises(ValueError, match="NaN"):
            scr.fit(y_true, y_pred)
    
    def test_fit_rejects_inf(self):
        """Test fit() raises ValueError on inf in y_pred."""
        scr = SplitConformalRegressor()
        
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, np.inf], [3.1, 4.1], [5.1, 6.1]])
        
        with pytest.raises(ValueError, match="inf"):
            scr.fit(y_true, y_pred)
    
    def test_fit_rejects_shape_mismatch(self):
        """Test fit() raises ValueError on shape mismatch."""
        scr = SplitConformalRegressor()
        
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1]])  # (2, 2)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            scr.fit(y_true, y_pred)
    
    def test_fit_rejects_n_less_than_2(self):
        """Test fit() raises ValueError when n < 2."""
        scr = SplitConformalRegressor()
        
        y_true = np.array([[1.0, 2.0]])  # n=1
        y_pred = np.array([[1.1, 2.1]])
        
        with pytest.raises(ValueError, match="n must be >= 2"):
            scr.fit(y_true, y_pred)


class TestQuantileOrderStatistic:
    """Test quantile() computes correct order statistics."""
    
    def test_quantile_exact_k_index(self):
        """Test quantile uses correct k_index formula for multi-output."""
        scr = SplitConformalRegressor()
        
        # k=2 outputs, n=5 samples
        # Residuals: col 0 = [0.1, 0.2, 0.3, 0.4, 0.5], col 1 = [0.5, 0.4, 0.3, 0.2, 0.1]
        y_true = np.array([
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
        ])
        y_pred = np.array([
            [1.1, 4.5],  # residuals: [0.1, 0.5]
            [2.2, 3.6],  # residuals: [0.2, 0.4]
            [3.3, 2.7],  # residuals: [0.3, 0.3]
            [4.4, 1.8],  # residuals: [0.4, 0.2]
            [5.5, 0.9],  # residuals: [0.5, 0.1]
        ])
        
        scr.fit(y_true, y_pred)
        
        # For alpha=0.1 (90% coverage):
        # k_index = ceil((5+1) * (1-0.1)) = ceil(5.4) = 6, clamped to 5
        # So we want the 5th smallest (i.e., the max) residual
        qhat = scr.quantile(alpha=0.1)
        assert qhat.shape == (2,)
        assert np.isclose(qhat[0], 0.5)  # max of col 0
        assert np.isclose(qhat[1], 0.5)  # max of col 1
        
        # For alpha=0.5 (50% coverage):
        # k_index = ceil((5+1) * (1-0.5)) = ceil(3.0) = 3
        # So we want the 3rd smallest residual
        qhat50 = scr.quantile(alpha=0.5)
        assert np.isclose(qhat50[0], 0.3)  # 3rd smallest of [0.1,0.2,0.3,0.4,0.5]
        assert np.isclose(qhat50[1], 0.3)  # 3rd smallest of [0.1,0.2,0.3,0.4,0.5]
    
    def test_quantile_uses_partition_not_sort(self):
        """Test that quantile works correctly (order statistic computation)."""
        scr = SplitConformalRegressor()
        
        # Simple case: k=1, n=10
        np.random.seed(42)
        residuals_raw = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0])
        y_true = np.zeros((10, 1))
        y_pred = -residuals_raw.reshape(-1, 1)  # so residuals = |0 - (-r)| = r
        
        scr.fit(y_true, y_pred)
        
        # alpha=0.1: k_index = ceil(11*0.9) = 10
        qhat = scr.quantile(alpha=0.1)
        assert np.isclose(qhat[0], 1.0)  # max
        
        # alpha=0.2: k_index = ceil(11*0.8) = 9
        sorted_residuals = np.sort(residuals_raw)  # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        qhat8 = scr.quantile(alpha=0.2)
        assert np.isclose(qhat8[0], sorted_residuals[8])  # 9th smallest (0-indexed: 8)


class TestPredictIntervalShapesAndTypes:
    """Test predict_interval preserves shapes and types."""
    
    @pytest.fixture
    def fitted_scr(self):
        """Create a fitted SplitConformalRegressor."""
        scr = SplitConformalRegressor()
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        scr.fit(y_true, y_pred)
        return scr
    
    def test_predict_interval_numpy_2d(self, fitted_scr):
        """Test predict_interval with 2D numpy array."""
        y_new = np.array([[10.0, 20.0], [30.0, 40.0]])
        lower, upper = fitted_scr.predict_interval(y_new, alpha=0.1)
        
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert lower.shape == (2, 2)
        assert upper.shape == (2, 2)
        assert np.all(lower < upper)
    
    def test_predict_interval_numpy_1d(self, fitted_scr):
        """Test predict_interval with 1D numpy array (k=1 case)."""
        # Need a k=1 fitted model
        scr = SplitConformalRegressor()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        scr.fit(y_true, y_pred)
        
        y_new = np.array([10.0, 20.0])
        lower, upper = scr.predict_interval(y_new, alpha=0.1)
        
        assert isinstance(lower, np.ndarray)
        assert lower.ndim == 1
        assert lower.shape == (2,)
    
    def test_predict_interval_dataframe(self, fitted_scr):
        """Test predict_interval preserves DataFrame type."""
        y_new = pd.DataFrame(
            [[10.0, 20.0], [30.0, 40.0]],
            index=["a", "b"],
            columns=["col1", "col2"]
        )
        lower, upper = fitted_scr.predict_interval(y_new, alpha=0.1)
        
        assert isinstance(lower, pd.DataFrame)
        assert isinstance(upper, pd.DataFrame)
        assert list(lower.index) == ["a", "b"]
        assert list(lower.columns) == ["col1", "col2"]
    
    def test_predict_interval_series(self):
        """Test predict_interval with Series input (k=1)."""
        scr = SplitConformalRegressor()
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 2.1, 3.1])
        scr.fit(y_true, y_pred)
        
        y_new = pd.Series([10.0, 20.0], index=["x", "y"])
        lower, upper = scr.predict_interval(y_new, alpha=0.1)
        
        assert isinstance(lower, pd.Series)
        assert isinstance(upper, pd.Series)
        assert list(lower.index) == ["x", "y"]


class TestAlphaValidationAndFitRequired:
    """Test alpha validation and fit-required checks."""
    
    def test_quantile_requires_fit(self):
        """Test quantile() raises RuntimeError if not fitted."""
        scr = SplitConformalRegressor()
        
        with pytest.raises(RuntimeError, match="fit"):
            scr.quantile(alpha=0.1)
    
    def test_predict_interval_requires_fit(self):
        """Test predict_interval() raises RuntimeError if not fitted."""
        scr = SplitConformalRegressor()
        
        with pytest.raises(RuntimeError, match="fit"):
            scr.predict_interval(np.array([1.0]), alpha=0.1)
    
    def test_invalid_alpha_zero(self):
        """Test alpha=0 raises ValueError."""
        scr = SplitConformalRegressor()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.1], [3.1]])
        scr.fit(y_true, y_pred)
        
        with pytest.raises(ValueError, match="alpha"):
            scr.quantile(alpha=0.0)
    
    def test_invalid_alpha_one(self):
        """Test alpha=1 raises ValueError."""
        scr = SplitConformalRegressor()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.1], [3.1]])
        scr.fit(y_true, y_pred)
        
        with pytest.raises(ValueError, match="alpha"):
            scr.quantile(alpha=1.0)
    
    def test_invalid_alpha_negative(self):
        """Test alpha<0 raises ValueError."""
        scr = SplitConformalRegressor()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.1], [3.1]])
        scr.fit(y_true, y_pred)
        
        with pytest.raises(ValueError, match="alpha"):
            scr.quantile(alpha=-0.1)
    
    def test_k_dimension_mismatch(self):
        """Test predict_interval rejects k dimension mismatch."""
        scr = SplitConformalRegressor()
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # k=2
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        scr.fit(y_true, y_pred)
        
        y_new = np.array([[10.0]])  # k=1
        
        with pytest.raises(ValueError, match="k dimension mismatch"):
            scr.predict_interval(y_new, alpha=0.1)


class TestDeterminism:
    """Test determinism of outputs."""
    
    def test_repeat_fit_identical(self):
        """Test repeated fit() calls produce identical residuals."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        
        scr1 = SplitConformalRegressor()
        scr1.fit(y_true, y_pred)
        
        scr2 = SplitConformalRegressor()
        scr2.fit(y_true, y_pred)
        
        np.testing.assert_array_equal(scr1.residuals_, scr2.residuals_)
    
    def test_repeat_quantile_identical(self):
        """Test repeated quantile() calls produce identical results."""
        scr = SplitConformalRegressor()
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])
        scr.fit(y_true, y_pred)
        
        qhat1 = scr.quantile(alpha=0.2)
        qhat2 = scr.quantile(alpha=0.2)
        
        np.testing.assert_array_equal(qhat1, qhat2)
    
    def test_repeat_predict_interval_identical(self):
        """Test repeated predict_interval() calls produce identical results."""
        scr = SplitConformalRegressor()
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])
        scr.fit(y_true, y_pred)
        
        y_new = np.array([[10.0], [20.0]])
        
        lower1, upper1 = scr.predict_interval(y_new, alpha=0.2)
        lower2, upper2 = scr.predict_interval(y_new, alpha=0.2)
        
        np.testing.assert_array_equal(lower1, lower2)
        np.testing.assert_array_equal(upper1, upper2)
