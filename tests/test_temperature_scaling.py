"""
Tests for src/temperature_scaling.py

Covers:
- Fit rejects invalid inputs (NaN, inf, non-binary, n<2, invalid input_type)
- Temperature grid search determinism
- Transform type preservation (Series, DataFrame, numpy)
- Temperature scaling improves or matches NLL on calibration set
- Proba input path works correctly
"""

import numpy as np
import pandas as pd
import pytest

from src.temperature_scaling import TemperatureScalerBinary, _stable_sigmoid


class TestFitRejectsInvalidInputs:
    """Test fit() input validation."""
    
    def test_rejects_nan_in_scores(self):
        """Test fit() raises ValueError on NaN in scores."""
        ts = TemperatureScalerBinary()
        
        scores = np.array([1.0, np.nan, 2.0])
        y_true = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="NaN"):
            ts.fit(scores, y_true)
    
    def test_rejects_inf_in_scores(self):
        """Test fit() raises ValueError on inf in scores."""
        ts = TemperatureScalerBinary()
        
        scores = np.array([1.0, np.inf, 2.0])
        y_true = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="inf"):
            ts.fit(scores, y_true)
    
    def test_rejects_non_binary_labels(self):
        """Test fit() raises ValueError when y_true is not binary."""
        ts = TemperatureScalerBinary()
        
        scores = np.array([1.0, 2.0, 3.0])
        y_true = np.array([0, 1, 2])  # Not binary!
        
        with pytest.raises(ValueError, match="binary"):
            ts.fit(scores, y_true)
    
    def test_rejects_n_less_than_2(self):
        """Test fit() raises ValueError when n < 2."""
        ts = TemperatureScalerBinary()
        
        scores = np.array([1.0])
        y_true = np.array([1])
        
        with pytest.raises(ValueError, match="n must be >= 2"):
            ts.fit(scores, y_true)
    
    def test_rejects_invalid_input_type(self):
        """Test fit() raises ValueError for invalid input_type."""
        ts = TemperatureScalerBinary()
        
        scores = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="input_type"):
            ts.fit(scores, y_true, input_type="invalid")
    
    def test_rejects_proba_out_of_range(self):
        """Test fit() raises ValueError when proba scores are out of [0,1]."""
        ts = TemperatureScalerBinary()
        
        scores = np.array([0.5, 1.5, 0.3])  # 1.5 is out of range
        y_true = np.array([1, 0, 1])
        
        with pytest.raises(ValueError, match="\\[0,1\\]"):
            ts.fit(scores, y_true, input_type="proba")


class TestTemperatureGridSearchDeterminism:
    """Test grid search is deterministic."""
    
    def test_fit_twice_identical_temperature(self):
        """Test fitting twice produces identical temperature."""
        scores = np.array([0.5, 1.5, -0.5, 2.0, -1.0])
        y_true = np.array([1, 1, 0, 1, 0])
        
        ts1 = TemperatureScalerBinary()
        ts1.fit(scores, y_true)
        
        ts2 = TemperatureScalerBinary()
        ts2.fit(scores, y_true)
        
        assert ts1.temperature_ == ts2.temperature_
    
    def test_transform_twice_identical(self):
        """Test transforming twice produces identical results."""
        scores_cal = np.array([0.5, 1.5, -0.5, 2.0, -1.0])
        y_cal = np.array([1, 1, 0, 1, 0])
        
        ts = TemperatureScalerBinary()
        ts.fit(scores_cal, y_cal)
        
        scores_test = np.array([1.0, -1.0, 0.5])
        
        proba1 = ts.transform(scores_test)
        proba2 = ts.transform(scores_test)
        
        np.testing.assert_array_equal(proba1, proba2)


class TestTransformTypePreservation:
    """Test transform() preserves input types."""
    
    @pytest.fixture
    def fitted_ts(self):
        """Create a fitted TemperatureScalerBinary."""
        ts = TemperatureScalerBinary()
        scores = np.array([0.5, 1.5, -0.5, 2.0, -1.0])
        y_true = np.array([1, 1, 0, 1, 0])
        ts.fit(scores, y_true)
        return ts
    
    def test_series_in_series_out(self, fitted_ts):
        """Test Series input returns Series output with same index."""
        scores = pd.Series([1.0, -1.0, 0.5], index=["a", "b", "c"])
        
        result = fitted_ts.transform(scores)
        
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b", "c"]
    
    def test_dataframe_in_dataframe_out(self, fitted_ts):
        """Test DataFrame input returns DataFrame output with same index/columns."""
        scores = pd.DataFrame(
            [[1.0], [-1.0], [0.5]],
            index=["x", "y", "z"],
            columns=["logit"]
        )
        
        result = fitted_ts.transform(scores)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["x", "y", "z"]
        assert list(result.columns) == ["logit"]
    
    def test_numpy_in_numpy_out(self, fitted_ts):
        """Test numpy input returns numpy output with same shape."""
        scores = np.array([1.0, -1.0, 0.5])
        
        result = fitted_ts.transform(scores)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
    
    def test_numpy_2d_preserved(self, fitted_ts):
        """Test 2D numpy input returns 2D output."""
        scores = np.array([[1.0], [-1.0], [0.5]])
        
        result = fitted_ts.transform(scores)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)


class TestTemperatureScalingImprovesNLL:
    """Test that temperature scaling improves or matches NLL."""
    
    def test_scaling_improves_overconfident_predictions(self):
        """Test that T > 1 helps overconfident wrong predictions."""
        # Create overconfident logits with some mislabeled
        # Very high logits but label is 0 -> overconfident and wrong
        logits = np.array([
            3.0,   # predicts 1 strongly -> label 1 (correct)
            -3.0,  # predicts 0 strongly -> label 0 (correct)
            3.0,   # predicts 1 strongly -> label 0 (WRONG, overconfident)
            -3.0,  # predicts 0 strongly -> label 1 (WRONG, overconfident)
            0.5,   # predicts 1 weakly -> label 1 (correct)
            -0.5,  # predicts 0 weakly -> label 0 (correct)
        ])
        y = np.array([1, 0, 0, 1, 1, 0])
        
        # Compute NLL at T=1 (baseline)
        p_t1 = _stable_sigmoid(logits)
        p_t1_clipped = np.clip(p_t1, 1e-12, 1 - 1e-12)
        nll_t1 = -np.mean(y * np.log(p_t1_clipped) + (1 - y) * np.log(1 - p_t1_clipped))
        
        # Fit temperature scaler
        ts = TemperatureScalerBinary()
        ts.fit(logits, y)
        
        # Compute NLL with best T
        p_best = ts.transform(logits)
        p_best_clipped = np.clip(p_best, 1e-12, 1 - 1e-12)
        nll_best = -np.mean(y * np.log(p_best_clipped) + (1 - y) * np.log(1 - p_best_clipped))
        
        # Best T should have NLL <= T=1 NLL
        assert nll_best <= nll_t1 + 1e-9, f"NLL best ({nll_best}) > NLL T=1 ({nll_t1})"
    
    def test_temperature_is_within_grid_bounds(self):
        """Test that found temperature is within the search grid."""
        logits = np.array([0.5, 1.5, -0.5, 2.0, -1.0, 0.0])
        y = np.array([1, 1, 0, 1, 0, 1])
        
        ts = TemperatureScalerBinary(t_min=0.1, t_max=5.0)
        ts.fit(logits, y)
        
        # Temperature should be within grid bounds
        assert 0.1 <= ts.temperature_ <= 5.0, f"T={ts.temperature_} not in [0.1, 5.0]"


class TestProbaInputPath:
    """Test input_type='proba' path."""
    
    def test_fit_with_proba_input(self):
        """Test fitting with probability inputs."""
        # Probabilities in (0, 1)
        proba = np.array([0.9, 0.1, 0.7, 0.3, 0.8])
        y_true = np.array([1, 0, 1, 0, 1])
        
        ts = TemperatureScalerBinary()
        ts.fit(proba, y_true, input_type="proba")
        
        assert ts.fitted_
        assert ts.temperature_ > 0
    
    def test_transform_with_proba_input(self):
        """Test transforming with probability inputs."""
        proba_cal = np.array([0.9, 0.1, 0.7, 0.3, 0.8])
        y_cal = np.array([1, 0, 1, 0, 1])
        
        ts = TemperatureScalerBinary()
        ts.fit(proba_cal, y_cal, input_type="proba")
        
        proba_test = np.array([0.6, 0.4, 0.8])
        result = ts.transform(proba_test, input_type="proba")
        
        # Result should be probabilities in (0, 1)
        assert np.all(result > 0)
        assert np.all(result < 1)
        assert np.all(np.isfinite(result))
    
    def test_predict_proba_alias(self):
        """Test predict_proba is an alias for transform."""
        proba_cal = np.array([0.9, 0.1, 0.7, 0.3, 0.8])
        y_cal = np.array([1, 0, 1, 0, 1])
        
        ts = TemperatureScalerBinary()
        ts.fit(proba_cal, y_cal, input_type="proba")
        
        proba_test = np.array([0.6, 0.4, 0.8])
        
        result_transform = ts.transform(proba_test, input_type="proba")
        result_predict = ts.predict_proba(proba_test, input_type="proba")
        
        np.testing.assert_array_equal(result_transform, result_predict)


class TestConstructorValidation:
    """Test constructor parameter validation."""
    
    def test_grid_size_too_small(self):
        """Test grid_size < 10 raises ValueError."""
        with pytest.raises(ValueError, match="grid_size"):
            TemperatureScalerBinary(grid_size=5)
    
    def test_invalid_t_min_t_max(self):
        """Test invalid t_min >= t_max raises ValueError."""
        with pytest.raises(ValueError, match="t_min"):
            TemperatureScalerBinary(t_min=2.0, t_max=1.0)
    
    def test_eps_out_of_range(self):
        """Test eps outside (0, 1e-3) raises ValueError."""
        with pytest.raises(ValueError, match="eps"):
            TemperatureScalerBinary(eps=0.01)


class TestTransformRequiresFit:
    """Test transform/predict_proba require fit."""
    
    def test_transform_before_fit_raises(self):
        """Test transform() raises RuntimeError if not fitted."""
        ts = TemperatureScalerBinary()
        
        with pytest.raises(RuntimeError, match="fit"):
            ts.transform(np.array([1.0, 2.0]))
    
    def test_predict_proba_before_fit_raises(self):
        """Test predict_proba() raises RuntimeError if not fitted."""
        ts = TemperatureScalerBinary()
        
        with pytest.raises(RuntimeError, match="fit"):
            ts.predict_proba(np.array([1.0, 2.0]))
