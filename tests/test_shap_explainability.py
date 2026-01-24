"""
Tests for src/shap_explainability.py

Covers:
- Linear model exact single-output Shapley values
- Linear model exact multi-output Shapley values
- Background mean baseline path
- Fail-fast invalid inputs
- Determinism (same seed = same result, different seed = different result)
"""

import numpy as np
import pandas as pd
import pytest

from src.shap_explainability import shapley_sampling_values, global_feature_importance


class TestLinearModelExactSingleOutput:
    """Test exact Shapley values for linear single-output model."""
    
    def test_linear_single_output_exact(self):
        """For linear model with baseline=0, Shapley = w * x_i exactly."""
        d = 4
        w = np.array([1.0, -2.0, 3.0, 0.5])
        b = 0.0
        
        def predict_fn(x):
            return x @ w + b
        
        x = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
        ])
        
        baseline = np.zeros(d)
        
        # With n_permutations=1, for linear model: Shapley = w * x_i
        # (because for linear functions, marginal contributions are additive)
        shap_values = shapley_sampling_values(
            predict_fn, x, baseline=baseline, n_permutations=1, seed=0
        )
        
        expected = x * w  # (n, d)
        np.testing.assert_allclose(shap_values, expected, atol=1e-10)
    
    def test_linear_many_permutations_converges(self):
        """Linear model: many permutations should also give exact Shapley."""
        d = 3
        w = np.array([2.0, -1.0, 0.5])
        
        def predict_fn(x):
            return x @ w
        
        x = np.array([[1.0, 2.0, 3.0]])
        baseline = np.zeros(d)
        
        # Any number of permutations should give same result for linear
        shap_values = shapley_sampling_values(
            predict_fn, x, baseline=baseline, n_permutations=64, seed=42
        )
        
        expected = x * w
        np.testing.assert_allclose(shap_values, expected, atol=1e-10)


class TestLinearModelExactMultiOutput:
    """Test exact Shapley values for linear multi-output model."""
    
    def test_linear_multi_output_exact(self):
        """For linear multi-output model, Shapley = x_i[:,None] * W."""
        d, k = 3, 2
        W = np.array([
            [1.0, 2.0],
            [-1.0, 0.5],
            [0.3, -0.3],
        ])  # (d, k)
        
        def predict_fn(x):
            return x @ W  # (n, k)
        
        x = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
        ])  # (n, d)
        
        baseline = np.zeros(d)
        
        shap_values = shapley_sampling_values(
            predict_fn, x, baseline=baseline, n_permutations=1, seed=0
        )
        
        # Expected: for each sample i, shap[i, j, :] = x[i, j] * W[j, :]
        expected = np.zeros((2, d, k))
        for i in range(2):
            for j in range(d):
                expected[i, j, :] = x[i, j] * W[j, :]
        
        assert shap_values.shape == (2, d, k)
        np.testing.assert_allclose(shap_values, expected, atol=1e-10)


class TestBackgroundMeanBaselinePath:
    """Test background dataset derives baseline via mean."""
    
    def test_background_mean_used_as_baseline(self):
        """When baseline=None and background provided, baseline=mean(background)."""
        d = 3
        w = np.array([1.0, 2.0, 3.0])
        
        def predict_fn(x):
            return x @ w
        
        # Background with mean = [2.0, 3.0, 4.0]
        background = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ])
        expected_baseline = np.array([2.0, 3.0, 4.0])
        
        # If x equals the baseline, shap values should be all zeros
        x = expected_baseline.reshape(1, -1)
        
        shap_values = shapley_sampling_values(
            predict_fn, x, background=background, n_permutations=1, seed=0
        )
        
        np.testing.assert_allclose(shap_values, np.zeros((1, d)), atol=1e-10)
    
    def test_baseline_wins_over_background(self):
        """When both baseline and background provided, baseline wins."""
        d = 2
        w = np.array([1.0, 1.0])
        
        def predict_fn(x):
            return x @ w
        
        background = np.array([[5.0, 5.0], [7.0, 7.0]])  # mean = [6, 6]
        explicit_baseline = np.zeros(d)  # Use zeros instead
        
        x = np.array([[2.0, 3.0]])
        
        shap_values = shapley_sampling_values(
            predict_fn, x, baseline=explicit_baseline, background=background,
            n_permutations=1, seed=0
        )
        
        # With baseline=0: shap = w * x = [2, 3]
        expected = x * w
        np.testing.assert_allclose(shap_values, expected, atol=1e-10)


class TestFailFastInvalidInputs:
    """Test fail-fast validation."""
    
    def test_nan_in_x_raises(self):
        """NaN in x raises ValueError."""
        def predict_fn(x):
            return x.sum(axis=1)
        
        x = np.array([[1.0, np.nan], [3.0, 4.0]])
        
        with pytest.raises(ValueError, match="NaN"):
            shapley_sampling_values(predict_fn, x)
    
    def test_inf_in_x_raises(self):
        """Inf in x raises ValueError."""
        def predict_fn(x):
            return x.sum(axis=1)
        
        x = np.array([[1.0, np.inf], [3.0, 4.0]])
        
        with pytest.raises(ValueError, match="inf"):
            shapley_sampling_values(predict_fn, x)
    
    def test_baseline_wrong_shape_raises(self):
        """Baseline with wrong shape raises ValueError."""
        def predict_fn(x):
            return x.sum(axis=1)
        
        x = np.array([[1.0, 2.0], [3.0, 4.0]])  # d=2
        baseline = np.array([1.0, 2.0, 3.0])  # d=3, wrong!
        
        with pytest.raises(ValueError, match="baseline"):
            shapley_sampling_values(predict_fn, x, baseline=baseline)
    
    def test_n_permutations_zero_raises(self):
        """n_permutations=0 raises ValueError."""
        def predict_fn(x):
            return x.sum(axis=1)
        
        x = np.array([[1.0, 2.0]])
        
        with pytest.raises(ValueError, match="n_permutations"):
            shapley_sampling_values(predict_fn, x, n_permutations=0)
    
    def test_predict_fn_wrong_n_raises(self):
        """predict_fn returning wrong n raises ValueError."""
        def bad_predict_fn(x):
            # Returns wrong shape
            return np.array([1.0, 2.0, 3.0])  # n=3 regardless of input
        
        x = np.array([[1.0, 2.0]])  # n=1
        
        with pytest.raises(ValueError, match="n"):
            shapley_sampling_values(bad_predict_fn, x)


class TestDeterminism:
    """Test determinism with seeds."""
    
    def test_same_seed_same_result(self):
        """Same seed produces identical results."""
        d = 4
        w = np.array([1.0, -0.5, 2.0, -1.5])
        
        def predict_fn(x):
            # Non-linear: sigmoid
            return 1.0 / (1.0 + np.exp(-x @ w))
        
        x = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
        
        shap1 = shapley_sampling_values(predict_fn, x, n_permutations=16, seed=42)
        shap2 = shapley_sampling_values(predict_fn, x, n_permutations=16, seed=42)
        
        np.testing.assert_array_equal(shap1, shap2)
    
    def test_different_seed_different_result(self):
        """Different seeds produce different results for non-linear model."""
        d = 4
        w = np.array([1.0, -0.5, 2.0, -1.5])
        
        def predict_fn(x):
            # Non-linear: sigmoid
            return 1.0 / (1.0 + np.exp(-x @ w))
        
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        
        shap1 = shapley_sampling_values(predict_fn, x, n_permutations=4, seed=0)
        shap2 = shapley_sampling_values(predict_fn, x, n_permutations=4, seed=123)
        
        # At least one element should differ
        assert not np.allclose(shap1, shap2), "Different seeds should produce different values"


class TestGlobalFeatureImportance:
    """Test global_feature_importance aggregation."""
    
    def test_mean_abs_aggregation(self):
        """Test mean_abs aggregation for single output."""
        shap_values = np.array([
            [1.0, -2.0, 3.0],
            [-1.0, 2.0, -3.0],
        ])  # (n=2, d=3)
        
        importance = global_feature_importance(shap_values, agg="mean_abs")
        
        expected = np.array([1.0, 2.0, 3.0])  # mean of abs
        np.testing.assert_allclose(importance, expected)
    
    def test_mean_aggregation(self):
        """Test mean aggregation (signed)."""
        shap_values = np.array([
            [1.0, -2.0, 3.0],
            [3.0, 2.0, -1.0],
        ])
        
        importance = global_feature_importance(shap_values, agg="mean")
        
        expected = np.array([2.0, 0.0, 1.0])  # mean (signed)
        np.testing.assert_allclose(importance, expected)
    
    def test_multi_output_importance(self):
        """Test importance for multi-output (n, d, k)."""
        shap_values = np.array([
            [[1.0, 2.0], [-1.0, -2.0]],  # sample 0: d=2, k=2
            [[3.0, 4.0], [3.0, 4.0]],    # sample 1
        ])  # (n=2, d=2, k=2)
        
        importance = global_feature_importance(shap_values, agg="mean_abs")
        
        # Shape should be (d, k)
        assert importance.shape == (2, 2)
        # mean of abs for feature 0: (|1|+|3|)/2 = 2, (|2|+|4|)/2 = 3
        # mean of abs for feature 1: (|-1|+|3|)/2 = 2, (|-2|+|4|)/2 = 3
        expected = np.array([[2.0, 3.0], [2.0, 3.0]])
        np.testing.assert_allclose(importance, expected)
    
    def test_invalid_agg_raises(self):
        """Invalid agg raises ValueError."""
        shap_values = np.array([[1.0, 2.0]])
        
        with pytest.raises(ValueError, match="agg"):
            global_feature_importance(shap_values, agg="invalid")


class TestTypePreservation:
    """Test DataFrame input/output type preservation."""
    
    def test_dataframe_in_dataframe_out_single(self):
        """DataFrame input returns DataFrame for single output."""
        d = 2
        w = np.array([1.0, 2.0])
        
        def predict_fn(x):
            return x @ w
        
        x = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=["a", "b"],
            columns=["f1", "f2"]
        )
        
        shap_values = shapley_sampling_values(
            predict_fn, x, baseline=np.zeros(d), n_permutations=1, seed=0
        )
        
        assert isinstance(shap_values, pd.DataFrame)
        assert list(shap_values.index) == ["a", "b"]
        assert list(shap_values.columns) == ["f1", "f2"]
    
    def test_dataframe_in_numpy_out_multi(self):
        """DataFrame input returns numpy for multi-output."""
        d, k = 2, 3
        W = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        
        def predict_fn(x):
            return x @ W
        
        x = pd.DataFrame([[1.0, 2.0]], columns=["f1", "f2"])
        
        shap_values = shapley_sampling_values(
            predict_fn, x, baseline=np.zeros(d), n_permutations=1, seed=0
        )
        
        # Multi-output always returns numpy
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == (1, d, k)
