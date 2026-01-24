"""
Tests for utils/math_tools.py

Covers weighted_harmonic_mean function with:
- Basic correctness
- Exclusion + weight rescaling
- All invalid -> NaN
- Shape mismatch error
- Negative weight error
"""

import numpy as np
import pytest

from utils.math_tools import weighted_harmonic_mean


class TestWeightedHarmonicMean:
    """Test suite for weighted_harmonic_mean function."""

    def test_basic_correctness(self):
        """Test basic weighted harmonic mean calculation.
        
        values=[2,4], weights=[0.5,0.5]
        expected = 1 / (0.5/2 + 0.5/4) = 1 / 0.375 = 2.666...
        """
        values = np.array([2.0, 4.0])
        weights = np.array([0.5, 0.5])
        
        result = weighted_harmonic_mean(values, weights)
        expected = 1.0 / (0.5 / 2.0 + 0.5 / 4.0)  # 2.666...
        
        assert np.isclose(result, expected, rtol=1e-9)
        assert np.isclose(result, 8.0 / 3.0, rtol=1e-9)  # 2.666...

    def test_exclusion_and_weight_rescale(self):
        """Test that negative values are excluded and weights rescaled.
        
        values=[2, -1, 4], weights=[0.2, 0.7, 0.1]
        -1 is excluded, remaining weights [0.2, 0.1] rescale to [0.666..., 0.333...]
        expected = 1 / (0.666.../2 + 0.333.../4) = 1 / (0.333... + 0.0833...) = 2.4
        """
        values = np.array([2.0, -1.0, 4.0])
        weights = np.array([0.2, 0.7, 0.1])
        
        result = weighted_harmonic_mean(values, weights)
        
        # After excluding -1, weights [0.2, 0.1] rescale to [2/3, 1/3]
        w1 = 0.2 / 0.3  # 2/3
        w2 = 0.1 / 0.3  # 1/3
        expected = 1.0 / (w1 / 2.0 + w2 / 4.0)  # 2.4
        
        assert np.isclose(result, expected, rtol=1e-9)
        assert np.isclose(result, 2.4, rtol=1e-9)

    def test_all_invalid_returns_nan(self):
        """Test that all invalid values return NaN.
        
        values=[0, -3, np.nan] - all invalid (0 excluded, -3 excluded, nan excluded)
        """
        values = np.array([0.0, -3.0, np.nan])
        weights = np.array([1.0, 1.0, 1.0])
        
        result = weighted_harmonic_mean(values, weights)
        
        assert np.isnan(result)

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises ValueError."""
        values = np.array([1.0, 2.0])
        weights = np.array([0.5, 0.3, 0.2])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            weighted_harmonic_mean(values, weights)

    def test_negative_weights_raises(self):
        """Test that negative weights raise ValueError."""
        values = np.array([2.0, 4.0, 6.0])
        weights = np.array([0.5, -0.1, 0.6])
        
        with pytest.raises(ValueError, match="weights must be non-negative"):
            weighted_harmonic_mean(values, weights)

    def test_zero_weights_excluded(self):
        """Test that zero weights are excluded (contribute nothing).
        
        values=[2, 4, 6], weights=[0.5, 0, 0.5]
        Zero weight entry is excluded, result same as [2, 6] with [0.5, 0.5]
        """
        values = np.array([2.0, 4.0, 6.0])
        weights = np.array([0.5, 0.0, 0.5])
        
        result = weighted_harmonic_mean(values, weights)
        
        # Rescaled weights: [0.5, 0.5] -> [0.5, 0.5] (already sum to 1)
        expected = 1.0 / (0.5 / 2.0 + 0.5 / 6.0)  # 3.0
        
        assert np.isclose(result, expected, rtol=1e-9)
        assert np.isclose(result, 3.0, rtol=1e-9)

    def test_inf_values_excluded(self):
        """Test that Inf values are excluded."""
        values = np.array([2.0, np.inf, 4.0])
        weights = np.array([0.5, 0.25, 0.25])
        
        result = weighted_harmonic_mean(values, weights)
        
        # Inf excluded, remaining weights [0.5, 0.25] rescale to [2/3, 1/3]
        w1 = 0.5 / 0.75
        w2 = 0.25 / 0.75
        expected = 1.0 / (w1 / 2.0 + w2 / 4.0)
        
        assert np.isclose(result, expected, rtol=1e-9)

    def test_single_valid_entry(self):
        """Test with only one valid entry after exclusion."""
        values = np.array([5.0, -1.0, 0.0])
        weights = np.array([1.0, 1.0, 1.0])
        
        result = weighted_harmonic_mean(values, weights)
        
        # Only 5.0 is valid, harmonic mean of single value is the value itself
        assert np.isclose(result, 5.0, rtol=1e-9)

    def test_all_weights_zero_returns_nan(self):
        """Test that all-zero weights return NaN."""
        values = np.array([2.0, 4.0])
        weights = np.array([0.0, 0.0])
        
        result = weighted_harmonic_mean(values, weights)
        
        assert np.isnan(result)
