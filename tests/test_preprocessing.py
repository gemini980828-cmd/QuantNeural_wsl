"""
Tests for src/preprocessing.py

Covers:
- HP lambda mapping (classic, ravn_uhlig, manual)
- RankGauss fit/transform contract + determinism
- Hamilton filter index alignment + non-empty result
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    HPParams,
    HamiltonParams,
    QuantDataProcessor,
)


class TestHPLambdaMapping:
    """Test HP filter lambda value mapping."""
    
    def test_classic_monthly(self):
        """Test classic lambda for monthly frequency."""
        hp = HPParams(mode="classic")
        lamb = QuantDataProcessor._hp_lambda("M", hp)
        assert lamb == 14400.0
    
    def test_classic_quarterly(self):
        """Test classic lambda for quarterly frequency."""
        hp = HPParams(mode="classic")
        lamb = QuantDataProcessor._hp_lambda("Q", hp)
        assert lamb == 1600.0
    
    def test_classic_yearly(self):
        """Test classic lambda for yearly frequency."""
        hp = HPParams(mode="classic")
        lamb = QuantDataProcessor._hp_lambda("Y", hp)
        assert lamb == 100.0
    
    def test_ravn_uhlig_monthly(self):
        """Test Ravn-Uhlig lambda for monthly frequency."""
        hp = HPParams(mode="ravn_uhlig")
        lamb = QuantDataProcessor._hp_lambda("M", hp)
        assert lamb == 129600.0
    
    def test_ravn_uhlig_quarterly(self):
        """Test Ravn-Uhlig lambda for quarterly frequency."""
        hp = HPParams(mode="ravn_uhlig")
        lamb = QuantDataProcessor._hp_lambda("Q", hp)
        assert lamb == 1600.0
    
    def test_ravn_uhlig_yearly(self):
        """Test Ravn-Uhlig lambda for yearly frequency."""
        hp = HPParams(mode="ravn_uhlig")
        lamb = QuantDataProcessor._hp_lambda("Y", hp)
        assert lamb == 6.25
    
    def test_manual_with_value(self):
        """Test manual lambda with provided value."""
        hp = HPParams(mode="manual", lamb_manual=5000.0)
        lamb = QuantDataProcessor._hp_lambda("M", hp)
        assert lamb == 5000.0
    
    def test_manual_without_value_raises(self):
        """Test manual mode without lamb_manual raises ValueError."""
        hp = HPParams(mode="manual")
        with pytest.raises(ValueError, match="requires lamb_manual"):
            QuantDataProcessor._hp_lambda("M", hp)


class TestRankGaussContract:
    """Test RankGauss fit/transform contract."""
    
    def test_transform_before_fit_raises(self):
        """Test that calling transform before fit raises RuntimeError."""
        proc = QuantDataProcessor(rankgauss=True)
        X = np.random.randn(100, 5)
        
        with pytest.raises(RuntimeError, match="fit_rankgauss.*not called"):
            proc.transform_rankgauss(X)
    
    def test_fit_transform_same_shape(self):
        """Test that transform returns same shape as input."""
        proc = QuantDataProcessor(rankgauss=True, random_state=42)
        X_train = np.random.randn(100, 5)
        
        proc.fit_rankgauss(X_train)
        X_transformed = proc.transform_rankgauss(X_train)
        
        assert X_transformed.shape == X_train.shape
    
    def test_fit_transform_finite_values(self):
        """Test that transformed values are finite."""
        proc = QuantDataProcessor(rankgauss=True, random_state=42)
        X_train = np.random.randn(100, 5)
        
        proc.fit_rankgauss(X_train)
        X_transformed = proc.transform_rankgauss(X_train)
        
        assert np.all(np.isfinite(X_transformed))
    
    def test_determinism_same_random_state(self):
        """Test that same random_state produces identical outputs."""
        np.random.seed(123)
        X_train = np.random.randn(100, 5)
        
        proc1 = QuantDataProcessor(rankgauss=True, random_state=42)
        proc1.fit_rankgauss(X_train)
        X1 = proc1.transform_rankgauss(X_train)
        
        proc2 = QuantDataProcessor(rankgauss=True, random_state=42)
        proc2.fit_rankgauss(X_train)
        X2 = proc2.transform_rankgauss(X_train)
        
        assert np.allclose(X1, X2)
    
    def test_rankgauss_false_returns_unchanged(self):
        """Test that rankgauss=False returns input unchanged."""
        proc = QuantDataProcessor(rankgauss=False)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        X_out = proc.transform_rankgauss(X)
        
        assert np.allclose(X_out, X)
    
    def test_rankgauss_false_fit_does_nothing(self):
        """Test that fit with rankgauss=False does nothing and doesn't break."""
        proc = QuantDataProcessor(rankgauss=False)
        X_train = np.random.randn(50, 3)
        
        # Should not raise
        proc.fit_rankgauss(X_train)
        
        # Transform should still work (returns unchanged)
        X_out = proc.transform_rankgauss(X_train)
        assert np.allclose(X_out, X_train)


class TestHamiltonFilter:
    """Test Hamilton regression filter."""
    
    @pytest.fixture
    def synthetic_series(self):
        """Create synthetic monthly series with 50 points."""
        dates = pd.date_range("2020-01-31", periods=50, freq="ME")
        # Simple AR(1) + trend for testing
        np.random.seed(42)
        values = np.cumsum(np.random.randn(50)) + np.arange(50) * 0.1
        return pd.Series(values, index=dates)
    
    def test_non_empty_result(self, synthetic_series):
        """Test that Hamilton filter returns non-empty result."""
        params = HamiltonParams(h=2, p=3)
        result = QuantDataProcessor.apply_hamilton_filter(
            synthetic_series, params=params
        )
        
        assert len(result) > 0
    
    def test_index_alignment_first_timestamp(self, synthetic_series):
        """Test that first result index matches expected target timestamp.
        
        With p=3, h=2:
        - First valid i is p-1 = 2 (so we use y[2], y[1], y[0] as lags)
        - Target is y[i+h] = y[2+2] = y[4]
        - Result index[0] should equal series.index[4]
        """
        params = HamiltonParams(h=2, p=3)
        result = QuantDataProcessor.apply_hamilton_filter(
            synthetic_series, params=params
        )
        
        # First target index: (p-1) + h = 2 + 2 = 4
        expected_first_idx = synthetic_series.index[4]
        assert result.index[0] == expected_first_idx
    
    def test_index_strictly_increasing(self, synthetic_series):
        """Test that result index is strictly increasing."""
        params = HamiltonParams(h=2, p=3)
        result = QuantDataProcessor.apply_hamilton_filter(
            synthetic_series, params=params
        )
        
        assert result.index.is_monotonic_increasing
    
    def test_finite_values(self, synthetic_series):
        """Test that all result values are finite."""
        params = HamiltonParams(h=2, p=3)
        result = QuantDataProcessor.apply_hamilton_filter(
            synthetic_series, params=params
        )
        
        assert np.all(np.isfinite(result.values))
    
    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty series."""
        dates = pd.date_range("2020-01-31", periods=5, freq="ME")
        series = pd.Series(np.arange(5.0), index=dates)
        
        # h=10, p=3 requires at least h+p = 13 points
        params = HamiltonParams(h=10, p=3)
        result = QuantDataProcessor.apply_hamilton_filter(series, params=params)
        
        assert len(result) == 0
    
    def test_result_length(self, synthetic_series):
        """Test that result has expected length.
        
        With n=50, h=2, p=3:
        - Valid i range: [p-1, n-h-1] = [2, 47]
        - Number of observations: 47 - 2 + 1 = 46
        """
        params = HamiltonParams(h=2, p=3)
        result = QuantDataProcessor.apply_hamilton_filter(
            synthetic_series, params=params
        )
        
        expected_length = len(synthetic_series) - params.h - (params.p - 1)
        assert len(result) == expected_length


class TestHPFilter:
    """Test HP filter application."""
    
    def test_hp_filter_returns_two_series(self):
        """Test that HP filter returns cycle and trend series."""
        proc = QuantDataProcessor()
        dates = pd.date_range("2020-01-31", periods=50, freq="ME")
        series = pd.Series(np.cumsum(np.random.randn(50)), index=dates)
        
        hp = HPParams(mode="classic")
        cycle, trend = proc.apply_hp_filter(series, freq="M", hp=hp)
        
        assert isinstance(cycle, pd.Series)
        assert isinstance(trend, pd.Series)
        assert len(cycle) == len(series)
        assert len(trend) == len(series)
    
    def test_hp_filter_index_preserved(self):
        """Test that HP filter preserves index."""
        proc = QuantDataProcessor()
        dates = pd.date_range("2020-01-31", periods=30, freq="ME")
        series = pd.Series(np.random.randn(30), index=dates)
        
        hp = HPParams(mode="classic")
        cycle, trend = proc.apply_hp_filter(series, freq="M", hp=hp)
        
        assert cycle.index.equals(series.index)
        assert trend.index.equals(series.index)
    
    def test_hp_leakage_guard_prevents_future_contamination(self):
        """Test that leakage guard prevents future data from affecting past outputs."""
        proc = QuantDataProcessor()
        
        # Build deterministic series of length 50
        dates = pd.date_range("2020-01-31", periods=50, freq="ME")
        np.random.seed(42)
        values_a = np.cumsum(np.random.randn(50))
        series_a = pd.Series(values_a, index=dates)
        
        # Compute with leakage guard enabled
        hp = HPParams(mode="classic", leakage_guard=True, lookback=120)
        cycle_a, trend_a = proc.apply_hp_filter(series_a, freq="M", hp=hp)
        
        # Create modified series: change LAST 10 points dramatically
        values_b = values_a.copy()
        values_b[-10:] += 1000.0  # Huge change
        series_b = pd.Series(values_b, index=dates)
        
        cycle_b, trend_b = proc.apply_hp_filter(series_b, freq="M", hp=hp)
        
        # Assert: First 40 points (all except last 10) should be IDENTICAL
        # because leakage guard ensures output at t doesn't depend on data after t
        k = 10
        np.testing.assert_allclose(
            trend_a.iloc[:-k], trend_b.iloc[:-k], 
            rtol=1e-10, atol=1e-10,
            err_msg="Leakage guard failed: past outputs changed when future data changed"
        )
        np.testing.assert_allclose(
            cycle_a.iloc[:-k], cycle_b.iloc[:-k],
            rtol=1e-10, atol=1e-10,
            err_msg="Leakage guard failed: past cycle outputs changed"
        )
    
    def test_hp_two_sided_affected_by_future(self):
        """Test that two-sided filter IS affected by future changes (guard is meaningful)."""
        proc = QuantDataProcessor()
        
        # Same setup as above
        dates = pd.date_range("2020-01-31", periods=50, freq="ME")
        np.random.seed(42)
        values_a = np.cumsum(np.random.randn(50))
        series_a = pd.Series(values_a, index=dates)
        
        # Compute with leakage guard DISABLED (two-sided filter)
        hp = HPParams(mode="classic", leakage_guard=False)
        cycle_a, trend_a = proc.apply_hp_filter(series_a, freq="M", hp=hp)
        
        # Modify last 10 points
        values_b = values_a.copy()
        values_b[-10:] += 1000.0
        series_b = pd.Series(values_b, index=dates)
        
        cycle_b, trend_b = proc.apply_hp_filter(series_b, freq="M", hp=hp)
        
        # Assert: At least one early point should differ significantly
        # This proves the guard is meaningful (two-sided filter does leak)
        k = 10
        max_diff = np.max(np.abs(trend_a.iloc[:-k].values - trend_b.iloc[:-k].values))
        
        assert max_diff > 1e-6, (
            f"Two-sided filter should be affected by future changes, "
            f"but max diff was only {max_diff:.2e}. "
            f"This suggests the filter is not actually two-sided."
        )

