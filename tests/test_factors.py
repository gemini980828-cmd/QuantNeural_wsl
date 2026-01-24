"""
Tests for src/factors.py

Covers:
- winsorize clips extremes correctly
- zscore produces ~0 mean and ~1 std within date bucket
- zscore with constant column returns zeros
- build_style_factors outputs required columns and finite values
- zscore with group_col (sector) produces independent normalization
- build_relative_earnings_momentum shape, columns, correctness
"""

import numpy as np
import pandas as pd
import pytest

from src.factors import (
    WinsorizeParams,
    winsorize_series,
    zscore_cross_section,
    build_style_factors,
    build_relative_earnings_momentum,
)


class TestWinsorizeSeries:
    """Test winsorize_series function."""
    
    def test_clips_extremes(self):
        """Test that winsorize clips outliers to quantile bounds."""
        # Series with outliers
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is extreme
        p = WinsorizeParams(lower_q=0.1, upper_q=0.9)
        
        result = winsorize_series(s, p)
        
        # Compute expected bounds
        lo = s.quantile(0.1)
        hi = s.quantile(0.9)
        
        assert result.max() <= hi + 1e-9
        assert result.min() >= lo - 1e-9
    
    def test_clips_to_quantile_threshold(self):
        """Test that clipped values equal the quantile threshold."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
        p = WinsorizeParams(lower_q=0.01, upper_q=0.99)
        
        result = winsorize_series(s, p)
        hi = s.quantile(0.99)
        
        # The max should be clipped to hi
        assert np.isclose(result.iloc[-1], hi, rtol=1e-9)
    
    def test_preserves_index(self):
        """Test that winsorize preserves the original index."""
        idx = pd.Index(["a", "b", "c", "d", "e"])
        s = pd.Series([1, 2, 3, 4, 100], index=idx)
        p = WinsorizeParams()
        
        result = winsorize_series(s, p)
        
        assert result.index.equals(idx)
    
    def test_no_change_within_bounds(self):
        """Test that values within bounds are unchanged."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        p = WinsorizeParams(lower_q=0.0, upper_q=1.0)  # No clipping
        
        result = winsorize_series(s, p)
        
        assert np.allclose(result.values, s.values)


class TestZscoreCrossSection:
    """Test zscore_cross_section function."""
    
    def test_zero_mean_unit_std(self):
        """Test that z-score produces ~0 mean and ~1 std within date bucket."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 10,
            "value": np.random.randn(10) * 5 + 10  # varied values
        })
        
        result = zscore_cross_section(df, cols=["value"], group_col=None, date_col="date")
        
        # Check mean ≈ 0 and std ≈ 1
        assert np.abs(result["value"].mean()) < 1e-9
        assert np.abs(result["value"].std(ddof=0) - 1.0) < 1e-9
    
    def test_constant_column_returns_zeros(self):
        """Test that constant column returns zeros (std=0 case)."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 5,
            "value": [5.0, 5.0, 5.0, 5.0, 5.0]  # constant
        })
        
        result = zscore_cross_section(df, cols=["value"], group_col=None, date_col="date")
        
        # All zeros
        assert np.allclose(result["value"].values, 0.0)
    
    def test_multiple_dates_independent(self):
        """Test that z-score is computed independently per date."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 3 + ["2020-02-01"] * 3,
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        })
        
        result = zscore_cross_section(df, cols=["value"], group_col=None, date_col="date")
        
        # Each date should have mean ≈ 0
        for date in ["2020-01-01", "2020-02-01"]:
            date_vals = result[df["date"] == date]["value"]
            assert np.abs(date_vals.mean()) < 1e-9
    
    def test_with_group_col(self):
        """Test that group_col produces independent normalization per group."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 6,
            "sector": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0]
        })
        
        result = zscore_cross_section(
            df, cols=["value"], group_col="sector", date_col="date"
        )
        
        # Each sector should have mean ≈ 0
        for sector in ["A", "B"]:
            sector_vals = result[df["sector"] == sector]["value"]
            assert np.abs(sector_vals.mean()) < 1e-9
            assert np.abs(sector_vals.std(ddof=0) - 1.0) < 1e-9
    
    def test_preserves_other_columns(self):
        """Test that non-transformed columns are preserved."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 3,
            "ticker": ["AAPL", "GOOG", "MSFT"],
            "value": [1.0, 2.0, 3.0]
        })
        
        result = zscore_cross_section(df, cols=["value"], group_col=None, date_col="date")
        
        assert "ticker" in result.columns
        assert result["ticker"].tolist() == ["AAPL", "GOOG", "MSFT"]


class TestBuildStyleFactors:
    """Test build_style_factors function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with required columns."""
        np.random.seed(42)
        n = 20  # 2 dates x 10 tickers
        
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 10 + ["2020-02-01"] * 10,
            "ticker": [f"T{i}" for i in range(10)] * 2,
            "PER": np.random.uniform(5, 30, n),
            "PBR": np.random.uniform(0.5, 5, n),
            "DivYield": np.random.uniform(0, 0.05, n),
            "SalesGrowth_3Y": np.random.uniform(-0.1, 0.3, n),
            "NetIncomeGrowth_3Y": np.random.uniform(-0.2, 0.4, n),
            "EPSGrowth_Fwd": np.random.uniform(-0.1, 0.2, n),
            "R_1M": np.random.uniform(-0.1, 0.1, n),
            "R_3M": np.random.uniform(-0.2, 0.2, n),
            "R_12M": np.random.uniform(-0.3, 0.5, n),
            "ROE": np.random.uniform(0.05, 0.3, n),
            "OPM": np.random.uniform(0.05, 0.25, n),
            "DebtRatio": np.random.uniform(0.1, 0.8, n),
            "EarningsVol": np.random.uniform(0.1, 0.5, n),
            "MarketCap": np.random.uniform(1e9, 1e12, n),
        })
        return df
    
    def test_output_columns_exist(self, sample_df):
        """Test that build_style_factors creates required output columns."""
        wins = WinsorizeParams()
        result = build_style_factors(sample_df, date_col="date", wins=wins)
        
        required_cols = ["Value", "Growth", "Momentum", "Quality", "Size"]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_output_finite_values(self, sample_df):
        """Test that factor columns have finite values for typical inputs."""
        wins = WinsorizeParams()
        result = build_style_factors(sample_df, date_col="date", wins=wins)
        
        factor_cols = ["Value", "Growth", "Momentum", "Quality", "Size"]
        for col in factor_cols:
            finite_ratio = np.isfinite(result[col]).mean()
            assert finite_ratio > 0.9, f"Column {col} has too many non-finite values"
    
    def test_preserves_row_count(self, sample_df):
        """Test that row count is preserved."""
        wins = WinsorizeParams()
        result = build_style_factors(sample_df, date_col="date", wins=wins)
        
        assert len(result) == len(sample_df)
    
    def test_intermediate_columns_present(self, sample_df):
        """Test that intermediate columns (inv_*) are present."""
        wins = WinsorizeParams()
        result = build_style_factors(sample_df, date_col="date", wins=wins)
        
        intermediate_cols = ["inv_PER", "inv_PBR", "inv_Debt", "inv_EarnVol", "inv_Size"]
        for col in intermediate_cols:
            assert col in result.columns, f"Missing intermediate column: {col}"
    
    def test_handles_zero_denominators(self):
        """Test that zero denominators are handled (become NaN, not inf)."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 3,
            "ticker": ["A", "B", "C"],
            "PER": [10.0, 0.0, 15.0],  # One zero
            "PBR": [2.0, 3.0, 4.0],
            "DivYield": [0.02, 0.03, 0.01],
            "SalesGrowth_3Y": [0.1, 0.2, 0.15],
            "NetIncomeGrowth_3Y": [0.1, 0.2, 0.15],
            "EPSGrowth_Fwd": [0.1, 0.2, 0.15],
            "R_1M": [0.05, 0.02, 0.03],
            "R_3M": [0.1, 0.05, 0.08],
            "R_12M": [0.2, 0.1, 0.15],
            "ROE": [0.15, 0.2, 0.18],
            "OPM": [0.1, 0.12, 0.11],
            "DebtRatio": [0.3, 0.4, 0.35],
            "EarningsVol": [0.2, 0.25, 0.22],
            "MarketCap": [1e10, 2e10, 1.5e10],
        })
        
        wins = WinsorizeParams()
        result = build_style_factors(df, date_col="date", wins=wins)
        
        # Should not have inf values
        assert not np.any(np.isinf(result["inv_PER"].values))


class TestBuildRelativeEarningsMomentum:
    """Test build_relative_earnings_momentum function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2020-01-31", periods=5, freq="ME")
        sectors = [f"S{i}" for i in range(10)]
        
        # Create DataFrames with positive values for log
        np.random.seed(42)
        sector_fy1 = pd.DataFrame(
            100 + np.random.randn(5, 10) * 10,
            index=dates,
            columns=sectors
        )
        sector_fy2 = pd.DataFrame(
            100 + np.random.randn(5, 10) * 10,
            index=dates,
            columns=sectors
        )
        market_fy1 = pd.Series(100 + np.random.randn(5) * 10, index=dates)
        market_fy2 = pd.Series(100 + np.random.randn(5) * 10, index=dates)
        
        return sector_fy1, sector_fy2, market_fy1, market_fy2, dates, sectors
    
    def test_output_shape(self, sample_data):
        """Test that output shape is (T, 20)."""
        sector_fy1, sector_fy2, market_fy1, market_fy2, dates, sectors = sample_data
        
        result = build_relative_earnings_momentum(
            sector_fy1, sector_fy2, market_fy1, market_fy2, method="logdiff"
        )
        
        assert result.shape == (5, 20)
    
    def test_output_columns(self, sample_data):
        """Test that output columns are in correct order."""
        sector_fy1, sector_fy2, market_fy1, market_fy2, dates, sectors = sample_data
        
        result = build_relative_earnings_momentum(
            sector_fy1, sector_fy2, market_fy1, market_fy2, method="logdiff"
        )
        
        expected_cols = [f"S{i}_FY1" for i in range(10)] + [f"S{i}_FY2" for i in range(10)]
        assert list(result.columns) == expected_cols
    
    def test_output_index(self, sample_data):
        """Test that output index matches input."""
        sector_fy1, sector_fy2, market_fy1, market_fy2, dates, sectors = sample_data
        
        result = build_relative_earnings_momentum(
            sector_fy1, sector_fy2, market_fy1, market_fy2, method="logdiff"
        )
        
        assert result.index.equals(dates)
    
    def test_logdiff_correctness(self):
        """Test logdiff method correctness on simple deterministic data."""
        dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
        sectors = ["S0"]
        
        # Sector FY1: [100, 110, 121] => log diff = [NaN, log(110/100), log(121/110)]
        # Market FY1: [100, 105, 110.25] => log diff = [NaN, log(1.05), log(1.05)]
        sector_fy1 = pd.DataFrame({"S0": [100.0, 110.0, 121.0]}, index=dates)
        sector_fy2 = pd.DataFrame({"S0": [100.0, 100.0, 100.0]}, index=dates)
        market_fy1 = pd.Series([100.0, 105.0, 110.25], index=dates)
        market_fy2 = pd.Series([100.0, 100.0, 100.0], index=dates)
        
        result = build_relative_earnings_momentum(
            sector_fy1, sector_fy2, market_fy1, market_fy2, method="logdiff"
        )
        
        # At t=2020-02-29:
        # sector_delta = log(110) - log(100) = log(1.10)
        # market_delta = log(105) - log(100) = log(1.05)
        # rel = log(1.10) - log(1.05) = log(1.10/1.05)
        expected_t1 = np.log(1.10) - np.log(1.05)
        assert np.isclose(result.loc[dates[1], "S0_FY1"], expected_t1, rtol=1e-9)
        
        # At t=2020-03-31:
        # sector_delta = log(121) - log(110) = log(121/110)
        # market_delta = log(110.25) - log(105) = log(110.25/105)
        # rel = log(121/110) - log(110.25/105)
        expected_t2 = np.log(121.0 / 110.0) - np.log(110.25 / 105.0)
        assert np.isclose(result.loc[dates[2], "S0_FY1"], expected_t2, rtol=1e-9)
        
        # First row should be NaN (from diff)
        assert np.isnan(result.loc[dates[0], "S0_FY1"])
    
    def test_pct_correctness(self):
        """Test pct method correctness on simple deterministic data."""
        dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
        sectors = ["S0"]
        
        # Sector: [100, 120, 150] => pct_change = [NaN, 0.2, 0.25]
        # Market: [100, 110, 121] => pct_change = [NaN, 0.1, 0.1]
        sector_fy1 = pd.DataFrame({"S0": [100.0, 120.0, 150.0]}, index=dates)
        sector_fy2 = pd.DataFrame({"S0": [100.0, 100.0, 100.0]}, index=dates)
        market_fy1 = pd.Series([100.0, 110.0, 121.0], index=dates)
        market_fy2 = pd.Series([100.0, 100.0, 100.0], index=dates)
        
        result = build_relative_earnings_momentum(
            sector_fy1, sector_fy2, market_fy1, market_fy2, method="pct"
        )
        
        # At t=2020-02-29: rel = 0.2 - 0.1 = 0.1
        assert np.isclose(result.loc[dates[1], "S0_FY1"], 0.1, rtol=1e-9)
        
        # At t=2020-03-31: rel = 0.25 - 0.1 = 0.15
        assert np.isclose(result.loc[dates[2], "S0_FY1"], 0.15, rtol=1e-9)
        
        # First row should be NaN
        assert np.isnan(result.loc[dates[0], "S0_FY1"])
    
    def test_invalid_method_raises(self, sample_data):
        """Test that invalid method raises ValueError."""
        sector_fy1, sector_fy2, market_fy1, market_fy2, _, _ = sample_data
        
        with pytest.raises(ValueError, match="method must be"):
            build_relative_earnings_momentum(
                sector_fy1, sector_fy2, market_fy1, market_fy2, method="bad"
            )
    
    def test_column_mismatch_raises(self, sample_data):
        """Test that mismatched sector columns raises ValueError."""
        sector_fy1, sector_fy2, market_fy1, market_fy2, _, _ = sample_data
        
        # Rename columns in FY2 to create mismatch
        sector_fy2_bad = sector_fy2.rename(columns={"S0": "SX"})
        
        with pytest.raises(ValueError, match="columns"):
            build_relative_earnings_momentum(
                sector_fy1, sector_fy2_bad, market_fy1, market_fy2
            )
    
    def test_index_mismatch_raises(self, sample_data):
        """Test that mismatched indices raises ValueError."""
        sector_fy1, sector_fy2, market_fy1, market_fy2, _, _ = sample_data
        
        # Create market with different index
        bad_dates = pd.date_range("2021-01-31", periods=5, freq="ME")
        market_fy1_bad = pd.Series([100, 100, 100, 100, 100], index=bad_dates)
        
        with pytest.raises(ValueError, match="identical indices"):
            build_relative_earnings_momentum(
                sector_fy1, sector_fy2, market_fy1_bad, market_fy2
            )
