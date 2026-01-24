"""
Tests for src/backtest_harness.py

Covers:
- Determinism: identical inputs produce identical outputs
- Execution lag / no look-ahead: weights become effective at correct lagged date
- Monthly vs quarterly rebalancing
- Costs reduce performance
- Turnover and trades log consistency
- Validation failures raise ValueError
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest_harness import (
    validate_prices_frame,
    validate_target_weights,
    resample_rebalance_dates,
    align_and_lag_weights,
    run_backtest,
)


class TestValidatePricesFrame:
    """Test price validation."""
    
    def test_valid_wide_format(self):
        """Test valid wide format prices pass."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        prices = pd.DataFrame({
            "AAPL": np.linspace(100, 110, 10),
            "MSFT": np.linspace(200, 220, 10),
        }, index=dates)
        
        # Should not raise
        validate_prices_frame(prices)
    
    def test_non_increasing_index_raises(self):
        """Test non-increasing index raises ValueError."""
        dates = pd.to_datetime(["2023-01-03", "2023-01-02", "2023-01-04"])
        prices = pd.DataFrame({
            "AAPL": [100, 101, 102],
        }, index=dates)
        
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_prices_frame(prices)
    
    def test_non_finite_prices_raises(self):
        """Test NaN prices raise ValueError."""
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100, np.nan, 102],
        }, index=dates)
        
        with pytest.raises(ValueError, match="finite"):
            validate_prices_frame(prices)
    
    def test_non_positive_prices_raises(self):
        """Test zero/negative prices raise ValueError."""
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100, 0, 102],
        }, index=dates)
        
        with pytest.raises(ValueError, match="> 0"):
            validate_prices_frame(prices)


class TestValidateTargetWeights:
    """Test weights validation."""
    
    def test_valid_weights(self):
        """Test valid weights pass."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
        weights = pd.DataFrame({
            "AAPL": [0.5, 0.6, 0.4],
            "MSFT": [0.5, 0.4, 0.6],
        }, index=dates)
        
        # Should not raise
        validate_target_weights(weights)
    
    def test_negative_weights_allowed(self):
        """Test negative weights (shorting) are allowed."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28"])
        weights = pd.DataFrame({
            "AAPL": [0.5, -0.2],
            "MSFT": [0.5, 1.2],
        }, index=dates)
        
        # Should not raise (negatives allowed for v1)
        validate_target_weights(weights)
    
    def test_non_finite_weights_raises(self):
        """Test NaN weights raise ValueError."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28"])
        weights = pd.DataFrame({
            "AAPL": [0.5, np.nan],
            "MSFT": [0.5, 0.5],
        }, index=dates)
        
        with pytest.raises(ValueError, match="finite"):
            validate_target_weights(weights)


class TestResampleRebalanceDates:
    """Test rebalance date selection."""
    
    def test_monthly_returns_all(self):
        """Test monthly frequency returns all dates."""
        dates = pd.to_datetime([
            "2023-01-31", "2023-02-28", "2023-03-31",
            "2023-04-30", "2023-05-31", "2023-06-30"
        ])
        
        result = resample_rebalance_dates(dates, rebalance="M")
        
        assert len(result) == 6
        assert result.equals(dates)
    
    def test_quarterly_returns_every_third(self):
        """Test quarterly frequency returns every 3rd date."""
        dates = pd.to_datetime([
            "2023-01-31", "2023-02-28", "2023-03-31",
            "2023-04-30", "2023-05-31", "2023-06-30"
        ])
        
        result = resample_rebalance_dates(dates, rebalance="Q")
        
        # Should return indices [0, 3] -> dates[0], dates[3]
        assert len(result) == 2
        assert result[0] == dates[0]
        assert result[1] == dates[3]
    
    def test_invalid_rebalance_raises(self):
        """Test invalid rebalance value raises ValueError."""
        dates = pd.to_datetime(["2023-01-31"])
        
        with pytest.raises(ValueError, match="'M' or 'Q'"):
            resample_rebalance_dates(dates, rebalance="W")


class TestAlignAndLagWeights:
    """Test weight alignment with execution lag."""
    
    def test_basic_alignment_with_lag(self):
        """Test weights are aligned and lagged correctly."""
        # Trading days: Mon 2nd, Tue 3rd, Wed 4th, Thu 5th, Fri 6th
        trading_days = pd.to_datetime([
            "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"
        ])
        
        # Signal on Sunday Jan 1st
        signal_dates = pd.to_datetime(["2023-01-01"])
        weights = pd.DataFrame({
            "AAPL": [1.0],
        }, index=signal_dates)
        
        # With lag=1: first trading day >= Jan 1 is Jan 2, then +1 = Jan 3
        result = align_and_lag_weights(weights, trading_days, execution_lag_days=1)
        
        assert len(result) == 1
        assert result.index[0] == pd.Timestamp("2023-01-03")
    
    def test_signal_beyond_calendar_dropped(self):
        """Test signal dates beyond trading calendar are dropped."""
        trading_days = pd.to_datetime(["2023-01-02", "2023-01-03"])
        
        # Signal on Jan 5, beyond last trading day
        signal_dates = pd.to_datetime(["2023-01-05"])
        weights = pd.DataFrame({"AAPL": [1.0]}, index=signal_dates)
        
        result = align_and_lag_weights(weights, trading_days, execution_lag_days=1)
        
        assert len(result) == 0


class TestRunBacktest:
    """Test complete backtest functionality."""
    
    @pytest.fixture
    def simple_prices(self):
        """Create simple rising prices for 20 trading days."""
        dates = pd.date_range("2023-01-02", periods=20, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 120, 20),  # 20% gain
            "B": np.linspace(100, 110, 20),  # 10% gain
        }, index=dates)
        return prices
    
    @pytest.fixture
    def simple_weights(self):
        """Create simple weights with 2 rebalance signals."""
        dates = pd.to_datetime(["2023-01-02", "2023-01-16"])
        weights = pd.DataFrame({
            "A": [1.0, 0.5],
            "B": [0.0, 0.5],
        }, index=dates)
        return weights
    
    def test_determinism(self, simple_prices, simple_weights):
        """Test same inputs produce identical outputs."""
        result1 = run_backtest(simple_prices, simple_weights)
        result2 = run_backtest(simple_prices, simple_weights)
        
        pd.testing.assert_series_equal(
            result1["equity_curve"], result2["equity_curve"]
        )
        pd.testing.assert_series_equal(
            result1["daily_returns"], result2["daily_returns"]
        )
    
    def test_execution_lag_no_lookahead(self, simple_prices):
        """Test execution lag prevents look-ahead."""
        # Signal on first day, should be effective on day after lag
        dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=dates)
        
        result = run_backtest(
            simple_prices, weights, execution_lag_days=1
        )
        
        # With lag=1, first rebalance should be on Jan 3 (second trading day)
        assert result["rebalance_dates"][0] == simple_prices.index[1]
    
    def test_quarterly_fewer_rebalances(self, simple_prices):
        """Test quarterly uses fewer rebalances than monthly."""
        # 4 monthly signals
        dates = pd.to_datetime([
            "2023-01-02", "2023-01-05", "2023-01-10", "2023-01-16"
        ])
        weights = pd.DataFrame({
            "A": [1.0, 0.8, 0.6, 0.4],
            "B": [0.0, 0.2, 0.4, 0.6],
        }, index=dates)
        
        result_m = run_backtest(simple_prices, weights, rebalance="M")
        result_q = run_backtest(simple_prices, weights, rebalance="Q")
        
        # Monthly should have 4 rebalances, quarterly should have 2
        assert len(result_m["rebalance_dates"]) == 4
        assert len(result_q["rebalance_dates"]) == 2
    
    def test_costs_reduce_performance(self, simple_prices, simple_weights):
        """Test costs reduce final equity."""
        result_no_cost = run_backtest(
            simple_prices, simple_weights, cost_bps=0, slippage_bps=0
        )
        result_with_cost = run_backtest(
            simple_prices, simple_weights, cost_bps=50, slippage_bps=50
        )
        
        assert result_with_cost["equity_curve"].iloc[-1] < result_no_cost["equity_curve"].iloc[-1]
    
    def test_turnover_consistency(self):
        """Test turnover matches expected calculation."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 110, 10),
        }, index=dates)
        
        # Two rebalances: [0,0] -> [0.6, 0.4] -> [0.4, 0.6]
        signal_dates = pd.to_datetime(["2023-01-02", "2023-01-06"])
        weights = pd.DataFrame({
            "A": [0.6, 0.4],
            "B": [0.4, 0.6],
        }, index=signal_dates)
        
        result = run_backtest(prices, weights, execution_lag_days=1)
        
        # First turnover: |0.6-0| + |0.4-0| = 1.0
        # Second turnover: |0.4-0.6| + |0.6-0.4| = 0.4
        assert len(result["turnover"]) == 2
        assert np.isclose(result["turnover"].iloc[0], 1.0)
        assert np.isclose(result["turnover"].iloc[1], 0.4)
    
    def test_trades_log(self):
        """Test trades DataFrame records weight changes."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 110, 10),
        }, index=dates)
        
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [0.6],
            "B": [0.4],
        }, index=signal_dates)
        
        result = run_backtest(prices, weights, execution_lag_days=1)
        
        trades = result["trades"]
        assert len(trades) == 2  # Both A and B get non-zero delta
        assert "date" in trades.columns
        assert "ticker" in trades.columns
        assert "delta_weight" in trades.columns
    
    def test_metrics_exist(self, simple_prices, simple_weights):
        """Test all required metrics are present."""
        result = run_backtest(simple_prices, simple_weights)
        
        expected_keys = [
            "cagr", "ann_vol", "sharpe", "max_drawdown",
            "total_turnover", "total_cost"
        ]
        for key in expected_keys:
            assert key in result["metrics"], f"Missing metric: {key}"
    
    def test_max_gross_leverage_scaling(self):
        """Test max_gross_leverage scales weights appropriately."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 110, 10),
        }, index=dates)
        
        # Weights sum to 1.6 gross
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [1.0],
            "B": [0.6],
        }, index=signal_dates)
        
        result = run_backtest(
            prices, weights, 
            execution_lag_days=1,
            max_gross_leverage=1.0
        )
        
        # Check that weights_used are scaled
        used_weights = result["weights_used"]
        gross = np.abs(used_weights.iloc[0]).sum()
        assert gross <= 1.0 + 1e-9  # Allow tiny floating point tolerance


class TestLongFormatValidation:
    """Test long format price validation."""
    
    def test_long_format_valid_data_accepted(self):
        """Test valid long format with multiple tickers per date passes."""
        # Build long format: duplicate dates are normal
        dates = pd.to_datetime([
            "2023-01-02", "2023-01-02",  # 2 tickers on same date
            "2023-01-03", "2023-01-03",
            "2023-01-04", "2023-01-04",
        ])
        prices = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "close": [100, 200, 101, 201, 102, 202],
        }, index=dates)
        
        # Should not raise
        validate_prices_frame(prices)
    
    def test_long_format_duplicate_date_ticker_rejected(self):
        """Test duplicate (date, ticker) pairs raise ValueError."""
        dates = pd.to_datetime([
            "2023-01-02", "2023-01-02",  # Same date
        ])
        prices = pd.DataFrame({
            "ticker": ["AAPL", "AAPL"],  # Same ticker -> duplicate pair
            "close": [100, 101],
        }, index=dates)
        
        with pytest.raises(ValueError, match="Duplicate.*date.*ticker"):
            validate_prices_frame(prices)
    
    def test_long_format_unsorted_rejected(self):
        """Test unsorted long format raises ValueError."""
        dates = pd.to_datetime([
            "2023-01-03",  # Out of order
            "2023-01-02",
        ])
        prices = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "close": [101, 200],
        }, index=dates)
        
        with pytest.raises(ValueError, match="sorted"):
            validate_prices_frame(prices)


class TestLongWideEquivalence:
    """Test that long and wide format produce same backtest results."""
    
    def test_long_wide_equivalence(self):
        """Test backtest results identical for long vs wide format."""
        # Create wide format
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices_wide = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 105, 10),
        }, index=dates)
        
        # Create equivalent long format
        long_records = []
        for dt in dates:
            long_records.append({"date": dt, "ticker": "A", "close": prices_wide.loc[dt, "A"]})
            long_records.append({"date": dt, "ticker": "B", "close": prices_wide.loc[dt, "B"]})
        
        prices_long = pd.DataFrame(long_records)
        prices_long = prices_long.set_index("date").sort_index()
        
        # Same weights for both
        signal_dates = pd.to_datetime(["2023-01-02", "2023-01-09"])
        weights = pd.DataFrame({
            "A": [0.6, 0.4],
            "B": [0.4, 0.6],
        }, index=signal_dates)
        
        result_wide = run_backtest(prices_wide, weights, execution_lag_days=1)
        result_long = run_backtest(prices_long, weights, execution_lag_days=1, price_col="close")
        
        # Equity curves should be numerically equal
        # Use check_names=False and check_freq=False because pivot may change index metadata
        pd.testing.assert_series_equal(
            result_wide["equity_curve"],
            result_long["equity_curve"],
            check_names=False,
            check_freq=False,
            atol=1e-10
        )
        pd.testing.assert_series_equal(
            result_wide["daily_returns"],
            result_long["daily_returns"],
            check_names=False,
            check_freq=False,
            atol=1e-10
        )


class TestExecutionLagGuard:
    """Test execution_lag_days edge guards."""
    
    def test_negative_execution_lag_align_raises(self):
        """Test align_and_lag_weights with negative lag raises ValueError."""
        trading_days = pd.to_datetime(["2023-01-02", "2023-01-03"])
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({"A": [1.0]}, index=signal_dates)
        
        with pytest.raises(ValueError, match="execution_lag_days must be >= 0"):
            align_and_lag_weights(weights, trading_days, execution_lag_days=-1)
    
    def test_negative_execution_lag_backtest_raises(self):
        """Test run_backtest with negative lag raises ValueError."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
        }, index=dates)
        
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({"A": [1.0]}, index=signal_dates)
        
        with pytest.raises(ValueError, match="execution_lag_days must be >= 0"):
            run_backtest(prices, weights, execution_lag_days=-1)
    
    def test_zero_execution_lag_allowed(self):
        """Test zero execution lag is valid (same-day execution)."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
        }, index=dates)
        
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({"A": [1.0]}, index=signal_dates)
        
        # Should not raise
        result = run_backtest(prices, weights, execution_lag_days=0)
        
        # With lag=0, first rebalance is on signal date itself
        assert result["rebalance_dates"][0] == pd.Timestamp("2023-01-02")


class TestMissingPricesFailFast:
    """Test fail-fast on missing prices (no NaN masking)."""
    
    def test_long_format_missing_observation_raises(self):
        """Test long format with missing (date, ticker) observation fails."""
        # Create long format with 2 tickers, but one is missing on one date
        # Date 2 (Jan 4): only A, no B
        dates = pd.to_datetime([
            "2023-01-02", "2023-01-02",  # Both A and B
            "2023-01-03", "2023-01-03",  # Both A and B
            "2023-01-04",                # Only A, B is missing!
            "2023-01-05", "2023-01-05",  # Both A and B
            "2023-01-06", "2023-01-06",  # Both A and B
        ])
        prices = pd.DataFrame({
            "ticker": ["A", "B", "A", "B", "A", "A", "B", "A", "B"],
            "close": [100, 200, 101, 201, 102, 103, 203, 104, 204],
        }, index=dates)
        
        # Weights use both A and B
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [0.5],
            "B": [0.5],  # B is used, but missing on Jan 4
        }, index=signal_dates)
        
        with pytest.raises(ValueError, match="Missing prices"):
            run_backtest(prices, weights, execution_lag_days=1, price_col="close")
    
    def test_wide_format_nan_price_raises(self):
        """Test wide format with NaN price in backtest window raises."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        price_values = np.linspace(100, 110, 10)
        price_values[5] = np.nan  # NaN in the middle
        
        prices = pd.DataFrame({
            "A": price_values,
            "B": np.linspace(100, 110, 10),  # B is fine
        }, index=dates)
        
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [1.0],  # Uses A which has NaN
            "B": [0.0],
        }, index=signal_dates)
        
        # validation should catch the NaN in prices
        with pytest.raises(ValueError, match="finite"):
            run_backtest(prices, weights, execution_lag_days=1)
    
    def test_complete_data_passes(self):
        """Test complete data with no NaN passes."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 105, 10),
        }, index=dates)
        
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [0.5],
            "B": [0.5],
        }, index=signal_dates)
        
        # Should not raise
        result = run_backtest(prices, weights, execution_lag_days=1)
        assert len(result["equity_curve"]) > 0


class TestUsedTickersRegression:
    """Regression tests for used_tickers selection fix (KeyError prevention)."""
    
    def test_extra_ticker_in_prices_does_not_crash(self):
        """Test prices with extra ticker not in weights does not crash."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 105, 10),
            "C": np.linspace(100, 108, 10),  # C is extra, not in weights
        }, index=dates)
        
        # Weights only use A and B
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [0.5],
            "B": [0.5],
        }, index=signal_dates)
        
        # Should NOT crash (would crash with old target_weights[t] logic)
        result = run_backtest(prices, weights, execution_lag_days=1)
        assert len(result["equity_curve"]) > 0
        assert result["metrics"]["cagr"] is not None
    
    def test_unused_ticker_with_nan_passes(self):
        """Test unused ticker with NaN does not fail and produces valid numeric output.
        
        Use long format where C is missing on one date (creates NaN after pivot)
        but C is not used in weights.
        
        Task 9.2.1: Also verify that equity_curve and daily_returns have no NaN/inf.
        """
        # Build long format: A and B complete, C missing on one date
        dates = pd.to_datetime([
            "2023-01-02", "2023-01-02", "2023-01-02",  # A, B, C
            "2023-01-03", "2023-01-03", "2023-01-03",  # A, B, C
            "2023-01-04", "2023-01-04",                # A, B only (C missing!)
            "2023-01-05", "2023-01-05", "2023-01-05",  # A, B, C
            "2023-01-06", "2023-01-06", "2023-01-06",  # A, B, C
        ])
        prices = pd.DataFrame({
            "ticker": ["A", "B", "C", "A", "B", "C", "A", "B", "A", "B", "C", "A", "B", "C"],
            "close": [100, 200, 300, 101, 201, 301, 102, 202, 103, 203, 303, 104, 204, 304],
        }, index=dates)
        
        # Weights only use A and B, never C
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [0.5],
            "B": [0.5],
            # C not included -> effective_weights[C] will be 0
        }, index=signal_dates)
        
        # Should PASS (C is not used, so NaN in C doesn't matter)
        result = run_backtest(prices, weights, execution_lag_days=1, price_col="close")
        
        # Basic check: we got results
        assert len(result["equity_curve"]) > 0
        
        # Task 9.2.1: Numeric sanity - no NaN or inf allowed in output
        equity_curve = result["equity_curve"]
        daily_returns = result["daily_returns"]
        
        assert np.all(np.isfinite(equity_curve.values)), (
            f"equity_curve contains NaN or inf: {equity_curve.values}"
        )
        assert np.all(np.isfinite(daily_returns.values)), (
            f"daily_returns contains NaN or inf: {daily_returns.values}"
        )
    
    def test_used_ticker_with_nan_fails(self):
        """Test used ticker with NaN in backtest window raises ValueError.
        
        Use long format where A is missing on one date within backtest window.
        """
        # Build long format: A missing on Jan 4, B complete
        dates = pd.to_datetime([
            "2023-01-02", "2023-01-02",  # A, B
            "2023-01-03", "2023-01-03",  # A, B
            "2023-01-04",                # B only (A missing!)
            "2023-01-05", "2023-01-05",  # A, B
            "2023-01-06", "2023-01-06",  # A, B
        ])
        prices = pd.DataFrame({
            "ticker": ["A", "B", "A", "B", "B", "A", "B", "A", "B"],
            "close": [100, 200, 101, 201, 202, 103, 203, 104, 204],
        }, index=dates)
        
        signal_dates = pd.to_datetime(["2023-01-02"])
        weights = pd.DataFrame({
            "A": [0.5],  # A is used (non-zero weight)
            "B": [0.5],
        }, index=signal_dates)
        
        # Should FAIL (A is used and has NaN after pivot)
        with pytest.raises(ValueError, match="Missing prices"):
            run_backtest(prices, weights, execution_lag_days=1, price_col="close")


class TestForwardReturnSemantics:
    """Test forward return semantics (Task 9.2.0).
    
    Verify that weights effective on day t earn the return from t to t+1,
    NOT the return from t-1 to t (which would be look-ahead bias).
    """
    
    def test_rebalance_on_jump_day_does_not_capture_jump(self):
        """Test setting weights on the big-jump day does NOT capture that jump.
        
        Setup:
        - Prices: [100, 200, 200] on three business days (100→200 is 100% jump)
        - Rebalance to 100% weight on day 2 (the day prices jump TO 200)
        
        Expected behavior (forward return semantics):
        - Weight set on day 2, so we earn return from day 2 to day 3 = 0%
        - The 100% jump from day 1→2 should NOT be captured
        - Final equity should be ~1.0 (no gain, ignoring costs)
        """
        dates = pd.bdate_range("2023-01-02", periods=3)  # 3 business days
        prices = pd.DataFrame({
            "A": [100.0, 200.0, 200.0],  # 100% jump on day 2, then flat
        }, index=dates)
        
        # Signal on day 1 (first day), but execution_lag=1 means effective on day 2
        signal_dates = pd.to_datetime([dates[0]])  # Signal on day 1
        weights = pd.DataFrame({
            "A": [1.0],  # 100% weight
        }, index=signal_dates)
        
        result = run_backtest(
            prices, weights,
            execution_lag_days=1,  # Effective on day 2
            cost_bps=0, slippage_bps=0,
            initial_equity=1.0
        )
        
        # With forward return semantics:
        # - Weight becomes effective on day 2
        # - Forward return on day 2 = (200→200)/200 = 0%
        # - So equity should stay at 1.0
        final_equity = result["equity_curve"].iloc[-1]
        assert abs(final_equity - 1.0) < 1e-10, (
            f"Expected ~1.0 (no return captured), got {final_equity}. "
            "This suggests weights are capturing same-day return (look-ahead bias)."
        )
    
    def test_rebalance_before_jump_captures_jump(self):
        """Test setting weights BEFORE the big-jump day DOES capture that jump.
        
        Setup:
        - Prices: [100, 200, 200] on three business days (100→200 is 100% jump)
        - Rebalance to 100% weight on day 1 (the day BEFORE prices jump)
        
        Expected behavior (forward return semantics):
        - Weight set on day 1, so we earn return from day 1 to day 2 = 100%
        - The 100% jump IS captured
        - Final equity should be ~2.0 (doubled)
        """
        dates = pd.bdate_range("2023-01-02", periods=3)  # 3 business days
        prices = pd.DataFrame({
            "A": [100.0, 200.0, 200.0],  # 100% jump on day 2, then flat
        }, index=dates)
        
        # Signal on day 0 (before first trading day), effective immediately on day 1
        signal_dates = pd.to_datetime([dates[0]])  # Signal on day 1
        weights = pd.DataFrame({
            "A": [1.0],  # 100% weight
        }, index=signal_dates)
        
        result = run_backtest(
            prices, weights,
            execution_lag_days=0,  # Effective on day 1 (same day as signal)
            cost_bps=0, slippage_bps=0,
            initial_equity=1.0
        )
        
        # With forward return semantics:
        # - Weight becomes effective on day 1
        # - Forward return on day 1 = (200-100)/100 = 100%
        # - Forward return on day 2 = (200-200)/200 = 0%
        # - So equity should be 1.0 * 2.0 * 1.0 = 2.0
        final_equity = result["equity_curve"].iloc[-1]
        assert abs(final_equity - 2.0) < 1e-10, (
            f"Expected ~2.0 (100% return captured), got {final_equity}. "
            "Forward return semantics should capture the 100% jump."
        )
    
    def test_last_day_forward_return_is_zero(self):
        """Test that forward returns on the last day are zero (no t+1 available)."""
        dates = pd.bdate_range("2023-01-02", periods=5)
        prices = pd.DataFrame({
            "A": [100.0, 110.0, 121.0, 133.1, 1000.0],  # Price goes up, then big jump on day 5
        }, index=dates)
        
        # Weight effective from day 1
        signal_dates = pd.to_datetime([dates[0]])
        weights = pd.DataFrame({"A": [1.0]}, index=signal_dates)
        
        result = run_backtest(
            prices, weights,
            execution_lag_days=0,
            cost_bps=0, slippage_bps=0,
            initial_equity=1.0
        )
        
        # Forward return semantics:
        # Day 1: forward_return = (110-100)/100 = 10% -> equity = 1.1
        # Day 2: forward_return = (121-110)/110 = 10% -> equity = 1.21
        # Day 3: forward_return = (133.1-121)/121 = 10% -> equity = 1.331
        # Day 4: forward_return = (1000-133.1)/133.1 = 651% -> equity = 1.331 * 7.51...
        # Day 5: forward_return = 0 (no day 6) -> equity stays same
        #
        # Wait - the issue is day 5 is the LAST day, so:
        # forward_return[day5] = 0 (because there's no day 6 price)
        # BUT forward_return[day4] = (1000-133.1)/133.1 = 6.51... IS captured
        #
        # So equity at end = 1 * 1.1 * 1.1 * 1.1 * (1000/133.1) * 1.0
        # = 1.331 * 7.5131... = 10.0 (approximately)
        #
        # The test should verify that the FINAL day's 0% forward return works correctly
        # by checking daily_returns on the last day
        daily_rets = result["daily_returns"]
        last_day_return = daily_rets.iloc[-1]
        
        # Last day forward return should be 0 (no t+1 available)
        assert abs(last_day_return - 0.0) < 1e-10, (
            f"Expected last day forward return to be 0.0, got {last_day_return}. "
            "Last day should have zero forward return."
        )


class TestBacktestSemanticsLock:
    """Regression tests to lock exact backtest semantics.
    
    These tests use deterministic inputs and verify exact output values,
    ensuring any optimization does not change numerical results.
    """
    
    def test_exact_equity_curve_values(self):
        """Test exact equity curve values match expected computation.
        
        Uses a small deterministic price panel and weights,
        manually computes expected equity, and asserts exact match.
        """
        # Deterministic setup: 3 tickers, 5 days
        dates = pd.bdate_range("2023-01-02", periods=5)
        prices = pd.DataFrame({
            "A": [100.0, 110.0, 121.0, 133.1, 146.41],  # +10% each day
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],   # flat
            "C": [100.0, 95.0, 90.25, 85.74, 81.45],    # -5% each day
        }, index=dates)
        
        # Weight 50% A, 50% B from day 1 (no C exposure)
        signal_dates = pd.to_datetime([dates[0]])
        weights = pd.DataFrame({
            "A": [0.5],
            "B": [0.5],
            "C": [0.0],
        }, index=signal_dates)
        
        result = run_backtest(
            prices, weights,
            execution_lag_days=0,
            cost_bps=0, slippage_bps=0,
            initial_equity=1.0
        )
        
        # Manual forward return computation:
        # Day 1: forward_return_A = (110-100)/100 = 10%, forward_return_B = 0%
        #        port_return = 0.5*0.10 + 0.5*0.0 = 5%
        #        equity = 1.0 * 1.05 = 1.05
        # Day 2: forward_return_A = (121-110)/110 = 10%, forward_return_B = 0%
        #        port_return = 5% -> equity = 1.05 * 1.05 = 1.1025
        # Day 3: forward_return_A = (133.1-121)/121 = 10%, forward_return_B = 0%
        #        port_return = 5% -> equity = 1.1025 * 1.05 = 1.157625
        # Day 4: forward_return_A = (146.41-133.1)/133.1 = 10%, forward_return_B = 0%
        #        port_return = 5% -> equity = 1.157625 * 1.05 = 1.21550625
        # Day 5: forward_return = 0 (last day) -> equity stays 1.21550625
        
        expected_equity = [1.05, 1.1025, 1.157625, 1.21550625, 1.21550625]
        
        actual_equity = result["equity_curve"].values
        np.testing.assert_allclose(actual_equity, expected_equity, rtol=1e-9, 
            err_msg="Equity curve values do not match expected computation")
    
    def test_manual_reference_computation_matches(self):
        """Compare backtest output vs manual dot-product computation.
        
        This is a regression test that will fail if dot-product semantics change.
        """
        dates = pd.bdate_range("2023-01-02", periods=4)
        prices = pd.DataFrame({
            "X": [100.0, 120.0, 108.0, 118.8],  # +20%, -10%, +10%
            "Y": [100.0, 90.0, 99.0, 94.05],    # -10%, +10%, -5%
        }, index=dates)
        
        signal_dates = pd.to_datetime([dates[0]])
        weights = pd.DataFrame({
            "X": [0.6],
            "Y": [0.4],
        }, index=signal_dates)
        
        result = run_backtest(
            prices, weights,
            execution_lag_days=0,
            cost_bps=0, slippage_bps=0,
            initial_equity=1.0
        )
        
        # Manual reference computation:
        # Forward returns:
        #   forward_X = [0.20, -0.10, 0.10, 0.0]  (last is 0)
        #   forward_Y = [-0.10, 0.10, -0.05, 0.0]
        # Portfolio returns (0.6*X + 0.4*Y):
        #   Day 1: 0.6*0.20 + 0.4*(-0.10) = 0.12 - 0.04 = 0.08
        #   Day 2: 0.6*(-0.10) + 0.4*0.10 = -0.06 + 0.04 = -0.02
        #   Day 3: 0.6*0.10 + 0.4*(-0.05) = 0.06 - 0.02 = 0.04
        #   Day 4: 0.6*0.0 + 0.4*0.0 = 0.0
        # Equity: 1.0 * 1.08 * 0.98 * 1.04 * 1.0 = 1.100736
        
        expected_final = 1.0 * 1.08 * 0.98 * 1.04 * 1.0
        actual_final = result["equity_curve"].iloc[-1]
        
        assert abs(actual_final - expected_final) < 1e-9, (
            f"Expected final equity {expected_final}, got {actual_final}"
        )
        
        # Also check daily returns
        expected_daily_returns = [0.08, -0.02, 0.04, 0.0]
        actual_daily_returns = result["daily_returns"].values
        np.testing.assert_allclose(actual_daily_returns, expected_daily_returns, rtol=1e-9)
    
    def test_costs_reduce_equity_at_rebalance(self):
        """Test costs are applied correctly at rebalance and reduce equity."""
        dates = pd.bdate_range("2023-01-02", periods=5)
        prices = pd.DataFrame({
            "A": [100.0, 110.0, 121.0, 133.1, 146.41],
        }, index=dates)
        
        signal_dates = pd.to_datetime([dates[0]])
        weights = pd.DataFrame({"A": [1.0]}, index=signal_dates)
        
        # With 100 bps total cost on 100% turnover = 1% cost
        result = run_backtest(
            prices, weights,
            execution_lag_days=0,
            cost_bps=50, slippage_bps=50,  # 100 bps total
            initial_equity=1.0
        )
        
        # Turnover at first rebalance = 1.0 (0 -> 100%)
        # Cost = 1.0 * 1.0 * 100/10000 = 0.01
        # Equity after cost = 1.0 - 0.01 = 0.99
        # Then forward returns apply
        
        # Check first day equity reflects cost
        first_day_equity = result["equity_curve"].iloc[0]
        # After 1% cost, then +10% forward return: 0.99 * 1.10 = 1.089
        expected_first = 0.99 * 1.10
        assert abs(first_day_equity - expected_first) < 1e-9, (
            f"Expected first day equity {expected_first}, got {first_day_equity}"
        )


class TestMetricsContract:
    """Test metrics contract and deprecation warnings (Task 9.2.4)."""
    
    def test_cagr_over_vol_exists_and_equals_sharpe(self):
        """Test cagr_over_vol metric exists and equals sharpe (for backward compat)."""
        dates = pd.bdate_range("2023-01-02", periods=60)
        np.random.seed(42)
        prices = pd.DataFrame({
            "A": 100.0 * np.cumprod(1 + np.random.randn(60) * 0.01),
            "B": 100.0 * np.cumprod(1 + np.random.randn(60) * 0.01),
        }, index=dates)
        
        signal_dates = pd.to_datetime([dates[0]])
        weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=signal_dates)
        
        result = run_backtest(prices, weights, execution_lag_days=0)
        
        metrics = result["metrics"]
        
        # cagr_over_vol exists
        assert "cagr_over_vol" in metrics, "cagr_over_vol metric missing"
        
        # sharpe still exists (backward compat)
        assert "sharpe" in metrics, "sharpe metric missing (backward compat)"
        
        # They are equal
        assert metrics["cagr_over_vol"] == metrics["sharpe"], (
            f"cagr_over_vol ({metrics['cagr_over_vol']}) != sharpe ({metrics['sharpe']})"
        )
    
    def test_deprecation_warning_present(self):
        """Test deprecation warning for sharpe metric is present."""
        dates = pd.bdate_range("2023-01-02", periods=60)
        np.random.seed(42)
        prices = pd.DataFrame({
            "A": 100.0 * np.cumprod(1 + np.random.randn(60) * 0.01),
        }, index=dates)
        
        signal_dates = pd.to_datetime([dates[0]])
        weights = pd.DataFrame({"A": [1.0]}, index=signal_dates)
        
        result = run_backtest(prices, weights, execution_lag_days=0)
        
        # warnings field exists
        assert "warnings" in result, "warnings field missing from result"
        
        warnings = result["warnings"]
        expected_warning = "METRIC_DEPRECATED:sharpe:use cagr_over_vol (sharpe here is CAGR/AnnVol, not standard Sharpe)."
        
        assert expected_warning in warnings, (
            f"Expected deprecation warning not found. Got: {warnings}"
        )





