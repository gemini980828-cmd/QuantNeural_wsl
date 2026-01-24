"""
Tests for src/e2e_backtest.py

Smoke tests that validate integration and determinism for the end-to-end
scores → target_weights → backtest pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.e2e_backtest import run_scores_backtest


def _make_wide_prices(n_days: int = 60, tickers: list = None) -> pd.DataFrame:
    """Create a simple wide price panel."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG"]
    
    # Strictly increasing dates (business days)
    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    
    # Create positive prices that vary deterministically
    np.random.seed(42)
    data = {}
    for i, ticker in enumerate(tickers):
        # Start at 100 + i*10, with small daily changes
        base = 100 + i * 10
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices
    
    return pd.DataFrame(data, index=dates)


def _make_monthly_scores(
    price_index: pd.DatetimeIndex,
    tickers: list,
    n_signals: int = 6
) -> pd.DataFrame:
    """Create monthly-ish scores aligned to price calendar."""
    # Pick signal dates that are within the price range
    # Use month-end-ish dates
    signal_dates = []
    current_month = price_index[0].month
    for i, date in enumerate(price_index):
        if date.month != current_month:
            signal_dates.append(price_index[i - 1])
            current_month = date.month
        if len(signal_dates) >= n_signals:
            break
    
    # Ensure we have enough signals
    if len(signal_dates) < n_signals:
        # Just use every 10th date
        signal_dates = list(price_index[::10][:n_signals])
    
    # Create deterministic scores
    np.random.seed(123)
    scores_data = {}
    for ticker in tickers:
        scores_data[ticker] = np.random.randn(len(signal_dates))
    
    return pd.DataFrame(scores_data, index=signal_dates)


def _convert_to_long(wide_prices: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Convert wide price panel to long format."""
    long_df = wide_prices.stack().reset_index()
    long_df.columns = ["date", "ticker", price_col]
    long_df = long_df.set_index("date").sort_index()
    return long_df


class TestE2EWideSmokeAndKeys:
    """Test end-to-end with wide price format."""
    
    def test_e2e_wide_smoke_and_keys(self):
        """Test basic e2e flow with wide prices and verify all output keys."""
        # Build test data
        prices = _make_wide_prices(n_days=60, tickers=["A", "B", "C"])
        scores = _make_monthly_scores(prices.index, ["A", "B", "C"], n_signals=6)
        
        # Run e2e backtest
        result = run_scores_backtest(
            prices,
            scores,
            rebalance="M",
            execution_lag_days=1,
            method="softmax",
        )
        
        # Verify all 8 harness keys
        required_keys = [
            "equity_curve",
            "daily_returns",
            "rebalance_dates",
            "weights_used",
            "turnover",
            "costs",
            "trades",
            "metrics",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify target_weights added
        assert "target_weights" in result
        
        # Verify types
        assert isinstance(result["equity_curve"], pd.Series)
        assert isinstance(result["daily_returns"], pd.Series)
        assert isinstance(result["rebalance_dates"], pd.DatetimeIndex)
        assert isinstance(result["weights_used"], pd.DataFrame)
        assert isinstance(result["turnover"], pd.Series)
        assert isinstance(result["costs"], pd.Series)
        assert isinstance(result["trades"], pd.DataFrame)
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["target_weights"], pd.DataFrame)
        
        # Verify equity_curve index is subset of prices index
        assert all(d in prices.index for d in result["equity_curve"].index)
        
        # Verify metrics has expected keys
        assert "cagr" in result["metrics"]
        assert "sharpe" in result["metrics"]
        assert "max_drawdown" in result["metrics"]


class TestE2ELongSmokeEquivalence:
    """Test wide vs long format equivalence."""
    
    def test_e2e_long_smoke_equivalence_basic(self):
        """Test that wide and long price formats produce identical results."""
        # Build test data
        wide_prices = _make_wide_prices(n_days=60, tickers=["X", "Y", "Z"])
        scores = _make_monthly_scores(wide_prices.index, ["X", "Y", "Z"], n_signals=6)
        
        # Convert to long format
        long_prices = _convert_to_long(wide_prices)
        
        # Run both
        result_wide = run_scores_backtest(
            wide_prices,
            scores,
            rebalance="M",
            execution_lag_days=1,
        )
        
        result_long = run_scores_backtest(
            long_prices,
            scores,
            rebalance="M",
            execution_lag_days=1,
        )
        
        # Assert numerical equality
        pd.testing.assert_series_equal(
            result_wide["equity_curve"],
            result_long["equity_curve"],
            check_names=False,
            check_freq=False,
            atol=1e-10,
        )
        
        pd.testing.assert_series_equal(
            result_wide["daily_returns"],
            result_long["daily_returns"],
            check_names=False,
            check_freq=False,
            atol=1e-10,
        )


class TestE2EDeterminism:
    """Test determinism and repeatability."""
    
    def test_e2e_determinism_repeatability(self):
        """Test that repeated calls with identical inputs produce identical results."""
        # Build test data
        prices = _make_wide_prices(n_days=60, tickers=["D", "E", "F"])
        scores = _make_monthly_scores(prices.index, ["D", "E", "F"], n_signals=6)
        
        # Run twice
        result1 = run_scores_backtest(
            prices,
            scores,
            rebalance="M",
            execution_lag_days=1,
            method="rank",
        )
        
        result2 = run_scores_backtest(
            prices,
            scores,
            rebalance="M",
            execution_lag_days=1,
            method="rank",
        )
        
        # Assert exact equality for equity curve
        pd.testing.assert_series_equal(
            result1["equity_curve"],
            result2["equity_curve"],
        )
        
        # Assert metrics dict is identical
        for key in result1["metrics"]:
            assert result1["metrics"][key] == result2["metrics"][key], f"Metric {key} differs"


class TestE2EQuarterlyRebalance:
    """Test quarterly rebalance reduces rebalance count."""
    
    def test_e2e_quarterly_rebalance_reduces_rebalances(self):
        """Test that quarterly rebalance uses fewer rebalance dates than monthly."""
        # Build test data with 6+ signal dates
        prices = _make_wide_prices(n_days=120, tickers=["G", "H", "I"])
        scores = _make_monthly_scores(prices.index, ["G", "H", "I"], n_signals=9)
        
        # Run monthly
        result_m = run_scores_backtest(
            prices,
            scores,
            rebalance="M",
            execution_lag_days=1,
        )
        
        # Run quarterly
        result_q = run_scores_backtest(
            prices,
            scores,
            rebalance="Q",
            execution_lag_days=1,
        )
        
        # Quarterly should have fewer or equal rebalances
        assert len(result_q["rebalance_dates"]) <= len(result_m["rebalance_dates"])
        
        # And specifically, quarterly picks every 3rd (indices 0, 3, 6, ...)
        # So it should be roughly 1/3
        if len(result_m["rebalance_dates"]) >= 3:
            assert len(result_q["rebalance_dates"]) < len(result_m["rebalance_dates"])


class TestE2EFailFast:
    """Test fail-fast behavior."""
    
    def test_e2e_out_of_range_scores_fail_fast(self):
        """Test that scores after last trading day raises ValueError."""
        # Create prices ending in 2023
        prices = _make_wide_prices(n_days=30, tickers=["J", "K", "L"])
        last_price_date = prices.index[-1]
        
        # Create scores starting after prices end
        future_dates = pd.bdate_range(
            last_price_date + pd.Timedelta(days=30),
            periods=3,
            freq="B"
        )
        scores = pd.DataFrame({
            "J": [1.0, 2.0, 3.0],
            "K": [2.0, 1.0, 3.0],
            "L": [3.0, 2.0, 1.0],
        }, index=future_dates)
        
        # Should raise ValueError from harness
        with pytest.raises(ValueError):
            run_scores_backtest(prices, scores)


class TestE2EMethods:
    """Test different weight methods work."""
    
    def test_e2e_topk_method(self):
        """Test e2e with topk method."""
        prices = _make_wide_prices(n_days=60, tickers=["M", "N", "O", "P"])
        scores = _make_monthly_scores(prices.index, ["M", "N", "O", "P"], n_signals=4)
        
        result = run_scores_backtest(
            prices,
            scores,
            method="topk",
            top_k=2,
        )
        
        # Verify it ran
        assert len(result["equity_curve"]) > 0
        
        # Verify target_weights structure
        assert result["target_weights"].shape[1] == 4  # 4 tickers
    
    def test_e2e_with_max_weight(self):
        """Test e2e with max_weight cap."""
        prices = _make_wide_prices(n_days=60, tickers=["Q", "R", "S"])
        scores = _make_monthly_scores(prices.index, ["Q", "R", "S"], n_signals=4)
        
        result = run_scores_backtest(
            prices,
            scores,
            method="softmax",
            max_weight=0.5,
        )
        
        # Verify max_weight is respected in target_weights
        assert result["target_weights"].max().max() <= 0.5 + 1e-12
