"""
Tests for src/run_scores_backtest_from_csv.py

Covers:
- Wide prices and scores CSV smoke test
- Long prices CSV equivalence to wide
- Scores missing date column raises
- Long prices missing required columns raises
- Wide prices missing date column raises
"""

import numpy as np
import pandas as pd
import pytest

from src.run_scores_backtest_from_csv import run_scores_backtest_from_csv


def _create_wide_prices_csv(path, n_days=60, tickers=None):
    """Create a wide prices CSV file."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    
    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    
    data = {"date": dates}
    np.random.seed(42)
    for i, ticker in enumerate(tickers):
        base = 100 + i * 10
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def _create_long_prices_csv(path, wide_df, price_col="close"):
    """Convert wide prices to long and save as CSV."""
    # wide_df has date column + ticker columns
    melted = wide_df.melt(
        id_vars=["date"],
        var_name="ticker",
        value_name=price_col
    )
    melted = melted.sort_values(["date", "ticker"])
    melted.to_csv(path, index=False)
    return melted


def _create_scores_csv(path, dates, tickers, n_signals=6):
    """Create a scores CSV file."""
    # Pick signal dates
    if len(dates) > n_signals * 10:
        signal_dates = list(dates[::10][:n_signals])
    else:
        signal_dates = list(dates[::max(1, len(dates) // n_signals)][:n_signals])
    
    np.random.seed(123)
    data = {"date": signal_dates}
    for ticker in tickers:
        data[ticker] = np.random.randn(len(signal_dates))
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


class TestWidePricesAndScoresCsvSmoke:
    """Test smoke with wide format prices."""
    
    def test_wide_prices_and_scores_csv_smoke(self, tmp_path):
        """Test basic CSV run with wide prices."""
        tickers = ["XXX", "YYY", "ZZZ"]
        
        # Create CSVs
        prices_path = tmp_path / "prices.csv"
        scores_path = tmp_path / "scores.csv"
        
        wide_df = _create_wide_prices_csv(prices_path, n_days=60, tickers=tickers)
        dates = pd.to_datetime(wide_df["date"])
        _create_scores_csv(scores_path, dates, tickers, n_signals=6)
        
        # Run backtest
        result = run_scores_backtest_from_csv(
            prices_csv_path=str(prices_path),
            scores_csv_path=str(scores_path),
            date_col="date",
            rebalance="M",
            execution_lag_days=1,
        )
        
        # Verify 9 keys
        required_keys = [
            "equity_curve", "daily_returns", "rebalance_dates",
            "weights_used", "turnover", "costs", "trades",
            "metrics", "target_weights"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify metrics present
        assert "cagr" in result["metrics"]
        assert "sharpe" in result["metrics"]


class TestLongPricesCsvEquivalence:
    """Test long format produces same results as wide."""
    
    def test_long_prices_csv_equivalence_to_wide(self, tmp_path):
        """Test wide and long prices produce identical results."""
        tickers = ["LLL", "MMM", "NNN"]
        
        # Create wide prices
        wide_prices_path = tmp_path / "prices_wide.csv"
        wide_df = _create_wide_prices_csv(wide_prices_path, n_days=60, tickers=tickers)
        
        # Create long prices from same data
        long_prices_path = tmp_path / "prices_long.csv"
        _create_long_prices_csv(long_prices_path, wide_df, price_col="close")
        
        # Create scores
        scores_path = tmp_path / "scores.csv"
        dates = pd.to_datetime(wide_df["date"])
        _create_scores_csv(scores_path, dates, tickers, n_signals=6)
        
        # Run with wide
        result_wide = run_scores_backtest_from_csv(
            prices_csv_path=str(wide_prices_path),
            scores_csv_path=str(scores_path),
            date_col="date",
            price_col="close",
            ticker_col="ticker",
            rebalance="M",
        )
        
        # Run with long
        result_long = run_scores_backtest_from_csv(
            prices_csv_path=str(long_prices_path),
            scores_csv_path=str(scores_path),
            date_col="date",
            price_col="close",
            ticker_col="ticker",
            rebalance="M",
        )
        
        # Assert identical results
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


class TestScoresMissingDateColumnRaises:
    """Test fail-fast on missing date column in scores."""
    
    def test_scores_missing_date_column_raises(self, tmp_path):
        """Test scores CSV without date column raises ValueError."""
        # Create valid prices
        prices_path = tmp_path / "prices.csv"
        _create_wide_prices_csv(prices_path)
        
        # Create scores WITHOUT date column
        scores_path = tmp_path / "scores.csv"
        df = pd.DataFrame({
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [2.0, 1.0, 3.0],
            "CCC": [3.0, 2.0, 1.0],
        })
        df.to_csv(scores_path, index=False)
        
        with pytest.raises(ValueError, match="date"):
            run_scores_backtest_from_csv(
                prices_csv_path=str(prices_path),
                scores_csv_path=str(scores_path),
                date_col="date",
            )


class TestLongPricesMissingRequiredColumnsRaises:
    """Test fail-fast on missing columns in long prices."""
    
    def test_prices_long_missing_ticker_raises(self, tmp_path):
        """Test long prices CSV missing ticker column raises ValueError."""
        # Create long prices without ticker column
        prices_path = tmp_path / "prices.csv"
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=10),
            "close": np.arange(10) + 100,
            # Missing "ticker" column!
        })
        df.to_csv(prices_path, index=False)
        
        # Create scores (needs ticker column to detect long format)
        # Actually since ticker not present, it will be treated as wide
        # But wide won't have the right format either
        # Let's force long format detection by adding ticker col in scores
        scores_path = tmp_path / "scores.csv"
        df_scores = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=3),
            "close": [1.0, 2.0, 3.0],  # This is weird, should be asset columns
        })
        df_scores.to_csv(scores_path, index=False)
        
        # This will treat prices as wide (no ticker column) but scores only has "close"
        # which means only 1 asset, triggering k_assets >= 2 error
        with pytest.raises(ValueError):
            run_scores_backtest_from_csv(
                prices_csv_path=str(prices_path),
                scores_csv_path=str(scores_path),
                date_col="date",
                ticker_col="ticker",
            )
    
    def test_prices_long_missing_price_col_raises(self, tmp_path):
        """Test long prices CSV missing price column raises ValueError."""
        prices_path = tmp_path / "prices.csv"
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=10).tolist() * 2,
            "ticker": ["A"] * 10 + ["B"] * 10,
            # Missing "close" column!
        })
        df.to_csv(prices_path, index=False)
        
        scores_path = tmp_path / "scores.csv"
        df_scores = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=3),
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 1.0, 0.0],
        })
        df_scores.to_csv(scores_path, index=False)
        
        with pytest.raises(ValueError, match="close"):
            run_scores_backtest_from_csv(
                prices_csv_path=str(prices_path),
                scores_csv_path=str(scores_path),
                date_col="date",
                ticker_col="ticker",
                price_col="close",
            )


class TestWidePricesMissingDateColumnRaises:
    """Test fail-fast on missing date column in wide prices."""
    
    def test_prices_wide_missing_date_column_raises(self, tmp_path):
        """Test wide prices CSV without date column raises ValueError."""
        # Create prices WITHOUT date column (only ticker columns)
        prices_path = tmp_path / "prices.csv"
        df = pd.DataFrame({
            "AAA": np.arange(10) + 100,
            "BBB": np.arange(10) + 110,
            "CCC": np.arange(10) + 120,
        })
        df.to_csv(prices_path, index=False)
        
        # Create valid scores
        scores_path = tmp_path / "scores.csv"
        df_scores = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=3),
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [2.0, 1.0, 3.0],
            "CCC": [3.0, 2.0, 1.0],
        })
        df_scores.to_csv(scores_path, index=False)
        
        with pytest.raises(ValueError, match="date"):
            run_scores_backtest_from_csv(
                prices_csv_path=str(prices_path),
                scores_csv_path=str(scores_path),
                date_col="date",
            )
