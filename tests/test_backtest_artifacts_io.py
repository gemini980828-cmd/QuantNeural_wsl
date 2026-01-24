"""
Tests for src/backtest_artifacts_io.py

Covers:
- Write scores CSV and round-trip smoke
- Write prices wide round-trip into runner
- Write prices long round-trip equivalence
- Fail-fast on invalid scores
- Fail-fast on invalid prices
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest_artifacts_io import write_scores_csv, write_prices_csv
from src.run_scores_backtest_from_csv import run_scores_backtest_from_csv


def _make_wide_prices_df(n_days=60, tickers=None, seed=42):
    """Create a valid wide prices DataFrame."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    
    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    
    np.random.seed(seed)
    data = {}
    for i, ticker in enumerate(tickers):
        base = 100 + i * 10
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices
    
    return pd.DataFrame(data, index=dates)


def _make_long_prices_df(n_days=60, tickers=None, seed=42):
    """Create a valid long prices DataFrame."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    
    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    
    np.random.seed(seed)
    rows = []
    for i, ticker in enumerate(tickers):
        base = 100 + i * 10
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        for d, p in zip(dates, prices):
            rows.append({"ticker": ticker, "close": p})
    
    # Create with DatetimeIndex
    all_dates = list(dates) * len(tickers)
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(all_dates)
    return df


def _make_scores_df(dates, tickers, n_signals=6, seed=123):
    """Create a valid scores DataFrame."""
    # Pick signal dates
    if len(dates) > n_signals * 10:
        signal_dates = list(dates[::10][:n_signals])
    else:
        signal_dates = list(dates[::max(1, len(dates) // n_signals)][:n_signals])
    
    np.random.seed(seed)
    data = {}
    for ticker in tickers:
        data[ticker] = np.random.randn(len(signal_dates))
    
    return pd.DataFrame(data, index=pd.DatetimeIndex(signal_dates))


class TestWriteScoresCsvRoundtrip:
    """Test write_scores_csv and read back."""
    
    def test_write_scores_csv_and_roundtrip_smoke(self, tmp_path):
        """Test writing and reading back scores CSV."""
        tickers = ["XXX", "YYY", "ZZZ"]
        dates = pd.bdate_range("2023-01-01", periods=30)
        scores = _make_scores_df(dates, tickers, n_signals=6)
        
        path = tmp_path / "scores.csv"
        write_scores_csv(scores, path=str(path), date_col="date")
        
        # Read back
        df = pd.read_csv(path)
        
        # Verify date_col exists
        assert "date" in df.columns
        
        # Verify columns match tickers
        assert set(tickers).issubset(set(df.columns))
        
        # Verify unique dates
        dates_read = pd.to_datetime(df["date"])
        assert dates_read.is_unique
        
        # Verify sorted
        assert dates_read.is_monotonic_increasing


class TestWritePricesWideRoundtripRunner:
    """Test writing wide prices and running through CSV runner."""
    
    def test_write_prices_wide_roundtrip_into_runner(self, tmp_path):
        """Test wide prices CSV round-trip into backtest runner."""
        tickers = ["PPP", "QQQ", "RRR"]
        
        # Create DataFrames
        prices_df = _make_wide_prices_df(n_days=60, tickers=tickers)
        scores_df = _make_scores_df(prices_df.index, tickers, n_signals=6)
        
        # Write CSVs
        prices_path = tmp_path / "prices.csv"
        scores_path = tmp_path / "scores.csv"
        
        write_prices_csv(prices_df, path=str(prices_path), format="wide", date_col="date")
        write_scores_csv(scores_df, path=str(scores_path), date_col="date")
        
        # Run backtest
        result = run_scores_backtest_from_csv(
            prices_csv_path=str(prices_path),
            scores_csv_path=str(scores_path),
            date_col="date",
            rebalance="M",
        )
        
        # Verify 9 keys
        required_keys = [
            "equity_curve", "daily_returns", "rebalance_dates",
            "weights_used", "turnover", "costs", "trades",
            "metrics", "target_weights"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestWritePricesLongRoundtripEquivalence:
    """Test long prices produces same result as wide."""
    
    def test_write_prices_long_roundtrip_equivalence(self, tmp_path):
        """Test long and wide prices produce identical results."""
        tickers = ["LLL", "MMM", "NNN"]
        
        # Create same underlying prices
        wide_df = _make_wide_prices_df(n_days=60, tickers=tickers, seed=42)
        
        # Create long version manually (same data)
        long_rows = []
        for date in wide_df.index:
            for ticker in tickers:
                long_rows.append({
                    "date": date,
                    "ticker": ticker,
                    "close": wide_df.loc[date, ticker]
                })
        long_df = pd.DataFrame(long_rows)
        long_df = long_df.set_index("date")
        
        # Create scores
        scores_df = _make_scores_df(wide_df.index, tickers, n_signals=6)
        
        # Write CSVs
        prices_wide_path = tmp_path / "prices_wide.csv"
        prices_long_path = tmp_path / "prices_long.csv"
        scores_path = tmp_path / "scores.csv"
        
        write_prices_csv(wide_df, path=str(prices_wide_path), format="wide", date_col="date")
        write_prices_csv(long_df, path=str(prices_long_path), format="long", 
                         date_col="date", ticker_col="ticker", price_col="close")
        write_scores_csv(scores_df, path=str(scores_path), date_col="date")
        
        # Run both
        result_wide = run_scores_backtest_from_csv(
            prices_csv_path=str(prices_wide_path),
            scores_csv_path=str(scores_path),
            date_col="date",
            ticker_col="ticker",
            price_col="close",
            rebalance="M",
        )
        
        result_long = run_scores_backtest_from_csv(
            prices_csv_path=str(prices_long_path),
            scores_csv_path=str(scores_path),
            date_col="date",
            ticker_col="ticker",
            price_col="close",
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


class TestFailFastInvalidScores:
    """Test fail-fast on invalid scores."""
    
    def test_scores_with_nan_raises(self, tmp_path):
        """Test scores with NaN raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [2.0, 1.0, 0.0],
        }, index=pd.bdate_range("2023-01-01", periods=3))
        
        path = tmp_path / "scores.csv"
        with pytest.raises(ValueError, match="finite"):
            write_scores_csv(scores, path=str(path))
    
    def test_scores_with_duplicate_dates_raises(self, tmp_path):
        """Test scores with duplicate dates raises ValueError."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-03"])
        scores = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 1.0, 0.0],
        }, index=dates)
        
        path = tmp_path / "scores.csv"
        with pytest.raises(ValueError, match="unique"):
            write_scores_csv(scores, path=str(path))
    
    def test_scores_with_less_than_2_assets_raises(self, tmp_path):
        """Test scores with < 2 assets raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
        }, index=pd.bdate_range("2023-01-01", periods=3))
        
        path = tmp_path / "scores.csv"
        with pytest.raises(ValueError, match="k_assets >= 2"):
            write_scores_csv(scores, path=str(path))


class TestFailFastInvalidPrices:
    """Test fail-fast on invalid prices."""
    
    def test_wide_prices_with_zero_raises(self, tmp_path):
        """Test wide prices with zero value raises ValueError."""
        prices = pd.DataFrame({
            "A": [100.0, 0.0, 102.0],  # Zero!
            "B": [110.0, 111.0, 112.0],
        }, index=pd.bdate_range("2023-01-01", periods=3))
        
        path = tmp_path / "prices.csv"
        with pytest.raises(ValueError, match="positive"):
            write_prices_csv(prices, path=str(path), format="wide")
    
    def test_wide_prices_with_nan_raises(self, tmp_path):
        """Test wide prices with NaN raises ValueError."""
        prices = pd.DataFrame({
            "A": [100.0, np.nan, 102.0],
            "B": [110.0, 111.0, 112.0],
        }, index=pd.bdate_range("2023-01-01", periods=3))
        
        path = tmp_path / "prices.csv"
        with pytest.raises(ValueError, match="finite"):
            write_prices_csv(prices, path=str(path), format="wide")
    
    def test_long_prices_with_duplicate_date_ticker_raises(self, tmp_path):
        """Test long prices with duplicate (date, ticker) raises ValueError."""
        dates = pd.to_datetime([
            "2023-01-01", "2023-01-01",  # dup date
            "2023-01-02", "2023-01-02",
        ])
        prices = pd.DataFrame({
            "ticker": ["A", "A", "A", "B"],  # dup (2023-01-01, A)
            "close": [100.0, 101.0, 102.0, 110.0],
        }, index=dates)
        
        path = tmp_path / "prices.csv"
        with pytest.raises(ValueError, match="unique"):
            write_prices_csv(prices, path=str(path), format="long")
    
    def test_long_prices_with_negative_raises(self, tmp_path):
        """Test long prices with negative value raises ValueError."""
        dates = pd.bdate_range("2023-01-01", periods=4)
        prices = pd.DataFrame({
            "ticker": ["A", "B", "A", "B"],
            "close": [100.0, -110.0, 102.0, 112.0],  # Negative!
        }, index=dates)
        
        path = tmp_path / "prices.csv"
        with pytest.raises(ValueError, match="positive"):
            write_prices_csv(prices, path=str(path), format="long")


class TestAutoFormatDetection:
    """Test auto format detection."""
    
    def test_auto_detects_long_format(self, tmp_path):
        """Test auto format detects long when ticker and close columns exist."""
        dates = pd.bdate_range("2023-01-01", periods=4)
        prices = pd.DataFrame({
            "ticker": ["A", "B", "A", "B"],
            "close": [100.0, 110.0, 102.0, 112.0],
        }, index=dates)
        
        path = tmp_path / "prices.csv"
        write_prices_csv(prices, path=str(path), format="auto")
        
        # Read back and verify long format columns
        df = pd.read_csv(path)
        assert "date" in df.columns
        assert "ticker" in df.columns
        assert "close" in df.columns
    
    def test_auto_detects_wide_format(self, tmp_path):
        """Test auto format detects wide when no ticker column."""
        prices = pd.DataFrame({
            "A": [100.0, 101.0, 102.0],
            "B": [110.0, 111.0, 112.0],
        }, index=pd.bdate_range("2023-01-01", periods=3))
        
        path = tmp_path / "prices.csv"
        write_prices_csv(prices, path=str(path), format="auto")
        
        # Read back and verify wide format columns
        df = pd.read_csv(path)
        assert "date" in df.columns
        assert "A" in df.columns
        assert "B" in df.columns
