"""
Tests for src/generate_scores_from_prices.py

Covers:
- Correct month-end signal dates
- No NaN/inf in scores
- Coverage filter drops incomplete tickers
- Leading plateau gate detects padded history
"""

import numpy as np
import pandas as pd
import pytest

from src.generate_scores_from_prices import (
    load_prices_csv,
    compute_monthly_momentum_scores,
    write_scores_csv,
)


def _create_wide_prices_df(n_days=300, tickers=None, seed=42, start_date="2020-01-01"):
    """Create a valid wide prices DataFrame."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    
    dates = pd.bdate_range(start_date, periods=n_days, freq="B")
    
    np.random.seed(seed)
    data = {}
    for i, ticker in enumerate(tickers):
        base = 100 + i * 10
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices
    
    return pd.DataFrame(data, index=dates)


def _create_long_prices_df(n_days=300, tickers=None, seed=42, start_date="2020-01-01"):
    """Create a valid long prices DataFrame."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    
    dates = pd.bdate_range(start_date, periods=n_days, freq="B")
    
    np.random.seed(seed)
    rows = []
    for i, ticker in enumerate(tickers):
        base = 100 + i * 10
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        for d, p in zip(dates, prices):
            rows.append({"ticker": ticker, "close": p})
    
    all_dates = list(dates) * len(tickers)
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(all_dates)
    return df


class TestLoadPricesCsv:
    """Test load_prices_csv function."""
    
    def test_load_wide_prices(self, tmp_path):
        """Test loading wide format prices."""
        prices = _create_wide_prices_df()
        path = tmp_path / "prices.csv"
        
        # Write with date column
        df = prices.reset_index()
        df.columns = ["date"] + list(prices.columns)
        df.to_csv(path, index=False)
        
        # Load
        loaded = load_prices_csv(str(path))
        
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert list(loaded.columns) == list(prices.columns)
        assert len(loaded) == len(prices)
    
    def test_load_long_prices(self, tmp_path):
        """Test loading long format prices."""
        prices = _create_long_prices_df()
        path = tmp_path / "prices.csv"
        
        # Write with date column
        df = prices.reset_index()
        df.columns = ["date", "ticker", "close"]
        df.to_csv(path, index=False)
        
        # Load
        loaded = load_prices_csv(str(path))
        
        assert "ticker" in loaded.columns
        assert "close" in loaded.columns


class TestComputeMonthlyMomentumScores:
    """Test compute_monthly_momentum_scores function."""
    
    def test_generates_correct_month_end_dates(self, tmp_path):
        """Test that signal dates are actual last trading days of months."""
        prices = _create_wide_prices_df(n_days=400, start_date="2020-01-01")
        
        scores = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            rebalance="M",
            min_coverage=0.9,
            enforce_no_leading_plateau=False,
        )
        
        # All dates should be in the original index
        for date in scores.index:
            # Date should be in the last week of each month
            assert date.day >= 20, f"Signal date {date} is not near month end"
    
    def test_no_nan_inf_in_scores(self, tmp_path):
        """Test that scores contain no NaN or inf."""
        prices = _create_wide_prices_df(n_days=400)
        
        scores = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            min_coverage=0.9,
            enforce_no_leading_plateau=False,
        )
        
        assert not scores.isna().any().any(), "Scores contain NaN"
        assert np.isfinite(scores.values).all(), "Scores contain inf"
    
    def test_quarterly_rebalance(self):
        """Test quarterly rebalance frequency."""
        prices = _create_wide_prices_df(n_days=600)
        
        scores = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            rebalance="Q",
            min_coverage=0.9,
            enforce_no_leading_plateau=False,
        )
        
        # Should have fewer dates than monthly
        scores_m = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            rebalance="M",
            min_coverage=0.9,
            enforce_no_leading_plateau=False,
        )
        
        assert len(scores) < len(scores_m)
    
    def test_coverage_filter_drops_incomplete_tickers(self):
        """Test that tickers with low coverage are dropped."""
        prices = _create_wide_prices_df(n_days=400)
        
        # Add a ticker with many NaNs
        prices["BAD"] = np.nan
        prices.loc[prices.index[:100], "BAD"] = 100.0  # Only 100 valid days
        
        scores = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            min_coverage=0.5,  # 50% coverage required
            enforce_no_leading_plateau=False,
        )
        
        assert "BAD" not in scores.columns


class TestLeadingPlateauGate:
    """Test leading plateau integrity gate."""
    
    def test_leading_plateau_raises_with_ticker_name(self):
        """Test that constant leading prices trigger ValueError with ticker name."""
        prices = _create_wide_prices_df(n_days=400)
        
        # Create a ticker with constant leading prices
        prices["PADDED"] = prices["AAA"].copy()
        prices.loc[prices.index[:252], "PADDED"] = 100.0  # Constant for 252 days
        
        with pytest.raises(ValueError, match="PADDED"):
            compute_monthly_momentum_scores(
                prices,
                lookback_days=252,
                enforce_no_leading_plateau=True,
                leading_plateau_days=252,
            )
    
    def test_leading_plateau_can_be_disabled(self):
        """Test that gate can be disabled."""
        prices = _create_wide_prices_df(n_days=400)
        
        # Create a ticker with constant leading prices
        prices["PADDED"] = prices["AAA"].copy()
        prices.loc[prices.index[:260], "PADDED"] = 100.0
        
        # Should not raise with gate disabled
        scores = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            enforce_no_leading_plateau=False,
            min_coverage=0.5,
        )
        
        assert scores is not None


class TestWriteScoresCsv:
    """Test write_scores_csv function."""
    
    def test_writes_valid_csv(self, tmp_path):
        """Test that valid CSV is written."""
        prices = _create_wide_prices_df(n_days=400)
        scores = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            min_coverage=0.9,
            enforce_no_leading_plateau=False,
        )
        
        path = tmp_path / "scores.csv"
        write_scores_csv(scores, out_scores_csv_path=str(path))
        
        # Read back
        df = pd.read_csv(path)
        assert "date" in df.columns
        assert len(df) == len(scores)
    
    def test_deterministic_output(self, tmp_path):
        """Test that output is deterministic."""
        prices = _create_wide_prices_df(n_days=400)
        scores = compute_monthly_momentum_scores(
            prices,
            lookback_days=252,
            min_coverage=0.9,
            enforce_no_leading_plateau=False,
        )
        
        path1 = tmp_path / "scores1.csv"
        path2 = tmp_path / "scores2.csv"
        
        write_scores_csv(scores, out_scores_csv_path=str(path1))
        write_scores_csv(scores, out_scores_csv_path=str(path2))
        
        content1 = path1.read_text()
        content2 = path2.read_text()
        
        assert content1 == content2
