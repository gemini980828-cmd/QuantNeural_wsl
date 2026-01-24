"""
Tests for src/stooq_prices.py

Covers:
- PIT cutoff enforcement (no future data)
- Schema validation (columns, dtypes)
- Duplicate handling (last wins deterministically)
- OHLC sanity validation
- Monthly resampling aggregation
"""

import numpy as np
import pandas as pd
import pytest

from src.stooq_prices import load_stooq_daily_prices, resample_to_monthly


class TestLoadStooqDailyPrices:
    """Tests for load_stooq_daily_prices function."""
    
    def test_load_parses_and_pit_cutoff(self, tmp_path):
        """Test basic parsing and PIT cutoff enforcement."""
        # Create test CSV with header and 5 rows, one after cutoff date
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,100.0,105.0,99.0,104.0,1000,0
AAPL.US,D,20230102,000000,104.0,106.0,103.0,105.0,1100,0
AAPL.US,D,20230103,000000,105.0,107.0,104.0,106.0,1200,0
AAPL.US,D,20230104,000000,106.0,108.0,105.0,107.0,1300,0
AAPL.US,D,20230110,000000,110.0,115.0,109.0,114.0,1500,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        # Load with cutoff that excludes last row
        as_of_date = "2023-01-05"
        df = load_stooq_daily_prices(str(csv_path), as_of_date=as_of_date)
        
        # Assert schema
        expected_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_cols
        
        # Assert all dates <= as_of_date
        cutoff_dt = pd.to_datetime(as_of_date)
        assert (df["date"] <= cutoff_dt).all()
        
        # Assert sorted by (ticker, date)
        assert df["ticker"].is_monotonic_increasing or len(df["ticker"].unique()) == 1
        for t in df["ticker"].unique():
            t_df = df[df["ticker"] == t]
            assert t_df["date"].is_monotonic_increasing
        
        # Assert dtypes
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["open"].dtype == np.float64
        assert df["high"].dtype == np.float64
        assert df["low"].dtype == np.float64
        assert df["close"].dtype == np.float64
        assert df["volume"].dtype == np.float64
        
        # Assert only 4 rows (last row excluded by cutoff)
        assert len(df) == 4
    
    def test_load_ticker_filter(self, tmp_path):
        """Test that ticker filter works correctly."""
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,100.0,105.0,99.0,104.0,1000,0
MSFT.US,D,20230101,000000,200.0,210.0,198.0,205.0,2000,0
AAPL.US,D,20230102,000000,104.0,106.0,103.0,105.0,1100,0
MSFT.US,D,20230102,000000,205.0,215.0,203.0,210.0,2100,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        df = load_stooq_daily_prices(
            str(csv_path), 
            as_of_date="2023-01-31",
            ticker="AAPL.US"
        )
        
        # Only AAPL rows
        assert (df["ticker"] == "AAPL.US").all()
        assert len(df) == 2
    
    def test_load_duplicate_last_wins_deterministically(self, tmp_path):
        """Test that duplicate (ticker, date) keeps last row deterministically."""
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,100.0,105.0,99.0,104.0,1000,0
AAPL.US,D,20230101,000000,101.0,106.0,100.0,105.0,1100,0
AAPL.US,D,20230102,000000,105.0,107.0,104.0,106.0,1200,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        df = load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
        
        # Should have only 2 rows (deduped)
        assert len(df) == 2
        
        # The 2023-01-01 row should be the last one (close=105.0, volume=1100)
        row_jan1 = df[df["date"] == pd.to_datetime("2023-01-01")].iloc[0]
        assert row_jan1["close"] == 105.0
        assert row_jan1["volume"] == 1100.0
    
    def test_load_validation_empty_after_cutoff(self, tmp_path):
        """Test that empty result after cutoff raises ValueError."""
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230110,000000,100.0,105.0,99.0,104.0,1000,0
AAPL.US,D,20230115,000000,104.0,106.0,103.0,105.0,1100,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        # All rows are after cutoff
        with pytest.raises(ValueError, match="No data remaining"):
            load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-05")
    
    def test_load_validation_nonfinite_close(self, tmp_path):
        """Test that non-finite OHLC values raise ValueError."""
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,100.0,105.0,99.0,nan,1000,0
AAPL.US,D,20230102,000000,104.0,106.0,103.0,105.0,1100,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        with pytest.raises(ValueError, match="Non-finite"):
            load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
    
    def test_load_validation_bad_ohlc_high_low(self, tmp_path):
        """Test that OHLC sanity violations raise ValueError."""
        # HIGH < CLOSE (violation)
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,100.0,104.0,99.0,110.0,1000,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        with pytest.raises(ValueError, match="OHLC sanity violation"):
            load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
    
    def test_load_validation_bad_ohlc_low_greater(self, tmp_path):
        """Test that LOW > OPEN/CLOSE raises ValueError."""
        # LOW > OPEN (violation)
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,100.0,110.0,105.0,102.0,1000,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        with pytest.raises(ValueError, match="OHLC sanity violation"):
            load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
    
    def test_load_filters_per_d_only(self, tmp_path):
        """Test that only PER=D rows are loaded."""
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,100.0,105.0,99.0,104.0,1000,0
AAPL.US,W,20230101,000000,100.0,110.0,98.0,108.0,5000,0
AAPL.US,D,20230102,000000,104.0,106.0,103.0,105.0,1100,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        df = load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
        
        # Only 2 daily rows
        assert len(df) == 2


class TestResampleToMonthly:
    """Tests for resample_to_monthly function."""
    
    def test_resample_to_monthly_aggregates_correctly(self, tmp_path):
        """Test monthly aggregation produces correct OHLCV values."""
        # Create daily data spanning 2 months
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230102,000000,100.0,105.0,99.0,104.0,1000,0
AAPL.US,D,20230103,000000,104.0,110.0,103.0,108.0,1100,0
AAPL.US,D,20230104,000000,108.0,112.0,97.0,106.0,1200,0
AAPL.US,D,20230201,000000,106.0,115.0,105.0,114.0,2000,0
AAPL.US,D,20230202,000000,114.0,118.0,110.0,112.0,2100,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        daily = load_stooq_daily_prices(str(csv_path), as_of_date="2023-02-28")
        monthly = resample_to_monthly(daily)
        
        # Assert schema
        expected_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        assert list(monthly.columns) == expected_cols
        
        # Should have 2 months
        assert len(monthly) == 2
        
        # Check January (3 rows)
        jan = monthly[monthly["date"] == pd.to_datetime("2023-01-31")].iloc[0]
        assert jan["open"] == 100.0   # first open
        assert jan["high"] == 112.0   # max high
        assert jan["low"] == 97.0     # min low
        assert jan["close"] == 106.0  # last close
        assert jan["volume"] == 3300  # sum: 1000+1100+1200
        
        # Check February (2 rows)
        feb = monthly[monthly["date"] == pd.to_datetime("2023-02-28")].iloc[0]
        assert feb["open"] == 106.0   # first open
        assert feb["high"] == 118.0   # max high
        assert feb["low"] == 105.0    # min low
        assert feb["close"] == 112.0  # last close
        assert feb["volume"] == 4100  # sum: 2000+2100
        
        # Assert sorted by (ticker, date)
        assert monthly["date"].is_monotonic_increasing
    
    def test_resample_to_monthly_multiple_tickers(self, tmp_path):
        """Test resampling with multiple tickers."""
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230102,000000,100.0,105.0,99.0,104.0,1000,0
MSFT.US,D,20230102,000000,200.0,210.0,198.0,205.0,2000,0
AAPL.US,D,20230103,000000,104.0,106.0,103.0,105.0,1100,0
MSFT.US,D,20230103,000000,205.0,215.0,203.0,210.0,2100,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        daily = load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
        monthly = resample_to_monthly(daily)
        
        # 2 tickers, 1 month each = 2 rows
        assert len(monthly) == 2
        assert set(monthly["ticker"]) == {"AAPL.US", "MSFT.US"}
    
    def test_resample_raises_on_missing_columns(self):
        """Test that missing columns raise ValueError."""
        # Missing 'high' column
        df = pd.DataFrame({
            "date": [pd.to_datetime("2023-01-01")],
            "ticker": ["AAPL.US"],
            "open": [100.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [1000.0],
        })
        
        with pytest.raises(ValueError, match="Missing required column"):
            resample_to_monthly(df)


class TestDtypeEnforcement:
    """Tests for float dtype enforcement."""
    
    def test_load_forces_float_dtypes_even_if_integers(self, tmp_path):
        """Test that loader returns float dtypes even for integer-looking values."""
        # Create CSV with integer values (no decimals)
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,10,11,9,10,100,0
AAPL.US,D,20230102,000000,11,12,10,11,110,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        df = load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
        
        # All numeric columns must be float dtype
        assert pd.api.types.is_float_dtype(df["open"]), f"open dtype: {df['open'].dtype}"
        assert pd.api.types.is_float_dtype(df["high"]), f"high dtype: {df['high'].dtype}"
        assert pd.api.types.is_float_dtype(df["low"]), f"low dtype: {df['low'].dtype}"
        assert pd.api.types.is_float_dtype(df["close"]), f"close dtype: {df['close'].dtype}"
        assert pd.api.types.is_float_dtype(df["volume"]), f"volume dtype: {df['volume'].dtype}"
    
    def test_resample_forces_float_dtypes(self, tmp_path):
        """Test that resample returns float dtypes for all numeric columns."""
        # Create CSV with integer values
        csv_content = """\
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20230101,000000,10,11,9,10,100,0
AAPL.US,D,20230102,000000,11,12,10,11,110,0
AAPL.US,D,20230103,000000,12,13,11,12,120,0
"""
        csv_path = tmp_path / "test_stooq.csv"
        csv_path.write_text(csv_content)
        
        daily = load_stooq_daily_prices(str(csv_path), as_of_date="2023-01-31")
        monthly = resample_to_monthly(daily)
        
        # All numeric columns must be float dtype
        assert pd.api.types.is_float_dtype(monthly["open"]), f"open dtype: {monthly['open'].dtype}"
        assert pd.api.types.is_float_dtype(monthly["high"]), f"high dtype: {monthly['high'].dtype}"
        assert pd.api.types.is_float_dtype(monthly["low"]), f"low dtype: {monthly['low'].dtype}"
        assert pd.api.types.is_float_dtype(monthly["close"]), f"close dtype: {monthly['close'].dtype}"
        assert pd.api.types.is_float_dtype(monthly["volume"]), f"volume dtype: {monthly['volume'].dtype}"
