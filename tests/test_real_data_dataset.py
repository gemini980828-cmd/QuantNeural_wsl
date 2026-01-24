"""
Tests for src/real_data_dataset.py

Covers:
- Label shift correctness (next-month return calculation)
- as_of_date cutoff prevents future label leakage
- X/Y alignment and shapes
- Determinism
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.real_data_dataset import (
    build_monthly_next_returns_from_stooq,
    build_real_data_xy_dataset,
)
from src.real_data_smoke import build_real_data_feature_frame


def _create_stooq_csv(tmp_path, ticker: str, rows: list[dict], filename: str = None) -> str:
    """Create a Stooq-format CSV file."""
    if filename is None:
        filename = f"{ticker.replace('.', '_')}.csv"
    
    lines = ["<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"]
    for row in rows:
        lines.append(
            f"{ticker},D,{row['date']},000000,"
            f"{row['open']},{row['high']},{row['low']},{row['close']},{row['vol']},0"
        )
    csv_path = tmp_path / filename
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return str(csv_path)


def _create_companyfacts_json(tmp_path, cik: str, facts: list[dict], filename: str) -> str:
    """Create a SEC companyfacts JSON file."""
    units = {}
    for fact in facts:
        tag = fact.get("tag", "NetIncomeLoss")
        unit = fact.get("unit", "USD")
        
        if tag not in units:
            units[tag] = {"units": {}}
        if unit not in units[tag]["units"]:
            units[tag]["units"][unit] = []
        
        units[tag]["units"][unit].append({
            "end": fact["end"],
            "filed": fact["filed"],
            "val": fact["val"],
            "fy": fact.get("fy", 2023),
            "fp": fact.get("fp", "Q4"),
        })
    
    data = {
        "cik": cik,
        "entityName": f"Test Company {cik}",
        "facts": {"us-gaap": units},
    }
    
    json_path = tmp_path / filename
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return str(json_path)


class TestLabelShiftCorrectness:
    """Test label shift (next-month return) calculation."""
    
    def test_label_shift_correctness_small_case(self, tmp_path):
        """Test that next-month returns are computed correctly."""
        # Create 10 tickers, each with 3 month-ends
        # closes = [100, 110, 121] -> returns = [0.10, 0.10, NaN]
        tickers = [f"TICK{i}.US" for i in range(10)]
        stooq_csv_by_ticker = {}
        
        # Create CSV for each ticker
        # Month-ends: 2023-01-31, 2023-02-28, 2023-03-31
        for i, ticker in enumerate(tickers):
            rows = [
                {"date": "20230131", "open": 100, "high": 100, "low": 100, "close": 100, "vol": 1000},
                {"date": "20230228", "open": 110, "high": 110, "low": 110, "close": 110, "vol": 1000},
                {"date": "20230331", "open": 121, "high": 121, "low": 121, "close": 121, "vol": 1000},
            ]
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"tick{i}.csv")
            stooq_csv_by_ticker[ticker] = csv_path
        
        result = build_monthly_next_returns_from_stooq(
            stooq_csv_by_ticker,
            tickers_in_order=tickers,
            as_of_date="2023-04-30",
        )
        
        # Check columns
        expected_cols = [f"S{i}_Y" for i in range(10)]
        assert list(result.columns) == expected_cols
        
        # Check shape
        assert result.shape == (3, 10)
        
        # Check first row: return = (110/100) - 1 = 0.10
        for col in expected_cols:
            assert abs(result.iloc[0][col] - 0.10) < 1e-9, f"First row {col} should be 0.10"
        
        # Check second row: return = (121/110) - 1 = 0.10
        for col in expected_cols:
            assert abs(result.iloc[1][col] - 0.10) < 1e-9, f"Second row {col} should be 0.10"
        
        # Check third (last) row: NaN (no t+1 available)
        for col in expected_cols:
            assert pd.isna(result.iloc[2][col]), f"Last row {col} should be NaN"
    
    def test_label_column_order_exact(self, tmp_path):
        """Test that column order is exactly S0_Y through S9_Y."""
        tickers = [f"T{i}.US" for i in range(10)]
        stooq_csv_by_ticker = {}
        
        for i, ticker in enumerate(tickers):
            rows = [
                {"date": "20230131", "open": 100, "high": 100, "low": 100, "close": 100, "vol": 1000},
                {"date": "20230228", "open": 110, "high": 110, "low": 110, "close": 110, "vol": 1000},
            ]
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"t{i}.csv")
            stooq_csv_by_ticker[ticker] = csv_path
        
        result = build_monthly_next_returns_from_stooq(
            stooq_csv_by_ticker,
            tickers_in_order=tickers,
            as_of_date="2023-03-31",
        )
        
        assert list(result.columns) == ["S0_Y", "S1_Y", "S2_Y", "S3_Y", "S4_Y",
                                         "S5_Y", "S6_Y", "S7_Y", "S8_Y", "S9_Y"]


class TestAsOfDateCutoff:
    """Test as_of_date PIT cutoff for labels."""
    
    def test_as_of_date_cutoff_prevents_future_label_leakage(self, tmp_path):
        """Test that as_of_date cutoff excludes future months from labels."""
        tickers = [f"CUT{i}.US" for i in range(10)]
        stooq_csv_by_ticker = {}
        
        # Create CSV with 4 months (Jan-Apr 2023)
        # as_of_date will be 2023-02-28, so only Jan and Feb should be visible
        for i, ticker in enumerate(tickers):
            rows = [
                {"date": "20230131", "open": 100, "high": 100, "low": 100, "close": 100, "vol": 1000},
                {"date": "20230228", "open": 110, "high": 110, "low": 110, "close": 110, "vol": 1000},
                {"date": "20230331", "open": 121, "high": 121, "low": 121, "close": 121, "vol": 1000},
                {"date": "20230430", "open": 133, "high": 133, "low": 133, "close": 133, "vol": 1000},
            ]
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"cut{i}.csv")
            stooq_csv_by_ticker[ticker] = csv_path
        
        # Cutoff at 2023-02-28
        result = build_monthly_next_returns_from_stooq(
            stooq_csv_by_ticker,
            tickers_in_order=tickers,
            as_of_date="2023-02-28",
        )
        
        # Should only have 2 months (Jan 31 and Feb 28)
        assert len(result) == 2
        assert result.index.max() <= pd.Timestamp("2023-02-28")
        
        # First row return should be (110/100)-1 = 0.10
        # Second row return should be NaN (no March data visible)
        for col in result.columns:
            assert abs(result.iloc[0][col] - 0.10) < 1e-9
            assert pd.isna(result.iloc[1][col])


class TestXYAlignment:
    """Test X/Y alignment and shapes."""
    
    def test_xy_alignment_and_shapes(self, tmp_path):
        """Test that X and Y are properly aligned with correct shapes."""
        # Create Stooq data for features (one ticker for price index)
        feature_stooq_rows = []
        month_dates = [
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"), ("2023", "03", "31"),
            ("2023", "04", "30"), ("2023", "05", "31"),
        ]
        for year, month, day in month_dates:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        # Create companyfacts for features
        quarters = [
            ("2021-12-31", "2022-02-15"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
        ]
        
        cik_facts = [
            {"end": end, "filed": filed, "val": 100 + i * 10, "tag": "NetIncomeLoss"}
            for i, (end, filed) in enumerate(quarters)
        ]
        
        cik_path = _create_companyfacts_json(tmp_path, "0000000001", cik_facts, "cik.json")
        cik_to_sector = {"0000000001": 0}
        
        # Create 10 label tickers
        label_tickers = [f"LABEL{i}.US" for i in range(10)]
        label_stooq_by_ticker = {}
        
        for i, ticker in enumerate(label_tickers):
            rows = []
            for year, month, day in month_dates:
                rows.append({
                    "date": f"{year}{month}{day}",
                    "open": 100 + i, "high": 105 + i, "low": 99 + i, "close": 102 + i, "vol": 1000,
                })
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"label{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        # Build X/Y dataset
        X, Y = build_real_data_xy_dataset(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": feature_stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik_path],
                "cik_to_sector": cik_to_sector,
            },
            stooq_csv_by_ticker_for_labels=label_stooq_by_ticker,
            label_tickers_in_order=label_tickers,
            as_of_date="2023-05-31",
        )
        
        # Check shapes
        assert X.shape[1] == 20, f"X should have 20 columns, got {X.shape[1]}"
        assert Y.shape[1] == 10, f"Y should have 10 columns, got {Y.shape[1]}"
        
        # Check alignment
        assert X.index.equals(Y.index), "X and Y indexes should be identical"
        assert len(X) > 0, "Should have at least one aligned row"
        
        # Check X column names
        expected_x_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
        assert list(X.columns) == expected_x_cols
        
        # Check Y column names
        expected_y_cols = [f"S{i}_Y" for i in range(10)]
        assert list(Y.columns) == expected_y_cols


class TestDeterminism:
    """Test deterministic behavior."""
    
    def test_label_determinism(self, tmp_path):
        """Test that build_monthly_next_returns_from_stooq is deterministic."""
        tickers = [f"DET{i}.US" for i in range(10)]
        stooq_csv_by_ticker = {}
        
        for i, ticker in enumerate(tickers):
            base = 100 + i * 10  # Different base per ticker
            rows = [
                {"date": "20230131", "open": base, "high": base + 5, "low": base - 1, "close": base, "vol": 1000},
                {"date": "20230228", "open": base + 10, "high": base + 15, "low": base + 9, "close": base + 10, "vol": 1000},
                {"date": "20230331", "open": base + 21, "high": base + 26, "low": base + 20, "close": base + 21, "vol": 1000},
            ]
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"det{i}.csv")
            stooq_csv_by_ticker[ticker] = csv_path
        
        result1 = build_monthly_next_returns_from_stooq(
            stooq_csv_by_ticker,
            tickers_in_order=tickers,
            as_of_date="2023-04-30",
        )
        
        result2 = build_monthly_next_returns_from_stooq(
            stooq_csv_by_ticker,
            tickers_in_order=tickers,
            as_of_date="2023-04-30",
        )
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_xy_dataset_determinism(self, tmp_path):
        """Test that build_real_data_xy_dataset is deterministic."""
        # Create minimal fixtures
        feature_stooq_rows = []
        for year, month, day in [
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-12-31", "2022-02-15"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
        ]
        
        cik_facts = [
            {"end": end, "filed": filed, "val": 100 + i * 10, "tag": "NetIncomeLoss"}
            for i, (end, filed) in enumerate(quarters)
        ]
        
        cik_path = _create_companyfacts_json(tmp_path, "0000000001", cik_facts, "cik.json")
        cik_to_sector = {"0000000001": 0}
        
        label_tickers = [f"L{i}.US" for i in range(10)]
        label_stooq_by_ticker = {}
        
        for i, ticker in enumerate(label_tickers):
            rows = feature_stooq_rows.copy()
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        common_kwargs = {
            "feature_builder": build_real_data_feature_frame,
            "feature_builder_kwargs": {
                "stooq_csv_path": feature_stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik_path],
                "cik_to_sector": cik_to_sector,
            },
            "stooq_csv_by_ticker_for_labels": label_stooq_by_ticker,
            "label_tickers_in_order": label_tickers,
            "as_of_date": "2023-02-28",
        }
        
        X1, Y1 = build_real_data_xy_dataset(**common_kwargs)
        X2, Y2 = build_real_data_xy_dataset(**common_kwargs)
        
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_frame_equal(Y1, Y2)


class TestValidation:
    """Test input validation and error handling."""
    
    def test_wrong_ticker_count_raises(self, tmp_path):
        """Test that non-10 ticker count raises ValueError."""
        tickers = [f"T{i}.US" for i in range(5)]  # Only 5 tickers
        stooq_csv_by_ticker = {}
        
        for i, ticker in enumerate(tickers):
            rows = [
                {"date": "20230131", "open": 100, "high": 100, "low": 100, "close": 100, "vol": 1000},
                {"date": "20230228", "open": 110, "high": 110, "low": 110, "close": 110, "vol": 1000},
            ]
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"t{i}.csv")
            stooq_csv_by_ticker[ticker] = csv_path
        
        with pytest.raises(ValueError, match="exactly 10 tickers"):
            build_monthly_next_returns_from_stooq(
                stooq_csv_by_ticker,
                tickers_in_order=tickers,
                as_of_date="2023-03-31",
            )
    
    def test_missing_ticker_raises(self, tmp_path):
        """Test that missing ticker in dict raises ValueError."""
        tickers = [f"T{i}.US" for i in range(10)]
        stooq_csv_by_ticker = {}
        
        # Only create 9 CSVs (missing T9.US)
        for i, ticker in enumerate(tickers[:9]):
            rows = [
                {"date": "20230131", "open": 100, "high": 100, "low": 100, "close": 100, "vol": 1000},
                {"date": "20230228", "open": 110, "high": 110, "low": 110, "close": 110, "vol": 1000},
            ]
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"t{i}.csv")
            stooq_csv_by_ticker[ticker] = csv_path
        
        with pytest.raises(ValueError, match="not found"):
            build_monthly_next_returns_from_stooq(
                stooq_csv_by_ticker,
                tickers_in_order=tickers,
                as_of_date="2023-03-31",
            )
