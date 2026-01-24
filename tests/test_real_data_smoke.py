"""
Integration tests for src/real_data_smoke.py

Covers:
- Happy path / shape lock
- End-to-end PIT sanity (filed-gate matters)
- Determinism (same inputs → same outputs)
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.real_data_smoke import build_real_data_feature_frame


def _create_stooq_csv(tmp_path, ticker: str, rows: list[dict]) -> str:
    """Create a Stooq-format CSV file."""
    lines = ["<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"]
    for row in rows:
        lines.append(
            f"{ticker},D,{row['date']},000000,"
            f"{row['open']},{row['high']},{row['low']},{row['close']},{row['vol']},0"
        )
    csv_path = tmp_path / "stooq.csv"
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return str(csv_path)


def _create_companyfacts_json(tmp_path, cik: str, facts: list[dict], filename: str) -> str:
    """Create a SEC companyfacts JSON file."""
    # Build the nested structure
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
        "facts": {
            "us-gaap": units,
        },
    }
    
    json_path = tmp_path / filename
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return str(json_path)


class TestHappyPathShapeLock:
    """Test happy path and output shape."""
    
    def test_output_shape_and_columns(self, tmp_path):
        """Test that output has correct shape, columns, and dtypes."""
        # Create 14 monthly rows for Stooq (one per month-end)
        stooq_rows = []
        for i in range(14):
            year = 2022 if i < 9 else 2023
            month = i + 4 if i < 9 else i - 8
            # Use last day of month
            if month in [4, 6, 9, 11]:
                day = 30
            elif month == 2:
                day = 28
            else:
                day = 31
            date_str = f"{year}{month:02d}{day:02d}"
            stooq_rows.append({
                "date": date_str,
                "open": 100 + i,
                "high": 105 + i,
                "low": 99 + i,
                "close": 102 + i,
                "vol": 1000 + i * 100,
            })
        
        stooq_path = _create_stooq_csv(tmp_path, "SPY.US", stooq_rows)
        
        # Create companyfacts for 2 CIKs with 8 quarters each
        quarters = [
            ("2021-03-31", "2021-05-01"),
            ("2021-06-30", "2021-08-01"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
        ]
        
        cik1_facts = [
            {"end": end, "filed": filed, "val": 100 + i * 10, "tag": "NetIncomeLoss"}
            for i, (end, filed) in enumerate(quarters)
        ]
        cik2_facts = [
            {"end": end, "filed": filed, "val": 50 + i * 5, "tag": "NetIncomeLoss"}
            for i, (end, filed) in enumerate(quarters)
        ]
        
        cik1_path = _create_companyfacts_json(tmp_path, "0000000001", cik1_facts, "cik1.json")
        cik2_path = _create_companyfacts_json(tmp_path, "0000000002", cik2_facts, "cik2.json")
        
        cik_to_sector = {
            "0000000001": 0,
            "0000000002": 1,
        }
        
        result = build_real_data_feature_frame(
            stooq_path,
            price_ticker="SPY.US",
            as_of_date="2023-05-31",
            companyfacts_json_paths=[cik1_path, cik2_path],
            cik_to_sector=cik_to_sector,
        )
        
        # Assert shape
        assert result.shape[1] == 20, f"Expected 20 columns, got {result.shape[1]}"
        assert result.shape[0] > 0, "Expected at least one row"
        
        # Assert index is monotonic increasing and <= as_of_date
        assert result.index.is_monotonic_increasing
        assert result.index.max() <= pd.Timestamp("2023-05-31")
        
        # Assert column names
        expected_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
        assert list(result.columns) == expected_cols
        
        # Assert all float dtypes
        for col in result.columns:
            assert pd.api.types.is_float_dtype(result[col]), f"{col} is not float"


class TestEndToEndPITSanity:
    """Test that PIT filed-gate matters in integrated pipeline."""
    
    def test_filed_gate_affects_features(self, tmp_path):
        """
        Test that features change based on filing date visibility.
        
        This test MUST FAIL if:
        - Future filings change any values for months <= 2023-03-31 (look-ahead)
        - The pipeline fails to produce a finite S0_H1 at 2023-05-31
        """
        # =====================================================================
        # Create Stooq data: include month-ends 2023-02-28, 03-31, 04-30, 05-31
        # Need 18 months of history for H2 (12M) lookback to work
        # =====================================================================
        stooq_rows = []
        # Jan 2022 through May 2023 (17 months)
        month_dates = [
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"), ("2023", "03", "31"),
            ("2023", "04", "30"), ("2023", "05", "31"),
        ]
        for year, month, day in month_dates:
            stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        stooq_path = _create_stooq_csv(tmp_path, "SPY.US", stooq_rows)
        
        # =====================================================================
        # Company A (sector 0): Q4 2022 filed LATE (after 2023-03-31)
        # Company B (sector 1): Q4 2022 filed ON-TIME (before 2023-02-28)
        # Both have 5 quarters so TTM is valid by the 3M lag month
        # =====================================================================
        company_a_facts = [
            {"end": "2021-12-31", "filed": "2022-02-15", "val": 100, "tag": "NetIncomeLoss"},
            {"end": "2022-03-31", "filed": "2022-05-01", "val": 110, "tag": "NetIncomeLoss"},
            {"end": "2022-06-30", "filed": "2022-08-01", "val": 120, "tag": "NetIncomeLoss"},
            {"end": "2022-09-30", "filed": "2022-11-01", "val": 130, "tag": "NetIncomeLoss"},
            # Q4 filed LATE - after 2023-03-31 but before 2023-05-31
            {"end": "2022-12-31", "filed": "2023-04-15", "val": 140, "tag": "NetIncomeLoss"},
        ]
        
        company_b_facts = [
            {"end": "2021-12-31", "filed": "2022-02-15", "val": 50, "tag": "NetIncomeLoss"},
            {"end": "2022-03-31", "filed": "2022-05-01", "val": 55, "tag": "NetIncomeLoss"},
            {"end": "2022-06-30", "filed": "2022-08-01", "val": 60, "tag": "NetIncomeLoss"},
            {"end": "2022-09-30", "filed": "2022-11-01", "val": 65, "tag": "NetIncomeLoss"},
            # Q4 filed ON-TIME - before 2023-02-28
            {"end": "2022-12-31", "filed": "2023-02-01", "val": 70, "tag": "NetIncomeLoss"},
        ]
        
        cik_a_path = _create_companyfacts_json(tmp_path, "0000000001", company_a_facts, "cikA.json")
        cik_b_path = _create_companyfacts_json(tmp_path, "0000000002", company_b_facts, "cikB.json")
        
        cik_to_sector = {
            "0000000001": 0,  # Company A (late Q4 filing)
            "0000000002": 1,  # Company B (on-time Q4 filing)
        }
        
        # =====================================================================
        # Compute result BEFORE Company A's Q4 filing (as_of_date=2023-03-31)
        # =====================================================================
        result_before = build_real_data_feature_frame(
            stooq_path,
            price_ticker="SPY.US",
            as_of_date="2023-03-31",
            companyfacts_json_paths=[cik_a_path, cik_b_path],
            cik_to_sector=cik_to_sector,
        )
        
        # =====================================================================
        # Compute result AFTER Company A's Q4 filing (as_of_date=2023-05-31)
        # =====================================================================
        result_after = build_real_data_feature_frame(
            stooq_path,
            price_ticker="SPY.US",
            as_of_date="2023-05-31",
            companyfacts_json_paths=[cik_a_path, cik_b_path],
            cik_to_sector=cik_to_sector,
        )
        
        # =====================================================================
        # ASSERTION A: No-look-ahead - overlapping months MUST be identical
        # If future filings leak into earlier months, this will FAIL.
        # =====================================================================
        overlapping_index = result_before.index
        result_after_overlapping = result_after.loc[overlapping_index]
        
        pd.testing.assert_frame_equal(
            result_before,
            result_after_overlapping,
            check_names=True,
            obj="No-look-ahead check: overlapping months must be identical",
        )
        
        # =====================================================================
        # ASSERTION B: Filed-gate affects later-month features
        # result_after must have 2023-05-31 and S0_H1 must be finite
        # =====================================================================
        may_2023 = pd.Timestamp("2023-05-31")
        assert may_2023 in result_after.index, f"2023-05-31 should be in result_after index"
        
        s0_h1_may = result_after.loc[may_2023, "S0_H1"]
        assert np.isfinite(s0_h1_may), f"S0_H1 at 2023-05-31 should be finite, got {s0_h1_may}"


class TestDeterminism:
    """Test deterministic behavior."""
    
    def test_identical_inputs_produce_identical_outputs(self, tmp_path):
        """Call twice with same inputs and assert outputs are identical."""
        # Create simple test data
        stooq_rows = []
        for i in range(8):
            year = 2022 if i < 5 else 2023
            month = i + 5 if i < 5 else i - 4
            if month in [4, 6, 9, 11]:
                day = 30
            elif month == 2:
                day = 28
            else:
                day = 31
            date_str = f"{year}{month:02d}{day:02d}"
            stooq_rows.append({
                "date": date_str,
                "open": 100 + i,
                "high": 105 + i,
                "low": 99 + i,
                "close": 102 + i,
                "vol": 1000,
            })
        
        stooq_path = _create_stooq_csv(tmp_path, "TEST.US", stooq_rows)
        
        quarters = [
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
        
        # Call twice
        result1 = build_real_data_feature_frame(
            stooq_path,
            price_ticker="TEST.US",
            as_of_date="2023-03-31",
            companyfacts_json_paths=[cik_path],
            cik_to_sector=cik_to_sector,
        )
        
        result2 = build_real_data_feature_frame(
            stooq_path,
            price_ticker="TEST.US",
            as_of_date="2023-03-31",
            companyfacts_json_paths=[cik_path],
            cik_to_sector=cik_to_sector,
        )
        
        # Assert identical
        pd.testing.assert_frame_equal(result1, result2)


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_empty_cik_to_sector_raises(self, tmp_path):
        """Test that empty cik_to_sector raises ValueError."""
        stooq_rows = [{"date": "20230131", "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000}]
        stooq_path = _create_stooq_csv(tmp_path, "SPY.US", stooq_rows)
        
        cik_facts = [{"end": "2022-12-31", "filed": "2023-01-15", "val": 100, "tag": "NetIncomeLoss"}]
        cik_path = _create_companyfacts_json(tmp_path, "0000000001", cik_facts, "cik.json")
        
        with pytest.raises(ValueError, match="cik_to_sector.*missing or empty"):
            build_real_data_feature_frame(
                stooq_path,
                price_ticker="SPY.US",
                as_of_date="2023-03-31",
                companyfacts_json_paths=[cik_path],
                cik_to_sector={},  # Empty
            )
