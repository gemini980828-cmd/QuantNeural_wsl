"""
Tests for src/real_data_health_gates.py

Covers:
- Expected column order
- Missing ratio computation
- Health gate failures on bad index/columns
- Missingness threshold with ignore_first_n_rows
- Valid frame passes all gates
- No-lookahead invariance check
- Exception handling in invariance check
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.real_data_health_gates import (
    expected_h1h2_columns,
    compute_missing_ratios,
    run_real_data_health_gates,
    check_no_lookahead_invariance,
    assert_real_data_health_gates,
)
from src.real_data_smoke import build_real_data_feature_frame


def _create_valid_feature_frame(n_rows: int = 20) -> pd.DataFrame:
    """Create a valid feature frame for testing."""
    index = pd.date_range("2022-01-31", periods=n_rows, freq="ME")
    cols = expected_h1h2_columns()
    data = np.random.default_rng(42).random((n_rows, len(cols)))
    return pd.DataFrame(data, index=index, columns=cols)


class TestExpectedColumns:
    """Test expected_h1h2_columns()."""
    
    def test_expected_columns_exact_order(self):
        """Test that expected columns are exactly 20 in correct order."""
        cols = expected_h1h2_columns()
        
        # Exactly 20 columns
        assert len(cols) == 20
        
        # First 10 are H1
        assert cols[:10] == [f"S{i}_H1" for i in range(10)]
        
        # Last 10 are H2
        assert cols[10:] == [f"S{i}_H2" for i in range(10)]
        
        # Full list
        expected = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
        assert cols == expected


class TestMissingRatios:
    """Test compute_missing_ratios()."""
    
    def test_missing_ratios_basic(self):
        """Test missing ratio computation."""
        index = pd.date_range("2022-01-31", periods=10, freq="ME")
        df = pd.DataFrame({
            "A": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 10% missing
            "B": [np.nan] * 5 + [1.0] * 5,  # 50% missing
            "C": [1.0] * 10,  # 0% missing
        }, index=index)
        
        ratios = compute_missing_ratios(df)
        
        assert abs(ratios["A"] - 0.1) < 1e-9
        assert abs(ratios["B"] - 0.5) < 1e-9
        assert abs(ratios["C"] - 0.0) < 1e-9
    
    def test_missing_ratios_preserves_column_order(self):
        """Test that column order is preserved."""
        index = pd.date_range("2022-01-31", periods=5, freq="ME")
        df = pd.DataFrame({
            "Z": [1.0] * 5,
            "A": [1.0] * 5,
            "M": [1.0] * 5,
        }, index=index)
        
        ratios = compute_missing_ratios(df)
        assert list(ratios.index) == ["Z", "A", "M"]
    
    def test_missing_ratios_empty_frame(self):
        """Test empty frame returns empty series."""
        df = pd.DataFrame()
        ratios = compute_missing_ratios(df)
        assert len(ratios) == 0


class TestHealthGates:
    """Test run_real_data_health_gates()."""
    
    def test_health_gates_pass_on_valid_frame(self):
        """Test that valid frame passes all gates."""
        df = _create_valid_feature_frame(n_rows=20)
        
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,
            max_feature_missing_ratio=0.20,
            ignore_first_n_rows_for_missing=12,
        )
        
        assert report["passed"] is True
        assert report["failed_gates"] == []
        assert report["metrics"]["n_rows"] == 20
        assert report["metrics"]["n_cols"] == 20
        assert report["metrics"]["columns_match_expected"] is True
        assert report["metrics"]["all_float_dtypes"] is True
    
    def test_health_gates_fail_on_bad_index_or_columns(self):
        """Test that bad index or columns cause failures."""
        # Bad index type (not DatetimeIndex)
        df_bad_index = pd.DataFrame(
            np.random.default_rng(42).random((20, 20)),
            columns=expected_h1h2_columns(),
        )
        
        report = run_real_data_health_gates(
            df_bad_index,
            as_of_date="2024-01-31",
            min_months=18,
        )
        
        assert "GATE_INDEX_TYPE" in report["failed_gates"]
        assert report["passed"] is False
        
        # Bad columns (wrong order)
        index = pd.date_range("2022-01-31", periods=20, freq="ME")
        df_bad_cols = pd.DataFrame(
            np.random.default_rng(42).random((20, 20)),
            index=index,
            columns=[f"X{i}" for i in range(20)],  # Wrong column names
        )
        
        report = run_real_data_health_gates(
            df_bad_cols,
            as_of_date="2024-01-31",
            min_months=18,
        )
        
        assert "GATE_COLUMNS_EXACT" in report["failed_gates"]
        assert report["passed"] is False
    
    def test_health_gates_fail_on_non_monotonic_index(self):
        """Test that non-monotonic index fails."""
        cols = expected_h1h2_columns()
        # Create non-monotonic dates spanning 2022-2023
        index = pd.DatetimeIndex([
            "2022-01-31", "2022-03-31", "2022-02-28",  # Not monotonic
            "2022-04-30", "2022-05-31", "2022-06-30",
            "2022-07-31", "2022-08-31", "2022-09-30",
            "2022-10-31", "2022-11-30", "2022-12-31",
            "2023-01-31", "2023-02-28", "2023-03-31",
            "2023-04-30", "2023-05-31", "2023-06-30",
        ])
        
        df = pd.DataFrame(
            np.random.default_rng(42).random((len(index), len(cols))),
            index=index,
            columns=cols,
        )
        
        report = run_real_data_health_gates(df, as_of_date="2024-01-31", min_months=18)
        assert "GATE_INDEX_MONOTONIC" in report["failed_gates"]
    
    def test_health_gates_fail_on_as_of_violation(self):
        """Test that max index > as_of_date fails."""
        df = _create_valid_feature_frame(n_rows=20)
        
        # as_of_date is BEFORE the frame's max index
        report = run_real_data_health_gates(
            df,
            as_of_date="2021-01-31",  # Before frame starts
            min_months=18,
        )
        
        assert "GATE_AS_OF_CUTOFF" in report["failed_gates"]
    
    def test_health_gates_fail_on_min_months(self):
        """Test that insufficient rows fail."""
        df = _create_valid_feature_frame(n_rows=10)
        
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,  # Requires 18, only have 10
        )
        
        assert "GATE_MIN_MONTHS" in report["failed_gates"]
    
    def test_health_gates_missingness_threshold_respects_ignore_first_n_rows(self):
        """Test that ignore_first_n_rows is respected for missingness."""
        index = pd.date_range("2022-01-31", periods=20, freq="ME")
        cols = expected_h1h2_columns()
        
        # First 12 rows have 100% NaN, last 8 rows have 0% NaN
        data = np.zeros((20, len(cols)))
        data[:12, :] = np.nan  # First 12 rows all NaN
        
        df = pd.DataFrame(data, index=index, columns=cols)
        
        # With ignore_first_n_rows_for_missing=12, should PASS
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,
            max_feature_missing_ratio=0.20,
            ignore_first_n_rows_for_missing=12,
        )
        
        assert "GATE_MISSINGNESS" not in report["failed_gates"]
        
        # With ignore_first_n_rows_for_missing=0, should FAIL
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,
            max_feature_missing_ratio=0.20,
            ignore_first_n_rows_for_missing=0,
        )
        
        assert "GATE_MISSINGNESS" in report["failed_gates"]


class TestAssertHealthGates:
    """Test assert_real_data_health_gates()."""
    
    def test_assert_passes_on_passed_report(self):
        """Test that passing report does not raise."""
        report = {"passed": True, "failed_gates": []}
        assert_real_data_health_gates(report)  # Should not raise
    
    def test_assert_raises_on_failed_report(self):
        """Test that failed report raises ValueError."""
        report = {"passed": False, "failed_gates": ["GATE_MIN_MONTHS", "GATE_INDEX_TYPE"]}
        
        with pytest.raises(ValueError, match="GATE_MIN_MONTHS"):
            assert_real_data_health_gates(report)


# =============================================================================
# Integration tests using real_data_smoke fixtures
# =============================================================================

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


class TestNoLookaheadInvariance:
    """Test check_no_lookahead_invariance()."""
    
    def test_check_no_lookahead_invariance_passes_on_real_data_smoke_fixture(self, tmp_path):
        """Test that no-lookahead check passes on correctly implemented pipeline."""
        # Create Stooq data
        stooq_rows = []
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
        
        # Create companyfacts
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
        
        # Run no-lookahead check
        result = check_no_lookahead_invariance(
            build_real_data_feature_frame,
            builder_kwargs={
                "stooq_csv_path": stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik_path],
                "cik_to_sector": cik_to_sector,
            },
            cutoff_dates=["2023-03-31", "2023-05-31"],
        )
        
        assert result["passed"] is True
        assert result["checked_pairs"] == [("2023-03-31", "2023-05-31")]
        assert result["failed_pair"] is None
        assert result["error"] is None
    
    def test_check_no_lookahead_invariance_never_crashes(self):
        """Test that exception in builder returns passed=False with error."""
        def bad_builder(**kwargs):
            raise RuntimeError("Intentional test failure")
        
        result = check_no_lookahead_invariance(
            bad_builder,
            builder_kwargs={},
            cutoff_dates=["2023-01-31", "2023-02-28"],
        )
        
        assert result["passed"] is False
        assert result["error"] is not None
        assert "Intentional test failure" in result["error"]
    
    def test_check_no_lookahead_invariance_single_date(self):
        """Test that single date returns passed=True with no checks."""
        def dummy_builder(as_of_date):
            return pd.DataFrame()
        
        result = check_no_lookahead_invariance(
            dummy_builder,
            builder_kwargs={},
            cutoff_dates=["2023-01-31"],
        )
        
        assert result["passed"] is True
        assert result["checked_pairs"] == []


class TestSectorCountsGate:
    """Test sector representativeness health gate."""
    
    def test_health_gate_does_not_fail_when_counts_missing(self):
        """Test that missing attrs does not cause GATE_SECTOR_COUNTS_MIN failure."""
        # Create valid feature frame WITHOUT attrs
        df = _create_valid_feature_frame(n_rows=20)
        assert "sector_counts" not in df.attrs
        
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,
            min_sector_firms=3,
            max_low_count_month_ratio=0.50,
        )
        
        # GATE_SECTOR_COUNTS_MIN should NOT be in failed_gates
        assert "GATE_SECTOR_COUNTS_MIN" not in report["failed_gates"]
        
        # sector_counts_present should be False
        assert report["sector_counts_present"] is False
        
        # sector_low_count_ratio should NOT be present
        assert "sector_low_count_ratio" not in report
    
    def test_health_gate_fails_when_counts_too_low(self):
        """Test that low sector counts trigger GATE_SECTOR_COUNTS_MIN."""
        # Create valid feature frame
        df = _create_valid_feature_frame(n_rows=20)
        
        # Create sector_counts with mostly zeros (low counts)
        counts_cols = [f"S{i}_n_firms" for i in range(10)]
        counts_data = np.zeros((20, 10), dtype=int)  # All zeros
        counts_df = pd.DataFrame(counts_data, index=df.index, columns=counts_cols)
        
        df.attrs["sector_counts"] = counts_df
        
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,
            ignore_first_n_rows_for_missing=12,
            min_sector_firms=3,  # Need >= 3 firms
            max_low_count_month_ratio=0.50,  # Max 50% of months can have <3 firms
        )
        
        # GATE_SECTOR_COUNTS_MIN should be in failed_gates
        assert "GATE_SECTOR_COUNTS_MIN" in report["failed_gates"]
        
        # sector_counts_present should be True
        assert report["sector_counts_present"] is True
        
        # sector_low_count_ratio should be reported
        assert "sector_low_count_ratio" in report
        
        # All ratios should be 1.0 (100% of eval rows have zero firms)
        for col in counts_cols:
            assert report["sector_low_count_ratio"][col] == 1.0
        
        # Thresholds should be reported
        assert report["sector_min_firms_threshold"] == 3
        assert report["sector_max_low_ratio_threshold"] == 0.50
    
    def test_health_gate_passes_when_counts_sufficient(self):
        """Test that sufficient sector counts do not trigger gate failure."""
        df = _create_valid_feature_frame(n_rows=20)
        
        # Create sector_counts with sufficient firms (>= 3 in all sectors)
        counts_cols = [f"S{i}_n_firms" for i in range(10)]
        counts_data = np.full((20, 10), 5, dtype=int)  # 5 firms per sector per month
        counts_df = pd.DataFrame(counts_data, index=df.index, columns=counts_cols)
        
        df.attrs["sector_counts"] = counts_df
        
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,
            ignore_first_n_rows_for_missing=12,
            min_sector_firms=3,
            max_low_count_month_ratio=0.50,
        )
        
        # GATE_SECTOR_COUNTS_MIN should NOT be in failed_gates
        assert "GATE_SECTOR_COUNTS_MIN" not in report["failed_gates"]
        
        # sector_counts_present should be True
        assert report["sector_counts_present"] is True
        
        # All ratios should be 0.0 (no months with low counts)
        for col in counts_cols:
            assert report["sector_low_count_ratio"][col] == 0.0
    
    def test_health_gate_handles_malformed_counts_gracefully(self):
        """Test that malformed attrs does not crash."""
        df = _create_valid_feature_frame(n_rows=20)
        
        # Attach malformed counts (wrong columns)
        df.attrs["sector_counts"] = pd.DataFrame({"wrong": [1, 2, 3]})
        
        # Should not crash
        report = run_real_data_health_gates(
            df,
            as_of_date="2024-01-31",
            min_months=18,
        )
        
        # GATE_SECTOR_COUNTS_MIN should NOT be in failed_gates (fail-safe)
        assert "GATE_SECTOR_COUNTS_MIN" not in report["failed_gates"]
        
        # sector_counts_present should be False (invalid structure)
        assert report["sector_counts_present"] is False
