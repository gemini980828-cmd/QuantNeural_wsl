"""
Tests for src/real_data_end_to_end.py

Covers:
- NaN-label row dropping
- End-to-end runner smoke test
- Determinism
- Health gate failure handling
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.real_data_end_to_end import (
    drop_nan_label_rows,
    run_real_data_end_to_end_baseline_mlp,
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


class TestDropNanLabelRows:
    """Test drop_nan_label_rows function."""
    
    def test_drop_nan_label_rows_drops_last_month(self):
        """Test that rows with NaN labels are dropped."""
        index = pd.date_range("2022-01-31", periods=3, freq="ME")
        
        # X: 20 columns, all finite
        X_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
        X = pd.DataFrame(
            np.random.default_rng(42).random((3, 20)),
            index=index,
            columns=X_cols,
        )
        
        # Y: 10 columns, last row has NaN
        Y_cols = [f"S{i}_Y" for i in range(10)]
        Y_data = np.random.default_rng(42).random((3, 10))
        Y_data[2, :] = np.nan  # Last row all NaN
        Y = pd.DataFrame(Y_data, index=index, columns=Y_cols)
        
        X2, Y2 = drop_nan_label_rows(X, Y)
        
        # Last month should be removed
        assert len(X2) == 2
        assert len(Y2) == 2
        
        # Indexes should be identical
        assert X2.index.equals(Y2.index)
        
        # Original first two months retained
        assert X2.index[0] == pd.Timestamp("2022-01-31")
        assert X2.index[1] == pd.Timestamp("2022-02-28")
    
    def test_drop_nan_label_rows_partial_nan(self):
        """Test that row with any NaN in Y is dropped."""
        index = pd.date_range("2022-01-31", periods=3, freq="ME")
        
        X_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
        X = pd.DataFrame(
            np.random.default_rng(42).random((3, 20)),
            index=index,
            columns=X_cols,
        )
        
        Y_cols = [f"S{i}_Y" for i in range(10)]
        Y_data = np.random.default_rng(42).random((3, 10))
        Y_data[1, 5] = np.nan  # Middle row has one NaN
        Y = pd.DataFrame(Y_data, index=index, columns=Y_cols)
        
        X2, Y2 = drop_nan_label_rows(X, Y)
        
        # Middle row dropped
        assert len(X2) == 2
        assert pd.Timestamp("2022-02-28") not in X2.index
    
    def test_drop_nan_label_rows_empty_raises(self):
        """Test that empty result raises ValueError."""
        index = pd.date_range("2022-01-31", periods=2, freq="ME")
        
        X_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
        X = pd.DataFrame(
            np.random.default_rng(42).random((2, 20)),
            index=index,
            columns=X_cols,
        )
        
        Y_cols = [f"S{i}_Y" for i in range(10)]
        Y = pd.DataFrame(
            [[np.nan] * 10, [np.nan] * 10],  # All NaN
            index=index,
            columns=Y_cols,
        )
        
        with pytest.raises(ValueError, match="No valid rows"):
            drop_nan_label_rows(X, Y)


class TestEndToEndRunner:
    """Test run_real_data_end_to_end_baseline_mlp function."""
    
    def test_end_to_end_runner_smoke_with_synthetic_builders(self, tmp_path):
        """Smoke test for complete end-to-end runner."""
        # Create feature Stooq data (20+ month-ends)
        feature_stooq_rows = []
        month_dates = [
            ("2021", "01", "31"), ("2021", "02", "28"), ("2021", "03", "31"),
            ("2021", "04", "30"), ("2021", "05", "31"), ("2021", "06", "30"),
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
        ]
        for year, month, day in month_dates:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        # Create companyfacts with enough quarters for TTM
        quarters = [
            ("2020-12-31", "2021-02-15"),
            ("2021-03-31", "2021-05-01"),
            ("2021-06-30", "2021-08-01"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
        
        # Create 10 label tickers
        label_tickers = [f"LABEL{i}.US" for i in range(10)]
        label_stooq_by_ticker = {}
        
        for i, ticker in enumerate(label_tickers):
            rows = []
            for year, month, day in month_dates:
                rows.append({
                    "date": f"{year}{month}{day}",
                    "open": 100 + i, "high": 105 + i, "low": 99, "close": 102 + i, "vol": 1000,
                })
            csv_path = _create_stooq_csv(tmp_path, ticker, rows, f"label{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        # Run end-to-end
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": feature_stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik1_path, cik2_path],
                "cik_to_sector": cik_to_sector,
            },
            stooq_csv_by_ticker_for_labels=label_stooq_by_ticker,
            label_tickers_in_order=label_tickers,
            as_of_date="2022-12-31",
            train_end="2022-04-30",
            val_end="2022-08-31",
            seed=42,
            rankgauss=True,
            epochs=1,
            batch_size=32,
            health_min_months=18,
            health_missing_threshold=1.0,  # Disable for synthetic data (features highly sparse)
            health_ignore_first_n_rows=12,
        )
        
        # Check required keys
        assert "health_report" in result
        assert "train_eval" in result
        assert "n_rows_xy_before_drop" in result
        assert "n_rows_xy_after_drop" in result
        
        # Health report should indicate passed
        assert result["health_report"]["passed"] is True
        
        # Train eval should have metrics
        assert "metrics" in result["train_eval"]
        assert np.isfinite(result["train_eval"]["metrics"]["mse"])
        assert np.isfinite(result["train_eval"]["metrics"]["mae"])
        
        # Last row should be dropped (NaN labels)
        assert result["n_rows_xy_after_drop"] == result["n_rows_xy_before_drop"] - 1
    
    def test_health_gate_failure_raises(self, tmp_path):
        """Test that health gate failure raises ValueError."""
        # Create minimal data
        feature_stooq_rows = []
        month_dates = [
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"), ("2023", "03", "31"),
            ("2023", "04", "30"), ("2023", "05", "31"), ("2023", "06", "30"),
            ("2023", "07", "31"), ("2023", "08", "31"),
        ]
        for year, month, day in month_dates:
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
            ("2022-12-31", "2023-02-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        # Set impossible health_min_months
        with pytest.raises(ValueError, match="Health gates failed"):
            run_real_data_end_to_end_baseline_mlp(
                feature_builder=build_real_data_feature_frame,
                feature_builder_kwargs={
                    "stooq_csv_path": feature_stooq_path,
                    "price_ticker": "SPY.US",
                    "companyfacts_json_paths": [cik_path],
                    "cik_to_sector": cik_to_sector,
                },
                stooq_csv_by_ticker_for_labels=label_stooq_by_ticker,
                label_tickers_in_order=label_tickers,
                as_of_date="2023-08-31",
                train_end="2023-02-28",
                val_end="2023-05-31",
                seed=42,
                epochs=1,
                health_min_months=9999,  # Impossible requirement
            )


class TestDeterminism:
    """Test deterministic behavior."""
    
    def test_determinism_same_seed_same_outputs(self, tmp_path):
        """Test that same seed produces same outputs."""
        # Create data
        feature_stooq_rows = []
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        for year, month, day in month_dates:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
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
            "train_end": "2022-04-30",
            "val_end": "2022-10-31",
            "seed": 42,
            "epochs": 1,
            "health_min_months": 18,
            "health_missing_threshold": 1.0,  # Disable for synthetic data
        }
        
        result1 = run_real_data_end_to_end_baseline_mlp(**common_kwargs)
        result2 = run_real_data_end_to_end_baseline_mlp(**common_kwargs)
        
        # Metrics should be equal
        assert result1["train_eval"]["metrics"]["mse"] == result2["train_eval"]["metrics"]["mse"]
        assert result1["train_eval"]["metrics"]["mae"] == result2["train_eval"]["metrics"]["mae"]
        
        # Predictions should be identical
        pd.testing.assert_frame_equal(
            result1["train_eval"]["y_pred_test"],
            result2["train_eval"]["y_pred_test"],
        )


class TestMetadataOutput:
    """Test metadata output in results."""
    
    def test_result_contains_metadata_fields(self, tmp_path):
        """Test that result dict contains label_tickers_in_order, label_semantics, feature_columns, y_columns."""
        # Create minimal data
        feature_stooq_rows = []
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        for year, month, day in month_dates:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": feature_stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik_path],
                "cik_to_sector": cik_to_sector,
            },
            stooq_csv_by_ticker_for_labels=label_stooq_by_ticker,
            label_tickers_in_order=label_tickers,
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
        )
        
        # Check metadata keys exist
        assert "label_tickers_in_order" in result
        assert "label_semantics" in result
        assert "feature_columns" in result
        assert "y_columns" in result
        
        # Check label_tickers_in_order equals input
        assert result["label_tickers_in_order"] == label_tickers
        
        # Check label_semantics exact string
        assert result["label_semantics"] == "next_month_returns_of_label_tickers"
        
        # Check feature_columns is list of 20 columns (S0_H1..S9_H1, S0_H2..S9_H2)
        expected_feature_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
        assert result["feature_columns"] == expected_feature_cols
        
        # Check y_columns is list of 10 columns (S0_Y..S9_Y)
        expected_y_cols = [f"S{i}_Y" for i in range(10)]
        assert result["y_columns"] == expected_y_cols


class TestHealthParamsWiring:
    """Test that health sector gate params are wired through correctly."""
    
    def test_result_contains_health_params_used(self, tmp_path):
        """Test that result contains health_params_used metadata."""
        # Create minimal data
        feature_stooq_rows = []
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        for year, month, day in month_dates:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": feature_stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik_path],
                "cik_to_sector": cik_to_sector,
            },
            stooq_csv_by_ticker_for_labels=label_stooq_by_ticker,
            label_tickers_in_order=label_tickers,
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            # Explicitly pass sector gate params (using permissive threshold for wiring test)
            health_min_sector_firms=5,
            health_max_low_count_month_ratio=1.0,  # Disabled to allow test to pass
        )
        
        # Check health_params_used exists with the values we passed
        assert "health_params_used" in result
        assert result["health_params_used"]["min_sector_firms"] == 5
        assert result["health_params_used"]["max_low_count_month_ratio"] == 1.0
    
    def test_default_health_params_do_not_fail_on_low_counts(self, tmp_path):
        """Test that default params (disabled gate) don't fail on low firm counts."""
        feature_stooq_rows = []
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        for year, month, day in month_dates:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        # Use DEFAULTS (gate disabled with max_low_count_month_ratio=1.0)
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": feature_stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik_path],
                "cik_to_sector": cik_to_sector,
            },
            stooq_csv_by_ticker_for_labels=label_stooq_by_ticker,
            label_tickers_in_order=label_tickers,
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            # Using defaults: health_min_sector_firms=3, health_max_low_count_month_ratio=1.0
        )
        
        # Should complete without failing sector counts gate
        assert "GATE_SECTOR_COUNTS_MIN" not in result["health_report"]["failed_gates"]
        assert result["health_report"]["passed"] is True
    
    def test_strict_sector_representativeness_gate_fails_end_to_end(self, tmp_path):
        """
        PROOF TEST: Strict sector representativeness settings MUST fail.
        
        This test uses max_low_count_month_ratio=0.0 (any low-count month fails)
        on synthetic data with only 1 firm in sector 0 and 0 in other sectors.
        This proves the gate is actually wired and enforced.
        """
        feature_stooq_rows = []
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        for year, month, day in month_dates:
            feature_stooq_rows.append({
                "date": f"{year}{month}{day}",
                "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000,
            })
        
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
        ]
        
        # Only 1 CIK in sector 0 -> low firm counts in all sectors
        cik_facts = [
            {"end": end, "filed": filed, "val": 100 + i * 10, "tag": "NetIncomeLoss"}
            for i, (end, filed) in enumerate(quarters)
        ]
        
        cik_path = _create_companyfacts_json(tmp_path, "0000000001", cik_facts, "cik.json")
        cik_to_sector = {"0000000001": 0}
        
        label_tickers = [f"L{i}.US" for i in range(10)]
        label_stooq_by_ticker = {}
        
        for i, ticker in enumerate(label_tickers):
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        # STRICT settings: any month with < 3 firms should fail
        with pytest.raises(ValueError, match="Health gates failed"):
            run_real_data_end_to_end_baseline_mlp(
                feature_builder=build_real_data_feature_frame,
                feature_builder_kwargs={
                    "stooq_csv_path": feature_stooq_path,
                    "price_ticker": "SPY.US",
                    "companyfacts_json_paths": [cik_path],
                    "cik_to_sector": cik_to_sector,
                },
                stooq_csv_by_ticker_for_labels=label_stooq_by_ticker,
                label_tickers_in_order=label_tickers,
                as_of_date="2023-02-28",
                train_end="2022-04-30",
                val_end="2022-10-31",
                seed=42,
                epochs=1,
                health_min_months=18,
                health_missing_threshold=1.0,
                # STRICT: require >= 3 firms, allow 0% of months with low counts
                health_min_sector_firms=3,
                health_max_low_count_month_ratio=0.0,
            )


class TestShadowMLPExportWiring:
    """Test optional shadow MLP export wiring in E2E runner (Task 9.3.1)."""
    
    def _create_minimal_e2e_fixtures(self, tmp_path):
        """Create minimal data fixtures for E2E tests."""
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        
        feature_stooq_rows = [
            {"date": f"{y}{m}{d}", "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000}
            for y, m, d in month_dates
        ]
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        return {
            "feature_stooq_path": feature_stooq_path,
            "cik_path": cik_path,
            "cik_to_sector": cik_to_sector,
            "label_tickers": label_tickers,
            "label_stooq_by_ticker": label_stooq_by_ticker,
        }
    
    def test_default_does_not_create_shadow_csv(self, tmp_path):
        """Test that default behavior (shadow_mlp_output_csv_path=None) does NOT create CSV."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            # NOT providing shadow_mlp_output_csv_path
        )
        
        # shadow_mlp_csv_exported should be None
        assert result["shadow_mlp_csv_exported"] is None
        
        # No scores_mlp.csv should exist
        potential_path = tmp_path / "scores_mlp.csv"
        assert not potential_path.exists(), "Shadow CSV should NOT be created by default"
    
    def test_optional_shadow_export_creates_csv(self, tmp_path):
        """Test that providing shadow_mlp_output_csv_path creates the CSV."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        shadow_csv_path = tmp_path / "shadow" / "scores_mlp.csv"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            # OPTIONAL: Enable shadow export
            shadow_mlp_output_csv_path=str(shadow_csv_path),
            shadow_mlp_epochs=1,  # Minimal for speed
        )
        
        # shadow_mlp_csv_exported should be the path
        assert result["shadow_mlp_csv_exported"] == str(shadow_csv_path)
        
        # CSV should exist
        assert shadow_csv_path.exists(), "Shadow CSV NOT created"
        
        # CSV should be valid
        loaded = pd.read_csv(shadow_csv_path, index_col="date", parse_dates=True)
        assert loaded.index.is_monotonic_increasing
        assert loaded.index.is_unique
        assert np.all(np.isfinite(loaded.values))
    
    def test_shadow_export_determinism_at_e2e_boundary(self, tmp_path):
        """Test that E2E shadow export is deterministic (same seed = identical CSV)."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        common_kwargs = {
            "feature_builder": build_real_data_feature_frame,
            "feature_builder_kwargs": {
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            "stooq_csv_by_ticker_for_labels": fixtures["label_stooq_by_ticker"],
            "label_tickers_in_order": fixtures["label_tickers"],
            "as_of_date": "2023-02-28",
            "train_end": "2022-04-30",
            "val_end": "2022-10-31",
            "seed": 42,
            "epochs": 1,
            "health_min_months": 18,
            "health_missing_threshold": 1.0,
            "shadow_mlp_epochs": 1,
        }
        
        # Run 1
        path1 = tmp_path / "run1" / "scores_mlp.csv"
        run_real_data_end_to_end_baseline_mlp(
            **common_kwargs,
            shadow_mlp_output_csv_path=str(path1),
        )
        
        # Run 2
        path2 = tmp_path / "run2" / "scores_mlp.csv"
        run_real_data_end_to_end_baseline_mlp(
            **common_kwargs,
            shadow_mlp_output_csv_path=str(path2),
        )
        
        # CSVs should be byte-identical
        content1 = path1.read_text()
        content2 = path2.read_text()
        assert content1 == content2, "E2E shadow export is NOT deterministic"


class TestShadowRiskE2E:
    """Tests for shadow risk export at E2E level."""
    
    def _create_minimal_e2e_fixtures(self, tmp_path):
        """Create minimal data fixtures for E2E tests."""
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        
        feature_stooq_rows = [
            {"date": f"{y}{m}{d}", "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000}
            for y, m, d in month_dates
        ]
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        return {
            "feature_stooq_path": feature_stooq_path,
            "cik_path": cik_path,
            "cik_to_sector": cik_to_sector,
            "label_tickers": label_tickers,
            "label_stooq_by_ticker": label_stooq_by_ticker,
        }
    
    def test_default_does_not_create_shadow_risk_csv(self, tmp_path):
        """Default behavior should NOT create shadow risk CSV."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        # Create output dir and a fake path to check for
        risk_path = tmp_path / "shadow_risk.csv"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            # shadow_risk_output_csv_path NOT provided
        )
        
        # Should not create CSV
        assert not risk_path.exists()
        
        # Result should have key but be None
        assert result.get("shadow_risk_csv_exported") is None
    
    def test_optional_shadow_risk_export_creates_csv(self, tmp_path):
        """When shadow_risk_output_csv_path provided, should create CSV."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        risk_path = tmp_path / "risk_output.csv"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            shadow_risk_output_csv_path=str(risk_path),
            shadow_risk_spy_ticker="L0",  # Use a ticker that exists in our test data
        )
        
        # CSV should be created (or fallback if SPY not available)
        assert risk_path.exists()
        
        # Result should have the path
        assert result.get("shadow_risk_csv_exported") == str(risk_path)


class TestShadowRiskMetricsE2E:
    """Tests for shadow risk metrics JSON export at E2E level."""
    
    def _create_minimal_e2e_fixtures(self, tmp_path):
        """Create minimal data fixtures for E2E tests."""
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        
        feature_stooq_rows = [
            {"date": f"{y}{m}{d}", "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000}
            for y, m, d in month_dates
        ]
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        return {
            "feature_stooq_path": feature_stooq_path,
            "cik_path": cik_path,
            "cik_to_sector": cik_to_sector,
            "label_tickers": label_tickers,
            "label_stooq_by_ticker": label_stooq_by_ticker,
        }
    
    def test_default_does_not_create_shadow_risk_metrics_json(self, tmp_path):
        """Default behavior should NOT create shadow risk metrics JSON."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        risk_csv_path = tmp_path / "risk.csv"
        risk_json_path = tmp_path / "risk_metrics.json"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            shadow_risk_output_csv_path=str(risk_csv_path),
            shadow_risk_spy_ticker="L0",
            # shadow_risk_metrics_output_json_path NOT provided
        )
        
        # JSON should NOT be created
        assert not risk_json_path.exists()
        
        # Result should have key but be None
        assert result.get("shadow_risk_metrics_json_exported") is None
    
    def test_optional_shadow_risk_metrics_export_creates_json(self, tmp_path):
        """When shadow_risk_metrics_output_json_path provided, should create JSON."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        risk_csv_path = tmp_path / "risk.csv"
        risk_json_path = tmp_path / "risk_metrics.json"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            shadow_risk_output_csv_path=str(risk_csv_path),
            shadow_risk_spy_ticker="L0",
            shadow_risk_metrics_output_json_path=str(risk_json_path),
        )
        
        # Both files should be created
        assert risk_csv_path.exists()
        assert risk_json_path.exists()
        
        # Result should have the path
        assert result.get("shadow_risk_metrics_json_exported") == str(risk_json_path)


class TestShadowRiskOverlayE2E:
    """Tests for shadow risk overlay export at E2E level."""
    
    def _create_minimal_e2e_fixtures(self, tmp_path):
        """Create minimal data fixtures for E2E tests."""
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        
        feature_stooq_rows = [
            {"date": f"{y}{m}{d}", "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000}
            for y, m, d in month_dates
        ]
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        return {
            "feature_stooq_path": feature_stooq_path,
            "cik_path": cik_path,
            "cik_to_sector": cik_to_sector,
            "label_tickers": label_tickers,
            "label_stooq_by_ticker": label_stooq_by_ticker,
        }
    
    def test_default_does_not_create_shadow_risk_overlay_artifacts(self, tmp_path):
        """Default behavior should NOT create overlay artifacts."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        risk_csv_path = tmp_path / "risk.csv"
        overlay_csv_path = tmp_path / "overlay.csv"
        overlay_json_path = tmp_path / "overlay_metrics.json"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            shadow_risk_output_csv_path=str(risk_csv_path),
            shadow_risk_spy_ticker="L0",
            # overlay paths NOT provided
        )
        
        # Overlay files should NOT be created
        assert not overlay_csv_path.exists()
        assert not overlay_json_path.exists()
        
        # Result should have keys but be None
        assert result.get("shadow_risk_overlay_csv_exported") is None
        assert result.get("shadow_risk_overlay_metrics_json_exported") is None
    
    def test_optional_shadow_risk_overlay_export_creates_files(self, tmp_path):
        """When overlay paths provided, should create both files."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        risk_csv_path = tmp_path / "risk.csv"
        overlay_csv_path = tmp_path / "overlay.csv"
        overlay_json_path = tmp_path / "overlay_metrics.json"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            shadow_risk_output_csv_path=str(risk_csv_path),
            shadow_risk_spy_ticker="L0",
            shadow_risk_overlay_output_csv_path=str(overlay_csv_path),
            shadow_risk_overlay_metrics_output_json_path=str(overlay_json_path),
        )
        
        # All shadow risk files should be created
        assert risk_csv_path.exists()
        assert overlay_csv_path.exists()
        assert overlay_json_path.exists()
        
        # Result should have the paths
        assert result.get("shadow_risk_overlay_csv_exported") == str(overlay_csv_path)
        assert result.get("shadow_risk_overlay_metrics_json_exported") == str(overlay_json_path)


class TestShadowRiskDecisionGateE2E:
    """Tests for shadow risk decision gate export at E2E level."""
    
    def _create_minimal_e2e_fixtures(self, tmp_path):
        """Create minimal data fixtures for E2E tests."""
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        
        feature_stooq_rows = [
            {"date": f"{y}{m}{d}", "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000}
            for y, m, d in month_dates
        ]
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        return {
            "feature_stooq_path": feature_stooq_path,
            "cik_path": cik_path,
            "cik_to_sector": cik_to_sector,
            "label_tickers": label_tickers,
            "label_stooq_by_ticker": label_stooq_by_ticker,
        }
    
    def test_default_does_not_create_shadow_risk_decision_gate_json(self, tmp_path):
        """Default behavior should NOT create decision gate JSON."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        risk_csv_path = tmp_path / "risk.csv"
        decision_path = tmp_path / "decision.json"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            shadow_risk_output_csv_path=str(risk_csv_path),
            shadow_risk_spy_ticker="L0",
            # decision gate path NOT provided
        )
        
        # Decision JSON should NOT be created
        assert not decision_path.exists()
        
        # Result should have key but be None
        assert result.get("shadow_risk_decision_gate_json_exported") is None
    
    def test_optional_shadow_risk_decision_gate_export_creates_json(self, tmp_path):
        """When all 9.5.x artifact paths + decision path provided, should create decision JSON."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        risk_csv_path = tmp_path / "risk.csv"
        risk_metrics_path = tmp_path / "risk_metrics.json"
        overlay_csv_path = tmp_path / "overlay.csv"
        overlay_json_path = tmp_path / "overlay_metrics.json"
        decision_path = tmp_path / "decision.json"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            shadow_risk_output_csv_path=str(risk_csv_path),
            shadow_risk_spy_ticker="L0",
            shadow_risk_metrics_output_json_path=str(risk_metrics_path),
            shadow_risk_overlay_output_csv_path=str(overlay_csv_path),
            shadow_risk_overlay_metrics_output_json_path=str(overlay_json_path),
            shadow_risk_decision_gate_output_json_path=str(decision_path),
        )
        
        # Decision JSON should be created
        assert decision_path.exists()
        
        # Result should have the path
        assert result.get("shadow_risk_decision_gate_json_exported") == str(decision_path)

    def test_shadow_risk_daily_prices_panel_has_datetime_index(self, tmp_path):
        """Shadow risk daily price panel should be indexed by dates (not RangeIndex)."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)

        risk_csv_path = tmp_path / "risk.csv"

        captured = {}

        def _fake_train_eval(*args, **kwargs):
            return {"ok": True}

        def _capture_shadow_risk(*, prices, **kwargs):
            captured["index"] = prices.index
            captured["columns"] = list(prices.columns)
            return pd.DataFrame()

        with patch("src.real_data_end_to_end.run_baseline_real_data_mlp_experiment", side_effect=_fake_train_eval):
            with patch("src.real_data_end_to_end.run_shadow_risk_exposure_logit", side_effect=_capture_shadow_risk):
                run_real_data_end_to_end_baseline_mlp(
                    feature_builder=build_real_data_feature_frame,
                    feature_builder_kwargs={
                        "stooq_csv_path": fixtures["feature_stooq_path"],
                        "price_ticker": "SPY.US",
                        "companyfacts_json_paths": [fixtures["cik_path"]],
                        "cik_to_sector": fixtures["cik_to_sector"],
                    },
                    stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
                    label_tickers_in_order=fixtures["label_tickers"],
                    as_of_date="2023-02-28",
                    train_end="2022-04-30",
                    val_end="2022-10-31",
                    seed=42,
                    epochs=1,
                    health_min_months=18,
                    health_missing_threshold=1.0,
                    shadow_risk_output_csv_path=str(risk_csv_path),
                    shadow_risk_spy_ticker="L0",
                )

        assert "index" in captured, "shadow risk function was not called"
        assert isinstance(captured["index"], pd.DatetimeIndex)
        assert captured["index"].is_monotonic_increasing
        assert "L0" in captured["columns"]


class TestShadowRiskMlpOptionalExport:
    """Tests for optional Shadow Risk MLP export."""
    
    def _create_minimal_e2e_fixtures(self, tmp_path):
        """Create minimal data fixtures for E2E tests."""
        month_dates = [
            ("2021", "07", "31"), ("2021", "08", "31"), ("2021", "09", "30"),
            ("2021", "10", "31"), ("2021", "11", "30"), ("2021", "12", "31"),
            ("2022", "01", "31"), ("2022", "02", "28"), ("2022", "03", "31"),
            ("2022", "04", "30"), ("2022", "05", "31"), ("2022", "06", "30"),
            ("2022", "07", "31"), ("2022", "08", "31"), ("2022", "09", "30"),
            ("2022", "10", "31"), ("2022", "11", "30"), ("2022", "12", "31"),
            ("2023", "01", "31"), ("2023", "02", "28"),
        ]
        
        feature_stooq_rows = [
            {"date": f"{y}{m}{d}", "open": 100, "high": 105, "low": 99, "close": 102, "vol": 1000}
            for y, m, d in month_dates
        ]
        feature_stooq_path = _create_stooq_csv(tmp_path, "SPY.US", feature_stooq_rows, "spy.csv")
        
        quarters = [
            ("2021-06-30", "2021-08-15"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
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
            csv_path = _create_stooq_csv(tmp_path, ticker, feature_stooq_rows, f"l{i}.csv")
            label_stooq_by_ticker[ticker] = csv_path
        
        return {
            "feature_stooq_path": feature_stooq_path,
            "cik_path": cik_path,
            "cik_to_sector": cik_to_sector,
            "label_tickers": label_tickers,
            "label_stooq_by_ticker": label_stooq_by_ticker,
        }
    
    def test_default_does_not_create_shadow_risk_mlp_artifacts(self, tmp_path):
        """Default behavior should not create MLP artifacts."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        mlp_csv_path = tmp_path / "shadow_risk_mlp.csv"
        mlp_json_path = tmp_path / "shadow_risk_mlp.json"
        
        # Run without MLP paths
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            # NOT providing MLP paths
        )
        
        # MLP files should not exist
        assert not mlp_csv_path.exists()
        assert not mlp_json_path.exists()
        
        # Return should have None for MLP exports
        assert result.get("shadow_risk_mlp_csv_exported") is None
        assert result.get("shadow_risk_mlp_metrics_json_exported") is None
    
    def test_optional_shadow_risk_mlp_export_creates_files(self, tmp_path):
        """When MLP paths provided (and shadow_risk CSV exists), MLP artifacts created."""
        fixtures = self._create_minimal_e2e_fixtures(tmp_path)
        
        # These are required for MLP export
        risk_csv_path = tmp_path / "shadow_risk.csv"
        risk_json_path = tmp_path / "shadow_risk.json"
        
        mlp_csv_path = tmp_path / "shadow_risk_mlp.csv"
        mlp_json_path = tmp_path / "shadow_risk_mlp.json"
        
        result = run_real_data_end_to_end_baseline_mlp(
            feature_builder=build_real_data_feature_frame,
            feature_builder_kwargs={
                "stooq_csv_path": fixtures["feature_stooq_path"],
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [fixtures["cik_path"]],
                "cik_to_sector": fixtures["cik_to_sector"],
            },
            stooq_csv_by_ticker_for_labels=fixtures["label_stooq_by_ticker"],
            label_tickers_in_order=fixtures["label_tickers"],
            as_of_date="2023-02-28",
            train_end="2022-04-30",
            val_end="2022-10-31",
            seed=42,
            epochs=1,
            health_min_months=18,
            health_missing_threshold=1.0,
            # Shadow Risk base paths (required for MLP to access prices_panel)
            shadow_risk_output_csv_path=str(risk_csv_path),
            shadow_risk_metrics_output_json_path=str(risk_json_path),
            shadow_risk_spy_ticker="L0",
            # MLP paths
            shadow_risk_mlp_output_csv_path=str(mlp_csv_path),
            shadow_risk_mlp_metrics_output_json_path=str(mlp_json_path),
        )
        
        # MLP files should exist
        assert mlp_csv_path.exists()
        assert mlp_json_path.exists()
        
        # Return should have MLP export paths
        assert result.get("shadow_risk_mlp_csv_exported") == str(mlp_csv_path)
        assert result.get("shadow_risk_mlp_metrics_json_exported") == str(mlp_json_path)

