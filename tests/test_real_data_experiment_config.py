"""
Tests for src/real_data_experiment_config.py

Covers:
- Validation rejects missing keys
- Validation rejects bad dates or ordering
- Validation requires 10 unique label tickers
- Validation requires label paths for all tickers
- run_id is deterministic
- Smoke test for run_real_data_experiment_from_config
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.real_data_experiment_config import (
    load_real_data_experiment_config,
    validate_real_data_experiment_config,
    run_real_data_experiment_from_config,
    _compute_run_id,
)


def _create_valid_config() -> dict:
    """Create a minimal valid config for testing."""
    return {
        "feature_builder": "real_data_smoke",
        "feature_builder_kwargs": {
            "stooq_csv_path": "/path/to/spy.csv",
            "price_ticker": "SPY.US",
            "companyfacts_json_paths": ["/path/to/cik1.json"],
            "cik_to_sector": {"0000000001": 0},
        },
        "labels": {
            "stooq_csv_by_ticker": {
                f"T{i}.US": f"/path/to/t{i}.csv" for i in range(10)
            },
            "tickers_in_order": [f"T{i}.US" for i in range(10)],
        },
        "dates": {
            "as_of_date": "2023-12-31",
            "train_end": "2023-04-30",
            "val_end": "2023-08-31",
        },
        "training": {
            "seed": 42,
            "rankgauss": True,
            "epochs": 1,
            "batch_size": 32,
        },
        "health": {
            "min_months": 18,
            "missing_threshold": 0.30,
            "ignore_first_n_rows": 12,
        },
    }


class TestValidateRejectsMissingKeys:
    """Test that validate rejects incomplete configs."""
    
    def test_validate_rejects_missing_feature_builder(self):
        """Test missing feature_builder raises ValueError."""
        cfg = _create_valid_config()
        del cfg["feature_builder"]
        
        with pytest.raises(ValueError, match="feature_builder"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_wrong_feature_builder(self):
        """Test wrong feature_builder value raises ValueError."""
        cfg = _create_valid_config()
        cfg["feature_builder"] = "wrong_builder"
        
        with pytest.raises(ValueError, match="real_data_smoke"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_missing_feature_builder_kwargs(self):
        """Test missing feature_builder_kwargs raises ValueError."""
        cfg = _create_valid_config()
        del cfg["feature_builder_kwargs"]
        
        with pytest.raises(ValueError, match="feature_builder_kwargs"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_missing_labels(self):
        """Test missing labels raises ValueError."""
        cfg = _create_valid_config()
        del cfg["labels"]
        
        with pytest.raises(ValueError, match="labels"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_missing_dates(self):
        """Test missing dates raises ValueError."""
        cfg = _create_valid_config()
        del cfg["dates"]
        
        with pytest.raises(ValueError, match="dates"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_missing_training(self):
        """Test missing training raises ValueError."""
        cfg = _create_valid_config()
        del cfg["training"]
        
        with pytest.raises(ValueError, match="training"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_missing_health(self):
        """Test missing health raises ValueError."""
        cfg = _create_valid_config()
        del cfg["health"]
        
        with pytest.raises(ValueError, match="health"):
            validate_real_data_experiment_config(cfg)


class TestValidateDateOrdering:
    """Test that validate rejects bad dates or ordering."""
    
    def test_validate_rejects_train_end_gte_val_end(self):
        """Test train_end >= val_end raises ValueError."""
        cfg = _create_valid_config()
        cfg["dates"]["train_end"] = "2023-08-31"  # Same as val_end
        cfg["dates"]["val_end"] = "2023-08-31"
        
        with pytest.raises(ValueError, match="train_end"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_train_end_after_val_end(self):
        """Test train_end > val_end raises ValueError."""
        cfg = _create_valid_config()
        cfg["dates"]["train_end"] = "2023-09-30"  # After val_end
        cfg["dates"]["val_end"] = "2023-08-31"
        
        with pytest.raises(ValueError, match="train_end"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_val_end_after_as_of(self):
        """Test val_end > as_of_date raises ValueError."""
        cfg = _create_valid_config()
        cfg["dates"]["val_end"] = "2024-01-31"  # After as_of_date
        
        with pytest.raises(ValueError, match="val_end"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_bad_date_format(self):
        """Test invalid date format raises ValueError."""
        cfg = _create_valid_config()
        cfg["dates"]["as_of_date"] = "12-31-2023"  # Wrong format
        
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            validate_real_data_experiment_config(cfg)


class TestValidateLabelTickers:
    """Test that validate requires 10 unique label tickers."""
    
    def test_validate_rejects_wrong_ticker_count(self):
        """Test non-10 ticker count raises ValueError."""
        cfg = _create_valid_config()
        cfg["labels"]["tickers_in_order"] = [f"T{i}.US" for i in range(5)]  # Only 5
        
        with pytest.raises(ValueError, match="exactly 10"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_duplicate_tickers(self):
        """Test duplicate tickers raises ValueError."""
        cfg = _create_valid_config()
        cfg["labels"]["tickers_in_order"] = ["T0.US"] * 10  # All duplicates
        
        with pytest.raises(ValueError, match="unique"):
            validate_real_data_experiment_config(cfg)
    
    def test_validate_rejects_missing_ticker_path(self):
        """Test missing ticker path in stooq_csv_by_ticker raises ValueError."""
        cfg = _create_valid_config()
        del cfg["labels"]["stooq_csv_by_ticker"]["T9.US"]  # Remove one
        
        with pytest.raises(ValueError, match="T9.US"):
            validate_real_data_experiment_config(cfg)


class TestValidateLabelTickersAlias:
    """Test label_tickers_in_order alias handling."""
    
    def test_config_accepts_label_tickers_in_order_alias(self):
        """Test alias label_tickers_in_order is accepted when tickers_in_order is absent."""
        cfg = _create_valid_config()
        # Use alias instead of original
        cfg["labels"]["label_tickers_in_order"] = cfg["labels"]["tickers_in_order"]
        del cfg["labels"]["tickers_in_order"]
        
        # Should not raise
        validated = validate_real_data_experiment_config(cfg)
        assert "labels" in validated
    
    def test_config_rejects_mismatch_between_alias_and_original(self):
        """Test that different values for alias and original raise ValueError."""
        cfg = _create_valid_config()
        # Provide both with different order
        cfg["labels"]["label_tickers_in_order"] = list(reversed(cfg["labels"]["tickers_in_order"]))
        
        with pytest.raises(ValueError, match="must be identical"):
            validate_real_data_experiment_config(cfg)
    
    def test_config_accepts_both_if_identical(self):
        """Test both keys are accepted if they are identical."""
        cfg = _create_valid_config()
        cfg["labels"]["label_tickers_in_order"] = list(cfg["labels"]["tickers_in_order"])
        
        # Should not raise
        validated = validate_real_data_experiment_config(cfg)
        assert "labels" in validated


class TestValidateHealthSectorGateParams:
    """Test health sector gate param validation."""
    
    def test_config_accepts_health_object_with_valid_values(self):
        """Test config with valid sector gate params passes."""
        cfg = _create_valid_config()
        cfg["health"]["min_sector_firms"] = 3
        cfg["health"]["max_low_count_month_ratio"] = 0.5
        
        # Should not raise
        validated = validate_real_data_experiment_config(cfg)
        assert validated["health"]["min_sector_firms"] == 3
        assert validated["health"]["max_low_count_month_ratio"] == 0.5
    
    def test_config_rejects_health_with_unknown_keys(self):
        """Test config with unknown health keys raises ValueError."""
        cfg = _create_valid_config()
        cfg["health"]["foo"] = 1
        
        with pytest.raises(ValueError, match="Unknown key in health"):
            validate_real_data_experiment_config(cfg)
    
    def test_config_rejects_health_min_sector_firms_zero(self):
        """Test config with min_sector_firms=0 raises ValueError."""
        cfg = _create_valid_config()
        cfg["health"]["min_sector_firms"] = 0
        
        with pytest.raises(ValueError, match="min_sector_firms must be int >= 1"):
            validate_real_data_experiment_config(cfg)
    
    def test_config_rejects_health_max_low_count_month_ratio_out_of_range(self):
        """Test config with ratio > 1.0 raises ValueError."""
        cfg = _create_valid_config()
        cfg["health"]["max_low_count_month_ratio"] = 1.5
        
        with pytest.raises(ValueError, match="max_low_count_month_ratio must be in"):
            validate_real_data_experiment_config(cfg)
    
    def test_config_accepts_absent_sector_gate_params(self):
        """Test config without sector gate params passes (backward compat)."""
        cfg = _create_valid_config()
        # Ensure sector gate params are absent
        assert "min_sector_firms" not in cfg["health"]
        assert "max_low_count_month_ratio" not in cfg["health"]
        
        # Should not raise
        validated = validate_real_data_experiment_config(cfg)
        assert "health" in validated


class TestRunIdDeterminism:
    """Test that run_id is deterministic."""
    
    def test_run_id_is_deterministic(self):
        """Test same config produces same run_id."""
        cfg1 = _create_valid_config()
        cfg2 = _create_valid_config()
        
        run_id1 = _compute_run_id(cfg1)
        run_id2 = _compute_run_id(cfg2)
        
        assert run_id1 == run_id2
        assert len(run_id1) == 64  # SHA256 hex length
    
    def test_run_id_changes_with_config(self):
        """Test different config produces different run_id."""
        cfg1 = _create_valid_config()
        cfg2 = _create_valid_config()
        cfg2["training"]["seed"] = 123  # Small change
        
        run_id1 = _compute_run_id(cfg1)
        run_id2 = _compute_run_id(cfg2)
        
        assert run_id1 != run_id2


class TestLoadConfig:
    """Test load_real_data_experiment_config."""
    
    def test_load_valid_json(self, tmp_path):
        """Test loading valid JSON file."""
        cfg = _create_valid_config()
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        
        loaded = load_real_data_experiment_config(str(config_path))
        assert loaded == cfg
    
    def test_load_raises_on_missing_file(self, tmp_path):
        """Test loading missing file raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            load_real_data_experiment_config(str(tmp_path / "nonexistent.json"))
    
    def test_load_raises_on_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises ValueError."""
        config_path = tmp_path / "bad.json"
        config_path.write_text("{invalid json", encoding="utf-8")
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_real_data_experiment_config(str(config_path))


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


class TestRunExperimentFromConfig:
    """Test run_real_data_experiment_from_config."""
    
    def test_run_real_data_experiment_from_config_smoke(self, tmp_path):
        """Smoke test for config-driven experiment runner."""
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
        
        # Create companyfacts
        quarters = [
            ("2020-12-31", "2021-02-15"),
            ("2021-03-31", "2021-05-01"),
            ("2021-06-30", "2021-08-01"),
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
        
        # Create config
        cfg = {
            "feature_builder": "real_data_smoke",
            "feature_builder_kwargs": {
                "stooq_csv_path": feature_stooq_path,
                "price_ticker": "SPY.US",
                "companyfacts_json_paths": [cik_path],
                "cik_to_sector": {"0000000001": 0},
            },
            "labels": {
                "stooq_csv_by_ticker": label_stooq_by_ticker,
                "tickers_in_order": label_tickers,
            },
            "dates": {
                "as_of_date": "2022-12-31",
                "train_end": "2022-04-30",
                "val_end": "2022-08-31",
            },
            "training": {
                "seed": 42,
                "rankgauss": True,
                "epochs": 1,
                "batch_size": 32,
            },
            "health": {
                "min_months": 18,
                "missing_threshold": 1.0,  # Relaxed for synthetic data
                "ignore_first_n_rows": 12,
            },
        }
        
        config_path = tmp_path / "experiment_config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        
        # Run experiment
        output = run_real_data_experiment_from_config(str(config_path))
        
        # Check required keys
        assert "config" in output
        assert "result" in output
        assert "run_id" in output
        
        # Check run_id is 64-char hex
        assert len(output["run_id"]) == 64
        
        # Check metrics are finite
        assert np.isfinite(output["result"]["train_eval"]["metrics"]["mse"])
        assert np.isfinite(output["result"]["train_eval"]["metrics"]["mae"])
        
        # Check determinism: run again
        output2 = run_real_data_experiment_from_config(str(config_path))
        
        # Same run_id
        assert output["run_id"] == output2["run_id"]
        
        # Same predictions
        pd.testing.assert_frame_equal(
            output["result"]["train_eval"]["y_pred_test"],
            output2["result"]["train_eval"]["y_pred_test"],
        )
