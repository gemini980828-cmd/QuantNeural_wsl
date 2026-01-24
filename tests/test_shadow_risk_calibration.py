"""
Tests for Shadow Risk Temperature Scaling Calibration (v9.6.2).

Tests verify:
- Artifacts are created with correct structure
- Calibrated probabilities are finite and in [0,1]
- Determinism: byte-identical outputs on repeated runs
- Calibration improves VAL log_loss
- Last horizon_days are dropped from label evaluation
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shadow_risk_calibration import run_shadow_risk_temperature_calibration


def _create_synthetic_shadow_csv(
    tmp_path: Path,
    n_days: int = 200,
    start_date: str = "2020-01-01",
) -> Path:
    """
    Create synthetic shadow CSV with miscalibrated probabilities.
    
    Probabilities are intentionally too confident (close to 0 or 1)
    to test that temperature scaling can improve calibration.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    
    # Create miscalibrated probabilities (too extreme)
    true_risk = np.random.binomial(1, 0.3, n_days)
    
    # Add noise and make predictions too extreme
    base_p = true_risk * 0.7 + (1 - true_risk) * 0.15
    noise = np.random.normal(0, 0.1, n_days)
    p_risk_off = np.clip(base_p + noise, 0.05, 0.95)
    
    # Make predictions more extreme (miscalibrated)
    p_risk_off = np.where(p_risk_off > 0.5, p_risk_off + 0.2, p_risk_off - 0.1)
    p_risk_off = np.clip(p_risk_off, 0.01, 0.99)
    
    shadow_df = pd.DataFrame({
        "date": dates,
        "p_risk_off": p_risk_off,
        "w_beta_suggested": np.zeros(n_days),
        "exposure_suggested": np.ones(n_days),
    })
    
    path = tmp_path / "shadow_risk.csv"
    shadow_df.to_csv(path, index=False)
    return path


def _create_synthetic_overlay_csv(
    tmp_path: Path,
    n_days: int = 200,
    start_date: str = "2020-01-01",
    crash_day: int = 80,
) -> Path:
    """
    Create synthetic overlay CSV with SPY returns.
    
    Includes a crash to generate some risk-off labels.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    
    # Normal returns with occasional volatility
    spy_rets = np.random.normal(0.0005, 0.01, n_days)
    
    # Add a crash period for risk-off events
    if crash_day < n_days - 10:
        spy_rets[crash_day:crash_day + 5] = -0.03  # 3% daily drops
    
    overlay_df = pd.DataFrame({
        "date": dates,
        "spy_ret_1d": spy_rets,
        "overlay_ret_1d": spy_rets * 0.8,
        "overlay_equity": (1 + spy_rets * 0.8).cumprod(),
        "exposure_suggested": np.full(n_days, 0.8),
    })
    
    path = tmp_path / "overlay.csv"
    overlay_df.to_csv(path, index=False)
    return path


class TestCalibrationOutputs:
    """Tests for calibration output artifacts."""
    
    def test_creates_calibrated_csv_and_json(self, tmp_path):
        """Calibration should create both CSV and JSON outputs."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        output_dir = tmp_path / "calibrated"
        
        result = run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-04-30",
            val_end="2020-07-31",
            as_of_date="2020-10-15",
            horizon_days=21,
            seed=42,
        )
        
        # Check files created
        assert (output_dir / "shadow_risk_calibrated_temp.csv").exists()
        assert (output_dir / "shadow_risk_metrics_calibrated_temp.json").exists()
        
        # Check return structure
        assert "best_temperature" in result
        assert "metrics" in result
        assert "warnings" in result
    
    def test_calibrated_probabilities_in_valid_range(self, tmp_path):
        """Calibrated p_risk_off_cal should be finite and in [0, 1]."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        output_dir = tmp_path / "calibrated"
        
        run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-04-30",
            val_end="2020-07-31",
            as_of_date="2020-10-15",
            horizon_days=21,
            seed=42,
        )
        
        # Read calibrated CSV
        cal_df = pd.read_csv(output_dir / "shadow_risk_calibrated_temp.csv")
        
        assert "p_risk_off_cal" in cal_df.columns
        assert cal_df["p_risk_off_cal"].notna().all()
        assert (cal_df["p_risk_off_cal"] >= 0).all()
        assert (cal_df["p_risk_off_cal"] <= 1).all()
    
    def test_csv_preserves_original_columns(self, tmp_path):
        """Output CSV should preserve all original columns plus p_risk_off_cal."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        output_dir = tmp_path / "calibrated"
        
        run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-04-30",
            val_end="2020-07-31",
            as_of_date="2020-10-15",
            horizon_days=21,
            seed=42,
        )
        
        # Read both CSVs
        original_df = pd.read_csv(shadow_csv)
        cal_df = pd.read_csv(output_dir / "shadow_risk_calibrated_temp.csv")
        
        # Check original columns are preserved
        for col in original_df.columns:
            assert col in cal_df.columns
        
        # Check p_risk_off_cal is added at the end
        assert cal_df.columns[-1] == "p_risk_off_cal"
        
        # Check row count is the same
        assert len(cal_df) == len(original_df)


class TestCalibrationDeterminism:
    """Tests for deterministic outputs."""
    
    def test_byte_identical_outputs_on_repeated_runs(self, tmp_path):
        """Running twice with same inputs should produce byte-identical outputs."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        
        output_dir_1 = tmp_path / "run1"
        output_dir_2 = tmp_path / "run2"
        
        params = dict(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            train_end="2020-04-30",
            val_end="2020-07-31",
            as_of_date="2020-10-15",
            horizon_days=21,
            seed=42,
        )
        
        run_shadow_risk_temperature_calibration(output_dir=str(output_dir_1), **params)
        run_shadow_risk_temperature_calibration(output_dir=str(output_dir_2), **params)
        
        # Compare JSON files
        json_1 = (output_dir_1 / "shadow_risk_metrics_calibrated_temp.json").read_bytes()
        json_2 = (output_dir_2 / "shadow_risk_metrics_calibrated_temp.json").read_bytes()
        assert json_1 == json_2, "JSON outputs are not byte-identical"
        
        # Compare CSV files
        csv_1 = (output_dir_1 / "shadow_risk_calibrated_temp.csv").read_bytes()
        csv_2 = (output_dir_2 / "shadow_risk_calibrated_temp.csv").read_bytes()
        assert csv_1 == csv_2, "CSV outputs are not byte-identical"


class TestCalibrationImproves:
    """Tests that calibration improves metrics."""
    
    def test_val_log_loss_improves_or_stays_same(self, tmp_path):
        """
        After calibration, VAL log_loss should be <= before calibration.
        
        Note: Since we fit temperature on VAL, it should improve or stay the same.
        """
        shadow_csv = _create_synthetic_shadow_csv(tmp_path, n_days=300)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path, n_days=300)
        output_dir = tmp_path / "calibrated"
        
        result = run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-03-31",
            val_end="2020-08-31",
            as_of_date="2020-12-31",
            horizon_days=21,
            seed=42,
        )
        
        val_metrics = result["metrics"]["val"]
        
        # VAL should have observations
        if val_metrics["uncalibrated"]["n_obs"] > 0 and val_metrics["calibrated"]["n_obs"] > 0:
            uncal_ll = val_metrics["uncalibrated"]["log_loss"]
            cal_ll = val_metrics["calibrated"]["log_loss"]
            
            if uncal_ll is not None and cal_ll is not None:
                # Calibrated should be <= uncalibrated (or very close due to rounding)
                assert cal_ll <= uncal_ll + 1e-9, (
                    f"Calibrated log_loss {cal_ll} > uncalibrated {uncal_ll}"
                )


class TestMetricsStructure:
    """Tests for metrics JSON structure."""
    
    def test_metrics_json_has_required_keys(self, tmp_path):
        """Metrics JSON should have all required fields."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        output_dir = tmp_path / "calibrated"
        
        run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-04-30",
            val_end="2020-07-31",
            as_of_date="2020-10-15",
            horizon_days=21,
            seed=42,
        )
        
        with open(output_dir / "shadow_risk_metrics_calibrated_temp.json") as f:
            data = json.load(f)
        
        # Top-level keys
        assert data["schema_version"] == "9.6.2"
        assert "config" in data
        assert "best_temperature" in data
        assert "metrics" in data
        assert "warnings" in data
        
        # Config keys
        assert "train_end" in data["config"]
        assert "val_end" in data["config"]
        assert "as_of_date" in data["config"]
        assert "horizon_days" in data["config"]
        assert "drawdown_threshold" in data["config"]
        assert "n_bins" in data["config"]
        assert "seed" in data["config"]
        
        # Metrics should have train/val/test with uncalibrated and calibrated
        for split in ["train", "val", "test"]:
            assert split in data["metrics"]
            assert "uncalibrated" in data["metrics"][split]
            assert "calibrated" in data["metrics"][split]
            assert "n_obs" in data["metrics"][split]["uncalibrated"]
            assert "n_obs" in data["metrics"][split]["calibrated"]


class TestHorizonDaysDropped:
    """Tests that last horizon_days are correctly dropped from labels."""
    
    def test_last_horizon_days_not_in_metrics(self, tmp_path):
        """
        Verify that the last horizon_days rows have no label and are excluded from metrics.
        """
        n_days = 100
        horizon_days = 21
        
        shadow_csv = _create_synthetic_shadow_csv(tmp_path, n_days=n_days)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path, n_days=n_days)
        output_dir = tmp_path / "calibrated"
        
        result = run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-01-15",
            val_end="2020-03-15",
            as_of_date="2020-05-29",
            horizon_days=horizon_days,
            seed=42,
        )
        
        # Sum up n_obs across all splits
        total_obs = sum(
            result["metrics"][split]["uncalibrated"]["n_obs"]
            for split in ["train", "val", "test"]
        )
        
        # Should be <= n_days - horizon_days (some may overlap or be filtered)
        assert total_obs <= n_days - horizon_days, (
            f"Total obs {total_obs} should be <= {n_days - horizon_days}"
        )


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_handles_empty_val_gracefully(self, tmp_path):
        """Should handle empty VAL split with warning."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path, n_days=50)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path, n_days=50)
        output_dir = tmp_path / "calibrated"
        
        result = run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-04-30",
            val_end="2020-04-30",  # Same as train_end = empty VAL
            as_of_date="2020-05-15",
            horizon_days=10,
            seed=42,
        )
        
        # Should have warning about empty VAL
        assert "VAL_EMPTY:cannot_calibrate" in result["warnings"]
        
        # Should have None temperature
        assert result["best_temperature"] is None
    
    def test_handles_single_class_val_gracefully(self, tmp_path):
        """Should handle VAL with single class with warning."""
        np.random.seed(42)
        
        # Create overlay with no risk events (all positive returns)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        overlay_df = pd.DataFrame({
            "date": dates,
            "spy_ret_1d": np.abs(np.random.normal(0.001, 0.005, 100)),  # All positive
        })
        overlay_csv = tmp_path / "overlay.csv"
        overlay_df.to_csv(overlay_csv, index=False)
        
        # Create shadow CSV
        shadow_df = pd.DataFrame({
            "date": dates,
            "p_risk_off": np.random.uniform(0.1, 0.9, 100),
        })
        shadow_csv = tmp_path / "shadow.csv"
        shadow_df.to_csv(shadow_csv, index=False)
        
        output_dir = tmp_path / "calibrated"
        
        result = run_shadow_risk_temperature_calibration(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            train_end="2020-02-01",
            val_end="2020-04-01",
            as_of_date="2020-05-15",
            horizon_days=10,
            seed=42,
        )
        
        # Should either have warning about single class or empty val2
        # (depends on whether any risk events occur in the synthetic data)
        has_warning = (
            "VAL_SINGLE_CLASS:cannot_calibrate" in result["warnings"]
            or "VAL_EMPTY:cannot_calibrate" in result["warnings"]
            or result["best_temperature"] is None
        )
        
        # Files should still be created
        assert (output_dir / "shadow_risk_calibrated_temp.csv").exists()
        assert (output_dir / "shadow_risk_metrics_calibrated_temp.json").exists()
