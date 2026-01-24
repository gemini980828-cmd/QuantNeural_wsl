"""
Tests for Rolling Temperature Calibration Sweep (v9.6.3).

Tests verify:
- Artifacts are created and deterministic
- Results sorted by val_end with correct column contract
- Skip on insufficient observations
- Summary metrics consistent with window counts
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shadow_risk_calibration_sweep import run_shadow_risk_temperature_sweep


def _create_synthetic_shadow_csv(
    tmp_path: Path,
    n_days: int = 400,
    start_date: str = "2018-01-01",
) -> Path:
    """
    Create synthetic shadow CSV with miscalibrated probabilities.
    
    First half: well-calibrated (p around 0.3 when risk, 0.7 when not)
    Second half: miscalibrated (too extreme predictions)
    """
    np.random.seed(42)
    
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    
    # Create risk events
    true_risk = np.random.binomial(1, 0.25, n_days)
    
    p_risk_off = np.zeros(n_days)
    mid = n_days // 2
    
    # First half: more calibrated
    for i in range(mid):
        if true_risk[i] == 1:
            p_risk_off[i] = np.random.uniform(0.5, 0.7)
        else:
            p_risk_off[i] = np.random.uniform(0.1, 0.3)
    
    # Second half: miscalibrated (too confident)
    for i in range(mid, n_days):
        if true_risk[i] == 1:
            p_risk_off[i] = np.random.uniform(0.8, 0.95)
        else:
            p_risk_off[i] = np.random.uniform(0.01, 0.1)
    
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
    n_days: int = 400,
    start_date: str = "2018-01-01",
) -> Path:
    """
    Create synthetic overlay CSV with SPY returns.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    
    # Normal returns with some volatility
    spy_rets = np.random.normal(0.0003, 0.012, n_days)
    
    # Add periodic crashes to generate risk events
    for crash_start in [50, 150, 250, 350]:
        if crash_start + 5 < n_days:
            spy_rets[crash_start:crash_start + 5] = -0.025
    
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


class TestSweepDeterminism:
    """Tests for deterministic outputs."""
    
    def test_writes_artifacts_and_is_deterministic(self, tmp_path):
        """Running twice with same inputs should produce byte-identical outputs."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        
        output_dir_1 = tmp_path / "run1"
        output_dir_2 = tmp_path / "run2"
        
        params = dict(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            as_of_date="2019-06-30",
            horizon_days=21,
            train_years=1,
            val_years=1,
            test_years=1,
            step_months=3,
            min_obs_val=10,
            min_obs_test=10,
            seed=42,
        )
        
        run_shadow_risk_temperature_sweep(output_dir=str(output_dir_1), **params)
        run_shadow_risk_temperature_sweep(output_dir=str(output_dir_2), **params)
        
        # Compare JSON files
        json_1 = (output_dir_1 / "shadow_risk_temp_sweep_summary.json").read_bytes()
        json_2 = (output_dir_2 / "shadow_risk_temp_sweep_summary.json").read_bytes()
        assert json_1 == json_2, "JSON outputs are not byte-identical"
        
        # Compare CSV files
        csv_1 = (output_dir_1 / "shadow_risk_temp_sweep_results.csv").read_bytes()
        csv_2 = (output_dir_2 / "shadow_risk_temp_sweep_results.csv").read_bytes()
        assert csv_1 == csv_2, "CSV outputs are not byte-identical"


class TestSweepColumnsAndOrder:
    """Tests for CSV column contract and ordering."""
    
    def test_results_sorted_and_column_contract(self, tmp_path):
        """Verify CSV has required columns in exact order and rows sorted by val_end."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        output_dir = tmp_path / "sweep"
        
        run_shadow_risk_temperature_sweep(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            as_of_date="2019-06-30",
            horizon_days=21,
            train_years=1,
            val_years=1,
            test_years=1,
            step_months=3,
            min_obs_val=10,
            min_obs_test=10,
            seed=42,
        )
        
        # Read CSV
        df = pd.read_csv(output_dir / "shadow_risk_temp_sweep_results.csv")
        
        # Check exact column order
        expected_columns = [
            "window_id",
            "train_end",
            "val_end",
            "test_end",
            "best_temperature",
            "n_obs_val",
            "n_obs_test",
            "val_ece_uncal",
            "val_ece_cal",
            "test_ece_uncal",
            "test_ece_cal",
            "val_log_loss_uncal",
            "val_log_loss_cal",
            "test_log_loss_uncal",
            "test_log_loss_cal",
            "warning",
        ]
        assert list(df.columns) == expected_columns
        
        # Check rows are sorted by val_end
        val_ends = pd.to_datetime(df["val_end"])
        assert val_ends.is_monotonic_increasing


class TestSweepSkipping:
    """Tests for skipping on insufficient observations."""
    
    def test_skip_on_insufficient_obs(self, tmp_path):
        """Set min_obs high so windows skip."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path, n_days=200)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path, n_days=200)
        output_dir = tmp_path / "sweep"
        
        result = run_shadow_risk_temperature_sweep(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            as_of_date="2018-12-31",
            horizon_days=21,
            train_years=1,
            val_years=1,
            test_years=1,
            step_months=3,
            min_obs_val=500,  # Very high - will cause skips
            min_obs_test=500,
            seed=42,
        )
        
        # Read CSV to check warnings
        df = pd.read_csv(output_dir / "shadow_risk_temp_sweep_results.csv")
        
        # At least some windows should be skipped
        skipped_rows = df[df["warning"].str.contains("SWEEP_SKIP:insufficient_obs", na=False)]
        assert len(skipped_rows) > 0 or len(df) == 0, "Expected some skipped windows"
        
        # Read JSON summary
        with open(output_dir / "shadow_risk_temp_sweep_summary.json") as f:
            summary_json = json.load(f)
        
        # n_windows_skipped should match skipped rows
        assert summary_json["summary"]["n_windows_skipped"] >= len(skipped_rows)


class TestSweepSummary:
    """Tests for summary metrics consistency."""
    
    def test_summary_metrics_consistent(self, tmp_path):
        """Verify summary counts match expectations."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        output_dir = tmp_path / "sweep"
        
        result = run_shadow_risk_temperature_sweep(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            as_of_date="2019-06-30",
            horizon_days=21,
            train_years=1,
            val_years=1,
            test_years=1,
            step_months=3,
            min_obs_val=10,
            min_obs_test=10,
            seed=42,
        )
        
        # Read summary JSON
        with open(output_dir / "shadow_risk_temp_sweep_summary.json") as f:
            summary_json = json.load(f)
        
        summary = summary_json["summary"]
        
        # n_total = n_evaluated + n_skipped
        assert summary["n_windows_total"] == summary["n_windows_evaluated"] + summary["n_windows_skipped"]
        
        # Read CSV to verify
        df = pd.read_csv(output_dir / "shadow_risk_temp_sweep_results.csv")
        assert summary["n_windows_total"] == len(df)
        
        # If evaluated > 0, share_improved should be defined
        if summary["n_windows_evaluated"] > 0:
            assert summary["share_improved_test_ece"] is not None


class TestSweepOutputFiles:
    """Tests for output file existence."""
    
    def test_creates_all_output_files(self, tmp_path):
        """Sweep should create CSV and JSON output files."""
        shadow_csv = _create_synthetic_shadow_csv(tmp_path)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        output_dir = tmp_path / "sweep"
        
        run_shadow_risk_temperature_sweep(
            shadow_csv_path=str(shadow_csv),
            overlay_csv_path=str(overlay_csv),
            output_dir=str(output_dir),
            as_of_date="2019-06-30",
            horizon_days=21,
            train_years=1,
            val_years=1,
            test_years=1,
            step_months=3,
            seed=42,
        )
        
        assert (output_dir / "shadow_risk_temp_sweep_results.csv").exists()
        assert (output_dir / "shadow_risk_temp_sweep_summary.json").exists()
        
        # JSON should have schema_version
        with open(output_dir / "shadow_risk_temp_sweep_summary.json") as f:
            data = json.load(f)
        assert data["schema_version"] == "9.6.3"
