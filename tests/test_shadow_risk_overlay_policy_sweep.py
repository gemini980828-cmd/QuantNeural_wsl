"""
Tests for shadow_risk_overlay_policy_sweep.py

Covers:
- Happy path with exact expected values
- Determinism (byte-identical outputs)
- Fail-safe behavior (missing files, missing columns)
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.shadow_risk_overlay_policy_sweep import run_overlay_policy_sweep


def _create_synthetic_csvs(
    tmp_path: Path,
    p_values: list[float],
    spy_returns: list[float],
) -> tuple[Path, Path]:
    """Create synthetic shadow and overlay CSVs."""
    n = len(p_values)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    shadow_path = tmp_path / "shadow.csv"
    shadow_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "p_risk_off": p_values,
    })
    shadow_df.to_csv(shadow_path, index=False)

    overlay_path = tmp_path / "overlay.csv"
    overlay_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "spy_ret_1d": spy_returns,
    })
    overlay_df.to_csv(overlay_path, index=False)

    return shadow_path, overlay_path


class TestHappyPath:
    """Test correct computation of metrics."""

    def test_exact_expected_values(self, tmp_path):
        """Verify metrics match expected values for known inputs."""
        # Simple test case with 5 days
        # p_values: [0.3, 0.6, 0.8, 0.5, 0.2]
        # SPY returns: [0.01, -0.02, 0.015, 0.005, -0.01]
        #
        # Using beta_cap=0.35, threshold=0.5:
        # p_adj = 0 if p <= 0.5 else (p - 0.5) / 0.5
        #   p=0.3 -> p_adj=0 -> w_beta=0 -> exposure=1.0
        #   p=0.6 -> p_adj=0.2 -> w_beta=0.07 -> exposure=0.93
        #   p=0.8 -> p_adj=0.6 -> w_beta=0.21 -> exposure=0.79
        #   p=0.5 -> p_adj=0 -> w_beta=0 -> exposure=1.0
        #   p=0.2 -> p_adj=0 -> w_beta=0 -> exposure=1.0
        # exposure = [1.0, 0.93, 0.79, 1.0, 1.0]

        p_values = [0.3, 0.6, 0.8, 0.5, 0.2]
        spy_returns = [0.01, -0.02, 0.015, 0.005, -0.01]

        shadow_path, overlay_path = _create_synthetic_csvs(
            tmp_path, p_values, spy_returns
        )
        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
            beta_caps=[0.35],
            thresholds=[0.5],
        )

        # Verify files exist
        assert results_path.exists()
        assert summary_path.exists()

        # Load results
        results_df = pd.read_csv(results_path)
        assert len(results_df) == 1  # single policy

        row = results_df.iloc[0]
        assert row["policy_id"] == 1
        assert row["beta_cap"] == 0.35
        assert row["threshold"] == 0.5
        assert row["n_obs"] == 5

        # Expected exposure: [1.0, 0.93, 0.79, 1.0, 1.0]
        expected_exposure = [1.0, 0.93, 0.79, 1.0, 1.0]
        expected_avg = sum(expected_exposure) / 5
        assert abs(row["avg_exposure"] - expected_avg) < 1e-6

        # frac_full_exposure: 3/5 = 0.6 (exposure >= 1-eps for p=0.3, 0.5, 0.2)
        assert abs(row["frac_full_exposure"] - 0.6) < 1e-6

        # Verify schema version
        assert result["schema_version"] == "9.6.9"
        assert len(result["warnings"]) == 0

    def test_shifted_weight_semantics(self, tmp_path):
        """Verify overlay_ret[0] = 0 and shifted-weight computation."""
        # With exposure = [1.0, 0.5, 1.0] and spy_ret = [0.01, 0.02, -0.01]
        # overlay_ret[0] = 0.0 (no prior exposure)
        # overlay_ret[1] = exposure[0] * spy_ret[1] = 1.0 * 0.02 = 0.02
        # overlay_ret[2] = exposure[1] * spy_ret[2] = 0.5 * (-0.01) = -0.005

        # To get exposure=[1.0, 0.5, 1.0]:
        # beta_cap=0.5, threshold=0.0
        # p=0.0 -> p_adj=0 -> exp=1.0
        # p=1.0 -> p_adj=1 -> w_beta=0.5 -> exp=0.5
        # p=0.0 -> p_adj=0 -> exp=1.0

        p_values = [0.0, 1.0, 0.0]
        spy_returns = [0.01, 0.02, -0.01]

        shadow_path, overlay_path = _create_synthetic_csvs(
            tmp_path, p_values, spy_returns
        )
        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
            beta_caps=[0.5],
            thresholds=[0.0],
        )

        results_df = pd.read_csv(results_path)
        row = results_df.iloc[0]

        # Total return from equity curve
        # overlay_ret = [0.0, 0.02, -0.005]
        # equity = [1.0, 1.02, 1.02 * 0.995] = [1.0, 1.02, 1.0149]
        expected_total_return = 1.0149 / 1.0 - 1
        assert abs(row["total_return"] - expected_total_return) < 1e-6

    def test_default_grid(self, tmp_path):
        """Verify default grid produces correct number of policies."""
        p_values = [0.5] * 10
        spy_returns = [0.01] * 10

        shadow_path, overlay_path = _create_synthetic_csvs(
            tmp_path, p_values, spy_returns
        )
        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
            # Use defaults: beta_caps=[0.10, 0.20, 0.35], thresholds=[0.0, 0.4, 0.5, 0.6]
        )

        results_df = pd.read_csv(results_path)
        # 3 beta_caps * 4 thresholds = 12 policies
        assert len(results_df) == 12

        # Verify config in result
        assert result["config"]["beta_caps"] == [0.10, 0.20, 0.35]
        assert result["config"]["thresholds"] == [0.0, 0.4, 0.5, 0.6]


class TestDeterminism:
    """Test byte-identical outputs across runs."""

    def test_byte_identical_json(self, tmp_path):
        """Running twice produces byte-identical summary JSON."""
        p_values = [0.1, 0.5, 0.9, 0.3, 0.7]
        spy_returns = [0.01, -0.02, 0.005, 0.015, -0.01]

        shadow_path, overlay_path = _create_synthetic_csvs(
            tmp_path, p_values, spy_returns
        )

        results_1 = tmp_path / "results_1.csv"
        summary_1 = tmp_path / "summary_1.json"
        results_2 = tmp_path / "results_2.csv"
        summary_2 = tmp_path / "summary_2.json"

        run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_1),
            output_summary_json_path=str(summary_1),
            beta_caps=[0.2, 0.35],
            thresholds=[0.4, 0.5],
        )

        run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_2),
            output_summary_json_path=str(summary_2),
            beta_caps=[0.2, 0.35],
            thresholds=[0.4, 0.5],
        )

        assert summary_1.read_bytes() == summary_2.read_bytes()
        assert results_1.read_bytes() == results_2.read_bytes()


class TestFailSafe:
    """Test fail-safe behavior on errors."""

    def test_missing_shadow_file(self, tmp_path):
        """Missing shadow file produces valid outputs with warnings."""
        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(tmp_path / "nonexistent.csv"),
            overlay_csv_path=str(tmp_path / "overlay.csv"),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
        )

        assert results_path.exists()
        assert summary_path.exists()
        assert len(result["warnings"]) > 0
        assert any("shadow_file_not_found" in w for w in result["warnings"])

        # Results CSV should have headers only
        results_df = pd.read_csv(results_path)
        assert len(results_df) == 0

    def test_missing_spy_ret_column(self, tmp_path):
        """Missing spy_ret column produces valid outputs with warnings."""
        # Create shadow CSV
        shadow_path = tmp_path / "shadow.csv"
        pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "p_risk_off": [0.5, 0.6],
        }).to_csv(shadow_path, index=False)

        # Create overlay CSV without spy_ret_1d
        overlay_path = tmp_path / "overlay.csv"
        pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "wrong_col": [0.01, 0.02],
        }).to_csv(overlay_path, index=False)

        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
        )

        assert summary_path.exists()
        assert len(result["warnings"]) > 0
        assert any("missing_column" in w for w in result["warnings"])

    def test_missing_p_column(self, tmp_path):
        """Missing p_risk_off column produces valid outputs with warnings."""
        # Create shadow CSV without p_risk_off
        shadow_path = tmp_path / "shadow.csv"
        pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "wrong_col": [0.5, 0.6],
        }).to_csv(shadow_path, index=False)

        # Create overlay CSV
        overlay_path = tmp_path / "overlay.csv"
        pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "spy_ret_1d": [0.01, 0.02],
        }).to_csv(overlay_path, index=False)

        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
        )

        assert summary_path.exists()
        assert len(result["warnings"]) > 0
        assert any("missing_column" in w for w in result["warnings"])


class TestBestPolicySelection:
    """Test best policy selection logic."""

    def test_best_by_metrics(self, tmp_path):
        """Verify best_by_* fields are correctly populated."""
        p_values = [0.3, 0.6, 0.4, 0.7, 0.5] * 20  # 100 days
        spy_returns = [0.01, -0.005, 0.008, -0.003, 0.012] * 20

        shadow_path, overlay_path = _create_synthetic_csvs(
            tmp_path, p_values, spy_returns
        )
        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
            beta_caps=[0.1, 0.2],
            thresholds=[0.3, 0.5],
        )

        # Should have best_by_* fields
        assert result["best_by_cagr_over_vol"] is not None
        assert result["best_by_max_dd"] is not None
        assert result["best_by_cagr"] is not None

        # Each should have required keys
        for best in [result["best_by_cagr_over_vol"], result["best_by_max_dd"], result["best_by_cagr"]]:
            assert "policy_id" in best
            assert "beta_cap" in best
            assert "threshold" in best
            assert "cagr" in best
            assert "vol" in best
            assert "max_dd" in best


class TestSchemaCompliance:
    """Test output schema compliance."""

    def test_schema_version(self, tmp_path):
        """Verify schema_version is 9.6.9."""
        p_values = [0.5] * 5
        spy_returns = [0.01] * 5

        shadow_path, overlay_path = _create_synthetic_csvs(
            tmp_path, p_values, spy_returns
        )
        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        result = run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
            beta_caps=[0.1],
            thresholds=[0.5],
        )

        with open(summary_path) as f:
            json_data = json.load(f)

        assert json_data["schema_version"] == "9.6.9"
        assert "config" in json_data
        assert "warnings" in json_data

    def test_csv_columns(self, tmp_path):
        """Verify results CSV has all required columns."""
        p_values = [0.5] * 5
        spy_returns = [0.01] * 5

        shadow_path, overlay_path = _create_synthetic_csvs(
            tmp_path, p_values, spy_returns
        )
        results_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.json"

        run_overlay_policy_sweep(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            output_results_csv_path=str(results_path),
            output_summary_json_path=str(summary_path),
            beta_caps=[0.1],
            thresholds=[0.5],
        )

        results_df = pd.read_csv(results_path)
        required_columns = [
            "policy_id", "beta_cap", "threshold",
            "n_obs", "avg_exposure", "std_exposure", "frac_full_exposure",
            "n_switches", "turnover_proxy", "avg_abs_delta_exposure",
            "total_return", "cagr", "vol", "cagr_over_vol", "max_dd",
        ]
        for col in required_columns:
            assert col in results_df.columns, f"Missing column: {col}"
