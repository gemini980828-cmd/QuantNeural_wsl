"""
Tests for shadow_risk_overlay_policy_apply.py

Covers:
- Happy path with exact expected values
- Shifted-weight semantics verification
- Determinism (byte-identical outputs)
- Fail-safe behavior (missing files)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.shadow_risk_overlay_policy_apply import apply_recommended_overlay_policy


def _create_test_files(
    tmp_path: Path,
    shadow_data: dict,
    overlay_data: dict,
    recommendation: dict,
) -> tuple[Path, Path, Path]:
    """Create test input files."""
    # Shadow CSV
    shadow_path = tmp_path / "shadow_risk.csv"
    pd.DataFrame(shadow_data).to_csv(shadow_path, index=False)

    # Overlay CSV
    overlay_path = tmp_path / "shadow_risk_overlay.csv"
    pd.DataFrame(overlay_data).to_csv(overlay_path, index=False)

    # Recommendation JSON
    recommendation_path = tmp_path / "overlay_policy_recommendation.json"
    with open(recommendation_path, "w") as f:
        json.dump(recommendation, f)

    return shadow_path, overlay_path, recommendation_path


class TestHappyPath:
    """Test correct application of policy."""

    def test_applies_policy_correctly(self, tmp_path):
        """Verify policy exposure is computed correctly."""
        # 5 days of data
        # p_risk_off: [0.2, 0.6, 0.8, 0.5, 0.3]
        # Policy: beta_cap=0.35, threshold=0.6
        #
        # Expected exposure:
        #   p=0.2 (<= 0.6) -> p_adj=0 -> w_beta=0 -> exp=1.0
        #   p=0.6 (<= 0.6) -> p_adj=0 -> w_beta=0 -> exp=1.0
        #   p=0.8 (> 0.6)  -> p_adj=(0.8-0.6)/(1-0.6)=0.5 -> w_beta=0.175 -> exp=0.825
        #   p=0.5 (<= 0.6) -> p_adj=0 -> w_beta=0 -> exp=1.0
        #   p=0.3 (<= 0.6) -> p_adj=0 -> w_beta=0 -> exp=1.0

        shadow_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "p_risk_off": [0.2, 0.6, 0.8, 0.5, 0.3],
        }
        overlay_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "spy_ret_1d": [0.01, -0.02, 0.015, 0.005, -0.01],
            "exposure_suggested": [0.7, 0.7, 0.7, 0.7, 0.7],  # baseline
        }
        recommendation = {
            "schema_version": "9.6.10",
            "recommended_policy": {
                "policy_id": 7,
                "beta_cap": 0.35,
                "threshold": 0.6,
            },
        }

        shadow_path, overlay_path, rec_path = _create_test_files(
            tmp_path, shadow_data, overlay_data, recommendation
        )

        output_csv = tmp_path / "output.csv"
        output_metrics = tmp_path / "metrics.json"
        output_diag = tmp_path / "diagnostics.json"

        result = apply_recommended_overlay_policy(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            policy_recommendation_json_path=str(rec_path),
            output_overlay_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            output_diagnostics_json_path=str(output_diag),
        )

        # Verify files exist
        assert output_csv.exists()
        assert output_metrics.exists()
        assert output_diag.exists()

        # Verify CSV content
        df = pd.read_csv(output_csv)
        assert len(df) == 5

        # Check column order
        expected_cols = [
            "date", "spy_ret_1d", "p_risk_off", "exposure_baseline",
            "exposure_suggested", "overlay_ret_1d", "overlay_equity",
            "policy_id", "beta_cap", "threshold",
        ]
        assert list(df.columns) == expected_cols

        # Check policy exposure values
        expected_exposure = [1.0, 1.0, 0.825, 1.0, 1.0]
        for i, exp_val in enumerate(expected_exposure):
            assert abs(df.iloc[i]["exposure_suggested"] - exp_val) < 1e-6, \
                f"Row {i}: expected {exp_val}, got {df.iloc[i]['exposure_suggested']}"

        # Check policy_id column
        assert all(df["policy_id"] == 7)

        # Verify schema version in metrics
        assert result["schema_version"] == "9.6.11"

    def test_shifted_weight_semantics(self, tmp_path):
        """Verify overlay_ret[0]=0 and shifted exposure."""
        # 3 days: exposure = [1.0, 0.5, 1.0]
        # spy_ret = [0.01, 0.02, -0.03]
        #
        # overlay_ret[0] = 0.0 (no prior exposure)
        # overlay_ret[1] = exposure[0] * spy_ret[1] = 1.0 * 0.02 = 0.02
        # overlay_ret[2] = exposure[1] * spy_ret[2] = 0.5 * (-0.03) = -0.015

        # To get exposure=[1.0, 0.5, 1.0]:
        # beta_cap=0.5, threshold=0.0
        # p=0.0 -> exp=1.0
        # p=1.0 -> p_adj=1 -> w_beta=0.5 -> exp=0.5
        # p=0.0 -> exp=1.0

        shadow_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "p_risk_off": [0.0, 1.0, 0.0],
        }
        overlay_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "spy_ret_1d": [0.01, 0.02, -0.03],
            "exposure_suggested": [0.8, 0.8, 0.8],
        }
        recommendation = {
            "schema_version": "9.6.10",
            "recommended_policy": {
                "policy_id": 1,
                "beta_cap": 0.5,
                "threshold": 0.0,
            },
        }

        shadow_path, overlay_path, rec_path = _create_test_files(
            tmp_path, shadow_data, overlay_data, recommendation
        )

        output_csv = tmp_path / "output.csv"
        output_metrics = tmp_path / "metrics.json"
        output_diag = tmp_path / "diagnostics.json"

        apply_recommended_overlay_policy(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            policy_recommendation_json_path=str(rec_path),
            output_overlay_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            output_diagnostics_json_path=str(output_diag),
        )

        df = pd.read_csv(output_csv)

        # Check overlay returns
        assert abs(df.iloc[0]["overlay_ret_1d"] - 0.0) < 1e-10  # First is 0
        assert abs(df.iloc[1]["overlay_ret_1d"] - 0.02) < 1e-10  # 1.0 * 0.02
        assert abs(df.iloc[2]["overlay_ret_1d"] - (-0.015)) < 1e-10  # 0.5 * (-0.03)

        # Check equity
        assert abs(df.iloc[0]["overlay_equity"] - 1.0) < 1e-10
        assert abs(df.iloc[1]["overlay_equity"] - 1.02) < 1e-10
        expected_eq2 = 1.02 * (1 + (-0.015))
        assert abs(df.iloc[2]["overlay_equity"] - expected_eq2) < 1e-10


class TestMetrics:
    """Test metrics computation."""

    def test_metrics_schema(self, tmp_path):
        """Verify metrics JSON has correct schema."""
        shadow_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "p_risk_off": [0.5, 0.5, 0.5],
        }
        overlay_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "spy_ret_1d": [0.01, -0.01, 0.005],
            "exposure_suggested": [0.7, 0.7, 0.7],
        }
        recommendation = {
            "schema_version": "9.6.10",
            "recommended_policy": {"policy_id": 1, "beta_cap": 0.2, "threshold": 0.5},
        }

        shadow_path, overlay_path, rec_path = _create_test_files(
            tmp_path, shadow_data, overlay_data, recommendation
        )

        output_csv = tmp_path / "output.csv"
        output_metrics = tmp_path / "metrics.json"
        output_diag = tmp_path / "diagnostics.json"

        result = apply_recommended_overlay_policy(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            policy_recommendation_json_path=str(rec_path),
            output_overlay_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            output_diagnostics_json_path=str(output_diag),
        )

        # Check required keys
        assert result["schema_version"] == "9.6.11"
        assert "recommended_policy" in result
        assert "baseline" in result
        assert "policy" in result
        assert "deltas_vs_baseline" in result
        assert "warnings" in result

        # Check policy metrics keys
        policy_metrics = result["policy"]
        required_keys = [
            "n_obs", "total_return", "cagr", "vol", "cagr_over_vol",
            "max_dd", "avg_exposure", "std_exposure", "turnover_proxy",
            "n_switches", "avg_abs_delta_exposure",
        ]
        for key in required_keys:
            assert key in policy_metrics, f"Missing key: {key}"


class TestDiagnostics:
    """Test diagnostics output."""

    def test_diagnostics_schema(self, tmp_path):
        """Verify diagnostics JSON has schema_version 9.6.7."""
        shadow_data = {
            "date": ["2024-01-01", "2024-01-02"],
            "p_risk_off": [0.5, 0.5],
        }
        overlay_data = {
            "date": ["2024-01-01", "2024-01-02"],
            "spy_ret_1d": [0.01, -0.01],
            "exposure_suggested": [0.7, 0.7],
        }
        recommendation = {
            "schema_version": "9.6.10",
            "recommended_policy": {"policy_id": 1, "beta_cap": 0.2, "threshold": 0.5},
        }

        shadow_path, overlay_path, rec_path = _create_test_files(
            tmp_path, shadow_data, overlay_data, recommendation
        )

        output_csv = tmp_path / "output.csv"
        output_metrics = tmp_path / "metrics.json"
        output_diag = tmp_path / "diagnostics.json"

        apply_recommended_overlay_policy(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            policy_recommendation_json_path=str(rec_path),
            output_overlay_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            output_diagnostics_json_path=str(output_diag),
        )

        with open(output_diag) as f:
            diag = json.load(f)

        assert diag["schema_version"] == "9.6.7"
        assert "diagnostics" in diag


class TestDeterminism:
    """Test byte-identical outputs across runs."""

    def test_byte_identical_metrics(self, tmp_path):
        """Running twice produces byte-identical metrics JSON."""
        shadow_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "p_risk_off": [0.3, 0.7, 0.5],
        }
        overlay_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "spy_ret_1d": [0.01, -0.02, 0.015],
            "exposure_suggested": [0.8, 0.6, 0.7],
        }
        recommendation = {
            "schema_version": "9.6.10",
            "recommended_policy": {"policy_id": 5, "beta_cap": 0.3, "threshold": 0.5},
        }

        shadow_path, overlay_path, rec_path = _create_test_files(
            tmp_path, shadow_data, overlay_data, recommendation
        )

        dir1 = tmp_path / "run1"
        dir1.mkdir()
        dir2 = tmp_path / "run2"
        dir2.mkdir()

        apply_recommended_overlay_policy(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            policy_recommendation_json_path=str(rec_path),
            output_overlay_csv_path=str(dir1 / "overlay.csv"),
            output_metrics_json_path=str(dir1 / "metrics.json"),
            output_diagnostics_json_path=str(dir1 / "diagnostics.json"),
        )

        apply_recommended_overlay_policy(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            policy_recommendation_json_path=str(rec_path),
            output_overlay_csv_path=str(dir2 / "overlay.csv"),
            output_metrics_json_path=str(dir2 / "metrics.json"),
            output_diagnostics_json_path=str(dir2 / "diagnostics.json"),
        )

        assert (dir1 / "metrics.json").read_bytes() == (dir2 / "metrics.json").read_bytes()


class TestFailSafe:
    """Test fail-safe behavior on errors."""

    def test_missing_recommendation_file(self, tmp_path):
        """Missing recommendation still produces valid outputs with warnings."""
        shadow_data = {
            "date": ["2024-01-01"],
            "p_risk_off": [0.5],
        }
        overlay_data = {
            "date": ["2024-01-01"],
            "spy_ret_1d": [0.01],
        }

        shadow_path = tmp_path / "shadow.csv"
        pd.DataFrame(shadow_data).to_csv(shadow_path, index=False)

        overlay_path = tmp_path / "overlay.csv"
        pd.DataFrame(overlay_data).to_csv(overlay_path, index=False)

        output_csv = tmp_path / "output.csv"
        output_metrics = tmp_path / "metrics.json"
        output_diag = tmp_path / "diagnostics.json"

        result = apply_recommended_overlay_policy(
            shadow_csv_path=str(shadow_path),
            overlay_csv_path=str(overlay_path),
            policy_recommendation_json_path=str(tmp_path / "nonexistent.json"),
            output_overlay_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            output_diagnostics_json_path=str(output_diag),
        )

        # Should not raise
        assert output_metrics.exists()
        assert result["schema_version"] == "9.6.11"
        assert len(result["warnings"]) > 0
        assert result["recommended_policy"] is None

    def test_missing_shadow_csv(self, tmp_path):
        """Missing shadow CSV produces valid outputs with warnings."""
        recommendation = {
            "schema_version": "9.6.10",
            "recommended_policy": {"policy_id": 1, "beta_cap": 0.2, "threshold": 0.5},
        }
        rec_path = tmp_path / "recommendation.json"
        with open(rec_path, "w") as f:
            json.dump(recommendation, f)

        output_csv = tmp_path / "output.csv"
        output_metrics = tmp_path / "metrics.json"
        output_diag = tmp_path / "diagnostics.json"

        result = apply_recommended_overlay_policy(
            shadow_csv_path=str(tmp_path / "nonexistent.csv"),
            overlay_csv_path=str(tmp_path / "overlay.csv"),
            policy_recommendation_json_path=str(rec_path),
            output_overlay_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            output_diagnostics_json_path=str(output_diag),
        )

        assert output_metrics.exists()
        assert len(result["warnings"]) > 0
