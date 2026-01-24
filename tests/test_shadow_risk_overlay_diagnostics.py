"""
Tests for shadow_risk_overlay_diagnostics.py

Covers:
- Happy path with exact expected values
- Determinism (byte-identical JSON across runs)
- Fail-safe behavior (missing file, missing columns)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.shadow_risk_overlay_diagnostics import compute_overlay_exposure_diagnostics


def _create_synthetic_overlay_csv(tmp_path: Path, exposures: list[float]) -> Path:
    """Create a synthetic overlay CSV with given exposure values."""
    csv_path = tmp_path / "overlay.csv"
    df = pd.DataFrame({
        "overlay_equity": [100.0 + i * 10 for i in range(len(exposures))],
        "exposure_suggested": exposures,
    })
    df.to_csv(csv_path, index=False)
    return csv_path


class TestHappyPath:
    """Test correct computation of all metrics."""

    def test_exact_expected_values(self, tmp_path):
        """Verify all metrics match expected values for known series."""
        # Series: [1.0, 1.0, 0.5, 0.5, 1.0]
        # n_obs = 5
        # avg_exposure = (1.0 + 1.0 + 0.5 + 0.5 + 1.0) / 5 = 0.8
        # std_exposure (population, ddof=0) = sqrt(mean((x - 0.8)^2))
        #   = sqrt(((0.2)^2 + (0.2)^2 + (-0.3)^2 + (-0.3)^2 + (0.2)^2) / 5)
        #   = sqrt((0.04 + 0.04 + 0.09 + 0.09 + 0.04) / 5)
        #   = sqrt(0.30 / 5) = sqrt(0.06) ≈ 0.2449
        # frac_exposure_lt_1 = 2/5 = 0.4
        # deltas = [0.0, -0.5, 0.0, +0.5] (consecutive differences)
        # n_switches = 2 (where abs(delta) > eps: -0.5 and +0.5)
        # turnover_proxy = 0.0 + 0.5 + 0.0 + 0.5 = 1.0
        # avg_abs_delta = 1.0 / 4 = 0.25

        exposures = [1.0, 1.0, 0.5, 0.5, 1.0]
        csv_path = _create_synthetic_overlay_csv(tmp_path, exposures)
        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path),
        )

        diag = result["diagnostics"]

        assert diag["n_obs"] == 5
        assert abs(diag["avg_exposure"] - 0.8) < 1e-9
        assert abs(diag["std_exposure"] - 0.24494897427831783) < 1e-9  # sqrt(0.06)
        assert abs(diag["frac_exposure_lt_1"] - 0.4) < 1e-9
        assert diag["n_switches"] == 2
        assert abs(diag["turnover_proxy"] - 1.0) < 1e-9
        assert abs(diag["avg_abs_delta_exposure"] - 0.25) < 1e-9

        # Verify JSON file was written
        assert output_path.exists()
        with open(output_path) as f:
            json_data = json.load(f)
        assert json_data["schema_version"] == "9.6.7"
        assert len(result["warnings"]) == 0

    def test_single_observation(self, tmp_path):
        """Single observation should have 0 switches and 0 turnover."""
        exposures = [0.75]
        csv_path = _create_synthetic_overlay_csv(tmp_path, exposures)
        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path),
        )

        diag = result["diagnostics"]
        assert diag["n_obs"] == 1
        assert abs(diag["avg_exposure"] - 0.75) < 1e-9
        assert diag["std_exposure"] == 0.0
        assert diag["frac_exposure_lt_1"] == 1.0
        assert diag["n_switches"] == 0
        assert diag["turnover_proxy"] == 0.0
        assert diag["avg_abs_delta_exposure"] == 0.0

    def test_all_same_exposure(self, tmp_path):
        """All same exposure should have 0 switches."""
        exposures = [1.0, 1.0, 1.0, 1.0]
        csv_path = _create_synthetic_overlay_csv(tmp_path, exposures)
        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path),
        )

        diag = result["diagnostics"]
        assert diag["n_obs"] == 4
        assert diag["n_switches"] == 0
        assert diag["turnover_proxy"] == 0.0
        assert diag["frac_exposure_lt_1"] == 0.0


class TestDeterminism:
    """Test that output is byte-identical across runs."""

    def test_byte_identical_json(self, tmp_path):
        """Running twice produces byte-identical JSON files."""
        exposures = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.5, 1.0]
        csv_path = _create_synthetic_overlay_csv(tmp_path, exposures)

        output_path_1 = tmp_path / "diagnostics_1.json"
        output_path_2 = tmp_path / "diagnostics_2.json"

        compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path_1),
        )

        compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path_2),
        )

        bytes_1 = output_path_1.read_bytes()
        bytes_2 = output_path_2.read_bytes()

        assert bytes_1 == bytes_2, "JSON output is not byte-identical across runs"


class TestFailSafe:
    """Test fail-safe behavior on errors."""

    def test_missing_file(self, tmp_path):
        """Missing CSV file should produce valid JSON with warnings."""
        missing_path = tmp_path / "nonexistent.csv"
        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(missing_path),
            output_json_path=str(output_path),
        )

        # JSON should be created despite missing file
        assert output_path.exists()

        # Diagnostics should have null/0 values
        diag = result["diagnostics"]
        assert diag["n_obs"] == 0
        assert diag["avg_exposure"] is None
        assert diag["std_exposure"] is None

        # Warnings should contain the error
        assert len(result["warnings"]) > 0
        assert any("SR_DIAG_FAIL:file_not_found" in w for w in result["warnings"])

    def test_missing_exposure_column(self, tmp_path):
        """Missing exposure column should produce valid JSON with warnings."""
        csv_path = tmp_path / "overlay.csv"
        df = pd.DataFrame({
            "overlay_equity": [100.0, 110.0, 120.0],
            "wrong_column": [1.0, 0.5, 1.0],
        })
        df.to_csv(csv_path, index=False)

        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path),
        )

        # JSON should be created
        assert output_path.exists()

        # Diagnostics should have null/0 values
        diag = result["diagnostics"]
        assert diag["n_obs"] == 0
        assert diag["avg_exposure"] is None

        # Warnings should contain the error
        assert len(result["warnings"]) > 0
        assert any("SR_DIAG_FAIL:missing_column" in w for w in result["warnings"])

    def test_empty_csv(self, tmp_path):
        """Empty CSV should produce valid JSON with warnings."""
        csv_path = tmp_path / "overlay.csv"
        df = pd.DataFrame(columns=["overlay_equity", "exposure_suggested"])
        df.to_csv(csv_path, index=False)

        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path),
        )

        assert output_path.exists()
        diag = result["diagnostics"]
        assert diag["n_obs"] == 0
        assert any("SR_DIAG_FAIL:empty_csv" in w for w in result["warnings"])

    def test_csv_with_nan_values(self, tmp_path):
        """CSV with NaN values should skip those rows."""
        csv_path = tmp_path / "overlay.csv"
        df = pd.DataFrame({
            "overlay_equity": [100.0, 110.0, 120.0, 130.0, 140.0],
            "exposure_suggested": [1.0, float("nan"), 0.5, float("nan"), 1.0],
        })
        df.to_csv(csv_path, index=False)

        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path),
        )

        diag = result["diagnostics"]
        # Only 3 usable rows: [1.0, 0.5, 1.0]
        assert diag["n_obs"] == 3
        assert abs(diag["avg_exposure"] - (1.0 + 0.5 + 1.0) / 3) < 1e-9
        # Deltas between usable rows: [0.5-1.0, 1.0-0.5] = [-0.5, 0.5]
        assert diag["n_switches"] == 2
        assert abs(diag["turnover_proxy"] - 1.0) < 1e-9


class TestSchemaCompliance:
    """Test that output JSON matches expected schema."""

    def test_schema_structure(self, tmp_path):
        """Verify JSON has all required keys."""
        exposures = [1.0, 0.5, 1.0]
        csv_path = _create_synthetic_overlay_csv(tmp_path, exposures)
        output_path = tmp_path / "diagnostics.json"

        result = compute_overlay_exposure_diagnostics(
            str(csv_path),
            output_json_path=str(output_path),
        )

        # Check top-level keys
        assert "schema_version" in result
        assert "diagnostics" in result
        assert "warnings" in result

        # Check schema version
        assert result["schema_version"] == "9.6.7"

        # Check diagnostics keys
        required_keys = [
            "n_obs",
            "avg_exposure",
            "std_exposure",
            "frac_exposure_lt_1",
            "n_switches",
            "turnover_proxy",
            "avg_abs_delta_exposure",
        ]
        for key in required_keys:
            assert key in result["diagnostics"], f"Missing diagnostics key: {key}"

        # Check warnings is a list
        assert isinstance(result["warnings"], list)
