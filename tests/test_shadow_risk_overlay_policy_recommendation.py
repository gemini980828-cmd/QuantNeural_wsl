"""
Tests for shadow_risk_overlay_policy_recommendation.py

Covers:
- Happy path with guardrails
- Tie-breaking logic
- Determinism
- Fail-safe behavior
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.shadow_risk_overlay_policy_recommendation import build_overlay_policy_recommendation


def _create_baseline_files(tmp_path: Path, metrics: dict, diagnostics: dict) -> tuple[Path, Path]:
    """Create baseline overlay metrics and diagnostics files."""
    metrics_path = tmp_path / "overlay_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    diagnostics_path = tmp_path / "overlay_diagnostics.json"
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f)

    return metrics_path, diagnostics_path


def _create_sweep_files(tmp_path: Path, policies: list[dict], summary: dict) -> tuple[Path, Path]:
    """Create policy sweep results CSV and summary JSON."""
    results_path = tmp_path / "sweep_results.csv"
    df = pd.DataFrame(policies)
    df.to_csv(results_path, index=False)

    summary_path = tmp_path / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f)

    return results_path, summary_path


class TestHappyPath:
    """Test correct recommendation selection."""

    def test_selects_best_passing_guardrails(self, tmp_path):
        """Selects policy that passes all guardrails with best cagr_over_vol."""
        # Baseline: avg_exposure=0.6, turnover=2.0, max_dd=-0.05
        metrics_path, diagnostics_path = _create_baseline_files(
            tmp_path,
            metrics={"test": {"cagr_over_vol": 2.0, "cagr": 0.08, "ann_vol": 0.04, "max_drawdown": -0.05}},
            diagnostics={"diagnostics": {"avg_exposure": 0.6, "turnover_proxy": 2.0}},
        )

        # Policies:
        # P1: avg_exp=0.7 (>baseline), turnover=2.5 (<3.0 cap), max_dd=-0.06 (>-0.07 threshold), cagr/vol=2.5
        # P2: avg_exp=0.5 (<baseline FAILS require_higher_avg_exposure)
        # P3: avg_exp=0.8, turnover=10.0 (>3.0 cap FAILS)
        # P4: avg_exp=0.65, turnover=1.5, max_dd=-0.10 (<-0.07 FAILS max_dd)
        policies = [
            {"policy_id": 1, "beta_cap": 0.1, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.7, "std_exposure": 0.1, "turnover_proxy": 2.5,
             "n_switches": 50, "cagr_over_vol": 2.5, "cagr": 0.10, "vol": 0.04, "max_dd": -0.06},
            {"policy_id": 2, "beta_cap": 0.2, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.5, "std_exposure": 0.1, "turnover_proxy": 1.0,
             "n_switches": 20, "cagr_over_vol": 3.0, "cagr": 0.12, "vol": 0.04, "max_dd": -0.03},
            {"policy_id": 3, "beta_cap": 0.1, "threshold": 0.4, "n_obs": 100,
             "avg_exposure": 0.8, "std_exposure": 0.1, "turnover_proxy": 10.0,
             "n_switches": 80, "cagr_over_vol": 2.8, "cagr": 0.11, "vol": 0.04, "max_dd": -0.04},
            {"policy_id": 4, "beta_cap": 0.3, "threshold": 0.6, "n_obs": 100,
             "avg_exposure": 0.65, "std_exposure": 0.1, "turnover_proxy": 1.5,
             "n_switches": 30, "cagr_over_vol": 2.6, "cagr": 0.10, "vol": 0.04, "max_dd": -0.10},
        ]
        results_path, summary_path = _create_sweep_files(
            tmp_path, policies,
            {"schema_version": "9.6.9", "best_by_cagr_over_vol": {"policy_id": 2}},
        )

        output_path = tmp_path / "recommendation.json"
        result = build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_path),
            max_dd_tolerance=0.02,  # threshold = -0.05 - 0.02 = -0.07
            turnover_multiplier_cap=1.5,  # threshold = 2.0 * 1.5 = 3.0
            require_higher_avg_exposure=True,
        )

        # Only P1 passes all guardrails
        assert result["recommended_policy"] is not None
        assert result["recommended_policy"]["policy_id"] == 1

        # Verify schema
        assert result["schema_version"] == "9.6.10"

    def test_tie_breaking(self, tmp_path):
        """Tie-break by max_dd, then turnover, then policy_id."""
        metrics_path, diagnostics_path = _create_baseline_files(
            tmp_path,
            metrics={"test": {"cagr_over_vol": 2.0, "max_drawdown": -0.05}},
            diagnostics={"diagnostics": {"avg_exposure": 0.5, "turnover_proxy": 5.0}},
        )

        # Two policies with same cagr_over_vol
        policies = [
            {"policy_id": 1, "beta_cap": 0.1, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.6, "std_exposure": 0.1, "turnover_proxy": 2.0,
             "n_switches": 50, "cagr_over_vol": 3.0, "cagr": 0.12, "vol": 0.04, "max_dd": -0.04},
            {"policy_id": 2, "beta_cap": 0.2, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.6, "std_exposure": 0.1, "turnover_proxy": 1.5,
             "n_switches": 40, "cagr_over_vol": 3.0, "cagr": 0.12, "vol": 0.04, "max_dd": -0.03},
        ]
        results_path, summary_path = _create_sweep_files(
            tmp_path, policies,
            {"schema_version": "9.6.9", "best_by_cagr_over_vol": {"policy_id": 1}},
        )

        output_path = tmp_path / "recommendation.json"
        result = build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_path),
        )

        # P2 should win: same cagr_over_vol but better max_dd (-0.03 > -0.04)
        assert result["recommended_policy"]["policy_id"] == 2

    def test_deltas_computed_correctly(self, tmp_path):
        """Verify deltas_vs_baseline are computed correctly."""
        metrics_path, diagnostics_path = _create_baseline_files(
            tmp_path,
            metrics={"test": {"cagr_over_vol": 2.0, "max_drawdown": -0.05}},
            diagnostics={"diagnostics": {"avg_exposure": 0.6, "turnover_proxy": 3.0}},
        )

        policies = [
            {"policy_id": 1, "beta_cap": 0.1, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.8, "std_exposure": 0.1, "turnover_proxy": 2.0,
             "n_switches": 50, "cagr_over_vol": 2.5, "cagr": 0.10, "vol": 0.04, "max_dd": -0.03},
        ]
        results_path, summary_path = _create_sweep_files(
            tmp_path, policies,
            {"schema_version": "9.6.9", "best_by_cagr_over_vol": {"policy_id": 1}},
        )

        output_path = tmp_path / "recommendation.json"
        result = build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_path),
        )

        # Check deltas in candidates
        top_cov = result["candidates"]["top_by_cagr_over_vol"][0]
        assert abs(top_cov["deltas_vs_baseline"]["delta_avg_exposure"] - 0.2) < 1e-6
        assert abs(top_cov["deltas_vs_baseline"]["delta_turnover_proxy"] - (-1.0)) < 1e-6
        assert abs(top_cov["deltas_vs_baseline"]["delta_cagr_over_vol"] - 0.5) < 1e-6
        assert abs(top_cov["deltas_vs_baseline"]["delta_max_dd"] - 0.02) < 1e-6


class TestDeterminism:
    """Test byte-identical outputs across runs."""

    def test_byte_identical_json(self, tmp_path):
        """Running twice produces byte-identical output."""
        metrics_path, diagnostics_path = _create_baseline_files(
            tmp_path,
            metrics={"test": {"cagr_over_vol": 2.0}},
            diagnostics={"diagnostics": {"avg_exposure": 0.6}},
        )

        policies = [
            {"policy_id": 1, "beta_cap": 0.1, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.7, "std_exposure": 0.1, "turnover_proxy": 2.0,
             "n_switches": 50, "cagr_over_vol": 2.5, "cagr": 0.10, "vol": 0.04, "max_dd": -0.04},
            {"policy_id": 2, "beta_cap": 0.2, "threshold": 0.4, "n_obs": 100,
             "avg_exposure": 0.8, "std_exposure": 0.15, "turnover_proxy": 1.5,
             "n_switches": 40, "cagr_over_vol": 2.8, "cagr": 0.11, "vol": 0.04, "max_dd": -0.03},
        ]
        results_path, summary_path = _create_sweep_files(
            tmp_path, policies,
            {"schema_version": "9.6.9", "best_by_cagr_over_vol": {"policy_id": 2}},
        )

        output_1 = tmp_path / "recommendation_1.json"
        output_2 = tmp_path / "recommendation_2.json"

        build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_1),
        )

        build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_2),
        )

        assert output_1.read_bytes() == output_2.read_bytes()


class TestFailSafe:
    """Test fail-safe behavior on errors."""

    def test_missing_files(self, tmp_path):
        """Missing input files produce valid output with warnings."""
        output_path = tmp_path / "recommendation.json"

        result = build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(tmp_path / "nonexistent_metrics.json"),
            overlay_diagnostics_json_path=str(tmp_path / "nonexistent_diag.json"),
            policy_sweep_results_csv_path=str(tmp_path / "nonexistent_results.csv"),
            policy_sweep_summary_json_path=str(tmp_path / "nonexistent_summary.json"),
            output_json_path=str(output_path),
        )

        assert output_path.exists()
        assert result["schema_version"] == "9.6.10"
        assert len(result["warnings"]) > 0
        assert result["recommended_policy"] is None

    def test_empty_sweep_results(self, tmp_path):
        """Empty sweep results produce valid output with warnings."""
        metrics_path, diagnostics_path = _create_baseline_files(
            tmp_path,
            metrics={"test": {"cagr_over_vol": 2.0}},
            diagnostics={"diagnostics": {"avg_exposure": 0.6}},
        )

        results_path = tmp_path / "sweep_results.csv"
        pd.DataFrame(columns=["policy_id", "cagr_over_vol"]).to_csv(results_path, index=False)

        summary_path = tmp_path / "sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"schema_version": "9.6.9"}, f)

        output_path = tmp_path / "recommendation.json"
        result = build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_path),
        )

        assert output_path.exists()
        assert len(result["warnings"]) > 0


class TestFallback:
    """Test fallback to sweep best when guardrails filter all."""

    def test_fallback_to_sweep_best(self, tmp_path):
        """Falls back to sweep best when all policies fail guardrails."""
        metrics_path, diagnostics_path = _create_baseline_files(
            tmp_path,
            metrics={"test": {"cagr_over_vol": 5.0, "max_drawdown": -0.01}},
            diagnostics={"diagnostics": {"avg_exposure": 0.9, "turnover_proxy": 0.5}},
        )

        # All policies fail: avg_exposure < 0.9
        policies = [
            {"policy_id": 1, "beta_cap": 0.1, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.6, "std_exposure": 0.1, "turnover_proxy": 2.0,
             "n_switches": 50, "cagr_over_vol": 2.5, "cagr": 0.10, "vol": 0.04, "max_dd": -0.04},
            {"policy_id": 2, "beta_cap": 0.2, "threshold": 0.4, "n_obs": 100,
             "avg_exposure": 0.7, "std_exposure": 0.1, "turnover_proxy": 1.5,
             "n_switches": 40, "cagr_over_vol": 3.0, "cagr": 0.12, "vol": 0.04, "max_dd": -0.03},
        ]
        results_path, summary_path = _create_sweep_files(
            tmp_path, policies,
            {"schema_version": "9.6.9", "best_by_cagr_over_vol": {"policy_id": 2}},
        )

        output_path = tmp_path / "recommendation.json"
        result = build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_path),
            require_higher_avg_exposure=True,
        )

        # Should fall back to sweep best (policy_id=2)
        assert result["recommended_policy"] is not None
        assert result["recommended_policy"]["policy_id"] == 2


class TestSchemaCompliance:
    """Test output schema compliance."""

    def test_schema_version(self, tmp_path):
        """Verify schema_version is 9.6.10."""
        metrics_path, diagnostics_path = _create_baseline_files(
            tmp_path,
            metrics={"test": {"cagr_over_vol": 2.0}},
            diagnostics={"diagnostics": {"avg_exposure": 0.6}},
        )

        policies = [
            {"policy_id": 1, "beta_cap": 0.1, "threshold": 0.5, "n_obs": 100,
             "avg_exposure": 0.7, "std_exposure": 0.1, "turnover_proxy": 2.0,
             "n_switches": 50, "cagr_over_vol": 2.5, "cagr": 0.10, "vol": 0.04, "max_dd": -0.04},
        ]
        results_path, summary_path = _create_sweep_files(
            tmp_path, policies, {"schema_version": "9.6.9"},
        )

        output_path = tmp_path / "recommendation.json"
        result = build_overlay_policy_recommendation(
            overlay_metrics_json_path=str(metrics_path),
            overlay_diagnostics_json_path=str(diagnostics_path),
            policy_sweep_results_csv_path=str(results_path),
            policy_sweep_summary_json_path=str(summary_path),
            output_json_path=str(output_path),
        )

        with open(output_path) as f:
            data = json.load(f)

        assert data["schema_version"] == "9.6.10"
        assert "baseline" in data
        assert "candidates" in data
        assert "recommended_policy" in data
        assert "selection_rules" in data
        assert "warnings" in data
