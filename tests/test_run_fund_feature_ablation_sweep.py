"""
Tests for run_fund_feature_ablation_sweep.py

Task 10.2.11: Tests for ablation stability sweep with monkeypatched stubs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Stubbed ablation suite result generator
def _make_stub_result(seed: int, window_id: str, make_fund_better: bool = True) -> dict:
    """Generate a stubbed ablation suite result."""
    np.random.seed(seed)
    
    modes = {}
    for mode in ["fund_full", "fund_zeroed", "fund_shuffled", "tech_only"]:
        base_ic = 0.02 + np.random.uniform(-0.005, 0.005)
        base_delta = 0.10 + np.random.uniform(-0.02, 0.02)
        
        # Make fund_full better if requested
        if make_fund_better and mode == "fund_full":
            base_ic += 0.005
            base_delta += 0.02
        
        # Make tech_only baseline
        if mode == "tech_only":
            base_ic -= 0.003
            base_delta -= 0.01
        
        # Make shuffled worse
        if mode == "fund_shuffled":
            base_ic -= 0.008
            base_delta -= 0.03
        
        modes[mode] = {
            "feature_mode": mode,
            "training_success": True,
            "backtest_success": True,
            "ic_mean": base_ic,
            "delta_cagr_vol_all": base_delta,
            "delta_cagr_vol_sec_covered": base_delta * 0.9,
            "delta_cagr_vol_sec_missing": base_delta * 1.1,
            "n_features_used": 25 if mode != "tech_only" else 15,
            "warnings": [],
        }
    
    return {
        "report_path": f"/tmp/stub/{window_id}_seed{seed}/REPORT_10_2_10.md",
        "summary_path": f"/tmp/stub/{window_id}_seed{seed}/ablation_summary.json",
        "modes": modes,
    }


class TestRunFundFeatureAblationSweep:
    """Tests for ablation stability sweep."""
    
    def test_sweep_generates_artifacts(self, tmp_path, monkeypatch):
        """Sweep should generate all three output files."""
        from scripts import run_fund_feature_ablation_sweep
        from scripts.run_fund_feature_ablation_sweep import run_fund_feature_ablation_sweep as sweep_fn
        
        call_count = [0]
        
        def mock_ablation_suite(**kwargs):
            call_count[0] += 1
            seed = kwargs.get("seed", 42)
            train_end = kwargs.get("train_end", "2020-01-01")
            return _make_stub_result(seed, f"W_{train_end[:4]}")
        
        monkeypatch.setattr(
            run_fund_feature_ablation_sweep,
            "run_fund_feature_ablation_suite",
            mock_ablation_suite
        )
        
        result = sweep_fn(
            fund_alpha_dataset_path="/tmp/fake/dataset.csv.gz",
            prices_csv_path="/tmp/fake/prices.csv",
            baseline_scores_csv_path="/tmp/fake/baseline.csv",
            out_dir=str(tmp_path),
            as_of_date="2024-10-01",
            windows=[
                {"id": "W1", "train_end": "2014-12-31", "val_end": "2016-12-31"},
            ],
            seeds=[42, 43],
        )
        
        # Verify files exist
        assert Path(result["csv_path"]).exists()
        assert Path(result["json_path"]).exists()
        assert Path(result["report_path"]).exists()
        
        # Verify call count (1 window × 2 seeds)
        assert call_count[0] == 2
        
        # Verify JSON schema
        with open(result["json_path"]) as f:
            summary = json.load(f)
        assert summary["schema_version"] == "10.2.11"
        assert "aggregated_metrics" in summary
        assert "per_mode" in summary["aggregated_metrics"]
        
        # Verify CSV has expected columns
        df = pd.read_csv(result["csv_path"])
        assert "window_id" in df.columns
        assert "seed" in df.columns
        assert "mode" in df.columns
        assert "ic_mean" in df.columns
        assert "delta_all" in df.columns
    
    def test_sweep_aggregates_correctly(self, tmp_path, monkeypatch):
        """Aggregated metrics should be computed correctly."""
        from scripts import run_fund_feature_ablation_sweep
        from scripts.run_fund_feature_ablation_sweep import run_fund_feature_ablation_sweep as sweep_fn
        
        def mock_ablation_suite(**kwargs):
            seed = kwargs.get("seed", 42)
            return _make_stub_result(seed, "W1", make_fund_better=True)
        
        monkeypatch.setattr(
            run_fund_feature_ablation_sweep,
            "run_fund_feature_ablation_suite",
            mock_ablation_suite
        )
        
        result = sweep_fn(
            fund_alpha_dataset_path="/tmp/fake/dataset.csv.gz",
            prices_csv_path="/tmp/fake/prices.csv",
            baseline_scores_csv_path="/tmp/fake/baseline.csv",
            out_dir=str(tmp_path),
            as_of_date="2024-10-01",
            windows=[{"id": "W1", "train_end": "2014-12-31", "val_end": "2016-12-31"}],
            seeds=[42, 43, 44],
        )
        
        agg = result["aggregated_metrics"]
        
        # All modes should have 3 runs each
        for mode in ["fund_full", "tech_only", "fund_zeroed", "fund_shuffled"]:
            assert agg["per_mode"][mode]["n_runs"] == 3
        
        # fund_full should beat tech_only in win rate (since make_fund_better=True)
        win_rate = agg["win_rates"]["fund_full_minus_tech_only"]["rate"]
        assert win_rate == 1.0, f"Expected 100% win rate, got {win_rate}"
        
        # fund_full delta should be higher than tech_only
        fund_delta = agg["per_mode"]["fund_full"]["delta_all_mean"]
        tech_delta = agg["per_mode"]["tech_only"]["delta_all_mean"]
        assert fund_delta > tech_delta
    
    def test_sweep_failsafe(self, tmp_path, monkeypatch):
        """Sweep should continue even if one run fails."""
        from scripts import run_fund_feature_ablation_sweep
        from scripts.run_fund_feature_ablation_sweep import run_fund_feature_ablation_sweep as sweep_fn
        
        call_count = [0]
        
        def mock_ablation_suite_with_failure(**kwargs):
            call_count[0] += 1
            seed = kwargs.get("seed", 42)
            
            # Fail on seed 43
            if seed == 43:
                raise ValueError("Simulated failure for seed 43")
            
            return _make_stub_result(seed, "W1")
        
        monkeypatch.setattr(
            run_fund_feature_ablation_sweep,
            "run_fund_feature_ablation_suite",
            mock_ablation_suite_with_failure
        )
        
        # Should not raise
        result = sweep_fn(
            fund_alpha_dataset_path="/tmp/fake/dataset.csv.gz",
            prices_csv_path="/tmp/fake/prices.csv",
            baseline_scores_csv_path="/tmp/fake/baseline.csv",
            out_dir=str(tmp_path),
            as_of_date="2024-10-01",
            windows=[{"id": "W1", "train_end": "2014-12-31", "val_end": "2016-12-31"}],
            seeds=[42, 43, 44],
        )
        
        # All seeds should be attempted
        assert call_count[0] == 3
        
        # Files should still exist
        assert Path(result["csv_path"]).exists()
        assert Path(result["json_path"]).exists()
        
        # Warning should be recorded
        with open(result["json_path"]) as f:
            summary = json.load(f)
        assert len(summary["warnings"]) > 0
        assert any("43" in w for w in summary["warnings"])
    
    def test_sweep_report_content(self, tmp_path, monkeypatch):
        """Report should contain expected sections."""
        from scripts import run_fund_feature_ablation_sweep
        from scripts.run_fund_feature_ablation_sweep import run_fund_feature_ablation_sweep as sweep_fn
        
        def mock_ablation_suite(**kwargs):
            seed = kwargs.get("seed", 42)
            return _make_stub_result(seed, "W1")
        
        monkeypatch.setattr(
            run_fund_feature_ablation_sweep,
            "run_fund_feature_ablation_suite",
            mock_ablation_suite
        )
        
        result = sweep_fn(
            fund_alpha_dataset_path="/tmp/fake/dataset.csv.gz",
            prices_csv_path="/tmp/fake/prices.csv",
            baseline_scores_csv_path="/tmp/fake/baseline.csv",
            out_dir=str(tmp_path),
            as_of_date="2024-10-01",
            windows=[{"id": "W1", "train_end": "2014-12-31", "val_end": "2016-12-31"}],
            seeds=[42, 43],
        )
        
        report = Path(result["report_path"]).read_text()
        
        # Check expected sections
        assert "# Task 10.2.11" in report
        assert "## Run Matrix" in report
        assert "## Aggregated Results" in report
        assert "## Pairwise Differences" in report
        assert "## Verdict" in report
        assert "VERDICT:" in report
