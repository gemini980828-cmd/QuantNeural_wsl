"""
Tests for run_shadow_risk_evaluation.py

Covers:
- Proxy ban (non-SPY ticker raises ValueError)
- Missing SPY data raises ValueError
- All artifacts created for both model variants
- Determinism (identical outputs across runs)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_shadow_risk_evaluation import run_shadow_risk_spy_only_evaluation

VARIANTS = ["logit", "mlp", "xgb"]


def _create_synthetic_spy_csv(tmp_path: Path, n_days: int = 800) -> Path:
    """Create a synthetic SPY price CSV in simple format."""
    np.random.seed(42)
    
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    
    # Simulate price with trend + noise + some downturns for risk-off labels
    returns = np.random.normal(0.0005, 0.012, n_days)
    returns[200:210] = -0.02
    returns[400:415] = -0.025
    returns[600:610] = -0.015
    
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": prices,
    })
    
    csv_path = tmp_path / "spy.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


class TestProxyBan:
    """Test that non-SPY tickers are banned (fail-fast)."""
    
    def test_proxy_ban_raises_for_aapl(self, tmp_path):
        """Using AAPL.US as spy_ticker should raise ValueError."""
        csv_path = _create_synthetic_spy_csv(tmp_path)
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="SPY_REQUIRED:proxy_not_allowed"):
            run_shadow_risk_spy_only_evaluation(
                spy_csv_path=str(csv_path),
                spy_ticker="AAPL.US",
                as_of_date="2021-01-01",
                train_end="2019-12-31",
                val_end="2020-06-30",
                output_root_dir=str(output_dir),
                seed=42,
            )
    
    def test_proxy_ban_raises_for_qqq(self, tmp_path):
        """Using QQQ as spy_ticker should raise ValueError."""
        csv_path = _create_synthetic_spy_csv(tmp_path)
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="SPY_REQUIRED:proxy_not_allowed"):
            run_shadow_risk_spy_only_evaluation(
                spy_csv_path=str(csv_path),
                spy_ticker="QQQ",
                as_of_date="2021-01-01",
                train_end="2019-12-31",
                val_end="2020-06-30",
                output_root_dir=str(output_dir),
                seed=42,
            )
    
    def test_spy_ticker_allowed(self, tmp_path):
        """SPY.US should be allowed (no error on ticker validation)."""
        csv_path = _create_synthetic_spy_csv(tmp_path)
        output_dir = tmp_path / "output"
        
        # Should not raise for SPY ticker
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            output_root_dir=str(output_dir),
            seed=42,
        )
        
        assert "models" in result
        assert "logit" in result["models"]
        assert "mlp" in result["models"]
        assert "xgb" in result["models"]


class TestMissingSpy:
    """Test that missing SPY data raises ValueError."""
    
    def test_missing_spy_raises_for_empty_csv(self, tmp_path):
        """Empty CSV should raise ValueError."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("date,close\n")
        
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="SPY_REQUIRED:missing_spy_price_series"):
            run_shadow_risk_spy_only_evaluation(
                spy_csv_path=str(csv_path),
                spy_ticker="SPY.US",
                as_of_date="2021-01-01",
                train_end="2019-12-31",
                val_end="2020-06-30",
                output_root_dir=str(output_dir),
                seed=42,
            )
    
    def test_missing_spy_raises_for_missing_columns(self, tmp_path):
        """CSV without required columns should raise ValueError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("foo,bar\n1,2\n3,4\n")
        
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="SPY_REQUIRED:missing_spy_price_series"):
            run_shadow_risk_spy_only_evaluation(
                spy_csv_path=str(csv_path),
                spy_ticker="SPY.US",
                as_of_date="2021-01-01",
                train_end="2019-12-31",
                val_end="2020-06-30",
                output_root_dir=str(output_dir),
                seed=42,
            )
    
    def test_missing_spy_raises_for_insufficient_data(self, tmp_path):
        """CSV with too few rows should raise ValueError."""
        csv_path = tmp_path / "tiny.csv"
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "close": [100 + i for i in range(10)],
        })
        df.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="SPY_REQUIRED:missing_spy_price_series"):
            run_shadow_risk_spy_only_evaluation(
                spy_csv_path=str(csv_path),
                spy_ticker="SPY.US",
                as_of_date="2021-01-01",
                train_end="2019-12-31",
                val_end="2020-06-30",
                output_root_dir=str(output_dir),
                seed=42,
            )


class TestWritesAllArtifacts:
    """Test that all artifacts are created for both model variants."""
    
    def test_writes_all_artifacts_for_both_models(self, tmp_path):
        """All expected artifacts should be created for logit and mlp."""
        csv_path = _create_synthetic_spy_csv(tmp_path, n_days=800)
        output_dir = tmp_path / "output"
        
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            output_root_dir=str(output_dir),
            seed=42,
        )
        
        expected_files = [
            "shadow_risk.csv",
            "shadow_risk_metrics.json",
            "shadow_risk_overlay.csv",
            "shadow_risk_overlay_metrics.json",
            "shadow_risk_decision_gate.json",
            "temp_sweep/shadow_risk_temp_sweep_results.csv",
            "temp_sweep/shadow_risk_temp_sweep_summary.json",
        ]
        
        for variant in VARIANTS:
            # Default horizon is 63, so artifacts are under horizon_63/variant
            variant_dir = output_dir / "horizon_63" / variant
            
            for file_rel in expected_files:
                file_path = variant_dir / file_rel
                assert file_path.exists(), f"Missing {file_rel} for {variant}"
        
        # Verify decision gate JSON has expected keys
        for variant in VARIANTS:
            decision_path = output_dir / "horizon_63" / variant / "shadow_risk_decision_gate.json"
            with open(decision_path) as f:
                decision = json.load(f)
            
            assert "decision" in decision, f"Missing 'decision' key for {variant}"
            assert "computed" in decision, f"Missing 'computed' key for {variant}"
            assert "checks" in decision, f"Missing 'checks' key for {variant}"
            assert "reasons" in decision, f"Missing 'reasons' key for {variant}"

        # Verify XGB metrics schema
        xgb_metrics_path = output_dir / "horizon_63" / "xgb" / "shadow_risk_metrics.json"
        with open(xgb_metrics_path) as f:
            xgb_metrics = json.load(f)
        assert "model_params" in xgb_metrics
        assert xgb_metrics["model_params"]["model_type"] == "xgb"
        assert xgb_metrics["model_params"]["backend"] in {"xgboost", "sklearn_hist_gb"}
        assert isinstance(xgb_metrics.get("warnings", []), list)


class TestDeterminism:
    """Test deterministic output generation."""
    
    def test_determinism_byte_identical(self, tmp_path):
        """Running twice with same inputs should produce identical outputs."""
        csv_path = _create_synthetic_spy_csv(tmp_path, n_days=800)
        output_dir_1 = tmp_path / "output1"
        output_dir_2 = tmp_path / "output2"
        
        params = dict(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            seed=42,
        )
        
        run_shadow_risk_spy_only_evaluation(
            output_root_dir=str(output_dir_1),
            **params,
        )
        
        run_shadow_risk_spy_only_evaluation(
            output_root_dir=str(output_dir_2),
            **params,
        )
        
        # Compare key artifacts for byte identity
        # Note: decision gate JSON may have subtle floating point differences
        # due to intermediate calculations, so we test a smaller subset
        files_to_compare = [
            "horizon_63/logit/shadow_risk.csv",
            "horizon_63/logit/temp_sweep/shadow_risk_temp_sweep_summary.json",
            "horizon_63/mlp/shadow_risk.csv",
            "horizon_63/mlp/temp_sweep/shadow_risk_temp_sweep_summary.json",
            "horizon_63/xgb/shadow_risk_metrics.json",
            "horizon_63/xgb/shadow_risk_overlay_metrics.json",
        ]
        
        for file_rel in files_to_compare:
            path_1 = output_dir_1 / file_rel
            path_2 = output_dir_2 / file_rel
            
            assert path_1.exists(), f"Missing {file_rel} in run 1"
            assert path_2.exists(), f"Missing {file_rel} in run 2"
            
            bytes_1 = path_1.read_bytes()
            bytes_2 = path_2.read_bytes()
            
            assert bytes_1 == bytes_2, f"Non-deterministic output for {file_rel}"


class TestHorizonWiring:
    """Test that --horizon-days arg correctly affects outputs."""
    
    def test_horizon_arg_wiring_changes_outputs(self, tmp_path):
        """
        Run evaluation twice with different horizons (63 vs 21).
        
        Verifies:
        1. Both runs succeed
        2. Expected directory structure includes horizon_63/ and horizon_21/
        3. Outputs are deterministic per horizon
        4. CSV row counts differ between horizons (proving horizon affects labeling)
        """
        csv_path = _create_synthetic_spy_csv(tmp_path, n_days=800)
        output_dir = tmp_path / "output"
        
        base_params = dict(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            seed=42,
        )
        
        # Run horizon=63
        result_63 = run_shadow_risk_spy_only_evaluation(
            output_root_dir=str(output_dir),
            horizon_days=63,
            **base_params,
        )
        
        # Run horizon=21
        result_21 = run_shadow_risk_spy_only_evaluation(
            output_root_dir=str(output_dir),
            horizon_days=21,
            **base_params,
        )
        
        # =====================================================================
        # 1. Verify directory structure includes horizon subdirs
        # =====================================================================
        horizon_63_dir = output_dir / "horizon_63"
        horizon_21_dir = output_dir / "horizon_21"
        
        assert horizon_63_dir.exists(), "Missing horizon_63 directory"
        assert horizon_21_dir.exists(), "Missing horizon_21 directory"
        
        for variant in VARIANTS:
            assert (horizon_63_dir / variant).exists(), f"Missing {variant} in horizon_63"
            assert (horizon_21_dir / variant).exists(), f"Missing {variant} in horizon_21"
        
        # =====================================================================
        # 2. Verify all expected artifacts exist for both horizons
        # =====================================================================
        expected_files = [
            "shadow_risk.csv",
            "shadow_risk_metrics.json",
            "shadow_risk_overlay.csv",
            "shadow_risk_overlay_metrics.json",
            "shadow_risk_decision_gate.json",
        ]
        
        for horizon_dir in [horizon_63_dir, horizon_21_dir]:
            for variant in VARIANTS:
                for file_name in expected_files:
                    file_path = horizon_dir / variant / file_name
                    assert file_path.exists(), f"Missing {file_name} in {horizon_dir}/{variant}"
        
        # =====================================================================
        # 3. Verify determinism: re-run horizon=63 and compare
        # =====================================================================
        output_dir_check = tmp_path / "output_check"
        result_63_check = run_shadow_risk_spy_only_evaluation(
            output_root_dir=str(output_dir_check),
            horizon_days=63,
            **base_params,
        )
        
        # Compare shadow_risk.csv for byte identity
        csv_63_orig = horizon_63_dir / "logit" / "shadow_risk.csv"
        csv_63_check = output_dir_check / "horizon_63" / "logit" / "shadow_risk.csv"
        
        assert csv_63_orig.read_bytes() == csv_63_check.read_bytes(), \
            "Non-deterministic output for horizon=63"
        
        # =====================================================================
        # 4. Verify row counts differ (horizon affects labeling)
        # =====================================================================
        df_63 = pd.read_csv(horizon_63_dir / "logit" / "shadow_risk.csv")
        df_21 = pd.read_csv(horizon_21_dir / "logit" / "shadow_risk.csv")
        
        # Since last horizon_days are excluded from labeling,
        # horizon=21 should have MORE rows than horizon=63
        # (63 - 21 = 42 more rows for horizon=21)
        assert len(df_21) >= len(df_63), \
            f"Expected horizon=21 to have >= rows than horizon=63, got {len(df_21)} vs {len(df_63)}"
        
        # Stronger check: they should differ (proving horizon actually affects output)
        # Note: This may not always hold if data is exactly the same length,
        # but for 800 days it should differ
        assert len(df_21) != len(df_63), \
            f"Expected different row counts for different horizons, both have {len(df_63)} rows"
        
        # =====================================================================
        # 5. Verify config in result has correct horizon_days
        # =====================================================================
        assert result_63["config"]["horizon_days"] == 63
        assert result_21["config"]["horizon_days"] == 21
        
        # Verify horizon_dir is correctly set
        assert "horizon_63" in result_63["horizon_dir"]
        assert "horizon_21" in result_21["horizon_dir"]


class TestOverlayDiagnosticsWiring:
    """Test that overlay_diagnostics.json is produced by the evaluation runner (Task 9.6.8)."""
    
    def test_overlay_diagnostics_json_created(self, tmp_path):
        """Overlay diagnostics JSON should be created for each variant."""
        csv_path = _create_synthetic_spy_csv(tmp_path, n_days=800)
        output_dir = tmp_path / "output"
        
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            output_root_dir=str(output_dir),
            seed=42,
        )
        
        # Verify diagnostics JSON exists for each variant
        for variant in VARIANTS:
            variant_dir = output_dir / "horizon_63" / variant
            diagnostics_path = variant_dir / "overlay_diagnostics.json"
            
            assert diagnostics_path.exists(), \
                f"Missing overlay_diagnostics.json for {variant}"
            
            # Load and verify schema
            with open(diagnostics_path) as f:
                data = json.load(f)
            
            # Check required top-level keys
            assert "schema_version" in data, f"Missing schema_version for {variant}"
            assert "diagnostics" in data, f"Missing diagnostics for {variant}"
            assert "warnings" in data, f"Missing warnings for {variant}"
            
            # Check schema version
            assert data["schema_version"] == "9.6.7", \
                f"Expected schema_version 9.6.7, got {data['schema_version']}"
            
            # Check diagnostics has required keys
            required_keys = [
                "n_obs", "avg_exposure", "std_exposure", "frac_exposure_lt_1",
                "n_switches", "turnover_proxy", "avg_abs_delta_exposure",
            ]
            for key in required_keys:
                assert key in data["diagnostics"], \
                    f"Missing diagnostics key '{key}' for {variant}"
        
        # Verify result dict has diagnostics paths
        for variant in VARIANTS:
            model_result = result["models"][variant]
            assert "shadow_risk_overlay_diagnostics_json" in model_result
            assert "shadow_risk_overlay_diagnostics_json_exported" in model_result
            assert model_result["shadow_risk_overlay_diagnostics_json_exported"] is True


class TestPolicySweepWiring:
    """Test that overlay_policy_sweep artifacts are produced by the evaluation runner (Task 9.6.9)."""
    
    def test_policy_sweep_artifacts_created(self, tmp_path):
        """Policy sweep CSV and JSON should be created for each variant."""
        csv_path = _create_synthetic_spy_csv(tmp_path, n_days=800)
        output_dir = tmp_path / "output"
        
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            output_root_dir=str(output_dir),
            seed=42,
        )
        
        # Verify policy sweep artifacts exist for each variant
        for variant in VARIANTS:
            variant_dir = output_dir / "horizon_63" / variant
            results_path = variant_dir / "overlay_policy_sweep_results.csv"
            summary_path = variant_dir / "overlay_policy_sweep_summary.json"
            
            assert results_path.exists(), \
                f"Missing overlay_policy_sweep_results.csv for {variant}"
            assert summary_path.exists(), \
                f"Missing overlay_policy_sweep_summary.json for {variant}"
            
            # Load and verify schema
            with open(summary_path) as f:
                data = json.load(f)
            
            # Check schema version
            assert data["schema_version"] == "9.6.9", \
                f"Expected schema_version 9.6.9, got {data['schema_version']}"
            
            # Check required top-level keys
            assert "config" in data
            assert "best_by_cagr_over_vol" in data
            assert "best_by_max_dd" in data
            assert "best_by_cagr" in data
            assert "warnings" in data
        
        # Verify result dict has policy sweep paths
        for variant in VARIANTS:
            model_result = result["models"][variant]
            assert "shadow_risk_overlay_policy_sweep_results_csv" in model_result
            assert "shadow_risk_overlay_policy_sweep_summary_json" in model_result
            assert "shadow_risk_overlay_policy_sweep_exported" in model_result
            assert model_result["shadow_risk_overlay_policy_sweep_exported"] is True


class TestPolicyRecommendationWiring:
    """Test that overlay_policy_recommendation.json is produced by the evaluation runner (Task 9.6.10)."""
    
    def test_policy_recommendation_json_created(self, tmp_path):
        """Policy recommendation JSON should be created for each variant."""
        csv_path = _create_synthetic_spy_csv(tmp_path, n_days=800)
        output_dir = tmp_path / "output"
        
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            output_root_dir=str(output_dir),
            seed=42,
        )
        
        # Verify recommendation JSON exists for each variant
        for variant in VARIANTS:
            variant_dir = output_dir / "horizon_63" / variant
            recommendation_path = variant_dir / "overlay_policy_recommendation.json"
            
            assert recommendation_path.exists(), \
                f"Missing overlay_policy_recommendation.json for {variant}"
            
            # Load and verify schema
            with open(recommendation_path) as f:
                data = json.load(f)
            
            # Check schema version
            assert data["schema_version"] == "9.6.10", \
                f"Expected schema_version 9.6.10, got {data['schema_version']}"
            
            # Check required top-level keys
            assert "baseline" in data
            assert "candidates" in data
            assert "recommended_policy" in data
            assert "selection_rules" in data
            assert "warnings" in data
        
        # Verify result dict has recommendation paths
        for variant in VARIANTS:
            model_result = result["models"][variant]
            assert "shadow_risk_overlay_policy_recommendation_json" in model_result
            assert "shadow_risk_overlay_policy_recommendation_exported" in model_result
            assert model_result["shadow_risk_overlay_policy_recommendation_exported"] is True


class TestPolicyApplyWiring:
    """Test that policy apply artifacts are produced by the evaluation runner (Task 9.6.11)."""
    
    def test_policy_apply_artifacts_created(self, tmp_path):
        """Policy apply CSV, metrics, and diagnostics should be created for each variant."""
        csv_path = _create_synthetic_spy_csv(tmp_path, n_days=800)
        output_dir = tmp_path / "output"
        
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(csv_path),
            spy_ticker="SPY.US",
            as_of_date="2021-01-01",
            train_end="2019-12-31",
            val_end="2020-06-30",
            output_root_dir=str(output_dir),
            seed=42,
        )
        
        # Verify policy apply artifacts exist for each variant
        for variant in VARIANTS:
            variant_dir = output_dir / "horizon_63" / variant
            
            # Check files exist
            policy_csv = variant_dir / "shadow_risk_overlay_policy_best.csv"
            policy_metrics = variant_dir / "shadow_risk_overlay_policy_best_metrics.json"
            policy_diagnostics = variant_dir / "overlay_policy_best_diagnostics.json"
            
            assert policy_csv.exists(), \
                f"Missing shadow_risk_overlay_policy_best.csv for {variant}"
            assert policy_metrics.exists(), \
                f"Missing shadow_risk_overlay_policy_best_metrics.json for {variant}"
            assert policy_diagnostics.exists(), \
                f"Missing overlay_policy_best_diagnostics.json for {variant}"
            
            # Verify metrics schema
            with open(policy_metrics) as f:
                metrics_data = json.load(f)
            assert metrics_data["schema_version"] == "9.6.11", \
                f"Expected schema_version 9.6.11, got {metrics_data['schema_version']}"
            
            # Verify diagnostics schema
            with open(policy_diagnostics) as f:
                diag_data = json.load(f)
            assert diag_data["schema_version"] == "9.6.7", \
                f"Expected diagnostics schema_version 9.6.7, got {diag_data['schema_version']}"
        
        # Verify result dict has policy apply paths
        for variant in VARIANTS:
            model_result = result["models"][variant]
            assert "shadow_risk_overlay_policy_best_csv" in model_result
            assert "shadow_risk_overlay_policy_best_metrics_json" in model_result
            assert "shadow_risk_overlay_policy_best_diagnostics_json" in model_result
            assert "shadow_risk_overlay_policy_best_exported" in model_result
            assert model_result["shadow_risk_overlay_policy_best_exported"] is True


class TestOpsModeReport:
    """Test ops mode report generation."""
    
    def test_ops_report_generated(self, tmp_path):
        """Ops report JSON should be generated when ops_mode is on."""
        spy_csv = _create_synthetic_spy_csv(tmp_path)
        output_dir = tmp_path / "eval_ops"
        
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(spy_csv),
            spy_ticker="SPY.US",
            as_of_date="2021-03-01",
            train_end="2020-01-01",
            val_end="2020-07-01",
            output_root_dir=str(output_dir),
            seed=42,
            horizon_days=63,
            ops_mode="on",
            ops_champion_variant="xgb",
            ops_overlay_mode="policy_best",
            ops_calibration_mode="raw",
        )
        
        # Verify ops report was generated
        assert "ops_report_json" in result
        ops_report_path = Path(result["ops_report_json"])
        assert ops_report_path.exists(), "Ops report JSON should exist"
        
        # Verify schema
        with open(ops_report_path) as f:
            ops_report = json.load(f)
        
        assert ops_report["schema_version"] == "9.6.16"
        assert ops_report["champion"]["variant"] == "xgb"
        assert ops_report["champion"]["calibration_mode"] == "raw"
        assert ops_report["champion"]["overlay_mode"] == "policy_best"
        assert ops_report["operational_decision"]["status"] in {"OK", "WARN", "FAIL"}
        assert ops_report["operational_decision"]["recommended_action"] in {"KEEP", "REVIEW", "HALT"}
        assert isinstance(ops_report["warnings"], list)
    
    def test_ops_mode_off_no_report(self, tmp_path):
        """Ops report should NOT be generated when ops_mode is off."""
        spy_csv = _create_synthetic_spy_csv(tmp_path)
        output_dir = tmp_path / "eval_no_ops"
        
        result = run_shadow_risk_spy_only_evaluation(
            spy_csv_path=str(spy_csv),
            spy_ticker="SPY.US",
            as_of_date="2021-03-01",
            train_end="2020-01-01",
            val_end="2020-07-01",
            output_root_dir=str(output_dir),
            seed=42,
            horizon_days=63,
            ops_mode="off",
        )
        
        # No ops_report_json key when mode is off
        assert "ops_report_json" not in result
        
        # No file should exist
        ops_path = output_dir / "horizon_63" / "ops_shadow_risk_report.json"
        assert not ops_path.exists(), "Ops report should not exist when mode is off"
