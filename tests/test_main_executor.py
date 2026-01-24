"""
Tests for main_executor.py

Covers:
- run_synthetic_smoke produces correct output structure
- X_mlp_shape is (T, 20) where T is the chosen constant
- yhat_shape has 10 columns
- selected_features are valid feature names
- regime_brier is finite
- regime_action_last is AGGRESSIVE or DEFENSIVE
- Config loading and validation
- Phase 5 fail-fast checks
"""

from pathlib import Path

import numpy as np
import pytest


class TestRunSyntheticSmoke:
    """Test run_synthetic_smoke function."""
    
    @pytest.fixture
    def smoke_result(self):
        """Run smoke test and return results."""
        from main_executor import run_synthetic_smoke
        return run_synthetic_smoke(seed=42)
    
    def test_x_mlp_shape(self, smoke_result):
        """Test that X_mlp shape is (T, 20) where T=24."""
        assert smoke_result["X_mlp_shape"] == (24, 20)
    
    def test_yhat_shape_columns(self, smoke_result):
        """Test that yhat has 10 columns."""
        assert smoke_result["yhat_shape"][1] == 10
    
    def test_yhat_shape_rows(self, smoke_result):
        """Test that yhat has positive number of rows."""
        assert smoke_result["yhat_shape"][0] > 0
    
    def test_selected_features_is_list(self, smoke_result):
        """Test that selected_features is a list of strings."""
        selected = smoke_result["selected_features"]
        assert isinstance(selected, list)
        assert all(isinstance(f, str) for f in selected)
    
    def test_selected_features_valid(self, smoke_result):
        """Test that selected features are within valid feature names."""
        selected = smoke_result["selected_features"]
        valid_names = smoke_result["feature_names_sel"]
        assert all(f in valid_names for f in selected)
    
    def test_regime_brier_finite(self, smoke_result):
        """Test that regime_brier is a finite float."""
        brier = smoke_result["regime_brier"]
        assert isinstance(brier, float)
        assert np.isfinite(brier)
    
    def test_regime_action_valid(self, smoke_result):
        """Test that regime_action_last is AGGRESSIVE or DEFENSIVE."""
        action = smoke_result["regime_action_last"]
        assert action in {"AGGRESSIVE", "DEFENSIVE"}
    
    def test_smoke_runs_without_exception(self):
        """Test that smoke run completes without exceptions."""
        from main_executor import run_synthetic_smoke
        # Different seed to test robustness
        result = run_synthetic_smoke(seed=123)
        assert "X_mlp_shape" in result
        assert "yhat_shape" in result
        assert "selected_features" in result
        assert "regime_brier" in result
        assert "regime_action_last" in result


class TestConfigLoading:
    """Test config loading and validation."""
    
    def test_run_from_config_happy_path(self, tmp_path):
        """Test that valid config loads and runs successfully."""
        from main_executor import run_from_config
        
        # Create minimal valid config
        config_content = """
project:
  seed: 123
  as_of_date: "2025-12-31"
models:
  type: "mlp"
regime:
  calibration:
    enabled: false
conformal:
  enabled: false
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        out = run_from_config(str(config_file))
        
        # Verify config echoes
        assert out["config_seed"] == 123
        assert out["config_as_of_date"] == "2025-12-31"
        
        # Verify smoke outputs
        assert out["X_mlp_shape"][1] == 20
        assert out["yhat_shape"][1] == 10
    
    def test_conformal_enabled_raises_runtime_error(self, tmp_path):
        """Test that conformal.enabled=true raises RuntimeError."""
        from main_executor import run_from_config
        
        config_content = """
project:
  seed: 42
  as_of_date: "2025-12-31"
conformal:
  enabled: true
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        with pytest.raises(RuntimeError, match="conformal"):
            run_from_config(str(config_file))
    
    def test_regime_calibration_enabled_raises_runtime_error(self, tmp_path):
        """Test that regime.calibration.enabled=true raises RuntimeError."""
        from main_executor import run_from_config
        
        config_content = """
project:
  seed: 42
  as_of_date: "2025-12-31"
regime:
  calibration:
    enabled: true
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        with pytest.raises(RuntimeError, match="calibration"):
            run_from_config(str(config_file))
    
    def test_model_type_not_mlp_raises_runtime_error(self, tmp_path):
        """Test that models.type != 'mlp' raises RuntimeError."""
        from main_executor import run_from_config
        
        config_content = """
project:
  seed: 42
  as_of_date: "2025-12-31"
models:
  type: "kan"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        with pytest.raises(RuntimeError, match="models.type"):
            run_from_config(str(config_file))
    
    def test_missing_project_seed_raises_value_error(self, tmp_path):
        """Test that missing project.seed raises ValueError."""
        from main_executor import run_from_config
        
        config_content = """
project:
  as_of_date: "2025-12-31"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        with pytest.raises(ValueError, match="project.seed"):
            run_from_config(str(config_file))
    
    def test_missing_project_as_of_date_raises_value_error(self, tmp_path):
        """Test that missing project.as_of_date raises ValueError."""
        from main_executor import run_from_config
        
        config_content = """
project:
  seed: 42
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        with pytest.raises(ValueError, match="project.as_of_date"):
            run_from_config(str(config_file))
    
    def test_missing_project_section_raises_value_error(self, tmp_path):
        """Test that missing project section raises ValueError."""
        from main_executor import run_from_config
        
        config_content = """
models:
  type: "mlp"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        with pytest.raises(ValueError, match="project"):
            run_from_config(str(config_file))


class TestArtifactSaving:
    """Test artifact saving functionality."""
    
    def test_saves_artifacts_when_artifacts_dir_provided(self, tmp_path):
        """Test that artifacts are saved when artifacts_dir is provided."""
        import json
        from main_executor import run_from_config
        
        # Create config
        config_content = """
project:
  seed: 123
  as_of_date: "2025-12-31"
models:
  type: "mlp"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        artifacts_dir = tmp_path / "artifacts"
        
        out = run_from_config(str(config_file), artifacts_dir=str(artifacts_dir))
        
        # Assert artifacts_run_dir is in output
        assert "artifacts_run_dir" in out
        run_dir = Path(out["artifacts_run_dir"])
        
        # Assert directory exists
        assert run_dir.exists()
        assert run_dir.is_dir()
        
        # Assert manifest.json exists and contains expected keys
        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert "run_id" in manifest
        assert manifest["config_seed"] == 123
        assert manifest["config_as_of_date"] == "2025-12-31"
        
        # Assert selected_features.json exists and contains a list
        features_path = run_dir / "selected_features.json"
        assert features_path.exists()
        with open(features_path, encoding="utf-8") as f:
            features = json.load(f)
        assert isinstance(features, list)
        
        # Assert model file exists
        model_path = run_dir / "mlp.keras"
        assert model_path.exists()
    
    def test_run_id_is_deterministic(self, tmp_path):
        """Test that run_id is deterministic based on seed and as_of_date."""
        from main_executor import run_from_config
        
        config_content = """
project:
  seed: 42
  as_of_date: "2024-06-15"
models:
  type: "mlp"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        artifacts_dir = tmp_path / "artifacts"
        
        out = run_from_config(str(config_file), artifacts_dir=str(artifacts_dir))
        
        # run_id should match seed/as_of_date exactly
        expected_run_id = "seed42_asof2024-06-15"
        run_dir = Path(out["artifacts_run_dir"])
        assert run_dir.name == expected_run_id
    
    def test_no_artifacts_when_dir_not_provided(self, tmp_path):
        """Test that no artifacts_run_dir when artifacts_dir is None."""
        from main_executor import run_from_config
        
        config_content = """
project:
  seed: 123
  as_of_date: "2025-12-31"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content, encoding="utf-8")
        
        out = run_from_config(str(config_file))
        
        # Should not have artifacts_run_dir
        assert "artifacts_run_dir" not in out
