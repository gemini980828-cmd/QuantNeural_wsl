"""
Tests for MLP defaults and metrics provenance in shadow_risk_exposure.

Locks the Task 9.6.12 MLP defaults and ensures model_params is recorded
in the metrics JSON for reproducibility and comparison.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _create_synthetic_prices(tmp_path: Path, n_days: int = 800) -> pd.DataFrame:
    """Create synthetic SPY price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2017-01-01", periods=n_days, freq="B")
    
    # Generate random returns
    returns = np.random.randn(n_days) * 0.01 + 0.0002
    
    # Convert to prices
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({"SPY.US": prices}, index=dates)
    return df


class TestMLPDefaults:
    """Test that MLP uses Task 9.6.12 defaults."""
    
    def test_model_params_in_metrics(self, tmp_path):
        """Metrics JSON should contain model_params with required keys."""
        from src.shadow_risk_exposure import run_shadow_risk_exposure_mlp_with_metrics
        
        prices = _create_synthetic_prices(tmp_path)
        output_csv = tmp_path / "shadow_risk.csv"
        output_metrics = tmp_path / "shadow_risk_metrics.json"
        
        run_shadow_risk_exposure_mlp_with_metrics(
            prices=prices,
            as_of_date="2020-01-01",
            train_end="2019-01-01",
            val_end="2019-07-01",
            output_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            spy_ticker="SPY.US",
            horizon_days=63,
            seed=42,
        )
        
        # Load metrics JSON
        with open(output_metrics) as f:
            metrics = json.load(f)
        
        # Check model_params exists
        assert "model_params" in metrics, "metrics must contain model_params"
        
        # Check required keys
        model_params = metrics["model_params"]
        required_keys = ["model_type", "hidden_layer_sizes", "alpha", "max_iter", "tol", "random_state"]
        for key in required_keys:
            assert key in model_params, f"model_params must contain {key}"
        
        # Verify model_type
        assert model_params["model_type"] == "mlp"
    
    def test_default_values_match_task_9612(self, tmp_path):
        """Default values should match Task 9.6.12 specification."""
        from src.shadow_risk_exposure import run_shadow_risk_exposure_mlp_with_metrics
        
        prices = _create_synthetic_prices(tmp_path)
        output_csv = tmp_path / "shadow_risk.csv"
        output_metrics = tmp_path / "shadow_risk_metrics.json"
        
        # Run with no explicit MLP params (use defaults)
        run_shadow_risk_exposure_mlp_with_metrics(
            prices=prices,
            as_of_date="2020-01-01",
            train_end="2019-01-01",
            val_end="2019-07-01",
            output_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            spy_ticker="SPY.US",
            seed=42,
        )
        
        with open(output_metrics) as f:
            metrics = json.load(f)
        
        model_params = metrics["model_params"]
        
        # Task 9.6.12 defaults
        assert model_params["hidden_layer_sizes"] == [8, 4], \
            f"Expected [8, 4], got {model_params['hidden_layer_sizes']}"
        assert model_params["alpha"] == 0.01, \
            f"Expected 0.01, got {model_params['alpha']}"
        assert model_params["max_iter"] == 1000, \
            f"Expected 1000, got {model_params['max_iter']}"
        assert model_params["tol"] == 0.01, \
            f"Expected 0.01, got {model_params['tol']}"
        assert model_params["random_state"] == 42, \
            f"Expected 42, got {model_params['random_state']}"
    
    def test_schema_version(self, tmp_path):
        """Metrics JSON should have schema_version 9.6.12.1."""
        from src.shadow_risk_exposure import run_shadow_risk_exposure_mlp_with_metrics
        
        prices = _create_synthetic_prices(tmp_path)
        output_csv = tmp_path / "shadow_risk.csv"
        output_metrics = tmp_path / "shadow_risk_metrics.json"
        
        run_shadow_risk_exposure_mlp_with_metrics(
            prices=prices,
            as_of_date="2020-01-01",
            train_end="2019-01-01",
            val_end="2019-07-01",
            output_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            spy_ticker="SPY.US",
            seed=42,
        )
        
        with open(output_metrics) as f:
            metrics = json.load(f)
        
        assert metrics["schema_version"] == "9.6.12.1"


class TestDeterminism:
    """Test that MLP metrics are deterministic."""
    
    def test_byte_identical_metrics(self, tmp_path):
        """Running twice with same seed produces byte-identical metrics JSON."""
        from src.shadow_risk_exposure import run_shadow_risk_exposure_mlp_with_metrics
        
        prices = _create_synthetic_prices(tmp_path)
        
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        dir2 = tmp_path / "run2"
        dir2.mkdir()
        
        # First run
        run_shadow_risk_exposure_mlp_with_metrics(
            prices=prices,
            as_of_date="2020-01-01",
            train_end="2019-01-01",
            val_end="2019-07-01",
            output_csv_path=str(dir1 / "shadow_risk.csv"),
            output_metrics_json_path=str(dir1 / "metrics.json"),
            spy_ticker="SPY.US",
            seed=42,
        )
        
        # Second run
        run_shadow_risk_exposure_mlp_with_metrics(
            prices=prices,
            as_of_date="2020-01-01",
            train_end="2019-01-01",
            val_end="2019-07-01",
            output_csv_path=str(dir2 / "shadow_risk.csv"),
            output_metrics_json_path=str(dir2 / "metrics.json"),
            spy_ticker="SPY.US",
            seed=42,
        )
        
        # Compare bytes
        bytes1 = (dir1 / "metrics.json").read_bytes()
        bytes2 = (dir2 / "metrics.json").read_bytes()
        assert bytes1 == bytes2, "Metrics JSON should be byte-identical across runs"


class TestOverriddenParams:
    """Test that overridden MLP params are recorded correctly."""
    
    def test_overridden_alpha_recorded(self, tmp_path):
        """Overriding alpha should be reflected in metrics JSON."""
        from src.shadow_risk_exposure import run_shadow_risk_exposure_mlp_with_metrics
        
        prices = _create_synthetic_prices(tmp_path)
        output_csv = tmp_path / "shadow_risk.csv"
        output_metrics = tmp_path / "shadow_risk_metrics.json"
        
        custom_alpha = 0.05
        
        run_shadow_risk_exposure_mlp_with_metrics(
            prices=prices,
            as_of_date="2020-01-01",
            train_end="2019-01-01",
            val_end="2019-07-01",
            output_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            spy_ticker="SPY.US",
            seed=42,
            alpha=custom_alpha,  # Override default
        )
        
        with open(output_metrics) as f:
            metrics = json.load(f)
        
        assert metrics["model_params"]["alpha"] == custom_alpha, \
            f"Expected alpha={custom_alpha}, got {metrics['model_params']['alpha']}"
    
    def test_overridden_hidden_layers_recorded(self, tmp_path):
        """Overriding hidden_layer_sizes should be reflected in metrics JSON."""
        from src.shadow_risk_exposure import run_shadow_risk_exposure_mlp_with_metrics
        
        prices = _create_synthetic_prices(tmp_path)
        output_csv = tmp_path / "shadow_risk.csv"
        output_metrics = tmp_path / "shadow_risk_metrics.json"
        
        custom_layers = (16, 8, 4)
        
        run_shadow_risk_exposure_mlp_with_metrics(
            prices=prices,
            as_of_date="2020-01-01",
            train_end="2019-01-01",
            val_end="2019-07-01",
            output_csv_path=str(output_csv),
            output_metrics_json_path=str(output_metrics),
            spy_ticker="SPY.US",
            seed=42,
            hidden_layer_sizes=custom_layers,  # Override default
        )
        
        with open(output_metrics) as f:
            metrics = json.load(f)
        
        assert metrics["model_params"]["hidden_layer_sizes"] == list(custom_layers), \
            f"Expected {list(custom_layers)}, got {metrics['model_params']['hidden_layer_sizes']}"
