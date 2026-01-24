"""
Tests for src/shadow_risk_decision_gate.py

Covers:
- Determinism (byte-identical JSON)
- Promotion when all thresholds pass
- Retention when ECE fails
- Fail-safe on missing inputs
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shadow_risk_decision_gate import evaluate_shadow_risk_promotion_decision


def _create_synthetic_risk_metrics_json(
    tmp_path: Path,
    test_ece: float,
    val_end: str = "2020-02-15",
    as_of_date: str = "2020-05-29",
) -> Path:
    """Create synthetic risk metrics JSON for testing."""
    risk_metrics = {
        "schema_version": "9.5.2",
        "config": {
            "val_end": val_end,
            "as_of_date": as_of_date,
        },
        "train": {"n_obs": 500, "ece": 0.04},
        "val": {"n_obs": 200, "ece": 0.05},
        "test": {"n_obs": 100, "ece": test_ece},
    }
    path = tmp_path / "risk_metrics.json"
    with open(path, "w") as f:
        json.dump(risk_metrics, f, indent=2, sort_keys=True)
    return path


def _create_synthetic_overlay_csv(
    tmp_path: Path,
    spy_max_dd: float = -0.30,
    n_days: int = 100,
) -> Path:
    """
    Create synthetic overlay CSV with controlled SPY max drawdown.
    
    We construct a specific return sequence that produces the exact target max drawdown.
    """
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    
    # Build a return sequence that produces exact target drawdown:
    # Start with small positive returns, then a big drop, then recovery
    spy_rets = np.zeros(n_days)
    
    # First half: small gains to build equity to 1.4 (40% gain)
    for i in range(n_days // 2):
        spy_rets[i] = 0.007  # ~0.7% daily, compounding to ~40%
    
    # Then a crash that produces target drawdown from peak
    # If peak is ~1.4 and we want -30% drawdown, we need to drop to ~0.98 (1.4 * 0.7 = 0.98)
    # Then we need return of (0.98/1.4) - 1 = -0.30
    crash_point = n_days // 2
    spy_rets[crash_point] = spy_max_dd  # Single day crash matching target
    
    # Rest: small positive returns for recovery
    for i in range(crash_point + 1, n_days):
        spy_rets[i] = 0.002
    
    # Build overlay DataFrame
    overlay_df = pd.DataFrame({
        "exposure_suggested": np.full(n_days, 0.8),
        "spy_ret_1d": spy_rets,
        "overlay_ret_1d": spy_rets * 0.8,
        "overlay_equity": (1 + spy_rets * 0.8).cumprod(),
    }, index=dates)
    overlay_df.index.name = "date"
    
    path = tmp_path / "overlay.csv"
    overlay_df.to_csv(path, float_format="%.10f")
    return path



def _create_synthetic_overlay_metrics_json(
    tmp_path: Path,
    test_cagr_over_vol: float = 1.2,
    test_max_drawdown: float = -0.18,
) -> Path:
    """Create synthetic overlay metrics JSON for testing."""
    overlay_metrics = {
        "schema_version": "9.5.3",
        "config": {},
        "train": {"n_obs": 500, "cagr_over_vol": 1.0, "max_drawdown": -0.15},
        "val": {"n_obs": 200, "cagr_over_vol": 1.1, "max_drawdown": -0.17},
        "test": {
            "n_obs": 100,
            "cagr_over_vol": test_cagr_over_vol,
            "max_drawdown": test_max_drawdown,
        },
    }
    path = tmp_path / "overlay_metrics.json"
    with open(path, "w") as f:
        json.dump(overlay_metrics, f, indent=2, sort_keys=True)
    return path


class TestDecisionJsonDeterminism:
    """Test decision JSON determinism."""
    
    def test_decision_json_determinism_byte_identical(self, tmp_path):
        """Same inputs produce byte-identical JSON."""
        risk_json = _create_synthetic_risk_metrics_json(tmp_path, test_ece=0.03)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        overlay_json = _create_synthetic_overlay_metrics_json(tmp_path)
        
        decision1 = tmp_path / "decision1.json"
        decision2 = tmp_path / "decision2.json"
        
        evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=str(risk_json),
            overlay_csv_path=str(overlay_csv),
            overlay_metrics_json_path=str(overlay_json),
            output_decision_json_path=str(decision1),
        )
        
        evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=str(risk_json),
            overlay_csv_path=str(overlay_csv),
            overlay_metrics_json_path=str(overlay_json),
            output_decision_json_path=str(decision2),
        )
        
        assert decision1.read_bytes() == decision2.read_bytes()


class TestPromoteWhenAllPass:
    """Test promotion when all thresholds pass."""
    
    def test_promote_when_all_thresholds_pass(self, tmp_path):
        """When all criteria pass, decision is PROMOTE_EXECUTION_CONTROL."""
        # Setup: test_ece=0.03 (<0.05), cagr_over_vol=1.5 (>=1.0), max_dd=-0.18 vs spy=-0.30 (>20% reduction)
        risk_json = _create_synthetic_risk_metrics_json(tmp_path, test_ece=0.03)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path, spy_max_dd=-0.30)
        overlay_json = _create_synthetic_overlay_metrics_json(
            tmp_path, test_cagr_over_vol=1.5, test_max_drawdown=-0.18
        )
        
        decision_path = tmp_path / "decision.json"
        
        result = evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=str(risk_json),
            overlay_csv_path=str(overlay_csv),
            overlay_metrics_json_path=str(overlay_json),
            output_decision_json_path=str(decision_path),
        )
        
        assert result["decision"] == "PROMOTE_EXECUTION_CONTROL"
        assert result["checks"]["pass_cagr_over_vol"] is True
        assert result["checks"]["pass_dd_reduction"] is True
        assert result["checks"]["pass_ece"] is True


class TestRetainWhenECEFails:
    """Test retention when ECE fails threshold."""
    
    def test_retain_when_ece_fails(self, tmp_path):
        """When ECE exceeds threshold, decision is RETAIN_SHADOW_ONLY."""
        # Setup: test_ece=0.10 (>0.05), other metrics pass
        risk_json = _create_synthetic_risk_metrics_json(tmp_path, test_ece=0.10)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        overlay_json = _create_synthetic_overlay_metrics_json(
            tmp_path, test_cagr_over_vol=1.5, test_max_drawdown=-0.18
        )
        
        decision_path = tmp_path / "decision.json"
        
        result = evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=str(risk_json),
            overlay_csv_path=str(overlay_csv),
            overlay_metrics_json_path=str(overlay_json),
            output_decision_json_path=str(decision_path),
        )
        
        assert result["decision"] == "RETAIN_SHADOW_ONLY"
        assert result["checks"]["pass_ece"] is False
        assert any("ECE" in r for r in result["reasons"])


class TestFailsafeMissingInputs:
    """Test fail-safe behavior on missing inputs."""
    
    def test_failsafe_missing_inputs_writes_json_and_warns(self, tmp_path, caplog):
        """Missing input files trigger fail-safe with warning and valid JSON."""
        decision_path = tmp_path / "decision.json"
        
        with caplog.at_level(logging.WARNING):
            result = evaluate_shadow_risk_promotion_decision(
                risk_metrics_json_path=str(tmp_path / "nonexistent_risk.json"),
                overlay_csv_path=str(tmp_path / "nonexistent_overlay.csv"),
                overlay_metrics_json_path=str(tmp_path / "nonexistent_metrics.json"),
                output_decision_json_path=str(decision_path),
            )
        
        # Decision JSON should be written
        assert decision_path.exists()
        
        # Decision should be RETAIN
        assert result["decision"] == "RETAIN_SHADOW_ONLY"
        
        # Warning should contain stable prefix
        assert any("SHADOW_RISK_GATE:" in msg for msg in caplog.messages)
        
        # Reasons should exist
        assert len(result["reasons"]) > 0


class TestRetainWhenCagrOverVolFails:
    """Test retention when CAGR/Vol fails threshold."""
    
    def test_retain_when_cagr_over_vol_below_threshold(self, tmp_path):
        """When CAGR/Vol below threshold, decision is RETAIN."""
        risk_json = _create_synthetic_risk_metrics_json(tmp_path, test_ece=0.03)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        overlay_json = _create_synthetic_overlay_metrics_json(
            tmp_path, test_cagr_over_vol=0.5, test_max_drawdown=-0.18
        )
        
        decision_path = tmp_path / "decision.json"
        
        result = evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=str(risk_json),
            overlay_csv_path=str(overlay_csv),
            overlay_metrics_json_path=str(overlay_json),
            output_decision_json_path=str(decision_path),
        )
        
        assert result["decision"] == "RETAIN_SHADOW_ONLY"
        assert result["checks"]["pass_cagr_over_vol"] is False


class TestDecisionJsonSchema:
    """Test decision JSON has required schema."""
    
    def test_decision_json_has_required_keys(self, tmp_path):
        """Decision JSON must have all required keys."""
        risk_json = _create_synthetic_risk_metrics_json(tmp_path, test_ece=0.03)
        overlay_csv = _create_synthetic_overlay_csv(tmp_path)
        overlay_json = _create_synthetic_overlay_metrics_json(tmp_path)
        
        decision_path = tmp_path / "decision.json"
        
        evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=str(risk_json),
            overlay_csv_path=str(overlay_csv),
            overlay_metrics_json_path=str(overlay_json),
            output_decision_json_path=str(decision_path),
        )
        
        with open(decision_path, "r") as f:
            data = json.load(f)
        
        # Check required keys
        assert data["schema_version"] == "9.6.0"
        assert "inputs" in data
        assert "computed" in data
        assert "thresholds" in data
        assert "checks" in data
        assert "decision" in data
        assert "reasons" in data
        assert "warnings" in data
        
        # Check computed sub-keys
        assert "overlay" in data["computed"]
        assert "spy_buy_hold" in data["computed"]
        assert "dd_reduction" in data["computed"]
        assert "test_ece" in data["computed"]


class TestSpyMetricsTestWindowOnly:
    """
    Regression test to verify SPY metrics are computed on TEST window only.
    
    This test would FAIL if SPY metrics were computed on the full window.
    """
    
    def test_spy_metrics_use_test_window_only(self, tmp_path):
        """
        SPY max_drawdown must come from TEST window only, not full window.
        
        Setup:
        - VAL segment (<= val_end): large drawdown (-0.50)
        - TEST segment (> val_end): small drawdown (-0.10)
        
        If SPY metrics use full window, max_dd would be ~ -0.50.
        If SPY metrics use TEST window only, max_dd should be ~ -0.10.
        """
        # Define split boundaries
        val_end = "2020-03-15"
        as_of_date = "2020-05-29"
        val_end_dt = pd.to_datetime(val_end)
        as_of_dt = pd.to_datetime(as_of_date)
        
        # Create date range spanning both VAL and TEST
        dates = pd.date_range("2020-01-01", "2020-05-31", freq="B")
        n_days = len(dates)
        spy_rets = np.zeros(n_days)
        
        # Mark each date as VAL or TEST
        for i, dt in enumerate(dates):
            if dt <= val_end_dt:
                # VAL period: include a big crash
                if i == 30:  # One day crash in VAL
                    spy_rets[i] = -0.50  # 50% drop in VAL
                else:
                    spy_rets[i] = 0.005  # Small gains otherwise
            else:
                # TEST period: mild volatility with small drawdown
                if i == len(dates) - 10:  # Small dip in TEST
                    spy_rets[i] = -0.10  # 10% drop in TEST
                else:
                    spy_rets[i] = 0.003
        
        # Build overlay CSV
        overlay_df = pd.DataFrame({
            "exposure_suggested": np.full(n_days, 0.8),
            "spy_ret_1d": spy_rets,
            "overlay_ret_1d": spy_rets * 0.8,
            "overlay_equity": (1 + spy_rets * 0.8).cumprod(),
        }, index=dates)
        overlay_df.index.name = "date"
        
        overlay_csv_path = tmp_path / "overlay.csv"
        overlay_df.to_csv(overlay_csv_path, float_format="%.10f")
        
        # Create risk metrics JSON with config dates
        risk_metrics = {
            "schema_version": "9.5.2",
            "config": {
                "val_end": val_end,
                "as_of_date": as_of_date,
            },
            "train": {"n_obs": 500, "ece": 0.04},
            "val": {"n_obs": 200, "ece": 0.05},
            "test": {"n_obs": 100, "ece": 0.03},
        }
        risk_json_path = tmp_path / "risk_metrics.json"
        with open(risk_json_path, "w") as f:
            json.dump(risk_metrics, f, indent=2, sort_keys=True)
        
        # Create overlay metrics JSON
        overlay_metrics = {
            "schema_version": "9.5.3",
            "config": {},
            "train": {"n_obs": 500, "cagr_over_vol": 1.0, "max_drawdown": -0.15},
            "val": {"n_obs": 200, "cagr_over_vol": 1.1, "max_drawdown": -0.17},
            "test": {"n_obs": 50, "cagr_over_vol": 1.5, "max_drawdown": -0.08},
        }
        overlay_json_path = tmp_path / "overlay_metrics.json"
        with open(overlay_json_path, "w") as f:
            json.dump(overlay_metrics, f, indent=2, sort_keys=True)
        
        # Run decision gate
        decision_path = tmp_path / "decision.json"
        result = evaluate_shadow_risk_promotion_decision(
            risk_metrics_json_path=str(risk_json_path),
            overlay_csv_path=str(overlay_csv_path),
            overlay_metrics_json_path=str(overlay_json_path),
            output_decision_json_path=str(decision_path),
        )
        
        # Key assertion: SPY max_drawdown must be from TEST only (~-0.10), NOT full window (~-0.50)
        spy_max_dd = result["computed"]["spy_buy_hold"]["max_drawdown"]
        
        # If using full window, spy_max_dd would be around -0.50
        # If using TEST window only, spy_max_dd should be around -0.10
        assert spy_max_dd is not None, "SPY max_drawdown should not be None"
        assert spy_max_dd > -0.25, f"SPY max_dd {spy_max_dd} suggests full window was used, not TEST only"
        assert spy_max_dd < 0, f"SPY max_dd {spy_max_dd} should be negative"
