"""
Main executor for QUANT-NEURAL pipeline.

Provides:
- set_global_seed: Reproducible seeding for numpy/tensorflow.
- load_yaml_config: Load YAML config file.
- validate_config_or_raise: Validate config, fail-fast for Phase 5 features.
- run_synthetic_smoke: End-to-end smoke test with synthetic data.
- run_from_config: Config-driven synthetic pipeline.
- build_run_id: Deterministic run ID from config.
- save_artifacts: Save run artifacts (manifest, features, model).

Entry point for running the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Any

import yaml

import numpy as np
import pandas as pd
import tensorflow as tf

# Project modules
from src.factors import build_relative_earnings_momentum
from src.models import SectorPredictorMLP, MLPParams
from src.selection import ModelSelector, LassoParams
from src.regime import RegimeDetector, RegimeParams


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Prefer Keras util if available
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    
    # Optional determinism (do not fail if unavailable)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def run_synthetic_smoke(
    seed: int = 42,
    return_objects: bool = False
) -> dict | Tuple[dict, dict]:
    """
    Run end-to-end synthetic smoke test.
    
    Exercises all core modules:
    - build_relative_earnings_momentum (factors)
    - SectorPredictorMLP (models)
    - ModelSelector (selection)
    - RegimeDetector (regime)
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    return_objects : bool
        If True, also return internal objects for artifact saving.
    
    Returns
    -------
    dict or Tuple[dict, dict]
        Results dict. If return_objects=True, returns (out, objects_dict).
    """
    set_global_seed(seed)
    
    # =========================================================================
    # 1) Build synthetic monthly data for relative earnings momentum
    # =========================================================================
    T = 24  # months
    sectors = [f"S{i}" for i in range(10)]
    dates = pd.date_range("2020-01-31", periods=T, freq="ME")
    
    # Create positive-valued operating profits (for logdiff)
    # Start at 100 and grow mildly with some noise
    np.random.seed(seed)
    
    def make_sector_df(base: float = 100.0) -> pd.DataFrame:
        """Create sector DataFrame with mild growth."""
        data = np.zeros((T, 10))
        for i in range(10):
            # Start at base, grow 1% per month on average with noise
            values = [base]
            for t in range(1, T):
                growth = 1.01 + np.random.randn() * 0.02  # ~1% growth, 2% vol
                values.append(values[-1] * max(growth, 0.9))  # floor to avoid negative
            data[:, i] = values
        return pd.DataFrame(data, index=dates, columns=sectors)
    
    def make_market_series(base: float = 100.0) -> pd.Series:
        """Create market Series with mild growth."""
        values = [base]
        for t in range(1, T):
            growth = 1.01 + np.random.randn() * 0.015
            values.append(values[-1] * max(growth, 0.9))
        return pd.Series(values, index=dates)
    
    sector_op_fy1 = make_sector_df(100.0)
    sector_op_fy2 = make_sector_df(105.0)
    market_op_fy1 = make_market_series(100.0)
    market_op_fy2 = make_market_series(105.0)
    
    # =========================================================================
    # 2) Compute relative earnings momentum features
    # =========================================================================
    X_mlp = build_relative_earnings_momentum(
        sector_op_fy1, sector_op_fy2,
        market_op_fy1, market_op_fy2,
        method="logdiff"
    )
    
    # Validate shape
    if X_mlp.shape != (T, 20):
        raise ValueError(f"Expected X_mlp shape ({T}, 20), got {X_mlp.shape}")
    
    # =========================================================================
    # 3) Create synthetic Y_mlp (target outputs)
    # =========================================================================
    Y_mlp = np.random.randn(T, 10).astype(np.float32)
    
    # Convert X_mlp to numpy, fill NaN with 0 for MLP
    X_mlp_np = X_mlp.fillna(0).values.astype(np.float32)
    
    # =========================================================================
    # 4) Time-ordered split (NO SHUFFLE)
    # =========================================================================
    n_train = int(0.6 * T)  # 14
    n_val = int(0.2 * T)    # 4
    # n_test = T - n_train - n_val  # 6
    
    X_train = X_mlp_np[:n_train]
    Y_train = Y_mlp[:n_train]
    X_val = X_mlp_np[n_train:n_train + n_val]
    Y_val = Y_mlp[n_train:n_train + n_val]
    X_test = X_mlp_np[n_train + n_val:]
    Y_test = Y_mlp[n_train + n_val:]
    
    # =========================================================================
    # 5) Train MLP (1 epoch for speed)
    # =========================================================================
    mlp_params = MLPParams(epochs=1, batch_size=16)
    mlp = SectorPredictorMLP(mlp_params)
    mlp.fit(X_train, Y_train, X_val, Y_val)
    yhat = mlp.predict(X_test)
    
    # =========================================================================
    # 6) Selection smoke test
    # =========================================================================
    n_sel = 200
    p_sel = 6
    np.random.seed(seed + 1)
    X_sel = np.random.randn(n_sel, p_sel).astype(np.float64)
    # Strong signal: y = 3*f0 - 2*f2 + noise
    y_sel = 3.0 * X_sel[:, 0] - 2.0 * X_sel[:, 2] + 0.01 * np.random.randn(n_sel)
    feature_names_sel = [f"f{i}" for i in range(p_sel)]
    
    selector = ModelSelector(LassoParams())
    selected_features = selector.select_features_lasso(X_sel, y_sel, feature_names_sel)
    
    # =========================================================================
    # 7) Regime smoke test
    # =========================================================================
    n_reg = 300
    np.random.seed(seed + 2)
    X_reg = np.random.randn(n_reg, 3).astype(np.float64)
    # y_reg = (2*X0 - X1 > 0).astype(int)
    logits = 2.0 * X_reg[:, 0] - X_reg[:, 1]
    y_reg = (logits > 0).astype(int)
    
    # Time-ordered split: first 200 train, last 100 test
    X_train_reg = X_reg[:200]
    y_train_reg = y_reg[:200]
    X_test_reg = X_reg[200:]
    y_test_reg = y_reg[200:]
    
    detector = RegimeDetector(RegimeParams())
    detector.fit(X_train_reg, y_train_reg)
    p_up = detector.predict_proba_up(X_test_reg)
    brier = detector.brier(X_test_reg, y_test_reg)
    action_last = detector.action(float(p_up[-1]))
    
    # =========================================================================
    # 8) Return results
    # =========================================================================
    out = {
        "X_mlp_shape": X_mlp.shape,
        "yhat_shape": yhat.shape,
        "selected_features": selected_features,
        "feature_names_sel": feature_names_sel,
        "regime_brier": brier,
        "regime_action_last": action_last,
    }
    
    if return_objects:
        objects = {
            "mlp_model": mlp.model,
            "selected_features": selected_features,
        }
        return out, objects
    
    return out


def load_yaml_config(path: str) -> dict:
    """
    Load YAML config file.
    
    Parameters
    ----------
    path : str
        Path to YAML config file.
    
    Returns
    -------
    dict
        Parsed configuration.
    
    Raises
    ------
    ValueError
        If config does not parse to a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must parse to a dict.")
    return cfg


def validate_config_or_raise(cfg: dict) -> dict:
    """
    Validate config, fail-fast for missing keys and unimplemented Phase 5 features.
    
    Parameters
    ----------
    cfg : dict
        Parsed configuration.
    
    Returns
    -------
    dict
        The config unchanged if valid.
    
    Raises
    ------
    ValueError
        If required keys are missing or have wrong types.
    RuntimeError
        If unimplemented Phase 5 features are enabled.
    """
    # Required: project section
    if "project" not in cfg or not isinstance(cfg["project"], dict):
        raise ValueError("Config must contain 'project' section as a dict.")
    
    project = cfg["project"]
    
    # Required: seed (int)
    if "seed" not in project:
        raise ValueError("Config missing required key: project.seed")
    if not isinstance(project["seed"], int):
        raise ValueError("project.seed must be an int.")
    
    # Required: as_of_date (str)
    if "as_of_date" not in project:
        raise ValueError("Config missing required key: project.as_of_date")
    if not isinstance(project["as_of_date"], str):
        raise ValueError("project.as_of_date must be a str.")
    
    # =========================================================================
    # FAIL-FAST: Unimplemented Phase 5 features (Phase 8.x)
    # =========================================================================
    
    # Conformal prediction
    if cfg.get("conformal", {}).get("enabled", False) is True:
        raise RuntimeError(
            "Config enables conformal prediction (Phase 8.x). "
            "Set conformal.enabled=false for 6.x."
        )
    
    # Regime calibration
    if cfg.get("regime", {}).get("calibration", {}).get("enabled", False) is True:
        raise RuntimeError(
            "Config enables regime calibration (Phase 8.x). "
            "Set regime.calibration.enabled=false for 6.x."
        )
    
    # Model type (only MLP supported)
    model_type = cfg.get("models", {}).get("type", "mlp")
    if model_type != "mlp":
        raise RuntimeError(
            f"Only models.type='mlp' is supported in 6.x. "
            f"Got '{model_type}'. Other types are Phase 8.x."
        )
    
    return cfg


def build_run_id(cfg: dict) -> str:
    """
    Build deterministic run ID from config.
    
    Parameters
    ----------
    cfg : dict
        Validated config.
    
    Returns
    -------
    str
        Run ID in format: seed{seed}_asof{as_of_date}
    """
    seed = cfg["project"]["seed"]
    as_of_date = cfg["project"]["as_of_date"]
    return f"seed{seed}_asof{as_of_date}"


def save_artifacts(
    *,
    artifacts_dir: str,
    run_id: str,
    out: dict,
    selected_features: list,
    mlp_model: Any,
) -> str:
    """
    Save run artifacts to disk.
    
    Parameters
    ----------
    artifacts_dir : str
        Base directory for artifacts.
    run_id : str
        Deterministic run identifier.
    out : dict
        Output dict from synthetic smoke run.
    selected_features : list
        Selected feature names from Lasso.
    mlp_model : keras.Model
        Trained MLP model.
    
    Returns
    -------
    str
        Path to the run directory.
    """
    # Create run directory
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save manifest.json
    manifest = {
        "run_id": run_id,
        "config_seed": out.get("config_seed"),
        "config_as_of_date": out.get("config_as_of_date"),
        "X_mlp_shape": list(out.get("X_mlp_shape", [])),
        "yhat_shape": list(out.get("yhat_shape", [])),
        "regime_brier": out.get("regime_brier"),
        "regime_action_last": out.get("regime_action_last"),
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    
    # Save selected_features.json
    features_path = run_dir / "selected_features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)
    
    # Save model
    model_path = run_dir / "mlp.keras"
    mlp_model.save(str(model_path), include_optimizer=False)
    
    return str(run_dir)


def run_from_config(config_path: str, artifacts_dir: Optional[str] = None) -> dict:
    """
    Run synthetic pipeline using config file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    artifacts_dir : Optional[str]
        If provided, save artifacts to this directory.
    
    Returns
    -------
    dict
        Results from synthetic smoke run + config echoes.
    """
    cfg = load_yaml_config(config_path)
    cfg = validate_config_or_raise(cfg)
    
    seed = cfg["project"]["seed"]
    as_of_date = cfg["project"]["as_of_date"]
    
    # Run with objects if we need to save artifacts
    if artifacts_dir:
        out, objects = run_synthetic_smoke(seed=seed, return_objects=True)
    else:
        out = run_synthetic_smoke(seed=seed)
    
    # Add config echoes for traceability
    out["config_seed"] = seed
    out["config_as_of_date"] = as_of_date
    
    # Save artifacts if requested
    if artifacts_dir:
        run_id = build_run_id(cfg)
        run_dir = save_artifacts(
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            out=out,
            selected_features=objects["selected_features"],
            mlp_model=objects["mlp_model"],
        )
        out["artifacts_run_dir"] = run_dir
    
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QUANT-NEURAL pipeline executor"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/hyperparameters.yaml)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory to save run artifacts (manifest, features, model)"
    )
    args = parser.parse_args()
    
    if args.config:
        out = run_from_config(args.config, artifacts_dir=args.artifacts_dir)
    else:
        out = run_synthetic_smoke(seed=42)
    
    print("=" * 60)
    print("QUANT-NEURAL Synthetic Smoke Test Results")
    print("=" * 60)
    for k, v in out.items():
        print(f"{k}: {v}")
