"""
Config-Driven Real-Data Experiment Runner for QUANT-NEURAL.

Provides:
- load_real_data_experiment_config: Load JSON config from file
- validate_real_data_experiment_config: Strict schema validation
- run_real_data_experiment_from_config: Execute experiment from config file

Config Schema (SSOT):
- feature_builder: "real_data_smoke" only
- feature_builder_kwargs: stooq_csv_path, price_ticker, companyfacts_json_paths, cik_to_sector
- labels: stooq_csv_by_ticker, tickers_in_order (exactly 10)
- dates: as_of_date, train_end, val_end
- training: seed, rankgauss, epochs, batch_size
- health: min_months, missing_threshold, ignore_first_n_rows

Run ID: SHA256 of canonical JSON dump for reproducibility tracking.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd

from src.real_data_end_to_end import run_real_data_end_to_end_baseline_mlp


def load_real_data_experiment_config(path: str) -> dict:
    """
    Load experiment config from JSON file.
    
    Parameters
    ----------
    path : str
        Path to JSON config file.
    
    Returns
    -------
    dict
        Parsed config dictionary.
    
    Raises
    ------
    ValueError
        If file cannot be read or JSON is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"Config file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}") from e
    
    return cfg


def validate_real_data_experiment_config(cfg: dict) -> dict:
    """
    Validate experiment config strictly.
    
    Parameters
    ----------
    cfg : dict
        Config dictionary to validate.
    
    Returns
    -------
    dict
        Normalized config (deep copy).
    
    Raises
    ------
    ValueError
        If any validation rule is violated.
    """
    # Deep copy to avoid mutation
    cfg = copy.deepcopy(cfg)
    
    # =========================================================================
    # feature_builder
    # =========================================================================
    if "feature_builder" not in cfg:
        raise ValueError("Missing required key: feature_builder")
    
    if cfg["feature_builder"] != "real_data_smoke":
        raise ValueError(
            f"feature_builder must be 'real_data_smoke', got '{cfg['feature_builder']}'"
        )
    
    # =========================================================================
    # feature_builder_kwargs
    # =========================================================================
    if "feature_builder_kwargs" not in cfg:
        raise ValueError("Missing required key: feature_builder_kwargs")
    
    fb_kwargs = cfg["feature_builder_kwargs"]
    if not isinstance(fb_kwargs, dict):
        raise ValueError("feature_builder_kwargs must be a dict")
    
    # Required keys in feature_builder_kwargs
    required_fb_keys = ["stooq_csv_path", "price_ticker", "companyfacts_json_paths", "cik_to_sector"]
    for key in required_fb_keys:
        if key not in fb_kwargs:
            raise ValueError(f"Missing required key in feature_builder_kwargs: {key}")
    
    if not isinstance(fb_kwargs["stooq_csv_path"], str):
        raise ValueError("feature_builder_kwargs.stooq_csv_path must be str")
    
    if not isinstance(fb_kwargs["price_ticker"], str):
        raise ValueError("feature_builder_kwargs.price_ticker must be str")
    
    if not isinstance(fb_kwargs["companyfacts_json_paths"], list):
        raise ValueError("feature_builder_kwargs.companyfacts_json_paths must be list")
    
    if len(fb_kwargs["companyfacts_json_paths"]) == 0:
        raise ValueError("feature_builder_kwargs.companyfacts_json_paths must be non-empty")
    
    for i, p in enumerate(fb_kwargs["companyfacts_json_paths"]):
        if not isinstance(p, str):
            raise ValueError(f"companyfacts_json_paths[{i}] must be str")
    
    if not isinstance(fb_kwargs["cik_to_sector"], dict):
        raise ValueError("feature_builder_kwargs.cik_to_sector must be dict")
    
    if len(fb_kwargs["cik_to_sector"]) == 0:
        raise ValueError("feature_builder_kwargs.cik_to_sector must be non-empty")
    
    for k, v in fb_kwargs["cik_to_sector"].items():
        if not isinstance(k, str):
            raise ValueError(f"cik_to_sector key must be str, got {type(k)}")
        if not isinstance(v, int):
            raise ValueError(f"cik_to_sector['{k}'] must be int, got {type(v)}")
    
    # =========================================================================
    # labels
    # =========================================================================
    if "labels" not in cfg:
        raise ValueError("Missing required key: labels")
    
    labels = cfg["labels"]
    if not isinstance(labels, dict):
        raise ValueError("labels must be a dict")
    
    # Accept either tickers_in_order or label_tickers_in_order (alias)
    has_original = "tickers_in_order" in labels
    has_alias = "label_tickers_in_order" in labels
    
    if not has_original and not has_alias:
        raise ValueError("Missing required key: labels.tickers_in_order or labels.label_tickers_in_order")
    
    # If both exist, they must be identical
    if has_original and has_alias:
        if labels["tickers_in_order"] != labels["label_tickers_in_order"]:
            raise ValueError(
                "labels.tickers_in_order and labels.label_tickers_in_order must be identical if both provided"
            )
    
    # Use whichever is present (prefer alias if both exist for validation)
    tickers = labels.get("label_tickers_in_order", labels.get("tickers_in_order"))
    
    if not isinstance(tickers, list):
        raise ValueError("labels.tickers_in_order (or label_tickers_in_order) must be list")
    
    if len(tickers) != 10:
        raise ValueError(f"labels.tickers_in_order must have exactly 10 items, got {len(tickers)}")
    
    for i, t in enumerate(tickers):
        if not isinstance(t, str):
            raise ValueError(f"labels.tickers_in_order[{i}] must be str")
    
    if len(set(tickers)) != 10:
        raise ValueError("labels.tickers_in_order must have 10 unique tickers")
    
    if "stooq_csv_by_ticker" not in labels:
        raise ValueError("Missing required key: labels.stooq_csv_by_ticker")
    
    stooq_by_ticker = labels["stooq_csv_by_ticker"]
    if not isinstance(stooq_by_ticker, dict):
        raise ValueError("labels.stooq_csv_by_ticker must be dict")
    
    for ticker in tickers:
        if ticker not in stooq_by_ticker:
            raise ValueError(f"labels.stooq_csv_by_ticker missing path for ticker: {ticker}")
    
    for k, v in stooq_by_ticker.items():
        if not isinstance(k, str):
            raise ValueError(f"stooq_csv_by_ticker key must be str")
        if not isinstance(v, str):
            raise ValueError(f"stooq_csv_by_ticker['{k}'] must be str")
    
    # =========================================================================
    # dates
    # =========================================================================
    if "dates" not in cfg:
        raise ValueError("Missing required key: dates")
    
    dates = cfg["dates"]
    if not isinstance(dates, dict):
        raise ValueError("dates must be a dict")
    
    required_date_keys = ["as_of_date", "train_end", "val_end"]
    for key in required_date_keys:
        if key not in dates:
            raise ValueError(f"Missing required key: dates.{key}")
        if not isinstance(dates[key], str):
            raise ValueError(f"dates.{key} must be str")
    
    # Parse and validate date format
    try:
        as_of_dt = pd.to_datetime(dates["as_of_date"], format="%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"dates.as_of_date must be YYYY-MM-DD format: {e}") from e
    
    try:
        train_end_dt = pd.to_datetime(dates["train_end"], format="%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"dates.train_end must be YYYY-MM-DD format: {e}") from e
    
    try:
        val_end_dt = pd.to_datetime(dates["val_end"], format="%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"dates.val_end must be YYYY-MM-DD format: {e}") from e
    
    # Date ordering: train_end < val_end <= as_of_date
    if train_end_dt >= val_end_dt:
        raise ValueError(
            f"dates.train_end ({dates['train_end']}) must be < dates.val_end ({dates['val_end']})"
        )
    
    if val_end_dt > as_of_dt:
        raise ValueError(
            f"dates.val_end ({dates['val_end']}) must be <= dates.as_of_date ({dates['as_of_date']})"
        )
    
    # =========================================================================
    # training
    # =========================================================================
    if "training" not in cfg:
        raise ValueError("Missing required key: training")
    
    training = cfg["training"]
    if not isinstance(training, dict):
        raise ValueError("training must be a dict")
    
    if "seed" not in training:
        raise ValueError("Missing required key: training.seed")
    if not isinstance(training["seed"], int):
        raise ValueError("training.seed must be int")
    
    if "rankgauss" not in training:
        raise ValueError("Missing required key: training.rankgauss")
    if not isinstance(training["rankgauss"], bool):
        raise ValueError("training.rankgauss must be bool")
    
    if "epochs" not in training:
        raise ValueError("Missing required key: training.epochs")
    if not isinstance(training["epochs"], int) or training["epochs"] <= 0:
        raise ValueError("training.epochs must be positive int")
    
    if "batch_size" not in training:
        raise ValueError("Missing required key: training.batch_size")
    if not isinstance(training["batch_size"], int) or training["batch_size"] <= 0:
        raise ValueError("training.batch_size must be positive int")
    
    # =========================================================================
    # health
    # =========================================================================
    if "health" not in cfg:
        raise ValueError("Missing required key: health")
    
    health = cfg["health"]
    if not isinstance(health, dict):
        raise ValueError("health must be a dict")
    
    if "min_months" not in health:
        raise ValueError("Missing required key: health.min_months")
    if not isinstance(health["min_months"], int) or health["min_months"] <= 0:
        raise ValueError("health.min_months must be positive int")
    
    if "missing_threshold" not in health:
        raise ValueError("Missing required key: health.missing_threshold")
    if not isinstance(health["missing_threshold"], (int, float)):
        raise ValueError("health.missing_threshold must be float")
    if not (0.0 <= health["missing_threshold"] <= 1.0):
        raise ValueError("health.missing_threshold must be in [0, 1]")
    
    if "ignore_first_n_rows" not in health:
        raise ValueError("Missing required key: health.ignore_first_n_rows")
    if not isinstance(health["ignore_first_n_rows"], int) or health["ignore_first_n_rows"] < 0:
        raise ValueError("health.ignore_first_n_rows must be non-negative int")
    
    # Optional sector representativeness params
    if "min_sector_firms" in health:
        if not isinstance(health["min_sector_firms"], int) or health["min_sector_firms"] < 1:
            raise ValueError("health.min_sector_firms must be int >= 1")
    
    if "max_low_count_month_ratio" in health:
        if not isinstance(health["max_low_count_month_ratio"], (int, float)):
            raise ValueError("health.max_low_count_month_ratio must be float")
        if not (0.0 <= health["max_low_count_month_ratio"] <= 1.0):
            raise ValueError("health.max_low_count_month_ratio must be in [0.0, 1.0]")
    
    # Strict schema: reject unknown keys in health
    allowed_health_keys = {
        "min_months", "missing_threshold", "ignore_first_n_rows",
        "min_sector_firms", "max_low_count_month_ratio"
    }
    for key in health:
        if key not in allowed_health_keys:
            raise ValueError(f"Unknown key in health: {key}")
    
    return cfg


def _compute_run_id(cfg: dict) -> str:
    """Compute deterministic run_id from config."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_real_data_experiment_from_config(path: str) -> dict:
    """
    Run real-data experiment from JSON config file.
    
    Parameters
    ----------
    path : str
        Path to JSON config file.
    
    Returns
    -------
    dict
        Results with keys:
        - "config": normalized config dict
        - "result": result from run_real_data_end_to_end_baseline_mlp
        - "run_id": SHA256 hex string of canonical config
    
    Raises
    ------
    ValueError
        If config is invalid or experiment fails.
    """
    # Load and validate config
    cfg = load_real_data_experiment_config(path)
    cfg = validate_real_data_experiment_config(cfg)
    
    # Compute deterministic run_id
    run_id = _compute_run_id(cfg)
    
    # Resolve feature_builder
    if cfg["feature_builder"] == "real_data_smoke":
        from src.real_data_smoke import build_real_data_feature_frame
        feature_builder = build_real_data_feature_frame
    else:
        raise ValueError(f"Unknown feature_builder: {cfg['feature_builder']}")
    
    # Run experiment
    # Prefer label_tickers_in_order alias if present, else fall back to tickers_in_order
    label_tickers = cfg["labels"].get("label_tickers_in_order", cfg["labels"].get("tickers_in_order"))
    
    result = run_real_data_end_to_end_baseline_mlp(
        feature_builder=feature_builder,
        feature_builder_kwargs=cfg["feature_builder_kwargs"],
        stooq_csv_by_ticker_for_labels=cfg["labels"]["stooq_csv_by_ticker"],
        label_tickers_in_order=label_tickers,
        as_of_date=cfg["dates"]["as_of_date"],
        train_end=cfg["dates"]["train_end"],
        val_end=cfg["dates"]["val_end"],
        seed=cfg["training"]["seed"],
        rankgauss=cfg["training"]["rankgauss"],
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        health_min_months=cfg["health"]["min_months"],
        health_missing_threshold=cfg["health"]["missing_threshold"],
        health_ignore_first_n_rows=cfg["health"]["ignore_first_n_rows"],
        # Optional sector representativeness params (use defaults if absent)
        health_min_sector_firms=cfg["health"].get("min_sector_firms", 3),
        health_max_low_count_month_ratio=cfg["health"].get("max_low_count_month_ratio", 1.0),
    )
    
    return {
        "config": cfg,
        "result": result,
        "run_id": run_id,
    }
