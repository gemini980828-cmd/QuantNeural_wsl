"""
Train alpha model from long-format alpha_dataset using walk-forward protocol.

Exports wide scores.csv compatible with existing backtest contracts.
Strict PIT/no-look-ahead: scoring for date t uses only rows with date < t.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Task 10.2.2.1: SEC fundamental columns (SSOT - must match build_alpha_dataset.py)
# Updated for V2.3 canonical layer column names
SEC_FUNDAMENTAL_COLS = [
    # V2.3 canonical columns
    "total_assets", "total_liabilities", "stockholders_equity",
    "revenues", "net_income", "operating_cash_flow", "shares_outstanding",
    # V1 legacy columns (for backward compatibility)
    "assets", "liabilities", "equity", "cash", "shares_out",
    "leverage", "cash_to_assets", "book_to_assets", "mktcap",
]

# Derived sets for eligibility gating
META_COLS = {"date", "ticker"}
FUND_COLS = set(SEC_FUNDAMENTAL_COLS)
FUND_MISS_COLS = {f"{c}_is_missing" for c in SEC_FUNDAMENTAL_COLS}

# Task 10.2.10: Feature modes for ablation study
FEATURE_MODES = ["fund_full", "fund_zeroed", "fund_shuffled", "tech_only"]


def _stable_hash(s: str) -> int:
    """Compute a stable hash that is consistent across Python runs."""
    import zlib
    return zlib.adler32(s.encode("utf-8"))


def _apply_feature_mode(
    df: pd.DataFrame,
    feature_mode: str,
    fund_value_cols: list[str],
    fund_miss_cols: list[str],
    seed: int,
    warnings_list: list[str],
) -> pd.DataFrame:
    """
    Apply feature mode transformation to DataFrame.
    
    Task 10.2.10: Transforms FUND columns based on feature_mode.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all feature columns.
    feature_mode : str
        One of: fund_full, fund_zeroed, fund_shuffled, tech_only
    fund_value_cols : list[str]
        List of fundamental value columns present in df.
    fund_miss_cols : list[str]
        List of *_is_missing indicator columns present in df.
    seed : int
        Global random seed.
    warnings_list : list[str]
        List to append warnings to.
    
    Returns
    -------
    pd.DataFrame
        Transformed DataFrame (copy).
    """
    result = df.copy()
    
    if feature_mode == "fund_full":
        # Default behavior - no transformation needed
        return result
    
    elif feature_mode == "tech_only":
        # Drop all FUND columns (they won't be used)
        # Note: actual column exclusion happens in feature_cols selection
        return result
    
    elif feature_mode == "fund_zeroed":
        # Set all fundamental value columns to 0.0
        for col in fund_value_cols:
            if col in result.columns:
                result[col] = 0.0
        # Set all *_is_missing indicators to 1.0 (everyone looks missing)
        for col in fund_miss_cols:
            if col in result.columns:
                result[col] = 1.0
        return result
    
    elif feature_mode == "fund_shuffled":
        # Shuffle FUND columns within each date (deterministically)
        all_fund_cols = fund_value_cols + fund_miss_cols
        present_cols = [c for c in all_fund_cols if c in result.columns]
        
        if not present_cols:
            warnings_list.append(f"ALPHA_FEATURE_MODE_FALLBACK:{feature_mode}:no_fund_cols_to_shuffle")
            return result
        
        # Group by date and shuffle within each date
        dates = result["date"].unique()
        
        for date in dates:
            date_mask = result["date"] == date
            date_indices = result[date_mask].index.tolist()
            
            if len(date_indices) <= 1:
                continue
            
            # Create deterministic seed for this date
            date_str = str(date)
            stable_seed = (seed * 1000003) ^ _stable_hash(date_str)
            rng = np.random.RandomState(stable_seed % (2**31))
            
            # Get the fund columns for this date
            fund_values = result.loc[date_indices, present_cols].values.copy()
            
            # Shuffle rows (tickers) for these columns
            shuffle_idx = rng.permutation(len(date_indices))
            fund_values_shuffled = fund_values[shuffle_idx]
            
            # Apply shuffled values back
            result.loc[date_indices, present_cols] = fund_values_shuffled
        
        return result
    
    else:
        warnings_list.append(f"ALPHA_FEATURE_MODE_FALLBACK:{feature_mode}:unknown_mode")
        return result


def _normalize_ticker(ticker: str) -> str:
    """Normalize ticker to uppercase, strip '.US' suffix."""
    t = str(ticker).strip().upper()
    if t.endswith(".US"):
        t = t[:-3]
    return t


def _build_xgb_regressor(seed: int) -> tuple[object, str, dict, list[str]]:
    """Build XGBoost regressor with deterministic settings."""
    warnings: list[str] = []
    xgb_params = {
        "random_state": seed,
        "n_jobs": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "verbosity": 0,
    }
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(**xgb_params), "xgboost", xgb_params, warnings
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingRegressor
        warnings.append(f"ALPHA_XGB_FALLBACK:sklearn_hist_gb:{str(e)[:30]}")
        hist_params = {
            "loss": "squared_error",
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 300,
            "random_state": seed,
            "early_stopping": False,
        }
        return HistGradientBoostingRegressor(**hist_params), "sklearn_hist_gb", hist_params, warnings


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman IC between two arrays."""
    if x.size < 5 or y.size < 5:
        return None
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    val = float(np.corrcoef(rx, ry)[0, 1])
    return val if np.isfinite(val) else None


def _get_rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Get rebalance dates (month-end or quarter-end) from available dates."""
    dates = dates.sort_values().unique()
    df = pd.DataFrame({"date": dates})
    df["date"] = pd.to_datetime(df["date"])
    
    if freq == "M":
        # Month-end: last date of each month
        df["period"] = df["date"].dt.to_period("M")
    else:  # Q
        df["period"] = df["date"].dt.to_period("Q")
    
    rebal = df.groupby("period")["date"].max().values
    return pd.DatetimeIndex(rebal)


def train_xgb_alpha_from_dataset(
    alpha_dataset_path: str,
    as_of_date: str,
    train_end: str,
    val_end: str,
    out_dir: str,
    *,
    rebalance: str = "Q",
    target_col: str = "fwd_ret_21d",
    seed: int = 42,
    top_k: int = 400,
    ineligible_penalty: float = 1e6,
    min_train_samples: int = 500,
    feature_mode: str = "fund_full",
) -> dict:
    """
    Train alpha model from long-format dataset using walk-forward protocol.
    
    Task 10.2.10: Supports feature_mode for ablation study:
    - fund_full: Use all features (default)
    - tech_only: Exclude FUND columns
    - fund_zeroed: Set FUND values to 0, _is_missing to 1
    - fund_shuffled: Shuffle FUND values within each date
    
    Returns dict with paths to output artifacts.
    """
    np.random.seed(seed)
    
    # Initialize warnings list early (needed for feature_mode transformations)
    warnings: list[str] = []
    
    # Read dataset
    dataset_path = Path(alpha_dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"alpha_dataset_path does not exist: {alpha_dataset_path}")
    
    if str(alpha_dataset_path).endswith(".csv.gz"):
        df = pd.read_csv(alpha_dataset_path, compression="gzip")
    elif str(alpha_dataset_path).endswith(".parquet"):
        df = pd.read_parquet(alpha_dataset_path)
    else:
        df = pd.read_csv(alpha_dataset_path)
    
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].apply(_normalize_ticker)
    
    # PIT cutoff
    as_of_dt = pd.to_datetime(as_of_date)
    df = df[df["date"] <= as_of_dt].copy()
    
    if df.empty:
        raise ValueError("No data remaining after PIT cutoff")
    
    # Identify feature columns (exclude date, ticker, and target columns)
    exclude_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
    target_cols = {c for c in df.columns if c.startswith("fwd_ret_")}
    feature_cols = [c for c in df.columns if c not in exclude_cols and c not in target_cols]
    feature_cols = sorted(feature_cols)
    
    # Task 10.2.10: Validate and apply feature_mode
    if feature_mode not in FEATURE_MODES:
        warnings.append(f"ALPHA_FEATURE_MODE_FALLBACK:{feature_mode}:invalid_mode_using_fund_full")
        feature_mode = "fund_full"
    
    # Identify FUND value columns and missing indicator columns present in dataset
    fund_value_cols_present = [c for c in feature_cols if c in FUND_COLS]
    fund_miss_cols_present = [c for c in feature_cols if c in FUND_MISS_COLS]
    
    # Task 10.2.10: For tech_only mode, exclude FUND columns from feature_cols
    if feature_mode == "tech_only":
        feature_cols = [c for c in feature_cols 
                       if c not in FUND_COLS and c not in FUND_MISS_COLS]
        fund_value_cols_present = []
        fund_miss_cols_present = []
    
    # Task 10.2.10: Apply transformation to entire dataset for zeroed/shuffled modes
    # (must be done before splitting into train/score to preserve consistency)
    if feature_mode in ["fund_zeroed", "fund_shuffled"]:
        df = _apply_feature_mode(
            df, feature_mode, fund_value_cols_present, fund_miss_cols_present, seed, warnings
        )
    
    # Task 10.2.2.1: Determine tech-required columns for eligibility
    # Fundamentals and their missing indicators do NOT affect eligibility
    tech_required_cols = [
        c for c in feature_cols 
        if c not in FUND_COLS and c not in FUND_MISS_COLS
    ]
    
    if not tech_required_cols:
        raise ValueError(
            "No technical features found for eligibility check. "
            f"feature_cols={feature_cols}, FUND_COLS={FUND_COLS}"
        )
    
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataset")
    
    # Parse date bounds
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    # Get rebalance dates
    all_dates = pd.DatetimeIndex(df["date"].unique())
    rebalance_dates = _get_rebalance_dates(all_dates, rebalance)
    
    # Filter rebalance dates to scoring window (after val_end, up to as_of_date)
    score_rebal_dates = rebalance_dates[(rebalance_dates > val_end_dt) & (rebalance_dates <= as_of_dt)]
    score_rebal_dates = score_rebal_dates.sort_values()
    
    if len(score_rebal_dates) == 0:
        raise ValueError("No rebalance dates in scoring window (val_end, as_of_date]")
    
    # Get all tickers
    all_tickers = sorted(df["ticker"].unique())
    
    # Build model
    model, backend, model_params, model_warnings = _build_xgb_regressor(seed)
    warnings.extend(model_warnings)
    
    # Walk-forward scoring
    scores_dict: dict[str, dict[str, float]] = {}
    ic_rows: list[dict] = []
    
    for rebal_date in score_rebal_dates:
        # TRAIN: date <= train_end AND date < rebal_date
        train_mask = (df["date"] <= train_end_dt) & (df["date"] < rebal_date)
        train_df = df[train_mask].copy()
        
        # Task 10.2.2.1: Only require tech features + target to be non-NaN
        # Fundamentals can be NaN since missing indicators capture "missingness"
        required_for_train = tech_required_cols + [target_col]
        train_df = train_df.dropna(subset=required_for_train)
        
        if len(train_df) < min_train_samples:
            warnings.append(f"ALPHA_XGB_SKIP_DATE:{rebal_date.date()}:insufficient_train:{len(train_df)}")
            continue
        
        # Fill NaN fundamentals with 0.0 for training (missing indicators already capture this)
        fund_cols_present = [c for c in feature_cols if c in FUND_COLS]
        for col in fund_cols_present:
            train_df[col] = train_df[col].fillna(0.0)
        
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[target_col].values.astype(np.float32)
        
        # Fit model (train-only fit, no shuffle)
        model.fit(X_train, y_train)
        
        # SCORE: date == rebal_date
        score_df = df[df["date"] == rebal_date].copy()
        
        if len(score_df) == 0:
            warnings.append(f"ALPHA_XGB_SKIP_DATE:{rebal_date.date()}:no_score_rows")
            continue
        
        # Fill NaN fundamentals with 0.0 for scoring (consistent with training)
        fund_cols_present = [c for c in feature_cols if c in FUND_COLS]
        for col in fund_cols_present:
            score_df[col] = score_df[col].fillna(0.0)
        
        # Predict scores
        score_df = score_df.set_index("ticker")
        
        date_scores: dict[str, float] = {}
        valid_scores: list[float] = []
        
        for ticker in all_tickers:
            if ticker in score_df.index:
                row = score_df.loc[ticker]
                features = row[feature_cols].values.astype(np.float32).reshape(1, -1)
                
                # Task 10.2.2.1: Eligibility based on tech_required_cols only
                # Check tech features for NaN (fundamentals do not affect eligibility)
                tech_vals = row[tech_required_cols].values.astype(np.float32)
                if np.isfinite(tech_vals).all() and np.isfinite(features).all():
                    pred = float(model.predict(features)[0])
                    if np.isfinite(pred):
                        date_scores[ticker] = pred
                        valid_scores.append(pred)
                        continue
            
            # Mark as ineligible (will assign penalty later)
            date_scores[ticker] = np.nan
        
        # Apply ineligible penalty
        if valid_scores:
            row_min = min(valid_scores)
            floor = row_min - ineligible_penalty
        else:
            floor = -ineligible_penalty
        
        eligible_count = sum(1 for v in date_scores.values() if np.isfinite(v))
        
        if eligible_count < top_k:
            raise ValueError(
                f"eligible_count({rebal_date.date()})={eligible_count} < top_k={top_k}"
            )
        
        for ticker in date_scores:
            if not np.isfinite(date_scores[ticker]):
                date_scores[ticker] = floor
        
        scores_dict[str(rebal_date.date())] = date_scores
        
        # Compute IC if realized target is available
        # For IC, we need actual future returns which exist in the dataset
        if target_col in score_df.columns:
            realized = score_df[target_col].reindex(all_tickers)
            predicted = pd.Series([date_scores.get(t, np.nan) for t in all_tickers], index=all_tickers)
            
            mask = realized.notna() & predicted.notna()
            if mask.sum() >= 10:
                ic = _spearman_ic(predicted[mask].values, realized[mask].values)
                ic_rows.append({
                    "date": str(rebal_date.date()),
                    "ic_spearman": ic,
                    "eligible_count": int(eligible_count),
                })
    
    if not scores_dict:
        raise ValueError("No dates were scored successfully")
    
    # Build wide scores DataFrame
    dates_sorted = sorted(scores_dict.keys())
    tickers_sorted = sorted(all_tickers)
    
    scores_wide = pd.DataFrame(index=dates_sorted, columns=tickers_sorted, dtype=np.float32)
    for d in dates_sorted:
        for t in tickers_sorted:
            scores_wide.loc[d, t] = scores_dict[d].get(t, np.nan)
    
    scores_wide = scores_wide.reset_index().rename(columns={"index": "date"})
    
    # Verify all finite
    numeric_cols = [c for c in scores_wide.columns if c != "date"]
    if not np.isfinite(scores_wide[numeric_cols].values.astype(float)).all():
        raise ValueError("scores.csv contains non-finite values")
    
    # Write outputs
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    scores_csv_path = out_path / "scores.csv"
    scores_wide.to_csv(scores_csv_path, index=False, float_format="%.10f", lineterminator="\n")
    
    ic_csv_path = out_path / "ic_by_date.csv"
    ic_df = pd.DataFrame(ic_rows) if ic_rows else pd.DataFrame(columns=["date", "ic_spearman", "eligible_count"])
    ic_df.to_csv(ic_csv_path, index=False, lineterminator="\n")
    
    ic_values = [r["ic_spearman"] for r in ic_rows if r["ic_spearman"] is not None]
    
    summary = {
        "schema_version": "10.2.10",
        "config": {
            "alpha_dataset_path": str(alpha_dataset_path),
            "as_of_date": as_of_date,
            "train_end": train_end,
            "val_end": val_end,
            "rebalance": rebalance,
            "target_col": target_col,
            "seed": seed,
            "top_k": top_k,
            "ineligible_penalty": ineligible_penalty,
            "min_train_samples": min_train_samples,
            "feature_mode": feature_mode,
        },
        "model": {
            "type": backend,
            "params": model_params,
        },
        "features": feature_cols,
        "n_features_used": len(feature_cols),
        "results": {
            "n_dates_scored": len(dates_sorted),
            "n_tickers": len(tickers_sorted),
            "ic_spearman_mean": float(np.mean(ic_values)) if ic_values else None,
            "ic_spearman_n": len(ic_values),
        },
        "warnings": warnings,
    }
    
    summary_path = out_path / "summary.json"
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    print(f"Wrote: {scores_csv_path}")
    print(f"Wrote: {ic_csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Summary: {len(dates_sorted)} dates scored, {len(tickers_sorted)} tickers")
    
    return {
        "scores_csv": str(scores_csv_path),
        "ic_csv": str(ic_csv_path),
        "summary_json": str(summary_path),
    }


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train alpha model from dataset (walk-forward)")
    parser.add_argument("--alpha-dataset-path", required=True, help="Path to alpha_dataset.csv.gz")
    parser.add_argument("--as-of-date", required=True, help="PIT cutoff date (YYYY-MM-DD)")
    parser.add_argument("--train-end", required=True, help="Training data end date (YYYY-MM-DD)")
    parser.add_argument("--val-end", required=True, help="Validation data end date (YYYY-MM-DD)")
    parser.add_argument("--rebalance", choices=["M", "Q"], default="Q", help="Rebalance frequency")
    parser.add_argument("--target-col", default="fwd_ret_21d", help="Target column name")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-k", type=int, default=400, help="Minimum eligible tickers per date")
    parser.add_argument("--ineligible-penalty", type=float, default=1e6, help="Penalty for ineligible tickers")
    parser.add_argument("--min-train-samples", type=int, default=500, help="Minimum training samples")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="fund_full",
                        help="Feature mode for ablation study (default: fund_full)")
    
    args = parser.parse_args()
    
    try:
        train_xgb_alpha_from_dataset(
            alpha_dataset_path=args.alpha_dataset_path,
            as_of_date=args.as_of_date,
            train_end=args.train_end,
            val_end=args.val_end,
            out_dir=args.out_dir,
            rebalance=args.rebalance,
            target_col=args.target_col,
            seed=args.seed,
            top_k=args.top_k,
            ineligible_penalty=args.ineligible_penalty,
            min_train_samples=args.min_train_samples,
            feature_mode=args.feature_mode,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
