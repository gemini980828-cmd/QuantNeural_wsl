"""
Real-Data Train/Eval Module for QUANT-NEURAL.

Provides:
- split_xy_by_date: Time-series split into train/val/test
- fit_predict_baseline_mlp: Train MLP with train-only RankGauss fit
- evaluate_regression: Compute MSE/MAE metrics
- run_baseline_real_data_mlp_experiment: End-to-end experiment runner

Point-in-Time (PIT) Rules:
- Train-only fit: RankGauss fitted on train, transform-only for val/test
- No shuffle: Keras fit uses shuffle=False
- Time-series discipline: strict chronological splits
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import QuantDataProcessor
from src.models import MLPParams, SectorPredictorMLP

# Try to import tensorflow for seed setting
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def split_xy_by_date(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split X and Y chronologically into train/val/test.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature frame with DatetimeIndex.
    Y : pd.DataFrame
        Label frame with DatetimeIndex (must match X.index).
    train_end : str
        End date for training set (inclusive), "YYYY-MM-DD".
    val_end : str
        End date for validation set (inclusive), "YYYY-MM-DD".
    
    Returns
    -------
    tuple
        (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    
    Raises
    ------
    ValueError
        If X and Y indexes don't match, or any split is empty.
    
    Notes
    -----
    Split boundaries:
    - Train: index <= train_end
    - Val:   train_end < index <= val_end
    - Test:  index > val_end
    """
    # Validate identical indexes
    if not X.index.equals(Y.index):
        raise ValueError("X and Y must have identical DatetimeIndex")
    
    train_end_dt = pd.to_datetime(train_end, format="%Y-%m-%d")
    val_end_dt = pd.to_datetime(val_end, format="%Y-%m-%d")
    
    # Create masks
    train_mask = X.index <= train_end_dt
    val_mask = (X.index > train_end_dt) & (X.index <= val_end_dt)
    test_mask = X.index > val_end_dt
    
    # Split
    X_train = X.loc[train_mask].copy()
    Y_train = Y.loc[train_mask].copy()
    X_val = X.loc[val_mask].copy()
    Y_val = Y.loc[val_mask].copy()
    X_test = X.loc[test_mask].copy()
    Y_test = Y.loc[test_mask].copy()
    
    # Validate non-empty splits
    if len(X_train) == 0:
        raise ValueError("Train split is empty")
    if len(X_val) == 0:
        raise ValueError("Val split is empty")
    if len(X_test) == 0:
        raise ValueError("Test split is empty")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def fit_predict_baseline_mlp(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    seed: int = 42,
    rankgauss: bool = True,
    epochs: int = 1,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Train baseline MLP and predict on test set.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (20 columns).
    Y_train : pd.DataFrame
        Training labels (10 columns).
    X_val : pd.DataFrame
        Validation features.
    Y_val : pd.DataFrame
        Validation labels.
    X_test : pd.DataFrame
        Test features.
    seed : int
        Random seed for reproducibility.
    rankgauss : bool
        Whether to apply RankGauss transformation.
    epochs : int
        Training epochs (default 1 for test speed).
    batch_size : int
        Training batch size.
    
    Returns
    -------
    pd.DataFrame
        Predictions with:
        - index == X_test.index
        - columns == Y_train.columns
        - float dtype, finite values
    
    Notes
    -----
    - RankGauss is fitted on TRAIN ONLY, then transforms val/test.
    - MLP training uses shuffle=False (enforced by SectorPredictorMLP).
    """
    # Set deterministic seeds
    np.random.seed(seed)
    if HAS_TF:
        tf.random.set_seed(seed)
    
    # Convert to numpy for processing
    X_train_arr = X_train.values.astype(np.float32)
    X_val_arr = X_val.values.astype(np.float32)
    X_test_arr = X_test.values.astype(np.float32)
    
    Y_train_arr = Y_train.values.astype(np.float32)
    Y_val_arr = Y_val.values.astype(np.float32)
    
    # Handle NaN in Y by replacing with 0 for training
    # (last row of Y typically has NaN from label shift)
    Y_train_arr = np.nan_to_num(Y_train_arr, nan=0.0)
    Y_val_arr = np.nan_to_num(Y_val_arr, nan=0.0)
    
    # RankGauss: fit on train only, transform all
    if rankgauss:
        processor = QuantDataProcessor(rankgauss=True, random_state=seed)
        
        # Fit on TRAIN ONLY
        processor.fit_rankgauss(X_train_arr)
        
        # Transform all sets
        X_train_arr = processor.transform_rankgauss(X_train_arr)
        X_val_arr = processor.transform_rankgauss(X_val_arr)
        X_test_arr = processor.transform_rankgauss(X_test_arr)
    
    # Create MLP with custom params
    params = MLPParams(
        epochs=epochs,
        batch_size=batch_size,
    )
    model = SectorPredictorMLP(params)
    
    # Train (shuffle=False enforced internally)
    model.fit(X_train_arr, Y_train_arr, X_val_arr, Y_val_arr)
    
    # Predict on test
    Y_pred_arr = model.predict(X_test_arr)
    
    # Convert to DataFrame
    Y_pred = pd.DataFrame(
        Y_pred_arr,
        index=X_test.index,
        columns=Y_train.columns,
    )
    
    # Ensure float dtype
    Y_pred = Y_pred.astype(float)
    
    return Y_pred


def evaluate_regression(
    Y_true: pd.DataFrame,
    Y_pred: pd.DataFrame,
) -> dict:
    """
    Evaluate regression predictions.
    
    Parameters
    ----------
    Y_true : pd.DataFrame
        Ground truth labels.
    Y_pred : pd.DataFrame
        Predicted labels.
    
    Returns
    -------
    dict
        Metrics including "mse" and "mae" (ignoring NaN).
    """
    # Align indexes
    common_idx = Y_true.index.intersection(Y_pred.index)
    y_true = Y_true.loc[common_idx].values.flatten()
    y_pred = Y_pred.loc[common_idx].values.flatten()
    
    # Remove NaN pairs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {"mse": np.nan, "mae": np.nan}
    
    mse = float(np.mean((y_true_clean - y_pred_clean) ** 2))
    mae = float(np.mean(np.abs(y_true_clean - y_pred_clean)))
    
    return {"mse": mse, "mae": mae}


def run_baseline_real_data_mlp_experiment(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    train_end: str,
    val_end: str,
    seed: int = 42,
    rankgauss: bool = True,
    epochs: int = 1,
    batch_size: int = 32,
) -> dict:
    """
    Run complete baseline MLP experiment on real data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Full feature frame (20 columns).
    Y : pd.DataFrame
        Full label frame (10 columns).
    train_end : str
        End date for training set.
    val_end : str
        End date for validation set.
    seed : int
        Random seed.
    rankgauss : bool
        Whether to apply RankGauss.
    epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    
    Returns
    -------
    dict
        Results with keys:
        - "metrics": dict with mse, mae
        - "n_train": int
        - "n_val": int
        - "n_test": int
        - "y_pred_test": pd.DataFrame
    """
    # Split
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_xy_by_date(
        X, Y, train_end=train_end, val_end=val_end
    )
    
    # Train and predict
    Y_pred_test = fit_predict_baseline_mlp(
        X_train, Y_train, X_val, Y_val, X_test,
        seed=seed,
        rankgauss=rankgauss,
        epochs=epochs,
        batch_size=batch_size,
    )
    
    # Evaluate
    metrics = evaluate_regression(Y_test, Y_pred_test)
    
    return {
        "metrics": metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "y_pred_test": Y_pred_test,
    }


def run_shadow_scoring_mlp(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    sector_to_tickers: dict[str, list[str]] | None = None,
    seed: int = 42,
    rankgauss: bool = True,
    epochs: int = 10,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Train MLP and export shadow scores to CSV artifact.
    
    This function generates "shadow" scores by:
    1) Training the Phase-4 MLP on train data only (PIT-safe)
    2) Predicting on val+test evaluation dates
    3) Optionally broadcasting sector scores to tickers
    4) Saving to CSV in standard scores format
    
    Parameters
    ----------
    X : pd.DataFrame
        Full feature frame (20 columns, DatetimeIndex).
    Y : pd.DataFrame
        Full label frame (10 columns, DatetimeIndex).
    train_end : str
        End date for training set.
    val_end : str
        End date for validation set.
    output_csv_path : str
        Path to save the shadow scores CSV.
    sector_to_tickers : dict[str, list[str]], optional
        Mapping from sector label (e.g. "S0") to list of tickers.
        If provided, broadcasts sector scores to ticker columns.
        If None, outputs sector-level scores only.
    seed : int
        Random seed for reproducibility.
    rankgauss : bool
        Whether to apply RankGauss transformation.
    epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    
    Returns
    -------
    pd.DataFrame
        Shadow scores DataFrame (also saved to CSV).
        - Index: DatetimeIndex (evaluation dates)
        - Columns: tickers (if sector_to_tickers provided) or sectors
    
    Notes
    -----
    - PIT/no-leakage enforced: RankGauss fit on train only
    - shuffle=False enforced in SectorPredictorMLP
    - Deterministic seeding: Python, NumPy, TensorFlow
    """
    import os
    import random
    
    # Set deterministic seeds
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TF:
        tf.random.set_seed(seed)
        # Additional TF determinism settings
        try:
            tf.keras.utils.set_random_seed(seed)
        except AttributeError:
            pass  # Older TF versions
    
    # Split data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_xy_by_date(
        X, Y, train_end=train_end, val_end=val_end
    )
    
    # Convert to numpy and handle NaN
    X_train_arr = X_train.values.astype(np.float32)
    X_val_arr = X_val.values.astype(np.float32)
    X_eval_arr = np.vstack([X_val.values, X_test.values]).astype(np.float32)
    
    Y_train_arr = np.nan_to_num(Y_train.values.astype(np.float32), nan=0.0)
    Y_val_arr = np.nan_to_num(Y_val.values.astype(np.float32), nan=0.0)
    
    # RankGauss: fit on TRAIN only
    if rankgauss:
        processor = QuantDataProcessor(rankgauss=True, random_state=seed)
        processor.fit_rankgauss(X_train_arr)
        X_train_arr = processor.transform_rankgauss(X_train_arr)
        X_val_arr = processor.transform_rankgauss(X_val_arr)
        X_eval_arr = processor.transform_rankgauss(X_eval_arr)
    
    # Create and train MLP
    params = MLPParams(epochs=epochs, batch_size=batch_size)
    model = SectorPredictorMLP(params)
    model.fit(X_train_arr, Y_train_arr, X_val_arr, Y_val_arr)
    
    # Predict on evaluation dates (val + test)
    Y_pred_arr = model.predict(X_eval_arr)
    
    # Build evaluation index (val + test dates)
    eval_index = X_val.index.append(X_test.index)
    
    # Create sector predictions DataFrame
    # Y_train columns are like ["S0_Y", "S1_Y", ..., "S9_Y"]
    sector_labels = [f"S{i}" for i in range(10)]
    sector_scores = pd.DataFrame(
        Y_pred_arr,
        index=eval_index,
        columns=sector_labels,
    )
    
    # Broadcast to tickers if mapping provided
    if sector_to_tickers is not None:
        ticker_scores_dict = {}
        for sector, tickers in sector_to_tickers.items():
            if sector in sector_scores.columns:
                for ticker in tickers:
                    ticker_scores_dict[ticker] = sector_scores[sector].values
        
        # Sort columns alphabetically for determinism
        sorted_tickers = sorted(ticker_scores_dict.keys())
        scores_df = pd.DataFrame(
            {t: ticker_scores_dict[t] for t in sorted_tickers},
            index=eval_index,
        )
    else:
        # Use sector-level scores directly
        scores_df = sector_scores
    
    # Ensure float dtype and handle any edge cases
    scores_df = scores_df.astype(float)
    
    # Verify data sanity
    if not np.all(np.isfinite(scores_df.values)):
        raise ValueError("Shadow scores contain NaN or inf values")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    
    # Reset index to include date column explicitly
    scores_out = scores_df.copy()
    scores_out.index.name = "date"
    scores_out.to_csv(output_csv_path)
    
    return scores_df

