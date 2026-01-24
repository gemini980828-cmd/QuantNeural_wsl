"""
Tests for src/real_data_train_eval.py

Covers:
- Time-series split boundaries
- Empty split validation
- MLP shapes and index alignment
- RankGauss train-only fit
- Experiment runner smoke test
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.real_data_train_eval import (
    split_xy_by_date,
    fit_predict_baseline_mlp,
    evaluate_regression,
    run_baseline_real_data_mlp_experiment,
)


def _create_synthetic_xy(n_months: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic aligned X and Y DataFrames."""
    index = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    
    # X: 20 features
    X_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
    X_data = np.random.default_rng(42).random((n_months, 20))
    X = pd.DataFrame(X_data, index=index, columns=X_cols)
    
    # Y: 10 labels
    Y_cols = [f"S{i}_Y" for i in range(10)]
    Y_data = np.random.default_rng(42).random((n_months, 10))
    Y = pd.DataFrame(Y_data, index=index, columns=Y_cols)
    
    return X, Y


class TestSplitXYByDate:
    """Test split_xy_by_date function."""
    
    def test_split_xy_by_date_boundaries(self):
        """Test that split respects exact date boundaries."""
        # Create 6 month-end rows
        X, Y = _create_synthetic_xy(n_months=6)
        # Index: 2022-01-31, 2022-02-28, 2022-03-31, 2022-04-30, 2022-05-31, 2022-06-30
        
        # train_end = 2022-02-28, val_end = 2022-04-30
        train_end = "2022-02-28"
        val_end = "2022-04-30"
        
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_xy_by_date(
            X, Y, train_end=train_end, val_end=val_end
        )
        
        # Train: <= 2022-02-28 (Jan, Feb) -> 2 rows
        assert len(X_train) == 2
        assert X_train.index.max() <= pd.Timestamp("2022-02-28")
        
        # Val: 2022-02-28 < idx <= 2022-04-30 (Mar, Apr) -> 2 rows
        assert len(X_val) == 2
        assert X_val.index.min() > pd.Timestamp("2022-02-28")
        assert X_val.index.max() <= pd.Timestamp("2022-04-30")
        
        # Test: > 2022-04-30 (May, Jun) -> 2 rows
        assert len(X_test) == 2
        assert X_test.index.min() > pd.Timestamp("2022-04-30")
        
        # No overlap
        all_indices = set(X_train.index) | set(X_val.index) | set(X_test.index)
        assert len(all_indices) == len(X)
        
        # Y splits match X splits
        assert X_train.index.equals(Y_train.index)
        assert X_val.index.equals(Y_val.index)
        assert X_test.index.equals(Y_test.index)
    
    def test_split_raises_on_empty_split(self):
        """Test that empty splits raise ValueError."""
        X, Y = _create_synthetic_xy(n_months=6)
        
        # Make val empty by setting val_end same as train_end
        with pytest.raises(ValueError, match="Val split is empty"):
            split_xy_by_date(X, Y, train_end="2022-02-28", val_end="2022-02-28")
        
        # Make test empty by setting val_end after all data
        with pytest.raises(ValueError, match="Test split is empty"):
            split_xy_by_date(X, Y, train_end="2022-02-28", val_end="2022-12-31")
    
    def test_split_raises_on_mismatched_index(self):
        """Test that mismatched X and Y indexes raise ValueError."""
        X, Y = _create_synthetic_xy(n_months=6)
        Y_bad = Y.iloc[:-1]  # Remove last row
        
        with pytest.raises(ValueError, match="identical DatetimeIndex"):
            split_xy_by_date(X, Y_bad, train_end="2022-02-28", val_end="2022-04-30")


class TestFitPredictBaselineMLP:
    """Test fit_predict_baseline_mlp function."""
    
    def test_fit_predict_baseline_mlp_shapes_and_index(self):
        """Test output shape and index alignment."""
        # Create synthetic data
        X, Y = _create_synthetic_xy(n_months=12)
        
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_xy_by_date(
            X, Y, train_end="2022-04-30", val_end="2022-08-31"
        )
        
        # Train and predict
        Y_pred = fit_predict_baseline_mlp(
            X_train, Y_train, X_val, Y_val, X_test,
            seed=42,
            rankgauss=True,
            epochs=1,
            batch_size=32,
        )
        
        # Output is DataFrame
        assert isinstance(Y_pred, pd.DataFrame)
        
        # Index matches X_test
        assert Y_pred.index.equals(X_test.index)
        
        # Columns match Y_train
        assert list(Y_pred.columns) == list(Y_train.columns)
        
        # Shape is (n_test, 10)
        assert Y_pred.shape == (len(X_test), 10)
        
        # All values finite
        assert np.isfinite(Y_pred.values).all()
    
    def test_fit_predict_without_rankgauss(self):
        """Test training without RankGauss transformation."""
        X, Y = _create_synthetic_xy(n_months=12)
        
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_xy_by_date(
            X, Y, train_end="2022-04-30", val_end="2022-08-31"
        )
        
        Y_pred = fit_predict_baseline_mlp(
            X_train, Y_train, X_val, Y_val, X_test,
            seed=42,
            rankgauss=False,  # No RankGauss
            epochs=1,
            batch_size=32,
        )
        
        assert Y_pred.shape == (len(X_test), 10)
        assert np.isfinite(Y_pred.values).all()


class TestRankGaussTrainOnlyFit:
    """Test that RankGauss is fitted on train only."""
    
    def test_rankgauss_fit_called_train_only(self):
        """Test that fit_rankgauss is called exactly once (on train)."""
        X, Y = _create_synthetic_xy(n_months=12)
        
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_xy_by_date(
            X, Y, train_end="2022-04-30", val_end="2022-08-31"
        )
        
        fit_call_count = 0
        original_fit = None
        
        # Capture original method for proper patching
        from src.preprocessing import QuantDataProcessor
        original_fit = QuantDataProcessor.fit_rankgauss
        
        def counting_fit(self, X_train_arr):
            nonlocal fit_call_count
            fit_call_count += 1
            return original_fit(self, X_train_arr)
        
        with patch.object(QuantDataProcessor, 'fit_rankgauss', counting_fit):
            fit_predict_baseline_mlp(
                X_train, Y_train, X_val, Y_val, X_test,
                seed=42,
                rankgauss=True,
                epochs=1,
            )
        
        assert fit_call_count == 1, f"fit_rankgauss should be called exactly once, got {fit_call_count}"


class TestEvaluateRegression:
    """Test evaluate_regression function."""
    
    def test_evaluate_regression_metrics(self):
        """Test MSE and MAE calculation."""
        index = pd.date_range("2022-01-31", periods=3, freq="ME")
        
        Y_true = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
        }, index=index)
        
        Y_pred = pd.DataFrame({
            "A": [1.1, 2.1, 3.1],
            "B": [4.1, 5.1, 6.1],
        }, index=index)
        
        metrics = evaluate_regression(Y_true, Y_pred)
        
        # All errors are 0.1
        expected_mse = 0.01  # 0.1^2
        expected_mae = 0.1
        
        assert abs(metrics["mse"] - expected_mse) < 1e-6
        assert abs(metrics["mae"] - expected_mae) < 1e-6
    
    def test_evaluate_regression_ignores_nan(self):
        """Test that NaN values are ignored in metric calculation."""
        index = pd.date_range("2022-01-31", periods=3, freq="ME")
        
        Y_true = pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
        }, index=index)
        
        Y_pred = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],  # Second row ignored due to NaN in true
        }, index=index)
        
        metrics = evaluate_regression(Y_true, Y_pred)
        
        # Only rows 0 and 2 are used, both have 0 error
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0


class TestRunBaselineExperiment:
    """Test run_baseline_real_data_mlp_experiment function."""
    
    def test_run_baseline_real_data_mlp_experiment_smoke(self):
        """Smoke test for complete experiment runner."""
        X, Y = _create_synthetic_xy(n_months=12)
        
        result = run_baseline_real_data_mlp_experiment(
            X, Y,
            train_end="2022-04-30",
            val_end="2022-08-31",
            seed=42,
            rankgauss=True,
            epochs=1,
            batch_size=32,
        )
        
        # Check required keys
        assert "metrics" in result
        assert "n_train" in result
        assert "n_val" in result
        assert "n_test" in result
        assert "y_pred_test" in result
        
        # Check metrics are finite
        assert np.isfinite(result["metrics"]["mse"])
        assert np.isfinite(result["metrics"]["mae"])
        
        # Check counts
        assert result["n_train"] == 4  # Jan-Apr
        assert result["n_val"] == 4    # May-Aug
        assert result["n_test"] == 4   # Sep-Dec
        
        # Check y_pred_test is DataFrame
        assert isinstance(result["y_pred_test"], pd.DataFrame)


class TestShadowScoringMLP:
    """Test run_shadow_scoring_mlp function (Task 9.3.0)."""
    
    def test_shadow_scoring_creates_csv(self, tmp_path):
        """Test that shadow scoring creates a CSV file with valid content."""
        from src.real_data_train_eval import run_shadow_scoring_mlp
        
        X, Y = _create_synthetic_xy(n_months=12)
        output_path = tmp_path / "scores_mlp.csv"
        
        # Run shadow scoring
        scores_df = run_shadow_scoring_mlp(
            X, Y,
            train_end="2022-04-30",
            val_end="2022-08-31",
            output_csv_path=str(output_path),
            seed=42,
            epochs=1,  # Minimal for speed
            batch_size=32,
        )
        
        # CSV file exists
        assert output_path.exists(), "CSV file not created"
        
        # Read back and validate
        loaded = pd.read_csv(output_path, index_col="date", parse_dates=True)
        
        # Index is datetime, monotonic, unique
        assert loaded.index.is_monotonic_increasing, "Index not monotonic"
        assert loaded.index.is_unique, "Index not unique"
        
        # Values are finite
        assert np.all(np.isfinite(loaded.values)), "CSV contains NaN/inf"
        
        # Returned DataFrame matches saved CSV
        assert len(scores_df) == len(loaded)
        assert set(scores_df.columns) == set(loaded.columns)
    
    def test_shadow_scoring_determinism(self, tmp_path):
        """Test that running twice with same seed produces identical outputs."""
        from src.real_data_train_eval import run_shadow_scoring_mlp
        
        X, Y = _create_synthetic_xy(n_months=12)
        
        # First run
        path1 = tmp_path / "run1" / "scores.csv"
        scores1 = run_shadow_scoring_mlp(
            X, Y,
            train_end="2022-04-30",
            val_end="2022-08-31",
            output_csv_path=str(path1),
            seed=42,
            epochs=1,
            batch_size=32,
        )
        
        # Second run with same seed
        path2 = tmp_path / "run2" / "scores.csv"
        scores2 = run_shadow_scoring_mlp(
            X, Y,
            train_end="2022-04-30",
            val_end="2022-08-31",
            output_csv_path=str(path2),
            seed=42,
            epochs=1,
            batch_size=32,
        )
        
        # DataFrames are exactly equal
        pd.testing.assert_frame_equal(scores1, scores2, check_exact=True)
        
        # CSV files are byte-identical
        content1 = path1.read_text()
        content2 = path2.read_text()
        assert content1 == content2, "CSV outputs differ between runs with same seed"
    
    def test_shadow_scoring_with_sector_to_ticker_broadcast(self, tmp_path):
        """Test broadcasting sector scores to ticker columns."""
        from src.real_data_train_eval import run_shadow_scoring_mlp
        
        X, Y = _create_synthetic_xy(n_months=12)
        output_path = tmp_path / "scores_tickers.csv"
        
        # Simple mapping: 2 tickers per sector
        sector_to_tickers = {
            "S0": ["AAPL", "MSFT"],
            "S1": ["GOOG", "META"],
            "S2": ["AMZN", "TSLA"],
        }
        
        scores_df = run_shadow_scoring_mlp(
            X, Y,
            train_end="2022-04-30",
            val_end="2022-08-31",
            output_csv_path=str(output_path),
            sector_to_tickers=sector_to_tickers,
            seed=42,
            epochs=1,
        )
        
        # Output has ticker columns (sorted alphabetically)
        expected_tickers = sorted(["AAPL", "MSFT", "GOOG", "META", "AMZN", "TSLA"])
        assert list(scores_df.columns) == expected_tickers
        
        # Tickers in same sector have same scores
        assert (scores_df["AAPL"] == scores_df["MSFT"]).all()
        assert (scores_df["GOOG"] == scores_df["META"]).all()
        assert (scores_df["AMZN"] == scores_df["TSLA"]).all()
    
    def test_shadow_scoring_eval_dates_cover_val_and_test(self, tmp_path):
        """Test that output covers val + test evaluation dates."""
        from src.real_data_train_eval import run_shadow_scoring_mlp
        
        X, Y = _create_synthetic_xy(n_months=12)
        output_path = tmp_path / "scores.csv"
        
        scores_df = run_shadow_scoring_mlp(
            X, Y,
            train_end="2022-04-30",  # 4 months train
            val_end="2022-08-31",     # 4 months val
            output_csv_path=str(output_path),
            seed=42,
            epochs=1,
        )
        
        # Val dates: May, Jun, Jul, Aug (4)
        # Test dates: Sep, Oct, Nov, Dec (4)
        # Total: 8 evaluation dates
        assert len(scores_df) == 8, f"Expected 8 eval dates, got {len(scores_df)}"
        
        # First date is after train_end
        assert scores_df.index.min() > pd.Timestamp("2022-04-30")

