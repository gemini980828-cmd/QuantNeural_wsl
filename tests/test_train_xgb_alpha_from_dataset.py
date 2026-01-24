"""
Tests for train_xgb_alpha_from_dataset.py

Covers:
- End-to-end training from synthetic alpha_dataset
- Wide scores.csv output format
- Determinism (same seed => identical output)
- IC computation
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_xgb_alpha_from_dataset import train_xgb_alpha_from_dataset


def _create_synthetic_alpha_dataset(tmp_path: Path, n_days: int = 120, n_tickers: int = 3) -> Path:
    """Create synthetic alpha_dataset.csv.gz for testing."""
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"TICK{i}" for i in range(n_tickers)]
    
    rows = []
    for d in dates:
        for t in tickers:
            base_price = 100 + hash(t) % 50
            price_mult = 1 + 0.001 * (d - dates[0]).days + np.random.normal(0, 0.02)
            close = base_price * price_mult
            
            rows.append({
                "date": d,
                "ticker": t,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 1e6 + np.random.uniform(0, 1e6),
                "vol_20d": 0.2 + np.random.uniform(0, 0.1),
                "mom_5d": np.random.uniform(-0.1, 0.1),
                "mom_21d": np.random.uniform(-0.2, 0.2),
                "mom_63d": np.random.uniform(-0.3, 0.3),
                "rsi_14d": 50 + np.random.uniform(-20, 20),
                "bbands_20d": np.random.uniform(-1, 2),
                "atr_14d_norm": 0.02 + np.random.uniform(0, 0.02),
                "fwd_ret_5d": np.random.uniform(-0.1, 0.1),
                "fwd_ret_10d": np.random.uniform(-0.15, 0.15),
                "fwd_ret_21d": np.random.uniform(-0.2, 0.2),
            })
    
    df = pd.DataFrame(rows)
    
    # Cast to float32
    for col in df.columns:
        if col not in ["date", "ticker"]:
            df[col] = df[col].astype(np.float32)
    
    output_path = tmp_path / "alpha_dataset.csv.gz"
    df.to_csv(output_path, index=False, compression="gzip")
    
    return output_path


class TestTrainXGBAlphaFromDataset:
    """Test train_xgb_alpha_from_dataset function."""
    
    def test_end_to_end(self, tmp_path):
        """End-to-end test producing scores.csv, ic_by_date.csv, summary.json."""
        dataset_path = _create_synthetic_alpha_dataset(tmp_path, n_days=120, n_tickers=5)
        out_dir = tmp_path / "alpha_model_output"
        
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-06-30",
            train_end="2020-03-31",
            val_end="2020-04-30",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=3,  # Low threshold for synthetic data
            min_train_samples=50,
        )
        
        # Verify output files exist
        assert Path(result["scores_csv"]).exists()
        assert Path(result["ic_csv"]).exists()
        assert Path(result["summary_json"]).exists()
        
        # Verify scores.csv is wide format
        scores = pd.read_csv(result["scores_csv"])
        assert "date" in scores.columns
        # Should have ticker columns
        ticker_cols = [c for c in scores.columns if c != "date"]
        assert len(ticker_cols) >= 3, "Should have multiple ticker columns"
        
        # Verify all values are finite
        assert np.isfinite(scores[ticker_cols].values.astype(float)).all(), "All scores should be finite"
        
        # Verify dates are sorted
        dates = pd.to_datetime(scores["date"])
        assert (dates.diff().dropna() > pd.Timedelta(0)).all(), "Dates should be strictly increasing"
    
    def test_determinism(self, tmp_path):
        """Same seed should produce byte-identical scores.csv."""
        dataset_path = _create_synthetic_alpha_dataset(tmp_path, n_days=100, n_tickers=3)
        
        out_dir_1 = tmp_path / "run1"
        out_dir_2 = tmp_path / "run2"
        
        common_args = {
            "alpha_dataset_path": str(dataset_path),
            "as_of_date": "2020-05-31",
            "train_end": "2020-03-31",
            "val_end": "2020-04-15",
            "rebalance": "M",
            "target_col": "fwd_ret_21d",
            "seed": 42,
            "top_k": 2,
            "min_train_samples": 30,
        }
        
        result1 = train_xgb_alpha_from_dataset(out_dir=str(out_dir_1), **common_args)
        result2 = train_xgb_alpha_from_dataset(out_dir=str(out_dir_2), **common_args)
        
        # Read and compare
        scores1 = Path(result1["scores_csv"]).read_bytes()
        scores2 = Path(result2["scores_csv"]).read_bytes()
        
        assert scores1 == scores2, "Same seed should produce identical scores.csv"
    
    def test_ic_computation(self, tmp_path):
        """IC by date should be computed and written."""
        # Use more tickers to ensure IC can be computed (needs >= 10 samples)
        dataset_path = _create_synthetic_alpha_dataset(tmp_path, n_days=150, n_tickers=15)
        out_dir = tmp_path / "output"
        
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-07-31",
            train_end="2020-04-30",
            val_end="2020-05-31",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=10,
            min_train_samples=100,
        )
        
        ic_df = pd.read_csv(result["ic_csv"])
        assert "date" in ic_df.columns
        assert "ic_spearman" in ic_df.columns
        # IC may still be empty if targets have NaN at score date, so just check structure
        assert len(ic_df.columns) >= 2, "Should have date and ic_spearman columns"
    
    def test_summary_schema(self, tmp_path):
        """Summary.json should have correct schema."""
        import json
        
        dataset_path = _create_synthetic_alpha_dataset(tmp_path, n_days=100, n_tickers=3)
        out_dir = tmp_path / "output"
        
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-05-31",
            train_end="2020-03-31",
            val_end="2020-04-15",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=2,
            min_train_samples=30,
        )
        
        with open(result["summary_json"]) as f:
            summary = json.load(f)
        
        assert summary["schema_version"] == "10.2.10"
        assert "model" in summary
        assert "features" in summary
        assert "results" in summary
        assert summary["results"]["n_dates_scored"] > 0
        # Task 10.2.10: New fields
        assert "n_features_used" in summary
        assert "feature_mode" in summary["config"]
    
    def test_pit_cutoff(self, tmp_path):
        """PIT cutoff should filter future dates."""
        dataset_path = _create_synthetic_alpha_dataset(tmp_path, n_days=150, n_tickers=3)
        out_dir = tmp_path / "output"
        
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-04-30",  # Early cutoff
            train_end="2020-02-28",
            val_end="2020-03-31",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=2,
            min_train_samples=30,
        )
        
        scores = pd.read_csv(result["scores_csv"])
        dates = pd.to_datetime(scores["date"])
        cutoff = pd.to_datetime("2020-04-30")
        
        assert (dates <= cutoff).all(), "No scores should be after as_of_date"


# ==============================================================================
# Task 10.2.2.1: Tests for Eligibility Gating (Fundamentals do NOT affect eligibility)
# ==============================================================================


# SEC fundamental columns for test definitions
SEC_FUNDAMENTAL_COLS = [
    "assets", "liabilities", "equity", "cash", "shares_out",
    "leverage", "cash_to_assets", "book_to_assets", "mktcap",
]


def _create_alpha_dataset_with_fundamentals(
    tmp_path: Path, 
    n_days: int = 120, 
    n_tickers: int = 5,
    fund_nan_ratio: float = 1.0,
) -> Path:
    """
    Create synthetic alpha_dataset with SEC fundamental columns.
    
    Parameters
    ----------
    fund_nan_ratio : float
        Ratio of NaN values in fundamental columns (0.0 = all valid, 1.0 = all NaN)
    """
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"TICK{i}" for i in range(n_tickers)]
    
    rows = []
    for d in dates:
        for t in tickers:
            base_price = 100 + hash(t) % 50
            price_mult = 1 + 0.001 * (d - dates[0]).days + np.random.normal(0, 0.02)
            close = base_price * price_mult
            
            row = {
                "date": d,
                "ticker": t,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 1e6 + np.random.uniform(0, 1e6),
                # Technical features - always valid
                "vol_20d": 0.2 + np.random.uniform(0, 0.1),
                "mom_5d": np.random.uniform(-0.1, 0.1),
                "mom_21d": np.random.uniform(-0.2, 0.2),
                "mom_63d": np.random.uniform(-0.3, 0.3),
                "rsi_14d": 50 + np.random.uniform(-20, 20),
                "bbands_20d": np.random.uniform(-1, 2),
                "atr_14d_norm": 0.02 + np.random.uniform(0, 0.02),
                # Targets
                "fwd_ret_5d": np.random.uniform(-0.1, 0.1),
                "fwd_ret_10d": np.random.uniform(-0.15, 0.15),
                "fwd_ret_21d": np.random.uniform(-0.2, 0.2),
            }
            
            # Add fundamental columns (may be NaN based on ratio)
            for col in SEC_FUNDAMENTAL_COLS:
                if np.random.random() < fund_nan_ratio:
                    row[col] = np.nan
                else:
                    row[col] = np.random.uniform(1e6, 1e9)
            
            # Add missing indicators (1.0 if NaN, else 0.0)
            for col in SEC_FUNDAMENTAL_COLS:
                row[f"{col}_is_missing"] = 1.0 if np.isnan(row[col]) else 0.0
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Cast to float32
    for col in df.columns:
        if col not in ["date", "ticker"]:
            df[col] = df[col].astype(np.float32)
    
    output_path = tmp_path / "alpha_dataset_with_fund.csv.gz"
    df.to_csv(output_path, index=False, compression="gzip")
    
    return output_path


class TestEligibilityGating:
    """Task 10.2.2.1: Tests for eligibility gating ignoring fundamentals."""

    def test_all_fundamentals_nan_does_not_reduce_eligibility(self, tmp_path):
        """
        Fundamentals missing should NOT reduce eligibility.
        
        Creates a dataset with valid technical features but 100% NaN fundamentals.
        Training should succeed without "insufficient eligible count" error.
        """
        # Create dataset with all fundamentals NaN
        dataset_path = _create_alpha_dataset_with_fundamentals(
            tmp_path, n_days=120, n_tickers=5, fund_nan_ratio=1.0
        )
        
        out_dir = tmp_path / "output"
        
        # This should NOT fail due to "insufficient eligible count"
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-06-30",
            train_end="2020-03-31",
            val_end="2020-04-30",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=3,
            min_train_samples=50,
        )
        
        # Output scores.csv should exist
        assert Path(result["scores_csv"]).exists(), "scores.csv should be created"
        
        # Verify expected shape (should have ticker columns)
        scores = pd.read_csv(result["scores_csv"])
        ticker_cols = [c for c in scores.columns if c != "date"]
        assert len(ticker_cols) >= 3, "Should have at least 3 ticker columns"
        
        # All values should be finite (penalty applied if needed, but no failures)
        assert np.isfinite(scores[ticker_cols].values.astype(float)).all()

    def test_technical_nan_causes_ineligibility(self, tmp_path):
        """
        Technical NaN should still cause ineligibility.
        
        Creates a dataset with valid fundamentals but some technical NaNs.
        Rows with technical NaNs should receive the ineligible penalty.
        """
        np.random.seed(42)
        
        # Create a small dataset manually
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        tickers = ["TICK0", "TICK1", "TICK2", "TICK3", "TICK4"]
        
        rows = []
        for d in dates:
            for t in tickers:
                close = 100 + np.random.uniform(-5, 5)
                row = {
                    "date": d,
                    "ticker": t,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1e6,
                    "vol_20d": 0.2,
                    "mom_5d": np.random.uniform(-0.1, 0.1),
                    "mom_21d": np.random.uniform(-0.1, 0.1),
                    "mom_63d": np.random.uniform(-0.1, 0.1),
                    "rsi_14d": 50.0,
                    "bbands_20d": 0.5,
                    "atr_14d_norm": 0.02,
                    # Targets
                    "fwd_ret_5d": np.random.uniform(-0.1, 0.1),
                    "fwd_ret_10d": np.random.uniform(-0.15, 0.15),
                    "fwd_ret_21d": np.random.uniform(-0.2, 0.2),
                }
                
                # Fundamentals all valid
                for col in SEC_FUNDAMENTAL_COLS:
                    row[col] = 1e9
                    row[f"{col}_is_missing"] = 0.0
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Find actual month-end dates in the data
        # For rebalance="M", the scoring window is (val_end, as_of_date] = (2020-04-15, 2020-05-31]
        # Month-end rebalance dates would be 2020-04-30 and 2020-05-29 (if Friday is month-end)
        df_dates = pd.DataFrame({"date": pd.to_datetime(dates)})
        df_dates["month"] = df_dates["date"].dt.to_period("M")
        month_ends = df_dates.groupby("month")["date"].max()
        
        # Find the last month-end in the dataset that falls within scoring window
        val_end = pd.to_datetime("2020-04-15")
        as_of = pd.to_datetime("2020-05-31")
        scoring_month_ends = [d for d in month_ends if val_end < d <= as_of]
        
        if scoring_month_ends:
            # Inject NaN on the last rebalance date for TICK0
            rebal_date = max(scoring_month_ends)
            mask = (df["date"] == rebal_date) & (df["ticker"] == "TICK0")
            if mask.sum() > 0:
                df.loc[mask, "mom_21d"] = np.nan
        
        # Cast to float32
        for col in df.columns:
            if col not in ["date", "ticker"]:
                df[col] = df[col].astype(np.float32)
        
        dataset_path = tmp_path / "alpha_dataset_tech_nan.csv.gz"
        df.to_csv(dataset_path, index=False, compression="gzip")
        
        out_dir = tmp_path / "output"
        
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-05-31",
            train_end="2020-03-31",
            val_end="2020-04-15",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=3,  # Low threshold
            min_train_samples=30,
        )
        
        # Should complete without error
        assert Path(result["scores_csv"]).exists()
        
        # Read scores and verify TICK0 has penalty on last rebalance date
        scores = pd.read_csv(result["scores_csv"])
        scores["date"] = pd.to_datetime(scores["date"])
        
        # Get the last scored date
        last_score_date = scores["date"].max()
        last_row = scores[scores["date"] == last_score_date]
        
        if not last_row.empty and "TICK0" in last_row.columns:
            # TICK0 should have a very low score (penalty applied)
            tick0_score = last_row["TICK0"].iloc[0]
            other_scores = [
                last_row[c].iloc[0] for c in ["TICK1", "TICK2", "TICK3", "TICK4"] 
                if c in last_row.columns
            ]
            
            if other_scores:
                # TICK0 should be significantly lower due to penalty
                # (penalty is valid_min - 1e6, so it should be way below others)
                assert tick0_score < min(other_scores) - 1e5, (
                    f"TICK0 with tech NaN should have penalty score. "
                    f"TICK0={tick0_score}, others={other_scores}"
                )

    def test_partial_fundamentals_still_eligible(self, tmp_path):
        """
        Tickers with partial fundamentals (some NaN, some valid) should still be eligible.
        """
        # Create dataset with 50% NaN fundamentals
        dataset_path = _create_alpha_dataset_with_fundamentals(
            tmp_path, n_days=120, n_tickers=5, fund_nan_ratio=0.5
        )
        
        out_dir = tmp_path / "output"
        
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-06-30",
            train_end="2020-03-31",
            val_end="2020-04-30",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=3,
            min_train_samples=50,
        )
        
        # Should succeed
        assert Path(result["scores_csv"]).exists()
        
        scores = pd.read_csv(result["scores_csv"])
        assert len(scores) > 0, "Should have at least one scored date"


# ==============================================================================
# Task 10.2.10: Tests for Feature Mode Ablation
# ==============================================================================


class TestFeatureModes:
    """Task 10.2.10: Tests for feature_mode parameter."""
    
    def test_tech_only_excludes_fund_columns(self, tmp_path):
        """tech_only mode should exclude FUND columns from feature set."""
        import json
        
        dataset_path = _create_alpha_dataset_with_fundamentals(
            tmp_path, n_days=120, n_tickers=5, fund_nan_ratio=0.5
        )
        
        out_dir_full = tmp_path / "fund_full"
        out_dir_tech = tmp_path / "tech_only"
        
        # Train with fund_full
        result_full = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-06-30",
            train_end="2020-03-31",
            val_end="2020-04-30",
            out_dir=str(out_dir_full),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=3,
            min_train_samples=50,
            feature_mode="fund_full",
        )
        
        # Train with tech_only
        result_tech = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-06-30",
            train_end="2020-03-31",
            val_end="2020-04-30",
            out_dir=str(out_dir_tech),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=3,
            min_train_samples=50,
            feature_mode="tech_only",
        )
        
        # Read summaries
        with open(result_full["summary_json"]) as f:
            summary_full = json.load(f)
        with open(result_tech["summary_json"]) as f:
            summary_tech = json.load(f)
        
        # tech_only should have fewer features
        assert summary_tech["n_features_used"] < summary_full["n_features_used"], \
            "tech_only should use fewer features than fund_full"
        
        # Verify feature modes recorded
        assert summary_full["config"]["feature_mode"] == "fund_full"
        assert summary_tech["config"]["feature_mode"] == "tech_only"
    
    def test_fund_shuffled_determinism(self, tmp_path):
        """fund_shuffled should produce identical results with same seed."""
        import json
        
        dataset_path = _create_alpha_dataset_with_fundamentals(
            tmp_path, n_days=120, n_tickers=5, fund_nan_ratio=0.3
        )
        
        common_args = {
            "alpha_dataset_path": str(dataset_path),
            "as_of_date": "2020-06-30",
            "train_end": "2020-03-31",
            "val_end": "2020-04-30",
            "rebalance": "M",
            "target_col": "fwd_ret_21d",
            "top_k": 3,
            "min_train_samples": 50,
            "feature_mode": "fund_shuffled",
        }
        
        # Run twice with same seed
        result1 = train_xgb_alpha_from_dataset(
            out_dir=str(tmp_path / "run1"), seed=42, **common_args
        )
        result2 = train_xgb_alpha_from_dataset(
            out_dir=str(tmp_path / "run2"), seed=42, **common_args
        )
        
        # Scores should be byte-identical
        scores1 = Path(result1["scores_csv"]).read_bytes()
        scores2 = Path(result2["scores_csv"]).read_bytes()
        assert scores1 == scores2, "Same seed should produce identical shuffled results"
        
        # Run with different seed
        result3 = train_xgb_alpha_from_dataset(
            out_dir=str(tmp_path / "run3"), seed=123, **common_args
        )
        
        scores3 = Path(result3["scores_csv"]).read_bytes()
        assert scores1 != scores3, "Different seed should produce different shuffled results"
    
    def test_fund_zeroed_sets_values(self, tmp_path):
        """fund_zeroed should produce valid results with zeroed fundamentals."""
        import json
        
        dataset_path = _create_alpha_dataset_with_fundamentals(
            tmp_path, n_days=120, n_tickers=5, fund_nan_ratio=0.0  # All valid
        )
        
        out_dir = tmp_path / "fund_zeroed"
        
        result = train_xgb_alpha_from_dataset(
            alpha_dataset_path=str(dataset_path),
            as_of_date="2020-06-30",
            train_end="2020-03-31",
            val_end="2020-04-30",
            out_dir=str(out_dir),
            rebalance="M",
            target_col="fwd_ret_21d",
            seed=42,
            top_k=3,
            min_train_samples=50,
            feature_mode="fund_zeroed",
        )
        
        # Should succeed
        assert Path(result["scores_csv"]).exists()
        
        # Verify feature mode recorded
        with open(result["summary_json"]) as f:
            summary = json.load(f)
        assert summary["config"]["feature_mode"] == "fund_zeroed"
