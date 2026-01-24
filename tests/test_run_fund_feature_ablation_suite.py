"""
Tests for run_fund_feature_ablation_suite.py

Task 10.2.10: Tests for ablation suite orchestrator.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _create_synthetic_fund_alpha_dataset(tmp_path: Path, n_days: int = 120, n_tickers: int = 5) -> Path:
    """Create synthetic FUND alpha dataset for ablation testing."""
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"TICK{i}" for i in range(n_tickers)]
    
    # SEC fundamental columns
    sec_cols = ["assets", "liabilities", "equity", "cash", "shares_out"]
    
    rows = []
    for d in dates:
        for t in tickers:
            close = 100 + hash(t) % 50 + np.random.uniform(-5, 5)
            
            # For ablation test: fundamentals have a predictive relationship
            # target ~ 0.1 * (assets rank) + noise
            asset_val = 1e9 * (1 + hash(t) % 10 / 10) + np.random.uniform(-1e8, 1e8)
            
            row = {
                "date": d,
                "ticker": t,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 1e6 + np.random.uniform(0, 1e6),
                # Technical features - pure noise
                "vol_20d": 0.2 + np.random.uniform(0, 0.1),
                "mom_5d": np.random.uniform(-0.1, 0.1),
                "mom_21d": np.random.uniform(-0.2, 0.2),
                "mom_63d": np.random.uniform(-0.3, 0.3),
                "rsi_14d": 50 + np.random.uniform(-20, 20),
                "bbands_20d": np.random.uniform(-1, 2),
                "atr_14d_norm": 0.02 + np.random.uniform(0, 0.02),
            }
            
            # Add fundamentals (with correlation to target)
            row["assets"] = asset_val
            row["liabilities"] = asset_val * 0.5
            row["equity"] = asset_val * 0.5
            row["cash"] = asset_val * 0.1
            row["shares_out"] = 1e8
            
            # Add missing indicators
            for col in sec_cols:
                row[f"{col}_is_missing"] = 0.0
            
            # Target has weak correlation with assets (for directional test)
            noise = np.random.uniform(-0.15, 0.15)
            row["fwd_ret_5d"] = noise
            row["fwd_ret_10d"] = noise
            row["fwd_ret_21d"] = 0.05 * (asset_val / 1e10 - 0.5) + noise
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Cast to float32
    for col in df.columns:
        if col not in ["date", "ticker"]:
            df[col] = df[col].astype(np.float32)
    
    output_path = tmp_path / "fund_alpha_dataset.csv.gz"
    df.to_csv(output_path, index=False, compression="gzip")
    
    return output_path


def _create_synthetic_prices_csv(tmp_path: Path, n_days: int = 200, tickers: list[str] | None = None) -> Path:
    """Create synthetic prices CSV for backtest."""
    if tickers is None:
        tickers = [f"TICK{i}" for i in range(5)]
    
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    
    data = {"date": [d.strftime("%Y-%m-%d") for d in dates]}
    for ticker in tickers:
        base = 100 + hash(ticker) % 50
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices
    
    df = pd.DataFrame(data)
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return path


def _create_synthetic_scores_csv(tmp_path: Path, filename: str, n_dates: int = 3, tickers: list[str] | None = None) -> Path:
    """Create synthetic baseline scores CSV."""
    if tickers is None:
        tickers = [f"TICK{i}" for i in range(5)]
    
    np.random.seed(42)
    # Use quarter-end dates
    dates = pd.date_range("2020-03-31", periods=n_dates, freq="QE")
    
    data = {"date": [d.strftime("%Y-%m-%d") for d in dates]}
    for ticker in tickers:
        data[ticker] = np.random.uniform(0, 1, n_dates)
    
    df = pd.DataFrame(data)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


class TestRunFundFeatureAblationSuite:
    """Tests for ablation suite orchestrator."""
    
    def test_suite_generates_report(self, tmp_path):
        """Suite should generate REPORT_10_2_10.md and ablation_summary.json."""
        from scripts.run_fund_feature_ablation_suite import run_fund_feature_ablation_suite
        
        tickers = [f"TICK{i}" for i in range(5)]
        
        # Create larger datasets to ensure sufficient training data and rebalance dates
        fund_path = _create_synthetic_fund_alpha_dataset(tmp_path, n_days=250, n_tickers=5)
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=4, tickers=tickers)
        
        out_dir = tmp_path / "ablation_output"
        
        # Use date ranges that ensure month-end dates exist in scoring window
        # Training: 2020-01 to 2020-04, Val: 2020-05, Scoring: 2020-06+
        result = run_fund_feature_ablation_suite(
            fund_alpha_dataset_path=str(fund_path),
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            out_dir=str(out_dir),
            as_of_date="2020-10-31",
            train_end="2020-05-31",
            val_end="2020-06-30",
            rebalance="M",
            target_col="fwd_ret_21d",
            top_k=3,
            seed=42,
            min_train_samples=30,
        )
        
        # Report and summary should exist
        assert Path(result["report_path"]).exists()
        assert Path(result["summary_path"]).exists()
        
        # Report should be markdown
        report_content = Path(result["report_path"]).read_text()
        assert "# Task 10.2.10" in report_content
        assert "fund_full" in report_content
        assert "tech_only" in report_content
        
        # Summary should be valid JSON with correct schema
        with open(result["summary_path"]) as f:
            summary = json.load(f)
        assert summary["schema_version"] == "10.2.10"
        assert "fund_full" in summary["modes"]
        assert "tech_only" in summary["modes"]
        assert "fund_zeroed" in summary["modes"]
        assert "fund_shuffled" in summary["modes"]
    
    def test_suite_mode_directories(self, tmp_path):
        """Suite should create per-mode directories with model and ab subdirs."""
        from scripts.run_fund_feature_ablation_suite import run_fund_feature_ablation_suite
        
        tickers = [f"TICK{i}" for i in range(5)]
        
        # Use same date ranges as test_suite_generates_report
        fund_path = _create_synthetic_fund_alpha_dataset(tmp_path, n_days=250, n_tickers=5)
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=4, tickers=tickers)
        
        out_dir = tmp_path / "ablation_output"
        
        run_fund_feature_ablation_suite(
            fund_alpha_dataset_path=str(fund_path),
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            out_dir=str(out_dir),
            as_of_date="2020-10-31",
            train_end="2020-05-31",
            val_end="2020-06-30",
            rebalance="M",
            target_col="fwd_ret_21d",
            top_k=3,
            seed=42,
            min_train_samples=30,
        )
        
        # Check per-mode directories
        for mode in ["fund_full", "tech_only", "fund_zeroed", "fund_shuffled"]:
            mode_dir = out_dir / mode
            assert mode_dir.is_dir(), f"{mode} directory should exist"
            assert (mode_dir / "model").is_dir(), f"{mode}/model should exist"
            assert (mode_dir / "model" / "scores.csv").exists(), f"{mode}/model/scores.csv should exist"
            assert (mode_dir / "model" / "summary.json").exists(), f"{mode}/model/summary.json should exist"
