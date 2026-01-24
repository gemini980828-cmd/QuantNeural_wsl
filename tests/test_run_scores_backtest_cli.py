"""
Tests for src/run_scores_backtest_cli.py

Covers:
- CLI runs successfully and creates output artifacts
- Determinism: running twice produces identical outputs
- Fail-fast: missing required args or files errors out
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.run_scores_backtest_cli import main


def _create_test_csvs(tmp_path, n_days=60, tickers=None, n_signals=6, n_tickers=500):
    """Create valid prices and scores CSV files for testing.
    
    n_tickers: number of tickers to generate (default 500 to allow top_k=400).
    """
    if tickers is None:
        # Generate enough tickers for top_k=400 default
        tickers = [f"T{i:04d}" for i in range(n_tickers)]
    
    # Wide prices
    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    np.random.seed(42)
    prices_data = {"date": dates}
    for i, ticker in enumerate(tickers):
        base = 100 + (i % 100)
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        prices_data[ticker] = prices
    
    prices_df = pd.DataFrame(prices_data)
    prices_path = tmp_path / "prices.csv"
    prices_df.to_csv(prices_path, index=False)
    
    # Scores
    signal_dates = list(dates[::10][:n_signals])
    np.random.seed(123)
    scores_data = {"date": signal_dates}
    for ticker in tickers:
        scores_data[ticker] = np.random.randn(len(signal_dates))
    
    scores_df = pd.DataFrame(scores_data)
    scores_path = tmp_path / "scores.csv"
    scores_df.to_csv(scores_path, index=False)
    
    return str(prices_path), str(scores_path)


class TestCLISuccessfulRun:
    """Test CLI runs and creates output artifacts."""
    
    def test_cli_creates_all_artifacts(self, tmp_path):
        """Test CLI creates all expected output files."""
        prices_path, scores_path = _create_test_csvs(tmp_path)
        out_dir = tmp_path / "output"
        
        # Run CLI
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir),
            "--rebalance", "M",
        ])
        
        assert exit_code == 0
        
        # Verify all expected files exist
        expected_files = [
            "equity_curve.csv",
            "trades.csv",
            "summary_metrics.json",
            "target_weights.csv",
            "weights_used.csv",
            "turnover.csv",
            "costs.csv",
            "returns.csv",
        ]
        
        for filename in expected_files:
            filepath = out_dir / filename
            assert filepath.exists(), f"Missing: {filename}"
        
        # Verify summary_metrics.json is valid JSON
        with open(out_dir / "summary_metrics.json") as f:
            summary = json.load(f)
        
        assert "metrics" in summary
        assert "params" in summary
        assert "cagr" in summary["metrics"]
        assert "sharpe" in summary["metrics"]
    
    def test_cli_with_method_rank(self, tmp_path):
        """Test CLI with rank method."""
        prices_path, scores_path = _create_test_csvs(tmp_path)
        out_dir = tmp_path / "output"
        
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir),
            "--method", "rank",
        ])
        
        assert exit_code == 0
        assert (out_dir / "equity_curve.csv").exists()
    
    def test_cli_with_topk_method(self, tmp_path):
        """Test CLI with topk method and --top_k arg."""
        prices_path, scores_path = _create_test_csvs(tmp_path)
        out_dir = tmp_path / "output"
        
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir),
            "--method", "topk",
            "--top_k", "2",
        ])
        
        assert exit_code == 0
        assert (out_dir / "equity_curve.csv").exists()
    
    def test_cli_with_softmax_topk_method(self, tmp_path):
        """Test CLI with softmax_topk method."""
        prices_path, scores_path = _create_test_csvs(tmp_path, n_tickers=50)
        out_dir = tmp_path / "output"
        
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir),
            "--method", "softmax_topk",
            "--top_k", "10",
            "--score_transform", "winsorize_zscore",
        ])
        
        assert exit_code == 0
        assert (out_dir / "equity_curve.csv").exists()
    
    def test_cli_softmax_with_score_transform_reduces_concentration(self, tmp_path):
        """Softmax + winsorize_zscore should reduce extreme single-name concentration."""
        prices_path, scores_path = _create_test_csvs(tmp_path, n_tickers=200)
        
        # Inject a single extreme outlier score for one ticker on the first signal date.
        df_scores = pd.read_csv(scores_path)
        ticker_cols = [c for c in df_scores.columns if c != "date"]
        df_scores.loc[0, ticker_cols[0]] = 100.0
        df_scores.to_csv(scores_path, index=False)
        
        out_plain = tmp_path / "out_plain"
        out_tx = tmp_path / "out_tx"
        
        exit_code_plain = main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_plain),
            "--rebalance", "M",
            "--method", "softmax",
            "--temperature", "1.0",
        ])
        assert exit_code_plain == 0
        
        exit_code_tx = main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_tx),
            "--rebalance", "M",
            "--method", "softmax",
            "--temperature", "1.0",
            "--score_transform", "winsorize_zscore",
        ])
        assert exit_code_tx == 0
        
        tw_plain = pd.read_csv(out_plain / "target_weights.csv")
        tw_tx = pd.read_csv(out_tx / "target_weights.csv")
        
        max_plain = float(tw_plain.drop(columns=["date"]).to_numpy().max())
        max_tx = float(tw_tx.drop(columns=["date"]).to_numpy().max())
        
        assert max_plain > 0.90
        assert max_tx < 0.90


class TestCLIDeterminism:
    """Test determinism of CLI output."""
    
    def test_running_twice_produces_identical_output(self, tmp_path):
        """Test two runs with same inputs produce identical files."""
        prices_path, scores_path = _create_test_csvs(tmp_path)
        out_dir1 = tmp_path / "output1"
        out_dir2 = tmp_path / "output2"
        
        # Run twice
        main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir1),
        ])
        
        main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir2),
        ])
        
        # Compare key files
        files_to_compare = [
            "equity_curve.csv",
            "trades.csv",
            "target_weights.csv",
        ]
        
        for filename in files_to_compare:
            content1 = (out_dir1 / filename).read_text()
            content2 = (out_dir2 / filename).read_text()
            assert content1 == content2, f"File {filename} differs between runs"


class TestCLIFailFast:
    """Test fail-fast behavior."""
    
    def test_missing_required_arg_raises(self, tmp_path):
        """Test missing required arg raises SystemExit."""
        with pytest.raises(SystemExit):
            main([
                "--scores_csv_path", "scores.csv",
                "--out_dir", str(tmp_path / "output"),
                # Missing --prices_csv_path
            ])
    
    def test_nonexistent_prices_file_raises(self, tmp_path):
        """Test nonexistent prices file raises exception."""
        _, scores_path = _create_test_csvs(tmp_path)
        
        with pytest.raises(Exception):  # FileNotFoundError or similar
            main([
                "--prices_csv_path", str(tmp_path / "nonexistent.csv"),
                "--scores_csv_path", scores_path,
                "--out_dir", str(tmp_path / "output"),
            ])
    
    def test_invalid_rebalance_choice_raises(self, tmp_path):
        """Test invalid rebalance choice raises SystemExit."""
        prices_path, scores_path = _create_test_csvs(tmp_path)
        
        with pytest.raises(SystemExit):
            main([
                "--prices_csv_path", prices_path,
                "--scores_csv_path", scores_path,
                "--out_dir", str(tmp_path / "output"),
                "--rebalance", "W",  # Invalid choice
            ])


class TestCLICSVReadability:
    """Test output CSVs are readable and have expected structure."""
    
    def test_equity_curve_csv_has_date_column(self, tmp_path):
        """Test equity_curve.csv has date column."""
        prices_path, scores_path = _create_test_csvs(tmp_path)
        out_dir = tmp_path / "output"
        
        main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir),
        ])
        
        df = pd.read_csv(out_dir / "equity_curve.csv")
        assert "date" in df.columns
        assert "equity" in df.columns
        assert len(df) > 0
    
    def test_trades_csv_readable(self, tmp_path):
        """Test trades.csv is readable."""
        prices_path, scores_path = _create_test_csvs(tmp_path)
        out_dir = tmp_path / "output"
        
        main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir),
        ])
        
        df = pd.read_csv(out_dir / "trades.csv")
        assert "date" in df.columns
        assert "ticker" in df.columns


class TestCLISafeDefaults:
    """Test safe CLI defaults (Task 9.0.4).
    
    Verifies that CLI defaults to production-safe settings:
    - method=topk (not softmax which can fail catastrophically)
    - rebalance=Q (lower cost than M)
    - top_k=400 (locked baseline)
    - cost_bps=10, slippage_bps=5
    """
    
    def test_minimal_args_uses_safe_defaults(self, tmp_path):
        """Test CLI with only required args uses safe defaults.
        
        Verifies:
        - method == "topk" (not softmax)
        - rebalance == "Q" (not M)
        - top_k == 400
        """
        # Create enough tickers for top_k=400
        prices_path, scores_path = _create_test_csvs(tmp_path, n_tickers=500)
        out_dir = tmp_path / "output"
        
        # Run with ONLY required args
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", str(out_dir),
        ])
        
        assert exit_code == 0
        
        # Verify defaults in summary_metrics.json
        with open(out_dir / "summary_metrics.json") as f:
            summary = json.load(f)
        
        params = summary["params"]
        
        # Safe defaults assertions
        assert params["method"] == "topk", f"Expected topk, got {params['method']}"
        assert params["rebalance"] == "Q", f"Expected Q, got {params['rebalance']}"
        assert params["top_k"] == 400, f"Expected 400, got {params['top_k']}"
        assert params["cost_bps"] == 10.0, f"Expected 10.0, got {params['cost_bps']}"
        assert params["slippage_bps"] == 5.0, f"Expected 5.0, got {params['slippage_bps']}"
        
        # Verify cagr_over_vol metric exists (Task 9.2.4)
        metrics = summary["metrics"]
        assert "cagr_over_vol" in metrics, "cagr_over_vol metric missing from CLI output"
        assert "sharpe" in metrics, "sharpe metric missing (backward compat)"
        assert metrics["cagr_over_vol"] == metrics["sharpe"], "cagr_over_vol != sharpe"
        
        # Verify warnings field exists
        assert "warnings" in summary, "warnings field missing from summary_metrics.json"
        
        # Verify numeric output sanity
        equity_curve = pd.read_csv(out_dir / "equity_curve.csv")
        assert np.all(np.isfinite(equity_curve["equity"].values)), "Equity contains NaN/inf"
        assert len(equity_curve) > 0


