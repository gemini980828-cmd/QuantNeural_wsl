"""
Tests for src/run_generate_scores_cli.py

Covers:
- Successful run creates output CSV
- Determinism: identical output for identical input
- Fail-fast: missing required args, missing file, invalid choice
"""

import numpy as np
import pandas as pd
import pytest

from src.run_generate_scores_cli import main


def _create_test_prices_csv(tmp_path, n_days=400, tickers=None):
    """Create a valid prices CSV for testing."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    
    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")
    
    np.random.seed(42)
    data = {"date": dates}
    for i, ticker in enumerate(tickers):
        base = 100 + i * 10
        returns = np.random.randn(n_days) * 0.01 + 0.001
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices
    
    df = pd.DataFrame(data)
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    
    return str(path)


class TestCLISuccessfulRun:
    """Test CLI runs and creates output."""
    
    def test_creates_output_csv_with_expected_columns(self, tmp_path):
        """Test CLI creates output CSV with date and ticker columns."""
        prices_path = _create_test_prices_csv(tmp_path)
        out_path = tmp_path / "scores.csv"
        
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--out_scores_csv_path", str(out_path),
            "--rebalance", "M",
            "--lookback_days", "252",
            "--no_leading_plateau_gate",
        ])
        
        assert exit_code == 0
        assert out_path.exists()
        
        df = pd.read_csv(out_path)
        assert "date" in df.columns
        assert "AAA" in df.columns
        assert "BBB" in df.columns
        assert "CCC" in df.columns
        assert len(df) > 0
    
    def test_quarterly_rebalance(self, tmp_path):
        """Test quarterly rebalance option."""
        prices_path = _create_test_prices_csv(tmp_path, n_days=600)
        out_path = tmp_path / "scores_q.csv"
        
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--out_scores_csv_path", str(out_path),
            "--rebalance", "Q",
            "--no_leading_plateau_gate",
        ])
        
        assert exit_code == 0
        assert out_path.exists()


class TestCLIDeterminism:
    """Test determinism of CLI output."""
    
    def test_same_input_yields_identical_output(self, tmp_path):
        """Test two runs with same inputs produce identical files."""
        prices_path = _create_test_prices_csv(tmp_path)
        out_path1 = tmp_path / "scores1.csv"
        out_path2 = tmp_path / "scores2.csv"
        
        main([
            "--prices_csv_path", prices_path,
            "--out_scores_csv_path", str(out_path1),
            "--no_leading_plateau_gate",
        ])
        
        main([
            "--prices_csv_path", prices_path,
            "--out_scores_csv_path", str(out_path2),
            "--no_leading_plateau_gate",
        ])
        
        content1 = out_path1.read_text()
        content2 = out_path2.read_text()
        
        assert content1 == content2


class TestCLIFailFast:
    """Test fail-fast behavior."""
    
    def test_missing_required_arg_raises(self, tmp_path):
        """Test missing required arg raises SystemExit."""
        with pytest.raises(SystemExit):
            main([
                "--out_scores_csv_path", str(tmp_path / "scores.csv"),
                # Missing --prices_csv_path
            ])
    
    def test_nonexistent_file_raises(self, tmp_path):
        """Test nonexistent prices file raises exception."""
        with pytest.raises(Exception):  # FileNotFoundError
            main([
                "--prices_csv_path", str(tmp_path / "nonexistent.csv"),
                "--out_scores_csv_path", str(tmp_path / "scores.csv"),
            ])
    
    def test_invalid_rebalance_choice_raises(self, tmp_path):
        """Test invalid rebalance choice raises SystemExit."""
        prices_path = _create_test_prices_csv(tmp_path)
        
        with pytest.raises(SystemExit):
            main([
                "--prices_csv_path", prices_path,
                "--out_scores_csv_path", str(tmp_path / "scores.csv"),
                "--rebalance", "W",  # Invalid
            ])


class TestCLICustomOptions:
    """Test CLI with custom options."""
    
    def test_custom_lookback_days(self, tmp_path):
        """Test custom lookback_days option."""
        prices_path = _create_test_prices_csv(tmp_path, n_days=200)
        out_path = tmp_path / "scores.csv"
        
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--out_scores_csv_path", str(out_path),
            "--lookback_days", "60",
            "--no_leading_plateau_gate",
        ])
        
        assert exit_code == 0
        
        df = pd.read_csv(out_path)
        assert len(df) > 0
    
    def test_min_coverage_option(self, tmp_path):
        """Test min_coverage option."""
        prices_path = _create_test_prices_csv(tmp_path)
        out_path = tmp_path / "scores.csv"
        
        exit_code = main([
            "--prices_csv_path", prices_path,
            "--out_scores_csv_path", str(out_path),
            "--min_coverage", "0.8",
            "--no_leading_plateau_gate",
        ])
        
        assert exit_code == 0
