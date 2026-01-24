"""
Tests for src/ab_backtest.py

Covers:
- A/B artifact creation
- Determinism (byte-identical outputs)
- Mismatch handling (mismatched dates/tickers)
- Delta sign verification
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ab_backtest import run_ab_backtest_from_score_csvs


def _create_prices_csv(tmp_path: Path, n_days: int = 60) -> str:
    """Create a synthetic prices CSV (wide format)."""
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    tickers = ["A", "B", "C", "D", "E"]
    
    np.random.seed(42)
    data = {"date": dates}
    for ticker in tickers:
        # Random walk prices
        returns = np.random.randn(n_days) * 0.01
        prices = 100 * np.cumprod(1 + returns)
        data[ticker] = prices
    
    df = pd.DataFrame(data)
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return str(path)


def _create_scores_csv(
    tmp_path: Path,
    filename: str,
    n_dates: int = 12,
    tickers: list[str] = None,
    score_seed: int = 42,
) -> str:
    """Create a synthetic scores CSV."""
    if tickers is None:
        tickers = ["A", "B", "C", "D", "E"]
    
    # Monthly signal dates
    dates = pd.date_range("2023-01-31", periods=n_dates, freq="ME")
    
    np.random.seed(score_seed)
    data = {"date": dates}
    for ticker in tickers:
        data[ticker] = np.random.randn(n_dates)
    
    df = pd.DataFrame(data)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


class TestABBacktestCreatesArtifacts:
    """Test that A/B backtest creates expected output files."""
    
    def test_ab_backtest_creates_artifacts(self, tmp_path):
        """Test that A/B backtest creates all expected JSON artifacts."""
        prices_csv = _create_prices_csv(tmp_path)
        baseline_csv = _create_scores_csv(tmp_path, "baseline_scores.csv", score_seed=42)
        variant_csv = _create_scores_csv(tmp_path, "variant_scores.csv", score_seed=123)
        
        output_dir = tmp_path / "ab_output"
        
        result = run_ab_backtest_from_score_csvs(
            prices_csv_path=prices_csv,
            baseline_scores_csv_path=baseline_csv,
            variant_scores_csv_path=variant_csv,
            output_dir=str(output_dir),
            rebalance="M",
            method="topk",
            top_k=3,
        )
        
        # Check files exist
        assert (output_dir / "baseline_summary.json").exists()
        assert (output_dir / "variant_summary.json").exists()
        assert (output_dir / "delta_summary.json").exists()
        
        # Check JSON is parseable
        with open(output_dir / "baseline_summary.json") as f:
            baseline_json = json.load(f)
        with open(output_dir / "variant_summary.json") as f:
            variant_json = json.load(f)
        with open(output_dir / "delta_summary.json") as f:
            delta_json = json.load(f)
        
        # Check expected keys
        assert "metrics" in baseline_json
        assert "cagr_over_vol" in baseline_json["metrics"]
        
        assert "metrics" in variant_json
        assert "cagr_over_vol" in variant_json["metrics"]
        
        assert "delta" in delta_json
        assert "cagr_over_vol" in delta_json["delta"]
        
        # Check result dict has expected keys
        assert "baseline_metrics" in result
        assert "variant_metrics" in result
        assert "delta" in result
        assert "dates_used" in result
        assert "tickers_used" in result


class TestABBacktestDeterminism:
    """Test that A/B backtest is deterministic."""
    
    def test_ab_backtest_determinism(self, tmp_path):
        """Test that same inputs produce identical outputs."""
        prices_csv = _create_prices_csv(tmp_path)
        baseline_csv = _create_scores_csv(tmp_path, "baseline_scores.csv", score_seed=42)
        variant_csv = _create_scores_csv(tmp_path, "variant_scores.csv", score_seed=123)
        
        # Run 1
        output_dir_1 = tmp_path / "run1"
        result_1 = run_ab_backtest_from_score_csvs(
            prices_csv_path=prices_csv,
            baseline_scores_csv_path=baseline_csv,
            variant_scores_csv_path=variant_csv,
            output_dir=str(output_dir_1),
            rebalance="M",
            method="topk",
            top_k=3,
            seed=42,
        )
        
        # Run 2
        output_dir_2 = tmp_path / "run2"
        result_2 = run_ab_backtest_from_score_csvs(
            prices_csv_path=prices_csv,
            baseline_scores_csv_path=baseline_csv,
            variant_scores_csv_path=variant_csv,
            output_dir=str(output_dir_2),
            rebalance="M",
            method="topk",
            top_k=3,
            seed=42,
        )
        
        # Check JSON content is identical
        with open(output_dir_1 / "delta_summary.json") as f:
            content_1 = f.read()
        with open(output_dir_2 / "delta_summary.json") as f:
            content_2 = f.read()
        
        assert content_1 == content_2, "A/B outputs not deterministic"
        
        # Check result dicts are equal
        assert result_1["delta"] == result_2["delta"]


class TestABBacktestMismatchHandling:
    """Test that A/B backtest handles mismatches correctly."""
    
    def test_ab_backtest_mismatch_dates(self, tmp_path):
        """Test that mismatched dates raise clear error."""
        prices_csv = _create_prices_csv(tmp_path, n_days=120)
        
        # Create baseline with one date range
        dates_baseline = pd.date_range("2023-01-31", periods=6, freq="ME")
        df_baseline = pd.DataFrame({
            "date": dates_baseline,
            "A": np.random.randn(6),
            "B": np.random.randn(6),
        })
        baseline_csv = tmp_path / "baseline.csv"
        df_baseline.to_csv(baseline_csv, index=False)
        
        # Create variant with completely different date range
        dates_variant = pd.date_range("2024-01-31", periods=6, freq="ME")
        df_variant = pd.DataFrame({
            "date": dates_variant,
            "A": np.random.randn(6),
            "B": np.random.randn(6),
        })
        variant_csv = tmp_path / "variant.csv"
        df_variant.to_csv(variant_csv, index=False)
        
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="No common dates"):
            run_ab_backtest_from_score_csvs(
                prices_csv_path=prices_csv,
                baseline_scores_csv_path=str(baseline_csv),
                variant_scores_csv_path=str(variant_csv),
                output_dir=str(output_dir),
            )
    
    def test_ab_backtest_mismatch_tickers(self, tmp_path):
        """Test that mismatched tickers raise clear error."""
        prices_csv = _create_prices_csv(tmp_path)
        
        # Create baseline with ticker set A
        dates = pd.date_range("2023-01-31", periods=6, freq="ME")
        df_baseline = pd.DataFrame({
            "date": dates,
            "AAA": np.random.randn(6),
            "BBB": np.random.randn(6),
        })
        baseline_csv = tmp_path / "baseline.csv"
        df_baseline.to_csv(baseline_csv, index=False)
        
        # Create variant with completely different tickers
        df_variant = pd.DataFrame({
            "date": dates,
            "XXX": np.random.randn(6),
            "YYY": np.random.randn(6),
        })
        variant_csv = tmp_path / "variant.csv"
        df_variant.to_csv(variant_csv, index=False)
        
        output_dir = tmp_path / "output"
        
        with pytest.raises(ValueError, match="No common tickers"):
            run_ab_backtest_from_score_csvs(
                prices_csv_path=prices_csv,
                baseline_scores_csv_path=str(baseline_csv),
                variant_scores_csv_path=str(variant_csv),
                output_dir=str(output_dir),
            )


class TestABBacktestDeltaSign:
    """Test that delta captures variant improvement correctly."""
    
    def test_ab_backtest_delta_sign(self, tmp_path):
        """Test that better variant produces positive delta."""
        # Create prices where A has positive returns, others negative
        n_days = 60
        dates = pd.bdate_range("2023-01-02", periods=n_days)
        
        data = {"date": dates}
        # A goes up steadily
        data["A"] = 100 * (1 + 0.01) ** np.arange(n_days)
        # B, C, D, E go down
        for ticker in ["B", "C", "D", "E"]:
            data[ticker] = 100 * (1 - 0.005) ** np.arange(n_days)
        
        prices_df = pd.DataFrame(data)
        prices_csv = tmp_path / "prices.csv"
        prices_df.to_csv(prices_csv, index=False)
        
        # Create score dates
        signal_dates = pd.date_range("2023-01-31", periods=4, freq="ME")
        
        # Baseline: uniform scores (equal weight all)
        baseline_df = pd.DataFrame({
            "date": signal_dates,
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [1.0, 1.0, 1.0, 1.0],
            "C": [1.0, 1.0, 1.0, 1.0],
            "D": [1.0, 1.0, 1.0, 1.0],
            "E": [1.0, 1.0, 1.0, 1.0],
        })
        baseline_csv = tmp_path / "baseline.csv"
        baseline_df.to_csv(baseline_csv, index=False)
        
        # Variant: heavily weights A (the winner)
        variant_df = pd.DataFrame({
            "date": signal_dates,
            "A": [10.0, 10.0, 10.0, 10.0],  # Very high score for winner
            "B": [1.0, 1.0, 1.0, 1.0],
            "C": [1.0, 1.0, 1.0, 1.0],
            "D": [1.0, 1.0, 1.0, 1.0],
            "E": [1.0, 1.0, 1.0, 1.0],
        })
        variant_csv = tmp_path / "variant.csv"
        variant_df.to_csv(variant_csv, index=False)
        
        output_dir = tmp_path / "output"
        
        result = run_ab_backtest_from_score_csvs(
            prices_csv_path=str(prices_csv),
            baseline_scores_csv_path=str(baseline_csv),
            variant_scores_csv_path=str(variant_csv),
            output_dir=str(output_dir),
            rebalance="M",
            method="topk",
            top_k=3,  # Will pick top 3
        )
        
        # Variant should outperform baseline (positive delta)
        # because variant weights A more, which has positive returns
        assert result["variant_metrics"]["cagr_over_vol"] >= result["baseline_metrics"]["cagr_over_vol"], (
            f"Expected variant >= baseline, got "
            f"variant={result['variant_metrics']['cagr_over_vol']:.4f}, "
            f"baseline={result['baseline_metrics']['cagr_over_vol']:.4f}"
        )


class TestWriteScoresPanelCSV:
    """Test _write_scores_panel_csv helper (Task 9.3.2.1)."""
    
    def test_write_scores_panel_csv_has_date_column(self, tmp_path):
        """Test that the helper writes a CSV with explicit 'date' column."""
        from src.ab_backtest import _write_scores_panel_csv
        
        # Create a tiny scores DataFrame with DatetimeIndex
        dates = pd.date_range("2023-01-31", periods=3, freq="ME")
        scores = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
            index=dates,
        )
        
        csv_path = tmp_path / "scores.csv"
        _write_scores_panel_csv(scores, str(csv_path))
        
        # Read first line
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()
        
        # Should start with "date," and include tickers
        assert first_line.startswith("date,"), f"First column is not 'date': {first_line}"
        assert "A" in first_line
        assert "B" in first_line
        
        # Also verify the full CSV is loadable correctly
        loaded = pd.read_csv(csv_path)
        assert "date" in loaded.columns
        assert list(loaded.columns) == ["date", "A", "B"]


class TestTempCleanupOnException:
    """Test temp file cleanup on exception (Task 9.3.2.1)."""
    
    def test_temp_cleanup_on_exception(self, tmp_path, monkeypatch):
        """Test that temp files are cleaned up even when backtest raises."""
        import os
        from src import ab_backtest
        
        prices_csv = _create_prices_csv(tmp_path)
        baseline_csv = _create_scores_csv(tmp_path, "baseline.csv", score_seed=42)
        variant_csv = _create_scores_csv(tmp_path, "variant.csv", score_seed=123)
        output_dir = tmp_path / "output"
        
        # Track temp file paths
        temp_baseline = output_dir / "_temp_baseline_scores.csv"
        temp_variant = output_dir / "_temp_variant_scores.csv"
        
        # Monkeypatch run_scores_backtest_from_csv to raise after temp files are written
        def mock_backtest(*args, **kwargs):
            # At this point temp files should exist
            raise RuntimeError("Simulated backtest failure")
        
        monkeypatch.setattr(
            ab_backtest,
            "run_scores_backtest_from_csv",
            mock_backtest,
        )
        
        # Call should raise
        with pytest.raises(RuntimeError, match="Simulated backtest failure"):
            run_ab_backtest_from_score_csvs(
                prices_csv_path=prices_csv,
                baseline_scores_csv_path=baseline_csv,
                variant_scores_csv_path=variant_csv,
                output_dir=str(output_dir),
                rebalance="M",
                method="topk",
                top_k=3,
            )
        
        # Temp files should be cleaned up
        assert not temp_baseline.exists(), "Temp baseline CSV not cleaned up"
        assert not temp_variant.exists(), "Temp variant CSV not cleaned up"

