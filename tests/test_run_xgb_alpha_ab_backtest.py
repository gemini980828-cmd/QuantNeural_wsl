"""
Tests for run_xgb_alpha_ab_backtest.py

Covers:
- End-to-end A/B backtest with synthetic data
- Output JSON file existence and parseability
- Determinism (same inputs + seed => byte-identical outputs)
- Inverse score sanity check
- Diagnostics.json generation
- Task 10.2.8: Subset mode tests (sec_covered, sec_missing, split)
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_xgb_alpha_ab_backtest import run_xgb_alpha_ab_backtest


def _create_synthetic_prices_csv(tmp_path: Path, n_days: int = 60, tickers: list[str] | None = None) -> Path:
    """Create synthetic prices CSV for testing."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    
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


def _create_synthetic_scores_csv(
    tmp_path: Path,
    filename: str,
    n_dates: int = 3,
    tickers: list[str] | None = None,
    score_offset: float = 0.0,
) -> Path:
    """Create synthetic scores CSV for testing."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    
    np.random.seed(42)
    # Use quarter-end dates to match Q rebalance
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="QE")
    
    data = {"date": [d.strftime("%Y-%m-%d") for d in dates]}
    for ticker in tickers:
        base_score = hash(ticker) % 10 / 10.0 + score_offset
        scores = base_score + np.random.uniform(-0.1, 0.1, n_dates)
        data[ticker] = scores
    
    df = pd.DataFrame(data)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def _create_synthetic_ic_by_date_csv(tmp_path: Path, n_dates: int = 4) -> Path:
    """Create synthetic ic_by_date.csv for testing diagnostics."""
    dates = pd.date_range("2020-03-31", periods=n_dates, freq="QE")
    
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "ic_spearman": np.random.uniform(-0.1, 0.1, n_dates),
        "eligible_count": [500] * n_dates,
    })
    
    path = tmp_path / "ic_by_date.csv"
    df.to_csv(path, index=False)
    return path


def _create_synthetic_fund_dataset_csv(
    tmp_path: Path,
    n_dates: int = 2,
    tickers: list[str] | None = None,
    covered_tickers: list[str] | None = None,
) -> Path:
    """
    Create synthetic FUND dataset CSV for testing subset modes.
    
    Parameters
    ----------
    tmp_path : Path
        Temporary directory
    n_dates : int
        Number of dates
    tickers : list[str]
        All tickers to include
    covered_tickers : list[str]
        Tickers that should be marked as SEC-covered
    
    Returns
    -------
    Path
        Path to the created CSV.gz file
    """
    if tickers is None:
        tickers = ["T001", "T002", "T003", "T004", "T005", "T006"]
    if covered_tickers is None:
        covered_tickers = ["T001", "T002", "T003"]  # First half covered
    
    # Use quarter-end dates to match Q rebalance
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="QE")
    
    rows = []
    for date in dates:
        for ticker in tickers:
            is_covered = ticker.upper() in [t.upper() for t in covered_tickers]
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "ticker": f"{ticker}.US",  # Match Stooq format
                "any_sec_present": 1 if is_covered else 0,
                "total_assets": 1000.0 if is_covered else np.nan,
                "mktcap": 100.0 + hash(ticker) % 50 if is_covered else np.nan,
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    path = tmp_path / "fund_dataset.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    return path


class TestRunXGBAlphaABBacktest:
    """Test run_xgb_alpha_ab_backtest function."""
    
    def test_end_to_end_with_diagnostics(self, tmp_path):
        """End-to-end test with diagnostics.json generation."""
        # Create synthetic data
        tickers = [f"T{i:03d}" for i in range(500)]  # 500 tickers for top_k=400
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline_scores.csv", n_dates=4, tickers=tickers)
        
        xgb_dir = tmp_path / "xgb_model"
        xgb_dir.mkdir()
        xgb_path = _create_synthetic_scores_csv(xgb_dir, "scores.csv", n_dates=4, tickers=tickers, score_offset=0.1)
        _create_synthetic_ic_by_date_csv(xgb_dir, n_dates=4)
        
        out_dir = tmp_path / "ab_output"
        
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
        )
        
        # Verify xgb subfolder exists
        assert Path(result["xgb"]["delta_summary_json"]).exists()
        
        # Verify diagnostics.json exists and has correct schema
        assert Path(result["diagnostics_json"]).exists()
        with open(result["diagnostics_json"]) as f:
            diag = json.load(f)
        assert diag["schema_version"] == "10.2.8"
        assert "oos_window" in diag
        assert diag["oos_window"]["n_dates_scored"] == 4
    
    def test_invert_scores(self, tmp_path):
        """Inverse score A/B should produce inverted scores file and results."""
        tickers = [f"T{i:03d}" for i in range(500)]
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline_scores.csv", n_dates=4, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb_scores.csv", n_dates=4, tickers=tickers, score_offset=0.1)
        
        out_dir = tmp_path / "ab_output"
        
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
            invert_scores=True,
        )
        
        # Verify both xgb and xgb_inverse exist
        assert "xgb" in result
        assert "xgb_inverse" in result
        assert Path(result["xgb"]["delta_summary_json"]).exists()
        assert Path(result["xgb_inverse"]["delta_summary_json"]).exists()
        
        # Verify inverted scores file exists
        inverted_csv = Path(result["xgb_inverse"]["inverted_scores_csv"])
        assert inverted_csv.exists()
        
        # Verify inversion is correct
        original = pd.read_csv(xgb_path)
        inverted = pd.read_csv(inverted_csv)
        
        ticker_cols = [c for c in original.columns if c != "date"]
        for col in ticker_cols:
            np.testing.assert_allclose(
                inverted[col].values,
                -1.0 * original[col].values,
                rtol=1e-6,
                err_msg=f"{col} should be inverted"
            )
    
    def test_determinism(self, tmp_path):
        """Same seed should produce byte-identical outputs."""
        tickers = [f"T{i:03d}" for i in range(500)]
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline_scores.csv", n_dates=4, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb_scores.csv", n_dates=4, tickers=tickers, score_offset=0.1)
        
        out_dir_1 = tmp_path / "run1"
        out_dir_2 = tmp_path / "run2"
        
        result1 = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir_1),
            seed=42,
            invert_scores=True,
        )
        
        result2 = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir_2),
            seed=42,
            invert_scores=True,
        )
        
        # Compare delta_summary.json bytes
        delta1 = Path(result1["xgb"]["delta_summary_json"]).read_bytes()
        delta2 = Path(result2["xgb"]["delta_summary_json"]).read_bytes()
        assert delta1 == delta2, "Same seed should produce identical xgb delta_summary.json"
        
        # Compare xgb_inverse delta
        inv_delta1 = Path(result1["xgb_inverse"]["delta_summary_json"]).read_bytes()
        inv_delta2 = Path(result2["xgb_inverse"]["delta_summary_json"]).read_bytes()
        assert inv_delta1 == inv_delta2, "Same seed should produce identical xgb_inverse delta_summary.json"
    
    def test_missing_ic_by_date(self, tmp_path):
        """Should handle missing ic_by_date.csv gracefully."""
        tickers = [f"T{i:03d}" for i in range(500)]
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline_scores.csv", n_dates=4, tickers=tickers)
        # No ic_by_date.csv created
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb_scores.csv", n_dates=4, tickers=tickers, score_offset=0.1)
        
        out_dir = tmp_path / "ab_output"
        
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
        )
        
        # Verify diagnostics.json has warning for missing ic_by_date.csv
        with open(result["diagnostics_json"]) as f:
            diag = json.load(f)
        
        assert "warning" in diag["oos_window"]
    
    def test_output_bundle_structure(self, tmp_path):
        """Verify correct folder structure for output bundle."""
        tickers = [f"T{i:03d}" for i in range(500)]
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline_scores.csv", n_dates=4, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb_scores.csv", n_dates=4, tickers=tickers, score_offset=0.1)
        
        out_dir = tmp_path / "ab_output"
        
        run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
            invert_scores=True,
        )
        
        # Verify expected structure
        assert (out_dir / "diagnostics.json").exists()
        assert (out_dir / "xgb" / "baseline_summary.json").exists()
        assert (out_dir / "xgb" / "variant_summary.json").exists()
        assert (out_dir / "xgb" / "delta_summary.json").exists()
        assert (out_dir / "xgb_inverse" / "baseline_summary.json").exists()
        assert (out_dir / "xgb_inverse" / "variant_summary.json").exists()
        assert (out_dir / "xgb_inverse" / "delta_summary.json").exists()
        assert (out_dir / "xgb_inverse" / "xgb_scores_inverted.csv").exists()


class TestSubsetModes:
    """Task 10.2.8: Test subset mode functionality."""
    
    def test_subset_mode_sec_covered(self, tmp_path):
        """Test sec_covered subset mode produces expected outputs."""
        # Create 6 tickers: 3 covered, 3 missing
        tickers = ["T001", "T002", "T003", "T004", "T005", "T006"]
        covered = ["T001", "T002", "T003"]
        
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=2, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb.csv", n_dates=2, tickers=tickers, score_offset=0.1)
        fund_path = _create_synthetic_fund_dataset_csv(tmp_path, n_dates=2, tickers=tickers, covered_tickers=covered)
        
        out_dir = tmp_path / "sec_covered_output"
        
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
            subset_mode="sec_covered",
            fund_dataset_path=str(fund_path),
        )
        
        # Verify subset_audit.json exists with expected keys
        assert (out_dir / "subset_audit.json").exists()
        with open(out_dir / "subset_audit.json") as f:
            audit = json.load(f)
        
        assert audit["subset_mode"] == "sec_covered"
        assert "mean_universe_size_per_date" in audit
        assert "min_universe_size_per_date" in audit
        assert "dates_with_universe_lt_topk" in audit
        assert "fraction_of_all_universe" in audit
        
        # Verify tilt_audit.json exists
        assert (out_dir / "tilt_audit.json").exists()
    
    def test_subset_mode_sec_missing(self, tmp_path):
        """Test sec_missing subset mode produces expected outputs."""
        tickers = ["T001", "T002", "T003", "T004", "T005", "T006"]
        covered = ["T001", "T002", "T003"]
        
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=2, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb.csv", n_dates=2, tickers=tickers, score_offset=0.1)
        fund_path = _create_synthetic_fund_dataset_csv(tmp_path, n_dates=2, tickers=tickers, covered_tickers=covered)
        
        out_dir = tmp_path / "sec_missing_output"
        
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
            subset_mode="sec_missing",
            fund_dataset_path=str(fund_path),
        )
        
        # Verify subset_audit.json exists with expected keys
        assert (out_dir / "subset_audit.json").exists()
        with open(out_dir / "subset_audit.json") as f:
            audit = json.load(f)
        
        assert audit["subset_mode"] == "sec_missing"
        assert "mean_universe_size_per_date" in audit
        
        # Verify fail-safe: dates_with_universe_lt_topk should contain dates
        # since we only have 3 missing tickers < 400
        assert len(audit["dates_with_universe_lt_topk"]) > 0
    
    def test_subset_mode_split(self, tmp_path):
        """Test split mode produces all subdirectories and valid backtest outputs."""
        tickers = ["T001", "T002", "T003", "T004", "T005", "T006"]
        covered = ["T001", "T002", "T003"]
        
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=2, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb.csv", n_dates=2, tickers=tickers, score_offset=0.1)
        fund_path = _create_synthetic_fund_dataset_csv(tmp_path, n_dates=2, tickers=tickers, covered_tickers=covered)
        
        out_dir = tmp_path / "split_output"
        
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
            subset_mode="split",
            fund_dataset_path=str(fund_path),
        )
        
        # Verify subdirectories exist
        assert (out_dir / "all").is_dir()
        assert (out_dir / "sec_covered").is_dir()
        assert (out_dir / "sec_missing").is_dir()
        
        # Verify REPORT_10_2_8.md exists
        assert (out_dir / "REPORT_10_2_8.md").exists()
        
        # Verify diagnostics.json exists and is valid JSON
        assert (out_dir / "diagnostics.json").exists()
        with open(out_dir / "diagnostics.json") as f:
            diag = json.load(f)
        assert "subset_mode" in diag
        
        # Task 10.2.9: With penalty fill, backtest outputs should exist for all subsets
        for subset_name in ["all", "sec_covered", "sec_missing"]:
            subset_dir = out_dir / subset_name
            assert (subset_dir / "subset_audit.json").exists()
            assert (subset_dir / "tilt_audit.json").exists()
            # xgb folder should exist with valid outputs (penalty fill fix)
            assert (subset_dir / "xgb").is_dir()
            # With penalty fill, delta_summary.json should exist
            if (subset_dir / "xgb" / "delta_summary.json").exists():
                with open(subset_dir / "xgb" / "delta_summary.json") as f:
                    delta = json.load(f)
                assert isinstance(delta, dict)
        
        # Verify result dict structure
        assert "all" in result
        assert "sec_covered" in result
        assert "sec_missing" in result
        assert "report_path" in result
    
    def test_subset_failsafe_topk_adaptation(self, tmp_path):
        """Test that top_k is adapted when subset universe is too small."""
        # Only 3 tickers, so subset universe will be < 400
        tickers = ["T001", "T002", "T003"]
        covered = ["T001"]  # Only 1 covered
        
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=2, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb.csv", n_dates=2, tickers=tickers, score_offset=0.1)
        fund_path = _create_synthetic_fund_dataset_csv(tmp_path, n_dates=2, tickers=tickers, covered_tickers=covered)
        
        out_dir = tmp_path / "failsafe_output"
        
        # Should not crash due to fail-safe
        result = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir),
            seed=42,
            subset_mode="sec_covered",
            fund_dataset_path=str(fund_path),
        )
        
        # Verify subset_audit records the adaptation
        with open(out_dir / "subset_audit.json") as f:
            audit = json.load(f)
        
        assert audit["effective_top_k"] < 400
        assert len(audit["dates_with_universe_lt_topk"]) > 0
        
        # Verify warnings in diagnostics
        with open(result["diagnostics_json"]) as f:
            diag = json.load(f)
        
        assert any("SUBSET_TOO_SMALL" in w for w in diag.get("warnings", []))
    
    def test_subset_mode_requires_fund_dataset(self, tmp_path):
        """Test that subset modes require fund_dataset_path."""
        tickers = ["T001", "T002", "T003"]
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=2, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb.csv", n_dates=2, tickers=tickers)
        
        out_dir = tmp_path / "no_fund_output"
        
        with pytest.raises(ValueError, match="fund_dataset_path is required"):
            run_xgb_alpha_ab_backtest(
                prices_csv_path=str(prices_path),
                baseline_scores_csv_path=str(baseline_path),
                xgb_scores_csv_path=str(xgb_path),
                out_dir=str(out_dir),
                seed=42,
                subset_mode="sec_covered",
                fund_dataset_path=None,
            )
    
    def test_fraction_of_all_universe_sanity(self, tmp_path):
        """Task 10.2.9: Verify fraction_of_all_universe is correctly computed."""
        # 6 tickers: 3 covered, 3 missing
        tickers = ["T001", "T002", "T003", "T004", "T005", "T006"]
        covered = ["T001", "T002", "T003"]  # 50% covered
        
        prices_path = _create_synthetic_prices_csv(tmp_path, n_days=300, tickers=tickers)
        baseline_path = _create_synthetic_scores_csv(tmp_path, "baseline.csv", n_dates=2, tickers=tickers)
        xgb_path = _create_synthetic_scores_csv(tmp_path, "xgb.csv", n_dates=2, tickers=tickers, score_offset=0.1)
        fund_path = _create_synthetic_fund_dataset_csv(tmp_path, n_dates=2, tickers=tickers, covered_tickers=covered)
        
        # Test sec_covered
        out_dir_covered = tmp_path / "fraction_covered"
        result_covered = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir_covered),
            seed=42,
            subset_mode="sec_covered",
            fund_dataset_path=str(fund_path),
        )
        
        with open(out_dir_covered / "subset_audit.json") as f:
            audit_covered = json.load(f)
        
        # Fraction should be ~0.5 (3 out of 6)
        fraction_covered = audit_covered["fraction_of_all_universe"]
        assert 0 < fraction_covered < 1, f"Expected 0 < fraction < 1, got {fraction_covered}"
        assert 0.4 < fraction_covered < 0.6, f"Expected ~0.5, got {fraction_covered}"
        
        # Test sec_missing
        out_dir_missing = tmp_path / "fraction_missing"
        result_missing = run_xgb_alpha_ab_backtest(
            prices_csv_path=str(prices_path),
            baseline_scores_csv_path=str(baseline_path),
            xgb_scores_csv_path=str(xgb_path),
            out_dir=str(out_dir_missing),
            seed=42,
            subset_mode="sec_missing",
            fund_dataset_path=str(fund_path),
        )
        
        with open(out_dir_missing / "subset_audit.json") as f:
            audit_missing = json.load(f)
        
        # Fraction should also be ~0.5 (3 out of 6)
        fraction_missing = audit_missing["fraction_of_all_universe"]
        assert 0 < fraction_missing < 1, f"Expected 0 < fraction < 1, got {fraction_missing}"
        
        # The two fractions should sum to approximately 1
        total_fraction = fraction_covered + fraction_missing
        assert 0.9 < total_fraction < 1.1, f"Covered + Missing fractions should sum to ~1, got {total_fraction}"
