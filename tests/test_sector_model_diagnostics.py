"""
Tests for src/sector_model_diagnostics.py

Covers:
- Sector realized returns computation
- IC timeseries (Spearman/Pearson)
- Hit rate calculation
- Tie statistics detection
- Full autopsy determinism
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sector_model_diagnostics import (
    load_sector_to_tickers_json,
    compute_sector_realized_next_returns,
    compute_sector_ic_timeseries,
    compute_sector_hit_rate,
    compute_tie_stats,
    run_sector_autopsy,
)


class TestLoadSectorToTickersJson:
    """Test load_sector_to_tickers_json function."""
    
    def test_loads_and_sorts(self, tmp_path):
        """Test that JSON is loaded with sorted keys and tickers."""
        data = {"S2": ["ZZZ", "AAA"], "S0": ["BBB", "CCC"]}
        path = tmp_path / "mapping.json"
        with open(path, "w") as f:
            json.dump(data, f)
        
        result = load_sector_to_tickers_json(str(path))
        
        assert list(result.keys()) == ["S0", "S2"]
        assert result["S0"] == ["BBB", "CCC"]
        assert result["S2"] == ["AAA", "ZZZ"]
    
    def test_invalid_json_raises(self, tmp_path):
        """Test invalid JSON raises ValueError."""
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            f.write("not valid json")
        
        with pytest.raises(ValueError, match="Failed to load"):
            load_sector_to_tickers_json(str(path))


class TestSectorRealizedReturns:
    """Test compute_sector_realized_next_returns function."""
    
    def test_sector_realized_returns_basic(self, tmp_path):
        """Verify exact returns with small synthetic prices panel."""
        # Create prices: 3 dates, 4 tickers
        prices_data = {
            "date": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "A": [100.0, 110.0, 121.0],  # 10% each month
            "B": [100.0, 105.0, 110.25],  # 5% each month
            "C": [100.0, 120.0, 144.0],  # 20% each month
            "D": [100.0, 90.0, 81.0],  # -10% each month
        }
        prices_path = tmp_path / "prices.csv"
        pd.DataFrame(prices_data).to_csv(prices_path, index=False)
        
        sector_to_tickers = {
            "S0": ["A", "B"],  # avg return: (10% + 5%) / 2 = 7.5%
            "S1": ["C", "D"],  # avg return: (20% - 10%) / 2 = 5%
        }
        
        dates = [pd.Timestamp("2023-01-31"), pd.Timestamp("2023-02-28")]
        
        result = compute_sector_realized_next_returns(
            prices_csv_path=str(prices_path),
            sector_to_tickers=sector_to_tickers,
            dates=dates,
        )
        
        # Check S0 return for 2023-01-31
        # A: 110/100 - 1 = 0.1, B: 105/100 - 1 = 0.05 -> mean = 0.075
        assert np.isclose(result.loc[pd.Timestamp("2023-01-31"), "S0"], 0.075, atol=1e-6)
        
        # Check S1 return for 2023-01-31
        # C: 120/100 - 1 = 0.2, D: 90/100 - 1 = -0.1 -> mean = 0.05
        assert np.isclose(result.loc[pd.Timestamp("2023-01-31"), "S1"], 0.05, atol=1e-6)


class TestSectorICTimeseries:
    """Test compute_sector_ic_timeseries function."""
    
    def test_sector_ic_timeseries_spearman(self):
        """Construct scores perfectly correlated with realized; IC == 1.0."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
        sectors = ["S0", "S1", "S2", "S3"]
        
        # Scores perfectly match realized (same ranking)
        scores = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0]] * 3,
            index=dates,
            columns=sectors,
        )
        realized = pd.DataFrame(
            [[0.01, 0.02, 0.03, 0.04]] * 3,
            index=dates,
            columns=sectors,
        )
        
        ic = compute_sector_ic_timeseries(scores, realized, method="spearman")
        
        assert len(ic) == 3
        for d in dates:
            assert np.isclose(ic.loc[d], 1.0, atol=1e-6)
    
    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        scores = pd.DataFrame([[1, 2]], columns=["S0", "S1"])
        realized = pd.DataFrame([[0.1, 0.2]], columns=["S0", "S1"])
        
        with pytest.raises(ValueError, match="must be 'spearman' or 'pearson'"):
            compute_sector_ic_timeseries(scores, realized, method="kendall")


class TestSectorHitRate:
    """Test compute_sector_hit_rate function."""
    
    def test_sector_hit_rate(self):
        """Verify hit_rate@1 behavior on controlled data."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28"])
        sectors = ["S0", "S1", "S2"]
        
        # Date 1: top by score = S2, top by realized = S2 -> HIT
        # Date 2: top by score = S0, top by realized = S1 -> MISS
        scores = pd.DataFrame(
            [[0.1, 0.2, 0.3], [0.9, 0.1, 0.5]],
            index=dates,
            columns=sectors,
        )
        realized = pd.DataFrame(
            [[0.01, 0.02, 0.05], [0.01, 0.10, 0.02]],
            index=dates,
            columns=sectors,
        )
        
        hit_rate = compute_sector_hit_rate(scores, realized, top_n=1)
        
        # 1 hit out of 2 dates = 0.5
        assert np.isclose(hit_rate, 0.5, atol=1e-6)
    
    def test_hit_rate_top2(self):
        """Test hit_rate@2 behavior."""
        dates = pd.to_datetime(["2023-01-31"])
        sectors = ["S0", "S1", "S2", "S3"]
        
        # Top 2 by score: S2, S3; Top 2 by realized: S2, S3 -> HIT
        scores = pd.DataFrame(
            [[0.1, 0.2, 0.9, 0.8]],
            index=dates,
            columns=sectors,
        )
        realized = pd.DataFrame(
            [[0.01, 0.02, 0.10, 0.09]],
            index=dates,
            columns=sectors,
        )
        
        hit_rate = compute_sector_hit_rate(scores, realized, top_n=2)
        assert np.isclose(hit_rate, 1.0, atol=1e-6)


class TestTieStats:
    """Test compute_tie_stats function."""
    
    def test_tie_stats_detects_large_ties(self):
        """Broadcast-like panel should show low unique values + large tie group."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28"])
        
        # 9 tickers, but only 3 unique values (simulating sector broadcast)
        # S0: 3 tickers with score 0.1
        # S1: 3 tickers with score 0.2
        # S2: 3 tickers with score 0.3
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        data = pd.DataFrame(
            [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]] * 2,
            index=dates,
            columns=tickers,
        )
        
        stats = compute_tie_stats(data)
        
        # Only 3 unique values
        assert stats["n_unique_mean"] == 3.0
        assert stats["n_unique_max"] == 3
        
        # Max tie group = 3
        assert stats["max_tie_group_mean"] == 3.0
        assert stats["max_tie_group_max"] == 3
        
        # Tie fraction = 3/9 = 0.333...
        assert np.isclose(stats["max_tie_frac_mean"], 1/3, atol=1e-6)
    
    def test_no_ties(self):
        """Panel with all unique values should show minimal ties."""
        dates = pd.to_datetime(["2023-01-31"])
        tickers = ["A", "B", "C", "D"]
        data = pd.DataFrame(
            [[0.1, 0.2, 0.3, 0.4]],
            index=dates,
            columns=tickers,
        )
        
        stats = compute_tie_stats(data)
        
        assert stats["n_unique_max"] == 4
        assert stats["max_tie_group_max"] == 1
        assert stats["max_tie_frac_max"] == 0.25


class TestRunSectorAutopsy:
    """Test run_sector_autopsy function."""
    
    def _create_test_files(self, tmp_path):
        """Create test files for autopsy."""
        # Prices
        prices_data = {
            "date": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "A": [100.0, 110.0, 121.0],
            "B": [100.0, 105.0, 110.25],
            "C": [100.0, 120.0, 144.0],
            "D": [100.0, 90.0, 81.0],
        }
        prices_path = tmp_path / "prices.csv"
        pd.DataFrame(prices_data).to_csv(prices_path, index=False)
        
        # Sector mapping
        sector_map = {"S0": ["A", "B"], "S1": ["C", "D"]}
        map_path = tmp_path / "sector_map.json"
        with open(map_path, "w") as f:
            json.dump(sector_map, f)
        
        # Baseline scores (broadcast-like)
        baseline_data = {
            "date": ["2023-01-31", "2023-02-28"],
            "A": [0.5, 0.6], "B": [0.5, 0.6],  # Same within S0
            "C": [0.3, 0.4], "D": [0.3, 0.4],  # Same within S1
        }
        baseline_path = tmp_path / "baseline_scores.csv"
        pd.DataFrame(baseline_data).to_csv(baseline_path, index=False)
        
        # MLP scores (broadcast-like)
        mlp_data = {
            "date": ["2023-01-31", "2023-02-28"],
            "A": [0.4, 0.5], "B": [0.4, 0.5],
            "C": [0.6, 0.7], "D": [0.6, 0.7],
        }
        mlp_path = tmp_path / "mlp_scores.csv"
        pd.DataFrame(mlp_data).to_csv(mlp_path, index=False)
        
        return prices_path, map_path, baseline_path, mlp_path
    
    def test_run_sector_autopsy_writes_json_and_deterministic(self, tmp_path):
        """Run twice -> byte-identical JSON."""
        prices_path, map_path, baseline_path, mlp_path = self._create_test_files(tmp_path)
        
        out_dir_1 = tmp_path / "out1"
        out_dir_2 = tmp_path / "out2"
        
        result1 = run_sector_autopsy(
            prices_csv_path=str(prices_path),
            sector_to_tickers_json_path=str(map_path),
            baseline_sector_broadcast_scores_csv_path=str(baseline_path),
            mlp_sector_broadcast_scores_csv_path=str(mlp_path),
            output_dir=str(out_dir_1),
        )
        
        result2 = run_sector_autopsy(
            prices_csv_path=str(prices_path),
            sector_to_tickers_json_path=str(map_path),
            baseline_sector_broadcast_scores_csv_path=str(baseline_path),
            mlp_sector_broadcast_scores_csv_path=str(mlp_path),
            output_dir=str(out_dir_2),
        )
        
        # Check JSON files are byte-identical
        json1 = (out_dir_1 / "sector_autopsy_summary.json").read_text()
        json2 = (out_dir_2 / "sector_autopsy_summary.json").read_text()
        
        assert json1 == json2, "Autopsy JSON is not deterministic"
        
        # Check result dicts match
        assert result1 == result2
        
        # Check structure
        assert "baseline" in result1
        assert "mlp" in result1
        assert "meta" in result1
        assert "ic_spearman_mean" in result1["baseline"]
        assert "hit_rate_top1" in result1["baseline"]
        assert "tie_stats" in result1["baseline"]
