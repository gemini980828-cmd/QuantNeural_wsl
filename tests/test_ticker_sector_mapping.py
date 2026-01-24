"""
Tests for src/ticker_sector_mapping.py

Covers:
- Basic ETL and determinism
- Sector-to-tickers dict building
- Bad JSON handling (fail-safe)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ticker_sector_mapping import (
    build_ticker_to_sector_csv,
    build_sector_to_tickers,
    sic_to_sector_name,
)


class TestSicToSectorName:
    """Test sic_to_sector_name function."""
    
    def test_energy_sic(self):
        """Test that oil/gas SIC codes map to Energy."""
        assert sic_to_sector_name(1311) == "Energy"  # Oil and Gas Extraction
        assert sic_to_sector_name(2911) == "Energy"  # Petroleum Refining
    
    def test_financials_sic(self):
        """Test that banking SIC codes map to Financials."""
        assert sic_to_sector_name(6022) == "Financials"  # State Commercial Banks
        assert sic_to_sector_name(6211) == "Financials"  # Security Brokers
    
    def test_technology_sic(self):
        """Test that tech SIC codes map to Information Technology."""
        assert sic_to_sector_name(3672) == "Information Technology"  # Electronics
        assert sic_to_sector_name(7370) == "Information Technology"  # Computer Services
    
    def test_unknown_sic(self):
        """Test that unknown SIC codes return empty string."""
        assert sic_to_sector_name(9999) == ""
        assert sic_to_sector_name(None) == ""
        assert sic_to_sector_name(-1) == ""


class TestBuildTickerToSectorCSV:
    """Test build_ticker_to_sector_csv function."""
    
    def _create_fake_companyfacts(self, tmp_path: Path, cik: str, sic: int, tickers: list[str]) -> Path:
        """Create a fake companyfacts JSON file."""
        data = {
            "cik": cik,
            "sic": sic,
            "tickers": tickers,
        }
        fname = f"CIK{cik}.json"
        fpath = tmp_path / fname
        with open(fpath, "w") as f:
            json.dump(data, f)
        return fpath
    
    def test_build_ticker_to_sector_csv_basic_and_deterministic(self, tmp_path):
        """Test basic ETL and that outputs are deterministic."""
        companyfacts_dir = tmp_path / "companyfacts"
        companyfacts_dir.mkdir()
        
        # Create 2 fake companyfacts JSON files
        self._create_fake_companyfacts(companyfacts_dir, "0000000001", 3672, ["AAA", "AAA2"])  # IT (Electronics)
        self._create_fake_companyfacts(companyfacts_dir, "0000000002", 1311, ["BBB"])  # Energy
        
        universe_tickers = ["AAA", "BBB", "MISSING"]
        sector_name_to_id = {
            "Information Technology": "S3",
            "Energy": "S1",
        }
        
        # Run 1
        output_csv_1 = tmp_path / "run1" / "mapping.csv"
        df1 = build_ticker_to_sector_csv(
            companyfacts_dir=str(companyfacts_dir),
            universe_tickers=universe_tickers,
            output_csv_path=str(output_csv_1),
            sector_name_to_id=sector_name_to_id,
        )
        
        # Run 2
        output_csv_2 = tmp_path / "run2" / "mapping.csv"
        df2 = build_ticker_to_sector_csv(
            companyfacts_dir=str(companyfacts_dir),
            universe_tickers=universe_tickers,
            output_csv_path=str(output_csv_2),
            sector_name_to_id=sector_name_to_id,
        )
        
        # Check CSV exists
        assert output_csv_1.exists()
        assert output_csv_2.exists()
        
        # Check DataFrames are identical
        assert df1.equals(df2), "ETL output is not deterministic"
        
        # Check CSV content is byte-identical
        content1 = output_csv_1.read_text()
        content2 = output_csv_2.read_text()
        assert content1 == content2, "CSV files are not byte-identical"
        
        # Check rows are sorted by ticker
        assert list(df1["ticker"]) == sorted(df1["ticker"].tolist())
        
        # Check correct sector assignments
        row_aaa = df1[df1["ticker"] == "AAA"].iloc[0]
        assert row_aaa["sector_id"] == "S3"
        assert row_aaa["sector_name"] == "Information Technology"
        
        row_bbb = df1[df1["ticker"] == "BBB"].iloc[0]
        assert row_bbb["sector_id"] == "S1"
        assert row_bbb["sector_name"] == "Energy"
        
        # Check MISSING ticker is not included (not in companyfacts)
        assert "MISSING" not in df1["ticker"].tolist()
    
    def test_duplicate_ticker_uses_smallest_source(self, tmp_path):
        """Test that duplicate tickers use the smallest source as tie-breaker."""
        companyfacts_dir = tmp_path / "companyfacts"
        companyfacts_dir.mkdir()
        
        # Same ticker in two files with different SIC codes
        self._create_fake_companyfacts(companyfacts_dir, "0000000002", 1311, ["AAA"])  # Energy
        self._create_fake_companyfacts(companyfacts_dir, "0000000001", 3672, ["AAA"])  # IT
        
        universe_tickers = ["AAA"]
        sector_name_to_id = {
            "Information Technology": "S3",
            "Energy": "S1",
        }
        
        output_csv = tmp_path / "mapping.csv"
        df = build_ticker_to_sector_csv(
            companyfacts_dir=str(companyfacts_dir),
            universe_tickers=universe_tickers,
            output_csv_path=str(output_csv),
            sector_name_to_id=sector_name_to_id,
        )
        
        # Should have only 1 row
        assert len(df) == 1
        
        # Should use CIK0000000001.json (smaller lexicographically)
        assert df.iloc[0]["source"] == "CIK0000000001.json"
        assert df.iloc[0]["sector_id"] == "S3"  # IT, not Energy


class TestBuildSectorToTickers:
    """Test build_sector_to_tickers function."""
    
    def test_build_sector_to_tickers_sorts_and_filters(self):
        """Test that sector_to_tickers dict is correct."""
        mapping_df = pd.DataFrame({
            "ticker": ["ZZZ", "AAA", "BBB", "CCC", "DDD"],
            "sector_id": ["S1", "S1", "S2", "", "S1"],
            "sector_name": ["Energy", "Energy", "Financials", "", "Energy"],
        })
        
        result = build_sector_to_tickers(mapping_df)
        
        # Check keys
        assert set(result.keys()) == {"S1", "S2"}
        
        # Check S1 tickers are sorted
        assert result["S1"] == ["AAA", "DDD", "ZZZ"]
        
        # Check S2
        assert result["S2"] == ["BBB"]
        
        # Empty sector_id should be excluded
        assert "" not in result
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        mapping_df = pd.DataFrame(columns=["ticker", "sector_id", "sector_name"])
        result = build_sector_to_tickers(mapping_df)
        assert result == {}


class TestBadJsonHandling:
    """Test that bad JSON files are skipped gracefully."""
    
    def test_bad_json_is_skipped_not_crash(self, tmp_path):
        """Test that invalid JSON files don't crash the ETL."""
        companyfacts_dir = tmp_path / "companyfacts"
        companyfacts_dir.mkdir()
        
        # Create a valid JSON
        valid_data = {"cik": "0000000001", "sic": 3672, "tickers": ["AAA"]}
        with open(companyfacts_dir / "CIK0000000001.json", "w") as f:
            json.dump(valid_data, f)
        
        # Create an invalid JSON
        with open(companyfacts_dir / "CIK_INVALID.json", "w") as f:
            f.write("{ this is not valid json }")
        
        universe_tickers = ["AAA"]
        sector_name_to_id = {"Information Technology": "S3"}
        
        output_csv = tmp_path / "mapping.csv"
        
        # Should NOT raise
        df = build_ticker_to_sector_csv(
            companyfacts_dir=str(companyfacts_dir),
            universe_tickers=universe_tickers,
            output_csv_path=str(output_csv),
            sector_name_to_id=sector_name_to_id,
        )
        
        # Should still have valid data
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAA"
