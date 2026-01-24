"""
Tests for audit_manifest_stooq_tickers.py

All implementer prompts must be written in English (must include this statement verbatim).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_manifest_stooq_tickers import (
    normalize_ticker,
    extract_stooq_tickers,
    load_manifest,
    build_matches,
    build_stooq_aligned_manifest,
    run_audit,
)


class TestNormalizeTicker:
    """Tests for normalize_ticker function."""
    
    def test_basic_uppercase(self):
        assert normalize_ticker("aapl") == "AAPL"
    
    def test_strip_whitespace(self):
        assert normalize_ticker("  AAPL  ") == "AAPL"
    
    def test_replace_dash_with_dot(self):
        assert normalize_ticker("BRK-B") == "BRK.B"
    
    def test_remove_us_suffix(self):
        assert normalize_ticker("AAPL.US") == "AAPL"
        assert normalize_ticker("aapl.us") == "AAPL"
    
    def test_complex_normalization(self):
        # Multiple transformations
        assert normalize_ticker("brk-b.us") == "BRK.B"
    
    def test_exchange_prefix_removal(self):
        assert normalize_ticker("NYSE:BRK.B") == "BRK.B"
        assert normalize_ticker("NASDAQ:AAPL") == "AAPL"
    
    def test_collapse_multiple_dots(self):
        assert normalize_ticker("A..B...C") == "A.B.C"
    
    def test_combined(self):
        # BRK-B.US -> BRK.B.US -> BRK.B (remove .US)
        assert normalize_ticker("BRK-B.US") == "BRK.B"


class TestExtractStooqTickers:
    """Tests for extract_stooq_tickers function."""
    
    def test_extract_from_directory(self, tmp_path):
        # Create dummy files
        (tmp_path / "AAPL.US.csv").touch()
        (tmp_path / "BRK-B.US.csv").touch()
        (tmp_path / "MSFT.US.txt").touch()
        (tmp_path / "XYZ.csv").touch()
        
        df = extract_stooq_tickers(tmp_path)
        
        assert len(df) == 4
        assert set(df['stooq_ticker_raw']) == {'AAPL.US', 'BRK-B.US', 'MSFT.US', 'XYZ'}
        assert 'AAPL' in df['stooq_ticker_norm'].values
        assert 'BRK.B' in df['stooq_ticker_norm'].values
    
    def test_empty_directory(self, tmp_path):
        df = extract_stooq_tickers(tmp_path)
        assert df.empty
    
    def test_nonexistent_directory(self, tmp_path):
        df = extract_stooq_tickers(tmp_path / "nonexistent")
        assert df.empty
    
    def test_deterministic_ordering(self, tmp_path):
        # Create files in random order
        for name in ["ZZZ.csv", "AAA.csv", "MMM.csv"]:
            (tmp_path / name).touch()
        
        df = extract_stooq_tickers(tmp_path)
        # Should be sorted by stooq_ticker_raw
        assert list(df['stooq_ticker_raw']) == ['AAA', 'MMM', 'ZZZ']


class TestLoadManifest:
    """Tests for load_manifest function."""
    
    def test_load_valid_manifest(self, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "ticker,companyfacts_status,companyfacts_path\n"
            "AAPL,ok,data/sec/aapl.json\n"
            "MSFT,ok,data/sec/msft.json\n"
            "BADSTATUS,pending,data/sec/bad.json\n"
        )
        
        df = load_manifest(manifest_path)
        
        assert len(df) == 2
        assert set(df['manifest_ticker_raw']) == {'AAPL', 'MSFT'}
    
    def test_filter_empty_path(self, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "ticker,companyfacts_status,companyfacts_path\n"
            "AAPL,ok,data/sec/aapl.json\n"
            "EMPTY,ok,\n"
        )
        
        df = load_manifest(manifest_path)
        assert len(df) == 1
    
    def test_missing_columns_raises(self, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text("ticker,some_column\nAAPL,value\n")
        
        with pytest.raises(ValueError, match="missing required columns"):
            load_manifest(manifest_path)
    
    def test_nonexistent_file(self, tmp_path):
        df = load_manifest(tmp_path / "nonexistent.csv")
        assert df.empty


class TestBuildMatches:
    """Tests for build_matches function."""
    
    def test_exact_matches(self):
        stooq_df = pd.DataFrame({
            'stooq_ticker_raw': ['AAPL.US', 'MSFT.US'],
            'stooq_ticker_norm': ['AAPL', 'MSFT'],
        })
        manifest_df = pd.DataFrame({
            'manifest_ticker_raw': ['AAPL', 'MSFT'],
            'manifest_ticker_norm': ['AAPL', 'MSFT'],
            'companyfacts_status': ['ok', 'ok'],
            'companyfacts_path': ['path/aapl.json', 'path/msft.json'],
        })
        
        matches, stooq_only, manifest_only = build_matches(stooq_df, manifest_df)
        
        assert len(matches) == 2
        assert len(stooq_only) == 0
        assert len(manifest_only) == 0
    
    def test_partial_overlap(self):
        stooq_df = pd.DataFrame({
            'stooq_ticker_raw': ['AAPL.US', 'XYZ.csv'],
            'stooq_ticker_norm': ['AAPL', 'XYZ'],
        })
        manifest_df = pd.DataFrame({
            'manifest_ticker_raw': ['AAPL', 'ONLY_MANIFEST'],
            'manifest_ticker_norm': ['AAPL', 'ONLY_MANIFEST'],
            'companyfacts_status': ['ok', 'ok'],
            'companyfacts_path': ['path/aapl.json', 'path/only.json'],
        })
        
        matches, stooq_only, manifest_only = build_matches(stooq_df, manifest_df)
        
        assert len(matches) == 1
        assert 'XYZ' in stooq_only['stooq_ticker_norm'].values
        assert 'ONLY_MANIFEST' in manifest_only['manifest_ticker_norm'].values


class TestBuildStooqAlignedManifest:
    """Tests for build_stooq_aligned_manifest function."""
    
    def test_aligned_manifest_structure(self):
        stooq_df = pd.DataFrame({
            'stooq_ticker_raw': ['AAPL.US', 'XYZ'],
            'stooq_ticker_norm': ['AAPL', 'XYZ'],
        })
        matches = pd.DataFrame({
            'stooq_ticker_raw': ['AAPL.US'],
            'companyfacts_path': ['path/aapl.json'],
        })
        
        aligned = build_stooq_aligned_manifest(stooq_df, matches)
        
        assert len(aligned) == 2
        assert set(aligned.columns) == {'ticker', 'companyfacts_status', 'companyfacts_path'}
        
        # AAPL.US should be ok
        aapl_row = aligned[aligned['ticker'] == 'AAPL.US'].iloc[0]
        assert aapl_row['companyfacts_status'] == 'ok'
        assert aapl_row['companyfacts_path'] == 'path/aapl.json'
        
        # XYZ should be missing
        xyz_row = aligned[aligned['ticker'] == 'XYZ'].iloc[0]
        assert xyz_row['companyfacts_status'] == 'missing'
        assert xyz_row['companyfacts_path'] == ''


class TestRunAuditEndToEnd:
    """End-to-end test for run_audit function."""
    
    def test_full_audit(self, tmp_path):
        # Create stooq directory with dummy files
        stooq_dir = tmp_path / "stooq"
        stooq_dir.mkdir()
        (stooq_dir / "AAPL.US.csv").touch()
        (stooq_dir / "BRK-B.US.csv").touch()
        (stooq_dir / "MSFT.US.txt").touch()
        (stooq_dir / "XYZ.csv").touch()
        
        # Create manifest CSV
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "ticker,companyfacts_status,companyfacts_path\n"
            "AAPL,ok,data/sec/aapl.json\n"
            "BRK.B,ok,data/sec/brkb.json\n"
            "MSFT.US,ok,data/sec/msft.json\n"
            "ONLYINMANIFEST,ok,data/sec/only.json\n"
        )
        
        # Run audit
        out_dir = tmp_path / "output"
        summary = run_audit(
            stooq_data_dir=str(stooq_dir),
            manifest_csv=str(manifest_path),
            out_dir=str(out_dir),
        )
        
        # Verify summary
        assert summary['stooq_count'] == 4
        assert summary['overlap_count'] == 3  # AAPL, BRK.B, MSFT
        assert summary['stooq_only_count'] == 1  # XYZ
        assert summary['manifest_only_count'] == 1  # ONLYINMANIFEST
        
        # Verify output files exist
        assert (out_dir / 'audit_summary.json').exists()
        assert (out_dir / 'matches.csv').exists()
        assert (out_dir / 'stooq_only.csv').exists()
        assert (out_dir / 'manifest_only.csv').exists()
        assert (out_dir / 'stooq_aligned_manifest.csv').exists()
        
        # Verify stooq_only contains XYZ
        stooq_only = pd.read_csv(out_dir / 'stooq_only.csv')
        assert 'XYZ' in stooq_only['stooq_ticker_norm'].values
        
        # Verify manifest_only contains ONLYINMANIFEST
        manifest_only = pd.read_csv(out_dir / 'manifest_only.csv')
        assert 'ONLYINMANIFEST' in manifest_only['manifest_ticker_norm'].values
        
        # Verify stooq_aligned_manifest
        aligned = pd.read_csv(out_dir / 'stooq_aligned_manifest.csv')
        assert len(aligned) == 4
        assert set(aligned.columns) == {'ticker', 'companyfacts_status', 'companyfacts_path'}
        
        # Verify deterministic ordering
        assert list(aligned['ticker']) == sorted(aligned['ticker'].tolist())
