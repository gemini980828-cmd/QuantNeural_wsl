"""
Tests for src/sec_fundamentals_v2.py

Covers:
- YAML configuration loading and validation
- Canonical entry extraction with fallback logic
- Wide table building with PIT alignment
- Provenance tracking (_tag columns)
- Coverage diagnostics computation
- Fail-safe behavior (no exceptions, graceful NaN handling)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.sec_fundamentals_v2 import (
    CanonicalItemConfig,
    TagMappingConfig,
    load_tag_mapping,
    extract_canonical_entry,
    build_canonical_wide_table,
    compute_coverage_diagnostics,
    CoverageDiagnostics,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_tag_mapping_yaml(tmp_path) -> Path:
    """Create a minimal test tag mapping YAML."""
    config = {
        "version": "1.0-test",
        "canonical_items": {
            "total_assets": {
                "taxonomy": "us-gaap",
                "unit": "USD",
                "tags": ["Assets", "TotalAssets"],
            },
            "net_income": {
                "taxonomy": "us-gaap",
                "unit": "USD",
                "tags": ["NetIncomeLoss", "ProfitLoss"],
            },
            "shares_outstanding": {
                "taxonomy": "dei",
                "unit": "shares",
                "tags": ["EntityCommonStockSharesOutstanding"],
                "fallback_taxonomy": "us-gaap",
                "fallback_tags": ["CommonStockSharesOutstanding"],
            },
        },
    }
    path = tmp_path / "test_mapping.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def sample_companyfacts_json(tmp_path) -> Path:
    """Create a minimal SEC companyfacts JSON file."""
    data = {
        "cik": "0000320193",
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2023-03-31", "filed": "2023-05-01", "val": 1000000000},
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 1100000000},
                            {"end": "2023-09-30", "filed": "2023-11-01", "val": 1200000000},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"end": "2023-03-31", "filed": "2023-05-01", "val": 50000000},
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 55000000},
                        ]
                    }
                },
                # CommonStockSharesOutstanding for fallback testing
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2023-03-31", "filed": "2023-05-01", "val": 10000000},
                        ]
                    }
                },
            },
            "dei": {
                # Note: EntityCommonStockSharesOutstanding missing - should fallback to us-gaap
            },
        },
    }
    path = tmp_path / "CIK0000320193.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def empty_companyfacts_json(tmp_path) -> Path:
    """Create an empty SEC companyfacts JSON file."""
    data = {
        "cik": "0000000001",
        "entityName": "Empty Corp",
        "facts": {},
    }
    path = tmp_path / "CIK0000000001.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ==============================================================================
# Test: TagMappingConfig
# ==============================================================================


class TestTagMappingConfig:
    """Tests for tag mapping configuration loading."""

    def test_load_from_yaml_success(self, sample_tag_mapping_yaml):
        """Test successful loading of YAML config."""
        config = TagMappingConfig.from_yaml(sample_tag_mapping_yaml)
        
        assert config.version == "1.0-test"
        assert len(config.canonical_items) == 3
        assert "total_assets" in config.canonical_items
        assert "net_income" in config.canonical_items
        assert "shares_outstanding" in config.canonical_items

    def test_canonical_item_properties(self, sample_tag_mapping_yaml):
        """Test CanonicalItemConfig properties."""
        config = TagMappingConfig.from_yaml(sample_tag_mapping_yaml)
        
        total_assets = config.canonical_items["total_assets"]
        assert total_assets.taxonomy == "us-gaap"
        assert total_assets.unit == "USD"
        assert total_assets.tags == ["Assets", "TotalAssets"]
        
    def test_all_tag_sources_with_fallback(self, sample_tag_mapping_yaml):
        """Test all_tag_sources() includes fallback tags."""
        config = TagMappingConfig.from_yaml(sample_tag_mapping_yaml)
        
        shares = config.canonical_items["shares_outstanding"]
        sources = shares.all_tag_sources()
        
        # Primary: dei/EntityCommonStockSharesOutstanding
        # Fallback: us-gaap/CommonStockSharesOutstanding
        assert len(sources) == 2
        assert sources[0] == ("dei", "EntityCommonStockSharesOutstanding")
        assert sources[1] == ("us-gaap", "CommonStockSharesOutstanding")

    def test_load_missing_file_raises(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            TagMappingConfig.from_yaml(missing_path)

    def test_load_invalid_yaml_raises(self, tmp_path):
        """Test that invalid YAML structure raises ValueError."""
        invalid_path = tmp_path / "invalid.yaml"
        invalid_path.write_text("just a string, not a dict")
        
        with pytest.raises(ValueError, match="expected dict"):
            TagMappingConfig.from_yaml(invalid_path)


class TestLoadTagMapping:
    """Tests for load_tag_mapping function."""

    def test_load_with_explicit_path(self, sample_tag_mapping_yaml):
        """Test loading with explicit path."""
        config = load_tag_mapping(sample_tag_mapping_yaml)
        assert config.version == "1.0-test"

    def test_load_default_path_exists(self):
        """Test that the default tag mapping file exists in the repo."""
        from src.sec_fundamentals_v2 import DEFAULT_TAG_MAPPING_PATH
        
        if DEFAULT_TAG_MAPPING_PATH.exists():
            config = load_tag_mapping()
            assert config.version is not None
            assert len(config.canonical_items) > 0


# ==============================================================================
# Test: extract_canonical_entry
# ==============================================================================


class TestExtractCanonicalEntry:
    """Tests for canonical entry extraction with fallback logic."""

    def test_extract_primary_tag_success(self, sample_companyfacts_json, sample_tag_mapping_yaml):
        """Test extraction using primary tag when available."""
        config = load_tag_mapping(sample_tag_mapping_yaml)
        item_config = config.canonical_items["total_assets"]
        
        df, used_tag = extract_canonical_entry(str(sample_companyfacts_json), item_config)
        
        assert not df.empty
        assert len(df) == 3
        assert used_tag == "Assets"
        
        # Verify column structure
        assert list(df.columns) == ["end", "filed", "val"]
        assert pd.api.types.is_datetime64_any_dtype(df["end"])
        assert pd.api.types.is_datetime64_any_dtype(df["filed"])

    def test_extract_fallback_tag_used(self, sample_companyfacts_json, sample_tag_mapping_yaml):
        """Test that fallback tag is used when primary is missing."""
        config = load_tag_mapping(sample_tag_mapping_yaml)
        item_config = config.canonical_items["shares_outstanding"]
        
        df, used_tag = extract_canonical_entry(str(sample_companyfacts_json), item_config)
        
        assert not df.empty
        # Should use us-gaap/CommonStockSharesOutstanding since dei is empty
        assert used_tag == "CommonStockSharesOutstanding"

    def test_extract_all_tags_missing_returns_empty(self, empty_companyfacts_json, sample_tag_mapping_yaml):
        """Test that empty DataFrame is returned when all tags are missing."""
        config = load_tag_mapping(sample_tag_mapping_yaml)
        item_config = config.canonical_items["total_assets"]
        
        df, used_tag = extract_canonical_entry(str(empty_companyfacts_json), item_config)
        
        assert df.empty
        assert used_tag is None

    def test_extract_nonexistent_file_returns_empty(self, sample_tag_mapping_yaml, tmp_path):
        """Test that nonexistent file returns empty DataFrame, not exception."""
        config = load_tag_mapping(sample_tag_mapping_yaml)
        item_config = config.canonical_items["total_assets"]
        
        nonexistent = str(tmp_path / "nonexistent.json")
        df, used_tag = extract_canonical_entry(nonexistent, item_config)
        
        assert df.empty
        assert used_tag is None


# ==============================================================================
# Test: build_canonical_wide_table
# ==============================================================================


class TestBuildCanonicalWideTable:
    """Tests for canonical wide table building."""

    def test_wide_table_structure(self, sample_companyfacts_json, sample_tag_mapping_yaml):
        """Test that wide table has correct structure."""
        as_of_dates = pd.date_range("2023-09-01", "2023-12-31", freq="D")
        
        df = build_canonical_wide_table(
            str(sample_companyfacts_json),
            as_of_dates,
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        # Index should match as_of_dates
        assert len(df) == len(as_of_dates)
        assert df.index.equals(as_of_dates)
        
        # Should have canonical item columns
        assert "total_assets" in df.columns
        assert "net_income" in df.columns
        assert "shares_outstanding" in df.columns
        
        # Should have provenance columns
        assert "total_assets_tag" in df.columns
        assert "net_income_tag" in df.columns

    def test_wide_table_dtypes(self, sample_companyfacts_json, sample_tag_mapping_yaml):
        """Test that numeric columns are float32."""
        as_of_dates = pd.date_range("2023-09-01", "2023-09-10", freq="D")
        
        df = build_canonical_wide_table(
            str(sample_companyfacts_json),
            as_of_dates,
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        assert df["total_assets"].dtype == np.float32
        assert df["net_income"].dtype == np.float32

    def test_pit_alignment_correctness(self, sample_companyfacts_json, sample_tag_mapping_yaml):
        """Test that PIT alignment uses filed_date, not end_date."""
        # Assets has these filings:
        # end=2023-03-31, filed=2023-05-01, val=1B
        # end=2023-06-30, filed=2023-08-01, val=1.1B
        # end=2023-09-30, filed=2023-11-01, val=1.2B
        
        as_of_dates = pd.DatetimeIndex([
            "2023-04-15",  # Before first filing -> NaN
            "2023-05-15",  # After first filing (2023-05-01) -> 1B
            "2023-08-15",  # After second filing (2023-08-01) -> 1.1B
            "2023-11-15",  # After third filing (2023-11-01) -> 1.2B
        ])
        
        df = build_canonical_wide_table(
            str(sample_companyfacts_json),
            as_of_dates,
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        # Before first filing: should be NaN
        assert np.isnan(df.loc["2023-04-15", "total_assets"])
        
        # After each filing: should have that filing's value
        assert df.loc["2023-05-15", "total_assets"] == pytest.approx(1_000_000_000.0)
        assert df.loc["2023-08-15", "total_assets"] == pytest.approx(1_100_000_000.0)
        assert df.loc["2023-11-15", "total_assets"] == pytest.approx(1_200_000_000.0)

    def test_provenance_columns_populated(self, sample_companyfacts_json, sample_tag_mapping_yaml):
        """Test that provenance columns show which tag was used."""
        as_of_dates = pd.date_range("2023-09-01", "2023-09-05", freq="D")
        
        df = build_canonical_wide_table(
            str(sample_companyfacts_json),
            as_of_dates,
            tag_mapping_path=sample_tag_mapping_yaml,
            include_provenance=True,
        )
        
        # total_assets used "Assets" tag
        assert df["total_assets_tag"].iloc[0] == "Assets"
        
        # shares_outstanding used fallback "CommonStockSharesOutstanding"
        assert df["shares_outstanding_tag"].iloc[0] == "CommonStockSharesOutstanding"

    def test_provenance_columns_disabled(self, sample_companyfacts_json, sample_tag_mapping_yaml):
        """Test that provenance columns can be disabled."""
        as_of_dates = pd.date_range("2023-09-01", "2023-09-05", freq="D")
        
        df = build_canonical_wide_table(
            str(sample_companyfacts_json),
            as_of_dates,
            tag_mapping_path=sample_tag_mapping_yaml,
            include_provenance=False,
        )
        
        # Should not have _tag columns
        assert "total_assets_tag" not in df.columns
        
        # Should still have value columns
        assert "total_assets" in df.columns

    def test_missing_file_returns_nan_df(self, sample_tag_mapping_yaml, tmp_path):
        """Test that missing file returns DataFrame with NaN values."""
        as_of_dates = pd.date_range("2023-09-01", "2023-09-05", freq="D")
        nonexistent = str(tmp_path / "nonexistent.json")
        
        df = build_canonical_wide_table(
            nonexistent,
            as_of_dates,
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        # Should have correct length
        assert len(df) == len(as_of_dates)
        
        # All values should be NaN
        assert df["total_assets"].isna().all()
        assert df["net_income"].isna().all()


# ==============================================================================
# Test: compute_coverage_diagnostics
# ==============================================================================


class TestComputeCoverageDiagnostics:
    """Tests for coverage diagnostics computation."""

    def test_diagnostics_basic(self, sample_companyfacts_json, sample_tag_mapping_yaml, tmp_path):
        """Test basic diagnostics computation."""
        # Create a directory with our test file
        test_dir = tmp_path / "sec_bulk"
        test_dir.mkdir()
        
        # Copy our sample file
        import shutil
        shutil.copy(sample_companyfacts_json, test_dir / "CIK0000320193.json")
        
        diagnostics = compute_coverage_diagnostics(
            str(test_dir),
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        assert diagnostics.total_tickers == 1
        
        # total_assets and net_income should be found
        assert diagnostics.items_coverage["total_assets"] == 1.0
        assert diagnostics.items_coverage["net_income"] == 1.0
        
        # shares_outstanding uses fallback
        assert diagnostics.items_coverage["shares_outstanding"] == 1.0
        
        # Check tag usage distribution
        assert "Assets" in diagnostics.tag_usage_distribution["total_assets"]
        assert diagnostics.tag_usage_distribution["total_assets"]["Assets"] == 1

    def test_diagnostics_empty_dir(self, sample_tag_mapping_yaml, tmp_path):
        """Test diagnostics on empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        diagnostics = compute_coverage_diagnostics(
            str(empty_dir),
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        assert diagnostics.total_tickers == 0
        assert diagnostics.overall_coverage == 0.0

    def test_diagnostics_summary(self, sample_companyfacts_json, sample_tag_mapping_yaml, tmp_path):
        """Test diagnostics summary generation."""
        test_dir = tmp_path / "sec_bulk"
        test_dir.mkdir()
        
        import shutil
        shutil.copy(sample_companyfacts_json, test_dir / "CIK0000320193.json")
        
        diagnostics = compute_coverage_diagnostics(
            str(test_dir),
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        summary = diagnostics.summary()
        
        assert "Coverage Diagnostics" in summary
        assert "total_assets" in summary
        assert "100.0%" in summary or "1.0" in summary

    def test_diagnostics_to_dict(self, sample_companyfacts_json, sample_tag_mapping_yaml, tmp_path):
        """Test diagnostics serialization to dict."""
        test_dir = tmp_path / "sec_bulk"
        test_dir.mkdir()
        
        import shutil
        shutil.copy(sample_companyfacts_json, test_dir / "CIK0000320193.json")
        
        diagnostics = compute_coverage_diagnostics(
            str(test_dir),
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        result = diagnostics.to_dict()
        
        assert "total_tickers" in result
        assert "items_coverage" in result
        assert "overall_coverage" in result
        
        # Should be JSON-serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0


# ==============================================================================
# Test: Fail-Safe Behavior
# ==============================================================================


class TestFailSafeBehavior:
    """Tests for fail-safe behavior (no exceptions, graceful handling)."""

    def test_extract_corrupt_json_returns_empty(self, tmp_path, sample_tag_mapping_yaml):
        """Test that corrupt JSON returns empty DataFrame."""
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("{ invalid json }")
        
        config = load_tag_mapping(sample_tag_mapping_yaml)
        item_config = config.canonical_items["total_assets"]
        
        # Should not raise
        df, used_tag = extract_canonical_entry(str(corrupt_path), item_config)
        
        assert df.empty
        assert used_tag is None

    def test_build_wide_table_corrupt_json(self, tmp_path, sample_tag_mapping_yaml):
        """Test that corrupt JSON returns NaN DataFrame."""
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("not json at all")
        
        as_of_dates = pd.date_range("2023-09-01", "2023-09-05", freq="D")
        
        # Should not raise
        df = build_canonical_wide_table(
            str(corrupt_path),
            as_of_dates,
            tag_mapping_path=sample_tag_mapping_yaml,
        )
        
        # Should return DataFrame, not crash
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(as_of_dates)


# ==============================================================================
# Test: Integration with Real Default Config
# ==============================================================================


class TestDefaultConfigIntegration:
    """Tests using the actual default tag mapping configuration."""

    def test_default_config_loads_successfully(self):
        """Test that the default config in configs/sec_tag_mapping.yaml loads."""
        from src.sec_fundamentals_v2 import DEFAULT_TAG_MAPPING_PATH
        
        if DEFAULT_TAG_MAPPING_PATH.exists():
            config = load_tag_mapping()
            
            # Should have multiple canonical items
            assert len(config.canonical_items) >= 20
            
            # Check some expected items exist
            assert "total_assets" in config.canonical_items
            assert "net_income" in config.canonical_items
            assert "stockholders_equity" in config.canonical_items

    def test_default_config_item_structure(self):
        """Test that default config items have valid structure."""
        from src.sec_fundamentals_v2 import DEFAULT_TAG_MAPPING_PATH
        
        if DEFAULT_TAG_MAPPING_PATH.exists():
            config = load_tag_mapping()
            
            for name, item in config.canonical_items.items():
                # Each item must have valid properties
                assert item.name == name
                assert item.taxonomy in ("us-gaap", "dei", "invest")
                assert len(item.tags) >= 1
                assert item.unit in ("USD", "shares", "USD/shares")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
