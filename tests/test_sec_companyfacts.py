"""
Tests for src/sec_companyfacts.py

Covers:
- Schema and dtype validation
- PIT cutoff uses filed date (not end date)
- Latest-filed wins within cutoff
- CIK mismatch validation
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.sec_companyfacts import load_companyfacts_json, select_latest_filed


def _create_companyfacts_json(tmp_path, cik: str, facts: dict) -> str:
    """Helper to create a companyfacts JSON file."""
    data = {
        "cik": cik,
        "entityName": "Test Company",
        "facts": facts,
    }
    path = tmp_path / "companyfacts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestLoadCompanyfactsJson:
    """Tests for load_companyfacts_json function."""
    
    def test_load_parses_schema_and_types(self, tmp_path):
        """Test that loader parses JSON and returns correct schema/dtypes."""
        facts = {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 1000000, "fy": 2023, "fp": "Q2"},
                            {"end": "2023-09-30", "filed": "2023-11-01", "val": 1100000, "fy": 2023, "fp": "Q3"},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 500000},
                        ]
                    }
                },
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        df = load_companyfacts_json(path, as_of_date="2023-12-31")
        
        # Required columns exist
        required_cols = ["cik", "taxonomy", "tag", "unit", "end", "filed", "val"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Dtypes
        assert pd.api.types.is_datetime64_any_dtype(df["end"])
        assert pd.api.types.is_datetime64_any_dtype(df["filed"])
        assert pd.api.types.is_float_dtype(df["val"])
        
        # Sorted by (taxonomy, tag, unit, end, filed)
        sort_cols = ["taxonomy", "tag", "unit", "end", "filed"]
        df_sorted = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(df, df_sorted)
        
        # Should have 3 rows total
        assert len(df) == 3
    
    def test_pit_cutoff_uses_filed_not_end(self, tmp_path):
        """Test that PIT cutoff uses filed date, not end date."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            # end is before cutoff, but filed is AFTER cutoff
                            {"end": "2023-06-30", "filed": "2023-12-15", "val": 1000},
                            # Both end and filed are before cutoff
                            {"end": "2023-03-31", "filed": "2023-05-01", "val": 900},
                        ]
                    }
                }
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        # Cutoff is 2023-12-01, so filed="2023-12-15" should be excluded
        df = load_companyfacts_json(path, as_of_date="2023-12-01")
        
        # Only the second row should remain
        assert len(df) == 1
        assert df.iloc[0]["val"] == 900.0
    
    def test_pit_cutoff_empty_raises_valueerror(self, tmp_path):
        """Test that empty result after cutoff raises ValueError."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-12-15", "val": 1000},
                        ]
                    }
                }
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        # Cutoff before any filed date
        with pytest.raises(ValueError, match="No data remaining"):
            load_companyfacts_json(path, as_of_date="2023-01-01")
    
    def test_cik_mismatch_raises(self, tmp_path):
        """Test that CIK mismatch raises ValueError."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 1000},
                        ]
                    }
                }
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        # Provide different CIK
        with pytest.raises(ValueError, match="CIK mismatch"):
            load_companyfacts_json(path, as_of_date="2023-12-31", cik="0000009999")
    
    def test_cik_normalization_matches(self, tmp_path):
        """Test that CIK normalization works (with/without leading zeros)."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 1000},
                        ]
                    }
                }
            }
        }
        # JSON has CIK without full padding
        path = _create_companyfacts_json(tmp_path, "1750", facts)
        
        # Provide with full padding - should match after normalization
        df = load_companyfacts_json(path, as_of_date="2023-12-31", cik="0000001750")
        
        assert len(df) == 1
        assert df.iloc[0]["cik"] == "0000001750"
    
    def test_nonfinite_val_raises(self, tmp_path):
        """Test that non-numeric val raises ValueError."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": "not_a_number"},
                        ]
                    }
                }
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        with pytest.raises(ValueError, match="NaN"):
            load_companyfacts_json(path, as_of_date="2023-12-31")


class TestSelectLatestFiled:
    """Tests for select_latest_filed function."""
    
    def test_select_latest_filed_wins_within_cutoff(self, tmp_path):
        """Test that latest filed within cutoff wins."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            # Same period, older filing
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 1000},
                            # Same period, newer filing (still within cutoff)
                            {"end": "2023-06-30", "filed": "2023-09-15", "val": 1100},
                            # Same period, even newer filing BUT after cutoff
                            {"end": "2023-06-30", "filed": "2023-12-20", "val": 1200},
                        ]
                    }
                }
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        # Load with cutoff that includes first two, excludes third
        df = load_companyfacts_json(path, as_of_date="2023-12-01")
        
        # Before select_latest_filed, should have 2 rows
        assert len(df) == 2
        
        # After select_latest_filed, should have 1 row with val=1100 (newer filing within cutoff)
        result = select_latest_filed(df, as_of_date="2023-12-01")
        
        assert len(result) == 1
        assert result.iloc[0]["val"] == 1100.0
        assert result.iloc[0]["filed"] == pd.to_datetime("2023-09-15")
    
    def test_select_latest_filed_multiple_keys(self, tmp_path):
        """Test latest-filed selection with multiple unique keys."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            # Key 1: Q1
                            {"end": "2023-03-31", "filed": "2023-05-01", "val": 500},
                            {"end": "2023-03-31", "filed": "2023-06-01", "val": 510},  # Later filing
                            # Key 2: Q2
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 600},
                        ]
                    }
                }
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        df = load_companyfacts_json(path, as_of_date="2023-12-31")
        result = select_latest_filed(df, as_of_date="2023-12-31")
        
        # Should have 2 rows (one per unique key)
        assert len(result) == 2
        
        # Q1 should have val=510 (later filing)
        q1_row = result[result["end"] == pd.to_datetime("2023-03-31")].iloc[0]
        assert q1_row["val"] == 510.0
        
        # Q2 should have val=600
        q2_row = result[result["end"] == pd.to_datetime("2023-06-30")].iloc[0]
        assert q2_row["val"] == 600.0
    
    def test_select_latest_filed_empty_after_cutoff_raises(self, tmp_path):
        """Test that empty after cutoff raises ValueError."""
        facts = {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 1000},
                        ]
                    }
                }
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        df = load_companyfacts_json(path, as_of_date="2023-12-31")
        
        # Apply stricter cutoff that excludes all rows
        with pytest.raises(ValueError, match="No data remaining"):
            select_latest_filed(df, as_of_date="2023-01-01")
    
    def test_select_latest_filed_deterministic_sorting(self, tmp_path):
        """Test that output is deterministically sorted."""
        facts = {
            "us-gaap": {
                "ZTag": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 100},
                        ]
                    }
                },
                "ATag": {
                    "units": {
                        "USD": [
                            {"end": "2023-06-30", "filed": "2023-08-01", "val": 200},
                        ]
                    }
                },
            }
        }
        path = _create_companyfacts_json(tmp_path, "0000001750", facts)
        
        df = load_companyfacts_json(path, as_of_date="2023-12-31")
        result = select_latest_filed(df, as_of_date="2023-12-31")
        
        # Should be sorted by tag (ATag before ZTag)
        assert result.iloc[0]["tag"] == "ATag"
        assert result.iloc[1]["tag"] == "ZTag"
