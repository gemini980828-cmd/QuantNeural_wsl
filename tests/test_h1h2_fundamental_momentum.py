"""
Tests for src/h1h2_fundamental_momentum.py

Covers:
- Output shape and column validation
- PIT filed-gate and forward-fill
- Latest-filed wins within cutoff
- Manual calculation verification
- n_sectors=10 enforcement
"""

import numpy as np
import pandas as pd
import pytest

from src.h1h2_fundamental_momentum import (
    build_h1h2_relative_fundamental_momentum,
    _get_pit_visible_facts,
    _compute_ttm_for_cik_at_month,
)


def _create_facts_df(rows: list[dict]) -> pd.DataFrame:
    """Helper to create a facts DataFrame from row dicts."""
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["val"] = df["val"].astype(float)
    return df


class TestOutputShapeAndColumns:
    """Test output structure."""
    
    def test_output_shape_and_columns(self):
        """Test that output has correct shape and column names."""
        # Create facts with 2 CIKs, 4+ quarters each
        facts_rows = []
        
        # CIK 1 (sector 0): 5 quarters of OperatingIncomeLoss
        for i, (end, filed) in enumerate([
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
            ("2023-03-31", "2023-05-01"),
        ]):
            facts_rows.append({
                "cik": "0000000001",
                "tag": "OperatingIncomeLoss",
                "unit": "USD",
                "end": end,
                "filed": filed,
                "val": 100 + i * 10,
            })
        
        # CIK 2 (sector 1): 5 quarters of NetIncomeLoss
        for i, (end, filed) in enumerate([
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
            ("2023-03-31", "2023-05-01"),
        ]):
            facts_rows.append({
                "cik": "0000000002",
                "tag": "NetIncomeLoss",
                "unit": "USD",
                "end": end,
                "filed": filed,
                "val": 50 + i * 5,
            })
        
        facts = _create_facts_df(facts_rows)
        
        # 14 month_ends (13 months span for H2 calculation)
        month_ends = pd.date_range("2022-04-30", periods=14, freq="ME")
        
        cik_to_sector = {
            "0000000001": 0,
            "0000000002": 1,
        }
        
        result = build_h1h2_relative_fundamental_momentum(
            facts,
            month_ends=month_ends,
            cik_to_sector=cik_to_sector,
            n_sectors=10,
        )
        
        # Check index
        assert result.index.equals(month_ends)
        
        # Check exactly 20 columns
        assert len(result.columns) == 20
        
        # Check column order
        expected_cols = (
            [f"S{i}_H1" for i in range(10)] +
            [f"S{i}_H2" for i in range(10)]
        )
        assert list(result.columns) == expected_cols
        
        # Check all float dtype
        for col in result.columns:
            assert pd.api.types.is_float_dtype(result[col]), f"{col} is not float"


class TestPITFiledGate:
    """Test PIT filed-date enforcement."""
    
    def test_pit_filed_gate_is_enforced(self):
        """Test that filings are only visible after their filed date - with explicit assertions."""
        # CIK with 4 quarters, but Q4 filed late
        facts_rows = [
            # Q1-Q3 filed promptly
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-03-31", "filed": "2022-05-01", "val": 100},
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-06-30", "filed": "2022-08-01", "val": 110},
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-09-30", "filed": "2022-11-01", "val": 120},
            # Q4 filed very late (March 2023)
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-12-31", "filed": "2023-03-15", "val": 130},
        ]
        facts = _create_facts_df(facts_rows)
        
        # Test dates: before and after Q4 filing
        t_before = pd.Timestamp("2023-02-28")  # Before Q4 filed
        t_after = pd.Timestamp("2023-04-30")   # After Q4 filed
        
        # Get PIT-visible facts at each date
        visible_before = _get_pit_visible_facts(facts, t_before)
        visible_after = _get_pit_visible_facts(facts, t_after)
        
        # ASSERTION A: Q4 row NOT visible before filed date
        q4_end = pd.Timestamp("2022-12-31")
        assert (visible_before["end"] == q4_end).sum() == 0, "Q4 should NOT be visible before filed"
        
        # ASSERTION B: Q4 row IS visible after filed date
        assert (visible_after["end"] == q4_end).sum() == 1, "Q4 should be visible after filed"
        
        # ASSERTION C: TTM before is NaN (only 3 periods)
        ttm_before = _compute_ttm_for_cik_at_month(
            visible_before, "0000000001", "NetIncomeLoss", "USD", t_before
        )
        assert np.isnan(ttm_before), f"TTM before should be NaN (only 3 periods), got {ttm_before}"
        
        # ASSERTION D: TTM after equals sum of 4 vals (100+110+120+130=460)
        ttm_after = _compute_ttm_for_cik_at_month(
            visible_after, "0000000001", "NetIncomeLoss", "USD", t_after
        )
        expected_ttm = 100 + 110 + 120 + 130  # 460
        assert abs(ttm_after - expected_ttm) < 1e-6, f"TTM after should be {expected_ttm}, got {ttm_after}"


class TestLatestFiledWins:
    """Test latest-filed-wins rule."""
    
    def test_latest_filed_wins_for_same_end(self):
        """Test that newer filing for same (cik, tag, unit, end) wins - with explicit assertions."""
        # Same quarter end, two filings with different values
        facts_rows = [
            # Q4: original filing
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-12-31", "filed": "2023-02-15", "val": 100},
            # Q4: amended/restated filing with different value (later date)
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-12-31", "filed": "2023-04-01", "val": 150},
        ]
        facts = _create_facts_df(facts_rows)
        
        # Test at May 2023 (after both filings)
        t_after = pd.Timestamp("2023-05-31")
        
        # Get PIT-visible facts
        visible = _get_pit_visible_facts(facts, t_after)
        
        # ASSERTION: Exactly one row for (cik, tag, unit, end), with val=150
        q4_end = pd.Timestamp("2022-12-31")
        q4_rows = visible[
            (visible["cik"] == "0000000001") &
            (visible["tag"] == "NetIncomeLoss") &
            (visible["unit"] == "USD") &
            (visible["end"] == q4_end)
        ]
        
        assert len(q4_rows) == 1, f"Expected exactly 1 row, got {len(q4_rows)}"
        assert q4_rows.iloc[0]["val"] == 150.0, f"Expected val=150.0, got {q4_rows.iloc[0]['val']}"


class TestRelativeMomentumCalculation:
    """Test momentum calculation correctness."""
    
    def test_relative_momentum_matches_manual_small_case(self):
        """Verify momentum calculation with a simple manual case (using n_sectors=10)."""
        # Simple setup: 2 CIKs in sectors 0 and 1, constant values
        # We'll verify that with constant values, momentum is ~0
        
        facts_rows = []
        
        # Build 8 quarters of data (to allow H2 lookback)
        quarters = [
            ("2021-03-31", "2021-05-01"),
            ("2021-06-30", "2021-08-01"),
            ("2021-09-30", "2021-11-01"),
            ("2021-12-31", "2022-02-01"),
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
        ]
        
        for end, filed in quarters:
            # CIK 1 (sector 0): constant 100
            facts_rows.append({
                "cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
                "end": end, "filed": filed, "val": 100.0,
            })
            # CIK 2 (sector 1): constant 50
            facts_rows.append({
                "cik": "0000000002", "tag": "NetIncomeLoss", "unit": "USD",
                "end": end, "filed": filed, "val": 50.0,
            })
        
        facts = _create_facts_df(facts_rows)
        
        cik_to_sector = {
            "0000000001": 0,
            "0000000002": 1,
        }
        
        # Choose a month where we have full data
        month_ends = pd.DatetimeIndex([pd.Timestamp("2023-03-31")])
        
        result = build_h1h2_relative_fundamental_momentum(
            facts,
            month_ends=month_ends,
            cik_to_sector=cik_to_sector,
            n_sectors=10,  # Must be 10
            method="logdiff",
        )
        
        # With constant values, TTM is always the same
        # So momentum (logdiff) should be ~0
        # And relative momentum (sector - market) should be ~0
        
        # Check that S0_H1 and S1_H1 are close to 0 (since no change)
        s0_h1 = result.loc[month_ends[0], "S0_H1"]
        s1_h1 = result.loc[month_ends[0], "S1_H1"]
        
        # They should be close to zero or NaN (if lag data unavailable)
        assert np.isnan(s0_h1) or abs(s0_h1) < 1e-6, f"S0_H1={s0_h1} should be ~0"
        assert np.isnan(s1_h1) or abs(s1_h1) < 1e-6, f"S1_H1={s1_h1} should be ~0"
    
    def test_method_pct_works(self):
        """Test that method='pct' produces valid output."""
        facts_rows = []
        
        quarters = [
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
        ]
        
        for i, (end, filed) in enumerate(quarters):
            facts_rows.append({
                "cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
                "end": end, "filed": filed, "val": 100.0 + i * 10,
            })
        
        facts = _create_facts_df(facts_rows)
        
        cik_to_sector = {"0000000001": 0}
        month_ends = pd.DatetimeIndex([pd.Timestamp("2023-03-31")])
        
        result = build_h1h2_relative_fundamental_momentum(
            facts,
            month_ends=month_ends,
            cik_to_sector=cik_to_sector,
            n_sectors=10,
            method="pct",
        )
        
        # Just verify it runs and produces float output
        assert len(result) == 1
        assert result["S0_H1"].dtype == np.float64


class TestEdgeCases:
    """Test edge cases."""
    
    def test_missing_cik_in_mapping_ignored(self):
        """Test that CIKs not in cik_to_sector are ignored."""
        facts_rows = [
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-12-31", "filed": "2023-02-01", "val": 100},
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-09-30", "filed": "2022-11-01", "val": 100},
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-06-30", "filed": "2022-08-01", "val": 100},
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-03-31", "filed": "2022-05-01", "val": 100},
            # CIK 999 is NOT in mapping
            {"cik": "0000000999", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-12-31", "filed": "2023-02-01", "val": 9999},
        ]
        facts = _create_facts_df(facts_rows)
        
        cik_to_sector = {"0000000001": 0}  # CIK 999 NOT included
        month_ends = pd.DatetimeIndex([pd.Timestamp("2023-03-31")])
        
        # Should not crash
        result = build_h1h2_relative_fundamental_momentum(
            facts,
            month_ends=month_ends,
            cik_to_sector=cik_to_sector,
            n_sectors=10,
        )
        
        assert len(result) == 1
    
    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        facts = _create_facts_df([
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-12-31", "filed": "2023-02-01", "val": 100},
        ])
        
        with pytest.raises(ValueError, match="method must be"):
            build_h1h2_relative_fundamental_momentum(
                facts,
                month_ends=pd.DatetimeIndex([pd.Timestamp("2023-03-31")]),
                cik_to_sector={"0000000001": 0},
                method="invalid",
            )
    
    def test_n_sectors_must_be_10(self):
        """Test that n_sectors != 10 raises ValueError."""
        facts = _create_facts_df([
            {"cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
             "end": "2022-12-31", "filed": "2023-02-01", "val": 100},
        ])
        
        with pytest.raises(ValueError, match="n_sectors must be 10"):
            build_h1h2_relative_fundamental_momentum(
                facts,
                month_ends=pd.DatetimeIndex([pd.Timestamp("2023-03-31")]),
                cik_to_sector={"0000000001": 0},
                n_sectors=2,  # Not 10 -> should raise
            )


class TestSectorCountsAttr:
    """Test sector_counts diagnostics attached via attrs."""
    
    def test_h1h2_attaches_sector_counts_attr(self):
        """Test that sector_counts is attached via attrs with correct structure."""
        facts_rows = []
        
        # 2 CIKs in sectors 0 and 1, 4+ quarters each
        quarters = [
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
        ]
        
        for end, filed in quarters:
            facts_rows.append({
                "cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
                "end": end, "filed": filed, "val": 100.0,
            })
            facts_rows.append({
                "cik": "0000000002", "tag": "NetIncomeLoss", "unit": "USD",
                "end": end, "filed": filed, "val": 50.0,
            })
        
        facts = _create_facts_df(facts_rows)
        
        cik_to_sector = {
            "0000000001": 0,
            "0000000002": 1,
        }
        
        month_ends = pd.date_range("2023-03-31", periods=3, freq="ME")
        
        result = build_h1h2_relative_fundamental_momentum(
            facts,
            month_ends=month_ends,
            cik_to_sector=cik_to_sector,
            n_sectors=10,
        )
        
        # 1) sector_counts in attrs
        assert "sector_counts" in result.attrs
        counts_df = result.attrs["sector_counts"]
        
        # 2) counts_df is a DataFrame
        assert isinstance(counts_df, pd.DataFrame)
        
        # 3) Same index as features_df
        assert counts_df.index.equals(result.index)
        
        # 4) Exactly 10 columns named S{i}_n_firms
        expected_cols = [f"S{i}_n_firms" for i in range(10)]
        assert list(counts_df.columns) == expected_cols
        
        # 5) All values are integers and >= 0
        for col in counts_df.columns:
            assert counts_df[col].dtype in [np.int32, np.int64, int], f"{col} is not int"
            assert (counts_df[col] >= 0).all(), f"{col} has negative values"
    
    def test_sector_counts_deterministic(self):
        """Test that sector counts are deterministic across runs."""
        facts_rows = []
        
        quarters = [
            ("2022-03-31", "2022-05-01"),
            ("2022-06-30", "2022-08-01"),
            ("2022-09-30", "2022-11-01"),
            ("2022-12-31", "2023-02-01"),
        ]
        
        for end, filed in quarters:
            facts_rows.append({
                "cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
                "end": end, "filed": filed, "val": 100.0,
            })
        
        facts = _create_facts_df(facts_rows)
        cik_to_sector = {"0000000001": 0}
        month_ends = pd.DatetimeIndex([pd.Timestamp("2023-03-31")])
        
        result1 = build_h1h2_relative_fundamental_momentum(
            facts,
            month_ends=month_ends,
            cik_to_sector=cik_to_sector,
            n_sectors=10,
        )
        
        result2 = build_h1h2_relative_fundamental_momentum(
            facts,
            month_ends=month_ends,
            cik_to_sector=cik_to_sector,
            n_sectors=10,
        )
        
        counts_df1 = result1.attrs["sector_counts"]
        counts_df2 = result2.attrs["sector_counts"]
        
        pd.testing.assert_frame_equal(counts_df1, counts_df2)
