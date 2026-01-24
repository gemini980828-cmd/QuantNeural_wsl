"""
Real-Data Smoke Wiring for QUANT-NEURAL.

Integrates:
- Stooq price loader (src/stooq_prices.py)
- SEC companyfacts loader (src/sec_companyfacts.py)
- H1/H2 fundamental momentum (src/h1h2_fundamental_momentum.py)

Point-in-Time (PIT) Rules:
- Prices: only dates <= as_of_date
- Fundamentals: only filed <= as_of_date (per month_end)
- No future data visible at any point in the pipeline
- No "now" or system time logic
"""

from __future__ import annotations

import pandas as pd

from src.stooq_prices import load_stooq_daily_prices, resample_to_monthly
from src.sec_companyfacts import load_companyfacts_json
from src.h1h2_fundamental_momentum import build_h1h2_relative_fundamental_momentum


def build_real_data_feature_frame(
    stooq_csv_path: str,
    *,
    price_ticker: str,
    as_of_date: str,
    companyfacts_json_paths: list[str],
    cik_to_sector: dict[str, int],
) -> pd.DataFrame:
    """
    Build real-data feature frame integrating Stooq prices and SEC fundamentals.
    
    Parameters
    ----------
    stooq_csv_path : str
        Path to Stooq bulk CSV file.
    price_ticker : str
        Ticker to filter in Stooq data (e.g., "SPY.US").
    as_of_date : str
        PIT cutoff date in "YYYY-MM-DD" format.
    companyfacts_json_paths : list[str]
        Paths to SEC companyfacts JSON files (one per company).
    cik_to_sector : dict[str, int]
        Mapping from CIK string to sector index (0-9).
    
    Returns
    -------
    pd.DataFrame
        Index: month-end timestamps (<= as_of_date)
        Columns: exactly 20 columns: S0_H1..S9_H1, S0_H2..S9_H2
        dtype: float
    
    Raises
    ------
    ValueError
        If no monthly dates produced, max(month_end) > as_of_date,
        empty facts after loading, or cik_to_sector is missing/empty.
    """
    # Validate inputs
    if not cik_to_sector:
        raise ValueError("cik_to_sector mapping is missing or empty")
    
    as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
    
    # =========================================================================
    # 1. Load and resample prices to monthly
    # =========================================================================
    daily_prices = load_stooq_daily_prices(
        stooq_csv_path,
        as_of_date=as_of_date,
        ticker=price_ticker,
    )
    
    monthly_prices = resample_to_monthly(daily_prices)
    
    if monthly_prices.empty:
        raise ValueError("No monthly prices produced from Stooq data")
    
    # Extract month-end timestamps
    month_ends = pd.DatetimeIndex(monthly_prices["date"].unique()).sort_values()
    
    # Validate PIT: max month_end must be <= as_of_date
    if month_ends.max() > as_of_dt:
        raise ValueError(
            f"max(month_end)={month_ends.max()} > as_of_date={as_of_date}"
        )
    
    # =========================================================================
    # 2. Load SEC companyfacts (do NOT globally deduplicate - let H1/H2 handle PIT)
    # =========================================================================
    all_facts = []
    
    for json_path in companyfacts_json_paths:
        # Load with as_of_date to apply initial PIT filter
        # But DON'T call select_latest_filed globally - that would remove
        # earlier filings needed for earlier month_ends
        facts = load_companyfacts_json(json_path, as_of_date=as_of_date)
        all_facts.append(facts)
    
    if not all_facts:
        raise ValueError("No companyfacts JSON files loaded")
    
    combined_facts = pd.concat(all_facts, ignore_index=True)
    
    if combined_facts.empty:
        raise ValueError("Empty facts after loading all companyfacts files")
    
    # =========================================================================
    # 3. Build H1/H2 relative fundamental momentum features
    # =========================================================================
    # The build function handles per-month_end PIT enforcement internally
    features = build_h1h2_relative_fundamental_momentum(
        combined_facts,
        month_ends=month_ends,
        cik_to_sector=cik_to_sector,
        n_sectors=10,  # Fixed 20-feature shape
    )
    
    # =========================================================================
    # 4. Validate output
    # =========================================================================
    expected_cols = [f"S{i}_H1" for i in range(10)] + [f"S{i}_H2" for i in range(10)]
    
    if list(features.columns) != expected_cols:
        raise ValueError(
            f"Output columns mismatch. Expected {expected_cols}, got {list(features.columns)}"
        )
    
    if len(features.columns) != 20:
        raise ValueError(f"Expected 20 columns, got {len(features.columns)}")
    
    # Ensure float dtype
    features = features.astype(float)
    
    return features
