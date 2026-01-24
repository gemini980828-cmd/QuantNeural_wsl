"""
Ticker-to-Sector Mapping ETL for QUANT-NEURAL.

Builds a deterministic mapping from tickers to sectors using SEC companyfacts JSON files.
This enables broadcasting sector-level MLP scores to ticker-level portfolios for A/B testing.

Design Principles:
- Deterministic: same inputs => byte-identical CSV
- Fail-safe: bad JSON files are skipped with warnings, not crashes
- Configurable: sector_name_to_id is caller-provided (no hardcoded S0..S9 ordering)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


# SIC-to-Sector Mapping Table
# Based on SEC SIC code ranges mapped to broad GICS-like sector buckets
# Reference: https://www.sec.gov/corpfin/division-of-corporation-finance-standard-industrial-classification-sic-code-list
_SIC_TO_SECTOR_TABLE = [
    # (range_start, range_end, sector_name)
    # Agriculture, Forestry, Fishing -> Materials
    (100, 999, "Materials"),
    # Mining -> Energy or Materials
    (1000, 1499, "Energy"),  # Oil, Gas, Coal
    (1500, 1799, "Industrials"),  # Construction
    # Manufacturing
    (2000, 2099, "Consumer Staples"),  # Food
    (2100, 2199, "Consumer Staples"),  # Tobacco
    (2200, 2399, "Consumer Discretionary"),  # Textiles, Apparel
    (2400, 2499, "Materials"),  # Lumber, Wood
    (2500, 2599, "Consumer Discretionary"),  # Furniture
    (2600, 2699, "Materials"),  # Paper
    (2700, 2799, "Communication Services"),  # Printing, Publishing
    (2800, 2899, "Health Care"),  # Chemicals (Pharma)
    (2900, 2999, "Energy"),  # Petroleum Refining
    (3000, 3099, "Materials"),  # Rubber, Plastics
    (3100, 3199, "Consumer Discretionary"),  # Leather
    (3200, 3299, "Materials"),  # Stone, Clay, Glass
    (3300, 3399, "Materials"),  # Primary Metals
    (3400, 3499, "Industrials"),  # Fabricated Metals
    (3500, 3599, "Industrials"),  # Industrial Machinery
    (3600, 3699, "Information Technology"),  # Electronics
    (3700, 3799, "Industrials"),  # Transportation Equipment
    (3800, 3899, "Health Care"),  # Medical Instruments
    (3900, 3999, "Consumer Discretionary"),  # Misc Manufacturing
    # Transportation -> Industrials
    (4000, 4799, "Industrials"),
    # Communications -> Communication Services
    (4800, 4899, "Communication Services"),
    # Utilities
    (4900, 4999, "Utilities"),
    # Wholesale Trade -> Consumer Discretionary
    (5000, 5199, "Consumer Discretionary"),
    # Retail Trade -> Consumer Discretionary/Staples
    (5200, 5399, "Consumer Discretionary"),
    (5400, 5499, "Consumer Staples"),  # Food Stores
    (5500, 5599, "Consumer Discretionary"),  # Auto Dealers
    (5600, 5699, "Consumer Discretionary"),  # Apparel
    (5700, 5799, "Consumer Discretionary"),  # Home Furnishings
    (5800, 5899, "Consumer Discretionary"),  # Eating/Drinking
    (5900, 5999, "Consumer Discretionary"),  # Misc Retail
    # Finance, Insurance, Real Estate
    (6000, 6199, "Financials"),  # Banks
    (6200, 6299, "Financials"),  # Securities
    (6300, 6499, "Financials"),  # Insurance
    (6500, 6599, "Real Estate"),  # Real Estate
    (6700, 6799, "Financials"),  # Holding Companies
    # Services
    (7000, 7099, "Consumer Discretionary"),  # Hotels
    (7200, 7299, "Consumer Discretionary"),  # Personal Services
    (7300, 7399, "Information Technology"),  # Business Services (IT)
    (7500, 7599, "Consumer Discretionary"),  # Auto Services
    (7600, 7699, "Consumer Discretionary"),  # Misc Repair
    (7800, 7899, "Communication Services"),  # Motion Pictures
    (7900, 7999, "Consumer Discretionary"),  # Recreation
    (8000, 8099, "Health Care"),  # Health Services
    (8100, 8199, "Industrials"),  # Legal Services
    (8200, 8299, "Consumer Discretionary"),  # Education
    (8300, 8399, "Health Care"),  # Social Services
    (8600, 8699, "Consumer Discretionary"),  # Membership Orgs
    (8700, 8799, "Information Technology"),  # Engineering/Research
]


def sic_to_sector_name(sic: int | None) -> str:
    """
    Map a SIC code to a broad sector name.
    
    Parameters
    ----------
    sic : int or None
        SIC code from SEC filings.
    
    Returns
    -------
    str
        Sector name (e.g., "Energy", "Financials"). Empty string if unknown.
    
    Notes
    -----
    Uses a deterministic mapping table based on SEC SIC code ranges.
    The mapping is approximate and designed for broad sector classification.
    """
    if sic is None:
        return ""
    
    try:
        sic_int = int(sic)
    except (ValueError, TypeError):
        return ""
    
    for range_start, range_end, sector_name in _SIC_TO_SECTOR_TABLE:
        if range_start <= sic_int <= range_end:
            return sector_name
    
    return ""


def _load_companyfacts_json(path: str) -> dict | None:
    """Load and parse a companyfacts JSON file. Returns None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.warning(f"Skipping invalid JSON file {path}: {e}")
        return None


def build_ticker_to_sector_csv(
    *,
    companyfacts_dir: str,
    universe_tickers: list[str],
    output_csv_path: str,
    sector_name_to_id: dict[str, str],
) -> pd.DataFrame:
    """
    Build a ticker-to-sector mapping CSV from SEC companyfacts JSON files.
    
    Parameters
    ----------
    companyfacts_dir : str
        Directory containing SEC companyfacts JSON files (e.g., CIK0000001234.json).
    universe_tickers : list[str]
        List of tickers to include in the output. Only these tickers will be mapped.
    output_csv_path : str
        Path to write the output CSV.
    sector_name_to_id : dict[str, str]
        Mapping from sector name (e.g., "Energy") to sector ID (e.g., "S1").
        Must be provided by caller to match project's sector ordering.
    
    Returns
    -------
    pd.DataFrame
        The mapping DataFrame with columns: ticker, sector_id, sector_name, sic, source
    
    Notes
    -----
    Determinism guarantees:
    - Output rows sorted by ticker ascending
    - Ties (ticker in multiple files) resolved by smallest source (lexicographic)
    - Bad JSON files are skipped with warnings
    """
    universe_set = set(universe_tickers)
    
    # Collect all (ticker, sector_id, sector_name, sic, source) entries
    entries = []
    
    # List all JSON files in directory
    try:
        files = sorted(os.listdir(companyfacts_dir))  # Sorted for determinism
    except OSError as e:
        logger.warning(f"Cannot read companyfacts_dir {companyfacts_dir}: {e}")
        files = []
    
    for fname in files:
        if not fname.endswith(".json"):
            continue
        
        fpath = os.path.join(companyfacts_dir, fname)
        data = _load_companyfacts_json(fpath)
        
        if data is None:
            continue
        
        # Extract fields
        tickers_raw = data.get("tickers", [])
        if not isinstance(tickers_raw, list):
            tickers_raw = []
        
        sic_raw = data.get("sic")
        try:
            sic = int(sic_raw) if sic_raw is not None else None
        except (ValueError, TypeError):
            sic = None
        
        sector_name = sic_to_sector_name(sic)
        sector_id = sector_name_to_id.get(sector_name, "")
        
        # Source for tie-breaking (use filename for consistency)
        source = fname
        
        for ticker in tickers_raw:
            if not isinstance(ticker, str):
                continue
            ticker_upper = ticker.upper()
            if ticker_upper in universe_set:
                entries.append({
                    "ticker": ticker_upper,
                    "sector_id": sector_id,
                    "sector_name": sector_name,
                    "sic": sic if sic is not None else "",
                    "source": source,
                })
    
    # Build DataFrame
    if entries:
        df = pd.DataFrame(entries)
    else:
        df = pd.DataFrame(columns=["ticker", "sector_id", "sector_name", "sic", "source"])
    
    # Handle duplicates: keep row with smallest source (lexicographic tie-break)
    if not df.empty:
        df = df.sort_values(["ticker", "source"])
        df = df.drop_duplicates(subset=["ticker"], keep="first")
    
    # Final sort by ticker for determinism
    df = df.sort_values("ticker").reset_index(drop=True)
    
    # Write CSV
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
    return df


def build_sector_to_tickers(
    mapping_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Build a sector_id -> [tickers] dict from a mapping DataFrame.
    
    Parameters
    ----------
    mapping_df : pd.DataFrame
        DataFrame with columns: ticker, sector_id (at minimum).
    
    Returns
    -------
    dict[str, list[str]]
        Mapping from sector_id to sorted list of tickers.
        Rows with empty sector_id are excluded.
    """
    result = {}
    
    if mapping_df.empty or "sector_id" not in mapping_df.columns:
        return result
    
    # Filter out empty sector_id
    valid = mapping_df[mapping_df["sector_id"].astype(str).str.strip() != ""]
    
    for sector_id, group in valid.groupby("sector_id"):
        tickers = sorted(group["ticker"].tolist())
        result[sector_id] = tickers
    
    return result
