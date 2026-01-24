"""
SEC Fundamental Features - PIT-safe extraction utilities.

Extracts point-in-time safe fundamental features from SEC companyfacts data.
Key principle: For any feature used at date t, only filings with filed_date <= t may be used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Required and optional SEC tags for fundamental features
REQUIRED_TAGS = [
    "NetIncomeLoss",
    "Revenues",
    "GrossProfit",
    "StockholdersEquity",
    "Assets",
    "Liabilities",
    "CommonStockSharesOutstanding",
]

OPTIONAL_TAGS = [
    "OperatingIncomeLoss",
    "CashAndCashEquivalentsAtCarryingValue",
]

ALL_TAGS = REQUIRED_TAGS + OPTIONAL_TAGS


def load_facts_json(path: str) -> pd.DataFrame:
    """
    Load SEC companyfacts JSON and extract fundamental data.
    
    Parameters
    ----------
    path : str
        Path to SEC companyfacts JSON file.
    
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - end_date (datetime64): Period end date
        - filed_date (datetime64): Filing date (PIT key)
        - tag (str): SEC XBRL tag name
        - value (float64): Reported value
        - form (str): Form type (10-K, 10-Q, etc.)
    
    Raises
    ------
    ValueError
        If file cannot be parsed or is empty.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"SEC companyfacts file not found: {path}")
    
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse SEC JSON: {path} - {e}")
    
    if not data:
        raise ValueError(f"Empty SEC companyfacts file: {path}")
    
    facts = data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    
    if not us_gaap:
        raise ValueError(f"No us-gaap facts found: {path}")
    
    records = []
    
    for tag in ALL_TAGS:
        tag_data = us_gaap.get(tag, {})
        units = tag_data.get("units", {})
        
        # Try USD first, then shares for share counts
        unit_data = units.get("USD", units.get("shares", []))
        
        for entry in unit_data:
            end_date = entry.get("end")
            filed_date = entry.get("filed")
            val = entry.get("val")
            form = entry.get("form", "")
            
            if end_date and filed_date and val is not None:
                records.append({
                    "end_date": end_date,
                    "filed_date": filed_date,
                    "tag": tag,
                    "value": float(val),
                    "form": form,
                })
    
    if not records:
        raise ValueError(f"No valid fundamental data extracted: {path}")
    
    df = pd.DataFrame(records)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df = df.dropna(subset=["end_date", "filed_date"])
    
    if df.empty:
        raise ValueError(f"No valid dates in SEC data: {path}")
    
    # Sort by filed_date for PIT alignment
    df = df.sort_values("filed_date").reset_index(drop=True)
    
    return df


def load_facts(path: str) -> pd.DataFrame:
    """
    Load SEC facts from either JSON or CSV format.
    
    Parameters
    ----------
    path : str
        Path to SEC data file (JSON or CSV).
    
    Returns
    -------
    pd.DataFrame
        Normalized long DataFrame with columns:
        [end_date, filed_date, tag, value, form]
    """
    path_obj = Path(path)
    
    if path_obj.suffix.lower() == ".json":
        return load_facts_json(path)
    elif path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        required_cols = ["end_date", "filed_date", "tag", "value"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
        df = df.dropna(subset=["end_date", "filed_date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        if "form" not in df.columns:
            df["form"] = ""
        
        return df.sort_values("filed_date").reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported file format: {path_obj.suffix}")


def _compute_ttm(
    df_tag: pd.DataFrame,
    filed_date: pd.Timestamp,
) -> float | None:
    """
    Compute trailing twelve months (TTM) sum for a tag.
    
    Uses the 4 most recent quarterly values filed on or before filed_date.
    Only considers 10-Q and 10-K forms with fiscal quarter data (3-month periods).
    
    Parameters
    ----------
    df_tag : pd.DataFrame
        DataFrame filtered to a single tag, sorted by filed_date.
    filed_date : pd.Timestamp
        Point-in-time cutoff date.
    
    Returns
    -------
    float or None
        TTM sum if 4 quarters available, else None.
    """
    # Filter to values available at filed_date
    available = df_tag[df_tag["filed_date"] <= filed_date].copy()
    
    if available.empty:
        return None
    
    # Filter to quarterly reports (typically 10-Q) with 3-month periods
    # We need to identify quarterly values vs annual
    available = available[available["form"].isin(["10-Q", "10-K/A", "10-Q/A", "10-K"])]
    
    if len(available) < 4:
        return None
    
    # Take the 4 most recent by filed_date, sorted by end_date to avoid duplicates
    available = available.sort_values(["end_date", "filed_date"])
    available = available.drop_duplicates(subset=["end_date"], keep="last")
    
    # Take the 4 most recent quarters by end_date
    recent = available.nlargest(4, "end_date")
    
    if len(recent) < 4:
        return None
    
    return float(recent["value"].sum())


def _get_latest_value(
    df_tag: pd.DataFrame,
    filed_date: pd.Timestamp,
) -> float | None:
    """
    Get the latest value for a tag available at filed_date.
    
    Parameters
    ----------
    df_tag : pd.DataFrame
        DataFrame filtered to a single tag.
    filed_date : pd.Timestamp
        Point-in-time cutoff date.
    
    Returns
    -------
    float or None
        Latest value if available, else None.
    """
    available = df_tag[df_tag["filed_date"] <= filed_date]
    
    if available.empty:
        return None
    
    # Get the most recently filed value
    latest_idx = available["filed_date"].idxmax()
    return float(available.loc[latest_idx, "value"])


def compute_pit_fundamentals(
    *,
    daily_index: pd.DatetimeIndex,
    facts: pd.DataFrame,
    close_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute PIT-safe fundamental features for each date in daily_index.
    
    For each date t, only uses filings with filed_date <= t.
    Uses merge_asof for vectorized PIT alignment.
    
    Parameters
    ----------
    daily_index : pd.DatetimeIndex
        Sequence of trading dates to compute features for.
    facts : pd.DataFrame
        Long-format facts DataFrame from load_facts().
    close_series : pd.Series, optional
        Close prices indexed by date for mktcap calculation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with fundamental feature columns (float32):
        - shares_out: Latest shares outstanding
        - mktcap: Market cap (shares_out * close)
        - ep_ttm: Earnings/Price TTM
        - bp: Book/Price
        - gp_to_assets: Gross Profit / Assets
        - roa_ttm: ROA TTM
        - leverage: Liabilities / Assets
        - gross_margin_ttm: Gross Margin TTM
    """
    if daily_index.empty:
        return pd.DataFrame(index=daily_index)
    
    # Prepare output
    result = pd.DataFrame(index=daily_index)
    
    # Pre-compute tag dataframes
    tag_dfs = {}
    for tag in ALL_TAGS:
        tag_df = facts[facts["tag"] == tag].copy()
        if not tag_df.empty:
            tag_dfs[tag] = tag_df
    
    # Use merge_asof for vectorized PIT alignment
    # For each tag, get the latest value as of each date
    
    for tag in ALL_TAGS:
        if tag not in tag_dfs:
            result[f"_raw_{tag}"] = np.nan
            continue
        
        tag_df = tag_dfs[tag].copy()
        
        # Get latest value for each filing date
        tag_df = tag_df.sort_values("filed_date")
        tag_df = tag_df.drop_duplicates(subset=["filed_date"], keep="last")
        tag_df = tag_df[["filed_date", "value"]].rename(columns={"value": f"_raw_{tag}"})
        
        # Merge onto daily index
        dates_df = pd.DataFrame({"date": daily_index})
        dates_df = dates_df.sort_values("date")
        tag_df = tag_df.rename(columns={"filed_date": "date"})
        
        merged = pd.merge_asof(
            dates_df,
            tag_df,
            on="date",
            direction="backward",
        )
        
        merged = merged.set_index("date")
        result[f"_raw_{tag}"] = merged[f"_raw_{tag}"].reindex(daily_index)
    
    # Compute derived features
    shares_out = result.get("_raw_CommonStockSharesOutstanding", pd.Series(np.nan, index=daily_index))
    equity = result.get("_raw_StockholdersEquity", pd.Series(np.nan, index=daily_index))
    assets = result.get("_raw_Assets", pd.Series(np.nan, index=daily_index))
    liabilities = result.get("_raw_Liabilities", pd.Series(np.nan, index=daily_index))
    net_income = result.get("_raw_NetIncomeLoss", pd.Series(np.nan, index=daily_index))
    gross_profit = result.get("_raw_GrossProfit", pd.Series(np.nan, index=daily_index))
    revenues = result.get("_raw_Revenues", pd.Series(np.nan, index=daily_index))
    
    # Core features
    result["shares_out"] = shares_out
    
    # Market cap
    if close_series is not None:
        close_aligned = close_series.reindex(daily_index)
        result["mktcap"] = shares_out * close_aligned
    else:
        result["mktcap"] = np.nan
    
    # Valuation ratios (using latest values as proxy for TTM)
    mktcap = result["mktcap"].replace(0, np.nan)
    
    # EP = Earnings / Price (inverse P/E)
    result["ep_ttm"] = net_income / mktcap
    
    # BP = Book / Price (inverse P/B)
    result["bp"] = equity / mktcap
    
    # Quality metrics
    assets_safe = assets.replace(0, np.nan)
    result["gp_to_assets"] = gross_profit / assets_safe
    result["roa_ttm"] = net_income / assets_safe
    result["leverage"] = liabilities / assets_safe
    
    # Gross margin
    revenues_safe = revenues.replace(0, np.nan)
    result["gross_margin_ttm"] = gross_profit / revenues_safe
    
    # Select output columns
    output_cols = [
        "shares_out",
        "mktcap",
        "ep_ttm",
        "bp",
        "gp_to_assets",
        "roa_ttm",
        "leverage",
        "gross_margin_ttm",
    ]
    
    output = result[output_cols].copy()
    
    # Cast to float32
    for col in output.columns:
        output[col] = output[col].astype(np.float32)
    
    return output


def compute_pit_fundamentals_for_ticker(
    *,
    ticker: str,
    facts_path: str,
    daily_dates: pd.DatetimeIndex,
    close_series: pd.Series | None = None,
) -> pd.DataFrame | None:
    """
    Convenience function to compute PIT fundamentals for a single ticker.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol for logging.
    facts_path : str
        Path to SEC companyfacts file.
    daily_dates : pd.DatetimeIndex
        Daily date index.
    close_series : pd.Series, optional
        Close prices for mktcap calculation.
    
    Returns
    -------
    pd.DataFrame or None
        Fundamental features DataFrame, or None if extraction fails.
    """
    try:
        facts = load_facts(facts_path)
        return compute_pit_fundamentals(
            daily_index=daily_dates,
            facts=facts,
            close_series=close_series,
        )
    except Exception as e:
        logger.warning(f"ALPHA_FACTS_SKIP:{ticker}:{str(e)[:50]}")
        return None


# ==============================================================================
# Task 10.2.1: SEC CompanyFacts Parser + PIT Snapshot Aligner (Required API)
# ==============================================================================


def extract_companyfacts_tag_entries(
    companyfacts_path: str,
    *,
    taxonomy: str,
    tag: str,
    unit_preference: str,
) -> pd.DataFrame:
    """
    Load a SEC companyfacts JSON file and extract entries for (taxonomy, tag, unit_preference).

    Returns a DataFrame with columns:
      - end (datetime64[ns])
      - filed (datetime64[ns])
      - val (float64)
    Sorted deterministically by (filed asc, end asc, val asc as tie-break).

    Fail-safe:
      - If file missing/corrupt or tag path missing, return empty DataFrame with the 3 columns.
      - Must NOT raise.
    """
    empty_df = pd.DataFrame(columns=["end", "filed", "val"])
    empty_df["end"] = pd.to_datetime(empty_df["end"])
    empty_df["filed"] = pd.to_datetime(empty_df["filed"])
    empty_df["val"] = empty_df["val"].astype(float)
    
    try:
        path_obj = Path(companyfacts_path)
        if not path_obj.exists():
            return empty_df
        
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            return empty_df
        
        facts = data.get("facts", {})
        taxonomy_data = facts.get(taxonomy, {})
        
        if not taxonomy_data:
            return empty_df
        
        tag_data = taxonomy_data.get(tag, {})
        units = tag_data.get("units", {})
        
        if not units:
            return empty_df
        
        # Try unit_preference first, then fallback to first available unit
        unit_entries = units.get(unit_preference)
        if unit_entries is None:
            # Fallback: use first available unit
            available_units = list(units.keys())
            if not available_units:
                return empty_df
            unit_entries = units.get(available_units[0], [])
        
        if not unit_entries:
            return empty_df
        
        records = []
        for entry in unit_entries:
            end_val = entry.get("end")
            filed_val = entry.get("filed")
            val = entry.get("val")
            
            if end_val and filed_val and val is not None:
                records.append({
                    "end": end_val,
                    "filed": filed_val,
                    "val": float(val),
                })
        
        if not records:
            return empty_df
        
        df = pd.DataFrame(records)
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        df = df.dropna(subset=["end", "filed"])
        
        if df.empty:
            return empty_df
        
        # Deterministic sort: filed asc, end asc, val asc for tie-break
        df = df.sort_values(["filed", "end", "val"], ascending=[True, True, True])
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception:
        return empty_df


def pit_latest_snapshot(
    entries: pd.DataFrame,
    as_of_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Compute PIT-safe latest snapshot series aligned to as_of_dates.

    Eligibility rule at date t:
      - Only rows with filed <= t are visible
      - Choose the row with maximum (end, filed) among visible rows
      - Return its val for date t

    Assumption:
      In SEC companyfacts, end <= filed generally holds. Enforce end <= filed by filtering.

    Output:
      - np.ndarray shape (len(as_of_dates),) dtype float
      - NaN when no visible row exists
    
    Implementation:
      - Vectorized using merge_asof, no per-date Python loops
      - Tracks cumulative "best-so-far" by (end, filed) key
    """
    n = len(as_of_dates)
    nan_result = np.full(n, np.nan, dtype=float)
    
    try:
        if entries.empty or n == 0:
            return nan_result
        
        # Validate required columns
        required_cols = {"end", "filed", "val"}
        if not required_cols.issubset(entries.columns):
            return nan_result
        
        df = entries.copy()
        
        # Enforce end <= filed to avoid edge-case leakage
        df = df[df["end"] <= df["filed"]].copy()
        
        if df.empty:
            return nan_result
        
        # Sort by filed asc, then end asc for determinism
        df = df.sort_values(["filed", "end"], ascending=[True, True]).reset_index(drop=True)
        
        # Create a composite key for "best-so-far" tracking: (end, filed)
        # We want the row with maximum (end, filed) tuple among rows filed so far
        # Track cumulative max (end, filed) and keep only rows where best updates
        
        # Convert to numeric for cummax tracking
        df["end_ord"] = df["end"].values.astype("datetime64[ns]").astype(np.int64)
        df["filed_ord"] = df["filed"].values.astype("datetime64[ns]").astype(np.int64)
        
        # Cumulative max of end_ord
        df["cummax_end"] = df["end_ord"].cummax()
        
        # For rows where end equals cummax, we use filed as secondary tie-breaker
        # Track rows where this row becomes the new "best" (max end, and if tied, max filed)
        best_end = df["cummax_end"].values
        end_ord = df["end_ord"].values
        filed_ord = df["filed_ord"].values
        
        # Build "best_filed_at_end" - for each row, the max filed among rows with end == cummax_end
        n_rows = len(df)
        keep = np.zeros(n_rows, dtype=bool)
        
        current_best_end = -1
        current_best_filed = -1
        
        for i in range(n_rows):
            is_new_best = False
            if end_ord[i] > current_best_end:
                # New best end
                is_new_best = True
                current_best_end = end_ord[i]
                current_best_filed = filed_ord[i]
            elif end_ord[i] == current_best_end and filed_ord[i] > current_best_filed:
                # Same end but newer filed
                is_new_best = True
                current_best_filed = filed_ord[i]
            
            if is_new_best:
                keep[i] = True
        
        df_best = df.loc[keep, ["filed", "val"]].copy()
        
        if df_best.empty:
            return nan_result
        
        # Now merge_asof from as_of_dates onto filed
        dates_df = pd.DataFrame({"date": as_of_dates})
        dates_df = dates_df.sort_values("date").reset_index(drop=True)
        
        df_best = df_best.rename(columns={"filed": "date"})
        df_best = df_best.sort_values("date").reset_index(drop=True)
        
        merged = pd.merge_asof(
            dates_df,
            df_best,
            on="date",
            direction="backward",
        )
        
        # Reorder to match original as_of_dates order
        merged = merged.set_index("date")
        result = merged["val"].reindex(as_of_dates).values
        
        return result.astype(float)
        
    except Exception:
        return nan_result


def compute_pit_fundamental_panel(
    *,
    companyfacts_path: str,
    as_of_dates: pd.DatetimeIndex,
    close: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a minimal PIT fundamental panel (aligned to as_of_dates) from companyfacts JSON.

    Required tags (use if available; missing allowed -> NaN):
      - Assets (us-gaap, USD)
      - Liabilities (us-gaap, USD)
      - StockholdersEquity (us-gaap, USD)
      - CashAndCashEquivalentsAtCarryingValue (us-gaap, USD) with fallback:
        CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents (us-gaap, USD)
      - Shares outstanding with priority:
        1) dei / EntityCommonStockSharesOutstanding
        2) us-gaap / CommonStockSharesOutstanding

    Derived columns:
      - assets, liabilities, equity, cash, shares_out
      - leverage, cash_to_assets, book_to_assets, mktcap

    Fail-safe:
      - Never raise; return empty DataFrame if anything fails.
    """
    # Deterministic column order
    output_cols = [
        "assets", "liabilities", "equity", "cash", "shares_out",
        "leverage", "cash_to_assets", "book_to_assets", "mktcap",
    ]
    
    empty_df = pd.DataFrame(index=as_of_dates, columns=output_cols, dtype=float)
    
    try:
        n = len(as_of_dates)
        if n == 0:
            return empty_df
        
        result = pd.DataFrame(index=as_of_dates)
        eps = 1e-9
        
        # Extract each required tag
        # Assets
        assets_entries = extract_companyfacts_tag_entries(
            companyfacts_path, taxonomy="us-gaap", tag="Assets", unit_preference="USD"
        )
        result["assets"] = pit_latest_snapshot(assets_entries, as_of_dates)
        
        # Liabilities
        liab_entries = extract_companyfacts_tag_entries(
            companyfacts_path, taxonomy="us-gaap", tag="Liabilities", unit_preference="USD"
        )
        result["liabilities"] = pit_latest_snapshot(liab_entries, as_of_dates)
        
        # StockholdersEquity
        equity_entries = extract_companyfacts_tag_entries(
            companyfacts_path, taxonomy="us-gaap", tag="StockholdersEquity", unit_preference="USD"
        )
        result["equity"] = pit_latest_snapshot(equity_entries, as_of_dates)
        
        # Cash: try primary tag, fallback to alternative
        cash_entries = extract_companyfacts_tag_entries(
            companyfacts_path, taxonomy="us-gaap", 
            tag="CashAndCashEquivalentsAtCarryingValue", unit_preference="USD"
        )
        cash_values = pit_latest_snapshot(cash_entries, as_of_dates)
        
        # Fallback if all NaN
        if np.all(np.isnan(cash_values)):
            cash_entries_alt = extract_companyfacts_tag_entries(
                companyfacts_path, taxonomy="us-gaap",
                tag="CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", 
                unit_preference="USD"
            )
            cash_values = pit_latest_snapshot(cash_entries_alt, as_of_dates)
        
        result["cash"] = cash_values
        
        # Shares outstanding: priority 1) dei, 2) us-gaap
        shares_entries = extract_companyfacts_tag_entries(
            companyfacts_path, taxonomy="dei", 
            tag="EntityCommonStockSharesOutstanding", unit_preference="shares"
        )
        shares_values = pit_latest_snapshot(shares_entries, as_of_dates)
        
        # Fallback to us-gaap if dei is all NaN
        if np.all(np.isnan(shares_values)):
            shares_entries_alt = extract_companyfacts_tag_entries(
                companyfacts_path, taxonomy="us-gaap",
                tag="CommonStockSharesOutstanding", unit_preference="shares"
            )
            shares_values = pit_latest_snapshot(shares_entries_alt, as_of_dates)
        
        result["shares_out"] = shares_values
        
        # Derived ratios
        assets_arr = result["assets"].values
        assets_safe = np.where(np.isnan(assets_arr) | (assets_arr == 0), eps, assets_arr)
        
        result["leverage"] = result["liabilities"].values / assets_safe
        result["cash_to_assets"] = result["cash"].values / assets_safe
        result["book_to_assets"] = result["equity"].values / assets_safe
        
        # Market cap (only if close provided)
        if close is not None and len(close) == n:
            result["mktcap"] = result["shares_out"].values * close
        else:
            result["mktcap"] = np.nan
        
        # Ensure column order
        result = result[output_cols]
        
        return result
        
    except Exception:
        return empty_df



# ==============================================================================
# SEC Fundamental Preprocessing for Coverage Enhancement
# ==============================================================================


SEC_FUNDAMENTAL_COLS = [
    "assets", "liabilities", "equity", "cash", "shares_out",
    "leverage", "cash_to_assets", "book_to_assets", "mktcap",
]


def preprocess_fundamentals_for_coverage(
    df: "pd.DataFrame",
    *,
    ffill: bool = True,
    fill_method: str = "median",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> "pd.DataFrame":
    """
    Preprocess SEC fundamental columns to improve coverage.
    
    PIT-safe preprocessing:
    1. Forward-fill within each ticker (only uses past values)
    2. Cross-sectional fill per date (median or zero)
    """
    if df.empty:
        return df
    
    sec_cols = [c for c in SEC_FUNDAMENTAL_COLS if c in df.columns]
    
    if not sec_cols:
        return df
    
    result = df.copy()
    
    # Step 1: Forward-fill within each ticker (PIT-safe)
    if ffill and ticker_col in result.columns:
        result = result.sort_values([ticker_col, date_col]).reset_index(drop=True)
        for col in sec_cols:
            result[col] = result.groupby(ticker_col)[col].ffill()
    
    # Step 2: Cross-sectional fill per date
    if fill_method == "median" and date_col in result.columns:
        for col in sec_cols:
            date_medians = result.groupby(date_col)[col].transform("median")
            result[col] = result[col].fillna(date_medians)
    elif fill_method == "zero":
        for col in sec_cols:
            result[col] = result[col].fillna(0.0)
    
    # Ensure float32 dtype
    for col in sec_cols:
        result[col] = result[col].astype("float32")
    
    return result
