"""
Build alpha dataset for XGBoost ranking model.

Reads per-ticker OHLCV files, calculates features and targets,
and exports a single memory-optimized dataset file.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.alpha_features import add_alpha_features, add_alpha_targets
from src.sec_fundamentals import compute_pit_fundamental_panel
# V2.3: Import canonical layer for low_confidence filtering support
from src.sec_fundamentals_v2 import build_canonical_wide_table, load_tag_mapping

logger = logging.getLogger(__name__)

# Single source of truth for SEC fundamental columns
SEC_FUNDAMENTAL_COLS = [
    "assets", "liabilities", "equity", "cash", "shares_out",
    "leverage", "cash_to_assets", "book_to_assets", "mktcap",
]

# V2.3 canonical columns (from sec_tag_mapping.yaml)
SEC_V2_CANONICAL_COLS = [
    "total_assets", "total_liabilities", "stockholders_equity", 
    "revenues", "net_income", "operating_cash_flow", "shares_outstanding",
]


def _normalize_ticker(ticker: str) -> str:
    """Normalize ticker to uppercase, strip '.US' suffix."""
    t = str(ticker).strip().upper()
    if t.endswith(".US"):
        t = t[:-3]
    return t


def _parse_ohlcv_file(filepath: Path) -> pd.DataFrame | None:
    """
    Parse a single OHLCV file (stooq or simple format).
    
    Returns DataFrame with columns [date, open, high, low, close, volume] or None on failure.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"ALPHA_DATASET_SKIP:{filepath.stem}:read_error:{str(e)[:50]}")
        return None
    
    if df.empty:
        logger.warning(f"ALPHA_DATASET_SKIP:{filepath.stem}:empty_file")
        return None
    
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Detect format - stooq uses <DATE>, simple uses date
    date_col = None
    for candidate in ["date", "<date>"]:
        if candidate in df.columns:
            date_col = candidate
            break
    
    if date_col is None:
        logger.warning(f"ALPHA_DATASET_SKIP:{filepath.stem}:no_date_column")
        return None
    
    # Rename columns to standard names
    col_map = {
        date_col: "date",
        "<open>": "open",
        "<high>": "high",
        "<low>": "low",
        "<close>": "close",
        "<vol>": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vol": "volume",
    }
    
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"ALPHA_DATASET_SKIP:{filepath.stem}:missing_columns:{missing}")
        return None
    
    # Parse date - handle both YYYYMMDD integer format and standard formats
    date_col_raw = df["date"]
    # Check if it's numeric (Stooq uses YYYYMMDD as integer like 19840907)
    if pd.api.types.is_numeric_dtype(date_col_raw):
        df["date"] = pd.to_datetime(date_col_raw.astype(int).astype(str), format="%Y%m%d", errors="coerce")
    else:
        # Try to parse as string - might be YYYYMMDD string or ISO format
        first_val = str(date_col_raw.iloc[0])
        if len(first_val) == 8 and first_val.isdigit():
            df["date"] = pd.to_datetime(date_col_raw.astype(str), format="%Y%m%d", errors="coerce")
        else:
            df["date"] = pd.to_datetime(date_col_raw, errors="coerce")
    
    df = df.dropna(subset=["date"])
    
    if df.empty:
        logger.warning(f"ALPHA_DATASET_SKIP:{filepath.stem}:no_valid_dates")
        return None
    
    # Ensure volume exists (fill with 0 if missing)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    
    # Select and order columns
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    
    # Convert to numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["open", "high", "low", "close"])
    
    if df.empty:
        logger.warning(f"ALPHA_DATASET_SKIP:{filepath.stem}:no_valid_ohlc")
        return None
    
    return df


def _load_manifest(manifest_path: str) -> dict[str, str]:
    """
    Load manifest CSV and return ticker -> companyfacts_path mapping.
    
    Parameters
    ----------
    manifest_path : str
        Path to manifest CSV with required columns:
        - ticker
        - companyfacts_status
        - companyfacts_path
    
    Returns
    -------
    dict[str, str]
        Ticker to companyfacts file path mapping (only status='ok' and non-empty path)
    
    Raises
    ------
    ValueError
        If manifest missing required columns (fail-fast on user input contract error).
    """
    df = pd.read_csv(manifest_path)
    
    # Validate required columns (strict contract per Task 10.2.2)
    required_cols = {"ticker", "companyfacts_status", "companyfacts_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest CSV missing required columns: {missing}")
    
    # Filter: only status == 'ok' AND non-empty path
    df = df[df["companyfacts_status"] == "ok"].copy()
    
    # Build mapping (normalize ticker)
    mapping = {}
    for _, row in df.iterrows():
        ticker = _normalize_ticker(row["ticker"])
        path = row["companyfacts_path"]
        if pd.notna(path) and str(path).strip():
            mapping[ticker] = str(path)
    
    return mapping


def build_alpha_dataset(
    data_dir: str,
    output_path: str,
    as_of_date: str,
    min_price: float = 5.0,
    min_volume: float = 1_000_000,
    manifest_csv: str | None = None,
    exclude_low_confidence: bool = False,
    use_v2_fundamentals: bool = True,
) -> None:
    """
    Build alpha dataset from per-ticker OHLCV files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing per-ticker files (*.txt, *.csv)
    output_path : str
        Output file path (csv.gz or parquet)
    as_of_date : str
        PIT cutoff date (YYYY-MM-DD) - no rows with date > as_of_date
    min_price : float
        Minimum close price filter (default 5.0)
    min_volume : float
        Minimum volume filter (default 1,000,000)
    manifest_csv : str | None
        Optional manifest CSV with ticker->companyfacts_path mapping
        for SEC fundamental features integration.
    exclude_low_confidence : bool
        If True, NaN-out SEC values that came from low_confidence sources
        (computed fallbacks, dimension backoff). Default False.
    use_v2_fundamentals : bool
        If True, use V2.3 canonical layer. Default True.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"data_dir does not exist: {data_dir}")
    
    # Load SEC manifest if provided
    sec_manifest: dict[str, str] = {}
    if manifest_csv:
        manifest_file = Path(manifest_csv)
        if manifest_file.exists():
            sec_manifest = _load_manifest(manifest_csv)
            logger.info(f"Loaded SEC manifest with {len(sec_manifest)} tickers")
        else:
            logger.warning(f"Manifest CSV not found: {manifest_csv}")

    
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Find all ticker files
    files = list(data_path.glob("*.txt")) + list(data_path.glob("*.csv"))
    if not files:
        raise ValueError(f"No .txt or .csv files found in: {data_dir}")
    
    all_dfs: list[pd.DataFrame] = []
    skipped_count = 0
    processed_count = 0
    
    for filepath in sorted(files):
        ticker = _normalize_ticker(filepath.stem)
        
        df = _parse_ohlcv_file(filepath)
        if df is None:
            skipped_count += 1
            continue
        
        # PIT cutoff
        df = df[df["date"] <= as_of_dt].copy()
        if df.empty:
            skipped_count += 1
            continue
        
        # Price and volume filters
        df = df[df["close"] > min_price].copy()
        if min_volume > 0 and "volume" in df.columns:
            df = df[df["volume"] >= min_volume].copy()
        
        if len(df) < 100:
            # Not enough data for feature calculation
            logger.warning(f"ALPHA_DATASET_SKIP:{ticker}:insufficient_rows:{len(df)}")
            skipped_count += 1
            continue
        
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        
        # Add features and targets
        df = add_alpha_features(df)
        df = add_alpha_targets(df)
        
        # Add SEC fundamental features if manifest provided
        if ticker in sec_manifest:
            facts_path = sec_manifest[ticker]
            # Manifest paths are project-root-relative; resolve against PROJECT_ROOT
            if not Path(facts_path).is_absolute():
                facts_path = str(Path(PROJECT_ROOT) / facts_path)
            
            try:
                # Create daily index from df dates
                daily_dates = pd.DatetimeIndex(df["date"])
                close_arr = df["close"].values.astype(float)
                
                if use_v2_fundamentals:
                    # V2.3 API: Use canonical layer with low_confidence support
                    sec_df = build_canonical_wide_table(
                        companyfacts_path=facts_path,
                        as_of_dates=daily_dates,
                        include_provenance=False,
                        include_confidence_flags=True,
                    )
                    
                    if sec_df is not None and not sec_df.empty:
                        # Apply low_confidence filtering if requested
                        for col in SEC_V2_CANONICAL_COLS:
                            if col in sec_df.columns:
                                if exclude_low_confidence:
                                    conf_col = f"{col}_low_confidence"
                                    if conf_col in sec_df.columns:
                                        # NaN-out low confidence values
                                        mask = sec_df[conf_col].fillna(False).astype(bool)
                                        sec_df.loc[mask, col] = np.nan
                                df[col] = sec_df[col].values.astype(np.float32)
                else:
                    # V1 API (legacy)
                    sec_df = compute_pit_fundamental_panel(
                        companyfacts_path=facts_path,
                        as_of_dates=daily_dates,
                        close=close_arr,
                    )
                    
                    if sec_df is not None and not sec_df.empty:
                        for col in sec_df.columns:
                            df[col] = sec_df[col].values.astype(np.float32)
            except Exception as e:
                logger.warning(f"ALPHA_FACTS_SKIP:{ticker}:{str(e)[:50]}")
        
        # Drop rows where ANY target is NaN
        target_cols = [c for c in df.columns if c.startswith("fwd_ret_")]
        df = df.dropna(subset=target_cols)
        
        if df.empty:
            skipped_count += 1
            continue
        
        # Add ticker column
        df["ticker"] = ticker
        
        all_dfs.append(df)
        processed_count += 1
    
    if not all_dfs:
        raise ValueError("No valid ticker data after processing")
    
    # Concatenate all tickers
    result = pd.concat(all_dfs, ignore_index=True)
    
    # Deterministic sort by (ticker, date) for groupby ffill
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Task 10.2.2.1: Apply ticker-level forward-fill for SEC fundamental columns ONLY
    # PIT-safe: only uses past values within each ticker
    # Support both V1 and V2.3 column names
    all_sec_cols = list(set(SEC_FUNDAMENTAL_COLS) | set(SEC_V2_CANONICAL_COLS))
    sec_cols_present = [c for c in all_sec_cols if c in result.columns]
    for col in sec_cols_present:
        result[col] = result.groupby("ticker")[col].ffill()
    
    # Task 10.2.2.1: Add missing indicator columns (after ffill)
    # Indicator = 1.0 if value is still NaN after ffill, else 0.0
    for col in sec_cols_present:
        result[f"{col}_is_missing"] = result[col].isna().astype(np.float32)
    
    # Cast numeric columns to float32 (including SEC cols and indicators)
    exclude_cols = {"date", "ticker"}
    for col in result.columns:
        if col not in exclude_cols:
            result[col] = result[col].astype(np.float32)
    
    # Final deterministic sort by (date, ticker)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Ensure (date, ticker) uniqueness
    if result.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Duplicate (date, ticker) pairs found - data integrity issue")
    
    # Reorder columns: date, ticker, OHLCV, features, other (including indicators), targets
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    feature_cols = ["vol_20d", "mom_5d", "mom_21d", "mom_63d", "rsi_14d", "bbands_20d", "atr_14d_norm"]
    target_cols = [c for c in result.columns if c.startswith("fwd_ret_")]
    # Sort other_cols to ensure determinism (SEC cols + indicators)
    other_cols = sorted([c for c in result.columns if c not in ["date", "ticker"] + ohlcv_cols + feature_cols + target_cols])
    
    ordered_cols = ["date", "ticker"] + ohlcv_cols + feature_cols + other_cols + sorted(target_cols)
    ordered_cols = [c for c in ordered_cols if c in result.columns]
    result = result[ordered_cols]
    
    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if str(output_path).endswith(".parquet"):
        try:
            result.to_parquet(output_path, index=False)
            print(f"Wrote: {output_path}")
        except Exception as e:
            logger.warning(f"ALPHA_DATASET_FALLBACK:parquet_unavailable:{str(e)[:50]}")
            fallback_path = str(output_path).replace(".parquet", ".csv.gz")
            result.to_csv(fallback_path, index=False, compression="gzip")
            print(f"Wrote: {fallback_path} (parquet fallback)")
    else:
        # Default: csv.gz
        if not str(output_path).endswith(".csv.gz"):
            output_path = str(output_path) + ".csv.gz"
        result.to_csv(output_path, index=False, compression="gzip")
        print(f"Wrote: {output_path}")
    
    print(f"Summary: {processed_count} tickers processed, {skipped_count} skipped, {len(result)} rows")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build alpha dataset for XGBoost ranking")
    parser.add_argument("--data-dir", required=True, help="Directory with per-ticker OHLCV files")
    parser.add_argument("--output-path", required=True, help="Output file path (csv.gz or parquet)")
    parser.add_argument("--as-of-date", required=True, help="PIT cutoff date (YYYY-MM-DD)")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum close price (default: 5.0)")
    parser.add_argument("--min-volume", type=float, default=1_000_000, help="Minimum volume (default: 1M)")
    parser.add_argument("--manifest-csv", type=str, default=None, help="Optional SEC manifest CSV with ticker->companyfacts_path")
    # V2.3: A/B testing options
    parser.add_argument("--exclude-low-confidence", action="store_true", 
                       help="NaN-out SEC values from low_confidence sources (computed fallbacks)")
    parser.add_argument("--use-v1-fundamentals", action="store_true",
                       help="Use legacy V1 API instead of V2.3 canonical layer")
    
    args = parser.parse_args()
    
    try:
        build_alpha_dataset(
            data_dir=args.data_dir,
            output_path=args.output_path,
            as_of_date=args.as_of_date,
            min_price=args.min_price,
            min_volume=args.min_volume,
            manifest_csv=args.manifest_csv,
            exclude_low_confidence=args.exclude_low_confidence,
            use_v2_fundamentals=not args.use_v1_fundamentals,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
