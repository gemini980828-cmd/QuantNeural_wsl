"""
Fast parallel alpha dataset builder using ThreadPoolExecutor.
"""

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.alpha_features import add_alpha_features, add_alpha_targets
from src.sec_fundamentals_v2 import build_canonical_wide_table


def normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.endswith(".US"):
        t = t[:-3]
    return t


def process_single_ticker(args):
    """Process a single ticker file. Returns DataFrame or None."""
    filepath, as_of_dt, min_price, min_volume, manifest_lookup = args
    
    ticker_raw = filepath.stem
    # Handle .csv.gz, .txt.gz
    if ticker_raw.endswith('.csv') or ticker_raw.endswith('.txt'):
        ticker_raw = ticker_raw[:-4]
    
    ticker = normalize_ticker(ticker_raw)
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Find date column
        date_col = None
        for candidate in ["date", "<date>"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            return None
        
        # Rename columns
        col_map = {
            date_col: "date", "<open>": "open", "<high>": "high", 
            "<low>": "low", "<close>": "close", "<vol>": "volume",
            "open": "open", "high": "high", "low": "low", 
            "close": "close", "volume": "volume", "vol": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        required = ["date", "open", "high", "low", "close"]
        if any(c not in df.columns for c in required):
            return None
        
        # Parse date
        date_col_raw = df["date"]
        if pd.api.types.is_numeric_dtype(date_col_raw):
            df["date"] = pd.to_datetime(date_col_raw.astype(int).astype(str), format="%Y%m%d", errors="coerce")
        else:
            first_val = str(date_col_raw.iloc[0])
            if len(first_val) == 8 and first_val.isdigit():
                df["date"] = pd.to_datetime(date_col_raw.astype(str), format="%Y%m%d", errors="coerce")
            else:
                df["date"] = pd.to_datetime(date_col_raw, errors="coerce")
        
        df = df.dropna(subset=["date"])
        if df.empty:
            return None
        
        # PIT cutoff
        df = df[df["date"] <= as_of_dt].copy()
        if df.empty:
            return None
        
        # Volume
        if "volume" not in df.columns:
            df["volume"] = 0.0
        
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["open", "high", "low", "close"])
        
        # Filters
        df = df[df["close"] > min_price].copy()
        if min_volume > 0:
            df = df[df["volume"] >= min_volume].copy()
        
        if len(df) < 100:
            return None
        
        df = df.sort_values("date").reset_index(drop=True)
        
        # Features and targets
        df = add_alpha_features(df)
        df = add_alpha_targets(df)
        
        # SEC fundamentals
        if ticker_raw in manifest_lookup:
            facts_path = manifest_lookup[ticker_raw]
            if not Path(facts_path).is_absolute():
                facts_path = str(Path(PROJECT_ROOT) / facts_path)
            
            try:
                daily_dates = pd.DatetimeIndex(df["date"])
                sec_df = build_canonical_wide_table(
                    companyfacts_path=facts_path,
                    as_of_dates=daily_dates,
                    include_provenance=False,
                    include_confidence_flags=True,
                )
                
                if sec_df is not None and not sec_df.empty:
                    v2_cols = ['total_assets', 'total_liabilities', 'stockholders_equity',
                              'revenues', 'net_income', 'operating_cash_flow', 'shares_outstanding']
                    for col in v2_cols:
                        if col in sec_df.columns:
                            df[col] = sec_df[col].values.astype(np.float32)
            except:
                pass
        
        # Drop NaN targets
        target_cols = [c for c in df.columns if c.startswith("fwd_ret_")]
        df = df.dropna(subset=target_cols)
        if df.empty:
            return None
        
        df["ticker"] = ticker
        return df
        
    except:
        return None


def build_fast(
    data_dir: str,
    output_path: str,
    as_of_date: str,
    min_price: float = 5.0,
    min_volume: float = 1_000_000,
    manifest_csv: str = None,
    max_workers: int = 8,
    manifest_only: bool = False,
):
    """Build alpha dataset with parallel processing."""
    
    data_path = Path(data_dir)
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Load manifest
    manifest_lookup = {}
    manifest_tickers = set()
    if manifest_csv and Path(manifest_csv).exists():
        mdf = pd.read_csv(manifest_csv)
        for _, row in mdf.iterrows():
            if row.get('companyfacts_status') == 'ok' and pd.notna(row.get('companyfacts_path')):
                manifest_lookup[str(row['ticker'])] = str(row['companyfacts_path'])
                manifest_tickers.add(str(row['ticker']))
        print(f"Loaded manifest: {len(manifest_lookup)} tickers")
    
    # Find files
    files = list(data_path.glob("*.txt")) + list(data_path.glob("*.csv"))
    
    # Filter to manifest-only if requested (MUCH faster for FUND builds)
    if manifest_only and manifest_tickers:
        filtered_files = []
        for f in files:
            ticker = f.stem.lower()
            if ticker in manifest_tickers:
                filtered_files.append(f)
        print(f"Filtered: {len(files)} -> {len(filtered_files)} files (manifest-only mode)")
        files = filtered_files
    
    print(f"Found {len(files)} files to process")
    
    # Prepare args
    args_list = [(f, as_of_dt, min_price, min_volume, manifest_lookup) for f in files]
    
    # Parallel process
    all_dfs = []
    processed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_ticker, args): args for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_dfs.append(result)
                processed += 1
            else:
                skipped += 1
            
            # Progress
            if (processed + skipped) % 500 == 0:
                print(f"Progress: {processed + skipped}/{len(files)} ({processed} ok, {skipped} skip)")
    
    if not all_dfs:
        raise ValueError("No valid data")
    
    # Combine
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # FFill and missing indicators
    v2_cols = ['total_assets', 'total_liabilities', 'stockholders_equity',
              'revenues', 'net_income', 'operating_cash_flow', 'shares_outstanding']
    sec_cols = [c for c in v2_cols if c in result.columns]
    
    for col in sec_cols:
        result[col] = result.groupby("ticker")[col].ffill()
        result[f"{col}_is_missing"] = result[col].isna().astype(np.float32)
    
    # Cast to float32
    for col in result.columns:
        if col not in ["date", "ticker"]:
            result[col] = result[col].astype(np.float32)
    
    # Final sort
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Write
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, compression="gzip")
    
    print(f"Wrote: {output_path}")
    print(f"Summary: {processed} tickers, {skipped} skipped, {len(result)} rows")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-volume", type=float, default=1_000_000)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--manifest-only", action="store_true", help="Only process tickers in manifest (5x faster)")
    
    args = parser.parse_args()
    
    build_fast(
        data_dir=args.data_dir,
        output_path=args.output_path,
        as_of_date=args.as_of_date,
        min_price=args.min_price,
        min_volume=args.min_volume,
        manifest_csv=args.manifest_csv,
        max_workers=args.max_workers,
        manifest_only=args.manifest_only,
    )


if __name__ == "__main__":
    main()
