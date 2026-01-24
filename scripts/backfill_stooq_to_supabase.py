#!/usr/bin/env python3
"""
Stooq Backfill Script for 200TQ Dashboard.

One-time script to load historical price data from local Stooq files
into Supabase prices_daily table.

Usage:
    python scripts/backfill_stooq_to_supabase.py

Requirements:
    pip install supabase pandas python-dotenv
    
Environment:
    SUPABASE_URL=https://xxx.supabase.co
    SUPABASE_SERVICE_ROLE_KEY=xxx
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from src.stooq_prices import load_stooq_daily_prices

# Configuration
# Stooq ticker mappings:
# - SPLG renamed to SPYM in Stooq
# - SGOV not available, using BIL (similar T-Bill ETF)
TICKERS = {
    "TQQQ": "tqqq.us.txt",
    "QQQ": "qqq.us.txt",
    "SPLG": "spym.us.txt",   # SPLG→SPYM in Stooq (store as SPLG)
    "SGOV": "bil.us.txt",    # SGOV→BIL replacement (store as SGOV for UI compatibility)
}
STOOQ_DIR = Path(__file__).parent.parent / "data" / "raw" / "stooq" / "us"
BATCH_SIZE = 1000


def get_supabase_client() -> Client:
    """Create Supabase client from environment variables."""
    load_dotenv()
    
    # Try service role first, fall back to anon key (RLS is disabled on target tables)
    url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL/NEXT_PUBLIC_SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY must be set"
        )
    
    return create_client(url, key)


def standardize_symbol(source_symbol: str) -> str:
    """Convert TQQQ.US -> TQQQ"""
    return source_symbol.replace(".US", "").upper()


def load_and_transform(symbol: str, filename: str) -> pd.DataFrame:
    """Load Stooq file and transform for upsert."""
    filepath = STOOQ_DIR / filename
    
    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath}")
        return pd.DataFrame()
    
    # Use today as PIT cutoff (load all available data)
    as_of_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        df = load_stooq_daily_prices(str(filepath), as_of_date=as_of_date)
    except ValueError as e:
        print(f"  ⚠️  Error loading {filename}: {e}")
        return pd.DataFrame()
    
    # Transform for Supabase
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["symbol"] = symbol
    df["source_symbol"] = df["ticker"]
    df["source"] = "stooq"
    df["fetched_at"] = datetime.now().isoformat()
    df["volume"] = df["volume"].fillna(0).astype(int)  # Cast to int for BIGINT column
    
    # Select columns for insert
    df = df[["date", "symbol", "open", "high", "low", "close", "volume", 
             "source", "source_symbol", "fetched_at"]]
    
    return df


def upsert_batch(client: Client, records: list[dict]) -> int:
    """Upsert a batch of records to prices_daily."""
    if not records:
        return 0
    
    # Supabase upsert with conflict on (date, symbol)
    result = client.table("prices_daily").upsert(
        records,
        on_conflict="date,symbol"
    ).execute()
    
    return len(result.data) if result.data else 0


def main():
    print("=" * 60)
    print("Stooq Backfill to Supabase")
    print("=" * 60)
    
    client = get_supabase_client()
    total_inserted = 0
    
    for symbol, filename in TICKERS.items():
        print(f"\n📊 Processing {symbol}...")
        
        df = load_and_transform(symbol, filename)
        
        if df.empty:
            continue
        
        print(f"  📅 Date range: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  📈 Rows: {len(df)}")
        
        # Upsert in batches
        records = df.to_dict(orient="records")
        batch_count = 0
        
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            inserted = upsert_batch(client, batch)
            batch_count += 1
            total_inserted += inserted
            print(f"  ✅ Batch {batch_count}: {inserted} rows upserted")
    
    print("\n" + "=" * 60)
    print(f"✅ Backfill complete! Total: {total_inserted} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
