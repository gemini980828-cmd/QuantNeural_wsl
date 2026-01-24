#!/usr/bin/env python3
"""
Download and ingest latest Stooq data from web.
One-time script to update prices_daily with data newer than local files.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# Ticker mapping: symbol -> stooq ticker
TICKERS = {
    "TQQQ": "tqqq.us",
    "QQQ": "qqq.us",
    "SPLG": "spym.us",  # SPLG → SPYM
    "SGOV": "bil.us",   # SGOV → BIL
}
STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}&i=d"
BATCH_SIZE = 500


def get_supabase_client() -> Client:
    load_dotenv()
    url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials not found")
    return create_client(url, key)


def get_max_date(client: Client, symbol: str) -> str | None:
    """Get latest date in prices_daily for symbol."""
    result = client.table("prices_daily")\
        .select("date")\
        .eq("symbol", symbol)\
        .order("date", desc=True)\
        .limit(1)\
        .execute()
    
    if result.data and len(result.data) > 0:
        return result.data[0]["date"]
    return None


def download_stooq(stooq_ticker: str) -> pd.DataFrame:
    """Download CSV from Stooq."""
    url = STOOQ_URL.format(ticker=stooq_ticker)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    
    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.lower() for c in df.columns]
    return df


def main():
    print("=" * 60)
    print("Stooq Web Download & Incremental Ingest")
    print("=" * 60)
    
    client = get_supabase_client()
    total_inserted = 0
    
    for symbol, stooq_ticker in TICKERS.items():
        print(f"\n📊 {symbol} ({stooq_ticker})...")
        
        # Get current max date
        max_date = get_max_date(client, symbol)
        print(f"  📅 Current max date: {max_date or 'None'}")
        
        # Download from Stooq
        try:
            df = download_stooq(stooq_ticker)
        except Exception as e:
            print(f"  ⚠️  Download failed: {e}")
            continue
        
        print(f"  📥 Downloaded: {len(df)} rows ({df['date'].min()} ~ {df['date'].max()})")
        
        # Filter to new rows only
        if max_date:
            df = df[df["date"] > max_date].copy()
        
        if df.empty:
            print("  ✅ Already up to date")
            continue
        
        print(f"  🆕 New rows: {len(df)} ({df['date'].min()} ~ {df['date'].max()})")
        
        # Transform
        df["symbol"] = symbol
        df["source"] = "stooq"
        df["fetched_at"] = datetime.now().isoformat()
        df["volume"] = df["volume"].fillna(0).astype(int)
        
        records = df[["date", "symbol", "open", "high", "low", "close", "volume", "source", "fetched_at"]].to_dict(orient="records")
        
        # Upsert in batches
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            result = client.table("prices_daily").upsert(batch, on_conflict="date,symbol").execute()
            inserted = len(result.data) if result.data else 0
            total_inserted += inserted
            print(f"  ✅ Batch: {inserted} rows upserted")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Total new rows: {total_inserted}")
    print("=" * 60)


if __name__ == "__main__":
    main()
