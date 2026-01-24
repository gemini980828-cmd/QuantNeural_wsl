#!/usr/bin/env python3
"""
Generate initial ops_snapshots_daily from existing prices_daily data.
Run once to create the first snapshot for UI testing.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


def main():
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError("Supabase credentials not found")
    
    client = create_client(url, key)
    
    print("=" * 60)
    print("Generate Initial Ops Snapshot")
    print("=" * 60)
    
    # 1. Get latest trading date from QQQ
    result = client.table("prices_daily")\
        .select("date, close")\
        .eq("symbol", "QQQ")\
        .order("date", desc=True)\
        .limit(220)\
        .execute()
    
    if not result.data or len(result.data) < 170:
        print(f"⚠️ Insufficient data: {len(result.data) if result.data else 0} rows")
        return
    
    qqq_data = result.data
    verdict_date = qqq_data[0]["date"]
    closes = [float(r["close"]) for r in qqq_data]
    
    print(f"📅 Verdict date: {verdict_date}")
    print(f"📊 QQQ closes: {len(closes)} rows")
    
    # 2. Calculate SMAs
    def sma(data, period):
        return sum(data[:period]) / period if len(data) >= period else None
    
    sma3 = sma(closes, 3)
    sma160 = sma(closes, 160)
    sma165 = sma(closes, 165)
    sma170 = sma(closes, 170)
    
    print(f"\n📈 SMA3: {sma3:.2f}")
    print(f"📈 SMA160: {sma160:.2f}")
    print(f"📈 SMA165: {sma165:.2f}")
    print(f"📈 SMA170: {sma170:.2f}")
    
    # 3. Determine verdict (2/3 majority)
    votes = [
        1 if sma3 > sma160 else 0,
        1 if sma3 > sma165 else 0,
        1 if sma3 > sma170 else 0,
    ]
    verdict = "ON" if sum(votes) >= 2 else "OFF10"
    
    print(f"\n🎯 Verdict: {verdict} (votes: {votes})")
    
    # 4. Get prices for all tickers
    prices_result = client.table("prices_daily")\
        .select("symbol, close")\
        .eq("date", verdict_date)\
        .execute()
    
    prices = {r["symbol"]: float(r["close"]) for r in prices_result.data}
    print(f"\n💰 Prices: {prices}")
    
    # 5. Find next trading day (estimate)
    from datetime import timedelta
    d = datetime.strptime(verdict_date, "%Y-%m-%d")
    d += timedelta(days=1)
    while d.weekday() >= 5:  # Skip weekends
        d += timedelta(days=1)
    execution_date = d.strftime("%Y-%m-%d")
    
    print(f"📅 Execution date: {execution_date}")
    
    # 6. Build payload
    payload = {
        "prices": prices,
        "sma": {
            "sma3": round(sma3, 4),
            "sma160": round(sma160, 4),
            "sma165": round(sma165, 4),
            "sma170": round(sma170, 4),
        },
        "verdict": verdict,
        "sourceMeta": {
            "source": "stooq",
            "lastTradingDate": verdict_date,
        },
    }
    
    # 7. Upsert snapshot
    client.table("ops_snapshots_daily").upsert({
        "verdict_date": verdict_date,
        "execution_date": execution_date,
        "health": "FRESH",
        "payload_json": payload,
        "computed_at": datetime.now().isoformat(),
    }).execute()
    
    print("\n" + "=" * 60)
    print("✅ Snapshot created!")
    print("=" * 60)


if __name__ == "__main__":
    main()
