"""
Build prices.csv from ALL Stooq files using parallel execution.
"""

import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allow running this script directly via `python scripts/build_universe_prices.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stooq_prices import load_stooq_daily_prices

STOOQ_DIR = "data/raw/stooq/us"
AS_OF_DATE = "2024-12-31"
START_DATE = "2010-01-01"
MIN_DAYS = 252 * 3  # At least 3 years of data
OUT_DIR = "data/backtest_universe_full"
LAST_DATE_MAX_GAP_DAYS = 10  # Drop tickers with stale/ended series


def load_single_ticker(args):
    """Load a single ticker file. Returns (ticker, series) or (ticker, None)."""
    fname, stooq_dir, as_of_date, start_date, min_days = args
    path = os.path.join(stooq_dir, fname)
    ticker = fname.replace(".us.txt", "").upper()
    
    try:
        df = load_stooq_daily_prices(path, as_of_date=as_of_date)
        series = df.set_index("date")["close"]
        
        # Filter by start_date
        series = series[series.index >= start_date]
        
        as_of_dt = pd.to_datetime(as_of_date, format="%Y-%m-%d")
        last_dt = series.index.max()
        is_recent = (as_of_dt - last_dt) <= pd.Timedelta(days=LAST_DATE_MAX_GAP_DAYS)

        if len(series) >= min_days and (series > 0).all() and is_recent:
            return (ticker, series.to_dict())
        return (ticker, None)
    except Exception as e:
        return (ticker, None)


def main():
    # Get all files
    files = [f for f in os.listdir(STOOQ_DIR) if f.endswith(".txt")]
    print(f"Total Stooq files: {len(files)}")
    
    # Prepare args
    args_list = [(f, STOOQ_DIR, AS_OF_DATE, START_DATE, MIN_DAYS) for f in files]
    
    # Parallel load
    all_series = {}
    loaded = 0
    errors = 0
    
    print(f"Loading with parallel workers...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_single_ticker, args): args[0] for args in args_list}
        
        for i, future in enumerate(as_completed(futures)):
            ticker, data = future.result()
            if data is not None:
                all_series[ticker] = pd.Series(data)
                loaded += 1
            else:
                errors += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/{len(files)}, loaded={loaded}, errors={errors}")
    
    print(f"\nSuccessfully loaded: {loaded} tickers")
    print(f"Errors/filtered: {errors}")
    
    if not all_series:
        print("No data loaded!")
        return
    
    # Build wide DataFrame
    print("Building wide DataFrame...")
    prices_wide = pd.DataFrame(all_series)
    prices_wide.index = pd.to_datetime(prices_wide.index)
    prices_wide = prices_wide.sort_index()
    prices_wide = prices_wide[sorted(prices_wide.columns)]
    
    print(f"Raw shape: {prices_wide.shape}")
    print(f"Date range: {prices_wide.index.min()} to {prices_wide.index.max()}")
    
    # NaN 처리: 90% 이상 커버리지
    coverage = prices_wide.notna().sum() / len(prices_wide)
    good_tickers = coverage[coverage >= 0.9].index.tolist()
    print(f"Tickers with >=90% coverage: {len(good_tickers)}")
    
    if len(good_tickers) < 2:
        # Try 50% coverage
        good_tickers = coverage[coverage >= 0.5].index.tolist()
        print(f"Fallback to >=50% coverage: {len(good_tickers)}")
    
    prices_wide = prices_wide[good_tickers]
    
    # Avoid collapsing to a tiny intersection across thousands of tickers.
    # Forward-fill is PIT-safe (no look-ahead) and preserves daily rows.
    prices_wide = prices_wide.ffill()

    # Drop initial rows where some tickers have not started yet (still NaN after ffill)
    prices_wide = prices_wide.dropna(how="any")

    if prices_wide.isna().any().any():
        raise ValueError("Unexpected NaN remains in prices after forward-fill + dropna")
    
    print(f"After ffill+dropna: {prices_wide.shape}")
    
    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "prices.csv")
    
    # Manual save to avoid validation issues
    df_out = prices_wide.reset_index()
    df_out = df_out.rename(columns={df_out.columns[0]: "date"})
    df_out.to_csv(out_path, index=False)
    
    print(f"\nSaved to: {out_path}")
    print(f"Final: {prices_wide.shape[0]} days × {prices_wide.shape[1]} tickers")
    
    # Generate scores
    print("\n=== Generating momentum scores ===")
    monthly = prices_wide.resample("ME").last()
    momentum = monthly.pct_change(12).dropna()
    
    scores_path = os.path.join(OUT_DIR, "scores.csv")
    scores_out = momentum.reset_index()
    scores_out = scores_out.rename(columns={scores_out.columns[0]: "date"})
    scores_out.to_csv(scores_path, index=False)
    
    print(f"Scores: {momentum.shape}")
    print(f"Saved to: {scores_path}")
    
    print("\n" + "=" * 50)
    print("COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
