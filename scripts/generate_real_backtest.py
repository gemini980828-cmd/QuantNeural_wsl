"""
Generate Real Data Backtest Inputs.

Uses Stooq price data and generates momentum-based scores for a backtest.
Outputs:
- data/backtest_real/prices.csv (wide format)
- data/backtest_real/scores.csv (wide format)

Then runs CLI backtest for both Monthly and Quarterly rebalancing.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from src.stooq_prices import load_stooq_daily_prices
from src.backtest_artifacts_io import write_prices_csv, write_scores_csv


def main():
    print("=" * 60)
    print("Real Data Backtest Input Generation")
    print("=" * 60)
    
    # ========================================
    # 1) Load configuration
    # ========================================
    with open("configs/real_data_experiment_1.json") as f:
        config = json.load(f)
    
    tickers_in_order = config["labels"]["tickers_in_order"]
    stooq_paths = config["labels"]["stooq_csv_by_ticker"]
    as_of_date = config["dates"]["as_of_date"]
    
    print(f"\nTickers: {tickers_in_order}")
    print(f"As-of date: {as_of_date}")
    
    # ========================================
    # 2) Build prices DataFrame (wide format)
    # ========================================
    print("\n=== Loading prices from Stooq...")
    
    all_prices = []
    for ticker in tickers_in_order:
        path = stooq_paths[ticker]
        print(f"  Loading {ticker} from {path}...")
        
        try:
            df = load_stooq_daily_prices(path, as_of_date=as_of_date, ticker=ticker.split(".")[0].upper() + ".US")
            # Actually the file has ticker as "AAPL.US" etc
            df = load_stooq_daily_prices(path, as_of_date=as_of_date)
            all_prices.append(df)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    if not all_prices:
        print("ERROR: No price data loaded!")
        return 1
    
    prices_long = pd.concat(all_prices, ignore_index=True)
    print(f"\nLoaded {len(prices_long)} daily rows for {prices_long['ticker'].nunique()} tickers")
    
    # Convert to wide format
    prices_wide = prices_long.pivot(
        index="date",
        columns="ticker",
        values="close"
    )
    
    # Rename columns to match tickers_in_order (use short names for simplicity)
    # Stooq tickers are like "AAPL.US", scores columns might want just "AAPL"
    prices_wide.columns = [c.replace(".US", "") for c in prices_wide.columns]
    
    # Fill NaN with forward-fill then backfill for any gaps
    prices_wide = prices_wide.ffill().bfill()
    
    # Drop any remaining NaN rows
    prices_wide = prices_wide.dropna()
    
    print(f"Wide prices shape: {prices_wide.shape}")
    print(f"Date range: {prices_wide.index.min()} to {prices_wide.index.max()}")
    
    # ========================================
    # 3) Build scores DataFrame (momentum-based)
    # ========================================
    print("\n=== Generating momentum-based scores...")
    
    # Use 12-month momentum as score: (close / close_12m_ago) - 1
    # Create monthly scores at month-end dates
    monthly_prices = prices_wide.resample("ME").last()
    
    # 12-month momentum
    momentum_12m = monthly_prices.pct_change(12)
    
    # Drop first 12 months (no momentum available)
    scores = momentum_12m.dropna()
    
    # Filter to only recent years for manageable backtest
    scores = scores[scores.index >= "2020-01-01"]
    
    print(f"Scores shape: {scores.shape}")
    print(f"Signal dates: {scores.index.min()} to {scores.index.max()}")
    print(f"Tickers: {list(scores.columns)}")
    
    # ========================================
    # 4) Write CSVs
    # ========================================
    output_dir = "data/backtest_real"
    os.makedirs(output_dir, exist_ok=True)
    
    prices_path = os.path.join(output_dir, "prices.csv")
    scores_path = os.path.join(output_dir, "scores.csv")
    
    # Save prices (wide format)
    write_prices_csv(prices_wide, path=prices_path, format="wide", date_col="date")
    print(f"\nPrices written to: {prices_path}")
    
    # Save scores (wide format)
    write_scores_csv(scores, path=scores_path, date_col="date")
    print(f"Scores written to: {scores_path}")
    
    # ========================================
    # 5) Run CLI backtest (Monthly and Quarterly)
    # ========================================
    print("\n" + "=" * 60)
    print("Running CLI Backtest")
    print("=" * 60)
    
    for rebalance in ["M", "Q"]:
        out_dir = f"results/real_backtest_{rebalance}"
        print(f"\n=== Backtest with rebalance={rebalance} -> {out_dir}")
        
        cmd = [
            sys.executable, "src/run_scores_backtest_cli.py",
            "--prices_csv_path", prices_path,
            "--scores_csv_path", scores_path,
            "--out_dir", out_dir,
            "--rebalance", rebalance,
            "--method", "softmax",
            "--cost_bps", "10",
            "--slippage_bps", "5",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
        else:
            print(result.stdout)
    
    # ========================================
    # 6) Verify determinism
    # ========================================
    print("\n=== Verifying determinism...")
    out_dir_m = "results/real_backtest_M"
    out_dir_m2 = "results/real_backtest_M_rerun"
    
    cmd = [
        sys.executable, "src/run_scores_backtest_cli.py",
        "--prices_csv_path", prices_path,
        "--scores_csv_path", scores_path,
        "--out_dir", out_dir_m2,
        "--rebalance", "M",
        "--method", "softmax",
        "--cost_bps", "10",
        "--slippage_bps", "5",
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
    eq1 = pd.read_csv(os.path.join(out_dir_m, "equity_curve.csv"))
    eq2 = pd.read_csv(os.path.join(out_dir_m2, "equity_curve.csv"))
    
    if eq1.equals(eq2):
        print("✅ Determinism verified: equity_curve.csv identical on re-run")
    else:
        print("❌ Determinism FAILED: equity_curve.csv differs")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Prices CSV: {prices_path}")
    print(f"Scores CSV: {scores_path}")
    print(f"Results (M): results/real_backtest_M/")
    print(f"Results (Q): results/real_backtest_Q/")
    
    # Load and print metrics
    for rebalance in ["M", "Q"]:
        metrics_path = f"results/real_backtest_{rebalance}/summary_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                summary = json.load(f)
            print(f"\n{rebalance} Metrics:")
            for k, v in summary["metrics"].items():
                print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Real Data Backtest Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
