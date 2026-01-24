"""
CLI for Score Generator.

Generates monthly momentum scores from price data.

Usage:
    python src/run_generate_scores_cli.py \\
        --prices_csv_path data/prices.csv \\
        --out_scores_csv_path data/scores.csv \\
        --rebalance M \\
        --lookback_days 252
"""

import argparse
import os
import sys
from typing import List, Optional

# Allow running via `python src/run_generate_scores_cli.py` from repo root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.generate_scores_from_prices import (
    load_prices_csv,
    compute_monthly_momentum_scores,
    write_scores_csv,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate monthly momentum scores from prices CSV."
    )
    
    # Required arguments
    parser.add_argument(
        "--prices_csv_path",
        type=str,
        required=True,
        help="Path to prices CSV (wide or long format)"
    )
    parser.add_argument(
        "--out_scores_csv_path",
        type=str,
        required=True,
        help="Output path for scores CSV"
    )
    
    # Optional arguments
    parser.add_argument(
        "--rebalance",
        type=str,
        default="M",
        choices=["M", "Q"],
        help="Rebalance frequency: M (monthly) or Q (quarterly)"
    )
    parser.add_argument(
        "--lookback_days",
        type=int,
        default=252,
        help="Lookback days for momentum (default: 252)"
    )
    parser.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Date column name (default: date)"
    )
    parser.add_argument(
        "--ticker_col",
        type=str,
        default="ticker",
        help="Ticker column name for long format (default: ticker)"
    )
    parser.add_argument(
        "--price_col",
        type=str,
        default="close",
        help="Price column name (default: close)"
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=1.0,
        help="Minimum coverage fraction (default: 1.0)"
    )
    parser.add_argument(
        "--no_leading_plateau_gate",
        action="store_true",
        help="Disable leading plateau integrity gate"
    )
    
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.
    
    Parameters
    ----------
    argv : list, optional
        Command line arguments. If None, uses sys.argv[1:].
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)
    
    # Load prices
    print(f"Loading prices from: {args.prices_csv_path}")
    prices = load_prices_csv(
        args.prices_csv_path,
        date_col=args.date_col,
        ticker_col=args.ticker_col,
        price_col=args.price_col,
    )
    
    print(f"Loaded prices: {len(prices)} rows")
    
    # Compute scores
    print(f"Computing {args.rebalance} momentum scores (lookback={args.lookback_days})...")
    scores = compute_monthly_momentum_scores(
        prices,
        lookback_days=args.lookback_days,
        rebalance=args.rebalance,
        min_coverage=args.min_coverage,
        enforce_no_leading_plateau=not args.no_leading_plateau_gate,
    )
    
    print(f"Generated scores: {scores.shape[0]} dates × {scores.shape[1]} tickers")
    
    # Write scores
    write_scores_csv(
        scores,
        out_scores_csv_path=args.out_scores_csv_path,
        date_col=args.date_col,
    )
    
    print(f"Scores written to: {args.out_scores_csv_path}")
    print(f"Signal date range: {scores.index.min()} to {scores.index.max()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
