"""
CLI: 200티큐 전략(200TQQ) 백테스트 (2010~2024 등).

예시
----
python scripts/backtest_200tqq_strategy.py --start_date 2010-01-01 --end_date 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import pandas as pd

# Allow running from repo root via `python scripts/...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.yahoo_prices import load_yahoo_daily
from src.strategy_200tqq import Strategy200TQQConfig, run_200tqq_backtest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest 200티큐 전략 (TQQQ/SPLG/SGOV switching).")
    p.add_argument("--start_date", type=str, default="2010-01-01")
    p.add_argument("--end_date", type=str, default="2024-12-31")
    p.add_argument("--initial_equity", type=float, default=1.0)

    p.add_argument("--sma_window", type=int, default=200)
    p.add_argument("--overheat_mult", type=float, default=1.05)
    p.add_argument("--no_entry_confirmation", action="store_true", help="Disable DOWN->FOCUS 1-day confirmation.")

    p.add_argument("--monthly_contribution", type=float, default=0.0)

    p.add_argument("--stop_loss_pct", type=float, default=0.05, help="0 to disable.")
    p.add_argument("--cost_bps", type=float, default=0.0)
    p.add_argument("--slippage_bps", type=float, default=0.0)

    p.add_argument("--take_profit_mode", type=str, default="official", choices=["none", "official", "high"])
    p.add_argument(
        "--take_profit_reinvest",
        type=str,
        default="all_splg",
        choices=["all_splg", "split_principal_profit"],
    )

    p.add_argument("--initial_mode", type=str, default="strategy", choices=["safe", "strategy"])
    p.add_argument("--overheat_start_splg_weight", type=float, default=1.0)

    p.add_argument("--tqqq", type=str, default="TQQQ")
    p.add_argument("--splg", type=str, default="SPLG")
    p.add_argument("--safe", type=str, default="SGOV")
    p.add_argument("--safe_proxy", type=str, default="BIL")

    p.add_argument("--cache_dir", type=str, default="data/raw/yahoo")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--out_dir", type=str, default="results/200tqq_backtest")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    tickers = [args.tqqq, args.splg, args.safe_proxy]
    if args.safe and args.safe not in tickers:
        tickers.append(args.safe)

    data_by_ticker: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        frame = load_yahoo_daily(
            tk,
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=args.cache_dir,
            refresh=bool(args.refresh),
        )
        data_by_ticker[tk] = frame.ohlcv

    cfg = Strategy200TQQConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=float(args.initial_equity),
        tqqq_ticker=args.tqqq,
        splg_ticker=args.splg,
        safe_ticker=args.safe,
        safe_proxy_ticker=args.safe_proxy,
        sma_window=int(args.sma_window),
        overheat_mult=float(args.overheat_mult),
        apply_entry_confirmation=not bool(args.no_entry_confirmation),
        monthly_contribution=float(args.monthly_contribution),
        stop_loss_pct=float(args.stop_loss_pct),
        cost_bps=float(args.cost_bps),
        slippage_bps=float(args.slippage_bps),
        take_profit_mode=args.take_profit_mode,
        take_profit_reinvest=args.take_profit_reinvest,
        initial_mode=args.initial_mode,
        overheat_start_splg_weight=float(args.overheat_start_splg_weight),
    )

    result = run_200tqq_backtest(data_by_ticker, cfg=cfg)

    # Artifacts
    daily = result["daily"].reset_index()
    daily.to_csv(os.path.join(args.out_dir, "daily.csv"), index=False)

    equity_curve = result["daily"]["equity"].reset_index()
    equity_curve.columns = ["date", "equity"]
    equity_curve.to_csv(os.path.join(args.out_dir, "equity_curve.csv"), index=False)

    trades = result["trades"]
    trades.to_csv(os.path.join(args.out_dir, "trades.csv"), index=False)

    state_df = result["state"].reset_index()
    state_df.to_csv(os.path.join(args.out_dir, "state.csv"), index=False)

    with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": result["metrics"], "config": cfg.__dict__}, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    print(f"Artifacts written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

