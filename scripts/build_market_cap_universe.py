"""
Build a PIT-safe market-cap gate using SEC companyfacts + Stooq prices.

Goal
----
Compute market cap per ticker per score-date:
  market_cap[t, i] = shares_outstanding_PIT(t) * price_ffill_to_date(t)

Then "gate" microcaps (and missing shares) by forcing their scores to a very
low value on that date so TopK selection cannot pick them (as long as
eligible_count >= top_k at every date).

Inputs (expected)
-----------------
- prices.csv: wide daily prices with a "date" column
- scores.csv: wide scores with a "date" column (monthly signal dates)
- universe_sec_manifest.csv: from `scripts/download_sec_data.py preprocess/filter_universe`

Outputs
-------
Writes a self-contained directory containing:
- prices.csv (optional copy)
- scores.csv (gated scores)
- market_cap.csv (optional)
- eligibility.csv (0/1 mask, optional)
- summary.json
- missing_shares_tickers.txt

Example
-------
python scripts/build_market_cap_universe.py ^
  --manifest_csv data/backtest_universe_sec/universe_sec_manifest.csv ^
  --prices_csv data/backtest_universe_sec/prices.csv ^
  --scores_csv data/backtest_universe_sec/scores.csv ^
  --min_market_cap_usd 300000000 ^
  --top_k 400 ^
  --out_dir data/backtest_universe_sec_mktcap
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow running this script directly via `python scripts/...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.endswith(".US"):
        t = t[: -len(".US")]
    return t


def _read_scores_csv(scores_csv: str, *, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if date_col not in df.columns:
        raise ValueError(f"scores_csv missing date column '{date_col}': {scores_csv}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    if not df.index.is_unique:
        raise ValueError("scores_csv has duplicate dates")
    return df


def _read_prices_csv(prices_csv: str, *, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(prices_csv)
    if date_col not in df.columns:
        raise ValueError(f"prices_csv missing date column '{date_col}': {prices_csv}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    if not df.index.is_unique:
        raise ValueError("prices_csv has duplicate dates (wide format expected)")
    return df


def _load_manifest(manifest_csv: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    required = [
        "ticker",
        "cik_status",
        "companyfacts_status",
        "companyfacts_path",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest_csv missing columns {missing}: {manifest_csv}")
    df = df.copy()
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    return df


def _extract_shares_entries(companyfacts_path: str) -> pd.DataFrame:
    """
    Extract shares outstanding entries from a companyfacts JSON file.

    Tries (in order):
    - facts.dei.EntityCommonStockSharesOutstanding.units.*
    - facts.us-gaap.CommonStockSharesOutstanding.units.*
    """
    path = Path(companyfacts_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame(columns=["end", "filed", "shares"])

    facts = data.get("facts", {})
    if not isinstance(facts, dict):
        return pd.DataFrame(columns=["end", "filed", "shares"])

    candidates: list[tuple[str, str]] = [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ]

    units = None
    for taxonomy, tag in candidates:
        tax = facts.get(taxonomy, {})
        if not isinstance(tax, dict):
            continue
        tag_obj = tax.get(tag, {})
        if not isinstance(tag_obj, dict):
            continue
        units_obj = tag_obj.get("units", {})
        if isinstance(units_obj, dict) and units_obj:
            units = units_obj
            break

    if not isinstance(units, dict) or not units:
        return pd.DataFrame(columns=["end", "filed", "shares"])

    unit_key = "shares" if "shares" in units else sorted(units.keys())[0]
    entries = units.get(unit_key, [])
    if not isinstance(entries, list) or not entries:
        return pd.DataFrame(columns=["end", "filed", "shares"])

    rows: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        end = e.get("end")
        filed = e.get("filed")
        val = e.get("val")
        rows.append({"end": end, "filed": filed, "shares": val})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["end", "filed", "shares"])

    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").astype(float)
    df = df.dropna(subset=["end", "filed", "shares"])
    df = df[np.isfinite(df["shares"].values)]
    if df.empty:
        return pd.DataFrame(columns=["end", "filed", "shares"])

    df = df.sort_values(["end", "filed"], kind="mergesort").reset_index(drop=True)
    return df


def _pit_shares_series(
    entries_sorted: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Compute PIT shares outstanding for each date in `dates`.

    Rule at date t:
      choose the last row (by end, then filed) with end <= t and filed <= t.
    """
    if entries_sorted.empty:
        return np.full(len(dates), np.nan, dtype=float)

    ends = entries_sorted["end"].to_numpy(dtype="datetime64[ns]")
    filed = entries_sorted["filed"].to_numpy(dtype="datetime64[ns]")
    shares = entries_sorted["shares"].to_numpy(dtype=float)

    out = np.full(len(dates), np.nan, dtype=float)
    for i, d in enumerate(dates.to_numpy(dtype="datetime64[ns]")):
        mask = (ends <= d) & (filed <= d)
        if mask.any():
            out[i] = shares[np.nonzero(mask)[0][-1]]
    return out


def _write_wide_csv(df: pd.DataFrame, out_csv: str, *, date_col: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: date_col})
    out.to_csv(out_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build market-cap gated universe scores.")
    p.add_argument("--manifest_csv", default="data/backtest_universe_sec/universe_sec_manifest.csv")
    p.add_argument("--prices_csv", default="data/backtest_universe_sec/prices.csv")
    p.add_argument("--scores_csv", default="data/backtest_universe_sec/scores.csv")
    p.add_argument("--date_col", default="date")
    p.add_argument("--min_market_cap_usd", type=float, default=300_000_000.0)
    p.add_argument("--min_price", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=400, help="Sanity: require eligible_count >= top_k at every date")
    p.add_argument("--ineligible_penalty", type=float, default=1e6)
    p.add_argument("--out_dir", default="data/backtest_universe_sec_mktcap")
    p.add_argument("--copy_prices", action="store_true", help="Copy prices.csv into out_dir")
    p.add_argument("--write_market_cap_csv", action="store_true")
    p.add_argument("--write_eligibility_csv", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    manifest = _load_manifest(args.manifest_csv)
    manifest = manifest[
        (manifest["cik_status"] == "ok") & (manifest["companyfacts_status"] == "ok")
    ].copy()
    if manifest.empty:
        raise SystemExit("No tickers with cik_status=ok and companyfacts_status=ok in manifest.")

    ticker_to_companyfacts = dict(
        zip(manifest["ticker"].tolist(), manifest["companyfacts_path"].tolist())
    )

    scores = _read_scores_csv(args.scores_csv, date_col=args.date_col)
    prices = _read_prices_csv(args.prices_csv, date_col=args.date_col)

    # Normalize columns and intersect
    scores_cols = [_normalize_ticker(c) for c in scores.columns]
    prices_cols = [_normalize_ticker(c) for c in prices.columns]
    scores.columns = scores_cols
    prices.columns = prices_cols

    tickers = sorted(set(scores.columns) & set(prices.columns) & set(ticker_to_companyfacts.keys()))
    if len(tickers) < 2:
        raise SystemExit(f"Need at least 2 tickers after intersection, got {len(tickers)}")

    scores = scores[tickers].astype(float)
    prices = prices[tickers].astype(float)

    if not np.isfinite(scores.values).all():
        raise SystemExit("scores contains NaN/inf; cannot apply gating safely.")
    if not np.isfinite(prices.values).all():
        raise SystemExit("prices contains NaN/inf; cannot compute market cap.")

    dates = scores.index
    prices_at = prices.reindex(dates, method="ffill")
    if prices_at.isna().any().any():
        raise SystemExit("prices_at has NaN after ffill alignment to score dates.")

    n_dates = len(dates)
    n_tickers = len(tickers)
    market_cap = np.full((n_dates, n_tickers), np.nan, dtype=float)
    missing_shares: list[str] = []

    for j, ticker in enumerate(tickers):
        cf_path = ticker_to_companyfacts.get(ticker, "")
        entries = _extract_shares_entries(cf_path)
        shares_arr = _pit_shares_series(entries, dates)
        if not np.isfinite(shares_arr).any():
            missing_shares.append(ticker)
            continue
        market_cap[:, j] = shares_arr * prices_at[ticker].to_numpy(dtype=float)

    eligible = np.isfinite(market_cap)
    eligible &= market_cap >= float(args.min_market_cap_usd)
    if args.min_price and args.min_price > 0:
        eligible &= prices_at.to_numpy(dtype=float) >= float(args.min_price)

    eligible_count = eligible.sum(axis=1).astype(int)
    if args.top_k and args.top_k > 0:
        bad = np.where(eligible_count < int(args.top_k))[0]
        if len(bad) > 0:
            first_bad = dates[bad[0]]
            min_elig = int(eligible_count.min())
            raise SystemExit(
                f"Market-cap gate too strict for top_k={args.top_k}: "
                f"min eligible_count={min_elig} (first failing date: {first_bad.date()}). "
                "Lower --min_market_cap_usd (or --top_k) and retry."
            )

    scores_arr = scores.to_numpy(dtype=float)
    row_min = scores_arr.min(axis=1)
    floor = (row_min - float(args.ineligible_penalty)).reshape(-1, 1)
    gated_arr = np.where(eligible, scores_arr, floor)
    gated_scores = pd.DataFrame(gated_arr, index=dates, columns=tickers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_prices:
        shutil.copyfile(args.prices_csv, out_dir / "prices.csv")

    _write_wide_csv(gated_scores, str(out_dir / "scores.csv"), date_col=args.date_col)

    if args.write_market_cap_csv:
        mc_df = pd.DataFrame(market_cap, index=dates, columns=tickers)
        _write_wide_csv(mc_df, str(out_dir / "market_cap.csv"), date_col=args.date_col)

    if args.write_eligibility_csv:
        elig_df = pd.DataFrame(eligible.astype(int), index=dates, columns=tickers)
        _write_wide_csv(elig_df, str(out_dir / "eligibility.csv"), date_col=args.date_col)

    with open(out_dir / "missing_shares_tickers.txt", "w", encoding="utf-8") as f:
        for t in missing_shares:
            f.write(f"{t}\n")

    summary = {
        "manifest_csv": args.manifest_csv,
        "prices_csv": args.prices_csv,
        "scores_csv": args.scores_csv,
        "min_market_cap_usd": float(args.min_market_cap_usd),
        "min_price": float(args.min_price),
        "top_k": int(args.top_k),
        "ineligible_penalty": float(args.ineligible_penalty),
        "tickers_total": int(len(tickers)),
        "dates_total": int(len(dates)),
        "eligible_count_min": int(eligible_count.min()) if len(eligible_count) else 0,
        "eligible_count_median": int(np.median(eligible_count)) if len(eligible_count) else 0,
        "eligible_count_max": int(eligible_count.max()) if len(eligible_count) else 0,
        "missing_shares_tickers": int(len(missing_shares)),
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Tickers: {len(tickers)}")
    print(f"Dates: {len(dates)}")
    print(
        "Eligible count (min/median/max): "
        f"{summary['eligible_count_min']}/{summary['eligible_count_median']}/{summary['eligible_count_max']}"
    )
    print(f"Missing shares tickers: {len(missing_shares)}")
    print(f"Wrote: {out_dir / 'scores.csv'}")
    if args.copy_prices:
        print(f"Wrote: {out_dir / 'prices.csv'}")
    if args.write_market_cap_csv:
        print(f"Wrote: {out_dir / 'market_cap.csv'}")
    if args.write_eligibility_csv:
        print(f"Wrote: {out_dir / 'eligibility.csv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

