"""
Build PIT-safe Form 345 (insider) features aligned to monthly score dates.

This script parses the SEC "structured data" insider transactions dataset
downloaded by `scripts/download_sec_data.py` into a backtest-ready feature panel.

Key PIT rule
------------
We bucket transactions by *filing date* (not transaction date):
  a transaction becomes usable at the first score date >= FILING_DATE.

This prevents look-ahead and matches the project's monthly score-date panels.

Output
------
Writes a long-format CSV with columns:
  - date (score date)
  - ticker (normalized)
  - buy_value, sell_value, net_value
  - buy_shares, sell_shares, net_shares
  - buy_count, sell_count

Only open-market purchase/sale codes are included by default (P, S).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
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


def _read_wide_csv_header_tickers(path: str, *, date_col: str) -> list[str]:
    df0 = pd.read_csv(path, nrows=0)
    cols = [c for c in df0.columns if str(c).strip()]
    cols = [_normalize_ticker(c) for c in cols]
    cols = [c for c in cols if c and c.lower() != date_col.lower()]
    return sorted(set(cols))


def _read_score_dates(score_dates_csv: str, *, date_col: str) -> pd.DatetimeIndex:
    df = pd.read_csv(score_dates_csv, usecols=[date_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    dates = pd.DatetimeIndex(df[date_col].values)
    if not dates.is_unique:
        raise ValueError(f"score_dates_csv has duplicate dates: {score_dates_csv}")
    return dates


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Form345 (insider) features aligned to score dates.")
    p.add_argument("--insiders_dir", default="data/raw/insiders")
    p.add_argument("--universe_prices_csv", default="data/backtest_universe_sec_mktcap/prices.csv")
    p.add_argument("--score_dates_csv", default="data/backtest_universe_sec_mktcap/market_cap.csv")
    p.add_argument("--date_col", default="date")
    p.add_argument("--out_csv", default="data/processed/insiders/insider_events_form345.csv")
    p.add_argument("--trans_codes", nargs="+", default=["P", "S"], help="Transaction codes to include (default: P S).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    insiders_dir = Path(args.insiders_dir)
    if not insiders_dir.exists():
        raise SystemExit(f"Missing insiders_dir: {insiders_dir}")

    if not Path(args.universe_prices_csv).exists():
        raise SystemExit(f"Missing universe_prices_csv: {args.universe_prices_csv}")
    if not Path(args.score_dates_csv).exists():
        raise SystemExit(f"Missing score_dates_csv: {args.score_dates_csv}")

    tickers = _read_wide_csv_header_tickers(args.universe_prices_csv, date_col=args.date_col)
    if not tickers:
        raise SystemExit("No tickers found in universe_prices_csv header")

    score_dates = _read_score_dates(args.score_dates_csv, date_col=args.date_col)
    if len(score_dates) < 2:
        raise SystemExit(f"Need >=2 score dates, got {len(score_dates)}")

    trans_codes = sorted({_normalize_ticker(c) for c in args.trans_codes if str(c).strip()})
    if not trans_codes:
        raise SystemExit("--trans_codes must contain at least one code")

    zip_paths = sorted(insiders_dir.glob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No .zip files found under: {insiders_dir}")

    min_d = score_dates.min()
    max_d = score_dates.max()

    chunks: list[pd.DataFrame] = []
    processed = 0
    skipped = 0

    for zpath in zip_paths:
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                names = set(z.namelist())
                if "SUBMISSION.tsv" not in names or "NONDERIV_TRANS.tsv" not in names:
                    skipped += 1
                    continue

                with z.open("SUBMISSION.tsv") as f:
                    sub = pd.read_csv(
                        f,
                        sep="\t",
                        usecols=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"],
                        dtype={"ACCESSION_NUMBER": str, "FILING_DATE": str, "ISSUERTRADINGSYMBOL": str},
                    )
                sub = sub.dropna(subset=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"]).copy()
                if sub.empty:
                    skipped += 1
                    continue

                sub["ticker"] = sub["ISSUERTRADINGSYMBOL"].map(_normalize_ticker)
                sub = sub[sub["ticker"].isin(tickers)]
                if sub.empty:
                    skipped += 1
                    continue

                sub["filing_date"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
                sub = sub.dropna(subset=["filing_date"])
                sub = sub[(sub["filing_date"] >= min_d) & (sub["filing_date"] <= max_d)]
                if sub.empty:
                    skipped += 1
                    continue

                with z.open("NONDERIV_TRANS.tsv") as f:
                    tx = pd.read_csv(
                        f,
                        sep="\t",
                        usecols=[
                            "ACCESSION_NUMBER",
                            "TRANS_CODE",
                            "TRANS_SHARES",
                            "TRANS_PRICEPERSHARE",
                            "TRANS_ACQUIRED_DISP_CD",
                        ],
                        dtype={
                            "ACCESSION_NUMBER": str,
                            "TRANS_CODE": str,
                            "TRANS_SHARES": str,
                            "TRANS_PRICEPERSHARE": str,
                            "TRANS_ACQUIRED_DISP_CD": str,
                        },
                    )
                tx = tx.dropna(subset=["ACCESSION_NUMBER", "TRANS_CODE"]).copy()
                if tx.empty:
                    skipped += 1
                    continue

                tx["TRANS_CODE"] = tx["TRANS_CODE"].astype(str).str.strip().str.upper()
                tx = tx[tx["TRANS_CODE"].isin(trans_codes)]
                if tx.empty:
                    skipped += 1
                    continue

                tx["TRANS_SHARES"] = pd.to_numeric(tx["TRANS_SHARES"], errors="coerce").astype(float)
                tx["TRANS_PRICEPERSHARE"] = pd.to_numeric(tx["TRANS_PRICEPERSHARE"], errors="coerce").astype(float)
                tx = tx.dropna(subset=["TRANS_SHARES"]).copy()
                if tx.empty:
                    skipped += 1
                    continue

                tx["TRANS_PRICEPERSHARE"] = tx["TRANS_PRICEPERSHARE"].fillna(0.0)
                tx["TRANS_ACQUIRED_DISP_CD"] = tx["TRANS_ACQUIRED_DISP_CD"].fillna("").astype(str).str.strip().str.upper()

                merged = tx.merge(sub[["ACCESSION_NUMBER", "ticker", "filing_date"]], on="ACCESSION_NUMBER", how="inner")
                if merged.empty:
                    skipped += 1
                    continue

                # Determine sign (+1 buy, -1 sell). Prefer acquired/disposed code; fallback to trans code.
                sign = np.zeros(len(merged), dtype=float)
                ad = merged["TRANS_ACQUIRED_DISP_CD"].to_numpy(dtype=str)
                sign[ad == "A"] = 1.0
                sign[ad == "D"] = -1.0
                tcode = merged["TRANS_CODE"].to_numpy(dtype=str)
                sign[(sign == 0.0) & (tcode == "P")] = 1.0
                sign[(sign == 0.0) & (tcode == "S")] = -1.0
                merged["sign"] = sign
                merged = merged[merged["sign"] != 0.0].copy()
                if merged.empty:
                    skipped += 1
                    continue

                merged["value"] = merged["TRANS_SHARES"].to_numpy(dtype=float) * merged["TRANS_PRICEPERSHARE"].to_numpy(dtype=float)
                merged["buy_value"] = np.where(merged["sign"] > 0, merged["value"], 0.0)
                merged["sell_value"] = np.where(merged["sign"] < 0, merged["value"], 0.0)
                merged["net_value"] = merged["buy_value"] - merged["sell_value"]

                merged["buy_shares"] = np.where(merged["sign"] > 0, merged["TRANS_SHARES"].to_numpy(dtype=float), 0.0)
                merged["sell_shares"] = np.where(merged["sign"] < 0, merged["TRANS_SHARES"].to_numpy(dtype=float), 0.0)
                merged["net_shares"] = merged["buy_shares"] - merged["sell_shares"]

                merged["buy_count"] = np.where(merged["sign"] > 0, 1.0, 0.0)
                merged["sell_count"] = np.where(merged["sign"] < 0, 1.0, 0.0)

                # Bucket to first score date >= filing_date
                filing_arr = merged["filing_date"].to_numpy(dtype="datetime64[ns]")
                idx = score_dates.searchsorted(filing_arr, side="left")
                keep = idx < len(score_dates)
                merged = merged.loc[keep].copy()
                if merged.empty:
                    skipped += 1
                    continue
                merged[args.date_col] = score_dates.to_numpy(dtype="datetime64[ns]")[idx[keep]]

                g = (
                    merged.groupby([args.date_col, "ticker"], sort=True)[
                        [
                            "buy_value",
                            "sell_value",
                            "net_value",
                            "buy_shares",
                            "sell_shares",
                            "net_shares",
                            "buy_count",
                            "sell_count",
                        ]
                    ]
                    .sum()
                    .reset_index()
                )
                chunks.append(g)
                processed += 1
        except Exception:
            skipped += 1
            continue

    if not chunks:
        raise SystemExit("No insider transactions parsed (check universe tickers, date range, and zip contents).")

    all_df = pd.concat(chunks, ignore_index=True)
    all_df[args.date_col] = pd.to_datetime(all_df[args.date_col])
    all_df["ticker"] = all_df["ticker"].map(_normalize_ticker)
    all_df = all_df.groupby([args.date_col, "ticker"], sort=True, as_index=False).sum()
    all_df = all_df.sort_values([args.date_col, "ticker"], kind="mergesort").reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)

    summary = {
        "insiders_dir": str(insiders_dir),
        "universe_prices_csv": args.universe_prices_csv,
        "score_dates_csv": args.score_dates_csv,
        "out_csv": str(out_csv),
        "trans_codes": trans_codes,
        "score_dates_min": str(pd.Timestamp(min_d).date()),
        "score_dates_max": str(pd.Timestamp(max_d).date()),
        "universe_tickers": int(len(tickers)),
        "zip_files_total": int(len(zip_paths)),
        "zip_files_processed": int(processed),
        "zip_files_skipped": int(skipped),
        "rows_out": int(len(all_df)),
        "unique_dates_out": int(all_df[args.date_col].nunique()),
        "unique_tickers_out": int(all_df["ticker"].nunique()),
    }
    summary_path = out_csv.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
