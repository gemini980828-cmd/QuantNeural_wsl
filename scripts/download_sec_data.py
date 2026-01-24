"""
Download free SEC datasets (companyfacts, submissions, 13F, Form 345) with caching.

This script intentionally uses only the Python stdlib (no extra deps).

SEC requires a descriptive User-Agent that includes contact info. Set it via
`--user_agent` or `SEC_USER_AGENT` env var.

Examples
--------
Download ticker<->CIK mapping:
  python scripts/download_sec_data.py tickers --user_agent "QuantNeural you@email.com"

Build universe manifests from downloaded SEC files:
  python scripts/download_sec_data.py preprocess ^
    --universe_prices_csv data/backtest_universe_full/prices.csv ^
    --out_dir data/processed/sec_universe

Filter prices/scores to fundamentals-ready tickers:
  python scripts/download_sec_data.py filter_universe ^
    --manifest_csv data/processed/sec_universe/universe_sec_manifest.csv ^
    --prices_csv data/backtest_universe_full/prices.csv ^
    --scores_csv data/backtest_universe_full/scores.csv ^
    --out_dir data/backtest_universe_sec

Download companyfacts for tickers in an existing universe prices CSV (header tickers):
  python scripts/download_sec_data.py companyfacts ^
    --universe_prices_csv data/backtest_universe_full/prices.csv ^
    --out_dir data/raw/sec_bulk ^
    --user_agent "QuantNeural you@email.com"

Download submissions JSON for a few tickers:
  python scripts/download_sec_data.py submissions --tickers AAPL MSFT ^
    --out_dir data/sec_submissions --user_agent "QuantNeural you@email.com"

Download + extract SEC 13F structured dataset for 2024Q4:
  python scripts/download_sec_data.py 13f --year 2024 --quarter 4 --extract ^
    --out_dir data/raw/sec_structured/13f --user_agent "QuantNeural you@email.com"
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Allow running this script directly via `python scripts/...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


_SIC_TO_SECTOR_TABLE = [
    (100, 999, "Materials"),
    (1000, 1499, "Energy"),
    (1500, 1799, "Industrials"),
    (2000, 2099, "Consumer Staples"),
    (2100, 2199, "Consumer Staples"),
    (2200, 2399, "Consumer Discretionary"),
    (2400, 2499, "Materials"),
    (2500, 2599, "Consumer Discretionary"),
    (2600, 2699, "Materials"),
    (2700, 2799, "Communication Services"),
    (2800, 2899, "Health Care"),
    (2900, 2999, "Energy"),
    (3000, 3099, "Materials"),
    (3100, 3199, "Consumer Discretionary"),
    (3200, 3299, "Materials"),
    (3300, 3399, "Materials"),
    (3400, 3499, "Industrials"),
    (3500, 3599, "Industrials"),
    (3600, 3699, "Information Technology"),
    (3700, 3799, "Industrials"),
    (3800, 3899, "Health Care"),
    (3900, 3999, "Consumer Discretionary"),
    (4000, 4799, "Industrials"),
    (4800, 4899, "Communication Services"),
    (4900, 4999, "Utilities"),
    (5000, 5199, "Consumer Discretionary"),
    (5200, 5399, "Consumer Discretionary"),
    (5400, 5499, "Consumer Staples"),
    (5500, 5599, "Consumer Discretionary"),
    (5600, 5699, "Consumer Discretionary"),
    (5700, 5799, "Consumer Discretionary"),
    (5800, 5899, "Consumer Discretionary"),
    (5900, 5999, "Consumer Discretionary"),
    (6000, 6199, "Financials"),
    (6200, 6299, "Financials"),
    (6300, 6499, "Financials"),
    (6500, 6599, "Real Estate"),
    (6700, 6799, "Financials"),
    (7000, 7099, "Consumer Discretionary"),
    (7200, 7299, "Consumer Discretionary"),
    (7300, 7399, "Information Technology"),
    (7500, 7599, "Consumer Discretionary"),
    (7600, 7699, "Consumer Discretionary"),
    (7800, 7899, "Communication Services"),
    (7900, 7999, "Consumer Discretionary"),
    (8000, 8099, "Health Care"),
    (8100, 8199, "Industrials"),
    (8200, 8299, "Consumer Discretionary"),
    (8300, 8399, "Health Care"),
    (8600, 8699, "Consumer Discretionary"),
    (8700, 8799, "Information Technology"),
]


def sic_to_sector_name(sic: int | str | None) -> str:
    if sic is None:
        return ""
    try:
        sic_int = int(sic)
    except (ValueError, TypeError):
        return ""
    for range_start, range_end, sector_name in _SIC_TO_SECTOR_TABLE:
        if range_start <= sic_int <= range_end:
            return sector_name
    return ""


def _normalize_cik(cik: str | int) -> str:
    digits_only = "".join(c for c in str(cik) if c.isdigit())
    return digits_only.zfill(10)


def _normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.endswith(".US"):
        t = t[: -len(".US")]
    return t


def _ticker_aliases(ticker: str) -> list[str]:
    t = _normalize_ticker(ticker)
    aliases = [t]
    if "." in t:
        aliases.append(t.replace(".", "-"))
    if "-" in t:
        aliases.append(t.replace("-", "."))
    # Deduplicate while preserving order
    out = []
    seen = set()
    for a in aliases:
        if a and a not in seen:
            out.append(a)
            seen.add(a)
    return out


def _read_csv_header_tickers(csv_path: str) -> list[str]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if not header:
        return []
    cols = [_normalize_ticker(c) for c in header]
    if cols and cols[0].lower() in {"date", "datetime", "time"}:
        cols = cols[1:]
    return [c for c in cols if c]


def _read_tickers_txt(path: str) -> list[str]:
    tickers: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(_normalize_ticker(s))
    return tickers


def _sec_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Encoding": "gzip",
        "Connection": "close",
    }


def _http_get_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read()
        hdrs = {k: v for k, v in resp.headers.items()}
        if hdrs.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw, hdrs


@dataclass(frozen=True)
class DownloadResult:
    status: str  # downloaded|skipped|failed
    path: str
    error: str = ""


def _download_to_path(
    url: str,
    *,
    out_path: Path,
    headers: dict[str, str],
    timeout_sec: float,
    retries: int,
    min_sleep_sec: float,
    overwrite: bool,
    dry_run: bool,
) -> DownloadResult:
    if out_path.exists() and not overwrite:
        return DownloadResult(status="skipped", path=str(out_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return DownloadResult(status="downloaded", path=str(out_path))

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            data, resp_headers = _http_get_bytes(
                url, headers=headers, timeout_sec=timeout_sec
            )
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, out_path)
            return DownloadResult(status="downloaded", path=str(out_path))
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get("Retry-After") if hasattr(e, "headers") else None
            code = getattr(e, "code", None)
            last_err = f"HTTPError {code}: {e}"
            if code in {403}:
                return DownloadResult(
                    status="failed",
                    path=str(out_path),
                    error=f"{last_err} (check --user_agent / SEC_USER_AGENT)",
                )
            if code in {404}:
                return DownloadResult(status="failed", path=str(out_path), error=last_err)
            if code not in {429, 500, 502, 503, 504}:
                return DownloadResult(status="failed", path=str(out_path), error=last_err)

            sleep_sec = min_sleep_sec * (2 ** (attempt - 1))
            if retry_after:
                try:
                    sleep_sec = max(sleep_sec, float(retry_after))
                except ValueError:
                    pass
            sleep_sec = min(120.0, sleep_sec) + random.random() * 0.25
            time.sleep(sleep_sec)
        except urllib.error.URLError as e:
            last_err = f"URLError: {e}"
            sleep_sec = min(120.0, min_sleep_sec * (2 ** (attempt - 1))) + random.random() * 0.25
            time.sleep(sleep_sec)
        except Exception as e:  # pragma: no cover
            last_err = str(e)
            break

    return DownloadResult(status="failed", path=str(out_path), error=last_err)


def _load_company_ticker_map(
    json_path: str,
) -> dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ticker_to_cik: dict[str, str] = {}

    # Handle fields/data array format (company_tickers_exchange.json)
    if isinstance(data, dict) and "fields" in data and "data" in data:
        fields = data["fields"]
        rows = data["data"]
        try:
            cik_idx = fields.index("cik")
            ticker_idx = fields.index("ticker")
        except ValueError:
            raise ValueError(f"Missing 'cik' or 'ticker' field in {json_path}")
        
        for row in rows:
            if not isinstance(row, list) or len(row) <= max(cik_idx, ticker_idx):
                continue
            cik = row[cik_idx]
            ticker = row[ticker_idx]
            if cik is not None and ticker:
                ticker_to_cik[_normalize_ticker(str(ticker))] = _normalize_cik(cik)
    
    # Handle dict of dicts format (company_tickers.json)
    elif isinstance(data, dict):
        entries = list(data.values())
        for row in entries:
            if not isinstance(row, dict):
                continue
            ticker = row.get("ticker")
            if not isinstance(ticker, str) or not ticker.strip():
                continue
            cik = row.get("cik_str")
            if cik is None:
                cik = row.get("cik")
            if cik is None:
                continue
            ticker_to_cik[_normalize_ticker(ticker)] = _normalize_cik(cik)
    
    # Handle list format
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            ticker = row.get("ticker")
            if not isinstance(ticker, str) or not ticker.strip():
                continue
            cik = row.get("cik_str")
            if cik is None:
                cik = row.get("cik")
            if cik is None:
                continue
            ticker_to_cik[_normalize_ticker(ticker)] = _normalize_cik(cik)
    else:
        raise ValueError(f"Unexpected JSON type for {json_path}: {type(data)}")

    if not ticker_to_cik:
        raise ValueError(f"No ticker->CIK rows parsed from {json_path}")

    return ticker_to_cik




def _resolve_ciks(
    tickers: Iterable[str],
    *,
    ticker_to_cik: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for t in tickers:
        found = None
        for a in _ticker_aliases(t):
            if a in ticker_to_cik:
                found = ticker_to_cik[a]
                break
        if found is None:
            missing.append(_normalize_ticker(t))
        else:
            resolved[_normalize_ticker(t)] = found
    return resolved, missing


def _collect_tickers_from_args(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = []
    if args.tickers:
        tickers.extend([_normalize_ticker(t) for t in args.tickers])
    if getattr(args, "tickers_file", None):
        tickers.extend(_read_tickers_txt(args.tickers_file))
    if getattr(args, "universe_prices_csv", None):
        tickers.extend(_read_csv_header_tickers(args.universe_prices_csv))
    if getattr(args, "universe_scores_csv", None):
        tickers.extend(_read_csv_header_tickers(args.universe_scores_csv))

    # Deduplicate (deterministic)
    return sorted({t for t in tickers if t})


def _require_user_agent(args: argparse.Namespace) -> str:
    ua = (args.user_agent or os.environ.get("SEC_USER_AGENT", "")).strip()
    if not ua:
        raise SystemExit(
            "Missing SEC User-Agent. Provide `--user_agent \"Name email\"` "
            "or set `SEC_USER_AGENT` env var."
        )
    return ua


def cmd_tickers(args: argparse.Namespace) -> int:
    ua = _require_user_agent(args)
    out_path = Path(args.out_json)
    url = SEC_TICKERS_EXCHANGE_URL if args.exchange else SEC_TICKERS_URL
    res = _download_to_path(
        url,
        out_path=out_path,
        headers=_sec_headers(ua),
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        min_sleep_sec=args.min_sleep_sec,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"{res.status}: {url} -> {res.path}")
    if res.error:
        print(res.error)
        return 1
    return 0


def _ensure_ticker_map_json(
    *,
    user_agent: str,
    ticker_map_json: str,
    timeout_sec: float,
    retries: int,
    min_sleep_sec: float,
    dry_run: bool,
) -> str:
    path = Path(ticker_map_json)
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    res = _download_to_path(
        SEC_TICKERS_EXCHANGE_URL,
        out_path=path,
        headers=_sec_headers(user_agent),
        timeout_sec=timeout_sec,
        retries=retries,
        min_sleep_sec=min_sleep_sec,
        overwrite=False,
        dry_run=dry_run,
    )
    if res.status == "failed":
        raise SystemExit(f"Failed to download ticker map: {res.error}")
    return str(path)


def _write_report_csv(path: str, rows: list[dict]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_csv_with_columns(path: str, rows: list[dict], columns: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _default_sector_name_to_id_10() -> dict[str, str]:
    """
    10-sector scheme (Real Estate folded into Financials).
    Keys are sector_name strings produced by sic_to_sector_name().
    """
    return {
        "Energy": "S0",
        "Materials": "S1",
        "Industrials": "S2",
        "Consumer Discretionary": "S3",
        "Consumer Staples": "S4",
        "Health Care": "S5",
        "Financials": "S6",
        "Real Estate": "S6",
        "Information Technology": "S7",
        "Communication Services": "S8",
        "Utilities": "S9",
    }


def _sector_id_to_index(sector_id: str) -> int | None:
    s = str(sector_id).strip().upper()
    if len(s) == 2 and s[0] == "S" and s[1].isdigit():
        return int(s[1])
    return None


def _safe_load_json(path: Path) -> tuple[dict | None, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), ""
    except Exception as e:
        return None, str(e)


def cmd_preprocess(args: argparse.Namespace) -> int:
    tickers = _collect_tickers_from_args(args)
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers / --tickers_file / --universe_prices_csv.")

    ticker_map_path = Path(args.ticker_map_json)
    if not ticker_map_path.exists():
        raise SystemExit(
            f"Missing ticker_map_json: {args.ticker_map_json}. "
            "Run `python scripts/download_sec_data.py tickers --exchange --user_agent \"Name email\"` first."
        )

    ticker_to_cik = _load_company_ticker_map(str(ticker_map_path))
    resolved, missing = _resolve_ciks(tickers, ticker_to_cik=ticker_to_cik)

    resolved_items = sorted(resolved.items())
    if args.max_tickers and args.max_tickers > 0:
        resolved_items = resolved_items[: args.max_tickers]
    unique_ciks = sorted({cik for _, cik in resolved_items})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    companyfacts_dir = Path(args.companyfacts_dir)
    submissions_dir = Path(args.submissions_dir)
    min_companyfacts_bytes = int(args.min_companyfacts_bytes)

    sector_name_to_id = _default_sector_name_to_id_10()

    cik_info: dict[str, dict] = {}
    for cik in unique_ciks:
        sub_path = submissions_dir / f"CIK{cik}.json"
        if not sub_path.exists():
            cik_info[cik] = {
                "submissions_status": "missing",
                "submissions_path": str(sub_path),
                "submissions_bytes": 0,
                "sic": "",
                "sector_name": "",
                "sector_id": "",
                "sector_index": "",
                "submissions_error": "",
            }
            continue

        size = sub_path.stat().st_size
        data, err = _safe_load_json(sub_path)
        if data is None or not isinstance(data, dict):
            cik_info[cik] = {
                "submissions_status": "bad_json",
                "submissions_path": str(sub_path),
                "submissions_bytes": size,
                "sic": "",
                "sector_name": "",
                "sector_id": "",
                "sector_index": "",
                "submissions_error": err,
            }
            continue

        sic = data.get("sic")
        sector_name = sic_to_sector_name(sic)
        sector_id = sector_name_to_id.get(sector_name, "")
        sector_index = _sector_id_to_index(sector_id)

        cik_info[cik] = {
            "submissions_status": "ok",
            "submissions_path": str(sub_path),
            "submissions_bytes": size,
            "sic": str(sic) if sic is not None else "",
            "sector_name": sector_name,
            "sector_id": sector_id,
            "sector_index": str(sector_index) if sector_index is not None else "",
            "submissions_error": "",
        }

    manifest_rows: list[dict] = []
    ticker_sector_rows: list[dict] = []
    cik_to_sector_index: dict[str, int] = {}

    for ticker in tickers:
        cik = resolved.get(ticker, "")
        if not cik:
            manifest_rows.append(
                {
                    "ticker": ticker,
                    "cik": "",
                    "cik_status": "missing",
                    "companyfacts_status": "n/a",
                    "companyfacts_path": "",
                    "companyfacts_bytes": 0,
                    "submissions_status": "n/a",
                    "submissions_path": "",
                    "submissions_bytes": 0,
                    "sic": "",
                    "sector_name": "",
                    "sector_id": "",
                    "sector_index": "",
                    "submissions_error": "",
                }
            )
            continue

        cf_path = companyfacts_dir / f"CIK{cik}.json"
        if not cf_path.exists():
            companyfacts_status = "missing"
            cf_bytes = 0
        else:
            cf_bytes = cf_path.stat().st_size
            companyfacts_status = "ok" if cf_bytes >= min_companyfacts_bytes else "too_small"

        info = cik_info.get(cik) or {
            "submissions_status": "missing",
            "submissions_path": str(submissions_dir / f"CIK{cik}.json"),
            "submissions_bytes": 0,
            "sic": "",
            "sector_name": "",
            "sector_id": "",
            "sector_index": "",
            "submissions_error": "",
        }

        manifest_rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                "cik_status": "ok",
                "companyfacts_status": companyfacts_status,
                "companyfacts_path": str(cf_path),
                "companyfacts_bytes": cf_bytes,
                "submissions_status": info["submissions_status"],
                "submissions_path": info["submissions_path"],
                "submissions_bytes": info["submissions_bytes"],
                "sic": info["sic"],
                "sector_name": info["sector_name"],
                "sector_id": info["sector_id"],
                "sector_index": info["sector_index"],
                "submissions_error": info["submissions_error"],
            }
        )

        sector_id = str(info.get("sector_id", "")).strip()
        sector_name = str(info.get("sector_name", "")).strip()
        sic = str(info.get("sic", "")).strip()
        submissions_status = str(info.get("submissions_status", "")).strip()

        if submissions_status == "ok" and sector_id:
            ticker_sector_rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "sector_id": sector_id,
                    "sector_name": sector_name,
                    "sic": sic,
                    "source": Path(info["submissions_path"]).name,
                }
            )
        sector_index = _sector_id_to_index(sector_id)
        if sector_index is not None and submissions_status == "ok":
            cik_to_sector_index[cik] = sector_index

    manifest_path = out_dir / "universe_sec_manifest.csv"
    _write_csv_with_columns(
        str(manifest_path),
        manifest_rows,
        columns=[
            "ticker",
            "cik",
            "cik_status",
            "companyfacts_status",
            "companyfacts_path",
            "companyfacts_bytes",
            "submissions_status",
            "submissions_path",
            "submissions_bytes",
            "sic",
            "sector_name",
            "sector_id",
            "sector_index",
            "submissions_error",
        ],
    )

    ticker_sector_rows = sorted(ticker_sector_rows, key=lambda r: (r["ticker"], r["sector_id"]))
    ticker_sector_path = out_dir / "universe_ticker_to_sector.csv"
    _write_csv_with_columns(
        str(ticker_sector_path),
        ticker_sector_rows,
        columns=["ticker", "cik", "sector_id", "sector_name", "sic", "source"],
    )

    sector_to_tickers: dict[str, list[str]] = {f"S{i}": [] for i in range(10)}
    for r in ticker_sector_rows:
        sid = r["sector_id"]
        if sid in sector_to_tickers:
            sector_to_tickers[sid].append(r["ticker"])
    for sid in sector_to_tickers:
        sector_to_tickers[sid] = sorted(set(sector_to_tickers[sid]))

    sector_to_tickers_path = out_dir / "universe_sector_to_tickers.json"
    with open(sector_to_tickers_path, "w", encoding="utf-8") as f:
        json.dump(sector_to_tickers, f, ensure_ascii=False, indent=2, sort_keys=True)

    cik_to_sector_index_path = out_dir / "universe_cik_to_sector_index.json"
    with open(cik_to_sector_index_path, "w", encoding="utf-8") as f:
        json.dump(cik_to_sector_index, f, ensure_ascii=False, indent=2, sort_keys=True)

    missing_cik_path = out_dir / "missing_cik_tickers.txt"
    with open(missing_cik_path, "w", encoding="utf-8") as f:
        for t in sorted(set(missing)):
            f.write(f"{t}\n")

    summary = {
        "tickers_total": len(tickers),
        "tickers_resolved_cik": len(resolved_items),
        "tickers_missing_cik": len(missing),
        "companyfacts_dir": str(companyfacts_dir),
        "submissions_dir": str(submissions_dir),
        "min_companyfacts_bytes": min_companyfacts_bytes,
        "tickers_with_sector_mapped": len(ticker_sector_rows),
        "sector_counts": {k: len(v) for k, v in sector_to_tickers.items()},
    }
    summary_path = out_dir / "universe_sec_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {ticker_sector_path}")
    print(f"Wrote: {sector_to_tickers_path}")
    print(f"Wrote: {cik_to_sector_index_path}")
    print(f"Wrote: {summary_path}")
    print(f"Tickers: total={len(tickers)} resolved={len(resolved_items)} missing_cik={len(missing)}")
    print(f"Sector-mapped tickers: {len(ticker_sector_rows)}")
    return 0


def _read_manifest_selected_tickers(
    manifest_csv: str,
    *,
    require_companyfacts_ok: bool,
    require_submissions_ok: bool,
    require_sector_id: bool,
) -> list[str]:
    selected: list[str] = []
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = _normalize_ticker(row.get("ticker", ""))
            if not ticker:
                continue
            if str(row.get("cik_status", "")).strip() != "ok":
                continue
            if require_companyfacts_ok and str(row.get("companyfacts_status", "")).strip() != "ok":
                continue
            if require_submissions_ok and str(row.get("submissions_status", "")).strip() != "ok":
                continue
            if require_sector_id and not str(row.get("sector_id", "")).strip():
                continue
            selected.append(ticker)
    return sorted(set(selected))


def _read_csv_header(in_csv: str) -> list[str]:
    with open(in_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _filter_wide_csv(
    in_csv: str,
    out_csv: str,
    *,
    date_col: str,
    tickers_to_keep: list[str],
    progress_every: int = 0,
) -> None:
    header = _read_csv_header(in_csv)
    if not header:
        raise ValueError(f"Empty CSV (no header): {in_csv}")
    if date_col not in header:
        raise ValueError(f"CSV missing date column '{date_col}': {in_csv}")

    col_to_idx = {c: i for i, c in enumerate(header)}
    date_idx = col_to_idx[date_col]

    keep_tickers = [t for t in tickers_to_keep if t in col_to_idx]
    if len(keep_tickers) < 2:
        raise ValueError(f"Need at least 2 tickers after filtering for {in_csv}, got {len(keep_tickers)}")

    idxs = [date_idx] + [col_to_idx[t] for t in keep_tickers]
    out_header = [header[i] for i in idxs]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_csv, "r", newline="", encoding="utf-8") as fin, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        _ = next(reader, None)  # consume header
        writer.writerow(out_header)

        for row_i, row in enumerate(reader, start=1):
            out_row = [row[i] if i < len(row) else "" for i in idxs]
            writer.writerow(out_row)
            if progress_every and row_i % progress_every == 0:
                print(f"  {out_path.name}: wrote {row_i} rows...")


def cmd_filter_universe(args: argparse.Namespace) -> int:
    manifest_csv = args.manifest_csv
    prices_csv = args.prices_csv
    scores_csv = args.scores_csv
    out_dir = Path(args.out_dir)

    for p in [manifest_csv, prices_csv, scores_csv]:
        if not Path(p).exists():
            raise SystemExit(f"Missing required file: {p}")

    selected = _read_manifest_selected_tickers(
        manifest_csv,
        require_companyfacts_ok=bool(args.require_companyfacts_ok),
        require_submissions_ok=bool(args.require_submissions_ok),
        require_sector_id=bool(args.require_sector_id),
    )

    if args.max_tickers and args.max_tickers > 0:
        selected = selected[: args.max_tickers]

    prices_header = _read_csv_header(prices_csv)
    scores_header = _read_csv_header(scores_csv)

    if args.date_col not in prices_header:
        raise SystemExit(f"prices_csv missing date column '{args.date_col}': {prices_csv}")
    if args.date_col not in scores_header:
        raise SystemExit(f"scores_csv missing date column '{args.date_col}': {scores_csv}")

    prices_tickers = set(prices_header) - {args.date_col}
    scores_tickers = set(scores_header) - {args.date_col}

    selected_set = set(selected)
    common = sorted(selected_set.intersection(prices_tickers).intersection(scores_tickers))
    if len(common) < 2:
        raise SystemExit(f"Too few tickers after intersection across manifest/prices/scores: {len(common)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    prices_out = out_dir / "prices.csv"
    scores_out = out_dir / "scores.csv"

    print(f"Selected tickers (manifest): {len(selected)}")
    print(f"Tickers kept (intersection): {len(common)}")
    print(f"Writing: {prices_out}")
    _filter_wide_csv(
        prices_csv,
        str(prices_out),
        date_col=args.date_col,
        tickers_to_keep=common,
        progress_every=int(args.progress_every_rows or 0),
    )
    print(f"Writing: {scores_out}")
    _filter_wide_csv(
        scores_csv,
        str(scores_out),
        date_col=args.date_col,
        tickers_to_keep=common,
        progress_every=int(args.progress_every_rows or 0),
    )

    tickers_txt = out_dir / "tickers.txt"
    with open(tickers_txt, "w", encoding="utf-8") as f:
        for t in common:
            f.write(f"{t}\n")

    filtered_manifest = out_dir / "universe_sec_manifest.csv"
    with open(manifest_csv, "r", newline="", encoding="utf-8") as fin, open(
        filtered_manifest, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {manifest_csv}")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        common_set = set(common)
        for row in reader:
            ticker = _normalize_ticker(row.get("ticker", ""))
            if ticker in common_set:
                writer.writerow(row)

    # Optional: filtered sector_to_tickers mapping
    sector_in = Path(args.sector_to_tickers_json_in)
    sector_out = out_dir / "sector_to_tickers.json"
    if sector_in.exists():
        with open(sector_in, "r", encoding="utf-8") as f:
            sector_to_tickers = json.load(f)
        if isinstance(sector_to_tickers, dict):
            filtered = {}
            common_set = set(common)
            for k in sorted(sector_to_tickers.keys()):
                tickers = sector_to_tickers.get(k, [])
                if isinstance(tickers, list):
                    filtered[k] = sorted({str(t).upper() for t in tickers if str(t).upper() in common_set})
            with open(sector_out, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2, sort_keys=True)

    summary = {
        "manifest_csv": manifest_csv,
        "prices_csv_in": prices_csv,
        "scores_csv_in": scores_csv,
        "out_dir": str(out_dir),
        "date_col": args.date_col,
        "require_companyfacts_ok": bool(args.require_companyfacts_ok),
        "require_submissions_ok": bool(args.require_submissions_ok),
        "require_sector_id": bool(args.require_sector_id),
        "tickers_selected_manifest": len(selected),
        "tickers_kept": len(common),
    }
    with open(out_dir / "universe_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote: {tickers_txt}")
    print(f"Wrote: {filtered_manifest}")
    print(f"Wrote: {out_dir / 'universe_build_summary.json'}")
    if sector_out.exists():
        print(f"Wrote: {sector_out}")
    return 0


def _download_sec_json_by_cik(
    *,
    ciks: list[str],
    url_template: str,
    out_dir: str,
    user_agent: str,
    timeout_sec: float,
    retries: int,
    min_sleep_sec: float,
    overwrite: bool,
    dry_run: bool,
    progress_every: int,
) -> dict[str, DownloadResult]:
    out = {}
    headers = _sec_headers(user_agent)
    out_dir_path = Path(out_dir)
    for i, cik in enumerate(ciks, start=1):
        url = url_template.format(cik=cik)
        out_path = out_dir_path / f"CIK{cik}.json"
        res = _download_to_path(
            url,
            out_path=out_path,
            headers=headers,
            timeout_sec=timeout_sec,
            retries=retries,
            min_sleep_sec=min_sleep_sec,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        out[cik] = res
        if progress_every and (i % progress_every == 0 or i == len(ciks)):
            done = sum(1 for r in out.values() if r.status in {"downloaded", "skipped"})
            failed = sum(1 for r in out.values() if r.status == "failed")
            print(f"Progress: {i}/{len(ciks)}  ok={done}  failed={failed}")
        if min_sleep_sec and not dry_run:
            time.sleep(min_sleep_sec)
    return out


def cmd_companyfacts(args: argparse.Namespace) -> int:
    ua = _require_user_agent(args)
    tickers = _collect_tickers_from_args(args)
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers / --tickers_file / --universe_prices_csv.")

    ticker_map_json = _ensure_ticker_map_json(
        user_agent=ua,
        ticker_map_json=args.ticker_map_json,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        min_sleep_sec=args.min_sleep_sec,
        dry_run=args.dry_run,
    )
    ticker_to_cik = _load_company_ticker_map(ticker_map_json)
    resolved, missing = _resolve_ciks(tickers, ticker_to_cik=ticker_to_cik)

    # Limit after resolving (deterministic)
    resolved_items = sorted(resolved.items())
    if args.max_tickers and args.max_tickers > 0:
        resolved_items = resolved_items[: args.max_tickers]
    unique_ciks = sorted({cik for _, cik in resolved_items})

    print(f"Tickers: {len(tickers)}  resolved: {len(resolved_items)}  missing: {len(missing)}")
    print(f"Unique CIKs to fetch: {len(unique_ciks)}")
    if missing and args.print_missing:
        print("Missing tickers (no CIK match):")
        for t in missing[:200]:
            print(f"  {t}")
        if len(missing) > 200:
            print(f"  ... ({len(missing) - 200} more)")

    cik_results = _download_sec_json_by_cik(
        ciks=unique_ciks,
        url_template="https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        out_dir=args.out_dir,
        user_agent=ua,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        min_sleep_sec=args.min_sleep_sec,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        progress_every=args.progress_every,
    )

    report_rows: list[dict] = []
    cik_to_status = {cik: r.status for cik, r in cik_results.items()}
    cik_to_path = {cik: r.path for cik, r in cik_results.items()}
    cik_to_error = {cik: r.error for cik, r in cik_results.items()}

    for ticker, cik in resolved_items:
        report_rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                "status": cik_to_status.get(cik, ""),
                "path": cik_to_path.get(cik, ""),
                "error": cik_to_error.get(cik, ""),
            }
        )
    for ticker in sorted(set(missing)):
        report_rows.append({"ticker": ticker, "cik": "", "status": "missing", "path": "", "error": ""})

    _write_report_csv(args.report_csv, report_rows)

    failed = [cik for cik, r in cik_results.items() if r.status == "failed"]
    if failed:
        print(f"Failed CIK downloads: {len(failed)} (see --report_csv for details)")
        return 1
    return 0


def cmd_submissions(args: argparse.Namespace) -> int:
    ua = _require_user_agent(args)
    tickers = _collect_tickers_from_args(args)
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers / --tickers_file / --universe_prices_csv.")

    ticker_map_json = _ensure_ticker_map_json(
        user_agent=ua,
        ticker_map_json=args.ticker_map_json,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        min_sleep_sec=args.min_sleep_sec,
        dry_run=args.dry_run,
    )
    ticker_to_cik = _load_company_ticker_map(ticker_map_json)
    resolved, missing = _resolve_ciks(tickers, ticker_to_cik=ticker_to_cik)

    resolved_items = sorted(resolved.items())
    if args.max_tickers and args.max_tickers > 0:
        resolved_items = resolved_items[: args.max_tickers]
    unique_ciks = sorted({cik for _, cik in resolved_items})

    print(f"Tickers: {len(tickers)}  resolved: {len(resolved_items)}  missing: {len(missing)}")
    print(f"Unique CIKs to fetch: {len(unique_ciks)}")

    cik_results = _download_sec_json_by_cik(
        ciks=unique_ciks,
        url_template="https://data.sec.gov/submissions/CIK{cik}.json",
        out_dir=args.out_dir,
        user_agent=ua,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        min_sleep_sec=args.min_sleep_sec,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        progress_every=args.progress_every,
    )

    report_rows: list[dict] = []
    cik_to_status = {cik: r.status for cik, r in cik_results.items()}
    cik_to_path = {cik: r.path for cik, r in cik_results.items()}
    cik_to_error = {cik: r.error for cik, r in cik_results.items()}

    for ticker, cik in resolved_items:
        report_rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                "status": cik_to_status.get(cik, ""),
                "path": cik_to_path.get(cik, ""),
                "error": cik_to_error.get(cik, ""),
            }
        )
    for ticker in sorted(set(missing)):
        report_rows.append({"ticker": ticker, "cik": "", "status": "missing", "path": "", "error": ""})

    _write_report_csv(args.report_csv, report_rows)

    failed = [cik for cik, r in cik_results.items() if r.status == "failed"]
    if failed:
        print(f"Failed CIK downloads: {len(failed)} (see --report_csv for details)")
        return 1
    return 0


def _download_zip_dataset(
    *,
    url: str,
    out_zip: Path,
    user_agent: str,
    timeout_sec: float,
    retries: int,
    min_sleep_sec: float,
    overwrite: bool,
    extract: bool,
    dry_run: bool,
) -> DownloadResult:
    res = _download_to_path(
        url,
        out_path=out_zip,
        headers=_sec_headers(user_agent),
        timeout_sec=timeout_sec,
        retries=retries,
        min_sleep_sec=min_sleep_sec,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if res.status == "failed" or not extract or dry_run:
        return res
    try:
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        extract_dir = out_zip.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        return DownloadResult(status="failed", path=str(out_zip), error=f"extract failed: {e}")
    return res


def cmd_13f(args: argparse.Namespace) -> int:
    ua = _require_user_agent(args)
    q = int(args.quarter)
    if q not in {1, 2, 3, 4}:
        raise SystemExit("--quarter must be 1..4")
    y = int(args.year)
    url = f"https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{y}q{q}_form13f.zip"
    out_zip = Path(args.out_dir) / f"{y}q{q}_form13f.zip"
    res = _download_zip_dataset(
        url=url,
        out_zip=out_zip,
        user_agent=ua,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        min_sleep_sec=args.min_sleep_sec,
        overwrite=args.overwrite,
        extract=args.extract,
        dry_run=args.dry_run,
    )
    print(f"{res.status}: {url} -> {res.path}")
    if res.error:
        print(res.error)
        return 1
    return 0


def cmd_form345(args: argparse.Namespace) -> int:
    ua = _require_user_agent(args)
    q = int(args.quarter)
    if q not in {1, 2, 3, 4}:
        raise SystemExit("--quarter must be 1..4")
    y = int(args.year)
    url = (
        "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/"
        f"{y}q{q}_form345.zip"
    )
    out_zip = Path(args.out_dir) / f"{y}q{q}_form345.zip"
    res = _download_zip_dataset(
        url=url,
        out_zip=out_zip,
        user_agent=ua,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        min_sleep_sec=args.min_sleep_sec,
        overwrite=args.overwrite,
        extract=args.extract,
        dry_run=args.dry_run,
    )
    print(f"{res.status}: {url} -> {res.path}")
    if res.error:
        print(res.error)
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download free SEC datasets with caching.")
    p.add_argument("--user_agent", default="", help="SEC User-Agent (or set SEC_USER_AGENT env var)")
    p.add_argument("--timeout_sec", type=float, default=60.0)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--min_sleep_sec", type=float, default=0.2, help="Base sleep between requests (seconds)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_tickers = sub.add_parser("tickers", help="Download SEC ticker<->CIK map JSON")
    p_tickers.add_argument("--exchange", action="store_true", help="Use company_tickers_exchange.json")
    p_tickers.add_argument("--out_json", default="data/raw/sec/company_tickers_exchange.json")
    p_tickers.set_defaults(func=cmd_tickers)

    def add_common_ticker_inputs(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--tickers", nargs="*", default=[])
        sp.add_argument("--tickers_file", default="", help="txt: one ticker per line")
        sp.add_argument("--universe_prices_csv", default="", help="prices.csv with tickers in header")
        sp.add_argument("--universe_scores_csv", default="", help="scores.csv with tickers in header")
        sp.add_argument("--ticker_map_json", default="data/raw/sec/company_tickers_exchange.json")
        sp.add_argument("--max_tickers", type=int, default=0, help="Limit resolved tickers for testing")
        sp.add_argument("--progress_every", type=int, default=100)
        sp.add_argument("--report_csv", default="")

    p_cf = sub.add_parser("companyfacts", help="Download SEC companyfacts JSON by CIK")
    add_common_ticker_inputs(p_cf)
    p_cf.add_argument("--out_dir", default="data/raw/sec_bulk")
    p_cf.add_argument("--print_missing", action="store_true")
    p_cf.set_defaults(func=cmd_companyfacts)

    p_sub = sub.add_parser("submissions", help="Download SEC submissions JSON by CIK")
    add_common_ticker_inputs(p_sub)
    p_sub.add_argument("--out_dir", default="data/sec_submissions")
    p_sub.set_defaults(func=cmd_submissions)

    p_13f = sub.add_parser("13f", help="Download SEC 13F structured dataset zip (by quarter)")
    p_13f.add_argument("--year", required=True)
    p_13f.add_argument("--quarter", required=True)
    p_13f.add_argument("--out_dir", default="data/raw/sec_structured/13f")
    p_13f.add_argument("--extract", action="store_true")
    p_13f.set_defaults(func=cmd_13f)

    p_345 = sub.add_parser("form345", help="Download SEC insider (Form 3/4/5) dataset zip (by quarter)")
    p_345.add_argument("--year", required=True)
    p_345.add_argument("--quarter", required=True)
    p_345.add_argument("--out_dir", default="data/raw/sec_structured/form345")
    p_345.add_argument("--extract", action="store_true")
    p_345.set_defaults(func=cmd_form345)

    p_prep = sub.add_parser("preprocess", help="Build universe manifests from downloaded SEC files")
    add_common_ticker_inputs(p_prep)
    p_prep.add_argument("--companyfacts_dir", default="data/raw/sec_bulk")
    p_prep.add_argument("--submissions_dir", default="data/sec_submissions")
    p_prep.add_argument("--min_companyfacts_bytes", type=int, default=2000)
    p_prep.add_argument("--out_dir", default="data/processed/sec_universe")
    p_prep.set_defaults(func=cmd_preprocess)

    p_filter = sub.add_parser("filter_universe", help="Filter prices/scores to fundamentals-ready tickers")
    p_filter.add_argument("--manifest_csv", default="data/processed/sec_universe/universe_sec_manifest.csv")
    p_filter.add_argument("--prices_csv", default="data/backtest_universe_full/prices.csv")
    p_filter.add_argument("--scores_csv", default="data/backtest_universe_full/scores.csv")
    p_filter.add_argument("--out_dir", default="data/backtest_universe_sec")
    p_filter.add_argument("--date_col", default="date")
    p_filter.add_argument("--max_tickers", type=int, default=0, help="Limit tickers for testing")
    p_filter.add_argument("--progress_every_rows", type=int, default=0, help="Print progress every N rows")
    p_filter.add_argument(
        "--require_companyfacts_ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require companyfacts_status == ok",
    )
    p_filter.add_argument(
        "--require_submissions_ok",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require submissions_status == ok",
    )
    p_filter.add_argument(
        "--require_sector_id",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require non-empty sector_id (from submissions SIC mapping)",
    )
    p_filter.add_argument(
        "--sector_to_tickers_json_in",
        default="data/processed/sec_universe/universe_sector_to_tickers.json",
        help="Optional input sector_to_tickers JSON to filter alongside prices/scores",
    )
    p_filter.set_defaults(func=cmd_filter_universe)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
