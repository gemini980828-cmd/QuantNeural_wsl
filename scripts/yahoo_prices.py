"""
Yahoo Finance daily OHLCV loader (chart API) with optional on-disk caching.

This is a small utility for strategy backtests that need:
- Daily OHLCV
- Dividend events (cash per share, ex-date)
- Split events (share multiplier on ex-date)

Implementation notes
--------------------
- Uses the public chart endpoint:
  https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}
- Trading dates are normalized to America/New_York midnight (naive).
- Cache format is a single CSV per ticker with columns:
  date, open, high, low, close, volume, dividend, split_ratio
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


NY_TZ = "America/New_York"


@dataclass(frozen=True)
class YahooDailyFrame:
    ohlcv: pd.DataFrame

    @property
    def dividends(self) -> pd.Series:
        if "dividend" not in self.ohlcv.columns:
            return pd.Series([], dtype=float)
        s = self.ohlcv["dividend"].copy()
        s.name = "dividend"
        return s

    @property
    def splits(self) -> pd.Series:
        if "split_ratio" not in self.ohlcv.columns:
            return pd.Series([], dtype=float)
        s = self.ohlcv["split_ratio"].copy()
        s.name = "split_ratio"
        return s


def _to_unix_seconds(date: str, *, add_days: int = 0) -> int:
    ts = pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=add_days)
    return int(ts.timestamp())


def _chart_url(ticker: str, *, start_date: str, end_date: str) -> str:
    # Yahoo chart uses unix seconds; use end_date+1d so end_date is included.
    p1 = _to_unix_seconds(start_date)
    p2 = _to_unix_seconds(end_date, add_days=1)
    events = "div%7Csplit"
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events={events}&includeAdjustedClose=true"
    )


def _http_get_json(
    url: str,
    *,
    timeout_sec: float = 30.0,
    retries: int = 3,
    min_sleep_sec: float = 0.5,
    user_agent: str = "Mozilla/5.0",
) -> dict:
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json,text/plain,*/*",
        "Connection": "close",
    }
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError {getattr(e, 'code', None)}: {e}"
            if getattr(e, "code", None) in {400, 401, 403, 404}:
                break
        except urllib.error.URLError as e:
            last_err = f"URLError: {e}"
        except Exception as e:  # pragma: no cover
            last_err = str(e)
            break

        if attempt < retries:
            time.sleep(min(10.0, min_sleep_sec * (2 ** (attempt - 1))))

    raise RuntimeError(f"Failed to fetch URL after {retries} retries: {last_err}\nURL={url}")


def _normalize_trading_dates(timestamps: list[int]) -> pd.DatetimeIndex:
    # Yahoo daily timestamps are at market open in UTC; normalize to NY date.
    dt = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(NY_TZ).normalize()
    return dt.tz_localize(None)


def _parse_chart_payload(payload: dict) -> YahooDailyFrame:
    chart = payload.get("chart", {})
    err = chart.get("error")
    if err:
        raise ValueError(f"Yahoo chart error: {err}")

    result = chart.get("result")
    if not result:
        raise ValueError("Yahoo chart payload missing chart.result")

    r0 = result[0]
    timestamps = r0.get("timestamp")
    if not timestamps:
        raise ValueError("Yahoo chart payload missing timestamp data")

    quotes = (r0.get("indicators", {}) or {}).get("quote", [])
    if not quotes:
        raise ValueError("Yahoo chart payload missing indicators.quote")
    q0 = quotes[0]

    idx = _normalize_trading_dates(list(timestamps))

    df = pd.DataFrame(
        {
            "open": q0.get("open", []),
            "high": q0.get("high", []),
            "low": q0.get("low", []),
            "close": q0.get("close", []),
            "volume": q0.get("volume", []),
        },
        index=idx,
    )

    # Drop rows with missing closes (no trading).
    df = df.dropna(subset=["close"]).copy()

    # Coerce types.
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df["volume"] = df["volume"].fillna(0.0)

    # Events: dividends + splits.
    df["dividend"] = 0.0
    df["split_ratio"] = 1.0

    events = r0.get("events", {}) or {}

    dividends = events.get("dividends", {}) or {}
    if dividends:
        div_rows = []
        for v in dividends.values():
            ts = v.get("date")
            amt = v.get("amount")
            if ts is None or amt is None:
                continue
            div_rows.append((int(ts), float(amt)))
        if div_rows:
            div_ts, div_amt = zip(*div_rows)
            div_idx = _normalize_trading_dates(list(div_ts))
            div_s = pd.Series(div_amt, index=div_idx, dtype=float).groupby(level=0).sum()
            div_s.index = div_s.index.tz_localize(None)
            df["dividend"] = div_s.reindex(df.index).fillna(0.0)

    splits = events.get("splits", {}) or {}
    if splits:
        split_rows = []
        for v in splits.values():
            ts = v.get("date")
            num = v.get("numerator")
            den = v.get("denominator")
            if ts is None or num is None or den in (None, 0):
                continue
            split_rows.append((int(ts), float(num) / float(den)))
        if split_rows:
            split_ts, split_ratio = zip(*split_rows)
            split_idx = _normalize_trading_dates(list(split_ts))
            split_s = pd.Series(split_ratio, index=split_idx, dtype=float).groupby(level=0).prod()
            split_s.index = split_s.index.tz_localize(None)
            df["split_ratio"] = split_s.reindex(df.index).fillna(1.0)

    df = df.sort_index()
    df.index.name = "date"
    return YahooDailyFrame(ohlcv=df)


def _cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("^", "_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe}_1d.csv")


def load_yahoo_daily(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> YahooDailyFrame:
    """
    Load daily OHLCV (+dividend/split events) from Yahoo chart API.

    If cache_dir is provided, results are cached to CSV and reused unless refresh=True.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, ticker)
        if os.path.exists(path) and not refresh:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.set_index("date").sort_index()
            for col in ["open", "high", "low", "close", "volume", "dividend", "split_ratio"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"]).copy()
            df["volume"] = df.get("volume", 0.0).fillna(0.0)
            df["dividend"] = df.get("dividend", 0.0).fillna(0.0)
            df["split_ratio"] = df.get("split_ratio", 1.0).fillna(1.0)
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()
            return YahooDailyFrame(ohlcv=df)

    url = _chart_url(ticker, start_date=start_date, end_date=end_date)
    payload = _http_get_json(url)
    frame = _parse_chart_payload(payload)
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    frame = YahooDailyFrame(ohlcv=frame.ohlcv[(frame.ohlcv.index >= start_dt) & (frame.ohlcv.index <= end_dt)].copy())

    if cache_dir:
        out_path = _cache_path(cache_dir, ticker)
        frame.ohlcv.reset_index().to_csv(out_path, index=False)

    return frame
