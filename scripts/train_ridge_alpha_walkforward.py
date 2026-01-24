"""
Walk-forward Ridge "alpha" baseline on a market-cap-gated universe.

This is a research script: it builds a PIT-safe feature panel using
  - Stooq prices (daily, wide)
  - SEC companyfacts (local JSON per ticker)
  - Market-cap eligibility mask (from scripts/build_market_cap_universe.py)

Then it trains a simple Ridge regression on past data only and produces a
backtest-ready `scores.csv` (wide: date + tickers).

Key invariants
--------------
- PIT-safe: fundamentals use only filings with filed <= date.
- No look-ahead: training at date t uses labels from dates < t only.
- Deterministic: no randomness; fixed ticker/date ordering.

Recommended workflow
--------------------
1) Build SEC + fundamentals-ready universe:
   python scripts/download_sec_data.py preprocess ...
   python scripts/download_sec_data.py filter_universe ...

2) Build market-cap gate (writes market_cap.csv + eligibility.csv):
   python scripts/build_market_cap_universe.py ...

3) Train / export ML scores:
   python scripts/train_ridge_alpha_walkforward.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# Allow running this script directly via `python scripts/...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.endswith(".US"):
        t = t[: -len(".US")]
    return t


def _read_wide_panel_csv(path: str, *, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"CSV missing date column '{date_col}': {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    if not df.index.is_unique:
        raise ValueError(f"CSV has duplicate dates: {path}")
    df.columns = [_normalize_ticker(c) for c in df.columns]
    df = df.loc[:, df.columns.notna()]
    return df


def _load_ticker_to_sector_ids(ticker_to_sector_csv: str, *, tickers: list[str]) -> list[str]:
    df = pd.read_csv(ticker_to_sector_csv)
    required = ["ticker", "sector_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ticker_to_sector_csv missing columns {missing}: {ticker_to_sector_csv}")

    df = df.copy()
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    df["sector_id"] = df["sector_id"].fillna("").astype(str)

    # Deterministic: smallest (sector_id, ticker) wins if duplicates exist
    df = df.sort_values(["ticker", "sector_id"], kind="mergesort").drop_duplicates(subset=["ticker"], keep="first")
    mapping = dict(zip(df["ticker"].tolist(), df["sector_id"].tolist()))

    out: list[str] = []
    for t in tickers:
        s = mapping.get(t, "")
        out.append(s if s else "UNKNOWN")
    return out


def _build_sector_members(sector_ids: list[str]) -> list[tuple[str, np.ndarray]]:
    sector_arr = np.asarray(sector_ids, dtype=str)
    unique = sorted(set(sector_arr.tolist()))
    return [(s, np.where(sector_arr == s)[0]) for s in unique]


def _sector_neutralize_rows_inplace(
    arr: np.ndarray,
    *,
    sector_members: list[tuple[str, np.ndarray]],
    eligible_mask: np.ndarray,
    min_count: int,
) -> None:
    """
    In-place per-row sector de-meaning using eligible & finite tickers only.
    """
    if min_count < 1:
        raise ValueError("--sector_min_count must be >= 1")
    if arr.ndim != 2:
        raise ValueError("arr must be 2D [date, ticker]")
    if eligible_mask.shape != arr.shape:
        raise ValueError("eligible_mask shape must match arr shape")

    n_rows = arr.shape[0]
    for i in range(n_rows):
        row = arr[i]
        base_mask = eligible_mask[i] & np.isfinite(row)
        if int(base_mask.sum()) < 2:
            continue
        for _, members in sector_members:
            m = base_mask[members]
            if int(m.sum()) < int(min_count):
                continue
            idx = members[m]
            mu = float(np.mean(row[idx]))
            if np.isfinite(mu):
                row[idx] = row[idx] - mu


def _zscore_rows_inplace(arr: np.ndarray, *, eligible_mask: np.ndarray) -> None:
    """
    In-place per-row z-score using eligible & finite tickers only.
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D [date, ticker]")
    if eligible_mask.shape != arr.shape:
        raise ValueError("eligible_mask shape must match arr shape")

    n_rows = arr.shape[0]
    for i in range(n_rows):
        row = arr[i]
        m = eligible_mask[i] & np.isfinite(row)
        if int(m.sum()) < 2:
            continue
        vals = row[m]
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if not np.isfinite(mu) or not np.isfinite(sd) or sd < 1e-12:
            continue
        row[m] = (vals - mu) / sd


def _rank_transform_rows(arr: np.ndarray, *, eligible_mask: np.ndarray) -> np.ndarray:
    """
    Per-row cross-sectional rank transform using eligible & finite tickers only.

    Returns centered percentile ranks in [-0.5, 0.5].
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D [date, ticker]")
    if eligible_mask.shape != arr.shape:
        raise ValueError("eligible_mask shape must match arr shape")

    out = np.full(arr.shape, np.nan, dtype=float)
    n_rows = arr.shape[0]
    for i in range(n_rows):
        m = eligible_mask[i] & np.isfinite(arr[i])
        if int(m.sum()) < 2:
            continue
        r = pd.Series(arr[i, m]).rank(method="average", pct=True).to_numpy(dtype=float)
        out[i, m] = r - 0.5
    return out


def _load_long_feature_panels(
    path: str,
    *,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    date_col: str,
    ticker_col: str,
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date_col='{date_col}' in long feature CSV: {path}")
    if ticker_col not in df.columns:
        raise ValueError(f"Missing ticker_col='{ticker_col}' in long feature CSV: {path}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[ticker_col] = df[ticker_col].map(_normalize_ticker)
    df = df.dropna(subset=[date_col, ticker_col])

    feature_cols = [c for c in df.columns if c not in {date_col, ticker_col}]
    if not feature_cols:
        raise ValueError(f"No feature columns found in long feature CSV: {path}")

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    df = df[df[date_col].isin(dates) & df[ticker_col].isin(tickers)]
    if df.empty:
        raise ValueError(f"No rows remain after filtering to (dates,tickers) intersection: {path}")

    g = df.groupby([date_col, ticker_col], sort=True, as_index=False)[feature_cols].sum()
    out: dict[str, pd.DataFrame] = {}
    for c in feature_cols:
        pvt = g.pivot(index=date_col, columns=ticker_col, values=c)
        pvt = pvt.reindex(index=dates, columns=tickers).fillna(0.0).astype(float)
        out[c] = pvt
    return out


def _load_manifest_companyfacts_paths(manifest_csv: str) -> dict[str, str]:
    df = pd.read_csv(manifest_csv)
    required = ["ticker", "cik_status", "companyfacts_status", "companyfacts_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest_csv missing columns {missing}: {manifest_csv}")
    df = df.copy()
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    df = df[(df["cik_status"] == "ok") & (df["companyfacts_status"] == "ok")]
    out = dict(zip(df["ticker"].tolist(), df["companyfacts_path"].tolist()))
    if not out:
        raise ValueError("No (ticker -> companyfacts_path) rows after filtering manifest.")
    return out


def _extract_tag_entries(
    companyfacts_path: str,
    *,
    taxonomy: str,
    tag: str,
    unit_preference: str,
) -> pd.DataFrame:
    """
    Extract a single (taxonomy, tag) series from a companyfacts JSON file.

    Returns a tidy DataFrame with columns: end, filed, val.
    Empty DataFrame if missing or invalid.
    """
    try:
        with open(companyfacts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame(columns=["end", "filed", "val"])

    facts = data.get("facts", {})
    if not isinstance(facts, dict):
        return pd.DataFrame(columns=["end", "filed", "val"])

    tax = facts.get(taxonomy, {})
    if not isinstance(tax, dict):
        return pd.DataFrame(columns=["end", "filed", "val"])

    tag_obj = tax.get(tag, {})
    if not isinstance(tag_obj, dict):
        return pd.DataFrame(columns=["end", "filed", "val"])

    units = tag_obj.get("units", {})
    if not isinstance(units, dict) or not units:
        return pd.DataFrame(columns=["end", "filed", "val"])

    entries = units.get(unit_preference)
    if not isinstance(entries, list) or not entries:
        # Fallback to any unit (deterministic)
        unit_key = sorted(units.keys())[0]
        entries = units.get(unit_key, [])
        if not isinstance(entries, list) or not entries:
            return pd.DataFrame(columns=["end", "filed", "val"])

    rows: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        rows.append({"end": e.get("end"), "filed": e.get("filed"), "val": e.get("val")})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])

    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype(float)
    df = df.dropna(subset=["end", "filed", "val"])
    if df.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])

    df = df[np.isfinite(df["val"].values)]
    df = df.sort_values(["end", "filed"], kind="mergesort").reset_index(drop=True)
    return df


def _pit_latest_snapshot(entries_sorted: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Snapshot PIT rule at date t:
      choose the latest row by (end, filed) among rows with end <= t and filed <= t.
    """
    if entries_sorted.empty:
        return np.full(len(dates), np.nan, dtype=float)

    ends = entries_sorted["end"].to_numpy(dtype="datetime64[ns]")
    filed = entries_sorted["filed"].to_numpy(dtype="datetime64[ns]")
    vals = entries_sorted["val"].to_numpy(dtype=float)

    out = np.full(len(dates), np.nan, dtype=float)
    date_arr = dates.to_numpy(dtype="datetime64[ns]")
    for i, d in enumerate(date_arr):
        mask = (ends <= d) & (filed <= d)
        if mask.any():
            out[i] = vals[np.nonzero(mask)[0][-1]]
    return out


def _winsorize_rows_inplace(
    arr: np.ndarray,
    *,
    eligible_mask: np.ndarray,
    q_low: float,
    q_high: float,
) -> None:
    """
    In-place per-row winsorization using eligible tickers only.
    """
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("winsorize quantiles must satisfy 0 <= q_low < q_high <= 1")
    n_rows = arr.shape[0]
    for i in range(n_rows):
        m = eligible_mask[i]
        if m.sum() < 2:
            continue
        row = arr[i, m]
        lo = np.quantile(row, q_low)
        hi = np.quantile(row, q_high)
        arr[i] = np.clip(arr[i], lo, hi)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward Ridge alpha baseline (price + fundamentals).")
    p.add_argument("--manifest_csv", default="data/backtest_universe_sec/universe_sec_manifest.csv")
    p.add_argument("--prices_csv", default="data/backtest_universe_sec_mktcap/prices.csv")
    p.add_argument("--market_cap_csv", default="data/backtest_universe_sec_mktcap/market_cap.csv")
    p.add_argument("--eligibility_csv", default="data/backtest_universe_sec_mktcap/eligibility.csv")
    p.add_argument(
        "--insider_events_csv",
        default="",
        help="Optional long CSV (date,ticker,...) from scripts/build_form345_insider_features.py",
    )
    p.add_argument("--insider_ticker_col", default="ticker")
    p.add_argument(
        "--ticker_to_sector_csv",
        default="data/processed/sec_universe/universe_ticker_to_sector.csv",
        help="Used only if sector-neutralization is enabled.",
    )
    p.add_argument("--date_col", default="date")
    p.add_argument("--out_dir", default="data/backtest_universe_sec_mktcap_ridge")

    p.add_argument("--min_train_periods", type=int, default=60, help="Min monthly periods before first prediction")
    p.add_argument("--retrain_every", type=int, default=1, help="Retrain model every N periods (1=monthly)")
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--top_k", type=int, default=400, help="Safety: eligible_count >= top_k at every predicted date")
    p.add_argument("--ineligible_penalty", type=float, default=1e6)

    p.add_argument("--winsorize_features", action="store_true")
    p.add_argument("--winsorize_q_low", type=float, default=0.01)
    p.add_argument("--winsorize_q_high", type=float, default=0.99)
    p.add_argument("--cs_zscore_features", action="store_true", help="Cross-sectional z-score per date (eligible-only).")

    p.add_argument("--label_horizon", type=int, default=1, help="Forward return horizon in months (e.g., 3 for quarterly).")
    p.add_argument(
        "--label_transform",
        choices=["raw", "rank", "winsorize_zscore"],
        default="raw",
        help="Cross-sectional label transform per date (eligible-only).",
    )

    p.add_argument("--sector_neutralize_labels", action="store_true")
    p.add_argument("--sector_neutralize_scores", action="store_true")
    p.add_argument("--sector_min_count", type=int, default=20, help="Min tickers per sector per date to neutralize.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    for p in [args.manifest_csv, args.prices_csv, args.market_cap_csv, args.eligibility_csv]:
        if not Path(p).exists():
            raise SystemExit(f"Missing required file: {p}")

    ticker_to_companyfacts = _load_manifest_companyfacts_paths(args.manifest_csv)

    prices = _read_wide_panel_csv(args.prices_csv, date_col=args.date_col)
    market_cap = _read_wide_panel_csv(args.market_cap_csv, date_col=args.date_col)
    eligibility = _read_wide_panel_csv(args.eligibility_csv, date_col=args.date_col)

    if not market_cap.index.equals(eligibility.index):
        raise SystemExit("market_cap and eligibility date indexes do not match")
    dates = market_cap.index

    # Normalize and intersect tickers
    tickers = sorted(set(prices.columns) & set(market_cap.columns) & set(eligibility.columns) & set(ticker_to_companyfacts.keys()))
    if len(tickers) < 2:
        raise SystemExit(f"Need at least 2 tickers after intersection, got {len(tickers)}")

    prices = prices[tickers].astype(float)
    market_cap = market_cap[tickers].astype(float)
    eligibility = eligibility[tickers].astype(int)
    eligible_mask = eligibility.to_numpy(dtype=bool)

    sector_members: list[tuple[str, np.ndarray]] | None = None
    if args.sector_neutralize_labels or args.sector_neutralize_scores:
        if not args.ticker_to_sector_csv or not Path(args.ticker_to_sector_csv).exists():
            raise SystemExit(
                f"Missing required --ticker_to_sector_csv for sector neutralization: {args.ticker_to_sector_csv}"
            )
        sector_ids = _load_ticker_to_sector_ids(args.ticker_to_sector_csv, tickers=tickers)
        sector_members = _build_sector_members(sector_ids)

    # Align daily prices to signal dates (PIT-safe ffill)
    prices_at = prices.reindex(dates, method="ffill")
    if prices_at.isna().any().any():
        raise SystemExit("prices_at has NaN after ffill alignment to score dates")

    # Price-derived features
    ret_1m = prices_at.pct_change(1)
    mom_3m = prices_at / prices_at.shift(3) - 1.0
    mom_6m = prices_at / prices_at.shift(6) - 1.0
    mom_12m = prices_at / prices_at.shift(12) - 1.0
    vol_3m = ret_1m.rolling(window=3, min_periods=3).std()

    # Shares (derived from market_cap / price) for a stable dilution proxy
    shares_est = market_cap / prices_at
    shares_chg_12m = shares_est / shares_est.shift(12) - 1.0

    # Fundamental snapshot features (PIT by filed<=date)
    n_dates = len(dates)
    n_tickers = len(tickers)
    assets = np.full((n_dates, n_tickers), np.nan, dtype=float)
    liabilities = np.full((n_dates, n_tickers), np.nan, dtype=float)
    equity = np.full((n_dates, n_tickers), np.nan, dtype=float)
    cash = np.full((n_dates, n_tickers), np.nan, dtype=float)

    for j, ticker in enumerate(tickers):
        path = ticker_to_companyfacts.get(ticker, "")
        if not path:
            continue

        a = _extract_tag_entries(path, taxonomy="us-gaap", tag="Assets", unit_preference="USD")
        l = _extract_tag_entries(path, taxonomy="us-gaap", tag="Liabilities", unit_preference="USD")
        e = _extract_tag_entries(path, taxonomy="us-gaap", tag="StockholdersEquity", unit_preference="USD")

        c = _extract_tag_entries(
            path,
            taxonomy="us-gaap",
            tag="CashAndCashEquivalentsAtCarryingValue",
            unit_preference="USD",
        )
        if c.empty:
            c = _extract_tag_entries(
                path,
                taxonomy="us-gaap",
                tag="CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                unit_preference="USD",
            )

        assets[:, j] = _pit_latest_snapshot(a, dates)
        liabilities[:, j] = _pit_latest_snapshot(l, dates)
        equity[:, j] = _pit_latest_snapshot(e, dates)
        cash[:, j] = _pit_latest_snapshot(c, dates)

        if (j + 1) % 200 == 0 or (j + 1) == len(tickers):
            print(f"Loaded fundamentals: {j+1}/{len(tickers)}")

    # Derived fundamental ratios
    mc_arr = market_cap.to_numpy(dtype=float)
    eps = 1e-18
    log_mktcap = np.log(np.maximum(mc_arr, eps))
    book_to_mkt = equity / np.maximum(mc_arr, eps)
    leverage = liabilities / np.maximum(assets, eps)
    cash_to_assets = cash / np.maximum(assets, eps)
    asset_growth_12m = assets / np.maximum(np.roll(assets, shift=12, axis=0), eps) - 1.0
    asset_growth_12m[:12, :] = np.nan

    insider_feature_frames: list[tuple[str, pd.DataFrame]] = []
    if args.insider_events_csv:
        if not Path(args.insider_events_csv).exists():
            raise SystemExit(f"Missing insider_events_csv: {args.insider_events_csv}")
        insider_panels = _load_long_feature_panels(
            args.insider_events_csv,
            dates=dates,
            tickers=tickers,
            date_col=args.date_col,
            ticker_col=args.insider_ticker_col,
        )

        mc_safe = market_cap.clip(lower=eps)
        if "buy_value" in insider_panels:
            buy_to_mc = insider_panels["buy_value"] / mc_safe
            insider_feature_frames.append(
                ("insider_buy_value_to_mktcap_3m", buy_to_mc.rolling(window=3, min_periods=1).sum())
            )
        if "sell_value" in insider_panels:
            sell_to_mc = insider_panels["sell_value"] / mc_safe
            insider_feature_frames.append(
                ("insider_sell_value_to_mktcap_3m", sell_to_mc.rolling(window=3, min_periods=1).sum())
            )
        if "net_value" in insider_panels:
            net_to_mc = insider_panels["net_value"] / mc_safe
            insider_feature_frames.append(
                ("insider_net_value_to_mktcap_3m", net_to_mc.rolling(window=3, min_periods=1).sum())
            )
        if "buy_count" in insider_panels:
            insider_feature_frames.append(
                ("insider_buy_count_3m", insider_panels["buy_count"].rolling(window=3, min_periods=1).sum())
            )
        if "sell_count" in insider_panels:
            insider_feature_frames.append(
                ("insider_sell_count_3m", insider_panels["sell_count"].rolling(window=3, min_periods=1).sum())
            )

    # Assemble feature tensor [date, ticker, feature]
    feature_frames: list[tuple[str, pd.DataFrame]] = [
        ("mom_1m", ret_1m),
        ("mom_3m", mom_3m),
        ("mom_6m", mom_6m),
        ("mom_12m", mom_12m),
        ("vol_3m", vol_3m),
        ("log_mktcap", pd.DataFrame(log_mktcap, index=dates, columns=tickers)),
        ("book_to_mkt", pd.DataFrame(book_to_mkt, index=dates, columns=tickers)),
        ("leverage", pd.DataFrame(leverage, index=dates, columns=tickers)),
        ("cash_to_assets", pd.DataFrame(cash_to_assets, index=dates, columns=tickers)),
        ("asset_growth_12m", pd.DataFrame(asset_growth_12m, index=dates, columns=tickers)),
        ("shares_chg_12m", shares_chg_12m),
    ]
    feature_frames.extend(insider_feature_frames)

    feature_names = [n for n, _ in feature_frames]
    X = np.stack([df.to_numpy(dtype=float) for _, df in feature_frames], axis=2)

    label_horizon = int(args.label_horizon)
    if label_horizon < 1:
        raise SystemExit("--label_horizon must be >= 1")

    # Label: forward return over label_horizon months (for diagnostics / training only)
    y_raw = (prices_at.shift(-label_horizon) / prices_at) - 1.0
    y_raw_arr = y_raw.to_numpy(dtype=float)

    # Optional sector neutralization (per-date) on labels
    y_arr = y_raw_arr.copy()
    if args.sector_neutralize_labels:
        if sector_members is None:
            raise SystemExit("sector_neutralize_labels requested but sector mapping was not loaded")
        _sector_neutralize_rows_inplace(
            y_arr,
            sector_members=sector_members,
            eligible_mask=eligible_mask,
            min_count=int(args.sector_min_count),
        )

    # Optional label transform (per-date cross-sectional)
    if args.label_transform == "raw":
        pass
    elif args.label_transform == "rank":
        y_arr = _rank_transform_rows(y_arr, eligible_mask=eligible_mask)
    elif args.label_transform == "winsorize_zscore":
        y_wz = y_arr.copy()
        _winsorize_rows_inplace(
            y_wz,
            eligible_mask=eligible_mask & np.isfinite(y_wz),
            q_low=args.winsorize_q_low,
            q_high=args.winsorize_q_high,
        )
        _zscore_rows_inplace(y_wz, eligible_mask=eligible_mask)
        y_arr = y_wz
    else:
        raise SystemExit(f"Unknown --label_transform: {args.label_transform}")

    # Mask: eligible and finite X/y (y is NaN on last date)
    X_finite = np.isfinite(X).all(axis=2)
    y_finite = np.isfinite(y_arr)
    sample_ok = eligible_mask & X_finite & y_finite

    # Optional feature winsorization (leakage-free: per-date cross-sectional)
    if args.winsorize_features:
        for k in range(X.shape[2]):
            _winsorize_rows_inplace(
                X[:, :, k],
                eligible_mask=eligible_mask & np.isfinite(X[:, :, k]),
                q_low=args.winsorize_q_low,
                q_high=args.winsorize_q_high,
            )

    # Optional feature cross-sectional z-score (leakage-free: per-date cross-sectional)
    if args.cs_zscore_features:
        for k in range(X.shape[2]):
            _zscore_rows_inplace(X[:, :, k], eligible_mask=eligible_mask)

    # Flatten once for efficient expanding-window training
    n_features = X.shape[2]
    X_flat = X.reshape(n_dates * n_tickers, n_features)
    y_flat = y_arr.reshape(n_dates * n_tickers)
    ok_flat = sample_ok.reshape(n_dates * n_tickers)

    # Walk-forward predictions
    min_train_periods = int(args.min_train_periods)
    retrain_every = int(args.retrain_every)
    if min_train_periods < 6:
        raise SystemExit("--min_train_periods must be >= 6")
    if retrain_every < 1:
        raise SystemExit("--retrain_every must be >= 1")

    pred_scores = np.full((n_dates, n_tickers), np.nan, dtype=float)

    scaler: StandardScaler | None = None
    model: Ridge | None = None
    last_fit_i: int | None = None

    for i in range(n_dates):
        # Train must use dates < i
        if i < min_train_periods:
            continue

        if last_fit_i is None or (i - last_fit_i) >= retrain_every:
            train_end = i * n_tickers  # exclusive
            train_mask = ok_flat[:train_end]
            if train_mask.sum() < 1000:
                raise SystemExit(f"Too few training samples at i={i}: {int(train_mask.sum())}")

            X_train = X_flat[:train_end][train_mask]
            y_train = y_flat[:train_end][train_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            model = Ridge(alpha=float(args.ridge_alpha))
            model.fit(X_train_s, y_train)
            last_fit_i = i

        assert scaler is not None and model is not None

        X_i = X[i]
        elig_i = eligible_mask[i]
        finite_i = np.isfinite(X_i).all(axis=1)
        pred_mask = elig_i & finite_i

        # Safety for TopK selection: ensure enough eligible tickers
        if args.top_k and int(pred_mask.sum()) < int(args.top_k):
            raise SystemExit(
                f"eligible_count({dates[i].date()})={int(pred_mask.sum())} < top_k={int(args.top_k)}"
            )

        preds = np.full(n_tickers, np.nan, dtype=float)
        if pred_mask.any():
            X_pred = X_i[pred_mask]
            X_pred_s = scaler.transform(X_pred)
            preds[pred_mask] = model.predict(X_pred_s)

        # Optional sector neutralization (per-date) on predicted scores
        if args.sector_neutralize_scores and sector_members is not None and pred_mask.any():
            tmp = preds.reshape(1, -1)
            tmp_mask = pred_mask.reshape(1, -1)
            _sector_neutralize_rows_inplace(
                tmp,
                sector_members=sector_members,
                eligible_mask=tmp_mask,
                min_count=int(args.sector_min_count),
            )
            preds = tmp.reshape(-1)

        # Gate ineligible/missing as very low finite values (per-row)
        row_min = np.nanmin(preds[pred_mask]) if pred_mask.any() else 0.0
        floor = row_min - float(args.ineligible_penalty)
        preds = np.where(np.isfinite(preds), preds, floor)
        pred_scores[i] = preds

        if (i + 1) % 12 == 0 or i == n_dates - 1:
            print(f"Predicted: {i+1}/{n_dates}  date={dates[i].date()}  train_end={dates[i-1].date()}")

    # Keep only dates where predictions exist (avoid NaN rows)
    keep_rows = np.isfinite(pred_scores).all(axis=1)
    pred_dates = dates[keep_rows]
    pred_scores = pred_scores[keep_rows]
    if len(pred_dates) == 0:
        raise SystemExit("No prediction rows produced (check --min_train_periods)")

    # Simple OOS diagnostics: per-date Spearman IC on eligible tickers
    ic_values_raw: list[float] = []
    ic_values_label: list[float] = []
    ic_by_date: list[dict[str, object]] = []
    for row_idx, d in enumerate(pred_dates):
        # Map back to original index
        i = int(dates.get_loc(d))
        if i >= n_dates - label_horizon:
            continue

        elig_i = eligible_mask[i]
        mask_raw = elig_i & np.isfinite(pred_scores[row_idx]) & np.isfinite(y_raw_arr[i])
        mask_label = elig_i & np.isfinite(pred_scores[row_idx]) & np.isfinite(y_arr[i])
        if int(mask_raw.sum()) < 10:
            continue

        s_raw = pd.Series(pred_scores[row_idx][mask_raw]).rank(method="average").to_numpy(dtype=float)
        r_raw = pd.Series(y_raw_arr[i][mask_raw]).rank(method="average").to_numpy(dtype=float)
        ic_raw = None
        if np.std(s_raw) >= 1e-12 and np.std(r_raw) >= 1e-12:
            v = float(np.corrcoef(s_raw, r_raw)[0, 1])
            if np.isfinite(v):
                ic_raw = v
                ic_values_raw.append(v)

        ic_label = None
        if int(mask_label.sum()) >= 10:
            s_lab = pd.Series(pred_scores[row_idx][mask_label]).rank(method="average").to_numpy(dtype=float)
            r_lab = pd.Series(y_arr[i][mask_label]).rank(method="average").to_numpy(dtype=float)
            if np.std(s_lab) >= 1e-12 and np.std(r_lab) >= 1e-12:
                v = float(np.corrcoef(s_lab, r_lab)[0, 1])
                if np.isfinite(v):
                    ic_label = v
                    ic_values_label.append(v)

        ic_by_date.append(
            {
                args.date_col: str(pd.Timestamp(d).date()),
                "ic_spearman_raw": ic_raw,
                "ic_spearman_label": ic_label,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame(pred_scores, index=pred_dates, columns=tickers)
    scores_out = scores_df.reset_index().rename(columns={"index": args.date_col})
    scores_out.to_csv(out_dir / "scores.csv", index=False)

    pd.DataFrame(ic_by_date).to_csv(out_dir / "ic_by_date.csv", index=False)

    summary = {
        "manifest_csv": args.manifest_csv,
        "prices_csv": args.prices_csv,
        "market_cap_csv": args.market_cap_csv,
        "eligibility_csv": args.eligibility_csv,
        "insider_events_csv": args.insider_events_csv,
        "insider_ticker_col": args.insider_ticker_col,
        "ticker_to_sector_csv": args.ticker_to_sector_csv,
        "out_dir": str(out_dir),
        "feature_names": feature_names,
        "insider_feature_names": [n for n, _ in insider_feature_frames],
        "min_train_periods": int(args.min_train_periods),
        "retrain_every": int(args.retrain_every),
        "ridge_alpha": float(args.ridge_alpha),
        "winsorize_features": bool(args.winsorize_features),
        "winsorize_q_low": float(args.winsorize_q_low),
        "winsorize_q_high": float(args.winsorize_q_high),
        "cs_zscore_features": bool(args.cs_zscore_features),
        "label_horizon": int(args.label_horizon),
        "label_transform": str(args.label_transform),
        "sector_neutralize_labels": bool(args.sector_neutralize_labels),
        "sector_neutralize_scores": bool(args.sector_neutralize_scores),
        "sector_min_count": int(args.sector_min_count),
        "top_k_safety": int(args.top_k),
        "tickers": int(len(tickers)),
        "dates_total": int(len(dates)),
        "pred_dates": int(len(pred_dates)),
        "ic_spearman_mean_raw": float(np.mean(ic_values_raw)) if ic_values_raw else None,
        "ic_spearman_std_raw": float(np.std(ic_values_raw)) if ic_values_raw else None,
        "ic_spearman_n_raw": int(len(ic_values_raw)),
        "ic_spearman_mean_label": float(np.mean(ic_values_label)) if ic_values_label else None,
        "ic_spearman_std_label": float(np.std(ic_values_label)) if ic_values_label else None,
        "ic_spearman_n_label": int(len(ic_values_label)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote: {out_dir / 'scores.csv'}")
    print(f"Wrote: {out_dir / 'ic_by_date.csv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    if ic_values_raw:
        print(f"Mean Spearman IC (raw): {np.mean(ic_values_raw):.4f} (n={len(ic_values_raw)})")
    if ic_values_label:
        print(f"Mean Spearman IC (label): {np.mean(ic_values_label):.4f} (n={len(ic_values_label)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
