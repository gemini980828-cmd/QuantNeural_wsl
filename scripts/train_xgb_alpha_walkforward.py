"""
Walk-forward XGBoost alpha model (deterministic, PIT-safe).

Builds a backtest-ready scores.csv for the existing universe gate.
This script is shadow-only: it does NOT change any execution layer logic.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _read_dates_from_csv(path: str, *, date_col: str) -> pd.DatetimeIndex:
    df = pd.read_csv(path, usecols=[date_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    dates = pd.DatetimeIndex(df[date_col].unique()).sort_values()
    if len(dates) == 0:
        raise ValueError(f"No valid dates found in {path}")
    return dates


def _derive_score_dates_from_prices(prices: pd.DataFrame, *, freq: str) -> pd.DatetimeIndex:
    if freq not in ("M", "Q"):
        raise ValueError(f"score_frequency must be 'M' or 'Q', got '{freq}'")
    rule = "M" if freq == "M" else "Q"
    return prices.resample(rule).last().index


def _load_manifest_companyfacts_paths(manifest_csv: str) -> dict[str, str]:
    df = pd.read_csv(manifest_csv)
    required = ["ticker", "companyfacts_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest_csv missing columns {missing}: {manifest_csv}")

    df = df.copy()
    df["ticker"] = df["ticker"].map(_normalize_ticker)

    if "cik_status" in df.columns and "companyfacts_status" in df.columns:
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


def _build_xgb_regressor(seed: int) -> tuple[object, str, dict, list[str]]:
    warnings: list[str] = []
    xgb_params = {
        "random_state": seed,
        "n_jobs": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "verbosity": 0,
    }
    try:
        from xgboost import XGBRegressor  # type: ignore

        return XGBRegressor(**xgb_params), "xgboost", xgb_params, warnings
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        warnings.append("XGB_FALLBACK:sklearn_hist_gb")
        hist_params = {
            "loss": "squared_error",
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 300,
            "random_state": seed,
            "early_stopping": False,
        }
        return HistGradientBoostingRegressor(**hist_params), "sklearn_hist_gb", hist_params, warnings


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    val = float(np.corrcoef(rx, ry)[0, 1])
    return val if np.isfinite(val) else None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward XGB alpha (price + optional fundamentals).")
    p.add_argument("--prices-csv-path", default="data/backtest_universe_sec_mktcap/prices.csv")
    p.add_argument("--baseline-scores-csv-path", default="data/backtest_universe_sec_mktcap/scores.csv")
    p.add_argument("--eligibility-csv-path", default="data/backtest_universe_sec_mktcap/eligibility.csv")
    p.add_argument("--market-cap-csv-path", default="data/backtest_universe_sec_mktcap/market_cap.csv")
    p.add_argument("--manifest-csv", default="", help="Optional manifest CSV with companyfacts paths")
    p.add_argument("--date-col", default="date")
    p.add_argument("--score-calendar-csv-path", default="")
    p.add_argument("--score-frequency", choices=["M", "Q"], default="M")
    p.add_argument("--rebalance", choices=["M", "Q"], default="Q")
    p.add_argument("--label-horizon-months", type=int, default=0)
    p.add_argument("--out-dir", default="data/backtest_universe_sec_mktcap_xgb")

    p.add_argument("--min-train-periods", type=int, default=24)
    p.add_argument("--retrain-every", type=int, default=1)
    p.add_argument("--min-train-samples", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=400)
    p.add_argument("--ineligible-penalty", type=float, default=1e6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-ic-by-date", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_date_col = "date"
    np.random.seed(args.seed)

    prices_path = Path(args.prices_csv_path)
    if not prices_path.exists():
        raise SystemExit(f"Missing prices_csv_path: {prices_path}")

    baseline_scores_path = Path(args.baseline_scores_csv_path) if args.baseline_scores_csv_path else None
    eligibility_path = Path(args.eligibility_csv_path) if args.eligibility_csv_path else None
    market_cap_path = Path(args.market_cap_csv_path) if args.market_cap_csv_path else None
    if baseline_scores_path and not baseline_scores_path.exists():
        raise SystemExit(f"Missing baseline_scores_csv_path: {baseline_scores_path}")
    if eligibility_path and not eligibility_path.exists():
        raise SystemExit(f"Missing eligibility_csv_path: {eligibility_path}")
    if market_cap_path and not market_cap_path.exists():
        raise SystemExit(f"Missing market_cap_csv_path: {market_cap_path}")

    prices = _read_wide_panel_csv(str(prices_path), date_col=args.date_col)

    # Score dates
    score_dates_source = "prices_resample"
    if args.score_calendar_csv_path:
        score_dates = _read_dates_from_csv(args.score_calendar_csv_path, date_col=args.date_col)
        score_dates_source = "score_calendar_csv"
    elif baseline_scores_path and baseline_scores_path.exists():
        score_dates = _read_dates_from_csv(str(baseline_scores_path), date_col=args.date_col)
        score_dates_source = "baseline_scores_csv"
    elif eligibility_path and eligibility_path.exists():
        eligibility_tmp = _read_wide_panel_csv(str(eligibility_path), date_col=args.date_col)
        score_dates = eligibility_tmp.index
        score_dates_source = "eligibility_csv"
    else:
        score_dates = _derive_score_dates_from_prices(prices, freq=args.score_frequency)

    score_dates = score_dates.sort_values()
    score_dates = score_dates[(score_dates >= prices.index.min()) & (score_dates <= prices.index.max())]
    if len(score_dates) == 0:
        raise SystemExit("No score dates remain after filtering to price range.")

    # Universe tickers
    tickers = sorted(prices.columns)
    if eligibility_path and eligibility_path.exists():
        eligibility_df = _read_wide_panel_csv(str(eligibility_path), date_col=args.date_col)
        tickers = sorted(set(tickers) & set(eligibility_df.columns))
    elif baseline_scores_path and baseline_scores_path.exists():
        base_cols = [c for c in pd.read_csv(baseline_scores_path, nrows=1).columns if c != args.date_col]
        base_cols = [_normalize_ticker(c) for c in base_cols]
        tickers = sorted(set(tickers) & set(base_cols))

    if len(tickers) < 2:
        raise SystemExit(f"Need at least 2 tickers after intersection, got {len(tickers)}")

    prices = prices[tickers].astype(float)

    # Align to score dates (PIT-safe)
    prices_at = prices.reindex(score_dates, method="ffill")
    if prices_at.isna().any().any():
        raise SystemExit("prices_at has NaN after ffill alignment to score dates")

    # Eligibility mask
    if eligibility_path and eligibility_path.exists():
        eligibility_df = _read_wide_panel_csv(str(eligibility_path), date_col=args.date_col)
        eligibility_df = eligibility_df.reindex(score_dates, method="ffill").fillna(0.0)
        eligibility_df = eligibility_df[tickers].astype(float)
        eligible_mask = eligibility_df.to_numpy(dtype=bool)
    else:
        eligible_mask = np.ones((len(score_dates), len(tickers)), dtype=bool)

    # Market cap (optional)
    log_mktcap_df = None
    if market_cap_path and market_cap_path.exists():
        market_cap = _read_wide_panel_csv(str(market_cap_path), date_col=args.date_col)
        market_cap = market_cap.reindex(score_dates, method="ffill")
        market_cap = market_cap[tickers].astype(float)
        eps = 1e-18
        log_mktcap_df = np.log(np.maximum(market_cap.to_numpy(dtype=float), eps))

    # Price-based features (monthly)
    ret_1m = prices_at.pct_change(1)
    mom_3m = prices_at / prices_at.shift(3) - 1.0
    mom_6m = prices_at / prices_at.shift(6) - 1.0
    mom_12m = prices_at / prices_at.shift(12) - 1.0
    vol_3m = ret_1m.rolling(window=3, min_periods=3).std()

    feature_frames: list[tuple[str, pd.DataFrame]] = [
        ("mom_1m", ret_1m),
        ("mom_3m", mom_3m),
        ("mom_6m", mom_6m),
        ("mom_12m", mom_12m),
        ("vol_3m", vol_3m),
    ]
    if log_mktcap_df is not None:
        feature_frames.append(("log_mktcap", pd.DataFrame(log_mktcap_df, index=score_dates, columns=tickers)))

    fundamentals_used = False
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        if not manifest_path.exists():
            raise SystemExit(f"Missing manifest_csv: {manifest_path}")

        ticker_to_companyfacts = _load_manifest_companyfacts_paths(str(manifest_path))
        n_dates = len(score_dates)
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

            assets[:, j] = _pit_latest_snapshot(a, score_dates)
            liabilities[:, j] = _pit_latest_snapshot(l, score_dates)
            equity[:, j] = _pit_latest_snapshot(e, score_dates)
            cash[:, j] = _pit_latest_snapshot(c, score_dates)

        eps = 1e-18
        mc_arr = None
        if log_mktcap_df is not None:
            mc_arr = np.exp(log_mktcap_df)

        if mc_arr is not None:
            book_to_mkt = equity / np.maximum(mc_arr, eps)
        else:
            book_to_mkt = np.full_like(equity, np.nan)

        leverage = liabilities / np.maximum(assets, eps)
        cash_to_assets = cash / np.maximum(assets, eps)
        asset_growth_12m = assets / np.maximum(np.roll(assets, shift=12, axis=0), eps) - 1.0
        asset_growth_12m[:12, :] = np.nan

        feature_frames.extend(
            [
                ("book_to_mkt", pd.DataFrame(book_to_mkt, index=score_dates, columns=tickers)),
                ("leverage", pd.DataFrame(leverage, index=score_dates, columns=tickers)),
                ("cash_to_assets", pd.DataFrame(cash_to_assets, index=score_dates, columns=tickers)),
                ("asset_growth_12m", pd.DataFrame(asset_growth_12m, index=score_dates, columns=tickers)),
            ]
        )
        fundamentals_used = True

    feature_names = [n for n, _ in feature_frames]
    X = np.stack([df.to_numpy(dtype=float) for _, df in feature_frames], axis=2)

    label_horizon = int(args.label_horizon_months)
    if label_horizon <= 0:
        label_horizon = 3 if args.rebalance == "Q" else 1

    y_raw = (prices_at.shift(-label_horizon) / prices_at) - 1.0
    y_arr = y_raw.to_numpy(dtype=float)

    X_finite = np.isfinite(X).all(axis=2)
    y_finite = np.isfinite(y_arr)
    sample_ok = eligible_mask & X_finite & y_finite

    n_dates, n_tickers, n_features = X.shape
    X_flat = X.reshape(n_dates * n_tickers, n_features)
    y_flat = y_arr.reshape(n_dates * n_tickers)
    ok_flat = sample_ok.reshape(n_dates * n_tickers)

    min_train_periods = int(args.min_train_periods)
    retrain_every = int(args.retrain_every)
    min_train_samples = int(args.min_train_samples)
    if min_train_periods < 6:
        raise SystemExit("--min-train-periods must be >= 6")
    if retrain_every < 1:
        raise SystemExit("--retrain-every must be >= 1")

    pred_scores = np.full((n_dates, n_tickers), np.nan, dtype=float)
    eligible_counts: list[int] = []

    model = None
    backend = "xgboost"
    model_params: dict = {}
    warnings: list[str] = []
    last_fit_i: int | None = None

    for i in range(n_dates):
        if i < min_train_periods:
            continue

        if last_fit_i is None or (i - last_fit_i) >= retrain_every:
            train_end = i * n_tickers
            train_mask = ok_flat[:train_end]
            if int(train_mask.sum()) < min_train_samples:
                raise SystemExit(f"Too few training samples at i={i}: {int(train_mask.sum())}")

            X_train = X_flat[:train_end][train_mask]
            y_train = y_flat[:train_end][train_mask]

            model, backend, model_params, w = _build_xgb_regressor(seed=args.seed)
            warnings.extend(w)
            model.fit(X_train, y_train)
            last_fit_i = i

        if model is None:
            raise SystemExit("Model was not initialized")

        X_i = X[i]
        elig_i = eligible_mask[i]
        finite_i = np.isfinite(X_i).all(axis=1)
        pred_mask = elig_i & finite_i

        if args.top_k and int(pred_mask.sum()) < int(args.top_k):
            raise SystemExit(
                f"eligible_count({score_dates[i].date()})={int(pred_mask.sum())} < top_k={int(args.top_k)}"
            )

        eligible_counts.append(int(pred_mask.sum()))
        preds = np.full(n_tickers, np.nan, dtype=float)
        if pred_mask.any():
            preds[pred_mask] = model.predict(X_i[pred_mask])

        row_min = float(np.nanmin(preds[pred_mask])) if pred_mask.any() else 0.0
        if not np.isfinite(row_min):
            row_min = 0.0
        floor = row_min - float(args.ineligible_penalty)
        preds[~pred_mask] = floor
        preds = np.where(np.isfinite(preds), preds, floor)
        pred_scores[i] = preds

    keep_rows = np.isfinite(pred_scores).all(axis=1)
    pred_dates = score_dates[keep_rows]
    pred_scores = pred_scores[keep_rows]
    if len(pred_dates) == 0:
        raise SystemExit("No prediction rows produced (check --min-train-periods)")

    scores_df = pd.DataFrame(pred_scores, index=pred_dates, columns=tickers)
    scores_out = scores_df.reset_index().rename(columns={"index": output_date_col})
    if not np.isfinite(scores_df.to_numpy()).all():
        raise SystemExit("scores.csv contains non-finite values")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_out.to_csv(out_dir / "scores.csv", index=False, float_format="%.10f", lineterminator="\n")

    ic_rows: list[dict[str, object]] = []
    ic_values: list[float] = []
    if args.write_ic_by_date:
        for row_idx, d in enumerate(pred_dates):
            i = int(score_dates.get_loc(d))
            if i >= n_dates - label_horizon:
                continue
            elig_i = eligible_mask[i]
            mask = elig_i & np.isfinite(pred_scores[row_idx]) & np.isfinite(y_arr[i])
            if int(mask.sum()) < 10:
                continue
            ic = _spearman_ic(pred_scores[row_idx][mask], y_arr[i][mask])
            if ic is not None:
                ic_values.append(ic)
            ic_rows.append(
                {
                    output_date_col: str(pd.Timestamp(d).date()),
                    "ic_spearman": ic,
                    "eligible_count": int(mask.sum()),
                }
            )
        pd.DataFrame(ic_rows).to_csv(out_dir / "ic_by_date.csv", index=False)

    eligible_stats = {
        "eligible_count_min": int(min(eligible_counts)) if eligible_counts else None,
        "eligible_count_mean": float(np.mean(eligible_counts)) if eligible_counts else None,
        "eligible_count_max": int(max(eligible_counts)) if eligible_counts else None,
    }

    summary = {
        "schema_version": "10.0.0",
        "config": {
            "prices_csv_path": str(prices_path),
            "baseline_scores_csv_path": str(baseline_scores_path) if baseline_scores_path else None,
            "eligibility_csv_path": str(eligibility_path) if eligibility_path else None,
            "market_cap_csv_path": str(market_cap_path) if market_cap_path else None,
            "score_calendar_csv_path": args.score_calendar_csv_path or None,
            "score_frequency": args.score_frequency,
            "score_dates_source": score_dates_source,
            "rebalance": args.rebalance,
            "label_horizon_months": int(label_horizon),
            "min_train_periods": int(args.min_train_periods),
            "retrain_every": int(args.retrain_every),
            "min_train_samples": int(args.min_train_samples),
            "top_k": int(args.top_k),
            "ineligible_penalty": float(args.ineligible_penalty),
            "seed": int(args.seed),
            "backend": backend,
            "model_params": model_params,
        },
        "features": feature_names,
        "fundamentals_used": fundamentals_used,
        "train_policy": {"type": "expanding", "retrain_every": int(args.retrain_every)},
        "eligibility_stats": eligible_stats,
        "universe": {"tickers": int(len(tickers)), "dates_total": int(len(score_dates)), "pred_dates": int(len(pred_dates))},
        "ic_spearman_mean": float(np.mean(ic_values)) if ic_values else None,
        "ic_spearman_n": int(len(ic_values)),
        "warnings": warnings,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote: {out_dir / 'scores.csv'}")
    if args.write_ic_by_date:
        print(f"Wrote: {out_dir / 'ic_by_date.csv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
