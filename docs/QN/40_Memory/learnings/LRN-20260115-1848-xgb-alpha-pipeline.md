---
id: LRN-20260115-1848-xgb-alpha-pipeline
date: 2026-01-15
as_of_date: 2026-01-15
related_taskspec: "Task 10.1.1–10.1.3"
tags: [learning, xgb, alpha, pipeline]
---

# Learning: XGB Alpha Model Pipeline Implementation

## Summary

Implemented a complete XGB alpha model pipeline for stock ranking, consisting of:

1. **Feature Engineering** (`src/alpha_features.py`)

   - Vectorized pandas/numpy operations (no `.apply()`)
   - vol_20d, mom_5d/21d/63d, rsi_14d, bbands_20d, atr_14d_norm
   - Wilder-style EWM smoothing for RSI/ATR

2. **Dataset Builder** (`src/build_alpha_dataset.py`)

   - PIT cutoff (`--as-of-date`)
   - Float32 memory optimization
   - csv.gz output with parquet fallback
   - Fail-safe file handling (skip corrupt files)

3. **Walk-Forward Training** (`scripts/train_xgb_alpha_from_dataset.py`)

   - Train on date ≤ train_end AND date < rebalance_date
   - Deterministic XGB params: n_jobs=1, subsample=1.0
   - Ineligible penalty: score = row_min - 1e6

4. **A/B Backtest Wrapper** (`scripts/run_xgb_alpha_ab_backtest.py`)
   - Locked Q400 baseline (rebalance=Q, top_k=400)
   - Thin wrapper calling existing harness

## Key Design Decisions

| Decision          | Rationale                                     |
| ----------------- | --------------------------------------------- |
| Pure pandas/numpy | No new dependencies (ta-lib forbidden)        |
| Float32           | 50% memory reduction for large datasets       |
| Wilder smoothing  | Standard RSI/ATR formula                      |
| Wrapper pattern   | Reuse existing A/B harness, avoid duplication |
| csv.gz default    | Universal compatibility; parquet optional     |

## Gotchas Discovered

1. **CSV read loses float32**: `pd.read_csv()` returns float64; test assertions must account for this
2. **IC requires ≥10 samples**: Spearman IC computation skips small groups
3. **Empty IC file**: Must write header even when no IC rows computed

## Test Coverage

- `tests/test_build_alpha_dataset.py`: 14 tests
- `tests/test_train_xgb_alpha_from_dataset.py`: 5 tests
- `tests/test_run_xgb_alpha_ab_backtest.py`: 3 tests

All tests passing (Exit Code 0).
