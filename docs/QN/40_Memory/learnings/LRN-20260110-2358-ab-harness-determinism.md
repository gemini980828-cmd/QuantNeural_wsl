---
id: LRN-20260110-2358-ab-harness-determinism
date: 2026-01-10
related_tasks: ["9.3.0", "9.3.1", "9.3.2", "9.4.0"]
tags: [learning, backtest, determinism, etl]
---

# A/B Backtest Harness Determinism + Sector Mapping

## Learning

When implementing A/B comparison between scoring models (momentum vs MLP), two key requirements emerged:

1. **Score Panel Alignment**: Baseline and variant must share common dates AND tickers. If the MLP outputs sector-level scores (S0-S9) while baseline has ticker-level scores (2753 tickers), A/B comparison fails due to empty intersection.

2. **Ticker-to-Sector Mapping ETL**: To broadcast sector-level predictions to ticker-level, need a deterministic SIC→Sector mapping. Key design choices:
   - `sector_name_to_id` is caller-provided (not hardcoded) to avoid coupling
   - Tie-break duplicates by smallest source (lexicographic)
   - Bad JSON files skipped with warning, not crash

## Code References

- `src/ab_backtest.py`: `run_ab_backtest_from_score_csvs()` - strict intersection + delta computation
- `src/ticker_sector_mapping.py`: `build_ticker_to_sector_csv()` - deterministic ETL

## Git Commit

`7fbcfa9d` - Phase 9.3: A/B Backtest Harness + Ticker-Sector ETL