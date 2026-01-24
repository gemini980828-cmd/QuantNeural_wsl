# RunReport: Phase 7.5 Large-Universe Backtest

**Date**: 2026-01-10
**Commit**: `f1fabf0`
**AS_OF_DATE**: 2024-12-31

## Summary

Built and tested large-universe backtest infrastructure:
- **10,501 Stooq files** → **2,753 tickers** (90%+ coverage filter)
- **Period**: 2011-06-30 ~ 2024-12-31 (~13.5 years, 3,398 trading days)
- **Scores**: 151 monthly momentum signals (12M pct_change)

## Backtest Results

| Metric | Value |
|--------|-------|
| CAGR | 15.48% |
| Ann. Vol | 28.49% |
| Sharpe | 0.54 |
| Max Drawdown | -57.73% |
| Total Turnover | 93.2 |
| Total Cost | 34.09% |

**Settings**: Monthly rebalance, Top 50 momentum, 10 bps costs, 5 bps slippage

## Files Changed (36 files, +7809 lines)

### New Modules
- `src/backtest_harness.py` - Daily simulation engine
- `src/weights_adapter.py` - Predictions-to-weights (softmax/rank/topk)
- `src/e2e_backtest.py` - E2E wiring
- `src/run_scores_backtest_cli.py` - CLI orchestrator
- `src/backtest_artifacts_io.py` - CSV I/O
- `src/generate_scores_from_prices.py` - Momentum score generator
- `src/temperature_scaling.py`, `conformal.py`, `shap_explainability.py` (Phase 8 utilities)

### Scripts
- `scripts/build_universe_prices.py` - Parallel Stooq loader

### Tests
- 40+ test cases covering all new modules

## AutoQC
- **pytest -q**: Exit code 0 ✅

## SSOT Tasks Completed
- 7.5.0-7.5.8 implemented and tested

## Notes
- Current backtest uses **simple 12M momentum**, NOT full ML pipeline
- Phase 5/6/7 Health Gates, Regime, ML model training NOT integrated yet
- `ffill()` used for daily price preservation (PIT-safe)