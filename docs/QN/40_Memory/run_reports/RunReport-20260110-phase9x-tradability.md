# RunReport: Phase 9.x Tradability Evaluation

## Summary

| Field | Value |
|-------|-------|
| **Date** | 2026-01-10 |
| **AS_OF_DATE** | 2024-12-31 |
| **Git Commit** | `09ba781` |
| **Status** | 🟢 Completed |

## Tasks Completed

| Task | Description | Result |
|------|-------------|--------|
| 9.0.0 | Evaluation Matrix | 12 runs, Rank dense best Sharpe but untradable |
| 9.0.1 | Tradability Diagnostics | Rank: 2,747 trades/reb, TopK: 106-200 |
| 9.0.2 | TopK Expansion Sweep | K=400 best MDD, K=200 best Sharpe |
| 9.0.3 | Cost Sensitivity | Q400 stable across 3 cost scenarios |
| 9.0.4 | Decision Lock | Q+topk400 fixed as production baseline |
| 9.1.0 | Tradable Rank | rank+top_k implemented but underperforms |

## Key Metrics (Locked Baseline: Q+topk400)

| Metric | Value |
|--------|-------|
| CAGR | 13.28% |
| Sharpe | 0.670 |
| Max DD | -44.0% |
| Turnover | 41.8 |
| Total Cost | 15.8% |

## Decision

> **Production Candidate**: `rebalance=Q, method=topk, top_k=400, cost_bps=10, slippage_bps=5`

### Rationale
1. Q400 matches Q300 Sharpe while improving MaxDD by ~1.5pp
2. Lower turnover reduces cost by ~2pp vs Q300
3. Rank-based methods excluded due to ~2,700 trades/rebalance
4. Monthly excluded for production due to higher cost

## Files Changed

- `src/weights_adapter.py` — rank+top_k, sparsity-preserving max_weight
- `tests/test_weights_adapter.py` — 6 new tests
- `docs/SSOT_TASKS.md` — F.5.8 (weights_adapter), F.5.9 (Decision Lock)
- `docs/PLANS.md` — Phase 9.x tasks, Locked Baseline section

## AutoQC

- **Status**: N/A (backtest harness, no data ETL)
- **pytest**: ✅ PASS (all tests)

## Forward Rules

1. All future model evaluations use Q+topk400 baseline
2. Any change to rebalance/method/top_k requires re-sweep