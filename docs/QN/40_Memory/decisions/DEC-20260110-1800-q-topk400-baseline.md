# DEC-20260110-1800: Q+TopK400 Production Candidate

## Decision

**Locked production baseline**: `rebalance=Q, method=topk, top_k=400, cost_bps=10, slippage_bps=5`

## Context

Phase 9.0.x experiments (24+ backtest runs) evaluated multiple configurations on 2,753-ticker Stooq universe (2011-2024).

## Evidence

| Config | Sharpe | MaxDD | Turnover | Cost |
|--------|--------|-------|----------|------|
| M_topk50 (baseline) | 0.54 | -57.7% | 93.2 | 34.1% |
| Q_topk300 | 0.670 | -45.2% | 43.7 | 17.0% |
| **Q_topk400** | **0.670** | **-44.0%** | **41.8** | **15.8%** |
| C1_M_rank (dense) | 0.78 | -37.0% | 26.3 | 9.4% |

## Rationale

1. Q400 matches Q300 Sharpe across all cost scenarios
2. Q400 has 1.5pp better MaxDD than Q300
3. Q400 has ~2pp lower cost due to lower turnover
4. Dense rank is untradable (~2,700 trades per rebalance)
5. Monthly has 2x cost of Quarterly

## Forward Rule

All future experiments use Q+topk400 as execution layer baseline. Any change requires re-sweep justification.

## Related

- [[RunReport-20260110-phase9x-tradability]]
- [[SSOT_TASKS]] § F.5.9