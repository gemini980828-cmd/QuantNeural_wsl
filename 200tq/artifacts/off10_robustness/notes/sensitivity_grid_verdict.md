# Sensitivity Grid Analysis

**Generated**: 2026-01-17 19:18
**Baseline**: E00_V0_Base_OFF10_CASH
**Candidate**: E03_Ensemble_SGOV

## Grid Parameters

- **Cost**: [10, 20, 30, 50] bps
- **Slippage**: [0, 5, 10, 20] bps
- **Tax**: 22% (fixed)

## Verdict: **ROBUST**

E03 maintains advantage across ALL cost/slippage scenarios (16/16)

## ΔCAGR Summary (E03 vs E00)

| Cost (bps) | Slip (bps) | Total | ΔCAGR | Status |
|----------:|----------:|------:|------:|:------:|
| 10 | 0 | 10 | +2.07%p | ✅ |
| 10 | 5 | 15 | +2.07%p | ✅ |
| 10 | 10 | 20 | +2.06%p | ✅ |
| 20 | 0 | 20 | +2.06%p | ✅ |
| 20 | 5 | 25 | +2.06%p | ✅ |
| 10 | 20 | 30 | +2.06%p | ✅ |
| 20 | 10 | 30 | +2.06%p | ✅ |
| 30 | 0 | 30 | +2.06%p | ✅ |
| 30 | 5 | 35 | +2.06%p | ✅ |
| 20 | 20 | 40 | +2.05%p | ✅ |
| 30 | 10 | 40 | +2.05%p | ✅ |
| 30 | 20 | 50 | +2.05%p | ✅ |
| 50 | 0 | 50 | +2.05%p | ✅ |
| 50 | 5 | 55 | +2.05%p | ✅ |
| 50 | 10 | 60 | +2.04%p | ✅ |
| 50 | 20 | 70 | +2.04%p | ✅ |

## Critical Finding

> **No Breakeven Point Found**: E03 maintains advantage across all tested scenarios.
> Strategy is robust to cost/slippage stress up to 50+20=70 bps total.
