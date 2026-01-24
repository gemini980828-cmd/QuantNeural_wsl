# RunReport: Task 10.2.3 Rev3 — SEC Fundamental Evaluation

**Date**: 2026-01-16
**AS_OF_DATE**: 2024-12-31
**Task**: Tech vs Fund Alpha Evaluation with V2.3 SEC Canonical Layer
**Status**: 🟢 Completed

---

## Summary

Task 10.2.3 Rev3 evaluated whether adding SEC fundamental features (V2.3 canonical layer) improves XGB alpha model performance vs tech-only features.

### Key Results

| Metric             | TECH       | FUND       | Delta     |
| ------------------ | ---------- | ---------- | --------- |
| **IC mean**        | 0.0042     | **0.0384** | +9x ✅    |
| **Delta CAGR/Vol** | **0.1317** | 0.1165     | -0.015 ❌ |
| **top_k used**     | 400        | 150        | mismatch  |

### Verdict: **NO_CLEAR_IMPROVEMENT**

- ✅ IC dramatically improved (9x)
- ❌ Backtest performance slightly declined
- ⚠️ top_k mismatch due to eligible_count constraints

---

## Coverage (V2.3 SEC Canonical Layer)

| Column              | Non-NaN Ratio |
| ------------------- | ------------- |
| total_assets        | 42.6%         |
| total_liabilities   | 40.1%         |
| stockholders_equity | 41.1%         |
| revenues            | 30.9%         |
| **any_sec_present** | **44.3%**     |

---

## Commands Executed

| Step | Command                          | Exit |
| ---- | -------------------------------- | ---- |
| A1   | build_alpha_dataset (TECH)       | ✅ 0 |
| A2   | build_alpha_dataset (FUND, V2.3) | ✅ 0 |
| B1   | train_xgb (TECH, top_k=400)      | ✅ 0 |
| B2   | train_xgb (FUND, top_k=150)      | ✅ 0 |
| C1   | run_ab_backtest (TECH)           | ✅ 0 |
| C2   | run_ab_backtest (FUND)           | ✅ 0 |

---

## Artifacts

- `artifacts/alpha_ab/REPORT_10_2_3_REV3.md`
- `artifacts/alpha_ab/xgb_q400_tech_eval_rev3/`
- `artifacts/alpha_ab/xgb_q400_fund_eval_rev3/`
- `data/alpha/xgb_model_q400_tech_rev3/`
- `data/alpha/xgb_model_q400_fund_rev3/`

---

## Next Steps

1. Investigate top_k mismatch — either normalize or use same top_k
2. Increase SEC coverage (currently 44%) before re-evaluation
3. Consider feature engineering (NaN→0 + is_missing indicator)
