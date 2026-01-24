# Task 10.2.7 — Q400 Parity Evaluation Report

**Date**: 2026-01-16
**AS_OF_DATE**: 2024-12-31
**Status**: ✅ COMPLETED

---

## 1. Goal

Evaluate TECH vs FUND XGB models under **identical Q400 conditions** (top_k=400) to ensure fair comparison.

---

## 2. Coverage Analysis (FUND Dataset)

| SEC Column          | Non-NaN Ratio |
| ------------------- | ------------- |
| total_assets        | 42.6%         |
| total_liabilities   | 40.1%         |
| stockholders_equity | 41.1%         |
| revenues            | 30.9%         |
| net_income          | 41.3%         |
| operating_cash_flow | 40.2%         |
| shares_outstanding  | 43.8%         |
| **any_sec_present** | **44.3%**     |

**Manifest Alignment**:

- stooq_total_tickers: 10,502
- manifest_ok_tickers: 1,951
- overlap: 100% (after ticker normalization fix)

---

## 3. Training Configuration

| Parameter  | Value            |
| ---------- | ---------------- |
| as_of_date | 2024-12-31       |
| train_end  | 2024-12-31       |
| val_end    | 2016-12-31       |
| rebalance  | Q (Quarterly)    |
| target_col | fwd_ret_63d      |
| top_k      | **400** (PARITY) |
| seed       | 42               |

---

## 4. Final Results

| Model    | target_col  | n_dates_scored | IC Mean    | Delta CAGR/Vol (XGB) | Delta CAGR/Vol (XGB_INV) | any_sec_present |
| -------- | ----------- | -------------- | ---------- | -------------------- | ------------------------ | --------------- |
| **TECH** | fwd_ret_63d | 32             | 0.0046     | +0.1340              | +0.1986                  | -               |
| **FUND** | fwd_ret_63d | 32             | **0.0205** | **+0.1448**          | +0.1986                  | 44.3%           |

### Improvement Metrics

| Metric         | TECH → FUND       | Change         |
| -------------- | ----------------- | -------------- |
| IC Mean        | 0.0046 → 0.0205   | **+4.4x** ✅   |
| Delta CAGR/Vol | +0.1340 → +0.1448 | **+0.0108** ✅ |

---

## 5. Verdict: **FUND_IMPROVES** ✅

Based on the verdict rule:

> FUND_IMPROVES iff FUND IC_mean > TECH IC_mean AND FUND Delta CAGR/Vol > TECH Delta CAGR/Vol

| Condition                                                     | Result |
| ------------------------------------------------------------- | ------ |
| FUND IC Mean (0.0205) > TECH IC Mean (0.0046)                 | ✅ YES |
| FUND Delta CAGR/Vol (+0.1448) > TECH Delta CAGR/Vol (+0.1340) | ✅ YES |

**Both conditions satisfied → FUND_IMPROVES**

---

## 6. Key Achievements

1. **Q400 Parity Achieved**: Both models trained with identical top_k=400
2. **Section 5 Minimal Fix Applied**: Updated SEC_FUNDAMENTAL_COLS to V2.3 column names
3. **Signal Injection without Attrition**: Fundamentals don't reduce eligibility

---

## 7. Evidence Bundle

### Artifacts:

- artifacts/alpha_ab/xgb_q400_tech_eval_v2_7/
- artifacts/alpha_ab/xgb_q400_fund_eval_v2_7/
- data/alpha/xgb_model_q400_tech_v2_7/
- data/alpha/xgb_model_q400_fund_v2_7/

---

## 8. Commands Executed

| Step | Command                               | Exit Code |
| ---- | ------------------------------------- | --------- |
| A1   | build_alpha_dataset (TECH)            | ✅ 0      |
| A2   | build_alpha_dataset (FUND, RAM cache) | ✅ 0      |
| C1   | train_xgb (TECH, top_k=400)           | ✅ 0      |
| C2   | train_xgb (FUND, top_k=400)           | ✅ 0      |
| D1   | run_ab_backtest (TECH)                | ✅ 0      |
| D2   | run_ab_backtest (FUND)                | ✅ 0      |
