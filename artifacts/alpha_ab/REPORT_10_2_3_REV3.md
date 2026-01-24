# Task 10.2.3 Rev3 — Tech vs Fund Evaluation Report

**Date**: 2026-01-16  
**Target Column**: `fwd_ret_63d` (Quarterly)  
**SEC Canonical Layer**: V2.3

---

## A) Coverage Table (FUND Dataset)

| Column                | Non-NaN Ratio | Is-Missing Ratio |
| --------------------- | ------------- | ---------------- |
| `total_assets`        | 42.6%         | 57.4%            |
| `total_liabilities`   | 40.1%         | 59.9%            |
| `stockholders_equity` | 41.1%         | 58.9%            |
| `revenues`            | 30.9%         | 69.1%            |
| `net_income`          | 41.3%         | 58.7%            |
| `operating_cash_flow` | 40.2%         | 59.8%            |
| `shares_outstanding`  | 43.8%         | 56.2%            |

**Overall `any_sec_present_ratio`**: 44.3%

> [!WARNING]
> Coverage is ~44%, lower than expected 74%. This is because manual dataset build
> uses all tickers (not just manifested ones), diluting coverage percentage.

---

## B) Performance Table (TECH vs FUND)

| Model    | Target      | n_dates | IC mean    | Delta CAGR/Vol | Delta CAGR/Vol (Inverse) |
| -------- | ----------- | ------- | ---------- | -------------- | ------------------------ |
| **TECH** | fwd_ret_63d | 32      | **0.0042** | **0.1317**     | 0.1986                   |
| **FUND** | fwd_ret_63d | 32      | **0.0384** | **0.1165**     | 0.1910                   |

### Key Observations

1. **IC mean**: FUND (0.0384) >> TECH (0.0042) — **9x improvement**
2. **Delta CAGR/Vol**: TECH (0.1317) > FUND (0.1165) — TECH wins by 0.015
3. **Inversion diagnostic**: Both improve with inversion (negative alpha direction)

---

## C) Verdict

### FUND_IMPROVES? **MIXED / NO_CLEAR_IMPROVEMENT**

- ✅ **IC dramatically improves** with Fund features (0.0042 → 0.0384)
- ❌ **Backtest performance slightly worse** (0.1317 → 0.1165)
- ⚠️ **Both directions show negative alpha** (inversion helps)

### Root Cause Analysis

1. **Higher IC doesn't translate to better returns** — possible factor decay
2. **Coverage still ~44%** — many rows have NaN fundamentals
3. **top_k mismatch** — TECH used top_k=400, FUND used top_k=150 (due to eligible_count constraints)

---

## D) File Tree

```
artifacts/alpha_ab/
├── xgb_q400_tech_eval_rev3/
│   ├── diagnostics.json
│   ├── xgb/
│   │   └── delta_summary.json
│   └── xgb_inverse/
│       └── delta_summary.json
└── xgb_q400_fund_eval_rev3/
    ├── diagnostics.json
    ├── xgb/
    │   └── delta_summary.json
    └── xgb_inverse/
        └── delta_summary.json

data/alpha/
├── xgb_model_q400_tech_rev3/
│   ├── scores.csv
│   ├── ic_by_date.csv
│   └── summary.json
└── xgb_model_q400_fund_rev3/
    ├── scores.csv
    ├── ic_by_date.csv
    └── summary.json
```

---

## E) Command Exit Codes

| Step | Command                     | Exit Code |
| ---- | --------------------------- | --------- |
| A1   | build_alpha_dataset (TECH)  | ✅ 0      |
| A2   | build_alpha_dataset (FUND)  | ✅ 0      |
| B1   | train_xgb (TECH, top_k=400) | ✅ 0      |
| B2   | train_xgb (FUND, top_k=150) | ✅ 0      |
| C1   | run_ab_backtest (TECH)      | ✅ 0      |
| C2   | run_ab_backtest (FUND)      | ✅ 0      |
