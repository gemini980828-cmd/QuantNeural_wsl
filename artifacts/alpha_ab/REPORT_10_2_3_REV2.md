# REPORT: Task 10.2.3 Rev2 - Tech vs Fund A/B Backtest

## Summary

**Verdict: `NO_CLEAR_IMPROVEMENT`**

Coverage fix (10.2.2.1) successfully implemented, but adding SEC fundamentals does **not** improve IC over the tech-only baseline in this 2-date sample.

---

## Configuration

| Parameter  | Value         |
| ---------- | ------------- |
| as_of_date | 2024-12-31    |
| train_end  | 2023-12-31    |
| val_end    | 2024-06-30    |
| rebalance  | Q (Quarterly) |
| target_col | fwd_ret_21d   |
| top_k      | 400           |
| seed       | 42            |

---

## Dataset Build Summary

| Dataset       | Tickers | Rows      | Fundamental Features | Missing Indicators |
| ------------- | ------- | --------- | -------------------- | ------------------ |
| Tech-only     | 2,457   | 4,396,822 | 0                    | 0                  |
| Fund-included | 2,457   | 4,396,822 | 9                    | 9                  |

**Coverage Fix Applied (10.2.2.1):**

- ✅ FFill-only for SEC columns (no median fill)
- ✅ Missing indicator columns (`<col>_is_missing`)
- ✅ Eligibility gating ignores fundamental NaNs

---

## IC Comparison

### IC by Date

| Date       | Tech IC | Fund IC | Δ (Fund - Tech) |
| ---------- | ------- | ------- | --------------- |
| 2024-09-30 | 0.1515  | 0.1255  | **-0.0260**     |
| 2024-10-01 | 0.1425  | 0.1077  | **-0.0348**     |

### IC Mean

| Metric               | Tech-only  | Fund-included | Δ           |
| -------------------- | ---------- | ------------- | ----------- |
| **IC Mean**          | **0.1470** | 0.1166        | **-0.0304** |
| N dates              | 2          | 2             | -           |
| Eligible count (avg) | 517        | 517           | 0           |

---

## Feature Comparison

### Tech-only Features (7)

```
atr_14d_norm, bbands_20d, mom_21d, mom_5d, mom_63d, rsi_14d, vol_20d
```

### Fund-included Features (25)

```
assets, assets_is_missing, atr_14d_norm, bbands_20d, book_to_assets,
book_to_assets_is_missing, cash, cash_is_missing, cash_to_assets,
cash_to_assets_is_missing, equity, equity_is_missing, leverage,
leverage_is_missing, liabilities, liabilities_is_missing, mktcap,
mktcap_is_missing, mom_21d, mom_5d, mom_63d, rsi_14d, shares_out,
shares_out_is_missing, vol_20d
```

---

## Analysis

### Why Fund IC is Lower

1. **Data Sparsity**: SEC fundamental coverage is ~43%, meaning most tickers have `<col>_is_missing=1.0` and `<col>=0.0` (imputed). This creates a "data availability bias" rather than true fundamental signal.

2. **Feature Dilution**: Adding 18 features (9 fundamentals + 9 missing indicators) to a 7-feature technical model increases dimensionality without proportional signal, reducing effective signal-to-noise ratio.

3. **Small Sample**: Only 2 rebalance dates scored, insufficient to draw robust conclusions.

4. **Missing Indicators Dominate**: The `*_is_missing` indicators may be encoding data quality/coverage rather than economic fundamentals.

### Judgment Criteria Check

| Criterion           | Passed? | Notes                               |
| ------------------- | ------- | ----------------------------------- |
| IC mean improvement | ❌ NO   | Fund IC (0.1166) < Tech IC (0.1470) |
| XGB delta CAGR/vol  | N/A     | Not measured (need full backtest)   |

**Final Judgment: `NO_CLEAR_IMPROVEMENT`**

---

## Recommendations

1. **Increase coverage before adding fundamentals**: Target >70% fundamental coverage before integrating into alpha model.

2. **Feature importance analysis**: Run SHAP/permutation importance to identify if any fundamental features provide marginal value.

3. **Longer backtest window**: Extend to 2012-2024 for more rebalance dates.

4. **Alternative imputation strategies**:

   - Cross-sectional percentile rank (within-date) instead of 0.0 fill
   - Industry-median imputation (requires SIC mapping)

5. **Consider simpler fundamental features first**: Start with `mktcap` and `leverage` only before adding all 9 features.

---

## Output Artifacts

| File         | Path                                            |
| ------------ | ----------------------------------------------- |
| Tech scores  | `data/alpha/xgb_model_q400_tech/scores.csv`     |
| Tech IC      | `data/alpha/xgb_model_q400_tech/ic_by_date.csv` |
| Tech summary | `data/alpha/xgb_model_q400_tech/summary.json`   |
| Fund scores  | `data/alpha/xgb_model_q400_fund/scores.csv`     |
| Fund IC      | `data/alpha/xgb_model_q400_fund/ic_by_date.csv` |
| Fund summary | `data/alpha/xgb_model_q400_fund/summary.json`   |

---

_Generated: 2026-01-15T21:XX (Task 10.2.3 Rev2)_
