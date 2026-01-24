# Task 10.2.10 — Fund Feature Ablation Report

## Ablation Results

| Mode | IC Mean | N Features | Delta (all) | Delta (sec_covered) | Delta (sec_missing) | Notes |
|------|---------|------------|-------------|---------------------|---------------------|-------|
| fund_full | 0.0116 | 21 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |
| fund_zeroed | 0.0098 | 21 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |
| fund_shuffled | 0.0026 | 21 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |
| tech_only | 0.0098 | 7 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |

## Verdict

- **FUND_FULL IC (0.0116) > TECH_ONLY IC (0.0098)**: Fundamentals add predictive value.
- **FUND_SHUFFLED IC (0.0026) < FUND_FULL IC (0.0116)**: Shuffling degrades IC, confirming signal.
