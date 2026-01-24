# Task 10.2.10 — Fund Feature Ablation Report

## Ablation Results

| Mode | IC Mean | N Features | Delta (all) | Delta (sec_covered) | Delta (sec_missing) | Notes |
|------|---------|------------|-------------|---------------------|---------------------|-------|
| fund_full | 0.0112 | 21 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |
| fund_zeroed | 0.0095 | 21 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |
| fund_shuffled | 0.0042 | 21 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |
| tech_only | 0.0095 | 7 | N/A | N/A | N/A | ALPHA_XGB_FALLBACK:sklearn_hist_gb:No module named |

## Verdict

- **FUND_FULL IC (0.0112) > TECH_ONLY IC (0.0095)**: Fundamentals add predictive value.
- **FUND_SHUFFLED IC (0.0042) < FUND_FULL IC (0.0112)**: Shuffling degrades IC, confirming signal.
