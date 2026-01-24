# XGB Alpha OOS Evidence Expansion + Sign Stability Sweep

**Generated**: 2026-01-15T19:55:00+09:00  
**Task**: 10.1.4  
**Schema Version**: 10.1.3

---

## Executive Summary

Three configurations were tested to evaluate OOS signal stability and direction:

| Config | Train End  | Val End    | OOS Dates | IC Mean | XGB Δ CAGR/Vol | XGB_INV Δ CAGR/Vol | Winner  |
| ------ | ---------- | ---------- | --------- | ------- | -------------- | ------------------ | ------- |
| **A**  | 2022-12-31 | 2023-12-31 | 4         | -0.0006 | -0.2518        | **+0.0386**        | INVERSE |
| **B**  | 2023-12-31 | 2023-12-31 | 4         | +0.0059 | -0.2518        | **+0.0386**        | INVERSE |
| **C**  | 2024-12-31 | 2016-12-31 | 32        | +0.0019 | **+0.1163**    | **+0.1986**        | BOTH ✅ |

---

## Direction Verdict

**UNSTABLE / NEED MORE**

Reasoning:

- IC mean is nearly zero across all configs (range: -0.0006 to +0.0059)
- IC is NOT consistently negative — in fact, CONFIG_B and CONFIG_C have positive IC
- XGB_INVERSE beats baseline in 3/3 configs, BUT XGB (non-inverted) also beats baseline in CONFIG_C
- The "inversion benefit" is NOT due to a consistently negative signal; it appears to be random/noise

> [!WARNING]
> The IC is essentially zero (mean ≈ 0), which means the XGB technical features have **no predictive power**.
> Both XGB and XGB_INVERSE can appear to beat baseline due to **random selection differences**, not true alpha.

---

## Detailed Results by Config

### CONFIG_A: Stale Training (train_end=2022-12-31)

| Metric             | Value                    |
| ------------------ | ------------------------ |
| OOS Date Range     | 2024-03-28 to 2024-10-01 |
| OOS Dates Scored   | 4                        |
| IC Mean            | -0.0006                  |
| XGB Δ CAGR/Vol     | -0.2518 ❌               |
| XGB_INV Δ CAGR/Vol | +0.0386 ✅               |

**Interpretation**: XGB significantly underperforms baseline. Inversion helps but only modestly.

---

### CONFIG_B: Fresh Training (train_end=2023-12-31)

| Metric             | Value                    |
| ------------------ | ------------------------ |
| OOS Date Range     | 2024-03-28 to 2024-10-01 |
| OOS Dates Scored   | 4                        |
| IC Mean            | +0.0059                  |
| XGB Δ CAGR/Vol     | -0.2518 ❌               |
| XGB_INV Δ CAGR/Vol | +0.0386 ✅               |

**Interpretation**: Same backtest period as CONFIG_A (only 4 common dates). IC slightly positive.
XGB still underperforms; inversion helps. Results identical to A due to overlapping common dates.

---

### CONFIG_C: Long OOS Window (train_end=2024-12-31, val_end=2016-12-31)

| Metric             | Value                    |
| ------------------ | ------------------------ |
| OOS Date Range     | 2017-03-31 to 2024-10-01 |
| OOS Dates Scored   | **32**                   |
| IC Mean            | +0.0019                  |
| XGB Δ CAGR/Vol     | +0.1163 ✅               |
| XGB_INV Δ CAGR/Vol | +0.1986 ✅               |

**Interpretation**: With 32 quarters of OOS data:

- **Both XGB and XGB_INVERSE beat baseline**
- XGB_INVERSE has higher delta (+0.1986 vs +0.1163)
- IC is essentially zero (+0.002) so the "signal" is effectively random

> [!IMPORTANT]
> CONFIG_C shows that both directions can outperform baseline when IC ≈ 0.
> This is likely due to **diversification effects** or **random variation**, not true predictive power.

---

## Key Observations

1. **IC ≈ 0 across all configs**: The XGB model with only technical features has no meaningful predictive signal.

2. **OOS window matters**: CONFIG_A/B with only 4 OOS dates is insufficient for conclusions. CONFIG_C with 32 dates provides better evidence.

3. **Both directions can win**: In CONFIG_C, both XGB (+0.116) and XGB_INVERSE (+0.199) beat baseline, proving lack of directional signal.

4. **Random noise dominates**: The performance differences are likely due to:
   - Different top-k selection from score noise
   - Diversification vs concentration effects
   - Not systematic alpha

---

## Recommendations

1. **Do NOT adopt XGB alpha (normal or inverted)** — no evidence of real predictive power (IC ≈ 0)

2. **Need fundamentally different features** to generate real alpha:

   - SEC fundamental data (CompanyFacts)
   - Cross-sectional normalization
   - Sector-neutral scoring

3. **Technical features alone are insufficient** for ranking alpha in this universe

---

## Artifact Locations

```
artifacts/alpha_ab/xgb_q400_63d_sweep/
├── REPORT.md (this file)
├── cfg_a/
│   ├── diagnostics.json
│   ├── xgb/
│   │   ├── baseline_summary.json
│   │   ├── variant_summary.json
│   │   └── delta_summary.json
│   └── xgb_inverse/
│       ├── baseline_summary.json
│       ├── variant_summary.json
│       ├── delta_summary.json
│       └── xgb_scores_inverted.csv
├── cfg_b/
│   └── (same structure)
└── cfg_c/
    └── (same structure)

data/alpha/
├── xgb_model_63d_cfg_a/
│   ├── scores.csv
│   ├── ic_by_date.csv
│   └── summary.json
├── xgb_model_63d_cfg_b/
│   └── (same structure)
└── xgb_model_63d_cfg_c/
    └── (same structure)
```

---

## Conclusion

**Direction Verdict: UNSTABLE / NEED MORE**

The XGB alpha signal built from technical features only has IC ≈ 0. Neither inversion nor non-inversion provides consistent, interpretable alpha. The observed backtest improvements in CONFIG_C are likely due to noise/diversification, not true predictive power.

**Next Steps**:

- Add fundamental features (SEC data)
- Implement sector-neutral scoring
- Re-run sweep with enriched feature set
