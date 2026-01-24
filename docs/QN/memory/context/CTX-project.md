# QuantNeural Project Context

**Last Updated**: 2026-01-07

## Project Overview

QUANT-NEURAL v2026.1 - Hybrid Quant + Neural investment system.

## Current Phase

**Implementation Phase** - Building core modules per SSOT specs.

## Completed Tasks

| Task     | Description                                         | Commit  |
| -------- | --------------------------------------------------- | ------- |
| Task 0   | Repo bootstrap + SSOT docs                          | 9f80ecd |
| Task 0.1 | SSOT alignment patch                                | -       |
| Task 0.2 | SSOT consistency sweep                              | -       |
| Prompt A | `utils/math_tools.py` (weighted_harmonic_mean)      | -       |
| Task 1.1 | File-scope compliance (remove utils/**init**.py)    | -       |
| Prompt B | `src/preprocessing.py` (RankGauss, HP, Hamilton)    | -       |
| Task 2.1 | Dependency declaration                              | 76ca2c3 |
| Prompt C | `src/factors.py` (winsorize, zscore, style factors) | 7cca194 |
| Task 3.1 | FutureWarning cleanup                               | c98320a |

## Next Tasks

- [ ] Prompt D: `src/selection.py` + `src/regime.py`
- [ ] Prompt E: `src/models.py` (Baseline MLP)
- [ ] Smoke Test: `main_executor.py`

## Test Status

- **Total tests**: 45
- **All passing**: ✅
- **Warnings**: None

## Key Modules Implemented

| Module                 | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `utils/math_tools.py`  | weighted_harmonic_mean for valuation aggregation           |
| `src/preprocessing.py` | QuantDataProcessor (RankGauss, HP filter, Hamilton filter) |
| `src/factors.py`       | WinsorizeParams, zscore_cross_section, build_style_factors |

## SSOT Documents

- `docs/PLANS.md` - Master plan and roadmap
- `docs/DATA_CONTRACT.md` - Data semantics and validation rules

| Task 4.0 | Real Data Experiment (Stooq+SEC) | fb209d2 |

## Status Update: 2026-01-07

## Status Update: 2026-01-08

## Status Update: 2026-01-08 (Task 7.4.2)

- **Last Task**: Task 7.4.2 - SSOT Documentation Lock
- **AS_OF_DATE**: 2026-01-08
- **Key Outcome**: Created `docs/SSOT_TASKS.md` as definitive contract document for Tasks 7.0-7.4. Locked non-negotiables, 7 BL public APIs, X=20/Y=10 shape contracts, HP leakage guard specs.
- **Decision**: [[DEC-20260108-1041-ssot-tasks-lock]]
- **Tests**: All passing ✅

- **Last Task**: Task 7.3.1.2 - BL Optimizer Constraint Safety
- **Key Outcome**: Black-Litterman module is now "never-crash" AND "never-invalid". All public functions return constraint-satisfying outputs.
- **Commit**: 3313a90
- **Tests**: 41 BL tests, 86+ total passing
- **Learning**: [[LRN-20260108-1000-bl-optimizer-constraint-projection]]

### Key Fixes
1. `calibrate_sector_views`: output shape always (K,)
2. `rmt_denoise_covariance`: use q=T/N (no forced >=1)
3. `optimize_portfolio_cvxpy`: proper constraint projection with slack-filling

- **Last Task**: Real Data Experiment Verification
- **AS_OF_DATE**: 2024-12-31
- **Key Outcome**: Config-driven end-to-end experiment passed (MSE~0.009). Fixed Stooq loader ticker format issue.
- **AutoQC**: ✅ PASS
