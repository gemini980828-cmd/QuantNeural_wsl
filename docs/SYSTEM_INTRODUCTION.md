# QUANT-NEURAL System Introduction

> **Purpose**: This document provides a comprehensive introduction to the QUANT-NEURAL system for external AI assistants or developers who need to understand, modify, or extend the codebase.

> **Last Updated**: 2026-01-14

---

## 1. System Overview

**QUANT-NEURAL** is a modular, test-gated framework for building hybrid quantitative + neural network investment systems. The system is designed for **institutional-grade backtesting and signal generation** with strict data integrity guarantees.

### 1.1 Core Philosophy

| Principle               | Description                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Point-in-Time (PIT)** | All computations use only data available at the specified `as_of_date`. No look-ahead bias allowed.                 |
| **Train-Only Fit**      | Scalers, transformers, and models are fitted exclusively on training data. Validation/test sets use transform-only. |
| **No Shuffle**          | Time-series data is never shuffled. Keras training uses `shuffle=False` mandatory.                                  |
| **Fail-Fast**           | Invalid inputs raise clear exceptions immediately. No silent fallbacks (except Black-Litterman layer).              |
| **Determinism**         | Same inputs + same seed = identical outputs. All randomness is seeded.                                              |

### 1.2 Project Location

```
c:\1234\QuantNeural\
```

---

## 2. Repository Structure

```
QuantNeural/
├── src/                          # Core modules
│   ├── preprocessing.py          # RankGauss, winsorization, HP filter
│   ├── factors.py                # Factor computations
│   ├── shadow_risk_exposure.py   # Shadow risk ML models (Logit/MLP)
│   ├── shadow_risk_overlay_diagnostics.py  # Exposure turnover analysis
│   ├── black_litterman_optimization.py     # BL portfolio optimization
│   └── ...
├── scripts/                      # ETL and batch processing scripts
│   ├── build_market_cap_universe.py
│   ├── train_ridge_alpha_walkforward.py
│   └── ...
├── tests/                        # Pytest test suite
├── configs/                      # Configuration files (YAML)
├── docs/                         # Documentation
│   ├── PLANS.md                  # Master plan and roadmap
│   └── SSOT_TASKS.md             # Detailed task contracts
├── data/
│   ├── raw/                      # Raw data (Stooq, SEC, etc.)
│   └── processed/                # Processed data
├── artifacts/                    # Generated outputs (backtests, scores)
├── run_shadow_risk_evaluation.py # Main evaluation runner
└── pyproject.toml                # Dependencies
```

---

## 3. Key Subsystems

### 3.1 Data Layer

| Component            | Description                                      | Key Files                              |
| -------------------- | ------------------------------------------------ | -------------------------------------- |
| **Stooq Prices**     | Daily OHLCV data from Stooq                      | `src/stooq_prices.py`                  |
| **SEC CompanyFacts** | Fundamental data from SEC EDGAR                  | `src/sec_companyfacts.py`              |
| **H1/H2 Momentum**   | Relative earnings momentum features (20 columns) | `src/h1h2_fundamental_momentum.py`     |
| **Market Cap Gate**  | PIT-safe market cap eligibility                  | `scripts/build_market_cap_universe.py` |

**PIT Rule for SEC Data**: Only rows with `filed <= as_of_date` are visible. Latest-filed wins for duplicates.

### 3.2 Preprocessing Pipeline

| Stage              | Description                       | PIT Safe                         |
| ------------------ | --------------------------------- | -------------------------------- |
| **RankGauss**      | Rank-based Gaussian transform     | ✅ Train-only fit                |
| **Winsorization**  | Cross-sectional per-date clipping | ✅ No future leakage             |
| **HP Filter**      | Hodrick-Prescott trend extraction | ✅ `leakage_guard=True` required |
| **StandardScaler** | Feature standardization           | ✅ Train-only fit                |

### 3.3 Backtest Harness

The backtest harness (`src/backtest_harness.py`) provides:

- **Rebalance frequencies**: Monthly (`M`) or Quarterly (`Q`)
- **Execution lag**: Configurable days between signal and trade
- **Cost model**: Transaction cost + slippage in basis points
- **Output**: Equity curve, daily returns, turnover, metrics (CAGR, Vol, Sharpe proxy)

**Locked Production Baseline**: `rebalance=Q, method=topk, top_k=400, cost_bps=10, slippage_bps=5`

### 3.4 Shadow Risk System (Phase 9.5-9.6)

The shadow risk system provides **risk exposure gating** without affecting alpha selection:

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  SPY Prices     │───▶│  Feature Engine  │───▶│  Logit/MLP Model   │
│  (single asset) │    │  (5 PIT features)│    │  (p_risk_off)      │
└─────────────────┘    └──────────────────┘    └────────┬───────────┘
                                                         │
                       ┌──────────────────┐              │
                       │  Overlay Backtest │◀────────────┘
                       │  (SPY/Cash switch)│    exposure = 1 - p_risk_off
                       └──────────────────┘
```

**Key Artifacts per Variant (logit/mlp)**:

- `shadow_risk.csv` — Daily predictions (p_risk_off, exposure_suggested)
- `shadow_risk_metrics.json` — Calibration metrics (ECE, Brier, ROC-AUC)
- `shadow_risk_overlay.csv` — Overlay equity curve
- `overlay_diagnostics.json` — Exposure turnover analysis

**Decision Gate Criteria** (for promotion to execution):

- ECE < 0.05
- Overlay CAGR/Vol ≥ 1.0
- Max drawdown reduction ≥ 20% vs SPY

---

## 4. Current Status (2026-01-14)

### 4.1 Completed Phases

| Phase   | Description                               | Status      |
| ------- | ----------------------------------------- | ----------- |
| 7.x     | Real-Data Integration (Stooq/SEC/H1H2)    | ✅          |
| 8.x     | Utilities (Conformal, Temp Scaling, SHAP) | ✅          |
| 9.0-9.4 | Backtest Harness Hardening                | ✅          |
| 9.5     | Shadow Risk ML (Logit/MLP)                | ✅          |
| 9.6.6   | Horizon Ablation Study (63d vs 21d)       | ✅          |
| 9.6.7-8 | Overlay Exposure Diagnostics              | ✅          |
| 9.7     | Universe Market Cap Gate                  | ✅          |
| 9.8-9.9 | Ridge Alpha & Insider Features            | ✅ Research |

### 4.2 Recent Experiment Results

**Shadow Risk Horizon Ablation (SPY-only, 2024-10-01)**:

| Model | Horizon | ECE       | CAGR/Vol | Max DD | Turnover |
| ----- | ------- | --------- | -------- | ------ | -------- |
| Logit | 63d     | 0.257     | 3.09     | -4.0%  | 3.46     |
| Logit | **21d** | **0.132** | 2.38     | -5.5%  | **0.82** |
| MLP   | 63d     | 0.447     | 3.52     | -3.5%  | 38.0     |
| MLP   | 21d     | 0.440     | 2.35     | -3.7%  | 41.4     |

**Finding**: Horizon=21 improves Logit ECE by 49% and reduces turnover by 76%, but lowers returns due to more conservative exposure.

---

## 5. Working with the Codebase

### 5.1 Running Tests

```powershell
cd c:\1234\QuantNeural
python -m pytest -q                           # All tests
python -m pytest tests/test_preprocessing.py  # Specific module
```

### 5.2 Running Shadow Risk Evaluation

```powershell
python run_shadow_risk_evaluation.py `
  --spy-csv-path data/raw/stooq/us/spy.us.txt `
  --spy-ticker SPY.US `
  --as-of-date 2024-10-01 `
  --train-end 2022-12-31 `
  --val-end 2023-12-31 `
  --horizon-days 63 `
  --output-root-dir artifacts/shadow_risk/spy_only_eval `
  --seed 42
```

### 5.3 Key Implementation Rules

1. **Always use `as_of_date`** parameter for PIT safety
2. **Fit on train only** — Never call `.fit()` on val/test data
3. **Keras training**: Always `model.fit(..., shuffle=False)`
4. **Test your changes**: Run `pytest -q` before committing
5. **Report format**: Files changed, rationale, pytest output

### 5.4 Adding New Modules

```
Allowed:
- Modify/create files in src/, tests/, scripts/, configs/
- Create artifacts in artifacts/

NOT Allowed:
- Add new top-level directories
- Add dependencies to pyproject.toml without explicit approval
- Change existing API signatures without updating SSOT
```

---

## 6. Important Files to Reference

| File                            | Purpose                                                        |
| ------------------------------- | -------------------------------------------------------------- |
| `docs/PLANS.md`                 | Master roadmap and current status                              |
| `docs/SSOT_TASKS.md`            | Detailed API contracts for each task                           |
| `pyproject.toml`                | Dependencies (numpy, pandas, sklearn, tensorflow, statsmodels) |
| `run_shadow_risk_evaluation.py` | Main CLI entrypoint for shadow risk                            |

---

## 7. Glossary

| Term          | Definition                                                            |
| ------------- | --------------------------------------------------------------------- |
| **PIT**       | Point-in-Time — using only data available at the reference date       |
| **ECE**       | Expected Calibration Error — measures probability calibration quality |
| **Overlay**   | Strategy that switches between SPY and cash based on risk signal      |
| **Shadow**    | Diagnostic-only mode — no impact on actual trading decisions          |
| **Train-End** | Last date of training period (exclusive for val/test)                 |
| **Val-End**   | Last date of validation period (exclusive for test)                   |
| **Horizon**   | Forward-looking period for label construction (e.g., 63 trading days) |
| **Fail-Safe** | Function that never raises to caller; returns safe defaults on error  |
| **Fail-Fast** | Function that raises immediately on invalid input                     |

---

## 8. Contact & Conventions

### 8.1 KIRA 5-Agent Workflow (Implicit)

This project uses the KIRA multi-agent workflow:

1. **Router** — Classifies request type
2. **Validator** — Checks PIT/design compliance
3. **Implementer** — Writes code and tests
4. **Analyst** — Analyzes results
5. **Memory Manager** — Records learnings

### 8.2 Response Format

All implementation responses should include:

```
A) Files Changed
B) Rationale
C) pytest -q output + Exit Code
D) Example usage
E) STOP
```

---

> **For detailed API contracts, see `docs/SSOT_TASKS.md`.** > **For project roadmap and status, see `docs/PLANS.md`.**
