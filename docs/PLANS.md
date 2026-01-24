# QUANT-NEURAL (v2026.1) — Master Plan

> **Single Source of Truth** for the Hybrid Quant + Neural project.

> [!IMPORTANT] > **For detailed task contracts (7.0–7.4)**, see [SSOT_TASKS.md](./SSOT_TASKS.md).
> That document is the definitive contract for APIs, I/O shapes, fail-safe policies, and determinism rules.

---

## 1. Purpose

QUANT-NEURAL is a modular, test-gated framework for building hybrid quantitative + neural network investment systems. The project emphasizes:

- **Point-in-Time (PIT) correctness**: No look-ahead bias.
- **Train-only fitting**: Scalers/transformers fit on training data only.
- **Reproducibility**: Deterministic seeds, no shuffle in time-series.
- **Fail-fast philosophy**: Raise clear exceptions on invalid inputs.

The system processes relative earnings momentum signals (X: T × 20) to predict 10-dim continuous targets (e.g., sector returns/scores); allocation mapping is a separate downstream layer.

---

## 2. Non-Goals (Explicitly Out of Scope)

The following are **NOT** in scope for the initial implementation phases:

- [ ] Production deployment infrastructure (Docker, Kubernetes)
- [ ] Real-time trading integration
- [ ] Alternative data sources (sentiment, news)
- [ ] Multi-asset class support (bonds, commodities)
- [ ] Complex ensemble models (beyond baseline MLP)
- [ ] Hyperparameter optimization frameworks (Optuna, Ray Tune)
- [ ] Web UI or dashboard

---

## 3. Non-Negotiables (Must Never Violate)

> [!CAUTION]
> Violating any of these rules will result in immediate PR rejection.

### 3.1 Point-in-Time (PIT) / No Look-Ahead

- All computations execute under a fixed `as_of_date`.
- Never use `datetime.now()` or system time implicitly.
- Time indices must be sorted ascending before transforms.

### 3.2 Train-Only Fit (No Data Leakage)

- Scalers, transformers, RankGauss: **fit on TRAIN only**.
- Validation/Test: **transform-only**.
- Tests must fail if fit is accidentally applied to val/test.

> [!NOTE] > **Winsorization** is cross-sectional per-date clipping (no future leakage by design), distinct from train-only-fitted transforms.

### 3.3 Time-Series Splits Only

- **NO** random KFold, shuffle, or random splits.
- Use `TimeSeriesSplit` or strictly ordered chronological splits.

### 3.4 Keras Training Constraints

- `shuffle=False` is **MANDATORY**.
- Use NumPy/Pandas arrays, not `tf.data` pipelines (to avoid silent shuffle).

### 3.5 Hamilton Filter Leakage Warning

- Hamilton filter uses forward values `y_{t+h}` (inherently non-causal).
- For backtest: allowed for comparison.
- For live logic: **enforce** `macro_lag_months >= h`.

### 3.6 Fail-Fast on Invalid Inputs

- Shape mismatches, negative weights, missing columns, non-finite values: **raise exceptions**.
- Do NOT silently drop data or guess defaults.

> [!NOTE] > **Exception — Black-Litterman Portfolio Layer (Task 7.3)**: The BL optimization module is intentionally **fail-safe**, not fail-fast. Public APIs must never raise to the caller; they return safe outputs and log `BL_FAILSAFE:*` warnings. See [SSOT_TASKS.md](./SSOT_TASKS.md) Section E for details.

---

## 4. Milestones / Task Roadmap

> [!NOTE]
> Items marked **Superseded** were part of the earlier synthetic/legacy plan and are replaced by Phase 7.x real-data track.

| Task           | Description                                                                               | Status        |
| -------------- | ----------------------------------------------------------------------------------------- | ------------- |
| **Task 0**     | Repo bootstrap + SSOT docs                                                                | ✅ Completed  |
| **Task 0.1**   | SSOT alignment patch                                                                      | ✅ Completed  |
| **Task 0.2**   | SSOT consistency sweep                                                                    | ✅ Completed  |
| **Prompt A**   | `utils/math_tools.py` + `tests/test_math_tools.py`                                        | ✅ Completed  |
| **Task 1.1**   | File-scope compliance (remove utils/**init**.py)                                          | ✅ Completed  |
| **Prompt B**   | `src/preprocessing.py` + `tests/test_preprocessing.py`                                    | ✅ Completed  |
| **Task 2.1**   | Dependency declaration (pyproject.toml)                                                   | ✅ Completed  |
| **Prompt C**   | `src/factors.py` + `tests/test_factors.py`                                                | ✅ Completed  |
| **Task 3.1**   | FutureWarning cleanup (groupby transform)                                                 | ✅ Completed  |
| **Prompt D**   | `src/selection.py` + `src/regime.py` (synthetic)                                          | ⚪ Superseded |
| **Prompt E**   | `src/models.py` (synthetic baseline)                                                      | ⚪ Superseded |
| **Smoke Test** | `main_executor.py` synthetic end-to-end                                                   | ⚪ Superseded |
| **Phase 7.x**  | Real-Data Integration (Stooq/SEC/H1H2/BL/HP guard) — see [SSOT_TASKS.md](./SSOT_TASKS.md) | ✅ Completed  |
| **Phase 8.x**  | Advanced upgrades (Conformal/MAPIE, Temperature scaling, SHAP, optional KAN)              | ⏳ Future     |

---

## 5. Definition of Done Philosophy

Every task completion requires:

1. **File-Scope Control**: Only allowlisted files modified.
2. **Spec-Matching Interfaces**: Names, signatures, behavior match spec exactly.
3. **PIT/No-Look-Ahead Preserved**: Verified through tests.
4. **Train-Only Fit Enforced**: Tested explicitly.
5. **`pytest -q` Passes**: Exit code 0, full output provided.
6. **Report Format**: Files changed, rationale, pytest output, diff summary.

---

## 6. Current Status

> **As of 2026-01-15**: Phase 10.x (XGB Alpha Model Pipeline) complete. Next: Run A/B backtest on real data.

### Completed

- [x] Task 0–3.1: Core infrastructure (repo skeleton, preprocessing, factors)
- [x] Task 7.0: Real-Data Integration (Stooq/SEC/H1H2 momentum features)
- [x] Task 7.1: Health Gates (coverage, missingness, PIT invariance)
- [x] Task 7.2: Train/Eval Pipeline (RankGauss, dataset building, MLP skeleton)
- [x] Task 7.2.4: Config alias (`label_tickers_in_order`) + explicit metadata keys
- [x] Task 7.2.5.x: Sector counts diagnostics & representativeness health gates
- [x] Task 7.3: Black-Litterman Portfolio Optimization (7 public APIs, never-crash)
- [x] Task 7.4.1: HP Filter Leakage Guard (`leakage_guard=True`, `lookback=120`)
- [x] Task 7.4.2: SSOT Documentation Lock (see [SSOT_TASKS.md](./SSOT_TASKS.md))
- [x] Task 7.4.3: Regime logistic reproducibility (`solver`, `random_state`)
- [x] Task 7.5.x: Backtest Harness (see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5)
- [x] Task 7.2.6: Predictions → target_weights Adapter (see [SSOT_TASKS.md](./SSOT_TASKS.md) § D.6)
- [x] Task 7.2.7: E2E Wiring Smoke (see [SSOT_TASKS.md](./SSOT_TASKS.md) § D.7)
- [x] Task 7.5.5: CSV Runner "run button" (see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.6)
- [x] Task 7.5.6: Backtest Artifact Writer (see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.6)
- [x] Task 7.5.7: CLI Runner (see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.7)
- [x] Task 8.1.0: Split Conformal Prediction Intervals (see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.6.1)
- [x] Task 8.2.0: Temperature Scaling Calibration (see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.6.2)
- [x] Task 8.3.0: SHAP-Style Explainability (see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.6.3)

### Test Status

- **All tests passing**: ✅

### Next Steps — Phase 9.x (Tradability + Harness Hardening + ML Shadow)

- [x] Task 9.0.0: Evaluation Matrix (CLI-only parameter sweep)
- [x] Task 9.0.1: Tradability Diagnostics (holdings/concentration analysis)
- [x] Task 9.0.2: TopK Expansion Sweep (K=200,300,400 × M,Q)
- [x] Task 9.0.3: Cost/Slippage Sensitivity (Q300/Q400 × 3 scenarios)
- [x] Task 9.0.4: Decision Lock + Safe CLI Defaults — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.9
- [x] Task 9.1.0: Tradable Rank (rank + top_k sparsification) — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.8
- [x] Task 9.2.0: Backtest Timing Semantics Lock — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.10
- [x] Task 9.2.1: Backtest NaN Immunity — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.11
- [x] Task 9.3.0: MLP Shadow Scoring + Artifact Export — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.14
- [x] Task 9.3.1: Wire shadow MLP into E2E runner (optional path) — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.15
- [x] Task 9.3.2: A/B Backtest Harness — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.16
- [x] Task 9.3.2.1: A/B Harness Hardening (temp CSV + cleanup) — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.17
- [x] Task 9.4.0: Ticker→Sector Mapping ETL — see [SSOT_TASKS.md](./SSOT_TASKS.md) § F.5.18
- [x] Task 9.4.1: Generate real-data ticker_to_sector.csv + export scores_mlp.csv + run A/B (9.3.3)
- [x] Task 9.3.10: SSOT/PLANS Lock — Close Sector-Fundamental MLP Alpha Track — see [SSOT_TASKS.md](./SSOT_TASKS.md) § G.4

### 🔒 Locked Baseline for Next Experiments

- **Baseline fixed**: `rebalance=Q, method=topk, top_k=400, cost_bps=10, slippage_bps=5`
- **CLI defaults updated**: `run_scores_backtest_cli.py` now uses safe production defaults
- **Shadow scoring available**: `run_shadow_scoring_mlp()` exports MLP predictions for A/B testing prep
- **A/B harness ready**: `run_ab_backtest_from_score_csvs()` compares baseline vs variant scores
- **Ticker-sector mapping ETL**: `build_ticker_to_sector_csv()` enables sector→ticker broadcast
- **Sector diagnostics**: `src/sector_model_diagnostics.py` provides IC/hit-rate/tie analysis
- **Decision locked**: H1/H2 Fundamental MLP NOT adopted for alpha (see SSOT § G.4)
- **Optional future lever** (only if needed): trade-band/hysteresis to further reduce churn.

### Next Milestone — Phase 9.5 (Shadow ML for Risk/Execution Control) ✅ COMPLETE

- [x] Task 9.5.0: Shadow Risk Exposure Logit — base model + deterministic CSV (p_risk_off, w_beta, exposure)
- [x] Task 9.5.1: Shadow Risk Hardening — label NaN fix, fallback VAL+TEST coverage, train-only tests
- [x] Task 9.5.2: Metrics JSON Export — Brier/AUC/log_loss/calibration bins/ECE per split
- [x] Task 9.5.3: Overlay Backtest + CSV Identity Lock — SPY/cash overlay, shifted-weight semantics
- [x] Task 9.5.4: SSOT/PLANS Sync — Decision lock + runbook (this task)

> **Scope Reminder**: Shadow-only diagnostics/artifacts; NO trading impact (no change to selection/weights/backtest outcomes unless optional export paths are explicitly enabled).

### Next Decision Gate — Phase 9.6 (Risk Gating Promotion Decision)

> [!NOTE] > **9.6.0 — Decision Gate**: Determine whether to promote risk gating from shadow-only to execution-control.
>
> **Inputs for Decision**:
>
> - Overlay metrics evidence (CAGR, Sharpe, max drawdown) from 9.5.3
> - Calibration quality (ECE, Brier) from 9.5.2
> - Comparison vs. buy-and-hold SPY baseline
>
> **Acceptance Criteria for Promotion** (to be validated before enabling):
>
> - Overlay CAGR/Vol ≥ 1.0 (risk-adjusted improvement)
> - Max drawdown reduction ≥ 20% vs. SPY buy-and-hold
> - ECE < 0.05 (well-calibrated probabilities)
>
> **If Promoted**: Enable exposure gating in portfolio construction (still no alpha selection impact).
> **If Not Promoted**: Retain as shadow diagnostics only; revisit with alternative signals.

### Next Milestone — Phase 9.6.x (Risk Signal Ablation) — IN PROGRESS

- [x] **Task 9.6.6: Shadow Risk Horizon Ablation (63d vs 21d)** — see [SSOT_TASKS.md](./SSOT_TASKS.md) § G.5.6
  - **Files Modified**: `run_shadow_risk_evaluation.py`, `tests/test_run_shadow_risk_evaluation.py`
  - **Implementation Details**:
    - Added `--horizon-days` CLI argument (default=63)
    - Output path structure changed to `<output_root>/horizon_<H>/{logit,mlp}/...`
    - New test class `TestHorizonWiring` with comprehensive ablation verification
  - **Experiment Config**: SPY-only, as_of_date=2024-10-01, train_end=2022-12-31, val_end=2023-12-31
  - **Results**:
    | Metric | H=63d (Baseline) | H=21d (Ablation) | Delta |
    |--------|------------------|------------------|-------|
    | **Logit test_ece** | 0.2574 | **0.1319** | **-48.8%** ✅ |
    | **MLP test_ece** | 0.4472 | 0.4402 | -1.6% ⚠️ |
    | Logit cagr/vol | 3.087 | 2.380 | -22.9% |
    | MLP cagr/vol | 3.519 | 2.351 | -33.2% |
    | Logit max_dd | -4.02% | -5.54% | +1.5%p |
    | MLP max_dd | -3.54% | -3.70% | +0.2%p |
  - **Findings**:
    - Logit model shows **significant ECE improvement** with shorter horizon
    - MLP model shows negligible change (model-specific issue requiring separate investigation)
    - Both models show reduced cagr/vol (risk-return tradeoff exists)
  - **Decision**: Partial success. Logit horizon=21 is adoption candidate; MLP requires hyperparameter tuning or architecture changes.

### Next Milestone — Phase 9.7 (Universe Market Cap Gate) — COMPLETE

- **PIT-safe market-cap gate**: `scripts/build_market_cap_universe.py` computes `market_cap = shares_outstanding_PIT * price_ffill_to_date`.
- **Microcap suppression without API changes**: ineligible tickers have their score set to `row_min - penalty` (finite), preserving existing `scores_to_target_weights()` fail-fast contracts.
- **Artifacts**: `data/backtest_universe_sec_mktcap` contains gated `scores.csv` (+ optional `market_cap.csv`, `eligibility.csv`, `summary.json`).

### Next Milestone — Phase 9.8 (Walk-Forward Ridge Alpha Baseline) — COMPLETE (Research-only)

- **Script**: `scripts/train_ridge_alpha_walkforward.py` trains a walk-forward Ridge model using PIT-safe SEC snapshots + price momentum on the market-cap gated universe.
- **Contracts preserved**: exports wide, finite `scores.csv` (gates ineligible/missing tickers to `row_min - penalty`) so the existing backtest harness/adapter requires no changes.
- **Artifacts**: `scores.csv` + `ic_by_date.csv` + `summary.json` (feature list + OOS IC diagnostics; `out_dir` configurable).
- **Current evidence**: label horizon/transform + sector-neutral options improve IC, but still trail momentum baseline in backtest → research track only.

### Next Milestone — Phase 9.9 (Form345 Insider Features + Ridge Integration) — COMPLETE (Research-only)

- **ETL**: `scripts/build_form345_insider_features.py` builds a PIT-safe (filing-date bucketed) insider activity panel from `data/raw/insiders`.
- **Artifacts**: `data/processed/insiders/insider_events_form345.csv` + `*.summary.json`.
- **Model hook**: `scripts/train_ridge_alpha_walkforward.py` supports `--insider_events_csv` to add rolling insider value/count features (scaled by market cap).
- **Current evidence**: small IC/backtest lift, but still below the topk momentum baseline → keep iterating before any adoption.

### Next Milestone — Phase 9.6.15 (Ops Mode Champion Lock) — COMPLETE

- **Champion locked**: XGB raw is the operational champion for shadow risk exposure
- **Ops report artifact**: `ops_shadow_risk_report.json` with schema_version 9.6.16
- **CLI args added**: `--ops-mode`, `--ops-champion-variant`, `--ops-overlay-mode`, `--ops-calibration-mode`
- **Decision gating**: status = OK/WARN/FAIL with recommended_action = KEEP/REVIEW/HALT

### Next Milestone — Phase 10.x (XGB Alpha Model Pipeline) — COMPLETE

- [x] **Task 10.0**: Review `scripts/train_xgb_alpha_walkforward.py` (walk-forward XGB regressor, price momentum features)
- [x] **Task 10.1.1**: Alpha Dataset Pipeline (`src/alpha_features.py`, `src/build_alpha_dataset.py`)
  - Vectorized features: vol_20d, mom_5d/21d/63d, rsi_14d, bbands_20d, atr_14d_norm
  - Forward return targets: fwd_ret_5d/10d/21d
  - PIT cutoff, float32 optimization, csv.gz output
- [x] **Task 10.1.2**: Train Alpha Model from Dataset (`scripts/train_xgb_alpha_from_dataset.py`)
  - Walk-forward training with PIT-safe scoring
  - Deterministic XGB params (n_jobs=1, subsample=1.0)
  - Wide scores.csv + ic_by_date.csv + summary.json
- [x] **Task 10.1.3**: A/B Backtest Wrapper (`scripts/run_xgb_alpha_ab_backtest.py`)
  - Locked Q400 baseline (rebalance=Q, top_k=400, cost_bps=10, slippage_bps=5)
  - Calls existing `run_ab_backtest_from_score_csvs()`
  - Outputs: baseline_summary.json, variant_summary.json, delta_summary.json

> **Next Gate**: Run full A/B backtest on real data to compare XGB alpha vs baseline scores.

### Future — Phase 8.x+ (Optional)

- [ ] 8.4: KAN architecture (optional, only if results justify)

> **vNext Candidates**: FT-Transformer, TabResNet — only after 8.x completion.

---

## 7. Repository Structure (Enforced)

```
Project_Root/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── factors.py
│   ├── selection.py
│   ├── regime.py
│   └── models.py
├── utils/
│   └── math_tools.py
├── configs/
│   ├── hyperparameters.yaml
│   └── schema.md
├── tests/
│   ├── test_math_tools.py
│   ├── test_preprocessing.py
│   ├── test_factors.py
│   ├── test_selection.py
│   ├── test_regime.py
│   └── test_models.py
├── docs/
│   ├── PLANS.md          ← This file (SSOT)
│   └── DATA_CONTRACT.md  ← Data contracts
├── main_executor.py
└── pyproject.toml
```

> [!IMPORTANT]
> Do NOT create additional top-level folders. All new modules must fit within this structure.

---

## 8. Contributing Guidelines

1. **Read the spec thoroughly** before implementing.
2. **Implement exactly as specified** — no extras, no renames, no refactors.
3. **Add tests** that lock in critical rules (PIT, train-only fit, no shuffle).
4. **Run `pytest -q`** and include full output in your report.
5. **Stop after completing the task** — do not proceed to the next task automatically.
