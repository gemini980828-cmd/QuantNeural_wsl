# QUANT-NEURAL — Single Source of Truth: Task Contracts

> **SSOT for Tasks 7.0–8.3** — Definitive contracts for APIs, I/O shapes, determinism, and fail-safe behavior.

---

## A) Non-Negotiables (Must Never Violate)

> [!CAUTION]
> Violating any of these rules results in immediate rejection.

### A.1 Point-in-Time (PIT) / No Look-Ahead

- All computations execute under a fixed `as_of_date`.
- **NEVER** use `datetime.now()` or `date.today()`.
- Time indices must be sorted ascending before transforms.
- SEC data: only rows with `filed <= as_of_date` are visible.

### A.2 Train-Only Fit (No Data Leakage)

- Scalers, RankGauss, calibration params: **fit on TRAIN only**.
- Validation/Test sets: **transform-only**.
- Tests must fail if fit is applied to val/test.

### A.3 No Shuffle in Time-Series

- **NO** random KFold, shuffle, or random splits.
- Keras: `model.fit(..., shuffle=False)` is **MANDATORY**.
- Use `TimeSeriesSplit` or strictly chronological ordering.

### A.4 Reproducibility

- Use deterministic seeds (`random_state`, `seed`) where applicable.
- Tests must be deterministic (no flaky randomness).

### A.5 Pytest-Gated Workflow

- Every implementation must run `pytest -q` and pass with exit code 0.
- No PR merges with failing tests.

---

## B) Roadmap Numbering

| Phase  | Description                                    | Status       |
| ------ | ---------------------------------------------- | ------------ |
| 6.x    | Core preprocessing, factors, selection, regime | ✅ Completed |
| 7.0    | Real-Data Integration (Stooq/SEC/H1H2)         | ✅ Completed |
| 7.1    | Health Gates & Dataset Building                | ✅ Completed |
| 7.2    | Train/Eval Pipeline (RankGauss, splits, MLP)   | ✅ Completed |
| 7.3    | Black-Litterman Portfolio Integration          | ✅ Completed |
| 7.4    | HP Filter Leakage Guard & SSOT Lock            | ✅ Completed |
| 7.5    | Backtest Harness (minimal deterministic loop)  | ✅ Completed |
| 8.1    | Split Conformal Prediction Intervals           | ✅ Completed |
| 8.2    | Temperature Scaling Calibration                | ✅ Completed |
| 8.3    | SHAP-Style Explainability                      | ✅ Completed |
| 8.4    | KAN architecture (optional)                    | ⏳ Future    |
| 9.6    | Shadow Risk Exposure Model (SPY-Only)          | ✅ Completed |
| 9.6.15 | Ops Mode Champion Lock (XGB)                   | ✅ Completed |
| 10.0   | XGB Alpha Walk-Forward (review)                | ✅ Completed |
| 10.1.1 | Alpha Dataset Pipeline (Features & Targets)    | ✅ Completed |
| 10.1.2 | Train Alpha Model (XGB) Walk-Forward           | ✅ Completed |
| 10.1.3 | A/B Backtest: Baseline vs XGB Alpha            | ✅ Completed |

---

## C) Task 7.0 Contracts — Real Data Integration

### C.1 Stooq Price Data

- **Module**: `src/stooq_prices.py`
- **PIT Rule**: `load_stooq_daily_prices(csv_path, as_of_date=...)` filters to dates ≤ `as_of_date`.
- **Output**: DataFrame with columns `[date, open, high, low, close, volume]`.

### C.2 SEC CompanyFacts

- **Module**: `src/sec_companyfacts.py`
- **PIT Rule**: Only rows with `filed <= as_of_date` are visible.
- **Latest-Filed Wins**: For duplicate `(cik, tag, unit, end)`, the row with newest `filed` wins.

### C.3 H1/H2 Fundamental Momentum Features

- **Module**: `src/h1h2_fundamental_momentum.py`
- **Function**: `build_h1h2_relative_fundamental_momentum(facts, *, month_ends, cik_to_sector, ...)`
- **Shape Contract**: Always outputs exactly **20 columns**:
  - `S0_H1, S1_H1, ..., S9_H1` (10 sectors × H1 short momentum)
  - `S0_H2, S1_H2, ..., S9_H2` (10 sectors × H2 long momentum)
- **n_sectors**: Must be exactly 10 (raises `ValueError` otherwise).
- **PIT Rules Applied**:
  - `_get_pit_visible_facts()`: filters `facts[facts["filed"] <= month_end]`
  - Latest-filed wins via sort + drop_duplicates

---

## D) Task 7.1/7.2 Contracts — Health Gates & Training Pipeline

### D.1 Health Gates

- **Module**: `src/real_data_health_gates.py`
- **Function**: `run_real_data_health_gates(frame, *, as_of_date, min_months=18, max_feature_missing_ratio=0.20, ...)`
- **Expected Columns**: Exactly 20 (`expected_h1h2_columns()` returns `S0_H1..S9_H1, S0_H2..S9_H2`).
- **Gates**:
  - Coverage: `min_months` threshold
  - Missingness: per-column NaN ratio ≤ `max_feature_missing_ratio`
  - PIT invariance: `check_no_lookahead_invariance()` verifies earlier cutoffs unchanged by later data

### D.2 X/Y Shape Contracts

| Tensor | Shape     | Columns                      |
| ------ | --------- | ---------------------------- |
| **X**  | `(n, 20)` | `S0_H1..S9_H1, S0_H2..S9_H2` |
| **Y**  | `(n, 10)` | `S0_Y, S1_Y, ..., S9_Y`      |

- **Module**: `src/real_data_dataset.py`
- **Function**: `build_real_data_xy_dataset(...)` → `(X_aligned, Y_aligned)`
- **Validation**: Raises `ValueError` if X ≠ 20 columns or Y ≠ 10 columns.

### D.3 Label Construction

- **Formula**: `y_t = (close[t+1] / close[t]) - 1` (next-month return)
- **Label Shift**: Last row has NaN (no t+1 available) — row is dropped during alignment.

### D.4 RankGauss — Train-Only Fit

- **Module**: `src/preprocessing.py`
- **Class**: `QuantDataProcessor`
- **Methods**:
  - `fit_rankgauss(X_train)` — fit on train only
  - `transform_rankgauss(X)` — apply to train/val/test
- **Raises**: `RuntimeError` if `transform_rankgauss` called before `fit_rankgauss`.

### D.5 Determinism

- `random_state` parameter controls reproducibility.
- Same `random_state` + same data → identical output.

### D.6 Task 7.2.6 Contracts — Predictions → target_weights Adapter

**Files:**

- `src/weights_adapter.py`
- `tests/test_weights_adapter.py`

**Public API:**

```python
def scores_to_target_weights(
    scores: pd.DataFrame,
    *,
    method: str = "softmax",      # "softmax" | "rank" | "topk"
    temperature: float = 1.0,     # for softmax (> 0)
    top_k: int | None = None,     # for topk method
    max_weight: float | None = None  # optional cap in (0, 1]
) -> pd.DataFrame
```

**Non-Negotiable Semantics:**

- **Input validation (fail-fast)**:
  - `scores` must be a pandas DataFrame (not numpy)
  - Index must be monotonic increasing, unique
  - Columns must be unique; `k_assets >= 2`
  - All values must be finite (no NaN/inf)
- **Output contract**:
  - Long-only: `w >= 0`
  - Each row sums to exactly `1.0`
  - Same shape, index, columns as input
- **Determinism**:
  - Identical inputs → identical outputs
  - Tie-breaking: stable sort by column name after score

**Method Definitions:**

- **softmax**: `w_i ∝ exp((s_i - max(s)) / T)`, normalized
- **rank**: rank-based weights (higher score = higher rank), normalized
- **topk**: equal weight `1/top_k` on top-k assets by score

**max_weight Enforcement:**

- Feasibility: `max_weight * k_assets >= 1.0`, else `ValueError`
- Redistribution: excess weight redistributed proportionally to uncapped assets

### D.7 Task 7.2.7 Contracts — E2E Wiring Smoke (Scores → target_weights → Backtest)

**Files:**

- `src/e2e_backtest.py`
- `tests/test_e2e_backtest.py`

**Public API:**

```python
def run_scores_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    price_col="close", rebalance="M", execution_lag_days=1,
    method="softmax", temperature=1.0, top_k=None, max_weight=None,
    cost_bps=0.0, slippage_bps=0.0, initial_equity=1.0, max_gross_leverage=None
) -> dict
```

**Non-Negotiable Semantics:**

- **Thin wrapper**: calls `scores_to_target_weights()` then `run_backtest()` (no duplicated logic)
- **Fail-fast pass-through**: does NOT catch exceptions; lets underlying validators raise
- **Deterministic**: no clock, no network, no randomness
- **Output contract**: returns the 8 harness keys unchanged + adds `"target_weights"` (pre-align weights)

**Test-Locked Behaviors:**

- Wide/long format equivalence (identical equity curves)
- Determinism repeatability (repeated calls identical)
- Quarterly rebalance yields fewer rebalances than monthly
- Out-of-range scores fail-fast (ValueError)
- TopK and max_weight integration works

---

## E) Task 7.3 Contracts — Black-Litterman Portfolio Optimization

### E.1 Module Overview

- **Module**: `src/black_litterman_optimization.py`
- **Policy**: "Never-Crash" + "Never-Invalid" — all functions return safe defaults on invalid input.

### E.2 Seven Public APIs

| #   | Function                    | Purpose                                    | Fail-Safe Behavior                               |
| --- | --------------------------- | ------------------------------------------ | ------------------------------------------------ |
| 1   | `enforce_monthly_units`     | Convert annualized → monthly units         | Returns safe defaults + warning                  |
| 2   | `compute_dynamic_ic`        | Dynamic IC from prob_up                    | Returns `ic_min` + warning                       |
| 3   | `calibrate_sector_views`    | Raw scores → monthly expected return views | Returns zeros (K,) + warning                     |
| 4   | `mc_dropout_sector_views`   | MC dropout for view uncertainty            | Returns fallback (zeros) + warning               |
| 5   | `rmt_denoise_covariance`    | RMT (Marcenko-Pastur) covariance denoising | Returns safe PSD + warning                       |
| 6   | `black_litterman_posterior` | BL posterior computation                   | Returns prior + warning                          |
| 7   | `optimize_portfolio_cvxpy`  | Portfolio optimization with fail-safe      | **Always** returns constraint-satisfying weights |

### E.3 Never-Crash & Never-Invalid Guarantees

- All functions catch exceptions internally.
- All functions return valid, usable outputs (never `None`, never crash).
- **Warning logs** indicate fallback activation.

### E.4 BL_FAILSAFE Warning Format

```
BL_FAILSAFE:<function_name>:<action> <key>=<value> ...
```

**Examples**:

```
BL_FAILSAFE:optimize_portfolio_cvxpy:relax_caps msw=0.15 mxsw=0.50
BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw=0.20 mxsw=0.60
BL_FAILSAFE:calibrate_sector_views:shape_mismatch len_scores=5 len_vol=10 returning_zeros
```

### E.5 Optimizer Fallback Tiers

The optimizer follows a strict tier order. Deterministic construction is attempted under **original caps** before any relaxation:

| Tier | Action                                             | Log Pattern                                                        |
| ---- | -------------------------------------------------- | ------------------------------------------------------------------ |
| 0    | cvxpy solve with original caps + γ-turnover        | (no log if success)                                                |
| 1    | cvxpy solve with γ × 0.8                           | `tier0_failed trying Tier 1`                                       |
| 2    | Deterministic construction with original caps      | `tier1_failed trying deterministic construction`                   |
| 3    | Iterative relax loop (msw × 1.1, mxsw × 1.1)       | `BL_FAILSAFE:optimize_portfolio_cvxpy:relax_caps msw=... mxsw=...` |
| 4    | Fallback to w_prev if feasible                     | `using_fallback_A`                                                 |
| 5    | Deterministic with fully relaxed caps (msw=mxsw=1) | `using_fallback_C`                                                 |
| 6    | Equal-weight 1/N                                   | (ultimate fallback)                                                |

**Parseable Log Lines** (always emitted on fallback):

```
BL_FAILSAFE:optimize_portfolio_cvxpy:relax_caps msw=<value> mxsw=<value> reason=<reason>
BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw=<value> mxsw=<value>
```

---

## F) Task 7.4 Contracts — HP Filter Leakage Guard

### F.1 HPParams Dataclass

```python
@dataclass
class HPParams:
    mode: HPMode = "classic"             # "classic", "ravn_uhlig", "manual"
    lamb_manual: Optional[float] = None  # Required if mode="manual"
    leakage_guard: bool = True           # NEW: enables PIT-safe rolling window
    lookback: Optional[int] = 120        # NEW: rolling window size (None = all past)
```

### F.2 PIT-Safe Definition

When `leakage_guard=True`:

- HP filter is applied in a **rolling window** fashion.
- At each index `i`, the filter sees only data `[max(0, i - lookback + 1) : i + 1]`.
- **Past outputs are invariant** to future data additions.
- If window is too short or hpfilter fails: `trend[i] = series[i]`, `cycle[i] = 0.0`.

When `leakage_guard=False`:

- Standard two-sided HP filter (uses all data including future).
- **Warning**: This is inherently non-causal. Use for offline analysis only.

### F.3 Test Contract

- `test_hp_leakage_guard_past_unchanged`: Past outputs must not change when future data is appended.
- `test_hp_two_sided_uses_future`: Two-sided mode (`leakage_guard=False`) produces different outputs than guarded mode.

---

## F.5) Task 7.5 Contracts — Backtest Harness

### F.5.1 Files

| File                             | Purpose                             |
| -------------------------------- | ----------------------------------- |
| `src/backtest_harness.py`        | Minimal deterministic backtest loop |
| `tests/test_backtest_harness.py` | Test suite (33+ tests)              |

### F.5.2 Public APIs

#### `validate_prices_frame(prices, *, price_col="close")`

- **Wide format**: DatetimeIndex strictly increasing, unique dates, all values finite and > 0.
- **Long format**: columns `["ticker", price_col]`, DatetimeIndex monotonic non-decreasing, allows duplicate dates. **(date, ticker) pairs must be unique**.
- Raises `ValueError` on violations.

#### `validate_target_weights(weights)`

- DatetimeIndex strictly increasing, unique.
- Values must be finite (negatives allowed for shorting).
- Raises `ValueError` on violations.

#### `resample_rebalance_dates(signal_dates, *, rebalance)`

- `"M"`: returns all signal dates (monthly).
- `"Q"`: returns every 3rd date (indices 0, 3, 6, ...) for quarterly.
- Raises `ValueError` for invalid `rebalance` value.

#### `align_and_lag_weights(weights, prices_index, *, execution_lag_days=1)`

- **execution_lag_days >= 0** required (raises `ValueError` if negative).
- Aligns each signal date to first trading day >= signal, then shifts by `execution_lag_days`.
- Drops signals that cannot be aligned (beyond calendar).

#### `run_backtest(prices, target_weights, *, ...)`

Full signature:

```python
run_backtest(
    prices, target_weights, *,
    price_col="close",
    rebalance="M",                  # "M" or "Q"
    execution_lag_days=1,           # >= 0
    cost_bps=0.0,
    slippage_bps=0.0,
    initial_equity=1.0,
    max_gross_leverage=None         # Optional cap
) -> dict
```

### F.5.3 Non-Negotiable Semantics

> [!IMPORTANT] > **Determinism**: No system clock, no network, no randomness. Same inputs => identical outputs.

> [!CAUTION] > **Fail-Fast on Missing Data**: After establishing backtest window, missing prices or returns for **USED tickers** must raise `ValueError`. No NaN masking (fillna(0.0)) allowed.

**used_tickers definition**:

```python
used_tickers = [t for t in tickers if (effective_weights[t].abs() > 0).any()]
```

Derived from `effective_weights` (post align+lag), NOT from raw `target_weights`. This avoids KeyError when prices have extra tickers not in weights.

**Turnover definition**:

```python
turnover = sum(|delta_w|) at each rebalance
```

**Cost model**:

```python
cost = equity * turnover * (cost_bps + slippage_bps) / 10000
```

Applied immediately at rebalance date, before computing that day's return.

### F.5.4 Output Contract

`run_backtest()` returns a `dict` with these **required keys**:

| Key               | Type               | Description                                                                       |
| ----------------- | ------------------ | --------------------------------------------------------------------------------- |
| `equity_curve`    | `pd.Series`        | Equity values indexed by trading days                                             |
| `daily_returns`   | `pd.Series`        | Daily portfolio returns                                                           |
| `rebalance_dates` | `pd.DatetimeIndex` | Effective trade dates after lag                                                   |
| `weights_used`    | `pd.DataFrame`     | Effective weights on rebalance dates                                              |
| `turnover`        | `pd.Series`        | Turnover at each rebalance date                                                   |
| `costs`           | `pd.Series`        | Costs applied at each rebalance                                                   |
| `trades`          | `pd.DataFrame`     | Columns: `date`, `ticker`, `delta_weight`                                         |
| `metrics`         | `dict`             | Keys: `cagr`, `ann_vol`, `sharpe`, `max_drawdown`, `total_turnover`, `total_cost` |

### F.5.5 Task Status

| Task  | Status       | Notes                                     |
| ----- | ------------ | ----------------------------------------- |
| 7.5.0 | ✅ Completed | Core harness with M/Q rebalance           |
| 7.5.1 | ✅ Completed | Long-format support + execution_lag guard |
| 7.5.2 | ✅ Completed | Fail-fast on missing prices (no NaN mask) |
| 7.5.3 | ✅ Completed | used_tickers bugfix (KeyError fix)        |
| 7.5.4 | ✅ Completed | SSOT documentation (this section)         |
| 7.5.5 | ✅ Completed | CSV Runner "run button" (see F.5.6)       |
| 7.5.6 | ✅ Completed | Artifact Writer (see F.5.6)               |
| 7.5.7 | ✅ Completed | CLI Runner (see F.5.7)                    |

### F.5.6 Task 7.5.5 Contracts — Backtest Runner from CSV Artifacts

**Files:**

- `src/run_scores_backtest_from_csv.py`
- `tests/test_run_scores_backtest_from_csv.py`

**Public API:**

```python
def run_scores_backtest_from_csv(
    *,
    prices_csv_path: str,
    scores_csv_path: str,
    price_col: str = "close",
    date_col: str = "date",
    ticker_col: str = "ticker",
    rebalance: str = "M",           # "M" | "Q"
    execution_lag_days: int = 1,    # >= 0
    method: str = "softmax",        # "softmax" | "rank" | "topk"
    temperature: float = 1.0,
    top_k: int | None = None,
    max_weight: float | None = None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_equity: float = 1.0,
    max_gross_leverage: float | None = None,
) -> dict
```

**Non-Negotiable Semantics:**

- **Deterministic**: no randomness, no system clock, no network
- **Fail-fast**: clear `ValueError` on invalid CSV formats or missing required columns
- **scores.csv format** (wide only):
  - Requires `date_col` + asset columns
  - Parsed to DatetimeIndex; must have unique dates
- **prices.csv format**:
  - Long if `ticker_col` present: requires `[date_col, ticker_col, price_col]`
  - Wide otherwise: requires `date_col` + ticker columns; unique dates
- **Thin wrapper semantics**:
  - Loads CSVs, then calls `src.e2e_backtest.run_scores_backtest()`
  - No duplicated backtest logic
  - No exception catching (pass-through fail-fast)

**Output Contract:**

- Returns the 8 harness keys + `"target_weights"` (from E2E wrapper), unchanged

**Test-Locked Behaviors:**

- Wide prices + wide scores smoke returns 9 keys
- Long prices equivalence to wide (identical equity curves)
- Missing required columns fail-fast (scores missing date, prices missing date/ticker/price_col)

---

### F.5.7 Task 7.5.7 Contracts — CLI Runner: CSV → Backtest → Output Artifacts

**Files:**

- `src/run_scores_backtest_cli.py`
- `tests/test_run_scores_backtest_cli.py`

**CLI Invocation:**

```bash
python src/run_scores_backtest_cli.py \
    --prices_csv_path data/prices.csv \
    --scores_csv_path data/scores.csv \
    --out_dir results/
```

**Required Arguments:**

- `--prices_csv_path` (str): Path to prices CSV
- `--scores_csv_path` (str): Path to scores CSV
- `--out_dir` (str): Output directory for artifacts

**Optional Arguments (pass-through):**

- `--rebalance` (M|Q), `--method` (softmax|rank|topk)
- `--temperature`, `--top_k`, `--max_weight`
- `--cost_bps`, `--slippage_bps`, `--initial_equity`, `--max_gross_leverage`
- `--execution_lag_days`, `--price_col`, `--date_col`, `--ticker_col`

**Non-Negotiable Semantics:**

- **Deterministic**: no randomness, no system clock, no network
- **Fail-fast**: argparse SystemExit on missing/invalid args; propagate file/validation errors
- **Thin wrapper**: calls `run_scores_backtest_from_csv()` (no duplicated logic)
- Output artifacts have explicit "date" columns for readability
- `summary_metrics.json` includes both metrics and CLI params used

**Output Artifacts (8 files):**
| File | Content |
|------|---------|
| `equity_curve.csv` | date, equity |
| `trades.csv` | date, ticker, delta_weight |
| `summary_metrics.json` | metrics + params |
| `target_weights.csv` | Pre-align weights |
| `weights_used.csv` | Applied weights |
| `turnover.csv` | date, turnover |
| `costs.csv` | date, cost |
| `returns.csv` | date, return |

**Test-Locked Behaviors:**

- Successful run creates all 8 artifacts
- Determinism: identical outputs for identical inputs
- Fail-fast: missing required arg, missing file, invalid rebalance choice
- Readability: equity_curve.csv has date/equity columns; trades.csv has date/ticker columns

---

### F.5.8 Task 9.1.0 Contracts — Weights Adapter (Tradable Rank)

**Files:**

- `src/weights_adapter.py`
- `tests/test_weights_adapter.py`

**Public API:**

```python
def scores_to_target_weights(
    scores: pd.DataFrame,
    *,
    method: str = "softmax",       # "softmax" | "rank" | "topk"
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    max_weight: Optional[float] = None,
) -> pd.DataFrame
```

**Method Semantics:**

| Method    | top_k    | Behavior                                           |
| --------- | -------- | -------------------------------------------------- |
| `softmax` | ignored  | Softmax(s/T), full dense weights                   |
| `topk`    | required | Equal weight 1/k on top_k assets, others 0         |
| `rank`    | None     | Dense rank-based weights (all assets)              |
| `rank`    | provided | **NEW (9.1.0)**: Sparse rank-weights on top_k only |

**Deterministic Tie-Breaking Rule:**

- Sort by score descending, then by column name ascending
- Guarantees identical outputs regardless of column ordering in input

**max_weight with Sparsity (9.1.0):**

- For sparse methods (topk, rank+top_k), max_weight redistributes only among active assets
- No leakage to zero-weight assets
- Feasibility uses active_count: `max_weight * active_count >= 1.0` required

**Fail-Fast Validation:**

- `ValueError` on: invalid method, temperature ≤ 0, top_k out of range
- `ValueError` on infeasible max_weight (active_count \* max_weight < 1.0)
- `ValueError` on NaN/inf in scores, non-monotonic index, duplicate index/columns

---

### F.5.9 Task 9.0.4 — Decision Lock: Default Production Candidate

**Status**: ✅ Complete

**Summary**:

Phase 9.0.x experiments (Evaluation Matrix, Tradability Diagnostics, TopK Expansion Sweep, Cost Sensitivity)
established the optimal execution-layer settings for the full-universe backtest harness.

**Decision Statements**:

1. Default (production candidate): `rebalance=Q, method=topk, top_k=400, cost_bps=10, slippage_bps=5`.
2. Rationale: Q400 matches Q300 Sharpe across cost scenarios while improving MaxDD and reducing total cost via lower turnover.
3. Rank-based methods are excluded for production due to non-tradable trade breadth (≈2,700 tickers traded per rebalance in dense rank).
4. Monthly remains research/aggressive only due to materially higher total cost in full universe runs.

**Evidence Block** (from Tasks 9.0.2 and 9.0.3):

| Run  | Scenario   | Sharpe | MaxDD  | Turnover | Cost  | CAGR   |
| ---- | ---------- | ------ | ------ | -------- | ----- | ------ |
| Q300 | S1 (10/5)  | 0.670  | -45.2% | 43.7     | 17.0% | 13.90% |
| Q400 | S1 (10/5)  | 0.670  | -44.0% | 41.8     | 15.8% | 13.28% |
| Q300 | S3 (15/10) | 0.651  | -45.5% | 43.7     | 27.6% | 13.50% |
| Q400 | S3 (15/10) | 0.651  | -44.3% | 41.8     | 25.7% | 12.90% |

**Forward Rules**:

- All subsequent model (MLP/Phase 8 utilities) evaluations **must use Q400 as the execution layer baseline** unless explicitly stated otherwise.
- Any change to `rebalance`/`method`/`top_k` must re-run 9.0.2-style sweep to justify the deviation.

**CLI Defaults** (updated in Task 9.0.4):

The CLI `src/run_scores_backtest_cli.py` now defaults to:

- `--method topk` (not softmax — softmax failed in full-universe tests)
- `--rebalance Q` (not M — lower transaction cost)
- `--top_k 400` (locked baseline from 9.0.2 sweep)
- `--cost_bps 10`, `--slippage_bps 5`

> [!NOTE]
> Tradability depends on account size; smaller accounts may need lower K (e.g., K=50 for <$100K).

---

### F.5.10 Task 9.2.0 — Backtest Timing Semantics Lock

**Status**: ✅ Complete

**Files**: `src/backtest_harness.py`, `tests/test_backtest_harness.py`

**Summary**: Locked forward return semantics. Weights effective on day t earn return from t to t+1 (not t-1 to t). Prevents look-ahead bias.

**Implementation**:

- Changed pct_change to `shift(-1)` for forward returns
- Last day return forced to 0.0 (no t+1 available)
- 3 tests lock the semantics

---

### F.5.11 Task 9.2.1 — Backtest NaN Immunity

**Status**: ✅ Complete

**Files**: `src/backtest_harness.py`, `tests/test_backtest_harness.py`

**Summary**: Unused ticker NaN cannot propagate to portfolio returns. Dot-product computed only over used_tickers.

**Implementation**:

- `pct_change(fill_method=None)` to avoid FutureWarning
- Used-ticker-only dot product
- Test verifies no NaN/inf in equity_curve or daily_returns

---

### F.5.12 Task 9.2.3 — Dead Code Cleanup

**Status**: ✅ Complete

**Files**: `src/backtest_harness.py`

**Summary**: Removed unused variables without changing semantics.

**Removed**:

- `portfolio_returns` (never used)
- `next_rebal` (never used)
- `rebalance_dates_set` (never used)
- `rebalance_date_to_row` (never used)

---

### F.5.13 Task 9.2.4 — Metrics Contract: cagr_over_vol

**Status**: ✅ Complete

**Files**: `src/backtest_harness.py`, `src/run_scores_backtest_cli.py`, tests

**Summary**: Renamed misleading "sharpe" metric to `cagr_over_vol` with backward compatibility.

**Metrics Contract**:

| Metric          | Definition                   | Status        |
| --------------- | ---------------------------- | ------------- |
| `cagr_over_vol` | CAGR / annualized_volatility | ✅ Primary    |
| `sharpe`        | Same as cagr_over_vol        | ⚠️ DEPRECATED |

> [!WARNING]
> The `sharpe` key is **NOT** standard Sharpe ratio (which uses excess return over risk-free rate).
> It is simply CAGR / AnnVol. Use `cagr_over_vol` for clarity.

**Output Contract**:

- `result["metrics"]["cagr_over_vol"]` — canonical metric
- `result["metrics"]["sharpe"]` — deprecated alias (same value)
- `result["warnings"]` — contains deprecation notice

---

### F.5.14 Task 9.3.0 — MLP Shadow Scoring + Artifact Export

**Status**: ✅ Complete

**Files**: `src/real_data_train_eval.py`, `tests/test_real_data_train_eval.py`

**Purpose**: Train Phase-4 MLP and export shadow scores CSV for A/B testing preparation. NO trading impact — default CLI behavior unchanged.

**Public API**:

```python
def run_shadow_scoring_mlp(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    sector_to_tickers: dict[str, list[str]] | None = None,
    seed: int = 42,
    rankgauss: bool = True,
    epochs: int = 10,
    batch_size: int = 32,
) -> pd.DataFrame:
```

**Invariants**:

| Rule           | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| PIT/No-Leakage | RankGauss fitted on TRAIN only; val/test transform-only              |
| No Shuffle     | `model.fit(..., shuffle=False)` enforced by `SectorPredictorMLP`     |
| Determinism    | Python `random.seed()` + `np.random.seed()` + `tf.random.set_seed()` |
| Train-Only Fit | Preprocessors never see val/test during fit                          |

**Output Artifact Contract**:

| Property              | Value                                            |
| --------------------- | ------------------------------------------------ |
| CSV path              | Caller-provided (`output_csv_path`)              |
| Index column          | `date` (DatetimeIndex)                           |
| Index properties      | Monotonic increasing, unique                     |
| Values                | Finite (no NaN/inf)                              |
| Columns (sector mode) | `["S0", "S1", ..., "S9"]`                        |
| Columns (ticker mode) | Sorted tickers (if `sector_to_tickers` provided) |

**Test Lock** (in `tests/test_real_data_train_eval.py`):

- `test_shadow_scoring_creates_csv` — CSV exists, valid index, finite values
- `test_shadow_scoring_determinism` — Same seed → byte-identical CSV
- `test_shadow_scoring_with_sector_to_ticker_broadcast` — Same sector → same score
- `test_shadow_scoring_eval_dates_cover_val_and_test` — Output covers val+test dates

---

### F.5.15 Task 9.3.1 — Wire Shadow MLP Export into E2E Runner

**Status**: ✅ Complete

**Files**: `src/real_data_end_to_end.py`, `tests/test_real_data_end_to_end.py`

**Purpose**: Add optional wiring of shadow MLP CSV export into the real-data E2E runner. NO change to default behavior — shadow export only occurs when explicitly enabled.

**Public API** (new optional parameters on `run_real_data_end_to_end_baseline_mlp`):

```python
# OPTIONAL: Shadow MLP export (no trading impact)
shadow_mlp_output_csv_path: str | None = None,  # Path to export CSV
shadow_mlp_epochs: int = 10,                     # Epochs for shadow training
shadow_mlp_sector_to_tickers: dict[str, list[str]] | None = None,  # Ticker broadcast
```

**Invariants**:

| Rule                   | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| Default unchanged      | If `shadow_mlp_output_csv_path` is None, no shadow export occurs |
| PIT discipline         | Delegated to `run_shadow_scoring_mlp` (train-only fit)           |
| Determinism            | Same seed → identical CSV output                                 |
| CLI defaults unchanged | No impact on production backtest path                            |

**Output Contract**:

| Key                       | Value                                    |
| ------------------------- | ---------------------------------------- |
| `shadow_mlp_csv_exported` | Path string if enabled, `None` otherwise |

**Test Lock** (in `tests/test_real_data_end_to_end.py`):

- `test_default_does_not_create_shadow_csv` — Default behavior unchanged
- `test_optional_shadow_export_creates_csv` — CSV created when path provided
- `test_shadow_export_determinism_at_e2e_boundary` — Same seed = byte-identical CSV

---

### F.5.16 Task 9.3.2 — A/B Backtest Harness

**Status**: ✅ Complete

**Files**: `src/ab_backtest.py`, `tests/test_ab_backtest.py`

**Purpose**: Compare baseline vs variant score panels via identical backtest pipeline, producing summary metrics and deltas.

**Public API**:

```python
def run_ab_backtest_from_score_csvs(
    *,
    prices_csv_path: str,
    baseline_scores_csv_path: str,
    variant_scores_csv_path: str,
    output_dir: str,
    rebalance: str = "Q",
    method: str = "topk",
    top_k: int = 400,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    max_weight: float | None = None,
    seed: int = 42,
) -> dict:
```

**Input Contract**:

| Requirement  | Description                                                |
| ------------ | ---------------------------------------------------------- |
| Date column  | Both CSVs must have `date` column                          |
| Index        | Monotonic increasing, unique                               |
| Values       | Finite (no NaN/inf)                                        |
| Intersection | Strict intersection of dates + tickers; fail-fast if empty |

**Output Contract**:

| Key                | Value                                                                                                                |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `baseline_metrics` | dict (includes `cagr_over_vol`)                                                                                      |
| `variant_metrics`  | dict (includes `cagr_over_vol`)                                                                                      |
| `delta`            | dict: `cagr_over_vol`, `cagr`, `ann_vol`, `max_drawdown`, `total_turnover`, `total_cost`, `cagr_over_vol_pct_change` |
| `dates_used`       | list of date strings                                                                                                 |
| `tickers_used`     | list of ticker strings                                                                                               |

**Artifacts Written**:

- `baseline_summary.json`
- `variant_summary.json`
- `delta_summary.json`

**Invariants**:

| Rule          | Description                                              |
| ------------- | -------------------------------------------------------- |
| Determinism   | Same inputs + seed → byte-identical `delta_summary.json` |
| CLI unchanged | No default CLI changes; baseline = Q/topk/400/10/5       |

**Test Lock** (in `tests/test_ab_backtest.py`):

- `test_ab_backtest_creates_artifacts` — JSON files exist + parseable
- `test_ab_backtest_determinism` — Same seed = identical output
- `test_ab_backtest_mismatch_dates` — Clear ValueError on no common dates
- `test_ab_backtest_mismatch_tickers` — Clear ValueError on no common tickers
- `test_ab_backtest_delta_sign` — Variant outperforms baseline when expected

---

### F.5.17 Task 9.3.2.1 — A/B Harness Hardening

**Status**: ✅ Complete

**Files**: `src/ab_backtest.py`, `tests/test_ab_backtest.py`

**Purpose**: Harden temp CSV serialization and cleanup.

**CSV Serialization Contract** (`_write_scores_panel_csv`):

| Property     | Value                                    |
| ------------ | ---------------------------------------- |
| First column | Named exactly `date`                     |
| Index        | Written with `index=False`               |
| Order        | Dates in deterministic (monotonic) order |

**Cleanup Contract**:

- Temp files cleaned up via `try/finally` even on backtest failure
- `_cleanup_temp_file()` ignores errors during removal

**Test Lock** (in `tests/test_ab_backtest.py`):

- `test_write_scores_panel_csv_has_date_column` — First line starts with "date,"
- `test_temp_cleanup_on_exception` — Temp files cleaned up on exception

---

### F.5.18 Task 9.4.0 — Ticker→Sector Mapping ETL

**Status**: ✅ Complete

**Files**: `src/ticker_sector_mapping.py`, `tests/test_ticker_sector_mapping.py`

**Purpose**: Build deterministic ticker→sector mapping from local SEC companyfacts JSON files to broadcast sector-level MLP scores (S0–S9) to ticker-level score panels for A/B backtesting.

**Public APIs**:

```python
def sic_to_sector_name(sic: int | None) -> str:
    """Map SIC code to sector name (e.g., 'Energy', 'Financials'). Returns '' if unknown."""

def build_ticker_to_sector_csv(
    *,
    companyfacts_dir: str,
    universe_tickers: list[str],
    output_csv_path: str,
    sector_name_to_id: dict[str, str],
) -> pd.DataFrame:
    """Build ticker-to-sector mapping CSV from SEC companyfacts."""

def build_sector_to_tickers(mapping_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {sector_id: [sorted tickers]}, excluding empty sector_id."""
```

**Determinism Invariants**:

| Rule               | Description                             |
| ------------------ | --------------------------------------- |
| Directory listing  | `os.listdir()` sorted before processing |
| Output rows        | Sorted by ticker ascending              |
| Duplicate handling | Smallest source (lexicographic) wins    |
| Reproducibility    | Same inputs → byte-identical CSV        |

**Fail-Safe Behavior**:

| Condition            | Behavior                        |
| -------------------- | ------------------------------- |
| Invalid JSON file    | Skipped with `logging.warning`  |
| Unreadable directory | Warns and returns empty mapping |

**Output Contract** (CSV columns):

| Column        | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| `ticker`      | Uppercase ticker symbol                                        |
| `sector_id`   | e.g., "S0"–"S9"; empty if sector_name not in sector_name_to_id |
| `sector_name` | e.g., "Energy", "Financials"; empty if SIC unknown             |
| `sic`         | SIC code from SEC; may be empty                                |
| `source`      | Filename used for tie-breaking                                 |

**Important Notes**:

> [!NOTE] > `sic_to_sector_name` may produce sector names outside the project's 10-sector S0–S9 set (e.g., "Real Estate"). The caller's `sector_name_to_id` dict must handle this by either omitting or collapsing such sectors.

**Test Lock** (in `tests/test_ticker_sector_mapping.py`):

- `test_energy_sic` — Energy SIC codes mapped correctly
- `test_financials_sic` — Financials SIC codes mapped correctly
- `test_technology_sic` — IT SIC codes mapped correctly
- `test_unknown_sic` — Unknown SIC returns empty string
- `test_build_ticker_to_sector_csv_basic_and_deterministic` — Basic ETL + determinism verified
- `test_duplicate_ticker_uses_smallest_source` — Tie-break by smallest source
- `test_build_sector_to_tickers_sorts_and_filters` — Dict sorted + empty excluded
- `test_empty_dataframe` — Empty input handled gracefully
- `test_bad_json_is_skipped_not_crash` — Invalid JSON skipped

---

### F.5.19 Task 9.4.2 — Universe Market Cap Gate (PIT-Safe)

**Status**: ✅ Complete

**Files**: `scripts/build_market_cap_universe.py`

**Purpose**: Reduce microcap / data-artifact concentration by applying a **PIT-safe market-cap eligibility gate**
before portfolio construction (without changing the backtest harness or weight adapter contracts).

**Inputs**:

- `manifest_csv`: `universe_sec_manifest.csv` (from `scripts/download_sec_data.py preprocess` / `filter_universe`)
  - Must include `ticker`, `cik_status`, `companyfacts_status`, `companyfacts_path`
- `prices_csv`: wide daily prices CSV with `date` column
- `scores_csv`: wide score panel CSV with `date` column (signal dates)

**Market Cap Definition (per date t, per ticker i)**:

- `price(t, i)` = last available close **<= t** (daily prices forward-filled to the score date)
- `shares(t, i)` from SEC companyfacts **PIT rule**:
  - Only rows with `filed <= t` are visible at date t
  - Choose the latest row by `(end, filed)` with `end <= t` and `filed <= t`
  - Shares tag priority:
    - `dei/EntityCommonStockSharesOutstanding`
    - fallback: `us-gaap/CommonStockSharesOutstanding`
- `market_cap(t, i) = shares(t, i) * price(t, i)`

**Eligibility Gate**:

- Eligible if `market_cap(t, i) >= min_market_cap_usd`
- Optional additional gate: `price(t, i) >= min_price`
- If shares cannot be computed at date t, ticker is **ineligible** at t.

**Score Gating Mechanism (per date t)**:

- For ineligible tickers, replace score with a very low **finite** value:
  - `score_gated(t, i) = min(score_row(t)) - ineligible_penalty`
- This preserves:
  - No NaN/inf in scores (keeps `scores_to_target_weights` fail-fast rules intact)
  - Deterministic tie-breaking behavior

**Safety Check (required for TopK correctness)**:

- Enforce `eligible_count(t) >= top_k` for every date t, otherwise fail-fast with a clear message.
  - Prevents accidental selection of ineligible tickers when `top_k` exceeds eligible universe size.

**Outputs** (written to `out_dir`):

- `scores.csv` (gated scores, same shape as input after intersection)
- `summary.json` (parameters + eligible_count stats + missing shares count)
- Optional:
  - `prices.csv` (copied for convenience)
  - `market_cap.csv` (wide panel aligned to score dates)
  - `eligibility.csv` (0/1 mask aligned to score dates)
  - `missing_shares_tickers.txt`

**Determinism / Safety**:

- No network calls, no system clock usage.
- Same inputs → byte-identical outputs (dates ascending; tickers sorted).
- PIT-safe by construction (`filed <= t` enforced for shares).

---

### F.5.20 Task 9.4.3 — Walk-Forward Ridge Alpha Baseline (PIT-Safe)

**Status**: ✅ Complete (Research Baseline)

**Files**: `scripts/train_ridge_alpha_walkforward.py`

**Purpose**: Build a **PIT-safe ML baseline** (walk-forward Ridge) on top of the **market-cap gated universe**, and export a backtest-ready wide `scores.csv` without modifying the backtest harness/adapter contracts.

**Inputs**:

- `manifest_csv`: `universe_sec_manifest.csv` (from `scripts/download_sec_data.py preprocess/filter_universe`)
  - Must include `ticker`, `cik_status`, `companyfacts_status`, `companyfacts_path`
- `prices_csv`: wide daily prices CSV with `date` column
- `market_cap_csv`: wide market-cap panel aligned to score dates (from `scripts/build_market_cap_universe.py`)
- `eligibility_csv`: 0/1 mask aligned to score dates (from `scripts/build_market_cap_universe.py`)
- Optional:
  - `insider_events_csv`: long (date,ticker,...) Form345 panel from `scripts/build_form345_insider_features.py`
  - `ticker_to_sector_csv`: ticker→sector mapping for sector neutralization (see § F.5.18)

**Feature Panel (per score date t, per ticker i)**:

- Price features (monthly on score dates):
  - `mom_1m`: `price(t)/price(t-1) - 1`
  - `mom_3m`, `mom_6m`, `mom_12m`
  - `vol_3m`: rolling std of `mom_1m` (3 periods)
- PIT fundamentals (SEC companyfacts):
  - Snapshot rule at date t: select the latest row by `(end, filed)` among rows with `end <= t` and `filed <= t`
  - Tags used (USD): `Assets`, `Liabilities`, `StockholdersEquity`, `CashAndCashEquivalentsAtCarryingValue` (fallback cash tag supported)
  - Derived ratios:
    - `log_mktcap = log(max(market_cap, eps))`
    - `book_to_mkt = equity / market_cap`
    - `leverage = liabilities / assets`
    - `cash_to_assets = cash / assets`
    - `asset_growth_12m = assets / assets(t-12) - 1`
    - `shares_chg_12m = (market_cap/price) / (market_cap/price)(t-12) - 1`

**Label (training only)**:

- Forward return over `h` months (monthly score dates): `y_raw(t, i) = price(t+h, i)/price(t, i) - 1`
  - `h` is controlled by `--label_horizon` (e.g., `3` for quarterly-aligned training)
- Optional label post-processing (per-date, eligible-only):
  - Sector de-meaning (`--sector_neutralize_labels`, requires `ticker_to_sector_csv`)
  - Cross-sectional transform (`--label_transform`): `raw | rank | winsorize_zscore`

**Walk-Forward Training / Scoring**:

- Expanding window: at each date index `i`, train on all dates `< i` (never uses future dates).
- Train-only transforms:
  - `StandardScaler` fit on training samples only.
- Model:
  - `sklearn.linear_model.Ridge(alpha=ridge_alpha)`
- Eligibility + safety:
  - Only eligible tickers participate in training/prediction (`eligibility_csv`).
  - Enforce `eligible_count(t) >= top_k` for every predicted date (fail-fast).
- Optional score post-processing:
  - Sector de-meaning of predicted scores (`--sector_neutralize_scores`, requires `ticker_to_sector_csv`)
- Score gating (per date t):
  - Ineligible / missing-feature tickers get a very low finite value: `row_min - ineligible_penalty` to preserve adapter contracts (no NaN/inf).

**Outputs** (written to `out_dir`):

- `scores.csv` (wide panel: `date` + tickers; finite values only)
- `ic_by_date.csv` (OOS per-date Spearman IC vs raw return and vs training label)
- `summary.json` (params + feature list + OOS Spearman IC summary)

**Determinism / Safety**:

- No randomness; stable ticker/date ordering.
- Same inputs → byte-identical `scores.csv`/`summary.json`.
- PIT-safe fundamentals by construction (`filed <= t` rule enforced).

---

### F.5.21 Task 9.4.4 — Form345 Insider Feature Panel (PIT-Safe)

**Status**: ✅ Complete (Research ETL)

**Files**: `scripts/build_form345_insider_features.py`

**Purpose**: Convert downloaded SEC Form 345 structured data zips into a **PIT-safe insider activity feature panel** aligned to the project's monthly score dates.

**Inputs**:

- `insiders_dir`: `data/raw/insiders` (zip files containing `SUBMISSION.tsv` + `NONDERIV_TRANS.tsv`)
- `universe_prices_csv`: used for universe ticker list (CSV header)
- `score_dates_csv`: used for the monthly score-date calendar (date column)

**PIT rule (filing-date bucketing)**:

- Each transaction becomes usable at the **first score date >= `FILING_DATE`** (from `SUBMISSION.tsv`).
- Transactions are filtered to `FILING_DATE` within the score-date range.
- By default, includes only non-derivative open-market codes: `TRANS_CODE in {P, S}`.

**Outputs**:

- `out_csv`: long-format CSV with columns:
  - `date`, `ticker`
  - `buy_value`, `sell_value`, `net_value`
  - `buy_shares`, `sell_shares`, `net_shares`
  - `buy_count`, `sell_count`
- `*.summary.json`: parse/coverage summary

**Determinism / Safety**:

- Zip processing order is sorted; grouping uses deterministic sorts.
- No network calls, no system clock usage.
- Same inputs → byte-identical outputs.

---

## F.6) Task 8.x Contracts — Phase 5 Upgrades (Utilities)

### F.6.1 Task 8.1.0 Contract — Split Conformal Prediction Intervals

**Files:**

- `src/conformal.py`
- `tests/test_conformal.py`

**Public API:**

```python
class SplitConformalRegressor:
    def fit(self, y_true, y_pred) -> self
    def quantile(self, *, alpha: float) -> np.ndarray  # shape (k,)
    def predict_interval(self, y_pred, *, alpha: float) -> (lower, upper)
```

**Non-Negotiable Semantics:**

> [!CAUTION]
> Calibration must be performed on a calibration/validation split, **NEVER on test data**.

- Deterministic (no randomness).
- Fail-fast on: NaN/inf, shape mismatch, alpha not in (0,1), not-fitted usage.
- Quantile computation: `k_index = ceil((n+1)*(1-alpha))`, clamped to [1,n], uses `np.partition`.

**Output Contract:**

- `predict_interval(y_pred)` returns `(lower, upper)` preserving input type (DataFrame/Series/numpy).
- Multi-output: arrays of shape (n, k).

---

### F.6.2 Task 8.2.0 Contract — Temperature Scaling Calibration

**Files:**

- `src/temperature_scaling.py`
- `tests/test_temperature_scaling.py`

**Public API:**

```python
class TemperatureScalerBinary:
    def __init__(self, *, grid_size=400, t_min=0.05, t_max=10.0, eps=1e-12)
    def fit(self, scores, y_true, *, input_type="logits") -> self
    def transform(self, scores, *, input_type="logits") -> proba
    def predict_proba(self, scores, *, input_type="logits") -> proba  # alias
```

**Non-Negotiable Semantics:**

> [!CAUTION]
> Fit temperature on calibration/validation split only, **NEVER on test data**.

- Deterministic grid search over log-spaced temperature range.
- Fail-fast on: labels not binary, NaN/inf, proba out of [0,1], invalid `input_type`, not-fitted usage.
- Supported `input_type`: `"logits"` or `"proba"`.

**Output Contract:**

- `transform()` returns calibrated probabilities in [0,1].
- Preserves input type (DataFrame/Series/numpy).

---

### F.6.3 Task 8.3.0 Contract — SHAP-Style Explainability

**Files:**

- `src/shap_explainability.py`
- `tests/test_shap_explainability.py`

**Public API:**

```python
def shapley_sampling_values(
    predict_fn, x, *,
    baseline=None, background=None,
    n_permutations=128, seed=0
) -> (n,d) or (n,d,k)

def global_feature_importance(shap_values, *, agg="mean_abs") -> (d,) or (d,k)
```

**Non-Negotiable Semantics:**

- Deterministic given seed (uses `np.random.default_rng(seed)`).
- Baseline selection priority:
  1. `baseline` if provided
  2. `mean(background)` if `background` provided
  3. zeros
- Fail-fast on: NaN/inf, invalid shapes, invalid `agg`, `predict_fn` output mismatch.

**Output Contract:**

- Single-output: returns `(n, d)` DataFrame if input is DataFrame, else numpy.
- Multi-output: always returns numpy `(n, d, k)`.
- `global_feature_importance` returns `(d,)` or `(d, k)` matching dimensionality.

---

## G) Current Status & Next Steps

### G.1 Current Status

| Task    | Status       | Notes                                         |
| ------- | ------------ | --------------------------------------------- |
| 7.0     | ✅ Completed | Stooq/SEC/H1H2 integration complete           |
| 7.1     | ✅ Completed | Health gates implemented                      |
| 7.2     | ✅ Completed | Train/eval pipeline with RankGauss            |
| 7.2.4   | ✅ Completed | Config alias + explicit metadata keys         |
| 7.2.5.x | ✅ Completed | Sector counts diagnostics & health gates      |
| 7.2.6   | ✅ Completed | Predictions → target_weights adapter          |
| 7.2.7   | ✅ Completed | E2E wiring (scores → weights → backtest)      |
| 7.3     | ✅ Completed | Black-Litterman with never-crash policy       |
| 7.4.1   | ✅ Completed | HP filter leakage guard                       |
| 7.4.2   | ✅ Completed | SSOT Documentation Lock (this document)       |
| 7.4.3   | ✅ Completed | Regime logistic reproducibility (solver/seed) |
| 7.5.x   | ✅ Completed | Backtest harness (M/Q rebalance, fail-fast)   |
| 8.1.0   | ✅ Completed | Split Conformal Prediction Intervals          |
| 8.2.0   | ✅ Completed | Temperature Scaling Calibration               |
| 8.3.0   | ✅ Completed | SHAP-Style Explainability                     |

### G.2 Next Steps — Phase 8.x (Remaining)

1. **8.4**: KAN architecture (optional, only if results justify)

### G.3 vNext Candidates (Not in Phase 8.x Scope)

The following are **NOT** part of Phase 8.x; only consider after Phase 8.x is complete and results justify expansion:

- FT-Transformer
- TabResNet
- Other advanced transformer architectures

### G.4 Decision Lock: Sector-Fundamental MLP Alpha Track (Task 9.3.10)

> [!CAUTION] > **DECISION**: Fundamental H1/H2 Sector MLP is **NOT ADOPTED** as alpha replacement or blend.

#### Evidence (Task 9.3.7 — Sector Autopsy, W1: 2023-01→2024-12)

| Metric                        | Baseline (Momentum) | MLP   | Conclusion    |
| ----------------------------- | ------------------- | ----- | ------------- |
| hit_rate_top1                 | **0.32**            | 0.00  | Baseline wins |
| hit_rate_top2                 | **0.08**            | 0.00  | Baseline wins |
| IC_spearman_mean              | **0.069**           | 0.041 | Baseline wins |
| IC_pearson_mean               | **0.060**           | 0.009 | Baseline wins |
| tie_stats (n_unique_mean)     | 9.0                 | 9.0   | Identical     |
| tie_stats (max_tie_group_max) | 410                 | 410   | Identical     |
| tie_stats (max_tie_frac_max)  | 0.251               | 0.251 | Identical     |

**Conclusion**: Predictive failure, NOT tie-breaking mechanics.

#### Evidence (Task 9.3.9 — Rolling OOS Sweep)

| Window | as_of_date | train_end  | val_end    | Baseline IC | Baseline hit@1 | MLP IC | MLP hit@1 |
| ------ | ---------- | ---------- | ---------- | ----------- | -------------- | ------ | --------- |
| W1     | 2024-12-31 | 2021-12-31 | 2022-12-31 | **0.069**   | **32%**        | 0.041  | 0%        |
| W2     | 2022-12-31 | 2019-12-31 | 2020-12-31 | **0.215**   | **36%**        | -0.044 | 4%        |
| W3     | 2020-12-31 | 2017-12-31 | 2018-12-31 | —           | —              | —      | —         |

> W3 failed due to `GATE_MISSINGNESS` (insufficient SEC data coverage for 2017-2018 training period).

#### Decision (Explicit)

1. **Fundamental H1/H2 sector MLP is NOT adopted** as an alpha replacement or blend in the current system.
2. This conclusion is **robust across multiple OOS windows**; NOT explained by short training length.
3. MLP shows **negative IC** in W2 (-0.044), indicating worse-than-random sector ranking.

#### Infrastructure Retained

The following validated infrastructure is retained for future comparisons:

- `src/sector_model_diagnostics.py` — IC/hit-rate/tie analysis
- `src/ab_backtest.py` — A/B harness for score panel comparison
- Shadow MLP export path — for future model experiments

---

## G.5 Task 9.5.x Contracts — Shadow Risk Exposure/Gating (NO Trading Impact)

> [!IMPORTANT] > **Scope**: Shadow-only diagnostics/artifacts; **NO trading impact** (no change to selection/weights/backtest outcomes unless optional export paths are explicitly enabled).

### G.5.0 Task 9.5.0 — Base Shadow Risk Exposure

**Files:**

- `src/shadow_risk_exposure.py`
- `tests/test_shadow_risk_exposure.py`

**Public API:**

```python
def run_shadow_risk_exposure_logit(
    prices: pd.DataFrame,
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    spy_ticker: str = "SPY",
    horizon_days: int = 63,
    seed: int = 42,
) -> pd.DataFrame
```

**Output CSV Columns:**

- `p_risk_off` — probability of risk-off regime
- `w_beta_suggested` — suggested beta tilt (0.0–0.35)
- `exposure_suggested` — suggested exposure level (0.0–1.0)
- `ret_21d, ret_63d, mom_252d, vol_63d, dd_126d` — PIT-safe features

**Non-Negotiable Semantics:**

- **PIT-Safe**: Features at date t use only information ≤ t
- **Train-Only Fit**: StandardScaler fitted on TRAIN split only
- **Deterministic**: Same inputs + same seed → byte-identical CSV
- **Fail-Safe**: Never crash; writes fallback CSV with safe defaults (p=0.5)

### G.5.1 Task 9.5.1 — Shadow Risk Hardening

**Corrections Applied:**

- **Label NaN Handling**: Rows with missing forward returns produce NaN labels (correctly excluded)
- **Fallback Coverage**: Fallback CSV covers VAL+TEST dates `(train_end, as_of_date]`
- **Train-Only Verification**: Tests use monkeypatch to verify scaler fit on TRAIN only
- **Horizon Exclusion Test**: Validates last `horizon_days` excluded from output

### G.5.2 Task 9.5.2 — Metrics JSON Export

**Public API:**

```python
def run_shadow_risk_exposure_logit_with_metrics(
    prices: pd.DataFrame,
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    output_csv_path: str,
    output_metrics_json_path: str,
    spy_ticker: str = "SPY",
    horizon_days: int = 63,
    seed: int = 42,
) -> Dict[str, Any]
```

**JSON Schema (version: "9.5.2"):**

```json
{
  "schema_version": "9.5.2",
  "config": { "as_of_date", "train_end", "val_end", "spy_ticker", "horizon_days", "seed" },
  "train": { "n_obs", "base_rate", "brier", "roc_auc", "log_loss", "calibration_bins", "ece" },
  "val": { ... },
  "test": { ... }
}
```

**Metrics Per Split:**

- `n_obs` — observation count
- `base_rate` — mean of risk_off labels
- `brier` — Brier score loss
- `roc_auc` — ROC AUC (null if single-class)
- `log_loss` — logarithmic loss (null if single-class)
- `calibration_bins` — 10 bins with count, mean_pred, frac_pos
- `ece` — Expected Calibration Error

### G.5.3 Task 9.5.3 — Overlay Backtest + CSV Identity Lock

**Public API:**

```python
def run_shadow_risk_overlay_spy_only(
    prices: pd.DataFrame,
    *,
    as_of_date: str,
    train_end: str,
    val_end: str,
    shadow_csv_path: str,
    output_overlay_csv_path: str,
    output_overlay_metrics_json_path: str,
    spy_ticker: str = "SPY",
    cash_daily_return: float = 0.0,
) -> Dict[str, Any]
```

**Overlay CSV Columns:**

- `exposure_suggested` — from shadow CSV
- `spy_ret_1d` — daily SPY return
- `overlay_ret_1d` — overlay return (shifted-weight semantics)
- `overlay_equity` — cumulative equity curve

**Shifted-Weight Semantics:**

```
overlay_ret[t] = exposure[t-1] * spy_ret[t] + (1 - exposure[t-1]) * cash_ret
```

**Overlay Metrics JSON (version: "9.5.3"):**

- `n_obs`, `total_return`, `cagr`, `ann_vol`, `cagr_over_vol`, `max_drawdown`
- `avg_exposure`, `turnover_exposure`

### Regression Locks (Explicit)

> [!CAUTION]
> The following regression tests MUST pass for any changes to shadow risk code.

| Lock                     | Test                                                         |
| ------------------------ | ------------------------------------------------------------ |
| CSV Identity             | `with_metrics` produces byte-identical CSV to plain function |
| Metrics JSON Determinism | Same inputs → byte-identical JSON                            |
| Overlay Determinism      | Same inputs → byte-identical overlay CSV/JSON                |
| Shifted-Weight Semantics | `overlay_ret[t]` uses `exposure[t-1]`                        |

### E2E Integration (Optional Parameters)

Added to `run_real_data_end_to_end_baseline_mlp()`:

| Parameter                                      | Default | Description                  |
| ---------------------------------------------- | ------- | ---------------------------- |
| `shadow_risk_output_csv_path`                  | None    | Base shadow CSV path         |
| `shadow_risk_spy_ticker`                       | "SPY"   | Ticker for SPY features      |
| `shadow_risk_horizon_days`                     | 63      | Forward horizon for labels   |
| `shadow_risk_metrics_output_json_path`         | None    | Metrics JSON path            |
| `shadow_risk_overlay_output_csv_path`          | None    | Overlay time series CSV path |
| `shadow_risk_overlay_metrics_output_json_path` | None    | Overlay metrics JSON path    |

**Return Dict Keys:**

- `shadow_risk_csv_exported`
- `shadow_risk_metrics_json_exported`
- `shadow_risk_overlay_csv_exported`
- `shadow_risk_overlay_metrics_json_exported`

### Fail-Safe Contract

> [!NOTE]
> Shadow risk functions NEVER crash. On error, they log stable warning prefixes and write valid artifacts.

| Condition           | Warning Prefix          | Fallback Behavior               |
| ------------------- | ----------------------- | ------------------------------- |
| Missing SPY         | `SHADOW_RISK_ML:*`      | CSV with p=0.5, JSON with nulls |
| Insufficient data   | `SHADOW_RISK_ML:*`      | CSV with p=0.5, JSON with nulls |
| Single-class labels | `SHADOW_RISK_METRICS:*` | JSON with roc_auc=null          |
| Missing shadow CSV  | `SHADOW_RISK_OVERLAY:*` | Overlay CSV/JSON with defaults  |

### G.5.6 Task 9.6.6 — Shadow Risk Horizon Ablation (63d vs 21d)

**Status**: ✅ Complete (Ablation Experiment)

**Files Modified**:

- `run_shadow_risk_evaluation.py`
- `tests/test_run_shadow_risk_evaluation.py`

**Purpose**: Test whether shortening labeling horizon from 63d to 21d improves OOS calibration (ECE).

**Implementation**:

- Added `--horizon-days` CLI arg (default=63)
- Output path structure: `<output_root>/horizon_<H>/{logit,mlp}/...`
- New test: `TestHorizonWiring.test_horizon_arg_wiring_changes_outputs`

**Experiment Results (SPY-only, 2024-10-01)**:

| Metric         | H=63d | H=21d     | Delta         |
| -------------- | ----- | --------- | ------------- |
| **Logit ECE**  | 0.257 | **0.132** | **-48.8%** ✅ |
| **MLP ECE**    | 0.447 | 0.440     | -1.6%         |
| Logit cagr/vol | 3.09  | 2.38      | -23%          |
| MLP cagr/vol   | 3.52  | 2.35      | -33%          |
| Logit MaxDD    | -4.0% | -5.5%     | +1.5%p        |
| MLP MaxDD      | -3.5% | -3.7%     | +0.2%p        |

**Findings**:

- Logit model shows **significant ECE improvement** with shorter horizon
- MLP model shows negligible change (model-specific issue)
- Both models show reduced cagr/vol (tradeoff)

**Decision**: Partial success. Logit horizon=21 adoption candidate; MLP requires further investigation.

---

### G.5.12 Task 9.6.12 — Shadow Risk MLP Stability & Calibration Diagnostics (2025-03-31 / 2025-06-30)

**Status**: Complete (diagnostic + default tuning)

**Files Modified**:

- `src/shadow_risk_exposure.py`

**Artifacts Added/Updated**:

- `artifacts/shadow_risk/spy_only_eval_asof_2025_03_31/`
- `artifacts/shadow_risk/spy_only_eval_asof_2025_06_30/`
- `artifacts/shadow_risk/spy_only_eval_asof_2025_06_30_opt1/`
- `artifacts/shadow_risk/spy_only_eval_asof_2025_06_30_opt1_tol1e2/`

**Purpose**: Validate MLP vs Logit under test windows with positive labels, check calibration behavior, and reduce MLP convergence warnings with simpler/regularized defaults.

**Implementation**:

- Re-ran SPY-only evaluation with `as_of_date=2025-03-31` and `2025-06-30` to restore test positives.
- Ran temperature scaling (VAL-fit/TEST-eval) using overlay-derived labels.
- Updated MLP defaults to `(8, 4)`, `alpha=1e-2`, `max_iter=1000`, `tol=1e-2`, and recorded `tol` in metrics JSON.

**Key Results (forward-return labels)**:

- 2025-03-31 (63d): Logit AUC ~0.53, MLP AUC ~0.43; MLP worse on ECE/Brier/LogLoss.
- 2025-06-30 (63d): Logit AUC ~0.63, MLP AUC ~0.39; logit dominates calibration.
- 2025-06-30 (21d, tol=1e-2): MLP AUC ~0.52 but ECE ~0.08 vs logit ECE ~0.05; MLP still less calibrated.

**Calibration (overlay labels)**:

- 63d window: logit test ECE improves with temperature scaling; MLP test ECE worsens.
- 21d window: temperature scaling does not improve ECE for either model.

**Decision**: Adopt smaller MLP + stronger regularization + looser tol to reduce convergence warnings; keep Logit as baseline for calibrated probabilities; treat calibration gains as window-dependent (not guaranteed for 21d).

---

## Quick Reference

### Critical Guards (Copy-Paste)

```python
# PIT guard
assert data.index.max() <= as_of_date

# Ascending order
assert df.index.is_monotonic_increasing

# Train-only fit
scaler.fit(X_train)  # Only on train

# No shuffle
model.fit(X, y, shuffle=False)

# HP leakage guard
hp_params = HPParams(leakage_guard=True, lookback=120)
```

### Shape Contracts

```python
# X features
assert X.shape[1] == 20  # S0_H1..S9_H1, S0_H2..S9_H2

# Y labels
assert Y.shape[1] == 10  # S0_Y..S9_Y
```
