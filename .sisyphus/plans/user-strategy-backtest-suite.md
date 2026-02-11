# User Strategy vs E03 Backtest Comparison Suite

## TL;DR

> **Quick Summary**: Build a comprehensive backtest suite comparing the E03 Ensemble baseline strategy against a new User Strategy with Trend Score, Volatility Lock, and Overheat logic. Run 11+ experiments with sensitivity analysis.
> 
> **Deliverables**:
> - `200tq/scripts/backtest_user_strategy_suite.py` - Main runner
> - `200tq/experiments/user_strategy_comparison/` - All outputs
> - leaderboard.csv/md, equity_curves.png, drawdowns.png, yearly_returns_heatmap.png, sensitivity_table.csv, tax_by_year.csv
> 
> **Estimated Effort**: Medium (4-6 hours)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
User wants to compare their custom trading strategy against the existing E03 Ensemble baseline across 11 experiments, with proper t+1 execution, 10bps costs, and 22% Korean tax on realized gains.

### Interview Summary
**Key Discussions**:
- E03 Baseline: Use OFF = 100% SGOV (not OFF10 variant) for fair comparison
- Entry Gate: "중간 탑승 금지" = only enter when previous day score was 0 (breakthrough from below)
- Overheat thresholds for E07: 2.30/2.51/2.70 with re-entry at 2.00/2.18/2.35
- No automated tests - manual inspection via leaderboard/equity curves

**Research Findings**:
- Existing `run_suite.py` provides ExperimentConfig dataclass pattern
- `backtest_hybrid_2_0.py` has QLD synthetic generation via `build_synthetic_2x()`
- Tax model uses iterative loop with cost basis tracking
- Signal lag via `signal.shift(1)` for t+1 execution

### Strategies to Implement

**Strategy A: E03 Ensemble (Baseline)**
- Signal: QQQ SMA(3) > SMA(window) ensemble majority vote (windows: 160, 165, 170)
- Position: ON = TQQQ 100%, OFF = SGOV 100%
- Execution: t+1 close

**Strategy B: User Strategy (Trend Score + Forced Exits)**

*Trend Score (0-2):*
- QQQ MA3 > QQQ MA161 → +1 point
- TQQQ price > TQQQ MA200 → +1 point
- Score 2 → TQQQ 100%
- Score 1 → QLD 100%
- Score 0 → SGOV 100%

*Forced Exits (override score):*
- Volatility Lock: 20-day volatility >= 6.2% → SGOV forced
- Overheat Sell: TQQQ / TQQQ_MA200 >= 2.51 → SGOV + overheat_mode=True
- Re-entry: In overheat_mode, only re-enter when ratio <= 2.18

*Entry Gate:*
- New entries only allowed when previous day score was 0 (breakthrough from below)

---

## Work Objectives

### Core Objective
Implement a comprehensive backtest suite that fairly compares the E03 baseline against the User Strategy across 11+ experiments with sensitivity analysis, producing publication-ready visualizations and metrics.

### Concrete Deliverables
- `200tq/scripts/backtest_user_strategy_suite.py` (~600-800 lines)
- `200tq/experiments/user_strategy_comparison/` directory with:
  - `leaderboard.csv` and `leaderboard.md`
  - `equity_curves.png` (all experiments overlaid)
  - `drawdowns.png` (all experiments)
  - `yearly_returns_heatmap.png`
  - `sensitivity_table.csv` (vol, overheat, MA thresholds)
  - `tax_by_year.csv` (annual tax breakdown)
  - Per-experiment subdirectories with individual artifacts

### Definition of Done
- [ ] `python 200tq/scripts/backtest_user_strategy_suite.py` runs without errors
- [ ] leaderboard.md shows all experiments sorted by CAGR
- [ ] equity_curves.png displays all experiment lines
- [ ] User's claimed results (CAGR ~54%, MDD ~49%, ~177 trades) can be verified or explained

### Must Have
- t+1 close execution via `signal.shift(1)`
- 10 bps transaction cost on weight changes
- 22% Korean tax on annual realized gains
- All 11 experiments implemented
- Synthetic QLD for pre-2006 dates

### Must NOT Have (Guardrails)
- NO look-ahead bias (all signals must be lagged)
- NO hardcoded expected results (let backtest compute)
- NO skipping the OFF=100% SGOV requirement for E03
- NO web UI or real-time components
- NO database persistence

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: NO (user chose no tests)
- **Automated tests**: None
- **Framework**: N/A

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

Verification will be done by:
1. Running the script and capturing stdout
2. Checking file existence and content
3. Validating CSV/JSON structure
4. Visual inspection of generated plots (via file size > 0)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Data loading and synthetic QLD generation
└── (No parallelization - foundation task)

Wave 2 (After Wave 1):
├── Task 2: User Strategy signal generation engine
└── Task 3: E03 Baseline signal generation (can reuse existing patterns)

Wave 3 (After Wave 2):
├── Task 4: Backtest engine with cost/tax model
└── (Sequential - depends on signals)

Wave 4 (After Wave 3):
├── Task 5: Experiment runner for all 11 experiments
└── (Sequential - runs all variants)

Wave 5 (After Wave 4):
├── Task 6: Visualization and reporting
└── (Sequential - requires results)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None (foundation) |
| 2 | 1 | 4 | 3 |
| 3 | 1 | 4 | 2 |
| 4 | 2, 3 | 5 | None |
| 5 | 4 | 6 | None |
| 6 | 5 | None | None (final) |

---

## TODOs

- [ ] 1. Data Loading and Synthetic Asset Generation

  **What to do**:
  - Create data loading function using yfinance for QQQ, TQQQ, SGOV, SHV, QLD
  - Implement `build_synthetic_2x()` for QLD (2x QQQ) pre-2006
  - Implement `build_synthetic_3x()` for TQQQ pre-2010
  - Handle SGOV → SHV → CASH fallback chain
  - Calculate 20-day rolling volatility: `returns.rolling(20).std() * np.sqrt(252)`
  - Align all price series and filter to analysis period (2010-01-01 to 2025-12-31)

  **Must NOT do**:
  - Do not cache to database
  - Do not use hardcoded file paths outside the project

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Data loading is well-defined with clear patterns from existing codebase
  - **Skills**: `[]`
    - No special skills needed - standard Python/pandas work

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (foundation)
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None

  **References**:
  - `200tq/scripts/run_suite.py:208-251` - `load_data()` function pattern
  - `200tq/scripts/backtest_hybrid_2_0.py:74-116` - Multi-ticker loading with QLD
  - `200tq/scripts/backtest_hybrid_2_0.py:118-131` - `build_synthetic_2x()` and `build_synthetic_3x()` patterns
  - `200tq/scripts/backtest_hybrid_2_0.py:134-211` - `prepare_prices()` with synthetic splicing

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Data loading returns aligned DataFrame
    Tool: Bash (python)
    Preconditions: Script file created
    Steps:
      1. Run: python -c "from backtest_user_strategy_suite import load_data; df = load_data(); print(df.shape, df.columns.tolist())"
      2. Assert: Output contains QQQ, TQQQ, QLD, SGOV
      3. Assert: Shape shows > 3000 rows (2010-2025)
    Expected Result: DataFrame with all required columns
    Evidence: stdout captured

  Scenario: 20-day volatility calculation works
    Tool: Bash (python)
    Preconditions: load_data() implemented
    Steps:
      1. Run: python -c "from backtest_user_strategy_suite import load_data, calc_volatility; df = load_data(); vol = calc_volatility(df); print(vol.tail())"
      2. Assert: Output shows numeric values
      3. Assert: Values are in reasonable range (0.1 to 1.5 annualized)
    Expected Result: Series of volatility values
    Evidence: stdout captured
  ```

  **Commit**: YES (groups with 2, 3)
  - Message: `feat(backtest): add data loading with synthetic QLD/TQQQ`
  - Files: `200tq/scripts/backtest_user_strategy_suite.py`
  - Pre-commit: `python -c "from 200tq.scripts.backtest_user_strategy_suite import load_data"`

---

- [ ] 2. User Strategy Signal Generation Engine

  **What to do**:
  - Implement `UserStrategyConfig` dataclass with all configurable parameters:
    - `ma_short`: int = 3
    - `ma_long_qqq`: int = 161
    - `ma_long_tqqq`: int = 200
    - `vol_threshold`: float = 0.062 (6.2%)
    - `overheat_ratio`: float = 2.51
    - `reentry_ratio`: float = 2.18
    - `use_vol_lock`: bool = True
    - `use_overheat`: bool = True
    - `use_qld_step`: bool = True
    - `use_entry_gate`: bool = True (strict vs relaxed)
  
  - Implement `calculate_trend_score()`:
    - Score component 1: QQQ MA3 > QQQ MA161 → +1
    - Score component 2: TQQQ > TQQQ MA200 → +1
    - Return score 0/1/2

  - Implement `apply_forced_exits()`:
    - Volatility Lock: If 20-day vol >= threshold → force score to 0
    - Overheat: If TQQQ/TQQQ_MA200 >= overheat_ratio → force to 0, set overheat_mode=True
    - Re-entry: While overheat_mode=True, only re-enter when ratio <= reentry_ratio

  - Implement `apply_entry_gate()`:
    - If use_entry_gate=True (strict): Only allow position when prev_score == 0
    - Track state machine: in_position, prev_score
    - Once in position, maintain until score drops to 0

  - Implement `generate_user_signal()`:
    - Combine all components
    - Return target allocation: TQQQ weight, QLD weight, SGOV weight

  **Must NOT do**:
  - Do not apply t+1 lag here (done in backtest engine)
  - Do not calculate returns or costs
  - Do not hardcode specific results

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: Complex state machine logic with multiple interacting conditions
  - **Skills**: `[]`
    - Standard Python - no special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:
  - `200tq/scripts/run_suite.py:257-348` - Signal generation patterns (plain, ensemble, hysteresis)
  - `200tq/scripts/run_suite.py:303-334` - `apply_confirmation()` state machine pattern
  - `200tq/scripts/backtest_hybrid_2_0.py:286-389` - `CoreState` dataclass and iterative state tracking
  - `200tq/scripts/backtest_hybrid_2_0.py:395-548` - `SatelliteState` with stage tracking (similar to overheat_mode)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Trend score calculation returns 0/1/2
    Tool: Bash (python)
    Preconditions: Signal generation implemented
    Steps:
      1. Run: python -c "from backtest_user_strategy_suite import load_data, calculate_trend_score; df = load_data(); score = calculate_trend_score(df); print(score.value_counts())"
      2. Assert: Output shows counts for values 0, 1, 2 only
      3. Assert: All three values present
    Expected Result: Distribution of scores across trading days
    Evidence: stdout captured

  Scenario: Volatility lock forces score to 0 during high vol
    Tool: Bash (python)
    Preconditions: Volatility lock implemented
    Steps:
      1. Create test with known high-vol period (e.g., March 2020)
      2. Assert: Score forced to 0 when vol >= 6.2%
      3. Assert: Without vol lock, score would be higher
    Expected Result: Vol lock correctly overrides trend score
    Evidence: Comparison output captured

  Scenario: Entry gate blocks mid-trend entries
    Tool: Bash (python)
    Preconditions: Entry gate implemented with test data
    Steps:
      1. Create scenario where score goes 0→1→2→1→2
      2. With gate: Position entered at 0→1, maintained through 1→2, closed at some point
      3. With gate: When score drops to 1 then rises to 2, should NOT re-enter
      4. Assert: Entry gate blocks the 1→2 re-entry
    Expected Result: Only breakthrough from 0 allows entry
    Evidence: State trace captured
  ```

  **Commit**: YES (groups with 1, 3)
  - Message: `feat(backtest): implement User Strategy signal engine with trend score, vol lock, overheat`
  - Files: `200tq/scripts/backtest_user_strategy_suite.py`
  - Pre-commit: `python -c "from 200tq.scripts.backtest_user_strategy_suite import generate_user_signal"`

---

- [ ] 3. E03 Baseline Signal Generation (100% SGOV OFF)

  **What to do**:
  - Implement `generate_e03_signal()` following existing pattern but with OFF = 100% SGOV
  - Ensemble majority vote: MA3 > MA(window) for windows [160, 165, 170]
  - Threshold: 2/3 votes required for ON
  - ON = 100% TQQQ, OFF = 100% SGOV (NOT the OFF10 variant)
  - Support configurable windows for E10 sensitivity test

  **Must NOT do**:
  - Do not use OFF = 10% TQQQ + 90% SGOV (that's the OFF10 variant)
  - Do not apply lag (done in backtest engine)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct adaptation of existing pattern with minor modification
  - **Skills**: `[]`
    - Standard Python

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:
  - `200tq/scripts/run_suite.py:265-276` - `generate_ensemble_signal()` exact pattern
  - `200tq/scripts/backtest_api.py:78-88` - Ensemble signal in API version

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: E03 signal generates binary ON/OFF
    Tool: Bash (python)
    Preconditions: E03 signal implemented
    Steps:
      1. Run: python -c "from backtest_user_strategy_suite import load_data, generate_e03_signal; df = load_data(); sig = generate_e03_signal(df); print(sig.value_counts())"
      2. Assert: Only values 0 and 1 present
      3. Assert: Reasonable distribution (not all 0 or all 1)
    Expected Result: Binary signal with mix of ON/OFF
    Evidence: stdout captured

  Scenario: E03 allocation is 100% TQQQ or 100% SGOV
    Tool: Bash (python)
    Preconditions: E03 implemented
    Steps:
      1. Check signal_to_weights mapping
      2. Assert: ON (1) → TQQQ=1.0, SGOV=0.0
      3. Assert: OFF (0) → TQQQ=0.0, SGOV=1.0
    Expected Result: No partial allocations
    Evidence: Weight arrays captured
  ```

  **Commit**: YES (groups with 1, 2)
  - Message: `feat(backtest): add E03 Ensemble baseline with 100% SGOV OFF`
  - Files: `200tq/scripts/backtest_user_strategy_suite.py`
  - Pre-commit: `python -c "from 200tq.scripts.backtest_user_strategy_suite import generate_e03_signal"`

---

- [ ] 4. Backtest Engine with Cost and Tax Model

  **What to do**:
  - Implement `run_backtest()` function accepting:
    - prices DataFrame
    - signal weights (TQQQ, QLD, SGOV weights per day)
    - config parameters
  
  - Apply t+1 execution lag: `weights.shift(1).fillna(0)`
  
  - Calculate daily portfolio return:
    ```python
    port_ret = (tqqq_w * tqqq_ret + qld_w * qld_ret + sgov_w * sgov_ret)
    ```
  
  - Transaction cost model (10 bps one-way):
    ```python
    weight_change = (tqqq_w.diff().abs() + qld_w.diff().abs()).fillna(0)
    cost_drag = weight_change * 0.001  # 10 bps
    ```
  
  - Tax model (22% on annual realized gains):
    - Track cost basis for TQQQ and QLD separately
    - On sells, calculate realized gain = shares_sold * (price - avg_cost)
    - Accumulate yearly gains
    - At year-end, apply 22% tax on positive gains only
    - Deduct tax from portfolio value
  
  - Track and return:
    - equity curve (daily values)
    - trades list (date, asset, side, value, cost, realized_gain)
    - yearly tax breakdown
    - total trades count

  **Must NOT do**:
  - Do not skip the lag (t+1 is critical)
  - Do not apply tax on unrealized gains
  - Do not use 20 bps (user spec says 10 bps)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: Complex iterative calculation with tax basis tracking
  - **Skills**: `[]`
    - Standard Python/pandas

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 5
  - **Blocked By**: Tasks 2, 3

  **References**:
  - `200tq/scripts/run_suite.py:353-524` - Full `run_backtest()` with iterative tax tracking
  - `200tq/scripts/run_suite.py:419-424` - Cost calculation pattern
  - `200tq/scripts/run_suite.py:426-505` - Tax basis tracking loop
  - `200tq/scripts/backtest_api.py:167-306` - Simplified version with same logic

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: T+1 lag is correctly applied
    Tool: Bash (python)
    Preconditions: Backtest engine implemented
    Steps:
      1. Create test signal that changes on specific date
      2. Verify position change happens on next trading day
      3. Assert: Signal on day T → Position change on day T+1
    Expected Result: Lag correctly applied
    Evidence: Signal vs position dates compared

  Scenario: Transaction costs reduce returns
    Tool: Bash (python)
    Preconditions: Backtest engine with cost toggle
    Steps:
      1. Run backtest with cost=10bps
      2. Run backtest with cost=0bps
      3. Assert: Final equity with cost < Final equity without cost
    Expected Result: Costs have measurable drag
    Evidence: Two final values compared

  Scenario: Tax is applied at year-end only
    Tool: Bash (python)
    Preconditions: Backtest engine with tax tracking
    Steps:
      1. Run backtest over 2 years
      2. Check equity curve for step-down at year boundaries
      3. Assert: Tax deducted on Dec 31 or last trading day of year
    Expected Result: Annual tax visible in equity curve
    Evidence: Year-end equity changes captured
  ```

  **Commit**: YES
  - Message: `feat(backtest): implement vectorized backtest engine with 10bps cost and 22% tax`
  - Files: `200tq/scripts/backtest_user_strategy_suite.py`
  - Pre-commit: `python -c "from 200tq.scripts.backtest_user_strategy_suite import run_backtest"`

---

- [ ] 5. Experiment Runner for All 11+ Experiments

  **What to do**:
  - Define `ExperimentConfig` dataclass with all parameters
  - Define all experiments:
  
  | ID | Name | Config |
  |----|------|--------|
  | E00 | E03_Ensemble_Baseline | E03 signal, OFF=100%SGOV |
  | E01 | User_Relaxed_Gate | Full user strategy, use_entry_gate=False |
  | E02 | User_Strict_Gate | Full user strategy, use_entry_gate=True |
  | E03 | User_No_VolLock | use_vol_lock=False |
  | E04 | User_No_Overheat | use_overheat=False |
  | E05 | User_No_QLD_Step | use_qld_step=False (score 1 → TQQQ) |
  | E06a | Vol_Sensitivity_5.0 | vol_threshold=0.050 |
  | E06b | Vol_Sensitivity_6.2 | vol_threshold=0.062 (baseline) |
  | E06c | Vol_Sensitivity_7.5 | vol_threshold=0.075 |
  | E07a | Overheat_2.30 | overheat_ratio=2.30, reentry_ratio=2.00 |
  | E07b | Overheat_2.51 | overheat_ratio=2.51, reentry_ratio=2.18 (baseline) |
  | E07c | Overheat_2.70 | overheat_ratio=2.70, reentry_ratio=2.35 |
  | E08a | MA161_Sensitivity_160 | ma_long_qqq=160 |
  | E08b | MA161_Sensitivity_161 | ma_long_qqq=161 (baseline) |
  | E08c | MA161_Sensitivity_165 | ma_long_qqq=165 |
  | E09a | TQQQ_MA200_Sens_180 | ma_long_tqqq=180 |
  | E09b | TQQQ_MA200_Sens_200 | ma_long_tqqq=200 (baseline) |
  | E09c | TQQQ_MA200_Sens_220 | ma_long_tqqq=220 |
  | E10a | E03_Ensemble_155_160_165 | E03 with windows [155, 160, 165] |
  | E10b | E03_Ensemble_160_165_170 | E03 with windows [160, 165, 170] (baseline) |
  
  - Implement `run_all_experiments()`:
    - Load data once
    - Loop through experiments
    - Run backtest for each
    - Collect results
    - Save per-experiment artifacts

  - Implement `calculate_metrics()`:
    - CAGR, MDD, Sharpe, Sortino, Calmar
    - Trades count, turnover
    - Total tax paid

  **Must NOT do**:
  - Do not skip any experiments
  - Do not reorder experiments (E00 must be first for delta calculation)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Loop orchestration with well-defined patterns
  - **Skills**: `[]`
    - Standard Python

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:
  - `200tq/scripts/run_suite.py:50-63` - `ExperimentConfig` dataclass
  - `200tq/scripts/run_suite.py:65-201` - EXPERIMENTS list definition
  - `200tq/scripts/run_suite.py:717-773` - `run_suite()` main loop pattern
  - `200tq/scripts/run_suite.py:530-578` - `calculate_metrics()` function

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All experiments run successfully
    Tool: Bash (python)
    Preconditions: Full script implemented
    Steps:
      1. Run: python 200tq/scripts/backtest_user_strategy_suite.py
      2. Assert: No Python errors
      3. Assert: stdout shows 20 experiments completed
    Expected Result: All experiments finish
    Evidence: stdout captured

  Scenario: E00 baseline produces reasonable metrics
    Tool: Bash (python)
    Preconditions: E00 runs
    Steps:
      1. Check E00 CAGR is in range 25-40%
      2. Check E00 MDD is in range -60% to -45%
      3. Assert: Metrics are not NaN or Inf
    Expected Result: Baseline metrics reasonable
    Evidence: Metrics JSON captured

  Scenario: User strategy results can be verified
    Tool: Bash (python)
    Preconditions: E02 (strict gate) runs
    Steps:
      1. Check E02 CAGR (user claimed ~54.23%)
      2. Check E02 MDD (user claimed ~48.92%)
      3. Check E02 trades (user claimed ~177)
      4. Note: Actual may differ due to tax/cost application
    Expected Result: Results captured for verification
    Evidence: E02 metrics captured
  ```

  **Commit**: YES
  - Message: `feat(backtest): add experiment runner with all 20 variants`
  - Files: `200tq/scripts/backtest_user_strategy_suite.py`
  - Pre-commit: `python 200tq/scripts/backtest_user_strategy_suite.py --dry-run` (if implemented)

---

- [ ] 6. Visualization and Reporting

  **What to do**:
  - Create output directory: `200tq/experiments/user_strategy_comparison/`
  
  - Generate `leaderboard.csv` and `leaderboard.md`:
    - Sort by CAGR descending
    - Include: Rank, Experiment, CAGR, ΔCAGR (vs E00), MDD, Sharpe, Calmar, Trades
    - Markdown format with proper table formatting

  - Generate `equity_curves.png`:
    - All experiments on single log-scale plot
    - Legend with experiment names
    - 14x7 inch figure, 150 DPI

  - Generate `drawdowns.png`:
    - All experiments drawdown curves
    - Fill between for visibility

  - Generate `yearly_returns_heatmap.png`:
    - Rows: Experiments
    - Columns: Years (2010-2025)
    - Color scale: Red (negative) to Green (positive)

  - Generate `sensitivity_table.csv`:
    - Group by parameter (vol_threshold, overheat_ratio, etc.)
    - Show how CAGR/MDD changes with parameter values

  - Generate `tax_by_year.csv`:
    - Rows: Years
    - Columns: Experiments
    - Values: Tax paid that year

  - Save per-experiment artifacts:
    - `{exp_name}/equity_curve.csv`
    - `{exp_name}/trades.csv`
    - `{exp_name}/metrics.json`

  **Must NOT do**:
  - Do not use interactive plots (save to file only)
  - Do not skip any output file

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard matplotlib/pandas visualization
  - **Skills**: `[]`
    - Standard Python

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5 (final)
  - **Blocks**: None
  - **Blocked By**: Task 5

  **References**:
  - `200tq/scripts/run_suite.py:606-662` - `save_artifacts()` per-experiment
  - `200tq/scripts/run_suite.py:665-711` - `generate_summary()` leaderboard
  - `200tq/scripts/run_suite.py:751-760` - Combined equity plot
  - `200tq/scripts/backtest_hybrid_2_0.py:747-844` - `generate_report()` multi-plot pattern

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All output files are created
    Tool: Bash (ls)
    Preconditions: Full run completed
    Steps:
      1. ls 200tq/experiments/user_strategy_comparison/
      2. Assert: leaderboard.csv exists
      3. Assert: leaderboard.md exists
      4. Assert: equity_curves.png exists (size > 100KB)
      5. Assert: drawdowns.png exists
      6. Assert: yearly_returns_heatmap.png exists
      7. Assert: sensitivity_table.csv exists
      8. Assert: tax_by_year.csv exists
    Expected Result: All files present
    Evidence: ls -la output captured

  Scenario: Leaderboard is properly sorted
    Tool: Bash (python)
    Preconditions: leaderboard.csv created
    Steps:
      1. Read leaderboard.csv
      2. Assert: Sorted by CAGR descending
      3. Assert: Rank column is 1, 2, 3, ...
    Expected Result: Proper ranking
    Evidence: First 5 rows captured

  Scenario: PNG files are valid images
    Tool: Bash (file)
    Preconditions: PNG files created
    Steps:
      1. Run: file 200tq/experiments/user_strategy_comparison/*.png
      2. Assert: Each file identified as "PNG image data"
    Expected Result: Valid PNG files
    Evidence: file command output
  ```

  **Commit**: YES
  - Message: `feat(backtest): add visualization and reporting for all experiments`
  - Files: `200tq/scripts/backtest_user_strategy_suite.py`
  - Pre-commit: `python 200tq/scripts/backtest_user_strategy_suite.py`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1, 2, 3 | `feat(backtest): add User Strategy backtest suite foundation` | backtest_user_strategy_suite.py | Import test |
| 4 | `feat(backtest): implement backtest engine with cost/tax model` | backtest_user_strategy_suite.py | Import run_backtest |
| 5 | `feat(backtest): add all 20 experiment configurations` | backtest_user_strategy_suite.py | Dry run |
| 6 | `feat(backtest): add visualization and reporting` | backtest_user_strategy_suite.py | Full run |

---

## Success Criteria

### Verification Commands
```bash
# Full run
python 200tq/scripts/backtest_user_strategy_suite.py

# Check outputs
ls -la 200tq/experiments/user_strategy_comparison/

# Verify leaderboard
cat 200tq/experiments/user_strategy_comparison/leaderboard.md

# Check file sizes (ensure plots generated)
du -h 200tq/experiments/user_strategy_comparison/*.png
```

### Final Checklist
- [ ] All 20 experiments complete without error
- [ ] E00 (E03 baseline) has CAGR ~30-35% (matching existing run_suite.py results)
- [ ] User strategy experiments (E01-E05) produce valid metrics
- [ ] Sensitivity analysis (E06-E10) shows parameter impact
- [ ] All PNG files are > 50KB (not empty)
- [ ] leaderboard.md is valid markdown table
- [ ] tax_by_year.csv shows annual tax for each experiment
- [ ] User's claimed results (CAGR ~54%, MDD ~49%) can be explained if different
