# Draft: User Strategy vs E03 Comparison Backtest Suite

## Requirements (confirmed)

### Strategy A: E03 Ensemble (Baseline)
- Signal: QQQ SMA(3) > SMA(window) ensemble majority vote (windows: 160, 165, 170)
- Position: ON = TQQQ 100%, OFF = SGOV 100% (or TQQQ 10% + SGOV 90% for OFF10 variant)
- Execution: t+1 close

### Strategy B: User Strategy (Trend Score + Forced Exits)

**Trend Score (0-2):**
- QQQ MA3 > QQQ MA161 → +1 point
- TQQQ price > TQQQ MA200 → +1 point
- Score 2 → TQQQ 100%
- Score 1 → QLD 100%
- Score 0 → SGOV 100%

**Forced Exits (override score):**
- Volatility Lock: 20-day volatility >= 6.2% → SGOV forced
- Overheat Sell: TQQQ / TQQQ_MA200 >= 2.51 (151% above) → SGOV + overheat_mode=True
- Re-entry: In overheat_mode, only re-enter when ratio <= 2.18 (118%)

**Entry Gate:**
- "중간 탑승 금지" - New entries only allowed when score transitions from 0 to higher

## 11 Experiments

| ID | Name | Description |
|----|------|-------------|
| E00 | E03 Ensemble baseline | Existing run_suite.py E03 strategy |
| E01 | User Relaxed Gate | Full user strategy, no entry restriction |
| E02 | User Strict Gate | Full user strategy, only breakthrough entries (0→1+) |
| E03 | No VolLock | User strategy without volatility lock |
| E04 | No Overheat | User strategy without overheat/re-entry logic |
| E05 | No QLD Step | Score 1 also goes to TQQQ (skip QLD intermediate) |
| E06a/b/c | Vol Sensitivity | Thresholds: 5.0%, 6.2%, 7.5% |
| E07 | Overheat Sensitivity | Multiple overheat thresholds |
| E08 | MA161 Sensitivity | Windows: 160, 161, 165 |
| E09 | TQQQ MA200 Sensitivity | Windows: 180, 200, 220 |
| E10 | E03 Ensemble Variants | Windows: 155, 160, 165 |

## Technical Decisions

- **Data source**: yfinance (following existing pattern)
- **Execution lag**: t+1 close via `signal.shift(1)`
- **Transaction cost**: 10 bps one-way (as existing)
- **Tax model**: 22% Korean overseas ETF on annual realized gains
- **QLD handling**: Synthetic 2x from QQQ for pre-inception dates (pattern from backtest_hybrid_2_0.py)
- **20-day volatility**: `returns.rolling(20).std() * np.sqrt(252)` annualized

## Research Findings

### From run_suite.py:
- ExperimentConfig dataclass for experiment definitions
- Iterative backtest loop for tax basis tracking
- Artifact generation: equity_curve.csv, trades.csv, yearly_returns.csv, metrics.json
- Combined plots: all_equity_curves.png, leaderboard.md

### From backtest_hybrid_2_0.py:
- QLD already implemented with synthetic fallback
- Pattern: `build_synthetic_2x()` function available
- F&G integration shows state machine pattern for complex entry logic

### From backtest_api.py:
- JSON API pattern for individual backtests
- Same cost/tax model

## Required Outputs
- leaderboard.csv/md
- equity_curves.png
- drawdowns.png  
- yearly_returns_heatmap.png
- sensitivity_table.csv
- tax_by_year.csv

## Open Questions (RESOLVED)
1. ✅ Test infrastructure: No tests - manual inspection via results
2. ✅ Entry gate: Score must be 0 on previous day (0→1 or 0→2 only)
3. ✅ E03 baseline: OFF = 100% SGOV for fair comparison

## User Decisions Recorded
- E03 Baseline: OFF = 100% SGOV (not OFF10 variant)
- Entry Gate: "중간 탑승 금지" = previous day score must be 0
- E07 Overheat thresholds: 2.30/2.51/2.70, Re-entry: 2.00/2.18/2.35
- No automated tests - manual inspection only
- E06 Vol thresholds: 5.0%, 6.2%, 7.5%
- E08 MA161: 160, 161, 165
- E09 TQQQ MA200: 180, 200, 220

## Scope Boundaries
- INCLUDE: All 11 experiments, visualizations, CSV outputs
- EXCLUDE: Web UI integration, real-time execution, database persistence, automated tests
