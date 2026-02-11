# E03 Dashboard v2026.3 Alignment

## TL;DR

> **Quick Summary**: Align the E03 Next.js dashboard (currently v2026.1: ON/OFF10 binary states) with the new E03_SSOT v2026.3 strategy that introduces F1 Signal Stability Filter (FlipCount), 4 operational states (ON-Normal, ON-Choppy, OFF, Emergency), updated Emergency thresholds (-5%/-15%), and Emergency cooldown. Scope includes core logic TDD, UI updates to Zones A/B/C, and syncing all 4 dashboard SSOT documents.
>
> **Deliverables**:
> - Extended type system with 4 strategy states + FlipCount fields
> - TDD-verified buildViewModel with F1 filter + Emergency + state priority logic
> - Updated mock scenarios covering all 4 states
> - FlipCount gauge + timeline visualization in Zone B
> - Emergency thresholds updated from -7%/-20% to -5%/-15%
> - ON-Choppy 70%/30% order calculation in Zone C
> - 4-state badge handling in Zone A
> - Choppy amber color tokens in design system
> - All 4 dashboard SSOT docs updated to v2026.3
>
> **Estimated Effort**: Large (7 tasks across 4 waves)
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 7

---

## Context

### Original Request
사용자가 새로 작성한 E03_SSOT v2026.3 전략 문서에 맞춰 기존 대시보드를 업데이트. F1 Signal Stability Filter(FlipCount), 4가지 운영 상태, 강화된 Emergency 임계값(-5%/-15%), Emergency 쿨다운 등 새로운 기능을 반영. 코드 + UI 리디자인 + SSOT 문서 동기화 범위.

### Interview Summary
**Key Discussions**:
- **Scope**: 코어 + UI 리디자인 (중간 범위) — 기존 Zone 레이아웃(A/B/C/D)은 유지
- **SSOT Docs**: 코드 + 문서 함께 업데이트
- **FlipCount Viz**: 게이지 + 타임라인 (40일 시그널 히스토리 표시)
- **Color System**: 기존 색상 토큰 확장 (amber for choppy)
- **Testing**: 핵심 로직만 TDD (buildViewModel.ts)
- **Approach**: 사용자 위임 ("당신이 판단하세요")

**Research Findings**:
- 4-state Market Structure Color System maps perfectly: Green(ON-Normal) → Amber(ON-Choppy) → Gray(OFF) → Red(Emergency)
- Korean fintech conventions: 만/억 formatting already implemented, red=rising already in place
- Test infrastructure verified: vitest + jsdom + react-testing-library ready
- Risk Monitor logic currently hardcoded in ZoneBSignalCore.tsx (lines 159-220) — should be centralized into buildViewModel

**Expert Judgment Decisions** (user delegated):
- FlipCount source: calculated in buildViewModel from `signalHistory: boolean[]` (past 40 days)
- State priority: Emergency > Cooldown > OFF > ON-Choppy > ON-Normal
- Cooldown: boolean modifier on state (not a 5th state); forces OFF10 for 1 extra day
- Emergency triggers: two independent conditions (QQQ daily -5%, TQQQ entry -15%), same result
- Missing flipCount: defaults to 0 (safe = no choppy detection)

### Metis Review
**Identified Gaps** (all addressed):
- FlipCount data source: → calculated in buildViewModel from signalHistory array
- State priority hierarchy: → Emergency > Cooldown > OFF > ON-Choppy > ON-Normal (per SSOT Part 2)
- Cooldown representation: → boolean modifier, not separate state
- Emergency sub-levels: → two independent triggers, same outcome
- Backward compatibility: → flipCount defaults to 0 when signalHistory missing

---

## Work Objectives

### Core Objective
Bring the E03 dashboard's core logic, UI, and documentation into full alignment with E03_SSOT v2026.3, enabling 4-state display, F1 Signal Stability Filter, and updated Emergency thresholds.

### Concrete Deliverables
- `lib/ops/e03/types.ts` — extended StrategyState union + new fields
- `lib/ops/e03/buildViewModel.ts` — F1 filter, Emergency state, state priority, 70% allocation
- `__tests__/buildViewModel.test.ts` — comprehensive TDD test suite
- `lib/ops/e03/mock.ts` — new scenarios (choppy, emergency_qqq, emergency_tqqq, cooldown)
- `lib/ops/dataSource.ts` — signalHistory passthrough + v2026.3 verdict types
- `components/e03/ZoneBSignalCore.tsx` — FlipCount gauge+timeline, 4-state verdict, -5%/-15%
- `components/e03/ZoneAHeader.tsx` — 4-state badge handling
- `components/e03/ZoneCOpsConsole.tsx` — ON-Choppy 70%/30% trade lines
- `app/globals.css` + `tailwind.config.ts` — choppy amber color tokens
- 4 SSOT docs updated to v2026.3

### Definition of Done
- [x] `npx vitest run` → all tests pass
- [x] `npm run build` → 0 errors
- [x] All 4 StrategyStates visually distinct on dashboard
- [x] FlipCount gauge renders with correct threshold indicator
- [x] Risk Monitor shows -5%/-15% thresholds (not -7%/-20%)
- [x] ON-Choppy scenario shows 70% TQQQ / 30% SGOV trades
- [x] All 4 SSOT docs reference v2026.3

### Must Have
- 4 strategy states: ON, ON_CHOPPY, OFF10, EMERGENCY
- F1 filter: FlipWindow=40d, FlipThreshold=3, ReducedWeight=0.70
- Emergency thresholds: QQQ -5%, TQQQ -15%
- Emergency cooldown: 1 day
- State priority: Emergency > Cooldown > OFF > ON-Choppy > ON-Normal
- FlipCount defaults to 0 when signalHistory absent

### Must NOT Have (Guardrails)
- ❌ Do NOT refactor Zone layout structure (A/B/C/D stays as-is)
- ❌ Do NOT change routing or page structure (pages remain Client Components)
- ❌ Do NOT add new state management beyond existing ViewModel pattern
- ❌ Do NOT implement real API integration for signalHistory (mock data only for now)
- ❌ Do NOT create a 5th "COOLDOWN" state — cooldown is a boolean modifier
- ❌ Do NOT touch performance charts, MacroStrip, or analysis pages
- ❌ Do NOT change the existing ON/OFF10 behavior when flipCount < 3 (backward compatible)
- ❌ Do NOT remove or rename EmergencyState ("NONE"|"SOFT_ALERT"|"HARD_CONFIRMED") — it remains separate from StrategyState

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> Every criterion is agent-executable using vitest, Playwright, or bash commands.

### Test Decision
- **Infrastructure exists**: YES (vitest 4.x + jsdom + react-testing-library)
- **Automated tests**: TDD for core logic (buildViewModel.ts only)
- **Framework**: vitest (already configured at `200tq/dashboard/vitest.config.ts`)
- **Test command**: `npx vitest run` (from `200tq/dashboard/`)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - Parallel):
├── Task 1: Types + buildViewModel TDD [no dependencies]
└── Task 3: Choppy color tokens [no dependencies]

Wave 2 (After Task 1):
└── Task 2: Mock scenarios + dataSource [depends: Task 1]

Wave 3 (After Tasks 1, 2, 3 - Parallel):
├── Task 4: ZoneBSignalCore update [depends: 1, 2, 3]
├── Task 5: ZoneAHeader update [depends: 1, 2, 3]
└── Task 6: ZoneCOpsConsole update [depends: 1, 2, 3]

Wave 4 (After All Code - Final):
└── Task 7: SSOT doc sync [depends: 1-6]

Critical Path: Task 1 → Task 2 → Task 4 → Task 7
Parallel Speedup: ~50% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 4, 5, 6 | 3 |
| 2 | 1 | 4, 5, 6 | — |
| 3 | None | 4, 5, 6 | 1 |
| 4 | 1, 2, 3 | 7 | 5, 6 |
| 5 | 1, 2, 3 | 7 | 4, 6 |
| 6 | 1, 2, 3 | 7 | 4, 5 |
| 7 | 1-6 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Dispatch |
|------|-------|---------------------|
| 1 | 1, 3 | task(category="ultrabrain") + task(category="quick") in parallel |
| 2 | 2 | task(category="unspecified-low") |
| 3 | 4, 5, 6 | 3x task(category="visual-engineering") in parallel |
| 4 | 7 | task(category="writing") |

---

## TODOs

- [x] 1. [TDD] Extend types.ts + buildViewModel.ts with F1 Filter, Emergency, and State Priority

  **What to do**:

  **1a. Update `lib/ops/e03/types.ts`**:
  - Extend `StrategyState` union: `"ON" | "ON_CHOPPY" | "OFF10" | "EMERGENCY"`
  - Add new fields to `E03ViewModel`:
    ```typescript
    flipCount: number;           // 0-N, from signalHistory
    isChoppy: boolean;           // flipCount >= 3 && strategyState involves ON
    targetTqqqWeight: number;    // 1.0 | 0.7 | 0.1
    cooldownActive: boolean;     // Emergency cooldown in effect
    signalHistory?: boolean[];   // Past 40 days ON/OFF for timeline viz
    ```
  - Update `verdictTitle` to handle all 4 states

  **1b. Write failing tests first (`__tests__/buildViewModel.test.ts`)** (RED phase):
  - Test: ON-Normal (FlipCount=0, signal ON) → strategyState="ON", targetTqqqWeight=1.0
  - Test: ON-Normal (FlipCount=2, signal ON) → strategyState="ON" (below threshold)
  - Test: ON-Choppy (FlipCount=3, signal ON) → strategyState="ON_CHOPPY", targetTqqqWeight=0.7
  - Test: ON-Choppy (FlipCount=5, signal ON) → strategyState="ON_CHOPPY"
  - Test: OFF (signal OFF, flipCount=0) → strategyState="OFF10", targetTqqqWeight=0.1
  - Test: OFF (signal OFF, flipCount=5) → strategyState="OFF10" (F1 does NOT apply to OFF)
  - Test: Emergency priority over ON → QQQ drop -6%, signal ON → strategyState="EMERGENCY"
  - Test: Emergency priority over OFF → TQQQ entry -16%, signal OFF → strategyState="EMERGENCY"
  - Test: Cooldown active → forces OFF10 even if signal is ON
  - Test: flipCount calculation from signalHistory: [T,T,F,T,F,T,...] → count flips
  - Test: empty/undefined signalHistory → flipCount=0, no choppy
  - Test: ON-Choppy trade lines: 70% TQQQ / 30% SGOV
  - Test: Emergency trade lines: 10% TQQQ / 90% SGOV (same as OFF10)
  - Test: verdictTitle strings for all 4 states

  **1c. Update `buildViewModel.ts`** (GREEN phase):
  - Add `signalHistory?: boolean[]` to `E03RawInputs`
  - Add `qqqDailyReturn?: number` to `E03RawInputs` (for Emergency QQQ check)
  - Add `tqqqEntryPrice?: number` to `E03RawInputs` (for Emergency TQQQ check)
  - Add `cooldownActive?: boolean` to `E03RawInputs`
  - Implement `calculateFlipCount(signalHistory: boolean[]): number` — count transitions in past 40 days
  - Implement state priority logic after voting:
    ```
    1. Check Emergency: qqqDailyReturn <= -5% OR tqqqCurrentPrice/tqqqEntryPrice <= 0.85
    2. Check Cooldown: cooldownActive → force OFF10
    3. Check OFF: voteCount < 2 → OFF10
    4. Check Choppy: voteCount >= 2 AND flipCount >= 3 → ON_CHOPPY
    5. Default: ON
    ```
  - Update trade line generation for ON_CHOPPY:
    - Calculate 70% target TQQQ shares
    - Calculate 30% SGOV shares from remainder
    - Different notes: "Choppy 구간: TQQQ 70%, SGOV 30%"
  - Update trade line generation for EMERGENCY:
    - Same as OFF10 (10%/90%)
    - Different notes: "Emergency Exit: 10% 잔류"
  - Centralize risk metrics into ViewModel output (move from ZoneBSignalCore hardcoded values):
    - `emergencyQqqThreshold: -5` (was -7)
    - `emergencyTqqqThreshold: -15` (was -20)
  - Set `verdictTitle` per state:
    - ON → "VERDICT: ON"
    - ON_CHOPPY → "VERDICT: ON (CHOPPY)"
    - OFF10 → "VERDICT: OFF10"
    - EMERGENCY → "VERDICT: EMERGENCY"

  **1d. Refactor** (REFACTOR phase): Clean up, ensure all tests still pass.

  **Must NOT do**:
  - Do NOT change EmergencyState type ("NONE"|"SOFT_ALERT"|"HARD_CONFIRMED")
  - Do NOT modify any UI component in this task
  - Do NOT break existing ON/OFF10 behavior when no signalHistory present
  - Do NOT implement real API calls for signalHistory

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: TDD cycle with complex state machine logic, multiple interacting conditions, state priority hierarchy
  - **Skills**: []
    - No special skills needed — pure TypeScript logic + vitest testing
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser interaction in this task
    - `frontend-ui-ux`: No UI work in this task

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Parallel Group**: Wave 1 (with Task 3)
  - **Blocks**: Tasks 2, 4, 5, 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `lib/ops/e03/types.ts:1-67` — Current type definitions including StrategyState union and E03ViewModel interface. Executor must ADD to these, not replace.
  - `lib/ops/e03/buildViewModel.ts:41-66` — Current voting logic (lines 55-66) that determines ON/OFF10. New F1 filter logic inserts AFTER this, before state assignment at line 66.
  - `lib/ops/e03/buildViewModel.ts:139-223` — Current trade line generation for ON/OFF10. Executor must ADD branches for ON_CHOPPY (70%/30%) and EMERGENCY (same as OFF10 but different notes).
  - `__tests__/example.test.ts` — Minimal vitest example confirming test infrastructure works. New test file follows same import pattern.

  **API/Type References** (contracts to implement against):
  - `200tq/E03_SSOT.md:92-115` — F1 Signal Stability Filter specification: FlipWindow=40, FlipThreshold=3, ReducedWeight=0.70. F1 does NOT apply to OFF state.
  - `200tq/E03_SSOT.md:117-131` — Emergency Exit rules: QQQ ≤-5%, TQQQ ≤-15% from entry, 1-day cooldown, target is OFF10 (10% not 0%).
  - `200tq/E03_SSOT.md:133-141` — Position allocation table: ON=100%, Choppy=70%/30%, OFF=10%/90%, Emergency=10%/90%.
  - `200tq/E03_SSOT.md:260-278` — Daily ops checklist showing exact decision flow order.

  **Test References** (testing patterns to follow):
  - `vitest.config.ts` — Test config: jsdom environment, globals=true, @/ alias configured, setupFiles at `__tests__/setup.ts`.

  **Documentation References**:
  - `200tq/E03_SSOT.md:280-288` — F1 Signal Stability calculation method (step by step).

  **Acceptance Criteria**:

  - [ ] `StrategyState` type is `"ON" | "ON_CHOPPY" | "OFF10" | "EMERGENCY"`
  - [ ] `E03ViewModel` has fields: `flipCount`, `isChoppy`, `targetTqqqWeight`, `cooldownActive`
  - [ ] `E03RawInputs` has fields: `signalHistory?`, `qqqDailyReturn?`, `tqqqEntryPrice?`, `cooldownActive?`
  - [ ] `npx vitest run __tests__/buildViewModel.test.ts` → PASS (14+ tests, 0 failures)
  - [ ] Test: buildViewModel({ signalHistory: [T,T,F,T,F,T,...40 items with 3 flips], signal=ON }) → strategyState="ON_CHOPPY"
  - [ ] Test: buildViewModel({ qqqDailyReturn: -6, signal=ON }) → strategyState="EMERGENCY"
  - [ ] Test: buildViewModel({ cooldownActive: true, signal=ON }) → strategyState="OFF10"
  - [ ] Test: buildViewModel({ signalHistory: undefined }) → flipCount=0, strategyState="ON"
  - [ ] Backward compatible: buildViewModel with NO new fields produces same ON/OFF10 as before

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All vitest tests pass for buildViewModel
    Tool: Bash
    Preconditions: Dependencies installed (npm install already done)
    Steps:
      1. Run: npx vitest run __tests__/buildViewModel.test.ts (from 200tq/dashboard/)
      2. Assert: exit code 0
      3. Assert: stdout contains "Tests  14 passed" (or more)
      4. Assert: stdout contains "0 failed"
    Expected Result: All tests pass, 0 failures
    Evidence: Terminal output captured

  Scenario: Build succeeds with new types
    Tool: Bash
    Preconditions: Types updated, buildViewModel updated
    Steps:
      1. Run: npm run build (from 200tq/dashboard/)
      2. Assert: exit code 0
      3. Assert: stdout contains "Compiled successfully" or "✓"
      4. Assert: no TypeScript errors in output
    Expected Result: Build passes with extended types
    Evidence: Build output captured
  ```

  **Commit**: YES
  - Message: `feat(e03): implement F1 Signal Stability Filter and 4-state strategy logic with TDD`
  - Files: `lib/ops/e03/types.ts`, `lib/ops/e03/buildViewModel.ts`, `__tests__/buildViewModel.test.ts`
  - Pre-commit: `npx vitest run __tests__/buildViewModel.test.ts`

---

- [x] 2. Extend mock scenarios and dataSource for v2026.3 fields

  **What to do**:

  **2a. Update `lib/ops/e03/mock.ts`**:
  - Extend `ScenarioId` type with new scenarios:
    ```typescript
    | "choppy_on"          // ON + FlipCount >= 3 → ON_CHOPPY state
    | "emergency_qqq"      // QQQ dropped -6% today
    | "emergency_tqqq"     // TQQQ -16% from entry
    | "cooldown"           // Emergency cooldown active, signal ON but forced OFF10
    ```
  - Add `signalHistory` generation helper:
    ```typescript
    function generateSignalHistory(flipCount: number): boolean[] {
      // Generate 40 booleans with exactly `flipCount` transitions
    }
    ```
  - Update base mock to include new E03RawInputs fields:
    - `signalHistory: generateSignalHistory(0)` (stable, no flips for base)
    - `qqqDailyReturn: 0.5` (normal day)
    - `tqqqEntryPrice: 105.50` (matches existing mock avgPrice)
    - `cooldownActive: false`
  - Add scenario implementations:
    - `choppy_on`: signalHistory with 4 flips, SMAs passing (signal ON)
    - `emergency_qqq`: qqqDailyReturn=-6, normal SMAs
    - `emergency_tqqq`: tqqqEntryPrice=137 (current ~115 = -16%)
    - `cooldown`: cooldownActive=true, normal SMAs (signal ON but forced OFF)
  - Update `getMockPortfolio` stopLossLevel from `* 0.80` to `* 0.85` (reflects -15% threshold)

  **2b. Update `lib/ops/dataSource.ts`**:
  - Update `OpsTodayResponse` type:
    - Change `verdict: "ON" | "OFF10"` → `verdict: "ON" | "ON_CHOPPY" | "OFF10" | "EMERGENCY"`
    - Add optional field: `signalHistory?: boolean[]`
    - Add optional field: `qqqDailyReturn?: number`
    - Add optional field: `tqqqEntryPrice?: number`
    - Add optional field: `cooldownActive?: boolean`
  - Update `buildRealInputs()` to pass through new fields:
    - `signalHistory: data.signalHistory`
    - `qqqDailyReturn: data.qqqDailyReturn`
    - `tqqqEntryPrice: data.tqqqEntryPrice`
    - `cooldownActive: data.cooldownActive`

  **Must NOT do**:
  - Do NOT change existing scenario behavior for "fresh_normal", "stale_or_closed", "soft_alert", "hard_confirmed"
  - Do NOT implement actual API calls for signalHistory data
  - Do NOT modify any UI component

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Straightforward data plumbing — extending types and adding mock data. No complex logic.
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser interaction

  **Parallelization**:
  - **Can Run In Parallel**: NO (Sequential after Task 1)
  - **Parallel Group**: Wave 2 (solo)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `lib/ops/e03/mock.ts:1-171` — Complete mock file. Executor must extend `ScenarioId` union (line 3-7), add cases to `getMockInputsByScenario` switch (line 136-170), keep all existing scenarios working.
  - `lib/ops/e03/mock.ts:107-134` — Base mock object pattern. New fields must be added to this base object so all scenarios inherit them.
  - `lib/ops/dataSource.ts:15-33` — `OpsTodayResponse` type that needs `verdict` union extended and new optional fields.
  - `lib/ops/dataSource.ts:60-93` — `buildRealInputs()` function that transforms API response to E03RawInputs. Must add passthrough for new fields.

  **API/Type References**:
  - `lib/ops/e03/types.ts` — Updated StrategyState union from Task 1 (must match exactly)
  - `lib/ops/e03/buildViewModel.ts` — Updated E03RawInputs from Task 1 (new fields to populate)
  - `200tq/E03_SSOT.md:133-141` — Position allocation table for each scenario's expected outcome

  **Acceptance Criteria**:

  - [ ] `ScenarioId` includes "choppy_on", "emergency_qqq", "emergency_tqqq", "cooldown"
  - [ ] `getMockInputsByScenario("choppy_on")` returns inputs where signalHistory has ≥3 flips and SMAs trigger ON
  - [ ] `getMockInputsByScenario("emergency_qqq")` returns inputs with qqqDailyReturn ≤ -5
  - [ ] `getMockInputsByScenario("emergency_tqqq")` returns inputs with TQQQ ≤ -15% from entry
  - [ ] `getMockInputsByScenario("cooldown")` returns inputs with cooldownActive=true
  - [ ] All existing scenarios ("fresh_normal", etc.) still work unchanged
  - [ ] `npm run build` → 0 errors
  - [ ] `npx vitest run` → all tests still pass (no regressions)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All mock scenarios produce valid ViewModels
    Tool: Bash
    Preconditions: Tasks 1 and 2 complete
    Steps:
      1. Create a quick validation script or add test:
         import each scenario → buildViewModel → assert no throws
      2. Run: npx vitest run
      3. Assert: all tests pass
    Expected Result: Every scenario produces a valid ViewModel without errors
    Evidence: Test output captured

  Scenario: Build succeeds with extended mock and dataSource
    Tool: Bash
    Steps:
      1. Run: npm run build (from 200tq/dashboard/)
      2. Assert: exit code 0
    Expected Result: No TypeScript errors
    Evidence: Build output captured
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `feat(e03): add v2026.3 mock scenarios and dataSource passthrough`
  - Files: `lib/ops/e03/mock.ts`, `lib/ops/dataSource.ts`
  - Pre-commit: `npx vitest run && npm run build`

---

- [x] 3. Add choppy state color tokens to design system

  **What to do**:

  **3a. Update `app/globals.css`**:
  - Add choppy/amber semantic color tokens in `:root`:
    ```css
    --choppy: 245 158 11;        /* amber-500 #F59E0B */
    ```
  - Add choppy tokens in `.dark`:
    ```css
    --choppy: 251 191 36;        /* amber-400 #FBBF24 (brighter for dark bg) */
    ```
  - Add strategy strip for choppy state (alongside existing `--strip-e03-on` and `--strip-e03-off`):
    ```css
    --strip-e03-choppy: #f59e0b;  /* amber-500 */
    ```
    And in `.dark`:
    ```css
    --strip-e03-choppy: #fbbf24;  /* amber-400 */
    ```

  **3b. Update `tailwind.config.ts`**:
  - Add `choppy` color token alongside existing `positive` and `negative`:
    ```typescript
    choppy: "rgb(var(--choppy) / <alpha-value>)",
    ```
  - Add `status.choppy` entry in the status group:
    ```typescript
    choppy: {
      bg: "rgb(var(--choppy) / <alpha-value>)",
      fg: "rgb(var(--fg) / <alpha-value>)",
    },
    ```

  **Must NOT do**:
  - Do NOT change any existing color values
  - Do NOT modify the font or shadow configurations
  - Do NOT add colors for states that already have tokens (ON=positive/emerald, OFF=status-inactive, Emergency=negative/red)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small surgical change — adding CSS variables and Tailwind config entries. ~20 lines total.
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Overkill for adding 4 CSS variables and 3 Tailwind entries

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `app/globals.css:7-42` — `:root` block with existing CSS custom property pattern. New `--choppy` variable must follow same `R G B` format (not hex) for Tailwind alpha-value compatibility.
  - `app/globals.css:44-73` — `.dark` block with corresponding dark-mode overrides. Must add dark variant of `--choppy`.
  - `app/globals.css:27-31` — Strategy strip CSS variables pattern (`--strip-e03-on`, `--strip-e03-off`). Add `--strip-e03-choppy` following same hex format.
  - `tailwind.config.ts:19-21` — Semantic color pattern (`positive`, `negative`). Add `choppy` following identical pattern.
  - `tailwind.config.ts:23-36` — Status color group pattern. Add `choppy` sub-group.

  **Acceptance Criteria**:

  - [ ] `--choppy` CSS variable defined in both `:root` and `.dark`
  - [ ] `--strip-e03-choppy` defined in both `:root` and `.dark`
  - [ ] `choppy` available as Tailwind utility: `text-choppy`, `bg-choppy`, `border-choppy`
  - [ ] `status-choppy-bg` and `status-choppy-fg` available as Tailwind utilities
  - [ ] `npm run build` → 0 errors
  - [ ] Existing color tokens unchanged

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Build succeeds with new color tokens
    Tool: Bash
    Steps:
      1. Run: npm run build (from 200tq/dashboard/)
      2. Assert: exit code 0
    Expected Result: Tailwind compiles with new tokens
    Evidence: Build output captured
  ```

  **Commit**: YES
  - Message: `style(e03): add choppy/amber color tokens for ON-Choppy state`
  - Files: `app/globals.css`, `tailwind.config.ts`
  - Pre-commit: `npm run build`

---

- [x] 4. Update ZoneBSignalCore: FlipCount gauge, 4-state verdict, Emergency thresholds

  **What to do**:

  **4a. Update verdict display** (lines 108-121 area):
  - Replace binary `isOff` logic with 4-state handling:
    ```typescript
    const stateConfig = {
      ON:         { actionLabel: "매수 유지",    stateLabel: "ON",        colorClass: "text-positive" },
      ON_CHOPPY:  { actionLabel: "매수 축소",    stateLabel: "CHOPPY",    colorClass: "text-choppy" },
      OFF10:      { actionLabel: "매도 대기",    stateLabel: "OFF",       colorClass: "text-amber-400" },  // keep existing amber for OFF
      EMERGENCY:  { actionLabel: "비상 전환",    stateLabel: "EMERGENCY", colorClass: "text-negative animate-pulse" },
    };
    ```
  - Update the `<h2>` heading and `<span>` badge to use the state config

  **4b. Update Risk Monitor** (lines 159-220):
  - Change TQQQ threshold from `-20%` to `-15%`:
    - `stopLossPrice = entryPrice * 0.85` (was 0.80)
    - Warning threshold: within 5% of stop (down 10-15% from entry)
    - Label: "TQQQ Flash Crash (-15%)" (was "-20%")
  - Change QQQ threshold from `-7%` to `-5%`:
    - `isQqqTriggered = qqqDropPct <= -5` (was -7)
    - `isQqqWarning = qqqDropPct <= -3 && qqqDropPct > -5` (was -5/-7)
    - Label: "QQQ Drop Alert (-5%)" (was "-7%")
  - **Ideally**: Read thresholds from `vm.emergencyQqqThreshold` and `vm.emergencyTqqqThreshold` (set in buildViewModel) instead of hardcoding. But if buildViewModel doesn't expose these yet, use the constants directly. Either way, the displayed values must be -5%/-15%.

  **4c. Add FlipCount Gauge section** (new section between risk monitor and verdict heading):
  - Only visible when `vm.signalHistory` exists and `vm.flipCount >= 0`
  - **Gauge component**: Horizontal bar showing FlipCount (0 to ~8+):
    - 0-2: Green zone (stable) — `bg-positive`
    - 3+: Amber zone (choppy) — `bg-choppy`
    - Marker at threshold=3 with label "Threshold"
    - Current value indicator with exact count displayed
  - **Timeline component** (below gauge): 40-day mini timeline:
    - Row of 40 small cells (each ~8px wide)
    - Color: green for ON, gray for OFF
    - Highlighted borders where flips occur
    - Label: "40일 시그널 히스토리" with FlipCount value: "N회 전환"
  - Wrap in collapsible section (default expanded when choppy, collapsed when not)

  **4d. Add cooldown indicator**:
  - When `vm.cooldownActive` is true, show subtle indicator near emergency badge:
    - "쿨다운 활성 (1일)" in amber text
    - Small clock icon

  **Must NOT do**:
  - Do NOT change the Evidence Cards grid (3-column grid with SMA160/165/170)
  - Do NOT modify Performance Strip or Settings panel
  - Do NOT change MacroStrip integration
  - Do NOT change the component's props interface (only read new fields from vm)
  - Do NOT move risk calculation logic out of this component in this task (just update thresholds)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI-heavy task — new gauge visualization, timeline component, color theming, responsive layout
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: FlipCount gauge and timeline are novel UI elements requiring design sensibility
  - **Skills Evaluated but Omitted**:
    - `playwright`: QA scenarios use Playwright but the skill is for the QA step, not implementation

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Parallel Group**: Wave 3 (with Tasks 5, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2, 3

  **References**:

  **Pattern References**:
  - `components/e03/ZoneBSignalCore.tsx:99-120` — Current verdict display logic (isOff binary). Must be replaced with 4-state config object pattern.
  - `components/e03/ZoneBSignalCore.tsx:159-220` — Current risk metrics calculation with hardcoded -7%/-20%. Update constants and labels.
  - `components/e03/ZoneBSignalCore.tsx:222-294` — Risk Monitor Banner JSX. Update threshold labels in the JSX text.
  - `components/e03/ZoneBSignalCore.tsx:296-320` — Verdict heading and state badge area where FlipCount gauge should be inserted.
  - `components/e03/ZoneBSignalCore.tsx:322-357` — Evidence Cards grid — do NOT modify.

  **API/Type References**:
  - `lib/ops/e03/types.ts:E03ViewModel` — Updated interface from Task 1 with `flipCount`, `isChoppy`, `targetTqqqWeight`, `cooldownActive`, `signalHistory`
  - `200tq/E03_SSOT.md:98-114` — F1 filter parameters (FlipWindow=40, FlipThreshold=3)
  - `200tq/E03_SSOT.md:121-131` — Emergency thresholds (-5%/-15%)

  **Documentation References**:
  - Previous session librarian research: Market Structure Color System (Green/Amber/Gray/Red maps to 4 states)
  - Previous session librarian research: gauge + timeline pattern for signal stability
  - `200tq/dashboard/E03_UI_SSOT.md` — Current UI spec (to be updated in Task 7)

  **Acceptance Criteria**:

  - [ ] 4-state verdict display: ON="매수 유지"+green, ON_CHOPPY="매수 축소"+amber, OFF10="매도 대기", EMERGENCY="비상 전환"+red+pulse
  - [ ] Risk Monitor shows "TQQQ Flash Crash (-15%)" label (not -20%)
  - [ ] Risk Monitor shows "QQQ Drop Alert (-5%)" label (not -7%)
  - [ ] FlipCount gauge visible when signalHistory present
  - [ ] FlipCount gauge shows threshold marker at 3
  - [ ] 40-day timeline renders with ON/OFF cells and flip highlights
  - [ ] Cooldown indicator shows when vm.cooldownActive=true
  - [ ] `npm run build` → 0 errors

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: ON-Normal state renders green verdict
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running on localhost:3000, MOCK mode with "fresh_normal" scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Wait for: h2 text visible (timeout: 10s)
      3. Assert: h2 text contains "매수 유지"
      4. Assert: h2 has class containing "text-positive"
      5. Assert: state badge text contains "ON"
      6. Screenshot: .sisyphus/evidence/task-4-on-normal.png
    Expected Result: Green "매수 유지" heading with ON badge
    Evidence: .sisyphus/evidence/task-4-on-normal.png

  Scenario: ON-Choppy state renders amber verdict
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, MOCK mode with "choppy_on" scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Select "choppy_on" from scenario picker
      3. Wait for: page update (timeout: 5s)
      4. Assert: h2 text contains "매수 축소"
      5. Assert: FlipCount gauge is visible
      6. Assert: gauge shows value >= 3 in amber zone
      7. Assert: 40-day timeline visible with colored cells
      8. Screenshot: .sisyphus/evidence/task-4-choppy-on.png
    Expected Result: Amber "매수 축소" heading, FlipCount gauge showing choppy zone, timeline visible
    Evidence: .sisyphus/evidence/task-4-choppy-on.png

  Scenario: Emergency state renders red pulsing verdict
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, MOCK mode with "emergency_qqq" scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Select "emergency_qqq" from scenario picker
      3. Wait for: page update (timeout: 5s)
      4. Assert: h2 text contains "비상 전환"
      5. Assert: Risk Monitor banner shows triggered state
      6. Assert: QQQ label contains "-5%" (not -7%)
      7. Screenshot: .sisyphus/evidence/task-4-emergency.png
    Expected Result: Red pulsing "비상 전환" with triggered risk monitor
    Evidence: .sisyphus/evidence/task-4-emergency.png

  Scenario: Risk Monitor thresholds updated
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, any scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Assert: page contains text "(-15%)" for TQQQ (not -20%)
      3. Assert: page contains text "(-5%)" for QQQ (not -7%)
      4. Screenshot: .sisyphus/evidence/task-4-thresholds.png
    Expected Result: Updated threshold labels
    Evidence: .sisyphus/evidence/task-4-thresholds.png
  ```

  **Commit**: YES
  - Message: `feat(e03): add FlipCount gauge/timeline and update Emergency thresholds in ZoneB`
  - Files: `components/e03/ZoneBSignalCore.tsx`
  - Pre-commit: `npm run build`

---

- [x] 5. Update ZoneAHeader: 4-state badge handling

  **What to do**:

  **5a. Update status badge logic** (lines 21-27):
  - Add choppy state detection alongside existing emergency/execution checks:
    ```typescript
    const isChoppy = vm.strategyState === "ON_CHOPPY";
    const isEmergencyState = vm.strategyState === "EMERGENCY";
    ```

  **5b. Add strategy state indicator** in the bottom row badges (lines 91-141):
  - Add a new badge between Emergency and Execution badges:
    - **ON**: Small green dot + "정상" (same as existing NONE emergency display)
    - **ON_CHOPPY**: Amber badge with waveform icon + "시그널 불안정" (FlipCount ≥3)
    - **OFF10**: Existing muted display
    - **EMERGENCY**: Merge with existing emergency badge (already handled for HARD_CONFIRMED)
  - When `vm.cooldownActive`:
    - Show "쿨다운" badge with clock icon in amber, between emergency and execution

  **5c. Update emergency display** (lines 96-110):
  - EMERGENCY strategyState should trigger the hard emergency display even if EmergencyState hasn't been explicitly set to HARD_CONFIRMED
  - Logic: `const showHardEmergency = isHardEmergency || isEmergencyState;`

  **Must NOT do**:
  - Do NOT change the top row (dates & toggles) layout
  - Do NOT modify ThemeToggle, SimMode, PrivacyMode, or ViewMode toggles
  - Do NOT change the notification bell behavior
  - Do NOT add new props to ZoneAHeaderProps (read everything from vm)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Badge UI work with conditional rendering and color theming
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Badge design for 4 states, consistent with existing badge pattern
  - **Skills Evaluated but Omitted**:
    - `playwright`: For QA only, not implementation

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Parallel Group**: Wave 3 (with Tasks 4, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2, 3

  **References**:

  **Pattern References**:
  - `components/e03/ZoneAHeader.tsx:16-28` — Current state detection logic. Add strategy state checks.
  - `components/e03/ZoneAHeader.tsx:91-141` — Bottom row badges (Emergency, Execution, Data priority order). Strategy state badge inserts into this priority chain.
  - `components/e03/ZoneAHeader.tsx:96-110` — Emergency badge rendering pattern (prominent when active, minimal dot when NONE). Follow same pattern for choppy state.

  **API/Type References**:
  - `lib/ops/e03/types.ts:E03ViewModel` — Updated with `strategyState` now including ON_CHOPPY and EMERGENCY, plus `cooldownActive` boolean.

  **Acceptance Criteria**:

  - [ ] ON_CHOPPY shows amber "시그널 불안정" badge with distinct icon
  - [ ] EMERGENCY strategyState shows hard emergency display (red, pulsing)
  - [ ] Cooldown shows "쿨다운" badge when vm.cooldownActive=true
  - [ ] Existing toggle buttons (Sim, Privacy, ViewMode, Notifications) unchanged
  - [ ] `npm run build` → 0 errors

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Choppy state badge visible in header
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, "choppy_on" scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Select "choppy_on" scenario
      3. Wait for: header badges visible (timeout: 5s)
      4. Assert: badge with text containing "시그널 불안정" or "CHOPPY" visible
      5. Assert: badge has amber/choppy coloring
      6. Screenshot: .sisyphus/evidence/task-5-choppy-badge.png
    Expected Result: Amber choppy badge visible in header
    Evidence: .sisyphus/evidence/task-5-choppy-badge.png

  Scenario: Emergency strategy state triggers hard emergency display
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, "emergency_qqq" scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Select "emergency_qqq" scenario
      3. Wait for: header badges visible (timeout: 5s)
      4. Assert: emergency badge shows red pulsing state
      5. Assert: badge text contains "비상" or "EMERGENCY"
      6. Screenshot: .sisyphus/evidence/task-5-emergency-badge.png
    Expected Result: Red emergency badge prominently displayed
    Evidence: .sisyphus/evidence/task-5-emergency-badge.png
  ```

  **Commit**: YES (groups with Task 4)
  - Message: `feat(e03): update ZoneA header for 4-state badge display`
  - Files: `components/e03/ZoneAHeader.tsx`
  - Pre-commit: `npm run build`

---

- [x] 6. Update ZoneCOpsConsole: ON-Choppy 70%/30% order calculation

  **What to do**:

  **6a. Update trade display context**:
  - The trade lines are already generated by `buildViewModel` (Task 1 handles the logic).
  - Zone C just **renders** `vm.expectedTrades` — it doesn't calculate.
  - Main change: visual indication of the Choppy state and its weight context.

  **6b. Add weight context indicator** above trade rows:
  - When `vm.strategyState === "ON_CHOPPY"`:
    - Show a subtle amber banner: "Choppy 구간: TQQQ 70% / SGOV 30%"
    - Visual: amber-left-border card with amber text
  - When `vm.strategyState === "EMERGENCY"`:
    - Show a red banner: "Emergency Exit: TQQQ 10% / SGOV 90%"
    - Visual: red-left-border card with red text, pulse animation
  - When `vm.strategyState === "ON"`:
    - Show green subtle text: "ON: TQQQ 100%"
  - When `vm.strategyState === "OFF10"`:
    - Show gray text: "OFF: TQQQ 10% / SGOV 90%"

  **6c. Update Position Impact Analysis** (lines 184-236):
  - The "Before → After" calculation already uses `vm.expectedTrades` to compute projected positions.
  - No logic change needed — it will automatically reflect 70%/30% trades from buildViewModel.
  - Add weight percentage labels to the "After" column for clarity:
    - Show the target weight percentage next to quantities (e.g., "TQQQ 350 (70%)")

  **6d. Update Risk Line** (lines 158-181):
  - Change `STOP(-20%)` label to `STOP(-15%)` to match new threshold
  - Update `stopLossDist` display if it references the old -20% value

  **Must NOT do**:
  - Do NOT change trade calculation logic (already in buildViewModel)
  - Do NOT modify RecordModal component
  - Do NOT change copy-to-clipboard behavior
  - Do NOT change CSV download functionality
  - Do NOT alter the Reality (기록) panel on the left side

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI changes to trade display with state-aware contextual banners
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Weight context banner design for trading interface
  - **Skills Evaluated but Omitted**:
    - `playwright`: For QA only

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Parallel Group**: Wave 3 (with Tasks 4, 5)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2, 3

  **References**:

  **Pattern References**:
  - `components/e03/ZoneCOpsConsole.tsx:45-48` — Current NO_ACTION detection. May need update for EMERGENCY/COOLDOWN states.
  - `components/e03/ZoneCOpsConsole.tsx:149-237` — Strategy panel with Position Impact Analysis. Weight labels should be added to "After" column.
  - `components/e03/ZoneCOpsConsole.tsx:158-181` — Risk line showing STOP(-20%). Update to -15%.
  - `components/e03/ZoneCOpsConsole.tsx:239-252` — Trade rows area. Weight context banner inserts above this.

  **API/Type References**:
  - `lib/ops/e03/types.ts:E03ViewModel` — `strategyState` (4 states), `targetTqqqWeight`, `expectedTrades` (already populated by buildViewModel)
  - `200tq/E03_SSOT.md:133-141` — Position allocation table

  **Acceptance Criteria**:

  - [ ] ON_CHOPPY scenario shows amber "Choppy 구간: TQQQ 70% / SGOV 30%" banner
  - [ ] EMERGENCY scenario shows red "Emergency Exit" banner
  - [ ] ON scenario shows green "ON: TQQQ 100%" subtle text
  - [ ] Risk line shows "STOP(-15%)" not "STOP(-20%)"
  - [ ] Position Impact "After" column shows weight percentages
  - [ ] Trade rows correctly display 70%/30% trades for choppy scenario
  - [ ] `npm run build` → 0 errors

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Choppy state shows weight context banner
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, "choppy_on" scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Select "choppy_on" scenario
      3. Wait for: strategy panel visible (timeout: 5s)
      4. Assert: amber banner text contains "Choppy" and "70%"
      5. Assert: trade rows show SELL TQQQ and BUY SGOV entries
      6. Screenshot: .sisyphus/evidence/task-6-choppy-trades.png
    Expected Result: Amber context banner with correct weight info, trade lines for 70/30
    Evidence: .sisyphus/evidence/task-6-choppy-trades.png

  Scenario: Emergency state shows exit banner
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, "emergency_qqq" scenario
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Select "emergency_qqq" scenario
      3. Wait for: strategy panel visible (timeout: 5s)
      4. Assert: red banner text contains "Emergency" and "10%"
      5. Screenshot: .sisyphus/evidence/task-6-emergency-trades.png
    Expected Result: Red context banner with emergency exit weight info
    Evidence: .sisyphus/evidence/task-6-emergency-trades.png

  Scenario: Risk line shows updated threshold
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running, any ON scenario with portfolio
    Steps:
      1. Navigate to: http://localhost:3000/command
      2. Assert: page contains text "STOP(-15%)" or "-15%"
      3. Assert: page does NOT contain text "STOP(-20%)"
      4. Screenshot: .sisyphus/evidence/task-6-risk-line.png
    Expected Result: Updated -15% threshold in risk line
    Evidence: .sisyphus/evidence/task-6-risk-line.png
  ```

  **Commit**: YES (groups with Tasks 4, 5)
  - Message: `feat(e03): add weight context banners and update thresholds in ZoneC`
  - Files: `components/e03/ZoneCOpsConsole.tsx`
  - Pre-commit: `npm run build`

---

- [x] 7. Update all 4 dashboard SSOT documents to v2026.3

  **What to do**:

  Update each of the 4 dashboard SSOT documents to reflect v2026.3 changes. Read each document first, then perform surgical edits.

  **7a. `E03_Command_Center_SSOT_v2.md`** (functional spec):
  - Update version references from v2026.1 to v2026.3
  - Add F1 Signal Stability Filter section
  - Update StrategyState descriptions: 2 states → 4 states
  - Update Emergency thresholds: -7%/-20% → -5%/-15%
  - Add FlipCount data requirements
  - Add cooldown behavior description
  - Update position allocation rules: add 70%/30% for ON-Choppy

  **7b. `E03_UI_SSOT.md`** (UI spec):
  - Add FlipCount gauge and timeline components specification
  - Update verdict display to document 4 states (label, color, icon for each)
  - Update Risk Monitor thresholds in spec
  - Add cooldown indicator specification
  - Add weight context banner spec for Zone C

  **7c. `E03_UX_SSOT.md`** (UX spec):
  - Add user flow for choppy state (what user sees, what it means, what to do)
  - Add user flow for emergency state (trigger → cooldown → recovery)
  - Update decision tree: 2-branch → 4-branch
  - Add FlipCount timeline interaction description (collapsed/expanded)

  **7d. `E03_DESIGN_SSOT.md`** (design system):
  - Add amber/choppy color token documentation
  - Add FlipCount gauge component specification (dimensions, colors, threshold marker)
  - Add timeline component specification (cell size, colors, gap between cells)
  - Add weight context banner component specification
  - Update state → color mapping table: 4 states

  **Must NOT do**:
  - Do NOT create new SSOT documents (update existing 4 only)
  - Do NOT change the fundamental document structure
  - Do NOT add implementation details (code snippets) to SSOT docs
  - Do NOT remove historical version information from documents

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Technical documentation update — requires reading existing docs and making precise edits while preserving structure
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - All technical skills: This is pure documentation work

  **Parallelization**:
  - **Can Run In Parallel**: NO (Sequential — final task)
  - **Parallel Group**: Wave 4 (solo, final)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 1-6 (needs to reflect actual implementation)

  **References**:

  **Pattern References**:
  - `200tq/dashboard/E03_Command_Center_SSOT_v2.md` — 432 lines, functional spec. Find all "ON"/"OFF10" references and update.
  - `200tq/dashboard/E03_UI_SSOT.md` — 401 lines, UI spec. Find component specs for Zone B and update.
  - `200tq/dashboard/E03_UX_SSOT.md` — 300 lines, UX spec. Find user flows and decision trees to update.
  - `200tq/dashboard/E03_DESIGN_SSOT.md` — 738 lines, design system. Find color tables and component specs to update.

  **Documentation References**:
  - `200tq/E03_SSOT.md` — THE authoritative strategy SSOT v2026.3. All dashboard docs must align with this.
  - `200tq/E03_SSOT.md:92-141` — F1 filter + Emergency + allocation table (primary source for updates)

  **Acceptance Criteria**:

  - [ ] All 4 docs reference "v2026.3" (not v2026.1)
  - [ ] All 4 docs describe 4 strategy states (ON, ON-Choppy, OFF, Emergency)
  - [ ] Command Center SSOT includes F1 filter section and -5%/-15% thresholds
  - [ ] UI SSOT includes FlipCount gauge and timeline component specs
  - [ ] UX SSOT includes choppy and emergency user flows
  - [ ] Design SSOT includes amber/choppy color tokens and new component specs
  - [ ] No references to old -7%/-20% thresholds remain in any doc
  - [ ] `grep -r "\-7%" 200tq/dashboard/E03_*.md` → 0 matches
  - [ ] `grep -r "\-20%" 200tq/dashboard/E03_*.md` → 0 matches (context-dependent, may still appear in historical sections)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: No old thresholds remain in SSOT docs
    Tool: Bash
    Steps:
      1. Run: grep -rn "\-7%" 200tq/dashboard/E03_*.md
      2. Assert: 0 matches (or only in version history sections)
      3. Run: grep -rn "v2026.1" 200tq/dashboard/E03_*.md
      4. Assert: only appears in version history, not in current spec sections
      5. Run: grep -rn "ON_CHOPPY\|ON-Choppy\|choppy" 200tq/dashboard/E03_*.md
      6. Assert: matches found in all 4 documents
    Expected Result: All docs updated, no stale references
    Evidence: grep output captured

  Scenario: All 4 docs mention v2026.3
    Tool: Bash
    Steps:
      1. Run: grep -l "v2026.3" 200tq/dashboard/E03_*.md
      2. Assert: 4 files listed
    Expected Result: All 4 SSOT docs reference v2026.3
    Evidence: grep output captured
  ```

  **Commit**: YES
  - Message: `docs(e03): sync all 4 dashboard SSOT documents to v2026.3`
  - Files: `200tq/dashboard/E03_Command_Center_SSOT_v2.md`, `200tq/dashboard/E03_UI_SSOT.md`, `200tq/dashboard/E03_UX_SSOT.md`, `200tq/dashboard/E03_DESIGN_SSOT.md`
  - Pre-commit: none (docs only)

---

## Commit Strategy

| After Task(s) | Message | Key Files | Verification |
|---------------|---------|-----------|--------------|
| 1 | `feat(e03): implement F1 Signal Stability Filter and 4-state strategy logic with TDD` | types.ts, buildViewModel.ts, buildViewModel.test.ts | `npx vitest run` |
| 2 | `feat(e03): add v2026.3 mock scenarios and dataSource passthrough` | mock.ts, dataSource.ts | `npx vitest run && npm run build` |
| 3 | `style(e03): add choppy/amber color tokens for ON-Choppy state` | globals.css, tailwind.config.ts | `npm run build` |
| 4 | `feat(e03): add FlipCount gauge/timeline and update Emergency thresholds in ZoneB` | ZoneBSignalCore.tsx | `npm run build` |
| 5 | `feat(e03): update ZoneA header for 4-state badge display` | ZoneAHeader.tsx | `npm run build` |
| 6 | `feat(e03): add weight context banners and update thresholds in ZoneC` | ZoneCOpsConsole.tsx | `npm run build` |
| 7 | `docs(e03): sync all 4 dashboard SSOT documents to v2026.3` | E03_*.md (4 files) | grep verification |

---

## Success Criteria

### Verification Commands
```bash
# From 200tq/dashboard/
npx vitest run                          # Expected: all tests pass
npm run build                           # Expected: 0 errors
grep -rn "ON_CHOPPY" lib/ops/e03/       # Expected: matches in types.ts, buildViewModel.ts, mock.ts
grep -rn "\-5%" components/e03/ZoneBSignalCore.tsx   # Expected: matches (new threshold)
grep -rn "\-7%" components/e03/ZoneBSignalCore.tsx   # Expected: 0 matches (old threshold removed)
grep -l "v2026.3" E03_*.md              # Expected: 4 files
```

### Final Checklist
- [x] All "Must Have" present (4 states, F1 filter, Emergency -5%/-15%, cooldown, state priority)
- [x] All "Must NOT Have" absent (no layout changes, no new state management, no real API)
- [x] All vitest tests pass
- [x] Build succeeds
- [x] All 4 SSOT docs updated
- [x] FlipCount gauge renders in Zone B
- [x] 4 states visually distinct
- [x] Backward compatible: missing signalHistory defaults to ON/OFF10 behavior
