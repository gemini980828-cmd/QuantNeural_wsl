# Dashboard View Modes Implementation

## TL;DR

> **Quick Summary**: Implement Simple/Pro mode toggle for the 200TQ dashboard. Simple Mode provides a minimal Robinhood-style view (total assets, action verdict, condensed macro indicators), while Pro Mode shows the full current dashboard.
> 
> **Deliverables**:
> - `viewMode` setting in Zustand store with localStorage persistence
> - Mode toggle button in ZoneAHeader
> - New SimpleView component for Simple Mode layout
> - Condensed MacroStrip variant (VIX, F&G, NQ only)
> - Conditional risk banner (shown only when triggered)
> - Smooth CSS transition animations between modes
> 
> **Estimated Effort**: Medium (4-6 hours)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Tasks 3,4,5 (parallel) → Task 6

---

## Context

### Original Request
Implement two dashboard modes for the 200TQ investment dashboard:
1. **Simple Mode** (Default) - Minimal, Robinhood-style
2. **Pro Mode** (Current Full Dashboard)

### Research Findings

**Architecture Discovery**:
- Zone-based component structure (ZoneA-D) with centralized ViewModel in CommandPage
- Zustand store handles persistence with `persist` middleware
- Existing Toggle pattern available in `settings/page.tsx`
- `tailwindcss-animate` available for transition animations
- Risk metrics (`isAnyTriggered`, `isWarning`) available in `ZoneBSignalCore`
- Portfolio data available via `vm.portfolio.derived`

**Relevant Files Examined**:
- `app/(shell)/command/page.tsx` - Main dashboard controller
- `components/e03/ZoneAHeader.tsx` - Header with toggles (142 lines)
- `components/e03/ZoneBSignalCore.tsx` - Signal display, risk metrics (462 lines)
- `components/e03/MacroStrip.tsx` - Macro indicators (177 lines)
- `components/portfolio/PortfolioSummaryStrip.tsx` - Portfolio display (142 lines)
- `lib/stores/settings-store.ts` - Zustand settings (123 lines)
- `lib/ops/e03/types.ts` - ViewModel types (106 lines)

### Gap Analysis (Self-Performed)

**Identified Gaps Addressed**:
1. **Hydration handling**: Use `_hasHydrated` pattern already in settings-store
2. **Default mode**: User specified "Simple Mode (Default)" - will set as default value
3. **URL persistence**: Not needed - localStorage is sufficient per requirements
4. **Next execution time display**: Available via `vm.executionBadge` and `vm.executionState`

---

## Work Objectives

### Core Objective
Add view mode switching to the 200TQ dashboard with Simple Mode as default, providing a minimal interface for quick decision validation.

### Concrete Deliverables
1. `viewMode: 'simple' | 'pro'` in Zustand settings-store
2. `useViewMode()` selector hook
3. Mode toggle button in ZoneAHeader
4. `SimpleView.tsx` component with hero layout
5. Condensed `MacroStrip` with `condensed` prop
6. Conditional risk banner in SimpleView
7. CSS transition animations for mode switching

### Definition of Done
- [ ] Toggle between Simple/Pro mode persists across browser sessions
- [ ] Simple Mode shows: total assets, daily change, action verdict, next execution, mini macro (3 indicators)
- [ ] Risk alerts appear in Simple Mode ONLY when `isAnyTriggered || isWarning` is true
- [ ] "상세 보기" / "Pro Mode" button switches to Pro Mode
- [ ] Animations are smooth (no jarring layout shifts)
- [ ] Both modes work on mobile (responsive)

### Must Have
- View mode state persistence in localStorage
- Seamless data sharing (no additional API calls when switching modes)
- Risk alert visibility when triggered in Simple Mode

### Must NOT Have (Guardrails)
- NO framer-motion installation (use existing tailwindcss-animate)
- NO duplicate data fetching between modes
- NO new context providers (use existing Zustand pattern)
- NO modification to the ViewModel structure
- NO changes to existing component props signatures (only additions)
- NO breaking changes to Pro Mode appearance

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (no test framework detected in package.json)
- **User wants tests**: Manual-only (not specified)
- **Framework**: none

### Automated Verification (ALWAYS include)

Each TODO includes executable verification procedures:

**For Frontend/UI changes** (using playwright skill):
- Navigate to dashboard, verify mode toggle exists
- Click toggle, verify mode changes
- Refresh page, verify mode persists
- Check responsive behavior on mobile viewport

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: Add viewMode to Zustand store [no dependencies]

Wave 2 (After Wave 1):
├── Task 2: Add MacroStrip condensed prop [depends: 1 for testing]
├── Task 3: Create SimpleView component [depends: 1]
└── Task 4: Add toggle to ZoneAHeader [depends: 1]

Wave 3 (After Wave 2):
└── Task 5: Integrate in command/page.tsx [depends: 2, 3, 4]

Wave 4 (After Wave 3):
└── Task 6: Add transition animations [depends: 5]

Critical Path: Task 1 → Task 3 → Task 5 → Task 6
Parallel Speedup: ~30% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4 | None (foundational) |
| 2 | 1 | 5 | 3, 4 |
| 3 | 1 | 5 | 2, 4 |
| 4 | 1 | 5 | 2, 3 |
| 5 | 2, 3, 4 | 6 | None (integration) |
| 6 | 5 | None | None (final polish) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Approach |
|------|-------|---------------------|
| 1 | 1 | Single focused task |
| 2 | 2, 3, 4 | Parallel execution possible |
| 3 | 5 | Integration - sequential |
| 4 | 6 | Final polish - sequential |

---

## TODOs

### Task 1: Add viewMode to Zustand Settings Store

**What to do**:
- Add `viewMode: 'simple' | 'pro'` type to `AppSettings` interface
- Add `viewMode: 'simple'` as default value in `defaultSettings`
- Add `useViewMode()` selector hook following existing pattern (`useSimulationMode`, etc.)

**Must NOT do**:
- Do not change any existing setting keys or defaults
- Do not modify the store structure beyond adding the new field

**Recommended Agent Profile**:
- **Category**: `quick`
  - Reason: Single file modification with clear pattern to follow
- **Skills**: None required
  - Existing patterns are clear in the file

**Parallelization**:
- **Can Run In Parallel**: NO (foundational)
- **Parallel Group**: Wave 1 (alone)
- **Blocks**: Tasks 2, 3, 4, 5, 6
- **Blocked By**: None

**References**:

**Pattern References**:
- `lib/stores/settings-store.ts:43-64` - Existing `defaultSettings` object showing how to add new settings
- `lib/stores/settings-store.ts:89-97` - Existing selector hooks pattern (`useSimulationMode`, `useDataSource`)

**Type References**:
- `lib/stores/settings-store.ts:5-31` - `AppSettings` interface where `viewMode` should be added

**Acceptance Criteria**:

```bash
# Agent runs:
grep -n "viewMode" /home/juwon/QuantNeural_wsl/200tq/dashboard/lib/stores/settings-store.ts
# Assert: Shows viewMode in AppSettings interface AND defaultSettings AND useViewMode hook

grep "viewMode.*simple" /home/juwon/QuantNeural_wsl/200tq/dashboard/lib/stores/settings-store.ts
# Assert: Default value is 'simple'
```

**For Frontend verification** (using playwright skill):
```
1. Navigate to: http://localhost:3000/command
2. Open DevTools Console
3. Execute: localStorage.getItem('200tq-settings')
4. Assert: JSON contains "viewMode" key
```

**Commit**: YES
- Message: `feat(settings): add viewMode setting for simple/pro dashboard toggle`
- Files: `lib/stores/settings-store.ts`
- Pre-commit: `npm run build` (type check)

---

### Task 2: Add Condensed Mode to MacroStrip

**What to do**:
- Add optional `condensed?: boolean` prop to `MacroStripProps` interface
- When `condensed` is true, show only: VIX, F&G, NQ (hide 10Y, DXY, 환율)
- Keep existing full display as default (`condensed` defaults to false)

**Must NOT do**:
- Do not change the styling or layout of the full MacroStrip
- Do not modify the MacroData interface
- Do not change how data is fetched or passed

**Recommended Agent Profile**:
- **Category**: `quick`
  - Reason: Single component prop addition with conditional rendering
- **Skills**: None required

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 2 (with Tasks 3, 4)
- **Blocks**: Task 5
- **Blocked By**: Task 1

**References**:

**Pattern References**:
- `components/e03/MacroStrip.tsx:88-175` - Current full render showing all 6 indicators

**Acceptance Criteria**:

```bash
# Agent runs:
grep -n "condensed" /home/juwon/QuantNeural_wsl/200tq/dashboard/components/e03/MacroStrip.tsx
# Assert: Shows condensed prop in interface and conditional rendering logic
```

**For Frontend verification** (using playwright skill):
```
1. Temporarily modify command/page.tsx to pass condensed={true} to MacroStrip
2. Navigate to: http://localhost:3000/command
3. Assert: Only VIX, F&G, NQ indicators visible
4. Assert: 10Y, DXY, 환율 are NOT visible
5. Revert temporary change
```

**Commit**: YES (groups with Task 3, 4)
- Message: `feat(macro): add condensed mode showing VIX, F&G, NQ only`
- Files: `components/e03/MacroStrip.tsx`
- Pre-commit: `npm run build`

---

### Task 3: Create SimpleView Component

**What to do**:
- Create new file `components/e03/SimpleView.tsx`
- Accept props: `vm: E03ViewModel`, `macroData`, `macroLoading`, `riskMetrics`, `onSwitchToPro: () => void`
- Implement layout:
  1. **Hero Section**: Total assets (large, centered) + daily change percentage
  2. **Action Card**: Verdict label ("매수 유지" or "매도 대기") + state badge (ON/OFF) + next execution time
  3. **Mini Macro Strip**: Use MacroStrip with `condensed={true}`
  4. **Risk Banner**: Only shown when `riskMetrics.isAnyTriggered || riskMetrics.isWarning`
  5. **Pro Mode Button**: "상세 보기" button at bottom

**Must NOT do**:
- Do not duplicate data fetching (receive all data as props)
- Do not create new API calls
- Do not duplicate formatters (import from existing files)

**Recommended Agent Profile**:
- **Category**: `visual-engineering`
  - Reason: New UI component requiring layout design and styling
- **Skills**: [`frontend-ui-ux`]
  - Reason: Creating a Robinhood-style hero layout requires UI design sense

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 2 (with Tasks 2, 4)
- **Blocks**: Task 5
- **Blocked By**: Task 1

**References**:

**Pattern References**:
- `components/portfolio/PortfolioSummaryStrip.tsx:31-58` - Pattern for displaying total assets with formatting
- `components/e03/ZoneBSignalCore.tsx:296-320` - Pattern for action label display ("매수 유지"/"매도 대기")
- `components/e03/ZoneBSignalCore.tsx:222-294` - Risk banner pattern with conditional styling

**Type References**:
- `lib/ops/e03/types.ts:33-67` - `E03ViewModel` interface
- `lib/ops/e03/types.ts:96-105` - `PortfolioSnapshot` interface

**External References**:
- Robinhood app screenshots for minimal financial dashboard inspiration

**Acceptance Criteria**:

```bash
# Agent runs:
ls -la /home/juwon/QuantNeural_wsl/200tq/dashboard/components/e03/SimpleView.tsx
# Assert: File exists

grep -c "totalEquity\|dailyPnL" /home/juwon/QuantNeural_wsl/200tq/dashboard/components/e03/SimpleView.tsx
# Assert: Count > 0 (uses portfolio data)

grep -c "매수 유지\|매도 대기" /home/juwon/QuantNeural_wsl/200tq/dashboard/components/e03/SimpleView.tsx
# Assert: Count > 0 (shows action verdict)

grep -c "상세 보기" /home/juwon/QuantNeural_wsl/200tq/dashboard/components/e03/SimpleView.tsx
# Assert: Count > 0 (has Pro Mode button)
```

**Commit**: YES (groups with Task 2, 4)
- Message: `feat(dashboard): create SimpleView component for minimal mode`
- Files: `components/e03/SimpleView.tsx`
- Pre-commit: `npm run build`

---

### Task 4: Add View Mode Toggle to ZoneAHeader

**What to do**:
- Import `useViewMode` and `useSettingsStore` from settings-store
- Add toggle button for view mode in the top row with other toggles (Sim, Privacy, Notifications)
- Use icon: `LayoutDashboard` (lucide-react) for Pro mode, `Minimize2` for Simple mode
- Button style: Follow existing toggle pattern (bg-blue-900/40 when active)
- On click: Toggle between 'simple' and 'pro'

**Must NOT do**:
- Do not change the existing toggle button positions or styles
- Do not modify the ZoneAHeaderProps interface (viewMode comes from Zustand, not props)

**Recommended Agent Profile**:
- **Category**: `quick`
  - Reason: Adding button to existing UI following clear pattern
- **Skills**: None required

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 2 (with Tasks 2, 3)
- **Blocks**: Task 5
- **Blocked By**: Task 1

**References**:

**Pattern References**:
- `components/e03/ZoneAHeader.tsx:38-55` - Existing toggle buttons (Sim, Privacy) pattern
- `app/(shell)/settings/page.tsx:12-25` - Toggle button implementation pattern

**Acceptance Criteria**:

```bash
# Agent runs:
grep -n "useViewMode\|viewMode" /home/juwon/QuantNeural_wsl/200tq/dashboard/components/e03/ZoneAHeader.tsx
# Assert: Shows import and usage of viewMode

grep -n "LayoutDashboard\|Minimize2" /home/juwon/QuantNeural_wsl/200tq/dashboard/components/e03/ZoneAHeader.tsx
# Assert: Shows icon imports for toggle
```

**For Frontend verification** (using playwright skill):
```
1. Navigate to: http://localhost:3000/command
2. Locate toggle button with view mode icon in header
3. Assert: Button is visible
4. Click the button
5. Assert: localStorage viewMode value changes
```

**Commit**: YES (groups with Task 2, 3)
- Message: `feat(header): add view mode toggle button`
- Files: `components/e03/ZoneAHeader.tsx`
- Pre-commit: `npm run build`

---

### Task 5: Integrate View Modes in CommandPage

**What to do**:
- Import `useViewMode` from settings-store
- Import `SimpleView` component
- Pass `riskMetrics` calculation result to SimpleView (extract from ZoneBSignalCore or compute at page level)
- Conditional render:
  - If `viewMode === 'simple'`: Render only ZoneAHeader + SimpleView
  - If `viewMode === 'pro'`: Render current full layout (all Zones)
- Handle hydration: Don't render mode-specific content until `_hasHydrated` is true

**Must NOT do**:
- Do not remove or modify the Pro Mode (current) layout
- Do not add duplicate data fetching
- Do not change how macroData is fetched (keep in ZoneBSignalCore or lift up)

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
  - Reason: Integration task requiring careful coordination of state and rendering
- **Skills**: None required
  - Pattern is straightforward once components are built

**Parallelization**:
- **Can Run In Parallel**: NO (integration)
- **Parallel Group**: Wave 3 (sequential)
- **Blocks**: Task 6
- **Blocked By**: Tasks 2, 3, 4

**References**:

**Pattern References**:
- `app/(shell)/command/page.tsx:250-259` - Existing loading state handling pattern
- `app/(shell)/command/page.tsx:261-377` - Current full dashboard layout to conditionally render
- `lib/stores/settings-store.ts:81-84` - Hydration handling with `_hasHydrated`

**Implementation Note**:
The macroData fetch currently happens inside ZoneBSignalCore. For SimpleView to access it, either:
1. Lift the fetch to CommandPage level (cleaner but more changes), or
2. Have SimpleView render MacroStrip which fetches its own data (simpler, current pattern)

Recommended: Option 2 - keep macroData fetch inside MacroStrip (already works this way)

**Acceptance Criteria**:

**For Frontend verification** (using playwright skill):
```
1. Navigate to: http://localhost:3000/command
2. Assert: Simple Mode displayed by default (hero section visible)
3. Click view mode toggle in header
4. Assert: Pro Mode displayed (all Zones visible)
5. Refresh page
6. Assert: Pro Mode persists (mode was saved)
7. Click toggle again
8. Assert: Simple Mode displayed
9. Check mobile viewport (375px width)
10. Assert: Layout is responsive in both modes
```

**Commit**: YES
- Message: `feat(dashboard): integrate simple/pro mode conditional rendering`
- Files: `app/(shell)/command/page.tsx`
- Pre-commit: `npm run build`

---

### Task 6: Add Transition Animations

**What to do**:
- Add CSS transition classes for mode switching
- Use `opacity` transition for smooth fade (150-200ms)
- Apply `transition-opacity duration-200 ease-in-out` to mode containers
- Consider using `AnimatePresence`-like pattern with conditional rendering:
  - Wrap SimpleView and ProView in transition containers
  - Use `opacity-0` / `opacity-100` with transitions

**Must NOT do**:
- Do not install framer-motion
- Do not add complex animations that affect performance
- Do not use transform animations that cause layout shifts

**Recommended Agent Profile**:
- **Category**: `quick`
  - Reason: Adding Tailwind classes for transitions
- **Skills**: [`frontend-ui-ux`]
  - Reason: Animation timing and easing should feel polished

**Parallelization**:
- **Can Run In Parallel**: NO (final polish)
- **Parallel Group**: Wave 4 (sequential)
- **Blocks**: None
- **Blocked By**: Task 5

**References**:

**Pattern References**:
- `tailwind.config.ts` - Check if custom transitions are defined
- Tailwind docs for transition utilities

**Acceptance Criteria**:

**For Frontend verification** (using playwright skill):
```
1. Navigate to: http://localhost:3000/command
2. Record video of mode toggle
3. Assert: Transition is visible (not instant snap)
4. Assert: Transition duration ~200ms (subjective visual check)
5. Assert: No layout shift or jank during transition
```

**Commit**: YES
- Message: `feat(dashboard): add smooth transition animations for mode switching`
- Files: `app/(shell)/command/page.tsx`
- Pre-commit: `npm run build`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(settings): add viewMode setting for simple/pro dashboard toggle` | settings-store.ts | `npm run build` |
| 2, 3, 4 | `feat(dashboard): implement simple mode components` | MacroStrip.tsx, SimpleView.tsx, ZoneAHeader.tsx | `npm run build` |
| 5 | `feat(dashboard): integrate simple/pro mode conditional rendering` | command/page.tsx | `npm run build` |
| 6 | `feat(dashboard): add smooth transition animations for mode switching` | command/page.tsx | `npm run build` |

---

## Success Criteria

### Verification Commands
```bash
# Type check passes
cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run build

# Required files exist
ls -la components/e03/SimpleView.tsx

# Settings store updated
grep -c "viewMode" lib/stores/settings-store.ts  # Expected: >= 3
```

### Final Checklist
- [ ] View mode toggle visible in ZoneAHeader
- [ ] Simple Mode is the default on first visit
- [ ] Simple Mode shows: total assets, daily change, verdict, next execution
- [ ] Simple Mode shows condensed macro strip (VIX, F&G, NQ only)
- [ ] Risk banner appears in Simple Mode only when triggered
- [ ] "상세 보기" button switches to Pro Mode
- [ ] Pro Mode shows full current dashboard (unchanged)
- [ ] Mode persists across browser sessions
- [ ] Transitions are smooth
- [ ] Both modes are mobile-responsive
