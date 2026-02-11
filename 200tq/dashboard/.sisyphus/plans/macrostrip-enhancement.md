# MacroStrip Enhancement: Previous Day Comparison + UI Refinements

## TL;DR

> **Quick Summary**: Enhance the MacroStrip component to show previous day change % for all 6 indicators, replace VIX/F&G StatusBadge with colored circle + yellow text, and reduce padding for a more compact layout.
> 
> **Deliverables**:
> - API returns `change` field for all 6 indicators
> - VIX/F&G display: colored circle (8px) + lime text value
> - Change % row below each indicator value with semantic coloring
> - Reduced padding (py-4 → py-2, p-3 → p-2)
> 
> **Estimated Effort**: Short (2-3 hours)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (API) → Task 3 (Component UI)

---

## Context

### Original Request
Modify MacroStrip component with:
1. VIX and F&G: Change from colored badge to yellow text with colored circle indicator
2. Reduce section padding (too much whitespace currently)
3. Add previous day comparison (change %) for ALL 6 indicators

### Interview Summary
**Key Discussions**:
- Circle position: Before value (● 15.2)
- Change % display: Below value in smaller text
- F&G change: Point change (e.g., +5pts), not percentage
- No tests: Manual browser verification

**Research Findings**:
- Yahoo Finance v8 Chart: Use `meta.chartPreviousClose` with `meta.previousClose` fallback
- Fear & Greed API: Supports `?limit=2` returning [today, yesterday]
- Semantic coloring per indicator (VIX up = red, NQ up = green)

### Metis Review
**Identified Gaps** (addressed):
- Change % directional coloring: Resolved with semantic per-indicator logic
- F&G change type: Resolved as point change
- Duplicate MacroData interface: Will sync both files
- Circle color mapping: Use existing ok/action/danger colors from StatusBadge

---

## Work Objectives

### Core Objective
Add previous day comparison data to all 6 macro indicators and refine the visual presentation of VIX/F&G indicators with reduced padding.

### Concrete Deliverables
- Modified `/app/api/macro/route.ts` returning `change` for all indicators
- Updated `/components/e03/MacroStrip.tsx` with new visual design
- Synced `MacroData` type in `/components/e03/ZoneBSignalCore.tsx`

### Definition of Done
- [ ] All 6 indicators show current value + change % below
- [ ] VIX/F&G use 8px colored circle + lime text (no StatusBadge)
- [ ] Container padding: py-2 (was py-4)
- [ ] Card padding: p-2 (was p-3)
- [ ] Change colors: semantic per indicator (VIX up = red, NQ up = green)
- [ ] Dashboard loads without errors at http://localhost:3000

### Must Have
- Previous close data from Yahoo Finance API (`chartPreviousClose`)
- F&G previous day via `?limit=2` parameter
- Fallback to `null` change if previous data unavailable
- Semantic coloring logic for change %

### Must NOT Have (Guardrails)
- NO trend arrows or icons (text only for change %)
- NO tooltip/hover states for previous close values
- NO refactoring of IndicatorCard into separate file
- NO changes to StatusBadge component itself
- NO mobile-specific breakpoint changes (use existing responsive grid)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO
- **User wants tests**: NO (manual verification)
- **Framework**: None

### Automated Verification (Agent-Executable)

Each TODO includes verification via Playwright browser automation:

**Verification Tool**: Playwright skill for browser automation

**Evidence Requirements**:
- Screenshots saved to `.sisyphus/evidence/`
- DOM assertions for expected elements
- No console errors in browser

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Update API route.ts (fetch previous close data)
└── Task 2: Update MacroData type in ZoneBSignalCore.tsx

Wave 2 (After Wave 1):
└── Task 3: Update MacroStrip.tsx UI (depends on API + Type)

Wave 3 (After Wave 2):
└── Task 4: Browser verification and cleanup

Critical Path: Task 1 → Task 3 → Task 4
Parallel Speedup: ~30% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3 | 2 |
| 2 | None | 3 | 1 |
| 3 | 1, 2 | 4 | None |
| 4 | 3 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | Two parallel quick agents |
| 2 | 3 | Single visual-engineering agent |
| 3 | 4 | Playwright verification |

---

## TODOs

### Task 1: Update API to Return Previous Close and Change %

- [ ] 1. Modify macro API route to include previous close and change data

  **What to do**:
  - Update `fetchYahooQuote` to return `{ value, prevClose, change }` instead of just `value`
  - Extract `meta.chartPreviousClose` with fallback to `meta.previousClose`
  - Calculate change: `((value - prevClose) / prevClose) * 100`
  - Update `fetchVix`, `fetchTreasury10Y`, `fetchDXY`, `fetchNasdaqFutures`, `fetchUsdKrw` to use new structure
  - Update `fetchFearGreed` to use `?limit=2` and calculate point change
  - Update `MacroData` interface to include `change: number | null` for each indicator

  **Must NOT do**:
  - Do NOT add any caching logic
  - Do NOT change the API endpoint path
  - Do NOT add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Focused API modification with clear scope
  - **Skills**: None required
    - API changes are straightforward TypeScript
  - **Skills Evaluated but Omitted**:
    - `git-master`: Not needed until commit phase

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `app/api/macro/route.ts:31-45` - Current `fetchYahooQuote` implementation
  - `app/api/macro/route.ts:47-53` - Current `fetchVix` structure to extend
  - `app/api/macro/route.ts:55-73` - Current `fetchFearGreed` with API call pattern

  **API/Type References**:
  - `app/api/macro/route.ts:8-16` - Current `MacroData` interface to update

  **External References**:
  - Yahoo Finance v8 response: `data.chart.result[0].meta.chartPreviousClose`
  - F&G API: `https://api.alternative.me/fng/?limit=2` returns `data[0]` (today), `data[1]` (yesterday)

  **WHY Each Reference Matters**:
  - `fetchYahooQuote:31-45`: This is the core function to modify - need to extend return type and extract prevClose from meta
  - `MacroData:8-16`: Interface defines API response shape - must add `change` field to each indicator

  **Acceptance Criteria**:

  **Automated Verification (curl via Bash)**:
  ```bash
  # Start dev server if not running, then test API
  curl -s http://localhost:3000/api/macro | jq '.vix | keys'
  # Assert: Returns ["change", "color", "value"]
  
  curl -s http://localhost:3000/api/macro | jq '.nq | keys'
  # Assert: Returns ["change", "value"]
  
  curl -s http://localhost:3000/api/macro | jq '.fng.change'
  # Assert: Returns a number (point change) or null
  ```

  **Evidence to Capture**:
  - [ ] Terminal output showing all 6 indicators have `change` field
  - [ ] Sample API response JSON

  **Commit**: YES
  - Message: `feat(api): add previous close and change % to macro indicators`
  - Files: `app/api/macro/route.ts`
  - Pre-commit: `curl localhost:3000/api/macro | jq .`

---

### Task 2: Sync MacroData Type in ZoneBSignalCore

- [ ] 2. Update MacroData interface in ZoneBSignalCore.tsx to match API changes

  **What to do**:
  - Add `change: number | null` to `vix`, `treasury`, `dxy`, `nq`, `usdkrw` objects
  - Add `change: number | null` to `fng` object (point change, not percentage)
  - Ensure interface matches exactly with route.ts definition

  **Must NOT do**:
  - Do NOT modify any component logic yet
  - Do NOT change any JSX
  - Do NOT touch the macro fetch useEffect

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple type definition update
  - **Skills**: None required
  - **Skills Evaluated but Omitted**:
    - None - trivial task

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `components/e03/ZoneBSignalCore.tsx:10-20` - Current `MacroData` interface (duplicate of route.ts)
  - `components/e03/MacroStrip.tsx:7-17` - Another copy of `MacroData` interface

  **WHY Each Reference Matters**:
  - `ZoneBSignalCore.tsx:10-20`: Primary consumer of MacroData - type must match API response
  - `MacroStrip.tsx:7-17`: Also has the interface - both must stay in sync

  **Acceptance Criteria**:

  **Automated Verification (TypeScript check via Bash)**:
  ```bash
  # Run TypeScript compiler to verify no type errors
  cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npx tsc --noEmit 2>&1 | head -20
  # Assert: No errors related to MacroData
  ```

  **Evidence to Capture**:
  - [ ] TypeScript compilation passes without MacroData errors

  **Commit**: NO (groups with Task 3)

---

### Task 3: Update MacroStrip UI - Circle Indicator, Change %, Reduced Padding

- [ ] 3. Implement new visual design for MacroStrip component

  **What to do**:
  
  **3a. Reduce Padding**:
  - Container: Change `py-4` to `py-2` (line 82)
  - IndicatorCard: Change `p-3` to `p-2` (line 37)
  
  **3b. VIX/F&G: Replace StatusBadge with Circle + Lime Text**:
  - Remove StatusBadge import usage for VIX and F&G
  - Add inline 8px colored circle before value: `<span className="w-2 h-2 rounded-full bg-{color}" />`
  - Value text: Use lime color `text-[#ABF43F]`
  - Circle colors: `bg-emerald-500` (ok), `bg-amber-500` (action), `bg-red-500` (danger)
  
  **3c. Add Change % Row Below All Values**:
  - Add new prop to IndicatorCard: `change?: number | null`
  - Add new prop to IndicatorCard: `changeColorInverted?: boolean` (for VIX, 10Y, DXY, USD/KRW)
  - Display format: `+1.2%` or `-0.5%` with 1 decimal place
  - Color logic:
    - Normal: positive = green, negative = red
    - Inverted: positive = red, negative = green
  - Zero change: gray text, no sign
  - F&G special: Show `+5pts` instead of `+5%`
  
  **3d. Semantic Coloring per Indicator**:
  - VIX: inverted (up = red)
  - F&G: normal (up = green), but display as points
  - 10Y Treasury: inverted (up = red)
  - DXY: inverted (up = red)
  - NQ: normal (up = green)
  - USD/KRW: inverted (up = red)

  **Must NOT do**:
  - Do NOT create separate component file for the circle
  - Do NOT add hover/tooltip states
  - Do NOT add trend arrows (↑/↓)
  - Do NOT remove F&G sublabel ("Extreme Fear" etc.)
  - Do NOT touch the loading state JSX

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI component modification with styling precision
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Needed for proper visual hierarchy and Tailwind styling
  - **Skills Evaluated but Omitted**:
    - `playwright`: Reserved for verification task

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1, Task 2

  **References**:

  **Pattern References**:
  - `components/e03/MacroStrip.tsx:35-54` - Current `IndicatorCard` component to modify
  - `components/e03/MacroStrip.tsx:96-147` - Current indicator grid layout
  - `components/e03/StatusBadge.tsx:27-35` - Circle implementation reference (w-1.5 h-1.5)

  **API/Type References**:
  - `components/e03/MacroStrip.tsx:7-17` - MacroData interface with new `change` field

  **Design References**:
  - Primary lime color: `#ABF43F` (from README design tokens)
  - Circle size: 8px = `w-2 h-2`
  - Status colors: emerald-500 (ok), amber-500 (action), red-500 (danger)

  **WHY Each Reference Matters**:
  - `IndicatorCard:35-54`: Core component to extend with circle and change display
  - `StatusBadge.tsx:27-35`: Shows existing circle pattern (but 6px, we need 8px)
  - Lines 96-147: Where each indicator is rendered - need to pass new props

  **Acceptance Criteria**:

  **Automated Verification (Playwright browser via skill)**:
  ```
  # Agent executes via playwright browser automation:
  1. Navigate to: http://localhost:3000 (or relevant page with MacroStrip)
  2. Wait for: MacroStrip component to load (selector: "[data-testid='macro-strip']" or ".bg-surface" containing "MACRO")
  3. Screenshot: .sisyphus/evidence/task-3-macrostrip-before.png
  
  # Verify VIX indicator structure:
  4. Assert: VIX card contains 8px circle element (w-2 h-2 rounded-full)
  5. Assert: VIX value text has lime color (#ABF43F or text-[#ABF43F])
  6. Assert: VIX shows change % below value
  
  # Verify change % coloring:
  7. Assert: Change % text is visible for all 6 indicators
  8. Screenshot: .sisyphus/evidence/task-3-macrostrip-final.png
  
  # Verify padding reduction:
  9. Assert: Container has py-2 class (not py-4)
  10. Assert: Cards have p-2 class (not p-3)
  ```

  **Evidence to Capture**:
  - [ ] Before/after screenshots showing visual changes
  - [ ] Console showing no errors

  **Commit**: YES
  - Message: `feat(ui): enhance MacroStrip with change %, circle indicators, compact padding`
  - Files: `components/e03/MacroStrip.tsx`, `components/e03/ZoneBSignalCore.tsx`
  - Pre-commit: `npx tsc --noEmit`

---

### Task 4: Browser Verification and Final Cleanup

- [ ] 4. Verify complete implementation in browser

  **What to do**:
  - Load the dashboard page with MacroStrip
  - Verify all 6 indicators display correctly
  - Verify VIX/F&G have circle + lime text (no StatusBadge)
  - Verify change % appears below each value
  - Verify semantic coloring (VIX up = red, NQ up = green)
  - Verify reduced padding looks appropriate
  - Check for console errors
  - Capture final evidence screenshots

  **Must NOT do**:
  - Do NOT make any code changes (verification only)
  - Do NOT adjust mobile breakpoints

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification task only
  - **Skills**: [`playwright`]
    - `playwright`: Browser automation for verification
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Not making changes, just verifying

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final)
  - **Blocks**: None (completion)
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - Dashboard URL: `http://localhost:3000` (or specific route with MacroStrip)

  **WHY Each Reference Matters**:
  - Need to navigate to correct page to see MacroStrip component

  **Acceptance Criteria**:

  **Automated Verification (Playwright browser via skill)**:
  ```
  # Full visual verification:
  1. Navigate to: http://localhost:3000
  2. Wait for: Network idle (all API calls complete)
  3. Wait for: MacroStrip visible
  
  # Capture evidence:
  4. Screenshot: .sisyphus/evidence/task-4-full-dashboard.png
  5. Screenshot: .sisyphus/evidence/task-4-macrostrip-closeup.png (element screenshot)
  
  # Verify no errors:
  6. Assert: No console errors
  7. Assert: All 6 indicator values are not "--" (data loaded)
  
  # Verify specific elements:
  8. Assert: VIX indicator has lime-colored value
  9. Assert: VIX indicator has colored circle (not StatusBadge)
  10. Assert: Each indicator shows change text below value
  ```

  **Evidence to Capture**:
  - [ ] Full dashboard screenshot
  - [ ] MacroStrip close-up screenshot
  - [ ] Console output showing no errors

  **Commit**: NO (verification only)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(api): add previous close and change % to macro indicators` | `app/api/macro/route.ts` | `curl localhost:3000/api/macro \| jq .` |
| 3 | `feat(ui): enhance MacroStrip with change %, circle indicators, compact padding` | `components/e03/MacroStrip.tsx`, `components/e03/ZoneBSignalCore.tsx` | `npx tsc --noEmit` |

---

## Success Criteria

### Verification Commands
```bash
# API returns change field
curl -s http://localhost:3000/api/macro | jq '.vix.change, .nq.change, .fng.change'

# TypeScript compiles
cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npx tsc --noEmit

# Dev server runs without errors
npm run dev
```

### Final Checklist
- [ ] All 6 indicators show change % below value
- [ ] VIX/F&G use 8px circle + lime text (StatusBadge removed)
- [ ] Container padding is py-2 (reduced from py-4)
- [ ] Card padding is p-2 (reduced from p-3)
- [ ] Semantic coloring correct (VIX up = red, NQ up = green)
- [ ] F&G shows point change (e.g., +5pts)
- [ ] No TypeScript errors
- [ ] No browser console errors
- [ ] Screenshots captured in `.sisyphus/evidence/`
