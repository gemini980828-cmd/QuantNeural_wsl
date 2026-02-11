# Macro Dashboard Overhaul - Phase 1 + 2 + UI Redesign

## TL;DR

> **Quick Summary**: Expand macro dashboard from 6 to 15 indicators (Yahoo Finance + FRED API), redesign UI with category grouping and sparklines, expand MacroStrip, and add test infrastructure.
>
> **Deliverables**:
> - Extended `/api/macro/route.ts` with 15 indicators and 7-day history
> - FRED API helper at `lib/api/fred.ts`
> - Redesigned macro page with 4 category sections + sparklines
> - Enhanced MacroStrip with 4 additional key indicators
> - Vitest test suite with API mocking
>
> **Estimated Effort**: Large (4-6 days)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Test Setup → API Extension → UI Redesign → MacroStrip

---

## Context

### Original Request
Complete overhaul of macro dashboard:
- Phase 1: Add 6 Yahoo Finance indicators
- Phase 2: Add 3 FRED API indicators
- Phase 3: UI redesign with category grouping, sparklines, responsive grid
- Bonus: Expand MacroStrip with key indicators

### Interview Summary
**Key Discussions**:
- Sparklines: Recharts (already in deps)
- FRED fallback: Show "--" with tooltip "FRED API key required"
- MacroStrip expansion: Add ES, Yield Curve, VIX3M, S&P500
- Testing: Add Vitest test infrastructure

**Research Findings**:
- Current API at `200tq/dashboard/app/api/macro/route.ts` fetches 6 indicators
- MacroStrip used in SimpleView and ZoneBSignalCore - must maintain backward compat
- `recharts` already installed (v2.11.0)
- Design system uses CSS variables (bg, surface, fg, muted)
- No test infrastructure currently exists

### Metis Review
**Identified Gaps** (addressed in this plan):
1. **Data freshness**: Manual refresh only, no auto-refresh (keeps complexity low)
2. **MacroStrip layout**: 8 total indicators too many → Use selective display with `condensed` prop
3. **Historical data**: Use 7 trading days (weekdays), not calendar days
4. **Error states**: Partial failure shows loaded data + error for failed indicators
5. **Test cases**: Defined explicit Vitest test cases with API mocking

---

## Work Objectives

### Core Objective
Expand macro dashboard to 15 indicators with enhanced UI featuring category grouping, sparklines, and responsive design while maintaining backward compatibility with MacroStrip component.

### Concrete Deliverables
1. `200tq/dashboard/app/api/macro/route.ts` - Extended with 15 indicators + history
2. `200tq/dashboard/lib/api/fred.ts` - FRED API helper
3. `200tq/dashboard/app/(shell)/macro/page.tsx` - Redesigned UI
4. `200tq/dashboard/components/e03/MacroStrip.tsx` - Expanded with new indicators
5. `200tq/dashboard/components/macro/Sparkline.tsx` - Recharts sparkline component
6. `200tq/dashboard/__tests__/` - Vitest test suite
7. `.env.local` addition: `FRED_API_KEY` (optional)

### Definition of Done
- [ ] `npm run build` passes with no errors
- [ ] API returns all 15 indicators (12 Yahoo + 3 FRED)
- [ ] Sparklines display 7-day trend for all indicators
- [ ] FRED gracefully degrades to "--" when API key missing
- [ ] MacroStrip renders without errors in SimpleView/ZoneBSignalCore
- [ ] Responsive grid works on mobile (<768px)
- [ ] Vitest suite passes with all tests green
- [ ] No TypeScript errors
- [ ] No console errors in browser

### Must Have
- All 15 indicators fetching correctly
- 7-day historical data for sparklines (trading days)
- 4 category sections in UI
- Color-coded thresholds for VIX, F&G, Yield Curve, HY Spread
- Backward-compatible MacroData interface
- FRED API fallback UI

### Must NOT Have (Guardrails)
- ❌ Real-time WebSocket updates (out of scope)
- ❌ Historical data persistence in database
- ❌ Custom date range selection for sparklines
- ❌ User preferences/customization
- ❌ News tab implementation
- ❌ Generic FRED client (only 3 specific series)
- ❌ E2E tests with Playwright (unit tests only)
- ❌ Refactoring unrelated parts of codebase
- ❌ IE11/old browser support

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO
- **User wants tests**: YES (Vitest)
- **Framework**: Vitest (ESM-native, fast, Next.js compatible)

### Test Setup Task
- [ ] 0. Setup Vitest
  - Install: `npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom`
  - Config: Create `vitest.config.ts`
  - Verify: `npm test` → shows help
  - Example: Create `__tests__/example.test.ts`
  - Verify: `npm test` → 1 test passes

### TDD Workflow
Each API test follows: Mock → Test → Implement
```
1. Write test with mocked API response
2. Run test (RED - not implemented)
3. Implement actual code
4. Run test (GREEN - passes)
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Setup Vitest infrastructure [no dependencies]
└── Task 2: Create FRED API helper [no dependencies]

Wave 2 (After Wave 1):
├── Task 3: Extend /api/macro route [depends: 2]
├── Task 4: Create Sparkline component [depends: 1]
└── Task 5: Write API tests [depends: 1, 3]

Wave 3 (After Wave 2):
├── Task 6: Redesign macro page UI [depends: 3, 4]
├── Task 7: Enhance MacroStrip [depends: 3]
└── Task 8: Write component tests [depends: 1, 6, 7]

Wave 4 (Final):
└── Task 9: Integration testing & polish [depends: all]

Critical Path: Task 1 → Task 3 → Task 6 → Task 9
Parallel Speedup: ~35% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3, 4, 5, 8 | 2 |
| 2 | None | 3 | 1 |
| 3 | 2 | 5, 6, 7 | 4 |
| 4 | 1 | 6 | 3, 5 |
| 5 | 1, 3 | 9 | 4, 6, 7 |
| 6 | 3, 4 | 8, 9 | 5, 7 |
| 7 | 3 | 8, 9 | 5, 6 |
| 8 | 1, 6, 7 | 9 | None |
| 9 | All | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Approach |
|------|-------|---------------------|
| 1 | 1, 2 | Parallel: Test setup + FRED helper |
| 2 | 3, 4, 5 | Parallel after Wave 1 |
| 3 | 6, 7, 8 | Parallel UI work |
| 4 | 9 | Sequential integration |

---

## TODOs

### Wave 1: Foundation

- [ ] 1. Setup Vitest Test Infrastructure

  **What to do**:
  - Install Vitest and testing dependencies
  - Configure vitest.config.ts for Next.js/React
  - Add test script to package.json
  - Create example test to verify setup
  - Setup MSW (Mock Service Worker) for API mocking

  **Must NOT do**:
  - Don't add Playwright/E2E tests
  - Don't configure coverage thresholds yet

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard dev tooling setup, well-documented
  - **Skills**: None needed
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: No UI work in this task

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 4, 5, 8
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `200tq/dashboard/package.json` - Current dependencies and scripts

  **External References**:
  - Vitest docs: `https://vitest.dev/guide/`
  - Testing Library React: `https://testing-library.com/docs/react-testing-library/intro/`

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  npm test -- --run
  # Assert: Output shows "1 passed" (example test)
  # Assert: Exit code 0
  
  # Verify config exists
  ls 200tq/dashboard/vitest.config.ts
  # Assert: File exists
  ```

  **Commit**: YES
  - Message: `feat(test): add Vitest infrastructure with React Testing Library`
  - Files: `package.json`, `vitest.config.ts`, `__tests__/example.test.ts`
  - Pre-commit: `npm test -- --run`

---

- [ ] 2. Create FRED API Helper

  **What to do**:
  - Create `lib/api/fred.ts` with typed fetch function
  - Implement `fetchFredSeries(seriesId: string): Promise<FredResult>`
  - Handle missing API key gracefully (return null)
  - Parse FRED JSON response to extract latest value
  - Add 7-day historical data extraction

  **Must NOT do**:
  - Don't create generic FRED client
  - Don't add caching layer
  - Don't add retry logic (keep simple)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple API wrapper, <100 lines
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `200tq/dashboard/app/api/macro/route.ts:37-60` - Yahoo Finance fetch pattern to follow

  **API References**:
  - FRED API endpoint: `https://api.stlouisfed.org/fred/series/observations`
  - Query params: `series_id`, `api_key`, `file_type=json`, `limit=10`, `sort_order=desc`

  **Type Definition**:
  ```typescript
  interface FredResult {
    value: number | null;
    change: number | null;
    date: string | null;
    history: number[];  // Last 7 trading days
  }
  ```

  **Acceptance Criteria**:

  ```bash
  # Agent runs TypeScript check:
  npx tsc --noEmit 200tq/dashboard/lib/api/fred.ts
  # Assert: No type errors
  
  # Test with mock (in test file):
  # Verify function returns null when FRED_API_KEY missing
  # Verify function parses valid response correctly
  ```

  **Commit**: YES
  - Message: `feat(api): add FRED API helper with graceful fallback`
  - Files: `lib/api/fred.ts`
  - Pre-commit: `npx tsc --noEmit`

---

### Wave 2: API & Components

- [ ] 3. Extend /api/macro Route with All 15 Indicators

  **What to do**:
  - Add 6 new Yahoo Finance symbols: `^VIX3M`, `^GSPC`, `ES=F`, `GC=F`, `CL=F`, `BTC-USD`
  - Integrate FRED helper for 3 indicators: `T10Y2Y`, `BAMLH0A0HYM2`, `DGS2`
  - Extend `MacroData` interface (keep all existing fields for backward compat!)
  - Add `history: number[]` field to each indicator (7 trading days)
  - Modify `fetchYahooQuote` to fetch 7-day range: `range=7d`
  - Add color logic for new indicators (Yield Curve, HY Spread)

  **Must NOT do**:
  - Don't change existing field names (backward compat!)
  - Don't remove any existing data
  - Don't add caching (keep stateless)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: API work with data transformation, moderate complexity
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Tasks 5, 6, 7
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `200tq/dashboard/app/api/macro/route.ts:37-60` - Existing fetchYahooQuote pattern
  - `200tq/dashboard/app/api/macro/route.ts:8-16` - Existing MacroData interface (KEEP!)

  **API References**:
  - Yahoo Finance 7d range: `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=7d`

  **Extended Interface** (append to existing, don't replace):
  ```typescript
  interface ExtendedMacroData extends MacroData {
    // Phase 1: Yahoo Finance
    vix3m: IndicatorData;
    sp500: IndicatorData;
    es: IndicatorData;
    gold: IndicatorData;
    oil: IndicatorData;
    btc: IndicatorData;
    
    // Phase 2: FRED
    yieldCurve: IndicatorData & { color: ColorTone };
    hySpread: IndicatorData & { color: ColorTone };
    treasury2y: IndicatorData;
  }
  
  interface IndicatorData {
    value: number | null;
    change: number | null;
    history: number[];
  }
  ```

  **Color Logic**:
  ```typescript
  function getYieldCurveColor(value: number): ColorTone {
    if (value > 0.5) return 'ok';      // Normal
    if (value >= 0) return 'action';   // Flattening
    return 'danger';                    // Inverted!
  }
  
  function getHySpreadColor(value: number): ColorTone {
    if (value < 3) return 'ok';        // Normal
    if (value <= 5) return 'action';   // Elevated
    return 'danger';                    // Credit stress
  }
  ```

  **Acceptance Criteria**:

  ```bash
  # Agent runs dev server and tests API:
  npm run dev &
  sleep 5
  curl -s http://localhost:3000/api/macro | jq '.vix3m.value, .sp500.value, .es.value, .gold.value, .oil.value, .btc.value'
  # Assert: 6 values returned (not null for most)
  
  curl -s http://localhost:3000/api/macro | jq '.yieldCurve, .hySpread, .treasury2y'
  # Assert: Returns objects (may be null if no API key)
  
  curl -s http://localhost:3000/api/macro | jq '.vix.history | length'
  # Assert: Returns 7 (or fewer for new listings)
  
  # Backward compat check:
  curl -s http://localhost:3000/api/macro | jq '.vix.value, .fng.value, .treasury.value, .dxy.value, .nq.value, .usdkrw.value'
  # Assert: All 6 original fields present
  ```

  **Commit**: YES
  - Message: `feat(api): expand macro API to 15 indicators with history`
  - Files: `app/api/macro/route.ts`, `lib/api/fred.ts`
  - Pre-commit: `npm run build`

---

- [ ] 4. Create Sparkline Component with Recharts

  **What to do**:
  - Create `components/macro/Sparkline.tsx`
  - Use Recharts `AreaChart` for gradient fill effect
  - Accept `data: number[]` and `color: 'positive' | 'negative' | 'neutral'`
  - Size: 80x32px, no axes, no labels
  - Responsive width

  **Must NOT do**:
  - Don't add tooltips (keep minimal)
  - Don't add interactivity
  - Don't create wrapper for all of Recharts

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI component with charting
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Sparkline needs visual polish

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 5)
  - **Blocks**: Task 6
  - **Blocked By**: Task 1 (for tests)

  **References**:

  **Pattern References**:
  - `200tq/dashboard/components/e03/MacroStrip.tsx` - Component structure pattern
  - `200tq/dashboard/tailwind.config.ts:20-21` - Semantic colors (positive, negative)

  **External References**:
  - Recharts AreaChart: `https://recharts.org/en-US/api/AreaChart`

  **Component Signature**:
  ```typescript
  interface SparklineProps {
    data: number[];
    color?: 'positive' | 'negative' | 'neutral';
    width?: number;
    height?: number;
  }
  
  export function Sparkline({ 
    data, 
    color = 'neutral', 
    width = 80, 
    height = 32 
  }: SparklineProps): JSX.Element
  ```

  **Acceptance Criteria**:

  ```bash
  # TypeScript check:
  npx tsc --noEmit
  # Assert: No errors
  
  # Visual verification via test:
  npm test -- --run sparkline
  # Assert: Sparkline test passes (renders without error)
  ```

  **Commit**: YES
  - Message: `feat(ui): add Sparkline component with Recharts`
  - Files: `components/macro/Sparkline.tsx`, `__tests__/components/Sparkline.test.tsx`
  - Pre-commit: `npm test -- --run`

---

- [ ] 5. Write API Tests with Mocked Responses

  **What to do**:
  - Create `__tests__/api/macro.test.ts`
  - Mock Yahoo Finance and FRED API responses
  - Test: All 15 indicators parsed correctly
  - Test: FRED returns null when no API key
  - Test: Partial failure handling (some indicators fail)
  - Test: History array has correct length

  **Must NOT do**:
  - Don't hit real APIs in tests
  - Don't test UI rendering (API only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard test writing
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Task 9
  - **Blocked By**: Tasks 1, 3

  **References**:

  **Pattern References**:
  - `200tq/dashboard/app/api/macro/route.ts` - Code under test

  **Test Cases**:
  ```typescript
  describe('/api/macro', () => {
    it('returns all 15 indicators', async () => { /* ... */ });
    it('preserves backward-compatible MacroData fields', async () => { /* ... */ });
    it('returns null for FRED when API key missing', async () => { /* ... */ });
    it('includes 7-day history for each indicator', async () => { /* ... */ });
    it('calculates change % correctly', async () => { /* ... */ });
    it('applies correct color for VIX thresholds', async () => { /* ... */ });
    it('applies correct color for Yield Curve (negative = danger)', async () => { /* ... */ });
  });
  ```

  **Acceptance Criteria**:

  ```bash
  npm test -- --run api/macro
  # Assert: All 7 tests pass
  # Assert: Output shows "7 passed"
  ```

  **Commit**: YES
  - Message: `test(api): add macro API tests with mocked responses`
  - Files: `__tests__/api/macro.test.ts`
  - Pre-commit: `npm test -- --run`

---

### Wave 3: UI Redesign

- [ ] 6. Redesign Macro Page with Category Sections

  **What to do**:
  - Refactor `app/(shell)/macro/page.tsx` with 4 sections:
    1. 📊 변동성 & 센티먼트 (VIX, VIX3M, F&G)
    2. 📈 금리 & 크레딧 (10Y, 2Y, Yield Curve, HY Spread)
    3. 💹 시장 & 선물 (S&P500, NQ, ES)
    4. 🌍 통화 & 원자재 (DXY, USD/KRW, Gold, Oil, BTC)
  - Create `CategorySection` component
  - Enhance `IndicatorCard` with sparkline integration
  - Implement responsive grid (mobile: 2col, tablet: 3col, desktop: 4col per section)
  - Add section collapse/expand (optional visual enhancement)

  **Must NOT do**:
  - Don't implement News tab (out of scope)
  - Don't add filtering/sorting
  - Don't add user preferences

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI redesign with significant layout changes
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Complex layout + visual polish required

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: Tasks 3, 4

  **References**:

  **Pattern References**:
  - `200tq/dashboard/app/(shell)/macro/page.tsx:204-241` - Existing IndicatorCard pattern
  - `200tq/dashboard/tailwind.config.ts` - Design tokens
  - `200tq/dashboard/components/e03/MacroStrip.tsx:24-37` - Color mapping functions

  **Layout Spec**:
  ```
  [Section Header: 📊 변동성 & 센티먼트]
  ┌─────────┬─────────┬─────────┬─────────┐  Desktop (4 col)
  │   VIX   │  VIX3M  │   F&G   │         │
  │ [value] │ [value] │ [value] │         │
  │ [spark] │ [spark] │ [spark] │         │
  └─────────┴─────────┴─────────┴─────────┘
  
  ┌─────────┬─────────┬─────────┐  Tablet (3 col)
  │   VIX   │  VIX3M  │   F&G   │
  └─────────┴─────────┴─────────┘
  
  ┌─────────┬─────────┐  Mobile (2 col)
  │   VIX   │  VIX3M  │
  ├─────────┼─────────┤
  │   F&G   │         │
  └─────────┴─────────┘
  ```

  **Acceptance Criteria**:

  ```bash
  npm run build
  # Assert: Build succeeds
  
  # Visual verification via playwright:
  # Navigate to http://localhost:3000/macro
  # Assert: 4 section headers visible
  # Assert: Cards show sparklines
  # Screenshot: .sisyphus/evidence/macro-page-desktop.png
  
  # Mobile verification:
  # Set viewport to 375x667 (iPhone)
  # Assert: 2-column grid
  # Screenshot: .sisyphus/evidence/macro-page-mobile.png
  ```

  **Commit**: YES
  - Message: `feat(ui): redesign macro page with category sections and sparklines`
  - Files: `app/(shell)/macro/page.tsx`, `components/macro/CategorySection.tsx`
  - Pre-commit: `npm run build`

---

- [ ] 7. Enhance MacroStrip with 4 New Indicators

  **What to do**:
  - Add ES Futures, S&P 500, VIX3M, Yield Curve to MacroStrip
  - Update MacroData type usage to ExtendedMacroData
  - Handle new indicators showing "--" if not loaded yet
  - Maintain `condensed` prop behavior (show subset in condensed mode)
  - Add Yield Curve color indicator (green/yellow/red based on value)

  **Must NOT do**:
  - Don't add sparklines to MacroStrip (too cluttered)
  - Don't change existing indicator positions
  - Don't break SimpleView or ZoneBSignalCore

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Component enhancement with backward compat concerns
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6, 8)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `200tq/dashboard/components/e03/MacroStrip.tsx` - Component to modify
  - `200tq/dashboard/components/e03/SimpleView.tsx` - Consumer (don't break!)
  - `200tq/dashboard/components/e03/ZoneBSignalCore.tsx` - Consumer (don't break!)

  **Layout Decision**:
  ```
  Existing (6): VIX | F&G | 10Y | DXY | NQ | 환율
  
  New Full Mode (10): VIX | VIX3M | F&G | 10Y | Yield | ES | SPX | DXY | NQ | 환율
  
  New Condensed (6): VIX | VIX3M | F&G | Yield | NQ | 환율
  ```

  **Acceptance Criteria**:

  ```bash
  npm run build
  # Assert: Build succeeds (no TypeScript errors)
  
  # Verify SimpleView still works:
  # Navigate to SimpleView page in browser
  # Assert: MacroStrip renders without errors
  # Assert: New indicators visible (ES, SPX, VIX3M, Yield)
  
  # Console check:
  # Assert: No console errors
  ```

  **Commit**: YES
  - Message: `feat(ui): expand MacroStrip with ES, S&P500, VIX3M, Yield Curve`
  - Files: `components/e03/MacroStrip.tsx`
  - Pre-commit: `npm run build`

---

- [ ] 8. Write Component Tests for UI

  **What to do**:
  - Create `__tests__/components/IndicatorCard.test.tsx`
  - Create `__tests__/components/MacroStrip.test.tsx`
  - Create `__tests__/components/Sparkline.test.tsx`
  - Test: Components render without crashing
  - Test: Loading states display correctly
  - Test: Color thresholds apply correctly
  - Test: MacroStrip backward compat (old props still work)

  **Must NOT do**:
  - Don't test visual styles (no snapshot tests)
  - Don't test API calls (covered in Task 5)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard component testing
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (after Tasks 6, 7)
  - **Parallel Group**: Sequential after Wave 3 UI tasks
  - **Blocks**: Task 9
  - **Blocked By**: Tasks 1, 6, 7

  **References**:

  **Pattern References**:
  - `__tests__/example.test.ts` - Test file structure (from Task 1)

  **Test Cases**:
  ```typescript
  describe('IndicatorCard', () => {
    it('renders value and change', () => { /* ... */ });
    it('shows loading skeleton when loading=true', () => { /* ... */ });
    it('applies correct color class for danger', () => { /* ... */ });
  });
  
  describe('MacroStrip', () => {
    it('renders all indicators in full mode', () => { /* ... */ });
    it('renders subset in condensed mode', () => { /* ... */ });
    it('handles null data gracefully', () => { /* ... */ });
    it('links to /macro page', () => { /* ... */ });
  });
  
  describe('Sparkline', () => {
    it('renders SVG element', () => { /* ... */ });
    it('handles empty data array', () => { /* ... */ });
    it('applies positive/negative colors', () => { /* ... */ });
  });
  ```

  **Acceptance Criteria**:

  ```bash
  npm test -- --run components/
  # Assert: All component tests pass
  # Assert: Output shows ≥9 tests passed
  ```

  **Commit**: YES
  - Message: `test(ui): add component tests for IndicatorCard, MacroStrip, Sparkline`
  - Files: `__tests__/components/*.test.tsx`
  - Pre-commit: `npm test -- --run`

---

### Wave 4: Integration & Polish

- [ ] 9. Integration Testing & Final Polish

  **What to do**:
  - Run full test suite
  - Verify all API endpoints work
  - Test responsive design at all breakpoints
  - Fix any TypeScript errors
  - Fix any console warnings
  - Test FRED fallback (with and without API key)
  - Update README with new environment variable
  - Add .env.example entry for FRED_API_KEY

  **Must NOT do**:
  - Don't add new features
  - Don't refactor working code

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Cross-cutting integration work
  - **Skills**: [`playwright`]
    - `playwright`: Visual verification across breakpoints

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: None (final task)
  - **Blocks**: None
  - **Blocked By**: All previous tasks

  **References**:

  **All previous files**

  **Acceptance Criteria**:

  ```bash
  # Full test suite:
  npm test -- --run
  # Assert: All tests pass (≥16 tests)
  
  # Build verification:
  npm run build
  # Assert: Exit code 0
  
  # TypeScript check:
  npx tsc --noEmit
  # Assert: No errors
  
  # Playwright visual verification:
  # Navigate to http://localhost:3000/macro
  # Screenshot desktop, tablet (768px), mobile (375px)
  # Assert: All 4 category sections visible on desktop
  # Assert: 2-column grid on mobile
  # Assert: Sparklines visible in all cards
  
  # FRED fallback test:
  # Temporarily remove FRED_API_KEY from .env.local
  # Refresh page
  # Assert: Yield Curve, HY Spread, 2Y show "--"
  # Assert: Tooltip on hover says "FRED API key required"
  ```

  **Commit**: YES
  - Message: `chore: integration testing and polish for macro dashboard`
  - Files: `README.md`, `.env.example`
  - Pre-commit: `npm run build && npm test -- --run`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(test): add Vitest infrastructure` | package.json, vitest.config.ts | `npm test -- --run` |
| 2 | `feat(api): add FRED API helper` | lib/api/fred.ts | `npx tsc --noEmit` |
| 3 | `feat(api): expand macro API to 15 indicators` | app/api/macro/route.ts | `npm run build` |
| 4 | `feat(ui): add Sparkline component` | components/macro/Sparkline.tsx | `npm test -- --run` |
| 5 | `test(api): add macro API tests` | __tests__/api/macro.test.ts | `npm test -- --run` |
| 6 | `feat(ui): redesign macro page with categories` | app/(shell)/macro/page.tsx | `npm run build` |
| 7 | `feat(ui): expand MacroStrip` | components/e03/MacroStrip.tsx | `npm run build` |
| 8 | `test(ui): add component tests` | __tests__/components/*.tsx | `npm test -- --run` |
| 9 | `chore: integration testing and polish` | README.md, .env.example | `npm run build` |

---

## Success Criteria

### Verification Commands
```bash
# All tests pass
npm test -- --run
# Expected: ≥16 tests, 0 failures

# Build succeeds
npm run build
# Expected: Exit code 0

# TypeScript clean
npx tsc --noEmit
# Expected: No errors

# API returns all 15 indicators
curl -s http://localhost:3000/api/macro | jq 'keys | length'
# Expected: 16 (15 indicators + updatedAt)
```

### Final Checklist
- [ ] All 15 indicators fetch and display correctly
- [ ] Sparklines show 7-day trend for all indicators
- [ ] 4 category sections render on macro page
- [ ] MacroStrip shows new indicators without breaking consumers
- [ ] FRED gracefully degrades when API key missing
- [ ] Responsive design works at mobile/tablet/desktop
- [ ] All Vitest tests pass
- [ ] npm run build succeeds
- [ ] No TypeScript errors
- [ ] No console errors in browser
- [ ] README updated with FRED_API_KEY instructions
