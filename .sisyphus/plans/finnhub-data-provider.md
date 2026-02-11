# Finnhub Multi-Provider Data Integration

## TL;DR

> **Quick Summary**: Replace delayed Polygon data with real-time Finnhub as primary source, keeping Polygon and adding yahoo-finance2 as cascading fallbacks for maximum data freshness and reliability.
> 
> **Deliverables**:
> - Finnhub fetch function using `/stock/candle` endpoint
> - yahoo-finance2 fetch function as tertiary fallback
> - Cascading fallback logic (Finnhub → Polygon → yahoo-finance2)
> - Dynamic `source` field tracking in `PriceRow`
> - Environment variable setup for `FINNHUB_API_KEY`
> 
> **Estimated Effort**: Medium (3-4 hours)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 4 → Task 5 → Task 6

---

## Context

### Original Request
User wants to improve data freshness for the 200tq dashboard by replacing the current Polygon data source (which has DELAYED status on free tier) with faster alternatives.

### Interview Summary
**Key Discussions**:
- **Tickers**: SPLG is correct (not SQQQ) - keep existing `['TQQQ', 'QQQ', 'SPLG', 'SGOV']`
- **Fallback chain**: Finnhub (primary) → Polygon (fallback) → yahoo-finance2 (tertiary)
- **Finnhub endpoint**: Use `/stock/candle` for proper EOD bars (matches Polygon structure)
- **yahoo-finance2**: Include as npm package fallback (no API key required)

**Research Findings**:
- **Finnhub**: Real-time on free tier, 60 calls/min, `/stock/candle?symbol={}&resolution=D&from={}&to={}`
- **yahoo-finance2**: `yahooFinance.historical(symbol, { period1, period2 })` returns `{ date, open, high, low, close, volume }`
- **Current code**: `PriceRow.source` already exists (line 40), currently hardcoded to `'polygon'`

### Metis Review
**Identified Gaps** (addressed):
- **Error isolation**: Each provider should fail independently without blocking others → Implemented per-ticker fallback
- **Rate limiting awareness**: Finnhub has 60/min limit → 4 tickers × 1 call each = well under limit
- **Type safety**: Need interfaces for Finnhub/Yahoo responses → Added to Task 1 & 2
- **Logging**: Need visibility into which provider succeeded → Source tracked in both `PriceRow` and `payload.sourceMeta`

---

## Work Objectives

### Core Objective
Add Finnhub as primary data source with cascading fallback to Polygon and yahoo-finance2, ensuring maximum data availability and freshness.

### Concrete Deliverables
- `fetchFinnhubCandle()` function with proper TypeScript interface
- `fetchYahooBar()` function using yahoo-finance2 package
- `fetchWithFallback()` orchestrator with per-ticker fallback logic
- Updated `source` field tracking actual provider used
- Updated `sourceMeta` in payload to reflect multi-provider status

### Definition of Done
- [ ] `npm run build` passes with no TypeScript errors
- [ ] Cron endpoint returns data with `source: 'finnhub'` when Finnhub succeeds
- [ ] Cron endpoint falls back to Polygon if Finnhub fails
- [ ] Cron endpoint falls back to yahoo-finance2 if both Finnhub and Polygon fail
- [ ] All 4 tickers (TQQQ, QQQ, SPLG, SGOV) fetch successfully

### Must Have
- Finnhub `/stock/candle` integration with proper date handling
- yahoo-finance2 npm package integration
- Per-ticker fallback (not all-or-nothing)
- Source tracking per ticker
- Existing idempotency logic preserved

### Must NOT Have (Guardrails)
- **NO changes to SMA calculation logic** (lines 90-107) - working correctly
- **NO changes to verdict determination logic** (lines 99-107)
- **NO changes to idempotency check logic** (lines 209-222)
- **NO new database schema changes** - `source` column already exists
- **NO UI changes** - backend only
- **NO removal of Polygon code** - demote to fallback, don't delete

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (no test files found in dashboard)
- **User wants tests**: Manual verification (standard for this project)
- **Framework**: None currently configured

### Manual QA Only

Each TODO includes detailed verification using curl and log inspection.

**Evidence Required:**
- Actual curl command output
- Console logs showing provider selection
- Database query showing `source` field values

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Add Finnhub fetch function [no dependencies]
├── Task 2: Add yahoo-finance2 package and fetch function [no dependencies]
└── Task 3: Add TypeScript interfaces for all providers [no dependencies]

Wave 2 (After Wave 1):
├── Task 4: Implement cascading fallback orchestrator [depends: 1, 2, 3]
└── Task 5: Update GET handler to use fallback orchestrator [depends: 4]

Wave 3 (After Wave 2):
└── Task 6: Add environment variable and deploy [depends: 5]

Critical Path: Task 1 → Task 4 → Task 5 → Task 6
Parallel Speedup: ~35% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 4 | 2, 3 |
| 2 | None | 4 | 1, 3 |
| 3 | None | 4 | 1, 2 |
| 4 | 1, 2, 3 | 5 | None |
| 5 | 4 | 6 | None |
| 6 | 5 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2, 3 | `category="quick"` - small focused functions |
| 2 | 4, 5 | `category="quick"` - orchestration logic |
| 3 | 6 | `category="quick"` - deployment |

---

## TODOs

- [ ] 1. Add Finnhub `/stock/candle` fetch function

  **What to do**:
  - Add `FinnhubCandle` interface matching API response
  - Implement `fetchFinnhubCandle(ticker: string, apiKey: string): Promise<PolygonDailyBar | null>`
  - Use endpoint: `https://finnhub.io/api/v1/stock/candle?symbol=${ticker}&resolution=D&from=${fromUnix}&to=${toUnix}&token=${apiKey}`
  - Convert Finnhub response to existing `PolygonDailyBar` format for compatibility
  - Return `null` on any error (let fallback handle it)

  **API Response Structure**:
  ```typescript
  interface FinnhubCandleResponse {
    c: number[];  // Close prices
    h: number[];  // High prices
    l: number[];  // Low prices
    o: number[];  // Open prices
    s: string;    // Status: "ok" or "no_data"
    t: number[];  // Unix timestamps
    v: number[];  // Volume
  }
  ```

  **Must NOT do**:
  - Don't modify existing `PolygonDailyBar` interface
  - Don't add retry logic (fallback handles failures)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single focused function, <50 lines, clear specification
  - **Skills**: None required
    - Plain TypeScript fetch, no specialized tooling needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 4
  - **Blocked By**: None (can start immediately)

  **References**:
  
  **Pattern References**:
  - `route.ts:47-67` - `fetchLatestBar()` pattern for Polygon (follow same error handling style)
  - `route.ts:72-85` - `barToPriceRow()` conversion pattern (output format to match)

  **API/Type References**:
  - `route.ts:23-30` - `PolygonDailyBar` interface (output must match this)
  - Finnhub docs: `https://finnhub.io/docs/api/stock-candles` - API spec

  **External References**:
  - Finnhub candle endpoint returns arrays; take last element for most recent bar

  **Acceptance Criteria**:

  **Manual Execution Verification:**
  - [ ] Function compiles: `npm run build` → no TypeScript errors
  - [ ] Verify Finnhub API manually:
    ```bash
    curl "https://finnhub.io/api/v1/stock/candle?symbol=QQQ&resolution=D&from=$(date -d '7 days ago' +%s)&to=$(date +%s)&token=${FINNHUB_API_KEY}"
    ```
    Expected: JSON with `s: "ok"` and arrays `c`, `h`, `l`, `o`, `t`, `v`

  **Commit**: NO (groups with Task 3)

---

- [ ] 2. Add yahoo-finance2 package and fetch function

  **What to do**:
  - Run `npm install yahoo-finance2`
  - Add `fetchYahooBar(ticker: string): Promise<PolygonDailyBar | null>`
  - Use `yahooFinance.historical(ticker, { period1: '7 days ago', period2: 'today' })`
  - Take last element from returned array
  - Convert to `PolygonDailyBar` format
  - Handle errors gracefully (return null)

  **yahoo-finance2 Response**:
  ```typescript
  interface YahooHistoricalRow {
    date: Date;
    open: number;
    high: number;
    low: number;
    close: number;
    adjClose: number;
    volume: number;
  }
  ```

  **Must NOT do**:
  - Don't use `quote()` endpoint (we need historical OHLCV)
  - Don't add complex error handling (fallback orchestrator handles it)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small function with npm package, clear input/output
  - **Skills**: None required
    - Standard npm package usage

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 4
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `route.ts:47-67` - Error handling pattern (try/catch, return null on error)
  - `route.ts:72-85` - Conversion to `PolygonDailyBar` pattern

  **API/Type References**:
  - `route.ts:23-30` - `PolygonDailyBar` interface (output must match)

  **External References**:
  - yahoo-finance2 docs: `yahooFinance.historical(query, { period1, period2 })` returns array of historical rows
  - Context7 snippet shows: `{ date, open, high, low, close, adjClose, volume }`

  **Acceptance Criteria**:

  **Manual Execution Verification:**
  - [ ] Package installed: `npm ls yahoo-finance2` → shows version
  - [ ] Function compiles: `npm run build` → no TypeScript errors
  - [ ] Verify package works in REPL:
    ```bash
    node -e "
      const yahooFinance = require('yahoo-finance2').default;
      yahooFinance.historical('QQQ', { period1: '2025-01-20' })
        .then(r => console.log('Last bar:', r[r.length-1]))
        .catch(e => console.error(e));
    "
    ```
    Expected: Object with date, open, high, low, close, volume

  **Commit**: NO (groups with Task 3)

---

- [ ] 3. Add TypeScript interfaces for multi-provider support

  **What to do**:
  - Add `FinnhubCandleResponse` interface at top of file
  - Add `DataSource = 'finnhub' | 'polygon' | 'yahoo'` type
  - Update `PriceRow.source` type from `string` to `DataSource`
  - Add `FetchResult` interface for orchestrator return type

  **Interfaces to add**:
  ```typescript
  type DataSource = 'finnhub' | 'polygon' | 'yahoo';

  interface FinnhubCandleResponse {
    c: number[];
    h: number[];
    l: number[];
    o: number[];
    s: string;
    t: number[];
    v: number[];
  }

  interface FetchResult {
    bar: PolygonDailyBar;
    source: DataSource;
  }
  ```

  **Must NOT do**:
  - Don't change `PolygonDailyBar` interface (it's the canonical format)
  - Don't change `PriceRow` structure (just tighten the type)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Type definitions only, no logic
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 4
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `route.ts:23-42` - Existing interface definitions (follow same style)

  **API/Type References**:
  - `route.ts:32-42` - `PriceRow` interface (update `source` type)
  - `route.ts:23-30` - `PolygonDailyBar` (don't modify)

  **Acceptance Criteria**:

  **Manual Execution Verification:**
  - [ ] TypeScript compiles: `npm run build` → no errors
  - [ ] Verify type is correctly narrowed:
    ```bash
    grep -n "DataSource" 200tq/dashboard/app/api/cron/ingest-close/route.ts
    ```
    Expected: Shows type definition and usage in PriceRow

  **Commit**: YES
  - Message: `feat(ingest): add Finnhub and yahoo-finance2 fetch functions with types`
  - Files: `route.ts`, `package.json`, `package-lock.json`
  - Pre-commit: `npm run build`

---

- [ ] 4. Implement cascading fallback orchestrator

  **What to do**:
  - Create `fetchWithFallback(ticker: string, finnhubKey: string | undefined, polygonKey: string): Promise<FetchResult | null>`
  - Try Finnhub first (if key exists)
  - Fall back to Polygon if Finnhub fails
  - Fall back to yahoo-finance2 if Polygon fails
  - Return `{ bar, source }` or `null` if all fail
  - Add console.log for each provider attempt (visibility)

  **Logic flow**:
  ```typescript
  async function fetchWithFallback(ticker, finnhubKey, polygonKey): Promise<FetchResult | null> {
    // 1. Try Finnhub (if key configured)
    if (finnhubKey) {
      const bar = await fetchFinnhubCandle(ticker, finnhubKey);
      if (bar) return { bar, source: 'finnhub' };
      console.log(`Finnhub failed for ${ticker}, trying Polygon...`);
    }
    
    // 2. Try Polygon
    const polygonBar = await fetchLatestBar(ticker, polygonKey);
    if (polygonBar) return { bar: polygonBar, source: 'polygon' };
    console.log(`Polygon failed for ${ticker}, trying Yahoo...`);
    
    // 3. Try Yahoo
    const yahooBar = await fetchYahooBar(ticker);
    if (yahooBar) return { bar: yahooBar, source: 'yahoo' };
    
    console.error(`All providers failed for ${ticker}`);
    return null;
  }
  ```

  **Must NOT do**:
  - Don't add retry logic within a single provider
  - Don't throw exceptions (return null for failures)
  - Don't add rate limiting (Finnhub 60/min is plenty for 4 tickers)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Orchestration logic, well-defined flow
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 2)
  - **Blocks**: Task 5
  - **Blocked By**: Tasks 1, 2, 3

  **References**:

  **Pattern References**:
  - `route.ts:47-67` - `fetchLatestBar()` - integrate into fallback chain
  - `route.ts:72-85` - `barToPriceRow()` - will be called with `FetchResult`

  **API/Type References**:
  - `route.ts:23-30` - `PolygonDailyBar` - common bar format
  - New `FetchResult` interface from Task 3

  **Acceptance Criteria**:

  **Manual Execution Verification:**
  - [ ] Function compiles: `npm run build` → no errors
  - [ ] Logic review: Verify fallback chain order in code

  **Commit**: NO (groups with Task 5)

---

- [ ] 5. Update GET handler to use fallback orchestrator

  **What to do**:
  - Add `FINNHUB_API_KEY` environment variable check (optional, not required)
  - Replace the ticker loop (lines 164-175) with `fetchWithFallback()` calls
  - Update `barToPriceRow()` to accept source parameter
  - Update `payload.sourceMeta` to reflect which providers were used
  - Preserve all existing idempotency and error handling logic

  **Changes needed**:
  1. Line 145-148: Add optional `finnhubKey` check
  2. Lines 164-175: Replace direct Polygon calls with `fetchWithFallback()`
  3. Line 72-85: Update `barToPriceRow()` signature to include source
  4. Lines 260-268: Update `sourceMeta` to track sources used

  **Updated sourceMeta example**:
  ```typescript
  sourceMeta: {
    sources: { QQQ: 'finnhub', TQQQ: 'finnhub', SPLG: 'polygon', SGOV: 'yahoo' },
    lastTradingDate: verdictDate,
    fetchedAt: new Date().toISOString(),
  }
  ```

  **Must NOT do**:
  - Don't change idempotency check logic (lines 209-222)
  - Don't change SMA calculation (lines 233-249)
  - Don't change verdict logic (line 252)
  - Don't require `FINNHUB_API_KEY` (graceful degradation to Polygon)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Refactoring existing handler, clear before/after
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 2)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `route.ts:136-175` - Current GET handler ticker loop (REPLACE)
  - `route.ts:72-85` - `barToPriceRow()` (MODIFY signature)
  - `route.ts:260-268` - `sourceMeta` structure (UPDATE)

  **API/Type References**:
  - `route.ts:32-42` - `PriceRow` interface with `source` field
  - New `FetchResult` type from Task 3

  **WHY Each Reference Matters**:
  - Lines 164-175: This is the exact code being replaced with fallback calls
  - Lines 260-268: This payload structure is consumed by the dashboard UI

  **Acceptance Criteria**:

  **Manual Execution Verification:**
  - [ ] Build succeeds: `npm run build` → no errors
  - [ ] Local test with Finnhub key:
    ```bash
    # Set env and test
    FINNHUB_API_KEY=your_key npm run dev
    # In another terminal:
    curl -H "Authorization: Bearer test" "http://localhost:3000/api/cron/ingest-close"
    ```
    Expected: Response with `status: "SUCCESS"` or `status: "SKIPPED"`
  - [ ] Verify source tracking in response payload

  **Commit**: YES
  - Message: `feat(ingest): implement Finnhub primary with Polygon/Yahoo fallback`
  - Files: `route.ts`
  - Pre-commit: `npm run build`

---

- [ ] 6. Add environment variable and deploy to Vercel

  **What to do**:
  - Document `FINNHUB_API_KEY` requirement in project
  - Add to Vercel environment variables
  - Deploy and verify cron job works
  - Check Supabase for new rows with correct `source` values

  **Vercel setup**:
  1. Go to Vercel dashboard → Project → Settings → Environment Variables
  2. Add `FINNHUB_API_KEY` with value from finnhub.io
  3. Redeploy (or push commit to trigger)

  **Must NOT do**:
  - Don't commit API key to code
  - Don't make Finnhub key required (Polygon fallback must work)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Deployment task, verification focused
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 3 - final)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 5

  **References**:

  **Documentation References**:
  - Vercel docs: Environment Variables in dashboard
  - Finnhub: Sign up at https://finnhub.io/ for free API key

  **Acceptance Criteria**:

  **Manual Execution Verification:**
  - [ ] Vercel env var set: Check Vercel dashboard → shows `FINNHUB_API_KEY`
  - [ ] Deploy succeeds: Vercel deployment → green checkmark
  - [ ] Trigger cron manually:
    ```bash
    curl -H "Authorization: Bearer ${CRON_SECRET}" "https://your-app.vercel.app/api/cron/ingest-close"
    ```
    Expected: `{"status":"SUCCESS",...}` or `{"status":"SKIPPED",...}`
  - [ ] Verify Supabase data:
    ```sql
    SELECT date, symbol, source, fetched_at 
    FROM prices_daily 
    ORDER BY fetched_at DESC 
    LIMIT 8;
    ```
    Expected: Recent rows with `source` = 'finnhub' (or fallback provider)

  **Commit**: NO (deployment only, no code changes)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 3 | `feat(ingest): add Finnhub and yahoo-finance2 fetch functions with types` | route.ts, package.json, package-lock.json | npm run build |
| 5 | `feat(ingest): implement Finnhub primary with Polygon/Yahoo fallback` | route.ts | npm run build |

---

## Success Criteria

### Verification Commands
```bash
# Build check
npm run build  # Expected: No errors

# Local endpoint test (with env vars set)
curl -H "Authorization: Bearer test" "http://localhost:3000/api/cron/ingest-close"
# Expected: {"status":"SUCCESS","pricesUpdated":4,...} or SKIPPED

# Supabase verification
# Run in Supabase SQL editor:
SELECT symbol, source, date, fetched_at FROM prices_daily ORDER BY fetched_at DESC LIMIT 10;
# Expected: Rows with source = 'finnhub' or fallback provider
```

### Final Checklist
- [ ] All "Must Have" present:
  - [ ] Finnhub `/stock/candle` integration
  - [ ] yahoo-finance2 package integration  
  - [ ] Per-ticker fallback logic
  - [ ] Source tracking per ticker
  - [ ] Idempotency preserved
- [ ] All "Must NOT Have" absent:
  - [ ] SMA calculation unchanged
  - [ ] Verdict logic unchanged
  - [ ] Idempotency logic unchanged
  - [ ] No new DB schema
  - [ ] Polygon code preserved (not deleted)
- [ ] Deployment verified on Vercel
