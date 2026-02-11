# Draft: Integrate ingest-close with Notification System

## Requirements (confirmed)

- **Verdict change notification**: When ON↔OFF10 changes, level: action
- **Error notification**: When cron fails, level: emergency
- **Stale notification**: When data fetch fails, level: action
- **Destinations**: ops_notifications table + Telegram

## Technical Decisions

- **Integration Point**: Modify `ingest-close/route.ts` directly
- **Trigger Functions**: Reuse existing `lib/ops/notifications/triggers.ts`
- **Deduplication**: Use `verdictDate` in dedupe_key for verdict changes

## Research Findings

### Current `ingest-close/route.ts` Flow:
1. Lines 266-277: Fetch prices for all tickers
2. Lines 279-308: Handle STALE case (no data) - **ADD DATA_STALE notification here**
3. Lines 353-354: Calculate verdict (sma3 vs sma160/165/170)
4. Lines 373-382: Upsert snapshot - **ADD VERDICT_CHANGED check after this**
5. Lines 393-419: Catch block - **ADD INGEST_FAIL notification here**

### Missing Piece:
- Need to fetch **previous verdict** before calculating new one
- Use `latestSnapshot` (already fetched at line 311) for comparison

### Existing Trigger Functions (ready to use):
- `checkVerdictChanged(prevVerdict, currVerdict, executionDate)` 
- `checkDataStale(lastPriceDate, reason)`
- `logIngestFail(jobType, errorMessage)`

### Deduplication Pattern:
- Current: Uses `getTodayKST()` for date in dedupe_key
- For `ingest-close`: Should use `verdictDate` (the actual trading date)
- This ensures one notification per trading day, not per calendar day

## Scope Boundaries

### INCLUDE:
1. Import notification triggers in `ingest-close/route.ts`
2. Add VERDICT_CHANGED check after snapshot upsert
3. Add DATA_STALE notification in no-data branch
4. Add INGEST_FAIL notification in catch block
5. Pass `verdictDate` for deduplication

### EXCLUDE:
- Changes to `triggers.ts` (functions already exist)
- Changes to `daily/route.ts` (separate job, can coexist)
- Schema changes (ops_notifications table already exists)
- New Telegram configuration (already set up)

## Test Strategy Decision

- **Infrastructure exists**: YES (Next.js app with Supabase)
- **User wants tests**: Need to confirm
- **QA approach**: Manual verification via curl + Supabase check + Telegram
