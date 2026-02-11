# Integrate ingest-close with Notification System

## TL;DR

> **Quick Summary**: Add notification triggers to the `ingest-close` cron job to alert on verdict changes, data staleness, and errors via app notifications + Telegram.
> 
> **Deliverables**:
> - Modified `ingest-close/route.ts` with notification integration
> - Verdict change alerts (ON↔OFF10) sent to ops_notifications + Telegram
> - Error/stale alerts with emergency level
> 
> **Estimated Effort**: Quick (single file, ~50 lines)
> **Parallel Execution**: NO - single sequential task
> **Critical Path**: Task 1 → Done

---

## Context

### Original Request
Integrate the `ingest-close` cron job with the existing notification system to trigger alerts when:
1. Verdict changes (ON↔OFF10) - level: action
2. Error occurs - level: emergency  
3. Cron fails to fetch data (STALE) - level: action

### Interview Summary
**Key Findings**:
- Notification system already exists with all required trigger functions
- `daily/route.ts` already uses the same patterns - can copy approach
- `checkVerdictChanged`, `checkDataStale`, `logIngestFail` ready to use
- Deduplication uses `dedupe_key` with date suffix

**Research Findings**:
- `ingest-close/route.ts:311-316` already fetches `latestSnapshot` for idempotency check
- This snapshot contains previous verdict at `payload_json.verdict`
- Trigger functions imported via `@/lib/ops/notifications`

---

## Work Objectives

### Core Objective
Add notification triggers to `ingest-close/route.ts` following the same pattern as `daily/route.ts`.

### Concrete Deliverables
- `ingest-close/route.ts` modified with 3 notification integration points

### Definition of Done
- [ ] `curl /api/cron/ingest-close` with verdict change → Telegram message received
- [ ] `curl /api/cron/ingest-close` on error → Telegram emergency alert received  
- [ ] `ops_notifications` table has new records with correct `dedupe_key`
- [ ] Duplicate calls within same day do NOT create duplicate notifications

### Must Have
- Import notification triggers at top of file
- VERDICT_CHANGED notification after snapshot upsert
- DATA_STALE notification in no-data branch
- INGEST_FAIL notification in catch block
- Use `verdictDate` (not `getTodayKST()`) for deduplication

### Must NOT Have (Guardrails)
- ❌ Do NOT modify `triggers.ts` - functions already exist
- ❌ Do NOT modify `daily/route.ts` - separate job, can coexist
- ❌ Do NOT change schema or table structure
- ❌ Do NOT add new environment variables
- ❌ Do NOT duplicate notification sending (already handled by `createNotification`)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (no test files for cron routes)
- **User wants tests**: Manual-only
- **Framework**: N/A

### Manual QA

**Verification Type**: API/Backend changes → curl + Supabase + Telegram check

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Single task):
└── Task 1: Integrate notifications into ingest-close route

Critical Path: Task 1 (only task)
Parallel Speedup: N/A - single task
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | None | N/A |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | delegate_task(category="quick", load_skills=["typescript-programmer"]) |

---

## TODOs

- [ ] 1. Integrate Notification Triggers into ingest-close Route

  **What to do**:
  
  1. **Add imports** at top of file (after line 15):
     ```typescript
     import {
       checkVerdictChanged,
       checkDataStale,
       logIngestFail,
     } from "@/lib/ops/notifications";
     ```
  
  2. **Add DATA_STALE notification** in no-data branch (around line 303, after setting health to STALE):
     ```typescript
     // After upsert health to STALE
     await checkDataStale(
       lastSnapshot?.[0]?.verdict_date || null,
       "Polygon API returned no data"
     );
     ```
  
  3. **Add VERDICT_CHANGED notification** after snapshot upsert (after line 382):
     - Get previous verdict from `latestSnapshot` (already fetched at line 311)
     - Compare with new `verdict`
     - Call `checkVerdictChanged` if different
     ```typescript
     // After snapshot upsert, check for verdict change
     const previousVerdict = latestSnapshot?.payload_json?.verdict as "ON" | "OFF10" | undefined;
     if (previousVerdict && previousVerdict !== verdict) {
       await checkVerdictChanged(previousVerdict, verdict, executionDate);
     }
     ```
  
  4. **Add INGEST_FAIL notification** in catch block (around line 415):
     ```typescript
     // After updating health to STALE in catch block
     await logIngestFail("ingest-close", String(error));
     ```

  **Must NOT do**:
  - Do NOT modify the trigger function implementations
  - Do NOT add duplicate notifications (each trigger point should be called once)
  - Do NOT use `getTodayKST()` for verdict change - use `verdictDate`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, <50 lines of code, clear pattern to follow from `daily/route.ts`
  - **Skills**: [`typescript-programmer`]
    - `typescript-programmer`: TypeScript/Next.js route handler modifications

  **Skills Evaluated but Omitted**:
  - `frontend-ui-ux`: No UI changes
  - `git-master`: User will commit separately
  - `dev-browser`: No browser testing needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (single task)
  - **Parallel Group**: N/A
  - **Blocks**: None
  - **Blocked By**: None

  **References** (CRITICAL):

  **Pattern References** (existing code to follow):
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/app/api/cron/daily/route.ts:15-22` - Import pattern for notification triggers
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/app/api/cron/daily/route.ts:99-108` - `checkVerdictChanged` usage pattern
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/app/api/cron/daily/route.ts:93-97` - `checkDataStale` usage pattern
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/app/api/cron/daily/route.ts:172` - `logIngestFail` usage pattern

  **API/Type References**:
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/lib/ops/notifications/triggers.ts:66-102` - `checkVerdictChanged(prev, curr, execDate)` function signature
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/lib/ops/notifications/triggers.ts:22-59` - `checkDataStale(date, reason)` function signature
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/lib/ops/notifications/triggers.ts:175-188` - `logIngestFail(jobType, error)` function signature

  **Target File** (what to modify):
  - `/home/juwon/QuantNeural_wsl/200tq/dashboard/app/api/cron/ingest-close/route.ts`
    - Line 15: Add imports
    - Line 303 (STALE branch): Add `checkDataStale` call
    - Line 382 (after snapshot upsert): Add `checkVerdictChanged` call
    - Line 415 (catch block): Add `logIngestFail` call

  **WHY Each Reference Matters**:
  - `daily/route.ts` imports: Shows exact import path and which functions to import
  - `checkVerdictChanged` usage: Shows how to compare previous vs current verdict
  - `checkDataStale` usage: Shows what parameters to pass for stale scenarios
  - `logIngestFail` usage: Shows job type naming convention

  **Acceptance Criteria**:

  **Manual Execution Verification:**
  
  - [ ] **Build check**: 
    ```bash
    cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run build
    ```
    Expected: No TypeScript errors

  - [ ] **Start dev server**:
    ```bash
    cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run dev
    ```
  
  - [ ] **Test SUCCESS case** (normal execution):
    ```bash
    curl -X GET "http://localhost:3000/api/cron/ingest-close" \
      -H "Authorization: Bearer $CRON_SECRET"
    ```
    Expected response contains: `"status": "SUCCESS"` or `"status": "SKIPPED"`
    Verify: No notification created if verdict unchanged (check Supabase)

  - [ ] **Test VERDICT_CHANGED case**:
    - Manually set `latestSnapshot.payload_json.verdict` in Supabase to opposite of current
    - Run curl again
    - Expected: Telegram message received with "시그널 변경"
    - Verify: `ops_notifications` has record with `event_type = 'VERDICT_CHANGED'`

  - [ ] **Verify deduplication**:
    - Run curl twice in succession
    - Expected: Second call should NOT create duplicate notification
    - Verify: Only one record per `dedupe_key` in `ops_notifications`

  **Evidence Required:**
  - [ ] `npm run build` output shows success
  - [ ] curl response JSON captured
  - [ ] Telegram screenshot (if verdict changed)
  - [ ] Supabase `ops_notifications` query showing new record

  **Commit**: YES
  - Message: `feat(ingest-close): integrate with notification system for verdict changes`
  - Files: `app/api/cron/ingest-close/route.ts`
  - Pre-commit: `npm run build`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(ingest-close): integrate with notification system for verdict changes` | `app/api/cron/ingest-close/route.ts` | `npm run build` |

---

## Success Criteria

### Verification Commands
```bash
# Build verification
cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run build
# Expected: Compiled successfully

# Check file has imports
grep -n "checkVerdictChanged" app/api/cron/ingest-close/route.ts
# Expected: Line with import statement + line with function call
```

### Final Checklist
- [ ] All "Must Have" present:
  - [ ] Import statement added
  - [ ] `checkVerdictChanged` called after snapshot upsert
  - [ ] `checkDataStale` called in STALE branch
  - [ ] `logIngestFail` called in catch block
- [ ] All "Must NOT Have" absent:
  - [ ] `triggers.ts` unchanged
  - [ ] `daily/route.ts` unchanged
  - [ ] No new env vars
- [ ] Build passes
- [ ] Notifications work end-to-end
