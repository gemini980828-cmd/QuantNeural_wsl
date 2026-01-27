# Records Compare Feature - IMPLEMENTATION COMPLETE ✅

**Date**: 2026-01-26
**Plan**: .sisyphus/plans/records-compare.md
**Status**: Implementation Complete (TODO 1-6), Manual Verification Pending (TODO 7)

---

## 🎯 Feature Overview

**Objective**: Enable "예상 vs 실제" comparison in Records page to calculate real quality metrics (accuracy, slippage) instead of hardcoded values.

**Completion**: 6/7 tasks complete (100% implementation, manual verification pending)

---

## ✅ Completed Tasks

### TODO 1: DB Schema ✅
**File**: `dashboard/supabase-schema.sql`
**Changes**:
- Added `expected_lines jsonb` column to `trade_executions` table (line 24-25)
- Nullable for backward compatibility
- Inline documentation of JSON structure

**SQL Migration** (user must run in Supabase):
```sql
ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS expected_lines jsonb;
COMMENT ON COLUMN trade_executions.expected_lines IS 'Expected trades from strategy: [{symbol, side, qty, expectedPrice?}]';
```

---

### TODO 2: Type Definitions ✅
**Files**: 
- `dashboard/lib/types.ts`
- `dashboard/lib/ops/e03/types.ts`
- `dashboard/lib/ops/e03/buildViewModel.ts`
- `dashboard/app/(shell)/records/page.tsx`

**Changes**:
- Added `expectedPrice?: number` to `TradeLine` interface (DB type)
- Added `inputPrices?: Record<string, number>` to `E03ViewModel`
- Added `expected_lines` to `TradeExecutionRecord` interface
- Exposed `inputPrices` from buildViewModel return value

---

### TODO 3: Storage Layer ✅
**File**: `dashboard/lib/ops/e03/storage.ts`

**Changes**:
- Extended `saveRecordToSupabase()` signature:
  ```typescript
  saveRecordToSupabase(
    executionDateLabel: string,
    record: ManualRecord,
    expectedTrades?: any[],        // NEW
    inputPrices?: Record<string, number>  // NEW
  )
  ```
- Implemented e03 → DB format conversion:
  - `ticker` → `symbol`
  - `action` → `side` (cast to 'BUY' | 'SELL')
  - `shares` → `qty`
  - `inputPrices[ticker]` → `expectedPrice`
- Conditionally includes `expectedLines` in POST body

---

### TODO 4: API Route ✅
**File**: `dashboard/app/api/record/route.ts`

**Changes**:
- Added `expectedLines` to request body destructuring (line 72)
- Type: `expectedLines?: TradeLine[]`
- Saves to `expected_lines` column: `expected_lines: expectedLines || null` (line 105)
- Backward compatible (null allowed)

---

### TODO 5: ZoneCOpsConsole ✅
**File**: `dashboard/components/e03/ZoneCOpsConsole.tsx`

**Changes**:
- Updated `handleRecordSave` to pass two additional arguments (line 91):
  ```typescript
  const result = await saveRecordToSupabase(
    vm.executionDateLabel,
    record,
    vm.expectedTrades,    // NEW
    vm.inputPrices        // NEW
  );
  ```
- Data flows from viewModel → storage → API → DB

---

### TODO 6: Records Page ✅
**File**: `dashboard/app/(shell)/records/page.tsx`

**Changes**:
- Removed hardcoded values:
  - ❌ `const accuracy = 98.5`
  - ❌ `const slippage = 0.12`
- Implemented real comparison logic in `useMemo` (lines 105-136):
  - Iterates through executed records
  - Compares `expected_lines` vs `lines`
  - **Accuracy**: `(matched trades / total expected) × 100`
  - **Slippage**: `avg(|actual.price - expected.expectedPrice| / expected.expectedPrice) × 100`
  - Null safety: handles records without `expected_lines`

---

### TODO 7: E2E Verification ⏳ PENDING
**Status**: Manual verification required (automated Playwright failed)

**Instructions**: See `verification-manual.md` for step-by-step manual testing guide

**What to verify**:
1. Command page → Save record → Check Network tab for `expectedLines`
2. Supabase → Verify `expected_lines` saved to database
3. Records page → Verify QualityAnalytics shows calculated values (not 98.5% / 0.12%)

---

## 📊 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Command Page (buildViewModel)                              │
│ ─────────────────────────────────────────────────────────── │
│ • vm.expectedTrades: [{ action, ticker, shares }]          │
│ • vm.inputPrices: { TQQQ: 85.23, SGOV: 100.50 }           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ ZoneCOpsConsole.handleRecordSave()                         │
│ ─────────────────────────────────────────────────────────── │
│ saveRecordToSupabase(date, record, expectedTrades, prices) │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Storage Layer (lib/ops/e03/storage.ts)                     │
│ ─────────────────────────────────────────────────────────── │
│ • Convert e03 → DB format                                  │
│ • { ticker, action, shares } → { symbol, side, qty,        │
│   expectedPrice }                                          │
│ • POST /api/record with expectedLines                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ API Route (app/api/record/route.ts)                        │
│ ─────────────────────────────────────────────────────────── │
│ INSERT INTO trade_executions (                             │
│   ...,                                                     │
│   expected_lines: [{ symbol, side, qty, expectedPrice }]  │
│ )                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Records Page (app/(shell)/records/page.tsx)                │
│ ─────────────────────────────────────────────────────────── │
│ • Load expected_lines from DB                              │
│ • Compare: expected_lines vs lines                         │
│ • Calculate accuracy = (matches / total) × 100             │
│ • Calculate slippage = avg(|actual - expected| /           │
│   expected) × 100                                          │
│ ✅ Display REAL metrics                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Files Modified

```
dashboard/
├── supabase-schema.sql                    [DB schema]
├── lib/
│   ├── types.ts                           [TradeLine type]
│   └── ops/e03/
│       ├── types.ts                       [E03ViewModel type]
│       ├── buildViewModel.ts              [Expose inputPrices]
│       └── storage.ts                     [Conversion logic]
├── app/
│   ├── api/record/route.ts                [Save expected_lines]
│   └── (shell)/records/page.tsx           [Comparison logic]
└── components/e03/
    └── ZoneCOpsConsole.tsx                [Pass data]
```

---

## 🔍 Verification Checklist

**Implementation (Complete)**
- [x] TODO 1: DB Schema
- [x] TODO 2: Type Definitions
- [x] TODO 3: Storage Layer
- [x] TODO 4: API Route
- [x] TODO 5: ZoneCOpsConsole
- [x] TODO 6: Records Page

**Manual Verification (Pending)**
- [ ] TODO 7: E2E Flow Test
  - [ ] Command page loads
  - [ ] expectedLines in POST request
  - [ ] expected_lines in Supabase
  - [ ] Real values in Records page
  - [ ] Screenshot captured

---

## 🚀 Next Steps

### 1. Run Database Migration
```sql
-- Execute in Supabase SQL Editor
ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS expected_lines jsonb;
COMMENT ON COLUMN trade_executions.expected_lines IS 'Expected trades from strategy: [{symbol, side, qty, expectedPrice?}]';
```

### 2. Manual Verification
Follow instructions in `verification-manual.md`:
1. Start dev server: `cd dashboard && bun run dev`
2. Test Command → Records flow
3. Verify expectedLines in Network tab
4. Check Supabase database
5. Confirm real metrics displayed

### 3. Commit Changes
After verification passes:
```bash
git add dashboard/
git commit -m "feat(records): implement expected vs actual comparison

- Add expected_lines column to trade_executions table
- Implement e03→DB format conversion in storage layer
- Save expectedLines via POST /api/record
- Calculate real accuracy/slippage in Records page
- Replace hardcoded 98.5%/0.12% with dynamic calculation

Implements: Records Compare feature (E03 SSOT Task #1)
"
```

---

## 📚 Technical Notes

### Type Duality
Two separate TradeLine types maintained:
- **e03 TradeLine** (UI): `{ action, ticker, shares, note }`
- **DB TradeLine** (Storage): `{ symbol, side, qty, price, expectedPrice, note }`

Conversion happens in `storage.ts` before API call.

### Backward Compatibility
- `expected_lines` column is nullable
- Old records without `expected_lines` handled gracefully
- Accuracy/slippage show 0% when no comparison data available

### Performance
- Comparison logic uses `useMemo` to recalculate only when records change
- No unnecessary re-renders in Records page

---

**Implementation Status**: ✅ COMPLETE
**Manual Verification**: ⏳ PENDING
**Ready for Production**: After manual verification passes
