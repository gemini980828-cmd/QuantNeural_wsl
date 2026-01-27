# Known Issues & Gotchas

## Current State Issues
1. **Hardcoded Stats**: Records page shows `accuracy = 98.5%`, `slippage = 0.12%` (lines 107-108 in records/page.tsx)
2. **Missing Storage**: `saveRecordToSupabase()` doesn't accept expectedTrades parameter
3. **API Gap**: POST /api/record doesn't save expected_lines
4. **DB Gap**: No expected_lines column in trade_executions table

## Potential Pitfalls
- **Type Confusion**: Don't modify e03 TradeLine - only DB TradeLine
- **Null Safety**: Must handle records where expected_lines is null
- **Price Lookup**: inputPrices may not have all tickers - handle gracefully
- **Side Conversion**: e03 uses "BUY"/"SELL", DB uses side enum - ensure consistency

## Dependencies
- Supabase must be running for DB migration
- Dev server needed for API testing
- Playwright for E2E verification
