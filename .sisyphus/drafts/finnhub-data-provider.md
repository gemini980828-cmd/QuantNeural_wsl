# Draft: Finnhub Data Provider Integration

## Requirements (confirmed)
- **Primary goal**: Replace/supplement Polygon (delayed data on free tier) with faster source
- **Target file**: `/home/juwon/QuantNeural_wsl/200tq/dashboard/app/api/cron/ingest-close/route.ts`
- **Tickers**: QQQ, TQQQ, SPLG, SGOV (per existing code, line 17)
- **Database**: Supabase `prices_daily` table
- **Deployment**: Vercel serverless

## Technical Decisions
- **Finnhub as PRIMARY**: Real-time on free tier, 60 calls/min
- **Polygon as FALLBACK**: Keep existing code, demote to fallback
- **yahoo-finance2**: OPTIONAL tertiary fallback (TBD by user)

## Research Findings
- **Finnhub API**: `https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}`
- **Response**: `{ c: currentPrice, pc: previousClose, o: open, h: high, l: low, t: timestamp }`
- **Note**: Finnhub returns QUOTE data (intraday), not EOD bars - may need different handling

## Current Implementation Notes
- `PriceRow` interface already has `source` field (line 40)
- Currently hardcoded to `'polygon'` (line 82)
- Idempotency logic checks `ops_snapshots_daily.verdict_date` (lines 209-222)
- Error handling marks data as STALE (lines 293-311)

## Open Questions
- [ ] Ticker confirmation: SPLG correct, or should be SQQQ?
- [ ] yahoo-finance2: Include in plan or skip?
- [ ] Finnhub NOTE: Returns quote (live), not EOD - is this acceptable for dashboard use case?

## Scope Boundaries
- INCLUDE: Finnhub integration, fallback logic, source tracking
- INCLUDE: Environment variable setup
- EXCLUDE: SMA calculation changes (already working)
- EXCLUDE: UI changes (this is backend only)
