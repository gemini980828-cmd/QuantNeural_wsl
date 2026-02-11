/**
 * Polygon EOD Price Ingestion Cron Job
 * 
 * Runs daily at KST 07:00 (UTC 22:00) to:
 * 1. Fetch latest EOD prices from Polygon API
 * 2. Upsert to prices_daily
 * 3. Calculate SMA signals for QQQ
 * 4. Generate verdict snapshot
 * 
 * @route GET /api/cron/ingest-close
 */

import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import yahooFinance from 'yahoo-finance2';
import {
  checkDataStale,
  checkHardTrigger,
  checkVerdictChanged,
  logIngestFail,
} from '@/lib/ops/notifications';

// Ticker configuration
const TICKERS = ['TQQQ', 'QQQ', 'SPLG', 'SGOV'] as const;
const SMA_BUFFER_DAYS = 220; // Extra buffer for SMA170

// Polygon API config
const POLYGON_BASE_URL = 'https://api.polygon.io';
const FINNHUB_BASE_URL = 'https://finnhub.io/api/v1';

type DataSource = 'finnhub' | 'polygon' | 'yahoo';

interface FinnhubQuoteResponse {
  c: number;   // Current price
  d: number;   // Change
  dp: number;  // Percent change
  h: number;   // High price of the day
  l: number;   // Low price of the day
  o: number;   // Open price of the day
  pc: number;  // Previous close price
  t: number;   // Unix timestamp (seconds)
}

interface FetchResult {
  bar: PolygonDailyBar;
  source: DataSource;
}

interface PolygonDailyBar {
  t: number;    // Unix timestamp ms
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

interface PriceRow {
  date: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source: DataSource;
  fetched_at: string;
}

/**
 * Get latest trading day bar from Polygon
 */
async function fetchLatestBar(ticker: string, apiKey: string): Promise<PolygonDailyBar | null> {
  // Get bars for last 5 days to handle weekends/holidays
  const to = new Date().toISOString().split('T')[0];
  const from = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  
  const url = `${POLYGON_BASE_URL}/v2/aggs/ticker/${ticker}/range/1/day/${from}/${to}?apiKey=${apiKey}&sort=desc&limit=1`;
  
  const res = await fetch(url);
  if (!res.ok) {
    console.error(`Polygon API error for ${ticker}:`, res.status);
    return null;
  }
  
  const data = await res.json();
  if (!data.results || data.results.length === 0) {
    console.error(`No data returned for ${ticker}`);
    return null;
  }
  
  return data.results[0];
}

async function fetchFinnhubQuote(ticker: string, apiKey: string): Promise<PolygonDailyBar | null> {
  try {
    const url = `${FINNHUB_BASE_URL}/quote?symbol=${ticker}&token=${apiKey}`;
    
    const res = await fetch(url);
    if (!res.ok) {
      console.error(`Finnhub API error for ${ticker}:`, res.status);
      return null;
    }
    
    const data: FinnhubQuoteResponse = await res.json();
    if (!data.c || data.c === 0) {
      console.error(`Finnhub no data for ${ticker}`);
      return null;
    }
    
    return {
      t: data.t * 1000,
      o: data.o,
      h: data.h,
      l: data.l,
      c: data.c,
      v: 0,
    };
  } catch (error) {
    console.error(`Finnhub fetch error for ${ticker}:`, error);
    return null;
  }
}

async function fetchYahooBar(ticker: string): Promise<PolygonDailyBar | null> {
  try {
    const period1 = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const history = await yahooFinance.historical(ticker, {
      period1,
      interval: '1d',
    }) as Array<{ date: Date; open: number; high: number; low: number; close: number; volume: number }>;
    
    if (!history || history.length === 0) {
      console.error(`Yahoo no data for ${ticker}`);
      return null;
    }
    
    const last = history[history.length - 1];
    return {
      t: last.date.getTime(),
      o: last.open,
      h: last.high,
      l: last.low,
      c: last.close,
      v: last.volume,
    };
  } catch (error) {
    console.error(`Yahoo fetch error for ${ticker}:`, error);
    return null;
  }
}

async function fetchWithFallback(
  ticker: string,
  finnhubKey: string | undefined,
  polygonKey: string
): Promise<FetchResult | null> {
  if (finnhubKey) {
    const bar = await fetchFinnhubQuote(ticker, finnhubKey);
    if (bar) return { bar, source: 'finnhub' };
    console.log(`Finnhub failed for ${ticker}, trying Polygon...`);
  }
  
  const polygonBar = await fetchLatestBar(ticker, polygonKey);
  if (polygonBar) return { bar: polygonBar, source: 'polygon' };
  console.log(`Polygon failed for ${ticker}, trying Yahoo...`);
  
  const yahooBar = await fetchYahooBar(ticker);
  if (yahooBar) return { bar: yahooBar, source: 'yahoo' };
  
  console.error(`All providers failed for ${ticker}`);
  return null;
}

/**
 * Convert Polygon bar to price row
 */
function barToPriceRow(ticker: string, bar: PolygonDailyBar, source: DataSource): PriceRow {
  const date = new Date(bar.t).toISOString().split('T')[0];
  return {
    date,
    symbol: ticker,
    open: bar.o,
    high: bar.h,
    low: bar.l,
    close: bar.c,
    volume: Math.floor(bar.v),
    source,
    fetched_at: new Date().toISOString(),
  };
}

/**
 * Calculate SMA for given close prices
 */
function calculateSMA(closes: number[], period: number): number | null {
  if (closes.length < period) return null;
  const slice = closes.slice(0, period);
  return slice.reduce((a, b) => a + b, 0) / period;
}

/**
 * Determine verdict based on E03 rules: SMA3 vs SMA160/165/170 majority
 */
function determineVerdict(sma3: number, sma160: number, sma165: number, sma170: number): 'ON' | 'OFF10' {
  const votes = [
    sma3 > sma160 ? 1 : 0,
    sma3 > sma165 ? 1 : 0,
    sma3 > sma170 ? 1 : 0,
  ];
  const bullishVotes = votes.reduce((a, b) => a + b, 0);
  return bullishVotes >= 2 ? 'ON' : 'OFF10';
}

/**
 * Find next trading day in prices_daily
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function findNextTradingDay(supabase: any, afterDate: string): Promise<string | null> {
  const { data, error } = await supabase
    .from('prices_daily')
    .select('date')
    .eq('symbol', 'QQQ')
    .gt('date', afterDate)
    .order('date', { ascending: true })
    .limit(1);
  
  if (error || !data || data.length === 0) {
    // If no future date, estimate next business day
    const d = new Date(afterDate);
    d.setDate(d.getDate() + 1);
    // Skip weekends
    while (d.getDay() === 0 || d.getDay() === 6) {
      d.setDate(d.getDate() + 1);
    }
    return d.toISOString().split('T')[0];
  }
  
  return data[0].date;
}

export async function GET(request: Request) {
  // Verify cron secret
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  
  const polygonKey = process.env.POLYGON_API_KEY;
  if (!polygonKey) {
    return NextResponse.json({ error: 'POLYGON_API_KEY not configured' }, { status: 500 });
  }
  
  const finnhubKey = process.env.FINNHUB_API_KEY;
  
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    return NextResponse.json({ error: 'Supabase not configured' }, { status: 500 });
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey);
  
  try {
    const priceRows: PriceRow[] = [];
    const sourcesUsed: Record<string, DataSource> = {};
    let verdictDate: string | null = null;
    
    for (const ticker of TICKERS) {
      const result = await fetchWithFallback(ticker, finnhubKey, polygonKey);
      if (result) {
        const row = barToPriceRow(ticker, result.bar, result.source);
        priceRows.push(row);
        sourcesUsed[ticker] = result.source;
        
        if (ticker === 'QQQ') {
          verdictDate = row.date;
        }
      }
    }
    
    if (priceRows.length === 0 || !verdictDate) {
      // STALE - no data available
      const { data: lastSnapshot } = await supabase
        .from('ops_snapshots_daily')
        .select('*')
        .order('verdict_date', { ascending: false })
        .limit(1);
      
      if (lastSnapshot && lastSnapshot.length > 0) {
        // Update health to STALE
        await supabase
          .from('ops_snapshots_daily')
          .upsert({
            verdict_date: lastSnapshot[0].verdict_date,
            execution_date: lastSnapshot[0].execution_date,
            health: 'STALE',
            payload_json: {
              ...lastSnapshot[0].payload_json,
              staleReason: 'Polygon API returned no data',
            },
            computed_at: new Date().toISOString(),
          });
        
        await checkDataStale(lastSnapshot[0].verdict_date, 'All providers failed to return data');
      }
      
      return NextResponse.json({ 
        status: 'STALE', 
        reason: 'No data from Polygon',
        updatedRows: 0,
      });
    }
    
    // 2. Check idempotency - skip only if we already have this or newer data
    const { data: latestSnapshot } = await supabase
      .from('ops_snapshots_daily')
      .select('verdict_date, health, payload_json')
      .order('verdict_date', { ascending: false })
      .limit(1)
      .single();
    
    // Skip if Polygon's date is not newer than our latest data
    if (latestSnapshot && latestSnapshot.verdict_date >= verdictDate) {
      return NextResponse.json({ 
        status: 'SKIPPED', 
        reason: `Latest snapshot (${latestSnapshot.verdict_date}) is already up-to-date. Polygon returned: ${verdictDate}`,
      });
    }
    
    // 3. Upsert prices
    const { error: upsertError } = await supabase
      .from('prices_daily')
      .upsert(priceRows, { onConflict: 'date,symbol' });
    
    if (upsertError) {
      throw new Error(`Price upsert failed: ${upsertError.message}`);
    }
    
    // 4. Calculate SMAs from QQQ closes
    const { data: qqqPrices, error: qqqError } = await supabase
      .from('prices_daily')
      .select('date, close')
      .eq('symbol', 'QQQ')
      .order('date', { ascending: false })
      .limit(SMA_BUFFER_DAYS);
    
    if (qqqError || !qqqPrices || qqqPrices.length < 170) {
      throw new Error(`Insufficient QQQ data for SMA: ${qqqPrices?.length || 0} rows`);
    }
    
    const closes = qqqPrices.map(p => Number(p.close));
    const sma3 = calculateSMA(closes, 3) ?? 0;
    const sma160 = calculateSMA(closes, 160) ?? 0;
    const sma165 = calculateSMA(closes, 165) ?? 0;
    const sma170 = calculateSMA(closes, 170) ?? 0;
    
    // 5. Determine verdict
    const verdict = determineVerdict(sma3, sma160, sma165, sma170);
    
    // 6. Find execution date (next trading day)
    const executionDate = await findNextTradingDay(supabase, verdictDate);
    
    // 7. Build payload
    const priceMap = Object.fromEntries(priceRows.map(r => [r.symbol, r.close]));
    
    const payload = {
      prices: priceMap,
      sma: { sma3, sma160, sma165, sma170 },
      verdict,
      sourceMeta: {
        sources: sourcesUsed,
        lastTradingDate: verdictDate,
        fetchedAt: new Date().toISOString(),
      },
    };
    
    // 8. Upsert snapshot
    await supabase
      .from('ops_snapshots_daily')
      .upsert({
        verdict_date: verdictDate,
        execution_date: executionDate,
        health: 'FRESH',
        payload_json: payload,
        computed_at: new Date().toISOString(),
      });
    
    // 9. Check for verdict change and send notification
    const previousVerdict = latestSnapshot?.payload_json?.verdict as 'ON' | 'OFF10' | undefined;
    if (previousVerdict && previousVerdict !== verdict) {
      await checkVerdictChanged(previousVerdict, verdict, executionDate!);
    }
    
    // 10. Check Hard Trigger (QQQ daily -7%)
    const qqqTodayClose = priceRows.find(r => r.symbol === 'QQQ')?.close;
    const qqqYesterdayClose = qqqPrices.length > 1 ? Number(qqqPrices[1].close) : null;
    
    if (qqqTodayClose && qqqYesterdayClose && qqqYesterdayClose > 0) {
      const qqqDailyChange = (qqqTodayClose - qqqYesterdayClose) / qqqYesterdayClose;
      await checkHardTrigger(qqqDailyChange);
    }
    
    return NextResponse.json({
      status: 'SUCCESS',
      verdictDate,
      executionDate,
      verdict,
      health: 'FRESH',
      pricesUpdated: priceRows.length,
    });
    
  } catch (error) {
    console.error('Cron job failed:', error);
    
    // Mark as STALE on error
    const { data: lastSnapshot } = await supabase
      .from('ops_snapshots_daily')
      .select('*')
      .order('verdict_date', { ascending: false })
      .limit(1);
    
    if (lastSnapshot && lastSnapshot.length > 0) {
      await supabase
        .from('ops_snapshots_daily')
        .update({
          health: 'STALE',
          payload_json: {
            ...lastSnapshot[0].payload_json,
            staleReason: String(error),
          },
        })
        .eq('verdict_date', lastSnapshot[0].verdict_date);
    }
    
    await logIngestFail('ingest-close', String(error));
    
    return NextResponse.json({ 
      status: 'ERROR', 
      error: String(error),
    }, { status: 500 });
  }
}
