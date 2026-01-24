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

// Ticker configuration
const TICKERS = ['TQQQ', 'QQQ', 'SPLG', 'SGOV'] as const;
const SMA_BUFFER_DAYS = 220; // Extra buffer for SMA170

// Polygon API config
const POLYGON_BASE_URL = 'https://api.polygon.io';

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
  source: string;
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

/**
 * Convert Polygon bar to price row
 */
function barToPriceRow(ticker: string, bar: PolygonDailyBar): PriceRow {
  const date = new Date(bar.t).toISOString().split('T')[0];
  return {
    date,
    symbol: ticker,
    open: bar.o,
    high: bar.h,
    low: bar.l,
    close: bar.c,
    volume: Math.floor(bar.v),
    source: 'polygon',
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
  
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    return NextResponse.json({ error: 'Supabase not configured' }, { status: 500 });
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey);
  
  try {
    // 1. Fetch latest bars from Polygon
    const priceRows: PriceRow[] = [];
    let verdictDate: string | null = null;
    
    for (const ticker of TICKERS) {
      const bar = await fetchLatestBar(ticker, polygonKey);
      if (bar) {
        const row = barToPriceRow(ticker, bar);
        priceRows.push(row);
        
        // Use QQQ's date as verdict date
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
      }
      
      return NextResponse.json({ 
        status: 'STALE', 
        reason: 'No data from Polygon',
        updatedRows: 0,
      });
    }
    
    // 2. Check idempotency - if FRESH snapshot already exists for this date, skip
    const { data: existingSnapshot } = await supabase
      .from('ops_snapshots_daily')
      .select('health')
      .eq('verdict_date', verdictDate)
      .single();
    
    if (existingSnapshot?.health === 'FRESH') {
      return NextResponse.json({ 
        status: 'SKIPPED', 
        reason: `FRESH snapshot already exists for ${verdictDate}`,
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
        source: 'polygon',
        lastTradingDate: verdictDate,
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
    
    return NextResponse.json({ 
      status: 'ERROR', 
      error: String(error),
    }, { status: 500 });
  }
}
