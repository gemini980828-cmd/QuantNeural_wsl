import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

type ColorTone = 'ok' | 'action' | 'danger';

interface MacroData {
  vix: { value: number | null; color: ColorTone; change: number | null };
  fng: { value: number | null; label: string; color: ColorTone; change: number | null };
  treasury: { value: number | null; change: number | null };
  dxy: { value: number | null; change: number | null };
  nq: { value: number | null; change: number | null };
  usdkrw: { value: number | null; change: number | null };
  vix3m: { value: number | null; change: number | null };
  sp500: { value: number | null; change: number | null };
  esFutures: { value: number | null; change: number | null };
  gold: { value: number | null; change: number | null };
  oil: { value: number | null; change: number | null };
  btc: { value: number | null; change: number | null };
  yieldCurve?: { value: number | null; color: ColorTone; change: number | null };
  hySpread?: { value: number | null; color: ColorTone; change: number | null };
  treasury2y?: { value: number | null; change: number | null };
  updatedAt: string;
}

function getVixColor(value: number): ColorTone {
  if (value < 15) return 'ok';
  if (value <= 25) return 'action';
  return 'danger';
}

function getFngColor(value: number): ColorTone {
  if (value <= 25) return 'danger';
  if (value <= 50) return 'action';
  if (value <= 75) return 'ok';
  return 'action';
}

function getYieldCurveColor(value: number): ColorTone {
  if (value > 0.5) return 'ok';
  if (value >= 0) return 'action';
  return 'danger';
}

function getHySpreadColor(value: number): ColorTone {
  if (value < 3) return 'ok';
  if (value <= 5) return 'action';
  return 'danger';
}

interface YahooQuoteResult {
  value: number | null;
  prevClose: number | null;
  change: number | null;
}

async function fetchYahooQuote(symbol: string): Promise<YahooQuoteResult> {
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=1d`;
    const res = await fetch(url, {
      cache: 'no-store',
      headers: { 'User-Agent': 'Mozilla/5.0' }
    });
    if (!res.ok) return { value: null, prevClose: null, change: null };
    
    const data = await res.json();
    const meta = data.chart?.result?.[0]?.meta;
    const value = meta?.regularMarketPrice ?? null;
    const prevClose = meta?.chartPreviousClose ?? meta?.previousClose ?? null;
    
    let change: number | null = null;
    if (value !== null && prevClose !== null && prevClose !== 0) {
      change = ((value - prevClose) / prevClose) * 100;
    }
    
    return { value, prevClose, change };
  } catch {
    return { value: null, prevClose: null, change: null };
  }
}

async function fetchFredSeries(seriesId: string): Promise<{ value: number | null; change: number | null }> {
  const apiKey = process.env.FRED_API_KEY;
  if (!apiKey) return { value: null, change: null };
  
  try {
    const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=${apiKey}&file_type=json&limit=2&sort_order=desc`;
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) return { value: null, change: null };
    
    const data = await res.json();
    const observations = data.observations || [];
    if (observations.length === 0) return { value: null, change: null };
    
    const latest = parseFloat(observations[0].value);
    const previous = observations.length > 1 ? parseFloat(observations[1].value) : null;
    const change = previous !== null ? latest - previous : null;
    
    return { value: isNaN(latest) ? null : latest, change };
  } catch {
    return { value: null, change: null };
  }
}

async function fetchVix(): Promise<{ value: number | null; color: ColorTone; change: number | null }> {
  const { value, change } = await fetchYahooQuote('^VIX');
  return {
    value,
    color: value !== null ? getVixColor(value) : 'action',
    change,
  };
}

async function fetchFearGreed(): Promise<{ value: number | null; label: string; color: ColorTone; change: number | null }> {
  try {
    const res = await fetch('https://api.alternative.me/fng/?limit=2', { cache: 'no-store' });
    if (!res.ok) throw new Error(`FNG API error: ${res.status}`);
    
    const json = await res.json();
    const today = json.data[0];
    const yesterday = json.data[1];
    
    const value = parseInt(today.value);
    const label = today.value_classification;
    const prevValue = yesterday ? parseInt(yesterday.value) : null;
    const change = prevValue !== null ? value - prevValue : null;
    
    return {
      value,
      label,
      color: getFngColor(value),
      change,
    };
  } catch (error) {
    console.error('Failed to fetch Fear & Greed:', error);
    return { value: null, label: '', color: 'action', change: null };
  }
}

async function fetchTreasury10Y(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('^TNX');
  return { value, change };
}

async function fetchDXY(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('DX-Y.NYB');
  return { value, change };
}

async function fetchNasdaqFutures(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('NQ=F');
  return { value, change };
}

async function fetchUsdKrw(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('USDKRW=X');
  return { value, change };
}

async function fetchVix3m(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('^VIX3M');
  return { value, change };
}

async function fetchSp500(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('^GSPC');
  return { value, change };
}

async function fetchEsFutures(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('ES=F');
  return { value, change };
}

async function fetchGold(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('GC=F');
  return { value, change };
}

async function fetchOil(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('CL=F');
  return { value, change };
}

async function fetchBtc(): Promise<{ value: number | null; change: number | null }> {
  const { value, change } = await fetchYahooQuote('BTC-USD');
  return { value, change };
}

async function fetchYieldCurve(): Promise<{ value: number | null; change: number | null }> {
  return await fetchFredSeries('T10Y2Y');
}

async function fetchHySpread(): Promise<{ value: number | null; change: number | null }> {
  return await fetchFredSeries('BAMLH0A0HYM2');
}

async function fetchTreasury2Y(): Promise<{ value: number | null; change: number | null }> {
  return await fetchFredSeries('DGS2');
}

async function fetchAllMacroData(): Promise<MacroData> {
  const [vix, fng, treasury, dxy, nq, usdkrw, vix3m, sp500, esFutures, gold, oil, btc, yieldCurveData, hySpreadData, treasury2y] = await Promise.all([
    fetchVix(),
    fetchFearGreed(),
    fetchTreasury10Y(),
    fetchDXY(),
    fetchNasdaqFutures(),
    fetchUsdKrw(),
    fetchVix3m(),
    fetchSp500(),
    fetchEsFutures(),
    fetchGold(),
    fetchOil(),
    fetchBtc(),
    fetchYieldCurve(),
    fetchHySpread(),
    fetchTreasury2Y(),
  ]);

  const data: MacroData = {
    vix,
    fng,
    treasury,
    dxy,
    nq,
    usdkrw,
    vix3m,
    sp500,
    esFutures,
    gold,
    oil,
    btc,
    updatedAt: new Date().toISOString(),
  };

  if (yieldCurveData.value !== null) {
    data.yieldCurve = {
      value: yieldCurveData.value,
      color: getYieldCurveColor(yieldCurveData.value),
      change: yieldCurveData.change,
    };
  }

  if (hySpreadData.value !== null) {
    data.hySpread = {
      value: hySpreadData.value,
      color: getHySpreadColor(hySpreadData.value),
      change: hySpreadData.change,
    };
  }

  if (treasury2y.value !== null) {
    data.treasury2y = treasury2y;
  }

  return data;
}

function getSupabaseClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) return null;
  return createClient(url, key);
}

async function getCachedMacro(): Promise<MacroData | null> {
  const supabase = getSupabaseClient();
  if (!supabase) return null;
  
  try {
    const { data, error } = await supabase
      .from('macro_cache')
      .select('data_json, updated_at')
      .eq('id', 'latest')
      .single();
    
    if (error || !data) return null;
    return data.data_json as MacroData;
  } catch {
    return null;
  }
}

async function saveMacroCache(macroData: MacroData): Promise<void> {
  const supabase = getSupabaseClient();
  if (!supabase) return;
  
  try {
    await supabase
      .from('macro_cache')
      .upsert({
        id: 'latest',
        data_json: macroData,
        updated_at: new Date().toISOString(),
      });
  } catch (error) {
    console.error('Failed to save macro cache:', error);
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const refresh = searchParams.get('refresh') === 'true';
  
  if (!refresh) {
    const cached = await getCachedMacro();
    if (cached) {
      return NextResponse.json(cached);
    }
  }
  
  const data = await fetchAllMacroData();
  
  await saveMacroCache(data);
  
  return NextResponse.json(data);
}
