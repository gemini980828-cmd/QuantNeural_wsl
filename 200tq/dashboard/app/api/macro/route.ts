import { NextResponse } from 'next/server';

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

export async function GET() {
  const [vix, fng, treasury, dxy, nq, usdkrw] = await Promise.all([
    fetchVix(),
    fetchFearGreed(),
    fetchTreasury10Y(),
    fetchDXY(),
    fetchNasdaqFutures(),
    fetchUsdKrw(),
  ]);

  const data: MacroData = {
    vix,
    fng,
    treasury,
    dxy,
    nq,
    usdkrw,
    updatedAt: new Date().toISOString(),
  };

  return NextResponse.json(data);
}
