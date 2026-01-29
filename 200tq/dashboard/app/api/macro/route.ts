import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

type ColorTone = 'ok' | 'action' | 'danger';

interface MacroData {
  vix: { value: number | null; color: ColorTone };
  fng: { value: number | null; label: string; color: ColorTone };
  treasury: { value: number | null };
  dxy: { value: number | null };
  nq: { value: number | null };
  usdkrw: { value: number | null };
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

async function fetchYahooQuote(symbol: string): Promise<number | null> {
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=1d`;
    const res = await fetch(url, {
      cache: 'no-store',
      headers: { 'User-Agent': 'Mozilla/5.0' }
    });
    if (!res.ok) return null;
    
    const data = await res.json();
    return data.chart?.result?.[0]?.meta?.regularMarketPrice || null;
  } catch {
    return null;
  }
}

async function fetchVix(): Promise<{ value: number | null; color: ColorTone }> {
  const value = await fetchYahooQuote('^VIX');
  return {
    value,
    color: value !== null ? getVixColor(value) : 'action',
  };
}

async function fetchFearGreed(): Promise<{ value: number | null; label: string; color: ColorTone }> {
  try {
    const res = await fetch('https://api.alternative.me/fng/', { cache: 'no-store' });
    if (!res.ok) throw new Error(`FNG API error: ${res.status}`);
    
    const json = await res.json();
    const value = parseInt(json.data[0].value);
    const label = json.data[0].value_classification;
    
    return {
      value,
      label,
      color: getFngColor(value),
    };
  } catch (error) {
    console.error('Failed to fetch Fear & Greed:', error);
    return { value: null, label: '', color: 'action' };
  }
}

async function fetchTreasury10Y(): Promise<{ value: number | null }> {
  const value = await fetchYahooQuote('^TNX');
  return { value };
}

async function fetchDXY(): Promise<{ value: number | null }> {
  const value = await fetchYahooQuote('DX-Y.NYB');
  return { value };
}

async function fetchNasdaqFutures(): Promise<{ value: number | null }> {
  const value = await fetchYahooQuote('NQ=F');
  return { value };
}

async function fetchUsdKrw(): Promise<{ value: number | null }> {
  const value = await fetchYahooQuote('USDKRW=X');
  return { value };
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
