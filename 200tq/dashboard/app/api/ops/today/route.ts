/**
 * Read API for Today's Ops Snapshot
 * 
 * Returns the latest ops snapshot for the Command page.
 * 
 * @route GET /api/ops/today
 */

import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

interface OpsSnapshot {
  verdictDateKst: string;
  executionDateKst: string;
  health: 'FRESH' | 'STALE' | 'CLOSED';
  prices: Record<string, number>;
  sma: {
    sma3: number;
    sma160: number;
    sma165: number;
    sma170: number;
  };
  verdict: 'ON' | 'OFF10';
  updatedAtKst: string;
  sourceMeta?: {
    source: string;
    lastTradingDate: string;
  };
  staleReason?: string;
}

export async function GET() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    return NextResponse.json({ error: 'Supabase not configured' }, { status: 500 });
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey);
  
  try {
    // Get latest snapshot
    const { data, error } = await supabase
      .from('ops_snapshots_daily')
      .select('*')
      .order('verdict_date', { ascending: false })
      .limit(1);
    
    if (error) {
      throw new Error(`DB query failed: ${error.message}`);
    }
    
    if (!data || data.length === 0) {
      // No snapshot exists - return empty with STALE
      return NextResponse.json({
        verdictDateKst: '',
        executionDateKst: '',
        health: 'STALE',
        prices: {},
        sma: { sma3: 0, sma160: 0, sma165: 0, sma170: 0 },
        verdict: 'OFF10',
        updatedAtKst: new Date().toISOString(),
        staleReason: 'No snapshot data available',
      } satisfies OpsSnapshot);
    }
    
    const snapshot = data[0];
    const payload = snapshot.payload_json;
    
    // Convert to KST timezone display
    const toKstString = (dateStr: string) => {
      if (!dateStr) return null;
      return `${dateStr}T09:00:00+09:00`;
    };
    
    const response: OpsSnapshot = {
      verdictDateKst: toKstString(snapshot.verdict_date) ?? '',
      executionDateKst: toKstString(snapshot.execution_date) ?? '',
      health: snapshot.health as 'FRESH' | 'STALE' | 'CLOSED',
      prices: payload.prices || {},
      sma: payload.sma || { sma3: 0, sma160: 0, sma165: 0, sma170: 0 },
      verdict: payload.verdict || 'OFF10',
      updatedAtKst: snapshot.computed_at,
      sourceMeta: payload.sourceMeta,
      staleReason: payload.staleReason,
    };
    
    return NextResponse.json(response);
    
  } catch (error) {
    console.error('Error fetching ops snapshot:', error);
    return NextResponse.json({ 
      error: String(error),
      health: 'STALE',
    }, { status: 500 });
  }
}
