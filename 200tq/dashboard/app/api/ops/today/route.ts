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
    
    // Fetch QQQ previous close for daily change calculation
    let qqqPrevClose: number | null = null;
    try {
      const { data: prevData } = await supabase
        .from('prices_daily')
        .select('close')
        .eq('symbol', 'QQQ')
        .lt('date', snapshot.verdict_date)
        .order('date', { ascending: false })
        .limit(1);
      
      if (prevData && prevData.length > 0) {
        qqqPrevClose = prevData[0].close;
      }
    } catch (e) {
      console.warn('Failed to fetch QQQ prev close:', e);
    }
    
    // Convert to KST timezone display
    const toKstString = (dateStr: string) => {
      if (!dateStr) return null;
      return `${dateStr}T09:00:00+09:00`;
    };
    
    // Build prices with prev close
    const prices = { ...(payload.prices || {}) };
    if (qqqPrevClose !== null) {
      prices.QQQ_PREV = qqqPrevClose;
    }
    
    // Fetch meta info (last ingest, unresolved alerts)
    let lastIngestSuccess: string | null = null;
    let lastIngestFail: string | null = null;
    let unresolvedAlerts = 0;
    
    try {
      // Get last successful ingest
      const { data: successRun } = await supabase
        .from('ops_job_runs')
        .select('ended_at')
        .eq('status', 'success')
        .order('ended_at', { ascending: false })
        .limit(1);
      if (successRun && successRun.length > 0) {
        lastIngestSuccess = successRun[0].ended_at;
      }
      
      // Get last failed ingest
      const { data: failRun } = await supabase
        .from('ops_job_runs')
        .select('ended_at')
        .eq('status', 'failed')
        .order('ended_at', { ascending: false })
        .limit(1);
      if (failRun && failRun.length > 0) {
        lastIngestFail = failRun[0].ended_at;
      }
      
      // Get unresolved notification count
      const { count } = await supabase
        .from('ops_notifications')
        .select('*', { count: 'exact', head: true })
        .eq('resolved', false);
      unresolvedAlerts = count || 0;
    } catch (e) {
      console.warn('Failed to fetch meta info:', e);
    }
    
    const response: OpsSnapshot & { meta?: object } = {
      verdictDateKst: toKstString(snapshot.verdict_date) ?? '',
      executionDateKst: toKstString(snapshot.execution_date) ?? '',
      health: snapshot.health as 'FRESH' | 'STALE' | 'CLOSED',
      prices,
      sma: payload.sma || { sma3: 0, sma160: 0, sma165: 0, sma170: 0 },
      verdict: payload.verdict || 'OFF10',
      updatedAtKst: snapshot.computed_at,
      sourceMeta: payload.sourceMeta,
      staleReason: payload.staleReason,
      meta: {
        lastIngestSuccess,
        lastIngestFail,
        unresolvedAlerts,
      },
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

