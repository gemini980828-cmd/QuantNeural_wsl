/**
 * Phase 5C: Trade Execution Records API
 * 
 * GET /api/record?date=YYYY-MM-DD - Fetch execution record for date
 * POST /api/record - Upsert execution record with snapshot
 */

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import type { TradeLine } from "@/lib/types";

// Force dynamic behavior - no caching for records (always need fresh data)
export const dynamic = "force-dynamic";

// Server-side Supabase client (uses service role for write operations)
function getSupabaseClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) {
    throw new Error("Supabase not configured");
  }
  
  return createClient(url, key);
}

/**
 * GET /api/record?date=2026-01-24
 * Returns execution record for the specified date, or empty indicator
 */
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const date = searchParams.get("date");
    
    if (!date) {
      return NextResponse.json({ error: "date parameter required" }, { status: 400 });
    }
    
    const supabase = getSupabaseClient();
    
    const { data, error } = await supabase
      .from("trade_executions")
      .select("*")
      .eq("execution_date", date)
      .maybeSingle();
    
    if (error) {
      throw error;
    }
    
    if (!data) {
      return NextResponse.json({ empty: true, execution_date: date });
    }
    
    return NextResponse.json(data);
    
  } catch (error) {
    console.error("GET /api/record error:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

/**
 * POST /api/record
 * Body: { executionDate, executed, lines, note }
 * Fetches current snapshot and stores with record
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { executionDate, executed, lines, note } = body as {
      executionDate: string;
      executed: boolean;
      lines: TradeLine[];
      note?: string;
    };
    
    if (!executionDate) {
      return NextResponse.json({ error: "executionDate required" }, { status: 400 });
    }
    
    const supabase = getSupabaseClient();
    
    // Fetch current ops snapshot for audit trail
    const { data: snapshot } = await supabase
      .from("ops_snapshots_daily")
      .select("*")
      .order("verdict_date", { ascending: false })
      .limit(1)
      .single();
    
    const payload = snapshot?.payload_json || {};
    
    // Build record with snapshot data
    const record = {
      execution_date: executionDate,
      verdict_date: snapshot?.verdict_date || executionDate,
      snapshot_verdict: payload.verdict || "OFF10",
      snapshot_health: snapshot?.health || "STALE",
      snapshot_json: payload,
      executed: executed ?? false,
      lines: lines || [],
      note: note || null,
      updated_at: new Date().toISOString(),
    };
    
    // Upsert (insert or update based on execution_date)
    const { data, error } = await supabase
      .from("trade_executions")
      .upsert(record, { onConflict: "execution_date" })
      .select()
      .single();
    
    if (error) {
      throw error;
    }
    
    return NextResponse.json({ success: true, data });
    
  } catch (error) {
    console.error("POST /api/record error:", error);
    return NextResponse.json({ success: false, error: String(error) }, { status: 500 });
  }
}
