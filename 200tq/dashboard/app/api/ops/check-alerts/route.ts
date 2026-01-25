/**
 * Alert Check API
 * 
 * POST /api/ops/check-alerts
 * 
 * Checks for alert conditions and creates notifications.
 * Should be called periodically (e.g., by cron or page load with throttling).
 */

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import {
  checkDataStale,
  checkVerdictChanged,
  checkHardTrigger,
  checkRecordMissing,
  getTodayKST,
} from "@/lib/ops/notifications";

export const dynamic = "force-dynamic";

function getSupabaseClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) {
    throw new Error("Supabase not configured");
  }
  
  return createClient(url, key);
}

interface CheckResult {
  checked: boolean;
  alerts: string[];
  timestamp: string;
}

export async function POST(req: NextRequest): Promise<NextResponse<CheckResult>> {
  const alerts: string[] = [];
  
  try {
    const supabase = getSupabaseClient();
    const today = getTodayKST();
    
    // 1. Get latest snapshot
    const { data: snapshots } = await supabase
      .from("ops_snapshots_daily")
      .select("*")
      .order("verdict_date", { ascending: false })
      .limit(2); // Get 2 for comparison
    
    const currentSnapshot = snapshots?.[0];
    const previousSnapshot = snapshots?.[1];
    
    if (!currentSnapshot) {
      // No snapshot - definitely stale
      await checkDataStale(null, "스냅샷 데이터 없음");
      alerts.push("DATA_STALE");
      
      return NextResponse.json({
        checked: true,
        alerts,
        timestamp: new Date().toISOString(),
      });
    }
    
    const payload = currentSnapshot.payload_json || {};
    const currentVerdict = payload.verdict as "ON" | "OFF10";
    const previousVerdict = previousSnapshot?.payload_json?.verdict as "ON" | "OFF10" | null;
    
    // 2. Check DATA_STALE
    const verdictDate = currentSnapshot.verdict_date;
    if (currentSnapshot.health === "STALE") {
      const isStale = await checkDataStale(verdictDate, payload.staleReason);
      if (isStale) alerts.push("DATA_STALE");
    }
    
    // 3. Check VERDICT_CHANGED
    if (previousVerdict && previousVerdict !== currentVerdict) {
      await checkVerdictChanged(
        previousVerdict,
        currentVerdict,
        currentSnapshot.execution_date
      );
      alerts.push("VERDICT_CHANGED");
      alerts.push("EXEC_SCHEDULED");
    }
    
    // 4. Check HARD_TRIGGER (QQQ -7%)
    const prices = payload.prices || {};
    const qqqClose = prices.QQQ;
    const qqqPrevClose = prices.QQQ_PREV;
    
    if (qqqClose && qqqPrevClose && qqqPrevClose > 0) {
      const qqqDailyChange = (qqqClose - qqqPrevClose) / qqqPrevClose;
      const triggered = await checkHardTrigger(qqqDailyChange);
      if (triggered) alerts.push("HARD_TRIGGER_CONFIRMED");
    }
    
    // 5. Check RECORD_MISSING (if today is execution date)
    const executionDate = currentSnapshot.execution_date;
    if (executionDate === today) {
      // Check if there's a record for today
      const { data: records } = await supabase
        .from("execution_logs")
        .select("id")
        .eq("execution_date", today)
        .limit(1);
      
      const hasRecord = !!(records && records.length > 0);
      const missing = await checkRecordMissing(executionDate, hasRecord);
      if (missing) alerts.push("RECORD_MISSING");
    }
    
    return NextResponse.json({
      checked: true,
      alerts,
      timestamp: new Date().toISOString(),
    });
    
  } catch (error) {
    console.error("POST /api/ops/check-alerts error:", error);
    return NextResponse.json({
      checked: false,
      alerts,
      timestamp: new Date().toISOString(),
    }, { status: 500 });
  }
}
