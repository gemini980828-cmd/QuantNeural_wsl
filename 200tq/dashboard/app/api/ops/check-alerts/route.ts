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

function getSupabaseConfig() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) {
    throw new Error("Supabase not configured");
  }
  
  return { url, key };
}

function getSupabaseClient() {
  const { url, key } = getSupabaseConfig();
  return createClient(url, key);
}

interface CheckResult {
  checked: boolean;
  alerts: string[];
  skipped: string[];
  timestamp: string;
}

interface PortfolioState {
  tqqq_shares: number;
  sgov_shares: number;
}

async function getPortfolioState(url: string, key: string): Promise<PortfolioState> {
  const supabase = createClient(url, key);
  const { data } = await supabase
    .from("portfolio_state")
    .select("tqqq_shares, sgov_shares")
    .eq("user_id", "default")
    .maybeSingle();
  
  const row = data as { tqqq_shares?: number; sgov_shares?: number } | null;
  return {
    tqqq_shares: row?.tqqq_shares ?? 0,
    sgov_shares: row?.sgov_shares ?? 0,
  };
}

export async function POST(req: NextRequest): Promise<NextResponse<CheckResult>> {
  const alerts: string[] = [];
  const skipped: string[] = [];
  
  try {
    const supabase = getSupabaseClient();
    const { url, key } = getSupabaseConfig();
    const today = getTodayKST();
    
    const snapshotsResult = await supabase
      .from("ops_snapshots_daily")
      .select("*")
      .order("verdict_date", { ascending: false })
      .limit(2);
    
    const portfolioState = await getPortfolioState(url, key);
    
    const snapshots = snapshotsResult.data;
    const currentSnapshot = snapshots?.[0];
    const previousSnapshot = snapshots?.[1];
    
    if (!currentSnapshot) {
      await checkDataStale(null, "스냅샷 데이터 없음");
      alerts.push("DATA_STALE");
      
      return NextResponse.json({
        checked: true,
        alerts,
        skipped,
        timestamp: new Date().toISOString(),
      });
    }
    
    const payload = currentSnapshot.payload_json || {};
    const currentVerdict = payload.verdict as "ON" | "OFF10";
    const previousVerdict = previousSnapshot?.payload_json?.verdict as "ON" | "OFF10" | null;
    
    const verdictDate = currentSnapshot.verdict_date;
    if (currentSnapshot.health === "STALE") {
      const isStale = await checkDataStale(verdictDate, payload.staleReason);
      if (isStale) alerts.push("DATA_STALE");
    }
    
    if (previousVerdict && previousVerdict !== currentVerdict) {
      const isOnToOff = previousVerdict === "ON" && currentVerdict === "OFF10";
      const isOffToOn = previousVerdict === "OFF10" && currentVerdict === "ON";
      
      if (isOnToOff && portfolioState.tqqq_shares > 0) {
        await checkVerdictChanged(
          previousVerdict,
          currentVerdict,
          currentSnapshot.execution_date
        );
        alerts.push("VERDICT_CHANGED");
        alerts.push("EXEC_SCHEDULED");
      } else if (isOffToOn && portfolioState.sgov_shares > 0) {
        await checkVerdictChanged(
          previousVerdict,
          currentVerdict,
          currentSnapshot.execution_date
        );
        alerts.push("VERDICT_CHANGED");
        alerts.push("EXEC_SCHEDULED");
      } else {
        const reason = isOnToOff 
          ? `SELL 신호이나 TQQQ 미보유 (${portfolioState.tqqq_shares}주)`
          : `BUY 신호이나 SGOV 미보유 (${portfolioState.sgov_shares}주)`;
        skipped.push(`VERDICT_CHANGED: ${reason}`);
      }
    }
    
    const prices = payload.prices || {};
    const qqqClose = prices.QQQ;
    const qqqPrevClose = prices.QQQ_PREV;
    
    if (qqqClose && qqqPrevClose && qqqPrevClose > 0) {
      const qqqDailyChange = (qqqClose - qqqPrevClose) / qqqPrevClose;
      const triggered = await checkHardTrigger(qqqDailyChange);
      if (triggered) alerts.push("HARD_TRIGGER_CONFIRMED");
    }
    
    const executionDate = currentSnapshot.execution_date;
    if (executionDate === today) {
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
      skipped,
      timestamp: new Date().toISOString(),
    });
    
  } catch (error) {
    console.error("POST /api/ops/check-alerts error:", error);
    return NextResponse.json({
      checked: false,
      alerts,
      skipped,
      timestamp: new Date().toISOString(),
    }, { status: 500 });
  }
}
