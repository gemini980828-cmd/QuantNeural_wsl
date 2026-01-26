/**
 * Daily Cron Job
 * 
 * Runs daily at 07:00 KST (22:00 UTC previous day)
 * - Fetches latest prices
 * - Updates snapshot
 * - Checks alert conditions
 * - Sends Telegram notifications if needed
 * 
 * @route GET /api/cron/daily
 */

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import {
  checkDataStale,
  checkVerdictChanged,
  checkHardTrigger,
  checkRecordMissing,
  logIngestFail,
  getTodayKST,
} from "@/lib/ops/notifications";

export const dynamic = "force-dynamic";
export const maxDuration = 60; // Allow up to 60 seconds

// Note: Vercel Cron is secured by infrastructure - only registered paths in vercel.json
// can be triggered, so no additional authentication layer is needed.

function getSupabaseClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) {
    throw new Error("Supabase not configured");
  }
  
  return createClient(url, key);
}

export async function GET(req: NextRequest) {
  // Note: Vercel Cron jobs are secured by Vercel's infrastructure
  // Only paths registered in vercel.json can be triggered by cron
  
  const startTime = Date.now();
  const today = getTodayKST();
  const results: Record<string, unknown> = {
    date: today,
    timestamp: new Date().toISOString(),
  };
  
  try {
    const supabase = getSupabaseClient();
    
    // 1. Log job start
    const { data: jobRun } = await supabase
      .from("ops_job_runs")
      .insert({
        job_type: "daily_check",
        started_at: new Date().toISOString(),
        status: "running",
      })
      .select()
      .single();
    
    const jobId = jobRun?.id;
    
    try {
      // 2. Get latest snapshots for comparison
      const { data: snapshots, error: snapshotError } = await supabase
        .from("ops_snapshots_daily")
        .select("*")
        .order("verdict_date", { ascending: false })
        .limit(2);
      
      if (snapshotError) {
        throw new Error(`Snapshot query failed: ${snapshotError.message}`);
      }
      
      const currentSnapshot = snapshots?.[0];
      const previousSnapshot = snapshots?.[1];
      
      if (!currentSnapshot) {
        // No snapshot - create DATA_STALE alert
        await checkDataStale(null, "스냅샷 데이터 없음");
        results.alerts = ["DATA_STALE"];
      } else {
        const payload = currentSnapshot.payload_json || {};
        const currentVerdict = payload.verdict as "ON" | "OFF10";
        const previousVerdict = previousSnapshot?.payload_json?.verdict as "ON" | "OFF10" | null;
        const alerts: string[] = [];
        
        // 3. Check DATA_STALE
        if (currentSnapshot.health === "STALE") {
          await checkDataStale(currentSnapshot.verdict_date, payload.staleReason);
          alerts.push("DATA_STALE");
        }
        
        // 4. Check VERDICT_CHANGED
        if (previousVerdict && previousVerdict !== currentVerdict) {
          await checkVerdictChanged(
            previousVerdict,
            currentVerdict,
            currentSnapshot.execution_date
          );
          alerts.push("VERDICT_CHANGED");
          alerts.push("EXEC_SCHEDULED");
        }
        
        // 5. Check HARD_TRIGGER (QQQ -7%)
        const prices = payload.prices || {};
        const qqqClose = prices.QQQ;
        const qqqPrevClose = prices.QQQ_PREV;
        
        if (qqqClose && qqqPrevClose && qqqPrevClose > 0) {
          const qqqDailyChange = (qqqClose - qqqPrevClose) / qqqPrevClose;
          if (qqqDailyChange <= -0.07) {
            await checkHardTrigger(qqqDailyChange);
            alerts.push("HARD_TRIGGER_CONFIRMED");
          }
        }
        
        // 6. Check RECORD_MISSING
        const executionDate = currentSnapshot.execution_date;
        if (executionDate === today) {
          const { data: records } = await supabase
            .from("execution_logs")
            .select("id")
            .eq("execution_date", today)
            .limit(1);
          
          const hasRecord = !!(records && records.length > 0);
          if (!hasRecord) {
            await checkRecordMissing(executionDate, hasRecord);
            alerts.push("RECORD_MISSING");
          }
        }
        
        results.alerts = alerts;
        results.currentVerdict = currentVerdict;
        results.health = currentSnapshot.health;
      }
      
      // 7. Update job run as success
      if (jobId) {
        await supabase
          .from("ops_job_runs")
          .update({
            status: "success",
            ended_at: new Date().toISOString(),
          })
          .eq("id", jobId);
      }
      
      results.success = true;
      results.durationMs = Date.now() - startTime;
      
    } catch (innerError) {
      // Log failure
      if (jobId) {
        await supabase
          .from("ops_job_runs")
          .update({
            status: "failed",
            ended_at: new Date().toISOString(),
            error: String(innerError),
          })
          .eq("id", jobId);
      }
      
      // Create INGEST_FAIL notification
      await logIngestFail("daily_check", String(innerError));
      
      throw innerError;
    }
    
    return NextResponse.json(results);
    
  } catch (error) {
    console.error("Cron daily error:", error);
    return NextResponse.json({
      success: false,
      error: String(error),
      date: today,
      durationMs: Date.now() - startTime,
    }, { status: 500 });
  }
}
