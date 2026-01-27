/**
 * Ops Job Runs API
 * 
 * GET /api/ops/runs
 * 
 * Fetches cron job execution history from ops_job_runs table.
 * Used in Notifications page to show cron execution history.
 * 
 * Query params:
 * - limit: number of records to fetch (default: 20)
 * - job_type: filter by job type (optional)
 */

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

export const dynamic = "force-dynamic";

interface JobRun {
  id: string;
  job_type: string;
  started_at: string;
  finished_at: string | null;
  status: "running" | "success" | "error";
  result_json: Record<string, unknown> | null;
  error_message: string | null;
  duration_ms: number | null;
}

interface RunsResponse {
  runs: JobRun[];
  total: number;
}

function getSupabaseClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) {
    throw new Error("Supabase not configured");
  }
  
  return createClient(url, key);
}

export async function GET(req: NextRequest): Promise<NextResponse<RunsResponse | { error: string }>> {
  try {
    const supabase = getSupabaseClient();
    const { searchParams } = new URL(req.url);
    
    const limit = parseInt(searchParams.get("limit") || "20", 10);
    const jobType = searchParams.get("job_type");
    
    let query = supabase
      .from("ops_job_runs")
      .select("*", { count: "exact" })
      .order("started_at", { ascending: false })
      .limit(limit);
    
    if (jobType) {
      query = query.eq("job_type", jobType);
    }
    
    const { data, error, count } = await query;
    
    if (error) {
      console.error("GET /api/ops/runs error:", error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    
    const runs: JobRun[] = (data ?? []).map((row) => ({
      id: row.id,
      job_type: row.job_type,
      started_at: row.started_at,
      finished_at: row.finished_at,
      status: row.status,
      result_json: row.result_json,
      error_message: row.error_message,
      duration_ms: row.finished_at
        ? new Date(row.finished_at).getTime() - new Date(row.started_at).getTime()
        : null,
    }));
    
    return NextResponse.json({
      runs,
      total: count ?? runs.length,
    });
  } catch (error) {
    console.error("GET /api/ops/runs error:", error);
    return NextResponse.json(
      { error: String(error) },
      { status: 500 }
    );
  }
}
