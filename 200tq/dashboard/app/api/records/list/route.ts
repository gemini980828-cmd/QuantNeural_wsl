/**
 * Records List API - Fetch trade execution records from Supabase
 * 
 * GET /api/records/list?limit=50
 * Returns list of trade execution records for Records page
 */

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

// Force dynamic behavior - no caching for records (always need fresh data)
export const dynamic = "force-dynamic";

function getSupabaseClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) {
    throw new Error("Supabase not configured");
  }
  
  return createClient(url, key);
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const limit = parseInt(searchParams.get("limit") || "50", 10);
    
    const supabase = getSupabaseClient();
    
    const { data, error } = await supabase
      .from("trade_executions")
      .select("*")
      .order("execution_date", { ascending: false })
      .limit(limit);
    
    if (error) {
      throw error;
    }
    
    return NextResponse.json({ 
      success: true, 
      records: data || [],
      count: data?.length || 0
    });
    
  } catch (error) {
    console.error("GET /api/records/list error:", error);
    return NextResponse.json({ 
      success: false, 
      error: String(error),
      records: [] 
    }, { status: 500 });
  }
}
