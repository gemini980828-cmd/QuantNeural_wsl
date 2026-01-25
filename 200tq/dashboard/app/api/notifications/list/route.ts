/**
 * Notifications List API
 * 
 * GET /api/notifications/list?level=action&resolved=false&limit=20
 * Returns list of notifications with filtering
 */

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

export const dynamic = "force-dynamic";

function getSupabaseClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!url || !key) {
    throw new Error("Supabase not configured");
  }
  
  return createClient(url, key);
}

export interface OpsNotification {
  id: string;
  dedupe_key: string;
  level: "info" | "action" | "emergency";
  event_type: string;
  title: string;
  body: string | null;
  resolved: boolean;
  resolved_at: string | null;
  external_sent: boolean;
  created_at: string;
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const level = searchParams.get("level"); // 'info' | 'action' | 'emergency'
    const resolved = searchParams.get("resolved"); // 'true' | 'false'
    const limit = parseInt(searchParams.get("limit") || "20", 10);
    
    const supabase = getSupabaseClient();
    
    let query = supabase
      .from("ops_notifications")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(limit);
    
    if (level) {
      query = query.eq("level", level);
    }
    
    if (resolved !== null) {
      query = query.eq("resolved", resolved === "true");
    }
    
    const { data, error } = await query;
    
    if (error) {
      throw error;
    }
    
    // Also get unresolved counts by level
    const { data: counts } = await supabase
      .from("ops_notifications")
      .select("level")
      .eq("resolved", false);
    
    const unresolvedCounts = {
      info: 0,
      action: 0,
      emergency: 0,
      total: 0,
    };
    
    if (counts) {
      counts.forEach((n) => {
        if (n.level in unresolvedCounts) {
          unresolvedCounts[n.level as keyof typeof unresolvedCounts]++;
        }
        unresolvedCounts.total++;
      });
    }
    
    return NextResponse.json({
      success: true,
      notifications: data || [],
      count: data?.length || 0,
      unresolvedCounts,
    });
    
  } catch (error) {
    console.error("GET /api/notifications/list error:", error);
    return NextResponse.json({
      success: false,
      error: String(error),
      notifications: [],
    }, { status: 500 });
  }
}
