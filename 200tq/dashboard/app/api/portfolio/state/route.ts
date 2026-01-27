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

export async function GET(): Promise<NextResponse> {
  try {
    const supabase = getSupabaseClient();
    
    const { data, error } = await supabase
      .from("portfolio_state")
      .select("*")
      .eq("user_id", "default")
      .maybeSingle();
    
    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    
    return NextResponse.json({
      state: data || { tqqq_shares: 0, sgov_shares: 0 },
    });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const supabase = getSupabaseClient();
    const body = await req.json();
    
    const { tqqq_shares, sgov_shares, source = "manual" } = body;
    
    const record = {
      user_id: "default",
      tqqq_shares: tqqq_shares ?? 0,
      sgov_shares: sgov_shares ?? 0,
      source,
      last_updated: new Date().toISOString(),
    };
    
    const { data, error } = await supabase
      .from("portfolio_state")
      .upsert(record, { onConflict: "user_id" })
      .select()
      .single();
    
    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    
    return NextResponse.json({ success: true, state: data });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}
