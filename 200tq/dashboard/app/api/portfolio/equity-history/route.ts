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

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const from = searchParams.get("from");
  const to = searchParams.get("to");

  try {
    const supabase = getSupabaseClient();

    let query = supabase
      .from("holdings_snapshots")
      .select("as_of_date, total_value_usd")
      .not("total_value_usd", "is", null)
      .order("as_of_date", { ascending: true });

    if (from) {
      query = query.gte("as_of_date", from);
    }
    if (to) {
      query = query.lte("as_of_date", to);
    }

    const { data, error } = await query;

    if (error) {
      return NextResponse.json(
        { success: false, error: error.message },
        { status: 500 }
      );
    }

    const equity = (data ?? []).map((row) => ({
      date: row.as_of_date,
      value: Number(row.total_value_usd),
    }));

    const range = {
      from: equity.length > 0 ? equity[0].date : null,
      to: equity.length > 0 ? equity[equity.length - 1].date : null,
    };

    return NextResponse.json({
      success: true,
      equity,
      count: equity.length,
      range,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: String(error) },
      { status: 500 }
    );
  }
}
