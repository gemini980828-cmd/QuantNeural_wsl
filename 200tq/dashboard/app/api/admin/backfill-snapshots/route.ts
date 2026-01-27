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

interface TradeExecution {
  id: string;
  execution_date: string;
  ticker: string;
  action: string;
  shares: number;
  price_usd: number;
}

interface HoldingsState {
  [ticker: string]: { shares: number; avgPrice: number };
}

export async function POST(req: NextRequest) {
  try {
    const supabase = getSupabaseClient();
    
    const { data: trades, error: tradesError } = await supabase
      .from("trade_executions")
      .select("*")
      .order("execution_date", { ascending: true });
    
    if (tradesError) throw tradesError;
    if (!trades || trades.length === 0) {
      return NextResponse.json({ 
        success: true, 
        message: "No trades to backfill",
        created: 0 
      });
    }

    const tradesByDate = new Map<string, TradeExecution[]>();
    for (const trade of trades) {
      const existing = tradesByDate.get(trade.execution_date) || [];
      existing.push(trade);
      tradesByDate.set(trade.execution_date, existing);
    }

    const holdings: HoldingsState = {};
    const snapshots: Array<{
      as_of_date: string;
      holdings_json: Record<string, { shares: number; price: number; value: number }>;
      total_value_usd: number;
      alloc_json: Record<string, number>;
    }> = [];

    const sortedDates = Array.from(tradesByDate.keys()).sort();
    
    for (const date of sortedDates) {
      const dayTrades = tradesByDate.get(date) || [];
      
      for (const trade of dayTrades) {
        const current = holdings[trade.ticker] || { shares: 0, avgPrice: 0 };
        
        if (trade.action === "BUY") {
          const totalShares = current.shares + trade.shares;
          const totalCost = (current.shares * current.avgPrice) + (trade.shares * trade.price_usd);
          current.avgPrice = totalShares > 0 ? totalCost / totalShares : 0;
          current.shares = totalShares;
        } else if (trade.action === "SELL") {
          current.shares = Math.max(0, current.shares - trade.shares);
          if (current.shares === 0) current.avgPrice = 0;
        }
        
        holdings[trade.ticker] = current;
      }

      const holdingsJson: Record<string, { shares: number; price: number; value: number }> = {};
      let totalValue = 0;
      
      for (const [ticker, pos] of Object.entries(holdings)) {
        if (pos.shares > 0) {
          const lastTrade = dayTrades.find(t => t.ticker === ticker);
          const price = lastTrade?.price_usd || pos.avgPrice;
          const value = pos.shares * price;
          holdingsJson[ticker] = { shares: pos.shares, price, value };
          totalValue += value;
        }
      }

      const allocJson: Record<string, number> = {};
      if (totalValue > 0) {
        for (const [ticker, pos] of Object.entries(holdingsJson)) {
          allocJson[ticker] = pos.value / totalValue;
        }
      }

      snapshots.push({
        as_of_date: date,
        holdings_json: holdingsJson,
        total_value_usd: totalValue,
        alloc_json: allocJson,
      });
    }

    let created = 0;
    for (const snapshot of snapshots) {
      const { error: upsertError } = await supabase
        .from("holdings_snapshots")
        .upsert({
          ...snapshot,
          user_id: "default",
        }, {
          onConflict: "user_id,as_of_date"
        });
      
      if (!upsertError) created++;
    }

    return NextResponse.json({ 
      success: true, 
      message: `Backfilled ${created} snapshots from ${trades.length} trades`,
      created,
      totalTrades: trades.length,
      dateRange: {
        from: sortedDates[0],
        to: sortedDates[sortedDates.length - 1]
      }
    });
    
  } catch (error) {
    console.error("Backfill error:", error);
    return NextResponse.json({ 
      success: false, 
      error: String(error) 
    }, { status: 500 });
  }
}
