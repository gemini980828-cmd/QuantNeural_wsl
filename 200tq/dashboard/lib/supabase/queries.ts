import { supabase } from "./client";
import type { 
  HoldingsSnapshotRow, 
  SignalRow, 
  TriggerEvaluationRow, 
  ExecutionChecklistRow, 
  DecisionLogRow,
  PortfolioStateRow
} from "@/lib/types";

// ... existing functions ...

export async function getLatestHoldings(): Promise<HoldingsSnapshotRow | null> {
  const { data, error } = await supabase
    .from("holdings_snapshots")
    .select("*")
    .order("as_of_date", { ascending: false })
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) {
    console.error("Error fetching holdings:", error);
    return null;
  }

  return data as HoldingsSnapshotRow;
}

export async function getRecentSignals(limit = 10): Promise<SignalRow[]> {
  const { data, error } = await supabase
    .from("signals")
    .select("*")
    .order("as_of_date", { ascending: false })
    .order("created_at", { ascending: false })
    .limit(limit);

  if (error) {
    console.error("Error fetching signals:", error);
    return [];
  }

  return (data ?? []) as SignalRow[];
}

export async function getRecentNews(limit = 20) {
  const { data, error } = await supabase
    .from("news_items")
    .select("*")
    .order("ts", { ascending: false })
    .limit(limit);

  if (error) {
    console.error("Error fetching news:", error);
    return [];
  }
  return data ?? [];
}

// [NEW] Fetch Today's Ops Data
export async function getTodayOpsData(dateStr: string) {
  // Parallel fetch for single interactions
  const [triggerRes, checklistRes, decisionRes] = await Promise.all([
    supabase.from("trigger_evaluations").select("*").eq("as_of_date", dateStr).maybeSingle(),
    supabase.from("execution_checklists").select("*").eq("as_of_date", dateStr).maybeSingle(),
    supabase.from("decision_logs").select("*").eq("as_of_date", dateStr).maybeSingle(),
  ]);

  return {
    trigger: (triggerRes.data as TriggerEvaluationRow) || null,
    checklist: (checklistRes.data as ExecutionChecklistRow) || null,
    decision: (decisionRes.data as DecisionLogRow) || null,
  };
}
// [NEW] Save Trigger Evaluation
export async function saveTriggerEvaluation(row: Partial<TriggerEvaluationRow>) {
  const { data, error } = await supabase
    .from("trigger_evaluations")
    .upsert(row)
    .select()
    .single();

  if (error) {
    throw error;
  }
  return data as TriggerEvaluationRow;
}

// ============================================
// Equity History for Heatmap
// ============================================

export interface EquityPoint {
  date: string;
  value: number;
}

export async function getEquityHistory(
  fromDate?: string,
  toDate?: string
): Promise<EquityPoint[]> {
  let query = supabase
    .from("holdings_snapshots")
    .select("as_of_date, total_value_usd")
    .order("as_of_date", { ascending: true });

  if (fromDate) {
    query = query.gte("as_of_date", fromDate);
  }
  if (toDate) {
    query = query.lte("as_of_date", toDate);
  }

  const { data, error } = await query;

  if (error) {
    console.error("Error fetching equity history:", error);
    return [];
  }

  return (data ?? [])
    .filter((row) => row.total_value_usd !== null)
    .map((row) => ({
      date: row.as_of_date,
      value: Number(row.total_value_usd),
    }));
}

export async function getHoldingsSnapshotsRange(
  fromDate: string,
  toDate: string
): Promise<HoldingsSnapshotRow[]> {
  const { data, error } = await supabase
    .from("holdings_snapshots")
    .select("*")
    .gte("as_of_date", fromDate)
    .lte("as_of_date", toDate)
    .order("as_of_date", { ascending: true });

  if (error) {
    console.error("Error fetching holdings snapshots range:", error);
    return [];
  }

  return (data ?? []) as HoldingsSnapshotRow[];
}

export async function upsertHoldingsSnapshot(
  snapshot: Partial<HoldingsSnapshotRow>
): Promise<HoldingsSnapshotRow | null> {
  const { data, error } = await supabase
    .from("holdings_snapshots")
    .upsert(snapshot, { 
      onConflict: "user_id,as_of_date",
      ignoreDuplicates: false 
    })
    .select()
    .single();

  if (error) {
    console.error("Error upserting holdings snapshot:", error);
    return null;
  }

  return data as HoldingsSnapshotRow;
}

// ============================================
// Price Data Helpers
// ============================================

export async function getPricesForDate(
  date: string,
  symbols: string[] = ["TQQQ", "SGOV", "SPLG", "QQQ"]
): Promise<Record<string, number>> {
  const { data, error } = await supabase
    .from("prices_daily")
    .select("symbol, close")
    .eq("date", date)
    .in("symbol", symbols);

  if (error) {
    console.error("Error fetching prices:", error);
    return {};
  }

  return (data ?? []).reduce((acc, row) => {
    acc[row.symbol] = Number(row.close);
    return acc;
  }, {} as Record<string, number>);
}

export async function getLatestPrices(
  symbols: string[] = ["TQQQ", "SGOV", "SPLG", "QQQ"]
): Promise<{ prices: Record<string, number>; date: string | null }> {
  const { data: dateData } = await supabase
    .from("prices_daily")
    .select("date")
    .eq("symbol", "QQQ")
    .order("date", { ascending: false })
    .limit(1)
    .single();

  if (!dateData) {
    return { prices: {}, date: null };
  }

  const prices = await getPricesForDate(dateData.date, symbols);
  return { prices, date: dateData.date };
}

// ============================================
// Trade Execution Records
// ============================================

import type { TradeExecutionRow } from "@/lib/types";
export async function getExecutionRecord(executionDate: string): Promise<TradeExecutionRow | null> {
  const { data, error } = await supabase
    .from("trade_executions")
    .select("*")
    .eq("execution_date", executionDate)
    .maybeSingle();

  if (error) {
    console.error("Error fetching execution record:", error);
    return null;
  }

  return data as TradeExecutionRow | null;
}

/**
 * Upsert execution record (insert or update based on execution_date)
 */
export async function upsertExecutionRecord(record: Partial<TradeExecutionRow>): Promise<TradeExecutionRow> {
  const { data, error } = await supabase
    .from("trade_executions")
    .upsert(record, { onConflict: "execution_date" })
    .select()
    .single();

  if (error) {
    throw error;
  }
  return data as TradeExecutionRow;
}

/**
 * Get recent execution records
 */
export async function getRecentExecutions(limit = 10): Promise<TradeExecutionRow[]> {
  const { data, error } = await supabase
    .from("trade_executions")
    .select("*")
    .order("execution_date", { ascending: false })
    .limit(limit);

  if (error) {
    console.error("Error fetching recent executions:", error);
    return [];
  }

  return (data ?? []) as TradeExecutionRow[];
}

export async function getPortfolioState(
  userId: string = "default"
): Promise<PortfolioStateRow | null> {
  const { data, error } = await supabase
    .from("portfolio_state")
    .select("*")
    .eq("user_id", userId)
    .maybeSingle();

  if (error) {
    console.error("Error fetching portfolio state:", error);
    return null;
  }

  return data as PortfolioStateRow | null;
}

export async function upsertPortfolioState(
  state: Partial<PortfolioStateRow> & { user_id?: string }
): Promise<PortfolioStateRow | null> {
  const record = {
    user_id: state.user_id || "default",
    tqqq_shares: state.tqqq_shares ?? 0,
    sgov_shares: state.sgov_shares ?? 0,
    source: state.source || "manual",
    last_updated: new Date().toISOString(),
  };

  const { data, error } = await supabase
    .from("portfolio_state")
    .upsert(record, { onConflict: "user_id" })
    .select()
    .single();

  if (error) {
    console.error("Error upserting portfolio state:", error);
    return null;
  }

  return data as PortfolioStateRow;
}

