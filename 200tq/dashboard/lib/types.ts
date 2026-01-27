export interface HoldingsJson {
  [ticker: string]: {
    shares: number;
    price: number;
    value: number;
  };
}

export interface AllocJson {
  [ticker: string]: number;
}

export interface HoldingsSnapshotRow {
  id: string;
  created_at: string;
  as_of_date: string; // YYYY-MM-DD
  total_value_usd: number | null;
  holdings_json: HoldingsJson;
  alloc_json: AllocJson | null;
  target_alloc_json: AllocJson | null;
  notes: string | null;
  user_id: string;
}

export interface SignalRow {
  id: string;
  created_at: string;
  as_of_date: string;
  regime: "ON" | "OFF";
  confirmed: boolean;
  confirm_day: number | null;
  reason_json: any;
  target_alloc_json: AllocJson | null;
  user_id: string;
}

export interface TriggerEvaluationRow {
  id: string;
  created_at: string;
  as_of_date: string;
  inputs_json: any;  // { close: number, ma200: number, ... }
  checks_json: any;  // { price_gt_ma: boolean, ... }
  trigger_ok: boolean;
  notes: string | null;
  user_id: string;
}

export interface ExecutionChecklistRow {
  id: string;
  created_at: string;
  as_of_date: string;
  pretrade_json: any;
  pretrade_complete: boolean;
  posttrade_json: any;
  executed_complete: boolean;
  override_used: boolean;
  user_id: string;
}

export interface DecisionLogRow {
  id: string;
  created_at: string;
  as_of_date: string;
  memo: string | null;
  override_reason: string | null;
  user_id: string;
}

// Phase 5C: Trade Execution Records
export interface TradeLine {
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  price?: number;
  expectedPrice?: number;
  note?: string;
}

export interface TradeExecutionRow {
  id?: string;
  execution_date: string;
  verdict_date: string;
  snapshot_verdict: 'ON' | 'OFF10';
  snapshot_health: 'FRESH' | 'STALE' | 'CLOSED';
  snapshot_json?: Record<string, unknown>;
  executed: boolean;
  lines: TradeLine[];
  note?: string;
  created_at?: string;
  updated_at?: string;
}

export interface PortfolioStateRow {
  id: string;
  user_id: string;
  tqqq_shares: number;
  sgov_shares: number;
  last_updated: string;
  source: 'manual' | 'ocr' | 'trade_record';
}

