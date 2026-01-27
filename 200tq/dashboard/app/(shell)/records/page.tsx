"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { Calendar, Download, ChevronRight, CheckCircle, XCircle, Clock, HelpCircle, ArrowLeft, BarChart3, LayoutGrid, Activity, ArrowRight, FlaskConical, Filter, RefreshCw, Database } from "lucide-react";
import { useDataSource } from "../../../lib/stores/settings-store";

// Supabase record type (from /api/records/list)
interface TradeExecutionRecord {
  id: string;
  execution_date: string;
  verdict_date: string;
  snapshot_verdict: 'ON' | 'OFF10';
  snapshot_health: 'FRESH' | 'STALE' | 'CLOSED';
  executed: boolean;
  lines: { symbol: string; side: string; qty: number; price?: number }[];
  expected_lines?: { symbol: string; side: string; qty: number; expectedPrice?: number }[];
  note?: string;
  created_at: string;
  updated_at: string;
}

interface RecordEntry {
  date: string;
  record: TradeExecutionRecord;
  status: "DONE" | "SKIPPED" | "DELAY" | "UNKNOWN";
}

// Fetch records from Supabase API
async function fetchRecords(): Promise<RecordEntry[]> {
  try {
    const res = await fetch("/api/records/list?limit=100");
    const data = await res.json();
    
    if (!data.success || !data.records) {
      return [];
    }
    
    return data.records.map((rec: TradeExecutionRecord) => {
      const hasLines = rec.lines && rec.lines.length > 0;
      const status: RecordEntry["status"] = rec.executed && hasLines ? "DONE" : "UNKNOWN";
      
      return {
        date: rec.execution_date,
        record: rec,
        status,
      };
    });
  } catch (error) {
    console.error("Failed to fetch records:", error);
    return [];
  }
}

// Mock records for MOCK mode demo
const MOCK_RECORDS: RecordEntry[] = [
  {
    date: "2026-01-24",
    status: "DONE",
    record: {
      id: "mock-1",
      execution_date: "2026-01-24",
      verdict_date: "2026-01-23",
      snapshot_verdict: "ON",
      snapshot_health: "FRESH",
      executed: true,
      lines: [
        { symbol: "TQQQ", side: "BUY", qty: 150, price: 54.38 },
        { symbol: "SGOV", side: "SELL", qty: 50, price: 100.25 },
      ],
      expected_lines: [
        { symbol: "TQQQ", side: "BUY", qty: 150, expectedPrice: 54.50 },
        { symbol: "SGOV", side: "SELL", qty: 50, expectedPrice: 100.20 },
      ],
      created_at: "2026-01-24T09:30:00Z",
      updated_at: "2026-01-24T09:30:00Z",
    },
  },
  {
    date: "2026-01-23",
    status: "DONE",
    record: {
      id: "mock-2",
      execution_date: "2026-01-23",
      verdict_date: "2026-01-22",
      snapshot_verdict: "OFF10",
      snapshot_health: "FRESH",
      executed: true,
      lines: [
        { symbol: "TQQQ", side: "SELL", qty: 100, price: 52.80 },
        { symbol: "SGOV", side: "BUY", qty: 200, price: 100.10 },
      ],
      expected_lines: [
        { symbol: "TQQQ", side: "SELL", qty: 100, expectedPrice: 53.00 },
        { symbol: "SGOV", side: "BUY", qty: 200, expectedPrice: 100.00 },
      ],
      created_at: "2026-01-23T09:30:00Z",
      updated_at: "2026-01-23T09:30:00Z",
    },
  },
  {
    date: "2026-01-22",
    status: "SKIPPED",
    record: {
      id: "mock-3",
      execution_date: "2026-01-22",
      verdict_date: "2026-01-21",
      snapshot_verdict: "OFF10",
      snapshot_health: "STALE",
      executed: false,
      lines: [],
      note: "시장 휴장",
      created_at: "2026-01-22T09:30:00Z",
      updated_at: "2026-01-22T09:30:00Z",
    },
  },
  {
    date: "2026-01-21",
    status: "DELAY",
    record: {
      id: "mock-4",
      execution_date: "2026-01-21",
      verdict_date: "2026-01-20",
      snapshot_verdict: "ON",
      snapshot_health: "FRESH",
      executed: true,
      lines: [
        { symbol: "TQQQ", side: "BUY", qty: 80, price: 51.20 },
      ],
      expected_lines: [
        { symbol: "TQQQ", side: "BUY", qty: 80, expectedPrice: 51.00 },
      ],
      note: "지연 체결",
      created_at: "2026-01-21T14:00:00Z",
      updated_at: "2026-01-21T14:00:00Z",
    },
  },
  {
    date: "2026-01-17",
    status: "DONE",
    record: {
      id: "mock-5",
      execution_date: "2026-01-17",
      verdict_date: "2026-01-16",
      snapshot_verdict: "ON",
      snapshot_health: "FRESH",
      executed: true,
      lines: [
        { symbol: "TQQQ", side: "BUY", qty: 200, price: 50.50 },
      ],
      expected_lines: [
        { symbol: "TQQQ", side: "BUY", qty: 200, expectedPrice: 50.45 },
      ],
      created_at: "2026-01-17T09:30:00Z",
      updated_at: "2026-01-17T09:30:00Z",
    },
  },
  {
    date: "2026-01-16",
    status: "UNKNOWN",
    record: {
      id: "mock-6",
      execution_date: "2026-01-16",
      verdict_date: "2026-01-15",
      snapshot_verdict: "OFF10",
      snapshot_health: "CLOSED",
      executed: false,
      lines: [],
      created_at: "2026-01-16T09:30:00Z",
      updated_at: "2026-01-16T09:30:00Z",
    },
  },
];

// Status badge component
function StatusBadge({ status }: { status: RecordEntry["status"] }) {
  const config = {
    DONE: { icon: CheckCircle, color: "text-emerald-400 bg-emerald-950/30 border border-emerald-900/50", label: "완료" },
    SKIPPED: { icon: XCircle, color: "text-neutral-400 bg-neutral-800 border border-neutral-700", label: "스킵" },
    DELAY: { icon: Clock, color: "text-amber-400 bg-amber-950/30 border border-amber-900/50", label: "지연" },
    UNKNOWN: { icon: HelpCircle, color: "text-neutral-500 bg-neutral-900 border border-neutral-800", label: "미확인" },
  };
  
  const { icon: Icon, color, label } = config[status];
  
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wider ${color}`}>
      <Icon size={10} />
      {label}
    </span>
  );
}

// Summary Strip (Portfolio Style - Horizontal, evenly distributed)
function RecordsSummaryStrip({ records }: { records: RecordEntry[] }) {
  const stats = useMemo(() => {
    const done = records.filter(r => r.status === "DONE").length;
    const skipped = records.filter(r => r.status === "SKIPPED").length;
    const unknown = records.filter(r => r.status === "UNKNOWN").length;
    return { total: records.length, done, skipped, unknown };
  }, [records]);

  return (
    <div className="rounded-xl border border-neutral-800 bg-surface p-5 grid grid-cols-2 md:grid-cols-4 gap-6">
      <div>
        <div className="text-xs text-muted mb-1">전체 기록</div>
        <div className="text-2xl font-bold font-mono text-fg">{stats.total}</div>
      </div>
      <div>
        <div className="text-xs text-muted mb-1">완료</div>
        <div className="text-2xl font-bold font-mono text-positive">{stats.done}</div>
      </div>
      <div>
        <div className="text-xs text-muted mb-1">스킵</div>
        <div className="text-2xl font-bold font-mono text-amber-400">{stats.skipped}</div>
      </div>
      <div>
        <div className="text-xs text-muted mb-1">미확인</div>
        <div className="text-2xl font-bold font-mono text-muted">{stats.unknown}</div>
      </div>
    </div>
  );
}

// Quality Analytics Section
function QualityAnalytics({ records }: { records: RecordEntry[] }) {
  const stats = useMemo(() => {
    const executed = records.filter(r => r.status === "DONE");
    const count = executed.length;
    
    let totalExpected = 0;
    let totalMatches = 0;
    let slippageSum = 0;
    let slippageCount = 0;

    executed.forEach(entry => {
      const expected = entry.record.expected_lines;
      const actual = entry.record.lines;
      if (!expected || !actual) return;
      
      expected.forEach(exp => {
        totalExpected++;
        const act = actual.find(l => l.symbol === exp.symbol);
        if (act && act.qty === exp.qty) totalMatches++;
        
        if (act?.price && exp.expectedPrice) {
          slippageSum += Math.abs(act.price - exp.expectedPrice) / exp.expectedPrice;
          slippageCount++;
        }
      });
    });

    const accuracy = totalExpected > 0 ? (totalMatches / totalExpected * 100) : 0;
    const slippage = slippageCount > 0 ? (slippageSum / slippageCount * 100) : 0;
    const delayed = records.filter(r => r.status === "DELAY").length;
    
    return { accuracy, slippage, delayed, count };
  }, [records]);

  // Show placeholder if no records
  if (records.length === 0) {
    return (
      <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
        <BarChart3 size={24} className="opacity-50" />
        <span className="text-sm">기록이 쌓이면 품질 분석이 표시됩니다.</span>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Accuracy Score */}
      <div className="bg-surface border border-neutral-800 rounded-xl p-5">
        <div className="text-xs text-muted mb-1">체결 정확도</div>
        <div className="flex items-baseline gap-2">
           <span className="text-2xl font-bold font-mono text-blue-400">{stats.accuracy}%</span>
        </div>
        <div className="w-full bg-neutral-800 h-1 rounded-full overflow-hidden mt-2">
           <div className="bg-blue-500 h-full rounded-full" style={{ width: `${stats.accuracy}%` }}></div>
        </div>
      </div>

      {/* Slippage */}
      <div className="bg-surface border border-neutral-800 rounded-xl p-5">
        <div className="text-xs text-muted mb-1">평균 슬리피지</div>
        <div className="flex items-baseline gap-2">
           <span className="text-2xl font-bold font-mono text-fg">{stats.slippage.toFixed(2)}%</span>
           <span className="text-xs text-muted">오차</span>
        </div>
        <div className="w-full bg-neutral-800 h-1 rounded-full overflow-hidden mt-2">
           <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.min(stats.slippage * 200, 100)}%` }}></div>
        </div>
      </div>

      {/* Delay */}
      <div className="bg-surface border border-neutral-800 rounded-xl p-5">
        <div className="text-xs text-muted mb-1">지연 기록</div>
        <div className="flex items-baseline gap-2">
           <span className="text-2xl font-bold font-mono text-fg">{stats.delayed}</span>
           <span className="text-xs text-muted">건</span>
        </div>
        <div className={`text-xs mt-2 ${stats.delayed === 0 ? "text-positive" : "text-amber-400"}`}>
           {stats.delayed === 0 ? "지연 없음 ✓" : "정시 기록 권장"}
        </div>
      </div>
    </div>
  );
}

export default function RecordsPage() {
  const dataSource = useDataSource();
  const [records, setRecords] = useState<RecordEntry[]>([]);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Filter state
  type PeriodFilter = "30d" | "3mo" | "all";
  type StatusFilter = "all" | "DONE" | "SKIPPED" | "DELAY" | "UNKNOWN";
  const [periodFilter, setPeriodFilter] = useState<PeriodFilter>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  
  // Fetch records based on dataSource
  const loadRecords = useCallback(async () => {
    setLoading(true);
    if (dataSource === "MOCK") {
      setRecords(MOCK_RECORDS);
    } else {
      const data = await fetchRecords();
      setRecords(data);
    }
    setLoading(false);
  }, [dataSource]);
  
  useEffect(() => {
    loadRecords();
  }, [loadRecords]);

  // Filtered records
  const filteredRecords = useMemo(() => {
    let result = records;
    
    // Period filter
    if (periodFilter !== "all") {
      const now = new Date();
      const cutoff = new Date();
      if (periodFilter === "30d") cutoff.setDate(now.getDate() - 30);
      if (periodFilter === "3mo") cutoff.setMonth(now.getMonth() - 3);
      const cutoffStr = cutoff.toISOString().split("T")[0];
      result = result.filter(r => r.date >= cutoffStr);
    }
    
    // Status filter
    if (statusFilter !== "all") {
      result = result.filter(r => r.status === statusFilter);
    }
    
    return result;
  }, [records, periodFilter, statusFilter]);
  
  const handleDownload = () => {
    if (records.length === 0) return;
    window.open("/api/record", "_blank");
  };

  // Helper to get line data by symbol
  const getLineData = (lines: TradeExecutionRecord["lines"], symbol: string) => {
    const line = lines?.find(l => l.symbol === symbol);
    return { qty: line?.qty || 0, price: line?.price };
  };

  const selectedRecord = records.find(r => r.date === selectedDate);

  // Detail View (Sub-page)
  if (selectedRecord) {
    return (
      <div className="space-y-6 pb-20">
        {/* Detail Header */}
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setSelectedDate(null)}
            className="w-8 h-8 flex items-center justify-center rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-700 transition-colors"
          >
            <ArrowLeft size={18} />
          </button>
          <h1 className="text-xl font-bold text-fg font-mono">{selectedRecord.date}</h1>
          <StatusBadge status={selectedRecord.status} />
        </div>

        {/* Executed Trades */}
        <div>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            체결 내역
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Executed Trades</span>
          </h2>
          <div className="rounded-xl border border-neutral-800 bg-surface overflow-hidden">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-muted uppercase bg-neutral-900 border-b border-neutral-800">
                <tr>
                  <th className="px-4 py-3 font-medium">티커</th>
                  <th className="px-4 py-3 font-medium text-right">수량</th>
                  <th className="px-4 py-3 font-medium text-right">체결가</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800">
                {selectedRecord.record.lines && selectedRecord.record.lines.length > 0 ? (
                  selectedRecord.record.lines.map((line, idx) => (
                    <tr key={idx} className="hover:bg-neutral-800/30 transition-colors">
                      <td className="px-4 py-3 font-bold text-fg">
                        <span className={line.side === 'BUY' ? 'text-positive' : 'text-negative'}>
                          {line.side}
                        </span>
                        {' '}{line.symbol}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">{line.qty.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted">
                        {line.price ? `$${line.price.toFixed(2)}` : "-"}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} className="px-4 py-6 text-center text-muted">거래 내역 없음</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Meta & Notes */}
        <div>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            기록 정보
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Meta</span>
          </h2>
          <div className="rounded-xl border border-neutral-800 bg-surface p-5 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-muted mb-1">판정일</div>
                <div className="font-mono text-fg">{selectedRecord.record.verdict_date}</div>
              </div>
              <div>
                <div className="text-xs text-muted mb-1">판정</div>
                <div className={`font-mono font-bold ${selectedRecord.record.snapshot_verdict === 'ON' ? 'text-positive' : 'text-amber-400'}`}>
                  {selectedRecord.record.snapshot_verdict}
                </div>
              </div>
              <div>
                <div className="text-xs text-muted mb-1">데이터 상태</div>
                <div className={`font-mono ${selectedRecord.record.snapshot_health === 'FRESH' ? 'text-positive' : 'text-amber-400'}`}>
                  {selectedRecord.record.snapshot_health}
                </div>
              </div>
              <div>
                <div className="text-xs text-muted mb-1">기록 시각</div>
                <div className="font-mono text-fg text-sm">{new Date(selectedRecord.record.created_at).toLocaleString("ko-KR")}</div>
              </div>
            </div>
            {selectedRecord.record.note && (
              <div className="pt-2 border-t border-neutral-800">
                <div className="text-xs text-muted mb-1">메모</div>
                <div className="text-fg whitespace-pre-wrap">{selectedRecord.record.note}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // List View (Main)
  return (
    <div className="space-y-8 pb-20">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          기록
          <span className="text-sm font-normal text-muted bg-neutral-800 px-2.5 py-0.5 rounded-full border border-neutral-700">Records</span>
          {dataSource === "MOCK" ? (
            <span className="text-[10px] font-bold text-amber-400 bg-amber-950/30 px-2 py-0.5 rounded border border-amber-900/50">
              MOCK
            </span>
          ) : (
            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-950/30 px-2 py-0.5 rounded border border-emerald-900/50">
              REAL
            </span>
          )}
        </h1>
                {records.length > 0 ? (
          <button 
            onClick={handleDownload}
            className="flex items-center gap-2 px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-xs font-medium transition-colors border border-neutral-700"
          >
            <Download size={14} />
            CSV Export
          </button>
        ) : (
          <div className="relative group">
            <button 
              disabled
              className="flex items-center gap-2 px-3 py-1.5 bg-neutral-900 text-neutral-600 rounded-lg text-xs font-medium border border-neutral-800 cursor-not-allowed"
            >
              <Download size={14} />
              CSV Export
            </button>
            <div className="absolute right-0 top-full mt-1 px-2 py-1 bg-neutral-800 text-neutral-400 text-[10px] rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
              기록이 없어 내보내기 불가
            </div>
          </div>
        )}
      </div>

      {/* 1. Summary */}
      <div>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <LayoutGrid size={18} className="text-neutral-400" />
          요약
          <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Summary</span>
        </h2>
        <RecordsSummaryStrip records={records} />
      </div>

      {/* 2. Quality Check */}
      <div>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Activity size={18} className="text-neutral-400" />
          운영 품질
          <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Quality</span>
        </h2>
        <QualityAnalytics records={records} />
      </div>
      
      {/* 3. Timeline */}
      <div>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Clock size={18} className="text-neutral-400" />
          최근 기록
          <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Timeline</span>
        </h2>
        
                        {records.length === 0 ? (
          <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-center gap-4">
            <Database size={32} className="text-neutral-600" />
            <div>
              <div className="text-fg font-medium mb-1">아직 기록이 없습니다</div>
              <div className="text-sm text-muted">Command 페이지에서 실행 완료 후 기록하세요</div>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 mt-2">
              <a 
                href="/command"
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors"
              >
                <ArrowRight size={16} />
                오늘 기록하러 가기
              </a>
              <button
                onClick={loadRecords}
                className="flex items-center gap-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-sm font-medium transition-colors border border-neutral-700"
              >
                <RefreshCw size={16} />
                새로고침
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Filter Controls */}
            <div className="flex flex-wrap items-center gap-2 mb-4 p-3 bg-neutral-900/50 rounded-lg border border-neutral-800">
              <Filter size={14} className="text-neutral-500" />
              
              {/* Period Filter */}
              <div className="flex gap-1 bg-neutral-900 p-0.5 rounded-md">
                {(["30d", "3mo", "all"] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setPeriodFilter(p)}
                    className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                      periodFilter === p
                        ? "bg-neutral-700 text-white"
                        : "text-neutral-500 hover:text-neutral-300"
                    }`}
                  >
                    {p === "30d" ? "30일" : p === "3mo" ? "3개월" : "전체"}
                  </button>
                ))}
              </div>
              
              <div className="w-px h-4 bg-neutral-700" />
              
              {/* Status Filter */}
              <div className="flex gap-1 bg-neutral-900 p-0.5 rounded-md">
                {(["all", "DONE", "SKIPPED", "DELAY", "UNKNOWN"] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setStatusFilter(s)}
                    className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                      statusFilter === s
                        ? "bg-neutral-700 text-white"
                        : "text-neutral-500 hover:text-neutral-300"
                    }`}
                  >
                    {s === "all" ? "전체" : s === "DONE" ? "완료" : s === "SKIPPED" ? "스킵" : s === "DELAY" ? "지연" : "미확인"}
                  </button>
                ))}
              </div>
              
              {/* Filter Result Count */}
              <span className="text-xs text-muted ml-auto">
                {filteredRecords.length}개 / 전체 {records.length}개
              </span>
            </div>
          <div className="rounded-xl border border-neutral-800 bg-surface overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-muted uppercase bg-neutral-900 border-b border-neutral-800">
                  <tr>
                    <th className="px-4 py-3 font-medium">날짜</th>
                    <th className="px-4 py-3 font-medium text-right">TQQQ</th>
                    <th className="px-4 py-3 font-medium text-right">SGOV</th>
                    <th className="px-4 py-3 font-medium text-right">상태</th>
                    <th className="px-4 py-3 font-medium"></th>
                  </tr>
                </thead>
                  <tbody className="divide-y divide-neutral-800">
                  {filteredRecords.map((entry) => {
                    const tqqq = getLineData(entry.record.lines, 'TQQQ');
                    const sgov = getLineData(entry.record.lines, 'SGOV');
                    return (
                      <tr 
                        key={entry.date} 
                        onClick={() => setSelectedDate(entry.date)}
                        className="hover:bg-neutral-800/30 transition-colors cursor-pointer group"
                      >
                        <td className="px-4 py-3 font-bold text-fg font-mono">{entry.date}</td>
                        <td className="px-4 py-3 text-right font-mono">{tqqq.qty.toLocaleString()}</td>
                        <td className="px-4 py-3 text-right font-mono">{sgov.qty.toLocaleString()}</td>
                        <td className="px-4 py-3 text-right">
                          <StatusBadge status={entry.status} />
                        </td>
                        <td className="px-4 py-3 text-right">
                          <ChevronRight size={16} className="text-neutral-600 group-hover:text-neutral-400" />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            </div>
          </>
        )}
      </div>

      {/* 4. Export (placeholder like Portfolio's 성과 분석) */}
      <div>
        <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Download size={18} className="text-neutral-400" />
          내보내기
          <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Export</span>
        </h2>
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-8 flex flex-col items-center justify-center text-muted gap-2 border-dashed">
          <Download size={24} className="opacity-50" />
          <span className="text-sm">CSV 다운로드</span>
          <button 
            onClick={handleDownload}
            className="mt-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-xs font-medium transition-colors border border-neutral-700"
          >
            다운로드
          </button>
        </div>
      </div>
    </div>
  );
}
