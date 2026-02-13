"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { Calendar, Download, ChevronRight, CheckCircle, XCircle, Clock, HelpCircle, ArrowLeft, BarChart3, LayoutGrid, Activity, ArrowRight, FlaskConical, Filter, RefreshCw, Database, FileText, GitCompare, ArrowUpDown } from "lucide-react";
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
    DONE: { icon: CheckCircle, color: "text-positive bg-positive-tint border border-positive/30", label: "완료" },
    SKIPPED: { icon: XCircle, color: "text-muted bg-surface border border-border", label: "스킵" },
    DELAY: { icon: Clock, color: "text-choppy bg-choppy-tint border border-choppy/30", label: "지연" },
    UNKNOWN: { icon: HelpCircle, color: "text-muted bg-inset border border-border", label: "미확인" },
  };
  
  const { icon: Icon, color, label } = config[status];
  
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold tracking-wider ${color}`}>
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
    <div className="rounded-xl border border-border bg-surface p-5 grid grid-cols-2 md:grid-cols-4 gap-6">
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
        <div className="text-2xl font-bold font-mono text-choppy">{stats.skipped}</div>
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
      <div className="rounded-xl border border-border bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
        <BarChart3 size={24} className="opacity-50" />
        <span className="text-sm">기록이 쌓이면 품질 분석이 표시됩니다.</span>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Accuracy Score */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="text-xs text-muted mb-1">체결 정확도</div>
        <div className="flex items-baseline gap-2">
           <span className="text-2xl font-bold font-mono text-info">{stats.accuracy}%</span>
        </div>
        <div className="w-full bg-surface h-1 rounded-full overflow-hidden mt-2">
           <div className="bg-info h-full rounded-full" style={{ width: `${stats.accuracy}%` }}></div>
        </div>
      </div>

      {/* Slippage */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="text-xs text-muted mb-1">평균 슬리피지</div>
        <div className="flex items-baseline gap-2">
           <span className="text-2xl font-bold font-mono text-fg">{stats.slippage.toFixed(2)}%</span>
           <span className="text-xs text-muted">오차</span>
        </div>
        <div className="w-full bg-surface h-1 rounded-full overflow-hidden mt-2">
           <div className="bg-positive h-full rounded-full" style={{ width: `${Math.min(stats.slippage * 200, 100)}%` }}></div>
        </div>
      </div>

      {/* Delay */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="text-xs text-muted mb-1">지연 기록</div>
        <div className="flex items-baseline gap-2">
           <span className="text-2xl font-bold font-mono text-fg">{stats.delayed}</span>
           <span className="text-xs text-muted">건</span>
        </div>
        <div className={`text-xs mt-2 ${stats.delayed === 0 ? "text-positive" : "text-choppy"}`}>
           {stats.delayed === 0 ? "지연 없음 ✓" : "정시 기록 권장"}
        </div>
      </div>
    </div>
  );
}

// --- Tab Type ---
type RecordsTab = "timeline" | "detail" | "compare" | "export";

// --- Main Page Component ---
export default function RecordsPage() {
  const dataSource = useDataSource();
  const [records, setRecords] = useState<RecordEntry[]>([]);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<RecordsTab>("timeline");
  
  // Filter state
  type PeriodFilter = "30d" | "3mo" | "all";
  type StatusFilter = "all" | "DONE" | "SKIPPED" | "DELAY" | "UNKNOWN";
  const [periodFilter, setPeriodFilter] = useState<PeriodFilter>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [exportPeriod, setExportPeriod] = useState<PeriodFilter>("all");
  
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

  // Tab navigation
  const handleSelectRecord = (date: string) => {
    setSelectedDate(date);
    setTab("detail");
  };

  const handleBackToTimeline = () => {
    setSelectedDate(null);
    setTab("timeline");
  };

  // Tab bar config
  const tabs: { id: RecordsTab; label: string; icon: typeof Clock }[] = [
    { id: "timeline", label: "타임라인", icon: Clock },
    { id: "detail", label: "상세", icon: FileText },
    { id: "compare", label: "비교", icon: GitCompare },
    { id: "export", label: "내보내기", icon: Download },
  ];

  return (
    <div className="space-y-6 pb-20">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          기록
          <span className="text-sm font-normal text-muted bg-surface px-2.5 py-0.5 rounded-full border border-border">Records</span>
          {dataSource === "MOCK" ? (
            <span className="text-[11px] font-bold text-choppy bg-choppy-tint px-2 py-0.5 rounded border border-choppy/30">
              MOCK
            </span>
          ) : (
            <span className="text-[11px] font-bold text-positive bg-positive-tint px-2 py-0.5 rounded border border-positive/30">
              REAL
            </span>
          )}
        </h1>
      </div>

      {/* Tab Bar */}
      <div className="flex gap-2 border-b border-border pb-2">
        {tabs.map((t) => {
          const Icon = t.icon;
          const isActive = tab === t.id;
          const isDetailDisabled = t.id === "detail" && !selectedDate;
          return (
            <button
              key={t.id}
              onClick={() => {
                if (isDetailDisabled) return;
                setTab(t.id);
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                isActive
                  ? "bg-inset text-fg"
                  : isDetailDisabled
                  ? "text-muted/40 cursor-not-allowed"
                  : "text-muted hover:bg-surface hover:text-fg"
              }`}
            >
              <Icon size={14} />
              {t.label}
              {t.id === "detail" && selectedDate && (
                <span className="text-[10px] font-mono text-muted bg-inset px-1 rounded">
                  {selectedDate.slice(5)}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* ============ TAB A: Timeline ============ */}
      {tab === "timeline" && (
        <div className="space-y-6">
          {/* Summary Strip */}
          <RecordsSummaryStrip records={records} />

          {records.length === 0 ? (
            <div className="rounded-xl border border-border bg-surface p-8 flex flex-col items-center justify-center text-center gap-4">
              <Database size={32} className="text-muted" />
              <div>
                <div className="text-fg font-medium mb-1">아직 기록이 없습니다</div>
                <div className="text-sm text-muted">Command 페이지에서 실행 완료 후 기록하세요</div>
              </div>
              <div className="flex flex-col sm:flex-row gap-2 mt-2">
                <a 
                  href="/command"
                  className="flex items-center gap-2 px-4 py-2 bg-info hover:bg-info/80 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  <ArrowRight size={16} />
                  오늘 기록하러 가기
                </a>
                <button
                  onClick={loadRecords}
                  className="flex items-center gap-2 px-4 py-2 bg-surface hover:bg-surface text-fg rounded-lg text-sm font-medium transition-colors border border-border"
                >
                  <RefreshCw size={16} />
                  새로고침
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* Filter Controls */}
              <div className="flex flex-wrap items-center gap-2 p-3 bg-inset/50 rounded-lg border border-border">
                <Filter size={14} className="text-muted" />
                
                {/* Period Filter */}
                <div className="flex gap-1 bg-inset p-0.5 rounded-md">
                  {(["30d", "3mo", "all"] as const).map((p) => (
                    <button
                      key={p}
                      onClick={() => setPeriodFilter(p)}
                      className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                        periodFilter === p
                          ? "bg-surface text-fg shadow-sm"
                          : "text-muted hover:text-fg"
                      }`}
                    >
                      {p === "30d" ? "30일" : p === "3mo" ? "3개월" : "전체"}
                    </button>
                  ))}
                </div>
                
                <div className="w-px h-4 bg-border" />
                
                {/* Status Filter */}
                <div className="flex gap-1 bg-inset p-0.5 rounded-md">
                  {(["all", "DONE", "SKIPPED", "DELAY", "UNKNOWN"] as const).map((s) => (
                    <button
                      key={s}
                      onClick={() => setStatusFilter(s)}
                      className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                        statusFilter === s
                          ? "bg-surface text-fg shadow-sm"
                          : "text-muted hover:text-fg"
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

              {/* Records Table */}
              <div className="rounded-xl border border-border bg-surface overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs text-muted uppercase bg-inset border-b border-border">
                      <tr>
                        <th className="px-4 py-3 font-medium">날짜</th>
                        <th className="px-4 py-3 font-medium">판정</th>
                        <th className="px-4 py-3 font-medium text-right">TQQQ</th>
                        <th className="px-4 py-3 font-medium text-right">SGOV</th>
                        <th className="px-4 py-3 font-medium text-right">상태</th>
                        <th className="px-4 py-3 font-medium"></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {filteredRecords.map((entry) => {
                        const tqqq = getLineData(entry.record.lines, 'TQQQ');
                        const sgov = getLineData(entry.record.lines, 'SGOV');
                        return (
                          <tr 
                            key={entry.date} 
                            onClick={() => handleSelectRecord(entry.date)}
                            className="hover:bg-inset/50 transition-colors cursor-pointer group"
                          >
                            <td className="px-4 py-3 font-bold text-fg font-mono">{entry.date}</td>
                            <td className="px-4 py-3">
                              <span className={`text-xs font-bold font-mono ${entry.record.snapshot_verdict === 'ON' ? 'text-positive' : 'text-muted'}`}>
                                {entry.record.snapshot_verdict}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-right font-mono">{tqqq.qty.toLocaleString()}</td>
                            <td className="px-4 py-3 text-right font-mono">{sgov.qty.toLocaleString()}</td>
                            <td className="px-4 py-3 text-right">
                              <StatusBadge status={entry.status} />
                            </td>
                            <td className="px-4 py-3 text-right">
                              <ChevronRight size={16} className="text-muted group-hover:text-fg transition-colors" />
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
      )}

      {/* ============ TAB B: Run Detail ============ */}
      {tab === "detail" && (
        <div className="space-y-6">
          {selectedRecord ? (
            <>
              {/* Detail Header */}
              <div className="flex items-center gap-4">
                <button 
                  onClick={handleBackToTimeline}
                  className="w-8 h-8 flex items-center justify-center rounded-lg bg-inset border border-border text-muted hover:text-fg hover:border-border transition-colors"
                >
                  <ArrowLeft size={18} />
                </button>
                <h2 className="text-xl font-bold text-fg font-mono">{selectedRecord.date}</h2>
                <StatusBadge status={selectedRecord.status} />
              </div>

              {/* 1. Decision Snapshot */}
              <div>
                <h3 className="text-base font-bold mb-3 flex items-center gap-2">
                  <Activity size={16} className="text-muted" />
                  판정 스냅샷
                  <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Decision Snapshot</span>
                </h3>
                <div className="rounded-xl border border-border bg-surface p-5">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <div className="text-xs text-muted mb-1">판정일</div>
                      <div className="font-mono text-fg text-sm">{selectedRecord.record.verdict_date}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted mb-1">판정</div>
                      <div className={`font-mono font-bold ${selectedRecord.record.snapshot_verdict === 'ON' ? 'text-positive' : 'text-choppy'}`}>
                        {selectedRecord.record.snapshot_verdict}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted mb-1">데이터 상태</div>
                      <div className={`font-mono text-sm ${selectedRecord.record.snapshot_health === 'FRESH' ? 'text-positive' : selectedRecord.record.snapshot_health === 'STALE' ? 'text-negative' : 'text-muted'}`}>
                        {selectedRecord.record.snapshot_health}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted mb-1">기록 시각</div>
                      <div className="font-mono text-fg text-sm">{new Date(selectedRecord.record.created_at).toLocaleString("ko-KR")}</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 2. Planned Orders (Expected) */}
              <div>
                <h3 className="text-base font-bold mb-3 flex items-center gap-2">
                  <LayoutGrid size={16} className="text-muted" />
                  예상 주문
                  <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Planned Orders</span>
                </h3>
                <div className="rounded-xl border border-border bg-surface overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs text-muted uppercase bg-inset border-b border-border">
                      <tr>
                        <th className="px-4 py-3 font-medium">티커</th>
                        <th className="px-4 py-3 font-medium text-right">예상 수량</th>
                        <th className="px-4 py-3 font-medium text-right">기준가</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {selectedRecord.record.expected_lines && selectedRecord.record.expected_lines.length > 0 ? (
                        selectedRecord.record.expected_lines.map((line, idx) => (
                          <tr key={idx}>
                            <td className="px-4 py-3 font-bold text-fg">
                              <span className={line.side === 'BUY' ? 'text-positive' : 'text-negative'}>
                                {line.side}
                              </span>
                              {' '}{line.symbol}
                            </td>
                            <td className="px-4 py-3 text-right font-mono">{line.qty.toLocaleString()}</td>
                            <td className="px-4 py-3 text-right font-mono text-muted">
                              {line.expectedPrice ? `$${line.expectedPrice.toFixed(2)}` : "-"}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={3} className="px-4 py-6 text-center text-muted">예상 주문 데이터 없음</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 3. Executed Trades (Actual) */}
              <div>
                <h3 className="text-base font-bold mb-3 flex items-center gap-2">
                  <CheckCircle size={16} className="text-muted" />
                  실제 체결
                  <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Executed Trades</span>
                </h3>
                <div className="rounded-xl border border-border bg-surface overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs text-muted uppercase bg-inset border-b border-border">
                      <tr>
                        <th className="px-4 py-3 font-medium">티커</th>
                        <th className="px-4 py-3 font-medium text-right">체결 수량</th>
                        <th className="px-4 py-3 font-medium text-right">체결가</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {selectedRecord.record.lines && selectedRecord.record.lines.length > 0 ? (
                        selectedRecord.record.lines.map((line, idx) => (
                          <tr key={idx}>
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

              {/* 4. Diff (Expected vs Actual) */}
              {selectedRecord.record.expected_lines && selectedRecord.record.expected_lines.length > 0 && selectedRecord.record.lines.length > 0 && (
                <div>
                  <h3 className="text-base font-bold mb-3 flex items-center gap-2">
                    <ArrowUpDown size={16} className="text-muted" />
                    예상 vs 실제
                    <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Diff</span>
                  </h3>
                  <div className="rounded-xl border border-border bg-surface overflow-hidden">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-muted uppercase bg-inset border-b border-border">
                        <tr>
                          <th className="px-4 py-3 font-medium">티커</th>
                          <th className="px-4 py-3 font-medium text-right">수량 오차</th>
                          <th className="px-4 py-3 font-medium text-right">가격 오차</th>
                          <th className="px-4 py-3 font-medium text-right">슬리피지</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {selectedRecord.record.expected_lines.map((exp, idx) => {
                          const act = selectedRecord.record.lines.find(l => l.symbol === exp.symbol);
                          const qtyDiff = act ? act.qty - exp.qty : 0;
                          const priceDiff = act?.price && exp.expectedPrice ? act.price - exp.expectedPrice : null;
                          const slippage = priceDiff !== null && exp.expectedPrice ? (Math.abs(priceDiff) / exp.expectedPrice * 100) : null;
                          
                          return (
                            <tr key={idx}>
                              <td className="px-4 py-3 font-bold text-fg">{exp.symbol}</td>
                              <td className="px-4 py-3 text-right font-mono">
                                <span className={qtyDiff === 0 ? "text-positive" : "text-negative"}>
                                  {qtyDiff === 0 ? "일치" : `${qtyDiff > 0 ? "+" : ""}${qtyDiff}`}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-right font-mono">
                                {priceDiff !== null ? (
                                  <span className={Math.abs(priceDiff) < 0.01 ? "text-positive" : "text-choppy"}>
                                    {priceDiff > 0 ? "+" : ""}{priceDiff.toFixed(2)}
                                  </span>
                                ) : (
                                  <span className="text-muted">-</span>
                                )}
                              </td>
                              <td className="px-4 py-3 text-right font-mono">
                                {slippage !== null ? (
                                  <span className={slippage < 0.1 ? "text-positive" : slippage < 0.5 ? "text-choppy" : "text-negative"}>
                                    {slippage.toFixed(3)}%
                                  </span>
                                ) : (
                                  <span className="text-muted">-</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 5. Notes & Evidence */}
              {selectedRecord.record.note && (
                <div>
                  <h3 className="text-base font-bold mb-3 flex items-center gap-2">
                    <FileText size={16} className="text-muted" />
                    메모
                    <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Notes</span>
                  </h3>
                  <div className="rounded-xl border border-border bg-surface p-5">
                    <div className="text-fg whitespace-pre-wrap">{selectedRecord.record.note}</div>
                  </div>
                </div>
              )}
            </>
          ) : (
            /* No record selected placeholder */
            <div className="rounded-xl border border-border bg-surface p-12 flex flex-col items-center justify-center text-center gap-3">
              <FileText size={32} className="text-muted opacity-50" />
              <div className="text-fg font-medium">기록을 선택하세요</div>
              <div className="text-sm text-muted">타임라인 탭에서 날짜를 클릭하면 상세 내용이 표시됩니다.</div>
              <button
                onClick={() => setTab("timeline")}
                className="mt-2 flex items-center gap-2 px-4 py-2 bg-info hover:bg-info/80 text-white rounded-lg text-sm font-medium transition-colors"
              >
                <Clock size={16} />
                타임라인으로 이동
              </button>
            </div>
          )}
        </div>
      )}

      {/* ============ TAB C: Compare ============ */}
      {tab === "compare" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold flex items-center gap-2">
              <Activity size={18} className="text-muted" />
              운영 품질
              <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Quality</span>
            </h2>
          </div>

          <QualityAnalytics records={records} />

          {/* Comparison Detail Table */}
          {records.length > 0 && (
            <div>
              <h3 className="text-base font-bold mb-3 flex items-center gap-2">
                <GitCompare size={16} className="text-muted" />
                예상 vs 실제 집계
                <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Aggregate Diff</span>
              </h3>
              <div className="rounded-xl border border-border bg-surface overflow-hidden">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-muted uppercase bg-inset border-b border-border">
                    <tr>
                      <th className="px-4 py-3 font-medium">날짜</th>
                      <th className="px-4 py-3 font-medium">판정</th>
                      <th className="px-4 py-3 font-medium text-right">예상 종목수</th>
                      <th className="px-4 py-3 font-medium text-right">체결 종목수</th>
                      <th className="px-4 py-3 font-medium text-right">일치</th>
                      <th className="px-4 py-3 font-medium text-right">상태</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {records.filter(r => r.status === "DONE").map((entry) => {
                      const expectedCount = entry.record.expected_lines?.length || 0;
                      const actualCount = entry.record.lines?.length || 0;
                      let matches = 0;
                      entry.record.expected_lines?.forEach(exp => {
                        const act = entry.record.lines?.find(l => l.symbol === exp.symbol);
                        if (act && act.qty === exp.qty) matches++;
                      });
                      const isMatch = expectedCount > 0 && matches === expectedCount;
                      
                      return (
                        <tr 
                          key={entry.date} 
                          onClick={() => handleSelectRecord(entry.date)}
                          className="hover:bg-inset/50 transition-colors cursor-pointer"
                        >
                          <td className="px-4 py-3 font-mono text-fg">{entry.date}</td>
                          <td className="px-4 py-3">
                            <span className={`text-xs font-bold font-mono ${entry.record.snapshot_verdict === 'ON' ? 'text-positive' : 'text-muted'}`}>
                              {entry.record.snapshot_verdict}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right font-mono">{expectedCount}</td>
                          <td className="px-4 py-3 text-right font-mono">{actualCount}</td>
                          <td className="px-4 py-3 text-right font-mono">
                            <span className={isMatch ? "text-positive" : "text-choppy"}>
                              {expectedCount > 0 ? `${matches}/${expectedCount}` : "-"}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <StatusBadge status={entry.status} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Skipped & Delay Summary */}
          {records.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-xl border border-border bg-surface p-5">
                <div className="text-xs text-muted mb-2">누락 (SKIPPED)</div>
                <div className="text-2xl font-bold font-mono text-choppy">
                  {records.filter(r => r.status === "SKIPPED").length}
                </div>
                <div className="text-xs text-muted mt-1">전체 {records.length}건 중</div>
              </div>
              <div className="rounded-xl border border-border bg-surface p-5">
                <div className="text-xs text-muted mb-2">지연 (DELAY)</div>
                <div className="text-2xl font-bold font-mono text-choppy">
                  {records.filter(r => r.status === "DELAY").length}
                </div>
                <div className="text-xs text-muted mt-1">전체 {records.length}건 중</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============ TAB D: Export ============ */}
      {tab === "export" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold flex items-center gap-2">
              <Download size={18} className="text-muted" />
              내보내기
              <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Export</span>
            </h2>
          </div>

          {/* Export Options Card */}
          <div className="rounded-xl border border-border bg-surface p-6 space-y-5">
            {/* Period Selection */}
            <div>
              <div className="text-sm font-medium text-fg mb-2">기간 선택</div>
              <div className="flex gap-2">
                {(["30d", "3mo", "all"] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setExportPeriod(p)}
                    className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors border ${
                      exportPeriod === p
                        ? "bg-inset text-fg border-border"
                        : "text-muted border-transparent hover:bg-inset hover:text-fg"
                    }`}
                  >
                    {p === "30d" ? "최근 30일" : p === "3mo" ? "최근 3개월" : "전체 기간"}
                  </button>
                ))}
              </div>
            </div>

            {/* Export Info */}
            <div className="border-t border-border pt-4">
              <div className="text-sm text-muted mb-3">
                내보내기 대상: <span className="font-mono text-fg font-bold">
                  {(() => {
                    if (exportPeriod === "all") return records.length;
                    const now = new Date();
                    const cutoff = new Date();
                    if (exportPeriod === "30d") cutoff.setDate(now.getDate() - 30);
                    if (exportPeriod === "3mo") cutoff.setMonth(now.getMonth() - 3);
                    const cutoffStr = cutoff.toISOString().split("T")[0];
                    return records.filter(r => r.date >= cutoffStr).length;
                  })()}건
                </span>
              </div>
            </div>

            {/* Download Button */}
            <div className="flex gap-3">
              {records.length > 0 ? (
                <button 
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-5 py-2.5 bg-info hover:bg-info/80 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  <Download size={16} />
                  CSV 다운로드
                </button>
              ) : (
                <button 
                  disabled
                  className="flex items-center gap-2 px-5 py-2.5 bg-inset text-muted rounded-lg text-sm font-medium border border-border cursor-not-allowed"
                >
                  <Download size={16} />
                  CSV 다운로드
                </button>
              )}
            </div>
          </div>

          {/* Future: Import placeholder */}
          <div className="rounded-xl border border-dashed border-border bg-inset/50 p-8 flex flex-col items-center justify-center text-muted gap-2">
            <Database size={24} className="opacity-50" />
            <span className="text-sm">CSV 가져오기 (추후 지원)</span>
          </div>
        </div>
      )}
    </div>
  );
}
