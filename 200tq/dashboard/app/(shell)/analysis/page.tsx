"use client";

import { useState, useMemo, useEffect } from "react";
import { TrendingUp, TrendingDown, BarChart3, Activity, Zap, FlaskConical, LineChart, Play, Loader2, Calendar, DollarSign, Layers, LayoutGrid, PieChart, AlertTriangle, ChevronDown, Database, Download, Check, ToggleLeft, ToggleRight } from "lucide-react";
import { useDataSource } from "../../../lib/stores/settings-store";
import { EquityCurveChart } from "@/components/analysis/EquityCurveChart";
import { ReturnsHeatmap } from "@/components/analysis/ReturnsHeatmap";
import { SingleStrategyPanel } from "@/components/analysis/SingleStrategyPanel";

type Period = "1M" | "3M" | "6M" | "YTD" | "ALL";
type Strategy = "200TQ" | "E00" | "E01" | "E02" | "E03" | "E04" | "E05" | "E06" | "E07" | "E08" | "E09" | "E10";

interface BacktestApiResult {
  status: "success" | "error";
  experiment?: string;
  params?: {
    strategy: string;
    startDate: string;
    endDate: string;
    capital: number;
  };
  metrics?: {
    CAGR: number;
    MDD: number;
    Sharpe: number;
    Sortino: number;
    Calmar: number;
    Final: number;
    FinalValue: number;
    TotalTax: number;
    TradesCount: number;
    TradingDays: number;
  };
  equity?: Array<{ date: string; value: number }>;
  elapsed_seconds?: number;
  message?: string;
}

function getMockKPIs(period: Period) {
  const data: Record<Period, { cagr: number; mdd: number; volatility: number; sharpe: number }> = {
    "1M": { cagr: 12.5, mdd: -3.2, volatility: 18.4, sharpe: 1.24 },
    "3M": { cagr: 28.7, mdd: -5.8, volatility: 22.1, sharpe: 1.45 },
    "6M": { cagr: 45.2, mdd: -8.4, volatility: 24.6, sharpe: 1.62 },
    "YTD": { cagr: 38.9, mdd: -12.1, volatility: 26.3, sharpe: 1.38 },
    "ALL": { cagr: 52.4, mdd: -18.7, volatility: 28.9, sharpe: 1.51 },
  };
  return data[period];
}

const STRATEGY_LABELS: Record<Strategy, string> = {
  "200TQ": "200TQ Original (MA200)",
  E00: "E00 Base (CASH)",
  E01: "E01 SGOV",
  E02: "E02 Ensemble",
  E03: "E03 Ensemble+SGOV",
  E04: "E04 Hysteresis 0.25%",
  E05: "E05 Hysteresis 0.50%",
  E06: "E06 Hysteresis 1.00%",
  E07: "E07 MA200Guard CASH",
  E08: "E08 MA200Guard SGOV",
  E09: "E09 BuyConfirm 2d",
  E10: "E10 BuySellConfirm 2d",
};

const STRATEGY_COLORS: Record<Strategy, string> = {
  "200TQ": "#ef4444",
  E00: "#94a3b8",
  E01: "#10b981",
  E02: "#f59e0b",
  E03: "#3b82f6",
  E04: "#a855f7",
  E05: "#ec4899",
  E06: "#84cc16",
  E07: "#06b6d4",
  E08: "#f97316",
  E09: "#6366f1",
  E10: "#14b8a6",
};

async function runBacktest(
  startDate: string,
  endDate: string,
  capital: number,
  strategy: Strategy
): Promise<BacktestApiResult> {
  const res = await fetch("/api/backtest/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strategy, startDate, endDate, capital }),
  });
  return res.json();
}

function PeriodSelector({ selected, onChange }: { selected: Period; onChange: (p: Period) => void }) {
  const periods: Period[] = ["1M", "3M", "6M", "YTD", "ALL"];
  
  return (
    <div className="flex gap-1 bg-inset p-1 rounded-lg">
      {periods.map((p) => (
        <button
          key={p}
          onClick={() => onChange(p)}
          className={`px-3 py-1.5 text-xs font-bold rounded-md transition-colors ${
            selected === p 
                ? "bg-inset text-fg"
              : "text-muted hover:text-fg"
          }`}
        >
          {p}
        </button>
      ))}
    </div>
  );
}

function KPICard({ 
  label, 
  value, 
  unit, 
  trend, 
  description 
}: { 
  label: string; 
  value: string; 
  unit?: string; 
  trend?: "up" | "down" | "neutral";
  description?: string;
}) {
  const trendColor = trend === "up" ? "text-positive" : trend === "down" ? "text-negative" : "text-fg";
  
  return (
    <div className="bg-surface border border-border rounded-xl p-5">
      <div className="text-xs text-muted mb-1">{label}</div>
      <div className={`text-2xl font-bold font-mono ${trendColor} flex items-baseline gap-1`}>
        {value}
        {unit && <span className="text-sm text-muted font-normal">{unit}</span>}
      </div>
      {description && <div className="text-xs text-muted mt-2">{description}</div>}
    </div>
  );
}

function BacktestControls({
  startDate,
  endDate,
  capital,
  strategy,
  isRunning,
  onStartDateChange,
  onEndDateChange,
  onCapitalChange,
  onStrategyChange,
  onRun,
}: {
  startDate: string;
  endDate: string;
  capital: number;
  strategy: Strategy;
  isRunning: boolean;
  onStartDateChange: (v: string) => void;
  onEndDateChange: (v: string) => void;
  onCapitalChange: (v: number) => void;
  onStrategyChange: (v: Strategy) => void;
  onRun: () => void;
}) {
  const strategies = Object.entries(STRATEGY_LABELS).map(([value, label]) => ({
    value: value as Strategy,
    label,
  }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
      <div className="space-y-1.5">
        <label className="text-xs text-muted flex items-center gap-1.5">
          <Calendar size={12} />
          시작일
        </label>
        <input
          type="date"
          value={startDate}
          onChange={(e) => onStartDateChange(e.target.value)}
          className="w-full bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg focus:outline-none focus:ring-2 focus:ring-info/50 focus:border-info"
        />
      </div>

      <div className="space-y-1.5">
        <label className="text-xs text-muted flex items-center gap-1.5">
          <Calendar size={12} />
          종료일
        </label>
        <input
          type="date"
          value={endDate}
          onChange={(e) => onEndDateChange(e.target.value)}
          className="w-full bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg focus:outline-none focus:ring-2 focus:ring-info/50 focus:border-info"
        />
      </div>

      <div className="space-y-1.5">
        <label className="text-xs text-muted flex items-center gap-1.5">
          <DollarSign size={12} />
          초기 자본 (만원)
        </label>
        <input
          type="number"
          value={capital / 10000}
          onChange={(e) => onCapitalChange(Number(e.target.value) * 10000)}
          min={100}
          step={100}
          className="w-full bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg focus:outline-none focus:ring-2 focus:ring-info/50 focus:border-info"
        />
      </div>

      <div className="space-y-1.5">
        <label className="text-xs text-muted flex items-center gap-1.5">
          <Layers size={12} />
          전략
        </label>
        <select
          value={strategy}
          onChange={(e) => onStrategyChange(e.target.value as Strategy)}
          className="w-full bg-inset border border-border rounded-lg px-3 py-2 text-sm text-fg focus:outline-none focus:ring-2 focus:ring-info/50 focus:border-info"
        >
          {strategies.map((s) => (
            <option key={s.value} value={s.value}>{s.label}</option>
          ))}
        </select>
      </div>

      <div className="space-y-1.5">
        <label className="text-xs text-muted invisible">실행</label>
        <button
          onClick={onRun}
          disabled={isRunning}
          className="w-full bg-info hover:bg-info/80 disabled:bg-inset disabled:cursor-not-allowed text-white font-bold rounded-lg px-4 py-2 text-sm flex items-center justify-center gap-2 transition-colors"
        >
          {isRunning ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              실행 중...
            </>
          ) : (
            <>
              <Play size={16} />
              Run Backtest
            </>
          )}
        </button>
      </div>
    </div>
  );
}

function BacktestResults({ results }: { results: BacktestApiResult | null }) {
  if (!results || results.status !== "success" || !results.metrics) return null;

  const { metrics, params, equity } = results;
  const totalReturn = ((metrics.Final - 1) * 100).toFixed(1);

  return (
    <div className="space-y-4 mt-6">
      <div className="flex items-center gap-2 px-4 py-2 bg-choppy-tint border border-choppy/30 rounded-lg text-choppy text-xs">
        <AlertTriangle size={14} />
        <span className="font-bold">오늘 Verdict와 무관</span>
        <span className="text-choppy/80">— 이 결과는 연구/검증용이며, 오늘 매매 결정에 사용하지 마세요</span>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-inset/50 rounded-xl border border-border">
        <div>
          <div className="text-xs text-muted">초기 자본</div>
          <div className="text-lg font-bold font-mono text-fg">{((params?.capital ?? 0) / 10000).toLocaleString()}만원</div>
        </div>
        <div>
          <div className="text-xs text-muted">최종 자산</div>
          <div className="text-lg font-bold font-mono text-positive">{(metrics.FinalValue / 10000).toLocaleString()}만원</div>
        </div>
        <div>
          <div className="text-xs text-muted">총 수익률</div>
          <div className={`text-lg font-bold font-mono ${Number(totalReturn) >= 0 ? "text-positive" : "text-negative"}`}>
            {Number(totalReturn) >= 0 ? "+" : ""}{totalReturn}%
          </div>
        </div>
        <div>
          <div className="text-xs text-muted">총 매매 횟수</div>
          <div className="text-lg font-bold font-mono text-fg">{metrics.TradesCount}회</div>
        </div>
      </div>

      {equity && equity.length > 0 && (
        <div className="rounded-xl border border-border bg-surface p-6">
          <div className="flex items-center gap-2 mb-4">
            <LineChart size={16} className="text-info" />
            <span className="text-sm font-bold text-fg">Equity Curve</span>
            <span className="text-xs text-muted ml-auto">{params?.startDate} ~ {params?.endDate}</span>
          </div>
          <EquityCurveChart
            strategies={[{
              name: results.experiment || "Strategy",
              color: "#3b82f6",
              data: equity,
            }]}
            height={300}
          />
        </div>
      )}

      <div className="rounded-xl border border-border bg-surface overflow-hidden">
        <div className="grid grid-cols-6 gap-4 p-4 border-b border-border bg-inset/50">
          <div className="text-xs text-muted uppercase tracking-wider">전략</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">CAGR</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">MDD</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">Sharpe</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">Sortino</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">Calmar</div>
        </div>
        <div className="divide-y divide-border">
          <div className="grid grid-cols-6 gap-4 p-4 hover:bg-surface/20 transition-colors">
            <div className="text-sm font-bold text-fg">{results.experiment}</div>
            <div className={`text-sm font-mono text-right ${metrics.CAGR >= 0 ? "text-positive" : "text-negative"}`}>
              {metrics.CAGR >= 0 ? "+" : ""}{metrics.CAGR.toFixed(1)}%
            </div>
            <div className="text-sm font-mono text-negative text-right">{metrics.MDD.toFixed(1)}%</div>
            <div className="text-sm font-mono text-fg text-right">{metrics.Sharpe.toFixed(2)}</div>
            <div className="text-sm font-mono text-fg text-right">{metrics.Sortino.toFixed(2)}</div>
            <div className="text-sm font-mono text-fg text-right">{metrics.Calmar.toFixed(2)}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-xs text-muted">
        <div className="bg-inset/30 rounded-lg p-3 border border-border">
          <span className="text-muted">거래일수:</span> {metrics.TradingDays}일
        </div>
        <div className="bg-inset/30 rounded-lg p-3 border border-border">
          <span className="text-muted">세금 (22%):</span> {(metrics.TotalTax / 10000).toLocaleString()}만원
        </div>
        <div className="bg-inset/30 rounded-lg p-3 border border-border">
          <span className="text-muted">실행시간:</span> {results.elapsed_seconds?.toFixed(1)}초
        </div>
      </div>

      <div className="flex gap-2 mt-4">
        <button
          onClick={() => {
            const csvRows = ["date,value"];
            equity?.forEach((p) => csvRows.push(`${p.date},${p.value}`));
            const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `backtest_${results.experiment}_${params?.startDate}_${params?.endDate}.csv`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="flex items-center gap-2 px-3 py-1.5 bg-surface hover:bg-surface text-fg rounded-lg text-xs font-medium transition-colors"
        >
          <Download size={14} />
          Export CSV
        </button>
        <button
          onClick={() => {
            const exportData = { experiment: results.experiment, params, metrics, equity };
            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `backtest_${results.experiment}_${params?.startDate}_${params?.endDate}.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="flex items-center gap-2 px-3 py-1.5 bg-surface hover:bg-surface text-fg rounded-lg text-xs font-medium transition-colors"
        >
          <Download size={14} />
          Export JSON
        </button>
      </div>
    </div>
  );
}

function StrategyCompareSection({
  compareStrategies,
  toggleCompareStrategy,
  compareResults,
  compareRunning,
  compareError,
  btStartDate,
  btEndDate,
  setBtStartDate,
  setBtEndDate,
  handleCompareStrategies,
}: {
  compareStrategies: Strategy[];
  toggleCompareStrategy: (s: Strategy) => void;
  compareResults: Record<Strategy, BacktestApiResult>;
  compareRunning: boolean;
  compareError: string | null;
  btStartDate: string;
  btEndDate: string;
  setBtStartDate: (v: string) => void;
  setBtEndDate: (v: string) => void;
  handleCompareStrategies: () => void;
}) {
  return (
    <div className="space-y-4 mt-6 pt-6 border-t border-border">
      <h3 className="text-sm font-bold text-fg flex items-center gap-2">
        <Layers size={14} className="text-muted" />
        복수 전략 비교
      </h3>
      <div className="flex flex-wrap gap-2">
        {(Object.keys(STRATEGY_LABELS) as Strategy[]).map((s) => (
          <button
            key={s}
            onClick={() => toggleCompareStrategy(s)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors flex items-center gap-1.5 ${
              compareStrategies.includes(s)
                ? "bg-info-tint border-info text-info"
                : "bg-surface border-border text-muted hover:border-border"
            }`}
          >
            {compareStrategies.includes(s) && <Check size={12} />}
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: STRATEGY_COLORS[s] }} />
            {s}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-4">
        <div className="grid grid-cols-2 gap-4 flex-1">
          <input
            type="date"
            value={btStartDate}
            onChange={(e) => setBtStartDate(e.target.value)}
            className="bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg"
          />
          <input
            type="date"
            value={btEndDate}
            onChange={(e) => setBtEndDate(e.target.value)}
            className="bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg"
          />
        </div>
        <button
          onClick={handleCompareStrategies}
          disabled={compareRunning || compareStrategies.length === 0}
          className="px-4 py-2 bg-info hover:bg-info/80 disabled:bg-inset text-white font-bold rounded-lg text-sm flex items-center gap-2"
        >
          {compareRunning ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
          Compare ({compareStrategies.length})
        </button>
      </div>
      {compareError && (
        <div className="text-negative text-xs">{compareError}</div>
      )}
      {Object.keys(compareResults).length > 0 && (
        <>
          <div className="rounded-xl border border-border overflow-hidden">
            <div className="grid grid-cols-7 gap-2 p-3 border-b border-border bg-inset/50 text-xs text-muted uppercase">
              <div>전략</div>
              <div className="text-right">CAGR</div>
              <div className="text-right">MDD</div>
              <div className="text-right">Sharpe</div>
              <div className="text-right">Sortino</div>
              <div className="text-right">Calmar</div>
              <div className="text-right">Final</div>
            </div>
            {Object.entries(compareResults).map(([s, r]) => r.metrics && (
              <div key={s} className="grid grid-cols-7 gap-2 p-3 border-b border-border/50 text-sm">
                <div className="font-bold flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: STRATEGY_COLORS[s as Strategy] }} />
                  {s}
                </div>
                <div className={`text-right font-mono ${r.metrics.CAGR >= 0 ? "text-positive" : "text-negative"}`}>
                  {r.metrics.CAGR >= 0 ? "+" : ""}{r.metrics.CAGR.toFixed(1)}%
                </div>
                <div className="text-right font-mono text-negative">{r.metrics.MDD.toFixed(1)}%</div>
                <div className="text-right font-mono">{r.metrics.Sharpe.toFixed(2)}</div>
                <div className="text-right font-mono">{r.metrics.Sortino.toFixed(2)}</div>
                <div className="text-right font-mono">{r.metrics.Calmar.toFixed(2)}</div>
                <div className="text-right font-mono">{r.metrics.Final.toFixed(2)}x</div>
              </div>
            ))}
          </div>
          <EquityCurveChart
            strategies={Object.entries(compareResults)
              .filter(([, r]) => r.equity && r.equity.length > 0)
              .map(([s, r]) => ({
                name: s,
                color: STRATEGY_COLORS[s as Strategy],
                data: r.equity || [],
              }))}
            height={400}
          />
        </>
      )}
    </div>
  );
}

export default function AnalysisPage() {
  const dataSource = useDataSource();
  const [period, setPeriod] = useState<Period>("YTD");
  const kpis = useMemo(() => dataSource === "MOCK" ? getMockKPIs(period) : null, [period, dataSource]);

  const [btStartDate, setBtStartDate] = useState("2020-01-01");
  const [btEndDate, setBtEndDate] = useState("2024-12-31");
  const [btCapital, setBtCapital] = useState(100000000);
  const [btStrategy, setBtStrategy] = useState<Strategy>("E03");
  const [btIsRunning, setBtIsRunning] = useState(false);
  const [btResults, setBtResults] = useState<BacktestApiResult | null>(null);
  const [btError, setBtError] = useState<string | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);

  const [compareStrategies, setCompareStrategies] = useState<Strategy[]>(["200TQ", "E03"]);
  const [compareResults, setCompareResults] = useState<Record<Strategy, BacktestApiResult>>({} as Record<Strategy, BacktestApiResult>);
  const [compareRunning, setCompareRunning] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  const [heatmapStrategy, setHeatmapStrategy] = useState<Strategy>("E03");
  const [heatmapEquity, setHeatmapEquity] = useState<Array<{ date: string; value: number }> | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [realEquity, setRealEquity] = useState<Array<{ date: string; value: number }> | null>(null);
  const [realEquityLoading, setRealEquityLoading] = useState(false);

  const [compareMode, setCompareMode] = useState(false);

  useEffect(() => {
    if (dataSource === "REAL") {
      setRealEquityLoading(true);
      fetch("/api/portfolio/equity-history")
        .then((res) => res.json())
        .then((data) => {
          if (data.success && data.equity) {
            setRealEquity(data.equity);
          }
        })
        .catch((err) => console.error("Failed to fetch real equity:", err))
        .finally(() => setRealEquityLoading(false));
    }
  }, [dataSource]);

  const handleRunBacktest = async () => {
    setShowConfirmDialog(false);
    setBtIsRunning(true);
    setBtResults(null);
    setBtError(null);
    
    try {
      const results = await runBacktest(btStartDate, btEndDate, btCapital, btStrategy);
      if (results.status === "error") {
        setBtError(results.message ?? "Unknown error");
      } else {
        setBtResults(results);
      }
    } catch (err) {
      setBtError(err instanceof Error ? err.message : String(err));
    } finally {
      setBtIsRunning(false);
    }
  };

  const handleCompareStrategies = async () => {
    if (compareStrategies.length === 0) return;
    setCompareRunning(true);
    setCompareError(null);
    setCompareResults({} as Record<Strategy, BacktestApiResult>);

    try {
      const promises = compareStrategies.map((s) => runBacktest(btStartDate, btEndDate, btCapital, s));
      const results = await Promise.all(promises);
      const newResults: Record<Strategy, BacktestApiResult> = {} as Record<Strategy, BacktestApiResult>;
      results.forEach((r, i) => {
        if (r.status === "success") {
          newResults[compareStrategies[i]] = r;
        }
      });
      setCompareResults(newResults);
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : String(err));
    } finally {
      setCompareRunning(false);
    }
  };

  const toggleCompareStrategy = (s: Strategy) => {
    setCompareStrategies((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : prev.length < 5 ? [...prev, s] : prev
    );
  };

  const handleRunHeatmap = async () => {
    setHeatmapLoading(true);
    try {
      const result = await runBacktest(btStartDate, btEndDate, btCapital, heatmapStrategy);
      if (result.status === "success" && result.equity) {
        setHeatmapEquity(result.equity);
      }
    } catch (err) {
      console.error("Heatmap generation failed:", err);
    } finally {
      setHeatmapLoading(false);
    }
  };

  return (
    <div className="space-y-8 pb-20">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          분석
          <span className="text-sm font-normal text-muted bg-surface px-2.5 py-0.5 rounded-full border border-border">Analysis</span>
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
        <PeriodSelector selected={period} onChange={setPeriod} />
      </div>

      {/* 1. Overview - KPI Cards */}
      <section>
        <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <LayoutGrid size={18} className="text-muted" />
          성과 요약
          <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Overview</span>
        </h2>
        {kpis ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <KPICard 
              label="CAGR (연환산)" 
              value={kpis.cagr > 0 ? `+${kpis.cagr}` : String(kpis.cagr)} 
              unit="%" 
              trend={kpis.cagr > 0 ? "up" : "down"}
              description="연간 복리 수익률"
            />
            <KPICard 
              label="MDD (최대낙폭)" 
              value={String(kpis.mdd)} 
              unit="%" 
              trend="down"
              description="최고점 대비 최대 손실"
            />
            <KPICard 
              label="변동성" 
              value={String(kpis.volatility)} 
              unit="%" 
              trend="neutral"
              description="연환산 표준편차"
            />
            <KPICard 
              label="샤프 비율" 
              value={String(kpis.sharpe)} 
              trend={kpis.sharpe > 1 ? "up" : "neutral"}
              description="위험 조정 수익률"
            />
          </div>
        ) : (
          <div className="rounded-xl border border-border bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
            <Database size={24} className="opacity-50" />
            <span className="text-sm">실제 성과 데이터가 없습니다</span>
            <span className="text-xs text-muted">거래 기록이 쌓이면 지표가 계산됩니다</span>
          </div>
        )}
      </section>

      {/* 2. Returns Heatmap */}
      <section>
        <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Calendar size={18} className="text-muted" />
          수익률 히트맵
          <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Returns</span>
          {dataSource === "REAL" ? (
            <span className="text-[11px] font-bold text-positive bg-positive-tint px-2 py-0.5 rounded border border-positive/30 ml-2">
              REAL
            </span>
          ) : (
            <span className="text-[11px] font-bold text-choppy bg-choppy-tint px-2 py-0.5 rounded border border-choppy/30 ml-2">
              SIMULATED
            </span>
          )}
        </h2>
        <div className="rounded-xl border border-border bg-surface p-6 space-y-4">
          {dataSource === "REAL" ? (
            <>
              {realEquity && realEquity.length > 0 ? (
                <>
                  <div className="flex items-center gap-2 px-3 py-2 bg-positive-tint border border-positive/20 rounded-lg text-positive text-xs">
                    <TrendingUp size={12} />
                    <span>실제 운용 성과입니다. 거래 기록 기반으로 계산됩니다.</span>
                  </div>
                  <ReturnsHeatmap
                    equity={realEquity}
                    isMock={false}
                    isLoading={realEquityLoading}
                  />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-muted gap-2">
                  <Database size={24} className="opacity-50" />
                  <span className="text-sm">아직 포트폴리오 스냅샷이 없습니다</span>
                  <span className="text-xs text-muted">Command에서 거래 기록을 저장하면 성과가 추적됩니다</span>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="flex items-center gap-2 px-3 py-2 bg-choppy-tint border border-choppy/20 rounded-lg text-choppy text-xs">
                <AlertTriangle size={12} />
                <span>백테스트 시뮬레이션입니다. REAL 모드로 전환하면 실제 성과가 표시됩니다.</span>
              </div>
              <div className="flex items-center gap-4">
                <select
                  value={heatmapStrategy}
                  onChange={(e) => setHeatmapStrategy(e.target.value as Strategy)}
                  className="bg-inset border border-border rounded-lg px-3 py-2 text-sm text-fg"
                >
                  {Object.entries(STRATEGY_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
                <div className="grid grid-cols-2 gap-2 flex-1 max-w-xs">
                  <input
                    type="date"
                    value={btStartDate}
                    onChange={(e) => setBtStartDate(e.target.value)}
                    className="bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg"
                  />
                  <input
                    type="date"
                    value={btEndDate}
                    onChange={(e) => setBtEndDate(e.target.value)}
                    className="bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg"
                  />
                </div>
                <button
                  onClick={handleRunHeatmap}
                  disabled={heatmapLoading}
                  className="px-4 py-2 bg-info hover:bg-info/80 disabled:bg-inset text-white font-bold rounded-lg text-sm flex items-center gap-2"
                >
                  {heatmapLoading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                  Generate
                </button>
              </div>
              <ReturnsHeatmap
                equity={heatmapEquity}
                isMock={true}
                isLoading={heatmapLoading}
              />
            </>
          )}
        </div>
      </section>

      {/* 3. Attribution */}
      <section>
        <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <PieChart size={18} className="text-muted" />
          성과 분해
          <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Attribution</span>
        </h2>
        {kpis ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-surface border border-border rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <Activity size={16} className="text-info" />
                  <span className="text-sm font-bold text-fg">Exposure</span>
                </div>
                <div className="text-2xl font-bold font-mono text-fg mb-1">78%</div>
                <div className="text-xs text-muted">평균 주식 노출 비중</div>
              </div>
              <div className="bg-surface border border-border rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <Zap size={16} className="text-choppy" />
                  <span className="text-sm font-bold text-fg">Timing</span>
                </div>
                <div className="text-2xl font-bold font-mono text-positive mb-1">+4.2%</div>
                <div className="text-xs text-muted">진입/이탈 타이밍 기여분</div>
              </div>
              <div className="bg-surface border border-border rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingDown size={16} className="text-muted" />
                  <span className="text-sm font-bold text-fg">Cash Drag</span>
                </div>
                <div className="text-2xl font-bold font-mono text-negative mb-1">-1.8%</div>
                <div className="text-xs text-muted">현금 보유로 인한 기회비용</div>
              </div>
            </div>
            
            <div className="mt-4 rounded-xl border border-border bg-surface p-5">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 size={16} className="text-purple-400" />
                <span className="text-sm font-bold text-fg">시그널 상태 분석</span>
                <span className="text-xs text-muted bg-surface px-2 py-0.5 rounded-full">E03 ON/OFF</span>
              </div>
              
              {/* Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="bg-inset/50 border border-border rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 rounded-full bg-positive" />
                    <span className="text-xs text-muted">ON 상태</span>
                  </div>
                  <div className="text-xl font-bold font-mono text-fg">245<span className="text-sm text-muted ml-1">일</span></div>
                  <div className="text-xs text-muted mt-1">60.5% · <span className="text-positive">+28.5%p</span> 기여</div>
                </div>
                <div className="bg-inset/50 border border-border rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 rounded-full bg-choppy" />
                    <span className="text-xs text-muted">OFF 상태</span>
                  </div>
                  <div className="text-xl font-bold font-mono text-fg">160<span className="text-sm text-muted ml-1">일</span></div>
                  <div className="text-xs text-muted mt-1">39.5% · <span className="text-positive">+4.6%p</span> 기여</div>
                </div>
                <div className="bg-inset/50 border border-border rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity size={12} className="text-info" />
                    <span className="text-xs text-muted">상태 전환</span>
                  </div>
                  <div className="text-xl font-bold font-mono text-fg">24<span className="text-sm text-muted ml-1">회</span></div>
                  <div className="text-xs text-muted mt-1">연평균 3.3회 전환</div>
                </div>
              </div>

              {/* Distribution Bar */}
              <div className="mb-3">
                <div className="text-xs text-muted mb-2">일수 분포</div>
                <div className="w-full h-6 rounded-lg overflow-hidden flex">
                  <div className="bg-positive h-full flex items-center justify-center" style={{ width: '60.5%' }}>
                    <span className="text-[11px] font-bold text-white">ON 60.5%</span>
                  </div>
                  <div className="bg-choppy h-full flex items-center justify-center" style={{ width: '39.5%' }}>
                    <span className="text-[11px] font-bold text-white">OFF 39.5%</span>
                  </div>
                </div>
              </div>

              {/* Current State */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted">현재 상태</span>
                <span className="px-2 py-1 rounded-full bg-positive-tint text-positive font-bold">ON</span>
              </div>
            </div>
          </>
        ) : (
          <div className="rounded-xl border border-border bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
            <PieChart size={24} className="opacity-50" />
            <span className="text-sm">성과 분해 데이터가 없습니다</span>
            <span className="text-xs text-muted">거래 기록이 쌓이면 분석이 가능합니다</span>
          </div>
        )}
      </section>

      {/* 4. Intel Lab (Backtest) - Collapsible */}
      <details className="group" open>
        <summary className="list-none cursor-pointer">
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <FlaskConical size={18} className="text-muted" />
            Intel Lab
            <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">Backtest</span>
            <span className="text-[11px] font-bold text-choppy bg-choppy-tint px-2 py-0.5 rounded border border-choppy/30 ml-2">
              NOT FOR TODAY&apos;S EXECUTION
            </span>
            <ChevronDown size={16} className="text-muted ml-auto group-open:rotate-180 transition-transform" />
          </h2>
        </summary>
        
        <div className="rounded-xl border border-border bg-surface p-6 space-y-6">
          {showConfirmDialog && (
            <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
              <div className="bg-inset border border-border rounded-xl p-6 max-w-md w-full shadow-2xl">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-choppy-tint rounded-full">
                    <AlertTriangle size={24} className="text-choppy" />
                  </div>
                  <h3 className="text-lg font-bold text-fg">연구 모드 실행</h3>
                </div>
                <p className="text-sm text-muted mb-4">
                  이 기능은 <strong className="text-fg">오늘 매매 결정을 위한 것이 아닙니다</strong>.<br/>
                  전략 검증 및 연구 목적으로만 사용하세요.
                </p>
                <p className="text-xs text-choppy bg-choppy-tint border border-choppy/30 rounded-lg p-3 mb-6">
                  ⚠️ 오늘의 실행 지시는 Command 페이지에서 확인하세요
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowConfirmDialog(false)}
                    className="flex-1 px-4 py-2 bg-surface hover:bg-surface text-fg rounded-lg text-sm font-medium transition-colors"
                  >
                    취소
                  </button>
                  <button
                    onClick={compareMode ? handleCompareStrategies : handleRunBacktest}
                    className="flex-1 px-4 py-2 bg-choppy hover:bg-choppy text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    계속 실행
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between">
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                compareMode
                  ? "bg-info-tint border border-info text-info"
                  : "bg-surface border border-border text-muted hover:border-border"
              }`}
            >
              {compareMode ? <ToggleRight size={18} /> : <ToggleLeft size={18} />}
              Compare Mode
              {compareMode && <span className="text-xs">({compareStrategies.length} selected)</span>}
            </button>
          </div>

          {compareMode ? (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {(Object.keys(STRATEGY_LABELS) as Strategy[]).map((s) => (
                  <button
                    key={s}
                    onClick={() => toggleCompareStrategy(s)}
                    className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors flex items-center gap-1.5 ${
                      compareStrategies.includes(s)
                        ? "bg-info-tint border-info text-info"
                        : "bg-surface border-border text-muted hover:border-border"
                    }`}
                  >
                    {compareStrategies.includes(s) && <Check size={12} />}
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: STRATEGY_COLORS[s] }} />
                    {s}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs text-muted flex items-center gap-1.5">
                    <Calendar size={12} />
                    시작일
                  </label>
                  <input
                    type="date"
                    value={btStartDate}
                    onChange={(e) => setBtStartDate(e.target.value)}
                    className="w-full bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs text-muted flex items-center gap-1.5">
                    <Calendar size={12} />
                    종료일
                  </label>
                  <input
                    type="date"
                    value={btEndDate}
                    onChange={(e) => setBtEndDate(e.target.value)}
                    className="w-full bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs text-muted flex items-center gap-1.5">
                    <DollarSign size={12} />
                    초기 자본 (만원)
                  </label>
                  <input
                    type="number"
                    value={btCapital / 10000}
                    onChange={(e) => setBtCapital(Number(e.target.value) * 10000)}
                    min={100}
                    step={100}
                    className="w-full bg-inset border border-border rounded-lg px-3 py-2 text-sm font-mono text-fg"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs text-muted invisible">실행</label>
                  <button
                    onClick={() => setShowConfirmDialog(true)}
                    disabled={compareRunning || compareStrategies.length === 0}
                    className="w-full bg-info hover:bg-info/80 disabled:bg-inset disabled:cursor-not-allowed text-white font-bold rounded-lg px-4 py-2 text-sm flex items-center justify-center gap-2 transition-colors"
                  >
                    {compareRunning ? (
                      <>
                        <Loader2 size={16} className="animate-spin" />
                        실행 중...
                      </>
                    ) : (
                      <>
                        <Play size={16} />
                        Compare ({compareStrategies.length})
                      </>
                    )}
                  </button>
                </div>
              </div>
              {compareError && (
                <div className="text-negative text-xs">{compareError}</div>
              )}
              {Object.keys(compareResults).length > 0 && (
                <>
                  <div className="rounded-xl border border-border overflow-hidden">
                    <div className="grid grid-cols-7 gap-2 p-3 border-b border-border bg-inset/50 text-xs text-muted uppercase">
                      <div>전략</div>
                      <div className="text-right">CAGR</div>
                      <div className="text-right">MDD</div>
                      <div className="text-right">Sharpe</div>
                      <div className="text-right">Sortino</div>
                      <div className="text-right">Calmar</div>
                      <div className="text-right">Final</div>
                    </div>
                    {Object.entries(compareResults).map(([s, r]) => r.metrics && (
                      <div key={s} className="grid grid-cols-7 gap-2 p-3 border-b border-border/50 text-sm">
                        <div className="font-bold flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: STRATEGY_COLORS[s as Strategy] }} />
                          {s}
                        </div>
                        <div className={`text-right font-mono ${r.metrics.CAGR >= 0 ? "text-positive" : "text-negative"}`}>
                          {r.metrics.CAGR >= 0 ? "+" : ""}{r.metrics.CAGR.toFixed(1)}%
                        </div>
                        <div className="text-right font-mono text-negative">{r.metrics.MDD.toFixed(1)}%</div>
                        <div className="text-right font-mono">{r.metrics.Sharpe.toFixed(2)}</div>
                        <div className="text-right font-mono">{r.metrics.Sortino.toFixed(2)}</div>
                        <div className="text-right font-mono">{r.metrics.Calmar.toFixed(2)}</div>
                        <div className="text-right font-mono">{r.metrics.Final.toFixed(2)}x</div>
                      </div>
                    ))}
                  </div>
                  <EquityCurveChart
                    strategies={Object.entries(compareResults)
                      .filter(([, r]) => r.equity && r.equity.length > 0)
                      .map(([s, r]) => ({
                        name: s,
                        color: STRATEGY_COLORS[s as Strategy],
                        data: r.equity || [],
                      }))}
                    height={400}
                  />
                </>
              )}
              {Object.keys(compareResults).length === 0 && !compareRunning && (
                <div className="rounded-xl border border-border bg-inset/50 p-8 flex flex-col items-center justify-center text-muted gap-2 border-dashed">
                  <Layers size={24} className="opacity-50" />
                  <span className="text-sm">전략을 선택하고 Compare를 클릭하세요</span>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <BacktestControls
                startDate={btStartDate}
                endDate={btEndDate}
                capital={btCapital}
                strategy={btStrategy}
                isRunning={btIsRunning}
                onStartDateChange={setBtStartDate}
                onEndDateChange={setBtEndDate}
                onCapitalChange={setBtCapital}
                onStrategyChange={setBtStrategy}
                onRun={() => setShowConfirmDialog(true)}
              />

              {btError && (
                <div className="rounded-xl border border-negative/30 bg-negative-tint p-4 flex items-center gap-3 text-negative">
                  <AlertTriangle size={20} />
                  <div>
                    <div className="font-bold text-sm">백테스트 실행 실패</div>
                    <div className="text-xs text-negative/80">{btError}</div>
                  </div>
                </div>
              )}

              {btResults && btResults.status === "success" && btResults.metrics && btResults.equity && (
                <SingleStrategyPanel
                  strategyName={btResults.experiment || btStrategy}
                  strategyColor={STRATEGY_COLORS[btStrategy]}
                  equity={btResults.equity}
                  metrics={btResults.metrics}
                  startDate={btStartDate}
                  endDate={btEndDate}
                />
              )}

              {!btResults && !btIsRunning && !btError && (
                <div className="rounded-xl border border-border bg-inset/50 p-8 flex flex-col items-center justify-center text-muted gap-2 border-dashed">
                  <FlaskConical size={24} className="opacity-50" />
                  <span className="text-sm">파라미터를 설정하고 Run Backtest를 클릭하세요</span>
                  <p className="text-xs text-muted mt-2 text-center max-w-md">
                    이 섹션은 전략 검증 및 백테스트용입니다. 
                    오늘의 매매 결정에는 Command 페이지를 사용하세요.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
