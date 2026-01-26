"use client";

import { useState, useMemo } from "react";
import { TrendingUp, TrendingDown, BarChart3, Activity, Zap, FlaskConical, Info, LineChart, Play, Loader2, Calendar, DollarSign, Layers, LayoutGrid, PieChart, AlertTriangle, ChevronDown, Database } from "lucide-react";
import { useDataSource } from "../../../lib/stores/settings-store";

type Period = "1M" | "3M" | "6M" | "YTD" | "ALL";
type Strategy = "E03" | "200TQ" | "BOTH";

// Mock KPI data
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

// Mock Backtest Results Generator
function runMockBacktest(startDate: string, endDate: string, capital: number, strategy: Strategy) {
  // Simulate processing delay and return mock results
  const daysDiff = Math.floor((new Date(endDate).getTime() - new Date(startDate).getTime()) / (1000 * 60 * 60 * 24));
  const years = daysDiff / 365;
  
  const baseResults = {
    E03: { cagr: 38.2 + Math.random() * 10, mdd: -12.5 - Math.random() * 5, sharpe: 1.42 + Math.random() * 0.3, volatility: 24.8 + Math.random() * 5 },
    "200TQ": { cagr: 32.1 + Math.random() * 8, mdd: -15.2 - Math.random() * 4, sharpe: 1.28 + Math.random() * 0.2, volatility: 22.3 + Math.random() * 4 },
  };
  
  const finalValue = capital * Math.pow(1 + (baseResults.E03.cagr / 100), years);
  const totalReturn = ((finalValue - capital) / capital) * 100;
  
  return {
    strategy,
    startDate,
    endDate,
    initialCapital: capital,
    finalValue: Math.round(finalValue),
    totalReturn: totalReturn.toFixed(1),
    metrics: strategy === "BOTH" ? baseResults : { [strategy]: baseResults[strategy] },
    trades: Math.floor(daysDiff / 20), // Approx 1 trade per 20 days
  };
}

// Period Selector
function PeriodSelector({ selected, onChange }: { selected: Period; onChange: (p: Period) => void }) {
  const periods: Period[] = ["1M", "3M", "6M", "YTD", "ALL"];
  
  return (
    <div className="flex gap-1 bg-neutral-900 p-1 rounded-lg">
      {periods.map((p) => (
        <button
          key={p}
          onClick={() => onChange(p)}
          className={`px-3 py-1.5 text-xs font-bold rounded-md transition-colors ${
            selected === p 
              ? "bg-neutral-700 text-white" 
              : "text-neutral-500 hover:text-neutral-300"
          }`}
        >
          {p}
        </button>
      ))}
    </div>
  );
}

// KPI Card
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
    <div className="bg-surface border border-neutral-800 rounded-xl p-5">
      <div className="text-xs text-muted mb-1">{label}</div>
      <div className={`text-2xl font-bold font-mono ${trendColor} flex items-baseline gap-1`}>
        {value}
        {unit && <span className="text-sm text-muted font-normal">{unit}</span>}
      </div>
      {description && <div className="text-xs text-neutral-600 mt-2">{description}</div>}
    </div>
  );
}

// Backtest Input Controls
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
  const strategies: { value: Strategy; label: string }[] = [
    { value: "E03", label: "E03 전략" },
    { value: "200TQ", label: "200TQ 전략" },
    { value: "BOTH", label: "둘 다 비교" },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
      {/* Start Date */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted flex items-center gap-1.5">
          <Calendar size={12} />
          시작일
        </label>
        <input
          type="date"
          value={startDate}
          onChange={(e) => onStartDateChange(e.target.value)}
          className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm font-mono text-fg focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500"
        />
      </div>

      {/* End Date */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted flex items-center gap-1.5">
          <Calendar size={12} />
          종료일
        </label>
        <input
          type="date"
          value={endDate}
          onChange={(e) => onEndDateChange(e.target.value)}
          className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm font-mono text-fg focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500"
        />
      </div>

      {/* Initial Capital */}
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
          className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm font-mono text-fg focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500"
        />
      </div>

      {/* Strategy Selector */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted flex items-center gap-1.5">
          <Layers size={12} />
          전략
        </label>
        <select
          value={strategy}
          onChange={(e) => onStrategyChange(e.target.value as Strategy)}
          className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-fg focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500"
        >
          {strategies.map((s) => (
            <option key={s.value} value={s.value}>{s.label}</option>
          ))}
        </select>
      </div>

      {/* Run Button */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted invisible">실행</label>
        <button
          onClick={onRun}
          disabled={isRunning}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-700 disabled:cursor-not-allowed text-white font-bold rounded-lg px-4 py-2 text-sm flex items-center justify-center gap-2 transition-colors"
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

// Backtest Results Display
function BacktestResults({ results }: { results: ReturnType<typeof runMockBacktest> | null }) {
  if (!results) return null;

  const metrics = results.strategy === "BOTH" 
    ? Object.entries(results.metrics as Record<string, { cagr: number; mdd: number; sharpe: number; volatility: number }>)
    : Object.entries(results.metrics);

    return (
    <div className="space-y-4 mt-6">
      {/* Research Warning Banner */}
      <div className="flex items-center gap-2 px-4 py-2 bg-amber-950/30 border border-amber-900/50 rounded-lg text-amber-400 text-xs">
        <AlertTriangle size={14} />
        <span className="font-bold">오늘 Verdict와 무관</span>
        <span className="text-amber-500/80">— 이 결과는 연구/검증용이며, 오늘 매매 결정에 사용하지 마세요</span>
      </div>
      
      {/* Summary Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-neutral-900/50 rounded-xl border border-neutral-800">
        <div>
          <div className="text-xs text-muted">초기 자본</div>
          <div className="text-lg font-bold font-mono text-fg">{(results.initialCapital / 10000).toLocaleString()}만원</div>
        </div>
        <div>
          <div className="text-xs text-muted">최종 자산</div>
          <div className="text-lg font-bold font-mono text-positive">{(results.finalValue / 10000).toLocaleString()}만원</div>
        </div>
        <div>
          <div className="text-xs text-muted">총 수익률</div>
          <div className="text-lg font-bold font-mono text-positive">+{results.totalReturn}%</div>
        </div>
        <div>
          <div className="text-xs text-muted">총 매매 횟수</div>
          <div className="text-lg font-bold font-mono text-fg">{results.trades}회</div>
        </div>
      </div>

      {/* Equity Curve Placeholder */}
      <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
        <LineChart size={32} className="text-blue-400 opacity-70" />
        <span className="text-sm">Equity Curve 차트</span>
        <p className="text-xs text-neutral-600 text-center">
          {results.startDate} ~ {results.endDate} 기간의 자산 곡선
        </p>
      </div>

      {/* Performance Metrics Table */}
      <div className="rounded-xl border border-neutral-800 bg-surface overflow-hidden">
        <div className="grid grid-cols-5 gap-4 p-4 border-b border-neutral-800 bg-neutral-900/50">
          <div className="text-xs text-muted uppercase tracking-wider">전략</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">CAGR</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">MDD</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">Sharpe</div>
          <div className="text-xs text-muted uppercase tracking-wider text-right">변동성</div>
        </div>
        <div className="divide-y divide-neutral-800">
          {metrics.map(([name, data]) => (
            <div key={name} className="grid grid-cols-5 gap-4 p-4 hover:bg-neutral-800/20 transition-colors">
              <div className="text-sm font-bold text-fg">{name}</div>
              <div className="text-sm font-mono text-positive text-right">+{data.cagr.toFixed(1)}%</div>
              <div className="text-sm font-mono text-negative text-right">{data.mdd.toFixed(1)}%</div>
              <div className="text-sm font-mono text-fg text-right">{data.sharpe.toFixed(2)}</div>
              <div className="text-sm font-mono text-muted text-right">{data.volatility.toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function AnalysisPage() {
  const dataSource = useDataSource();
  const [period, setPeriod] = useState<Period>("YTD");
  const kpis = useMemo(() => dataSource === "MOCK" ? getMockKPIs(period) : null, [period, dataSource]);

  // Backtest state
  const [btStartDate, setBtStartDate] = useState("2024-01-01");
  const [btEndDate, setBtEndDate] = useState("2024-12-31");
  const [btCapital, setBtCapital] = useState(100000000); // 1억원
  const [btStrategy, setBtStrategy] = useState<Strategy>("E03");
  const [btIsRunning, setBtIsRunning] = useState(false);
    const [btResults, setBtResults] = useState<ReturnType<typeof runMockBacktest> | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);

  const handleRunBacktest = async () => {
    setShowConfirmDialog(false);
    setBtIsRunning(true);
    setBtResults(null);
    
    // Simulate async processing
    await new Promise((resolve) => setTimeout(resolve, 1500));
    
    const results = runMockBacktest(btStartDate, btEndDate, btCapital, btStrategy);
    setBtResults(results);
    setBtIsRunning(false);
  };

  return (
    <div className="space-y-8 pb-20">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          분석
          <span className="text-sm font-normal text-muted bg-neutral-800 px-2.5 py-0.5 rounded-full border border-neutral-700">Analysis</span>
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
        <PeriodSelector selected={period} onChange={setPeriod} />
      </div>

      {/* 1. Overview - KPI Cards */}
      <section>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <LayoutGrid size={18} className="text-neutral-400" />
          성과 요약
          <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Overview</span>
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
          <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
            <Database size={24} className="opacity-50" />
            <span className="text-sm">실제 성과 데이터가 없습니다</span>
            <span className="text-xs text-neutral-600">거래 기록이 쌓이면 지표가 계산됩니다</span>
          </div>
        )}
      </section>

      {/* 2. Strategies Comparison */}
      <section>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Layers size={18} className="text-neutral-400" />
          전략 비교
          <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Strategies</span>
        </h2>
        {kpis ? (
          <>
            <div className="rounded-xl border border-neutral-800 bg-surface overflow-hidden">
              {/* Comparison Header */}
              <div className="grid grid-cols-3 gap-4 p-5 border-b border-neutral-800 bg-neutral-900/50">
                <div className="text-xs text-muted uppercase tracking-wider">지표</div>
                <div className="text-xs text-muted uppercase tracking-wider text-center">E03 전략</div>
                <div className="text-xs text-muted uppercase tracking-wider text-center">200TQ 전략</div>
              </div>
              
              {/* Comparison Rows */}
              <div className="divide-y divide-neutral-800">
                <div className="grid grid-cols-3 gap-4 p-5 hover:bg-neutral-800/20">
                  <div className="text-sm text-muted">수익률</div>
                  <div className="text-sm font-mono text-positive text-center">+{kpis.cagr}%</div>
                  <div className="text-sm font-mono text-positive text-center">+{(kpis.cagr * 0.85).toFixed(1)}%</div>
                </div>
                <div className="grid grid-cols-3 gap-4 p-5 hover:bg-neutral-800/20">
                  <div className="text-sm text-muted">MDD</div>
                  <div className="text-sm font-mono text-negative text-center">{kpis.mdd}%</div>
                  <div className="text-sm font-mono text-negative text-center">{(kpis.mdd * 1.2).toFixed(1)}%</div>
                </div>
                <div className="grid grid-cols-3 gap-4 p-5 hover:bg-neutral-800/20">
                  <div className="text-sm text-muted">샤프</div>
                  <div className="text-sm font-mono text-fg text-center">{kpis.sharpe}</div>
                  <div className="text-sm font-mono text-fg text-center">{(kpis.sharpe * 0.9).toFixed(2)}</div>
                </div>
              </div>
            </div>
            
            {/* Equity Curve Placeholder */}
            <div className="mt-4 rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
              <LineChart size={24} className="opacity-50" />
              <span className="text-sm">Equity Curve 차트 준비 중...</span>
            </div>
          </>
        ) : (
          <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
            <Layers size={24} className="opacity-50" />
            <span className="text-sm">전략 비교 데이터가 없습니다</span>
            <span className="text-xs text-neutral-600">거래 기록이 쌓이면 비교 분석이 가능합니다</span>
          </div>
        )}
      </section>

      {/* 3. Attribution */}
      <section>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <PieChart size={18} className="text-neutral-400" />
          성과 분해
          <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Attribution</span>
        </h2>
        {kpis ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-surface border border-neutral-800 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <Activity size={16} className="text-blue-400" />
                  <span className="text-sm font-bold text-fg">Exposure</span>
                </div>
                <div className="text-2xl font-bold font-mono text-fg mb-1">78%</div>
                <div className="text-xs text-muted">평균 주식 노출 비중</div>
              </div>
              <div className="bg-surface border border-neutral-800 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <Zap size={16} className="text-amber-400" />
                  <span className="text-sm font-bold text-fg">Timing</span>
                </div>
                <div className="text-2xl font-bold font-mono text-positive mb-1">+4.2%</div>
                <div className="text-xs text-muted">진입/이탈 타이밍 기여분</div>
              </div>
              <div className="bg-surface border border-neutral-800 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingDown size={16} className="text-neutral-400" />
                  <span className="text-sm font-bold text-fg">Cash Drag</span>
                </div>
                <div className="text-2xl font-bold font-mono text-negative mb-1">-1.8%</div>
                <div className="text-xs text-muted">현금 보유로 인한 기회비용</div>
              </div>
            </div>
            
            {/* Event Analysis Placeholder */}
            <div className="mt-4 rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
              <BarChart3 size={24} className="opacity-50" />
              <span className="text-sm">이벤트 기반 분석 (Down/Focus/Overheat) 준비 중...</span>
            </div>
          </>
        ) : (
          <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
            <PieChart size={24} className="opacity-50" />
            <span className="text-sm">성과 분해 데이터가 없습니다</span>
            <span className="text-xs text-neutral-600">거래 기록이 쌓이면 분석이 가능합니다</span>
          </div>
        )}
      </section>

            {/* 4. Intel Lab (Backtest) - Collapsible */}
      <details className="group">
        <summary className="list-none cursor-pointer">
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <FlaskConical size={18} className="text-neutral-400" />
            Intel Lab
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Backtest</span>
            <span className="text-[10px] font-bold text-amber-400 bg-amber-950/30 px-2 py-0.5 rounded border border-amber-900/50 ml-2">
              NOT FOR TODAY'S EXECUTION
            </span>
            <ChevronDown size={16} className="text-neutral-500 ml-auto group-open:rotate-180 transition-transform" />
          </h2>
        </summary>
        
        <div className="rounded-xl border border-neutral-800 bg-surface p-6">
          {/* Confirmation Dialog */}
          {showConfirmDialog && (
            <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
              <div className="bg-neutral-900 border border-neutral-700 rounded-xl p-6 max-w-md w-full shadow-2xl">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-amber-500/20 rounded-full">
                    <AlertTriangle size={24} className="text-amber-400" />
                  </div>
                  <h3 className="text-lg font-bold text-fg">연구 모드 실행</h3>
                </div>
                <p className="text-sm text-muted mb-4">
                  이 기능은 <strong className="text-fg">오늘 매매 결정을 위한 것이 아닙니다</strong>.<br/>
                  전략 검증 및 연구 목적으로만 사용하세요.
                </p>
                <p className="text-xs text-amber-400 bg-amber-950/30 border border-amber-900/50 rounded-lg p-3 mb-6">
                  ⚠️ 오늘의 실행 지시는 Command 페이지에서 확인하세요
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowConfirmDialog(false)}
                    className="flex-1 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-sm font-medium transition-colors"
                  >
                    취소
                  </button>
                  <button
                    onClick={handleRunBacktest}
                    className="flex-1 px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    계속 실행
                  </button>
                </div>
              </div>
            </div>
          )}
          
          {/* Backtest Controls */}
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

          {/* Results */}
          <BacktestResults results={btResults} />

          {/* Empty State */}
          {!btResults && !btIsRunning && (
            <div className="mt-6 rounded-xl border border-neutral-800 bg-neutral-900/50 p-8 flex flex-col items-center justify-center text-muted gap-2 border-dashed">
              <FlaskConical size={24} className="opacity-50" />
              <span className="text-sm">파라미터를 설정하고 Run Backtest를 클릭하세요</span>
              <p className="text-xs text-neutral-600 mt-2 text-center max-w-md">
                이 섹션은 전략 검증 및 백테스트용입니다. 
                오늘의 매매 결정에는 Command 페이지를 사용하세요.
              </p>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}

