"use client";

import { useMemo } from "react";
import { TrendingDown, Activity, Clock, AlertTriangle, Target, Zap } from "lucide-react";
import { EquityCurveChart } from "./EquityCurveChart";
import { DrawdownChart } from "./DrawdownChart";
import { RollingMetricsChart } from "./RollingMetricsChart";
import { calculateRiskMetrics, calculateMaxDrawdown, calculateDrawdownPeriods } from "./utils/risk-metrics";

interface EquityPoint {
  date: string;
  value: number;
}

interface BacktestMetrics {
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
}

interface SingleStrategyPanelProps {
  strategyName: string;
  strategyColor: string;
  equity: EquityPoint[];
  metrics: BacktestMetrics;
  startDate: string;
  endDate: string;
}

function MetricCard({ 
  icon: Icon, 
  label, 
  value, 
  subValue,
  valueColor = "text-fg",
  description 
}: { 
  icon: React.ElementType;
  label: string; 
  value: string; 
  subValue?: string;
  valueColor?: string;
  description?: string;
}) {
  return (
    <div className="bg-neutral-900/50 border border-neutral-800 rounded-lg p-4">
      <div className="flex items-center gap-2 text-xs text-muted mb-2">
        <Icon size={12} />
        {label}
      </div>
      <div className={`font-mono font-bold text-lg ${valueColor}`}>
        {value}
        {subValue && <span className="text-xs text-muted ml-1">{subValue}</span>}
      </div>
      {description && <div className="text-[10px] text-neutral-600 mt-1">{description}</div>}
    </div>
  );
}

export function SingleStrategyPanel({
  strategyName,
  strategyColor,
  equity,
  metrics,
  startDate,
  endDate,
}: SingleStrategyPanelProps) {
  const riskMetrics = useMemo(() => {
    if (!equity || equity.length === 0) return null;
    return calculateRiskMetrics(equity);
  }, [equity]);

  const { mdd } = useMemo(() => {
    if (!equity || equity.length === 0) return { mdd: { mdd: 0, peakDate: "", troughDate: "" } };
    return { mdd: calculateMaxDrawdown(equity) };
  }, [equity]);

  const drawdownPeriods = useMemo(() => {
    if (!equity || equity.length === 0) return [];
    return calculateDrawdownPeriods(equity);
  }, [equity]);

  const longestDrawdown = useMemo(() => {
    if (drawdownPeriods.length === 0) return null;
    return drawdownPeriods.reduce((longest, period) => {
      const totalDuration = period.duration + (period.recovery || 0);
      const longestDuration = longest.duration + (longest.recovery || 0);
      return totalDuration > longestDuration ? period : longest;
    }, drawdownPeriods[0]);
  }, [drawdownPeriods]);

  if (!equity || equity.length === 0 || !riskMetrics) {
    return (
      <div className="text-center text-muted py-8">
        No equity data available
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-neutral-800 bg-surface p-6">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: strategyColor }} />
          <span className="text-sm font-bold text-fg">{strategyName}</span>
          <span className="text-xs text-muted ml-auto">{startDate} ~ {endDate}</span>
        </div>
        <EquityCurveChart
          strategies={[{ name: strategyName, color: strategyColor, data: equity }]}
          height={300}
        />
      </div>

      <div className="rounded-xl border border-neutral-800 bg-surface p-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingDown size={16} className="text-red-400" />
          <span className="text-sm font-bold text-fg">Drawdown (Underwater)</span>
          <span className="text-xs text-muted ml-2">MDD: {(mdd.mdd * 100).toFixed(1)}%</span>
        </div>
        <DrawdownChart equity={equity} height={180} showMDD={true} />
      </div>

      <div className="rounded-xl border border-neutral-800 bg-surface p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity size={16} className="text-blue-400" />
          <span className="text-sm font-bold text-fg">Rolling 12M Metrics</span>
        </div>
        <RollingMetricsChart equity={equity} height={200} metric="both" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          icon={TrendingDown}
          label="Max Drawdown"
          value={`${(metrics.MDD).toFixed(1)}%`}
          valueColor="text-red-400"
          description="Peak-to-trough decline"
        />
        <MetricCard
          icon={Clock}
          label="Longest Recovery"
          value={longestDrawdown?.recovery ? `${longestDrawdown.recovery}d` : "N/A"}
          subValue={longestDrawdown?.duration ? `(${longestDrawdown.duration}d down)` : ""}
          description="Time to recover from worst DD"
        />
        <MetricCard
          icon={Target}
          label="Sharpe Ratio"
          value={metrics.Sharpe.toFixed(2)}
          valueColor={metrics.Sharpe >= 1 ? "text-positive" : "text-fg"}
          description="Risk-adjusted return"
        />
        <MetricCard
          icon={Target}
          label="Sortino Ratio"
          value={metrics.Sortino.toFixed(2)}
          valueColor={metrics.Sortino >= 1 ? "text-positive" : "text-fg"}
          description="Downside risk-adjusted"
        />
        <MetricCard
          icon={Zap}
          label="Calmar Ratio"
          value={metrics.Calmar.toFixed(2)}
          valueColor={metrics.Calmar >= 1 ? "text-positive" : "text-fg"}
          description="CAGR / MDD"
        />
        <MetricCard
          icon={AlertTriangle}
          label="Ulcer Index"
          value={riskMetrics.ulcerIndex.toFixed(2)}
          description="Drawdown pain index"
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <div className="bg-neutral-900/30 rounded-lg p-3 border border-neutral-800">
          <span className="text-neutral-500">Volatility:</span>{" "}
          <span className="font-mono">{(riskMetrics.volatility * 100).toFixed(1)}%</span>
        </div>
        <div className="bg-neutral-900/30 rounded-lg p-3 border border-neutral-800">
          <span className="text-neutral-500">Downside Dev:</span>{" "}
          <span className="font-mono">{(riskMetrics.downsideDeviation * 100).toFixed(1)}%</span>
        </div>
        <div className="bg-neutral-900/30 rounded-lg p-3 border border-neutral-800">
          <span className="text-neutral-500">Trades:</span>{" "}
          <span className="font-mono">{metrics.TradesCount}</span>
        </div>
        <div className="bg-neutral-900/30 rounded-lg p-3 border border-neutral-800">
          <span className="text-neutral-500">Tax (22%):</span>{" "}
          <span className="font-mono">{(metrics.TotalTax / 10000).toLocaleString()}만원</span>
        </div>
      </div>
    </div>
  );
}
