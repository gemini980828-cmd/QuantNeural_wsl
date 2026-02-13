"use client";

import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot,
} from "recharts";
import { calculateDrawdownSeries, calculateMaxDrawdown } from "./utils/risk-metrics";

interface EquityPoint {
  date: string;
  value: number;
}

interface DrawdownChartProps {
  equity: EquityPoint[] | null;
  height?: number;
  showMDD?: boolean;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

interface TooltipPayloadItem {
  value: number;
  dataKey: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const drawdown = payload[0]?.value ?? 0;

  return (
    <div className="bg-surface border border-border rounded-lg p-3 shadow-xl">
      <p className="text-xs text-muted mb-1 font-mono">{label}</p>
      <div className="flex items-center gap-2 text-sm">
        <span className="text-fg">Drawdown:</span>
        <span className="font-mono font-bold text-negative">
          {(drawdown * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

export function DrawdownChart({ 
  equity, 
  height = 200, 
  showMDD = true 
}: DrawdownChartProps) {
  const { chartData, mddPoint } = useMemo(() => {
    if (!equity || equity.length === 0) {
      return { chartData: [], mddPoint: null };
    }

    const drawdownSeries = calculateDrawdownSeries(equity);
    const mdd = calculateMaxDrawdown(equity);

    const sampleInterval = Math.max(1, Math.floor(drawdownSeries.length / 100));
    const sampled = drawdownSeries.filter((_: unknown, i: number) => i % sampleInterval === 0);
    
    const lastPoint = drawdownSeries[drawdownSeries.length - 1];
    if (drawdownSeries.length > 0 && sampled[sampled.length - 1] !== lastPoint) {
      sampled.push(lastPoint);
    }

    const mddSampledPoint = showMDD && mdd.troughDate
      ? sampled.find((p: { date: string; drawdown: number }) => p.date === mdd.troughDate) || null
      : null;

    return {
      chartData: sampled,
      mddPoint: mddSampledPoint ? { 
        date: mddSampledPoint.date, 
        drawdown: mdd.mdd 
      } : null,
    };
  }, [equity, showMDD]);

  if (!equity || equity.length === 0 || chartData.length === 0) {
    return (
      <div 
        className="flex items-center justify-center text-muted text-sm"
        style={{ height }}
      >
        No drawdown data available
      </div>
    );
  }

  const minDrawdown = Math.min(...chartData.map((d: { date: string; drawdown: number }) => d.drawdown), -0.01);

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
          <defs>
            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" opacity={0.3} vertical={false} />
          <XAxis
            dataKey="date"
            stroke="rgb(var(--border))"
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickFormatter={formatDate}
            tickLine={{ stroke: "rgb(var(--border))" }}
          />
          <YAxis
            stroke="rgb(var(--border))"
            tick={{ fill: "rgb(var(--muted))", fontSize: 11 }}
            tickLine={{ stroke: "rgb(var(--border))" }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            domain={[minDrawdown * 1.1, 0]}
          />
          <ReferenceLine
            y={0}
            stroke="rgb(var(--border))"
            strokeWidth={1}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "rgb(var(--border))", strokeWidth: 1 }} />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#ef4444"
            strokeWidth={1.5}
            fill="url(#colorDrawdown)"
            fillOpacity={1}
          />
          {showMDD && mddPoint && (
            <ReferenceDot
              x={mddPoint.date}
              y={mddPoint.drawdown}
              r={6}
              fill="#dc2626"
              stroke="rgb(var(--bg))"
              strokeWidth={2}
              label={{
                value: `MDD: ${(mddPoint.drawdown * 100).toFixed(1)}%`,
                position: "top",
                fill: "#dc2626",
                fontSize: 10,
                fontWeight: "bold",
              }}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
