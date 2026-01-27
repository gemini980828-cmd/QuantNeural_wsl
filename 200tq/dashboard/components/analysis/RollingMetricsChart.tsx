"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { calculateRollingReturns, calculateRollingSharpe } from "./utils/risk-metrics";

interface EquityPoint {
  date: string;
  value: number;
}

interface RollingMetricsChartProps {
  equity: EquityPoint[] | null;
  height?: number;
  windowDays?: number;
  riskFreeRate?: number;
  metric?: "return" | "sharpe" | "both";
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

interface TooltipPayloadItem {
  name: string;
  value: number;
  color: string;
  dataKey: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-3 shadow-xl">
      <p className="text-xs text-neutral-400 mb-2 font-mono">{label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2 text-sm">
          <div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-neutral-300">{entry.name}:</span>
          <span className="font-mono font-bold" style={{ color: entry.color }}>
            {entry.dataKey === "return" 
              ? `${(entry.value * 100).toFixed(1)}%`
              : entry.value.toFixed(2)
            }
          </span>
        </div>
      ))}
    </div>
  );
}

export function RollingMetricsChart({
  equity,
  height = 200,
  windowDays = 252,
  riskFreeRate = 0.02,
  metric = "both",
}: RollingMetricsChartProps) {
  const chartData = useMemo(() => {
    if (!equity || equity.length < windowDays) {
      return [];
    }

    const showReturn = metric === "return" || metric === "both";
    const showSharpe = metric === "sharpe" || metric === "both";

    const rollingReturns = showReturn 
      ? calculateRollingReturns(equity, windowDays) 
      : [];
    const rollingSharpe = showSharpe 
      ? calculateRollingSharpe(equity, windowDays, riskFreeRate) 
      : [];

    const merged = rollingReturns.map((r: { date: string; value: number }, i: number) => ({
      date: r.date,
      return: showReturn ? r.value : undefined,
      sharpe: showSharpe ? rollingSharpe[i]?.value : undefined,
    }));

    const sampleInterval = Math.max(1, Math.floor(merged.length / 100));
    const sampled = merged.filter((_: unknown, i: number) => i % sampleInterval === 0);
    
    const lastPoint = merged[merged.length - 1];
    if (merged.length > 0 && sampled[sampled.length - 1] !== lastPoint) {
      sampled.push(lastPoint);
    }

    return sampled;
  }, [equity, windowDays, riskFreeRate, metric]);

  if (!equity || equity.length < windowDays || chartData.length === 0) {
    return (
      <div 
        className="flex items-center justify-center text-neutral-500 text-sm"
        style={{ height }}
      >
        Not enough data for {windowDays}-day rolling metrics
      </div>
    );
  }

  const showReturn = metric === "return" || metric === "both";
  const showSharpe = metric === "sharpe" || metric === "both";

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#475569"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickFormatter={formatDate}
            tickLine={{ stroke: "#475569" }}
          />
          {showReturn && (
            <YAxis
              yAxisId="return"
              stroke="#3b82f6"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={{ stroke: "#475569" }}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              orientation="left"
            />
          )}
          {showSharpe && (
            <YAxis
              yAxisId="sharpe"
              stroke="#f59e0b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={{ stroke: "#475569" }}
              tickFormatter={(v: number) => v.toFixed(1)}
              orientation={showReturn ? "right" : "left"}
            />
          )}
          {showReturn && (
            <ReferenceLine
              yAxisId="return"
              y={0}
              stroke="#475569"
              strokeDasharray="5 5"
            />
          )}
          {showSharpe && (
            <ReferenceLine
              yAxisId="sharpe"
              y={1}
              stroke="#10b981"
              strokeDasharray="5 5"
              label={{
                value: "Sharpe=1",
                fill: "#10b981",
                fontSize: 10,
                position: "insideTopRight",
              }}
            />
          )}
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "#475569", strokeWidth: 1 }} />
          <Legend
            wrapperStyle={{ color: "#94a3b8", paddingTop: "8px" }}
            iconType="line"
          />
          {showReturn && (
            <Line
              yAxisId="return"
              type="monotone"
              dataKey="return"
              name="12M Return"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "#3b82f6", stroke: "#1e293b", strokeWidth: 2 }}
            />
          )}
          {showSharpe && (
            <Line
              yAxisId="sharpe"
              type="monotone"
              dataKey="sharpe"
              name="12M Sharpe"
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              activeDot={{ r: 4, fill: "#f59e0b", stroke: "#1e293b", strokeWidth: 2 }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
