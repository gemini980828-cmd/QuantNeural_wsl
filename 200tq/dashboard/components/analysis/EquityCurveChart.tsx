"use client";

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

interface EquityPoint {
  date: string;
  value: number;
}

interface StrategyEquity {
  name: string;
  color: string;
  data: EquityPoint[];
}

interface EquityCurveChartProps {
  strategies: StrategyEquity[];
  height?: number;
}

const COLORS = [
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#a855f7",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
];

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
            {entry.value.toFixed(2)}x
          </span>
          <span className="text-xs text-neutral-500">
            ({((entry.value - 1) * 100).toFixed(1)}%)
          </span>
        </div>
      ))}
    </div>
  );
}

export function EquityCurveChart({ strategies, height = 400 }: EquityCurveChartProps) {
  if (strategies.length === 0) return null;

  const allDates = new Set<string>();
  strategies.forEach((s) => s.data.forEach((d) => allDates.add(d.date)));
  const sortedDates = Array.from(allDates).sort();

  const chartData = sortedDates.map((date) => {
    const point: Record<string, string | number> = { date };
    strategies.forEach((s) => {
      const match = s.data.find((d) => d.date === date);
      if (match) point[s.name] = match.value;
    });
    return point;
  });

  const sampleInterval = Math.max(1, Math.floor(chartData.length / 30));
  const sampledData = chartData.filter((_, i) => i % sampleInterval === 0);
  if (chartData.length > 0 && sampledData[sampledData.length - 1] !== chartData[chartData.length - 1]) {
    sampledData.push(chartData[chartData.length - 1]);
  }

  const allValues = strategies.flatMap((s) => s.data.map((d) => d.value));
  const minValue = Math.min(...allValues, 1);
  const maxValue = Math.max(...allValues);
  const yMin = Math.max(0.1, Math.floor(minValue * 10) / 10 - 0.1);
  const yMax = Math.ceil(maxValue * 10) / 10 + 0.1;

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={sampledData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#475569"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickFormatter={formatDate}
            tickLine={{ stroke: "#475569" }}
          />
          <YAxis
            stroke="#475569"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={{ stroke: "#475569" }}
            tickFormatter={(v: number) => `${v.toFixed(1)}x`}
            domain={[yMin, yMax]}
          />
          <ReferenceLine
            y={1.0}
            stroke="#10b981"
            strokeWidth={1.5}
            strokeDasharray="5 5"
            label={{
              value: "Breakeven",
              fill: "#10b981",
              fontSize: 10,
              position: "insideTopRight",
            }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "#475569", strokeWidth: 1 }} />
          <Legend
            wrapperStyle={{ color: "#94a3b8", paddingTop: "12px" }}
            iconType="line"
          />
          {strategies.map((s, i) => (
            <Line
              key={s.name}
              type="monotone"
              dataKey={s.name}
              stroke={s.color || COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={false}
              name={s.name}
              activeDot={{ r: 5, fill: s.color || COLORS[i % COLORS.length], stroke: "#1e293b", strokeWidth: 2 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
