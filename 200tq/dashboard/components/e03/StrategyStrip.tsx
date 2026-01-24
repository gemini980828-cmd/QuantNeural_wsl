"use client";

import React from "react";
import type { Segment, TQ200StateEn } from "../../lib/ops/e03/indicators";

/**
 * StrategyStrip - Horizontal state strips for Story Mode chart
 * Renders E03 Target or 200TQ State as colored segments
 */

type E03TargetState = "ON" | "OFF";

interface StrategyStripProps {
  type: "e03" | "200tq";
  segments: Segment<E03TargetState | TQ200StateEn>[];
  xScale: (idx: number) => number;
  height?: number; // 10-14px (default 12)
  chartWidth: number;
  label?: string;
}

// Semantic color classes (CSS variables in globals.css)
const E03_COLORS: Record<E03TargetState, string> = {
  ON: "fill-[var(--strip-e03-on,#10b981)]", // emerald-500
  OFF: "fill-[var(--strip-e03-off,#6b7280)]", // gray-500
};

const TQ200_COLORS: Record<TQ200StateEn, string> = {
  Bear: "fill-[var(--strip-200tq-bear,#ef4444)]", // red-500
  Focus: "fill-[var(--strip-200tq-focus,#3b82f6)]", // blue-500
  Overheat: "fill-[var(--strip-200tq-overheat,#f97316)]", // orange-500
};

export default function StrategyStrip({
  type,
  segments,
  xScale,
  height = 12,
  chartWidth,
  label,
}: StrategyStripProps) {
  const colorMap = type === "e03" ? E03_COLORS : TQ200_COLORS;

  return (
    <g className="strategy-strip">
      {/* Label on left */}
      {label && (
        <text
          x={-8}
          y={height / 2}
          fill="#9ca3af"
          fontSize={9}
          textAnchor="end"
          dominantBaseline="middle"
        >
          {label}
        </text>
      )}
      
      {/* Segments */}
      {segments.map((seg, i) => {
        const x1 = xScale(seg.start);
        const x2 = xScale(seg.end);
        const width = Math.max(0, x2 - x1);
        
        // Skip if segment is outside visible range
        if (x2 < 0 || x1 > chartWidth) return null;
        
        // Clamp to visible range
        const clampedX = Math.max(0, x1);
        const clampedW = Math.min(chartWidth, x2) - clampedX;
        
        const colorClass = colorMap[seg.value as keyof typeof colorMap] || "fill-gray-400";
        
        return (
          <rect
            key={`${type}-${i}`}
            x={clampedX}
            y={0}
            width={clampedW}
            height={height}
            className={colorClass}
            rx={2}
          />
        );
      })}
    </g>
  );
}
