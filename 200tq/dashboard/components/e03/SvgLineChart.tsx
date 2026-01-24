"use client";

import { useState, useRef, useEffect } from "react";
import StrategyStrip from "./StrategyStrip";
import type { Segment, TQ200StateEn } from "../../lib/ops/e03/indicators";

interface LineData {
  data: (number | null)[];
  color: string;
  label: string;
  dashed?: boolean;
  isSecondary?: boolean;  // For dual Y-axis: renders on right Y-axis with separate scale
}

// Signal marker at a specific index
export interface SignalMarker {
  idx: number;
  type: "BUY" | "SELL";
  price: number;
}

// Zone band for signal state
export interface SignalZone {
  startIdx: number;
  endIdx: number;
  state: "ON" | "OFF10";
}

// V/E day markers
export interface VEMarker {
  idx: number;
  type: "V" | "E";  // Verdict or Exec day
}

interface SvgLineChartProps {
  lines: LineData[];
  dates?: string[];
  width?: number;
  height?: number;
  showLegend?: boolean;
  markers?: SignalMarker[];
  zones?: SignalZone[];
  // Story Mode props (v1.2)
  e03Segments?: Segment<"ON" | "OFF">[];
  tqqSegments?: Segment<TQ200StateEn>[];
  veMarkers?: VEMarker[];
  showTrades?: boolean;  // false = hide BUY/SELL (Story Mode default)
  showStrips?: boolean;  // true = show E03/200TQ strips (default)
}

export default function SvgLineChart({
  lines,
  dates = [],
  width = 600,
  height = 300,
  showLegend = true,
  markers = [],
  zones = [],
  e03Segments = [],
  tqqSegments = [],
  veMarkers = [],
  showTrades = false,  // Story Mode: hide BUY/SELL by default
  showStrips = true,   // Story Mode: show strips by default
}: SvgLineChartProps) {
  // Responsive Width Logic
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width, height });

  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newWidth } = entry.contentRect;
        // Enforce 2.2:1 aspect ratio, but respect min height
        // If prop height provided is meant to be logic-based, we might override it, 
        // but user asked for aspect-ratio enforcement.
        // Let's use the container width and fixed aspect ratio.
        const newHeight = newWidth / 2.2; 
        
        setDimensions({
           width: newWidth,
           height: Math.max(300, newHeight) // Enforce min height
        });
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  // Use dynamic dimensions if available, else props
  // We use the state 'dimensions' which initializes with props.
  const chartW_Dynamic = dimensions.width;
  const chartH_Dynamic = dimensions.height;

  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Zoom state: [startIdx, endIdx] as percentage of data range (0-1)
  const [zoomRange, setZoomRange] = useState<[number, number]>([0, 1]);
  const zoomRangeRef = useRef(zoomRange);
  zoomRangeRef.current = zoomRange;
  
  // Zoom helper function using ref to avoid stale closure
  const applyZoomRef = useRef<(direction: 1 | -1) => void>(() => {});
  applyZoomRef.current = (direction: 1 | -1) => {
    const zoomFactor = 0.15;
    const [start, end] = zoomRangeRef.current;
    const range = end - start;
    const center = (start + end) / 2;
    
    let newRange = direction > 0 ? range * (1 + zoomFactor) : range * (1 - zoomFactor);
    newRange = Math.max(0.1, Math.min(1, newRange));
    
    let newStart = center - newRange / 2;
    let newEnd = center + newRange / 2;
    
    if (newStart < 0) { newStart = 0; newEnd = newRange; }
    if (newEnd > 1) { newEnd = 1; newStart = 1 - newRange; }
    
    setZoomRange([newStart, newEnd]);
  };
  
  const applyZoom = (direction: 1 | -1) => applyZoomRef.current(direction);
  
  // Native wheel event listener to properly prevent scroll (passive: false required)
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    
    const handleNativeWheel = (e: WheelEvent) => {
      e.preventDefault();
      const direction = e.deltaY > 0 ? 1 : -1;
      applyZoomRef.current(direction as 1 | -1);
    };
    
    svg.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => svg.removeEventListener('wheel', handleNativeWheel);
  }, []);
  
  // Drag-to-pan state
  const [isDragging, setIsDragging] = useState(false);
  const dragStartX = useRef(0);
  const dragStartRange = useRef<[number, number]>([0, 1]);
  
  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoomRange[0] === 0 && zoomRange[1] === 1) return; // No panning when not zoomed
    setIsDragging(true);
    dragStartX.current = e.clientX;
    dragStartRange.current = zoomRange;
  };
  
  const handleDrag = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    const dx = e.clientX - dragStartX.current;
    const chartWidth = width - 70; // padding.left + padding.right
    const rangeWidth = dragStartRange.current[1] - dragStartRange.current[0];
    
    // Calculate pan amount (negative dx = pan right = increase range)
    const panAmount = -(dx / chartWidth) * rangeWidth;
    
    let newStart = dragStartRange.current[0] + panAmount;
    let newEnd = dragStartRange.current[1] + panAmount;
    
    // Clamp to bounds
    if (newStart < 0) { newStart = 0; newEnd = rangeWidth; }
    if (newEnd > 1) { newEnd = 1; newStart = 1 - rangeWidth; }
    
    setZoomRange([newStart, newEnd]);
  };
  
  const handleMouseUp = () => setIsDragging(false);
  
  // Button handlers for manual zoom
  const handleZoomIn = () => applyZoom(-1);
  const handleZoomOut = () => applyZoom(1);
  
  // Reset zoom on double click
  const handleDoubleClick = () => setZoomRange([0, 1]);

  if (lines.length === 0 || lines.every(l => l.data.length === 0)) {
    return (
      <div className="flex items-center justify-center text-neutral-600 text-sm" style={{ width, height }}>
        No data
      </div>
    );
  }

  // Padding: left reduced (for X labels), right increased (for Y labels - Toss style)
  const padding = { top: 30, right: 60, bottom: 40, left: 20 };
  const chartW = chartW_Dynamic - padding.left - padding.right;
  const chartH = chartH_Dynamic - padding.top - padding.bottom;

  // Get max length first
  let maxLen = 0;
  for (const line of lines) {
    if (line.data.length > maxLen) maxLen = line.data.length;
  }
  
  // Calculate visible index range based on zoom
  const visibleStartIdx = Math.floor(zoomRange[0] * (maxLen - 1));
  const visibleEndIdx = Math.ceil(zoomRange[1] * (maxLen - 1));
  const visibleLen = visibleEndIdx - visibleStartIdx + 1;

  // Find min/max for Y scale within visible range (PRIMARY lines only)
  let minY = Infinity;
  let maxY = -Infinity;
  
  // Find min/max for SECONDARY Y scale (right axis)
  let minY2 = Infinity;
  let maxY2 = -Infinity;
  const hasSecondary = lines.some(l => l.isSecondary);

  for (const line of lines) {
    for (let i = visibleStartIdx; i <= visibleEndIdx && i < line.data.length; i++) {
      const v = line.data[i];
      if (v !== null) {
        if (line.isSecondary) {
          if (v < minY2) minY2 = v;
          if (v > maxY2) maxY2 = v;
        } else {
          if (v < minY) minY = v;
          if (v > maxY) maxY = v;
        }
      }
    }
  }
  
  // Track min/max/current for first NON-secondary line (main series) - for annotations
  let rawMin = Infinity;
  let rawMax = -Infinity;
  let minIdx = 0;
  let maxIdx = 0;
  let currentPrice: number | null = null;
  const primaryLine = lines.find(l => !l.isSecondary);
  if (primaryLine && primaryLine.data.length > 0) {
    for (let i = visibleStartIdx; i <= visibleEndIdx && i < primaryLine.data.length; i++) {
      const v = primaryLine.data[i];
      if (v !== null) {
        if (v < rawMin) { rawMin = v; minIdx = i; }
        if (v > rawMax) { rawMax = v; maxIdx = i; }
        currentPrice = v; // Last valid value becomes current
      }
    }
  }

  // Apply padding to Y scales
  const yRange = maxY - minY || 1;
  minY -= yRange * 0.2; // Increased padding to 20% for clearance
  maxY += yRange * 0.2;
  
  const yRange2 = maxY2 - minY2 || 1;
  minY2 -= yRange2 * 0.2;
  maxY2 += yRange2 * 0.2;

  // Scale functions adjusted for zoom
  // Use full chart width (Toss style has data end slightly before right edge naturally)
  const drawWidth = chartW;
  const xScale = (i: number) => ((i - visibleStartIdx) / (visibleLen - 1)) * drawWidth;
  const yScale = (v: number) => chartH - ((v - minY) / (maxY - minY)) * chartH;
  const yScale2 = (v: number) => chartH - ((v - minY2) / (maxY2 - minY2)) * chartH;

  // Build paths (only for visible range)
  const paths = lines.map((line) => {
    const points: string[] = [];
    let started = false;
    const scale = line.isSecondary ? yScale2 : yScale;
    for (let i = visibleStartIdx; i <= visibleEndIdx && i < line.data.length; i++) {
      const v = line.data[i];
      if (v === null) continue;
      const x = xScale(i);
      const y = scale(v);
      if (!started) {
        points.push(`M${x.toFixed(1)},${y.toFixed(1)}`);
        started = true;
      } else {
        points.push(`L${x.toFixed(1)},${y.toFixed(1)}`);
      }
    }
    return points.join(" ");
  });

  // Y-axis ticks (5 ticks) - PRIMARY (left axis)
  const yTicks: number[] = [];
  for (let i = 0; i <= 4; i++) {
    yTicks.push(minY + ((maxY - minY) * i) / 4);
  }
  
  // Y-axis ticks - SECONDARY (right axis)
  const yTicks2: number[] = [];
  if (hasSecondary) {
    for (let i = 0; i <= 4; i++) {
      yTicks2.push(minY2 + ((maxY2 - minY2) * i) / 4);
    }
  }

  // X-axis date ticks - responsive formatting based on visible range
  const xTickCount = 5;
  const xTicks: { idx: number; label: string }[] = [];
  if (dates.length > 0 && visibleLen > 0) {
    // Determine date format based on visible span
    const firstDate = dates[visibleStartIdx];
    const lastDate = dates[visibleEndIdx] || dates[dates.length - 1];
    
    let spanDays = 0;
    if (firstDate && lastDate) {
      const d1 = new Date(firstDate);
      const d2 = new Date(lastDate);
      spanDays = Math.abs((d2.getTime() - d1.getTime()) / (1000 * 60 * 60 * 24));
    }
    
    // Format based on span:
    // > 365 days: Year only (2024)
    // > 90 days: Year-Month (24년 6월)
    // <= 90 days: Month-Day (6/15)
    const formatDate = (d: string): string => {
      if (!d) return "";
      const date = new Date(d);
      if (spanDays > 365) {
        return date.getFullYear().toString();
      } else if (spanDays > 90) {
        const year = date.getFullYear().toString().slice(2);
        const month = date.getMonth() + 1;
        return `${year}년 ${month}월`;
      } else {
        const month = date.getMonth() + 1;
        const day = date.getDate();
        return `${month}/${day}`;
      }
    };
    
    // Generate ticks, avoiding duplicate year labels
    const usedYears = new Set<string>();
    for (let i = 0; i < xTickCount; i++) {
      const idx = visibleStartIdx + Math.floor((i / (xTickCount - 1)) * (visibleLen - 1));
      const d = dates[idx];
      let label = formatDate(d);
      
      // Skip duplicate year labels
      if (spanDays > 365) {
        if (usedYears.has(label)) {
          continue;
        }
        usedYears.add(label);
      }
      
      xTicks.push({ idx, label });
    }
  }

  // Mouse handling
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - padding.left;
    const drawWidth = chartW - 40; // Match the xScale width
    
    // Reverse scale: idx = (x / drawWidth) * (visibleLen - 1) + visibleStartIdx
    let idx = Math.round((x / drawWidth) * (visibleLen - 1)) + visibleStartIdx;
    
    // Clamp to valid range
    if (idx < 0) idx = 0;
    if (idx >= maxLen) idx = maxLen - 1;
    
    // Only set hover if within draw area (roughly)
    if (x >= -10 && x <= drawWidth + 10) {
      setHoverIdx(idx);
    } else {
      setHoverIdx(null);
    }
  };

  const handleMouseLeave = () => setHoverIdx(null);

  // Hover values - only show overlays in header (tickers shown on Y-axis badges)
  const hoverDate = hoverIdx !== null && dates[hoverIdx] ? dates[hoverIdx] : null;
  const tickerLabels = ["TQQQ", "QQQ", "VOO"];
  const hoverValues = hoverIdx !== null 
    ? lines
        .filter(l => !tickerLabels.includes(l.label)) // Exclude tickers from header
        .map(l => ({ label: l.label, color: l.color, value: l.data[hoverIdx] }))
    : [];
  
  // Format hover date for display
  const formatHoverDate = (d: string | null) => {
    if (!d) return "";
    const date = new Date(d);
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  return (
    <div ref={containerRef} className="relative w-full" style={{ height: chartH_Dynamic }}>
      {/* Top Price Header (No date - Toss style) */}
      {hoverIdx !== null && (
        <div className="absolute top-0 left-0 right-20 h-7 px-2 flex items-center gap-3 text-xs z-20 pointer-events-none">
        {hoverValues.map((v, i) => (
            v.value !== null && v.value !== undefined && (
              <span key={i} className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: v.color }} />
                <span className="text-neutral-400 font-medium">{v.label}</span>
                <span className="font-mono font-bold" style={{ color: v.color }}>{v.value.toFixed(2)}</span>
              </span>
            )
          ))}
        </div>
      )}

      {/* Zoom Controls */}
      <div className="absolute top-1 right-1 flex items-center gap-1 z-10">
        <button
          onClick={handleZoomIn}
          className="w-6 h-6 rounded bg-surface/80 hover:bg-surface border border-border text-fg text-sm font-bold flex items-center justify-center transition-colors"
          title="확대 (Zoom In)"
        >
          +
        </button>
        <button
          onClick={handleZoomOut}
          className="w-6 h-6 rounded bg-surface/80 hover:bg-surface border border-border text-fg text-sm font-bold flex items-center justify-center transition-colors"
          title="축소 (Zoom Out)"
        >
          −
        </button>
        {zoomRange[0] !== 0 || zoomRange[1] !== 1 ? (
          <button
            onClick={handleDoubleClick}
            className="px-1.5 h-6 rounded bg-surface/80 hover:bg-surface border border-border text-muted text-[10px] flex items-center justify-center transition-colors"
            title="리셋 (Reset)"
          >
            Reset
          </button>
        ) : null}
      </div>
      
      <svg 
        ref={svgRef}
        width={chartW_Dynamic} 
        height={chartH_Dynamic} 
        viewBox={`0 0 ${chartW_Dynamic} ${chartH_Dynamic}`} 
        className={`overflow-visible ${isDragging ? 'cursor-grabbing' : zoomRange[0] !== 0 || zoomRange[1] !== 1 ? 'cursor-grab' : 'cursor-crosshair'}`}
        onMouseMove={(e) => { handleMouseMove(e); handleDrag(e); }}
        onMouseLeave={(e) => { handleMouseLeave(); handleMouseUp(); }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onDoubleClick={handleDoubleClick}
      >
        <defs>
          {/* Dynamic gradient based on first line color */}
          <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={lines[0]?.color || "#3b82f6"} stopOpacity="0.20" />
            <stop offset="100%" stopColor={lines[0]?.color || "#3b82f6"} stopOpacity="0" />
          </linearGradient>
          {/* ClipPath to prevent overflow */}
          <clipPath id="chartClip">
            <rect x={0} y={0} width={chartW} height={chartH} />
          </clipPath>
        </defs>
        <g transform={`translate(${padding.left},${padding.top})`} clipPath="url(#chartClip)">
          {/* Area Fill for the first line (Main Series) */}
          {lines.length > 0 && lines[0].data.length > 0 && !lines[0].dashed && (() => {
             const firstLine = lines[0];
             const areaPoints: string[] = [];
             let started = false;
             let firstX = 0;
             let lastX = 0;
             
             for (let i = 0; i < firstLine.data.length; i++) {
               const v = firstLine.data[i];
               if (v === null) continue;
               const x = xScale(i);
               const y = yScale(v);
               if (!started) {
                 areaPoints.push(`M${x.toFixed(1)},${y.toFixed(1)}`);
                 firstX = x;
                 started = true;
               } else {
                 areaPoints.push(`L${x.toFixed(1)},${y.toFixed(1)}`);
               }
               lastX = x;
             }
             
             if (started) {
               // Close the area path
               areaPoints.push(`L${lastX.toFixed(1)},${chartH}`); // Bottom Right
               areaPoints.push(`L${firstX.toFixed(1)},${chartH}`); // Bottom Left
               areaPoints.push("Z"); // Close
             }
             
             return (
               <path d={areaPoints.join(" ")} fill="url(#areaGradient)" stroke="none" />
             );
          })()}
        </g>
        
        {/* Axis labels (outside clipPath) */}
        <g transform={`translate(${padding.left},${padding.top})`}>
          {/* Grid lines (primary scale) */}
          {yTicks.map((tick, i) => (
            <line key={`grid-${i}`} x1={0} y1={yScale(tick)} x2={chartW} y2={yScale(tick)} stroke="#404040" strokeOpacity="0.08" strokeDasharray="3,3" />
          ))}
          
          {/* Y-axis tick labels - PRIMARY (right side for main ticker) */}
          {yTicks.map((tick, i) => {
            const y = yScale(tick);
            // Skip labels too close to bottom to avoid overlap
            if (y > chartH - 15) return null;
            const label = tick >= 1000 ? `${(tick / 1000).toFixed(0)}K` : tick.toFixed(tick >= 100 ? 0 : 1);
            return (
              <text key={`y-${i}`} x={chartW + 8} y={y} fill="#9ca3af" fontSize={10} textAnchor="start" dominantBaseline="middle" className="font-mono">
                {label}
              </text>
            );
          })}
          
          {/* Y-axis tick labels - SECONDARY (left side for TQQQ) */}
          {hasSecondary && yTicks2.map((tick, i) => {
            const y = yScale2(tick);
            // Skip labels too close to bottom to avoid overlap
            if (y > chartH - 15) return null;
            const label = tick >= 1000 ? `${(tick / 1000).toFixed(0)}K` : tick.toFixed(tick >= 100 ? 0 : 1);
            return (
              <text key={`y2-${i}`} x={-8} y={y} fill="#22c55e" fontSize={10} textAnchor="end" dominantBaseline="middle" className="font-mono">
                {label}
              </text>
            );
          })}
          
          {/* Current price label (red badge on right - Toss style) */}
          {currentPrice !== null && (
            <g>
              <rect 
                x={chartW + 2} 
                y={yScale(currentPrice) - 9} 
                width={50} 
                height={18} 
                fill="#ef4444" 
                rx={3}
              />
              <text x={chartW + 27} y={yScale(currentPrice)} fill="white" fontSize={10} fontWeight="bold" textAnchor="middle" dominantBaseline="middle" className="font-mono">
                {currentPrice >= 1000 ? `${(currentPrice / 1000).toFixed(1)}K` : currentPrice.toFixed(1)}
              </text>
            </g>
          )}
          
          {/* Min price point annotation (Adaptive positioning) */}
          {rawMin !== Infinity && rawMin !== rawMax && (
            (() => {
              const x = xScale(minIdx);
              const y = yScale(rawMin);
              const pctFromCurrent = currentPrice ? (((currentPrice - rawMin) / rawMin) * 100) : 0;
              const dateStr = dates[minIdx] ? formatHoverDate(dates[minIdx]) : "";
              
              // Collision detection
              const isNearLeft = x < 140; // Increased threshold for long text
              const isNearRight = x > chartW - 140;
              
              // Standard: Text to right. Near Right Edge: Text to left.
              const textAnchor = isNearRight ? "end" : "start";
              const textX = isNearRight ? x - 6 : x + 6;
              const textY = y + 16; // Pivot below dot

              return (
                <g>
                  <circle cx={x} cy={y} r={4} fill="#3b82f6" stroke="white" strokeWidth={1} />
                  <text 
                    x={textX} 
                    y={textY} 
                    fill="#3b82f6"
                    fontSize={11} 
                    fontWeight="bold" 
                    textAnchor={textAnchor}
                    className="font-mono"
                  >
                    {rawMin.toFixed(1)} ({pctFromCurrent >= 0 ? '+' : ''}{pctFromCurrent.toFixed(1)}%, {dateStr})
                  </text>
                </g>
              );
            })()
          )}
          
          {/* Max price point annotation (Adaptive positioning) */}
          {rawMax !== -Infinity && rawMin !== rawMax && (
            (() => {
              const x = xScale(maxIdx);
              const y = yScale(rawMax);
              const pctFromCurrent = currentPrice ? (((currentPrice - rawMax) / rawMax) * 100) : 0;
              const dateStr = dates[maxIdx] ? formatHoverDate(dates[maxIdx]) : "";
              
              // Collision detection
              const isNearLeft = x < 140;
              const isNearRight = x > chartW - 140;
              const isNearTop = y < 20;

              // If Near Left: Anchor Start (Right of dot). If Near Right: Anchor End (Left of dot).
              const textAnchor = isNearRight ? "end" : "start";
              const textX = isNearRight ? x - 6 : x + 6;
              const textY = isNearTop ? y + 16 : y - 6; // If too high, push text below

              return (
                <g>
                  <circle cx={x} cy={y} r={4} fill="#ef4444" stroke="white" strokeWidth={1} />
                  <text 
                    x={textX} 
                    y={textY} 
                    fill="#ef4444"
                    fontSize={11} 
                    fontWeight="bold" 
                    textAnchor={textAnchor}
                    className="font-mono"
                  >
                    {rawMax.toFixed(1)} ({pctFromCurrent >= 0 ? '+' : ''}{pctFromCurrent.toFixed(1)}%, {dateStr})
                  </text>
                </g>
              );
            })()
          )}
          
          {/* Hover price badges on Y-axis for each ticker line */}
          {hoverIdx !== null && lines.map((line, i) => {
            const hoverPrice = line.data[hoverIdx];
            if (hoverPrice === null) return null;
            
            // Only show badges for ticker lines (TQQQ, QQQ, VOO), not overlays
            const isTicker = ["TQQQ", "QQQ", "VOO"].includes(line.label);
            if (!isTicker) return null;
            
            // Use appropriate Y scale for secondary lines
            const yPos = line.isSecondary ? yScale2(hoverPrice) : yScale(hoverPrice);
            
            // Position: primary on right, secondary on left
            const xPos = line.isSecondary ? -55 : chartW + 2;
            const textX = line.isSecondary ? -30 : chartW + 27;
            
            return (
              <g key={`ybadge-${i}`}>
                <rect 
                  x={xPos} 
                  y={yPos - 9} 
                  width={50} 
                  height={18} 
                  fill={line.color} 
                  rx={3}
                />
                <text 
                  x={textX} 
                  y={yPos} 
                  fill="white" 
                  fontSize={10} 
                  fontWeight="500" 
                  textAnchor="middle" 
                  dominantBaseline="middle"
                  className="font-mono"
                >
                  {hoverPrice >= 1000 ? `${(hoverPrice / 1000).toFixed(1)}K` : hoverPrice.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* X-axis labels (skip if too close to right edge) */}
          {xTicks.map((t, i) => {
            const x = xScale(t.idx);
            // Skip labels too close to right edge
            if (x > chartW - 20) return null;
            return (
              <text key={i} x={x} y={chartH + 20} fill="#737373" fontSize={10} textAnchor="middle">
                {t.label}
              </text>
            );
          })}
          
          {/* X-axis hover date badge (Toss style - outside clipPath) */}
          {hoverIdx !== null && (
            <g>
              <rect 
                x={xScale(hoverIdx) - 40} 
                y={chartH + 10} 
                width={80} 
                height={20} 
                fill="#3b82f6" 
                rx={4}
              />
              <text 
                x={xScale(hoverIdx)} 
                y={chartH + 20} 
                fill="white" 
                fontSize={11} 
                fontWeight="600"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {dates[hoverIdx] ? formatHoverDate(dates[hoverIdx]) : ""}
              </text>
            </g>
          )}
        </g>
        
        {/* Chart content (inside clipPath) */}
        <g transform={`translate(${padding.left},${padding.top})`} clipPath="url(#chartClip)">

          {/* E03 Signal Zone Bands (background) */}
          {zones.map((zone, i) => {
            const x1 = xScale(zone.startIdx);
            const x2 = xScale(zone.endIdx);
            const color = zone.state === "ON" ? "#22c55e" : "#ef4444"; // Green for ON, Red for OFF
            return (
              <rect 
                key={`zone-${i}`}
                x={x1}
                y={0}
                width={x2 - x1}
                height={chartH}
                fill={color}
                fillOpacity={0.08}
              />
            );
          })}

          {/* Lines */}
          {paths.map((d, i) => (
            <path key={i} d={d} fill="none" stroke={lines[i].color} strokeWidth={lines[i].dashed ? 1 : 1.5}
              strokeDasharray={lines[i].dashed ? "3,3" : undefined} 
              strokeOpacity={lines[i].dashed ? 0.7 : 1}
            />
          ))}

          {/* E03 Signal Markers (▲BUY / ▼SELL) - Hidden in Story Mode by default */}
          {showTrades && markers.map((marker, i) => {
            const x = xScale(marker.idx);
            const y = yScale(marker.price);
            const isBuy = marker.type === "BUY";
            const color = isBuy ? "#22c55e" : "#ef4444";
            // Triangle pointing up (BUY) or down (SELL)
            const size = 8;
            const path = isBuy 
              ? `M${x},${y - size} L${x - size/1.5},${y + size/2} L${x + size/1.5},${y + size/2} Z`  // Up
              : `M${x},${y + size} L${x - size/1.5},${y - size/2} L${x + size/1.5},${y - size/2} Z`; // Down
            return (
              <g key={`marker-${i}`}>
                <path d={path} fill={color} stroke="white" strokeWidth={0.5} />
                <text 
                  x={x} 
                  y={isBuy ? y - size - 4 : y + size + 10} 
                  fill={color} 
                  fontSize={8} 
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {isBuy ? "BUY" : "SELL"}
                </text>
              </g>
            );
          })}

          {/* V/E Day Markers (Verdict / Exec vertical lines) */}
          {veMarkers.map((ve, i) => {
            const x = xScale(ve.idx);
            const isVerdict = ve.type === "V";
            const color = isVerdict ? "#8b5cf6" : "#f59e0b"; // purple for V, amber for E
            return (
              <g key={`ve-${i}`}>
                <line 
                  x1={x} y1={0} x2={x} y2={chartH}
                  stroke={color} strokeWidth={1} strokeDasharray="4,2" strokeOpacity={0.7}
                />
                <text
                  x={x + 4} y={12}
                  fill={color} fontSize={10} fontWeight="bold"
                >
                  {ve.type}
                </text>
              </g>
            );
          })}

          {/* Hover crosshair and X-axis date badge */}
          {hoverIdx !== null && (
            <>
              {/* Vertical crosshair */}
              <line x1={xScale(hoverIdx)} y1={0} x2={xScale(hoverIdx)} y2={chartH}
                stroke="#3b82f6" strokeWidth={1} strokeDasharray="4,2" strokeOpacity={0.7} />
              {/* Horizontal crosshairs for each ticker line */}
              {lines.map((line, i) => {
                const val = line.data[hoverIdx];
                if (val === null) return null;
                
                // Only show crosshairs for ticker lines
                const isTicker = ["TQQQ", "QQQ", "VOO"].includes(line.label);
                if (!isTicker) return null;
                
                const yPos = line.isSecondary ? yScale2(val) : yScale(val);
                
                return (
                  <line 
                    key={`hcross-${i}`}
                    x1={0} 
                    y1={yPos} 
                    x2={chartW} 
                    y2={yPos}
                    stroke={line.color} 
                    strokeWidth={1} 
                    strokeDasharray="4,2" 
                    strokeOpacity={0.5} 
                  />
                );
              })}
            </>
          )}

          {/* Hover dots */}
          {hoverIdx !== null && lines.map((l, i) => {
            const v = l.data[hoverIdx];
            if (v === null) return null;
            return <circle key={i} cx={xScale(hoverIdx)} cy={yScale(v)} r={4} fill={l.color} />;
          })}
        </g>
        
        {/* Strategy Strips (Story Mode) - positioned at very bottom below x-axis labels */}
        {showStrips && (e03Segments.length > 0 || tqqSegments.length > 0) && (
          <g transform={`translate(${padding.left}, ${padding.top + chartH + 24})`}>
            {/* E03 Target Strip */}
            {e03Segments.length > 0 && (
              <g transform="translate(0, 0)">
                <StrategyStrip
                  type="e03"
                  segments={e03Segments}
                  xScale={(idx) => {
                    // Convert data index to visible x position
                    if (idx <= visibleStartIdx) return 0;
                    if (idx >= visibleEndIdx) return chartW;
                    return ((idx - visibleStartIdx) / (visibleEndIdx - visibleStartIdx)) * chartW;
                  }}
                  height={10}
                  chartWidth={chartW}
                  label="E03"
                />
              </g>
            )}
            {/* 200TQ State Strip */}
            {tqqSegments.length > 0 && (
              <g transform="translate(0, 12)">
                <StrategyStrip
                  type="200tq"
                  segments={tqqSegments}
                  xScale={(idx) => {
                    if (idx <= visibleStartIdx) return 0;
                    if (idx >= visibleEndIdx) return chartW;
                    return ((idx - visibleStartIdx) / (visibleEndIdx - visibleStartIdx)) * chartW;
                  }}
                  height={10}
                  chartWidth={chartW}
                  label="200TQ"
                />
              </g>
            )}
          </g>
        )}
      </svg>
    </div>
  );
}
