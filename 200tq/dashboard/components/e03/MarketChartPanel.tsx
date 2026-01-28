"use client";

import { useState, useMemo, useEffect } from "react";
import SvgLineChart, { VEMarker } from "./SvgLineChart";
import { MoreHorizontal, Loader2 } from "lucide-react";
import { getMockSeries, filterByPeriod, getClosePrices, Period, Ticker } from "../../lib/ops/e03/mockPrices";
import { usePriceData, getClosesFromSeries, type PricePoint, type RealTicker } from "../../lib/ops/e03/usePriceData";
import { 
  sma, envelope, normalizeSeries, get200TQState, getE03Signal,
  computeE03SignalArray, shiftSignalToTarget, compute200TQStateArray, runLengthEncode
} from "../../lib/ops/e03/indicators";
import { summarizePerfWithData, type PriceData } from "../../lib/ops/e03/performance";

interface MarketChartPanelProps {
  selectedPeriod: Period;
  onPeriodChange: (p: Period) => void;
  e03State: "ON" | "OFF10";
  privacyMode: boolean;
  startCapital?: number;
}

const PERIODS: Period[] = ["3M", "6M", "1Y", "3Y", "5Y"];
const TICKERS: Ticker[] = ["TQQQ", "QQQ"];

const OVERLAY_CONFIGS = [
  { key: "sma3", label: "SMA3", defaultOn: false },
  { key: "sma160", label: "SMA160", defaultOn: false },
  { key: "sma165", label: "SMA165", defaultOn: false },
  { key: "sma170", label: "SMA170", defaultOn: false },
  { key: "sma200", label: "SMA200", defaultOn: true },
  { key: "env5", label: "+5%", defaultOn: true },
];

const COLORS: Record<string, string> = {
  TQQQ: "#3b82f6", // Blue-500 (Primary)
  QQQ: "#9ca3af",  // Neutral-400
  sma3: "#facc15",
  sma160: "#f97316",
  sma165: "#ef4444",
  sma170: "#ec4899",
  sma200: "#06b6d4",
  env5: "#06b6d4",
};

export default function MarketChartPanel({
  selectedPeriod,
  onPeriodChange,
  e03State,
  privacyMode,
  startCapital = 10000000,
}: MarketChartPanelProps) {
  // --- Real Data Integration ---
  const { series: priceData, loading: priceLoading } = usePriceData();
  
  // Helper to get series (uses hook data)
  const getSeries = (ticker: Ticker): PricePoint[] => {
    return priceData[ticker as RealTicker] || [];
  };
  
  // Helper to get close prices array
  const getCloses = (ticker: Ticker): number[] => {
    return getSeries(ticker).map(p => p.close);
  };
  
  // Ticker toggles - Default: E03 preset (QQQ + TQQQ)
  const [activeTickers, setActiveTickers] = useState<Set<Ticker>>(new Set<Ticker>(["QQQ", "TQQQ"]));
  // Overlay toggles - Default: E03 overlays
  const [activeOverlays, setActiveOverlays] = useState<Set<string>>(
    new Set(["sma3", "sma160", "sma165", "sma170"])
  );
  // Focus ticker for MA calculations - Default: QQQ (E03 signal source)
  const [focusTicker, setFocusTicker] = useState<Ticker>("QQQ");
  
  // Strategy Mode State - Default: E03
  type StrategyMode = "200TQ" | "E03";
  const [strategyMode, setStrategyMode] = useState<StrategyMode>("E03");

  // Handle strategy change with presets
  const handleStrategyChange = (mode: StrategyMode) => {
    setStrategyMode(mode);
    if (mode === "200TQ") {
      // 200TQ Preset: TQQQ only, SMA200, +5%
      setActiveTickers(new Set<Ticker>(["TQQQ"]));
      setFocusTicker("TQQQ");
      setActiveOverlays(new Set(["sma200", "env5"]));
    } else {
      // E03 Preset: QQQ+TQQQ, SMA3, SMA160, 165, 170
      setActiveTickers(new Set<Ticker>(["QQQ", "TQQQ"]));
      setFocusTicker("QQQ");
      setActiveOverlays(new Set(["sma3", "sma160", "sma165", "sma170"]));
    }
  };

  // Fixed green color for TQQQ (Market style)
  const currentColors = useMemo(() => ({
    ...COLORS,
    TQQQ: "#22c55e", // Green (Market)
  } as Record<string, string>), []);
  
  // Story Mode toggles
  const [showTrades, setShowTrades] = useState(true); // BUY/SELL shown by default
  const [showStrips, setShowStrips] = useState(false);  // Strategy strips hidden by default

  // Auto-sync focus ticker when single ticker is selected
  useEffect(() => {
    if (activeTickers.size === 1) {
      const singleTicker = Array.from(activeTickers)[0];
      setFocusTicker(singleTicker);
    }
  }, [activeTickers]);

  const toggleTicker = (t: Ticker) => {
    const next = new Set(activeTickers);
    if (next.has(t)) {
      next.delete(t);
      if (next.size === 0) next.add(t); // At least one
    } else {
      next.add(t);
    }
    setActiveTickers(next);
  };

  const toggleOverlay = (key: string) => {
    const next = new Set(activeOverlays);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    setActiveOverlays(next);
  };

  // Compute chart data
  const chartData = useMemo(() => {
    const focusSeries = getSeries(focusTicker);
    const filtered = filterByPeriod(focusSeries, selectedPeriod);
    const startIdx = focusSeries.length - filtered.length;
    
    const focusPrices = getCloses(focusTicker).slice(startIdx);
    
    // Always normalize when multiple tickers (Index=100 style)
    const shouldNormalize = activeTickers.size > 1;

    const lines: { data: (number | null)[]; color: string; label: string; dashed?: boolean; isSecondary?: boolean }[] = [];

    // Ticker price lines
    for (const ticker of TICKERS) {
      if (!activeTickers.has(ticker)) continue;
      const series = getSeries(ticker);
      const tickerFiltered = filterByPeriod(series, selectedPeriod);
      let prices = tickerFiltered.map(p => p.close);
      
      // Normalize when multiple tickers shown
      if (shouldNormalize) {
        prices = normalizeSeries(prices) as number[];
      }
      
      lines.push({
        data: prices,
        color: currentColors[ticker],
        label: ticker,
      });
    }

    // Overlays (based on focus ticker)
    const fullPrices = getCloses(focusTicker);
    const sma3Full = sma(fullPrices, 3);
    const sma160Full = sma(fullPrices, 160);
    const sma165Full = sma(fullPrices, 165);
    const sma170Full = sma(fullPrices, 170);
    const sma200Full = sma(fullPrices, 200);
    const env5Full = envelope(sma200Full, 1.05);

    const sliceOverlay = (arr: (number | null)[]) => arr.slice(startIdx);
    
    // Get base price for consistent normalization (first price of focus ticker in the period)
    const focusBasePrice = focusPrices.find(p => p !== null && p !== undefined) ?? 100;

    if (activeOverlays.has("sma3")) {
      let data = sliceOverlay(sma3Full);
      if (shouldNormalize) data = normalizeSeries(data, undefined, focusBasePrice) as (number | null)[];
      lines.push({ data, color: currentColors.sma3, label: "SMA3" });
    }
    if (activeOverlays.has("sma160")) {
      let data = sliceOverlay(sma160Full);
      if (shouldNormalize) data = normalizeSeries(data, undefined, focusBasePrice) as (number | null)[];
      lines.push({ data, color: currentColors.sma160, label: "SMA160" });
    }
    if (activeOverlays.has("sma165")) {
      let data = sliceOverlay(sma165Full);
      if (shouldNormalize) data = normalizeSeries(data, undefined, focusBasePrice) as (number | null)[];
      lines.push({ data, color: currentColors.sma165, label: "SMA165" });
    }
    if (activeOverlays.has("sma170")) {
      let data = sliceOverlay(sma170Full);
      if (shouldNormalize) data = normalizeSeries(data, undefined, focusBasePrice) as (number | null)[];
      lines.push({ data, color: currentColors.sma170, label: "SMA170" });
    }
    if (activeOverlays.has("sma200")) {
      let data = sliceOverlay(sma200Full);
      if (shouldNormalize) data = normalizeSeries(data, undefined, focusBasePrice) as (number | null)[];
      lines.push({ data, color: currentColors.sma200, label: "SMA200" });
    }
    if (activeOverlays.has("env5")) {
      let data = sliceOverlay(env5Full);
      if (shouldNormalize) data = normalizeSeries(data, undefined, focusBasePrice) as (number | null)[];
      lines.push({ data, color: currentColors.env5, label: "+5%", dashed: true });
    }

    // Current 200TQ state
    const lastIdx = fullPrices.length - 1;
    const currentClose = fullPrices[lastIdx];
    const currentSma200 = sma200Full[lastIdx];
    const currentEnv5 = env5Full[lastIdx];
    const tq200State = get200TQState(currentClose, currentSma200, currentEnv5);
    const margin200 = currentSma200 ? ((currentClose - currentSma200) / currentSma200 * 100).toFixed(1) : "-";

    // Get dates for X-axis
    const dates = filtered.map(p => p.date);

    // === Signal Computation based on Strategy Mode ===
    
    // Common types
    type SignalMarker = { idx: number; type: "BUY" | "SELL"; price: number };
    type SignalZone = { startIdx: number; endIdx: number; state: "ON" | "OFF10" };
    
    const markers: SignalMarker[] = [];
    const zones: SignalZone[] = [];

    if (strategyMode === "E03") {
      // === E03 Strategy Logic ===
      // E03 uses QQQ for signal generation per SSOT
      const qqqPricesFull = getCloses("QQQ");
      const qqqSma3 = sma(qqqPricesFull, 3);
      const qqqSma160 = sma(qqqPricesFull, 160);
      const qqqSma165 = sma(qqqPricesFull, 165);
      const qqqSma170 = sma(qqqPricesFull, 170);
      
      // Track signal state changes
      let prevSignal: "ON" | "OFF10" | null = null;
      let zoneStartIdx = 0;
      
      for (let i = startIdx; i < qqqPricesFull.length; i++) {
        const chartIdx = i - startIdx;
        const signal = getE03Signal(
          qqqSma3[i],
          qqqSma160[i],
          qqqSma165[i],
          qqqSma170[i]
        );
        
        // Detect signal transitions
        if (prevSignal !== null && prevSignal !== signal) {
          zones.push({ startIdx: zoneStartIdx, endIdx: chartIdx, state: prevSignal });
          zoneStartIdx = chartIdx;
          
          // Marker
          const rawPrice = focusPrices[chartIdx];
          if (rawPrice !== undefined && rawPrice !== null) {
            let markerPrice = rawPrice;
            if (shouldNormalize) {
              let basePrice: number | null = null;
              for (const p of focusPrices) {
                if (p !== null && p !== undefined) { basePrice = p; break; }
              }
              if (basePrice && basePrice !== 0) {
                markerPrice = (rawPrice / basePrice) * 100;
              }
            }
            markers.push({
              idx: chartIdx,
              type: signal === "ON" ? "BUY" : "SELL",
              price: markerPrice,
            });
          }
        }
        prevSignal = signal;
      }
      
      if (prevSignal !== null) {
        zones.push({ startIdx: zoneStartIdx, endIdx: dates.length - 1, state: prevSignal });
      }

    } else {
      // === 200TQ Strategy Logic ===
      // Uses TQQQ 200SMA state (Price vs 200SMA)
      // Zone ON: Price >= 200SMA, Zone OFF: Price < 200SMA (simplified for visualization)
      const tqqqPricesFull = getCloses("TQQQ");
      const tqqqSma200 = sma(tqqqPricesFull, 200);
      
      let prevIsBull: boolean | null = null; // bullish (above 200)
      let zoneStartIdx = 0;

      for (let i = startIdx; i < tqqqPricesFull.length; i++) {
        const chartIdx = i - startIdx;
        const price = tqqqPricesFull[i];
        const ma = tqqqSma200[i];
        
        if (price === null || ma === null) continue;
        
        const isBull = price >= ma;
        
        if (prevIsBull !== null && prevIsBull !== isBull) {
          // Close previous zone
          // If prev was bull (true) -> ON zone. If bear (false) -> OFF zone.
          zones.push({ startIdx: zoneStartIdx, endIdx: chartIdx, state: prevIsBull ? "ON" : "OFF10" });
          zoneStartIdx = chartIdx;
          
          // Marker
          // Bull -> Bear : SELL
          // Bear -> Bull : BUY
          const rawPrice = focusPrices[chartIdx]; 
          // Note: In 200TQ mode, focus is TQQQ, so rawPrice is TQQQ price.
          
          if (rawPrice !== undefined && rawPrice !== null) {
             markers.push({
               idx: chartIdx,
               type: isBull ? "BUY" : "SELL",
               price: rawPrice, // No normalization in 200TQ mode (single ticker)
             });
          }
        }
        prevIsBull = isBull;
      }
      
      if (prevIsBull !== null) {
        zones.push({ startIdx: zoneStartIdx, endIdx: dates.length - 1, state: prevIsBull ? "ON" : "OFF10" });
      }
    }

    // === Story Mode: Segment computation ===
    // Slice SMAs to match chart range (startIdx onwards)
    const sma3Chart = sma3Full.slice(startIdx);
    const sma160Chart = sma160Full.slice(startIdx);
    const sma165Chart = sma165Full.slice(startIdx);
    const sma170Chart = sma170Full.slice(startIdx);
    const sma200Chart = sma200Full.slice(startIdx);
    
    // E03 Signal → Target (t+1 shift) → RLE segments for strip
    const e03Signal = computeE03SignalArray(sma3Chart, sma160Chart, sma165Chart, sma170Chart);
    const e03Target = shiftSignalToTarget(e03Signal, 1, "OFF");
    const e03Segments = runLengthEncode(e03Target);
    
    // 200TQ State → RLE segments for strip (uses TQQQ prices)
    const tqqqPricesFull = getCloses("TQQQ");
    const tqqqSma200Full = sma(tqqqPricesFull, 200);
    const tqqqPricesChart = tqqqPricesFull.slice(startIdx) as (number | null)[];
    const tqqqSma200Chart = tqqqSma200Full.slice(startIdx);
    const tqStateArray = compute200TQStateArray(tqqqPricesChart, tqqqSma200Chart);
    const tqqSegments = runLengthEncode(tqStateArray);
    
    // V/E markers (Verdict = today, Exec = tomorrow in trading days)
    // For now, mark the last day as E (exec, assuming t+1 applied), second-to-last as V
    const veMarkers: VEMarker[] = [];
    if (dates.length >= 2) {
      veMarkers.push({ idx: dates.length - 2, type: "V" }); // Verdict (t)
      veMarkers.push({ idx: dates.length - 1, type: "E" }); // Exec (t+1)
    }

    return { 
      lines, dates, tq200State, currentClose, currentSma200, margin200, markers, zones,
      e03Segments, tqqSegments, veMarkers, e03Signal, e03Target
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTickers, activeOverlays, focusTicker, selectedPeriod, currentColors, strategyMode, priceData]);



  // Build PriceData from usePriceData hook
  const perfPriceData: PriceData = useMemo(() => ({
    QQQ: priceData.QQQ,
    TQQQ: priceData.TQQQ,
    SGOV: priceData.SGOV,
    SPLG: priceData.SPLG,
  }), [priceData]);
  
  // Performance summary using real/mock data from usePriceData
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const perf = useMemo(() => summarizePerfWithData(selectedPeriod, startCapital, perfPriceData) as any, [selectedPeriod, startCapital, perfPriceData]);

  const formatKrw = (val: number) => {
    if (privacyMode) return "***";
    const abs = Math.abs(val);
    if (abs >= 100000000) return `${(val / 100000000).toFixed(1)}억`;
    if (abs >= 10000) return `${(val / 10000).toFixed(0)}만`;
    return val.toLocaleString();
  };

  // Loading state
  if (priceLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-center h-64 bg-surface rounded-xl border border-neutral-800">
          <Loader2 className="animate-spin text-neutral-500" size={32} />
          <span className="ml-2 text-neutral-500 text-sm">Loading chart data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header: State Badges + Trades Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2 min-w-0">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <h4 className="text-xs sm:text-sm font-semibold text-fg whitespace-nowrap">Market Charts</h4>
          {/* Strategy Mode Selector */}
          <div className="flex bg-surface dark:bg-neutral-900 rounded-lg p-0.5 border border-border shrink-0">
            <button
               onClick={() => handleStrategyChange("200TQ")}
               className={`px-2 sm:px-3 py-1 text-[10px] font-bold rounded-md transition-all ${
                 strategyMode === "200TQ"
                   ? "bg-blue-600 text-white shadow-sm"
                   : "text-neutral-500 hover:text-neutral-300"
               }`}
            >
              200TQ
            </button>
            <button
               onClick={() => handleStrategyChange("E03")}
               className={`px-2 sm:px-3 py-1 text-[10px] font-bold rounded-md transition-all ${
                 strategyMode === "E03"
                   ? "bg-purple-600 text-white shadow-sm"
                   : "text-neutral-500 hover:text-neutral-300"
               }`}
            >
              E03
            </button>
          </div>
        </div>
        <div className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs overflow-x-auto">
          <span className={`px-1.5 sm:px-2 py-0.5 rounded whitespace-nowrap ${e03State === "ON" ? "bg-green-900/40 text-green-400" : "bg-amber-900/40 text-amber-400"}`}>
            E03: {e03State}
          </span>
          <span className={`px-1.5 sm:px-2 py-0.5 rounded font-sans whitespace-nowrap ${
            chartData.tq200State === "하락" ? "bg-red-900/60 text-red-200" :
            chartData.tq200State === "과열" ? "bg-orange-900/40 text-orange-300" :
            "bg-blue-900/40 text-blue-300"
          }`}>
            200TQ: {chartData.tq200State}
            <span className="opacity-70 ml-1 text-[9px] hidden sm:inline">
              {chartData.tq200State === "하락" ? "(<200MA)" : chartData.tq200State === "과열" ? "(>+5%)" : "(200MA~+5%)"}
            </span>
          </span>
          {/* Trades toggle (Advanced) */}
          <button
            onClick={() => setShowTrades(!showTrades)}
            className={`px-1.5 sm:px-2 py-0.5 rounded text-[10px] font-medium transition-colors whitespace-nowrap ${
              showTrades 
                ? "bg-purple-900/40 text-purple-300" 
                : "bg-neutral-800/60 text-neutral-500 hover:text-neutral-400"
            }`}
          >
            Trades {showTrades ? "ON" : "OFF"}
          </button>
        </div>
      </div>



      {/* Controls Container using Segmented Groups */}
      <div className="flex flex-col gap-3 pb-2 border-b border-border mb-4">
        <div className="flex flex-wrap items-center justify-between gap-y-3">
          
          {/* Group 1: Tickers */}
          <div className="flex items-center gap-2">
            <div className="flex p-0.5 bg-inset rounded-lg">
              {TICKERS.map(t => (
                <button
                  key={t}
                  onClick={() => toggleTicker(t)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                    activeTickers.has(t)
                      ? "bg-surface dark:bg-neutral-700 text-fg shadow-sm"
                      : "text-muted hover:text-fg"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            {activeTickers.size > 1 && (
               <span className="text-[9px] text-cyan-400 bg-cyan-950/30 px-1.5 py-0.5 rounded border border-cyan-700">
                 Index=100
               </span>
            )}
          </div>

          {/* Group 2: Period */}
          <div className="flex p-0.5 bg-inset rounded-lg">
            {PERIODS.map(p => (
              <button
                key={p}
                onClick={() => onPeriodChange(p)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                  selectedPeriod === p
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-y-2">
          {/* Group 3: Indicators (All visible) */}
          <div className="flex flex-wrap items-center gap-1.5 flex-1">
             <span className="text-[10px] text-muted uppercase tracking-wider font-semibold mr-1">Overlay</span>
             {OVERLAY_CONFIGS.map(o => (
                <button
                  key={o.key}
                  onClick={() => toggleOverlay(o.key)}
                  className={`px-2 py-0.5 text-[10px] rounded border transition-all ${
                    activeOverlays.has(o.key)
                      ? "bg-surface dark:bg-neutral-200 border-border text-fg font-bold shadow-sm"
                      : "bg-transparent border-transparent text-muted hover:bg-surface/50"
                  }`}
                  style={{ borderColor: activeOverlays.has(o.key) ? COLORS[o.key] : undefined, color: activeOverlays.has(o.key) ? COLORS[o.key] : undefined }}
                >
                  {o.label}
                </button>
             ))}
          </div>

          {/* Group 4: MA Basis */}
          <div className="flex items-center gap-2">
             <span className="text-[10px] text-muted">MA 기준</span>
              <select 
                value={focusTicker}
                onChange={e => setFocusTicker(e.target.value as Ticker)}
                className="bg-inset rounded px-2 py-1 text-fg font-medium text-[10px] focus:outline-none focus:ring-1 focus:ring-neutral-500 cursor-pointer"
              >
                {TICKERS.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-surface rounded-xl p-3 shadow-sm">
        <SvgLineChart 
          lines={chartData.lines} 
          dates={chartData.dates} 
          width={700} 
          height={407}  /* Increased height (1.1x of 370) */
          markers={chartData.markers}
          zones={chartData.zones}
          e03Segments={chartData.e03Segments}
          tqqSegments={chartData.tqqSegments}
          veMarkers={chartData.veMarkers}
          showTrades={showTrades}
          showStrips={showStrips}
        />
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-surface p-3 rounded-xl shadow-sm">
          <div className="text-muted mb-1">Focus: {focusTicker}</div>
          <div className="text-fg">
            Close: <span className="font-sans tabular-nums">${chartData.currentClose?.toFixed(2)}</span>
          </div>
          <div className="text-fg">
            SMA200: <span className="font-sans tabular-nums">${chartData.currentSma200?.toFixed(2) ?? "-"}</span>
          </div>
          <div className="text-fg">
            Margin: <span className={`font-mono tabular-nums ${Number(chartData.margin200) > 0 ? "text-positive" : "text-negative"}`}>
              {chartData.margin200}%
            </span>
          </div>
        </div>
        <div className="bg-surface p-3 rounded-xl shadow-sm">
          <div className="text-muted mb-1">Performance ({selectedPeriod})</div>
          <div className={perf.tq200.returnPct >= 0 ? "text-positive" : "text-negative"}>
            200TQ: <span className="font-mono">{perf.tq200.returnPct > 0 ? "+" : ""}{perf.tq200.returnPct.toFixed(1)}%</span>
            <span className="text-muted ml-1">({formatKrw(perf.tq200.pnl)})</span>
          </div>
          <div className={perf.e03.returnPct >= 0 ? "text-positive" : "text-negative"}>
            E03: <span className="font-mono">{perf.e03.returnPct > 0 ? "+" : ""}{perf.e03.returnPct.toFixed(1)}%</span>
            <span className="text-muted ml-1">({formatKrw(perf.e03.pnl)})</span>
          </div>
        </div>
      </div>
    </div>
  );
}
