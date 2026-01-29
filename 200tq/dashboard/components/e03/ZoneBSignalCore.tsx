import { E03ViewModel, EvidenceItem } from "../../lib/ops/e03/types";
import { CheckCircle2, XCircle, AlertTriangle, TrendingUp, Wallet, Settings, ChevronDown } from "lucide-react";
import { Period } from "../../lib/ops/e03/mockPrices";
import { PerfSummary } from "../../lib/ops/e03/performance";
import { useState, useMemo, useEffect } from "react";
import { getClosePrices } from "../../lib/ops/e03/mockPrices";
import { sma } from "../../lib/ops/e03/indicators";
import MacroStrip from "./MacroStrip";

type ColorTone = 'ok' | 'action' | 'danger';

interface MacroData {
  vix: { value: number | null; color: ColorTone };
  fng: { value: number | null; label: string; color: ColorTone };
  treasury: { value: number | null };
  dxy: { value: number | null };
  nq: { value: number | null };
  usdkrw: { value: number | null };
  updatedAt: string;
}

interface ZoneBSignalCoreProps {
  vm: E03ViewModel;
  selectedPeriod: Period;
  perfSummary: PerfSummary | null;
  startCapital?: number;
  onCapitalChange?: (capital: number) => void;
  realPrices?: Record<string, number>; // Real API prices (TQQQ, QQQ, etc.)
  onDateRangeChange?: (start: string, end: string) => void;
}

function EvidenceCard({ item }: { item: EvidenceItem }) {
  return (
    <div className={`flex flex-col items-center justify-center p-4 rounded-xl shadow-sm transition-all ${
      item.pass 
        ? "bg-surface text-fg ring-1 ring-inset/50" 
        : "bg-surface text-muted opacity-60"
    }`}>
      <div className="mb-2">
        {item.pass ? <CheckCircle2 size={24} className="text-emerald-500" /> : <XCircle size={24} />}
      </div>
      <div className="text-sm font-semibold mb-1 uppercase tracking-wider">{item.label}</div>
      <div className={`text-xs font-mono font-medium ${item.marginPct > 0 ? "text-positive" : "text-negative"}`}>
        {item.marginPct > 0 ? "+" : ""}{item.marginPct.toFixed(2)}%
      </div>
    </div>
  );
}

function formatKrw(val: number, privacyMode: boolean): string {
  if (privacyMode) return "***";
  const sign = val >= 0 ? "+" : "";
  const abs = Math.abs(val);
  if (abs >= 100000000) return `${sign}${(val / 100000000).toFixed(1)}억`;
  if (abs >= 10000) {
    const man = Math.round(val / 10000);
    return `${man >= 0 ? "+" : ""}${man.toLocaleString()}만`;
  }
  return `${sign}${val.toLocaleString()}`;
}

// Calculate date range for period
function getDateRange(period: Period): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  
  const periodDays: Record<Period, number> = {
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "3Y": 1095,
    "5Y": 1825,
  };
  
  start.setDate(start.getDate() - periodDays[period]);
  
  const format = (d: Date) => d.toISOString().slice(0, 10);
  return { start: format(start), end: format(end) };
}

// Calculate dynamic period label from date range
function getPeriodLabel(startDate: string, endDate: string): string {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const diffMs = end.getTime() - start.getTime();
  const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));
  
  if (diffDays <= 0) return "0D";
  if (diffDays <= 90) return `${diffDays}D`;
  if (diffDays <= 365) {
    const months = Math.round(diffDays / 30);
    return `${months}M`;
  }
  const years = (diffDays / 365).toFixed(1);
  // Remove trailing .0
  return years.endsWith('.0') ? `${Math.round(diffDays / 365)}Y` : `${years}Y`;
}

export default function ZoneBSignalCore({ 
  vm, 
  selectedPeriod, 
  perfSummary, 
  startCapital = 10000000,
  onCapitalChange,
  realPrices,
  onDateRangeChange
}: ZoneBSignalCoreProps) {
  const isOff = vm.strategyState === "OFF10";
  const [showSettings, setShowSettings] = useState(false);
  const [includeFees, setIncludeFees] = useState(false);
  const [localCapital, setLocalCapital] = useState(startCapital / 10000); // 만원 단위
  
  // Custom date range state
  const defaultDateRange = getDateRange(selectedPeriod);
  const [localStartDate, setLocalStartDate] = useState(defaultDateRange.start);
  const [localEndDate, setLocalEndDate] = useState(defaultDateRange.end);
  
  // Action guidance based on strategy state
  const actionLabel = isOff ? "매도 대기" : "매수 유지";
  const stateLabel = isOff ? "OFF" : "ON";
  
  // Use local dates for display
  const dateRange = { start: localStartDate, end: localEndDate };
  
  const handleCapitalChange = (value: string) => {
    const num = parseInt(value, 10);
    if (!isNaN(num) && num > 0) {
      setLocalCapital(num);
      onCapitalChange?.(num * 10000);
    }
  };
  
  const handleDateChange = (type: 'start' | 'end', value: string) => {
    if (type === 'start') {
      setLocalStartDate(value);
      onDateRangeChange?.(value, localEndDate);
    } else {
      setLocalEndDate(value);
      onDateRangeChange?.(localStartDate, value);
    }
  };
  
  const [macroData, setMacroData] = useState<MacroData | null>(null);
  const [macroLoading, setMacroLoading] = useState(true);
  
  useEffect(() => {
    fetch('/api/macro')
      .then(res => res.ok ? res.json() : Promise.reject('API Error'))
      .then(data => setMacroData(data))
      .catch(err => {
        console.error('Macro fetch failed:', err);
        setMacroData(null);
      })
      .finally(() => setMacroLoading(false));
  }, []);
  


  // Risk Calculation (SSOT: TQQQ -20% from entry price)
  // Uses realPrices if available (REAL mode), otherwise falls back to mock prices
  const riskMetrics = useMemo(() => {
    // Get prices: prefer realPrices (API) over mock data
    let currentPrice: number;
    let qqqCurrent: number;
    let qqqPrev: number;
    
    if (realPrices && realPrices.TQQQ && realPrices.QQQ) {
      // REAL mode: use API prices
      currentPrice = realPrices.TQQQ;
      qqqCurrent = realPrices.QQQ;
      // For QQQ prev close, we'd need historical data - estimate ~0% change for now
      // In production, this should come from a separate API call or stored prev close
      qqqPrev = realPrices.QQQ_PREV || realPrices.QQQ; // Fallback to same (0% change)
    } else {
      // MOCK mode: use getClosePrices
      const tqqqFull = getClosePrices("TQQQ");
      const qqqFull = getClosePrices("QQQ");
      currentPrice = tqqqFull[tqqqFull.length - 1];
      qqqCurrent = qqqFull[qqqFull.length - 1];
      qqqPrev = qqqFull[qqqFull.length - 2];
    }
    
    // Entry price: In REAL mode, should come from user's actual entry log
    // For now, use current price as entry (showing 0% from entry for safety)
    const entryPrice = realPrices?.TQQQ_ENTRY || currentPrice;
    
    // Stop loss line per SSOT: -20% from entry
    const stopLossPrice = entryPrice * 0.80;
    
    // Distance to stop loss (positive = safe buffer, negative = breached)
    const distToStop = currentPrice && entryPrice 
      ? ((currentPrice - stopLossPrice) / currentPrice) * 100 
      : 0;
    
    // Warning threshold: within 5% of stop loss (i.e., down 15-20% from entry)
    const isWarning = distToStop > 0 && distToStop < 5;
    const isTriggered = distToStop <= 0;
    
    // QQQ Drop Percentage (Negative value means drop)
    const qqqDropPct = (qqqCurrent && qqqPrev) 
      ? ((qqqCurrent - qqqPrev) / qqqPrev) * 100 
      : 0;
      
    // QQQ Trigger: Drop > 7% (i.e., less than -7%)
    const isQqqTriggered = qqqDropPct <= -7;
    const isQqqWarning = qqqDropPct <= -5 && qqqDropPct > -7; // Warn at -5%
    
    return { 
      currentPrice, 
      entryPrice,
      stopLossPrice, 
      distToStop,
      isWarning,
      isTriggered,
      qqqDropPct,
      isQqqTriggered,
      isQqqWarning,
      isAnyTriggered: isTriggered || isQqqTriggered
    };
  }, [selectedPeriod, realPrices]);

  return (
    <section className="space-y-4">
      {/* Risk Monitor Banner (High Priority) */}
      <div className={`rounded-xl border px-4 py-3 transition-all shadow-sm ${
        riskMetrics.isAnyTriggered 
          ? "bg-red-950 border-red-500 animate-pulse text-white shadow-lg shadow-red-900/40" // Emergency/Exit Now
          : riskMetrics.isWarning || riskMetrics.isQqqWarning
            ? "bg-amber-950/40 border-amber-500/50" // Caution (subtle)
            : "bg-surface border-border"
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
             <div className={`p-2 rounded-full ${
               riskMetrics.isAnyTriggered ? "bg-red-500/30 text-red-100" : 
               riskMetrics.isWarning || riskMetrics.isQqqWarning ? "bg-amber-500/20 text-amber-400" : "bg-inset text-neutral-400"
             }`}>
               <AlertTriangle size={18} className={riskMetrics.isAnyTriggered ? "animate-pulse" : ""} />
             </div>
             
             <div className="flex flex-col gap-1">
               {/* TQQQ Status */}
               <div className="flex items-center gap-3">
                 <span className={`text-[10px] font-bold uppercase tracking-wider ${
                   riskMetrics.isTriggered ? "text-red-200" : "text-muted"
                 }`}>
                   TQQQ Flash Crash (-20%)
                 </span>
                 <div className="flex items-baseline gap-2">
                    <span className={`text-sm font-bold font-mono tracking-tight ${
                      riskMetrics.isTriggered ? "text-white" : 
                      riskMetrics.isWarning ? "text-amber-400" : "text-fg"
                    }`}>
                      {riskMetrics.isTriggered ? "🚨 TRIGGERED" : `${riskMetrics.distToStop.toFixed(1)}%`}
                    </span>
                    <span className={`text-[10px] font-sans tabular-nums ${
                      riskMetrics.isTriggered ? "text-red-300" : "text-muted"
                    }`}>
                      (${riskMetrics.currentPrice?.toFixed(2)} → SL ${riskMetrics.stopLossPrice?.toFixed(2)})
                    </span>
                 </div>
               </div>

               {/* QQQ Status */}
               <div className="flex items-center gap-3">
                 <span className={`text-[10px] font-bold uppercase tracking-wider ${
                   riskMetrics.isQqqTriggered ? "text-red-200" : "text-muted"
                 }`}>
                   QQQ Drop Alert (-7%)
                 </span>
                 <div className="flex items-baseline gap-2">
                    <span className={`text-sm font-bold font-mono tracking-tight ${
                      riskMetrics.isQqqTriggered ? "text-white" : 
                      riskMetrics.isQqqWarning ? "text-amber-400" : "text-fg"
                    }`}>
                      {riskMetrics.isQqqTriggered ? "🚨 TRIGGERED" : `${riskMetrics.qqqDropPct.toFixed(2)}%`}
                    </span>
                    <span className={`text-[10px] font-sans tabular-nums bg-inset px-1 rounded ${
                      riskMetrics.isQqqTriggered ? "text-red-300" : "text-muted"
                    }`}>
                      vs Prev Close
                    </span>
                 </div>
               </div>
             </div>
          </div>
          
          {riskMetrics.isAnyTriggered && (
            <div className="px-3 py-1.5 bg-red-600 text-white text-sm font-bold rounded-lg animate-bounce shadow-lg">
              EXIT NOW
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-baseline gap-4">
             {/* Primary: Action Label (what to do) */}
             <h2 className={`text-4xl font-bold tracking-tight font-sans ${
               isOff ? "text-amber-400" : "text-positive"
             }`}>
               {actionLabel}
             </h2>
             {/* Secondary: System State (technical status) */}
             <span className={`text-lg px-2 py-0.5 rounded font-medium font-sans ${
               isOff 
                 ? "bg-status-inactive-bg text-status-inactive-fg" 
                 : "bg-positive/10 text-positive"
             }`}>
               {stateLabel}
             </span>
        </div>
        
        {vm.emergencyState === "SOFT_ALERT" && (
           <div className="flex items-center gap-2 text-status-action-fg text-sm font-medium bg-status-action-bg/10 px-3 py-1 rounded-lg border border-status-action-bg/20">
             <AlertTriangle size={16} />
             {vm.softAlertNote || "Soft Alert Active"}
           </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4">
        {vm.evidence.map((item) => (
          <div 
            key={item.window} 
            className="flex flex-col items-center justify-center p-4 rounded-xl bg-surface shadow-sm relative overflow-hidden group"
          >
            {/* Hover Tooltip - fully opaque overlay */}
            <div className="absolute inset-0 bg-surface rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col items-center justify-center p-4 z-20 pointer-events-none">
              <div className="text-xs text-center space-y-2">
                <div className="font-semibold text-fg font-sans">계산식</div>
                <div className="text-muted font-sans text-[11px]">SMA3 ÷ {item.label} - 1</div>
                <div className={`text-lg font-bold font-mono tabular-nums ${item.marginPct >= 0 ? "text-positive" : "text-negative"}`}>
                  = {item.marginPct >= 0 ? "+" : ""}{item.marginPct.toFixed(2)}%
                </div>
                <div className="pt-2 border-t border-border text-[10px] text-muted font-sans">
                  SMA3 {">"} {item.label} 이면 ON 투표
                </div>
              </div>
            </div>
            
            <div className="mb-3 p-3 rounded-full bg-inset">
               {item.pass ? <CheckCircle2 size={24} className="text-positive/80" /> : <XCircle size={24} className="text-muted/50" />}
            </div>
            
            <div className="text-muted font-medium mb-1 tracking-wide text-xs uppercase font-sans">{item.label}</div>
            
            <div className={`text-2xl font-bold tracking-tight font-mono tabular-nums ${item.marginPct >= 0 ? "text-positive" : "text-negative"}`}>
              {item.marginPct > 0 ? "+" : ""}{item.marginPct.toFixed(2)}%
            </div>
            
            {/* Clarifying text */}
            <div className="text-[10px] text-muted/70 mt-2 font-sans text-center">
              SMA3 대비 괴리율
            </div>
          </div>
        ))}
      </div>

      {/* Performance Strip with Settings */}
      {perfSummary && (
        <div className="bg-surface rounded-lg shadow-sm">
          {/* Main Strip */}
          <div className="flex items-center justify-between px-4 py-2 text-xs">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-muted border-r border-border pr-3 font-sans">
                <Wallet size={12} />
                <span>기준: {localCapital.toLocaleString()}만</span>
              </div>
              <TrendingUp size={14} className="text-neutral-500" />
              <span className="text-muted font-medium font-sans">PERF ({getPeriodLabel(localStartDate, localEndDate)}):</span>
              <span className={perfSummary.tq200.returnPct >= 0 ? "text-positive" : "text-negative"}>
                <span className="font-sans">200TQ</span> <span className="font-mono">{perfSummary.tq200.returnPct > 0 ? "+" : ""}{perfSummary.tq200.returnPct.toFixed(1)}%</span>
                <span className="text-muted ml-1">({formatKrw(perfSummary.tq200.pnl, vm.privacyMode)})</span>
              </span>
              <span className="text-border">|</span>
              <span className={perfSummary.e03.returnPct >= 0 ? "text-positive" : "text-negative"}>
                <span className="font-sans">E03</span> <span className="font-mono">{perfSummary.e03.returnPct > 0 ? "+" : ""}{perfSummary.e03.returnPct.toFixed(1)}%</span>
                <span className="text-muted ml-1">({formatKrw(perfSummary.e03.pnl, vm.privacyMode)})</span>
              </span>
              {/* Fee Badge */}
              <span className={`text-[9px] px-1.5 py-0.5 rounded font-sans ${
                includeFees ? "bg-amber-900/30 text-amber-400" : "bg-inset text-muted"
              }`}>
                {includeFees ? "세금포함" : "세전"}
              </span>
            </div>
            
            {/* Settings Toggle */}
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center gap-1 text-muted hover:text-fg transition-colors p-1 rounded hover:bg-inset"
            >
              <Settings size={12} />
              <ChevronDown size={10} className={`transition-transform ${showSettings ? "rotate-180" : ""}`} />
            </button>
          </div>
          
          {/* Settings Panel */}
          {showSettings && (
            <div className="border-t border-border px-4 py-3 space-y-3 text-xs">
              {/* Date Range Input */}
              <div className="flex items-center justify-between">
                <span className="text-muted font-sans">기간</span>
                <div className="flex items-center gap-2">
                  <input
                    type="date"
                    value={localStartDate}
                    onChange={(e) => handleDateChange('start', e.target.value)}
                    className="px-2 py-1 bg-inset rounded border border-border text-fg font-mono text-xs focus:outline-none focus:ring-1 focus:ring-positive"
                  />
                  <span className="text-muted">~</span>
                  <input
                    type="date"
                    value={localEndDate}
                    onChange={(e) => handleDateChange('end', e.target.value)}
                    className="px-2 py-1 bg-inset rounded border border-border text-fg font-mono text-xs focus:outline-none focus:ring-1 focus:ring-positive"
                  />
                </div>
              </div>
              
              {/* Capital Input */}
              <div className="flex items-center justify-between">
                <span className="text-muted font-sans">기준 금액</span>
                <div className="flex items-center gap-1">
                  <input
                    type="number"
                    value={localCapital}
                    onChange={(e) => handleCapitalChange(e.target.value)}
                    className="w-20 px-2 py-1 bg-inset rounded border border-border text-fg font-mono text-right focus:outline-none focus:ring-1 focus:ring-positive"
                  />
                  <span className="text-muted font-sans">만원</span>
                </div>
              </div>
              
              {/* Fee Toggle */}
              <div className="flex items-center justify-between">
                <span className="text-muted font-sans">수수료/세금 포함</span>
                <button
                  onClick={() => setIncludeFees(!includeFees)}
                  className={`relative w-10 h-5 rounded-full transition-colors ${
                    includeFees ? "bg-positive" : "bg-inset"
                  }`}
                >
                  <span className={`absolute top-0.5 ${includeFees ? "right-0.5" : "left-0.5"} w-4 h-4 bg-white rounded-full shadow transition-all`} />
                </button>
              </div>
              
              {/* Info Note */}
              <div className="text-[10px] text-muted/70 font-sans pt-2 border-t border-border">
                ※ 백테스트 기준. 실제 수익과 다를 수 있습니다.
              </div>
            </div>
          )}
        </div>
      )}
      
      <MacroStrip data={macroData} isLoading={macroLoading} />
    </section>
  );
}
