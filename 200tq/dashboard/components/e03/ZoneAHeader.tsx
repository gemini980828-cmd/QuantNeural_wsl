"use client";

import { E03ViewModel, HeaderBadge, BadgeTone } from "../../lib/ops/e03/types";
import { AlertCircle, CheckCircle, Info, Shield, ShieldAlert, MonitorPlay, Clock, Database, AlertTriangle } from "lucide-react";
import { ThemeToggle } from "../ThemeToggle";

interface ZoneAHeaderProps {
  vm: E03ViewModel;
  onToggleSimulation?: (next: boolean) => void;
  onTogglePrivacy?: (next: boolean) => void;
}

export default function ZoneAHeader({ vm, onToggleSimulation, onTogglePrivacy }: ZoneAHeaderProps) {
  const needsRecord = vm.executionState === "DUE_TODAY" || vm.executionState === "UNKNOWN";
  
  // Determine emergency state
  const hasEmergency = vm.emergencyState !== "NONE";
  const isHardEmergency = vm.emergencyState === "HARD_CONFIRMED";
  
  // Determine execution context
  const isExecScheduled = vm.executionBadge.detail?.includes("SCHEDULED") || vm.executionBadge.detail?.includes("DUE");
  const isExecDueToday = vm.executionState === "DUE_TODAY";

  return (
    <header className="bg-bg/90 backdrop-blur-md border-b border-border pb-4 mb-6">
      <div className="flex flex-col gap-4">
        {/* Top Row: Dates & Toggles */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 sm:gap-4 text-sm text-muted">
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-sans min-w-0">
             <span className="whitespace-nowrap">Verdict: <span className="text-fg font-mono">{vm.verdictDateLabel}</span> <span className="text-muted">(KST)</span></span>
             <span className="whitespace-nowrap">Exec: <span className="text-fg font-mono">{vm.executionDateLabel}</span> <span className="text-muted">(KST)</span></span>
          </div>
          
          <div className="flex flex-wrap gap-2 shrink-0">
            <ThemeToggle />
            <button 
               onClick={() => onToggleSimulation?.(!vm.simulationMode)}
               className={`flex items-center gap-1.5 px-2 py-1 rounded border transition-colors text-xs sm:text-sm ${
                 vm.simulationMode ? "bg-amber-900/40 text-amber-500 border-amber-800" : "border-neutral-800 hover:border-neutral-700"
               }`}
            >
              <MonitorPlay size={14} />
              <span className="hidden xs:inline">{vm.simulationMode ? "SIM MODE" : "Sim"}</span>
            </button>
            <button 
               onClick={() => onTogglePrivacy?.(!vm.privacyMode)}
               className={`flex items-center gap-1.5 px-2 py-1 rounded border transition-colors text-xs sm:text-sm ${
                 vm.privacyMode ? "bg-indigo-900/40 text-indigo-400 border-indigo-800" : "border-neutral-800 hover:border-neutral-700"
               }`}
            >
              <Shield size={14} />
              <span className="hidden xs:inline">{vm.privacyMode ? "Privacy" : "Visible"}</span>
            </button>
          </div>
        </div>

        {/* Bottom Row: Action-Priority Status (EMERGENCY > EXEC > DATA) */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
          <div className="flex flex-wrap items-center gap-2 min-w-0">
            
            {/* 1. EMERGENCY - Priority 1 (Prominent when active, minimal when NONE) */}
            {hasEmergency ? (
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-bold text-xs ${
                isHardEmergency 
                  ? "bg-red-500 text-white animate-pulse shadow-lg shadow-red-900/40" 
                  : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
              }`}>
                <AlertTriangle size={14} className={isHardEmergency ? "animate-bounce" : ""} />
                <span>{isHardEmergency ? "🚨 비상 확정" : "⚠️ 긴급 점검"}</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-neutral-600 text-xs" title="비상 상황 없음">
                <div className="w-2 h-2 rounded-full bg-neutral-700" />
                <span className="hidden sm:inline">정상</span>
              </div>
            )}
            
            {/* 2. EXEC - Priority 2 (Prominent when scheduled/due) */}
            {isExecScheduled || isExecDueToday ? (
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-bold text-xs ${
                isExecDueToday
                  ? "bg-blue-500 text-white"
                  : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
              }`}>
                <Clock size={14} />
                <span>{isExecDueToday ? "오늘 실행 예정" : "내일 아침 실행 예정"}</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-neutral-600 text-xs" title="실행 예정 없음">
                <Clock size={12} className="opacity-50" />
                <span className="hidden sm:inline">실행 없음</span>
              </div>
            )}
            
            {/* 3. DATA - Priority 3 (Subtle, tooltip) */}
            <div 
              className="flex items-center gap-1.5 text-neutral-500 text-xs cursor-help group relative"
              title={`데이터: ${vm.dataBadge.detail || "FRESH"}`}
            >
              <Database size={12} className={vm.dataBadge.detail?.includes("STALE") ? "text-amber-500" : "opacity-50"} />
              {vm.dataBadge.detail?.includes("STALE") ? (
                <span className="text-amber-500">데이터 오래됨</span>
              ) : (
                <span className="hidden lg:inline opacity-60">데이터 정상</span>
              )}
            </div>
          </div>

          {/* Record Required CTA */}
          {needsRecord && (
            <div className="animate-pulse flex items-center gap-2 px-3 py-1.5 bg-status-action-bg text-status-action-fg rounded-lg text-xs font-bold uppercase tracking-wide whitespace-nowrap shrink-0 shadow-lg">
              <AlertCircle size={14} />
              <span className="hidden sm:inline">기록 필요</span>
              <span className="sm:hidden">기록</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
