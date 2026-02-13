"use client";

import Link from "next/link";
import { E03ViewModel } from "../../lib/ops/e03/types";
import { AlertCircle, Shield, MonitorPlay, Clock, Database, AlertTriangle, Bell, LayoutDashboard, Minimize2, Activity, Clock3 } from "lucide-react";
import { useViewMode, useSettingsStore } from "@/lib/stores/settings-store";

interface ZoneAHeaderProps {
  vm: E03ViewModel;
  unresolvedAlerts?: number;
}

export default function ZoneAHeader({ vm, unresolvedAlerts = 0 }: ZoneAHeaderProps) {
  const viewMode = useViewMode();
  const store = useSettingsStore();
  const needsRecord = vm.executionState === "DUE_TODAY" || vm.executionState === "UNKNOWN";
  const isChoppy = vm.strategyState === "ON_CHOPPY";
  const isEmergencyState = vm.strategyState === "EMERGENCY";
  
  // Determine emergency state
  const isHardEmergency = vm.emergencyState === "HARD_CONFIRMED";
  const showHardEmergency = isHardEmergency || isEmergencyState;
  const hasEmergency = vm.emergencyState !== "NONE" || isEmergencyState;
  
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
            {vm.simulationMode && (
              <span className="flex items-center gap-1.5 px-2 py-1 rounded border text-xs bg-choppy-tint text-choppy border-choppy/40">
                <MonitorPlay size={14} />
                <span className="hidden xs:inline">SIM</span>
              </span>
            )}
            {vm.privacyMode && (
              <span className="flex items-center gap-1.5 px-2 py-1 rounded border text-xs bg-indigo-900/40 text-indigo-400 border-indigo-800">
                <Shield size={14} />
                <span className="hidden xs:inline">Privacy</span>
              </span>
            )}
            <Link 
              href="/notifications"
              className={`relative flex items-center gap-1.5 px-2 py-1 rounded border transition-colors text-xs sm:text-sm ${
                unresolvedAlerts > 0 
                  ? "bg-negative-tint text-negative border-negative/40" 
                   : "border-border hover:border-fg/20"
               }`}
            >
              <Bell size={14} />
              {unresolvedAlerts > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-[11px] font-bold px-1.5 py-0.5 rounded-full min-w-[18px] text-center">
                  {unresolvedAlerts > 9 ? "9+" : unresolvedAlerts}
                </span>
              )}
              <span className="hidden lg:inline">알림</span>
            </Link>
            
            {/* View Mode Toggle */}
            <button 
               onClick={() => store.setSetting('viewMode', viewMode === 'simple' ? 'pro' : 'simple')}
               className={`flex items-center gap-1.5 px-2 py-1 rounded border transition-colors text-xs sm:text-sm ${
                  viewMode === 'pro' ? "bg-purple-900/40 text-purple-400 border-purple-800" : "border-border hover:border-fg/20"
               }`}
            >
              {viewMode === 'simple' ? <LayoutDashboard size={14} /> : <Minimize2 size={14} />}
              <span className="hidden xs:inline">{viewMode === 'simple' ? "Simple" : "Pro"}</span>
            </button>
          </div>
        </div>

        {/* Bottom Row: Action-Priority Status (EMERGENCY > EXEC > DATA) */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
          <div className="flex flex-wrap items-center gap-2 min-w-0">
            
            {/* 1. EMERGENCY - Priority 1 (Prominent when active, minimal when NONE) */}
            {hasEmergency && (
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-bold text-xs ${
                showHardEmergency 
                  ? "bg-negative text-white animate-pulse shadow-lg shadow-negative/30" 
                  : "bg-choppy-tint text-choppy border border-choppy/30"
              }`}>
                <AlertTriangle size={14} className={showHardEmergency ? "animate-bounce" : ""} />
                <span>{showHardEmergency ? "🚨 비상 확정" : "⚠️ 긴급 점검"}</span>
              </div>
            )}

            {isChoppy && (
              <div className="flex items-center gap-1.5 px-2 py-1 rounded border text-xs font-medium bg-choppy-tint text-choppy border-choppy/40">
                <Activity size={14} />
                <span>시그널 불안정</span>
              </div>
            )}
            {vm.strategyState === "OFF10" && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-md text-xs bg-status-inactive-bg text-status-inactive-fg">
                <span>OFF10</span>
              </div>
            )}

            {vm.cooldownActive && (
              <div className="flex items-center gap-1.5 px-2 py-1 rounded border text-xs bg-choppy-tint text-choppy border-choppy/40">
                <Clock3 size={12} />
                <span>쿨다운</span>
              </div>
            )}
            
            {(isExecScheduled || isExecDueToday) && (
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-bold text-xs ${
                isExecDueToday
                  ? "bg-info text-white"
                  : "bg-info-tint text-info border border-info/30"
              }`}>
                <Clock size={14} />
                <span>{isExecDueToday ? "오늘 실행 예정" : "내일 아침 실행 예정"}</span>
              </div>
            )}
            
            {vm.dataBadge.detail?.includes("STALE") && (
              <div className="flex items-center gap-1.5 text-choppy text-xs">
                <Database size={12} />
                <span>데이터 오래됨</span>
              </div>
            )}
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
