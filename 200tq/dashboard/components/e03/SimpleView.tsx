"use client";

import { E03ViewModel } from "@/lib/ops/e03/types";
import MacroStrip from "./MacroStrip";
import { AlertTriangle, TrendingUp, TrendingDown, ArrowRight } from "lucide-react";

type ColorTone = 'ok' | 'action' | 'danger';

interface MacroData {
  vix: { value: number | null; color: ColorTone; change: number | null };
  fng: { value: number | null; label: string; color: ColorTone; change: number | null };
  treasury: { value: number | null; change: number | null };
  dxy: { value: number | null; change: number | null };
  nq: { value: number | null; change: number | null };
  usdkrw: { value: number | null; change: number | null };
  updatedAt: string;
}

interface SimpleViewProps {
  vm: E03ViewModel;
  macroData: MacroData | null;
  macroLoading: boolean;
  onSwitchToPro: () => void;
}

function getVerdictStyle(strategyState: "ON" | "OFF10"): { bg: string; text: string; icon: React.ReactNode; label: string } {
  if (strategyState === "ON") {
    return { 
      bg: "bg-emerald-500/20 border-emerald-500/30", 
      text: "text-emerald-400",
      icon: <TrendingUp className="w-6 h-6" />,
      label: "ON (100% TQQQ)"
    };
  }
  return { 
    bg: "bg-amber-500/20 border-amber-500/30", 
    text: "text-amber-400",
    icon: <TrendingDown className="w-6 h-6" />,
    label: "OFF10 (10% TQQQ)"
  };
}

function formatKRW(value: number, privacyMode: boolean): string {
  if (privacyMode) return "***";
  return `₩${value.toLocaleString('ko-KR')}`;
}

export default function SimpleView({ vm, macroData, macroLoading, onSwitchToPro }: SimpleViewProps) {
  const verdictStyle = getVerdictStyle(vm.strategyState);
  const hasEmergency = vm.emergencyState !== "NONE";
  const isHardEmergency = vm.emergencyState === "HARD_CONFIRMED";
  
  const totalValue = vm.portfolio?.derived?.totalEquity ?? vm.inputTotalValueKrw ?? 0;
  const dailyPnL = vm.portfolio?.derived?.dailyPnL ?? 0;
  const dailyPnLPct = vm.portfolio?.derived?.dailyPnLPct ?? 0;
  const isPositive = dailyPnL >= 0;

  return (
    <div className="flex flex-col items-center gap-6 py-8">
      <div className="text-center space-y-1">
        <p className="text-muted text-sm">총 자산</p>
        <p className="text-4xl sm:text-5xl font-bold text-fg font-mono">
          {formatKRW(totalValue, vm.privacyMode)}
        </p>
        <p className={`text-lg font-mono ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
          {vm.privacyMode ? '***' : (
            <>
              {isPositive ? '+' : ''}{dailyPnLPct.toFixed(2)}%
              <span className="text-sm ml-2">
                ({isPositive ? '+' : ''}₩{dailyPnL.toLocaleString('ko-KR')})
              </span>
            </>
          )}
        </p>
      </div>

      <div className={`flex items-center gap-3 px-6 py-4 rounded-xl border ${verdictStyle.bg}`}>
        <span className={verdictStyle.text}>{verdictStyle.icon}</span>
        <div className="text-center">
          <p className={`text-2xl font-bold ${verdictStyle.text}`}>{verdictStyle.label}</p>
          <p className="text-muted text-xs mt-1">오늘의 판정</p>
        </div>
      </div>

      {hasEmergency && (
        <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
          isHardEmergency 
            ? "bg-red-500/20 text-red-400 border border-red-500/30" 
            : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
        }`}>
          <AlertTriangle className={`w-4 h-4 ${isHardEmergency ? "animate-pulse" : ""}`} />
          <span className="text-sm font-medium">
            {isHardEmergency ? "비상 상황 확정" : "긴급 점검 필요"}
          </span>
        </div>
      )}

      <div className="w-full max-w-md">
        <MacroStrip data={macroData} isLoading={macroLoading} condensed={true} />
      </div>

      <button
        onClick={onSwitchToPro}
        className="flex items-center gap-2 text-muted hover:text-fg transition-colors text-sm mt-4"
      >
        <span>자세히 보기</span>
        <ArrowRight className="w-4 h-4" />
      </button>
    </div>
  );
}
