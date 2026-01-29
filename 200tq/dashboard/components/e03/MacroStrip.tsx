"use client";

import Link from "next/link";
import { Globe } from "lucide-react";

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

interface MacroStripProps {
  data: MacroData | null;
  isLoading: boolean;
}

function getCircleColor(color: ColorTone): string {
  switch (color) {
    case 'ok': return 'bg-emerald-500';
    case 'action': return 'bg-amber-500';
    case 'danger': return 'bg-red-500';
  }
}

function getTextColor(color: ColorTone): string {
  switch (color) {
    case 'ok': return 'text-emerald-400';
    case 'action': return 'text-amber-400';
    case 'danger': return 'text-red-400';
  }
}

function getChangeColor(change: number | null | undefined, inverted: boolean): string {
  if (change === null || change === undefined || change === 0) return 'text-muted';
  const isPositive = change > 0;
  const isGood = inverted ? !isPositive : isPositive;
  return isGood ? 'text-emerald-400' : 'text-red-400';
}

function formatChange(change: number | null | undefined, isPoints: boolean = false): string {
  if (change === null || change === undefined) return '--';
  if (change === 0) return '0%';
  const sign = change > 0 ? '+' : '';
  if (isPoints) {
    return `${sign}${change}`;
  }
  return `${sign}${change.toFixed(1)}%`;
}

function Separator() {
  return <span className="text-border mx-2">|</span>;
}

export default function MacroStrip({ data, isLoading }: MacroStripProps) {
  const formatValue = (val: number | null | undefined, decimals: number = 1, suffix: string = '') => {
    if (val === null || val === undefined) return '--';
    return val.toFixed(decimals) + suffix;
  };

  const formatWithCommas = (val: number | null | undefined, decimals: number = 0) => {
    if (val === null || val === undefined) return '--';
    return val.toLocaleString('en-US', { 
      minimumFractionDigits: decimals, 
      maximumFractionDigits: decimals 
    });
  };

  if (isLoading) {
    return (
      <div className="bg-surface rounded-lg shadow-sm px-4 py-2 mt-2">
        <div className="flex items-center gap-2 text-xs text-muted">
          <Globe size={12} />
          <span>매크로 지표 로딩...</span>
        </div>
      </div>
    );
  }

  const vixColor = data?.vix.color || 'action';
  const fngColor = data?.fng.color || 'action';

  return (
    <div className="bg-surface rounded-lg shadow-sm px-4 py-3 mt-3">
      <div className="flex items-center text-sm">
        <Globe size={14} className="text-muted shrink-0 mr-4" />
        
        <div className="flex items-center flex-wrap gap-y-1">
          <div className="flex items-center gap-1.5">
            <span className="text-muted text-xs">VIX</span>
            <span className={`w-2 h-2 rounded-full ${getCircleColor(vixColor)}`} />
            <span className={`font-mono font-medium ${getTextColor(vixColor)}`}>
              {formatValue(data?.vix.value)}
            </span>
            <span className={`text-xs font-mono ${getChangeColor(data?.vix.change, true)}`}>
              {formatChange(data?.vix.change)}
            </span>
          </div>
          
          <Separator />
          
          <div className="flex items-center gap-1.5">
            <span className="text-muted text-xs">F&G</span>
            <span className={`w-2 h-2 rounded-full ${getCircleColor(fngColor)}`} />
            <span className={`font-mono font-medium ${getTextColor(fngColor)}`}>
              {data?.fng.value ?? '--'}
            </span>
            <span className={`text-xs font-mono ${getChangeColor(data?.fng.change, false)}`}>
              {formatChange(data?.fng.change, true)}
            </span>
          </div>
          
          <Separator />
          
          <div className="flex items-center gap-1.5">
            <span className="text-muted text-xs">10Y</span>
            <span className="text-fg font-mono font-medium">
              {formatValue(data?.treasury.value, 2, '%')}
            </span>
            <span className={`text-xs font-mono ${getChangeColor(data?.treasury.change, true)}`}>
              {formatChange(data?.treasury.change)}
            </span>
          </div>
          
          <Separator />
          
          <div className="flex items-center gap-1.5">
            <span className="text-muted text-xs">DXY</span>
            <span className="text-fg font-mono font-medium">
              {formatValue(data?.dxy.value, 1)}
            </span>
            <span className={`text-xs font-mono ${getChangeColor(data?.dxy.change, true)}`}>
              {formatChange(data?.dxy.change)}
            </span>
          </div>

          <Separator />

          <div className="flex items-center gap-1.5">
            <span className="text-muted text-xs">NQ</span>
            <span className="text-fg font-mono font-medium">
              {formatWithCommas(data?.nq.value, 0)}
            </span>
            <span className={`text-xs font-mono ${getChangeColor(data?.nq.change, false)}`}>
              {formatChange(data?.nq.change)}
            </span>
          </div>

          <Separator />

          <div className="flex items-center gap-1.5">
            <span className="text-muted text-xs">환율</span>
            <span className="text-fg font-mono font-medium">
              {formatWithCommas(data?.usdkrw.value, 0)}
            </span>
            <span className={`text-xs font-mono ${getChangeColor(data?.usdkrw.change, true)}`}>
              {formatChange(data?.usdkrw.change)}
            </span>
          </div>
        </div>
        
        <Link 
          href="/macro" 
          className="text-muted hover:text-fg text-xs transition-colors shrink-0 ml-auto"
        >
          상세 →
        </Link>
      </div>
    </div>
  );
}
