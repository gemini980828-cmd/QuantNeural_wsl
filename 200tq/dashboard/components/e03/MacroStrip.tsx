"use client";

import Link from "next/link";
import { Globe, Activity, Heart, Landmark, DollarSign, TrendingUp, Banknote } from "lucide-react";

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

function getCircleColorClass(color: ColorTone): string {
  switch (color) {
    case 'ok': return 'bg-emerald-500';
    case 'action': return 'bg-amber-500';
    case 'danger': return 'bg-red-500';
  }
}

function getChangeColor(change: number | null, inverted: boolean): string {
  if (change === null || change === 0) return 'text-muted';
  const isPositive = change > 0;
  const isGood = inverted ? !isPositive : isPositive;
  return isGood ? 'text-emerald-400' : 'text-red-400';
}

function formatChange(change: number | null, isPoints: boolean = false): string {
  if (change === null) return '--';
  if (change === 0) return '0';
  const sign = change > 0 ? '+' : '';
  if (isPoints) {
    return `${sign}${change}pts`;
  }
  return `${sign}${change.toFixed(1)}%`;
}

interface IndicatorCardProps {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  change?: number | null;
  changeInverted?: boolean;
  isPointChange?: boolean;
  sublabel?: string;
}

function IndicatorCard({ icon, label, value, change, changeInverted = false, isPointChange = false, sublabel }: IndicatorCardProps) {
  return (
    <div className="flex flex-col items-center justify-center p-2 rounded-xl bg-inset shadow-sm">
      <div className="mb-1.5 p-1.5 rounded-full bg-surface">
        {icon}
      </div>
      <div className="text-muted font-medium mb-0.5 tracking-wide text-[10px] uppercase font-sans">
        {label}
      </div>
      <div className="text-base font-bold tracking-tight font-mono tabular-nums text-fg">
        {value}
      </div>
      {change !== undefined && (
        <div className={`text-[10px] font-mono mt-0.5 ${getChangeColor(change, changeInverted)}`}>
          {formatChange(change, isPointChange)}
        </div>
      )}
      {sublabel && (
        <div className="text-[9px] text-muted/70 font-sans">
          {sublabel}
        </div>
      )}
    </div>
  );
}

interface CircleValueProps {
  value: string;
  color: ColorTone;
}

function CircleValue({ value, color }: CircleValueProps) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full ${getCircleColorClass(color)}`} />
      <span className="text-[#ABF43F]">{value}</span>
    </div>
  );
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
      <div className="bg-surface rounded-lg shadow-sm px-3 py-1.5 mt-2">
        <div className="flex items-center gap-2 text-xs text-muted">
          <Globe size={12} />
          <span>매크로 지표 로딩...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-lg shadow-sm px-3 py-2 mt-2">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-muted">
          <Globe size={12} />
          <span className="text-[10px] font-medium">MACRO</span>
        </div>
        <Link 
          href="/macro" 
          className="text-muted hover:text-fg text-[10px] transition-colors"
        >
          상세 →
        </Link>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-6 gap-1.5">
        <IndicatorCard
          icon={<Activity size={14} className="text-muted" />}
          label="VIX"
          value={
            data?.vix.value !== null ? (
              <CircleValue 
                value={formatValue(data?.vix.value)} 
                color={data?.vix.color || 'action'} 
              />
            ) : '--'
          }
          change={data?.vix.change}
          changeInverted={true}
        />

        <IndicatorCard
          icon={<Heart size={14} className="text-muted" />}
          label="F&G"
          value={
            data?.fng.value !== null ? (
              <CircleValue 
                value={String(data?.fng.value)} 
                color={data?.fng.color || 'action'} 
              />
            ) : '--'
          }
          change={data?.fng.change}
          changeInverted={false}
          isPointChange={true}
          sublabel={data?.fng.label || undefined}
        />

        <IndicatorCard
          icon={<Landmark size={14} className="text-muted" />}
          label="10Y"
          value={formatValue(data?.treasury.value, 2, '%')}
          change={data?.treasury.change}
          changeInverted={true}
        />

        <IndicatorCard
          icon={<DollarSign size={14} className="text-muted" />}
          label="DXY"
          value={formatValue(data?.dxy.value, 1)}
          change={data?.dxy.change}
          changeInverted={true}
        />

        <IndicatorCard
          icon={<TrendingUp size={14} className="text-muted" />}
          label="NQ"
          value={formatWithCommas(data?.nq.value, 0)}
          change={data?.nq.change}
          changeInverted={false}
          sublabel="선물"
        />

        <IndicatorCard
          icon={<Banknote size={14} className="text-muted" />}
          label="환율"
          value={formatWithCommas(data?.usdkrw.value, 0)}
          change={data?.usdkrw.change}
          changeInverted={true}
          sublabel="원"
        />
      </div>
    </div>
  );
}
