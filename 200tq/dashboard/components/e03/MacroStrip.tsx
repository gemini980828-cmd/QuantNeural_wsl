"use client";

import Link from "next/link";
import { Globe, Activity, Heart, Landmark, DollarSign, TrendingUp, Banknote } from "lucide-react";
import { StatusBadge, StatusTone } from "./StatusBadge";

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

interface MacroStripProps {
  data: MacroData | null;
  isLoading: boolean;
}

function mapToStatusTone(color: ColorTone): StatusTone {
  return color;
}

interface IndicatorCardProps {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  sublabel?: string;
}

function IndicatorCard({ icon, label, value, sublabel }: IndicatorCardProps) {
  return (
    <div className="flex flex-col items-center justify-center p-3 rounded-xl bg-inset shadow-sm">
      <div className="mb-2 p-2 rounded-full bg-surface">
        {icon}
      </div>
      <div className="text-muted font-medium mb-1 tracking-wide text-[10px] uppercase font-sans">
        {label}
      </div>
      <div className="text-lg font-bold tracking-tight font-mono tabular-nums text-fg">
        {value}
      </div>
      {sublabel && (
        <div className="text-[9px] text-muted/70 mt-0.5 font-sans">
          {sublabel}
        </div>
      )}
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
      <div className="bg-surface rounded-lg shadow-sm px-4 py-2 mt-2">
        <div className="flex items-center gap-2 text-xs text-muted">
          <Globe size={12} />
          <span>매크로 지표 로딩...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-lg shadow-sm px-4 py-4 mt-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-muted">
          <Globe size={14} />
          <span className="text-xs font-medium">MACRO</span>
        </div>
        <Link 
          href="/macro" 
          className="text-muted hover:text-fg text-xs transition-colors"
        >
          상세 →
        </Link>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
        <IndicatorCard
          icon={<Activity size={16} className="text-muted" />}
          label="VIX"
          value={
            data?.vix.value !== null ? (
              <StatusBadge tone={mapToStatusTone(data?.vix.color || 'action')}>
                {formatValue(data?.vix.value)}
              </StatusBadge>
            ) : '--'
          }
        />

        <IndicatorCard
          icon={<Heart size={16} className="text-muted" />}
          label="F&G"
          value={
            data?.fng.value !== null ? (
              <StatusBadge tone={mapToStatusTone(data?.fng.color || 'action')}>
                {data?.fng.value}
              </StatusBadge>
            ) : '--'
          }
          sublabel={data?.fng.label || undefined}
        />

        <IndicatorCard
          icon={<Landmark size={16} className="text-muted" />}
          label="10Y"
          value={formatValue(data?.treasury.value, 2, '%')}
        />

        <IndicatorCard
          icon={<DollarSign size={16} className="text-muted" />}
          label="DXY"
          value={formatValue(data?.dxy.value, 1)}
        />

        <IndicatorCard
          icon={<TrendingUp size={16} className="text-muted" />}
          label="NQ"
          value={formatWithCommas(data?.nq.value, 0)}
          sublabel="선물"
        />

        <IndicatorCard
          icon={<Banknote size={16} className="text-muted" />}
          label="환율"
          value={`${formatWithCommas(data?.usdkrw.value, 0)}`}
          sublabel="원"
        />
      </div>
    </div>
  );
}
