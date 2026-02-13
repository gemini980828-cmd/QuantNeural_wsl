"use client";

import { useState, useEffect, useCallback } from "react";
import { Globe, RefreshCw, ArrowUpRight, ArrowDownRight, Minus, TrendingUp, Percent, DollarSign, BarChart3 } from "lucide-react";

type ColorTone = 'ok' | 'action' | 'danger';

interface MacroData {
  // Core
  vix: { value: number | null; color: ColorTone; change: number | null };
  fng: { value: number | null; label: string; color: ColorTone; change: number | null };
  treasury: { value: number | null; change: number | null };
  dxy: { value: number | null; change: number | null };
  nq: { value: number | null; change: number | null };
  usdkrw: { value: number | null; change: number | null };
  // New Yahoo Finance
  vix3m: { value: number | null; change: number | null };
  sp500: { value: number | null; change: number | null };
  esFutures: { value: number | null; change: number | null };
  gold: { value: number | null; change: number | null };
  oil: { value: number | null; change: number | null };
  btc: { value: number | null; change: number | null };
  // FRED (optional)
  yieldCurve?: { value: number | null; color: ColorTone; change: number | null };
  hySpread?: { value: number | null; color: ColorTone; change: number | null };
  treasury2y?: { value: number | null; change: number | null };
  updatedAt: string;
}

type TabId = "indicators" | "news";

export default function MacroPage() {
  const [data, setData] = useState<MacroData | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<TabId>("indicators");

  const fetchData = useCallback(async (forceRefresh = false) => {
    setLoading(true);
    try {
      const url = forceRefresh ? "/api/macro?refresh=true" : "/api/macro";
      const res = await fetch(url);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (e) {
      console.error("Failed to fetch macro data:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRefresh = () => {
    fetchData(true);
  };

  const getCircleColor = (color: ColorTone): string => {
    switch (color) {
      case 'ok': return 'bg-positive';
      case 'action': return 'bg-choppy';
      case 'danger': return 'bg-negative';
      default: return 'bg-muted';
    }
  };

  const getTextColor = (color: ColorTone): string => {
    switch (color) {
      case 'ok': return 'text-positive';
      case 'action': return 'text-choppy';
      case 'danger': return 'text-negative';
      default: return 'text-muted';
    }
  };

  const getChangeColor = (change: number | null | undefined, inverted: boolean): string => {
    if (change === null || change === undefined || change === 0) return 'text-muted';
    const isPositive = change > 0;
    const isGood = inverted ? !isPositive : isPositive;
    return isGood ? 'text-positive' : 'text-negative';
  };

  const formatChange = (change: number | null | undefined, isPoints: boolean = false): string => {
    if (change === null || change === undefined) return '--';
    if (change === 0) return '0%';
    const sign = change > 0 ? '+' : '';
    if (isPoints) return `${sign}${change}`;
    return `${sign}${change.toFixed(1)}%`;
  };

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

  return (
    <div className="space-y-6 pb-20">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          <Globe size={24} className="text-accent" />
          매크로 상황판
          <span className="text-sm font-normal text-muted bg-surface px-2.5 py-0.5 rounded-full border border-border">
            Macro Intel
          </span>
        </h1>
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="p-2 rounded-lg hover:bg-surface text-muted transition-colors"
        >
          <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border pb-2" data-testid="macro-tabs">
        <button
          onClick={() => setTab("indicators")}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            tab === "indicators" ? "bg-inset text-white" : "text-muted hover:bg-surface"
          }`}
        >
          지표
        </button>
        <button
          onClick={() => setTab("news")}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            tab === "news" ? "bg-inset text-white" : "text-muted hover:bg-surface"
          }`}
        >
          뉴스
        </button>
      </div>

      {/* Content */}
      {tab === "indicators" ? (
        <div className="space-y-6">
          {/* Category 1: Volatility & Sentiment */}
          <CategorySection 
            icon={<BarChart3 size={16} className="text-choppy" />} 
            title="변동성 & 센티먼트"
          >
            <IndicatorCard 
              title="VIX" 
              value={formatValue(data?.vix.value)} 
              change={formatChange(data?.vix.change)}
              changeColor={getChangeColor(data?.vix.change, true)}
              statusColor={getCircleColor(data?.vix.color || 'action')}
              valueColor={getTextColor(data?.vix.color || 'action')}
              loading={loading}
            />
            <IndicatorCard 
              title="VIX 3M" 
              value={formatValue(data?.vix3m?.value)} 
              change={formatChange(data?.vix3m?.change)}
              changeColor={getChangeColor(data?.vix3m?.change, true)}
              loading={loading}
            />
            <IndicatorCard 
              title="Fear & Greed" 
              value={data?.fng.value?.toString() || '--'} 
              label={data?.fng.label}
              change={formatChange(data?.fng.change, true)}
              changeColor={getChangeColor(data?.fng.change, false)}
              statusColor={getCircleColor(data?.fng.color || 'action')}
              valueColor={getTextColor(data?.fng.color || 'action')}
              loading={loading}
            />
          </CategorySection>

          {/* Category 2: Rates & Credit */}
          <CategorySection 
            icon={<Percent size={16} className="text-info" />} 
            title="금리 & 크레딧"
          >
            <IndicatorCard 
              title="US 10Y" 
              value={formatValue(data?.treasury.value, 2, '%')} 
              change={formatChange(data?.treasury.change)}
              changeColor={getChangeColor(data?.treasury.change, true)}
              loading={loading}
            />
            {data?.treasury2y && (
              <IndicatorCard 
                title="US 2Y" 
                value={formatValue(data.treasury2y.value, 2, '%')} 
                change={formatChange(data.treasury2y.change)}
                changeColor={getChangeColor(data.treasury2y.change, true)}
                loading={loading}
              />
            )}
            {data?.yieldCurve && (
              <IndicatorCard 
                title="Yield Curve" 
                subtitle="10Y-2Y"
                value={formatValue(data.yieldCurve.value, 2, '%')} 
                change={formatChange(data.yieldCurve.change)}
                changeColor={getChangeColor(data.yieldCurve.change, false)}
                statusColor={getCircleColor(data.yieldCurve.color)}
                valueColor={getTextColor(data.yieldCurve.color)}
                loading={loading}
              />
            )}
            {data?.hySpread && (
              <IndicatorCard 
                title="HY Spread" 
                value={formatValue(data.hySpread.value, 2, '%')} 
                change={formatChange(data.hySpread.change)}
                changeColor={getChangeColor(data.hySpread.change, true)}
                statusColor={getCircleColor(data.hySpread.color)}
                valueColor={getTextColor(data.hySpread.color)}
                loading={loading}
              />
            )}
          </CategorySection>

          {/* Category 3: Markets & Futures */}
          <CategorySection 
            icon={<TrendingUp size={16} className="text-positive" />} 
            title="시장 & 선물"
          >
            <IndicatorCard 
              title="S&P 500" 
              value={formatWithCommas(data?.sp500?.value)} 
              change={formatChange(data?.sp500?.change)}
              changeColor={getChangeColor(data?.sp500?.change, false)}
              loading={loading}
            />
            <IndicatorCard 
              title="ES Futures" 
              value={formatWithCommas(data?.esFutures?.value)} 
              change={formatChange(data?.esFutures?.change)}
              changeColor={getChangeColor(data?.esFutures?.change, false)}
              loading={loading}
            />
            <IndicatorCard 
              title="NQ Futures" 
              value={formatWithCommas(data?.nq?.value)} 
              change={formatChange(data?.nq?.change)}
              changeColor={getChangeColor(data?.nq?.change, false)}
              loading={loading}
            />
          </CategorySection>

          {/* Category 4: FX & Commodities */}
          <CategorySection 
            icon={<DollarSign size={16} className="text-accent" />} 
            title="통화 & 원자재"
          >
            <IndicatorCard 
              title="DXY" 
              value={formatValue(data?.dxy?.value, 1)} 
              change={formatChange(data?.dxy?.change)}
              changeColor={getChangeColor(data?.dxy?.change, true)}
              loading={loading}
            />
            <IndicatorCard 
              title="USD/KRW" 
              value={formatWithCommas(data?.usdkrw?.value)} 
              change={formatChange(data?.usdkrw?.change)}
              changeColor={getChangeColor(data?.usdkrw?.change, true)}
              loading={loading}
            />
            <IndicatorCard 
              title="Gold" 
              value={formatWithCommas(data?.gold?.value, 1)} 
              change={formatChange(data?.gold?.change)}
              changeColor={getChangeColor(data?.gold?.change, false)}
              loading={loading}
            />
            <IndicatorCard 
              title="Crude Oil" 
              value={formatValue(data?.oil?.value, 2)} 
              change={formatChange(data?.oil?.change)}
              changeColor={getChangeColor(data?.oil?.change, false)}
              loading={loading}
            />
            <IndicatorCard 
              title="Bitcoin" 
              value={formatWithCommas(data?.btc?.value)} 
              change={formatChange(data?.btc?.change)}
              changeColor={getChangeColor(data?.btc?.change, false)}
              loading={loading}
            />
          </CategorySection>
        </div>
      ) : (
        <div className="bg-surface rounded-xl border border-border p-12 text-center">
          <div className="text-4xl mb-4">🗞️</div>
          <h3 className="text-lg font-medium text-fg mb-1">AI 뉴스 분석 준비중</h3>
          <p className="text-muted text-sm">주요 매크로 뉴스 및 시장 영향을 분석하여 제공할 예정입니다.</p>
        </div>
      )}

      {data?.updatedAt && (
        <p className="text-[11px] text-muted text-right">
          Last updated: {new Date(data.updatedAt).toLocaleString()}
        </p>
      )}
    </div>
  );
}

function IndicatorCard({ 
  title, value, label, subtitle, change, changeColor, statusColor, valueColor, loading 
}: { 
  title: string; 
  value: string; 
  label?: string; 
  subtitle?: string;
  change: string; 
  changeColor?: string; 
  statusColor?: string; 
  valueColor?: string; 
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="bg-surface rounded-xl border border-border p-4 h-32 animate-pulse">
        <div className="h-4 w-20 bg-inset rounded mb-4" />
        <div className="h-8 w-24 bg-inset rounded mb-2" />
        <div className="h-3 w-16 bg-inset rounded" />
      </div>
    );
  }

  const isPositive = change.startsWith('+');
  const isZero = change === '0%';
  const Icon = isZero ? Minus : (isPositive ? ArrowUpRight : ArrowDownRight);

  return (
    <div className="bg-surface rounded-xl border border-border p-4 transition-all hover:border-border">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-muted uppercase tracking-tight">{title}</span>
          {subtitle && <span className="text-[11px] text-muted">{subtitle}</span>}
        </div>
        {statusColor && <div className={`w-2 h-2 rounded-full ${statusColor}`} />}
      </div>
      <div className="flex items-baseline gap-2">
        <div className={`text-2xl font-bold tracking-tight ${valueColor || 'text-fg'}`}>
          {value}
        </div>
        {label && <span className="text-[11px] text-muted font-medium">{label}</span>}
      </div>
      <div className={`flex items-center gap-0.5 text-xs font-mono mt-1 ${changeColor}`}>
        <Icon size={12} />
        {change}
      </div>
    </div>
  );
}

function CategorySection({ 
  icon, 
  title, 
  children 
}: { 
  icon: React.ReactNode; 
  title: string; 
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <h2 className="text-sm font-semibold text-fg">{title}</h2>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {children}
      </div>
    </div>
  );
}
