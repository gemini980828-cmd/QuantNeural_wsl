"use client";

import { useState, useEffect, useCallback } from "react";
import { E03RawInputs, buildViewModel } from "../../../lib/ops/e03/buildViewModel";
import { getInputs } from "../../../lib/ops/dataSource";
import { useDataSource } from "../../../lib/stores/settings-store";
import PortfolioSummaryStrip from "../../../components/portfolio/PortfolioSummaryStrip";
import PortfolioPositionsTable from "../../../components/portfolio/PortfolioPositionsTable";
import { Info, History, LayoutGrid, Layers, TrendingUp, RefreshCw, AlertTriangle } from "lucide-react";

export default function PortfolioPage() {
  const dataSource = useDataSource();
  
  // Data state
  const [rawInputs, setRawInputs] = useState<E03RawInputs | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load data based on dataSource
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const inputs = await getInputs({ 
        dataSource, 
        scenario: "fresh_normal" 
      });
      setRawInputs(inputs);
    } catch (e) {
      setError(String(e));
      console.error("Failed to load portfolio data:", e);
    } finally {
      setLoading(false);
    }
  }, [dataSource]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const vm = rawInputs ? buildViewModel(rawInputs) : null;
  const portfolio = vm?.portfolio;

  // Loading state
  if (loading) {
    return (
      <div className="p-4">
        <div className="mt-8 text-center text-muted flex flex-col items-center gap-2">
          <RefreshCw className="animate-spin" size={24} />
          <span>포트폴리오 로딩 중...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="p-4">
        <div className="mt-8 text-center text-negative flex flex-col items-center gap-2">
          <AlertTriangle size={24} />
          <span>데이터 로드 실패: {error}</span>
          <button 
            onClick={loadData}
            className="mt-2 px-4 py-2 bg-neutral-800 rounded-lg text-sm hover:bg-neutral-700"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  if (!portfolio) {
    return (
      <div className="p-4">
        <div className="mt-8 text-center text-muted">포트폴리오 데이터가 없습니다.</div>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-20">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          포트폴리오
          <span className="text-sm font-normal text-muted bg-neutral-800 px-2.5 py-0.5 rounded-full border border-neutral-700">Portfolio</span>
        </h1>
        
        {/* Data Source Indicator */}
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-1 rounded-full ${
            dataSource === "REAL" 
              ? "bg-positive/20 text-positive border border-positive/30" 
              : "bg-amber-900/30 text-amber-400 border border-amber-700/30"
          }`}>
            {dataSource}
          </span>
          <button
            onClick={loadData}
            disabled={loading}
            className="p-1.5 rounded-lg hover:bg-neutral-800 text-muted transition-colors"
            title="새로고침"
          >
            <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      <main className="space-y-8">
        
        {/* 1. Summary Section */}
        <div>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <LayoutGrid size={18} className="text-neutral-400" />
            요약
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Overview</span>
          </h2>
          <PortfolioSummaryStrip portfolio={portfolio} />
        </div>

        {/* 2. Positions Section */}
        <div>
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Layers size={18} className="text-neutral-400" />
            보유 종목
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Holdings</span>
          </h2>
          <PortfolioPositionsTable 
             positions={portfolio.positions} 
             totalEquity={portfolio.derived.totalEquity} 
          />
        </div>

        {/* 3. Execution Logs (Placeholder for MVP) */}
        <div>
             <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <History size={18} className="text-neutral-400" />
            최근 체결
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Execution Logs</span>
          </h2>
          <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
             <History size={24} className="opacity-50" />
             <span className="text-sm">최근 체결 내역이 없습니다.</span>
          </div>
        </div>

        {/* 4. Performance (Placeholder for MVP) */}
        <div>
             <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <TrendingUp size={18} className="text-neutral-400" />
            성과 분석
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Performance</span>
          </h2>
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-8 flex flex-col items-center justify-center text-muted gap-2 border-dashed">
             <Info size={24} className="opacity-50" />
             <span className="text-sm">성과 분석 차트 준비 중...</span>
          </div>
        </div>

      </main>
    </div>
  );
}
