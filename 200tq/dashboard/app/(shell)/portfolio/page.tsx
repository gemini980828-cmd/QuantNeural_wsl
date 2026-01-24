"use client";

import { useState } from "react";
import { getMockInputsByScenario, ScenarioId } from "../../../lib/ops/e03/mock"; 
import { buildViewModel } from "../../../lib/ops/e03/buildViewModel";
import PortfolioSummaryStrip from "../../../components/portfolio/PortfolioSummaryStrip";
import PortfolioPositionsTable from "../../../components/portfolio/PortfolioPositionsTable";
import { Info, History, LayoutGrid, Layers, TrendingUp } from "lucide-react";

export default function PortfolioPage() {
  // Use same mock logic as Command Dashboard for consistency
  const [scenario, setScenario] = useState<ScenarioId>("fresh_normal");
  const inputs = getMockInputsByScenario(scenario);
  const vm = buildViewModel(inputs);
  const portfolio = vm.portfolio;

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
             <span className="text-sm">최근 체결 내역이 없습니다. (Mock)</span>
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
