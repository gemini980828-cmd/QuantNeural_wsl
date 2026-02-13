"use client";

import Link from "next/link";
import { PortfolioSnapshot } from "../../lib/ops/e03/types";
import { ArrowRight, Info } from "lucide-react";

interface PortfolioSummaryStripProps {
  portfolio?: PortfolioSnapshot;
  hideCta?: boolean;
}

export default function PortfolioSummaryStrip({ portfolio, hideCta = false }: PortfolioSummaryStripProps) {
  if (!portfolio) {
    // Empty State
    return (
      <div className="w-full bg-surface border border-border rounded-xl p-4 flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-muted">
          <Info size={16} />
          <span>보유 포트폴리오 데이터가 없습니다</span>
        </div>
        <Link 
          href="/portfolio" 
          className="text-info hover:text-info transition-colors text-xs font-bold flex items-center gap-1"
        >
          입력하기 <ArrowRight size={12} />
        </Link>
      </div>
    );
  }

  const { derived, positions } = portfolio;
  const tqqq = positions.find(p => p.ticker === "TQQQ");
  const sgov = positions.find(p => p.ticker === "SGOV");
  
  // Formatters
  const fmtKrw = (n: number) => n.toLocaleString();
  const fmtUsd = (n: number) => n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtPct = (n: number) => `${(n * 100).toFixed(2)}%`;
  const getColor = (n: number) => n > 0 ? "text-positive" : n < 0 ? "text-negative" : "text-muted";

  return (
    <section className="w-full bg-surface border border-border rounded-xl p-4 shadow-sm mb-6">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        
        {/* Metric Group 
            - Mobile (<md): Grid 2 cols for better density than stack
            - Tablet (md): Flex wrap (2 rows if needed)
            - Desktop (xl): Flex nowrap
        */}
        <div className="flex-1 grid grid-cols-2 md:flex md:flex-wrap xl:flex-nowrap md:items-center gap-y-4 gap-x-4 md:gap-x-8 w-full md:w-auto min-w-0">
          
          {/* 1. Total Equity (Always Visible) */}
          <div className="flex flex-col min-w-0">
            <span className="text-xs text-muted mb-0.5 whitespace-nowrap">총자산 (KRW)</span>
            <span className="text-lg font-bold font-mono tracking-tight truncate">
              ₩{fmtKrw(derived.totalEquity)}
            </span>
          </div>

          {/* 2. Today's P/L (Always Visible) */}
          <div className="flex flex-col min-w-0">
            <span className="text-xs text-muted mb-0.5 whitespace-nowrap">오늘 손익</span>
            <div className={`font-mono font-medium truncate ${getColor(derived.dailyPnL)}`}>
              {derived.dailyPnL > 0 ? "+" : ""}₩{fmtKrw(derived.dailyPnL)}
              <span className="text-xs ml-1 opacity-80">
                ({derived.dailyPnL > 0 ? "+" : ""}{fmtPct(derived.dailyPnLPct)})
              </span>
            </div>
          </div>

          {/* 3. TQQQ Holdings (Always Visible) */}
          <div className="flex flex-col min-w-0">
            <span className="text-xs text-muted mb-0.5 whitespace-nowrap">TQQQ 보유</span>
            <div className="font-mono font-medium text-fg truncate">
              {tqqq ? `${tqqq.qty.toLocaleString()} 주` : "-"}
              <span className="text-xs text-muted ml-1">
                ({tqqq ? fmtPct(derived.weights["TQQQ"] || 0) : "0%"})
              </span>
            </div>
          </div>

          {/* 4. TQQQ Price (Hide on very narrow screens if needed, mostly visible) */}
          <div className="flex flex-col min-w-0 hidden xs:flex">
            <span className="text-xs text-muted mb-0.5 whitespace-nowrap">평단 / 현재가</span>
            <div className="font-mono text-sm text-fg truncate">
              ${tqqq ? fmtUsd(tqqq.avgPrice) : "0.00"}
              <span className="text-muted mx-1">/</span>
              <span className={tqqq && tqqq.currentPrice > tqqq.avgPrice ? "text-positive" : "text-fg"}>
                ${tqqq ? fmtUsd(tqqq.currentPrice) : "0.00"}
              </span>
            </div>
          </div>

          {/* 5. Unrealized P/L (Always Visible) */}
          <div className="flex flex-col min-w-0 col-span-2 md:col-span-1">
            <span className="text-xs text-muted mb-0.5 whitespace-nowrap">평가 손익</span>
            <div className={`font-mono font-medium truncate ${getColor(derived.unrealizedPnL)}`}>
               {derived.unrealizedPnL > 0 ? "+" : ""}₩{fmtKrw(derived.unrealizedPnL)}
               <span className="text-xs ml-1 opacity-80">
                 ({derived.unrealizedPnL > 0 ? "+" : ""}{fmtPct(derived.unrealizedPnLPct)})
               </span>
            </div>
          </div>

        </div>

        {hideCta ? (
           <div className="hidden md:flex flex-col justify-center border-l border-border pl-4 min-w-0">
            <span className="text-xs text-muted mb-0.5 whitespace-nowrap">SGOV 보유</span>
            <div className="font-mono font-medium text-fg truncate">
              {sgov ? `${sgov.qty.toLocaleString()} 주` : "-"}
              <span className="text-xs text-muted ml-1">
                ({sgov ? fmtPct(derived.weights["SGOV"] || 0) : "0%"})
              </span>
            </div>
          </div>
        ) : (
           <div className="hidden md:flex flex-col justify-center border-l border-border pl-4 h-10 w-32 shrink-0">
            <Link 
              href="/portfolio" 
               className="w-full h-full flex items-center justify-center gap-1.5 bg-inset hover:bg-border text-muted hover:text-fg text-xs font-bold rounded-lg transition-all"
            >
              포트폴리오
              <ArrowRight size={14} />
            </Link>
          </div>
        )}
      </div>
      
      {!hideCta && (
        <Link 
          href="/portfolio" 
           className="md:hidden mt-4 w-full py-3 flex items-center justify-center gap-1.5 bg-inset hover:bg-border text-muted text-xs font-bold rounded-lg transition-all"
        >
          포트폴리오 자세히 보기
          <ArrowRight size={14} />
        </Link>
      )}
    </section>
  );
}
