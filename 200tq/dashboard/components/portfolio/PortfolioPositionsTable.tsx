"use client";

import { PortfolioPosition, StrategyState } from "../../lib/ops/e03/types";
import { ChevronDown, Eye } from "lucide-react";
import { useState } from "react";

interface PortfolioPositionsTableProps {
  positions: PortfolioPosition[];
  totalEquity: number;
  strategyState?: StrategyState;
  targetTqqqWeight?: number;
}

function getTargetWeight(ticker: string, targetTqqqWeight?: number): number | null {
  if (targetTqqqWeight == null) return null;
  if (ticker === "TQQQ") return targetTqqqWeight;
  if (ticker === "SGOV" || ticker === "SPLG") return 1 - targetTqqqWeight;
  return null;
}

function getDeviationColor(deviation: number): string {
  const abs = Math.abs(deviation);
  if (abs < 0.03) return "text-muted";
  if (abs < 0.10) return "text-choppy";
  return "text-negative";
}

export default function PortfolioPositionsTable({ positions, totalEquity, strategyState, targetTqqqWeight }: PortfolioPositionsTableProps) {
  const KRW_USD = 1380.0; // Fixed for MVP Mock
  const [showWatchlist, setShowWatchlist] = useState(false);
  const hasStrategy = strategyState != null && targetTqqqWeight != null;

  // Split positions: held (qty > 0) vs watchlist (qty === 0)
  const heldPositions = positions.filter(p => p.qty > 0);
  const watchlistPositions = positions.filter(p => p.qty === 0);

  const renderRow = (pos: PortfolioPosition, isWatchlist = false) => {
    const value = pos.qty * pos.currentPrice * KRW_USD;
    const cost = pos.qty * pos.avgPrice * KRW_USD;
    const pnl = value - cost;
    const pnlPct = cost > 0 ? pnl / cost : 0;
    const weight = totalEquity > 0 ? value / totalEquity : 0;
    const target = getTargetWeight(pos.ticker, targetTqqqWeight);
    const deviation = target != null ? weight - target : null;

    return (
      <tr 
        key={pos.ticker} 
        className={`transition-colors ${isWatchlist ? "bg-inset/50 text-muted" : "hover:bg-inset/30"}`}
      >
        <td className={`px-4 py-3 font-bold font-sans ${isWatchlist ? "text-muted" : "text-fg"}`}>
          {pos.ticker}
          {isWatchlist && <span className="ml-2 text-[11px] text-muted">(미보유)</span>}
        </td>
        <td className={`px-4 py-3 text-right font-mono ${isWatchlist ? "text-muted" : ""}`}>
          {pos.qty.toLocaleString()}
        </td>
        <td className="px-4 py-3 text-right font-mono text-muted">
          ${pos.avgPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </td>
        <td className={`px-4 py-3 text-right font-mono ${isWatchlist ? "text-muted" : "text-fg"}`}>
          ${pos.currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </td>
        <td className="px-4 py-3 text-right font-mono font-medium">
          {isWatchlist ? "-" : `₩${Math.round(value).toLocaleString()}`}
        </td>
        <td className="px-4 py-3 text-right font-mono">
          {isWatchlist ? (
            <span className="text-muted">-</span>
          ) : (
            <>
              <div className={pnl > 0 ? "text-positive" : pnl < 0 ? "text-negative" : "text-muted"}>
                {pnl > 0 ? "+" : ""}₩{Math.round(pnl).toLocaleString()}
              </div>
              <div className={`text-[11px] ${pnl > 0 ? "text-positive/70" : pnl < 0 ? "text-negative/70" : "text-muted"}`}>
                {pnl > 0 ? "+" : ""}{(pnlPct * 100).toFixed(2)}%
              </div>
            </>
          )}
        </td>
        <td className="px-4 py-3 text-right font-mono text-muted">
          {isWatchlist ? "-" : `${(weight * 100).toFixed(1)}%`}
        </td>
        {hasStrategy && (
          <>
            <td className="px-4 py-3 text-right font-mono text-muted">
              {isWatchlist || target == null ? "-" : `${(target * 100).toFixed(0)}%`}
            </td>
            <td className={`px-4 py-3 text-right font-mono ${isWatchlist || deviation == null ? "text-muted" : getDeviationColor(deviation)}`}>
              {isWatchlist || deviation == null ? "-" : (
                <>{deviation > 0 ? "+" : ""}{(deviation * 100).toFixed(1)}%p</>
              )}
            </td>
          </>
        )}
      </tr>
    );
  };

  return (
    <div className="space-y-4">
      {/* Main Holdings Table */}
      <div className="rounded-xl border border-border bg-surface overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-muted uppercase bg-inset border-b border-border">
              <tr>
                <th className="px-4 py-3 font-medium">티커</th>
                <th className="px-4 py-3 font-medium text-right">수량</th>
                <th className="px-4 py-3 font-medium text-right">평단가</th>
                <th className="px-4 py-3 font-medium text-right">현재가</th>
                <th className="px-4 py-3 font-medium text-right">평가금액 (KRW)</th>
                <th className="px-4 py-3 font-medium text-right">평가손익</th>
                <th className="px-4 py-3 font-medium text-right">비중</th>
                {hasStrategy && (
                  <>
                    <th className="px-4 py-3 font-medium text-right">목표</th>
                    <th className="px-4 py-3 font-medium text-right">괴리</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              {heldPositions.map((pos) => renderRow(pos, false))}
              
              {heldPositions.length === 0 && (
                <tr>
                  <td colSpan={hasStrategy ? 9 : 7} className="px-4 py-8 text-center text-muted">
                    보유 종목이 없습니다.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Watchlist Section (0 qty positions) */}
      {watchlistPositions.length > 0 && (
        <div className="rounded-xl border border-border/50 bg-inset/30 overflow-hidden">
          <button
            onClick={() => setShowWatchlist(!showWatchlist)}
            className="w-full flex items-center justify-between px-4 py-3 text-xs text-muted hover:text-muted transition-colors"
          >
            <div className="flex items-center gap-2">
              <Eye size={14} className="opacity-50" />
              <span className="font-medium">전략 자산 (미보유)</span>
              <span className="text-muted">• {watchlistPositions.length}개</span>
            </div>
            <ChevronDown size={14} className={`transition-transform ${showWatchlist ? "rotate-180" : ""}`} />
          </button>
          
          {showWatchlist && (
            <div className="border-t border-border/50">
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <tbody className="divide-y divide-neutral-800/50">
                    {watchlistPositions.map((pos) => renderRow(pos, true))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
