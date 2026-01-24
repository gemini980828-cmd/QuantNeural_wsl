"use client";

import { PortfolioPosition } from "../../lib/ops/e03/types";
import { ChevronDown, Eye } from "lucide-react";
import { useState } from "react";

interface PortfolioPositionsTableProps {
  positions: PortfolioPosition[];
  totalEquity: number;
}

export default function PortfolioPositionsTable({ positions, totalEquity }: PortfolioPositionsTableProps) {
  const KRW_USD = 1380.0; // Fixed for MVP Mock
  const [showWatchlist, setShowWatchlist] = useState(false);

  // Split positions: held (qty > 0) vs watchlist (qty === 0)
  const heldPositions = positions.filter(p => p.qty > 0);
  const watchlistPositions = positions.filter(p => p.qty === 0);

  const renderRow = (pos: PortfolioPosition, isWatchlist = false) => {
    const value = pos.qty * pos.currentPrice * KRW_USD;
    const cost = pos.qty * pos.avgPrice * KRW_USD;
    const pnl = value - cost;
    const pnlPct = cost > 0 ? pnl / cost : 0;
    const weight = totalEquity > 0 ? value / totalEquity : 0;

    return (
      <tr 
        key={pos.ticker} 
        className={`transition-colors ${isWatchlist ? "bg-neutral-900/50 text-neutral-500" : "hover:bg-neutral-800/30"}`}
      >
        <td className={`px-4 py-3 font-bold font-sans ${isWatchlist ? "text-neutral-500" : "text-fg"}`}>
          {pos.ticker}
          {isWatchlist && <span className="ml-2 text-[10px] text-neutral-600">(미보유)</span>}
        </td>
        <td className={`px-4 py-3 text-right font-mono ${isWatchlist ? "text-neutral-600" : ""}`}>
          {pos.qty.toLocaleString()}
        </td>
        <td className="px-4 py-3 text-right font-mono text-muted">
          ${pos.avgPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </td>
        <td className={`px-4 py-3 text-right font-mono ${isWatchlist ? "text-neutral-500" : "text-fg"}`}>
          ${pos.currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </td>
        <td className="px-4 py-3 text-right font-mono font-medium">
          {isWatchlist ? "-" : `₩${Math.round(value).toLocaleString()}`}
        </td>
        <td className="px-4 py-3 text-right font-mono">
          {isWatchlist ? (
            <span className="text-neutral-600">-</span>
          ) : (
            <>
              <div className={pnl > 0 ? "text-positive" : pnl < 0 ? "text-negative" : "text-muted"}>
                {pnl > 0 ? "+" : ""}₩{Math.round(pnl).toLocaleString()}
              </div>
              <div className={`text-[10px] ${pnl > 0 ? "text-positive/70" : pnl < 0 ? "text-negative/70" : "text-muted"}`}>
                {pnl > 0 ? "+" : ""}{(pnlPct * 100).toFixed(2)}%
              </div>
            </>
          )}
        </td>
        <td className="px-4 py-3 text-right font-mono text-muted">
          {isWatchlist ? "-" : `${(weight * 100).toFixed(1)}%`}
        </td>
      </tr>
    );
  };

  return (
    <div className="space-y-4">
      {/* Main Holdings Table */}
      <div className="rounded-xl border border-neutral-800 bg-surface overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-muted uppercase bg-neutral-900 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3 font-medium">티커</th>
                <th className="px-4 py-3 font-medium text-right">수량</th>
                <th className="px-4 py-3 font-medium text-right">평단가</th>
                <th className="px-4 py-3 font-medium text-right">현재가</th>
                <th className="px-4 py-3 font-medium text-right">평가금액 (KRW)</th>
                <th className="px-4 py-3 font-medium text-right">평가손익</th>
                <th className="px-4 py-3 font-medium text-right">비중</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              {heldPositions.map((pos) => renderRow(pos, false))}
              
              {heldPositions.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-muted">
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
        <div className="rounded-xl border border-neutral-800/50 bg-neutral-900/30 overflow-hidden">
          <button
            onClick={() => setShowWatchlist(!showWatchlist)}
            className="w-full flex items-center justify-between px-4 py-3 text-xs text-neutral-500 hover:text-neutral-400 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Eye size={14} className="opacity-50" />
              <span className="font-medium">전략 자산 (미보유)</span>
              <span className="text-neutral-600">• {watchlistPositions.length}개</span>
            </div>
            <ChevronDown size={14} className={`transition-transform ${showWatchlist ? "rotate-180" : ""}`} />
          </button>
          
          {showWatchlist && (
            <div className="border-t border-neutral-800/50">
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
