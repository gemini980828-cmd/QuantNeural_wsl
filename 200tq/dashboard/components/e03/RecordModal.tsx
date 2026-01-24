"use client";

import { useState, useEffect } from "react";
import { TradeLine } from "../../lib/ops/e03/types";
import { X, Check, Sparkles, AlertCircle } from "lucide-react";

interface RecordModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (fills: Record<string, number>, prices: Record<string, number>, note?: string) => void;
  executionDateLabel: string;
  expectedTrades: TradeLine[];
}

export default function RecordModal({ isOpen, onClose, onSave, executionDateLabel, expectedTrades }: RecordModalProps) {
  // Build unique tickers with their expected values
  const tradeSuggestions = expectedTrades
    .filter(t => t.action !== "HOLD" && t.shares > 0)
    .reduce((acc, t) => {
      acc[t.ticker] = { shares: t.shares, action: t.action };
      return acc;
    }, {} as Record<string, { shares: number; action: string }>);
  
  const uniqueTickers = Array.from(new Set([
    ...Object.keys(tradeSuggestions),
    "TQQQ", "SGOV" 
  ]));

  const [fills, setFills] = useState<Record<string, string>>({});
  const [prices, setPrices] = useState<Record<string, string>>({});
  const [note, setNote] = useState("");
  const [autoFilled, setAutoFilled] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Auto-fill when modal opens with expected trades
  useEffect(() => {
    if (isOpen && Object.keys(tradeSuggestions).length > 0 && !autoFilled) {
      const initialFills: Record<string, string> = {};
      Object.entries(tradeSuggestions).forEach(([ticker, { shares }]) => {
        initialFills[ticker] = shares.toString();
      });
      setFills(initialFills);
      setAutoFilled(true);
      
      // Build auto note
      const summary = Object.entries(tradeSuggestions)
        .map(([ticker, { shares, action }]) => `${action} ${ticker} ${shares}주`)
        .join(", ");
      setNote(summary);
    }
  }, [isOpen, tradeSuggestions, autoFilled]);

  // Reset when modal closes
  useEffect(() => {
    if (!isOpen) {
      setAutoFilled(false);
      setValidationError(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    // Validate: at least one ticker must have both shares and price
    const tickersWithData = uniqueTickers.filter(t => {
      const hasShares = fills[t] && parseInt(fills[t]) > 0;
      const hasPrice = prices[t] && parseFloat(prices[t]) > 0;
      return hasShares && hasPrice;
    });
    
    if (tickersWithData.length === 0) {
      setValidationError("최소 1개 종목의 수량과 가격을 입력해 주세요.");
      return;
    }
    
    // Convert string inputs to numbers
    const numFills: Record<string, number> = {};
    const numPrices: Record<string, number> = {};
    
    tickersWithData.forEach(t => {
      numFills[t] = parseInt(fills[t]);
      numPrices[t] = parseFloat(prices[t]);
    });
    
    setValidationError(null);
    onSave(numFills, numPrices, note);
    onClose();
  };

  const hasAutoFillData = Object.keys(tradeSuggestions).length > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
       <div className="bg-surface border border-border rounded-2xl w-full max-w-md shadow-2xl p-6 space-y-5 max-h-[90vh] overflow-y-auto">
          <div className="flex justify-between items-center">
             <div className="flex items-center gap-2">
               <h3 className="text-lg font-bold text-fg font-sans">실행 기록</h3>
               {hasAutoFillData && autoFilled && (
                 <span className="flex items-center gap-1 text-[10px] bg-positive/10 text-positive px-2 py-0.5 rounded-full font-sans">
                   <Sparkles size={10} />
                   자동 입력됨
                 </span>
               )}
             </div>
             <button onClick={onClose} className="text-muted hover:text-fg transition-colors">
                <X size={20} />
             </button>
          </div>

          <div className="space-y-4">
             <div className="p-3 bg-inset rounded-lg border border-border text-sm text-muted font-sans">
                기준일: <span className="text-fg font-mono ml-2">{executionDateLabel}</span> (KST)
             </div>

             {/* Validation Error */}
             {validationError && (
               <div className="flex items-center gap-2 p-3 bg-negative/10 text-negative text-sm rounded-lg border border-negative/20">
                 <AlertCircle size={16} />
                 {validationError}
               </div>
             )}

             <div className="space-y-4">
                {uniqueTickers.map(ticker => {
                  const suggestion = tradeSuggestions[ticker];
                  return (
                    <div key={ticker} className="p-3 bg-inset/50 rounded-lg border border-border space-y-2">
                       <div className="flex items-center justify-between">
                         <span className="text-sm font-bold text-fg font-sans">{ticker}</span>
                         {suggestion && (
                           <span className={`text-[10px] px-1.5 py-0.5 rounded font-sans ${
                             suggestion.action === "BUY" ? "bg-positive/10 text-positive" : "bg-negative/10 text-negative"
                           }`}>
                             예상: {suggestion.action} {suggestion.shares}주
                           </span>
                         )}
                       </div>
                       
                       <div className="grid grid-cols-2 gap-2">
                         {/* Shares Input */}
                         <div className="space-y-1">
                           <label className="text-[10px] font-medium text-muted font-sans">체결 수량</label>
                           <input 
                             type="number" 
                             className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-fg font-mono text-sm focus:border-positive focus:outline-none focus:ring-1 focus:ring-positive placeholder-muted"
                             placeholder="0"
                             value={fills[ticker] || ""}
                             onChange={(e) => setFills(prev => ({...prev, [ticker]: e.target.value}))}
                           />
                         </div>
                         
                         {/* Price Input */}
                         <div className="space-y-1">
                           <label className="text-[10px] font-medium text-muted font-sans">
                             체결가 <span className="text-negative">*</span>
                           </label>
                           <div className="relative">
                             <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted text-sm">$</span>
                             <input 
                               type="number" 
                               step="0.01"
                               className="w-full bg-surface border border-border rounded-lg pl-7 pr-3 py-2 text-fg font-mono text-sm focus:border-positive focus:outline-none focus:ring-1 focus:ring-positive placeholder-muted"
                               placeholder="0.00"
                               value={prices[ticker] || ""}
                               onChange={(e) => setPrices(prev => ({...prev, [ticker]: e.target.value}))}
                             />
                           </div>
                         </div>
                       </div>
                    </div>
                  );
                })}
            </div>

            <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-muted font-sans">메모 (선택)</label>
                <textarea 
                   className="bg-inset border border-border rounded-lg px-3 py-2 text-fg text-sm font-sans focus:border-positive focus:outline-none focus:ring-1 focus:ring-positive placeholder-muted resize-none h-16"
                   placeholder="예: 장 마감 직전 체결..."
                   value={note}
                   onChange={(e) => setNote(e.target.value)}
                />
            </div>
          </div>

          <button 
             onClick={handleSave}
             className="w-full py-3 bg-positive hover:bg-positive/90 text-white rounded-lg font-bold font-sans flex items-center justify-center gap-2 transition-colors"
          >
             <Check size={18} />
             기록 저장
          </button>
       </div>
    </div>
  );
}
