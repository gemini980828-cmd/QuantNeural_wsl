"use client";

import { useState } from "react";
import { E03ViewModel, TradeLine } from "../../lib/ops/e03/types";
import { Copy, Lock, Info, ExternalLink, Check, Download } from "lucide-react";
import { formatTradeForClipboard, getHumanReadableReason } from "../../lib/ops/e03/formatters";
import { useToast } from "@/lib/stores/toast-store";
import RecordModal from "./RecordModal";
import { saveRecord, saveRecordToSupabase, ManualRecord } from "../../lib/ops/e03/storage";

interface ZoneCOpsConsoleProps {
  vm: E03ViewModel;
  onRecordSuccess?: () => void; // Callback to refresh parent
}

function TradeRow({ item, privacyMode }: { item: TradeLine, privacyMode: boolean }) {
  const isBuy = item.action === "BUY";
  const isSell = item.action === "SELL";
  const displayShares = privacyMode ? "***" : (item.shares > 0 ? `${item.shares.toLocaleString()} 주` : "-");

  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-3 border-b border-border last:border-0 text-sm gap-2">
      <div className="flex items-center gap-3">
        <span className={`font-bold w-12 font-sans ${
          isBuy ? "text-positive" : isSell ? "text-negative" : "text-muted"
        }`}>
          {item.action}
        </span>
        <span className="font-medium text-fg font-sans">{item.ticker}</span>
        <span className="text-fg font-mono">
           {displayShares}
        </span>
      </div>
      
      {/* Note: Full text, no truncation - critical for trading decisions */}
      {item.note && (
        <div className="text-xs text-muted font-sans leading-relaxed sm:text-right sm:max-w-[250px]">
          {item.note}
        </div>
      )}
    </div>
  );
}

export default function ZoneCOpsConsole({ vm, onRecordSuccess }: ZoneCOpsConsoleProps) {
  const isNoAction = vm.executionState === "NO_ACTION" || 
     (vm.expectedTrades.length === 0) || 
     (vm.expectedTrades.length === 1 && vm.expectedTrades[0].action === "HOLD");

  const highlightRecord = vm.primaryCtaLabel === "기록 필요";
  const stopLossLabel = vm.emergencyTqqqThreshold ?? -15;
  
  const toast = useToast();
  const [isRecordModalOpen, setIsRecordModalOpen] = useState(false);

  // Copy Handler
  const handleCopy = async (mode: "FULL" | "NUMBERS") => {
    if (vm.copyLock.locked) return;
    
    const text = formatTradeForClipboard(vm.expectedTrades, mode);
    try {
       await navigator.clipboard.writeText(text);
       // Privacy mode: do not reveal numbers in toast
       const content = vm.privacyMode 
         ? "클립보드에 복사됨" 
         : (mode === "NUMBERS" ? "숫자만 복사됨" : vm.expectedTrades.map(t => `${t.ticker} ${t.action === 'SELL' ? '-' : '+'}${Math.round(t.shares)}`).join(', '));
       toast.success(`복사 완료: ${content}`);
    } catch (err) {
       // Fallback for old browsers or non-secure contexts
       const ta = document.createElement('textarea');
       ta.value = text;
       document.body.appendChild(ta);
       ta.select();
       document.execCommand('copy');
       document.body.removeChild(ta);
       toast.success("복사 완료");
    }
  };

  const handleRecordSave = async (fills: Record<string, number>, prices: Record<string, number>, note?: string) => {
    const record: ManualRecord = {
      recordedAt: new Date().toISOString(),
      fills,
      prices,
      note
    };
    
    // Save to localStorage
    saveRecord(vm.executionDateLabel, record);
    
    // Save to Supabase
    const result = await saveRecordToSupabase(vm.executionDateLabel, record, vm.expectedTrades, vm.inputPrices);
    if (result.success) {
      toast.success("기록 완료 (localStorage + Supabase)");
    } else {
      toast.error("기록 완료 (Supabase 저장 실패)");
    }
    
    onRecordSuccess?.();
  };

  return (
    <section className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start relative">
      
      <RecordModal 
         isOpen={isRecordModalOpen} 
         onClose={() => setIsRecordModalOpen(false)}
         onSave={handleRecordSave}
         executionDateLabel={vm.executionDateLabel}
         expectedTrades={vm.expectedTrades}
      />

      {/* LEFT: Reality */}
      <div className="space-y-3">
          <h3 className="text-lg font-bold font-sans text-muted flex items-center gap-2">
          실제 체결(수동)
          <span className="text-[11px] bg-inset px-1.5 py-0.5 rounded text-muted">READ-ONLY</span>
          <a 
            href="/api/record" 
            download="execution_records.csv"
            className="text-[11px] bg-inset hover:bg-positive/20 px-1.5 py-0.5 rounded text-muted hover:text-positive transition-colors flex items-center gap-1"
            title="CSV 다운로드"
          >
            <Download size={10} />
            CSV
          </a>
        </h3>
        <div className="p-4 rounded-xl bg-surface border border-border min-h-[200px] flex items-center justify-center text-sm shadow-sm">
           {vm.executionState === "RECORDED" ? (
             <div className="flex flex-col items-center gap-2 text-positive">
               <Check size={32} />
               <span className="font-bold font-sans">기록 완료</span>
               <span className="text-xs text-muted font-sans">localStorage + CSV</span>
             </div>
           ) : (
             <div className="flex flex-col items-center gap-3 text-center">
               <div className="text-muted text-sm font-sans">아직 기록되지 않음</div>
               <button
                 onClick={() => setIsRecordModalOpen(true)}
                 className="px-4 py-2 bg-positive-tint hover:bg-positive/20 text-positive rounded-lg font-medium font-sans text-sm transition-colors flex items-center gap-2"
               >
                 <Check size={16} />
                 첫 실행 기록하기
               </button>
             </div>
           )}
        </div>
      </div>

      {/* RIGHT: Strategy */}
      <div className="space-y-3">
         <div className="flex flex-col gap-2">
            <h3 className="text-lg font-bold font-sans text-info flex flex-col lg:flex-row lg:items-center justify-between gap-2">
               <div className="flex items-center gap-2">
                   예상 주문(자동)
                  <span className="text-[11px] bg-info-tint px-1.5 py-0.5 rounded text-info">Execution Checklist</span>
               </div>
               
               {/* Risk Line (Compact) - Only show if TQQQ expected > 0 */}
               {(() => {
                  // Quick calc for risk visibility
                  const port = vm.portfolio;
                  if (!port) return null;
                  
                  const afterTqqq = (port.positions.find(p => p.ticker === "TQQQ")?.qty || 0) + 
                                    (vm.expectedTrades.find(t => t.ticker === "TQQQ" && t.action === "BUY")?.shares || 0) -
                                    (vm.expectedTrades.find(t => t.ticker === "TQQQ" && t.action === "SELL")?.shares || 0);

                  if (afterTqqq > 0 && port.derived.stopLossLevel) {
                     const dist = port.derived.stopLossDist || -0.15;
                     const stopPrice = port.derived.stopLossLevel;
                     
                     return (
                        <div className="text-[11px] font-mono text-negative bg-negative-tint px-2 py-0.5 rounded border border-negative/20 flex items-center gap-2 w-fit">
                           <span>STOP({stopLossLabel}%): ${stopPrice.toFixed(2)}</span>
                           <span className="opacity-50">|</span>
                           <span>Dist: {(dist * 100).toFixed(1)}%</span>
                        </div>
                     );
                  }
                  return null;
               })()}
            </h3>

            {/* Position Impact Analysis (Before -> After) */}
            {vm.portfolio && (
               <div className="bg-surface border border-border rounded-lg p-2.5 text-xs font-mono grid grid-cols-2 gap-4">
                  {/* Before */}
                  <div className="space-y-1">
                     <div className="text-muted text-[11px] uppercase tracking-wider">Current</div>
                     <div className="flex justify-between">
                         <span className="text-muted">TQQQ</span>
                         <span className="text-fg font-bold">
                            {vm.portfolio.positions.find(p => p.ticker === "TQQQ")?.qty.toLocaleString() || 0}
                         </span>
                      </div>
                      <div className="flex justify-between">
                         <span className="text-muted">SGOV</span>
                        <span className="text-fg font-bold">
                           {vm.portfolio.positions.find(p => p.ticker === "SGOV")?.qty.toLocaleString() || 0}
                        </span>
                     </div>
                  </div>

                  {/* After (Projected) */}
                   <div className="space-y-1 border-l border-border pl-4 relative">
                     <div className="text-info text-[11px] uppercase tracking-wider">After (Est)</div>
                     {(() => {
                        const p = vm.portfolio!.positions;
                        const trades = vm.expectedTrades;
                        
                        const getQty = (ticker: string) => {
                           const curr = p.find(x => x.ticker === ticker)?.qty || 0;
                           const buy = trades.find(x => x.ticker === ticker && x.action === "BUY")?.shares || 0;
                           const sell = trades.find(x => x.ticker === ticker && x.action === "SELL")?.shares || 0;
                           return curr + buy - sell;
                        }

                        const tqqqAfter = getQty("TQQQ");
                        const sgovAfter = getQty("SGOV");
                        const tqqqWeightPct = Math.round(vm.targetTqqqWeight * 100);
                        const sgovWeightPct = 100 - tqqqWeightPct;

                        return (
                           <>
                              <div className="flex justify-between">
                                  <span className="text-muted">TQQQ</span>
                                  <span className="text-info font-bold">{tqqqAfter.toLocaleString()} ({tqqqWeightPct}%)</span>
                                </div>
                                <div className="flex justify-between">
                                   <span className="text-muted">SGOV</span>
                                  <span className="text-info font-bold">{sgovAfter.toLocaleString()} ({sgovWeightPct}%)</span>
                               </div>
                           </>
                        );
                     })()}
                  </div>
               </div>
            )}
         </div>

         <div className="rounded-xl bg-surface border border-border p-4 shadow-sm">
            {vm.strategyState === "ON_CHOPPY" && (
              <div className="mb-3 text-xs text-choppy font-medium">Choppy 구간: TQQQ 70% / SGOV 30%</div>
            )}
            {vm.strategyState === "EMERGENCY" && (
              <div className="mb-3 text-xs text-negative font-medium animate-pulse">Emergency Exit: TQQQ 10% / SGOV 90%</div>
            )}
            {vm.strategyState === "ON" && (
              <div className="mb-3 text-xs text-positive font-medium">ON: TQQQ 100%</div>
            )}
            {vm.strategyState === "OFF10" && (
              <div className="mb-3 text-xs text-muted font-medium">OFF: TQQQ 10% / SGOV 90%</div>
            )}
             {isNoAction ? (
                <div className="flex flex-col items-center justify-center py-8 text-muted gap-2">
                  <Info size={24} />
                  <span>오늘 주문 없음 (NO_ACTION)</span>
               </div>
            ) : (
              <div>
                 {vm.expectedTrades.map((trade, idx) => (
                    <TradeRow key={idx} item={trade} privacyMode={vm.privacyMode} />
                 ))}
              </div>
            )}
         </div>

         {/* Actions */}
         <div className="space-y-2 pt-2">
            {isNoAction ? (
               // P0-2: Info Card for NO_ACTION
               <div className="flex items-center justify-between p-4 rounded-lg bg-surface/50 text-muted text-sm">
                  <div className="flex flex-col">
                     <span className="text-xs text-muted mb-0.5">다음 실행일</span>
                     <span className="font-mono text-fg">내일 장 시작 전</span>
                  </div>
                  <div className="flex flex-col items-end">
                     <span className="text-xs text-muted mb-0.5">현재 상태</span>
                     <span className="font-bold text-muted">NO ACTION</span>
                  </div>
               </div>
            ) : (
               // Normal Actions
               <div className="flex flex-col md:flex-row gap-2">
                   <button
                     disabled={vm.copyLock.locked}
                     onClick={() => handleCopy("FULL")}
                      className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-all ${
                        vm.copyLock.locked 
                           ? "bg-e03-surface text-muted cursor-not-allowed border border-e03-border"
                          : "bg-info hover:bg-info/80 text-white shadow-lg shadow-info/20 border border-info/30"
                      }`}
                   >
                     {vm.copyLock.locked ? <Lock size={16} /> : <Copy size={16} />}
                     주문 복사
                   </button>
                   
                   <button
                     disabled={vm.copyLock.locked}
                     onClick={() => handleCopy("NUMBERS")}
                      className={`px-4 py-3 rounded-lg font-medium text-sm transition-all ${
                         vm.copyLock.locked
                          ? "bg-surface text-muted cursor-not-allowed border border-border w-full md:w-auto" 
                          : "bg-surface hover:bg-inset text-muted border border-border w-full md:w-auto"
                      }`}
                   >
                     숫자만
                   </button>
               </div>
            )}

            {/* Lock Reasons */}
            {vm.copyLock.locked && !isNoAction && (
              <div className="flex flex-col gap-1 px-1 mt-1">
                 {vm.copyLock.reasons.map(r => (
                    <span key={r} className="text-xs text-negative flex items-center gap-1.5">
                      <Lock size={10} className="shrink-0" /> 
                      {getHumanReadableReason(r)}
                    </span>
                 ))}
              </div>
            )}

            {/* Record Execution */}
            <div className={`mt-4 pt-4 border-t border-e03-border/50 ${highlightRecord ? 'opacity-100' : 'opacity-70'}`}>
                <button 
                  onClick={() => setIsRecordModalOpen(true)}
                   className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg text-sm font-bold transition-all shadow-md ${
                      highlightRecord 
                         ? "bg-negative hover:bg-negative/80 text-white animate-pulse-slow shadow-red-900/20" 
                         : "bg-surface text-muted hover:bg-inset border border-border"
                   }`}
                >
                   <ExternalLink size={14} />
                   {vm.primaryCtaLabel === "기록 필요" ? "실행 결과 기록하기 (필수)" : "실행 결과 기록 (Manual)"}
                </button>
            </div>
         </div>
      </div>
    </section>
  );
}
