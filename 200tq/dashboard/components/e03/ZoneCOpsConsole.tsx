"use client";

import { useState } from "react";
import { E03ViewModel, TradeLine } from "../../lib/ops/e03/types";
import { Copy, Lock, Info, ExternalLink, Check, Download } from "lucide-react";
import { formatTradeForClipboard, getHumanReadableReason } from "../../lib/ops/e03/formatters";
import Toast from "./Toast";
import RecordModal from "./RecordModal";
import { saveRecord, saveRecordToCsv, ManualRecord } from "../../lib/ops/e03/storage";

interface ZoneCOpsConsoleProps {
  vm: E03ViewModel;
  onRecordSuccess?: () => void; // Callback to refresh parent
}

function TradeRow({ item, privacyMode }: { item: TradeLine, privacyMode: boolean }) {
  const isBuy = item.action === "BUY";
  const isSell = item.action === "SELL";
  const displayShares = privacyMode ? "***" : (item.shares > 0 ? `${item.shares.toLocaleString()} 주` : "-");

  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-3 border-b border-neutral-800 last:border-0 text-sm gap-2">
      <div className="flex items-center gap-3">
        <span className={`font-bold w-12 font-sans ${
          isBuy ? "text-positive" : isSell ? "text-negative" : "text-neutral-400"
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
  
  const [toastMsg, setToastMsg] = useState<string | null>(null);
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
         : (mode === "NUMBERS" ? "숫자만 복사됨" : `${vm.expectedTrades.length}개 주문 복사됨`);
       setToastMsg(`복사 완료: ${content}`);
    } catch (err) {
       // Fallback for old browsers or non-secure contexts
       const ta = document.createElement('textarea');
       ta.value = text;
       document.body.appendChild(ta);
       ta.select();
       document.execCommand('copy');
       document.body.removeChild(ta);
       setToastMsg("복사 완료");
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
    
    // Save to CSV file
    const csvResult = await saveRecordToCsv(vm.executionDateLabel, record);
    if (csvResult.success) {
      setToastMsg("기록 완료 (localStorage + CSV)");
    } else {
      setToastMsg("기록 완료 (CSV 저장 실패)");
    }
    
    onRecordSuccess?.();
  };

  return (
    <section className="grid grid-cols-1 md:grid-cols-2 gap-6 relative">
      {toastMsg && <Toast message={toastMsg} onClose={() => setToastMsg(null)} />}
      
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
          현실 (기록)
          <span className="text-[10px] bg-inset px-1.5 py-0.5 rounded text-muted">READ-ONLY</span>
          <a 
            href="/api/record" 
            download="execution_records.csv"
            className="text-[10px] bg-inset hover:bg-positive/20 px-1.5 py-0.5 rounded text-muted hover:text-positive transition-colors flex items-center gap-1"
            title="CSV 다운로드"
          >
            <Download size={10} />
            CSV
          </a>
        </h3>
        <div className="p-4 rounded-xl bg-surface h-full min-h-[200px] flex items-center justify-center text-sm shadow-sm">
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
                 className="px-4 py-2 bg-positive/10 hover:bg-positive/20 text-positive rounded-lg font-medium font-sans text-sm transition-colors flex items-center gap-2"
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
            <h3 className="text-lg font-bold font-sans text-blue-400 flex flex-col lg:flex-row lg:items-center justify-between gap-2">
               <div className="flex items-center gap-2">
                  전략 (예상)
                  <span className="text-[10px] bg-blue-500/10 px-1.5 py-0.5 rounded text-blue-400">Execution Checklist</span>
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
                     const dist = port.derived.stopLossDist || -0.2;
                     const stopPrice = port.derived.stopLossLevel;
                     
                     return (
                        <div className="text-[10px] font-mono text-red-400 bg-red-950/20 px-2 py-0.5 rounded border border-red-900/30 flex items-center gap-2 w-fit">
                           <span>STOP(-20%): ${stopPrice.toFixed(2)}</span>
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
               <div className="bg-surface border border-neutral-800 rounded-lg p-2.5 text-xs font-mono grid grid-cols-2 gap-4">
                  {/* Before */}
                  <div className="space-y-1">
                     <div className="text-muted text-[10px] uppercase tracking-wider">Current</div>
                     <div className="flex justify-between">
                        <span className="text-neutral-400">TQQQ</span>
                        <span className="text-fg font-bold">
                           {vm.portfolio.positions.find(p => p.ticker === "TQQQ")?.qty.toLocaleString() || 0}
                        </span>
                     </div>
                     <div className="flex justify-between">
                        <span className="text-neutral-400">SGOV</span>
                        <span className="text-fg font-bold">
                           {vm.portfolio.positions.find(p => p.ticker === "SGOV")?.qty.toLocaleString() || 0}
                        </span>
                     </div>
                  </div>

                  {/* After (Projected) */}
                  <div className="space-y-1 border-l border-neutral-800 pl-4 relative">
                     <div className="text-blue-400 text-[10px] uppercase tracking-wider">After (Est)</div>
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

                        return (
                           <>
                              <div className="flex justify-between">
                                 <span className="text-neutral-500">TQQQ</span>
                                 <span className="text-blue-100 font-bold">{tqqqAfter.toLocaleString()}</span>
                              </div>
                              <div className="flex justify-between">
                                 <span className="text-neutral-500">SGOV</span>
                                 <span className="text-blue-100 font-bold">{sgovAfter.toLocaleString()}</span>
                              </div>
                           </>
                        );
                     })()}
                  </div>
               </div>
            )}
         </div>

         <div className="rounded-xl bg-surface p-4 shadow-sm">
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
                         ? "bg-e03-surface text-neutral-600 cursor-not-allowed border border-e03-border" 
                         : "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20"
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
                         ? "bg-surface text-muted cursor-not-allowed w-full md:w-auto" 
                         : "bg-surface hover:bg-inset text-muted ring-1 ring-inset w-full md:w-auto"
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
                    <span key={r} className="text-xs text-red-400 flex items-center gap-1.5">
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
                        ? "bg-red-600 hover:bg-red-500 text-white animate-pulse-slow shadow-red-900/20" 
                        : "bg-surface text-muted hover:bg-inset"
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
