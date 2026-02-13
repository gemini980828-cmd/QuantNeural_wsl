"use client";

import { E03ViewModel } from "../../lib/ops/e03/types";
import { ChevronRight } from "lucide-react";
import { Period } from "../../lib/ops/e03/mockPrices";
import MarketChartPanel from "./MarketChartPanel";

interface ZoneDIntelLabProps {
  vm: E03ViewModel;
  selectedPeriod: Period;
  onPeriodChange: (p: Period) => void;
  startCapital?: number;
}

export default function ZoneDIntelLab({ vm, selectedPeriod, onPeriodChange, startCapital = 10000000 }: ZoneDIntelLabProps) {
  const e03State = vm.strategyState === "OFF10" ? "OFF10" : "ON";
  
  return (
    <details className="group border-t border-border pt-8 mt-0">
      <summary className="list-none cursor-pointer flex items-center justify-between text-muted hover:text-fg transition-colors">
         <span className="text-lg font-bold font-sans">차트 분석</span>
         <ChevronRight size={16} className="group-open:rotate-90 transition-transform" />
      </summary>
      
      <div className="pt-4 pb-8 space-y-6">
         {/* D1: Market Charts */}
         <div>
           <MarketChartPanel 
             selectedPeriod={selectedPeriod}
             onPeriodChange={onPeriodChange}
             e03State={e03State}
             privacyMode={vm.privacyMode}
             startCapital={startCapital}
           />
         </div>
         
         {/* D2: Advanced Info (collapsed) */}
         <details className="text-xs text-muted">
            <summary className="cursor-pointer font-sans text-muted hover:text-fg text-[11px] uppercase tracking-wider">
             고급 정보
           </summary>
           <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-3">
               <div className="p-5 bg-surface rounded-xl border border-border shadow-sm">
                 <h4 className="font-semibold font-sans text-muted mb-2">Currency</h4>
                 <p className="text-fg">USD/KRW: 1,380.0 (Mock)</p>
              </div>
               <div className="p-5 bg-surface rounded-xl border border-border shadow-sm">
                 <h4 className="font-semibold font-sans text-muted mb-2">Data Source</h4>
                 <p className="text-fg">Mock Price Series (5Y)</p>
              </div>
           </div>
         </details>
      </div>
    </details>
  );
}

