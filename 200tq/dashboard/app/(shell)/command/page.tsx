"use client";

import Link from "next/link";
import { useEffect, useState, useMemo, useCallback } from "react";
import { ScenarioId } from "../../../lib/ops/e03/mock";
import { E03RawInputs, buildViewModel } from "../../../lib/ops/e03/buildViewModel";
import { loadToggle, saveToggle, loadRecord, STORAGE_KEYS } from "../../../lib/ops/e03/storage";
import { Period } from "../../../lib/ops/e03/mockPrices";
import { summarizePerf } from "../../../lib/ops/e03/performance";
import { getInputs, DataSourceMode } from "../../../lib/ops/dataSource";
import { useDataSource, useDevScenario } from "../../../lib/stores/settings-store";
import ZoneAHeader from "../../../components/e03/ZoneAHeader";
import ZoneBSignalCore from "../../../components/e03/ZoneBSignalCore";
import ZoneCOpsConsole from "../../../components/e03/ZoneCOpsConsole";
import ZoneDIntelLab from "../../../components/e03/ZoneDIntelLab";
import PortfolioSummaryStrip from "../../../components/portfolio/PortfolioSummaryStrip";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function CommandPage({
  searchParams,
}: {
  searchParams: { scenario?: string };
}) {
  const currentScenario = (searchParams.scenario as ScenarioId) || "fresh_normal";
  const dataSource = useDataSource();
  const devScenarioEnabled = useDevScenario();

  // Data state
  const [rawInputs, setRawInputs] = useState<E03RawInputs | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Client State
  const [simMode, setSimMode] = useState(false);
  const [privMode, setPrivMode] = useState(false);
  const [recordUpdateTrigger, setRecordUpdateTrigger] = useState(0);
  const [selectedPeriod, setSelectedPeriod] = useState<Period>("1Y");
  const [startCapital, setStartCapital] = useState(10000000);

  // Load data based on dataSource
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const inputs = await getInputs({ 
        dataSource, 
        scenario: currentScenario 
      });
      setRawInputs(inputs);
      setStartCapital(inputs.inputTotalValueKrw || 10000000);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      // Fallback to MOCK on error
      const fallback = await getInputs({ dataSource: "MOCK", scenario: currentScenario });
      setRawInputs(fallback);
    } finally {
      setLoading(false);
    }
  }, [dataSource, currentScenario]);

  // Initial load
  useEffect(() => {
    loadData();
  }, [loadData]);

  // Load persisted toggles on mount
  useEffect(() => {
    setSimMode(loadToggle(STORAGE_KEYS.SIMULATION_MODE, false));
    setPrivMode(loadToggle(STORAGE_KEYS.PRIVACY_MODE, false));
  }, []);

  // Handlers
  const toggleSim = (val: boolean) => {
    setSimMode(val);
    saveToggle(STORAGE_KEYS.SIMULATION_MODE, val);
  };
  const togglePriv = (val: boolean) => {
    setPrivMode(val);
    saveToggle(STORAGE_KEYS.PRIVACY_MODE, val);
  };

  // Build ViewModel
  const vm = useMemo(() => {
    if (!rawInputs) return null;
    
    const inputsWithToggles = {
      ...rawInputs,
      simulationMode: simMode,
      privacyMode: privMode,
    };
    
    return buildViewModel(inputsWithToggles);
  }, [rawInputs, simMode, privMode]);

  // Apply record override
  const vmWithRecord = useMemo(() => {
    if (!vm) return null;
    
    const record = typeof window !== "undefined" ? loadRecord(vm.executionDateLabel) : null;
    if (record) {
      vm.executionState = "RECORDED";
      vm.executionBadge.label = "Exec: RECORDED";
      vm.executionBadge.tone = "neutral";
      if (vm.primaryCtaLabel === "기록 필요") {
        vm.primaryCtaLabel = "주문 복사";
      }
    }
    return vm;
  }, [vm, recordUpdateTrigger]);

  // Performance Summary
  const perfSummary = useMemo(() => summarizePerf(selectedPeriod, startCapital), [selectedPeriod, startCapital]);

  const scenarios: ScenarioId[] = [
    "fresh_normal", 
    "stale_or_closed", 
    "soft_alert", 
    "hard_confirmed"
  ];

  const scenarioLabels: Record<ScenarioId, string> = {
    fresh_normal: "데이터 정상",
    stale_or_closed: "장 마감",
    soft_alert: "긴급 점검",
    hard_confirmed: "비상 확정",
  };

  // Loading state
  if (loading || !vmWithRecord) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 text-muted">
          <RefreshCw className="w-8 h-8 animate-spin" />
          <span className="text-sm">데이터 로딩 중...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-20">
      {/* REAL mode indicator + STALE warning */}
      {dataSource === "REAL" && (
        <div className="flex items-center gap-2 text-xs mb-4">
          <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded-full font-medium">
            REAL DATA
          </span>
          {vmWithRecord.dataState === "STALE" && (
            <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 rounded-full font-medium flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              데이터 지연 - 전일 유지
            </span>
          )}
          {error && (
            <span className="px-2 py-0.5 bg-red-500/20 text-red-400 rounded-full font-medium flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              API 오류 - MOCK으로 대체
            </span>
          )}
        </div>
      )}

      {/* Dev Tools: Scenario Switcher (only in MOCK mode or when devScenario enabled) */}
      {(dataSource === "MOCK" || devScenarioEnabled) && (
        <div className="flex gap-3 text-xs items-center mb-6 pl-1 opacity-70 hover:opacity-100 transition-opacity duration-300">
            <span className="uppercase font-bold tracking-widest text-neutral-500 dark:text-neutral-400 select-none">Dev Scenario</span>
            <div className="flex gap-1">
              {scenarios.map((id) => (
                <Link
                  key={id}
                  href={`/command?scenario=${id}`}
                  className={`px-2 py-0.5 rounded-full transition-all ${
                    currentScenario === id 
                      ? "bg-neutral-200 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-200 font-medium shadow-sm" 
                      : "text-neutral-500 dark:text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-300 hover:bg-neutral-200/50 dark:hover:bg-neutral-900/50"
                  }`}
                >
                  {scenarioLabels[id]}
                </Link>
              ))}
            </div>
        </div>
      )}

      {/* Main Command Center UI */}
      <div className="relative">
        {simMode && (
           <div className="absolute inset-0 pointer-events-none z-0 flex items-center justify-center overflow-hidden opacity-5">
              <div className="text-[15vw] font-black text-amber-500 -rotate-12 select-none whitespace-nowrap">
                 SIMULATION
              </div>
           </div>
        )}

        <div>
          <ZoneAHeader 
             vm={vmWithRecord} 
             onToggleSimulation={toggleSim}
             onTogglePrivacy={togglePriv}
          />
          
          <div>
            <ZoneBSignalCore 
              vm={vmWithRecord} 
              selectedPeriod={selectedPeriod}
              perfSummary={perfSummary}
              startCapital={startCapital}
              onCapitalChange={setStartCapital}
            />
            
            <div className="h-8" /> {/* Spacer */}

            {/* Portfolio Summary Strip */}
            <PortfolioSummaryStrip portfolio={vmWithRecord.portfolio} />
            
            <div className="h-4" /> {/* Spacer */}
            
            <ZoneCOpsConsole 
               vm={vmWithRecord} 
               onRecordSuccess={() => setRecordUpdateTrigger(p => p + 1)} 
            />
            
            <div className="h-24" /> {/* Large Spacer for Zone D separation */}
            
            <ZoneDIntelLab 
              vm={vmWithRecord} 
              selectedPeriod={selectedPeriod}
              onPeriodChange={setSelectedPeriod}
              startCapital={startCapital}
            />
          </div>
        </div>
      </div>

      {/* Collapsed Debug View */}
      <details className="mt-12 border-t border-border pt-4">
        <summary className="text-xs text-muted cursor-pointer hover:text-fg">
           Debug JSON View
        </summary>
        <pre suppressHydrationWarning className="mt-4 p-4 bg-black rounded-lg border border-neutral-800 text-green-900 text-[10px] overflow-auto max-h-[400px] font-mono whitespace-pre">
           {JSON.stringify({ 
             ...vmWithRecord,
             dataSource,
           }, null, 2)}
        </pre>
      </details>
    </div>
  );
}
