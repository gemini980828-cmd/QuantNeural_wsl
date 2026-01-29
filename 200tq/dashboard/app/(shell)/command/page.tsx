"use client";

import Link from "next/link";
import { useEffect, useState, useMemo, useCallback } from "react";
import { ScenarioId } from "../../../lib/ops/e03/mock";
import { E03RawInputs, buildViewModel } from "../../../lib/ops/e03/buildViewModel";
import { loadToggle, saveToggle, loadRecord, STORAGE_KEYS } from "../../../lib/ops/e03/storage";
import { Period, PricePoint } from "../../../lib/ops/e03/mockPrices";
import { summarizePerf, summarizePerfWithDateRange, DateRange, PriceData } from "../../../lib/ops/e03/performance";
import { getInputs, DataSourceMode } from "../../../lib/ops/dataSource";
import { useDataSource, useDevScenario, useViewMode, useSettingsStore } from "../../../lib/stores/settings-store";
import ZoneAHeader from "../../../components/e03/ZoneAHeader";
import ZoneBSignalCore from "../../../components/e03/ZoneBSignalCore";
import ZoneCOpsConsole from "../../../components/e03/ZoneCOpsConsole";
import ZoneDIntelLab from "../../../components/e03/ZoneDIntelLab";
import PortfolioSummaryStrip from "../../../components/portfolio/PortfolioSummaryStrip";
import SimpleView from "../../../components/e03/SimpleView";
import { AlertTriangle, RefreshCw, Wallet } from "lucide-react";

export default function CommandPage({
  searchParams,
}: {
  searchParams: { scenario?: string };
}) {
  const currentScenario = (searchParams.scenario as ScenarioId) || "fresh_normal";
  const dataSource = useDataSource();
  const devScenarioEnabled = useDevScenario();
  const viewMode = useViewMode();
  const settingsStore = useSettingsStore();

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
  const [unresolvedAlerts, setUnresolvedAlerts] = useState(0);
  
  const [portfolioState, setPortfolioState] = useState<{ tqqq: number; sgov: number } | null>(null);
  
  // Macro data for SimpleView
  type ColorTone = 'ok' | 'action' | 'danger';
  interface MacroData {
    vix: { value: number | null; color: ColorTone; change: number | null };
    fng: { value: number | null; label: string; color: ColorTone; change: number | null };
    treasury: { value: number | null; change: number | null };
    dxy: { value: number | null; change: number | null };
    nq: { value: number | null; change: number | null };
    usdkrw: { value: number | null; change: number | null };
    updatedAt: string;
  }
  const [macroData, setMacroData] = useState<MacroData | null>(null);
  const [macroLoading, setMacroLoading] = useState(true);
  
  // Custom date range for backtest
  const getDefaultDateRange = (): DateRange => {
    const end = new Date();
    const start = new Date();
    start.setFullYear(end.getFullYear() - 1); // Default 1Y
    return {
      start: start.toISOString().split('T')[0],
      end: end.toISOString().split('T')[0],
    };
  };
  const [customDateRange, setCustomDateRange] = useState<DateRange>(getDefaultDateRange);
  
  // Real price data for backtest (fetched from API)
  const [realPriceData, setRealPriceData] = useState<PriceData | null>(null);
  const [priceLoading, setPriceLoading] = useState(false);
  
  // Fetch real price data when date range changes
  useEffect(() => {
    const fetchRealPrices = async () => {
      setPriceLoading(true);
      try {
        // Need extra history for SMA calculations (SMA200 needs 200 trading days before start)
        // Add 300 calendar days buffer to ensure enough data
        const bufferStart = new Date(customDateRange.start);
        bufferStart.setDate(bufferStart.getDate() - 300);
        const bufferedStartDate = bufferStart.toISOString().split('T')[0];
        
        // Fetch all required symbols in parallel with buffered start date
        const [qqqRes, tqqqRes, sgovRes] = await Promise.all([
          fetch(`/api/prices/history?symbol=QQQ&from=${bufferedStartDate}&to=${customDateRange.end}&limit=4000`),
          fetch(`/api/prices/history?symbol=TQQQ&from=${bufferedStartDate}&to=${customDateRange.end}&limit=4000`),
          fetch(`/api/prices/history?symbol=SGOV&from=${bufferedStartDate}&to=${customDateRange.end}&limit=4000`),
        ]);
        
        const [qqqData, tqqqData, sgovData] = await Promise.all([
          qqqRes.json(),
          tqqqRes.json(),
          sgovRes.json(),
        ]);
        
        // Convert API format to PriceData format
        const priceData: PriceData = {
          QQQ: qqqData.bars?.map((b: { date: string; close: number }) => ({ date: b.date, close: b.close })) || [],
          TQQQ: tqqqData.bars?.map((b: { date: string; close: number }) => ({ date: b.date, close: b.close })) || [],
          SGOV: sgovData.bars?.map((b: { date: string; close: number }) => ({ date: b.date, close: b.close })) || [],
        };
        
        setRealPriceData(priceData);
      } catch (e) {
        console.error('Failed to fetch real prices:', e);
        setRealPriceData(null); // Fall back to mock
      } finally {
        setPriceLoading(false);
      }
    };
    
    fetchRealPrices();
  }, [customDateRange]);

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
      
      // Fetch unresolved alerts count
      try {
        const notifRes = await fetch("/api/notifications/list?resolved=false&limit=1");
        const notifData = await notifRes.json();
        if (notifData.unresolvedCounts) {
          setUnresolvedAlerts(notifData.unresolvedCounts.total || 0);
        }
      } catch (e) {
        console.warn("Failed to fetch notification counts", e);
      }
      
      try {
        const portfolioRes = await fetch("/api/portfolio/state");
        const portfolioData = await portfolioRes.json();
        if (portfolioData.state) {
          setPortfolioState({
            tqqq: portfolioData.state.tqqq_shares || 0,
            sgov: portfolioData.state.sgov_shares || 0,
          });
        }
      } catch (e) {
        console.warn("Failed to fetch portfolio state", e);
      }
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

  // Fetch macro data for SimpleView
  useEffect(() => {
    const fetchMacro = async () => {
      setMacroLoading(true);
      try {
        const res = await fetch("/api/macro");
        const data = await res.json();
        setMacroData(data);
      } catch (e) {
        console.warn("Failed to fetch macro data", e);
      } finally {
        setMacroLoading(false);
      }
    };
    fetchMacro();
  }, []);

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

  // Check if record exists for today (Supabase + localStorage fallback)
  const [hasRecordForToday, setHasRecordForToday] = useState(false);
  
  useEffect(() => {
    if (!vm) return;
    
    // Check localStorage first (fast)
    const localRecord = loadRecord(vm.executionDateLabel);
    if (localRecord) {
      setHasRecordForToday(true);
      return;
    }
    
    // Then check Supabase
    const checkSupabaseRecord = async () => {
      try {
        const res = await fetch(`/api/record?date=${vm.executionDateLabel}`);
        const data = await res.json();
        setHasRecordForToday(!data.empty && data.executed);
      } catch {
        setHasRecordForToday(false);
      }
    };
    
    checkSupabaseRecord();
  }, [vm?.executionDateLabel, recordUpdateTrigger]);

  // Apply record override
  const vmWithRecord = useMemo(() => {
    if (!vm) return null;
    
    if (hasRecordForToday) {
      vm.executionState = "RECORDED";
      vm.executionBadge.label = "Exec: RECORDED";
      vm.executionBadge.tone = "neutral";
      if (vm.primaryCtaLabel === "기록 필요") {
        vm.primaryCtaLabel = "주문 복사";
      }
    }
    return vm;
  }, [vm, hasRecordForToday]);

  // Performance Summary - now using custom date range with real price data
  const perfSummary = useMemo(
    () => summarizePerfWithDateRange(customDateRange, startCapital, realPriceData || undefined), 
    [customDateRange, startCapital, realPriceData]
  );
  
  // Handler for date range changes from ZoneBSignalCore
  const handleDateRangeChange = (start: string, end: string) => {
    setCustomDateRange({ start, end });
  };

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
        <div className="flex items-center gap-2 text-xs mb-4 flex-wrap">
          <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded-full font-medium">
            REAL DATA
          </span>
          {portfolioState && (
            <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded-full font-medium flex items-center gap-1">
              <Wallet className="w-3 h-3" />
              TQQQ {portfolioState.tqqq}주 / SGOV {portfolioState.sgov}주
            </span>
          )}
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
            <span className="uppercase font-bold tracking-widest text-muted dark:text-muted select-none">Dev Scenario</span>
            <div className="flex gap-1">
              {scenarios.map((id) => (
                <Link
                  key={id}
                  href={`/command?scenario=${id}`}
                  className={`px-2 py-0.5 rounded-full transition-all ${
                    currentScenario === id 
                      ? "bg-neutral-200 dark:bg-surface text-neutral-900 dark:text-neutral-200 font-medium shadow-sm" 
                      : "text-muted dark:text-muted hover:text-neutral-900 dark:hover:text-fg hover:bg-neutral-200/50 dark:hover:bg-inset/50"
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
             unresolvedAlerts={unresolvedAlerts}
          />
          
          {viewMode === 'simple' ? (
            <SimpleView 
              vm={vmWithRecord}
              macroData={macroData}
              macroLoading={macroLoading}
              onSwitchToPro={() => settingsStore.setSetting('viewMode', 'pro')}
            />
          ) : (
            <div>
              <ZoneBSignalCore 
                vm={vmWithRecord} 
                selectedPeriod={selectedPeriod}
                perfSummary={perfSummary}
                startCapital={startCapital}
                onCapitalChange={setStartCapital}
                realPrices={rawInputs?.inputPrices}
                onDateRangeChange={handleDateRangeChange}
              />
              
              <div className="h-8" />

              <PortfolioSummaryStrip portfolio={vmWithRecord.portfolio} />
              
              <div className="h-4" />
              
              <ZoneCOpsConsole 
                 vm={vmWithRecord} 
                 onRecordSuccess={() => setRecordUpdateTrigger(p => p + 1)} 
              />
              
              <div className="h-24" />
              
              <ZoneDIntelLab 
                vm={vmWithRecord} 
                selectedPeriod={selectedPeriod}
                onPeriodChange={setSelectedPeriod}
                startCapital={startCapital}
              />
            </div>
          )}
        </div>
      </div>

      {/* Collapsed Debug View */}
      <details className="mt-12 border-t border-border pt-4">
        <summary className="text-xs text-muted cursor-pointer hover:text-fg">
           Debug JSON View
        </summary>
        <pre suppressHydrationWarning className="mt-4 p-4 bg-black rounded-lg border border-border text-green-900 text-[10px] overflow-auto max-h-[400px] font-mono whitespace-pre">
           {JSON.stringify({ 
             ...vmWithRecord,
             dataSource,
           }, null, 2)}
        </pre>
      </details>
    </div>
  );
}
