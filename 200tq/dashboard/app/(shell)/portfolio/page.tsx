"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { E03RawInputs, buildViewModel } from "../../../lib/ops/e03/buildViewModel";
import { getInputs } from "../../../lib/ops/dataSource";
import { useDataSource } from "../../../lib/stores/settings-store";
import PortfolioSummaryStrip from "../../../components/portfolio/PortfolioSummaryStrip";
import PortfolioPositionsTable from "../../../components/portfolio/PortfolioPositionsTable";
import { Info, History, LayoutGrid, Layers, TrendingUp, RefreshCw, AlertTriangle, Database, Wallet, Camera, Check, CheckCircle2, XCircle } from "lucide-react";

interface TradeRecord {
  id: string;
  execution_date: string;
  ticker: string;
  action: string;
  shares: number;
  price_usd: number;
  note?: string;
}

interface EquityPoint {
  date: string;
  value: number;
}

export default function PortfolioPage() {
  const dataSource = useDataSource();
  
  const [rawInputs, setRawInputs] = useState<E03RawInputs | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [recentTrades, setRecentTrades] = useState<TradeRecord[]>([]);
  const [tradesLoading, setTradesLoading] = useState(false);
  
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [equityLoading, setEquityLoading] = useState(false);

  const [portfolioStateLoading, setPortfolioStateLoading] = useState(true);
  const [portfolioSaving, setPortfolioSaving] = useState(false);
  const [tqqqShares, setTqqqShares] = useState(0);
  const [sgovShares, setSgovShares] = useState(0);
  const [portfolioLastUpdated, setPortfolioLastUpdated] = useState<string | null>(null);
  const [portfolioSaveResult, setPortfolioSaveResult] = useState<{ success: boolean; message: string } | null>(null);
  
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrResult, setOcrResult] = useState<{ success: boolean; message: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load data based on dataSource
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const inputs = await getInputs({ 
        dataSource, 
        scenario: "fresh_normal" 
      });
      setRawInputs(inputs);
    } catch (e) {
      setError(String(e));
      console.error("Failed to load portfolio data:", e);
    } finally {
      setLoading(false);
    }
  }, [dataSource]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (dataSource === "REAL") {
      setTradesLoading(true);
      fetch("/api/records/list?limit=10")
        .then(res => res.json())
        .then(data => {
          if (data.success) setRecentTrades(data.records || []);
        })
        .catch(console.error)
        .finally(() => setTradesLoading(false));

      setEquityLoading(true);
      fetch("/api/portfolio/equity-history")
        .then(res => res.json())
        .then(data => {
          if (data.success) setEquityHistory(data.equity || []);
        })
        .catch(console.error)
        .finally(() => setEquityLoading(false));
    } else {
      setRecentTrades([]);
      setEquityHistory([]);
    }
  }, [dataSource]);

  useEffect(() => {
    fetch("/api/portfolio/state")
      .then(res => res.json())
      .then(data => {
        if (data.state) {
          setTqqqShares(data.state.tqqq_shares || 0);
          setSgovShares(data.state.sgov_shares || 0);
          setPortfolioLastUpdated(data.state.last_updated || null);
        }
      })
      .catch(() => {})
      .finally(() => setPortfolioStateLoading(false));
  }, []);

  const savePortfolioState = async () => {
    setPortfolioSaving(true);
    setPortfolioSaveResult(null);
    try {
      const res = await fetch("/api/portfolio/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tqqq_shares: tqqqShares, sgov_shares: sgovShares }),
      });
      const data = await res.json();
      if (data.success) {
        setPortfolioSaveResult({ success: true, message: "저장 완료" });
        setPortfolioLastUpdated(data.state.last_updated);
      } else {
        setPortfolioSaveResult({ success: false, message: data.error || "저장 실패" });
      }
    } catch (e) {
      setPortfolioSaveResult({ success: false, message: String(e) });
    } finally {
      setPortfolioSaving(false);
    }
  };

  const handleOcrUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setOcrLoading(true);
    setOcrResult(null);
    
    try {
      const formData = new FormData();
      formData.append("image", file);
      
      const res = await fetch("/api/portfolio/ocr", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      if (data.success) {
        setTqqqShares(data.tqqq_shares);
        setSgovShares(data.sgov_shares);
        setOcrResult({ success: true, message: `TQQQ ${data.tqqq_shares}주, SGOV ${data.sgov_shares}주 추출` });
      } else {
        setOcrResult({ success: false, message: data.error || "OCR 분석 실패" });
      }
    } catch (e) {
      setOcrResult({ success: false, message: String(e) });
    } finally {
      setOcrLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const vm = rawInputs ? buildViewModel(rawInputs) : null;
  const portfolio = vm?.portfolio;

  // Loading state
  if (loading) {
    return (
      <div className="p-4">
        <div className="mt-8 text-center text-muted flex flex-col items-center gap-2">
          <RefreshCw className="animate-spin" size={24} />
          <span>포트폴리오 로딩 중...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="p-4">
        <div className="mt-8 text-center text-negative flex flex-col items-center gap-2">
          <AlertTriangle size={24} />
          <span>데이터 로드 실패: {error}</span>
          <button 
            onClick={loadData}
            className="mt-2 px-4 py-2 bg-neutral-800 rounded-lg text-sm hover:bg-neutral-700"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

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
        
        {/* Data Source Indicator */}
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-1 rounded-full ${
            dataSource === "REAL" 
              ? "bg-positive/20 text-positive border border-positive/30" 
              : "bg-amber-900/30 text-amber-400 border border-amber-700/30"
          }`}>
            {dataSource}
          </span>
          <button
            onClick={loadData}
            disabled={loading}
            className="p-1.5 rounded-lg hover:bg-neutral-800 text-muted transition-colors"
            title="새로고침"
          >
            <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
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

        {/* Portfolio State Input */}
        <div>
          <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Wallet size={18} className="text-neutral-400" />
            보유 현황 입력
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Holdings Input</span>
          </h2>
          <div className="rounded-xl border border-neutral-800 bg-surface divide-y divide-neutral-800">
            <div className="flex items-center justify-between p-4">
              <div>
                <div className="text-sm font-medium text-fg">TQQQ 보유량</div>
                <div className="text-xs text-muted mt-0.5">현재 보유중인 TQQQ 수량</div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={0}
                  value={tqqqShares}
                  onChange={(e) => setTqqqShares(Math.max(0, parseInt(e.target.value) || 0))}
                  disabled={portfolioStateLoading}
                  className="w-24 bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-1.5 text-sm text-fg text-right focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                <span className="text-xs text-muted">주</span>
              </div>
            </div>
            <div className="flex items-center justify-between p-4">
              <div>
                <div className="text-sm font-medium text-fg">SGOV 보유량</div>
                <div className="text-xs text-muted mt-0.5">현금 대용 (SGOV 수량)</div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={0}
                  value={sgovShares}
                  onChange={(e) => setSgovShares(Math.max(0, parseInt(e.target.value) || 0))}
                  disabled={portfolioStateLoading}
                  className="w-24 bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-1.5 text-sm text-fg text-right focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                <span className="text-xs text-muted">주</span>
              </div>
            </div>
            <div className="flex items-center justify-between p-4">
              <div>
                <div className="text-sm font-medium text-fg">스크린샷 OCR</div>
                <div className="text-xs text-muted mt-0.5">삼성증권 앱 스크린샷에서 보유량 자동 추출</div>
              </div>
              <div className="flex items-center gap-2">
                {ocrResult && (
                  <span className={`text-xs flex items-center gap-1 ${
                    ocrResult.success ? "text-green-400" : "text-red-400"
                  }`}>
                    {ocrResult.success ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                    {ocrResult.message}
                  </span>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleOcrUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={ocrLoading || portfolioStateLoading}
                  className="text-xs font-bold px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all bg-purple-600 hover:bg-purple-500 text-white disabled:bg-neutral-700 disabled:text-neutral-400"
                >
                  {ocrLoading ? (
                    <RefreshCw size={12} className="animate-spin" />
                  ) : (
                    <Camera size={12} />
                  )}
                  스크린샷 분석
                </button>
              </div>
            </div>
            <div className="flex items-center justify-between p-4">
              <div>
                <div className="text-sm font-medium text-fg">저장</div>
                <div className="text-xs text-muted mt-0.5">
                  {portfolioLastUpdated 
                    ? `마지막 업데이트: ${new Date(portfolioLastUpdated).toLocaleString("ko-KR")}` 
                    : "저장된 데이터 없음"}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {portfolioSaveResult && (
                  <span className={`text-xs flex items-center gap-1 ${
                    portfolioSaveResult.success ? "text-green-400" : "text-red-400"
                  }`}>
                    {portfolioSaveResult.success ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                    {portfolioSaveResult.message}
                  </span>
                )}
                <button
                  onClick={savePortfolioState}
                  disabled={portfolioSaving || portfolioStateLoading}
                  className="text-xs font-bold px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all bg-blue-600 hover:bg-blue-500 text-white disabled:bg-neutral-700 disabled:text-neutral-400"
                >
                  {portfolioSaving ? (
                    <RefreshCw size={12} className="animate-spin" />
                  ) : (
                    <Check size={12} />
                  )}
                  저장
                </button>
              </div>
            </div>
            <div className="p-4 bg-blue-900/10">
              <div className="text-xs text-blue-400">
                <strong className="block mb-1">알림 조건:</strong>
                <ul className="list-disc list-inside space-y-0.5 text-blue-400/80">
                  <li>BUY 신호 + SGOV 보유 → 알림 발송</li>
                  <li>SELL 신호 + TQQQ 보유 → 알림 발송</li>
                  <li>해당 자산이 없으면 알림 없음</li>
                </ul>
              </div>
            </div>
          </div>
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

        {/* 3. Execution Logs */}
        <div>
             <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <History size={18} className="text-neutral-400" />
            최근 체결
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Execution Logs</span>
            {dataSource === "REAL" && recentTrades.length > 0 && (
              <span className="text-[10px] text-emerald-400 bg-emerald-950/30 px-2 py-0.5 rounded border border-emerald-900/50">
                {recentTrades.length}건
              </span>
            )}
          </h2>
          {dataSource === "MOCK" ? (
            <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
               <Database size={24} className="opacity-50" />
               <span className="text-sm">MOCK 모드에서는 체결 기록이 표시되지 않습니다</span>
               <span className="text-xs text-neutral-600">Settings에서 REAL 모드로 전환하세요</span>
            </div>
          ) : tradesLoading ? (
            <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex items-center justify-center text-muted gap-2">
               <RefreshCw size={20} className="animate-spin" />
               <span className="text-sm">로딩 중...</span>
            </div>
          ) : recentTrades.length === 0 ? (
            <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex flex-col items-center justify-center text-muted gap-2">
               <History size={24} className="opacity-50" />
               <span className="text-sm">최근 체결 내역이 없습니다</span>
               <span className="text-xs text-neutral-600">Command에서 거래를 기록하면 여기에 표시됩니다</span>
            </div>
          ) : (
            <div className="rounded-xl border border-neutral-800 bg-surface overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-neutral-900/50">
                  <tr className="border-b border-neutral-800">
                    <th className="text-left py-3 px-4 text-muted font-medium">날짜</th>
                    <th className="text-left py-3 px-4 text-muted font-medium">종목</th>
                    <th className="text-left py-3 px-4 text-muted font-medium">구분</th>
                    <th className="text-right py-3 px-4 text-muted font-medium">수량</th>
                    <th className="text-right py-3 px-4 text-muted font-medium">단가</th>
                  </tr>
                </thead>
                <tbody>
                  {recentTrades.map((trade) => (
                    <tr key={trade.id} className="border-b border-neutral-800/50 hover:bg-neutral-900/30">
                      <td className="py-3 px-4 font-mono text-xs text-muted">{trade.execution_date}</td>
                      <td className="py-3 px-4 font-bold text-fg">{trade.ticker}</td>
                      <td className="py-3 px-4">
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          trade.action === "BUY" 
                            ? "bg-positive/10 text-positive" 
                            : "bg-negative/10 text-negative"
                        }`}>
                          {trade.action}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-mono">{trade.shares}</td>
                      <td className="py-3 px-4 text-right font-mono">${trade.price_usd?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* 4. Performance */}
        <div>
             <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
            <TrendingUp size={18} className="text-neutral-400" />
            성과 분석
            <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">Performance</span>
            {dataSource === "REAL" && equityHistory.length > 0 && (
              <span className="text-[10px] text-emerald-400 bg-emerald-950/30 px-2 py-0.5 rounded border border-emerald-900/50">
                REAL
              </span>
            )}
          </h2>
          {dataSource === "MOCK" ? (
            <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-8 flex flex-col items-center justify-center text-muted gap-2 border-dashed">
               <Database size={24} className="opacity-50" />
               <span className="text-sm">MOCK 모드에서는 성과 분석이 표시되지 않습니다</span>
               <span className="text-xs text-neutral-600">Settings에서 REAL 모드로 전환하세요</span>
            </div>
          ) : equityLoading ? (
            <div className="rounded-xl border border-neutral-800 bg-surface p-8 flex items-center justify-center text-muted gap-2">
               <RefreshCw size={20} className="animate-spin" />
               <span className="text-sm">성과 데이터 로딩 중...</span>
            </div>
          ) : equityHistory.length === 0 ? (
            <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-8 flex flex-col items-center justify-center text-muted gap-2 border-dashed">
               <Info size={24} className="opacity-50" />
               <span className="text-sm">아직 성과 데이터가 없습니다</span>
               <span className="text-xs text-neutral-600">거래 기록이 쌓이면 성과가 표시됩니다</span>
            </div>
          ) : (
            <div className="rounded-xl border border-neutral-800 bg-surface p-6">
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-neutral-900/50 rounded-lg p-4">
                  <div className="text-xs text-muted mb-1">시작 자산</div>
                  <div className="font-mono font-bold text-fg">
                    ${equityHistory[0]?.value.toLocaleString(undefined, { minimumFractionDigits: 0 })}
                  </div>
                  <div className="text-[10px] text-muted mt-1">{equityHistory[0]?.date}</div>
                </div>
                <div className="bg-neutral-900/50 rounded-lg p-4">
                  <div className="text-xs text-muted mb-1">현재 자산</div>
                  <div className="font-mono font-bold text-fg">
                    ${equityHistory[equityHistory.length - 1]?.value.toLocaleString(undefined, { minimumFractionDigits: 0 })}
                  </div>
                  <div className="text-[10px] text-muted mt-1">{equityHistory[equityHistory.length - 1]?.date}</div>
                </div>
                <div className="bg-neutral-900/50 rounded-lg p-4">
                  <div className="text-xs text-muted mb-1">수익률</div>
                  {(() => {
                    const startVal = equityHistory[0]?.value || 1;
                    const endVal = equityHistory[equityHistory.length - 1]?.value || 1;
                    const returnPct = ((endVal / startVal) - 1) * 100;
                    return (
                      <div className={`font-mono font-bold ${returnPct >= 0 ? "text-positive" : "text-negative"}`}>
                        {returnPct >= 0 ? "+" : ""}{returnPct.toFixed(2)}%
                      </div>
                    );
                  })()}
                  <div className="text-[10px] text-muted mt-1">{equityHistory.length}개 데이터포인트</div>
                </div>
              </div>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}
