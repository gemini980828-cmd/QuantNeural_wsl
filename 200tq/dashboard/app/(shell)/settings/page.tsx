"use client";

import { useState, useEffect } from "react";
import { 
  Palette, Eye, EyeOff, Monitor, Moon, Sun, Bell, BellOff, 
  Database, Shield, AlertTriangle, Check, Send, MessageSquare, RefreshCw, CheckCircle2, XCircle, Archive
} from "lucide-react";
import { useSettingsStore, type AppSettings } from "@/lib/stores/settings-store";
import { applyThemeToDOM } from "@/lib/theme";

// Toggle Switch Component
function Toggle({ 
  enabled, 
  onChange, 
  disabled = false 
}: { 
  enabled: boolean; 
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={() => !disabled && onChange(!enabled)}
      disabled={disabled}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"
      } ${enabled ? "bg-info" : "bg-inset"}`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          enabled ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}

// Settings Section Component
function SettingsSection({ 
  id,
  title, 
  badge, 
  icon: Icon, 
  children 
}: { 
  id?: string;
  title: string; 
  badge: string; 
  icon: React.ElementType; 
  children: React.ReactNode;
}) {
  return (
    <section id={id}>
      <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
        <Icon size={18} className="text-muted" />
        {title}
        <span className="text-xs font-normal text-muted bg-surface px-2 py-0.5 rounded-full">{badge}</span>
      </h2>
      <div className="rounded-xl border border-border bg-surface divide-y divide-border">
        {children}
      </div>
    </section>
  );
}

// Settings Row Component
function SettingsRow({ 
  label, 
  description, 
  children 
}: { 
  label: string; 
  description?: string; 
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between p-4 hover:bg-surface/20 transition-colors">
      <div>
        <div className="text-sm font-medium text-fg">{label}</div>
        {description && <div className="text-xs text-muted mt-0.5">{description}</div>}
      </div>
      <div>{children}</div>
    </div>
  );
}

// Select Dropdown Component
function Select<T extends string>({ 
  value, 
  onChange, 
  options 
}: { 
  value: T; 
  onChange: (v: T) => void; 
  options: { value: T; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className="bg-inset border border-border rounded-lg px-3 py-1.5 text-sm text-fg focus:outline-none focus:ring-2 focus:ring-info/50"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>{opt.label}</option>
      ))}
    </select>
  );
}

export default function SettingsPage() {
  // Use Zustand store
  const {
    theme,
    amountMasking,
    decimalPlaces,
    currency,
    compactMode,
    defaultChartTab,
    simulationMode,
    privacyMode,
    devScenario,
    notificationsEnabled,
    notifyVerdictChange,
    notifyExecScheduled,
    notifyStopLoss,
    dataSource,
    confirmBeforeCopy,
    simOnlyCopy,
    forceHighRiskWarning,
    setSetting,
  } = useSettingsStore();

  // Telegram state
  const [telegramConfigured, setTelegramConfigured] = useState<boolean | null>(null);
  const [telegramLoading, setTelegramLoading] = useState(false);
  const [telegramTestResult, setTelegramTestResult] = useState<{ success: boolean; message: string } | null>(null);
  
  const [telegramEmergency, setTelegramEmergency] = useState(true);
  const [telegramAction, setTelegramAction] = useState(true);
  
  const [backfillLoading, setBackfillLoading] = useState(false);
  const [backfillResult, setBackfillResult] = useState<{ success: boolean; message: string } | null>(null);
  
  // Check Telegram configuration on mount
  useEffect(() => {
    fetch("/api/telegram/send")
      .then(res => res.json())
      .then(data => setTelegramConfigured(data.configured))
      .catch(() => setTelegramConfigured(false));
  }, []);
  
  // Send test message
  const sendTestMessage = async () => {
    setTelegramLoading(true);
    setTelegramTestResult(null);
    try {
      const res = await fetch("/api/telegram/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: `✅ 테스트 성공!\n\n현재 시각: ${new Date().toLocaleString("ko-KR")}\n\n200TQ Dashboard 연동이 정상적으로 작동합니다.`,
          isTest: true,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setTelegramTestResult({ success: true, message: "테스트 메시지 발송 완료!" });
      } else {
        setTelegramTestResult({ success: false, message: data.error || "발송 실패" });
      }
    } catch (e) {
      setTelegramTestResult({ success: false, message: String(e) });
    } finally {
      setTelegramLoading(false);
    }
  };

  const runBackfill = async () => {
    setBackfillLoading(true);
    setBackfillResult(null);
    try {
      const res = await fetch("/api/admin/backfill-snapshots", { method: "POST" });
      const data = await res.json();
      if (data.success) {
        setBackfillResult({ success: true, message: data.message });
      } else {
        setBackfillResult({ success: false, message: data.error || "Backfill 실패" });
      }
    } catch (e) {
      setBackfillResult({ success: false, message: String(e) });
    } finally {
      setBackfillLoading(false);
    }
  };

  return (
    <div className="space-y-8 pb-20">
            {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          설정
          <span className="text-sm font-normal text-muted bg-surface px-2.5 py-0.5 rounded-full border border-border">Settings</span>
        </h1>
      </div>

      {/* Current Mode Summary Strip */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <div className="text-xs text-muted mb-2 font-medium">현재 모드</div>
        <div className="flex flex-wrap gap-2">
          <a 
            href="#mode-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              simulationMode 
                ? "bg-choppy-tint text-choppy border border-choppy/30 hover:bg-choppy/30" 
                : "bg-surface text-muted hover:bg-surface"
            }`}
          >
            SIM: {simulationMode ? "ON" : "OFF"}
          </a>
          <a 
            href="#mode-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              devScenario 
                ? "bg-purple-500/20 text-purple-400 border border-purple-500/30 hover:bg-purple-500/30" 
                : "bg-surface text-muted hover:bg-surface"
            }`}
          >
            DEV: {devScenario ? "ON" : "OFF"}
          </a>
          <a 
            href="#mode-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              privacyMode 
                ? "bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/30" 
                : "bg-surface text-muted hover:bg-surface"
            }`}
          >
            Privacy: {privacyMode ? "ON" : "OFF"}
          </a>
          <a 
            href="#notifications-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              notificationsEnabled 
                ? "bg-positive-tint text-positive border border-positive/30 hover:bg-positive/20" 
                : "bg-surface text-muted hover:bg-surface"
            }`}
          >
            Notifications: {notificationsEnabled ? "ON" : "OFF"}
          </a>
        </div>
      </div>

      {/* A. App */}
      <SettingsSection id="app-section" title="앱" badge="App" icon={Palette}>
        <SettingsRow label="테마" description="시스템/다크/라이트 모드 전환">
          <div className="flex gap-1 bg-inset dark:bg-inset p-1 rounded-lg">
            <button
              onClick={() => { applyThemeToDOM("system"); setSetting("theme", "system"); }}
              className={`px-3 py-1 text-xs font-bold rounded-md flex items-center gap-1.5 transition-colors ${
                theme === "system" ? "bg-inset text-fg" : "text-muted"
              }`}
            >
              <Monitor size={12} />
              System
            </button>
            <button
              onClick={() => { applyThemeToDOM("dark"); setSetting("theme", "dark"); }}
              className={`px-3 py-1 text-xs font-bold rounded-md flex items-center gap-1.5 transition-colors ${
                theme === "dark" ? "bg-inset text-fg" : "text-muted"
              }`}
            >
              <Moon size={12} />
              Dark
            </button>
            <button
              onClick={() => { applyThemeToDOM("light"); setSetting("theme", "light"); }}
              className={`px-3 py-1 text-xs font-bold rounded-md flex items-center gap-1.5 transition-colors ${
                theme === "light" ? "bg-inset text-fg" : "text-muted"
              }`}
            >
              <Sun size={12} />
              Light
            </button>
          </div>
        </SettingsRow>
        <SettingsRow label="금액 마스킹" description="민감한 금액 정보를 *** 로 표시">
          <Toggle enabled={amountMasking} onChange={(v) => setSetting("amountMasking", v)} />
        </SettingsRow>
        <SettingsRow label="소수점 자리">
          <Select 
            value={String(decimalPlaces) as "0" | "1" | "2"} 
            onChange={(v) => setSetting("decimalPlaces", Number(v) as 0 | 1 | 2)}
            options={[
              { value: "0", label: "0자리" },
              { value: "1", label: "1자리" },
              { value: "2", label: "2자리" },
            ]}
          />
        </SettingsRow>
        <SettingsRow label="통화">
          <Select 
            value={currency} 
            onChange={(v) => setSetting("currency", v)}
            options={[
              { value: "KRW", label: "KRW (₩)" },
              { value: "USD", label: "USD ($)" },
            ]}
          />
        </SettingsRow>
        <SettingsRow label="컴팩트 모드" description="UI 요소 간격 축소">
          <Toggle enabled={compactMode} onChange={(v) => setSetting("compactMode", v)} />
        </SettingsRow>
        <SettingsRow label="차트 기본 탭" description="Analysis 페이지 기본 표시">
          <Select
            value={defaultChartTab}
            onChange={(v) => setSetting("defaultChartTab", v)}
            options={[
              { value: "equity", label: "수익률 곡선" },
              { value: "heatmap", label: "히트맵" },
              { value: "decomposition", label: "성과 분해" },
            ]}
          />
        </SettingsRow>
      </SettingsSection>

      {/* B. Mode */}
      <SettingsSection id="mode-section" title="모드" badge="Mode" icon={Monitor}>
        <SettingsRow label="시뮬레이션 모드" description="실제 주문 없이 테스트 실행">
          <div className="flex items-center gap-2">
            {simulationMode && (
              <span className="text-[11px] font-bold text-choppy bg-choppy-tint px-2 py-0.5 rounded border border-choppy/30">
                SIM
              </span>
            )}
            <Toggle enabled={simulationMode} onChange={(v) => setSetting("simulationMode", v)} />
          </div>
        </SettingsRow>
        <SettingsRow label="프라이버시 모드" description="민감 정보 숨김">
          <div className="flex items-center gap-2">
            {privacyMode ? <EyeOff size={14} className="text-muted" /> : <Eye size={14} className="text-muted" />}
            <Toggle enabled={privacyMode} onChange={(v) => setSetting("privacyMode", v)} />
          </div>
        </SettingsRow>
        <SettingsRow label="Dev Scenario" description="개발용 시나리오 모드">
          <div className="flex items-center gap-2">
            {devScenario && (
              <span className="text-[11px] font-bold text-purple-700 dark:text-purple-400 bg-purple-100 dark:bg-purple-950/30 px-2 py-0.5 rounded border border-purple-300 dark:border-purple-900/50">
                DEV
              </span>
            )}
            <Toggle enabled={devScenario} onChange={(v) => setSetting("devScenario", v)} />
          </div>
        </SettingsRow>
      </SettingsSection>

      {/* C. Notifications */}
      <SettingsSection id="notifications-section" title="알림" badge="Notifications" icon={Bell}>
        <SettingsRow label="알림 활성화" description="모든 알림 끄기/켜기">
          <div className="flex items-center gap-2">
            {notificationsEnabled ? <Bell size={14} className="text-info" /> : <BellOff size={14} className="text-muted" />}
            <Toggle enabled={notificationsEnabled} onChange={(v) => setSetting("notificationsEnabled", v)} />
          </div>
        </SettingsRow>
        <SettingsRow label="Verdict 변경 알림" description="BUY/SELL/HOLD 변경 시">
          <Toggle enabled={notifyVerdictChange} onChange={(v) => setSetting("notifyVerdictChange", v)} disabled={!notificationsEnabled} />
        </SettingsRow>
        <SettingsRow label="Exec Scheduled 알림" description="체결 예정 시점에">
          <Toggle enabled={notifyExecScheduled} onChange={(v) => setSetting("notifyExecScheduled", v)} disabled={!notificationsEnabled} />
        </SettingsRow>
        <SettingsRow label="Stop-Loss 임박 알림" description="손절선 접근 시">
          <Toggle enabled={notifyStopLoss} onChange={(v) => setSetting("notifyStopLoss", v)} disabled={!notificationsEnabled} />
        </SettingsRow>
      </SettingsSection>

      {/* Telegram External Notifications */}
      <SettingsSection id="telegram-section" title="텔레그램" badge="Telegram" icon={MessageSquare}>
        <SettingsRow label="연결 상태" description="Vercel 환경변수 설정 필요">
          <div className="flex items-center gap-2">
            {telegramConfigured === null ? (
              <span className="text-xs text-muted flex items-center gap-1">
                <RefreshCw size={12} className="animate-spin" />
                확인 중...
              </span>
            ) : telegramConfigured ? (
              <span className="text-xs text-positive flex items-center gap-1 bg-positive-tint px-2 py-1 rounded-lg border border-positive/30">
                <CheckCircle2 size={12} />
                연결됨
              </span>
            ) : (
              <span className="text-xs text-negative flex items-center gap-1 bg-negative-tint px-2 py-1 rounded-lg border border-negative/30">
                <XCircle size={12} />
                미설정
              </span>
            )}
          </div>
        </SettingsRow>
        <SettingsRow label="Emergency 알림" description="비상 상황 발생 시 발송">
          <Toggle 
            enabled={telegramEmergency} 
            onChange={setTelegramEmergency} 
            disabled={!telegramConfigured} 
          />
        </SettingsRow>
        <SettingsRow label="Action 알림" description="조치 필요 상황 발생 시 발송">
          <Toggle 
            enabled={telegramAction} 
            onChange={setTelegramAction} 
            disabled={!telegramConfigured} 
          />
        </SettingsRow>
        <SettingsRow label="테스트 메시지" description="텔레그램으로 테스트 메시지 발송">
          <div className="flex items-center gap-2">
            {telegramTestResult && (
              <span className={`text-xs flex items-center gap-1 ${
                telegramTestResult.success ? "text-positive" : "text-negative"
              }`}>
                {telegramTestResult.success ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                {telegramTestResult.message}
              </span>
            )}
            <button
              onClick={sendTestMessage}
              disabled={!telegramConfigured || telegramLoading}
              className={`text-xs font-bold px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all ${
                telegramConfigured 
                  ? "bg-info hover:bg-info/80 text-white" 
                  : "bg-surface text-muted cursor-not-allowed"
              }`}
            >
              {telegramLoading ? (
                <RefreshCw size={12} className="animate-spin" />
              ) : (
                <Send size={12} />
              )}
              테스트
            </button>
          </div>
        </SettingsRow>
        {!telegramConfigured && (
          <div className="p-4 bg-choppy-tint border-t border-choppy/30">
            <div className="text-xs text-choppy">
              <strong className="block mb-1">설정 방법:</strong>
              Vercel Dashboard → Settings → Environment Variables에서:
              <ul className="list-disc list-inside mt-1 space-y-0.5 text-choppy/80">
                <li><code className="bg-surface px-1 rounded">TELEGRAM_BOT_TOKEN</code></li>
                <li><code className="bg-surface px-1 rounded">TELEGRAM_CHAT_ID</code></li>
              </ul>
            </div>
          </div>
        )}
      </SettingsSection>

      {/* D. Data & Integrations */}
      <SettingsSection id="data-section" title="데이터 & 연동" badge="Integrations" icon={Database}>
        <SettingsRow label="데이터 소스" description="MOCK: 시뮬레이션 / REAL: 실시간 API">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setSetting("dataSource", "MOCK")}
              className={`text-xs font-bold px-3 py-1 rounded transition-all ${
                dataSource === "MOCK"
                  ? "text-choppy bg-choppy-tint border border-choppy/40"
                  : "text-muted hover:text-fg hover:bg-surface"
              }`}
            >
              MOCK
            </button>
            <button
              onClick={() => setSetting("dataSource", "REAL")}
              className={`text-xs font-bold px-3 py-1 rounded transition-all ${
                dataSource === "REAL"
                  ? "text-positive bg-positive-tint border border-positive/40"
                  : "text-muted hover:text-fg hover:bg-surface"
              }`}
            >
              REAL
            </button>
          </div>
        </SettingsRow>
        <SettingsRow label="API 연동" description="브로커, 가격 데이터">
          <span className="text-xs text-muted bg-surface px-2 py-1 rounded">추후 지원</span>
        </SettingsRow>
        <SettingsRow label="Import/Export" description="데이터 내보내기 형식">
          <span className="text-xs text-muted bg-surface px-2 py-1 rounded">CSV</span>
        </SettingsRow>
        <SettingsRow label="스냅샷 Backfill" description="기존 거래 기록에서 포트폴리오 스냅샷 재생성">
          <div className="flex items-center gap-2">
            {backfillResult && (
              <span className={`text-xs flex items-center gap-1 ${
                backfillResult.success ? "text-positive" : "text-negative"
              }`}>
                {backfillResult.success ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                {backfillResult.message}
              </span>
            )}
            <button
              onClick={runBackfill}
              disabled={backfillLoading}
              className="text-xs font-bold px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all bg-info hover:bg-info/80 text-white disabled:bg-inset disabled:text-muted"
            >
              {backfillLoading ? (
                <RefreshCw size={12} className="animate-spin" />
              ) : (
                <Archive size={12} />
              )}
              Backfill
            </button>
          </div>
        </SettingsRow>
      </SettingsSection>

      {/* E. Safety */}
      <SettingsSection id="safety-section" title="안전" badge="Safety" icon={Shield}>
        <SettingsRow label="주문 복사 전 확인" description="클립보드 복사 전 확인 대화상자">
          <div className="flex items-center gap-2">
            <Check size={14} className="text-positive" />
            <Toggle enabled={confirmBeforeCopy} onChange={(v) => setSetting("confirmBeforeCopy", v)} />
          </div>
        </SettingsRow>
        <SettingsRow label="시뮬 모드에서만 복사 허용" description="LIVE 모드에서 복사 차단">
          <Toggle enabled={simOnlyCopy} onChange={(v) => setSetting("simOnlyCopy", v)} />
        </SettingsRow>
        <SettingsRow label="고위험 상태 경고 강제" description="Down 등에서 경고 항상 표시">
          <div className="flex items-center gap-2">
            <AlertTriangle size={14} className="text-negative" />
            <Toggle enabled={forceHighRiskWarning} onChange={(v) => setSetting("forceHighRiskWarning", v)} />
          </div>
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
