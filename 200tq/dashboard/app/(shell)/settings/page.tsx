"use client";

import { 
  Palette, Eye, EyeOff, Monitor, Moon, Sun, Bell, BellOff, 
  Database, Shield, AlertTriangle, Check
} from "lucide-react";
import { useSettingsStore, type AppSettings } from "@/lib/stores/settings-store";

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
      } ${enabled ? "bg-blue-600" : "bg-neutral-700"}`}
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
        <Icon size={18} className="text-neutral-400" />
        {title}
        <span className="text-xs font-normal text-muted bg-neutral-800 px-2 py-0.5 rounded-full">{badge}</span>
      </h2>
      <div className="rounded-xl border border-neutral-800 bg-surface divide-y divide-neutral-800">
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
    <div className="flex items-center justify-between p-4 hover:bg-neutral-800/20 transition-colors">
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
      className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-1.5 text-sm text-fg focus:outline-none focus:ring-2 focus:ring-blue-500/50"
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

  return (
    <div className="space-y-8 pb-20">
            {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          설정
          <span className="text-sm font-normal text-muted bg-neutral-800 px-2.5 py-0.5 rounded-full border border-neutral-700">Settings</span>
        </h1>
      </div>

      {/* Current Mode Summary Strip */}
      <div className="rounded-xl border border-neutral-800 bg-surface p-4">
        <div className="text-xs text-muted mb-2 font-medium">현재 모드</div>
        <div className="flex flex-wrap gap-2">
          <a 
            href="#mode-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              simulationMode 
                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30" 
                : "bg-neutral-800 text-neutral-500 hover:bg-neutral-700"
            }`}
          >
            SIM: {simulationMode ? "ON" : "OFF"}
          </a>
          <a 
            href="#mode-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              devScenario 
                ? "bg-purple-500/20 text-purple-400 border border-purple-500/30 hover:bg-purple-500/30" 
                : "bg-neutral-800 text-neutral-500 hover:bg-neutral-700"
            }`}
          >
            DEV: {devScenario ? "ON" : "OFF"}
          </a>
          <a 
            href="#mode-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              privacyMode 
                ? "bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/30" 
                : "bg-neutral-800 text-neutral-500 hover:bg-neutral-700"
            }`}
          >
            Privacy: {privacyMode ? "ON" : "OFF"}
          </a>
          <a 
            href="#notifications-section"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              notificationsEnabled 
                ? "bg-green-500/20 text-green-400 border border-green-500/30 hover:bg-green-500/30" 
                : "bg-neutral-800 text-neutral-500 hover:bg-neutral-700"
            }`}
          >
            Notifications: {notificationsEnabled ? "ON" : "OFF"}
          </a>
        </div>
      </div>

      {/* A. App */}
      <SettingsSection id="app-section" title="앱" badge="App" icon={Palette}>
        <SettingsRow label="테마" description="다크/라이트 모드 전환">
          <div className="flex gap-1 bg-neutral-900 p-1 rounded-lg">
            <button
              onClick={() => setSetting("theme", "dark")}
              className={`px-3 py-1 text-xs font-bold rounded-md flex items-center gap-1.5 transition-colors ${
                theme === "dark" ? "bg-neutral-700 text-white" : "text-neutral-500"
              }`}
            >
              <Moon size={12} />
              Dark
            </button>
            <button
              onClick={() => setSetting("theme", "light")}
              className={`px-3 py-1 text-xs font-bold rounded-md flex items-center gap-1.5 transition-colors ${
                theme === "light" ? "bg-neutral-700 text-white" : "text-neutral-500"
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
      </SettingsSection>

      {/* B. Mode */}
      <SettingsSection id="mode-section" title="모드" badge="Mode" icon={Monitor}>
        <SettingsRow label="시뮬레이션 모드" description="실제 주문 없이 테스트 실행">
          <div className="flex items-center gap-2">
            {simulationMode && (
              <span className="text-[10px] font-bold text-amber-400 bg-amber-950/30 px-2 py-0.5 rounded border border-amber-900/50">
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
              <span className="text-[10px] font-bold text-purple-400 bg-purple-950/30 px-2 py-0.5 rounded border border-purple-900/50">
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
            {notificationsEnabled ? <Bell size={14} className="text-blue-400" /> : <BellOff size={14} className="text-muted" />}
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

      {/* D. Data & Integrations */}
      <SettingsSection id="data-section" title="데이터 & 연동" badge="Integrations" icon={Database}>
        <SettingsRow label="데이터 소스" description="MOCK: 시뮬레이션 / REAL: 실시간 API">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setSetting("dataSource", "MOCK")}
              className={`text-xs font-bold px-3 py-1 rounded transition-all ${
                dataSource === "MOCK"
                  ? "text-amber-400 bg-amber-950/50 border border-amber-800"
                  : "text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800"
              }`}
            >
              MOCK
            </button>
            <button
              onClick={() => setSetting("dataSource", "REAL")}
              className={`text-xs font-bold px-3 py-1 rounded transition-all ${
                dataSource === "REAL"
                  ? "text-emerald-400 bg-emerald-950/50 border border-emerald-800"
                  : "text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800"
              }`}
            >
              REAL
            </button>
          </div>
        </SettingsRow>
        <SettingsRow label="API 연동" description="브로커, 가격 데이터">
          <span className="text-xs text-muted bg-neutral-800 px-2 py-1 rounded">추후 지원</span>
        </SettingsRow>
        <SettingsRow label="Import/Export" description="데이터 내보내기 형식">
          <span className="text-xs text-muted bg-neutral-800 px-2 py-1 rounded">CSV</span>
        </SettingsRow>
      </SettingsSection>

      {/* E. Safety */}
      <SettingsSection id="safety-section" title="안전" badge="Safety" icon={Shield}>
        <SettingsRow label="주문 복사 전 확인" description="클립보드 복사 전 확인 대화상자">
          <div className="flex items-center gap-2">
            <Check size={14} className="text-green-400" />
            <Toggle enabled={confirmBeforeCopy} onChange={(v) => setSetting("confirmBeforeCopy", v)} />
          </div>
        </SettingsRow>
        <SettingsRow label="시뮬 모드에서만 복사 허용" description="LIVE 모드에서 복사 차단">
          <Toggle enabled={simOnlyCopy} onChange={(v) => setSetting("simOnlyCopy", v)} />
        </SettingsRow>
        <SettingsRow label="고위험 상태 경고 강제" description="Down 등에서 경고 항상 표시">
          <div className="flex items-center gap-2">
            <AlertTriangle size={14} className="text-red-400" />
            <Toggle enabled={forceHighRiskWarning} onChange={(v) => setSetting("forceHighRiskWarning", v)} />
          </div>
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}

