import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

// Settings Types
export type ViewMode = "simple" | "pro";

export interface AppSettings {
  // A. App
  theme: "system" | "dark" | "light";
  amountMasking: boolean;
  decimalPlaces: 0 | 1 | 2;
  currency: "KRW" | "USD";
  compactMode: boolean;
  viewMode: ViewMode;
  defaultChartTab: "equity" | "heatmap" | "decomposition";
  
  // B. Mode
  simulationMode: boolean;
  privacyMode: boolean;
  devScenario: boolean;
  
  // C. Notifications
  notificationsEnabled: boolean;
  notifyVerdictChange: boolean;
  notifyExecScheduled: boolean;
  notifyStopLoss: boolean;
  
  // D. Data
  dataSource: "MOCK" | "REAL";
  
  // E. Safety
  confirmBeforeCopy: boolean;
  simOnlyCopy: boolean;
  forceHighRiskWarning: boolean;
}

interface SettingsStore extends AppSettings {
  // Actions
  setSetting: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
  resetSettings: () => void;
  
  // Hydration state
  _hasHydrated: boolean;
  setHasHydrated: (state: boolean) => void;
}

const defaultSettings: AppSettings = {
  theme: "system",
  amountMasking: false,
  decimalPlaces: 2,
  currency: "KRW",
  compactMode: false,
  viewMode: "simple",
  defaultChartTab: "equity",
  
  simulationMode: true,
  privacyMode: false,
  devScenario: false,
  
  notificationsEnabled: true,
  notifyVerdictChange: true,
  notifyExecScheduled: true,
  notifyStopLoss: true,
  
  dataSource: "MOCK",
  
  confirmBeforeCopy: true,
  simOnlyCopy: true,
  forceHighRiskWarning: true,
};

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      ...defaultSettings,
      _hasHydrated: false,
      
      setSetting: (key, value) => set({ [key]: value }),
      
      resetSettings: () => set(defaultSettings),
      
      setHasHydrated: (state) => set({ _hasHydrated: state }),
    }),
    {
      name: "200tq-settings",
      storage: createJSONStorage(() => localStorage),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    }
  )
);

// Selector hooks for convenience
export const useTheme = () => useSettingsStore((s) => s.theme);
export const useSimulationMode = () => useSettingsStore((s) => s.simulationMode);
export const useDevScenario = () => useSettingsStore((s) => s.devScenario);
export const usePrivacyMode = () => useSettingsStore((s) => s.privacyMode);
export const useAmountMasking = () => useSettingsStore((s) => s.amountMasking);
export const useCompactMode = () => useSettingsStore((s) => s.compactMode);
export const useCurrency = () => useSettingsStore((s) => s.currency);
export const useDecimalPlaces = () => useSettingsStore((s) => s.decimalPlaces);
export const useDataSource = () => useSettingsStore((s) => s.dataSource);
export const useViewMode = () => useSettingsStore((s) => s.viewMode);

// Utility functions
export function formatAmount(
  amount: number,
  settings: Pick<AppSettings, "amountMasking" | "privacyMode" | "currency" | "decimalPlaces">
): string {
  if (settings.amountMasking || settings.privacyMode) {
    return "***";
  }
  
  const formatted = amount.toLocaleString(settings.currency === "KRW" ? "ko-KR" : "en-US", {
    minimumFractionDigits: settings.decimalPlaces,
    maximumFractionDigits: settings.decimalPlaces,
  });
  
  const symbol = settings.currency === "KRW" ? "₩" : "$";
  return `${symbol}${formatted}`;
}

export function canCopyOrder(settings: Pick<AppSettings, "simOnlyCopy" | "simulationMode">): boolean {
  if (settings.simOnlyCopy && !settings.simulationMode) {
    return false;
  }
  return true;
}
