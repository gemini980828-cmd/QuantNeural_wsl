"use client";

import { useEffect } from "react";
import { useSettingsStore } from "@/lib/stores/settings-store";

/**
 * SettingsEffects component applies settings to the DOM.
 * Should be placed once at the root of the app.
 */
export function SettingsEffects() {
  const theme = useSettingsStore((s) => s.theme);
  const compactMode = useSettingsStore((s) => s.compactMode);
  const hasHydrated = useSettingsStore((s) => s._hasHydrated);

  // Apply theme
  useEffect(() => {
    if (!hasHydrated) return;
    
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.remove("light");
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
      root.classList.add("light");
    }
  }, [theme, hasHydrated]);

  // Apply compact mode
  useEffect(() => {
    if (!hasHydrated) return;
    
    const root = document.documentElement;
    if (compactMode) {
      root.classList.add("compact");
    } else {
      root.classList.remove("compact");
    }
  }, [compactMode, hasHydrated]);

  return null;
}
