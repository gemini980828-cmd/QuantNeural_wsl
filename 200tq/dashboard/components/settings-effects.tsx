"use client";

import { useEffect, useLayoutEffect } from "react";
import { useSettingsStore } from "@/lib/stores/settings-store";
import { applyThemeToDOM } from "@/lib/theme";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export function SettingsEffects() {
  const theme = useSettingsStore((s) => s.theme);
  const compactMode = useSettingsStore((s) => s.compactMode);
  const hasHydrated = useSettingsStore((s) => s._hasHydrated);

  useIsomorphicLayoutEffect(() => {
    if (!hasHydrated) return;
    applyThemeToDOM(theme);

    if (theme === "system") {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      const handler = () => applyThemeToDOM("system");
      mq.addEventListener("change", handler);
      return () => mq.removeEventListener("change", handler);
    }
  }, [theme, hasHydrated]);

  useIsomorphicLayoutEffect(() => {
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
