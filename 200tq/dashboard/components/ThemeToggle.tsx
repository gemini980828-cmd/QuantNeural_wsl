"use client";
import { useSettingsStore } from "@/lib/stores/settings-store";
import { applyThemeToDOM } from "@/lib/theme";

export function ThemeToggle() {
  const theme = useSettingsStore((s) => s.theme);
  const setSetting = useSettingsStore((s) => s.setSetting);

  const cycle = () => {
    const order = ["dark", "light", "system"] as const;
    const idx = order.indexOf(theme);
    const next = order[(idx + 1) % order.length];
    applyThemeToDOM(next);
    setSetting("theme", next);
  };

  const label = theme === "dark" ? "Dark" : theme === "light" ? "Light" : "System";

  return (
    <button
      onClick={cycle}
      className="rounded-lg bg-inset px-3 py-2 text-sm text-fg hover:opacity-90"
      aria-label="Toggle theme"
    >
      {label}
    </button>
  );
}
