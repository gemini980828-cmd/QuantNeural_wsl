import type { Metadata } from "next";
import "./globals.css";
import { SettingsEffects } from "@/components/settings-effects";
import { ToastProvider } from "@/components/ToastProvider";

export const metadata: Metadata = {
  title: "200TQ α Dashboard",
  description: "E03 Strategy Operations Dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="color-scheme" content="only light" />
        <script
          dangerouslySetInnerHTML={{
            __html: `
(function() {
  try {
    var stored = localStorage.getItem("200tq-settings");
    var theme = "system";
    var compactMode = false;
    
    if (stored) {
      var settings = JSON.parse(stored);
      var state = settings && settings.state ? settings.state : settings;
      if (state && (state.theme === "dark" || state.theme === "light" || state.theme === "system")) {
        theme = state.theme;
      }
      compactMode = !!(state && state.compactMode);
    }
    
    var isDark = false;
    if (theme === "system") {
      isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    } else {
      isDark = theme === "dark";
    }
    
    var d = document.documentElement;
    var meta = document.querySelector('meta[name="color-scheme"]');
    d.classList.remove("dark", "light");
    if (isDark) {
      d.classList.add("dark");
      d.setAttribute("data-theme", "dark");
      d.style.setProperty('color-scheme', 'dark');
      if (meta) meta.setAttribute('content', 'dark');
    } else {
      d.classList.add("light");
      d.setAttribute("data-theme", "light");
      d.style.setProperty('color-scheme', 'only light');
      if (meta) meta.setAttribute('content', 'only light');
    }
    
    if (compactMode) {
      d.classList.add("compact");
    } else {
      d.classList.remove("compact");
    }
  } catch (e) {
    var d = document.documentElement;
    var fallbackDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    d.classList.toggle('dark', fallbackDark);
    d.classList.toggle('light', !fallbackDark);
    d.setAttribute('data-theme', fallbackDark ? 'dark' : 'light');
    d.style.setProperty('color-scheme', fallbackDark ? 'dark' : 'only light');
    if (meta) meta.setAttribute('content', fallbackDark ? 'dark' : 'only light');
  }
})();
`,
          }}
        />
      </head>
      <body className="bg-bg text-fg font-sans antialiased">
        <SettingsEffects />
        <ToastProvider />
        {children}
      </body>
    </html>
  );
}
