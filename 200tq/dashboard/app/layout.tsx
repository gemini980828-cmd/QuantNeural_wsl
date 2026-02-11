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
      var state = settings.state || settings;
      theme = state.theme || "system";
      compactMode = state.compactMode;
    }
    
    var isDark = false;
    if (theme === "system") {
      isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    } else {
      isDark = theme === "dark";
    }
    
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    
    if (compactMode) {
      document.documentElement.classList.add("compact");
    }
  } catch (e) {
    document.documentElement.classList.add("dark");
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
