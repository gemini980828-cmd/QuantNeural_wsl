import type { Metadata } from "next";
import "./globals.css";
import { SettingsEffects } from "@/components/settings-effects";

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
    if (stored) {
      var settings = JSON.parse(stored);
      var state = settings.state || settings;
      if (state.theme === "dark") document.documentElement.classList.add("dark");
      else document.documentElement.classList.remove("dark");
      if (state.compactMode) document.documentElement.classList.add("compact");
    } else {
      document.documentElement.classList.add("dark");
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
        {children}
      </body>
    </html>
  );
}
