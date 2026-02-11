import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        inset: "rgb(var(--inset) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",

        // Semantic colors for data display
        positive: "rgb(var(--positive) / <alpha-value>)",
        choppy: "rgb(var(--choppy) / <alpha-value>)",
        negative: "rgb(var(--negative) / <alpha-value>)",

        status: {
          inactive: {
            bg: "rgb(var(--status-inactive-bg) / <alpha-value>)",
            fg: "rgb(var(--status-inactive-fg) / <alpha-value>)",
          },
          action: {
            bg: "rgb(var(--status-action-bg) / <alpha-value>)",
            fg: "rgb(var(--status-action-fg) / <alpha-value>)",
          },
          choppy: {
            bg: "rgb(var(--choppy) / <alpha-value>)",
            fg: "rgb(var(--fg) / <alpha-value>)",
          },
          danger: {
            bg: "rgb(var(--status-danger-bg) / <alpha-value>)",
            fg: "rgb(var(--status-danger-fg) / <alpha-value>)",
          },
        },
        // Legacy e03 mapping for transition - map to new tokens
        e03: {
           bg: "rgb(var(--bg) / <alpha-value>)",
           surface: "rgb(var(--surface) / <alpha-value>)",
           inset: "rgb(var(--inset) / <alpha-value>)",
           border: "rgb(var(--border) / <alpha-value>)",
        }
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      boxShadow: {
        'highlight': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.06)',
      }
    },
  },
  plugins: [],
};

export default config;
