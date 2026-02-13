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
        "muted-subtle": "rgb(var(--muted-subtle) / <alpha-value>)",

        // Semantic colors for data display
        positive: "rgb(var(--positive) / <alpha-value>)",
        choppy: "rgb(var(--choppy) / <alpha-value>)",
        negative: "rgb(var(--negative) / <alpha-value>)",

        // Informational / accent
        info: "rgb(var(--info) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",

        "positive-tint": "rgb(var(--positive-tint) / <alpha-value>)",
        "negative-tint": "rgb(var(--negative-tint) / <alpha-value>)",
        "choppy-tint": "rgb(var(--choppy-tint) / <alpha-value>)",
        "info-tint": "rgb(var(--info-tint) / <alpha-value>)",
        "accent-tint": "rgb(var(--accent-tint) / <alpha-value>)",

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
        'card': '0 1px 3px 0 rgba(0,0,0,0.06), 0 1px 2px -1px rgba(0,0,0,0.04)',
      }
    },
  },
  plugins: [],
};

export default config;
