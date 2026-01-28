# Theme System Upgrade: 3-Mode Support + Full Light/Dark Compatibility

## TL;DR

> **Quick Summary**: Upgrade theme system from 2-mode (dark/light) to 3-mode (system/dark/light) with OS preference detection, then fix all hardcoded dark-mode colors across 12+ files so light mode actually works.
> 
> **Deliverables**:
> - Pre-commit hook Python 3 compatibility fix
> - 3-mode theme toggle with OS preference detection and live listening
> - All pages and chart components use semantic tokens or dark: variants
> - Light mode renders correctly across entire dashboard
> 
> **Estimated Effort**: Medium (2-3 hours)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (store) → Task 2 (layout) → Task 3 (effects) → Task 4 (settings UI)

---

## Context

### Original Request
1. Fix pre-commit hook shebang from `python` to `python3`
2. Upgrade theme system to 3 modes (system/dark/light) with proper OS preference detection
3. Fix all hardcoded colors so light mode actually works

### Problem Analysis
The user reported: "브라우저가 라이트모드일때 다크/라이트모드랑 브라우저가 다크모드일때 다크/라이트 모드가 색이 다 다르거든요"

**Root Causes Identified:**
1. Theme system only supports 2 modes (dark/light), no "system" auto-detection
2. Many components use hardcoded dark-mode colors like `bg-neutral-800`, `text-neutral-400` without `dark:` variants
3. Charts use hex colors directly (`#3b82f6`, `#ef4444`) instead of CSS variables
4. Mixing semantic tokens (`bg-surface`) with hardcoded colors (`border-neutral-800`)

### Files Requiring Changes

**Core Theme Infrastructure (4 files):**
| File | Change |
|------|--------|
| `settings-store.ts` | Add "system" to theme type, default to "system" |
| `layout.tsx` | Update inline script to handle "system" + matchMedia |
| `settings-effects.tsx` | Listen to prefers-color-scheme when theme="system" |
| `settings/page.tsx` | Add System button to theme toggle UI |

**Pages with Hardcoded Colors (4 files):**
- `portfolio/page.tsx` - `bg-neutral-800`, `border-neutral-800`, `text-neutral-*`
- `settings/page.tsx` - `bg-neutral-900`, `bg-neutral-700`, `text-neutral-500`
- `command/page.tsx` - `bg-neutral-*`, `border-neutral-*`
- `records/page.tsx` - `text-neutral-*`, `bg-neutral-*`

**Chart Components with Hardcoded Hex (6 files):**
- `SvgLineChart.tsx` - `#404040`, `#9ca3af`, `#737373`, `#22c55e`, etc.
- `EquityCurveChart.tsx` - `#334155`, `#475569`, `#94a3b8`, tooltip bg
- `DrawdownChart.tsx` - `#334155`, `#475569`, `#94a3b8`, tooltip bg
- `RollingMetricsChart.tsx` - `#334155`, `#475569`, `#94a3b8`, tooltip bg
- `MarketChartPanel.tsx` - `border-neutral-800`, `bg-neutral-900`
- `StrategyStrip.tsx` - `#9ca3af` for label text

**Other Components:**
- `ReturnsHeatmap.tsx` - `bg-neutral-900/50`, `border-neutral-800`

---

## Work Objectives

### Core Objective
Make the dashboard theme system robust with 3-mode support (system/dark/light) and ensure all UI components render correctly in both dark and light modes.

### Concrete Deliverables
1. `.git/hooks/pre-commit` - shebang updated to python3
2. Theme type extended to include "system"
3. OS preference detection + live listener for system mode
4. All 12+ files converted to semantic tokens or dark: variants

### Definition of Done
- [ ] `python3 .git/hooks/pre-commit` runs without error
- [ ] New users default to "system" theme
- [ ] Changing OS preference immediately updates theme when set to "system"
- [ ] Light mode renders with proper contrast (no dark backgrounds on light)
- [ ] Dark mode remains unchanged (no regressions)

### Must Have
- "System" mode that auto-follows OS preference
- Live listener for OS preference changes (matchMedia change event)
- All visible UI uses semantic tokens or dark: variants
- Backward compatibility (existing "dark"/"light" settings still work)

### Must NOT Have (Guardrails)
- No changes to actual theme colors (keep existing CSS variables)
- No changes to chart logic/data - only colors
- No new dependencies
- No breaking changes to localStorage format (zustand handles migration)
- Do NOT add dark: variants to components that already use semantic tokens correctly

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (no test framework configured for this dashboard)
- **User wants tests**: Manual-only (user will visually verify)
- **Framework**: N/A

### Manual QA Procedures

**Theme Toggle Verification:**
1. Open Settings page, verify 3 buttons visible: System | Dark | Light
2. Click each button, verify `<html>` element class changes appropriately
3. For "System": verify class matches OS preference

**OS Preference Detection:**
1. Set theme to "System"
2. Change OS to dark mode → verify dashboard switches to dark
3. Change OS to light mode → verify dashboard switches to light
4. Verify transition is immediate (no refresh needed)

**Light Mode Visual Check (per page):**
1. Set theme to "Light"
2. Navigate each page: Portfolio, Command, Records, Settings, Analysis
3. Verify: no dark backgrounds on light theme, text is readable, charts visible

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately) - Independent Tasks:
├── Task 0: Pre-commit hook fix [trivial, no deps]
├── Task 5: Fix SvgLineChart.tsx [independent component]
├── Task 6: Fix EquityCurveChart.tsx [independent component]
├── Task 7: Fix DrawdownChart.tsx [independent component]
├── Task 8: Fix RollingMetricsChart.tsx [independent component]
└── Task 9: Fix StrategyStrip.tsx [independent component]

Wave 2 (After Wave 1 starts) - Core Theme:
├── Task 1: Update settings-store.ts [no deps]
├── Task 2: Update layout.tsx [depends: 1 conceptually, can start in parallel]
└── Task 3: Update settings-effects.tsx [depends: 1]

Wave 3 (After Wave 2):
├── Task 4: Update Settings page UI [depends: 1, 3]
├── Task 10: Fix MarketChartPanel.tsx [after chart fixes for consistency]
├── Task 11: Fix portfolio/page.tsx [after effects work]
├── Task 12: Fix command/page.tsx [parallel with 11]
├── Task 13: Fix records/page.tsx [parallel with 11]
├── Task 14: Fix settings/page.tsx [parallel with 11]
└── Task 15: Fix ReturnsHeatmap.tsx [parallel with 11]

Critical Path: Task 1 → Task 3 → Task 4 (theme infra must complete before UI)
Parallel Speedup: ~50% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 0 | None | None | All |
| 1 | None | 2, 3, 4 | 0, 5-9 |
| 2 | 1 | 4 | 3, 5-9 |
| 3 | 1 | 4 | 2, 5-9 |
| 4 | 1, 2, 3 | None | 10-15 |
| 5-9 | None | 10 | 0-4, each other |
| 10-15 | None | None | 4, each other |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Dispatch |
|------|-------|---------------------|
| 1 | 0, 5-9 | 6 parallel agents (quick category) |
| 2 | 1, 2, 3 | 3 parallel agents (quick category) |
| 3 | 4, 10-15 | 6 parallel agents (quick category) |

---

## TODOs

### Task 0: Fix Pre-commit Hook Shebang (Trivial)

- [ ] 0. Update pre-commit hook Python shebang

  **What to do**:
  - Change `#!/usr/bin/env python` to `#!/usr/bin/env python3`
  - First line only, preserve rest of file

  **Must NOT do**:
  - Do not modify any other content in the hook

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-line change, trivial edit
  - **Skills**: None needed
  - **Skills Evaluated but Omitted**:
    - `git-master`: Not needed for file edit

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 5-9)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `.git/hooks/pre-commit:1` - Line to modify (current: `#!/usr/bin/env python`)

  **Acceptance Criteria**:
  - [ ] Line 1 reads exactly `#!/usr/bin/env python3`
  - [ ] Rest of file unchanged
  - [ ] Manual verify: `python3 .git/hooks/pre-commit` → runs without "python not found" error

  **Commit**: YES
  - Message: `fix(hooks): use python3 shebang for pre-commit hook`
  - Files: `.git/hooks/pre-commit`

---

### Task 1: Extend Theme Type in Settings Store

- [ ] 1. Add "system" to theme type and update default

  **What to do**:
  - Change type from `"dark" | "light"` to `"system" | "dark" | "light"`
  - Change default from `"dark"` to `"system"`
  - Add helper function `getResolvedTheme(theme, prefersDark)` that returns actual theme

  **Must NOT do**:
  - Do not change localStorage key name (backward compat)
  - Do not modify other settings

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small TypeScript changes, well-scoped
  - **Skills**: None needed
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Not UI work, just type changes

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3)
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/lib/stores/settings-store.ts:7` - Current type definition
  - `200tq/dashboard/lib/stores/settings-store.ts:44` - Default value

  **Acceptance Criteria**:
  - [ ] `AppSettings.theme` type is `"system" | "dark" | "light"`
  - [ ] `defaultSettings.theme` is `"system"`
  - [ ] TypeScript compiles without errors
  - [ ] Existing "dark"/"light" values in localStorage still work (no migration needed)

  **Commit**: NO (groups with Task 4)

---

### Task 2: Update Layout Inline Script for System Mode

- [ ] 2. Handle "system" preference in SSR inline script

  **What to do**:
  - Modify the inline script in `<head>` to:
    1. Check if `state.theme === "system"`
    2. If system, use `window.matchMedia('(prefers-color-scheme: dark)').matches`
    3. Apply `.dark` class based on resolved preference
  - Preserve existing dark/light handling

  **Must NOT do**:
  - Do not add React code here (must remain vanilla JS for SSR)
  - Do not change the fallback behavior (default to dark if no storage)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small vanilla JS changes in template literal
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 1, 3)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1 (conceptually, but can start in parallel)

  **References**:
  - `200tq/dashboard/app/layout.tsx:16-32` - Current inline script
  - Pattern: `window.matchMedia('(prefers-color-scheme: dark)').matches`

  **Acceptance Criteria**:
  - [ ] When localStorage has `theme: "system"` and OS is dark → `.dark` class applied
  - [ ] When localStorage has `theme: "system"` and OS is light → no `.dark` class
  - [ ] When localStorage has `theme: "dark"` → `.dark` class applied (unchanged)
  - [ ] When localStorage has `theme: "light"` → no `.dark` class (unchanged)
  - [ ] No flash of wrong theme on page load

  **Commit**: NO (groups with Task 4)

---

### Task 3: Update SettingsEffects for System Mode with Live Listener

- [ ] 3. Add matchMedia listener for system preference changes

  **What to do**:
  - When `theme === "system"`:
    1. Get initial OS preference via `matchMedia`
    2. Apply appropriate class based on OS preference
    3. Add `change` event listener to `matchMedia` query
    4. Clean up listener on unmount or theme change
  - When theme is "dark" or "light": existing behavior unchanged

  **Must NOT do**:
  - Do not poll for changes (use event listener)
  - Do not store resolved theme separately (compute on the fly)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: React useEffect pattern, well-defined scope
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 1, 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:
  - `200tq/dashboard/components/settings-effects.tsx:16-27` - Current theme effect
  - Pattern reference: `const mq = window.matchMedia('(prefers-color-scheme: dark)'); mq.addEventListener('change', handler);`

  **Acceptance Criteria**:
  - [ ] When theme is "system" and OS switches dark→light → UI updates immediately
  - [ ] When theme is "system" and OS switches light→dark → UI updates immediately
  - [ ] When theme changes from "system" to "dark" → listener is cleaned up
  - [ ] No memory leaks (listener removed on unmount)

  **Commit**: NO (groups with Task 4)

---

### Task 4: Update Settings Page Theme Toggle UI

- [ ] 4. Add System button to theme toggle

  **What to do**:
  - Add a third button "System" with Monitor icon (from lucide-react)
  - Reorder buttons: System | Dark | Light
  - Update button styling to highlight active state for all 3
  - Import Monitor icon from lucide-react

  **Must NOT do**:
  - Do not change button styling approach (keep existing pattern)
  - Do not add tooltip or explanation (keep minimal)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: UI copy-paste pattern, add one button
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10-15)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2, 3

  **References**:
  - `200tq/dashboard/app/(shell)/settings/page.tsx:253-274` - Current theme toggle UI
  - Pattern: Copy existing button, change icon to Monitor, set value to "system"
  - Lucide icon: `Monitor` from `lucide-react`

  **Acceptance Criteria**:
  - [ ] Three buttons visible: System (Monitor icon) | Dark (Moon icon) | Light (Sun icon)
  - [ ] Clicking System sets theme to "system"
  - [ ] Active button has `bg-neutral-700 text-white` styling
  - [ ] All three buttons work correctly

  **Commit**: YES
  - Message: `feat(settings): add 3-mode theme support (system/dark/light)`
  - Files: `settings-store.ts`, `layout.tsx`, `settings-effects.tsx`, `settings/page.tsx`
  - Pre-commit: `cd 200tq/dashboard && npm run build` (type check)

---

### Task 5: Fix SvgLineChart.tsx Hardcoded Colors

- [ ] 5. Replace hardcoded hex colors with CSS variables

  **What to do**:
  - Replace hardcoded colors in SVG elements with CSS variable references
  - Use `rgb(var(--muted))` pattern for text colors
  - Use `rgb(var(--border))` for grid lines
  - Keep semantic colors (green/red for signals) as-is (they're intentional)

  **Color Mapping**:
  | Current | Replace With | Usage |
  |---------|--------------|-------|
  | `#404040` | `rgb(var(--border))` | Grid lines |
  | `#9ca3af` | `rgb(var(--muted))` | Axis labels |
  | `#737373` | `rgb(var(--muted))` | X-axis labels |
  | `text-neutral-400` | `text-muted` | Hover info |
  | `text-neutral-600` | `text-muted/60` | Low priority text |

  **Keep unchanged** (intentional semantic colors):
  - `#22c55e`, `#ef4444` - Signal colors (green/red)
  - `#3b82f6` - Crosshair/highlight blue
  - Line colors passed as props

  **Must NOT do**:
  - Do not change signal marker colors (they're semantic: buy=green, sell=red)
  - Do not modify chart logic or data processing

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Find-replace pattern, no logic changes
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 0, 6-9)
  - **Blocks**: Task 10
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/components/e03/SvgLineChart.tsx:185` - No data text
  - `200tq/dashboard/components/e03/SvgLineChart.tsx:405` - Hover text neutral-400
  - `200tq/dashboard/components/e03/SvgLineChart.tsx:504` - Grid stroke #404040
  - `200tq/dashboard/components/e03/SvgLineChart.tsx:514` - Y-axis label fill #9ca3af
  - `200tq/dashboard/components/e03/SvgLineChart.tsx:671` - X-axis label fill #737373
  - `200tq/dashboard/app/globals.css:19-20` - CSS variable definitions for --muted

  **Acceptance Criteria**:
  - [ ] Grid lines use `rgb(var(--border))`
  - [ ] Axis labels use `rgb(var(--muted))`
  - [ ] Chart renders correctly in dark mode (no regression)
  - [ ] Chart renders correctly in light mode (visible labels)

  **Commit**: NO (groups with Task 10)

---

### Task 6: Fix EquityCurveChart.tsx Hardcoded Colors

- [ ] 6. Replace Recharts hardcoded colors with CSS variables

  **What to do**:
  - Update CartesianGrid stroke
  - Update XAxis/YAxis stroke and tick fills
  - Update tooltip background/border
  - Use CSS variable pattern for Recharts: `stroke="rgb(var(--border))"`

  **Color Mapping**:
  | Current | Replace With | Element |
  |---------|--------------|---------|
  | `#334155` | `rgb(var(--border))` | CartesianGrid stroke |
  | `#475569` | `rgb(var(--border))` | Axis stroke |
  | `#94a3b8` | `rgb(var(--muted))` | Tick fill |
  | `bg-neutral-900` | `bg-surface` | Tooltip container |
  | `border-neutral-700` | `border-border` | Tooltip border |
  | `text-neutral-400` | `text-muted` | Tooltip date |
  | `text-neutral-300` | `text-fg` | Tooltip labels |

  **Must NOT do**:
  - Do not change line colors (passed as props)
  - Do not modify data processing

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Find-replace in Recharts props
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 0, 5, 7-9)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/components/analysis/EquityCurveChart.tsx:64` - Tooltip bg
  - `200tq/dashboard/components/analysis/EquityCurveChart.tsx:117-128` - Axis colors

  **Acceptance Criteria**:
  - [ ] Tooltip uses `bg-surface border-border`
  - [ ] Axes use CSS variables
  - [ ] Chart visible in both dark and light modes

  **Commit**: NO (groups with Task 10)

---

### Task 7: Fix DrawdownChart.tsx Hardcoded Colors

- [ ] 7. Replace Recharts hardcoded colors with CSS variables

  **What to do**:
  - Same pattern as Task 6
  - Update tooltip, axes, grid

  **Color Mapping**: Same as Task 6

  **Must NOT do**:
  - Do not change red gradient (semantic: drawdown = negative)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/components/analysis/DrawdownChart.tsx:50` - Tooltip bg
  - `200tq/dashboard/components/analysis/DrawdownChart.tsx:119-130` - Axis colors

  **Acceptance Criteria**:
  - [ ] Tooltip uses semantic classes
  - [ ] Axes use CSS variables
  - [ ] Red gradient unchanged (semantic)

  **Commit**: NO (groups with Task 10)

---

### Task 8: Fix RollingMetricsChart.tsx Hardcoded Colors

- [ ] 8. Replace Recharts hardcoded colors with CSS variables

  **What to do**:
  - Same pattern as Tasks 6, 7

  **Color Mapping**: Same as Task 6

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/components/analysis/RollingMetricsChart.tsx:52` - Tooltip
  - `200tq/dashboard/components/analysis/RollingMetricsChart.tsx:130-143` - Axes

  **Acceptance Criteria**:
  - [ ] Consistent with other chart components

  **Commit**: NO (groups with Task 10)

---

### Task 9: Fix StrategyStrip.tsx Hardcoded Colors

- [ ] 9. Replace label text color with CSS variable

  **What to do**:
  - Change `fill="#9ca3af"` to `fill="rgb(var(--muted))"`

  **Must NOT do**:
  - Do not change segment colors (they use CSS variables already via `var(--strip-*)`)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/components/e03/StrategyStrip.tsx:51` - Label fill color

  **Acceptance Criteria**:
  - [ ] Label uses `rgb(var(--muted))`
  - [ ] Segment colors unchanged (already correct)

  **Commit**: NO (groups with Task 10)

---

### Task 10: Fix MarketChartPanel.tsx Hardcoded Colors

- [ ] 10. Replace Tailwind hardcoded colors with semantic tokens or dark: variants

  **What to do**:
  - Replace `bg-neutral-900` with `bg-inset`
  - Replace `border-neutral-800` with `border-border`
  - Replace `text-neutral-*` with `text-muted` or `text-fg`
  - Add `dark:` variants where semantic tokens don't fit

  **Color Mapping**:
  | Current | Replace With |
  |---------|--------------|
  | `bg-neutral-900` | `bg-inset` |
  | `border-neutral-800` | `border-border` |
  | `text-neutral-300` | `text-fg` |
  | `text-neutral-400` | `text-muted` |
  | `text-neutral-500` | `text-muted` |
  | `text-neutral-600` | `text-muted/60` or `dark:text-neutral-600 text-neutral-400` |

  **Must NOT do**:
  - Do not change status badge colors (semantic: green/amber/red)
  - Do not change button active states (blue/purple are intentional brand colors)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Tailwind class replacements
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 4, 11-15)
  - **Blocks**: None
  - **Blocked By**: Tasks 5-9 (for consistency)

  **References**:
  - `200tq/dashboard/components/e03/MarketChartPanel.tsx:393` - Loading state bg
  - `200tq/dashboard/components/e03/MarketChartPanel.tsx:408` - Strategy mode selector
  - `200tq/dashboard/components/e03/MarketChartPanel.tsx:462-504` - Controls container

  **Acceptance Criteria**:
  - [ ] All `bg-neutral-*` replaced with semantic tokens
  - [ ] All `border-neutral-*` replaced with `border-border`
  - [ ] Panel renders correctly in light mode

  **Commit**: YES
  - Message: `style(charts): use semantic tokens for theme compatibility`
  - Files: `SvgLineChart.tsx`, `EquityCurveChart.tsx`, `DrawdownChart.tsx`, `RollingMetricsChart.tsx`, `StrategyStrip.tsx`, `MarketChartPanel.tsx`

---

### Task 11: Fix portfolio/page.tsx Hardcoded Colors

- [ ] 11. Replace Tailwind hardcoded colors with semantic tokens

  **What to do**:
  - Apply same mapping as Task 10
  - Focus on card backgrounds, borders, text colors

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/app/(shell)/portfolio/page.tsx` - Full file scan needed

  **Acceptance Criteria**:
  - [ ] No `bg-neutral-*` or `border-neutral-*` without dark: variant
  - [ ] Page renders correctly in light mode

  **Commit**: NO (groups with Task 14)

---

### Task 12: Fix command/page.tsx Hardcoded Colors

- [ ] 12. Replace Tailwind hardcoded colors with semantic tokens

  **What to do**:
  - Same pattern as Task 11

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/app/(shell)/command/page.tsx` - Full file scan needed

  **Acceptance Criteria**:
  - [ ] No hardcoded neutral colors without dark: variant
  - [ ] Page renders correctly in light mode

  **Commit**: NO (groups with Task 14)

---

### Task 13: Fix records/page.tsx Hardcoded Colors

- [ ] 13. Replace Tailwind hardcoded colors with semantic tokens

  **What to do**:
  - Same pattern as Tasks 11, 12

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/app/(shell)/records/page.tsx` - Full file scan needed

  **Acceptance Criteria**:
  - [ ] No hardcoded neutral colors without dark: variant

  **Commit**: NO (groups with Task 14)

---

### Task 14: Fix settings/page.tsx Hardcoded Colors

- [ ] 14. Replace Tailwind hardcoded colors with semantic tokens

  **What to do**:
  - Same pattern as other page tasks
  - Note: Also include theme toggle from Task 4 if not already done

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 4 (theme toggle changes)

  **References**:
  - `200tq/dashboard/app/(shell)/settings/page.tsx:254` - Theme toggle bg-neutral-900
  - `200tq/dashboard/app/(shell)/settings/page.tsx:242-244` - Notification toggle colors

  **Acceptance Criteria**:
  - [ ] All neutral colors use semantic tokens or dark: variants
  - [ ] Settings page renders correctly in light mode

  **Commit**: YES
  - Message: `style(pages): use semantic tokens for theme compatibility`
  - Files: `portfolio/page.tsx`, `command/page.tsx`, `records/page.tsx`, `settings/page.tsx`

---

### Task 15: Fix ReturnsHeatmap.tsx Hardcoded Colors

- [ ] 15. Replace Tailwind hardcoded colors with semantic tokens

  **What to do**:
  - Replace `bg-neutral-900/50` with `bg-inset`
  - Replace `border-neutral-800` with `border-border`
  - Replace `text-neutral-*` with semantic equivalents

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `200tq/dashboard/components/analysis/ReturnsHeatmap.tsx:37` - StatCard bg
  - `200tq/dashboard/components/analysis/ReturnsHeatmap.tsx:76-81` - Empty state text
  - `200tq/dashboard/components/analysis/ReturnsHeatmap.tsx:110` - Border color

  **Acceptance Criteria**:
  - [ ] All neutral colors replaced
  - [ ] Heatmap visible in light mode

  **Commit**: YES
  - Message: `style(analysis): use semantic tokens in ReturnsHeatmap`
  - Files: `ReturnsHeatmap.tsx`

---

## Commit Strategy

| After Task(s) | Message | Files | Verification |
|---------------|---------|-------|--------------|
| 0 | `fix(hooks): use python3 shebang` | `.git/hooks/pre-commit` | `python3 .git/hooks/pre-commit` |
| 4 (with 1,2,3) | `feat(settings): add 3-mode theme support` | 4 files | Visual: toggle works |
| 10 (with 5-9) | `style(charts): semantic tokens for theme` | 6 files | Visual: charts in light mode |
| 14 (with 11-13) | `style(pages): semantic tokens for theme` | 4 files | Visual: pages in light mode |
| 15 | `style(analysis): semantic tokens in heatmap` | 1 file | Visual: heatmap in light mode |

---

## Success Criteria

### Verification Commands
```bash
# Pre-commit hook works
python3 .git/hooks/pre-commit

# TypeScript compiles
cd 200tq/dashboard && npm run build
```

### Final Checklist
- [ ] Pre-commit hook uses python3 shebang
- [ ] Theme toggle shows 3 options: System | Dark | Light
- [ ] "System" mode follows OS preference
- [ ] OS preference changes update theme immediately (no refresh)
- [ ] All pages render correctly in light mode
- [ ] All charts render correctly in light mode
- [ ] Dark mode has no regressions
- [ ] No TypeScript errors
