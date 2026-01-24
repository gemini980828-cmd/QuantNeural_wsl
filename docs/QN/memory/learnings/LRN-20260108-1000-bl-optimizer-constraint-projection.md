---
id: LRN-20260108-1000-bl-optimizer-constraint-projection
date: 2026-01-08
related_task: "Task 7.3.1.2: BL Optimizer Constraint Safety"
commit: 3313a90
tags: [learning, black-litterman, portfolio-optimization, constraint-safety]
---

# Learning: BL Optimizer Normalize-After-Clip Bug

## Problem Discovered
The Black-Litterman optimizer's fallback logic had a critical bug: **normalizing weights to sum==1 AFTER applying caps would re-break the caps**.

Example:
1. N=10, max_stock_weight=0.10
2. Equal weight: w = [0.1, 0.1, ..., 0.1] (sum=1.0) ✓
3. After clipping to 0.10: w = [0.1, 0.1, ..., 0.1] ✓
4. If caps are relaxed to 0.11 → w = [0.11, 0.11, ...] → sum=1.1
5. Normalize to sum=1: w *= 1/1.1 → w = [0.1, 0.1, ...] ✓

But trouble occurs when:
1. Initial caps infeasible (e.g., N=10, max_stock=0.02 → max sum=0.2 < 1)
2. Caps get relaxed to 0.11
3. Weights normalized → caps violated again!

## Root Cause
The normalize step `w = w / sum(w)` was applied globally, potentially scaling UP weights that were already at cap, causing cap violations.

## Solution
Implemented `_project_feasible()` helper with proper constraint-aware projection:
1. **Step A**: Initialize from w0 or equal weight
2. **Step B**: Apply stock cap (clip down only)
3. **Step C**: Enforce sector caps by scaling down within each sector
4. **Step D**: 
   - If allow_cash=True: scale down if sum>1, never scale up
   - If allow_cash=False: fill remaining mass into assets with SLACK capacity only
5. `_fill_remaining_mass()`: deterministic slot-filling by ascending index, respecting both stock and sector slack

## Key Insight
> **Never do a global normalize that can scale UP weights. Only fill into slack capacity.**

## Additional Fixes in Same Task
- `calibrate_sector_views`: output shape ALWAYS (K,), coerce raw_scores to match
- `rmt_denoise_covariance`: use q=T/N without forcing q>=1 for correct Marcenko-Pastur
- Added parseable logs: `final_caps msw=X mxsw=Y` for test verification
