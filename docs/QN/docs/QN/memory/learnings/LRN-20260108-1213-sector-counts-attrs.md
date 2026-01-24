---
id: LRN-20260108-1213-sector-counts-attrs
date: 2026-01-08
related_tasks: Tasks 7.2.5.1–7.2.5.3.1
tags: [learning, pandas, diagnostics, health-gates]
---

# Learning: DataFrame.attrs for Non-Critical Diagnostics (Fail-Safe Pattern)

## Context
When adding per-month per-sector firm counts to the H1/H2 feature builder, we needed a way to pass diagnostic metadata without changing the 20-column feature structure.

## Solution
- **Attach diagnostics via `DataFrame.attrs["sector_counts"]`** instead of adding columns
- **Fail-safe in health gates**: If `attrs` missing or malformed, log warning and skip gate (don't crash)
- **Default gate threshold `max_low_count_month_ratio=1.0`** (disabled) for backward compatibility

## Key Pattern
```python
# Producer (h1h2_fundamental_momentum.py)
result.attrs["sector_counts"] = counts_df

# Consumer (real_data_health_gates.py) - FAIL-SAFE
try:
    counts_df = frame.attrs.get("sector_counts")
    if isinstance(counts_df, pd.DataFrame) and valid_structure(counts_df):
        # Apply gate logic
except Exception:
    sector_counts_present = False  # Don't crash, just skip
```

## Why This Matters
- `attrs` can be lost through `.copy()`, `.loc[]`, concat, etc.
- Never crash on missing diagnostics — they're nice-to-have, not critical
- Gate only triggers when explicitly configured with stricter threshold

## Related Commits
- `a9f8d73`: feat: Tasks 7.2.4-7.2.5.3.1
