---
id: LRN-20260116-0020-sec-v23-canonical-layer
date: 2026-01-16
related_runreport: "[[RunReport-20260116-task-10-2-3-rev3]]"
tags: [learning, sec-fundamentals, coverage, implementation]
---

# Learning: SEC Canonical Layer V2.3 Implementation

## Overview

V2.3 extends the SEC canonical layer with secondary computed fallbacks and enhanced quality gates.

## New Features Implemented

### 1. Secondary Computed Fallback (Assets - SE)

```python
def _compute_liabilities_from_assets_minus_se():
    """
    Liabilities = Assets - StockholdersEquity
    Priority: 4th (after primary tags + LSE-SE)
    """
```

**Result**: C \ (A ∪ B) = **0** — No additional coverage (LSE-SE already covers all cases)

### 2. Balance Sheet Identity Gate

```python
# Gate 4: abs(assets - (liab + equity)) <= 5% * assets
if balance_diff > config.balance_identity_tolerance * total_assets:
    flags["balance_identity_fail"] = True
```

### 3. 4-Stage Priority Hierarchy for `total_liabilities`

1. Primary tags (Liabilities, LiabilitiesNoncurrent)
2. COMPUTED_LSE_MINUS_SE
3. COMPUTED_ASSETS_MINUS_SE (new)

### 4. Enhanced Diagnostics

- `primary_lse_se_counts` — Track LSE-SE usage
- `secondary_assets_se_counts` — Track Assets-SE usage
- `analyze_secondary_fallback_potential()` — C \ (A ∪ B) analysis

## Coverage Analysis Results

| Item                | Coverage | Expansion Potential                          |
| ------------------- | -------- | -------------------------------------------- |
| stockholders_equity | 98.9%    | None (2 companies with data lack SE)         |
| revenues            | 94.8%    | Low (11 alt tags are "Income" not "Revenue") |
| operating_cash_flow | 98.5%    | None                                         |

## Key Files Modified

- `src/sec_fundamentals_v2.py` — V2.3 implementation
- `src/build_alpha_dataset.py` — V2.3 API integration + `--exclude-low-confidence`

## Takeaway

> V2.3 adds safety nets (balance identity gate, secondary fallback) but
> coverage is already near ceiling under low-false-positive constraints.
