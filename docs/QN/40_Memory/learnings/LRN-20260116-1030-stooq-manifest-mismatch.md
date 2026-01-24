---
id: LRN-20260116-1030-stooq-manifest-mismatch
date: 2026-01-16
tags: [learning, data-integration, manifest, coverage]
---

# Learning: Stooq Manifest Ticker Mismatch Root Cause

## Problem

SEC coverage was only 44% instead of expected 74% in final alpha dataset.

## Root Cause Analysis

### 1. Manifest vs Dataset Ticker Mismatch

| Metric                             | Count             |
| ---------------------------------- | ----------------- |
| Stooq files in data/raw/stooq/us   | 10,502            |
| SEC Manifest OK tickers            | 1,951             |
| **Overlap (before normalization)** | **1,221 (49.7%)** |

### 2. Why Mismatch Occurred

- Stooq filename format: `BRK-B.US.csv`
- Manifest ticker format: `BRK.B`
- Without normalization, matching fails

### 3. Solution: Ticker Normalization

```python
def normalize_ticker(s: str) -> str:
    s = s.strip().upper()
    s = s.replace('-', '.')  # BRK-B -> BRK.B
    if ':' in s:
        s = s.split(':')[-1]  # NYSE:BRK.B -> BRK.B
    s = re.sub(r'\.+', '.', s)  # Collapse dots
    if s.endswith('.US'):
        s = s[:-3]  # Remove .US suffix
    return s
```

### 4. Result After Fix

- Created `scripts/audit_manifest_stooq_tickers.py`
- Created `stooq_aligned_manifest.csv` (248 KB)
- **Overlap: 1,951/1,951 (100%)** - All manifest tickers now match

## Key Takeaway

> Always normalize ticker symbols when joining data from different sources.
> Stooq uses `.US` suffix and `-` for class shares; SEC uses bare symbols with `.`
