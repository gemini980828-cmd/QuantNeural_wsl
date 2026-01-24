---
id: LRN-20260116-0115-sec-v2-eligible-count
date: 2026-01-16
as_of_date: 2024-12-31
related_runreport: "[[RunReport-20260116-task-10-2-3-rev3]]"
tags: [learning, sec-fundamentals, xgb-training, coverage]
---

# Learning: SEC Fundamentals Reduce XGB Eligible Count

## Context

When training XGB alpha model with SEC fundamental features (V2.3 canonical layer),
the eligible_count per rebalance date drops significantly compared to tech-only.

## Problem

Training script has fail-fast check:

```python
if eligible_count < top_k:
    raise RuntimeError(f"eligible_count({date})={eligible_count} < top_k={top_k}")
```

With FUND dataset (V2.3 SEC columns):

- top_k=400 → FAIL (eligible=286 on 2017-03-31)
- top_k=250 → FAIL (eligible=229 on 2017-12-29)
- top_k=200 → FAIL (eligible=196 on 2024-10-01)
- top_k=150 → SUCCESS

## Root Cause

SEC coverage is ~44%, meaning ~56% of rows have NaN in fundamental columns.
These rows are likely dropped or have reduced weight during training.

## Solution Options

1. **Same top_k for fair comparison** — Use top_k=150 for both TECH and FUND
2. **Feature engineering** — Replace NaN with 0.0 + use `_is_missing` indicator
3. **Increase coverage** — Expand SEC tag mapping or use more fallbacks

## Key Takeaway

> When comparing Tech-only vs Fund-included models, ensure same top_k constraint
> or the comparison is invalid due to different universe sizes per date.
