---
id: LRN-20260116-1030-parallel-dataset-builder
date: 2026-01-16
tags: [learning, performance, parallel-processing]
---

# Learning: Parallel Alpha Dataset Builder

## Problem

Sequential `build_alpha_dataset.py` takes ~10+ minutes for 10,502 files.

## Solution

Created `scripts/build_alpha_dataset_fast.py` with ThreadPoolExecutor:

```python
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(process_single_ticker, args): args for args in args_list}
    for future in as_completed(futures):
        result = future.result()
        # ...
```

## Key Features

1. **Parallel I/O** - 16 workers process files concurrently
2. **Progress reporting** - Every 500 files
3. **Unbuffered output** - Use `python -u` for immediate feedback
4. **Same logic** - Mirrors original build_alpha_dataset behavior

## Performance

| Metric                | Sequential | Parallel (16 workers) |
| --------------------- | ---------- | --------------------- |
| Time for 10,502 files | ~10+ min   | ~20 min\*             |

\*SEC data loading is the bottleneck, not parallelizable per-ticker.

## Usage

```bash
python -u scripts/build_alpha_dataset_fast.py \
  --data-dir data/raw/stooq/us \
  --output-path data/processed/alpha_dataset.csv.gz \
  --as-of-date 2024-12-31 \
  --manifest-csv data/processed/manifest_audit/stooq_aligned_manifest.csv \
  --max-workers 16
```

## Takeaway

> SEC data loading dominates build time. For further optimization,
> consider batch-loading SEC facts or caching compiled fundamentals.
