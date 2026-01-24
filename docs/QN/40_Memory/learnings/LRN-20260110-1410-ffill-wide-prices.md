# Learning: ffill() for Multi-Ticker Wide DataFrames

**ID**: LRN-20260110-1410-ffill-wide-prices
**Date**: 2026-01-10
**Related Commit**: f1fabf0

## Context
When building a wide prices DataFrame from 2,753 tickers spanning different date ranges:
- IPO dates differ significantly (some start 2010, others 2020+)
- `dropna()` collapses to intersection → only 298 rows (monthly!)
- Need daily rows for proper backtest simulation

## Problem
```python
prices_wide = prices_wide.dropna()  # ❌ Collapsed to tiny intersection
```

## Solution
```python
# Forward-fill is PIT-safe (only uses past data)
prices_wide = prices_wide.ffill()

# Drop leading NaN (where no ticker has started yet)
prices_wide = prices_wide.dropna(how="any")
```

## Why PIT-Safe
- `ffill()` only propagates **past** values forward
- No look-ahead bias
- Preserves daily granularity (3,398 rows vs 298)

## Gotcha
- Must also filter for "recent" tickers to avoid stale series:
```python
is_recent = (as_of_dt - last_dt) <= pd.Timedelta(days=10)
```

## Tags
#learning #pit-safety #pandas #backtest