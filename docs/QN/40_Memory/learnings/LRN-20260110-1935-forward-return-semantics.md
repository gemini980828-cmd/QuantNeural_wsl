# LRN-20260110-1935: Forward Return Semantics for PIT Integrity

## Learning

**Problem**: Standard `pct_change()` gives backward returns (day t-1 → t), but weights set on day t would capture a return that includes information from the future.

**Solution**: Use forward returns with `shift(-1)`:
```python
forward_returns = price_wide.pct_change(fill_method=None).shift(-1)
forward_returns.iloc[-1] = 0.0  # No t+1 on last day
```

**Semantics**:
- Weights effective on day t earn return from t to t+1
- Last day forward return is always 0.0 (no future price)

## Test Pattern

```python
# Prices: [100, 200, 200] — 100% jump on day 2
# Weight set on day 2 (the jump day)
# Expected: equity = 1.0 (jump NOT captured)
# If weight set on day 1: equity = 2.0 (jump IS captured)
```

## Impact

- Critical for PIT integrity in backtests
- Prevents optimistic bias from capturing same-day moves

## Related

- [[RunReport-20260110-phase92x-harness]]
- Task 9.2.0