# QUANT-NEURAL — Data Contract

> **Single Source of Truth** for data semantics, formats, and validation rules.
>
> See also: [SSOT_TASKS.md](./SSOT_TASKS.md) for complete task contracts (7.0–7.4).

---

## 1. Global Time Semantics

### 1.1 `as_of_date` Meaning and PIT Cutoff Rule

- **`as_of_date`**: The reference date for all data queries and computations.
- **PIT Cutoff Rule**: Only data available **on or before** `as_of_date` may be used.
- **No Future Data**: Any data timestamped after `as_of_date` is strictly forbidden.

```python
# Example: Correct PIT query
data = load_data(end_date=as_of_date)  # Includes data up to as_of_date

# WRONG: This leaks future data
data = load_data()  # Uses all available data including future
```

### 1.2 Time Index Ordering Requirement

- All time-indexed DataFrames/arrays **MUST** be sorted in **ascending** order by date.
- Before applying any time-series transforms, validate:
  ```python
  assert df.index.is_monotonic_increasing, "Time index must be ascending"
  ```

### 1.3 Frequency Assumption

- **Default frequency**: Monthly (end-of-month).
- **Extension**: Weekly or daily frequencies may be added later but are NOT in scope for initial phases.
- **Alignment**: When merging datasets, align to the common frequency (typically monthly).

---

## 2. Dataset Split Policy

### 2.1 Train/Val/Test Order Constraints

> [!CAUTION] > **NO RANDOM SPLITS. NO SHUFFLE.**

- **Chronological ordering**: Train → Validation → Test (strictly in time order).
- Split ratios (configurable defaults, not hard spec):
  - Train: 70% (placeholder)
  - Validation: 15% (placeholder)
  - Test: 15% (placeholder)

```
Timeline: ─────────────────────────────────────────────────►
          │◄────── Train ──────►│◄─ Val ─►│◄─ Test ─►│
          t0                    t1        t2         t3
```

### 2.2 Train-Only Fit, Val/Test Transform-Only

| Operation        | Train | Validation | Test |
| ---------------- | ----- | ---------- | ---- |
| Scaler fit       | ✅    | ❌         | ❌   |
| Scaler transform | ✅    | ✅         | ✅   |
| RankGauss fit    | ✅    | ❌         | ❌   |

```python
# Correct pattern
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)   # transform only
X_test_scaled = scaler.transform(X_test)  # transform only

# WRONG: Leaks val/test statistics
scaler.fit(X_all)  # ❌ Includes val and test
```

### 2.3 Winsorization Semantics

> [!NOTE] > **Winsorization is NOT a train-only-fitted transform.**

Winsorization is **cross-sectional per-date clipping**:

- At each date `t`, compute quantiles from that date's cross-section only.
- No future leakage by design (only uses data from time `t`).
- Distinct from train-only-fitted transforms (RankGauss, scalers, calibration).

```python
# Per-date cross-sectional winsorization (correct)
for date in df.index.unique():
    mask = df.index == date
    lower = df.loc[mask, col].quantile(0.01)
    upper = df.loc[mask, col].quantile(0.99)
    df.loc[mask, col] = df.loc[mask, col].clip(lower, upper)
```

---

## 3. Core Tables / Tensors

> [!IMPORTANT]
> For complete API contracts, see [SSOT_TASKS.md](./SSOT_TASKS.md).

### 3.1 Factor/Feature Table Expectations

| Column                    | Type     | Description                    |
| ------------------------- | -------- | ------------------------------ |
| `date`                    | datetime | Observation date (monthly end) |
| `ticker`                  | str      | Stock identifier               |
| `factor_1` ... `factor_N` | float    | Factor/feature values          |

**Requirements**:

- Index: MultiIndex `(date, ticker)` or separate columns.
- No duplicate `(date, ticker)` pairs.
- NaN handling: Explicit (see Section 5).

### 3.2 Feature Tensor X — H1/H2 Fundamental Momentum

- **Shape**: `(n, 20)` where `n` = number of months
- **Columns** (exact order):
  - `S0_H1, S1_H1, S2_H1, ..., S9_H1` (10 sectors × H1 short momentum)
  - `S0_H2, S1_H2, S2_H2, ..., S9_H2` (10 sectors × H2 long momentum)
- **Sector Ordering**: S0–S9 mapped to sectors via `cik_to_sector` dictionary.
- **Dtype**: `float64`
- **Constraints**:
  - May contain NaN in early rows (insufficient lag history).
  - Post-preprocessing: RankGauss normalized (train-only fit).

### 3.3 Label Tensor Y — Next-Month Returns

- **Shape**: `(n, 10)` where `n` = number of months
- **Columns** (exact order): `S0_Y, S1_Y, S2_Y, ..., S9_Y`
- **Construction**: `y_t = (close[t+1] / close[t]) - 1`
- **Dtype**: `float64`
- **Constraints**:
  - Last available month has NaN (no t+1) — dropped during alignment.
  - Finite numeric values after alignment.

### 3.4 Monthly Index Assumptions

- **Frequency**: Month-end dates (`freq="ME"`)
- **Index Type**: `pd.DatetimeIndex`, monotonically increasing.
- **Alignment**: X and Y share identical index after `build_real_data_xy_dataset()`.

> [!NOTE] > **Allocation/weights normalization** (softmax, sum=1.0) is a **separate downstream layer** in the Black-Litterman module.

---

## 4. Leakage Hazards and Explicit Guards

### 4.1 Hamilton Filter Forward-Looking Caveat

> [!WARNING]
> Hamilton filter is inherently non-causal. Use with extreme care.

The Hamilton filter uses forward values `y_{t+h}` where `h` is the horizon parameter.

**Rules**:

- **Backtest use**: Allowed for historical comparison.
- **Live/production use**: **MUST** enforce `macro_lag_months >= h`.

```python
# Guard for live logic
assert macro_lag_months >= hamilton_h, (
    f"Hamilton filter uses h={hamilton_h} forward periods. "
    f"macro_lag_months={macro_lag_months} must be >= h for live logic."
)
```

### 4.2 No "Now" Logic

- **FORBIDDEN**:
  ```python
  now = datetime.now()  # ❌ Non-deterministic
  today = date.today()  # ❌ Changes daily
  ```
- **REQUIRED**:
  ```python
  # Explicit as_of_date parameter
  def process_data(as_of_date: date) -> pd.DataFrame:
      ...
  ```

### 4.3 Leakage Checklist

Before any implementation, verify:

- [ ] No future data used in feature calculation
- [ ] Scalers/transformers fit on train only
- [ ] No `datetime.now()` or `date.today()` calls
- [ ] Time indices sorted ascending
- [ ] Hamilton filter guarded for live use

---

## 5. Missing Data Policy

### 5.1 Two Categories of NaN

> [!IMPORTANT]
> NaN handling depends on **cause** and **stage**. Not all NaNs are defects.

#### (A) Expected Warmup NaNs — Allowed at Raw Stage

These NaNs are **structurally expected** during feature construction:

| Source                      | Reason                                 | Handling Point                    |
| --------------------------- | -------------------------------------- | --------------------------------- |
| H1/H2 momentum (early rows) | TTM requires 4+ quarters of history    | Dropped during dataset alignment  |
| Label Y (final row)         | Next-month return undefined for last t | Dropped during dataset alignment  |
| HP filter (initial rows)    | Rolling window needs warmup if guarded | Filled as `trend=series, cycle=0` |

**Contract**: Warmup NaNs are trimmed by `build_real_data_xy_dataset()` or health gates **before** training. They must not reach the model.

#### (B) Unexpected NaNs — Fail-Fast

These NaNs are **data defects** and must be rejected:

| Stage                      | Symptom                           | Action                                   |
| -------------------------- | --------------------------------- | ---------------------------------------- |
| Post-alignment X/Y         | NaN in feature/label after warmup | **RAISE** `ValueError`                   |
| Required columns missing   | Column not in DataFrame           | **RAISE** `ValueError` with column names |
| Non-finite after RankGauss | Inf/NaN after preprocessing       | **RAISE** — should not happen            |

```python
# Post-alignment fail-fast guard
if X_aligned.isna().any().any() or Y_aligned.isna().any().any():
    raise ValueError("Unexpected NaN after warmup trimming")
```

### 5.2 Missingness Threshold (Health Gates)

- `max_feature_missing_ratio` (default 0.20): Per-column NaN ratio after ignoring first N warmup rows.
- If exceeded, health gates fail with diagnostic report.

### 5.3 Explicit Forward-Fill (Exception)

When forward-fill is intentional, document explicitly:

```python
# EXPLICIT: Forward-fill for price gaps on non-trading days (documented)
df['price'] = df['price'].ffill()
```

---

## 6. Validation Functions (Placeholder)

> Task 0 placeholder; validation utilities implemented in later prompts.

```python
def validate_time_index(df: pd.DataFrame) -> None:
    """Validate time index is monotonically increasing."""
    # TODO: Implement in Prompt A/B

def validate_no_leakage(
    train_end: date,
    val_start: date,
    test_start: date
) -> None:
    """Validate chronological ordering of splits."""
    # TODO: Implement in Prompt A/B

def validate_pit_cutoff(
    data_date: date,
    as_of_date: date
) -> None:
    """Validate data respects PIT cutoff."""
    # TODO: Implement in Prompt A/B
```

---

## 7. Quick Reference

### Input Tensor X

- Shape: `(T, 20)`
- Content: Relative earnings momentum features
- Preprocessing: RankGauss (train-only fit), winsorize (per-date cross-sectional)

### Output Tensor Y

- Shape: `(T, 10)`
- Content: 10-dim continuous targets (sector returns/scores)
- Constraints: Finite numeric values; allocation normalization is downstream

### Critical Guards

```python
# PIT
assert data.index.max() <= as_of_date

# Ascending order
assert df.index.is_monotonic_increasing

# Train-only fit
scaler.fit(X_train)  # Only on train

# No shuffle
model.fit(X, y, shuffle=False)

# Hamilton guard
assert macro_lag_months >= hamilton_h
```
