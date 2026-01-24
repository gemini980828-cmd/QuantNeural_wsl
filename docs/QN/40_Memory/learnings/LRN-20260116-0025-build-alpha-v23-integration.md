---
id: LRN-20260116-0025-build-alpha-v23-integration
date: 2026-01-16
tags: [learning, build-alpha-dataset, v23-api, cli]
---

# Learning: build_alpha_dataset V2.3 API Integration

## Change Summary

`src/build_alpha_dataset.py` was updated to integrate V2.3 SEC canonical layer.

## New Parameters

| Parameter                | Default | Description                                |
| ------------------------ | ------- | ------------------------------------------ |
| `exclude_low_confidence` | False   | NaN-out SEC values from computed fallbacks |
| `use_v2_fundamentals`    | True    | Use V2.3 canonical layer (vs legacy V1)    |

## CLI Options

```bash
# Exclude low-confidence (computed fallback) values
python -m src.build_alpha_dataset \
  --exclude-low-confidence \
  --manifest-csv data/.../manifest.csv

# Use legacy V1 API
python -m src.build_alpha_dataset \
  --use-v1-fundamentals
```

## V2.3 Column Names

| V1 Column   | V2.3 Column         |
| ----------- | ------------------- |
| assets      | total_assets        |
| liabilities | total_liabilities   |
| equity      | stockholders_equity |
| (new)       | revenues            |
| (new)       | net_income          |
| (new)       | operating_cash_flow |
| (new)       | shares_outstanding  |

## \_is_missing Indicators

Both V1 and V2.3 columns generate `{col}_is_missing` indicators:

- V1: `assets_is_missing`, `liabilities_is_missing`
- V2.3: `total_assets_is_missing`, `stockholders_equity_is_missing`

## Test Compatibility

Tests were updated to accept both V1 and V2.3 column names:

```python
expected_indicators = [
    "assets_is_missing",  # V1
    "total_assets_is_missing",  # V2.3
]
```
