---
id: LRN-20260107-2230-stooq-ticker-format
date: 2026-01-07
as_of_date: 2024-12-31
related_taskspec: "[[Real Data Experiment]]"
related_runreport: "[[RunReport-20260107-real_data_experiment]]"
tags: [learning, integration, data_loader]
---

# Learning: Stooq Ticker Format Handling

## Observation

The Stooq data loader (`stooq_prices.py`) initially attempted to be "smart" by stripping the `.US` suffix from tickers in the raw file (e.g., converting `AAPL.US` to `AAPL`). This caused mismatches when the configuration provided `AAPL.US` or when the file content validation expected the original format.

## Insight

Implicit data cleaning in the loader layer can lead to confusion about the expected identifier format. It is better to handle tickers "raw" in the lower-level loader and let the configuration or upper-level logic decide on the naming convention.

## Action Taken

- Removed `.str.replace(r"\.US$", "", regex=True)` from `load_stooq_daily_prices`.
- Updated tests and config to use strictly matching `AAPL.US` format.
