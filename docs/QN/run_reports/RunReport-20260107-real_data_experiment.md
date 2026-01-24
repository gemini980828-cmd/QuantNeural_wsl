# RunReport-20260107-real_data_experiment

**Date**: 2026-01-07
**Task**: Real Data Experiment Verification
**Run ID**: fb209d21e02e0abfa2414a2cbb433c61468a701426c321d4bbd55a339db5817f

## AutoQC Summary

- **Status**: ✅ PASS
- **Evidence**: Health gates passed (passed=True, failed_gates=[])
- **QC Table**: `data/raw/stooq/us` & `data/raw/sec_bulk` verified

## Execution Details

- **Script**: `scripts/run_real_data_experiment.py`
- **Config**: `configs/real_data_experiment_1.json`
- **As Of Date**: 2024-12-31

### Results

- **MSE**: 0.008939323747148077
- **MAE**: 0.06921655575923558
- **Data Shape**:
  - `n_train`: 82
  - `n_val`: 24
  - `n_test`: 23
  - `n_rows_xy_after_drop`: 129
- **Determinism**: Verified (run_id_same=True, outputs identical)

## Fixes Implemented

- **Stooq Loader**: Removed implicit `.US` suffix stripping. Loader now respects the raw ticker format in the file (`AAPL.US`), requiring the config to specify the exact ticker.

## Next Steps

- Expand universe to more tickers.
- Integrate full 20+ feature set.
