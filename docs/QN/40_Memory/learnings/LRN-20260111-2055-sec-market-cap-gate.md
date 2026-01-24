# LRN-20260111-2055: SEC Market Cap Gating Pipeline

## Summary
Implemented automated market cap gating using SEC companyfacts data to exclude microcaps from backtest universe, preventing TopK from selecting illiquid/risky small-caps.

## Key Implementation Details

### Market Cap Calculation (PIT-Safe)
- **Formula**: `shares_outstanding × price(date, ffill)`
- **shares_outstanding**: From SEC companyfacts `EntityCommonStockSharesOutstanding`
- **Price**: From universe prices CSV with forward-fill for missing dates
- **Threshold**: $300M USD (configurable via `--min_market_cap_usd`)

### Ineligible Stock Penalty
- Stocks below threshold get score penalized: `row_min - 1,000,000`
- This ensures TopK never selects ineligible stocks regardless of original score
- Preserves score ordering for eligible stocks

### Pipeline Flow
```
SEC companyfacts → shares_outstanding (PIT)
                     ↓
prices.csv → aligned price grid
                     ↓
market_cap = shares × price
                     ↓
eligibility mask (market_cap >= threshold)
                     ↓
scores penalized for ineligible → TopK selection
```

## Results (First Run)
| Metric | Value |
|--------|-------|
| Total Tickers | 1,951 |
| Dates | 151 |
| Eligible (min/median/max) | 1,163 / 1,452 / 1,568 |
| Missing shares_outstanding | 103 tickers |

## Generated Artifacts
- `data/backtest_universe_sec_mktcap/scores.csv` - Penalized scores
- `data/backtest_universe_sec_mktcap/prices.csv` - Aligned prices
- `data/backtest_universe_sec_mktcap/market_cap.csv` - Daily market caps
- `data/backtest_universe_sec_mktcap/eligibility.csv` - Binary eligibility mask

## Related Files
- `scripts/build_market_cap_universe.py`
- `scripts/download_sec_data.py` (preprocess, filter_universe commands)

## Tags
#learning #market-cap #universe-construction #sec-data #pit-safe
