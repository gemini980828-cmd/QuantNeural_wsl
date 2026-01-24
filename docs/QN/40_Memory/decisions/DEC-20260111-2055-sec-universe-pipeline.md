# DEC-20260111-2055: SEC-Based Universe Construction Pipeline

## Decision
Implemented a complete SEC data processing pipeline for fundamentals-ready backtesting universes.

## Rationale
1. **Microcap Risk**: Original universe included microcaps that could dominate TopK selection
2. **SEC as SSOT**: SEC companyfacts provides authoritative shares_outstanding data
3. **PIT Integrity**: Market cap calculated using PIT-safe shares × historical price

## Pipeline Components

### 1. SEC Data Download (`download_sec_data.py`)
- `tickers`: Ticker ↔ CIK mapping
- `companyfacts`: Financial statements (19,063 files)
- `submissions`: Filing metadata + SIC codes (14,594 files)

### 2. Universe Preprocessing (`download_sec_data.py preprocess`)
- Builds manifest linking tickers → CIK → companyfacts/submissions
- Extracts SIC → GICS-style sector mapping (S0-S9)
- Outputs: `universe_sec_manifest.csv`, `universe_ticker_to_sector.csv`

### 3. Universe Filtering (`download_sec_data.py filter_universe`)
- Filters prices/scores to tickers with valid SEC data
- Requires: companyfacts OK, optionally sector mapping
- Outputs: Clean `prices.csv`, `scores.csv`, `tickers.txt`

### 4. Market Cap Gating (`build_market_cap_universe.py`)
- Calculates daily market caps from SEC shares × price
- Penalizes sub-threshold stocks to exclude from TopK
- Configurable threshold (default $300M)

## Impact
- **1,951 tickers** with SEC fundamentals coverage
- **10 sectors** mapped from SIC codes
- **Market cap floor** prevents microcap selection

## Alternatives Considered
- Yahoo Finance market cap: Not PIT-safe, stale data
- Manual filtering: Not scalable, error-prone

## Tags
#decision #universe #sec-data #market-cap #fundamentals
