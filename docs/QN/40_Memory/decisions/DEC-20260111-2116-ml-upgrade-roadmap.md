# DEC-20260111-2116: ML Upgrade Roadmap Priority

## Decision
Established ML upgrade priority roadmap after Ridge walkforward baseline.

## Priority Order

### 1️⃣ Label & Normalization (Highest Priority)
- **Current**: Raw 1M forward return → noisy
- **Target**: 3M forward return for Q rebalancing
- **Transform**: Cross-sectional winsorize → zscore/rank per date
- **Rationale**: Reduces noise, improves signal stability

### 2️⃣ Sector Neutralization
- **Method**: Sector (or market) demean/rank
- **Goal**: Remove "sector betting", learn stock-specific alpha only
- **Implementation**: Subtract sector median before model training

### 3️⃣ 13F/Form345 Features (Lower Priority but Valid)
- **13F**: Slow/medium-term institutional positioning
- **Form345**: Faster event-driven insider signals
- **Key**: PIT based on `filed_date`, not report period
- **Lag**: 13F is 45 days delayed, Form4 is 2 days

## Ridge Baseline Status
- ✅ Walkforward implementation complete
- ✅ All pytest tests passing
- ✅ SSOT/PLANS updated

## Tags
#decision #ml #roadmap #ridge #normalization #neutralization
