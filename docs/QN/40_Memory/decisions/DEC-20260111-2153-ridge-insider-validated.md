# DEC-20260111-2153: Ridge + Insider Alpha Validated

## Decision
Insider trading features from Form345 provide statistically significant alpha improvement over momentum baseline. Recommended for production inclusion.

## Evidence

### OOS Performance Comparison
| Configuration | Sharpe (0 cost) | Sharpe (10/5 bps) | Delta |
|---------------|-----------------|-------------------|-------|
| Momentum baseline (w=0) | 0.4366 | 0.4144 | - |
| Ridge + insider (w=1) | **0.5012** | **0.4778** | **+15%** |

### Blend Grid (w = Ridge weight)
Best configurations from `blend_grid_summary.csv`:
- w=1.0 dominates across cost scenarios
- Insider features robust to transaction costs

## Implementation Details

### New Scripts Added
1. `build_form345_insider_features.py` - Form345 ETL
2. `tune_score_blend.py` - Blend grid evaluation
3. `train_ridge_alpha_walkforward.py` - Upgraded with:
   - Label horizon config (1M/3M)
   - Cross-sectional normalization
   - Sector neutralization option
   - Insider feature integration

### Production Recommendation
```bash
# Generate insider features
python scripts/build_form345_insider_features.py

# Train Ridge with insider
python scripts/train_ridge_alpha_walkforward.py \
  --include_insider_features \
  --label_horizon 3M \
  --sector_neutralize
```

## Risks
- Form345 data lag: 2 business days minimum
- Coverage: Not all tickers have insider activity
- Regime change: Insider signal may decay over time

## Tags
#decision #insider #ridge #alpha #validated
