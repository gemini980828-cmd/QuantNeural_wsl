# LRN-20260111-2116: Cross-Sectional Normalization for ML

## Summary
Raw forward returns are noisy for ML training. Cross-sectional normalization per date dramatically improves signal quality.

## Problem
- **Raw 1M forward return**: High variance, outliers dominate
- **Time-series variation**: Market regime changes obscure stock-specific alpha
- **Sector confounding**: Model learns sector momentum, not stock selection

## Solution: Per-Date (Cross-Sectional) Transform

```python
# For each rebalance date:
def normalize_for_ml(returns_date: pd.Series) -> pd.Series:
    # 1. Winsorize extremes (1%, 99%)
    clipped = returns_date.clip(
        lower=returns_date.quantile(0.01),
        upper=returns_date.quantile(0.99)
    )
    # 2. Z-score (mean=0, std=1)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-8)
```

### Optional: Sector Neutralization
```python
def neutralize_sector(returns: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    for ticker in returns.columns:
        sector = sector_map.get(ticker)
        if sector:
            sector_median = returns[sector_tickers].median(axis=1)
            returns[ticker] -= sector_median
    return returns
```

## Label Horizon Alignment
| Rebalance | Label Horizon | Rationale |
|-----------|---------------|-----------|
| Monthly (M) | 1M forward | Aligned |
| Quarterly (Q) | **3M forward** | Match holding period |

## Impact
- Reduces outlier influence
- Removes market-wide trends
- Forces model to learn relative rankings

## Tags
#learning #ml #normalization #cross-sectional #sector-neutral
