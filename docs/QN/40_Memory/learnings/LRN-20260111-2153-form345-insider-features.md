# LRN-20260111-2153: Form345 Insider Features (PIT-Safe)

## Summary
Implemented PIT-safe insider trading feature extraction from SEC Form 3/4/5 filings. Insider buying signals provide predictive alpha for Ridge model.

## ETL Pipeline

### Input
- SEC Form345 ZIP files: `data/raw/insiders/2012q1_form345.zip` ... `2025q4_form345.zip`
- 81 quarterly files spanning 2006-2025

### PIT Key: `filed_date`
```python
# CRITICAL: Use FILED date, not transaction date
# filed_date = when SEC received the filing (2 business days after trade)
# This is the first date the information becomes public
event_date = row["filed_date"]  # NOT row["transaction_date"]
```

### Output Features (per ticker, per date)
| Feature | Description |
|---------|-------------|
| `insider_buy_count_90d` | Count of buy transactions in last 90 days |
| `insider_sell_count_90d` | Count of sell transactions in last 90 days |
| `insider_net_count_90d` | Buys - Sells |
| `insider_buy_value_90d` | Total USD value of buys |
| `insider_sell_value_90d` | Total USD value of sells |
| `insider_net_value_90d` | Net USD value |

### Generated Artifact
- `data/processed/insider_events_form345.csv`

## Ridge Integration Results

| Model | Sharpe (no cost) | Sharpe (10/5 bps) |
|-------|------------------|-------------------|
| Momentum only (w=0) | 0.4366 | 0.4144 |
| **Ridge + insider (w=1)** | **0.5012** | **0.4778** |

**Improvement: +14.8% Sharpe (no cost), +15.3% (with cost)**

## Related Files
- `scripts/build_form345_insider_features.py` - ETL pipeline
- `scripts/train_ridge_alpha_walkforward.py` - Ridge with insider option
- `scripts/tune_score_blend.py` - Blend grid evaluation

## Tags
#learning #insider #form345 #sec #pit-safe #features
