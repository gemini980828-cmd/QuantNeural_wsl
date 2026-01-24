"""
Reverse split adjustment for TQQQ.

Split-adjusted prices need to be UN-adjusted to get raw historical prices.
Formula: raw_price = adjusted_price * cumulative_split_factor_from_that_date_to_present

For example, if there was a 2:1 split on 2022-01-13:
- Adjusted price on 2022-01-12: $35
- Raw price on 2022-01-12: $35 * 2 = $70 (actual trading price that day)
"""
import pandas as pd
import numpy as np

# Load Yahoo TQQQ data with split info
tqqq = pd.read_csv('data/raw/yahoo/TQQQ_1d.csv', parse_dates=['date'])
tqqq = tqqq.set_index('date').sort_index()

print("=== Split Events ===")
splits = tqqq[tqqq['split_ratio'] != 1.0][['close', 'split_ratio']]
for dt, row in splits.iterrows():
    print(f"  {dt.date()}: ratio={row['split_ratio']:.2f}")

print(f"\nTotal cumulative split: {splits['split_ratio'].prod():.1f}x")

# Calculate cumulative split factor from each date to present
# We need to multiply prices BEFORE each split by the split ratio
# Working backwards: the most recent price is correct, earlier prices need adjustment

# Create reverse cumulative split factor
# For each date, we need the product of all splits AFTER that date
tqqq['split_factor'] = 1.0

# Calculate cumulative split factor going backwards
# The factor starts at 1.0 for the most recent date
# For dates before a split, we multiply by the split ratio
cumulative = 1.0
factors = []
for dt in reversed(tqqq.index):
    factors.append(cumulative)
    sr = tqqq.loc[dt, 'split_ratio']
    if sr != 1.0:
        cumulative *= sr
factors = list(reversed(factors))
tqqq['split_factor'] = factors

print("\n=== Split Factor Examples ===")
print(f"Latest date factor: {tqqq.iloc[-1]['split_factor']:.4f} (should be 1.0)")
print(f"Earliest date factor: {tqqq.iloc[0]['split_factor']:.4f} (should be ~{splits['split_ratio'].prod():.1f})")

# Create raw (unadjusted) prices
tqqq['raw_open'] = tqqq['open'] * tqqq['split_factor']
tqqq['raw_high'] = tqqq['high'] * tqqq['split_factor']
tqqq['raw_low'] = tqqq['low'] * tqqq['split_factor']
tqqq['raw_close'] = tqqq['close'] * tqqq['split_factor']

print("\n=== Price Comparison ===")
print("Adjusted vs Raw prices:")
print(f"  2012-01-03: Adj=${tqqq.loc['2012-01-03', 'close']:.4f} → Raw=${tqqq.loc['2012-01-03', 'raw_close']:.2f}")
print(f"  2024-12-31: Adj=${tqqq.loc['2024-12-31', 'close']:.4f} → Raw=${tqqq.loc['2024-12-31', 'raw_close']:.2f}")

# Check around a split date
print("\nAround 2022-01-13 split (2:1):")
for dt in ['2022-01-12', '2022-01-13', '2022-01-14']:
    if dt in tqqq.index:
        row = tqqq.loc[dt]
        print(f"  {dt}: Adj=${row['close']:.2f} → Raw=${row['raw_close']:.2f}, factor={row['split_factor']:.4f}")

# Calculate returns to verify
adj_return = tqqq.iloc[-1]['close'] / tqqq.iloc[0]['close'] - 1
raw_return = tqqq.iloc[-1]['raw_close'] / tqqq.iloc[0]['raw_close'] - 1
print(f"\n=== Return Verification ===")
print(f"Adjusted total return: {adj_return*100:.1f}%")
print(f"Raw total return: {raw_return*100:.1f}%")
print(f"(These should be equal - just different price scales)")

# Save raw prices
output = tqqq[['raw_open', 'raw_high', 'raw_low', 'raw_close', 'volume', 'dividend', 'split_ratio', 'split_factor']].copy()
output.columns = ['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio', 'split_factor']
output.to_csv('data/raw/yahoo/TQQQ_raw_1d.csv')
print(f"\nRaw prices saved to: data/raw/yahoo/TQQQ_raw_1d.csv")
