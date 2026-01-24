"""Detailed split analysis."""
import pandas as pd

tqqq = pd.read_csv('data/raw/yahoo/TQQQ_1d.csv', parse_dates=['date'])
tqqq = tqqq.set_index('date')

# All splits
splits = tqqq[tqqq['split_ratio'] != 1.0][['close', 'split_ratio']]
print("=== TQQQ Split Events ===")
for dt, row in splits.iterrows():
    print(f"  {dt.date()}: close=${row['close']:.2f}, ratio={row['split_ratio']}")

print(f"\nCumulative split product: {splits['split_ratio'].prod():.1f}")

# Check price continuity around 2022-01-13 split
print("\n=== Prices around 2022-01-13 (2:1 split) ===")
around_split = tqqq.loc['2022-01-10':'2022-01-18', ['close', 'split_ratio']]
for dt, row in around_split.iterrows():
    marker = " <-- SPLIT" if row['split_ratio'] != 1.0 else ""
    print(f"  {dt.date()}: close=${row['close']:.2f}{marker}")

# Key insight: if Yahoo provides split-adjusted prices, 
# the close should NOT jump/drop significantly on split date
jan12 = tqqq.loc['2022-01-12', 'close']
jan13 = tqqq.loc['2022-01-13', 'close']
pct_change = (jan13 - jan12) / jan12 * 100
print(f"\n=== Split Detection ===")
print(f"2022-01-12 close: ${jan12:.2f}")
print(f"2022-01-13 close: ${jan13:.2f}")
print(f"Change: {pct_change:.1f}%")
if abs(pct_change) < 10:
    print("VERDICT: Prices appear SPLIT-ADJUSTED (no sudden drop)")
elif pct_change < -40:
    print("VERDICT: Prices appear RAW (sudden drop on split)")
