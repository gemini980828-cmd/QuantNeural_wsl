"""
Diagnose Synthetic TQQQ Model Discrepancy

Investigate why synthetic TQQQ (3x QQQ daily reset) has:
- Correlation: 0.86 (should be >0.99)
- CAGR difference: +47.67%
- Tracking error: 35.6%

Possible causes:
1. Real TQQQ is split-adjusted but we're comparing raw prices
2. Dividend handling
3. The synthetic model itself is correct but we're not accounting for splits in comparison
"""
import os
import numpy as np
import pandas as pd

print("=" * 70)
print("SYNTHETIC TQQQ MODEL DISCREPANCY DIAGNOSIS")
print("=" * 70)

# Load data
qqq_path = 'data/raw/stooq/us/qqq.us.txt'
qqq = pd.read_csv(qqq_path)
qqq.columns = [c.strip('<>').lower() for c in qqq.columns]
qqq['date'] = pd.to_datetime(qqq['date'], format='%Y%m%d')
qqq = qqq.set_index('date').sort_index()

tqqq_real_path = 'data/raw/yahoo_raw/TQQQ_1d.csv'
tqqq_real = pd.read_csv(tqqq_real_path, parse_dates=['date']).set_index('date')

print("\n[1] Data Overview")
print(f"QQQ: {qqq.index[0].date()} to {qqq.index[-1].date()} ({len(qqq)} days)")
print(f"TQQQ Real: {tqqq_real.index[0].date()} to {tqqq_real.index[-1].date()} ({len(tqqq_real)} days)")

# Check if TQQQ has split data
print("\n[2] TQQQ Split Events in Yahoo Data")
if 'split_ratio' in tqqq_real.columns:
    splits = tqqq_real[tqqq_real['split_ratio'] != 1.0]
    print(f"Split events: {len(splits)}")
    for d, row in splits.iterrows():
        print(f"  {d.date()}: split_ratio = {row['split_ratio']}")
else:
    print("No split_ratio column in TQQQ data")

# Check if we should use 'close' or 'adj_close'
print("\n[3] TQQQ Price Columns")
print(f"Available columns: {list(tqqq_real.columns)}")

# Look at specific split dates
print("\n[4] Price Behavior Around Splits")
split_dates = ['2012-05-11', '2014-01-24', '2017-01-12', '2018-05-24', '2021-01-21', '2022-01-13']
for sd in split_dates:
    if pd.Timestamp(sd) in tqqq_real.index:
        idx = tqqq_real.index.get_loc(pd.Timestamp(sd))
        if idx > 0:
            prev = tqqq_real.iloc[idx - 1]
            curr = tqqq_real.iloc[idx]
            print(f"\n  {sd}:")
            print(f"    Prev close: {prev['close']:.2f}")
            print(f"    Curr close: {curr['close']:.2f}")
            print(f"    Ratio: {curr['close']/prev['close']:.4f}")
            if 'adj_close' in tqqq_real.columns:
                print(f"    Prev adj_close: {prev.get('adj_close', 'N/A')}")
                print(f"    Curr adj_close: {curr.get('adj_close', 'N/A')}")

# Compare using adj_close vs close
print("\n[5] Comparison: Using adj_close vs close for TQQQ")
LEVERAGE = 3.0
DAILY_EXPENSE = 0.0095 / 252
DAILY_BORROWING = 0.005 / 252
DAILY_COST = DAILY_EXPENSE + DAILY_BORROWING

qqq_ret = qqq['close'].pct_change()
synth_tqqq_ret = LEVERAGE * qqq_ret - DAILY_COST

overlap_start = pd.Timestamp('2012-01-03')
overlap_end = pd.Timestamp('2024-12-31')

common_idx = synth_tqqq_ret.index.intersection(tqqq_real.index)
common_idx = common_idx[(common_idx >= overlap_start) & (common_idx <= overlap_end)]

# Using 'close' (raw)
real_ret_close = tqqq_real.loc[common_idx, 'close'].pct_change().dropna()

# Using 'adj_close' if available
if 'adj_close' in tqqq_real.columns:
    real_ret_adj = tqqq_real.loc[common_idx, 'adj_close'].pct_change().dropna()
else:
    # Calculate adjusted returns by adjusting for splits
    tqqq_split = tqqq_real.loc[common_idx, 'split_ratio'].fillna(1.0)
    # Adjusted return = (close / prev_close) * split_ratio - 1
    adj_close = tqqq_real.loc[common_idx, 'close'].copy()
    # Cumulative split adjustment (backward)
    cum_split = tqqq_split[::-1].cumprod()[::-1]
    adj_close_calc = tqqq_real.loc[common_idx, 'close'] * cum_split
    real_ret_adj = adj_close_calc.pct_change().dropna()

synth_ret = synth_tqqq_ret.loc[common_idx].dropna()

# Align all
final_idx = synth_ret.index.intersection(real_ret_close.index)
synth_ret = synth_ret.loc[final_idx]
real_ret_close = real_ret_close.loc[final_idx]

if len(real_ret_adj) > 0:
    real_ret_adj = real_ret_adj.loc[real_ret_adj.index.intersection(final_idx)]

print(f"\nDays compared: {len(final_idx)}")

# Correlation with raw close
corr_close = synth_ret.corr(real_ret_close)
print(f"\nCorrelation (synth vs TQQQ close): {corr_close:.6f}")

# Check: maybe the issue is that Yahoo's 'close' is actually adjusted?
# Let's look at cumulative returns
synth_cum = (1 + synth_ret).cumprod()
real_cum_close = (1 + real_ret_close).cumprod()

years = len(final_idx) / 252
synth_total = synth_cum.iloc[-1] - 1
real_total_close = real_cum_close.iloc[-1] - 1

synth_cagr = (1 + synth_total) ** (1/years) - 1
real_cagr_close = (1 + real_total_close) ** (1/years) - 1

print(f"\nSynthetic CAGR: {synth_cagr*100:.2f}%")
print(f"Real TQQQ (close) CAGR: {real_cagr_close*100:.2f}%")
print(f"CAGR Difference: {(synth_cagr - real_cagr_close)*100:.2f}%")

# Now let's also check what happens if we properly account for splits in TQQQ
print("\n[6] Checking if Yahoo 'close' is already split-adjusted")
# If 'close' is raw, we'd see big jumps on split dates
# If 'close' is adjusted, we'd see smooth prices

# Check a specific split date
sd = pd.Timestamp('2022-01-13')  # 2:1 split
if sd in tqqq_real.index:
    idx = tqqq_real.index.get_loc(sd)
    prev_close = tqqq_real.iloc[idx-1]['close']
    curr_close = tqqq_real.iloc[idx]['close']
    ratio = curr_close / prev_close
    
    print(f"\n2022-01-13 Split Analysis:")
    print(f"  Prev close: {prev_close:.2f}")
    print(f"  Curr close: {curr_close:.2f}")
    print(f"  Price ratio: {ratio:.4f}")
    
    if abs(ratio - 0.5) < 0.1:
        print("  → Price HALVED - this is RAW (unadjusted) prices")
        print("  → Need to adjust for splits when calculating returns!")
    elif abs(ratio - 1.0) < 0.1:
        print("  → Price UNCHANGED - this is ADJUSTED prices")
        print("  → Should not adjust for splits")

# Recalculate with split-adjusted returns for TQQQ
print("\n[7] Recalculating with Split-Adjusted Returns")

# For raw prices, we need: adjusted_return = (close_t * split_ratio_t) / close_{t-1} - 1
tqqq_close = tqqq_real.loc[common_idx, 'close']
tqqq_split = tqqq_real.loc[common_idx, 'split_ratio'].fillna(1.0)

# Adjusted return accounts for splits
tqqq_adj_ret = (tqqq_close * tqqq_split / tqqq_close.shift(1)) - 1
tqqq_adj_ret = tqqq_adj_ret.dropna()

# Align
final_idx2 = synth_ret.index.intersection(tqqq_adj_ret.index)
synth_ret2 = synth_ret.loc[final_idx2]
tqqq_adj_ret2 = tqqq_adj_ret.loc[final_idx2]

corr_adj = synth_ret2.corr(tqqq_adj_ret2)
print(f"\nCorrelation (synth vs TQQQ split-adjusted returns): {corr_adj:.6f}")

# CAGR
synth_cum2 = (1 + synth_ret2).cumprod()
real_cum_adj = (1 + tqqq_adj_ret2).cumprod()

years2 = len(final_idx2) / 252
synth_total2 = synth_cum2.iloc[-1] - 1
real_total_adj = real_cum_adj.iloc[-1] - 1

synth_cagr2 = (1 + synth_total2) ** (1/years2) - 1
real_cagr_adj = (1 + real_total_adj) ** (1/years2) - 1

print(f"\nSynthetic CAGR: {synth_cagr2*100:.2f}%")
print(f"Real TQQQ (split-adjusted) CAGR: {real_cagr_adj*100:.2f}%")
print(f"CAGR Difference: {(synth_cagr2 - real_cagr_adj)*100:.2f}%")

# Tracking error
te = (synth_ret2 - tqqq_adj_ret2).std() * np.sqrt(252)
print(f"Tracking Error: {te*100:.2f}%")

print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
if corr_adj > 0.99:
    print("✅ HIGH CORRELATION achieved with split-adjusted returns")
    print("   → The previous audit was comparing raw prices incorrectly")
else:
    print(f"⚠️ Correlation still low: {corr_adj:.4f}")
    print("   → Need further investigation")
