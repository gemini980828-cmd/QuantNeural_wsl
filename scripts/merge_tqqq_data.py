"""
Step 1: Merge Synthetic TQQQ (1999-2010) with Real TQQQ (2010+)

We use Synthetic TQQQ up to 2010-02-08 (day before real TQQQ listing),
then switch to real TQQQ data from 2010-02-09 onwards.

The synthetic data is scaled so that the transition is seamless.
"""
import os
import pandas as pd
import numpy as np

print("=== Step 1: Merging Synthetic and Real TQQQ ===")

# Load synthetic TQQQ
synthetic = pd.read_csv('data/raw/synthetic/TQQQ_synthetic_1d.csv', parse_dates=['date'])
synthetic = synthetic.set_index('date').sort_index()
print(f"Synthetic TQQQ: {synthetic.index[0].date()} to {synthetic.index[-1].date()}")

# Load real TQQQ (raw prices with split adjustment)
real = pd.read_csv('data/raw/yahoo_raw/TQQQ_1d.csv', parse_dates=['date'])
real = real.set_index('date').sort_index()
print(f"Real TQQQ (raw): {real.index[0].date()} to {real.index[-1].date()}")

# Cutoff date: use synthetic before this, real from this date onwards
CUTOFF = pd.Timestamp('2010-02-09')  # Real TQQQ listing date

# Get synthetic data before cutoff
synthetic_before = synthetic[synthetic.index < CUTOFF].copy()
print(f"Synthetic before cutoff: {len(synthetic_before)} days")

# Get real data from cutoff onwards
real_from = real[real.index >= CUTOFF].copy()
print(f"Real from cutoff: {len(real_from)} days")

# Scale synthetic to match real at transition
# Find the last synthetic day and first real day
last_synthetic_date = synthetic_before.index[-1]
first_real_date = real_from.index[0]

last_synthetic_close = synthetic_before.loc[last_synthetic_date, 'close']
first_real_open = real_from.loc[first_real_date, 'open']

# Scale factor to make transition seamless
# We want: synthetic_close * scale_factor ≈ real_open
scale_factor = first_real_open / last_synthetic_close
print(f"\nTransition point:")
print(f"  Last synthetic ({last_synthetic_date.date()}): ${last_synthetic_close:.4f}")
print(f"  First real ({first_real_date.date()}): ${first_real_open:.4f}")
print(f"  Scale factor: {scale_factor:.6f}")

# Apply scale to synthetic
for col in ['open', 'high', 'low', 'close']:
    synthetic_before[col] = synthetic_before[col] * scale_factor

# Ensure real has all required columns
if 'split_factor' not in real_from.columns:
    real_from['split_factor'] = 1.0
if 'dividend' not in real_from.columns:
    real_from['dividend'] = 0.0
if 'split_ratio' not in real_from.columns:
    real_from['split_ratio'] = 1.0

# Align columns
cols = ['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']
synthetic_before = synthetic_before[cols]
real_from = real_from[cols]

# Merge
merged = pd.concat([synthetic_before, real_from])
merged = merged.sort_index()

# Add source column for debugging
merged['source'] = 'real'
merged.loc[merged.index < CUTOFF, 'source'] = 'synthetic'

print(f"\n=== Merged TQQQ ===")
print(f"Total days: {len(merged)}")
print(f"Date range: {merged.index[0].date()} to {merged.index[-1].date()}")
print(f"Synthetic days: {(merged['source'] == 'synthetic').sum()}")
print(f"Real days: {(merged['source'] == 'real').sum()}")

# Verify transition
print(f"\n=== Transition Verification ===")
around_cutoff = merged.loc['2010-02-05':'2010-02-12', ['close', 'source']]
print(around_cutoff)

# Save merged data
output_dir = 'data/raw/merged'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'TQQQ_merged_1d.csv')
merged.to_csv(output_path)
print(f"\nSaved: {output_path}")

# Also need to create merged versions of SPLG and BIL
# For now, we'll just use synthetic proxies for pre-2010 period
print("\n=== Creating merged SPLG and BIL ===")

# Load QQQ as proxy for generating synthetic SPLG (S&P 500 proxy)
# Actually, we need SPY data for SPLG. Let's check what we have.
# For simplicity, we'll use QQQ returns scaled appropriately

# Load real SPLG and BIL
splg_real = pd.read_csv('data/raw/yahoo_raw/SPLG_1d.csv', parse_dates=['date'])
splg_real = splg_real.set_index('date').sort_index()

bil_real = pd.read_csv('data/raw/yahoo_raw/BIL_1d.csv', parse_dates=['date'])
bil_real = bil_real.set_index('date').sort_index()

print(f"Real SPLG range: {splg_real.index[0].date()} to {splg_real.index[-1].date()}")
print(f"Real BIL range: {bil_real.index[0].date()} to {bil_real.index[-1].date()}")

# For pre-SPLG period, we need to create synthetic data
# SPLG tracks S&P 500, so ideally we'd use SPY data
# For now, let's check if we have SPY in stooq

spy_path = 'data/raw/stooq/us/spy.us.txt'
if os.path.exists(spy_path):
    print("Found SPY data in Stooq")
    spy = pd.read_csv(spy_path)
    spy.columns = [c.strip('<>').lower() for c in spy.columns]
    spy['date'] = pd.to_datetime(spy['date'], format='%Y%m%d')
    spy = spy.set_index('date').sort_index()
    spy = spy.rename(columns={'vol': 'volume'})
    
    # Use SPY directly as SPLG proxy for pre-SPLG period
    splg_cutoff = splg_real.index[0]
    spy_before = spy[spy.index < splg_cutoff].copy()
    
    # Scale to match transition
    if len(spy_before) > 0:
        last_spy = spy_before.iloc[-1]['close']
        first_splg = splg_real.iloc[0]['open']
        splg_scale = first_splg / last_spy
        
        for col in ['open', 'high', 'low', 'close']:
            spy_before[col] = spy_before[col] * splg_scale
        
        spy_before['volume'] = spy_before['volume']
        spy_before['dividend'] = 0.0
        spy_before['split_ratio'] = 1.0
        
        # Merge
        splg_merged = pd.concat([spy_before[['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']], 
                                  splg_real[['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']]])
        splg_merged = splg_merged.sort_index()
        splg_merged.to_csv(os.path.join(output_dir, 'SPLG_merged_1d.csv'))
        print(f"Saved: SPLG_merged_1d.csv (SPY proxy before {splg_cutoff.date()})")
else:
    print("SPY not found, using real SPLG only")
    splg_real[['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']].to_csv(
        os.path.join(output_dir, 'SPLG_merged_1d.csv'))

# For BIL (T-bill ETF), use constant growth rate for pre-BIL period
bil_cutoff = bil_real.index[0]
# Create synthetic BIL with ~0.1% monthly return (low risk-free rate)
dates_before_bil = merged.index[merged.index < bil_cutoff]
if len(dates_before_bil) > 0:
    bil_synthetic = pd.DataFrame(index=dates_before_bil)
    # Start at price that scales to first real BIL
    first_bil = bil_real.iloc[0]['open']
    days_before = len(dates_before_bil)
    daily_rate = 0.02 / 252  # ~2% annual (historical short-term rate)
    
    # Work backwards
    prices = [first_bil]
    for _ in range(days_before):
        prices.insert(0, prices[0] / (1 + daily_rate))
    prices = prices[:-1]  # Remove the duplicated first_bil
    
    bil_synthetic['close'] = prices
    bil_synthetic['open'] = bil_synthetic['close']
    bil_synthetic['high'] = bil_synthetic['close']
    bil_synthetic['low'] = bil_synthetic['close']
    bil_synthetic['volume'] = 0.0
    bil_synthetic['dividend'] = 0.0
    bil_synthetic['split_ratio'] = 1.0
    
    bil_merged = pd.concat([bil_synthetic, bil_real[['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']]])
    bil_merged = bil_merged.sort_index()
    bil_merged.to_csv(os.path.join(output_dir, 'BIL_merged_1d.csv'))
    print(f"Saved: BIL_merged_1d.csv (synthetic 2% annual before {bil_cutoff.date()})")
else:
    bil_real[['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio']].to_csv(
        os.path.join(output_dir, 'BIL_merged_1d.csv'))

print("\n=== Step 1 Complete ===")
print(f"Merged data saved to: {output_dir}/")
