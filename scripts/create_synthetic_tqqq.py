"""
Create Synthetic TQQQ from QQQ with realistic costs.

Formula:
  daily_return = 3 × QQQ_return - daily_borrowing_cost - daily_expense_ratio
  
Costs:
  - Expense ratio: 0.95% per year (TQQQ actual)
  - Borrowing cost: ~0.5% per year (Fed Funds rate proxy, varies)
  - Total: ~1.45% per year → ~0.0058% per day
"""
import os
import pandas as pd
import numpy as np

# Load QQQ data from Stooq
def load_stooq_price(ticker: str, path: str = "data/raw/stooq/us") -> pd.DataFrame:
    """Load Stooq price data."""
    file_path = os.path.join(path, f"{ticker.lower()}.us.txt")
    df = pd.read_csv(file_path)
    df.columns = [c.strip('<>').lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date').sort_index()
    df = df.rename(columns={'openint': 'openinterest'})
    return df[['open', 'high', 'low', 'close', 'vol']].rename(columns={'vol': 'volume'})

print("=== Loading QQQ from Stooq ===")
qqq = load_stooq_price("qqq")
print(f"QQQ data range: {qqq.index[0]} to {qqq.index[-1]}")
print(f"Total days: {len(qqq)}")

# Parameters
LEVERAGE = 3.0
EXPENSE_RATIO_ANNUAL = 0.0095  # 0.95% per year
BORROWING_COST_ANNUAL = 0.005  # 0.5% per year (conservative estimate)
TRADING_DAYS_PER_YEAR = 252

# Daily costs
DAILY_EXPENSE = EXPENSE_RATIO_ANNUAL / TRADING_DAYS_PER_YEAR
DAILY_BORROWING = BORROWING_COST_ANNUAL / TRADING_DAYS_PER_YEAR
DAILY_TOTAL_COST = DAILY_EXPENSE + DAILY_BORROWING

print(f"\n=== Cost Parameters ===")
print(f"Leverage: {LEVERAGE}x")
print(f"Expense ratio: {EXPENSE_RATIO_ANNUAL*100:.2f}% per year")
print(f"Borrowing cost: {BORROWING_COST_ANNUAL*100:.2f}% per year")
print(f"Total daily cost: {DAILY_TOTAL_COST*100:.4f}%")

# Calculate QQQ daily returns
qqq['return'] = qqq['close'].pct_change()

# Generate synthetic TQQQ
# synthetic_daily_return = LEVERAGE * QQQ_return - daily_costs
qqq['tqqq_return'] = LEVERAGE * qqq['return'] - DAILY_TOTAL_COST

# Build price series (start at 100 for convenience)
START_PRICE = 100.0
synthetic_tqqq = pd.DataFrame(index=qqq.index)
synthetic_tqqq['close'] = START_PRICE * (1 + qqq['tqqq_return']).cumprod()
synthetic_tqqq.loc[synthetic_tqqq.index[0], 'close'] = START_PRICE

# Generate OHLC (approximate: open=prev close, high/low based on qqq ratios)
synthetic_tqqq['open'] = synthetic_tqqq['close'].shift(1).fillna(START_PRICE)

# High/Low approximation using QQQ's intraday range scaled by leverage
qqq_range_ratio = (qqq['high'] / qqq['close']).fillna(1.0)
qqq_low_ratio = (qqq['low'] / qqq['close']).fillna(1.0)
synthetic_tqqq['high'] = synthetic_tqqq['close'] * (1 + LEVERAGE * (qqq_range_ratio - 1))
synthetic_tqqq['low'] = synthetic_tqqq['close'] * (1 + LEVERAGE * (qqq_low_ratio - 1))

# Ensure high >= close >= low
synthetic_tqqq['high'] = synthetic_tqqq[['high', 'close', 'open']].max(axis=1)
synthetic_tqqq['low'] = synthetic_tqqq[['low', 'close', 'open']].min(axis=1)

synthetic_tqqq['volume'] = qqq['volume'] * 0.1  # Approximate volume
synthetic_tqqq['dividend'] = 0.0
synthetic_tqqq['split_ratio'] = 1.0

# Verify against actual TQQQ (overlap period)
print("\n=== Verification against actual TQQQ ===")
try:
    actual_tqqq = pd.read_csv('data/raw/yahoo/TQQQ_1d.csv', parse_dates=['date'])
    actual_tqqq = actual_tqqq.set_index('date').sort_index()
    
    # Compare returns from 2012-01-01 to 2024-12-31
    start = '2012-01-01'
    end = '2024-12-31'
    
    actual_ret = actual_tqqq.loc[start:end, 'close'].pct_change()
    synth_ret = qqq.loc[start:end, 'tqqq_return']
    
    # Align indices
    common = actual_ret.index.intersection(synth_ret.index)
    actual_ret = actual_ret.loc[common]
    synth_ret = synth_ret.loc[common]
    
    corr = actual_ret.corr(synth_ret)
    print(f"Daily return correlation (2012-2024): {corr:.4f}")
    
    # Total return comparison
    actual_total = (1 + actual_ret).prod() - 1
    synth_total = (1 + synth_ret).prod() - 1
    print(f"Actual TQQQ total return: {actual_total*100:.1f}%")
    print(f"Synthetic TQQQ total return: {synth_total*100:.1f}%")
    print(f"Difference: {(synth_total - actual_total)*100:.1f}%")
except Exception as e:
    print(f"Could not compare: {e}")

# Save synthetic TQQQ
output_dir = 'data/raw/synthetic'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'TQQQ_synthetic_1d.csv')
synthetic_tqqq.to_csv(output_path)
print(f"\n=== Saved ===")
print(f"Synthetic TQQQ: {output_path}")
print(f"Date range: {synthetic_tqqq.index[0]} to {synthetic_tqqq.index[-1]}")

# Show sample
print("\n=== Sample Data ===")
print("First 5 rows:")
print(synthetic_tqqq.head())
print("\nLast 5 rows:")
print(synthetic_tqqq.tail())

# Check drawdowns
synthetic_tqqq['peak'] = synthetic_tqqq['close'].cummax()
synthetic_tqqq['drawdown'] = synthetic_tqqq['close'] / synthetic_tqqq['peak'] - 1
max_dd = synthetic_tqqq['drawdown'].min()
max_dd_date = synthetic_tqqq['drawdown'].idxmin()
print(f"\n=== Max Drawdown ===")
print(f"Max drawdown: {max_dd*100:.1f}% on {max_dd_date.date()}")

# Key periods
print("\n=== Key Historical Events ===")
events = [
    ('2000-03-10', 'Dot-com peak'),
    ('2002-10-09', 'Dot-com bottom'),
    ('2007-10-09', 'Pre-crisis peak'),
    ('2009-03-09', 'GFC bottom'),
    ('2020-02-19', 'Pre-COVID peak'),
    ('2020-03-23', 'COVID bottom'),
]
for date, event in events:
    if date in synthetic_tqqq.index:
        price = synthetic_tqqq.loc[date, 'close']
        dd = synthetic_tqqq.loc[date, 'drawdown']
        print(f"  {date} ({event}): ${price:.2f}, DD: {dd*100:.1f}%")
