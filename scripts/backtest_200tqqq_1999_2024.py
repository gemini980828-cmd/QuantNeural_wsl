"""
Step 2: Run 200TQQ strategy backtest from 1999 to 2024 using merged data.
"""
import json
import os
import pandas as pd
import numpy as np

print("=== Step 2: 200TQQ Backtest 1999-2024 ===")

# Load merged data
data_dir = 'data/raw/merged'

tqqq = pd.read_csv(os.path.join(data_dir, 'TQQQ_merged_1d.csv'), parse_dates=['date'])
tqqq = tqqq.set_index('date').sort_index()

splg = pd.read_csv(os.path.join(data_dir, 'SPLG_merged_1d.csv'), parse_dates=['date'])
splg = splg.set_index('date').sort_index()

bil = pd.read_csv(os.path.join(data_dir, 'BIL_merged_1d.csv'), parse_dates=['date'])
bil = bil.set_index('date').sort_index()

print(f"TQQQ: {tqqq.index[0].date()} to {tqqq.index[-1].date()}")
print(f"SPLG: {splg.index[0].date()} to {splg.index[-1].date()}")
print(f"BIL: {bil.index[0].date()} to {bil.index[-1].date()}")

# Parameters
SMA_WINDOW = 200
OVERHEAT_MULT = 1.05
STOP_LOSS_PCT = 0.05

# Use common dates
common_dates = tqqq.index.intersection(splg.index).intersection(bil.index)
tqqq = tqqq.loc[common_dates]
splg = splg.loc[common_dates]
bil = bil.loc[common_dates]

print(f"Common dates: {len(common_dates)} days")
print(f"Date range: {common_dates[0].date()} to {common_dates[-1].date()}")

# Calculate SMA and state
tqqq['sma'] = tqqq['close'].rolling(window=SMA_WINDOW, min_periods=SMA_WINDOW).mean()
tqqq['upper'] = tqqq['sma'] * OVERHEAT_MULT

def get_state(row):
    if pd.isna(row['sma']):
        return 'NO_SIGNAL'
    if row['close'] < row['sma']:
        return 'DOWN'
    if row['close'] > row['upper']:
        return 'OVERHEAT'
    return 'FOCUS'

tqqq['state'] = tqqq.apply(get_state, axis=1)

# Count states
state_counts = tqqq['state'].value_counts()
print(f"\nState distribution:")
for s, c in state_counts.items():
    print(f"  {s}: {c} ({c/len(tqqq)*100:.1f}%)")

# Backtest
cash = 1.0
tqqq_shares = 0.0
splg_shares = 0.0
bil_shares = 0.0
tqqq_cost_basis = 0.0

prev_state = 'NO_SIGNAL'
pending_entry = False
start_consumed = False

daily_rows = []
trades = []

for i, (d, row) in enumerate(tqqq.iterrows()):
    # Apply splits at start of day
    tqqq_sr = row.get('split_ratio', 1.0)
    if tqqq_sr != 1.0 and tqqq_shares > 0:
        tqqq_shares *= tqqq_sr
        
    splg_sr = splg.loc[d].get('split_ratio', 1.0) if d in splg.index else 1.0
    if splg_sr != 1.0 and splg_shares > 0:
        splg_shares *= splg_sr
        
    bil_sr = bil.loc[d].get('split_ratio', 1.0) if d in bil.index else 1.0
    if bil_sr != 1.0 and bil_shares > 0:
        bil_shares *= bil_sr
    
    # Prices
    tqqq_close = row['close']
    splg_close = splg.loc[d, 'close'] if d in splg.index else 100.0
    bil_close = bil.loc[d, 'close'] if d in bil.index else 100.0
    tqqq_open = row['open']
    tqqq_low = row['low']
    splg_open = splg.loc[d, 'open'] if d in splg.index else 100.0
    bil_open = bil.loc[d, 'open'] if d in bil.index else 100.0
    
    state = row['state']
    
    # Execute pending entry at open
    if pending_entry and state in ['FOCUS', 'OVERHEAT']:
        # Sell BIL/SPLG
        if bil_shares > 0:
            proceeds = bil_shares * bil_open
            trades.append({'date': d, 'action': 'SELL', 'ticker': 'BIL', 'shares': bil_shares, 'price': bil_open})
            bil_shares = 0.0
            cash += proceeds
        if splg_shares > 0:
            proceeds = splg_shares * splg_open
            trades.append({'date': d, 'action': 'SELL', 'ticker': 'SPLG', 'shares': splg_shares, 'price': splg_open})
            splg_shares = 0.0
            cash += proceeds
        # Buy TQQQ
        if cash > 0:
            tqqq_shares = cash / tqqq_open
            tqqq_cost_basis = cash
            trades.append({'date': d, 'action': 'BUY', 'ticker': 'TQQQ', 'shares': tqqq_shares, 'price': tqqq_open})
            cash = 0.0
        pending_entry = False
    
    # Stop loss check
    if tqqq_shares > 0 and STOP_LOSS_PCT > 0:
        avg_cost = tqqq_cost_basis / tqqq_shares if tqqq_shares > 0 else 0
        stop_price = avg_cost * (1 - STOP_LOSS_PCT)
        if tqqq_low <= stop_price:
            fill_price = min(tqqq_open, stop_price)
            proceeds = tqqq_shares * fill_price
            trades.append({'date': d, 'action': 'STOP', 'ticker': 'TQQQ', 'shares': tqqq_shares, 'price': fill_price})
            tqqq_shares = 0.0
            tqqq_cost_basis = 0.0
            cash += proceeds
            # Sell SPLG too
            if splg_shares > 0:
                proceeds = splg_shares * splg_close
                trades.append({'date': d, 'action': 'STOP', 'ticker': 'SPLG', 'shares': splg_shares, 'price': splg_close})
                splg_shares = 0.0
                cash += proceeds
            # Go to BIL
            if cash > 0:
                bil_shares = cash / bil_close
                trades.append({'date': d, 'action': 'BUY', 'ticker': 'BIL', 'shares': bil_shares, 'price': bil_close})
                cash = 0.0
            # Check re-entry
            if state in ['FOCUS', 'OVERHEAT']:
                pending_entry = True
    
    # First signal handling (one-time)
    if state != 'NO_SIGNAL' and not start_consumed:
        start_consumed = True
        if state == 'OVERHEAT':
            # Start with SPLG
            if cash > 0 or bil_shares > 0:
                if bil_shares > 0:
                    cash += bil_shares * bil_close
                    bil_shares = 0.0
                splg_shares = cash / splg_close
                trades.append({'date': d, 'action': 'BUY', 'ticker': 'SPLG', 'shares': splg_shares, 'price': splg_close})
                cash = 0.0
        elif state == 'FOCUS':
            pending_entry = True  # Wait one more day
        elif state == 'DOWN':
            # Stay in safe
            if cash > 0:
                bil_shares = cash / bil_close
                trades.append({'date': d, 'action': 'BUY', 'ticker': 'BIL', 'shares': bil_shares, 'price': bil_close})
                cash = 0.0
    
    # Close valuation
    equity = cash + tqqq_shares * tqqq_close + splg_shares * splg_close + bil_shares * bil_close
    
    daily_rows.append({
        'date': d,
        'state': state,
        'equity': equity,
        'tqqq_shares': tqqq_shares,
        'splg_shares': splg_shares,
        'bil_shares': bil_shares,
        'cash': cash,
    })
    
    # End of day: determine next action
    if state == 'NO_SIGNAL':
        pass
    elif state == 'DOWN':
        if tqqq_shares > 0 or splg_shares > 0:
            pending_entry = False  # Will exit at next open
    elif state == 'FOCUS':
        if prev_state == 'DOWN' and tqqq_shares <= 0:
            pending_entry = True
        elif prev_state == 'OVERHEAT':
            pending_entry = True  # Immediate entry from OVERHEAT
    
    prev_state = state

# Calculate metrics
daily_df = pd.DataFrame(daily_rows)
final_equity = daily_df.iloc[-1]['equity']
start_equity = 1.0  # Initial investment
total_return = final_equity / start_equity - 1
years = (daily_df.iloc[-1]['date'] - daily_df.iloc[0]['date']).days / 365.25
cagr = (final_equity / start_equity) ** (1/years) - 1 if years > 0 else 0

daily_df['returns'] = daily_df['equity'].pct_change()
ann_vol = daily_df['returns'].std() * (252 ** 0.5)

daily_df['peak'] = daily_df['equity'].cummax()
daily_df['drawdown'] = daily_df['equity'] / daily_df['peak'] - 1
max_dd = daily_df['drawdown'].min()
max_dd_date = daily_df.loc[daily_df['drawdown'].idxmin(), 'date']

metrics = {
    'total_return': total_return,
    'cagr': cagr,
    'ann_vol': ann_vol,
    'cagr_over_vol': cagr / ann_vol if ann_vol > 0 else 0,
    'max_drawdown': max_dd,
    'max_dd_date': str(max_dd_date.date()),
    'n_trades': len(trades),
    'years': years,
}

print("\n=== Results: 200TQQ Strategy 1999-2024 ===")
print(f"Total Return: {total_return*100:.1f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Annual Volatility: {ann_vol*100:.1f}%")
print(f"CAGR/Vol: {metrics['cagr_over_vol']:.2f}")
print(f"Max Drawdown: {max_dd*100:.1f}% (on {max_dd_date.date()})")
print(f"Total Trades: {len(trades)}")
print(f"Years: {years:.1f}")

# Save results
output_dir = 'results/200tqq_1999_2024'
os.makedirs(output_dir, exist_ok=True)

daily_df.to_csv(os.path.join(output_dir, 'daily.csv'), index=False)
daily_df[['date', 'equity']].to_csv(os.path.join(output_dir, 'equity_curve.csv'), index=False)
pd.DataFrame(trades).to_csv(os.path.join(output_dir, 'trades.csv'), index=False)

with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
    json.dump({'metrics': metrics}, f, indent=2)

print(f"\nSaved to: {output_dir}/")

# Compare with Buy-and-Hold
print("\n=== Comparison: Buy-and-Hold ===")
tqqq_bh_return = tqqq.iloc[-1]['close'] / tqqq.iloc[0]['close'] - 1
tqqq_bh_cagr = (tqqq.iloc[-1]['close'] / tqqq.iloc[0]['close']) ** (1/years) - 1
print(f"TQQQ Buy-and-Hold CAGR: {tqqq_bh_cagr*100:.2f}%")
print(f"Strategy CAGR: {cagr*100:.2f}%")
print(f"Strategy vs B&H: {(cagr - tqqq_bh_cagr)*100:+.2f}% per year")
