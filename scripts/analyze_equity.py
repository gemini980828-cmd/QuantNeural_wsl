"""Deep dive into equity curve behavior."""
import pandas as pd

# Load equity curve
eq = pd.read_csv('results/200tqq_official_2011_2024/equity_curve.csv', parse_dates=['date'])
eq = eq.set_index('date')

# Load daily details
daily = pd.read_csv('results/200tqq_official_2011_2024/daily.csv', parse_dates=['date'])
daily = daily.set_index('date')

# Load trades
trades = pd.read_csv('results/200tqq_official_2011_2024/trades.csv', parse_dates=['date'])

print("=== Equity Curve Overview ===")
print(f"Start: {eq.index[0]} - Equity: {eq.iloc[0]['equity']:.4f}")
print(f"End: {eq.index[-1]} - Equity: {eq.iloc[-1]['equity']:.4f}")
print(f"Total Return: {(eq.iloc[-1]['equity']/eq.iloc[0]['equity'] - 1)*100:.1f}%")

print("\n=== First 10 Days ===")
print(daily[['state', 'equity', 'TQQQ_shares', 'SPLG_shares', 'cash']].head(10))

print("\n=== Trade Summary ===")
print(trades['action'].value_counts())

# Check how much time spent in each state
print("\n=== Time in Each State ===")
state_counts = daily['state'].value_counts()
print(state_counts)
print(f"\nDOWN fraction: {state_counts.get('DOWN', 0) / len(daily) * 100:.1f}%")
print(f"FOCUS fraction: {state_counts.get('FOCUS', 0) / len(daily) * 100:.1f}%")
print(f"OVERHEAT fraction: {state_counts.get('OVERHEAT', 0) / len(daily) * 100:.1f}%")

# When did we first enter TQQQ?
first_tqqq_buy = trades[(trades['action'] == 'BUY') & (trades['ticker'] == 'TQQQ')].iloc[0] if len(trades[(trades['action'] == 'BUY') & (trades['ticker'] == 'TQQQ')]) > 0 else None
if first_tqqq_buy is not None:
    print(f"\n=== First TQQQ Entry ===")
    print(first_tqqq_buy)
