"""Compare TQQQ Buy-and-Hold vs 200TQQ Strategy returns."""
import pandas as pd

# Load TQQQ price data
tqqq = pd.read_csv('data/raw/yahoo/TQQQ_1d.csv', parse_dates=['date'])
tqqq = tqqq.set_index('date').sort_index()

# Filter to 2011-2024
tqqq = tqqq.loc['2011-01-01':'2024-12-31']

print("=== TQQQ Price Data Range ===")
print(f"Start: {tqqq.index[0]} - Close: ${tqqq.iloc[0]['close']:.4f}")
print(f"End: {tqqq.index[-1]} - Close: ${tqqq.iloc[-1]['close']:.4f}")

# Calculate buy-and-hold return
start_price = tqqq.iloc[0]['close']
end_price = tqqq.iloc[-1]['close']
total_return = end_price / start_price - 1
years = (tqqq.index[-1] - tqqq.index[0]).days / 365.25
cagr = (end_price / start_price) ** (1/years) - 1

print(f"\n=== TQQQ Buy-and-Hold (Split-Adjusted Prices) ===")
print(f"Total Return: {total_return*100:.1f}%")
print(f"Years: {years:.2f}")
print(f"CAGR: {cagr*100:.2f}%")

# Check if this makes sense
print(f"\n=== Sanity Check ===")
print(f"TQQQ is 3x leveraged QQQ. If QQQ returned ~15%/year,")
print(f"TQQQ should return ~30-40%/year in bull market (minus decay).")
print(f"Our calculated TQQQ Buy-and-Hold CAGR: {cagr*100:.2f}%")

# Load strategy equity curve
eq = pd.read_csv('results/200tqq_official_2011_2024/equity_curve.csv', parse_dates=['date'])
eq = eq.set_index('date')
strategy_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
strategy_years = (eq.index[-1] - eq.index[0]).days / 365.25
strategy_cagr = (eq['equity'].iloc[-1] / eq['equity'].iloc[0]) ** (1/strategy_years) - 1

print(f"\n=== 200TQQ Strategy ===")
print(f"Total Return: {strategy_return*100:.1f}%")
print(f"CAGR: {strategy_cagr*100:.2f}%")

print(f"\n=== Comparison ===")
print(f"TQQQ B&H CAGR: {cagr*100:.2f}%")
print(f"200TQQ Strategy CAGR: {strategy_cagr*100:.2f}%")
print(f"Strategy underperformance: {(cagr - strategy_cagr)*100:.2f}% per year")
