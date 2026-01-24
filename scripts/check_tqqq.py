import pandas as pd

t = pd.read_csv('data/raw/yahoo/TQQQ_1d.csv', parse_dates=['date'])
t = t.set_index('date').sort_index()
t = t.loc['2012-01-01':'2024-12-31']

print(f"Start: {t.index[0]} @ {t.iloc[0]['close']:.4f}")
print(f"End: {t.index[-1]} @ {t.iloc[-1]['close']:.4f}")

r = t.iloc[-1]['close'] / t.iloc[0]['close']
y = (t.index[-1] - t.index[0]).days / 365.25

print(f"Total Return: {(r-1)*100:.1f}%")
print(f"Years: {y:.2f}")
print(f"TQQQ Buy-and-Hold CAGR: {(r**(1/y)-1)*100:.2f}%")
