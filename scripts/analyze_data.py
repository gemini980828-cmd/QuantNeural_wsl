"""Analyze data files."""
import pandas as pd

# Load files
prices = pd.read_csv('data/backtest_universe_full/prices.csv')
scores = pd.read_csv('data/backtest_universe_full/scores.csv')

print('=== PRICES.CSV ===')
print(f'Shape: {prices.shape}')
print(f'Columns count: {len(prices.columns)}')
print(f'First 5 columns: {list(prices.columns[:5])}')
print(f'Date range: {prices["date"].min()} to {prices["date"].max()}')
print(f'Ticker columns: {len(prices.columns) - 1}')

print()
print('=== SCORES.CSV ===')
print(f'Shape: {scores.shape}')
print(f'Columns count: {len(scores.columns)}')
print(f'First 5 columns: {list(scores.columns[:5])}')
print(f'Date range: {scores["date"].min()} to {scores["date"].max()}')
print(f'Ticker columns: {len(scores.columns) - 1}')

# Check for duplicates
print()
print('=== DUPLICATE CHECKS ===')
print(f'Prices duplicate dates: {prices["date"].duplicated().sum()}')
print(f'Scores duplicate dates: {scores["date"].duplicated().sum()}')

# Check ticker overlap
prices_tickers = set(prices.columns) - {'date'}
scores_tickers = set(scores.columns) - {'date'}
common = prices_tickers & scores_tickers
only_prices = prices_tickers - scores_tickers
only_scores = scores_tickers - prices_tickers

print()
print('=== TICKER ALIGNMENT ===')
print(f'Prices tickers: {len(prices_tickers)}')
print(f'Scores tickers: {len(scores_tickers)}')
print(f'Common tickers: {len(common)}')
print(f'Only in prices: {len(only_prices)}')
print(f'Only in scores: {len(only_scores)}')

if only_scores:
    print(f'Sample tickers only in scores: {list(only_scores)[:10]}')
