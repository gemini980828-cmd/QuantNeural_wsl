"""
Create raw (unadjusted) prices for all 200TQQ strategy tickers.
"""
import pandas as pd
import os

tickers = ['TQQQ', 'SPLG', 'BIL', 'SGOV']
output_dir = 'data/raw/yahoo_raw'
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    input_path = f'data/raw/yahoo/{ticker}_1d.csv'
    if not os.path.exists(input_path):
        print(f"SKIP: {ticker} - file not found")
        continue
    
    df = pd.read_csv(input_path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # Get splits
    splits = df[df['split_ratio'] != 1.0]
    total_split = splits['split_ratio'].prod() if len(splits) > 0 else 1.0
    
    print(f"=== {ticker} ===")
    print(f"  Splits: {len(splits)}")
    if len(splits) > 0:
        for dt, row in splits.iterrows():
            print(f"    {dt.date()}: {row['split_ratio']:.2f}x")
    print(f"  Total factor: {total_split:.1f}x")
    
    # Calculate cumulative split factor going backwards
    cumulative = 1.0
    factors = []
    for dt in reversed(df.index):
        factors.append(cumulative)
        sr = df.loc[dt, 'split_ratio']
        if sr != 1.0:
            cumulative *= sr
    factors = list(reversed(factors))
    df['split_factor'] = factors
    
    # Calculate raw prices
    df['raw_open'] = df['open'] * df['split_factor']
    df['raw_high'] = df['high'] * df['split_factor']
    df['raw_low'] = df['low'] * df['split_factor']
    df['raw_close'] = df['close'] * df['split_factor']
    
    # Verify returns are unchanged
    adj_ret = df.iloc[-1]['close'] / df.iloc[0]['close']
    raw_ret = df.iloc[-1]['raw_close'] / df.iloc[0]['raw_close']
    print(f"  Return ratio check: adj={adj_ret:.4f}, raw={raw_ret:.4f}, match={abs(adj_ret-raw_ret)<0.001}")
    
    # Save
    output = df[['raw_open', 'raw_high', 'raw_low', 'raw_close', 'volume', 'dividend', 'split_ratio', 'split_factor']].copy()
    output.columns = ['open', 'high', 'low', 'close', 'volume', 'dividend', 'split_ratio', 'split_factor']
    output_path = os.path.join(output_dir, f'{ticker}_1d.csv')
    output.to_csv(output_path)
    print(f"  Saved: {output_path}")
    print()

print("Done! Raw prices saved to:", output_dir)
