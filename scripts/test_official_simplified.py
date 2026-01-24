"""Test official engine with simplified settings to match the simple script."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.strategy_200tqq import Strategy200TQQConfig, run_200tqq_backtest

# Load merged data
data_dir = 'data/raw/merged'
tqqq = pd.read_csv(os.path.join(data_dir, 'TQQQ_merged_1d.csv'), parse_dates=['date']).set_index('date').sort_index()
splg = pd.read_csv(os.path.join(data_dir, 'SPLG_merged_1d.csv'), parse_dates=['date']).set_index('date').sort_index()
bil = pd.read_csv(os.path.join(data_dir, 'BIL_merged_1d.csv'), parse_dates=['date']).set_index('date').sort_index()

for df in [tqqq, splg, bil]:
    if 'dividend' not in df.columns: df['dividend'] = 0.0
    if 'split_ratio' not in df.columns: df['split_ratio'] = 1.0

data_by_ticker = {'TQQQ': tqqq, 'SPLG': splg, 'BIL': bil}

print("=== Test 1: Simplified Config (no take_profit, no entry confirmation) ===")
cfg1 = Strategy200TQQConfig(
    start_date='1999-03-10',
    end_date='2024-12-31',
    initial_equity=1.0,
    tqqq_ticker='TQQQ',
    splg_ticker='SPLG',
    safe_ticker='SGOV',
    safe_proxy_ticker='BIL',
    sma_window=200,
    overheat_mult=1.05,
    apply_entry_confirmation=False,
    monthly_contribution=0.0,
    stop_loss_pct=0.05,
    cost_bps=0.0,
    slippage_bps=0.0,
    take_profit_mode='none',
    take_profit_reinvest='all_splg',
    initial_mode='strategy',
    overheat_start_splg_weight=1.0,
)

result1 = run_200tqq_backtest(data_by_ticker, cfg=cfg1)
m1 = result1['metrics']
print(f"CAGR: {m1['cagr']*100:.2f}%")
print(f"Total Return: {m1['total_return']*100:.1f}%")
print(f"Max Drawdown: {m1['max_drawdown']*100:.1f}%")
print(f"Trades: {m1['n_trades']}")

print("\n=== Test 2: No stop-loss ===")
cfg2 = Strategy200TQQConfig(
    start_date='1999-03-10',
    end_date='2024-12-31',
    initial_equity=1.0,
    tqqq_ticker='TQQQ',
    splg_ticker='SPLG',
    safe_ticker='SGOV',
    safe_proxy_ticker='BIL',
    sma_window=200,
    overheat_mult=1.05,
    apply_entry_confirmation=False,
    monthly_contribution=0.0,
    stop_loss_pct=0.0,  # No stop loss
    cost_bps=0.0,
    slippage_bps=0.0,
    take_profit_mode='none',
    take_profit_reinvest='all_splg',
    initial_mode='strategy',
    overheat_start_splg_weight=1.0,
)

result2 = run_200tqq_backtest(data_by_ticker, cfg=cfg2)
m2 = result2['metrics']
print(f"CAGR: {m2['cagr']*100:.2f}%")
print(f"Total Return: {m2['total_return']*100:.1f}%")
print(f"Max Drawdown: {m2['max_drawdown']*100:.1f}%")
print(f"Trades: {m2['n_trades']}")
