"""
Run official 200TQQ backtest 1999-2024 with ALL rules using merged data.
Uses the main strategy engine with full official settings.
"""
import json
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.strategy_200tqq import Strategy200TQQConfig, run_200tqq_backtest

print("=== Official 200TQQ Backtest 1999-2024 ===")

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

# Prepare data dict for strategy engine
# Add required columns if missing
for df in [tqqq, splg, bil]:
    if 'dividend' not in df.columns:
        df['dividend'] = 0.0
    if 'split_ratio' not in df.columns:
        df['split_ratio'] = 1.0

data_by_ticker = {
    'TQQQ': tqqq,
    'SPLG': splg,
    'BIL': bil,
}

# Official config with ALL rules
cfg = Strategy200TQQConfig(
    start_date='1999-03-10',
    end_date='2024-12-31',
    initial_equity=1.0,
    tqqq_ticker='TQQQ',
    splg_ticker='SPLG',
    safe_ticker='SGOV',  # Will fall back to proxy
    safe_proxy_ticker='BIL',
    sma_window=200,
    overheat_mult=1.05,
    apply_entry_confirmation=True,  # DOWN→FOCUS 하루 더 확인
    monthly_contribution=0.0,
    stop_loss_pct=0.05,
    cost_bps=0.0,
    slippage_bps=0.0,
    take_profit_mode='official',  # 공식 익절 (10%, 25%, 50%, 100%+)
    take_profit_reinvest='all_splg',  # 익절금 SPLG 재투자
    initial_mode='strategy',
    overheat_start_splg_weight=1.0,
)

print(f"\n=== Config ===")
print(f"  take_profit_mode: {cfg.take_profit_mode}")
print(f"  take_profit_reinvest: {cfg.take_profit_reinvest}")
print(f"  apply_entry_confirmation: {cfg.apply_entry_confirmation}")
print(f"  stop_loss_pct: {cfg.stop_loss_pct}")
print(f"  sma_window: {cfg.sma_window}")
print(f"  overheat_mult: {cfg.overheat_mult}")

# Run backtest
result = run_200tqq_backtest(data_by_ticker, cfg=cfg)

# Extract results
metrics = result['metrics']
daily = result['daily']
trades = result['trades']

print(f"\n=== Results: Official 200TQQ 1999-2024 ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# Save results
output_dir = 'results/200tqq_official_1999_2024'
os.makedirs(output_dir, exist_ok=True)

daily_df = daily.reset_index()
daily_df.to_csv(os.path.join(output_dir, 'daily.csv'), index=False)

equity_curve = daily['equity'].reset_index()
equity_curve.columns = ['date', 'equity']
equity_curve.to_csv(os.path.join(output_dir, 'equity_curve.csv'), index=False)

trades.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)

with open(os.path.join(output_dir, 'summary_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump({'metrics': metrics, 'config': cfg.__dict__}, f, indent=2)

print(f"\nSaved to: {output_dir}/")

# Count trade types
print(f"\n=== Trade Breakdown ===")
print(trades['action'].value_counts())
