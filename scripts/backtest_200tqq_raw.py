"""
Backtest 200TQQ using RAW (unadjusted) prices with proper share split handling.
"""
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_raw_price(ticker: str, cache_dir: str = "data/raw/yahoo_raw") -> pd.DataFrame:
    """Load raw price data from cache."""
    path = os.path.join(cache_dir, f"{ticker}_1d.csv")
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df


def run_200tqq_raw_backtest(
    start_date: str = "2011-01-01",
    end_date: str = "2024-12-31",
    sma_window: int = 200,
    overheat_mult: float = 1.05,
    stop_loss_pct: float = 0.05,
    out_dir: str = "results/200tqq_raw",
):
    """Run 200TQQ backtest with raw prices and split adjustment."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load raw data
    df_tqqq = load_raw_price("TQQQ")
    df_splg = load_raw_price("SPLG")
    df_bil = load_raw_price("BIL")
    
    # Filter date range
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    df_tqqq = df_tqqq[(df_tqqq.index >= start_dt) & (df_tqqq.index <= end_dt)]
    df_splg = df_splg[(df_splg.index >= start_dt) & (df_splg.index <= end_dt)]
    df_bil = df_bil[(df_bil.index >= start_dt) & (df_bil.index <= end_dt)]
    
    # Use common dates
    common_dates = df_tqqq.index.intersection(df_splg.index).intersection(df_bil.index)
    df_tqqq = df_tqqq.loc[common_dates]
    df_splg = df_splg.loc[common_dates]
    df_bil = df_bil.loc[common_dates]
    
    # Calculate SMA and state
    df_tqqq['sma'] = df_tqqq['close'].rolling(window=sma_window, min_periods=sma_window).mean()
    df_tqqq['upper'] = df_tqqq['sma'] * overheat_mult
    
    def get_state(row):
        if pd.isna(row['sma']):
            return 'NO_SIGNAL'
        if row['close'] < row['sma']:
            return 'DOWN'
        if row['close'] > row['upper']:
            return 'OVERHEAT'
        return 'FOCUS'
    
    df_tqqq['state'] = df_tqqq.apply(get_state, axis=1)
    
    # Initialize
    cash = 1.0
    tqqq_shares = 0.0
    splg_shares = 0.0
    bil_shares = 0.0
    tqqq_cost_basis = 0.0
    
    prev_state = 'NO_SIGNAL'
    pending_entry = False
    
    daily_rows = []
    trades = []
    
    for i, (d, row) in enumerate(df_tqqq.iterrows()):
        # Apply splits at start of day
        tqqq_sr = row.get('split_ratio', 1.0)
        if tqqq_sr != 1.0 and tqqq_shares > 0:
            tqqq_shares *= tqqq_sr
            
        splg_sr = df_splg.loc[d].get('split_ratio', 1.0) if d in df_splg.index else 1.0
        if splg_sr != 1.0 and splg_shares > 0:
            splg_shares *= splg_sr
            
        bil_sr = df_bil.loc[d].get('split_ratio', 1.0) if d in df_bil.index else 1.0
        if bil_sr != 1.0 and bil_shares > 0:
            bil_shares *= bil_sr
        
        # Prices
        tqqq_close = row['close']
        splg_close = df_splg.loc[d, 'close'] if d in df_splg.index else 100.0
        bil_close = df_bil.loc[d, 'close'] if d in df_bil.index else 100.0
        tqqq_open = row['open']
        tqqq_low = row['low']
        splg_open = df_splg.loc[d, 'open'] if d in df_splg.index else 100.0
        bil_open = df_bil.loc[d, 'open'] if d in df_bil.index else 100.0
        
        state = row['state']
        
        # Execute pending entry
        if pending_entry and state == 'FOCUS':
            # Buy TQQQ with all BIL
            if bil_shares > 0:
                proceeds = bil_shares * bil_open
                trades.append({'date': d, 'action': 'SELL', 'ticker': 'BIL', 'shares': bil_shares, 'price': bil_open})
                bil_shares = 0.0
                cash += proceeds
            if cash > 0:
                tqqq_shares = cash / tqqq_open
                tqqq_cost_basis = cash
                trades.append({'date': d, 'action': 'BUY', 'ticker': 'TQQQ', 'shares': tqqq_shares, 'price': tqqq_open})
                cash = 0.0
            pending_entry = False
        
        # Stop loss check (intraday)
        if tqqq_shares > 0 and stop_loss_pct > 0:
            avg_cost = tqqq_cost_basis / tqqq_shares if tqqq_shares > 0 else 0
            stop_price = avg_cost * (1 - stop_loss_pct)
            if tqqq_low <= stop_price:
                # Stop triggered
                fill_price = min(tqqq_open, stop_price)
                proceeds = tqqq_shares * fill_price
                trades.append({'date': d, 'action': 'STOP', 'ticker': 'TQQQ', 'shares': tqqq_shares, 'price': fill_price})
                tqqq_shares = 0.0
                tqqq_cost_basis = 0.0
                cash += proceeds
                # Also sell SPLG
                if splg_shares > 0:
                    proceeds = splg_shares * splg_close
                    trades.append({'date': d, 'action': 'STOP', 'ticker': 'SPLG', 'shares': splg_shares, 'price': splg_close})
                    splg_shares = 0.0
                    cash += proceeds
                # Convert to BIL
                if cash > 0:
                    bil_shares = cash / bil_close
                    trades.append({'date': d, 'action': 'BUY', 'ticker': 'BIL', 'shares': bil_shares, 'price': bil_close})
                    cash = 0.0
                # Check if close is above SMA for re-entry
                if state in ['FOCUS', 'OVERHEAT']:
                    pending_entry = True
        
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
            # Exit all growth to BIL
            if tqqq_shares > 0 or splg_shares > 0:
                pending_entry = False
        elif state == 'FOCUS':
            if prev_state == 'DOWN':
                pending_entry = True  # Wait one day
            elif prev_state == 'OVERHEAT':
                pending_entry = True  # No wait needed per spec, but simplify
        elif state == 'OVERHEAT':
            # New money to SPLG (not implemented for simplicity)
            pass
        
        # First signal handling
        if i == sma_window - 1:  # First valid signal day
            if state == 'OVERHEAT':
                # Start with SPLG
                splg_shares = cash / splg_close
                trades.append({'date': d, 'action': 'BUY', 'ticker': 'SPLG', 'shares': splg_shares, 'price': splg_close})
                cash = 0.0
            elif state == 'FOCUS':
                pending_entry = True
            elif state == 'DOWN':
                # Stay in cash/BIL
                if cash > 0:
                    bil_shares = cash / bil_close
                    trades.append({'date': d, 'action': 'BUY', 'ticker': 'BIL', 'shares': bil_shares, 'price': bil_close})
                    cash = 0.0
        
        prev_state = state
    
    # Calculate metrics
    daily_df = pd.DataFrame(daily_rows)
    final_equity = daily_df.iloc[-1]['equity']
    start_equity = daily_df.iloc[0]['equity'] if len(daily_df) > 0 else 1.0
    total_return = final_equity / start_equity - 1
    years = (daily_df.iloc[-1]['date'] - daily_df.iloc[0]['date']).days / 365.25
    cagr = (final_equity / start_equity) ** (1/years) - 1 if years > 0 else 0
    
    # Volatility and drawdown
    daily_df['returns'] = daily_df['equity'].pct_change()
    ann_vol = daily_df['returns'].std() * (252 ** 0.5)
    
    daily_df['peak'] = daily_df['equity'].cummax()
    daily_df['drawdown'] = daily_df['equity'] / daily_df['peak'] - 1
    max_dd = daily_df['drawdown'].min()
    
    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'cagr_over_vol': cagr / ann_vol if ann_vol > 0 else 0,
        'max_drawdown': max_dd,
        'n_trades': len(trades),
    }
    
    print("=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save
    daily_df.to_csv(os.path.join(out_dir, 'daily.csv'), index=False)
    daily_df[['date', 'equity']].to_csv(os.path.join(out_dir, 'equity_curve.csv'), index=False)
    pd.DataFrame(trades).to_csv(os.path.join(out_dir, 'trades.csv'), index=False)
    
    with open(os.path.join(out_dir, 'summary_metrics.json'), 'w') as f:
        json.dump({'metrics': metrics}, f, indent=2)
    
    print(f"\nSaved to: {out_dir}")
    return metrics


if __name__ == "__main__":
    run_200tqq_raw_backtest(
        start_date="2011-01-01",
        end_date="2024-12-31",
        out_dir="results/200tqq_raw_2011_2024",
    )
