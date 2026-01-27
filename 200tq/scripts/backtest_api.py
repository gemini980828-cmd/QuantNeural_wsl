#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

COST_BPS = 10
TAX_RATE = 0.22
SHORT_MA = 3
OFF_TQQQ_WEIGHT = 0.10
TRADING_DAYS = 252

STRATEGY_CONFIG = {
    "200TQ": {"name": "200TQ_Original", "off_asset": "CASH", "signal_type": "tq200", "ensemble_windows": [], "use_ma200_guard": False, "off_tqqq_weight": 0.0},
    "E00": {"name": "E00_V0_Base_OFF10_CASH", "off_asset": "CASH", "signal_type": "plain", "ensemble_windows": [], "use_ma200_guard": False},
    "E01": {"name": "E01_V1_OFF10_SGOV", "off_asset": "SGOV", "signal_type": "plain", "ensemble_windows": [], "use_ma200_guard": False},
    "E02": {"name": "E02_V4_Ensemble_160_165_170", "off_asset": "CASH", "signal_type": "ensemble", "ensemble_windows": [160, 165, 170], "use_ma200_guard": False},
    "E03": {"name": "E03_Ensemble_SGOV", "off_asset": "SGOV", "signal_type": "ensemble", "ensemble_windows": [160, 165, 170], "use_ma200_guard": False},
    "E04": {"name": "E04_Hysteresis_0p25", "off_asset": "CASH", "signal_type": "hysteresis", "hysteresis_band": 0.0025, "ensemble_windows": [], "use_ma200_guard": False},
    "E05": {"name": "E05_Hysteresis_0p50", "off_asset": "CASH", "signal_type": "hysteresis", "hysteresis_band": 0.005, "ensemble_windows": [], "use_ma200_guard": False},
    "E06": {"name": "E06_Hysteresis_1p00", "off_asset": "CASH", "signal_type": "hysteresis", "hysteresis_band": 0.01, "ensemble_windows": [], "use_ma200_guard": False},
    "E07": {"name": "E07_MA200Guard_CASH", "off_asset": "CASH", "signal_type": "plain", "ensemble_windows": [], "use_ma200_guard": True},
    "E08": {"name": "E08_MA200Guard_SGOV", "off_asset": "SGOV", "signal_type": "plain", "ensemble_windows": [], "use_ma200_guard": True},
    "E09": {"name": "E09_BuyConfirm_2days", "off_asset": "CASH", "signal_type": "plain", "ensemble_windows": [], "use_ma200_guard": False, "buy_confirm_days": 2},
    "E10": {"name": "E10_BuySellConfirm_2days", "off_asset": "CASH", "signal_type": "plain", "ensemble_windows": [], "use_ma200_guard": False, "buy_confirm_days": 2, "sell_confirm_days": 2},
}

def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf
    
    buffer_start = pd.to_datetime(start_date) - pd.Timedelta(days=400)
    
    tickers = ["QQQ", "TQQQ", "SGOV", "SHV"]
    raw = yf.download(
        tickers=tickers,
        start=buffer_start.strftime("%Y-%m-%d"),
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    
    prices = pd.DataFrame()
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if (t, "Close") in raw.columns:
                prices[t] = raw[(t, "Close")]
    
    prices["CASH"] = 100.0
    
    if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
        if "SHV" in prices.columns:
            prices["SGOV"] = prices["SHV"]
        else:
            prices["SGOV"] = 100.0
    
    prices["SGOV"] = prices["SGOV"].ffill()
    if "SHV" in prices.columns:
        prices["SHV"] = prices["SHV"].ffill()
    
    prices = prices.dropna(subset=["QQQ", "TQQQ"])
    
    return prices, start_date, end_date

def generate_plain_signal(prices: pd.DataFrame, long_ma: int = 161) -> pd.Series:
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    ma_long = qqq.rolling(long_ma).mean()
    return (ma_short > ma_long).astype(int)

def generate_ensemble_signal(prices: pd.DataFrame, windows: list) -> pd.Series:
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    
    votes = pd.DataFrame(index=prices.index)
    for lw in windows:
        ma_long = qqq.rolling(lw).mean()
        votes[f"w{lw}"] = (ma_short > ma_long).astype(int)
    
    threshold = len(windows) // 2 + 1
    return (votes.sum(axis=1) >= threshold).astype(int)

def generate_hysteresis_signal(prices: pd.DataFrame, band: float) -> pd.Series:
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(SHORT_MA).mean()
    ma_long = qqq.rolling(161).mean()
    spread = (ma_short / ma_long) - 1.0
    
    signal = pd.Series(index=prices.index, dtype=int)
    state = 0
    
    for dt in prices.index:
        if pd.isna(spread.loc[dt]):
            signal.loc[dt] = state
            continue
        s = spread.loc[dt]
        if s > band:
            state = 1
        elif s < -band:
            state = 0
        signal.loc[dt] = state
    
    return signal

def apply_confirmation(signal: pd.Series, buy_days: int, sell_days: int) -> pd.Series:
    if buy_days == 1 and sell_days == 1:
        return signal
    
    result = pd.Series(index=signal.index, dtype=int)
    current_state = 0
    pending_state = None
    pending_count = 0
    
    for dt in signal.index:
        target = signal.loc[dt]
        
        if target != current_state:
            if pending_state == target:
                pending_count += 1
            else:
                pending_state = target
                pending_count = 1
            
            confirm_days = buy_days if target == 1 else sell_days
            if pending_count >= confirm_days:
                current_state = target
                pending_state = None
                pending_count = 0
        else:
            pending_state = None
            pending_count = 0
        
        result.loc[dt] = current_state
    
    return result

def generate_tq200_signal(prices: pd.DataFrame) -> pd.Series:
    tqqq = prices["TQQQ"]
    ma200 = tqqq.rolling(200).mean()
    return (tqqq > ma200).astype(int)

def generate_signal(prices: pd.DataFrame, config: dict) -> pd.Series:
    signal_type = config.get("signal_type", "plain")
    
    if signal_type == "tq200":
        signal = generate_tq200_signal(prices)
    elif signal_type == "ensemble":
        signal = generate_ensemble_signal(prices, config["ensemble_windows"])
    elif signal_type == "hysteresis":
        signal = generate_hysteresis_signal(prices, config.get("hysteresis_band", 0.0))
    else:
        signal = generate_plain_signal(prices)
    
    buy_days = config.get("buy_confirm_days", 1)
    sell_days = config.get("sell_confirm_days", 1)
    signal = apply_confirmation(signal, buy_days, sell_days)
    
    return signal

def run_backtest(prices: pd.DataFrame, config: dict, start_date: str, end_date: str, capital: float) -> dict:
    off_asset = config["off_asset"]
    if off_asset == "SGOV":
        if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
            off_asset = "SHV" if "SHV" in prices.columns else "CASH"
    
    off_tqqq_weight = config.get("off_tqqq_weight", OFF_TQQQ_WEIGHT)
    
    signal_raw = generate_signal(prices, config)
    signal_lagged = signal_raw.shift(1).fillna(0).astype(int)
    
    if config.get("use_ma200_guard", False):
        qqq = prices["QQQ"]
        ma200 = qqq.rolling(200).mean()
        guard = (qqq < ma200).shift(1).fillna(False).astype(int)
        off_weight_series = pd.Series(off_tqqq_weight, index=prices.index)
        off_weight_series[(signal_lagged == 0) & (guard == 1)] = 0.0
        target_tqqq = signal_lagged * 1.0 + (1 - signal_lagged) * off_weight_series
    else:
        target_tqqq = signal_lagged * 1.0 + (1 - signal_lagged) * off_tqqq_weight
    
    target_off = 1.0 - target_tqqq
    
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    off_ret = prices[off_asset].pct_change().fillna(0.0) if off_asset in prices.columns else pd.Series(0.0, index=prices.index)
    
    weight_change = target_tqqq.diff().abs().fillna(0.0)
    cost_drag = weight_change * (COST_BPS / 10000.0)
    
    port_ret_gross = target_tqqq * tqqq_ret + target_off * off_ret - cost_drag
    
    analysis_mask = (prices.index >= start_date) & (prices.index <= end_date)
    
    equity_list = []
    portfolio_value = capital
    tqqq_cost_basis = 0.0
    tqqq_shares = 0.0
    yearly_gains = {}
    total_tax_paid = 0.0
    prev_tqqq_weight = 0.0
    trades_count = 0
    
    analysis_prices = prices[analysis_mask]
    
    for i, dt in enumerate(analysis_prices.index):
        px_tqqq = float(prices.loc[dt, "TQQQ"])
        r = float(port_ret_gross.loc[dt])
        portfolio_value *= (1 + r)
        
        curr_weight = float(target_tqqq.loc[dt])
        weight_chg = curr_weight - prev_tqqq_weight
        
        year = dt.year
        if year not in yearly_gains:
            yearly_gains[year] = 0.0
        
        if abs(weight_chg) > 1e-6:
            trades_count += 1
            
            if weight_chg < 0 and tqqq_shares > 0 and tqqq_cost_basis > 0:
                avg_cost = tqqq_cost_basis / tqqq_shares
                sold_fraction = abs(weight_chg)
                sold_value = sold_fraction * portfolio_value
                sold_shares = sold_value / px_tqqq if px_tqqq > 0 else 0
                gain = sold_shares * (px_tqqq - avg_cost)
                yearly_gains[year] += gain
                
                sell_ratio = min(1.0, sold_shares / tqqq_shares) if tqqq_shares > 0 else 0
                tqqq_cost_basis *= (1 - sell_ratio)
                tqqq_shares -= sold_shares
            elif weight_chg > 0:
                buy_value = weight_chg * portfolio_value
                buy_shares = buy_value / px_tqqq if px_tqqq > 0 else 0
                tqqq_cost_basis += buy_value
                tqqq_shares += buy_shares
        
        prev_tqqq_weight = curr_weight
        
        is_year_end = (i == len(analysis_prices.index) - 1) or \
                      (analysis_prices.index[i+1].year != year if i < len(analysis_prices.index) - 1 else True)
        
        if is_year_end and year in yearly_gains:
            taxable = max(0, yearly_gains[year])
            tax = taxable * TAX_RATE
            portfolio_value -= tax
            total_tax_paid += tax
        
        equity_list.append({"date": dt.strftime("%Y-%m-%d"), "value": portfolio_value / capital})
    
    if len(equity_list) == 0:
        return {"status": "error", "message": "No data in date range"}
    
    equity_values = [e["value"] for e in equity_list]
    n_years = len(equity_values) / TRADING_DAYS
    final = equity_values[-1]
    
    cagr = (final ** (1.0 / n_years) - 1.0) if n_years > 0 and final > 0 else 0.0
    
    equity_series = pd.Series(equity_values)
    returns = equity_series.pct_change().fillna(0.0)
    peak = equity_series.cummax()
    drawdown = equity_series / peak - 1.0
    mdd = float(drawdown.min())
    
    daily_std = returns.std(ddof=0)
    sharpe = (returns.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0
    
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0.0
    sortino = (cagr / downside_std) if downside_std > 0 else 0.0
    
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0.0
    
    sampled_equity = equity_list[::max(1, len(equity_list) // 500)]
    if equity_list[-1] not in sampled_equity:
        sampled_equity.append(equity_list[-1])
    
    return {
        "status": "success",
        "experiment": config["name"],
        "params": {
            "strategy": config["name"],
            "startDate": start_date,
            "endDate": end_date,
            "capital": capital,
        },
        "metrics": {
            "CAGR": round(cagr * 100, 2),
            "MDD": round(mdd * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Sortino": round(sortino, 2),
            "Calmar": round(calmar, 2),
            "Final": round(final, 4),
            "FinalValue": round(portfolio_value, 0),
            "TotalTax": round(total_tax_paid, 0),
            "TradesCount": trades_count,
            "TradingDays": len(equity_list),
        },
        "equity": sampled_equity,
    }

def main():
    parser = argparse.ArgumentParser(description="Run backtest with specified parameters")
    parser.add_argument("--strategy", type=str, default="E03", help="Strategy code (E00-E10)")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000000, help="Initial capital in KRW")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.strategy not in STRATEGY_CONFIG:
        result = {"status": "error", "message": f"Unknown strategy: {args.strategy}. Available: {list(STRATEGY_CONFIG.keys())}"}
        print(json.dumps(result))
        sys.exit(1)
    
    config = STRATEGY_CONFIG[args.strategy]
    
    try:
        prices, start_date, end_date = load_data(args.start, args.end)
        result = run_backtest(prices, config, args.start, args.end, args.capital)
        result["elapsed_seconds"] = round(time.time() - start_time, 2)
        print(json.dumps(result))
    except Exception as e:
        result = {"status": "error", "message": str(e)}
        print(json.dumps(result))
        sys.exit(1)

if __name__ == "__main__":
    main()
