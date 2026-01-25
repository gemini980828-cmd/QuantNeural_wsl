# -*- coding: utf-8 -*-
"""
TQQQ Hybrid 2.0 Backtest
========================

전략 개요:
- 코어 (70%): QQQ MA3/MA161 추세추종 → TQQQ, -5% 하드스탑
- 위성 (30%): VIX 기반 F&G 지수로 3단계 분할매수 (TQQQ:QLD:QQQM = 3:4:3)

핵심 규칙:
- F&G 프록시: VIX 롤링 백분위 (3년 창)
- 리밸런싱: 분기 체크 + ±5% 밴드 조건부
- 체결: Close-to-Close, 1-day lag

Author: QuantNeural v2026.1
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
TRADING_DAYS = 252

# Cost & Tax
COST_BPS = 10  # 10 bps one-way
TAX_RATE = 0.22  # Korean overseas ETF 22%

# Core Strategy (Hybrid 2.0 original: MA161 single)
CORE_WEIGHT = 0.70
SHORT_MA = 3
LONG_MA = 161  # Original Hybrid 2.0 uses single MA161
LONG_MA_WINDOWS = [160, 165, 170]  # For E03 benchmark only
CORE_STOP_LOSS = -0.05  # -5% hard stop

# Satellite Strategy
SATELLITE_WEIGHT = 0.30
FG_LOOKBACK = 756  # 3 years rolling window
FG_STAGE1 = 20  # F&G <= 20 → 1st buy
FG_STAGE2 = 15  # F&G <= 15 → 2nd buy
FG_STAGE3 = 10  # F&G <= 10 → 3rd buy
FG_RESET = 25   # F&G >= 25 → reset stages
FG_TAKE_PROFIT = 60  # F&G >= 60 for take profit
SAT_TAKE_PROFIT_RET = 0.30  # +30% return
SAT_FREEZE_LOSS = -0.20  # -20% → freeze additional buys

# Satellite allocation: TQQQ:QLD:QQQM = 3:4:3
SAT_TQQQ_RATIO = 0.30
SAT_QLD_RATIO = 0.40
SAT_QQQM_RATIO = 0.30

# Rebalancing
REBAL_BAND = 0.05  # ±5% drift band

# Output
OUTPUT_DIR = "/home/juwon/QuantNeural_wsl/200tq/artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> Dict[str, pd.DataFrame]:
    """Load price data from yfinance"""
    import yfinance as yf
    
    print("📥 Downloading data from yfinance...")
    tickers = ["QQQ", "TQQQ", "QLD", "QQQM", "BIL", "^VIX"]
    
    raw = yf.download(
        tickers=tickers,
        start="2007-01-01",  # Extra for VIX lookback
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    
    prices = {}
    for t in tickers:
        col_name = t.replace("^", "")  # ^VIX → VIX
        if isinstance(raw.columns, pd.MultiIndex):
            if (t, "Close") in raw.columns:
                prices[col_name] = raw[(t, "Close")].dropna()
        else:
            if "Close" in raw.columns:
                prices[col_name] = raw["Close"].dropna()
    
    # Fallback for missing data
    if "QQQM" not in prices or prices["QQQM"].isna().all():
        print("   ⚠️ QQQM not available, using QQQ as proxy")
        prices["QQQM"] = prices["QQQ"].copy()
    
    if "BIL" not in prices or prices["BIL"].isna().all():
        print("   ⚠️ BIL not available, using constant 100")
        prices["BIL"] = pd.Series(100.0, index=prices["QQQ"].index)
    
    # Forward fill and filter to analysis period
    for k in prices:
        prices[k] = prices[k].ffill()
    
    print(f"   Loaded: {list(prices.keys())}")
    return prices


def build_synthetic_2x(qqq_close: pd.Series) -> pd.Series:
    """Build synthetic 2x leveraged ETF (QLD proxy) from QQQ"""
    r_qqq = qqq_close.pct_change().fillna(0)
    r_2x = 2.0 * r_qqq
    price_2x = 100 * (1 + r_2x).cumprod()
    return price_2x


def build_synthetic_3x(qqq_close: pd.Series) -> pd.Series:
    """Build synthetic 3x leveraged ETF (TQQQ proxy) from QQQ"""
    r_qqq = qqq_close.pct_change().fillna(0)
    r_3x = 3.0 * r_qqq
    price_3x = 100 * (1 + r_3x).cumprod()
    return price_3x


def prepare_prices(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    """Prepare aligned price DataFrame with synthetic backfill for leveraged ETFs"""
    # Use QQQ + VIX as base (both available from 2007+)
    base_idx = prices["QQQ"].index
    if "VIX" in prices:
        base_idx = base_idx.intersection(prices["VIX"].index)
    
    # Filter to analysis period
    base_idx = base_idx[(base_idx >= START_DATE) & (base_idx <= END_DATE)]
    
    df = pd.DataFrame(index=base_idx)
    df["QQQ"] = prices["QQQ"].reindex(base_idx).ffill()
    df["VIX"] = prices["VIX"].reindex(base_idx).ffill()
    
    # BIL: use actual if available, else assume ~0% return
    if "BIL" in prices:
        df["BIL"] = prices["BIL"].reindex(base_idx).ffill()
        # Fill early period with constant
        if df["BIL"].isna().any():
            df["BIL"] = df["BIL"].fillna(100.0)
    else:
        df["BIL"] = 100.0
    
    # Build synthetic leveraged series from QQQ
    synth_3x = build_synthetic_3x(df["QQQ"])
    synth_2x = build_synthetic_2x(df["QQQ"])
    synth_1x = df["QQQ"] / df["QQQ"].iloc[0] * 100  # Normalized
    
    # TQQQ: splice synthetic with actual
    if "TQQQ" in prices:
        actual_tqqq = prices["TQQQ"].reindex(base_idx)
        tqqq_start = actual_tqqq.first_valid_index()
        if tqqq_start and tqqq_start > base_idx[0]:
            # Scale synthetic to match actual at splice point
            scale = actual_tqqq.loc[tqqq_start] / synth_3x.loc[tqqq_start]
            df["TQQQ"] = synth_3x * scale
            df.loc[tqqq_start:, "TQQQ"] = actual_tqqq.loc[tqqq_start:].ffill()
            print(f"   TQQQ: synthetic until {tqqq_start.date()}, actual after")
        else:
            df["TQQQ"] = actual_tqqq.ffill()
    else:
        df["TQQQ"] = synth_3x
        print("   TQQQ: fully synthetic (3x QQQ)")
    
    # QLD: splice synthetic with actual
    if "QLD" in prices:
        actual_qld = prices["QLD"].reindex(base_idx)
        qld_start = actual_qld.first_valid_index()
        if qld_start and qld_start > base_idx[0]:
            scale = actual_qld.loc[qld_start] / synth_2x.loc[qld_start]
            df["QLD"] = synth_2x * scale
            df.loc[qld_start:, "QLD"] = actual_qld.loc[qld_start:].ffill()
            print(f"   QLD: synthetic until {qld_start.date()}, actual after")
        else:
            df["QLD"] = actual_qld.ffill()
    else:
        df["QLD"] = synth_2x
        print("   QLD: fully synthetic (2x QQQ)")
    
    # QQQM: use QQQ as proxy (QQQM launched 2020, tracks same index)
    if "QQQM" in prices:
        actual_qqqm = prices["QQQM"].reindex(base_idx)
        qqqm_start = actual_qqqm.first_valid_index()
        if qqqm_start and qqqm_start > base_idx[0]:
            scale = actual_qqqm.loc[qqqm_start] / synth_1x.loc[qqqm_start]
            df["QQQM"] = synth_1x * scale
            df.loc[qqqm_start:, "QQQM"] = actual_qqqm.loc[qqqm_start:].ffill()
            print(f"   QQQM: QQQ proxy until {qqqm_start.date()}, actual after")
        else:
            df["QQQM"] = actual_qqqm.ffill()
    else:
        df["QQQM"] = df["QQQ"].copy()
        print("   QQQM: using QQQ as proxy")
    
    print(f"   Period: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"   Trading days: {len(df)}")
    
    return df


# ============================================================
# FEAR & GREED: REAL DATA + SYNTHESIS FALLBACK
# ============================================================
FG_DATA_PATH = "/home/juwon/QuantNeural_wsl/200tq/data/fear_greed_historical.csv"


def load_real_fear_greed() -> pd.Series:
    """Load real CNN Fear & Greed Index data from CSV"""
    if not os.path.exists(FG_DATA_PATH):
        print(f"   ⚠️ Real F&G data not found at {FG_DATA_PATH}")
        return pd.Series(dtype=float)
    
    fg_df = pd.read_csv(FG_DATA_PATH)
    # Parse date column (format: M/D/YYYY)
    fg_df["Date"] = pd.to_datetime(fg_df["Date"], format="%m/%d/%Y")
    fg_df = fg_df.set_index("Date").sort_index()
    fg_series = fg_df["Fear Greed"].astype(float)
    
    print(f"   Real F&G data: {fg_series.index[0].date()} ~ {fg_series.index[-1].date()} ({len(fg_series)} days)")
    return fg_series


def synth_fear_greed(vix: pd.Series, lookback: int = FG_LOOKBACK) -> pd.Series:
    """
    VIX rolling percentile → F&G conversion
    - Higher VIX (fear) → Lower F&G
    - Lower VIX (greed) → Higher F&G
    """
    def pct_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1] * 100
    
    vix_pct = vix.rolling(lookback, min_periods=60).apply(pct_rank, raw=False)
    fg = 100 - vix_pct
    return fg.fillna(50)  # Neutral for warmup period


def get_fear_greed(df: pd.DataFrame) -> pd.Series:
    """
    Get F&G index: use real CNN data where available, VIX synthesis for rest
    """
    # Load real F&G data
    real_fg = load_real_fear_greed()
    
    # Generate synthetic F&G from VIX
    synth_fg = synth_fear_greed(df["VIX"])
    
    # Create combined series
    fg = pd.Series(index=df.index, dtype=float)
    
    if len(real_fg) > 0:
        # Use real data where available
        common_dates = df.index.intersection(real_fg.index)
        fg.loc[common_dates] = real_fg.loc[common_dates]
        print(f"   Using real F&G for {len(common_dates)} days")
        
        # Fill remaining with synthetic
        missing = fg.isna()
        fg.loc[missing] = synth_fg.loc[missing]
        synth_count = missing.sum()
        if synth_count > 0:
            print(f"   Using synthetic F&G (VIX-based) for {synth_count} days")
    else:
        # Fallback to all synthetic
        fg = synth_fg
        print(f"   Using fully synthetic F&G (VIX-based)")
    
    return fg


# ============================================================
# CORE STRATEGY (70%)
# ============================================================
@dataclass
class CoreState:
    position: float = 0.0  # TQQQ position value
    cash: float = 0.0  # BIL position value
    cost_basis: float = 0.0  # Average cost for stop-loss
    shares: float = 0.0


def run_core_backtest(
    df: pd.DataFrame,
    initial_capital: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Core strategy: Hybrid 2.0 (MA3/MA161) with TQQQ
    - ON: MA3 > MA161 → 100% TQQQ
    - OFF: MA3 < MA161 → 100% BIL (전량 청산)
    - Emergency exit: -5% stop-loss
    """
    qqq = df["QQQ"]
    tqqq = df["TQQQ"]
    bil = df["BIL"]
    
    # Single MA161 Signal (Hybrid 2.0 original)
    ma_short = qqq.rolling(SHORT_MA).mean()
    ma_long = qqq.rolling(LONG_MA).mean()
    signal_raw = (ma_short > ma_long).astype(int)
    signal = signal_raw.shift(1).fillna(0).astype(int)  # 1-day lag
    
    # Backtest
    equity = pd.Series(index=df.index, dtype=float)
    trades = []
    
    state = CoreState(cash=initial_capital)
    bil_shares = initial_capital / float(bil.iloc[0])
    
    for i, dt in enumerate(df.index):
        px_tqqq = float(tqqq.loc[dt])
        px_bil = float(bil.loc[dt])
        sig = int(signal.loc[dt])
        
        # Current portfolio value
        port_value = state.shares * px_tqqq + bil_shares * px_bil
        
        # Check stop-loss
        if state.shares > 0 and state.cost_basis > 0:
            avg_cost = state.cost_basis / state.shares
            current_ret = (px_tqqq / avg_cost) - 1.0
            if current_ret <= CORE_STOP_LOSS:
                # Stop-loss triggered
                proceeds = state.shares * px_tqqq
                cost = proceeds * (COST_BPS / 10000)
                bil_shares += (proceeds - cost) / px_bil
                trades.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Action": "STOP_LOSS",
                    "Asset": "TQQQ",
                    "Value": proceeds,
                    "Cost": cost,
                })
                state.shares = 0
                state.cost_basis = 0
                sig = 0  # Force OFF
        
        # Signal transition
        prev_in_tqqq = state.shares > 0
        want_tqqq = sig == 1
        
        if want_tqqq and not prev_in_tqqq:
            # Buy TQQQ
            cash_available = bil_shares * px_bil
            buy_value = cash_available
            cost = buy_value * (COST_BPS / 10000)
            shares_bought = (buy_value - cost) / px_tqqq
            state.shares = shares_bought
            state.cost_basis = buy_value - cost
            bil_shares = 0
            trades.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Action": "BUY",
                "Asset": "TQQQ",
                "Value": buy_value,
                "Cost": cost,
            })
        
        elif not want_tqqq and prev_in_tqqq:
            # Sell TQQQ
            sell_value = state.shares * px_tqqq
            cost = sell_value * (COST_BPS / 10000)
            bil_shares = (sell_value - cost) / px_bil
            trades.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Action": "SELL",
                "Asset": "TQQQ",
                "Value": sell_value,
                "Cost": cost,
            })
            state.shares = 0
            state.cost_basis = 0
        
        # Record equity
        equity.loc[dt] = state.shares * px_tqqq + bil_shares * px_bil
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return equity, trades_df


# ============================================================
# SATELLITE STRATEGY (30%)
# ============================================================
@dataclass
class SatelliteState:
    stage: int = 0  # 0=waiting, 1/2/3=invested stages
    total_invested: float = 0.0
    tqqq_shares: float = 0.0
    qld_shares: float = 0.0
    qqqm_shares: float = 0.0
    bil_shares: float = 0.0
    frozen: bool = False  # -20% reached
    cycle_complete: bool = False


def run_satellite_backtest(
    df: pd.DataFrame,
    fg: pd.Series,
    initial_capital: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Satellite strategy: F&G-based 3-stage DCA
    - Stage 1: F&G <= 20
    - Stage 2: F&G <= 15
    - Stage 3: F&G <= 10
    - Take profit: +30% AND F&G >= 60
    - Freeze: -20%
    """
    tqqq = df["TQQQ"]
    qld = df["QLD"]
    qqqm = df["QQQM"]
    bil = df["BIL"]
    
    equity = pd.Series(index=df.index, dtype=float)
    trades = []
    
    # Initial: all in BIL
    state = SatelliteState()
    state.bil_shares = initial_capital / float(bil.iloc[0])
    stage_capital = initial_capital / 3.0  # Each stage uses 1/3
    
    for i, dt in enumerate(df.index):
        px_tqqq = float(tqqq.loc[dt])
        px_qld = float(qld.loc[dt])
        px_qqqm = float(qqqm.loc[dt])
        px_bil = float(bil.loc[dt])
        fg_val = float(fg.loc[dt])
        
        # Current portfolio value
        port_value = (
            state.tqqq_shares * px_tqqq +
            state.qld_shares * px_qld +
            state.qqqm_shares * px_qqqm +
            state.bil_shares * px_bil
        )
        
        # Calculate return if invested
        if state.total_invested > 0:
            invested_value = (
                state.tqqq_shares * px_tqqq +
                state.qld_shares * px_qld +
                state.qqqm_shares * px_qqqm
            )
            current_ret = (invested_value / state.total_invested) - 1.0
        else:
            current_ret = 0.0
        
        # Check freeze condition (-20%)
        if not state.frozen and state.total_invested > 0 and current_ret <= SAT_FREEZE_LOSS:
            state.frozen = True
            trades.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Action": "FREEZE",
                "Reason": f"Return {current_ret*100:.1f}% <= -20%",
            })
        
        # Check take profit (+30% AND F&G >= 60)
        if state.total_invested > 0 and current_ret >= SAT_TAKE_PROFIT_RET and fg_val >= FG_TAKE_PROFIT:
            # Sell all positions
            sell_value = (
                state.tqqq_shares * px_tqqq +
                state.qld_shares * px_qld +
                state.qqqm_shares * px_qqqm
            )
            cost = sell_value * (COST_BPS / 10000)
            state.bil_shares += (sell_value - cost) / px_bil
            trades.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Action": "TAKE_PROFIT",
                "Value": sell_value,
                "Return": current_ret,
                "FG": fg_val,
                "Cost": cost,
            })
            state.tqqq_shares = 0
            state.qld_shares = 0
            state.qqqm_shares = 0
            state.total_invested = 0
            state.stage = 0
            state.frozen = False
            state.cycle_complete = True
        
        # Check reset condition (F&G >= 25 after cycle complete)
        if state.cycle_complete and fg_val >= FG_RESET:
            state.cycle_complete = False
        
        # Stage entry (only if not frozen and not in cycle_complete waiting)
        if not state.frozen and not state.cycle_complete:
            cash_available = state.bil_shares * px_bil
            
            # Check stage transitions
            if state.stage == 0 and fg_val <= FG_STAGE1 and cash_available >= stage_capital * 0.9:
                # Stage 1 entry
                buy_amount = min(stage_capital, cash_available)
                _execute_satellite_buy(state, buy_amount, px_tqqq, px_qld, px_qqqm, px_bil)
                state.stage = 1
                trades.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Action": "STAGE1_BUY",
                    "FG": fg_val,
                    "Amount": buy_amount,
                })
            
            elif state.stage == 1 and fg_val <= FG_STAGE2 and cash_available >= stage_capital * 0.9:
                # Stage 2 entry
                buy_amount = min(stage_capital, cash_available)
                _execute_satellite_buy(state, buy_amount, px_tqqq, px_qld, px_qqqm, px_bil)
                state.stage = 2
                trades.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Action": "STAGE2_BUY",
                    "FG": fg_val,
                    "Amount": buy_amount,
                })
            
            elif state.stage == 2 and fg_val <= FG_STAGE3 and cash_available >= stage_capital * 0.9:
                # Stage 3 entry
                buy_amount = min(stage_capital, cash_available)
                _execute_satellite_buy(state, buy_amount, px_tqqq, px_qld, px_qqqm, px_bil)
                state.stage = 3
                trades.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Action": "STAGE3_BUY",
                    "FG": fg_val,
                    "Amount": buy_amount,
                })
        
        # Record equity
        equity.loc[dt] = (
            state.tqqq_shares * px_tqqq +
            state.qld_shares * px_qld +
            state.qqqm_shares * px_qqqm +
            state.bil_shares * px_bil
        )
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return equity, trades_df


def _execute_satellite_buy(
    state: SatelliteState,
    amount: float,
    px_tqqq: float,
    px_qld: float,
    px_qqqm: float,
    px_bil: float,
):
    """Execute satellite buy with 3:4:3 allocation"""
    cost = amount * (COST_BPS / 10000)
    net_amount = amount - cost
    
    tqqq_amount = net_amount * SAT_TQQQ_RATIO
    qld_amount = net_amount * SAT_QLD_RATIO
    qqqm_amount = net_amount * SAT_QQQM_RATIO
    
    state.tqqq_shares += tqqq_amount / px_tqqq
    state.qld_shares += qld_amount / px_qld
    state.qqqm_shares += qqqm_amount / px_qqqm
    state.total_invested += net_amount
    state.bil_shares -= amount / px_bil


# ============================================================
# HYBRID PORTFOLIO
# ============================================================
def run_hybrid_backtest(df: pd.DataFrame, fg: pd.Series) -> Dict:
    """
    Combined Core (70%) + Satellite (30%) with quarterly rebalancing
    """
    initial_capital = 1.0
    core_capital = initial_capital * CORE_WEIGHT
    sat_capital = initial_capital * SATELLITE_WEIGHT
    
    # Run individual strategies
    core_equity, core_trades = run_core_backtest(df, core_capital)
    sat_equity, sat_trades = run_satellite_backtest(df, fg, sat_capital)
    
    # Combined equity (simple sum for now, rebalancing below)
    combined = core_equity + sat_equity
    
    # Apply quarterly rebalancing with band
    rebalanced = apply_quarterly_rebalancing(df, core_equity, sat_equity)
    
    return {
        "core_equity": core_equity,
        "satellite_equity": sat_equity,
        "combined_equity": combined,
        "rebalanced_equity": rebalanced,
        "core_trades": core_trades,
        "satellite_trades": sat_trades,
    }


def apply_quarterly_rebalancing(
    df: pd.DataFrame,
    core_eq: pd.Series,
    sat_eq: pd.Series,
) -> pd.Series:
    """Apply quarterly rebalancing with ±5% band"""
    result = pd.Series(index=df.index, dtype=float)
    
    # Identify quarter ends
    quarter_ends = df.index.to_series().groupby(pd.Grouper(freq='QE')).last()
    
    core_mult = 1.0
    sat_mult = 1.0
    prev_core = float(core_eq.iloc[0])
    prev_sat = float(sat_eq.iloc[0])
    
    for i, dt in enumerate(df.index):
        curr_core = float(core_eq.loc[dt]) * core_mult
        curr_sat = float(sat_eq.loc[dt]) * sat_mult
        total = curr_core + curr_sat
        
        # Check if quarter end
        is_quarter_end = dt in quarter_ends.values
        
        if is_quarter_end and total > 0:
            core_weight = curr_core / total
            drift = abs(core_weight - CORE_WEIGHT)
            
            if drift >= REBAL_BAND:
                # Rebalance to 70:30
                target_core = total * CORE_WEIGHT
                target_sat = total * SATELLITE_WEIGHT
                
                # Adjust multipliers
                if core_eq.loc[dt] > 0:
                    core_mult = target_core / float(core_eq.loc[dt])
                if sat_eq.loc[dt] > 0:
                    sat_mult = target_sat / float(sat_eq.loc[dt])
                
                curr_core = target_core
                curr_sat = target_sat
        
        result.loc[dt] = curr_core + curr_sat
        prev_core = curr_core
        prev_sat = curr_sat
    
    return result


# ============================================================
# METRICS
# ============================================================
def calculate_metrics(equity: pd.Series) -> Dict:
    """Calculate performance metrics"""
    n_years = len(equity) / TRADING_DAYS
    final = float(equity.iloc[-1])
    
    cagr = (final ** (1.0 / n_years) - 1.0) if n_years > 0 and final > 0 else 0.0
    
    returns = equity.pct_change().fillna(0.0)
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    mdd = float(drawdown.min())
    
    daily_std = returns.std(ddof=0)
    sharpe = (returns.mean() / daily_std * np.sqrt(TRADING_DAYS)) if daily_std > 0 else 0.0
    
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0.0
    
    return {
        "Final": final,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Calmar": calmar,
        "Years": n_years,
    }


# ============================================================
# BENCHMARK COMPARISON
# ============================================================
def run_benchmarks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Run benchmark strategies for comparison"""
    benchmarks = {}
    
    # TQQQ Buy & Hold
    tqqq_ret = df["TQQQ"].pct_change().fillna(0)
    benchmarks["TQQQ_BuyHold"] = (1 + tqqq_ret).cumprod()
    
    # BIL (Cash)
    bil_ret = df["BIL"].pct_change().fillna(0)
    benchmarks["BIL_Cash"] = (1 + bil_ret).cumprod()
    
    # E03: Ensemble (MA160/165/170) with 100%/10% allocation
    qqq = df["QQQ"]
    ma3 = qqq.rolling(3).mean()
    
    # Ensemble voting
    votes = pd.DataFrame(index=df.index)
    for window in [160, 165, 170]:
        ma_long = qqq.rolling(window).mean()
        votes[f"MA{window}"] = (ma3 > ma_long).astype(int)
    
    vote_sum = votes.sum(axis=1)
    signal = (vote_sum >= 2).astype(int).shift(1).fillna(0)  # 2/3 majority + 1-day lag
    
    tqqq_w = signal * 1.0 + (1 - signal) * 0.10  # ON: 100%, OFF: 10%
    bil_w = 1.0 - tqqq_w
    port_ret = tqqq_w * tqqq_ret + bil_w * bil_ret
    benchmarks["E03_Ensemble"] = (1 + port_ret).cumprod()
    
    # E03 + Leverage Down: Use lower leverage in OFF state
    qld_ret = df["QLD"].pct_change().fillna(0)
    qqqm_ret = df["QQQM"].pct_change().fillna(0)
    
    # Version 1: OFF → 10% QLD (2x instead of 3x)
    tqqq_w_lev1 = signal * 1.0  # ON: 100% TQQQ
    qld_w_lev1 = (1 - signal) * 0.10  # OFF: 10% QLD
    bil_w_lev1 = (1 - signal) * 0.90  # OFF: 90% BIL
    port_ret_lev1 = tqqq_w_lev1 * tqqq_ret + qld_w_lev1 * qld_ret + bil_w_lev1 * bil_ret
    benchmarks["E03_OFF_QLD"] = (1 + port_ret_lev1).cumprod()
    
    # Version 2: OFF → 5% TQQQ + 5% QQQM (blended 2x)
    tqqq_w_lev2 = signal * 1.0 + (1 - signal) * 0.05  # ON: 100%, OFF: 5%
    qqqm_w_lev2 = (1 - signal) * 0.05  # OFF: 5% QQQM
    bil_w_lev2 = (1 - signal) * 0.90  # OFF: 90% BIL
    port_ret_lev2 = tqqq_w_lev2 * tqqq_ret + qqqm_w_lev2 * qqqm_ret + bil_w_lev2 * bil_ret
    benchmarks["E03_OFF_Mix"] = (1 + port_ret_lev2).cumprod()
    
    # Version 3: OFF → 0% (completely out, no 10% residual)
    tqqq_w_lev3 = signal * 1.0  # ON: 100% TQQQ, OFF: 0%
    bil_w_lev3 = 1.0 - tqqq_w_lev3  # ON: 0% BIL, OFF: 100%
    port_ret_lev3 = tqqq_w_lev3 * tqqq_ret + bil_w_lev3 * bil_ret
    benchmarks["E03_OFF_0pct"] = (1 + port_ret_lev3).cumprod()
    
    return benchmarks


# ============================================================
# REPORTING
# ============================================================
def generate_report(results: Dict, benchmarks: Dict[str, pd.Series], df: pd.DataFrame, fg: pd.Series):
    """Generate report and plots"""
    
    # Calculate metrics for all strategies
    all_metrics = {}
    
    for name, eq in [
        ("Hybrid_2.0_Core", results["core_equity"]),
        ("Hybrid_2.0_Satellite", results["satellite_equity"]),
        ("Hybrid_2.0_Combined", results["combined_equity"]),
        ("Hybrid_2.0_Rebalanced", results["rebalanced_equity"]),
    ]:
        all_metrics[name] = calculate_metrics(eq)
    
    for name, eq in benchmarks.items():
        all_metrics[name] = calculate_metrics(eq)
    
    # Print summary
    print("\n" + "="*80)
    print("           TQQQ Hybrid 2.0 Backtest Results")
    print("="*80)
    print(f"Period: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"Trading Days: {len(df)}")
    print("="*80)
    
    print(f"\n{'Strategy':<25} {'CAGR':>10} {'MDD':>10} {'Sharpe':>8} {'Calmar':>8} {'Final':>10}")
    print("-"*80)
    
    for name, m in all_metrics.items():
        print(f"{name:<25} {m['CAGR']*100:>9.2f}% {m['MDD']*100:>9.2f}% "
              f"{m['Sharpe']:>8.2f} {m['Calmar']:>8.2f} {m['Final']:>9.2f}x")
    
    print("="*80)
    
    # Plot equity curves
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Equity curves
    ax1 = axes[0]
    ax1.plot(results["rebalanced_equity"], label="Hybrid 2.0 (Rebalanced)", linewidth=2, color='blue')
    ax1.plot(results["core_equity"], label="Core (70%)", linewidth=1, alpha=0.7, color='green')
    ax1.plot(results["satellite_equity"], label="Satellite (30%)", linewidth=1, alpha=0.7, color='orange')
    for name, eq in benchmarks.items():
        ax1.plot(eq, label=name, linewidth=1, linestyle='--', alpha=0.6)
    ax1.set_yscale('log')
    ax1.set_title(f"Equity Curves (Log Scale) | {df.index[0].date()} ~ {df.index[-1].date()}")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Drawdown
    ax2 = axes[1]
    rebal_eq = results["rebalanced_equity"]
    peak = rebal_eq.cummax()
    dd = (rebal_eq / peak - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color='red')
    ax2.set_title(f"Hybrid 2.0 Drawdown (MDD: {all_metrics['Hybrid_2.0_Rebalanced']['MDD']*100:.1f}%)")
    ax2.set_ylabel("Drawdown %")
    ax2.set_ylim(-60, 5)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Fear & Greed Index
    ax3 = axes[2]
    ax3.plot(fg, label="Synthetic F&G (VIX-based)", color='purple', alpha=0.7)
    ax3.axhline(20, color='red', linestyle='--', alpha=0.5, label="Stage 1 (F&G=20)")
    ax3.axhline(15, color='orange', linestyle='--', alpha=0.5, label="Stage 2 (F&G=15)")
    ax3.axhline(10, color='darkred', linestyle='--', alpha=0.5, label="Stage 3 (F&G=10)")
    ax3.axhline(60, color='green', linestyle='--', alpha=0.5, label="Take Profit (F&G=60)")
    ax3.set_title("Synthetic Fear & Greed Index (VIX Rolling Percentile)")
    ax3.set_ylabel("F&G")
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "compare_hybrid_2_0.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Plot saved to: {plot_path}")
    plt.close()
    
    # Save metrics JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_hybrid_2_0.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"📋 Metrics saved to: {metrics_path}")
    
    # Save equity CSV
    equity_df = pd.DataFrame({
        "Hybrid_2.0": results["rebalanced_equity"],
        "Core": results["core_equity"],
        "Satellite": results["satellite_equity"],
    })
    for name, eq in benchmarks.items():
        equity_df[name] = eq
    equity_path = os.path.join(OUTPUT_DIR, "equity_hybrid_2_0.csv")
    equity_df.to_csv(equity_path)
    print(f"📈 Equity saved to: {equity_path}")
    
    return all_metrics


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*80)
    print("       TQQQ Hybrid 2.0 Backtest Engine")
    print("="*80)
    
    # Load data
    prices = load_data()
    df = prepare_prices(prices)
    
    # Generate Fear & Greed (real CNN data + VIX fallback)
    print("\n📊 Loading Fear & Greed index...")
    fg = get_fear_greed(df)
    print(f"   F&G range: {fg.min():.1f} ~ {fg.max():.1f}")
    print(f"   F&G mean: {fg.mean():.1f}")
    
    # Run hybrid backtest
    print("\n🚀 Running Hybrid 2.0 backtest...")
    results = run_hybrid_backtest(df, fg)
    
    # Run benchmarks
    print("\n📊 Running benchmarks...")
    benchmarks = run_benchmarks(df)
    
    # Generate report
    metrics = generate_report(results, benchmarks, df, fg)
    
    print("\n✅ Backtest complete!")
    
    return results, metrics


if __name__ == "__main__":
    main()
