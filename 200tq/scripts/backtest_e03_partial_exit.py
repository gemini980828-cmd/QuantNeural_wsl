# -*- coding: utf-8 -*-
"""
E03 다단계 익절 백테스트 실험
==============================

E03 전략에 200TQ v1.0의 다단계 익절 규칙을 추가하여 CAGR/MDD 비교

다단계 익절 규칙 (200TQ v1.0):
- +10% 수익 시: 30% 익절
- +25% 수익 시: 50% 익절
- +50% 수익 시: 70% 익절
- +100% 수익 시: 90% 익절

Author: QuantNeural Backtest Engine v2026.1
Date: 2026-01-25
"""

import os
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
ARTIFACTS_DIR = "/home/juwon/QuantNeural_wsl/200tq/artifacts/backtest_e03_partial_exit"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Date ranges (E03 표준)
PRIMARY_START = "2010-02-01"
PRIMARY_END = "2025-12-31"

# 다단계 익절 thresholds
PARTIAL_EXIT_TIERS = [
    (0.10, 0.30),  # +10% 수익 시 30% 익절
    (0.25, 0.50),  # +25% 수익 시 50% 익절
    (0.50, 0.70),  # +50% 수익 시 70% 익절
    (1.00, 0.90),  # +100% 수익 시 90% 익절
]

# Cost/Tax (E03 표준 조건 - 세금 없이 순수 비교)
COST_BPS = 10
TAX_RATE = 0.0  # Tax0 for fair comparison (SSOT 검증은 Net이지만, 익절 비교는 Gross로)

# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class BacktestResult:
    """Backtest result container"""
    name: str
    equity: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    signals: pd.Series
    trades: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================
# DATA LOADING
# ============================================================
def download_data(start: str, end: str) -> pd.DataFrame:
    """Download price data from yfinance"""
    import yfinance as yf
    
    tickers = ["QQQ", "TQQQ", "SGOV", "SHV"]
    
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance download failed")
    
    prices = pd.DataFrame()
    
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if (t, "Close") in raw.columns:
                prices[t] = raw[(t, "Close")]
        else:
            if "Close" in raw.columns:
                prices[t] = raw["Close"]
    
    prices["CASH"] = 100.0
    
    if "SGOV" not in prices.columns or prices["SGOV"].isna().all():
        print("WARNING: SGOV not available, using SHV as proxy")
        if "SHV" in prices.columns:
            prices["SGOV"] = prices["SHV"]
        else:
            prices["SGOV"] = 100.0
    
    prices["SGOV"] = prices["SGOV"].ffill()
    prices = prices.dropna(subset=["QQQ", "TQQQ"])
    
    return prices


# ============================================================
# SIGNAL GENERATION (E03 앙상블)
# ============================================================
def generate_ensemble_signal(prices: pd.DataFrame, short_ma: int = 3,
                             long_windows: List[int] = [160, 165, 170]) -> pd.Series:
    """
    E03 ensemble signal: majority vote among long windows
    ON if at least 2 of 3 are ON else OFF
    """
    qqq = prices["QQQ"]
    ma_short = qqq.rolling(short_ma).mean()
    
    votes = pd.DataFrame(index=prices.index)
    for lw in long_windows:
        ma_long = qqq.rolling(lw).mean()
        votes[f"w{lw}"] = (ma_short > ma_long).astype(int)
    
    # Majority vote (at least 2 of 3)
    signal = (votes.sum(axis=1) >= 2).astype(int)
    
    # Shift by 1 day (apply signal to next day's return)
    signal = signal.shift(1).fillna(0).astype(int)
    
    return signal


# ============================================================
# METRICS CALCULATION
# ============================================================
def calculate_metrics(equity: pd.Series, returns: pd.Series,
                      trades: pd.Series, trading_days: int = 252) -> Dict[str, float]:
    """Calculate all required metrics"""
    
    n_years = len(equity) / trading_days
    cagr = (equity.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    mdd = drawdown.min()
    
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=0)
    sharpe = (daily_mean / daily_std * np.sqrt(trading_days)) if daily_std > 0 else 0.0
    
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0.0
    
    trades_per_year = trades.sum() / n_years if n_years > 0 else 0
    
    return {
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Calmar": calmar,
        "TradesPerYear": trades_per_year,
        "FinalValue": equity.iloc[-1],
        "NumYears": n_years,
    }


# ============================================================
# BACKTEST ENGINES
# ============================================================
def run_e03_baseline(prices: pd.DataFrame, cost_bps: float = 10, 
                     tax_rate: float = 0.22) -> BacktestResult:
    """
    E03 Baseline 백테스트 (다단계 익절 없음)
    - ON: 100% TQQQ
    - OFF: 10% TQQQ + 90% SGOV
    """
    signal = generate_ensemble_signal(prices)
    
    # Position weights
    positions = pd.DataFrame(index=prices.index)
    positions["TQQQ"] = signal * 1.0 + (1 - signal) * 0.10
    positions["SGOV"] = 1.0 - positions["TQQQ"]
    
    # Returns
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0.0)
    sgov_ret = prices["SGOV"].pct_change().fillna(0.0)
    
    # Portfolio return
    port_ret = positions["TQQQ"] * tqqq_ret + positions["SGOV"] * sgov_ret
    
    # Transaction costs
    weight_change = positions["TQQQ"].diff().abs().fillna(0.0)
    trades = (weight_change > 0.001).astype(int)
    cost_drag = weight_change * (cost_bps / 10000.0)
    port_ret = port_ret - cost_drag
    
    # Tax (simplified TaxB - annual)
    port_ret = apply_simple_tax(port_ret, tax_rate)
    
    # Equity curve
    equity = (1.0 + port_ret).cumprod()
    
    metrics = calculate_metrics(equity, port_ret, trades)
    
    return BacktestResult(
        name="E03_Baseline",
        equity=equity,
        daily_returns=port_ret,
        positions=positions,
        signals=signal,
        trades=trades,
        metrics=metrics
    )


def run_e03_with_partial_exit(prices: pd.DataFrame, cost_bps: float = 10, 
                               tax_rate: float = 0.22) -> BacktestResult:
    """
    E03 + 다단계 익절 백테스트
    
    다단계 익절 규칙:
    - +10% 수익 시: 30% 포지션 익절
    - +25% 수익 시: 50% 포지션 익절
    - +50% 수익 시: 70% 포지션 익절
    - +100% 수익 시: 90% 포지션 익절
    
    익절된 자금은 SGOV로 이동
    """
    signal = generate_ensemble_signal(prices)
    
    # Initialize tracking variables
    equity_values = []
    daily_rets = []
    tqqq_weights = []
    sgov_weights = []
    trade_flags = []
    
    # State variables
    portfolio_value = 1.0
    tqqq_position = 0.0  # $ value in TQQQ
    sgov_position = 0.0  # $ value in SGOV
    entry_price = None   # TQQQ entry price for gain tracking
    current_tier = -1    # Which tier we've already triggered (-1 = none)
    
    # Track which tiers have been triggered during current ON cycle
    triggered_tiers = set()
    
    tqqq_prices = prices["TQQQ"].values
    sgov_prices = prices["SGOV"].values
    signals = signal.values
    
    for i, dt in enumerate(prices.index):
        sig = signals[i]
        tqqq_price = tqqq_prices[i]
        sgov_price = sgov_prices[i]
        
        # Calculate daily returns first (if not first day)
        if i > 0:
            prev_tqqq_price = tqqq_prices[i-1]
            prev_sgov_price = sgov_prices[i-1]
            
            tqqq_ret = (tqqq_price / prev_tqqq_price - 1.0) if prev_tqqq_price > 0 else 0.0
            sgov_ret = (sgov_price / prev_sgov_price - 1.0) if prev_sgov_price > 0 else 0.0
            
            # Update positions based on returns
            tqqq_position *= (1 + tqqq_ret)
            sgov_position *= (1 + sgov_ret)
        
        portfolio_value = tqqq_position + sgov_position
        if portfolio_value <= 0:
            portfolio_value = 1e-10
        
        is_trade = False
        
        # ON signal handling
        if sig == 1:
            # First day of ON: Entry
            if i == 0 or signals[i-1] == 0:
                # New ON cycle - reset tier tracking
                entry_price = tqqq_price
                triggered_tiers = set()
                current_tier = -1
                
                # Go to 100% TQQQ
                prev_tqqq_weight = tqqq_position / portfolio_value if portfolio_value > 0 else 0
                tqqq_position = portfolio_value
                sgov_position = 0.0
                is_trade = abs(1.0 - prev_tqqq_weight) > 0.001
            
            # Check for partial exit triggers
            elif entry_price is not None and entry_price > 0:
                gain = (tqqq_price / entry_price) - 1.0
                
                for tier_idx, (threshold, exit_pct) in enumerate(PARTIAL_EXIT_TIERS):
                    if tier_idx in triggered_tiers:
                        continue  # Already triggered this tier
                    
                    if gain >= threshold:
                        # Trigger partial exit
                        exit_amount = tqqq_position * exit_pct
                        tqqq_position -= exit_amount
                        sgov_position += exit_amount * (1 - cost_bps / 10000.0)  # Apply cost
                        triggered_tiers.add(tier_idx)
                        is_trade = True
                        current_tier = tier_idx
        
        # OFF signal handling
        else:
            # Transition to OFF10: 10% TQQQ, 90% SGOV
            target_tqqq = portfolio_value * 0.10
            target_sgov = portfolio_value * 0.90
            
            prev_tqqq_weight = tqqq_position / portfolio_value if portfolio_value > 0 else 0
            
            # Apply transaction cost on weight change
            weight_change = abs(0.10 - prev_tqqq_weight)
            cost = portfolio_value * weight_change * (cost_bps / 10000.0)
            
            tqqq_position = target_tqqq
            sgov_position = target_sgov - cost
            
            is_trade = weight_change > 0.001
            
            # Reset entry tracking
            entry_price = None
            triggered_tiers = set()
            current_tier = -1
        
        # Record state
        portfolio_value = tqqq_position + sgov_position
        tqqq_weight = tqqq_position / portfolio_value if portfolio_value > 0 else 0
        sgov_weight = sgov_position / portfolio_value if portfolio_value > 0 else 0
        
        equity_values.append(portfolio_value)
        tqqq_weights.append(tqqq_weight)
        sgov_weights.append(sgov_weight)
        trade_flags.append(1 if is_trade else 0)
        
        if i > 0:
            daily_ret = (portfolio_value / equity_values[-2]) - 1.0
        else:
            daily_ret = 0.0
        daily_rets.append(daily_ret)
    
    # Create result dataframes
    equity = pd.Series(equity_values, index=prices.index)
    daily_returns = pd.Series(daily_rets, index=prices.index)
    
    positions = pd.DataFrame({
        "TQQQ": tqqq_weights,
        "SGOV": sgov_weights
    }, index=prices.index)
    
    trades = pd.Series(trade_flags, index=prices.index)
    
    # Apply tax
    daily_returns = apply_simple_tax(daily_returns, tax_rate)
    equity = (1.0 + daily_returns).cumprod()
    
    metrics = calculate_metrics(equity, daily_returns, trades)
    
    return BacktestResult(
        name="E03_PartialExit",
        equity=equity,
        daily_returns=daily_returns,
        positions=positions,
        signals=signal,
        trades=trades,
        metrics=metrics
    )


def apply_simple_tax(returns: pd.Series, tax_rate: float) -> pd.Series:
    """Simplified TaxB: Apply annual tax on positive gains"""
    if tax_rate <= 0:
        return returns
    
    result = returns.copy()
    years = returns.index.year.unique()
    
    for year in years:
        year_mask = returns.index.year == year
        year_gain = returns[year_mask].sum()
        
        if year_gain > 0:
            tax_drag = year_gain * tax_rate / year_mask.sum()
            result[year_mask] -= tax_drag
    
    return result


# ============================================================
# VISUALIZATION
# ============================================================
def plot_comparison(baseline: BacktestResult, partial: BacktestResult, save_path: str):
    """Plot equity curves comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Equity curves (log scale)
    ax1 = axes[0]
    ax1.plot(baseline.equity, label=f"{baseline.name} (CAGR: {baseline.metrics['CAGR']*100:.1f}%)",
             color='#1f77b4', linewidth=2)
    ax1.plot(partial.equity, label=f"{partial.name} (CAGR: {partial.metrics['CAGR']*100:.1f}%)",
             color='#ff7f0e', linewidth=2)
    ax1.set_yscale("log")
    ax1.set_title("E03 Baseline vs E03 + Partial Exit (Log Scale)", fontsize=14)
    ax1.set_ylabel("Portfolio Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Drawdowns
    ax2 = axes[1]
    
    peak_base = baseline.equity.cummax()
    dd_base = (baseline.equity / peak_base - 1.0) * 100
    
    peak_partial = partial.equity.cummax()
    dd_partial = (partial.equity / peak_partial - 1.0) * 100
    
    ax2.fill_between(dd_base.index, dd_base.values, 0, alpha=0.3, color='#1f77b4',
                     label=f"Baseline (MDD: {baseline.metrics['MDD']*100:.1f}%)")
    ax2.fill_between(dd_partial.index, dd_partial.values, 0, alpha=0.3, color='#ff7f0e',
                     label=f"Partial Exit (MDD: {partial.metrics['MDD']*100:.1f}%)")
    ax2.set_title("Drawdown Comparison", fontsize=14)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-70, 5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {save_path}")


def generate_report(baseline: BacktestResult, partial: BacktestResult, save_path: str):
    """Generate markdown report"""
    
    report = f"""# E03 다단계 익절 백테스트 결과

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Period**: {PRIMARY_START} ~ {PRIMARY_END}  
**Transaction Cost**: {COST_BPS} bps  
**Tax Rate**: {TAX_RATE*100:.0f}% (TaxB)

---

## 다단계 익절 규칙 (200TQ v1.0)

| 수익률 | 익절 비중 |
|:-------|:----------|
| +10%   | 30% 익절  |
| +25%   | 50% 익절  |
| +50%   | 70% 익절  |
| +100%  | 90% 익절  |

---

## 결과 비교

| 지표 | E03 Baseline | E03 + Partial Exit | 차이 |
|:-----|-------------:|-------------------:|-----:|
| **CAGR** | {baseline.metrics['CAGR']*100:.2f}% | {partial.metrics['CAGR']*100:.2f}% | {(partial.metrics['CAGR'] - baseline.metrics['CAGR'])*100:+.2f}%p |
| **MDD** | {baseline.metrics['MDD']*100:.2f}% | {partial.metrics['MDD']*100:.2f}% | {(partial.metrics['MDD'] - baseline.metrics['MDD'])*100:+.2f}%p |
| **Sharpe** | {baseline.metrics['Sharpe']:.2f} | {partial.metrics['Sharpe']:.2f} | {partial.metrics['Sharpe'] - baseline.metrics['Sharpe']:+.2f} |
| **Calmar** | {baseline.metrics['Calmar']:.2f} | {partial.metrics['Calmar']:.2f} | {partial.metrics['Calmar'] - baseline.metrics['Calmar']:+.2f} |
| **Trades/Year** | {baseline.metrics['TradesPerYear']:.1f} | {partial.metrics['TradesPerYear']:.1f} | {partial.metrics['TradesPerYear'] - baseline.metrics['TradesPerYear']:+.1f} |
| **Final Value** | {baseline.metrics['FinalValue']:.2f}x | {partial.metrics['FinalValue']:.2f}x | {partial.metrics['FinalValue'] - baseline.metrics['FinalValue']:+.2f}x |

---

## 분석

"""
    
    cagr_diff = (partial.metrics['CAGR'] - baseline.metrics['CAGR']) * 100
    mdd_diff = (partial.metrics['MDD'] - baseline.metrics['MDD']) * 100
    
    if cagr_diff < 0:
        report += f"- **CAGR 하락**: 다단계 익절로 인해 CAGR이 **{abs(cagr_diff):.2f}%p 하락**했습니다.\n"
    else:
        report += f"- **CAGR 상승**: 다단계 익절로 인해 CAGR이 **{cagr_diff:.2f}%p 상승**했습니다.\n"
    
    if mdd_diff > 0:  # MDD is negative, so positive diff means shallower
        report += f"- **MDD 개선**: 다단계 익절로 인해 MDD가 **{abs(mdd_diff):.2f}%p 개선**되었습니다.\n"
    else:
        report += f"- **MDD 악화**: 다단계 익절로 인해 MDD가 **{abs(mdd_diff):.2f}%p 악화**되었습니다.\n"
    
    report += f"""
---

## 결론

E03 SSOT에서 다단계 익절을 미적용한 결정의 근거:

> ❌ **다단계 익절**: E03에서 미사용 (시그널 기반 전환만 적용)

본 백테스트 결과:

"""
    
    if cagr_diff < -1.0:
        report += "- ✅ **SSOT 결정 유지 권장**: 다단계 익절이 CAGR을 유의미하게 하락시킴\n"
    elif mdd_diff > 3.0 and cagr_diff > -0.5:
        report += "- ⚠️ **재검토 필요**: MDD 개선 효과가 CAGR 손실보다 클 수 있음\n"
    else:
        report += "- ✅ **SSOT 결정 유지 권장**: 다단계 익절의 이점이 명확하지 않음\n"
    
    report += f"""
---

_Generated by QuantNeural Backtest Engine v2026.1_
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("E03 다단계 익절 백테스트 실험")
    print("=" * 60)
    
    # Download data
    print("\n[1/4] Downloading data...")
    prices = download_data(PRIMARY_START, PRIMARY_END)
    print(f"     Data range: {prices.index[0]} ~ {prices.index[-1]}")
    print(f"     Total days: {len(prices)}")
    
    # Run baseline
    print("\n[2/4] Running E03 Baseline...")
    baseline = run_e03_baseline(prices, cost_bps=COST_BPS, tax_rate=TAX_RATE)
    print(f"     CAGR: {baseline.metrics['CAGR']*100:.2f}%")
    print(f"     MDD:  {baseline.metrics['MDD']*100:.2f}%")
    
    # Run with partial exit
    print("\n[3/4] Running E03 + Partial Exit...")
    partial = run_e03_with_partial_exit(prices, cost_bps=COST_BPS, tax_rate=TAX_RATE)
    print(f"     CAGR: {partial.metrics['CAGR']*100:.2f}%")
    print(f"     MDD:  {partial.metrics['MDD']*100:.2f}%")
    
    # Generate outputs
    print("\n[4/4] Generating outputs...")
    
    plot_comparison(baseline, partial, os.path.join(ARTIFACTS_DIR, "equity_comparison.png"))
    generate_report(baseline, partial, os.path.join(ARTIFACTS_DIR, "REPORT_E03_PARTIAL_EXIT.md"))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Baseline':>12} {'Partial Exit':>14} {'Diff':>10}")
    print("-" * 60)
    print(f"{'CAGR':<20} {baseline.metrics['CAGR']*100:>11.2f}% {partial.metrics['CAGR']*100:>13.2f}% {(partial.metrics['CAGR']-baseline.metrics['CAGR'])*100:>+9.2f}%p")
    print(f"{'MDD':<20} {baseline.metrics['MDD']*100:>11.2f}% {partial.metrics['MDD']*100:>13.2f}% {(partial.metrics['MDD']-baseline.metrics['MDD'])*100:>+9.2f}%p")
    print(f"{'Sharpe':<20} {baseline.metrics['Sharpe']:>12.2f} {partial.metrics['Sharpe']:>14.2f} {partial.metrics['Sharpe']-baseline.metrics['Sharpe']:>+10.2f}")
    print(f"{'Trades/Year':<20} {baseline.metrics['TradesPerYear']:>12.1f} {partial.metrics['TradesPerYear']:>14.1f} {partial.metrics['TradesPerYear']-baseline.metrics['TradesPerYear']:>+10.1f}")
    print("=" * 60)
    
    print(f"\n📁 Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
