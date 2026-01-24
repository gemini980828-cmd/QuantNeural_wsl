# -*- coding: utf-8 -*-
"""추가 분석: 연도별 수익률, Drawdown 차트, 통계 상세"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/juwon/QuantNeural/scripts')
from compare_200tq_optimized import (
    download_ohlc, build_synth_tqqq_ohlc_from_qqq, splice_actual_into_synth_ohlc,
    backtest_akitqq_ssot_simple, backtest_akitqq_ssot_full, backtest_qqq_3_161_off_exposure,
    max_drawdown, cagr, annual_vol, sharpe
)


def analyze_yearly_performance(curves):
    """연도별 수익률 계산"""
    yearly = {}
    for eq in curves:
        yearly_rets = eq.resample('YE').last().pct_change().dropna()
        yearly[eq.name] = yearly_rets
    
    df = pd.DataFrame(yearly)
    df.index = df.index.year
    return df


def plot_drawdowns(curves, save_path):
    """Drawdown 차트"""
    fig, axes = plt.subplots(len(curves), 1, figsize=(14, 3*len(curves)), sharex=True)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    for ax, eq, c in zip(axes, curves, colors):
        peak = eq.cummax()
        dd = (eq / peak - 1.0) * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color=c)
        ax.plot(dd, color=c, linewidth=0.8)
        ax.set_ylabel('Drawdown %')
        ax.set_title(f'{eq.name} (MDD: {dd.min():.1f}%)', fontsize=11)
        ax.set_ylim(-70, 5)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Drawdown chart saved to: {save_path}")
    plt.close()


def extended_statistics(curves):
    """확장 통계"""
    stats = {}
    for eq in curves:
        r = eq.pct_change().dropna()
        
        # 승률 (일간)
        win_rate = (r > 0).sum() / len(r) * 100
        
        # 최고/최악 일간 수익률
        best_day = r.max() * 100
        worst_day = r.min() * 100
        
        # 최고/최악 월간 수익률
        monthly = eq.resample('ME').last().pct_change().dropna()
        best_month = monthly.max() * 100
        worst_month = monthly.min() * 100
        
        # 연속 상승/하락일
        positive = (r > 0).astype(int)
        negative = (r < 0).astype(int)
        
        # Calmar Ratio
        calmar = cagr(eq) / abs(max_drawdown(eq)) if max_drawdown(eq) != 0 else np.nan
        
        # Sortino Ratio (downside deviation only)
        downside = r[r < 0]
        downside_std = downside.std(ddof=0) * np.sqrt(252) if len(downside) > 0 else np.nan
        sortino = (r.mean() * 252) / downside_std if downside_std > 0 else np.nan
        
        stats[eq.name] = {
            'CAGR': f"{cagr(eq)*100:.2f}%",
            'MDD': f"{max_drawdown(eq)*100:.2f}%",
            'Volatility': f"{annual_vol(r)*100:.2f}%",
            'Sharpe': f"{sharpe(r):.2f}",
            'Sortino': f"{sortino:.2f}",
            'Calmar': f"{calmar:.2f}",
            'Win Rate(Daily)': f"{win_rate:.1f}%",
            'Best Day': f"{best_day:.2f}%",
            'Worst Day': f"{worst_day:.2f}%",
            'Best Month': f"{best_month:.2f}%",
            'Worst Month': f"{worst_month:.2f}%",
        }
    
    return pd.DataFrame(stats).T


def main():
    print("="*80)
    print("         200TQ vs Optimized 확장 분석")
    print("="*80)
    
    # 데이터 로드
    print("\n📥 Downloading data...")
    data = download_ohlc(["QQQ", "TQQQ", "SPY", "SHV"], "2000-01-01", "2025-12-31")
    
    qqq = data["QQQ"].copy()
    spy = data["SPY"].copy()
    shv = data["SHV"].copy()
    
    # 합성 TQQQ
    synth_tqqq = build_synth_tqqq_ohlc_from_qqq(qqq, annual_fee=0.02, base=100.0)
    if "TQQQ" in data:
        synth_tqqq = synth_tqqq.reindex(qqq.index).ffill().dropna()
        actual_tqqq = data["TQQQ"].reindex(qqq.index).dropna()
        tqqq = splice_actual_into_synth_ohlc(synth_tqqq, actual_tqqq)
    else:
        tqqq = synth_tqqq.reindex(qqq.index).ffill().dropna()
    
    # 공통 인덱스
    common_idx = qqq.index.intersection(tqqq.index).intersection(spy.index).intersection(shv.index)
    qqq = qqq.loc[common_idx]
    tqqq = tqqq.loc[common_idx]
    spy = spy.loc[common_idx]
    shv = shv.loc[common_idx]
    
    qqq_close = qqq["Close"]
    tqqq_close = tqqq["Close"]
    spy_close = spy["Close"].rename("SPLG")
    shv_close = shv["Close"].rename("CASH")
    
    # 전략 실행
    print("📈 Running backtests...")
    curves = [
        backtest_akitqq_ssot_simple(tqqq, spy_close, shv_close),
        backtest_akitqq_ssot_full(tqqq, spy_close, shv_close),
        backtest_qqq_3_161_off_exposure(qqq_close, tqqq_close, shv_close, 0.0),
        backtest_qqq_3_161_off_exposure(qqq_close, tqqq_close, shv_close, 0.10),
        backtest_qqq_3_161_off_exposure(qqq_close, tqqq_close, shv_close, 0.20),
    ]
    
    # 공통 인덱스
    common = curves[0].index
    for s in curves[1:]:
        common = common.intersection(s.index)
    curves = [s.loc[common] for s in curves]
    
    # 1. 확장 통계
    print("\n" + "="*80)
    print("                      📊 확장 통계 (Extended Statistics)")
    print("="*80)
    ext_stats = extended_statistics(curves)
    pd.set_option('display.width', 200)
    print(ext_stats.to_string())
    
    # 2. 연도별 수익률
    print("\n" + "="*80)
    print("                      📅 연도별 수익률 (Yearly Returns)")
    print("="*80)
    yearly = analyze_yearly_performance(curves)
    yearly_fmt = (yearly * 100).round(2).astype(str) + '%'
    print(yearly_fmt.to_string())
    
    # 3. 연도별 승패
    print("\n" + "="*80)
    print("                      🏆 연도별 승률 (Positive Years)")
    print("="*80)
    for eq in curves:
        yearly_rets = eq.resample('YE').last().pct_change().dropna()
        positive_years = (yearly_rets > 0).sum()
        total_years = len(yearly_rets)
        print(f"  {eq.name}: {positive_years}/{total_years} ({positive_years/total_years*100:.1f}%)")
    
    # 4. Drawdown 차트
    print("\n📊 Generating drawdown charts...")
    plot_drawdowns(curves, "/home/juwon/QuantNeural/artifacts/drawdown_comparison.png")
    
    # 5. 연도별 히트맵 저장
    print("\n📊 Generating yearly heatmap...")
    fig, ax = plt.subplots(figsize=(12, 8))
    yearly_pct = yearly * 100
    
    # 히트맵
    im = ax.imshow(yearly_pct.T.values, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    
    ax.set_xticks(range(len(yearly_pct.index)))
    ax.set_xticklabels(yearly_pct.index, rotation=45, ha='right')
    ax.set_yticks(range(len(yearly_pct.columns)))
    ax.set_yticklabels(yearly_pct.columns)
    
    # 값 표시
    for i in range(len(yearly_pct.columns)):
        for j in range(len(yearly_pct.index)):
            val = yearly_pct.iloc[j, i]
            color = 'white' if abs(val) > 30 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=8)
    
    ax.set_title('Yearly Returns Heatmap (%)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Return %')
    plt.tight_layout()
    plt.savefig('/home/juwon/QuantNeural/artifacts/yearly_returns_heatmap.png', dpi=150, bbox_inches='tight')
    print("📊 Heatmap saved to: /home/juwon/QuantNeural/artifacts/yearly_returns_heatmap.png")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)
    print("\n📁 생성된 파일:")
    print("  - /home/juwon/QuantNeural/artifacts/compare_200tq_optimized.png (Equity Curves)")
    print("  - /home/juwon/QuantNeural/artifacts/drawdown_comparison.png (Drawdowns)")
    print("  - /home/juwon/QuantNeural/artifacts/yearly_returns_heatmap.png (Heatmap)")
    print("  - /home/juwon/QuantNeural/artifacts/REPORT_200TQ_vs_Optimized.md (Report)")


if __name__ == "__main__":
    main()
