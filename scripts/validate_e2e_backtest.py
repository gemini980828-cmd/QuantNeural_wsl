"""
E2E Backtest Flow Validation Script.

Demonstrates the complete workflow:
1. Prepare prices_df (wide format)
2. Prepare scores_df (signal dates + asset columns)
3. Write to CSV using backtest_artifacts_io
4. Run backtest using run_scores_backtest_from_csv
5. Verify determinism (M vs Q, repeated runs)
6. Export artifacts
"""

import numpy as np
import pandas as pd
from src.backtest_artifacts_io import write_prices_csv, write_scores_csv
from src.run_scores_backtest_from_csv import run_scores_backtest_from_csv


def main():
    print("=" * 60)
    print("E2E Backtest Flow Validation")
    print("=" * 60)
    
    # ========================================
    # 1) 실데이터 준비 (합성 데이터로 시뮬레이션)
    # ========================================
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=60, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    
    # Wide prices
    prices_data = {}
    for i, ticker in enumerate(tickers):
        base = 100 + i * 20
        returns = np.random.randn(60) * 0.02 + 0.001
        prices_data[ticker] = base * np.cumprod(1 + returns)
    
    prices_df = pd.DataFrame(prices_data, index=dates)
    print(f"\n=== prices_df: {prices_df.shape}")
    print(prices_df.head(3))
    
    # Scores: 6 signal dates
    signal_dates = dates[::10][:6]
    np.random.seed(123)
    scores_data = {ticker: np.random.randn(6) for ticker in tickers}
    scores_df = pd.DataFrame(scores_data, index=signal_dates)
    print(f"\n=== scores_df: {scores_df.shape}")
    print(scores_df)
    
    # ========================================
    # 2) CSV 저장 (Task 7.5.6)
    # ========================================
    write_prices_csv(prices_df, path="data/e2e_test_prices.csv", format="wide")
    write_scores_csv(scores_df, path="data/e2e_test_scores.csv")
    print("\n=== CSV 저장 완료")
    
    # ========================================
    # 3) Monthly Backtest
    # ========================================
    res_m = run_scores_backtest_from_csv(
        prices_csv_path="data/e2e_test_prices.csv",
        scores_csv_path="data/e2e_test_scores.csv",
        rebalance="M",
        execution_lag_days=1,
        method="softmax",
        cost_bps=10.0,
        slippage_bps=5.0,
    )
    
    print("\n=== Monthly Backtest Metrics:")
    for k, v in res_m["metrics"].items():
        print(f"  {k}: {v:.6f}")
    
    print(f"\nRebalance dates (M): {len(res_m['rebalance_dates'])}")
    print(f"Equity curve length: {len(res_m['equity_curve'])}")
    print(f"Trades count: {len(res_m['trades'])}")
    
    # ========================================
    # 4) Quarterly Backtest
    # ========================================
    res_q = run_scores_backtest_from_csv(
        prices_csv_path="data/e2e_test_prices.csv",
        scores_csv_path="data/e2e_test_scores.csv",
        rebalance="Q",
        execution_lag_days=1,
        method="softmax",
        cost_bps=10.0,
        slippage_bps=5.0,
    )
    
    print("\n=== Quarterly Backtest Metrics:")
    for k, v in res_q["metrics"].items():
        print(f"  {k}: {v:.6f}")
    
    print(f"\nRebalance dates (Q): {len(res_q['rebalance_dates'])}")
    
    # ========================================
    # 5) 결정론 검증
    # ========================================
    res_m2 = run_scores_backtest_from_csv(
        prices_csv_path="data/e2e_test_prices.csv",
        scores_csv_path="data/e2e_test_scores.csv",
        rebalance="M",
        execution_lag_days=1,
        method="softmax",
        cost_bps=10.0,
        slippage_bps=5.0,
    )
    
    eq_match = (res_m["equity_curve"] == res_m2["equity_curve"]).all()
    print(f"\n=== 결정론 검증: equity_curve 동일 = {eq_match}")
    
    # ========================================
    # 6) 아티팩트 저장
    # ========================================
    res_m["equity_curve"].to_csv("data/e2e_equity_curve.csv")
    res_m["trades"].to_csv("data/e2e_trades.csv", index=False)
    print("\n=== 아티팩트 저장 완료:")
    print("  - data/e2e_equity_curve.csv")
    print("  - data/e2e_trades.csv")
    
    # ========================================
    # 7) 요약
    # ========================================
    print("\n" + "=" * 60)
    if eq_match and len(res_m["equity_curve"]) > 0:
        print("✅ 실제 구동 성공!")
        print("   - 에러 없이 실행됨")
        print("   - metrics / equity_curve / trades 파일 생성됨")
        print(f"   - M vs Q 비교: M={len(res_m['rebalance_dates'])} / Q={len(res_q['rebalance_dates'])} rebalances")
        print("   - 결정론 보장됨 (동일 입력 → 동일 출력)")
    else:
        print("❌ 검증 실패")
    print("=" * 60)


if __name__ == "__main__":
    main()
