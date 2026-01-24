# -*- coding: utf-8 -*-
# 필요 라이브러리: yfinance, pandas, numpy, matplotlib
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Metrics
# -----------------------------
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def cagr(equity: pd.Series, trading_days_per_year: int = 252) -> float:
    if len(equity) < 2:
        return np.nan
    years = len(equity) / trading_days_per_year
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)

def annual_vol(daily_ret: pd.Series, trading_days_per_year: int = 252) -> float:
    return float(daily_ret.std(ddof=0) * np.sqrt(trading_days_per_year))

def sharpe(daily_ret: pd.Series, rf_daily: float = 0.0, trading_days_per_year: int = 252) -> float:
    ex = daily_ret - rf_daily
    sd = ex.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(ex.mean() / sd * np.sqrt(trading_days_per_year))


# -----------------------------
# Data Download (Adjusted OHLC)
# -----------------------------
def download_ohlc(tickers, start, end):
    """
    auto_adjust=True로 조정 반영 OHLC를 받아서
    분할/배당 영향을 최대한 반영한 가격으로 백테스트.
    """
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
        raise RuntimeError("yfinance 다운로드 결과가 비어 있습니다(네트워크/차단/야후 오류 가능).")

    out = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            cols = [(t, c) for c in ["Open", "High", "Low", "Close"] if (t, c) in raw.columns]
            if len(cols) != 4:
                # 일부 환경에서 예외가 있을 수 있어 방어
                continue
            df = raw[cols].copy()
            df.columns = ["Open", "High", "Low", "Close"]
        else:
            # 단일 티커 요청 시
            if set(["Open", "High", "Low", "Close"]).issubset(set(raw.columns)):
                df = raw[["Open", "High", "Low", "Close"]].copy()
            else:
                continue

        df = df.dropna()
        out[t] = df

    if len(out) == 0:
        raise RuntimeError("다운로드된 OHLC 데이터가 없습니다. 티커/기간을 확인하세요.")

    return out


# -----------------------------
# Synthetic TQQQ (OHLC) from QQQ OHLC
# -----------------------------
def build_synth_tqqq_ohlc_from_qqq(qqq_ohlc: pd.DataFrame, annual_fee: float = 0.02, base: float = 100.0):
    """
    합성 TQQQ OHLC:
    - 기준: 전일 QQQ 종가 대비 당일 QQQ의 O/H/L/C 변화율을 계산한 뒤 3배로 확대
    - 비용: 일간 fee = annual_fee/252 를 각 O/H/L/C 변화율에 동일하게 차감(근사)
    - OHLC 일관성: Low <= min(Open,Close), High >= max(Open,Close) 보정(미세 오차 방어)

    주의: 레버리지 ETF의 실제 구조(리밸런싱/괴리/경로의존)를 완벽히 재현하지는 못합니다.
    """
    fee_d = annual_fee / 252.0
    q = qqq_ohlc.copy()

    prev_c = q["Close"].shift(1)
    # 전일 종가 대비 각 값의 변화율
    r_o = (q["Open"] / prev_c) - 1.0
    r_h = (q["High"] / prev_c) - 1.0
    r_l = (q["Low"] / prev_c) - 1.0
    r_c = (q["Close"] / prev_c) - 1.0

    # 3배 + 비용 근사
    tr_o = 3.0 * r_o - fee_d
    tr_h = 3.0 * r_h - fee_d
    tr_l = 3.0 * r_l - fee_d
    tr_c = 3.0 * r_c - fee_d

    # 가격 생성(전일 합성 종가 기반)
    idx = q.index
    out = pd.DataFrame(index=idx, columns=["Open", "High", "Low", "Close"], dtype=float)
    out.iloc[0] = [base, base, base, base]

    for i in range(1, len(idx)):
        prev_close = float(out["Close"].iloc[i - 1])
        o = prev_close * (1.0 + float(tr_o.iloc[i]))
        h = prev_close * (1.0 + float(tr_h.iloc[i]))
        l = prev_close * (1.0 + float(tr_l.iloc[i]))
        c = prev_close * (1.0 + float(tr_c.iloc[i]))

        # OHLC 일관성 보정
        lo = min(l, o, c)
        hi = max(h, o, c)
        out.iloc[i] = [o, hi, lo, c]

    out = out.dropna()
    out.index.name = q.index.name
    return out


def splice_actual_into_synth_ohlc(synth: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    합성 OHLC에 실제 OHLC를 스플라이스.
    - 최초 겹치는 날짜 t0에서 Close가 연속되도록 스케일링 후
    - t0 이후 실제값(스케일된)을 사용.
    """
    common = synth.index.intersection(actual.index)
    if len(common) == 0:
        return synth.copy()

    t0 = common[0]
    s0 = float(synth.loc[t0, "Close"])
    a0 = float(actual.loc[t0, "Close"])
    if s0 <= 0 or a0 <= 0:
        return synth.copy()

    scale = s0 / a0
    actual_scaled = actual.copy()
    for c in ["Open", "High", "Low", "Close"]:
        actual_scaled[c] = actual_scaled[c] * scale

    out = synth.copy()
    out.loc[actual_scaled.index, ["Open", "High", "Low", "Close"]] = actual_scaled[["Open", "High", "Low", "Close"]]
    return out


# -----------------------------
# Backtest Engine: Optimized QQQ 3/161 with OFF exposure
# -----------------------------
def backtest_qqq_3_161_off_exposure(
    qqq_close: pd.Series,
    tqqq_close: pd.Series,
    cash_close: pd.Series,
    off_weight: float = 0.0,
):
    """
    최적화 버전:
    - 신호: QQQ SMA(3) > SMA(161) 이면 ON
    - 비중: ON이면 TQQQ 100%, OFF이면 TQQQ off_weight(0/0.1/0.2), 나머지 CASH
    - 신호는 종가 기준 확정, 다음 거래일에 반영(shift(1))

    반환: equity curve
    """
    ma3 = qqq_close.rolling(3).mean()
    ma161 = qqq_close.rolling(161).mean()
    on = (ma3 > ma161).astype(int).shift(1).fillna(0).astype(int)

    w_t = on * 1.0 + (1 - on) * float(off_weight)
    w_c = 1.0 - w_t

    r_t = tqqq_close.pct_change().fillna(0.0)
    r_c = cash_close.pct_change().fillna(0.0)

    r = w_t * r_t + w_c * r_c
    eq = (1.0 + r).cumprod()
    return eq.rename(f"Optim_QQQ_3_161_OFF{int(off_weight*100)}")


# -----------------------------
# Backtest Engine: Akitqq SSOT (200TQ) - Simplified version
# -----------------------------
def backtest_akitqq_ssot_simple(
    tqqq_ohlc: pd.DataFrame,
    splg_close: pd.Series,
    cash_close: pd.Series,
    monthly_contribution: float = 0.0,
):
    """
    아기티큐 SSOT 간소화 버전(200TQ):
    - 기본 규칙: TQQQ Close > MA200 이면 TQQQ 100%, 아니면 CASH 100%
    - 신호는 종가 기준 확정, 다음 거래일에 반영(shift(1))
    - 스탑로스/익절 없이 단순 MA 스위칭
    """
    t_close = tqqq_ohlc["Close"]
    
    # MA200
    ma200 = t_close.rolling(200).mean()
    
    # 신호: 종가 > MA200이면 ON, 다음 거래일에 반영
    on = (t_close > ma200).astype(int).shift(1).fillna(0).astype(int)
    
    w_t = on * 1.0  # TQQQ weight
    w_c = 1.0 - w_t  # CASH weight
    
    r_t = t_close.pct_change().fillna(0.0)
    r_c = cash_close.pct_change().fillna(0.0)
    
    r = w_t * r_t + w_c * r_c
    eq = (1.0 + r).cumprod()
    
    return eq.rename("200TQ_Simple")


def backtest_akitqq_ssot_full(
    tqqq_ohlc: pd.DataFrame,
    splg_close: pd.Series,
    cash_close: pd.Series,
    monthly_contribution: float = 0.0,
):
    """
    아기티큐 SSOT(일봉 근사) - 버그 수정 버전:
    - 상태판단: TQQQ Close vs MA200, upper=MA200*1.05
        * 하락: close < MA200  -> TQQQ/SPLG 전량 매도, CASH 100%
        * 집중: MA200 <= close <= upper -> CASH를 TQQQ로 전환(기존 SPLG는 유지)
        * 과열: close > upper -> 기존 TQQQ 유지, 신규자금은 SPLG (월납입금만 SPLG로)
    - 매수(하락->상승 전환): '하루 더 확인' (2일 연속 MA200 위) 후 진입
    - 스탑로스: TQQQ 평균단가 대비 -5% 스탑
    - 부분익절: +10/+25/+50에서 10%씩
    """
    t = tqqq_ohlc.copy()
    t_close = t["Close"]
    t_low = t["Low"]

    # MA
    ma200 = t_close.rolling(200).mean()
    upper = ma200 * 1.05
    above = (t_close >= ma200)

    idx = t.index

    # 포트(달러 기준)
    cash = 0.0
    sh_tqqq = 0.0
    sh_splg = 0.0
    sh_cash = 0.0  # cash ETF(SHV) shares

    # 시작은 CASH(현금성 ETF)로 100%
    cash_px0 = float(cash_close.loc[idx[0]])
    sh_cash = 1.0 / cash_px0  # $1 worth of CASH ETF
    
    # TQQQ 평균단가(주수 기반)
    tqqq_shares_cost_basis = 0.0  # 총 원가(달러)

    # 익절 상태
    tp_10_done = False
    tp_25_done = False
    tp_50_done = False
    next_mult = 1.0  # +100%부터

    # '하루 더 확인' 상태
    pending_cross_up = False

    equity = pd.Series(index=idx, dtype=float)

    # 월 첫 거래일 플래그
    first_day_of_month = idx.to_series().dt.to_period("M").ne(idx.to_series().shift(1).dt.to_period("M"))
    first_day_of_month.iloc[0] = True

    for i, dt in enumerate(idx):
        # 오늘 가격
        px_t = float(t_close.loc[dt])
        px_s = float(splg_close.loc[dt])
        px_c = float(cash_close.loc[dt])
        
        # 포트 가치 평가
        port = cash + sh_tqqq * px_t + sh_splg * px_s + sh_cash * px_c
        equity.loc[dt] = port

        # 워밍업 구간은 스킵 (MA200 필요)
        if np.isnan(ma200.loc[dt]):
            continue

        # 장중 스탑로스: 오늘 Low가 스탑가 이하이면 즉시 청산
        if sh_tqqq > 0 and tqqq_shares_cost_basis > 0:
            avg = tqqq_shares_cost_basis / sh_tqqq
            stop_price = avg * 0.95
            if float(t_low.loc[dt]) <= stop_price:
                # TQQQ는 stop_price에 체결 가정
                cash += sh_tqqq * stop_price
                sh_tqqq = 0.0
                tqqq_shares_cost_basis = 0.0

                # SPLG도 종가에 매도
                cash += sh_splg * px_s
                sh_splg = 0.0

                # 익절 상태 리셋
                tp_10_done = tp_25_done = tp_50_done = False
                next_mult = 1.0
                pending_cross_up = False
                
                # **버그 수정**: 현금을 CASH ETF로 전환
                if cash > 0:
                    sh_cash += cash / px_c
                    cash = 0.0
                continue

        close = float(t_close.loc[dt])
        m200 = float(ma200.loc[dt])
        up = float(upper.loc[dt])

        is_bear = close < m200
        is_focus = (close >= m200) and (close <= up)
        is_overheat = close > up

        # 월 납입금 처리
        if first_day_of_month.loc[dt] and monthly_contribution > 0:
            cash += float(monthly_contribution)
            if is_bear:
                sh_cash += cash / px_c
                cash = 0.0
            elif is_focus:
                sh_tqqq += cash / px_t
                tqqq_shares_cost_basis += cash
                cash = 0.0
            else:  # 과열
                sh_splg += cash / px_s
                cash = 0.0

        # 하락 상태: 전량 CASH로 이동
        if is_bear:
            if sh_tqqq > 0:
                cash += sh_tqqq * px_t
                sh_tqqq = 0.0
                tqqq_shares_cost_basis = 0.0
                tp_10_done = tp_25_done = tp_50_done = False
                next_mult = 1.0

            if sh_splg > 0:
                cash += sh_splg * px_s
                sh_splg = 0.0

            # 모든 현금을 CASH ETF로
            if cash > 0:
                sh_cash += cash / px_c
                cash = 0.0

            pending_cross_up = False
            continue

        # 상승 전환 '하루 더 확인' 규칙
        if i >= 1:
            prev_dt = idx[i - 1]
            if not np.isnan(ma200.loc[prev_dt]):
                was_bear_yday = float(t_close.loc[prev_dt]) < float(ma200.loc[prev_dt])
                now_above = bool(above.loc[dt])

                if was_bear_yday and now_above:
                    pending_cross_up = True
                elif pending_cross_up and now_above:
                    # 확인 완료: CASH를 TQQQ로 전환
                    cash += sh_cash * px_c
                    sh_cash = 0.0
                    if cash > 0:
                        sh_tqqq += cash / px_t
                        tqqq_shares_cost_basis += cash
                        cash = 0.0
                    pending_cross_up = False
                elif pending_cross_up and not now_above:
                    pending_cross_up = False

        # 집중 구간: CASH를 TQQQ로 전환
        if is_focus and not pending_cross_up:
            cash += sh_cash * px_c
            sh_cash = 0.0
            if cash > 0:
                sh_tqqq += cash / px_t
                tqqq_shares_cost_basis += cash
                cash = 0.0

        # 부분익절 트리거
        if sh_tqqq > 0 and tqqq_shares_cost_basis > 0:
            avg = tqqq_shares_cost_basis / sh_tqqq
            ret = (px_t / avg) - 1.0

            sell_frac = 0.0
            if (not tp_10_done) and ret >= 0.10:
                sell_frac += 0.10
                tp_10_done = True
            if (not tp_25_done) and ret >= 0.25:
                sell_frac += 0.10
                tp_25_done = True
            if (not tp_50_done) and ret >= 0.50:
                sell_frac += 0.10
                tp_50_done = True

            while ret >= next_mult:
                sell_frac += 0.50
                next_mult += 1.0

            if sell_frac > 0:
                sell_frac = min(1.0, sell_frac)
                sell_sh = sh_tqqq * sell_frac
                proceeds = sell_sh * px_t
                sh_tqqq -= sell_sh
                tqqq_shares_cost_basis *= (1.0 - sell_frac)
                # 익절 대금을 SPLG로 재투자
                sh_splg += proceeds / px_s

        # 남은 현금을 CASH ETF로 보관
        if cash > 0:
            sh_cash += cash / px_c
            cash = 0.0

    return equity.dropna().rename("200TQ_SSOT_Full")


# -----------------------------
# Runner: Compare
# -----------------------------
def run_compare(
    start="2000-01-01",
    end="2025-12-31",
    monthly_contribution=0.0,
):
    print("Downloading OHLC data via yfinance ...")
    tickers = ["QQQ", "TQQQ", "SPY", "SHV"]
    data = download_ohlc(tickers, start, end)

    if "QQQ" not in data:
        raise RuntimeError("QQQ OHLC 데이터가 필요합니다.")
    if "SPY" not in data:
        raise RuntimeError("SPY(=SPLG proxy) OHLC 데이터가 필요합니다.")
    if "SHV" not in data:
        print("WARNING: SHV 데이터가 없어 CASH 수익률을 0%로 근사합니다.")
        data["SHV"] = pd.DataFrame(index=data["QQQ"].index, data={
            "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0
        })

    qqq = data["QQQ"].copy()
    spy = data["SPY"].copy()
    shv = data["SHV"].copy()

    # 합성 TQQQ OHLC
    synth_tqqq = build_synth_tqqq_ohlc_from_qqq(qqq, annual_fee=0.02, base=100.0)

    # 실제 TQQQ가 있으면 스플라이스
    if "TQQQ" in data and len(data["TQQQ"]) > 0:
        actual_tqqq = data["TQQQ"].copy()
        synth_tqqq = synth_tqqq.reindex(qqq.index).ffill().dropna()
        actual_tqqq = actual_tqqq.reindex(qqq.index).dropna()
        tqqq = splice_actual_into_synth_ohlc(synth_tqqq, actual_tqqq)
    else:
        tqqq = synth_tqqq.reindex(qqq.index).ffill().dropna()

    # 공통 인덱스
    common_idx = qqq.index.intersection(tqqq.index).intersection(spy.index).intersection(shv.index)
    qqq = qqq.loc[common_idx]
    tqqq = tqqq.loc[common_idx]
    spy = spy.loc[common_idx]
    shv = shv.loc[common_idx]

    # 시리즈 추출
    qqq_close = qqq["Close"]
    tqqq_close = tqqq["Close"]
    spy_close = spy["Close"].rename("SPLG")
    shv_close = shv["Close"].rename("CASH")

    # 200TQ Simple (MA200 스위칭만)
    eq_200tq_simple = backtest_akitqq_ssot_simple(
        tqqq_ohlc=tqqq,
        splg_close=spy_close,
        cash_close=shv_close,
        monthly_contribution=float(monthly_contribution),
    )

    # 200TQ Full (스탑/익절 포함)
    eq_200tq_full = backtest_akitqq_ssot_full(
        tqqq_ohlc=tqqq,
        splg_close=spy_close,
        cash_close=shv_close,
        monthly_contribution=float(monthly_contribution),
    )

    # Optimized variants
    eq_opt_0  = backtest_qqq_3_161_off_exposure(qqq_close, tqqq_close, shv_close, off_weight=0.0)
    eq_opt_10 = backtest_qqq_3_161_off_exposure(qqq_close, tqqq_close, shv_close, off_weight=0.10)
    eq_opt_20 = backtest_qqq_3_161_off_exposure(qqq_close, tqqq_close, shv_close, off_weight=0.20)

    # Align curves
    curves = [eq_200tq_simple, eq_200tq_full, eq_opt_0, eq_opt_10, eq_opt_20]
    common = curves[0].index
    for s in curves[1:]:
        common = common.intersection(s.index)
    curves = [s.loc[common] for s in curves]

    # Summary
    rows = {}
    for s in curves:
        r = s.pct_change().fillna(0.0)
        rows[s.name] = {
            "CAGR": cagr(s),
            "MDD": max_drawdown(s),
            "AnnVol": annual_vol(r),
            "Sharpe": sharpe(r),
            "FinalMult": float(s.iloc[-1]),
            "Days": int(len(s)),
        }

    summary = pd.DataFrame(rows).T
    
    # Format for display
    out = summary.copy()
    out["CAGR"] = (out["CAGR"] * 100).round(2).astype(str) + "%"
    out["MDD"] = (out["MDD"] * 100).round(2).astype(str) + "%"
    out["AnnVol"] = (out["AnnVol"] * 100).round(2).astype(str) + "%"
    out["Sharpe"] = out["Sharpe"].round(2)
    out["FinalMult"] = out["FinalMult"].round(2).astype(str) + "x"
    out["Days"] = out["Days"].astype(int)
    
    print("\n" + "="*80)
    print("                      200TQ vs Optimized Backtest Summary")
    print("="*80)
    print(f"Period: {common[0].date()} ~ {common[-1].date()} ({len(common)} trading days)")
    print(f"Monthly Contribution: ${monthly_contribution:,.0f}")
    print("="*80)
    
    # Full width display
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 10)
    print(out.to_string())
    print("="*80)
    
    # Strategy descriptions
    print("\n📌 Strategy Descriptions:")
    print("  - 200TQ_Simple:     TQQQ MA(200) 단순 스위칭 (Close > MA200 → TQQQ, else CASH)")
    print("  - 200TQ_SSOT_Full:  SSOT 풀버전 (MA200 + 집중/과열 구간 + 5% 스탑 + 익절)")
    print("  - Optim_QQQ_3_161:  QQQ MA(3) vs MA(161) 크로스 기반 (OFF0/10/20 = OFF시 TQQQ 비중)")
    print()

    # Plot
    plt.figure(figsize=(14, 7))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    for s, c in zip(curves, colors):
        plt.plot(s, label=s.name, color=c, linewidth=1.5)
    plt.yscale("log")
    plt.title(f"Equity Curves (log scale) | {common[0].date()} ~ {common[-1].date()}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig("/home/juwon/QuantNeural/artifacts/compare_200tq_optimized.png", dpi=150, bbox_inches="tight")
    print("📊 Plot saved to: /home/juwon/QuantNeural/artifacts/compare_200tq_optimized.png")
    plt.show()

    return summary, curves


if __name__ == "__main__":
    run_compare(start="2000-01-01", end="2025-12-31", monthly_contribution=0.0)
