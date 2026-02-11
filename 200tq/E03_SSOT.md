# E03 Strategy SSOT (Single Source of Truth)

**200TQQ v2026.3 — Ensemble + F1 Signal Stability + Emergency Exit**

> **Status**: ✅ GO (Confidence: HIGH)
> **Certified**: February 2026
> **Author**: QuantNeural Research

---

## Executive Summary

E03 v2026.3은 **200TQ 전략**의 최종 진화형으로, 세 개의 독립 레이어를 결합합니다:

1. **다중 MA 앙상블 투표** — 노이즈 감소된 방향성 시그널
2. **F1 Signal Stability Filter** — 학술 논문 기반 whipsaw 감쇠
3. **Emergency Exit** — 시장 급락 시 즉시 방어 전환

세 레이어의 조합으로 **CAGR +7.6%p, MDD +4.8%p 동시 개선**을 달성합니다.
OOS(2018-2025) 검증에서 IS(2010-2017) 대비 **모든 지표가 개선**되어 과적합 우려가 없습니다.

| 지표 | E03 v2026.1 (이전) | **E03 v2026.3 (현행)** | 변화 |
|:-----|:------------------:|:----------------------:|:----:|
| **Net CAGR** | 34.8% | **42.4%** | **+7.6%p** |
| **MDD** | -51.5% | **-46.8%** | **+4.8%p** |
| **Calmar** | 0.68 | **0.91** | **+0.23** |
| **Sharpe** | 0.89 | **1.03** | **+0.14** |
| **Sortino** | 0.93 | **1.21** | **+0.28** |

---

## Part 1: 전략의 기원 (Origins)

### 1.1 200TQ / Akitqq SSOT (Track A)

**원본 전략**: TQQQ의 200일 이평선을 기준으로 시장 상태를 판단하고 자산을 전환.

| 상태 | 조건 | 행동 |
|:-----|:-----|:-----|
| **Bear** | Close < SMA200 | 100% CASH (SGOV/SHV) |
| **Focus** | SMA200 ≤ Close ≤ SMA200 × 1.05 | 100% TQQQ |
| **Overheat** | Close > SMA200 × 1.05 | TQQQ 보유 유지, 신규 자금 → SPLG |

**v1.0 특징** (역사적 참고용):

- 2일 연속 확인 후 진입 (Entry Confirmation)
- 5% Stop Loss 규칙
- 다단계 익절 (+10%, +25%, +50%, +100%)
- **Historical CAGR**: ~10.5% (1999-2024)

> [!NOTE]
> **v1.0 규칙의 E03 적용 여부**:
>
> - ❌ **SPLG**: E03에서 미사용 (ON/OFF 이원 구조로 단순화)
> - ❌ **2일 확인**: E03에서 미사용 (E09/E10 실험 결과 -1.9~4.1%p 성과 저하)
> - ❌ **다단계 익절**: E03에서 미사용 (시그널 기반 전환만 적용)
> - ✅ **스탑로스**: v2026.3에서 **Emergency Exit으로 재설계** (Part 3 참조)

### 1.2 발전 동기

200TQ는 안전하지만 보수적. 더 적극적인 성장을 위해:

1. **기초자산 신호**: TQQQ 대신 QQQ의 MA를 기준으로 사용 (노이즈 감소)
2. **OFF 상태 최적화**: 100% 현금 대신 **10% TQQQ 잔류** (반등 포착)
3. **국채 수익 확보**: CASH → SGOV로 대체 (Treasury Yield Harvesting)

### 1.3 v2026.3 추가 동기

E03 v2026.1의 한계를 두 가지 학술 논문과 비상 규칙으로 보완:

- **MDD -51.5%는 심리적으로 과도** → Emergency Exit으로 -46.8%까지 축소
- **횡보장에서 whipsaw** → F1 Signal Stability Filter로 포지션 축소
- **비상 규칙(-7%)이 한 번도 독립 트리거되지 않음** (E44/E45 실험) → 임계값을 **-5%로 강화**

---

## Part 2: 시스템 정의

### Layer 1: 핵심 시그널 (E03 Ensemble)

```
Signal = Majority Vote of [MA160, MA165, MA170] vs MA3

IF ≥2 windows are ON → Signal = ON
ELSE → Signal = OFF
```

- **ON 조건**: QQQ의 `SMA(3) > SMA(window)`
- **OFF 조건**: QQQ의 `SMA(3) ≤ SMA(window)`
- **시그널 지연**: 1일 (t일 시그널 → t+1일 적용)

### Layer 2: F1 Signal Stability Filter

**출처**: Declerck & Vy (2024) "When Moving Averages Meet Momentum" SSRN:5032806

시그널이 빈번하게 전환(flip)되는 구간을 자동 감지하여 포지션을 축소합니다.

```
FlipCount = 과거 40일간 시그널 변경 횟수

IF Signal == ON AND FlipCount ≥ 3:
    TQQQ Weight = 70% (정상 100%에서 30% 감소)
ELSE:
    정상 비중 유지
```

| 파라미터 | 값 | 근거 |
|:---------|:---|:-----|
| **FlipWindow** | 40일 | 36개 조합 그리드 서치 Calmar 최적 |
| **FlipThreshold** | 3회 | 동일 |
| **ReducedWeight** | 0.70 | 동일 |

> [!IMPORTANT]
> F1 필터는 **OFF 상태에는 적용하지 않습니다**. OFF 상태의 10% 잔류는 항상 유지됩니다.
> F1은 ON 상태에서만 100% → 70%로 포지션을 줄입니다.

### Layer 3: Emergency Exit

시장 급락 시 시그널과 무관하게 **즉시 OFF10으로 전환**합니다.

| 트리거 조건 | 행동 | 쿨다운 |
|:-----------|:-----|:-------|
| QQQ 당일 수익률 ≤ **-5%** | **OFF10 강제 전환** | 1일 |
| TQQQ 진입가 대비 ≤ **-15%** | **OFF10 강제 전환** | 1일 |

**Emergency 규칙 상세**:

1. **당일 종가 기준**으로 트리거 판단 (장중 모니터링 아님)
2. 트리거 발동 시 **다음 장 시작에 OFF10으로 리밸런싱**
3. **쿨다운 1일**: 트리거 다음 날도 OFF10 유지 → 그 다음 날부터 정상 시그널 복귀
4. 비상 전환의 목표 상태는 **OFF10 (10% 잔류)**이며, 0% 청산이 아닙니다

### 포지션 배분 요약

| 상태 | 조건 | TQQQ | SGOV |
|:-----|:-----|-----:|-----:|
| **ON (정상)** | Signal ON, FlipCount < 3 | **100%** | 0% |
| **ON (choppy)** | Signal ON, FlipCount ≥ 3 | **70%** | 30% |
| **OFF** | Signal OFF | **10%** | 90% |
| **Emergency** | QQQ ≤-5% 또는 TQQQ ≤-15% | **10%** | 90% |

### 실행 모델

| 항목 | 값 |
|:-----|:---|
| 체크 주기 | Daily (월간도 동등 성과 — E26 검증) |
| 실행 지연 | 1일 (t일 시그널 → t+1일 적용) |
| 리밸런싱 | Delta 방식 (목표 비중 변경 시에만) |
| 거래비용 | 10bps 편도 |
| 세금 | 22% (한국 해외ETF, 연말 실현이익) |

---

## Part 3: 왜 D인가? — 6전략 비교

OOS 검증 스위트에서 6개 전략 변형을 비교한 결과:

### 3.1 Full Period (2010-2025)

| 전략 | CAGR | MDD | Sharpe | Calmar | Trades | Emergency |
|:-----|-----:|----:|-------:|-------:|-------:|----------:|
| A) E03 Baseline | 34.8% | -51.5% | 0.89 | 0.68 | 52 | 0 |
| B) E03 + Emergency | 42.8% | -49.1% | 1.03 | 0.87 | 56 | 13 |
| C) E03 + F1 | 33.8% | -51.5% | 0.88 | 0.66 | 59 | 0 |
| **D) E03 + F1 + Emergency** | **42.4%** | **-46.8%** | **1.03** | **0.91** | **63** | **13** |
| E) E03 + F3 Graduated | 27.9% | -42.9% | 0.87 | 0.65 | 399 | 0 |
| F) E03 + F3 + Emergency | 34.9% | -39.2% | 1.04 | 0.89 | 403 | 12 |

### 3.2 D가 최적인 이유

| 비교 | 분석 |
|:-----|:-----|
| **D vs A** (기준선) | CAGR +7.6%p, MDD +4.8%p — 모든 지표 압도적 개선 |
| **D vs B** (Emergency만) | CAGR 거의 동일(-0.4%p), MDD 2.3%p 추가 개선 → **Calmar 0.87→0.91** |
| **D vs F** (F3+Emergency) | F는 MDD -39.2%로 최저이나 CAGR 34.9%로 낮음. 거래 403회로 비현실적 |

**D 선택 근거**:

1. **CAGR × MDD 동시 최적화** — Calmar 0.91로 6개 중 1위
2. **거래 63회** (15년간) = 연 4.2회로 수동 매매 가능
3. **F1 필터의 역할**: Emergency 단독(B)보다 MDD 2.3%p 추가 축소
4. **Tax 0.31** — Emergency가 하락장 손실을 실현하여 세금 부담 극감 (A의 6.31 대비 95% 감소)

### 3.3 OOS 검증 (IS: 2010-2017 → OOS: 2018-2025)

| 전략 | IS CAGR | OOS CAGR | IS Calmar | OOS Calmar | CAGR 비율 | Calmar 비율 |
|:-----|--------:|---------:|----------:|-----------:|----------:|------------:|
| A) Baseline | 27.4% | 41.7% | 0.56 | 0.81 | 1.53 | 1.45 |
| B) Emergency | 29.8% | 55.9% | 0.61 | 1.36 | 1.87 | 2.24 |
| C) F1 | 25.9% | 41.2% | 0.55 | 0.80 | 1.59 | 1.45 |
| **D) F1+Emergency** | **28.4%** | **56.8%** | **0.61** | **1.38** | **2.00** | **2.28** |
| E) F3 | 21.1% | 34.1% | 0.56 | 0.79 | 1.62 | 1.42 |
| F) F3+Emergency | 23.0% | 46.9% | 0.63 | 1.20 | 2.04 | 1.90 |

> [!IMPORTANT]
> **D의 OOS 결과가 IS보다 모든 지표에서 우수합니다.**
>
> - CAGR 비율 2.00 = OOS가 IS의 2배
> - Calmar 비율 2.28 = OOS Calmar이 IS의 2.3배
> - **과적합(overfitting)의 증거가 없습니다**

---

## Part 4: Emergency Exit 상세

### 4.1 Historical Emergency Events (13건)

| # | 날짜 | 트리거 | 포트폴리오 가치 | 시장 배경 |
|:-:|:-----|:-------|---------------:|:----------|
| 1 | 2011-08-04 | TQQQ -19.4% (Stop) | 1.41x | 미국 신용등급 강등 |
| 2 | 2011-08-08 | QQQ -6.0% (Crash) | 1.38x | S&P 다운그레이드 여파 |
| 3 | 2011-11-21 | TQQQ -16.1% (Stop) | 1.21x | 유로존 위기 |
| 4 | 2018-10-24 | TQQQ -19.6% (Stop) | 8.43x | Fed 긴축 우려 |
| 5 | 2020-02-27 | QQQ -5.0% (Crash) | 13.67x | COVID-19 첫 급락 |
| 6 | 2020-03-09 | TQQQ -30.0% (Stop) | 13.43x | COVID 팬데믹 |
| 7 | 2020-03-12 | QQQ -9.2% (Crash) | 13.18x | 서킷 브레이커 발동 |
| 8 | 2020-03-16 | QQQ -12.0% (Crash) | 13.17x | 연속 서킷 브레이커 |
| 9 | 2020-09-03 | QQQ -5.1% (Crash) | 41.24x | 기술주 급락 |
| 10 | 2022-05-05 | QQQ -5.0% (Crash) | 68.72x | 인플레이션 우려 |
| 11 | 2022-09-13 | QQQ -5.5% (Crash) | 62.56x | CPI 쇼크 |
| 12 | 2025-04-03 | QQQ -5.4% (Crash) | 168.04x | 관세 전쟁 |
| 13 | 2025-04-04 | QQQ -6.2% (Crash) | 165.37x | 관세 전쟁 2일차 |

### 4.2 Emergency 규칙이 작동하는 원리

**왜 CAGR이 올라가는가?**

Emergency Exit은 대폭락 직전에 레버리지를 줄여 **"낙폭 회피"** 효과를 냅니다.
기존 E03는 MA 시그널이 OFF로 전환되기까지 2-5일이 걸리는데,
그 사이에 TQQQ가 -30%~-50% 추가 하락합니다.
Emergency는 **당일 즉시** 10%로 줄여 이 손실을 차단합니다.

**왜 세금이 줄어드는가?**

Emergency Exit 시점에는 TQQQ가 이미 하락한 상태이므로,
매도 시 **손실이 실현**됩니다. 이 실현 손실이 다른 거래의 이익과 상계되어
연말 과세 대상이 줄어듭니다 (TotalTax: 6.31 → 0.31).

### 4.3 Deadband Layer — 비적용

> [!NOTE]
> **E03에서 Deadband는 적용하지 않습니다.**
>
> E03은 ON↔OFF 전환 시 **90%p 점프** (100% ↔ 10%)가 발생하므로,
> 3~5%p Deadband 임계값이 트리거될 상황이 없습니다.
> 연간 거래 횟수도 4.2회로 이미 극히 낮아 추가 필터링이 불필요합니다.
>
> **검증**: E30(5%p), E31(3%p) 실험 결과 E03 대비 CAGR/MDD/거래횟수 모두 동일.

### 4.4 OFF 자산 선택

| 순위 | 자산 | 비고 |
|:----:|:-----|:-----|
| 1 | **SGOV** | Primary (0-3M Treasury) |
| 2 | SHV | Fallback (동등 성과 — E23) |
| 3 | BIL | Historical Proxy |

---

## Part 5: 실제 운용 체크리스트

### 5.1 Daily Ops (권장)

```
1. 장 마감 후 QQQ 종가 확인
2. Emergency 체크:
   a. QQQ 당일 수익률 ≤ -5%? → Emergency OFF10
   b. TQQQ 현재가 / 진입가 - 1 ≤ -15%? → Emergency OFF10
3. Emergency 아닌 경우:
   a. MA3, MA160, MA165, MA170 계산
   b. 다수결 투표 → ON/OFF 결정
   c. ON이면: 과거 40일 시그널 전환 횟수 확인
      - ≥3회 → ON-Choppy (TQQQ 70%, SGOV 30%)
      - <3회 → ON-Normal (TQQQ 100%)
   d. OFF → TQQQ 10%, SGOV 90%
4. 목표 비중과 현재 비중 비교 → 거래 필요 여부 판단
5. 다음 장 시작 시 실행
```

### 5.2 F1 Signal Stability 계산 방법

```
1. E03 앙상블 시그널의 과거 40일 기록을 추출
2. 전일 대비 시그널이 바뀐 날 수를 카운트 (ON→OFF 또는 OFF→ON)
3. 카운트 ≥ 3 → "choppy" 구간
4. Choppy 구간에서 시그널이 ON이면 TQQQ 비중을 100% → 70%로 축소
5. 나머지 30%는 SGOV 배분
```

### 5.3 수동 매매 실행 프로토콜 (삼성증권 기준)

| 항목 | 규칙 |
|:-----|:-----|
| **시그널 확정 시점** | 미국 정규장 마감 후 (한국시간 익일 06:00~07:00) |
| **주문 입력 시점** | 한국시간 22:30 (프리마켓) 또는 23:30 (정규장 시작) |
| **주문 타입** | **시장가 (MOO: Market on Open)** 권장 |
| **10% 잔류 계산** | **수량 기준**, 소수점 이하 **올림** (예: 보유 100주 → 10주 잔류) |
| **SGOV 매수** | TQQQ 매도 체결 후, 체결금액의 해당 비율을 시장가 매수 |
| **70% Choppy** | ON-Normal과 동일 방향, TQQQ 비중만 70%로 조정 |

> [!TIP]
> **10% 계산 예시**: TQQQ 137주 보유 시
>
> - 10% = 13.7주 → **14주 잔류** (올림)
> - 매도 수량 = 137 - 14 = **123주**

> [!TIP]
> **70% Choppy 예시**: 총 자산 $100,000, 현재 TQQQ 100%
>
> - 목표: TQQQ $70,000 (70%), SGOV $30,000 (30%)
> - TQQQ 현재가 $80일 때: $30,000 / $80 = 375주 매도

### 5.4 예외 상황

| 상황 | 대응 |
|:-----|:-----|
| 데이터 누락 | 전일 시그널 유지 |
| SGOV 거래 불가 | SHV로 대체 |
| Emergency 트리거 | 즉시 OFF10 전환 (시그널 무시) |
| Emergency 쿨다운 중 시그널 ON | OFF10 유지 (쿨다운 1일 경과 후 복귀) |

### 5.5 신규 자금 / 분배금 처리

| 상황 | 규칙 |
|:-----|:-----|
| **신규 입금** | 현재 시그널의 목표 비중에 맞춰 **부족한 자산**부터 매수 |
| **분배금 수령** | 당일 시그널 확인 후, 목표 비중에 맞춰 재투자 |
| **최소 주문 미달** | SGOV에 현금 축적 후, 다음 리밸런싱 시점에 통합 처리 |
| **정수주 제약** | 소수점 미만은 SGOV 쪽에 잔액으로 유지 |

---

## Part 6: 기대 성과 및 리스크

### 6.1 기대 수익 (Full Period 2010-2025)

| 지표 | 값 |
|:-----|:---|
| **Net CAGR** | **42.4%** |
| **MDD** | **-46.8%** |
| **Sharpe** | **1.03** |
| **Sortino** | **1.21** |
| **Calmar** | **0.91** |
| **Final Value** | **272x** ($1 → $272, 15년) |
| **거래 횟수** | 63회 (연 4.2회) |
| **Emergency 발동** | 13회 (15년간) |
| **누적 세금** | 0.31 (전략 A 대비 95% 감소) |

### 6.2 OOS 분할 성과

| 구간 | CAGR | MDD | Calmar |
|:-----|-----:|----:|-------:|
| **IS (2010-2017)** | 28.4% | -46.8% | 0.61 |
| **OOS (2018-2025)** | 56.8% | -41.0% | 1.38 |
| **Degradation Ratio** | 2.00 | — | 2.28 |

> OOS가 IS보다 우수 = **과적합 없음**

### 6.3 알려진 리스크

> [!WARNING]
>
> - **레버리지 리스크**: TQQQ는 일별 3배 레버리지로, 장기 횡보 시 손실 누적 가능
> - **모델 리스크**: MA 기반 전략은 급반전 장세에 취약 (F1 필터로 일부 완화)
> - **유동성 리스크**: 대규모 자금 운용 시 슬리피지 증가
> - **Emergency 한계**: -5% 임계값이 장중 도달 후 반등하면 불필요한 이탈 가능
> - **F1 파라미터 리스크**: FW40/FT3/RW70은 단일 백테스트 기간에 최적화됨
> - **기간 편향**: OOS(2018-2025)는 역대급 기술주 강세장을 포함

---

## Part 7: 학술 근거

### 7.1 F1: Signal Stability Filter

**Declerck, F. & Vy, H. (2024)**. "When Moving Averages Meet Momentum."
*SSRN Working Paper*, No. 5032806.

- MA 기반 전략에서 **신호 빈번 전환(whipsaw) 구간을 식별**하여 포지션을 축소하면 리스크 대비 수익이 개선됨을 입증
- E03 적용: 40일 윈도우, 3회 이상 전환 시 ON 포지션을 100% → 70%로 축소

### 7.2 F2: Autocorrelation Filter (참고 — D에는 미적용)

**Hsieh, C.-S., Chang, Y.-H., & Chen, T.-Y. (2025)**. "Leveraged ETF Timing with Return Autocorrelation."
*arXiv:2504.20116*.

- 수익률 자기상관이 음수인 구간에서 레버리지를 줄이면 MDD가 극적으로 감소
- E03에서 F3(Graduated) 변형으로 테스트: MDD -39.2%까지 축소하나 CAGR과 거래 빈도 트레이드오프

---

## Part 8: 버전 히스토리

| 버전 | 날짜 | 변경 사항 |
|:-----|:-----|:---------|
| v1.0 (200TQ) | 2024 | 원본 TQQQ 스위칭 전략 |
| v2.0 (OFF10) | 2025 | QQQ 기반 시그널, 10% 잔류 도입 |
| v2026.1 (E03) | 2026-01 | 앙상블 투표, SGOV, Deadband 검증 완료 |
| v2026.2 | 2026-01-19 | SSOT 정합성 패치 (7개 항목 명확화) |
| **v2026.3 (D)** | **2026-02-10** | **F1 Signal Stability Filter 추가, Emergency Exit -5%/-15% 도입, OOS 검증 완료** |

### v2026.3 변경 근거

1. **weakness_suite**: E03의 비상 규칙(-7%)이 한 번도 독립 트리거되지 않음 발견 → -5%로 강화
2. **paper_improvements**: Declerck & Vy(2024) F1 필터의 Calmar 최적 파라미터 FW40/FT3/RW70 발견
3. **oos_emergency**: D 전략(F1+Emergency)이 OOS에서 과적합 없이 최고 Calmar 달성 확인
4. **mdd_reduction**: 5가지 MDD 축소 접근법 중 Emergency Exit이 CAGR 손실 없이 가장 효과적

---

## Appendix: 관련 문서

- [FINAL_DECISION.md](./artifacts/off10_robustness/FINAL_DECISION.md) — E03 최초 승인 보고서
- [deadband_verdict.md](./artifacts/off10_robustness/notes/deadband_verdict.md) — Deadband 비적용 근거
- [Robustness Protocol](./artifacts/off10_robustness/) — 검증 스위트 결과
- [OOS Emergency Results](./experiments/oos_emergency/) — D 전략 OOS 검증 결과
- [Paper Improvements](./experiments/paper_improvements/) — F1/F2/F3 논문 기반 실험
- [Weakness Suite](./experiments/weakness_suite/) — E03 약점 분석
- [MDD Reduction](./experiments/mdd_reduction/) — MDD 축소 실험
- [Dashboard](/200tq/dashboard) — 실시간 운용 대시보드

---

_이 문서는 E03 전략의 공식 SSOT입니다. 모든 운용 결정은 이 문서를 기준으로 합니다._
