# E03 Strategy Google Sheets 운용 시스템

## TL;DR

> **Quick Summary**: E03 v2026.3 전략(Ensemble + F1 Filter + Emergency Exit)의 Daily Ops를 Google Sheets 스프레드시트로 관리하는 시스템. Google Apps Script(.gs)를 실행하면 6개 탭(Dashboard, Signal, Emergency, TradeLog, Portfolio, Settings)이 자동 생성되며, GOOGLEFINANCE()로 가격 데이터를 자동 수집하고 SMA/투표/FlipCount/비상감지를 수식으로 자동 계산.
> 
> **Deliverables**:
> - `200tq/sheets/e03_sheet_builder.gs` — Google Apps Script 파일 (Sheet Editor에 붙여넣고 실행)
> - `200tq/sheets/E03_SHEET_SSOT.md` — 스프레드시트 블루프린트 문서 (탭/열/수식/서식 완전 명세)
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 (sequential .gs file build)

---

## Context

### Original Request
"어제 새로 만든 E03_SSOT를 가지고 대시보드 이전에 스프레드시트로 관리를 하려고 하는데요. 어떤식으로 스프레드 시트를 만들면 좋을지 계획을 세워주세요."

### Interview Summary
**Key Discussions**:
- 도구: Google Sheets + GOOGLEFINANCE() 자동 연동
- 기능 범위: 5개 모듈 전체 (Signal, Emergency, TradeLog, Portfolio, F1 FlipCount)
- 데이터 범위: 최근 1년 (2025~) — F1 필터 40일 윈도우 충족
- 통화: USD 기본 + KRW 환산 열 추가
- 현재 상태: 이미 TQQQ/SGOV 보유 중 → 초기 포지션 입력 필요
- 시트 구성: 탭 분리 (6개 탭)

**Research Findings**:
- E03 SSOT (v2026.3)는 3-Layer 전략: Ensemble → F1 Filter → Emergency Exit
- 포지션 상태: ON(100%), ON-Choppy(70%/30%), OFF10(10%/90%), Emergency(→OFF10)
- Daily Ops 체크리스트가 SSOT Part 5에 이미 정의됨
- 실행 모델: 1일 지연, 10bps 편도, 22% 세금, 10% 잔류(올림)
- 대시보드의 types.ts/buildViewModel.ts에 이미 ViewModel 패턴이 구현되어 있어 참고 가능

### Metis Review
**Identified Gaps (all resolved)**:
- 히스토리컬 데이터 초기화: GOOGLEFINANCE 자동 백필 (1년 데이터로 F1 40일 윈도우 충족)
- Emergency 진입가 정의: 가중평균 평균단가(삼성증권 표준) 적용
- 세금 추적 범위: 실현이익만 (SSOT 22% 기준 부합)
- Emergency 쿨다운: 자동 추적 열 포함
- GOOGLEFINANCE 데이터 신뢰성: IFERROR 래핑 + 수동 입력 폴백
- FlipCount Cold Start: "FlipCount 유효까지 N일" 카운터 포함
- 10% 잔류 소수점: CEILING() 함수 사용 + 최소 10주 경고

---

## Work Objectives

### Core Objective
E03 v2026.3 전략의 Daily Ops 체크리스트(SSOT Part 5.1)를 Google Sheets로 완전히 옮겨, 매일 시트를 열면 오늘의 시그널·비상상태·추천거래를 바로 확인할 수 있는 운용 도구를 만든다.

### Concrete Deliverables
1. `200tq/sheets/e03_sheet_builder.gs` — Google Apps Script 파일
2. `200tq/sheets/E03_SHEET_SSOT.md` — 블루프린트 명세 문서

### Definition of Done
- [x] Apps Script 파일이 JavaScript 문법 오류 없이 작성됨
- [x] 6개 탭 생성 함수가 모두 포함됨 (Dashboard, Signal, Emergency, TradeLog, Portfolio, Settings)
- [x] 블루프린트 문서가 모든 탭/열/수식/조건부서식을 완전히 명세함
- [x] SSOT Part 5.1의 Daily Ops 체크리스트가 100% 반영됨

### Must Have
- GOOGLEFINANCE()로 QQQ/TQQQ/SGOV 가격 자동 수집
- SMA(3/160/165/170) 자동 계산
- 앙상블 다수결 투표 → 전략 상태 자동 판정
- F1 FlipCount (40일 롤링) 자동 추적
- Emergency 조건(QQQ ≤-5%, TQQQ ≤-15%) 자동 감지
- 포트폴리오 목표 비중 대비 추천 거래 수량 자동 계산
- USD + KRW 이중 표시 (환율 GOOGLEFINANCE 자동)
- 조건부 서식: ON(초록), OFF10(빨강), Choppy(노랑), Emergency(보라)
- IFERROR() 래핑으로 GOOGLEFINANCE 장애 시 graceful degradation
- 10% 잔류 계산에 CEILING() 함수 사용 (올림)
- Emergency 쿨다운 자동 추적

### Must NOT Have (Guardrails)
- ❌ 차트/그래프 (대시보드에서 처리 예정)
- ❌ 백테스트/시뮬레이션 기능 (Python 스크립트로 이미 완료)
- ❌ 멀티 전략 지원 (E03 v2026.3 전용으로 하드코딩)
- ❌ 이메일/SMS 알림 자동화
- ❌ 성과 분석 (Sharpe, MDD 등 — 대시보드에서 처리)
- ❌ 세금 최적화 로직 (세금 추적은 정보 제공만)
- ❌ 외부 API 연동 (GOOGLEFINANCE만 사용)
- ❌ 자동 매매 기능 (수동 실행만)
- ❌ 개별 셀 조작 루프 (반드시 batch operation 사용)

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> 모든 검증은 agent가 도구(Bash, grep)를 사용하여 수행합니다.

### Test Decision
- **Infrastructure exists**: NO (Google Apps Script는 로컬 테스트 불가)
- **Automated tests**: NO (GAS는 Google Sheets 환경에서만 실행)
- **Framework**: N/A
- **Agent-Executed QA**: 파일 존재·구문 검증·구조 검증·블루프린트 정합성 검사

### QA 전략
Google Apps Script는 로컬에서 실행 불가하므로 QA는 다음 방식으로 수행:
1. **구문 검증**: `node --check` 로 JavaScript 문법 에러 검사 (GAS는 JS 슈퍼셋)
2. **구조 검증**: grep으로 필수 함수/탭 이름/수식 키워드 존재 확인
3. **블루프린트 정합성**: 블루프린트에 명세된 모든 탭/열이 .gs 파일에 구현되어 있는지 교차 확인
4. **SSOT 정합성**: E03_SSOT.md의 핵심 상수(160/165/170, 40일, 3회, 70%, -5%, -15%)가 .gs 파일에 포함되어 있는지 확인

---

## Spreadsheet Architecture (6-Tab Design)

### Tab 1: 📊 Dashboard (첫 번째 탭 — 매일 여는 화면)
| 영역 | 내용 |
|:-----|:-----|
| **Header** | 오늘 날짜, 데이터 상태(FRESH/STALE), 마지막 업데이트 시각 |
| **Verdict** | 전략 상태 (ON / ON-Choppy / OFF10 / Emergency) — 큰 글씨, 색상 배경 |
| **Evidence** | SMA160/165/170 투표 결과 (PASS/FAIL), Margin % |
| **Emergency** | QQQ 당일수익률, TQQQ 진입가대비, 쿨다운 상태 |
| **F1 Filter** | FlipCount 값, Choppy 여부, "유효까지 N일" 카운터 |
| **Action** | 추천 거래 (Sell X TQQQ, Buy Y SGOV 등), 목표 비중 |
| **Portfolio** | 현재 보유 현황 (USD/KRW), 총 자산, 일일 손익 |

### Tab 2: 📈 Signal (시그널 히스토리)
| 열 | 내용 | 수식 여부 |
|:---|:-----|:----------|
| A | Date | 수동/자동 |
| B | QQQ Close | GOOGLEFINANCE |
| C | SMA3 | =AVERAGE(B열 3일) |
| D | SMA160 | =AVERAGE(B열 160일) |
| E | SMA165 | =AVERAGE(B열 165일) |
| F | SMA170 | =AVERAGE(B열 170일) |
| G | Vote160 | =IF(C>D, "PASS", "FAIL") |
| H | Vote165 | =IF(C>E, "PASS", "FAIL") |
| I | Vote170 | =IF(C>F, "PASS", "FAIL") |
| J | Ensemble | =IF(COUNTIF(G:I,"PASS")>=2, "ON", "OFF") |
| K | FlipCount | =SUMPRODUCT(40일 윈도우 시그널 변경 횟수) |
| L | State | =IF(Emergency, "EMERGENCY", IF(J="OFF","OFF10", IF(K>=3,"ON-CHOPPY","ON"))) |
| M | Target TQQQ% | =SWITCH(L, "ON",100%, "ON-CHOPPY",70%, "OFF10",10%, "EMERGENCY",10%) |

### Tab 3: 🚨 Emergency (비상 감지)
| 열 | 내용 | 수식 여부 |
|:---|:-----|:----------|
| A | Date | Signal탭 참조 |
| B | QQQ Close | Signal탭 참조 |
| C | QQQ Daily Return | =(B_today - B_yesterday) / B_yesterday |
| D | QQQ Crash? | =IF(C <= -0.05, "TRIGGER", "SAFE") |
| E | TQQQ Current | GOOGLEFINANCE |
| F | TQQQ Entry (Avg) | Settings탭 참조 (가중평균 평균단가) |
| G | TQQQ Drawdown% | =(E-F)/F |
| H | TQQQ Stop? | =IF(G <= -0.15, "TRIGGER", "SAFE") |
| I | Emergency | =IF(OR(D="TRIGGER", H="TRIGGER"), "ACTIVE", "NONE") |
| J | Cooldown | 쿨다운 1일 자동 추적 |

### Tab 4: 📝 TradeLog (거래 기록)
| 열 | 내용 | 입력 방식 |
|:---|:-----|:----------|
| A | Date | 수동 입력 |
| B | Ticker | 드롭다운: TQQQ, SGOV |
| C | Action | 드롭다운: BUY, SELL, HOLD |
| D | Shares | 수동 입력 (양수 정수, 데이터 검증) |
| E | Price USD | 수동 입력 (양수) |
| F | Total USD | =D*E (자동) |
| G | USD/KRW | GOOGLEFINANCE("CURRENCY:USDKRW") |
| H | Total KRW | =F*G (자동) |
| I | Commission | =F*0.001 (10bps) |
| J | Signal State | 거래 시점의 전략 상태 (수동 또는 Signal탭 참조) |
| K | Note | 수동 입력 (메모) |

### Tab 5: 💼 Portfolio (포트폴리오 현황)
| 열 | 내용 | 수식 여부 |
|:---|:-----|:----------|
| A | Ticker | TQQQ, SGOV, CASH |
| B | Qty | TradeLog에서 집계 또는 수동 입력 |
| C | Avg Entry (USD) | 가중평균 (TradeLog 기반 또는 수동) |
| D | Current Price (USD) | GOOGLEFINANCE |
| E | Market Value (USD) | =B*D |
| F | Market Value (KRW) | =E*환율 |
| G | Weight % | =E/총자산 |
| H | Target Weight % | Dashboard의 목표 비중 참조 |
| I | Deviation % | =G-H |
| J | Unrealized PnL (USD) | =(D-C)*B |
| K | Unrealized PnL (KRW) | =J*환율 |
| L | Daily PnL (USD) | =(D_today - D_yesterday)*B |
| M | Recommended Trade | =IF(I>threshold, "Sell X shares", "Hold") |

### Tab 6: ⚙️ Settings (설정)
| 영역 | 내용 |
|:-----|:-----|
| **Strategy Constants** | SMA 윈도우(160,165,170), F1 윈도우(40), F1 임계값(3), Reduced Weight(0.70), Emergency QQQ(-0.05), Emergency TQQQ(-0.15), OFF 잔류(0.10) |
| **Portfolio Initial** | TQQQ 초기 수량, TQQQ 평균단가, SGOV 초기 수량, SGOV 평균단가, 현금잔고(KRW) |
| **Live Data** | QQQ 현재가, TQQQ 현재가, SGOV 현재가, USD/KRW 환율 (모두 GOOGLEFINANCE) |
| **OFF Asset** | Primary: SGOV, Fallback: SHV |
| **Execution** | 거래비용 10bps, 세금 22%, 실행지연 1일 |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: .gs 파일 생성 — Settings + PriceData 레이어
└── Task 5: 블루프린트 문서 작성 (독립)

Wave 2 (After Task 1):
└── Task 2: Signal 탭 — 앙상블 투표 + F1 FlipCount

Wave 3 (After Task 2):
└── Task 3: Emergency + TradeLog 탭

Wave 4 (After Task 3):
└── Task 4: Portfolio + Dashboard 탭 + 전체 Polish

Critical Path: Task 1 → Task 2 → Task 3 → Task 4
Parallel Speedup: Task 5 동시 실행으로 ~15% 시간 절약
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4 | 5 |
| 2 | 1 | 3 | 5 |
| 3 | 2 | 4 | 5 |
| 4 | 3 | None | 5 |
| 5 | None | None | 1, 2, 3, 4 |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 5 | task(category="unspecified-high") / task(category="writing") |
| 2 | 2 | task(category="unspecified-high") |
| 3 | 3 | task(category="unspecified-high") |
| 4 | 4 | task(category="unspecified-high") |

---

## TODOs

- [x] 1. Foundation — Settings + Price Data Layer (.gs 파일 생성)

  **What to do**:
  - `200tq/sheets/` 디렉터리 생성 (없으면)
  - `e03_sheet_builder.gs` 파일 생성
  - 메인 진입점 함수 `initializeE03Sheet()` 작성
    - 6개 탭 생성 함수를 순차 호출하는 오케스트레이터
    - 기존 시트가 있으면 덮어쓸지 확인하는 안전장치
  - 헬퍼 함수 작성:
    - `batchSetValues(sheet, range, values)` — 배치 쓰기
    - `batchSetFormulas(sheet, range, formulas)` — 배치 수식 설정
    - `safeGoogleFinance(ticker, attr)` — IFERROR 래핑된 GOOGLEFINANCE 수식 생성
    - `formatAsPercent(range)`, `formatAsCurrency(range)` — 서식 헬퍼
  - `createSettingsTab()` 함수 구현:
    - Strategy Constants 섹션: SMA 윈도우(160,165,170), F1 파라미터(40,3,0.70), Emergency 임계값(-0.05,-0.15), OFF 잔류(0.10)
    - Portfolio Initial 섹션: TQQQ/SGOV 초기 수량·평균단가 입력란, 현금잔고
    - Live Data 섹션: GOOGLEFINANCE 수식으로 QQQ/TQQQ/SGOV 현재가 + USD/KRW 환율 자동 갱신
    - 셀에 Named Range 설정 (예: `CFG_SMA160_WINDOW`, `CFG_TQQQ_ENTRY_PRICE` 등)
  - `createPriceHistoryHelper()` 함수 구현:
    - GOOGLEFINANCE("QQQ", "close", DATE(2025,1,1), TODAY(), "DAILY")로 과거 1년 QQQ 종가 자동 수집
    - 날짜순 정렬 (최신 → 과거)
    - SMA 계산의 기반 데이터로 사용
    - TQQQ/SGOV 현재가도 GOOGLEFINANCE로 설정
  - `onDailyUpdate()` 시간 트리거 함수 (선택적):
    - 매일 미국 장 마감 후 자동 실행되어 최신 데이터 반영
    - 트리거 설정 함수: `setupDailyTrigger()`

  **Must NOT do**:
  - 개별 셀 조작 루프 (반드시 batch operation)
  - GOOGLEFINANCE 외 외부 API 호출
  - 차트/그래프 추가

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Google Apps Script 파일 작성은 프론트엔드도 아니고 일반적 코딩 작업
  - **Skills**: []
    - Google Apps Script는 특정 skill이 필요 없음 (일반 JavaScript 기반)
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: 스프레드시트는 UI 작업이 아님

  **Parallelization**:
  - **Can Run In Parallel**: YES (Task 5와 동시 실행 가능)
  - **Parallel Group**: Wave 1 (with Task 5)
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None

  **References** (CRITICAL):

  **Pattern References**:
  - `200tq/E03_SSOT.md:135-141` — 포지션 배분 요약 (ON/Choppy/OFF10/Emergency 비중 표)
  - `200tq/E03_SSOT.md:144-151` — 실행 모델 (체크주기, 실행지연, 리밸런싱, 거래비용, 세금)
  - `200tq/E03_SSOT.md:293-299` — 삼성증권 실행 프로토콜 (시장가 MOO, 10% 잔류 올림)

  **API/Type References**:
  - `200tq/dashboard/lib/ops/e03/types.ts:1-4` — StrategyState, EmergencyState 타입 정의
  - `200tq/dashboard/lib/ops/e03/types.ts:71-105` — PortfolioPosition, PortfolioDerived, PortfolioSnapshot 인터페이스 (스프레드시트 Portfolio 탭 열 설계의 참조)

  **External References**:
  - Google Apps Script SpreadsheetApp: `https://developers.google.com/apps-script/reference/spreadsheet/spreadsheet-app`
  - GOOGLEFINANCE 함수 문법: `https://support.google.com/docs/answer/3093281`
  - Batch operations 패턴: `setValues()`, `setFormulas()` 사용 (공식 문서 Best Practices)

  **WHY Each Reference Matters**:
  - SSOT 135-141: Settings탭의 strategy constants 값과 Portfolio탭의 목표비중 계산 공식의 원천
  - SSOT 293-299: 10% 잔류 CEILING() 계산과 거래비용 10bps 상수의 근거
  - types.ts: 스프레드시트의 Portfolio 탭 열 구조를 대시보드 ViewModel과 일관되게 설계하기 위한 참조

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: .gs 파일 존재 및 구문 유효성
    Tool: Bash
    Preconditions: Task 1 완료
    Steps:
      1. ls -la 200tq/sheets/e03_sheet_builder.gs → 파일 존재 확인
      2. node --check 200tq/sheets/e03_sheet_builder.gs → 구문 에러 없음 확인
         (참고: GAS 전용 API는 node에서 undefined이지만 syntax error는 아님.
          SpreadsheetApp 등은 런타임 에러이지 구문 에러가 아니므로 --check 통과해야 함)
      3. wc -l 200tq/sheets/e03_sheet_builder.gs → 최소 200줄 이상
    Expected Result: 파일 존재, 구문 유효, 충분한 코드 분량
    Evidence: 각 명령어 stdout/stderr 캡쳐

  Scenario: 필수 함수 존재 확인
    Tool: Bash (grep)
    Preconditions: .gs 파일 존재
    Steps:
      1. grep -c "function initializeE03Sheet" 200tq/sheets/e03_sheet_builder.gs → 1
      2. grep -c "function createSettingsTab" 200tq/sheets/e03_sheet_builder.gs → 1
      3. grep -c "GOOGLEFINANCE" 200tq/sheets/e03_sheet_builder.gs → ≥ 4 (QQQ, TQQQ, SGOV, USDKRW)
      4. grep -c "IFERROR" 200tq/sheets/e03_sheet_builder.gs → ≥ 4 (모든 GOOGLEFINANCE에 래핑)
    Expected Result: 모든 필수 함수와 키워드 존재
    Evidence: grep 결과 캡쳐

  Scenario: 전략 상수 하드코딩 확인
    Tool: Bash (grep)
    Preconditions: .gs 파일 존재
    Steps:
      1. grep "160" 200tq/sheets/e03_sheet_builder.gs → SMA160 윈도우 존재
      2. grep "165" 200tq/sheets/e03_sheet_builder.gs → SMA165 윈도우 존재
      3. grep "170" 200tq/sheets/e03_sheet_builder.gs → SMA170 윈도우 존재
      4. grep "40" 200tq/sheets/e03_sheet_builder.gs → F1 윈도우 40일 존재
      5. grep "0.70\|70%" 200tq/sheets/e03_sheet_builder.gs → Reduced Weight 존재
      6. grep "0.05\|-5%" 200tq/sheets/e03_sheet_builder.gs → Emergency QQQ 임계값
      7. grep "0.15\|-15%" 200tq/sheets/e03_sheet_builder.gs → Emergency TQQQ 임계값
    Expected Result: SSOT의 모든 전략 상수가 코드에 존재
    Evidence: grep 매칭 결과
  ```

  **Evidence to Capture:**
  - [ ] Bash output: node --check 결과
  - [ ] Bash output: grep 필수함수 결과
  - [ ] Bash output: grep 전략상수 결과

  **Commit**: YES
  - Message: `feat(sheets): create E03 spreadsheet builder — Settings + PriceData layer`
  - Files: `200tq/sheets/e03_sheet_builder.gs`
  - Pre-commit: `node --check 200tq/sheets/e03_sheet_builder.gs`

---

- [x] 2. Signal Layer — 앙상블 투표 + F1 FlipCount + 상태 판정

  **What to do**:
  - `e03_sheet_builder.gs`에 `createSignalTab()` 함수 추가
  - Signal 탭 열 구조 구현:
    - Col A: Date (PriceHistory에서 참조 또는 자동 생성)
    - Col B: QQQ Close (GOOGLEFINANCE 또는 PriceHistory 참조)
    - Col C: SMA3 = AVERAGE(최근 3일 QQQ Close)
    - Col D: SMA160 = AVERAGE(최근 160일 QQQ Close)
    - Col E: SMA165 = AVERAGE(최근 165일 QQQ Close)
    - Col F: SMA170 = AVERAGE(최근 170일 QQQ Close)
    - Col G: Vote160 = IF(C > D, "PASS", "FAIL")
    - Col H: Vote165 = IF(C > E, "PASS", "FAIL")
    - Col I: Vote170 = IF(C > F, "PASS", "FAIL")
    - Col J: Ensemble = IF(COUNTIF(G:I, "PASS") >= 2, "ON", "OFF")
    - Col K: FlipCount = 과거 40일간 J열(Ensemble)의 시그널 전환 횟수 (SUMPRODUCT 패턴)
    - Col L: State = 복합 수식 (Emergency 확인 → OFF 확인 → Choppy 확인 → ON)
    - Col M: Target TQQQ% = SWITCH(State, "ON"→100%, "ON-CHOPPY"→70%, "OFF10"→10%, "EMERGENCY"→10%)
    - Col N: FlipCount 유효성 = IF(현재행 < 40, "N일 후 유효", "VALID") — Cold Start 표시
  - SMA 수식이 데이터 부족 시 IFERROR로 빈 문자열 반환하도록 처리
  - 1년분 데이터 행에 대해 수식을 배치로 설정 (약 250행)
  - 조건부 서식 적용:
    - Vote 열: PASS → 초록 배경, FAIL → 빨강 배경
    - State 열: ON → 진한 초록, ON-CHOPPY → 노랑, OFF10 → 빨강, EMERGENCY → 보라
    - FlipCount ≥ 3 → 노랑 강조
  - 열 너비/고정행(헤더) 설정

  **Must NOT do**:
  - Signal 탭에 차트 추가
  - F1 파라미터를 Signal 탭에 하드코딩 (Settings 탭의 Named Range 참조)
  - 개별 셀 루프로 수식 설정

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 복잡한 수식 로직과 Apps Script API 이해 필요
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (Task 1과 같은 파일)
  - **Parallel Group**: Wave 2 (sequential after Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References** (CRITICAL):

  **Pattern References**:
  - `200tq/E03_SSOT.md:79-91` — Layer 1 핵심 시그널 (앙상블 투표 로직, ON/OFF 조건, strict inequality)
  - `200tq/E03_SSOT.md:93-115` — Layer 2 F1 Signal Stability Filter (FlipWindow=40, FlipThreshold=3, ReducedWeight=0.70)
  - `200tq/E03_SSOT.md:280-288` — F1 Signal Stability 계산 방법 (5단계 절차)
  - `200tq/dashboard/lib/ops/e03/buildViewModel.ts:54-66` — 투표 로직 구현 패턴 (sma3 > smaWindow, strict inequality, voteCount >= 2)

  **WHY Each Reference Matters**:
  - SSOT 79-91: `SMA(3) > SMA(window)` (strict >)를 정확히 구현해야 함. >=가 아님
  - SSOT 93-115: F1 필터는 **OFF 상태에는 적용하지 않음** — 이 규칙을 State 수식에 반영해야 함
  - buildViewModel.ts 54-66: TypeScript 구현의 투표 로직 패턴을 Google Sheets 수식으로 동일하게 번역

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Signal 탭 함수 존재 확인
    Tool: Bash (grep)
    Preconditions: Task 2 완료
    Steps:
      1. grep -c "function createSignalTab" 200tq/sheets/e03_sheet_builder.gs → 1
      2. grep "SMA3\|SMA160\|SMA165\|SMA170" 200tq/sheets/e03_sheet_builder.gs → 4개 이상 매칭
      3. grep "COUNTIF\|countif" 200tq/sheets/e03_sheet_builder.gs → ≥1 (앙상블 다수결 수식)
      4. grep "SUMPRODUCT\|sumproduct" 200tq/sheets/e03_sheet_builder.gs → ≥1 (FlipCount 수식)
      5. grep "FlipCount\|flipCount\|flip_count" 200tq/sheets/e03_sheet_builder.gs → ≥1
    Expected Result: Signal 탭의 모든 핵심 수식 키워드 존재
    Evidence: grep 결과

  Scenario: 조건부 서식 코드 존재 확인
    Tool: Bash (grep)
    Preconditions: Task 2 완료
    Steps:
      1. grep "ConditionalFormatRule\|conditionalFormatRule\|newConditionalFormatRule" 200tq/sheets/e03_sheet_builder.gs → ≥1
      2. grep "setBackground\|setBackgroundRGB" 200tq/sheets/e03_sheet_builder.gs → ≥3 (ON/OFF/Choppy 각각)
    Expected Result: 조건부 서식 설정 코드 존재
    Evidence: grep 결과

  Scenario: strict inequality 확인 (> not >=)
    Tool: Bash (grep)
    Preconditions: Task 2 완료
    Steps:
      1. 수식 문자열에서 SMA3 > SMA160 패턴이 > (strict)인지 확인
      2. ">=" 가 SMA 비교에 사용되지 않았는지 확인
    Expected Result: SMA 비교는 strict greater-than 사용
    Evidence: 관련 코드 라인 출력
  ```

  **Commit**: YES
  - Message: `feat(sheets): add Signal tab — Ensemble voting, F1 FlipCount, state determination`
  - Files: `200tq/sheets/e03_sheet_builder.gs`
  - Pre-commit: `node --check 200tq/sheets/e03_sheet_builder.gs`

---

- [x] 3. Safety Layer — Emergency 감지 + Trade Log

  **What to do**:
  - `e03_sheet_builder.gs`에 `createEmergencyTab()` 함수 추가:
    - Col A: Date (Signal 탭 참조)
    - Col B: QQQ Close (Signal 탭 참조)
    - Col C: QQQ Daily Return = (B_today - B_prev) / B_prev
    - Col D: Crash Trigger = IF(C <= Settings!Emergency_QQQ, "🚨 TRIGGER", "✅ SAFE")
    - Col E: TQQQ Current Price (GOOGLEFINANCE)
    - Col F: TQQQ Entry Price (Settings탭의 가중평균 평균단가 참조)
    - Col G: TQQQ Drawdown % = (E - F) / F
    - Col H: Stop Trigger = IF(G <= Settings!Emergency_TQQQ, "🚨 TRIGGER", "✅ SAFE")
    - Col I: Emergency Status = IF(OR(D="🚨 TRIGGER", H="🚨 TRIGGER"), "🔴 ACTIVE", "🟢 NONE")
    - Col J: Cooldown = 이전일 Emergency ACTIVE였으면 "COOLDOWN", 아니면 "CLEAR"
    - 조건부 서식: TRIGGER → 빨강 배경+흰 글씨, ACTIVE → 보라 배경
  - `createTradeLogTab()` 함수 추가:
    - 열 구조: Date, Ticker, Action, Shares, Price(USD), Total(USD), Rate, Total(KRW), Commission, Signal State, Note
    - 데이터 검증(Data Validation):
      - Ticker: 드롭다운 ["TQQQ", "SGOV"]
      - Action: 드롭다운 ["BUY", "SELL", "HOLD"]
      - Shares: 양수 정수만 (> 0)
      - Price: 양수만 (> 0)
    - 자동 계산 열:
      - Total USD = Shares × Price
      - USD/KRW = GOOGLEFINANCE("CURRENCY:USDKRW")
      - Total KRW = Total USD × Rate
      - Commission = Total USD × 0.001 (10bps)
    - 헤더 행 고정, 열 너비 자동 조정

  **Must NOT do**:
  - Emergency 탭에 자동 매매 트리거 추가
  - TradeLog에 세금 최적화 로직 추가
  - 이메일 알림 기능

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Emergency 감지 로직과 Data Validation API 사용
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References** (CRITICAL):

  **Pattern References**:
  - `200tq/E03_SSOT.md:117-131` — Layer 3 Emergency Exit (트리거 조건, 쿨다운 규칙, 목표 상태 OFF10)
  - `200tq/E03_SSOT.md:206-222` — Historical Emergency Events 13건 (실제 트리거 사례 — 테스트 데이터로 활용 가능)
  - `200tq/E03_SSOT.md:314-320` — 예외 상황 대응 (데이터 누락, SGOV 거래불가, 쿨다운 중 시그널 ON)
  - `200tq/dashboard/lib/ops/e03/types.ts:21-26` — TradeLine 인터페이스 (action, ticker, shares, note)

  **WHY Each Reference Matters**:
  - SSOT 117-131: "당일 종가 기준" 트리거 판단, "다음 장 시작에 OFF10", "쿨다운 1일" — 이 3가지 시간 규칙이 정확히 수식에 반영되어야 함
  - SSOT 314-320: 쿨다운 중 시그널 ON이면 OFF10 유지 — 이 예외 규칙이 Cooldown 열 수식에 포함되어야 함
  - SSOT 206-222: Emergency 13건 이벤트에서 QQQ -5.0%~-12.0%, TQQQ -15.0%~-30.0% 범위 확인 → 수식의 임계값 검증

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Emergency + TradeLog 함수 존재
    Tool: Bash (grep)
    Preconditions: Task 3 완료
    Steps:
      1. grep -c "function createEmergencyTab" 200tq/sheets/e03_sheet_builder.gs → 1
      2. grep -c "function createTradeLogTab" 200tq/sheets/e03_sheet_builder.gs → 1
      3. grep "TRIGGER\|trigger" 200tq/sheets/e03_sheet_builder.gs → ≥2 (Crash + Stop)
      4. grep "Cooldown\|cooldown\|COOLDOWN" 200tq/sheets/e03_sheet_builder.gs → ≥1
      5. grep "requireValueInList\|newDataValidation\|DataValidation" 200tq/sheets/e03_sheet_builder.gs → ≥2 (Ticker + Action 드롭다운)
    Expected Result: Emergency와 TradeLog의 핵심 로직 존재
    Evidence: grep 결과

  Scenario: Emergency 임계값 정확성
    Tool: Bash (grep)
    Preconditions: Task 3 완료
    Steps:
      1. grep 에서 -0.05 또는 -5% 가 Emergency QQQ 판정에 사용되는지 확인
      2. grep 에서 -0.15 또는 -15% 가 Emergency TQQQ 판정에 사용되는지 확인
    Expected Result: SSOT의 임계값과 정확히 일치
    Evidence: 매칭 코드 라인
  ```

  **Commit**: YES
  - Message: `feat(sheets): add Emergency monitoring + TradeLog with data validation`
  - Files: `200tq/sheets/e03_sheet_builder.gs`
  - Pre-commit: `node --check 200tq/sheets/e03_sheet_builder.gs`

---

- [x] 4. Operations Layer — Portfolio + Dashboard + Global Polish

  **What to do**:
  - `e03_sheet_builder.gs`에 `createPortfolioTab()` 함수 추가:
    - 행 구조: TQQQ, SGOV, CASH (3행 고정)
    - 열: Ticker, Qty, Avg Entry(USD), Current Price(USD), Value(USD), Value(KRW), Weight%, Target%, Deviation%, Unrealized PnL(USD), Unrealized PnL(KRW), Daily PnL(USD), Recommended Trade
    - 자동 계산:
      - Current Price: GOOGLEFINANCE 참조
      - Value USD: =Qty × Current Price
      - Value KRW: =Value USD × 환율
      - Weight: =Value / Total Value
      - Target%: Signal 탭의 최신 State에서 참조 (ON→100%, Choppy→70%, OFF10→10%)
      - Deviation: =Weight - Target
      - Recommended Trade: Delta 계산 → "Sell X shares" 또는 "Buy Y shares" 또는 "HOLD"
        - 10% 잔류 계산 시 CEILING() 사용 (SSOT Part 5.3)
    - Total 행: 합계 Value(USD), Value(KRW)
    - 환율: Settings탭의 GOOGLEFINANCE("CURRENCY:USDKRW") 참조
  - `createDashboardTab()` 함수 추가:
    - **Header 영역**: 오늘 날짜 (=TODAY()), 데이터 상태 (GOOGLEFINANCE 정상/비정상), 마지막 업데이트
    - **Verdict 영역**: 
      - 현재 전략 상태 (Signal 탭 최신행 L열 참조) — 큰 글씨 (24pt)
      - 배경색: ON→초록, OFF10→빨강, Choppy→노랑, Emergency→보라
    - **Evidence 영역**: SMA160/165/170 투표 결과 + Margin % (Signal 탭 참조)
    - **F1 영역**: FlipCount 값 + "유효까지 N일" 카운터
    - **Emergency 영역**: QQQ 당일수익률, TQQQ Drawdown%, 쿨다운 상태
    - **Action 영역**: 추천 거래 (Portfolio 탭의 Recommended Trade 참조)
    - **Portfolio Summary 영역**: 총자산(USD/KRW), TQQQ/SGOV 비중, 일일 손익
    - Dashboard를 스프레드시트의 **첫 번째 탭**으로 이동 (setIndex(0))
  - `applyGlobalFormatting()` 함수 추가:
    - 모든 탭의 헤더 행 고정 (freezeRows(1))
    - 통화 열 서식 ($#,##0.00 / ₩#,##0)
    - 퍼센트 열 서식 (0.00%)
    - 열 너비 자동 조정
    - 시트 보호: Settings 탭의 Constants 영역 보호 (실수 수정 방지)

  **Must NOT do**:
  - Dashboard에 차트/스파크라인 추가
  - Portfolio에 세금 최적화 계산
  - 성과 분석 (Sharpe, Calmar 등)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Portfolio 계산, Dashboard 레이아웃, 전체 Polish — 가장 복합적인 태스크
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final, sequential)
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References** (CRITICAL):

  **Pattern References**:
  - `200tq/E03_SSOT.md:260-278` — Daily Ops 체크리스트 (6단계: 종가확인→Emergency체크→MA계산→투표→비중비교→실행)
  - `200tq/E03_SSOT.md:296-306` — 10% 잔류 올림 계산 예시 (137주→14주 잔류→123주 매도)
  - `200tq/E03_SSOT.md:307-311` — 70% Choppy 계산 예시 ($100,000 기준)
  - `200tq/dashboard/lib/ops/e03/types.ts:33-67` — E03ViewModel 인터페이스 (Dashboard 구조의 참조)
  - `200tq/dashboard/lib/ops/e03/buildViewModel.ts:139-223` — Expected Trades 계산 로직 (ON→SGOV매도+TQQQ매수, OFF10→10%잔류+SGOV매수)

  **WHY Each Reference Matters**:
  - SSOT 260-278: Dashboard 탭의 표시 순서는 이 Daily Ops 체크리스트의 순서를 따라야 함
  - SSOT 296-306: Portfolio 탭의 Recommended Trade 열에서 10% 잔류 올림 계산 (CEILING 함수) 구현 시 이 예시를 참조
  - buildViewModel.ts 139-223: ON→OFF10 전환 시 매도 수량 계산, OFF10→ON 전환 시 SGOV 매도+TQQQ 매수 로직을 수식으로 번역

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Portfolio + Dashboard + GlobalFormatting 함수 존재
    Tool: Bash (grep)
    Preconditions: Task 4 완료
    Steps:
      1. grep -c "function createPortfolioTab" 200tq/sheets/e03_sheet_builder.gs → 1
      2. grep -c "function createDashboardTab" 200tq/sheets/e03_sheet_builder.gs → 1
      3. grep -c "function applyGlobalFormatting" 200tq/sheets/e03_sheet_builder.gs → 1
      4. grep "CEILING\|ceiling\|Math.ceil" 200tq/sheets/e03_sheet_builder.gs → ≥1 (10% 잔류 올림)
      5. grep "CURRENCY:USDKRW\|USDKRW" 200tq/sheets/e03_sheet_builder.gs → ≥1 (환율)
      6. grep "freezeRows\|setFrozenRows" 200tq/sheets/e03_sheet_builder.gs → ≥1 (헤더 고정)
    Expected Result: Portfolio, Dashboard, GlobalFormatting 핵심 구현 존재
    Evidence: grep 결과

  Scenario: 전체 .gs 파일 6개 탭 함수 완성도
    Tool: Bash (grep)
    Preconditions: Task 4 완료 (전체 파일 완성)
    Steps:
      1. grep -c "function create.*Tab" 200tq/sheets/e03_sheet_builder.gs → 정확히 6 (또는 5+Helper)
      2. 6개 탭 이름 확인: grep "Dashboard\|Signal\|Emergency\|TradeLog\|Portfolio\|Settings" 200tq/sheets/e03_sheet_builder.gs → 각각 ≥1
      3. node --check 200tq/sheets/e03_sheet_builder.gs → Exit 0 (최종 구문 검증)
      4. wc -l 200tq/sheets/e03_sheet_builder.gs → 최소 500줄 이상 (6개 탭 + 헬퍼)
    Expected Result: 6개 탭 모두 구현, 구문 유효, 충분한 코드 분량
    Evidence: 각 명령어 결과

  Scenario: SSOT 정합성 최종 확인
    Tool: Bash (grep)
    Preconditions: 전체 .gs 파일 완성
    Steps:
      1. SSOT 핵심 상수 존재: 160, 165, 170, 40, 3, 0.70 (또는 70), -0.05 (또는 -5), -0.15 (또는 -15), 0.10 (또는 10)
      2. GOOGLEFINANCE 호출 최소 4개: QQQ, TQQQ, SGOV, USDKRW
      3. 6개 탭 이름 문자열 모두 존재
      4. initializeE03Sheet에서 모든 create 함수 호출 확인
    Expected Result: E03 SSOT v2026.3과 완벽히 일치하는 구현
    Evidence: grep/search 결과 종합
  ```

  **Commit**: YES
  - Message: `feat(sheets): complete E03 spreadsheet — Portfolio, Dashboard, global formatting`
  - Files: `200tq/sheets/e03_sheet_builder.gs`
  - Pre-commit: `node --check 200tq/sheets/e03_sheet_builder.gs`

---

- [x] 5. Blueprint Documentation — 스프레드시트 명세 문서

  **What to do**:
  - `200tq/sheets/E03_SHEET_SSOT.md` 작성
  - 문서 구조:
    1. **Status & Authority**: SSOT 패턴 (200tq/E03_SSOT.md 참조하는 authority chain)
    2. **Overview**: 목적, 6개 탭 요약, 데이터 흐름 다이어그램 (ASCII)
    3. **Tab Specifications**: 각 탭별 상세 명세
       - 열 정의 (이름, 타입, 수식/수동, 설명)
       - 조건부 서식 규칙 (색상 코드, 조건)
       - 데이터 검증 규칙 (드롭다운, 범위 제한)
       - 탭 간 참조 관계
    4. **Formula Reference**: 핵심 수식 목록
       - SMA 계산, 앙상블 투표, FlipCount, Emergency 감지, 목표비중, 추천거래
    5. **User Guide**: 
       - 초기 설정 방법 (Script Editor에 .gs 붙여넣기 → initializeE03Sheet() 실행)
       - 초기 포트폴리오 입력 (Settings 탭)
       - 일일 운용 워크플로우 (Daily Ops)
       - 수동 거래 기록 방법 (TradeLog 탭)
    6. **Limitations & Known Issues**:
       - FlipCount Cold Start (첫 40일 부정확)
       - GOOGLEFINANCE 데이터 지연/누락 가능
       - 주말/공휴일 데이터 처리
    7. **SSOT Cross-Reference**: E03_SSOT.md 대비 매핑 테이블

  **Must NOT do**:
  - 백테스트 결과/성과 분석 포함
  - 대시보드 마이그레이션 계획
  - 전략 변경 제안

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 기술 문서 작성이 주 업무
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: None
  - **Blocked By**: None (SSOT 참조만으로 작성 가능)

  **References** (CRITICAL):

  **Pattern References**:
  - `200tq/E03_SSOT.md` — 전체 문서 (전략 명세의 원천, 모든 수치/규칙 참조)
  - `200tq/dashboard/E03_Command_Center_SSOT_v2.md` — 기능 명세 SSOT 패턴 (문서 구조 참조)
  - `200tq/dashboard/E03_UX_SSOT.md` — UX SSOT 패턴 (Zone 구조 참조)

  **WHY Each Reference Matters**:
  - E03_SSOT.md: 블루프린트에서 인용할 모든 전략 상수, 규칙, 예시의 원천
  - Command Center SSOT: SSOT 문서 작성 패턴 (Status/Authority/Non-Negotiables 구조)
  - UX SSOT: 정보 계층 구조 (Zone A-D 패턴)을 참고하여 Dashboard 탭 레이아웃 명세

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: 블루프린트 문서 완성도
    Tool: Bash (grep)
    Preconditions: Task 5 완료
    Steps:
      1. ls -la 200tq/sheets/E03_SHEET_SSOT.md → 파일 존재
      2. wc -l 200tq/sheets/E03_SHEET_SSOT.md → 최소 200줄
      3. grep "Dashboard\|Signal\|Emergency\|TradeLog\|Portfolio\|Settings" 200tq/sheets/E03_SHEET_SSOT.md → 6개 탭 이름 모두 존재
      4. grep "GOOGLEFINANCE" 200tq/sheets/E03_SHEET_SSOT.md → ≥1
      5. grep "SMA160\|SMA165\|SMA170" 200tq/sheets/E03_SHEET_SSOT.md → ≥3
      6. grep "FlipCount\|Flip Count" 200tq/sheets/E03_SHEET_SSOT.md → ≥1
      7. grep "Emergency" 200tq/sheets/E03_SHEET_SSOT.md → ≥3
      8. grep "initializeE03Sheet\|initial" 200tq/sheets/E03_SHEET_SSOT.md → ≥1 (User Guide 포함)
    Expected Result: 6개 탭, 핵심 수식, 유저 가이드가 모두 문서화됨
    Evidence: grep 결과

  Scenario: SSOT 교차 참조 정합성
    Tool: Bash (grep)
    Preconditions: Task 5 완료
    Steps:
      1. grep "E03_SSOT\|SSOT" 200tq/sheets/E03_SHEET_SSOT.md → ≥2 (원본 SSOT 참조)
      2. grep "v2026.3" 200tq/sheets/E03_SHEET_SSOT.md → ≥1 (버전 명시)
    Expected Result: 원본 SSOT 참조 및 버전이 명확히 기재됨
    Evidence: grep 결과
  ```

  **Commit**: YES (Task 4와 그룹으로 커밋 가능)
  - Message: `docs(sheets): add E03 spreadsheet blueprint SSOT document`
  - Files: `200tq/sheets/E03_SHEET_SSOT.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(sheets): create E03 sheet builder — Settings + PriceData` | `200tq/sheets/e03_sheet_builder.gs` | `node --check` |
| 2 | `feat(sheets): add Signal tab — Ensemble, F1 FlipCount, state` | `200tq/sheets/e03_sheet_builder.gs` | `node --check` |
| 3 | `feat(sheets): add Emergency + TradeLog tabs` | `200tq/sheets/e03_sheet_builder.gs` | `node --check` |
| 4 | `feat(sheets): complete sheet — Portfolio, Dashboard, formatting` | `200tq/sheets/e03_sheet_builder.gs` | `node --check` |
| 5 | `docs(sheets): add E03 spreadsheet blueprint SSOT` | `200tq/sheets/E03_SHEET_SSOT.md` | N/A |

---

## Success Criteria

### Verification Commands
```bash
# 1. 파일 존재
ls -la 200tq/sheets/e03_sheet_builder.gs  # Expected: file exists
ls -la 200tq/sheets/E03_SHEET_SSOT.md     # Expected: file exists

# 2. JavaScript 구문 유효성
node --check 200tq/sheets/e03_sheet_builder.gs  # Expected: no output (success)

# 3. 필수 함수 존재 (6개 탭 + 메인 + 글로벌 = 최소 8개 함수)
grep -c "function " 200tq/sheets/e03_sheet_builder.gs  # Expected: >= 8

# 4. 6개 탭 이름 모두 존재
for tab in Dashboard Signal Emergency TradeLog Portfolio Settings; do
  grep -c "$tab" 200tq/sheets/e03_sheet_builder.gs
done  # Expected: each >= 1

# 5. SSOT 핵심 상수 존재
grep -c "160\|165\|170" 200tq/sheets/e03_sheet_builder.gs  # Expected: >= 6

# 6. 블루프린트 분량
wc -l 200tq/sheets/E03_SHEET_SSOT.md  # Expected: >= 200 lines
```

### Final Checklist
- [x] 모든 "Must Have" 항목이 .gs 파일에 구현됨
- [x] 모든 "Must NOT Have" 항목이 없음 (차트, 백테스트, 멀티전략 등)
- [x] E03_SSOT.md의 Daily Ops 체크리스트(Part 5.1)가 100% 반영됨
- [x] 모든 GOOGLEFINANCE 호출에 IFERROR 래핑됨
- [x] 10% 잔류 계산에 CEILING 사용됨
- [x] node --check 통과
- [x] 블루프린트가 모든 탭/열/수식을 명세함
