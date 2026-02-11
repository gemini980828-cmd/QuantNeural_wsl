# E03 Command Center — UX SSOT v3.0 (Functional-First, v2026.3 aligned)

- **Document Status**: `SSOT_V3.0_DRAFT`
- **Product**: E03 Command Center (Personal Ops Dashboard)
- **Strategy Authority**: `E03_SSOT.md` (E03 Strategy SSOT)
- **Functional Authority**: `E03_Command_Center_SSOT_v2.md` (기능 SSOT v2.0)
- **Hard Constraint**: **브로커(삼성증권) 자동 연동 없음**
- **Timezone**: KST 표기 기본(옵션: ET 병기)
- **Last Updated**: 2026-01-19 (KST)

> 본 문서는 **디자인(색/그리드/타이포/애니메이션) 결정 없이도** 개발이 가능한 수준으로,
> **기능적 요구사항과 UX의 유기성(동선/상태/가드레일/입력 최소화)**에만 집중한 SSOT입니다.

---

## 1) UX 비가역 원칙(Non‑Negotiables)

### 1.1 Rule is King (규칙 우선)
- UI는 사용자가 **규칙을 의심하도록 만드는 정보(뉴스/차트/의견)**를 판단/실행 전에 노출하지 않는다.
- 정보 소비는 **항상 하단(Zone D) 또는 ‘의도적 진입’**으로 격리한다.

### 1.2 Decision → Execution → Information (인지 흐름 강제)
사용자가 항상 아래 순서로만 진행하도록 UX를 강제한다.
1) **시스템/데이터 상태**(비상 여부, 데이터 신선도)
2) **오늘 판정(Verdict)** (종가 기반, 확정)
3) **내일 실행(Execution)** (t+1)
4) 실행 이후에만 **정보/회고/시뮬레이션**을 소비

### 1.3 Two Truths Model (전략 진실 vs 현실 진실)
- **전략 진실(자동)**: 신호/목표 비중/예상 주문(“해야 하는 것”)
- **현실 진실(수동)**: 실제 보유/실제 체결/실행 여부(“실제로 한 것”)
- 둘은 **데이터 모델, UI 구조, 문구**에서 절대 섞지 않는다.
  - 예: “예상 주문(자동)”을 “실제 체결(수동)”처럼 보이게 만들지 않는다.

### 1.4 Execution Lag 강제 (t일 판정 → t+1 실행)
- 모든 핵심 출력(신호, 주문, 경보)은 **Verdict Date / Execution Date**를 **항상 동시 표기**한다.
- 사용자가 “오늘 신호를 오늘 체결해야 한다”고 오해할 여지를 **0으로** 만든다.

### 1.5 Fail‑safe UX (누락/오류에서도 안전)
- 데이터 지연/결측, 휴장/조기폐장, 입력 누락, 시뮬레이터 조작 상태에서:
  - **잘못된 주문 생성/복사**가 발생하면 안 된다.
  - 항상 “보류 + 경고 + 최소 행동(기록/재시도)”로 수렴한다.

---

## 2) 정보 구조(IA) — Zone 기반 (디자인 비결정)

### 2.1 Zone 정의
- **Zone A: Global Watchdog**
  - 핵심 질문: “지금 비상인가?” “데이터가 최신인가?” “오늘/내일 실행이 필요한가?”
- **Zone B: Signal Core**
  - 핵심 질문: “오늘 판정은?” “내일 실행은?”
- **Zone C: Ops Console**
  - 핵심 질문: “내일 주문은 몇 주/얼마인가?” “실행/기록은 완료됐나?”
- **Zone D: Intel Lab**
  - 핵심 질문: “회고/분석/시뮬(선택)”

### 2.2 상호작용 우선순위
- Zone A/B/C는 **매일 사용**(1~2분)
- Zone D는 **선택적 사용**(정보는 노이즈가 될 수 있으므로 ‘의도적 진입’ 필요)

---

## 3) 핵심 상태 모델(State Model) — UX가 명확해야 하는 “정의역”

브로커 연동이 없기 때문에, UX는 아래 상태를 **명확히 분리/표현**해야 한다.

### 3.1 Data State (데이터 신뢰)
- `FRESH_CLOSE`: 종가 기반 데이터가 최신이며 신호 계산 가능
- `STALE`: 종가 업데이트 지연/결측(신호는 전일 유지 또는 계산 보류)
- `MARKET_CLOSED`: 휴장/조기폐장 등(신호/실행 타임라인 자동 조정)

**UX 규칙**
- `STALE` 또는 `MARKET_CLOSED` 상태에서는
  - 주문 생성/복사는 원칙적으로 **잠금(LOCK)** 또는 **보수적 모드(전일 유지 + 강한 경고)** 로 처리한다.

### 3.2 Strategy State (전략 판정)
- `ON` / `ON_CHOPPY` / `OFF10` / `EMERGENCY`
- 판정 근거는 항상: `SMA(3) vs SMA(160/165/170)` + “2/3 투표 결과”
- ON 계열은 F1 Filter(FlipWindow=40, FlipThreshold=3)로 `ON`과 `ON_CHOPPY`를 분기

### 3.3 Execution State (실행 상태)
- `NO_ACTION`: 목표 비중 변화 없음 → 실행할 주문 없음
- `SCHEDULED`: 내일 실행 필요(Execution Date 존재)
- `DUE_TODAY`: 오늘이 Execution Date
- `RECORDED`: 사용자가 실행/체결 기록까지 완료
- `UNKNOWN`: 실행 여부를 확인할 수 없음(기록 누락/미입력)

**UX 규칙**
- `DUE_TODAY`일 때는 **‘주문 카드’와 ‘기록’**이 최상단 우선순위를 가져야 한다.
- `UNKNOWN`은 실패가 아니라 **연동 부재 환경의 정상 상태**이므로, “기록으로 개선할 수 있는 상태”로 안내한다.

### 3.4 Emergency State (비상 상태)
- `NONE`
- `SOFT_ALERT`: 장중 조기 경보(자동 전환/강제 주문 금지)
- `HARD_CONFIRMED`: 종가 확정 트리거(내일 OFF10 실행 준비)

**UX 규칙**
- `SOFT_ALERT`에서는 **“준비”만** 허용(계산/체크리스트), **강제 복사 버튼 금지**
- `HARD_CONFIRMED`에서만 “내일 OFF10 주문 초안 생성/복사” 허용
- Hard 해제 직후 1일은 `cooldownActive=true`로 OFF10 유지(쿨다운 배지 표기)

---

## 4) 사용자 핵심 동선(User Flows) — 60초/120초 목표

### 4.1 Daily Ops (매일 아침, 60초)
목표: “오늘 판정/내일 실행” 확인 + 필요 시 준비.

1) Zone A에서 `Data State` 확인(FRESH/STALE/휴장)
2) Zone B에서 `오늘 Verdict` 확인(ON/ON_CHOPPY/OFF10/EMERGENCY) + 근거
3) `Execution State = SCHEDULED`면 Zone C에서 **내일 주문 초안** 확인/복사(준비)
4) 실행일이 아니라면 종료(정보 소비는 선택)

### 4.2 Execution Day (실행일, 120초)
목표: 실수 없는 주문 + 기록 누락 방지.

1) Zone A: “오늘은 Execution Day”임을 1초 내 인지
2) Zone C: 주문 초안 확인 → 복사 → 삼성증권에서 주문
3) 복귀 후 Zone C의 **최소 기록 폼**으로 “체결 수량(필수)”만 입력하여 `RECORDED`로 전환
4) 기록이 어려우면 `UNKNOWN`으로 남기되, “나중에 보정 큐”로 저장

### 4.3 Emergency (비상, 60~180초)
목표: 패닉 매매 방지 + 규칙 준수.

- **SOFT_ALERT(장중)**
  1) 경보 노출(“확정은 종가 이후” 문구 필수)
  2) 행동은 “OFF10 준비(계산/체크리스트)”까지만
  3) 주문 복사는 잠금

- **HARD_CONFIRMED(종가 확정)**
  1) “내일 OFF10 실행 준비”가 명확히 표시
  2) Zone C에서 OFF10 주문 초안 생성/복사 허용
  3) 실행일에 기록까지 완료하도록 리마인드 큐 등록

- **ON_CHOPPY(필터 발동)**
  1) Zone B에서 FlipCount 게이지와 40일 타임라인으로 불안정 상태를 설명
  2) Zone C에서 70/30 배분 배지와 주문 컨텍스트를 동시에 노출
  3) 사용자는 규칙 변경 없이 비중 축소 상태임을 즉시 인지

### 4.4 신규 자금/분배금(선택)
- 입력 시, 목표 비중 대비 **부족 자산 우선**으로 배분 초안을 생성
- 정수주/최소주문 제약으로 남는 잔액은 “현금/SGOV 유지”로 처리(명확히 표기)

### 4.5 데이터 누락/휴장
- `STALE`일 때:
  - 신호는 “전일 유지(보수적)” 또는 “계산 보류(잠금)” 중 하나로 SSOT에서 고정(권장: 전일 유지 + 강한 경고)
  - 주문 생성/복사는 기본 잠금
- 휴장/조기폐장일 때:
  - Verdict/Execution 날짜 표기를 자동 보정(사용자 혼동 방지)

---

## 5) Zone별 기능‑UX 결합 명세 (Module Spec)

### 5.1 Zone A — Global Watchdog (상단, 항상 보이는 상태 요약)
**목적**: 사용자가 스크롤/탐색 없이도 “오늘 해야 할 일”을 1초 내 파악.

**반드시 제공해야 하는 ‘상태 요약 3종’**
1) `Data State`: 기준(종가/장중), 마지막 업데이트 시각
2) `Emergency State`: NONE / SOFT / HARD_CONFIRMED
3) `Execution State`: NO_ACTION / SCHEDULED / DUE_TODAY / RECORDED / UNKNOWN

**가드레일**
- `STALE` 또는 `SOFT_ALERT` 상태에서는 “주문 복사”를 유도하는 문구/동작 금지
- `DUE_TODAY` + `UNKNOWN`이면 “기록 누락”을 최우선으로 상기(브로커 연동이 없어도 UX로 해결해야 함)

---

### 5.2 Zone B — Signal Core (판정과 근거, ‘확정 vs 가정’ 분리)
**목적**: 오늘 Verdict와 내일 Execution을 “한 문장”으로 이해.

**B0. Signal Summary (필수)**
- 출력:
  - `State`: ON / ON_CHOPPY / OFF10 / EMERGENCY
  - `Verdict Date`: (QQQ Close 기반)
  - `Execution Date`: (t+1 기준)
  - `Basis`: SMA3 vs SMA160/165/170 + 2/3 투표 결과

**B0-2. FlipCount Interaction (필수)**
- 접힘/펼침 토글 제공
- ON_CHOPPY일 때 기본 펼침
- 텍스트 기준: "40일 시그널 히스토리 · N회 전환"

**B1. Ensemble Details (필수)**
- 각 window별:
  - TRUE/FALSE
  - Margin(%)
  - 참고값(선택): SMA3, SMAwindow

**B2. Simulator (선택이지만 강추)**
- 목적: 예측이 아니라 **시나리오 대비**
- 입력: “오늘 종가가 Δ%라면?”
- 출력: 가정 기반 내일 신호(“SIMULATION”으로 명시)

**Simulator 가드레일(필수)**
- 시뮬레이션 활성화 상태에서는:
  - Zone C 주문 생성/복사 **잠금**
  - Zone B 상단에 “확정 신호 아님(종가 이후 확정)” 문구 **고정**
  - 사용자가 ‘실수로’ 시뮬 상태를 확정으로 오인할 수 없게 한다.

---

### 5.3 Zone C — Ops Console (실행·기록의 중심, 연동 부재를 UX로 보완)
**목적**: “내일/오늘” 실행을 고민 없이 완료시키고, 기록 누락을 최소화.

**C0. 입력(필수, 최소화)**
- 필수 입력(Quick Mode):
  - `총 평가액(원)` 또는 `투입 금액(원)` (프로덕트 정책에 따라 1개로 통일 권장)
- 선택 입력(정확도 개선):
  - `현재 보유 수량`(TQQQ/SGOV)
  - `TQQQ 평균단가`(비상 트리거 -15% 계산에 필요)
  - `신규 입금/분배금`

**입력 UX 최적화(기능 요구)**
- 마지막 입력값 자동 채움(연동 불가의 대체)
- 오입력 방지: 한글 독음/자리수 확인 등은 “기능적 안전장치”로 간주(디자인 제외)

**C1. 주문 초안 생성/출력(필수)**
- 출력은 반드시 구분:
  - “예상 주문(자동, 전략 진실)” vs “실제 체결(수동, 현실 진실)”
- `Execution Date`가 없는 날은 “주문 없음(No Action)”을 명확히.

**C2. Copy (필수)**
- 복사 데이터는 MTS 입력 호환을 위해 “숫자만” 복사 옵션 제공
- 복사 직후 “무엇이 복사되었는지(티커/수량)” 확인을 UX에 포함(실수 방지)

**C3. Execution Confirmation (강추, 연동 부재의 핵심 보완)**
- 실행일(또는 사용자가 체크한 날)에 최소 기록 폼을 노출:
  - 필수: `체결 수량`(종목별)
  - 선택: 체결가/수수료/메모
- 기록이 없으면 `UNKNOWN`으로 남기되, “나중에 보정” 큐로 쌓는다.

---

### 5.4 Zone D — Intel Lab (정보/회고/시뮬, ‘의도적 진입’)
**목적**: 규칙을 흔들지 않으면서도 사용자의 정보 욕구를 해소.

**운영 원칙**
- Zone D는 판단/실행을 방해하지 않도록 “의도적 진입”을 요구한다(기본 접힘/잠금 등 디자인은 별도 결정).
- AI Briefing은 “팩트 기반 요약”만 허용(추천/예측/선동 금지).

**권장 구성(기능 레벨)**
- AI Briefing(팩트/출처 포함, 실패 시 폴백)
- Compliance/기록 상태(UNKNOWN을 정상으로 취급)
- 회고 지표(Underwater, 월간 성과 등)
- 시뮬/리플레이는 반드시 “가정”임을 명시하고 Zone C 실행과 분리

---

## 6) 기록 시스템(Manual Ledger) — 연동 없는 환경의 ‘진짜 엔진’

### 6.1 데이터 엔티티(권장 최소 모델)
- `HoldingsSnapshot`(선택): 날짜, 종목, 수량, 평균단가(선택)
- `TradeLog`(필수): 날짜, 종목, 매수/매도, 수량(필수), 체결가(선택), 수수료/세금(선택), 메모(선택)
- `OpsChecklist`(권장): Execution Day 체크(주문 완료/기록 완료)

### 6.2 UX 정책
- 기록은 “완벽”이 아니라 “지속”이 목표:
  - 최소 입력만으로도 `RECORDED`가 되게 한다(체결 수량만 필수)
- 수정/보정은 허용하되:
  - 변경 이력(누가/언제/무엇)을 최소한으로 남긴다(감사 가능성)

---

## 7) 알림(Notifications) — 연동 없이도 운영 품질을 좌우

### 7.1 트리거(필수)
- Verdict 변경(ON↔OFF10)
- Verdict 변경(ON↔ON_CHOPPY↔OFF10↔EMERGENCY)
- Execution 스케줄 생성(SCHEDULED)
- Execution Day(DUE_TODAY)
- 기록 누락(UNKNOWN 지속)
- Emergency: SOFT_ALERT / HARD_CONFIRMED

### 7.2 UX 정책
- 알림은 “확정/비확정” 문구를 반드시 분리한다.
- 중복 알림은 억제(쿨다운/확인 배지)하여 노이즈가 되지 않게 한다.
- 알림은 사용자를 ‘앱으로 호출’하는 용도이지, 매매 판단 근거가 아니다.

---

## 8) 시뮬/리플레이 UX 정책(강제 안전장치)

- 시뮬 결과는 절대 “주문 생성/복사”로 바로 연결되지 않는다.
- 시뮬은 Zone D 또는 Zone B의 ‘가정 모드’에서만 제공하되,
  - **가정 상태 표시**
  - **확정 신호와 분리**
  - **실행 기능 잠금**
  을 SSOT 레벨에서 고정한다.

---

## 9) 완료 조건(Acceptance Criteria)

### 9.1 Daily Ops (60초)
- 사용자는 3번 클릭/입력 이내로 다음을 확인할 수 있어야 한다.
  - 데이터 최신 여부
  - 오늘 Verdict
  - 내일 Execution 유무

### 9.2 Execution Day (120초)
- 주문 복사 → 삼성증권 주문 → 최소 기록(수량)까지 완료 가능해야 한다.
- 기록 누락 시 시스템이 `UNKNOWN`을 명확히 표시하고 “보정 큐”로 관리한다.

### 9.3 Fail‑safe
- `STALE` 데이터, 시뮬레이션 모드, SOFT_ALERT에서는 잘못된 주문 복사가 발생하지 않는다.

---

## 10) 디자인 범위 외(Out of Scope for this SSOT)
- 색상, 타이포, 그리드/레이아웃 수치, 애니메이션, 아이콘, 컴포넌트 스타일은 별도 UI SSOT에서 결정한다.
