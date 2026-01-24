# E03 Command Center — UI Design SSOT v1.1

- Document Status: SSOT_UI_V1.1
- Target Strategy: E03 (Ensemble Vote + OFF10 + SGOV)
- Audience: Frontend Engineer / Publisher
- Authoring Role: Senior UI Engineer
- Timezone: KST 기본 표기 (옵션: ET 병기)
- Last Updated: 2026-01-19 (KST)

---

## 0) Authority & Non-Goals

### 0.1 Authority (상위 권위)
UI는 아래 SSOT 정의를 변경할 수 없다.
- Strategy Authority: `E03_SSOT.md`
- Functional Authority: `E03_Command_Center_SSOT_v2.md`
- UX Authority: `E03_UX_SSOT_v3.md`

### 0.2 Non-Goals (비목표)
- 브로커(삼성증권) 자동 연동/자동 매매 없음
- 뉴스/차트/의견 기반 매수·매도 추천, 예측, 선동성 정보 제공 없음
- 전략 룰 임의 수정 UI 없음(룰 변경은 “버전 관리 산출물”로만 허용)

---

## 1) Design Philosophy

### 1.1 Enforced Linearity (강제된 선형성)
사용자가 다음 인지 순서를 거스를 수 없도록 **구조와 시각 위계로 강제**한다.

1) 상태 확인 (Zone A: 데이터 신뢰 / 비상 / 실행 필요)
2) 판정 (Zone B: 오늘 Verdict, 종가 기반 확정)
3) 실행 (Zone C: 내일/오늘 주문 초안 + 기록)
4) 정보 (Zone D: 회고/시뮬/브리핑 — 선택)

### 1.2 Rule is King + Two Truths Model
- 전략 진실(자동): 신호/목표 비중/예상 주문(해야 하는 것)
- 현실 진실(수동): 실제 체결/실제 보유/실행 여부(실제로 한 것)
- 두 진실은 **레이아웃, 라벨, 데이터 모델, 문구**에서 절대 섞지 않는다.

### 1.3 Execution Lag 강제 (t일 판정 → t+1 실행)
모든 핵심 출력에 **Verdict Date(판정일) + Execution Date(실행일)**를 항상 동시 표기한다.
- 사용자가 “오늘 신호를 오늘 실행”이라고 오해할 여지를 0으로 만든다.

### 1.4 Fail-safe UX
데이터 결측/지연, 휴장/조기폐장, 입력 누락, 시뮬레이션 상태에서:
- 잘못된 주문 생성/복사 방지
- 항상 “잠금(LOCK) + 사유(WHY) + 최소 행동(기록/재시도)”로 수렴

---

## 2) Lexicon (용어/표기 고정)

### 2.1 상태명 표기(고정)
- ON
- OFF10  ← 공백 금지(OFF 10 금지)

### 2.2 앙상블 비교식(고정)
- 비교는 “현재가 vs MA”가 아니라 **SMA(3) vs SMA(window)**.
- Vote(window) = (SMA(3) > SMA(window))
- windows = {160, 165, 170}
- 2/3 이상 true → ON, else → OFF10

### 2.3 시간/날짜 표기(고정)
- 기본: KST
- 권장 템플릿
  - 판정일(Verdict): YYYY-MM-DD (요일) HH:MM KST
  - 실행일(Execution): YYYY-MM-DD (요일) (t+1)

---

## 3) Layout System (구조)

### 3.1 Desktop IA (Zone A→B→C→D)
- Vertical Stack 구조(상→하 흐름 우선)
- Zone B(판정)는 Zone C(실행)보다 항상 위
- Zone D(정보)는 기본 접힘(Accordion/Drawer)로 격리

### 3.2 Grid / Container
- Container: max-width 1200px, center aligned
- Grid: 12 columns, gutter 24px

### 3.3 Persistent Areas
| Area | Spec | Rationale |
|---|---|---|
| Zone A Header | Sticky Top, height 64px | 데이터/비상/실행 상태를 진입 즉시 인지 |
| LNB | Left fixed, width 64px (icon bar) | 단일 목적 도구, 메인 작업 공간 확보 |
| Zone D | Bottom collapsed | 판단/실행 후 정보 소비를 강제 |

---

## 4) State Model → UI Mapping (필수)

UI는 아래 4종 상태를 반드시 분리 표현한다.

### 4.1 Data State (데이터 신뢰)
- FRESH_CLOSE: 종가 기반 데이터 최신, 신호 계산 가능
- STALE: 업데이트 지연/결측
- MARKET_CLOSED: 휴장/조기폐장 등

UI 규칙
- STALE/MARKET_CLOSED는 반드시 “사유”를 표시한다.
- STALE에서는 기본적으로 주문 생성/복사 LOCK.
- 제품 정책이 “전일 신호 유지”일 경우, 표시 문구를 고정:
  - “데이터 지연: 전일 신호 유지 중(보수적)”

### 4.2 Strategy State (전략 판정)
- ON / OFF10

### 4.3 Execution State (실행 상태)
- NO_ACTION: 목표 비중 변화 없음(주문 없음)
- SCHEDULED: 내일 실행 필요(Execution Date 존재)
- DUE_TODAY: 오늘이 Execution Date
- RECORDED: 실행/체결 기록 완료
- UNKNOWN: 실행 여부 미확인(기록 누락/미입력)

우선순위 규칙
- DUE_TODAY면 Zone C가 화면 최강 CTA를 가져야 한다.
- DUE_TODAY + UNKNOWN은 헤더와 Zone C에서 “기록 누락”을 최우선 상기한다.

### 4.4 Emergency State (비상 상태)
- NONE
- SOFT_ALERT: 장중 조기 경보(자동 전환 아님)
- HARD_CONFIRMED: 종가 확정 트리거(내일 OFF10 실행 준비)

UI 규칙(핵심)
- SOFT_ALERT에서는 “준비”만 허용:
  - Copy 버튼 Disabled(주문 복사 유도 금지)
  - “확정은 종가 이후” 고정 문구 표시
- HARD_CONFIRMED에서만 OFF10 주문 초안 생성/복사 허용

---

## 5) Zone Specs (구역별 UI 명세)

## 5.1 Zone A — Global Watchdog (Sticky Header)
목적: 스크롤 없이도 “지금 해야 할 일”을 1초 내 인지.

필수 노출(항상)
1) Data Badge
- Data Basis: QQQ Close
- Updated: YYYY-MM-DD HH:MM KST
- Signal Basis: SMA3 vs SMA160/165/170
- 상태: FRESH / STALE / MARKET_CLOSED (+ 사유)

2) Emergency Badge
- NONE / SOFT_ALERT / HARD_CONFIRMED
- SOFT_ALERT 시 “확정은 종가 이후” 보조 텍스트

3) Execution Badge
- NO_ACTION / SCHEDULED / DUE_TODAY / RECORDED / UNKNOWN
- DUE_TODAY는 가장 강한 강조(시각적 경고)

권장 옵션
- Privacy Toggle(금액/수량 마스킹)
- Simulation Indicator(활성 시 항상 노출)

---

## 5.2 Zone B — Signal Core (판정 Hero)
목적: 오늘 Verdict와 내일 Execution을 “한 문장”으로 이해.

B0. Hero Summary (필수)
- Verdict (가장 큰 타이포): ON 또는 OFF10
- Subline:
  - 판정일(Verdict Date)
  - 실행일(Execution Date, t+1)
- 상태 문구(상황별):
  - NO_ACTION: “주문 없음”
  - SCHEDULED: “내일 실행 준비”
  - DUE_TODAY: “오늘 실행일”
  - STALE: “데이터 지연: 전일 신호 유지 중(보수적)”

B1. Evidence Grid (필수)
- 3개 카드: SMA160 / SMA165 / SMA170
- 각 카드 필드:
  - Vote: ✅ / ❌  (SMA3 > SMAwindow 여부)
  - Margin(%): SMA3 대비 SMAwindow 괴리(양/음)
  - Gauge: 0% 기준 좌(음수)/우(양수)로 뻗는 막대(직관 인지)

B2. Guardrail Copy (필수)
- “비상 경보(SOFT_ALERT)는 확정이 아니며, 확정은 종가 이후입니다.”
- “주문 실행은 t+1일(실행일)에 합니다.”

---

## 5.3 Zone C — Ops Console (실행 + 기록)
목적: 실행 고민 제거 + 기록 누락 최소화(연동 부재를 UX로 보완).

레이아웃(권장)
- Split: 좌 4 (Input) : 우 8 (Order Ticket)

C0. Input Panel (Reality, 수동 입력)
필수 입력(정책적으로 1개로 통일 권장)
- 총 평가액(원)  ← 권장 기본

선택 입력(정확도/비상 계산)
- 현재 보유 수량: TQQQ, SGOV
- TQQQ 평균단가(비상 -20% 트리거 계산용)
- 신규 입금/분배금(선택)

Auto-fill UX
- 이전 기록 자동 채움 시 “자동 채움” 표시(사용자 확인 필요)
- 자동 채움 필드 배경 강조(연한 하이라이트)

C1. Order Ticket (Strategy, 자동 산출)
표기 원칙(고정)
- 상단 라벨: “예상 주문(자동 · 전략 진실)”
- Execution Date가 없으면:
  - “오늘 주문 없음(NO_ACTION)” 상태 카드로 대체(복사 버튼 숨김 또는 disabled)

주문 리스트 표시(실수 방지)
- 종목명보다 **Qty를 크게** 표시(모노스페이스 권장)
- OFF10 계산 규칙 각주(작게 고정):
  - “OFF10 잔류 수량은 10%를 올림(ceiling)”
  - “SGOV 불가 시 SHV로 대체”
  - “(선택) 0.2% 현금 버퍼 반영”

C2. Copy (Primary Action)
- 버튼 라벨: “주문 복사”
- 옵션 토글: “숫자만 복사”
- 복사 완료 토스트(필수):
  - “복사 완료: TQQQ -123, SGOV +320”처럼 ‘무엇이 복사됐는지’ 표시
- Disabled 조건(필수)
  - Data State = STALE/MARKET_CLOSED
  - Emergency State = SOFT_ALERT
  - Simulation Mode = ON
  - (선택) HARD_LOCK(수동 잠금) 활성화

C3. Execution Log (Confirm, 현실 진실)
표기 원칙(고정)
- 섹션 라벨: “실제 체결(수동 · 현실 진실)”
- DUE_TODAY 또는 UNKNOWN일 때 우선 노출(CTA 승격)

최소 기록(권장)
- 필수: 종목별 체결 수량
- 선택: 체결가/수수료/세금/메모
- 저장 완료 시 Execution State → RECORDED

UNKNOWN 처리
- 실패가 아니라 “연동 부재의 정상 상태”로 안내
- “나중에 보정” 큐로 임시 저장 지원(선택 기능)

---

## 5.4 Zone D — Intel Lab (정보/회고, 기본 접힘)
목적: 규칙을 흔들지 않고 정보 욕구를 해소(의도적 진입).

원칙(고정)
- 기본 접힘(Accordion/Drawer)
- 판단/실행 전에 정보가 전면에 나오지 않음

허용 콘텐츠(권장)
- 팩트 기반 요약(가능하면 출처/근거 표기)
- 회고 지표(규율/기록 완성도/일탈 비용 등)
- 시뮬/리플레이는 반드시 “가정”임을 명시하고 실행과 분리

금지
- 매수·매도 추천, 예측, 선동성 문구

---

## 6) Global Modes (전역 모드)

### 6.1 Simulation Mode (Safety UI)
목적: 사용자가 “가정”을 “확정”으로 오인하지 못하게 강제.

필수 UI
- 화면 테두리: 4px dashed orange border
- 워터마크: “SIMULATION MODE” 반복 패턴
- Zone A에 “SIMULATION” 배지 상시 노출

필수 가드레일
- 주문 생성/복사 LOCK
- Zone B 상단에 “확정 신호 아님(종가 이후 확정)” 고정

### 6.2 Privacy Mode (권장)
- 금액/수량을 마스킹(예: *** 또는 약식 표기)
- 스크린샷/화면공유 환경에서 안전

---

## 7) Component Detail (구현 레벨)

### 7.1 Badges (Zone A)
- Pill 형태(텍스트만 금지)
- Data:
  - FRESH: 녹색 점 + “데이터 최신”
  - STALE: 빨간 점 + “데이터 지연(소스 확인)”
  - MARKET_CLOSED: 회색 점 + “휴장/조기폐장”
- Emergency:
  - SOFT_ALERT: “비상 경보(장중)”
  - HARD_CONFIRMED: “비상 확정(종가)”
- Execution:
  - DUE_TODAY: 채워진 배경(강 경고)
  - SCHEDULED: 아웃라인(준비 상태)
  - UNKNOWN: “기록 필요” 라벨 동반

### 7.2 Vote Gauge (Zone B)
- 0% 기준 좌/우로 확장(음수/양수 직관)
- 각 카드에 Margin% 텍스트 + 게이지 동시 제공(텍스트 단독 금지)

### 7.3 Copy Button (Primary Action)
- Default: Primary filled
- Hover: 미세한 강조(밝기/그림자)
- Active: “Copied!” 상태로 1.5초(또는 토스트로 대체)
- Success 상태(권장): 버튼 색상/아이콘 체크로 단기 피드백
- Disabled: opacity + not-allowed + 툴팁(사유 고정)
  - 예: “데이터 업데이트가 필요합니다.” / “비상 경보 중에는 복사할 수 없습니다.” / “시뮬레이션 모드에서는 복사할 수 없습니다.”

### 7.4 Inputs (Auto-fill UX)
- 높이 56px 권장(터치/가독)
- Auto-filled: 연한 하이라이트 + “자동 채움” 아이콘/라벨
- Focus: 강조(바닥선 2px 등)

---

## 8) Copy Dictionary (문구 고정 — 한국어 중심)

반드시 고정(의미 변형 금지)
- “OFF10” (공백 금지)
- “판정일(Verdict)” / “실행일(Execution)”
- “예상 주문(자동 · 전략 진실)” / “실제 체결(수동 · 현실 진실)”
- “확정은 종가 이후입니다.”
- “주문 없음(NO_ACTION)”
- “기록 필요(UNKNOWN)”

토스트/툴팁 템플릿(권장)
- Disabled(데이터): “데이터 지연으로 주문 복사가 잠겼습니다. 업데이트 후 다시 시도하세요.”
- Disabled(비상 경보): “비상 경보는 확정이 아닙니다. 종가 확정 후에 주문이 활성화됩니다.”
- Disabled(시뮬): “시뮬레이션 모드에서는 주문 복사가 잠겼습니다.”

---

## 9) Accessibility & QA Gates

### 9.1 A11y
- 키보드 포커스 순서: Zone A → Zone B → Zone C → Zone D
- 버튼/입력의 포커스 링 제공
- 배지는 색상만으로 의미 전달 금지(텍스트 동반)
- 숫자 입력은 스크린리더 라벨(단위 포함: 원/주)

### 9.2 Error-proofing (필수)
- Copy 토스트에 “복사된 내용(티커/수량)”을 반드시 노출
- DUE_TODAY일 때 Zone C “기록” CTA가 항상 화면에서 강하게 보일 것
- STALE/SOFT_ALERT/SIMULATION에서 Copy는 반드시 잠길 것

---

## 10) ASCII Wireframe (Reference)

+-----------------------------------------------------------------------+
| [Zone A: Header] (Sticky)                                             |
| Data: ● FRESH (06:30 KST)  Basis: SMA3 vs 160/165/170                 |
| Emergency: NONE | Execution: SCHEDULED (Exec: 01.20)                  |
+-----------------------------------------------------------------------+
| [LNB Icon Bar]                                                        |
|                                                                       |
| [Zone B: Signal Core]                                                 |
| +-------------------------------------------------------------------+ |
| | VERDICT: OFF10                                                    | |
| | 판정일: 01.19 (KST 06:45)   실행일: 01.20 (t+1)                   | |
| |                                                                   | |
| | [Evidence]  MA160 [✅] +3.2% [<-0%->]                             | |
| |            MA165 [✅] +1.5% [<-0%->]                             | |
| |            MA170 [❌] -0.2% [<-0%->]                             | |
| +-------------------------------------------------------------------+ |
|                                                                       |
| [Zone C: Ops Console]                                                 |
| +-------------------------+  +--------------------------------------+ |
| | Input (Reality)         |  | Order Ticket (Strategy)              | |
| | Total Asset (KRW)       |  | 예상 주문(자동 · 전략 진실)          | |
| | [ 150,000,000 ]         |  | [SELL] TQQQ  1,200 ea                | |
| | (Auto-filled)           |  | [BUY ] SGOV    320 ea                | |
| | TQQQ Qty (opt)          |  |                                      | |
| | [ 1,215 ]               |  | [주문 복사] (숫자만 복사 토글)       | |
| +-------------------------+  +--------------------------------------+ |
|                               +-------------------------------------+|
|                               | Execution Log (Reality)             ||
|                               | 실제 체결(수동 · 현실 진실)         ||
|                               | [ ] All orders executed             ||
|                               | Final Qty: [____] [SAVE]            ||
|                               +-------------------------------------+|
|                                                                       |
| [Zone D: Intel Lab] (Collapsed) v                                     |
+-----------------------------------------------------------------------+

---

## 11) Implementation Checklist (PR 리뷰용)
- [ ] 용어/토큰: OFF10 공백 금지
- [ ] Evidence: SMA3 vs SMA160/165/170 표기/툴팁 고정
- [ ] Zone A: Data/Emergency/Execution 3종 배지 항상 노출
- [ ] SOFT_ALERT에서 Copy 잠금, HARD_CONFIRMED에서만 OFF10 주문 복사 허용
- [ ] STALE/MARKET_CLOSED/Simulation에서 주문 생성·복사 잠금 + 사유 메시지
- [ ] NO_ACTION일 때 “주문 없음” 명확히(복사/주문 UI 비활성)
- [ ] Two Truths 라벨: 예상 주문(자동) vs 실제 체결(수동) 영구 분리
- [ ] 복사 완료 토스트에 “무엇이 복사됐는지(티커/수량)” 표시
- [ ] 실행일(DUE_TODAY)+UNKNOWN이면 기록 CTA 승격(헤더+Zone C 동시)
