E03 Command Center — Design SSOT v1.0 (Web + Mobile, Light/Dark)

Status: DESIGN_SSOT_V1.0

Authority(변경 불가): Strategy / Functional / UX SSOT를 UI가 침범하지 않는다. 

E03_UI_SSOT

핵심 UX 불변식(Non-Negotiables)

Rule is King, Decision→Execution→Information, Two Truths, t→t+1 Execution Lag, Fail-safe UX 

E03_UX_SSOT

Zone 구조(Zone A→B→C→D)와 Zone D 격리는 절대 유지 

E03_Command_Center_SSOT_v2

0) 디자인 목표와 비목표
디자인 목표

1초 인지: Zone A에서 데이터/비상/실행 상태를 즉시 파악 

E03_UX_SSOT

실행 실수 방지: 잠금(LOCK) + 사유(WHY) + 최소 행동으로 수렴 

E03_UI_SSOT

정보는 뒤로: Zone D는 “의도적 진입”만 허용(기본 접힘) 

E03_UI_SSOT

비목표(디자인이 하지 말아야 할 것)

자동매매/브로커 연동처럼 보이게 만드는 UI 금지 

E03_Command_Center_SSOT_v2

뉴스/차트/의견 기반 추천·예측·선동성 노출 금지 

E03_UI_SSOT

룰 임의 수정 UI 금지(버전 관리 산출물로만) 

E03_UI_SSOT

1) 크로스플랫폼 레이아웃 SSOT (Web / Mobile)
1.1 Zone 순서(절대)

Web/Desktop: Zone A(Sticky) → Zone B → Zone C → Zone D(기본 접힘) 

E03_UI_SSOT

Mobile: Zone A(고정 바) → Zone B → Zone C → Zone D(기본 접힘) 

E03_Command_Center_SSOT_v2

1.2 Web Grid / Container

Container: max-width: 1200px, center 

E03_UI_SSOT

Grid: 12 columns, gutter 24px 

E03_UI_SSOT

Zone A Header: sticky top, 64px 

E03_UI_SSOT

1.3 Global Header & Drawer Navigation (v2.0)

브랜드: "200TQ α" (모노스페이스, α는 cyan 강조)

로고 아이콘: Gradient badge (cyan→blue) + α 심볼

Header Bar (항상 표시):
- height: 56px, fixed top, z-50
- 왼쪽: 햄버거 버튼 (드로어 토글)
- 로고: "200TQ α" (클릭 시 /command로 이동)

Drawer Navigation (기본 숨김):
- 너비: 256px, 왼쪽 슬라이드
- Backdrop 오버레이

Navigation Items (5개 고정):

| 순서 | 이름       | 경로        | 용도                    |
|------|------------|-------------|-------------------------|
| 1    | Command    | /command    | 일일 의사결정 (메인)     |
| 2    | Portfolio  | /portfolio  | 포트폴리오 현황          |
| 3    | Records    | /records    | 실행 기록 조회           |
| 4    | Analysis   | /analysis   | 성과 분석 & 백테스트     |
| 5    | Settings   | /settings   | 시스템 설정              |

Mobile/Web 통일: 동일 드로어 구조 (Bottom Tab Bar 미사용)


2) Design Tokens SSOT (Light/Dark + Web/Mobile 공통)

원칙: “색상/폰트/간격”은 컴포넌트 내부에서 하드코딩 금지, 모두 토큰으로만 사용.

2.1 Typography (KR/EN 혼용 최적)

Font Stack

KR: Pretendard Variable → Apple SD Gothic Neo → system

EN/Number: Inter Variable → system

Number 규칙

금액/수량/퍼센트는 font-variant-numeric: tabular-nums; 고정

Type Scale (Mobile / Web 공통)

Display: 32/40 (Zone B Verdict “ON/OFF10”)

H1: 24/32

H2: 20/28

Body: 16/24

Body2: 14/20

Caption: 12/16

Micro: 11/14 (보조 라벨/각주)

2.2 Spacing & Layout

Base: 8pt grid

Spacing tokens: 4, 8, 12, 16, 20, 24, 32, 40, 48

Card padding: 16(모바일)/20(웹)

Section gap(Zone B↔C 등): 24(모바일)/32(웹)

2.3 Radius, Border, Shadow

Radius:

Card: 16

Input/Button: 12

Badge(Pill): 999

Modal/Drawer: 20

Border width: 1 (기본), 2(집중 상태/강조 링)

Elevation(Shadow):

E1(카드): 얕은 그림자

E2(팝오버/툴팁)

E3(모달)

2.4 Motion

Duration: 120 / 200 / 320ms

Easing: standard ease-out, 강조(경고)만 spring-like 허용

“비상” 애니메이션은 1초 이내, 반복은 최소(피로도 방지)

3) Color System SSOT (라이트/다크)

“트레이딩 앱” 레퍼런스 이미지의 공통 인상(강한 Primary 블루 + 뉴트럴 기반 + 절제된 강조)을 유지하되, 의미 색은 상태 모델에 종속.

3.1 Core Palette (권장 HEX)

Primary Blue

primary-600: #0A84FF (주 CTA, iOS 계열 감성)

primary-700: #0066CC (hover/pressed)

Neutrals (Light)

bg: #F6F7F9

surface: #FFFFFF

text-1: #0B1220

text-2: #5B667A

border: #E6EAF0

Neutrals (Dark)

bg: #0B0F14

surface: #111827

text-1: #EAF0FF

text-2: #A9B4C7

border: #243044

Semantic

positive: #16A34A

negative: #DC2626

warning: #F59E0B

info: #2563EB

3.2 Semantic Tokens (CSS Variables 권장)

--bg, --surface, --surface-2, --border

--text-1, --text-2, --text-3

--primary, --primary-hover, --primary-pressed

--success, --danger, --warn, --info

--focus-ring (primary 기반)

--shadow-e1/e2/e3 (모드별 별도 값)

4) State Model → Visual Mapping (가장 중요한 디자인 규칙)

UI는 아래 상태를 절대 섞지 않고, 상태별로 색/강조/잠금을 고정한다. 

E03_UI_SSOT

4.1 전역 상태 배지(Zone A: 항상 보임)

Zone A는 “상태 요약 3종(Data/Emergency/Execution)”을 고정 표시해야 한다. 

E03_UX_SSOT

Badge 스타일(공통)

Pill(높이 28~32), 좌측 Dot(6px), 텍스트 + 짧은 서브텍스트(옵션)

색상만으로 의미 전달 금지(텍스트 필수) 

E03_UI_SSOT

Data State

FRESH_CLOSE: Dot=Success, label=“데이터 최신”

STALE: Dot=Danger, label=“데이터 지연”, 사유(WHY) 필수 

E03_UI_SSOT

MARKET_CLOSED: Dot=Neutral, label=“휴장/조기폐장”

Emergency State

NONE: Neutral

SOFT_ALERT: Warning, 보조 문구 “확정은 종가 이후입니다.” 

E03_UI_SSOT

HARD_CONFIRMED: Danger, 보조 문구 “내일 OFF10 실행 준비”

Execution State

NO_ACTION: Neutral

SCHEDULED: Info(아웃라인)

DUE_TODAY: Danger(채움 배경) + 가장 강한 강조 

E03_UI_SSOT

RECORDED: Success

UNKNOWN: Warning + “기록 필요” 라벨 동반 

E03_UI_SSOT

4.2 “주문 복사(Primary CTA)” 잠금 규칙(절대)

아래 조건이면 Copy 버튼은 Disabled + Tooltip(사유 고정) 처리. 

E03_UI_SSOT

Data = STALE 또는 MARKET_CLOSED

Emergency = SOFT_ALERT

Simulation Mode = ON

(선택) HARD_LOCK(수동 잠금)

4.3 Simulation Mode (안전장치 UI)

화면 테두리: 4px dashed(경고색)

워터마크: “SIMULATION MODE”

Zone A에 “SIMULATION” 배지 상시 노출

실행 기능(주문 생성/복사) 잠금 

E03_UI_SSOT

5) Component SSOT (구현 단위)
5.1 Buttons

Sizes:

Primary CTA: height 52~56, radius 12, label 16/semibold

Secondary: height 44~48

Variants:

Primary Filled(Blue)

Secondary Outline

Tertiary Ghost

Destructive(빨강)은 “삭제/계정” 같은 비업무에만 제한

Disabled:

opacity 40~50% + cursor not-allowed + tooltip(WHY) 

E03_UI_SSOT

5.2 Cards (Zone B/C 기본 컨테이너)

Card radius 16, padding 16/20

헤더(Title+Meta)와 바디(내용) 구분

“전략 진실(자동)” 카드와 “현실 진실(수동)” 카드는 라벨/아이콘/배경 톤을 다르게(혼동 방지) 

E03_UX_SSOT

5.3 Badges (Zone A 핵심)

Pill 형태 강제 

E03_UI_SSOT

Data/Emergency/Execution 각각 색+텍스트 고정(위 4.1)

5.4 Inputs

Height 56 권장(터치/가독) 

E03_UI_SSOT

Auto-filled 상태: 연한 하이라이트 + “자동 채움” 라벨 

E03_UI_SSOT

숫자 입력:

우측 단위(원/주/%)

천 단위 구분 표시

에러는 즉시(Inline) + 요약(Toast 금지)

5.5 Evidence Vote Cards (Zone B)

3개 카드(SMA160/165/170) 고정 

E03_UI_SSOT

카드 구성:

Vote(✅/❌)

Margin(%)

Gauge(0% 기준 좌/우 확장) — 텍스트 단독 금지 

E03_UI_SSOT

5.6 Order Ticket (Zone C)

상단 라벨 고정: “예상 주문(자동 · 전략 진실)” 

E03_UI_SSOT

수량(Qty)을 종목명보다 크게(실수 방지) 

E03_UI_SSOT

Copy 토스트에 “복사된 티커/수량” 반드시 노출 

E03_UI_SSOT

5.7 Execution Log (Zone C)

라벨 고정: “실제 체결(수동 · 현실 진실)” 

E03_UI_SSOT

DUE_TODAY 또는 UNKNOWN이면 이 섹션 CTA 승격 

E03_UI_SSOT

6) Copy & Microcopy Dictionary (한국어 고정)
6.1 절대 고정 용어

“OFF10” (공백 금지) 

E03_UI_SSOT

“판정일(Verdict)” / “실행일(Execution)” 

E03_UI_SSOT

“예상 주문(자동 · 전략 진실)” / “실제 체결(수동 · 현실 진실)” 

E03_UI_SSOT

“확정은 종가 이후입니다.” 

E03_UI_SSOT

“주문 없음(NO_ACTION)” / “기록 필요(UNKNOWN)” 

E03_UI_SSOT

6.2 Disabled Tooltip 템플릿(권장)

데이터: “데이터 지연으로 주문 복사가 잠겼습니다. 업데이트 후 다시 시도하세요.” 

E03_UI_SSOT

비상 경보: “비상 경보는 확정이 아닙니다. 종가 확정 후에 주문이 활성화됩니다.” 

E03_UI_SSOT

시뮬: “시뮬레이션 모드에서는 주문 복사가 잠겼습니다.” 

E03_UI_SSOT

7) Light/Dark Mode 구현 규칙(Design → Dev 핸드오프)

토글 방식: system(기본) + 사용자 강제 토글

다크 모드에서 금지:

순백 텍스트(눈부심) 과다 사용 → text-1는 약간 낮춘 off-white 사용

과도한 채도(특히 빨강/주황) 면적을 넓게 사용(피로도)

차트:

Gridline은 border보다 약하게(시각적 노이즈 최소)

Positive/Negative는 semantic 색 고정

8) Accessibility & QA Gates (필수 통과 조건)
8.1 A11y

포커스 순서: Zone A → Zone B → Zone C → Zone D 

E03_UI_SSOT

배지는 색상만으로 의미 전달 금지(텍스트 동반) 

E03_UI_SSOT

숫자 입력은 스크린리더 라벨에 단위 포함(원/주)

8.2 Error-proofing

STALE/SOFT_ALERT/SIMULATION 상태에서 Copy는 반드시 잠김 

E03_UI_SSOT

Copy 완료 토스트에 “복사된 티커/수량” 필수 

E03_UI_SSOT

DUE_TODAY에는 Zone C “기록” CTA가 항상 강하게 노출 

E03_UI_SSOT

9) (권장) Tailwind/CSS Variables 스캐폴딩 예시
:root {
  --bg: #F6F7F9;
  --surface: #FFFFFF;
  --border: #E6EAF0;
  --text-1: #0B1220;
  --text-2: #5B667A;

  --primary: #0A84FF;
  --primary-hover: #167CFF;
  --primary-pressed: #0066CC;

  --success: #16A34A;
  --danger: #DC2626;
  --warn: #F59E0B;
}

.dark {
  --bg: #0B0F14;
  --surface: #111827;
  --border: #243044;
  --text-1: #EAF0FF;
  --text-2: #A9B4C7;

  --primary: #0A84FF;
  --primary-hover: #3396FF;
  --primary-pressed: #0066CC;
}
---

## 10) Records 페이지 IA (v1.0)

목표: "과거에 내가 실제로 무엇을 했는지"를 증거/감사 관점에서 빠르게 찾고 비교.

### 10.1 Records 탭 구조 (4개)

| 탭 | 이름 | 용도 |
|---|------|------|
| A | Timeline | 날짜별 실행 기록 리스트 (메인) |
| B | Run Detail | 개별 실행 상세 (Timeline 선택 시 진입) |
| C | Compare | 예상 vs 실제 집계/품질 점검 |
| D | Export | CSV 다운로드/가져오기 |

---

### 10.2 A. Timeline (기본 화면)

날짜별 실행 기록 리스트 (최신순)

각 항목:
- 상태 배지: DONE / SKIPPED / DELAY / UNKNOWN
- 요약: Verdict Date, Exec Date, 주문 유형(SELL/BUY), 핵심 수치(예상 수량/기준가)
- 클릭 → Run Detail로 이동

---

### 10.3 B. Run Detail (상세)

Timeline에서 선택 시 자동 진입

섹션 구성:

| 섹션 | 내용 |
|------|------|
| Decision Snapshot | 당시 Verdict/State, 경고, 기준 가격 |
| Planned Orders | "예상 주문" (전략 카드 기준) |
| Executed Trades | "실제 체결" 입력/불러오기 |
| Diff | 예상 vs 실제 (수량/가격/슬리피지/오차) |
| Notes & Evidence | 메모, 스크린샷 링크/첨부 (선택) |

CTA:
- 실행 기록 수정
- CSV 내보내기 (해당일)

---

### 10.4 C. Compare (비교/집계)

필터: 기간 / 티커 / 상태

집계 지표:
- 예상 대비 체결 오차 평균
- 누락(SKIPPED) 횟수
- DELAY 빈도

목적: 운영 품질 점검 ("기록 잘했나?")

---

### 10.5 D. Export

- CSV 다운로드 (기간 지정)
- 필터링 후 다운로드
- 가져오기(옵션): CSV import (추후)


---

## 11) Analysis 페이지 IA (v1.0)

목표: "전략이 유효했는지"를 사후 검증하고 개선 아이디어를 얻는 곳.

> [!IMPORTANT]
> 여기는 절대 "오늘 행동"을 유도하면 안 됩니다. Command 플로우 보호.

### 11.1 Analysis 탭 구조 (4개)

| 탭 | 이름 | 용도 |
|---|------|------|
| A | Overview | 기간별 KPI 대시보드 |
| B | Strategies | E03 vs 200TQ 비교 |
| C | Attribution | 성과 분해/요인 분석 |
| D | Intel Lab | Research 모드 차트 |

---

### 11.2 A. Overview (기본 화면)

기간 선택: 1M / 3M / 6M / YTD / ALL + 커스텀

KPI 카드:
- CAGR/수익률 (연환산)
- MDD (Maximum Drawdown)
- 변동성
- 샤프/소르티노 (선택)
- 회전율 (Proxy)
- 현금비중 평균 (있다면)

벤치마크 선택: SPLG, Buy&Hold 등

---

### 11.3 B. Strategies (전략 비교)

E03 vs 200TQ 비교 (Command의 Zone D 비교 확장판)

그래프:
- 누적 수익률 (Equity Curve)
- 드로우다운

표:
- 기간별 성과 테이블 (월/분기)

핵심 질문: "누가 이겼는지" + "언제/왜 약했는지"

---

### 11.4 C. Attribution (성과 분해)

성과 분해 (가능한 범위에서):
- Exposure (노출) vs Timing (진입/이탈) vs Cash drag

이벤트 기반 분석:
- Down / Focus / Overheat 상태별 성과
- 실행 누락/지연이 성과에 미친 영향 (Records와 연결)

---

### 11.5 D. Intel Lab (Research View)

Zone D 확장, 단 문맥은 '검증/리서치'로 고정 (오늘 판단 아님)

포함:
- 차트 확대 (오버레이/주기/지표)
- 특정 구간 하이라이트 (드로우다운 구간 클릭 → 해당 기간 차트로 점프)

> [!WARNING]
> "Research Mode / Not for today's execution" 배지로 Command와 명확히 분리


---

## 12) Settings 페이지 IA (v1.0)

목표: "운영 안정성"을 높이는 설정만. "전략 변경"은 금지 또는 버전 관리로만.

> [!NOTE]
> Settings는 "사용자가 매일 안 들어가도 되는 곳"이어야 합니다.
> 항목 수를 늘리기보다 섹션 5개 고정 권장.

### 12.1 Settings 섹션 구조 (5개)

| 섹션 | 이름 | 용도 |
|------|------|------|
| A | App | 테마, 표시 옵션, 레이아웃 |
| B | Mode | 시뮬레이션, 프라이버시, Dev Scenario |
| C | Notifications | 알림 조건 및 채널 |
| D | Data & Integrations | 데이터 소스, API 연동 |
| E | Safety | 실수 방지 옵션 |

---

### 12.2 A. App

- Theme: Dark / Light
- 표시 옵션:
  - 금액 마스킹
  - 소수점 자리
  - 통화 (KRW / USD)
- 레이아웃 옵션:
  - 컴팩트 모드
  - 차트 기본 탭

---

### 12.3 B. Mode

- Simulation Mode 토글
- Visible / Privacy (민감 정보 숨김) 토글
- Dev Scenario (개발 전용):
  - 여기서만 노출
  - "DEV 배지" 상시 표시

---

### 12.4 C. Notifications

- 알림 on/off
- 알림 조건:
  - Verdict 변경
  - Exec scheduled 시점
  - Stop-loss 임박 (거리 기준)
- 채널 (추후): 이메일 / 푸시 / 웹훅

---

### 12.5 D. Data & Integrations

- (현재) 데이터 소스 상태 표시 (MOCK / REAL)
- (추후) API 연동: 브로커, 가격 데이터
- Import/Export 기본 경로/형식 설정

---

### 12.6 E. Safety

"실수 방지" 옵션들:

- 주문 복사 전 확인 체크 (고정)
- 시뮬 모드에서만 주문 복사 허용 (옵션)
- 고위험 상태 (Down 등)에서 경고 강제 표시

