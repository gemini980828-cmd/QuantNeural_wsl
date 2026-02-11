# QuantNeural Mobile Ops Dashboard

E03 전략을 무결점 실행하기 위한 개인용 트레이딩 운영 대시보드.

## 📱 Pages (7)

- **Command** — 오늘의 시그널 판정, 4-state 실행 지시 (ON/ON_CHOPPY/OFF10/EMERGENCY), 체결 기록
- **Portfolio** — 보유 종목 현황, 목표 vs 실제 비중, OCR 스크린샷 분석
- **Macro** — 15개 매크로 지표 (VIX, 장단기 금리차, USD/KRW 등)
- **Records** — 체결 타임라인, 운영 품질 분석 (정확도/슬리피지/지연), CSV 내보내기
- **Notifications** — 데이터 신선도, 작업 알림, 시스템 상태
- **Analysis** — 백테스트 (단일/복수 전략 비교), 수익률 히트맵, 성과 분해
- **Settings** — 데이터 소스 (MOCK/REAL), 시뮬레이션 모드, 통화 설정

## 🛠️ Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS + Holo dark glassmorphism 디자인 시스템
- **Icons**: Lucide React
- **Charts**: Recharts
- **Database**: Supabase
- **Backtest Engine**: Python (FastAPI)

## 🚀 Setup

1. **Install Dependencies**

   ```bash
   npm install
   ```

2. **Run Local Dev**

   ```bash
   npm run dev
   ```

3. **Supabase Setup**
   - Create a new project on Supabase.
   - Run the contents of `supabase-schema.sql` in the SQL Editor.
   - Set environment variables in `.env.local`.

## 📂 Structure

- `app/(shell)/` — 7개 페이지 (command, portfolio, macro, records, notifications, analysis, settings)
- `app/api/` — API routes (ops/today, backtest/run, macro, portfolio, records)
- `components/e03/` — E03 전략 전용 UI (ZoneA~C, SimpleView, RecordModal)
- `components/analysis/` — 백테스트 차트 (EquityCurveChart, ReturnsHeatmap, SingleStrategyPanel)
- `components/portfolio/` — 포트폴리오 UI (SummaryStrip, PositionsTable, EquityChart)
- `components/ui/` — 공통 UI (Toast)
- `lib/ops/e03/` — 코어 로직 (buildViewModel, types, mock)
- `lib/stores/` — Zustand 상태 관리

## 🎨 Design Tokens

- **Background**: `#090909`
- **Primary (Lime)**: `#ABF43F`
- **Secondary (Cyan)**: `#3FF4E5`
- **Card**: `rounded-xl bg-surface border-border`
- **Badge (active)**: `bg-{color}-900/40 text-{color}-400 border-{color}-800`

## 🚧 Planned Features

- **Tax Jar** (`/tax`, MOD-C3) — FIFO tax lots, USD/KRW 환율 자동 조회, 연간 양도차익 계산, CSV 내보내기
- **PDF/CSV Reports** (MOD-D5) — 월간/연간 운영 리포트 자동 생성
