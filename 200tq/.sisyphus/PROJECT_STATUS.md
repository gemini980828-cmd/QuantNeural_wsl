# 200TQ Dashboard - Project Status

> **Last Updated**: 2026-01-28
> **Version**: Alpha (Development)
> **Deploy URL**: https://dashboard-five-tau-66.vercel.app

---

## IMPORTANT: AI Agent Instructions

**새 세션 시작 시**: 이 파일을 먼저 읽고 현재 상태를 파악하세요.

**큰 작업 완료 후**: 반드시 이 문서를 업데이트하세요:
1. 완료된 기능을 "Implemented Features"로 이동
2. 새로 발견된 이슈를 "Known Issues"에 추가
3. Last Updated 날짜 갱신

---

## Project Overview

200TQ는 TQQQ/SGOV 스위칭 전략(E03 Ensemble+SGOV) 운영을 위한 개인 트레이딩 대시보드입니다.

### Tech Stack
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS
- **Backend**: Supabase (PostgreSQL), Vercel Serverless Functions
- **Data**: Finnhub (Primary), Polygon (Fallback), yahoo-finance2 (Tertiary)
- **Notifications**: Telegram Bot API + In-App Notification Center

---

## Implemented Features (완료된 기능)

### Pages

#### 1. Command Page (`/command`)
- [x] **Zone A (Header)**: 날짜, 시나리오 탭, TQQQ Flash Crash 알림
- [x] **Zone B (Signal Core)**: 매수유지 ON/OFF, SMA160/165/170 마진율, 앙상블 다수결
- [x] **Zone C (Ops Console)**: 현실(기록) vs 전략(예상), 주문 복사, 실행 기록
- [x] **Zone D (Chart Analysis)**: TQQQ/QQQ 차트, SMA 오버레이, 200TQ/E03 모드 전환
- [x] 포트폴리오 요약 바 (총자산, 오늘 손익, TQQQ 보유, 평가 손익)
- [x] PERF 12M 성과 비교 (200TQ vs E03)

#### 2. Portfolio Page (`/portfolio`)
- [x] 요약 (Overview): 총자산, 오늘 손익, TQQQ 보유량
- [x] 보유 현황 입력 (Holdings Input): TQQQ/SGOV 수량 입력
- [x] 스크린샷 OCR: 삼성증권 앱 스크린샷에서 보유량 자동 추출
- [x] 보유 종목 테이블: 티커, 수량, 평단가, 현재가, 평가금액, 평가손익, 비중
- [x] 알림 조건 표시 (BUY+SGOV보유 → 알림, SELL+TQQQ보유 → 알림)
- [x] MOCK/REAL 모드별 체결 기록/성과 분석 표시

#### 3. Records Page (`/records`)
- [x] 요약 (Summary): 전체 기록, 완료, 스킵, 미확인 카운트
- [x] 운영 품질 (Quality): 체결 정확도, 평균 슬리피지, 지연 기록
- [x] 최근 기록 (Timeline): 날짜별 TQQQ/SGOV 수량, 상태(완료/스킵/지연/미확인)
- [x] 필터링: 30일, 3개월, 전체 / 상태별 필터
- [x] CSV Export 기능
- [x] 예상 vs 실제 비교 로직 (expected_lines 저장)

#### 4. Notifications Page (`/notifications`)
- [x] 오늘 상태 요약: DATA(FRESH/STALE), OPS(성공/실패), EXEC(실행 예정일)
- [x] 알림 타임라인: 전체, Action, Emergency, 해결됨 탭
- [x] 운영 로그 (최근 7일): Cron ingest 실행 기록

#### 5. Analysis Page (`/analysis`)
- [x] 성과 요약 (Overview): CAGR, MDD, 변동성, 샤프 비율
- [x] 수익률 히트맵 (Returns Heatmap): 월별 수익률 색상 매트릭스
- [x] 성과 분해 (Attribution): Exposure, Timing, Cash Drag
- [x] Intel Lab (Backtest): 
  - 시작일/종료일/초기자본/전략 선택
  - Run Backtest 버튼 → Python 백테스트 실행
  - E00~E10 전략 선택 가능
  - Compare Mode (복수 전략 비교)

#### 6. Settings Page (`/settings`)
- [x] 앱 설정: 테마(System/Dark/Light 3-mode), 금액 마스킹, 소수점 자리, 통화, 컴팩트 모드
- [x] 모드 설정: 시뮬레이션, Dev Scenario, 프라이버시 모드
- [x] 알림 설정: 알림 활성화, Verdict 변경, Exec Scheduled, Stop-Loss 임박
- [x] 텔레그램: 연결 상태, Emergency/Action 알림, 테스트 메시지
- [x] 데이터 & 연동: 데이터 소스(MOCK/REAL), API 연동, Import/Export, 스냅샷 Backfill
- [x] 안전: 주문 복사 전 확인, 시뮬 모드에서만 복사 허용, 고위험 상태 경고

### Backend & Infrastructure

#### API Routes
- [x] `/api/prices/history` - 가격 데이터 조회 (Supabase)
- [x] `/api/backtest/run` - Python 백테스트 실행
- [x] `/api/record` - 체결 기록 저장/조회
- [x] `/api/portfolio/state` - 포트폴리오 상태 저장/조회
- [x] `/api/portfolio/ocr` - 스크린샷 OCR 분석
- [x] `/api/cron/ingest-close` - 일별 종가 수집 (Finnhub → Polygon → Yahoo 폴백)
- [x] `/api/cron/daily` - 일일 알림 체크 (Cron)
- [x] `/api/ops/check-alerts` - 수동 알림 체크
- [x] `/api/telegram/send` - 텔레그램 메시지 발송
- [x] `/api/notifications/list` - 알림 목록 조회
- [x] `/api/notifications/ack` - 알림 확인 처리

#### Database (Supabase)
- [x] `prices_daily` - 일별 가격 데이터 (source 필드로 provider 추적)
- [x] `trade_executions` - 체결 기록 (expected_lines 포함)
- [x] `portfolio_states` - 포트폴리오 스냅샷
- [x] `ops_snapshots_daily` - 운영 스냅샷 (verdict, SMA, health)
- [x] `ops_notifications` - 알림 기록 (dedupe_key로 중복 방지)

#### Backtest Engine (Python)
- [x] `scripts/run_suite.py` - 11개 전략 백테스트
- [x] `scripts/backtest_api.py` - API용 백테스트 래퍼
- [x] `scripts/backtest_flash_crash.py` - Flash Crash 영향 분석
- [x] `experiments/` - E00~E10 실험 결과

---

## Known Issues (알려진 이슈)

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| 1 | Records 슬리피지 소수점 너무 김 | Low | `0.1693100408311205%` → 포맷팅 필요 |

---

## Pending Features (대기 중인 기능)

| # | Feature | Priority | Notes |
|---|---------|----------|-------|
| 1 | Portfolio Equity Curve 차트 | Medium | 시간별 자산 가치 시각화 |
| 2 | Hard Trigger 모니터링 | Medium | QQQ -7% / TQQQ -20% 실시간 체크 |
| 3 | 체결 기록 자동 입력 | Low | 증권사 API 연동 |
| 4 | LLM 통합 (AI 분석) | Low | 시장 분석, 전략 제안 |

---

## Recent Changes (최근 변경)

### 2026-01-28
- **Multi-Provider 데이터 통합 완료**
  - Finnhub `/quote` API를 Primary로 전환 (실시간 데이터)
  - Polygon을 Fallback으로 강등
  - yahoo-finance2 npm 패키지를 Tertiary fallback으로 추가
  - `source` 필드로 각 ticker별 데이터 출처 추적
- **알림센터 연동 완료**
  - `ingest-close` 크론에서 직접 알림 트리거
  - VERDICT_CHANGED: 시그널 변경시 앱+Telegram 알림
  - DATA_STALE: 데이터 수집 실패시 알림
  - INGEST_FAIL: 크론 에러시 Emergency 알림
  - dedupe_key로 중복 알림 방지
- 3-mode 테마 시스템 구현 (System/Dark/Light)
  - System 모드: OS 다크/라이트 설정 자동 감지 및 실시간 반영
  - 설정 페이지에 System | Dark | Light 토글 추가
- 전체 UI 테마 호환성 개선
  - 6개 차트 컴포넌트 CSS 변수 적용
  - 6개 페이지 semantic token 적용
  - 하드코딩된 neutral-* 색상 → semantic tokens (bg-surface, border-border, text-fg, text-muted)
- pre-commit hook shebang 수정 (python → python3)

### 2026-01-27
- Flash crash 백테스트 분석 + vibe-kit tooling 추가
- 포트폴리오 데이터 없어도 입력칸 표시 수정
- 포트폴리오 상태 섹션 Settings → Portfolio 페이지로 이동

### 2026-01-26
- Records Compare 기능 완성 (예상 vs 실제 비교)
- MOCK/REAL 모드 분리 완성
- Real Equity Heatmap 인프라 + Cron ingest-close
- 삼성증권 스크린샷 OCR 분석 기능
- 포트폴리오 상태 기반 조건부 알림

---

## File Structure

```
200tq/
├── dashboard/                 # Next.js 프론트엔드
│   ├── app/
│   │   ├── (shell)/          # 메인 페이지들
│   │   │   ├── command/
│   │   │   ├── portfolio/
│   │   │   ├── records/
│   │   │   ├── notifications/
│   │   │   ├── analysis/
│   │   │   └── settings/
│   │   └── api/              # API Routes
│   ├── components/
│   │   ├── e03/              # E03 전략 컴포넌트
│   │   └── analysis/         # 분석 컴포넌트
│   └── lib/
│       ├── ops/e03/          # E03 비즈니스 로직
│       └── stores/           # 상태 관리
├── scripts/                   # Python 백테스트
├── experiments/               # 실험 결과
└── .sisyphus/                # 작업 관리
    ├── PROJECT_STATUS.md     # 이 파일
    ├── plans/                # 작업 계획
    └── notepads/             # 작업 노트
```

---

## Quick Reference

### 환경변수 (Vercel)
| 변수 | 용도 |
|------|------|
| `FINNHUB_API_KEY` | Finnhub 실시간 데이터 (Primary) |
| `POLYGON_API_KEY` | Polygon 데이터 (Fallback) |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 알림 |
| `TELEGRAM_CHAT_ID` | 텔레그램 채팅 ID |
| `CRON_SECRET` | Cron job 인증 |
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase 서비스 키 |

### 로컬 개발
```bash
cd dashboard && bun run dev
```

### 백테스트 실행
```bash
python3 scripts/run_suite.py
```

### 배포
```bash
git push origin main  # Vercel 자동 배포
# 또는
npx vercel --prod --yes
```

---

## Contact

- **Repository**: github.com/gemini980828-cmd/QuantNeural_wsl
- **Deploy**: https://dashboard-five-tau-66.vercel.app
