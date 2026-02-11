# 200TQ Dashboard - Project Status

> Last Updated: 2026-01-29

---

## Recent Session Summary (2026-01-29)

### Completed Today

| Task | Description | Commit |
|------|-------------|--------|
| **Cache Issue Fix** | `.next` 폴더 캐시로 인한 로컬 데이터 불일치 해결 | - |
| **RECORD_MISSING Alert Fix** | SGOV/TQQQ 미보유 시 실행일 알림 스킵 | `6f33d7e2` |
| **Macro Page** | 15개 매크로 지표 + 카테고리 그룹핑 UI | `8e913cfd`, `bdc0885f` |
| **Macro Auto-Update** | 캐싱 시스템 + Cron 자동 갱신 | `182c668a` |
| **macro_cache Table** | Supabase 테이블 생성 완료 | SQL 실행 |
| **Simple/Pro View** | 대시보드 뷰 모드 토글 | `ec76efe9` |

### Cron Schedule (KST)

| Endpoint | Schedule | Description |
|----------|----------|-------------|
| `/api/cron/daily` | 매일 07:00 | 일일 알림 체크 |
| `/api/cron/ingest-close` | 평일 07:00 | 주식 가격 업데이트 |
| `/api/macro?refresh=true` | 매일 07:30 | 매크로 지표 캐시 갱신 |

---

## Completed Features

### Data & Backend

| Feature | Description | Status |
|---------|-------------|--------|
| **Multi-Provider Data** | Finnhub -> Polygon -> Yahoo fallback chain | Production |
| **Notification Center** | Verdict change, Data stale, Cron error triggers | Production |
| **Telegram Integration** | Emergency/Action alerts via Telegram | Production |
| **Hard Trigger Monitoring** | QQQ -7%, TQQQ -20% emergency alerts | Production |
| **Idempotency Logic** | Prevents duplicate data ingestion | Fixed |
| **Macro Data Pipeline** | 15개 지표 fetch + 24h 캐싱 | Production |
| **Smart RECORD_MISSING** | 포트폴리오 상태 기반 알림 필터링 | Production |

### Frontend - Already Implemented (DO NOT RE-RECOMMEND)

| Category | Feature | Location |
|----------|---------|----------|
| **분석 기능** | 200TQ vs E03 벤치마크 비교 | 차트 분석 페이지 |
| **분석 기능** | 수익률 계산 (시작자산 -> 현재자산) | 포트폴리오 페이지 |
| **분석 기능** | SMA 오버레이 (3/160/165/170/200/+5%) | 차트 분석 페이지 |
| **분석 기능** | 체결 정확도, 평균 슬리피지 | 기록 페이지 |
| **UI/UX** | Dark/Light/System 테마 토글 | 설정 페이지 |
| **UI/UX** | 모바일 최적화 레이아웃 | 전체 |
| **UI/UX** | Execution Checklist | 메인 대시보드 |
| **UI/UX** | CSV Export | 기록 페이지 |
| **UI/UX** | 금액 마스킹, 소수점 자리 설정 | 설정 페이지 |
| **UI/UX** | Simple/Pro 뷰 모드 토글 | 대시보드 |
| **포트폴리오** | Equity Curve 차트 | 포트폴리오 페이지 |
| **포트폴리오** | OCR 스크린샷 분석 | 포트폴리오 페이지 |
| **포트폴리오** | 보유종목 테이블 (평단가, 현재가, 손익, 비중) | 포트폴리오 페이지 |
| **매크로** | VIX, F&G, 10Y, DXY 등 15개 지표 | Macro 페이지 |
| **매크로** | 카테고리별 그룹핑 (Fear, Rates, Liquidity 등) | Macro 페이지 |
| **매크로** | 자동 캐싱 + Refresh 버튼 | Macro 페이지 |
| **알림** | DATA/OPS/EXEC 상태 카드 | 알림센터 |
| **알림** | 알림 타임라인 + 필터 | 알림센터 |
| **알림** | Verdict/Exec/Stop-Loss 알림 설정 | 설정 페이지 |
| **안전** | 주문 복사 전 확인, 시뮬 모드 제한 | 설정 페이지 |

---

## NOT YET Implemented (Available for Future Work)

### A. 분석 기능 강화

| Feature | Description | Priority |
|---------|-------------|----------|
| **Performance Metrics** | CAGR, MDD, Sharpe Ratio, Sortino Ratio | Medium |
| **Monthly/Yearly Returns Heatmap** | 월별/연도별 수익률 히트맵 | Low |
| **Drawdown Analysis** | 현재 DD 깊이, 회복 예상 일수 | Medium |

### B. AI/ML + 매크로 인텔리전스

| Feature | Description | Priority |
|---------|-------------|----------|
| **News Sentiment Analysis** | 금융 뉴스 AI 분석 (긍정/부정/중립) | High |
| **Market Regime Detection** | Bull/Bear/Sideways 자동 분류 | Medium |
| **AI Trade Commentary** | 오늘의 시그널에 대한 AI 해설 | Medium |

### C. UI/UX 추가 개선

| Feature | Description | Priority |
|---------|-------------|----------|
| **Widget Dashboard** | 드래그앤드롭 커스텀 레이아웃 | Low |
| **PWA/Push Notifications** | 앱 설치 없이 푸시 알림 | Low |

---

## Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `prices_daily` | 일별 종가 데이터 (QQQ, TQQQ, SGOV) |
| `ops_snapshots_daily` | 일별 운영 스냅샷 (verdict, signals) |
| `portfolio_state` | 현재 포트폴리오 상태 |
| `holdings_snapshots` | 보유종목 히스토리 |
| `execution_logs` | 체결 기록 |
| `notifications` | 알림 히스토리 |
| `macro_cache` | 매크로 지표 캐시 (24h TTL) |

### macro_cache Schema

```sql
create table if not exists macro_cache (
  id text primary key default 'latest',
  data_json jsonb not null,
  updated_at timestamptz default now()
);

create index if not exists macro_cache_updated_idx on macro_cache(updated_at desc);
```

---

## File Structure

```
/home/juwon/QuantNeural_wsl/200tq/dashboard/
├── app/
│   ├── api/
│   │   ├── cron/
│   │   │   ├── ingest-close/route.ts    # 주가 업데이트 cron
│   │   │   └── daily/route.ts           # 일일 알림 체크
│   │   ├── macro/route.ts               # 매크로 API + 캐싱
│   │   └── ops/check-alerts/route.ts    # 알림 체크 API
│   └── macro/page.tsx                   # 매크로 대시보드 페이지
├── lib/
│   ├── ops/notifications/
│   │   ├── triggers.ts                  # 알림 트리거
│   │   └── createNotification.ts        # 알림 생성 + Telegram
│   └── macro/
│       └── indicators.ts                # 15개 매크로 지표 정의
├── components/macro/                    # 매크로 UI 컴포넌트
└── .sisyphus/PROJECT_STATUS.md          # This file
```

---

## Environment

| Item | Value |
|------|-------|
| Supabase URL | `https://jttvzicwuqdhgjowpwhf.supabase.co` |
| Vercel Production | https://dashboard-five-tau-66.vercel.app |
| Telegram | Configured |

---

## Known Issues / Notes

1. **로컬 개발 시 `.next` 캐시 주의**: 데이터 불일치 발생 시 `rm -rf .next && npm run dev`
2. **Macro API Rate Limits**: 일부 소스(FRED, Fear&Greed)는 rate limit 있음 - 캐싱으로 해결
3. **RECORD_MISSING 알림**: SGOV/TQQQ 미보유 시 자동 스킵됨

---

## Session History Reference

| Date | Session ID | Key Work |
|------|------------|----------|
| 2026-01-29 | `ses_3f8155412ffezame92SexWNE1d` | Macro 페이지, 알림 개선 |
| 2026-01-28 | `ses_3fe0b8b5dffeDQFOx9gg5s0P0f` | 테마 시스템, Vercel 배포 확인 |
| 2026-01-27 | `ses_3fe23cf0dffe6sFUxD5Ff3IF2V` | 3-Mode 테마 구현 |
| 2026-01-27 | `ses_4008a7c50ffeBoh50zNi2ldb1G` | Portfolio equity, SGOV 지표 |
