# Draft: E03 Strategy 스프레드시트 운용 시스템

## 배경
- 사용자는 대시보드 완성 이전에 스프레드시트로 E03 전략을 운용/관리하고자 함
- E03 v2026.3 (D전략): Ensemble + F1 Filter + Emergency Exit
- 전략 상태: ON (100%), ON-Choppy (70%), OFF10 (10%), Emergency (→OFF10)
- 연 4.2회 거래, 삼성증권 기준 수동 매매

## E03 SSOT 핵심 데이터 포인트
- QQQ 종가, SMA3, SMA160, SMA165, SMA170
- 3-window 투표 결과 (pass/fail)
- F1: 과거 40일 시그널 전환 횟수 (flip count)
- Emergency: QQQ 당일 수익률 ≤ -5%, TQQQ 진입가 대비 ≤ -15%
- 포트폴리오: TQQQ/SGOV 보유 수량, 평균단가, 현재가, 비중

## Requirements (confirmed)
- **도구**: Google Sheets
- **가격 데이터**: GOOGLEFINANCE() 함수 연동으로 자동 수집
- **기능 범위**: 전체 5개 모듈 모두 포함
  1. Daily Signal 기록 (SMA3/160/165/170 투표, 상태 판정)
  2. Emergency 모니터링 (QQQ 당일 수익률, TQQQ 진입가 대비)
  3. 거래 로그 (BUY/SELL/HOLD, 수량, 가격, 수수료)
  4. 포트폴리오 현황 (보유 수량, 비중, 평가금액, 손익)
  5. F1 Flip Count 추적 (과거 40일 시그널 전환 횟수 → Choppy 판단)

## Technical Decisions
- Google Sheets + GOOGLEFINANCE() 자동 연동

## Open Questions
- (모두 해결됨)

## Additional Decisions
- **히스토리컬 데이터**: 최근 1년 (2025~) — F1 필터 40일 윈도우 충족
- **통화**: USD 기본 + KRW 환산 열 추가 (GOOGLEFINANCE로 환율 자동)
- **현재 포지션**: 이미 TQQQ/SGOV 보유 중 → 초기값 입력 필요
- **시트 구성**: 탭 분리 (시그널/거래로그/포트폴리오/설정 등)

## Scope Boundaries
- INCLUDE:
  - Google Sheets 워크북 설계 (탭 구조, 열 정의, 수식)
  - GOOGLEFINANCE() 연동 (QQQ/TQQQ/SGOV 가격, USD/KRW 환율)
  - SMA 자동 계산 (3/160/165/170)
  - 앙상블 투표 + 상태 판정 자동화
  - F1 Flip Count 자동 추적
  - Emergency 조건 자동 판단
  - 거래 로그 + 포트폴리오 현황
  - 조건부 서식 (ON/OFF10/Choppy/Emergency 색상 구분)
- EXCLUDE:
  - 백테스트 시뮬레이션 (이미 Python으로 완료)
  - 대시보드 코드와의 자동 연동
  - 알림/이메일 자동화 (향후 확장 가능)
  - 2010~2024 히스토리컬 데이터 백필
