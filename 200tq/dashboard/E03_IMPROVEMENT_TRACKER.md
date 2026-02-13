# E03 Dashboard 개선점 트래커

> 최종 검토: 2026-02-13
> SSOT 교차검증 기반 23개 개선 항목 — ✅ 완료

## 🔴 Critical (안전)

- [x] 1. 복사 토스트에 종목/수량 표시 — `ZoneCOpsConsole.tsx`
- [x] 2. 복사 잠금 사유 SSOT 원문 일치 — `formatters.ts`
- [x] 3. 시뮬레이션 워터마크 강화 — `command/page.tsx`
- [x] 4. Two Truths 라벨 SSOT 원문 일치 — `ZoneCOpsConsole.tsx`

## 🟠 High (속도)

- [x] 5. Verdict 상태 최대 타이포그래피 — `ZoneBSignalCore.tsx:267-273`
- [x] 6. PortfolioStrip을 Zone C 아래로 이동 — `command/page.tsx:390-394`
- [x] 7. 정상 상태 뱃지 숨기기 — `ZoneAHeader.tsx:96-167`
- [x] 8. Sim/Privacy/Theme 토글 Settings 이동 — `ZoneAHeader.tsx:42-91`
- [x] 9. Pro View Hero Metric 추가 — 신규
- [x] 10. MacroStrip Zone B 시각 분리 — `command/page.tsx`

## 🟡 Medium (품질)

- [x] 11. Evidence 카드 게이지 바 추가 — `ZoneBSignalCore.tsx:292-327`
- [x] 12. 하드코딩 색상 토큰 수정 — `SimpleView, ZoneBSignalCore, AppHeader`
- [x] 13. 섹션 간격 SSOT 스펙 적용 — 전체
- [x] 14. 카드 배경/테두리 명암 강화 — `globals.css`
- [x] 15. tabular-nums 전역 적용 — `globals.css`
- [x] 16. Dev Scenario 바 프로덕션 숨기기 — 관련 컴포넌트
- [x] 17. text-[10px] → 최소 11px — 전체

## 🟢 Lower (확장)

- ~~18. Records 4탭 구조~~ — 스킵 (사용자 결정: Records 탭 미생성)
- [x] 19. Portfolio 전략 vs 실제 비교 — `PortfolioPositionsTable.tsx`, `portfolio/page.tsx`
- ~~20. Analysis 탭 분리~~ — 스킵 (사용자 결정: 분리 불필요)
- [x] 21. Notifications 트리거 완성 — `notifications/page.tsx`
- [x] 22. Settings 5섹션 검증 — `settings/page.tsx`, `settings-store.ts`
- ~~23. 다크모드 순백 텍스트 완화~~ — 이전 세션에서 해결 완료
