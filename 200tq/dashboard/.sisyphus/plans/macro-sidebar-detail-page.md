# Macro Sidebar + Detail Page Implementation

## TL;DR

> **Quick Summary**: 사이드바에 Macro 메뉴 추가하고 /macro 페이지를 탭 기반 UI(지표/뉴스)로 구현. 지표 탭은 6개 매크로 카드, 뉴스 탭은 placeholder.
> 
> **Deliverables**:
> - AppHeader.tsx에 Macro 네비게이션 항목 추가
> - macro/page.tsx 전면 교체 (탭 UI + 6개 지표 카드)
> 
> **Estimated Effort**: Quick (~30분)
> **Parallel Execution**: NO - sequential (2개 파일만 수정)
> **Critical Path**: Task 1 → Task 2 (독립적, 순서 무관)

---

## Context

### Original Request
사이드바에 Macro 메뉴 추가하고 매크로 상세 페이지 구현

### Interview Summary
**Key Discussions**:
- 사이드바 위치: Portfolio 다음, Records 앞
- 아이콘: Globe (lucide-react)
- 탭 구조: [지표] [뉴스] - 뉴스는 placeholder
- 지표 카드: VIX, F&G, 10Y, DXY, NQ, USD/KRW

**Research Findings**:
- `MacroStrip.tsx`: ColorTone, MacroData 타입 + 색상 유틸리티 함수
- `notifications/page.tsx`: useState<TabId> 탭 UI 패턴
- `/api/macro`: 이미 6개 지표 데이터 반환 중

### Metis Review
**Identified Gaps** (addressed):
- 새로고침 정책 → 페이지 진입 시 fetch (수동 refresh 버튼 미포함)
- 로딩/에러 상태 → MacroStrip 패턴 재사용 (스켈레톤 + "--" fallback)
- 반응형 레이아웃 → grid-cols-2 sm:grid-cols-3 lg:grid-cols-3
- API 실패 시 → 개별 지표 null 처리 (이미 API에서 처리됨)

---

## Work Objectives

### Core Objective
사이드바에서 Macro 페이지로 이동하여 주요 매크로 지표를 한눈에 확인할 수 있도록 구현

### Concrete Deliverables
- `components/AppHeader.tsx`: navItems 배열에 Macro 항목 1줄 추가
- `app/(shell)/macro/page.tsx`: 탭 UI + 지표 카드 페이지 전체 교체

### Definition of Done
- [ ] `npm run build` 성공 (exit code 0)
- [ ] `/macro` 경로 접근 시 탭 UI 렌더링
- [ ] 지표 탭에서 6개 카드 표시
- [ ] 사이드바에서 Macro 클릭 시 /macro로 이동

### Must Have
- 사이드바 Macro 메뉴 (Portfolio 다음 위치)
- 탭 전환 기능 (지표/뉴스)
- 6개 매크로 지표 카드 (실시간 데이터)
- 로딩/에러 상태 처리

### Must NOT Have (Guardrails)
- 차트/스파크라인 구현 (Phase 2 예정)
- WebSocket 실시간 스트리밍
- 개별 지표 알림 설정
- 외부 사이트 링크
- 새로운 API 엔드포인트 생성

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (npm run build만 사용)
- **User wants tests**: NO
- **Framework**: none
- **QA approach**: Manual verification via build + browser

### Automated Verification

각 TODO는 빌드 명령과 Playwright를 통한 브라우저 검증 포함:

```bash
# 빌드 검증
cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run build
# Assert: Exit code 0, no TypeScript errors
```

**For Frontend/UI changes** (using playwright skill):
```
1. Navigate to: http://localhost:3000/macro
2. Wait for: selector "[data-testid='macro-tabs']" to be visible
3. Assert: text "지표" visible on tab
4. Assert: 6 indicator cards visible
5. Click: tab "뉴스"
6. Assert: text "AI 뉴스 분석 준비중" visible
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Add Macro to sidebar navItems
└── Task 2: Implement macro/page.tsx (can run parallel)

Both tasks are independent - no dependencies.
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | None | 2 |
| 2 | None | None | 1 |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | Single agent can handle both sequentially (small scope) |

---

## TODOs

- [ ] 1. Add Macro item to sidebar navigation

  **What to do**:
  - Import `Globe` icon from lucide-react
  - Add Macro item to navItems array between Portfolio and Records
  
  **Must NOT do**:
  - 다른 navItems 순서 변경 금지
  - 아이콘 외 다른 import 추가 금지

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 단순 1줄 추가 작업, 복잡한 로직 없음
  - **Skills**: None needed
    - 파일 편집만 필요, 특수 도메인 지식 불필요

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: None
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `components/AppHeader.tsx:15-22` - navItems 배열 구조 (name, href, icon 형식)
  - `components/AppHeader.tsx:6` - lucide-react import 패턴

  **Acceptance Criteria**:

  **Automated Verification:**
  ```bash
  # Agent runs:
  cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run build
  # Assert: Exit code 0
  # Assert: No "Globe" import error
  ```

  ```bash
  # Agent runs:
  grep -n "Macro" /home/juwon/QuantNeural_wsl/200tq/dashboard/components/AppHeader.tsx
  # Assert: Output contains "Macro", "/macro", "Globe"
  ```

  **Commit**: YES
  - Message: `feat(nav): add Macro menu to sidebar`
  - Files: `components/AppHeader.tsx`
  - Pre-commit: `npm run build`

---

- [ ] 2. Implement Macro detail page with tabs and indicator cards

  **What to do**:
  1. Replace entire `macro/page.tsx` content
  2. Implement tab UI with useState<"indicators" | "news">
  3. Create indicator card grid (2 cols mobile, 3 cols desktop)
  4. Fetch data from `/api/macro` using useEffect
  5. Display 6 indicator cards with:
     - Title (지표명)
     - Current value (대형 폰트)
     - Change % with color coding
     - Status indicator (색상 원)
  6. News tab: placeholder "AI 뉴스 분석 준비중"
  7. Loading state: skeleton cards
  8. Error handling: graceful fallback with "--"

  **Must NOT do**:
  - 차트/스파크라인 추가 금지 (Phase 2)
  - 새로운 API 엔드포인트 생성 금지
  - 외부 라이브러리 추가 금지

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI 컴포넌트 구현, 반응형 레이아웃, 색상 시스템
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: 카드 UI, 탭 전환, 반응형 그리드 구현

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: None
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `app/(shell)/notifications/page.tsx:254-271` - 탭 UI 패턴 (useState, onClick, 스타일링)
  - `app/(shell)/notifications/page.tsx:180-194` - 카드 컴포넌트 스타일 (bg-surface, rounded-xl, border)
  - `components/e03/MacroStrip.tsx:24-45` - 색상 유틸리티 함수 (getCircleColor, getTextColor, getChangeColor)
  - `components/e03/MacroStrip.tsx:61-84` - 로딩 상태 UI 패턴

  **API/Type References** (contracts to implement against):
  - `components/e03/MacroStrip.tsx:6-16` - MacroData 타입 정의 (복사해서 사용)
  - `app/api/macro/route.ts:8-16` - API 응답 스키마

  **WHY Each Reference Matters**:
  - `notifications/page.tsx` 탭 패턴: 동일한 프로젝트의 검증된 탭 전환 UX
  - `MacroStrip.tsx` 색상 함수: VIX/F&G 색상 기준 일관성 유지
  - 기존 카드 스타일: Holo 디자인 시스템 준수

  **Acceptance Criteria**:

  **Automated Verification:**
  ```bash
  # Build verification
  cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run build
  # Assert: Exit code 0
  # Assert: No TypeScript errors
  ```

  **For Frontend/UI changes** (using playwright skill):
  ```
  # Agent executes via playwright browser automation:
  1. Start dev server: npm run dev (background)
  2. Navigate to: http://localhost:3000/macro
  3. Wait for: page load complete
  4. Assert: Tab buttons visible ("지표", "뉴스")
  5. Assert: 6 indicator cards visible (VIX, F&G, 10Y, DXY, NQ, USD/KRW)
  6. Click: "뉴스" tab
  7. Assert: text "AI 뉴스 분석 준비중" visible
  8. Screenshot: .sisyphus/evidence/task-2-macro-page.png
  ```

  **Evidence to Capture:**
  - [ ] Terminal output from `npm run build` (success)
  - [ ] Screenshot of /macro page with indicator cards

  **Commit**: YES
  - Message: `feat(macro): implement detail page with tabs and indicator cards`
  - Files: `app/(shell)/macro/page.tsx`
  - Pre-commit: `npm run build`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(nav): add Macro menu to sidebar` | `components/AppHeader.tsx` | `npm run build` |
| 2 | `feat(macro): implement detail page with tabs and indicator cards` | `app/(shell)/macro/page.tsx` | `npm run build` |

---

## Success Criteria

### Verification Commands
```bash
cd /home/juwon/QuantNeural_wsl/200tq/dashboard && npm run build
# Expected: Build successful, exit code 0

grep "Macro" components/AppHeader.tsx
# Expected: { name: "Macro", href: "/macro", icon: Globe }
```

### Final Checklist
- [ ] Macro appears in sidebar between Portfolio and Records
- [ ] Clicking Macro navigates to /macro
- [ ] /macro page shows tab UI with [지표] [뉴스] tabs
- [ ] 지표 tab shows 6 indicator cards with real data
- [ ] 뉴스 tab shows "AI 뉴스 분석 준비중" placeholder
- [ ] Build passes (npm run build)
- [ ] No TypeScript errors
