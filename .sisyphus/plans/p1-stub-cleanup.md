# P1 Stub Cleanup + SSOT Tax Decision + README 현행화

## TL;DR

> **Quick Summary**: 스텁 페이지(/history, /reports)와 죽은 코드(LNB.tsx) 삭제, Tax Jar 구현 결정을 SSOT에 기록, 낙후된 README를 현재 상태로 업데이트
> 
> **Deliverables**:
> - SSOT에 Tax Jar 결정 문서화 (MOD-C3 섹션)
> - 3개 파일 삭제 (history/page.tsx, reports/page.tsx, LNB.tsx)
> - dashboard/README.md 전면 현행화
> 
> **Estimated Effort**: Quick (30분 이내)
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Task 1, 2 병렬 → Task 3 (README) → Build 검증

---

## Context

### Original Request
사용자가 P0-P3 스코어카드에서 남은 P1 작업을 확인 후, 스텁 페이지를 구현할지 제거할지 판단을 요청.

### Interview Summary
**Key Discussions**:
- `/history` → `/records`가 100% 대체 (체결 타임라인, 품질 분석, CSV 내보내기)
- `/reports` → `/analysis`가 100% 대체 (CAGR/MDD, 히트맵, 백테스트, 전략 비교)
- `LNB.tsx` → 스텁을 참조하지만 아무 곳에서도 import되지 않는 죽은 코드
- 세금 기능 → 사용자가 Excel을 쓰지 않을 것이므로 대시보드에 구현 필요. 단, 현재 스텁과는 별개 모듈(MOD-C3 Tax Jar)

**Research Findings**:
- Wealthfolio(⭐7k), Robinhood, Wealthfront: lean nav 패턴, history/reports 분리 없음
- 한국 해외ETF 세금: 22%, 연 250만원 공제, 매년 5월 신고, USD→KRW 당일환율 변환 필요
- E03 거래 빈도: 연 3~7회 → 수동 가능하나 사용자가 Excel 미사용 확인
- OSS 참고: Rotki(⭐3.7k)만 History/Reports 분리 — 세무/회계 목적일 때만

### Metis Review
**Identified Gaps** (addressed):
- SSOT 업데이트 위치 → MOD-C3 섹션 아래에 결정 기록 (자동 해결)
- README 범위 → 현재 README가 "Overview, Signals, News, Reports" 등 존재하지 않는 페이지 언급. 전면 현행화 필요 (자동 해결)
- `/api/prices/history` API 엔드포인트와 `/history` 페이지 라우트 구분 → 수용 기준에 반영
- LNB 동적 import 가능성 → grep 충분 (자동 해결)

---

## Work Objectives

### Core Objective
P1 스코어카드 항목 정리: 죽은 코드 제거 + 미래 기능 결정 문서화 + 문서 현행화

### Concrete Deliverables
- `E03_Command_Center_SSOT_v2.md` MOD-C3 섹션 업데이트
- `app/(shell)/history/` 디렉터리 삭제
- `app/(shell)/reports/` 디렉터리 삭제
- `components/e03/LNB.tsx` 삭제
- `dashboard/README.md` 현행화

### Definition of Done
- [ ] SSOT에 Tax Jar 결정이 명시되어 있음
- [ ] 3개 파일이 파일시스템에 존재하지 않음
- [ ] README가 현재 7개 페이지 구조를 반영함
- [ ] `npm run build` 성공 (exit code 0)

### Must Have
- SSOT에 "Tax Jar 구현 확정 (Excel 미사용), 별도 /tax 페이지로 구현" 명시
- 스텁 3개 파일 완전 삭제
- README에 현재 존재하는 7개 페이지 목록

### Must NOT Have (Guardrails)
- AppHeader.tsx 수정 (이미 올바른 7개 페이지 네비게이션)
- 데이터베이스 스키마 변경
- 삭제 대상 외 추가 파일 삭제
- Tax Jar 기능 구현 (문서화만)
- 코드 리팩토링

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**

### Test Decision
- **Infrastructure exists**: YES (vitest)
- **Automated tests**: None needed (삭제 + 문서 작업, 코드 변경 없음)
- **Framework**: bun test / vitest (기존)

### Agent-Executed QA Scenarios (MANDATORY)

**모든 Task에 적용:**

```
Scenario: Build succeeds after all changes
  Tool: Bash
  Preconditions: All 3 tasks completed
  Steps:
    1. npm run build
    2. Assert: exit code 0
    3. Assert: no "Module not found" errors in output
  Expected Result: Clean build
  Evidence: Build output captured
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: SSOT Tax Jar 결정 기록
└── Task 2: 스텁 + LNB 삭제

Wave 2 (After Wave 1):
└── Task 3: README 현행화

Wave 3 (After Wave 2):
└── Build 검증 + 최종 grep 확인
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3 (README needs updated page list) | 2 |
| 2 | None | 3 (README needs to not mention deleted pages) | 1 |
| 3 | 1, 2 | None (final) | None |

---

## TODOs

- [ ] 1. SSOT에 Tax Jar 결정 기록

  **What to do**:
  - `E03_Command_Center_SSOT_v2.md`의 `MOD-C3. Tax Jar (선택/고급)` 섹션에 아래 내용 추가:
    - "(선택/고급)" → "(확정 — Excel 미사용)" 로 변경
    - 결정 내용 추가:
      ```
      #### 구현 결정 (2026-02)
      - **결정**: Tax Jar 모듈 구현 확정. 사용자가 Excel을 사용하지 않으므로 대시보드가 유일한 세금 데이터 관리 도구.
      - **삭제된 스텁**: `/reports` 스텁 삭제됨. Tax 기능은 /reports와 무관한 별도 모듈.
      - **구현 형태**: 별도 `/tax` 페이지 (MOD-C3 모듈)
      - **필수 기능**:
        - FIFO tax lots 관리 (수동 체결 입력 기반)
        - 거래일 기준 USD/KRW 환율 자동 조회 (한국은행 매매기준율)
        - 연간 순양도차익 계산 (250만원 기본공제 적용)
        - SGOV 배당 소득 추적 (종합소득세용)
        - 세금/정산용 CSV 내보내기
      - **우선순위**: P2 (현재 연 3~7회 거래로 급하지 않으나, 거래 데이터 축적 시점에 구현)
      - **참고**: 기존 tax_lots 테이블 스키마 및 daily_audit.tax_liability_est 필드 활용
      ```

  **Must NOT do**:
  - SSOT의 다른 섹션 수정
  - 스키마 변경
  - 다른 SSOT 파일(DESIGN, UX, UI) 수정

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 단일 파일의 단일 섹션에 텍스트 추가만
  - **Skills**: []
    - 특별한 스킬 불필요

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `200tq/dashboard/E03_Command_Center_SSOT_v2.md:298-302` — 현재 MOD-C3 Tax Jar 섹션 (여기에 추가)
  - `200tq/dashboard/E03_Command_Center_SSOT_v2.md:391-400` — tax_lots 테이블 스키마 (참조용)
  - `200tq/dashboard/E03_Command_Center_SSOT_v2.md:55-58` — §0.7 리포트 자동 생성 로드맵 (컨텍스트)

  **Acceptance Criteria**:

  - [ ] MOD-C3 제목이 "(확정 — Excel 미사용)"으로 변경됨
  - [ ] "구현 결정 (2026-02)" 서브섹션이 존재함
  - [ ] 결정 내용에 "별도 `/tax` 페이지", "Excel 미사용", "USD/KRW 환율" 키워드 포함

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: SSOT Tax Jar decision recorded correctly
    Tool: Bash (grep)
    Preconditions: SSOT file updated
    Steps:
      1. grep "확정" E03_Command_Center_SSOT_v2.md
      2. Assert: Output contains "Tax Jar" and "확정"
      3. grep "Excel 미사용" E03_Command_Center_SSOT_v2.md
      4. Assert: Output found
      5. grep "/tax" E03_Command_Center_SSOT_v2.md
      6. Assert: Output contains "별도 /tax 페이지"
    Expected Result: All 3 keywords present in MOD-C3 section
    Evidence: grep output captured
  ```

  **Commit**: YES (groups with Task 2)
  - Message: `chore(dashboard): record Tax Jar decision in SSOT, delete stub pages and dead LNB`
  - Files: `E03_Command_Center_SSOT_v2.md`, deleted files
  - Pre-commit: `npm run build`

---

- [ ] 2. 스텁 페이지 + LNB.tsx 삭제

  **What to do**:
  - 다음 3개 파일/디렉터리 삭제:
    1. `app/(shell)/history/page.tsx` (및 빈 history/ 디렉터리)
    2. `app/(shell)/reports/page.tsx` (및 빈 reports/ 디렉터리)
    3. `components/e03/LNB.tsx`
  - 삭제 전 grep으로 참조 확인:
    ```bash
    grep -r "from.*LNB\|import.*LNB" --include="*.tsx" --include="*.ts"
    # 결과 없음 확인
    ```

  **Must NOT do**:
  - AppHeader.tsx 수정 (이미 올바름)
  - 추가 파일 삭제
  - `/api/prices/history` API 엔드포인트 건드리기 (페이지 라우트와 다름)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 파일 3개 삭제 + grep 확인만
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `200tq/dashboard/app/(shell)/history/page.tsx` — 삭제 대상 (7줄, "History (Stub)" 텍스트만)
  - `200tq/dashboard/app/(shell)/reports/page.tsx` — 삭제 대상 (7줄, "Reports (Stub)" 텍스트만)
  - `200tq/dashboard/components/e03/LNB.tsx` — 삭제 대상 (죽은 코드, 아무 곳에서도 import 안 됨)
  - `200tq/dashboard/components/AppHeader.tsx:15-23` — 현재 네비게이션 (7개 페이지, history/reports 없음 — 수정 불필요 확인용)

  **Acceptance Criteria**:

  - [ ] `app/(shell)/history/` 디렉터리 존재하지 않음
  - [ ] `app/(shell)/reports/` 디렉터리 존재하지 않음
  - [ ] `components/e03/LNB.tsx` 존재하지 않음
  - [ ] `app/(shell)/` 하위에 7개 페이지 디렉터리만 존재: analysis, command, macro, notifications, portfolio, records, settings

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Stub files deleted and no orphaned references
    Tool: Bash
    Preconditions: Files deleted
    Steps:
      1. test ! -d app/(shell)/history && echo "history dir deleted"
      2. Assert: "history dir deleted" printed
      3. test ! -d app/(shell)/reports && echo "reports dir deleted"
      4. Assert: "reports dir deleted" printed
      5. test ! -f components/e03/LNB.tsx && echo "LNB deleted"
      6. Assert: "LNB deleted" printed
      7. ls app/(shell)/ | sort
      8. Assert: exactly 7 directories (analysis, command, macro, notifications, portfolio, records, settings)
      9. grep -r "href.*['\"]\/history" --include="*.tsx" app/ components/
      10. Assert: no matches (API routes like /api/prices/history are NOT href links)
      11. grep -r "href.*['\"]\/reports" --include="*.tsx" app/ components/
      12. Assert: no matches
      13. grep -r "from.*LNB\|import.*LNB" --include="*.tsx" --include="*.ts" .
      14. Assert: no matches
    Expected Result: All files deleted, zero orphaned references
    Evidence: Terminal output captured

  Scenario: Build succeeds after deletion
    Tool: Bash
    Preconditions: All 3 files deleted
    Steps:
      1. npm run build
      2. Assert: exit code 0
      3. Assert: no "Module not found" errors mentioning history, reports, or LNB
    Expected Result: Clean build
    Evidence: Build output captured
  ```

  **Commit**: YES (grouped with Task 1)
  - Message: `chore(dashboard): record Tax Jar decision in SSOT, delete stub pages and dead LNB`
  - Files: deleted `history/page.tsx`, `reports/page.tsx`, `LNB.tsx`; modified `E03_Command_Center_SSOT_v2.md`
  - Pre-commit: `npm run build`

---

- [ ] 3. README.md 현행화

  **What to do**:
  - `dashboard/README.md` 전면 업데이트:
    - 제목: "QuantNeural Mobile Ops Dashboard" 유지
    - Features 섹션: 현재 기능 반영 (Overview/Signals/News/Reports → 7개 실제 페이지)
    - Stack 섹션: 현행 의존성 확인 및 업데이트
    - Structure 섹션: 현재 디렉터리 구조 반영
    - Design Tokens 섹션: 현재 토큰 유지 (변경 없음)
    - 추가: "Planned Features" 섹션 (Tax Jar, PDF Reports)
  
  - **현재 README 문제점**:
    - "Overview, Signals, News, Reports" → 이 페이지들은 존재하지 않음
    - `components/overview/` → 이 디렉터리는 존재하지 않음
    - `supabase-schema.sql` → 확인 필요
    - 7개 실제 페이지(Command, Portfolio, Macro, Records, Notifications, Analysis, Settings) 미언급

  **Must NOT do**:
  - README를 과도하게 장문으로 작성 (간결하게 유지)
  - 기존 Setup 절차 변경 (동작하는 것은 건드리지 않기)
  - package.json 수정

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 단일 마크다운 파일 업데이트
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Tasks 1, 2)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 1 (SSOT 결정 내용 참조), Task 2 (삭제된 페이지 확인)

  **References**:

  **Pattern References**:
  - `200tq/dashboard/README.md` — 현재 README (53줄, 심각하게 낙후)
  - `200tq/dashboard/components/AppHeader.tsx:15-23` — 현재 네비게이션 7개 페이지 목록 (README 페이지 목록의 정본)
  - `200tq/dashboard/E03_Command_Center_SSOT_v2.md:298-303` — MOD-C3 Tax Jar (Planned Features 섹션 참조)
  - `200tq/dashboard/E03_Command_Center_SSOT_v2.md:55-58` — §0.7 리포트 자동 생성 (Planned Features 참조)

  **Acceptance Criteria**:

  - [ ] README에 7개 현재 페이지 목록 포함 (Command, Portfolio, Macro, Records, Notifications, Analysis, Settings)
  - [ ] "Overview", "Signals", "News" 등 존재하지 않는 페이지 미언급
  - [ ] "Planned Features" 섹션에 Tax Jar 언급
  - [ ] `components/overview/` 등 존재하지 않는 디렉터리 미언급

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: README reflects current page structure
    Tool: Bash (grep)
    Preconditions: README updated
    Steps:
      1. grep -c "Command\|Portfolio\|Macro\|Records\|Notifications\|Analysis\|Settings" dashboard/README.md
      2. Assert: count >= 7 (all 7 pages mentioned)
      3. grep -i "Signals\b" dashboard/README.md
      4. Assert: no matches (deleted page)
      5. grep "Overview" dashboard/README.md
      6. Assert: no matches or only in project description context, not as a page name
      7. grep "Tax Jar\|/tax" dashboard/README.md
      8. Assert: at least 1 match in Planned Features
    Expected Result: README accurately reflects 7 pages + planned features
    Evidence: grep output captured

  Scenario: No references to deleted/nonexistent pages
    Tool: Bash (grep)
    Preconditions: README updated
    Steps:
      1. grep "components/overview" dashboard/README.md
      2. Assert: no matches
      3. grep "Reports (Stub)\|History (Stub)" dashboard/README.md
      4. Assert: no matches
    Expected Result: Zero references to nonexistent directories or stubs
    Evidence: grep output captured
  ```

  **Commit**: YES (separate)
  - Message: `docs(dashboard): modernize README to reflect current 7-page structure`
  - Files: `README.md`
  - Pre-commit: none (documentation only)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 + 2 | `chore(dashboard): record Tax Jar decision in SSOT, delete stub pages and dead LNB` | E03_Command_Center_SSOT_v2.md, -history/page.tsx, -reports/page.tsx, -LNB.tsx | `npm run build` |
| 3 | `docs(dashboard): modernize README to reflect current 7-page structure` | README.md | grep 검증 |

---

## Success Criteria

### Verification Commands
```bash
# 1. Build passes
cd 200tq/dashboard && npm run build    # Expected: exit code 0

# 2. Stubs deleted
test ! -d app/\(shell\)/history         # Expected: exit 0
test ! -d app/\(shell\)/reports         # Expected: exit 0
test ! -f components/e03/LNB.tsx       # Expected: exit 0

# 3. SSOT updated
grep "확정" E03_Command_Center_SSOT_v2.md     # Expected: "Tax Jar" context
grep "/tax" E03_Command_Center_SSOT_v2.md     # Expected: "별도 /tax 페이지"

# 4. README current
grep -c "Command" README.md                   # Expected: >= 1
grep "Signals" README.md                      # Expected: no match (exit 1)

# 5. No orphaned references
grep -r "href.*'/history'" --include="*.tsx" . # Expected: no match
grep -r "import.*LNB" --include="*.tsx" .      # Expected: no match
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Build passes (`npm run build` → 0)
- [ ] 7 pages remain in `app/(shell)/`
- [ ] SSOT documents Tax Jar as confirmed future work
- [ ] README reflects reality
