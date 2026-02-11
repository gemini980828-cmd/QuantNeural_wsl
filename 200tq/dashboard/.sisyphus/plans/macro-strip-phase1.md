# Macro Indicators Strip - Phase 1

## TL;DR

> **Quick Summary**: PERF 요약 줄 아래에 VIX, Fear&Greed, 10Y Treasury, DXY 4개 매크로 지표를 한 줄로 표시하는 MacroStrip 컴포넌트 추가. API 캐싱 1시간, 에러 시 graceful degradation.
> 
> **Deliverables**:
> - `app/api/macro/route.ts` - 매크로 데이터 API 엔드포인트
> - `components/e03/MacroStrip.tsx` - 콤팩트 매크로 표시 컴포넌트
> - `ZoneBSignalCore.tsx` 수정 - MacroStrip 통합
> 
> **Estimated Effort**: Short (2-3 hours)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (API) → Task 3 (Integration)

---

## Context

### Original Request
메인 페이지의 PERF 요약 줄 아래에 매크로 지표를 한 줄로 통합 추가.
- Phase 1: 메인 페이지에 Tier 1 매크로 위젯 (VIX, Fear&Greed, 10Y Treasury, DXY)
- Phase 2: `/macro` 상세 페이지 (나중에)

### Target Layout
```
📊 기준: 10,000만  PERF(12M): 200TQ +22.2% | E03 +33.6%  [세팅]
🌐 VIX 18.5 🟡 │ F&G 65 🟢 │ 10Y 4.25% │ DXY 104  [상세→]
```

### Interview Summary
**Key Decisions**:
- VIX 색상: 🟢 <15 | 🟡 15-25 | 🔴 >25
- Fear&Greed 색상: 🔴 0-25 | 🟡 26-50 | 🟢 51-75 | 🟡 76-100 (극단 경고)
- 10Y/DXY: 중립 표시 (색상 없음, 값만 표시)
- 모바일: 2행 래핑 (VIX|F&G + 10Y|DXY)
- API 실패 시: 부분 표시 (실패한 지표만 "--")

**Research Findings**:
- `ZoneBSignalCore.tsx` lines 333-370에 PERF strip 위치
- `fetchWithFallback` 패턴이 `ingest-close/route.ts`에 존재
- yahoo-finance2 설치됨 (VIX: ^VIX, 10Y: ^TNX, DXY: DX-Y.NYB)
- Alternative.me API로 Fear&Greed 조회 가능 (무료, 키 불필요)
- `StatusBadge` 컴포넌트 존재 (ok/danger/action/neutral/info tones)

### Metis Review
**Identified Gaps** (addressed):
- 색상 기준값 → 사용자 확인 완료
- 모바일 레이아웃 → 2행 래핑 확정
- 부분 실패 처리 → 부분 표시 확정
- 10Y/DXY 색상 해석 어려움 → 중립 표시로 결정

---

## Work Objectives

### Core Objective
PERF 요약 줄 바로 아래에 4개 매크로 지표를 한 줄로 표시하여 시장 상황을 한눈에 파악 가능하게 함.

### Concrete Deliverables
- `app/api/macro/route.ts` - 4개 지표 데이터 반환 API
- `components/e03/MacroStrip.tsx` - 매크로 지표 표시 컴포넌트
- `ZoneBSignalCore.tsx` 수정 - PERF strip 아래 MacroStrip 추가
- `app/(shell)/macro/page.tsx` - Phase 2용 플레이스홀더 페이지

### Definition of Done
- [ ] `npm run build` 성공 (0 errors)
- [ ] PERF 줄 바로 아래에 MacroStrip 표시
- [ ] 4개 지표 모두 표시 (VIX, F&G, 10Y, DXY)
- [ ] VIX/F&G에 색상 인디케이터 표시
- [ ] API 실패 시 "--" 표시, UI 정상 동작

### Must Have
- VIX, Fear&Greed, 10Y Treasury, DXY 4개 지표 표시
- VIX 색상: 🟢 <15 | 🟡 15-25 | 🔴 >25
- F&G 색상: 🔴 0-25 (Extreme Fear) | 🟡 26-50 | 🟢 51-75 | 🟡 76-100 (Extreme Greed)
- 10Y/DXY: 값만 표시 (색상 없음)
- 1시간 캐싱 (Next.js revalidate 사용)
- API 실패 시 graceful degradation ("--" 표시)
- 모바일 2행 래핑

### Must NOT Have (Guardrails)
- 차트, 그래프, 히스토리 뷰 (Phase 2)
- 5번째 이상 지표 추가
- 실시간 업데이트 (WebSocket/Polling)
- 임계값 설정 관리 패널
- 개별 지표 상세 페이지 (Phase 2)
- 데이터베이스 테이블 생성
- PERF strip 스타일/로직 수정

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (테스트 파일 없음)
- **User wants tests**: Manual-only
- **Framework**: none

### Manual QA Procedures

**By Deliverable Type:**
| Type | Verification Tool | Procedure |
|------|------------------|-----------|
| **API** | curl / browser | GET 요청, JSON 응답 확인 |
| **Frontend** | Browser devtools | 시각적 확인, 반응형 테스트 |
| **Build** | npm run build | 빌드 성공 여부 |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: API 엔드포인트 생성 (/api/macro)
└── Task 2: MacroStrip 컴포넌트 생성

Wave 2 (After Wave 1):
├── Task 3: ZoneBSignalCore에 MacroStrip 통합
└── Task 4: /macro 플레이스홀더 페이지 생성

Wave 3 (After Wave 2):
└── Task 5: 최종 검증 및 빌드 테스트
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3 | 2, 4 |
| 2 | None | 3 | 1, 4 |
| 3 | 1, 2 | 5 | 4 |
| 4 | None | 5 | 1, 2, 3 |
| 5 | 3, 4 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | `category="quick"` with parallel background tasks |
| 2 | 3, 4 | `category="quick"` after Wave 1 completes |
| 3 | 5 | Final verification task |

---

## TODOs

### Task 1: 매크로 데이터 API 엔드포인트 생성

- [ ] 1. Create `/api/macro` route

  **What to do**:
  - `app/api/macro/route.ts` 생성
  - yahoo-finance2로 VIX(^VIX), 10Y(^TNX), DXY(DX-Y.NYB) 조회
  - Alternative.me API로 Fear&Greed 조회
  - 각 지표별 try-catch로 개별 에러 처리
  - Next.js revalidate: 3600 (1시간 캐싱)
  - 응답 스키마:
    ```typescript
    interface MacroData {
      vix: { value: number | null; color: 'ok' | 'action' | 'danger' };
      fng: { value: number | null; label: string; color: 'ok' | 'action' | 'danger' };
      treasury: { value: number | null };
      dxy: { value: number | null };
      updatedAt: string;
    }
    ```

  **Must NOT do**:
  - 데이터베이스 저장
  - 복잡한 fallback 체인 (yahoo만 사용)
  - WebSocket 실시간 업데이트

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 단일 API 파일 생성, 명확한 패턴 존재
  - **Skills**: []
    - No special skills needed for API route creation
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: API 로직만 다루므로 불필요

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `app/api/cron/ingest-close/route.ts:15-16` - yahoo-finance2 import 패턴
  - `app/api/cron/ingest-close/route.ts:153-173` - fetchWithFallback 에러 처리 패턴
  - `app/api/cron/ingest-close/route.ts:130-150` - fetchYahooBar 함수 패턴

  **API/Type References**:
  - yahoo-finance2 quote API: `yahooFinance.quote(symbol)`
  - Alternative.me F&G: `https://api.alternative.me/fng/`

  **Yahoo Finance Symbol Validation** (사전 검증 필수):
  
  **작업 시작 전 반드시 확인**:
  ```bash
  # 개발 서버에서 Node REPL로 검증
  node -e "
    const yahooFinance = require('yahoo-finance2').default;
    (async () => {
      try {
        const vix = await yahooFinance.quote('^VIX');
        console.log('VIX:', vix.regularMarketPrice);
        const tnx = await yahooFinance.quote('^TNX');
        console.log('10Y:', tnx.regularMarketPrice);
        const dxy = await yahooFinance.quote('DX-Y.NYB');
        console.log('DXY:', dxy.regularMarketPrice);
      } catch (e) { console.error(e); }
    })();
  "
  ```
  
  **검증 체크리스트**:
  - [ ] ^VIX 응답 확인 (regularMarketPrice 존재)
  - [ ] ^TNX 응답 확인 (regularMarketPrice 존재)
  - [ ] DX-Y.NYB 응답 확인 (regularMarketPrice 존재)
  
  **만약 심볼 실패 시 대체**:
  - ^VIX → VIX (심볼 변형 시도)
  - ^TNX → 하드코딩 fallback 또는 제외
  - DX-Y.NYB → DX=F (선물 심볼 시도)

  **Alternative.me API Response Format** (검증 완료):
  ```json
  // GET https://api.alternative.me/fng/
  {
    "name": "Fear and Greed Index",
    "data": [{
      "value": "29",
      "value_classification": "Fear",
      "timestamp": "1769558400"
    }]
  }
  ```
  
  **Parsing Logic**:
  ```typescript
  const res = await fetch('https://api.alternative.me/fng/');
  const json = await res.json();
  const value = parseInt(json.data[0].value);
  const label = json.data[0].value_classification; // "Fear", "Greed", etc.
  ```

  **Color Logic**:
  ```typescript
  // VIX color logic
  function getVixColor(value: number): 'ok' | 'action' | 'danger' {
    if (value < 15) return 'ok';      // 🟢 Low volatility
    if (value <= 25) return 'action'; // 🟡 Normal
    return 'danger';                   // 🔴 High volatility
  }

  // Fear & Greed color logic (극단 = 경고)
  function getFngColor(value: number): 'ok' | 'action' | 'danger' {
    if (value <= 25) return 'danger';  // 🔴 Extreme Fear
    if (value <= 50) return 'action';  // 🟡 Fear to Neutral
    if (value <= 75) return 'ok';      // 🟢 Greed (optimal)
    return 'action';                    // 🟡 Extreme Greed
  }
  ```

  **Acceptance Criteria**:

  **API Verification**:
  - [ ] `curl http://localhost:3000/api/macro` 실행
  - [ ] Response status: 200
  - [ ] Response body contains: `{"vix":{"value":...,"color":"..."},"fng":{...},"treasury":{...},"dxy":{...},"updatedAt":"..."}`
  - [ ] 각 value가 null이 아닌 숫자값 (정상 케이스)
  - [ ] 두 번째 요청 시 1초 이내 응답 (캐시 히트 확인)

  **Error Handling Verification**:
  - [ ] Alternative.me API 응답 시간 확인: `curl -w "%{time_total}" https://api.alternative.me/fng/`
  - [ ] yahoo-finance2 심볼 확인: ^VIX, ^TNX, DX-Y.NYB 모두 유효

  **Commit**: YES
  - Message: `feat(macro): add /api/macro endpoint for market indicators`
  - Files: `app/api/macro/route.ts`
  - Pre-commit: `npm run build`

---

### Task 2: MacroStrip 컴포넌트 생성

- [ ] 2. Create MacroStrip component

  **What to do**:
  - `components/e03/MacroStrip.tsx` 생성
  - Props: `data: MacroData | null`, `isLoading: boolean`
  - 4개 지표를 한 줄로 표시 (VIX │ F&G │ 10Y │ DXY)
  - StatusBadge 컴포넌트 활용하여 VIX/F&G 색상 표시
  - 10Y/DXY는 색상 없이 값만 표시 (일반 텍스트로)
  - 모바일(sm 이하): 2행 래핑
  - [상세→] 버튼 → `/macro` 링크
  - 로딩 상태: "..." 표시
  - 에러 상태: "--" 표시

  **Color Rendering Implementation** (StatusBadge tone 매핑):
  ```tsx
  // VIX/F&G - StatusBadge 사용 (색상 dot 표시)
  {data.vix.value !== null ? (
    <StatusBadge tone={data.vix.color}>
      VIX {data.vix.value.toFixed(1)}
    </StatusBadge>
  ) : (
    <span className="text-muted">VIX --</span>
  )}

  // 10Y/DXY - 일반 텍스트 사용 (색상 없음)
  <span className="text-fg font-mono text-xs">
    10Y {data.treasury.value !== null ? `${data.treasury.value.toFixed(2)}%` : '--'}
  </span>
  <span className="text-fg font-mono text-xs">
    DXY {data.dxy.value !== null ? data.dxy.value.toFixed(0) : '--'}
  </span>
  ```

  **Mobile Layout Implementation** (Tailwind flex-wrap):
  ```tsx
  {/* 컨테이너: sm 이상에서 한 줄, sm 미만에서 2행 래핑 */}
  <div className="bg-surface rounded-lg shadow-sm px-4 py-2">
    <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap sm:gap-3 text-xs">
      {/* 첫 번째 그룹: VIX + F&G */}
      <div className="flex items-center gap-2">
        <Globe size={12} className="text-muted" />
        <StatusBadge tone={vixColor}>VIX {vixValue}</StatusBadge>
        <span className="text-border">│</span>
        <StatusBadge tone={fngColor}>F&G {fngValue} {fngLabel}</StatusBadge>
      </div>
      
      {/* 두 번째 그룹: 10Y + DXY + 상세 링크 */}
      <div className="flex items-center gap-2">
        <span className="text-border hidden sm:inline">│</span>
        <span className="text-fg font-mono">10Y {treasuryValue}%</span>
        <span className="text-border">│</span>
        <span className="text-fg font-mono">DXY {dxyValue}</span>
        <Link href="/macro" className="text-muted hover:text-fg text-[10px]">[상세→]</Link>
      </div>
    </div>
  </div>
  ```

  **Must NOT do**:
  - 차트/그래프 추가
  - 툴팁/설명 추가
  - 개별 지표 클릭 상세 페이지

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 단일 컴포넌트 생성, 기존 패턴 따름
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: 반응형 레이아웃, StatusBadge 활용 스타일링
  - **Skills Evaluated but Omitted**:
    - `playwright`: 테스트 자동화 불필요 (수동 검증)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `components/e03/ZoneBSignalCore.tsx:335-370` - PERF strip 레이아웃 패턴 (동일한 스타일링 따라야 함)
  - `components/e03/StatusBadge.tsx:12-41` - StatusBadge 사용법 (tone prop)
  - `components/e03/StrategyStrip.tsx` - 유사한 한 줄 요약 컴포넌트 패턴

  **Type References**:
  - `components/e03/StatusBadge.tsx:3` - StatusTone 타입 (ok | danger | action | neutral | info)

  **Layout Spec**:
  ```
  Desktop (sm+):
  ┌─────────────────────────────────────────────────────────┐
  │ 🌐 VIX 18.5 🟡 │ F&G 65 Greed 🟢 │ 10Y 4.25% │ DXY 104 │ [상세→] │
  └─────────────────────────────────────────────────────────┘

  Mobile (<sm):
  ┌────────────────────────────────────┐
  │ 🌐 VIX 18.5 🟡 │ F&G 65 Greed 🟢  │
  │    10Y 4.25%   │ DXY 104  [상세→] │
  └────────────────────────────────────┘
  ```

  **Acceptance Criteria**:

  **Visual Verification (Browser)**:
  - [ ] 컴포넌트가 에러 없이 렌더링됨
  - [ ] VIX, F&G, 10Y, DXY 4개 지표 모두 표시
  - [ ] VIX/F&G에 StatusBadge 색상 dot 표시
  - [ ] 10Y/DXY에 색상 없이 값만 표시
  - [ ] [상세→] 버튼 클릭 시 `/macro`로 이동
  - [ ] 모바일 뷰포트(375px)에서 2행 래핑 확인

  **Error State Verification**:
  - [ ] `data.vix.value = null` 시 "VIX --" 표시
  - [ ] `data = null` 시 전체 스켈레톤/로딩 표시

  **Commit**: NO (groups with Task 3)

---

### Task 3: ZoneBSignalCore에 MacroStrip 통합

- [ ] 3. Integrate MacroStrip into ZoneBSignalCore

  **What to do**:
  - `ZoneBSignalCore.tsx` 수정
  - **useEffect + fetch 패턴 사용** (SWR 미설치됨)
  - PERF strip (`bg-surface rounded-lg shadow-sm`) 바로 아래에 MacroStrip 추가
  - 동일한 컨테이너 스타일 (`bg-surface rounded-lg shadow-sm`) 적용
  - 로딩/에러 상태 처리

  **Data Fetching Implementation** (SWR 미설치, useEffect + fetch 사용):
  ```tsx
  // 1. 상단에 import 추가
  import MacroStrip from './MacroStrip';
  
  // 2. MacroData 타입 정의 (컴포넌트 상단 또는 별도 타입 파일)
  interface MacroData {
    vix: { value: number | null; color: 'ok' | 'action' | 'danger' };
    fng: { value: number | null; label: string; color: 'ok' | 'action' | 'danger' };
    treasury: { value: number | null };
    dxy: { value: number | null };
    updatedAt: string;
  }
  
  // 3. 컴포넌트 내부에 state 및 useEffect 추가
  const [macroData, setMacroData] = useState<MacroData | null>(null);
  const [macroLoading, setMacroLoading] = useState(true);
  
  useEffect(() => {
    fetch('/api/macro')
      .then(res => res.ok ? res.json() : Promise.reject('API Error'))
      .then(data => setMacroData(data))
      .catch(err => {
        console.error('Macro fetch failed:', err);
        setMacroData(null);
      })
      .finally(() => setMacroLoading(false));
  }, []);
  ```

  **Integration Point** (정확한 JSX 트리 구조):
  ```tsx
  {/* 
    ZoneBSignalCore.tsx 구조:
    <section> (최상위)
      ...
      {/* PERF strip (lines 333-429) */}
      {perfSummary && (
        <div className="bg-surface rounded-lg shadow-sm">
          {/* PERF content + Settings panel */}
        </div>
      )}
      
      {/* ↓ 여기에 MacroStrip 추가 (line 429 직후, </section> 직전) */}
      <MacroStrip data={macroData} isLoading={macroLoading} />
      
    </section>
  */}
  
  // 실제 삽입 위치: line 429의 닫는 )}와 line 430의 </section> 사이
  // 즉, PERF strip의 닫는 div와 같은 레벨(형제 요소)로 추가
  ```

  **Must NOT do**:
  - PERF strip 자체 수정
  - SWR 설치 (현재 프로젝트에서 사용하지 않음)
  - 복잡한 캐싱 로직 (서버 API에서 Next.js revalidate 처리)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 기존 파일에 import + JSX 추가만 필요
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: 레이아웃 통합, 스타일 일관성
  - **Skills Evaluated but Omitted**:
    - `git-master`: 단순 파일 수정으로 커밋 자동화 불필요

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1, Task 2

  **References**:

  **Pattern References**:
  - `components/e03/ZoneBSignalCore.tsx:333-429` - PERF strip 위치 (이 바로 아래에 추가)
  - `components/e03/ZoneBSignalCore.tsx:1` - "use client" 디렉티브 확인
  - `components/e03/ZoneBSignalCore.tsx:429` - PERF strip 닫는 괄호 `)}` 위치
  - `components/e03/ZoneBSignalCore.tsx:430-431` - `</section>` 및 컴포넌트 종료

  **Integration Point**:
  ```tsx
  // 기존 PERF strip (lines 333-370)
  {perfSummary && (
    <div className="bg-surface rounded-lg shadow-sm">
      {/* ... PERF content ... */}
    </div>
  )}

  // 새로 추가할 MacroStrip (PERF 바로 아래)
  <MacroStrip data={macroData} isLoading={macroLoading} />
  ```

  **Acceptance Criteria**:

  **Visual Verification (Browser at localhost:3000)**:
  - [ ] PERF strip 표시 확인
  - [ ] MacroStrip이 PERF strip 바로 아래에 표시
  - [ ] 두 strip 사이 간격 일관성 (같은 margin/gap)
  - [ ] 페이지 로드 시 MacroStrip 데이터 표시 (1-2초 내)

  **Error Handling Verification**:
  - [ ] Network 탭에서 /api/macro 요청 성공 확인 (200)
  - [ ] API 실패 시 (Network throttle) MacroStrip에 "--" 표시

  **Commit**: YES
  - Message: `feat(macro): integrate MacroStrip below PERF summary`
  - Files: `components/e03/ZoneBSignalCore.tsx`, `components/e03/MacroStrip.tsx`
  - Pre-commit: `npm run build`

---

### Task 4: /macro 플레이스홀더 페이지 생성

- [ ] 4. Create /macro placeholder page

  **What to do**:
  - `app/(shell)/macro/page.tsx` 생성
  - Phase 2 안내 텍스트 표시: "상세 매크로 지표 페이지 (Phase 2에서 구현 예정)"
  - 기본 레이아웃만 적용 (shell layout 활용)

  **Must NOT do**:
  - 차트/데이터 표시
  - API 호출
  - 복잡한 UI

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 단순 플레이스홀더 페이지
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: 단순 텍스트 페이지로 불필요

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: None (독립적)

  **References**:

  **Pattern References**:
  - `app/(shell)/settings/page.tsx` - 기존 shell 페이지 패턴

  **Acceptance Criteria**:

  **Visual Verification**:
  - [ ] `/macro` 접속 시 페이지 표시
  - [ ] "Phase 2에서 구현 예정" 텍스트 확인
  - [ ] 사이드바/헤더 등 shell 레이아웃 정상 적용

  **Navigation Verification**:
  - [ ] MacroStrip의 [상세→] 버튼 클릭 시 `/macro`로 이동

  **Commit**: NO (groups with Task 3)

---

### Task 5: 최종 검증 및 빌드 테스트

- [ ] 5. Final verification and build test

  **What to do**:
  - `npm run build` 실행하여 빌드 성공 확인
  - 로컬 개발 서버에서 전체 흐름 테스트
  - 모바일 뷰포트 테스트

  **Must NOT do**:
  - 새로운 기능 추가
  - 코드 수정 (버그 발견 시 별도 커밋)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 검증 작업만 수행
  - **Skills**: [`playwright`]
    - `playwright`: 브라우저 자동화로 시각적 검증
  - **Skills Evaluated but Omitted**:
    - `git-master`: 검증 단계에서 커밋 불필요

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final)
  - **Blocks**: None
  - **Blocked By**: Task 3, Task 4

  **References**:

  **Verification Commands**:
  - `npm run build` - 빌드 성공 확인
  - `npm run dev` - 개발 서버 시작

  **Acceptance Criteria**:

  **Build Verification**:
  - [ ] `npm run build` 실행
  - [ ] Exit code: 0
  - [ ] "✓ Compiled successfully" 메시지 확인
  - [ ] 0 type errors, 0 warnings (lint 제외)

  **Browser Verification**:
  - [ ] http://localhost:3000 접속
  - [ ] PERF strip 표시 확인
  - [ ] MacroStrip 4개 지표 표시 확인
  - [ ] VIX/F&G 색상 인디케이터 표시 확인
  - [ ] [상세→] 클릭 → /macro 이동 확인
  - [ ] DevTools 375px에서 2행 래핑 확인

  **API Verification**:
  - [ ] Network 탭에서 /api/macro 요청 확인
  - [ ] Response 200, JSON 데이터 정상

  **Commit**: YES (if any fixes needed)
  - Message: `fix(macro): [specific fix description]`
  - Pre-commit: `npm run build`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(macro): add /api/macro endpoint for market indicators` | `app/api/macro/route.ts` | curl test |
| 3 | `feat(macro): integrate MacroStrip below PERF summary` | `ZoneBSignalCore.tsx`, `MacroStrip.tsx`, `macro/page.tsx` | npm run build |
| 5 | `fix(macro): ...` (if needed) | varies | npm run build |

---

## Success Criteria

### Verification Commands
```bash
# Build verification
npm run build  # Expected: ✓ Compiled successfully

# API verification
curl http://localhost:3000/api/macro  # Expected: JSON with vix, fng, treasury, dxy

# Dev server
npm run dev  # Start and visually verify
```

### Final Checklist
- [ ] All "Must Have" present (4 indicators, colors, caching, error handling)
- [ ] All "Must NOT Have" absent (no charts, no 5th indicator, no realtime)
- [ ] Build succeeds with 0 errors
- [ ] Mobile layout works (2-row wrap)
- [ ] [상세→] navigates to /macro placeholder
