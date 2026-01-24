---
description: Vibe 시스템 기반 코드 작업 워크플로우
---

# /vibe Workflow

> **이 워크플로우는 모든 코드 작업에 자동 적용됩니다.**

## 1. 작업 시작 전 (필수)

// turbo

```bash
# 최신 컨텍스트 확인
cat .vibe/context/LATEST_CONTEXT.md
```

// turbo

```bash
# 금지사항 확인
cat .vibe/agent_memory/DONT_DO_THIS.md
```

// turbo

```bash
# 수정할 파일의 영향 분석 (파일 경로 변경)
.venv/bin/python .vibe/brain/impact_analyzer.py <target_file>
```

---

## 2. 작업 중 규칙

- **JSDoc 자동생성 금지** - 전역 JSDoc 생성/편집 절대 금지
- **직접 수정하는 boundary 함수만** @param/@returns 추가/갱신
- **순환 의존성 금지** - deep imports 피하기
- **Diff 최소화** - 리포지토리 전체 변경 피하기

---

## 3. 작업 완료 전 (필수)

// turbo

```bash
# 타입체크 베이스라인 확인 (에러 증가 금지)
.venv/bin/python .vibe/brain/typecheck_baseline.py
```

// turbo

```bash
# Pre-commit 체크 (staged 파일만)
.venv/bin/python .vibe/brain/precommit.py
```

---

## 4. 선택적 검증

```bash
# 전체 정밀 점검
.venv/bin/python .vibe/brain/doctor.py --full
```

```bash
# 성능 의심 시 프로파일링
.venv/bin/python .vibe/brain/doctor.py --full --profile --mode node --entry <entry.js>
```

---

## 체크리스트

작업 완료 전 확인:

- [ ] LATEST_CONTEXT.md 읽음
- [ ] DONT_DO_THIS.md 읽음
- [ ] impact_analyzer 실행함 (해당 시)
- [ ] typecheck baseline 통과
- [ ] precommit 통과
