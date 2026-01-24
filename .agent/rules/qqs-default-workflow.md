---
description: QQS Default Workflow - 모든 작업에 KIRA 5-Agent 패턴 자동 적용
alwaysApply: true
---

# QQS Default Workflow (Always On)

모든 비-QUICK_ANSWER 작업에 5-Agent 패턴을 암묵적으로 적용:
1. Router (내부 판단) - 요청 유형 분류
2. Validator - PIT 규칙, 설계 검토 (코드 변경 시)
3. Implementer - 코드 작성/실행
4. Analyst - 결과 검증 (ETL/분석 시)
5. Memory Manager - 작업 완료 후 기록

## 요청 유형 분류
- QUICK_ANSWER: 뭐야, 어디, 설명해줘  바로 답변
- IMPLEMENTATION: 구현, 추가, 수정  Validator  Implementer
- ETL: 로드, 업데이트, 백필  Validator  Implementer  Analyst
- ANALYSIS: 분석, 백테스트  Analyst
- VERIFICATION: 검증, 확인  Validator

## 작업 완료 시 Memory Manager
- Decision (최대 1개): 중요 설계 결정  40_Memory/decisions/
- Learning (최대 1개): 버그/해결책 발견  40_Memory/learnings/
- Context 업데이트: 40_Memory/context/CTX-project.md
