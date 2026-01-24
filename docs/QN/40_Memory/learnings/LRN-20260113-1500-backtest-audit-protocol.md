---
id: LRN-20260113-1500-backtest-audit-protocol
date: 2026-01-13
tags: [learning, backtest, audit, methodology]
related_ki: strategy_200tqq_implementation
---

# Backtest Integrity Audit Protocol (BIAP)

## 6-Point Verification Framework

200TQQ 감사 과정에서 표준화된 백테스트 검증 프로토콜.

### [0] Bundle Consistency

- Config SHA256 해시로 실행 인스턴스 일치 확인
- `summary_metrics.json`, `daily.csv`, `trades.csv` 동일 실행 검증

### [1] Proxy/Synthetic Definition

- 합성 데이터 생성 수식 명시
- 전환점 (Transition Date) 스케일링 팩터 기록

### [2] Synthetic Validation

- 실제 데이터 vs 합성 데이터 상관관계 검증
- Tracking Error 임계값: < 5%

### [3] Split/Dividend Accountability

- 스플릿 날짜 equity jump 검증
- 거래 없는 날 equity 불연속 = 0% 확인

### [4] Rule Implementation Audit

- Signal timing: T_close 생성 → T+1_open 체결
- Stop-loss, Take-profit 논리 샘플 검증

### [5] Final Verdict

- TRUSTED / UNTRUSTED 판정
- Benchmark sanity check (Buy & Hold 대비)

## Takeaway

> 모든 장기 백테스트는 BIAP 6단계를 통과해야 공식 결과로 인정.
