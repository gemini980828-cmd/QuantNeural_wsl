---
id: DEC-20260113-1200-200tqq-audit-trusted
date: 2026-01-13
tags: [decision, backtest, strategy, audit]
related_ki: strategy_200tqq_implementation
---

# 200TQQ 25년 백테스트 TRUSTED 판정

## Decision

200TQQ 전략의 1999-2024 장기 백테스트 결과를 **공식 검증(TRUSTED)**으로 확정.

## Verified Metrics

| Metric     | Value            |
| ---------- | ---------------- |
| **CAGR**   | 10.52%           |
| **MDD**    | -42.8%           |
| **Period** | 1999-2024 (25yr) |

## Audit Protocol (6-Point BIAP)

1. ✅ **Bundle Consistency**: Config SHA256 일치 확인
2. ✅ **Synthetic Validation**: Method B correlation 0.9989
3. ✅ **Split Accountability**: 모든 스플릿 날짜 equity jump 검증
4. ✅ **Rule Implementation**: Signal timing, stop-loss, take-profit 확인
5. ✅ **Benchmark Sanity**: BH TQQQ (0.85%), SPY (6.5%) 대비 합리적
6. ✅ **Proxy Definition**: SPLG/BIL 1999-2012 합성 명세 문서화

## Implications

- 200TQQ 전략은 ML Reranker 대비 벤치마크로 사용 가능
- 장기 성과: Buy & Hold TQQQ (-96% MDD) 대비 우수한 리스크 관리
