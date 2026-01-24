---
id: LRN-20260113-1200-synthetic-tqqq-split-handling
date: 2026-01-13
tags: [learning, backtest, data-quality, splits]
related_ki: strategy_200tqq_implementation
---

# Synthetic TQQQ & Adaptive Split Handling

## Problem Discovered

200TQQ 장기 백테스트 (1999-2024) 검증 중 초기 결과가 비현실적으로 높은 CAGR (2200%+)을 보임.

## Root Cause

**Double-Count Bug**: Split-adjusted 가격 (Path B)을 사용하면서 동시에 주식 수 조정 로직 (Path A)을 적용.

- TQQQ의 누적 스플릿 배수 (96x)로 인해 equity가 7.47배 과대 계상됨.

## Solution

**Adaptive Split Handling Protocol** 도입:

1. `split_ratio` 컬럼 확인
2. `split_ratio != 1.0`일 때만 주식 수 조정
3. Yahoo adjusted 데이터는 `split_ratio = 1.0`으로 안전 처리

## Validation

- Synthetic TQQQ (Method B) vs Real TQQQ: **Correlation 0.9989**, TE 2.87%
- Path C (Raw + Share Adj) ↔ Path B (Adjusted) 교차 검증 완료

## Takeaway

> Split 처리는 반드시 **하나의 경로**만 선택. 혼용 시 치명적 오류 발생.
