---
id: DEC-20260108-1041-ssot-tasks-lock
date: 2026-01-08
as_of_date: 2026-01-08
related_taskspec: "Task 7.4.2"
tags: [decision, documentation, ssot, contracts]
---

# Decision: SSOT_TASKS.md as Definitive Contract

## Context
As Tasks 7.0–7.4 (Real Data Integration, Black-Litterman, HP Leakage Guard) were completed, there was no single authoritative document specifying exact API contracts, I/O shapes, fail-safe behavior, and determinism rules.

## Decision
Created `docs/SSOT_TASKS.md` as the **definitive contract document** for:
- Non-Negotiables (PIT, train-only fit, no shuffle, reproducibility)
- 7 Black-Litterman public APIs with "never-crash" policy
- `BL_FAILSAFE:` parseable warning format
- X=(n,20) / Y=(n,10) shape contracts
- HP filter `leakage_guard=True`, `lookback=120` defaults

## Rationale
- Single source of truth prevents contradictory implementations
- Enforces pytest-gated workflow before PR merge
- Lock down contracts before Phase 8.x upgrades

## Consequences
- All future implementations MUST comply with SSOT_TASKS.md
- PLANS.md and DATA_CONTRACT.md link to SSOT_TASKS.md for detailed contracts
