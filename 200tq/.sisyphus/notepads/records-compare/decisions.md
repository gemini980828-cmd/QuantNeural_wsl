# Architectural Decisions - Records Compare

## Decision 1: Save Expected Prices
**Choice**: YES - save `expectedPrice` field
**Reason**: Required for slippage calculation (|actual - expected| / expected × 100)
**Impact**: Adds one field to DB TradeLine type

## Decision 2: Implementation Approach
**Choice**: Single PR with all changes (DB + API + UI)
**Reason**: Feature is tightly coupled, hard to split meaningfully
**Impact**: Larger changeset but complete feature delivery

## Decision 3: Type System
**Choice**: Keep two separate TradeLine types (e03 vs DB)
**Reason**: Avoid large refactor, maintain UI/storage separation
**Impact**: Requires conversion function in storage layer

## Decision 4: Backward Compatibility
**Choice**: expected_lines is nullable (jsonb NULL)
**Reason**: Existing records don't have this data
**Impact**: UI must handle null gracefully (show 0% or N/A)

## Decision 5: Testing Strategy
**Choice**: Manual QA only (no automated tests)
**Reason**: No test infrastructure exists, user preference
**Impact**: Require thorough manual verification in TODO 7
