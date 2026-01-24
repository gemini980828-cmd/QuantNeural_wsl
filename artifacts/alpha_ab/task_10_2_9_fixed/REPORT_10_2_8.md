# Task 10.2.8 — Attribution Split Report

## Subset Comparison

| Subset | IC Mean | Delta CAGR/Vol (XGB) | Delta CAGR/Vol (XGB_INV) | Mean Universe | Fraction |
|--------|---------|----------------------|--------------------------|---------------|----------|
| all | 0.0205 | +0.1448 | +0.1986 | 2457 | 100.0% |
| sec_covered | N/A | +0.1181 | +0.1676 | 534 | 21.7% |
| sec_missing | N/A | +0.1849 | +0.1183 | 1923 | 78.3% |

## Verdict

- **Both subsets show improvement**: The model provides value across both SEC-covered and SEC-missing universes.
- **Warning**: SEC-COVERED subset had 3 dates with universe < top_k (top_k adapted).
