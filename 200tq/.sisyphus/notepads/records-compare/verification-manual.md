# Manual E2E Verification Instructions

## TODO 7: E2E Verification (Manual)

Since automated Playwright verification encountered technical issues, please perform manual verification:

### Prerequisites
1. Start the development server:
   ```bash
   cd dashboard
   bun run dev
   ```
2. Ensure Supabase is running and accessible
3. Open browser to http://localhost:3000

### Verification Steps

#### Step 1: Command Page Test
1. Navigate to http://localhost:3000/command
2. Wait for data to load (check viewModel populates)
3. Verify no console errors
4. Look for "기록 저장" button
5. Click the button and save a trade record
6. **Check Network Tab** in DevTools:
   - Look for POST request to `/api/record`
   - Verify request body contains `expectedLines` array
   - Example:
     ```json
     {
       "executionDate": "2026-01-27",
       "executed": true,
       "lines": [...],
       "expectedLines": [
         {
           "symbol": "TQQQ",
           "side": "BUY",
           "qty": 100,
           "expectedPrice": 85.23
         }
       ]
     }
     ```

#### Step 2: Database Verification
1. Open Supabase SQL Editor
2. Run query:
   ```sql
   SELECT execution_date, expected_lines, lines 
   FROM trade_executions 
   ORDER BY execution_date DESC 
   LIMIT 5;
   ```
3. Verify `expected_lines` column contains data (not null)
4. Example expected result:
   ```json
   [
     {
       "symbol": "TQQQ",
       "side": "BUY", 
       "qty": 100,
       "expectedPrice": 85.23
     }
   ]
   ```

#### Step 3: Records Page Test
1. Navigate to http://localhost:3000/records
2. Wait for records to load
3. Locate the **QualityAnalytics** section (3 cards: 체결 정확도, 평균 슬리피지, 지연 기록)
4. **Verify the values are NOT hardcoded**:
   - ❌ NOT 98.5% accuracy
   - ❌ NOT 0.12% slippage
   - ✅ Should show calculated values OR 0% if no comparison data
5. Check browser console - verify no errors
6. Take screenshot and save to `.sisyphus/evidence/records-compare-e2e.png`

### Expected Results

**If no records with expected_lines exist:**
- Accuracy: 0%
- Slippage: 0%
- (This is correct - no data to compare)

**If records with expected_lines exist:**
- Accuracy: Calculated percentage (e.g., 95.5%, 100%, etc.)
- Slippage: Calculated percentage (e.g., 0.15%, 1.2%, etc.)
- Values should change based on actual data

### Success Criteria
- [ ] Command page loads without errors
- [ ] POST /api/record includes expectedLines in request body
- [ ] Supabase shows expected_lines saved in database
- [ ] Records page shows calculated values (not 98.5% / 0.12%)
- [ ] No console errors in browser
- [ ] Screenshot saved to evidence folder

### Troubleshooting

**Issue: expectedLines not in request body**
- Check: ZoneCOpsConsole.tsx line 91 passes vm.expectedTrades
- Check: vm.inputPrices exists in viewModel
- Solution: Verify TODO 5 changes applied correctly

**Issue: expected_lines null in database**
- Check: API route accepts expectedLines parameter
- Check: Storage layer conversion logic working
- Solution: Verify TODOs 3 & 4 changes applied correctly

**Issue: Still shows 98.5% / 0.12%**
- Check: Records page comparison logic implemented
- Check: Browser cache cleared (hard refresh: Ctrl+Shift+R)
- Solution: Verify TODO 6 changes applied correctly

---

## Manual Verification Report Template

After completing manual verification, document results here:

```markdown
## [DATE TIME] Manual E2E Verification Results

**Tester**: [Your Name]
**Environment**: [Dev/Staging]

### Command Page
- Status: [PASS/FAIL]
- Notes: [Any observations]

### Network Request
- expectedLines present: [YES/NO]
- Sample data: [Paste request body]

### Database
- expected_lines saved: [YES/NO]
- Sample data: [Paste query result]

### Records Page
- Accuracy shown: [X.XX%]
- Slippage shown: [X.XX%]
- Console errors: [NONE/List errors]
- Screenshot: [Path or attached]

### Overall Result: [PASS/FAIL]
```
