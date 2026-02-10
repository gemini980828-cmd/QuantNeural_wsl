# E03 Spreadsheet SSOT — Operations Tool Blueprint

**Status**: ✅ ACTIVE  
**Version**: v2026.3  
**Authority**: This document specifies the Google Sheets implementation of `200tq/E03_SSOT.md`  
**Last Updated**: 2026-02-10  
**Implementation**: `200tq/sheets/e03_sheet_builder.gs`

---

## Overview

### Purpose
Daily operations spreadsheet for the E03 v2026.3 trading strategy. Provides real-time signal monitoring, emergency detection, trade logging, and portfolio tracking using GOOGLEFINANCE() for automatic price updates.

### 6-Tab Architecture

| Tab | Emoji | Purpose |
|-----|-------|---------|
| **Dashboard** | 📊 | Daily ops command center — verdict, action, portfolio summary |
| **Signal** | 📈 | Ensemble voting, F1 FlipCount tracking, state determination |
| **Emergency** | 🚨 | Crash detection (QQQ -5%, TQQQ -15%), cooldown tracking |
| **TradeLog** | 📝 | Manual trade recording with automatic commission calculation |
| **Portfolio** | 💼 | Holdings, value, weight, unrealized P&L, recommended trades |
| **Settings** | ⚙️ | Strategy constants, initial positions, live GOOGLEFINANCE feeds |

### Data Flow Diagram

```
┌─────────────┐
│  Settings   │──┐ (Constants: SMA windows, F1 params, Emergency thresholds)
└─────────────┘  │
                 │
┌─────────────┐  │
│ PriceData   │  │ (Hidden tab: GOOGLEFINANCE QQQ history from 2025-01-01)
│  (Hidden)   │──┼──┐
└─────────────┘  │  │
                 │  ▼
┌─────────────┐  │  ┌─────────────┐
│  TradeLog   │  ├─▶│   Signal    │◀─┐ (Calculates SMA, Ensemble vote, FlipCount, State)
└─────────────┘  │  └─────────────┘  │
      │          │         │          │
      │          │         ▼          │
      ▼          │  ┌─────────────┐  │
┌─────────────┐  ├─▶│  Emergency  │──┘ (QQQ/TQQQ crash triggers, Cooldown status)
│  Portfolio  │  │  └─────────────┘
└─────────────┘  │         │
      │          │         │
      └──────────┼─────────┘
                 ▼
           ┌─────────────┐
           │  Dashboard  │ (Aggregates: Verdict, F1, Emergency, Action, Portfolio)
           └─────────────┘
```

---

## Tab Specifications

### Tab 1: ⚙️ Settings

**Purpose**: Centralized configuration — strategy constants, portfolio initialization, live price feeds.

**Column Structure**:
| Row | Col A (Label) | Col B (Value) | Type | Notes |
|-----|---------------|---------------|------|-------|
| 1 | Header | Header | — | Bold, gray background |
| 2-13 | **Strategy Constants** | — | Section | — |
| 2 | SMA Window 1 | 160 | Constant | Named: `CFG_SMA_WIN1` |
| 3 | SMA Window 2 | 165 | Constant | Named: `CFG_SMA_WIN2` |
| 4 | SMA Window 3 | 170 | Constant | Named: `CFG_SMA_WIN3` |
| 5 | F1 Flip Window | 40 | Constant | Named: `CFG_F1_WINDOW` |
| 6 | F1 Flip Threshold | 3 | Constant | Named: `CFG_F1_THRESHOLD` |
| 7 | F1 Reduced Weight | 0.70 | Constant | Named: `CFG_F1_REDUCED` (% format) |
| 8 | Emergency QQQ Threshold | -0.05 | Constant | Named: `CFG_EMERGENCY_QQQ` (% format) |
| 9 | Emergency TQQQ Threshold | -0.15 | Constant | Named: `CFG_EMERGENCY_TQQQ` (% format) |
| 10 | OFF Residual | 0.10 | Constant | Named: `CFG_OFF_RESIDUAL` (% format) |
| 11 | Commission Rate | 0.001 | Constant | Named: `CFG_COMMISSION` |
| 12 | Tax Rate | 0.22 | Constant | Named: `CFG_TAX` |
| 14-19 | **Portfolio Initial** | — | Section | User input required |
| 15 | TQQQ Qty | [user input] | Manual | Named: `CFG_TQQQ_QTY` |
| 16 | TQQQ Avg Entry | [user input] | Manual | Named: `CFG_TQQQ_ENTRY` ($ format) |
| 17 | SGOV Qty | [user input] | Manual | Named: `CFG_SGOV_QTY` |
| 18 | SGOV Avg Entry | [user input] | Manual | Named: `CFG_SGOV_ENTRY` ($ format) |
| 19 | Cash Balance KRW | [user input] | Manual | Named: `CFG_CASH_KRW` (₩ format) |
| 21 | **Live Data** | — | Section | — |
| 23 | QQQ Price | `=IFERROR(GOOGLEFINANCE("QQQ","price"),"")` | Auto | Named: `LIVE_QQQ` |
| 24 | TQQQ Price | `=IFERROR(GOOGLEFINANCE("TQQQ","price"),"")` | Auto | Named: `LIVE_TQQQ` |
| 25 | SGOV Price | `=IFERROR(GOOGLEFINANCE("SGOV","price"),"")` | Auto | Named: `LIVE_SGOV` |
| 26 | USD/KRW | `=IFERROR(GOOGLEFINANCE("CURRENCY:USDKRW"),"")` | Auto | Named: `LIVE_USDKRW` |

**Named Ranges**: 15 total (all CFG_* and LIVE_* cells)

**Tab Dependencies**:
- **Read by**: Signal, Emergency, Portfolio, Dashboard
- **Writes**: Manual user input for rows 15-19 only

---

### Tab 2: 📊 PriceData (Hidden)

**Purpose**: GOOGLEFINANCE historical data feed for QQQ (2025-01-01 to TODAY).

**Column Structure**:
| Col | Name | Formula |
|-----|------|---------|
| A | Date | Auto (from GOOGLEFINANCE) |
| B | QQQ Close | `=IFERROR(GOOGLEFINANCE("QQQ","close",DATE(2025,1,1),TODAY(),"DAILY"),"")` |

**Notes**:
- Sheet is **hidden** after creation
- Provides ~400 days of data (signal tab rows 2-400 reference this)
- Auto-updates when sheet recalculates (Ctrl+R or reopen)

---

### Tab 3: 📈 Signal

**Purpose**: Core ensemble logic — SMA calculations, majority voting, F1 FlipCount, state determination.

**Column Structure** (Rows 2-400, headers in row 1):
| Col | Name | Formula | Description |
|-----|------|---------|-------------|
| A | Date | `='📊 PriceData'!A[row]` | From PriceData tab |
| B | QQQ Close | `='📊 PriceData'!B[row]` | From PriceData tab |
| C | SMA3 | `=IFERROR(AVERAGE(OFFSET(B[row],0,0,-3,1)),"")` | 3-day simple moving average |
| D | SMA160 | `=IFERROR(AVERAGE(OFFSET(B[row],0,0,-CFG_SMA_WIN1,1)),"")` | 160-day SMA |
| E | SMA165 | `=IFERROR(AVERAGE(OFFSET(B[row],0,0,-CFG_SMA_WIN2,1)),"")` | 165-day SMA |
| F | SMA170 | `=IFERROR(AVERAGE(OFFSET(B[row],0,0,-CFG_SMA_WIN3,1)),"")` | 170-day SMA |
| G | Vote160 | `=IF(C[row]>D[row],"PASS","FAIL")` | SMA3 > SMA160 (strict >) |
| H | Vote165 | `=IF(C[row]>E[row],"PASS","FAIL")` | SMA3 > SMA165 |
| I | Vote170 | `=IF(C[row]>F[row],"PASS","FAIL")` | SMA3 > SMA170 |
| J | Ensemble | `=IF(COUNTIF(G[row]:I[row],"PASS")>=2,"ON","OFF")` | Majority vote (≥2/3) |
| K | FlipCount | `=SUMPRODUCT(--(...))` | Signal flips in past 40 days (see Formula Reference) |
| L | State | Complex IFS | EMERGENCY / OFF10 / ON-CHOPPY / ON (see below) |
| M | Target TQQQ% | `=IFS(L[row]="ON",1, L[row]="ON-CHOPPY",CFG_F1_REDUCED, ...)` | 100% / 70% / 10% / 10% |
| N | FlipCount Valid | `=IF(ROW()-1<CFG_F1_WINDOW, CFG_F1_WINDOW-(ROW()-1)&" days left","VALID")` | Cold start indicator |

**State Formula (Col L)**:
```
=IF(J[row]="","",
   IF('🚨 Emergency'!I[row]="🔴 ACTIVE","EMERGENCY",
      IF(J[row]="OFF","OFF10",
         IF(AND(J[row]="ON",K[row]>=CFG_F1_THRESHOLD),"ON-CHOPPY","ON")
      )
   )
)
```

**Conditional Formatting**:
- **Vote Columns (G:I)**: "PASS" → Green (`#34A853`), "FAIL" → Red (`#EA4335`)
- **Ensemble (J)**: "ON" → Green, "OFF" → Red
- **State (L)**: 
  - "ON" → Dark Green (`#0F9D58`)
  - "ON-CHOPPY" → Yellow (`#FBBC04`)
  - "OFF10" → Red (`#EA4335`)
  - "EMERGENCY" → Purple (`#9C27B0`)
- **FlipCount (K)**: ≥3 → Yellow background

**Number Formats**:
- B:F → `$#,##0.00` (currency)
- M → `0.00%` (percent)

**Tab Dependencies**:
- **Reads from**: PriceData, Settings (named ranges), Emergency (col I)
- **Read by**: Dashboard, Emergency, Portfolio

---

### Tab 4: 🚨 Emergency

**Purpose**: Real-time crash detection (QQQ -5%, TQQQ -15% from entry), cooldown tracking.

**Column Structure** (Rows 2-400):
| Col | Name | Formula | Description |
|-----|------|---------|-------------|
| A | Date | `='📈 Signal'!A[row]` | From Signal tab |
| B | QQQ Close | `='📈 Signal'!B[row]` | From Signal tab |
| C | QQQ Daily Return | `=(B[row]-B[row-1])/B[row-1]` | Day-over-day % change |
| D | Crash Trigger? | `=IF(C[row]<=CFG_EMERGENCY_QQQ,"🚨 TRIGGER","✅ SAFE")` | -5% threshold |
| E | TQQQ Current | `=LIVE_TQQQ` | From Settings live data |
| F | TQQQ Entry (Avg) | `=CFG_TQQQ_ENTRY` | From Settings portfolio |
| G | TQQQ Drawdown% | `=(E[row]-F[row])/F[row]` | Current DD from entry |
| H | Stop Trigger? | `=IF(G[row]<=CFG_EMERGENCY_TQQQ,"🚨 TRIGGER","✅ SAFE")` | -15% threshold |
| I | Emergency Status | `=IF(OR(D[row]="🚨 TRIGGER",H[row]="🚨 TRIGGER"),"🔴 ACTIVE","🟢 NONE")` | Combined status |
| J | Cooldown | `=IF(I[row-1]="🔴 ACTIVE","COOLDOWN","CLEAR")` | 1-day cooldown tracker |

**Conditional Formatting**:
- **Trigger Columns (D, H)**: "🚨 TRIGGER" → Red background + white text
- **Emergency Status (I)**: "🔴 ACTIVE" → Purple background

**Number Formats**:
- C, G → `0.00%`
- E, F → `$#,##0.00`

**Tab Dependencies**:
- **Reads from**: Signal, Settings
- **Read by**: Signal (col I), Dashboard

---

### Tab 5: 📝 TradeLog

**Purpose**: Manual trade recording with automatic USD/KRW/commission calculations.

**Column Structure**:
| Col | Name | Input Type | Formula/Validation | Description |
|-----|------|------------|-------------------|-------------|
| A | Date | Manual | — | Trade execution date |
| B | Ticker | Manual | Dropdown: `["TQQQ", "SGOV"]` | Asset ticker |
| C | Action | Manual | Dropdown: `["BUY", "SELL", "HOLD"]` | Trade action |
| D | Shares | Manual | Validation: > 0, integer | Number of shares |
| E | Price USD | Manual | Validation: > 0 | Execution price |
| F | Total USD | Auto | `=D[row]*E[row]` | Gross amount |
| G | USD/KRW | Auto | `=LIVE_USDKRW` | Exchange rate (from Settings) |
| H | Total KRW | Auto | `=F[row]*G[row]` | KRW equivalent |
| I | Commission | Auto | `=F[row]*CFG_COMMISSION` | 10bps (0.1%) |
| J | Signal State | Manual | — | State at time of trade (ON/CHOPPY/OFF10) |
| K | Note | Manual | — | Free-form memo |

**Data Validation**:
- **Col B**: `requireValueInList(["TQQQ", "SGOV"])`
- **Col C**: `requireValueInList(["BUY", "SELL", "HOLD"])`
- **Col D**: `requireNumberGreaterThan(0)`, integer only
- **Col E**: `requireNumberGreaterThan(0)`

**Number Formats**:
- E, F, I → `$#,##0.00`
- G → `₩#,##0.00`
- H → `₩#,##0`

**Tab Dependencies**:
- **Reads from**: Settings (LIVE_USDKRW, CFG_COMMISSION)
- **Read by**: Portfolio (for trade aggregation)

---

### Tab 6: 💼 Portfolio

**Purpose**: Current holdings, market value, weight vs target, unrealized P&L, recommended trades.

**Row Structure** (Fixed 3 rows + header):
| Row | Ticker | Qty | Avg Entry (USD) | Current Price (USD) | Value (USD) | Value (KRW) | Weight % | Target % | Deviation % | Unrealized PnL (USD) | Daily PnL (USD) | Recommended Trade |
|-----|--------|-----|-----------------|---------------------|-------------|-------------|----------|----------|-------------|----------------------|-----------------|-------------------|
| 2 | TQQQ | `=CFG_TQQQ_QTY` | `=CFG_TQQQ_ENTRY` | `=LIVE_TQQQ` | `=B2*D2` | `=E2*LIVE_USDKRW` | `=E2/SUM(E:E)` | Latest Signal state | `=G2-H2` | `=(D2-C2)*B2` | `=(D2-prev_D)*B2` | See formula below |
| 3 | SGOV | `=CFG_SGOV_QTY` | `=CFG_SGOV_ENTRY` | `=LIVE_SGOV` | `=B3*D3` | `=E3*LIVE_USDKRW` | `=E3/SUM(E:E)` | Latest Signal state | `=G3-H3` | `=(D3-C3)*B3` | `=(D3-prev_D)*B3` | See formula below |
| 4 | CASH | `=CFG_CASH_KRW/LIVE_USDKRW` | — | 1 | `=B4` | `=CFG_CASH_KRW` | `=E4/SUM(E:E)` | — | — | — | — | — |
| 5 | **TOTAL** | — | — | — | `=SUM(E2:E4)` | `=SUM(F2:F4)` | 100% | — | — | `=SUM(J2:J3)` | `=SUM(K2:K3)` | — |

**Recommended Trade Formula**:
```
IF(Target_TQQQ% = Current_Weight%,
   "HOLD",
   IF(Target_TQQQ% > Current_Weight%,
      "BUY " & CEILING((Target% - Current%) * TotalValue / TQQQ_Price) & " TQQQ",
      "SELL " & FLOOR((Current% - Target%) * TotalValue / TQQQ_Price - CEILING(TotalQty*0.10)) & " TQQQ"
   )
)
```
Notes:
- Uses `CEILING()` for 10% residual calculation (E03_SSOT.md Part 5.3)
- Target % comes from Signal tab latest State (ON=100%, Choppy=70%, OFF10=10%)

**Number Formats**:
- C, D, E, J, K → `$#,##0.00`
- F → `₩#,##0`
- G, H, I → `0.00%`

**Tab Dependencies**:
- **Reads from**: Settings (CFG_*, LIVE_*), Signal (latest State), TradeLog (optional aggregation)
- **Read by**: Dashboard

---

### Tab 7: 📊 Dashboard

**Purpose**: Daily ops command center — single-screen view of verdict, evidence, emergency, action, portfolio.

**Layout** (Zones):

#### Zone A: Header (Rows 1-3)
| Cell | Content | Formula |
|------|---------|---------|
| A1 | "E03 Strategy Dashboard" | Static |
| A2 | "As of:" | Static |
| B2 | [Today's Date] | `=TEXT(TODAY(),"YYYY-MM-DD")` |
| A3 | "Data Status:" | Static |
| B3 | FRESH / STALE | `=IF(ISBLANK(LIVE_QQQ),"❌ STALE","✅ FRESH")` |

#### Zone B: Verdict (Rows 5-7)
| Cell | Content | Formula | Formatting |
|------|---------|---------|------------|
| A5 | "VERDICT" | Static | Bold, 14pt |
| A6 | [Current State] | `=LOOKUP(2,1/('📈 Signal'!$L$2:$L$400<>""),'📈 Signal'!$L$2:$L$400)` | **24pt, bold, colored background** |
| A7 | [Target TQQQ%] | `=LOOKUP(2,1/('📈 Signal'!$M$2:$M$400<>""),'📈 Signal'!$M$2:$M$400)` | 18pt, percent format |

**Conditional Formatting for A6**:
- "ON" → Green background
- "ON-CHOPPY" → Yellow background
- "OFF10" → Red background
- "EMERGENCY" → Purple background

#### Zone C: Evidence (Rows 9-13)
| Row | Label | Value Formula |
|-----|-------|---------------|
| 9 | "SMA3" | Latest from Signal tab |
| 10 | "SMA160 / 165 / 170" | Latest 3 values |
| 11 | "Vote Results" | `=COUNTIF(Signal!G_latest:I_latest,"PASS")&"/3 PASS"` |
| 12 | "Margin %" | Smallest margin among 3 windows |

#### Zone D: F1 Filter (Rows 15-17)
| Row | Label | Value Formula |
|-----|-------|---------------|
| 15 | "FlipCount (40d)" | `=Signal!K_latest` |
| 16 | "Threshold" | `=CFG_F1_THRESHOLD` |
| 17 | "Status" | `=Signal!N_latest` (VALID or "X days left") |

#### Zone E: Emergency (Rows 19-22)
| Row | Label | Value Formula |
|-----|-------|---------------|
| 19 | "QQQ Daily Return" | `=Emergency!C_latest` |
| 20 | "TQQQ Drawdown" | `=Emergency!G_latest` |
| 21 | "Emergency Status" | `=Emergency!I_latest` |
| 22 | "Cooldown" | `=Emergency!J_latest` |

#### Zone F: Action (Rows 24-26)
| Row | Label | Value Formula |
|-----|-------|---------------|
| 24 | "📋 Recommended Trade" | Static header |
| 25 | [Trade instruction] | `=Portfolio!L2` (TQQQ recommended trade) |
| 26 | [SGOV instruction] | `=Portfolio!L3` (SGOV recommended trade) |

#### Zone G: Portfolio Summary (Rows 28-32)
| Row | Label | Value Formula |
|-----|-------|---------------|
| 28 | "Total Value (USD)" | `=Portfolio!E5` |
| 29 | "Total Value (KRW)" | `=Portfolio!F5` |
| 30 | "TQQQ Weight" | `=Portfolio!G2` |
| 31 | "Unrealized P&L" | `=Portfolio!J5` |
| 32 | "Daily P&L" | `=Portfolio!K5` |

**Tab Position**: Moved to index 1 (first tab) via `moveActiveSheet(1)`

**Tab Dependencies**:
- **Reads from**: Signal, Emergency, Portfolio, Settings
- **Read by**: User (primary interface)

---

## Formula Reference

### SMA Calculation (OFFSET pattern)
```
=IFERROR(AVERAGE(OFFSET(B[row], 0, 0, -[window], 1)), "")
```
- `OFFSET(B[row], 0, 0, -window, 1)`: Creates range from B[row-window+1]:B[row]
- `AVERAGE(...)`: Computes mean
- `IFERROR(..., "")`: Returns blank if insufficient data

### Ensemble Vote (COUNTIF majority)
```
=IF(COUNTIF(G[row]:I[row], "PASS") >= 2, "ON", "OFF")
```
- Counts "PASS" in 3 vote columns
- ≥2 → "ON", else "OFF"

### FlipCount (SUMPRODUCT pattern)
```
=IF(ROW()-1 < CFG_F1_WINDOW, "",
   SUMPRODUCT(--(
      OFFSET(J[row], -CFG_F1_WINDOW+1, 0, CFG_F1_WINDOW-1, 1)
      <>
      OFFSET(J[row], -CFG_F1_WINDOW+2, 0, CFG_F1_WINDOW-1, 1)
   ))
)
```
- Compares each pair of adjacent Ensemble values in 40-day window
- Counts how many differ (`<>`)
- Returns blank if row < 40

### Emergency Detection (OR logic)
```
=IF(OR(D[row]="🚨 TRIGGER", H[row]="🚨 TRIGGER"), "🔴 ACTIVE", "🟢 NONE")
```
- D: QQQ crash (daily return ≤ -5%)
- H: TQQQ stop (DD from entry ≤ -15%)
- Either triggers → ACTIVE

### Target Weight (IFS switch)
```
=IFS(
  L[row]="ON", 1,
  L[row]="ON-CHOPPY", CFG_F1_REDUCED,
  L[row]="OFF10", CFG_OFF_RESIDUAL,
  L[row]="EMERGENCY", CFG_OFF_RESIDUAL,
  TRUE, ""
)
```
- Maps State → Target TQQQ%: ON=100%, Choppy=70%, OFF10=10%, Emergency=10%

### 10% Residual (CEILING)
```
=CEILING(TotalQty * CFG_OFF_RESIDUAL)
```
- E03_SSOT.md Part 5.3: "10% 잔류는 올림"
- Example: 137 shares → CEILING(13.7) = 14 shares residual

### GOOGLEFINANCE Wrapper (IFERROR)
```
=IFERROR(GOOGLEFINANCE("ticker", "attribute"), "")
```
- Graceful degradation: Returns blank if API fails
- All GOOGLEFINANCE calls wrapped for robustness

---

## User Guide

### 5.1 Initial Setup

**Step 1: Create New Google Sheet**
1. Go to sheets.google.com
2. Create blank spreadsheet
3. Name it "E03 Strategy Operations"

**Step 2: Install Apps Script**
1. Tools → Script Editor
2. Delete default `function myFunction() {}`
3. Paste entire contents of `200tq/sheets/e03_sheet_builder.gs`
4. Save (Ctrl+S), name it "E03 Sheet Builder"

**Step 3: Run Initialization**
1. Select function dropdown → `initializeE03Sheet`
2. Click ▶ Run button
3. Grant permissions when prompted:
   - ⚠️ "This app isn't verified" → Advanced → Go to [project] (unsafe)
   - Allow: See/edit/create/delete your spreadsheets
4. Wait ~10-15 seconds for script to complete
5. Confirmation dialog: "E03 spreadsheet initialized successfully"

**Result**: 6 tabs created, Dashboard tab active, all formulas set.

### 5.2 Initial Configuration (Settings Tab)

Navigate to ⚙️ Settings tab:

**Portfolio Initial (Rows 15-19) — REQUIRED INPUT:**
1. **TQQQ Qty** (B15): Enter current TQQQ shares (integer)
2. **TQQQ Avg Entry** (B16): Enter weighted average cost basis (USD)
3. **SGOV Qty** (B17): Enter current SGOV shares (integer)
4. **SGOV Avg Entry** (B18): Enter weighted average cost basis (USD)
5. **Cash Balance KRW** (B19): Enter current cash (KRW)

**Verify Live Data (Rows 23-26):**
- All 4 cells (QQQ, TQQQ, SGOV, USD/KRW) should show numbers
- If blank: Wait 10 seconds or press Ctrl+R to refresh
- If still blank: Check GOOGLEFINANCE API status

**Strategy Constants (Rows 2-12) — DO NOT MODIFY** unless changing strategy:
- SMA windows: 160, 165, 170
- F1 params: Window=40, Threshold=3, Weight=0.70
- Emergency: QQQ=-5%, TQQQ=-15%
- OFF Residual: 10%

### 5.3 Daily Operations Workflow

**Maps to E03_SSOT.md Part 5.1 Daily Ops Checklist:**

1. **Open Dashboard Tab** (📊 Dashboard)
2. **Check Data Status** (Row 3): Must show "✅ FRESH"
   - If "❌ STALE": Press Ctrl+R or reopen sheet
3. **Read Verdict** (Row 6): ON / ON-CHOPPY / OFF10 / EMERGENCY
   - Background color indicates state
4. **Check Emergency Status** (Zone E, Row 21):
   - "🟢 NONE": Normal operation
   - "🔴 ACTIVE": Emergency triggered → immediate action required
5. **Review F1 Filter** (Zone D):
   - FlipCount value (Row 15)
   - If "X days left" (Row 17): FlipCount not yet valid, ignore Choppy state
6. **Check Recommended Trade** (Zone F, Rows 25-26):
   - "HOLD": No action needed
   - "BUY X TQQQ": Execute market order (or limit near close)
   - "SELL Y TQQQ": Execute market order, keep CEILING(10%) residual
7. **Execute in Broker** (if trade recommended):
   - Place order in brokerage account
   - Note execution price
8. **Record Trade in TradeLog Tab** (📝 TradeLog):
   - Enter: Date, Ticker, Action, Shares, Price
   - Verify auto-calculated: Total USD, KRW, Commission
   - Note Signal State at time of trade
9. **Update Portfolio** (if manual tracking):
   - Settings tab → Update TQQQ/SGOV Qty and Avg Entry (if using manual method)
   - Or: Let TradeLog auto-aggregate (if implemented)

**Frequency**: Check once per day after US market close (before next open).

### 5.4 Recording Trades (TradeLog Tab)

**When to Record**: After every BUY or SELL execution in broker.

**Steps**:
1. Navigate to 📝 TradeLog tab
2. Find first empty row
3. Enter:
   - **Date** (Col A): Execution date (YYYY-MM-DD)
   - **Ticker** (Col B): Select from dropdown (TQQQ or SGOV)
   - **Action** (Col C): Select from dropdown (BUY, SELL, HOLD)
   - **Shares** (Col D): Number of shares (integer)
   - **Price USD** (Col E): Execution price per share
4. **Auto-calculated** (do not edit):
   - Total USD (Col F)
   - USD/KRW rate (Col G)
   - Total KRW (Col H)
   - Commission 10bps (Col I)
5. **Optional**:
   - **Signal State** (Col J): Copy from Dashboard Verdict at time of trade
   - **Note** (Col K): Free-form (e.g., "OFF10→ON transition", "Emergency exit")

**Example Entry**:
| Date | Ticker | Action | Shares | Price USD | Total USD | USD/KRW | Total KRW | Commission | Signal State | Note |
|------|--------|--------|--------|-----------|-----------|---------|-----------|------------|--------------|------|
| 2026-02-10 | TQQQ | SELL | 50 | 62.40 | 3,120.00 | 1,450 | 4,524,000 | 3.12 | OFF10 | ON→OFF10 transition |

---

## Limitations & Known Issues

### FlipCount Cold Start
- **Issue**: First 40 rows show "X days left" in FlipCount Valid column
- **Impact**: Cannot detect Choppy state until row 41+
- **Mitigation**: Signal tab starts from 2025-01-01, so by Feb 2025 FlipCount is valid
- **User Action**: Ignore "ON-CHOPPY" state during cold start period

### GOOGLEFINANCE Data Delays
- **Issue**: Weekends/holidays may show stale data
- **Impact**: Dashboard shows "❌ STALE" status
- **Mitigation**: Formulas use `IFERROR()` to avoid errors
- **User Action**: Manual refresh (Ctrl+R) or wait for next market day

### Manual Price Refresh Required
- **Issue**: GOOGLEFINANCE doesn't auto-update intraday
- **Impact**: Dashboard may show yesterday's close until refresh
- **Mitigation**: Recalculate sheet before making decisions
- **User Action**: Press Ctrl+R or reopen sheet to force refresh

### TQQQ Entry Price Manual Maintenance
- **Issue**: Portfolio weighted average entry price not auto-updated
- **Impact**: Emergency stop-loss uses outdated entry price
- **Mitigation**: User must manually update Settings tab after each buy
- **User Action**: After BUY trade, recalculate weighted avg: `(OldQty*OldEntry + NewQty*NewPrice) / (OldQty+NewQty)`

### Emergency Cooldown Manual Verification
- **Issue**: Cooldown formula checks previous day only
- **Impact**: If manually overriding, verify cooldown status
- **Mitigation**: Dashboard shows cooldown status in Zone E
- **User Action**: Wait 1 day after Emergency before re-entering ON state

### No Historical TradeLog Aggregation
- **Issue**: TradeLog is append-only, no auto-aggregation to Portfolio
- **Impact**: Portfolio Qty/Entry must be manually updated
- **Mitigation**: Settings tab serves as single source of truth
- **User Action**: Update Settings B15-B18 after each trade

### GOOGLEFINANCE API Limits
- **Issue**: Google may throttle/block excessive refreshes
- **Impact**: Cells show #N/A or blank
- **Mitigation**: Limit refreshes to 1-2 per hour during market hours
- **User Action**: If errors persist, wait 1 hour and try again

---

## SSOT Cross-Reference

**Authority Chain**: This spreadsheet implements the strategy defined in `200tq/E03_SSOT.md v2026.3`.

| E03_SSOT.md Section | Spreadsheet Tab | Column/Formula | Implementation Notes |
|---------------------|-----------------|----------------|----------------------|
| **Part 2: Layer 1 — Ensemble** | Signal | Cols D-F (SMA 160/165/170), Col J (Ensemble) | Windows: 160, 165, 170. Strict `>` comparison (not `>=`). Majority vote: ≥2/3 PASS. |
| **Part 2: Layer 2 — F1 Filter** | Signal | Col K (FlipCount), Col L (State), Col N (Valid) | Window: 40 days. Threshold: 3 flips. Reduced weight: 0.70. Cold start: "X days left". |
| **Part 2: Layer 3 — Emergency Exit** | Emergency | Cols D, H, I (Triggers), Col J (Cooldown) | QQQ threshold: -5%. TQQQ threshold: -15%. Cooldown: 1 day. Overrides ON state. |
| **Part 5.1: Daily Ops Checklist** | Dashboard | All zones (A-G) | 6-step workflow mapped: (1) Data status → (2) Verdict → (3) Emergency → (4) F1 → (5) Recommended trade → (6) Portfolio summary. |
| **Part 5.3: 10% Residual Calculation** | Portfolio | Col L (Recommended Trade) | Uses `CEILING()` for upward rounding. Example: 137 shares → 14 residual (not 13). |
| **Part 5.3: Execution Model** | Settings | Rows 11-12 (Commission, Tax) | Commission: 10bps (0.1%). Tax: 22%. TradeLog auto-calculates commission. |
| **Part 3.1: Position Allocation Table** | Signal | Col M (Target TQQQ%) | ON=100%, ON-CHOPPY=70%, OFF10=10%, EMERGENCY=10%. Settings rows 7, 10 store 0.70, 0.10. |
| **Part 4.1: Backtest Parameters** | Settings | Rows 2-10 (Strategy Constants) | SMA windows (160, 165, 170), F1 params (40, 3, 0.70), Emergency (-0.05, -0.15), OFF residual (0.10). |
| **Part 6: Historical Events** | Emergency | Cols C-D (QQQ Crash), Cols G-H (TQQQ Stop) | Real-time detection. Historical events (2020-03-12, 2022-02-24, etc.) would trigger if data present. |
| **Part 7.2: SGOV as OFF Asset** | Settings | Row 17-18 (SGOV Qty/Entry), TradeLog (Ticker dropdown) | SGOV hardcoded as OFF asset. Dashboard recommends SGOV buys on OFF10 transitions. |

**Version Consistency**: All constants match E03_SSOT.md v2026.3 exactly. Any deviation = implementation error.

**Testing Reference**: To verify correctness, compare Signal tab output on known dates (e.g., 2020-03-12 crash) against E03_SSOT.md Part 6 historical events.

---

## Appendix: Apps Script Execution

**File**: `200tq/sheets/e03_sheet_builder.gs` (953 lines)

**Main Functions**:
1. `initializeE03Sheet()` — Entry point, orchestrates all tab creation
2. `createSettingsTab()` — Creates ⚙️ Settings (constants, portfolio, live data)
3. `createPriceDataTab()` — Creates hidden 📊 PriceData (GOOGLEFINANCE history)
4. `createSignalTab()` — Creates 📈 Signal (SMA, voting, FlipCount, state)
5. `createEmergencyTab()` — Creates 🚨 Emergency (crash detection, cooldown)
6. `createTradeLogTab()` — Creates 📝 TradeLog (manual entry, data validation)
7. `createPortfolioTab()` — Creates 💼 Portfolio (holdings, P&L, recommendations)
8. `createDashboardTab()` — Creates 📊 Dashboard (daily ops UI)
9. `applyGlobalFormatting()` — Freezes headers, protects Settings constants

**Execution Time**: ~10-15 seconds (creates 6 tabs, 400 rows × 14 columns of formulas, ~20 conditional format rules, 15 named ranges).

**Permissions Required**:
- `https://www.googleapis.com/auth/spreadsheets` (read/write sheets)
- `https://www.googleapis.com/auth/script.container.ui` (show dialogs)

**Error Handling**: All GOOGLEFINANCE calls wrapped in `IFERROR()`. Named ranges checked before creation. Existing tabs detected before overwrite.

---

**END OF BLUEPRINT**

_For strategy details, see: `200tq/E03_SSOT.md`_  
_For implementation code, see: `200tq/sheets/e03_sheet_builder.gs`_  
_For questions, contact: QuantNeural Research_
