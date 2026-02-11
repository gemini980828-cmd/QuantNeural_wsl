// ──────────────────────────────────────────────────────────
// E03 시트 빌더 — 자동 연속 실행 아키텍처
// ──────────────────────────────────────────────────────────
//
// 사용법: initE03() 한 번 실행 → 모든 단계 자동 완료
//
// 단계 1–8:  탭 생성 + 포맷 (GOOGLEFINANCE 제외)
// 단계 9–11: GOOGLEFINANCE 수식 삽입 (지연 실행)
//
// PropertiesService + ScriptApp.newTrigger()를 사용하여
// 4.5분 GAS 제한에 근접하면 자동으로 이어서 실행합니다.
// ──────────────────────────────────────────────────────────

var E03_TABS = [
  '⚙️ Settings',
  '📊 PriceData',
  '📈 Signal',
  '🚨 Emergency',
  '📝 TradeLog',
  '💼 Portfolio',
  '📊 Dashboard'
];

var E03_MAX_MS = 4.5 * 60 * 1000; // 270,000ms (4.5분)

// ── 오케스트레이션 ──────────────────────────────────────

/**
 * 단일 진입점. 기존 E03 탭을 삭제하고 모든 빌드 단계를 실행.
 * 4.5분 제한에 근접하면 시간 기반 트리거로 자동 이어서 실행.
 */
function initE03() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();

  // 이전 실행에서 남은 트리거/상태 정리
  _clearE03Triggers();

  var props = PropertiesService.getScriptProperties();
  props.setProperty('E03_STEP', '1');
  props.setProperty('E03_SS_ID', ss.getId());

  _processE03(ss, 1);
}

/**
 * 트리거 핸들러 — 시간 기반 트리거에 의해 호출되어
 * 저장된 단계부터 처리를 재개합니다.
 */
function continueE03() {
  var props = PropertiesService.getScriptProperties();
  var step = parseInt(props.getProperty('E03_STEP'), 10);
  var ssId = props.getProperty('E03_SS_ID');

  if (!ssId || isNaN(step)) {
    console.log('continueE03: 저장된 상태 없음 — 중단합니다.');
    _clearE03Triggers();
    return;
  }

  var ss = SpreadsheetApp.openById(ssId);
  _processE03(ss, step);
}

/**
 * 메인 처리 루프. 단계를 순차적으로 실행하며,
 * 각 단계 후 경과 시간을 확인합니다. 4.5분 예산을
 * 초과하면 진행 상황을 저장하고 1분 후 트리거를 예약합니다.
 *
 * 단계:
 *   1  deleteTargetTabs           (기존 탭 삭제)
 *   2  createSettingsTab          (GF 없음)
 *   3  createPriceDataTab         (GF 없음)
 *   4  createSignalTab            (신호 탭)
 *   5  createEmergencyTab         (비상 탭)
 *   6  createTradeLogTab          (거래 기록 탭)
 *   7  createPortfolioTab         (GF 없음)
 *   8  createDashboardTab + 글로벌 포맷
 *   9  _insertGF_Settings         (GF 4개)
 *  10  _insertGF_PriceData        (GF 1개 — 가장 무거움)
 *  11  _insertGF_Portfolio        (GF 2개)
 */
function _processE03(ss, startStep) {
  var t0 = Date.now();
  var props = PropertiesService.getScriptProperties();

  // 이 함수를 호출한 트리거 정리
  _clearE03Triggers();

  var step = startStep;
  var totalSteps = 11;

  while (step <= totalSteps) {
    console.log('E03 단계 ' + step + '/' + totalSteps + ' 시작…');

    switch (step) {
      case 1:
        deleteTargetTabs(ss, E03_TABS);
        // 삭제 후 ss 참조가 오래되므로 다시 가져옴
        ss = SpreadsheetApp.openById(ss.getId());
        break;
      case 2:  createSettingsTab(ss); break;
      case 3:  createPriceDataTab(ss); break;
      case 4:  createSignalTab(ss); break;
      case 5:  createEmergencyTab(ss); break;
      case 6:  createTradeLogTab(ss); break;
      case 7:  createPortfolioTab(ss); break;
      case 8:
        createDashboardTab(ss);
        applyGlobalFormatting(ss);
        break;
      case 9:  _insertGF_Settings(ss); break;
      case 10: _insertGF_PriceData(ss); break;
      case 11: _insertGF_Portfolio(ss); break;
    }

    console.log('E03 단계 ' + step + '/' + totalSteps + ' 완료.');
    step += 1;

    // 경과 시간 확인 — 초과 시 연속 실행 예약
    if (step <= totalSteps && (Date.now() - t0) >= E03_MAX_MS) {
      props.setProperty('E03_STEP', String(step));
      ScriptApp.newTrigger('continueE03')
        .timeBased()
        .after(60 * 1000)
        .create();
      console.log('E03 시간 제한 도달. 단계 ' + step + '에서 연속 실행을 예약했습니다.');
      return;
    }
  }

  // 모든 단계 완료 — 정리
  props.deleteProperty('E03_STEP');
  props.deleteProperty('E03_SS_ID');
  _clearE03Triggers();

  console.log('E03 시트 빌드 완료 — 총 ' + totalSteps + '단계 모두 완료.');
}

function _clearE03Triggers() {
  var triggers = ScriptApp.getProjectTriggers();
  var i;
  for (i = 0; i < triggers.length; i += 1) {
    if (triggers[i].getHandlerFunction() === 'continueE03') {
      ScriptApp.deleteTrigger(triggers[i]);
    }
  }
}

// ── 헬퍼 함수 ──────────────────────────────────────────

function safeGF(ticker, attr) {
  return '=IFERROR(GOOGLEFINANCE("' + ticker + '","' + attr + '"),"")';
}

function safeGFHistory(ticker, attr, startDate) {
  return '=IFERROR(GOOGLEFINANCE("' + ticker + '","' + attr + '",' + startDate + ',TODAY(),"DAILY"),"")';
}

function getSheetOrThrow(ss, sheetName) {
  var sheet = ss.getSheetByName(sheetName);
  if (!sheet) {
    throw new Error('시트를 찾을 수 없음: ' + sheetName);
  }
  return sheet;
}

function setHeaderRow(sheet, headers) {
  var headerValues = [headers];
  var range = sheet.getRange(1, 1, 1, headers.length);
  range.setValues(headerValues);
  range.setFontWeight('bold');
  range.setBackground('#E8EAED');
  range.setHorizontalAlignment('center');
}

function setSectionTitle(sheet, a1, title) {
  var range = sheet.getRange(a1);
  range.setValue(title);
  range.setFontWeight('bold');
  range.setBackground('#F1F3F4');
}

function setNamedRangeSafe(ss, name, range) {
  var existing = ss.getRangeByName(name);
  if (existing) {
    ss.removeNamedRange(name);
  }
  ss.setNamedRange(name, range);
}

function deleteTargetTabs(ss, tabNames) {
  // 스프레드시트를 다시 가져와 최신 시트 참조 확보
  ss = SpreadsheetApp.openById(ss.getId());
  var i;
  for (i = 0; i < tabNames.length; i += 1) {
    var sheet = ss.getSheetByName(tabNames[i]);
    if (sheet) {
      try {
        console.log('기존 탭 삭제 중: ' + tabNames[i]);
        ss.deleteSheet(sheet);
      } catch (e) {
        console.log('삭제 건너뜀 (이미 없음): ' + tabNames[i] + ' — ' + e.message);
      }
    }
  }
  // 삭제를 확정한 후 insertSheet 호출 가능하도록 flush
  SpreadsheetApp.flush();
}

// ── 탭 생성 (GOOGLEFINANCE 없음) ────────────────────────

function createSettingsTab(ss) {
  var sheet = ss.insertSheet('⚙️ Settings');

  var headerRange = sheet.getRange(1, 1, 1, 2);
  headerRange.setValues([['Label', 'Value']]);
  headerRange.setFontWeight('bold');
  headerRange.setBackground('#DDE3EA');

  setSectionTitle(sheet, 'A2', '전략 상수');

  var constantRows = [
    ['SMA Window 1', 160],
    ['SMA Window 2', 165],
    ['SMA Window 3', 170],
    ['F1 Flip Window', 40],
    ['F1 Flip Threshold', 3],
    ['F1 Reduced Weight', 0.70],
    ['Emergency QQQ Threshold', -0.05],
    ['Emergency TQQQ Threshold', -0.15],
    ['OFF Residual', 0.10],
    ['Commission Rate', 0.001],
    ['Tax Rate', 0.22]
  ];
  sheet.getRange(2, 1, constantRows.length, 2).setValues(constantRows);

  setSectionTitle(sheet, 'A14', '포트폴리오 초기값');
  var portfolioRows = [
    ['TQQQ 수량', ''],
    ['TQQQ 평균 매수가', ''],
    ['SGOV 수량', ''],
    ['SGOV 평균 매수가', ''],
    ['현금 잔고 (KRW)', ''],
    ['', ''],
    ['실시간 데이터', '']
  ];
  sheet.getRange(15, 1, portfolioRows.length, 2).setValues(portfolioRows);

  var liveLabels = [
    ['QQQ 현재가', ''],
    ['TQQQ 현재가', ''],
    ['SGOV 현재가', ''],
    ['USD/KRW 환율', '']
  ];
  sheet.getRange(23, 1, liveLabels.length, 2).setValues(liveLabels);

  // 참고: GOOGLEFINANCE 수식은 단계 9 (_insertGF_Settings)에서 삽입

  sheet.getRange('A:A').setHorizontalAlignment('left');
  sheet.getRange('B:B').setHorizontalAlignment('right');
  sheet.getRange('B7:B10').setNumberFormat('0.00%');
  sheet.getRange('B11:B12').setNumberFormat('0.000');
  sheet.getRange('B16:B16').setNumberFormat('$#,##0.00');
  sheet.getRange('B18:B18').setNumberFormat('$#,##0.00');
  sheet.getRange('B19:B19').setNumberFormat('₩#,##0');
  sheet.getRange('B23:B25').setNumberFormat('$#,##0.00');
  sheet.getRange('B26:B26').setNumberFormat('₩#,##0.00');

  sheet.setColumnWidths(1, 1, 260);
  sheet.setColumnWidths(2, 1, 180);

  setNamedRangeSafe(ss, 'CFG_SMA_WIN1', sheet.getRange('B2'));
  setNamedRangeSafe(ss, 'CFG_SMA_WIN2', sheet.getRange('B3'));
  setNamedRangeSafe(ss, 'CFG_SMA_WIN3', sheet.getRange('B4'));
  setNamedRangeSafe(ss, 'CFG_F1_WINDOW', sheet.getRange('B5'));
  setNamedRangeSafe(ss, 'CFG_F1_THRESHOLD', sheet.getRange('B6'));
  setNamedRangeSafe(ss, 'CFG_F1_REDUCED', sheet.getRange('B7'));
  setNamedRangeSafe(ss, 'CFG_EMERGENCY_QQQ', sheet.getRange('B8'));
  setNamedRangeSafe(ss, 'CFG_EMERGENCY_TQQQ', sheet.getRange('B9'));
  setNamedRangeSafe(ss, 'CFG_OFF_RESIDUAL', sheet.getRange('B10'));
  setNamedRangeSafe(ss, 'CFG_COMMISSION', sheet.getRange('B11'));
  setNamedRangeSafe(ss, 'CFG_TAX', sheet.getRange('B12'));

  setNamedRangeSafe(ss, 'CFG_TQQQ_QTY', sheet.getRange('B15'));
  setNamedRangeSafe(ss, 'CFG_TQQQ_ENTRY', sheet.getRange('B16'));
  setNamedRangeSafe(ss, 'CFG_SGOV_QTY', sheet.getRange('B17'));
  setNamedRangeSafe(ss, 'CFG_SGOV_ENTRY', sheet.getRange('B18'));
  setNamedRangeSafe(ss, 'CFG_CASH_KRW', sheet.getRange('B19'));

  setNamedRangeSafe(ss, 'LIVE_QQQ', sheet.getRange('B23'));
  setNamedRangeSafe(ss, 'LIVE_TQQQ', sheet.getRange('B24'));
  setNamedRangeSafe(ss, 'LIVE_SGOV', sheet.getRange('B25'));
  setNamedRangeSafe(ss, 'LIVE_USDKRW', sheet.getRange('B26'));
}

function createPriceDataTab(ss) {
  var sheet = ss.insertSheet('📊 PriceData');
  // 참고: GOOGLEFINANCE 수식은 단계 10 (_insertGF_PriceData)에서 삽입
  sheet.hideSheet();
}

function createSignalTab(ss) {
  var sheet = ss.insertSheet('📈 Signal');
  var headers = [
    'Date',
    'QQQ Close',
    'SMA3',
    'SMA160',
    'SMA165',
    'SMA170',
    'Vote160',
    'Vote165',
    'Vote170',
    'Ensemble',
    'FlipCount',
    'State',
    'Target TQQQ%',
    'FlipCount Valid'
  ];
  setHeaderRow(sheet, headers);

  var row2Formulas = [[
    '=IFERROR(\'📊 PriceData\'!A2,"")',
    '=IFERROR(\'📊 PriceData\'!B2,"")',
    '=IFERROR(AVERAGE(OFFSET(B2,-2,0,3,1)),\"\")',
    '=IFERROR(AVERAGE(OFFSET(B2,-(CFG_SMA_WIN1-1),0,CFG_SMA_WIN1,1)),\"\")',
    '=IFERROR(AVERAGE(OFFSET(B2,-(CFG_SMA_WIN2-1),0,CFG_SMA_WIN2,1)),\"\")',
    '=IFERROR(AVERAGE(OFFSET(B2,-(CFG_SMA_WIN3-1),0,CFG_SMA_WIN3,1)),\"\")',
    '=IF(C2="","",IF(C2>D2,"PASS","FAIL"))',
    '=IF(C2="","",IF(C2>E2,"PASS","FAIL"))',
    '=IF(C2="","",IF(C2>F2,"PASS","FAIL"))',
    '=IF(G2="","",IF(COUNTIF(G2:I2,"PASS")>=2,"ON","OFF"))',
    '=IF(ROW()-1<CFG_F1_WINDOW,"",SUMPRODUCT(--(OFFSET(J2,-CFG_F1_WINDOW+1,0,CFG_F1_WINDOW-1,1)<>OFFSET(J2,-CFG_F1_WINDOW+2,0,CFG_F1_WINDOW-1,1))))',
    '=IF(J2="","",IF(\'🚨 Emergency\'!I2="🔴 ACTIVE","EMERGENCY",IF(J2="OFF","OFF10",IF(AND(J2="ON",K2>=CFG_F1_THRESHOLD),"ON-CHOPPY","ON"))))',
    '=IFS(L2="ON",1,L2="ON-CHOPPY",CFG_F1_REDUCED,L2="OFF10",CFG_OFF_RESIDUAL,L2="EMERGENCY",CFG_OFF_RESIDUAL,TRUE,"")',
    '=IF(ROW()-1<CFG_F1_WINDOW,CFG_F1_WINDOW-(ROW()-1)&" days left","VALID")'
  ]];
  var row2Range = sheet.getRange(2, 1, 1, headers.length);
  row2Range.setFormulas(row2Formulas);
  row2Range.copyTo(sheet.getRange(3, 1, 298, headers.length));

  sheet.getRange('B2:F300').setNumberFormat('$#,##0.00');
  sheet.getRange('M2:M300').setNumberFormat('0.00%');

  var voteRange = sheet.getRange('G2:I300');
  var stateRange = sheet.getRange('L2:L300');
  var flipRange = sheet.getRange('K2:K300');

  var rules = sheet.getConditionalFormatRules();
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('PASS')
      .setBackground('#00CC00')
      .setRanges([voteRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('FAIL')
      .setBackground('#FF0000')
      .setRanges([voteRange])
      .build()
  );

  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('ON')
      .setBackground('#00AA00')
      .setRanges([stateRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('ON-CHOPPY')
      .setBackground('#FFAA00')
      .setRanges([stateRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('OFF10')
      .setBackground('#CC0000')
      .setFontColor('#FFFFFF')
      .setRanges([stateRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('EMERGENCY')
      .setBackground('#9900CC')
      .setFontColor('#FFFFFF')
      .setRanges([stateRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenFormulaSatisfied('=AND($K2<>"",$K2>=3)')
      .setBackground('#FFAA00')
      .setRanges([flipRange])
      .build()
  );
  sheet.setConditionalFormatRules(rules);
}

function createEmergencyTab(ss) {
  var sheet = ss.insertSheet('🚨 Emergency');
  var headers = [
    'Date',
    'QQQ Close',
    'QQQ 일간수익률',
    '폭락 트리거',
    'TQQQ 현재가',
    'TQQQ 매수가',
    'TQQQ 하락률',
    '스탑 트리거',
    '비상 상태',
    '쿨다운'
  ];
  setHeaderRow(sheet, headers);

  var row2Formulas = [[
    '=IFERROR(\'📈 Signal\'!A2,"")',
    '=IFERROR(\'📈 Signal\'!B2,"")',
    '=IF(OR(B2="",B1=""),"",(B2-B1)/B1)',
    '=IF(C2="","",IF(C2<=CFG_EMERGENCY_QQQ,"🚨 TRIGGER","✅ SAFE"))',
    '=LIVE_TQQQ',
    '=CFG_TQQQ_ENTRY',
    '=IF(OR(E2="",F2="",F2=0),"",(E2-F2)/F2)',
    '=IF(G2="","",IF(G2<=CFG_EMERGENCY_TQQQ,"🚨 TRIGGER","✅ SAFE"))',
    '=IF(OR(D2="🚨 TRIGGER",H2="🚨 TRIGGER"),"🔴 ACTIVE","🟢 NONE")',
    '=IF(ROW()<=2,"CLEAR",IF(I1="🔴 ACTIVE","⏳ COOLDOWN","CLEAR"))'
  ]];
  var row2Range = sheet.getRange(2, 1, 1, headers.length);
  row2Range.setFormulas(row2Formulas);
  row2Range.copyTo(sheet.getRange(3, 1, 298, headers.length));

  sheet.getRange('C2:C300').setNumberFormat('0.00%');
  sheet.getRange('G2:G300').setNumberFormat('0.00%');
  sheet.getRange('B2:B300').setNumberFormat('$#,##0.00');
  sheet.getRange('E2:F300').setNumberFormat('$#,##0.00');

  var rules = sheet.getConditionalFormatRules();
  var crashRange = sheet.getRange('D2:D300');
  var stopRange = sheet.getRange('H2:H300');
  var emergencyRange = sheet.getRange('I2:I300');

  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('🚨 TRIGGER')
      .setBackground('#CC0000')
      .setFontColor('#FFFFFF')
      .setRanges([crashRange, stopRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('🔴 ACTIVE')
      .setBackground('#9900CC')
      .setFontColor('#FFFFFF')
      .setRanges([emergencyRange])
      .build()
  );
  sheet.setConditionalFormatRules(rules);
}

function createTradeLogTab(ss) {
  var sheet = ss.insertSheet('📝 TradeLog');
  var headers = [
    '날짜',
    '종목',
    '매매구분',
    '수량',
    '단가(USD)',
    '금액(USD)',
    'USD/KRW',
    '금액(KRW)',
    '수수료',
    '신호 상태',
    '메모'
  ];
  setHeaderRow(sheet, headers);

  var row2Formulas = [[
    '=IF(D2="","",D2*E2)',
    '=LIVE_USDKRW',
    '=IF(F2="","",F2*G2)',
    '=IF(F2="","",F2*CFG_COMMISSION)'
  ]];
  var row2Range = sheet.getRange(2, 6, 1, 4);
  row2Range.setFormulas(row2Formulas);
  row2Range.copyTo(sheet.getRange(3, 6, 198, 4));

  var tickerValidation = SpreadsheetApp.newDataValidation()
    .requireValueInList(['TQQQ', 'SGOV'], true)
    .setAllowInvalid(false)
    .build();
  sheet.getRange('B2:B200').setDataValidation(tickerValidation);

  var actionValidation = SpreadsheetApp.newDataValidation()
    .requireValueInList(['BUY', 'SELL', 'HOLD'], true)
    .setAllowInvalid(false)
    .build();
  sheet.getRange('C2:C200').setDataValidation(actionValidation);

  var sharesValidation = SpreadsheetApp.newDataValidation()
    .requireNumberGreaterThan(0)
    .setAllowInvalid(false)
    .build();
  sheet.getRange('D2:D200').setDataValidation(sharesValidation);

  var priceValidation = SpreadsheetApp.newDataValidation()
    .requireNumberGreaterThan(0)
    .setAllowInvalid(false)
    .build();
  sheet.getRange('E2:E200').setDataValidation(priceValidation);

  sheet.getRange('E2:G200').setNumberFormat('$#,##0.00');
  sheet.getRange('H2:H200').setNumberFormat('₩#,##0');
  sheet.getRange('I2:I200').setNumberFormat('$#,##0.00');
}

function createPortfolioTab(ss) {
  var sheet = ss.insertSheet('💼 Portfolio');
  var headers = [
    '종목',
    '수량',
    '평균매수가(USD)',
    '현재가(USD)',
    '평가금액(USD)',
    '평가금액(KRW)',
    '비중%',
    '목표비중%',
    '편차%',
    '미실현손익(USD)',
    '미실현손익(KRW)',
    '일간손익(USD)',
    '추천 거래'
  ];
  setHeaderRow(sheet, headers);

  var labels = [
    ['TQQQ'],
    ['SGOV'],
    ['CASH'],
    ['합계']
  ];
  sheet.getRange(2, 1, labels.length, 1).setValues(labels);
  sheet.getRange('A2:A5').setFontWeight('bold');

  var formulas = [
    [
      '=CFG_TQQQ_QTY',
      '=CFG_TQQQ_ENTRY',
      '=LIVE_TQQQ',
      '=IF(OR(B2="",D2=""),"",B2*D2)',
      '=IF(E2="","",E2*LIVE_USDKRW)',
      '=IF($E$5=0,"",E2/$E$5)',
      '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$M$2:$M$300<>""),\'📈 Signal\'!$M$2:$M$300),"")',
      '=IF(OR(G2="",H2=""),"",G2-H2)',
      '=IF(OR(B2="",C2="",D2=""),"",(D2-C2)*B2)',
      '=IF(J2="","",J2*LIVE_USDKRW)',
      '=""',
      '=IF(H2="","",IF(ABS(I2)<0.01,"HOLD",IF(I2>0,"SELL "&MAX(0,B2-CEILING(B2*H2,1))&" TQQQ","BUY "&MAX(0,CEILING(($E$5*H2-E2)/D2,1))&" TQQQ")))'
    ],
    [
      '=CFG_SGOV_QTY',
      '=CFG_SGOV_ENTRY',
      '=LIVE_SGOV',
      '=IF(OR(B3="",D3=""),"",B3*D3)',
      '=IF(E3="","",E3*LIVE_USDKRW)',
      '=IF($E$5=0,"",E3/$E$5)',
      '=IF(H2="","",1-H2)',
      '=IF(OR(G3="",H3=""),"",G3-H3)',
      '=IF(OR(B3="",C3="",D3=""),"",(D3-C3)*B3)',
      '=IF(J3="","",J3*LIVE_USDKRW)',
      '=""',
      '=IF(H3="","",IF(ABS(I3)<0.01,"HOLD",IF(I3>0,"SELL "&MAX(0,B3-CEILING(B3*H3,1))&" SGOV","BUY "&MAX(0,CEILING(($E$5*H3-E3)/D3,1))&" SGOV")))'
    ],
    [
      '=CFG_CASH_KRW',
      '',
      '',
      '=IF(B4="","",B4/LIVE_USDKRW)',
      '=B4',
      '=IF($E$5=0,"",E4/$E$5)',
      '=0',
      '=IF(OR(G4="",H4=""),"",G4-H4)',
      '=0',
      '=0',
      '=0',
      '=IF(B4="","","HOLD CASH")'
    ],
    [
      '',
      '',
      '',
      '=SUM(E2:E4)',
      '=SUM(F2:F4)',
      '=IF(E5=0,"",1)',
      '=IF(H2="","",H2+H3)',
      '=IF(OR(G5="",H5=""),"",G5-H5)',
      '=SUM(J2:J4)',
      '=SUM(K2:K4)',
      '=SUM(L2:L4)',
      '=TEXTJOIN(" | ",TRUE,M2,M3,M4)'
    ]
  ];

  sheet.getRange(2, 2, formulas.length, 12).setFormulas(formulas);

  // 참고: L2, L3은 플레이스홀더(""). 실제 GOOGLEFINANCE 일간손익
  // 수식은 단계 11 (_insertGF_Portfolio)에서 삽입됩니다.

  sheet.getRange('C2:E5').setNumberFormat('$#,##0.00');
  sheet.getRange('F2:F5').setNumberFormat('₩#,##0');
  sheet.getRange('G2:I5').setNumberFormat('0.00%');
  sheet.getRange('J2:J5').setNumberFormat('$#,##0.00');
  sheet.getRange('K2:K5').setNumberFormat('₩#,##0');
  sheet.getRange('L2:L5').setNumberFormat('$#,##0.00');
}

function createDashboardTab(ss) {
  var sheet = ss.insertSheet('📊 Dashboard');

  sheet.setColumnWidths(1, 1, 170);
  sheet.setColumnWidths(2, 1, 170);
  sheet.setColumnWidths(3, 1, 170);
  sheet.setColumnWidths(4, 1, 180);
  sheet.setColumnWidths(5, 1, 180);
  sheet.setColumnWidths(6, 1, 180);
  sheet.setColumnWidths(7, 1, 180);
  sheet.setColumnWidths(8, 1, 180);

  sheet.getRange('A1:H1').merge();
  sheet.getRange('A1').setValue('E03 v2026.3 트레이딩 전략 대시보드');
  sheet.getRange('A1').setFontSize(18).setFontWeight('bold').setHorizontalAlignment('center');
  sheet.getRange('A1').setBackground('#102A43').setFontColor('#FFFFFF');

  var headerRows = [
    ['오늘', '=TODAY()', '데이터 상태', '=IF(LIVE_QQQ="","⚠️ 오래됨","✅ 최신")', '마지막 업데이트', '=NOW()', '', ''],
    ['', '', '', '', '', '', '', '']
  ];
  sheet.getRange(2, 1, headerRows.length, 8).setValues(headerRows);
  sheet.getRange('B2:B2').setNumberFormat('yyyy-mm-dd');
  sheet.getRange('F2:F2').setNumberFormat('yyyy-mm-dd hh:mm:ss');

  sheet.getRange('A4:C5').merge();
  sheet.getRange('A4').setValue('판정');
  sheet.getRange('A4').setFontWeight('bold').setFontSize(16).setHorizontalAlignment('center');
  sheet.getRange('A4').setBackground('#D9E2EC');

  sheet.getRange('D4:H5').merge();
  sheet.getRange('D4').setFormula('=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$L$2:$L$300<>""),\'📈 Signal\'!$L$2:$L$300),"")');
  sheet.getRange('D4').setFontSize(24).setFontWeight('bold').setHorizontalAlignment('center').setVerticalAlignment('middle');

  var evidenceRows = [
    ['근거: Vote160', '', '', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$G$2:$G$300<>""),\'📈 Signal\'!$G$2:$G$300),"")', '근거: Vote165', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$H$2:$H$300<>""),\'📈 Signal\'!$H$2:$H$300),"")', '근거: Vote170', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$I$2:$I$300<>""),\'📈 Signal\'!$I$2:$I$300),"")'],
    ['SMA3', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$C$2:$C$300<>""),\'📈 Signal\'!$C$2:$C$300),"")', 'SMA160', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$D$2:$D$300<>""),\'📈 Signal\'!$D$2:$D$300),"")', 'SMA165', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$E$2:$E$300<>""),\'📈 Signal\'!$E$2:$E$300),"")', 'SMA170', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$F$2:$F$300<>""),\'📈 Signal\'!$F$2:$F$300),"")'],
    ['', '', '', '', '', '', '', '']
  ];
  sheet.getRange(7, 1, evidenceRows.length, 8).setValues(evidenceRows);
  sheet.getRange('B8:H8').setNumberFormat('$#,##0.00');

  var f1Rows = [
    ['F1 플립 횟수', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$K$2:$K$300<>""),\'📈 Signal\'!$K$2:$K$300),"")', 'F1 유효성', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$N$2:$N$300<>""),\'📈 Signal\'!$N$2:$N$300),"")', '목표 TQQQ%', '=IFERROR(LOOKUP(2,1/(\'📈 Signal\'!$M$2:$M$300<>""),\'📈 Signal\'!$M$2:$M$300),"")', '', ''],
    ['횡보 상태', '=IF(D4="ON-CHOPPY","횡보중","정상")', '', '', '', '', '', '']
  ];
  sheet.getRange(11, 1, f1Rows.length, 8).setValues(f1Rows);
  sheet.getRange('F11:F11').setNumberFormat('0.00%');

  var emergencyRows = [
    ['비상 QQQ 수익률', '=IFERROR(LOOKUP(2,1/(\'🚨 Emergency\'!$C$2:$C$300<>""),\'🚨 Emergency\'!$C$2:$C$300),"")', '비상 하락률', '=IFERROR(LOOKUP(2,1/(\'🚨 Emergency\'!$G$2:$G$300<>""),\'🚨 Emergency\'!$G$2:$G$300),"")', '비상 상태', '=IFERROR(LOOKUP(2,1/(\'🚨 Emergency\'!$I$2:$I$300<>""),\'🚨 Emergency\'!$I$2:$I$300),"")', '', ''],
    ['쿨다운', '=IFERROR(LOOKUP(2,1/(\'🚨 Emergency\'!$J$2:$J$300<>""),\'🚨 Emergency\'!$J$2:$J$300),"")', '', '', '', '', '', '']
  ];
  sheet.getRange(14, 1, emergencyRows.length, 8).setValues(emergencyRows);
  sheet.getRange('B14:B14').setNumberFormat('0.00%');
  sheet.getRange('D14:D14').setNumberFormat('0.00%');

  sheet.getRange('A17:C18').merge();
  sheet.getRange('A17').setValue('실행 액션');
  sheet.getRange('A17').setFontWeight('bold').setFontSize(16).setHorizontalAlignment('center').setVerticalAlignment('middle');
  sheet.getRange('A17').setBackground('#D9E2EC');
  sheet.getRange('D17:H18').merge();
  sheet.getRange('D17').setFormula('=IFERROR(TEXTJOIN(" | ",TRUE,\'💼 Portfolio\'!M2:M4),"")');
  sheet.getRange('D17').setFontWeight('bold').setWrap(true);

  var summaryRows = [
    ['포트폴리오 가치 USD', '=IFERROR(\'💼 Portfolio\'!E5,"")', '포트폴리오 가치 KRW', '=IFERROR(\'💼 Portfolio\'!F5,"")', '', '', '', ''],
    ['TQQQ 비중', '=IFERROR(\'💼 Portfolio\'!G2,"")', 'SGOV 비중', '=IFERROR(\'💼 Portfolio\'!G3,"")', '', '', '', ''],
    ['일간 손익 USD', '=IFERROR(\'💼 Portfolio\'!L5,"")', '목표 상태', '=IFERROR(D4,"")', '', '', '', ''],
    ['', '', '', '', '', '', '', '']
  ];
  sheet.getRange(20, 1, summaryRows.length, 8).setValues(summaryRows);

  sheet.getRange('B20:B20').setNumberFormat('$#,##0.00');
  sheet.getRange('D20:D20').setNumberFormat('₩#,##0');
  sheet.getRange('B21:D21').setNumberFormat('0.00%');
  sheet.getRange('B22:B22').setNumberFormat('$#,##0.00');

  sheet.getRange('A2:A23').setFontWeight('bold');
  sheet.getRange('A7:A22').setBackground('#F0F4F8');

  var rules = sheet.getConditionalFormatRules();
  var verdictRange = sheet.getRange('D4:H5');
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('ON')
      .setBackground('#00AA00')
      .setFontColor('#FFFFFF')
      .setRanges([verdictRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('OFF10')
      .setBackground('#CC0000')
      .setFontColor('#FFFFFF')
      .setRanges([verdictRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('ON-CHOPPY')
      .setBackground('#FFAA00')
      .setFontColor('#000000')
      .setRanges([verdictRange])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenTextEqualTo('EMERGENCY')
      .setBackground('#9900CC')
      .setFontColor('#FFFFFF')
      .setRanges([verdictRange])
      .build()
  );
  sheet.setConditionalFormatRules(rules);

  try {
    sheet.activate();
    ss.moveActiveSheet(1);
  } catch (e) {
    console.log('Dashboard를 첫 번째 위치로 이동 실패: ' + e.message);
  }
}

// ── GOOGLEFINANCE 수식 지연 삽입 ────────────────────────

function _insertGF_Settings(ss) {
  var sheet = getSheetOrThrow(ss, '⚙️ Settings');
  var liveFormulas = [
    [safeGF('QQQ', 'price')],
    [safeGF('TQQQ', 'price')],
    [safeGF('SGOV', 'price')],
    ['=IFERROR(GOOGLEFINANCE("CURRENCY:USDKRW"),"")']
  ];
  sheet.getRange(23, 2, liveFormulas.length, 1).setFormulas(liveFormulas);
}

function _insertGF_PriceData(ss) {
  var sheet = getSheetOrThrow(ss, '📊 PriceData');
  var formula = [[safeGFHistory('QQQ', 'close', 'DATE(2025,1,1)')]];
  sheet.getRange(1, 1, 1, 1).setFormulas(formula);
}

function _insertGF_Portfolio(ss) {
  var sheet = getSheetOrThrow(ss, '💼 Portfolio');
  sheet.getRange('L2').setFormula(
    '=IF(B2="","",B2*(LIVE_TQQQ-IFERROR(INDEX(GOOGLEFINANCE("TQQQ","close",TODAY()-7,TODAY(),"DAILY"),2,2),LIVE_TQQQ)))'
  );
  sheet.getRange('L3').setFormula(
    '=IF(B3="","",B3*(LIVE_SGOV-IFERROR(INDEX(GOOGLEFINANCE("SGOV","close",TODAY()-7,TODAY(),"DAILY"),2,2),LIVE_SGOV)))'
  );
}

// ── 글로벌 포맷팅 ──────────────────────────────────────

function applyGlobalFormatting(ss) {
  var tabNames = [
    '⚙️ Settings',
    '📊 PriceData',
    '📈 Signal',
    '🚨 Emergency',
    '📝 TradeLog',
    '💼 Portfolio',
    '📊 Dashboard'
  ];

  var i;
  for (i = 0; i < tabNames.length; i += 1) {
    var sheet = ss.getSheetByName(tabNames[i]);
    if (!sheet) {
      continue;
    }
    sheet.setFrozenRows(1);
  }

  // 전략 상수 보호 (경고만 표시)
  var settings = ss.getSheetByName('⚙️ Settings');
  if (settings) {
    var protectRange = settings.getRange('A2:B12');
    var protection = protectRange.protect();
    protection.setDescription('E03 전략 상수 (보호됨)');
    protection.setWarningOnly(true);
  }
}

// ── 유틸리티 함수 ──────────────────────────────────────

function clearExistingDailyTrigger() {
  var triggers = ScriptApp.getProjectTriggers();
  var i;
  for (i = 0; i < triggers.length; i += 1) {
    if (triggers[i].getHandlerFunction() === 'onDailyUpdate') {
      ScriptApp.deleteTrigger(triggers[i]);
    }
  }
}

function setupDailyTrigger() {
  clearExistingDailyTrigger();
  ScriptApp.newTrigger('onDailyUpdate')
    .timeBased()
    .everyDays(1)
    .atHour(7)
    .create();
}

function onDailyUpdate() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('📊 Dashboard');
  if (!dashboard) {
    return;
  }

  dashboard.getRange('F2').setFormula('=NOW()');
  SpreadsheetApp.flush();
}

function refreshNamedRanges() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var settings = getSheetOrThrow(ss, '⚙️ Settings');

  setNamedRangeSafe(ss, 'CFG_SMA_WIN1', settings.getRange('B2'));
  setNamedRangeSafe(ss, 'CFG_SMA_WIN2', settings.getRange('B3'));
  setNamedRangeSafe(ss, 'CFG_SMA_WIN3', settings.getRange('B4'));
  setNamedRangeSafe(ss, 'CFG_F1_WINDOW', settings.getRange('B5'));
  setNamedRangeSafe(ss, 'CFG_F1_THRESHOLD', settings.getRange('B6'));
  setNamedRangeSafe(ss, 'CFG_F1_REDUCED', settings.getRange('B7'));
  setNamedRangeSafe(ss, 'CFG_EMERGENCY_QQQ', settings.getRange('B8'));
  setNamedRangeSafe(ss, 'CFG_EMERGENCY_TQQQ', settings.getRange('B9'));
  setNamedRangeSafe(ss, 'CFG_OFF_RESIDUAL', settings.getRange('B10'));
  setNamedRangeSafe(ss, 'CFG_COMMISSION', settings.getRange('B11'));
  setNamedRangeSafe(ss, 'CFG_TAX', settings.getRange('B12'));
  setNamedRangeSafe(ss, 'CFG_TQQQ_QTY', settings.getRange('B15'));
  setNamedRangeSafe(ss, 'CFG_TQQQ_ENTRY', settings.getRange('B16'));
  setNamedRangeSafe(ss, 'CFG_SGOV_QTY', settings.getRange('B17'));
  setNamedRangeSafe(ss, 'CFG_SGOV_ENTRY', settings.getRange('B18'));
  setNamedRangeSafe(ss, 'CFG_CASH_KRW', settings.getRange('B19'));
  setNamedRangeSafe(ss, 'LIVE_QQQ', settings.getRange('B23'));
  setNamedRangeSafe(ss, 'LIVE_TQQQ', settings.getRange('B24'));
  setNamedRangeSafe(ss, 'LIVE_SGOV', settings.getRange('B25'));
  setNamedRangeSafe(ss, 'LIVE_USDKRW', settings.getRange('B26'));
}

function resetOnlySignals() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var signal = ss.getSheetByName('📈 Signal');
  var emergency = ss.getSheetByName('🚨 Emergency');

  if (signal) {
    signal.clear();
    ss.deleteSheet(signal);
  }
  if (emergency) {
    emergency.clear();
    ss.deleteSheet(emergency);
  }

  createSignalTab(ss);
  createEmergencyTab(ss);
  applyGlobalFormatting(ss);
}

function applyThemeToAllTabs() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var tabNames = [
    '⚙️ Settings',
    '📊 PriceData',
    '📈 Signal',
    '🚨 Emergency',
    '📝 TradeLog',
    '💼 Portfolio',
    '📊 Dashboard'
  ];
  var i;
  for (i = 0; i < tabNames.length; i += 1) {
    var sheet = ss.getSheetByName(tabNames[i]);
    if (!sheet) {
      continue;
    }
    var lastCol = Math.max(sheet.getLastColumn(), 2);
    var header = sheet.getRange(1, 1, 1, lastCol);
    header.setBackground('#E8EAED');
    header.setFontWeight('bold');
    header.setFontColor('#111111');
  }
}

function compactColumnsForMobile() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('📊 Dashboard');
  if (!dashboard) {
    return;
  }
  dashboard.setColumnWidths(1, 8, 150);
}

function expandColumnsForDesktop() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('📊 Dashboard');
  if (!dashboard) {
    return;
  }
  dashboard.setColumnWidths(1, 8, 200);
}

function rebuildDashboardOnly() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('📊 Dashboard');
  if (dashboard) {
    ss.deleteSheet(dashboard);
  }
  createDashboardTab(ss);
  applyGlobalFormatting(ss);
}

function hidePriceDataTab() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var priceData = ss.getSheetByName('📊 PriceData');
  if (priceData) {
    priceData.hideSheet();
  }
}

function unhidePriceDataTab() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var priceData = ss.getSheetByName('📊 PriceData');
  if (priceData) {
    priceData.showSheet();
  }
}

function validateSetupQuick() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var required = [
    '⚙️ Settings',
    '📊 PriceData',
    '📈 Signal',
    '🚨 Emergency',
    '📝 TradeLog',
    '💼 Portfolio',
    '📊 Dashboard'
  ];
  var missing = [];
  var i;
  for (i = 0; i < required.length; i += 1) {
    if (!ss.getSheetByName(required[i])) {
      missing.push(required[i]);
    }
  }

  if (missing.length > 0) {
    console.log('누락된 탭: ' + missing.join(', '));
    return;
  }
  console.log('E03 탭 구조가 완전합니다.');
}

function forceRecalcNow() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var settings = ss.getSheetByName('⚙️ Settings');
  if (!settings) {
    return;
  }
  settings.getRange('B27').setFormula('=NOW()');
  SpreadsheetApp.flush();
}

function freezeAllHeaders() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheets = ss.getSheets();
  var i;
  for (i = 0; i < sheets.length; i += 1) {
    sheets[i].setFrozenRows(1);
  }
}

function autoResizeAllColumns() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheets = ss.getSheets();
  var i;
  for (i = 0; i < sheets.length; i += 1) {
    var colCount = Math.max(sheets[i].getLastColumn(), 2);
    sheets[i].autoResizeColumns(1, colCount);
  }
}

function applyNumberFormatsAgain() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  applyGlobalFormatting(ss);
}

function moveDashboardFirst() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('📊 Dashboard');
  if (dashboard) {
    dashboard.activate();
    ss.moveActiveSheet(1);
  }
}
