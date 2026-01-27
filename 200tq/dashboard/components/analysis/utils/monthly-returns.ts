export interface EquityPoint {
  date: string;
  value: number;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return: number;
}

export interface YearlyTotal {
  year: number;
  total: number;
}

export interface MonthlyReturnsData {
  returns: MonthlyReturn[];
  yearlyTotals: YearlyTotal[];
  bestMonth: MonthlyReturn | null;
  worstMonth: MonthlyReturn | null;
  positiveMonths: number;
  totalMonths: number;
}

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export function getMonthName(month: number): string {
  return MONTH_NAMES[month - 1] || '';
}

export function calculateMonthlyReturns(equity: EquityPoint[]): MonthlyReturnsData {
  if (!equity || equity.length === 0) {
    return { 
      returns: [], 
      yearlyTotals: [], 
      bestMonth: null, 
      worstMonth: null, 
      positiveMonths: 0, 
      totalMonths: 0 
    };
  }

  const sorted = [...equity].sort((a, b) => 
    new Date(a.date).getTime() - new Date(b.date).getTime()
  );

  const monthlyGroups = new Map<string, { start: number; end: number }>();
  const yearlyGroups = new Map<number, { start: number; end: number }>();

  sorted.forEach(point => {
    const d = new Date(point.date);
    const year = d.getFullYear();
    const month = d.getMonth() + 1;
    const monthKey = `${year}-${month}`;

    const existingMonth = monthlyGroups.get(monthKey);
    if (!existingMonth) {
      monthlyGroups.set(monthKey, { start: point.value, end: point.value });
    } else {
      existingMonth.end = point.value;
    }

    const existingYear = yearlyGroups.get(year);
    if (!existingYear) {
      yearlyGroups.set(year, { start: point.value, end: point.value });
    } else {
      existingYear.end = point.value;
    }
  });

  const returns: MonthlyReturn[] = [];
  monthlyGroups.forEach((values, key) => {
    const [yearStr, monthStr] = key.split('-');
    const year = parseInt(yearStr, 10);
    const month = parseInt(monthStr, 10);
    const ret = values.start !== 0 
      ? ((values.end - values.start) / values.start) * 100 
      : 0;
    returns.push({ year, month, return: ret });
  });
  returns.sort((a, b) => a.year - b.year || a.month - b.month);

  const yearlyTotals: YearlyTotal[] = [];
  yearlyGroups.forEach((values, year) => {
    const total = values.start !== 0 
      ? ((values.end - values.start) / values.start) * 100 
      : 0;
    yearlyTotals.push({ year, total });
  });
  yearlyTotals.sort((a, b) => a.year - b.year);

  let bestMonth: MonthlyReturn | null = null;
  let worstMonth: MonthlyReturn | null = null;
  let positiveMonths = 0;

  returns.forEach(r => {
    if (r.return > 0) positiveMonths++;
    if (!bestMonth || r.return > bestMonth.return) bestMonth = r;
    if (!worstMonth || r.return < worstMonth.return) worstMonth = r;
  });

  return { 
    returns, 
    yearlyTotals, 
    bestMonth, 
    worstMonth, 
    positiveMonths, 
    totalMonths: returns.length 
  };
}

export function generateMockMonthlyReturns(
  startYear: number, 
  endYear: number
): MonthlyReturnsData {
  const returns: MonthlyReturn[] = [];
  const yearlyTotals: YearlyTotal[] = [];

  const MOCK_VALUES = [
    [12.3, -3.2, 8.7, 15.1, -7.8, 4.2, 9.5, -2.1, 11.3, 6.8, -4.5, 18.2],
    [5.6, 14.2, -8.9, 3.1, 7.4, -1.2, 22.1, 4.8, -6.3, 9.7, 2.1, 13.5],
    [-4.1, 8.9, 16.3, -2.7, 5.8, 11.2, -9.4, 7.6, 3.2, -5.8, 19.4, 6.1],
    [10.5, -6.2, 4.8, 12.7, -3.9, 8.1, 5.4, -1.8, 14.6, 2.3, 7.9, -4.2],
    [7.2, 3.5, -5.1, 9.8, 11.4, -2.6, 6.3, 15.7, -8.2, 4.1, 8.6, 1.9],
  ];

  for (let year = startYear; year <= endYear; year++) {
    let yearlyReturn = 0;
    const yearIndex = (year - startYear) % MOCK_VALUES.length;
    const mockRow = MOCK_VALUES[yearIndex];

    for (let month = 1; month <= 12; month++) {
      const ret = mockRow[month - 1];
      returns.push({ year, month, return: ret });
      
      yearlyReturn = (1 + yearlyReturn / 100) * (1 + ret / 100) - 1;
      yearlyReturn *= 100;
    }
    yearlyTotals.push({ year, total: Math.round(yearlyReturn * 10) / 10 });
  }

  let bestMonth = returns[0];
  let worstMonth = returns[0];
  let positiveMonths = 0;
  
  returns.forEach(r => {
    if (r.return > 0) positiveMonths++;
    if (r.return > bestMonth.return) bestMonth = r;
    if (r.return < worstMonth.return) worstMonth = r;
  });

  return { 
    returns, 
    yearlyTotals, 
    bestMonth, 
    worstMonth, 
    positiveMonths, 
    totalMonths: returns.length 
  };
}

export function getReturnColor(returnValue: number): string {
  if (returnValue < -20) return '#991b1b';
  if (returnValue < -10) return '#b91c1c';
  if (returnValue < -5) return '#dc2626';
  if (returnValue < 0) return 'rgba(239, 68, 68, 0.7)';
  if (returnValue < 5) return 'rgba(34, 197, 94, 0.7)';
  if (returnValue < 10) return '#16a34a';
  if (returnValue < 20) return '#15803d';
  return '#166534';
}

export function formatReturn(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
}

export function formatMonthRef(month: MonthlyReturn | null): string {
  if (!month) return '-';
  return `${getMonthName(month.month)} ${month.year} (${formatReturn(month.return)})`;
}

export function getYears(data: MonthlyReturnsData): number[] {
  const years = new Set<number>();
  data.returns.forEach(r => years.add(r.year));
  return Array.from(years).sort((a, b) => a - b);
}

export function getReturnForMonth(
  data: MonthlyReturnsData, 
  year: number, 
  month: number
): number | null {
  const found = data.returns.find(r => r.year === year && r.month === month);
  return found ? found.return : null;
}

export function getYearlyTotal(data: MonthlyReturnsData, year: number): number | null {
  const found = data.yearlyTotals.find(y => y.year === year);
  return found ? found.total : null;
}
