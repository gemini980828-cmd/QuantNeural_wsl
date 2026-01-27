export interface EquityPoint {
  date: string;
  value: number;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

export interface DrawdownPeriod {
  startDate: string;
  troughDate: string;
  endDate: string | null;
  peakValue: number;
  troughValue: number;
  drawdown: number;
  duration: number;
  recovery: number | null;
}

export interface RollingDataPoint {
  date: string;
  value: number;
}

export interface RiskMetrics {
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  ulcerIndex: number;
  maxDrawdown: number;
  avgRecoveryDays: number | null;
  volatility: number;
  downsideDeviation: number;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

const toSortedEquity = (equity: EquityPoint[]): EquityPoint[] =>
  [...equity].sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
  );

const diffDays = (start: Date, end: Date): number =>
  Math.round((end.getTime() - start.getTime()) / MS_PER_DAY);

const calculateDailyReturns = (equity: EquityPoint[]): number[] => {
  if (equity.length < 2) return [];
  const returns: number[] = [];
  for (let i = 1; i < equity.length; i += 1) {
    const prev = equity[i - 1].value;
    const curr = equity[i].value;
    returns.push(prev !== 0 ? curr / prev - 1 : 0);
  }
  return returns;
};

const mean = (values: number[]): number =>
  values.length === 0
    ? 0
    : values.reduce((sum, v) => sum + v, 0) / values.length;

const standardDeviation = (values: number[]): number => {
  if (values.length === 0) return 0;
  const avg = mean(values);
  const variance =
    values.reduce((sum, v) => sum + (v - avg) ** 2, 0) / values.length;
  return Math.sqrt(variance);
};

/**
 * Calculate drawdown series from equity curve.
 */
export function calculateDrawdownSeries(
  equity: EquityPoint[]
): DrawdownPoint[] {
  if (!equity || equity.length === 0) return [];
  const sorted = toSortedEquity(equity);
  let peak = sorted[0].value;
  return sorted.map(point => {
    if (point.value > peak) peak = point.value;
    const raw = peak !== 0 ? point.value / peak - 1 : 0;
    const drawdown = Math.min(0, raw);
    return { date: point.date, drawdown };
  });
}

/**
 * Calculate maximum drawdown and peak/trough dates.
 */
export function calculateMaxDrawdown(equity: EquityPoint[]): {
  mdd: number;
  peakDate: string;
  troughDate: string;
} {
  if (!equity || equity.length === 0) {
    return { mdd: 0, peakDate: '', troughDate: '' };
  }

  const sorted = toSortedEquity(equity);
  let peakValue = sorted[0].value;
  let peakDate = sorted[0].date;
  let troughDate = sorted[0].date;
  let maxDrawdown = 0;

  sorted.forEach(point => {
    if (point.value > peakValue) {
      peakValue = point.value;
      peakDate = point.date;
    }
    const dd = peakValue !== 0 ? point.value / peakValue - 1 : 0;
    if (dd < maxDrawdown) {
      maxDrawdown = dd;
      troughDate = point.date;
    }
  });

  return { mdd: maxDrawdown, peakDate, troughDate };
}

/**
 * Calculate discrete drawdown periods with recovery info.
 */
export function calculateDrawdownPeriods(
  equity: EquityPoint[]
): DrawdownPeriod[] {
  if (!equity || equity.length === 0) return [];
  const sorted = toSortedEquity(equity);
  const periods: DrawdownPeriod[] = [];

  let peakValue = sorted[0].value;
  let peakDate = sorted[0].date;
  let troughValue = sorted[0].value;
  let troughDate = sorted[0].date;
  let maxDrawdown = 0;
  let inDrawdown = false;

  for (let i = 1; i < sorted.length; i += 1) {
    const point = sorted[i];

    if (point.value > peakValue) {
      if (inDrawdown && maxDrawdown < 0) {
        const start = new Date(peakDate);
        const trough = new Date(troughDate);
        const recoveryDate = new Date(point.date);
        periods.push({
          startDate: peakDate,
          troughDate,
          endDate: point.date,
          peakValue,
          troughValue,
          drawdown: maxDrawdown,
          duration: diffDays(start, trough),
          recovery: diffDays(trough, recoveryDate)
        });
      }

      peakValue = point.value;
      peakDate = point.date;
      troughValue = point.value;
      troughDate = point.date;
      maxDrawdown = 0;
      inDrawdown = false;
      continue;
    }

    const dd = peakValue !== 0 ? point.value / peakValue - 1 : 0;
    const drawdown = Math.min(0, dd);
    if (drawdown < maxDrawdown) {
      maxDrawdown = drawdown;
      troughValue = point.value;
      troughDate = point.date;
      inDrawdown = true;
    }
  }

  if (inDrawdown && maxDrawdown < 0) {
    const start = new Date(peakDate);
    const trough = new Date(troughDate);
    periods.push({
      startDate: peakDate,
      troughDate,
      endDate: null,
      peakValue,
      troughValue,
      drawdown: maxDrawdown,
      duration: diffDays(start, trough),
      recovery: null
    });
  }

  return periods;
}

/**
 * Calculate rolling total returns over a window of calendar days.
 */
export function calculateRollingReturns(
  equity: EquityPoint[],
  windowDays = 252
): RollingDataPoint[] {
  if (!equity || equity.length === 0) return [];
  const sorted = toSortedEquity(equity);
  const result: RollingDataPoint[] = [];
  let startIndex = 0;

  for (let i = 0; i < sorted.length; i += 1) {
    const currentDate = new Date(sorted[i].date).getTime();
    const windowStart = currentDate - windowDays * MS_PER_DAY;
    while (
      startIndex < i &&
      new Date(sorted[startIndex].date).getTime() < windowStart
    ) {
      startIndex += 1;
    }

    const base = sorted[startIndex].value;
    const value = base !== 0 ? sorted[i].value / base - 1 : 0;
    result.push({ date: sorted[i].date, value });
  }

  return result;
}

/**
 * Calculate rolling Sharpe ratio over a window of calendar days.
 */
export function calculateRollingSharpe(
  equity: EquityPoint[],
  windowDays = 252,
  riskFreeRate = 0.02
): RollingDataPoint[] {
  if (!equity || equity.length === 0) return [];
  const sorted = toSortedEquity(equity);
  const result: RollingDataPoint[] = [];
  let startIndex = 0;
  const periodsPerYear = 252;

  for (let i = 0; i < sorted.length; i += 1) {
    const currentDate = new Date(sorted[i].date).getTime();
    const windowStart = currentDate - windowDays * MS_PER_DAY;
    while (
      startIndex < i &&
      new Date(sorted[startIndex].date).getTime() < windowStart
    ) {
      startIndex += 1;
    }

    const windowEquity = sorted.slice(startIndex, i + 1);
    const returns = calculateDailyReturns(windowEquity);
    if (returns.length === 0) {
      result.push({ date: sorted[i].date, value: 0 });
      continue;
    }

    const avg = mean(returns);
    const std = standardDeviation(returns);
    const annualizedReturn = avg * periodsPerYear;
    const annualizedStd = std * Math.sqrt(periodsPerYear);
    const sharpe =
      annualizedStd !== 0
        ? (annualizedReturn - riskFreeRate) / annualizedStd
        : 0;

    result.push({ date: sorted[i].date, value: sharpe });
  }

  return result;
}

/**
 * Calculate full set of risk metrics for an equity curve.
 */
export function calculateRiskMetrics(
  equity: EquityPoint[],
  periodsPerYear = 252,
  riskFreeRate = 0.02
): RiskMetrics {
  if (!equity || equity.length === 0) {
    return {
      sharpeRatio: 0,
      sortinoRatio: 0,
      calmarRatio: 0,
      ulcerIndex: 0,
      maxDrawdown: 0,
      avgRecoveryDays: null,
      volatility: 0,
      downsideDeviation: 0
    };
  }

  const sorted = toSortedEquity(equity);
  const returns = calculateDailyReturns(sorted);
  const totalReturn =
    sorted[0].value !== 0
      ? sorted[sorted.length - 1].value / sorted[0].value - 1
      : 0;
  const annualizedReturn =
    returns.length > 0
      ? Math.pow(1 + totalReturn, periodsPerYear / returns.length) - 1
      : 0;

  const volatility = standardDeviation(returns) * Math.sqrt(periodsPerYear);
  const sharpeRatio =
    volatility !== 0 ? (annualizedReturn - riskFreeRate) / volatility : 0;

  const downsideReturns = returns.filter(r => r < 0);
  const downsideDeviation =
    downsideReturns.length > 0
      ? Math.sqrt(
          downsideReturns.reduce((sum, r) => sum + r ** 2, 0) /
            downsideReturns.length
        ) * Math.sqrt(periodsPerYear)
      : 0;
  const sortinoRatio =
    downsideDeviation !== 0
      ? (annualizedReturn - 0) / downsideDeviation
      : 0;

  const { mdd: maxDrawdown } = calculateMaxDrawdown(sorted);
  const calmarRatio =
    maxDrawdown !== 0 ? annualizedReturn / Math.abs(maxDrawdown) : 0;

  const drawdownSeries = calculateDrawdownSeries(sorted);
  const ulcerIndex =
    drawdownSeries.length > 0
      ? Math.sqrt(
          drawdownSeries.reduce(
            (sum, point) => sum + (point.drawdown * 100) ** 2,
            0
          ) / drawdownSeries.length
        )
      : 0;

  const periods = calculateDrawdownPeriods(sorted);
  const recoveryDays = periods
    .map(period => period.recovery)
    .filter((value): value is number => value !== null);
  const avgRecoveryDays =
    recoveryDays.length > 0
      ? recoveryDays.reduce((sum, value) => sum + value, 0) /
        recoveryDays.length
      : null;

  return {
    sharpeRatio,
    sortinoRatio,
    calmarRatio,
    ulcerIndex,
    maxDrawdown,
    avgRecoveryDays,
    volatility,
    downsideDeviation
  };
}
