"use client";

import { useMemo } from "react";
import { Calendar, TrendingUp, TrendingDown, Loader2 } from "lucide-react";
import {
  EquityPoint,
  MonthlyReturnsData,
  calculateMonthlyReturns,
  generateMockMonthlyReturns,
  getReturnColor,
  formatReturn,
  formatMonthRef,
  getMonthName,
  getYears,
  getReturnForMonth,
  getYearlyTotal,
} from "./utils/monthly-returns";

interface ReturnsHeatmapProps {
  equity: EquityPoint[] | null;
  isMock?: boolean;
  isLoading?: boolean;
}

function StatCard({ 
  label, 
  value, 
  icon: Icon, 
  valueColor 
}: { 
  label: string; 
  value: string; 
  icon: React.ElementType;
  valueColor?: string;
}) {
  return (
    <div className="bg-neutral-900/50 border border-neutral-800 rounded-lg p-4">
      <div className="flex items-center gap-2 text-xs text-muted mb-2">
        <Icon size={12} />
        {label}
      </div>
      <div className={`font-mono font-bold text-sm ${valueColor || "text-fg"}`}>
        {value}
      </div>
    </div>
  );
}

export function ReturnsHeatmap({ equity, isMock, isLoading }: ReturnsHeatmapProps) {
  const data: MonthlyReturnsData | null = useMemo(() => {
    if (equity && equity.length > 0) {
      return calculateMonthlyReturns(equity);
    }
    if (isMock) {
      return generateMockMonthlyReturns(2020, new Date().getFullYear());
    }
    return null;
  }, [equity, isMock]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12 text-muted">
        <Loader2 className="animate-spin mr-2" size={20} />
        <span className="text-sm">{isMock ? "히트맵 생성 중..." : "성과 데이터 로딩 중..."}</span>
      </div>
    );
  }

  if (!data || data.returns.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted gap-2">
        <Calendar size={24} className="opacity-50" />
        {isMock ? (
          <>
            <span className="text-sm">백테스트를 실행하면 히트맵이 표시됩니다</span>
            <span className="text-xs text-neutral-600">Generate 버튼을 클릭하세요</span>
          </>
        ) : (
          <>
            <span className="text-sm">아직 포트폴리오 데이터가 없습니다</span>
            <span className="text-xs text-neutral-600">거래 기록이 쌓이면 성과가 표시됩니다</span>
          </>
        )}
      </div>
    );
  }

  const years = getYears(data);
  const months = Array.from({ length: 12 }, (_, i) => i + 1);

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left py-2 px-2 font-medium text-muted w-16">Year</th>
              {months.map(m => (
                <th key={m} className="text-center py-2 px-1 font-medium text-muted w-12">
                  {getMonthName(m)}
                </th>
              ))}
              <th className="text-right py-2 px-2 font-medium text-muted w-16">Total</th>
            </tr>
          </thead>
          <tbody>
            {years.map(year => {
              const yearTotal = getYearlyTotal(data, year);
              return (
                <tr key={year} className="border-t border-neutral-800/50">
                  <td className="py-1.5 px-2 font-mono font-bold text-fg">{year}</td>
                  {months.map(month => {
                    const ret = getReturnForMonth(data, year, month);
                    if (ret === null) {
                      return (
                        <td key={month} className="py-1.5 px-1">
                          <div className="w-full h-8 rounded bg-neutral-900/50" />
                        </td>
                      );
                    }
                    return (
                      <td key={month} className="py-1.5 px-1">
                        <div
                          className="w-full h-8 rounded flex items-center justify-center font-mono text-[10px] font-bold text-white transition-transform hover:scale-105 cursor-default"
                          style={{ backgroundColor: getReturnColor(ret) }}
                          title={`${getMonthName(month)} ${year}: ${formatReturn(ret)}`}
                        >
                          {ret >= 0 ? "+" : ""}{ret.toFixed(0)}
                        </div>
                      </td>
                    );
                  })}
                  <td className="py-1.5 px-2 text-right">
                    <span 
                      className={`font-mono font-bold ${
                        yearTotal !== null && yearTotal >= 0 ? "text-positive" : "text-negative"
                      }`}
                    >
                      {yearTotal !== null ? formatReturn(yearTotal) : "-"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <StatCard
          icon={TrendingUp}
          label="최고 월"
          value={formatMonthRef(data.bestMonth)}
          valueColor="text-positive"
        />
        <StatCard
          icon={TrendingDown}
          label="최저 월"
          value={formatMonthRef(data.worstMonth)}
          valueColor="text-negative"
        />
        <StatCard
          icon={Calendar}
          label="상승 월"
          value={`${data.positiveMonths}/${data.totalMonths} (${Math.round(data.positiveMonths / data.totalMonths * 100)}%)`}
          valueColor="text-fg"
        />
      </div>
    </div>
  );
}
