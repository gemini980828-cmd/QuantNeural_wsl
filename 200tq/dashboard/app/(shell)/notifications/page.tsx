"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { 
  Bell, RefreshCw, CheckCircle2, AlertTriangle, Info, 
  Database, Server, ClipboardList, ExternalLink, Clock, Check,
  ArrowRightLeft, Calendar, CalendarClock, HelpCircle, Zap
} from "lucide-react";

// Types
interface OpsNotification {
  id: string;
  dedupe_key: string;
  level: "info" | "action" | "emergency";
  event_type: string;
  title: string;
  body: string | null;
  resolved: boolean;
  resolved_at: string | null;
  created_at: string;
}

interface OpsJobRun {
  id: string;
  job_type: string;
  started_at: string;
  ended_at: string | null;
  status: "running" | "success" | "failed";
  max_price_date: string | null;
  rows_upserted: number;
  error: string | null;
}

type TabId = "all" | "action" | "emergency" | "resolved";

// Helper functions
function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return "방금 전";
  if (diffMins < 60) return `${diffMins}분 전`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}시간 전`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}일 전`;
}

function getLevelColor(level: string): string {
  switch (level) {
    case "emergency": return "bg-negative-tint text-negative border-negative/30";
    case "action": return "bg-choppy-tint text-choppy border-choppy/30";
    default: return "bg-info-tint text-info border-info/30";
  }
}

function getLevelIcon(level: string) {
  switch (level) {
    case "emergency": return <AlertTriangle size={14} />;
    case "action": return <Bell size={14} />;
    default: return <Info size={14} />;
  }
}

const EVENT_TYPE_META: Record<string, { label: string; icon: React.ReactNode; badgeClass: string }> = {
  VERDICT_CHANGE:   { label: "Verdict 변경",  icon: <ArrowRightLeft size={13} />, badgeClass: "bg-info-tint text-info border-info/30" },
  EXEC_SCHEDULED:   { label: "스케줄 생성",   icon: <CalendarClock size={13} />,  badgeClass: "bg-choppy-tint text-choppy border-choppy/30" },
  DUE_TODAY:        { label: "오늘 실행일",    icon: <Calendar size={13} />,       badgeClass: "bg-positive-tint text-positive border-positive/30" },
  UNKNOWN_PERSIST:  { label: "기록 누락",      icon: <HelpCircle size={13} />,     badgeClass: "bg-choppy-tint text-choppy border-choppy/30" },
  SOFT_ALERT:       { label: "장중 경보",      icon: <Zap size={13} />,            badgeClass: "bg-negative-tint text-negative border-negative/30" },
  HARD_CONFIRMED:   { label: "비상 확정",      icon: <AlertTriangle size={13} />,  badgeClass: "bg-negative-tint text-negative border-negative/30" },
};

function getEventTypeMeta(eventType: string) {
  return EVENT_TYPE_META[eventType] || { label: eventType, icon: <Info size={13} />, badgeClass: "bg-inset text-muted border-border" };
}

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<OpsNotification[]>([]);
  const [jobRuns, setJobRuns] = useState<OpsJobRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<TabId>("all");
  const [unresolvedCounts, setUnresolvedCounts] = useState({ info: 0, action: 0, emergency: 0, total: 0 });
  
  // Meta from /api/ops/today
  const [meta, setMeta] = useState<{
    lastIngestSuccess: string | null;
    lastIngestFail: string | null;
    unresolvedAlerts: number;
    health: string;
    staleReason: string | null;
    verdictDateKst: string;
    executionDateKst: string;
  } | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch notifications
      const notifRes = await fetch("/api/notifications/list?limit=50");
      const notifData = await notifRes.json();
      if (notifData.success) {
        setNotifications(notifData.notifications);
        setUnresolvedCounts(notifData.unresolvedCounts);
      }
      
      // Fetch ops meta
      const opsRes = await fetch("/api/ops/today");
      const opsData = await opsRes.json();
      setMeta({
        lastIngestSuccess: opsData.meta?.lastIngestSuccess,
        lastIngestFail: opsData.meta?.lastIngestFail,
        unresolvedAlerts: opsData.meta?.unresolvedAlerts || 0,
        health: opsData.health,
        staleReason: opsData.staleReason,
        verdictDateKst: opsData.verdictDateKst,
        executionDateKst: opsData.executionDateKst,
      });
      
      // Fetch job runs (would need a new API, using placeholder for now)
      setJobRuns([]);
      
    } catch (e) {
      console.error("Failed to load notifications:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleAck = async (id: string) => {
    try {
      const res = await fetch("/api/notifications/ack", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
      });
      if (res.ok) {
        // Optimistic update
        setNotifications(prev => 
          prev.map(n => n.id === id ? { ...n, resolved: true, resolved_at: new Date().toISOString() } : n)
        );
        setUnresolvedCounts(prev => ({ ...prev, total: prev.total - 1 }));
      }
    } catch (e) {
      console.error("Failed to acknowledge:", e);
    }
  };

  // Filter notifications by tab
  const filteredNotifications = notifications.filter(n => {
    if (tab === "all") return !n.resolved;
    if (tab === "resolved") return n.resolved;
    return n.level === tab && !n.resolved;
  });

  return (
    <div className="space-y-8 pb-20">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-fg flex items-center gap-3">
          <Bell size={24} />
          알림센터
          <span className="text-sm font-normal text-muted bg-surface px-2.5 py-0.5 rounded-full border border-border">
            Notifications
          </span>
          {unresolvedCounts.total > 0 && (
            <span className="text-xs bg-negative text-white px-2 py-0.5 rounded-full">
              {unresolvedCounts.total}
            </span>
          )}
        </h1>
        <button
          onClick={loadData}
          disabled={loading}
          className="p-2 rounded-lg hover:bg-surface text-muted transition-colors"
        >
          <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* A. Status Summary Cards */}
      <section>
        <h2 className="text-sm font-medium text-muted mb-3 uppercase tracking-wide">오늘 상태 요약</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* DATA Card */}
          <div className="bg-surface rounded-xl border border-border p-4">
            <div className="flex items-center gap-2 mb-2">
              <Database size={16} className="text-muted" />
              <span className="text-sm font-medium text-muted">DATA</span>
            </div>
            <div className={`text-lg font-bold ${meta?.health === "FRESH" ? "text-positive" : "text-choppy"}`}>
              {meta?.health || "—"}
            </div>
            {meta?.staleReason && (
              <div className="text-xs text-muted mt-1">{meta.staleReason}</div>
            )}
            <div className="text-xs text-muted mt-2">
              판정일: {meta?.verdictDateKst?.split("T")[0] || "—"}
            </div>
          </div>

          {/* OPS Card */}
          <div className="bg-surface rounded-xl border border-border p-4">
            <div className="flex items-center gap-2 mb-2">
              <Server size={16} className="text-muted" />
              <span className="text-sm font-medium text-muted">OPS</span>
            </div>
            {meta?.lastIngestSuccess ? (
              <>
                <div className="text-lg font-bold text-positive flex items-center gap-1">
                  <CheckCircle2 size={16} />
                  성공
                </div>
                <div className="text-xs text-muted mt-1">
                  {formatRelativeTime(meta.lastIngestSuccess)}
                </div>
              </>
            ) : meta?.lastIngestFail ? (
              <>
                <div className="text-lg font-bold text-negative flex items-center gap-1">
                  <AlertTriangle size={16} />
                  실패
                </div>
                <div className="text-xs text-muted mt-1">
                  {formatRelativeTime(meta.lastIngestFail)}
                </div>
              </>
            ) : (
              <div className="text-lg font-bold text-muted">기록 없음</div>
            )}
          </div>

          {/* EXEC Card */}
          <div className="bg-surface rounded-xl border border-border p-4">
            <div className="flex items-center gap-2 mb-2">
              <ClipboardList size={16} className="text-muted" />
              <span className="text-sm font-medium text-muted">EXEC</span>
            </div>
            <div className="text-lg font-bold text-fg">
              {meta?.executionDateKst?.split("T")[0] || "—"}
            </div>
            <div className="text-xs text-muted mt-1">
              실행 예정일
            </div>
            <Link 
              href="/records" 
              className="text-xs text-info hover:underline mt-2 inline-flex items-center gap-1"
            >
              기록 확인 <ExternalLink size={10} />
            </Link>
          </div>
        </div>
      </section>

      {/* B. Notifications Timeline */}
      <section>
        <h2 className="text-sm font-medium text-muted mb-3 uppercase tracking-wide">알림 타임라인</h2>
        
        {/* Tabs */}
        <div className="flex gap-2 mb-4 border-b border-border pb-2">
          {(["all", "action", "emergency", "resolved"] as TabId[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                tab === t 
                  ? "bg-inset text-fg" 
                  : "text-muted hover:bg-surface"
              }`}
            >
              {t === "all" && "전체"}
              {t === "action" && `Action (${unresolvedCounts.action})`}
              {t === "emergency" && `Emergency (${unresolvedCounts.emergency})`}
              {t === "resolved" && "해결됨"}
            </button>
          ))}
        </div>

        {/* Notification Cards */}
        <div className="space-y-3">
          {filteredNotifications.length === 0 ? (
            <div className="bg-surface rounded-xl border border-border p-8 text-center text-muted">
              <Bell size={24} className="mx-auto mb-2 opacity-50" />
              <span className="text-sm">알림이 없습니다</span>
            </div>
          ) : (
            filteredNotifications.map((notif) => (
              <div 
                key={notif.id}
                className={`bg-surface rounded-xl border p-4 ${
                  notif.resolved ? "border-border opacity-60" : "border-border"
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${getLevelColor(notif.level)}`}>
                        {getLevelIcon(notif.level)}
                        {notif.level.toUpperCase()}
                      </span>
                      {(() => {
                        const meta = getEventTypeMeta(notif.event_type);
                        return (
                          <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${meta.badgeClass}`}>
                            {meta.icon}
                            {meta.label}
                          </span>
                        );
                      })()}
                      <span className="text-xs text-muted">{formatRelativeTime(notif.created_at)}</span>
                    </div>
                    <div className="font-medium text-fg">{notif.title}</div>
                    {notif.body && (
                      <div className="text-sm text-muted mt-1 line-clamp-2">{notif.body}</div>
                    )}
                  </div>
                  {!notif.resolved && (
                    <button
                      onClick={() => handleAck(notif.id)}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs bg-surface hover:bg-surface rounded-lg text-muted transition-colors"
                    >
                      <Check size={12} />
                      확인
                    </button>
                  )}
                  {notif.resolved && (
                    <span className="text-xs text-positive flex items-center gap-1">
                      <CheckCircle2 size={12} />
                      해결됨
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      {/* C. Operations Log */}
      <section>
        <h2 className="text-sm font-medium text-muted mb-3 uppercase tracking-wide">
          운영 로그 (최근 7일)
        </h2>
        
        {jobRuns.length === 0 ? (
          <div className="bg-surface rounded-xl border border-border p-8 text-center text-muted">
            <Clock size={24} className="mx-auto mb-2 opacity-50" />
            <span className="text-sm">운영 로그가 없습니다</span>
            <div className="text-xs mt-2">Cron ingest 실행 시 기록됩니다</div>
          </div>
        ) : (
          <div className="bg-surface rounded-xl border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-surface/50">
                <tr>
                  <th className="text-left px-4 py-2 text-muted font-medium">작업</th>
                  <th className="text-left px-4 py-2 text-muted font-medium">시작</th>
                  <th className="text-left px-4 py-2 text-muted font-medium">상태</th>
                  <th className="text-left px-4 py-2 text-muted font-medium">결과</th>
                </tr>
              </thead>
              <tbody>
                {jobRuns.map((run) => (
                  <tr key={run.id} className="border-t border-border">
                    <td className="px-4 py-2">{run.job_type}</td>
                    <td className="px-4 py-2 text-muted">{formatRelativeTime(run.started_at)}</td>
                    <td className="px-4 py-2">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        run.status === "success" ? "bg-positive-tint text-positive" :
                        run.status === "failed" ? "bg-negative-tint text-negative" :
                        "bg-info-tint text-info"
                      }`}>
                        {run.status}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-muted">
                      {run.rows_upserted > 0 && `${run.rows_upserted} rows`}
                      {run.error && <span className="text-negative">{run.error}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
