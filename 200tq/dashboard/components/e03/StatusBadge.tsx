import { ReactNode } from "react";

export type StatusTone = "inactive" | "action" | "danger" | "ok" | "neutral" | "info";

interface StatusBadgeProps {
  tone: StatusTone;
  children: ReactNode;
  className?: string;
  detail?: string;
}

export function StatusBadge({ tone, children, className = "", detail }: StatusBadgeProps) {
  const getColors = (t: StatusTone) => {
    switch (t) {
      case "inactive": return "bg-status-inactive-bg text-status-inactive-fg border-transparent";
      case "action": return "bg-status-action-bg/10 text-status-action-fg border-status-action-bg/20";
      case "danger": return "bg-status-danger-bg/10 text-status-danger-bg border-status-danger-bg/20";
      case "ok": return "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20";
      case "neutral": return "bg-neutral-500/10 text-neutral-600 dark:text-neutral-300 border-neutral-500/20";
      case "info": return "bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/20";
    }
  };

  const colors = getColors(tone);

  return (
    <div className={`flex items-center gap-1.5 px-2.5 h-[24px] rounded-full border text-[11px] font-bold tracking-wide uppercase ${colors} ${className}`}>
      <div className={`w-1.5 h-1.5 rounded-full ${
         tone === "inactive" ? "bg-status-inactive-fg" :
         tone === "action" ? "bg-status-action-bg" :
         tone === "danger" ? "bg-status-danger-bg" :
         tone === "ok" ? "bg-emerald-500" :
         "bg-current"
      }`} />
      <span>{children}</span>
      {detail && (
         <span className="opacity-60 border-l border-current pl-1.5 ml-0.5">{detail}</span>
      )}
    </div>
  );
}
