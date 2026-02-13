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
      case "action": return "bg-choppy-tint text-status-action-fg border-status-action-bg/20";
      case "danger": return "bg-negative-tint text-status-danger-bg border-status-danger-bg/20";
      case "ok": return "bg-positive-tint text-positive dark:text-positive border-positive/20";
      case "neutral": return "bg-muted/10 text-muted border-muted/20";
      case "info": return "bg-accent-tint text-accent dark:text-accent border-accent/20";
    }
  };

  const colors = getColors(tone);

  return (
    <div className={`flex items-center gap-1.5 px-2.5 h-[24px] rounded-full border text-[11px] font-bold tracking-wide uppercase ${colors} ${className}`}>
      <div className={`w-1.5 h-1.5 rounded-full ${
         tone === "inactive" ? "bg-status-inactive-fg" :
         tone === "action" ? "bg-status-action-bg" :
         tone === "danger" ? "bg-status-danger-bg" :
         tone === "ok" ? "bg-positive" :
         "bg-current"
      }`} />
      <span>{children}</span>
      {detail && (
         <span className="opacity-60 border-l border-current pl-1.5 ml-0.5">{detail}</span>
      )}
    </div>
  );
}
