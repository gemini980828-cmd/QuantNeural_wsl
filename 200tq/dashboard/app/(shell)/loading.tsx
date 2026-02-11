import { RefreshCw } from "lucide-react";

export default function Loading() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="flex flex-col items-center gap-3 text-muted">
        <RefreshCw className="w-8 h-8 animate-spin" />
        <span className="text-sm">로딩 중...</span>
      </div>
    </div>
  );
}
