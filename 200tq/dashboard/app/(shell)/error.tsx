'use client';

import { useEffect } from "react";
import { AlertTriangle } from "lucide-react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[ShellError]", error);
  }, [error]);

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="flex flex-col items-center gap-3 text-center">
        <AlertTriangle className="w-8 h-8 text-negative" />
        <h2 className="text-lg font-semibold text-fg">오류가 발생했습니다</h2>
        <p className="text-sm text-muted max-w-md">{error.message}</p>
        {error.digest && (
          <p className="text-xs text-muted font-mono">Digest: {error.digest}</p>
        )}
        <button
          onClick={reset}
          className="mt-2 px-4 py-2 bg-surface rounded-lg text-sm text-fg hover:bg-inset border border-border transition-colors"
        >
          다시 시도
        </button>
      </div>
    </div>
  );
}
