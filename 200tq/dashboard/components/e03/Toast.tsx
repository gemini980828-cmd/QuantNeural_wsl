"use client";

/**
 * @deprecated Use `useToast()` from `@/lib/stores/toast-store` instead.
 * This component is no longer imported anywhere.
 * Retained for reference — safe to delete.
 */

import { useEffect } from "react";

interface ToastProps {
  message: string;
  onClose: () => void;
  duration?: number;
}

export default function Toast({ message, onClose, duration = 2500 }: ToastProps) {
  useEffect(() => {
    const timer = setTimeout(onClose, duration);
    return () => clearTimeout(timer);
  }, [duration, onClose]);

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[100] animate-fade-in-up">
       <div className="bg-neutral-800 text-white px-4 py-2 rounded-lg shadow-xl border border-neutral-700 flex items-center gap-2 text-sm font-medium">
         {message}
       </div>
    </div>
  );
}
