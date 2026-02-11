"use client";

import { useEffect } from "react";
import { CheckCircle, AlertTriangle, Info } from "lucide-react";
import { useToastStore, type ToastItem } from "@/lib/stores/toast-store";

const ICON_MAP = {
  success: <CheckCircle size={16} className="text-positive shrink-0" />,
  error: <AlertTriangle size={16} className="text-negative shrink-0" />,
  info: <Info size={16} className="text-blue-400 shrink-0" />,
};

function ToastSlot({ toast }: { toast: ToastItem }) {
  const removeToast = useToastStore((s) => s.removeToast);

  useEffect(() => {
    const timer = setTimeout(() => removeToast(toast.id), toast.duration);
    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, removeToast]);

  return (
    <div className="animate-fade-in-up">
      <div className="bg-neutral-800 text-white px-4 py-2 rounded-lg shadow-xl border border-neutral-700 flex items-center gap-2 text-sm font-medium">
        {ICON_MAP[toast.type]}
        {toast.message}
      </div>
    </div>
  );
}

export function ToastProvider() {
  const toasts = useToastStore((s) => s.toasts);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[100] flex flex-col-reverse gap-2">
      {toasts.map((toast) => (
        <ToastSlot key={toast.id} toast={toast} />
      ))}
    </div>
  );
}
