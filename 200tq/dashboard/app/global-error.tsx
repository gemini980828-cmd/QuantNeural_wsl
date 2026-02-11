'use client';

import { AlertTriangle } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#0B0F14",
          color: "#E6E6E8",
          fontFamily: "'Inter', system-ui, sans-serif",
        }}
      >
        <div style={{ textAlign: "center", maxWidth: 400, padding: "0 1rem" }}>
          <AlertTriangle
            style={{ width: 48, height: 48, color: "#FCA5A5", margin: "0 auto 1rem" }}
          />
          <h1 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "0.5rem" }}>
            치명적 오류
          </h1>
          <p style={{ fontSize: "0.875rem", color: "#B4B9BF", marginBottom: "1.5rem" }}>
            페이지를 표시할 수 없습니다. 잠시 후 다시 시도해주세요.
          </p>
          {error.digest && (
            <p style={{ fontSize: "0.75rem", color: "#787E87", fontFamily: "monospace", marginBottom: "1rem" }}>
              Ref: {error.digest}
            </p>
          )}
          <button
            onClick={reset}
            style={{
              padding: "0.5rem 1.5rem",
              backgroundColor: "#111827",
              color: "#E6E6E8",
              border: "1px solid #1F2937",
              borderRadius: "0.5rem",
              fontSize: "0.875rem",
              cursor: "pointer",
            }}
          >
            다시 시도
          </button>
        </div>
      </body>
    </html>
  );
}
