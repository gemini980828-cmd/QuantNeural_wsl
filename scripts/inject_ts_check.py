#!/usr/bin/env python
"""Safely inject `// @ts-check` into JavaScript files.

IMPORTANT: This can create a large diff. Therefore:
- Default is DRY RUN.
- Use --apply to write changes.

Requirements (per spec):
- Add `// @ts-check` at file top unless already present.
- If a shebang exists, insert on the next line.
- Respect exclude_dirs.
- Idempotent.
- Print changed/skipped stats.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

TS_CHECK_LINE = "// @ts-check\n"
SHEBANG_RE = re.compile(r"^#!")
TS_CHECK_RE = re.compile(r"^\s*//\s*@ts-check\b")


def load_config(root: Path) -> Dict:
    cfg_path = root / ".vibe" / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # fallback minimal
    return {
        "exclude_dirs": ["node_modules", ".git", ".vibe", "dist", "build", "out", "coverage"],
        "include_globs": ["**/*.js"],
        "root": ".",
    }


def iter_js_files(root: Path, exclude_dirs: List[str], include_globs: List[str]) -> List[Path]:
    exclude_set = {d.strip("/\\") for d in exclude_dirs}

    def is_excluded(p: Path) -> bool:
        parts = {part for part in p.parts}
        return any(x in parts for x in exclude_set)

    seen = set()
    out: List[Path] = []
    for glob_pat in include_globs:
        for p in root.glob(glob_pat):
            if not p.is_file() or p.suffix.lower() != ".js":
                continue
            if is_excluded(p):
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
    return sorted(out)


def inject_one(path: Path) -> Tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"read_error:{e}"

    lines = text.splitlines(keepends=True)
    if not lines:
        # empty file
        return False, "empty"

    # already present in first few lines
    for i in range(min(10, len(lines))):
        if TS_CHECK_RE.search(lines[i]):
            return False, "already"

    insert_at = 0
    if SHEBANG_RE.match(lines[0]):
        insert_at = 1

    # idempotency: if the insert position already contains it
    if insert_at < len(lines) and TS_CHECK_RE.search(lines[insert_at]):
        return False, "already"

    lines.insert(insert_at, TS_CHECK_LINE)
    new_text = "".join(lines)

    if new_text == text:
        return False, "noop"

    # Normalize line endings to \n only (git can manage CRLF if needed)
    new_text = new_text.replace("\r\n", "\n")
    return True, new_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true", help="Write changes to disk")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N files (0=all)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cfg = load_config(root)
    exclude_dirs = cfg.get("exclude_dirs") or ["node_modules", ".git", ".vibe", "dist", "build", "out", "coverage"]
    include_globs = cfg.get("include_globs") or ["**/*.js"]

    files = iter_js_files(root, exclude_dirs, include_globs)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    changed = 0
    skipped = 0
    errors = 0

    for p in files:
        ok, result = inject_one(p)
        if not ok:
            if result.startswith("read_error"):
                errors += 1
            else:
                skipped += 1
            continue

        if args.apply:
            try:
                p.write_text(result, encoding="utf-8", newline="\n")
                changed += 1
            except Exception:
                errors += 1
        else:
            changed += 1

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(f"[vibe][inject_ts_check] mode={mode}")
    print(f"[vibe][inject_ts_check] files_total={len(files)}")
    print(f"[vibe][inject_ts_check] would_change={changed}" if not args.apply else f"[vibe][inject_ts_check] changed={changed}")
    print(f"[vibe][inject_ts_check] skipped={skipped}")
    print(f"[vibe][inject_ts_check] errors={errors}")

    if not args.apply:
        print("[vibe][inject_ts_check] NOTE: Dry-run only. Re-run with --apply to write.")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
