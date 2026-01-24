#!/usr/bin/env python
"""Setup the agent-friendly development environment (.vibe).

Design goals:
- Safe-by-default (no mass code changes unless explicitly requested elsewhere).
- Windows compatible.
- No external dependencies besides those in .vibe/brain/requirements.txt.

Usage:
  python scripts/setup_vibe_env.py --project-name MyProject

Optional:
  --update-package-json   Add a safe `typecheck` npm script if missing (OFF by default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_EXCLUDE_DIRS = [
    "node_modules",
    ".git",
    ".vibe",
    "dist",
    "build",
    "out",
    "coverage",
]
DEFAULT_INCLUDE_GLOBS = ["**/*.js"]

JSCONFIG_TEMPLATE = {
    "compilerOptions": {"checkJs": True, "target": "ESNext"},
    "exclude": ["node_modules", ".vibe", "dist", "build", "out"],
}

TSCONFIG_CHECK_TEMPLATE = {
    "compilerOptions": {
        "allowJs": True,
        "checkJs": True,
        "noEmit": True,
        "target": "ESNext",
        "module": "ESNext",
        "moduleResolution": "Node",
        "skipLibCheck": True,
        "strict": False,
    },
    "exclude": ["node_modules", ".vibe", "dist", "build", "out"],
}

CONFIG_TEMPLATE = {
    "project_name": "CHANGE_ME",
    "root": ".",
    "exclude_dirs": DEFAULT_EXCLUDE_DIRS,
    "include_globs": DEFAULT_INCLUDE_GLOBS,
    "critical_tags": ["@critical"],
    "context": {"latest_file": ".vibe/context/LATEST_CONTEXT.md", "max_recent_files": 12},
    "quality_gates": {
        "cycle_block": True,
        "typecheck_block_on_increase": True,
        "complexity_warn_threshold": 15,
    },
    "profiling": {"enabled_by_default": False, "mode": "node", "entry": None},
    "resolver": {
        "aliases": {},
        "root_relative_enabled": False,
        "use_tsconfig_paths": True,
        "use_jsconfig_paths": True,
        "extensions": [".js", ".mjs", ".cjs"],
        "index_files": ["index.js", "index.mjs", "index.cjs"],
    },
    "typecheck": {
        "incremental": True,
        "tsconfig": "tsconfig.check.json",
        "tsbuildinfo": ".vibe/locks/tsc_tsconfig_check_tsbuildinfo",
    },
}

DONT_DO_THIS_TEMPLATE = """# DONT_DO_THIS (Agent Memory)

This file is intentionally short. Keep it actionable.

- Do NOT auto-generate or auto-edit JSDoc across the codebase.
- Do NOT run a repo-wide formatter/linter that creates a massive diff.
- Do NOT introduce circular dependencies. (pre-commit will block.)
- Do NOT increase the TypeScript typecheck error baseline.
- Do NOT run profiling in pre-commit. Profiling is doctor-only.
"""

AGENT_CHECKLIST_TEMPLATE = """# AGENT_CHECKLIST

## Before you start
- Read `.vibe/context/LATEST_CONTEXT.md`
- Read `.vibe/agent_memory/DONT_DO_THIS.md`
- Run impact analysis for your target file(s):
  - `python .vibe/brain/impact_analyzer.py path/to/file.js`

## While you work
- Do NOT auto-generate JSDoc.
- For exported boundary functions, write JSDoc manually (at least @param/@returns where relevant).
- If you touch @critical paths, create a checkpoint commit/tag (manual; automation OFF by default).

## Before you finish
- Ensure typecheck baseline did not increase:
  - `python .vibe/brain/typecheck_baseline.py`
- Ensure pre-commit passes:
  - `git commit` (hook runs staged-only fast checks)
- (Optional) Run fast smoke tests:
  - `python .vibe/brain/run_core_tests.py --fast`
- (Optional) If you suspect perf regressions:
  - `python .vibe/brain/doctor.py --full --profile --mode node --entry <entry.js>`
"""

AGENT_SYSTEM_PROMPT_TEMPLATE = """# AGENT_SYSTEM_PROMPT

You are operating inside a large vanilla JavaScript repository with an agent-friendly environment (`.vibe`).

Non-negotiables:
- I read `.vibe/context/LATEST_CONTEXT.md` before writing code.
- I do not auto-generate JSDoc. I may detect missing docs and propose templates only.
- I do not increase typecheck errors (baseline gate).
- I never introduce circular dependencies. If I do, the commit is blocked.
- I keep the staged-only fast loop. Full scans run via `doctor`.
- Performance profiling runs only via `doctor --profile`; source-code injection is forbidden.

Working discipline:
- Prefer minimal diffs; never mass-format the repo.
- If a change has high impact (many dependents), I recommend a checkpoint.
- I log warnings instead of crashing the tooling.
"""

PROFILE_GUIDE_TEMPLATE = """# PROFILE_GUIDE

This guide covers profiling WITHOUT source-code injection.

## Node (automated via doctor)

1) Identify an entry file (example: `src/index.js` or `server.js`).
2) Run:

```bash
python .vibe/brain/doctor.py --full --profile --mode node --entry path/to/entry.js --seconds 10
```

Artifacts:
- `.vibe/reports/performance.log`
- `.vibe/reports/performance_stats.json`

Notes:
- Node profiling is heuristic. Treat it as a "where to look first" signal.
- Keep seconds low (5–20) for developer feedback loops.

## Browser (manual)

Automated browser profiling is highly environment-dependent. Use Chrome DevTools:

1) Open DevTools → Performance.
2) Record a representative interaction.
3) Export or copy a summary of the hottest functions/files.
4) Paste the summary into `.vibe/reports/performance.log`.

Then run:

```bash
python .vibe/brain/summarizer.py --full
```

The summarizer will extract "Top 5" slow items if a performance report exists.
"""


@dataclass
class ScanStats:
    js_files: int = 0
    total_loc: int = 0


def _iter_js_files(root: Path, exclude_dirs: List[str], include_globs: List[str]) -> Iterable[Path]:
    exclude_set = {d.strip("/\\") for d in exclude_dirs}

    def is_excluded(p: Path) -> bool:
        parts = {part for part in p.parts}
        return any(x in parts for x in exclude_set)

    seen = set()
    for glob_pat in include_globs:
        for p in root.glob(glob_pat):
            if not p.is_file():
                continue
            if p.suffix.lower() != ".js":
                continue
            if is_excluded(p):
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            yield p


def scan_repo(root: Path, exclude_dirs: List[str], include_globs: List[str]) -> ScanStats:
    stats = ScanStats()
    for p in _iter_js_files(root, exclude_dirs, include_globs):
        stats.js_files += 1
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                stats.total_loc += sum(1 for _ in f)
        except Exception:
            # LOC is best-effort.
            pass
    return stats


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_json_if_missing(path: Path, obj, *, force: bool) -> Tuple[bool, str]:
    if path.exists() and not force:
        return False, "exists"
    write_json(path, obj)
    return True, "written"


def _update_package_json_typecheck(root: Path) -> Tuple[bool, str]:
    """Best-effort package.json update. Safe-by-default:

    - Only touches package.json when explicitly requested.
    - Only adds scripts.typecheck when missing.
    - Never rewrites other fields.
    """
    pkg = root / "package.json"
    if not pkg.exists():
        return False, "missing"

    obj = _read_json(pkg)
    if not isinstance(obj, dict):
        return False, "parse_error"
    scripts = obj.get("scripts")
    if scripts is None:
        scripts = {}
        obj["scripts"] = scripts
    if not isinstance(scripts, dict):
        return False, "bad_scripts"
    if "typecheck" in scripts:
        return False, "already"

    scripts["typecheck"] = "tsc -p tsconfig.check.json"
    try:
        pkg.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        return True, "updated"
    except Exception as e:
        return False, f"write_error:{e}"


def ensure_file(path: Path, content: str, *, force: bool) -> Tuple[bool, str]:
    if path.exists() and not force:
        return False, "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True, "written"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-name", default=None)
    ap.add_argument("--root", default=".")
    ap.add_argument("--force", action="store_true", help="Overwrite existing template files")
    ap.add_argument(
        "--update-package-json",
        action="store_true",
        help="If package.json exists, add scripts.typecheck when missing (OFF by default)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    vibe = root / ".vibe"

    # Directories
    for d in ["brain", "context", "db", "reports", "agent_memory", "locks"]:
        (vibe / d).mkdir(parents=True, exist_ok=True)

    # Config
    cfg_path = vibe / "config.json"
    cfg_obj = dict(CONFIG_TEMPLATE)
    if args.project_name:
        cfg_obj["project_name"] = args.project_name
    write_json(cfg_path, cfg_obj) if (not cfg_path.exists() or args.force) else None

    # Root TypeScript/JS config (safe: only create if missing unless --force)
    _write_json_if_missing(root / "jsconfig.json", JSCONFIG_TEMPLATE, force=args.force)
    _write_json_if_missing(root / "tsconfig.check.json", TSCONFIG_CHECK_TEMPLATE, force=args.force)

    # Requirements
    req_path = vibe / "brain" / "requirements.txt"
    ensure_file(req_path, "watchdog\npython-dateutil\n", force=args.force)

    # Agent docs
    ensure_file(vibe / "agent_memory" / "DONT_DO_THIS.md", DONT_DO_THIS_TEMPLATE, force=args.force)
    ensure_file(vibe / "AGENT_CHECKLIST.md", AGENT_CHECKLIST_TEMPLATE, force=args.force)

    # System prompt doc (root-level)
    ensure_file(root / "AGENT_SYSTEM_PROMPT.md", AGENT_SYSTEM_PROMPT_TEMPLATE, force=args.force)

    # Profiling guide
    ensure_file(vibe / "context" / "PROFILE_GUIDE.md", PROFILE_GUIDE_TEMPLATE, force=args.force)

    # Initial LATEST_CONTEXT placeholder (summarizer will overwrite; this avoids first-run confusion)
    ensure_file(
        vibe / "context" / "LATEST_CONTEXT.md",
        "# LATEST_CONTEXT\n\nGenerated: (not yet)\n\nRun: `python .vibe/brain/summarizer.py --full`\n",
        force=False,
    )

    # Step 0 scan report
    stats = scan_repo(root, cfg_obj["exclude_dirs"], cfg_obj["include_globs"])
    scan_report = {
        "project_root": str(root),
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "js_files": stats.js_files,
        "approx_loc": stats.total_loc,
        "exclude_dirs": cfg_obj["exclude_dirs"],
        "include_globs": cfg_obj["include_globs"],
    }
    write_json(vibe / "reports" / "scan_report.json", scan_report)

    # Optional: safe package.json script addition (explicit flag only)
    if args.update_package_json:
        ok, msg = _update_package_json_typecheck(root)
        if ok:
            print("[vibe] package.json: added scripts.typecheck")
        else:
            if msg == "missing":
                print("[vibe] package.json: not found (skipped)")
            elif msg == "already":
                print("[vibe] package.json: scripts.typecheck already exists (skipped)")
            else:
                print(f"[vibe] package.json: could not update ({msg})")

    print("[vibe] setup complete")
    print(f"[vibe] root: {root}")
    print(f"[vibe] js files (approx): {stats.js_files}")
    print(f"[vibe] loc (approx): {stats.total_loc}")
    print("[vibe] next:")
    print("  1) pip install -r .vibe/brain/requirements.txt")
    print("  2) python .vibe/brain/indexer.py --scan-all")
    print("  3) python .vibe/brain/summarizer.py --full")
    print("  4) python .vibe/brain/typecheck_baseline.py --init")
    print("  5) python scripts/install_hooks.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
