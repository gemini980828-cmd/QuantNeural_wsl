#!/usr/bin/env python
"""Install Git hooks for .vibe.

Requirements:
- Create/update .git/hooks/pre-commit
- Hook must call: python .vibe/brain/precommit.py
- Windows compatible; avoid bash dependencies.

This script does NOT enable any auto-commit/auto-rollback behavior.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

HOOK_BODY = """#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    # Resolve repo root as the directory containing this hook's parent (../..)
    hook_path = Path(__file__).resolve()
    repo_root = hook_path.parent.parent

    precommit = repo_root / '.vibe' / 'brain' / 'precommit.py'
    if not precommit.exists():
        print('[vibe][hook] missing .vibe/brain/precommit.py', file=sys.stderr)
        return 1

    cmd = [sys.executable, str(precommit)]
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == '__main__':
    raise SystemExit(main())
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    git_dir = root / '.git'
    if not git_dir.exists():
        print('[vibe][install_hooks] .git not found. Run this at repo root.', file=sys.stderr)
        return 1

    hooks_dir = git_dir / 'hooks'
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_path = hooks_dir / 'pre-commit'
    hook_path.write_text(HOOK_BODY, encoding='utf-8', newline='\n')

    # Best-effort executable bit (harmless on Windows)
    try:
        mode = hook_path.stat().st_mode
        hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

    print('[vibe][install_hooks] installed .git/hooks/pre-commit')
    print('[vibe][install_hooks] hook runs staged-only checks via .vibe/brain/precommit.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
