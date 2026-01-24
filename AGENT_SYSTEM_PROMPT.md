# AGENT_SYSTEM_PROMPT

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
