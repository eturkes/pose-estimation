# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

---

## 2026-05-16 — `.claude/` infrastructure seeded

Initial setup of `.claude/` (INDEX, tech/, memory/, prompts/) and migration off `AGENTS.md`. No code changes. Watch for future drift between `.claude/tech/*.md` and the code itself; the `test_public_api.py` test guards `__init__.py` re-exports but nothing else guards the other tech notes.

Open follow-ups (not yet decisions):
- Consider a lightweight CI/precommit doc-drift check (e.g. assert key file paths mentioned in `tech/*.md` exist).
- `analysis_summary.html` is committed and large (~5 MB); future audit can decide if it should be tracked via Git LFS or moved to a release artifact.
