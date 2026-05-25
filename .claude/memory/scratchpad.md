# Scratchpad

Transient working notes — anything from "current investigation" to "half-finished idea". Safe to prune at session boundaries. Never treat as a source of truth; promote stable findings to `tech/`, decisions to `decisions.md`, lessons to `lessons.md`.

## How to use

- Start each session by skimming the most-recent entries (top of file).
- Append `## YYYY-MM-DD HH:MM — <session topic>` and write freely.
- When closing a session, either prune the entry or summarise it into a longer-lived file.

---

## 2026-05-25 — CLAUDE.md alignment audit

User updated CLAUDE.md with expanded directives (commit `2fe51fd`). This session verified downstream file alignment:

- **conventions.md**: Updated git section — added commit-timing rule ("commit before end-of-turn that closes cohesive work; defer mid-iteration") and LLM-optimized message framing.
- **INDEX.md**: Added `prompts/sessions.md` to manifest (was missing despite file existing).
- **AGENTS.md, kickoff.md, INDEX.md agent-writable line**: Already propagated by previous session — verified correct.
- **decisions.md**: Previous session's `2026-05-24 — CLAUDE.md revision` entry covers the policy change. No new decision needed.
- **Scratchpad**: Pruned old 2026-05-24 entries (all findings were already promoted to tech notes and decisions).

Key new CLAUDE.md directives beyond agent-write permission:
- Commit timing: end-of-cohesive-work, defer mid-iteration. Messages optimized for LLM parsing.
- Directory scoping: constrain to launch dir + children.
- Security audit scheduling + software updates.
- Test suite guidance: permissible but avoid overtesting.
- KISS/UNIX/overengineering awareness.
- Expanded objectivity: first principles, scientific method, benchmarking.
- Memory drift prevention: "must diligently keep up-to-date."

**Project status:** Phase 1 ✓ (stability), Phase 2 ✓ (clinical metrics), Phase 3 pending (3D pipeline, awaiting footage), Phase 4 ongoing (maintenance).

---

## Prior session summary (2026-05-24)

All entries pruned — findings promoted to:
- `decisions.md`: 10 entries covering roadmap, jitter fixes, adaptive smoothing, clinical metrics (2A-2D), session orchestration, refactor analysis, path traversal fix, R 4.6 migration, clinical pipeline E2E, CLAUDE.md revision.
- `lessons.md`: 4 entries covering path traversal validation, R 4.6 renv workflow, if_else vector length, AGENTS.md drift.
- `tech/analysis.md`: Full clinical metric documentation (bilateral, quality, trunk, segmentation).
- `tech/conventions.md`: Agent-writable files section, updated git guidance.
- Test count: 252 passing.
