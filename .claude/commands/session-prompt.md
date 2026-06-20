Continue this project (fresh session). The task below: non-empty ⇒ it is your sole task; do exactly it and leave `.agent/roadmap.md` untouched unless it directs otherwise. Empty ⇒ take the next open step in `.agent/roadmap.md`.

Load `.agent/roadmap.md` (plan + status), then `.agent/memory.md` (lessons + decisions); CLAUDE.md is already in context. Read only what the step implicates — subsystem reference is `.claude/tech/*.md` (filenames index it). Navigate Python via Serena/LSP, R via grep.

Execution loop:
1. Restate the step + its acceptance in one line.
2. Implement; reuse existing modules, match surrounding style.
3. Phase 2 needs real footage — if `rehab/data/videos` holds none, stop and report the block; every result must trace to real captures.
4. Verify: `uv run ruff check` / `ruff format --check` / `ty check` / `pytest`; touched R scripts exit 0.
5. Record durable lessons/decisions in `.agent/memory.md`; advance `.agent/roadmap.md` status if you closed a step; scoped local commit.

Task (may be empty): $ARGUMENTS
