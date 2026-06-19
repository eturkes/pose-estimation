Continue this project (fresh session). `CLAUDE.md`: operating contract.

`$ARGUMENTS` — when set, that is the task; ignore the roadmap. When empty, take the next open step in `.agent/roadmap.md`.

Load first: `CLAUDE.md` (how you operate), `.agent/roadmap.md` (plan + status), `.agent/memory.md` (lessons + decisions). Then read only what the step implicates — subsystem reference lives in `.claude/tech/*.md` (filenames are the index).

Then:
1. Restate the step + its acceptance in one line.
2. Implement. Reuse existing modules; match surrounding style. Navigate Python via Serena/LSP, R via grep.
3. If a Phase 2 step needs real footage and `rehab/data/videos` holds none, stop and report the block — never fabricate results.
4. Verify: `uv run ruff check` / `ruff format --check` / `ty check` / `pytest`; touched R scripts exit 0.
5. Record durable lessons/decisions in `.agent/memory.md`; update status + next in `.agent/roadmap.md`.
6. Scoped commit (local only). Watch context with `sh .agent/compaction.sh`; wrap up near 80%.
