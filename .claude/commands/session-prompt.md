Continue this project (fresh session). Task below non-empty ⇒ sole task; do exactly it, leave `.agent/roadmap.md` untouched unless it directs otherwise. Empty ⇒ run the MODE selected from the roadmap's active milestone (first one not DONE/REVIEWED).

Load `.agent/roadmap.md` (milestone ledger + active-milestone detail), then `.agent/memory.md` (lessons + decisions); CLAUDE.md is auto-injected. Read only what the step implicates — subsystem reference `.claude/tech/*.md` (filenames index it). Navigate Python via Serena/LSP, R via grep.

MODE ← active milestone status (each mode advances status, then ends on a scoped commit; convention below):
- UNPLANNED (incl. a not-yet-split future milestone) → PLANNING
- IN-PROGRESS (has an OPEN unit) → WORK-UNIT (lowest OPEN unit)
- IMPLEMENTED (units all DONE, unreviewed) → MILESTONE-REVIEW

Record context-usage in WORK-UNIT only.

PLANNING — splits the outline into milestones if not yet split, then plans ONLY the next milestone.
- Read: the prior milestone's commit range, esp. recorded context-usage (right-sizes units); for the first planned milestone, the outline-seed commit(s) the roadmap names.
- Gate first: a footage-gated milestone with no confirmed real 3-cam session stops here — record the standing block, no workflow. Confirm footage functionally (resolve the session via the pipeline), not by reading the deny-listed `videos/` tree.
- Plan (once unblocked): always a dynamic workflow + web search. Break the milestone into units each completable within a 200K window; sequence footage-independent prep first; flag any still-gated unit (e.g. agreement needs a baseline) BLOCKED — planned, not yet runnable.
- Close: set the milestone IN-PROGRESS (units enumerated), then commit `roadmap (M<m> plan): …`. I compact, then `/codex-review`; fix accepted findings → follow-up commit (same tag + `Codex-Review:` trailer).

WORK-UNIT.
- Read: the last completed unit's commit(s) — or the planning commit(s) if this is the milestone's first unit.
- Do: (1) restate the unit + acceptance in one line; (2) implement, reusing modules, matching surrounding style; (3) FOOTAGE GATE — a footage-gated unit needs a real 3-cam session (`session.json` + `calibration.json` + synced per-camera streams); confirm it functionally (resolve via the pipeline), not by reading the deny-listed `videos/` tree; unconfirmed ⇒ stop and report the block, so every result traces to real captures; (4) VERIFY `uv run ruff check` / `uv run ruff format --check` / `uv run ty check` / `uv run pytest`, touched R scripts exit 0; (5) record durable lessons/decisions in `.agent/memory.md`.
- Close: record the unit's context-usage (`.agent/compaction.sh`, full `pct used/window`) into the roadmap; set the unit DONE — and the milestone IMPLEMENTED if no OPEN unit remains; then commit `<scope> (M<m>.<u>): …`. I compact, then `/codex-review`; fix accepted findings → follow-up commit (same tag + `Codex-Review:` trailer).

MILESTONE-REVIEW — I launch this with 1M context (the only 1M session).
- Read: every commit of the milestone, planning commits included.
- Do: adversarially review the milestone's whole body — correctness, claim-vs-guarantee gaps, cross-unit consistency; fix what you find. Skip context-usage recording.
- Close: set the milestone REVIEWED, then commit `<scope> (M<m> review): …`. I call `/codex-review` WITHOUT compacting; fix accepted findings → follow-up commit (same tag + `Codex-Review:` trailer). Next session PLANS the next milestone.

Commit convention — scoped (`<scope>: …`), trace key in parens: unit `(M<m>.<u>)`, plan `(M<m> plan)`, review `(M<m> review)`. Codex-review follow-ups keep the tag and add a `Codex-Review: <accepted findings>` trailer. Grep a milestone's history: `git log --grep "(M<m>[. ]"`.

Task (may be empty): $ARGUMENTS
