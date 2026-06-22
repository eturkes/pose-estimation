Continue this project (fresh session). Task below non-empty ⇒ sole task; do exactly it, leave `.agent/roadmap.md` untouched unless it directs otherwise. Empty ⇒ run the MODE selected from the roadmap's active milestone (first one not DONE/REVIEWED).

Load `.agent/roadmap.md` (milestone ledger + active-milestone detail), then `.agent/memory.md` (lessons + decisions); CLAUDE.md is auto-injected. Read only what the step implicates — subsystem reference `.claude/tech/*.md` (filenames index it). Navigate Python via Serena/LSP, R via grep.

MODE ← active milestone status:
- UNPLANNED → PLANNING
- IN-PROGRESS (an open unit) → WORK-UNIT (lowest open unit)
- IMPLEMENTED (all units done, not yet reviewed) → MILESTONE-REVIEW

All modes advance milestone/unit status in the roadmap and end on a scoped local commit (convention below). Record context-usage in WORK-UNIT only.

PLANNING — always a dynamic workflow + web search.
- Read: the prior milestone's commit range, esp. its recorded context-usage (right-sizes units); for the first milestone, the seed/outline commit(s).
- Do: divide the outline into milestones if not yet split; plan ONLY the next milestone — break it into units each completable within a 200K window, footage-independent units sequenced first. Footage-gated milestone: if no real 3-camera session exists yet under `videos/`, record the standing block and stop (skip the workflow); once one exists, plan its footage-independent units first and defer the gated ones.
- Close: commit `roadmap (M<m> plan): …`. I compact, then call `/codex-review`; fix accepted findings → follow-up commit (same tag + `Codex-Review:` trailer).

WORK-UNIT.
- Read: the last completed unit's commit(s) — or the planning commit(s) if this is the milestone's first unit.
- Do: (1) restate the unit + acceptance in one line; (2) implement, reusing modules, matching surrounding style; (3) FOOTAGE GATE — a footage-gated unit needs a real 3-camera session (`session.json` + `calibration.json` + synced per-camera streams under `videos/`); absent it, stop and report the block — every result traces to real captures; (4) VERIFY `uv run ruff check` / `ruff format --check` / `ty check` / `pytest`, touched R scripts exit 0; (5) record durable lessons/decisions in `.agent/memory.md`.
- Close: record the unit's context-usage (`.agent/compaction.sh`) into the roadmap; commit `<scope> (M<m>.<u>): …`. I compact, then call `/codex-review`; fix accepted findings → follow-up commit (same tag + `Codex-Review:` trailer).

MILESTONE-REVIEW — I launch this with 1M context (the only 1M session).
- Read: every commit of the milestone, planning commits included.
- Do: adversarially review the milestone's whole body — correctness, claim-vs-guarantee gaps, cross-unit consistency; fix what you find. Skip context-usage recording.
- Close: commit `<scope> (M<m> review): …` and set the milestone REVIEWED. I call `/codex-review` WITHOUT compacting; fix accepted findings → follow-up commit (same tag + `Codex-Review:` trailer). Next session PLANS the next milestone.

Commit convention — scoped (`<scope>: …`), trace key in parens: unit `(M<m>.<u>)`, plan `(M<m> plan)`, review `(M<m> review)`. Codex-review follow-ups keep the tag and add a `Codex-Review: <accepted findings>` trailer. Grep a milestone's history: `git log --grep "(M<m>"`.

Task (may be empty): $ARGUMENTS
