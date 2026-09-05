# Upstream instruction drop-ins

`CLAUDE.md` + `.claude/commands/*.md` arrive as upstream drop-ins landing over this repo's local
adaptations. This file is the refresh-safe copy of the clauses they overwrite: it is project law,
outranks conflicting drop-in text, and loads for MAIN + teammates alike.

Reconciliation, every refresh: diff each refreshed file against its prior commit, re-apply every
clause below, commit `state: …`. `git log --grep upstream` lists prior reconciliations. Verify with
`rg -n 'DESCOPED|agent/contracts' .claude/commands/ CLAUDE.md` → the DESCOPED clause present, zero
`agent/contracts` hits.

1. **DESCOPED = a terminal milestone status** (`session-roadmap.md:3`; three successive refreshes
   dropped it). Active milestone = first lacking a terminal status (DONE / REVIEWED / DESCOPED). A
   DESCOPED milestone is closed by decision rather than by completion: its retained detail lives in
   the `.agent/archive/` record the roadmap stub names, and reviving any part of it is a PLANNING
   ruling, never a dispatch outcome. M3 is DESCOPED, and its roadmap heading declares its own
   terminality as the second guard.
2. **Acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`** — never upstream's
   `.agent/contracts/m<m>u<u>*.md`. Upstream's requirements (committed, outside the attached set,
   read on demand, dispatchable by MILESTONE-REVIEW) are all met at the archive path, and ≥15 files
   bind it: `.agent/`, `tests/`, `scripts/`, plus a generated data field —
   `scripts/make_calibration_qc_fixtures.py` writes the path into
   `tests/fixtures/calibration_qc_set/manifest.json`, and `check_calibration_qc_fixtures.py`
   validates digests without resolving that field, so a rename missing the generator leaves a
   dangling pointer no gate reports. Sweep with `rg -c --hidden 'archive/contract-'`; `.agent/` is
   hidden, so an unflagged sweep reports the non-dot files alone.
