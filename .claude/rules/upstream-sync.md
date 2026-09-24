---
paths:
  - "CLAUDE.md"
  - ".claude/rules/*.md"
  - ".agent/*.md"
  - ".agent/archive/*.md"
---

# Upstream instruction sync

`CLAUDE.md` arrives as an upstream drop-in landing over this repo's local adaptations.
**The clauses below are the only surviving copy.**

**Procedure, every refresh:** diff the refreshed file against its prior commit, re-apply every
clause below, keep every upstream change no clause contradicts, commit `state: …`. Verify with
`rg -l 'archive/contract-' scripts/ src/ tests/` = 7, re-derived rather than trusted — a
whole-tree sweep counts every document that merely mentions the path and drifts on every edit, so
it carries no figure here (Clause 3). `git log -- CLAUDE.md` lists priors (a message grep on
`upstream` also hits non-refresh commits). A refresh that
retires a constant also falsifies claims elsewhere → sweep `.agent/` and `.claude/rules/` for the
retired term and correct what depended on it, since a stale sizing datum reaches planning as a
budget. Both dot-dirs must be named or `--hidden` passed, or the sweep reads clean over unread
files (→ `evidence.md`). **A purely additive clause is not a no-op either**: contradicting nothing, it
still binds mechanisms this repo already runs its own way, so resolve every new clause against the
local mechanism before recording a refresh clean — `CLAUDE.md`'s `Verification integrity` bullet
binds a red-witness rule whose local form is a targeted run, because the decisive gate has to close
green (→ `gates.md`). **A cut is not a reversal either**: guidance upstream drops as redundant
with the current model (the UI/UX style + report-every-issue lines) stays expected behaviour, so
keep every local application of it standing. A retired mechanism is the other kind: its local
dependents re-derive (`/goal` → plain phase sessions, flow kept). The upstream commit + the
user's refresh note decide which; ask when neither does.

- **Clause 1 — acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`**, never at
  `.agent/contracts/`. Upstream's requirements — committed, outside the attached set, read on
  demand — are all met at the archive path, and **7 files under `scripts/ src/ tests/` break if
  it moves**, one of them a generated data field: `scripts/make_calibration_qc_fixtures.py` writes
  the path into `tests/fixtures/calibration_qc_set/manifest.json`, and
  `check_calibration_qc_fixtures.py` validates digests without resolving that field, so a rename
  missing the generator leaves a dangling pointer no gate reports.
- **Clause 2 — the grain splits: a review PASS terminates on rows adjudicated, a review WAVE
  closes on fixes applied.** `CLAUDE.md`'s termination rule quantifies over the pass and is kept
  whole — check set fixed before the diff is read, every row adjudicated, an all-`pass` table
  complete, no open-ended re-review after it. Closure quantifies over the wave: an adjudicated row
  whose ruling accepts a fix stays open until that fix closes on its own acceptance check under
  MAIN's rerun, so a wave that rules every row and applies none has not closed and re-enters to
  apply them. Measured: M2's wave 1 adjudicated 193 rows / 55 fails in one pass and needed three
  further sessions to close 40 of its 43 accepted fixes. Reading the clause as an override instead
  of a grain split takes upstream's anti-flip-flop mechanism down as collateral (grain law →
  `evidence.md`).
- **Clause 3 — mutable state belongs in `.agent/spec.md` + `.agent/deferred.md`, never in
  `.claude/rules/`.** A rule file is read as standing law, so a mid-session ledger parked there
  reads as current long after it stops being true. Anything MAIN rewrites while it works — status,
  open rows, counts — stays in those two; `.claude/rules/` takes only what holds until new
  evidence reverses it, and points at the queue rather than restating a row.
- **Clause 4 — `.agent/archive/` is this repo's detail store and its records are frozen.**
  `roadmap.md`, `polish.md`, `review-m2.md`, the 14 unit contracts and the reviewer reports live
  there and keep their own stale pointers (`/session-roadmap`, `.agent/memory.md`, a `Read()`
  deny list). Read an archive pointer as a citation of its own time. The live surfaces are
  `.agent/spec.md`, `.agent/deferred.md` and `.claude/rules/`.

**Superseded by upstream — never restore.** The `Read()` path-exclusion control (→
`data-boundary.md`). The `N% NK/1M` gauge convention: MAIN's window is whatever
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` clamps it to, so no literal belongs in it. `.agent/roadmap.md`
and `.agent/polish.md` as attached state, and the `/session-roadmap`, `/session-prompt`,
`/session-polish` commands that consumed them — `.agent/spec.md` is now the sole attached state
and the phase flow replaces the MODE dispatch. The **`≤ 8 KB` cap on `.agent/spec.md`**: liveness
replaces it — every line binds current or future work, superseded text dies in the commit that
supersedes it, size is emergent. A byte budget rewards compressing live prose over deleting dead
rows, which is the opposite of the ranking. **The deferral queue as a `spec.md` section**: a queue
is monotonic, so attached it is a permanent growth term; it lives at `.agent/deferred.md`, which
every queue write names by path, while bare `Deferred` = the `spec.md` section, carrying the queue
pointer plus the unfinished units = the current spine. **The dispatch class map as project scope**:
trigger, shape→role map and the six solo licences live in global `CLAUDE.md` `Subagents` b1, so
`CLAUDE.md` and `.claude/rules/delegation.md` point at it and never restate it.
`.serena/project.yml` `ignored_paths` and the `read-guard.sh` volume budget: both mechanisms are
deregistered toolchain-wide.

**A `scripts/check_drop_ins.py` gate was weighed and declined**: the user announces every
refresh, and the clauses self-evidence against the tree — 7 shipped files name the archive path,
so a reverted `CLAUDE.md` contradicts them on sight. Rigor concentrates where no re-check exists.
