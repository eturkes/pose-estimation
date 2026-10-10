---
paths:
  - "CLAUDE.md"
  - ".claude/rules/*.md"
  - ".agent/*.md"
  - ".agent/archive/*.md"
---

# Upstream instruction sync

`CLAUDE.md` = upstream `~/.local/app/agents/claude/CLAUDE.project.md` byte for byte; every local
adaptation lives in `.claude/rules/`, keyed on the template clause it overrides and indexed below;
Clauses 1-5 are the only surviving copy of the ones that bind a refresh.
**`last-sync = agents@4d22203`.**

**Template invariants — hold each after every refresh:**

- `cmp CLAUDE.md ~/.local/app/agents/claude/CLAUDE.project.md` = 0, line 1 = `@.agent/spec.md`.
- `.agent/spec.md` = `Intent` (user-edited alone) · `Artifacts` · `Decisions` · `Tasks` · `Phase`,
  in that order. `Artifacts` = path each + run command where it runs.
  `Tasks` = `- [ ]` open units in order, `- [x] <sha>` once committed, on-path finds appended,
  ticked rows cleared at phase close, last line = the `.agent/deferred.md` pointer; an open
  resume note = one `- [ ] RESUME: …` row at the head of its unit (`pause.md`; the statusline reads
  that form). `Phase` = phase + scope.
- Every phase = one session per pasted body (`prompts/{auto,steered}/<phase>.md`), ITERATE
  included, run until its `Met when`. Advisor on through IMPLEMENT + MAINTAIN.
- Deferral queue = `.agent/deferred.md`, unattached, one row + acceptance check each.
- A template structure (prototype location, CI, review ledger, spec layout) holds its default unless
  a ruling in the index adapts, retires or marks it inapplicable, naming the replacement or the
  user's waiver.
- Teammate triggers + mechanics = global `CLAUDE.md` `Subagents`; role rules =
  `~/.claude/agents/<role>.md`; a commit body names each teammate its unit used (name, role,
  verdict). The phase sets dispatch rate; closing diff → `reviewer`: one per lens in IMPLEMENT +
  MAINTAIN, one covering every lens elsewhere. Thinking depth = the launch `--effort`. No
  `.claude/settings*.json` env pin or `.claude/agents/` definition overrides the user-level models,
  effort or roles.

**Recipe, every refresh** = upstream's session body
`~/.local/app/agents/claude/prompts/{auto,steered}/refresh.md` (`last-sync` derivation + recording,
delta, old-form sweep, commit subject). This repo adds: re-apply every ruling in the index; keep
every upstream change no ruling contradicts. Verify Clause 1 with
`rg -l 'archive/contract-' scripts/ src/ tests/` = 9, re-derived rather than trusted — a
whole-tree sweep counts every document that merely mentions the path and drifts on every edit.

**A purely additive clause is not a no-op either**: contradicting nothing, it still binds
mechanisms this repo already runs its own way, so resolve every new clause against the local
mechanism before recording a refresh clean — `CLAUDE.md`'s `Verification integrity` bullet binds a
red-witness rule whose local form is a targeted run, because the decisive gate has to close green
(→ `gates.md`). **A cut is not a reversal either**: guidance upstream drops as redundant with the
current model (the UI/UX style + report-every-issue lines) stays expected
behaviour, so keep every local application of it standing. A retired mechanism is the other kind:
its local dependents re-derive against the upstream commit that retired it. That commit + the
user's refresh note decide which; ask when neither does.

**Rulings on template clauses — index.** Every structure absent here holds its template default:
prototype at `prototype/<name>/` (`prototype/review-ui/`), review ledger `.agent/review.md`, spec
layout as above. `Phase` scope = the user's ruling, recorded in `.agent/spec.md` `Phase`.

| Template clause | Ruling | Effect → replacement |
| --- | --- | --- |
| `Session flow` IMPLEMENT: scanning + update automation "in gate + CI" | `gates.md` *Hosted CI* (user) | adapts → full local gate replaces hosted CI; scanners + update automation owed (`.agent/deferred.md`) |
| `Engineering` verification integrity: red on the unfixed revision | `gates.md` red-witness bullet | adapts → targeted `P pytest tests/<file>` red on the unfixed tree, never a red commit; body names both refs |
| `Engineering` verification integrity: skipped case → row; `green` | `gates.md` `green` bullet | adapts → environment-completeness `skipif` = entry gate, no row; A32's zero-skip reconciliation carries `green` |
| `Engineering` deterministic checks ship their firing input | `gates.md` *Check firing evidence* | gap recorded → green-only checks + mutation campaigns owe firing inputs (`.agent/deferred.md`) |
| `Engineering` review termination | Clause 2 | adapts grain → a pass terminates on rows adjudicated, a wave closes on fixes applied |
| `Engineering` assurance tier "with its contract" | Clause 1 | fixes the path → `.agent/archive/contract-m<m>u<u>.md` |
| `Session flow` finished work → `.agent/archive/` | Clause 4 + `retention.md` | adapts → archive records frozen, stale pointers kept |
| `Authoring` durable-guidance routing | Clause 3 | adapts → mutable state stays in `spec.md` + `deferred.md`, never in rules |
| `Session flow` Teammates + Advisor, `Execution` research, `Engineering` `data` tier: depth keyed on phase | Clause 5 (user) | adapts scope → spine rows run MAINTAIN law whatever phase the review UI holds |
| `Session flow` PROTOTYPE + ITERATE: finalist screenshot, `operator` visual QA | `data-boundary.md` blanked-stage bullet | adapts → player view captured as the blanked stage alone (`.scratch/player_shot.mjs`); census + cohort by view fragment |

- **Clause 1 — acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`**; the template
  names no contract path. **9 files under `scripts/ src/ tests/` break if it moves**, one of them a generated data field: `scripts/make_calibration_qc_fixtures.py` writes
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
  open rows, counts, `last-sync` excepted — stays in those two; `.claude/rules/` takes only what
  holds until new evidence reverses it, and points at the queue rather than restating a row.
- **Clause 4 — `.agent/archive/` is this repo's detail store and its records are frozen.**
  `roadmap.md`, `polish.md`, `review-m2.md`, the 14 unit contracts and the reviewer reports live
  there and keep their own stale pointers (`/session-roadmap`, `.agent/memory.md`, a `Read()`
  deny list). Read an archive pointer as a citation of its own time. The live surfaces are
  `.agent/spec.md`, `.agent/deferred.md` and `.claude/rules/`.
- **Clause 5 — spine rows = MAINTAIN requests on the shipped spine, whatever phase the review UI
  holds** (user ruling). The template keys teammate depth, research count, the `data`-tier
  `reviewer` and the advisor on phase, while `Phase` scopes the review UI alone. A spine row in
  the same `Tasks` therefore runs IMPLEMENT + MAINTAIN law: `researcher` at any source count,
  `consultant` per kernel contract, `reviewer` + `tester` per kernel unit, `reviewer` per `data`
  unit, `scientist` per shipped analysis result (after-tables, cohort), advisor on. Review-UI rows
  follow the phase `Phase` names.

**Superseded by upstream — never restore.** The `Read()` path-exclusion control (→
`data-boundary.md`). The `N% NK/1M` gauge convention: MAIN and a teammate read against different
windows (→ `evidence.md`), so no literal belongs in it. `.agent/roadmap.md` and `.agent/polish.md`
as attached state, and the `/session-roadmap`, `/session-prompt`, `/session-polish` commands that
consumed them — `.agent/spec.md` is the sole attached state and the phase flow replaces the MODE
dispatch. The **`≤ 8 KB` cap on `.agent/spec.md`**: liveness replaces it — every line binds current
or future work, superseded text dies in the commit that supersedes it, size is emergent. A byte
budget rewards compressing live prose over deleting dead rows, which is the opposite of the ranking.
**The deferral queue as a `spec.md` section**: a queue is monotonic, so attached it is a permanent
growth term; it lives at `.agent/deferred.md`, which every queue write names by path, while
`spec.md` `Tasks` carries the unfinished units + the queue pointer. **Teammate law as project
scope**: triggers + mechanics live in global `CLAUDE.md` `Subagents` and role rules in
`~/.claude/agents/<role>.md`, so `.claude/rules/delegation.md` points at them and never restates
them. `.serena/project.yml` `ignored_paths` and the `read-guard.sh` volume budget: both mechanisms
are deregistered toolchain-wide.

**A `scripts/check_drop_ins.py` gate was weighed and declined**: the user announces every
refresh, and the clauses self-evidence against the tree — every Clause 1 file names the archive path,
so a reverted `CLAUDE.md` contradicts them on sight. Rigor concentrates where no re-check exists.
