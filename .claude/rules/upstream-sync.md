---
paths:
  - "CLAUDE.md"
  - ".claude/rules/*.md"
  - ".agent/*.md"
  - ".agent/archive/*.md"
---

# Upstream instruction sync

`CLAUDE.md` = upstream `~/.local/app/agents/claude/CLAUDE.project.md` byte for byte, `.agent/spec-lint`
= upstream `claude/spec-lint` byte for byte; every local adaptation lives in `.claude/rules/`, keyed
on the template clause it overrides and indexed below; Clauses 1-5 are the only surviving copy of
the ones that bind a refresh.
**`last-sync = agents@8bc2c15`.**

**Template invariants — hold each after every refresh:**

- `cmp CLAUDE.md ~/.local/app/agents/claude/CLAUDE.project.md` = 0, line 1 = `@.agent/spec.md`;
  `cmp .agent/spec-lint ~/.local/app/agents/claude/spec-lint` = 0.
- `.agent/spec.md` = `Intent` (human-facing, outranks the rest) · `Artifacts` · `Decisions` ·
  `Tasks` · `Phase`, in that order; `.agent/spec-lint` grades the shape inside the decisive gate
  (→ `spec-lint.md`). `Artifacts` = path each + run command where it runs.
  `Tasks` = `- [ ]` open units in order, `- [x] <sha>` once committed, on-path finds appended,
  ticked rows cleared at each phase close + each MAINTAIN request's close, last line = the
  `.agent/deferred.md` pointer; an open resume note = one `- [ ] RESUME: …` row at the head of its
  unit (`pause.md`, `pause-now.md`; the statusline reads that form). `Phase` = `<PHASE> — <scope>`
  on line 1, then routed state alone (`>` request lines, `Suspended:` lines); history →
  `.agent/archive/`.
- Every phase = one session per pasted body (`prompts/{auto,steered}/<phase>.md`), ITERATE
  included, run until its `Met when`; a request needing a session or more of another phase →
  `prompts/auto/route.md`.
- Deferral queue = `.agent/deferred.md`, unattached, one entry + acceptance check each.
- A template structure (prototype location, CI, review ledger, spec layout) holds its default unless
  a ruling in the index adapts, retires or marks it inapplicable, naming the replacement or the
  user's waiver.
- Teammate fits, their reasons + mechanics = global `CLAUDE.md` `Subagents`; role rules =
  `~/.claude/agents/<role>.md`; each dispatch + advisor call = MAIN's per-case call over the fits,
  a user request binding; a commit body names each teammate its unit used (name, role, verdict).
  The phase sets teammate cost + its own phase fits; the dispatch bar = the phase's in PROTOTYPE +
  ITERATE, each unit's depth in IMPLEMENT + MAINTAIN, and the advisor bar follows it. Closing-diff
  `reviewer` → one per lens in IMPLEMENT, one covering every lens elsewhere, MAINTAIN's where
  needed. Advisor tool absent where a call would pay → ask the user for
  `/advisor claude-fable-5-1`. Thinking depth = the launch `--effort`. No `.claude/settings*.json`
  env pin or `.claude/agents/` definition overrides the user-level models, effort or roles.
- Collaboration, Authoring, Engineering, research, briefs, git + read-cost law = global
  `CLAUDE.md`; a ruling on one of those clauses keys on its global section.

**Recipe, every refresh** = upstream's session body
`~/.local/app/agents/claude/prompts/{auto,steered}/refresh.md` (`last-sync` derivation + recording,
delta, old-form sweep, commit subject). This repo adds: re-apply every ruling in the index; keep
every upstream change no ruling contradicts. Verify Clause 1 with
`rg -l 'archive/contract-' scripts/ src/ tests/` = 9, re-derived rather than trusted — a
whole-tree sweep counts every document that merely mentions the path and drifts on every edit.

**A purely additive clause is not a no-op either**: contradicting nothing, it still binds
mechanisms this repo already runs its own way, so resolve every new clause against the local
mechanism before recording a refresh clean — global `CLAUDE.md` `Engineering`'s verification-integrity
bullet binds a red-witness rule whose local form is a targeted run, because the decisive gate has to close green
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
| global `Engineering` verification integrity: red on the unfixed revision | `gates.md` red-witness bullet | adapts → targeted `P pytest tests/<file>` red on the unfixed tree, never a red commit; body names both refs |
| global `Engineering` verification integrity: skipped case → row; `green` | `gates.md` `green` bullet | adapts → environment-completeness `skipif` = entry gate, no row; A32's zero-skip reconciliation carries `green` |
| global `Engineering` deterministic checks ship their firing input | `checks.md` | gap recorded → green-only checks + mutation campaigns owe firing inputs (`.agent/deferred.md`) |
| global `Engineering` review termination | Clause 2 | adapts grain → a pass terminates on rows adjudicated, a wave closes on fixes applied |
| global `Engineering` assurance tier "with its contract" | Clause 1 | fixes the path → `.agent/archive/contract-m<m>u<u>.md` |
| `Session flow` finished work → `.agent/archive/`; `Execution` every file = yours to edit | Clause 4 + `retention.md` | adapts → archive records frozen, stale pointers kept |
| global `Authoring` durable-guidance routing | Clause 3 | adapts → mutable state stays in `spec.md` + `deferred.md`, never in rules |
| `Session flow` Teammates + Advisor: cost, dispatch bar, advisor bar + phase fits keyed on phase | Clause 5 (user) | adapts scope → spine rows run MAINTAIN law whatever phase the review UI holds |
| `Session flow` PROTOTYPE + ITERATE: finalist screenshot + `operator` QA fit; IMPLEMENT `operator` QA fit (+ global `Subagents` `operator` fit) | `data-boundary.md` blanked-stage bullet | adapts → player view captured as the blanked stage alone (`.scratch/player_shot.mjs`); census + cohort by view fragment |

- **Clause 1 — acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`**; the template
  names no contract path. **9 files under `scripts/ src/ tests/` break if it moves**, one of them a generated data field: `scripts/make_calibration_qc_fixtures.py` writes
  the path into `tests/fixtures/calibration_qc_set/manifest.json`, and
  `check_calibration_qc_fixtures.py` validates digests without resolving that field, so a rename
  missing the generator leaves a dangling pointer no gate reports.
- **Clause 2 — the grain splits: a review PASS terminates on rows adjudicated, a review WAVE
  closes on fixes applied.** Global `CLAUDE.md` `Engineering`'s termination rule quantifies over the
  pass and is kept whole — check set fixed before the diff is read, every row adjudicated, an all-`pass` table
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
  `.agent/spec.md`, `.agent/deferred.md` and `.claude/rules/`. The template's `every file there =
  yours to edit` grants the edit; the freeze is this repo's ruling on what an edit would falsify.
- **Clause 5 — spine rows = MAINTAIN requests on the shipped spine, whatever phase the review UI
  holds** (user ruling). The template keys teammate cost, the dispatch bar and the advisor bar on
  phase, while `Phase` scopes the review UI alone. A spine row in the same `Tasks` therefore runs
  MAINTAIN law, depth per unit: a unit over pipeline, publisher or analysis code is `kernel`
  (`.agent/spec.md` `Decisions`), so deep — dispatch wherever a teammate adds assurance MAIN's own
  pass would miss (a close call → dispatch), `consultant` on its contract before code, the advisor
  at the low bar, the closing-diff `reviewer` where needed (one covering every lens). Review-UI
  rows follow the phase `Phase` names.
  Global fits (`tester` on a committed kernel contract, `reviewer` on a kernel-unit diff,
  `scientist` on an after-table or a cohort republication) hold in every phase, scope-free.

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
scope**: fits + mechanics live in global `CLAUDE.md` `Subagents` and role rules in
`~/.claude/agents/<role>.md`, so `.claude/rules/delegation.md` points at them and never restates
them. **Per-phase teammate + advisor mandates** — every-phase `consultant` on the phase plan, a
`researcher` at any source count, `reviewer` + `tester` per kernel unit, a `scientist` reproducing
each result, the advisor on through IMPLEMENT + MAINTAIN: each call is MAIN's per-case judgment over
the fits, so Clause 5 rules scope, never a dispatch list. `.serena/project.yml` `ignored_paths`
and the `read-guard.sh` volume budget: both mechanisms are deregistered toolchain-wide. **`Intent`
edited by the user alone**: every file in the repo is the agent's to edit, `Intent` kept current
in the human-facing register. **Closing-diff `reviewer` per lens in MAINTAIN**: one covering every
lens, where needed. **Firing + scratch-validator records beside the bare gate invocation**: each
lives in a `paths:` rule on the files it grades, so the bare tier carries invocation law alone.
**The ≤20/≤25-word sentence caps**: human-facing text takes short sentences, and a sentence-length
flag is a lead for judgment, never a gate.

**A `scripts/check_drop_ins.py` gate was weighed and declined**: the user announces every
refresh, and the clauses self-evidence against the tree — every Clause 1 file names the archive path,
so a reverted `CLAUDE.md` contradicts them on sight. Rigor concentrates where no re-check exists.
