# Delegation — traps measured in this project

Teammate fits + mechanics = `CLAUDE.md` `Teammates`; each role's standing rules =
`.claude/agents/<role>.md`. These are the project's own rulings + measured corrections.

## Scope + review grain

- **Phase scopes the spine; review-UI rows run PROTOTYPE law** (user ruling: the review UI is the
  analysis's review instrument). Spine work (pipeline, publisher, analysis = `kernel`, `.agent/spec.md`
  `Decisions`) follows the phase `Phase` names, deep. A `tools/review-ui/` row is instrument upkeep
  under PROTOTYPE law, outside the gates, in any phase. Role fits (`tester` on a committed kernel
  contract, `reviewer` on a kernel-unit diff, `scientist` on an after-table or a cohort
  republication) hold in every phase.
- **A review PASS terminates on rows adjudicated; a review WAVE closes on fixes applied** (ruling on
  `CLAUDE.md` `Engineering` review termination, kept whole for the pass: check set fixed before the
  diff is read, every row adjudicated, an all-`pass` table complete, no open-ended re-review). An
  adjudicated row accepting a fix stays open until that fix closes on its own acceptance check
  under MAIN's rerun → a wave that rules every row and applies none re-enters to apply them (M2's
  first wave: 193 rows / 55 fails in one pass, three more sessions to close 40 of 43 fixes).
  Reading this as an override rather than a grain split drops the anti-flip-flop mechanism.

## Worktree teammates

- **Poll a worktree teammate's report inside its worktree**:
  `.scratch/worktrees/<name>/.scratch/agents/<name>.md`. MAIN's `.scratch/agents/<name>.md` keeps
  the unfilled seed forever, so a counter reading it reports `0/N` against an agent on pace. Copy
  the worktree file into MAIN's `.scratch/agents/` before removing the worktree (`.scratch/` is
  gitignored → teardown deletes it).
- **A worktree branch tip moves after the poll that read it** → a deletion's `was <sha>` names a
  commit the poll never saw; check ancestry (`git log --oneline -3 <sha>`) before concluding work
  was lost.
- **Keep a rejected design's worktree after picking the winner**: an independent implementation
  sharing no line with the winner is a free differential oracle (a rejected 184-line generator
  matched the shipped partition over 193 blocks + 379 assets), unreproducible once deleted.

## Rulings + reviewer output

- **Amend a superseded ruling in place, in the same turn as the code it rules.** A "later rulings
  win" note does not hold: the table is what gets read, and a `tester` conforming to a stale row
  produced a suite that read as a regression against correct code.
- **A reviewer's worktree diff against `main` carries its own staleness** (it can show a revert of
  a MAIN change landed after its branch point) → read it as findings, never as a patch.
- **A saturated reviewer's persistence is a context artifact, not evidence.** When the same row
  returns from an agent near its ceiling (> ~97 %), restate the ruling once and stop; prefer a
  fresh successor. Weigh a finding by its argument; the discriminator is the reviewer's gauge, not
  its insistence.
- **A reviewer's DONE marker is not the end of its yield** (nine accepted defects once arrived
  after both markers) → while it has context, point it at surfaces MAIN just changed and take
  findings by message; ask for the report file last.
- **Copy a reviewer's red suite into the primary tree and rerun it; the pass/fail split is the
  adjudication ledger.** Reds against an older branch point separate into fixed / open-scope / live
  only under execution, and the rerun catches half-fixes a report's tally cannot.
- **A restored red is evidence, not law**: 3 of 27 restored reds were defective (one contradicted
  its own ledger row with the test right) → read each against the shipped contract and its own
  ledger row before crediting it.
- **Reds from one reviewer are no more mutually consistent than reds from two** → restore a
  battery whole and run it as a set; a per-row restore hides a contradiction as a fix that "did
  not take".

## Review-wave sizing + grader vocabulary

- **5 reviewers × ~40 rows (193 rows / 55 fails) fit one MAIN window** because harvest is
  mechanical: archive the report, script-extract the verdict table, read `fail` rows alone (~25K
  tokens for 193 rows). Seeding + both-ways grading = `CLAUDE.md` `Teammates`.
- **Write the brief in the grader's own vocabulary** (`scripts/check_review_report.py` →
  `checks.md`): rows `| <ID> |` with `<ID>` = `[A-Z]\d{2}`; literal headings `## Verdict table`,
  `## Detail`, `## Register`; verdicts `pass` | `fail` | `unknown` — map the role's `finding`
  (ruling on `.claude/agents/reviewer.md` Deliverable) to `fail`, since a `finding` row grades
  `FAIL P02c`.
