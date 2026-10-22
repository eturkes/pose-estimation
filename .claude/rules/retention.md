# Artifact retention — standing user rule

The pipeline's numerical output is destined for the `../rehab` dashboard over the hospital SCI
database.

- **An artifact serving that path is kept whatever the status of the work that produced it,
  DESCOPED included; one off it is deleted outright**, never tagged or archived — a superseded
  artifact costs a future agent a wrong start. The question is which of the two, never age.
- **A retained branch, tag or oracle needs one named open dependency**, decided on content, not
  label. **Prove no-dependency before deleting** by sweeping `.agent/ scripts/ tests/ src/ docs/`
  with a positive control — an empty search and a broken one print the same bytes.
- **A retained commit survives on an `archive/*` tag, never on its `wt/` branch.** The Close sweeps
  `wt/` branches wholesale, and `git gc` prunes unreachable commits past `gc.pruneExpire` (2 weeks
  default) → a row naming a
  `.scratch/worktrees/<name>` path resolves it to a commit and tags it the moment the row is
  written. Tag a live wave's tip as its unit closes; recover an untagged one by matching `git fsck
  --no-progress | grep 'dangling commit'` against subjects (tip = the dangling commit with no
  dangling descendant).
- **Tag classes** — census = `git tag --list 'archive/*'`, read it before trusting any retention
  claim:
  - oracles: `archive/m2u3-spike-audio` (`scripts/spike_audio_offsets.py` `_family_coverage`, the
    P38 oracle of R6's connectivity-reconciliation row) · `archive/m2u3-spike-visual`
    (`scripts/visual_sync_spike.py` `family_recovery_summary`, the only copy of the rule yielding
    26/137);
  - diff-blind + reviewer suites: `archive/m<m>u<u>-{test,review,rev}[-<n>]` +
    `archive/m2-review-<name>` = red-witness test bytes that commit bodies cite (→ `gates.md`) and
    adversarial-review dispatch inputs; `archive/m3u3-test` = the diff-blind cases of the five cut
    3D clinical groups on the `../rehab` path.
- A red diff-blind or reviewer suite is retained on its tag rather than in `tests/`, because one red
  file fails the decisive gate (→ `gates.md`).
- **Acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`** (ruling on `CLAUDE.md`
  `Engineering` "with its contract", which names no path). Files that break on a move =
  `rg -l 'archive/contract-' scripts/ src/ tests/` (a whole-tree sweep also counts documents that
  merely mention it). One is a generated data field: `scripts/make_calibration_qc_fixtures.py`
  writes the path into `tests/fixtures/calibration_qc_set/manifest.json`, which
  `check_calibration_qc_fixtures.py` validates without resolving → a rename missing the generator
  leaves a dangling pointer no gate reports.
- **`.agent/archive/` records are frozen, stale pointers included** (ruling on `CLAUDE.md`
  `Session flow` finished work → `.agent/archive/` + `Execution` every file = yours to edit):
  rewriting one falsifies it as evidence of what was ruled; the live surface carries the
  correction. Read an archive pointer as a citation of its own time (`.agent/memory.md`, a `Read()`
  deny list, `/session-*` commands and live-state self-descriptions are all retired). Live state =
  `.agent/spec.md` + `.agent/deferred.md` + `.claude/rules/`; reviving a frozen `polish.md` row =
  writing it into `.agent/deferred.md` with an acceptance check.
