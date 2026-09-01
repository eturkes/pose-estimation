# Roadmap

Live long-horizon state only; completed trajectory belongs in git. Closed-unit detail, frozen contracts → `.agent/archive/`, read on demand.

**Repo scope = `videos/3-cam/`.** Sibling directories under the same data root are out of scope: `harness/` holds schematics for a capture harness that was never built, `database/` holds the hospital's SCI clinical records. `videos/initial/` is preliminary data, retired from active work.

## M2 — three-camera corpus: inventory, qualification, 3D ruling

**Status: IN-PROGRESS** — M2.1-M2.5 DONE; M2.6 and M2.6b both closed negative and ruled. **M2.7 and M2.8 are PLANNED into 7 units** — M2.7.1 DONE; M2.7.2-M2.7.4 and M2.8.1-M2.8.3 OPEN, none BLOCKED, lowest OPEN unit = M2.7.2. M2.7 publishes the negative through a new `calibration_qc/` publisher (F1a); **M2.8 publishes cohort aggregates only, by user ruling** — no per-subject rows, no patient identifier, no join column, no join to `../rehab`, and it stops at this repo's boundary. The 3D line is closed; it reopens only on prospective calibrated capture. The old clearance precondition is met: full decode clearance covers the whole `videos/3-cam/` tree, for MAIN and teammates. Chat and reports carry redacted aggregates only — never imagery, filenames, or subject identifiers.

**Goal:** turn 382 uncontrolled clips into an addressable, measured corpus; establish by evidence whether 3D reconstruction is recoverable from it; then execute that ruling under a claim boundary the data can carry.

**Corpus, measured.** Header-only census over all 382 files, from the committed tool (`pose-estimation-inventory`; cv2 container properties, no pixel decode; ~11 s including a SHA-256 of every file). Every number below reruns from `inventory/census.json`:

- 382 files, 339 743 frames, 186.8 min, 18.28 GB, 16 subject directories, all readable, all 382 SHA-256 distinct. Per-file nominal duration median 19.3 s (p25 13.9, p75 33.1, p95 62.7, max 274.8).
- Stems `<n>_<view>_<task>_<side>` → **188 task-side families** over 379 canonical files. `<n>` is the **subject** ordinal — 16 values, one per subject directory, bijective — so family identity is `(subject, task, side)` with no take component. View coverage **52 three-view / 85 two-view / 51 one-view**, so 137 families are multi-view. **3 files quarantine** (2 empty side token, 1 unknown side token), 0 excluded. A plan assuming 188 three-camera sessions overstates by 3.6×.
- **The intended design is 16 subjects × 6 tasks × 2 sides = 192 families; 188 exist, so 4 are absent outright** — `s02-cap-r`, `s06-glass-r`, `s07-coin-l`, `s12-glass-l`. Family view-sets: `above|left|right` 52, `above|right` 46, `above` 37, `left|right` 20, `above|left` 19, `right` 13, `left` 1. Canonical assets per view: above 155, left 93, right 131 — **`left` is the camera most often missing**, and every one of the 16 subjects owns at least one three-view family.
- **The 49/87/52 versus 52/85/51 disagreement is settled: doubled media suffixes.** 4 files are named `x.mov.MOV`; stripping one suffix leaves `.mov` occupying the side slot. Typo and repeat-marker handling moved 0 of 188 families. Both earlier splits are dead.
- Name repair is small and now measured: 379 case folds, 15 task-token repairs over 4 misspellings, 4 doubled suffixes, 2 whitespace collapses, 1 repeat marker.
- Resolution 1920×1080, or 1080×1920 for the 28 portrait clips, plus one 1280×720 outlier. **38 clips carry non-zero rotation metadata** — 28 at 90/270°, 10 at 180°. Codecs h264 + hevc, and **all 16 subject directories mix both**.
- **Nominal 30 fps, but every file differs**: 29.963-29.987 Hz, plus one 119.97 fps 720p outlier. Within-family fps agreement to 3 dp: **7 of 137** multi-view families.
- **The views of one family are not one recording.** Frame-count parity within 5%: 40 of 137; within 20%: 74 of 137. Equal resolution across views: 122 of 137. Within-family duration spread median **3.92 s**, p75 13.0 s, p95 25.7 s, max 210.2 s.
- **Orientation varies inside a single view label**: `above` = 129 rot0 / 15 rot90 / 10 rot180 / 1 rot270; `left` = 88/5; `right` = 124/7.
- **2 families carry a view conflict** — two files claiming one view of one family. Both pairs are distinct recordings rather than copies: one is repeat-marked, one merges under underscore collapse. The registry flags them; it does not choose between them.
- **Header facts are the demuxer's claims.** A 10-clip full decode spanning both codecs and all four rotation values matched the header frame count exactly, 7 061 of 7 061 frames, so counts are trustworthy on that sample. The same decode found inter-frame presentation timestamps non-uniform on 10 of 10 clips, so cv2 alone supports no constant-frame-rate claim.

**No fixed rig ever existed** — the camera harness was designed and never built, which is what the orientation, codec, parity and duration spread independently measure. Three cameras were started and stopped by hand and re-oriented between takes. Two consequences bind the whole milestone: a single rig calibration reused across the corpus is incoherent, so calibration is **per recording event** at best — a grain M2.2 still has to resolve, since a `view_conflict` family holds more than one take; and the repo's alignment model, one non-negative integer `sync_offset` per camera, cannot express unequal frame rates, so cross-view drift is unrepresentable today.

**Claim boundary, set by evidence, not by ambition.** Published upper-limb validation puts three RGB cameras at 5-9 cm joint-position error with no clinical angle validation; the credible functional-task protocols use 8-10 synchronized cameras at 60-85 Hz and still report 3-20°+ angle RMSD. One frame at 30 Hz is 33 ms, which is 33 mm of hand travel at 1 m/s, so even exact integer alignment caps timing contribution near ±17 mm. **M2 may claim retrospective 3D recovery feasibility with internal geometric and QC evidence — reprojection, triangulation angle, visibility, offset confidence, scale provenance, sensitivity. It may not claim clinical validity, absolute metric accuracy, or marker-based equivalence.** Crossing that floor needs a prospective calibrated capture, which M2.7 specifies rather than performs.

**Metric scale is unavailable in this corpus — surveyed and ruled (R3, closed negative).** Image geometry has a gauge freedom, so SfM, essential matrices and pose-only calibration recover shape up to one unknown factor. The survey ran over a stratified 52/379 asset sample spanning all 18 task×view cells: the task apparatus is widely visible, yet exact dimensional identity resolved in **0/52**, every fallback (anthropometrics, furniture, rig baseline, calibration target, audio time-of-flight, skeletal priors) is absent rather than imprecise, and the best conditional route floors at **±17.7%** before lens distortion. 3D output is therefore permanently arbitrary-scale for this corpus: angles, angular velocities, timing, normalized trajectory shape and dimensionless ratios survive; every metre-valued distance, velocity and jerk does not. The negative is *sampled*, so the axis publishes `scale_unmeasured` on all 379 rows rather than a measured `none`. Reopening needs acquisition outside this corpus — caliper measurement of the retained apparatus, a filmed ruler, or mapped participant anthropometry — and is a user decision, not a schedulable analysis unit.

**Units.** Tier `kernel` throughout; each closes with a scoped commit and its own primary-tree gate run.

| id | unit | spine result |
| -- | ---- | ------------ |
| M2.1 | Canonical capture record + corpus census | Stem grammar, normalization, quarantine, deterministic `capture_id`; per-file container facts; one committed inventory tool replacing the two scratch censuses. |
| M2.2 | Session materialization + discovery | Idempotent generator emitting a discoverable session tree; partial-view policy; `--list-sessions` enumerates the real corpus. |
| M2.3 | Capture qualification + 3D-route ruling | Decode-sampled evidence on scale reference, background rigidity, view↔geometry stability, detectability, recoverable offset/drift, intrinsics metadata → MAIN's ruling, which shapes M2.5-M2.7. |
| M2.4 | Timebase truth | Adopt `nominal_fs()` at the call sites; regenerate goldens; per-file cadence replaces the `1/median(diff(ts))` estimate. |
| M2.5 | Cross-view alignment | One float `offset_s` per camera against the event reference, no rate term (M2.3 R5); per-recording-event `sync_qc` evidence. |
| M2.6 | Calibration recovery | **Planned result unreachable — F0 closed negative.** Extrinsics by BA over time-synchronized 2D keypoints was the route (scene-feature SfM already eliminated by measurement); the corpus carries 15-20 px systematic cross-view keypoint bias, which the shipped estimator and independent bundle adjustment both fail to repair, and which no disjointly selected RTMW-L subset beats all 65 keypoints on. Shipped instead: the measurement closing it, `scripts/probe_calibration_bias.py`. **Ruled: publish the negative through F1a**, executed in M2.7. |
| M2.6b | Bias-modeling repair | **User-funded, closed negative at its gate (G0).** The one repair route A14 left open — a joint bias-and-pose parameterization under an identifiability constraint. Its premise is a bias with fewer degrees of freedom than the data, which needs the bias to be shared across events. Measured absent at every grouping. Shipped: `scripts/probe_bias_transfer.py`. |
| M2.7.1 | `calibration_qc/` publisher (F1a) | Corpus-level negative ruling published beside `qualification/`, never patching it. One corpus row + evidence pointers to the two committed probes. |
| M2.7.2 | De-identified regression fixtures | Idempotently generated fixtures for one valid generation plus the contract/integrity failure matrix; a privacy scan proves no corpus identifier reaches them. |
| M2.7.3 | Claim-bounded negative report | Human-facing report stating exactly what the two probes decide, with a claim matrix mapping every conclusion to evidence and to permitted/prohibited wording. |
| M2.7.4 | Prospective capture specification | The future calibrated acquisition that can reopen 3D. Specified, never run. |
| M2.8.1 | Corpus-run preconditions + instrumented pilot | Two measured blockers fixed, CFR fallback rate instrumented, stratified pilot green — before the 6.5 h run is funded. |
| M2.8.2 | Full corpus 2D run | ~6.5 h resumable run over 379 assets; per-asset clinical features plus a manifest giving every asset an explicit disposition. |
| M2.8.3 | Cohort aggregate publisher + bilingual descriptor | 12 `(task, side)` cohort rows, no subject rows; append-ready `columns.yaml` fragment with full `ja`/`en` labels. |

**Unit status.** M2.1, M2.2 and **M2.3 DONE** — M2.3 across ten windows, closing on P29. **M2.4 DONE** and **M2.5 DONE** — see below. **M2.6 closed negative at F0 and ruled** — extrinsic recovery is measured unachievable on this corpus, and the verdict ships through F1a, see below. **M2.6b closed negative at G0** — the funded repair route is refused by measurement, see below. **M2.7.1 DONE** across four windows — the `calibration_qc/` publisher, both campaigns and its documentation ship. **M2.7.2-M2.7.4 and M2.8.1-M2.8.3 OPEN**: the 3D spine is gone, the negative is published, and the 2D line delivers cohort aggregates.

### M2.7 — publish the closed-negative 3D ruling — IN-PROGRESS, 4 units, M2.7.1 DONE

**F1a ships as the FIFTH copy of the publisher idiom, and the extraction is deliberately declined.**
`src/pose_estimation/publication.py` (polish, pri 3/M) anticipated exactly this trigger, and M2.7 is
the trigger. It is still refused: extraction touches three shipped publishers whose mutation gates
are pinned to them, its own acceptance bar demands all three published trees stay byte-identical, and
it is blocked behind the pri-1 row recording that `scripts/run_inventory_mutations.py` cannot replay
from committed state (recorded target `inventory.py` `96a6d0…` against committed `66d3ef…`).
Refactoring three working publishers is not the shortest path to publishing a negative. The polish row
stands; this ruling is why, so it is not re-litigated each window.

**Sizing against M2.5, the nearest publisher actual (22 files / 4761 insertions / 4 windows).** F1a is
smaller than that analog for one structural reason: **M2.5's four windows bought a 379-row solved
artifact with a least-squares solver, a reference-selection rule, and a census that had to be
recounted; F1a publishes ONE corpus-level row plus evidence pointers, and owns no estimator.** The
publisher machinery — ownership, atomic swap, digests, alphabets, consumer boundary — is the whole
cost, and it is a known idiom with four worked examples. `map-m2` extracted that contract as a
seven-point checklist with per-publisher anchors and found **no safe literal common denominator**:
`qualify` is the whole-tree oracle, `measure` the marker-parsing oracle, and copying `inventory` or
`sessions` inherits known trust-root gaps. Build from `qualify`.

**Hard carrier constraints, measured this window and non-negotiable.** `events_qc.geom_qualified`,
`events_qc.qualified` and `assets_qc.scale_ref_class` are **blank by design**; the sentinels live in
`events_qc.reason` (`geom_unmeasured` on 193/193) and `assets_qc.qc_flags` (`scale_unmeasured` on
379/379). So F1a **publishes beside `qualification/` and never patches it**, `session.json` stays
unpatchable, `sync_offset` is never written, and no per-event geometry verdict is emitted.

| id | tier | spine result | sizing |
| -- | ---- | ------------ | ------ |
| M2.7.1 | kernel | `calibration_qc/` publisher: contract + validator, canonical tree, integrity + consumer boundary, ownership + atomic publication. Accepts one corpus row and the probe-evidence rows; rejects every event/asset/capture/subject key. | **DONE**; 3 windows estimated, **4 spent** — contract + publisher (1-2), adversarial wave (3), rulings + campaigns + docs (4) |
| M2.7.2 | data | De-identified regression fixtures: one valid generation + the smallest contract/integrity failure matrix, each negative failing for its named predicate. Privacy scan proves no corpus filename, path, subject token or row-level statistic. | **1 spent, contract + validator + seed shipped; 1 more for fixture production** |
| M2.7.3 | docs | Claim-bounded negative report. Claim matrix maps every conclusion to evidence and to permitted/prohibited wording. Excludes clinical validity, absolute metric accuracy, marker equivalence, other-detector impossibility, and prospective-capture impossibility. | 1 window |
| M2.7.4 | docs | Prospective capture specification, 20 normative sections. Five non-negotiables: intrinsic/extrinsic calibration, synchronization residuals, orientation/drift control, traceable metric scale, identifiable-video governance. Specified, never run. | 1-2 windows |

**Order.** M2.7.1 gates M2.7.2 and M2.7.3 (the report releases only against a validated generation).
M2.7.4 is independent of all three and parallelizable from the start.

#### M2.7.2 — OPEN, window 1 spent on the contract, the validator and the seed

**The unit's own contract is frozen at `.agent/archive/contract-m2u72.md`** — 7 design decisions,
12 predicates P01-P12, a 30-row probe-corpus seed. Tier `data`, so assurance = one structural
validator + a live spot-check at close, defects fix forward.

**What M2.7.2 buys that M2.7.1's 104-case suite cannot, and this is the whole justification.**
Those cases build their inputs in Python under `tmp_path` and destroy them at teardown, and every
one asserts a *property* — row count, header order, idempotence, refusal class. **Nothing in the
tree pins the publisher's published BYTES.** A change to `_cell()`'s number formatting, to
`_canonical`'s sort key or to `inventory.render_json`'s separators passes all 104 and silently moves
every published byte. The committed generation catches it as a diff. Two further gains: refusal
identity becomes data rather than code (M2.7.3 cites the matrix instead of restating it, and a
renamed reason code fails by name), and the privacy predicate finally covers what this repo
*commits* — `_assert_cells_carry_no_identifier` guards run-time output only.

**Shipped this window.** `.agent/archive/contract-m2u72.md`;
`scripts/check_calibration_qc_fixtures.py` (MAIN-authored validator, 12 predicates, single
implementation behind both the CLI and the suite); the deliverable-first seed
`tests/fixtures/calibration_qc_set/{manifest.json,README.md}` whose `matrix` names all 30 reason
codes at `unknown`. **Graded both ways at seed**: absent tree → rc=1, seed present → **2/12 green,
rc=1**. Static gate `ruff check` / `ruff format --check` / `ty check` all **rc=0**; collection
**1438, unmoved** from the `02c4239` baseline.

**Deliberately NOT shipped: `tests/test_calibration_qc_fixtures.py` (P10).** A suite case asserting
`rc == 0` would be red until the fixtures land, and the alternative — a skip guard — breaks the
zero-skip invariant C8.08 reconciles. **P10 stays FAIL by design; that is what a deliverable-first
seed looks like.** The test lands in the same commit as the fixtures.

**The 30 reason codes are the failure matrix's row set**, derived from `src/` rather than
transcribed, and re-derived by P06 at check time so a new reason fails the matrix instead of being
omitted. MAIN's expected split, to be corrected by measurement: 18 run-path, 9 validate-path,
4 candidate `not_file_only` (`claim_missing`, `claim_prohibited`, `corpus_cardinality` each need a
monkeypatched module constant, since A02/A05 put the ruling and the claim set beyond any argument;
`tree_unreadable` needs a permission state git does not carry).

**One blocker found in MAIN's own contract before dispatch, and it would have eaten a whole
deliverable silently.** D07 first placed the fixtures at `tests/fixtures/calibration_qc/`.
Measured with `git check-ignore -v`: that path is ignored by `.gitignore:94`, and the corrected
root's inner `inputs/qualification/` is ignored by `.gitignore:85`. **Four `.gitignore` rows are
slash-free component names — `inventory:60`, `sessions:78`, `qualification:85`,
`calibration_qc:94` — so they match at ANY depth**, and `git add` reports success while committing
nothing. Layout amended: root `tests/fixtures/calibration_qc_set/`, upstream tree `inputs/upstream/`,
golden `expected/published/`. All eight paths verified NOT ignored.

**A second trap, one level down from an M2.7.1 ruling.** The first privacy scan failed on the
fixture's own `README.md`, because the README quoted the path text the scan forbids. Same shape as
A05's ruling that `PROHIBITED_PARAPHRASES` stays module-side and unpublished — **a document inside
the scanned set carries every string it quotes.** Ruled the same way: the needles live in the
checker, the README describes the rule without spelling them, and the scan stays total over every
committed byte rather than buying its own exemption.

**Next window's whole job.** Wave = one `prod-m2u72` teammate behind the committed validator:
write `scripts/make_calibration_qc_fixtures.py` (idempotent, default destination pinned), generate
`inputs/` + `expected/published/` + one `negatives/<reason>/` overlay per file-only reason, fill
`manifest.json`, and drive `python scripts/check_calibration_qc_fixtures.py` to rc=0. MAIN then
adds `tests/test_calibration_qc_fixtures.py`, spot-checks the golden and one overlay per path, runs
the decisive suite detached, and commits. **The minimum valid input set is the load-bearing
unknown** — `tests/test_calibration_qc.py:90-109` builds it by running `qualify.run()` over
synthetic media, but `run()` takes `sessions_dir`/`inventory_dir` as optional, so the fixture
likely needs the published qualification tree alone. `map-m2u72` was stopped at 1/6 sections with
S1 (the constructor graph and that minimum set) partly filled →
`.scratch/agents/map-m2u72.md`; it is browse context, not evidence.

`main=` **77% 185K/240K** — the window bought a frozen contract, a 12-predicate validator and a
graded seed, and no fixture bytes. `mate=` 32% 77K/240K (`map-m2u72`, stopped).

**Sizing datum.** Entry cost alone was **53% 127K/240K** before the first tool call: `.agent/`
attached state is now large enough that a WORK-UNIT window starts past half its budget. That is the
one-window aim's real constraint on this milestone, not the unit's line count. **M2.7.1's closed
detail belongs in `.agent/archive/` at the next MILESTONE-REVIEW**, and until it moves, size every
remaining M2 unit against ~110K of usable window rather than 240K.

#### M2.7.1 — DONE in window 4: both campaigns re-derived, 45 review rows ruled, docs registered

**The review closes 45 rows — 38 accepted, 7 rejected with recorded rulings.** Both reviewer red
files were copied into the primary tree and **rerun** rather than trusted from their reports, which
was decisive: several window-3 "already fixed" rulings were only half-fixed (C01/C11 fixed the module
docstring and left the roadmap headline). Score before fixes **41 failed / 4 passed**, after
**7 failed / 38 passed**, and the 7 are exactly the rejected set — **N10** (restoring the sibling to
`out` is stronger than survival at the retiring path), **N40-extra and N51** (*the arm set is open by
design*: the evidence table transcribes what the probe emitted, and the real capture carries 21 arms
against 5 required), **N57** (`calibration_qc/` ruled not deny-listed; the `Read(qualification/*.csv)`
form stands), and the three encoding-only rows **N38, N64, A05**, whose substance ships under a
different spelling. Both red files are retained at `.scratch/reds/`, byte-identical to their worktree
tips.

**The accepted rows changed `src/` in eleven places.** The docstring guarantee narrowed to the
`bias_transfer`-only check, stating that `calibration_bias` output is never checked and that a digest
match does not authenticate provenance (A03/A04, N71). `RECORD_KEYS` closes the accepted
evidence-record key set, so an unknown key is **refused** (`forbidden_key`) rather than silently
discarded (D04/N44-N49) — the existing token check could not be reused, because the probe's own
`between_event_r` carries a forbidden token. `REQUIRED_ARMS` 3 → 5 (the four groupings plus the null)
with `RULED_POPULATION` pinned at 178 pairs / 103 events (C05); duplicate arm labels refused (N40);
`REQUIRED_STATISTIC_FIELDS` closed with `above_0p5` the one nullable field (N52); a corpus table that
is not exactly one row refused before `mkdir` (N39); each capture and its `.sha256` sidecar checked
for disjointness from `out` (N17 — the directory check resolves the directory, not its entries);
reason codes on all 10 `validate_generation` refusals plus `_assert_owned`/`_assert_disjoint`
(N28-N32); `_fold` folding case, `_` **and** `-` on both haystack and needle, with `UNRUN_ARM_OUTCOMES`
deriving new `PROHIBITED_PARAPHRASES` entries (C13/A05); and `run()` reordered so every input is
validated **before** the retiring-sibling restore and `_assert_owned` (N50/N53). Suite 865 → 1032 lines, ~26 new cases; the publisher 980 → 1114.

**Both campaigns are re-derived in the primary tree, and the second re-derivation is the lesson.**
Determinism regenerated after the `wt/rev2-m2u71` squash-merge reproduced rev2's worktree run **byte
for byte** — all 39 verdicts and all three `baseline_sha256` digests identical, with only the two
`source_digests` keys moving — proving path-independence and proving the window-3 fixes move no
published byte. Mutation grew **42 → 51** for the new predicates, all killed; `validate_catalogue`
rejects a patch matching ≠1 time and a no-op patch before scoring, and it caught M46's anchor after
`ruff format` collapsed `RULED_POPULATION` to one line. Then the new cases failed `ruff check` (one
SIM300) and `ruff format --check`, and formatting `calibration_qc.py` **moved source bytes** —
invalidating every committed digest and every patch anchor — so both campaigns ran a second time.
**Order the static stages before campaign regeneration**; `conventions.md` now states it.

**Real inputs re-verified after the reformat.** 84 evidence rows over 21 arms, 15 claims, rc=0;
republish **byte-identical** across all three entries; `qualification/` **byte-unmoved** over its 5
files; **no staging or retiring sibling**; `validate_generation(...)` returns **7 keys at v1**. The
real capture was read before the new guards were written and confirms each of them: exactly the 9
expected record keys, all four grouping arms plus the null at 178 pairs / 103 events, and no null
required statistic field across all 21 arms.

**Six documentation surfaces applied from `doc-m2u71`'s drafts, corrected against shipped state.**
New `docs/technical/calibration_qc.md` carries the refusal-reason table, the five cited arms, the
population floor, the open-arm-set statement, `above_0p5`-only nullability, all four handled error
classes, the claim-scan fold, the pid-reuse restore, and "does not authenticate a capture".
`entrypoints.md` eight → ten console scripts with two sections; `architecture.md` two module-map
rows; `tests.md` ten inventory rows (five suites, five campaign scripts); `conventions.md` a new
`### Auxiliary campaigns` section carrying **measured** wall times, not the draft's estimates;
`qualification.md` the pointer D03 mandates. Roadmap claims corrected in the same pass: C03/C04
rescoped to the shipped estimator plus independent BA over disjoint RTMW-L subsets, C01/C11's M2.6
headline rescoped to RTMW-L/1080p/per-model-prior, A03's overstatement replaced with what is and is
not checked, R02's `~780 lines` removed, `source_sha256` → `source_digests`.

**Window 3 dispatched the adversarial wave and closed four defects.** Four teammates in one block:
`rev-m2u71` (N01-N80 conformance), `rev2-m2u71` (both campaigns), `rev3-m2u71` (claim soundness,
C01-C15 + D01-D06 + A01-A06), `doc-m2u71` (six documentation surfaces). **`rev3-m2u71` closed
27/27 — 16 pass, 11 fail — `rev-m2u71` closed 80/80 — 50 pass, 30 fail, 31 reds — and `doc-m2u71`
drafted 6/6.** Only `rev2-m2u71` was still filling at MAIN's reserve (36/81, tip `9e39cf7`); **every
worktree and `wt/` branch is retained**, and window 4 revives `rev2-m2u71` by `SendMessage` rather
than funding a successor. Full state, tips, the ruling set and the close-order recipe →
`.scratch/agents/main-checkpoint-m2u71-w3.md`.

**`rev-m2u71`'s 30 fails need triage, not acceptance.** It scored strictly against `05208f6` rather
than the `e494659` target my standing ruling moved it to, so three classes separate. **Already
fixed**: N10, N14. **Open scope rather than defect**: N57-N65, the campaign, documentation and config
carriers — both campaigns are committed seeds `rev2-m2u71` owns, and the six drafts await MAIN's
application. **Live**: N17 symlinked input overlap; N28-N32; N38-N40 and N44-N52 non-closed headers,
cardinality, arms and row keys; N50/N53 evidence validation ordered after ownership; N71. Those plus
`rev3-m2u71`'s eight are window 4's ruling set.

**Four defects fixed with committed acceptance checks; suite 76 → 80 cases.** (1) **N10, severe** —
`run()` opened by removing the same-pid `retiring` sibling, and pids are reused, so it could delete
the sole complete generation a kill left between the two renames. That is the exact state the
post-swap comment says the sweep is ordered to protect. The sibling is now restored to `out` before
ownership is judged. (2) **N14** — `_remove` used `rmtree(ignore_errors=True)`, which swallows
`NotADirectoryError`, so a regular file at either sibling path blocked the staging mkdir and the
swap. (3) **C01/C11** — the marker's `CLAIMS` are correctly qualified, but the module docstring
dropped RTMW-L, 1080p and the per-model intrinsic prior; an unscoped negative contradicts C11's own
exclusion. (4) **`main()`** caught `CalibrationQcError` and `qualify.QualifyError` alone, so a
tampered registry or session tree reached the operator as a traceback; `sessions.SessionsError` and
`inventory.InventoryError` join the handler.

**The teammate-conclusion rule earned a second datum.** `rev-m2u71`'s N10 red test asserted the
generation survives **at the retiring path**; restoring it to `out` is the stronger outcome, so the
committed case asserts that instead. **A reviewer's red test encodes one acceptable outcome, not the
only one — adjudicate the requirement, then write the acceptance check.**

**Window 2 shipped the publisher.** `src/pose_estimation/calibration_qc.py`,
`tests/test_calibration_qc.py` **74 cases, all green**, CLI `pose-estimation-calibration-qc`
registered in `pyproject.toml`, `.gitignore` two-pattern `calibration_qc` + `calibration_qc.*/`.
`ruff check`, `ruff format --check`, `ty check` all rc=0. Contract gained **A01-A06**
(`.agent/archive/contract-m2u71.md` §10), which close §5's column list and §6's implementation half.

**The design crux, and it inverts a wave-1 conclusion.** `scout-m2u71` E08 measured that no
corpus-row schema satisfies "every field has a probe-output key" and proposed dropping the scope
columns or changing the acceptance rule. **No such rule exists in the frozen contract** — §1 forbids
F1a *computing* a statistic, D06 fixes how *evidence* enters, and neither says the ruling must be
derived. So the corpus row is the **ruling**, a module constant no CLI argument can respell (A02),
and the evidence table alone traces to probe keys. That makes the ruling deterministic, D04
unrepresentability provable over the schema, and C13's unrun arm publishable as `unrun_arm` +
`unrun_arm_status` whose alphabet admits `unrun` alone. E08's ruling that C13 cannot enter a CSV is
overturned.

**What the tool actually guarantees, and the gap between that and the ruling's width.** It computes
nothing. It refuses to publish when the **`bias_transfer`** evidence has gone missing or moved: a
cited arm absent (the four groupings the transfer claim names, plus the permutation null), a
reference band absent (shared-image-bias / per-event-bias / noise prefixes), a required arm short of
the ruled 178-pair/103-event population, a duplicate arm label, a record key outside the closed set,
a statistic key or required field dropped, a capture taken under a different script version, or a
cited script missing. **What it does not check is `calibration_bias` output** — cited and digested,
never ingested — so the claims resting on that probe publish without their numbers being seen.
**A digest match binds a capture to one script version; it authenticates nothing**, so hand-written
stdout carrying the live digest reads like a real run.

**Shape.** Three entries. `corpus_qc.csv` = 1 row x 10 ruling columns. `evidence_qc.csv` = **long
form**, one row per (probe, arm, statistic) x 9 columns; wide-per-arm was refused because the arms do
not share a statistic set (A01). `calibration_qc.json` = marker, excluded from `tree_digest` and
carrying it, with `generation` = {two CSV digests, tree, qualification, probes, generator_version,
census}. `GENERATOR_VERSION = "v1"`.

**Two mechanisms that are new to this project's publisher idiom.** (1) **Evidence-to-script binding**
(A04): each capture is `<probe>.jsonl` + `<probe>.sha256`, and `validate_generation(..., probes_dir=)`
re-runs the comparison, so a probe edited after publication makes the set stale. (2) **Claim
conformance enforced at publication** (A05): `CLAIMS` (the C01-C15 required statements) publish into
the marker so a consumer reads the bound from the artifact; `PROHIBITED_PARAPHRASES` stays module-side
and is never published, because a set carrying that list would contain the text the scan exists to
keep out; `_assert_claim_conformance` runs over the **staged bytes before the swap**, case-folded with
`_` folded to space so a snake_case cell cannot smuggle an overreach. Both carry negative controls in
the suite.

**`calibration_bias` is cited and digested, never ingested** (A03). `probe_bias_transfer.py` emits one
uniform record per arm; `probe_calibration_bias.py` emits four differently-shaped record families, and
flattening them needs a per-family adapter — F1a taking a position on what each family means, which is
the estimator knowledge §1 forbids. Its numbers reach humans through M2.7.3.

**Carrier invariant holds, pinned.** `test_the_ruling_never_fills_a_qualification_sentinel` byte-
compares the whole `qualification/` tree across a publish and asserts `geom_qualified` blank +
`geom_unmeasured` present on every event row.

**Wave-1 correction, recorded because the previous window's record was wrong.** `map-m2u71` left
**S07-S12 unfilled — 6 of 12 sections, not the 2 the last window recorded**. S01-S06 are complete and
were the decisive input (the seven-point publisher contract, the crash-state table, the alphabet
traps, the consumer boundary, the 17-row registration surface). `map-m2u71-2` was dispatched this
window for S07 + S08 + S11 + S12 and **completed all four** — S12 carries **80 contiguous rows
N01-N80**, the anchored normative checklist §6's implementation half needs, and S08 carries the
determinism/mutation campaign shapes the remaining work consumes. S09 is superseded by D01 and S10
was MAIN-computed in one command (`sessions` M2.2 = 20 files / +3447; `qualify` M2.3 = 36 files /
+12301). Its one reported implementation gap was **validated and fixed in-window**: `_read_capture`
skipped a compact arm record cut before its closing brace, because the guard distinguishing the
probes' pretty-printed closing summary keyed on `endswith("}")` — so a killed run's truncated final
record was dropped instead of refused, and the truncation case in the suite had used a *malformed*
line carrying a trailing brace, which is a different input. The guard now keys on the summary
opener's real shape (a bare `{`), with two cases pinning both sides.

**The publisher has now run on real inputs, and window 2's evidence was entirely synthetic.** MAIN
captured `scripts/probe_bias_transfer.py --cache .scratch/calib-obs-wide` (51.9 s, 27 lines) into
`.scratch/f1a-evidence/bias_transfer.jsonl` with a `sha256sum` sidecar, then published against the
real `qualification/` tree: **84 evidence rows over 21 arms, 15 claims into the marker, one corpus
row, 1.556 s cold and 0.664 s warm**. A second publish is **byte-identical across all three
entries**, leaves `qualification/` **byte-unmoved**, and leaves **no staging or retiring sibling**.
`validate_generation(..., qualification_dir=, sessions_dir=, inventory_dir=, probes_dir=)` accepts,
returning all seven generation keys at `generator_version = v1`. The corpus row reads
`corpus,unachievable,cross_view_keypoint_bias,absent,rtmw_l,1080,per_model_prior,per_event_double_centered_bias_and_pose,unrun,bias_transfer|calibration_bias`.
Only `bias_transfer` needs a capture — `run()` ingests `INGESTED_PROBES` alone, so `calibration_bias`
is digested from the script and needs no `.jsonl`. **`pose-estimation-calibration-qc` is not on the
shared venv's path**: the environment predates the `pyproject.toml` row and every gate call uses
`--no-sync`. That is a stale install, not a defect, but the installed command is unexercised until a
`uv sync` runs — which mutates the shared environment and must wait for the wave to close.

**The `Read()` exclusion gap S06 found is closed.** `.claude/settings.json` `permissions.deny` gains
`Read(qualification/*.csv)` + `Read(qualification.*/)`, 29 entries to 31, verified both ways —
`qualification/assets_qc.csv` blocked, `qualification/qualification.json` still readable, matching the
`inventory/` precedent where the marker stays quotable. **`calibration_qc/` is ruled NOT denied**: it
is redaction-safe by contract, three small entries, and the artifact every review reads. The deny set
is a context-cost guard, and this tree costs nothing.

**Shipped surface.** `src/pose_estimation/calibration_qc.py` (1114 lines) + CLI
`pose-estimation-calibration-qc`; `tests/test_calibration_qc.py` and its focused oracle
`tests/test_calibration_qc_mutants.py`; `scripts/check_calibration_qc_determinism.py` with its
committed `tests/calibration_qc_determinism_results.json` and the `source_digests` tripwire;
`scripts/run_calibration_qc_mutations.py`; `docs/technical/calibration_qc.md` plus the four
registration edits S06 names and the `qualification.md` pointer D03 mandates. **S08 was the input for
both campaigns**, which shipped as runnable seeds at `e494659` graded both ways — 21 sweeps + 18
tampers at `unknown` (rc=1) and 42 mutants `UNENCODED` (rc=1) — and closed green in window 4.

**Sizing datum.** The publisher itself cost roughly half a window once S01-S06 were in hand — the
idiom is genuinely reusable and the crash-state table transfers line for line. What the window
actually bought beyond code: the A02 ruling, which a wave-1 report had settled the wrong way. **A
teammate's conclusion about the contract is attention-directing, never binding — validate it against
the frozen text before building on it.**

**Gate, window 4 and decisive.** `ruff check`, `ruff format --check`, `ty check` all rc=0; suite
**1438 passed / 0 skipped / 0 failed, rc=0** in 1216.78 s, primary tree, alone. Baseline **1406** at
`9f198f3` plus **32**: 26 authored across the two calibration-QC suites (87 → 104 and 41 → 50) and
**6 free**, because the existing negative-control case parametrizes over `PROHIBITED_PARAPHRASES`,
which the 6 `UNRUN_ARM_OUTCOMES` entries widen 17 → 23. Campaigns **51/51 mutants killed** in
162.96 s and **39/39 determinism verdicts** in 20.98 s, both re-derived after the reformat. Window-3
trend for sizing: 1358 in 916.39 s, then 1360 in 1018.37 s, then 1406 at `9f198f3`. **The suite now
costs roughly two passes** — `test_c8_08` spawns a nested full-suite subprocess — so run it detached;
a foreground call hits the 10-minute Bash cap.

`main=` window 2 100% 240K/240K, window 3 93% 222K/240K at reserve close, **window 4 59% 141K/240K**
— the whole unit closed inside one window with reserve to spare, because no teammate was funded.
`mate=` 72% 173K/240K (`rev3-m2u71`; `doc-m2u71` 67%, `rev-m2u71` 64%, `rev2-m2u71` 63%), unchanged:
**window 4 ran MAIN-only**, every teammate having been stopped in window 3.

**Sizing datum, and it is the one PLANNING should carry forward.** A window that only *rules and
applies* — no dispatch, no worktree, no polling — closed 45 review rows, 11 source changes, two
campaign re-derivations and six documentation surfaces at **59%**. The window-3 wave cost 93% to
produce those rows. Reviewing is what needs teammates; adjudicating and applying does not, and
splitting them across windows is what keeps both inside the one-window aim.

**Sizing datum, and it is the expensive one.** All four teammates reached 44-52% context with **zero
cells flushed and zero commits**, then filled 30-100% of their tables within one poll of a single
flush directive. The seeds were committed and runnable before dispatch, the grade command was named,
and the batch size was stated — and none of that started the flush on its own. **A deliverable-first
brief buys the recovery, not the cadence: budget one flush directive per wave at roughly the
half-context mark and send it before the counters are flat twice.**

### M2.8 — 2D cohort aggregate delivery — PLANNED, 3 units

**USER RULED: cohort aggregates only.** No per-subject rows, no patient identifier, no join column, no
join template, and deliberately no join to `../rehab`. The user chose this with the non-consumability
tradeoff stated. The join surface is **cut**; `analysis/make_templates.R`'s operator-filled
`sessions.csv` pattern is not extended, and the two pri-1 polish rows serving that path stay deferred.

**The ruling SHRINKS the consumer delta rather than only making it safer.** `res-m2-1` priced full
ingest into `../rehab` at one untracked table plus six tracked files, whose three heaviest items are
the authorized identifier map, a join-validating loader, and the dataset merge. A cohort artifact
carries no `IDNumber`/`TIMES`, so **all three vanish** and the delta falls to its two cheapest items:
land a table, append descriptors.

**The aggregate is well-conditioned, measured.** `(task, side)` = **12 cohort cells** over tasks
`cap coin glass key nut peg` x sides `l r`, each holding **15 or 16 distinct subjects**, one family
per subject per cell; 188 = 12 x 16 - 4 absent. **Zero cells below 5 subjects**, so no small-cell
suppression is needed at that grain. A finer grain — adding `view` — shrinks cells and reopens the
question, so the published grain is a contract decision, not a convenience.

**Consumer shape, measured against `../rehab` as it stands.** Deliver an **append-ready
`columns.yaml` descriptor fragment** (`group: pose`, `role: feature`, full `ja` + `en`, dtype/range/
unit); a separate `schema/pose_features.yaml` is worse, because it forces a `schema.py` load/merge
change. **`short_ja`/`short_en` do not exist** anywhere in that repo — comment-only aspiration, absent
from descriptors, from `ColumnSpec` and from dashboard code — so the earlier "optional short labels"
premise is retired. `ui_strings.yaml` carries 395 keys, every one exactly `ja`+`en`, 0 missing arms;
a column's label belongs in its descriptor while pose-specific headings belong in `ui_strings.yaml`.
Fonts ship as **10 subset WOFF2 faces**, so new Japanese glyphs plausibly force a consumer-side font
rebuild — keep new JA text inside existing coverage and state the risk in the delivery note.
`data/processed/` is gitignored and pyarrow is installed, so Parquet is natural and CSV acceptable.

| id | tier | spine result | sizing |
| -- | ---- | ------------ | ------ |
| M2.8.1 | kernel | Corpus-run preconditions + instrumented pilot. Fix `--output-dir` (`run.py:663-668`); make the three silent-drop guards (`analysis/clinical_features.R:1294,1300,1305`) emit a keyed QC row; instrument the CFR fallback rate; stratified pilot spanning both codecs, all four device configs and all four rotations. | 1-2 windows |
| M2.8.2 | data | Full corpus 2D run, ~6.5 h, resumable. Per-asset clinical features plus a run manifest giving every one of the 379 assets an explicit disposition, so no asset is silently absent from a denominator. | 1-2 windows |
| M2.8.3 | kernel | Cohort aggregate publisher + append-ready bilingual descriptor. 12 `(task, side)` rows, `n_subjects`/`n_events`, per-feature distribution statistics. | 2 windows |

**M2.8.1 exists because M2.6 recorded why it must.** "Where a milestone's spine rests on an unmeasured
empirical assumption, the feasibility probe is its own unit with its own budget." M2.8's spine assumes
the 2D path runs clean over 379 real assets; **it has only ever run over 12 preliminary clips from the
retired `videos/initial/` tree**. Two blockers are already measured rather than suspected, and both
are silent: `--output-dir` is ignored in native rtmlib session mode, which is the entry point M2
drives the corpus through, so a 6.5 h run lands somewhere other than where it was asked; and three
enumerator guards drop a whole person with **zero rows and no record**, which for a cohort aggregate
moves `n_subjects`/`n_events` with nothing saying so. Funding the run before these close is the exact
mistake M2.6 paid three windows to learn.

**What the 2D path already gives, so it is priced as delivery and not development.**
`analysis/clinical_features.R` is pinned by six committed goldens (`2d_csv4dp_*`, `2d_cumsum_*`,
`2d_idx_*`, each with a `_windows` companion), and M2.4 already replaced `1/median(diff(ts))` with
`nominal_fs()` at the call sites, so the cadence under every rate-based feature is the corrected one.

**Corpus readiness, measured.** 379 canonical assets, `decode_status = ok` on **379/379**;
`frames_decoded == frames_reported` on **379/379**, totalling **337 090 frames**; `detect_rate` mean
**0.9899**, median 1.0, min 0.3333, with 367/379 at >= 0.90. **Container PTS reordering is not a
risk**: `pts_monotonic = 0` on 123/379, but `SourceClock.timestamp()`
(`src/pose_estimation/video_io.py:28-77`) guarantees strictly-increasing timestamps — a regressing or
repeated `cv2.CAP_PROP_POS_MSEC` falls back to `idx/fps`, then a second guard forces `last + 1/fps`.
The open predicate is the **CFR fallback rate**, which M2.8.1 instruments; 123 is an upper bound on
exposure, since qualification measured PyAV demux order while the run path reads cv2 presentation
order.

**What the negative leaves standing, and it is more than the 2D line.** M2.1 inventory (`capture_id`,
container facts), M2.2 sessions (193 events / 382 assets), M2.3 qualification (379 assets), M2.4
timebase truth, M2.5 alignment (`offset_s`, realized residual median 6.31 ms — still meaningful with
no 3D, since it supports cross-view comparison and best-view selection per event), and the
`measurements/` sidecar. Only the extrinsics line died.

**Registry facts that bind the descriptor** (`res-m2-1`, measured, superseding the earlier summary).
`schema/columns.yaml` = 67 direct descriptors + 152 family-expanded = **219 `ColumnSpec`**; observed
key union `{raw, ja, en, group, role, dtype, unit, range, levels}`, six required. Groups today =
`demographics id injury isncsci isncsci_motor isncsci_sensory meta scim`; roles =
`feature id meta outcome`; dtypes = `categorical datetime numeric ordinal`. `missing_sentinels` =
`["_", "", "NA", "NT", "ND"]`. **A new `group: pose` string enters automatically** — `load_schema()`
enforces required keys by `KeyError` alone, with no unique-raw check and no enum check, and duplicate
raw names silently keep the last. **Schema rows alone do not make fields into dashboard features**:
`ADMISSION_FEATURES`/`NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` are hard-coded at
`../rehab/src/rehab_sci/data/dataset.py:40-107`. `schema.py` carries no patient data, so the
descriptor stays separable from the table and the same posture holds here.

**Planning record.** One window. Wave 1 = `map-m2` (publisher idiom + F1a shape; stopped at 97%
saturation, 6/10 sections, harvested), `res-m2-1` (`../rehab` study, 8/8, complete), `res-m2-2`
(prospective-capture research). Wave 2 = `plan-m2`, which drafted an **8-unit split for M2.7 alone**;
MAIN arbitrated it to 4. **Wave 3 `planrev-m2` was NOT funded** — MAIN reached context reserve first,
so the split carries no adversarial review and its sizing is MAIN's alone. That is the one soft spot
in this plan: treat every window estimate as untested, and let M2.7.1's own wave-1 map correct it.

**Two sizing data for the next PLANNING window.** (1) **A planning agent splits by risk, not by
value.** `plan-m2` produced a defensible 8-unit M2.7 from the every-unit-overran record, but F1a
publishes one corpus row and owns no estimator, while the M2.5 analog it was sized against bought a
379-row solved artifact with a solver. **Size a publisher by what it computes, not by the idiom it
reuses.** (2) **MAIN out-measured the wave again**, for the third consecutive window: the cohort-cell
census, corpus readiness, sizing archaeology, the sentinel verification and the PTS resolution were
all MAIN scripts, and the PTS question had already cost `map-m2` most of a window when MAIN answered
it in two reads. **On a planning window, delegate the reading and the external research; keep every
script-derivable census in MAIN's hands.**

**Out of scope, explicitly.** Editing `../rehab`. Integration is that repo's work under its own
`CLAUDE.md` and roadmap. M2.8 ends at a published artifact plus its schema descriptor.

### M2.6b — G0 CLOSED NEGATIVE; the bias does not transfer, so there is nothing to model

**The user funded the one repair route A14 left open, and it is refused by measurement in one
window, at its premise, before an estimator was built.** Route = a joint bias-and-pose
parameterization under an identifiability constraint, Malleson-style (IJCV 128, 2020 §3.3.2). Probe
`scripts/probe_bias_transfer.py`, reading the same caches and reusing `pair_structure` verbatim, so
the instrument is A10's.

**G0 — is the bias shared across recording events?** The route needs a bias with fewer degrees of
freedom than the data. A per-event bias field is 2 x C x K = **390 free parameters against 11 pose
DoF with no external anchor** — A14/Q03's exactly-degenerate case. The only two ways to reduce it are
sharing across events, or Malleson's anchor (already-calibrated cameras plus ground-truth optical
bone transforms), which no retrospective corpus can acquire. So the whole route rests on transfer.

**Measured on the FULL eligible population, not a sample.** `--stratum-events 25` collected all
**115 eligible events** (74 two-camera + 41 three-camera) at 32 frames, det-CPU / pose-NPU, yielding
**178 camera pairs over 103 events**; 12 events yield no usable pair. Residual magnitude median
**17.04 px**. A09-A13's 22-event sample is no longer a limit on this unit's negative.

**A10's statistic, re-split across EVENTS instead of frame blocks, reads the null — and the contrast
sharpens on the full population.** Within-event ceiling rises to **r median 0.8138, 129/178 pairs
above 0.5**, so the bias is *more* clearly reproducible inside an event here than in the 39-pair
sample (0.7029). Between events it is gone, at four successively stricter groupings:

| grouping | n | signed r | above 0.5 |
| -------- | - | -------- | --------- |
| same view pair | 4341 | **0.0108** | 787 |
| + same device-model pair | 2738 | 0.0102 | 527 |
| + same task | 1071 | 0.0103 | 189 |
| + same subject | 275 | **-0.0296** | 52 |
| keypoints permuted (null) | 4692 | 0.0051 | 185 |

Per view pair: `above|left` 0.0311 (n=1207), `above|right` -0.0029 (n=1858), `left|right` 0.0117
(n=1276). **The corpus sits at its own permutation null on a statistic returning 0.81 within an
event**, and the subject grouping — the last live variant, since a subject-anatomy bias would pool
that subject's ~7 events per field — is the one that goes negative.

**Calibrated references separate completely, and the corpus is on the wrong side of the gap.** Every
synthetic arm runs the identical statistic through the real cache's validity masks, image sizes and
device models, with the rig re-jittered per event so the correlation is measured across different
camera placements exactly as the corpus one is, and each arm pools **3 independent field draws**
because one draw is one realization of the mechanism.

- **Shared** bias fields — the repairable mechanism: image-fixed 8 px **0.9662**, 32 px **0.7632**;
  Malleson's own parameterization, a constant 3D per-(camera, keypoint) offset, at 20/40/80 mm
  **0.9411 / 0.2192 / 0.6264**. Under rig jitter widened to 0.6 m and 1.2 m — far past any plausible
  placement change — 8 px holds **0.9236 / 0.7961** and 32 px **0.6429 / 0.1799**.
- **Not shared**: per-event bias 8/32 px **-0.0106 / 0.0052**; zero-mean noise 8/32 px
  **0.0159 / 0.0303**.
- Shared arms span **0.180-0.966**; non-shared arms span **-0.011-0.030**; the corpus spans
  **-0.030-0.031** across five groupings. No overlap, and the corpus sits inside the non-shared band.
  The shared arms are **not monotone in magnitude** — field realization dominates at 3 draws, which
  is why the claim rests on the 0.180 floor across every shared arm rather than on any single value.

**The magnitude split names what the corpus does share, and it is not correctable.** Correlating
|residual| instead of signed residual gives **0.1499** pooled against a permutation null of -0.0324
and a per-event-bias reference of -0.0033, and it stays at **0.1462** within subject and 0.2191 on
`left|right`. So **the same keypoints are hard everywhere, at every grouping, while the direction of
the offset is redrawn every event.** A difficulty ranking is not a correctable offset — magnitude
cannot be subtracted from a coordinate — which is also why A13 found no keypoint subset that
transfers.

**Bound of the negative.** It bounds a bias model keyed on any of view pair, device-model pair, task
or subject, measured through signed epipolar residuals over the full eligible population. Its power
is established by the synthetic arms, which hold r >= 0.180 under the corpus's own masks and 1.2 m
placement jitter; those arms do not span subject-to-subject anatomical variation or the 7 assets that
change orientation mid-clip. That limit cuts toward the negative rather than away from it: a bias
transferring only within one subject **and** one viewing geometry is estimable only per event, which
is the degenerate case the route had to escape.

**A second refusal, independent of G0, and it survives even if a per-event solve is identifiable.**
Under a joint multi-camera parameterization, rotation cycle closure is an algebraic identity of the
solve, not a check — the same reason contract A02 kept A12's pairwise BA poses independent. The
corpus offers no other acceptance statistic: held-out reprojection on the solve's own keypoints is
self-consistency and is prohibited as accuracy (P05, Pätzold et al.), and cross-event transfer, the
one out-of-sample target that would have replaced closure, is what G0 measures absent. **A per-event
joint solve could therefore be built and could never be credited on this corpus.**

**Unrun, and named.** Whether a per-event double-centered bias-and-pose solve recovers known
extrinsics on the synthetic control is not measured — the parameter count argues it is degenerate,
and G0 plus the acceptance gap make it undecidable on the corpus either way, so it was not built.
That is the one arm a future attempt could still run, and it would refine the claim from "no
transferable bias exists" to "identifiable in principle, unverifiable here"; it would not reopen the
route.

**Two caches, kept apart on purpose.** `.scratch/calib-obs-f32/` holds the 22-event
`--stratum-events 2` sample and is A09-A13's population — verified after the widening to still return
39 pairs / r 0.7029 / 26 above 0.5 / 20.773 px / variance fraction 0.4327, matching M2.6's record
exactly. `.scratch/calib-obs-wide/` holds all 115 and is M2.6b's. Both are gitignored and regenerate
from the committed probe; the split is by `_event_key` over the deterministic selection, so it is
reproducible rather than a manual sort. `scripts/probe_calibration_observability.py` gained
`--stratum-events` (default 2 unchanged) to make the wide population re-derivable; the sample is a
hash-ranked prefix, so raising it only ADDS events to the selection — the 22 narrow entries are a
strict subset of the 115 wide ones by cache key. It does **not** make entries reusable: measured,
`stratum_events` joins the cache fingerprint (2 -> `2b84d350…`, 25 -> `d427f95f…`), and `load_event`
rejects a mismatch, so raising the value re-collects the entire sample. Separate `--cache`
directories are the mechanism, not a convenience. Deferred to polish: the binding is over-broad,
since no cached per-event value depends on how many events were selected.

**Gate at close.** `ruff check`, `ruff format --check`, `ty check` all rc=0. Decisive suite
**1284 passed / 0 skipped / rc=0** in 1232.97 s, unmoved from the `41efc55` baseline — correct for a
window shipping one probe and no production surface. A first gate run failed
`tests/test_r_timebase_truth.py::test_c8_08` on `subprocess.TimeoutExpired` while the 115-event
collection held the CPU; **never run the gate beside a decode/inference sweep** — the R cases carry
subprocess timeouts that CPU contention alone can blow. Re-verified alone. The suite predates one
later `--stratum-events` help-text edit; `rg -l` over `tests/` finds no reference to any of the three
probes, so no case can observe it, and the three static checks reran green on the final bytes.

**The whole M2.6b record re-derives from the shipped probe.**
`probe_bias_transfer.py --cache .scratch/calib-obs-wide` returns 178 pairs / 103 events, signed r
0.0108 (n=4341) with 787 above 0.5, within-event 0.8138 (129/178), residual median 17.038 px,
magnitude 0.1499 pooled / 0.1462 within subject / -0.0324 null, and every synthetic arm at the value
recorded above. Re-derivation is the acceptance for these numbers, since the cache is gitignored and
no test covers a probe.

**One claim of this unit was refuted by its own scale-up and is recorded rather than quietly fixed.**
On the 22-event cache the within-subject magnitude correlation read 0.0025, which supported "the
shared difficulty is between-subject". On all 115 it reads **0.1462**, indistinguishable from the
pooled 0.1499 — shared keypoint difficulty is present *within* subject too. The signed result is
unchanged at every grouping; only that secondary reading moved. Same lesson as A02: **a number
measured on a sample gets re-derived at scale, not carried forward.**

**Sizing, recorded for PLANNING.** One window, one probe, one fork closed. **The new datum: a repair
route names a premise, and the premise is cheaper to test than the repair.** The route needed a bias
with fewer degrees of freedom than the data; that reduces to transfer or to an external anchor, and
transfer was one re-split of a statistic already shipped. No estimator was written. Where a funded
route rests on a structural precondition, **measure the precondition first and budget the estimator
only after it survives.**

### M2.6 — F0 CLOSED NEGATIVE; verdict ships through F1a

**Extrinsic recovery from RTMW-L keypoints on this corpus at 1080p under per-model intrinsic priors
is measured unachievable, and the cause is measured rather than assumed.** Those three qualifiers
travel with the negative: a lower-bias keypoint source, a detector trained for multi-view
consistency and prospective calibrated capture all stay outside the bound. Three windows against F0 ("does M2.6 exist as scoped?"). Contract at
`.agent/archive/contract-m2u6.md`: §1-§6 bind as the record of the route tested, amendments A01-A08
(windows 1-2) and **A09-A13 (this window, the closure)**. F1-F3 are moot as written — they configure
a publisher for extrinsics the corpus cannot yield. Baseline `7fcf329`.

Every number below reruns from committed state via `scripts/probe_calibration_bias.py`, which reads
the caches `scripts/probe_calibration_observability.py` writes and reuses that probe's estimator
verbatim, so the instrument under test is the shipped one.

**The instrument is exonerated (A09).** A synthetic positive control drives *known* extrinsics through
the real cache's own per-(camera, frame, keypoint) validity masks, image sizes, device models and
10-event three-camera population — only the geometry becomes known. At zero correspondence error the
shipped pooled-`recoverPose` route returns cycle closure **median 0.000 deg (max 0.004)**. It holds
10/10 events inside the 10 deg bound out to **sigma = 8 px** (median 2.746) and degrades
monotonically: 16 px → 4.925 (7/10), 32 px → 39.913 (1/10); the anatomical-bias arm arrives at the
same place at 20-40 mm (19.07 / 28.18 deg). The corpus reads 2/10 at median 39.0-47.4 deg, which
prices its effective cross-view correspondence error near **30 px at 1080p**. **The estimator was
never the blocker** — §6b's "initialization, not observability" alternative is refuted.

**The mechanism is confirmed by direct measurement (A10).** One pooled pose per pair is fit on a
training frame block; per-keypoint mean **signed** epipolar residuals are then correlated across two
**disjoint** held-out blocks. The identical statistic on synthetic data calibrates it — zero-mean
noise at sigma 2/8/32 px gives split-half **r = 0.010 / 0.007 / 0.120**, fixed bias (image-fixed
8-32 px or anatomical 20-80 mm) gives **r = 0.993-0.998**. The corpus, 39 pairs, gives **r median
0.703, 26/39 above 0.5**, between-keypoint variance fraction **0.433** against noise 0.070-0.087 and
bias 0.88-0.96, at residual magnitude median **20.8 px**. The confound is controlled: the sigma-32
noise arm carries pair rotation error median 13.7 deg — a pose as wrong as the corpus's — and still
returns 0.120, so **a wrong pose alone does not manufacture split-half reproducible per-keypoint
structure**. Systematic component **15-20 px** by two decompositions, remainder random; both orders
above the 1-4 px regime published keypoint calibration works in.

**Two repair routes priced and refused.** Independent pairwise BA (robust Sampson, 5 DoF, poses
independent so A02's condition holds) makes closure **worse**: 8-frame median 37.17 → 40.53, 2/10 →
1/10; 32-frame **39.00 → 78.89**, 2/10 → 1/10, median max pose move 43.32 deg (A12). A better fit to
biased correspondences moves further from truth and the damage grows with data — RANSAC's rejection
was partly shielding the estimate. And no keypoint subset rescues it (A13): ranked on one event fold,
scored on the disjoint fold, both directions, **no subset beats all 65 on held-out events** (fold 1
all-65 47.42 vs cleanest-40 41.87 / cleanest-24 108.96; fold 2 all-65 21.04 (2/5) vs cleanest-40
64.35 / cleanest-24 49.72 (2/5)), and the **cleanest ten keypoints still carry 49.6-53.9 px** mean
absolute residual. The bias is corpus-wide, not concentrated.

**A08's uncovered population gains a statistic (A11).** Split-half r is per **pair**, so it reaches
events with no cycle: 2-camera pairs **r median 0.776** (7/10 above 0.5, 20.4 px) against 3-camera
pairs 0.638 (19/29, 22.2 px). The bias is a corpus property, not a three-camera artifact, and the 80
two-camera events A08 named now have an internal consistency statistic needing no third camera.

**Bound of the negative, stated as precisely as a positive would be.** It bounds extrinsic recovery
from **RTMW-L keypoints** on **this corpus** at 1080p under **per-model intrinsic priors**. It does
not bound a keypoint source with lower cross-view bias, a detector trained for multi-view consistency,
a joint bias-and-pose parameterization under an identifiability constraint M2.6 never specified, or
any prospectively calibrated capture. **The detector and the viewpoint separation are what a future
attempt must change — not the solver and not the sample size.**

**Five causes refuted, one confirmed.** Refuted by measurement: planar degeneracy (homography inliers
0-1 vs 39 essential), low parallax (median 72-129 deg), alignment (sync residual median 7.85 ms,
p95 25.09, max 32.71, n=336 — inside one 33.3 ms frame, so M2.5's offsets are confirmed working),
undersampling (4x frames left closure at 2/10), estimator/initialization (A09, A12). Confirmed by
measurement: cross-view keypoint correspondence bias (A10).

**Three rulings from windows 1-2 that still bind, whatever replaces M2.6.**

- **The metres chain.** An arbitrary-scale extrinsic written into today's `CameraCalibration`
  propagates a false metric claim through six shipped surfaces: `_types.py:91` → calibration docs →
  `triangulation.py:12,454` → `export.py:443-487` (`_x_m/_y_m/_z_m`) → `validation.py:1265-1293`
  (×1000, millimetres) → `analysis/clinical_features.R` (`coord_space="world-metric-3d"`). The
  projection math is unit-agnostic, so each accepts arbitrary units numerically while its contract
  lies. Any arbitrary-scale geometry needs its own type with explicit scale provenance.
- **The carrier.** `session.json` may not be patched (`sessions.validate_generation` hashes the tree).
  `sync_offset` may not be written (frame domain vs time domain). `events_qc.geom_qualified` cannot be
  filled by a publisher reading `qualification/` — that is the publisher cycle M2.5 refused. All 193
  rows still read `geom_unmeasured`; `geom` remains an M2.6-owned token.
- **The acceptance statistic.** Held-out reprojection on the solve's own keypoint family is
  self-consistency, never accuracy (Pätzold et al. GCPR 2022: beat the reference on human reprojection
  4.01 vs 4.57 px while losing to it on independent AprilTags by 3.05 px). Same discipline as M2.3's
  acoustic closure.

**Candidate population, still valid and MAIN-derived.** **121 events carry ≥ 2 offset-bearing
cameras** (80 two-camera + 41 three-camera) — the ceiling. 115 `sync_status = connected`, 6
`unconnected` (P07 partial publication). Over the **283 offset-bearing cameras** inside them:
**223 `rigid`, 46 `unmeasurable`, 9 `camera_motion`, 5 `excluded_orientation`**; `detect_rate` median
1.0000 / min 0.7083. **64 of 121 events have every offset-bearing camera `rigid`.** Ruled:
`unmeasurable` does not disqualify on its own.

**The user ruled: fund the repair route.** That became M2.6b, which closed it negative at its own
gate. **Ruled since, on that result:** publish the negative through **F1a** — a new `calibration_qc/`
publisher carrying the verdict as a corpus-level ruling, with `events_qc.geom_qualified` staying
`UNMEASURED`. See A16 for why F1a rather than F1b/F1c.

**The literature corroborates it and names the one repair route (A14).** Malleson, Collomosse &
Hilton, IJCV 128 (2020) §3.3.2 Eq. 17-18 models a per-camera keypoint offset explicitly, **calls the
discrepancy "systematic bias"**, measures offsets reaching **6.5 cm** (50-61 px at this corpus's ~2 m
and fx 1553-1873), and reports its zero-offset arm at **1.22x** the full model's global pose error —
so M2.6's measured 15-20 px systematic component is modest against published values. **None of the
located keypoint-extrinsic methods models signed view-dependent bias**: Lee et al. use a Gaussian
per-joint sigma plus RANSAC, Pätzold et al. a heatmap covariance. Both model **noise**, the wrong
estimator for a fixed offset — which is why A12's robust BA moved further from truth, not closer. The
one live repair route is a joint bias-and-pose parameterization under an identifiability constraint;
that is a different unit from M2.6 as planned and this ruling does not fund it.

**Wave state.** `res-m2u6-3` (this window) answered Q01-Q03 of 5 at 32% before MAIN stopped it at
reserve; Q04 (accuracy vs camera angular separation) and Q05 (published negatives, closure precision
in practice) are unowed → `.scratch/agents/res-m2u6-3.md`. **`wt/spike-m2u6-ba` and
`wt/spike-m2u6-sweep` are REMOVED** — worktrees and branches both, at the M2 plan close. Their named
open dependency was F0, now closed, and the M2.7/M2.8 scope does not claim them; the evidence they
held re-derives from committed state, since shipped `scripts/probe_calibration_bias.py` carries all
four arms (`control`/`structure`/`ba`/`subset`) that produced A09-A13. Wave-1 reports
preserved: `.scratch/agents/map-m2u6.md` (12-section map, 110-row anchored checklist, S12 unfilled),
`res-m2u6.md` (Q05-Q08 unowed). Caches survive at `.scratch/calibration-observability/` (8 frames)
and `.scratch/calib-obs-f32/` (32 frames); both are gitignored and regenerate from the committed
probe.

**Gate at close.** `ruff check`, `ruff format --check`, `ty check` all rc=0. The decisive suite is
**unmoved from the `41efc55` baseline of 1284 passed / 0 skipped** — correct, since this window
shipped two probe scripts and no production surface. `scripts/probe_calibration_bias.py` is new and
`scripts/probe_calibration_observability.py` gained `--frames-per-event` (default 8 unchanged, value
already inside the cache fingerprint), which closes the standing polish row and makes the 32-frame
cache — the primary input to every A09-A13 number — re-derivable from committed state.

`main=` 83% 200K/240K at reserve close. `mate=` 32% 77K/240K (`res-m2u6-3`).

**Sizing, recorded for PLANNING.** Three windows on one fork. **Four new data.**

1. **A positive control is the cheapest thing that could have been run first, and it was run third.**
   Windows 1 and 2 spent two full windows narrowing hypotheses against an instrument nobody had
   calibrated. The control cost one script and minutes, and it simultaneously exonerated the
   estimator, priced the error budget in px, and supplied the reference distributions that made the
   bias statistic readable. **On any empirical fork, calibrate the instrument against known ground
   truth before interpreting a single result from it.** M2.3 learned the same lesson through P21's
   unadjudicable gate; this is its constructive form.
2. **A null needs a mechanism, and a mechanism needs a discriminating statistic with calibrated
   references.** "Bias is the only surviving hypothesis" (window 2) and "bias is measured at r = 0.703
   against 0.010 noise / 0.997 bias references" (window 3) are different epistemic objects. Only the
   second can be published, and only the second refuses the repair routes.
3. **MAIN out-measured its own wave for the second consecutive window.** Every decisive number in
   windows 2 and 3 came from MAIN running scripts directly. This window funded exactly one teammate,
   on the one genuinely parallel track (literature), and spent the rest of the window measuring —
   which closed in one window a fork two mixed waves had left open. **Script-derivable empirical
   forks belong in MAIN's hands; delegate the reading, not the measuring.**
4. **Scope a unit's first window to ask whether the unit exists.** M2.6 was planned as "recover
   extrinsics, publish them" and its real content was "find out whether extrinsics are recoverable."
   The plan carried no unit for the second question, so it was answered inside a unit budgeted for the
   first. **Where a milestone's spine rests on an unmeasured empirical assumption, the feasibility
   probe is its own unit with its own budget.**

### M2.5 — DONE

**Product.** `qualification/cameras_qc.csv` at `GENERATOR_VERSION` v4, P03's eight columns, one row per
placed asset: **379 rows over 193 events = 355 carrying an offset + 24 `unreachable`**. `events_qc`
gains `sync_status`. Offsets come from gauge-fixed unweighted least squares over accepted edges
(`QUALIFIED_PAIR_STATUSES`), restricted to the reference's connected component. `graph_connected`,
`sync_status` and `offset_span_s` are read back out of the published camera rows, so the two tables
cannot disagree; `graph_connected` stays 173/193. Alignment ships here rather than in `sessions/`
because writing offsets back into a registry-derived manifest closes a measured `sessions` → `qualify`
→ alignment → `sessions` digest cycle.

**M2.6 consumes this.** Offsets live in `cameras_qc.csv` alone. Manifests keep `sync_offset: 0`, the
legacy integer pre-roll trim in the fusion reader's frame domain. The fusion frame reader does not
apply the published offsets — that is later work, recorded in `docs/technical/validation.md`.

**Sign proved, not assumed.** `offset_s = t_camera − t_reference`; positive means that camera started
earlier; reference exactly `0`; application `t_ref = t_camera − offset_s`. A synthetic 375 ms lead
recovered `+0.375000058` A→B and `−0.375000058` B→A, antisymmetry error 0. A sign flip yields a fully
connected, digest-valid artifact whose cameras move twice as far apart in time, and no structural
check can see it.

**Reference ruled on totality.** View hierarchy `above` > `left` > `right`, tie-broken by lowest
`asset_id` — the only semantically meaningful rule total over 193 events (155/24/14). Latest-start is
undefined on exactly the 20 unconnected events; `above` alone covers 155/193; highest degree is unique
on only 69/193.

**Two populations, permanently named apart: 379 / 355 / 24, never 329 / 50.** 329 is `events_qc`'s
population — cameras inside a graph-connected event — and 355 is `cameras_qc`'s offset-bearing rows.
`scripts/check_cameras_qc_census.py` pins both side by side, because this project has now conflated a
pair of same-shaped counts three times.

**Solver ruled on measurement.** Unweighted least-squares over every accepted edge beats the shipped
breadth-first tree: it differs by median 0, p95 3.708 ms, **max 10.095 ms = 0.303 nominal frame**,
moving 60 of the 329 cameras the probe compares and 6 nearest-frame indices across 5 events, all
inside the 30 events carrying a redundant edge. **329 is the probe's own population** — every camera
in a graph-connected event, which is what `scripts/probe_alignment_solver.py` reports as
`cameras_compared` — and never a count of solved cameras. The 26 further offsets P07 publishes sit in
single-edge components, where a tree solve and a least-squares solve coincide, so the mover count is
unchanged over the full 355. Dropping
any one redundant edge moves **0/90 solves** by more than one frame. Confidence weighting is
**rejected**: Spearman `peak_rms` against absolute audio-visual disagreement is **+0.4141** — the
wrong sign for a precision weight — and `peak_ratio` is +0.0659, so neither is an inverse-variance
estimate.

**No per-camera uncertainty is publishable.** 74 connected two-camera events and 11 connected
three-camera trees carry **0 residual degrees of freedom** — 85 of the 115 connected multi-camera
events. Only the 30 closed triangles reach 1 df, where correlated acoustic and rolling-shutter bias
violates the model and closure is structurally bias-blind. Event-level closure stays the published
self-consistency statistic, labelled self-consistency and never accuracy.

**Rulings.** Contract and all three amendments at `.agent/archive/contract-m2u5.md`. **A01**: P07
stands — 6 of the 10 three-camera failures hold the view-hierarchy reference inside the two-camera
component, so partial publication recovers cameras `spike-m2u5-solve` Q08 would have nulled. **A02**:
P19's frozen census was derived under a rule P07 replaced; recomputed to 355/24. **A03**: 14 phase-1
readings batch-ruled, with Q01, Q04, Q05, Q08 and Q12 ruled against.

**Review closed.** `rev-m2u5-2` 38/38 rows, `rev2-m2u5` 33/33, `doc-m2u5` 14/14, `test-m2u5-2` 55/55
cases; both registers empty. Accepted findings and their fixes: **C09** cell alphabets built from the
status frozensets, because `[a-z_]+` is a shape that accepted every token the partition excludes;
**C31** `peak_ratio` +0.0659 added to the solver docstring; **D06** `_canonical(rows, key)` fixes
pairs/cameras/events row order at the publish site, with `assets_qc` ruled exempt in registry order
and pinned by a new D09 row; **P20** determinism regenerated; **X02/X04** the 329→355 denominator
above; **X10** a 27-word two-action instruction split. **C46, C47 and C49 were ruled test defects** —
the docs state the required facts, and C49's regex fired on the very disclaimer O24 asked for.

**Campaign gates, every one credited by MAIN's own rerun.** `scripts/run_m2u5_mutations.py` **25
mutants / 25 killed / 0 survived**. `scripts/check_m2u5_determinism.py` **D06-D09 all PASS, 0
failures** across flagless and measured modes, with both negative controls firing — `cameras_qc` FAILs
in both modes without `_canonical`, and D09 FAILs when asset rows are sorted.
`scripts/check_qualify_determinism.py` **40/40 sweeps, 19/19 tamper classes**.
`scripts/check_cameras_qc_census.py` **21/21**, plus a 379-row differential against
`scripts/orc_cameras_qc.py` — an implementation sharing no solver line with `qualify` — at **0
findings**, worst offset delta 3.333e-10 s.

**Gate.** `ruff`, `ruff format`, `ty` clean; decisive gate **1284 passed / 0 skipped** in 771.92 s,
primary tree, package path printed before collection. The v4 republish is byte-identical to the
shipped tree across all five artifacts, so the row-order fix changed no published byte.
`scripts/probe_alignment_solver.py` and `scripts/orc_cameras_qc.py` both ship here, so every solver
and census number above reruns from committed state rather than from a worktree.

`main=` 90% 215K/240K. `mate=` 90% 216K/240K (`rev2-m2u5`).

**Sizing, recorded for PLANNING.** Four windows against a one-unit plan; the fourth bought review
harvest, six accepted findings and the campaign reruns alone. M2.1, M2.3 and M2.4 overran the same
way — **on adversarial surface, never on line count**. Two new data: a frozen contract's stated
censuses need re-deriving at implementation time rather than trusting (A02), and a diff-blind suite's
documentation cases arrive pinned to invented phrasings, costing one adjudication round each.

### M2.4 — gate green, review adjudication closing

Contract at `.agent/archive/contract-m2u4.md`: **20 predicates P01-P20**, 4 invariant surfaces, gate
identity, an 8-class probe seed, and **35 amendments A01-A35** ruling `test-m2u4`'s phase-1
ambiguities over its **82 cases**. Baseline `6bbd50e`, whose gate read 1116 passed / 0 skipped.

**Gate green, MAIN-verified: `ruff`, `ruff format`, `ty` clean and `1199 passed` in 988.79 s, with
collected == passed and zero skipped, xfailed, xpassed, deselected or errored** — A32's
reconciliation, not a floor. Implementation, tolerance split, schema widening, golden regeneration,
the 82-case red suite and the corpus artifacts are all merged and green.

That 1199 was measured at `09e8dab`. Two post-review cases landed since, so the **decisive rerun
reads `ruff`, `ruff format`, `ty` clean and 1200 collected = 1200 passed, zero skipped, xfailed,
xpassed or errored** — outcome characters counted from the run and the count confirmed by an
independent `--collect-only`. `scripts/check_m2u4_suite_seed.py` still reports **82/82 encoded,
29 red, 53 control**: post-review cases carry no `kind:` marker, so the frozen census is untouched.
The final case merged after that run makes the next count 1201.

**Both reviews adjudicated.** `rev-m2u4-3`: 73 rows, 61 pass, 12 fail. `rev2-m2u4-2`: 33 rows —
23 mutants killed, M07 survived-unencoded, M20 survived with a credential now merged and green,
7 sweeps stable, D06 ruled `varied-contract-conflict` (a sweep over byte-distinct inputs cannot
demand an identical `source_sha256`; raw QC is byte-identical across three orders once that
mandated tag is excluded, alpha 300/300, beta 108/108).

Fixed in-unit: X12's label-fed oracle, A09's contiguous one-ULP control, X17's two over-cap M2.4
sentences, A34's documentation half, the uncovered `magnitude` guard, and M20's tolerance-read pin.
Deferred with acceptance checks: A21/X16, A34/X01/P07/A09-generalized, X10, X09, X14, X17's three
pre-existing sentences, and M07.

Not gated by this unit, and deliberately so: the generalized bound `k*q/S_retained` has no oracle
deriving `k` or `S_retained`. Its consumer is the grid residual, which `trajectory_grid_status()`
measures directly at 21,651/21,651 windows against 21,571/21,651 legacy — an unpinned inequality
whose only consumer is separately measured is not a gate.

`main=` 89% 215K/240K (implementation plus the full review harvest; the unit ran past the one-window
aim across a compaction boundary). `mate=` 100% 240K/240K (`rev-m2u4-3`, saturated at hand-off, so
its one re-review round on the in-unit fixes never ran — the fixes rest on demonstrated mutant kills
and the decisive gate instead). All six worktrees and every `wt/*` branch are removed.

**The defect is larger than a precision improvement — it decides whether the real corpus is
processable.** `1/median(diff(ts))` carries ~1e-3 relative bias against 4-decimal timestamps, and
`trajectory_grid_status` residual grows linearly with span under it: ~0.03 slots per second at
29.97 Hz, crossing `GRID_SLOT_TOLERANCE = 0.25` at ~8.3 s. `nominal_fs` holds ~0.002 flat. On real
decode timestamps (`scout-m2u4` pilot, 10 assets / 13 043 frames) `nominal_fs` places **610/610**
one-second windows on the grid against **530/610** for the biased estimator. The producer checks
windows, never whole clips, so that window figure is the one describing published rows.

**The finding that shaped the contract.** Adopting any estimator makes `gap_too_long` depend on that
estimator's residual, and `QC_POLICY_TOLERANCE = 1e-9` is sized for IEEE754 slack, four orders too
tight. At nominal 30 Hz the 3-slot gap verdict then cycles pass/pass/**FAIL** with clip length
mod 3 — a QC verdict with no physical meaning. Ruled (A10, A11): split the tolerance, keep `1e-9`
representation slack on coverage, add a `1e-4` estimator slack on the gap comparison alone, publish
it as `qc_policy_tolerance` so a consumer reproduces the verdict from the row. Two flips are
intended and pinned — 30 Hz 3-slot stays `pass`, and 60 Hz 6-slot moves `FAIL` → `pass`, which closes
the standing `gap6`/`gap7` polish row.

Scope: `analysis/clinical_features.R` is the only production surface, plus its goldens, the R suites,
`docs/technical/analysis.md` and the new `tests/test_r_timebase_truth.py` seed.
`src/pose_estimation/` timestamp production and rounding stay unchanged;
`analysis/data_extraction.R:100-112` and `analysis/arthrose_diag.R:77-100` divide angular change by
each rounded interval and go to `.agent/polish.md` rather than widening this unit.

**Corpus-scale evidence, 379 assets** (`scout-m2u4-2`). Grid placement under `nominal_fs`
passes **21 651/21 651** windows against **21 571/21 651** legacy, and every one of the 80 legacy
failures comes from a single 119.97 fps asset — the predicted shape, since the bias grows with frame
count per unit time. Every asset's `nominal_fs` residual is no worse than legacy's, so the swap costs
nothing anywhere. The container-header cross-check disagrees on **4/379**, worst 1.46938e-4, across
h264 and hevc and 3 of 4 device configs; A31 rules that a reported outlier, not a gate, because the
header divides `n_frames` by a duration counting the terminal frame while `nominal_fs` divides
`n_frames - 1` by the span that omits it — identical under constant frame rate, separating by
`(terminal_frame_duration - mean_interval) / span` under VFR.

**Next session's whole job.** Harvest the two live reviews, merge their red tests, adjudicate what
they filed, rerun the gate, set M2.4 DONE and commit. Both worktrees carry committed work.

| teammate | worktree | state at handoff | what remains |
| -------- | -------- | ---------------- | ------------ |
| `rev-m2u4-3` | `wt/rev-m2u4` | **40 of 73** rows adjudicated through batch 5 (`9483165`); P01-P20 + A01-A20 done, A13-A20 all pass on a 19/19 cadence lattice; gauge 78%, checkpoint directed | spawn `rev-m2u4-4` on the same worktree for A21-A35 + X01-X19; open `fail` rows are P07 + A09 (below), and P01 + P18 need rescoring because both were fixed after it scored them |
| `rev2-m2u4-2` | `wt/rev2-m2u4` | **25 of 33** mutants scored through `5236a28`, gauge 60% | finish 8 mutants, then merge its red tests as cases beyond the frozen 82 |

`rev2-m2u4-2`'s two accepted findings, both already carrying evidence. **M07 survived-unencoded**:
`return(NA_real_)` -> `return(0)` in `nominal_fs`'s final guard survives every case, and it is
probably pipeline-equivalent since both call sites guard `fs <= 0` — kill it with a unit assertion on
the documented `NA_real_` return, not a pipeline case. **M20 survived-encoded**: C5.18 passed when
both published tolerance reads were replaced by the literals `1e-4` and `1e-9`, because the producer
values equal them; it shipped `test_threshold_oracle_uses_published_tolerances`, green on MAIN and
red on the mutant.

**The one open judgment call.** A34's generalized bound `k * TIMESTAMP_QUANTUM / S_retained` is
stated in the contract and the roxygen, but no test derives `k` or `S_retained`, and A09 also wants
exact one-ULP rejection control. Decide whether that case lands in M2.4 or in `.agent/polish.md`.

Merge `rev2-m2u4-2`'s red tests as cases beyond the frozen 82: the 82 bind to the phase-1 table, and
a mutation-derived case has different provenance. Ruling A09/P07's open row is the one judgment call
left — decide whether a `k`/`S_retained` case lands in this unit or in `.agent/polish.md`.

**Wave-1 state, superseded by the table above; worktrees retained.**

| teammate | branch tip | delivered | open dependency |
| -------- | ---------- | --------- | --------------- |
| `map-m2u4` | (primary) | 12-unit surface map + normative checklist, `.scratch/agents/map-m2u4.md` | none, harvested |
| `scout-m2u4` | `bf09826` | `scripts/probe_timebase_grid.py` + 10-asset pilot; full 379-asset sweep unfinished | P20's sample + byte-identical rerun |
| `spike-m2u4-adopt` | `bc49910` | prototype adoption, U1-U3 filled | golden/QC-verdict blast-radius cells U4-U8 |
| `test-m2u4` | `3bf5f0f` | 80 candidate cases + 26 ambiguities, diff-blind | phase 2 = encode the ruled table as the red suite |
| `rev2-m2u4` | `1bdd315` | 25-mutant catalogue + 8 determinism sweeps, fixed pre-diff | phase 2 = run the campaign against MAIN's diff |
| `rev-m2u4` | `6bbd50e` | nothing committed; phase-1 work was transcript-only and is lost | re-dispatch from the contract |

**Two corrections to teammate output, recorded so they are not re-inherited.** `map-m2u4` U6 claimed
the 30 Hz 3-slot gap "still passes because the comparison is inclusive" — false for
`(n-1) mod 3 == 2`, and the contract's §3 sweep is the disproof. `map-m2u4` U11 rows 6-7 cite
`78352e1` and `2977cec` as prior art; both are this wave's own teammate branches, and **no prior
adoption attempt exists in project history**.

**Sizing, recorded for PLANNING.** One window bought the surface map, the real-corpus probe, the
prototype, the contract and its 26 rulings — and no implementation. M2.4 read as a small unit in the
M2 plan (two call sites, a golden regeneration) and is not one: the two-line estimator swap forces a
QC-policy change, a published-schema change, three version bumps, a golden regeneration and a test
oracle that currently encodes the defect it is meant to catch. **A unit that moves a shipped
threshold's semantics is a kernel unit whatever its line count** — the same lesson M2.1 and M2.3
recorded, arriving here through a different door.

### M2.3 — closed, and what it leaves standing

Contract frozen at `.agent/archive/contract-m2u3.md` — **39 predicates P01-P39**, 4 invariant surfaces, gate identity, an 8-class probe-corpus seed. Rulings R1-R10 are all ruled and **all closed**. **All 39 predicates are green**, P29 closing last in window 10. Per-window trajectory for windows 5-10 → `.agent/archive/m2u3-windows.md`; verdict tables → `.agent/archive/rulings-m2u3.md`. Retained worktrees: `wt/spike-m2u3-audio`, kept because its `_family_coverage` is the P38 oracle until R6's connectivity reconciliation polish row closes. Reports for every stopped teammate are preserved under `.scratch/agents/`.

**Sizing, recorded for PLANNING.** Ten windows against a one-window plan, on a user ruling that the unit run whole rather than be re-split. Window 1 alone (`main=` 87%) bought tooling, the metadata axis, two offset spikes, the cross-modality cross-check and the contract, and shipped one commit. **Size a `kernel` unit by its adversarial surface, not its line count**: 39 predicates over six evidence axes, four needing their own measurement pipeline, is a milestone's worth of contract wearing a unit's label. The suspension of the one-window aim was granted to this unit alone and does not carry forward.

**Settled by measurement** (redaction-safe aggregates, `.scratch/m2u3/*_agg.json`, from committed PyAV):

- **No camera intrinsics metadata exists anywhere in the corpus.** Every `mebx` key over all 1010 timed-metadata tracks: `video-orientation` (376 files), `live-photo-info` (376), `detected-face` + sub-keys (135), `segment-identifier` (123). Intrinsics can only come from a per-model prior — `iPad (5th generation)` fx ≈ 1873.3 px, `iPad Air 11-inch (M2)` fx ≈ 1553.2 px, 4:3→16:9 crop 1.08947×, readout/stabilisation factor unreported — or from self-calibration.
- **The cameras are 2 iPad models over 4 (model, OS) configurations, about 3 tablets, and the `above` and `left` labels were served by different tablets in two eras.** `right` = iPad(5)/16.7 on all 131 assets. Every 3-view family draws from 3 distinct configurations; every subject used exactly 3. Codec tracks device: h264 = iPad(5), hevc = iPad Air. 48 kHz audio = iPad Air/26.5 exactly, so 55 of 137 multi-view families mix audio sample rates.
- **Every canonical asset carries mono AAC audio**, so the audio route covers the whole corpus.
- **Sync QC is stratified by `(model, OS, sample_rate)`** (P29) — `pairs_qc.csv` publishes `stratum_a`/`stratum_b`, `assets_qc.csv` publishes `audio_rate_hz`, and `qualification.json` publishes a `pairs.sync_strata` census keyed on the two strata sorted. Measured over 379 assets: **4 configurations, each carrying exactly one rate** — iPad(5)/16.7→44100 (131), iPad(5)/16.6→44100 (125), Air-M2/26.5→48000 (66), Air-M2/18.1.1→44100 (57). **The rate component therefore adds no partition on this corpus**; it is published anyway, because a stratum that is assumed rather than measured cannot show when that stops being true. **Stratum medians are dominated by manual camera start times and are not a device-latency measurement** — the stratification makes a per-configuration constant *visible* and measures none, which is the whole reason exact iPad input-to-timestamp latency stays unbenchmarked.
- **Cross-view offsets are recoverable.** Audio: 210/246 pairs accepted, confidence ROC AUC 0.96083, 2 false positives per 100 held-out controls, 122/137 multi-view families graph-connected, **35/35 accepted three-view triangles close under one 33.4 ms frame (median 4.451 ms)**, full-corpus cold run 8.256 s. An independent visual motion-energy estimator sharing no code and no signal agrees with audio to **median 12.89 ms, 86.2% within one frame, 41.5% under 10 ms, on the 65 pairs both accepted** (p75 23.10, p95 50.72, max 74.8 ms), measured against the visual spike's corrected control-optimal, creationdate-independent gate at `f82a9a9`: 74/246 pairs, 26/137 families, 0/200 controls, closure on 9/52 families with |r| median 8.08 ms and max 34.12 ms. The pre-correction basis (67/246 gate) read median 10.86 ms / 88.3% / n=60 — **superseded, conclusion unchanged**. **The visual gate's 0/200 control result does not bound its gross-error rate**: of the 9 pairs it accepts and audio rejects, one disagrees by 87.4 s. That evidence bounds the **visual** estimator alone, so R6 gives it a veto and not a vote — audio estimates, the corroborator vetoes only where it cleared its own gate. Requiring agreement was priced and refused: it leaves 111/137 families unrecoverable and 2 closing triangles. The audio figures are unaffected. No drift term is needed: 0/132 qualified audio drifts move alignment by more than one frame.
- **Closure is blind to acoustic-path bias** — propagation delays form an exact cocycle around a triangle — so closure certifies self-consistency, never accuracy. The cross-modality number is the only accuracy statistic this corpus yields, and its magnitude matches the 6-9 ms acoustic bias expected at these camera separations.
- **Rolling shutter bounds every timing claim above and synchronisation never removes it** (P27). Neither iPad model publishes a readout time → the artifact carries a **sweep, not a value: 0–33.33 ms**, with Apple-mobile 1080p line-scan evidence of **12.4–30.9 ms** (37–93% of one 30 Hz frame period) named as a proxy from other devices, never as a measurement of these two iPads. Every closure and cross-modality figure here sits inside that sweep, so none of them shows sub-readout camera agreement. Calling this contribution negligible is prohibited in every document.
- **AAC priming is a measured 0 ms residual, not the predicted bias** (P28). Prediction was rate-dependent — 2112 samples = 47.891 ms @ 44.1 kHz, 44.000 ms @ 48 kHz → a raw untrimmed mixed-rate pair carries a fixed **3.891 ms** bias, and 55/137 multi-view families mix rates. The decode path cancels it: PyAV trims priming, skip = 2112 samples on 379/379 and first decoded PTS = 0 on 379/379. Quote the measured 0 ms; never quote 3.891 ms as a live bias.
- **7 assets change device orientation mid-clip**, which the single display matrix cv2 applies on decode cannot express; 3 assets carry no orientation track at all.
- `com.apple.quicktime.creationdate` is a **coarse sanity check, not an alignment prior**: whole-second, and residuals against measured offsets show multi-second per-tablet clock biases fitting neither a recording-start nor a file-finalize hypothesis.
- **123 assets carry GPS coordinates** (`location.ISO6709`, the iPad Air files). Values never read. Flagged to the user as a data-boundary matter.
- Device-side face metadata on 135 assets shows **7 assets ever holding more than one face**, so the one-subject assumption is not free.

**Ruled.** Verdict table → `.agent/archive/rulings-m2u3.md`; contract §7 carries the one-line summary. R1 view labels, R2 rigidity, R4 the 3D route, R5 the offset representation and A1 the axis wiring are all decided on measurement. R3 (metric scale reference) closed negative in window 6 — see the sampled-negative survey above.

- **Scene-feature extrinsics is eliminated, and M2.6's route is re-specified.** All 246 within-family pairs: **0 recoverable**, cross-view mutual SIFT matches median **13.5**, F-inliers median **8.0** — the algebraic minimum, so those inliers carry no evidence. Two controls make the null geometric rather than procedural: a baseline ladder falling 2812 (same frame) → 1252 → 962 (same asset, far) → **12–19** (cross-view) with 1740–3355 keypoints present per view, and the only 2 rich pairs of 244 being the `above|above` (298.5) and `left|left` (732.5) **view-conflict** pairs — correspondence returns exactly where two cameras share a viewpoint. **M2.6 recovers extrinsics from the subject's own keypoints**, where correspondence is assigned rather than matched.
- **P21's rigidity gate was unadjudicable and is replaced.** Its 4 px accept threshold also served as the MAGSAC inlier threshold, so `residual_p95` could never exceed the gate judging it. Across an 8× threshold sweep `residual_p95` tracks the threshold monotonically while inliers grow 6% → **no gate may be built on it**. `drift_median` moves 5.0% over that range. New gate `drift_p95 ≤ 20 px` = the reprojection tolerance already applied at `triangulation.py:423-424`: **280/298 assets pass, 71/137 families keep every member rigid** (was 76/286 and 3/137 under P21 as frozen). The 278/286 first recorded here was the 4 px instrument's population and is retired.
- **The view label is not a stable camera geometry.** `above` is 85% rigidity-unmeasurable on iPad(5)/16.6 and 3% on Air-M2/26.5, while the same iPad(5)/16.6 tablet's `left` is 6% — so view matters within a configuration and configuration matters within a view. **iPad(5)/16.6 `above` (89 assets, 23% of the corpus) is an unstable camera**: `valid_fraction` median 0.212 against 1.000 everywhere else, highest quiet-border motion energy in the corpus, `decode_status` ok on 89/89, and where 27 of the visual spike's 28 independent flags land. No per-view prior crosses the era boundary.
- **Detectability re-measured after a defect.** `detect_rate` median **1.0** (mean 0.989886, min 0.333333, n=379), 0 inference-failure frames, all four device configurations uniformly high. The prior 0.0 median was an artefact of rtmlib `PoseTracker` IoU-matching seconds-apart samples → `.agent/memory.md`.

**Coverage limits, carried forward:** assets with no rigidity verdict under any gate concentrate in the unstable cell above → `.agent/polish.md`. The 93/379 recorded here is the 4 px instrument's population and is **retired** — R2's amendment puts it at **81/379** (71 support-unmeasurable + 10 orientation-excluded). R3's sampled negative means every artifact permanently states arbitrary scale.

| unit | close | gate (passed / skipped) | `main=` | `mate=` |
| ---- | ----- | ----------------------- | ------- | ------- |
| — | baseline at M2 plan | 621 / 0 | — | — |
| M2.1 | `30280c3`..`6e363a0` | 734 / 0 | 98% 236K/240K, 3 windows | 100% 240K/240K |
| M2.2 | `d9f6c65`..`05fe55a` | 844 / 0 | 100% 240K/240K, 3 windows | 99% 238K/240K |
| M2.3 | `1ae599c`..`5e40922` + close | 1116 / 0 | 90% 216K/240K, 10 windows | 100% 240K/240K (M2.1 peak); 76% 183K/240K in window 10 |

**Sizing analogs** (unique files touched, summed churn; gauges where recorded). M3.2 `16e6fab` = 9 files, +891/−117, `main=95%` — the schema/identity analog for M2.1. M3.3a `a6218e5` = 13 files, +1694/−152, `main=58%` — a full artifact slice. Multi-camera fusion `62685e0` = 14 files, +1040/−164, and calibration `4d4df80` = 18 files, +1472/−156 — the integration band for M2.5/M2.6. Uncalibrated QA `20c36a0` = 14 files, +1225/−152 and adversarial failure modes `36f28a2` = 11 files, +981/−392 — the analogs for M2.7. **M3.3 was planned as one unit and did not fit one MAIN window**; M2.1/M2.2 are split at the same kind of boundary for the same reason.

**M2.1 actual, `30280c3` = 14 files, +4960/−22, `main=98%` across three windows.** It overran the one-window aim, and the overrun was not implementation churn — `inventory.py` is 1225 lines and the two suites are 3262 — it was the review loop: `rev-m2u1` returned 30 findings in phase 1 and 12 more in phase 2, each one costing a ruling, a fix, a contract amendment and a corpus rerun. **Size a `kernel` unit by its adversarial surface, not its line count.** A unit that publishes a durable artifact with a frozen digest pays for every predicate twice. M2.2 is the same shape and should be planned to close in one window only if its contract surface is materially smaller than M2.1's 31 amendments.

**M2.2 actual, `d9f6c65`..`05fe55a` = 16 files, +3006/−22, `main=100%` across three windows.** Its contract surface *was* smaller — 10 sections and 9 amendments against M2.1's 31 — and it overran anyway, for a different reason. `inventory` publishes a table; `sessions` publishes a **tree**, so every predicate has a filesystem failure mode behind it: a swap that fails, a sibling left under a dead pid, a symlinked `--out`, a corpus that moves between planning and linking, an output that contains its own input. Two reviewers filled 14 rows each in phase 1, then kept finding — nine more accepted defects arrived after both phase-2 markers, clustered on exactly those crash states plus the alphabets that keep hostile registry cells out of published names. **Budget a publishing unit by its crash states, not its contract sections**, and expect the review loop to outlive its own completion marker: an adversarial reviewer with context left is still the cheapest defect source in the wave.

**M2.1 → M2.2 handoff.** The census tool writes three artifacts to `inventory/`, self-verifying through `validate_generation()`: `assets.csv` (one row per discovered file — canonical corpus-relative path, disposition, reason code, SHA-256, container facts), `captures.csv` (one row per task-side family), `census.json` (redaction-safe aggregates plus a `generation` block digesting all three, the census entry being a digest of the census minus its own key). M2.2 reads `assets.csv`; it does not walk the corpus again.

- **`capture_id` names a task-side family, never a recording event** — `(subject, task, side)`, no take component. **The instance grain is ruled: `event_id = f"{capture_id}_run-{run_index:02d}"`**, so no event key can ever equal a family key and the standing "never bind calibration to `capture_id`" constraint becomes unrepresentable rather than merely documented. `run-<index>` is BIDS's own entity for an otherwise-identical repeated acquisition; prior art (Pose2Sim, OpenCap, Anipose, EasyMocap, FreeMoCap, MMPose) uniformly puts the recording event below any participant or visit grouping and never treats a semantic family as proof that clips are one event. `run_index` is **not chronological and asserts no provenance** — assignment order is the registry's `source_path` code-point order, which only makes it deterministic. Full contract → `.agent/archive/contract-m2u2.md`; MAIN's verdict table → `.agent/archive/rulings-m2u2.md`.
- **Take resolution for the 2 view-conflict families is ruled: none is asserted.** No published pipeline infers same-take membership from filename, file order, duration, frame count, or creation-time proximity — membership is declared at acquisition, and alignment is a separate later step off a decoded shared signal. Header facts separate neither conflict. So each asset of a conflicted family becomes its own single-camera run with `take_resolution = "unresolved"`, and its run count must never be read as a performance count. M2.3 may resolve them by decode. Unequal frame counts across views are compatible with one event after offset estimation, so **the frame-parity figures above are evidence of neither sameness nor difference**.
- Path text is re-decoded strictly as UTF-8 once at discovery, so classification, parsing, ordering and published text are a function of corpus bytes rather than of filesystem locale; a non-UTF-8 name keeps its surrogate form. M2.2's symlink names must come from that canonical column, never from a fresh directory walk.
- 379 canonical / 3 quarantined / 0 excluded. Quarantine is a **stem-grammar** verdict, not a readability one — all three files open and probe. M2.2 holds all three out as `quarantined_stem`; republish the registry to admit them.
- `census.json` is the one redaction-safe artifact: no filename, no path, no subject directory name, recognized extensions only. `assets.csv` and `captures.csv` are patient-adjacent; `inventory/` is gitignored, and so are `sessions/` and `sessions.*/`.
- **Two committed gates back every claim above and both rerun from committed state** — `scripts/run_inventory_mutations.py` (72 mutants, 71 killed, `M028` alone surviving as a ruled equivalent) and `scripts/check_inventory_determinism.py` (20 sweeps, 0 failures, plus 13 tamper classes the consumer boundary rejects by exception class). A predicate M2.2 adds to the registry earns a mutant in the same commit.

**M2.2 → M2.3 handoff.** `pose-estimation-sessions` publishes `sessions/` from `inventory/` and never walks the corpus: **193 events over 382 assets** — 58 one-camera, 84 two-camera, 51 three-camera; 186 `family` and 7 `unresolved`; 379 placed and 3 held out. Each event directory holds one `session.json` and one `cam-*` symlink per camera, `discover_sessions` returns all 193, and the tree regenerates byte-identically under a changed locale, hash seed, time zone and `--out` name. Every consumer calls `validate_generation(out, inventory_dir=…)` before reading a row: the two-argument form is the only check that catches a registry rebuilt under a tree which still looks internally consistent. Shipped surface → `docs/technical/sessions.md`; contract and rulings → `.agent/archive/contract-m2u2.md`, `.agent/archive/rulings-m2u2.md`.

- **The 7 unresolved families are M2.3's to resolve or to leave.** `take_resolution = "unresolved"` asserts no multi-camera event, so each of their assets is its own single-camera event and their run counts understate the true grouping. Nothing in the tree infers take membership from a filename, a file order, a duration, a frame count, or a creation time; M2.3's decode evidence is the first thing that could.
- **`sync_offset` is 0 and unmeasured on every camera, and no manifest declares `calibration`.** Both fields exist and assert nothing. **M2.5 does not fill `sync_offset`** — its contract P16 keeps the manifest field as the legacy integer pre-roll trim and publishes alignment as `qualification/cameras_qc.csv` instead, because writing a derived offset back into a registry-derived manifest closes a publisher cycle. M2.6 still fills `calibration`.
- **Nothing is decoded yet.** The generator asks the filesystem whether each listed path is a regular file and stops there, so container facts still come from M2.1's cv2 probe alone.
- **Two distinct orientation policies, and only the probe's is explicit.** `probe_container` sets `CAP_PROP_ORIENTATION_AUTO=1` and records `CAP_PROP_ORIENTATION_META` + `CAP_PROP_ORIENTATION_AUTO` (`src/pose_estimation/video_io.py:243-254`); every decode path relies on the backend default instead, which happens to be auto-rotate, so frames arrive upright by convention rather than by assertion. The hazard is that a rotated view has different image geometry from its siblings, not that frames are sideways. **7 assets change orientation mid-clip** (`[1,8]`, `[1,6,8]`×2, `[3,6]`, `[1,6]`×2, `[1,3,6]`) and 3 carry no orientation track at all — a single display matrix cannot express either case, so a per-asset rotation constant is wrong for 10 of 379.
- Decoder/tool matrix: **cv2 + PyAV** (`av>=17.1.0`, `pyproject.toml:35`; measured 18.1.0). `ffmpeg`, `ffprobe` and `exiftool` remain absent. PyAV supplies true PTS, creation timestamps, audio tracks and Apple `mebx` metadata; the metadata axis is already measured off it.

**Standing constraints.**

- **Capture identity has no schema home.** Producer keys are `video`/`person_idx`/`window`; `session.json` carries no task, side, or family field; `world3d.csv` reduces `video` to `session_id`. M2.1's registry is the single source of family identity, bound by `capture_id`. Legacy 2D and 3D producer schemas stay unwidened — `analysis/utils.R:59-87` treats every numeric non-metadata column as a feature.
- **Calibration identity is unbound.** Discovery accepts any calibration whose camera names match; nothing compares rig or session identity (`src/pose_estimation/multicam.py:364-383,579-588`). Per-recording-event calibration makes this a live hazard, so M2.6 must bind calibration to the instance grain M2.2 resolves. It may **never** bind calibration to `capture_id`: a `view_conflict` family holds more than one take, so that key does not name a recording event.
- **View labels are lexical, not geometric.** `above`/`left`/`right` are filename tokens. M2.3 verifies them against measured geometry before any calibration reuse; a mismatch projects pixels through the wrong camera.
- **Provisional QC thresholds.** `coverage ≥ 0.80`, `max_gap ≤ 0.10 s` are engineering defaults carried under `qc_policy_version`, not validated standards. M2.7 is where evidence replaces them.
- **One subject only.** Fusion reads `person_idx == 0`; cross-camera identity matching does not exist.
- **Decisive gate is primary-tree.** `renv/library/` is gitignored, so worktrees skip R cases unless symlinked; a green worktree run is no evidence for `analysis/*.R`.

**Acceptance:** every one of the 382 files reaches exactly one explicit outcome — canonical family member, quarantined stem, or recorded exclusion — with nothing silently dropped; the session tree regenerates byte-identical from a clean base and `--list-sessions` enumerates it; every corpus claim traces to a committed rerunnable command rather than a scratch script; the claim boundary above is honored in every artifact and document; full suite passes in the primary tree with 0 skips.

## M3 — analysis-ready 3D aggregation

**Status: DESCOPED** — terminal, never a dispatch target; reviving any part is a PLANNING ruling. M3.1, M3.2 and M3.3a shipped and stay in the tree with their gates. M3.3b and M3.4-M3.6 are cut: real data replaced the synthetic development surface, and no remaining unit is forced by it. `clinical_3d_video_aggregate.csv` was never built and nothing references it, so the cut falsifies no shipped claim. **Range** `b1f5b81`..`429b0f4`, descope ruling in `89b4fdd`. **Gauge band** `main=` 58% (M3.3a) to 95% (M3.2); M3.1 closed before gauges were recorded.

What survives in the tree: the timestamp-aware trajectory kernel (`zoo` dropped), the 3D producer identity schema, and `<stem>_clinical_3d_window_qc.csv` over the four trajectory groups. The QC artifact explains 12 trajectory metrics and is silent on `bilateral_*`, `trunk`, `shoulders`, `cpi` — `docs/technical/analysis.md` *Current scope* now states that as standing scope.

The cut released M2.4: M3's "2D goldens byte-identical" acceptance was the only reason `nominal_fs()` shipped unadopted, and re-deriving `output/rtmw-l_body_single/` stopped mattering when `videos/initial/` was retired. Full record, including the retained unmerged branches and the frozen M3.3 contract → `.agent/archive/m3.md`, `.agent/archive/contract-m3u3.md`.

## Produced datasets

- `output/rtmw-l_body_single/` — **preliminary**, from the retired `videos/initial/` clips. 12 single-camera clips, RTMW-L / `--tracking body` / `--single-subject`, det-CPU + pose-NPU; 15 430 rows over 15 455 frames, 99.7% mean coverage, 100% body-wrist observation. Kept on disk, not regenerated. Its clinical features predate both the M3.1 gap fix and M2.4's cadence fix, so **any normalized-jerk or velocity figure from it is suspect** — recompute before citing it anywhere, including a paper's preliminary-work section. Schema conformance and coverage figures are unaffected and remain quotable.

## Backlog

Scope seed for the milestone after M2.

- **Clinical join surface** — the eventual destination for M2's numerical output: the hospital SCI database (`database/ALL_SCIDATA.csv` + `SCI_DATABASE_HEADER.xlsx`), currently analyzed as a dashboard in `Projects/rehab/`. **The subject↔patient mapping is unknown and has to be established first**; nothing in `videos/3-cam/` identifies a database record. Needs capture/session metadata, then a capture→assessment bridge with instrument/version/domain/side/status and cardinality safety. ISNCSCI is side/myotome-resolved while SCIM is whole-person, so the join grain cannot be settled against synthetic data.
- Cross-camera identity matching for multi-person scenes; fusion assumes one subject.
- Gap-aware movement-phase metrics — M3.1's kernel covers frame/window scope only; `analysis/clinical_features.R:918-1097` phase speed/path/SAL/NJ/efficiency stay gap-unsafe and explicitly unqualified.
- Prospective calibrated capture — the only route past M2's claim boundary. M2.7 specifies it; running it is a separate milestone with its own clearance and ethics footprint.
