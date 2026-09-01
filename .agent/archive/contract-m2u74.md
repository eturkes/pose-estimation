# Contract — M2.7.4, prospective capture specification

Tier `docs` ⇒ assurance = `doc` + consistency pass, and the pass ships as a committed checker
rather than as one session's reading (M2.7.3 precedent). Baseline `bfdf1ba`.

**Product.** `docs/prospective_capture.md` — a 20-section normative specification for the future
calibrated acquisition that can reopen 3D. Specified, never run.

**Gate identity.** `env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync` over
`ruff check` · `ruff format --check` · `ty check` · `pytest`. Decisive suite = primary tree.
Unit validator = `python scripts/check_prospective_capture.py`, driven identically from
`tests/test_prospective_capture.py` through `runpy`.

## 1. Design decisions

**D01 — the spec ships BESIDE `docs/capture_protocol.md` and never patches it.** Ruled on the two
documents' scopes, not on taste. `capture_protocol.md` is the operating procedure for *this repo's
shipped tooling*: `pose-estimation-calibrate board|capture|solve`, `pose-estimation-validate
--qa-only`, `charuco.py`'s 6×9 / `DICT_4X4_250` / 40 mm board, `MIN_SHARED_FRAMES`, and an extrinsic
topology that solves direct pairs against one world camera. The prospective specification requires
what that tooling does not support — 8 cameras, 120 fps, global shutter, a wired trigger, traceable
scale carriers and sealed held-out check targets. Merging them would make a shipped operational
document state requirements its own commands cannot meet. Same idiom as F1a publishing beside
`qualification/` rather than patching it.

**D02 — each document names the other, and says which capture it governs.** Two overlapping
normative capture documents drift, and the drift is invisible when nothing references either. That
is measured, not feared: M2.7.3 found `docs/technical/calibration_qc.md` had already drifted on 6 of
15 claims while no test or script referenced it. P10 pins the banner in both directions.

**D03 — 20 sections, fixed ids `S01`-`S20`, fixed titles.** The spine is the research report's Q8
inventory adopted whole. Each section carries five fields — **Owner**, **Record**, **Threshold**,
**Failure action**, **Evidence**. A section missing one cannot be operated: a threshold with no
failure action is advice, and a record with no owner is nobody's.

**D04 — every threshold resting on a measured absence is labelled a local decision.** Five absences
are measured over a 40-call search budget, and each one sits under a number this spec must still
choose. The spec chooses, and says it chose. Presenting a project choice as a standard is the one
defect a `docs`-tier unit can ship that no downstream consumer re-checks.

**D05 — the spec asserts nothing about the M2 corpus beyond the published claim set.** It specifies a
future acquisition. It may state that this acquisition is the route that could reopen 3D; it may not
state that running it will. `calibration_qc.PROHIBITED_PARAPHRASES` under `_fold` stays total over
the document, with no excluded span — the M2.7.2/M2.7.3 ruling, reached the same way both times.

**D06 — never run, and the text may not read as though it were.** No command is reported as having
been executed, no result is reported, and the unrun status is stated verbatim once.

**D07 — human register.** A specification is a surface a person reads at consumption time ⇒
ASD-STE100, and registration in the `docs/technical/conventions.md` text-register inventory is what
puts it under that rule. Unregistered defaults to the agent register, which is a silent register
change rather than a missing line.

**D08 — every external citation carries a resolvable identifier.** URL or DOI, verified by
`res-m2u74-2` before MAIN ships it as authority. Teammate research is attention-directing; a
citation MAIN acts on is a citation MAIN validated.

## 2. Section spine (D03)

| id | title | binds |
| -- | ----- | ----- |
| S01 | Scope, estimands and claim boundary | L1 |
| S02 | Document control and responsibilities | |
| S03 | Study design, population and sampling | |
| S04 | Ethics, consent and identifiable-video governance | **N5**, L4 |
| S05 | Task script, safety and trial schedule | |
| S06 | Room, lighting and scene controls | |
| S07 | Hardware and software inventory | |
| S08 | Camera layout and visibility proof | |
| S09 | Sensor mode and image-quality qualification | |
| S10 | Mechanical mounting, orientation and drift epochs | **N3** |
| S11 | Intrinsic calibration | **N1**, L2, L3 |
| S12 | Extrinsic calibration and coordinate frame | **N1**, L2 |
| S13 | Metric-scale traceability | **N4** |
| S14 | Synchronization and rolling-shutter model | **N2**, L5 |
| S15 | Session, trial and provenance manifest | |
| S16 | Preflight, capture and postflight disposition | |
| S17 | Independent reference and validation acquisition | |
| S18 | Processing and model contract | |
| S19 | Acceptance statistics, uncertainty and exclusions | |
| S20 | Security, release, reproducibility and change control | **N5** |

**Five non-negotiables (N1-N5).** N1 intrinsic + extrinsic calibration (S11, S12) · N2
synchronization residuals (S14) · N3 orientation and drift control (S10) · N4 traceable metric scale
(S13) · N5 identifiable-video governance (S04, S20).

**Five measured absences (L1-L5), each a labelled local decision.** L1 no markerless-specific
reporting checklist ⇒ the section spine itself is synthesis · L2 no universal clinical reprojection
threshold · L3 no sourced board-to-volume ratio · L4 no generic Japanese retention period for
non-invasive research video · L5 no published millisecond synchronization figure for OpenCap,
Pose2Sim or Anipose.

## 3. Predicates

| id | predicate |
| -- | --------- |
| P01 | All 20 sections present, ids `S01`-`S20`, contract titles verbatim, in ascending order. |
| P02 | Every section carries **Owner**, **Record**, **Threshold**, **Failure action**, **Evidence**, each non-empty. |
| P03 | Each of N1-N5 names its contract section, and every bound section states an obligation with `MUST`. |
| P04 | No `calibration_qc.PROHIBITED_PARAPHRASES` entry under `_fold`, over the spec and the `capture_protocol.md` banner. Total, no excluded span. |
| P05 | All five absences L1-L5 appear, each inside a span labelled a local decision; at least 5 labels. |
| P06 | Every citation carries a URL or DOI; non-empty floor; no citation left bare. |
| P07 | Backticked repo paths and Markdown link targets all resolve; non-empty floor on both kinds. |
| P08 | No capture id, corpus path segment or media filename. |
| P09 | `docs/prospective_capture.md` listed in the `docs/technical/conventions.md` text-register inventory. |
| P10 | Scope banner both ways: the spec names `docs/capture_protocol.md`, and `capture_protocol.md` names the spec, each with its governing-scope sentence. |
| P11 | No execution claim under `_fold`; the unrun status stated verbatim once. |
| P12 | `tests/test_prospective_capture.py` drives this checker. |
| P13 | Every prose sentence outside fences, tables and headings is ≤ 25 words (ASD-STE100 description bound). |

**Vacuity rule, standing.** Every set-quantified predicate carries a non-empty floor, and its detail
line reports the count it ranged over. A green predicate whose detail line reports zero items is a
failing predicate — M2.7.3's P07 shipped that defect and it was caught by reading output, not rc.

## 4. Invariant surfaces

1. `docs/capture_protocol.md` — content unchanged except the added scope banner. It remains the
   procedure for the shipped tooling.
2. `src/pose_estimation/calibration_qc.py` — read-only. The claim set and the prohibited-paraphrase
   list are consumed, never edited.
3. Published trees (`inventory/`, `sessions/`, `qualification/`, `calibration_qc/`) — untouched. This
   unit publishes no artifact and computes nothing.
4. Suite collection moves by exactly the new test file.

## 5. Negative-control seed

Each seeded, observed to fire, and reverted.

| id | injection | must fire |
| -- | --------- | --------- |
| NC1 | Drop one section heading | P01 |
| NC2 | Blank one section's **Failure action** | P02 |
| NC3 | Remove `MUST` from an N-bound section | P03 |
| NC4 | Insert a prohibited paraphrase | P04 |
| NC5 | Delete one local-decision label | P05 |
| NC6 | Strip the identifier from one citation | P06 |
| NC7 | Point a backticked path at a non-existent file | P07 |
| NC8 | Add an execution claim in past tense | P11 |
| NC9 | Extend one prose sentence past 25 words | P13 |

## 6. Amendments

**A01 — P04 is scoped to `docs/prospective_capture.md` alone, correcting §3 as frozen.** The frozen
predicate ranged over "the spec and the `capture_protocol.md` banner". Measured at implementation:
`capture_protocol.md` already carries *"This clinical-validity gap stays open until a gravity
reference exists"*, and `_fold` flattens `-` to a space, so `clinical-validity` reads as the
prohibited needle `clinical validity`. That sentence concedes a gap; it is the opposite of the
overreach the needle exists to catch. `PROHIBITED_PARAPHRASES` was written for `calibration_qc`'s
claim surface, and ranging it over a document written to a different contract buys a false positive
rather than coverage. `capture_protocol.md` is an invariant surface (§4.1), so the alternative —
rewording a shipped human-facing document to satisfy a rule it was never written to — is scope creep
on a DONE unit. The banner is pinned by P10 instead, which checks its content directly.

**A02 — NC1 trips P02 as well as P01, and that coupling is correct.** Dropping a section heading
orphans its five fields, so the section-spine predicate and the field predicate both fail. Recorded
so a later run does not read the second failure as a defect in the control.

**A03 — S11 shipped with no `MUST` and P03 caught it.** The section was written entirely in
imperatives, which the register prefers, but N1 binds S11 and a non-negotiable needs definite
modality. Repaired at implementation: the target rigidity and the pose count now state `MUST`. This
is the predicate earning its place — the defect is invisible to a reading pass, because imperative
prose reads as obligation to a human and states none to a checker.

## 7. Verdict table

| id | verdict | evidence |
| -- | ------- | -------- |
| P01 | pass | 20 sections present, titles verbatim, ascending |
| P02 | pass | 20 sections x 5 fields, all present and non-empty |
| P03 | pass | 5 non-negotiables bound to 7 sections, all stating `MUST` (A03) |
| P04 | pass | 0 of 23 prohibited paraphrases, total over the spec, no excluded span (A01) |
| P05 | pass | 5 absences stated, 13 local-decision labels |
| P06 | pass | 15 citations, every one carrying a URL or DOI, none listed-but-uncited |
| P07 | pass | 1 backticked repo path, resolves; prospective artifact names deliberately outside `PATH_ROOTS` |
| P08 | pass | no capture id, corpus path segment or media filename |
| P09 | pass | registered in the `conventions.md` human-facing inventory |
| P10 | pass | both documents carry the banner and name each other |
| P11 | pass | 0 of 10 execution claims; unrun status stated exactly once |
| P12 | pass | `tests/test_prospective_capture.py` drives the checker through `runpy` |
| P13 | pass | 266 prose sentences, all within 25 words |

**Negative controls: 9 of 9 fire**, each naming its own predicate, document restored byte-identical
(`.scratch/nc_m2u74.py`, baseline and post-restore both clean). NC1→P01 (+P02, A02) · NC2→P02 ·
NC3→P03 · NC4→P04 · NC5→P05 · NC6→P06 · NC7→P07 · NC8→P11 · NC9→P13.

**Dispatch inputs for MILESTONE-REVIEW.** Contract = this file. Verdict table = above. No `test`,
`orc` or `diff` branch tips: tier `docs` runs no oracle, and the unit's whole product is one
document plus the committed checker that grades it.
