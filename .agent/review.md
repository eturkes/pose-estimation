# Review ledger

Adjudication record per closing diff; a ruling holds until new evidence reverses it. Reports =
`.scratch/agents/<name>.md` (gitignored), graded by `scripts/check_review_report.py`.

## OpenVINO 2026.4.1 adoption (archive build + `.venv` wheel)

Check set R01-R10 fixed at seed; `reviewer-5` + `reviewer-6` blind to each other. Verdict =
first pass → re-review of the accepted fix against its acceptance check.

| ID | check | reviewer-5 | reviewer-6 | ruling |
| --- | --- | --- | --- | --- |
| R01 | `uv.lock`: 3.13 resolution moves `openvino` alone; other deltas explained | pass | pass | pass — onnxruntime py<3.11 fork 1.24.3 had no cp310 wheel → 1.23.2 |
| R02 | `uv lock --check` + `uv sync --dry-run` clean | pass | pass | pass |
| R03 | `gates.md` 4-way prefix table reproduces | pass | pass | pass |
| R04 | `gates.md` `.scratch/ovupgrade/` record matches the scripts | pass | pass | pass |
| R05 | detector figures ≤ their evidence's precision | fail → pass | fail → pass | fixed — `≤ 3e-6` from a 6-dp print → `< 3.5e-6`; box Δ frame named |
| R06 | build-drift figures re-derive | fail → pass | fail → pass | fixed — one quantum → 576 coords at 6 dp + 17 confidences at 4 dp |
| R07 | run side loads 2026.4.1 on CPU/GPU/NPU; placement per run log | pass | pass | pass |
| R08 | resume note accurate | pass | pass | pass |
| R09 | `deferred.md` rows carry failing acceptance checks | pass | pass | pass |
| R10 | no live surface keeps a contradicted claim | fail → pass | pass | fail ruled (anchor rematched) → fixed — `rtmlib_openvino.py` comment said "exactly … score delta 0" |

Register, ruled: pilot census 4 → 5 events in live law (selection code + publishers predate the
sweep; frozen archive keeps 4); Clause 1 count 7 → 8 (`tests/test_rev4_m2u95.py`); carried
41.6/318.7 ms timings labelled as an earlier build's.

## Corpus rerun downstream + review-UI layout (b8ccd71, aed1e7d + closing commit)

Check set R01-R12 fixed at seed; `reviewer-1` alone, report `.scratch/agents/reviewer-1/rev.md`.
Verdict = first pass → re-review of the accepted fix against its acceptance check (F rows).

| ID | check | reviewer-1 | ruling |
| --- | --- | --- | --- |
| R01 | drift: no self-sized canvas widens a track at any width/ratio | pass | pass |
| R02 | ratchet: `fitStage`'s room independent of the stage in both layouts | pass | pass |
| R03 | layout: stage + transport on screen, list usable, README true | fail → pass | fixed — claims narrowed to the measured domain (`.scratch/floor_probe.mjs`: 550 px tall side by side, 700 px stacked) |
| R04 | retired `--fit-viewport`/68dvh/40dvh leaves no dead code or text | pass | pass |
| R05 | `gates.md` QA records match script sources | fail → pass | fixed — 48/48 after-only includes the 4 seed rows |
| R06 | catalog § M2.9 figures traced, inferences labelled | fail → pass | fixed — P08 = point-side comparisons (4/frame); blink attribution labelled inference, D02 counter row decides |
| R07 | `corpus-run.md` measured-whole = `run_report.json` | pass | pass |
| R08 | `cohort.md` + spec cohort line = published cohort | fail → pass | fixed — 10 `trunk_*` + compensatory = 11 |
| R09 | new `deferred.md` rows fallible + decidable, no duplicates | pass | pass |
| R10 | spec `Artifacts` current | pass | pass |
| R11 | data boundary: no identifier, ordinal or media committed | pass | pass |
| R12 | commit bodies consistent; subject format | pass | pass |
Register, ruled: N01 blink row gains a per-clip hand-presence floor (all-absent candidate fails);
N02 drift/ratchet preamble names the mechanism, ratio-1 rows kept as controls.

## ITERATE close — review UI (`17e8c1e..HEAD`)

Lenses fixed before reading: R01 every `Artifacts` prototype entry runs by its recorded command ·
R02 `Decisions` hold every UI ruling, readable without the prototype · R03 each stub, shortcut +
hard-coded value reads as one (`reviewer-9`, S01-S10, `.scratch/agents/reviewer-9.md`) · R04
`spec.md` matches what shipped + passes `.agent/spec-lint`.

| ID | check | verdict | ruling |
| --- | --- | --- | --- |
| R01 | `:8791` + `:8792` comparison run; 2/2 committed + 7 scratch QA suites, glyph check | pass | pass |
| R02 | ITERATE UI rulings in `Decisions` | fail | fixed — `Review UI shape` bullet (6 rulings from commit bodies) |
| R04 | `spec.md` current + lint | fail | fixed — Intent, Artifacts, Decisions, Tasks cleared, Phase set; lint rc 0 |
| S01 S03 S05 S07 S10 | clip list, census, cohort figures, copy, prototype badge | pass | pass |
| S02 | null fps / rotation shown as `0.000 fps` / `0°` | fail | fixed — `—`; red witness green |
| S04 | manifest census hides zero dispositions unstated | fail | fixed — zero-hidden note; red witness green |
| S06 | "every distance and velocity a lower bound" — false under perspective | fail | fixed — UI ja/en + `docs/technical/cohort.md:82` |
| S08 | README overlay parity + pipeline dot colour | fail | fixed — darkened limbs, splines, yellow dots stated |
| S09 | player silent on absent trees + missing media | fail | fixed — absent-tree banner, `player.no_video`; red witness green |

Red witness = reviewer-9's capture-free script against `0a876d8` (red, rc 1) → fixed tree: target
rows green; its reason-note positive control reads 2, the S04 note being the second.


## M4.6 + M4.3a + calibration_qc relabel (uncommitted diff vs `487f790`)

Check set R01-R10 fixed at seed; `reviewer-10` (report `.scratch/agents/reviewer-10.md`, red tests
`.scratch/agents/reviewer-10/`). Verdict = first pass → MAIN's rerun of the reviewer's reds on the
fixed tree.

| ID | check | reviewer-10 | ruling |
| --- | --- | --- | --- |
| R01 | M4.6 D01 exported-stream gate | fail → pass | fixed — a malformed decoded frame now resets the subject stream like any unexported frame (`run.py` skip branch); reviewer red green |
| R02 | M4.6 D02 export hygiene + counter | pass | pass |
| R03 | M4.6 D03 duplicate census | pass | pass |
| R04 | build + calibration in the generation; resume; report allowlist | pass | pass |
| R05 | registry names, alias, devices | pass | pass |
| R06 | Curve / PAV / apply | fail → pass | fixed — non-finite knots refused; ±inf scores clamp to the curve ends, NaN stays NaN; new test |
| R07 | `--pose-device` default resolution | fail → pass | fixed — MediaPipe delegation defaults NPU before the registry branch; reviewer red green + new test |
| R08 | tests pin predicates through shipped symbols | fail → pass | fixed — `fit` itself now graded against brute-force isotonic regression at every distinct score; kills the reviewer's constant-curve mutant |
| R09 | calibration_qc relabel text-only | pass | pass |
| R10 | data + claim boundary | pass | pass |

Register, ruled: `bench/replay.py` forced `--pose-device=NPU` → removed, device follows the
registry; `rtmlib-runtime.md` still said the build is not in the config → updated; M4.3a P04
"byte-identical CSV + diagnostics" → CSV bytes stay pinned by the existing M2.10.1 baseline-bytes
case, diagnostics differ only by M4.6's two census fields (and latency, which never repeats) —
contract A02; M4.6 P07 (stage attribution on real data) = MAIN-owned, open until its replay runs.

## IMPLEMENT closing review (301bd38..56b30c4 + 0bc4094)

Lenses fixed at seed: `reviewer-11` correctness C01-C12, `reviewer-12` cleanup U01-U12,
`reviewer-13` altitude A01-A12 (reports `.scratch/agents/reviewer-1{1,2,3}.md`, grader rc 0;
red batteries `.scratch/reviewer-1{1,2,3}/`); MAIN security lens S01. 36 rows + S01 → 15
pass rows, 21 fail rows; duplicates merged below. Verdict = first pass → re-review of each
accepted fix against its acceptance check.

| ID | finding | rows | ruling | fix |
| --- | --- | --- | --- | --- |
| F01 | repeated-image bootstrap broke tied scores unlike COCOeval | C04 | accepted | detections repeat as blocks, copies adjacent; check.py + 2 tied rows (old code: 2 FAIL) |
| F02 | confirm E1 capped observed errors at the miss value | C05a | accepted | cap removed; test |
| F03 | confirm E3 pooled sides instead of frame → subject | C05b, C12 | accepted | per-frame share then subject mean; test expectation 2/3 → 3/4 |
| F04 | pilot G4 admitted frames with one exported hand | C05c | accepted | both hands exported; test |
| F05 | confirm consumed the label iterator on the first arm | C06a, A06.3, U-R01 | accepted | `list(labels.load)`; CLI tests |
| F06 | `--from-run` presence / pilot-eval wrote into missing dirs | C06b, A06.3 | accepted | mkdir; CLI test |
| F07 | stdout + reports carried run-tree paths | C06c, U06, A06.2 | accepted | arm names only; CLI test |
| F08 | judgment logic (consensus, calibration targets, pre-roll equivalence) lived in the driver | A06.1 | accepted | moved to `bench.labels.consensus`, `bench.screen.calibration_targets`, `bench.replay.read_point_delta` |
| F09 | zero quantile + floor dropped clean sparse motion | C07 | accepted | level falls back to threshold × series max; test; qualification rerun |
| F10 | noise-proxy floor counted pre-selection stencils | C08 | accepted | floor counts selected centres; test; measure rerun (unchanged) |
| F11 | noise reader diverged from R masking on non-finite cells | A08 | accepted | `bout_qualification.read_wrist` (src) + R differential test |
| F12 | cohort bout reader: no key / bounds check | C09a/b, A09 | accepted | `BOUT_KEYS` required; finite ordered bounds; 14 tests |
| F13 | vectorised ICC resamples cancelled catastrophically at large offsets | C09c, U-R02 | accepted | sums centred; offset test |
| F14 | report stamped the on-disk R digest on untouched outputs | C10a, A10 | accepted | marker records `analysis_sha256` per event; report = the recorded one or `mixed`; verdict `clinical_method_uniform`; test |
| F15 | an interrupted refresh left rewritten files under a complete marker | C10b | accepted | marker failed before R; test |
| F16 | A32 missed collection-time skips | C11 | accepted | `pytest_collectreport`; C8.08 case (old plugin: red) |
| F17 | bench + runtime named RTMW-X twice; calibrate wrote an unused file | A03, U03 | accepted | bench arms derive from `run.MODEL_REGISTRY`; caches renamed; test |
| F18 | `population.features` counted declared, not published | U09 | accepted | published count; test |
| F19 | clinical stage duplicated in attempt + refresh | U10 | accepted | one `_clinical_stage` |
| F20 | DexYCB joint test used its own table as oracle | U12 | accepted | named join to the codebook; test |
| F21 | driver-report build test exercised the helper only | A12 | accepted | both drivers' reports asserted |
| F22 | development-lookup control absent from P08 | A-R02 | accepted | lookup detector fails admissibility; test |
| S01 | review UI stored a non-finite click as a visible point | S01 | accepted | stored `?` (0bc4094) |

Re-review (`reviewer-14`, fresh context; `.scratch/agents/reviewer-14.md`, grader rc 0): 21 pass,
2 fail → fixed again and closed on MAIN's rerun of the reviewer's own probes:
F07 (major) — replay still printed the runner's source + output paths → `replay_window` sends
the runner's stdout + stderr to the window's `replay.log`; the probe's two commands pass, ported
to `tests/test_m4u4_bench_cli.py` with the F05 three-arm case. F20 (minor) — the name join
covered 5 slots → the oracle is mmpose's `coco_wholebody_hand` list for all 21; all 210 pairwise
exchanges fail it. Register: R01 unclosed diagnostics stream → `_diag_row` closes it; R02 the
corpus-run rule said a source edit makes a tree `mixed` → corrected (stale ≠ mixed).
Full gate on 71fe581: 2719 passed / 5 failed — four test fixtures predating `--refresh-clinical`
and the per-event digest (namespaces + marker) and M2.9.4's SPARC label case pinned to the
retired window column → fixtures given the new field; the label case now pins `*_bout_sparc`
(old expectation fails on v2 code, the recorded firing).

## IMPLEMENT final closing review (301bd38..eccbe39) + MAIN security lens

Lenses fixed at seed (`.scratch/impl/seed_reviews.py`, seeds graded `FAIL P03` both ways):
`reviewer-15` correctness/spec K01-K12 (code `f4667db..eccbe39`, contracts whole phase; F01-F22
stand), `reviewer-16` claim soundness N01-N13, `reviewer-17` guarantee-vs-claim G01-G13,
`reviewer-18` verification integrity V01-V13, `reviewer-19` `CLAUDE.md` conformance M01-M13
(reports `.scratch/agents/reviewer-1{5..9}.md`, grader PASS rc 0 each); MAIN security S02-S12
(`.scratch/agents/security-main.md`). 64 reviewer rows: 34 pass, 30 fail → 24 distinct findings.
Verdict = first pass → MAIN's rerun of the reviewer's red (or the ported test) → re-review of each
fix against its acceptance check.

| ID | finding | rows | ruling | fix | state |
| --- | --- | --- | --- | --- | --- |
| F23 | `labels.validate` never checked the frame object's keys | K06a G02a G09a | accepted | named; parametrized row | fixed |
| F24 | `screen` ranked arms outside the registry (A04-dropped caches → finalist) | K07a K12 | accepted | refused before ranking; test | fixed |
| F25 | whole-clip replay skipped the production camera survey → posed trimmed setup frames | K08a | accepted | `CORPUS_ARGS` + `--camera-survey`; test | fixed |
| F26 | an A01 heading naming nobody passed the candidate freeze | K09a | accepted | refusal before labels load; test | fixed |
| F27 | P07: "bone-length adds on 2 clips" vs the table's 3 of 6 | K11a N10c N12b N13a | accepted | 3 of 6 in pose-weak-points + spec | fixed |
| F28 | `score_calibration` promised P(correct); shipped reference maps emit raw-scale scores > 1 | G01a | accepted | docstring names both fits | fixed |
| F29 | DexYCB bootstrap resampled zero-evidence hands → ÷0 on a sparse draw | G03a | accepted | units = hands with a valid joint; equal coverage enforced; test; table re-derived (Δ ≤ 0.001) | fixed |
| F30 | `--from-run` scored markerless / failed / other-generation events | G04a G04b | accepted | complete marker + recorded model + calibration digest; tests | fixed |
| F31 | pilot replay cache survived a hand-gate refit | G04c | accepted | cache keyed by replay args + calibration digest; test | fixed |
| F32 | review-UI README: global "writes nothing" / "read-only" vs `/label` | G11a V11a M11b | accepted | scoped (`9417b9e`) | fixed |
| F33 | "no finite pre-roll" for a 0-8 s grid | N04a N10f | accepted | "no tested pre-roll (0-8 s)" in CLI + benchmark.md | fixed |
| F34 | per-class state qualification claimed from pooled field intervals | N09a N10a N13b | accepted | `bench.py agree` prints per-class lower bounds; benchmark.md table; test | fixed |
| F35 | M4.3 pilot "8 subjects" = main quota; combined = 14 | N09b | accepted | benchmark.md + contract-m4u3 A02: main 8, stress 12, pilot 14 | fixed |
| F36 | consensus coordinates "qualified at 0.3" from inter-labeller precision | N10b V10a V12 | accepted | consensus vs GT measured (`public-score --consensus`); user ruling: G2 at 0.6 (contract-m4u3 A02) | fixed |
| F37 | judge "near chance / mostly wrong" vs 90/145 decisive correct | N10d N12b | accepted | decisive 90/145, no bin at the bar (benchmark.md, bench.md) | fixed |
| F38 | cross-view "not a population artifact" from a count restriction | N10e N13c | accepted | count held, support not: composition unmeasured | fixed |
| F39 | README: "No method can calibrate these cameras" | N11a M11a | accepted | README: the tested methods, a calibrated capture can change it | fixed |
| F40 | README: M4.3/M4.4 confirmation stated as done | N11b | accepted | README states what is measured; final wording at close | fixed |
| F41 | `cohort.md` 88 / 1056 stated as every publication's census | N12a | accepted | cohort.md: declares 88; published count = `population.features` | fixed |
| F42 | calibration P01: 5 % bound + fitted-model oracle vs the frozen 2 % | V01a V08 V09 | accepted | generating-model oracle at 2 % (1e6/arm: ≤ 0.7 % over 6 seed pairs; kills the 4 % slope mutant) | fixed |
| F43 | cocowb self-check passed a scorer missing its wholebody pass | V03a | accepted | exact six-part set | fixed |
| F44 | censored-fragment test passed with zero fragments | V05a V08 | accepted | exact `[True, True]`; kills the row-drop mutant | fixed |
| F45 | "run under the gate prefix" covered inference commands | M04a M10 M12 | accepted | split by command kind (CLI, benchmark.md, spec) | fixed |
| F46 | subproject `uv run` inherited `UV_PROJECT_ENVIRONMENT` → the primary `.venv` replaced (MAIN, live) | — | accepted | unset in the gate step + `bench._evaluate`; test; `gates.md` | fixed |
| S02-S06 | review-UI request boundary | S | accepted | `9417b9e` (`security_qa.py` 2/9 → 9/9) | fixed |
| S12 | report allowlist admitted a marker's `analysis_sha256` verbatim | G-R05 | accepted | 64-hex shape, else `mixed`; test | fixed |
| M03a | `probe_crossview_arm --cache KEY` echoed into report keys | M03 | rejected | operator-typed CLI text = trusted (withdrawn by reviewer-19) | — |

Register, ruled: M4.5a noise named white + 1/f, generator = white + AR(1) ρ 0.8 → live surfaces name
AR(1); `scanners.md` / `gates.md` / spec "both" uv lockfiles → three; `run.py` comment
`calibration/` → `score_calibrations/`; bout test "~0.6 %" → 1.18 %; `sample.py` reads the
published CSVs unvalidated (unconfirmed, no red) → kept as register; 2/35 historic subjects
without `→` → no rewrite; README accelerator path = the ruled one.
