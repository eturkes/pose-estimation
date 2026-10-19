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

