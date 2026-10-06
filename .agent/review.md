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
