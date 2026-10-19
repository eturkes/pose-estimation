# Contract — M4.6 pipeline defects before calibration

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
User rulings (this session): any `.agent/deferred.md` row may join the phase; features + pipeline
free to change. Consultant-2 register: these repairs must precede every per-arm calibration, or
the calibration reruns. Four deferred rows close here: P08 gate keying, P09 out-of-frame residue,
the vacuous D02 census, the OpenVINO build missing from `pose_config.json`.

## 1. Artifact

- `src/pose_estimation/run.py` `process_source`: (a) under `--single-subject` the hand gate runs
  on the selected subject row with the one key `SUBJECT_KEY = "subject"`, after
  `filter_single_subject`; without `--single-subject` it keeps per-track keys before export, as
  today. (b) after the gate and the subject selection, `keypoint_hygiene.zero_out_of_frame` runs
  again on the exported keypoints (`EXPORT_HYGIENE`), so no stage after the first hygiene pass
  can export a point outside `[0, w) × [0, h)` at a positive score. (c) diagnostics gain
  `duplicate_hand_fired` (frames where `suppress_duplicate_hand` zeroed a hand) and
  `exported_out_of_frame_zeroed` (points the export pass zeroed).
- `src/pose_estimation/keypoint_hygiene.py`: `suppress_duplicate_hand(keypoints, scores, *,
  fired=None)` and `apply_hygiene(..., fired=None)` — `fired` (a list) receives the person index
  once per person whose weaker hand is zeroed; scores returned unchanged by its presence.
- `src/pose_estimation/corpus_run.py`: `POSE_CONFIG_FIELDS` gains `openvino_build`;
  `pose_config(args)` records `openvino.get_version()` under it; a recorded config with another
  build is not a match, so resume re-runs that event.
- `scripts/corpus_run_2d.py` + `scripts/pilot_corpus_run.py`: forward + report the build
  (`configuration.openvino_build`; `REPORT_FIELDS` gains it).

## 2. Design decisions

**D01 — gate key = the exported subject stream.** One key per source in single-subject mode: the
gate's hysteresis follows what is exported, whichever smoother track supplies it, so a transient
second track can neither switch the subject's hand off nor hand its own state over. The state
resets when a frame exports no row (the stream broke), mirroring the per-track expiry it
replaces. Multi-person export keeps per-track keys (rows there are identities).

**D02 — export hygiene.** The out-of-frame ruling ("an out-of-frame keypoint is not an
observation") binds the exported point, wherever it moved. Coordinates stay as computed; only the
score drops to 0, so the row carries visibility / confidence 0 for that point (the writer blanks
hand coordinates only when the whole hand is absent).

**D03 — duplicate census.** Counted per person-frame at the shipped hygiene call; the diag row
publishes it per asset, so D02's firing is measured instead of inferred.

**D04 — build in the configuration.** A build swap moves landmarks by up to 4.1e-5 and features by
up to 4.1e-4 of column max (`rtmlib-runtime.md`); a resumed run must not mix builds.

## 3. Predicates

- **P01** Single-subject gate keying: a synthetic two-track source where the subject's own track
  carries a handless row for k frames while a transient track takes the exported row with its
  hands, then the subject returns with hand level in [0.5, 0.65): the exported hand stays
  admitted (on the base revision it is switched off — the red witness); a hand that drops below
  0.5 still switches off; a frame with no exported row resets the state.
- **P02** Multi-person export keeps per-track keys (the M2.10.1 P03/P05 keyed-state cases hold
  unchanged without `--single-subject`).
- **P03** Export hygiene: a stage after the first hygiene pass (bone-length projection, filter
  extrapolation, carried prediction) placing a scored point outside the frame exports score 0
  (visibility 0 / hand confidence 0); in-frame points unchanged; `exported_out_of_frame_zeroed`
  counts it.
- **P04** `duplicate_hand_fired` > 0 on a synthetic duplicate-hand row and 0 with no duplicate;
  `apply_hygiene` output unchanged by passing the counter.
- **P05** `openvino_build` in `POSE_CONFIG_FIELDS` and `pose_config.json`; a recorded config with a
  foreign build string is due on resume; the report's `configuration` carries the build.
- **P06** Determinism campaigns re-measured where `corpus_run.py` or `video_io.py` sit in a
  `SOURCE_FILES`/`SOURCES` tuple (`rg -ln` over `tests/*_results.json`), regenerated last.

MAIN-owned (real data): **P07** on the P09 hit set, one replay (`bench.replay`) per moving-stage
hypothesis names which stage placed the residue (bone-length vs filter), recorded in
`docs/technical/pose-weak-points.md`.

## 4. Invariant surfaces

The superseded M2.10.1 P05 clause "gate precedes subject selection" is replaced by D01 for
single-subject export (its test is rewritten to pin the new order; its red on the new code is the
recorded firing). No feature or R code changes.

## 6. Negative controls

- NC1 per-track keying restored under `--single-subject` reds P01.
- NC2 export hygiene removed reds P03.
- NC3 a collector that appends per zeroed point rather than per firing person reds P04 on a
  one-person, one-duplicate fixture (`fired == [0]`).

## 7. Verdict table

| ID | verdict | evidence |
| --- | --- | --- |
| P01-P05 | pass | `tests/test_m4u6_pipeline_defects.py` 10 cases (7 red on `487f790` + bench, the 3 green = guards the old code already met) + reviewer-10's malformed-frame reset (fixed, R01); M2.10.1's order case rewritten to pin select → gate (its old assertion reds on the new code = the recorded firing) |
| P06 | pass | `tests/cohort_determinism_results.json` regenerated last (`corpus_run.py` in its `SOURCES`) |
| P07 | open | stage attribution replay on real hit clips — MAIN, after the GPU queue |

## 8. Amendments
