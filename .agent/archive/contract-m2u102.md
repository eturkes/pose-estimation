# Contract — M2.10.2 camera survey: task span + camera-motion compensation

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
User rulings (this session): trim each clip to its settled task span, export no rows outside it,
count the exclusion in the run report, bridge no interior gap; validate camera compensation on a
synthetic-shake control first and compensate only if it recovers the truth (it did: §2 D07).

## 1. Artifact

- `src/pose_estimation/camera_survey.py` (new) — `CameraSurvey` (`n_frames`, `speeds(start, end)`),
  `estimate_step`, `survey_source`, `net_motion`, `choose_reference`, `register`, `in_bounds`,
  `in_view`, `task_span`, `to_reference`, `runs`, constants `GMC_WORK = 640`, `ORB_WORK = 480`,
  `SAMPLE_EVERY`, `PERSON_AREA`, `SETTLED`, `PERSON_SHARE`, `ANCHOR_INLIERS`, `T_MAX`, `R_MAX`,
  `S_MAX`, `TASK_GAP_S`.
- `src/pose_estimation/run.py` — `--camera-survey` (store_true, default off), `--task-gap-s`
  (float, default `camera_survey.TASK_GAP_S`); `process_source(..., camera_survey=False,
  task_gap_s=TASK_GAP_S)`; diagnostics fields (D10).
- `src/pose_estimation/export.py` — `CAMERA_COLUMNS`; `make_csv_header(tracking, camera=False)`;
  `open_csv_writer(path, tracking, camera=False)`; `camera_values(transform, frame_h, frame_w)`
  (the four normalized cells, 6 dp). `frame_to_rows` is unchanged; `process_source` adds the cells.
- `analysis/clinical_features.R` — `stabilize_camera(df)` applied to 2D input after `read_csv`.
- `src/pose_estimation/corpus_run.py` — `POSE_CONFIG_FIELDS` gains `camera_survey`, `task_gap_s`.
- `scripts/corpus_run_2d.py`, `scripts/pilot_corpus_run.py` — `--camera-survey/--no-camera-survey`
  (default on), `--task-gap-s` (default 2.5), forwarded; `configuration`, `REPORT_FIELDS`,
  `pose_config.json`; report block `task_span` (D11); `GENERATOR_VERSION` corpus v6, pilot v5.
- `docs/technical/tracking-modes.md`, `docs/technical/entrypoints.md` — reference updates.

## 2. Design decisions

**D01 — camera step.** Per consecutive decoded frame pair, at 640 px max-dimension grayscale
(INTER_AREA): Shi-Tomasi (`maxCorners` 600, `qualityLevel` 0.01, `minDistance` 8, `blockSize` 3)
on the earlier frame, **no person mask**; pyramidal LK (21 × 21, 3 levels) forward and back,
keep points with both statuses 1 and forward-backward error < 1 px; >= 20 kept points; RANSAC
`estimateAffinePartial2D` (1.5 px, 2000 iterations, confidence 0.995) with >= 15 inliers. The step
is the 3 × 3 similarity mapping frame t-1 to frame t in full-resolution pixels, else unmeasured.
Evidence (synthetic-shake control, known similarity trajectories on 6 heavily occupied clips,
`.scratch/messy/shake_control.py`): person-masked steps failed on 1960/7209 frames and left SPARC
abs error median 0.20 (max 6.8) at camera RMS speed 0.15 max-dim/s; the open variant failed on
0/7209 and left 0.051 (max 0.19).

**D02 — sampling.** Every `SAMPLE_EVERY = 5`th decoded frame (frame 0 included): ORB (1500
features, FAST threshold 10) on the 480 px grayscale, and the detector's person boxes; a frame has
a person when a box covers >= 1 % of the frame area.

**D03 — reference.** `net_motion[t]` = the composed steps over the next 1 s (`round(fps)` steps)
applied to the frame centre: centre displacement / max-dim + |rotation| deg / 100 + |log scale|;
infinite when any of those steps is unmeasured. A placement = a maximal run with
`net_motion < SETTLED = 0.05`, >= 1 s long, whose sampled frames have a person on >= 0.6 of them.
Reference = the sampled frame nearest the middle of the longest placement (ties → earlier).
No placement → no reference.

**D04 — anchors.** Each sampled frame registers to the reference: ORB descriptors matched by
Hamming kNN (k = 2) with ratio 0.8, >= 10 matches, RANSAC similarity (3 px at 480 px) frame →
reference. An anchor = >= `ANCHOR_INLIERS = 15` inliers and inside the bounds: centre displacement
< `T_MAX = 0.25` max-dim, |rotation| < `R_MAX = 20` deg, |log scale| < `S_MAX = 0.35`. The reference
is an anchor with the identity.

**D05 — in view.** A forward pass and a backward pass carry frame → reference transforms through
the steps (an unmeasured step counts as the identity); an anchor replaces the carried transform. A
frame is in view when either pass gives it a transform inside the D04 bounds.

**D06 — task span.** The in-view run holding the reference, merged with any in-view run whose gap
to it is <= `task_gap_s` (default `TASK_GAP_S = 2.5` s, `round(gap × fps)` frames), repeated until
no run merges. `[start, end)` frame indices. No reference → the span is the whole clip, and
diagnostics say so (D10). Graded on labels (MAIN, P11) — the one-span rule loses task frames that
a long interruption separates from the span; that loss is ruled policy, reported apart.

**D07 — compensation transform.** `C_t` = frame t → reference frame, composed from the D01 steps
alone (unmeasured = identity) outward from the reference in both directions — no anchor resets, so
`C_t` changes only by the measured step between consecutive frames. Exported per row as
`cam_a, cam_b, cam_tx, cam_ty` in the CSV's normalized coordinates (pixels / `export.coord_scale`):
`x_ref = cam_a·x − cam_b·y + cam_tx`, `y_ref = cam_b·x + cam_a·y + cam_ty`. 6 decimal places.
Evidence (same control, 14 clips, side + above): compensated mean speed rel error <= 0.017 and
SPARC abs error median <= 0.11 at RMS camera speed 0.005-0.4 max-dim/s, against 0.06-14× and 0.3-1.9
uncompensated; with no synthetic shake it adds SPARC error median 0.04-0.12.

**D08 — R stabilization.** `analysis/clinical_features.R` applies `stabilize_camera(df)` to 2D input
right after `read_csv`: when all four `cam_*` columns exist, every `<landmark>_x`/`<landmark>_y`
pair (body and hands; `_z` untouched) is replaced by the D07 mapping per row; a row with any `cam_*`
NA keeps raw coordinates; input without the columns is returned unchanged (the shipped goldens
carry none). Similarity mapping ⇒ image-plane angles are unchanged; velocity, jerk, SPARC, reach
and displacement read reference-frame coordinates.

**D09 — two passes.** With `camera_survey` and a file source (a live camera index is never
surveyed), `process_source` opens the source twice: `survey_source` decodes it first with the same
`max_frames` limit, indexing frames like `process_source` (every successful read is a frame index, a
malformed frame included as an unmeasured identity step; `max_frames` counts well-formed frames),
and calls the tracker's `det_model` on sampled frames; then the main decode: frames outside `[start, end)` are not posed and
export no row; the tracker, smoother, bone smoother and hand gate keep their per-source reset, so
the first posed frame is the span start. Every exported row carries D07's columns; without
`camera_survey` the header and rows stay exactly as before (304 body columns).

**D10 — diagnostics.** `SOURCE_DIAGNOSTIC_FIELDS` gains `task_start_frame`, `task_end_frame`,
`frames_outside_task`, `camera_reference_frame` (blank = none), `camera_steps_unmeasured`,
`camera_speed_p50`, `camera_speed_p90` (|step displacement of the frame centre| / max-dim × fps over
the span, 6 dp). Without `camera_survey`: span = whole clip, the rest blank or 0.

**D11 — run report.** Both drivers publish `task_span` = `{frames_decoded, frames_outside_task,
assets_trimmed, assets_without_reference}` summed over assets with diagnostics, plus `camera_survey`
and `task_gap_s` in `configuration`; field names join `REPORT_FIELDS`.

**D12 — generation identity.** `camera_survey` + `task_gap_s` join `POSE_CONFIG_FIELDS`; an event
recorded otherwise is due on resume and refused by `--analyse-only` / `--reuse-run`.

## 3. Predicates

Tester-owned (synthetic, no media, no model):

- **P01** `estimate_step` recovers a known similarity between two synthetic textured frames
  (random texture + a moving foreground patch covering < 30 % of the frame) to < 0.5 px at the frame
  corners; returns `None` for a textureless pair.
- **P02** `net_motion`: composition order, the 1 s horizon, infinity across an unmeasured step,
  the three terms' weights.
- **P03** `choose_reference`: longest qualifying placement; the >= 1 s, < 0.05 and person-share
  >= 0.6 conditions each exclude; nearest sampled frame to the middle; ties → earlier; none → `None`.
- **P04** `in_view` + `task_span`: anchors reset carried transforms; bounds exclusive as stated;
  unmeasured = identity; gap merge at exactly `round(gap·fps)` frames merges and one more does not;
  repeated merging; span holds the reference; no reference → whole clip.
- **P05** `to_reference`: composed from steps alone, outward from the reference, identity at the
  reference; export in normalized units reproduces `x_ref` for a known pixel transform.
- **P06** CSV: with a survey the header = the 304 body columns + `CAMERA_COLUMNS` in that order;
  rows exist only for `[start, end)`; without one the bytes equal the pre-change output.
- **P07** `process_source`: no pose call outside the span; the tracker's first call is the span
  start; diagnostics carry D10 fields with exact values.
- **P08** R `stabilize_camera`: formula per row on body + hand pairs, `_z` untouched, NA row kept raw,
  no-op without columns; `compute_frame_features` angle columns unchanged under a pure rotation +
  translation + scale; R goldens unchanged.
- **P09** CLI + drivers + report: flags, defaults (run off; drivers on, 2.5), forwarding,
  `configuration`, `REPORT_FIELDS`, `task_span` sums over diagnostics rows, `pose_config.json`,
  resume/refusal under another setting.

MAIN-owned, real frames:

- **P10** Shipped `estimate_step` + `to_reference` on the synthetic-shake control (≥ 12 clips, side
  + above): compensated SPARC abs error median <= 0.15 at RMS camera speed 0.15 max-dim/s.
- **P11** Shipped `survey_source` + `task_span` on the watch set against MAIN's dense setup labels
  and the labellers' sparse `setup` column (>= 20 labelled assets): setup frames excluded >= 0.90;
  labelled task frames lost < 0.02, with frames lost to the one-span rule reported apart.
- **P12** Watch-set pilot through the drivers: rows only inside spans, `cam_*` columns present,
  R stage completes on every event, `task_span` block equals the diagnostics' sums.

## 4. Invariant surfaces

1. Every existing suite passes; callers without `camera_survey` keep M2.10.1 output byte for byte.
2. R goldens unchanged; R input without `cam_*` columns computes exactly as before.

## 5. Gate identity

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <ruff check|ruff format --check|ty check|pytest>
```

## 6. Negative controls

1. Person mask restored in `estimate_step` (boxes passed) → P01's foreground case still passes
   while P10 degrades — P10 is the instrument for the mask, P01 for recovery.
2. Gap merge with `<` instead of `<=` → P04 fires.
3. Anchors reset `C_t` → P05 fires (a step change at the anchor).
4. R stabilization maps `y` with `+ cam_b·y` → P08 fires.
5. Drivers default the survey off → P09 fires.

## 7. Verdict table

| row | verdict | evidence |
| --- | --- | --- |
| D01-D12 + A01-A03 | pass | `tester-2` suite 118/118 green on the implementation, 117 red / 1 control green on `e996025` (test bytes `archive/m2u102-test-2` = `3e43b4e`); `reviewer-2` 12 rows: 9 pass, B06/B07/B09 fixed |
| P01-P09 | pass | `tests/test_m2u102_camera_survey.py` (118 cases) |
| P10 | pass | shipped `estimate_step` + `to_reference`, 14 clips (7 side, 7 above), RMS camera speed 0.15 max-dim/s: SPARC abs error median 0.070 side / 0.012 above (uncompensated 0.663 / 1.292); mean speed rel error 0.017 / 0.006; 0/18 819 steps unmeasured |
| P11 | pass | shipped `survey_source` over 83 cached clips (spans equal the prototype's on 78/81; 0 without reference; 11 trimmed); 82 labelled assets: dense (13 clips) setup excluded 6613/6836 = 0.967, task lost 75/6695 = 0.011; sparse setup excluded 41/44, task lost 1/855 = 0.001 (+4 frames the one-span rule drops) |
| P12 | pass | driver pilot 11 assets / 11 289 decoded frames: every row inside its span, `cam_*` present, R completed on every event, verdicts all true, `task_span` {1493 outside, 1 trimmed, 0 without reference} = the diagnostics' sums |
| NC1 | pass (MAIN) | person-masked steps on the synthetic-shake control: SPARC error median 0.202, max 6.78, 1960/7209 unmeasured vs open 0.051 / 0.19 / 0 |
| NC2-NC5 | pass | `<` gap merge 5 red; anchor-reset `to_reference` 3 red; R `+ b*y` 3 red; drivers default off 1 red; sources restored byte-identical |
| reviewer-2 B06/B07/B09 | fixed | `tests/test_rev2_m2u102.py` 8 red / 1 green before, 9 green after |

## 8. Amendments

Appended as ruled.

### A01 — the API, frozen (`tester-2` request)

```python
CameraSurvey(width: int, height: int, fps: float, steps: np.ndarray, measured: np.ndarray,
             reference: int | None, span: tuple[int, int], to_reference: np.ndarray)  # frozen dataclass
  .n_frames -> int                      # len(steps)
  .speeds(start, end) -> np.ndarray     # centre speed of each MEASURED step t in [max(start,1), end), max-dim/s
# steps[t]: 3x3 similarity frame t-1 -> frame t, full-res px; steps[0] = identity, measured[0] = False.
estimate_step(previous: gray, current: gray, scale: float = 1.0) -> np.ndarray | None  # 3x3; translation x scale
in_bounds(transform: 3x3, width, height) -> bool                      # D04 bounds, all strict "<"
net_motion(steps, measured, width, height, fps) -> np.ndarray (n,)
  # horizon = max(1, round(fps)) steps t+1..min(n-1, t+horizon); inf if any of them unmeasured;
  # last frame (no steps ahead) = the previous frame's value (0.0 when n == 1).
runs(mask) -> list[tuple[int, int]]                                   # maximal True runs, [a, b)
choose_reference(motion: (n,), person: dict[int, bool], fps) -> int | None
  # person = {sampled frame index: has a person}; placement = run of motion < SETTLED with
  # b - a >= fps and mean(person over sampled frames inside) >= PERSON_SHARE (no sampled frame -> skip);
  # longest placement wins (first on ties); reference = sampled frame minimising (|f - (a+b)/2|, f).
register(frame_features, reference_features, scale) -> (3x3 | None, inliers: int)
  # features = (points (k,2) full-res float32, ORB descriptors | None)
in_view(steps, anchors: dict[int, 3x3], width, height) -> np.ndarray bool (n,)
  # forward: anchor replaces; else carried @ inv(steps[t]); backward: anchor replaces; else carried @ steps[t+1]
task_span(view, reference: int | None, fps, gap_s=TASK_GAP_S) -> tuple[int, int]
  # gap = round(gap_s * fps) frames; merge when distance <= gap; no reference / no run holding it -> (0, n)
to_reference(steps, reference: int | None) -> np.ndarray (n,3,3)
  # identity at the reference; t > ref: out[t-1] @ inv(steps[t]); t < ref: out[t+1] @ steps[t+1]; None -> all identity
survey_source(capture, detector=None, *, max_frames=0, gap_s=TASK_GAP_S) -> CameraSurvey | None  # None when no frame
export.camera_values(transform, frame_h, frame_w) -> dict[str, str]   # 6-dp strings, keys CAMERA_COLUMNS
```

### A02 — tester harness calls repaired against shipped signatures; reading rulings

`tests/test_m2u102_camera_survey.py` (`tester-2`, `archive/m2u102-test-2`) called
`corpus_run_2d._attempt_event` without its `logs` argument, `pilot_corpus_run._run_event` with a
`logs` attribute the args namespace does not carry and into a directory not yet created, and wrote a
list through `export.open_csv_writer`, which returns a `csv.DictWriter`. Each call now matches the
shipped signature; no assertion changed. Rulings on `tester-2`'s readings: `assets_without_reference`
counts every asset whose diagnostics carry no reference, survey on or off (the literal D11 sum);
the survey stops reading once `max_frames` well-formed frames are in (no extra read); the export
inside a merged span is continuous (out-of-view gaps <= `task_gap_s` are posed and exported);
P08's similarity invariance covers inter-segment angles, not image-axis orientations.


### A03 — `reviewer-2` fixes (B06, B07, B09)

- B06: a malformed read clears the survey's previous frame, so the next step is unmeasured rather
  than a two-frame displacement booked as one.
- B07.1: D11 sums over every asset whose own (owner-matching) diagnostics row exists, any
  disposition — `_artifacts(...)["task_counters"]`, kept apart from the `ok`-only CFR population;
  the reviewer's witness reads that key.
- B07.2: `frames_outside_task` counts decoded indices `< min(n_frames_decoded, survey.n_frames)`
  outside the span, so an early end never counts frames the pass did not reach.
- B09: the survey reads the frame rate through `video_io.safe_fps`, the pose pass's own policy.
