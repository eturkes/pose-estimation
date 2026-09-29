# Contract — M2.9.1 subject tracker: detector-convention crops, sticky subject, identity-keyed smoothing

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
Catalog of the weak points this unit repairs → `docs/technical/pose-weak-points.md` (C02, C03,
C08, C11; C04 in part).

## 1. Artifact

- `src/pose_estimation/subject_tracker.py` (new) — `SubjectTracker(PoseTracker)`, `box_iou`,
  `pose_shift`.
- `src/pose_estimation/rtmlib_smoothing.py` — `KeypointSmoother.__call__(keypoints, scores, t,
  track_ids=None)`.
- `src/pose_estimation/run.py` — `--tracker {subject,rtmlib}`; `process_source` forwards the
  tracker's identities to the smoother.
- `src/pose_estimation/rtmlib_openvino.py` — GPU compile config.
- `scripts/corpus_run_2d.py`, `scripts/pilot_corpus_run.py` — tracker flag, defaults, report field.
- `docs/technical/tracking-modes.md`, `docs/technical/entrypoints.md`,
  `docs/technical/architecture.md` — reference updates (conventions.md:46).

## 2. Design decisions

**D01 — every crop is a detector box; the pose never re-sizes it.** rtmlib's stateless branch
crops from `pose_to_bbox(kpts)` between detector calls and from the detector box on detector
frames; the two conventions disagree in extent, the crop jumps, and the top-down model returns a
different skeleton (509 px translation + 321 px deformation; `rtmlib-runtime.md`). Here the box a
track carries is always the last detector box matched to it, translated by the pose's own motion.
Its width and height change only when a detector box is matched to the track.

**D02 — transport = median displacement of keypoints confident in both consecutive poses.**
After a track is posed, its box moves by `pose_shift(prev_kps, prev_scores, kps, scores,
floor=confidence_floor)`: the per-axis median of `kps - prev_kps` over keypoints whose score is
`>= floor` in both poses and whose coordinates are finite in both; fewer than 3 such keypoints, or
no previous pose → zero shift. The shift applies to both corners (`box + [dx, dy, dx, dy]`). On a
detector frame the matched detector box is posed first and then shifted, so the box carried into
the next frame is a constant-velocity prediction.

**D03 — the detector runs every frame on GPU at f32.** Cost basis of the superseded "reconcile
subclass" ruling (8.23× = ~64 h for `det_frequency=1`) was the CPU detector at 318.7 ms/call.
Measured on 240 corpus frames / 40 clips: GPU f32 = 41.6 ms/call and **bit-identical to CPU**
(IoU 1.0000, score delta 0, 0 count mismatches); GPU f16 (plugin default) = 19.8 ms but IoU min
0.982 and 1 box flipped at the 0.3 score cut. So the GPU compile carries
`INFERENCE_PRECISION_HINT = f32` beside `PERFORMANCE_HINT = LATENCY`; CPU + NPU configs are
unchanged. `det_frequency` stays a parameter (default 1): at N > 1 the box is transported on
the N − 1 frames between detector calls, never re-sized.

**D04 — association = box IoU through `gated_assignment`.** Cost `1 - IoU`, admitted iff
`IoU >= iou_threshold` (0.3; `threshold = 1 - iou_threshold + 1e-12` under the helper's strict
`<`), maximum cardinality then minimum cost. Matched track → box = detector box, `misses = 0`,
area EMA update. Unmatched track → `misses += 1`. Unmatched box → new track, `id = next_id++`,
area = its own box area. A track is removed once `misses > max_misses` (15). Misses count
detector frames. A missed track that is still live keeps being posed at its transported box.

**D05 — subject = the largest detector box, sticky.** A track's `area` = EMA (`area_alpha` 0.2)
of detector-box area / frame area, seeded at the first box. No live subject → subject = the
track with the largest `area` (first in track order on a tie), or none. With a live subject, on
detector frames only: the rival with the largest `area` challenges when
`rival.area > switch_ratio * subject.area` (1.5); `switch_frames` (15) consecutive challenging
detector frames by the **same** rival make it the subject; any frame without a challenge, or a
different challenger, restarts the count. Evidence: the shipped single-subject rule is the argmax
of mean score over all 133 keypoints, 68 facial — on the C02 probe clip the patient's box is
0.46-0.87 of the frame against a bystander's 0.06-0.26, and the argmax picked the bystander on
20/31 frames; arms-only subjects (C11) are the largest detector box in all 3 probed clips
(0.13-0.32 against 0.012-0.032).

**D06 — output.** `single_subject=True` → the subject alone is posed and returned (1 row), or
nothing. `single_subject=False` → every live track is posed and returned, subject first, the rest
in track order. `last_track_ids` = the ids of the returned rows, in row order. No row → arrays of
shape `(0, 133, 2)` and `(0, 133)`, float64, and the pose model is not called. The pose model is
always called with exactly one box, never an empty list (RTMPose's whole-frame fallback).

**D07 — identity-keyed smoothing.** `KeypointSmoother.__call__(..., track_ids=ids)` associates row
`i` with the live smoother track carrying `ids[i]` (after the existing drop of rows with no
observed keypoint), regardless of centroid distance; an id with no live track opens a new track.
A track opened under an id is emitted from its first frame — `min_track_age` gates unnamed tracks
alone. Carried tracks keep their id. `len(track_ids) != rows` → `ValueError`. `track_ids=None` →
behaviour unchanged, centroid matching included. Evidence: the shipped chain withholds the first
2 frames of every track and re-keys a subject whose confidence mass moves; first exported row
>= frame 2 on 379/379 clips.

**D08 — runner.** `run.py --tracker {subject,rtmlib}`, default `subject`. `subject` builds
`SubjectTracker(..., single_subject=args.single_subject)`; `rtmlib` builds the upstream
`PoseTracker` exactly as before. Both run with `tracking=False`; `SubjectTracker` refuses a truthy
`tracking` (`ValueError`) and a solution without a detector (`ValueError`). `process_source`
passes `pose_tracker.last_track_ids` to the smoother whenever the tracker exposes that attribute,
and calls the smoother without it otherwise. `filter_single_subject` stays downstream unchanged.

**D09 — drivers + provenance.** `corpus_run_2d.py` and `pilot_corpus_run.py` take `--tracker`
(default `subject`), pass it to `pose_estimation.run`, and publish it as
`configuration.tracker` (`REPORT_FIELDS` gains `tracker`; `GENERATOR_VERSION` bumps). Defaults
become `--det-device GPU --det-frequency 1`. Two landmark generations under different trackers
are shaped identically; the report is where they separate (same rule as `coord_normalization`).

## 3. Predicates

Tester-owned (diff-blind suite, stub detector + stub pose model, no media, no accelerator):

- **P01** Crop provenance: over a stimulus whose pose output implies a box of a different extent
  than the detector's, every pose call receives exactly one box whose width and height equal the
  last detector box matched to that track.
- **P02** Transport: poses moving by `(dx, dy)` with >= 3 keypoints confident in both frames move
  the next crop by exactly the per-axis median displacement; < 3 such keypoints, or scores below
  `confidence_floor`, leave it in place; non-finite keypoints are excluded from the median.
- **P03** Cadence + no starvation: detector calls = `ceil(frames / det_frequency)` for
  `det_frequency` in {1, 2, 5, 7, 13}; pose calls = frames × posed tracks; zero pose calls with an
  empty box list.
- **P04** Association (`single_subject=False`): IoU >= 0.3 keeps the id (0.3 exactly included),
  IoU < 0.3 mints a new id;
  a track missed for `max_misses` consecutive detector frames is still returned, for
  `max_misses + 1` it is gone; ids are never reused.
- **P05** Subject selection: at the first detection the largest box is the subject; a smaller
  person with higher keypoint scores never becomes the subject while the larger one is live.
- **P06** Stickiness: a rival over `switch_ratio × subject` for `switch_frames - 1` detector
  frames does not take over, for `switch_frames` it does; one non-challenging detector frame
  restarts the count; a change of challenger restarts it; losing the subject re-picks the
  largest live track at once.
- **P07** Output shape: `single_subject` → 1 row or 0, `last_track_ids` = `[subject]` or `[]`;
  otherwise all live tracks, subject first; `len(last_track_ids)` = returned rows on every frame;
  no track → `(0, 133, 2)` / `(0, 133)` float64 and no pose call.
- **P08** Construction: `SubjectTracker` is a `PoseTracker` with `tracking is False`; truthy
  `tracking` → `ValueError`; a solution without a detector → `ValueError`; `reset()` clears
  tracks, subject, challenger, ids and `frame_cnt`.
- **P09** Identity-keyed smoothing: a row keeps its smoother track across a 1000 px jump under one
  id (same `output_track_keys()` entry); a new id is exported on its first frame; unnamed rows
  still wait `min_track_age` frames; mismatched lengths → `ValueError`.
- **P10** GPU config: compiling on `GPU` passes `INFERENCE_PRECISION_HINT: f32` and
  `PERFORMANCE_HINT: LATENCY`; `CPU` and `NPU` pass `PERFORMANCE_HINT: LATENCY` alone (fake
  `openvino.Core`).
- **P11** Runner: `run.main` builds a `SubjectTracker` by default with `single_subject` forwarded;
  `--tracker rtmlib` builds a plain `PoseTracker` (not a `SubjectTracker`); `process_source`
  passes the tracker's `last_track_ids` to the smoother, and calls it without `track_ids` when the
  tracker has no such attribute.
- **P12** End to end, the C02 shape: two people, the patient's box larger with lower mean
  keypoint score, the bystander's smaller with higher mean score. The default runner chain
  (tracker → smoother → `filter_single_subject`) exports the patient on every frame from frame 0;
  the `--tracker rtmlib` chain at `det_frequency=7` exports the bystander on at least one frame
  (red at base, where the default chain is the rtmlib chain).
- **P13** Drivers: both scripts pass `--tracker <value>` to `pose_estimation.run`, publish
  `configuration.tracker`, carry `tracker` in `REPORT_FIELDS`, and default to `det_device=GPU`,
  `det_frequency=1`, `tracker=subject`.

MAIN-owned, real corpus — the user's own acceptance for the instability repair
(`.agent/deferred.md` *User rulings on the instability repair*): re-run the seven-arm pilot sweep
with the repaired tracker. Accelerator recipe, `scripts/pilot_corpus_run.py --seed 20260922
--min-assets 4 --max-frames 400` (4 events / 11 assets, the sweep's own sample), `--tracker
subject --det-device GPU`, `det_frequency` in {1, 2, 3, 7, 14, 21, 35}; statistics by the sweep's
own instruments (`.scratch/arm_stats.py`, `.scratch/cadence_peak.py`); reference arms
`.scratch/detfreq/f1` (quality target) + `f7` (shipped baseline, 327.3 s).

- **P14** Quality at the shipped arm (`det_frequency=1`): median alternation <= 0.55 and
  isolated relocations = 0 (the ruling's "~0"); whole-skeleton rate below the old f1 arm's
  3.79 % of observed frames.
- **P15** Cadence at the shipped arm: no peak at the old shipped cadence (4.29 Hz) above the
  6.7 Hz control column's range; every other arm's cadence ratio reported.
- **P16** Cost: the shipped arm's `outer_wall_s` <= 1.25 × 327.3 s.
- **P17** The curve: all seven arms tabulated beside the old sweep (wall, relocation rates,
  alternation, cadence); the shipped `det_frequency` is the arm that meets P14 at the lowest wall.

## 4. Invariant surfaces

1. `--tracker rtmlib` = the shipped behaviour: `tests/test_corpus_run_2d.py` P01-P04 pass
   unmodified.
2. `KeypointSmoother` without `track_ids`: `tests/test_smoothing.py` +
   `tests/test_rtmlib_csv_export.py` pass unmodified.
3. Landmark CSV schema (304 columns), diagnostics schema, R stage: unchanged.
4. Published trees (`output/`, `cohort/`, every publisher tree): untouched in this unit; the
   corpus rerun is its own unit.

## 5. Gate identity

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <ruff check|ruff format --check|ty check|pytest>
```

Close expects the base collection + the new cases, zero skipped (A32). Red witness = targeted
`pytest tests/<suite>` on the base tree (→ `gates.md`).

## 6. Negative controls

Each seeded into the implementation at close, graded, reverted by digest:

1. Crop from `pose_to_bbox(kps)` between detector frames → P01 fires.
2. Subject = argmax of mean keypoint score → P05 + P12 fire.
3. `count >= switch_frames - 1` → P06 fires.
4. Drop `or ext_id is not None` from the smoother's emit gate → P09 fires.
5. Drop the f32 hint → P10 fires.
6. Default `--tracker rtmlib` → P11 + P12 fire.

## 7. Verdict table

Appended at close.

## 8. Amendments

Appended as ruled.

### A01 — D04: association scores the better of the carried box and the last detector box

`IoU = max(IoU(carried box, detection), IoU(last matched detector box, detection))`; a match
sets both to the detection. Found by the shipped
`tests/test_corpus_run_2d.py::test_p04_shipped_detector_cadence_is_frames_over_det_frequency[miss-at-3-1]`:
one frame whose pose jumps 990 px carries the box off the person (D02), the next detection
scores IoU 0 against the carried box, a new id is minted, and the orphan is posed at an empty
box for `max_misses` detector frames — 2 pose calls per frame, and under `single_subject` the
orphan stays the subject. Scoring the anchor as well re-attaches the detection to its track.

### A02 — D02: transport needs consecutive poses

A track not posed on a frame (a non-subject under `single_subject`) forgets its pose, so the
next time it is posed its box moves by zero. Otherwise a pose from many frames earlier pairs
with the current one and moves the box by the whole interval's displacement.

### A03 — D09: the report may name only the generation the tree carries (`reviewer-1` R09)

`reviewer-1` reproduced on `51e2094`: corpus resume, `--analyse-only` and pilot `--reuse-run` each
published `configuration.tracker = subject` over landmarks an rtmlib run produced (one pose
invocation, three green reports) — `configuration` was the invocation's, which breaks
`corpus-run.md`'s "the published report is a function of STATE". Every field of `configuration`
had the defect; D09 made it likelier by moving the defaults. Repair: after an event's run stage
passes, the driver writes `pose_config.json` (`corpus_run.POSE_CONFIG_FIELDS`: model, tracking,
tracker, det_device, pose_device, det_frequency, single_subject) into the event directory; the
pilot does the same after each run. A complete event whose record is absent or differs from the
invocation's is due again on resume; `--analyse-only` and any report refuse to publish while a
complete event carries another configuration; `--reuse-run` refuses the same way. Two fixtures
that build complete events by hand (`tests/test_review_m2u82.py` throughput case, the P13 pilot
case) now write the record.

### A04 — D07: an id list is measured by length, never truthiness (`reviewer-1` R07)

`if track_ids:` misread NumPy ids (an empty array raised on truth-testing, `np.array([0])` read
as empty). The empty-input path now refuses `len(track_ids) != 0`.

### Verdict table (appended at close)

| row | verdict | evidence |
| --- | --- | --- |
| D01-D09 + A01-A04 | pass | `tester-1` suite 98/98 green on the implementation, 98/98 red on `e667022`; `reviewer-1` 14 rows: 9 pass, 4 findings fixed + re-reviewed |
| P01-P13 | pass | `tests/test_m2u91_subject_tracker.py` (98 cases); R14 hardening: the unnamed-smoothing case grades against `ff117de`'s own class (`tests/fixtures/m2u91/`) and reds on a passthrough |
| P14 | pass | shipped arm (`det_frequency=1`): alternation 0.497 (<= 0.55), isolated 0.000 %, whole 0.000 % (old f1 3.79 %), 2502 observed frames |
| P15 | pass | 4.29 Hz peak/bg 0.850 inside the 6.7 Hz control range 0.840-1.161; positive control old f7 1.336 |
| P16 | pass | 283.6 s <= 409.1 s (1.25 × 327.3 s) |
| P17 | pass | arms 1/2/3/7/14/21/35: wall 283.6/192.6/155.9/125.1/117.0/104.4/97.5 s, alternation 0.497/0.612/0.612/0.611/0.604/0.564/0.556, isolated 0.000/0.039/0.078/0.039/0.082/0.162/0.000 %; only arm 1 meets P14 |
| reviewer-1 R07 R09 R12 R14 | fixed | A03, A04, `entrypoints.md` reworded, golden oracle; `tests/test_rev1_m2u91.py` 6/6 green (red on `51e2094`/`0fca7cd`) |
| NC1 | pass | pose-derived crop: 14 red (P01 ×2, P02 ×9, P04 ×3) |
| NC2 | pass, weaker than written | argmax-mean subject: 16 red (P05 ×3, P06 ×10, P02, P07, P08); P12 stays green — under `single_subject` the seeded rule never poses the bystander, so it never sees its score; P05 is the predicate that grades the rule |
| NC3 | pass | switch one frame early: 9 red (P06 ×8, P08) |
| NC4 | pass | named tracks age-gated: 5 red (P09 ×4, P12) |
| NC5 | pass | no f32 hint: 1 red (P10) |
| NC6 | pass | default `rtmlib`: 3 red (P11 ×2, P12) |
