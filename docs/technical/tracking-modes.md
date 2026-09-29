# Tracking modes and schemas

Selected with `--tracking {hands|hands-arms|body}`. Mode constants live in `processing.tracking_pose_indices()`.

| Mode | Body keypoints | Hand keypoints | Pose detection | CSV body prefix |
|------|----------------|----------------|----------------|-----------------|
| `hands` | — | 2 × 21 | Skipped | — |
| `hands-arms` (default) | 12 (shoulders → finger bases) | 2 × 21 | Yes | `arm_` |
| `body` | 33 (all MediaPipe pose) | 2 × 21 | Yes | `body_` |

## Mode constants

- `TRACKING_HANDS`, `TRACKING_HANDS_ARMS`, `TRACKING_BODY` — string keys (also re-exported from package root).
- `BONE_SEGMENTS` vs `BONE_SEGMENTS_BODY` — arm-only vs full-body bone graph.
- `ANGLE_LIMITS` vs `ANGLE_LIMITS_BODY` — joint-angle clamps.
- Wrist/shoulder index pairs are mode-specific; consult `processing.tracking_pose_indices()` rather than hardcoding indices.

## Hand assignment

- `hands` mode: uses the hand model's mirror-corrected handedness when labels
  are at least 0.6-confidence and unambiguous, including when wrists cross;
  wrist x-order is retained as a compatibility fallback for missing/duplicate
  labels. An uncertain sole hand is left unlabeled instead of being fabricated
  as left. The rtmlib whole-body schema supplies anatomical side directly.
- `hands-arms` / `body`: validity-gated Hungarian matching against arm wrists
  (`match_hands_to_arms`). Distance and distality are applied before solving,
  so an inadmissible low-cost edge cannot block a valid alternative. Distality
  requires the hand to be closer to the wrist than to the shoulder midpoint.

## Temporal association and confidence

- Both MediaPipe and rtmlib trackers associate detections against
  velocity-predicted anchors. The shared gated assignment first maximizes the
  count of below-threshold, finite matches, then minimizes cost within that
  feasible set.
- One Euro confidence weights gate both the derivative innovation and the
  position update. A zero-confidence spike therefore cannot freeze the visible
  point while contaminating velocity used by later association/carry-forward.
  Non-finite observations hold the last finite state.
- Displacement caps and adaptive rest-speed decisions use image-space x/y only;
  model-relative z does not enter pixel thresholds.
- In single-subject MediaPipe tracking, `max_tracks=2` limits total hand tracks,
  including dormant tracks in their grace period, rather than only limiting
  newly emitted results.

## rtmlib tracker (`run.py --tracker`)

- `subject` (default) = `SubjectTracker` (`subject_tracker.py`). Every crop is a detector box.
  Between detector calls and across detector misses the box moves by the median displacement
  of keypoints scoring >= 0.3 in consecutive poses; the pose never re-sizes it. Tracks associate
  by box IoU >= 0.3 (`gated_assignment`) and drop after 15 missed detector frames.
- Subject = the track with the largest detector box (EMA of box area / frame area). It is
  sticky: a rival must stay 1.5× larger for 15 consecutive detector frames to take over; a lost
  subject is replaced by the largest live track. `--single-subject` poses the subject alone.
- `KeypointSmoother` receives the tracker's ids (`track_ids`) and associates by identity, so a
  named track exports from its first frame; `min_track_age` gates unnamed tracks only.
- `rtmlib` = upstream `PoseTracker(tracking=False)`: pose-derived crops between detector calls,
  replaced by the detector box every `--det-frequency` frames — the instability mechanism in
  `.claude/rules/rtmlib-runtime.md`. Kept for comparison runs.
- The corpus drivers run the detector every frame (`--det-device GPU --det-frequency 1`); the GPU
  plugin compiles at f32, which reproduces the CPU detector's boxes exactly.

## Score hygiene (rtmlib path, `keypoint_hygiene.py`)

`process_source` applies `apply_hygiene` to the tracker output before the smoother:

- **Out-of-frame points score 0.** A keypoint outside `[0, width) × [0, height)` of the decoded
  frame, or with non-finite coordinates, is not an observation. The smoother holds its position
  while other points of the person stay observed, and carries a person with every point zeroed;
  either way the export is visibility 0. The R stage reads any score above 0 as evidence, so these
  extrapolations used to enter the features.
- **A duplicate hand is suppressed.** When one hand is hidden, RTMW draws it on the visible one at
  lower confidence. If >= 10 hand points are present in both hands, the median distance between
  them is below 0.3 of the larger hand extent, and the weaker hand's mean score is below 0.5 and
  below 0.6 of the stronger's, the weaker hand's 21 scores become 0.
- Out-of-frame zeroing runs first, so a zeroed point never counts as present in the duplicate test.

## Single-subject mode (`--single-subject`)

Three resilience layers for unreliable body detection (e.g. top-down views):

1. **Primary body selection** — keep the largest body. Hands that pass age/spatial filters are preserved; body-level matches re-indexed to primary.
2. **Body carry-forward** — when body detection drops, reuse the last known body for up to ~0.5 s so hands-arms matching continues. Tuned by `carry_grace` and `carry_damping`.
3. **Hand-only fallback** — when carry-forward expires (or no body was ever seen), export a row with blank arm columns and model-assigned left/right hands (x-order fallback).

Hand-only fallback uses the same model handedness policy as `hands` mode;
x-order is only the ambiguity fallback. Live-camera input is mirrored by
default, while unflipped file, batch, and multi-camera inputs swap the model's
selfie-oriented label.

Carried/extrapolated tracks retain their internal geometry and decayed tracking
state. Carried bodies emit zero visibility, and carried rtmlib hands have zero
scores. MediaPipe carried hands are likewise omitted from export; observed hand
presence and rtmlib per-keypoint scores populate explicit hand confidence
columns. Predictions can therefore preserve local display/association
continuity without masquerading as fresh camera evidence in confidence-gated
multi-view fusion.

## Plausibility constraints

- `BoneLengthSmoother` maintains a clipped EMA per finite x/y segment. A bad
  observation cannot arbitrarily move the learned length, missing segments
  initialize independently when they recover, and repeated projections repair
  adjacent segments moved by an earlier correction.
- Joint-angle clamps operate in x/y and rigidly rotate the complete distal
  branch (wrist plus mapped finger points, or ankle plus foot points). This
  preserves downstream geometry. Default elbow/knee upper limits allow a
  straight 180° joint; the lower limit remains 30°.
- z is preserved by these constraints because it is relative model depth, not
  a calibrated Euclidean coordinate.

## rtmlib schema mapping

The face-derived MediaPipe points read the iBUG-300W layout from the subject's own
side: eyes 1 ← 42, 3 ← 45, 4 ← 39, 6 ← 36 and mouth 9 ← 54, 10 ← 48 (face sub-index;
COCO index = 23 + sub).

COCO-WholeBody's 12-point arm projection intentionally uses MCP/base joints for
the existing `*_base` columns. Its 33-point MediaPipe-body projection instead
maps body indices 17–22 to the actual pinky, index, and thumb tips (COCO hand
offsets 20, 8, and 4), matching the MediaPipe body semantics.

## CSV column counts

| Mode | Body columns | Hand columns | Metadata | Total |
|------|--------------|--------------|----------|-------|
| `hands` | — | 2 × 21 × 4 = 168 | 4 | 172 |
| `hands-arms` | 12 × 4 = 48 | 168 | 4 | 220 |
| `body` | 33 × 4 = 132 | 168 | 4 | 304 |

Body keypoints export `x, y, z, visibility`. Hand keypoints export `x, y, z,
confidence`; missing coordinates are blank with confidence zero. Read-back also
supports legacy hand CSVs without confidence columns by using finite-coordinate
presence. With `--single-subject`, body columns may be blank on hand-only
fallback frames.

All association, carry, and plausibility parameters above are provisional.
They are covered by unit/synthetic fixtures, but were not tuned or evaluated on
sensitive recordings or real clinical footage.
