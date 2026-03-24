# Algorithm Improvement Plan

Each section below is a self-contained task to be completed in a single
chat session.  Mark the status field as `done` once the section is
finished and committed.

---

## 1 · Biomechanical constraints — bone-length consistency

**Status:** `done`

### Goal

Enforce temporal bone-length consistency so that estimated segment
lengths (shoulder→elbow, elbow→wrist, etc.) do not jump implausibly
between frames.  This catches frames where a keypoint drifts to an
anatomically impossible position — a common artefact with top-down
cameras and partial occlusions.

### Design

Create a new module **`constraints.py`**.

**`BoneLengthSmoother`** — maintains a running exponential average of
each bone segment length per tracked body.  Each call receives the
latest smoothed arm landmarks `(12, 3)` and:

1. Computes the Euclidean distance for every segment pair (6 pairs:
   shoulder→elbow, elbow→wrist, wrist→index-base for each arm side).
2. Updates the running-average length with EMA (α ≈ 0.05, so the
   estimate moves slowly).
3. On each frame, for every segment where the measured length deviates
   from the running average by more than a tolerance (e.g. 40 %),
   projects the distal keypoint along the segment direction so its
   distance from the proximal keypoint matches the expected length.
   **Only the distal keypoint is adjusted** — corrections propagate
   outward from shoulder to wrist to finger base.
4. Returns the corrected landmarks array.

The correction operates on pixel-space landmarks (before CSV
normalisation) and should be called in `main.py` right after
`smoother.smooth_bodies()`.

**Segment pairs** (using the project's arm keypoint indices 0–11):

| Index | Name              |
|-------|-------------------|
| 0 → 2 | left shoulder → left elbow  |
| 2 → 4 | left elbow → left wrist     |
| 4 → 6 | left wrist → left index base|
| 1 → 3 | right shoulder → right elbow|
| 3 → 5 | right elbow → right wrist   |
| 5 → 7 | right wrist → right index base|

### Integration

In `main.py`, after the `smoother.smooth_bodies()` call and before
`match_hands_to_arms()`, run the bone-length correction on each body in
`body_lm`.

### Testing

Add a lightweight test script or inline assertions that verify:

- Repeated calls with constant landmarks produce no correction.
- A single keypoint perturbed by 2× its expected distance is pulled
  back to within tolerance.
- The EMA converges within ~20 frames.

---

## 2 · Biomechanical constraints — joint-angle limits

**Status:** `done`

### Goal

Reject or clamp anatomically implausible joint angles for elbows and
wrists.  A human elbow cannot extend beyond ~170° or flex below ~30°;
wrist lateral deviation is similarly bounded.

### Design

Add to **`constraints.py`**:

**`clamp_joint_angles(landmarks, limits)`**

Takes the arm landmarks `(12, 3)` and a dictionary of angle limits:

```python
ANGLE_LIMITS = {
    # (proximal, joint, distal): (min_degrees, max_degrees)
    (0, 2, 4): (30, 170),   # left elbow
    (1, 3, 5): (30, 170),   # right elbow
}
```

For each triplet, compute the angle at the middle joint using the
2D (x, y) coordinates.  If outside the allowed range, rotate the
distal keypoint around the joint to the nearest limit.

Return the corrected landmarks.

### Integration

Call `clamp_joint_angles()` immediately after `BoneLengthSmoother` in
`main.py`, on each body.

### Notes

- Only clamp, never discard entire frames — the downstream CSV should
  always have a row for every frame.
- Use 2D angles (x, y) because the z-coordinate from MediaPipe is
  relative depth and not metrically reliable for angle computation.
- Keep limits conservative; the goal is to reject clearly wrong
  estimates, not to enforce textbook biomechanics.

---

## 3 · Velocity-based extrapolation during carry-forward

**Status:** `done`

### Goal

When body detection drops out temporarily (the carry-forward / grace
period), instead of replaying the last static landmarks, extrapolate
using the velocity estimated by the One Euro Filter.  This produces
more natural motion continuity during brief occlusions.

### Design

Modify **`smoothing.py`**:

1. Change the track tuple from `(filter, anchor, age, misses,
   last_output)` to `(filter, anchor, age, misses, last_output,
   last_velocity)`.
2. When the One Euro Filter processes a frame, store the filter's
   internal `dx_prev` (velocity estimate) alongside `last_output`.
3. During carry-forward (`emit_carry=True` and the track is in its
   grace period), instead of emitting `last_output` unchanged:
   - Compute `dt` since the last real observation.
   - Extrapolate: `predicted = last_output + last_velocity * dt *
     damping` where `damping` decays exponentially with misses
     (e.g. `0.8 ** misses`).
   - Update `last_output` to the extrapolated value so subsequent
     carry-forward frames continue from the new position.
   - Update the anchor as well so re-matching still works.

### Integration

The changes are internal to `PoseSmoother._match_and_smooth` and
`smooth_bodies`.  No API changes — callers already receive smoothed
landmark arrays.

### Edge cases

- If velocity is very large (probably spurious), cap extrapolation
  magnitude to e.g. the match threshold per frame.
- If `last_velocity` is None (first frame), fall back to static carry.
- Damping ensures extrapolation decays to static after a few frames,
  avoiding runaway drift.

---

## 4 · Savitzky-Golay post-processing for batch CSVs

**Status:** `done`

### Goal

For offline / batch video processing, add a second-pass Savitzky-Golay
smoothing filter over the exported CSV data.  Unlike the real-time One
Euro Filter, Savitzky-Golay uses both past and future samples and
preserves peaks better (polynomial fit vs. exponential smoothing),
making it ideal for post-hoc refinement.

### Design

Create a new module **`postprocess.py`** with:

**`savgol_smooth_csv(input_path, output_path, window=11, polyorder=3)`**

1. Read the CSV into a pandas DataFrame.
2. Group by `(video, person_idx)`.
3. For each numeric landmark column, apply
   `scipy.signal.savgol_filter(column, window_length, polyorder)`.
   - Skip columns that are entirely blank (unmatched hands).
   - Handle NaN gaps (short gaps: interpolate first, then filter;
     long gaps: filter each contiguous segment independently).
4. Write the smoothed DataFrame to `output_path`.

The window length should be odd and defaults to 11 frames (~0.37 s at
30 fps).  Polynomial order defaults to 3.

### Integration

Add a **`--postprocess`** flag to `main.py`:

- When used with `--batch-dir` or `--source <file>`, after the main
  processing loop finishes, run `savgol_smooth_csv()` on each output
  CSV, writing to `<stem>_smooth.csv` alongside the original.
- Print a summary: which files were post-processed, the window/order
  used.

Also make `postprocess.py` usable standalone:
```
python postprocess.py output/video1.csv --window 15 --polyorder 3
```

### Dependencies

This requires `pandas` and `scipy`.  **Do not add them to
`requirements.txt`** (they are only needed for post-processing).
Instead, import them inside the function and raise a clear error if
missing.

---

## 5 · Arm-guided hand ROI fallback

**Status:** `done`

### Goal

When the palm SSD detector fails to detect a hand (common from top-down
cameras or when the hand is partially occluded), synthesise a hand
detection from the arm wrist keypoints so the hand landmark model still
gets a chance to run.

Research shows that the MediaPipe Holistic hand ROI heuristic breaks
when the hand plane is not parallel to the camera.  Using shoulder →
elbow → wrist geometry to estimate hand position, orientation, and
scale is more robust for non-frontal views.

### Design

Modify **`processing.py`**:

**`_synthesise_hand_detections(body_landmarks, existing_palm_dets,
frame_h, frame_w)`**

For each detected body, for each arm side (wrist keypoint 4 and 5):

1. Skip if a palm detection already overlaps this wrist (distance
   between palm detection centre and wrist < some threshold, e.g. 0.1
   in normalised coordinates).
2. Extract the arm chain: shoulder → elbow → wrist.
3. Compute forearm direction: `(wrist - elbow)` normalised.
4. Estimate hand centre: `wrist + forearm_direction * forearm_length *
   0.4` (the hand extends ~40 % of the forearm length beyond the
   wrist).
5. Estimate rotation from forearm direction (same formula as
   `get_hand_crop`).
6. Estimate box size from forearm length × 0.8.
7. Build a synthetic detection dict with `box`, `keypoints` (wrist
   as kp 0, estimated middle-finger as kp 2 along the forearm
   direction), and `score` (use arm wrist visibility).

Append the synthetic detections to the palm detection list before hand
landmark inference.

### Integration

In `process_frame()`, after `run_detection()` for palms and before
the hand landmark loop, call `_synthesise_hand_detections()` if body
landmarks are available.  Only add synthetic detections for wrists
that aren't already covered by a real palm detection.

### Notes

- The synthetic detection is a best-guess; the hand landmark model
  will still reject the crop if no hand is present (via `hand_flag`).
  So false positives are filtered naturally.
- Mark synthetic detections with a flag (e.g. `"synthetic": True`) so
  they can be excluded from detection smoothing state (they shouldn't
  anchor future real detections).
- This requires body landmarks to be available in `process_frame()`
  before the palm detection path runs, which is already the case in the
  current pipeline order.

---

## 6 · Confidence-weighted temporal smoothing

**Status:** `pending`

### Goal

Use MediaPipe's per-keypoint visibility scores to modulate temporal
smoothing intensity.  Low-confidence keypoints should be smoothed more
aggressively (pulled toward the previous estimate); high-confidence
keypoints should pass through with less filtering.

This is particularly valuable for partially occluded arms where some
keypoints have high confidence and others are guesses.

### Design

Modify **`smoothing.py`**:

1. Add a `ConfidenceWeightedOneEuroFilter` subclass (or extend
   `OneEuroFilter`) that accepts an optional `confidence` array
   alongside the landmark array `x`.
2. After computing the standard One Euro filtered output `x_hat`,
   blend between `x_hat` (the smoothed value) and `self.x_prev`
   (the previous value) based on confidence:

   ```python
   # confidence in [0, 1] per keypoint, broadcast to (N, 3)
   w = confidence[:, None] ** gamma   # gamma > 1 sharpens the curve
   result = w * x_hat + (1 - w) * self.x_prev
   ```

   When confidence is high (≈ 1), the result is the normal One Euro
   output.  When confidence is low (≈ 0), the result stays near the
   previous position, resisting the noisy input.

3. `gamma` defaults to 2.0 (so confidence 0.5 → weight 0.25, heavily
   pulling toward the previous estimate).

### Integration

In `PoseSmoother.smooth_bodies()`, pass `body_visibilities` through to
the filter.  This requires threading visibility arrays through
`_match_and_smooth()`.  Hand landmarks have no per-keypoint visibility,
so the hand path is unchanged.

### Notes

- This must compose correctly with the bone-length and angle-limit
  corrections from sections 1–2 (those run after smoothing, so the
  order is: filter → confidence weighting → bone length → angle
  limits).
- Store the blended result as `x_prev` for the next frame, not the
  raw filter output, so the filter state stays consistent.
