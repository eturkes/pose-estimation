# Port main.py Smoothing/Tracking Features to prototype_rtmw.py

## Context

`prototype_rtmw.py` uses RTMW (133-keypoint whole-body model) and produces
more accurate detections than `main.py`, but suffers from flickery/less smooth
output and lower FPS.  The items below port proven techniques from `main.py`'s
`smoothing.py` and `constraints.py` into `prototype_rtmw.py`'s
`KeypointSmoother` class.

Each item is independent and can be implemented in any order, though items 1-3
are highest priority.  All changes are confined to `prototype_rtmw.py` unless
noted otherwise.

---

## Items

### 1. Confidence-weighted One Euro Filter
- **Status:** DONE
- **Goal:** Reduce flicker on low-confidence keypoints.
- **What:** Modify `_OneEuro.__call__` to accept an optional `confidence`
  array (shape `(n_keypoints,)`).  When provided, blend the filtered output
  with the previous estimate using `w = clip(conf, 0, 1)[:, None] ** gamma`,
  producing `result = w * x_hat + (1 - w) * x_prev`.  Add a `gamma`
  constructor parameter (default 2.0).
- **Wire-up:** In `KeypointSmoother.__call__`, pass the per-person `scores[i]`
  (already available, shape `(133,)`) as the `confidence` argument when
  calling the filter.
- **Reference:** `smoothing.py` `OneEuroFilter.__call__` lines 30-59.

### 2. Differentiated smoothing parameters per body region
- **Status:** TODO
- **Goal:** Heavier smoothing on body/arms, lighter on hands/fingers.
- **What:** Replace the single `_OneEuro` per person with per-region filters.
  Split RTMW's 133 keypoints into regions and apply different parameters:
  - Body (indices 0-16): `min_cutoff=0.3, beta=0.5`
  - Face (indices 23-90): `min_cutoff=0.3, beta=0.5`
  - Hands (indices 91-132): `min_cutoff=1.0, beta=0.3`
  - Feet (indices 17-22): `min_cutoff=0.3, beta=0.5`
- **Implementation:** Each track stores a dict of region filters instead of a
  single `_OneEuro`.  In the smoothing call, slice the keypoint array by
  region, filter each slice with its own `_OneEuro`, and reassemble.
- **Reference:** `smoothing.py` `PoseSmoother.smooth_bodies` (line 244)
  and `smooth_hands` (line 272) for the parameter values.

### 3. Velocity-based carry-forward extrapolation
- **Status:** TODO
- **Goal:** Smooth motion continuity during detection dropouts instead of
  freezing.
- **What:** Store each track's last velocity (`filter.dx_prev`) and last
  timestamp.  During carry-forward (the `_carry` method and the unmatched-track
  loop), extrapolate position:
  ```
  damping = carry_damping ** misses
  step = last_velocity * dt * damping
  # Cap per-keypoint step magnitude at match_thresh
  predicted = last_kps + step
  ```
  Add `carry_damping` parameter to `KeypointSmoother.__init__` (default 0.8).
- **Reference:** `smoothing.py` `PoseSmoother._extrapolate` lines 193-216.

### 4. Hungarian matching
- **Status:** TODO
- **Goal:** Optimal track-to-detection assignment (matters when multiple
  people are close together or crossing).
- **What:** Replace the greedy loop in `KeypointSmoother._match` with
  `scipy.optimize.linear_sum_assignment` on the same centroid distance
  cost matrix.  Filter assignments by `match_thresh` after solving.
- **Reference:** `smoothing.py` `PoseSmoother._match_and_smooth` lines
  129-144.

### 5. Track age gating
- **Status:** TODO
- **Goal:** Prevent one-frame false-positive detections from flashing a
  skeleton.
- **What:** Add a `misses` counter (already exists) and an `age` counter to
  each track.  Increment `age` on each matched frame, reset to 0 on creation.
  In the output assembly, only emit tracks whose `age >= min_track_age`.
  Add `min_track_age` parameter to `KeypointSmoother.__init__` (default 3).
  Carried tracks decrement age by 1 per miss (clamped to 0) so intermittent
  false positives cannot accumulate age across gaps.
- **Reference:** `smoothing.py` `PoseSmoother._match_and_smooth` age logic,
  and `main.py` `min_track_age` / `min_hand_age` usage.

### 6. Bone-length constraints
- **Status:** TODO
- **Goal:** Reduce visual rubber-banding of limbs by enforcing temporal
  bone-length consistency.
- **What:** Port `constraints.py` `BoneLengthSmoother` and adapt bone
  segment index pairs to COCO-WholeBody 133 layout.  Relevant segments:
  - Shoulders→elbows: (5,7), (6,8)
  - Elbows→wrists: (7,9), (8,10)
  - Wrists→index-finger MCP: (9,91), (10,112)
  - Hips→knees: (11,13), (12,14)  (body mode)
  - Knees→ankles: (13,15), (14,16)  (body mode)
  Apply after smoothing, before rendering.  Add a `--no-constraints` CLI flag
  to disable.
- **Reference:** `constraints.py` full file (234 lines).
