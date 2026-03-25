# Pose Estimation — Optimisation Plan

Each section below is a self-contained task.  Complete them in
order; each section states its file targets, rationale, and
verification steps.  Mark a section **[DONE]** in the heading
once all its verification steps pass.

---

## Section 1 — Fix broken extrapolation tests [DONE]

### Status: [DONE]

### Rationale

`test_extrapolation.py` has two API mismatches introduced when
`smooth_hands` was changed to return `(smoothed, n_active)` and
hand carry-forward was enabled (`static_carry=True`):

1. All `smooth_hands` calls assign the *tuple* to `smoothed`
   instead of unpacking `(smoothed, n_active)`.
2. `test_hand_tracks_unaffected` asserts `len(smoothed) == 0`
   after dropping the hand for one frame, but hands now carry
   forward via `static_carry=True` (up to `grace` frames).

### Files

- `test_extrapolation.py`

### Changes

1. Unpack every `smoother.smooth_hands(...)` call:

   ```python
   smoothed, n_active = smoother.smooth_hands(...)
   ```

2. In `test_hand_tracks_unaffected`:
   - After dropping the hand for one frame, **expect**
     `len(smoothed) == 1` (static carry emits the last output).
   - Assert `n_active == 0` to confirm no real detection matched.
   - On re-acquire, assert the track age is > 1 (re-matched the
     existing track, not a fresh one).

### Verification

```bash
python3 test_extrapolation.py
# Expect: "All extrapolation tests passed."
python3 tests/test_smoothing.py
python3 tests/test_constraints.py
# Expect: both pass unchanged.
```

---

## Section 2 — Hungarian matching for hands-to-arms

### Status: TODO

### Rationale

`match_hands_to_arms()` in `processing.py` uses a greedy
nested loop: each arm wrist grabs the closest unmatched hand in
order.  Every other matching routine in the project uses
`scipy.optimize.linear_sum_assignment` for optimal assignment.
With multiple arms/hands the greedy order can produce suboptimal
pairings (e.g., arm 0 steals a hand that is the only good match
for arm 1's wrist).

### Files

- `processing.py` — `match_hands_to_arms()`
- `tests/test_matching.py` — **new file**

### Changes

1. Replace the greedy loop with a cost-matrix + Hungarian
   assignment approach:

   - Build a cost matrix of size `(n_wrists, n_hands)` where
     each row is one arm-wrist and each column is one hand.
   - Populate with Euclidean distances between arm wrist and
     hand wrist (`hand_lm[0, :2]`).
   - Run `linear_sum_assignment(cost)`.
   - Filter matches by `threshold` distance.
   - Apply the existing distality check (hand must be closer
     to wrist than to shoulder midpoint).
   - Return the same `(arm_idx, wrist_kp, hand_idx)` tuples.

2. Preserve the existing function signature exactly:

   ```python
   def match_hands_to_arms(body_landmarks, hand_landmarks,
                           threshold=100, wrist_kps=None,
                           shoulder_kps=None):
   ```

3. Create `tests/test_matching.py` with at least these cases:
   - **No bodies or no hands** → empty result.
   - **Single body, two hands** → both wrists matched.
   - **Two hands equidistant** — verify optimal (not greedy)
     assignment: place hand A at (60, 300) near left wrist and
     hand B at (240, 300) near right wrist; the greedy order
     should not matter.
   - **Hand beyond threshold** → not matched.
   - **Distality check** — a hand near the shoulder midpoint
     is rejected even if within threshold.
   - **Cross-body scenario** — two bodies, one hand between
     them; verify it's matched to the nearest wrist globally.

### Verification

```bash
python3 tests/test_matching.py
# All tests pass.
python3 tests/test_smoothing.py
python3 tests/test_constraints.py
python3 test_extrapolation.py
# All previously passing tests still pass.
```

---

## Section 3 — Confidence-weighted hand smoothing

### Status: TODO

### Rationale

Body smoothing passes per-keypoint `visibility` scores to the
One Euro Filter so low-confidence keypoints resist noise.
Hand smoothing currently passes `confidences=None`, so all
21 hand keypoints are treated identically regardless of the
model's `hand_flag` confidence.

Plumbing `hand_flag` through gives the filter a signal to hold
steady when the model is uncertain about a hand crop (e.g.,
motion blur, partial occlusion).

### Files

- `processing.py` — `process_frame()` return value
- `smoothing.py` — `smooth_hands()`
- `main.py` — call site for `smooth_hands()`
- `tests/test_smoothing.py` — extend `test_hand_path_unaffected`

### Changes

1. **`process_frame()` already returns `hand_landmarks`** as a
   list of `(21, 3)` arrays.  Add a parallel return value
   `hand_flags`: a list of floats, one per accepted hand (the
   `hand_flag` confidence from `detect_hand_landmarks`).

   Update the return signature:

   ```python
   return (body_landmarks, body_visibilities,
           hand_landmarks, hand_flags, state, diag)
   ```

   Collect `hand_flag` alongside each accepted hand landmark
   in the existing loop (lines ~657-681).

2. **`smooth_hands()`** — accept an optional `hand_flags`
   parameter (list of floats, one per hand).  Broadcast each
   flag to a `(21,)` confidence array and pass to
   `_match_and_smooth` via `confidences`:

   ```python
   def smooth_hands(self, hand_landmarks, t,
                    hand_flags=None, grace=None,
                    max_tracks=None):
       confs = None
       if hand_flags is not None:
           confs = [np.full(21, f) for f in hand_flags]
       ...
       self.hand_tracks, smoothed, n_active = self._match_and_smooth(
           ..., confidences=confs,
       )
   ```

3. **`main.py`** — update the `process_frame` call site to
   unpack the new `hand_flags` return value.  Pass
   `hand_flags` to `smoother.smooth_hands()`.

4. **Update all other callers** that unpack the `process_frame`
   return value (search for `process_frame(` in all files
   including `benchmark.py` if needed).  `frame_diag` already
   stores the per-detection hand_flag in `hand_diag`, so no
   change is needed in `metrics.py`.

5. **Test**: extend `test_hand_path_unaffected` in
   `tests/test_smoothing.py` to verify that low hand_flag
   values produce less movement:

   ```python
   def test_hand_confidence_smoothing():
       smoother_hi = PoseSmoother()
       smoother_lo = PoseSmoother()
       lm = [_make_landmarks(21)]

       smoother_hi.smooth_hands(lm, 0.0, hand_flags=[1.0])
       smoother_lo.smooth_hands(lm, 0.0, hand_flags=[0.2])

       shifted = [lm[0] + 30.0]
       r_hi, _ = smoother_hi.smooth_hands(shifted, 0.1,
                                           hand_flags=[1.0])
       r_lo, _ = smoother_lo.smooth_hands(shifted, 0.1,
                                           hand_flags=[0.2])

       move_hi = np.linalg.norm(r_hi[0] - lm[0])
       move_lo = np.linalg.norm(r_lo[0] - lm[0])
       assert move_lo < move_hi
   ```

### Verification

```bash
python3 tests/test_smoothing.py
python3 tests/test_constraints.py
python3 test_extrapolation.py
# All pass.
```

---

## Section 4 — Proportional bone-length correction

### Status: TODO

### Rationale

`BoneLengthSmoother.update()` enforces bone lengths by
projecting only the **distal** keypoint.  When the correction
is large this causes the distal end to jump visually while the
proximal end stays fixed.  Splitting the correction between
both endpoints (weighted toward the distal end) produces
a more natural adjustment.

### Files

- `constraints.py` — `BoneLengthSmoother.update()`
- `tests/test_constraints.py`

### Changes

1. Add a `distal_weight` parameter to `BoneLengthSmoother`
   (default `0.8`).  The remaining `1 - distal_weight` is
   applied to the proximal keypoint.

   In the correction loop, after computing the expected length
   and the overshoot direction, split the displacement:

   ```python
   overshoot = landmarks[d] - (landmarks[p] + direction * expected)
   landmarks[d] -= self.distal_weight * overshoot
   landmarks[p] += (1 - self.distal_weight) * overshoot
   ```

   **Important**: because segments are ordered
   shoulder→outward, a proximal correction on one segment
   affects the next.  The chain already propagates outward, so
   this is consistent — the shoulder moves slightly, which then
   shifts the elbow baseline, which then shifts the wrist.

2. Add `POSE_BENCH_BONE_DISTAL_WEIGHT` environment variable
   override, and register it in `benchmark.py`'s
   `TUNEABLE_PARAMS`.

3. Update tests:
   - `test_constant_landmarks_no_correction` — unchanged
     (no correction means no split).
   - `test_perturbed_keypoint_corrected` — verify **both**
     the distal and proximal keypoints shifted, and the bone
     length is within tolerance.
   - New: `test_proportional_correction_direction` — verify
     the proximal keypoint moved toward the perturbed distal
     keypoint (not away from it).

### Verification

```bash
python3 tests/test_constraints.py
# All tests pass including new test.
python3 tests/test_smoothing.py
python3 test_extrapolation.py
# Unchanged tests still pass.
```

---

## Section 5 — Configurable carry-forward damping

### Status: TODO

### Rationale

The velocity damping factor in `PoseSmoother._extrapolate()`
is hardcoded to `0.8`.  Making it configurable via environment
variable lets the benchmark sweep test different decay rates,
and a higher value (e.g., 0.9) may improve motion continuity
for slow subjects while a lower value (e.g., 0.6) prevents
drift for fast/erratic motion.

### Files

- `smoothing.py` — `PoseSmoother.__init__()` and
  `_extrapolate()`
- `benchmark.py` — `TUNEABLE_PARAMS`
- `test_extrapolation.py`

### Changes

1. Add `carry_damping` parameter to `PoseSmoother.__init__`:

   ```python
   def __init__(self, match_threshold=150, carry_damping=None):
       ...
       if carry_damping is None:
           carry_damping = float(
               os.environ.get("POSE_BENCH_CARRY_DAMPING", "0.8"))
       self.carry_damping = carry_damping
   ```

2. In `_extrapolate`, replace `0.8 ** misses` with
   `self.carry_damping ** misses`.

3. Register in `benchmark.py`:

   ```python
   "carry_damping": ("smoothing", "carry_damping", 0.8),
   ```

4. Add a test in `test_extrapolation.py`:

   ```python
   def test_damping_factor_configurable():
       s_fast = PoseSmoother(carry_damping=0.5)
       s_slow = PoseSmoother(carry_damping=0.95)
       # ... build velocity for both ...
       # After 3 carry frames, slow damping should have moved
       # further than fast damping.
   ```

### Verification

```bash
python3 test_extrapolation.py
python3 tests/test_smoothing.py
# All pass.
```

---

## Section 6 — Detection-level carry-forward

### Status: TODO

### Rationale

`_smooth_detections()` blends new detections against previous
ones via EMA, but when a detection drops for a single frame the
EMA chain breaks.  On reappearance the detection starts fresh,
causing a transient crop jump that propagates to landmark
jitter.

Adding 1-frame carry-forward at the detection level preserves
the EMA chain through brief SSD dropouts.  This is
complementary to (not redundant with) landmark-level
carry-forward and re-crop fallback, which operate downstream.

### Files

- `processing.py` — `_smooth_detections()`
- `tests/test_detection.py` — **new file**

### Changes

1. `_smooth_detections` currently receives `(new_dets,
   prev_dets, ...)` and returns only matched/new entries.
   Extend it to also carry forward any `prev_dets` that had
   no match in `new_dets`, for one frame only:

   - Track a `carried` flag on each detection dict.
   - Previous detections that are already carried
     (`det.get("_carried")`) are not carried again.
   - Carried detections get `score *= 0.7` (decayed
     confidence) so they lose NMS competition against real
     detections next frame.
   - Add `"_carried": True` to the dict.

2. The returned list is `smoothed + carried_over`.

3. Create `tests/test_detection.py` with:
   - **Stable detection** — two consecutive frames with the
     same detection produce EMA-blended output.
   - **One-frame dropout** — frame 1 has detection, frame 2
     has none: verify the carried detection is returned.
   - **Two-frame dropout** — frame 1 has detection, frames 2
     and 3 have none: verify no carry on frame 3 (only
     1-frame grace).
   - **Carry score decay** — carried detection's score is
     reduced by the decay factor.
   - **Re-acquisition** — after carry, a new matching
     detection blends smoothly with the carried entry.

### Verification

```bash
python3 tests/test_detection.py
python3 tests/test_smoothing.py
python3 tests/test_constraints.py
python3 test_extrapolation.py
# All pass.
```

---

## Section 7 — Benchmark sweep configuration

### Status: TODO

### Rationale

The benchmark infrastructure is built but there is no default
sweep configuration.  A curated YAML config encodes the most
impactful parameter ranges identified from code analysis,
enabling systematic optimisation with a single command.

### Files

- `sweep_default.yaml` — **new file**
- `AGENTS.md` — add sweep config reference

### Changes

1. Create `sweep_default.yaml`:

   ```yaml
   # Default parameter sweep — run with:
   #   python benchmark.py --source video.mp4 --config sweep_default.yaml

   # One-Euro body filter: lower min_cutoff = more smoothing
   body_min_cutoff: [0.15, 0.3, 0.5, 0.8]

   # One-Euro body beta: higher = more responsive to speed
   body_beta: [0.3, 0.5, 0.8]

   # One-Euro hand filter
   hand_min_cutoff: [0.5, 1.0, 2.0]
   hand_beta: [0.2, 0.3, 0.5]

   # Confidence gamma: higher = low-conf keypoints resist more
   confidence_gamma: [1.5, 2.0, 3.0]

   # Detection smoothing EMA weight
   det_smooth_alpha: [0.3, 0.5, 0.7]

   # Bone-length constraint tolerance
   bone_tolerance: [0.25, 0.4, 0.6]

   # Carry-forward grace period (frames)
   carry_grace: [5, 10, 15]
   ```

   **Note**: running the full Cartesian product is
   prohibitively large (4×3×3×3×3×3×3×3 = 8748 combos).
   The YAML is a reference; actual sweeps should target
   1-2 parameters at a time.

2. Create `sweep_quick.yaml` for a targeted first pass:

   ```yaml
   # Quick sweep: body smoothing + detection EMA only
   body_min_cutoff: [0.15, 0.3, 0.5]
   det_smooth_alpha: [0.3, 0.5]
   ```

3. Update AGENTS.md to reference the sweep configs.

### Verification

```bash
# Dry run: verify YAML parses and grid generates correctly
python3 -c "
import yaml, json
with open('sweep_quick.yaml') as f:
    spec = yaml.safe_load(f)
print(json.dumps(spec, indent=2))
print(f'{len(spec)} parameters')
"
```

---

## Section 8 — Expand test coverage

### Status: TODO

### Rationale

Critical pipeline components lack unit tests: synthetic hand
generation, landmark re-crop, and detection smoothing.
Coverage gaps mean regressions from earlier sections may go
undetected.

### Files

- `tests/test_detection.py` — extend (if created in Sec 6)
  or create
- `tests/test_processing.py` — **new file**

### Changes

1. **`tests/test_processing.py`** — tests for helper functions
   in `processing.py`:

   a. `test_synthesise_hand_from_arm` — provide a single body
      landmark array with known arm geometry; verify the
      synthetic detection's box centre is ~40% of forearm
      length beyond the wrist, and the box size is ~80% of
      forearm length.

   b. `test_synthesise_skips_covered_wrist` — provide a body
      landmark plus a real palm detection near the wrist;
      verify no synthetic is generated (overlap suppression).

   c. `test_recrop_from_landmarks` — provide previous hand
      landmarks and no real palm detections; verify a re-crop
      detection is returned with correct centre and size.

   d. `test_recrop_skips_covered_hand` — provide previous hand
      landmarks plus a real palm detection nearby; verify no
      re-crop is generated.

   e. `test_affine_matrix_degenerate` — zero-size crop, NaN
      inputs, etc. should return None.

2. **`tests/test_detection.py`** — if not already created in
   Section 6, create with tests for `_smooth_detections`
   as described there.  Additionally:

   a. `test_nms_no_overlap` — two non-overlapping boxes both
      survive.

   b. `test_nms_full_overlap` — two identical boxes: only the
      higher-scored one survives.

   c. `test_decode_no_detections` — scores below threshold
      produce empty list.

### Verification

```bash
python3 tests/test_processing.py
python3 tests/test_detection.py
python3 tests/test_matching.py
python3 tests/test_smoothing.py
python3 tests/test_constraints.py
python3 test_extrapolation.py
# All pass.
```

---

## Section 9 — Housekeeping

### Status: TODO

### Rationale

After all code changes, documentation and metadata must stay
in sync.

### Files

- `AGENTS.md`
- `README.md`
- `.gitignore`

### Changes

1. Update `AGENTS.md`:
   - Add `sweep_default.yaml` and `sweep_quick.yaml` to the
     optimisation suite list.
   - Add `carry_damping` and `bone_distal_weight` to the
     tuneable parameter notes.
   - Mention the new test files.

2. Update `README.md`:
   - Add the new tuneable parameters to the parameter table.
   - Add the sweep config files to the usage examples.
   - Add the new matching algorithm note.

3. Verify `.gitignore` covers `benchmark_output/`, `output/`,
   and `__pycache__/`.

4. Remove this file (`PLAN.md`) once all sections are done.

### Verification

```bash
git diff --stat   # review all changed files
git status        # nothing untracked except intended new files
```

---

## Commit guidance

Each completed section should be committed separately:

```
Sec 1: Fix extrapolation test API mismatch
Sec 2: Use Hungarian matching for hands-to-arms
Sec 3: Add confidence-weighted hand smoothing
Sec 4: Proportional bone-length correction
Sec 5: Configurable carry-forward damping
Sec 6: Add detection-level carry-forward
Sec 7: Add benchmark sweep configurations
Sec 8: Expand unit test coverage
Sec 9: Update documentation
```

Keep commit subjects under 50 characters and body lines under
72 characters per project convention.
