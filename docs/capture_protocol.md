# 3-camera capture & QA protocol

How to record a calibration session and a subject session that the 3D
clinical pipeline can validate, and how to grade a capture **before** its
clinical metrics are trusted. The automated gate is
`pose-estimation-validate --qa-only` (see `.claude/tech/validation.md` →
*qa_check*); this document is the human procedure behind it.

Two recordings make one study session:

1. a **calibration session** — the ChArUco board swept through the volume;
2. a **subject session** — the patient performing the task script.

Both are ordinary multi-camera sessions (`.claude/tech/multicam.md`): a
directory of `cam*.{mp4,avi,…}` clips, optionally a `session.json` manifest.

---

## 1. Physical setup

**Cameras (N = 3).** One camera is the **world-frame** camera (defines the
origin); the deployed default is `cam1`.

- **Placement / FOV overlap.** Arrange the three cameras around the working
  volume so the subject's tracked region (torso + both arms, or the whole
  body) is visible to **≥ 2 cameras at all times** — every keypoint needs
  two views to triangulate, and a spare third view lets fusion reject an
  outlier (`fusion.n_views_median` wants ≥ 3). Aim for ~45–90° between
  adjacent optical axes: too small a baseline triangulates poorly (depth
  is ill-conditioned), too large loses shared coverage and worsens
  cross-view matching.
- **World-frame camera is level.** Trunk lean/rotation assume world "up" =
  −y of the world camera (`.claude/tech/analysis.md`). Mount the world
  camera level (spirit level / tripod bubble); a tilted reference biases
  every trunk angle. This is a standing clinical-validity gap until a
  gravity reference is added — keep it level.
- **Working volume.** Define the box the subject moves in and place cameras
  so it sits comfortably inside every frame with margin (subjects drift).
  Keep the subject ≳ 1.5 m from each camera to limit lens distortion at the
  frame edges.
- **Rigidity.** Cameras must not move between the calibration capture and
  the subject capture — extrinsics are solved once and reused. Lock tripods;
  re-calibrate after any bump.
- **Frame rate.** Record all cameras at the **same nominal fps** (≥ 30 fps;
  60 fps for fast reaching). Clinical kinematics need temporal resolution
  for smoothness/velocity metrics.
- **Lighting.** Bright, diffuse, flicker-free. Avoid backlighting (windows
  behind the subject), hard shadows, and rolling-shutter banding under
  mains-frequency lights. Even illumination across the volume keeps 2D
  detection confidence above the floor on every camera.
- **Shutter / sync.** No hardware genlock is assumed — sync is software-only
  (`.claude/tech/multicam.md`). Use a **global-shutter** camera if available
  (rolling shutter smears fast motion and desyncs rows). Align clips by one
  of: (a) a shared visual cue at the start (a clap/marker all cameras see)
  trimmed via per-camera `sync_offset` in `session.json`; or (b) starting
  all recorders together and trusting frame-index alignment. Sub-frame
  desync degrades reprojection on fast motion — the QA gate's frame-count
  parity check is the desync proxy.

---

## 2. ChArUco calibration capture

Produces per-camera intrinsics + extrinsics. Board geometry defaults
(`charuco.py`): 6×9 squares, `DICT_4X4_250`, 40 mm squares, 30 mm markers.

**Print the board.** `pose-estimation-calibrate board --output board.png`
then print at **100 % scale** and verify one square with a ruler (a
mis-scaled board silently corrupts metric units). Mount it rigidly flat
(foam-board / clipboard) — any flex breaks the planar assumption.

**Record.** Use `pose-estimation-calibrate capture --devices 0,1,2` (SPACE
appends one synchronized frame per camera — frame index = press index, so
the clips are inherently synchronized) or record freely and align later.
Then **move the board through the entire working volume** while pressing:

- **Translation diversity** — visit the centre, all four corners, near and
  far planes of the volume. A board confined to the centre weakly
  constrains oblique cameras' intrinsics and couples focal-length error
  into the stereo translation (`.claude/memory/lessons.md` 2026-06-08). The
  QA `board_coverage` metric grades how much of each frame the board swept.
- **Tilt diversity** — also rotate the board (pitch/yaw/roll, ~±30°) at each
  location, not just translate it. Tilt variety is what separates focal
  length from distance in the solve.
- **Scale** — keep the board large enough in frame: **≳ 25 px per square**
  (boards at 2+ m on a 1080p camera fall below this and stop detecting).
- **Topology (hard requirement)** — extrinsics are solved as **direct pairs
  against the world camera only**; chained A↔B↔C is unsupported
  (`.claude/tech/calibration.md`). So the **world camera must co-see the
  board simultaneously with each other camera** for enough frames
  (≥ `MIN_SHARED_FRAMES`). Sweep the board through the overlap region of
  world∩cam2 and world∩cam3 deliberately.
- **Count** — collect well above the floor: ≥ ~25 usable board views per
  camera (the solver needs ≥ 8 for intrinsics; more is better).

**Solve.** `pose-estimation-calibrate solve --session-dir <calib_dir>
--output calibration.json`. Check the reported global reprojection RMS
(target < 1 px, usable < 2 px). Re-sweep and re-solve if high.

---

## 3. Subject task script

Record the patient performing a **structured, repeatable** task so trials
are comparable across sessions (longitudinal tracking) and within a session
(repeatability is the strongest evidence absent a ground-truth baseline —
`.claude/tech/validation.md` gap register).

Per trial:

1. **Rest hold** (~2 s, still) — anchors the temporal-jitter / rest-period
   metrics. The subject holds the start posture motionless.
2. **Task** — the clinical movement, e.g. seated forward reach-grasp-
   transport-release, or a bilateral arm-raise. Keep the script fixed across
   sessions; the R segmentation classifies reach/grasp/transport/release
   phases (`.claude/tech/analysis.md`).
3. **Return to rest** (~2 s, still).
4. **Repeat** the identical trial ≥ 3× per subject — repeated identical
   trials feed the inter-trial repeatability (ICC / CoV) evidence.

Keep the subject within the calibrated working volume and facing the world
camera. One person in frame only — fusion uses `person_idx == 0`; a second
person in view is a scope limit, not handled.

---

## 4. Per-capture acceptance checklist

Run the automated gate first, then confirm by eye:

```bash
pose-estimation-validate --session-dir <subject_dir> \
    --calibration <calib_dir> --qa-only --out qa.json --markdown qa.md
```

Exit code **0** = PASS/WARN (usable), **1** = FAIL (recapture), **2** =
harness error. The gate grades (thresholds + rationale in
`.claude/tech/validation.md`):

- [ ] **Calibration RMS** within band (< 1 px ideal, < 2 px usable).
- [ ] **Board coverage** — each camera's sweep lit up enough of the frame
      (no centre-bound capture). WARN ⇒ sweep wider and re-solve.
- [ ] **ChArUco detection** — enough usable board views per camera (above
      the intrinsic floor).
- [ ] **Frame-count parity** — cameras recorded ~equal frame counts (desync
      proxy). A large mismatch ⇒ a camera dropped frames or started late.
- [ ] **Subject 2D detection** — the subject is tracked in most frames of
      every camera; low-confidence fraction within band.

Manual confirmation (the gate cannot see these):

- [ ] World camera **level** (trunk-angle validity).
- [ ] Cameras **unmoved** since calibration.
- [ ] Subject **inside the working volume** for the whole trial, visible to
      ≥ 2 cameras throughout.
- [ ] **One** person in frame.
- [ ] Lighting even; no backlight/flicker; no motion blur on fast phases.

Any FAIL (or an unchecked manual item) ⇒ recapture before trusting clinical
metrics.

---

## 5. Anonymization & data-sharing strategy

Patient video is biometric. The rule is **raw imagery never leaves the
capture host; only de-identified derived coordinates are shareable.**

**Never committed (already enforced by `.gitignore`):**

- Raw patient video (`videos/` is git-ignored; it is a symlink, and git
  refuses to traverse it).
- Pipeline output for real subjects (`output/`).
- Calibration is patient-adjacent (it encodes identifying lab/rig geometry):
  `calibration/` and `calibration.json` are git-ignored at **any depth** by
  default (decision 2026-06-15).

**Shareable fixtures = de-identified derived artifacts only:**

- Per-camera keypoint **CSVs**, `calibration.json`, and `world3d.csv` —
  these are coordinates and camera parameters, **no imagery**.
- Any **committed** calibration fixture must be a vetted, de-identified file
  added against the default-deny ignore via a scoped negation
  (`!tests/fixtures/<x>/calibration.json`) or `git add -f` — a deliberate,
  reviewed act, never automatic.
- Retained **imagery** in the repo must be **synthetic** only — a rendered
  ChArUco board or a synthetic skeleton (as the test fixtures already do);
  **never** patient frames.
- **Strip capture metadata** (camera serials, timestamps, GPS/EXIF, file
  paths that name the patient) from any shared artifact.
- Derive fixtures **only from recordings the subject consented** to share,
  under the study's ethics approval.

This is the source of truth for Session 2C's anonymized-fixture step; see
`.claude/memory/decisions.md` for the recorded decision.
