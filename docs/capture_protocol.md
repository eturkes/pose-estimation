# 3-camera capture and QA protocol

This document tells you how to record a calibration session and a subject
session that the 3D clinical pipeline can validate. It also tells you how to
grade a capture before you trust its clinical metrics. The automated gate is
`pose-estimation-validate --qa-only` (see `docs/technical/validation.md` →
*qa_check*). This document is the human procedure behind that gate.

Two recordings make one study session:

1. **Calibration session.** You sweep the ChArUco board through the volume.
2. **Subject session.** The patient does the task script.

Both are ordinary multi-camera sessions (`docs/technical/multicam.md`). Each
one is a directory of `cam*.{mp4,avi,…}` clips, with an optional
`session.json` manifest.

---

## 1. Physical setup

**Cameras (N = 3).** One camera is the **world camera**, which defines the
origin. The deployed default is `cam1`.

- **Placement and FOV overlap.** Arrange the three cameras around the
  working volume. At least 2 cameras must see the tracked region of the
  subject at all times. That region is the torso and both arms, or the whole
  body. Every keypoint needs two views for triangulation, and a spare third
  view lets fusion reject an outlier (`fusion.n_views_median` wants ≥ 3).
  Aim for 45-90° between adjacent optical axes. A smaller baseline
  triangulates poorly, because depth becomes ill-conditioned. A larger
  baseline loses shared coverage and degrades cross-view matching.
- **World camera level.** Mount the world camera level with a spirit level
  or a tripod bubble. The trunk lean and rotation metrics assume that world
  "up" = -y of the world camera (`docs/technical/analysis.md`). A tilted
  reference biases every trunk angle. This clinical-validity gap stays open
  until a gravity reference exists, so keep the camera level.
- **Working volume.** Define the box in which the subject moves. Place the
  cameras so that the box sits inside every frame with margin, because
  subjects drift. Keep the subject approximately 1.5 m or more from each
  camera, which limits lens distortion at the frame edges.
- **Rigidity.** The cameras must not move between the calibration capture
  and the subject capture. The solver computes the extrinsics once, and the
  pipeline reuses them. Lock the tripods. If a camera moves, calibrate again.
- **Frame rate.** Record all cameras at the same nominal frame rate. Use
  30 fps or more. Use 60 fps for fast reaching. Clinical kinematics need
  that temporal resolution for the smoothness and velocity metrics.
- **Lighting.** Use bright, diffuse, flicker-free light. Avoid backlighting,
  for example a window behind the subject. Avoid hard shadows. Avoid
  rolling-shutter banding under mains-frequency lights. Even illumination
  across the volume keeps the 2D detection confidence above the floor on
  every camera.
- **Shutter and synchronization.** The rig has no hardware genlock. Its frame
  reader uses software synchronization (`docs/technical/multicam.md`). Use a
  **global-shutter** camera when one is available. A rolling shutter smears
  fast motion and shifts rows in time. Give all cameras one shared visual cue,
  such as a clap. Trim only whole-frame pre-roll with
  `session.json:cameras[*].sync_offset`. Alternatively, start all recorders
  together and use frame-index alignment. These methods are coarse capture
  setup. They do not establish sub-frame synchronization. The QA frame-count
  parity check detects gross duration or start differences only. It does not
  prove temporal alignment. `qualification/cameras_qc.csv` publishes the
  measured audio offsets. The fusion frame reader does not apply those offsets
  yet.

---

## 2. ChArUco calibration capture

This capture produces the per-camera intrinsics and extrinsics. The board
geometry defaults (`charuco.py`) are 6×9 squares, `DICT_4X4_250`, 40 mm
squares and 30 mm markers.

**Print the board.** Run `pose-estimation-calibrate board --output board.png`.
Print the image at **100 % scale**. Measure one square with a ruler, because
a mis-scaled board corrupts the metric units silently.
Mount the board flat and rigid, for example on foam-board or a clipboard.
Any flex breaks the planar assumption.

**Record.** Run `pose-estimation-calibrate capture --devices 0,1,2`. Each
SPACE press appends one synchronized frame per camera, and the frame index
equals the press index, so the clips stay synchronized by construction. As an
alternative, record freely and align the clips later. **Move the board
through the entire working volume** while you press SPACE:

- **Translation diversity.** Visit the centre, all four corners, and the
  near and far planes of the volume. A board that stays in the centre
  constrains the intrinsics of the oblique cameras weakly, and couples the
  focal-length error into the stereo translation. The QA `board_coverage`
  metric grades how much of each frame the board swept.
- **Tilt diversity.** Rotate the board at each location through
  approximately ±30° of pitch, yaw and roll. Do not only translate it. Tilt
  variety is what separates focal length from distance in the solve.
- **Scale.** Keep the board large in the frame: **25 px per square or
  more**. A board at 2 m or more from a 1080p camera falls below that limit,
  and the detector stops finding it.
- **Topology (hard requirement).** The solver computes the extrinsics as
  **direct pairs against the world camera only**, and does not support a
  chained A↔B↔C rig (`docs/technical/calibration.md`). Therefore the
  **world camera must see the board at the same time as each other
  camera**. That shared view must last `MIN_SHARED_FRAMES` frames or more.
  Sweep the board deliberately through the overlap of world∩cam2 and the
  overlap of world∩cam3.
- **Count.** Collect approximately 25 usable board views per camera or more,
  which stays well above the floor. The solver needs 8 views for the
  intrinsics, and more views give a better result.

**Solve.** Run `pose-estimation-calibrate solve --session-dir <calib_dir>
--output calibration.json`. Read the global reprojection RMS that the solver
reports. Target less than 1 px. Accept up to 2 px. If the RMS is high, sweep
the board again. Then solve again.

---

## 3. Subject task script

Record the patient during a **structured, repeatable** task. A fixed task
makes the trials comparable across sessions, which supports longitudinal
tracking. It also makes the trials comparable within one session.
Repeatability is the strongest evidence available without a ground-truth
baseline (see the gap register in `docs/technical/validation.md`).

Per trial:

1. **Rest hold** (approximately 2 s, still). The subject holds the start
   posture without motion. This hold anchors the temporal-jitter and
   rest-period metrics.
2. **Task.** The subject does the clinical movement, for example a seated
   forward reach-grasp-transport-release, or a bilateral arm-raise. Keep the
   script fixed across sessions. The R segmentation classifies the reach,
   grasp, transport and release phases (`docs/technical/analysis.md`).
3. **Return to rest** (approximately 2 s, still).
4. **Repeat** the identical trial three times or more per subject. Repeated
   identical trials feed the inter-trial repeatability evidence (ICC and
   CoV).

Keep the subject inside the calibrated working volume. Keep the subject
facing the world camera. Record one person only, because fusion uses
`person_idx == 0`. A second person in view is a scope limit, and the
pipeline does not handle it.

---

## 4. Per-capture acceptance checklist

Run the automated gate first. Then confirm the manual items by eye:

```bash
pose-estimation-validate --session-dir <subject_dir> \
    --calibration <calib_dir> --qa-only --out qa.json --markdown qa.md
```

The exit code is **0** for PASS or WARN (usable), **1** for FAIL
(recapture), and **2** for a harness error. The gate grades the items below.
For the thresholds and the rationale, see `docs/technical/validation.md`.

- [ ] **Calibration RMS** inside the band (less than 1 px is ideal, less
      than 2 px is usable).
- [ ] **Board coverage.** The sweep of each camera lit up enough of the
      frame, with no centre-bound capture. If the gate reports WARN, sweep
      wider and solve again.
- [ ] **ChArUco detection.** Each camera has enough usable board views,
      above the intrinsic floor.
- [ ] **Frame-count parity.** Treat similar counts as coarse capture QA only.
      A large mismatch can mean that one camera dropped frames or started late.
      Similar counts do not prove temporal alignment.
- [ ] **Subject 2D detection.** Every camera tracks the subject in most
      frames, and the low-confidence fraction stays inside the band.

Confirm these items yourself, because the gate cannot see them:

- [ ] The world camera is **level** (trunk-angle validity).
- [ ] No camera moved since the calibration.
- [ ] The subject stays **inside the working volume** for the whole trial,
      and at least 2 cameras see the subject throughout.
- [ ] **One** person is in the frame.
- [ ] The lighting is even, with no backlight, no flicker, and no motion
      blur on the fast phases.

If the gate reports any FAIL, or if a manual item stays unchecked, record
the session again. Trust the clinical metrics only after every item passes.

---

## 5. Anonymization and data-sharing strategy

Patient video is biometric data. One rule governs it: **raw imagery never
leaves the capture host, and only de-identified derived coordinates are
shareable.**

**Never commit these (`.gitignore` already enforces the rule):**

- Raw patient video. `videos/` is git-ignored, and git refuses to traverse
  it because it is a symlink.
- Pipeline output for real subjects (`output/`).
- Calibration data, which is patient-adjacent because it encodes identifying
  lab and rig geometry. `.gitignore` excludes `calibration/` and
  `calibration.json` at **any depth** by default.

**Share de-identified derived artifacts only:**

- Per-camera keypoint **CSV files**, `calibration.json` and `world3d.csv`
  hold coordinates and camera parameters, with **no imagery**.
- Every **committed** calibration fixture must be a vetted, de-identified
  file. Add it against the default-deny ignore with a scoped negation
  (`!tests/fixtures/<x>/calibration.json`) or with `git add -f`. This act
  must stay deliberate and reviewed, and never automatic.
- Retained **imagery** in the repo must be **synthetic** only, for example a
  rendered ChArUco board or a synthetic skeleton, as the test fixtures
  already are. Never retain patient frames.
- **Strip the capture metadata** from every shared artifact. That metadata
  covers camera serials, timestamps, GPS and EXIF data, and file paths that
  name the patient.
- Derive a fixture **only from a recording that the subject consented** to
  share, under the ethics approval of the study.

This section is the source of truth for any review of a real-data fixture.
