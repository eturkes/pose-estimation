# Session prompts

Paste the kickoff prompt from `prompts/kickoff.md`, then append one of the task blocks below as the `<TASK>` section. Sessions are designed to run autonomously with minimal user input.

## Completed roadmaps

### Clinical Pipeline E2E (2026-05-24) — all 8 tasks done

1. COCO-WholeBody -> MediaPipe keypoint mapping
2. Wire CSV export into rtmlib process_source()
3. Test rtmlib CSV schema compat with R pipeline
4. Harden R scripts for edge cases
5. E2E clinical pipeline smoke test
6. Dependency update + security audit
7. Refactor main.py/run.py (analysis: not worthwhile)
8. Tech notes drift audit

## Current roadmap: Stability + Clinical Metrics + 3D Pipeline (2026-05-24)

Four phases. Phases 1-2 are the immediate priority (jitter fixes + clinical metrics). Phase 3 (3D pipeline) starts when 3-cam footage timeline firms up (~weeks away). Phase 4 is periodic maintenance.

### Phase 1: Tracking stability ✓
- 1A: ✓ Investigate and fix remaining jitter/drops (all backends/modes)
- 1B: ✓ Adaptive smoothing based on movement phase

### Phase 2: Clinical metrics expansion
- 2A: ✓ Bilateral comparison metrics
- 2B: ✓ Movement quality scores (normalized jerk, movement efficiency, CPI)
- 2C: Trunk/torso metrics — body mode only (independent)
- 2D: Temporal movement segmentation (unblocked — 2B complete)

### Phase 3: 3D pipeline implementation
- 3A: fuse_session_frame() with synthetic tests (independent)
- 3B: solve_charuco() + calibration CLI (independent)
- 3C: 3D CSV export + R analysis (blocked by 3A)

### Phase 4: Maintenance (periodic, interleave freely)
- Dependency update + security audit + tech notes

---

## Session 1A: Investigate and fix tracking jitter/drops

```
Execute: Deep-dive into remaining tracking jitter and detection drops across all backends and modes.

Load tech notes: architecture.md, tracking-modes.md, optimization.md.

Context: Previous session (2026-05-24) added outlier rejection (cap=30px), lowered hand min_cutoff to 0.5, lowered detection EMA alpha to 0.35, extended carry-forward to 3 frames. User reports jitter/drops persist across multiple backends and modes.

1. Profile jitter sources quantitatively. Use available footage to measure:
   - Per-keypoint position variance frame-to-frame (identify worst offenders).
   - Detection-carry oscillation frequency (how often does detection drop and recover?).
   - Confidence distribution per backend — are low-confidence keypoints dragging quality down?
2. Test whether outlier_cap=30 is too permissive for fine hand movements (try 15-20px range).
3. Investigate rtmlib internal smoothing interaction — rtmlib's PoseTracker has its own smoothing that may fight our One Euro filter.
4. Test confidence-gated filtering: keypoints below a visibility threshold get pulled harder toward the previous estimate.
5. Benchmark parameter combinations on available footage; update sweep_default.yaml with findings.
6. Update smoothing.py / run.py / processing.py as needed. Add tests for new behaviors.
7. Update tech notes: optimization.md, tracking-modes.md.
```

---

## Session 1B: Adaptive smoothing based on movement phase

```
Execute: Extend smoothing pipeline with movement-phase-aware parameter adaptation.

Load tech notes: architecture.md, tracking-modes.md, optimization.md.

Prerequisites: Session 1A must be complete (jitter investigation provides the baseline).

Context: Fixed-parameter One Euro filtering trades off jitter (needs low min_cutoff) vs lag (needs high min_cutoff). The clinical use case has distinct phases: near-stationary rest periods (need aggressive smoothing) vs fast reaching/grasping (need minimal lag).

1. Implement velocity-regime detection in PoseSmoother: classify each keypoint's current state as REST / SLOW / FAST based on filtered velocity magnitude and configurable thresholds.
2. Adapt min_cutoff per regime: REST uses a very low cutoff (heavy smoothing), FAST uses the current default. SLOW interpolates.
3. Implement as a wrapper/mixin that PoseSmoother delegates to — preserve the existing non-adaptive path as the default for backwards compatibility.
4. Add env vars for regime thresholds (POSE_BENCH_REST_VELOCITY, POSE_BENCH_FAST_VELOCITY).
5. Test with synthetic trajectories: rest-then-reach-then-rest pattern. Verify smoothing adapts and jitter is reduced during rest without adding lag during reach.
6. Benchmark against 1A results on available footage.
7. Update sweep configs and tech notes.
```

---

## Session 2A: Bilateral comparison metrics

```
Execute: Add left-vs-right symmetry analysis to clinical_features.R.

Load tech notes: analysis.md, tracking-modes.md.

Context: Clinical assessment of upper-limb rehabilitation needs L/R comparison. All existing per-side metrics (elbow flexion, wrist deviation, reach distance, grasp aperture, wrist velocity, displacement) are computed independently per side but never compared.

1. In compute_frame_features(): for each per-side metric, compute:
   - symmetry_ratio = min(L, R) / max(L, R) (1.0 = perfect symmetry, 0 = one side absent).
   - dominance_index = (R - L) / (R + L) (positive = right-dominant, handles division-by-zero).
   - absolute_difference = abs(R - L).
2. In compute_window_features(): aggregate symmetry metrics per window (mean, SD).
3. Handle NA gracefully — if one side is missing, symmetry metrics are NA (already handled by R's NA propagation, but verify).
4. Add to both hands-arms and body modes. Hands-only mode: derive from wrist x-coordinate assignment (less reliable — document this caveat).
5. Update analysis.md with new metric descriptions.
6. Test with edge cases: single-hand data, both hands missing, perfectly symmetric input.
```

---

## Session 2B: Movement quality scores

```
Execute: Add movement quality metrics beyond SAL to clinical_features.R.

Load tech notes: analysis.md, tracking-modes.md.

Context: SAL (spectral arc length) is already computed for wrist velocity in windows. Clinical literature uses additional smoothness/quality measures: normalized jerk, movement efficiency, and compensatory pattern detection.

1. Implement normalized_jerk(): dimensionless jerk metric. Compute as:
   NJ = sqrt(0.5 * integral(jerk^2 dt) * duration^5 / amplitude^2)
   where jerk = d^3(position)/dt^3, duration = window length, amplitude = path length.
   Lower NJ = smoother movement. Apply to wrist and fingertip trajectories.
2. Implement movement_efficiency(): path_length / straight_line_distance for each movement window. 1.0 = perfectly straight; higher = more curved/corrective. Apply to wrist trajectory.
3. Implement compensatory_pattern_index() (body mode only): within each window, compute correlation between trunk lean angle and reach distance. High positive correlation suggests trunk compensation for limited shoulder/elbow ROM. Skip in hands-arms mode (no hip keypoints for trunk lean).
4. Improve SAL: verify frequency cutoff (currently uses default) matches clinical literature recommendations (typically 10-20 Hz cutoff for upper limb). Add cutoff parameter to spectral_arc_length().
5. Add all metrics to compute_window_features(). normalized_jerk and movement_efficiency also work per-frame (using a short sliding window).
6. Update analysis.md. Test with synthetic smooth and jerky trajectories.
```

---

## Session 2C: Trunk/torso metrics (body mode)

```
Execute: Add trunk and postural metrics to clinical_features.R, gated to body tracking mode.

Load tech notes: analysis.md, tracking-modes.md.

Context: Body mode tracks 33 keypoints including left_hip (idx 23) and right_hip (idx 24). These enable trunk/postural analysis essential for rehabilitation: trunk compensation is a primary concern in upper-limb rehab (patients lean forward or laterally to compensate for limited arm ROM).

Important: hands-arms mode (12 keypoints: shoulders through finger bases) has NO hip keypoints. All trunk metrics must be gated behind detect_tracking() returning "body". Emit an informative skip message for other modes.

1. Implement trunk_lean_sagittal(): angle between the shoulder-midpoint→hip-midpoint line and vertical (positive Y axis in image coords, remembering +Y is down). In degrees; 0 = upright, positive = leaning forward.
2. Implement trunk_lean_lateral(): frontal-plane component of the same vector. Positive = leaning right.
3. Implement trunk_rotation(): angle between the shoulder line (left_shoulder→right_shoulder) and the hip line (left_hip→right_hip) projected onto the transverse plane. 0 = shoulders aligned with hips.
4. Implement posture_symmetry(): (left_shoulder_y - right_shoulder_y) / shoulder_width. Captures shoulder drop/elevation asymmetry.
5. Add all to compute_frame_features() inside a body-mode gate. Add windowed summaries (mean, SD, range) to compute_window_features().
6. Update analysis.md with metric definitions and body-mode-only caveat.
7. Test with edge cases: missing hip keypoints, upright vs leaning poses.
```

---

## Session 2D: Temporal movement segmentation

```
Execute: Add automatic movement phase detection and per-phase feature extraction to clinical_features.R.

Load tech notes: analysis.md, tracking-modes.md.

Prerequisites: Sessions 2A (bilateral) and 2B (quality scores) should be complete — segmentation benefits from having symmetry ratios and quality metrics available for per-phase extraction.

Context: Clinical trials involve structured tasks (reach-grasp-transport-release). Automated segmentation enables per-phase analysis: "is the reach phase getting smoother over sessions?" rather than whole-trial averages that dilute phase-specific improvements.

1. Implement segment_movements(): velocity-profile-based segmentation of wrist trajectory.
   - Compute wrist speed (Euclidean norm of velocity vector).
   - Smooth speed with a short median filter to suppress noise.
   - Detect movement onset/offset via threshold crossing (e.g., 5% of peak speed within a window).
   - Within each movement: sub-segment into REACH (hand moving away from body, aperture opening), GRASP (aperture closing), TRANSPORT (hand moving with closed aperture), RELEASE (aperture opening while stationary or returning).
   - Use grasp_aperture (thumb-index distance) derivative to distinguish reach from transport.
2. State machine: REST -> REACH -> GRASP -> TRANSPORT -> RELEASE -> REST.
   - Allow transitions to skip states (e.g., reach without grasp for pointing tasks).
   - Configurable velocity and aperture thresholds via function parameters.
3. Output: *_movement_phases.csv with columns: video, movement_idx, phase, start_frame, end_frame, duration_sec, peak_velocity, path_length, smoothness (NJ and SAL), mean_symmetry_ratio (if bilateral metrics available).
4. Per-movement summary: total movement time, number of phases detected, movement efficiency.
5. Update analysis.md with the segmentation algorithm description.
6. Test with synthetic reach-grasp-release trajectories. Verify correct phase boundaries.
```

---

## Session 3A: Implement fuse_session_frame()

```
Execute: Implement the 3D triangulation fusion policy layer in triangulation.py.

Load tech notes: architecture.md, multicam.md, calibration.md.

Context: Math primitives (projection_matrix, undistort_points, triangulate_views) are implemented and tested. The stub fuse_session_frame() needs the policy layer: which views to use, confidence weighting, missing-view handling, outlier rejection. Can be fully implemented and tested with synthetic multi-view data (no real footage needed).

1. Implement fuse_session_frame():
   a. For each camera, undistort its 2D keypoints using undistort_points().
   b. Build projection matrices via session_projection_matrices().
   c. Assemble per-view arrays for triangulate_views(), using confidence scores as weights.
   d. Handle missing keypoints: if a camera has NaN/zero-confidence for a keypoint, exclude that view for that keypoint. Require min 2 views per keypoint; NaN for insufficient views.
   e. Cheirality check: after triangulation, project each 3D point back into each camera and verify it's in front (Z > 0 in camera frame). Flag violations.
   f. Compute per-keypoint reprojection error (mean across views).
2. Return: np.ndarray (N, 3) world-space keypoints + metadata dict with per-keypoint confidence and reprojection error.
3. Wire into process_session(): after per-camera 2D processing completes, if calibration is present, call fuse_session_frame() for each synchronized frame.
4. Tests: create synthetic 3D skeleton, project through 3 known cameras with noise, triangulate, verify reconstruction error < 5mm. Test with 1 camera missing, with occluded keypoints, with an outlier view.
5. Update architecture.md and multicam.md (remove "stub" references).
```

---

## Session 3B: Implement solve_charuco() and calibration CLI

```
Execute: Implement ChArUco-based camera calibration solver.

Load tech notes: calibration.md, multicam.md.

Context: calibration.py has IO + validation working. solve_charuco() and calibration_cli.py solve/capture subcommands are stubs. Can be implemented and tested with synthetically rendered ChArUco board images.

1. Define CHARUCO_BOARD_DEFAULT: cv2.aruco.CharucoBoard with sensible defaults (6x9, DICT_4X4_250, 40mm squares, 30mm markers). Make configurable via function params.
2. Implement _detect_charuco_corners(): given a video path, iterate frames, detect ChArUco corners via cv2.aruco.CharucoDetector. Return per-frame corner arrays + IDs. Subsample to ~50 well-distributed frames.
3. Implement _solve_intrinsics(): per-camera cv2.aruco.calibrateCameraCharucoExtended -> K, distortion, per-frame rvec/tvec.
4. Implement _solve_extrinsics(): for camera pairs that share board-visible frames, compute relative pose via cv2.stereoCalibrate. Express all cameras relative to world_frame camera.
5. Assemble into solve_charuco(): iterate cameras, solve intrinsics, solve pairwise extrinsics, compute reprojection error, return SessionCalibration.
6. Wire calibration_cli.py:
   - solve: call solve_charuco(), save result, print summary.
   - capture: open camera feeds, overlay detected corners in real-time (pygame display), save frames on keypress.
7. Test with synthetic renders: create a ChArUco board image, warp it with known camera matrices, verify solve recovers the matrices within tolerance.
8. Update calibration.md (replace "planned" with "implemented").
```

---

## Session 3C: 3D CSV export schema + R analysis extension

```
Execute: Design 3D output format and extend R analysis for 3D keypoints.

Load tech notes: architecture.md, multicam.md, analysis.md, tracking-modes.md.

Prerequisites: Session 3A (fuse_session_frame) must be complete.

1. Define world3d.csv schema in export.py (new function):
   - Metadata: video/session_id, frame_idx, timestamp_sec, person_idx.
   - Per-keypoint: {name}_x_m, {name}_y_m, {name}_z_m, {name}_confidence, {name}_reproj_err_px.
   - Keypoint names match 2D schema (arm_* or body_* depending on mode).
   - Units: metres for xyz, pixels for reproj_err.
2. Wire export into process_session(): after fuse_session_frame(), write world3d.csv.
3. Extend clinical_features.R:
   - Detect 3D input (check for _x_m columns).
   - 3D joint angles: use true 3D vectors for angle_at_vertex() instead of 2D projections. More accurate for out-of-plane movements.
   - 3D reach distance: true Euclidean distance in metres (not normalised pixel distance).
   - 3D velocity: in m/s (physically meaningful, comparable across camera setups).
   - 3D trunk metrics: true sagittal/frontal/transverse plane decomposition using world coordinates.
4. Output: *_clinical_3d.csv alongside existing *_clinical.csv.
5. Update analysis.md, multicam.md, architecture.md.
6. Test with synthetic 3D data (known joint angles, verify computed angles match).
```

---

## Session 4: Maintenance cycle

```
Execute: Periodic maintenance — dependency updates, security audit, tech notes drift check.

Load tech notes: environment.md, architecture.md, tests.md.

1. Update Python dependencies: uv lock --upgrade, then uv sync. Verify all tests pass.
2. Update R packages: renv::update(), then renv::snapshot(). Verify R scripts exit 0.
3. Security audit:
   - Review CLI argument handling for injection vectors.
   - Review session.json / calibration.json parsing for path traversal (recheck _safe_resolve coverage).
   - Check for new CVEs in openvino, onnxruntime, opencv, rtmlib.
4. Run full test suite: uv run pytest. Run linter: uv run ruff check. Run type checker: uv run ty check.
5. Tech notes drift audit: verify all .claude/tech/*.md files match current code (module map, CLI flags, test inventory, API surface).
6. Update .claude/memory/scratchpad.md with maintenance log.
7. Commit all changes.
```
