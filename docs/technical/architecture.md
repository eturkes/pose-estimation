# System architecture

## Module map (`src/pose_estimation/`)

| File | Role |
|------|------|
| `main.py` | MediaPipe pipeline entry point; CLI, capture loop, pygame display, CSV export. |
| `run.py` | Unified entry point with rtmlib backends (RTMW-L, DWPose-M, RTMPose-M) + MediaPipe delegate. Holds `MODEL_REGISTRY`, CLI, capture loop. Smoother → `rtmlib_smoothing.py`; OpenVINO patch → `rtmlib_openvino.py` (both re-imported). |
| `models.py` | Downloads MediaPipe TFLite, converts to OpenVINO IR, compiles. Checksum-validated. |
| `detection.py` | SSD anchor generation and decoding, including MediaPipe-style score-weighted NMS for overlapping boxes and crop keypoints. |
| `processing.py` | Detector letterboxing/de-letterboxing, graph-compatible pose/palm crops, landmark-output decoding, hands→arms matching, `process_frame`, `tracking_pose_indices()`, `select_primary_body`. |
| `drawing.py` | Catmull-Rom splines, skeleton rendering, overlay blending. |
| `assignment.py` | Shared validity-gated Hungarian assignment: maximize the number of admissible pairs, then minimize their total cost. |
| `smoothing.py` | One Euro Filter (`OneEuroFilter`, `PoseSmoother`) — confidence-safe position/velocity filtering, prediction-aware association, x/y outlier caps, carry-forward, and adaptive min_cutoff. `OneEuroFilter` is shared by both paths. |
| `rtmlib_smoothing.py` | rtmlib-path multi-person smoother: confidence-weighted centroids, prediction-aware gated assignment, carry-forward, `REGION_PARAMS`, and per-region `OneEuroFilter`s from `smoothing.py`. Re-exported from `run.py` (tests import via `pose_estimation.run`). |
| `video_io.py` | Shared cv2 video-IO helpers for both entry points, including `SourceTimestampClock`: media presentation timestamps with a deterministic frame/FPS fallback for files, and monotonic elapsed time for live sources. |
| `rtmlib_openvino.py` | Self-contained OpenVINO-backend monkey-patch for rtmlib (`_patch_rtmlib_openvino`). No `run.py` globals. |
| `constraints.py` | Robust clipped bone-length estimates with iterative x/y projection, plus rigid distal-branch joint-angle clamps; exposes `BoneLengthSmoother`, `clamp_joint_angles`, `BONE_SEGMENTS{,_BODY}`, and `ANGLE_LIMITS{,_BODY}`. |
| `mapping.py` | COCO-WholeBody → MediaPipe keypoint schema mapping (`coco_to_mediapipe`), including distinct arm-base and full-body fingertip semantics. Translates rtmlib output to `frame_to_rows()` interface. |
| `export.py` | CSV schema (`frame_to_rows`, `open_csv_writer`, `wrist_to_side`) + read-back for 3D fusion (`read_csv_keypoints`, incl. timestamps) + world3d.csv writer (`make_world3d_header`, `write_world3d_csv` — duck-typed, no multicam import). |
| `postprocess.py` | Savitzky-Golay offline smoothing (`savgol_smooth_csv`). |
| `metrics.py` | `MetricsCollector`, `ConstraintDiagnostics`, `SmoothingDiagnostics` — per-frame quality metrics. |
| `benchmark.py` | Parameter sweep harness (subprocess fan-out, `--config` YAML). |
| `multicam.py` | Multi-camera `Session` discovery + synchronized iteration + CLI session resolution (`resolve_cli_sessions`). `process_session` orchestrates per-camera processing via callback, then 3D-fuses CSVs when calibration present (`fuse_session_outputs`, `SessionFusion`) and writes `world3d.csv`. See `multicam.md`. |
| `calibration.py` | Camera-calibration JSON IO + validation (cv2-free). See `calibration.md`. |
| `charuco.py` | ChArUco board build/render, corner detection, `solve_charuco` (intrinsics + pairwise extrinsics + global RMS). See `calibration.md`. |
| `calibration_cli.py` | `pose-estimation-calibrate` console script (`verify`/`solve`/`board`/`capture`). |
| `triangulation.py` | 3D triangulation: projection/undistortion primitives plus confidence-weighted DLT, two-view minimal-set consensus, bounded geometric refinement, residual/ray-angle gates, cheirality, and diagnostics. |
| `validation.py` | End-to-end pipeline validation harness: `run_validation` orchestrates calibration → 2D tracking → fusion → R clinical metrics on one session and emits a `ValidationReport` (JSON + Markdown). `qa_check` is the pre-flight capture-QA gate (`QAReport`; `--qa-only`). Orchestrates + measures only; reuses the pipeline blocks. `pose-estimation-validate` console script. See `validation.md`. |
| `_types.py` | `Detection`, `HandDetectionDiag`, `PipelineState`, `CameraCalibration`, `SessionCalibration`, `SessionFrame`, `MultiCamPipelineState`, `FusionDiagnostics` TypedDicts. Fusion diagnostics distinguish eligible candidate views from final consensus contributors. |

## Public API (re-exported from `src/pose_estimation/__init__.py`)

`ANGLE_LIMITS`, `ANGLE_LIMITS_BODY`, `BONE_SEGMENTS`, `BONE_SEGMENTS_BODY`, `TRACKING_BODY`, `TRACKING_HANDS`, `TRACKING_HANDS_ARMS`, `BoneLengthSmoother`, `CalibrationError`, `CameraCalibration`, `Detection`, `FusionDiagnostics`, `HandDetectionDiag`, `MultiCamPipelineState`, `OneEuroFilter`, `PipelineState`, `PoseSmoother`, `Session`, `SessionCalibration`, `SessionCamera`, `SessionError`, `SessionFrame`, `coco_to_mediapipe`, `clamp_joint_angles`, `discover_session`, `discover_sessions`, `download_and_compile_models`, `fuse_session_frame`, `fuse_session_outputs`, `iter_synchronized_frames`, `load_calibration`, `load_session_calibration`, `make_charuco_board`, `match_hands_to_arms`, `process_frame`, `process_session`, `save_calibration`, `savgol_smooth_csv`, `select_primary_body`, `solve_charuco`, `tracking_pose_indices`.

Treat this list as the stable surface. Internal helpers (leading `_`) may move freely.

## Frame pipeline (MediaPipe path, `main.py` + `processing.process_frame`)

1. Capture (BGR) → optional flip → resize.
2. Detector preprocessing — aspect-preserving letterbox to 224 px for pose and
   192 px for palm; pose RGB is scaled to `[-1, 1]`, palm RGB to `[0, 1]`.
3. SSD decode — apply the configured score threshold, merge overlapping
   anchor predictions with score-weighted NMS, then remove letterbox padding
   from boxes and detector keypoints before computing image-space crops. Pose
   detection is skipped in `hands` mode.
4. ROI construction follows the model graphs:
   - pose: detector keypoint 0 is the centre, keypoints 0→1 set rotation and
     circle diameter, and the square ROI expands by 1.25;
   - palm: wrist keypoint 0→middle-finger MCP keypoint 2 sets rotation, the
     raw rect shifts by `shift_y=-0.5` toward the fingers, becomes square-long,
     and expands by 2.6.
   Landmark crops replicate source-edge pixels outside the image, matching the
   graph tensor transform; full-frame detector letterboxing remains zero-filled.
   Arm-guided and previous-landmark hand crops remain fallbacks when palm
   detection is weak.
5. Landmark decode — pose x/y receives local 7×7 heatmap refinement;
   visibility and presence logits are sigmoid-decoded and combined
   conservatively. The pose flag and hand-presence flag model outputs are
   already probabilities and are not sigmoid-applied a second time. Inverse
   crop scale is applied to z (including the hand model's 0.4 z
   normalisation), without applying x/y translation or rotation to depth.
   The hand model's third output supplies handedness; labels are swapped for
   unmirrored file/session input to match MediaPipe's mirrored-input contract.
6. Temporal smoothing and association — `PoseSmoother` uses confidence-safe
   One Euro velocity/position updates, prediction-aware validity-gated
   assignment, carry-forward, and an active-plus-dormant `max_tracks` cap in
   single-subject hand tracking. Confidence-zero placeholders do not initialize
   filter coordinates, so the first genuine observation bypasses smoothing and
   outlier caps instead of being pulled toward arbitrary model output. Carried predictions retain geometry for
   internal continuity. Carried bodies emit zero visibility; carried rtmlib
   hands have zero scores, and MediaPipe carried hands are omitted from CSV
   export. Fresh MediaPipe presence and rtmlib per-keypoint scores are retained
   in explicit hand confidence columns.
7. Bone-length constraints — clipped temporal estimates plus iterative x/y
   projections repair linked segments without treating relative z as metric.
8. Joint-angle limits — `clamp_joint_angles` rotates the entire distal branch
   rigidly so a correction does not disconnect fingers or feet.
9. Hand↔arm matching — validity-gated Hungarian assignment with the distality
   rule (hand closer to wrist than shoulder midpoint) built into the feasible
   edge set.
10. Optional single-subject filter — keeps largest-bbox body and re-indexes
    hand matches.
11. Draw overlays, BGR→RGB, pygame surface blit.
12. Export row via `export.frame_to_rows`.

The preprocessing constants, association thresholds, filter settings, and
constraint limits are provisional engineering defaults. These changes were
verified with unit/synthetic fixtures; no sensitive recordings or real
clinical footage were inspected or used to tune them.

## Display backend

- `pygame-ce` (SDL2) — chosen because OpenCV's bundled Qt backend does not render on Wayland.
- `opencv-python-headless` — no GUI module needed.

## Inter-frame state

`PipelineState` (TypedDict): `pose_dets`, `palm_dets`, `hand_diag`. Threaded through `process_frame` calls.

## Cross-references

- Modes: `tracking-modes.md`
- Multi-camera sessions: `multicam.md`
- Calibration format + workflow: `calibration.md`
- Entry points & CLI: `entrypoints.md`
- Validation harness + report schema: `validation.md`
- Tests: `tests.md`
