# Architecture

## Module map (`src/pose_estimation/`)

| File | Role |
|------|------|
| `main.py` | MediaPipe pipeline entry point; CLI, capture loop, pygame display, CSV export. |
| `run.py` | Unified entry point with rtmlib backends (RTMW-L, DWPose-M, RTMPose-M) + MediaPipe delegate. Holds `MODEL_REGISTRY`, CLI, capture loop. Smoother → `rtmlib_smoothing.py`; OpenVINO patch → `rtmlib_openvino.py` (both re-imported). |
| `models.py` | Downloads MediaPipe TFLite, converts to OpenVINO IR, compiles. Checksum-validated. |
| `detection.py` | SSD anchor generation, NMS, decoding. |
| `processing.py` | Preprocess, crop, landmark inference, hands→arms matching, `process_frame`, `tracking_pose_indices()`, `select_primary_body`. |
| `drawing.py` | Catmull-Rom splines, skeleton rendering, overlay blending. |
| `smoothing.py` | One Euro Filter (`OneEuroFilter`, `PoseSmoother`) — confidence-weighted temporal smoothing with velocity-aware outlier rejection and adaptive min_cutoff (heavier smoothing during rest, normal during movement). MediaPipe path. |
| `rtmlib_smoothing.py` | rtmlib-path keypoint smoother: `_OneEuro`, `KeypointSmoother`, `REGION_PARAMS`, `_KP_*`. Parallel to `smoothing.py`; re-exported from `run.py` (tests import via `pose_estimation.run`). |
| `rtmlib_openvino.py` | Self-contained OpenVINO-backend monkey-patch for rtmlib (`_patch_rtmlib_openvino`). No `run.py` globals. |
| `constraints.py` | `BoneLengthSmoother`, `clamp_joint_angles`, `BONE_SEGMENTS{,_BODY}`, `ANGLE_LIMITS{,_BODY}`. |
| `mapping.py` | COCO-WholeBody → MediaPipe keypoint schema mapping (`coco_to_mediapipe`). Translates rtmlib output to `frame_to_rows()` interface. |
| `export.py` | CSV schema (`frame_to_rows`, `open_csv_writer`, `wrist_to_side`) + read-back for 3D fusion (`read_csv_keypoints`, incl. timestamps) + world3d.csv writer (`make_world3d_header`, `write_world3d_csv` — duck-typed, no multicam import). |
| `postprocess.py` | Savitzky-Golay offline smoothing (`savgol_smooth_csv`). |
| `metrics.py` | `MetricsCollector`, `ConstraintDiagnostics`, `SmoothingDiagnostics` — per-frame quality metrics. |
| `benchmark.py` | Parameter sweep harness (subprocess fan-out, `--config` YAML). |
| `multicam.py` | Multi-camera `Session` discovery + synchronized iteration. `process_session` orchestrates per-camera processing via callback, then 3D-fuses CSVs when calibration present (`fuse_session_outputs`, `SessionFusion`) and writes `world3d.csv`. See `tech/multicam.md`. |
| `calibration.py` | Camera-calibration JSON IO + validation (cv2-free). See `tech/calibration.md`. |
| `charuco.py` | ChArUco board build/render, corner detection, `solve_charuco` (intrinsics + pairwise extrinsics + global RMS). See `tech/calibration.md`. |
| `calibration_cli.py` | `pose-estimation-calibrate` console script (`verify`/`solve`/`board`/`capture`). |
| `triangulation.py` | 3D triangulation: math primitives (DLT, projection, undistort) + `fuse_session_frame` policy layer (validity masking, confidence-weighted DLT, min-view enforcement, greedy outlier-view rejection, cheirality flag, reprojection diagnostics). |
| `_types.py` | `Detection`, `HandDetectionDiag`, `PipelineState`, `CameraCalibration`, `SessionCalibration`, `SessionFrame`, `MultiCamPipelineState`, `FusionDiagnostics` TypedDicts. |

## Public API (re-exported from `src/pose_estimation/__init__.py`)

`ANGLE_LIMITS`, `ANGLE_LIMITS_BODY`, `BONE_SEGMENTS`, `BONE_SEGMENTS_BODY`, `TRACKING_BODY`, `TRACKING_HANDS`, `TRACKING_HANDS_ARMS`, `BoneLengthSmoother`, `CalibrationError`, `CameraCalibration`, `Detection`, `FusionDiagnostics`, `HandDetectionDiag`, `MultiCamPipelineState`, `OneEuroFilter`, `PipelineState`, `PoseSmoother`, `Session`, `SessionCalibration`, `SessionCamera`, `SessionError`, `SessionFrame`, `coco_to_mediapipe`, `clamp_joint_angles`, `discover_session`, `discover_sessions`, `download_and_compile_models`, `fuse_session_frame`, `fuse_session_outputs`, `iter_synchronized_frames`, `load_calibration`, `load_session_calibration`, `make_charuco_board`, `match_hands_to_arms`, `process_frame`, `process_session`, `save_calibration`, `savgol_smooth_csv`, `select_primary_body`, `solve_charuco`, `tracking_pose_indices`.

Treat this list as the stable surface. Internal helpers (leading `_`) may move freely.

## Frame pipeline (MediaPipe path, `main.py` + `processing.process_frame`)

1. Capture (BGR) → optional flip → resize.
2. Pose detection (skipped in `hands` mode).
3. Arm-guided hand ROI fallback when palm detection is weak.
4. Landmark inference.
5. Smoothing — `PoseSmoother` (One Euro, confidence-weighted, velocity-aware outlier rejection).
6. Bone-length constraints — `BoneLengthSmoother`.
7. Joint-angle limits — `clamp_joint_angles`.
8. Hand↔arm matching — Hungarian assignment via `scipy.optimize.linear_sum_assignment`, with distality reject (hand closer to shoulder midpoint than wrist).
9. Optional single-subject filter — keeps largest-bbox body, re-indexes hand matches.
10. Draw overlays, BGR→RGB, pygame surface blit.
11. Export row via `export.frame_to_rows`.

## Display backend

- `pygame-ce` (SDL2) — chosen because OpenCV's bundled Qt backend does not render on Wayland.
- `opencv-python-headless` — no GUI module needed.

## Inter-frame state

`PipelineState` (TypedDict): `pose_dets`, `palm_dets`, `hand_diag`. Threaded through `process_frame` calls.

## Cross-references

- Modes: `tech/tracking-modes.md`
- Multi-camera sessions: `tech/multicam.md`
- Calibration format + workflow: `tech/calibration.md`
- Entry points & CLI: `tech/entrypoints.md`
- Tests: `tech/tests.md`
