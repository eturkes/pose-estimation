# Tests

All tests live in `tests/`. Run with `uv run pytest`. Pytest is configured strict (warnings → errors).

## Core pipeline tests

| File | Covers |
|------|--------|
| `tests/test_smoothing.py` | `OneEuroFilter`, hand and body smoothing paths, adaptive min_cutoff (rest/fast regime, transition, per-keypoint independence, disable-via-None). |
| `tests/test_constraints.py` | Bone-length smoothing, joint-angle clamping. |
| `tests/test_matching.py` | Hungarian hand→arm matching, distality reject. |
| `tests/test_detection.py` | NMS, detection EMA smoothing, carry-forward. |
| `tests/test_processing.py` | Synthetic hands, re-crop, affine helpers. |
| `tests/test_extrapolation.py` | Carry-forward extrapolation behaviour. |

## RTMW / rtmlib path

| File | Covers |
|------|--------|
| `tests/test_rtmlib_csv_export.py` | `run.py` CSV export via `process_source()`: schema correctness, row counts, coordinate normalization, video_name handling. Uses mock tracker + synthetic video. |
| `tests/test_rtmw_age_gating.py` | Age-based gating of stale detections. |
| `tests/test_rtmw_confidence.py` | Confidence handling for the 133-keypoint output. |
| `tests/test_rtmw_constraints.py` | Constraint behaviour on rtmlib keypoints. |
| `tests/test_rtmw_extrapolation.py` | Carry-forward for rtmlib path. |
| `tests/test_rtmw_matching.py` | Matching specific to rtmlib outputs. |
| `tests/test_rtmw_regions.py` | Region cropping & keypoint subset extraction. |

## Keypoint mapping

| File | Covers |
|------|--------|
| `tests/test_mapping.py` | `mapping.py` COCO-WholeBody→MediaPipe translation: output shapes for 133/17-kp x each tracking mode, coordinate correctness, edge cases, round-trip through `frame_to_rows`. |

## Multi-camera

| File | Covers |
|------|--------|
| `tests/test_calibration.py` | `calibration.py` JSON IO, schema validation, `utc_timestamp`. |
| `tests/test_charuco.py` | `charuco.py` solver on synthetic warped-board renders (3 cams, known GT, MJPG videos): intrinsics (f < 2%, c < 12 px), extrinsics (rot < 1°, trans < 15 mm), world-frame zero pose + metadata, global RMS bound, save/load roundtrip, sync-offset arithmetic, error paths (unknown world frame, no detections, insufficient overlap, missing video, marker ≥ square), render dimensions, `_subsample`. Module-scoped solve fixture; render = 3× supersample warpPerspective + INTER_AREA (plain warp aliases marker interiors). |
| `tests/test_calibration_cli.py` | `calibration_cli.py` wiring: verify summary/exit codes, solve passthrough (monkeypatched solver) + save + empty-session error, board PNG E2E re-detection (all 40 interior corners) + custom geometry dims + marker-size rejection, `_parse_devices`/`_parse_squares`, `_compose_grid`, capture name-count mismatch. |
| `tests/test_multicam.py` | `multicam.py` session discovery (manifest + glob), calibration auto-load, sync offsets, path traversal rejection (camera file, calibration path, camera name), `iter_synchronized_frames`, `process_session` callback invocation + output dir creation + 3D-fusion hook (summary print, world3d.csv on disk + header/columns, non-fatal failure), `read_csv_keypoints` round-trip (kps + conf + timestamps), `write_world3d_csv` round-trip (rounding, blank-NaN, n_views/cheirality ints), `fuse_session_outputs` (sync-offset alignment, per-frame timestamps from world-frame camera, missing-CSV/calibration errors). Uses MJPG/AVI synthetic videos + synthetic projected CSVs. |
| `tests/test_triangulation.py` | `triangulation.py` projection / undistort / weighted DLT primitives; `fuse_session_frame` policy layer (noisy 3-cam <5mm, 2-cam <1cm, occlusion, insufficient views, outlier-view rejection, min_views, cheirality flag, confidence aggregation, validation errors). |
| `tests/test_validation.py` | `validation.py` end-to-end: `run_validation` over a synthetic session (3× supersampled ChArUco render for the solve + projected synthetic skeleton via injected `camera_processor`). Solve and load-calibration branches, all report sections finite + in sane synthetic ranges (reproj < 2 px, bone CoV/symmetry < 0.05, jitter < 5 mm), JSON NaN-free, CSV-reuse branch, `pose-estimation-validate` CLI writes JSON + Markdown, missing-calibration `ValidationError` + exit code 2. R clinical leg `skipif`-guarded (`_HAS_R`). **Verdict grading (1B)** on constructed reports (`_good_report()` all-PASS, single-field mutations): WARN/FAIL bands per metric, n_views floor, informational checks (timing/symmetry) never escalating, non-finite→WARN, baseline `_deg` agreement graded + non-angle noted, custom-thresholds override, verdict in JSON/Markdown, CLI exit code 0/0/1 for PASS/WARN/FAIL (+`--strict` WARN→1). **QA gate (1C)**: good capture (rendered session + fully-detected arms+hands subject via `_full_skeleton_processor`) grades not-FAIL and clears every sufficiency check; bad capture (`_render_bad_capture`: 6 centre-clustered board poses + cam3 truncated to half its frames) FAILs board coverage + ChArUco frame floor + frame-count parity; QA `to_json` carries the verdict and is NaN-free; `--qa-only` CLI exits 0/1. |

## R pipeline integration

| File | Covers |
|------|--------|
| `tests/test_r_pipeline.py` | End-to-end R pipeline compatibility: verifies rtmlib-mapped CSVs (via `coco_to_mediapipe` + `frame_to_rows`) are consumable by `clinical_features.R`. Tests CSV schema for hands-arms/body/17-kp modes, runs R clinical pipeline on synthetic data, verifies output columns including movement-phase segmentation (smoke test + crafted reach-grasp trajectory). 3D mode (`TestWorld3DClinical`): synthetic world3d.csv with known geometry — exact 90° elbows, 21.80° trunk lean (total + sagittal), 0.3 m/s wrist velocity, metric reach, reproj/cheirality gating to NA, `_3d` output suffixes, no rescan of own outputs. Skipped when R unavailable. |

## Infrastructure

| File | Covers |
|------|--------|
| `tests/test_public_api.py` | Stability of the package-level re-exports in `pose_estimation/__init__.py`. |
| `tests/test_models_checksum.py` | Model download URLs + checksums (guards against silent registry drift). |
| `tests/test_benchmark_config.py` | YAML sweep config parsing. |
| `tests/test_helpers.py` | Shared helpers (`video_io.safe_fps` clamp, `video_io.frame_count` count/missing-file, processing/postprocess validators); assertions that bench fixtures match production shapes. |
| `tests/conftest.py` | Shared fixtures. |

## Pytest options (from `pyproject.toml`)

- `-ra --strict-config --strict-markers --import-mode=importlib`
- `pythonpath = ["src"]`
- `filterwarnings = ["error", "ignore::DeprecationWarning:pkg_resources.*"]` — any unexpected warning fails the test.

## Coverage

```bash
uv run pytest --cov=pose_estimation
```

## When changing the public API

Update `tests/test_public_api.py` in the same commit. The package-level re-export list in `src/pose_estimation/__init__.py` is the canonical surface — anything not there is internal.
