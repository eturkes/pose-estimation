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
| `tests/test_calibration.py` | `calibration.py` JSON IO, schema validation, `solve_charuco` stub. |
| `tests/test_multicam.py` | `multicam.py` session discovery (manifest + glob), calibration auto-load, sync offsets, path traversal rejection (camera file, calibration path, camera name), `iter_synchronized_frames`, `process_session` callback invocation + output dir creation + 3D-fusion hook (summary print, non-fatal failure), `read_csv_keypoints` round-trip, `fuse_session_outputs` (sync-offset alignment, missing-CSV/calibration errors). Uses MJPG/AVI synthetic videos + synthetic projected CSVs. |
| `tests/test_triangulation.py` | `triangulation.py` projection / undistort / weighted DLT primitives; `fuse_session_frame` policy layer (noisy 3-cam <5mm, 2-cam <1cm, occlusion, insufficient views, outlier-view rejection, min_views, cheirality flag, confidence aggregation, validation errors). |

## R pipeline integration

| File | Covers |
|------|--------|
| `tests/test_r_pipeline.py` | End-to-end R pipeline compatibility: verifies rtmlib-mapped CSVs (via `coco_to_mediapipe` + `frame_to_rows`) are consumable by `clinical_features.R`. Tests CSV schema for hands-arms/body/17-kp modes, runs R clinical pipeline on synthetic data, verifies output columns including movement-phase segmentation (smoke test + crafted reach-grasp trajectory). Skipped when R unavailable. |

## Infrastructure

| File | Covers |
|------|--------|
| `tests/test_public_api.py` | Stability of the package-level re-exports in `pose_estimation/__init__.py`. |
| `tests/test_models_checksum.py` | Model download URLs + checksums (guards against silent registry drift). |
| `tests/test_benchmark_config.py` | YAML sweep config parsing. |
| `tests/test_helpers.py` | Shared helpers; assertions that bench fixtures match production shapes. |
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
