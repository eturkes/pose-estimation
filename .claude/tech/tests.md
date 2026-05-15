# Tests

All tests live in `tests/`. Run with `uv run pytest`. Pytest is configured strict (warnings → errors).

## Core pipeline tests

| File | Covers |
|------|--------|
| `tests/test_smoothing.py` | `OneEuroFilter`, hand and body smoothing paths. |
| `tests/test_constraints.py` | Bone-length smoothing, joint-angle clamping. |
| `tests/test_matching.py` | Hungarian hand→arm matching, distality reject. |
| `tests/test_detection.py` | NMS, detection EMA smoothing, carry-forward. |
| `tests/test_processing.py` | Synthetic hands, re-crop, affine helpers. |
| `tests/test_extrapolation.py` | Carry-forward extrapolation behaviour. |

## RTMW / rtmlib path

| File | Covers |
|------|--------|
| `tests/test_rtmw_age_gating.py` | Age-based gating of stale detections. |
| `tests/test_rtmw_confidence.py` | Confidence handling for the 133-keypoint output. |
| `tests/test_rtmw_constraints.py` | Constraint behaviour on rtmlib keypoints. |
| `tests/test_rtmw_extrapolation.py` | Carry-forward for rtmlib path. |
| `tests/test_rtmw_matching.py` | Matching specific to rtmlib outputs. |
| `tests/test_rtmw_regions.py` | Region cropping & keypoint subset extraction. |

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
