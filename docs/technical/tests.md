# Test suite

All tests live in `tests/`. Run with `uv run pytest`. Pytest is configured strict (warnings → errors).

## Core pipeline tests

| File | Covers |
|------|--------|
| `tests/test_smoothing.py` | Confidence-safe `OneEuroFilter`; non-finite recovery with zero emitted observation confidence; prediction-aware association; x/y outlier caps; adaptive min-cutoff; active/dormant track limits; zero-confidence carry-forward. |
| `tests/test_constraints.py` | Clipped robust bone-length estimates, missing-segment recovery, iterative linked projections, observation-validity isolation, and rigid distal-branch joint-angle clamps. |
| `tests/test_matching.py` | Validity-gated Hungarian hand→arm matching, including maximum-cardinality behaviour and distality rejection. |
| `tests/test_detection.py` | SSD decode, score stability, ordinary NMS, MediaPipe-compatible score-weighted NMS, detection EMA smoothing, and carry-forward. |
| `tests/test_processing.py` | Detector value ranges and exact letterbox inversion, graph-compatible pose/palm ROIs and replicated crop borders, finite-safe heatmap and landmark-output/handedness decode contracts, synthetic hands, re-crop, and affine helpers. |
| `tests/test_extrapolation.py` | Carry-forward extrapolation behaviour, prediction-assisted association, and emitted confidence integrity. |

## RTMW / rtmlib path

| File | Covers |
|------|--------|
| `tests/test_rtmlib_csv_export.py` | `run.py` CSV export via `process_source()`: schema correctness, per-keypoint hand confidence, source-media timestamps, per-source state reset, coordinate normalization, and video-name handling. Uses mock tracker + synthetic video. |
| `tests/test_rtmw_age_gating.py` | Age-based gating of stale detections. |
| `tests/test_rtmw_confidence.py` | Confidence handling for the 133-keypoint output. |
| `tests/test_rtmw_constraints.py` | Constraint behaviour on rtmlib keypoints. |
| `tests/test_rtmw_extrapolation.py` | Carry-forward and zero-confidence stale geometry for the rtmlib path, including current-score caps on held coordinates. |
| `tests/test_rtmw_matching.py` | Finite confidence-weighted centroids and validity-gated, prediction-aware matching for rtmlib outputs. |
| `tests/test_rtmw_regions.py` | Region cropping & keypoint subset extraction. |

## Keypoint mapping

| File | Covers |
|------|--------|
| `tests/test_mapping.py` | `mapping.py` COCO-WholeBody→MediaPipe translation: output shapes for 133/17-kp x each tracking mode, coordinate correctness, authoritative COCO anatomical side, per-keypoint hand confidence, edge cases, and round-trip through `frame_to_rows`. |

## Multi-camera

| File | Covers |
|------|--------|
| `tests/test_calibration.py` | `calibration.py` JSON IO, schema validation, `utc_timestamp`. |
| `tests/test_charuco.py` | `charuco.py` solver on synthetic warped-board renders (3 cams, known GT, MJPG videos): intrinsics (f < 2%, c < 12 px), extrinsics (rot < 1°, trans < 15 mm), world-frame zero pose + metadata, global RMS bound, save/load roundtrip, sync-offset arithmetic, error paths (unknown world frame, no detections, insufficient overlap, missing video, marker ≥ square), render dimensions, `_subsample`. Module-scoped solve fixture; render = 3× supersample warpPerspective + INTER_AREA (plain warp aliases marker interiors). |
| `tests/test_calibration_cli.py` | `calibration_cli.py` wiring: verify summary/exit codes, solve passthrough (monkeypatched solver) + save + empty-session error, board PNG E2E re-detection (all 40 interior corners) + custom geometry dims + marker-size rejection, `_parse_devices`/`_parse_squares`, `_compose_grid`, capture name-count mismatch. |
| `tests/test_multicam.py` | `multicam.py` session discovery (manifest + glob), calibration auto-load, sync offsets, path traversal rejection (camera file, calibration path, camera name), `iter_synchronized_frames`, `process_session` callback invocation + output dir creation + 3D-fusion hook (summary print, world3d.csv on disk + header/columns, non-fatal failure), `read_csv_keypoints` round-trip (body/hand confidence, legacy hand presence, timestamps), `write_world3d_csv` round-trip (rounding, blank-NaN, n_views/cheirality ints), `fuse_session_outputs` (sync-offset alignment, per-frame timestamps from world-frame camera, missing-CSV/calibration errors). Uses MJPG/AVI synthetic videos + synthetic projected CSVs. |
| `tests/test_triangulation.py` | Projection/undistortion and confidence-weighted DLT primitives; deterministic two-view minimal-set consensus; bounded geometric refinement; final residual, cheirality, and viewing-ray-angle gates; occlusion/outlier recovery; and strict input validation. |
| `tests/test_validation.py` | End-to-end synthetic-session validation and QA, including solve/load branches, true expected-frame denominators, zero-confidence detection accounting, zero/one-based keypoint CSVs, trusted-fusion geometry metrics, viewing-ray-angle diagnostics, schema/threshold serialization, report grading, and CLI exit codes. The R clinical leg is `skipif`-guarded (`_HAS_R`). |
| `tests/test_validation_failuremodes.py` | Failure-mode suite: one injected degradation per test on the shared synthetic session, asserting the harness *identifies* the fault (report field crosses threshold + verdict degrades; bad data → NaN, never garbage). **Camera dropout** (cam3 → 2 frames) → `n_views_median`=2 WARN + cam3 frame-count short; **miscalibration** (cam2 +5° yaw, projected through the true rig, fused through the perturbed one) → `reproj_err_px_median`≈12.8 FAIL with the bad view *kept* (n_views=3, below the rejection cliff at ~6°); **desync** (5× velocity, cam2 `sync_offset`=2 vs a 0-offset control) → reprojection rises into WARN; **low confidence** (all conf=0.1) → `worst_low_confidence_fraction`=1 FAIL while fusion still reconstructs (flagged, not dropped), plus a zero-confidence-region case proving the validity gate routes to NaN (active set 12→8); **occlusion** (region in 1 cam → recovered from the other 2; wide region in 2 cams part-way → `unfused_keypoint_fraction`≈0.5 FAIL, surviving fusion still low-CoV); **degenerate calibration** (1 cm baseline + 1 px noise) → `bone_length_cv`/`temporal_jitter_mm` FAIL while reproj p95 stays PASS — the "2D fine, 3D garbage" signature. Magnitudes are deterministic and empirically calibrated against fusion outlier rejection. |

## R pipeline integration

| File | Covers |
|------|--------|
| `tests/test_r_pipeline.py` | End-to-end R pipeline compatibility for mapped 2D CSVs and synthetic 3D geometry. Covers zero-confidence observation masking in feature selection, clinical, and standalone finger-mobility scripts; legacy hand-confidence fallback; insufficient-observation diagnosis; movement phases; known joint/trunk/speed metrics; fail-closed reprojection/cheirality/viewing-angle trust gates; legacy angle-column absence; `_3d` suffixes; and output-rescan avoidance. Skipped when R is unavailable. |

## Infrastructure

| File | Covers |
|------|--------|
| `tests/test_inventory.py` | Covers the grammar, eight normalization traces, dispositions, exclusion precedence, family identities, all three artifact schemas, determinism, and hostile inputs. It also covers no-decode probes, full-byte fixity, backend provenance, path redaction, and three-artifact generation validation. |
| `tests/test_sessions.py` | Covers the 18 acceptance predicates of the session generator: placement conservation over every registry row, event-identity type safety and its two-digit grammar, take resolution for view-conflict families, discovery and `--list-sessions` agreement, byte-identical regeneration across locale, hash seed, time zone and output name, idempotency with stale removal, every generation tamper class, the closed marker schema, registry-first validation, synthetic escaped-path decoding, symlink extension and containment, console and manifest redaction, the two-rename publication swap, the instrumented no-corpus-listing proof, and marker-shaped output ownership. |
| `tests/test_public_api.py` | Stability of the package-level re-exports in `pose_estimation/__init__.py`. |
| `tests/test_models_checksum.py` | Model download URLs + checksums (guards against silent registry drift). |
| `tests/test_benchmark_config.py` | YAML sweep config parsing. |
| `tests/test_postprocess.py` | Savitzky-Golay observation gating for zero/nonfinite `_vis` and `_conf`, unchanged evidence columns, and legacy coordinate-only behavior. |
| `tests/test_helpers.py` | Shared helpers (`video_io.safe_fps`, `frame_count`, `SourceTimestampClock`, processing/postprocess validators); assertions that benchmark fixtures match production shapes. |
| `tests/test_source_timestamps.py` | Both pipeline entry points use source presentation time for file input, preserve malformed-frame index/timestamp gaps, and reset temporal state between independent sources; MediaPipe CSV export excludes carried hands, preserves observed match indexing, and carries hand-presence confidence. |
| `tests/conftest.py` | Shared fixtures, incl. the session-scoped `rendered_session` (a rendered 3-camera ChArUco session + its `solve_charuco` result) consumed by both validation suites. Built once per test session (the render + solve is the most expensive fixture). |
| `tests/synthetic_session.py` | Not a test module — the shared synthetic-session builders behind `rendered_session` and the validation suites: ChArUco render (`render_calibration_session`), projected-skeleton CSV writer (`write_skeleton_csv` with fault-injection hooks: `confidence`, `occlude`/`occlude_frames`, `zero_conf`, `frames`, `velocity`, `project_with`, `noise_px`) + its `skeleton_processor`, the fully-detected QA variant, and `render_bad_capture`. Imports as a top-level module (it is on `pythonpath`). |

## Pytest options (from `pyproject.toml`)

- `-ra --strict-config --strict-markers --import-mode=importlib`
- `pythonpath = ["src", "tests"]` — `tests` is on the path so cross-test helper modules (`synthetic_session.py`) import as top-level under `importlib` mode. The same root is mirrored in `[tool.ty.environment].root` so `ty` resolves the import too.
- `filterwarnings = ["error", "ignore::DeprecationWarning:pkg_resources.*"]` — any unexpected warning fails the test.

## Coverage

```bash
uv run pytest --cov=pose_estimation
```

## When changing the public API

Update `tests/test_public_api.py` in the same commit. The package-level re-export list in `src/pose_estimation/__init__.py` is the canonical surface — anything not there is internal.
