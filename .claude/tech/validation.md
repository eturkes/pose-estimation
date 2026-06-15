# Validation harness

End-to-end validation of the 3D clinical pipeline on one session, emitting a
structured report. `src/pose_estimation/validation.py`; console script
`pose-estimation-validate`. Roadmap Session 1A (`prompts/sessions.md`).

## What it does

`run_validation(session_dir, *, calibration=None, baseline=None, device="NPU",
backend="onnxruntime", output_dir=None, camera_processor=None,
run_clinical=True, confidence_floor=…)` (`validation.py:247`) runs the full
chain on a single session and returns a `ValidationReport` (`validation.py:218`):

1. **Calibration** — external `calibration` arg wins; a calibration *file* is
   loaded, a *directory* with `calibration.json` is loaded, a directory without
   one is solved via `charuco.solve_charuco` and saved. No calibration anywhere
   → `ValidationError` (`validation.py:117`).
2. **2D tracking** — three-way branch (`_run_tracking`): an injected
   `camera_processor` (tests), else reuse if ≥2 per-camera CSVs already exist,
   else a real backend run via `python -m pose_estimation.run --headless` per
   camera.
3. **3D fusion** — `multicam.fuse_session_outputs` + `export.write_world3d_csv`
   (run directly, *not* via `process_session`, so each stage gets its own
   wall-clock).
4. **Clinical metrics** — `analysis/clinical_features.R` on `world3d.csv` (skipped
   gracefully when `Rscript` is absent or `run_clinical=False`).

It **orchestrates and measures only** — no pipeline maths reimplemented. The
substantive blocks (`solve_charuco`, `fuse_session_outputs`,
`write_world3d_csv`, `read_csv_keypoints`, the R script) are reused.

## Report schema

`ValidationReport` is a dataclass tree. `to_json()` → CI-parseable dict
(non-finite floats → `null` via `_native`); `to_markdown()` → human summary.
`REPORT_SCHEMA_VERSION` bumps on layout change. Sections:

| Section | Key numbers |
|---------|-------------|
| `calibration` | reprojection RMS, per-camera intrinsics (fx/fy/cx/cy, ‖dist‖), `world_frame`, camera count, `solved` (solved-this-run vs loaded). |
| `tracking_2d` | per-camera frame count, detection rate, low-confidence fraction (below `confidence_floor`), dropped (zero-detection) frames; `reused_existing_csvs`. |
| `fusion_3d` | reproj-err median/p95/max, n_views median/min, cheirality-violation rate, unfused fraction. |
| `timing` | per-stage wall-clock, throughput (fused fps), device + backend. |
| `agreement` | baseline-optional (see below) + `clinical_csv_produced`. |

**Raw numbers only — no PASS/WARN/FAIL verdict.** Grading + thresholds land in
Session 1B.

### Agreement leg (baseline-optional, by design)

No external ground truth exists yet (decision 2026-06-15). With a `baseline`
JSON (`{metric_column: reference_value}`) → per-metric absolute error of clinical
aggregates. Without → internal self-consistency surrogates straight from
`world3d.csv` (`_build_agreement`, `validation.py:694`):

- **bone-length CoV** — std/mean of each rigid bone's 3D length over frames.
- **L/R symmetry** — relative |L−R| per symmetric bone pair.
- **temporal jitter (mm)** — median magnitude of the *second* temporal
  difference (acceleration), which isolates frame-to-frame noise from steady
  movement.

`_BONES` / `_SYMMETRIC_BONES` cover both arm-mode (`arm_*`) and body-mode
(`body_*`) names; the harness keeps only bones whose both endpoints are in the
fused set, so the active mode self-selects. Untracked skeleton keypoints surface
as all-NaN columns and are dropped from the surrogates by design.

`unfused_keypoint_fraction` is computed over **active** keypoints only (fused in
≥1 frame). A keypoint never tracked in 2D (e.g. legs in arm mode) is a tracking
gap, visible in `tracking_2d.detection_rate`, not a fusion failure.

## Running

```bash
pose-estimation-validate --session-dir videos/session_a \
    --calibration calib.json --out report.json --markdown report.md
```

Flags (`main`, `validation.py:982`): `--session-dir` (required), `--calibration`
(file | dir | omit to reuse `<session>/calibration.json`), `--baseline`,
`--device`, `--backend`, `--out` (JSON), `--markdown`. Exit code `2` on
`ValidationError` (e.g. no calibration); `0` on success.

## Tests

`tests/test_validation.py` runs the harness end-to-end on a synthetic session:
a 3× supersampled ChArUco warp render (mirrors `test_charuco`) for the
calibration solve, plus a projected synthetic skeleton fed through an injected
`camera_processor`. Fusion is near-exact (same calibration solves both the rig
and the fusion), so surrogate thresholds are tight (bone CoV < 0.05, symmetry
< 0.05, jitter < 5 mm). The R-dependent test is `skipif` guarded
(`_HAS_R`). See `tech/tests.md`.
