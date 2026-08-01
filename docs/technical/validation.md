# Clinical validation harness

End-to-end validation of the 3D clinical pipeline on one session, emitting a
structured report. `src/pose_estimation/validation.py`; console script
`pose-estimation-validate`; live real-data validation state in `.agent/roadmap.md`.

## What it does

`run_validation(session_dir, *, calibration=None, baseline=None, device="NPU",
backend="onnxruntime", output_dir=None, camera_processor=None,
run_clinical=True, confidence_floor=…)` runs the full chain on a single session
and returns a `ValidationReport`:

1. **Calibration** — external `calibration` arg wins; a calibration *file* is
   loaded, a *directory* with `calibration.json` is loaded, a directory without
   one is solved via `charuco.solve_charuco` and saved. No calibration anywhere
   → `ValidationError`.
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
`REPORT_SCHEMA_VERSION` (currently **3**) bumps on JSON *layout* change;
threshold-value changes bump `THRESHOLDS_VERSION` instead. Sections:

| Section | Key numbers |
|---------|-------------|
| `calibration` | reprojection RMS, per-camera intrinsics (fx/fy/cx/cy, ‖dist‖), `world_frame`, camera count, `solved` (solved-this-run vs loaded). |
| `tracking_2d` | per-camera **observed and expected** post-sync source-frame counts, detection rate, low-confidence fraction (below `confidence_floor`), dropped (zero-detection) frames; `reused_existing_csvs`. Expected counts come from source videos, not sparse CSV rows. |
| `fusion_3d` | fused and expected logical-frame counts + fused-frame fraction, trusted-keypoint fraction, reproj-err median/p95/max, accepted/candidate view diagnostics + view-rejection fraction, cheirality-violation rate, triangulation-angle p05/median, and unfused fraction. |
| `timing` | per-stage wall-clock, throughput (fused fps), device + backend. |
| `agreement` | baseline-optional (see below) + `clinical_csv_produced`. |
| `verdict` | overall PASS/WARN/FAIL + per-check grades vs `THRESHOLDS` (see below). Computed on demand by `ValidationReport.verdict()` and merged into `to_json()` / `to_markdown()`. |

### Agreement leg (baseline-optional, by design)

No external ground truth exists yet. With a `baseline`
JSON (`{metric_column: reference_value}`) → per-metric absolute error of clinical
aggregates. Without → internal self-consistency surrogates straight from
`world3d.csv` (`_build_agreement`):

- **bone-length CoV** — std/mean of each rigid bone's trusted 3D length over
  frames; one observation is insufficient and reports unmeasured rather than a
  false zero-variance PASS.
- **L/R symmetry** — relative |L−R| per symmetric bone pair.
- **temporal jitter (mm)** — median magnitude of the *second* temporal
  difference (acceleration) over contiguous frame-index triples only, so a CSV
  gap is not treated as a one-frame jump.

`_BONES` / `_SYMMETRIC_BONES` cover both arm-mode (`arm_*`) and body-mode
(`body_*`) names; the harness keeps only bones whose both endpoints are in the
fused set, so the active mode self-selects. Untracked skeleton keypoints surface
as all-NaN columns and are dropped from the surrogates by design.

Self-consistency uses **trusted** samples only: finite XYZ, finite reprojection
error ≤ `REPROJ_GATE_PX` (20 px), cheirality true, and, whenever the angle
column exists, a finite triangulation angle ≥
`TRIANGULATION_ANGLE_GATE_DEG` (1°). Only a legacy schema lacking the angle
column entirely opts out; present-but-blank diagnostics fail closed.

`unfused_keypoint_fraction` is computed over **active** keypoints only. Active
means fused in ≥1 frame **or attempted with at least two views**; thus a point
rejected every frame by residual/angle gates cannot disappear from the
denominator. A keypoint never observed in two views remains a 2D tracking gap,
visible in `tracking_2d.detection_rate`, not a fusion failure.
`trusted_keypoint_fraction` uses expected logical frames × active keypoints, so
missing world3d rows also reduce coverage.
Coverage accepts current zero-based and legacy one-based logical indices, but
counts only unique rows inside the inferred source-derived interval; stray
negative or high indices cannot substitute for missing expected frames.

`view_rejection_fraction` is
`sum(candidate_n_views - n_views) / sum(candidate_n_views)` over keypoint
attempts with at least two pre-consensus views. This separates robust consensus
rejecting an inconsistent camera (3→2 contributes 1/3) from a genuine
two-camera observation (2→2 contributes zero). Legacy world3d files without
candidate-view columns remain readable, but this diagnostic is explicitly
unavailable and ungraded rather than assumed zero.

## Acceptance thresholds + verdict

`THRESHOLDS` (`validation.py`, currently version **2**) is the
single-source-of-truth `Thresholds` dataclass; `THRESHOLDS_VERSION` bumps on any
value change (independent of the report schema version). Each metric is a
`Band(warn, fail, direction)`:
`direction="max"` → lower is better (`≤warn` PASS, `≤fail` WARN, else FAIL);
`direction="min"` → higher is better (`≥warn` PASS, `≥fail` WARN, else FAIL). A
**non-finite** metric grades WARN (surfaced for review, never silently passed).

`ValidationReport.verdict(thresholds=THRESHOLDS)` → `Verdict` (overall grade =
worst **non-informational** check; per-check `Check` list; explanatory notes).
Surfaced in `to_json` (`verdict` block) and `to_markdown` (verdict table at the
top). CLI exit code: **0** PASS/WARN, **1** FAIL, **2** harness error;
`--strict` promotes WARN→1 for stricter CI gating.

| Metric (check name) | warn | fail | dir | Rationale + source |
|---------------------|------|------|-----|--------------------|
| `calibration.reprojection_error_px` | 1.0 | 2.0 | max | <1 px photogrammetry gold standard; ~2 px (~1 cm) still yields usable kinematics. Pose2Sim robustness (Pagnon et al. 2021, *Sensors* 21(19):6530). |
| `fusion.reproj_err_px_median` | 8.0 | 12.0 | max | Pose2Sim triangulation cutoff ~10 px; Anipose <12 px in >75 % of frames (Karashchuk et al. 2021, *Cell Reports*). |
| `fusion.reproj_err_px_p95` | 15.0 | 20.0 | max | Anipose treats >20 px as missing (== `REPROJ_GATE_PX`, the fusion gate); <18 px in >90 % of frames. |
| `fusion.n_views_min` (hard floor) | — | <2 | — | DLT triangulation needs ≥2 views; below = malformed/degenerate fusion. Discrete FAIL guard. |
| `fusion.n_views_median` | 3.0 | 2.0 | min | 3-camera deployment: median <3 means the typical keypoint has no spare view to reject an outlier (`multicam.md`). |
| `tracking.worst_detection_rate` | 0.80 | 0.50 | min | Positive-confidence detected keypoints / expected post-sync source frame-keypoint slots, worst camera. Missing rows and zero-confidence carry/extrapolation count as misses. **Provisional**. |
| `tracking.worst_low_confidence_fraction` | 0.2 | 0.4 | max | Worst camera's fraction of detected keypoints below `confidence_floor` (0.3). **Provisional** pending cleared real footage. |
| `fusion.fused_frame_fraction` | 0.95 | 0.80 | min | Unique world3d logical frames / expected fusable frames. For `min_views=2`, expectation is the second-largest post-sync camera frame count. **Provisional**. |
| `fusion.trusted_keypoint_fraction` | 0.90 | 0.75 | min | Trusted 3D samples / expected active-keypoint slots, including missing logical frames in the denominator. **Provisional**. |
| `fusion.triangulation_angle_deg_p05` | 2.0 | 1.0 | min | Fifth percentile of available viewing-ray angles. Fusion hard-gates below 1°; <2° warns that accepted geometry remains close to that ill-conditioned floor. **Provisional**. |
| `fusion.view_rejection_fraction` | 0.05 | 0.20 | max | Pre-consensus candidate views rejected by robust consensus. Persistent 3→2 rejection = 0.333 and FAILs; genuine 2→2 occlusion contributes zero. **Provisional**. |
| `fusion.unfused_keypoint_fraction` | 0.1 | 0.25 | max | Active-keypoint frame slots that failed to fuse. **Provisional**. |
| `fusion.cheirality_violation_rate` | 0.01 | 0.05 | max | Points should reconstruct in front of all contributing cameras; violations ≈ 0 in a healthy solve. |
| `agreement.mean_bone_length_cv` | 0.05 | 0.10 | max | Rigid-bone length CoV across frames = cross-camera reconstruction precision; ~1 cm markerless keypoint noise on a ~0.25 m forearm ≈ 4–5 % (dual-camera OA RMSD ~11 mm, *Ann. Biomed. Eng.* 2025). |
| `agreement.temporal_jitter_mm` | 5.0 | 15.0 | max | Static-pose temporal noise via 2nd-difference (acceleration) magnitude. |
| `timing.throughput_fps` *(info)* | 15.0 | 5.0 | min | Whole-pipeline fused fps incl. one-time solve/R — a coarse perf-regression signal. **Provisional** pending a real per-device study. Excluded from the overall grade. |
| `agreement.mean_symmetry_rel_diff` *(info)* | 0.05 | 0.10 | max | L/R bone-length relative difference. Valid **only** for symmetric-by-construction input; excluded from the overall grade (real anatomical asymmetry would confound it). |
| `agreement.<metric>_deg` (baseline only) | 5.0 | 10.0 | max | ~5° clinical precision threshold for joint-angle agreement; OpenCap ~6° flagged borderline (OpenCap validation, *J. Biomech.* 2024). Graded only when a baseline supplies `_deg` metrics — none exists yet. |

*(info)* = informational: graded and surfaced, but does **not** raise the overall
verdict. Provisional values are engineering gates, **not clinically validated
thresholds**. They must be recalibrated against cleared real captures before
any clinical-accuracy claim.

## Capture QA gate

`qa_check(session_dir, *, calibration=None, output_dir=None,
camera_processor=None, device, backend, confidence_floor, board=None) ->
QAReport` (`validation.py`) is a **pre-flight gate**: it grades a *raw
capture* before its clinical metrics are trusted, without running the
fusion/clinical chain. The human procedure behind it is
`../capture_protocol.md`. Run it via `pose-estimation-validate
--qa-only`.

It assesses three failure surfaces, reusing the harness building blocks
(`detect_charuco_corners`, `_resolve_external_calibration`, `_run_tracking`
+ `_measure_tracking`):

| `QAReport` section | What it measures |
|--------------------|------------------|
| `calibration` (`CalibrationQA`) | Solved/loaded reprojection RMS + per-camera (`CharucoCameraQA`): board-detection count, detection rate, and **FOV coverage** (fraction of a `COVERAGE_GRID` = (8, 6) image grid the pooled board corners touched). Coverage + detection need the raw ChArUco *session directory*; a `calibration.json` file (or `None`) yields RMS only. |
| `parity` (`ParityQA`) | Raw per-camera frame counts + `disparity` = (max−min)/max — a software-sync **desync proxy** (`multicam.md`: no genlock). Zero-frame cameras remain in the calculation and cannot disappear from the denominator. |
| `subject` (`SubjectQA`) | Per-camera 2D detection rate + low-confidence fraction on the subject clip, measured against source-derived expected post-sync frames (same three-way tracking source as `run_validation`; degrades to *unassessed* rather than raising when no CSVs/backend are available). |

`QAReport.verdict()` grades against `QA_THRESHOLDS` (capture-specific) plus
the shared `THRESHOLDS` bands (calib RMS, confidence floor), reusing the
same `Band`/`Check`/`Grade`/`Verdict` vocabulary as the report verdict;
overall = worst check. `to_json` / `to_markdown` / CLI exit code mirror
`ValidationReport`. `QA_REPORT_SCHEMA_VERSION` and `QA_THRESHOLDS_VERSION`
are independent of the report-side versions.

### QA thresholds (`QA_THRESHOLDS`, version 1 — provisional)

All **provisional**: literature/engineering-grounded but unproven on real
captures; the first cleared real session recalibrates them (bump
`QA_THRESHOLDS_VERSION`). Calibration RMS is shared with the report verdict
(see the table above).

| Check | warn | fail | dir | Rationale |
|-------|------|------|-----|-----------|
| `calibration.min_charuco_frames` (hard floor) | — | < `MIN_INTRINSIC_FRAMES` (8) | — | Below the solver's intrinsic-frame minimum the calibration cannot solve. Discrete FAIL guard. |
| `calibration.worst_charuco_detection_rate` | 0.30 | 0.10 | min | Board detected / total frames, worst camera. Capture-style dependent (a fast varied sweep detects in fewer frames yet constrains geometry better), so lenient — the frame floor is the real sufficiency gate. |
| `calibration.worst_board_coverage` | 0.60 | 0.35 | min | FOV-grid occupancy, worst camera. A centre-bound board weakly constrains oblique-camera intrinsics and couples focal-length error into stereo translation. |
| `parity.frame_count_disparity` | 0.05 | 0.20 | max | (max−min)/max raw frame counts. Declared `sync_offset`s trim pre-roll, so a few frames is normal; a large mismatch signals a dropped/desynced recording. |
| `subject.worst_detection_rate` | 0.80 | 0.50 | min | The subject should track in most expected source frame-keypoint slots. Missing rows and zero-confidence carry count as misses. |
| `subject.worst_low_confidence_fraction` | 0.20 | 0.40 | max | Shared `THRESHOLDS.max_low_confidence_fraction`. |

## Clinical-validity gap register

Honest status of what the harness can and cannot yet prove. The headline gap is
agreement-vs-ground-truth: there is **no external baseline**, so clinical-metric
*correctness* is unproven; the verdict currently
rests on internal evidence (reprojection, cross-camera consistency, temporal
stability) only.

| Gap | Status | Detail / impact | Resolved by |
|-----|--------|-----------------|-------------|
| Clinical-metric agreement vs ground truth | **UNVALIDATED** | No known-geometry object, goniometer, or second reference system exists. Joint angles / reach / aperture are internally consistent but not proven *accurate*. `agreement_tolerance_deg` is dormant until a baseline arrives. | Baseline study; cheapest viable = known-length rod for scale + goniometer for ≥1 joint. |
| L/R symmetry surrogate | **Conditional** | Valid only when the input is symmetric by construction (phantom/synthetic). On a real subject, genuine anatomical asymmetry confounds it — hence graded *informational*, never failing the verdict. | A baseline or a known-symmetric phantom. |
| 2D bone-length constraints under foreshortening | **Known-approximate** | `BoneLengthSmoother` constrains 2D limbs that foreshorten under projection; `--no-constraints` disables it. 3D fusion does not suffer this, but 2D inputs feeding fusion still do. | Inherent to 2D; mitigated by 3D fusion + redundancy. |
| World-frame "up" assumption (trunk metrics) | **Assumption** | Trunk lean/rotation assume world −y = vertical, true only if the `world_frame` camera is level (`multicam.md`, `analysis.md`). A tilted reference camera biases all trunk angles. | Mitigated by `../capture_protocol.md` (level + verify the world camera, manual checklist item — the QA gate cannot see it); a gravity/level reference would close it fully. |
| Single-person fusion | **Scope limit** | Fusion uses `person_idx == 0` only; multi-subject scenes are out of scope (`multicam.md`). Two people in frame → only one fuses. | Future cross-camera person matching (not roadmapped). |
| Synchronization | **Software-only** | No hardware genlock; sync is recorder-aligned or integer `sync_offset` frames (`multicam.md`). Sub-frame desync degrades reprojection on fast motion; audio-xcorr sync is FUTURE. | Partially caught: `qa_check` frame-count-parity desync proxy + `../capture_protocol.md` sync procedure. Sub-frame desync still unmodelled (audio-xcorr FUTURE). |
| Throughput budget | **Provisional** | `throughput_fps` denominator includes one-time solve + R, so it is a coarse regression signal, not a real-time SLA. Graded *informational*. | Multi-session per-device performance study. |
| Thresholds calibrated on synthetic data only | **Provisional** | Bands are literature- or engineering-grounded but unproven on real captures; near-exact synthetic fusion sits far inside every band. | First cleared capture, then multi-session recalibration. |

## Running

```bash
# Full validation:
pose-estimation-validate --session-dir videos/session_a \
    --calibration calib.json --out report.json --markdown report.md
# Pre-flight capture QA only (no fusion/clinical chain):
pose-estimation-validate --session-dir videos/subject_a \
    --calibration videos/calib_a --qa-only --out qa.json --markdown qa.md
```

Flags (`main`): `--session-dir` (required), `--calibration` (file | dir | omit to
reuse `<session>/calibration.json`), `--baseline`, `--device`, `--backend`,
`--output-dir`, `--out` (JSON), `--markdown`, `--no-clinical`, `--qa-only`,
`--strict`. `--qa-only` runs `qa_check` instead of `run_validation` (pass the
raw ChArUco *session directory* as `--calibration` for board coverage). Exit
code: **0** PASS/WARN, **1** FAIL verdict, **2** `ValidationError` (e.g. no
calibration); `--strict` makes WARN exit **1** too. Both paths share `_emit`
(JSON + Markdown + verdict + exit code), duck-typed over the two report types.

## Tests

`tests/test_validation.py` runs the harness end-to-end on a synthetic session:
a 3× supersampled ChArUco warp render (mirrors `test_charuco`) for the
calibration solve, plus a projected synthetic skeleton fed through an injected
`camera_processor`. Fusion is near-exact (same calibration solves both the rig
and the fusion), so surrogate thresholds are tight (bone CoV < 0.05, symmetry
< 0.05, jitter < 5 mm) and the synthetic session grades **PASS**. The R-dependent
test is `skipif` guarded (`_HAS_R`).

Verdict grading is unit-tested on **constructed** reports (no harness run):
`_good_report()` builds an all-PASS report; tests mutate one field at a time to
assert each metric crosses its WARN/FAIL band, that informational checks (timing,
symmetry) never escalate the overall grade, that non-finite metrics grade WARN,
that baseline `_deg` agreement grades while non-angle metrics are noted, and that
the CLI exit code matches the verdict (incl. `--strict`). See `tests.md`.

The **QA gate** is tested in the same file: a good capture (the
rendered session + a *fully-detected* arms+hands subject via
`_full_skeleton_processor`) grades not-FAIL and clears every sufficiency check;
a deliberately bad capture (`_render_bad_capture`: 6 centre-clustered board
poses + cam3 truncated to half its frames) FAILs on board coverage, the ChArUco
frame floor, and frame-count parity; `--qa-only` exits 0 / 1 accordingly.
`frame_count` has its own unit test in `tests/test_helpers.py`.

The shared synthetic builders now live in `tests/synthetic_session.py` (render
+ projected-skeleton CSV with fault-injection hooks) and the `rendered_session`
fixture in `tests/conftest.py` (session-scoped: one render + solve for the whole
suite).

**Failure-mode suite** — `tests/test_validation_failuremodes.py`
proves the harness *surfaces* a degraded capture rather than emitting a
plausible-but-wrong report. One injected fault per test, each asserting the
matching report field crosses its threshold, the verdict degrades, and bad data
becomes NaN (never fabricated): camera dropout (`n_views`↓), miscalibration
and desynchronization (bad views are rejected, so `view_rejection_fraction`
FAILs while trusted reprojection stays low), low confidence
(`worst_low_confidence_fraction` FAIL + zero-conf NaN gate), occlusion
(`unfused_keypoint_fraction` FAIL, 1-camera occlusion recovered from the
remaining views), and degenerate geometry (angle/instability gates reject or
flag the "2D fine, 3D garbage" case). Injection magnitudes are deterministic
and explicitly account for robust minimal-set consensus.
