# R analysis pipeline

Scripts in `analysis/` consume metrics + landmark CSVs from `output/`. Invoke with `Rscript analysis/<script>.R <args>`.

R deps are managed by `renv` (lockfile: `renv.lock`). Install with `renv::restore()`.

## Shared helpers

- `analysis/utils.R` — `script_dir()`, `aggregate_per_video()`, `METADATA_COLS`, `WINDOW_META`. Sourced by most other scripts via `source(file.path(script_dir(), "utils.R"))`.

## Diagnostic / quality

| Script | Inputs | Outputs |
|--------|--------|---------|
| `summary.R` | `*_metrics.csv` in dir | Text report + JSON. |
| `timeseries.R` | metrics CSVs | Temporal diagnostic plots (PNG). |
| `keypoint_detail.R` | `*_kp_detail.csv` | Per-keypoint heatmaps + trajectory plots. |
| `compare.R` | two JSON run summaries | Side-by-side run comparison. |

## Finger mobility / osteoarthritis screen (standalone)

Separate from the `clinical_features.R` ecosystem: each reads one capture's raw landmark CSV directly (`timestamp_sec` + `{left,right}_hand_{0..20}_{x,y,z,conf}`; legacy files may lack `conf`). Current-schema coordinates with blank/nonfinite/zero confidence are masked before analysis, so carried predictions do not become clinical evidence; legacy files retain finite-coordinate presence semantics. Degenerate joints and non-increasing timestamps yield `NA` rather than infinite or fabricated mobility. Finger flexion per frame = sum of the two inter-segment joint angles over a finger's four MediaPipe landmarks (`id..id+3`; MCP start thumb=1, index=5, middle=9, ring=13, pinky=17).

| Script | Inputs | Outputs |
|--------|--------|---------|
| `data_extraction.R` | one landmark CSV `[out_dir]` | `<stem>_angle_data.csv` (per-frame flexion, 5 fingers × both hands), `<stem>_mobility_analysis.csv` (frame-to-frame Δ + angular speed). |
| `arthrose_diag.R` | one landmark CSV `[out_dir]` | stdout: index range-of-motion, mean angular speed, mobility diagnosis (thresholds amplitude ≥ 40°, speed ≥ 60°/s), or an explicit insufficient-observations result; `<stem>_closed_hand.png` (thumb–index distance over time). Needs `zoo`. |

Live-camera captures now feed these: `pose-estimation-run <idx> --output-dir output/` exports `output/camera<idx>.csv` (file sources still use the file stem).

## Feature engineering

| Script | Inputs | Outputs |
|--------|--------|---------|
| `features.R` | landmark CSVs | Confidence-gated variance ranking, correlation heatmap, scree plot, biplot, UMAP, feature ranking CSV. Requires `uwot`, `tidyverse`. |
| `clinical_features.R` | landmark CSVs (hands-arms or body) **or `world3d.csv`** (auto-detected) | `*_clinical.csv` (per-frame): elbow flexion, wrist deviation, finger spread, reach distance (raw + shoulder-normalised), grasp aperture (thumb–index, thumb–pinky), wrist/fingertip displacement, **bilateral comparison** (symmetry ratio, dominance index, absolute difference for each metric pair), **trunk/torso metrics** (body mode only: trunk lean, lateral lean, sagittal lean [3D only], trunk rotation, posture symmetry). `*_clinical_windows.csv` (1 s windows, 50 % overlap): spectral arc length (SAL, configurable fc), mean + peak wrist velocity, **normalized jerk** (wrist + fingertip), **movement efficiency** (wrist), **compensatory pattern index** (body mode only), **trunk windowed summaries** (body mode only: mean/sd/range), **bilateral comparison** for each window metric. 3D inputs get `_3d` output suffixes (see below). Hands-only CSVs skipped (no arm keypoints). Helpers include `adapt_2d_confidence`, `adapt_world3d`, `angle_at_vertex`, `dist_3d`, `spectral_arc_length`, `normalized_jerk`, `movement_efficiency`, and the trunk/bilateral helpers. |

## 2D observation gating

Raw 2D consumers gate coordinates before deriving features.
`clinical_features.R::adapt_2d_confidence()` and
`features.R::mask_unobserved_coordinates()` mask body/arm coordinates whose
explicit `_vis` is blank, nonfinite, or zero. Current hand schemas receive the
same treatment from `_conf`, so carried/held coordinates cannot become clinical
evidence or drive variance, PCA, and UMAP. Positive scores are retained; the
validation report separately flags low-but-positive confidence. Legacy hand
CSVs without `_conf` keep finite-coordinate presence semantics.

## 3D input mode (world3d.csv)

`clinical_features.R` auto-detects fused 3D inputs (schema: `multicam.md`) via `is_world3d()` — any column ending `_x_m`. Same script, same feature path; differences:

- **Gating first** (`adapt_world3d()`): a keypoint-frame is masked to NA when reprojection or cheirality diagnostics are blank/nonfinite, `reproj_err_px > REPROJ_GATE_PX` (constant, 20 px — matches the fusion-side `max_view_reproj_px`; required because at exactly `min_views` fusion cannot drop an outlier view), or `cheirality_ok == 0`. When the `triangulation_angle_deg` column exists, blank/nonfinite values also fail closed and finite values must meet the provisional 1° fusion gate. Legacy files lacking the angle column entirely remain readable. Diagnostic columns, including candidate/final view counts, are then dropped and `_{x,y,z}_m` renamed to `_{x,y,z}`, after which the existing 3D-capable helpers (`angle_at_vertex`, `dist_3d`, window speed) operate unchanged.
- **Units are physical**: angles deg (true 3D, not projected), distances m, velocities m/s, path lengths m. 2D inputs remain normalised-coordinate units.
- **Trunk metrics use true plane decomposition** (z available): `trunk_lean_angle_3d` (total, vs −y vertical), `trunk_lean_sagittal_3d` → new column `trunk_lean_sagittal_deg` (positive = leaning away from camera; NA in 2D mode — out-of-plane is unmeasurable), `trunk_rotation_3d` (shoulder vs hip line in x–z plane), `posture_symmetry_3d` (3D shoulder width). Lateral lean formula is shared (x–y, identical in both modes). Windowed `trunk_lean_sagittal_mean/sd` added in both branches (NA in 2D).
- **Vertical assumption**: world −y = up holds only if the `world_frame` camera is level (documented in `multicam.md`).
- **Output suffix partition**: `*_clinical_3d.csv`, `*_clinical_3d_windows.csv`, `*_movement_phases_3d.csv`. Downstream aggregation scripts glob `_clinical.csv`/`_clinical_windows.csv` and therefore skip `_3d` outputs by construction — metre-unit rows must stay out of normalised-unit aggregations. Downstream 3D aggregation is deliberately not built yet.
- **Artifact identity tags (3D outputs only)**: each 3D output carries nine character columns, appended last and constant within the file — `artifact_kind`, `source_sha256`, `coord_space`, `distance_unit`, `producer_version`, `metric_method_version`, `qc_policy_version`, `metric_qualification`, `provenance_class`. 2D outputs carry none of them, so the partition above stays intact and metre-unit provenance can never reach a normalised-unit aggregation. Values are character precisely so `utils.R::aggregate_per_video()`, which promotes every numeric non-metadata column to a feature, cannot pick them up: one numeric tag would become five feature columns.
  - `artifact_kind` ∈ `clinical-frame-3d` / `clinical-window-3d` / `movement-phase-3d`. Artifact identity is the pair (`source_sha256`, `artifact_kind`); capture identity stays `video`, which a 3D input must carry exactly once and non-blank or the run fails closed.
  - `source_sha256` is the SHA-256 of the input CSV's bytes (`openssl::sha256`). It is a pure function of content, so reruns are byte-stable and goldens hold, a copy under a new name is recognisably the same artifact, and changed input bytes are recognisably a different one.
  - `coord_space` = `world-metric-3d`, `distance_unit` = `m`. Per-metric units (m, m/s, deg, dimensionless) are a registry concern — these outputs are wide and mix all four, so no per-row `unit` column is representable here.
  - `metric_qualification` states gap semantics in the artifact rather than only here: `gap-aware` on windows (timestamp-aware kernel), `gap-unsafe` on movement phases (still differentiates across holes), `frame-instantaneous` on per-frame values, whose displacements are row-adjacent steps that go NA on a masked sample without checking the interval.
  - `provenance_class` = `unverified`: rig, model and filter identity are absent from `world3d.csv`, so provenance is declared unknown rather than guessed, and a reader refuses to pool across unverified artifacts instead of assuming equivalence.
  - The three version tags are independent. `producer_version` tracks the emitted column set, `metric_method_version` a metric's computation, `qc_policy_version` the gate values (`REPROJ_GATE_PX`, `TRIANGULATION_ANGLE_GATE_DEG`, `OBSERVATION_CONFIDENCE_GATE`). Values are `v<n>`, never bare digits — a bare `1` is guessed `double` by a default `read_csv`, which would re-open the numeric-feature hazard.
  - **The triplet identifies the producer release, not the individual file.** `attach_artifact_tags()` writes one triplet across every 3D output of a run, so a bump in any of the three moves all 3D files — including a file whose own content is byte-identical to the previous release. Read the tags as "which release emitted this", never as "this file changed". A reader that needs per-file change detection must compare content or hashes; a reader that needs release compatibility compares tags. Per-file version vectors are a deliberate non-goal here, because one triplet per release is what makes a set of 3D artifacts safe to pool.
- **Typed empty outputs (3D only)**: all three 3D artifacts are always written, carrying the full ordered header with zero data rows when nothing qualifies, which also overwrites any stale file from an earlier run. 2D keeps its skip-if-empty behaviour. CSV carries no type channel — a default `read_csv` of a header-only file returns every column as character — so a reader recovers types by supplying explicit `readr::cols()` collectors, and `window_schema()`/`phase_schema()` are the ordered column owners it builds them from. A zero-row artifact cannot carry tag values; identity binds by stem to the frame artifact, which is non-empty whenever the input has at least one row.
- Input discovery excludes its own outputs via regex `clinical[_a-z0-9]*|movement_phases[_a-z0-9]*` (digit class covers `_3d`).
- Window stats use `safe_mean`/`safe_sd` (all-NA → NA, warning-free); CPI now reuses per-frame `trunk_lean_deg` instead of recomputing 2D lean, so it is mode-appropriate automatically.

## Window QC evidence artifact

**Artifact.** `clinical_features.R` writes `<stem>_clinical_3d_window_qc.csv` beside the three existing 3D artifacts. It never writes this artifact for 2D input.

The producer writes the file for every processed 3D input. When no window qualifies, it writes the complete header with zero rows. A rerun therefore clears stale content.

Each row describes one attempted metric in one emitted window. Rows remain when the matching estimate is `NA`.

The unique key is `video × person_idx × window_start_sec × window_end_sec × metric_id`. The artifact never duplicates estimate values. Read estimates from `<stem>_clinical_3d_windows.csv` by using the shared window key and `metric_id`.

**Fields.** The first 24 fields form the QC record. The nine `ARTIFACT_TAG_COLS` follow them as the final block. The record widened from 22 to 24 fields. `qc_policy_tolerance` and `qc_coverage_tolerance` were published between `max_gap_sec` and `qc_status`. A consumer therefore reproduces both verdicts from the row alone, not from a code constant.

| Column | Type | Description |
|--------|------|-------------|
| `video` | string | Capture identity from the input |
| `person_idx` | int | Tracked person index |
| `window_start_sec` | float | Inclusive window start time, in seconds |
| `window_end_sec` | float | Exclusive window end time, in seconds |
| `metric_id` | string | Canonical window-estimate column name |
| `source_group` | string | Required-keypoint group that supplies the metric |
| `n_expected_frames` | int | Nominal slots in the half-open window |
| `n_valid_frames` | int | Slots where every required coordinate passes the gate |
| `frame_coverage` | float | `n_valid_frames / n_expected_frames` |
| `n_expected_intervals` | int | Adjacent intervals available on the nominal grid |
| `n_valid_intervals` | int | Adjacent expected-slot pairs with two valid endpoints |
| `interval_coverage` | float | `n_valid_intervals / n_expected_intervals` |
| `valid_duration_sec` | float | Valid interval count divided by `fs` |
| `longest_gap_frames` | int | Largest run of consecutive missing nominal slots |
| `longest_gap_sec` | float | `longest_gap_frames / fs` |
| `n_gaps` | int | Number of missing-slot runs, including edge runs |
| `required_keypoints` | string | Comma-separated canonical keypoint prefixes, in dependency order and without spaces |
| `n_required_keypoints_present` | int | Required keypoints with at least one gate-passed sample in the window |
| `min_coverage` | float | Literal minimum frame-coverage threshold |
| `max_gap_sec` | float | Literal maximum gap threshold, in seconds |
| `qc_policy_tolerance` | float | Relative estimator slack applied to the gap comparison |
| `qc_coverage_tolerance` | float | Relative representation slack applied to the coverage comparison |
| `qc_status` | string | `pass` or `fail` |
| `qc_reason` | string | Precedence-selected primary usability cause |
| `artifact_kind` | string | Fixed value `window_qc` |
| `source_sha256` | string | SHA-256 of the input CSV bytes, shared with the other 3D artifacts |
| `coord_space` | string | Fixed value `world-metric-3d` |
| `distance_unit` | string | Fixed value `m` |
| `producer_version` | string | Producer layout identifier; `v3` for this artifact set |
| `metric_method_version` | string | Metric computation identifier; `v2` after cadence adoption |
| `qc_policy_version` | string | QC policy identifier; `v3` for this policy |
| `metric_qualification` | string | Fixed value `gap-aware` |
| `provenance_class` | string | Fixed value `unverified` |

**Arithmetic.** Each window uses the half-open interval `[window_start_sec, window_end_sec)`. Nominal slots include missing samples, so missing input rows still count as expected.

Let `N` be the nominal-slot count. Let `valid[i]` mean that every metric-required coordinate passes the gate at slot `i`.

| Field | Exact definition |
|-------|------------------|
| `n_expected_frames` | `N`: nominal slots in `[window_start_sec, window_end_sec)` |
| `n_valid_frames` | `sum(valid)`: slots where every metric-required coordinate passes the gate |
| `n_expected_intervals` | `max(n_expected_frames - 1, 0)` |
| `n_valid_intervals` | Count of adjacent expected-slot pairs where `valid[i] && valid[i + 1]` |
| `frame_coverage` | `n_valid_frames / n_expected_frames` |
| `interval_coverage` | `n_valid_intervals / n_expected_intervals` |
| `valid_duration_sec` | `n_valid_intervals / fs` |
| `longest_gap_sec` | `longest_gap_frames / fs` |

`interval_coverage` is `NA` when `n_expected_intervals` is zero. Its count and `valid_duration_sec` are then zero.

Coverage denominators always use nominal-slot counts. They never use the observed row count. A row absent at a window edge counts as an expected slot, the same as an absent interior row.

The estimators keep a narrower grid that starts at the first observed sample. Evidence therefore describes the whole window, while an estimate describes the observed span inside it. This keeps the jerk duration term stable when a clip loses its edge rows.

**Status and reason precedence.** `qc_status` is `pass` or `fail`. A pass requires every policy condition and a finite matching estimate. A passing row uses `qc_reason = none`.

For a failing row, the producer selects the first applicable reason in this precedence order:

| Precedence | `qc_reason` | Trigger |
|------------|-------------|---------|
| 1 | `invalid_timebase` | The producer cannot build the nominal grid for the window |
| 2 | `missing_required_keypoints` | A required coordinate column is absent, or `n_valid_frames == 0` |
| 3 | `insufficient_observations` | `n_valid_frames < 2` or `n_valid_intervals < 1` |
| 4 | `gap_too_long` | `longest_gap_sec` exceeds the tolerance-adjusted maximum |
| 5 | `insufficient_coverage` | `frame_coverage` is below the tolerance-adjusted minimum |
| 6 | `estimator_undefined` | Earlier conditions pass, but the matching estimate is not finite |

An `invalid_timebase` row keeps its key, dependency, thresholds, and tags. Its count, coverage, duration, and gap fields are `NA`.

The artifact records only the highest-precedence cause. Concurrent support causes remain reconstructable from the independent evidence fields.

**Gap definition.** A gap is a run of consecutive missing nominal slots. Leading and trailing runs also count.

`longest_gap_frames` is the largest run length. `longest_gap_sec` equals that count divided by `fs`.

If observed samples flank `k` missing slots, their unobserved span covers `k + 1` intervals. The field reports the missing-slot duration, not that larger span.

The producer estimates `fs` with `nominal_fs(t, magnitude = TRUE)`. That estimator averages the non-gap intervals. The average cancels the four-decimal export rounding.

The reciprocal of the median interval does not cancel that rounding. The earlier estimator read about 30.03 Hz for a nominal 30 fps capture.

The magnitude keeps a cadence for an out-of-order clip. The window is then keyed and reported as `invalid_timebase`. A signed estimate would drop the clip before any window existed.

The estimator needs two usable positive intervals. A single interval carries a whole quantum of rounding error, which is worse than the estimator it replaces.

Accuracy has a floor. Each endpoint carries up to half of the 1e-4 second export quantum. Rounding errors telescope only inside one uninterrupted run of retained intervals.

One retained run obeys `abs(delta_fs / fs) <= 1e-4 / span`. The producer claims that bound for a span of 1 second or more.

The gap filter drops long intervals, so a clip with gaps keeps `k` separate runs. Those runs do not telescope into each other. The bound then loosens to `k * 1e-4 / S_retained`, where `S_retained` is the retained exposure. A clip with gaps must not use the full endpoint span as the denominator.

Grid placement consumes this error. `trajectory_grid_status()` checks the displacement per window, so the lever arm is `WINDOW_SEC` and the budget is `GRID_SLOT_TOLERANCE`. Measured margins run 135× at 30 Hz down to 17× at 119.88 Hz. The same errors leave 1.686× over a 20-second clip, so this margin is a per-window result and not a whole-clip guarantee. The residual measures the displacement directly and decides whether an estimate describes the data.

**Policy and tolerance.** The shipped thresholds are engineering-provisional. They are not clinically validated.

Each row carries `min_coverage = 0.80` and `max_gap_sec = 0.10`. These literal values make the support-policy result re-derivable without a registry.

Each row also carries two slacks, because the two comparisons carry different error:

- `longest_gap_sec <= max_gap_sec * (1 + qc_policy_tolerance)`, with `qc_policy_tolerance = 1e-4`
- `frame_coverage >= min_coverage * (1 - qc_coverage_tolerance)`, with `qc_coverage_tolerance = 1e-9`

The gap comparison divides a slot count by an estimated cadence, so it carries the estimator residual. The coverage comparison divides two integer counts, so it carries representation error alone.

One shared 1e-9 slack made the nominal 30 Hz three-slot verdict follow the clip length. The verdict cycled pass, pass, fail as the frame count moved through the residues of 3. That result had no physical meaning.

The gap slack stays far below one frame period. It admits 1e-5 seconds against 8.3e-3 seconds at 120 Hz, so it cannot hide a real gap.

The first comparison is inclusive. `interval_coverage` remains evidence and does not gate status.

A policy change must update `qc_policy_version`. This artifact ships with `qc_policy_version = v3`.

**Limitations.** `qc_reason` reports metric-usability causes only. It never attributes upstream fusion failures to reprojection, cheirality, triangulation angle, absent source views, or confidence.

That attribution requires new diagnostic fields from `src/pose_estimation/export.py`. The current adapter exposes only the resulting validity mask.

QC evidence is advisory. It never suppresses or overwrites an estimate. A failed QC row can therefore accompany a finite estimate.

An estimate is `NA` only when the metric kernel cannot compute it. Estimate values remain exclusively in `<stem>_clinical_3d_windows.csv`.

SAL reconstructs missing interior speed intervals linearly. Therefore, the artifact makes no zero-interpolation claim and emits no interpolation-count field.

The `n_valid_intervals` field records observed interval support for SAL. A leading or trailing speed gap still makes SAL undefined.

**Current scope.** The artifact currently emits 12 trajectory metrics across four source groups:

| `source_group` | `required_keypoints` | `metric_id` values |
|----------------|----------------------|--------------------|
| `left_wrist` | `body_left_wrist` | `left_wrist_sal`, `left_wrist_velocity_mean`, `left_wrist_velocity_peak`, `left_wrist_normalized_jerk`, `left_wrist_movement_efficiency` |
| `right_wrist` | `body_right_wrist` | `right_wrist_sal`, `right_wrist_velocity_mean`, `right_wrist_velocity_peak`, `right_wrist_normalized_jerk`, `right_wrist_movement_efficiency` |
| `left_fingertip` | `left_hand_8` | `left_fingertip_normalized_jerk` |
| `right_fingertip` | `right_hand_8` | `right_fingertip_normalized_jerk` |

The producer emits no rows for `bilateral_wrist`, `bilateral_fingertip`, `trunk`, `shoulders` and `cpi`. Their estimates in `<stem>_clinical_3d_windows.csv` carry no usability evidence. A bilateral estimate reads its two side groups, so the rows for those groups bound it. The trunk, shoulder and CPI estimates have no evidence source in this artifact.

## Clinical comparison / longitudinal

| Script | Inputs | Outputs |
|--------|--------|---------|
| `clinical_correlation.R` | `*_clinical*.csv` + `clinical_scores.csv` | `*_correlation_table.csv` (Pearson, Spearman, BH-FDR), `*_correlation_matrix.png`, `*_scatter_top.png`. |
| `longitudinal.R` | `*_clinical*.csv` + `sessions.csv` (+ optional `clinical_scores.csv`) | `*_longitudinal_summary.csv`, per-patient line plots. Flags Δ > 1 SD from baseline. |
| `compare_clinical.R` | `*_clinical*.csv` | `all_clinical_video_summary.csv`, `all_clinical_radar.png`, `all_clinical_heatmap.png`. Outlier flag at >2 SD. |
| `clinical_dimreduce.R` | `*_clinical*.csv` | `all_clinical_pca_scree.png`, `all_clinical_pca_biplot.png`, `all_clinical_umap.png`, `all_clinical_pca_loadings.csv`. Requires `uwot`, `tidyverse`. |
| `temporal_clinical.R` | `*_clinical*.csv` | `<stem>_clinical_timeseries.png` per video, `all_clinical_timeseries_overview.png`. Skips videos with <10 rows. Requires `patchwork`, `tidyverse`. |
| `explore_clinical.R` | `*_clinical*.csv` | `all_clinical_distributions.png`, `all_clinical_na_heatmap.png`, `all_clinical_boxplots.png`, `all_clinical_window_distributions.png`. Sanity checks. |

## Metadata management

| Script | Purpose |
|--------|---------|
| `make_templates.R` | Scans output dir for unique videos; writes `clinical_scores_template.csv` (`video, GRASSP, UEMS, SCIM`) and `sessions_template.csv` (`video, patient_id, session_date`). |
| `validate_metadata.R` | Validates a completed scores/sessions CSV. Auto-detects type. Checks columns, duplicates, video matches, ISO 8601 dates, numeric scores. Exit 0 valid, 1 errors. |

## Bundled report

- `analysis/analysis_summary.Rmd` — R Markdown report; renders to `analysis/analysis_summary.html` (committed for browsing).

## Edge-case resilience

All scripts handle degenerate inputs gracefully:

- **Short videos** (<10 frames): `clinical_features.R` emits per-frame features and no windows — 2D writes no windows file, 3D writes a typed empty one. `temporal_clinical.R` skips videos <10 rows with a message.
- **Zero-variance features**: `compare_clinical.R`, `clinical_dimreduce.R`, `features.R` warn and skip heatmap/PCA/UMAP plots when insufficient variable features remain.
- **Missing hand data**: columns filled with NA/blank; R scripts use safe column extraction (`ex()` returns NA vector for absent columns).
- **Single video/patient**: correlation/longitudinal scripts produce output but flag insufficient data.

## Bilateral comparison metrics

Added by `compute_bilateral()` in `analysis/clinical_features.R`. Applied to all per-side metric pairs in both per-frame and per-window outputs.

### Formulas (using abs() internally for sign-agnostic handling)

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| `{metric}_symmetry_ratio` | min(abs(L), abs(R)) / max(abs(L), abs(R)) | [0, 1] | 1.0 = perfect symmetry; 0 = one side absent/zero |
| `{metric}_dominance_index` | (abs(R) − abs(L)) / (abs(R) + abs(L)) | [−1, 1] | Positive = right has larger magnitude; 0 = symmetric |
| `{metric}_abs_diff` | abs(R − L) | [0, ∞) | Raw asymmetry in original metric units |

### Per-frame bilateral metrics (9 pairs × 3 = 27 columns)

Applied to: `elbow_angle_deg`, `wrist_deviation_deg`, `finger_spread_deg`, `reach_raw`, `reach_norm`, `grasp_aperture_thumb_index`, `grasp_aperture_thumb_pinky`, `wrist_displacement`, `fingertip_displacement`.

### Edge cases

- One side NA → all three bilateral metrics are NA (R's NA propagation).
- Both sides zero → symmetry_ratio = NA, dominance_index = NA, abs_diff = 0 (guarded by denom > 1e-12).
- SAL (negative values): abs() ensures correct ratio/dominance computation. Positive dominance_index for SAL means right side has larger |SAL| = less smooth.

## Movement quality metrics

Added to `compute_window_features()` in `clinical_features.R`. Provide smoothness, efficiency, and compensation analysis per sliding window.

### Normalized Jerk (Hogan & Sternad 2009)

Dimensionless jerk metric: `NJ = sqrt(T^5 / (2 * a^2) * integral(||jerk||^2 dt))`.
- `T` = window duration (seconds), `a` = path length (amplitude), jerk = 3rd derivative of position.
- Lower NJ = smoother movement; minimum-jerk trajectory gives ~18.97.
- Applied to wrist (`{side}_wrist_normalized_jerk`) and index fingertip (`{side}_fingertip_normalized_jerk`).
- Jerk is summed over fully observed 4-sample stencils only; `T` is the true grid span, and amplitude counts observed intervals alone.
- Guards: returns NA when n < 5 frames, amplitude < 1e-10, or no stencil is gap-free.

### Movement Efficiency

Path curvature ratio: `ME = path_length / straight_line_distance`.
- 1.0 = perfectly straight start-to-end movement; higher = more curved/corrective.
- Applied to wrist trajectory (`{side}_wrist_movement_efficiency`).
- Guards: returns NA when start ≈ end (straight_line < 1e-10), or when the observed path is broken by an interior gap. Leading and trailing gaps are trimmed instead, since they shorten the span without breaking it.

### Gap handling (window scope)

Window metrics run through `trajectory_metrics()`, which places samples on the nominal frame grid (`trajectory_grid()`) and masks each derivative wherever its stencil touches a hole. Quality-gated NA samples and absent rows are treated alike.

Why it matters: differentiating across a gap as though survivors were adjacent moved normalized jerk from 16.456 to 2437 on a *single* dropped frame, and to 23 605 on a 15-frame hole. Masking holds those at +0.8 % and +22.9 %.

The estimands deliberately differ in how they treat an unobserved span:
- **NJ** — fully observed stencils, fixed `dt`, true span duration.
- **SAL** — interior missing speed intervals are filled linearly; a leading or trailing gap returns NA rather than extrapolated motion. Needs ≥4 observed intervals.
- **Velocity mean/peak** — observed support only, so both are biased low under loss; a peak hidden inside a gap is unrecoverable.
- **Efficiency** — NA on a broken path. Bridging a hole with a straight chord biases the ratio toward 1.0, reporting a straighter, healthier movement than was observed.
- **Dropout** — missing nominal duration over the full span, plus the longest gap run. Returned by the kernel; it becomes an output column in M3.3.

On a complete grid every metric reduces to the previous operation order and stays bit-identical — enforced by `tests/goldens/r_clinical/` and `tests/test_r_trajectory_kernel.py`.

The movement-phase block (`analysis/clinical_features.R`) still calls the legacy gap-unsafe primitives and is explicitly out of scope for this work.

`fs` comes from `nominal_fs()` at every call site. Window enumeration reads interval magnitudes. Movement segmentation reads signed intervals. The estimator averages the non-gap intervals, which cancels the exporter's four-decimal rounding. The reciprocal of the median interval amplified that rounding. It read about 30.03 Hz for a 30 fps capture.

### Compensatory Pattern Index (body mode only)

Pearson correlation between `trunk_lean_angle` and `max(left_reach, right_reach)` within each window.
- `trunk_lean_angle`: unsigned angle (degrees) between shoulder-midpoint→hip-midpoint vector and vertical. 0 = upright, 90 = horizontal.
- High positive CPI suggests trunk compensation for limited arm ROM.
- Requires hip keypoints → body mode only; NA in hands-arms mode.
- Guard: requires ≥5 non-NA frame pairs for meaningful correlation.
- Column: `compensatory_pattern_index` (not lateralised — single value per window).

### SAL frequency cutoff

`spectral_arc_length(v, fs, fc = SAL_FREQ_CUTOFF)` — the `fc` parameter (default 10 Hz) is now configurable. 10 Hz matches Balasubramanian et al. (2012/2015) for upper-limb movements. Higher cutoffs (up to 20 Hz) may be appropriate for fast movements; the function clamps to Nyquist automatically.

### Per-window bilateral metrics (6 pairs × 3 = 18 columns)

Applied to: `wrist_sal`, `wrist_velocity_mean`, `wrist_velocity_peak`, `wrist_normalized_jerk`, `wrist_movement_efficiency`, `fingertip_normalized_jerk`.

## Trunk/torso metrics (body mode only)

Added to `compute_frame_features()` in `clinical_features.R`, gated behind `tracking == "body"`. Hands-arms and hands modes receive NA for all trunk columns (columns are still emitted for schema consistency).

### Per-frame columns

| Column | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| `trunk_lean_deg` | `trunk_lean_angle()`: unsigned angle between shoulder-midpoint→hip-midpoint vector and vertical | [0, 90] deg | 0 = upright, 90 = horizontal |
| `trunk_lean_lateral_deg` | `trunk_lean_lateral()`: `atan2(dx, -dy)` where dx = sh_mid_x − hip_mid_x, dy = sh_mid_y − hip_mid_y | (−90, 90) deg | Positive = leaning right, negative = leaning left |
| `trunk_rotation_deg` | `trunk_rotation()`: angle difference between shoulder line (L→R) and hip line (L→R) | (−180, 180] deg | Positive = shoulders rotated clockwise relative to hips (viewed from front) |
| `posture_symmetry` | `posture_symmetry()`: (lsh_y − rsh_y) / shoulder_width_2d | (−1, 1) | Positive = right shoulder higher (left dropped); NA when shoulder width ≈ 0 |

### Per-window columns (mean + SD of per-frame values; trunk_lean also gets range)

| Column | Source |
|--------|--------|
| `trunk_lean_mean`, `trunk_lean_sd`, `trunk_lean_range` | `trunk_lean_deg` |
| `trunk_lean_lateral_mean`, `trunk_lean_lateral_sd` | `trunk_lean_lateral_deg` |
| `trunk_rotation_mean`, `trunk_rotation_sd` | `trunk_rotation_deg` |
| `posture_symmetry_mean`, `posture_symmetry_sd` | `posture_symmetry` |

### Helpers (`analysis/clinical_features.R`)

- `trunk_lean_angle()` — unsigned total lean (existing, also used by CPI).
- `trunk_lean_lateral()` — signed lateral lean in frontal plane.
- `trunk_rotation()` — shoulder vs hip line angle difference.
- `posture_symmetry()` — normalised shoulder height asymmetry.

### Body-mode gate

Requires hip keypoints (`body_left_hip_*`, `body_right_hip_*`) which only exist in body mode (33 MediaPipe keypoints). Hands-arms mode has 12 arm keypoints (shoulders → finger bases) — no hips. The gate checks `tracking == "body"` in both `compute_frame_features()` and `compute_window_features()`.

## Temporal movement segmentation

Added by `segment_movements()` in `clinical_features.R`. Produces `*_movement_phases.csv` alongside the existing per-frame and per-window outputs.

### Algorithm

1. **Movement detection**: per-side wrist speed (coord-units/sec) smoothed with `running_median(k=5)`. Above-threshold segments detected via RLE where threshold = `speed_thresh_pct` × peak speed (default 5%). Close segments merged (gap ≤ `min_gap_frames`, default 3). Short segments rejected (< `min_movement_frames`, default 5).

2. **Phase classification** (`classify_movement_phases()`): state machine within each movement using smoothed grasp-aperture derivative:
   - **REACH** — default (hand moving, aperture stable/open)
   - **GRASP** — first sustained run of aperture derivative < −threshold (closing)
   - **TRANSPORT** — between GRASP end and RELEASE start (moving with closed aperture)
   - **RELEASE** — sustained run of aperture derivative > +threshold (opening)
   - Transitions require `min_phase_frames` (default 3) consecutive frames. Aperture threshold is adaptive: 5% of aperture range within the movement. Without hand data, entire movement stays REACH.

3. **Per-phase feature extraction**: peak/mean velocity, path length, normalized jerk, SAL, mean bilateral reach symmetry ratio.

4. **Per-movement summary** (denormalized across phase rows): total duration, number of phases, peak velocity, total path length, movement efficiency.

### Output schema (`*_movement_phases.csv`)

Phase metrics are **not** gap-qualified: they still drop or bridge tracking holes, unlike the window kernel above. The 3D artifact says so in its own `metric_qualification = gap-unsafe` column, so a copied file carries the caveat with it; the `_3d` variant also carries the other eight identity tags after the columns below.

| Column | Type | Description |
|--------|------|-------------|
| `video` | string | Source video name |
| `person_idx` | int | Person index |
| `side` | string | "left" or "right" |
| `movement_idx` | int | Movement number (per side, 1-based) |
| `phase` | string | REACH, GRASP, TRANSPORT, or RELEASE |
| `start_frame`, `end_frame` | int | Frame range (inclusive) |
| `duration_sec` | float | Phase duration |
| `peak_velocity`, `mean_velocity` | float | Speed statistics (coord/sec) |
| `path_length` | float | Cumulative wrist displacement |
| `smoothness_nj` | float | Normalized jerk (NA if < 5 frames) |
| `smoothness_sal` | float | Spectral arc length (NA if < 4 frames) |
| `mean_reach_symmetry` | float | Mean reach_raw_symmetry_ratio during phase |
| `movement_duration_sec` | float | Total movement duration |
| `movement_n_phases` | int | Distinct phases in this movement |
| `movement_peak_velocity` | float | Peak speed across entire movement |
| `movement_path_length` | float | Total path length across movement |
| `movement_efficiency` | float | Path length / straight-line distance |

### Helpers

- `running_median(x, k)` — sliding median filter preserving edges.
- `classify_movement_phases()` — aperture-derivative state machine.
- `segment_movements()` — main orchestrator (movement detection + phase classification + feature extraction).

### Edge cases

- **No hand data** → all phases labelled REACH (pointing/reaching tasks).
- **Low aperture variation** (range < 1e-8) → all phases labelled REACH.
- **Very short video** (< `min_movement_frames`) → no movements detected.
- **Static wrist** (peak speed < 1e-10) → no movements detected.
- **Hands-only mode** → skipped (same as all other clinical features).

## Aggregation convention

Per-video aggregation (used by correlation / longitudinal / dimreduce / compare): mean, median, SD, min, max for frame features; mean, SD for window features. Implemented once in `aggregate_per_video()` (`utils.R`); always reuse rather than duplicate.
