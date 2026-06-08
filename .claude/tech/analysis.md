# Analysis pipeline (R)

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

## Feature engineering

| Script | Inputs | Outputs |
|--------|--------|---------|
| `features.R` | landmark CSVs | Variance ranking, correlation heatmap, scree plot, biplot, UMAP, feature ranking CSV. Requires `uwot`, `tidyverse`. |
| `clinical_features.R` | landmark CSVs (hands-arms or body) **or `world3d.csv`** (auto-detected) | `*_clinical.csv` (per-frame): elbow flexion, wrist deviation, finger spread, reach distance (raw + shoulder-normalised), grasp aperture (thumb–index, thumb–pinky), wrist/fingertip displacement, **bilateral comparison** (symmetry ratio, dominance index, absolute difference for each metric pair), **trunk/torso metrics** (body mode only: trunk lean, lateral lean, sagittal lean [3D only], trunk rotation, posture symmetry). `*_clinical_windows.csv` (1 s windows, 50 % overlap): spectral arc length (SAL, configurable fc), mean + peak wrist velocity, **normalized jerk** (wrist + fingertip), **movement efficiency** (wrist), **compensatory pattern index** (body mode only), **trunk windowed summaries** (body mode only: mean/sd/range), **bilateral comparison** for each window metric. 3D inputs get `_3d` output suffixes (see below). Hands-only CSVs skipped (no arm keypoints). Helpers: `angle_at_vertex`, `dist_3d`, `spectral_arc_length`, `normalized_jerk`, `movement_efficiency`, `trunk_lean_angle`, `trunk_lean_lateral`, `trunk_rotation`, `posture_symmetry`, `compute_bilateral`, `is_world3d`, `adapt_world3d`, `trunk_lean_angle_3d`, `trunk_lean_sagittal_3d`, `trunk_rotation_3d`, `posture_symmetry_3d`. |

## 3D input mode (world3d.csv)

`clinical_features.R` auto-detects fused 3D inputs (schema: `tech/multicam.md`) via `is_world3d()` — any column ending `_x_m`. Same script, same feature path; differences:

- **Gating first** (`adapt_world3d()`): a keypoint-frame is masked to NA when `reproj_err_px > REPROJ_GATE_PX` (constant, 20 px — matches the fusion-side `max_view_reproj_px`; required because at exactly `min_views` fusion cannot drop an outlier view) **or** `cheirality_ok == 0`. Diagnostic columns are then dropped and `_{x,y,z}_m` renamed to `_{x,y,z}`, after which the existing 3D-capable helpers (`angle_at_vertex`, `dist_3d`, window speed) operate unchanged.
- **Units are physical**: angles deg (true 3D, not projected), distances m, velocities m/s, path lengths m. 2D inputs remain normalised-coordinate units.
- **Trunk metrics use true plane decomposition** (z available): `trunk_lean_angle_3d` (total, vs −y vertical), `trunk_lean_sagittal_3d` → new column `trunk_lean_sagittal_deg` (positive = leaning away from camera; NA in 2D mode — out-of-plane is unmeasurable), `trunk_rotation_3d` (shoulder vs hip line in x–z plane), `posture_symmetry_3d` (3D shoulder width). Lateral lean formula is shared (x–y, identical in both modes). Windowed `trunk_lean_sagittal_mean/sd` added in both branches (NA in 2D).
- **Vertical assumption**: world −y = up holds only if the `world_frame` camera is level (documented in `tech/multicam.md`).
- **Output suffix partition**: `*_clinical_3d.csv`, `*_clinical_3d_windows.csv`, `*_movement_phases_3d.csv`. Downstream aggregation scripts glob `_clinical.csv`/`_clinical_windows.csv` and therefore skip `_3d` outputs by construction — metre-unit rows must stay out of normalised-unit aggregations. Downstream 3D aggregation is deliberately not built yet.
- Input discovery excludes its own outputs via regex `clinical[_a-z0-9]*|movement_phases[_a-z0-9]*` (digit class covers `_3d`).
- Window stats use `safe_mean`/`safe_sd` (all-NA → NA, warning-free); CPI now reuses per-frame `trunk_lean_deg` instead of recomputing 2D lean, so it is mode-appropriate automatically.

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

All scripts handle degenerate inputs gracefully after 2026-05-24 hardening:

- **Short videos** (<10 frames): `clinical_features.R` emits per-frame features, skips windowed features. `temporal_clinical.R` skips videos <10 rows with a message.
- **Zero-variance features**: `compare_clinical.R`, `clinical_dimreduce.R`, `features.R` warn and skip heatmap/PCA/UMAP plots when insufficient variable features remain.
- **Missing hand data**: columns filled with NA/blank; R scripts use safe column extraction (`ex()` returns NA vector for absent columns).
- **Single video/patient**: correlation/longitudinal scripts produce output but flag insufficient data.

## Bilateral comparison metrics

Added by `compute_bilateral()` in `clinical_features.R:58-78`. Applied to all per-side metric pairs in both per-frame and per-window outputs.

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
- Guards: returns NA when n < 5 frames or amplitude < 1e-10.

### Movement Efficiency

Path curvature ratio: `ME = path_length / straight_line_distance`.
- 1.0 = perfectly straight start-to-end movement; higher = more curved/corrective.
- Applied to wrist trajectory (`{side}_wrist_movement_efficiency`).
- Guard: returns NA when start ≈ end (straight_line < 1e-10).

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

### Helpers (`clinical_features.R:194-262`)

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
