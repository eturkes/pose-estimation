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
| `clinical_features.R` | landmark CSVs (hands-arms or body) | `*_clinical.csv` (per-frame): elbow flexion, wrist deviation, finger spread, reach distance (raw + shoulder-normalised), grasp aperture (thumb–index, thumb–pinky), wrist/fingertip displacement, **bilateral comparison** (symmetry ratio, dominance index, absolute difference for each metric pair). `*_clinical_windows.csv` (1 s windows, 50 % overlap): spectral arc length (SAL, configurable fc), mean + peak wrist velocity, **normalized jerk** (wrist + fingertip), **movement efficiency** (wrist), **compensatory pattern index** (body mode only), **bilateral comparison** for each window metric. Hands-only CSVs skipped (no arm keypoints). Helpers: `angle_at_vertex`, `dist_3d`, `spectral_arc_length`, `normalized_jerk`, `movement_efficiency`, `trunk_lean_angle`, `compute_bilateral`. |

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

## Aggregation convention

Per-video aggregation (used by correlation / longitudinal / dimreduce / compare): mean, median, SD, min, max for frame features; mean, SD for window features. Implemented once in `aggregate_per_video()` (`utils.R`); always reuse rather than duplicate.
