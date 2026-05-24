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
| `clinical_features.R` | landmark CSVs (hands-arms or body) | `*_clinical.csv` (per-frame): elbow flexion, wrist deviation, finger spread, reach distance (raw + shoulder-normalised), grasp aperture (thumb–index, thumb–pinky), wrist/fingertip displacement. `*_clinical_windows.csv` (1 s windows, 50 % overlap): spectral arc length (SAL), mean + peak wrist velocity. Hands-only CSVs skipped (no arm keypoints). Helpers: `angle_at_vertex`, `dist_3d`, `spectral_arc_length`. |

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

## Aggregation convention

Per-video aggregation (used by correlation / longitudinal / dimreduce / compare): mean, median, SD, min, max for frame features; mean, SD for window features. Implemented once in `aggregate_per_video()` (`utils.R`); always reuse rather than duplicate.
