# Pose Estimation — Agent Context

See README.md for project overview, architecture, usage, and dependencies.

## Optimisation Suite

The project includes instrumentation and analysis tools for the
assistant to iteratively improve the pipeline without seeing
the live video:

- `metrics.py` — `MetricsCollector` writes `*_metrics.csv` and
  optionally `*_kp_detail.csv` per video with jitter,
  confidence, smoothing delta, constraint corrections, detection
  counts, and carry-forward state.
- `analysis/summary.R` — text report + JSON from metrics CSV.
- `analysis/timeseries.R` — temporal diagnostic plots (PNG).
- `analysis/keypoint_detail.R` — per-keypoint heatmaps and
  trajectory plots from `*_kp_detail.csv`.
- `analysis/compare.R` — side-by-side comparison of two runs.
- `analysis/features.R` — feature selection and UMAP/PCA
  visualisation from landmark CSVs (variance, correlation
  heatmap, scree plot, biplot, UMAP by time/video, feature
  ranking CSV).  Requires R packages `uwot` and `tidyverse`.
- `analysis/clinical_features.R` — derives clinically meaningful
  kinematic features from landmark CSVs (hands-arms or body
  mode).  Per-frame outputs (`*_clinical.csv`): joint angles
  (elbow flexion, wrist deviation, finger spread), reach
  distance (raw and shoulder-width-normalised), grasp aperture
  (thumb–index, thumb–pinky), wrist and fingertip displacement.
  Per-window outputs (`*_clinical_windows.csv`, 1 s windows,
  50 % overlap): spectral arc length (SAL), mean and peak wrist
  velocity.  Hands-only CSVs are skipped (no arm keypoints).
  Helper functions (`angle_at_vertex`, `dist_3d`,
  `spectral_arc_length`) are unit-testable at the top of the
  file.
- `analysis/clinical_correlation.R` — correlates aggregated
  clinical features with external clinical scores
  (`clinical_scores.csv`).  Aggregates per-frame and per-window
  features per video (mean, median, SD, min, max), joins on
  `video`, computes pairwise Pearson and Spearman correlations
  with Benjamini-Hochberg FDR correction.  Outputs:
  `*_correlation_table.csv` (feature, score, pearson_r,
  spearman_rho, p_value, p_adj_bh, n),
  `*_correlation_matrix.png` (Spearman heatmap with
  significance stars), `*_scatter_top.png` (top 6 pairs).
  Warns and continues on unmatched videos or missing scores.
- `analysis/longitudinal.R` — compares clinical features for
  the same patient across multiple sessions (time points) to
  track recovery.  Requires `sessions.csv` with columns:
  `video`, `patient_id`, `session_date` (ISO 8601).  Aggregates
  per-video features (reuses `aggregate_per_video` pattern),
  computes session-to-session deltas and percentage changes from
  baseline, flags changes exceeding 1 SD of baseline.  Outputs:
  `*_longitudinal_summary.csv` (patient_id, feature,
  session_date, value, delta_from_baseline, pct_change, flagged),
  `*_longitudinal_<patient_id>.png` (line plots of top 6 most
  variable features per patient).  Optionally overlays clinical
  scores from `clinical_scores.csv` on plots when a third CLI
  argument is provided.  Patients with a single session are
  included in the summary (no deltas) but skipped for flagging.
- `analysis/temporal_clinical.R` — per-video temporal
  visualisation of clinical features.  For each video with ≥10
  per-frame rows, produces a multi-panel time-series plot
  (`<stem>_clinical_timeseries.png`) with left/right sides as
  coloured lines on shared panels (faceted by feature, free
  y-scales).  Window-level features (SAL, velocity) shown as
  step-function panels beneath the per-frame panels using
  `patchwork`.  Also produces
  `all_clinical_timeseries_overview.png` comparing normalised
  reach across all qualifying videos.  Videos with <10 rows are
  skipped.  Requires R packages `patchwork` and `tidyverse`.
- `analysis/compare_clinical.R` — between-video comparison of
  aggregated clinical features.  Aggregates per-frame features
  per video (mean, median, SD, min, max) and window features
  (mean, SD) into `all_clinical_video_summary.csv`.  Produces
  `all_clinical_radar.png` (parallel-coordinate plot of z-scored
  means) and `all_clinical_heatmap.png` (clustered heatmap).
  Flags outlier videos (>2 SD from group mean).
- `analysis/make_templates.R` — generates template metadata CSVs
  for downstream analyses.  Scans output directory for unique
  `video` values across `*_clinical.csv` files, writes
  `clinical_scores_template.csv` (columns: `video`, `GRASSP`,
  `UEMS`, `SCIM`) and `sessions_template.csv` (columns: `video`,
  `patient_id`, `session_date`) with blank values for the user
  to fill in.  Prints console instructions for next steps.
  Usage: `Rscript analysis/make_templates.R output/`.
- `analysis/validate_metadata.R` — validates a completed
  `clinical_scores.csv` or `sessions.csv` against the clinical
  feature CSVs.  Auto-detects file type (sessions if
  `patient_id`/`session_date` columns present, scores otherwise).
  Checks: required columns present, no duplicate videos, video
  names match clinical CSVs, dates parse as ISO 8601, scores are
  numeric.  Reports errors and warnings, exits 0 if valid, 1 if
  errors found.  Usage:
  `Rscript analysis/validate_metadata.R <metadata.csv> <output_dir>`.
- `analysis/clinical_dimreduce.R` — PCA and UMAP on per-video
  aggregated clinical features.  Loads `*_clinical.csv` and
  `*_clinical_windows.csv`, aggregates per video (mean, median,
  SD, min, max for frame features; mean, SD for window features),
  drops zero-variance / >50 % NA columns, imputes remaining NAs
  with column medians, z-scores, then runs PCA and UMAP.
  Outputs: `all_clinical_pca_scree.png` (scree plot),
  `all_clinical_pca_biplot.png` (PC1 vs PC2 with video labels
  and top-10 loading arrows),
  `all_clinical_umap.png` (2D UMAP with video labels),
  `all_clinical_pca_loadings.csv` (feature loadings on first
  PCs).  Prints console summary of variance explained and top
  features on PC1/PC2.  Requires R packages `uwot` and
  `tidyverse`.
- `analysis/explore_clinical.R` — exploratory summary and
  sanity-check of clinical features.  Loads all `*_clinical.csv`
  and `*_clinical_windows.csv` from a directory, prints per-video
  row counts, per-column NA rates, summary statistics, and
  data-quality warnings (>50 % NA features, zero-row videos,
  constant-valued features).  Outputs:
  `all_clinical_distributions.png` (per-feature density by video),
  `all_clinical_na_heatmap.png` (video × feature missingness),
  `all_clinical_boxplots.png` (box plots by video),
  `all_clinical_window_distributions.png` (window feature
  densities).
- `benchmark.py` — parameter sweep harness using `--headless`
  mode and env-var overrides (`POSE_BENCH_*`).
- `--headless` flag on `main.py` skips pygame for batch metrics.
- `--metrics-detail` flag enables per-keypoint detail CSV.
- `sweep_default.yaml` — full parameter grid (8 params);
  use 1-2 at a time to keep runs tractable.
- `sweep_quick.yaml` — targeted first pass
  (`body_min_cutoff` × `det_smooth_alpha`, 6 combos).

Tuneable parameters are overridden via environment variables
(`POSE_BENCH_BODY_MIN_CUTOFF`, etc.) so that `benchmark.py`
can spawn isolated subprocess runs per combination.  Notable
additions: `carry_damping` (velocity decay for carry-forward
extrapolation, default 0.8) and `bone_distal_weight`
(proportion of bone-length correction applied to the distal
keypoint, default 0.8).

## Tests

Unit tests live in `tests/` and `test_extrapolation.py`:

- `tests/test_smoothing.py` — One Euro Filter, hand/body paths
- `tests/test_constraints.py` — bone-length and joint-angle
- `tests/test_matching.py` — Hungarian hand-to-arm matching
- `tests/test_detection.py` — NMS, detection smoothing,
  carry-forward
- `tests/test_processing.py` — synthetic hands, re-crop,
  affine helpers
- `test_extrapolation.py` — carry-forward extrapolation

## Tracking Modes

The `--tracking` CLI flag controls body-part scope:

- `hands` — palm + hand landmarks only, no pose detection
- `hands-arms` — 12 arm keypoints + hands (default)
- `body` — all 33 pose keypoints + hands

Mode-specific constants (keypoint indices, wrist/shoulder pairs,
arm chains) are defined in `processing.py:tracking_pose_indices()`.
Export column prefix is `arm_` for hands-arms, `body_` for body mode.

## Git Conventions

- Commit subject line: under 50 characters
- Commit body lines: under 72 characters
- Before committing, check whether README.md, .gitignore, or other
  housekeeping files need updates to stay consistent with the changes

## Environment

- Managed by `uv` (`pyproject.toml` + `uv.lock`)
- Python virtual environment in `.venv/`
- Host runs Linux with GNOME Wayland and Homebrew Python 3.14
- The `.venv` must be created on the host, not inside a container
  (absolute symlinks to the Python binary are not portable)

## Package Installation

- Python deps live in `pyproject.toml`; add new deps there and
  run `uv sync` on the host (the assistant cannot do this in
  the container because `.venv` symlinks are host-specific)

- `uv.lock` is committed for reproducible installs
- When installing R packages, use `renv` (not the global library)
