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
