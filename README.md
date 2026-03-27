# Pose Estimation

Real-time pose estimation combining MediaPipe TFLite models with
Intel OpenVINO inference. Supports three tracking modes — hands only,
arms + hands (default), or whole body + hands — from a webcam or video,
with temporal smoothing and skeleton visualization.

## Requirements

- Python 3.10+
- An OpenVINO-compatible device (NPU, CPU, or GPU)

## Setup

```bash
uv sync
```



## Usage

```bash
# Default: webcam 0, NPU device, hands-arms tracking
python main.py

# Use a different camera
python main.py --source 1

# Use a video file
python main.py --source video.mp4

# Run inference on CPU instead of NPU
python main.py --device CPU

# Disable horizontal flip (useful for rear cameras)
python main.py --no-flip

# Specify a custom model cache directory
python main.py --model-dir /path/to/models
```

Models are downloaded automatically on first run and cached in the `model/`
directory (TFLite from Google MediaPipe, converted to OpenVINO IR).

Close the window or press **ESC** to exit.

### Tracking Modes

Use `--tracking` to select what body parts are tracked:

```bash
# Hands only — palm detection + 21 hand keypoints per hand
python main.py --tracking hands

# Arms + hands (default) — 12 arm keypoints + hand landmarks
python main.py --tracking hands-arms

# Whole body + hands — all 33 pose keypoints + hand landmarks
python main.py --tracking body
```

| Mode | Body keypoints | Hand keypoints | Pose detection |
|------|---------------|----------------|----------------|
| `hands` | — | 2 × 21 | Skipped |
| `hands-arms` | 12 (shoulders → finger bases) | 2 × 21 | Yes |
| `body` | 33 (face, torso, arms, legs) | 2 × 21 | Yes |

In **hands** mode, pose detection is skipped entirely, which improves
performance but disables arm-guided hand detection fallbacks.  Hands are
assigned left/right by wrist x-coordinate.

In **body** mode, all 33 MediaPipe Pose landmarks are tracked (including
face, torso, and leg keypoints), with biomechanical constraints extended
to knee joints.

## Batch Video Processing

Place videos in `videos/` and run:

```bash
python main.py --batch-dir videos/
```

Each video is displayed in real-time during processing. One CSV per video is
written to `output/` (configurable with `--output-dir`).

Add `--postprocess` to apply offline Savitzky-Golay smoothing to the exported
CSVs (writes `<stem>_smooth.csv` alongside each original):

```bash
python main.py --batch-dir videos/ --postprocess
python main.py --batch-dir videos/ --postprocess --savgol-window 15 --savgol-polyorder 3
```

The post-processing script can also be run standalone on existing CSVs:

```bash
python postprocess.py output/video1.csv --window 15 --polyorder 3
```

Both `videos/` and `output/` are git-ignored to prevent patient data from being
committed.

### Single-Subject Mode

To track only the most prominent person (e.g. the patient), add
`--single-subject`. This selects the body with the largest landmark bounding box
each frame and discards other detections:

```bash
python main.py --batch-dir videos/ --single-subject
```

Single-subject mode has three resilience layers for unreliable body detection
(e.g. top-down views with partial body visibility):

1. **Primary body selection** — when bodies are genuinely detected, keep the
   largest. All hands that passed the age/spatial filters are preserved; only
   body-level matches are re-indexed to the primary body.
2. **Body carry-forward** — when body detection drops out, reuse the last known
   body for up to 0.5 s so hands-arms matching can continue.
3. **Hand-only fallback** — when carry-forward expires (or no body was ever
   seen), export a row with blank arm columns and hand data assigned left/right
   by x-coordinate.

## CSV Output

CSVs contain one row per person per frame with normalised (0–1) landmark
coordinates. The number of columns depends on the tracking mode:

| Mode | Body columns | Hand columns | Metadata | Total |
|------|-------------|-------------|----------|-------|
| `hands` | — | 2 × 21 × 3 = 126 | 4 | 130 |
| `hands-arms` | 12 × 4 = 48 | 2 × 21 × 3 = 126 | 4 | 178 |
| `body` | 33 × 4 = 132 | 2 × 21 × 3 = 126 | 4 | 262 |

Body columns use the prefix `arm_` in hands-arms mode and `body_` in body mode.
Each body keypoint has x, y, z, and visibility values. Hand keypoints have
x, y, z only.

Missing hand data is left blank. With `--single-subject`, body columns may also
be blank in hand-only fallback frames.

## Optimisation & Analysis

### Headless mode

Run without display (faster, no pygame dependency at runtime):

```bash
python main.py --source video.mp4 --headless
```

This produces the same CSVs plus a `*_metrics.csv` with per-frame quality
metrics (jitter, confidence, smoothing deltas, constraint corrections, etc.).
Add `--metrics-detail` to also write a per-keypoint `*_kp_detail.csv` (large).

### Analysis suite (R)

R scripts in `analysis/` consume the metrics and landmark CSVs:

```bash
Rscript analysis/summary.R output/            # text report + JSON
Rscript analysis/timeseries.R output/          # temporal diagnostic plots
Rscript analysis/keypoint_detail.R output/     # per-keypoint analysis
Rscript analysis/compare.R run_a.json run_b.json  # before/after comparison
Rscript analysis/clinical_features.R output/  # clinical kinematic features
Rscript analysis/clinical_correlation.R output/ clinical_scores.csv  # correlate with scores
Rscript analysis/longitudinal.R output/ sessions.csv                # longitudinal recovery tracking
Rscript analysis/longitudinal.R output/ sessions.csv clinical_scores.csv  # with score overlay
Rscript analysis/explore_clinical.R output/         # exploratory summary & sanity-check plots
Rscript analysis/temporal_clinical.R output/        # per-video temporal feature plots
```

`clinical_features.R` reads landmark CSVs (hands-arms or body mode) and
computes per-frame joint angles, reach distances, grasp apertures, and
frame-to-frame displacement, plus per-window (1 s, 50 % overlap) spectral
arc length and wrist velocity statistics.  Outputs `*_clinical.csv`
(per-frame) and `*_clinical_windows.csv` (per-window) alongside each input.

`clinical_correlation.R` joins aggregated clinical features (mean, median,
SD, min, max per video) with an external clinical scores CSV on the `video`
column.  Computes pairwise Pearson and Spearman correlations with
Benjamini-Hochberg FDR correction.  Outputs a tidy
`*_correlation_table.csv`, a Spearman rho heatmap
(`*_correlation_matrix.png`), and scatter plots for the top 6 pairs
(`*_scatter_top.png`).

`longitudinal.R` compares clinical features from the same patient across
multiple sessions to track recovery.  Requires a `sessions.csv` mapping
each video to a `patient_id` and `session_date` (ISO 8601).  Aggregates
per-video features, computes session-to-session deltas and percentage
changes from baseline, and flags changes exceeding 1 SD of the baseline
session.  Outputs `*_longitudinal_summary.csv` and per-patient line plots
(`*_longitudinal_<patient_id>.png`).  Optionally overlays clinical scores
from `clinical_scores.csv` on a secondary axis.

`explore_clinical.R` loads all `*_clinical.csv` and `*_clinical_windows.csv`
files from a directory and produces an exploratory summary: per-video row
counts, per-column NA rates, summary statistics, and data-quality warnings.
Outputs `all_clinical_distributions.png` (density plots by video),
`all_clinical_na_heatmap.png` (missingness heatmap),
`all_clinical_boxplots.png` (box plots by video), and
`all_clinical_window_distributions.png` (window feature densities).

`temporal_clinical.R` visualises how clinical features evolve over time
within each video.  For every video with ≥10 per-frame rows, it produces
a multi-panel time-series plot with left and right sides overlaid, plus
window-level SAL and velocity panels where available.  Outputs one
`<stem>_clinical_timeseries.png` per video and a summary
`all_clinical_timeseries_overview.png` comparing one key feature
(normalised reach) across all videos.

### Parameter benchmarking

Sweep pipeline parameters and compare results:

```bash
python benchmark.py --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5 1.0
python benchmark.py --source video.mp4 --sweep body_beta 0.3 0.5 0.7 --sweep hand_min_cutoff 0.5 1.0 2.0
```

Available sweep parameters: `det_score_thresh`, `hand_flag_thresh`,
`body_min_cutoff`, `body_beta`, `hand_min_cutoff`, `hand_beta`,
`confidence_gamma`, `det_smooth_alpha`, `bone_ema_alpha`, `bone_tolerance`,
`bone_distal_weight`, `carry_grace`, `carry_damping`.

Curated sweep configurations are provided as YAML files:

```bash
# Full grid (8 params — run 1-2 at a time)
python benchmark.py --source video.mp4 --config sweep_default.yaml

# Quick first pass (body smoothing × detection EMA, 6 combos)
python benchmark.py --source video.mp4 --config sweep_quick.yaml
```

## Architecture

| File | Role |
|------|------|
| `main.py` | Entry point, CLI, video capture loop, pygame display |
| `models.py` | Downloads MediaPipe TFLite models, converts to OpenVINO IR, compiles |
| `detection.py` | SSD anchor generation, NMS, detection decoding |
| `processing.py` | Preprocessing, crop extraction, landmark inference, hands-arms matching |
| `drawing.py` | Catmull-Rom splines, skeleton rendering, overlay blending |
| `smoothing.py` | One Euro Filter with confidence-weighted temporal smoothing |
| `constraints.py` | Biomechanical constraints (bone-length consistency, joint-angle limits) |
| `export.py` | CSV schema definition, per-frame landmark row conversion |
| `postprocess.py` | Savitzky-Golay offline smoothing for exported CSVs |
| `metrics.py` | Per-frame quality metrics collection for optimisation |
| `benchmark.py` | Parameter sweep harness (headless) |
| `analysis/` | R scripts for metrics summarisation, visualisation, comparison |

## Technical Details

- **Display**: Uses `pygame-ce` (SDL2) instead of `cv2.imshow` because
  OpenCV's bundled Qt backend does not render on Wayland.
- **Image processing**: Uses `opencv-python-headless` (no GUI module needed).
- **Inference devices**: NPU (default), CPU, GPU via OpenVINO.
- **Hand-to-arm matching**: Uses Hungarian (optimal) assignment via
  `scipy.optimize.linear_sum_assignment` to pair detected hands with arm
  wrists, with a distality check to reject hands closer to the shoulder
  midpoint than the wrist.
- **Frame pipeline**: BGR capture → flip → resize → detect → arm-guided
  hand ROI fallback → landmark → smooth (confidence-weighted) →
  bone-length constraints → joint-angle limits → match →
  (optional single-subject filter) → draw overlays → convert to RGB →
  pygame surface.

## Dependencies

Defined in `pyproject.toml`. Core dependencies:

| Package | Purpose |
|---------|---------|
| `openvino` | Model compilation and inference |
| `opencv-python-headless` | Image processing (no GUI) |
| `numpy` | Numerical operations |
| `scipy` | Hungarian matching, Savitzky-Golay filter |
| `pygame-ce` | Display (SDL2, Wayland-compatible) |
| `tqdm` | Progress bars for model downloads |
| `requests` | HTTP for model downloads |
| `rtmlib` | RTMW/RTMPose/DWPose whole-body pose estimation (ONNX/OpenVINO) |
| `pandas` | CSV reading/writing for post-processing |
| `pyyaml` | YAML sweep config parsing for benchmarking |

## Unified Entry Point

`run.py` is the main entry point for pose estimation. It supports multiple
model backends via the `--model` flag:

| Model | Keypoints | Description |
|-------|-----------|-------------|
| `rtmw-l` (default) | 133 | RTMW-L wholebody (body + hands + face + feet) |
| `dwpose-m` | 133 | DWPose-M wholebody |
| `rtmpose-m` | 17 | RTMPose-M body only |
| `mediapipe` | — | MediaPipe pose + hand (delegates to `main.py`) |

rtmlib-based models use [rtmlib](https://github.com/Tau-J/rtmlib) for
lightweight ONNX/OpenVINO inference without mmcv/mmpose dependencies.

```bash
# Quick test on webcam (default: rtmw-l)
python run.py

# Try different models
python run.py --model dwpose-m
python run.py --model rtmpose-m
python run.py --model mediapipe

# Video with OpenVINO backend
python run.py --source video.mp4 --backend openvino

# Benchmark latency without display
python run.py --source video.mp4 --headless
```

Models are downloaded automatically on first run.
