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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

Four R scripts in `analysis/` consume the metrics CSVs:

```bash
Rscript analysis/summary.R output/            # text report + JSON
Rscript analysis/timeseries.R output/          # temporal diagnostic plots
Rscript analysis/keypoint_detail.R output/     # per-keypoint analysis
Rscript analysis/compare.R run_a.json run_b.json  # before/after comparison
```

### Parameter benchmarking

Sweep pipeline parameters and compare results:

```bash
python benchmark.py --source video.mp4 --sweep body_min_cutoff 0.1 0.3 0.5 1.0
python benchmark.py --source video.mp4 --sweep body_beta 0.3 0.5 0.7 --sweep hand_min_cutoff 0.5 1.0 2.0
```

Available sweep parameters: `det_score_thresh`, `hand_flag_thresh`,
`body_min_cutoff`, `body_beta`, `hand_min_cutoff`, `hand_beta`,
`confidence_gamma`, `det_smooth_alpha`, `bone_ema_alpha`, `bone_tolerance`,
`carry_grace`.

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
- **Frame pipeline**: BGR capture → flip → resize → detect → arm-guided
  hand ROI fallback → landmark → smooth (confidence-weighted) →
  bone-length constraints → joint-angle limits → match →
  (optional single-subject filter) → draw overlays → convert to RGB →
  pygame surface.

## Dependencies

Defined in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `openvino` | Model compilation and inference |
| `opencv-python-headless` | Image processing (no GUI) |
| `numpy` | Numerical operations |
| `scipy` | Hungarian matching, Savitzky-Golay filter |
| `pygame-ce` | Display (SDL2, Wayland-compatible) |
| `tqdm` | Progress bars for model downloads |
| `requests` | HTTP for model downloads |

Optional (for `--postprocess` / `postprocess.py`):

| Package | Purpose |
|---------|---------|
| `pandas` | CSV reading/writing for post-processing |
