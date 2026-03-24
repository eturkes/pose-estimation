# Arm & Hand Pose Estimation

Real-time arm and hand pose estimation combining MediaPipe TFLite models with
Intel OpenVINO inference. Detects arm poses (12 keypoints) and hand landmarks
(21 keypoints per hand) from a webcam or video, with temporal smoothing and
skeleton visualization.

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
# Default: webcam 0, NPU device
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

## Batch Video Processing

Place videos in `videos/` and run:

```bash
python main.py --batch-dir videos/
```

Each video is displayed in real-time during processing. One CSV per video is
written to `output/` (configurable with `--output-dir`).

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
   body for up to 0.5 s so hand-arm matching can continue.
3. **Hand-only fallback** — when carry-forward expires (or no body was ever
   seen), export a row with blank arm columns and hand data assigned left/right
   by x-coordinate.

## CSV Output

CSVs contain one row per person per frame with normalised (0–1) landmark
coordinates (178 columns total):

| Component | Columns |
|-----------|---------|
| Arm keypoints | 12 × 4 values (x, y, z, visibility) |
| Hand keypoints | 2 × 21 × 3 values (x, y, z per hand) |
| Metadata | 4 columns |

Missing hand data is left blank. With `--single-subject`, arm columns may also
be blank in hand-only fallback frames.

## Architecture

| File | Role |
|------|------|
| `main.py` | Entry point, CLI, video capture loop, pygame display |
| `models.py` | Downloads MediaPipe TFLite models, converts to OpenVINO IR, compiles |
| `detection.py` | SSD anchor generation, NMS, detection decoding |
| `processing.py` | Preprocessing, crop extraction, landmark inference, hand-arm matching |
| `drawing.py` | Catmull-Rom splines, skeleton rendering, overlay blending |
| `smoothing.py` | One Euro Filter for temporal smoothing of landmarks |
| `constraints.py` | Biomechanical constraints (bone-length consistency) |
| `export.py` | CSV schema definition, per-frame landmark row conversion |

## Technical Details

- **Display**: Uses `pygame-ce` (SDL2) instead of `cv2.imshow` because
  OpenCV's bundled Qt backend does not render on Wayland.
- **Image processing**: Uses `opencv-python-headless` (no GUI module needed).
- **Inference devices**: NPU (default), CPU, GPU via OpenVINO.
- **Frame pipeline**: BGR capture → flip → resize → detect → landmark →
  smooth → bone-length constraints → match → (optional single-subject filter) → draw overlays →
  convert to RGB → pygame surface.

## Dependencies

Defined in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `openvino` | Model compilation and inference |
| `opencv-python-headless` | Image processing (no GUI) |
| `numpy` | Numerical operations |
| `pygame-ce` | Display (SDL2, Wayland-compatible) |
| `tqdm` | Progress bars for model downloads |
| `requests` | HTTP for model downloads |
