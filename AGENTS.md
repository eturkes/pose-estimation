# Arm & Hand Pose Estimation — Agent Context

## Project Overview

Real-time arm and hand pose estimation combining MediaPipe TFLite models with
Intel OpenVINO inference. Detects arm poses (12 keypoints) and hand landmarks
(21 keypoints per hand) from webcam or video, with temporal smoothing and
skeleton visualization.

## Architecture

| File             | Role                                                      |
|------------------|-----------------------------------------------------------|
| `main.py`        | Entry point, CLI, video capture loop, pygame display      |
| `models.py`      | Downloads MediaPipe TFLite models, converts to OpenVINO IR, compiles |
| `detection.py`   | SSD anchor generation, NMS, detection decoding            |
| `processing.py`  | Preprocessing, crop extraction, landmark inference, hand-arm matching |
| `drawing.py`     | Catmull-Rom splines, skeleton rendering, overlay blending |
| `smoothing.py`   | One Euro Filter for temporal smoothing of landmarks       |
| `export.py`      | CSV schema definition, per-frame landmark row conversion  |

## Batch Video Processing

Place videos in `videos/` and run:

```
python main.py --batch-dir videos/
```

To track only the most prominent person (e.g. the patient), add
`--single-subject`. This selects the body with the largest landmark
bounding box each frame and discards other detections:

```
python main.py --batch-dir videos/ --single-subject
```

Single-subject mode has three resilience layers for unreliable body
detection (e.g. top-down views with partial body visibility):

1. **Primary body selection** — when bodies are genuinely detected
   (not smoother carry-forward), keep the largest.  All hands that
   passed the age / spatial filters are preserved; only body-level
   matches are re-indexed to the primary body.
2. **Body carry-forward** — when real body detection drops out (as
   reported by `smooth_bodies`' `n_active` return value), reuse the
   last known body for up to 0.5 s so hand-arm matching can continue.
3. **Hand-only fallback** — when carry-forward expires (or no body was
   ever seen), export a row with blank arm columns and hand data
   assigned left/right by x-coordinate.

`smooth_bodies` returns a third value `n_active` (count of genuinely
matched bodies, excluding grace-period carry-forward).  Single-subject
mode uses this instead of `len(body_lm)` to decide whether body
detection succeeded, preventing the smoother's own carry-forward from
masking real detection dropouts.

Each video is displayed in real-time during processing. One CSV per video
is written to `output/` (configurable with `--output-dir`). CSVs contain
one row per person per frame with normalised (0–1) landmark coordinates
(178 columns total: 12 arm keypoints × 4 values + 2 × 21 hand keypoints
× 3 values + 4 metadata columns). Missing hand data is left blank.
With `--single-subject`, arm columns may also be blank in hand-only
fallback frames.

Both `videos/` and `output/` are git-ignored to prevent patient data from
being committed.

## Key Technical Details

- **Display**: Uses `pygame-ce` (SDL2) instead of `cv2.imshow` because
  OpenCV's bundled Qt backend does not render on Wayland.
- **Image processing**: Uses `opencv-python-headless` (no GUI module needed).
- **Models**: Downloaded from Google MediaPipe on first run, cached in `model/`.
  Converted from TFLite → OpenVINO IR (XML/BIN).
- **Inference devices**: NPU (default), CPU, GPU via OpenVINO.
- **Frame pipeline**: BGR capture → flip → resize → detect → landmark →
  smooth → match → (optional single-subject filter) → draw overlays →
  convert to RGB → pygame surface.

## Dependencies

Defined in `requirements.txt`:
- `openvino` — model compilation and inference
- `opencv-python-headless` — image processing (no GUI)
- `numpy` — numerical operations
- `pygame-ce` — display (SDL2, Wayland-compatible)
- `tqdm` — progress bars for model downloads
- `requests` — HTTP for model downloads

## Git Conventions

- Commit subject line: under 50 characters
- Commit body lines: under 72 characters
- Before committing, check whether README.md, .gitignore, or other
  housekeeping files need updates to stay consistent with the changes

## Environment

- Python virtual environment in `.venv/`
- Host runs Linux with GNOME Wayland and Homebrew Python 3.14
- The `.venv` must be created on the host, not inside a container
  (absolute symlinks to the Python binary are not portable)
