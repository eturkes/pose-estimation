# Arm & Hand Pose Estimation

Real-time arm and hand pose estimation using MediaPipe models with Intel OpenVINO inference.

Detects and tracks arm poses and hand landmarks from a webcam or video file, with temporal smoothing and skeleton visualization.

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

Models are downloaded automatically on first run and cached in the `model/` directory.

Press **ESC** to exit.
