"""Pose estimation — unified entry point.

Supports rtmlib-based models (RTMW, DWPose, RTMPose) and MediaPipe.

Usage:
    python -m pose_estimation.run                                     # webcam 0, default model
    python -m pose_estimation.run --model dwpose-m                    # DWPose wholebody
    python -m pose_estimation.run --model rtmpose-m                   # body-only (17 kps)
    python -m pose_estimation.run --model mediapipe                   # MediaPipe pose + hand
    python -m pose_estimation.run --source video.mp4 --backend openvino --device GPU
    python -m pose_estimation.run --backend openvino --device NPU
    python -m pose_estimation.run --source video.mp4 --backend openvino --device NPU --headless
    python -m pose_estimation.run --batch-dir videos/ --backend openvino --device NPU
    python -m pose_estimation.run --batch-dir videos/ --single-subject --backend openvino --device NPU --tracking hands-arms

Requirements:
    pip install rtmlib openvino  # or: pip install rtmlib onnxruntime
"""

import argparse
import collections
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .constraints import BoneLengthSmoother

# ---------------------------------------------------------------------------
# Model registry — NPU-compatible models (verified via scripts/npu_compat.py)
# ---------------------------------------------------------------------------
# Largest variant per model family; all use YOLOX-m for detection.

_DET_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
)
_DET_INPUT_SIZE = (640, 640)

MODEL_REGISTRY = {
    "rtmw-l": {
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"
        ),
        "pose_input_size": (192, 256),
        "pose_class": "RTMPose",
        "n_kps": 133,
        "label": "Wholebody 133 kps (RTMW-L, 256x192)",
    },
    "dwpose-m": {
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192"
            "-c8b76419_20230728.zip"
        ),
        "pose_input_size": (192, 256),
        "pose_class": "RTMPose",
        "n_kps": 133,
        "label": "Wholebody 133 kps (DWPose-M, 256x192)",
    },
    "rtmpose-m": {
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192"
            "-e48f03d0_20230504.zip"
        ),
        "pose_input_size": (192, 256),
        "pose_class": "RTMPose",
        "n_kps": 17,
        "label": "Body 17 kps (RTMPose-M, 256x192)",
    },
}

DEFAULT_MODEL = "rtmw-l"


# ---------------------------------------------------------------------------
# COCO-WholeBody 133 keypoint tracking masks
# ---------------------------------------------------------------------------
_KP_ARMS = {5, 6, 7, 8, 9, 10}  # shoulders, elbows, wrists
_KP_LHAND = set(range(91, 112))  # 21 left-hand landmarks
_KP_RHAND = set(range(112, 133))  # 21 right-hand landmarks

TRACKING_INDICES = {
    "hands": _KP_LHAND | _KP_RHAND,
    "hands-arms": _KP_ARMS | _KP_LHAND | _KP_RHAND,
    "body": set(range(133)),
}

# Per-region smoothing parameters for 133-keypoint COCO-WholeBody layout.
# Hands/fingers get lighter smoothing (higher min_cutoff) to preserve fast
# articulation; body, feet, and face get heavier smoothing.
# (name, start_index, end_index_exclusive, min_cutoff, beta)
REGION_PARAMS = [
    ("body", 0, 17, 0.3, 0.5),
    ("feet", 17, 23, 0.3, 0.5),
    ("face", 23, 91, 0.3, 0.5),
    ("hands", 91, 133, 1.0, 0.3),
]

# ---------------------------------------------------------------------------
# Bone-length constraint segments for COCO-WholeBody 133 layout
# ---------------------------------------------------------------------------
# Ordered proximal→distal so corrections propagate outward.
BONE_SEGMENTS_WB = [
    (5, 7),  # left shoulder → left elbow
    (7, 9),  # left elbow → left wrist
    (9, 91),  # left wrist → left index-finger MCP
    (6, 8),  # right shoulder → right elbow
    (8, 10),  # right elbow → right wrist
    (10, 112),  # right wrist → right index-finger MCP
]

BONE_SEGMENTS_WB_BODY = [
    *BONE_SEGMENTS_WB,
    (11, 13),  # left hip → left knee
    (13, 15),  # left knee → left ankle
    (12, 14),  # right hip → right knee
    (14, 16),  # right knee → right ankle
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
WINDOW_TITLE = "Pose Estimation"


def _frame_to_surface(frame):
    """Convert a BGR OpenCV frame to a pygame Surface."""
    import pygame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# Temporal smoothing — reduces frame-to-frame keypoint jitter
# ---------------------------------------------------------------------------


class _OneEuro:
    """Minimal One Euro Filter for array-valued signals."""

    def __init__(self, min_cutoff=0.5, beta=0.5, d_cutoff=1.0, gamma=2.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.gamma = gamma
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t, confidence=None):
        if self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        assert self.x_prev is not None
        assert self.dx_prev is not None
        dt = max(t - self.t_prev, 1e-6)
        a_d = 1.0 / (1.0 + 1.0 / (2 * np.pi * self.d_cutoff * dt))
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))
        x_hat = a * x + (1 - a) * self.x_prev

        # Confidence weighting: low-confidence keypoints are pulled toward
        # the previous position, resisting noisy input.
        if confidence is not None:
            w = np.clip(confidence, 0.0, 1.0)[:, None] ** self.gamma
            result = w * x_hat + (1 - w) * self.x_prev
        else:
            result = x_hat

        # ``result`` is returned and may be mutated by the caller, so keep
        # a private copy as state.  ``dx_hat`` is fresh and never returned.
        self.x_prev = result.copy()
        self.dx_prev = dx_hat
        self.t_prev = t
        return result


class KeypointSmoother:
    """Multi-person temporal smoother with track matching and carry-forward.

    Reduces jitter via One Euro Filters on keypoint positions and EMA on
    confidence scores.  Greedy nearest-centroid matching associates
    detections with persistent tracks across frames.  During brief
    detection dropouts, tracks carry forward with gradual score decay
    so the skeleton fades rather than vanishing abruptly.
    """

    SCORE_DECAY = 0.9  # per-frame score multiplier during carry-forward

    def __init__(
        self,
        min_cutoff=0.5,
        beta=0.5,
        score_alpha=0.5,
        carry_frames=5,
        match_thresh=150,
        carry_damping=0.8,
        min_track_age=3,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.score_alpha = score_alpha
        self.carry_frames = carry_frames
        self.match_thresh = match_thresh
        self.carry_damping = carry_damping
        self.min_track_age = min_track_age
        self.tracks = []

    def reset(self):
        """Clear all track state (e.g. between video sources)."""
        self.tracks = []

    def _make_filters(self, n_kps):
        """Create per-region or single filter depending on keypoint count."""
        if n_kps == 133:
            return {name: _OneEuro(min_cutoff=mc, beta=b) for name, _, _, mc, b in REGION_PARAMS}
        return {"all": _OneEuro(min_cutoff=self.min_cutoff, beta=self.beta)}

    def _apply_filters(self, filters, kp, t, confidence):
        """Apply region-aware or single filter to keypoints."""
        if "all" in filters:
            return filters["all"](kp, t, confidence=confidence)
        result = np.empty_like(kp)
        for name, start, end, _, _ in REGION_PARAMS:
            conf_slice = confidence[start:end] if confidence is not None else None
            result[start:end] = filters[name](kp[start:end], t, confidence=conf_slice)
        return result

    def _get_velocity(self, filters):
        """Extract concatenated velocity from region or single filters."""
        if "all" in filters:
            v = filters["all"].dx_prev
            return v.copy() if v is not None else None
        parts = []
        for name, _, _, _, _ in REGION_PARAMS:
            v = filters[name].dx_prev
            if v is None:
                return None
            parts.append(v)
        return np.concatenate(parts, axis=0)

    def _extrapolate(self, last_kps, last_velocity, last_t, t, misses):
        """Velocity-based extrapolation with exponential damping.

        Falls back to static carry when no velocity is available.
        Per-keypoint displacement is capped at match_thresh to
        prevent runaway drift from spurious velocity estimates.
        """
        if last_velocity is None:
            return last_kps
        dt = t - last_t
        if dt <= 0:
            return last_kps
        damping = self.carry_damping**misses
        step = last_velocity * dt * damping
        # Cap per-keypoint displacement magnitude
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        scale = np.minimum(1.0, self.match_thresh / norms)
        step *= scale
        return last_kps + step

    def __call__(self, keypoints, scores, t):
        """Return (smoothed_keypoints, smoothed_scores) or (None, None)."""
        if keypoints is None or len(keypoints.shape) != 3 or keypoints.shape[0] == 0:
            return self._carry(t)

        n_det = keypoints.shape[0]
        det_centroids = keypoints.mean(axis=1)

        matched, used_tracks = self._match(det_centroids)

        new_tracks = []
        out_kps = []
        out_scores = []

        for i in range(n_det):
            kp = keypoints[i]
            sc = scores[i]

            if i in matched:
                tr = self.tracks[matched[i]]
                filt = tr["filter"]
                prev_sc = tr["scores"]
                age = tr["age"] + 1
            else:
                filt = self._make_filters(kp.shape[0])
                prev_sc = sc
                age = 1

            smooth_kp = self._apply_filters(filt, kp, t, sc)
            smooth_sc = self.score_alpha * sc + (1 - self.score_alpha) * prev_sc

            new_tracks.append(
                {
                    "filter": filt,
                    "centroid": smooth_kp.mean(axis=0).copy(),
                    "scores": smooth_sc.copy(),
                    "misses": 0,
                    "age": age,
                    "last_kps": smooth_kp.copy(),
                    "last_velocity": self._get_velocity(filt),
                    "last_t": t,
                }
            )
            if age >= self.min_track_age:
                out_kps.append(smooth_kp)
                out_scores.append(smooth_sc)

        # Carry forward unmatched tracks within grace period.
        # Decrement age each missed frame so intermittent false
        # positives cannot accumulate age across grace gaps.
        for j, tr in enumerate(self.tracks):
            if j in used_tracks or tr["misses"] >= self.carry_frames:
                continue
            misses = tr["misses"] + 1
            age = max(0, tr["age"] - 1)
            predicted = self._extrapolate(
                tr["last_kps"], tr.get("last_velocity"), tr.get("last_t", 0), t, misses
            )
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append(
                {
                    "filter": tr["filter"],
                    "centroid": predicted.mean(axis=0).copy(),
                    "scores": decayed,
                    "misses": misses,
                    "age": age,
                    "last_kps": predicted,
                    "last_velocity": tr.get("last_velocity"),
                    "last_t": t,
                }
            )
            if age >= self.min_track_age:
                out_kps.append(predicted)
                out_scores.append(decayed)

        self.tracks = new_tracks
        if out_kps:
            return np.stack(out_kps), np.stack(out_scores)
        return None, None

    def _match(self, det_centroids):
        """Optimal nearest-centroid matching via Hungarian algorithm."""
        matched = {}
        used_tracks = set()
        if not self.tracks or len(det_centroids) == 0:
            return matched, used_tracks

        trk_c = np.array([tr["centroid"] for tr in self.tracks])
        cost = np.linalg.norm(det_centroids[:, None, :] - trk_c[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind, strict=False):
            if cost[r, c] < self.match_thresh:
                matched[int(r)] = int(c)
                used_tracks.add(int(c))

        return matched, used_tracks

    def _carry(self, t=None):
        """Emit carry-forward tracks when no detections are present."""
        new_tracks = []
        out_kps = []
        out_scores = []
        for tr in self.tracks:
            if tr["misses"] >= self.carry_frames:
                continue
            misses = tr["misses"] + 1
            age = max(0, tr["age"] - 1)
            if t is not None:
                predicted = self._extrapolate(
                    tr["last_kps"], tr.get("last_velocity"), tr.get("last_t", 0), t, misses
                )
            else:
                predicted = tr["last_kps"]
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append(
                {
                    "filter": tr["filter"],
                    "centroid": predicted.mean(axis=0).copy(),
                    "scores": decayed,
                    "misses": misses,
                    "age": age,
                    "last_kps": predicted,
                    "last_velocity": tr.get("last_velocity"),
                    "last_t": t if t is not None else tr.get("last_t", 0),
                }
            )
            if age >= self.min_track_age:
                out_kps.append(predicted)
                out_scores.append(decayed)
        self.tracks = new_tracks
        if out_kps:
            return np.stack(out_kps), np.stack(out_scores)
        return None, None


# ---------------------------------------------------------------------------
# Monkey-patch rtmlib's OpenVINO backend to support NPU / GPU devices
# ---------------------------------------------------------------------------
# rtmlib hardcodes device_name='CPU' in its OpenVINO backend.  The patch
# below overrides that so we can pass --device NPU (or GPU) and have it
# forwarded to OpenVINO's compile_model().  For NPU, models are also
# reshaped to static shapes (batch=1) before compilation.
_ORIG_BASE_INIT = None  # set lazily after import


def _patch_rtmlib_openvino():
    """Allow rtmlib's OpenVINO backend to use non-CPU devices.

    Also generalises the output-layer handling so that models with any
    number of outputs (e.g. 3 for RTMW3D) work correctly.
    """
    from rtmlib.tools import base as rtmlib_base

    global _ORIG_BASE_INIT
    if _ORIG_BASE_INIT is not None:
        return  # already patched

    _ORIG_BASE_INIT = rtmlib_base.BaseTool.__init__

    def _patched_init(
        self,
        onnx_model=None,
        model_input_size=None,
        mean=None,
        std=None,
        backend="opencv",
        device="cpu",
    ):
        if backend == "openvino":
            from pathlib import Path

            from openvino import Core
            from rtmlib.tools.file import download_checkpoint

            if onnx_model is None:
                raise ValueError("onnx_model is required for the openvino backend")
            if not Path(onnx_model).exists():
                onnx_model = download_checkpoint(onnx_model)

            core = Core()
            model_onnx = core.read_model(model=onnx_model)

            ov_device = device.upper() if device else "CPU"

            # NPU requires static shapes — freeze any dynamic dimensions
            if ov_device == "NPU":
                input_shape = model_onnx.input(0).partial_shape
                if input_shape.is_dynamic:
                    static = []
                    for dim in input_shape:
                        if dim.is_dynamic:
                            static.append(1)
                        else:
                            static.append(dim.get_length())
                    print(f"  Reshaping to static {static} for NPU")
                    model_onnx.reshape(static)

            try:
                self.compiled_model = core.compile_model(
                    model=model_onnx, device_name=ov_device, config={"PERFORMANCE_HINT": "LATENCY"}
                )
            except RuntimeError as exc:
                if ov_device != "CPU":
                    print(
                        f"WARNING: Failed to compile on {ov_device} ({exc}), falling back to CPU."
                    )
                    model_onnx = core.read_model(model=onnx_model)
                    self.compiled_model = core.compile_model(
                        model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY"}
                    )
                    ov_device = "CPU"
                else:
                    raise

            n_outputs = len(self.compiled_model.outputs)
            self.input_layer = self.compiled_model.input(0)
            self._ov_output_layers = [self.compiled_model.output(i) for i in range(n_outputs)]
            # Backward compat for rtmlib code that uses these directly
            self.output_layer0 = self._ov_output_layers[0]
            self.output_layer1 = self._ov_output_layers[1]

            print(f"load {onnx_model} with openvino/{ov_device} backend ({n_outputs} outputs)")

            self.onnx_model = onnx_model
            self.model_input_size = model_input_size
            self.mean = mean
            self.std = std
            self.backend = backend
            self.device = device
        else:
            # rtmlib typed these `str = None` / `tuple = None`; in practice
            # non-None is always passed when the non-openvino branch runs.
            assert onnx_model is not None
            assert model_input_size is not None
            assert mean is not None
            assert std is not None
            _ORIG_BASE_INIT(
                self,
                onnx_model=onnx_model,
                model_input_size=model_input_size,
                mean=mean,
                std=std,
                backend=backend,
                device=device,
            )

    rtmlib_base.BaseTool.__init__ = _patched_init  # ty: ignore[invalid-assignment]

    # Patch inference() so models with >2 outputs work.
    _orig_inference = rtmlib_base.BaseTool.inference

    def _patched_inference(self, img):
        if self.backend != "openvino" or not hasattr(self, "_ov_output_layers"):
            return _orig_inference(self, img)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input_tensor = img[None, :, :, :]

        results = self.compiled_model(input_tensor)
        return [results[layer] for layer in self._ov_output_layers]

    rtmlib_base.BaseTool.inference = _patched_inference  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


FALLBACK_FPS = 30.0
MIN_REASONABLE_FPS = 1.0
MAX_REASONABLE_FPS = 240.0


def _open_capture(source, source_str):
    """Open a VideoCapture with diagnostic error messages.

    *source* may be an int (camera index) or path string.  Returns the
    open capture or None after printing a context-aware reason.
    """
    if isinstance(source, str):
        path = pathlib.Path(source)
        if not path.exists():
            print(f"WARNING: file not found: {source_str}, skipping.")
            return None
        if not path.is_file():
            print(f"WARNING: not a regular file: {source_str}, skipping.")
            return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            print(f"WARNING: cannot open camera index {source}, skipping.")
        else:
            print(f"WARNING: cannot open {source_str} (codec issue?), skipping.")
        return None
    return cap


def _safe_fps(raw_fps):
    """Clamp/validate an FPS reading; fall back to FALLBACK_FPS."""
    if not np.isfinite(raw_fps) or raw_fps <= 0:
        return FALLBACK_FPS
    if raw_fps < MIN_REASONABLE_FPS or raw_fps > MAX_REASONABLE_FPS:
        print(f"WARNING: unusual FPS reported ({raw_fps:.2f}); using {FALLBACK_FPS}.")
        return FALLBACK_FPS
    return float(raw_fps)


def collect_video_files(batch_dir):
    """Return sorted list of video file paths in *batch_dir*."""
    batch_path = pathlib.Path(batch_dir)
    files = sorted(str(p) for p in batch_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise RuntimeError(f"No video files found in {batch_dir}")
    return files


def filter_single_subject(keypoints, scores):
    """Keep only the highest-confidence person."""
    if keypoints is None or len(keypoints.shape) != 3:
        return keypoints, scores
    if keypoints.shape[0] <= 1:
        return keypoints, scores
    mean_scores = scores.mean(axis=1)
    best = np.argmax(mean_scores)
    return keypoints[best : best + 1], scores[best : best + 1]


def mask_tracking_scores(scores, tracking_mode):
    """Zero out scores for keypoints outside the tracking scope.

    This causes draw_skeleton's kpt_thr filter to hide them.
    """
    if tracking_mode is None or tracking_mode == "body":
        return scores
    visible = TRACKING_INDICES[tracking_mode]
    n_kps = scores.shape[-1]
    masked = scores.copy()
    for i in range(n_kps):
        if i not in visible:
            masked[:, i] = 0.0
    return masked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Pose estimation")
    p.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    p.add_argument(
        "--batch-dir",
        default=None,
        help="Process all video files in a directory (overrides --source, implies --headless)",
    )
    p.add_argument(
        "--single-subject", action="store_true", help="Track only the highest-confidence person"
    )
    p.add_argument(
        "--backend",
        default="openvino",
        choices=["onnxruntime", "openvino", "opencv"],
        help="Inference backend (default: openvino)",
    )
    p.add_argument(
        "--device", default="NPU", help="Device for inference: NPU, CPU, GPU (default: NPU)"
    )
    p.add_argument(
        "--mode",
        default="balanced",
        choices=["performance", "balanced", "lightweight"],
        help="Model quality/speed tier (default: balanced)",
    )
    model_names = [*list(MODEL_REGISTRY.keys()), "mediapipe"]
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=model_names,
        help=(
            f"Pose model (default: {DEFAULT_MODEL}).  "
            + ", ".join(f"{k}: {v['label']}" for k, v in MODEL_REGISTRY.items())
            + ", mediapipe: MediaPipe pose + hand (TFLite)"
        ),
    )
    # Kept for backward compatibility; --model takes precedence.
    p.add_argument("--body-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--tracking",
        default="hands-arms",
        choices=["hands", "hands-arms", "body"],
        help="Keypoint scope (default: hands-arms). hands/hands-arms require Wholebody.",
    )
    p.add_argument(
        "--det-frequency", type=int, default=7, help="Run detector every N frames (default: 7)"
    )
    p.add_argument("--headless", action="store_true", help="No display — just print latency stats")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    p.add_argument("--no-smooth", action="store_true", help="Disable temporal smoothing")
    p.add_argument("--no-constraints", action="store_true", help="Disable bone-length constraints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------


def process_source(
    args, pose_tracker, source_str, draw_skeleton, smoother=None, bone_smoother=None, screen=None
):
    """Process a single video/camera source.  Returns latency list (ms)."""
    source = int(source_str) if source_str.isdigit() else source_str
    cap = _open_capture(source, source_str)
    if cap is None:
        return []

    fps_video = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Source:  {source_str} ({w}x{h} @ {fps_video:.1f} fps"
        f"{f', {total_frames} frames' if total_frames > 0 else ''})"
    )
    print()

    use_pygame = not args.headless and screen is not None
    if use_pygame:
        import pygame

    latencies = []
    processing_times = collections.deque(maxlen=60)
    frame_idx = 0
    try:
        while cap.isOpened():
            if use_pygame:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        cap.release()
                        return latencies
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        cap.release()
                        return latencies

            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                print(f"WARNING: skipping malformed frame {frame_idx}")
                continue
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            t0 = time.perf_counter()
            keypoints, scores = pose_tracker(frame)
            dt = time.perf_counter() - t0
            latencies.append(dt * 1000)

            if smoother is not None:
                keypoints, scores = smoother(keypoints, scores, t0)

            if args.single_subject:
                keypoints, scores = filter_single_subject(keypoints, scores)

            if bone_smoother is not None and keypoints is not None:
                for pi in range(keypoints.shape[0]):
                    keypoints[pi], _ = bone_smoother.update(pi, keypoints[pi])
                bone_smoother.prune(range(keypoints.shape[0]))

            n_persons = (
                keypoints.shape[0] if keypoints is not None and len(keypoints.shape) == 3 else 0
            )
            n_kps = keypoints.shape[1] if n_persons > 0 else 0

            if frame_idx <= 5 or frame_idx % 50 == 0:
                mean_lat = np.mean(latencies[-50:])
                print(
                    f"Frame {frame_idx:5d} | "
                    f"{dt * 1000:6.1f} ms | "
                    f"avg {mean_lat:6.1f} ms | "
                    f"{n_persons} person(s), {n_kps} kps"
                )

            if not args.headless:
                if n_persons > 0:
                    draw_scores = mask_tracking_scores(scores, args.tracking)
                    img_show = draw_skeleton(
                        frame.copy(), keypoints, draw_scores, openpose_skeleton=False, kpt_thr=0.3
                    )
                else:
                    img_show = frame

                processing_times.append(dt)
                avg_ms = np.mean(processing_times) * 1000
                fps = 1000 / avg_ms
                _, f_width = img_show.shape[:2]
                label = f"Inference: {avg_ms:.1f}ms ({fps:.1f} FPS)"
                if total_frames > 0:
                    pct = frame_idx / total_frames * 100
                    label += f"  |  Frame {frame_idx}/{total_frames} ({pct:.0f}%)"
                cv2.putText(
                    img_show,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_COMPLEX,
                    f_width / 1000,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                if use_pygame:
                    assert screen is not None  # implied by use_pygame
                    # Resize window to match first frame
                    if frame_idx == 1:
                        fh, fw = img_show.shape[:2]
                        screen = pygame.display.set_mode((fw, fh))
                        video_name = (
                            pathlib.Path(source_str).name if not source_str.isdigit() else None
                        )
                        if video_name:
                            pygame.display.set_caption(f"{WINDOW_TITLE} — {video_name}")
                    screen.blit(_frame_to_surface(img_show), (0, 0))
                    pygame.display.flip()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()

    return latencies


def print_latency_summary(latencies):
    """Print latency statistics."""
    if not latencies:
        return
    arr = np.array(latencies)
    # Skip first few frames (model warmup)
    warm = arr[min(3, len(arr)) :]
    print()
    print("─── Latency summary ───")
    print(f"  Frames processed: {len(arr)}")
    print(f"  Warmup (first 3):  {np.mean(arr[:3]):.1f} ms avg")
    if len(warm) > 0:
        print(
            f"  Steady-state:      {np.mean(warm):.1f} ms avg, "
            f"{np.median(warm):.1f} ms median, "
            f"{np.percentile(warm, 95):.1f} ms p95"
        )
        print(f"  Effective FPS:     {1000 / np.mean(warm):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_mediapipe(args):
    """Delegate to pose_estimation.main for the MediaPipe pipeline."""
    cmd = [sys.executable, "-m", "pose_estimation.main"]
    if args.batch_dir:
        cmd += ["--batch-dir", args.batch_dir]
    elif args.source != "0":
        cmd += ["--source", args.source]
    cmd += ["--device", args.device]
    cmd += ["--tracking", args.tracking]
    if args.single_subject:
        cmd.append("--single-subject")
    if args.headless:
        cmd.append("--headless")
    for flag, val in (("--no-smooth", args.no_smooth), ("--max-frames", args.max_frames)):
        if val:
            print(f"WARNING: {flag} is not supported by the MediaPipe pipeline; ignoring.")
    print(f"Delegating to MediaPipe pipeline: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    args = parse_args()

    # ── MediaPipe delegates to main.py ──────────────────────────────
    if args.model == "mediapipe":
        sys.exit(_run_mediapipe(args))

    # ── Resolve model — legacy --body-only maps to rtmpose-m ────────
    model_name = args.model
    if args.body_only and model_name == DEFAULT_MODEL:
        model_name = "rtmpose-m"

    model = MODEL_REGISTRY[model_name]

    # --tracking hands/hands-arms needs wholebody (133 kps)
    if args.tracking != "body" and model["n_kps"] == 17:
        print(
            f"NOTE: --tracking {args.tracking} requires wholebody; "
            f"switching from {model_name} to {DEFAULT_MODEL}."
        )
        model_name = DEFAULT_MODEL
        model = MODEL_REGISTRY[model_name]

    # ── Patch rtmlib before importing its classes ────────────────────
    if args.backend == "openvino":
        _patch_rtmlib_openvino()

    # ── Import rtmlib (deferred so --help works without it) ─────────
    from functools import partial

    from rtmlib import Custom, PoseTracker, draw_skeleton

    # ── Set up model (explicit URLs from registry for all devices) ──
    solution_cls = partial(
        Custom,
        det_class="YOLOX",
        det=_DET_URL,
        det_input_size=_DET_INPUT_SIZE,
        pose_class=model["pose_class"],
        pose=model["pose"],
        pose_input_size=model["pose_input_size"],
    )
    print(f"Model:   {model['label']} [{model_name}]")

    tracking_label = f", tracking={args.tracking}"
    single_label = ", single-subject" if args.single_subject else ""
    smooth_label = ", no-smooth" if args.no_smooth else ", smooth"
    constraint_label = ", no-constraints" if args.no_constraints else ""
    print(
        f"Backend: {args.backend}, device={args.device}"
        f"{tracking_label}{single_label}{smooth_label}"
        f"{constraint_label}"
    )

    pose_tracker = PoseTracker(
        solution_cls,  # ty: ignore[invalid-argument-type]  # rtmlib accepts any callable
        mode=args.mode,
        det_frequency=args.det_frequency,
        backend=args.backend,
        device=args.device,
        to_openpose=False,
    )

    smoother = None if args.no_smooth else KeypointSmoother()

    bone_smoother = None
    if not args.no_constraints:
        segments = BONE_SEGMENTS_WB_BODY if args.tracking == "body" else BONE_SEGMENTS_WB
        bone_smoother = BoneLengthSmoother(segments=segments)

    # ── Collect sources ─────────────────────────────────────────────
    if args.batch_dir:
        sources = collect_video_files(args.batch_dir)
        print(f"Batch:   {len(sources)} video(s) in {args.batch_dir}")
    else:
        sources = [args.source]

    # ── Display ─────────────────────────────────────────────────────
    screen = None
    if not args.headless:
        import pygame as _pg

        _pg.init()
        screen = _pg.display.set_mode((640, 480))
        _pg.display.set_caption(WINDOW_TITLE)

    # ── Process each source ─────────────────────────────────────────
    all_latencies = []
    try:
        for i, src in enumerate(sources):
            if len(sources) > 1:
                print(f"\n{'=' * 60}")
                print(f"[{i + 1}/{len(sources)}] {src}")
                print("=" * 60)

            if smoother is not None:
                smoother.reset()
            latencies = process_source(
                args,
                pose_tracker,
                src,
                draw_skeleton,
                smoother=smoother,
                bone_smoother=bone_smoother,
                screen=screen,
            )
            print_latency_summary(latencies)
            all_latencies.extend(latencies)
    finally:
        if not args.headless:
            import pygame as _pg

            _pg.quit()

    # ── Batch summary ───────────────────────────────────────────────
    if len(sources) > 1 and all_latencies:
        print(f"\n{'=' * 60}")
        print("BATCH SUMMARY")
        print("=" * 60)
        print_latency_summary(all_latencies)


if __name__ == "__main__":
    main()
