"""Prototype: RTMW whole-body pose estimation via rtmlib.

Quick proof-of-concept to evaluate RTMW model quality and inference
speed on your hardware before integrating into the main pipeline.

Usage:
    # Webcam (default, onnxruntime CPU)
    python prototype_rtmw.py

    # Video file on GPU
    python prototype_rtmw.py --source video.mp4 --backend openvino --device GPU

    # NPU (uses yolox-m + rtmw-m, the NPU-compatible pair)
    python prototype_rtmw.py --backend openvino --device NPU

    # Headless benchmark
    python prototype_rtmw.py --source video.mp4 --backend openvino --device NPU --headless

    # Body-only (17 kps) instead of whole-body (133 kps)
    python prototype_rtmw.py --body-only --backend openvino --device NPU

    # Batch processing all videos in a directory
    python prototype_rtmw.py --batch-dir videos/ --backend openvino --device NPU

    # Single subject, hands-arms tracking scope
    python prototype_rtmw.py --batch-dir videos/ --single-subject --backend openvino --device NPU --tracking hands-arms

Requirements:
    pip install rtmlib openvino  # or: pip install rtmlib onnxruntime
"""

import argparse
import os
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from constraints import BoneLengthSmoother


# ---------------------------------------------------------------------------
# NPU-compatible model URLs (verified via test_npu_compat.py)
# ---------------------------------------------------------------------------
# NPU requires static shapes and doesn't support all ops.  These specific
# model variants have been tested and compile successfully on NPU.
NPU_MODELS = {
    # Detector: yolox-m is the only YOLOX variant that passes NPU compilation
    "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "det_input_size": (640, 640),
    # Whole-body pose: rtmw-l-m (133 kps) — best quality that compiles on NPU
    "pose_wholebody": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
    "pose_wholebody_input_size": (192, 256),
    # Body-only pose: rtmpose-m (17 kps)
    "pose_body": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    "pose_body_input_size": (192, 256),
}


# ---------------------------------------------------------------------------
# COCO-WholeBody 133 keypoint tracking masks
# ---------------------------------------------------------------------------
_KP_ARMS = {5, 6, 7, 8, 9, 10}           # shoulders, elbows, wrists
_KP_LHAND = set(range(91, 112))           # 21 left-hand landmarks
_KP_RHAND = set(range(112, 133))          # 21 right-hand landmarks

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
BONE_SEGMENTS_RTMW = [
    (5, 7),    # left shoulder → left elbow
    (7, 9),    # left elbow → left wrist
    (9, 91),   # left wrist → left index-finger MCP
    (6, 8),    # right shoulder → right elbow
    (8, 10),   # right elbow → right wrist
    (10, 112), # right wrist → right index-finger MCP
]

BONE_SEGMENTS_RTMW_BODY = BONE_SEGMENTS_RTMW + [
    (11, 13),  # left hip → left knee
    (13, 15),  # left knee → left ankle
    (12, 14),  # right hip → right knee
    (14, 16),  # right knee → right ankle
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


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

        self.x_prev = result.copy()
        self.dx_prev = dx_hat.copy()
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

    def __init__(self, min_cutoff=0.5, beta=0.5, score_alpha=0.5,
                 carry_frames=5, match_thresh=150, carry_damping=0.8,
                 min_track_age=3):
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
            return {name: _OneEuro(min_cutoff=mc, beta=b)
                    for name, _, _, mc, b in REGION_PARAMS}
        return {"all": _OneEuro(min_cutoff=self.min_cutoff, beta=self.beta)}

    def _apply_filters(self, filters, kp, t, confidence):
        """Apply region-aware or single filter to keypoints."""
        if "all" in filters:
            return filters["all"](kp, t, confidence=confidence)
        result = np.empty_like(kp)
        for name, start, end, _, _ in REGION_PARAMS:
            conf_slice = (confidence[start:end]
                          if confidence is not None else None)
            result[start:end] = filters[name](
                kp[start:end], t, confidence=conf_slice)
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
        damping = self.carry_damping ** misses
        step = last_velocity * dt * damping
        # Cap per-keypoint displacement magnitude
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        scale = np.minimum(1.0, self.match_thresh / norms)
        step *= scale
        return last_kps + step

    def __call__(self, keypoints, scores, t):
        """Return (smoothed_keypoints, smoothed_scores) or (None, None)."""
        if (keypoints is None or len(keypoints.shape) != 3
                or keypoints.shape[0] == 0):
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
            smooth_sc = (self.score_alpha * sc
                         + (1 - self.score_alpha) * prev_sc)

            new_tracks.append({
                "filter": filt,
                "centroid": smooth_kp.mean(axis=0).copy(),
                "scores": smooth_sc.copy(),
                "misses": 0,
                "age": age,
                "last_kps": smooth_kp.copy(),
                "last_velocity": self._get_velocity(filt),
                "last_t": t,
            })
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
                tr["last_kps"], tr.get("last_velocity"),
                tr.get("last_t", 0), t, misses)
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append({
                "filter": tr["filter"],
                "centroid": predicted.mean(axis=0).copy(),
                "scores": decayed,
                "misses": misses,
                "age": age,
                "last_kps": predicted,
                "last_velocity": tr.get("last_velocity"),
                "last_t": t,
            })
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
        cost = np.linalg.norm(
            det_centroids[:, None, :] - trk_c[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
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
                    tr["last_kps"], tr.get("last_velocity"),
                    tr.get("last_t", 0), t, misses)
            else:
                predicted = tr["last_kps"]
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append({
                "filter": tr["filter"],
                "centroid": predicted.mean(axis=0).copy(),
                "scores": decayed,
                "misses": misses,
                "age": age,
                "last_kps": predicted,
                "last_velocity": tr.get("last_velocity"),
                "last_t": t if t is not None else tr.get("last_t", 0),
            })
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
    """Allow rtmlib's OpenVINO backend to use non-CPU devices."""
    from rtmlib.tools import base as rtmlib_base

    global _ORIG_BASE_INIT
    if _ORIG_BASE_INIT is not None:
        return  # already patched

    _ORIG_BASE_INIT = rtmlib_base.BaseTool.__init__

    def _patched_init(self, onnx_model=None, model_input_size=None,
                      mean=None, std=None, backend='opencv', device='cpu'):
        if backend == 'openvino':
            import os

            from openvino import Core

            from rtmlib.tools.file import download_checkpoint

            if not os.path.exists(onnx_model):
                onnx_model = download_checkpoint(onnx_model)

            core = Core()
            model_onnx = core.read_model(model=onnx_model)

            ov_device = device.upper() if device else 'CPU'

            # NPU requires static shapes — freeze any dynamic dimensions
            if ov_device == 'NPU':
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
                    model=model_onnx,
                    device_name=ov_device,
                    config={'PERFORMANCE_HINT': 'LATENCY'})
            except RuntimeError as exc:
                if ov_device != 'CPU':
                    print(f"WARNING: Failed to compile on {ov_device} "
                          f"({exc}), falling back to CPU.")
                    # Re-read without reshape for CPU fallback
                    model_onnx = core.read_model(model=onnx_model)
                    self.compiled_model = core.compile_model(
                        model=model_onnx,
                        device_name='CPU',
                        config={'PERFORMANCE_HINT': 'LATENCY'})
                    ov_device = 'CPU'
                else:
                    raise

            self.input_layer = self.compiled_model.input(0)
            self.output_layer0 = self.compiled_model.output(0)
            self.output_layer1 = self.compiled_model.output(1)

            print(f'load {onnx_model} with openvino/{ov_device} backend')

            self.onnx_model = onnx_model
            self.model_input_size = model_input_size
            self.mean = mean
            self.std = std
            self.backend = backend
            self.device = device
        else:
            _ORIG_BASE_INIT(self, onnx_model=onnx_model,
                            model_input_size=model_input_size,
                            mean=mean, std=std,
                            backend=backend, device=device)

    rtmlib_base.BaseTool.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_video_files(batch_dir):
    """Return sorted list of video file paths in *batch_dir*."""
    files = []
    for f in sorted(os.listdir(batch_dir)):
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
            files.append(os.path.join(batch_dir, f))
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
    return keypoints[best:best + 1], scores[best:best + 1]


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
    p = argparse.ArgumentParser(description="RTMW prototype")
    p.add_argument("--source", default="0",
                   help="Camera index or video path (default: 0)")
    p.add_argument("--batch-dir", default=None,
                   help="Process all video files in a directory "
                        "(overrides --source, implies --headless)")
    p.add_argument("--single-subject", action="store_true",
                   help="Track only the highest-confidence person")
    p.add_argument("--backend", default="onnxruntime",
                   choices=["onnxruntime", "openvino", "opencv"],
                   help="Inference backend (default: onnxruntime)")
    p.add_argument("--device", default="cpu",
                   help="Device for inference: cpu, NPU, GPU (default: cpu)")
    p.add_argument("--mode", default="balanced",
                   choices=["performance", "balanced", "lightweight"],
                   help="Model quality/speed tier (default: balanced)")
    p.add_argument("--body-only", action="store_true",
                   help="Use Body (17 kps) instead of Wholebody (133 kps)")
    p.add_argument("--tracking", default=None,
                   choices=["hands", "hands-arms", "body"],
                   help="Keypoint scope for visualization "
                        "(hands/hands-arms require Wholebody, "
                        "overrides --body-only)")
    p.add_argument("--det-frequency", type=int, default=7,
                   help="Run detector every N frames (default: 7)")
    p.add_argument("--headless", action="store_true",
                   help="No display — just print latency stats")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited)")
    p.add_argument("--no-smooth", action="store_true",
                   help="Disable temporal smoothing")
    p.add_argument("--no-constraints", action="store_true",
                   help="Disable bone-length constraints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

def process_source(args, pose_tracker, source_str, draw_skeleton,
                   smoother=None, bone_smoother=None):
    """Process a single video/camera source.  Returns latency list (ms)."""
    source = int(source_str) if source_str.isdigit() else source_str
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"WARNING: Cannot open {source_str}, skipping.")
        return []

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source:  {source_str} ({w}x{h} @ {fps_video:.1f} fps"
          f"{f', {total_frames} frames' if total_frames > 0 else ''})")
    print()

    latencies = []
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
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
                    keypoints[pi], _ = bone_smoother.update(
                        pi, keypoints[pi])
                bone_smoother.prune(range(keypoints.shape[0]))

            n_persons = (keypoints.shape[0]
                         if keypoints is not None
                         and len(keypoints.shape) == 3
                         else 0)
            n_kps = keypoints.shape[1] if n_persons > 0 else 0

            # Print periodic stats
            if frame_idx <= 5 or frame_idx % 50 == 0:
                mean_lat = np.mean(latencies[-50:])
                print(f"Frame {frame_idx:5d} | "
                      f"{dt*1000:6.1f} ms | "
                      f"avg {mean_lat:6.1f} ms | "
                      f"{n_persons} person(s), {n_kps} kps")

            if not args.headless:
                img_show = frame.copy()
                if n_persons > 0:
                    draw_scores = mask_tracking_scores(
                        scores, args.tracking)
                    img_show = draw_skeleton(
                        img_show, keypoints, draw_scores,
                        openpose_skeleton=False, kpt_thr=0.3)

                # Overlay FPS
                fps_text = (f"{1000 / latencies[-1]:.0f} fps"
                            if latencies[-1] > 0 else "")
                cv2.putText(img_show, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img_show, f"{n_persons} person(s)", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("RTMW Prototype", img_show)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    return latencies


def print_latency_summary(latencies):
    """Print latency statistics."""
    if not latencies:
        return
    arr = np.array(latencies)
    # Skip first few frames (model warmup)
    warm = arr[min(3, len(arr)):]
    print()
    print("─── Latency summary ───")
    print(f"  Frames processed: {len(arr)}")
    print(f"  Warmup (first 3):  {np.mean(arr[:3]):.1f} ms avg")
    if len(warm) > 0:
        print(f"  Steady-state:      {np.mean(warm):.1f} ms avg, "
              f"{np.median(warm):.1f} ms median, "
              f"{np.percentile(warm, 95):.1f} ms p95")
        print(f"  Effective FPS:     {1000 / np.mean(warm):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --tracking hands/hands-arms needs Wholebody (133 kps)
    if args.tracking and args.tracking != "body" and args.body_only:
        print("NOTE: --tracking overrides --body-only; "
              "using Wholebody model.")
        args.body_only = False

    # ── Patch rtmlib before importing its classes ────────────────────
    if args.backend == "openvino":
        _patch_rtmlib_openvino()

    # ── Import rtmlib (deferred so --help works without it) ─────────
    from rtmlib import Body, Custom, PoseTracker, Wholebody, draw_skeleton

    # ── Set up model ────────────────────────────────────────────────
    is_npu = args.device.upper() == "NPU"

    if is_npu:
        # Use NPU-verified models explicitly via Custom class
        from functools import partial

        pose_key = "pose_body" if args.body_only else "pose_wholebody"
        solution_cls = partial(
            Custom,
            det_class="YOLOX",
            det=NPU_MODELS["det"],
            det_input_size=NPU_MODELS["det_input_size"],
            pose_class="RTMPose",
            pose=NPU_MODELS[pose_key],
            pose_input_size=NPU_MODELS[f"{pose_key}_input_size"],
        )
        label = "Body (17 kps)" if args.body_only else "Wholebody (133 kps)"
        print(f"Model:   {label}, NPU-verified (yolox-m + "
              f"{'rtmpose-m' if args.body_only else 'rtmw-m'})")
    else:
        # Use rtmlib's built-in mode selection
        solution_cls = Body if args.body_only else Wholebody
        label = "Body (17 kps)" if args.body_only else "Wholebody (133 kps)"
        print(f"Model:   {label}, mode={args.mode}")

    tracking_label = f", tracking={args.tracking}" if args.tracking else ""
    single_label = ", single-subject" if args.single_subject else ""
    smooth_label = ", no-smooth" if args.no_smooth else ", smooth"
    constraint_label = ", no-constraints" if args.no_constraints else ""
    print(f"Backend: {args.backend}, device={args.device}"
          f"{tracking_label}{single_label}{smooth_label}"
          f"{constraint_label}")

    pose_tracker = PoseTracker(
        solution_cls,
        mode=args.mode,
        det_frequency=args.det_frequency,
        backend=args.backend,
        device=args.device,
        to_openpose=False,
    )

    smoother = None if args.no_smooth else KeypointSmoother()

    bone_smoother = None
    if not args.no_constraints:
        segments = (BONE_SEGMENTS_RTMW_BODY
                    if args.tracking == "body" or args.body_only
                    else BONE_SEGMENTS_RTMW)
        bone_smoother = BoneLengthSmoother(segments=segments)

    # ── Collect sources ─────────────────────────────────────────────
    if args.batch_dir:
        sources = collect_video_files(args.batch_dir)
        print(f"Batch:   {len(sources)} video(s) in {args.batch_dir}")
    else:
        sources = [args.source]

    # ── Process each source ─────────────────────────────────────────
    all_latencies = []
    for i, src in enumerate(sources):
        if len(sources) > 1:
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(sources)}] {src}")
            print("=" * 60)

        if smoother is not None:
            smoother.reset()
        latencies = process_source(args, pose_tracker, src, draw_skeleton,
                                   smoother=smoother,
                                   bone_smoother=bone_smoother)
        print_latency_summary(latencies)
        all_latencies.extend(latencies)

    # ── Batch summary ───────────────────────────────────────────────
    if len(sources) > 1 and all_latencies:
        print(f"\n{'=' * 60}")
        print("BATCH SUMMARY")
        print("=" * 60)
        print_latency_summary(all_latencies)


if __name__ == "__main__":
    main()
