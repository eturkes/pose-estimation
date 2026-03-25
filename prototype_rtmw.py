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


# ---------------------------------------------------------------------------
# NPU-compatible model URLs (verified via test_npu_compat.py)
# ---------------------------------------------------------------------------
# NPU requires static shapes and doesn't support all ops.  These specific
# model variants have been tested and compile successfully on NPU.
NPU_MODELS = {
    # Detector: yolox-m is the only YOLOX variant that passes NPU compilation
    "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "det_input_size": (640, 640),
    # Whole-body pose: rtmw-m (133 kps) — best quality that compiles on NPU
    "pose_wholebody": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip",
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

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


# ---------------------------------------------------------------------------
# Temporal smoothing — reduces frame-to-frame keypoint jitter
# ---------------------------------------------------------------------------

class _OneEuro:
    """Minimal One Euro Filter for array-valued signals."""

    def __init__(self, min_cutoff=0.5, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
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

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t
        return x_hat


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
                 carry_frames=5, match_thresh=150):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.score_alpha = score_alpha
        self.carry_frames = carry_frames
        self.match_thresh = match_thresh
        self.tracks = []

    def reset(self):
        """Clear all track state (e.g. between video sources)."""
        self.tracks = []

    def __call__(self, keypoints, scores, t):
        """Return (smoothed_keypoints, smoothed_scores) or (None, None)."""
        if (keypoints is None or len(keypoints.shape) != 3
                or keypoints.shape[0] == 0):
            return self._carry()

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
            else:
                filt = _OneEuro(min_cutoff=self.min_cutoff, beta=self.beta)
                prev_sc = sc

            smooth_kp = filt(kp, t)
            smooth_sc = (self.score_alpha * sc
                         + (1 - self.score_alpha) * prev_sc)

            new_tracks.append({
                "filter": filt,
                "centroid": smooth_kp.mean(axis=0).copy(),
                "scores": smooth_sc.copy(),
                "misses": 0,
                "last_kps": smooth_kp.copy(),
            })
            out_kps.append(smooth_kp)
            out_scores.append(smooth_sc)

        # Carry forward unmatched tracks within grace period
        for j, tr in enumerate(self.tracks):
            if j in used_tracks or tr["misses"] >= self.carry_frames:
                continue
            misses = tr["misses"] + 1
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append({
                "filter": tr["filter"],
                "centroid": tr["centroid"],
                "scores": decayed,
                "misses": misses,
                "last_kps": tr["last_kps"],
            })
            out_kps.append(tr["last_kps"])
            out_scores.append(decayed)

        self.tracks = new_tracks
        if out_kps:
            return np.stack(out_kps), np.stack(out_scores)
        return None, None

    def _match(self, det_centroids):
        """Greedy nearest-centroid matching."""
        matched = {}
        used_tracks = set()
        if not self.tracks or len(det_centroids) == 0:
            return matched, used_tracks

        trk_c = np.array([tr["centroid"] for tr in self.tracks])
        cost = np.linalg.norm(
            det_centroids[:, None, :] - trk_c[None, :, :], axis=2)
        for _ in range(min(len(det_centroids), len(self.tracks))):
            val = cost.min()
            if val >= self.match_thresh:
                break
            i, j = np.unravel_index(cost.argmin(), cost.shape)
            matched[int(i)] = int(j)
            used_tracks.add(int(j))
            cost[i, :] = np.inf
            cost[:, j] = np.inf

        return matched, used_tracks

    def _carry(self):
        """Emit carry-forward tracks when no detections are present."""
        new_tracks = []
        out_kps = []
        out_scores = []
        for tr in self.tracks:
            if tr["misses"] >= self.carry_frames:
                continue
            misses = tr["misses"] + 1
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append({
                "filter": tr["filter"],
                "centroid": tr["centroid"],
                "scores": decayed,
                "misses": misses,
                "last_kps": tr["last_kps"],
            })
            out_kps.append(tr["last_kps"])
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

def process_source(args, pose_tracker, source_str, draw_skeleton,
                   smoother=None):
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
    print(f"Backend: {args.backend}, device={args.device}"
          f"{tracking_label}{single_label}{smooth_label}")

    pose_tracker = PoseTracker(
        solution_cls,
        mode=args.mode,
        det_frequency=args.det_frequency,
        backend=args.backend,
        device=args.device,
        to_openpose=False,
    )

    smoother = None if args.no_smooth else KeypointSmoother()

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
                                   smoother=smoother)
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
