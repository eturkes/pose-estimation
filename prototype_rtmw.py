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

Requirements:
    pip install rtmlib openvino  # or: pip install rtmlib onnxruntime
"""

import argparse
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RTMW prototype")
    p.add_argument("--source", default="0",
                   help="Camera index or video path (default: 0)")
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
    p.add_argument("--det-frequency", type=int, default=7,
                   help="Run detector every N frames (default: 7)")
    p.add_argument("--headless", action="store_true",
                   help="No display — just print latency stats")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited)")
    return p.parse_args()


def main():
    args = parse_args()

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

    print(f"Backend: {args.backend}, device={args.device}")

    pose_tracker = PoseTracker(
        solution_cls,
        mode=args.mode,
        det_frequency=args.det_frequency,
        backend=args.backend,
        device=args.device,
        to_openpose=False,
    )

    # ── Open video source ───────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source:  {args.source} ({w}x{h} @ {fps_video:.1f} fps"
          f"{f', {total_frames} frames' if total_frames > 0 else ''})")
    print()

    # ── Processing loop ─────────────────────────────────────────────
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

            n_persons = keypoints.shape[0] if keypoints is not None and len(keypoints.shape) == 3 else 0
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
                    img_show = draw_skeleton(
                        img_show, keypoints, scores,
                        openpose_skeleton=False, kpt_thr=0.3)

                # Overlay FPS
                fps_text = f"{1000 / latencies[-1]:.0f} fps" if latencies[-1] > 0 else ""
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

    # ── Summary ─────────────────────────────────────────────────────
    if latencies:
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


if __name__ == "__main__":
    main()
