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
import os
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np

from .constraints import BoneLengthSmoother
from .export import frame_to_rows, open_csv_writer
from .mapping import coco_to_mediapipe
from .multicam import (
    SessionError,
    process_session,
    resolve_cli_sessions,
)
from .rtmlib_openvino import _patch_rtmlib_openvino
from .rtmlib_smoothing import (
    _KP_ARMS,
    _KP_LHAND,
    _KP_RHAND,
    REGION_PARAMS,  # noqa: F401  # re-exported for tests
    KeypointSmoother,
    OneEuroFilter,  # noqa: F401  # re-exported for tests (shared smoothing.OneEuroFilter)
)
from .video_io import collect_video_files, frame_to_surface, open_capture, safe_fps

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
TRACKING_INDICES = {
    "hands": _KP_LHAND | _KP_RHAND,
    "hands-arms": _KP_ARMS | _KP_LHAND | _KP_RHAND,
    "body": set(range(133)),
}

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

WINDOW_TITLE = "Pose Estimation"


def _parse_rest_cutoff(env_var, default):
    """Parse an env var as optional float (returns None for 'none' or empty)."""
    val = os.environ.get(env_var, "")
    if val == "":
        return default
    if val.lower() == "none":
        return None
    return float(val)


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
        help="Process all video files in a directory (overrides --source)",
    )
    p.add_argument(
        "--session-dir",
        default=None,
        help=(
            "Multi-camera session directory (cam*.{mp4,avi,mov,mkv,webm} + "
            "optional session.json + optional calibration.json). "
            "Per-view processing + 3D fusion are not yet wired."
        ),
    )
    p.add_argument(
        "--sessions-dir",
        default=None,
        help="Parent directory holding multiple session subdirectories (batch mode).",
    )
    p.add_argument(
        "--calibration",
        default=None,
        help=(
            "Override path to calibration.json for the selected session(s); "
            "defaults to <session_dir>/calibration.json when present."
        ),
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
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV output. Per-source CSVs are written here.",
    )
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    p.add_argument("--no-smooth", action="store_true", help="Disable temporal smoothing")
    p.add_argument("--no-constraints", action="store_true", help="Disable bone-length constraints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------


def process_source(
    args,
    pose_tracker,
    source_str,
    draw_skeleton,
    smoother=None,
    bone_smoother=None,
    screen=None,
    output_csv=None,
    video_name=None,
):
    """Process a single video/camera source.  Returns latency list (ms).

    When *output_csv* is a path, per-frame keypoints are mapped to the
    MediaPipe CSV schema and written to that file.  *video_name* is the
    label written into the CSV ``video`` column (defaults to filename).
    """
    source = int(source_str) if source_str.isdigit() else source_str
    cap = open_capture(source, display=source_str)
    if cap is None:
        return []

    fps_video = safe_fps(cap.get(cv2.CAP_PROP_FPS))
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

    # CSV export setup
    csv_fh = None
    csv_writer = None
    csv_video_name = video_name or (
        pathlib.Path(source_str).name if not source_str.isdigit() else "webcam"
    )
    if output_csv is not None:
        csv_fh, csv_writer = open_csv_writer(output_csv, tracking=args.tracking)

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

            # CSV export
            if csv_writer is not None and n_persons > 0:
                timestamp = (frame_idx - 1) / fps_video if fps_video > 0 else 0.0
                body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(
                    keypoints, scores, n_kps, args.tracking
                )
                rows = frame_to_rows(
                    video_name=csv_video_name,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp,
                    frame_h=h,
                    frame_w=w,
                    body_landmarks=body_lm,
                    body_visibilities=body_vis,
                    hand_landmarks=hand_lm,
                    matches=matches,
                    tracking=args.tracking,
                )
                for row in rows:
                    csv_writer.writerow(row)

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
                        _caption = (
                            pathlib.Path(source_str).name if not source_str.isdigit() else None
                        )
                        if _caption:
                            pygame.display.set_caption(f"{WINDOW_TITLE} — {_caption}")
                    screen.blit(frame_to_surface(img_show), (0, 0))
                    pygame.display.flip()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if csv_fh is not None:
            csv_fh.close()
            print(f"  CSV written: {output_csv}")

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
    if args.session_dir:
        cmd += ["--session-dir", args.session_dir]
    elif args.sessions_dir:
        cmd += ["--sessions-dir", args.sessions_dir]
    elif args.batch_dir:
        cmd += ["--batch-dir", args.batch_dir]
    elif args.source != "0":
        cmd += ["--source", args.source]
    if args.calibration:
        cmd += ["--calibration", args.calibration]
    cmd += ["--device", args.device]
    cmd += ["--tracking", args.tracking]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    if args.single_subject:
        cmd.append("--single-subject")
    if args.headless:
        cmd.append("--headless")
    for flag, val in (("--no-smooth", args.no_smooth), ("--max-frames", args.max_frames)):
        if val:
            print(f"WARNING: {flag} is not supported by the MediaPipe pipeline; ignoring.")
    print(f"Delegating to MediaPipe pipeline: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _dispatch_sessions(args, *, pose_tracker, draw_skeleton, smoother, bone_smoother, screen):
    """Resolve --session-dir / --sessions-dir and run per-camera processing.

    Constructs an rtmlib camera processor closure that wraps
    ``process_source`` with smoother reset, then hands off to
    ``process_session`` for per-camera orchestration.
    """
    sessions = resolve_cli_sessions(args.session_dir, args.sessions_dir, args.calibration)

    def _camera_processor(*, source, output_csv, output_diag, video_name, **_kw):
        if smoother is not None:
            smoother.reset()
        latencies = process_source(
            args,
            pose_tracker,
            source,
            draw_skeleton,
            smoother=smoother,
            bone_smoother=bone_smoother,
            screen=screen,
            output_csv=str(output_csv),
            video_name=video_name,
        )
        print_latency_summary(latencies)
        return latencies

    for s in sessions:
        process_session(
            s,
            camera_processor=_camera_processor,
        )


def main():
    args = parse_args()

    # ── MediaPipe delegates to main.py (forwards session flags too) ─
    if args.model == "mediapipe":
        sys.exit(_run_mediapipe(args))

    if args.calibration is not None and not (args.session_dir or args.sessions_dir):
        print(
            "WARNING: --calibration has no effect without --session-dir/--sessions-dir; ignoring."
        )

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

    smoother = (
        None
        if args.no_smooth
        else KeypointSmoother(
            rest_cutoff=_parse_rest_cutoff("POSE_BENCH_BODY_REST_CUTOFF", 0.05),
            hand_rest_cutoff=_parse_rest_cutoff("POSE_BENCH_HAND_REST_CUTOFF", 0.15),
            rest_speed=float(os.environ.get("POSE_BENCH_REST_SPEED", "2.0")),
            fast_speed=float(os.environ.get("POSE_BENCH_FAST_SPEED", "10.0")),
        )
    )

    bone_smoother = None
    if not args.no_constraints:
        segments = BONE_SEGMENTS_WB_BODY if args.tracking == "body" else BONE_SEGMENTS_WB
        bone_smoother = BoneLengthSmoother(segments=segments)

    # ── Multi-camera session dispatch ─────────────────────────────
    if args.session_dir or args.sessions_dir:
        screen = None
        if not args.headless:
            import pygame as _pg

            _pg.init()
            screen = _pg.display.set_mode((640, 480))
            _pg.display.set_caption(WINDOW_TITLE)
        try:
            _dispatch_sessions(
                args,
                pose_tracker=pose_tracker,
                draw_skeleton=draw_skeleton,
                smoother=smoother,
                bone_smoother=bone_smoother,
                screen=screen,
            )
        except SessionError as exc:
            print(f"ERROR: {exc}")
            sys.exit(2)
        finally:
            if not args.headless:
                import pygame as _pg

                _pg.quit()
        return

    # ── Collect sources ─────────────────────────────────────────────
    if args.batch_dir:
        sources = [str(p) for p in collect_video_files(args.batch_dir)]
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

    # ── Resolve output directory ───────────────────────────────────
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each source ─────────────────────────────────────────
    all_latencies = []
    try:
        for i, src in enumerate(sources):
            if len(sources) > 1:
                print(f"\n{'=' * 60}")
                print(f"[{i + 1}/{len(sources)}] {src}")
                print("=" * 60)

            # Derive per-source CSV path
            csv_path = None
            if out_dir is not None and not src.isdigit():
                csv_path = str(out_dir / (pathlib.Path(src).stem + ".csv"))

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
                output_csv=csv_path,
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
