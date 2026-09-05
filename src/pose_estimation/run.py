"""Pose estimation — unified entry point.

Supports rtmlib-based models (RTMW, DWPose, RTMPose) and MediaPipe.

Usage:
    python -m pose_estimation.run                                     # webcam 0, default model
    python -m pose_estimation.run --model dwpose-m                    # DWPose wholebody
    python -m pose_estimation.run --model rtmpose-m                   # body-only (17 kps)
    python -m pose_estimation.run --model mediapipe                   # MediaPipe pose + hand
    python -m pose_estimation.run --source video.mp4 --headless
    python -m pose_estimation.run --batch-dir videos/
    python -m pose_estimation.run --batch-dir videos/ --single-subject --tracking hands-arms
    python -m pose_estimation.run --pose-device CPU --det-device CPU   # no accelerator

Requirements:
    pip install rtmlib openvino  # or: pip install rtmlib onnxruntime
"""

import argparse
import collections
import csv
import os
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np

from .calibration import CalibrationError
from .constraints import BoneLengthSmoother
from .export import frame_to_rows, open_csv_writer
from .mapping import coco_hand_confidences, coco_hand_handedness, coco_to_mediapipe
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
from .video_io import (
    SourceTimestampClock,
    collect_video_files,
    frame_to_surface,
    open_capture,
    safe_fps,
)

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


# ---------------------------------------------------------------------------
# Detector / pose model device placement
# ---------------------------------------------------------------------------
class SplitDeviceSolution:
    """rtmlib solution that compiles the detector and pose model on separate devices.

    rtmlib's ``Custom`` takes a single ``device`` for both models, but the two do
    not share device compatibility.  YOLOX exports its NMS into the graph, so its
    ``dets`` output shape is dynamic (one row per surviving box).  A device that
    demands static shapes — the NPU — instead returns a fixed-size buffer whose
    unused rows are never written, so their scores read as uninitialised memory:
    every frame saturates at the padded row count, with scores outside [0, 1].
    The pose models are static-shaped and agree with CPU to well under a pixel.

    ``PoseTracker`` consumes ``det_model``, ``pose_model`` and ``det_categories``;
    it also reads ``det_model.mode``, which ``YOLOX`` sets from its own default.
    """

    def __init__(
        self,
        *,
        det,
        det_input_size,
        det_device,
        pose_class,
        pose,
        pose_input_size,
        pose_device,
        backend,
        mode=None,
        to_openpose=False,
        device=None,
    ):
        # ``mode`` and ``device`` arrive from PoseTracker and are deliberately
        # unused: explicit registry URLs make ``mode`` moot, and device placement
        # is already resolved per model by the two device arguments.
        del mode, device
        import rtmlib

        self.det_model = rtmlib.YOLOX(
            det, model_input_size=det_input_size, backend=backend, device=det_device
        )
        self.pose_model = getattr(rtmlib, pose_class)(
            pose,
            model_input_size=pose_input_size,
            to_openpose=to_openpose,
            backend=backend,
            device=pose_device,
        )
        self.det_categories = None
        self.one_stage = False


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


def _reset_if_supported(component):
    """Reset source-local state without depending on a concrete backend type."""
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run pose estimation.")
    p.add_argument(
        "--source",
        default="0",
        help="Select a camera index or video-file path (default: 0).",
    )
    p.add_argument(
        "--batch-dir",
        default=None,
        help="Process all video files in this directory (overrides --source).",
    )
    p.add_argument(
        "--session-dir",
        default=None,
        help=(
            "Select a multi-camera session directory with cam*.{mp4,avi,mov,mkv,webm}. "
            "The directory can include session.json and calibration.json. "
            "The pipeline processes each view and fuses to 3D when calibration is present."
        ),
    )
    p.add_argument(
        "--sessions-dir",
        default=None,
        help="Select a parent directory that contains multiple session subdirectories (batch mode).",
    )
    p.add_argument(
        "--list-sessions",
        action="store_true",
        help=(
            "Run a read-only session probe. "
            "The probe discovers session(s) from filenames, session.json, and calibration.json "
            "without frame decoding. It prints the camera count and calibration presence for "
            "each session, then exits. If you omit both --session-dir and --sessions-dir, "
            "the probe uses the sessions/ root that pose-estimation-sessions publishes."
        ),
    )
    p.add_argument(
        "--calibration",
        default=None,
        help=(
            "Set an override path to calibration.json for the selected session(s). "
            "The default is <session_dir>/calibration.json when that file is present."
        ),
    )
    p.add_argument(
        "--single-subject",
        action="store_true",
        help="Track only the highest-confidence person.",
    )
    p.add_argument(
        "--backend",
        default="openvino",
        choices=["onnxruntime", "openvino", "opencv"],
        help="Select the inference backend (default: openvino).",
    )
    p.add_argument(
        "--det-device",
        default="CPU",
        help=(
            "Select the person-detector device: NPU, CPU, or GPU (default: CPU). "
            "YOLOX exports in-graph NMS with a dynamic output shape. "
            "The NPU cannot honor this shape; see SplitDeviceSolution."
        ),
    )
    p.add_argument(
        "--pose-device",
        default="NPU",
        help="Select the pose-model device: NPU, CPU, or GPU (default: NPU).",
    )
    p.add_argument(
        "--mode",
        default="balanced",
        choices=["performance", "balanced", "lightweight"],
        help="Select the model quality/speed tier (default: balanced).",
    )
    model_names = [*list(MODEL_REGISTRY.keys()), "mediapipe"]
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=model_names,
        help=(
            f"Select the pose model (default: {DEFAULT_MODEL}). Available models: "
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
        help=(
            "Select the keypoint scope (default: hands-arms). "
            "'hands' and 'hands-arms' require a Wholebody model."
        ),
    )
    p.add_argument(
        "--det-frequency",
        type=int,
        default=7,
        help="Run the detector every N frames (default: 7).",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Skip the display. Print only latency statistics.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Write per-source CSVs to this directory.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = unlimited).",
    )
    p.add_argument("--no-smooth", action="store_true", help="Disable temporal smoothing.")
    p.add_argument("--no-constraints", action="store_true", help="Disable bone-length constraints.")
    return p.parse_args(argv)


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
    output_diag=None,
    video_name=None,
):
    """Process a single video/camera source.  Returns latency list (ms).

    When *output_csv* is a path, per-frame keypoints are mapped to the
    MediaPipe CSV schema and written to that file.  *video_name* is the
    label written into the CSV ``video`` column (defaults to filename).

    When *output_diag* is a path, a one-row source summary is written there,
    carrying the timestamp dispositions ``SourceTimestampClock`` recorded.  A
    corpus run needs the CFR fallback rate measured per asset: the container's
    PTS monotonicity flag counts demux order while this path reads
    presentation order, so it bounds exposure without measuring it.
    """
    source = int(source_str) if source_str.isdigit() else source_str
    cap = open_capture(source, display=source_str)
    if cap is None:
        return []

    _reset_if_supported(pose_tracker)
    _reset_if_supported(smoother)
    _reset_if_supported(bone_smoother)

    fps_video = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    source_clock = SourceTimestampClock(
        cap,
        fps_video,
        live=isinstance(source, int),
    )
    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # Banner only.  Every exported row normalises against the decoded frame's
    # own shape instead, so a backend whose header disagrees with its pixels —
    # or an asset that changes orientation mid-clip — cannot desynchronise the
    # coordinate scale from the image it describes.
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
    source_frame_idx = 0
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
            decoded_frame_idx = source_frame_idx
            source_frame_idx += 1
            timestamp = source_clock.timestamp(decoded_frame_idx)
            if frame is None or frame.size == 0:
                print(f"WARNING: The pipeline skips malformed source frame {decoded_frame_idx}.")
                continue
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            t0 = time.perf_counter()
            keypoints, scores = pose_tracker(frame)
            dt = time.perf_counter() - t0
            latencies.append(dt * 1000)

            if smoother is not None:
                keypoints, scores = smoother(keypoints, scores, timestamp)

            if bone_smoother is not None:
                if smoother is not None:
                    if keypoints is not None and keypoints.ndim == 3 and keypoints.shape[0] > 0:
                        track_keys = smoother.output_track_keys()
                        if len(track_keys) != keypoints.shape[0]:
                            raise RuntimeError(
                                "smoother output track keys are not aligned with keypoints"
                            )
                        if scores is None or scores.shape != keypoints.shape[:2]:
                            raise RuntimeError("scores are not aligned with constrained keypoints")
                        for track_key, person_keypoints, person_scores in zip(
                            track_keys, keypoints, scores, strict=True
                        ):
                            validity = (
                                np.isfinite(person_keypoints).all(axis=1)
                                & np.isfinite(person_scores)
                                & (person_scores > 0.0)
                            )
                            bone_smoother.update(track_key, person_keypoints, validity=validity)
                    bone_smoother.prune(smoother.live_track_keys())
                else:
                    # Without temporal association, detector row order is not
                    # an identity, even when only one row happens to be present.
                    # Cross-applying learned proportions is worse than skipping
                    # the temporal constraint.
                    bone_smoother.prune([])

            if args.single_subject:
                keypoints, scores = filter_single_subject(keypoints, scores)

            n_persons = (
                keypoints.shape[0] if keypoints is not None and len(keypoints.shape) == 3 else 0
            )
            n_kps = keypoints.shape[1] if n_persons > 0 else 0

            # CSV export
            if csv_writer is not None and n_persons > 0:
                body_lm, body_vis, hand_lm, matches = coco_to_mediapipe(
                    keypoints, scores, n_kps, args.tracking
                )
                hand_handedness = (
                    coco_hand_handedness(scores, n_kps) if args.tracking == "hands" else None
                )
                hand_confidences = coco_hand_confidences(keypoints, scores, n_kps)
                rows = frame_to_rows(
                    video_name=csv_video_name,
                    frame_idx=decoded_frame_idx,
                    timestamp_sec=timestamp,
                    frame_h=frame.shape[0],
                    frame_w=frame.shape[1],
                    body_landmarks=body_lm,
                    body_visibilities=body_vis,
                    hand_landmarks=hand_lm,
                    matches=matches,
                    tracking=args.tracking,
                    hand_handedness=hand_handedness,
                    hand_confidences=hand_confidences,
                )
                for row in rows:
                    csv_writer.writerow(row)

            if frame_idx <= 5 or frame_idx % 50 == 0:
                mean_lat = np.mean(latencies[-50:])
                print(
                    f"Frame {frame_idx:5d} | "
                    f"{dt * 1000:6.1f} ms | "
                    f"average {mean_lat:6.1f} ms | "
                    f"people {n_persons} | keypoints {n_kps}"
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
        print("\nThe run stopped after an interruption.")
    finally:
        cap.release()
        if csv_fh is not None:
            csv_fh.close()
            print(f"  Wrote CSV: {output_csv}")
        if output_diag is not None:
            write_source_diagnostics(
                output_diag,
                video=csv_video_name,
                clock=source_clock,
                fps_nominal=fps_video,
                latencies=latencies,
            )

    return latencies


SOURCE_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "video",
    "n_frames_decoded",
    "pts_accepted",
    "index_fallback",
    "monotonic_forced",
    "cfr_fallback_rate",
    "fps_nominal",
    "latency_ms_mean",
    "latency_ms_p95",
)


def write_source_diagnostics(path, *, video, clock, fps_nominal, latencies):
    """Write the one-row per-source diagnostics summary.

    Written from the ``finally`` arm, so an interrupted run still reports what
    it decoded — the counts describe the frames processed, never the frames
    the asset holds. That arm is also why the parent is created here: a missing
    destination directory would raise over whatever exception is unwinding.
    """
    row = {
        "video": video,
        "n_frames_decoded": clock.n_timestamps,
        "pts_accepted": clock.pts_accepted,
        "index_fallback": clock.index_fallback,
        "monotonic_forced": clock.monotonic_forced,
        "cfr_fallback_rate": f"{clock.cfr_fallback_rate:.6f}",
        "fps_nominal": f"{fps_nominal:.6f}",
        "latency_ms_mean": f"{float(np.mean(latencies)):.3f}" if latencies else "",
        "latency_ms_p95": f"{float(np.percentile(latencies, 95)):.3f}" if latencies else "",
    }
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:  # noqa: PTH123
        writer = csv.DictWriter(handle, fieldnames=SOURCE_DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(
        f"  CFR fallback rate: {row['cfr_fallback_rate']} "
        f"(index {clock.index_fallback}, forced {clock.monotonic_forced} "
        f"of {clock.n_timestamps} timestamps)"
    )
    print(f"  Wrote diagnostics: {path}")


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
    print(f"  Warmup (first 3):  {np.mean(arr[:3]):.1f} ms average")
    if len(warm) > 0:
        print(
            f"  Steady-state:      {np.mean(warm):.1f} ms average, "
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
    cmd += ["--det-device", args.det_device]
    cmd += ["--pose-device", args.pose_device]
    cmd += ["--tracking", args.tracking]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    if args.single_subject:
        cmd.append("--single-subject")
    if args.headless:
        cmd.append("--headless")
    for flag, val in (("--no-smooth", args.no_smooth), ("--max-frames", args.max_frames)):
        if val:
            print(
                f"WARNING: The MediaPipe pipeline does not support {flag}. "
                "The runner ignores this flag."
            )
    print(f"The runner delegates to the MediaPipe pipeline: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _dispatch_sessions(args, *, pose_tracker, draw_skeleton, smoother, bone_smoother, screen):
    """Resolve --session-dir / --sessions-dir and run per-camera processing.

    Constructs an rtmlib camera processor closure that wraps
    ``process_source`` with smoother reset, then hands off to
    ``process_session`` for per-camera orchestration.
    """
    sessions = resolve_cli_sessions(args.session_dir, args.sessions_dir, args.calibration)

    def _camera_processor(*, source, output_csv, output_diag, video_name, **_kw):
        latencies = process_source(
            args,
            pose_tracker,
            source,
            draw_skeleton,
            smoother=smoother,
            bone_smoother=bone_smoother,
            screen=screen,
            output_csv=str(output_csv),
            output_diag=str(output_diag),
            video_name=video_name,
        )
        print_latency_summary(latencies)
        return latencies

    for s in sessions:
        process_session(
            s,
            camera_processor=_camera_processor,
            output_dir=args.output_dir,
        )


def main(argv=None):
    # argv rides through, matching the other console scripts, so --list-sessions
    # is testable in-process instead of only through a subprocess.
    args = parse_args(argv)

    if args.list_sessions:
        # The probe reports an ordinal, a camera count and calibration presence
        # alone: the tree is patient-adjacent, so session ids, camera names and
        # every frame stay out of an agent's context.  Its default root is the
        # published tree, never the raw media root, which is non-recursive and
        # so could only ever discover nothing.
        session_dir = args.session_dir
        sessions_dir = args.sessions_dir
        if session_dir is None and sessions_dir is None:
            sessions_dir = "sessions"
        try:
            resolve_cli_sessions(
                session_dir,
                sessions_dir,
                args.calibration,
                summary_label="Discovered sessions",
                redact_identifiers=True,
            )
        except (SessionError, CalibrationError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # ── MediaPipe delegates to main.py (forwards session flags too) ─
    if args.model == "mediapipe":
        sys.exit(_run_mediapipe(args))

    if args.calibration is not None and not (args.session_dir or args.sessions_dir):
        print(
            "WARNING: --calibration has no effect without --session-dir/--sessions-dir. "
            "The runner ignores it."
        )

    # ── Resolve model — legacy --body-only maps to rtmpose-m ────────
    model_name = args.model
    if args.body_only and model_name == DEFAULT_MODEL:
        model_name = "rtmpose-m"

    model = MODEL_REGISTRY[model_name]

    # --tracking hands/hands-arms needs wholebody (133 kps)
    if args.tracking != "body" and model["n_kps"] == 17:
        print(
            f"NOTE: --tracking {args.tracking} requires a Wholebody model. "
            f"The runner switches from {model_name} to {DEFAULT_MODEL}."
        )
        model_name = DEFAULT_MODEL
        model = MODEL_REGISTRY[model_name]

    # ── Patch rtmlib before importing its classes ────────────────────
    if args.backend == "openvino":
        _patch_rtmlib_openvino()

    # ── Import rtmlib (deferred so --help works without it) ─────────
    from functools import partial

    from rtmlib import PoseTracker, draw_skeleton

    # ── Set up model (explicit URLs from registry for all devices) ──
    solution_cls = partial(
        SplitDeviceSolution,
        det=_DET_URL,
        det_input_size=_DET_INPUT_SIZE,
        det_device=args.det_device,
        pose_class=model["pose_class"],
        pose=model["pose"],
        pose_input_size=model["pose_input_size"],
        pose_device=args.pose_device,
    )
    print(f"Model:   {model['label']} [{model_name}]")

    tracking_label = f", tracking={args.tracking}"
    single_label = ", single-subject" if args.single_subject else ""
    smooth_label = ", no-smooth" if args.no_smooth else ", smooth"
    constraint_label = ", no-constraints" if args.no_constraints else ""
    print(
        f"Backend: {args.backend}, det-device={args.det_device}, "
        f"pose-device={args.pose_device}"
        f"{tracking_label}{single_label}{smooth_label}"
        f"{constraint_label}"
    )

    pose_tracker = PoseTracker(
        solution_cls,  # ty: ignore[invalid-argument-type]  # rtmlib accepts any callable
        mode=args.mode,
        det_frequency=args.det_frequency,
        backend=args.backend,
        to_openpose=False,
        # rtmlib's IoU tracking indexes the CURRENT frame's keypoint array by a
        # PERSISTENT track id, so one missed match raises IndexError on a path
        # that returns before ``frame_cnt += 1`` and before ``bboxes_last_frame``
        # is replaced -- freezing both for the rest of the source.  The residue
        # of the frozen counter then picks the failure: ``% det_frequency == 0``
        # re-runs the detector on every frame, anything else starves the box list
        # and RTMPose silently falls back to the WHOLE FRAME as its crop.
        # KeypointSmoother already owns temporal association (Hungarian
        # ``gated_assignment``), so rtmlib's tracker is redundant as well as
        # unsound.  ``tracking=False`` takes the stateless branch, which needs
        # ``det_frequency != 1`` to engage.
        tracking=False,
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

            # Derive per-source CSV path: file sources use the file stem; live
            # camera sources (a numeric device index) use "camera<idx>" so the
            # pose CSV is still exported and stays unique across cameras.
            csv_path = None
            if out_dir is not None:
                stem = f"camera{src}" if src.isdigit() else pathlib.Path(src).stem
                csv_path = str(out_dir / (stem + ".csv"))

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
