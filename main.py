#!/usr/bin/env python3
"""Arm & hand pose estimation using OpenVINO and MediaPipe models.

Usage:
    python main.py                          # webcam 0, NPU device
    python main.py --source 1               # webcam 1
    python main.py --source video.mp4       # single video file
    python main.py --batch-dir videos/      # process all videos in a directory
    python main.py --device CPU             # run on CPU
    python main.py --no-flip                # disable mirror for front camera
"""

import argparse
import collections
import pathlib
import time

import cv2
import numpy as np
import pygame

from models import download_and_compile_models
from detection import generate_anchors, PALM_INPUT_SIZE, POSE_INPUT_SIZE
from processing import process_frame, match_hands_to_arms
from smoothing import PoseSmoother
from drawing import draw_body_landmarks, draw_hand_landmarks, draw_arm_hand_bridges
from export import open_csv_writer, frame_to_rows

WINDOW_TITLE = "Arm & Hand Pose Estimation"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def frame_to_surface(frame):
    """Convert a BGR OpenCV frame to a pygame Surface."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))


def process_video(source, flip, models, palm_anchors, pose_anchors,
                  screen, csv_writer=None, video_name=None):
    """Run pose estimation on a single video source with real-time display.

    Returns True if the user requested quit (ESC / window close),
    False if the video simply ended.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  Cannot open: {source}")
        return False

    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Optionally set resolution for cameras
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother = PoseSmoother()
    processing_times = collections.deque(maxlen=200)
    frame_idx = 0

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    return True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    cap.release()
                    return True

            ret, frame = cap.read()
            if not ret:
                break

            if flip:
                frame = cv2.flip(frame, 1)

            # Cap resolution for performance
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            frame_h, frame_w = frame.shape[:2]

            # Resize pygame window to match first frame
            if frame_idx == 0:
                screen = pygame.display.set_mode((frame_w, frame_h))
                if video_name:
                    pygame.display.set_caption(f"{WINDOW_TITLE} — {video_name}")

            # Inference
            start = time.time()
            body_lm, body_vis, hand_lm = process_frame(
                frame, models, palm_anchors, pose_anchors
            )
            elapsed = time.time() - start

            # Temporal smoothing
            t = time.time()
            body_lm, body_vis = smoother.smooth_bodies(body_lm, body_vis, t)
            hand_lm = smoother.smooth_hands(hand_lm, t)
            matches = match_hands_to_arms(body_lm, hand_lm)

            # Export landmarks
            if csv_writer is not None:
                timestamp_sec = frame_idx / fps_source
                rows = frame_to_rows(
                    video_name or str(source), frame_idx, timestamp_sec,
                    frame_h, frame_w, body_lm, body_vis, hand_lm, matches,
                )
                for row in rows:
                    csv_writer.writerow(row)

            # Draw overlays
            frame = draw_body_landmarks(frame, body_lm, body_vis)
            frame = draw_arm_hand_bridges(frame, body_lm, hand_lm, matches)
            frame = draw_hand_landmarks(frame, hand_lm)

            # FPS / progress overlay
            processing_times.append(elapsed)
            avg_ms = np.mean(processing_times) * 1000
            fps = 1000 / avg_ms
            _, f_width = frame.shape[:2]
            label = f"Inference: {avg_ms:.1f}ms ({fps:.1f} FPS)"
            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                label += f"  |  Frame {frame_idx}/{total_frames} ({pct:.0f}%)"
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000,
                        (0, 0, 255), 1, cv2.LINE_AA)

            screen.blit(frame_to_surface(frame), (0, 0))
            pygame.display.flip()
            frame_idx += 1

    finally:
        cap.release()

    return False


def collect_video_files(directory):
    """Return sorted list of video file paths in a directory."""
    d = pathlib.Path(directory)
    if not d.is_dir():
        raise RuntimeError(f"Not a directory: {directory}")
    files = sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No video files found in: {directory}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Arm & hand pose estimation")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source", default=None,
        help="Video source: camera index (int) or file path (default: 0)")
    source_group.add_argument(
        "--batch-dir", default=None,
        help="Directory of video files to process sequentially")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for CSV output (default: output/)")
    parser.add_argument("--device", default="NPU",
                        help="OpenVINO inference device (default: NPU)")
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal flip (useful for rear cameras)")
    parser.add_argument("--model-dir", default="model",
                        help="Directory for downloaded/converted models")
    args = parser.parse_args()

    # Download, convert, and compile models
    models = download_and_compile_models(args.model_dir, args.device)

    # Pre-generate detection anchors
    palm_anchors = generate_anchors(PALM_INPUT_SIZE, strides=[8, 16, 16, 16])
    pose_anchors = generate_anchors(POSE_INPUT_SIZE, strides=[8, 16, 32, 32, 32])

    pygame.init()
    # Placeholder size; process_video resizes on first frame
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption(WINDOW_TITLE)

    try:
        if args.batch_dir:
            video_files = collect_video_files(args.batch_dir)
            print(f"Found {len(video_files)} video(s) in {args.batch_dir}")

            for i, vpath in enumerate(video_files, 1):
                print(f"\nProcessing {i}/{len(video_files)}: {vpath.name}")
                csv_path = pathlib.Path(args.output_dir) / f"{vpath.stem}.csv"
                fh, writer = open_csv_writer(csv_path)
                try:
                    user_quit = process_video(
                        str(vpath), False, models, palm_anchors, pose_anchors,
                        screen, csv_writer=writer, video_name=vpath.name,
                    )
                finally:
                    fh.close()
                print(f"  Saved: {csv_path}")
                if user_quit:
                    print("User quit — stopping batch.")
                    break

            print("\nBatch complete.")

        else:
            # Single source mode (camera or file)
            source_arg = args.source if args.source is not None else "0"
            try:
                source = int(source_arg)
                flip = not args.no_flip
            except ValueError:
                source = source_arg
                flip = False

            # For single file mode, also export CSV
            csv_writer = None
            fh = None
            if isinstance(source, str):
                vpath = pathlib.Path(source)
                csv_path = pathlib.Path(args.output_dir) / f"{vpath.stem}.csv"
                fh, csv_writer = open_csv_writer(csv_path)

            video_name = pathlib.Path(source).name if isinstance(source, str) else None
            print(f"Source: {source} | Device: {args.device} | Flip: {flip}")
            print("Close the window or press ESC to exit.")

            try:
                process_video(source, flip, models, palm_anchors, pose_anchors,
                              screen, csv_writer=csv_writer,
                              video_name=video_name)
            finally:
                if fh is not None:
                    fh.close()
                    print(f"Saved: {csv_path}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
