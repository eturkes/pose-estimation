#!/usr/bin/env python3
"""Live arm & hand pose estimation using OpenVINO and MediaPipe models.

Usage:
    python main.py                        # webcam 0, NPU device
    python main.py --source 1             # webcam 1
    python main.py --source video.mp4     # video file
    python main.py --device CPU           # run on CPU
    python main.py --no-flip              # disable mirror for front camera
"""

import argparse
import collections
import time

import cv2
import numpy as np

from models import download_and_compile_models
from detection import generate_anchors, PALM_INPUT_SIZE, POSE_INPUT_SIZE
from processing import process_frame, match_hands_to_arms
from smoothing import PoseSmoother
from drawing import draw_body_landmarks, draw_hand_landmarks, draw_arm_hand_bridges


def main():
    parser = argparse.ArgumentParser(description="Live arm & hand pose estimation")
    parser.add_argument("--source", default="0",
                        help="Video source: camera index (int) or file path (default: 0)")
    parser.add_argument("--device", default="NPU",
                        help="OpenVINO inference device (default: NPU)")
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal flip (useful for rear cameras)")
    parser.add_argument("--model-dir", default="model",
                        help="Directory for downloaded/converted models (default: model)")
    args = parser.parse_args()

    # Parse source: integer → camera index, string → file path
    try:
        source = int(args.source)
        flip = not args.no_flip
    except ValueError:
        source = args.source
        flip = False

    # Download, convert, and compile models
    models = download_and_compile_models(args.model_dir, args.device)

    # Pre-generate detection anchors
    palm_anchors = generate_anchors(PALM_INPUT_SIZE, strides=[8, 16, 16, 16])
    pose_anchors = generate_anchors(POSE_INPUT_SIZE, strides=[8, 16, 32, 32, 32])

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother = PoseSmoother()
    processing_times = collections.deque(maxlen=200)
    window = "Arm & Hand Pose Estimation — press ESC to exit"

    print(f"Source: {source} | Device: {args.device} | Flip: {flip}")
    print("Press ESC to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Source ended.")
                break

            if flip:
                frame = cv2.flip(frame, 1)

            # Cap resolution for performance
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Inference
            start = time.time()
            body_lm, body_vis, hand_lm = process_frame(frame, models, palm_anchors, pose_anchors)
            elapsed = time.time() - start

            # Temporal smoothing
            t = time.time()
            body_lm, body_vis = smoother.smooth_bodies(body_lm, body_vis, t)
            hand_lm = smoother.smooth_hands(hand_lm, t)
            matches = match_hands_to_arms(body_lm, hand_lm)

            # Draw overlays
            frame = draw_body_landmarks(frame, body_lm, body_vis)
            frame = draw_arm_hand_bridges(frame, body_lm, hand_lm, matches)
            frame = draw_hand_landmarks(frame, hand_lm)

            # FPS counter
            processing_times.append(elapsed)
            avg_ms = np.mean(processing_times) * 1000
            fps = 1000 / avg_ms
            _, f_width = frame.shape[:2]
            cv2.putText(frame, f"Inference: {avg_ms:.1f}ms ({fps:.1f} FPS)",
                        (20, 40), cv2.FONT_HERSHEY_COMPLEX, f_width / 1000,
                        (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
