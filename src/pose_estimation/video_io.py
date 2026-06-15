"""Shared cv2 video-IO helpers for the pipeline entry points.

Single home for the capture/FPS/display/batch helpers that ``main.py``
(MediaPipe path) and ``run.py`` (rtmlib path) both need — previously
duplicated per entry point, which let the two copies drift.
"""

import pathlib

import cv2
import numpy as np

FALLBACK_FPS = 30.0
MIN_REASONABLE_FPS = 1.0
MAX_REASONABLE_FPS = 240.0
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def open_capture(source, display=None):
    """Open a VideoCapture with diagnostic error messages.

    *source* may be an int (camera index) or path string.  *display*
    overrides the name used in messages.  Returns the open capture or
    None after printing a context-aware reason.
    """
    label = source if display is None else display
    if isinstance(source, str):
        path = pathlib.Path(source)
        if not path.exists():
            print(f"WARNING: file not found: {label}.")
            return None
        if not path.is_file():
            print(f"WARNING: not a regular file: {label}.")
            return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            print(f"WARNING: cannot open camera index {source} (no device or driver?).")
        else:
            print(f"WARNING: cannot open {label} (codec issue or file integrity?).")
        return None
    return cap


def frame_count(source):
    """Return the number of frames in *source*, or 0 if it cannot be opened.

    Trusts the container's ``CAP_PROP_FRAME_COUNT`` metadata when it is
    positive (exact for the MJPG AVIs the calibration ``capture`` writes
    and the mp4/cv2-written clips this pipeline reads); falls back to a
    decode-free ``grab`` count when the metadata is missing or unreliable.
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return 0
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            return n
        n = 0
        while cap.grab():
            n += 1
        return n
    finally:
        cap.release()


def safe_fps(raw_fps):
    """Clamp/validate an FPS reading from cv2; fall back to FALLBACK_FPS."""
    if not np.isfinite(raw_fps) or raw_fps <= 0:
        return FALLBACK_FPS
    if raw_fps < MIN_REASONABLE_FPS or raw_fps > MAX_REASONABLE_FPS:
        print(f"WARNING: unusual FPS reported ({raw_fps:.2f}); using {FALLBACK_FPS}.")
        return FALLBACK_FPS
    return float(raw_fps)


def frame_to_surface(frame):
    """Convert a BGR OpenCV frame to a pygame Surface."""
    import pygame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))


def collect_video_files(directory):
    """Return the sorted video-file Paths in *directory*; raise if none."""
    d = pathlib.Path(directory)
    if not d.is_dir():
        raise RuntimeError(f"Not a directory: {directory}")
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise RuntimeError(f"No video files found in: {directory}")
    return files
