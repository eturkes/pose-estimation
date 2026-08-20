"""Shared cv2 video-IO helpers for the pipeline entry points.

Single home for the capture/FPS/display/batch helpers that ``main.py``
(MediaPipe path) and ``run.py`` (rtmlib path) both need — previously
duplicated per entry point, which let the two copies drift.
"""

import contextlib
import dataclasses
import os
import pathlib
import time

import cv2
import numpy as np

FALLBACK_FPS = 30.0
MIN_REASONABLE_FPS = 1.0
MAX_REASONABLE_FPS = 240.0
# Ordered tuple for reporting; VIDEO_EXTS is the membership form the batch
# helpers use.  multicam.VIDEO_EXTENSIONS is deliberately narrower: a session
# manifest accepts fewer containers than a corpus census recognises.
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")
VIDEO_EXTS = set(VIDEO_EXTENSIONS)


class SourceTimestampClock:
    """Return a stable source timestamp after each decoded frame.

    File-backed captures prefer the media presentation timestamp reported by
    OpenCV.  Missing, duplicate, or regressing timestamps fall back to the
    source frame index and nominal FPS.  Live captures use elapsed monotonic
    time, anchored to zero on the first decoded frame.

    The returned values are strictly increasing after the first frame.  That
    matters for temporal filters, which otherwise become dependent on inference
    latency or receive a near-zero interval from an unsupported ``POS_MSEC``
    backend that repeatedly reports zero.
    """

    def __init__(self, capture, fps, *, live, monotonic=None):
        self._capture = capture
        self._fps = safe_fps(fps)
        self._live = live
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._live_origin = None
        self._last = None

    def timestamp(self, source_frame_idx):
        """Return seconds for a zero-based decoded-frame index."""
        fallback = float(source_frame_idx) / self._fps
        candidate = None

        if self._live:
            now = float(self._monotonic())
            if np.isfinite(now):
                if self._live_origin is None:
                    self._live_origin = now
                    candidate = 0.0
                else:
                    candidate = now - self._live_origin
        else:
            pos_msec = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            if np.isfinite(pos_msec) and pos_msec >= 0.0:
                candidate = pos_msec / 1000.0

        # A repeated timestamp is unusable for temporal filtering even though
        # it is technically non-regressing, so use the CFR fallback as well.
        if candidate is None or (self._last is not None and candidate <= self._last):
            candidate = fallback

        if self._last is not None and candidate <= self._last:
            candidate = self._last + 1.0 / self._fps

        self._last = candidate
        return candidate


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
            print(f"WARNING: The file does not exist: {label}.")
            return None
        if not path.is_file():
            print(f"WARNING: The path is not a regular file: {label}.")
            return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            print(f"WARNING: OpenCV cannot open camera index {source} (no device or driver?).")
        else:
            print(f"WARNING: OpenCV cannot open {label} (codec issue or file integrity?).")
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
        print(
            f"WARNING: OpenCV reported unusual FPS ({raw_fps:.2f}). "
            f"The pipeline uses {FALLBACK_FPS}."
        )
        return FALLBACK_FPS
    return float(raw_fps)


@dataclasses.dataclass(frozen=True)
class ContainerFacts:
    """Raw cv2 header readings for one media file.

    Nothing here is repaired, clamped, or substituted: an unreadable FPS stays
    0.0 rather than becoming ``FALLBACK_FPS``, and a missing frame count stays
    0 rather than triggering a decode.  Callers that want usable values call
    ``safe_fps``/``frame_count`` instead; callers that want to *describe* a
    file, such as the corpus census, need the container's own answers.

    Every field name says ``reported`` or ``auto`` because OpenCV's FFmpeg
    backend averages the frame rate and may estimate the frame count from
    ``duration * fps``.  These are claims by the demuxer, not measurements.
    """

    probe_status: str
    backend_name: str
    reported_width: int
    reported_height: int
    reported_avg_fps: float
    reported_frame_count: int
    reported_rotation_deg: int
    reported_fourcc: str
    orientation_auto: bool


PROBE_OPENED = "opened"
PROBE_OPEN_FAILED = "open_failed"
PROBE_SKIPPED = "skipped"

_UNPROBED = ContainerFacts(
    probe_status=PROBE_OPEN_FAILED,
    backend_name="",
    reported_width=0,
    reported_height=0,
    reported_avg_fps=0.0,
    reported_frame_count=0,
    reported_rotation_deg=0,
    reported_fourcc="",
    orientation_auto=False,
)


def _finite_int(raw):
    """Return *raw* as an int, or 0 when the backend reports NaN/inf."""
    value = float(raw)
    return int(value) if np.isfinite(value) else 0


def _fourcc_text(raw):
    """Return the codec tag verbatim, or "" when any of its bytes is unprintable.

    The four bytes are never trimmed.  ``DIB `` is a real tag whose fourth
    character is a space, so trimming publishes a repaired value under a
    contract that promises the backend's raw reading, and it contradicts the
    fixed four-character width the schema states.
    """
    code = _finite_int(raw)
    if code <= 0:
        return ""
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() else ""


def _utf8_path_text(path):
    """Return *path* as a UTF-8-clean string, or None when its bytes are not.

    Under a C locale Python decodes filenames with ``surrogateescape``, and a
    lone surrogate reaching OpenCV's C++ boundary terminates the process
    instead of raising.  Recovering the original bytes and decoding them
    strictly yields either a string OpenCV can take or an honest refusal.
    """
    try:
        return os.fsencode(path).decode("utf-8")
    except UnicodeDecodeError:
        return None


def probe_container(path):
    """Read one file's container header without decoding a frame.

    Auto-rotation is requested explicitly and then read back, because OpenCV
    changed its default across 4.10/4.11/4.12 and the setting decides whether
    the reported width and height are the display or the coded size.

    This never raises, and that covers the whole capture lifecycle: the
    constructor, ``isOpened``, every property read and ``release``.  A census
    has to give every file a disposition, so a backend that throws on one file
    answers ``open_failed`` for it rather than ending the run.  A throw during
    teardown alone keeps the facts already read, because the header answered
    and reporting ``open_failed`` for it would be a false statement.
    """
    text = _utf8_path_text(path)
    if text is None:
        return _UNPROBED
    try:
        cap = cv2.VideoCapture(text)
    except Exception:
        return _UNPROBED
    try:
        if not cap.isOpened():
            return _UNPROBED
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        return ContainerFacts(
            probe_status=PROBE_OPENED,
            backend_name=cap.getBackendName(),
            reported_width=_finite_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            reported_height=_finite_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            reported_avg_fps=float(cap.get(cv2.CAP_PROP_FPS)),
            reported_frame_count=_finite_int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            reported_rotation_deg=_finite_int(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
            reported_fourcc=_fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)),
            orientation_auto=bool(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)),
        )
    except Exception:
        return _UNPROBED
    finally:
        # A backend that throws on teardown must not end the run either, so
        # the handle is abandoned rather than the exception propagated.  The
        # header was already read, so the facts stand on their own.
        with contextlib.suppress(Exception):
            cap.release()


def frame_to_surface(frame):
    """Convert a BGR OpenCV frame to a pygame Surface."""
    import pygame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))


def collect_video_files(directory):
    """Return the sorted video-file Paths in *directory*; raise if none."""
    d = pathlib.Path(directory)
    if not d.is_dir():
        raise RuntimeError(f"The path is not a directory: {directory}.")
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise RuntimeError(f"The directory contains no video files: {directory}.")
    return files
