"""Camera survey: per-frame camera motion, the task view and its span, one decode pass ahead of pose.

Hand-held clips open and close on footage that is not the task — a sync close-up, the camera being
placed or picked up (floor, ceiling, the operator) — and shake, re-aim or zoom in between.  The
survey measures the camera directly: a similarity step between consecutive frames from background
feature tracks, and, on every fifth frame, an ORB registration to a reference view of the settled
task.  The task span is the run of frames that map into that view, and the steps compose into a
per-frame transform onto the reference frame, so motion features can be read with the camera's
own motion removed.  Thresholds and their measurements → `.agent/archive/contract-m2u102.md`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import cv2
import numpy as np

from .video_io import safe_fps

GMC_WORK = 640
ORB_WORK = 480
SAMPLE_EVERY = 5
PERSON_AREA = 0.01
SETTLED = 0.05
PERSON_SHARE = 0.6
ANCHOR_INLIERS = 15
T_MAX = 0.25
R_MAX = 20.0
S_MAX = 0.35
TASK_GAP_S = 2.5

_EYE = np.eye(3)


@dataclasses.dataclass(frozen=True)
class CameraSurvey:
    """One source's survey.  ``steps[t]`` maps frame t-1 to frame t (``steps[0]`` = identity)."""

    width: int
    height: int
    fps: float
    steps: np.ndarray  # (n, 3, 3) full-resolution pixels
    measured: np.ndarray  # (n,) bool; steps[0] is never measured
    reference: int | None
    span: tuple[int, int]  # [start, end) frame indices
    to_reference: np.ndarray  # (n, 3, 3) frame t -> reference frame, steps alone

    @property
    def n_frames(self) -> int:
        return len(self.steps)

    def speeds(self, start: int, end: int) -> np.ndarray:
        """Frame-centre speed of each measured step in ``[start, end)``, max-dim per second."""
        steps = [t for t in range(max(start, 1), min(end, self.n_frames)) if self.measured[t]]
        if not steps:
            return np.zeros(0)
        centre = np.array([self.width / 2.0, self.height / 2.0, 1.0])
        moved = self.steps[steps] @ centre
        return (
            np.hypot(moved[:, 0] - centre[0], moved[:, 1] - centre[1])
            / max(self.width, self.height)
            * self.fps
        )


def _gray(frame: np.ndarray, work: int) -> np.ndarray:
    h, w = frame.shape[:2]
    k = work / max(w, h)
    small = cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small


def estimate_step(
    previous: np.ndarray, current: np.ndarray, scale: float = 1.0
) -> np.ndarray | None:
    """Similarity mapping ``previous`` to ``current`` (both grayscale at one size), else ``None``.

    ``scale`` = full-resolution pixels per working pixel; the returned 3 x 3 is in full pixels.
    No foreground mask: on frames a seated subject and a therapist fill, a mask leaves too few
    background points, and RANSAC already rejects the minority that moves on its own.
    """
    points = cv2.goodFeaturesToTrack(
        previous, maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=3
    )
    if points is None or len(points) < 20:
        return None
    # The stubs type `nextPts` as an array; None asks OpenCV to allocate it.
    forward, status, _ = cv2.calcOpticalFlowPyrLK(  # ty: ignore[no-matching-overload]
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(  # ty: ignore[no-matching-overload]
        current,
        previous,
        forward,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    error = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
    keep = (status.reshape(-1) == 1) & (status_back.reshape(-1) == 1) & (error < 1.0)
    if int(keep.sum()) < 20:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        points[keep],
        forward[keep],
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=2000,
        confidence=0.995,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 15:
        return None
    step = np.vstack([matrix, [0.0, 0.0, 1.0]])
    step[:2, 2] *= scale
    return step


def _magnitude(transform: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    centre = np.array([width / 2.0, height / 2.0, 1.0])
    moved = transform @ centre
    shift = float(np.hypot(*(moved - centre)[:2])) / max(width, height)
    rotation = abs(float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))))
    zoom = abs(float(np.log(max(np.hypot(transform[0, 0], transform[1, 0]), 1e-12))))
    return shift, rotation, zoom


def in_bounds(transform: np.ndarray, width: int, height: int) -> bool:
    shift, rotation, zoom = _magnitude(transform, width, height)
    return shift < T_MAX and rotation < R_MAX and zoom < S_MAX


def net_motion(
    steps: np.ndarray, measured: np.ndarray, width: int, height: int, fps: float
) -> np.ndarray:
    """Per frame: the next second's composed motion, shift / max-dim + rotation deg / 100 + |log scale|."""
    n = len(steps)
    horizon = max(1, round(fps))
    out = np.full(n, np.inf)
    for t in range(n):
        last = min(n - 1, t + horizon)
        if last == t:
            out[t] = out[t - 1] if t else 0.0
            continue
        if not measured[t + 1 : last + 1].all():
            continue
        composed = _EYE.copy()
        for k in range(t + 1, last + 1):
            composed = steps[k] @ composed
        shift, rotation, zoom = _magnitude(composed, width, height)
        out[t] = shift + rotation / 100.0 + zoom
    return out


def runs(mask) -> list[tuple[int, int]]:
    out, start = [], None
    for i, value in enumerate([*list(mask), False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    return out


def choose_reference(motion: np.ndarray, person: dict[int, bool], fps: float) -> int | None:
    """Sampled frame nearest the middle of the longest settled placement holding a person."""
    best = None
    for a, b in runs(motion < SETTLED):
        sampled = [f for f in person if a <= f < b]
        if b - a < fps or not sampled or np.mean([person[f] for f in sampled]) < PERSON_SHARE:
            continue
        if best is None or b - a > best[1] - best[0]:
            best = (a, b, sampled)
    if best is None:
        return None
    a, b, sampled = best
    middle = (a + b) / 2.0
    return min(sampled, key=lambda f: (abs(f - middle), f))


def register(frame_features, reference_features, scale: float) -> tuple[np.ndarray | None, int]:
    """ORB similarity frame -> reference in full pixels and its inlier count."""
    points, descriptors = frame_features
    ref_points, ref_descriptors = reference_features
    if (
        descriptors is None
        or ref_descriptors is None
        or len(descriptors) < 10
        or len(ref_descriptors) < 2
    ):
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, ref_descriptors, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.8 * rest[0].distance]
    if len(good) < 10:
        return None, 0
    src = np.array([points[m.queryIdx] for m in good], dtype=np.float32)
    dst = np.array([ref_points[m.trainIdx] for m in good], dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0 * scale, maxIters=2000
    )
    if matrix is None or inliers is None:
        return None, 0
    return np.vstack([matrix, [0.0, 0.0, 1.0]]), int(inliers.sum())


def in_view(
    steps: np.ndarray, anchors: dict[int, np.ndarray], width: int, height: int
) -> np.ndarray:
    """Frames whose frame -> reference transform, carried through the steps from the nearest anchor
    in either direction (an anchor replaces it), lies inside the bounds."""
    n = len(steps)
    view = np.zeros(n, bool)
    carried = None
    for t in range(n):
        if t in anchors:
            carried = anchors[t]
        elif carried is not None:
            carried = carried @ np.linalg.inv(steps[t])
        if carried is not None and in_bounds(carried, width, height):
            view[t] = True
    carried = None
    for t in range(n - 1, -1, -1):
        if t in anchors:
            carried = anchors[t]
        elif carried is not None:
            carried = carried @ steps[t + 1]
        if carried is not None and in_bounds(carried, width, height):
            view[t] = True
    return view


def task_span(
    view: np.ndarray, reference: int | None, fps: float, gap_s: float = TASK_GAP_S
) -> tuple[int, int]:
    """In-view run holding the reference, merged across gaps of <= ``gap_s``; whole clip without one."""
    n = len(view)
    if reference is None:
        return 0, n
    pieces = runs(view)
    holding = [piece for piece in pieces if piece[0] <= reference < piece[1]]
    if not holding:
        return 0, n
    lo, hi = holding[0]
    gap = round(gap_s * fps)
    merged = True
    while merged:
        merged = False
        for a, b in pieces:
            if b <= lo and lo - b <= gap and a < lo:
                lo, merged = a, True
            if a >= hi and a - hi <= gap and b > hi:
                hi, merged = b, True
    return lo, hi


def to_reference(steps: np.ndarray, reference: int | None) -> np.ndarray:
    """Frame t -> reference frame, composed from the steps alone (identity everywhere without one)."""
    n = len(steps)
    out = np.repeat(_EYE[None], n, axis=0)
    if reference is None:
        return out
    for t in range(reference + 1, n):
        out[t] = out[t - 1] @ np.linalg.inv(steps[t])
    for t in range(reference - 1, -1, -1):
        out[t] = out[t + 1] @ steps[t + 1]
    return out


def survey_source(
    capture,
    detector: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    max_frames: int = 0,
    gap_s: float = TASK_GAP_S,
) -> CameraSurvey | None:
    """One decode pass over an opened capture: steps, sampled ORB + persons, then reference and span."""
    fps = safe_fps(capture.get(cv2.CAP_PROP_FPS))
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)  # ty: ignore[unresolved-attribute]
    steps, measured, features, person = [], [], {}, {}
    previous, width, height, t = None, 0, 0, 0
    posed = 0
    # Indexed like `run.process_source`: every successful read is a frame index, a malformed
    # frame included, and `max_frames` counts well-formed frames; the survey stops reading at it.
    while not (max_frames and posed >= max_frames):
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            steps.append(_EYE.copy())
            measured.append(False)
            # The next frame's predecessor is this unreadable one: its step stays unmeasured.
            previous = None
            t += 1
            continue
        posed += 1
        height, width = frame.shape[:2]
        gray = _gray(frame, GMC_WORK)
        step = (
            estimate_step(previous, gray, max(width, height) / GMC_WORK)
            if previous is not None
            else None
        )
        steps.append(step if step is not None else _EYE.copy())
        measured.append(step is not None)
        if t % SAMPLE_EVERY == 0:
            small = _gray(frame, ORB_WORK)
            keypoints, descriptors = orb.detectAndCompute(small, None)
            k = max(width, height) / ORB_WORK
            points = (
                np.array([kp.pt for kp in keypoints], dtype=np.float32) * k
                if keypoints
                else np.zeros((0, 2), np.float32)
            )
            features[t] = (points, descriptors)
            boxes = (
                np.asarray(detector(frame), dtype=np.float64)
                if detector is not None
                else np.zeros((0, 4))
            )
            boxes = boxes.reshape(len(boxes), -1)[:, :4] if boxes.size else np.zeros((0, 4))
            area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
                boxes[:, 3] - boxes[:, 1], 0, None
            )
            person[t] = bool((area >= PERSON_AREA * width * height).any())
        previous = gray
        t += 1
    if not steps:
        return None
    steps_array, measured_array = np.array(steps), np.array(measured)
    motion = net_motion(steps_array, measured_array, width, height, fps)
    reference = choose_reference(motion, person, fps)
    anchors: dict[int, np.ndarray] = {}
    if reference is not None:
        anchors[reference] = _EYE.copy()
        scale = max(width, height) / ORB_WORK
        for f, feats in features.items():
            if f == reference:
                continue
            matrix, inliers = register(feats, features[reference], scale)
            if (
                matrix is not None
                and inliers >= ANCHOR_INLIERS
                and in_bounds(matrix, width, height)
            ):
                anchors[f] = matrix
    span = task_span(in_view(steps_array, anchors, width, height), reference, fps, gap_s)
    return CameraSurvey(
        width,
        height,
        fps,
        steps_array,
        measured_array,
        reference,
        span,
        to_reference(steps_array, reference),
    )
