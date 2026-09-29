"""Top-down pose tracker that keeps one box convention and one subject identity.

rtmlib's ``PoseTracker`` derives the next crop from the current pose between
detector calls (``pose_to_bbox`` over every keypoint, x1.25) and replaces it with
the detector's box every ``det_frequency`` frames.  The two conventions disagree
in extent, so the crop jumps on every detector frame and the top-down model
returns a different skeleton — a periodic artifact at ``fps / det_frequency``.
Its single-subject choice downstream was the argmax of the mean score over all
133 keypoints, 68 of them facial, so a small, fully visible bystander beat the
large patient whose head the frame cuts off.

Here every crop is a detector box.  Between detections, and when a detection
misses its track, the last box is carried and moved by the pose's own motion
(the median displacement of confident keypoints), never re-sized from the pose.
Tracks associate by box IoU against the carried box or the last detector box,
whichever overlaps more.  The subject is the track whose detector box is
largest, and it stays the subject until it is lost or another track stays
``switch_ratio`` times larger for ``switch_frames`` detector frames.
"""

from __future__ import annotations

import numpy as np
from rtmlib import PoseTracker

from .assignment import gated_assignment

EMPTY_KEYPOINTS = np.zeros((0, 133, 2), dtype=np.float64)
EMPTY_SCORES = np.zeros((0, 133), dtype=np.float64)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(n, 4)`` and ``(m, 4)`` xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def pose_shift(
    previous: np.ndarray | None,
    previous_scores: np.ndarray | None,
    current: np.ndarray,
    current_scores: np.ndarray,
    *,
    floor: float = 0.3,
    minimum: int = 3,
) -> np.ndarray:
    """Median displacement of keypoints confident in both poses, else zero."""
    if previous is None or previous_scores is None:
        return np.zeros(2)
    both = (
        (previous_scores >= floor)
        & (current_scores >= floor)
        & np.isfinite(previous).all(axis=1)
        & np.isfinite(current).all(axis=1)
    )
    if int(both.sum()) < minimum:
        return np.zeros(2)
    return np.median(current[both] - previous[both], axis=0)


class _Track:
    __slots__ = ("anchor", "area", "box", "id", "keypoints", "misses", "scores")

    def __init__(self, track_id: int, box: np.ndarray, area: float) -> None:
        self.id = track_id
        self.box = box
        # Last matched detector box.  Association scores against it as well as
        # the transported box, so one glitched pose that carries the box off the
        # person cannot orphan the track at the next detection.
        self.anchor = box.copy()
        self.area = area
        self.misses = 0
        self.keypoints: np.ndarray | None = None
        self.scores: np.ndarray | None = None


class SubjectTracker(PoseTracker):
    """``PoseTracker`` with detector-convention crops and a sticky subject.

    ``single_subject`` poses and returns the subject alone; otherwise every live
    track is posed and returned, subject first.  ``last_track_ids`` names the
    returned rows, so a downstream smoother can associate by identity.
    """

    def __init__(
        self,
        solution,
        *,
        det_frequency: int = 1,
        single_subject: bool = False,
        iou_threshold: float = 0.3,
        max_misses: int = 15,
        area_alpha: float = 0.2,
        switch_ratio: float = 1.5,
        switch_frames: int = 15,
        confidence_floor: float = 0.3,
        **kwargs,
    ) -> None:
        self.single_subject = single_subject
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.area_alpha = area_alpha
        self.switch_ratio = switch_ratio
        self.switch_frames = switch_frames
        self.confidence_floor = confidence_floor
        # rtmlib's IoU branch is unsound (rtmlib-runtime.md); association lives here.
        if kwargs.pop("tracking", False):
            raise ValueError("SubjectTracker owns association; tracking=True is not accepted")
        super().__init__(solution, det_frequency=det_frequency, tracking=False, **kwargs)
        # PoseTracker degrades to detector-less mode with a printed warning; every
        # crop here is a detector box, so that mode has nothing to crop from.
        if self.det_model is None:
            raise ValueError("SubjectTracker needs a top-down solution with a detector")
        self._detector = self.det_model

    def reset(self) -> None:
        super().reset()
        self.tracks: list[_Track] = []
        self.subject_id: int | None = None
        self.challenger: tuple[int, int] | None = None
        self.last_track_ids: list[int] = []

    # ------------------------------------------------------------------ association

    def _associate(self, boxes: np.ndarray, frame_area: float) -> None:
        matched_tracks: set[int] = set()
        matched_boxes: set[int] = set()
        if self.tracks and len(boxes):
            iou = np.maximum(
                box_iou(np.array([t.box for t in self.tracks]), boxes),
                box_iou(np.array([t.anchor for t in self.tracks]), boxes),
            )
            rows, cols = gated_assignment(1.0 - iou, threshold=1.0 - self.iou_threshold + 1e-12)
            for r, c in zip(rows, cols, strict=True):
                track = self.tracks[int(r)]
                track.box = boxes[int(c)].copy()
                track.anchor = boxes[int(c)].copy()
                area = _area(track.box) / frame_area
                track.area += self.area_alpha * (area - track.area)
                track.misses = 0
                matched_tracks.add(int(r))
                matched_boxes.add(int(c))
        for index, track in enumerate(self.tracks):
            if index not in matched_tracks:
                track.misses += 1
        for index, box in enumerate(boxes):
            if index not in matched_boxes:
                self.tracks.append(_Track(self.next_id, box.copy(), _area(box) / frame_area))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    def _select_subject(self, detected: bool) -> None:
        by_id = {t.id: t for t in self.tracks}
        if self.subject_id not in by_id:
            self.subject_id = max(self.tracks, key=lambda t: t.area).id if self.tracks else None
            self.challenger = None
            return
        if not detected:
            return
        subject = by_id[self.subject_id]
        rivals = [t for t in self.tracks if t.id != subject.id]
        best = max(rivals, key=lambda t: t.area) if rivals else None
        if best is None or best.area <= self.switch_ratio * subject.area:
            self.challenger = None
            return
        count = self.challenger[1] + 1 if self.challenger and self.challenger[0] == best.id else 1
        if count >= self.switch_frames:
            self.subject_id = best.id
            self.challenger = None
        else:
            self.challenger = (best.id, count)

    # ------------------------------------------------------------------ frame

    def __call__(self, image: np.ndarray):
        height, width = image.shape[:2]
        detected = self.frame_cnt % self.det_frequency == 0
        if detected:
            self._associate(_boxes(self._detector(image)), float(width * height))
        self._select_subject(detected)

        posed = (
            [t for t in self.tracks if t.id == self.subject_id]
            if self.single_subject
            else (sorted(self.tracks, key=lambda t: t.id != self.subject_id))
        )
        # A track that sat out a frame has no consecutive pose to move by.
        for track in self.tracks:
            if track not in posed:
                track.keypoints = track.scores = None
        keypoints, scores, ids = [], [], []
        for track in posed:
            kps, sc = self.pose_model(image, bboxes=[track.box])
            kps = np.asarray(kps, dtype=np.float64)[0]
            sc = np.asarray(sc, dtype=np.float64)[0]
            shift = pose_shift(track.keypoints, track.scores, kps, sc, floor=self.confidence_floor)
            track.box = track.box + np.concatenate([shift, shift])
            track.keypoints, track.scores = kps, sc
            keypoints.append(kps)
            scores.append(sc)
            ids.append(track.id)

        self.frame_cnt += 1
        self.last_track_ids = ids
        if not keypoints:
            return EMPTY_KEYPOINTS.copy(), EMPTY_SCORES.copy()
        return np.stack(keypoints), np.stack(scores)


def _area(box: np.ndarray) -> float:
    return float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))


def _boxes(raw) -> np.ndarray:
    """Detector boxes as ``(n, 4)`` xyxy; any trailing score column is dropped."""
    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4))
    return array.reshape(len(array), -1)[:, :4]
