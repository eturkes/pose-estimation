"""M2.9.1 P01-P13: contract-derived, synthetic-only, implementation-blind.

Independent scalar median/rectangle/assignment oracles exercise generated inputs;
runner and driver checks reach their shipped call paths rather than source spelling.
"""

from __future__ import annotations

import csv
import itertools
import math
import pathlib
import runpy
import statistics
import sys
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from rtmlib import PoseTracker

from pose_estimation import export
from pose_estimation import run as run_module
from pose_estimation.rtmlib_smoothing import KeypointSmoother

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_K = 133
_BOX = np.array([100.0, 100.0, 300.0, 400.0])
_EMPTY = np.empty((0, 4), dtype=np.float64)
_Pose = Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]]
_Detections = Callable[[int], np.ndarray]


def _image(index: int = 0, *, height: int = 600, width: int = 1200) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[0, 0, 0] = index
    return image


def _keypoints(x: float = 150.0, y: float = 200.0) -> np.ndarray:
    index = np.arange(_K, dtype=np.float64)
    return np.column_stack((x + index % 11, y + index % 13))


def _box_pose(_index: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoints = _keypoints(*box[:2])
    keypoints[0], keypoints[1] = box[:2], box[2:4]
    return keypoints, np.ones(_K)


def _constant_pose(_index: int, _box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _keypoints(), np.ones(_K)


class _Detector:
    mode = "balanced"

    def __init__(self, detections: _Detections) -> None:
        self.detections = detections
        self.frames: list[int] = []

    def __call__(self, image: np.ndarray) -> np.ndarray:
        index = int(image[0, 0, 0])
        self.frames.append(index)
        return np.asarray(self.detections(index), dtype=np.float64).copy()


class _PoseModel:
    def __init__(self, pose: _Pose) -> None:
        self.pose = pose
        self.calls: list[tuple[int, np.ndarray]] = []
        self.whole_frame_calls = 0

    def __call__(self, image: np.ndarray, bboxes: Any = ()) -> tuple[np.ndarray, np.ndarray]:
        index = int(image[0, 0, 0])
        boxes = np.asarray(bboxes, dtype=np.float64)
        self.calls.append((index, boxes.copy()))
        if len(boxes) == 0:
            self.whole_frame_calls += 1
            boxes = np.array([[0.0, 0.0, image.shape[1], image.shape[0]]])
        rows = [self.pose(index, box[:4]) for box in boxes]
        return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def _solution_factory(
    detections: _Detections, pose: _Pose = _constant_pose, *, detector: bool = True
) -> type:
    class _Solution:
        def __init__(self, **_kwargs: Any) -> None:
            self.det_model = _Detector(detections) if detector else None
            self.pose_model = _PoseModel(pose)
            self.det_categories = None
            self.one_stage = False

    return _Solution


def _subject(
    detections: _Detections = lambda _index: _BOX[None],
    pose: _Pose = _constant_pose,
    **kwargs: Any,
) -> Any:
    # Keep the new import inside each test's call path: a missing module is a
    # separate red per predicate, never a collection failure for the whole file.
    from pose_estimation.subject_tracker import SubjectTracker

    return SubjectTracker(_solution_factory(detections, pose), **kwargs)


def _ids(tracker: Any) -> list[int]:
    return list(tracker.last_track_ids)


def _drive(tracker: Any, frames: int, *, start: int = 0) -> list[list[int]]:
    observed = []
    for index in range(start, start + frames):
        keypoints, scores = tracker(_image(index))
        assert len(keypoints) == len(scores) == len(_ids(tracker))
        observed.append(_ids(tracker))
    return observed


def _reference_shift(
    previous: np.ndarray | None,
    previous_scores: np.ndarray | None,
    current: np.ndarray,
    scores: np.ndarray,
    floor: float,
) -> tuple[float, float]:
    if previous is None or previous_scores is None:
        return 0.0, 0.0
    deltas = [
        (float(now[0] - before[0]), float(now[1] - before[1]))
        for before, old_score, now, score in zip(
            previous, previous_scores, current, scores, strict=True
        )
        if old_score >= floor
        and score >= floor
        and all(math.isfinite(float(value)) for value in (*before, *now))
    ]
    if len(deltas) < 3:
        return 0.0, 0.0
    return statistics.median(row[0] for row in deltas), statistics.median(row[1] for row in deltas)


def _reference_iou(
    left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray
) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    areas = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (
        right[3] - right[1]
    )
    return overlap / (areas - overlap)


def _reference_assignment(
    old: np.ndarray, new: np.ndarray, threshold: float = 0.3, anchors: np.ndarray | None = None
) -> dict[int, int]:
    # Exhaustion is independent of gated_assignment/Hungarian's implementation.
    if anchors is None:
        anchors = old
    candidates = []
    for choices in itertools.product(range(-1, len(old)), repeat=len(new)):
        used = [index for index in choices if index >= 0]
        if len(used) != len(set(used)):
            continue
        overlaps = [
            max(
                _reference_iou(old[old_index], new[new_index]),
                _reference_iou(anchors[old_index], new[new_index]),
            )
            for new_index, old_index in enumerate(choices)
            if old_index >= 0
        ]
        if any(value < threshold for value in overlaps):
            continue
        candidates.append(((-len(used), sum(1.0 - value for value in overlaps)), choices))
    _, best = min(candidates)
    return {new_index: old_index for new_index, old_index in enumerate(best) if old_index >= 0}


def _rows_by_box(tracker: Any, index: int) -> dict[tuple[float, ...], int]:
    keypoints, scores = tracker(_image(index))
    assert len(keypoints) == len(scores) == len(_ids(tracker)) > 0
    return {
        tuple(row[:2].reshape(-1)): track_id
        for row, track_id in zip(keypoints, _ids(tracker), strict=True)
    }


def _box_id(rows: dict[tuple[float, ...], int], box: np.ndarray) -> int:
    matches = [
        track_id
        for marker, track_id in rows.items()
        if np.allclose(marker, box, rtol=2e-6, atol=1e-4)
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("columns", [4, 5])
def test_p01_crop_extent_changes_only_on_a_matched_detector_box(columns: int) -> None:
    boxes = (_BOX, _BOX + np.array([-10.0, -10.0, 20.0, 20.0]))

    def detections(index: int) -> np.ndarray:
        box = boxes[int(index >= 3)]
        return np.array([box if columns == 4 else np.append(box, 0.95)])

    def deform(index: int, _box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = _keypoints()
        points = np.array([200.0, 250.0]) + (base - np.median(base, axis=0)) * (1 + index * 0.7)
        return points, np.ones(_K)

    tracker = _subject(detections, deform, det_frequency=3)
    _drive(tracker, 6)
    assert len(tracker.pose_model.calls) == 6
    for index, received in tracker.pose_model.calls:
        assert len(received) == 1
        expected = boxes[int(index >= 3)]
        np.testing.assert_allclose(received[0, 2:4] - received[0, :2], expected[2:] - expected[:2])
    assert tracker.pose_model.whole_frame_calls == 0


@pytest.mark.parametrize("valid", [0, 1, 2, 3, 4, _K])
def test_p02_transport_uses_joint_confidence_and_per_axis_median(valid: int) -> None:
    previous = _keypoints()
    current = previous.copy()
    displacement = np.array([[1.0, 90.0], [7.0, 3.0], [11.0, -2.0], [15.0, 5.0]])
    current += displacement[np.arange(_K) % len(displacement)]
    old_scores = np.zeros(_K)
    scores = np.zeros(_K)
    old_scores[:valid] = scores[:valid] = 0.4
    # A score above floor in only one of the two frames is not an observation pair.
    old_scores[valid::2] = 1.0
    scores[valid + 1 :: 2] = 1.0
    tracker = _subject(
        pose=lambda index, _box: (previous, old_scores) if index == 0 else (current, scores),
        det_frequency=7,
        confidence_floor=0.4,
    )
    _drive(tracker, 3)
    expected = _reference_shift(previous, old_scores, current, scores, 0.4)
    np.testing.assert_array_equal(tracker.pose_model.calls[1][1][0, :4], _BOX)
    np.testing.assert_allclose(tracker.pose_model.calls[2][1][0, :4], _BOX + np.tile(expected, 2))


def test_p02_detector_box_is_posed_before_transport_prediction() -> None:
    matched = _BOX + np.array([15.0, 4.0, 25.0, 14.0])
    shift = np.array([3.0, -2.0])
    tracker = _subject(
        lambda index: _BOX[None] if index == 0 else matched[None],
        lambda index, _box: (_keypoints() + index * shift, np.ones(_K)),
        det_frequency=2,
    )
    _drive(tracker, 4)
    np.testing.assert_array_equal(tracker.pose_model.calls[2][1][0, :4], matched)
    np.testing.assert_allclose(tracker.pose_model.calls[3][1][0, :4], matched + np.tile(shift, 2))


def test_p02_pose_shift_generated_oracle_and_translation_invariance() -> None:
    from pose_estimation.subject_tracker import pose_shift

    rng = np.random.default_rng(209102)
    exercised: set[str] = set()
    for size in (0, 1, 2, 3, 4, 17, _K):
        for _ in range(20):
            previous = rng.normal(0.0, 100.0, (size, 2))
            current = previous + rng.normal(0.0, 30.0, (size, 2))
            old_scores = rng.choice([0.0, 0.49, 0.5, 0.8, np.inf, np.nan], size)
            scores = rng.choice([0.0, 0.49, 0.5, 0.8, np.inf, np.nan], size)
            if size:
                previous[rng.integers(size), rng.integers(2)] = rng.choice([np.nan, np.inf])
                current[rng.integers(size), rng.integers(2)] = rng.choice([np.nan, -np.inf])
            expected = _reference_shift(previous, old_scores, current, scores, 0.5)
            exercised.add("zero" if expected == (0.0, 0.0) else "median")
            actual = pose_shift(previous, old_scores, current, scores, floor=0.5)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
            offset = rng.uniform(-1000, 1000, 2)
            translated = pose_shift(
                previous + offset, old_scores, current + offset, scores, floor=0.5
            )
            np.testing.assert_allclose(translated, expected, rtol=0, atol=1e-10)
            order = rng.permutation(size)
            np.testing.assert_allclose(
                pose_shift(
                    previous[order], old_scores[order], current[order], scores[order], floor=0.5
                ),
                expected,
                rtol=0,
                atol=1e-10,
            )
    assert exercised == {"zero", "median"}
    np.testing.assert_array_equal(pose_shift(None, None, _keypoints(), np.ones(_K)), [0.0, 0.0])


def test_p02_nonfinite_coordinates_cannot_transport_the_crop() -> None:
    previous, current = _keypoints(), _keypoints() + np.array([8.0, -6.0])
    previous[3:, 0] = np.nan
    current[4::2, 1] = np.inf
    tracker = _subject(
        pose=lambda index, _box: (previous if index == 0 else current, np.ones(_K)),
        det_frequency=5,
    )
    _drive(tracker, 3)
    np.testing.assert_array_equal(
        tracker.pose_model.calls[2][1][0, :4], _BOX + np.array([8, -6, 8, -6])
    )


def test_p02_unposed_track_forgets_pose_before_subject_reentry() -> None:
    def detections(index: int) -> np.ndarray:
        subject = [50, 50, 350 if index >= 4 else 150, 250]
        rival = [500, 50, 680, 250] if index >= 2 else [500, 50, 650, 150]
        return np.array([subject, rival], dtype=np.float64)

    def poses(index: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = 75.0 if box[0] < 400 else 550.0
        if index >= 4 and box[0] < 400:
            position += 1000.0
        return _keypoints(position), np.ones(_K)

    tracker = _subject(
        detections,
        poses,
        det_frequency=2,
        single_subject=True,
        area_alpha=1.0,
        switch_frames=1,
    )
    trace = _drive(tracker, 6)
    assert trace[0] == trace[1] == trace[4] == trace[5]
    assert trace[2] == trace[3]
    assert trace[0] != trace[2]
    assert len(tracker.pose_model.calls) == 6
    np.testing.assert_array_equal(tracker.pose_model.calls[5][1][0, :4], detections(4)[0])


@pytest.mark.parametrize("frequency", [1, 2, 5, 7, 13])
@pytest.mark.parametrize("single_subject", [False, True])
def test_p03_cadence_and_pose_calls_are_independent_of_pose_jumps(
    frequency: int, single_subject: bool
) -> None:
    boxes = np.array([_BOX, _BOX + np.array([550.0, 0.0, 550.0, 0.0])])

    def discontinuous(index: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jump = 1000.0 if index in (3, 7, 21) else 0.0
        return _keypoints(*box[:2]) + jump, np.zeros(_K)

    tracker = _subject(
        lambda _index: boxes,
        discontinuous,
        det_frequency=frequency,
        single_subject=single_subject,
    )
    frames = 37
    _drive(tracker, frames)
    assert tracker.frame_cnt == frames
    assert tracker.det_model.frames == list(range(0, frames, frequency))
    assert len(tracker.det_model.frames) == math.ceil(frames / frequency)
    assert len(tracker.pose_model.calls) == frames * (1 if single_subject else len(boxes))
    assert tracker.pose_model.whole_frame_calls == 0
    assert all(len(boxes) == 1 for _, boxes in tracker.pose_model.calls)


@pytest.mark.parametrize("overlap", [0.299999, 0.3, 0.300001, 1.0])
def test_p04_iou_gate_includes_exactly_point_three(overlap: float) -> None:
    # Same-height containment gives IoU = new width / old width, exactly 3/10 at the gate.
    old = np.array([100.0, 100.0, 1100.0, 200.0])
    new = np.array([100.0, 100.0, 100.0 + 1000.0 * overlap, 200.0])
    tracker = _subject(
        lambda index: (old if index == 0 else new)[None],
        det_frequency=1,
        single_subject=False,
    )
    trace = _drive(tracker, 2)
    assert len(trace[0]) == 1
    if overlap >= 0.3:
        assert trace[1] == trace[0]
    else:
        assert len(trace[1]) == 2
        assert trace[0][0] in trace[1]
        assert len(set(trace[1]) - set(trace[0])) == 1


@pytest.mark.parametrize("frequency", [1, 5])
@pytest.mark.parametrize("max_misses", [0, 1, 3, 15])
def test_p04_misses_count_detector_frames_and_ids_never_recycle(
    frequency: int, max_misses: int
) -> None:
    reappear = frequency * (max_misses + 2)
    tracker = _subject(
        lambda index: _BOX[None] if index in (0, reappear) else _EMPTY,
        det_frequency=frequency,
        single_subject=False,
        max_misses=max_misses,
    )
    first = _drive(tracker, 1)[0]
    assert len(first) == 1
    for index in range(1, reappear + 1):
        keypoints, scores = tracker(_image(index))
        assert len(keypoints) == len(scores) == len(_ids(tracker))
        if index < frequency * (max_misses + 1):
            assert _ids(tracker) == first
        elif index < reappear:
            assert _ids(tracker) == []
        else:
            assert len(_ids(tracker)) == 1
            assert _ids(tracker)[0] > first[0]
    assert tracker.pose_model.whole_frame_calls == 0


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_p04_configured_iou_threshold_endpoints(threshold: float) -> None:
    moved = _BOX + np.array([600.0, 0.0, 600.0, 0.0])
    tracker = _subject(
        lambda index: _BOX[None] if index < 2 else moved[None],
        iou_threshold=threshold,
        single_subject=False,
    )
    trace = _drive(tracker, 3)
    assert trace[1] == trace[0]
    assert len(trace[0]) == 1
    if threshold == 0.0:
        assert trace[2] == trace[0]
    else:
        assert len(trace[2]) == 2
        assert trace[0][0] in trace[2]


def test_p04_detector_match_resets_consecutive_misses() -> None:
    tracker = _subject(
        lambda index: _BOX[None] if index % 3 == 0 else _EMPTY,
        det_frequency=1,
        max_misses=2,
    )
    trace = _drive(tracker, 13)
    assert len(trace[0]) == 1
    assert all(ids == trace[0] for ids in trace)


def test_p04_assignment_maximizes_cardinality_before_iou_cost() -> None:
    # IoU matrix [[.818, .429], [.333, 0]]: greedy takes .818 then loses a track.
    old = np.array([[0.0, 0.0, 100.0, 100.0], [60.0, 0.0, 160.0, 100.0]])
    new = np.array([[10.0, 0.0, 110.0, 100.0], [-40.0, 0.0, 60.0, 100.0]])
    expected = _reference_assignment(old, new)
    assert expected == {0: 1, 1: 0}
    tracker = _subject(
        lambda index: old if index == 0 else new,
        _box_pose,
        det_frequency=1,
        single_subject=False,
    )
    first, second = _rows_by_box(tracker, 0), _rows_by_box(tracker, 1)
    assert len(first) == len(second) == 2
    for new_index, old_index in expected.items():
        assert _box_id(second, new[new_index]) == _box_id(first, old[old_index])


def test_p04_generated_assignments_match_exhaustive_oracle_and_affine_scaling() -> None:
    rng = np.random.default_rng(209104)
    exercised: set[str] = set()
    for _ in range(36):
        count = int(rng.integers(2, 5))
        origins = rng.uniform([80.0, 80.0], [500.0, 250.0], (count, 2))
        sizes = rng.uniform([70.0, 80.0], [250.0, 200.0], (count, 2))
        old = np.column_stack((origins, origins + sizes))
        new = old[rng.permutation(count)] + np.tile(rng.uniform(-85.0, 85.0, (count, 2)), (1, 2))
        expected = _reference_assignment(old, new)
        exercised.add("full" if len(expected) == count else "partial")
        for scale, offset in ((1.0, np.zeros(4)), (2.5, np.array([30.0, -20.0] * 2))):
            before, after = old * scale + offset, new * scale + offset
            tracker = _subject(
                lambda index, before=before, after=after: before if index == 0 else after,
                _box_pose,
                det_frequency=1,
                single_subject=False,
            )
            first, second = _rows_by_box(tracker, 0), _rows_by_box(tracker, 1)
            assert len(first) == count
            assert len(second) == count + count - len(expected)
            for new_index, box in enumerate(after):
                if new_index in expected:
                    assert _box_id(second, box) == _box_id(first, before[expected[new_index]])
                else:
                    assert _box_id(second, box) not in first.values()
    assert exercised == {"full", "partial"}


def test_p04_anchor_recovers_identity_after_glitched_pose_transport() -> None:
    tracker = _subject(
        lambda _index: _BOX[None],
        lambda index, _box: (_keypoints() + (1000.0 if index == 1 else 0.0), np.ones(_K)),
        det_frequency=2,
        single_subject=False,
    )
    trace = _drive(tracker, 5)
    assert len(trace[0]) == 1
    assert all(ids == trace[0] for ids in trace)
    assert len(tracker.pose_model.calls) == 5
    np.testing.assert_array_equal(tracker.pose_model.calls[2][1][0, :4], _BOX)
    np.testing.assert_array_equal(tracker.pose_model.calls[4][1][0, :4], _BOX)


def test_p04_carried_box_match_updates_anchor_before_later_miss() -> None:
    moved = _BOX + np.array([250.0, 0.0, 250.0, 0.0])
    absent = _BOX + np.array([700.0, 0.0, 700.0, 0.0])

    def poses(index: int, _box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shift = 0.0 if index == 0 else 1250.0 if index == 3 else 250.0
        return _keypoints() + np.array([shift, 0.0]), np.ones(_K)

    tracker = _subject(
        lambda index: (_BOX if index == 0 else moved if index < 6 else absent)[None],
        poses,
        det_frequency=2,
        single_subject=False,
    )
    trace = _drive(tracker, 7)
    assert len(trace[0]) == 1
    assert trace[:6] == [trace[0]] * 6
    np.testing.assert_array_equal(tracker.pose_model.calls[4][1][0, :4], moved)
    assert len(trace[6]) == 2
    assert trace[0][0] in trace[6]
    assert len(set(trace[6]) - set(trace[0])) == 1


def test_p04_generated_anchor_or_transport_assignments_use_max_iou() -> None:
    rng = np.random.default_rng(209141)
    exercised: set[str] = set()
    for _ in range(24):
        origins = rng.uniform([100.0, 100.0], [600.0, 250.0], (3, 2))
        sizes = rng.uniform([80.0, 100.0], [180.0, 200.0], (3, 2))
        anchors = np.column_stack((origins, origins + sizes))
        shifts = rng.uniform(-250.0, 250.0, (3, 2))
        transported = anchors + np.tile(shifts, (1, 2))
        use_anchor = rng.integers(0, 2, 3).astype(bool)
        new = np.where(use_anchor[:, None], anchors, transported)
        new = new + np.tile(rng.uniform(-35.0, 35.0, (3, 2)), (1, 2))
        new = new[rng.permutation(3)]
        expected = _reference_assignment(transported, new, anchors=anchors)

        def poses(
            index: int, box: np.ndarray, anchors: np.ndarray = anchors, shifts: np.ndarray = shifts
        ) -> tuple[np.ndarray, np.ndarray]:
            points, scores = _box_pose(index, box)
            if index == 1:
                track = next(i for i, anchor in enumerate(anchors) if np.array_equal(box, anchor))
                points += shifts[track]
            return points, scores

        tracker = _subject(
            lambda index, anchors=anchors, new=new: anchors if index == 0 else new,
            poses,
            det_frequency=2,
            single_subject=False,
        )
        first = _rows_by_box(tracker, 0)
        tracker(_image(1))
        second = _rows_by_box(tracker, 2)
        assert len(first) == 3
        assert len(second) == 6 - len(expected)
        for new_index, old_index in expected.items():
            assert _box_id(second, new[new_index]) == _box_id(first, anchors[old_index])
            anchor_iou = _reference_iou(anchors[old_index], new[new_index])
            motion_iou = _reference_iou(transported[old_index], new[new_index])
            exercised.add("anchor" if anchor_iou > motion_iou else "transport")
    assert exercised == {"anchor", "transport"}


def test_p04_box_iou_matches_scalar_oracle_and_geometric_properties() -> None:
    from pose_estimation.subject_tracker import box_iou

    rng = np.random.default_rng(209140)
    pairs = [
        (_BOX, _BOX),
        (_BOX, _BOX + np.array([200.0, 0.0, 200.0, 0.0])),
        (_BOX, _BOX + np.array([300.0, 0.0, 300.0, 0.0])),
        (_BOX, np.array([150.0, 120.0, 200.0, 200.0])),
    ]
    for _ in range(100):
        origins = rng.uniform(-1000.0, 1000.0, (2, 2))
        sizes = rng.uniform(1.0, 1500.0, (2, 2))
        rectangles = np.column_stack((origins, origins + sizes))
        pairs.append((rectangles[0], rectangles[1]))
    observed: set[str] = set()
    for left, right in pairs:
        expected = _reference_iou(left, right)
        observed.add("overlap" if expected > 0 else "disjoint")
        assert box_iou(left, right) == pytest.approx(expected, abs=1e-12)
        assert box_iou(right, left) == pytest.approx(expected, abs=1e-12)
        for scale in (0.01, 1.0, 100.0):
            offset = np.array([30.0, -70.0, 30.0, -70.0])
            assert box_iou(left * scale + offset, right * scale + offset) == pytest.approx(
                expected, abs=1e-12
            )
    assert observed == {"overlap", "disjoint"}


@pytest.mark.parametrize("single_subject", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_p05_largest_box_wins_despite_lower_pose_confidence(
    single_subject: bool, reverse: bool
) -> None:
    small = np.array([700.0, 100.0, 800.0, 200.0])
    boxes = np.array([small, _BOX] if reverse else [_BOX, small])

    def scored_pose(_index: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        large = box[0] < 500
        return _keypoints(150 if large else 750), np.full(_K, 0.55 if large else 0.99)

    tracker = _subject(lambda _index: boxes, scored_pose, single_subject=single_subject)
    for index in range(20):
        keypoints, scores = tracker(_image(index))
        assert len(keypoints) == (1 if single_subject else 2)
        assert keypoints[0, 0, 0] == 150
        assert scores[0, 0] == pytest.approx(0.55)
    assert len(tracker.pose_model.calls) == 20 * (1 if single_subject else 2)


def test_p05_equal_area_chooses_first_track_not_detector_score() -> None:
    boxes = np.array([np.append(_BOX, 0.4), np.append(_BOX + np.array([600, 0, 600, 0]), 0.99)])
    tracker = _subject(lambda _index: boxes, _box_pose, single_subject=True)
    keypoints, _ = tracker(_image())
    np.testing.assert_array_equal(keypoints[0, :2].reshape(-1), boxes[0, :4])


def _challenge_boxes(index: int, *, rival_area: float = 18000.0) -> np.ndarray:
    subject = np.array([50.0, 50.0, 150.0, 150.0])
    area = 9000.0 if index == 0 else rival_area
    return np.array([subject, [500.0, 50.0, 500.0 + area / 100.0, 150.0]])


@pytest.mark.parametrize("frequency", [1, 3])
@pytest.mark.parametrize("switch_frames", [1, 3, 15])
def test_p06_switch_requires_consecutive_detector_frames(
    frequency: int, switch_frames: int
) -> None:
    tracker = _subject(
        _challenge_boxes,
        det_frequency=frequency,
        single_subject=False,
        area_alpha=1.0,
        switch_frames=switch_frames,
    )
    trace = _drive(tracker, frequency * switch_frames + 1)
    assert len(trace[0]) == 2
    subject, rival = trace[0]
    assert all(ids[0] == subject for ids in trace[:-1])
    assert trace[-1] == [rival, subject]


def test_p06_exact_switch_ratio_is_not_a_challenge() -> None:
    tracker = _subject(
        lambda index: _challenge_boxes(index, rival_area=15000.0),
        det_frequency=1,
        single_subject=False,
        area_alpha=1.0,
        switch_ratio=1.5,
        switch_frames=2,
    )
    trace = _drive(tracker, 8)
    assert len(trace[0]) == 2
    assert all(ids == trace[0] for ids in trace)


def test_p06_nonchallenging_detector_frame_restarts_count() -> None:
    tracker = _subject(
        lambda index: _challenge_boxes(index, rival_area=14000.0 if index == 3 else 18000.0),
        det_frequency=1,
        single_subject=False,
        area_alpha=1.0,
        switch_frames=3,
    )
    trace = _drive(tracker, 7)
    subject, rival = trace[0]
    assert [ids[0] for ids in trace] == [subject] * 6 + [rival]


def test_p06_different_challenger_restarts_count() -> None:
    def detections(index: int) -> np.ndarray:
        first, second = (90.0, 90.0) if index == 0 else (180.0, 160.0)
        if index >= 3:
            first, second = 160.0, 180.0
        return np.array(
            [[50, 50, 150, 150], [400, 50, 400 + first, 150], [800, 50, 800 + second, 150]],
            dtype=np.float64,
        )

    tracker = _subject(detections, single_subject=False, area_alpha=1.0, switch_frames=3)
    trace = _drive(tracker, 6)
    subject, _, challenger = trace[0]
    assert [ids[0] for ids in trace] == [subject] * 5 + [challenger]


def test_p06_area_ema_not_raw_area_controls_the_challenge() -> None:
    tracker = _subject(_challenge_boxes, single_subject=False, switch_frames=2)
    subject_area, rival_area = 10000.0, 9000.0
    count = 0
    switch_at = None
    for index in range(1, 20):
        rival_area = 0.2 * 18000.0 + 0.8 * rival_area
        count = count + 1 if rival_area > 1.5 * subject_area else 0
        if count == 2:
            switch_at = index
            break
    assert switch_at is not None
    assert switch_at > 2
    trace = _drive(tracker, switch_at + 1)
    subject, rival = trace[0]
    assert all(ids[0] == subject for ids in trace[:-1])
    assert trace[-1][0] == rival


def test_p06_area_ema_uses_fraction_of_each_frames_area() -> None:
    tracker = _subject(_challenge_boxes, single_subject=False, switch_frames=2)
    subject_area, rival_area = 10000.0 / 160000.0, 9000.0 / 160000.0
    tracker(_image(0, height=200, width=800))
    subject, rival = _ids(tracker)
    challenges = 0
    observed_switch = False
    for index in range(1, 20):
        subject_area = 0.2 * (10000.0 / 640000.0) + 0.8 * subject_area
        rival_area = 0.2 * (18000.0 / 640000.0) + 0.8 * rival_area
        challenges = challenges + 1 if rival_area > 1.5 * subject_area else 0
        tracker(_image(index, height=400, width=1600))
        if challenges < 2:
            assert _ids(tracker)[0] == subject
        else:
            assert _ids(tracker)[0] == rival
            assert index > 6, "raw-pixel area EMA switches at frame 6; this stimulus must differ"
            observed_switch = True
            break
    assert observed_switch


def test_p06_losing_subject_immediately_selects_largest_survivor() -> None:
    boxes = np.array([_BOX, [500, 50, 650, 150], [850, 50, 950, 150]], dtype=np.float64)
    tracker = _subject(
        lambda index: boxes if index == 0 else boxes[1:],
        single_subject=False,
        max_misses=1,
        switch_frames=15,
    )
    trace = _drive(tracker, 3)
    subject, largest, smallest = trace[0]
    assert trace[1][0] == subject
    assert trace[2] == [largest, smallest]


@pytest.mark.parametrize("single_subject", [False, True])
def test_p07_empty_shapes_no_fallback_and_recovery(single_subject: bool) -> None:
    tracker = _subject(
        lambda index: _BOX[None] if index == 2 else _EMPTY,
        single_subject=single_subject,
        max_misses=0,
    )
    for index in (0, 1):
        keypoints, scores = tracker(_image(index))
        assert keypoints.shape == (0, _K, 2)
        assert scores.shape == (0, _K)
        assert keypoints.dtype == scores.dtype == np.dtype(np.float64)
        assert _ids(tracker) == []
    assert tracker.pose_model.calls == []
    keypoints, scores = tracker(_image(2))
    assert keypoints.shape == (1, _K, 2)
    assert scores.shape == (1, _K)
    assert len(_ids(tracker)) == 1
    keypoints, scores = tracker(_image(3))
    assert (keypoints.shape, scores.shape) == ((0, _K, 2), (0, _K))
    assert _ids(tracker) == []
    assert len(tracker.pose_model.calls) == 1
    assert tracker.pose_model.whole_frame_calls == 0


def test_p07_subject_first_then_track_order_survives_detector_reordering() -> None:
    boxes = np.array([[30, 30, 130, 130], _BOX + np.array([300, 0, 300, 0]), [900, 50, 1020, 170]])
    tracker = _subject(
        lambda index: boxes if index == 0 else boxes[[2, 0, 1]],
        _box_pose,
        single_subject=False,
    )
    first, _ = tracker(_image())
    initial_ids = _ids(tracker)
    assert len(initial_ids) == 3
    np.testing.assert_array_equal(first[:, :2].reshape(3, 4), boxes[[1, 0, 2]])
    second, _ = tracker(_image(1))
    assert _ids(tracker) == initial_ids
    np.testing.assert_array_equal(second[:, :2].reshape(3, 4), boxes[[1, 0, 2]])


def test_p08_construction_is_a_stateless_upstream_subclass() -> None:
    tracker = _subject()
    assert isinstance(tracker, PoseTracker)
    assert tracker.tracking is False


@pytest.mark.parametrize("tracking", [True, 1, "enabled"])
def test_p08_truthy_tracking_is_refused(tracking: Any) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - Contract fixes type, not wording.
        _subject(tracking=tracking)


def test_p08_missing_detector_is_refused() -> None:
    from pose_estimation.subject_tracker import SubjectTracker

    with pytest.raises(ValueError):  # noqa: PT011 - Contract fixes type, not wording.
        SubjectTracker(_solution_factory(lambda _index: _EMPTY, detector=False))


def test_p08_reset_clears_identity_misses_challenger_and_clock() -> None:
    tracker = _subject(
        _challenge_boxes, single_subject=False, area_alpha=1.0, switch_frames=3, max_misses=0
    )
    original = _drive(tracker, 3)
    assert original[0] == original[-1]
    # Mint extra identities before reset; the fresh run must not inherit next_id.
    tracker.det_model.detections = lambda _index: _EMPTY
    assert _drive(tracker, 1, start=3) == [[]]
    tracker.det_model.detections = _challenge_boxes
    reborn = _drive(tracker, 1, start=4)[0]
    assert min(reborn) > max(original[0])
    tracker.reset()
    assert tracker.frame_cnt == 0
    assert _ids(tracker) == []
    replay = _drive(tracker, 4)
    fresh = _subject(
        _challenge_boxes, single_subject=False, area_alpha=1.0, switch_frames=3, max_misses=0
    )
    assert replay == _drive(fresh, 4)
    assert replay[:3] == [original[0]] * 3
    assert replay[3] == original[0][::-1]


def test_p09_explicit_identity_survives_thousand_pixel_jump() -> None:
    smoother = KeypointSmoother(min_track_age=1, match_thresh=20)
    first = _keypoints()[None]
    scores = np.ones((1, _K))
    smoother(first, scores, 0.0, track_ids=[37])
    keys = smoother.output_track_keys()
    assert len(keys) == 1
    output, _ = smoother(first + 1000, scores, 1 / 30, track_ids=[37])
    assert output is not None
    assert len(output) == 1
    assert smoother.output_track_keys() == keys
    assert len(smoother.tracks) == 1


def test_p09_named_first_frame_bypasses_age_gate_unnamed_does_not() -> None:
    named, unnamed = KeypointSmoother(min_track_age=3), KeypointSmoother(min_track_age=3)
    points, scores = _keypoints()[None], np.ones((1, _K))
    for index in range(3):
        plain, _ = unnamed(points, scores, index / 30)
        assert (plain is None) == (index < 2)
        identified, _ = named(points, scores, index / 30, track_ids=[0])
        assert identified is not None
        assert len(identified) == 1


def test_p09_identity_overrides_centroid_and_row_order() -> None:
    smoother = KeypointSmoother(min_track_age=1, carry_frames=0, match_thresh=10000)
    initial = np.stack([_keypoints(100), _keypoints(500), _keypoints(900)])
    scores = np.ones((3, _K))
    smoother(initial, scores, 0.0, track_ids=[4, 8, 12])
    keys = smoother.output_track_keys()
    assert len(keys) == 3
    # Leave each position unchanged while rotating ids: centroid-only matching
    # would keep the original keys, which is exactly the competing rule.
    smoother(initial, scores, 1 / 30, track_ids=[12, 4, 8])
    assert smoother.output_track_keys() == [keys[2], keys[0], keys[1]]
    smoother(initial[:1], scores[:1], 2 / 30, track_ids=[99])
    assert len(smoother.output_track_keys()) == 1
    assert smoother.output_track_keys()[0] not in keys


@pytest.mark.parametrize("missing", ["score", "coordinates"])
def test_p09_dropped_unobserved_row_does_not_shift_identity_alignment(missing: str) -> None:
    smoother = KeypointSmoother(min_track_age=3, carry_frames=0)
    points = np.stack([_keypoints(100), _keypoints(400), _keypoints(700)])
    scores = np.ones((3, _K))
    if missing == "score":
        scores[1] = 0.0
    else:
        points[1] = np.nan
    out, _ = smoother(points, scores, 0.0, track_ids=[10, 20, 30])
    keys = smoother.output_track_keys()
    assert out is not None
    assert len(out) == len(keys) == 2
    smoother(points[[2, 0]], scores[[2, 0]], 1 / 30, track_ids=[30, 10])
    assert smoother.output_track_keys() == keys[::-1]


def test_p09_carried_identity_survives_gap_then_expires() -> None:
    smoother = KeypointSmoother(min_track_age=3, carry_frames=1, match_thresh=1)
    points, scores = _keypoints()[None], np.ones((1, _K))
    smoother(points, scores, 0.0, track_ids=[71])
    first = smoother.output_track_keys()
    assert len(first) == 1
    carried, _ = smoother(np.empty((0, _K, 2)), np.empty((0, _K)), 1 / 30, track_ids=[])
    assert carried is not None
    assert len(carried) == 1
    assert smoother.output_track_keys() == first
    smoother(points + 1000, scores, 2 / 30, track_ids=[71])
    assert smoother.output_track_keys() == first
    smoother(None, None, 3 / 30, track_ids=[])
    expired, _ = smoother(None, None, 4 / 30, track_ids=[])
    assert expired is None
    assert smoother.output_track_keys() == []
    smoother(points, scores, 5 / 30, track_ids=[71])
    assert len(smoother.output_track_keys()) == 1
    assert smoother.output_track_keys()[0] != first[0]


@pytest.mark.parametrize(("rows", "ids"), [(0, [1]), (1, []), (1, [1, 2]), (2, [1])])
def test_p09_mismatched_raw_row_count_is_refused(rows: int, ids: list[int]) -> None:
    smoother = KeypointSmoother()
    with pytest.raises(ValueError):  # noqa: PT011 - Contract fixes type, not wording.
        smoother(np.zeros((rows, _K, 2)), np.zeros((rows, _K)), 0.0, track_ids=ids)
    if rows == 0:
        with pytest.raises(ValueError):  # noqa: PT011 - Contract fixes type, not wording.
            smoother(None, None, 0.0, track_ids=ids)


_UNNAMED_GOLDEN = (
    pathlib.Path(__file__).parent / "fixtures" / "m2u91" / "unnamed_smoother_golden.npz"
)


def test_p09_none_preserves_unnamed_smoothing_over_generated_sequences() -> None:
    # Oracle = the pre-M2.9.1 class on this exact stimulus (make_unnamed_smoother_golden.py);
    # comparing two instances of the new code alone passes a passthrough (reviewer-1 R14).
    golden = np.load(_UNNAMED_GOLDEN)
    implicit, explicit = KeypointSmoother(), KeypointSmoother()
    rng = np.random.default_rng(209109)
    observed = 0
    identities: list[dict[Any, int]] = [{}, {}]
    for index in range(30):
        points = np.stack([_keypoints(100), _keypoints(500)]) + rng.normal(0, 3, (2, _K, 2))
        scores = rng.uniform(0.5, 1.0, (2, _K))
        if index % 7 == 4:
            points, scores = None, None
        left = implicit(points, scores, index / 30)
        right = explicit(points, scores, index / 30, track_ids=None)
        for first, second in zip(left, right, strict=True):
            assert (first is None) == (second is None)
            if first is not None:
                observed += 1
                np.testing.assert_array_equal(first, second)
        if f"kps_{index}" in golden:
            np.testing.assert_array_equal(left[0], golden[f"kps_{index}"])
            np.testing.assert_array_equal(left[1], golden[f"sc_{index}"])
        else:
            assert left[0] is None
            assert left[1] is None
        canonical = [
            [known.setdefault(key, len(known)) for key in smoother.output_track_keys()]
            for smoother, known in zip((implicit, explicit), identities, strict=True)
        ]
        assert canonical[0] == canonical[1] == golden[f"keys_{index}"].tolist()
    assert observed > 0


class _CompileObserved(BaseException):
    pass


def test_p10_gpu_requests_f32_without_changing_cpu_or_npu(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openvino
    from rtmlib.tools.base import BaseTool

    from pose_estimation.rtmlib_openvino import _patch_rtmlib_openvino

    captured: list[tuple[str, dict[str, Any]]] = []
    shape = openvino.PartialShape([1, 3, 64, 64])
    port = SimpleNamespace(
        partial_shape=shape,
        shape=[1, 3, 64, 64],
        get_partial_shape=lambda: shape,
        get_any_name=lambda: "input",
    )
    model = SimpleNamespace(inputs=[port], input=lambda *_args: port, reshape=lambda *_args: None)

    class _Core:
        def read_model(self, *_args: Any, **_kwargs: Any) -> Any:
            return model

        def compile_model(self, model: Any, device_name: str, config: dict[str, Any]) -> Any:
            captured.append((device_name, dict(config)))
            raise _CompileObserved

    class _Tool(BaseTool):
        def __call__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(openvino, "Core", _Core)
    _patch_rtmlib_openvino()
    path = tmp_path / "synthetic.onnx"
    path.write_bytes(b"synthetic-model-read-by-fake-core")
    for device in ("CPU", "NPU", "GPU"):
        with pytest.raises(_CompileObserved):
            _Tool(
                onnx_model=str(path), model_input_size=(64, 64), backend="openvino", device=device
            )
    assert [device for device, _ in captured] == ["CPU", "NPU", "GPU"]
    configs = dict(captured)
    assert configs["CPU"] == configs["NPU"] == {"PERFORMANCE_HINT": "LATENCY"}
    assert set(configs["GPU"]) == {"PERFORMANCE_HINT", "INFERENCE_PRECISION_HINT"}
    assert configs["GPU"]["PERFORMANCE_HINT"] == "LATENCY"
    assert configs["GPU"]["INFERENCE_PRECISION_HINT"] in ("f32", openvino.Type.f32)


def _runner_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    *,
    detections: _Detections = lambda _index: _BOX[None],
    pose: _Pose = _constant_pose,
    options: Sequence[str] = (),
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    session = tmp_path / "synthetic-session"
    session.mkdir(exist_ok=True)
    monkeypatch.setattr(run_module, "SplitDeviceSolution", _solution_factory(detections, pose))
    monkeypatch.setattr(
        run_module, "_dispatch_sessions", lambda _args, **kwargs: captured.update(kwargs)
    )
    run_module.main(
        ["--session-dir", str(session), "--headless", "--backend", "onnxruntime", *options]
    )
    assert "pose_tracker" in captured
    return captured


@pytest.mark.parametrize("single_subject", [False, True])
def test_p11_runner_defaults_to_subject_and_forwards_single_subject(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, single_subject: bool
) -> None:
    built = _runner_components(
        monkeypatch,
        tmp_path,
        detections=lambda _index: np.array([_BOX, _BOX + np.array([500, 0, 500, 0])]),
        options=["--single-subject"] if single_subject else [],
    )
    tracker = built["pose_tracker"]
    assert type(tracker) is not PoseTracker, "P11: default runner still constructs upstream"
    from pose_estimation.subject_tracker import SubjectTracker

    assert isinstance(tracker, SubjectTracker)
    assert tracker.tracking is False
    keypoints, _ = tracker(_image())
    assert len(keypoints) == (1 if single_subject else 2)


def test_p11_explicit_rtmlib_builds_plain_stateless_upstream(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _runner_components(monkeypatch, tmp_path, options=["--tracker", "rtmlib"])[
        "pose_tracker"
    ]
    assert type(tracker) is PoseTracker
    assert tracker.tracking is False
    with pytest.raises(SystemExit) as invalid:
        run_module.parse_args(["--tracker", "unsupported"])
    assert invalid.value.code == 2


class _Capture:
    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: self.frames,
            cv2.CAP_PROP_FRAME_WIDTH: 1200.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 600.0,
            cv2.CAP_PROP_POS_MSEC: max(0, self.index - 1) * 1000 / 30,
        }.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= self.frames:
            return False, None
        image = _image(self.index)
        self.index += 1
        return True, image

    def release(self) -> None:
        self.released = True


def _source_args() -> SimpleNamespace:
    return SimpleNamespace(tracking="body", single_subject=True, headless=True, max_frames=0)


def test_p11_process_source_forwards_ids_only_when_tracker_exposes_them(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    class _SpySmoother:
        def reset(self) -> None:
            pass

        def __call__(
            self, keypoints: np.ndarray, scores: np.ndarray, t: float, **kwargs: Any
        ) -> Any:
            calls.append(kwargs)
            return keypoints, scores

    class _Tracker:
        def __call__(self, _image: np.ndarray) -> Any:
            if hasattr(self, "last_track_ids"):
                self.last_track_ids = [73]
            return _keypoints()[None], np.ones((1, _K))

    for identified in (False, True):
        capture, tracker = _Capture(3), _Tracker()
        if identified:
            tracker.last_track_ids = []
        monkeypatch.setattr(
            run_module, "open_capture", lambda *_args, capture=capture, **_kwargs: capture
        )
        calls.clear()
        run_module.process_source(
            _source_args(),
            tracker,
            "synthetic.mp4",
            draw_skeleton=None,
            smoother=_SpySmoother(),
            output_csv=tmp_path / f"ids-{identified}.csv",
        )
        assert capture.released
        assert calls == ([{"track_ids": [73]}] if identified else [{}]) * 3


def _export_chain(
    monkeypatch: pytest.MonkeyPatch, destination: pathlib.Path, built: dict[str, Any], frames: int
) -> list[dict[str, str]]:
    capture = _Capture(frames)
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: capture)
    run_module.process_source(
        _source_args(),
        built["pose_tracker"],
        "synthetic.mp4",
        draw_skeleton=None,
        smoother=built["smoother"],
        bone_smoother=built.get("bone_smoother"),
        output_csv=destination,
    )
    assert capture.released
    with destination.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_p12_default_chain_exports_larger_lower_confidence_subject_from_frame_zero(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boxes = np.array([_BOX, [700.0, 100.0, 800.0, 200.0]])

    def two_people(_index: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        patient = box[0] < 500
        # Wide-enough extent keeps legacy pose_to_bbox above its MIN_AREA gate.
        points = _keypoints(150 if patient else 750, 150)
        points[:, 0] += (np.arange(_K) % 2) * 50
        points[:, 1] += (np.arange(_K) % 3) * 50
        return points, np.full(_K, 0.6 if patient else 0.99)

    default = _runner_components(
        monkeypatch,
        tmp_path,
        detections=lambda _index: boxes,
        pose=two_people,
        options=["--single-subject", "--det-frequency", "7"],
    )
    rows = _export_chain(monkeypatch, tmp_path / "default.csv", default, 10)
    assert rows, "P12: the actual CSV must carry exported rows"
    prefix, names = export._body_keypoint_names(export.TRACKING_BODY)
    x_column = f"{prefix}_{names[0]}_x"
    observed_x = [float(row[x_column]) for row in rows]
    assert max(observed_x) < 0.3, f"P12: larger subject at x<.3; observed {observed_x}"
    assert [int(row["frame_idx"]) for row in rows] == list(range(10))
    legacy = _runner_components(
        monkeypatch,
        tmp_path,
        detections=lambda _index: boxes,
        pose=two_people,
        options=["--tracker", "rtmlib", "--single-subject", "--det-frequency", "7"],
    )
    legacy_rows = _export_chain(monkeypatch, tmp_path / "legacy.csv", legacy, 10)
    assert legacy_rows
    assert any(float(row[x_column]) > 0.5 for row in legacy_rows)


def _driver(name: str) -> dict[str, Any]:
    namespace = runpy.run_path(str(_ROOT / "scripts" / name), run_name="_m2u91_driver")
    return namespace["main"].__globals__


_DRIVERS = ("corpus_run_2d.py", "pilot_corpus_run.py")


def _driver_args(
    driver: dict[str, Any], monkeypatch: pytest.MonkeyPatch, root: pathlib.Path
) -> Any:
    monkeypatch.setattr(sys, "argv", ["driver"])
    args = driver["_parse_args"]()
    for name in ("inventory", "qualification", "sessions", "out"):
        path = root / name
        path.mkdir(exist_ok=True)
        setattr(args, name, path)
    args.report = root / "report.json"
    return args


@pytest.mark.parametrize("name", _DRIVERS)
def test_p13_driver_defaults_select_gpu_every_frame_subject(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(name)
    monkeypatch.setattr(sys, "argv", [name])
    args = driver["_parse_args"]()
    assert (args.det_device, args.det_frequency, getattr(args, "tracker", None)) == (
        "GPU",
        1,
        "subject",
    )


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize("tracker", ["subject", "rtmlib"])
def test_p13_driver_parses_and_forwards_tracker_to_real_run_command(
    name: str, tracker: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(name)
    args = _driver_args(driver, monkeypatch, tmp_path)
    args.tracker = tracker
    commands: list[list[str]] = []

    def call(command: list[str], **_kwargs: Any) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(driver["subprocess"], "call", call)
    if name == "corpus_run_2d.py":
        driver["_attempt_event"]("synthetic-event", args, tmp_path / "logs")
    else:
        driver["_run_event"](
            args.sessions / "synthetic-event", args.out, tmp_path / "run.log", args
        )
    run_commands = [command for command in commands if "pose_estimation.run" in command]
    assert len(run_commands) == 1
    command = run_commands[0]
    assert "--tracker" in command, f"P13: forwarding absent in {command}"
    assert command[command.index("--tracker") + 1] == tracker
    monkeypatch.setattr(sys, "argv", [name, "--tracker", tracker])
    assert driver["_parse_args"]().tracker == tracker
    monkeypatch.setattr(sys, "argv", [name, "--tracker", "unsupported"])
    with pytest.raises(SystemExit) as invalid:
        driver["_parse_args"]()
    assert invalid.value.code == 2


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize("tracker", ["subject", "rtmlib"])
def test_p13_driver_publishes_tracker_in_emitted_configuration(
    name: str, tracker: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    driver = _driver(name)
    args = _driver_args(driver, monkeypatch, tmp_path)
    args.tracker = tracker
    asset = SimpleNamespace(
        asset_id="synthetic-asset",
        event_id="synthetic-event",
        camera_name="cam-a",
        codec="h264",
        device_config="synthetic",
        rotation_deg=0,
        pts_monotonic=1,
        reported_frames=3,
    )
    (args.sessions / "generation.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(driver, "_parse_args", lambda: args)
    monkeypatch.setitem(driver, "tree_digest", lambda _path: "tree")
    monkeypatch.setitem(driver, "validate_generation", lambda *_args, **_kwargs: {})
    if name == "corpus_run_2d.py":
        args.analyse_only = True
        monkeypatch.setattr(driver["pilot"], "_load_assets", lambda *_args: [asset])
        monkeypatch.setitem(driver, "_canonical_asset_ids", lambda _path: [asset.asset_id])
        monkeypatch.setitem(driver, "generation_digest", lambda _path: "marker")
    else:
        args.reuse_run = True
        args.min_assets = 1
        # A03: reused output must name the pose configuration it was produced under.
        driver["write_pose_config"](args.out / asset.event_id, driver["pose_config"](args))
        monkeypatch.setitem(driver, "_assert_sources_validated", lambda *_args: None)
        monkeypatch.setitem(driver, "_load_assets", lambda *_args: [asset])
        monkeypatch.setitem(
            driver,
            "_guard_verdicts",
            lambda *_args: {"events_probed": 1, "default_output_refused": 1},
        )
        monkeypatch.setitem(driver, "_diagnostics", lambda *_args: [])
        monkeypatch.setitem(driver, "_partition", lambda *_args: driver["Partition"]())
    assert driver["main"]() in (0, 1)
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    assert payload["population"]["events"] == 1
    assert payload["configuration"].get("tracker") == tracker


@pytest.mark.parametrize("name", _DRIVERS)
def test_p13_report_vocabulary_and_generator_identity_include_tracker(name: str) -> None:
    driver = _driver(name)
    base_versions = {"corpus_run_2d.py": "v2", "pilot_corpus_run.py": "v1"}
    assert (
        "tracker" in driver["REPORT_FIELDS"],
        driver["GENERATOR_VERSION"] != base_versions[name],
    ) == (True, True)
