"""M2.9.2 P01-P07: contract-derived hygiene oracle, properties and pipeline witnesses."""

from __future__ import annotations

import csv
import importlib
import math
from statistics import mean, median
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

_LEFT = slice(91, 112)
_RIGHT = slice(112, 133)
_HANDS = (_LEFT, _RIGHT)
_FACE_TABLE = {1: 42, 3: 45, 4: 39, 6: 36, 9: 54, 10: 48}


def _hygiene():
    # Import during the call, so the unimplemented contract collects before failing.
    return importlib.import_module("pose_estimation.keypoint_hygiene")


def _pair(*, weak_side=0, weak=0.3, strong=0.8, offset=1.0, dtype=np.float64):
    points = np.full((1, 133, 2), 20.0, dtype=dtype)
    scores = np.full((1, 133), 0.75, dtype=dtype)
    points[0, _LEFT, 0] = np.linspace(0.0, 10.0, 21)
    points[0, _LEFT, 1] = 0.0
    points[0, _RIGHT] = points[0, _LEFT] + [0.0, offset]
    scores[0, _HANDS[weak_side]] = weak
    scores[0, _HANDS[1 - weak_side]] = strong
    return points, scores


def _zero_reference(points, scores, width, height):
    result = scores.copy()
    for person in range(len(points)):
        for index, (x, y) in enumerate(points[person]):
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                result[person, index] = 0
    return result


def _duplicate_reference(points, scores):
    """Scalar D02: independent loops, scalar geometry and statistics functions."""
    result = scores.copy()
    if points.shape[1] != 133:
        return result
    for person in range(len(points)):
        present = []
        for start in (91, 112):
            indices = {
                index
                for index in range(21)
                if all(math.isfinite(value) for value in points[person, start + index])
                and math.isfinite(scores[person, start + index])
                and scores[person, start + index] > 0
            }
            present.append(indices)
        common = sorted(present[0] & present[1])
        if len(common) < 10:
            continue
        extents = []
        means = []
        for start, indices in zip((91, 112), present, strict=True):
            xs, ys = zip(*(points[person, start + index] for index in indices), strict=True)
            extents.append(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
            means.append(mean(float(scores[person, start + index]) for index in indices))
        extent = max(extents)
        if extent == 0:
            continue
        distances = [
            math.dist(points[person, 91 + index], points[person, 112 + index]) for index in common
        ]
        weak_side = 0 if means[0] < means[1] else 1
        weak, strong = means[weak_side], means[1 - weak_side]
        if median(distances) / extent < 0.3 and weak < 0.5 and weak < 0.6 * strong:
            result[person, _HANDS[weak_side]] = 0
    return result


def _reference(points, scores, width, height):
    return _duplicate_reference(points, _zero_reference(points, scores, width, height))


def _assert_unchanged(points, scores, before):
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(scores, before[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_p01_half_open_bounds_and_nonfinite_coordinates(dtype):
    hygiene = _hygiene()
    width, height = 64, 48
    points = np.array(
        [
            [0, 0],
            [width - 0.5, height - 0.5],
            [0, height - 0.5],
            [width - 0.5, 0],
            [-0.5, 12],
            [12, -0.5],
            [width, 12],
            [12, height],
            [width + 0.5, 12],
            [12, height + 0.5],
            [np.nan, 12],
            [12, np.nan],
            [np.inf, 12],
            [12, np.inf],
            [-np.inf, 12],
            [12, -np.inf],
        ],
        dtype=dtype,
    )[None]
    scores = np.linspace(0.1, 0.9, points.shape[1], dtype=dtype)[None]
    before = points.copy(), scores.copy()
    actual = hygiene.zero_out_of_frame(points, scores, width, height)
    expected = scores.copy()
    expected[:, 4:] = 0
    np.testing.assert_array_equal(actual, expected)
    _assert_unchanged(points, scores, before)
    assert not np.shares_memory(actual, scores)


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
def test_p01_next_representable_values_straddle_each_frame_edge(axis):
    hygiene = _hygiene()
    bounds = (64.0, 48.0)
    bound = bounds[axis]
    points = np.full((1, 6, 2), 10.0)
    points[0, :, axis] = [
        np.nextafter(0.0, -np.inf),
        0.0,
        np.nextafter(0.0, np.inf),
        np.nextafter(bound, -np.inf),
        bound,
        np.nextafter(bound, np.inf),
    ]
    scores = np.full((1, 6), 0.875)
    actual = hygiene.zero_out_of_frame(points, scores, *bounds)
    np.testing.assert_array_equal(actual, [[0, 0.875, 0.875, 0.875, 0, 0]])


@pytest.mark.parametrize("seed", [17, 133, 2092])
def test_p01_generated_scores_obey_only_the_coordinate_mask(seed):
    hygiene = _hygiene()
    rng = np.random.default_rng(seed)
    observed_inside = observed_outside = 0
    for _ in range(80):
        n, k = int(rng.integers(1, 5)), int(rng.integers(1, 170))
        width, height = rng.integers(1, 2049, size=2)
        points = rng.uniform(-0.5, 1.5, size=(n, k, 2)) * [width, height]
        scores = rng.uniform(-1.0, 2.0, size=(n, k))
        points.reshape(-1)[::19] = np.nan
        points.reshape(-1)[::31] = np.inf
        scores.reshape(-1)[::23] = np.nan
        scores.reshape(-1)[::29] = np.inf
        before = points.copy(), scores.copy()
        expected = _zero_reference(points, scores, width, height)
        inside = (
            np.isfinite(points).all(axis=-1)
            & (points[..., 0] >= 0)
            & (points[..., 0] < width)
            & (points[..., 1] >= 0)
            & (points[..., 1] < height)
        )
        observed_inside += int(np.count_nonzero(inside))
        observed_outside += int(np.count_nonzero(~inside))
        actual = hygiene.zero_out_of_frame(points, scores, width, height)
        np.testing.assert_array_equal(actual, expected)
        _assert_unchanged(points, scores, before)
        assert not np.shares_memory(actual, scores)
    assert observed_inside > 0
    assert observed_outside > 0


@pytest.mark.parametrize("weak_side", [0, 1], ids=["left", "right"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_p02_duplicate_zeroes_all_and_only_the_weaker_hand(weak_side, dtype):
    hygiene = _hygiene()
    points, scores = _pair(weak_side=weak_side, dtype=dtype)
    before = points.copy(), scores.copy()
    actual = hygiene.suppress_duplicate_hand(points, scores)
    expected = scores.copy()
    expected[0, _HANDS[weak_side]] = 0
    np.testing.assert_array_equal(actual, expected)
    _assert_unchanged(points, scores, before)
    assert not np.shares_memory(actual, scores)


@pytest.mark.parametrize("weak_side", [0, 1], ids=["left", "right"])
def test_p02_exactly_ten_common_indices_suffice(weak_side):
    hygiene = _hygiene()
    points, scores = _pair(weak_side=weak_side)
    scores[0, 101:112] = 0
    scores[0, 122:133] = 0
    assert np.count_nonzero((scores[0, _LEFT] > 0) & (scores[0, _RIGHT] > 0)) == 10
    actual = hygiene.suppress_duplicate_hand(points, scores)
    expected = scores.copy()
    expected[0, _HANDS[weak_side]] = 0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("weak_side", [0, 1], ids=["larger-weaker", "larger-stronger"])
def test_p02_overlap_uses_the_larger_own_present_extent(weak_side):
    hygiene = _hygiene()
    points, scores = _pair(weak_side=weak_side, offset=3.0)
    points[0, _LEFT, 0] = np.r_[np.arange(10), np.full(11, 100.0)]
    points[0, 112:122, 0] = np.arange(10)
    scores[0, 122:133] = 0
    # Common-only or smaller-hand extent gives 3/9 >= 0.3; D02 gives 3/100.
    actual = hygiene.suppress_duplicate_hand(points, scores)
    expected = scores.copy()
    expected[0, _HANDS[weak_side]] = 0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("diagonal_extent", [False, True], ids=["distance", "extent"])
def test_p02_overlap_uses_euclidean_distance_and_bbox_diagonal(diagonal_extent):
    hygiene = _hygiene()
    points, scores = _pair(offset=0.0)
    if diagonal_extent:
        points[0, _LEFT, 1] = points[0, _LEFT, 0]
        points[0, _RIGHT] = points[0, _LEFT] + [0.0, 3.25]
        assert 3.25 / math.hypot(10, 10) < 0.3 < 3.25 / 10
    else:
        points[0, _RIGHT] = points[0, _LEFT] + [2.0, 2.0]
        assert math.hypot(2, 2) / 10 < 0.3 < (2 + 2) / 10
    expected = scores.copy()
    expected[0, _LEFT] = 0
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), expected)


@pytest.mark.parametrize(("distance", "fires"), [(6.0, True), (8.0, False)])
def test_p02_p03_even_common_count_uses_mean_of_two_middle_distances(distance, fires):
    hygiene = _hygiene()
    points, scores = _pair(offset=0.0)
    points[0, 91:101, 0] = points[0, 112:122, 0] = np.linspace(0, 10, 10)
    points[0, 117:122, 1] = distance
    scores[0, 101:112] = scores[0, 122:133] = 0
    overlap = (distance / 2) / math.hypot(10, distance)
    assert (overlap < 0.3) == fires
    expected = scores.copy()
    if fires:
        expected[0, _LEFT] = 0
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), expected)


@pytest.mark.parametrize("field", ["x", "y", "score"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg-inf"])
def test_p02_nonfinite_values_cannot_poison_ten_valid_correspondences(field, value):
    hygiene = _hygiene()
    points, scores = _pair()
    if field == "score":
        scores[0, 101:112] = value
    else:
        points[0, 101:112, 0 if field == "x" else 1] = value
    before = points.copy(), scores.copy()
    expected = scores.copy()
    expected[0, _LEFT] = 0
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), expected)
    _assert_unchanged(points, scores, before)


def test_p02_overlap_uses_median_instead_of_mean_distance():
    hygiene = _hygiene()
    points, scores = _pair(offset=0.0)
    points[0, 123:133, 1] = 100.0
    distances = np.linalg.norm(points[0, _LEFT] - points[0, _RIGHT], axis=1)
    extent = math.hypot(10.0, 100.0)
    assert np.median(distances) / extent == 0
    assert np.mean(distances) / extent > 0.3
    actual = hygiene.suppress_duplicate_hand(points, scores)
    expected = scores.copy()
    expected[0, _LEFT] = 0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fires", [False, True], ids=["own-mean-keeps", "own-mean-fires"])
def test_p02_p03_hand_mean_uses_own_present_indices_not_common_indices(fires):
    hygiene = _hygiene()
    points, scores = _pair(offset=0.0, strong=0.875)
    scores[0, 122:133] = 0
    scores[0, 91:101] = 0.75 if fires else 0.25
    scores[0, 101:112] = 0.0625 if fires else 0.875
    own_mean = float(np.mean(scores[0, _LEFT]))
    assert (own_mean < 0.5 and own_mean < 0.6 * 0.875) == fires
    actual = hygiene.suppress_duplicate_hand(points, scores)
    expected = scores.copy()
    if fires:
        expected[0, _LEFT] = 0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("offset", "weak", "strong", "common"),
    [
        (3.0, 0.3, 0.8, 21),
        (3.125, 0.3, 0.8, 21),
        (1.0, 0.5, 0.9375, 21),
        (1.0, 0.5625, 1.0, 21),
        (1.0, 0.375, 0.625, 21),
        (1.0, 0.4375, 0.625, 21),
        (1.0, 0.3, 0.8, 9),
        (1.0, 0.3, 0.8, 0),
        (1.0, 0.25, 0.25, 21),
    ],
    ids=[
        "overlap-equal",
        "overlap-above",
        "weak-equal",
        "weak-above",
        "ratio-equal",
        "ratio-above",
        "nine-common",
        "zero-common",
        "tie",
    ],
)
def test_p03_each_duplicate_gate_keeps_scores(offset, weak, strong, common):
    hygiene = _hygiene()
    points, scores = _pair(offset=offset, weak=weak, strong=strong)
    scores[0, 91 + common : 112] = 0
    scores[0, 112 + common : 133] = 0
    before = points.copy(), scores.copy()
    actual = hygiene.suppress_duplicate_hand(points, scores)
    np.testing.assert_array_equal(actual, scores)
    _assert_unchanged(points, scores, before)
    assert not np.shares_memory(actual, scores)


@pytest.mark.parametrize(
    ("offset", "weak", "strong"),
    [(2.999, 0.3, 0.8), (1.0, 0.499, 0.9375), (1.0, 0.374, 0.625)],
    ids=["below-overlap", "below-weak", "below-ratio"],
)
def test_p03_strict_boundaries_still_fire_just_below(offset, weak, strong):
    hygiene = _hygiene()
    points, scores = _pair(offset=offset, weak=weak, strong=strong)
    expected = scores.copy()
    expected[0, _LEFT] = 0
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), expected)


def test_p03_common_indices_are_corresponding_not_nearest_points():
    hygiene = _hygiene()
    points, scores = _pair(offset=0.0)
    points[0, _RIGHT] = points[0, _LEFT][::-1]
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), scores)


def test_p03_unobserved_points_do_not_inflate_extent_or_dilute_mean():
    hygiene = _hygiene()
    points, scores = _pair(offset=4.0)
    points[0, 91:101, 0] = np.linspace(0, 10, 10)
    points[0, 112:122, 0] = np.linspace(0, 10, 10)
    points[0, 101:112] = 1000.0
    points[0, 122:133] = -1000.0
    scores[0, 101:112] = scores[0, 122:133] = 0
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), scores)

    points, scores = _pair(offset=0.0, weak=0.5, strong=0.9375)
    scores[0, 101:112] = 0
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), scores)


@pytest.mark.parametrize("field", ["x", "y", "score"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg-inf"])
def test_p03_nonfinite_values_are_not_present(field, value):
    hygiene = _hygiene()
    points, scores = _pair(offset=0.0)
    if field == "score":
        scores[0, 100:112] = value
    else:
        points[0, 100:112, 0 if field == "x" else 1] = value
    before = points.copy(), scores.copy()
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), scores)
    _assert_unchanged(points, scores, before)


@pytest.mark.parametrize("value", [0.0, -0.1], ids=["zero", "negative"])
def test_p03_nonpositive_scores_are_not_present(value):
    hygiene = _hygiene()
    points, scores = _pair(offset=0.0)
    scores[0, 100:112] = value
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), scores)


@pytest.mark.parametrize("offset", [0.0, 1.0], ids=["coincident", "separate"])
def test_p03_zero_extent_hands_do_not_fire(offset):
    hygiene = _hygiene()
    points, scores = _pair()
    points[0, _LEFT] = [20.0, 20.0]
    points[0, _RIGHT] = [20.0, 20.0 + offset]
    with np.errstate(all="raise"):
        actual = hygiene.suppress_duplicate_hand(points, scores)
    np.testing.assert_array_equal(actual, scores)


@pytest.mark.parametrize("weak_side", [0, 1], ids=["left", "right"])
def test_p04_out_of_frame_zeroing_precedes_common_index_count(weak_side):
    hygiene = _hygiene()
    points, scores = _pair(weak_side=weak_side, offset=0.0)
    points[0, 100:112, 0] = 100.0
    points[0, 121:133, 0] = 100.0
    expected = _zero_reference(points, scores, 64, 48)
    assert np.count_nonzero((expected[0, _LEFT] > 0) & (expected[0, _RIGHT] > 0)) == 9
    before = points.copy(), scores.copy()
    actual = hygiene.apply_hygiene(points, scores, 64, 48)
    np.testing.assert_array_equal(actual, expected)
    _assert_unchanged(points, scores, before)


@pytest.mark.parametrize("n_keypoints", [0, 1, 17, 21, 91, 112, 132, 134, 170])
def test_p05_non_wholebody_counts_only_zero_out_of_frame(n_keypoints):
    hygiene = _hygiene()
    points = np.full((2, n_keypoints, 2), 10.0)
    scores = np.full((2, n_keypoints), 0.3)
    if n_keypoints:
        points[0, 0] = [64, 20]
    if n_keypoints >= 133:
        pair, pair_scores = _pair()
        points[:, :133] = pair
        scores[:, :133] = pair_scores
        points[0, 0] = [64, 20]
    expected = _zero_reference(points, scores, 64, 48)
    np.testing.assert_array_equal(hygiene.apply_hygiene(points, scores, 64, 48), expected)
    np.testing.assert_array_equal(hygiene.suppress_duplicate_hand(points, scores), scores)


@pytest.mark.parametrize("n_keypoints", [0, 17, 133, 134])
def test_p05_empty_person_axis_preserves_shape(n_keypoints):
    hygiene = _hygiene()
    points = np.empty((0, n_keypoints, 2))
    scores = np.empty((0, n_keypoints))
    for function, args in [
        (hygiene.zero_out_of_frame, (64, 48)),
        (hygiene.suppress_duplicate_hand, ()),
        (hygiene.apply_hygiene, (64, 48)),
    ]:
        actual = function(points, scores, *args)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == scores.shape
        assert actual.size == 0
        assert actual is not scores


def test_p05_each_person_is_independent_and_row_permutation_equivariant():
    hygiene = _hygiene()
    cases = [_pair(weak_side=0), _pair(weak_side=1), _pair(offset=4.0), _pair(offset=0.0)]
    cases[-1][1][0, 100:112] = 0
    points = np.concatenate([case[0] for case in cases])
    scores = np.concatenate([case[1] for case in cases])
    expected = _reference(points, scores, 64, 48)
    assert np.count_nonzero(expected[0, _LEFT]) == 0
    assert np.count_nonzero(expected[1, _RIGHT]) == 0
    np.testing.assert_array_equal(expected[2:], scores[2:])
    actual = hygiene.apply_hygiene(points, scores, 64, 48)
    np.testing.assert_array_equal(actual, expected)
    order = [2, 0, 3, 1]
    permuted = hygiene.apply_hygiene(points[order], scores[order], 64, 48)
    np.testing.assert_array_equal(permuted, actual[order])
    for index in range(len(points)):
        single = hygiene.apply_hygiene(points[index : index + 1], scores[index : index + 1], 64, 48)
        np.testing.assert_array_equal(single, actual[index : index + 1])


@pytest.mark.parametrize(
    "function", ["zero_out_of_frame", "suppress_duplicate_hand", "apply_hygiene"]
)
def test_p05_readonly_strided_inputs_return_independent_scores(function):
    hygiene = _hygiene()
    points, scores = _pair()
    storage_points = np.zeros((2, 266, 2))
    storage_scores = np.zeros((2, 266))
    storage_points[::2, ::2] = points
    storage_scores[::2, ::2] = scores
    points = storage_points[::2, ::2]
    scores = storage_scores[::2, ::2]
    before = points.copy(), scores.copy()
    points.flags.writeable = scores.flags.writeable = False
    args = () if function == "suppress_duplicate_hand" else (64, 48)
    actual = getattr(hygiene, function)(points, scores, *args)
    expected = (
        _zero_reference(points, scores, 64, 48)
        if function == "zero_out_of_frame"
        else _reference(points, scores, 64, 48)
    )
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, storage_scores)
    _assert_unchanged(points, scores, before)


def _generated_pairs(seed, count=180):
    rng = np.random.default_rng(seed)
    for case in range(count):
        n = int(rng.integers(1, 5))
        points = rng.uniform(-10, 110, size=(n, 133, 2))
        scores = rng.uniform(0.6, 1.0, size=(n, 133))
        for person in range(n):
            points[person, _LEFT] = rng.uniform(20, 70, size=(21, 2))
            points[person, _RIGHT] = points[person, _LEFT] + rng.normal(
                0, (0.3, 3.0, 30.0)[case % 3], size=(21, 2)
            )
            weak_side = int(rng.integers(2))
            scores[person, _HANDS[weak_side]] = rng.uniform(0.05, 0.45, size=21)
            missing = rng.choice(21, size=int(rng.integers(0, 16)), replace=False)
            scores[person, 91 + missing] = 0
            if case % 7 == 0:
                points[person, 112 + missing, 0] = np.nan
            if case % 11 == 0:
                scores[person, 112 + missing] = np.inf
        yield points, scores


@pytest.mark.parametrize("seed", [92, 133, 2092])
def test_p02_p05_generated_differential_and_hygiene_properties(seed):
    hygiene = _hygiene()
    fired = kept = 0
    for points, scores in _generated_pairs(seed):
        before = points.copy(), scores.copy()
        bounded = _zero_reference(points, scores, 100, 100)
        expected = _duplicate_reference(points, bounded)
        for person in range(len(points)):
            if np.array_equal(expected[person], bounded[person]):
                kept += 1
            else:
                fired += 1
        actual = hygiene.apply_hygiene(points, scores, 100, 100)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(hygiene.zero_out_of_frame(points, scores, 100, 100), bounded)
        np.testing.assert_array_equal(
            hygiene.suppress_duplicate_hand(points, scores), _duplicate_reference(points, scores)
        )
        np.testing.assert_array_equal(hygiene.apply_hygiene(points, actual, 100, 100), actual)
        np.testing.assert_array_equal(hygiene.apply_hygiene(points * 4, scores, 400, 400), actual)
        order = np.arange(len(points))[::-1]
        np.testing.assert_array_equal(
            hygiene.apply_hygiene(points[order], scores[order], 100, 100), actual[order]
        )
        _assert_unchanged(points, scores, before)
        assert not np.shares_memory(actual, scores)
    assert fired > 0
    assert kept > 0


class _Capture:
    def __init__(self, shapes):
        self.shapes = shapes
        self.index = 0
        self.released = False

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: len(self.shapes),
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_POS_MSEC: max(0, self.index - 1) * 1000 / 30,
        }.get(prop, 0.0)

    def isOpened(self):
        return not self.released

    def read(self):
        if self.index == len(self.shapes):
            return False, None
        shape = self.shapes[self.index]
        self.index += 1
        return True, np.zeros((*shape, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _Tracker:
    def __init__(self, outputs):
        self.outputs = outputs
        self.index = 0
        self.seen_shapes = []
        self.reset_calls = 0

    def reset(self):
        self.index = 0
        self.reset_calls += 1

    def __call__(self, frame):
        self.seen_shapes.append(frame.shape[:2])
        result = self.outputs[self.index]
        self.index += 1
        return result


class _RecordingSmoother:
    def __init__(self):
        self.seen = []

    def reset(self):
        self.seen.clear()

    def __call__(self, points, scores, timestamp, **_kwargs):
        self.seen.append((points.copy(), scores.copy(), timestamp))
        return points, scores


def _pipeline(
    tmp_path,
    monkeypatch,
    outputs,
    *,
    shapes=None,
    smoother=None,
    tracking="body",
    tracker_kind="stateful",
):
    run = importlib.import_module("pose_estimation.run")
    export = importlib.import_module("pose_estimation.export")
    shapes = shapes if shapes is not None else [(48, 64)] * len(outputs)
    capture = _Capture(shapes)
    tracker = _Tracker(outputs)
    before = [(points.copy(), scores.copy()) for points, scores in outputs]
    monkeypatch.setattr(run, "open_capture", lambda *_args, **_kwargs: capture)
    output_csv = tmp_path / "synthetic-hygiene.csv"
    run.process_source(
        SimpleNamespace(tracking=tracking, headless=True, single_subject=False, max_frames=0),
        tracker if tracker_kind == "stateful" else lambda frame: tracker(frame),
        "synthetic-hygiene.mp4",
        draw_skeleton=None,
        smoother=smoother,
        output_csv=output_csv,
    )
    with output_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        assert reader.fieldnames == export.make_csv_header(tracking)
        if tracking == "body":
            assert len(reader.fieldnames) == 304
        rows = list(reader)
    assert len(rows) == len(outputs)
    assert tracker.seen_shapes == shapes
    assert capture.released
    for (points, scores), snapshot in zip(outputs, before, strict=True):
        _assert_unchanged(points, scores, snapshot)
    return rows


@pytest.mark.parametrize("tracker_kind", ["stateful", "callable"])
def test_p06_each_decoded_size_reaches_smoother_after_hygiene(tmp_path, monkeypatch, tracker_kind):
    points, scores = _pair()
    points[0, 5] = [60, 25]
    points[0, 6] = [25, 60]
    points[0, 7] = [64, 10]
    smoother = _RecordingSmoother()
    shapes = [(48, 64), (64, 48)]
    _pipeline(
        tmp_path,
        monkeypatch,
        [(points, scores)] * 2,
        shapes=shapes,
        smoother=smoother,
        tracker_kind=tracker_kind,
    )
    assert len(smoother.seen) == len(shapes)
    for (seen_points, seen_scores, _timestamp), (height, width) in zip(
        smoother.seen, shapes, strict=True
    ):
        np.testing.assert_array_equal(seen_points, points)
        np.testing.assert_array_equal(seen_scores, _reference(points, scores, width, height))


@pytest.mark.parametrize("n_keypoints", [17, 133])
@pytest.mark.parametrize("tracking", ["body", "hands-arms"])
def test_p06_out_of_frame_point_exports_zero_visibility(
    tmp_path, monkeypatch, n_keypoints, tracking
):
    points = np.full((1, n_keypoints, 2), 20.0)
    scores = np.full((1, n_keypoints), 0.8125)
    points[0, 5] = [64.0, 25.0]
    rows = _pipeline(tmp_path, monkeypatch, [(points, scores)], tracking=tracking)
    prefix = "body" if tracking == "body" else "arm"
    assert float(rows[0][f"{prefix}_left_shoulder_vis"]) == 0
    assert float(rows[0][f"{prefix}_right_shoulder_vis"]) == 0.8125
    assert float(rows[0][f"{prefix}_left_shoulder_x"]) == 1.0


@pytest.mark.parametrize("weak_side", [0, 1], ids=["left", "right"])
@pytest.mark.parametrize("tracking", ["body", "hands-arms", "hands"])
def test_p06_suppressed_hand_exports_21_zero_confidences(
    tmp_path, monkeypatch, weak_side, tracking
):
    points, scores = _pair(weak_side=weak_side)
    points[:, 91:] += [10, 20]
    rows = _pipeline(tmp_path, monkeypatch, [(points, scores)], tracking=tracking)
    weak_name, strong_name = ("left", "right") if weak_side == 0 else ("right", "left")
    weak_columns = [f"{weak_name}_hand_{index}_conf" for index in range(21)]
    strong_columns = [f"{strong_name}_hand_{index}_conf" for index in range(21)]
    assert len(weak_columns) == 21
    np.testing.assert_array_equal([float(rows[0][column]) for column in weak_columns], np.zeros(21))
    np.testing.assert_allclose(
        [float(rows[0][column]) for column in strong_columns], 0.8, rtol=0, atol=1e-15
    )


def test_p06_hygiene_before_real_smoother_holds_previous_position(tmp_path, monkeypatch):
    run = importlib.import_module("pose_estimation.run")
    points = np.full((1, 17, 2), 20.0)
    points[0, 0] = [60.0, 24.0]
    scores = np.full((1, 17), 0.875)
    outside = points.copy()
    outside[0, 0] = [64.0, 24.0]
    rows = _pipeline(
        tmp_path,
        monkeypatch,
        [(points, scores), (outside, scores)],
        smoother=run.KeypointSmoother(min_track_age=1),
    )
    assert float(rows[0]["body_nose_x"]) == 60 / 64
    assert float(rows[1]["body_nose_x"]) == float(rows[0]["body_nose_x"])
    assert float(rows[1]["body_nose_vis"]) == 0
    assert float(rows[1]["body_left_shoulder_vis"]) == 0.875


@pytest.mark.parametrize(("mediapipe", "face_sub"), list(_FACE_TABLE.items()))
def test_p07_face_table_and_mapping_follow_ibug_subject_sides(mediapipe, face_sub):
    mapping = importlib.import_module("pose_estimation.mapping")
    processing = importlib.import_module("pose_estimation.processing")
    points = np.arange(266, dtype=float).reshape(1, 133, 2)
    scores = np.linspace(0.1, 0.9, 133)[None]
    body, visibility, _hands, _matches = mapping.coco_to_mediapipe(
        points, scores, 133, processing.TRACKING_BODY
    )
    np.testing.assert_array_equal(body[0][mediapipe, :2], points[0, 23 + face_sub])
    assert visibility[0][mediapipe] == scores[0, 23 + face_sub]
    assert dict(mapping._COCO_TO_BODY_FACE) == _FACE_TABLE


def test_p07_generated_ibug_faces_preserve_the_subjects_left_and_right():
    mapping = importlib.import_module("pose_estimation.mapping")
    processing = importlib.import_module("pose_estimation.processing")
    rng = np.random.default_rng(300)
    for _ in range(60):
        midline, scale = rng.uniform(100, 500), rng.uniform(1, 50)
        points = np.zeros((1, 133, 2))
        points[..., 0] = midline
        points[..., 1] = 50
        points[0, 1, 0] = midline + scale
        points[0, 2, 0] = midline - scale
        points[0, 23 + 36 : 23 + 42, 0] = midline - rng.uniform(0.5, 1.5, 6) * scale
        points[0, 23 + 42 : 23 + 48, 0] = midline + rng.uniform(0.5, 1.5, 6) * scale
        points[0, 23 + 48, 0] = midline - scale
        points[0, 23 + 54, 0] = midline + scale
        body, _visibility, _hands, _matches = mapping.coco_to_mediapipe(
            points, np.ones((1, 133)), 133, processing.TRACKING_BODY
        )
        assert len(body) == 1
        assert np.all(body[0][[1, 2, 3, 9], 0] > midline)
        assert np.all(body[0][[4, 5, 6, 10], 0] < midline)
