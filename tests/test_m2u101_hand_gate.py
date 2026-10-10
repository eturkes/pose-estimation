"""M2.10.1 P01-P06: diff-blind gate properties and synthetic pipeline/driver witnesses."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_HANDS = (slice(91, 112), slice(112, 133))
_COUNTERS = ("hand_frames_present", "hand_frames_gated")
_DRIVERS = ("corpus_run_2d.py", "pilot_corpus_run.py")
_OMITTED = object()


def _gate(**kwargs):
    module = importlib.import_module("pose_estimation.keypoint_hygiene")
    cls = getattr(module, "HandPresenceGate", None)
    assert callable(cls), "P01-P05: shipped keypoint_hygiene.HandPresenceGate is missing"
    return cls(**kwargs)


def _scores(left=0.8, right=0.8, *, rows=1, dtype=np.float64):
    result = np.full((rows, 133), 0.875, dtype=dtype)
    result[:, _HANDS[0]], result[:, _HANDS[1]] = left, right
    return result


def _counts(gate):
    return tuple(getattr(gate, field) for field in _COUNTERS)


def test_p01_frozen_constants():
    module = importlib.import_module("pose_estimation.keypoint_hygiene")
    assert getattr(module, "HAND_GATE_ON", None) == 0.65
    assert getattr(module, "HAND_GATE_OFF", None) == 0.5
    assert getattr(module, "HAND_GATE_MIN_POINTS", None) == 10


def test_p01_nc1_exact_on_boundary_is_inclusive():
    gate = _gate()
    scores = _scores(0, 0)
    # Binary-exact summands: 6.5 / 10 is exactly the float used by HAND_GATE_ON.
    scores[0, 91:101] = [0.5] * 6 + [0.875] * 4
    assert scores[0, 91:101].mean() == 0.65
    np.testing.assert_array_equal(gate(scores, [7]), scores)
    assert _counts(gate) == (1, 0)


@pytest.mark.parametrize("hand", [0, 1], ids=["left", "right"])
def test_p01_hysteresis_threshold_trace(hand):
    gate = _gate()
    levels = (0.64, 0.65, 0.64, 0.50, 0.49, 0.50, 0.64, 0.65)
    admitted = (False, True, True, True, False, False, False, True)
    for level, keep in zip(levels, admitted, strict=True):
        scores = _scores(0, 0)
        scores[0, _HANDS[hand]] = level
        actual = gate(scores, [7])
        np.testing.assert_array_equal(actual[0, _HANDS[hand]], level if keep else 0)
    assert _counts(gate) == (len(levels), admitted.count(False))


@pytest.mark.parametrize("n_positive", [0, 1, 9, 10, 20, 21])
@pytest.mark.parametrize("other", [0.0, -0.0, -1.0, np.nan, np.inf, -np.inf])
def test_p01_presence_counts_only_positive_finite_points(n_positive, other):
    gate = _gate()
    scores = _scores(0, 0)
    scores[0, _HANDS[0]] = other
    scores[0, 91 : 91 + n_positive] = 0.75
    actual = gate(scores, [7])
    np.testing.assert_array_equal(
        actual[0, _HANDS[0]], scores[0, _HANDS[0]] if n_positive >= 10 else 0
    )
    assert _counts(gate) == ((1, 0) if n_positive >= 10 else (0, 0))


@pytest.mark.parametrize(
    ("level", "admitted"),
    [(np.nextafter(0.0, 1.0), False), (np.finfo(np.float64).max, True)],
    ids=["smallest-positive", "largest-finite"],
)
def test_p01_finite_positive_score_extremes(level, admitted):
    gate = _gate()
    scores = _scores(level, level)
    actual = gate(scores, [7])
    np.testing.assert_array_equal(actual[:, 91:], scores[:, 91:] if admitted else 0)
    assert _counts(gate) == (2, 0 if admitted else 2)


def test_p01_nine_points_turn_an_on_hand_off_and_require_reentry():
    gate = _gate()
    gate(_scores(), [7])
    sparse = _scores(0, 0.8)
    sparse[0, 91:100] = 1.0
    actual = gate(sparse, [7])
    np.testing.assert_array_equal(actual[0, _HANDS[0]], 0)
    actual = gate(_scores(0.6, 0.8), [7])
    np.testing.assert_array_equal(actual[0, _HANDS[0]], 0)
    np.testing.assert_array_equal(actual[0, _HANDS[1]], 0.8)


def test_p01_positive_score_mean_not_all_point_mean_or_count_above_threshold():
    gate = _gate()
    scores = _scores(0, 0)
    scores[0, 91:101] = [0.5] * 5 + [1.0] * 5
    scores[0, 101:112] = [0, -1, np.nan, np.inf, -np.inf, 0, 0, 0, 0, 0, 0]
    assert np.mean(scores[0, 91:101]) == 0.75
    np.testing.assert_array_equal(gate(scores, [7]), scores)
    assert _counts(gate) == (1, 0)


@pytest.mark.parametrize("min_points", [1, 5, 21])
def test_p01_constructor_thresholds_and_min_points_are_effective(min_points):
    gate = _gate(on=0.875, off=0.25, min_points=min_points)
    scores = _scores(0, 0)
    scores[0, 91 : 91 + min_points] = 0.875
    np.testing.assert_array_equal(gate(scores, [7]), scores)
    scores[0, 91 : 91 + min_points] = 0.25
    np.testing.assert_array_equal(gate(scores, [7]), scores)
    scores[0, 91 : 91 + min_points] = 0.125
    np.testing.assert_array_equal(gate(scores, [7])[0, _HANDS[0]], 0)
    scores[0, 91 : 91 + min_points] = 0.75
    np.testing.assert_array_equal(gate(scores, [7])[0, _HANDS[0]], 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_p02_generated_copy_and_untouched_score_property(dtype):
    gate = _gate()
    rng = np.random.default_rng(210101)
    n_off = n_on = 0
    for _ in range(40):
        gate.reset()
        rows = int(rng.integers(1, 8))
        backing = rng.uniform(-1, 1, (rows, 266)).astype(dtype)
        scores = backing[:, ::2]
        off = rng.integers(0, 2, (rows, 2)).astype(bool)
        for row in range(rows):
            for hand, indices in enumerate(_HANDS):
                scores[row, indices] = (
                    rng.uniform(0.1, 0.4, 21) if off[row, hand] else rng.uniform(0.75, 1, 21)
                )
        before = backing.copy()
        scores.flags.writeable = False
        actual = gate(scores, list(range(rows)))
        assert actual.shape == scores.shape
        assert not np.shares_memory(actual, scores)
        np.testing.assert_array_equal(backing, before)
        np.testing.assert_array_equal(actual[:, :91], scores[:, :91])
        for row in range(rows):
            for hand, indices in enumerate(_HANDS):
                np.testing.assert_array_equal(
                    actual[row, indices], 0 if off[row, hand] else scores[row, indices]
                )
        n_off += int(off.sum())
        n_on += int((~off).sum())
    assert n_off > 0
    assert n_on > 0


@pytest.mark.parametrize("shape", [(0, 133), (0, 17), (2, 0), (2, 1), (2, 17), (2, 132), (2, 134)])
def test_p02_empty_and_non_wholebody_are_independent_unchanged_arrays(shape):
    gate = _gate()
    scores = np.arange(np.prod(shape), dtype=float).reshape(shape)
    before = scores.copy()
    scores.flags.writeable = False
    actual = gate(scores, list(range(shape[0])))
    assert actual is not scores
    assert actual.shape == scores.shape
    assert not np.shares_memory(actual, scores)
    np.testing.assert_array_equal(actual, before)
    np.testing.assert_array_equal(scores, before)
    assert _counts(gate) == (0, 0)


def test_p03_nc3_state_follows_keys_and_each_hand_through_row_swap():
    gate = _gate()
    scores = np.vstack([_scores(0.8, 0.4), _scores(0.4, 0.8)])
    gate(scores, [11, 29])
    scores = _scores(0.55, 0.55, rows=2)
    actual = gate(scores, [29, 11])
    for row, off_hand in ((0, 0), (1, 1)):
        np.testing.assert_array_equal(actual[row, _HANDS[off_hand]], 0)
        np.testing.assert_array_equal(actual[row, _HANDS[1 - off_hand]], 0.55)


def test_p03_prune_drops_only_absent_keys_without_resetting_counters():
    gate = _gate()
    gate(_scores(rows=2), [11, 29])
    before = _counts(gate)
    gate.prune([11])
    assert _counts(gate) == before
    actual = gate(_scores(0.55, 0.55, rows=2), [29, 11])
    np.testing.assert_array_equal(actual[0, 91:], 0)
    np.testing.assert_array_equal(actual[1, 91:], 0.55)
    gate.prune([])
    np.testing.assert_array_equal(gate(_scores(0.55, 0.55), [11])[0, 91:], 0)


def test_p03_reset_clears_both_hand_states_and_both_counters():
    gate = _gate()
    gate(_scores(), [11])
    gate(_scores(0.8, 0.4), [11])
    assert _counts(gate) == (4, 1)
    gate.reset()
    assert _counts(gate) == (0, 0)
    np.testing.assert_array_equal(gate(_scores(0.55, 0.55), [11])[0, 91:], 0)
    assert _counts(gate) == (2, 2)


@pytest.mark.parametrize(
    ("on", "off"), [(0.5, 0.5), (0.49, 0.5), (0.0, 0.0), (-2.0, -1.0), (2.0, 3.0)]
)
def test_p03_invalid_threshold_order_raises_value_error(on, off):
    # Resolve the class before the exception context; a missing symbol is not validation.
    cls = type(_gate())
    with pytest.raises(ValueError, match=r"(?s).*"):
        cls(on=on, off=off)


@pytest.mark.parametrize("seed", [210103, 210104])
def test_p03_generated_row_permutation_key_renaming_and_reset_properties(seed):
    canonical, permuted = _gate(), _gate()
    rng = np.random.default_rng(seed)
    population = np.arange(1, 7)
    visited = set()
    for frame in range(80):
        if frame % 19 == 0:
            canonical.reset()
            permuted.reset()
        live = rng.choice(population, size=int(rng.integers(1, 7)), replace=False)
        renamed = [f"key-{key}" for key in live]
        canonical.prune(live.tolist())
        permuted.prune(renamed)
        scores = rng.choice([0, 0.4, 0.55, 0.9, np.nan, np.inf], size=(len(live), 133))
        order = rng.permutation(len(live))
        before = scores.copy()
        expected = canonical(scores, live.tolist())
        actual = permuted(scores[order], [renamed[index] for index in order])
        np.testing.assert_array_equal(actual, expected[order])
        np.testing.assert_array_equal(scores, before)
        assert _counts(canonical) == _counts(permuted)
        visited.update(live)
    assert len(visited) == len(population) > 0
    canonical.reset()
    fresh = _gate()
    probe = _scores(0.55, 0.55, rows=len(population))
    np.testing.assert_array_equal(
        canonical(probe, population.tolist()), fresh(probe, population.tolist())
    )


def test_p04_counters_are_hand_frames_not_points_rows_or_absent_hands():
    gate = _gate()
    first = np.vstack([_scores(0.8, 0.4), _scores(0, 0)])
    first[1, 91:100] = 1.0
    first[1, 112:122] = 0.75
    gate(first, [11, 29])
    assert _counts(gate) == (3, 1)
    second = np.vstack([_scores(0.55, 0.55), _scores(0, 0.8)])
    second[1, 91:100] = 0.9
    gate(second, [29, 11])
    assert _counts(gate) == (6, 2)
    gate(np.empty((0, 133)), [])
    gate(np.ones((3, 17)), [11, 29, 37])
    gate.prune([])
    assert _counts(gate) == (6, 2)


def _points(rows=1):
    points = np.full((rows, 133, 2), 20.0)
    # Separated, nondegenerate hands keep duplicate-hand hygiene out of this witness.
    points[:, _HANDS[0], 0] = np.linspace(3, 13, 21)
    points[:, _HANDS[0], 1] = np.linspace(10, 15, 21)
    points[:, _HANDS[1], 0] = np.linspace(43, 53, 21)
    points[:, _HANDS[1], 1] = np.linspace(25, 30, 21)
    return points


class _Tracker:
    def __init__(self, outputs, keys):
        self.outputs, self.keys = outputs, keys
        self.index = 0
        self.last_track_ids = []

    def reset(self):
        self.index = 0
        self.last_track_ids = []

    def __call__(self, _frame):
        points, scores = self.outputs[self.index]
        self.last_track_ids = self.keys[self.index]
        self.index += 1
        return points.copy(), scores.copy()


class _KeyedSmoother:
    def __init__(self, keys, live=None):
        self.keys = keys
        self.live = live if live is not None else keys
        self.index = -1

    def reset(self):
        self.index = -1

    def __call__(self, points, scores, _timestamp, **_kwargs):
        self.index += 1
        return points, scores

    def output_track_keys(self):
        return self.keys[self.index]

    def live_track_keys(self):
        return self.live[self.index]


def _pipeline(
    root,
    monkeypatch,
    outputs,
    *,
    gate=_OMITTED,
    smoother=None,
    keys=None,
    tracker_ids=True,
    bone_smoother=None,
    single_subject=False,
    empty_expected=False,
):
    from test_m2u92_keypoint_hygiene import _Capture

    run = importlib.import_module("pose_estimation.run")
    export = importlib.import_module("pose_estimation.export")
    root.mkdir(parents=True, exist_ok=True)
    source = root / "synthetic.avi"
    source.touch()
    capture = _Capture([(48, 64)] * len(outputs))
    keys = keys if keys is not None else [list(range(len(points))) for points, _ in outputs]
    tracker = _Tracker(outputs, keys)
    monkeypatch.setattr(run, "open_capture", lambda *_args, **_kwargs: capture)
    before = [(points.copy(), scores.copy()) for points, scores in outputs]
    kwargs = {} if gate is _OMITTED else {"hand_gate": gate}
    csv_path, diag_path = root / "landmarks.csv", root / "diagnostics.csv"
    run.process_source(
        SimpleNamespace(
            tracking="body", headless=True, single_subject=single_subject, max_frames=0
        ),
        tracker if tracker_ids else lambda frame: tracker(frame),
        str(source),
        draw_skeleton=None,
        smoother=smoother,
        bone_smoother=bone_smoother,
        output_csv=csv_path,
        output_diag=diag_path,
        video_name="synthetic-event/cam-a",
        **kwargs,
    )
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == export.make_csv_header("body")
        assert len(reader.fieldnames or ()) == 304
        rows = list(reader)
    with diag_path.open(newline="") as stream:
        diagnostics = list(csv.DictReader(stream))
    assert len(diagnostics) == 1
    assert capture.released
    assert tracker.index == len(outputs)
    assert (len(outputs) == 0) is empty_expected
    for (points, scores), (old_points, old_scores) in zip(outputs, before, strict=True):
        np.testing.assert_array_equal(points, old_points)
        np.testing.assert_array_equal(scores, old_scores)
    return rows, diagnostics[0], csv_path.read_bytes()


def _hand_values(row, side, suffix):
    columns = [f"{side}_hand_{index}_{suffix}" for index in range(21)]
    assert len(columns) == 21
    assert set(columns) <= row.keys()
    return [row[column] for column in columns]


def _assert_absent(row, side):
    np.testing.assert_array_equal([float(value) for value in _hand_values(row, side, "conf")], 0)
    assert _hand_values(row, side, "x") == [""] * 21
    assert _hand_values(row, side, "y") == [""] * 21
    assert _hand_values(row, side, "z") == [""] * 21


def test_p05_gated_csv_loses_coordinates_admitted_values_and_body_stay_exact(tmp_path, monkeypatch):
    outputs = [(_points(), _scores(0.6, 0.8))]
    baseline, _, _ = _pipeline(tmp_path / "off", monkeypatch, outputs)
    gate = _gate()
    actual, diag, _ = _pipeline(tmp_path / "on", monkeypatch, outputs, gate=gate)
    assert len(actual) == len(baseline) == 1
    _assert_absent(actual[0], "left")
    # A01: MediaPipe body points 17-22 derive from the hand's COCO points, so a gated
    # hand's body proxies export at visibility 0 too.
    proxies = [f"body_left_{name}_vis" for name in ("pinky", "index", "thumb")]
    assert [actual[0][column] for column in proxies] == ["0.0"] * 3
    preserved = [
        column
        for column in baseline[0]
        if not column.startswith("left_hand_") and column not in proxies
    ]
    assert len(preserved) > 0
    assert {column: actual[0][column] for column in preserved} == {
        column: baseline[0][column] for column in preserved
    }
    assert tuple(int(diag[field]) for field in _COUNTERS) == _counts(gate) == (2, 1)


def test_p05_nc2_gate_after_real_smoother_preserves_admitted_score(tmp_path, monkeypatch):
    from pose_estimation.rtmlib_smoothing import KeypointSmoother

    outputs = [(_points(), _scores(level, 0.9)) for level in (0.6, 0.9, 0.9, 0.9, 0.9, 0.9)]
    baseline, _, _ = _pipeline(
        tmp_path / "off", monkeypatch, outputs, smoother=KeypointSmoother(min_track_age=1)
    )
    assert len(baseline) == len(outputs)
    assert min(map(float, _hand_values(baseline[-1], "left", "conf"))) >= 0.65
    gate = _gate()
    actual, _, _ = _pipeline(
        tmp_path / "on", monkeypatch, outputs, gate=gate, smoother=KeypointSmoother(min_track_age=1)
    )
    assert len(actual) == len(baseline)
    _assert_absent(actual[0], "left")
    assert actual[-1] == baseline[-1], "N02: pre-smoother zeroing contaminates EMA scores"


@pytest.mark.parametrize("key_source", ["smoother", "tracker", "row"])
def test_p05_nc3_pipeline_key_source_survives_swapped_rows(tmp_path, monkeypatch, key_source):
    gate = _gate()
    first = np.vstack([_scores(0.8, 0.4), _scores(0.4, 0.8)])
    second = _scores(0.55, 0.55, rows=2)
    smoother = _KeyedSmoother([[11, 29], [29, 11]]) if key_source == "smoother" else None
    keys = [[29, 11], [11, 29]] if key_source == "tracker" else [[101, 103], [101, 103]]
    rows, _, _ = _pipeline(
        tmp_path,
        monkeypatch,
        [(_points(2), first), (_points(2), second)],
        gate=gate,
        smoother=smoother,
        keys=keys,
        tracker_ids=key_source != "row",
    )
    assert len(rows) == 4
    off_sides = ("right", "left") if key_source == "row" else ("left", "right")
    for row, absent in zip(rows[2:], off_sides, strict=True):
        _assert_absent(row, absent)
        present = "right" if absent == "left" else "left"
        np.testing.assert_array_equal(list(map(float, _hand_values(row, present, "conf"))), 0.55)


@pytest.mark.parametrize("key_source", ["tracker", "smoother-kept", "smoother-pruned"])
def test_p05_prunes_reborn_keys_but_keeps_smoother_live_keys(tmp_path, monkeypatch, key_source):
    gate = _gate()
    outputs = [
        (_points(), _scores()),
        (_points(0), _scores(rows=0)),
        (_points(), _scores(0.55, 0.55)),
    ]
    live = [[7], [7], [7]] if key_source == "smoother-kept" else [[7], [], [7]]
    smoother = _KeyedSmoother([[7], [], [7]], live) if key_source != "tracker" else None
    rows, _, _ = _pipeline(
        tmp_path, monkeypatch, outputs, gate=gate, smoother=smoother, keys=[[7], [], [7]]
    )
    assert len(rows) == 2
    for side in ("left", "right"):
        if key_source == "smoother-kept":
            np.testing.assert_array_equal(
                list(map(float, _hand_values(rows[-1], side, "conf"))), 0.55
            )
        else:
            _assert_absent(rows[-1], side)


def test_p05_process_source_resets_a_reused_gate_and_diagnostic_counters(tmp_path, monkeypatch):
    gate = _gate()
    first, diag, _ = _pipeline(tmp_path / "first", monkeypatch, [(_points(), _scores())], gate=gate)
    assert len(first) == 1
    assert _counts(gate) == (2, 0)
    assert tuple(int(diag[field]) for field in _COUNTERS) == (2, 0)
    second, diag, _ = _pipeline(
        tmp_path / "second", monkeypatch, [(_points(), _scores(0.55, 0.55))], gate=gate
    )
    assert len(second) == 1
    _assert_absent(second[0], "left")
    _assert_absent(second[0], "right")
    assert tuple(int(diag[field]) for field in _COUNTERS) == _counts(gate) == (2, 2)


def test_p05_zero_frame_source_resets_gate_and_publishes_zero_counters(tmp_path, monkeypatch):
    baseline, _, _ = _pipeline(tmp_path / "off", monkeypatch, [], empty_expected=True)
    assert baseline == []
    gate = _gate()
    gate(_scores(0.6, 0.8), [7])
    assert _counts(gate) == (2, 1)
    rows, diag, _ = _pipeline(tmp_path / "on", monkeypatch, [], gate=gate, empty_expected=True)
    assert rows == []
    assert tuple(int(diag[field]) for field in _COUNTERS) == _counts(gate) == (0, 0)


def test_p05_ungated_baseline_bytes_and_zero_diagnostics(tmp_path, monkeypatch):
    run = importlib.import_module("pose_estimation.run")
    outputs = [
        (_points(), _scores(left, right)) for left, right in ((0.6, 0.8), (0.4, 0.9), (0.9, 0.55))
    ]
    rows, diag, payload = _pipeline(tmp_path, monkeypatch, outputs)
    assert len(rows) == 3
    # Frozen synthetic output from the unfixed contract baseline, 1e335a4.
    assert (
        hashlib.sha256(payload).hexdigest()
        == "16f52aac63bbe34172b3fe3ed14e99ebac347c9a9bdc44009469225e2ba09cba"
    )
    assert set(_COUNTERS) <= set(run.SOURCE_DIAGNOSTIC_FIELDS), (
        "P05: firing counters absent from diagnostic schema"
    )
    assert tuple(int(diag[field]) for field in _COUNTERS) == (0, 0)
    explicit, explicit_diag, explicit_payload = _pipeline(
        tmp_path / "explicit-none", monkeypatch, outputs, gate=None
    )
    assert explicit_payload == payload
    assert explicit == rows
    assert tuple(int(explicit_diag[field]) for field in _COUNTERS) == (0, 0)


@pytest.mark.parametrize("enabled", [False, True])
def test_p06_run_main_constructs_only_on_flag_and_forwards_through_dispatch(
    tmp_path, monkeypatch, enabled
):
    from test_corpus_run_preconditions import _make_session
    from test_m2u91_subject_tracker import _solution_factory

    run = importlib.import_module("pose_estimation.run")
    cls = type(_gate())
    session = _make_session(tmp_path, cameras=("cam-a", "cam-b"))
    calls = []
    built = []

    initialize = cls.__init__

    def constructor(instance, *args, **kwargs):
        initialize(instance, *args, **kwargs)
        built.append(instance)

    monkeypatch.setattr(cls, "__init__", constructor)
    monkeypatch.setattr(
        run, "SplitDeviceSolution", _solution_factory(lambda _index: np.empty((0, 4)))
    )
    monkeypatch.setattr(run, "resolve_cli_sessions", lambda *_args, **_kwargs: [session])
    monkeypatch.setattr(run, "process_source", lambda *args, **kwargs: calls.append(kwargs) or [])
    options = ["--hand-gate"] if enabled else []
    parsed = run.parse_args(options)
    assert parsed.hand_gate is enabled
    run.main(
        [
            "--session-dir",
            str(session.directory),
            "--output-dir",
            str(tmp_path / "out"),
            "--headless",
            "--backend",
            "onnxruntime",
            *options,
        ]
    )
    assert len(calls) == 2
    assert len(built) == int(enabled)
    assert all(call.get("hand_gate") is (built[0] if enabled else None) for call in calls)


def test_p05_gate_follows_bones_and_precedes_subject_selection(tmp_path, monkeypatch):
    run = importlib.import_module("pose_estimation.run")
    events = []

    class Smoother(_KeyedSmoother):
        def __call__(self, points, scores, _timestamp, **_kwargs):
            events.append("smooth")
            return super().__call__(points, scores, _timestamp, **_kwargs)

    class Bones:
        def reset(self):
            pass

        def prune(self, _keys):
            pass

        def update(self, _key, points, **_kwargs):
            events.append("bone")
            return points, 0.0

    class ObservedGate:
        def __getattr__(self, name):
            return getattr(gate, name)

        def __call__(self, scores, keys):
            events.append("gate")
            return gate(scores, keys)

    select = run.filter_single_subject

    def selected(*args, **kwargs):
        events.append("select")
        return select(*args, **kwargs)

    monkeypatch.setattr(run, "filter_single_subject", selected)
    baseline, _, _ = _pipeline(
        tmp_path / "off",
        monkeypatch,
        [(_points(2), _scores(0.6, 0.8, rows=2))],
        smoother=Smoother([[11, 29]]),
        bone_smoother=Bones(),
        single_subject=True,
    )
    assert len(baseline) == 1
    assert events == ["smooth", "bone", "bone", "select"]
    events.clear()
    gate = _gate()
    rows, diag, _ = _pipeline(
        tmp_path / "on",
        monkeypatch,
        [(_points(2), _scores(0.6, 0.8, rows=2))],
        gate=ObservedGate(),
        smoother=Smoother([[11, 29]]),
        bone_smoother=Bones(),
        single_subject=True,
    )
    assert len(rows) == 1
    assert events == ["smooth", "bone", "bone", "gate", "select"]
    assert tuple(int(diag[field]) for field in _COUNTERS) == (4, 2)


def _driver(name):
    return runpy.run_path(str(_ROOT / "scripts" / name), run_name="_m2u101_driver")[
        "main"
    ].__globals__


def test_p06_pose_config_fields_include_hand_gate():
    from pose_estimation.corpus_run import POSE_CONFIG_FIELDS

    assert "hand_gate" in POSE_CONFIG_FIELDS


@pytest.mark.parametrize("name", _DRIVERS)
def test_p06_nc4_driver_defaults_on_and_declares_generation_identity(name, monkeypatch):
    driver = _driver(name)
    monkeypatch.setattr(sys, "argv", [name])
    args = driver["_parse_args"]()
    assert getattr(args, "hand_gate", None) is True, "N04: driver must default hand gate on"
    assert "hand_gate" in driver["REPORT_FIELDS"]
    # M2.10.1 introduced `hand_gate` at corpus v5 / pilot v4; later units bump past it.
    assert int(driver["GENERATOR_VERSION"].lstrip("v")) >= (5 if name == "corpus_run_2d.py" else 4)


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize("enabled", [False, True])
def test_p06_driver_boolean_flags_forward_effective_setting(name, enabled, tmp_path, monkeypatch):
    from test_m2u91_subject_tracker import _driver_args

    driver = _driver(name)
    monkeypatch.setattr(sys, "argv", [name, "--hand-gate" if enabled else "--no-hand-gate"])
    parsed = driver["_parse_args"]()
    assert parsed.hand_gate is enabled
    args = _driver_args(driver, monkeypatch, tmp_path)
    args.hand_gate = enabled
    commands = []
    monkeypatch.setattr(
        driver["subprocess"], "call", lambda command, **_kwargs: commands.append(command) or 0
    )
    if name == "corpus_run_2d.py":
        driver["_attempt_event"]("synthetic-event", args, tmp_path / "logs")
    else:
        driver["_run_event"](
            args.sessions / "synthetic-event", args.out, tmp_path / "run.log", args
        )
    commands = [command for command in commands if "pose_estimation.run" in command]
    assert len(commands) == 1
    assert ("--hand-gate" in commands[0]) is enabled
    run = importlib.import_module("pose_estimation.run")
    assert (
        run.parse_args(commands[0][commands[0].index("pose_estimation.run") + 1 :]).hand_gate
        is enabled
    )


def _generation(name, tmp_path, monkeypatch, *, enabled):
    from test_m2u91_subject_tracker import _driver_args
    from test_m2u95_body_drop import _csv

    driver = _driver(name)
    args = _driver_args(driver, monkeypatch, tmp_path)
    args.hand_gate = enabled
    event_id, camera = "synthetic-event", "cam-a"
    video = f"{event_id}/{camera}"
    _csv(
        args.inventory / "assets.csv",
        ("asset_id", "disposition", "reported_rotation_deg", "reported_frame_count"),
        [
            {
                "asset_id": "synthetic-asset",
                "disposition": "canonical",
                "reported_rotation_deg": "0",
                "reported_frame_count": "3",
            }
        ],
    )
    _csv(
        args.qualification / "assets_qc.csv",
        ("asset_id", "codec", "device_config", "pts_monotonic"),
        [
            {
                "asset_id": "synthetic-asset",
                "codec": "h264",
                "device_config": "synthetic",
                "pts_monotonic": "1",
            }
        ],
    )
    _csv(
        args.sessions / "placements.csv",
        ("asset_id", "event_id", "camera_name", "placement"),
        [
            {
                "asset_id": "synthetic-asset",
                "event_id": event_id,
                "camera_name": camera,
                "placement": "placed",
            }
        ],
    )
    (args.sessions / "generation.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(driver, "_parse_args", lambda: args)
    monkeypatch.setitem(driver, "validate_generation", lambda *_args, **_kwargs: {})
    if name == "pilot_corpus_run.py":
        args.min_assets = 1
        monkeypatch.setitem(driver, "_assert_sources_validated", lambda *_args: None)
        monkeypatch.setitem(
            driver,
            "_guard_verdicts",
            lambda *_args: {"events_probed": 1, "default_output_refused": 1},
        )
    pilot = driver["pilot"].__dict__ if name == "corpus_run_2d.py" else driver
    qc_header, _ = pilot["_r_constants"]()
    generated = []
    event_out = args.out / event_id
    run = importlib.import_module("pose_estimation.run")

    def subprocess_call(command, **_kwargs):
        if "pose_estimation.run" in command:
            generated.append(command.copy())
            _csv(
                event_out / f"{camera}.csv",
                ("video", "person_idx", "frame_idx"),
                [{"video": video, "person_idx": "0", "frame_idx": "0"}],
            )
            diagnostic = dict.fromkeys(run.SOURCE_DIAGNOSTIC_FIELDS, "0")
            diagnostic.update(
                video=video,
                n_frames_decoded="3",
                pts_accepted="3",
                latency_ms_mean="1",
                latency_ms_p95="1",
            )
            _csv(event_out / f"{camera}_diag.csv", tuple(diagnostic), [diagnostic])
        else:
            assert command[0] == "Rscript"
            _csv(event_out / f"{camera}_clinical_group_qc.csv", qc_header, [])
            _csv(
                event_out / f"{camera}_clinical_windows.csv",
                ("video", "person_idx"),
                [{"video": video, "person_idx": "0"}],
            )
        return 0

    monkeypatch.setattr(driver["subprocess"], "call", subprocess_call)
    assert driver["main"]() == 0
    assert len(generated) == 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["population"]["events"] == 1
    assert report["verdicts"]
    assert all(report["verdicts"].values())
    return driver, args, generated, event_out


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize("enabled", [False, True])
def test_p06_report_configuration_and_pose_marker_publish_setting(
    name, enabled, tmp_path, monkeypatch
):
    driver, args, _, event_out = _generation(name, tmp_path, monkeypatch, enabled=enabled)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    marker = json.loads((event_out / "pose_config.json").read_text(encoding="utf-8"))
    assert report["configuration"].get("hand_gate") is enabled
    assert marker.get("hand_gate") is enabled
    assert "hand_gate" in driver["REPORT_FIELDS"]


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize("enabled", [False, True])
def test_p06_same_setting_reuses_a_complete_generation(name, enabled, tmp_path, monkeypatch):
    driver, args, generated, event_out = _generation(name, tmp_path, monkeypatch, enabled=enabled)
    marker = event_out / "pose_config.json"
    before = marker.read_bytes()
    assert json.loads(before).get("hand_gate") is enabled
    if name == "pilot_corpus_run.py":
        args.reuse_run = True
    assert driver["main"]() == 0
    assert len(generated) == 1
    assert marker.read_bytes() == before


def test_p06_disabling_gate_also_makes_event_due(tmp_path, monkeypatch):
    driver, args, generated, event_out = _generation(
        "corpus_run_2d.py", tmp_path, monkeypatch, enabled=True
    )
    args.hand_gate = False
    assert driver["main"]() == 0
    assert len(generated) == 2
    assert "--hand-gate" not in generated[-1]
    assert (
        json.loads((event_out / "pose_config.json").read_text(encoding="utf-8"))["hand_gate"]
        is False
    )


@pytest.mark.parametrize("old", [False, "missing"])
def test_p06_ungated_complete_event_is_due_on_resume(tmp_path, monkeypatch, old):
    driver, args, generated, event_out = _generation(
        "corpus_run_2d.py", tmp_path, monkeypatch, enabled=False
    )
    marker_path = event_out / "pose_config.json"
    if old == "missing":
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.pop("hand_gate", None)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    args.hand_gate = True
    assert driver["main"]() == 0
    assert len(generated) == 2, "P06: complete ungated event reused after gate enabled"
    assert "--hand-gate" in generated[-1]
    assert json.loads(marker_path.read_text(encoding="utf-8"))["hand_gate"] is True


@pytest.mark.parametrize(
    ("name", "mode"), [("corpus_run_2d.py", "analyse_only"), ("pilot_corpus_run.py", "reuse_run")]
)
@pytest.mark.parametrize("old", [False, "missing"])
def test_p06_ungated_generation_refuses_analysis_or_reuse(
    name, mode, old, tmp_path, monkeypatch, capsys
):
    driver, args, generated, event_out = _generation(name, tmp_path, monkeypatch, enabled=False)
    marker_path = event_out / "pose_config.json"
    if old == "missing":
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.pop("hand_gate", None)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in event_out.glob("*") if path.is_file()}
    assert {"cam-a.csv", "pose_config.json"} <= before.keys()
    args.hand_gate = True
    setattr(args, mode, True)
    refusal = driver.get("RunError", driver.get("PilotError"))
    assert refusal is not None
    capsys.readouterr()
    try:
        result = driver["main"]()
    except refusal as error:
        message = str(error)
    else:
        captured = capsys.readouterr()
        message = captured.out + captured.err
        assert result != 0, f"P06: {mode} accepted an ungated generation"
    assert any(
        term in message.lower()
        for term in ("hand", "config", "generation", "provenance", "mismatch")
    )
    assert len(generated) == 1
    for filename in ("cam-a.csv", "pose_config.json"):
        assert (event_out / filename).read_bytes() == before[filename]
