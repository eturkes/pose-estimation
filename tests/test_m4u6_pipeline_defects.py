"""M4.6 P01-P05: subject-stream hand gate, export hygiene, duplicate census, build identity."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np

from test_m2u101_hand_gate import (
    _HANDS,
    _gate,
    _hand_values,
    _KeyedSmoother,
    _pipeline,
    _points,
    _scores,
)

_EMPTY = (np.zeros((0, 133, 2)), np.zeros((0, 133)))


def _two_rows(subject_hands, other_hands):
    """Row 0 = the subject's track, row 1 = a transient track (same body, other hand level)."""
    return _points(2), np.vstack(
        [_scores(subject_hands, subject_hands), _scores(other_hands, other_hands)]
    )


def _present(row, side="left"):
    return all(value != "" for value in _hand_values(row, side, "x"))


def _transient_then_return(level_back):
    outputs = (
        [(_points(), _scores(0.8, 0.8))] * 3
        + [_two_rows(0.0, 0.8)] * 3
        + [(_points(), _scores(level_back, level_back))] * 3
    )
    keys = [[7]] * 3 + [[7, 9]] * 3 + [[7]] * 3
    return outputs, keys


def test_p01_transient_track_cannot_switch_the_subject_hand_off(tmp_path, monkeypatch):
    outputs, keys = _transient_then_return(0.6)
    rows, _, _ = _pipeline(
        tmp_path, monkeypatch, outputs, gate=_gate(), smoother=_KeyedSmoother(keys),
        keys=keys, single_subject=True,
    )  # fmt: skip
    assert len(rows) == 9
    # Back on the subject's own track at a level inside the hysteresis band, the exported hand
    # stays admitted: the stream's state was on when the transient track exported it.
    assert [_present(r) for r in rows] == [True] * 9


def test_p01_a_hand_below_off_still_switches_off(tmp_path, monkeypatch):
    outputs, keys = _transient_then_return(0.4)
    rows, _, _ = _pipeline(
        tmp_path, monkeypatch, outputs, gate=_gate(), smoother=_KeyedSmoother(keys),
        keys=keys, single_subject=True,
    )  # fmt: skip
    assert [_present(r) for r in rows] == [True] * 6 + [False] * 3


def test_p01_a_frame_without_an_exported_row_resets_the_stream(tmp_path, monkeypatch):
    outputs = [(_points(), _scores(0.8, 0.8))] * 2 + [_EMPTY] + [(_points(), _scores(0.6, 0.6))] * 2
    keys = [[7], [7], [], [7], [7]]
    rows, _, _ = _pipeline(
        tmp_path, monkeypatch, outputs, gate=_gate(), smoother=_KeyedSmoother(keys),
        keys=keys, single_subject=True,
    )  # fmt: skip
    # 0.6 cannot turn a reset hand on (needs 0.65); without the reset it would stay on.
    assert [_present(r) for r in rows] == [True, True, False, False]


def test_p02_multi_person_export_keeps_per_track_keys(tmp_path, monkeypatch):
    outputs, keys = _transient_then_return(0.6)
    rows, _, _ = _pipeline(
        tmp_path, monkeypatch, outputs, gate=_gate(), smoother=_KeyedSmoother(keys),
        keys=keys, single_subject=False,
    )  # fmt: skip
    subject = [r for r in rows if r["person_idx"] == "0"]
    # Per-track state: track 7 saw its own handless rows, so 0.6 cannot readmit its hands.
    assert [_present(r) for r in subject][-3:] == [False] * 3


class _Mover(_KeyedSmoother):
    """A post-hygiene stage that drags the left wrist + one left-hand point out of the frame."""

    def __call__(self, points, scores, _timestamp, **_kwargs):
        points, scores = super().__call__(points, scores, _timestamp)
        points = np.array(points, copy=True)
        points[:, 9] = (70.0, 20.0)  # left wrist, frame is 64 x 48
        points[:, _HANDS[0].start] = (-3.0, 12.0)
        return points, scores


def test_p03_a_point_moved_out_of_frame_after_hygiene_exports_no_observation(tmp_path, monkeypatch):
    rows, diag, _ = _pipeline(
        tmp_path, monkeypatch, [(_points(), _scores())], smoother=_Mover([[0]]), keys=[[0]]
    )
    assert len(rows) == 1
    row = rows[0]
    assert float(row["body_left_wrist_vis"]) == 0.0
    assert float(row["body_right_wrist_vis"]) > 0.0  # in-frame neighbour untouched
    # Per point, only the evidence drops: conf 0, coordinates as computed (the writer blanks a
    # hand's coordinates only when the whole hand is absent).
    assert float(row["left_hand_0_conf"]) == 0.0
    assert float(row["left_hand_1_conf"]) > 0.0
    assert int(diag["exported_out_of_frame_zeroed"]) == 2


def test_p03_in_frame_export_is_unchanged_and_counts_zero(tmp_path, monkeypatch):
    rows, diag, _ = _pipeline(tmp_path, monkeypatch, [(_points(), _scores())])
    assert len(rows) == 1
    assert int(diag["exported_out_of_frame_zeroed"]) == 0


def _duplicate_hands():
    points = _points()
    points[:, _HANDS[0]] = points[:, _HANDS[1]]  # the hidden hand drawn on the visible one
    scores = _scores(0.3, 0.9)
    return points, scores


def test_p04_duplicate_collector_names_each_firing_person_once():
    hygiene = importlib.import_module("pose_estimation.keypoint_hygiene")
    points, scores = _duplicate_hands()
    fired: list[int] = []
    with_collector = hygiene.suppress_duplicate_hand(points, scores, fired=fired)
    np.testing.assert_array_equal(with_collector, hygiene.suppress_duplicate_hand(points, scores))
    assert fired == [0]
    assert not with_collector[0, _HANDS[0]].any()
    clean: list[int] = []
    hygiene.apply_hygiene(_points(), _scores(), 64, 48, fired=clean)
    assert clean == []


def test_p04_diagnostics_publish_the_duplicate_census(tmp_path, monkeypatch):
    _, diag, _ = _pipeline(tmp_path / "dup", monkeypatch, [_duplicate_hands()] * 3)
    assert int(diag["duplicate_hand_fired"]) == 3
    _, diag, _ = _pipeline(tmp_path / "clean", monkeypatch, [(_points(), _scores())] * 3)
    assert int(diag["duplicate_hand_fired"]) == 0


def test_p05_build_is_part_of_the_generation(tmp_path):
    import openvino

    corpus = importlib.import_module("pose_estimation.corpus_run")
    assert "openvino_build" in corpus.POSE_CONFIG_FIELDS
    # The build comes from the process, never from an argument a caller could leave stale.
    args = SimpleNamespace(**{name: f"value-{name}" for name in corpus.POSE_CONFIG_FIELDS})
    config = corpus.pose_config(args)
    corpus.write_pose_config(tmp_path, config)
    recorded = json.loads((tmp_path / corpus.POSE_CONFIG_FILENAME).read_text())
    assert recorded["openvino_build"] == openvino.get_version()
    foreign = dict(config, openvino_build="2026.0.0-foreign")
    assert corpus.read_pose_config(tmp_path) != foreign  # a resume under another build is due


def test_p05_drivers_record_the_running_build():
    import openvino

    corpus = importlib.import_module("pose_estimation.corpus_run")
    assert corpus.openvino_build() == openvino.get_version()
