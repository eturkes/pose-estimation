"""Entrypoint integration tests for source-derived frame timestamps."""

from __future__ import annotations

import csv
import importlib
import types
from typing import ClassVar

import cv2
import numpy as np
import pytest

from pose_estimation.processing import TRACKING_HANDS

main_module = importlib.import_module("pose_estimation.main")
run_module = importlib.import_module("pose_estimation.run")


class _Capture:
    def __init__(self, pos_msec, malformed=()):
        self._pos_msec = list(pos_msec)
        self._malformed = set(malformed)
        self._index = -1
        self.released = False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 10.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._pos_msec))
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._pos_msec[self._index]
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 64.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 48.0
        raise AssertionError(f"unexpected capture property: {prop}")

    def read(self):
        self._index += 1
        if self._index >= len(self._pos_msec):
            return False, None
        if self._index in self._malformed:
            return True, np.empty((0,), dtype=np.uint8)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        return True, frame

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True


class _Writer:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


class _RecordingPoseSmoother:
    instances: ClassVar[list[_RecordingPoseSmoother]] = []

    def __init__(self):
        self.timestamps = []
        self.n_hands = 0
        self.instances.append(self)

    def smooth_hands(self, hand_landmarks, timestamp, **_kwargs):
        self.timestamps.append(timestamp)
        self.n_hands = len(hand_landmarks)
        return hand_landmarks, len(hand_landmarks)

    def hand_track_ages(self):
        return [1] * self.n_hands

    def body_track_ages(self):
        return []

    def hand_observation_indices(self):
        return list(range(self.n_hands))


def test_mediapipe_uses_one_media_timestamp_for_smoothing_csv_and_diagnostics(monkeypatch):
    capture = _Capture([0.0, 100.0, 200.0])
    csv_writer = _Writer()
    diag_writer = _Writer()
    _RecordingPoseSmoother.instances.clear()

    monkeypatch.setattr(main_module, "open_capture", lambda _source: capture)
    monkeypatch.setattr(main_module, "PoseSmoother", _RecordingPoseSmoother)
    monkeypatch.setattr(
        main_module,
        "process_frame",
        lambda *_args, **_kwargs: ([], [], [], [], {"hand_diag": []}, object()),
    )
    # Irregular inference durations must not alter file-source timestamps.
    perf_ticks = iter([0.0, 50.0, 100.0, 101.0, 1000.0, 1200.0])
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: next(perf_ticks))

    main_module.process_video(
        "synthetic.mp4",
        False,
        models={},
        palm_anchors=None,
        pose_anchors=None,
        screen=None,
        csv_writer=csv_writer,
        diag_writer=diag_writer,
        tracking=TRACKING_HANDS,
        headless=True,
    )

    smoother = _RecordingPoseSmoother.instances[0]
    expected = [0.0, 0.1, 0.2]
    assert smoother.timestamps == pytest.approx(expected)
    assert [row["timestamp_sec"] for row in csv_writer.rows] == pytest.approx(expected)
    assert [row["timestamp"] for row in diag_writer.rows] == pytest.approx(expected)
    assert [row["frame_idx"] for row in csv_writer.rows] == [0, 1, 2]
    assert capture.released is True


def test_mediapipe_export_drops_carried_hands_and_remaps_observed_matches():
    carried = np.zeros((21, 3))
    observed = np.ones((21, 3))
    hands, matches = main_module._observed_hands_for_export(
        [carried, observed],
        [(0, 4, 0), (0, 5, 1)],
        {id(observed)},
    )

    assert len(hands) == 1
    assert hands[0] is observed
    assert matches == [(0, 5, 0)]


def test_mediapipe_malformed_frame_keeps_source_index_and_timestamp_gap(monkeypatch):
    capture = _Capture([0.0, 100.0, 200.0], malformed={1})
    csv_writer = _Writer()
    _RecordingPoseSmoother.instances.clear()

    monkeypatch.setattr(main_module, "open_capture", lambda _source: capture)
    monkeypatch.setattr(main_module, "PoseSmoother", _RecordingPoseSmoother)
    monkeypatch.setattr(
        main_module,
        "process_frame",
        lambda *_args, **_kwargs: ([], [], [], [], {"hand_diag": []}, object()),
    )
    perf_ticks = iter([0.0, 0.01, 0.2, 0.21])
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: next(perf_ticks))

    main_module.process_video(
        "synthetic.mp4",
        False,
        models={},
        palm_anchors=None,
        pose_anchors=None,
        screen=None,
        csv_writer=csv_writer,
        tracking=TRACKING_HANDS,
        headless=True,
    )

    assert [row["frame_idx"] for row in csv_writer.rows] == [0, 2]
    assert [row["timestamp_sec"] for row in csv_writer.rows] == pytest.approx([0.0, 0.2])
    assert _RecordingPoseSmoother.instances[0].timestamps == pytest.approx([0.0, 0.2])


def test_mediapipe_hand_presence_reaches_csv_confidence(monkeypatch):
    capture = _Capture([0.0])
    csv_writer = _Writer()
    hand = np.zeros((21, 3), dtype=np.float64)
    diagnostics = types.SimpleNamespace(
        raw_hand_handedness=[("left", 0.95)],
        raw_hand_confidences=[0.37],
    )
    _RecordingPoseSmoother.instances.clear()
    monkeypatch.setattr(main_module, "open_capture", lambda _source: capture)
    monkeypatch.setattr(main_module, "PoseSmoother", _RecordingPoseSmoother)
    monkeypatch.setattr(
        main_module,
        "process_frame",
        lambda *_args, **_kwargs: (
            [],
            [],
            [hand],
            [0.37],
            {"hand_diag": []},
            diagnostics,
        ),
    )
    perf_ticks = iter([0.0, 0.01])
    monkeypatch.setattr(main_module.time, "perf_counter", lambda: next(perf_ticks))

    main_module.process_video(
        "synthetic.mp4",
        False,
        models={},
        palm_anchors=None,
        pose_anchors=None,
        screen=None,
        csv_writer=csv_writer,
        tracking=TRACKING_HANDS,
        headless=True,
    )

    assert csv_writer.rows[0]["left_hand_0_conf"] == pytest.approx(0.37)
    assert csv_writer.rows[0]["right_hand_0_conf"] == 0.0


def test_rtmlib_malformed_frame_keeps_source_index_and_timestamp_gap(tmp_path, monkeypatch):
    capture = _Capture([0.0, 100.0, 200.0], malformed={1})
    csv_path = tmp_path / "rtm.csv"
    keypoints = np.zeros((1, 133, 2), dtype=np.float64)
    scores = np.ones((1, 133), dtype=np.float64)
    args = types.SimpleNamespace(
        tracking=TRACKING_HANDS,
        headless=True,
        single_subject=False,
        max_frames=0,
    )
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: capture)

    run_module.process_source(
        args,
        lambda _frame: (keypoints, scores),
        "synthetic.mp4",
        draw_skeleton=None,
        output_csv=str(csv_path),
    )

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [int(row["frame_idx"]) for row in rows] == [0, 2]
    assert [float(row["timestamp_sec"]) for row in rows] == pytest.approx([0.0, 0.2])
