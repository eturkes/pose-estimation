"""Test CSV export from the rtmlib pipeline (process_source)."""

import csv
import pathlib
import types

import cv2
import numpy as np
import pytest

from pose_estimation.export import make_csv_header
from pose_estimation.processing import TRACKING_BODY, TRACKING_HANDS, TRACKING_HANDS_ARMS
from pose_estimation.run import process_source

_FRAME_SIZE = (160, 120)  # w, h
_FPS = 10.0
_N_FRAMES = 5


def _write_video(path: pathlib.Path, n_frames: int = _N_FRAMES) -> bool:
    """Write a tiny MJPG/AVI video."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, _FPS, _FRAME_SIZE)
    if not writer.isOpened():
        return False
    try:
        for i in range(n_frames):
            frame = np.full(
                (_FRAME_SIZE[1], _FRAME_SIZE[0], 3),
                fill_value=(100 + i * 10) % 256,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def _make_args(tracking="hands-arms", headless=True, single_subject=False):
    """Create a minimal args namespace for process_source."""
    return types.SimpleNamespace(
        tracking=tracking,
        headless=headless,
        single_subject=single_subject,
        max_frames=0,
    )


def _mock_tracker_133(frame):
    """Return 1 person with 133 keypoints in pixel range."""
    rng = np.random.default_rng(0)
    kps = rng.uniform(10, 100, (1, 133, 2)).astype(np.float32)
    scores = rng.uniform(0.5, 1.0, (1, 133)).astype(np.float32)
    return kps, scores


def _mock_tracker_17(frame):
    """Return 1 person with 17 keypoints."""
    rng = np.random.default_rng(0)
    kps = rng.uniform(10, 100, (1, 17, 2)).astype(np.float32)
    scores = rng.uniform(0.5, 1.0, (1, 17)).astype(np.float32)
    return kps, scores


class _ResettableTracker:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def __call__(self, frame):
        return _mock_tracker_133(frame)


class _RecordingSmoother:
    def __init__(self):
        self.reset_calls = 0
        self.timestamps = []

    def reset(self):
        self.reset_calls += 1

    def __call__(self, keypoints, scores, timestamp):
        self.timestamps.append(timestamp)
        return keypoints, scores


class _RecordingBoneSmoother:
    def __init__(self):
        self.reset_calls = 0
        self.pruned = []

    def reset(self):
        self.reset_calls += 1

    def update(self, _track_key, keypoints):
        return keypoints, 0.0

    def prune(self, active_keys):
        self.pruned.append(list(active_keys))


@pytest.fixture
def video_path(tmp_path):
    vpath = tmp_path / "test_video.avi"
    if not _write_video(vpath):
        pytest.skip("MJPG codec unavailable")
    return vpath


class TestCSVExport133HandsArms:
    def test_csv_created(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="hands-arms")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )
        assert csv_path.exists()

    def test_csv_columns_match_header(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="hands-arms")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )
        expected = make_csv_header(TRACKING_HANDS_ARMS)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert list(reader.fieldnames) == expected

    def test_hands_mode_writes_per_keypoint_confidence(self, video_path, tmp_path):
        csv_path = tmp_path / "hands.csv"
        keypoints, _scores = _mock_tracker_133(np.empty((1, 1, 3)))
        scores = np.full((1, 133), 0.8, dtype=np.float32)
        scores[0, 91:112] = np.linspace(0.2, 0.4, 21)
        scores[0, 112:133] = np.linspace(0.6, 0.9, 21)

        process_source(
            _make_args(tracking=TRACKING_HANDS),
            lambda _frame: (keypoints, scores),
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )

        with csv_path.open() as fh:
            row = next(csv.DictReader(fh))
        assert float(row["left_hand_0_conf"]) == pytest.approx(scores[0, 91])
        assert float(row["right_hand_20_conf"]) == pytest.approx(scores[0, 132])

    def test_csv_row_count(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="hands-arms")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )
        with csv_path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        # header + 1 row per frame (1 person per frame)
        assert len(rows) == 1 + _N_FRAMES

    def test_csv_video_name_default(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="hands-arms")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["video"] == "test_video.avi"

    def test_csv_video_name_override(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="hands-arms")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
            video_name="session1/cam1",
        )
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["video"] == "session1/cam1"

    def test_csv_coordinates_normalized(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="hands-arms")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # Arm coordinates should be in [0, 1] range (synth data is 10-100 in 160x120 frame)
        x = float(row["arm_left_shoulder_x"])
        assert 0.0 <= x <= 1.0


class TestCSVExport133Body:
    def test_csv_columns_body_mode(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="body")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )
        expected = make_csv_header(TRACKING_BODY)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert list(reader.fieldnames) == expected


class TestCSVExport17Body:
    def test_csv_columns_17kp(self, video_path, tmp_path):
        csv_path = tmp_path / "out.csv"
        args = _make_args(tracking="body")
        process_source(
            args,
            _mock_tracker_17,
            str(video_path),
            draw_skeleton=None,
            output_csv=str(csv_path),
        )
        expected = make_csv_header(TRACKING_BODY)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert list(reader.fieldnames) == expected


class TestNoCSVWhenDisabled:
    def test_no_csv_without_flag(self, video_path, tmp_path):
        """process_source without output_csv produces no file."""
        args = _make_args(tracking="hands-arms")
        process_source(
            args,
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
        )
        csv_files = list(tmp_path.glob("*.csv"))
        # Only the video file should be in tmp_path
        assert not any(f.name.endswith(".csv") for f in csv_files)


def test_media_timestamp_drives_smoothing_and_csv_independent_of_inference_delay(
    video_path, tmp_path, monkeypatch
):
    csv_path = tmp_path / "timed.csv"
    tracker = _ResettableTracker()
    smoother = _RecordingSmoother()
    # Two perf_counter reads per frame. Deliberately huge, irregular inference
    # durations must not leak into the file-backed smoother timebase.
    perf_ticks = iter([0.0, 10.0, 100.0, 130.0, 500.0, 501.0, 900.0, 950.0, 2000.0, 2100.0])
    monkeypatch.setattr("pose_estimation.run.time.perf_counter", lambda: next(perf_ticks))

    process_source(
        _make_args(),
        tracker,
        str(video_path),
        draw_skeleton=None,
        smoother=smoother,
        output_csv=str(csv_path),
    )

    assert tracker.reset_calls == 1
    assert smoother.reset_calls == 1
    assert smoother.timestamps == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [float(row["timestamp_sec"]) for row in rows] == pytest.approx(smoother.timestamps)
    # Both backends use zero-based decoded-source indices so sync offsets and
    # malformed-frame gaps have identical semantics.
    assert [int(row["frame_idx"]) for row in rows] == [0, 1, 2, 3, 4]


def test_pose_tracker_resets_at_every_source_boundary(video_path):
    tracker = _ResettableTracker()

    for _ in range(2):
        process_source(
            _make_args(),
            tracker,
            str(video_path),
            draw_skeleton=None,
        )

    assert tracker.reset_calls == 2


def test_bone_state_resets_at_every_source_boundary(video_path):
    bone_smoother = _RecordingBoneSmoother()

    for _ in range(2):
        process_source(
            _make_args(),
            _mock_tracker_133,
            str(video_path),
            draw_skeleton=None,
            bone_smoother=bone_smoother,
        )

    assert bone_smoother.reset_calls == 2
