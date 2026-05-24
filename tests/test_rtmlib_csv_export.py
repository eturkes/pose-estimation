"""Test CSV export from the rtmlib pipeline (process_source)."""

import csv
import pathlib
import types

import cv2
import numpy as np
import pytest

from pose_estimation.export import make_csv_header
from pose_estimation.processing import TRACKING_BODY, TRACKING_HANDS_ARMS
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
