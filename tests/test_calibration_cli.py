"""Tests for calibration_cli.py — subcommand wiring + capture helpers.

The full ChArUco solve is covered by test_charuco.py; here ``solve`` is
exercised at the wiring level (monkeypatched solver → save + summary)
plus its error path, and ``board`` end-to-end (the rendered PNG must be
re-detectable, guarding print fidelity).
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import pytest

from pose_estimation import calibration_cli
from pose_estimation._types import CameraCalibration, SessionCalibration
from pose_estimation.calibration import load_calibration, save_calibration
from pose_estimation.calibration_cli import (
    _compose_grid,
    _parse_devices,
    _parse_squares,
    main,
)


def _synthetic_calibration() -> SessionCalibration:
    cameras = {}
    for i, name in enumerate(["cam1", "cam2"]):
        cameras[name] = CameraCalibration(
            name=name,
            resolution=(1280, 720),
            K=np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]),
            distortion=np.zeros(5),
            rvec=np.zeros(3) if i == 0 else np.array([0.0, 0.4, 0.0]),
            tvec=np.zeros(3) if i == 0 else np.array([-0.8, 0.0, 0.2]),
        )
    return SessionCalibration(
        format_version=1,
        session_id="cli_test",
        world_frame="cam1",
        cameras=cameras,
        reprojection_error_px=0.5,
        solver="opencv-charuco",
        solved_at="2026-06-08T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_prints_summary(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    path = tmp_path / "calib.json"
    save_calibration(_synthetic_calibration(), path)
    assert main(["verify", "--calibration", str(path)]) == 0
    out = capsys.readouterr().out
    assert "cli_test" in out
    assert "cam2" in out


def test_verify_missing_file_errors(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    assert main(["verify", "--calibration", str(tmp_path / "nope.json")]) == 2
    assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# solve (wiring; the solver itself is covered by test_charuco.py)
# ---------------------------------------------------------------------------


def test_solve_saves_and_summarises(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    seen: dict = {}

    def fake_solve(session_dir, *, board, world_frame, max_frames):
        seen.update(world_frame=world_frame, max_frames=max_frames, board=board)
        return _synthetic_calibration()

    monkeypatch.setattr(calibration_cli, "solve_charuco", fake_solve)
    out_path = tmp_path / "calib.json"
    rc = main(
        [
            "solve",
            "--session-dir",
            str(tmp_path),
            "--output",
            str(out_path),
            "--world-frame",
            "cam1",
            "--max-frames",
            "30",
        ]
    )
    assert rc == 0
    assert seen["world_frame"] == "cam1"
    assert seen["max_frames"] == 30
    assert load_calibration(out_path)["session_id"] == "cli_test"
    assert "cli_test" in capsys.readouterr().out


def test_solve_empty_session_errors(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    session = tmp_path / "empty"
    session.mkdir()
    rc = main(["solve", "--session-dir", str(session), "--output", str(tmp_path / "c.json")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# board
# ---------------------------------------------------------------------------


def test_board_writes_redetectable_png(tmp_path: pathlib.Path):
    out = tmp_path / "board.png"
    assert main(["board", "--output", str(out), "--px-per-square", "120"]) == 0
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    from pose_estimation.charuco import make_charuco_board

    board = make_charuco_board()
    corners, _ids, _mc, _mi = cv2.aruco.CharucoDetector(board).detectBoard(img)
    sx, sy = board.getChessboardSize()
    assert corners is not None
    assert len(corners) == (sx - 1) * (sy - 1)  # all interior corners


def test_board_custom_geometry_dimensions(tmp_path: pathlib.Path):
    out = tmp_path / "board.png"
    rc = main(
        [
            "board",
            "--output",
            str(out),
            "--px-per-square",
            "60",
            "--squares",
            "5x7",
            "--square-size-m",
            "0.05",
            "--marker-size-m",
            "0.037",
        ]
    )
    assert rc == 0
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    margin = 15  # 60 px/square * default 0.25 margin_squares
    assert img.shape == (7 * 60 + 2 * margin, 5 * 60 + 2 * margin)


def test_board_rejects_marker_not_smaller_than_square(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    rc = main(
        [
            "board",
            "--output",
            str(tmp_path / "b.png"),
            "--square-size-m",
            "0.03",
            "--marker-size-m",
            "0.03",
        ]
    )
    assert rc == 2
    assert "marker_size_m" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# capture helpers (the live loop needs hardware; logic is unit-tested)
# ---------------------------------------------------------------------------


def test_parse_devices_valid_and_invalid():
    assert _parse_devices("0,1,2") == [0, 1, 2]
    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        _parse_devices("0,abc")
    with pytest.raises(argparse.ArgumentTypeError, match="duplicate"):
        _parse_devices("0,0")


def test_parse_squares_valid_and_invalid():
    assert _parse_squares("6x9") == (6, 9)
    assert _parse_squares("5X7") == (5, 7)
    with pytest.raises(argparse.ArgumentTypeError, match="COLSxROWS"):
        _parse_squares("6")
    with pytest.raises(argparse.ArgumentTypeError, match="3x3"):
        _parse_squares("2x9")


def test_compose_grid_scales_to_common_height():
    frames = [
        np.zeros((720, 1280, 3), dtype=np.uint8),
        np.zeros((540, 960, 3), dtype=np.uint8),
    ]
    grid = _compose_grid(frames, cell_height=270)
    assert grid.shape[0] == 270
    assert grid.shape[1] == 480 + 480  # both are 16:9 → 480 px wide at 270 px


def test_capture_name_count_mismatch_errors(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    rc = main(
        [
            "capture",
            "--session-dir",
            str(tmp_path),
            "--devices",
            "0,1",
            "--names",
            "cam1",
        ]
    )
    assert rc == 2
    assert "names" in capsys.readouterr().err
