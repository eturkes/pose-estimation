"""Tests for charuco.py — synthetic-render calibration solve.

A ChArUco board with known geometry is warped through known camera
models (zero distortion) into MJPG videos; ``solve_charuco`` must
recover intrinsics, extrinsics, and a small global reprojection RMS.
Rendering supersamples 3x then INTER_AREA-downsamples — plain
``warpPerspective`` aliases the 4x4 marker interiors into undetectable
mush at realistic board scales.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import cv2
import numpy as np
import pytest

from pose_estimation.calibration import (
    CalibrationError,
    load_calibration,
    save_calibration,
)
from pose_estimation.charuco import (
    CHARUCO_SOLVER_TAG,
    _subsample,
    detect_charuco_corners,
    make_charuco_board,
    render_charuco_board,
    solve_charuco,
)

BOARD = make_charuco_board()
PX_PER_SQUARE = 120
BOARD_IMG = render_charuco_board(BOARD, px_per_square=PX_PER_SQUARE, margin_squares=0.0)
_SX, _SY = BOARD.getChessboardSize()
_SQ = BOARD.getSquareLength()
BOARD_W_M, BOARD_H_M = _SX * _SQ, _SY * _SQ

# Ground-truth rig: cam1 = world; cam2/cam3 yawed toward the volume.
# Mixed resolutions exercise per-camera size handling.  (Values typed
# Any: heterogeneous ndarray/tuple members otherwise infer as unions.)
GT: dict[str, dict[str, Any]] = {
    "cam1": {
        "K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]),
        "size": (1280, 720),
        "rvec": np.zeros(3),
        "tvec": np.zeros(3),
    },
    "cam2": {
        "K": np.array([[880.0, 0, 650], [0, 880.0, 350], [0, 0, 1]]),
        "size": (1280, 720),
        "rvec": np.array([0.0, np.deg2rad(25.0), 0.0]),
        "tvec": np.array([-0.8, 0.0, 0.25]),
    },
    "cam3": {
        "K": np.array([[700.0, 0, 480], [0, 700.0, 270], [0, 0, 1]]),
        "size": (960, 540),
        "rvec": np.array([0.0, np.deg2rad(-20.0), 0.05]),
        "tvec": np.array([0.7, 0.0, 0.2]),
    },
}
CAM3_SYNC_OFFSET = 2

_BG_GRAY = 180


def _board_poses(n: int, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    """Centre-parametrised board poses facing the rig with varied tilt.

    Identity orientation faces the camera: object x right, y down,
    +z INTO the board (OpenCV planar-target convention).
    """
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.6)])
        tvec = np.array(
            [rng.uniform(-0.40, 0.25), rng.uniform(-0.15, 0.15), rng.uniform(0.70, 1.35)]
        )
        poses.append((rvec, tvec))
    return poses


def _render_view(
    K: np.ndarray,
    rvec_cam: np.ndarray,
    tvec_cam: np.ndarray,
    rvec_board: np.ndarray,
    tvec_board: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Render the board seen by a camera; plain background if off-view."""
    w, h = size
    blank = np.full((h, w), _BG_GRAY, dtype=np.uint8)
    corners_m = np.array(
        [[0, 0, 0], [BOARD_W_M, 0, 0], [BOARD_W_M, BOARD_H_M, 0], [0, BOARD_H_M, 0]]
    )
    centred = corners_m - np.array([BOARD_W_M / 2, BOARD_H_M / 2, 0.0])
    R_b = cv2.Rodrigues(np.asarray(rvec_board, dtype=np.float64))[0]
    R_c = cv2.Rodrigues(np.asarray(rvec_cam, dtype=np.float64))[0]
    pts_world = (R_b @ centred.T).T + tvec_board
    pts_cam = (R_c @ pts_world.T).T + tvec_cam
    if np.any(pts_cam[:, 2] <= 0.1):
        return blank
    proj = (K @ pts_cam.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    if proj[:, 0].min() < 10 or proj[:, 0].max() > w - 10:
        return blank
    if proj[:, 1].min() < 10 or proj[:, 1].max() > h - 10:
        return blank
    bh, bw = BOARD_IMG.shape[:2]
    src = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
    ss = 3  # supersample: anti-alias marker interiors
    H = cv2.getPerspectiveTransform(src, proj.astype(np.float32) * ss)
    canvas = cv2.warpPerspective(
        BOARD_IMG,
        H,
        (w * ss, h * ss),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_BG_GRAY,
    )
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def _write_video(path: pathlib.Path, frames: list[np.ndarray], size: tuple[int, int]) -> None:
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, size)
    assert writer.isOpened()
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


@pytest.fixture(scope="module")
def solved_session(tmp_path_factory: pytest.TempPathFactory):
    """Synthetic 3-camera session (cam3 sync-offset 2) solved once."""
    session_dir = tmp_path_factory.mktemp("calib_session")
    poses = _board_poses(60)
    for name, gt in GT.items():
        frames = [
            _render_view(gt["K"], gt["rvec"], gt["tvec"], rb, tb, gt["size"]) for rb, tb in poses
        ]
        if name == "cam3":  # prepend pre-roll the manifest's sync_offset trims
            w, h = gt["size"]
            frames = [np.full((h, w), _BG_GRAY, dtype=np.uint8)] * CAM3_SYNC_OFFSET + frames
        _write_video(session_dir / f"{name}.avi", frames, gt["size"])
    manifest = {
        "format_version": 1,
        "session_id": "calib_synthetic",
        "cameras": [
            {"name": "cam1", "file": "cam1.avi", "sync_offset": 0},
            {"name": "cam2", "file": "cam2.avi", "sync_offset": 0},
            {"name": "cam3", "file": "cam3.avi", "sync_offset": CAM3_SYNC_OFFSET},
        ],
    }
    (session_dir / "session.json").write_text(json.dumps(manifest))
    return solve_charuco(session_dir)


def test_solve_recovers_intrinsics(solved_session):
    for name, gt in GT.items():
        K = np.asarray(solved_session["cameras"][name]["K"])
        f_gt = gt["K"][0, 0]
        assert abs(K[0, 0] - gt["K"][0, 0]) < 0.02 * f_gt, name
        assert abs(K[1, 1] - gt["K"][1, 1]) < 0.02 * f_gt, name
        assert abs(K[0, 2] - gt["K"][0, 2]) < 12.0, name
        assert abs(K[1, 2] - gt["K"][1, 2]) < 12.0, name


def test_solve_recovers_extrinsics(solved_session):
    for name in ("cam2", "cam3"):
        cam = solved_session["cameras"][name]
        R_est = cv2.Rodrigues(np.asarray(cam["rvec"], dtype=np.float64))[0]
        R_gt = cv2.Rodrigues(GT[name]["rvec"])[0]
        angle_err = np.rad2deg(np.linalg.norm(cv2.Rodrigues(R_est @ R_gt.T)[0]))
        t_err = np.linalg.norm(np.asarray(cam["tvec"]) - GT[name]["tvec"])
        assert angle_err < 1.0, f"{name}: rotation error {angle_err:.3f} deg"
        assert t_err < 0.015, f"{name}: translation error {t_err * 1000:.1f} mm"


def test_solve_world_frame_and_metadata(solved_session):
    assert solved_session["world_frame"] == "cam1"
    assert np.allclose(solved_session["cameras"]["cam1"]["rvec"], 0.0)
    assert np.allclose(solved_session["cameras"]["cam1"]["tvec"], 0.0)
    assert solved_session["solver"] == CHARUCO_SOLVER_TAG
    assert solved_session["session_id"] == "calib_synthetic"
    for name, gt in GT.items():
        assert solved_session["cameras"][name]["resolution"] == gt["size"]


def test_solve_global_rms_small(solved_session):
    assert 0.0 < solved_session["reprojection_error_px"] < 1.5


def test_solve_roundtrips_through_save_load(solved_session, tmp_path: pathlib.Path):
    out = tmp_path / "calibration.json"
    save_calibration(solved_session, out)
    loaded = load_calibration(out)
    assert loaded["world_frame"] == "cam1"
    assert set(loaded["cameras"]) == set(GT)
    np.testing.assert_allclose(
        loaded["cameras"]["cam2"]["K"], solved_session["cameras"]["cam2"]["K"]
    )


def test_solve_unknown_world_frame_raises(solved_session, tmp_path: pathlib.Path):
    # Reuse an empty dir: world_frame validation fires before detection.
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    _write_video(session_dir / "cam1.avi", [np.zeros((48, 64), dtype=np.uint8)], (64, 48))
    with pytest.raises(CalibrationError, match="world_frame"):
        solve_charuco(session_dir, world_frame="nope")


def test_solve_no_detections_raises(tmp_path: pathlib.Path):
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    blank = [np.full((480, 640), _BG_GRAY, dtype=np.uint8)] * 10
    _write_video(session_dir / "cam1.avi", blank, (640, 480))
    with pytest.raises(CalibrationError, match=r"cam1.*usable frames"):
        solve_charuco(session_dir)


def test_solve_insufficient_overlap_raises(tmp_path: pathlib.Path):
    """Both cameras calibrate individually but share no board frames."""
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    size = (640, 480)
    poses = _board_poses(48, seed=11)
    rv0, tv0 = np.zeros(3), np.zeros(3)
    blank = np.full((size[1], size[0]), _BG_GRAY, dtype=np.uint8)
    cam1 = [
        _render_view(K, rv0, tv0, rb, tb, size) if i < 24 else blank
        for i, (rb, tb) in enumerate(poses)
    ]
    cam2 = [
        _render_view(K, rv0, tv0, rb, tb, size) if i >= 24 else blank
        for i, (rb, tb) in enumerate(poses)
    ]
    _write_video(session_dir / "cam1.avi", cam1, size)
    _write_video(session_dir / "cam2.avi", cam2, size)
    with pytest.raises(CalibrationError, match=r"cam2.*world-frame camera"):
        solve_charuco(session_dir)


def test_detect_charuco_corners_applies_sync_offset(tmp_path: pathlib.Path):
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    size = (640, 480)
    # Deterministic central poses: this test checks index arithmetic only.
    poses = [
        (np.array([0.1 * i, -0.1 * i, 0.0]), np.array([0.0, 0.0, 0.8 + 0.1 * i])) for i in range(4)
    ]
    frames = [_render_view(K, np.zeros(3), np.zeros(3), rb, tb, size) for rb, tb in poses]
    path = tmp_path / "cam1.avi"
    _write_video(path, frames, size)
    plain, _ = detect_charuco_corners(path, BOARD)
    shifted, _ = detect_charuco_corners(path, BOARD, sync_offset=2)
    assert [d.frame_idx for d in plain] == [0, 1, 2, 3]
    assert [d.frame_idx for d in shifted] == [0, 1]  # raw 2,3 → logical 0,1


def test_detect_charuco_corners_missing_video_raises(tmp_path: pathlib.Path):
    with pytest.raises(CalibrationError, match="cannot open"):
        detect_charuco_corners(tmp_path / "missing.avi", BOARD)


def test_make_charuco_board_rejects_oversized_marker():
    with pytest.raises(CalibrationError, match="marker_size_m"):
        make_charuco_board(square_size_m=0.04, marker_size_m=0.04)


def test_render_charuco_board_dimensions():
    img = render_charuco_board(BOARD, px_per_square=50, margin_squares=0.5)
    sx, sy = BOARD.getChessboardSize()
    assert img.shape == (sy * 50 + 50, sx * 50 + 50)  # margin = 25px per side


def test_subsample_spreads_and_caps():
    assert _subsample(list(range(10)), 20) == list(range(10))
    picked = _subsample(list(range(100)), 10)
    assert len(picked) == 10
    assert picked[0] == 0
    assert picked[-1] == 99
    assert picked == sorted(picked)
