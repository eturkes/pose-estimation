"""Tests for ``pose_estimation.triangulation`` math primitives.

The fusion integration (``fuse_session_frame``) is a stub; the
primitives below — ``projection_matrix``, ``project_points``,
``undistort_points``, ``triangulate_views`` — are pure math and
testable against synthetic ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from pose_estimation._types import CameraCalibration, SessionCalibration, SessionFrame
from pose_estimation.triangulation import (
    fuse_session_frame,
    project_points,
    projection_matrix,
    session_projection_matrices,
    triangulate_views,
    undistort_points,
)

# ---------------------------------------------------------------------------
# Synthetic camera factories
# ---------------------------------------------------------------------------


def _make_cam(
    name: str,
    rvec: list[float],
    tvec: list[float],
    fx: float = 1000.0,
    cx: float = 960.0,
    cy: float = 540.0,
) -> CameraCalibration:
    return CameraCalibration(
        name=name,
        resolution=(1920, 1080),
        K=np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        distortion=np.zeros(5, dtype=np.float64),
        rvec=np.asarray(rvec, dtype=np.float64),
        tvec=np.asarray(tvec, dtype=np.float64),
    )


def _three_camera_session() -> SessionCalibration:
    """Three cameras in a wide-baseline arc looking at the origin."""
    cams = [
        _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        # cam2: 1m to the right, rotated slightly toward origin.
        _make_cam("cam2", [0.0, -0.3, 0.0], [-1.0, 0.0, 0.0]),
        # cam3: 1m to the left, rotated toward origin.
        _make_cam("cam3", [0.0, 0.3, 0.0], [1.0, 0.0, 0.0]),
    ]
    return SessionCalibration(
        format_version=1,
        session_id="synthetic",
        world_frame="cam1",
        cameras={c["name"]: c for c in cams},
        reprojection_error_px=0.0,
        solver="test",
        solved_at="2026-05-18T12:00:00Z",
    )


# ---------------------------------------------------------------------------
# projection_matrix / session_projection_matrices
# ---------------------------------------------------------------------------


def test_projection_matrix_world_frame_is_K_times_Rt():
    cam = _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    P = projection_matrix(cam)
    assert P.shape == (3, 4)
    # World-frame camera: P = K [I | 0].
    np.testing.assert_allclose(P[:, :3], cam["K"])
    np.testing.assert_allclose(P[:, 3], np.zeros(3), atol=1e-9)


def test_session_projection_matrices_one_per_camera():
    calib = _three_camera_session()
    Ps = session_projection_matrices(calib)
    assert set(Ps.keys()) == {"cam1", "cam2", "cam3"}
    for P in Ps.values():
        assert P.shape == (3, 4)


# ---------------------------------------------------------------------------
# project_points / undistort_points
# ---------------------------------------------------------------------------


def test_project_points_world_origin_at_principal_point():
    cam = _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 5.0], cx=960.0, cy=540.0)
    pts = project_points(np.array([[0.0, 0.0, 0.0]]), cam)
    assert pts.shape == (1, 2)
    np.testing.assert_allclose(pts[0], [960.0, 540.0], atol=1e-6)


def test_undistort_points_is_identity_for_zero_distortion():
    cam = _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    image_pts = np.array([[100.0, 200.0], [800.0, 600.0]])
    undist = undistort_points(image_pts, cam)
    np.testing.assert_allclose(undist, image_pts, atol=1e-4)


# ---------------------------------------------------------------------------
# triangulate_views — round-trip on synthetic data
# ---------------------------------------------------------------------------


def test_triangulate_views_recovers_world_points_under_two_views():
    """Project known 3D points via two cameras, then triangulate them back."""
    cam1 = _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cam2 = _make_cam("cam2", [0.0, -0.2, 0.0], [-0.5, 0.0, 0.0])
    world = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.1, -0.05, 2.5],
            [-0.2, 0.1, 4.0],
        ]
    )
    pts_1 = project_points(world, cam1)
    pts_2 = project_points(world, cam2)
    recovered = triangulate_views(
        [projection_matrix(cam1), projection_matrix(cam2)],
        [pts_1, pts_2],
    )
    assert recovered.shape == world.shape
    np.testing.assert_allclose(recovered, world, atol=1e-4)


def test_triangulate_views_three_views_improve_recovery():
    """Three views: noisy 2D inputs should still triangulate near the truth."""
    rng = np.random.default_rng(2026)
    calib = _three_camera_session()
    world = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.5, 0.0, 2.5],
            [-0.3, 0.2, 3.5],
            [0.1, -0.1, 4.0],
        ]
    )
    cams = list(calib["cameras"].values())
    Ps = [projection_matrix(c) for c in cams]
    noisy = [project_points(world, c) + rng.normal(0, 0.1, size=(world.shape[0], 2)) for c in cams]
    recovered = triangulate_views(Ps, noisy)
    np.testing.assert_allclose(recovered, world, atol=5e-3)


def test_triangulate_views_marks_insufficient_views_as_nan():
    """Keypoints with <2 visible views should come back as NaN rows."""
    cam1 = _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cam2 = _make_cam("cam2", [0.0, -0.2, 0.0], [-0.5, 0.0, 0.0])
    world = np.array([[0.0, 0.0, 3.0], [0.2, 0.0, 4.0]])
    pts_1 = project_points(world, cam1)
    pts_2 = project_points(world, cam2)
    # Drop keypoint 1 from view 2 (NaN) — only 1 view sees it.
    pts_2[1] = np.nan
    recovered = triangulate_views(
        [projection_matrix(cam1), projection_matrix(cam2)],
        [pts_1, pts_2],
    )
    np.testing.assert_allclose(recovered[0], world[0], atol=1e-4)
    assert np.all(np.isnan(recovered[1]))


def test_triangulate_views_shape_mismatch_raises():
    with pytest.raises(ValueError, match="point arrays"):
        triangulate_views(
            [np.eye(3, 4), np.eye(3, 4)],
            [np.zeros((2, 2))],
        )


def test_triangulate_views_per_view_shape_raises():
    with pytest.raises(ValueError, match=r"expected.*\(2, 2\)"):
        triangulate_views(
            [np.eye(3, 4), np.eye(3, 4)],
            [np.zeros((2, 2)), np.zeros((3, 2))],
        )


def test_triangulate_views_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        triangulate_views([], [])


# ---------------------------------------------------------------------------
# fuse_session_frame stub
# ---------------------------------------------------------------------------


def test_fuse_session_frame_is_stub():
    calib = _three_camera_session()
    sf = SessionFrame(frame_index=0, frames={})
    with pytest.raises(NotImplementedError, match="not yet wired"):
        fuse_session_frame(sf, per_camera_keypoints={}, calibration=calib)
