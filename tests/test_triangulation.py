"""Tests for ``pose_estimation.triangulation``.

Math primitives (``projection_matrix``, ``project_points``,
``undistort_points``, ``triangulate_views``) and the
``fuse_session_frame`` policy layer are both tested against synthetic
multi-view ground truth (known 3D points projected through known
cameras).
"""

from __future__ import annotations

import numpy as np
import pytest

from pose_estimation._types import CameraCalibration, SessionCalibration
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
    distortion: list[float] | None = None,
) -> CameraCalibration:
    return CameraCalibration(
        name=name,
        resolution=(1920, 1080),
        K=np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        distortion=(
            np.zeros(5, dtype=np.float64)
            if distortion is None
            else np.asarray(distortion, dtype=np.float64)
        ),
        rvec=np.asarray(rvec, dtype=np.float64),
        tvec=np.asarray(tvec, dtype=np.float64),
    )


def _three_camera_session(distortion: list[float] | None = None) -> SessionCalibration:
    """Three cameras in a wide-baseline arc looking at the origin."""
    cams = [
        _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], distortion=distortion),
        # cam2: 1m to the right, rotated slightly toward origin.
        _make_cam("cam2", [0.0, -0.3, 0.0], [-1.0, 0.0, 0.0], distortion=distortion),
        # cam3: 1m to the left, rotated toward origin.
        _make_cam("cam3", [0.0, 0.3, 0.0], [1.0, 0.0, 0.0], distortion=distortion),
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
# fuse_session_frame — policy layer on synthetic multi-view data
# ---------------------------------------------------------------------------

_MILD_DISTORTION = [0.05, -0.02, 0.001, 0.001, 0.0]

# A 12-keypoint synthetic "skeleton" spread through the working volume.
_SKELETON = np.array(
    [
        [-0.40, -0.30, 2.8],
        [0.40, -0.30, 2.8],
        [-0.45, 0.00, 3.0],
        [0.45, 0.00, 3.0],
        [-0.40, 0.30, 3.2],
        [0.40, 0.30, 3.2],
        [-0.15, 0.35, 2.9],
        [0.15, 0.35, 2.9],
        [-0.10, 0.40, 3.1],
        [0.10, 0.40, 3.1],
        [-0.05, 0.45, 3.0],
        [0.05, 0.45, 3.0],
    ]
)


def _project_skeleton(
    calib: SessionCalibration,
    world: np.ndarray = _SKELETON,
    noise_px: float = 0.0,
    seed: int = 31,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for name, cam in calib["cameras"].items():
        pts = project_points(world, cam)
        if noise_px > 0.0:
            pts = pts + rng.normal(0.0, noise_px, size=pts.shape)
        out[name] = pts
    return out


def test_fuse_recovers_noisy_skeleton_within_5mm():
    """3 cameras, lens distortion, 0.3px detector noise → < 5mm error."""
    calib = _three_camera_session(distortion=_MILD_DISTORTION)
    kps = _project_skeleton(calib, noise_px=0.3)
    world, diag = fuse_session_frame(kps, calib)

    assert world.shape == _SKELETON.shape
    assert np.abs(world - _SKELETON).max() < 5e-3
    assert np.all(diag["n_views"] == 3)
    assert np.all(diag["cheirality_ok"])
    assert np.all(np.isfinite(diag["reprojection_error_px"]))
    assert diag["reprojection_error_px"].max() < 2.0
    np.testing.assert_allclose(diag["confidence"], 1.0)


def test_fuse_works_with_one_camera_missing():
    """Two of three cameras still triangulate within tolerance.

    Tolerance is wider than the 3-view case: with a 1m baseline at
    ~3m depth and 0.3px noise, two-view depth uncertainty is roughly
    4mm per noise-sigma (z^2/(f*b) * sigma_px * sqrt(2)), so 1cm
    covers the max over 12 keypoints.
    """
    calib = _three_camera_session(distortion=_MILD_DISTORTION)
    kps = _project_skeleton(calib, noise_px=0.3)
    del kps["cam3"]
    world, diag = fuse_session_frame(kps, calib)

    assert np.abs(world - _SKELETON).max() < 1e-2
    assert np.all(diag["n_views"] == 2)
    assert np.all(diag["cheirality_ok"])


def test_fuse_excludes_occluded_keypoints_per_view():
    """Zero confidence in one view excludes only that view's keypoint."""
    calib = _three_camera_session()
    kps = _project_skeleton(calib, noise_px=0.2)
    conf = {n: np.ones(len(_SKELETON)) for n in kps}
    conf["cam1"][2] = 0.0  # cam1 cannot see keypoint 2

    world, diag = fuse_session_frame(kps, calib, confidences=conf)

    assert np.abs(world - _SKELETON).max() < 5e-3
    assert diag["n_views"][2] == 2
    assert np.all(np.delete(diag["n_views"], 2) == 3)


def test_fuse_insufficient_views_yield_nan():
    """A keypoint seen by fewer than min_views cameras comes back NaN."""
    calib = _three_camera_session()
    kps = _project_skeleton(calib)
    kps["cam2"][1] = np.nan
    kps["cam3"][1] = np.nan  # keypoint 1: only cam1 left

    world, diag = fuse_session_frame(kps, calib)

    assert np.all(np.isnan(world[1]))
    assert diag["n_views"][1] == 1
    assert diag["confidence"][1] == 0.0
    assert np.isnan(diag["reprojection_error_px"][1])
    assert not diag["cheirality_ok"][1]
    # Remaining keypoints are unaffected.
    mask = np.arange(len(_SKELETON)) != 1
    assert np.abs(world[mask] - _SKELETON[mask]).max() < 1e-4


def test_fuse_outlier_view_is_rejected():
    """A grossly wrong view is dropped and the keypoint re-triangulated."""
    calib = _three_camera_session()
    kps = _project_skeleton(calib)
    kps["cam3"][0] += np.array([80.0, -60.0])  # corrupt one view of keypoint 0

    world, diag = fuse_session_frame(kps, calib)

    assert np.abs(world - _SKELETON).max() < 1e-3
    assert diag["n_views"][0] == 2
    assert np.all(np.delete(diag["n_views"], 0) == 3)
    assert diag["reprojection_error_px"][0] < 1.0


def test_fuse_min_views_three_drops_two_view_keypoints():
    """min_views=3 rejects keypoints that only two cameras observed."""
    calib = _three_camera_session()
    kps = _project_skeleton(calib)
    kps["cam2"][0] = np.nan  # keypoint 0: two views remain

    world, diag = fuse_session_frame(kps, calib, min_views=3)

    assert np.all(np.isnan(world[0]))
    assert diag["n_views"][0] == 2
    mask = np.arange(len(_SKELETON)) != 0
    assert np.abs(world[mask] - _SKELETON[mask]).max() < 1e-4
    assert np.all(diag["n_views"][mask] == 3)


def test_fuse_flags_cheirality_violations():
    """A point triangulated behind the cameras is flagged, not dropped."""
    cams = [
        _make_cam("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _make_cam("cam2", [0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]),
    ]
    calib = SessionCalibration(
        format_version=1,
        session_id="behind",
        world_frame="cam1",
        cameras={c["name"]: c for c in cams},
        reprojection_error_px=0.0,
        solver="test",
        solved_at="2026-05-18T12:00:00Z",
    )
    behind = np.array([[0.1, 0.05, -3.0]])  # behind both cameras
    kps = {}
    for name, cam in calib["cameras"].items():
        P = projection_matrix(cam)
        x_h = P @ np.append(behind[0], 1.0)
        kps[name] = (x_h[:2] / x_h[2]).reshape(1, 2)

    world, diag = fuse_session_frame(kps, calib)

    np.testing.assert_allclose(world, behind, atol=1e-6)
    assert not diag["cheirality_ok"][0]
    assert diag["n_views"][0] == 2


def test_fuse_confidence_is_mean_of_contributing_views():
    calib = _three_camera_session()
    kps = _project_skeleton(calib)
    conf = {
        "cam1": np.full(len(_SKELETON), 0.9),
        "cam2": np.full(len(_SKELETON), 0.6),
        "cam3": np.full(len(_SKELETON), 0.3),
    }
    _world, diag = fuse_session_frame(kps, calib, confidences=conf)
    np.testing.assert_allclose(diag["confidence"], (0.9 + 0.6 + 0.3) / 3, atol=1e-9)


def test_fuse_validation_errors():
    calib = _three_camera_session()
    kps = _project_skeleton(calib)

    with pytest.raises(ValueError, match="empty"):
        fuse_session_frame({}, calib)
    with pytest.raises(ValueError, match="missing from calibration"):
        fuse_session_frame({"cam9": kps["cam1"]}, calib)
    with pytest.raises(ValueError, match=r"expected \(N, 2\)"):
        fuse_session_frame({"cam1": np.zeros((4, 3))}, calib)
    with pytest.raises(ValueError, match="expected 12"):
        fuse_session_frame({"cam1": kps["cam1"], "cam2": kps["cam2"][:5]}, calib)
    with pytest.raises(ValueError, match=r"confidences\['cam1'\]"):
        fuse_session_frame(kps, calib, confidences={"cam1": np.ones(3)})
    with pytest.raises(ValueError, match="min_views must be >= 2"):
        fuse_session_frame(kps, calib, min_views=1)
