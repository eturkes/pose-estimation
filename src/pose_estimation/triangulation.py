"""3D triangulation primitives for multi-camera pose fusion.

Provides standalone, unit-testable math helpers (projection matrices,
projection, undistortion, weighted linear DLT) plus a session-level
``fuse_session_frame`` integration point that is currently a
``NotImplementedError`` stub.  Policy (which views to use, confidence
weighting, outlier rejection, missing-view handling) belongs in the
integration layer; the primitives below are pure math.

See ``.claude/tech/multicam.md`` for the planned data flow and
``.claude/tech/calibration.md`` for coordinate-frame conventions
(OpenCV: +X right, +Y down, +Z forward; metres for tvec / world).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ._types import CameraCalibration, SessionCalibration, SessionFrame


def projection_matrix(camera: CameraCalibration) -> np.ndarray:
    """Build a 3x4 projection matrix ``P = K [R | t]`` for *camera*.

    ``R`` is the rotation matrix derived from the calibration's
    Rodrigues ``rvec``.  The result maps a homogeneous world point
    ``(X, Y, Z, 1)`` to homogeneous image coordinates
    ``(u, v, w)`` (before distortion).
    """
    R, _ = cv2.Rodrigues(np.asarray(camera["rvec"], dtype=np.float64))
    t = np.asarray(camera["tvec"], dtype=np.float64).reshape(3, 1)
    Rt = np.hstack([R, t])  # (3, 4)
    K = np.asarray(camera["K"], dtype=np.float64)
    return K @ Rt


def session_projection_matrices(calibration: SessionCalibration) -> dict[str, np.ndarray]:
    """Compute one projection matrix per camera in *calibration*."""
    return {name: projection_matrix(cam) for name, cam in calibration["cameras"].items()}


def project_points(world_points: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Project ``world_points`` (N, 3) into *camera*'s image plane.

    Applies the camera's distortion model.  Returns a ``(N, 2)``
    array in pixel coordinates.
    """
    pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 1, 3)
    image_pts, _ = cv2.projectPoints(
        pts,
        np.asarray(camera["rvec"], dtype=np.float64),
        np.asarray(camera["tvec"], dtype=np.float64),
        np.asarray(camera["K"], dtype=np.float64),
        np.asarray(camera["distortion"], dtype=np.float64),
    )
    return image_pts.reshape(-1, 2)


def undistort_points(image_points: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Undistort 2D pixel-space ``image_points`` (N, 2).

    Returns pixel-space coordinates with the camera's lens distortion
    removed (suitable for direct use with ``projection_matrix``).
    """
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(camera["K"], dtype=np.float64)
    dist = np.asarray(camera["distortion"], dtype=np.float64)
    # P=K projects the normalised result back to pixel coords.
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    return undistorted.reshape(-1, 2)


def triangulate_views(
    projection_matrices: list[np.ndarray],
    points_per_view: list[np.ndarray],
    weights: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Linear DLT triangulation of one 3D point per keypoint across views.

    Given ``V`` views, ``N`` keypoints per view:
    - ``projection_matrices``: list of ``V`` (3, 4) matrices.
    - ``points_per_view``: list of ``V`` (N, 2) pixel-space arrays
      (typically undistorted via ``undistort_points``).
    - ``weights`` (optional): list of ``V`` (N,) per-keypoint weights
      (e.g. detector confidences).  ``None`` → uniform weights.

    Returns ``(N, 3)`` world-space points.  ``NaN`` rows indicate
    keypoints with insufficient visible views.

    The DLT system for each keypoint is::

        [ w_i * (x_i * P_i[2] - P_i[0]) ]
        [ w_i * (y_i * P_i[2] - P_i[1]) ]   X = 0
        [           ...                  ]

    solved by SVD; the world point is the right singular vector
    matching the smallest singular value (homogenised).
    """
    if not projection_matrices:
        raise ValueError("triangulate_views: projection_matrices is empty")
    if len(projection_matrices) != len(points_per_view):
        raise ValueError(
            f"triangulate_views: got {len(projection_matrices)} projection matrices "
            f"but {len(points_per_view)} point arrays"
        )
    V = len(projection_matrices)
    N = int(points_per_view[0].shape[0])
    for v, pts in enumerate(points_per_view):
        if pts.shape != (N, 2):
            raise ValueError(
                f"triangulate_views: points_per_view[{v}] has shape {pts.shape}, expected ({N}, 2)"
            )
    if weights is None:
        weights = [np.ones(N, dtype=np.float64) for _ in range(V)]
    elif len(weights) != V:
        raise ValueError(f"triangulate_views: got {len(weights)} weight arrays for {V} views")

    Ps = [np.asarray(P, dtype=np.float64) for P in projection_matrices]
    pts_arr = [np.asarray(p, dtype=np.float64) for p in points_per_view]
    w_arr = [np.asarray(w, dtype=np.float64) for w in weights]

    world = np.full((N, 3), np.nan, dtype=np.float64)
    for k in range(N):
        rows: list[np.ndarray] = []
        for v in range(V):
            wv = float(w_arr[v][k])
            if wv <= 0.0 or not np.all(np.isfinite(pts_arr[v][k])):
                continue
            x, y = pts_arr[v][k]
            P = Ps[v]
            rows.append(wv * (x * P[2] - P[0]))
            rows.append(wv * (y * P[2] - P[1]))
        if len(rows) < 4:
            # Need at least two views (4 equations) to triangulate.
            continue
        A = np.vstack(rows)
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        X_h = vh[-1]
        if abs(X_h[3]) < 1e-12:
            continue
        world[k] = X_h[:3] / X_h[3]
    return world


# ---------------------------------------------------------------------------
# Integration (stub)
# ---------------------------------------------------------------------------


def fuse_session_frame(
    session_frame: SessionFrame,
    per_camera_keypoints: dict[str, np.ndarray],
    calibration: SessionCalibration,
    *,
    confidences: dict[str, np.ndarray] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Fuse per-camera 2D keypoints into one world-space 3D pose.

    Not yet wired.  The primitives above (``projection_matrix``,
    ``undistort_points``, ``triangulate_views``) cover the math; this
    integration layer must decide on confidence policy, missing-view
    handling, outlier rejection (RANSAC / cheirality check), and
    optional bundle refinement.  See ``.claude/tech/multicam.md``.
    """
    raise NotImplementedError(
        "fuse_session_frame is not yet wired. Primitives are in place "
        "(projection_matrix, undistort_points, triangulate_views); the "
        "policy layer is tracked as a follow-up. Called with "
        f"frame_index={session_frame['frame_index']}, "
        f"cameras_with_kps={sorted(per_camera_keypoints)}, "
        f"calibration_world_frame={calibration['world_frame']!r}, "
        f"confidences={'provided' if confidences is not None else 'absent'}, "
        f"extra kwargs={sorted(kwargs)}."
    )


__all__ = [
    "fuse_session_frame",
    "project_points",
    "projection_matrix",
    "session_projection_matrices",
    "triangulate_views",
    "undistort_points",
]
