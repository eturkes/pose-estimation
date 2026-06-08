"""3D triangulation for multi-camera pose fusion.

Provides standalone, unit-testable math helpers (projection matrices,
projection, undistortion, weighted linear DLT) plus the session-level
``fuse_session_frame`` policy layer: per-view validity masking,
confidence weighting, minimum-view enforcement, greedy outlier-view
rejection, cheirality flagging, and per-keypoint reprojection-error
diagnostics.

See ``.claude/tech/multicam.md`` for the session data flow and
``.claude/tech/calibration.md`` for coordinate-frame conventions
(OpenCV: +X right, +Y down, +Z forward; metres for tvec / world).
"""

from __future__ import annotations

import cv2
import numpy as np

from ._types import CameraCalibration, FusionDiagnostics, SessionCalibration


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
# Fusion policy layer
# ---------------------------------------------------------------------------


def _rotation_translation(camera: CameraCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, t)`` mapping world points into *camera*'s frame."""
    R, _ = cv2.Rodrigues(np.asarray(camera["rvec"], dtype=np.float64))
    t = np.asarray(camera["tvec"], dtype=np.float64).reshape(3)
    return R, t


def fuse_session_frame(
    per_camera_keypoints: dict[str, np.ndarray],
    calibration: SessionCalibration,
    *,
    confidences: dict[str, np.ndarray] | None = None,
    min_views: int = 2,
    min_confidence: float = 0.0,
    max_view_reproj_px: float = 20.0,
) -> tuple[np.ndarray, FusionDiagnostics]:
    """Fuse per-camera 2D keypoints into one world-space 3D pose.

    Inputs:
    - ``per_camera_keypoints``: camera name → ``(N, 2)`` *distorted*
      pixel coordinates (the raw detector output frame).  ``NaN``
      marks a keypoint the camera did not observe.  All cameras must
      share the same keypoint count ``N`` and exist in *calibration*.
    - ``confidences``: camera name → ``(N,)`` per-keypoint scores.
      Cameras absent from the dict (or the whole dict being ``None``)
      get uniform weight 1.0.

    Policy, per keypoint:
    1. A view is *valid* when its coordinates are finite and its
       confidence exceeds ``min_confidence``.
    2. Valid views are undistorted and triangulated via weighted DLT
       (weights = confidences).  Fewer than ``min_views`` valid views
       → ``NaN`` world point.
    3. Greedy outlier rejection: while any contributing view reprojects
       worse than ``max_view_reproj_px`` *and* more than ``min_views``
       views remain, drop the worst view and re-triangulate.  At
       exactly ``min_views`` views nothing is dropped — a residual
       error above the threshold stays visible in the diagnostics.
    4. Cheirality: the fused point is flagged ``ok`` only when it lies
       in front (camera-frame ``Z > 0``) of every contributing camera.

    Returns ``(world, diag)``: ``(N, 3)`` world-space points (metres,
    ``NaN`` rows where fusion failed) and per-keypoint
    ``FusionDiagnostics``.  Reprojection errors are measured in the
    original distorted pixel frame.
    """
    if not per_camera_keypoints:
        raise ValueError("fuse_session_frame: per_camera_keypoints is empty")
    if min_views < 2:
        raise ValueError(f"fuse_session_frame: min_views must be >= 2 (got {min_views})")
    unknown = sorted(set(per_camera_keypoints) - set(calibration["cameras"]))
    if unknown:
        raise ValueError(f"fuse_session_frame: cameras missing from calibration: {unknown}")

    names = sorted(per_camera_keypoints)
    cams = {n: calibration["cameras"][n] for n in names}

    pixel_pts: dict[str, np.ndarray] = {}
    n_kps = -1
    for name in names:
        arr = np.asarray(per_camera_keypoints[name], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"fuse_session_frame: keypoints[{name!r}] has shape {arr.shape}, expected (N, 2)"
            )
        if n_kps == -1:
            n_kps = int(arr.shape[0])
        elif arr.shape[0] != n_kps:
            raise ValueError(
                f"fuse_session_frame: keypoints[{name!r}] has {arr.shape[0]} keypoints, "
                f"expected {n_kps}"
            )
        pixel_pts[name] = arr

    conf: dict[str, np.ndarray] = {}
    for name in names:
        if confidences is not None and name in confidences:
            c = np.asarray(confidences[name], dtype=np.float64)
            if c.shape != (n_kps,):
                raise ValueError(
                    f"fuse_session_frame: confidences[{name!r}] has shape {c.shape}, "
                    f"expected ({n_kps},)"
                )
        else:
            c = np.ones(n_kps, dtype=np.float64)
        conf[name] = c

    # Per-view validity → weights; undistort only valid points.
    undist: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for name in names:
        pts = pixel_pts[name]
        valid = np.isfinite(pts).all(axis=1) & (conf[name] > min_confidence)
        u = np.full((n_kps, 2), np.nan, dtype=np.float64)
        if valid.any():
            u[valid] = undistort_points(pts[valid], cams[name])
        undist[name] = u
        weights[name] = np.where(valid, conf[name], 0.0)

    Ps = {n: projection_matrix(cams[n]) for n in names}
    Rts = {n: _rotation_translation(cams[n]) for n in names}

    def _triangulate_keypoint(k: int) -> np.ndarray:
        return triangulate_views(
            [Ps[n] for n in names],
            [undist[n][k : k + 1] for n in names],
            [weights[n][k : k + 1] for n in names],
        )[0]

    def _view_errors(k: int, X: np.ndarray, active: list[str]) -> np.ndarray:
        errs = np.empty(len(active), dtype=np.float64)
        for i, name in enumerate(active):
            proj = project_points(X[np.newaxis, :], cams[name])[0]
            errs[i] = float(np.linalg.norm(proj - pixel_pts[name][k]))
        return errs

    world = triangulate_views(
        [Ps[n] for n in names], [undist[n] for n in names], [weights[n] for n in names]
    )

    n_views = np.zeros(n_kps, dtype=np.int64)
    mean_conf = np.zeros(n_kps, dtype=np.float64)
    reproj = np.full(n_kps, np.nan, dtype=np.float64)
    cheirality = np.zeros(n_kps, dtype=bool)

    for k in range(n_kps):
        active = [n for n in names if weights[n][k] > 0.0]
        n_views[k] = len(active)
        if len(active) < min_views:
            world[k] = np.nan
            continue
        if not np.all(np.isfinite(world[k])):
            continue  # degenerate DLT (e.g. coincident rays)

        errs = _view_errors(k, world[k], active)
        while len(active) > min_views and float(np.max(errs)) > max_view_reproj_px:
            worst = active[int(np.argmax(errs))]
            weights[worst][k] = 0.0
            active.remove(worst)
            world[k] = _triangulate_keypoint(k)
            if not np.all(np.isfinite(world[k])):
                break
            errs = _view_errors(k, world[k], active)
        n_views[k] = len(active)
        if not np.all(np.isfinite(world[k])):
            continue

        mean_conf[k] = float(np.mean([conf[n][k] for n in active]))
        reproj[k] = float(np.mean(errs))
        cheirality[k] = all((Rts[n][0] @ world[k] + Rts[n][1])[2] > 0.0 for n in active)

    diag = FusionDiagnostics(
        n_views=n_views,
        confidence=mean_conf,
        reprojection_error_px=reproj,
        cheirality_ok=cheirality,
    )
    return world, diag


__all__ = [
    "fuse_session_frame",
    "project_points",
    "projection_matrix",
    "session_projection_matrices",
    "triangulate_views",
    "undistort_points",
]
