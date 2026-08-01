"""3D triangulation for multi-camera pose fusion.

Provides standalone, unit-testable math helpers (projection matrices,
projection, undistortion, weighted linear DLT) plus the session-level
``fuse_session_frame`` policy layer: per-view validity masking,
confidence weighting, minimum-view enforcement, deterministic minimal-set
consensus, bounded geometric refinement, triangulation-angle gating,
cheirality flagging, and per-keypoint reprojection-error diagnostics.

See ``docs/technical/multicam.md`` for the session data flow and
``docs/technical/calibration.md`` for coordinate-frame conventions
(OpenCV: +X right, +Y down, +Z forward; metres for tvec / world).
"""

from __future__ import annotations

import itertools
import math

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

        [ sqrt(w_i) * (x_i * P_i[2] - P_i[0]) ]
        [ sqrt(w_i) * (y_i * P_i[2] - P_i[1]) ]   X = 0
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

    Ps: list[np.ndarray] = []
    for v, projection in enumerate(projection_matrices):
        P = np.asarray(projection, dtype=np.float64)
        if P.shape != (3, 4):
            raise ValueError(
                f"triangulate_views: projection_matrices[{v}] has shape {P.shape}, expected (3, 4)"
            )
        if not np.isfinite(P).all():
            raise ValueError(f"triangulate_views: projection_matrices[{v}] must be finite")
        Ps.append(P)

    pts_arr: list[np.ndarray] = []
    N = -1
    for v, points in enumerate(points_per_view):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            expected = "(N, 2)" if N == -1 else f"({N}, 2)"
            raise ValueError(
                f"triangulate_views: points_per_view[{v}] has shape {pts.shape}, "
                f"expected {expected}"
            )
        if N == -1:
            N = int(pts.shape[0])
        elif pts.shape != (N, 2):
            raise ValueError(
                f"triangulate_views: points_per_view[{v}] has shape {pts.shape}, expected ({N}, 2)"
            )
        pts_arr.append(pts)

    if weights is None:
        w_arr = [np.ones(N, dtype=np.float64) for _ in range(V)]
    elif len(weights) != V:
        raise ValueError(f"triangulate_views: got {len(weights)} weight arrays for {V} views")
    else:
        w_arr = []
        for v, weight in enumerate(weights):
            w = np.asarray(weight, dtype=np.float64)
            if w.shape != (N,):
                raise ValueError(
                    f"triangulate_views: weights[{v}] has shape {w.shape}, expected ({N},)"
                )
            if not np.isfinite(w).all():
                raise ValueError(f"triangulate_views: weights[{v}] must be finite")
            if np.any(w < 0.0):
                raise ValueError(f"triangulate_views: weights[{v}] must be non-negative")
            w_arr.append(w)

    world = np.full((N, 3), np.nan, dtype=np.float64)
    for k in range(N):
        rows: list[np.ndarray] = []
        for v in range(V):
            wv = float(w_arr[v][k])
            if wv <= 0.0 or not np.all(np.isfinite(pts_arr[v][k])):
                continue
            x, y = pts_arr[v][k]
            P = Ps[v]
            row_scale = math.sqrt(wv)
            rows.append(row_scale * (x * P[2] - P[0]))
            rows.append(row_scale * (y * P[2] - P[1]))
        if len(rows) < 4:
            # Need at least two views (4 equations) to triangulate.
            continue
        A = np.vstack(rows)
        try:
            _, _, vh = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
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


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _backproject_rays(image_points: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Return unit world-frame rays for finite distorted pixel coordinates."""
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(camera["K"], dtype=np.float64)
    dist = np.asarray(camera["distortion"], dtype=np.float64)
    normalised = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    camera_rays = np.column_stack([normalised, np.ones(len(normalised), dtype=np.float64)])
    R, _t = _rotation_translation(camera)
    world_rays = camera_rays @ R
    norms = np.linalg.norm(world_rays, axis=1, keepdims=True)
    return world_rays / norms


def _max_acute_ray_angle_deg(rays: list[np.ndarray]) -> float:
    """Maximum acute angle between any two world-frame viewing rays."""
    if len(rays) < 2:
        return float("nan")
    dots = [abs(float(a @ b)) for a, b in itertools.combinations(rays, 2)]
    min_abs_dot = min(dots)
    return math.degrees(math.acos(float(np.clip(min_abs_dot, 0.0, 1.0))))


def _reprojection_residuals(
    point: np.ndarray,
    active: list[str],
    cameras: dict[str, CameraCalibration],
    pixel_points: dict[str, np.ndarray],
    keypoint_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked 2D residuals and per-view magnitudes."""
    residuals = np.empty((len(active), 2), dtype=np.float64)
    for i, name in enumerate(active):
        projected = project_points(point[np.newaxis, :], cameras[name])[0]
        residuals[i] = projected - pixel_points[name][keypoint_index]
    errors = np.linalg.norm(residuals, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return residuals, errors


def _truncated_reprojection_loss(
    errors: np.ndarray, view_confidences: np.ndarray, gate_px: float
) -> float:
    clipped = np.minimum(errors, gate_px)
    return float(np.sum(view_confidences * clipped * clipped))


def _batch_reprojection_residuals(
    points: np.ndarray,
    active: list[str],
    cameras: dict[str, CameraCalibration],
    pixel_points: dict[str, np.ndarray],
    keypoint_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch distorted-pixel residuals for points sharing a consensus view set."""
    residuals = np.empty((len(points), len(active), 2), dtype=np.float64)
    for view_index, name in enumerate(active):
        residuals[:, view_index] = (
            project_points(points, cameras[name]) - pixel_points[name][keypoint_indices]
        )
    errors = np.linalg.norm(residuals, axis=2)
    errors[~np.isfinite(errors)] = np.inf
    return residuals, errors


def _batch_numeric_projection_jacobian(points: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Central-difference distorted-pixel Jacobians, vectorised over points and XYZ."""
    epsilon = 1e-5 * np.maximum(1.0, np.linalg.norm(points, axis=1))
    offsets = epsilon[:, np.newaxis, np.newaxis] * np.eye(3, dtype=np.float64)
    samples = np.concatenate(
        [points[:, np.newaxis, :] + offsets, points[:, np.newaxis, :] - offsets], axis=1
    )
    projected = project_points(samples.reshape(-1, 3), camera).reshape(-1, 6, 2)
    derivatives = (projected[:, :3] - projected[:, 3:]) / (2.0 * epsilon[:, np.newaxis, np.newaxis])
    return derivatives.transpose(0, 2, 1)


def _batch_huber_reprojection_loss(
    errors: np.ndarray, view_confidences: np.ndarray, delta_px: float
) -> np.ndarray:
    quadratic = errors <= delta_px
    loss = np.where(
        quadratic,
        0.5 * errors * errors,
        delta_px * (errors - 0.5 * delta_px),
    )
    values = np.sum(view_confidences * loss, axis=1)
    values[~np.isfinite(errors).all(axis=1)] = np.inf
    return values


def _solve_batched_normals(normal: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve independent 3x3 systems without letting one singular item poison a batch."""
    try:
        return np.linalg.solve(normal, rhs[..., np.newaxis])[..., 0]
    except np.linalg.LinAlgError:
        return np.einsum("mij,mj->mi", np.linalg.pinv(normal), rhs)


def _refine_points_geometric(
    initial: np.ndarray,
    active: list[str],
    cameras: dict[str, CameraCalibration],
    pixel_points: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    keypoint_indices: np.ndarray,
    gate_px: float,
    *,
    max_iterations: int = 8,
) -> np.ndarray:
    """Vectorised bounded IRLS refinement in distorted pixel space.

    Keypoints sharing a consensus view set are refined as a batch.  Each
    proposed step is backtracked independently, and a seed is retained unless
    both robust loss and mean geometric reprojection are non-worsening.
    """
    points = np.asarray(initial, dtype=np.float64).copy()
    view_conf = np.column_stack([confidences[name][keypoint_indices] for name in active]).astype(
        np.float64, copy=False
    )
    residuals, errors = _batch_reprojection_residuals(
        points, active, cameras, pixel_points, keypoint_indices
    )
    initial_loss = _batch_huber_reprojection_loss(errors, view_conf, gate_px)
    initial_mean = np.mean(errors, axis=1)
    refinable = np.isfinite(initial_loss) & np.isfinite(initial_mean)
    current_loss = initial_loss.copy()
    current_mean = initial_mean.copy()

    for _ in range(max_iterations):
        jacobian = np.concatenate(
            [_batch_numeric_projection_jacobian(points, cameras[name]) for name in active],
            axis=1,
        )
        residual_vector = residuals.reshape(len(points), -1)
        robust_scale = np.ones_like(errors)
        beyond_gate = errors > gate_px
        robust_scale[beyond_gate] = gate_px / errors[beyond_gate]
        row_scale = np.repeat(np.sqrt(view_conf * robust_scale), 2, axis=1)
        weighted_jacobian = np.zeros_like(jacobian)
        np.multiply(
            jacobian,
            row_scale[:, :, np.newaxis],
            out=weighted_jacobian,
            where=row_scale[:, :, np.newaxis] > 0.0,
        )
        weighted_residual = np.zeros_like(residual_vector)
        np.multiply(
            residual_vector,
            row_scale,
            out=weighted_residual,
            where=row_scale > 0.0,
        )
        normal = np.einsum("mri,mrj->mij", weighted_jacobian, weighted_jacobian)
        rhs = -np.einsum("mri,mr->mi", weighted_jacobian, weighted_residual)
        damping = 1e-8 * np.maximum(1.0, np.trace(normal, axis1=1, axis2=2) / 3.0)
        normal += damping[:, np.newaxis, np.newaxis] * np.eye(3, dtype=np.float64)
        delta = _solve_batched_normals(normal, rhs)

        delta_norm = np.linalg.norm(delta, axis=1)
        max_step = 0.25 * np.maximum(1.0, np.linalg.norm(points, axis=1))
        scale = np.minimum(1.0, max_step / np.maximum(delta_norm, 1e-15))
        delta *= scale[:, np.newaxis]
        viable = refinable & np.isfinite(delta).all(axis=1) & (delta_norm >= 1e-8)
        if not viable.any():
            break

        accepted = np.zeros(len(points), dtype=bool)
        next_points = points.copy()
        next_residuals = residuals.copy()
        next_errors = errors.copy()
        next_loss = current_loss.copy()
        next_mean = current_mean.copy()
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = points + fraction * delta
            candidate_residuals, candidate_errors = _batch_reprojection_residuals(
                candidate, active, cameras, pixel_points, keypoint_indices
            )
            candidate_loss = _batch_huber_reprojection_loss(candidate_errors, view_conf, gate_px)
            candidate_mean = np.mean(candidate_errors, axis=1)
            take = (
                viable
                & ~accepted
                & np.isfinite(candidate_loss)
                & (candidate_loss <= current_loss + 1e-12)
                & (candidate_mean <= current_mean + 1e-12)
            )
            next_points[take] = candidate[take]
            next_residuals[take] = candidate_residuals[take]
            next_errors[take] = candidate_errors[take]
            next_loss[take] = candidate_loss[take]
            next_mean[take] = candidate_mean[take]
            accepted |= take
        if not accepted.any():
            break
        points = next_points
        residuals = next_residuals
        errors = next_errors
        current_loss = next_loss
        current_mean = next_mean

    improved = (
        np.isfinite(current_loss)
        & np.isfinite(current_mean)
        & (current_loss <= initial_loss + 1e-12)
        & (current_mean <= initial_mean + 1e-12)
    )
    points[~improved] = initial[~improved]
    return points


def fuse_session_frame(
    per_camera_keypoints: dict[str, np.ndarray],
    calibration: SessionCalibration,
    *,
    confidences: dict[str, np.ndarray] | None = None,
    min_views: int = 2,
    min_confidence: float = 0.0,
    max_view_reproj_px: float = 20.0,
    min_triangulation_angle_deg: float = 1.0,
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
    2. Every valid two-view minimal set proposes a point.  Hypotheses are
       ranked deterministically by reprojection-inlier cardinality,
       contributing confidence, cheirality, then truncated reprojection
       loss.  Fewer than ``min_views`` consensus inliers → ``NaN``.
    3. The winning consensus is refit with confidence-weighted DLT, then
       refined for at most eight bounded iterations in the original
       distorted pixel frame.  A final residual above
       ``max_view_reproj_px`` invalidates the result.
    4. The maximum acute pairwise viewing-ray angle must be at least
       ``min_triangulation_angle_deg`` (default 1 degree); smaller angles
       are too ill-conditioned for reliable depth.
    5. Cheirality: the fused point is flagged ``ok`` only when it lies
       in front (camera-frame ``Z > 0``) of every contributing camera.

    Returns ``(world, diag)``: ``(N, 3)`` world-space points (metres,
    ``NaN`` rows where fusion failed) and per-keypoint
    ``FusionDiagnostics``.  Reprojection errors are measured in the
    original distorted pixel frame.
    """
    if not per_camera_keypoints:
        raise ValueError("fuse_session_frame: per_camera_keypoints is empty")
    if isinstance(min_views, bool) or not isinstance(min_views, (int, np.integer)):
        raise ValueError(f"fuse_session_frame: min_views must be an integer (got {min_views!r})")
    if min_views < 2:
        raise ValueError(f"fuse_session_frame: min_views must be >= 2 (got {min_views})")
    if not _is_finite_number(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            "fuse_session_frame: min_confidence must be finite and within [0, 1] "
            f"(got {min_confidence!r})"
        )
    if not _is_finite_number(max_view_reproj_px) or max_view_reproj_px <= 0.0:
        raise ValueError(
            "fuse_session_frame: max_view_reproj_px must be finite and > 0 "
            f"(got {max_view_reproj_px!r})"
        )
    if (
        not _is_finite_number(min_triangulation_angle_deg)
        or not 0.0 <= min_triangulation_angle_deg <= 90.0
    ):
        raise ValueError(
            "fuse_session_frame: min_triangulation_angle_deg must be finite and within "
            f"[0, 90] (got {min_triangulation_angle_deg!r})"
        )
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

    if confidences is not None:
        unknown_conf = sorted(set(confidences) - set(names))
        if unknown_conf:
            raise ValueError(
                f"fuse_session_frame: confidence cameras have no keypoints: {unknown_conf}"
            )

    conf: dict[str, np.ndarray] = {}
    for name in names:
        if confidences is not None and name in confidences:
            c = np.asarray(confidences[name], dtype=np.float64)
            if c.shape != (n_kps,):
                raise ValueError(
                    f"fuse_session_frame: confidences[{name!r}] has shape {c.shape}, "
                    f"expected ({n_kps},)"
                )
            if not np.isfinite(c).all():
                raise ValueError(
                    f"fuse_session_frame: confidences[{name!r}] must contain only finite values"
                )
            c = np.clip(c, 0.0, 1.0)
        else:
            c = np.ones(n_kps, dtype=np.float64)
        conf[name] = c

    # Per-view validity → weights; undistort and back-project only valid points.
    undist: dict[str, np.ndarray] = {}
    rays: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for name in names:
        pts = pixel_pts[name]
        valid = np.isfinite(pts).all(axis=1) & (conf[name] > min_confidence)
        u = np.full((n_kps, 2), np.nan, dtype=np.float64)
        r = np.full((n_kps, 3), np.nan, dtype=np.float64)
        if valid.any():
            u[valid] = undistort_points(pts[valid], cams[name])
            r[valid] = _backproject_rays(pts[valid], cams[name])
        undist[name] = u
        rays[name] = r
        weights[name] = np.where(valid, conf[name], 0.0)

    Ps = {n: projection_matrix(cams[n]) for n in names}
    Rts = {n: _rotation_translation(cams[n]) for n in names}

    def _triangulate_keypoint(k: int, active: list[str]) -> np.ndarray:
        return triangulate_views(
            [Ps[n] for n in active],
            [undist[n][k : k + 1] for n in active],
            [weights[n][k : k + 1] for n in active],
        )[0]

    world = np.full((n_kps, 3), np.nan, dtype=np.float64)
    candidate_n_views = np.zeros(n_kps, dtype=np.int64)
    n_views = np.zeros(n_kps, dtype=np.int64)
    mean_conf = np.zeros(n_kps, dtype=np.float64)
    reproj = np.full(n_kps, np.nan, dtype=np.float64)
    cheirality = np.zeros(n_kps, dtype=bool)
    triangulation_angle = np.full(n_kps, np.nan, dtype=np.float64)
    refit_seeds = np.full((n_kps, 3), np.nan, dtype=np.float64)
    consensus_by_keypoint: list[tuple[str, ...] | None] = [None] * n_kps

    for k in range(n_kps):
        active = [n for n in names if weights[n][k] > 0.0]
        candidate_n_views[k] = len(active)
        n_views[k] = len(active)
        if len(active) < min_views:
            continue

        active_conf = np.asarray([conf[name][k] for name in active], dtype=np.float64)
        best_score: tuple[int, float, int, float] | None = None
        best_consensus: list[str] = []
        for pair in itertools.combinations(active, 2):
            hypothesis = _triangulate_keypoint(k, list(pair))
            if not np.isfinite(hypothesis).all():
                continue
            _residuals, errors = _reprojection_residuals(hypothesis, active, cams, pixel_pts, k)
            inlier_mask = errors <= max_view_reproj_px
            inliers = [name for name, keep in zip(active, inlier_mask, strict=True) if keep]
            inlier_confidence = float(np.sum(active_conf[inlier_mask]))
            cheiral_count = sum(
                (Rts[name][0] @ hypothesis + Rts[name][1])[2] > 0.0 for name in inliers
            )
            robust_loss = _truncated_reprojection_loss(errors, active_conf, max_view_reproj_px)
            score = (len(inliers), inlier_confidence, cheiral_count, -robust_loss)
            if best_score is None or score > best_score:
                best_score = score
                best_consensus = inliers

        if len(best_consensus) < min_views:
            continue

        n_views[k] = len(best_consensus)
        angle = _max_acute_ray_angle_deg([rays[name][k] for name in best_consensus])
        triangulation_angle[k] = angle
        if not math.isfinite(angle) or angle < min_triangulation_angle_deg:
            continue

        refit = _triangulate_keypoint(k, best_consensus)
        if not np.isfinite(refit).all():
            continue
        refit_seeds[k] = refit
        consensus_by_keypoint[k] = tuple(best_consensus)

    consensus_groups: dict[tuple[str, ...], list[int]] = {}
    for keypoint_index, consensus_key in enumerate(consensus_by_keypoint):
        if consensus_key is not None:
            consensus_groups.setdefault(consensus_key, []).append(keypoint_index)

    for consensus_key, grouped_indices in consensus_groups.items():
        indices = np.asarray(grouped_indices, dtype=np.intp)
        active = list(consensus_key)
        refined = _refine_points_geometric(
            refit_seeds[indices],
            active,
            cams,
            pixel_pts,
            conf,
            indices,
            max_view_reproj_px,
        )
        _residuals, final_errors = _batch_reprojection_residuals(
            refined, active, cams, pixel_pts, indices
        )
        accepted = np.isfinite(final_errors).all(axis=1) & (
            np.max(final_errors, axis=1) <= max_view_reproj_px
        )
        accepted_indices = indices[accepted]
        world[accepted_indices] = refined[accepted]
        reproj[accepted_indices] = np.mean(final_errors[accepted], axis=1)
        mean_conf[accepted_indices] = np.mean(
            np.column_stack([conf[name][accepted_indices] for name in active]), axis=1
        )
        for keypoint_index in accepted_indices:
            cheirality[keypoint_index] = all(
                (Rts[name][0] @ world[keypoint_index] + Rts[name][1])[2] > 0.0 for name in active
            )

    diag = FusionDiagnostics(
        candidate_n_views=candidate_n_views,
        n_views=n_views,
        confidence=mean_conf,
        reprojection_error_px=reproj,
        cheirality_ok=cheirality,
        triangulation_angle_deg=triangulation_angle,
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
