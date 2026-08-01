"""Biomechanical constraints for landmark plausibility."""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Bone-length consistency — 12-keypoint arm scheme
# ---------------------------------------------------------------------------

# Segment pairs: (proximal_index, distal_index)
# Ordered shoulder→outward so corrections propagate distally.
BONE_SEGMENTS = [
    (0, 2),  # left shoulder → left elbow
    (2, 4),  # left elbow → left wrist
    (4, 6),  # left wrist → left index base
    (1, 3),  # right shoulder → right elbow
    (3, 5),  # right elbow → right wrist
    (5, 7),  # right wrist → right index base
]

# ---------------------------------------------------------------------------
# Bone-length consistency — 33-keypoint full body scheme
# ---------------------------------------------------------------------------

BONE_SEGMENTS_BODY = [
    # Arms (same joints, full-body indices)
    (11, 13),  # left shoulder → left elbow
    (13, 15),  # left elbow → left wrist
    (15, 19),  # left wrist → left index
    (12, 14),  # right shoulder → right elbow
    (14, 16),  # right elbow → right wrist
    (16, 20),  # right wrist → right index
    # Legs
    (23, 25),  # left hip → left knee
    (25, 27),  # left knee → left ankle
    (24, 26),  # right hip → right knee
    (26, 28),  # right knee → right ankle
]


class BoneLengthSmoother:
    """Enforce robust temporal bone-length consistency per tracked body.

    Maintains a clipped exponential moving average of each 2-D bone segment
    length.  Clipping prevents one bad landmark from moving the learned body
    proportions toward the outlier.  When a measured length deviates from the
    pre-update estimate by more than *tolerance*, iterative projections split
    the correction between both endpoints (weighted by *distal_weight* toward
    the distal keypoint).  Repeated passes repair adjacent segments moved by an
    earlier projection while leaving in-tolerance perspective changes alone.

    Only image-space x/y coordinates participate.  MediaPipe's z is scaled
    with the crop into a pixel-like relative-depth value, but it is not a
    calibrated Euclidean coordinate; it therefore must not drive a 2-D
    projected bone-length constraint.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (small = slow adaptation).
    tolerance : float
        Maximum allowed fractional deviation from the running average
        before correction is applied (e.g. 0.4 = 40 %).
    segments : list of (int, int), optional
        Bone segment index pairs.  Defaults to :data:`BONE_SEGMENTS`
        (12-keypoint arm scheme).
    distal_weight : float
        Fraction of the correction applied to the distal keypoint
        (default 0.8).  The remaining ``1 - distal_weight`` is applied
        to the proximal keypoint.
    max_iterations : int
        Maximum linked-segment projection passes per frame (default 8).
    """

    def __init__(
        self,
        alpha=None,
        tolerance=None,
        segments=None,
        distal_weight=None,
        max_iterations=8,
    ):
        if alpha is None:
            alpha = float(os.environ.get("POSE_BENCH_BONE_EMA_ALPHA", "0.05"))
        if tolerance is None:
            tolerance = float(os.environ.get("POSE_BENCH_BONE_TOLERANCE", "0.4"))
        if distal_weight is None:
            distal_weight = float(os.environ.get("POSE_BENCH_BONE_DISTAL_WEIGHT", "0.8"))
        self.alpha = alpha
        self.tolerance = tolerance
        self.distal_weight = distal_weight
        self.max_iterations = max(1, int(max_iterations))
        self.segments = segments if segments is not None else BONE_SEGMENTS
        # Pre-stash proximal / distal index arrays for vectorised lookups.
        self._seg_p_idx = np.array([p for p, _ in self.segments], dtype=np.intp)
        self._seg_d_idx = np.array([d for _, d in self.segments], dtype=np.intp)
        self._averages = {}  # body_id -> np.array of average lengths

    def reset(self):
        """Clear all learned body proportions between independent sources."""
        self._averages = {}

    def update(self, body_id, landmarks, validity=None):
        """Apply bone-length correction to *landmarks* in-place.

        Parameters
        ----------
        body_id : int
            Stable identifier for the tracked body (e.g. track index).
        landmarks : np.ndarray
            Shape (N, 2) or (N, 3) keypoints.  The image-space x/y columns are
            modified in-place; any additional columns are preserved.
        validity : array-like of bool, optional
            Per-keypoint observation validity.  Segments touching an invalid
            point neither update their learned length nor move either endpoint.

        Returns
        -------
        tuple of (np.ndarray, float)
            The (possibly corrected) landmarks array and the total
            correction magnitude in pixels (sum of endpoint displacements).
        """
        seg_p = self._seg_p_idx
        seg_d = self._seg_d_idx

        # x/y are image pixels for both backends. MediaPipe z remains
        # model-relative rather than calibrated world depth.
        xy = landmarks[:, :2]
        if validity is None:
            valid_keypoints = np.ones(landmarks.shape[0], dtype=bool)
        else:
            valid_keypoints = np.asarray(validity, dtype=bool)
            if valid_keypoints.shape != (landmarks.shape[0],):
                raise ValueError(
                    f"validity shape {valid_keypoints.shape} does not match {(landmarks.shape[0],)}"
                )
        diffs = xy[seg_d] - xy[seg_p]
        lengths = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        finite = (
            np.isfinite(xy[seg_p]).all(axis=1)
            & np.isfinite(xy[seg_d]).all(axis=1)
            & valid_keypoints[seg_p]
            & valid_keypoints[seg_d]
            & np.isfinite(lengths)
            & (lengths > 1e-6)
        )

        if body_id not in self._averages:
            initial = np.full_like(lengths, np.nan)
            initial[finite] = lengths[finite]
            self._averages[body_id] = initial
            return landmarks, 0.0

        avg = self._averages[body_id]
        tolerance = self.tolerance
        distal_weight = self.distal_weight
        prox_weight = 1.0 - distal_weight
        eps = 1e-6

        # Initialise segments independently: one missing keypoint must not
        # poison the other bones, and a later valid observation may recover it.
        uninitialised = finite & ~np.isfinite(avg)
        avg[uninitialised] = lengths[uninitialised]

        established = finite & np.isfinite(avg) & ~uninitialised & (avg > eps)
        if not established.any():
            return landmarks, 0.0

        # Keep the pre-update estimate as this frame's projection target.  A
        # winsorised EMA can adapt gradually to genuine scale/perspective
        # changes without learning an arbitrarily large one-frame outlier.
        target = avg.copy()
        lo = np.maximum(eps, target[established] * (1.0 - tolerance))
        hi = target[established] * (1.0 + tolerance)
        clipped = np.clip(lengths[established], lo, hi)
        avg[established] += self.alpha * (clipped - avg[established])

        initially_violated = established & (np.abs(lengths - target) > tolerance * target)
        if not initially_violated.any():
            return landmarks, 0.0

        total_correction = 0.0
        # A stale filtered coordinate may be finite even when its current raw
        # observation is invalid.  Only this frame's established segments are
        # eligible for linked projection.
        repairable = established
        for _ in range(self.max_iterations):
            changed = False
            for i in np.flatnonzero(repairable):
                p = int(seg_p[i])
                d = int(seg_d[i])
                delta = xy[d] - xy[p]
                if not np.isfinite(delta).all():
                    continue
                norm = math.hypot(float(delta[0]), float(delta[1]))
                if norm < eps:
                    continue
                expected = float(target[i])
                diff_n = norm - expected
                if abs(diff_n) <= tolerance * expected:
                    continue

                overshoot = delta * (diff_n / norm)
                xy[d] -= distal_weight * overshoot
                xy[p] += prox_weight * overshoot
                total_correction += abs(diff_n)
                changed = True
            if not changed:
                break

        return landmarks, total_correction

    def prune(self, active_ids):
        """Remove state for body IDs no longer being tracked."""
        stale = set(self._averages) - set(active_ids)
        for bid in stale:
            del self._averages[bid]


# ---------------------------------------------------------------------------
# Joint-angle limits — 12-keypoint arm scheme
# ---------------------------------------------------------------------------

# (proximal, joint, distal): (min_degrees, max_degrees)
ANGLE_LIMITS = {
    (0, 2, 4): (30, 180),  # left elbow
    (1, 3, 5): (30, 180),  # right elbow
}

# ---------------------------------------------------------------------------
# Joint-angle limits — 33-keypoint full body scheme
# ---------------------------------------------------------------------------

ANGLE_LIMITS_BODY = {
    (11, 13, 15): (30, 180),  # left elbow
    (12, 14, 16): (30, 180),  # right elbow
    (23, 25, 27): (30, 180),  # left knee
    (24, 26, 28): (30, 180),  # right knee
}

# A joint correction is a rigid rotation of the complete distal branch.  Moving
# only the wrist/ankle would immediately break the wrist-to-finger or ankle-to-
# foot segments even though their relative geometry was valid.
_DISTAL_BRANCHES = {
    (0, 2, 4): (4, 6, 8, 10),
    (1, 3, 5): (5, 7, 9, 11),
    (11, 13, 15): (15, 17, 19, 21),
    (12, 14, 16): (16, 18, 20, 22),
    (23, 25, 27): (27, 29, 31),
    (24, 26, 28): (28, 30, 32),
}


def clamp_joint_angles(landmarks, limits=None, validity=None):
    """Clamp joint angles to anatomically plausible ranges.

    For each joint triplet, compute the 2D angle at the middle keypoint.
    If the angle falls outside [min, max], rigidly rotate the complete distal
    branch around the joint to the nearest limit.  This preserves downstream
    segment geometry instead of disconnecting a wrist from its finger bases or
    an ankle from its foot landmarks.

    Only x/y coordinates are used for the angle calculation (MediaPipe's
    z is relative depth and metrically unreliable).  The z value of the
    distal keypoint is left unchanged.

    Parameters
    ----------
    landmarks : np.ndarray
        Shape (N, 3) keypoints in pixel space.  Modified in-place.
    limits : dict, optional
        ``{(proximal, joint, distal): (min_deg, max_deg), ...}``.
        Defaults to :data:`ANGLE_LIMITS`.
    validity : array-like of bool, optional
        Per-keypoint observation validity.  Invalid joint triplets are skipped,
        as are invalid landmarks elsewhere on a rotated distal branch.

    Returns
    -------
    tuple of (np.ndarray, int)
        The (possibly corrected) landmarks array and the number of
        joint angles that were clamped.
    """
    if limits is None:
        limits = ANGLE_LIMITS
    if validity is None:
        valid_keypoints = np.ones(landmarks.shape[0], dtype=bool)
    else:
        valid_keypoints = np.asarray(validity, dtype=bool)
        if valid_keypoints.shape != (landmarks.shape[0],):
            raise ValueError(
                f"validity shape {valid_keypoints.shape} does not match {(landmarks.shape[0],)}"
            )

    n_clamped = 0
    eps = 1e-6

    for (prox, joint, dist), (min_deg, max_deg) in limits.items():
        if not (valid_keypoints[prox] and valid_keypoints[joint] and valid_keypoints[dist]):
            continue
        # Scalar-math hot path: avoids per-iteration numpy dispatch for the
        # tiny 2-D vectors involved in each joint angle.
        jx = float(landmarks[joint, 0])
        jy = float(landmarks[joint, 1])
        v1x = float(landmarks[prox, 0]) - jx
        v1y = float(landmarks[prox, 1]) - jy
        v2x = float(landmarks[dist, 0]) - jx
        v2y = float(landmarks[dist, 1]) - jy

        len_v1 = math.hypot(v1x, v1y)
        len_v2 = math.hypot(v2x, v2y)
        if len_v1 < eps or len_v2 < eps:
            continue

        cross = v1x * v2y - v1y * v2x
        dot_val = v1x * v2x + v1y * v2y
        signed_angle = math.atan2(cross, dot_val)
        unsigned_angle = -signed_angle if signed_angle < 0.0 else signed_angle

        min_rad = math.radians(min_deg)
        max_rad = math.radians(max_deg)

        if unsigned_angle < min_rad:
            target = min_rad
        elif unsigned_angle > max_rad:
            target = max_rad
        else:
            continue

        n_clamped += 1

        target_signed = math.copysign(target, signed_angle)
        rotation = target_signed - signed_angle
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        branch = _DISTAL_BRANCHES.get((prox, joint, dist), (dist,))
        for idx in branch:
            if idx >= landmarks.shape[0] or not valid_keypoints[idx]:
                continue
            dx = float(landmarks[idx, 0]) - jx
            dy = float(landmarks[idx, 1]) - jy
            if not (math.isfinite(dx) and math.isfinite(dy)):
                continue
            landmarks[idx, 0] = jx + dx * cos_r - dy * sin_r
            landmarks[idx, 1] = jy + dx * sin_r + dy * cos_r

    return landmarks, n_clamped
