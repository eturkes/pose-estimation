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
    """Enforce temporal bone-length consistency per tracked body.

    Maintains an exponential moving average of each bone segment length.
    When the measured length deviates from the running average beyond
    *tolerance*, the correction is split between both endpoints
    (weighted by *distal_weight* toward the distal keypoint).
    Corrections propagate outward (shoulder → elbow → wrist → finger
    base).

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
    """

    def __init__(self, alpha=None, tolerance=None, segments=None, distal_weight=None):
        if alpha is None:
            alpha = float(os.environ.get("POSE_BENCH_BONE_EMA_ALPHA", "0.05"))
        if tolerance is None:
            tolerance = float(os.environ.get("POSE_BENCH_BONE_TOLERANCE", "0.4"))
        if distal_weight is None:
            distal_weight = float(os.environ.get("POSE_BENCH_BONE_DISTAL_WEIGHT", "0.8"))
        self.alpha = alpha
        self.tolerance = tolerance
        self.distal_weight = distal_weight
        self.segments = segments if segments is not None else BONE_SEGMENTS
        # Pre-stash proximal / distal index arrays for vectorised lookups.
        self._seg_p_idx = np.array([p for p, _ in self.segments], dtype=np.intp)
        self._seg_d_idx = np.array([d for _, d in self.segments], dtype=np.intp)
        self._averages = {}  # body_id -> np.array of average lengths

    def update(self, body_id, landmarks):
        """Apply bone-length correction to *landmarks* in-place.

        Parameters
        ----------
        body_id : int
            Stable identifier for the tracked body (e.g. track index).
        landmarks : np.ndarray
            Shape (N, 3) keypoints in pixel space.  Modified in-place.

        Returns
        -------
        tuple of (np.ndarray, float)
            The (possibly corrected) landmarks array and the total
            correction magnitude in pixels (sum of distal-keypoint
            displacements).
        """
        seg_p = self._seg_p_idx
        seg_d = self._seg_d_idx

        # Vectorised initial segment lengths: one fancy-index + one
        # einsum reduces 10 ``np.linalg.norm`` calls to a single pass.
        diffs = landmarks[seg_d] - landmarks[seg_p]
        lengths = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))

        if body_id not in self._averages:
            self._averages[body_id] = lengths.copy()
            return landmarks, 0.0

        avg = self._averages[body_id]
        # avg = alpha * lengths + (1 - alpha) * avg, in place
        avg *= 1.0 - self.alpha
        avg += self.alpha * lengths

        tolerance = self.tolerance
        distal_weight = self.distal_weight
        prox_weight = 1.0 - distal_weight
        eps = 1e-6

        # Vectorised violated mask using *initial* lengths (matches the original
        # semantics which compared pre-correction lengths against the EMA).
        valid = avg > eps
        violated_mask = valid & (np.abs(lengths - avg) > tolerance * avg)
        if not violated_mask.any():
            return landmarks, 0.0

        total_correction = 0.0
        for i in np.flatnonzero(violated_mask):
            p = int(seg_p[i])
            d = int(seg_d[i])
            expected = float(avg[i])

            # Recompute direction from current landmarks — prior corrections in
            # this loop may have moved the shared endpoint of an earlier segment.
            ddx = float(landmarks[d, 0] - landmarks[p, 0])
            ddy = float(landmarks[d, 1] - landmarks[p, 1])
            ddz = float(landmarks[d, 2] - landmarks[p, 2])
            norm = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            if norm < eps:
                continue

            # overshoot vector = (direction) * (norm - expected); equivalently
            # scale the (ddx, ddy, ddz) raw vector by (norm - expected)/norm.
            diff_n = norm - expected
            scale = diff_n / norm
            ox = ddx * scale
            oy = ddy * scale
            oz = ddz * scale

            landmarks[d, 0] -= distal_weight * ox
            landmarks[d, 1] -= distal_weight * oy
            landmarks[d, 2] -= distal_weight * oz
            landmarks[p, 0] += prox_weight * ox
            landmarks[p, 1] += prox_weight * oy
            landmarks[p, 2] += prox_weight * oz

            # |distal_change| + |proximal_change| = |overshoot| = |norm - expected|
            total_correction += abs(diff_n)

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
    (0, 2, 4): (30, 170),  # left elbow
    (1, 3, 5): (30, 170),  # right elbow
}

# ---------------------------------------------------------------------------
# Joint-angle limits — 33-keypoint full body scheme
# ---------------------------------------------------------------------------

ANGLE_LIMITS_BODY = {
    (11, 13, 15): (30, 170),  # left elbow
    (12, 14, 16): (30, 170),  # right elbow
    (23, 25, 27): (30, 170),  # left knee
    (24, 26, 28): (30, 170),  # right knee
}


def clamp_joint_angles(landmarks, limits=None):
    """Clamp joint angles to anatomically plausible ranges.

    For each joint triplet, compute the 2D angle at the middle keypoint.
    If the angle falls outside [min, max], rotate the distal keypoint
    around the joint to the nearest limit while preserving segment length.

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

    Returns
    -------
    tuple of (np.ndarray, int)
        The (possibly corrected) landmarks array and the number of
        joint angles that were clamped.
    """
    if limits is None:
        limits = ANGLE_LIMITS

    n_clamped = 0
    eps = 1e-6

    for (prox, joint, dist), (min_deg, max_deg) in limits.items():
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
        inv_l1 = 1.0 / len_v1
        cos_t = math.cos(target_signed)
        sin_t = math.sin(target_signed)
        # Rotate the unit v1 direction by target_signed, scale to len_v2.
        dir_x = (v1x * cos_t - v1y * sin_t) * inv_l1
        dir_y = (v1x * sin_t + v1y * cos_t) * inv_l1

        landmarks[dist, 0] = jx + dir_x * len_v2
        landmarks[dist, 1] = jy + dir_y * len_v2

    return landmarks, n_clamped
