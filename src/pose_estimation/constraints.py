"""Biomechanical constraints for landmark plausibility."""

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

    def __init__(self, alpha=None, tolerance=None, segments=None,
                 distal_weight=None):
        if alpha is None:
            alpha = float(os.environ.get("POSE_BENCH_BONE_EMA_ALPHA", "0.05"))
        if tolerance is None:
            tolerance = float(os.environ.get("POSE_BENCH_BONE_TOLERANCE", "0.4"))
        if distal_weight is None:
            distal_weight = float(
                os.environ.get("POSE_BENCH_BONE_DISTAL_WEIGHT", "0.8"))
        self.alpha = alpha
        self.tolerance = tolerance
        self.distal_weight = distal_weight
        self.segments = segments if segments is not None else BONE_SEGMENTS
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
        lengths = np.array([
            np.linalg.norm(landmarks[d] - landmarks[p])
            for p, d in self.segments
        ])

        if body_id not in self._averages:
            self._averages[body_id] = lengths.copy()
            return landmarks, 0.0

        avg = self._averages[body_id]
        avg[:] = self.alpha * lengths + (1 - self.alpha) * avg

        total_correction = 0.0
        for i, (p, d) in enumerate(self.segments):
            expected = avg[i]
            if expected < 1e-6:
                continue
            deviation = abs(lengths[i] - expected) / expected
            if deviation > self.tolerance:
                direction = landmarks[d] - landmarks[p]
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                direction /= norm
                overshoot = landmarks[d] - (landmarks[p] + direction * expected)
                old_d = landmarks[d].copy()
                old_p = landmarks[p].copy()
                landmarks[d] -= self.distal_weight * overshoot
                landmarks[p] += (1 - self.distal_weight) * overshoot
                total_correction += float(
                    np.linalg.norm(landmarks[d] - old_d)
                    + np.linalg.norm(landmarks[p] - old_p))

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

    for (prox, joint, dist), (min_deg, max_deg) in limits.items():
        v1 = landmarks[prox, :2] - landmarks[joint, :2]
        v2 = landmarks[dist, :2] - landmarks[joint, :2]

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        if len_v1 < 1e-6 or len_v2 < 1e-6:
            continue

        # Signed angle from v1 to v2 (positive = counterclockwise)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot_val = v1[0] * v2[0] + v1[1] * v2[1]
        signed_angle = np.arctan2(cross, dot_val)
        unsigned_angle = abs(signed_angle)

        min_rad = np.radians(min_deg)
        max_rad = np.radians(max_deg)

        if unsigned_angle < min_rad:
            target = min_rad
        elif unsigned_angle > max_rad:
            target = max_rad
        else:
            continue

        n_clamped += 1

        # Rotate the unit v1 direction by the clamped signed angle
        target_signed = np.copysign(target, signed_angle)
        v1_hat = v1 / len_v1
        cos_t = np.cos(target_signed)
        sin_t = np.sin(target_signed)
        new_dir = np.array([
            v1_hat[0] * cos_t - v1_hat[1] * sin_t,
            v1_hat[0] * sin_t + v1_hat[1] * cos_t,
        ])

        landmarks[dist, :2] = landmarks[joint, :2] + new_dir * len_v2

    return landmarks, n_clamped
