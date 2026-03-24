"""Biomechanical constraints for arm landmark plausibility."""

import numpy as np

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


class BoneLengthSmoother:
    """Enforce temporal bone-length consistency per tracked body.

    Maintains an exponential moving average of each bone segment length.
    When the measured length deviates from the running average beyond
    *tolerance*, the distal keypoint is projected along the segment
    direction to the expected length.  Corrections propagate outward
    (shoulder → elbow → wrist → finger base).

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (small = slow adaptation).
    tolerance : float
        Maximum allowed fractional deviation from the running average
        before correction is applied (e.g. 0.4 = 40 %).
    """

    def __init__(self, alpha=0.05, tolerance=0.4):
        self.alpha = alpha
        self.tolerance = tolerance
        self._averages = {}  # body_id -> np.array of average lengths

    def update(self, body_id, landmarks):
        """Apply bone-length correction to *landmarks* (12, 3) in-place.

        Parameters
        ----------
        body_id : int
            Stable identifier for the tracked body (e.g. track index).
        landmarks : np.ndarray
            Shape (12, 3) arm keypoints in pixel space.  Modified in-place.

        Returns
        -------
        np.ndarray
            The (possibly corrected) landmarks array.
        """
        lengths = np.array([
            np.linalg.norm(landmarks[d] - landmarks[p])
            for p, d in BONE_SEGMENTS
        ])

        if body_id not in self._averages:
            self._averages[body_id] = lengths.copy()
            return landmarks

        avg = self._averages[body_id]
        avg[:] = self.alpha * lengths + (1 - self.alpha) * avg

        for i, (p, d) in enumerate(BONE_SEGMENTS):
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
                landmarks[d] = landmarks[p] + direction * expected

        return landmarks

    def prune(self, active_ids):
        """Remove state for body IDs no longer being tracked."""
        stale = set(self._averages) - set(active_ids)
        for bid in stale:
            del self._averages[bid]
