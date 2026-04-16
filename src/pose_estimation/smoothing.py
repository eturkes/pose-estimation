"""Temporal smoothing for pose landmarks using One Euro Filters."""

import os

import numpy as np
from scipy.optimize import linear_sum_assignment


class OneEuroFilter:
    """One Euro Filter for smoothing noisy real-time signals.

    Adapts cutoff frequency based on signal speed: slow movements are smoothed
    aggressively while fast movements pass through with minimal lag.
    Works on numpy arrays of any shape.

    Optionally accepts per-keypoint confidence scores to modulate smoothing:
    low-confidence keypoints are pulled toward the previous estimate while
    high-confidence keypoints pass through with standard filtering.
    """

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0, gamma=2.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.gamma = gamma
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t, confidence=None):
        if self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = max(t - self.t_prev, 1e-6)

        a_d = 1.0 / (1.0 + 1.0 / (2 * np.pi * self.d_cutoff * dt))
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))
        x_hat = a * x + (1 - a) * self.x_prev

        # Confidence weighting: low-confidence keypoints are pulled toward
        # the previous position, resisting noisy input.
        if confidence is not None:
            w = np.clip(confidence, 0.0, 1.0)[:, None] ** self.gamma
            result = w * x_hat + (1 - w) * self.x_prev
        else:
            result = x_hat

        self.x_prev = result.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t

        return result


class PoseSmoother:
    """Temporal smoothing for body and hand landmarks.

    Tracks detections across frames by anchor point proximity and applies
    One Euro Filters to reduce jitter while preserving responsiveness.
    Body uses heavier smoothing (min_cutoff=0.3) with confidence-weighted
    blending from per-keypoint visibility scores; hands use moderate
    smoothing (min_cutoff=1.0) for fast finger movements.

    During brief detection dropouts (carry-forward), body tracks
    extrapolate using the last velocity estimate from the One Euro Filter
    with exponential damping, producing smoother motion continuity than
    static replay.
    """

    def __init__(self, match_threshold=150, carry_damping=None):
        self.match_threshold = match_threshold
        if carry_damping is None:
            carry_damping = float(
                os.environ.get("POSE_BENCH_CARRY_DAMPING", "0.8"))
        self.carry_damping = carry_damping
        self.body_tracks = []
        self.hand_tracks = []
        self._n_active_bodies = 0
        self._n_active_hands = 0

    def _match_and_smooth(self, tracks, landmarks, get_anchor, new_filter_fn,
                          t, grace=0, max_tracks=None, emit_carry=False,
                          confidences=None, static_carry=False):
        """Match landmarks to existing tracks, smooth, and return.

        Each track is a 7-tuple:
            (filter, anchor, age, misses, last_output, last_velocity, last_t).
        *age* counts consecutive matched frames.  *misses* counts
        consecutive frames without a match.  Unmatched tracks survive
        up to *grace* missed frames so their filter state (and age) is
        preserved when the detection briefly drops out.

        When *emit_carry* is True, tracks in their grace period
        extrapolate using their last velocity estimate (with exponential
        damping) so the skeleton moves naturally during brief detection
        dropouts.  If no velocity is available (first-frame track), the
        last output is emitted unchanged (static carry).

        When *static_carry* is True (implies *emit_carry*), carried
        tracks always emit their last output unchanged — no velocity
        extrapolation.  Useful for hands where per-finger extrapolation
        is unreliable.

        When *max_tracks* is set, no new tracks are created once the
        total number of tracks (active + dormant) reaches the limit.
        Detections that cannot match an existing track are discarded.

        *confidences* is an optional list parallel to *landmarks*; each
        entry is a 1-D array of per-keypoint confidence scores passed to
        the One Euro Filter for confidence-weighted blending.

        Returns (new_tracks, smoothed, n_active) where *n_active* is
        the count of freshly matched landmarks (excludes carry-forward
        entries appended when *emit_carry* is True).
        """
        if static_carry:
            emit_carry = True
        smoothed = []
        new_tracks = []
        used = set()

        # --- Hungarian (optimal) landmark-to-track assignment ---
        n_lm = len(landmarks)
        n_tr = len(tracks)
        lm_to_track = {}  # landmark index -> track index

        if n_lm > 0 and n_tr > 0:
            anchors_lm = np.array([get_anchor(lm) for lm in landmarks])
            anchors_tr = np.array([tr[1] for tr in tracks])
            # Cost matrix: Euclidean distance between each (landmark, track)
            diff = anchors_lm[:, None, :] - anchors_tr[None, :, :]
            cost = np.linalg.norm(diff, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.match_threshold:
                    lm_to_track[r] = c
                    used.add(c)

        for lm_idx, lm in enumerate(landmarks):
            if lm_idx in lm_to_track:
                tr_idx = lm_to_track[lm_idx]
                filt = tracks[tr_idx][0]
                age = tracks[tr_idx][2] + 1
            elif max_tracks is None or len(new_tracks) < max_tracks:
                filt = new_filter_fn()
                age = 1
            else:
                continue

            conf = confidences[lm_idx] if confidences is not None else None
            s = filt(lm, t, confidence=conf)
            velocity = filt.dx_prev.copy() if filt.dx_prev is not None else None
            new_tracks.append(
                (filt, get_anchor(s).copy(), age, 0, s.copy(), velocity, t))
            smoothed.append(s)

        n_active = len(smoothed)

        # Carry forward unmatched tracks within grace period.
        # Decrement age each missed frame so intermittent false
        # positives cannot accumulate age across grace gaps.
        for i, (filt, prev_anchor, age, misses, last_out,
                last_vel, last_t) in enumerate(tracks):
            if i in used or misses >= grace:
                continue
            new_misses = misses + 1
            decayed_age = max(0, age - 1)
            if emit_carry and last_out is not None:
                if static_carry:
                    predicted = last_out
                else:
                    predicted = self._extrapolate(
                        last_out, last_vel, last_t, t, new_misses)
                new_anchor = get_anchor(predicted).copy()
                new_tracks.append(
                    (filt, new_anchor, decayed_age, new_misses,
                     predicted.copy(), last_vel, t))
                smoothed.append(predicted)
            else:
                new_tracks.append(
                    (filt, prev_anchor, decayed_age, new_misses,
                     last_out, last_vel, last_t))

        return new_tracks, smoothed, n_active

    def _extrapolate(self, last_output, last_velocity, last_t, t, misses):
        """Velocity-based extrapolation with exponential damping.

        Falls back to static carry when no velocity is available.
        Per-keypoint displacement is capped at match_threshold to
        prevent runaway drift from spurious velocity estimates.
        """
        if last_velocity is None:
            return last_output

        dt = t - last_t
        if dt <= 0:
            return last_output

        damping = self.carry_damping ** misses
        step = last_velocity * dt * damping

        # Cap per-keypoint displacement magnitude
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        scale = np.minimum(1.0, self.match_threshold / norms)
        step *= scale

        return last_output + step

    def body_track_ages(self):
        """Return the age (in frames) of each active body track."""
        return [age for _, _, age, _, _, _, _ in
                self.body_tracks[:self._n_active_bodies]]

    def hand_track_ages(self):
        """Return the age (in frames) of each active hand track."""
        return [age for _, _, age, _, _, _, _ in
                self.hand_tracks[:self._n_active_hands]]

    def smooth_bodies(self, body_landmarks, body_visibilities, t,
                      shoulder_indices=(0, 1)):
        """Smooth body landmarks and return (landmarks, visibilities, n_detected).

        *n_detected* is the number of bodies that were genuinely matched
        (or newly created) this frame — i.e. **not** carry-forward ghosts.
        Callers that need to know whether body detection actually fired
        (e.g. single-subject mode) should inspect this value.

        *shoulder_indices* selects the two keypoints whose midpoint is
        used as the track anchor.  Defaults to ``(0, 1)`` for the
        12-keypoint arm scheme; use ``(11, 12)`` for the 33-keypoint
        full body scheme.
        """
        si = shoulder_indices
        lm_list = body_landmarks or []
        body_mc = float(os.environ.get("POSE_BENCH_BODY_MIN_CUTOFF", "0.3"))
        body_b = float(os.environ.get("POSE_BENCH_BODY_BETA", "0.5"))
        body_g = float(os.environ.get("POSE_BENCH_CONFIDENCE_GAMMA", "2.0"))
        grace = int(os.environ.get("POSE_BENCH_CARRY_GRACE", "10"))
        self.body_tracks, smoothed, n_active = self._match_and_smooth(
            self.body_tracks, lm_list,
            get_anchor=lambda lm: (lm[si[0], :2] + lm[si[1], :2]) / 2,
            new_filter_fn=lambda: OneEuroFilter(
                min_cutoff=body_mc, beta=body_b, gamma=body_g),
            t=t,
            grace=grace,
            emit_carry=True,
            confidences=body_visibilities if lm_list else None,
        )
        self._n_active_bodies = n_active
        # Actively matched bodies use the provided visibility;
        # carried bodies (during detection dropout) assume full visibility.
        vis = list(body_visibilities[:n_active])
        n_carried = len(smoothed) - n_active
        if n_carried > 0:
            n_kp = smoothed[0].shape[0] if smoothed else 12
            vis.extend([np.ones(n_kp)] * n_carried)
        return smoothed, vis, n_active

    def smooth_hands(self, hand_landmarks, t, hand_flags=None,
                     grace=None, max_tracks=None):
        if grace is None:
            grace = int(os.environ.get("POSE_BENCH_CARRY_GRACE", "10"))
        hand_mc = float(os.environ.get("POSE_BENCH_HAND_MIN_CUTOFF", "1.0"))
        hand_b = float(os.environ.get("POSE_BENCH_HAND_BETA", "0.3"))
        hand_g = float(os.environ.get("POSE_BENCH_CONFIDENCE_GAMMA", "2.0"))
        confs = None
        if hand_flags is not None:
            confs = [np.full(21, f) for f in hand_flags]
        self.hand_tracks, smoothed, n_active = self._match_and_smooth(
            self.hand_tracks, hand_landmarks or [],
            get_anchor=lambda lm: lm[0, :2],
            new_filter_fn=lambda: OneEuroFilter(
                min_cutoff=hand_mc, beta=hand_b, gamma=hand_g),
            t=t,
            grace=grace,
            static_carry=True,
            max_tracks=max_tracks,
            confidences=confs,
        )
        self._n_active_hands = n_active
        return smoothed, n_active

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------

    def body_carry_state(self):
        """Return (is_carrying, n_carry_frames) for the first body track.

        A track is "carrying" when its miss counter is > 0 (i.e. it was
        not matched this frame but is still within its grace period).
        """
        if not self.body_tracks:
            return False, 0
        _, _, _, misses, _, _, _ = self.body_tracks[0]
        return misses > 0, misses

    def hand_carry_flags(self):
        """Return a list of bools: whether each active hand track is carrying."""
        return [misses > 0
                for _, _, _, misses, _, _, _ in self.hand_tracks]

    @staticmethod
    def compute_smooth_delta(raw_landmarks, smoothed_landmarks):
        """Sum of per-keypoint L2 distance between raw and smoothed (pixels).

        Returns 0.0 if inputs are None or shape-mismatched.
        """
        if raw_landmarks is None or smoothed_landmarks is None:
            return 0.0
        if len(raw_landmarks) == 0 or len(smoothed_landmarks) == 0:
            return 0.0
        # Compare the first entry (primary body / hand)
        raw = raw_landmarks[0] if isinstance(raw_landmarks, list) else raw_landmarks
        smo = smoothed_landmarks[0] if isinstance(smoothed_landmarks, list) else smoothed_landmarks
        if raw.shape != smo.shape:
            return 0.0
        return float(np.sum(np.linalg.norm(raw[:, :2] - smo[:, :2], axis=1)))
