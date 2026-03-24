"""Temporal smoothing for pose landmarks using One Euro Filters."""

import numpy as np


class OneEuroFilter:
    """One Euro Filter for smoothing noisy real-time signals.

    Adapts cutoff frequency based on signal speed: slow movements are smoothed
    aggressively while fast movements pass through with minimal lag.
    Works on numpy arrays of any shape.
    """

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
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

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t

        return x_hat


class PoseSmoother:
    """Temporal smoothing for body and hand landmarks.

    Tracks detections across frames by anchor point proximity and applies
    One Euro Filters to reduce jitter while preserving responsiveness.
    Body uses heavier smoothing (min_cutoff=0.3); hands use moderate
    smoothing (min_cutoff=1.0) for fast finger movements.
    """

    def __init__(self, match_threshold=250):
        self.match_threshold = match_threshold
        self.body_tracks = []
        self.hand_tracks = []
        self._n_active_hands = 0

    def _match_and_smooth(self, tracks, landmarks, get_anchor, new_filter_fn,
                          t, grace=0, max_tracks=None, emit_carry=False):
        """Match landmarks to existing tracks, smooth, and return.

        Each track is a 5-tuple: (filter, anchor, age, misses, last_output).
        *age* counts consecutive matched frames.  *misses* counts
        consecutive frames without a match.  Unmatched tracks survive
        up to *grace* missed frames so their filter state (and age) is
        preserved when the detection briefly drops out.

        When *emit_carry* is True, tracks in their grace period emit
        their last smoothed output so the skeleton remains visible
        during brief detection dropouts.

        When *max_tracks* is set, no new tracks are created once the
        total number of tracks (active + dormant) reaches the limit.
        Detections that cannot match an existing track are discarded.

        Returns (new_tracks, smoothed, n_active) where *n_active* is
        the count of freshly matched landmarks (excludes carry-forward
        entries appended when *emit_carry* is True).
        """
        smoothed = []
        new_tracks = []
        used = set()

        for lm in landmarks:
            anchor = get_anchor(lm)
            best_i, best_d = None, float('inf')
            for i, (filt, prev_anchor, age, misses, _) in enumerate(tracks):
                if i in used:
                    continue
                d = np.linalg.norm(anchor - prev_anchor)
                if d < best_d:
                    best_d = d
                    best_i = i

            if best_i is not None and best_d < self.match_threshold:
                filt = tracks[best_i][0]
                age = tracks[best_i][2] + 1
                used.add(best_i)
            elif max_tracks is None or len(tracks) < max_tracks:
                filt = new_filter_fn()
                age = 1
            else:
                continue

            s = filt(lm, t)
            new_tracks.append((filt, get_anchor(s).copy(), age, 0, s.copy()))
            smoothed.append(s)

        n_active = len(smoothed)

        # Carry forward unmatched tracks within grace period
        for i, (filt, prev_anchor, age, misses, last_out) in enumerate(tracks):
            if i not in used and misses < grace:
                new_tracks.append((filt, prev_anchor, age, misses + 1, last_out))
                if emit_carry and last_out is not None:
                    smoothed.append(last_out)

        return new_tracks, smoothed, n_active

    def hand_track_ages(self):
        """Return the age (in frames) of each active hand track.

        Length matches the most recent ``smooth_hands`` output.
        """
        return [age for _, _, age, _, _ in
                self.hand_tracks[:self._n_active_hands]]

    def smooth_bodies(self, body_landmarks, body_visibilities, t):
        self.body_tracks, smoothed, n_active = self._match_and_smooth(
            self.body_tracks, body_landmarks or [],
            get_anchor=lambda lm: (lm[0, :2] + lm[1, :2]) / 2,
            new_filter_fn=lambda: OneEuroFilter(min_cutoff=0.3, beta=0.5),
            t=t,
            grace=10,
            emit_carry=True,
        )
        # Actively matched bodies use the provided visibility;
        # carried bodies (during detection dropout) assume full visibility.
        vis = list(body_visibilities[:n_active])
        n_carried = len(smoothed) - n_active
        if n_carried > 0:
            n_kp = smoothed[0].shape[0] if smoothed else 12
            vis.extend([np.ones(n_kp)] * n_carried)
        return smoothed, vis

    def smooth_hands(self, hand_landmarks, t, grace=10, max_tracks=None):
        self.hand_tracks, smoothed, _ = self._match_and_smooth(
            self.hand_tracks, hand_landmarks or [],
            get_anchor=lambda lm: lm[0, :2],
            new_filter_fn=lambda: OneEuroFilter(min_cutoff=1.0, beta=0.3),
            t=t,
            grace=grace,
            max_tracks=max_tracks,
        )
        self._n_active_hands = len(smoothed)
        return smoothed
