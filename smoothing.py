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
    Body uses heavier smoothing (min_cutoff=1.0); hands use lighter
    smoothing (min_cutoff=3.0) for fast finger movements.
    """

    def __init__(self, match_threshold=150):
        self.match_threshold = match_threshold
        self.body_tracks = []
        self.hand_tracks = []
        self._n_active_hands = 0

    def _match_and_smooth(self, tracks, landmarks, get_anchor, new_filter_fn,
                          t, grace=0, max_tracks=None):
        """Match landmarks to existing tracks, smooth, and return.

        Each track is a 4-tuple: (filter, anchor, age, misses).
        *age* counts consecutive matched frames.  *misses* counts
        consecutive frames without a match.  Unmatched tracks survive
        up to *grace* missed frames so their age is preserved when the
        detection briefly drops out.

        When *max_tracks* is set, no new tracks are created once the
        total number of tracks (active + dormant) reaches the limit.
        Detections that cannot match an existing track are discarded.
        """
        smoothed = []
        new_tracks = []
        used = set()

        for lm in landmarks:
            anchor = get_anchor(lm)
            best_i, best_d = None, float('inf')
            for i, (filt, prev_anchor, age, misses) in enumerate(tracks):
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
            new_tracks.append((filt, get_anchor(s).copy(), age, 0))
            smoothed.append(s)

        # Carry forward unmatched tracks within grace period
        for i, (filt, prev_anchor, age, misses) in enumerate(tracks):
            if i not in used and misses < grace:
                new_tracks.append((filt, prev_anchor, age, misses + 1))

        return new_tracks, smoothed

    def hand_track_ages(self):
        """Return the age (in frames) of each active hand track.

        Length matches the most recent ``smooth_hands`` output.
        """
        return [age for _, _, age, _ in
                self.hand_tracks[:self._n_active_hands]]

    def smooth_bodies(self, body_landmarks, body_visibilities, t):
        if not body_landmarks:
            self.body_tracks = []
            return [], []

        self.body_tracks, smoothed = self._match_and_smooth(
            self.body_tracks, body_landmarks,
            get_anchor=lambda lm: (lm[0, :2] + lm[1, :2]) / 2,
            new_filter_fn=lambda: OneEuroFilter(min_cutoff=1.0, beta=0.5),
            t=t,
        )
        return smoothed, body_visibilities

    def smooth_hands(self, hand_landmarks, t, grace=10, max_tracks=None):
        self.hand_tracks, smoothed = self._match_and_smooth(
            self.hand_tracks, hand_landmarks or [],
            get_anchor=lambda lm: lm[0, :2],
            new_filter_fn=lambda: OneEuroFilter(min_cutoff=3.0, beta=0.3),
            t=t,
            grace=grace,
            max_tracks=max_tracks,
        )
        self._n_active_hands = len(smoothed)
        return smoothed
