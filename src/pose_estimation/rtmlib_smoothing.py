"""rtmlib-path keypoint smoother for the 133-keypoint COCO-WholeBody layout.

Parallel to ``smoothing.py`` (the MediaPipe-path filters); provides the One
Euro Filter and the multi-person ``KeypointSmoother`` used by the rtmlib path.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# COCO-WholeBody 133 keypoint tracking masks
# ---------------------------------------------------------------------------
_KP_ARMS = {5, 6, 7, 8, 9, 10}  # shoulders, elbows, wrists
_KP_LHAND = set(range(91, 112))  # 21 left-hand landmarks
_KP_RHAND = set(range(112, 133))  # 21 right-hand landmarks

# Per-region smoothing parameters for 133-keypoint COCO-WholeBody layout.
# Hands/fingers get lighter smoothing (higher min_cutoff) to preserve fast
# articulation; body, feet, and face get heavier smoothing.
# (name, start_index, end_index_exclusive, min_cutoff, beta)
REGION_PARAMS = [
    ("body", 0, 17, 0.3, 0.5),
    ("feet", 17, 23, 0.3, 0.5),
    ("face", 23, 91, 0.3, 0.5),
    ("hands", 91, 133, 0.5, 0.3),
]


# ---------------------------------------------------------------------------
# Temporal smoothing — reduces frame-to-frame keypoint jitter
# ---------------------------------------------------------------------------


class _OneEuro:
    """Minimal One Euro Filter for array-valued signals."""

    def __init__(
        self,
        min_cutoff=0.5,
        beta=0.5,
        d_cutoff=1.0,
        gamma=2.0,
        outlier_cap=0.0,
        rest_cutoff=None,
        rest_speed=2.0,
        fast_speed=10.0,
        speed_alpha=0.1,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.gamma = gamma
        self.outlier_cap = outlier_cap
        self.rest_cutoff = rest_cutoff
        self.rest_speed = rest_speed
        self.fast_speed = fast_speed
        self.speed_alpha = speed_alpha
        self._smoothed_speed = None
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t, confidence=None):
        if self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        assert self.x_prev is not None
        assert self.dx_prev is not None
        dt = max(t - self.t_prev, 1e-6)

        # Outlier rejection: cap the unexpected component of displacement.
        x_use = x
        if self.outlier_cap > 0 and x.ndim == 2:
            diff = x - self.x_prev
            predicted_step = self.dx_prev * dt
            unexpected = diff - predicted_step
            norms_sq = np.einsum("ij,ij->i", unexpected, unexpected)
            cap_sq = self.outlier_cap * self.outlier_cap
            mask = norms_sq > cap_sq
            if mask.any():
                norms = np.sqrt(norms_sq)
                s = np.ones(norms.shape)
                np.divide(self.outlier_cap, norms, out=s, where=mask)
                x_use = self.x_prev + predicted_step + unexpected * s[:, None]

        a_d = 1.0 / (1.0 + 1.0 / (2 * np.pi * self.d_cutoff * dt))
        dx = (x_use - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive min_cutoff: during rest, lower the cutoff floor.
        if self.rest_cutoff is not None and x.ndim == 2:
            kp_speed = np.sqrt(np.einsum("ij,ij->i", dx_hat, dx_hat)) * dt
            if self._smoothed_speed is None:
                self._smoothed_speed = kp_speed.copy()
            else:
                a_s = self.speed_alpha
                self._smoothed_speed += a_s * (kp_speed - self._smoothed_speed)
            rng = self.fast_speed - self.rest_speed
            if rng > 0:
                t_interp = np.clip((self._smoothed_speed - self.rest_speed) / rng, 0.0, 1.0)
                effective_mc = self.rest_cutoff + t_interp * (self.min_cutoff - self.rest_cutoff)
            else:
                effective_mc = np.full_like(self._smoothed_speed, self.min_cutoff)
            cutoff = effective_mc[:, None] + self.beta * np.abs(dx_hat)
        else:
            cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        a = 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))
        x_hat = a * x_use + (1 - a) * self.x_prev

        # Confidence weighting: low-confidence keypoints are pulled toward
        # the previous position, resisting noisy input.
        if confidence is not None:
            w = np.clip(confidence, 0.0, 1.0)[:, None] ** self.gamma
            result = w * x_hat + (1 - w) * self.x_prev
        else:
            result = x_hat

        # ``result`` is returned and may be mutated by the caller, so keep
        # a private copy as state.  ``dx_hat`` is fresh and never returned.
        self.x_prev = result.copy()
        self.dx_prev = dx_hat
        self.t_prev = t
        return result


class KeypointSmoother:
    """Multi-person temporal smoother with track matching and carry-forward.

    Reduces jitter via One Euro Filters on keypoint positions and EMA on
    confidence scores.  Greedy nearest-centroid matching associates
    detections with persistent tracks across frames.  During brief
    detection dropouts, tracks carry forward with gradual score decay
    so the skeleton fades rather than vanishing abruptly.
    """

    SCORE_DECAY = 0.9  # per-frame score multiplier during carry-forward

    def __init__(
        self,
        min_cutoff=0.5,
        beta=0.5,
        score_alpha=0.5,
        carry_frames=5,
        match_thresh=150,
        carry_damping=0.8,
        min_track_age=3,
        outlier_cap=30.0,
        rest_cutoff=None,
        hand_rest_cutoff=None,
        rest_speed=2.0,
        fast_speed=10.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.score_alpha = score_alpha
        self.carry_frames = carry_frames
        self.match_thresh = match_thresh
        self.carry_damping = carry_damping
        self.min_track_age = min_track_age
        self.outlier_cap = outlier_cap
        self.rest_cutoff = rest_cutoff
        self.hand_rest_cutoff = hand_rest_cutoff
        self.rest_speed = rest_speed
        self.fast_speed = fast_speed
        self.tracks = []

    def reset(self):
        """Clear all track state (e.g. between video sources)."""
        self.tracks = []

    def _make_filters(self, n_kps):
        """Create per-region or single filter depending on keypoint count."""
        oc = self.outlier_cap
        rs, fs = self.rest_speed, self.fast_speed
        if n_kps == 133:
            return {
                name: _OneEuro(
                    min_cutoff=mc,
                    beta=b,
                    outlier_cap=oc,
                    rest_cutoff=self.hand_rest_cutoff if name == "hands" else self.rest_cutoff,
                    rest_speed=rs,
                    fast_speed=fs,
                )
                for name, _, _, mc, b in REGION_PARAMS
            }
        return {
            "all": _OneEuro(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                outlier_cap=oc,
                rest_cutoff=self.rest_cutoff,
                rest_speed=rs,
                fast_speed=fs,
            )
        }

    def _apply_filters(self, filters, kp, t, confidence):
        """Apply region-aware or single filter to keypoints."""
        if "all" in filters:
            return filters["all"](kp, t, confidence=confidence)
        result = np.empty_like(kp)
        for name, start, end, _, _ in REGION_PARAMS:
            conf_slice = confidence[start:end] if confidence is not None else None
            result[start:end] = filters[name](kp[start:end], t, confidence=conf_slice)
        return result

    def _get_velocity(self, filters):
        """Extract concatenated velocity from region or single filters."""
        if "all" in filters:
            v = filters["all"].dx_prev
            return v.copy() if v is not None else None
        parts = []
        for name, _, _, _, _ in REGION_PARAMS:
            v = filters[name].dx_prev
            if v is None:
                return None
            parts.append(v)
        return np.concatenate(parts, axis=0)

    def _extrapolate(self, last_kps, last_velocity, last_t, t, misses):
        """Velocity-based extrapolation with exponential damping.

        Falls back to static carry when no velocity is available.
        Per-keypoint displacement is capped at match_thresh to
        prevent runaway drift from spurious velocity estimates.
        """
        if last_velocity is None:
            return last_kps
        dt = t - last_t
        if dt <= 0:
            return last_kps
        damping = self.carry_damping**misses
        step = last_velocity * dt * damping
        # Cap per-keypoint displacement magnitude
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        scale = np.minimum(1.0, self.match_thresh / norms)
        step *= scale
        return last_kps + step

    def __call__(self, keypoints, scores, t):
        """Return (smoothed_keypoints, smoothed_scores) or (None, None)."""
        if keypoints is None or len(keypoints.shape) != 3 or keypoints.shape[0] == 0:
            return self._carry(t)

        n_det = keypoints.shape[0]
        det_centroids = keypoints.mean(axis=1)

        matched, used_tracks = self._match(det_centroids)

        new_tracks = []
        out_kps = []
        out_scores = []

        for i in range(n_det):
            kp = keypoints[i]
            sc = scores[i]

            if i in matched:
                tr = self.tracks[matched[i]]
                filt = tr["filter"]
                prev_sc = tr["scores"]
                age = tr["age"] + 1
            else:
                filt = self._make_filters(kp.shape[0])
                prev_sc = sc
                age = 1

            smooth_kp = self._apply_filters(filt, kp, t, sc)
            smooth_sc = self.score_alpha * sc + (1 - self.score_alpha) * prev_sc

            new_tracks.append(
                {
                    "filter": filt,
                    "centroid": smooth_kp.mean(axis=0).copy(),
                    "scores": smooth_sc.copy(),
                    "misses": 0,
                    "age": age,
                    "last_kps": smooth_kp.copy(),
                    "last_velocity": self._get_velocity(filt),
                    "last_t": t,
                }
            )
            if age >= self.min_track_age:
                out_kps.append(smooth_kp)
                out_scores.append(smooth_sc)

        # Carry forward unmatched tracks within grace period.
        # Decrement age each missed frame so intermittent false
        # positives cannot accumulate age across grace gaps.
        for j, tr in enumerate(self.tracks):
            if j in used_tracks or tr["misses"] >= self.carry_frames:
                continue
            misses = tr["misses"] + 1
            age = max(0, tr["age"] - 1)
            predicted = self._extrapolate(
                tr["last_kps"], tr.get("last_velocity"), tr.get("last_t", 0), t, misses
            )
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append(
                {
                    "filter": tr["filter"],
                    "centroid": predicted.mean(axis=0).copy(),
                    "scores": decayed,
                    "misses": misses,
                    "age": age,
                    "last_kps": predicted,
                    "last_velocity": tr.get("last_velocity"),
                    "last_t": t,
                }
            )
            if age >= self.min_track_age:
                out_kps.append(predicted)
                out_scores.append(decayed)

        self.tracks = new_tracks
        if out_kps:
            return np.stack(out_kps), np.stack(out_scores)
        return None, None

    def _match(self, det_centroids):
        """Optimal nearest-centroid matching via Hungarian algorithm."""
        matched = {}
        used_tracks = set()
        if not self.tracks or len(det_centroids) == 0:
            return matched, used_tracks

        trk_c = np.array([tr["centroid"] for tr in self.tracks])
        cost = np.linalg.norm(det_centroids[:, None, :] - trk_c[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind, strict=False):
            if cost[r, c] < self.match_thresh:
                matched[int(r)] = int(c)
                used_tracks.add(int(c))

        return matched, used_tracks

    def _carry(self, t=None):
        """Emit carry-forward tracks when no detections are present."""
        new_tracks = []
        out_kps = []
        out_scores = []
        for tr in self.tracks:
            if tr["misses"] >= self.carry_frames:
                continue
            misses = tr["misses"] + 1
            age = max(0, tr["age"] - 1)
            if t is not None:
                predicted = self._extrapolate(
                    tr["last_kps"], tr.get("last_velocity"), tr.get("last_t", 0), t, misses
                )
            else:
                predicted = tr["last_kps"]
            decayed = tr["scores"] * self.SCORE_DECAY
            new_tracks.append(
                {
                    "filter": tr["filter"],
                    "centroid": predicted.mean(axis=0).copy(),
                    "scores": decayed,
                    "misses": misses,
                    "age": age,
                    "last_kps": predicted,
                    "last_velocity": tr.get("last_velocity"),
                    "last_t": t if t is not None else tr.get("last_t", 0),
                }
            )
            if age >= self.min_track_age:
                out_kps.append(predicted)
                out_scores.append(decayed)
        self.tracks = new_tracks
        if out_kps:
            return np.stack(out_kps), np.stack(out_scores)
        return None, None
