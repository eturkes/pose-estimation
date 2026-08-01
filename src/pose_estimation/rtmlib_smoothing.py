"""rtmlib-path keypoint smoother for the 133-keypoint COCO-WholeBody layout.

Provides the multi-person ``KeypointSmoother`` used by the rtmlib path; the
One Euro Filter itself is the shared ``smoothing.OneEuroFilter``.
"""

import numpy as np

from .assignment import gated_assignment
from .smoothing import OneEuroFilter

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


class KeypointSmoother:
    """Multi-person temporal smoother with track matching and carry-forward.

    Reduces jitter via One Euro Filters on keypoint positions and EMA on
    confidence scores.  Validity-gated Hungarian matching associates
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
        self._last_output_track_keys = []

    def reset(self):
        """Clear all track state (e.g. between video sources)."""
        self.tracks = []
        self._last_output_track_keys = []

    def output_track_keys(self):
        """Return stable keys aligned with the most recent output arrays."""
        return list(self._last_output_track_keys)

    def live_track_keys(self):
        """Return stable keys for all tracks still inside their grace period."""
        return [id(track["filter"]) for track in self.tracks]

    def _make_filters(self, n_kps):
        """Create per-region or single filter depending on keypoint count."""
        oc = self.outlier_cap
        rs, fs = self.rest_speed, self.fast_speed
        if n_kps == 133:
            return {
                name: OneEuroFilter(
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
            "all": OneEuroFilter(
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
        if not np.isfinite(step).all():
            step = np.where(np.isfinite(step), step, 0.0)
        # Cap pixel displacement in the image plane only.
        step_xy = step[:, :2]
        norms = np.linalg.norm(step_xy, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        scale = np.minimum(1.0, self.match_thresh / norms)
        step[:, :2] *= scale
        return last_kps + step

    @staticmethod
    def _detection_centroids(keypoints, scores):
        """Return finite, confidence-weighted image-plane centroids."""
        coords = keypoints[..., :2]
        finite = np.isfinite(coords).all(axis=2)
        safe_coords = np.where(finite[..., None], coords, 0.0)

        if scores is not None and scores.shape == finite.shape:
            weights = np.where(finite & np.isfinite(scores), np.clip(scores, 0.0, 1.0), 0.0)
        else:
            weights = finite.astype(np.float64)

        denom = weights.sum(axis=1)
        # All-zero confidence still gets a finite geometric fallback; this
        # preserves historical trackability while excluding malformed points.
        fallback = denom <= 0.0
        if fallback.any():
            weights = weights.copy()
            weights[fallback] = finite[fallback]
            denom = weights.sum(axis=1)

        centroids = np.full((keypoints.shape[0], 2), np.nan, dtype=np.float64)
        np.divide(
            np.einsum("nkd,nk->nd", safe_coords, weights),
            denom[:, None],
            out=centroids,
            where=denom[:, None] > 0.0,
        )
        return centroids

    def __call__(self, keypoints, scores, t):
        """Return (smoothed_keypoints, smoothed_scores) or (None, None)."""
        if keypoints is None or len(keypoints.shape) != 3 or keypoints.shape[0] == 0:
            return self._carry(t)

        score_array = np.asarray(scores)
        if score_array.shape != keypoints.shape[:2]:
            raise ValueError(
                f"scores shape {score_array.shape} does not match keypoints {keypoints.shape[:2]}"
            )
        finite_xy = np.isfinite(keypoints[..., :2]).all(axis=2)
        valid_scores = np.isfinite(score_array) & (score_array > 0.0)
        observed_people = (finite_xy & valid_scores).any(axis=1)
        if not observed_people.any():
            return self._carry(t)
        keypoints = keypoints[observed_people]
        scores = score_array[observed_people]

        n_det = keypoints.shape[0]
        det_centroids = self._detection_centroids(keypoints, scores)

        matched, used_tracks = self._match(det_centroids, t=t)

        new_tracks = []
        out_kps = []
        out_scores = []
        output_track_keys = []

        for i in range(n_det):
            kp = keypoints[i]
            raw_sc = scores[i]
            finite_measurement = np.isfinite(kp[..., :2]).all(axis=1)
            sc = np.where(finite_measurement & np.isfinite(raw_sc), np.clip(raw_sc, 0.0, 1.0), 0.0)

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
            smooth_centroid = self._detection_centroids(
                smooth_kp[np.newaxis], smooth_sc[np.newaxis]
            )[0]

            new_tracks.append(
                {
                    "filter": filt,
                    "centroid": smooth_centroid.copy(),
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
                # EMA is useful internal state, but exported evidence may not
                # be more confident than this frame's actual observation.
                out_scores.append(np.minimum(smooth_sc, sc))
                output_track_keys.append(id(filt))

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
            predicted_centroid = self._detection_centroids(
                predicted[np.newaxis], decayed[np.newaxis]
            )[0]
            new_tracks.append(
                {
                    "filter": tr["filter"],
                    "centroid": predicted_centroid.copy(),
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
                # Preserve predicted geometry for display/association, but a
                # carried track is not a fresh image observation.  Zero output
                # confidence prevents CSV fusion from triangulating multiple
                # extrapolations as independent camera evidence.
                out_scores.append(np.zeros_like(decayed))
                output_track_keys.append(id(tr["filter"]))

        self.tracks = new_tracks
        self._last_output_track_keys = output_track_keys
        if out_kps:
            return np.stack(out_kps), np.stack(out_scores)
        return None, None

    def _match(self, det_centroids, t=None):
        """Prediction-aware valid nearest-centroid matching."""
        matched = {}
        used_tracks = set()
        if not self.tracks or len(det_centroids) == 0:
            return matched, used_tracks

        trk_c = np.array([tr["centroid"][:2] for tr in self.tracks], dtype=np.float64)
        if t is not None:
            for i, tr in enumerate(self.tracks):
                velocity = tr.get("last_velocity")
                dt = t - tr.get("last_t", t)
                if velocity is None or dt <= 0:
                    continue
                velocity_xy = velocity[:, :2]
                finite = np.isfinite(velocity_xy).all(axis=1)
                if not finite.any():
                    continue
                global_velocity = np.median(velocity_xy[finite], axis=0)
                misses = tr.get("misses", 0)
                damping = self.carry_damping ** (misses + 1) if misses else 1.0
                step = global_velocity * (dt * damping)
                norm = float(np.hypot(step[0], step[1]))
                if self.match_thresh > 0 and norm > self.match_thresh:
                    step *= self.match_thresh / norm
                trk_c[i] += step
        cost = np.linalg.norm(det_centroids[:, None, :] - trk_c[None, :, :], axis=2)
        row_ind, col_ind = gated_assignment(cost, threshold=self.match_thresh)
        for r, c in zip(row_ind, col_ind, strict=False):
            matched[int(r)] = int(c)
            used_tracks.add(int(c))

        return matched, used_tracks

    def _carry(self, t=None):
        """Emit carry-forward tracks when no detections are present."""
        new_tracks = []
        out_kps = []
        out_scores = []
        output_track_keys = []
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
            predicted_centroid = self._detection_centroids(
                predicted[np.newaxis], decayed[np.newaxis]
            )[0]
            new_tracks.append(
                {
                    "filter": tr["filter"],
                    "centroid": predicted_centroid.copy(),
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
                out_scores.append(np.zeros_like(decayed))
                output_track_keys.append(id(tr["filter"]))
        self.tracks = new_tracks
        self._last_output_track_keys = output_track_keys
        if out_kps:
            return np.stack(out_kps), np.stack(out_scores)
        return None, None
