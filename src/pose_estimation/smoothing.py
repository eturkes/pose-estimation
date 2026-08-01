"""Temporal smoothing for pose landmarks using One Euro Filters."""

import os

import numpy as np

from .assignment import gated_assignment

_TWO_PI = 2.0 * np.pi


def _parse_optional_float(env_var, default):
    """Parse an env var as float, returning None if set to 'none' or empty."""
    val = os.environ.get(env_var, "")
    if val == "":
        return default
    if val.lower() == "none":
        return None
    return float(val)


def _body_anchor(landmarks, shoulder_indices):
    """Return shoulder midpoint, falling back to a finite landmark median."""
    values = np.asarray(landmarks)
    shoulders = values[..., list(shoulder_indices), :2]
    shoulder_anchor = shoulders.mean(axis=-2)
    if values.ndim == 2:
        if np.isfinite(shoulder_anchor).all():
            return shoulder_anchor
        finite_shoulders = np.isfinite(shoulders).all(axis=1)
        if finite_shoulders.any():
            return shoulders[finite_shoulders].mean(axis=0)
        xy = values[:, :2]
        finite = np.isfinite(xy).all(axis=1)
        return np.median(xy[finite], axis=0) if finite.any() else shoulder_anchor

    result = shoulder_anchor.copy()
    invalid = ~np.isfinite(result).all(axis=1)
    for index in np.flatnonzero(invalid):
        finite_shoulders = np.isfinite(shoulders[index]).all(axis=1)
        if finite_shoulders.any():
            result[index] = shoulders[index, finite_shoulders].mean(axis=0)
            continue
        xy = values[index, :, :2]
        finite = np.isfinite(xy).all(axis=1)
        if finite.any():
            result[index] = np.median(xy[finite], axis=0)
    return result


class OneEuroFilter:
    """One Euro Filter for smoothing noisy real-time signals.

    Adapts cutoff frequency based on signal speed: slow movements are smoothed
    aggressively while fast movements pass through with minimal lag.
    Works on numpy arrays of any shape.

    Optionally accepts per-keypoint confidence scores to modulate smoothing:
    low-confidence keypoints are pulled toward the previous estimate while
    high-confidence keypoints pass through with standard filtering.
    Scalar confidence values are also accepted and take a fast path that
    skips per-call array allocation.
    """

    __slots__ = (
        "_initialized",
        "_smoothed_speed",
        "_tau_d",
        "beta",
        "d_cutoff",
        "dx_prev",
        "fast_speed",
        "gamma",
        "min_cutoff",
        "outlier_cap",
        "rest_cutoff",
        "rest_speed",
        "speed_alpha",
        "t_prev",
        "x_prev",
    )

    def __init__(
        self,
        min_cutoff=1.0,
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
        # Adaptive min_cutoff: when rest_cutoff is set, the effective
        # min_cutoff interpolates between rest_cutoff (at low velocity)
        # and min_cutoff (at high velocity) per keypoint.  Velocity is
        # tracked via an EMA of per-keypoint speed in px/frame.
        self.rest_cutoff = rest_cutoff
        self.rest_speed = rest_speed
        self.fast_speed = fast_speed
        self.speed_alpha = speed_alpha
        self._smoothed_speed = None
        # Per-coordinate observation state. Detector tensors may contain
        # arbitrary finite placeholders for confidence-zero keypoints; those
        # values can be returned with confidence zero for continuity, but must
        # not become the origin used to clamp the first genuine observation.
        self._initialized = None
        # Cache the derivative-filter time constant; only depends on d_cutoff.
        self._tau_d = 1.0 / (_TWO_PI * d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t, confidence=None):
        x = np.asarray(x)

        # Parse confidence before initialization so a finite detector
        # placeholder is not mistaken for the first real observation. The same
        # weights later gate derivative innovation and position updates.
        w_scalar = None
        w = None
        if confidence is not None:
            if getattr(confidence, "ndim", 0) == 0:
                w_scalar = float(confidence)
                if not np.isfinite(w_scalar) or w_scalar < 0.0:
                    w_scalar = 0.0
                elif w_scalar > 1.0:
                    w_scalar = 1.0
                if self.gamma != 1.0 and w_scalar > 0.0:
                    w_scalar = w_scalar**self.gamma
            else:
                w = np.clip(np.asarray(confidence), 0.0, 1.0)
                w = np.where(np.isfinite(w), w, 0.0)
                if self.gamma != 1.0:
                    powered = np.zeros_like(w)
                    np.power(w, self.gamma, out=powered, where=w > 0.0)
                    w = powered

        finite_x = np.isfinite(x)
        if confidence is None:
            observed = finite_x
        elif w_scalar is not None:
            observed = finite_x & (w_scalar > 0.0)
        else:
            assert w is not None
            if x.ndim < 2 or w.shape != x.shape[:-1]:
                raise ValueError(
                    f"confidence shape {w.shape} does not match keypoint array {x.shape}"
                )
            observed = finite_x & (w > 0.0)[..., np.newaxis]

        if self.t_prev is None:
            initial = x.copy()
            if not np.isfinite(initial).all():
                initial = np.nan_to_num(initial, nan=0.0, posinf=0.0, neginf=0.0)
            self.x_prev = initial.copy()
            self.dx_prev = np.zeros_like(initial)
            self._initialized = observed.copy()
            self.t_prev = t
            return initial

        x_prev = self.x_prev
        assert x_prev is not None
        assert self.dx_prev is not None
        assert self._initialized is not None
        if x.shape != x_prev.shape:
            raise ValueError(f"input shape {x.shape} does not match filter state {x_prev.shape}")

        # Bypass smoothing and the displacement cap independently for every
        # coordinate on its first finite, positive-confidence observation.
        # This prevents arbitrary confidence-zero model output from biasing a
        # later genuine measurement while retaining initialized neighbours.
        newly_observed = observed & ~self._initialized
        if newly_observed.any():
            x_prev = x_prev.copy()
            dx_prev = self.dx_prev.copy()
            x_prev[newly_observed] = x[newly_observed]
            dx_prev[newly_observed] = 0.0
            self.x_prev = x_prev
            self.dx_prev = dx_prev
        self._initialized |= observed

        dt = t - self.t_prev
        if dt < 1e-6:
            dt = 1e-6

        # Derivative low-pass: a_d = dt / (dt + tau_d)
        a_d = dt / (dt + self._tau_d)

        # Compute the (x - x_prev) delta once and reuse it for both
        # dx_hat (derivative) and x_hat (state update) — saves an alloc
        # and a subtract op vs recomputing.  Non-finite coordinates hold their
        # previous value, preventing one malformed keypoint from permanently
        # poisoning the filter state.
        measurement = x if finite_x.all() else np.where(finite_x, x, x_prev)
        with np.errstate(over="ignore", invalid="ignore"):
            diff = measurement - x_prev
        if not np.isfinite(diff).all():
            diff = np.where(np.isfinite(diff), diff, 0.0)
        if not np.issubdtype(diff.dtype, np.floating):
            diff = diff.astype(np.float64)

        # Outlier rejection: cap the unexpected component of displacement.
        # Predicted movement (from velocity) passes through; only the
        # surprise beyond outlier_cap pixels is clamped per keypoint.
        if self.outlier_cap > 0 and x.ndim == 2 and x.shape[1] >= 2:
            predicted_step = self.dx_prev * dt
            unexpected = diff - predicted_step
            unexpected_xy = unexpected[:, :2]
            norms_sq = np.einsum("ij,ij->i", unexpected_xy, unexpected_xy)
            cap_sq = self.outlier_cap * self.outlier_cap
            mask = norms_sq > cap_sq
            if mask.any():
                norms = np.sqrt(norms_sq)
                s = np.ones(norms.shape)
                np.divide(self.outlier_cap, norms, out=s, where=mask)
                diff[:, :2] = predicted_step[:, :2] + unexpected_xy * s[:, None]

        scale = a_d / dt
        candidate_dx = diff * scale + self.dx_prev * (1.0 - a_d)
        if confidence is None or w_scalar == 1.0 or (w is not None and np.all(w == 1.0)):
            dx_hat = candidate_dx
        elif w_scalar is not None:
            if w_scalar == 0.0:
                dx_hat = self.dx_prev.copy()
            else:
                dx_hat = self.dx_prev + (candidate_dx - self.dx_prev) * w_scalar
        else:
            assert w is not None
            dx_hat = self.dx_prev + (candidate_dx - self.dx_prev) * w[:, None]

        # Per-element cutoff and gain.  Compute |dx_hat|*beta + min_cutoff
        # then convert directly to a = dt/(dt + 1/(2π*cutoff)).
        abs_dx = np.abs(dx_hat)
        abs_dx *= self.beta

        # Adaptive min_cutoff: during rest (low velocity), lower the
        # cutoff floor for heavier smoothing.  During fast movement,
        # beta*|speed| dominates and min_cutoff is irrelevant.
        if self.rest_cutoff is not None and x.ndim == 2:
            # Rest/fast thresholds are pixel quantities, so depth or other
            # non-image coordinates must not change the adaptive regime.
            velocity_xy = dx_hat[:, : min(2, dx_hat.shape[1])]
            kp_speed = np.sqrt(np.einsum("ij,ij->i", velocity_xy, velocity_xy)) * dt
            if self._smoothed_speed is None:
                self._smoothed_speed = kp_speed.copy()
            else:
                a_s = self.speed_alpha
                self._smoothed_speed += a_s * (kp_speed - self._smoothed_speed)
            rng = self.fast_speed - self.rest_speed
            if rng > 0:
                t_interp = (self._smoothed_speed - self.rest_speed) / rng
                np.clip(t_interp, 0.0, 1.0, out=t_interp)
                effective_mc = self.rest_cutoff + t_interp * (self.min_cutoff - self.rest_cutoff)
            else:
                effective_mc = np.full_like(self._smoothed_speed, self.min_cutoff)
            abs_dx += effective_mc[:, None]
        else:
            abs_dx += self.min_cutoff

        # a = dt / (dt + 1/(2π*cutoff)) = (2π*cutoff*dt) / (2π*cutoff*dt + 1)
        abs_dx *= _TWO_PI * dt
        a = abs_dx / (abs_dx + 1.0)

        # x_hat = a * diff + x_prev (reuse precomputed diff).
        x_hat = a * diff
        x_hat += x_prev

        if confidence is not None:
            # 0-D scalars (Python or numpy) take a math-only fast path
            # that skips np.clip / np.power and the (n_kp,) weight array.
            if w_scalar is not None:
                # result = w*(x_hat - x_prev) + x_prev (in-place on x_hat,
                # which is local scratch we can safely mutate).
                result = x_hat
                result -= x_prev
                result *= w_scalar
                result += x_prev
            else:
                assert w is not None
                # result = w * (x_hat - x_prev) + x_prev
                result = (x_hat - x_prev) * w[:, None]
                result += x_prev
        else:
            result = x_hat

        # ``result`` is the value we return; callers may mutate it (e.g.
        # bone-length smoother), so keep a private copy as our state.
        # ``dx_hat`` is fresh and never returned, so it can be aliased.
        self.x_prev = result.copy()
        self.dx_prev = dx_hat
        self.t_prev = t

        return result


class PoseSmoother:
    """Temporal smoothing for body and hand landmarks.

    Tracks detections across frames by anchor point proximity and applies
    One Euro Filters to reduce jitter while preserving responsiveness.
    Body uses heavier smoothing (min_cutoff=0.3) with confidence-weighted
    blending from per-keypoint visibility scores; hands use moderate
    smoothing (min_cutoff=0.5) for fast finger movements.

    During brief detection dropouts (carry-forward), body tracks
    extrapolate using the last velocity estimate from the One Euro Filter
    with exponential damping, producing smoother motion continuity than
    static replay.
    """

    def __init__(self, match_threshold=150, carry_damping=None):
        self.match_threshold = match_threshold
        if carry_damping is None:
            carry_damping = float(os.environ.get("POSE_BENCH_CARRY_DAMPING", "0.8"))
        self.carry_damping = carry_damping
        # Per-run tuning knobs; env vars are only re-read when a new
        # PoseSmoother is constructed (i.e. once per subprocess invocation).
        self._body_mc = float(os.environ.get("POSE_BENCH_BODY_MIN_CUTOFF", "0.3"))
        self._body_b = float(os.environ.get("POSE_BENCH_BODY_BETA", "0.5"))
        self._hand_mc = float(os.environ.get("POSE_BENCH_HAND_MIN_CUTOFF", "0.5"))
        self._hand_b = float(os.environ.get("POSE_BENCH_HAND_BETA", "0.3"))
        self._gamma = float(os.environ.get("POSE_BENCH_CONFIDENCE_GAMMA", "2.0"))
        self._grace = int(os.environ.get("POSE_BENCH_CARRY_GRACE", "10"))
        self._outlier_cap = float(os.environ.get("POSE_BENCH_OUTLIER_CAP", "30"))
        # Adaptive smoothing: lower effective min_cutoff during rest for
        # heavier smoothing of stationary keypoints.  "none" disables.
        self._body_rest_cutoff = _parse_optional_float("POSE_BENCH_BODY_REST_CUTOFF", 0.05)
        self._hand_rest_cutoff = _parse_optional_float("POSE_BENCH_HAND_REST_CUTOFF", 0.15)
        self._rest_speed = float(os.environ.get("POSE_BENCH_REST_SPEED", "2.0"))
        self._fast_speed = float(os.environ.get("POSE_BENCH_FAST_SPEED", "10.0"))
        self.body_tracks = []
        self.hand_tracks = []
        self._n_active_bodies = 0
        self._n_active_hands = 0
        self._last_hand_observation_indices = []

    def _predict_anchor(self, track, get_anchor, t):
        """Predict one track anchor at *t* from its reliable filter velocity."""
        _, anchor, _, misses, _, velocity, last_t = track
        if velocity is None:
            return anchor
        dt = t - last_t
        if dt <= 0:
            return anchor

        anchor_velocity = np.asarray(get_anchor(velocity), dtype=np.float64)
        if anchor_velocity.shape != (2,) or not np.isfinite(anchor_velocity).all():
            return anchor
        damping = self.carry_damping ** (misses + 1) if misses else 1.0
        if not np.isfinite(damping):
            return anchor
        step = anchor_velocity * (dt * damping)
        step_norm = float(np.hypot(step[0], step[1]))
        if self.match_threshold > 0 and step_norm > self.match_threshold:
            step *= self.match_threshold / step_norm
        return anchor + step

    def _match_and_smooth(
        self,
        tracks,
        landmarks,
        get_anchor,
        new_filter_fn,
        t,
        grace=0,
        max_tracks=None,
        emit_carry=False,
        confidences=None,
        static_carry=False,
    ):
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

        Returns (new_tracks, smoothed, n_active, active_indices) where *n_active* is
        the count of freshly matched landmarks (excludes carry-forward
        entries appended when *emit_carry* is True), and *active_indices*
        maps those outputs back to the input landmark list.
        """
        if static_carry:
            emit_carry = True
        smoothed = []
        new_tracks = []
        used = set()
        active_indices = []

        # --- Hungarian (optimal) landmark-to-track assignment ---
        n_lm = len(landmarks)
        n_tr = len(tracks)
        lm_to_track = {}  # landmark index -> track index

        threshold = self.match_threshold
        if n_lm > 0 and n_tr > 0:
            # Batched anchor extraction: ``get_anchor`` is Ellipsis-aware so a
            # single call on the stacked landmarks produces the full (n_lm, 2)
            # anchor array — replacing n_lm per-element lambda dispatches.
            if n_lm == 1:
                # ``[None]`` makes a free 3-D view of the single landmark so
                # the Ellipsis-aware ``get_anchor`` produces the (1, 2) array
                # without copying.
                stacked = landmarks[0][None]
            else:
                stacked = np.stack(landmarks)
            anchors_lm = get_anchor(stacked)
            # Track anchors are 2-element arrays stored at tuple index 1.
            # An empty buffer + per-row write beats np.stack at n_tr ≤ 4
            # because the latter has a fixed Python-side validation cost.
            anchors_tr = np.empty((n_tr, 2), dtype=anchors_lm.dtype)
            for i, tr in enumerate(tracks):
                anchors_tr[i] = self._predict_anchor(tr, get_anchor, t)
            # Cost matrix: Euclidean distance between each (landmark, track).
            dx = anchors_lm[:, 0:1] - anchors_tr[None, :, 0]
            dy = anchors_lm[:, 1:2] - anchors_tr[None, :, 1]
            cost = np.hypot(dx, dy)
            row_ind, col_ind = gated_assignment(cost, threshold=threshold)
            pairs = list(zip(row_ind, col_ind, strict=False))
            if max_tracks is not None and len(pairs) > max_tracks:
                pairs = sorted(pairs, key=lambda pair: (cost[pair], pair[0], pair[1]))[:max_tracks]
                pairs.sort()
            for r, c in pairs:
                lm_to_track[int(r)] = int(c)
                used.add(int(c))

        surviving_existing = sum(i in used or track[3] < grace for i, track in enumerate(tracks))
        birth_slots = None if max_tracks is None else max(0, max_tracks - surviving_existing)
        n_births = 0

        for lm_idx, lm in enumerate(landmarks):
            if lm_idx in lm_to_track:
                tr_idx = lm_to_track[lm_idx]
                filt = tracks[tr_idx][0]
                age = tracks[tr_idx][2] + 1
            elif birth_slots is None or n_births < birth_slots:
                filt = new_filter_fn()
                age = 1
                n_births += 1
            else:
                continue

            conf = confidences[lm_idx] if confidences is not None else None
            s = filt(lm, t, confidence=conf)
            # ``filt.x_prev`` is a stable internal copy of ``s`` (see
            # OneEuroFilter.__call__) and ``filt.dx_prev`` is a fresh array
            # that the filter never mutates after assignment, so we can
            # alias them here without extra copies.
            last_output_state = filt.x_prev
            velocity = filt.dx_prev
            new_tracks.append(
                (filt, get_anchor(last_output_state).copy(), age, 0, last_output_state, velocity, t)
            )
            smoothed.append(s)
            active_indices.append(lm_idx)

        n_active = len(smoothed)

        # Carry forward unmatched tracks within grace period.
        # Decrement age each missed frame so intermittent false
        # positives cannot accumulate age across grace gaps.
        for i, (filt, prev_anchor, age, misses, last_out, last_vel, last_t) in enumerate(tracks):
            if i in used or misses >= grace:
                continue
            if max_tracks is not None and len(new_tracks) >= max_tracks:
                continue
            new_misses = misses + 1
            decayed_age = max(0, age - 1)
            if emit_carry and last_out is not None:
                if static_carry:
                    predicted = last_out
                    new_anchor = self._predict_anchor(
                        (filt, prev_anchor, age, misses, last_out, last_vel, last_t),
                        get_anchor,
                        t,
                    ).copy()
                    next_last_out = last_out
                else:
                    predicted = self._extrapolate(last_out, last_vel, last_t, t, new_misses)
                    new_anchor = get_anchor(predicted).copy()
                    # ``predicted`` is fresh from _extrapolate (or last_out
                    # when extrapolation bails); the caller receives the
                    # same reference as our stored snapshot, so callers
                    # must not mutate carry-forward outputs in place.
                    next_last_out = predicted
                new_tracks.append(
                    (filt, new_anchor, decayed_age, new_misses, next_last_out, last_vel, t)
                )
                smoothed.append(predicted)
            else:
                new_tracks.append(
                    (filt, prev_anchor, decayed_age, new_misses, last_out, last_vel, last_t)
                )

        return new_tracks, smoothed, n_active, active_indices

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

        damping = self.carry_damping**misses
        # When damping is negligible the step is sub-pixel, so skip the
        # work entirely and return a static carry.
        if damping < 1e-3:
            return last_output

        step = last_velocity * (dt * damping)
        if not np.isfinite(step).all():
            step = np.where(np.isfinite(step), step, 0.0)

        # Cap per-keypoint image-plane displacement magnitude.  Depth and
        # other non-image coordinates are not pixels and must not contribute
        # to, or be rescaled by, the pixel threshold.
        step_xy = step[:, :2]
        # Compute squared norms
        # directly (np.linalg.norm has Python-level overhead that dominates
        # at the small array sizes we see here) and only apply the cap to
        # rows that actually exceed the threshold.
        threshold = self.match_threshold
        threshold_sq = threshold * threshold
        norms_sq = np.einsum("ij,ij->i", step_xy, step_xy)
        too_long = norms_sq > threshold_sq
        if too_long.any():
            scale = np.ones_like(norms_sq)
            np.sqrt(norms_sq, out=norms_sq, where=too_long)
            np.divide(threshold, norms_sq, out=scale, where=too_long)
            step[:, :2] *= scale[:, None]

        return last_output + step

    def body_track_ages(self):
        """Return the age (in frames) of each active body track."""
        return [age for _, _, age, _, _, _, _ in self.body_tracks[: self._n_active_bodies]]

    def hand_track_ages(self):
        """Return the age (in frames) of each active hand track."""
        return [age for _, _, age, _, _, _, _ in self.hand_tracks[: self._n_active_hands]]

    def smooth_bodies(self, body_landmarks, body_visibilities, t, shoulder_indices=(0, 1)):
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
        visibility_list = list(body_visibilities or [])
        safe_visibilities = []
        for index, landmarks in enumerate(lm_list):
            n_keypoints = landmarks.shape[0]
            if index >= len(visibility_list):
                visibility = np.zeros(n_keypoints, dtype=np.float64)
            else:
                visibility = np.asarray(visibility_list[index], dtype=np.float64)
                if visibility.shape != (n_keypoints,):
                    raise ValueError(
                        f"body visibility shape {visibility.shape} does not match {(n_keypoints,)}"
                    )
            # OneEuro deliberately holds the previous coordinate when a raw
            # measurement is nonfinite.  That held geometry is useful for the
            # tracker, but it is not fresh evidence and must carry confidence 0.
            coordinate_valid = np.isfinite(landmarks).all(axis=1)
            safe_visibilities.append(
                np.where(
                    coordinate_valid & np.isfinite(visibility),
                    np.clip(visibility, 0.0, 1.0),
                    0.0,
                )
            )

        self.body_tracks, smoothed, n_active, active_indices = self._match_and_smooth(
            self.body_tracks,
            lm_list,
            get_anchor=lambda lm: _body_anchor(lm, si),
            new_filter_fn=lambda: OneEuroFilter(
                min_cutoff=self._body_mc,
                beta=self._body_b,
                gamma=self._gamma,
                outlier_cap=self._outlier_cap,
                rest_cutoff=self._body_rest_cutoff,
                rest_speed=self._rest_speed,
                fast_speed=self._fast_speed,
            ),
            t=t,
            grace=self._grace,
            emit_carry=True,
            confidences=safe_visibilities if lm_list else None,
        )
        self._n_active_bodies = n_active
        # Actively matched bodies use the provided visibility.  A carried pose
        # is a prediction, not a detector observation, so advertise zero
        # visibility: downstream multi-camera fusion must not treat an
        # extrapolated skeleton as independent image evidence.
        vis = [safe_visibilities[index] for index in active_indices]
        n_carried = len(smoothed) - n_active
        if n_carried > 0:
            n_kp = smoothed[0].shape[0] if smoothed else 12
            vis.extend([np.zeros(n_kp)] * n_carried)
        return smoothed, vis, n_active

    def smooth_hands(self, hand_landmarks, t, hand_flags=None, grace=None, max_tracks=None):
        if grace is None:
            grace = self._grace
        # ``hand_flag`` is a single scalar per hand (uniform across all 21
        # keypoints) so pass the scalars straight through — OneEuroFilter's
        # scalar fast path avoids the per-frame (21,) array allocation.
        self.hand_tracks, smoothed, n_active, active_indices = self._match_and_smooth(
            self.hand_tracks,
            hand_landmarks or [],
            get_anchor=lambda lm: lm[..., 0, :2],
            new_filter_fn=lambda: OneEuroFilter(
                min_cutoff=self._hand_mc,
                beta=self._hand_b,
                gamma=self._gamma,
                outlier_cap=self._outlier_cap,
                rest_cutoff=self._hand_rest_cutoff,
                rest_speed=self._rest_speed,
                fast_speed=self._fast_speed,
            ),
            t=t,
            grace=grace,
            static_carry=True,
            max_tracks=max_tracks,
            confidences=hand_flags,
        )
        self._n_active_hands = n_active
        self._last_hand_observation_indices = active_indices
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

    def body_track_keys(self):
        """Return stable keys aligned with the latest smoothed body outputs."""
        return [id(track[0]) for track in self.body_tracks]

    def hand_carry_flags(self):
        """Return a list of bools: whether each active hand track is carrying."""
        return [misses > 0 for _, _, _, misses, _, _, _ in self.hand_tracks]

    def hand_observation_indices(self):
        """Map fresh smoothed hand outputs to the current raw detection list."""
        return list(self._last_hand_observation_indices)

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
        # np.hypot on explicit dx/dy beats np.linalg.norm's Python-level
        # dispatch at the (12-33, 2) sizes we see per frame.
        dx = raw[:, 0] - smo[:, 0]
        dy = raw[:, 1] - smo[:, 1]
        return float(np.hypot(dx, dy).sum())
