"""Score hygiene for top-down whole-body keypoints before smoothing and export.

A top-down model returns every keypoint inside its crop, so a crop at the frame
edge places points outside the image at confident-looking scores, and a hidden
hand is drawn on top of the visible one.  The R stage reads every score above
zero as an observation, so both enter the features as measurements.  Seated
subjects add a third case: legs under the table and hips below an overhead
camera are placed at scores no cut separates from real points.  These
functions zero the scores; coordinates stay as the model returned them.
"""

from __future__ import annotations

import numpy as np

HIPS = slice(11, 13)
LOWER_BODY = slice(13, 23)  # knees, ankles, then the six COCO-WholeBody foot points
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)
WHOLE_BODY_KEYPOINTS = 133

DUPLICATE_OVERLAP = 0.3
DUPLICATE_WEAK = 0.5
DUPLICATE_RELATIVE = 0.6
DUPLICATE_MIN_COMMON = 10

# Calibrated on the smoother's exported score (min(EMA, raw)) over 148 labelled frames: hallucinated
# hands median 0.43, real hands 0.83.  Hysteresis keeps a real hand through brief dips.
HAND_GATE_ON = 0.65
HAND_GATE_OFF = 0.5
HAND_GATE_MIN_POINTS = 10


def zero_out_of_frame(keypoints, scores, width, height):
    """Scores with 0 wherever a keypoint is non-finite or outside ``[0, w) x [0, h)``."""
    xy = np.asarray(keypoints, dtype=np.float64)[..., :2]
    out = np.array(scores, dtype=np.float64, copy=True)
    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out[~inside] = 0.0
    return out


def _extent(points):
    span = points.max(axis=0) - points.min(axis=0)
    return float(np.hypot(span[0], span[1]))


def suppress_duplicate_hand(keypoints, scores):
    """Scores with the weaker hand zeroed wherever it duplicates the stronger one.

    Present = finite and score > 0.  Fires when >= 10 indices are present in both
    hands, the median distance between them is under 0.3 of the larger hand
    extent, and the weaker hand's mean score is under 0.5 and under 0.6 of the
    stronger's.
    """
    xy = np.asarray(keypoints, dtype=np.float64)[..., :2]
    out = np.array(scores, dtype=np.float64, copy=True)
    if out.ndim != 2 or out.shape[1] != WHOLE_BODY_KEYPOINTS:
        return out
    for person in range(out.shape[0]):
        hands = []
        for part in (LEFT_HAND, RIGHT_HAND):
            points, conf = xy[person, part], out[person, part]
            present = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0.0)
            hands.append((points, conf, present))
        (lp, lc, lok), (rp, rc, rok) = hands
        common = lok & rok
        if int(common.sum()) < DUPLICATE_MIN_COMMON:
            continue
        extent = max(_extent(lp[lok]), _extent(rp[rok]))
        if extent <= 0.0:
            continue
        overlap = float(np.median(np.linalg.norm(lp[common] - rp[common], axis=1))) / extent
        left_mean, right_mean = float(lc[lok].mean()), float(rc[rok].mean())
        weaker, stronger = min(left_mean, right_mean), max(left_mean, right_mean)
        if (
            overlap < DUPLICATE_OVERLAP
            and weaker < DUPLICATE_WEAK
            and weaker < DUPLICATE_RELATIVE * stronger
        ):
            out[person, LEFT_HAND if left_mean < right_mean else RIGHT_HAND] = 0.0
    return out


def drop_body_parts(scores, *, lower_body=False, hips=False):
    """Scores with knees, ankles and feet (``lower_body``) and the hips (``hips``) at 0."""
    out = np.array(scores, dtype=np.float64, copy=True)
    if out.ndim == 2:
        if lower_body:
            out[:, LOWER_BODY] = 0.0
        if hips:
            out[:, HIPS] = 0.0
    return out


def apply_hygiene(keypoints, scores, width, height, *, drop_lower_body=False, drop_hips=False):
    """Out-of-frame zeroing, the requested body-part drops, then duplicate-hand suppression."""
    out = zero_out_of_frame(keypoints, scores, width, height)
    out = drop_body_parts(out, lower_body=drop_lower_body, hips=drop_hips)
    return suppress_duplicate_hand(keypoints, out)


def _finite_mean(values):
    """Mean of positive finite values; rescaled by the maximum only when the plain sum overflows."""
    with np.errstate(over="ignore"):
        mean = float(values.mean())
    if np.isfinite(mean):
        return mean
    peak = float(values.max())
    return peak * float((values / peak).mean())


class HandPresenceGate:
    """Export admission per hand with hysteresis, keyed by track.

    A hand is present when at least ``min_points`` of its 21 scores are finite
    and positive; its level is their mean.  Off -> on at ``level >= on``; on ->
    off when absent or ``level < off``.  An off hand's scores return as 0.
    Runs after the smoother: the thresholds are calibrated on its output score.
    """

    def __init__(self, on=HAND_GATE_ON, off=HAND_GATE_OFF, min_points=HAND_GATE_MIN_POINTS):
        if on <= off:
            raise ValueError(f"hand gate needs on > off, got on={on} off={off}")
        self.on, self.off, self.min_points = float(on), float(off), int(min_points)
        self.reset()

    def reset(self):
        self.state: dict = {}
        self.hand_frames_present = 0
        self.hand_frames_gated = 0

    def prune(self, live_keys):
        live = set(live_keys)
        self.state = {key: value for key, value in self.state.items() if key in live}

    def __call__(self, scores, keys):
        out = np.array(scores, dtype=np.float64, copy=True)
        if out.ndim != 2 or out.shape[1] != WHOLE_BODY_KEYPOINTS:
            return out
        keys = list(keys)
        if len(keys) != out.shape[0]:
            raise ValueError(f"{len(keys)} keys for {out.shape[0]} rows")
        for row, key in enumerate(keys):
            states = self.state.setdefault(key, [False, False])
            for hand, part in enumerate((LEFT_HAND, RIGHT_HAND)):
                conf = out[row, part]
                positive = np.isfinite(conf) & (conf > 0.0)
                present = int(positive.sum()) >= self.min_points
                level = _finite_mean(conf[positive]) if present else 0.0
                if present:
                    self.hand_frames_present += 1
                states[hand] = present and level >= (self.off if states[hand] else self.on)
                if not states[hand]:
                    if present:
                        self.hand_frames_gated += 1
                    out[row, part] = 0.0
        return out
