"""Per-arm keypoint score calibration: raw model score → P(keypoint correct) (contract M4.3a).

Raw SimCC and heatmap scores are checkpoint-specific — RTMW-L 384's raw median reads 4.2 where
RTMW-m's reads 0.74 on the same crops — and the smoother clips every score to [0, 1], so an
uncalibrated arm saturates each gate downstream. A calibration maps each keypoint group's raw
score through a monotone piecewise-linear curve fit on public ground truth, giving every arm one
meaning for a score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).resolve().parent / "score_calibrations"
#: COCO-WholeBody keypoint index ranges per group.
GROUPS = {"body": (0, 17), "foot": (17, 23), "face": (23, 91), "hand": (91, 133)}
MAX_KNOTS = 64


@dataclass(frozen=True)
class Curve:
    x: tuple[float, ...]  # strictly increasing raw scores
    # Non-decreasing, >= 0: P(correct) in [0, 1], or a reference arm's equivalent raw score.
    y: tuple[float, ...]

    def __post_init__(self):
        x, y = np.asarray(self.x, float), np.asarray(self.y, float)
        if len(x) < 2 or len(x) != len(y):
            raise ValueError("a curve needs >= 2 matching knots")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("curve knots must be finite")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
            raise ValueError("curve knots must rise in x and not fall in y")
        if y.min() < 0:
            raise ValueError("calibrated values must be >= 0")

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        # np.interp clamps outside the knots; NaN passes through as NaN.
        return np.interp(raw, self.x, self.y)


@dataclass(frozen=True)
class ScoreCalibration:
    curves: dict[str, Curve]  # absent group = identity

    def apply(self, scores: np.ndarray) -> np.ndarray:
        out = np.array(scores, dtype=np.float64, copy=True)
        for group, curve in self.curves.items():
            lo, hi = GROUPS[group]
            part = out[..., lo:hi]
            # NaN stays NaN; +-inf clamps to the curve's ends like any out-of-range score.
            known = ~np.isnan(part)
            part[known] = curve(part[known])
        return out

    def to_json(self) -> str:
        return json.dumps(
            {g: {"x": list(c.x), "y": list(c.y)} for g, c in sorted(self.curves.items())},
            indent=1,
            sort_keys=True,
        )


def identity() -> ScoreCalibration:
    return ScoreCalibration({})


def load(name: str | None) -> ScoreCalibration:
    """The named calibration under `CALIBRATION_DIR`, or identity for `None`."""
    if name is None:
        return identity()
    raw = json.loads((CALIBRATION_DIR / name).read_text(encoding="utf-8"))
    return ScoreCalibration({g: Curve(tuple(c["x"]), tuple(c["y"])) for g, c in raw.items()})


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the weighted least-squares non-decreasing fit to `y`."""
    blocks: list[list[float]] = []  # [value, weight, count]
    for value, weight in zip(y, w, strict=True):
        blocks.append([float(value), float(weight), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(int(n), v) for v, _, n in blocks])


def fit(raw: np.ndarray, correct: np.ndarray) -> Curve:
    """Isotonic P(correct | raw), compressed to <= `MAX_KNOTS` knots at the fit's breakpoints."""
    raw, correct = np.asarray(raw, float), np.asarray(correct, float)
    keep = np.isfinite(raw)
    raw, correct = raw[keep], correct[keep]
    if len(raw) < 2:
        raise ValueError("calibration needs >= 2 finite observations")
    order = np.argsort(raw, kind="mergesort")
    xs, ys = raw[order], correct[order]
    ux, inverse = np.unique(xs, return_inverse=True)
    sums = np.bincount(inverse, weights=ys)
    counts = np.bincount(inverse).astype(float)
    level = _pav(sums / counts, counts)
    if len(ux) < 2:
        return Curve((float(ux[0]), float(ux[0]) + 1.0), (float(level[0]),) * 2)
    # Knots where the step function changes value, plus both ends.
    change = np.flatnonzero(np.diff(level) != 0)
    idx = np.unique(np.concatenate([[0], change, change + 1, [len(ux) - 1]]))
    if len(idx) > MAX_KNOTS:
        idx = np.unique(np.round(np.linspace(0, len(ux) - 1, MAX_KNOTS)).astype(int))
    x, y = ux[idx], np.clip(level[idx], 0.0, 1.0)
    y = np.maximum.accumulate(y)
    return Curve(tuple(float(v) for v in x), tuple(float(v) for v in y))


def _logistic(raw: np.ndarray, correct: np.ndarray, iterations: int = 50) -> tuple[float, float]:
    """Maximum-likelihood (a, b) of P(correct | raw) = sigmoid(a * raw + b), Newton-Raphson."""
    x, y = np.asarray(raw, float), np.asarray(correct, float)
    keep = np.isfinite(x)
    x, y = x[keep], y[keep]
    if len(x) < 2 or y.min() == y.max():
        raise ValueError("a logistic fit needs both outcomes")
    a, b = 0.0, float(np.log((y.mean() + 1e-9) / (1 - y.mean() + 1e-9)))
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        w = p * (1 - p) + 1e-12
        grad = np.array([np.sum((y - p) * x), np.sum(y - p)])
        hess = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        step = np.linalg.solve(hess, grad)
        a, b = a + float(step[0]), b + float(step[1])
        if np.abs(step).max() < 1e-10:
            break
    return a, b


def fit_reference(
    raw: np.ndarray, correct: np.ndarray, ref_raw: np.ndarray, ref_correct: np.ndarray
) -> Curve:
    """Map an arm's raw score onto the reference arm's raw score of equal P(correct) (M4.3b D01).

    Both sides are logistic, so the map is affine: (a * raw + b - b_ref) / a_ref, clamped at 0.
    Knots span the arm's observed raw range, plus the zero crossing when it falls inside.
    """
    a, b = _logistic(raw, correct)
    a_ref, b_ref = _logistic(ref_raw, ref_correct)
    if a <= 0 or a_ref <= 0:
        raise ValueError("a score that does not rise with correctness cannot be calibrated")
    finite = np.asarray(raw, float)[np.isfinite(np.asarray(raw, float))]
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0

    def mapped(r: float) -> float:
        return (a * r + b - b_ref) / a_ref

    xs = [lo, hi]
    zero = (b_ref - b) / a  # raw where the map crosses 0
    if lo < zero < hi:
        xs.insert(1, zero)
    ys = [max(0.0, mapped(x)) for x in xs]
    return Curve(tuple(xs), tuple(np.maximum.accumulate(ys)))
