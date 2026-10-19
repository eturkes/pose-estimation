"""Cross-view agreement of same-event asset summaries (contract `.agent/archive/contract-m4u5b.md`).

Descriptive, not reliability: views of one recording event differ in projection, observed time
support, segmentation and tracking, and the statistic carries all of them (D01). Each event is a
target, each of its views a rating; events are never imputed a missing view.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

RESAMPLES = 1000
#: A range needs a finite point estimate and at least this share of finite resamples (D02).
MIN_FINITE_SHARE = 0.95
REASONS = (
    "agreement_below_floor",
    "agreement_no_multiview",
    "agreement_zero_variance",
    "agreement_insufficient_resamples",
)


def icc_oneway(groups: Sequence[Sequence[float]], weights: Sequence[float] | None = None):
    """One-way random-effects ICC(1,1) over unbalanced targets, `None` when undefined.

    `weights[i]` = how many copies of target i the sample holds (a resample draws a subject
    more than once); copies are separate targets with identical values.
    """
    y = [np.asarray(g, float) for g in groups]
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    k = np.array([len(g) for g in y], float)
    g = w.sum()
    n = float(np.dot(w, k))
    if g < 2 or n <= g:
        return None
    means = np.array([v.mean() for v in y])
    grand = float(np.dot(w, k * means)) / n
    ssb = float(np.dot(w, k * (means - grand) ** 2))
    ssw = float(np.dot(w, [((v - m) ** 2).sum() for v, m in zip(y, means, strict=True)]))
    msb, msw = ssb / (g - 1), ssw / (n - g)
    k0 = (n - float(np.dot(w, k**2)) / n) / (g - 1)
    denominator = msb + (k0 - 1) * msw
    if not denominator > 1e-300:
        return None
    return (msb - msw) / denominator


def mean_pair_abs_diff(values: Sequence[float]) -> float:
    v = np.asarray(values, float)
    i, j = np.triu_indices(len(v), 1)
    return float(np.abs(v[i] - v[j]).mean())


def seed_for(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def resample_icc(
    events: Mapping[str, Sequence[float]],
    subject_of: Mapping[str, str],
    seed: int,
    n: int = RESAMPLES,
) -> list[float | None]:
    """ICC per subject-cluster resample: draw subjects with replacement, every drawn copy adds
    all of that subject's events (multiplicity kept); canonical order before indexing.

    Vectorised `icc_oneway` over the draws: per event k, mean and within sum of squares, then
    each draw weights its events by the draw count of their subject."""
    names = sorted(events)
    subjects = sorted({subject_of[e] for e in names})
    index = {s: i for i, s in enumerate(subjects)}
    owner = np.array([index[subject_of[e]] for e in names])
    k = np.array([len(events[e]) for e in names], float)
    # Centred first: the ICC is translation-invariant, and the uncentred sums of squares below
    # cancel catastrophically at a large common offset (review C09c: 26/50 draws wrong at 1e8).
    centre = float(np.mean([np.mean(events[e]) for e in names]))
    mean = np.array([np.mean(events[e]) for e in names]) - centre
    ssw_e = np.array(
        [
            ((np.asarray(events[e], float) - centre - m) ** 2).sum()
            for e, m in zip(names, mean, strict=True)
        ]
    )
    rng = np.random.default_rng(seed)
    counts = np.stack(
        [
            np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects))
            for _ in range(n)
        ]
    )
    w = counts[:, owner].astype(float)
    g = w.sum(axis=1)
    total = w @ k
    with np.errstate(divide="ignore", invalid="ignore"):
        grand = (w @ (k * mean)) / total
        ssb = w @ (k * mean**2) - total * grand**2
        msb = ssb / (g - 1)
        msw = (w @ ssw_e) / (total - g)
        k0 = (total - (w @ k**2) / total) / (g - 1)
        denominator = msb + (k0 - 1) * msw
        icc = (msb - msw) / denominator
    ok = (g >= 2) & (total > g) & (denominator > 1e-300) & np.isfinite(icc)
    return [float(v) if good else None for v, good in zip(icc, ok, strict=True)]


@dataclass(frozen=True)
class Agreement:
    abs_diff: float | None
    icc: float | None
    range_lo: float | None
    range_hi: float | None
    n_events: int
    n_subjects: int
    reasons: tuple[str, ...]


def agreement(
    events: Mapping[str, Sequence[float]],
    subject_of: Mapping[str, str],
    key: str,
    floor: int,
) -> Agreement:
    """Agreement fields for one (level, feature, task, side) row (D01-D03).

    `events` = event id -> finite asset values (one per view); events with < 2 views drop out.
    """
    multi = {e: list(v) for e, v in events.items() if len(v) >= 2}
    n_subjects = len({subject_of[e] for e in multi})
    if not multi or n_subjects < floor:
        reason = "agreement_no_multiview" if not multi else "agreement_below_floor"
        return Agreement(None, None, None, None, len(multi), n_subjects, (reason,))
    abs_diff = float(np.median([mean_pair_abs_diff(v) for v in multi.values()]))
    icc = icc_oneway(list(multi.values()))
    if icc is None:
        return Agreement(abs_diff, None, None, None, len(multi), n_subjects,
                         ("agreement_zero_variance",))  # fmt: skip
    draws = [d for d in resample_icc(multi, subject_of, seed_for(key)) if d is not None]
    if len(draws) < math.ceil(MIN_FINITE_SHARE * RESAMPLES):
        return Agreement(abs_diff, icc, None, None, len(multi), n_subjects,
                         ("agreement_insufficient_resamples",))  # fmt: skip
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Agreement(abs_diff, icc, float(lo), float(hi), len(multi), n_subjects, ())
