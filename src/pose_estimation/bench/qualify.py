"""Instrument qualification statistics (contract M4.1 D05-D06)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np

NON_DECISIVE = frozenset({"both-ok", "both-wrong", "?"})


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        raise ValueError("Wilson interval over an empty population")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.5, 0.9)) -> list[float]:
    if len(values) == 0:
        raise ValueError("quantiles of an empty population")
    return [float(v) for v in np.quantile(np.asarray(values, dtype=float), qs)]


def normalized_errors(pred: np.ndarray, gt: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """Per-point Euclidean error divided by its normaliser (hand / person box diagonal)."""
    pred, gt, norm = (np.asarray(a, dtype=float) for a in (pred, gt, norm))
    if pred.size == 0:
        raise ValueError("errors over an empty population")
    if np.any(norm <= 0):
        raise ValueError("non-positive normaliser")
    return np.hypot(*(pred - gt).T) / norm


def agreement(a: Sequence[str], b: Sequence[str]) -> tuple[int, int]:
    """(agreeing, compared) over paired categorical labels, `?` on either side excluded."""
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x != "?" and y != "?"]
    if not pairs:
        raise ValueError("no decided pair to compare")
    return sum(x == y for x, y in pairs), len(pairs)


def class_agreement(a: Sequence[str], b: Sequence[str], classes: Sequence[str]) -> dict[str, dict]:
    """Per class: how often one rater's call of it is repeated by the other, both directions.

    The bar (contract-m4u1 D06) is per class — a pooled interval lets a common class carry a rare
    one. `?` on either side is excluded; `lower` = the smaller Wilson lower bound of the two
    directions, None when either rater never calls the class.
    """
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x != "?" and y != "?"]
    out = {}
    for c in classes:
        both = sum(x == c and y == c for x, y in pairs)
        first, second = sum(x == c for x, _ in pairs), sum(y == c for _, y in pairs)
        lower = min(wilson(both, first)[0], wilson(both, second)[0]) if first and second else None
        out[c] = {"agree": both, "first": first, "second": second, "lower": lower}
    return out


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two `[x0, y0, x1, y1]` boxes."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("box_iou needs two [x0, y0, x1, y1] boxes")
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def judge_bins(
    margins: Sequence[float], truths: Sequence[str], outcomes: Sequence[str], bins: Sequence[float]
) -> list[dict[str, float | int]]:
    """Per margin bin: decisive answers, correct decisive answers, abstentions, Wilson bounds.

    `truths` = the arm with the smaller GT error; `outcomes` = decoded answers (an arm name or a
    non-decisive class). A decisive answer is correct when it names the truth.
    """
    if not (len(margins) == len(truths) == len(outcomes)):
        raise ValueError("margins, truths and outcomes differ in length")
    if not margins:
        raise ValueError("judge bins over an empty population")
    rows = []
    for lo, hi in pairwise(bins):
        idx = [i for i, m in enumerate(margins) if lo <= m < hi]
        decisive = [i for i in idx if outcomes[i] not in NON_DECISIVE]
        correct = sum(outcomes[i] == truths[i] for i in decisive)
        row: dict[str, float | int] = {
            "lo": lo,
            "hi": hi,
            "shown": len(idx),
            "decisive": len(decisive),
            "correct": correct,
            "abstain": len(idx) - len(decisive),
        }
        if decisive:
            row["acc_lo"], row["acc_hi"] = wilson(correct, len(decisive))
        rows.append(row)
    return rows


def paired_subject_sd(diffs_by_subject: dict[int, Sequence[float]]) -> tuple[float, int]:
    """SD over subjects of each subject's mean paired difference, and the subject count."""
    means = [float(np.mean(v)) for v in diffs_by_subject.values() if len(v)]
    if len(means) < 2:
        raise ValueError("paired-subject SD needs two subjects")
    return float(np.std(means, ddof=1)), len(means)


#: Agreed labeller hand state → present; `?` (labellers disagree) + unlabelled states drop out.
PRESENT = {"vis": True, "part": True, "occ": False, "oof": False}


def exported_hands(row: Mapping[str, str] | None) -> dict[str, bool]:
    """Per side: the exported landmark row carries >= 1 hand point with a positive confidence."""
    return {
        side: bool(row)
        and any(
            row.get(f"{word}_hand_{i}_x", "") != ""
            and float(row.get(f"{word}_hand_{i}_conf") or 0) > 0
            for i in range(21)
        )
        for side, word in (("L", "left"), ("R", "right"))
    }


def presence_rows(subject, states: dict[str, str], exported: dict[str, bool]) -> list[dict]:
    return [
        {"subject": subject, "side": side, "present": PRESENT[state], "exported": exported[side]}
        for side, state in states.items()
        if state in PRESENT
    ]


def _counts(rows: Sequence[dict]) -> tuple[int, int, int]:
    tp = sum(r["exported"] and r["present"] for r in rows)
    fp = sum(r["exported"] and not r["present"] for r in rows)
    fn = sum(not r["exported"] and r["present"] for r in rows)
    return tp, fp, fn


def presence_f1(rows: Sequence[dict]) -> float:
    tp, fp, fn = _counts(rows)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def presence_summary(rows: Sequence[dict], seed: int, n_boot: int = 2000) -> dict:
    """Precision, recall, F1 + a 95 % interval from resampling subjects (rows cluster by subject)."""
    by_subject: dict = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(seed)
    boots = [
        presence_f1([r for s in rng.choice(subjects, len(subjects)) for r in by_subject[s]])
        for _ in range(n_boot)
    ]
    tp, fp, fn = _counts(rows)
    return {
        "sides": len(rows),
        "subjects": len(subjects),
        "f1": presence_f1(rows),
        "f1_interval": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }
