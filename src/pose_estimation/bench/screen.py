"""Public screening (contract M4.2): result rows, PCK, AUROC, paired bootstraps, the finalist rule.

Pose-only protocol: every arm sees the same GT boxes, so arms pair person by person and image by
image — every interval here is a paired bootstrap over the same draws.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from .. import score_calibration
from .labels import HAND_JOINTS
from .public import GtPerson

#: mmpose `results2json` field split of the 133 COCO-WholeBody keypoints.
RESULT_FIELDS = (
    ("keypoints", 17),
    ("foot_kpts", 6),
    ("face_kpts", 68),
    ("lefthand_kpts", 21),
    ("righthand_kpts", 21),
)
HAND_FIRST = {"L": 91, "R": 112}
ARM_JOINTS = range(5, 11)  # shoulders, elbows, wrists
PCK_HAND = (0.05, 0.1, 0.2)
PCK_ARM = (0.02, 0.05, 0.1)
#: D03: a finalist's arm-joint PCK@0.05 stays within this of the best arm's.
ARM_PCK_SLACK = 0.01


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie) — the Mann-Whitney form of the ROC area."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUROC needs both classes")
    allv = np.concatenate([pos, neg])
    _, inverse, counts = np.unique(allv, return_inverse=True, return_counts=True)
    # Average rank of each distinct value = rank of its first copy + (copies - 1) / 2.
    first = np.cumsum(counts) - counts + 1
    ranks = (first + (counts - 1) / 2)[inverse]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def result_rows(persons: Sequence[GtPerson], kps: np.ndarray, scores: np.ndarray) -> list[dict]:
    """One mmpose result per person: `(x, y, raw score)` triples per field, score = mean raw."""
    rows = []
    cuts = np.cumsum([0] + [n for _, n in RESULT_FIELDS])
    for p, k, sc in zip(persons, kps, scores, strict=True):
        flat = np.concatenate([k, sc[:, None]], axis=1)
        row = {"image_id": p.image_id, "category_id": 1, "score": float(sc.mean())}
        for i, (name, _) in enumerate(RESULT_FIELDS):
            row[name] = flat[cuts[i] : cuts[i + 1]].ravel().round(3).tolist()
        rows.append(row)
    return rows


def pck(errors: Sequence[float], thresholds: Sequence[float]) -> dict[str, float | int]:
    e = np.asarray(errors, dtype=float)
    if not len(e):
        raise ValueError("PCK over no joints")
    return {f"@{t}": float(np.mean(e < t)) for t in thresholds} | {"n": len(e)}


def hand_errors(persons: Sequence[GtPerson], kps: np.ndarray) -> dict[str, list[float]]:
    """Annotated hand joints (`c > 0`): error / GT hand-box diagonal, all 21 + the 5 labelled."""
    feature = np.zeros(21, bool)
    feature[list(HAND_JOINTS.values())] = True
    out: dict[str, list[float]] = {"all21": [], "feature5": []}
    for p, k in zip(persons, kps, strict=True):
        for h in p.hands:
            first = HAND_FIRST[h.side]
            ok = h.kpts[:, 2] > 0
            err = np.hypot(*(k[first : first + 21] - h.kpts[:, :2]).T) / h.diagonal
            out["all21"] += err[ok].tolist()
            out["feature5"] += err[ok & feature].tolist()
    return out


def arm_errors(
    persons: Sequence[GtPerson], kps: np.ndarray, scores: np.ndarray
) -> tuple[list[float], list[float], list[float]]:
    """Arm joints: error / person-box diagonal where `v = 2`; raw scores at `v = 2` and `v = 1`."""
    errors, visible, occluded = [], [], []
    for p, k, sc in zip(persons, kps, scores, strict=True):
        diag = float(np.hypot(p.box[2], p.box[3]))
        for j in ARM_JOINTS:
            if p.body[j, 2] == 2:
                errors.append(float(np.hypot(*(k[j] - p.body[j, :2])) / diag))
                visible.append(float(sc[j]))
            elif p.body[j, 2] == 1:
                occluded.append(float(sc[j]))
    return errors, visible, occluded


def interval(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_pck_bootstrap(
    correct: Mapping[str, np.ndarray], valid: np.ndarray, n: int, seed: int
) -> dict[str, dict]:
    """Per arm PCK over `(units, joints)` + paired percentile intervals, units resampled jointly.

    `correct[arm]` and `valid` share one shape; a unit = one image (one hand per image here).
    Units without a valid joint (e.g. hands the detector missed) are outside the estimand, so
    they are dropped before resampling: every draw then has a nonzero denominator.
    """
    valid = np.asarray(valid, bool)
    if not valid.any():
        raise ValueError("PCK bootstrap over no valid joints")
    units = valid.any(axis=1)
    valid = valid[units]
    names = sorted(correct)
    hits = {a: (np.asarray(correct[a], bool)[units] & valid).sum(axis=1) for a in names}
    totals = valid.sum(axis=1)
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {a: [] for a in names}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(totals), len(totals)), minlength=len(totals))
        denominator = float(np.dot(w, totals))
        for a in names:
            boots[a].append(float(np.dot(w, hits[a])) / denominator)
    return {
        "arms": {
            a: {"pck": float(hits[a].sum() / totals.sum()), "interval": interval(boots[a])}
            for a in names
        },
        "pairs": {
            f"{x}-{y}": {
                "difference": float((hits[x].sum() - hits[y].sum()) / totals.sum()),
                "interval": interval(np.subtract(boots[x], boots[y])),
            }
            for x, y in itertools.permutations(names, 2)
        },
    }


def finalists(
    arm_pck: Mapping[str, float],
    beats: Sequence[Mapping[str, Mapping]],
    baseline: str,
) -> dict[str, list[str]]:
    """D03 → `{arm: reasons}`; an arm with no reason is a finalist, the baseline always is.

    `beats` = one pair table per screening metric (`"<x>-<y>"` → paired `interval` of x - y). An
    arm falls when another arm's interval lies above zero on EVERY metric (dominance), or when its
    arm-joint PCK@0.05 trails the best arm's by more than `ARM_PCK_SLACK`.
    """
    best = max(arm_pck.values())
    out: dict[str, list[str]] = {}
    for arm in sorted(arm_pck):
        reasons = []
        if arm != baseline:
            for other in sorted(arm_pck):
                key = f"{other}-{arm}"
                if other != arm and all(table[key]["interval"][0] > 0 for table in beats):
                    reasons.append(f"dominated by {other}")
            if arm_pck[arm] < best - ARM_PCK_SLACK:
                reasons.append(
                    f"arm PCK@0.05 {arm_pck[arm]:.4f} < best {best:.4f} - {ARM_PCK_SLACK}"
                )
        out[arm] = reasons
    return out


#: Correct = normalised error under the group's tolerance (M4.3a): body/foot of the person-box
#: diagonal, face likewise but tighter, hand of the hand-box diagonal.
CAL_TOLERANCE = {"body": 0.05, "foot": 0.05, "face": 0.02, "hand": 0.1}


def calibration_targets(persons, kps, scores) -> tuple[dict, dict]:
    """Per keypoint group: raw scores + whether each prediction is correct (CAL_TOLERANCE)."""
    raw: dict[str, list[np.ndarray]] = defaultdict(list)
    correct: dict[str, list[np.ndarray]] = defaultdict(list)
    for p, k, sc in zip(persons, kps, scores, strict=True):
        diag = float(np.hypot(p.box[2], p.box[3]))
        for group, (lo, hi) in score_calibration.GROUPS.items():
            if group == "hand":
                sides = {h.side for h in p.hands}
                for h in p.hands:
                    first = 91 if h.side == "L" else 112
                    err = np.hypot(*(k[first : first + 21] - h.kpts[:, :2]).T) / h.diagonal
                    raw[group].append(sc[first : first + 21])
                    correct[group].append((h.kpts[:, 2] > 0) & (err < CAL_TOLERANCE["hand"]))
                for side, first in (("L", 91), ("R", 112)):
                    if side not in sides:  # unannotated hand: no prediction there is correct
                        raw[group].append(sc[first : first + 21])
                        correct[group].append(np.zeros(21, bool))
                continue
            gt = gt_part(p, group)
            err = np.hypot(*(k[lo:hi] - gt[:, :2]).T) / diag
            raw[group].append(sc[lo:hi])
            correct[group].append((gt[:, 2] > 0) & (err < CAL_TOLERANCE[group]))
    return (
        {g: np.concatenate(v) for g, v in raw.items()},
        {g: np.concatenate(v) for g, v in correct.items()},
    )


def gt_part(person, group: str) -> np.ndarray:
    """The person's GT rows for a non-hand group, from the raw COCO-WholeBody annotation."""
    if group == "body":
        return person.body
    return person.extra[group]
