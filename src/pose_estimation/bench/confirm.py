"""Locked confirmation scorer (contract M4.4): user-clicked frames vs exported rows.

Endpoints per frame → per subject means → paired subject-cluster bootstrap of candidate - baseline
→ the frozen decision table. Exported rows are landmark-CSV rows (isotropic coordinates: pixel =
value x max(frame width, height)); user records are codebook v2 in source pixels.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import HAND_JOINTS, SIDES

SIDE_WORD = {"L": "left", "R": "right"}
#: Non-inferiority margins + the superiority test, frozen in contract M4.4 §3.
MARGINS = {"E1": 0.02, "E2": 0.02, "E3": -0.05, "E4": 0.02}
N_BOOT = 10_000
BOOT_SEED = 20261021


@dataclass(frozen=True)
class FrameScore:
    subject: str
    e1: float | None  # median normalised hand-joint error (misses = 1.0)
    e2: float | None  # median arm-joint error / upper-arm length
    e3: tuple[bool, ...]  # per decided side: exported presence == user presence
    e4: bool | None  # side swap


def _xy(row: Mapping[str, str], prefix: str, scale: float) -> np.ndarray | None:
    """The exported point in pixels, or None when the row carries no observation there.

    An observation = coordinates + a positive score (`_vis` body, `_conf` hand): the R stage
    reads a zero score as missing, so a zero-score point is no export.
    """
    x, y = row.get(f"{prefix}_x", ""), row.get(f"{prefix}_y", "")
    score = row.get(f"{prefix}_vis", row.get(f"{prefix}_conf", ""))
    if "" in (x, y, score) or None in (x, y, score) or not float(score) > 0:
        return None
    return np.array([float(x) * scale, float(y) * scale])


def _visible(joint: Mapping) -> np.ndarray | None:
    return np.asarray(joint["xy"], float) if joint.get("state") == "vis" else None


def score_frame(record: Mapping, row: Mapping[str, str] | None, subject: str) -> FrameScore:
    """E1-E4 for one user record against the exported row at that frame (`None` = no row)."""
    row = row or {}
    scale = float(max(record["frame"]["width"], record["frame"]["height"]))
    hand_errors: list[float] = []
    arm_errors: list[float] = []
    presence: list[bool] = []
    sides = record["sides"]
    upper_arm = {}
    for side in SIDES:
        sh, el = _visible(sides[side]["body"]["shoulder"]), _visible(sides[side]["body"]["elbow"])
        if sh is not None and el is not None:
            upper_arm[side] = float(np.hypot(*(sh - el)))
    for side in SIDES:
        word = SIDE_WORD[side]
        joints = sides[side]["joints"]
        visible = {n: _visible(j) for n, j in joints.items() if _visible(j) is not None}
        exported = any(_xy(row, f"{word}_hand_{i}", scale) is not None for i in range(21))
        states = {j.get("state") for j in joints.values()}
        if states != {"?"}:
            presence.append(exported == bool(visible))
        if len(visible) >= 2:
            pts = np.array(list(visible.values()))
            diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
            for name, xy in visible.items():
                got = _xy(row, f"{word}_hand_{HAND_JOINTS[name]}", scale)
                # 1.0 = a miss (contract §2); an exported point keeps its own error, however large.
                hand_errors.append(1.0 if got is None else float(np.hypot(*(got - xy))) / diag)
        norm = upper_arm.get(side) or upper_arm.get("R" if side == "L" else "L")
        if norm:
            for name in ("shoulder", "elbow", "wrist"):
                xy = _visible(sides[side]["body"][name])
                if xy is None:
                    continue
                got = _xy(row, f"body_{word}_{name}", scale)
                arm_errors.append(1.0 if got is None else float(np.hypot(*(got - xy))) / norm)
    swap = None
    wl, wr = _visible(sides["L"]["body"]["wrist"]), _visible(sides["R"]["body"]["wrist"])
    gl, gr = _xy(row, "body_left_wrist", scale), _xy(row, "body_right_wrist", scale)
    if wl is not None and wr is not None and gl is not None and gr is not None:
        swap = bool(
            np.hypot(*(gl - wr)) < np.hypot(*(gl - wl))
            and np.hypot(*(gr - wl)) < np.hypot(*(gr - wr))
        )
    return FrameScore(
        subject,
        float(np.median(hand_errors)) if hand_errors else None,
        float(np.median(arm_errors)) if arm_errors else None,
        tuple(presence),
        swap,
    )


def subject_means(scores: Sequence[FrameScore]) -> dict[str, dict[str, float]]:
    """Per subject = mean over frames: E1, E2, E3 (each frame's agreeing share of sides), E4."""
    out: dict[str, dict[str, float]] = {}
    for subject in sorted({s.subject for s in scores}):
        mine = [s for s in scores if s.subject == subject]
        values: dict[str, list[float]] = {
            "E1": [s.e1 for s in mine if s.e1 is not None],
            "E2": [s.e2 for s in mine if s.e2 is not None],
            # Per frame first (share of its decided sides), then the subject mean (§2 grain).
            "E3": [float(np.mean(s.e3)) for s in mine if s.e3],
            "E4": [float(s.e4) for s in mine if s.e4 is not None],
        }
        out[subject] = {k: float(np.mean(v)) if v else math.nan for k, v in values.items()}
    return out


def bootstrap(
    candidate: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
    endpoint: str,
    n: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """(point, lower, upper) of mean(candidate - baseline) over subjects, subject resampling."""
    subjects = [
        s
        for s in sorted(set(candidate) & set(baseline))
        if not (math.isnan(candidate[s][endpoint]) or math.isnan(baseline[s][endpoint]))
    ]
    if len(subjects) < 2:
        raise ValueError(f"{endpoint}: fewer than 2 subjects with both values")
    d = np.array([candidate[s][endpoint] - baseline[s][endpoint] for s in subjects])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def decide(intervals: Mapping[str, tuple[float, float, float]]) -> str:
    """The contract's decision table over E1-E4 intervals (point, lower, upper)."""
    e1, e2, e3, e4 = (intervals[k] for k in ("E1", "E2", "E3", "E4"))
    holds = (
        e1[2] <= MARGINS["E1"]
        and e2[2] <= MARGINS["E2"]
        and e3[1] >= MARGINS["E3"]
        and e4[2] <= MARGINS["E4"]
    )
    if holds:
        return "superior" if e1[2] < 0 else "non-inferior"
    wrong = (
        (e1[2] > MARGINS["E1"] and e1[0] > 0)
        or (e2[2] > MARGINS["E2"] and e2[0] > 0)
        or (e3[1] < MARGINS["E3"] and e3[0] < 0)
        or (e4[2] > MARGINS["E4"] and e4[0] > 0)
    )
    return "regression" if wrong else "inconclusive"


def adopt(states: Mapping[str, str], public_rank: Sequence[str], baseline: str) -> str:
    """First candidate in public-evidence order whose state is non-inferior or superior."""
    for name in public_rank:
        if states.get(name) in ("non-inferior", "superior"):
            return name
    return baseline


CANDIDATES_LINE = re.compile(r"^- \*\*A01 \(candidates\)\.\*\*(?P<names>.*)$", re.M)


def frozen_candidates(contract: Path) -> list[str]:
    """Candidate names from the contract's A01 amendment; refuses when it is absent (P04)."""
    match = CANDIDATES_LINE.search(contract.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("contract-m4u4 names no candidates (amendment A01): refusing to score")
    return re.findall(r"`([^`]+)`", match.group("names"))
