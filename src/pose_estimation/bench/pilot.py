"""M4.3 pilot evaluation: a replayed configuration's exported row vs the labellers' consensus.

Contract `.agent/archive/contract-m4u3.md`. Exported coordinates are normalised by the frame's
max dimension (`export.coord_scale`), labels are pixels of the same auto-oriented frame, so one
scalar converts; a 90-degree display rotation leaves `max(w, h)` unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .labels import HAND_JOINTS
from .qualify import PRESENT, exported_hands, presence_f1

GROSS_HAND = 0.3  # of the labelled hand extent diagonal (M4.1 qualified resolution)
GROSS_ARM = 0.1  # of the consensus patient-box diagonal
ENDPOINTS = ("G1", "G2", "G3", "G4")
#: Decision margins (contract §3): lower bound for G1-G3 differences, upper for G4.
MARGIN = 0.05
REFIT_POINT = -0.02  # M4.3b D02: a presence-F1 point loss beyond this refits the gate
_WORD = {"L": "left", "R": "right"}


def check_request(arm: str) -> None:
    """P04: refuse an arm the pipeline cannot run."""
    from pose_estimation.run import MODEL_REGISTRY  # deferred: run imports the inference stack

    if arm not in MODEL_REGISTRY:
        raise ValueError(f"{arm!r} is not a run.MODEL_REGISTRY key")


def _point(row: Mapping[str, str] | None, prefix: str, confidence: str) -> np.ndarray | None:
    """Exported (x, y) in normalised units, or None where the R stage would mask it."""
    if not row or row.get(f"{prefix}_x", "") == "":
        return None
    score = row.get(f"{prefix}_{confidence}") or ""
    if not score or not math.isfinite(float(score)) or float(score) <= 0:
        return None
    return np.array([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])])


def _diagonal(box: Sequence[float]) -> float:
    return float(math.hypot(box[2] - box[0], box[3] - box[1]))


def _hits(row, joints: Mapping, group: str, column, confidence: str, scale: float, tol: float):
    """(exported within `tol` px, labelled `vis`) over `group` joints; not exported = a miss."""
    hits = n = 0
    for name, j in joints.items():
        if not name.startswith(group) or j["state"] != "vis":
            continue
        p = _point(row, column(name.split(".", 1)[1]), confidence)
        n += 1
        hits += p is not None and math.dist(p * scale, j["xy"]) < tol
    return hits, n


def side_rows(
    subject, consensus: Mapping, row: Mapping[str, str] | None, scale: float
) -> list[dict]:
    """One record per side: presence, gross hand + arm hits, swap evidence (`?` states out)."""
    present = exported_hands(row)
    hands = {}
    for side in ("L", "R"):
        pts = [
            j["xy"]
            for n, j in consensus["sides"][side]["joints"].items()
            if n.startswith("joints.") and j["state"] == "vis"
        ]
        if consensus["sides"][side]["hand"] in ("vis", "part") and pts:
            hands[side] = np.mean(pts, axis=0)
    patient = consensus["patient"]
    out = []
    for side in ("L", "R"):
        s = consensus["sides"][side]
        word = _WORD[side]
        rec: dict = {"subject": subject, "side": side}
        if s["hand"] in PRESENT:
            rec |= {"present": PRESENT[s["hand"]], "exported": present[side]}
        hand_ok = s["hand"] in ("vis", "part") and bool(s["extent"])
        rec["hand_hits"], rec["hand_n"] = _hits(
            row,
            s["joints"] if hand_ok else {},
            "joints.",
            lambda name, w=word: f"{w}_hand_{HAND_JOINTS[name]}",
            "conf",
            scale,
            GROSS_HAND * _diagonal(s["extent"]) if hand_ok else 0.0,
        )
        rec["arm_hits"], rec["arm_n"] = _hits(
            row,
            s["joints"] if patient else {},
            "body.",
            lambda name, w=word: f"body_{w}_{name}",
            "vis",
            scale,
            GROSS_ARM * _diagonal(patient) if patient else 0.0,
        )
        if len(hands) == 2 and present[side]:
            pts = [_point(row, f"{_WORD[side]}_hand_{i}", "conf") for i in range(21)]
            pts = [p * scale for p in pts if p is not None]
            other = "R" if side == "L" else "L"
            centre = np.mean(pts, axis=0)
            rec["swap"] = bool(math.dist(centre, hands[other]) < math.dist(centre, hands[side]))
        out.append(rec)
    return out


def endpoints(rows: Sequence[dict]) -> dict[str, float]:
    """G1-G4 pooled over the rows; NaN where an endpoint has no eligible side."""

    def share(hits: str, n: str) -> float:
        total = sum(r[n] for r in rows)
        return sum(r[hits] for r in rows) / total if total else math.nan

    labelled = [r for r in rows if "present" in r]
    swaps = [r["swap"] for r in rows if "swap" in r]
    return {
        "G1": presence_f1(labelled) if labelled else math.nan,
        "G2": share("hand_hits", "hand_n"),
        "G3": share("arm_hits", "arm_n"),
        "G4": float(np.mean(swaps)) if swaps else math.nan,
    }


def paired_intervals(
    arm: Sequence[dict], base: Sequence[dict], n: int, seed: int
) -> dict[str, tuple[float, float, float]]:
    """(arm - baseline) point + 95 % percentile interval per endpoint, subjects resampled."""
    subjects = sorted({r["subject"] for r in base} | {r["subject"] for r in arm})
    by = {
        name: {s: [r for r in rows if r["subject"] == s] for s in subjects}
        for name, rows in (("arm", arm), ("base", base))
    }
    point_a, point_b = endpoints(arm), endpoints(base)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {g: [] for g in ENDPOINTS}
    for _ in range(n):
        pick = rng.choice(len(subjects), len(subjects))
        a = endpoints([r for i in pick for r in by["arm"][subjects[i]]])
        b = endpoints([r for i in pick for r in by["base"][subjects[i]]])
        for g in ENDPOINTS:
            if not (math.isnan(a[g]) or math.isnan(b[g])):
                draws[g].append(a[g] - b[g])
    return {
        g: (
            point_a[g] - point_b[g],
            float(np.percentile(draws[g], 2.5)) if draws[g] else math.nan,
            float(np.percentile(draws[g], 97.5)) if draws[g] else math.nan,
        )
        for g in ENDPOINTS
    }


def decide(intervals: Mapping[str, tuple[float, float, float]]) -> str:
    """Contract §3: `pass`, `refit` (presence alone breaches) or `fail`."""
    (g1, g1_lo, _), (_, g2_lo, _), (_, g3_lo, _), (_, _, g4_hi) = (intervals[g] for g in ENDPOINTS)
    others_hold = g2_lo >= -MARGIN and g3_lo >= -MARGIN and g4_hi <= MARGIN
    presence_holds = g1_lo >= -MARGIN and g1 >= REFIT_POINT
    if others_hold and presence_holds:
        return "pass"
    return "refit" if others_hold else "fail"
