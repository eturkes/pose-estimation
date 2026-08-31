#!/usr/bin/env python3
"""Decide whether this corpus's cross-view keypoint bias is a GLOBAL field or a per-event artifact.

M2.6 closed F0 negative on a measured cause: per-keypoint mean signed epipolar residuals reproduce
across two DISJOINT frame blocks of one event at r median 0.703, calibrated against 0.010-0.120 for
zero-mean noise and 0.993-0.998 for fixed bias. That measured a bias inside an event. It said
nothing about whether the same bias returns on a DIFFERENT recording of the same view pair.

The distinction gates the whole repair route. A bias that is a function of (view, keypoint) can be
estimated on training events and applied to held-out ones, which is the only form of the repair that
recovers anything: a bias estimable only jointly with the pose it corrupts, per event, buys nothing.
A bias that is a per-event artifact is unmodelable no matter how the estimator is written, and the
repair is dead before an estimator exists.

  reproduce  BETWEEN-event correlation of per-keypoint mean signed epipolar residual vectors,
             grouped by ordered view pair, read against the WITHIN-event split-half r that is this
             statistic's own ceiling on the same pairs. Synthetic arms calibrate it, each run
             through the identical statistic: a bias field shared across every event (the repairable
             mechanism), as an image-fixed offset and as Malleson's 3D per-(camera, keypoint) offset;
             an independent field per event (the unrepairable one); and zero-mean noise (no bias at
             all), at magnitudes bracketing the corpus's measured 15-20 px systematic component. The
             corpus is also grouped by device-model pair, task and subject, each stricter than the
             last, and read against a keypoint-permutation null on its own vectors.

Synthetic events reuse the real cache's validity masks, image sizes and device models; only the rig,
the structure and the bias become known. Rigs are jittered per event, so a synthetic between-event
correlation is measured across different camera placements exactly as the corpus one is, and a rig
jitter sweep bounds how much placement variation the statistic survives.

Output is redaction-safe: distributions and counts only, no identifier, filename or per-event key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_calibration_bias import (
    KP,
    MIN_KEYPOINTS,
    _pairs_of,
    _stats,
    load_event,
    pair_structure,
    synthesize,
)
from probe_calibration_observability import SEED

SYNTHETIC_VIEWS = ("above", "left", "right")  # `_rig`'s nominal camera order
# `_event_key` is sha256(event_id)[:24] and `event_id` is `s<NN>-<task>-<side>_run-<NN>` over a
# closed alphabet, so the cache's hashed keys inverte by enumeration -- which recovers the subject
# grouping without reading one patient-adjacent table. `load_labelled` asserts every key resolves,
# so a grammar change fails loudly instead of silently emptying the subject arm.
SUBJECTS = 16
TASKS = ("cap", "peg", "nut", "coin", "glass", "key")
SIDES = ("l", "r")
RUNS = 9
BIAS_SWEEP_PX = (8.0, 32.0)
ANATOMICAL_SWEEP_MM = (20.0, 40.0, 80.0)
RIG_JITTER_SWEEP_M = (0.60, 1.20)  # no rig was ever built; placement moved between takes
SYNTHETIC_EVENTS = 20
FIELD_REALIZATIONS = 3
STRONG_R = 0.5


def _subject_by_event_key() -> dict[str, str]:
    return {
        hashlib.sha256(f"s{subject:02d}-{task}-{side}_run-{run:02d}".encode()).hexdigest()[
            :24
        ]: f"s{subject:02d}"
        for subject in range(1, SUBJECTS + 1)
        for task in TASKS
        for side in SIDES
        for run in range(RUNS)
    }


def load_labelled(cache: Path) -> list[dict[str, Any]]:
    """Cache entries carrying their view labels, device models, task and subject.

    Camera index order is view order, which is what makes an ordered view-pair label comparable
    across events.
    """
    subjects = _subject_by_event_key()
    events = []
    for path in sorted(cache.glob("*.npz")):
        event = load_event(path)
        if event is None:
            continue
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["meta"].tobytes().decode())
        views = list(meta["views"])
        if views != sorted(views):
            raise RuntimeError("cache camera order is not view order; pair labels would not align")
        if meta["event_key"] not in subjects:
            raise RuntimeError("cache key does not invert to an event id; the grammar has moved")
        event["views"] = views
        event["task"] = meta["task"]
        event["subject"] = subjects[meta["event_key"]]
        events.append(event)
    return events


def _vectors(
    event: dict[str, Any], views: list[str], task: str, subject: str
) -> list[dict[str, Any]]:
    rows = []
    for left, right in _pairs_of(event["cameras"]):
        row = pair_structure(event, left, right)
        if row is None:
            continue
        rows.append(
            {
                "view_pair": f"{views[left]}|{views[right]}",
                "model_pair": "|".join(sorted((event["models"][left], event["models"][right]))),
                "task": task,
                "subject": subject,
                "bias_px": row["keypoint_bias_px"],
                "split_r": row["split_r"],
                "median_abs_px": row["median_abs_px"],
            }
        )
    return rows


def real_vectors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, event in enumerate(events):
        rows.extend(
            {**row, "event": index}
            for row in _vectors(event, event["views"], event["task"], event["subject"])
        )
    return rows


def synthetic_vectors(
    events: list[dict[str, Any]], *, count: int, **kwargs
) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        event = events[index % len(events)]
        observed, focals, _ = synthesize(
            event["mask"], event["sizes"], event["models"], seed=SEED + index, **kwargs
        )
        probe = {
            "observed": observed,
            "mask": event["mask"],
            "focals": focals,
            "cameras": 3,
            "models": event["models"],
        }
        rows.extend(
            {**row, "event": index}
            for row in _vectors(probe, list(SYNTHETIC_VIEWS), event["task"], event["subject"])
        )
    return rows


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    usable = np.isfinite(left) & np.isfinite(right)
    if int(usable.sum()) < MIN_KEYPOINTS:
        return None
    a, b = left[usable], right[usable]
    if a.std() <= 0 or b.std() <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _between(
    rows: list[dict[str, Any]],
    *keys: str,
    permute: bool = False,
    absolute: bool = False,
    seed: int = SEED,
) -> list[float]:
    """Correlations over every pair of DISTINCT events; `keys` restrict to matching groups.

    `permute` shuffles the second vector's keypoint order, which destroys any shared per-keypoint
    structure while preserving the marginal distribution and the keypoint count. It is the null this
    statistic has to be read against: with ~40 usable keypoints a sample correlation carries real
    spread, so a small median is only evidence once the permutation spread is known.

    `absolute` correlates residual MAGNITUDE instead of signed offset, which separates the two ways
    keypoints can agree across events. Shared magnitude alone is shared difficulty -- the same joints
    are hard everywhere -- and carries no correctable offset. Only shared SIGN is a modelable bias.
    """
    rng = np.random.default_rng(seed)
    values = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["event"] == rows[j]["event"]:
                continue
            if any(rows[i][key] != rows[j][key] for key in keys):
                continue
            left, right = rows[i]["bias_px"], rows[j]["bias_px"]
            if permute:
                right = right[rng.permutation(len(right))]
            if absolute:
                left, right = np.abs(left), np.abs(right)
            value = _corr(left, right)
            if value is not None:
                values.append(value)
    return values


def _r_stats(values: list[float]) -> dict[str, Any]:
    out = _stats(values, 4)
    out["above_0p5"] = sum(value > STRONG_R for value in values)
    return out


def _emit(
    label: str, groups: list[list[dict[str, Any]]], *keys: str, **kwargs: Any
) -> dict[str, Any]:
    """Correlate within each group, pool the values. One group per independent bias realization."""
    signed = [value for group in groups for value in _between(group, *keys, **kwargs)]
    magnitude = [
        value for group in groups for value in _between(group, *keys, absolute=True, **kwargs)
    ]
    rows = [row for group in groups for row in group]
    out: dict[str, Any] = {
        "label": label,
        "pairs": len(rows),
        "events": len(groups[0]) and len({row["event"] for row in groups[0]}),
        "realizations": len(groups),
        "between_event_r": _r_stats(signed),
        "between_event_r_abs": _r_stats(magnitude),
        "within_event_r": _r_stats([row["split_r"] for row in rows]),
        "median_abs_px": _stats([row["median_abs_px"] for row in rows]),
    }
    ceiling = [row["split_r"] for row in rows if math.isfinite(row["split_r"])]
    finite = [value for value in signed if math.isfinite(value)]
    out["shared_fraction"] = (
        round(float(np.median(finite) / np.median(ceiling)), 4)
        if finite and ceiling and np.median(ceiling) > 0
        else None
    )
    print(json.dumps(out), flush=True)
    return out


def arm_reproduce(cache: Path, *, synthetic_events: int, realizations: int) -> dict[str, Any]:
    results = []
    labelled = load_labelled(cache)
    real = real_vectors(labelled)
    results.append(_emit("REAL same view pair", [real], "view_pair"))
    for view_pair in sorted({row["view_pair"] for row in real}):
        subset = [row for row in real if row["view_pair"] == view_pair]
        results.append(_emit(f"REAL {view_pair}", [subset], "view_pair"))
    results.append(
        _emit("REAL same view pair + same model pair", [real], "view_pair", "model_pair")
    )
    results.append(_emit("REAL same view pair + same task", [real], "view_pair", "task"))
    # The last live variant of the repair: a bias that is a property of the SUBJECT's anatomy would
    # transfer across that subject's own events, which pools ~7 events per bias field instead of one.
    results.append(_emit("REAL same view pair + same subject", [real], "view_pair", "subject"))
    results.append(
        _emit("REAL same view pair, keypoints permuted (null)", [real], "view_pair", permute=True)
    )

    events = [event for event in labelled if event["cameras"] == 3]
    if not events:
        return {"results": results}
    keypoints = len(KP)
    rng = np.random.default_rng(SEED)
    # One field draw is one realization of the mechanism, and the reference varies across draws, so
    # every synthetic arm pools `realizations` independent fields before its median is read.
    arms: list[tuple[str, list[dict[str, Any]]]] = []
    for magnitude in BIAS_SWEEP_PX:
        fields = [rng.normal(0.0, magnitude, (3, keypoints, 2)) for _ in range(realizations)]
        arms.append(
            (
                f"SYNTH shared image bias {magnitude}px",
                [{"bias_field": field} for field in fields],
            )
        )
        arms.extend(
            (
                f"SYNTH shared image bias {magnitude}px, rig jitter {jitter}m",
                [{"bias_field": field, "rig_jitter_m": jitter} for field in fields],
            )
            for jitter in RIG_JITTER_SWEEP_M
        )
    # Malleson's parameterization: a constant 3D detector-to-bone offset per (camera, keypoint). Its
    # image residual depends on viewing direction, so placement variation can decorrelate a bias that
    # IS shared -- which is the one way the corpus null could be an artifact of this statistic.
    arms.extend(
        (
            f"SYNTH shared anatomical bias {millimetres}mm",
            [
                {"anatomical_field": rng.normal(0.0, millimetres / 1000.0, (3, keypoints, 3))}
                for _ in range(realizations)
            ],
        )
        for millimetres in ANATOMICAL_SWEEP_MM
    )
    arms += [
        (f"SYNTH per-event bias {value}px", [{"image_bias_px": value}]) for value in BIAS_SWEEP_PX
    ]
    arms += [(f"SYNTH noise sigma={value}px", [{"sigma_px": value}]) for value in BIAS_SWEEP_PX]
    for label, draws in arms:
        groups = [synthetic_vectors(events, count=synthetic_events, **kwargs) for kwargs in draws]
        results.append(_emit(label, groups, "view_pair"))
    return {"results": results}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / ".scratch" / "calib-obs-f32")
    parser.add_argument("--synthetic-events", type=int, default=SYNTHETIC_EVENTS)
    parser.add_argument("--field-realizations", type=int, default=FIELD_REALIZATIONS)
    args = parser.parse_args(argv)
    if not args.cache.is_dir():
        raise SystemExit(
            "no keypoint cache found; run probe_calibration_observability.py collect first"
        )
    report = {
        "seed": SEED,
        "reproduce": arm_reproduce(
            args.cache,
            synthetic_events=args.synthetic_events,
            realizations=args.field_realizations,
        ),
    }
    print(json.dumps({"summary": sorted(report)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
