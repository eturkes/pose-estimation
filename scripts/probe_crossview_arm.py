#!/usr/bin/env python3
"""Cross-view keypoint residual per pose arm: the benchmark half of the 3D-negative ruling.

`.agent/spec.md` Decisions closes 3D on RTMW-m keypoints (15-20 px systematic cross-view bias) and
binds the benchmark to report each arm's cross-view residual, a material drop returning the ruling
to the user. `collect` recollects M2.6's observability cache with the pose model swapped for a
`run.MODEL_REGISTRY` arm. Sampling, synchronisation, the CPU detector and the cache layout stay
the shipped probe's own, imported unchanged: an edit there moves `calibration_qc`'s bound
digests. `compare` scores every arm through `probe_calibration_bias.pair_structure`, the
statistic behind the ruling.

Scores pass through the arm's `score_calibrations/` map before caching, so the probe's 0.5
validity cut keeps one meaning across arms (raw median: RTMW-L 384 4.2, RTMW-m 0.74).

Decision rule, frozen before any arm result: over the (event, camera pair) units every arm scores,
ratio = median `median_abs_px` arm / baseline; material drop = ratio <= 0.75 with its
event-clustered bootstrap 95 % upper bound < 1. 0.75 takes the ~17 px residual the ruling rests on
below its 15-20 px band.

  collect --arm KEY --device DEV --cache DIR [probe flags]   accelerator recipe for NPU/GPU
  compare --cache KEY=DIR ... --baseline KEY                 gate prefix

Output is redaction-safe: counts and distributions only, no event key, filename or identifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_calibration_bias as bias
import probe_calibration_observability as obs
from pose_estimation import score_calibration
from pose_estimation.bench import arms
from pose_estimation.run import MODEL_REGISTRY, CalibratedPose

MATERIAL_RATIO = 0.75
BOOTSTRAP = 2000
SEED = 20261020
ARM_FILE = "arm.json"


def _arm_models(key: str, device: str):
    shipped = obs.make_models

    def make_models() -> obs.Models:
        # The shipped builder keeps its detector + device census; its RTMW-m pose is replaced.
        base = shipped()
        pose = arms.load(arms.POSE_ARMS[key], device)
        placement = obs._execution_devices(pose)
        placed = obs._devices(placement)
        if not placed or not all(device in name for name in placed):
            raise RuntimeError(f"pose did not execute on {device}: {placed!r}")
        calibration = MODEL_REGISTRY[key]["calibration"]
        if calibration is not None:
            pose = CalibratedPose(pose, score_calibration.load(str(calibration)))
        return obs.Models(
            base.detector, pose, {**base.placements, "pose": placement}, base.available_devices
        )

    return make_models


def collect(args: argparse.Namespace, probe_flags: list[str]) -> int:
    args.cache.mkdir(parents=True, exist_ok=True)
    record = {"arm": args.arm, "device": args.device}
    marker = args.cache / ARM_FILE
    if marker.exists() and json.loads(marker.read_text()) != record:
        raise SystemExit(f"{marker} names another arm; collect into its own --cache")
    marker.write_text(json.dumps(record, sort_keys=True) + "\n")
    obs.make_models = _arm_models(args.arm, args.device)
    return obs.main([*probe_flags, "--cache", str(args.cache), "collect"])


def _units(cache: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    units = {}
    for path in sorted(cache.glob("*.npz")):
        event = bias.load_event(path)
        if event is None:
            continue
        for left, right in bias._pairs_of(event["cameras"]):
            row = bias.pair_structure(event, left, right)
            if row is not None:
                units[(path.stem, left, right)] = row
    return units


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "median_abs_px": bias._stats([r["median_abs_px"] for r in rows]),
        "split_r": bias._stats([r["split_r"] for r in rows], 4),
        "split_r_above_0p5": sum(r["split_r"] > 0.5 for r in rows),
    }


def _ratio(arm: np.ndarray, base: np.ndarray) -> float:
    return float(np.median(arm) / np.median(base))


def paired_ratio(
    events: list[str], arm: np.ndarray, base: np.ndarray, seed: int = SEED
) -> dict[str, Any]:
    """Ratio of medians arm / base over paired units, 95 % interval resampling whole events."""
    names = sorted(set(events))
    rows = {name: [i for i, e in enumerate(events) if e == name] for name in names}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP):
        picked = np.concatenate([rows[names[i]] for i in rng.integers(0, len(names), len(names))])
        draws.append(_ratio(arm[picked], base[picked]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    ratio = _ratio(arm, base)
    return {
        "events": len(names),
        "units": len(events),
        "ratio": round(ratio, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "material_drop": bool(ratio <= MATERIAL_RATIO and hi < 1.0),
    }


def compare(args: argparse.Namespace) -> int:
    caches = dict(item.split("=", 1) for item in args.cache)
    if args.baseline not in caches:
        raise SystemExit("--baseline must name one --cache KEY")
    units = {}
    for key, directory in caches.items():
        marker = Path(directory) / ARM_FILE
        if marker.exists() and json.loads(marker.read_text())["arm"] != key:
            raise SystemExit(f"{marker} names another arm than {key}")
        units[key] = _units(Path(directory))
    shared = sorted(set.intersection(*(set(u) for u in units.values())))
    if not shared:
        raise SystemExit("no (event, pair) unit is scored by every arm")
    report: dict[str, Any] = {
        "rule": {"material_ratio": MATERIAL_RATIO, "bootstrap": BOOTSTRAP, "seed": SEED},
        "arms": {
            key: {
                "provenance": "arm.json" if (Path(caches[key]) / ARM_FILE).exists() else "dir",
                "all": _summary(list(u.values())),
                "paired": _summary([u[k] for k in shared]),
            }
            for key, u in units.items()
        },
    }
    events = [k[0] for k in shared]
    base = np.array([units[args.baseline][k]["median_abs_px"] for k in shared])
    report["versus_baseline"] = {
        key: paired_ratio(events, np.array([units[key][k]["median_abs_px"] for k in shared]), base)
        for key in units
        if key != args.baseline
    }
    print(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="recollect the observability cache with one pose arm")
    c.add_argument("--arm", required=True, choices=sorted(arms.POSE_ARMS))
    c.add_argument("--device", required=True, choices=("CPU", "GPU", "NPU"))
    c.add_argument("--cache", required=True, type=Path)
    k = sub.add_parser("compare", help="paired cross-view residual per arm vs a baseline")
    k.add_argument("--cache", action="append", required=True, metavar="KEY=DIR")
    k.add_argument("--baseline", required=True)
    args, probe_flags = parser.parse_known_args(argv)
    if args.command == "collect":
        return collect(args, probe_flags)
    if probe_flags:
        parser.error(f"unrecognized arguments: {' '.join(probe_flags)}")
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
