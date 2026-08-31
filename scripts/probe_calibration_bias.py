#!/usr/bin/env python3
"""Decide whether M2.6's closure null is an estimator defect or cross-view correspondence bias.

`probe_calibration_observability.py` measures that three-camera rotation cycles do not close on this
corpus. That result alone cannot say why. This probe answers it with four arms, all reading the
keypoint caches that probe writes, and all reusing its estimator verbatim so the instrument under
test is the shipped one.

  control    Synthetic positive control. Known extrinsics, the REAL cache's per-(camera, frame,
             keypoint) validity masks, image sizes and device models, and swept correspondence error.
             Calibrates the instrument and prices the error budget: what closure does the shipped
             estimator return at a known error level?
  structure  Is the epipolar residual zero-mean noise or a reproducible per-keypoint offset? One
             pooled pose is fit on a training frame block; per-keypoint mean signed residuals are
             then correlated between two DISJOINT held-out blocks. Synthetic noise and bias arms run
             the identical statistic, so the corpus number reads against calibrated references.
  ba         Independent pairwise bundle adjustment, robust Sampson, cycle recomposed. Contract A02:
             poses stay independently estimated, so closure remains an acceptance statistic rather
             than an algebraic identity of a joint solve.
  subset     Does restricting to low-bias keypoints rescue closure? Keypoints are ranked on one fold
             of events and closure evaluated on the disjoint fold, so the subset is never selected on
             the outcome it is scored against.

Output is redaction-safe: distributions and counts only, no identifier, filename or per-event key.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_calibration_observability import (
    CALIBRATION_INDICES,
    FX_PRIOR_1920,
    PRIMARY_CONFIDENCE,
    RANSAC_THRESHOLD_PX,
    SEED,
    _estimate_pose,
    _normalized,
    _rotation_angle,
    _valid,
)

KP = np.asarray(CALIBRATION_INDICES, dtype=int)
N_BODY = 23  # CALIBRATION_INDICES = range(23) + range(91,133): 23 body + 2 x 21 hand
N_HAND = 21
MIN_OBS_PER_HALF = 3
MIN_KEYPOINTS = 8
CLOSURE_BOUND_DEG = 10.0
MOTION_EXTENT_M = 0.40  # seated upper-limb working volume; Lee et al.'s small-motion regime
SUBSET_SIZES = (16, 24, 32, 40)
NOISE_SWEEP_PX = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
IMAGE_BIAS_SWEEP_PX = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
ANATOMICAL_BIAS_SWEEP_MM = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)


# --- synthetic rig ------------------------------------------------------------------------------


def _rig(rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """World->camera (R, t) for above/left/right at corpus-like separations, jittered per event."""
    nominal = np.array([[0.0, 1.60, 1.20], [-1.60, 0.90, 1.10], [1.60, 0.90, 1.10]])
    poses = []
    for position in nominal + rng.uniform(-0.25, 0.25, size=(3, 3)):
        forward = -position / np.linalg.norm(position)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(forward, right), forward])
        poses.append((rotation, -rotation @ position))
    return poses


def _structure(frames: int, rng: np.random.Generator) -> np.ndarray:
    """(frames, 65, 3) world points: seated upper body plus two moving hand clusters."""
    body = np.column_stack(
        [
            rng.uniform(-0.22, 0.22, N_BODY),
            rng.uniform(-0.25, 0.42, N_BODY),
            rng.uniform(-0.14, 0.14, N_BODY),
        ]
    )
    hands = [rng.normal(0.0, 0.035, size=(N_HAND, 3)) for _ in range(2)]
    half = MOTION_EXTENT_M / 2.0
    out = np.empty((frames, N_BODY + 2 * N_HAND, 3))
    for frame in range(frames):
        out[frame, :N_BODY] = body + rng.normal(0.0, 0.02, size=(N_BODY, 3))
        for hand in range(2):
            centre = np.array([(-1) ** hand * 0.20, 0.05, 0.28]) + rng.uniform(-half, half, size=3)
            start = N_BODY + hand * N_HAND
            out[frame, start : start + N_HAND] = hands[hand] + centre
    return out


def _fx(model: str, width: int, height: int) -> float:
    return FX_PRIOR_1920[model] * max(width, height) / 1920.0


def synthesize(
    mask: np.ndarray,
    sizes: np.ndarray,
    models: list[str],
    *,
    sigma_px: float = 0.0,
    image_bias_px: float = 0.0,
    anatomical_bias_mm: float = 0.0,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Normalized observations under a known rig.

    `anatomical_bias_mm` is the hypothesised mechanism: a per-(camera, keypoint) offset fixed in the
    BODY frame, so the same joint means a different physical point to each camera and rides that
    disagreement through every frame. `image_bias_px` is its image-fixed counterpart, and `sigma_px`
    is zero-mean measurement noise.
    """
    rng = np.random.default_rng(seed)
    cameras, frames, keypoints = mask.shape
    poses = _rig(rng)
    world = _structure(frames, rng)
    focals = np.array([_fx(models[c], int(sizes[c, 0]), int(sizes[c, 1])) for c in range(cameras)])
    flat = rng.normal(0.0, image_bias_px, (cameras, keypoints, 2)) if image_bias_px > 0 else None
    solid = (
        rng.normal(0.0, anatomical_bias_mm / 1000.0, (cameras, keypoints, 3))
        if anatomical_bias_mm > 0
        else None
    )
    observed = np.empty((cameras, frames, keypoints, 2))
    for camera in range(cameras):
        width, height = int(sizes[camera, 0]), int(sizes[camera, 1])
        for frame in range(frames):
            points = world[frame] if solid is None else world[frame] + solid[camera]
            in_camera = points @ poses[camera][0].T + poses[camera][1]
            depth = np.maximum(in_camera[:, 2], 1e-6)
            pixels = np.column_stack(
                [
                    in_camera[:, 0] / depth * focals[camera] + width / 2.0,
                    in_camera[:, 1] / depth * focals[camera] + height / 2.0,
                ]
            )
            if flat is not None:
                pixels = pixels + flat[camera]
            if flat is not None or solid is not None:
                pixels = pixels + rng.normal(0.0, 1.0, pixels.shape)
            if sigma_px > 0:
                pixels = pixels + rng.normal(0.0, sigma_px, pixels.shape)
            observed[camera, frame] = _normalized(pixels, width, height, models[camera])
    return observed, focals, poses


# --- cache reading ------------------------------------------------------------------------------


def load_event(path: Path) -> dict[str, Any] | None:
    """Real observations, validity mask, focals and sizes from one observability cache entry."""
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["meta"].tobytes().decode())
        points, scores = archive["points"], archive["scores"]
        widths, heights = archive["widths"], archive["heights"]
    if points.ndim != 4:
        return None
    cameras, frames = points.shape[0], points.shape[1]
    sizes = np.stack([[widths[c, 0], heights[c, 0]] for c in range(cameras)])
    if sizes.min() <= 0:
        return None
    models = meta["models"]
    mask = np.stack(
        [
            np.stack(
                [
                    _valid(points[c, f, KP], scores[c, f, KP], PRIMARY_CONFIDENCE)
                    for f in range(frames)
                ]
            )
            for c in range(cameras)
        ]
    )
    observed = np.empty((cameras, frames, len(KP), 2))
    for camera in range(cameras):
        width, height = int(sizes[camera, 0]), int(sizes[camera, 1])
        for frame in range(frames):
            observed[camera, frame] = _normalized(
                points[camera, frame, KP], width, height, models[camera]
            )
    focals = np.array([_fx(models[c], int(sizes[c, 0]), int(sizes[c, 1])) for c in range(cameras)])
    return {
        "observed": observed,
        "mask": mask,
        "focals": focals,
        "sizes": sizes,
        "models": models,
        "cameras": cameras,
    }


def load_events(cache: Path, *, cameras: int | None = None) -> list[dict[str, Any]]:
    events = []
    for path in sorted(cache.glob("*.npz")):
        event = load_event(path)
        if event is not None and (cameras is None or event["cameras"] == cameras):
            events.append(event)
    return events


# --- shared geometry ----------------------------------------------------------------------------


def _pairs_of(cameras: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(cameras) for b in range(a + 1, cameras)]


def _pool(observed, mask, left, right, frames):
    lefts, rights, labels = [], [], []
    for frame in frames:
        shared = mask[left, frame] & mask[right, frame]
        if not shared.any():
            continue
        lefts.append(observed[left, frame, shared])
        rights.append(observed[right, frame, shared])
        labels.append(np.flatnonzero(shared))
    if not lefts:
        return None
    return np.concatenate(lefts), np.concatenate(rights), np.concatenate(labels)


def _pose(event, left, right, frames):
    pooled = _pool(event["observed"], event["mask"], left, right, frames)
    if pooled is None or len(pooled[0]) < 8:
        return None, None, None
    focal = float((event["focals"][left] + event["focals"][right]) / 2.0)
    estimate = _estimate_pose(pooled[0], pooled[1], RANSAC_THRESHOLD_PX / focal)
    return estimate, pooled, focal


def _cycle(rotations: dict[tuple[int, int], np.ndarray]) -> float:
    if len(rotations) != 3:
        return math.nan
    return _rotation_angle(rotations[(0, 2)], rotations[(1, 2)] @ rotations[(0, 1)])


def _signed_residual_px(essential, left, right, focal):
    """Signed point-to-epipolar-line distance in the right image, in pixels."""
    homogeneous_left = np.column_stack([left, np.ones(len(left))])
    homogeneous_right = np.column_stack([right, np.ones(len(right))])
    lines = (essential @ homogeneous_left.T).T
    norm = np.maximum(np.hypot(lines[:, 0], lines[:, 1]), np.finfo(np.float64).eps)
    return np.sum(homogeneous_right * lines, axis=1) / norm * focal


def _stats(values, digits: int = 3) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"n": 0}
    return {
        "n": len(finite),
        "median": round(float(np.median(finite)), digits),
        "min": round(finite[0], digits),
        "max": round(finite[-1], digits),
    }


def _closure_stats(values) -> dict[str, Any]:
    out = _stats(values)
    out["within_10deg"] = sum(
        value <= CLOSURE_BOUND_DEG for value in values if math.isfinite(value)
    )
    return out


# --- arm: control -------------------------------------------------------------------------------


def _synthetic_closure(events, *, seed_base: int, **kwargs) -> dict[str, Any]:
    cycles, errors = [], []
    for index, event in enumerate(events):
        observed, focals, poses = synthesize(
            event["mask"], event["sizes"], event["models"], seed=seed_base + index, **kwargs
        )
        probe = {"observed": observed, "mask": event["mask"], "focals": focals}
        frames = list(range(event["mask"].shape[1]))
        rotations = {}
        for left, right in _pairs_of(3):
            estimate, _, _ = _pose(probe, left, right, frames)
            if estimate is not None:
                rotations[(left, right)] = estimate.rotation
                errors.append(
                    _rotation_angle(estimate.rotation, poses[right][0] @ poses[left][0].T)
                )
        cycles.append(_cycle(rotations))
    return {"cycle_deg": _closure_stats(cycles), "pair_rotation_error_deg": _closure_stats(errors)}


def arm_control(caches: dict[str, Path]) -> dict[str, Any]:
    conditions = [("noise", {"sigma_px": value}) for value in NOISE_SWEEP_PX]
    conditions += [("image_bias", {"image_bias_px": value}) for value in IMAGE_BIAS_SWEEP_PX]
    conditions += [
        ("anatomical_bias", {"anatomical_bias_mm": value}) for value in ANATOMICAL_BIAS_SWEEP_MM
    ]
    rows = []
    for name, cache in caches.items():
        events = load_events(cache, cameras=3)
        if not events:
            continue
        for arm, kwargs in conditions:
            row = {
                "cache": name,
                "frames": int(events[0]["mask"].shape[1]),
                "events": len(events),
                "arm": arm,
                **dict(kwargs),
                **_synthetic_closure(events, seed_base=SEED, **kwargs),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {"conditions": rows}


# --- arm: structure -----------------------------------------------------------------------------


def _keypoint_means(residual, labels, keypoints):
    means = np.full(keypoints, np.nan)
    counts = np.zeros(keypoints, dtype=int)
    for keypoint in range(keypoints):
        selected = labels == keypoint
        counts[keypoint] = int(selected.sum())
        if counts[keypoint] >= MIN_OBS_PER_HALF:
            means[keypoint] = float(residual[selected].mean())
    return means, counts


def _splits(frames: int):
    half = frames // 2
    quarter = (frames - half) // 2
    return (
        list(range(half)),
        list(range(half, half + quarter)),
        list(range(half + quarter, frames)),
    )


def pair_structure(event, left, right) -> dict[str, Any] | None:
    train, test_a, test_b = _splits(event["mask"].shape[1])
    estimate, _, focal = _pose(event, left, right, train)
    if estimate is None:
        return None
    halves = []
    for frames in (test_a, test_b):
        block = _pool(event["observed"], event["mask"], left, right, frames)
        if block is None:
            return None
        halves.append(
            (_signed_residual_px(estimate.essential, block[0], block[1], focal), block[2])
        )
    keypoints = event["mask"].shape[2]
    means_a, _ = _keypoint_means(*halves[0], keypoints)
    means_b, _ = _keypoint_means(*halves[1], keypoints)
    usable = np.isfinite(means_a) & np.isfinite(means_b)
    if int(usable.sum()) < MIN_KEYPOINTS:
        return None
    a, b = means_a[usable], means_b[usable]
    if a.std() <= 0 or b.std() <= 0:
        return None
    combined = np.concatenate([halves[0][0], halves[1][0]])
    pooled_means, pooled_counts = _keypoint_means(
        combined, np.concatenate([halves[0][1], halves[1][1]]), keypoints
    )
    covered = np.isfinite(pooled_means)
    return {
        "split_r": float(np.corrcoef(a, b)[0, 1]),
        "median_abs_px": float(np.median(np.abs(combined))),
        "keypoints": int(usable.sum()),
        "between_keypoint_variance_fraction": (
            float(np.var(pooled_means[covered]) / np.var(combined))
            if np.var(combined) > 0
            else math.nan
        ),
        "median_obs_per_keypoint": float(np.median(pooled_counts[covered])),
    }


def _structure_summary(rows, label) -> dict[str, Any]:
    if not rows:
        return {"label": label, "pairs": 0}
    return {
        "label": label,
        "pairs": len(rows),
        "split_r": _stats([row["split_r"] for row in rows], 4),
        "median_abs_px": _stats([row["median_abs_px"] for row in rows]),
        "between_keypoint_variance_fraction": _stats(
            [row["between_keypoint_variance_fraction"] for row in rows], 4
        ),
        "split_r_above_0p5": sum(row["split_r"] > 0.5 for row in rows),
    }


def arm_structure(cache: Path) -> dict[str, Any]:
    results = []
    real = load_events(cache)
    by_count: dict[int, list] = {}
    for event in real:
        for left, right in _pairs_of(event["cameras"]):
            row = pair_structure(event, left, right)
            if row is not None:
                by_count.setdefault(event["cameras"], []).append(row)
    results.append(_structure_summary([row for rows in by_count.values() for row in rows], "REAL"))
    print(json.dumps(results[-1]), flush=True)
    for cameras, rows in sorted(by_count.items()):
        results.append(_structure_summary(rows, f"REAL {cameras}-camera events"))
        print(json.dumps(results[-1]), flush=True)

    events = load_events(cache, cameras=3)
    arms: list[tuple[str, dict[str, float]]] = [
        (f"synth noise sigma={value}px", {"sigma_px": value}) for value in (2.0, 8.0, 32.0)
    ]
    arms += [(f"synth image-bias {value}px", {"image_bias_px": value}) for value in (8.0, 32.0)]
    arms += [
        (f"synth anatomical-bias {value}mm", {"anatomical_bias_mm": value})
        for value in (20.0, 40.0, 80.0)
    ]
    for label, kwargs in arms:
        rows = []
        for index, event in enumerate(events):
            observed, focals, _ = synthesize(
                event["mask"], event["sizes"], event["models"], seed=SEED + index, **kwargs
            )
            probe = {"observed": observed, "mask": event["mask"], "focals": focals}
            for left, right in _pairs_of(3):
                row = pair_structure(probe, left, right)
                if row is not None:
                    rows.append(row)
        results.append(_structure_summary(rows, label))
        print(json.dumps(results[-1]), flush=True)
    return {"results": results}


# --- arm: independent pairwise bundle adjustment --------------------------------------------------


def _skew(vector):
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def _rodrigues(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    cross = _skew(vector / angle)
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _log_rotation(rotation):
    angle = math.acos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def _sampson(essential, left, right):
    homogeneous_left = np.column_stack([left, np.ones(len(left))])
    homogeneous_right = np.column_stack([right, np.ones(len(right))])
    lines_right = (essential @ homogeneous_left.T).T
    lines_left = (essential.T @ homogeneous_right.T).T
    denominator = np.sqrt(
        np.maximum(
            lines_right[:, 0] ** 2
            + lines_right[:, 1] ** 2
            + lines_left[:, 0] ** 2
            + lines_left[:, 1] ** 2,
            np.finfo(np.float64).eps,
        )
    )
    return np.sum(homogeneous_right * lines_right, axis=1) / denominator


def refine_pair(estimate, left, right, threshold):
    """Robust Sampson minimization over the 5 relative-pose DoF, independent of every other pair."""
    unit = estimate.translation / np.linalg.norm(estimate.translation)
    start = np.concatenate(
        [
            _log_rotation(estimate.rotation),
            [math.acos(np.clip(unit[2], -1.0, 1.0)), math.atan2(unit[1], unit[0])],
        ]
    )

    def residuals(parameters):
        theta, phi = parameters[3:]
        direction = np.array(
            [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
        )
        return _sampson(_skew(direction) @ _rodrigues(parameters[:3]), left, right)

    try:
        solution = least_squares(
            residuals, start, loss="soft_l1", f_scale=threshold, max_nfev=400, xtol=1e-12
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    return _rodrigues(solution.x[:3])


def arm_ba(caches: dict[str, Path]) -> dict[str, Any]:
    out = {}
    for name, cache in caches.items():
        baseline, refined, moves = [], [], []
        for event in load_events(cache, cameras=3):
            frames = list(range(event["mask"].shape[1]))
            initial, adjusted = {}, {}
            for left, right in _pairs_of(3):
                estimate, pooled, focal = _pose(event, left, right, frames)
                if estimate is None:
                    continue
                initial[(left, right)] = estimate.rotation
                threshold = RANSAC_THRESHOLD_PX / focal
                improved = refine_pair(estimate, pooled[0], pooled[1], threshold)
                if improved is not None:
                    adjusted[(left, right)] = improved
            if len(initial) == 3 and len(adjusted) == 3:
                baseline.append(_cycle(initial))
                refined.append(_cycle(adjusted))
                moves.append(max(_rotation_angle(initial[key], adjusted[key]) for key in initial))
        out[name] = {
            "events": len(baseline),
            "cycle_recoverPose_deg": _closure_stats(baseline),
            "cycle_after_independent_BA_deg": _closure_stats(refined),
            "max_pose_move_deg": _stats(moves),
        }
        print(json.dumps({name: out[name]}), flush=True)
    return out


# --- arm: low-bias keypoint subset ----------------------------------------------------------------


def _keypoint_bias(events) -> np.ndarray:
    """Per-keypoint mean absolute epipolar residual px, pooled over every pair of every event."""
    keypoints = events[0]["mask"].shape[2]
    totals, counts = np.zeros(keypoints), np.zeros(keypoints, dtype=int)
    for event in events:
        frames = list(range(event["mask"].shape[1]))
        for left, right in _pairs_of(3):
            estimate, pooled, focal = _pose(event, left, right, frames)
            if estimate is None:
                continue
            residual = _signed_residual_px(estimate.essential, pooled[0], pooled[1], focal)
            for keypoint in range(keypoints):
                selected = pooled[2] == keypoint
                if selected.any():
                    totals[keypoint] += float(np.abs(residual[selected]).mean())
                    counts[keypoint] += 1
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.inf)


def _closure_on_subset(events, subset) -> list[float]:
    values = []
    for event in events:
        blocked = np.ones(event["mask"].shape[2], dtype=bool)
        blocked[subset] = False
        restricted = event["mask"].copy()
        restricted[:, :, blocked] = False
        probe = {"observed": event["observed"], "mask": restricted, "focals": event["focals"]}
        frames = list(range(restricted.shape[1]))
        rotations = {}
        for left, right in _pairs_of(3):
            estimate, _, _ = _pose(probe, left, right, frames)
            if estimate is not None:
                rotations[(left, right)] = estimate.rotation
        cycle = _cycle(rotations)
        if math.isfinite(cycle):
            values.append(cycle)
    return values


def arm_subset(cache: Path) -> dict[str, Any]:
    events = load_events(cache, cameras=3)
    keypoints = events[0]["mask"].shape[2]
    even = list(range(0, len(events), 2))
    odd = list(range(1, len(events), 2))
    folds = []
    for select, evaluate in ((even, odd), (odd, even)):
        bias = _keypoint_bias([events[i] for i in select])
        order = np.argsort(bias)
        held_out = [events[i] for i in evaluate]
        fold = {
            "select_events": len(select),
            "evaluate_events": len(evaluate),
            "bias_px_cleanest10_median": round(float(np.median(bias[order[:10]])), 3),
            "all_keypoints": _closure_stats(_closure_on_subset(held_out, np.arange(keypoints))),
            "subsets": {},
        }
        for size in SUBSET_SIZES:
            fold["subsets"][f"cleanest{size}"] = _closure_stats(
                _closure_on_subset(held_out, order[:size])
            )
            fold["subsets"][f"noisiest{size}"] = _closure_stats(
                _closure_on_subset(held_out, order[-size:])
            )
        folds.append(fold)
        print(json.dumps(fold), flush=True)
    return {"events": len(events), "keypoints": keypoints, "folds": folds}


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / ".scratch" / "calib-obs-f32")
    parser.add_argument(
        "--sparse-cache", type=Path, default=root / ".scratch" / "calibration-observability"
    )
    parser.add_argument(
        "arm", choices=("control", "structure", "ba", "subset", "all"), default="all", nargs="?"
    )
    args = parser.parse_args(argv)
    caches = {
        name: path
        for name, path in (("sparse", args.sparse_cache), ("dense", args.cache))
        if path.is_dir()
    }
    if not caches:
        raise SystemExit(
            "no keypoint cache found; run probe_calibration_observability.py collect first"
        )
    report: dict[str, Any] = {"seed": SEED, "caches": sorted(caches)}
    if args.arm in ("control", "all"):
        report["control"] = arm_control(caches)
    if args.arm in ("structure", "all"):
        report["structure"] = arm_structure(args.cache)
    if args.arm in ("ba", "all"):
        report["ba"] = arm_ba(caches)
    if args.arm in ("subset", "all"):
        report["subset"] = arm_subset(args.cache)
    print(json.dumps({"summary": sorted(report)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
