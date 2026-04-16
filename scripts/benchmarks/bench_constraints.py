"""Benchmarks for constraints.BoneLengthSmoother and clamp_joint_angles."""

from __future__ import annotations

import numpy as np

from pose_estimation.constraints import (
    ANGLE_LIMITS,
    ANGLE_LIMITS_BODY,
    BONE_SEGMENTS,
    BONE_SEGMENTS_BODY,
    BoneLengthSmoother,
    clamp_joint_angles,
)

from ._fixtures import ARM_KP, BODY_KP, make_body_landmarks
from ._harness import run_group


def _bone_smoother_stream_case(n_kp: int, n_bodies: int, n_frames: int, tolerance: float):
    """Simulate *n_frames* per-body updates."""
    segments = BONE_SEGMENTS_BODY if n_kp == BODY_KP else BONE_SEGMENTS
    frames = [
        [make_body_landmarks(n_kp=n_kp, seed=1000 + i * 10 + b) for b in range(n_bodies)]
        for i in range(n_frames)
    ]

    def _run():
        sm = BoneLengthSmoother(tolerance=tolerance, segments=segments)
        for body_group in frames:
            for bid, lm in enumerate(body_group):
                sm.update(bid, lm.copy())

    return _run


def _bone_single_update_case(n_kp: int):
    """Measure a single BoneLengthSmoother.update on a warm state."""
    segments = BONE_SEGMENTS_BODY if n_kp == BODY_KP else BONE_SEGMENTS
    sm = BoneLengthSmoother(segments=segments)
    lm = make_body_landmarks(n_kp=n_kp, seed=1500)
    # Warm the EMA state with a couple of updates
    for _ in range(3):
        sm.update(0, lm.copy())

    def _run():
        sm.update(0, lm.copy())

    return _run


def _clamp_angles_case(n_kp: int, shuffle_pct: float):
    """Landmarks where a fraction of angles are outside anatomical limits."""
    limits = ANGLE_LIMITS_BODY if n_kp == BODY_KP else ANGLE_LIMITS
    rng = np.random.default_rng(1600)
    lm = make_body_landmarks(n_kp=n_kp, seed=1700)

    # Push some distal keypoints out-of-range
    triplets = list(limits.keys())
    n_bad = max(1, int(len(triplets) * shuffle_pct))
    bad_triplets = rng.choice(len(triplets), n_bad, replace=False)
    for idx in bad_triplets:
        _, joint, dist = triplets[idx]
        # Place distal near the joint so the angle is far below min_deg
        lm[dist, :2] = lm[joint, :2] + rng.uniform(-2, 2, 2)

    def _run():
        # Work on a copy so each call sees the same starting state
        clamp_joint_angles(lm.copy(), limits=limits)

    return _run


def _prune_case(n_bodies_alive: int, n_bodies_dead: int):
    """BoneLengthSmoother.prune with a mix of alive and stale IDs."""
    sm = BoneLengthSmoother()
    lm = make_body_landmarks(n_kp=ARM_KP, seed=1800)
    # Seed state for all IDs (alive + dead)
    total = n_bodies_alive + n_bodies_dead
    for bid in range(total):
        sm.update(bid, lm.copy())
    active = list(range(n_bodies_alive))

    def _run():
        # prune only mutates state it finds stale, so repeated calls are
        # idempotent after the first; re-seed each iteration to keep work constant
        for bid in range(n_bodies_alive, total):
            sm._averages[bid] = sm._averages.get(bid, lm[:, 0].copy())
        sm.prune(active)

    return _run


def build_cases():
    cases: list[tuple[str, object, dict]] = []

    # Streaming update
    for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
        for n_bodies in (1, 2):
            for n_frames in (30, 120):
                for tol in (0.4, 0.1):
                    cases.append(
                        (
                            "BoneLengthSmoother.update (stream)",
                            _bone_smoother_stream_case(n_kp, n_bodies, n_frames, tol),
                            {"kp": label, "bodies": n_bodies, "frames": n_frames, "tol": tol},
                        )
                    )

    # Single warm update
    for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
        cases.append(
            (
                "BoneLengthSmoother.update (single, warm)",
                _bone_single_update_case(n_kp),
                {"kp": label},
            )
        )

    # Joint-angle clamping at varied violation rates
    for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
        for pct in (0.0, 0.5, 1.0):
            cases.append(
                (
                    "clamp_joint_angles",
                    _clamp_angles_case(n_kp, pct),
                    {"kp": label, "violated": pct},
                )
            )

    # Prune
    for alive, dead in [(1, 0), (1, 3), (2, 6), (4, 12)]:
        cases.append(
            (
                "BoneLengthSmoother.prune",
                _prune_case(alive, dead),
                {"alive": alive, "dead": dead},
            )
        )

    return cases


def run(iters: int = 80, warmup: int = 5):
    return run_group("constraints", build_cases(), iters=iters, warmup=warmup)
