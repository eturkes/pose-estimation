"""Benchmarks for processing.match_hands_to_arms and select_primary_body."""

from __future__ import annotations

import numpy as np

from pose_estimation.processing import (
    SHOULDER_KPS_12,
    SHOULDER_KPS_33,
    WRIST_KPS_12,
    WRIST_KPS_33,
    match_hands_to_arms,
    select_primary_body,
)

from ._fixtures import ARM_KP, BODY_KP, make_body_list, make_hand_list, make_visibility_list
from ._harness import run_group


def _match_case(n_bodies: int, n_hands: int, n_kp: int, close: bool):
    """Build closure that calls match_hands_to_arms repeatedly.

    *close* toggles whether we place hand wrists near real wrists so the
    threshold check passes (true hot path) or far from them (all pairs
    rejected — tests the cost-matrix path).
    """
    bodies = make_body_list(n_bodies=n_bodies, n_kp=n_kp, seed=2000)
    hands = make_hand_list(n_hands=n_hands, seed=2100)

    if close and bodies:
        # Anchor each hand[0] close to an arm wrist for realistic matching
        wrist_kps = WRIST_KPS_33 if n_kp == BODY_KP else WRIST_KPS_12
        rng = np.random.default_rng(2200)
        for hi, h in enumerate(hands):
            if hi >= 2 * n_bodies:
                # No arm wrist left — push far away
                h[0, :2] = np.array([-500.0, -500.0])
                continue
            body_idx = hi // 2
            w_kp = wrist_kps[hi % 2]
            h[0, :2] = bodies[body_idx][w_kp, :2] + rng.normal(0, 5, 2)

    wrist_kps = WRIST_KPS_33 if n_kp == BODY_KP else WRIST_KPS_12
    shoulder_kps = SHOULDER_KPS_33 if n_kp == BODY_KP else SHOULDER_KPS_12

    def _run():
        match_hands_to_arms(
            bodies,
            hands,
            threshold=100,
            wrist_kps=wrist_kps,
            shoulder_kps=shoulder_kps,
        )

    return _run


def _select_primary_case(n_bodies: int, n_hands: int, n_kp: int):
    bodies = make_body_list(n_bodies=n_bodies, n_kp=n_kp, seed=2300)
    vis = make_visibility_list(n_bodies=n_bodies, n_kp=n_kp, seed=2400)
    hands = make_hand_list(n_hands=n_hands, seed=2500)
    # Build a matches list touching each body
    matches: list[tuple[int, int, int]] = []
    wrist_kps = WRIST_KPS_33 if n_kp == BODY_KP else WRIST_KPS_12
    for bi in range(n_bodies):
        for hi_offset, wk in enumerate(wrist_kps):
            hand_idx = (2 * bi + hi_offset) % max(1, n_hands)
            matches.append((bi, wk, hand_idx))

    def _run():
        select_primary_body(bodies, vis, hands, matches)

    return _run


def build_cases():
    cases: list[tuple[str, object, dict]] = []

    # Realistic (close) matching: 1-4 bodies × 0-8 hands × both kp schemes
    for n_bodies in (1, 2, 4):
        for n_hands in (0, 2, 4, 8):
            for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
                cases.append(
                    (
                        "match_hands_to_arms (close)",
                        _match_case(n_bodies, n_hands, n_kp, close=True),
                        {"bodies": n_bodies, "hands": n_hands, "kp": label},
                    )
                )

    # Stress: hands far from all wrists (every pair rejected at threshold)
    for n_bodies in (1, 4):
        for n_hands in (2, 8):
            cases.append(
                (
                    "match_hands_to_arms (far)",
                    _match_case(n_bodies, n_hands, ARM_KP, close=False),
                    {"bodies": n_bodies, "hands": n_hands, "kp": "arm12"},
                )
            )

    # Primary-body selection
    for n_bodies in (1, 2, 4):
        for n_hands in (2, 4):
            for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
                cases.append(
                    (
                        "select_primary_body",
                        _select_primary_case(n_bodies, n_hands, n_kp),
                        {"bodies": n_bodies, "hands": n_hands, "kp": label},
                    )
                )

    return cases


def run(iters: int = 150, warmup: int = 10):
    return run_group("matching", build_cases(), iters=iters, warmup=warmup)
