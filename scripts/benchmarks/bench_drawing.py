"""Benchmarks for drawing module — spline, body/hand overlays."""

from __future__ import annotations

import numpy as np

from pose_estimation.drawing import (
    catmull_rom_spline,
    draw_arm_hand_bridges,
    draw_body_landmarks,
    draw_hand_landmarks,
)

from ._fixtures import (
    ARM_KP,
    BODY_KP,
    FRAME_H,
    FRAME_W,
    make_body_list,
    make_frame,
    make_hand_list,
    make_visibility_list,
)
from ._harness import run_group


def _spline_case(n_points: int, samples: int):
    rng = np.random.default_rng(5000 + n_points)
    pts = rng.uniform(100, 600, (n_points, 2))

    def _run():
        catmull_rom_spline(pts, num_samples=samples)

    return _run


def _draw_body_case(n_bodies: int, n_kp: int):
    bodies = make_body_list(n_bodies=n_bodies, n_kp=n_kp, seed=5100)
    vis = make_visibility_list(n_bodies=n_bodies, n_kp=n_kp, seed=5200)
    frame = make_frame()

    def _run():
        draw_body_landmarks(frame.copy(), bodies, vis)

    return _run


def _draw_hand_case(n_hands: int):
    hands = make_hand_list(n_hands=n_hands, seed=5300)
    frame = make_frame()

    def _run():
        draw_hand_landmarks(frame.copy(), hands)

    return _run


def _draw_bridges_case(n_pairs: int):
    # One body per pair (worst case for cache coherence)
    n_bodies = max(1, n_pairs)
    n_hands = max(1, n_pairs)
    bodies = make_body_list(n_bodies=n_bodies, n_kp=ARM_KP, seed=5400)
    hands = make_hand_list(n_hands=n_hands, seed=5500)
    matches = [(i % n_bodies, 4 + (i % 2), i % n_hands) for i in range(n_pairs)]
    frame = make_frame()

    def _run():
        draw_arm_hand_bridges(frame.copy(), bodies, hands, matches)

    return _run


def build_cases():
    cases: list[tuple[str, object, dict]] = []

    for n in (2, 3, 5, 8):
        for samples in (10, 20, 40):
            cases.append(
                (
                    "catmull_rom_spline",
                    _spline_case(n, samples),
                    {"points": n, "samples": samples},
                )
            )

    for n_bodies in (1, 2, 4):
        for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
            cases.append(
                (
                    "draw_body_landmarks",
                    _draw_body_case(n_bodies, n_kp),
                    {"bodies": n_bodies, "kp": label, "frame": f"{FRAME_W}x{FRAME_H}"},
                )
            )

    for n_hands in (1, 2, 4, 8):
        cases.append(
            (
                "draw_hand_landmarks",
                _draw_hand_case(n_hands),
                {"hands": n_hands, "frame": f"{FRAME_W}x{FRAME_H}"},
            )
        )

    for n_pairs in (1, 2, 4, 8):
        cases.append(
            (
                "draw_arm_hand_bridges",
                _draw_bridges_case(n_pairs),
                {"pairs": n_pairs},
            )
        )

    return cases


def run(iters: int = 60, warmup: int = 5):
    return run_group("drawing", build_cases(), iters=iters, warmup=warmup)
