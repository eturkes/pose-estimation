"""Benchmarks for smoothing.OneEuroFilter and PoseSmoother."""

from __future__ import annotations

from pose_estimation.smoothing import OneEuroFilter, PoseSmoother

from ._fixtures import (
    ARM_KP,
    BODY_KP,
    HAND_KP,
    make_body_landmarks,
    make_body_list,
    make_hand_flags,
    make_hand_list,
    make_visibilities,
    make_visibility_list,
)
from ._harness import run_group


def _one_euro_case(n_kp: int, with_conf: bool, n_frames: int):
    """Build a closure that filters *n_frames* synthetic frames."""
    landmarks = [make_body_landmarks(n_kp=n_kp, seed=100 + i) for i in range(n_frames)]
    conf = make_visibilities(n_kp, seed=900) if with_conf else None

    def _run():
        f = OneEuroFilter(min_cutoff=0.3, beta=0.5)
        t = 0.0
        for lm in landmarks:
            f(lm, t, confidence=conf)
            t += 1 / 30

    return _run


def _pose_smoother_bodies_case(n_bodies: int, n_kp: int, n_frames: int):
    """Simulate a *n_frames* run with constant body count."""
    per_frame = [
        make_body_list(n_bodies=n_bodies, n_kp=n_kp, seed=200 + i) for i in range(n_frames)
    ]
    vis_per_frame = [
        make_visibility_list(n_bodies=n_bodies, n_kp=n_kp, seed=300 + i) for i in range(n_frames)
    ]
    si = (11, 12) if n_kp == BODY_KP else (0, 1)

    def _run():
        sm = PoseSmoother()
        t = 0.0
        for bodies, vis in zip(per_frame, vis_per_frame, strict=False):
            sm.smooth_bodies(bodies, vis, t, shoulder_indices=si)
            t += 1 / 30

    return _run


def _pose_smoother_hands_case(n_hands: int, n_frames: int):
    per_frame = [make_hand_list(n_hands=n_hands, seed=400 + i) for i in range(n_frames)]
    flag_per_frame = [make_hand_flags(n_hands, seed=500 + i) for i in range(n_frames)]

    def _run():
        sm = PoseSmoother()
        t = 0.0
        for hands, flags in zip(per_frame, flag_per_frame, strict=False):
            sm.smooth_hands(hands, t, hand_flags=flags)
            t += 1 / 30

    return _run


def _compute_smooth_delta_case(n_kp: int, n_bodies: int):
    raw = [make_body_landmarks(n_kp=n_kp, seed=600 + i) for i in range(n_bodies)]
    smoothed = [make_body_landmarks(n_kp=n_kp, seed=700 + i) for i in range(n_bodies)]
    return lambda: PoseSmoother.compute_smooth_delta(raw, smoothed)


def _extrapolate_case(n_kp: int):
    sm = PoseSmoother()
    last = make_body_landmarks(n_kp=n_kp, seed=800)
    vel = make_body_landmarks(n_kp=n_kp, seed=801) * 0.01
    return lambda: sm._extrapolate(last, vel, 0.0, 1 / 30, 3)


def build_cases():
    cases: list[tuple[str, object, dict]] = []

    # OneEuroFilter single calls across 120-frame runs
    for n_kp, label in [(ARM_KP, "arm12"), (HAND_KP, "hand21"), (BODY_KP, "body33")]:
        for with_conf in (False, True):
            for n_frames in (1, 30, 120):
                cases.append(
                    (
                        "OneEuroFilter.__call__",
                        _one_euro_case(n_kp, with_conf, n_frames),
                        {"kp": label, "conf": with_conf, "frames": n_frames},
                    )
                )

    # PoseSmoother.smooth_bodies across body counts
    for n_bodies in (1, 2, 4):
        for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
            for n_frames in (30, 120):
                cases.append(
                    (
                        "PoseSmoother.smooth_bodies",
                        _pose_smoother_bodies_case(n_bodies, n_kp, n_frames),
                        {"bodies": n_bodies, "kp": label, "frames": n_frames},
                    )
                )

    # PoseSmoother.smooth_hands
    for n_hands in (1, 2, 4, 8):
        for n_frames in (30, 120):
            cases.append(
                (
                    "PoseSmoother.smooth_hands",
                    _pose_smoother_hands_case(n_hands, n_frames),
                    {"hands": n_hands, "frames": n_frames},
                )
            )

    # compute_smooth_delta
    for n_bodies in (1, 2, 4):
        for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
            cases.append(
                (
                    "PoseSmoother.compute_smooth_delta",
                    _compute_smooth_delta_case(n_kp, n_bodies),
                    {"bodies": n_bodies, "kp": label},
                )
            )

    # _extrapolate
    for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
        cases.append(
            (
                "PoseSmoother._extrapolate",
                _extrapolate_case(n_kp),
                {"kp": label},
            )
        )

    return cases


def run(iters: int = 80, warmup: int = 5):
    return run_group("smoothing", build_cases(), iters=iters, warmup=warmup)
