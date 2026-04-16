"""Benchmarks for processing module — affine helpers, re-crop, synthesis."""

from __future__ import annotations

from pose_estimation.processing import (
    HAND_INPUT_SIZE,
    POSE_LM_INPUT_SIZE,
    _affine_matrix,
    _recrop_from_landmarks,
    _synthesise_hand_detections,
    get_hand_crop,
    get_pose_crop,
    tracking_pose_indices,
    transform_landmarks_to_image,
)

from ._fixtures import (
    ARM_KP,
    BODY_KP,
    FRAME_H,
    FRAME_W,
    HAND_KP,
    PALM_DET_KP,
    POSE_DET_KP,
    make_body_landmarks,
    make_body_list,
    make_detection,
    make_detections,
    make_frame,
    make_hand_landmarks,
    make_hand_list,
    make_visibility_list,
)
from ._harness import run_group


def _tracking_indices_case(mode: str):
    return lambda: tracking_pose_indices(mode)


def _affine_matrix_case():
    return lambda: _affine_matrix(640.0, 360.0, 25.0, 400.0, 256)


def _get_pose_crop_case():
    frame = make_frame()
    det = make_detection(POSE_DET_KP, seed=4000)
    return lambda: get_pose_crop(frame, det)


def _get_hand_crop_case():
    frame = make_frame()
    det = make_detection(PALM_DET_KP, seed=4100)
    return lambda: get_hand_crop(frame, det)


def _transform_landmarks_case(n_kp: int):
    lm = make_body_landmarks(n_kp=n_kp, seed=4200)
    M = _affine_matrix(640.0, 360.0, 10.0, 400.0, POSE_LM_INPUT_SIZE)

    def _run():
        transform_landmarks_to_image(lm, M)

    return _run


def _transform_hand_case():
    lm = make_hand_landmarks(seed=4300)
    M = _affine_matrix(600.0, 400.0, 30.0, 200.0, HAND_INPUT_SIZE)

    def _run():
        transform_landmarks_to_image(lm, M)

    return _run


def _synthesise_hands_case(n_bodies: int, n_palm_dets: int, mode: str):
    n_kp = BODY_KP if mode == "body" else ARM_KP
    bodies = make_body_list(n_bodies=n_bodies, n_kp=n_kp, seed=4400)
    vis = make_visibility_list(n_bodies=n_bodies, n_kp=n_kp, seed=4500)
    existing = make_detections(n_palm_dets, PALM_DET_KP, seed=4600)
    _, _, _, arm_chains = tracking_pose_indices(mode)

    def _run():
        _synthesise_hand_detections(
            bodies, vis, list(existing), FRAME_H, FRAME_W, arm_chains=arm_chains
        )

    return _run


def _recrop_case(n_hands: int, n_palm_dets: int):
    prev = make_hand_list(n_hands=n_hands, seed=4700)
    existing = make_detections(n_palm_dets, PALM_DET_KP, seed=4800)

    def _run():
        _recrop_from_landmarks(prev, list(existing), FRAME_H, FRAME_W)

    return _run


def build_cases():
    cases: list[tuple[str, object, dict]] = []

    for mode in ("hands", "hands-arms", "body"):
        cases.append(
            (
                "tracking_pose_indices",
                _tracking_indices_case(mode),
                {"mode": mode},
            )
        )

    cases.append(("_affine_matrix", _affine_matrix_case(), {}))

    cases.append(("get_pose_crop", _get_pose_crop_case(), {"frame": f"{FRAME_W}x{FRAME_H}"}))
    cases.append(("get_hand_crop", _get_hand_crop_case(), {"frame": f"{FRAME_W}x{FRAME_H}"}))

    for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33"), (HAND_KP, "hand21")]:
        if n_kp == HAND_KP:
            cases.append(
                (
                    "transform_landmarks_to_image",
                    _transform_hand_case(),
                    {"kp": label},
                )
            )
        else:
            cases.append(
                (
                    "transform_landmarks_to_image",
                    _transform_landmarks_case(n_kp),
                    {"kp": label},
                )
            )

    for n_bodies in (1, 2, 4):
        for n_palm in (0, 2, 6):
            for mode in ("hands-arms", "body"):
                cases.append(
                    (
                        "_synthesise_hand_detections",
                        _synthesise_hands_case(n_bodies, n_palm, mode),
                        {"bodies": n_bodies, "palm_dets": n_palm, "mode": mode},
                    )
                )

    for n_hands in (0, 1, 2, 4):
        for n_palm in (0, 2, 4):
            cases.append(
                (
                    "_recrop_from_landmarks",
                    _recrop_case(n_hands, n_palm),
                    {"hands": n_hands, "palm_dets": n_palm},
                )
            )

    return cases


def run(iters: int = 100, warmup: int = 5):
    return run_group("processing", build_cases(), iters=iters, warmup=warmup)
