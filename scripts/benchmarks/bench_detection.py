"""Benchmarks for detection: generate_anchors, nms, decode_detections."""

from __future__ import annotations

from pose_estimation.detection import (
    PALM_INPUT_SIZE,
    POSE_INPUT_SIZE,
    decode_detections,
    generate_anchors,
    nms,
)
from pose_estimation.processing import _smooth_detections

from ._fixtures import (
    PALM_DET_KP,
    POSE_DET_KP,
    make_detections,
    make_nms_boxes,
    make_raw_ssd_outputs,
)
from ._harness import run_group

# MediaPipe SSD stride configurations (real values used by the pipeline)
PALM_STRIDES = [8, 16, 16, 16]
POSE_STRIDES = [8, 16, 32, 32, 32]


def _generate_anchors_case(input_size: int, strides: list[int]):
    return lambda: generate_anchors(input_size, strides)


def _nms_case(n: int, iou: float):
    boxes, scores = make_nms_boxes(n)

    def _run():
        # nms mutates ``order`` internally but we pass fresh arrays; make copies
        # just in case to keep benchmarks isolated.
        nms(boxes.copy(), scores.copy(), iou_threshold=iou)

    return _run


def _decode_case(n_anchors: int, num_keypoints: int):
    raw_boxes, raw_scores, anchors = make_raw_ssd_outputs(n_anchors, num_keypoints)
    input_size = PALM_INPUT_SIZE if num_keypoints == PALM_DET_KP else POSE_INPUT_SIZE

    def _run():
        decode_detections(
            raw_boxes,
            raw_scores,
            anchors,
            input_size,
            num_keypoints,
            score_threshold=0.5,
            iou_threshold=0.3,
        )

    return _run


def _smooth_detections_case(n_new: int, n_prev: int, num_keypoints: int):
    new_dets = make_detections(n_new, num_keypoints, seed=3000)
    prev_dets = make_detections(n_prev, num_keypoints, seed=3100)

    def _run():
        _smooth_detections(list(new_dets), list(prev_dets))

    return _run


def build_cases():
    cases: list[tuple[str, object, dict]] = []

    # Anchor generation (done once at startup, but cheap to measure)
    cases.append(
        (
            "generate_anchors (palm)",
            _generate_anchors_case(PALM_INPUT_SIZE, PALM_STRIDES),
            {"size": PALM_INPUT_SIZE, "layers": len(PALM_STRIDES)},
        )
    )
    cases.append(
        (
            "generate_anchors (pose)",
            _generate_anchors_case(POSE_INPUT_SIZE, POSE_STRIDES),
            {"size": POSE_INPUT_SIZE, "layers": len(POSE_STRIDES)},
        )
    )

    # NMS across detection counts (post-threshold, so O(10)–O(few hundred))
    for n in (4, 16, 64, 256):
        for iou in (0.3, 0.5):
            cases.append(("nms", _nms_case(n, iou), {"n": n, "iou": iou}))

    # decode_detections across typical palm/pose anchor counts
    # Palm SSD with strides [8,16,16,16] on 192 input has 2016 anchors
    for n_anchors in (2016, 896):
        for num_kp in (PALM_DET_KP, POSE_DET_KP):
            cases.append(
                (
                    "decode_detections",
                    _decode_case(n_anchors, num_kp),
                    {"anchors": n_anchors, "kp": num_kp},
                )
            )

    # _smooth_detections (Hungarian on small n, with carry-forward)
    for n_new in (0, 1, 2, 4):
        for n_prev in (0, 1, 2, 4):
            cases.append(
                (
                    "_smooth_detections",
                    _smooth_detections_case(n_new, n_prev, PALM_DET_KP),
                    {"new": n_new, "prev": n_prev},
                )
            )

    return cases


def run(iters: int = 100, warmup: int = 5):
    return run_group("detection", build_cases(), iters=iters, warmup=warmup)
