"""Candidate pose models for the benchmark, by exact checkpoint (contracts M4.1-M4.2).

Names follow mmpose's own table, never the checkpoint's distillation prefix: `rtmw-dw-l-m` = m,
`rtmw-dw-x-l` = L. SimCC scores = rtmlib's raw decode (mean of the x and y maximum logits); heatmap
scores = peak value. Every checkpoint has its own scale, so nothing compares raw scores across arms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseArm:
    name: str
    url: str
    input_size: tuple[int, int]  # (width, height)
    note: str
    kind: str = "simcc"
    flip_test: bool = True


#: Screened arms = `run.MODEL_REGISTRY` keys, so a benchmark name, its runtime configuration and
#: its calibration file (`score_calibrations/<key>.json`) can never name one checkpoint twice.
_NOTES = {
    "rtmw-m-256": "shipped through M2.10 (registry alias rtmw-l)",
    "rtmw-l-256": "",
    "rtmw-l-384": "export emits large raw logits (COCO median 4.2)",
    "rtmw-x-384": "cocktail13 ONNX; mmpose's table scores the cocktail14 .pth",
    "dwpose-l-384": "",
}


def _size(value) -> tuple[int, int]:
    width, height = value
    return int(width), int(height)


def _arms() -> dict[str, PoseArm]:
    from pose_estimation.run import MODEL_REGISTRY  # deferred: run imports the inference stack

    return {
        key: PoseArm(
            key,
            str(MODEL_REGISTRY[key]["pose"]),
            _size(MODEL_REGISTRY[key]["pose_input_size"]),
            note,
        )
        for key, note in _NOTES.items()
    }


POSE_ARMS = _arms()

#: Screening-only heatmap arms (M4.2 D01): f16 IR on GPU, no runtime pipeline path → kept apart
#: from `POSE_ARMS`; a carried-forward arm enters `run.MODEL_REGISTRY` under the same name.
SCREEN_ONLY = {
    f"sapiens-{size}{suffix}": PoseArm(
        f"sapiens-{size}{suffix}",
        f"model/sapiens/sapiens-{size}-f16.xml",
        (768, 1024),
        "Sapiens v1 COCO-WholeBody, UDP/DARK" + (", flip TTA" if flip else ", no flip"),
        "heatmap",
        flip,
    )
    for size in ("0.3b", "0.6b", "1b", "2b")
    for suffix, flip in (("", True), ("-noflip", False))
}

SCREENED_ARMS = POSE_ARMS | SCREEN_ONLY


def load(arm: PoseArm, device: str = "GPU"):
    """An rtmlib `RTMPose` for `arm` on the OpenVINO backend (GPU compiles at f32); heatmap arms
    compile their f16 IR at f16."""
    if arm.kind == "heatmap":
        from pose_estimation.bench.heatmap_arm import HeatmapArm

        return HeatmapArm(arm.url, device, precision="f16", flip_test=arm.flip_test)
    import rtmlib

    from pose_estimation.rtmlib_openvino import _patch_rtmlib_openvino

    _patch_rtmlib_openvino()
    return rtmlib.RTMPose(
        arm.url, model_input_size=arm.input_size, backend="openvino", device=device
    )


def predict(model, image: np.ndarray, boxes_xyxy) -> tuple[np.ndarray, np.ndarray]:
    """`(n, 133, 2)` keypoints + `(n, 133)` raw scores, one row per box."""
    kps, scores = model(image, bboxes=[list(map(float, b)) for b in boxes_xyxy])
    return np.asarray(kps, dtype=float), np.asarray(scores, dtype=float)
