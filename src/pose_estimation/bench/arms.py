"""Candidate pose models for the benchmark, by exact checkpoint (contracts M4.1-M4.2).

Names follow mmpose's own table, never the checkpoint's distillation prefix: `rtmw-dw-l-m` = m,
`rtmw-dw-x-l` = L. Scores are rtmlib's raw SimCC decode (mean of the x and y maximum logits); every
checkpoint has its own scale, so nothing compares raw scores across arms.
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


def load(arm: PoseArm, device: str = "GPU"):
    """An rtmlib `RTMPose` for `arm` on the OpenVINO backend (GPU compiles at f32)."""
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
