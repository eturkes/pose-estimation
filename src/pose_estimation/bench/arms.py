"""Candidate pose models for the benchmark, by exact checkpoint (contracts M4.1-M4.2).

Names follow mmpose's own table, never the checkpoint's distillation prefix: `rtmw-dw-l-m` = m,
`rtmw-dw-x-l` = L. Scores are rtmlib's raw SimCC decode (mean of the x and y maximum logits); every
checkpoint has its own scale, so nothing compares raw scores across arms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MMPOSE = "https://download.openmmlab.com/mmpose/v1/projects/"


@dataclass(frozen=True)
class PoseArm:
    name: str
    url: str
    input_size: tuple[int, int]  # (width, height)
    note: str


POSE_ARMS = {
    a.name: a
    for a in (
        PoseArm(
            "rtmw-m-256",
            _MMPOSE + "rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
            (192, 256),
            "shipped through M2.10 (registry key rtmw-l)",
        ),
        PoseArm(
            "rtmw-l-256",
            _MMPOSE + "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
            (192, 256),
            "",
        ),
        PoseArm(
            "rtmw-l-384",
            _MMPOSE + "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
            (288, 384),
            "export emits large raw logits (COCO median 4.2)",
        ),
        PoseArm(
            "rtmw-x-384-c13",
            _MMPOSE
            + "rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip",
            (288, 384),
            "cocktail13 ONNX; mmpose's table scores the cocktail14 .pth",
        ),
        PoseArm(
            "dwpose-l-384",
            _MMPOSE
            + "rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip",
            (288, 384),
            "",
        ),
    )
}


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
