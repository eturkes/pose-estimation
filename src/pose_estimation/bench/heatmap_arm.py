"""Sapiens COCO-WholeBody: UDP affine → RGB normalisation → flip TTA → DARK-UDP.

Specification: facebookresearch/sapiens pose/configs/sapiens_pose/coco_wholebody/
and pose/mmpose/codecs/{udp_heatmap,utils/refinement}.py (Meta, CC-BY-NC-4.0).
The IR emits raw heatmaps, not probabilities. Score = peak before DARK modulation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = (768, 1024)
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
# Anatomical swaps from the upstream coco_wholebody dataset metadata, including face-68.
FLIP_INDICES = (
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19,
    39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
    49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 50, 51, 52, 53, 58, 57, 56, 55, 54,
    68, 67, 66, 65, 70, 69, 62, 61, 60, 59, 64, 63, 77, 76, 75, 74, 73, 72, 71,
    82, 81, 80, 79, 78, 87, 86, 85, 84, 83, 90, 89, 88,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
    127, 128, 129, 130, 131, 132, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
)  # fmt: skip


def crop(image: np.ndarray, box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upstream GetBBoxCenterScale(padding=1.25), TopdownAffine(use_udp=True)."""
    box = np.asarray(box, np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("box must be finite xyxy with positive width and height")
    center = (box[:2] + box[2:]) * 0.5
    scale = (box[2:] - box[:2]) * 1.25
    aspect = INPUT_SIZE[0] / INPUT_SIZE[1]
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    factors = (np.asarray(INPUT_SIZE, np.float32) - 1) / scale
    matrix = np.zeros((2, 3), np.float32)
    matrix[0, 0], matrix[1, 1] = factors
    matrix[:, 2] = factors * (scale * 0.5 - center)
    patch = cv2.warpAffine(image, matrix, INPUT_SIZE, flags=cv2.INTER_LINEAR)
    rgb = (patch[:, :, ::-1].astype(np.float32) - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), center, scale


def flip_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Restore spatial + anatomical order; upstream shift_heatmap=False."""
    return heatmaps[list(FLIP_INDICES), :, ::-1]


def decode(
    heatmaps: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian UDP codec (blur=11) + estimator's inverse bbox transform."""
    heatmaps = np.asarray(heatmaps, np.float32)
    if heatmaps.ndim != 3 or heatmaps.shape[0] != 133 or not np.isfinite(heatmaps).all():
        raise ValueError("expected finite (133,H,W) heatmaps")
    k, h, w = heatmaps.shape
    if min(h, w) < 2:
        raise ValueError("heatmap spatial dimensions must exceed one")
    flat = heatmaps.reshape(k, -1)
    argmax = flat.argmax(axis=1)
    scores = flat.max(axis=1)
    xy = np.column_stack((argmax % w, argmax // w)).astype(np.float32)
    maps = np.empty_like(heatmaps)
    for j, hm in enumerate(heatmaps):
        padded = np.pad(hm, 5)
        blurred = cv2.GaussianBlur(padded, (11, 11), 0)[5:-5, 5:-5]
        peak = blurred.max()
        maps[j] = blurred * (scores[j] / peak) if peak > 0 else 0
    np.clip(maps, 1e-3, 50, out=maps)
    np.log(maps, out=maps)
    padded = np.pad(maps, ((0, 0), (1, 1), (1, 1)), mode="edge").ravel()
    idx = (xy[:, 0] + 1 + (xy[:, 1] + 1) * (w + 2)).astype(int)
    idx += (w + 2) * (h + 2) * np.arange(k)
    mid = padded[idx]
    xp, xm = padded[idx + 1], padded[idx - 1]
    yp, ym = padded[idx + w + 2], padded[idx - w - 2]
    pp, mm = padded[idx + w + 3], padded[idx - w - 3]
    gradient = np.column_stack((0.5 * (xp - xm), 0.5 * (yp - ym)))
    cross = 0.5 * (pp - xp - yp + 2 * mid - xm - ym + mm)
    hessian = np.stack((xp - 2 * mid + xm, cross, cross, yp - 2 * mid + ym), axis=1)
    hessian = hessian.reshape(k, 2, 2) + np.finfo(np.float32).eps * np.eye(2)
    xy -= np.linalg.solve(hessian, gradient[..., None])[..., 0]
    # Codec x/(W-1)*input_width, then estimator /input_width*scale: sizes cancel.
    xy = xy / np.array([w - 1, h - 1]) * scale + center - scale * 0.5
    xy[scores <= 0] = -1
    return xy, scores


class HeatmapArm:
    """One exact OpenVINO device; compile failures stay loud (no CPU fallback)."""

    def __init__(
        self,
        path: str | Path,
        device: str = "GPU",
        *,
        precision: str = "f32",
        flip_test: bool = True,
    ) -> None:
        import openvino as ov

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing pose IR {path}; run uv run --script scripts/convert/sapiens.py "
                "--size <arm size> --out model/sapiens --stage export"
            )
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if device in ("GPU", "CPU"):
            config["INFERENCE_PRECISION_HINT"] = precision
        self.compiled = ov.Core().compile_model(str(path), device, config)
        self.execution_devices = self.compiled.get_property("EXECUTION_DEVICES")
        self.flip_test = flip_test
        print(
            f"Sapiens: {path.name} requested={device} actual={self.execution_devices}", flush=True
        )

    def heatmaps(self, tensor: np.ndarray) -> np.ndarray:
        return self.compiled([tensor])[self.compiled.output(0)][0].copy()

    def __call__(self, image: np.ndarray, bboxes) -> tuple[np.ndarray, np.ndarray]:
        keypoints = np.empty((len(bboxes), 133, 2), np.float32)
        scores = np.empty((len(bboxes), 133), np.float32)
        for i, box in enumerate(bboxes):
            tensor, center, scale = crop(image, box)
            hm = self.heatmaps(tensor)
            if self.flip_test:
                flipped = self.heatmaps(np.ascontiguousarray(tensor[..., ::-1]))
                hm = (hm + flip_heatmaps(flipped)) * 0.5
            keypoints[i], scores[i] = decode(hm, center, scale)
        return keypoints, scores
