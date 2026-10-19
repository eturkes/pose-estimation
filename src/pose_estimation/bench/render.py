"""Labelling sheets: a frame or a square crop under a labelled coordinate grid (contract M4.1 D02).

A labeller reads positions in grid units; `SheetMap` converts them to source-frame pixels and
back. Grid units: frame sheets = percent of frame width / height (0-100), crop sheets = cells of a
`grid` x `grid` lattice over the crop (0-`grid`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD = 40  # px of margin on every side; tick labels on all four
_MAJOR = (0, 0, 255)
_MINOR = (0, 200, 255)


@dataclass(frozen=True)
class SheetMap:
    """frame px = origin + units x scale; sheet px = PAD + (frame px - origin) x zoom."""

    origin_x: float
    origin_y: float
    scale_x: float  # frame px per grid unit
    scale_y: float
    zoom: float  # sheet px per frame px

    def to_frame(self, u: float, v: float) -> tuple[float, float]:
        return self.origin_x + u * self.scale_x, self.origin_y + v * self.scale_y

    def to_units(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.origin_x) / self.scale_x, (y - self.origin_y) / self.scale_y

    def to_sheet(self, x: float, y: float) -> tuple[float, float]:
        return PAD + (x - self.origin_x) * self.zoom, PAD + (y - self.origin_y) * self.zoom


def _label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)


def _grid(canvas: np.ndarray, sm: SheetMap, n_u: int, n_v: int, step: int, major: int) -> None:
    h, w = canvas.shape[:2]
    for i in range(0, n_u + 1, step):
        x = round(sm.to_sheet(*sm.to_frame(i, 0))[0])
        col = _MAJOR if i % major == 0 else _MINOR
        cv2.line(canvas, (x, PAD), (x, h - PAD - 1), col, 1)
        _label(canvas, str(i), x - 8, 16)
        _label(canvas, str(i), x - 8, h - 8)
    for j in range(0, n_v + 1, step):
        y = round(sm.to_sheet(*sm.to_frame(0, j))[1])
        col = _MAJOR if j % major == 0 else _MINOR
        cv2.line(canvas, (PAD, y), (w - PAD - 1, y), col, 1)
        _label(canvas, str(j), 2, y + 5)
        _label(canvas, str(j), w - PAD + 4, y + 5)


def frame_sheet(
    image: np.ndarray, max_side: int = 1600, step: int = 5
) -> tuple[np.ndarray, SheetMap]:
    """The whole frame, scaled to `max_side`, ticks every `step` percent (majors every 25)."""
    h, w = image.shape[:2]
    zoom = max_side / max(w, h)
    body = cv2.resize(image, (round(w * zoom), round(h * zoom)), interpolation=cv2.INTER_AREA)
    canvas = np.full((body.shape[0] + 2 * PAD, body.shape[1] + 2 * PAD, 3), 255, np.uint8)
    canvas[PAD:-PAD, PAD:-PAD] = body
    sm = SheetMap(0.0, 0.0, w / 100.0, h / 100.0, zoom)
    _grid(canvas, sm, 100, 100, step, 25)
    return canvas, sm


def crop_sheet(
    image: np.ndarray, centre: tuple[float, float], side: float, out: int = 640, grid: int = 16
) -> tuple[np.ndarray, SheetMap]:
    """A `side`-px square around `centre`, resampled to `out` px, a `grid` x `grid` lattice.

    Area outside the frame reads mid-grey, so a crop clipped by the frame edge stays square and
    its grid stays uniform.
    """
    zoom = out / side
    x0, y0 = centre[0] - side / 2, centre[1] - side / 2
    affine = np.array([[zoom, 0.0, -x0 * zoom], [0.0, zoom, -y0 * zoom]])
    body = cv2.warpAffine(
        image, affine, (out, out), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128)
    )
    canvas = np.full((out + 2 * PAD, out + 2 * PAD, 3), 255, np.uint8)
    canvas[PAD:-PAD, PAD:-PAD] = body
    sm = SheetMap(x0, y0, side / grid, side / grid, zoom)
    _grid(canvas, sm, grid, grid, 1, 4)
    return canvas, sm
