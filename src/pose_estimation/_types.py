"""Shared structural types for the pose-estimation pipeline.

The pipeline passes detection records and pipeline state as plain
``dict``s for performance and JSON-friendliness.  These TypedDicts
document the contract so static analysers (and humans) can spot
typos and missing keys without changing runtime behaviour.
"""

from __future__ import annotations

import sys
from typing import Any, TypedDict

import numpy as np

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # pragma: no cover - Python 3.10 fallback
    # ``NotRequired`` arrived in 3.11.  On 3.10 we model the optional
    # fields with a separate ``total=False`` TypedDict and combine via
    # multiple inheritance so the public name is unchanged.

    class _OptionalDetectionFlags(TypedDict, total=False):
        _carried: bool
        synthetic: bool
        recrop: bool

    NotRequired = None  # type: ignore[assignment]


if sys.version_info >= (3, 11):

    class Detection(TypedDict):
        """Output of an SSD detection model (pose or palm).

        ``box`` and ``keypoints`` are normalised to ``[0, 1]`` along the
        image dimensions.  ``score`` is the post-sigmoid confidence.
        The optional flags are added by downstream pipeline stages.
        """

        box: np.ndarray  # shape (4,)
        keypoints: np.ndarray  # shape (num_keypoints, 2)
        score: float
        _carried: NotRequired[bool]
        synthetic: NotRequired[bool]
        recrop: NotRequired[bool]
else:  # pragma: no cover - Python 3.10 fallback

    class _DetectionRequired(TypedDict):
        box: np.ndarray
        keypoints: np.ndarray
        score: float

    class Detection(_DetectionRequired, _OptionalDetectionFlags):
        """SSD detection record.  See the 3.11+ definition above."""


class HandDetectionDiag(TypedDict):
    """Per-hand diagnostic record produced inside ``process_frame``."""

    kind: str  # "real", "synthetic", or "recrop"
    det_score: float
    hand_flag: float
    accepted: bool


class PipelineState(TypedDict):
    """Inter-frame state threaded through ``process_frame``."""

    pose_dets: list[dict[str, Any]]
    palm_dets: list[dict[str, Any]]
    hand_diag: list[HandDetectionDiag]


__all__ = ["Detection", "HandDetectionDiag", "PipelineState"]
