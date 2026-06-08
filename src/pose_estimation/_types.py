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


class CameraCalibration(TypedDict):
    """Single-camera intrinsics + extrinsics for a multi-camera session.

    All matrix/vector fields hold ``np.ndarray`` instances after a
    successful ``load_calibration`` call.  JSON-on-disk uses nested
    lists; the loader materialises arrays so downstream code does not
    branch on type.
    """

    name: str
    resolution: tuple[int, int]  # (width, height) in pixels
    K: np.ndarray  # (3, 3) intrinsic matrix
    distortion: np.ndarray  # (k,) OpenCV distortion coefficients
    rvec: np.ndarray  # (3,) Rodrigues rotation, camera-from-world
    tvec: np.ndarray  # (3,) translation, camera-from-world (metres)


class SessionCalibration(TypedDict):
    """Collection of camera calibrations for one session.

    ``cameras`` is keyed by camera ``name`` for O(1) lookup.  The
    on-disk representation is a list; ``load_calibration`` builds the
    dict.  ``world_frame`` names the camera whose pose is the world
    origin (its rvec/tvec must be zero).
    """

    format_version: int
    session_id: str
    world_frame: str
    cameras: dict[str, CameraCalibration]
    reprojection_error_px: float
    solver: str
    solved_at: str  # ISO-8601 UTC timestamp


class SessionFrame(TypedDict):
    """One synchronized multi-camera frame yielded by the session iterator.

    ``frames`` is keyed by camera name so consumers do not depend on
    camera order.  ``frame_index`` is the *logical* index (post
    sync-offset alignment); per-camera raw indices may differ.
    """

    frame_index: int
    frames: dict[str, np.ndarray]  # BGR images, varying resolutions allowed


class FusionDiagnostics(TypedDict):
    """Per-keypoint diagnostics returned by ``fuse_session_frame``.

    All arrays are length ``N`` (one entry per keypoint).  A keypoint
    that could not be triangulated has ``NaN`` reprojection error,
    zero confidence, and ``cheirality_ok=False``; its ``n_views``
    still reports how many valid views were available.
    """

    n_views: np.ndarray  # (N,) int — views contributing to the estimate
    confidence: np.ndarray  # (N,) float — mean confidence of contributing views
    reprojection_error_px: np.ndarray  # (N,) float — mean over views; NaN if unfused
    cheirality_ok: np.ndarray  # (N,) bool — fused and in front of all contributing cameras


if sys.version_info >= (3, 11):

    class MultiCamPipelineState(TypedDict):
        """Inter-frame state for the multi-camera pipeline.

        Holds one ``PipelineState`` per camera (keyed by name) plus
        room for shared 3D state once triangulation is wired.
        ``world_keypoints`` is populated only when fusion runs.
        """

        per_camera: dict[str, PipelineState]
        world_keypoints: NotRequired[np.ndarray]  # (n_keypoints, 3)
else:  # pragma: no cover - Python 3.10 fallback

    class _MultiCamRequired(TypedDict):
        per_camera: dict[str, PipelineState]

    class _MultiCamOptional(TypedDict, total=False):
        world_keypoints: np.ndarray

    class MultiCamPipelineState(_MultiCamRequired, _MultiCamOptional):
        """Multi-camera pipeline state.  See 3.11+ definition above."""


__all__ = [
    "CameraCalibration",
    "Detection",
    "FusionDiagnostics",
    "HandDetectionDiag",
    "MultiCamPipelineState",
    "PipelineState",
    "SessionCalibration",
    "SessionFrame",
]
