"""Pose estimation pipeline combining MediaPipe TFLite and OpenVINO."""

from ._types import (
    CameraCalibration,
    Detection,
    HandDetectionDiag,
    MultiCamPipelineState,
    PipelineState,
    SessionCalibration,
    SessionFrame,
)
from .calibration import (
    CalibrationError,
    load_calibration,
    load_session_calibration,
    save_calibration,
)
from .constraints import (
    ANGLE_LIMITS,
    ANGLE_LIMITS_BODY,
    BONE_SEGMENTS,
    BONE_SEGMENTS_BODY,
    BoneLengthSmoother,
    clamp_joint_angles,
)
from .mapping import coco_to_mediapipe
from .models import download_and_compile_models
from .multicam import (
    Session,
    SessionCamera,
    SessionError,
    discover_session,
    discover_sessions,
    iter_synchronized_frames,
    process_session,
)
from .postprocess import savgol_smooth_csv
from .processing import (
    TRACKING_BODY,
    TRACKING_HANDS,
    TRACKING_HANDS_ARMS,
    match_hands_to_arms,
    process_frame,
    select_primary_body,
    tracking_pose_indices,
)
from .smoothing import OneEuroFilter, PoseSmoother
from .triangulation import fuse_session_frame

__all__ = [
    "ANGLE_LIMITS",
    "ANGLE_LIMITS_BODY",
    "BONE_SEGMENTS",
    "BONE_SEGMENTS_BODY",
    "TRACKING_BODY",
    "TRACKING_HANDS",
    "TRACKING_HANDS_ARMS",
    "BoneLengthSmoother",
    "CalibrationError",
    "CameraCalibration",
    "Detection",
    "HandDetectionDiag",
    "MultiCamPipelineState",
    "OneEuroFilter",
    "PipelineState",
    "PoseSmoother",
    "Session",
    "SessionCalibration",
    "SessionCamera",
    "SessionError",
    "SessionFrame",
    "clamp_joint_angles",
    "coco_to_mediapipe",
    "discover_session",
    "discover_sessions",
    "download_and_compile_models",
    "fuse_session_frame",
    "iter_synchronized_frames",
    "load_calibration",
    "load_session_calibration",
    "match_hands_to_arms",
    "process_frame",
    "process_session",
    "save_calibration",
    "savgol_smooth_csv",
    "select_primary_body",
    "tracking_pose_indices",
]
