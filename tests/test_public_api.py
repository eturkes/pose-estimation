"""Smoke tests for the package's public API surface.

Locks in the names exported from ``pose_estimation`` so future
refactors don't accidentally break library consumers.
"""

from __future__ import annotations

import pose_estimation as pe


def test_smoothers_exported():
    assert pe.OneEuroFilter is not None
    assert pe.PoseSmoother is not None
    assert pe.BoneLengthSmoother is not None


def test_pipeline_entry_points_exported():
    assert callable(pe.process_frame)
    assert callable(pe.match_hands_to_arms)
    assert callable(pe.select_primary_body)
    assert callable(pe.tracking_pose_indices)


def test_tracking_constants_exported():
    assert pe.TRACKING_HANDS == "hands"
    assert pe.TRACKING_HANDS_ARMS == "hands-arms"
    assert pe.TRACKING_BODY == "body"


def test_constraint_constants_exported():
    assert isinstance(pe.BONE_SEGMENTS, list)
    assert isinstance(pe.BONE_SEGMENTS_BODY, list)
    assert isinstance(pe.ANGLE_LIMITS, dict)
    assert isinstance(pe.ANGLE_LIMITS_BODY, dict)


def test_typed_dicts_exported():
    # TypedDicts are not instantiable for runtime checks, but the names
    # must be importable so downstream users can annotate against them.
    assert pe.Detection is not None
    assert pe.PipelineState is not None
    assert pe.HandDetectionDiag is not None


def test_savgol_helper_exported():
    assert callable(pe.savgol_smooth_csv)


def test_models_helper_exported():
    assert callable(pe.download_and_compile_models)


def test_multicam_surface_exported():
    # Dataclasses / helpers for multi-camera sessions.
    assert pe.Session is not None
    assert pe.SessionCamera is not None
    assert callable(pe.discover_session)
    assert callable(pe.discover_sessions)
    assert callable(pe.iter_synchronized_frames)
    assert callable(pe.process_session)
    assert issubclass(pe.SessionError, ValueError)


def test_calibration_surface_exported():
    assert callable(pe.load_calibration)
    assert callable(pe.load_session_calibration)
    assert callable(pe.save_calibration)
    assert issubclass(pe.CalibrationError, ValueError)
    # TypedDicts: importable so callers can annotate.
    assert pe.CameraCalibration is not None
    assert pe.SessionCalibration is not None
    assert pe.SessionFrame is not None
    assert pe.MultiCamPipelineState is not None


def test_mapping_surface_exported():
    assert callable(pe.coco_to_mediapipe)


def test_triangulation_surface_exported():
    assert callable(pe.fuse_session_frame)


def test_all_lists_only_existing_names():
    for name in pe.__all__:
        assert hasattr(pe, name), f"__all__ lists {name!r} but it isn't defined"
