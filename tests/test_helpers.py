"""Tests for the small refactored helpers and CLI validators.

Covers:
- ``processing._detection_centres`` and ``_make_palm_keypoints``
- ``processing._carry_detection`` (single source of truth for decay)
- ``postprocess._odd_int`` argparse validator
- ``main._safe_fps`` clamp behaviour
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from pose_estimation.main import (
    FALLBACK_FPS,
    MAX_REASONABLE_FPS,
    _safe_fps,
)
from pose_estimation.postprocess import _odd_int
from pose_estimation.processing import (
    CARRIED_DET_SCORE_DECAY,
    PALM_FINGER_KP_IDX,
    PALM_KP_COUNT,
    PALM_WRIST_KP_IDX,
    _carry_detection,
    _detection_centre,
    _detection_centres,
    _make_palm_keypoints,
)

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _det(cx, cy, size=0.1):
    half = size / 2
    return {
        "box": np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32),
        "keypoints": np.array([[cx, cy]] * 7, dtype=np.float32),
        "score": 0.9,
    }


def test_detection_centre_matches_box_midpoint():
    det = _det(0.5, 0.4, size=0.2)
    centre = _detection_centre(det)
    np.testing.assert_allclose(centre, [0.5, 0.4], atol=1e-6)


def test_detection_centres_empty():
    assert _detection_centres([]) == []


def test_detection_centres_multiple():
    dets = [_det(0.1, 0.2), _det(0.7, 0.8)]
    centres = _detection_centres(dets)
    assert len(centres) == 2
    np.testing.assert_allclose(centres[0], [0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(centres[1], [0.7, 0.8], atol=1e-6)


# ---------------------------------------------------------------------------
# Palm keypoint factory
# ---------------------------------------------------------------------------


def test_make_palm_keypoints_shape_and_dtype():
    centre = np.array([0.5, 0.5], dtype=np.float32)
    wrist = np.array([0.45, 0.55], dtype=np.float32)
    finger = np.array([0.50, 0.40], dtype=np.float32)

    kps = _make_palm_keypoints(centre, wrist, finger)
    assert kps.shape == (PALM_KP_COUNT, 2)
    assert kps.dtype == np.float32


def test_make_palm_keypoints_anchor_indices():
    """Wrist and finger MCP land at the canonical palm-detection slots."""
    centre = np.array([0.5, 0.5], dtype=np.float32)
    wrist = np.array([0.1, 0.2], dtype=np.float32)
    finger = np.array([0.3, 0.4], dtype=np.float32)

    kps = _make_palm_keypoints(centre, wrist, finger)
    np.testing.assert_allclose(kps[PALM_WRIST_KP_IDX], wrist)
    np.testing.assert_allclose(kps[PALM_FINGER_KP_IDX], finger)
    # All other slots default to the centre
    for i in range(PALM_KP_COUNT):
        if i in (PALM_WRIST_KP_IDX, PALM_FINGER_KP_IDX):
            continue
        np.testing.assert_allclose(kps[i], centre)


def test_make_palm_keypoints_does_not_alias_centre():
    """Mutating the result must not leak into the centre array."""
    centre = np.array([0.5, 0.5], dtype=np.float32)
    wrist = np.array([0.1, 0.2], dtype=np.float32)
    finger = np.array([0.3, 0.4], dtype=np.float32)

    kps = _make_palm_keypoints(centre, wrist, finger)
    kps[3] = [0.99, 0.99]
    np.testing.assert_allclose(centre, [0.5, 0.5])


# ---------------------------------------------------------------------------
# Carry detection
# ---------------------------------------------------------------------------


def test_carry_detection_decays_score_and_marks_carried():
    det = _det(0.5, 0.5)
    det["score"] = 0.8
    out = _carry_detection(det)

    assert out["_carried"] is True
    assert abs(out["score"] - 0.8 * CARRIED_DET_SCORE_DECAY) < 1e-9
    # Original is untouched
    assert "_carried" not in det
    assert det["score"] == 0.8


# ---------------------------------------------------------------------------
# _odd_int argparse validator
# ---------------------------------------------------------------------------


def test_odd_int_accepts_odd_positive():
    assert _odd_int("11") == 11
    assert _odd_int("3") == 3


def test_odd_int_rejects_even():
    with pytest.raises(argparse.ArgumentTypeError, match="must be odd"):
        _odd_int("10")


def test_odd_int_rejects_too_small():
    with pytest.raises(argparse.ArgumentTypeError, match="≥ 3"):
        _odd_int("1")


def test_odd_int_rejects_non_integer():
    with pytest.raises(argparse.ArgumentTypeError, match="expected an integer"):
        _odd_int("abc")


# ---------------------------------------------------------------------------
# _safe_fps
# ---------------------------------------------------------------------------


def test_safe_fps_passes_through_normal():
    assert _safe_fps(30.0) == 30.0
    assert _safe_fps(60.0) == 60.0
    assert _safe_fps(24.0) == 24.0


def test_safe_fps_replaces_zero_or_negative():
    assert _safe_fps(0.0) == FALLBACK_FPS
    assert _safe_fps(-1.0) == FALLBACK_FPS


def test_safe_fps_replaces_nonfinite():
    assert _safe_fps(float("nan")) == FALLBACK_FPS
    assert _safe_fps(float("inf")) == FALLBACK_FPS


def test_safe_fps_replaces_outliers(capsys):
    # Above the upper bound — falls back to FALLBACK_FPS with a warning
    result = _safe_fps(MAX_REASONABLE_FPS + 100)
    assert result == FALLBACK_FPS
    captured = capsys.readouterr()
    assert "unusual FPS" in captured.out
