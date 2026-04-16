"""Tests for bone-length constraints on COCO-WholeBody 133 layout."""

import numpy as np

from pose_estimation.constraints import BoneLengthSmoother
from pose_estimation.run import (
    BONE_SEGMENTS_WB,
    BONE_SEGMENTS_WB_BODY,
    parse_args,
)


def _make_wholebody_landmarks():
    """Return a plausible (133, 3) COCO-WholeBody landmark array.

    Places body keypoints at approximate positions and fills hand/face
    keypoints with small offsets from their anchor joints.
    """
    lm = np.zeros((133, 3), dtype=np.float64)

    # Body keypoints (0-16): rough standing pose
    lm[5] = [100, 100, 0]   # left shoulder
    lm[6] = [200, 100, 0]   # right shoulder
    lm[7] = [80, 200, 0]    # left elbow
    lm[8] = [220, 200, 0]   # right elbow
    lm[9] = [60, 300, 0]    # left wrist
    lm[10] = [240, 300, 0]  # right wrist
    lm[11] = [120, 250, 0]  # left hip
    lm[12] = [180, 250, 0]  # right hip
    lm[13] = [115, 380, 0]  # left knee
    lm[14] = [185, 380, 0]  # right knee
    lm[15] = [110, 500, 0]  # left ankle
    lm[16] = [190, 500, 0]  # right ankle

    # Left hand (91-111): cluster near left wrist
    for i in range(91, 112):
        lm[i] = lm[9] + [(i - 91) * 2, (i - 91) * 1.5, 0]

    # Right hand (112-132): cluster near right wrist
    for i in range(112, 133):
        lm[i] = lm[10] + [(i - 112) * 2, (i - 112) * 1.5, 0]

    return lm


def test_rtmw_segment_indices_valid():
    """All segment indices must be within [0, 133)."""
    for p, d in BONE_SEGMENTS_WB_BODY:
        assert 0 <= p < 133, f"proximal index {p} out of range"
        assert 0 <= d < 133, f"distal index {d} out of range"


def test_rtmw_arm_segments_subset_of_body():
    """Arm-only segments should be a prefix of the body segments."""
    n = len(BONE_SEGMENTS_WB)
    assert BONE_SEGMENTS_WB_BODY[:n] == BONE_SEGMENTS_WB


def test_rtmw_constant_landmarks_no_correction():
    """Repeated identical 133-kp frames should produce no correction."""
    smoother = BoneLengthSmoother(
        alpha=0.05, tolerance=0.4, segments=BONE_SEGMENTS_WB)
    lm = _make_wholebody_landmarks()
    original = lm.copy()

    for _ in range(30):
        result, correction = smoother.update(0, lm.copy())

    np.testing.assert_allclose(result, original, atol=1e-10)
    assert correction == 0.0


def test_rtmw_perturbed_wrist_corrected():
    """Doubling shoulder-elbow distance should trigger correction."""
    smoother = BoneLengthSmoother(
        alpha=0.05, tolerance=0.4, segments=BONE_SEGMENTS_WB)
    lm = _make_wholebody_landmarks()

    for _ in range(20):
        smoother.update(0, lm.copy())

    perturbed = lm.copy()
    elbow = lm[7]
    wrist = lm[9]
    direction = wrist - elbow
    perturbed[9] = elbow + direction * 2.0

    result, correction = smoother.update(0, perturbed)
    assert correction > 0

    corrected_len = np.linalg.norm(result[9] - result[7])
    expected_len = np.linalg.norm(lm[9] - lm[7])
    assert abs(corrected_len - expected_len) / expected_len < 0.5


def test_rtmw_body_segments_include_legs():
    """Body segment list should contain hip-knee and knee-ankle pairs."""
    leg_pairs = {(11, 13), (13, 15), (12, 14), (14, 16)}
    body_set = set(BONE_SEGMENTS_WB_BODY)
    assert leg_pairs.issubset(body_set)


def test_rtmw_body_smoother_with_legs():
    """Body-mode smoother should correct perturbed knee position."""
    smoother = BoneLengthSmoother(
        alpha=0.05, tolerance=0.4, segments=BONE_SEGMENTS_WB_BODY)
    lm = _make_wholebody_landmarks()

    for _ in range(20):
        smoother.update(0, lm.copy())

    perturbed = lm.copy()
    hip = lm[11]
    knee = lm[13]
    direction = knee - hip
    perturbed[13] = hip + direction * 2.0

    result, correction = smoother.update(0, perturbed)
    assert correction > 0


def test_no_constraints_flag():
    """--no-constraints flag should be parsed correctly."""
    import sys
    orig = sys.argv
    try:
        sys.argv = ["prog", "--no-constraints"]
        args = parse_args()
        assert args.no_constraints is True

        sys.argv = ["prog"]
        args = parse_args()
        assert args.no_constraints is False
    finally:
        sys.argv = orig


if __name__ == "__main__":
    test_rtmw_segment_indices_valid()
    test_rtmw_arm_segments_subset_of_body()
    test_rtmw_constant_landmarks_no_correction()
    test_rtmw_perturbed_wrist_corrected()
    test_rtmw_body_segments_include_legs()
    test_rtmw_body_smoother_with_legs()
    test_no_constraints_flag()
    print("All tests passed.")
