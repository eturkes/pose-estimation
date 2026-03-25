"""Tests for biomechanical constraints (bone length and joint angles)."""

import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from constraints import BoneLengthSmoother, BONE_SEGMENTS, clamp_joint_angles


def _make_landmarks():
    """Return a plausible (12, 3) arm landmark array.

    Layout (approximate pixel positions):
        0  left shoulder   (100, 100)
        1  right shoulder  (200, 100)
        2  left elbow      (80,  200)
        3  right elbow     (220, 200)
        4  left wrist      (60,  300)
        5  right wrist     (240, 300)
        6  left index base (55,  330)
        7  right index base(245, 330)
        8-11 other finger bases (not involved in bone segments)
    """
    lm = np.zeros((12, 3), dtype=np.float64)
    lm[0] = [100, 100, 0]
    lm[1] = [200, 100, 0]
    lm[2] = [80, 200, 0]
    lm[3] = [220, 200, 0]
    lm[4] = [60, 300, 0]
    lm[5] = [240, 300, 0]
    lm[6] = [55, 330, 0]
    lm[7] = [245, 330, 0]
    lm[8] = [50, 325, 0]
    lm[9] = [250, 325, 0]
    lm[10] = [45, 320, 0]
    lm[11] = [255, 320, 0]
    return lm


def test_constant_landmarks_no_correction():
    """Repeated calls with identical landmarks should produce no change."""
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.4)
    lm = _make_landmarks()
    original = lm.copy()

    for _ in range(30):
        result, correction = smoother.update(0, lm.copy())

    np.testing.assert_allclose(result, original, atol=1e-10)
    assert correction == 0.0


def test_perturbed_keypoint_corrected():
    """A keypoint displaced 2x its expected distance is pulled back."""
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.4)
    lm = _make_landmarks()

    # Prime the EMA with 20 consistent frames
    for _ in range(20):
        smoother.update(0, lm.copy())

    # Perturb left wrist (index 4) — double the distance from elbow
    perturbed = lm.copy()
    elbow = lm[2]
    wrist = lm[4]
    direction = wrist - elbow
    perturbed[4] = elbow + direction * 2.0  # 2x normal distance

    result, correction = smoother.update(0, perturbed)
    assert correction > 0

    # After correction the left elbow→wrist distance should be close
    # to the EMA (within tolerance), not 2x.
    corrected_len = np.linalg.norm(result[4] - result[2])
    expected_len = np.linalg.norm(lm[4] - lm[2])
    assert abs(corrected_len - expected_len) / expected_len < 0.5, (
        f"corrected_len={corrected_len:.1f}, expected≈{expected_len:.1f}"
    )

    # Verify direction is preserved
    orig_dir = direction / np.linalg.norm(direction)
    corr_dir = (result[4] - result[2])
    corr_dir /= np.linalg.norm(corr_dir)
    np.testing.assert_allclose(corr_dir, orig_dir, atol=1e-6)


def test_ema_converges():
    """EMA should converge to the true bone lengths within ~20 frames."""
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.4)
    lm = _make_landmarks()

    true_lengths = np.array([
        np.linalg.norm(lm[d] - lm[p]) for p, d in BONE_SEGMENTS
    ])

    for i in range(25):
        _, _ = smoother.update(0, lm.copy())

    avg = smoother._averages[0]
    # After 25 frames at alpha=0.05, EMA should be within ~30% of true
    # (1 - 0.95^25 ≈ 0.72 of the way there from any starting point,
    # but we start at the first observation so convergence is immediate).
    np.testing.assert_allclose(avg, true_lengths, rtol=0.01)


def test_prune_removes_stale():
    """prune() should drop state for IDs not in the active set."""
    smoother = BoneLengthSmoother()
    lm = _make_landmarks()
    _, _ = smoother.update(0, lm.copy())
    _, _ = smoother.update(1, lm.copy())
    _, _ = smoother.update(2, lm.copy())

    smoother.prune([0, 2])
    assert 1 not in smoother._averages
    assert 0 in smoother._averages
    assert 2 in smoother._averages


def test_small_movements_within_tolerance_pass_through():
    """Perturbations within tolerance should not trigger correction."""
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.4)
    lm = _make_landmarks()

    for _ in range(20):
        smoother.update(0, lm.copy())

    # Small perturbation (10% of bone length)
    perturbed = lm.copy()
    elbow = lm[2]
    wrist = lm[4]
    direction = (wrist - elbow)
    direction /= np.linalg.norm(direction)
    perturbed[4] = wrist + direction * np.linalg.norm(wrist - elbow) * 0.1

    result, correction = smoother.update(0, perturbed)

    # Should NOT be corrected — within 40% tolerance
    np.testing.assert_allclose(result[4], perturbed[4], atol=1e-10)
    assert correction == 0.0


# -----------------------------------------------------------------------
# Joint-angle clamping tests
# -----------------------------------------------------------------------

def _make_bent_landmarks():
    """Return (12, 3) landmarks with naturally bent elbows (~120°).

    The default _make_landmarks() has perfectly straight arms (180°),
    which is outside the 170° limit.  This variant bends the elbows
    so all joint angles sit comfortably inside the allowed range.
    """
    lm = _make_landmarks()
    # Bend left arm: move wrist rightward so elbow angle ≈ 120°
    lm[4] = [100, 290, 0]
    lm[6] = [105, 320, 0]
    # Bend right arm: move wrist leftward
    lm[5] = [200, 290, 0]
    lm[7] = [195, 320, 0]
    return lm


def _angle_at_joint(landmarks, prox, joint, dist):
    """Return the unsigned angle (degrees) at *joint* in 2D."""
    v1 = landmarks[prox, :2] - landmarks[joint, :2]
    v2 = landmarks[dist, :2] - landmarks[joint, :2]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def test_angle_within_limits_unchanged():
    """Landmarks with valid joint angles should not be modified."""
    lm = _make_bent_landmarks()
    original = lm.copy()
    _, n_clamped = clamp_joint_angles(lm)
    np.testing.assert_allclose(lm, original, atol=1e-10)
    assert n_clamped == 0


def test_angle_below_minimum_clamped():
    """An elbow angle below 30° should be opened to 30°."""
    lm = _make_landmarks()
    # Place left wrist very close to left shoulder direction (tiny angle)
    # Shoulder at (100,100), elbow at (80,200)
    # v1 = shoulder - elbow = (20, -100)
    # Place wrist so v2 is nearly parallel to v1 → angle ≈ 0°
    v1 = lm[0, :2] - lm[2, :2]  # (20, -100)
    v1_hat = v1 / np.linalg.norm(v1)
    forearm_len = np.linalg.norm(lm[4, :2] - lm[2, :2])
    # Angle = 10° from v1 direction
    angle_10 = np.radians(10)
    cos_a, sin_a = np.cos(angle_10), np.sin(angle_10)
    rotated = np.array([
        v1_hat[0] * cos_a - v1_hat[1] * sin_a,
        v1_hat[0] * sin_a + v1_hat[1] * cos_a,
    ])
    lm[4, :2] = lm[2, :2] + rotated * forearm_len

    assert _angle_at_joint(lm, 0, 2, 4) < 30

    _, n_clamped = clamp_joint_angles(lm)
    assert n_clamped >= 1

    result_angle = _angle_at_joint(lm, 0, 2, 4)
    assert abs(result_angle - 30) < 0.5, f"Expected ~30°, got {result_angle:.1f}°"

    # Segment length should be preserved
    new_len = np.linalg.norm(lm[4, :2] - lm[2, :2])
    assert abs(new_len - forearm_len) < 1e-6


def test_angle_above_maximum_clamped():
    """An elbow angle beyond 170° should be pulled back to 170°."""
    lm = _make_landmarks()
    # Place left wrist so the elbow angle is ~175°
    v1 = lm[0, :2] - lm[2, :2]
    v1_hat = v1 / np.linalg.norm(v1)
    forearm_len = np.linalg.norm(lm[4, :2] - lm[2, :2])
    angle_175 = np.radians(175)
    cos_a, sin_a = np.cos(angle_175), np.sin(angle_175)
    rotated = np.array([
        v1_hat[0] * cos_a - v1_hat[1] * sin_a,
        v1_hat[0] * sin_a + v1_hat[1] * cos_a,
    ])
    lm[4, :2] = lm[2, :2] + rotated * forearm_len

    assert _angle_at_joint(lm, 0, 2, 4) > 170

    _, n_clamped = clamp_joint_angles(lm)
    assert n_clamped >= 1

    result_angle = _angle_at_joint(lm, 0, 2, 4)
    assert abs(result_angle - 170) < 0.5, f"Expected ~170°, got {result_angle:.1f}°"


def test_angle_clamp_preserves_z():
    """Clamping should only modify x/y; z is left unchanged."""
    lm = _make_landmarks()
    lm[:, 2] = np.arange(12) * 10.0  # give each keypoint a distinct z

    # Force an out-of-range angle
    v1 = lm[0, :2] - lm[2, :2]
    v1_hat = v1 / np.linalg.norm(v1)
    forearm_len = np.linalg.norm(lm[4, :2] - lm[2, :2])
    angle_10 = np.radians(10)
    cos_a, sin_a = np.cos(angle_10), np.sin(angle_10)
    rotated = np.array([
        v1_hat[0] * cos_a - v1_hat[1] * sin_a,
        v1_hat[0] * sin_a + v1_hat[1] * cos_a,
    ])
    lm[4, :2] = lm[2, :2] + rotated * forearm_len
    z_before = lm[4, 2]

    _, _ = clamp_joint_angles(lm)

    assert lm[4, 2] == z_before, "z coordinate should not change"


def test_angle_clamp_right_elbow():
    """Verify the right elbow triplet (1, 3, 5) is also clamped."""
    lm = _make_landmarks()
    v1 = lm[1, :2] - lm[3, :2]
    v1_hat = v1 / np.linalg.norm(v1)
    forearm_len = np.linalg.norm(lm[5, :2] - lm[3, :2])
    angle_10 = np.radians(10)
    cos_a, sin_a = np.cos(angle_10), np.sin(angle_10)
    rotated = np.array([
        v1_hat[0] * cos_a - v1_hat[1] * sin_a,
        v1_hat[0] * sin_a + v1_hat[1] * cos_a,
    ])
    lm[5, :2] = lm[3, :2] + rotated * forearm_len

    _, n_clamped = clamp_joint_angles(lm)
    assert n_clamped >= 1

    result_angle = _angle_at_joint(lm, 1, 3, 5)
    assert abs(result_angle - 30) < 0.5, f"Expected ~30°, got {result_angle:.1f}°"


if __name__ == "__main__":
    test_constant_landmarks_no_correction()
    test_perturbed_keypoint_corrected()
    test_ema_converges()
    test_prune_removes_stale()
    test_small_movements_within_tolerance_pass_through()
    test_angle_within_limits_unchanged()
    test_angle_below_minimum_clamped()
    test_angle_above_maximum_clamped()
    test_angle_clamp_preserves_z()
    test_angle_clamp_right_elbow()
    print("All tests passed.")
