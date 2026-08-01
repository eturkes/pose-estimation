"""Tests for biomechanical constraints (bone length and joint angles)."""

import numpy as np

from pose_estimation.constraints import BONE_SEGMENTS, BoneLengthSmoother, clamp_joint_angles


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

    elbow_before = perturbed[2].copy()
    wrist_before = perturbed[4].copy()

    result, correction = smoother.update(0, perturbed)
    assert correction > 0

    # Both endpoints should have shifted (proportional correction)
    assert np.linalg.norm(result[4] - wrist_before) > 1e-6, "distal keypoint should have moved"
    assert np.linalg.norm(result[2] - elbow_before) > 1e-6, (
        "proximal keypoint should also have moved"
    )

    # After correction the left elbow→wrist distance should be close
    # to the EMA (within tolerance), not 2x.
    corrected_len = np.linalg.norm(result[4] - result[2])
    expected_len = np.linalg.norm(lm[4] - lm[2])
    assert abs(corrected_len - expected_len) / expected_len < 0.5, (
        f"corrected_len={corrected_len:.1f}, expected≈{expected_len:.1f}"
    )


def test_ema_converges():
    """EMA should converge to the true bone lengths within ~20 frames."""
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.4)
    lm = _make_landmarks()

    true_lengths = np.array([np.linalg.norm(lm[d] - lm[p]) for p, d in BONE_SEGMENTS])

    for _i in range(25):
        _, _ = smoother.update(0, lm.copy())

    avg = smoother._averages[0]
    # After 25 frames at alpha=0.05, EMA should be within ~30% of true
    # (1 - 0.95^25 ≈ 0.72 of the way there from any starting point,
    # but we start at the first observation so convergence is immediate).
    np.testing.assert_allclose(avg, true_lengths, rtol=0.01)


def test_outlier_does_not_poison_running_bone_length():
    """A corrected observation must not drag the learned length toward it."""
    smoother = BoneLengthSmoother(alpha=0.2, tolerance=0.1, segments=[(0, 1)])
    baseline = np.array([[0.0, 0.0], [10.0, 0.0]])
    for _ in range(10):
        smoother.update(0, baseline.copy())

    outlier = np.array([[0.0, 0.0], [1000.0, 0.0]])
    result, correction = smoother.update(0, outlier)

    assert correction > 0.0
    assert smoother._averages[0][0] <= 10.2
    assert np.linalg.norm(result[1] - result[0]) <= 11.0


def test_nan_segment_initialises_after_recovery():
    """One missing first-frame segment must not poison its EMA forever."""
    smoother = BoneLengthSmoother(segments=[(0, 1), (1, 2)])
    missing = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan]])
    smoother.update(0, missing)

    assert np.isfinite(smoother._averages[0][0])
    assert np.isnan(smoother._averages[0][1])

    valid = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    smoother.update(0, valid)
    assert np.all(np.isfinite(smoother._averages[0]))


def test_constraints_use_image_xy_and_preserve_z():
    """Crop-relative z must neither mask an x/y outlier nor be modified."""
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.1, segments=[(0, 1)])
    baseline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 1000.0]])
    for _ in range(5):
        smoother.update(0, baseline.copy())

    stretched = baseline.copy()
    stretched[1, 0] = 100.0
    z_before = stretched[:, 2].copy()
    result, correction = smoother.update(0, stretched)

    assert correction > 0.0
    assert np.linalg.norm(result[1, :2] - result[0, :2]) <= 11.0
    np.testing.assert_array_equal(result[:, 2], z_before)


def test_iterative_projection_repairs_connected_chain():
    """Correcting a distal bone must not leave its upstream neighbour broken."""
    smoother = BoneLengthSmoother(
        alpha=0.05,
        tolerance=0.1,
        distal_weight=0.8,
        segments=[(0, 1), (1, 2)],
        max_iterations=20,
    )
    baseline = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    for _ in range(5):
        smoother.update(0, baseline.copy())

    outlier = baseline.copy()
    outlier[2, 0] = 100.0
    result, correction = smoother.update(0, outlier)
    lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)

    assert correction > 0.0
    np.testing.assert_allclose(lengths, 10.0, rtol=0.1)


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


def test_proportional_correction_direction():
    """Proximal keypoint should move toward the perturbed distal keypoint."""
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.4)
    lm = _make_landmarks()

    for _ in range(20):
        smoother.update(0, lm.copy())

    perturbed = lm.copy()
    elbow = lm[2]
    wrist = lm[4]
    direction = wrist - elbow
    perturbed[4] = elbow + direction * 2.0

    elbow_before = perturbed[2].copy()
    result, _ = smoother.update(0, perturbed)

    # Proximal (elbow) should have moved toward the distal (wrist)
    toward_distal = perturbed[4] - elbow_before
    proximal_shift = result[2] - elbow_before
    # Positive dot product means same general direction
    assert np.dot(toward_distal, proximal_shift) > 0, (
        "proximal keypoint should move toward the distal keypoint"
    )


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
    direction = wrist - elbow
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

    This variant bends the elbows so all joint angles sit comfortably inside
    the allowed range and can exercise the no-op path away from a boundary.
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
    rotated = np.array(
        [
            v1_hat[0] * cos_a - v1_hat[1] * sin_a,
            v1_hat[0] * sin_a + v1_hat[1] * cos_a,
        ]
    )
    lm[4, :2] = lm[2, :2] + rotated * forearm_len

    assert _angle_at_joint(lm, 0, 2, 4) < 30

    _, n_clamped = clamp_joint_angles(lm)
    assert n_clamped >= 1

    result_angle = _angle_at_joint(lm, 0, 2, 4)
    assert abs(result_angle - 30) < 0.5, f"Expected ~30°, got {result_angle:.1f}°"

    # Segment length should be preserved
    new_len = np.linalg.norm(lm[4, :2] - lm[2, :2])
    assert abs(new_len - forearm_len) < 1e-6


def test_custom_angle_above_maximum_clamped():
    """A caller-supplied 170° maximum is still enforced."""
    lm = _make_landmarks()
    # Place left wrist so the elbow angle is ~175°
    v1 = lm[0, :2] - lm[2, :2]
    v1_hat = v1 / np.linalg.norm(v1)
    forearm_len = np.linalg.norm(lm[4, :2] - lm[2, :2])
    angle_175 = np.radians(175)
    cos_a, sin_a = np.cos(angle_175), np.sin(angle_175)
    rotated = np.array(
        [
            v1_hat[0] * cos_a - v1_hat[1] * sin_a,
            v1_hat[0] * sin_a + v1_hat[1] * cos_a,
        ]
    )
    lm[4, :2] = lm[2, :2] + rotated * forearm_len

    assert _angle_at_joint(lm, 0, 2, 4) > 170

    _, n_clamped = clamp_joint_angles(lm, limits={(0, 2, 4): (30, 170)})
    assert n_clamped >= 1

    result_angle = _angle_at_joint(lm, 0, 2, 4)
    assert abs(result_angle - 170) < 0.5, f"Expected ~170°, got {result_angle:.1f}°"


def test_default_angle_limit_preserves_straight_limb_without_side_flip():
    """Near-180° limbs are valid and must not jump between bend sides."""
    outputs = []
    for epsilon in (-1e-6, 1e-6):
        lm = np.zeros((12, 3), dtype=np.float64)
        lm[0, :2] = (-100.0, 0.0)
        lm[2, :2] = (0.0, 0.0)
        lm[4, :2] = (100.0, epsilon)
        before = lm.copy()
        _, n_clamped = clamp_joint_angles(lm)
        assert n_clamped == 0
        np.testing.assert_allclose(lm, before)
        outputs.append(lm[4, :2].copy())

    assert np.linalg.norm(outputs[0] - outputs[1]) < 1e-4


def test_angle_clamp_rotates_complete_distal_branch():
    """Wrist-to-finger geometry remains rigid when the elbow is clamped."""
    lm = _make_landmarks()
    distances_before = [np.linalg.norm(lm[i, :2] - lm[4, :2]) for i in (6, 8, 10)]

    _, n_clamped = clamp_joint_angles(lm, limits={(0, 2, 4): (30, 120)})
    distances_after = [np.linalg.norm(lm[i, :2] - lm[4, :2]) for i in (6, 8, 10)]

    assert n_clamped == 1
    np.testing.assert_allclose(distances_after, distances_before, atol=1e-10)


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
    rotated = np.array(
        [
            v1_hat[0] * cos_a - v1_hat[1] * sin_a,
            v1_hat[0] * sin_a + v1_hat[1] * cos_a,
        ]
    )
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
    rotated = np.array(
        [
            v1_hat[0] * cos_a - v1_hat[1] * sin_a,
            v1_hat[0] * sin_a + v1_hat[1] * cos_a,
        ]
    )
    lm[5, :2] = lm[3, :2] + rotated * forearm_len

    _, n_clamped = clamp_joint_angles(lm)
    assert n_clamped >= 1

    result_angle = _angle_at_joint(lm, 1, 3, 5)
    assert abs(result_angle - 30) < 0.5, f"Expected ~30°, got {result_angle:.1f}°"


def test_bone_constraint_does_not_move_segment_with_invalid_endpoint():
    smoother = BoneLengthSmoother(alpha=0.05, tolerance=0.1)
    baseline = _make_landmarks()
    smoother.update(7, baseline.copy(), validity=np.ones(12, dtype=bool))

    current = baseline.copy()
    current[4, 0] += 200.0
    validity = np.ones(12, dtype=bool)
    validity[4] = False
    before = current.copy()

    _, correction = smoother.update(7, current, validity=validity)

    assert correction == 0.0
    np.testing.assert_array_equal(current, before)


def test_angle_constraint_skips_invalid_triplet_and_branch_point():
    lm = _make_landmarks()
    original = lm.copy()
    validity = np.ones(12, dtype=bool)
    validity[4] = False

    _, n_clamped = clamp_joint_angles(lm, limits={(0, 2, 4): (30, 60)}, validity=validity)

    assert n_clamped == 0
    np.testing.assert_array_equal(lm, original)

    # With a valid triplet, an invalid downstream finger must remain fixed
    # while the rest of the branch rotates.
    validity[4] = True
    validity[6] = False
    finger_before = lm[6].copy()
    _, n_clamped = clamp_joint_angles(lm, limits={(0, 2, 4): (30, 60)}, validity=validity)
    assert n_clamped == 1
    np.testing.assert_array_equal(lm[6], finger_before)
