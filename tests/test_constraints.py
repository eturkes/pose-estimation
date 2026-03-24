"""Tests for biomechanical bone-length constraints."""

import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from constraints import BoneLengthSmoother, BONE_SEGMENTS


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
        result = smoother.update(0, lm.copy())

    np.testing.assert_allclose(result, original, atol=1e-10)


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

    result = smoother.update(0, perturbed)

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
        smoother.update(0, lm.copy())

    avg = smoother._averages[0]
    # After 25 frames at alpha=0.05, EMA should be within ~30% of true
    # (1 - 0.95^25 ≈ 0.72 of the way there from any starting point,
    # but we start at the first observation so convergence is immediate).
    np.testing.assert_allclose(avg, true_lengths, rtol=0.01)


def test_prune_removes_stale():
    """prune() should drop state for IDs not in the active set."""
    smoother = BoneLengthSmoother()
    lm = _make_landmarks()
    smoother.update(0, lm.copy())
    smoother.update(1, lm.copy())
    smoother.update(2, lm.copy())

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

    result = smoother.update(0, perturbed)

    # Should NOT be corrected — within 40% tolerance
    np.testing.assert_allclose(result[4], perturbed[4], atol=1e-10)


if __name__ == "__main__":
    test_constant_landmarks_no_correction()
    test_perturbed_keypoint_corrected()
    test_ema_converges()
    test_prune_removes_stale()
    test_small_movements_within_tolerance_pass_through()
    print("All tests passed.")
