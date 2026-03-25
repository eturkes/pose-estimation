"""Tests for confidence-weighted temporal smoothing."""

import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from smoothing import OneEuroFilter, PoseSmoother


def _make_landmarks(n=12):
    """Return a simple (n, 3) landmark array."""
    rng = np.random.RandomState(4817)
    return rng.rand(n, 3) * 300


def test_no_confidence_unchanged_behaviour():
    """Without confidence, the filter should behave identically to before."""
    lm = _make_landmarks()
    f1 = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    f2 = OneEuroFilter(min_cutoff=1.0, beta=0.5)

    results_no_conf = []
    results_none = []
    for t in np.linspace(0, 1, 20):
        noisy = lm + np.random.RandomState(int(t * 1000)).randn(*lm.shape) * 5
        results_no_conf.append(f1(noisy, t))
        results_none.append(f2(noisy, t, confidence=None))

    for a, b in zip(results_no_conf, results_none):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_full_confidence_matches_standard():
    """Confidence of 1.0 for all keypoints should equal standard filtering."""
    lm = _make_landmarks()
    f_std = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    f_conf = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    conf = np.ones(lm.shape[0])

    for t in np.linspace(0, 1, 20):
        noisy = lm + np.random.RandomState(int(t * 1000)).randn(*lm.shape) * 5
        r_std = f_std(noisy, t)
        r_conf = f_conf(noisy, t, confidence=conf)
        np.testing.assert_allclose(r_conf, r_std, atol=1e-12)


def test_zero_confidence_stays_at_previous():
    """Confidence of 0.0 should keep the keypoint at the previous value."""
    lm = _make_landmarks()
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    conf = np.zeros(lm.shape[0])

    # First frame initialises
    r0 = filt(lm, 0.0, confidence=conf)
    np.testing.assert_allclose(r0, lm)

    # Second frame with zero confidence: output should stay at r0
    noisy = lm + 50.0
    r1 = filt(noisy, 0.1, confidence=conf)
    np.testing.assert_allclose(r1, r0, atol=1e-10)

    # Third frame: still pinned
    r2 = filt(noisy + 20.0, 0.2, confidence=conf)
    np.testing.assert_allclose(r2, r0, atol=1e-10)


def test_low_confidence_smoothed_more():
    """Low-confidence keypoints should deviate less from the previous value."""
    lm = _make_landmarks()
    f_high = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    f_low = OneEuroFilter(min_cutoff=1.0, beta=0.5)

    conf_high = np.ones(lm.shape[0])
    conf_low = np.full(lm.shape[0], 0.3)

    # Initialise both with the same frame
    f_high(lm, 0.0, confidence=conf_high)
    f_low(lm, 0.0, confidence=conf_low)

    # Feed a significantly shifted frame
    shifted = lm + 40.0
    r_high = f_high(shifted, 0.1, confidence=conf_high)
    r_low = f_low(shifted, 0.1, confidence=conf_low)

    # Low-confidence result should be closer to the initial landmarks
    dist_high = np.linalg.norm(r_high - lm)
    dist_low = np.linalg.norm(r_low - lm)
    assert dist_low < dist_high, (
        f"Low-conf should stay closer to previous: "
        f"dist_low={dist_low:.2f}, dist_high={dist_high:.2f}"
    )


def test_mixed_confidence():
    """Per-keypoint confidence: high-conf keypoints move more than low-conf."""
    lm = _make_landmarks(4)
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    # Keypoints 0,1 high confidence; 2,3 low
    conf = np.array([1.0, 1.0, 0.1, 0.1])

    filt(lm, 0.0, confidence=conf)
    shifted = lm + 50.0
    result = filt(shifted, 0.1, confidence=conf)

    # High-confidence keypoints (0,1) should have moved further from lm
    move_high = np.mean(np.linalg.norm(result[:2] - lm[:2], axis=1))
    move_low = np.mean(np.linalg.norm(result[2:] - lm[2:], axis=1))
    assert move_low < move_high, (
        f"Low-conf kps should move less: "
        f"move_low={move_low:.2f}, move_high={move_high:.2f}"
    )


def test_gamma_controls_sharpness():
    """Higher gamma should make low confidence pull harder toward previous."""
    lm = _make_landmarks()
    f_low_gamma = OneEuroFilter(min_cutoff=1.0, beta=0.5, gamma=1.0)
    f_high_gamma = OneEuroFilter(min_cutoff=1.0, beta=0.5, gamma=4.0)
    conf = np.full(lm.shape[0], 0.5)

    f_low_gamma(lm, 0.0, confidence=conf)
    f_high_gamma(lm, 0.0, confidence=conf)

    shifted = lm + 40.0
    r_low_g = f_low_gamma(shifted, 0.1, confidence=conf)
    r_high_g = f_high_gamma(shifted, 0.1, confidence=conf)

    # Higher gamma → weight = 0.5^4 = 0.0625 vs 0.5^1 = 0.5
    # So high gamma output should be closer to lm (the initial value)
    dist_low_g = np.linalg.norm(r_low_g - lm)
    dist_high_g = np.linalg.norm(r_high_g - lm)
    assert dist_high_g < dist_low_g


def test_confidence_clipped():
    """Confidence values outside [0, 1] should be clipped, not crash."""
    lm = _make_landmarks()
    filt = OneEuroFilter()
    conf = np.array([1.5, -0.5] + [0.8] * (lm.shape[0] - 2))

    filt(lm, 0.0, confidence=conf)
    result = filt(lm + 10.0, 0.1, confidence=conf)
    assert not np.any(np.isnan(result))


def test_smooth_bodies_passes_confidence():
    """PoseSmoother.smooth_bodies should use visibilities for confidence."""
    smoother = PoseSmoother()

    # Two bodies
    lm1 = _make_landmarks(12)
    lm2 = _make_landmarks(12) + 500  # far apart so they get separate tracks
    vis1 = np.ones(12)
    vis2 = np.full(12, 0.1)

    # First frame — initialise
    smoother.smooth_bodies([lm1, lm2], [vis1, vis2], 0.0)

    # Second frame with shift — low-vis body should resist the shift more
    shifted1 = lm1 + 30.0
    shifted2 = lm2 + 30.0
    result, _, _ = smoother.smooth_bodies(
        [shifted1, shifted2], [vis1, vis2], 0.1)

    move1 = np.linalg.norm(result[0] - lm1)
    move2 = np.linalg.norm(result[1] - lm2)
    assert move2 < move1, (
        f"Low-vis body should move less: move1={move1:.2f}, move2={move2:.2f}"
    )


def test_hand_path_unaffected():
    """smooth_hands should work without confidence (no regression)."""
    smoother = PoseSmoother()
    lm = [_make_landmarks(21)]

    r1, n1 = smoother.smooth_hands(lm, 0.0)
    assert len(r1) == 1

    r2, n2 = smoother.smooth_hands([lm[0] + 5.0], 0.1)
    assert len(r2) == 1
    # Should be smoothed, not identical to input
    assert not np.allclose(r2[0], lm[0] + 5.0)


if __name__ == "__main__":
    test_no_confidence_unchanged_behaviour()
    test_full_confidence_matches_standard()
    test_zero_confidence_stays_at_previous()
    test_low_confidence_smoothed_more()
    test_mixed_confidence()
    test_gamma_controls_sharpness()
    test_confidence_clipped()
    test_smooth_bodies_passes_confidence()
    test_hand_path_unaffected()
    print("All tests passed.")
