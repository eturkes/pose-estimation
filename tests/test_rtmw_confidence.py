"""Tests for confidence-weighted One Euro Filter."""

import numpy as np

from pose_estimation.run import KeypointSmoother, _OneEuro


def _make_kps(n=133):
    """Return a simple (n, 2) keypoint array."""
    rng = np.random.RandomState(7291)
    return rng.rand(n, 2) * 300


def test_no_confidence_unchanged():
    """Without confidence, filter behaves identically to before."""
    kp = _make_kps()
    f1 = _OneEuro(min_cutoff=0.5, beta=0.5)
    f2 = _OneEuro(min_cutoff=0.5, beta=0.5)

    for t in np.linspace(0, 1, 20):
        noisy = kp + np.random.RandomState(int(t * 1000)).randn(*kp.shape) * 5
        r1 = f1(noisy, t)
        r2 = f2(noisy, t, confidence=None)
    np.testing.assert_allclose(r1, r2, atol=1e-12)


def test_full_confidence_matches_standard():
    """Confidence of 1.0 everywhere should equal standard filtering."""
    kp = _make_kps()
    f_std = _OneEuro(min_cutoff=0.5, beta=0.5)
    f_conf = _OneEuro(min_cutoff=0.5, beta=0.5)
    conf = np.ones(kp.shape[0])

    for t in np.linspace(0, 1, 20):
        noisy = kp + np.random.RandomState(int(t * 1000)).randn(*kp.shape) * 5
        r_std = f_std(noisy, t)
        r_conf = f_conf(noisy, t, confidence=conf)
    np.testing.assert_allclose(r_conf, r_std, atol=1e-12)


def test_zero_confidence_stays_at_previous():
    """Confidence of 0.0 should pin output to the previous estimate."""
    kp = _make_kps()
    filt = _OneEuro(min_cutoff=0.5, beta=0.5)
    conf = np.zeros(kp.shape[0])

    r0 = filt(kp, 0.0, confidence=conf)
    np.testing.assert_allclose(r0, kp)

    r1 = filt(kp + 50.0, 0.1, confidence=conf)
    np.testing.assert_allclose(r1, r0, atol=1e-10)


def test_low_confidence_smoothed_more():
    """Low-confidence keypoints should deviate less from previous."""
    kp = _make_kps()
    f_high = _OneEuro(min_cutoff=0.5, beta=0.5)
    f_low = _OneEuro(min_cutoff=0.5, beta=0.5)

    conf_high = np.ones(kp.shape[0])
    conf_low = np.full(kp.shape[0], 0.3)

    f_high(kp, 0.0, confidence=conf_high)
    f_low(kp, 0.0, confidence=conf_low)

    shifted = kp + 40.0
    r_high = f_high(shifted, 0.1, confidence=conf_high)
    r_low = f_low(shifted, 0.1, confidence=conf_low)

    dist_high = np.linalg.norm(r_high - kp)
    dist_low = np.linalg.norm(r_low - kp)
    assert dist_low < dist_high


def test_gamma_controls_sharpness():
    """Higher gamma makes low confidence pull harder toward previous."""
    kp = _make_kps()
    f_low_g = _OneEuro(min_cutoff=0.5, beta=0.5, gamma=1.0)
    f_high_g = _OneEuro(min_cutoff=0.5, beta=0.5, gamma=4.0)
    conf = np.full(kp.shape[0], 0.5)

    f_low_g(kp, 0.0, confidence=conf)
    f_high_g(kp, 0.0, confidence=conf)

    shifted = kp + 40.0
    r_low_g = f_low_g(shifted, 0.1, confidence=conf)
    r_high_g = f_high_g(shifted, 0.1, confidence=conf)

    dist_low_g = np.linalg.norm(r_low_g - kp)
    dist_high_g = np.linalg.norm(r_high_g - kp)
    assert dist_high_g < dist_low_g


def test_confidence_clipped():
    """Out-of-range confidence values are clipped, not crash."""
    kp = _make_kps(10)
    filt = _OneEuro()
    conf = np.array([1.5, -0.5] + [0.8] * 8)

    filt(kp, 0.0, confidence=conf)
    result = filt(kp + 10.0, 0.1, confidence=conf)
    assert not np.any(np.isnan(result))


def test_smoother_passes_confidence():
    """KeypointSmoother should wire scores through to the filter.

    Two persons with different score levels: the low-score person's
    smoothed keypoints should resist movement more than the high-score
    person's.
    """
    smoother = KeypointSmoother(min_track_age=1)

    n_kps = 133
    rng = np.random.RandomState(5832)
    kp1 = rng.rand(n_kps, 2) * 300
    kp2 = rng.rand(n_kps, 2) * 300 + 600  # far apart

    sc_high = np.ones(n_kps)
    sc_low = np.full(n_kps, 0.1)

    keypoints = np.stack([kp1, kp2])
    scores = np.stack([sc_high, sc_low])

    # Initialise
    smoother(keypoints, scores, 0.0)

    # Shift both persons equally
    shifted_kps = keypoints + 40.0
    smooth_kps, _ = smoother(shifted_kps, scores, 0.1)

    move_high = np.linalg.norm(smooth_kps[0] - kp1)
    move_low = np.linalg.norm(smooth_kps[1] - kp2)
    assert move_low < move_high, (
        f"Low-score person should move less: move_low={move_low:.2f}, move_high={move_high:.2f}"
    )
