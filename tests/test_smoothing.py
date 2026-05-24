"""Tests for confidence-weighted temporal smoothing."""

import numpy as np

from pose_estimation.smoothing import OneEuroFilter, PoseSmoother


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

    for a, b in zip(results_no_conf, results_none, strict=False):
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
        f"Low-conf kps should move less: move_low={move_low:.2f}, move_high={move_high:.2f}"
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
    result, _, _ = smoother.smooth_bodies([shifted1, shifted2], [vis1, vis2], 0.1)

    move1 = np.linalg.norm(result[0] - lm1)
    move2 = np.linalg.norm(result[1] - lm2)
    assert move2 < move1, f"Low-vis body should move less: move1={move1:.2f}, move2={move2:.2f}"


def test_hand_path_unaffected():
    """smooth_hands should work without confidence (no regression)."""
    smoother = PoseSmoother()
    lm = [_make_landmarks(21)]

    r1, _n1 = smoother.smooth_hands(lm, 0.0)
    assert len(r1) == 1

    r2, _n2 = smoother.smooth_hands([lm[0] + 5.0], 0.1)
    assert len(r2) == 1
    # Should be smoothed, not identical to input
    assert not np.allclose(r2[0], lm[0] + 5.0)


def test_hand_confidence_smoothing():
    """Low hand_flag values should produce less movement than high ones."""
    smoother_hi = PoseSmoother()
    smoother_lo = PoseSmoother()
    lm = [_make_landmarks(21)]

    smoother_hi.smooth_hands(lm, 0.0, hand_flags=[1.0])
    smoother_lo.smooth_hands(lm, 0.0, hand_flags=[0.2])

    shifted = [lm[0] + 30.0]
    r_hi, _ = smoother_hi.smooth_hands(shifted, 0.1, hand_flags=[1.0])
    r_lo, _ = smoother_lo.smooth_hands(shifted, 0.1, hand_flags=[0.2])

    move_hi = np.linalg.norm(r_hi[0] - lm[0])
    move_lo = np.linalg.norm(r_lo[0] - lm[0])
    assert move_lo < move_hi, (
        f"Low hand_flag should move less: move_lo={move_lo:.2f}, move_hi={move_hi:.2f}"
    )


# ---- Outlier rejection --------------------------------------------------------


def test_outlier_cap_suppresses_spike():
    """A single-frame spike beyond outlier_cap should be clamped."""
    lm = _make_landmarks(12)[:, :2]  # (12, 2) for 2D keypoints
    filt = OneEuroFilter(min_cutoff=0.3, beta=0.5, outlier_cap=20.0)

    filt(lm, 0.0)
    filt(lm + 1.0, 1 / 30)  # small movement to establish velocity

    # Spike: one keypoint jumps 100 pixels in one frame
    spiked = lm + 2.0
    spiked[5] += 100.0
    result = filt(spiked, 2 / 30)

    # Keypoint 5 should be clamped well below the 100px spike
    displacement_5 = np.linalg.norm(result[5] - lm[5])
    assert displacement_5 < 50.0, f"Spike was not suppressed: {displacement_5:.1f}px"


def test_outlier_cap_zero_disables():
    """outlier_cap=0 should produce identical results to no cap."""
    lm = _make_landmarks(12)[:, :2]
    f_off = OneEuroFilter(min_cutoff=0.3, beta=0.5, outlier_cap=0.0)
    f_default = OneEuroFilter(min_cutoff=0.3, beta=0.5)

    for t in np.linspace(0, 1, 20):
        noisy = lm + np.random.RandomState(int(t * 1000)).randn(*lm.shape) * 10
        r_off = f_off(noisy, t)
        r_default = f_default(noisy, t)
        np.testing.assert_allclose(r_off, r_default, atol=1e-12)


def test_outlier_cap_allows_predicted_movement():
    """Movement consistent with velocity should pass through unclamped."""
    lm = np.zeros((5, 2))
    filt = OneEuroFilter(min_cutoff=0.3, beta=0.5, outlier_cap=10.0)
    dt = 1 / 30

    filt(lm, 0.0)
    # Establish a strong rightward velocity (50 px/s → ~1.67 px/frame)
    for i in range(1, 10):
        filt(lm + np.array([i * 1.67, 0.0]), i * dt)

    # Large step consistent with built-up velocity: should NOT be clamped
    consistent_step = lm + np.array([10 * 1.67 + 2.0, 0.0])
    result = filt(consistent_step, 10 * dt)
    # Result should track near the input (within filter smoothing)
    displacement = np.abs(result[:, 0] - consistent_step[:, 0])
    assert np.all(displacement < 10.0), f"Predicted movement was wrongly clamped: {displacement}"


# ---- Adaptive smoothing (movement-phase-aware min_cutoff) --------------------


def test_adaptive_rest_smooths_more_than_fixed():
    """During stationary input, adaptive mode should smooth more than fixed."""
    lm = _make_landmarks(12)[:, :2]
    dt = 1 / 30
    f_fixed = OneEuroFilter(min_cutoff=0.3, beta=0.5)
    f_adapt = OneEuroFilter(min_cutoff=0.3, beta=0.5, rest_cutoff=0.05)

    # Warm up both filters with stationary data so smoothed_speed settles
    for i in range(30):
        noise = np.random.RandomState(i).randn(*lm.shape) * 2
        f_fixed(lm + noise, i * dt)
        f_adapt(lm + noise, i * dt)

    # Apply a small perturbation — adaptive (at rest) should resist more
    bump = lm + 8.0
    r_fixed = f_fixed(bump, 30 * dt)
    r_adapt = f_adapt(bump, 30 * dt)

    move_fixed = np.linalg.norm(r_fixed - lm)
    move_adapt = np.linalg.norm(r_adapt - lm)
    assert move_adapt < move_fixed, (
        f"Adaptive should resist more during rest: adapt={move_adapt:.2f}, fixed={move_fixed:.2f}"
    )


def test_adaptive_fast_movement_matches_fixed():
    """During sustained fast movement, adaptive cutoff should equal min_cutoff."""
    lm = np.zeros((8, 2))
    dt = 1 / 30
    f_fixed = OneEuroFilter(min_cutoff=0.3, beta=0.5)
    f_adapt = OneEuroFilter(
        min_cutoff=0.3, beta=0.5, rest_cutoff=0.05, rest_speed=2.0, fast_speed=10.0
    )

    # Sustained fast movement (20 px/frame) for many frames
    for i in range(60):
        pos = lm + np.array([i * 20.0, 0.0])
        f_fixed(pos, i * dt)
        f_adapt(pos, i * dt)

    # After sustained fast movement, both should produce nearly identical output
    final_pos = lm + np.array([60 * 20.0, 0.0])
    r_fixed = f_fixed(final_pos, 60 * dt)
    r_adapt = f_adapt(final_pos, 60 * dt)

    np.testing.assert_allclose(r_adapt, r_fixed, atol=0.5)


def test_adaptive_rest_to_movement_recovers():
    """Transition from rest to fast movement: lag should clear within frames."""
    lm = np.zeros((6, 2))
    dt = 1 / 30
    filt = OneEuroFilter(
        min_cutoff=0.3,
        beta=0.5,
        rest_cutoff=0.05,
        rest_speed=2.0,
        fast_speed=10.0,
        speed_alpha=0.15,
    )

    # 30 frames at rest
    for i in range(30):
        filt(lm + np.random.RandomState(i).randn(*lm.shape) * 0.5, i * dt)

    # Start fast movement (25 px/frame rightward)
    lag_frames = 0
    for i in range(30, 50):
        frame = i - 30
        pos = lm + np.array([(frame + 1) * 25.0, 0.0])
        result = filt(pos, i * dt)
        tracking_error = np.abs(result[:, 0] - pos[:, 0]).mean()
        if tracking_error > 50.0:
            lag_frames += 1

    # Adaptive mode should recover from rest-level smoothing quickly:
    # at most a few frames of elevated lag during transition
    assert lag_frames <= 5, f"Too many high-lag frames during transition: {lag_frames}"


def test_adaptive_disabled_with_none():
    """rest_cutoff=None should produce identical results to no adaptive args."""
    lm = _make_landmarks(12)[:, :2]
    f_plain = OneEuroFilter(min_cutoff=0.3, beta=0.5)
    f_none = OneEuroFilter(min_cutoff=0.3, beta=0.5, rest_cutoff=None)

    for t in np.linspace(0, 1, 30):
        noisy = lm + np.random.RandomState(int(t * 1000)).randn(*lm.shape) * 5
        r_plain = f_plain(noisy, t)
        r_none = f_none(noisy, t)
        np.testing.assert_allclose(r_none, r_plain, atol=1e-12)


def test_adaptive_per_keypoint_independence():
    """Keypoints at different speeds should get different effective cutoffs."""
    dt = 1 / 30
    filt = OneEuroFilter(
        min_cutoff=0.3,
        beta=0.0,
        rest_cutoff=0.05,
        rest_speed=2.0,
        fast_speed=10.0,
    )

    # Keypoint 0 stays still, keypoint 1 moves fast
    frames = 40
    for i in range(frames):
        pos = np.array([[0.0, 0.0], [i * 20.0, 0.0]])
        filt(pos, i * dt)

    # After warmup, check smoothed_speed: kp0 ≈ rest, kp1 ≈ fast
    speed = filt._smoothed_speed
    assert speed is not None
    assert speed[0] < 3.0, f"Stationary kp speed too high: {speed[0]:.2f}"
    assert speed[1] > 5.0, f"Moving kp speed too low: {speed[1]:.2f}"


def test_pose_smoother_adaptive_default():
    """PoseSmoother with default env vars should create adaptive filters."""
    smoother = PoseSmoother()
    lm = [_make_landmarks(12)]
    vis = [np.ones(12)]

    # Initialize tracks
    smoother.smooth_bodies(lm, vis, 0.0)

    # The body filter should have rest_cutoff set
    filt = smoother.body_tracks[0][0]
    assert filt.rest_cutoff is not None
    assert filt.rest_cutoff < filt.min_cutoff
