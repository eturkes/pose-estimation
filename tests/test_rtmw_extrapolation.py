"""Tests for velocity-based carry-forward extrapolation."""

import numpy as np

from pose_estimation.run import KeypointSmoother


def _make_kps(n=133, seed=3741):
    """Return a simple (n, 2) keypoint array."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2) * 300


def test_static_carry_no_drift():
    """Repeated static keypoints produce negligible carry-forward drift."""
    sm = KeypointSmoother()
    kp = _make_kps()
    sc = np.ones(133)

    for i in range(20):
        sm(kp[np.newaxis], sc[np.newaxis], i * 0.033)

    last = sm.tracks[0]["last_kps"].copy()

    # Drop detection for 3 frames
    for i in range(3):
        out_kps, _ = sm(np.empty((0, 133, 2)), np.empty((0, 133)), (20 + i) * 0.033)
        assert out_kps is not None, "should emit carry-forward"
        diff = np.max(np.abs(out_kps[0] - last))
        assert diff < 1.0, f"static carry drifted by {diff}"


def test_moving_carry_extrapolates():
    """Carry-forward with built-up velocity produces movement."""
    sm = KeypointSmoother()
    base = _make_kps()
    sc = np.ones(133)

    # Feed 20 frames of steady rightward motion
    for i in range(20):
        shifted = base + np.array([[i * 5.0, 0.0]])
        sm(shifted[np.newaxis], sc[np.newaxis], i * 0.033)

    last = sm.tracks[0]["last_kps"].copy()

    # Drop detection — carry-forward should extrapolate
    carry = []
    for i in range(4):
        out_kps, _ = sm(np.empty((0, 133, 2)), np.empty((0, 133)), (20 + i) * 0.033)
        assert out_kps is not None
        carry.append(out_kps[0].copy())

    # First carry should have moved right from last
    dx = carry[0][0, 0] - last[0, 0]
    assert dx > 0, f"expected rightward extrapolation, got dx={dx}"

    # Subsequent frames should continue advancing
    for j in range(1, len(carry)):
        assert carry[j][0, 0] > carry[j - 1][0, 0], \
            f"carry frame {j} didn't advance"


def test_damping_decelerates():
    """Step size should decrease over consecutive carry frames."""
    sm = KeypointSmoother()
    base = _make_kps()
    sc = np.ones(133)

    for i in range(20):
        shifted = base + np.array([[i * 5.0, 0.0]])
        sm(shifted[np.newaxis], sc[np.newaxis], i * 0.033)

    carry = []
    for i in range(5):
        out_kps, _ = sm(np.empty((0, 133, 2)), np.empty((0, 133)), (20 + i) * 0.033)
        carry.append(out_kps[0][0, 0])

    steps = [carry[j] - carry[j - 1] for j in range(1, len(carry))]
    for j in range(1, len(steps)):
        assert steps[j] < steps[j - 1], \
            f"step {j} ({steps[j]:.4f}) >= step {j-1} ({steps[j-1]:.4f})"


def test_extrapolation_capped():
    """Large velocity is capped at match_thresh per keypoint."""
    sm = KeypointSmoother(match_thresh=80, min_track_age=0)
    base = _make_kps()
    sc = np.ones(133)

    sm(base[np.newaxis], sc[np.newaxis], 0.0)
    # Huge jump to produce large velocity
    big_shift = base + 5000.0
    sm(big_shift[np.newaxis], sc[np.newaxis], 0.033)
    last = sm.tracks[0]["last_kps"].copy()

    out_kps, _ = sm(np.empty((0, 133, 2)), np.empty((0, 133)), 0.066)
    step = out_kps[0] - last
    max_norm = np.max(np.linalg.norm(step, axis=1))
    assert max_norm <= 80 + 1e-3, \
        f"per-keypoint step {max_norm:.1f} exceeds match_thresh 80"


def test_carry_damping_configurable():
    """Different carry_damping values produce different extrapolation."""
    base = _make_kps()
    sc = np.ones(133)

    def _run(damping):
        sm = KeypointSmoother(carry_damping=damping)
        for i in range(20):
            shifted = base + np.array([[i * 5.0, 0.0]])
            sm(shifted[np.newaxis], sc[np.newaxis], i * 0.033)
        positions = []
        for i in range(3):
            out_kps, _ = sm(np.empty((0, 133, 2)), np.empty((0, 133)), (20 + i) * 0.033)
            positions.append(out_kps[0][0, 0])
        return positions[-1] - positions[0]

    travel_fast = _run(0.5)   # fast decay
    travel_slow = _run(0.95)  # slow decay
    assert travel_slow > travel_fast, \
        f"slow damping ({travel_slow:.3f}) should travel further than fast ({travel_fast:.3f})"


def test_no_detection_carry_with_time():
    """_carry(t) path (no detections at all) also extrapolates."""
    sm = KeypointSmoother()
    base = _make_kps()
    sc = np.ones(133)

    for i in range(20):
        shifted = base + np.array([[i * 5.0, 0.0]])
        sm(shifted[np.newaxis], sc[np.newaxis], i * 0.033)

    last = sm.tracks[0]["last_kps"].copy()

    # Pass None keypoints (triggers _carry path)
    out_kps, _ = sm(None, None, 20 * 0.033)
    assert out_kps is not None
    dx = out_kps[0][0, 0] - last[0, 0]
    assert dx > 0, f"_carry path should extrapolate, got dx={dx}"


def test_carry_centroid_updates():
    """Carried track centroid should reflect extrapolated position."""
    sm = KeypointSmoother()
    base = _make_kps()
    sc = np.ones(133)

    for i in range(20):
        shifted = base + np.array([[i * 5.0, 0.0]])
        sm(shifted[np.newaxis], sc[np.newaxis], i * 0.033)

    old_centroid = sm.tracks[0]["centroid"].copy()

    sm(np.empty((0, 133, 2)), np.empty((0, 133)), 20 * 0.033)
    new_centroid = sm.tracks[0]["centroid"]

    # Centroid should have moved right
    assert new_centroid[0] > old_centroid[0], \
        "carried centroid should reflect extrapolated position"


def test_17kp_carry_extrapolates():
    """Extrapolation works for non-133 (body-only) keypoints too."""
    sm = KeypointSmoother()
    rng = np.random.RandomState(8124)
    base = rng.rand(17, 2) * 300
    sc = np.ones(17)

    for i in range(20):
        shifted = base + np.array([[i * 5.0, 0.0]])
        sm(shifted[np.newaxis], sc[np.newaxis], i * 0.033)

    last = sm.tracks[0]["last_kps"].copy()

    out_kps, _ = sm(np.empty((0, 17, 2)), np.empty((0, 17)), 20 * 0.033)
    assert out_kps is not None
    dx = out_kps[0][0, 0] - last[0, 0]
    assert dx > 0, f"17-kp carry should extrapolate, got dx={dx}"


if __name__ == "__main__":
    test_static_carry_no_drift()
    test_moving_carry_extrapolates()
    test_damping_decelerates()
    test_extrapolation_capped()
    test_carry_damping_configurable()
    test_no_detection_carry_with_time()
    test_carry_centroid_updates()
    test_17kp_carry_extrapolates()
    print("All RTMW extrapolation tests passed.")
