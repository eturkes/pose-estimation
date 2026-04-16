"""Tests for velocity-based extrapolation during carry-forward."""

import numpy as np

from pose_estimation.smoothing import PoseSmoother, OneEuroFilter


def make_body(x_offset=0.0, y_offset=0.0):
    """Create a synthetic body landmark array (12, 3)."""
    base = np.zeros((12, 3))
    for i in range(12):
        base[i] = [100.0 + i * 10 + x_offset, 200.0 + i * 5 + y_offset, 0.5]
    return base


def test_static_landmarks_no_extrapolation():
    """Repeated static landmarks produce no carry-forward drift."""
    smoother = PoseSmoother()
    lm = make_body()
    vis = np.ones(12)

    # Feed 20 frames of identical landmarks
    for i in range(20):
        t = i * 0.033
        smoothed, _, _ = smoother.smooth_bodies([lm.copy()], [vis], t)

    last_smoothed = smoothed[0].copy()

    # Now drop detection for 5 frames (carry-forward)
    for i in range(5):
        t = (20 + i) * 0.033
        smoothed, _, n_det = smoother.smooth_bodies([], [], t)
        assert n_det == 0, "should have no real detections"
        assert len(smoothed) == 1, "should emit carry-forward"
        # With zero velocity, extrapolation should be negligible
        diff = np.max(np.abs(smoothed[0] - last_smoothed))
        assert diff < 1e-3, f"static carry drifted by {diff}"
    print("PASS: static_landmarks_no_extrapolation")


def test_moving_landmarks_extrapolate():
    """Carry-forward with velocity produces movement, not static replay."""
    smoother = PoseSmoother()
    vis = np.ones(12)

    # Feed 20 frames of steadily moving landmarks
    for i in range(20):
        lm = make_body(x_offset=i * 5.0)  # 5 px/frame rightward
        t = i * 0.033
        smoothed, _, _ = smoother.smooth_bodies([lm.copy()], [vis], t)

    last_real = smoothed[0].copy()

    # Now drop detection — carry-forward should extrapolate
    carry_outputs = []
    for i in range(5):
        t = (20 + i) * 0.033
        smoothed, _, n_det = smoother.smooth_bodies([], [], t)
        assert n_det == 0
        assert len(smoothed) == 1
        carry_outputs.append(smoothed[0].copy())

    # First carry-forward should have moved rightward from last_real
    dx = carry_outputs[0][0, 0] - last_real[0, 0]
    assert dx > 0, f"expected rightward extrapolation, got dx={dx}"

    # Each subsequent carry frame should move further right (but slower)
    for j in range(1, len(carry_outputs)):
        assert carry_outputs[j][0, 0] > carry_outputs[j - 1][0, 0], \
            f"carry frame {j} didn't advance"

    # Damping: step size should decrease over consecutive carry frames
    steps = [carry_outputs[j][0, 0] - carry_outputs[j - 1][0, 0]
             for j in range(1, len(carry_outputs))]
    for j in range(1, len(steps)):
        assert steps[j] < steps[j - 1], \
            f"step {j} ({steps[j]:.3f}) >= step {j-1} ({steps[j-1]:.3f})"
    print("PASS: moving_landmarks_extrapolate")


def test_extrapolation_capped():
    """Extremely large velocity is capped at match_threshold."""
    smoother = PoseSmoother(match_threshold=100)
    vis = np.ones(12)

    # Create two frames with a huge jump to produce large velocity
    lm1 = make_body(x_offset=0)
    lm2 = make_body(x_offset=5000)  # enormous jump

    smoother.smooth_bodies([lm1.copy()], [vis], 0.0)
    smoothed, _, _ = smoother.smooth_bodies([lm2.copy()], [vis], 0.033)
    last_real = smoothed[0].copy()

    # Carry-forward
    smoothed, _, _ = smoother.smooth_bodies([], [], 0.066)
    step = smoothed[0] - last_real
    max_norm = np.max(np.linalg.norm(step, axis=1))
    assert max_norm <= 100 + 1e-3, \
        f"per-keypoint step {max_norm:.1f} exceeds match_threshold 100"
    print("PASS: extrapolation_capped")


def test_hand_tracks_unaffected():
    """Hand tracking (emit_carry=False) still works with 7-tuple tracks."""
    smoother = PoseSmoother()
    hand = np.random.RandomState(42).rand(21, 3) * 100

    for i in range(10):
        t = i * 0.033
        smoothed, n_active = smoother.smooth_hands([hand.copy()], t)
        assert len(smoothed) == 1

    # Drop hand — static carry emits the last output for one grace frame
    smoothed, n_active = smoother.smooth_hands([], 10 * 0.033)
    assert len(smoothed) == 1, "static carry should emit one hand"
    assert n_active == 0, "no real detection matched"

    # Re-acquire near original position — should match existing track
    smoothed, n_active = smoother.smooth_hands([hand.copy()], 11 * 0.033)
    assert len(smoothed) == 1

    ages = smoother.hand_track_ages()
    assert len(ages) == 1
    assert ages[0] > 1, "should have re-matched existing track"
    print("PASS: hand_tracks_unaffected")


def test_damping_converges_to_static():
    """After many misses the extrapolation effectively stops."""
    smoother = PoseSmoother()
    vis = np.ones(12)

    # Build velocity
    for i in range(20):
        lm = make_body(x_offset=i * 5.0)
        smoother.smooth_bodies([lm.copy()], [vis], i * 0.033)

    # Carry for 10 frames (full grace period)
    positions = []
    for i in range(10):
        t = (20 + i) * 0.033
        smoothed, _, _ = smoother.smooth_bodies([], [], t)
        if smoothed:
            positions.append(smoothed[0][0, 0])

    # Last few steps should be very small (damping ≈ 0.8^10 ≈ 0.107)
    if len(positions) >= 3:
        final_step = abs(positions[-1] - positions[-2])
        first_step = abs(positions[1] - positions[0])
        ratio = final_step / max(first_step, 1e-9)
        assert ratio < 0.5, \
            f"damping ratio {ratio:.3f} — expected significant decay"
    print("PASS: damping_converges_to_static")


def test_damping_factor_configurable():
    """Different carry_damping values produce different extrapolation distances."""
    s_fast = PoseSmoother(carry_damping=0.5)
    s_slow = PoseSmoother(carry_damping=0.95)
    vis = np.ones(12)

    # Build velocity with identical motion for both smoothers
    for i in range(20):
        lm = make_body(x_offset=i * 5.0)
        t = i * 0.033
        s_fast.smooth_bodies([lm.copy()], [vis], t)
        s_slow.smooth_bodies([lm.copy()], [vis], t)

    # Carry-forward for 3 frames
    positions_fast = []
    positions_slow = []
    for i in range(3):
        t = (20 + i) * 0.033
        sm_fast, _, _ = s_fast.smooth_bodies([], [], t)
        sm_slow, _, _ = s_slow.smooth_bodies([], [], t)
        positions_fast.append(sm_fast[0][0, 0])
        positions_slow.append(sm_slow[0][0, 0])

    # Slow damping (0.95) should have moved further than fast damping (0.5)
    total_fast = positions_fast[-1] - positions_fast[0]
    total_slow = positions_slow[-1] - positions_slow[0]
    assert total_slow > total_fast, \
        f"slow damping ({total_slow:.3f}) should move further than fast ({total_fast:.3f})"
    print("PASS: damping_factor_configurable")


if __name__ == "__main__":
    test_static_landmarks_no_extrapolation()
    test_moving_landmarks_extrapolate()
    test_extrapolation_capped()
    test_hand_tracks_unaffected()
    test_damping_converges_to_static()
    test_damping_factor_configurable()
    print("\nAll extrapolation tests passed.")
