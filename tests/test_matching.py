"""Tests for Hungarian hand-to-arm matching."""

import numpy as np

from pose_estimation.processing import match_hands_to_arms


def _make_body():
    """Return a plausible (12, 3) arm landmark array.

    Layout (approximate pixel positions):
        0  left shoulder   (100, 100)
        1  right shoulder  (200, 100)
        2  left elbow      (80,  200)
        3  right elbow     (220, 200)
        4  left wrist      (60,  300)
        5  right wrist     (240, 300)
        6-11 finger bases  (not used by matching)
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


def _make_hand(wrist_x, wrist_y):
    """Return a (21, 3) hand landmark array with the given wrist position."""
    lm = np.zeros((21, 3), dtype=np.float64)
    lm[0] = [wrist_x, wrist_y, 0]
    for i in range(1, 21):
        lm[i] = [wrist_x + i * 2, wrist_y + i, 0]
    return lm


def test_no_bodies_empty():
    """No bodies → empty result."""
    hand = _make_hand(60, 300)
    result = match_hands_to_arms([], [hand])
    assert result == []


def test_no_hands_empty():
    """No hands → empty result."""
    body = _make_body()
    result = match_hands_to_arms([body], [])
    assert result == []


def test_single_body_two_hands():
    """Single body, two hands near the wrists → both matched."""
    body = _make_body()
    hand_left = _make_hand(60, 305)    # near left wrist (60, 300)
    hand_right = _make_hand(240, 305)  # near right wrist (240, 300)

    result = match_hands_to_arms([body], [hand_left, hand_right])
    assert len(result) == 2

    matched_wrist_kps = {wrist_kp for _, wrist_kp, _ in result}
    matched_hand_idxs = {hand_idx for _, _, hand_idx in result}
    assert matched_wrist_kps == {4, 5}
    assert matched_hand_idxs == {0, 1}

    # Verify correct pairing: hand 0 (left) → wrist 4, hand 1 (right) → wrist 5
    for arm_idx, wrist_kp, hand_idx in result:
        if wrist_kp == 4:
            assert hand_idx == 0
        elif wrist_kp == 5:
            assert hand_idx == 1


def test_optimal_not_greedy():
    """Hungarian assignment avoids the suboptimal greedy pairing.

    Place hand A near left wrist and hand B near right wrist.
    A greedy algorithm processing the left wrist first could steal
    the wrong hand if positions were adversarial, but with clean
    placement the optimal assignment should always get it right
    regardless of processing order.
    """
    body = _make_body()
    hand_a = _make_hand(60, 300)   # near left wrist (60, 300)
    hand_b = _make_hand(240, 300)  # near right wrist (240, 300)

    # Regardless of input order, optimal assignment should pair correctly
    for hands in [[hand_a, hand_b], [hand_b, hand_a]]:
        result = match_hands_to_arms([body], hands)
        assert len(result) == 2
        for arm_idx, wrist_kp, hand_idx in result:
            hand_wrist = hands[hand_idx][0, :2]
            arm_wrist = body[wrist_kp, :2]
            assert np.linalg.norm(hand_wrist - arm_wrist) < 5


def test_hand_beyond_threshold():
    """A hand too far from any wrist is not matched."""
    body = _make_body()
    far_hand = _make_hand(900, 900)

    result = match_hands_to_arms([body], [far_hand], threshold=100)
    assert result == []


def test_distality_check():
    """A hand near the shoulder midpoint is rejected even if within threshold.

    Shoulder midpoint is at (150, 100).  Place a hand at (150, 110),
    which is close to the shoulder midpoint but far from the wrists.
    With a generous threshold, the distance check would pass, but the
    distality check should reject it.
    """
    body = _make_body()
    # Hand at (150, 110): distance to shoulder midpoint (150, 100) = 10
    # Distance to left wrist (60, 300) ≈ 212, right wrist (240, 300) ≈ 210
    # Both wrist distances are greater than shoulder-midpoint distance → reject
    hand = _make_hand(150, 110)

    result = match_hands_to_arms([body], [hand], threshold=300)
    assert result == [], "hand near shoulder midpoint should be rejected"


def test_cross_body_nearest_wrist():
    """Two bodies, one hand between them → matched to the nearest wrist."""
    body1 = _make_body()
    body2 = _make_body()
    # Shift body2 to the right by 400px
    body2[:, 0] += 400

    # Place hand near body1's right wrist (240, 300)
    hand = _make_hand(245, 300)

    result = match_hands_to_arms([body1, body2], [hand])
    assert len(result) == 1

    arm_idx, wrist_kp, hand_idx = result[0]
    assert arm_idx == 0, "should match to body 0 (closer)"
    assert wrist_kp == 5, "should match to right wrist"
    assert hand_idx == 0


if __name__ == "__main__":
    test_no_bodies_empty()
    test_no_hands_empty()
    test_single_body_two_hands()
    test_optimal_not_greedy()
    test_hand_beyond_threshold()
    test_distality_check()
    test_cross_body_nearest_wrist()
    print("All matching tests passed.")
