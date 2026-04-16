"""Tests for processing.py helper functions.

Covers synthetic hand generation, landmark re-crop, and affine matrix
degenerate-input handling.
"""

import numpy as np

from pose_estimation.processing import (
    _synthesise_hand_detections,
    _recrop_from_landmarks,
    _affine_matrix,
    _ARM_CHAINS_12,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_body(shoulder_l=(100, 200), elbow_l=(100, 300),
               wrist_l=(100, 400), shoulder_r=(300, 200),
               elbow_r=(300, 300), wrist_r=(300, 400)):
    """Create a (12, 3) arm landmark array with known geometry.

    12-keypoint arm scheme index mapping:
      0=left shoulder, 1=right shoulder,
      2=left elbow,    3=right elbow,
      4=left wrist,    5=right wrist,
      6-11=finger bases (unused here).
    """
    lm = np.zeros((12, 3), dtype=np.float32)
    lm[0, :2] = shoulder_l
    lm[1, :2] = shoulder_r
    lm[2, :2] = elbow_l
    lm[3, :2] = elbow_r
    lm[4, :2] = wrist_l
    lm[5, :2] = wrist_r
    return lm


def _make_vis(n=12, value=0.9):
    return np.full(n, value, dtype=np.float32)


def _make_hand_landmarks(wrist_px, mcp_px):
    """Create a (21, 3) hand landmark array in pixel coordinates.

    kp[0] = wrist, kp[9] = middle MCP.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0, :2] = wrist_px
    lm[9, :2] = mcp_px
    # Fill remaining with midpoint for realism
    mid = (np.array(wrist_px) + np.array(mcp_px)) / 2
    for i in range(21):
        if i not in (0, 9):
            lm[i, :2] = mid
    return lm


def _make_palm_det(cx_norm, cy_norm, size=0.1, score=0.9):
    """Create a minimal palm detection dict in normalised coordinates."""
    half = size / 2
    return {
        "box": np.array([cx_norm - half, cy_norm - half,
                         cx_norm + half, cy_norm + half], dtype=np.float32),
        "keypoints": np.array([[cx_norm, cy_norm]] * 7, dtype=np.float32),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Synthetic hand detection tests
# ---------------------------------------------------------------------------

def test_synthesise_hand_from_arm():
    """Synthetic detection box centre ≈ 40% of forearm beyond wrist,
    box size ≈ 80% of forearm length."""
    body = _make_body(
        elbow_l=(100, 300), wrist_l=(100, 400),
        elbow_r=(300, 300), wrist_r=(300, 400),
    )
    vis = _make_vis()
    frame_h, frame_w = 640, 480

    result = _synthesise_hand_detections(
        [body], [vis], [], frame_h, frame_w,
        arm_chains=_ARM_CHAINS_12,
    )

    assert len(result) == 2, f"Expected 2 synthetic dets, got {len(result)}"

    for det in result:
        assert det.get("synthetic") is True

    # Check the left arm detection (forearm is vertical, length 100 px)
    left_det = result[0]
    forearm_len = 100.0
    box = left_det["box"]
    box_px = box * np.array([frame_w, frame_h, frame_w, frame_h])
    box_centre_x = (box_px[0] + box_px[2]) / 2
    box_centre_y = (box_px[1] + box_px[3]) / 2
    box_w = box_px[2] - box_px[0]
    box_h = box_px[3] - box_px[1]

    # Centre should be ~40 px beyond wrist (wrist is at y=400, forearm points down)
    expected_centre_y = 400 + forearm_len * 0.4
    assert abs(box_centre_y - expected_centre_y) < 2.0, (
        f"Centre Y {box_centre_y} != expected {expected_centre_y}")

    # Box size should be ~80% of forearm length = 80 px (square)
    expected_size = forearm_len * 0.8
    assert abs(box_w - expected_size) < 2.0
    assert abs(box_h - expected_size) < 2.0


def test_synthesise_skips_covered_wrist():
    """No synthetic generated when a real palm detection covers the wrist."""
    body = _make_body(wrist_l=(100, 400), wrist_r=(300, 400))
    vis = _make_vis()
    frame_h, frame_w = 640, 480

    # Place a real palm detection right at the left wrist (normalised)
    left_wrist_norm_x = 100.0 / frame_w
    left_wrist_norm_y = 400.0 / frame_h
    palm_det = _make_palm_det(left_wrist_norm_x, left_wrist_norm_y, size=0.05)

    result = _synthesise_hand_detections(
        [body], [vis], [palm_det], frame_h, frame_w,
        arm_chains=_ARM_CHAINS_12,
    )

    # Left wrist is covered → only the right arm generates a synthetic
    assert len(result) == 1
    assert result[0].get("synthetic") is True


# ---------------------------------------------------------------------------
# Re-crop from landmarks tests
# ---------------------------------------------------------------------------

def test_recrop_from_landmarks():
    """Re-crop detection returned with correct centre and size."""
    frame_h, frame_w = 640, 480
    wrist_px = np.array([200, 300])
    mcp_px = np.array([200, 240])     # 60 px palm length, above wrist
    hand_lm = _make_hand_landmarks(wrist_px, mcp_px)

    result = _recrop_from_landmarks([hand_lm], [], frame_h, frame_w)

    assert len(result) == 1
    det = result[0]
    assert det.get("recrop") is True

    # Centre should be midpoint of wrist and MCP in normalised coords
    expected_cx = ((wrist_px[0] + mcp_px[0]) / 2) / frame_w
    expected_cy = ((wrist_px[1] + mcp_px[1]) / 2) / frame_h
    box = det["box"]
    det_cx = (box[0] + box[2]) / 2
    det_cy = (box[1] + box[3]) / 2
    assert abs(det_cx - expected_cx) < 0.01
    assert abs(det_cy - expected_cy) < 0.01

    # Box size should be 2× palm length (box_half = palm_len)
    palm_len = np.linalg.norm(mcp_px - wrist_px)
    expected_w_norm = (2 * palm_len) / frame_w
    actual_w = box[2] - box[0]
    assert abs(actual_w - expected_w_norm) < 0.01


def test_recrop_skips_covered_hand():
    """No re-crop when a real palm detection covers the hand."""
    frame_h, frame_w = 640, 480
    wrist_px = np.array([200, 300])
    mcp_px = np.array([200, 240])
    hand_lm = _make_hand_landmarks(wrist_px, mcp_px)

    # Real palm detection near the wrist
    palm_det = _make_palm_det(200 / frame_w, 300 / frame_h, size=0.05)

    result = _recrop_from_landmarks([hand_lm], [palm_det], frame_h, frame_w)

    assert len(result) == 0


# ---------------------------------------------------------------------------
# Affine matrix degenerate input tests
# ---------------------------------------------------------------------------

def test_affine_matrix_zero_size():
    """Zero-size crop returns None."""
    assert _affine_matrix(100, 100, 0, 0, 256) is None


def test_affine_matrix_nan_inputs():
    """NaN in any positional input returns None."""
    assert _affine_matrix(float("nan"), 100, 0, 100, 256) is None
    assert _affine_matrix(100, float("nan"), 0, 100, 256) is None
    assert _affine_matrix(100, 100, float("nan"), 100, 256) is None
    assert _affine_matrix(100, 100, 0, float("nan"), 256) is None


def test_affine_matrix_inf_inputs():
    """Infinity in any input returns None."""
    assert _affine_matrix(float("inf"), 100, 0, 100, 256) is None
    assert _affine_matrix(100, 100, 0, float("inf"), 256) is None


def test_affine_matrix_valid():
    """Valid inputs produce a (2, 3) finite matrix."""
    M = _affine_matrix(100, 100, 0, 200, 256)
    assert M is not None
    assert M.shape == (2, 3)
    assert np.all(np.isfinite(M))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_funcs = [v for k, v in sorted(globals().items())
                  if k.startswith("test_") and callable(v)]
    for fn in test_funcs:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(test_funcs)} processing tests passed.")
