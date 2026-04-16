"""Tests for detection-level carry-forward, NMS, and decode_detections."""

import numpy as np

from pose_estimation.processing import _smooth_detections
from pose_estimation.detection import nms, decode_detections, generate_anchors


def _make_det(cx, cy, size=0.1, score=0.9):
    """Create a minimal detection dict centred at (cx, cy) in normalised coords."""
    half = size / 2
    return {
        "box": np.array([cx - half, cy - half, cx + half, cy + half],
                        dtype=np.float32),
        "keypoints": np.array([[cx, cy]] * 7, dtype=np.float32),
        "score": score,
    }


# ---- Stable detection: EMA blending ----------------------------------------

def test_stable_detection_blends():
    """Two consecutive frames with the same detection produce EMA-blended output."""
    det1 = _make_det(0.5, 0.5, score=0.8)
    det2 = _make_det(0.52, 0.50, score=0.85)
    alpha = 0.5

    result = _smooth_detections([det2], [det1], alpha=alpha)

    assert len(result) == 1
    expected_box = alpha * det2["box"] + (1 - alpha) * det1["box"]
    np.testing.assert_allclose(result[0]["box"], expected_box, atol=1e-6)
    assert result[0]["score"] == det2["score"]


# ---- One-frame dropout: carry forward --------------------------------------

def test_one_frame_dropout_carries():
    """Frame 1 has a detection, frame 2 has none: carried detection returned."""
    det1 = _make_det(0.5, 0.5, score=0.8)

    result = _smooth_detections([], [det1])

    assert len(result) == 1
    assert result[0].get("_carried") is True


# ---- Two-frame dropout: no second carry ------------------------------------

def test_two_frame_dropout_no_carry():
    """A detection already carried is not carried again (1-frame grace)."""
    det1 = _make_det(0.5, 0.5, score=0.8)

    # Frame 2: dropout → carry
    frame2 = _smooth_detections([], [det1])
    assert len(frame2) == 1
    assert frame2[0].get("_carried") is True

    # Frame 3: dropout again with the carried detection as prev
    frame3 = _smooth_detections([], frame2)
    assert len(frame3) == 0


# ---- Carry score decay ------------------------------------------------------

def test_carry_score_decay():
    """Carried detection's score is reduced by 0.7 decay factor."""
    original_score = 0.8
    det1 = _make_det(0.5, 0.5, score=original_score)

    result = _smooth_detections([], [det1])

    assert len(result) == 1
    expected_score = original_score * 0.7
    assert abs(result[0]["score"] - expected_score) < 1e-6


# ---- Re-acquisition: blends with carried entry -----------------------------

def test_reacquisition_blends_with_carried():
    """After carry, a new matching detection blends smoothly with the carried entry."""
    det1 = _make_det(0.5, 0.5, score=0.8)
    alpha = 0.5

    # Frame 2: dropout → carry
    carried = _smooth_detections([], [det1])
    assert len(carried) == 1

    # Frame 3: new detection near the carried position
    det3 = _make_det(0.51, 0.50, score=0.85)
    result = _smooth_detections([det3], carried, alpha=alpha)

    assert len(result) >= 1
    # The new detection should have blended with the carried entry
    blended = result[0]
    expected_box = alpha * det3["box"] + (1 - alpha) * carried[0]["box"]
    np.testing.assert_allclose(blended["box"], expected_box, atol=1e-6)
    # Score comes from the new detection, not the carried one
    assert blended["score"] == det3["score"]


# ---- Edge cases -------------------------------------------------------------

def test_empty_both():
    """No previous and no new detections returns empty list."""
    result = _smooth_detections([], [])
    assert result == []


def test_no_prev_returns_new():
    """No previous detections: new detections pass through unchanged."""
    det = _make_det(0.3, 0.4, score=0.7)
    result = _smooth_detections([det], [])
    assert len(result) == 1
    np.testing.assert_allclose(result[0]["box"], det["box"])


def test_carry_preserves_box_and_keypoints():
    """Carried detection retains original box and keypoints."""
    det1 = _make_det(0.5, 0.5, score=0.8)
    result = _smooth_detections([], [det1])

    assert len(result) == 1
    np.testing.assert_allclose(result[0]["box"], det1["box"])
    np.testing.assert_allclose(result[0]["keypoints"], det1["keypoints"])


def test_partial_carry_with_new_dets():
    """When some prev_dets match and others don't, unmatched are carried."""
    det_a = _make_det(0.2, 0.2, score=0.8)
    det_b = _make_det(0.8, 0.8, score=0.9)

    # Frame 2: only one new detection near det_a; det_b should carry
    det_a2 = _make_det(0.21, 0.21, score=0.85)
    result = _smooth_detections([det_a2], [det_a, det_b])

    # Should have the blended match plus the carried det_b
    assert len(result) == 2
    carried = [d for d in result if d.get("_carried")]
    assert len(carried) == 1
    assert abs(carried[0]["score"] - 0.9 * 0.7) < 1e-6


# ---- NMS tests --------------------------------------------------------------

def test_nms_no_overlap():
    """Two non-overlapping boxes both survive NMS."""
    boxes = np.array([
        [0.0, 0.0, 0.1, 0.1],
        [0.8, 0.8, 0.9, 0.9],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.3)

    assert len(keep) == 2


def test_nms_full_overlap():
    """Two identical boxes: only the higher-scored one survives."""
    boxes = np.array([
        [0.1, 0.1, 0.5, 0.5],
        [0.1, 0.1, 0.5, 0.5],
    ], dtype=np.float32)
    scores = np.array([0.7, 0.95], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.3)

    assert len(keep) == 1
    assert keep[0] == 1   # higher score


# ---- decode_detections tests -----------------------------------------------

def test_decode_no_detections():
    """All scores below threshold produce empty list."""
    input_size = 192
    num_keypoints = 7
    values_per_anchor = 4 + num_keypoints * 2   # 18

    anchors = generate_anchors(input_size, [8, 16, 16, 16])
    n_anchors = len(anchors)

    # Raw scores: large negative → sigmoid ≈ 0 (well below default 0.5)
    raw_scores = np.full((1, n_anchors, 1), -10.0, dtype=np.float32)
    raw_boxes = np.zeros((1, n_anchors, values_per_anchor), dtype=np.float32)

    result = decode_detections(raw_boxes, raw_scores, anchors, input_size,
                               num_keypoints, score_threshold=0.5)

    assert result == []


# ---- Run all tests ----------------------------------------------------------

if __name__ == "__main__":
    test_funcs = [v for k, v in sorted(globals().items())
                  if k.startswith("test_") and callable(v)]
    for fn in test_funcs:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(test_funcs)} detection tests passed.")
