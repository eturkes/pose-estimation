"""Tests for processing.py helper functions.

Covers synthetic hand generation, landmark re-crop, and affine matrix
degenerate-input handling.
"""

import numpy as np
import pytest

from pose_estimation.processing import (
    _ARM_CHAINS_12,
    _affine_matrix,
    _preprocess_detector,
    _recrop_from_landmarks,
    _refine_pose_landmarks_from_heatmap,
    _synthesise_hand_detections,
    detect_hand_landmarks,
    detect_pose_landmarks,
    get_hand_crop,
    get_pose_crop,
    run_detection,
    transform_landmarks_to_image,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_body(
    shoulder_l=(100, 200),
    elbow_l=(100, 300),
    wrist_l=(100, 400),
    shoulder_r=(300, 200),
    elbow_r=(300, 300),
    wrist_r=(300, 400),
):
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
        "box": np.array(
            [cx_norm - half, cy_norm - half, cx_norm + half, cy_norm + half], dtype=np.float32
        ),
        "keypoints": np.array([[cx_norm, cy_norm]] * 7, dtype=np.float32),
        "score": score,
    }


class _FakeInput:
    def __init__(self, size):
        self.shape = (1, size, size, 3)


class _FakeModel:
    def __init__(self, size, arrays):
        self._input = _FakeInput(size)
        self.outputs = [object() for _ in arrays]
        self._results = dict(zip(self.outputs, arrays, strict=True))
        self.last_tensor = None

    def input(self, _index):
        return self._input

    def output(self, index):
        return self.outputs[index]

    def __call__(self, inputs):
        self.last_tensor = inputs[0]
        return self._results


# ---------------------------------------------------------------------------
# Synthetic hand detection tests
# ---------------------------------------------------------------------------


def test_synthesise_hand_from_arm():
    """Synthetic detection box centre ≈ 40% of forearm beyond wrist,
    box size ≈ 80% of forearm length."""
    body = _make_body(
        elbow_l=(100, 300),
        wrist_l=(100, 400),
        elbow_r=(300, 300),
        wrist_r=(300, 400),
    )
    vis = _make_vis()
    frame_h, frame_w = 640, 480

    result = _synthesise_hand_detections(
        [body],
        [vis],
        [],
        frame_h,
        frame_w,
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
    box_centre_y = (box_px[1] + box_px[3]) / 2
    box_w = box_px[2] - box_px[0]
    box_h = box_px[3] - box_px[1]

    # Centre should be ~40 px beyond wrist (wrist is at y=400, forearm points down)
    expected_centre_y = 400 + forearm_len * 0.4
    assert abs(box_centre_y - expected_centre_y) < 2.0, (
        f"Centre Y {box_centre_y} != expected {expected_centre_y}"
    )

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
        [body],
        [vis],
        [palm_det],
        frame_h,
        frame_w,
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
    mcp_px = np.array([200, 240])  # 60 px palm length, above wrist
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

    # Box size should be 2x palm length (box_half = palm_len)
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
# MediaPipe model-contract geometry and decoding
# ---------------------------------------------------------------------------


def test_detector_preprocess_letterboxes_and_uses_model_value_range():
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    model = _FakeModel(224, [])

    pose_tensor, padding = _preprocess_detector(frame, 224, model, (-1.0, 1.0))
    palm_tensor, palm_padding = _preprocess_detector(frame, 224, model, (0.0, 1.0))

    np.testing.assert_allclose(padding, [0.0, 0.25, 0.0, 0.25])
    np.testing.assert_array_equal(palm_padding, padding)
    assert pose_tensor.shape == (1, 224, 224, 3)
    assert pose_tensor[0, 0, 0, 0] == -1.0  # zero letterbox in [-1, 1]
    assert pose_tensor[0, 112, 112, 0] == 1.0
    assert palm_tensor[0, 0, 0, 0] == 0.0
    assert palm_tensor[0, 112, 112, 0] == 1.0


def test_detector_preprocess_keeps_odd_aspect_padding_symmetric():
    frame = np.full((101, 200, 3), 255, dtype=np.uint8)
    model = _FakeModel(224, [])

    tensor, padding = _preprocess_detector(frame, 224, model, (0.0, 1.0))

    expected_vertical_padding = (1.0 - 101.0 / 200.0) * 0.5
    np.testing.assert_allclose(
        padding,
        [0.0, expected_vertical_padding, 0.0, expected_vertical_padding],
        atol=1e-7,
    )
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor[0, 0, 0, 0] == 0.0
    assert tensor[0, 112, 112, 0] == 1.0


def test_run_detection_removes_letterbox_and_forwards_threshold():
    size = 224
    values = np.zeros((1, 1, 12), dtype=np.float32)
    values[0, 0, 2:4] = 0.2 * size
    scores = np.zeros((1, 1, 1), dtype=np.float32)  # sigmoid = 0.5
    model = _FakeModel(size, [values, scores])
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    anchors = np.array([[0.5, 0.5]], dtype=np.float32)

    detections = run_detection(frame, model, size, anchors, 4, score_threshold=0.5)
    rejected = run_detection(frame, model, size, anchors, 4, score_threshold=0.6)

    assert len(detections) == 1
    np.testing.assert_allclose(detections[0]["box"], [0.4, 0.3, 0.6, 0.7], atol=1e-6)
    np.testing.assert_allclose(detections[0]["keypoints"], 0.5, atol=1e-6)
    assert rejected == []


def test_pose_crop_uses_virtual_hip_centre_and_circle_diameter():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.zeros((4, 2), dtype=np.float32)
    keypoints[0] = (0.5, 0.5)
    keypoints[1] = (0.5, 0.3)

    _crop, matrix = get_pose_crop(frame, {"keypoints": keypoints}, target_size=100)

    assert matrix is not None
    centre = matrix @ np.array([50.0, 50.0, 1.0])
    np.testing.assert_allclose(centre, [50.0, 50.0], atol=1e-6)
    # radius=20 px -> diameter=40 -> graph expansion 1.25 -> 50 px ROI.
    np.testing.assert_allclose(matrix[:, :2], np.eye(2) * 2.0, atol=1e-6)


def test_hand_crop_applies_graph_shift_before_expansion():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = _make_palm_det(0.5, 0.5, size=0.2)
    detection["keypoints"][0] = (0.5, 0.55)
    detection["keypoints"][2] = (0.5, 0.45)

    _crop, matrix = get_hand_crop(frame, detection, target_size=104)

    assert matrix is not None
    # Raw height=20 px and shift_y=-0.5 moves the centre 10 px upward;
    # square/scale then produces a 52 px ROI.
    centre = matrix @ np.array([50.0, 40.0, 1.0])
    np.testing.assert_allclose(centre, [52.0, 52.0], atol=1e-5)
    np.testing.assert_allclose(matrix[:, :2], np.eye(2) * 2.0, atol=1e-5)


def test_landmark_crops_replicate_image_border():
    """MediaPipe landmark ROIs extend edge pixels outside the source image."""
    frame = np.full((20, 20, 3), 77, dtype=np.uint8)
    pose_keypoints = np.zeros((4, 2), dtype=np.float32)
    pose_keypoints[0] = (0.0, 0.0)
    pose_keypoints[1] = (0.0, 0.25)
    pose_crop, _ = get_pose_crop(frame, {"keypoints": pose_keypoints}, target_size=32)

    hand_detection = _make_palm_det(0.0, 0.0, size=0.2)
    hand_detection["keypoints"][0] = (0.0, 0.05)
    hand_detection["keypoints"][2] = (0.0, -0.05)
    hand_crop, _ = get_hand_crop(frame, hand_detection, target_size=32)

    assert pose_crop is not None
    assert np.all(pose_crop == 77)
    assert hand_crop is not None
    assert np.all(hand_crop == 77)


def test_transform_landmarks_scales_depth_with_crop_and_hand_normalization():
    matrix = _affine_matrix(50.0, 50.0, 0.0, 50.0, 100)
    landmarks = np.array([[50.0, 50.0, 10.0]], dtype=np.float32)

    pose = transform_landmarks_to_image(landmarks, matrix)
    hand = transform_landmarks_to_image(landmarks, matrix, z_normalization=0.4)

    np.testing.assert_allclose(pose, [[50.0, 50.0, 5.0]], atol=1e-6)
    np.testing.assert_allclose(hand, [[50.0, 50.0, 12.5]], atol=1e-6)


def test_pose_heatmap_refinement_uses_local_weighted_centroid():
    landmarks = np.zeros((39, 5), dtype=np.float32)
    landmarks[:, :2] = 128.0
    heatmap = np.full((64, 64, 39), -100.0, dtype=np.float32)
    heatmap[30, 35, 0] = 10.0

    refined = _refine_pose_landmarks_from_heatmap(landmarks, heatmap)

    np.testing.assert_allclose(refined[0, :2], [140.0, 120.0], atol=1e-3)
    np.testing.assert_array_equal(refined[1:], landmarks[1:])


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_pose_heatmap_refinement_skips_nonfinite_kernel(nonfinite):
    landmarks = np.zeros((39, 5), dtype=np.float32)
    landmarks[:, :2] = 128.0
    heatmap = np.full((64, 64, 39), -100.0, dtype=np.float32)
    heatmap[32, 32, 0] = nonfinite

    refined = _refine_pose_landmarks_from_heatmap(landmarks, heatmap)

    np.testing.assert_array_equal(refined, landmarks)


def test_pose_flag_is_already_probability_and_presence_limits_visibility():
    raw = np.zeros((1, 195), dtype=np.float32)
    raw.reshape(39, 5)[:, :3] = (128.0, 128.0, 10.0)
    raw.reshape(39, 5)[:, 3] = 4.0
    raw.reshape(39, 5)[:, 4] = -2.0
    model = _FakeModel(
        256,
        [
            raw,
            np.array([[0.8]], dtype=np.float32),
            np.zeros((1, 1, 1, 1), dtype=np.float32),
            np.full((1, 64, 64, 39), -100.0, dtype=np.float32),
            np.zeros((1, 117), dtype=np.float32),
        ],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.zeros((4, 2), dtype=np.float32)
    keypoints[0] = (0.5, 0.5)
    keypoints[1] = (0.5, 0.3)

    landmarks, visibility, flag = detect_pose_landmarks(
        frame, {"keypoints": keypoints}, model, keypoint_indices=[0]
    )

    assert landmarks.shape == (1, 3)
    assert flag == pytest.approx(0.8)
    assert visibility[0] == pytest.approx(1.0 / (1.0 + np.exp(2.0)))


def test_hand_decoder_uses_image_landmarks_and_already_sigmoided_flag():
    image_landmarks = np.zeros((1, 63), dtype=np.float32)
    image_landmarks.reshape(21, 3)[:, :2] = 112.0
    world_landmarks = np.full((1, 63), 999.0, dtype=np.float32)
    model = _FakeModel(
        224,
        [
            image_landmarks,
            np.array([[0.9]], dtype=np.float32),
            np.array([[0.25]], dtype=np.float32),
            world_landmarks,
        ],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = _make_palm_det(0.5, 0.5, size=0.2)
    detection["keypoints"][0] = (0.5, 0.55)
    detection["keypoints"][2] = (0.5, 0.45)

    landmarks, flag, handedness = detect_hand_landmarks(frame, detection, model)

    assert flag == pytest.approx(0.9)
    # Raw 0.25 means model class Right with score 0.75. The default input is
    # unmirrored, so MediaPipe's selfie-oriented label is swapped to Left.
    assert handedness is not None
    assert handedness[0] == "left"
    assert handedness[1] == pytest.approx(0.75)
    assert landmarks.shape == (21, 3)
    assert np.max(np.abs(landmarks[:, :2])) < 100.0

    _landmarks, _flag, mirrored_handedness = detect_hand_landmarks(
        frame, detection, model, input_mirrored=True
    )
    assert mirrored_handedness is not None
    assert mirrored_handedness[0] == "right"
    assert mirrored_handedness[1] == pytest.approx(0.75)
