"""Crop extraction, landmark detection, and per-frame processing pipeline.

Detection-crop smoothing: instead of using raw SSD bounding boxes (which
jitter frame-to-frame), detections are matched across frames and their
keypoints / boxes are exponentially smoothed before crop extraction.  This
stabilises the input to the landmark model, eliminating the main source of
landmark flickering.
"""

import cv2
import numpy as np

from detection import (
    PALM_INPUT_SIZE,
    HAND_INPUT_SIZE,
    POSE_INPUT_SIZE,
    POSE_LM_INPUT_SIZE,
    decode_detections,
)

ARM_KEYPOINT_INDICES = list(range(11, 23))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(frame, size, compiled_model):
    """Resize, convert to RGB float32, and batch for the given compiled model."""
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if list(compiled_model.input(0).shape)[-1] == 3:
        return np.expand_dims(img, 0)
    return np.expand_dims(img.transpose(2, 0, 1), 0)


# ---------------------------------------------------------------------------
# SSD detection
# ---------------------------------------------------------------------------

def run_detection(frame, compiled_model, input_size, anchors, num_keypoints):
    """Run an SSD detection model (pose or palm) on a frame."""
    tensor = _preprocess(frame, input_size, compiled_model)
    results = compiled_model([tensor])

    values_per_anchor = 4 + num_keypoints * 2
    out0 = results[compiled_model.output(0)]
    out1 = results[compiled_model.output(1)]
    if out0.shape[-1] == values_per_anchor:
        raw_boxes, raw_scores = out0, out1
    else:
        raw_boxes, raw_scores = out1, out0

    return decode_detections(raw_boxes, raw_scores, anchors, input_size, num_keypoints)


# ---------------------------------------------------------------------------
# Detection smoothing
# ---------------------------------------------------------------------------

def _smooth_detections(new_dets, prev_dets, match_threshold=0.15, alpha=0.5):
    """Smooth detection keypoints and boxes with an exponential moving average.

    Matches each new detection to the nearest previous detection (by box-
    centre distance in normalised [0, 1] coordinates).  Matched pairs have
    their keypoints and boxes blended; unmatched detections pass through
    as-is.
    """
    if not prev_dets or not new_dets:
        return new_dets

    def _center(det):
        b = det["box"]
        return (b[:2] + b[2:]) / 2

    prev_centers = [_center(d) for d in prev_dets]
    smoothed = []
    used = set()

    for new_det in new_dets:
        nc = _center(new_det)
        best_j, best_d = None, float("inf")
        for j, pc in enumerate(prev_centers):
            if j in used:
                continue
            d = float(np.linalg.norm(nc - pc))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is not None and best_d < match_threshold:
            used.add(best_j)
            prev = prev_dets[best_j]
            smoothed.append({
                "keypoints": alpha * new_det["keypoints"] + (1 - alpha) * prev["keypoints"],
                "box": alpha * new_det["box"] + (1 - alpha) * prev["box"],
                "score": new_det["score"],
            })
        else:
            smoothed.append(new_det)

    return smoothed


# ---------------------------------------------------------------------------
# Affine crop helpers
# ---------------------------------------------------------------------------

def _affine_matrix(cx, cy, rotation, size, target_size):
    """Compute the 2x3 affine warp matrix for a rotation-aware crop.

    Returns *None* when any input is non-finite or the crop size is
    degenerate, so callers can bail out instead of producing a singular
    matrix.
    """
    if size < 1 or not (np.isfinite(cx) and np.isfinite(cy)
                        and np.isfinite(rotation) and np.isfinite(size)):
        return None

    cos_r = np.cos(np.radians(-rotation))
    sin_r = np.sin(np.radians(-rotation))
    scale = target_size / size

    M = np.array([
        [cos_r * scale, -sin_r * scale,
         target_size / 2 - (cx * cos_r - cy * sin_r) * scale],
        [sin_r * scale, cos_r * scale,
         target_size / 2 - (cx * sin_r + cy * cos_r) * scale],
    ], dtype=np.float32)

    if not np.all(np.isfinite(M)):
        return None
    return M


def get_pose_crop(img, detection, scale_factor=2.6, target_size=POSE_LM_INPUT_SIZE):
    """Extract a rotation-aware person crop using pose detection keypoints."""
    img_h, img_w = img.shape[:2]
    kp_hip = detection["keypoints"][0] * np.array([img_w, img_h])
    kp_full = detection["keypoints"][1] * np.array([img_w, img_h])

    cx = (kp_hip[0] + kp_full[0]) / 2
    cy = (kp_hip[1] + kp_full[1]) / 2
    dx = kp_full[0] - kp_hip[0]
    dy = kp_full[1] - kp_hip[1]
    rotation = np.degrees(np.arctan2(dx, -dy))
    body_size = np.sqrt(dx**2 + dy**2) * scale_factor

    M = _affine_matrix(cx, cy, rotation, body_size, target_size)
    if M is None:
        return None, None
    return cv2.warpAffine(img, M, (target_size, target_size)), M


def get_hand_crop(img, detection, scale_factor=2.6, target_size=HAND_INPUT_SIZE):
    """Extract a rotation-aware hand crop using palm detection keypoints."""
    img_h, img_w = img.shape[:2]
    kp_wrist = detection["keypoints"][0] * np.array([img_w, img_h])
    kp_middle = detection["keypoints"][2] * np.array([img_w, img_h])

    box = detection["box"] * np.array([img_w, img_h, img_w, img_h])
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    box_size = max(box[2] - box[0], box[3] - box[1]) * scale_factor

    rotation = np.degrees(np.arctan2(
        kp_middle[0] - kp_wrist[0], -(kp_middle[1] - kp_wrist[1])))
    shift = box_size * 0.05
    cx += shift * np.sin(np.radians(rotation))
    cy -= shift * np.cos(np.radians(rotation))

    M = _affine_matrix(cx, cy, rotation, box_size, target_size)
    if M is None:
        return None, None
    return cv2.warpAffine(img, M, (target_size, target_size)), M


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def transform_landmarks_to_image(landmarks, M):
    """Transform landmarks from crop coordinates back to original image coordinates."""
    M_full = np.vstack([M, [0, 0, 1]])
    try:
        M_inv = np.linalg.inv(M_full)[:2]
    except np.linalg.LinAlgError:
        return landmarks.copy()

    ones = np.ones((landmarks.shape[0], 1))
    pts = np.hstack([landmarks[:, :2], ones])
    original_pts = (M_inv @ pts.T).T

    result = np.copy(landmarks)
    result[:, 0] = original_pts[:, 0]
    result[:, 1] = original_pts[:, 1]
    return result


# ---------------------------------------------------------------------------
# Landmark inference
# ---------------------------------------------------------------------------

def detect_pose_landmarks(frame, detection, pose_lm_compiled):
    """Run pose landmark model. Returns 12 arm keypoints (re-indexed 0-11) and visibility."""
    cropped, M = get_pose_crop(frame, detection)
    if cropped is None:
        return None, None, 0.0

    tensor = _preprocess(cropped, POSE_LM_INPUT_SIZE, pose_lm_compiled)
    results = pose_lm_compiled([tensor])

    landmarks = None
    pose_flag = None

    for output in pose_lm_compiled.outputs:
        data = results[output].squeeze()
        if data.size == 195:
            landmarks = data.reshape(39, 5)
        elif data.size == 1 and pose_flag is None:
            pose_flag = 1.0 / (1.0 + np.exp(-float(data)))

    if landmarks is None or pose_flag is None:
        return None, None, 0.0

    arm_lm = landmarks[ARM_KEYPOINT_INDICES][:, :3].copy()
    arm_lm = transform_landmarks_to_image(arm_lm, M)
    arm_vis = 1.0 / (1.0 + np.exp(-landmarks[ARM_KEYPOINT_INDICES][:, 3]))

    return arm_lm, arm_vis, float(pose_flag)


def detect_hand_landmarks(frame, detection, hand_compiled):
    """Run hand landmark model. Returns 21 keypoints and confidence."""
    cropped, M = get_hand_crop(frame, detection)
    if cropped is None:
        return None, 0.0

    tensor = _preprocess(cropped, HAND_INPUT_SIZE, hand_compiled)
    results = hand_compiled([tensor])

    hand_flag = None
    landmark_candidates = []
    for output in hand_compiled.outputs:
        data = results[output].squeeze()
        if data.size == 63:
            landmark_candidates.append(data.reshape(21, 3))
        elif data.size == 1 and hand_flag is None:
            hand_flag = 1.0 / (1.0 + np.exp(-float(data)))

    if not landmark_candidates or hand_flag is None:
        return None, 0.0

    if len(landmark_candidates) == 1:
        landmarks = landmark_candidates[0]
    else:
        landmarks = max(landmark_candidates,
                        key=lambda lm: np.abs(lm[:, :2]).max())

    landmarks = transform_landmarks_to_image(landmarks, M)
    return landmarks, float(hand_flag)


# ---------------------------------------------------------------------------
# Hand-arm matching & primary-body selection
# ---------------------------------------------------------------------------

def match_hands_to_arms(body_landmarks, hand_landmarks, threshold=100):
    """Match detected hands to the nearest arm wrist by proximity.

    A match is only accepted when the hand is closer to the wrist than
    to the shoulder midpoint, ensuring the hand is at the distal end of
    the arm rather than near the torso.

    Returns list of (arm_idx, arm_wrist_kp, hand_idx) tuples.
    """
    matches = []
    if not body_landmarks or not hand_landmarks:
        return matches

    used_hands = set()
    for arm_idx, arm_lm in enumerate(body_landmarks):
        shoulder_mid = (arm_lm[0, :2] + arm_lm[1, :2]) / 2
        for wrist_kp in [4, 5]:
            arm_wrist = arm_lm[wrist_kp, :2]
            best_hand = None
            best_dist = float('inf')
            for hand_idx, hand_lm in enumerate(hand_landmarks):
                if hand_idx in used_hands:
                    continue
                dist = np.linalg.norm(arm_wrist - hand_lm[0, :2])
                if dist < best_dist:
                    best_dist = dist
                    best_hand = hand_idx
            if best_hand is not None and best_dist < threshold:
                hand_wrist = hand_landmarks[best_hand][0, :2]
                if best_dist < np.linalg.norm(hand_wrist - shoulder_mid):
                    matches.append((arm_idx, wrist_kp, best_hand))
                    used_hands.add(best_hand)
    return matches


def select_primary_body(body_landmarks, body_visibilities, hand_landmarks, matches):
    """Keep only the body with the largest landmark bounding box and its matched hands.

    Returns (body_landmarks, body_visibilities, hand_landmarks, matches)
    with at most one body and only its matched hands, re-indexed to position 0.
    """
    if not body_landmarks:
        return [], [], [], []

    best_idx = 0
    best_area = 0.0
    for i, lm in enumerate(body_landmarks):
        xs, ys = lm[:, 0], lm[:, 1]
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        if area > best_area:
            best_area = area
            best_idx = i

    primary_body = [body_landmarks[best_idx]]
    primary_vis = [body_visibilities[best_idx]]

    hand_index_map = {}
    primary_hands = []
    primary_matches = []
    for arm_idx, wrist_kp, hand_idx in matches:
        if arm_idx == best_idx:
            if hand_idx not in hand_index_map:
                hand_index_map[hand_idx] = len(primary_hands)
                primary_hands.append(hand_landmarks[hand_idx])
            primary_matches.append((0, wrist_kp, hand_index_map[hand_idx]))

    return primary_body, primary_vis, primary_hands, primary_matches


# ---------------------------------------------------------------------------
# Main per-frame pipeline
# ---------------------------------------------------------------------------

def process_frame(frame, models, palm_anchors, pose_anchors,
                  prev_state=None,
                  det_score_threshold=0.5, lm_score_threshold=0.5):
    """Full pipeline: detect arm poses and hand landmarks.

    Detection bounding boxes and keypoints are smoothed against the
    previous frame (*prev_state*) before crop extraction so that the
    landmark model receives a stable input.

    Returns ``(body_landmarks, body_visibilities, hand_landmarks, state)``.
    """
    pose_det_model = models["pose_detection"]
    pose_lm_model = models["pose_landmark"]
    palm_det_model = models["palm_detection"]
    hand_lm_model = models["hand_landmark"]

    # --- Arm pose estimation -----------------------------------------------
    pose_detections = run_detection(
        frame, pose_det_model, POSE_INPUT_SIZE, pose_anchors, 4)
    prev_pose = prev_state.get("pose_dets", []) if prev_state else []
    pose_detections = _smooth_detections(pose_detections, prev_pose)

    body_landmarks = []
    body_visibilities = []
    kept_pose_dets = []
    for det in pose_detections:
        if det["score"] < det_score_threshold:
            continue
        lm, vis, confidence = detect_pose_landmarks(frame, det, pose_lm_model)
        if lm is not None and confidence > lm_score_threshold:
            body_landmarks.append(lm)
            body_visibilities.append(vis)
            kept_pose_dets.append(det)

    # --- Hand pose estimation ----------------------------------------------
    palm_detections = run_detection(
        frame, palm_det_model, PALM_INPUT_SIZE, palm_anchors, 7)
    prev_palm = prev_state.get("palm_dets", []) if prev_state else []
    palm_detections = _smooth_detections(palm_detections, prev_palm)

    hand_landmarks = []
    kept_palm_dets = []
    for det in palm_detections:
        if det["score"] < det_score_threshold:
            continue
        lm, confidence = detect_hand_landmarks(frame, det, hand_lm_model)
        if lm is not None and confidence > lm_score_threshold:
            hand_landmarks.append(lm)
            kept_palm_dets.append(det)

    state = {"pose_dets": kept_pose_dets, "palm_dets": kept_palm_dets}
    return body_landmarks, body_visibilities, hand_landmarks, state
