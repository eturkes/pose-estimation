"""Crop extraction, landmark detection, and per-frame processing pipeline."""

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


def _preprocess(frame, size, compiled_model):
    """Resize, convert to RGB float32, and batch for the given compiled model."""
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if list(compiled_model.input(0).shape)[-1] == 3:
        return np.expand_dims(img, 0)
    return np.expand_dims(img.transpose(2, 0, 1), 0)


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

    cos_r = np.cos(np.radians(-rotation))
    sin_r = np.sin(np.radians(-rotation))
    scale = target_size / body_size

    M = np.array([
        [cos_r * scale, -sin_r * scale, target_size / 2 - (cx * cos_r - cy * sin_r) * scale],
        [sin_r * scale,  cos_r * scale, target_size / 2 - (cx * sin_r + cy * cos_r) * scale],
    ], dtype=np.float32)

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

    rotation = np.degrees(np.arctan2(kp_middle[0] - kp_wrist[0], -(kp_middle[1] - kp_wrist[1])))
    shift = box_size * 0.05
    cx += shift * np.sin(np.radians(rotation))
    cy -= shift * np.cos(np.radians(rotation))

    cos_r = np.cos(np.radians(-rotation))
    sin_r = np.sin(np.radians(-rotation))
    scale = target_size / box_size

    M = np.array([
        [cos_r * scale, -sin_r * scale, target_size / 2 - (cx * cos_r - cy * sin_r) * scale],
        [sin_r * scale,  cos_r * scale, target_size / 2 - (cx * sin_r + cy * cos_r) * scale],
    ], dtype=np.float32)

    return cv2.warpAffine(img, M, (target_size, target_size)), M


def transform_landmarks_to_image(landmarks, M):
    """Transform landmarks from crop coordinates back to original image coordinates."""
    M_full = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_full)[:2]

    ones = np.ones((landmarks.shape[0], 1))
    pts = np.hstack([landmarks[:, :2], ones])
    original_pts = (M_inv @ pts.T).T

    result = np.copy(landmarks)
    result[:, 0] = original_pts[:, 0]
    result[:, 1] = original_pts[:, 1]
    return result


def detect_pose_landmarks(frame, detection, pose_lm_compiled):
    """Run pose landmark model. Returns 12 arm keypoints (re-indexed 0-11) and visibility."""
    cropped, M = get_pose_crop(frame, detection)
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
    tensor = _preprocess(cropped, HAND_INPUT_SIZE, hand_compiled)
    results = hand_compiled([tensor])

    landmarks = None
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
        landmarks = max(landmark_candidates, key=lambda lm: np.abs(lm[:, :2]).max())

    landmarks = transform_landmarks_to_image(landmarks, M)
    return landmarks, float(hand_flag)


def match_hands_to_arms(body_landmarks, hand_landmarks, threshold=100):
    """Match detected hands to the nearest arm wrist by proximity.

    Returns list of (arm_idx, arm_wrist_kp, hand_idx) tuples.
    """
    matches = []
    if not body_landmarks or not hand_landmarks:
        return matches

    used_hands = set()
    for arm_idx, arm_lm in enumerate(body_landmarks):
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
                matches.append((arm_idx, wrist_kp, best_hand))
                used_hands.add(best_hand)
    return matches


def process_frame(frame, models, palm_anchors, pose_anchors,
                  det_score_threshold=0.5, lm_score_threshold=0.5):
    """Full pipeline: detect arm poses and hand landmarks.

    Args:
        models: dict from download_and_compile_models() with keys
            "pose_detection", "pose_landmark", "palm_detection", "hand_landmark".
    """
    pose_det = models["pose_detection"]
    pose_lm = models["pose_landmark"]
    palm_det = models["palm_detection"]
    hand_lm = models["hand_landmark"]

    # Arm pose estimation
    pose_detections = run_detection(frame, pose_det, POSE_INPUT_SIZE, pose_anchors, 4)
    body_landmarks = []
    body_visibilities = []
    for det in pose_detections:
        if det["score"] < det_score_threshold:
            continue
        lm, vis, confidence = detect_pose_landmarks(frame, det, pose_lm)
        if lm is not None and confidence > lm_score_threshold:
            body_landmarks.append(lm)
            body_visibilities.append(vis)

    # Hand pose estimation
    palm_detections = run_detection(frame, palm_det, PALM_INPUT_SIZE, palm_anchors, 7)
    hand_landmarks = []
    for det in palm_detections:
        if det["score"] < det_score_threshold:
            continue
        lm, confidence = detect_hand_landmarks(frame, det, hand_lm)
        if lm is not None and confidence > lm_score_threshold:
            hand_landmarks.append(lm)

    return body_landmarks, body_visibilities, hand_landmarks
