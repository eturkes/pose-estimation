"""Crop extraction, landmark detection, and per-frame processing pipeline.

Detection-crop smoothing: instead of using raw SSD bounding boxes (which
jitter frame-to-frame), detections are matched across frames and their
keypoints / boxes are exponentially smoothed before crop extraction.  This
stabilises the input to the landmark model, eliminating the main source of
landmark flickering.
"""

import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection import (
    PALM_INPUT_SIZE,
    HAND_INPUT_SIZE,
    POSE_INPUT_SIZE,
    POSE_LM_INPUT_SIZE,
    decode_detections,
)
from metrics import FrameDiagnostics

# ---------------------------------------------------------------------------
# Tracking modes
# ---------------------------------------------------------------------------

TRACKING_HANDS = "hands"
TRACKING_HANDS_ARMS = "hands-arms"
TRACKING_BODY = "body"

# Pose landmark indices to extract per mode
ARM_KEYPOINT_INDICES = list(range(11, 23))   # 12 arm keypoints
BODY_KEYPOINT_INDICES = list(range(33))      # all 33 pose keypoints

# (shoulder, elbow, wrist) triplets used for synthetic hand detections,
# indexed into the extracted keypoint array for each mode.
_ARM_CHAINS_12 = [(0, 2, 4), (1, 3, 5)]
_ARM_CHAINS_33 = [(11, 13, 15), (12, 14, 16)]

# Wrist / shoulder indices in each keypoint scheme, used for hands-arms
# matching and the body-tracking anchor.
WRIST_KPS_12 = (4, 5)
WRIST_KPS_33 = (15, 16)
SHOULDER_KPS_12 = (0, 1)
SHOULDER_KPS_33 = (11, 12)


def tracking_pose_indices(tracking):
    """Return (keypoint_indices, wrist_kps, shoulder_kps, arm_chains)."""
    if tracking == TRACKING_BODY:
        return (BODY_KEYPOINT_INDICES, WRIST_KPS_33,
                SHOULDER_KPS_33, _ARM_CHAINS_33)
    return (ARM_KEYPOINT_INDICES, WRIST_KPS_12,
            SHOULDER_KPS_12, _ARM_CHAINS_12)


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

def _smooth_detections(new_dets, prev_dets, match_threshold=0.15, alpha=None):
    """Smooth detection keypoints and boxes with an exponential moving average.

    Matches each new detection to the nearest previous detection (by box-
    centre distance in normalised [0, 1] coordinates).  Matched pairs have
    their keypoints and boxes blended; unmatched detections pass through
    as-is.
    """
    if alpha is None:
        alpha = float(os.environ.get("POSE_BENCH_DET_SMOOTH_ALPHA", "0.5"))
    if not prev_dets or not new_dets:
        return new_dets

    def _center(det):
        b = det["box"]
        return (b[:2] + b[2:]) / 2

    new_centers = np.array([_center(d) for d in new_dets])
    prev_centers = np.array([_center(d) for d in prev_dets])

    # Optimal assignment via Hungarian algorithm
    diff = new_centers[:, None, :] - prev_centers[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched = {}  # new index -> prev index
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < match_threshold:
            matched[r] = c

    smoothed = []
    for i, new_det in enumerate(new_dets):
        if i in matched:
            prev = prev_dets[matched[i]]
            smoothed.append({
                "keypoints": alpha * new_det["keypoints"] + (1 - alpha) * prev["keypoints"],
                "box": alpha * new_det["box"] + (1 - alpha) * prev["box"],
                "score": new_det["score"],
            })
        else:
            smoothed.append(new_det)

    return smoothed


# ---------------------------------------------------------------------------
# Arm-guided synthetic hand detections
# ---------------------------------------------------------------------------

def _synthesise_hand_detections(body_landmarks, body_visibilities,
                                existing_palm_dets, frame_h, frame_w,
                                arm_chains=None, overlap_threshold=0.1):
    """Create synthetic palm detections from arm wrist keypoints.

    When the palm SSD detector fails (e.g. top-down camera, partial
    occlusion), the arm pose gives a reasonable estimate of hand
    position and orientation.  Synthetic detections are fed to the hand
    landmark model, which will reject bad crops via ``hand_flag``.

    *arm_chains* is a list of (shoulder, elbow, wrist) index triplets
    into the body landmark array.

    Returns a list of detection dicts with ``"synthetic": True``.
    """
    if arm_chains is None:
        arm_chains = _ARM_CHAINS_12
    synthetic = []
    if not body_landmarks:
        return synthetic

    scale = np.array([frame_w, frame_h], dtype=np.float32)

    # Pre-compute centres of existing real palm detections (normalised)
    palm_centres = []
    for det in existing_palm_dets:
        b = det["box"]
        palm_centres.append((b[:2] + b[2:]) / 2)

    for body_lm, body_vis in zip(body_landmarks, body_visibilities):
        for shoulder_idx, elbow_idx, wrist_idx in arm_chains:
            wrist_px = body_lm[wrist_idx, :2]
            elbow_px = body_lm[elbow_idx, :2]

            wrist_norm = wrist_px / scale

            # Skip if a real palm detection already covers this wrist
            if any(np.linalg.norm(wrist_norm - pc) < overlap_threshold
                   for pc in palm_centres):
                continue

            # Forearm vector and length (pixel space)
            forearm = wrist_px - elbow_px
            forearm_len = float(np.linalg.norm(forearm))
            if forearm_len < 1:
                continue
            forearm_dir = forearm / forearm_len

            # Hand centre: ~40 % of forearm length beyond the wrist
            hand_centre_px = wrist_px + forearm_dir * forearm_len * 0.4
            hand_centre_norm = hand_centre_px / scale

            # Middle-finger base estimate: further along forearm
            middle_finger_norm = (wrist_px + forearm_dir * forearm_len * 0.7) / scale

            # Square box sized at 80 % of forearm length, centred on hand
            box_half = forearm_len * 0.4   # half of 0.8 * forearm_len
            x1 = (hand_centre_px[0] - box_half) / frame_w
            y1 = (hand_centre_px[1] - box_half) / frame_h
            x2 = (hand_centre_px[0] + box_half) / frame_w
            y2 = (hand_centre_px[1] + box_half) / frame_h

            # Build 7-keypoint array (get_hand_crop uses kp[0] and kp[2])
            keypoints = np.broadcast_to(
                hand_centre_norm, (7, 2)).astype(np.float32).copy()
            keypoints[0] = wrist_norm
            keypoints[2] = middle_finger_norm

            synthetic.append({
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                "keypoints": keypoints,
                "score": float(body_vis[wrist_idx]),
                "synthetic": True,
            })

    return synthetic


# ---------------------------------------------------------------------------
# Landmark-based re-crop detections
# ---------------------------------------------------------------------------

def _recrop_from_landmarks(prev_hand_landmarks, real_palm_dets,
                           frame_h, frame_w, overlap_threshold=0.1):
    """Create palm-format detections from previous-frame hand landmarks.

    When a tracked hand has no nearby *real* palm detection (e.g. due to
    wrist rotation), the previous frame's landmarks provide a well-centred,
    correctly-rotated crop.  The hand landmark model validates each crop
    via ``hand_flag``, so bad crops self-reject.

    *real_palm_dets* should contain only genuine SSD palm detections —
    **not** synthetic or other fallback entries — so that arm-guided
    detections (which use the forearm direction rather than true hand
    orientation) do not suppress the re-crop.

    Returns a list of detection dicts with ``"recrop": True``.
    """
    recrops = []
    if not prev_hand_landmarks:
        return recrops

    scale = np.array([frame_w, frame_h], dtype=np.float32)

    palm_centres = []
    for det in real_palm_dets:
        b = det["box"]
        palm_centres.append((b[:2] + b[2:]) / 2)

    for hand_lm in prev_hand_landmarks:
        wrist_px = hand_lm[0, :2]
        middle_mcp_px = hand_lm[9, :2]
        wrist_norm = wrist_px / scale

        # Skip if a real palm detection already covers this hand
        if any(np.linalg.norm(wrist_norm - pc) < overlap_threshold
               for pc in palm_centres):
            continue

        # Palm-centred box sized from wrist-to-MCP distance
        palm_len = float(np.linalg.norm(middle_mcp_px - wrist_px))
        if palm_len < 1:
            continue
        center_px = (wrist_px + middle_mcp_px) / 2
        box_half = palm_len

        x1 = (center_px[0] - box_half) / frame_w
        y1 = (center_px[1] - box_half) / frame_h
        x2 = (center_px[0] + box_half) / frame_w
        y2 = (center_px[1] + box_half) / frame_h

        # 7-keypoint array matching palm detection format
        center_norm = center_px / scale
        keypoints = np.broadcast_to(
            center_norm, (7, 2)).astype(np.float32).copy()
        keypoints[0] = wrist_norm
        keypoints[2] = middle_mcp_px / scale

        recrops.append({
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "keypoints": keypoints,
            "score": 0.9,
            "recrop": True,
        })

    return recrops


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

def detect_pose_landmarks(frame, detection, pose_lm_compiled,
                          keypoint_indices=None):
    """Run pose landmark model and extract the requested keypoints.

    *keypoint_indices* selects which of the 39 raw landmarks to return.
    Defaults to the 12 arm keypoints (indices 11–22) for backward
    compatibility.

    Returns (landmarks, visibility, pose_flag).
    """
    if keypoint_indices is None:
        keypoint_indices = ARM_KEYPOINT_INDICES

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

    lm = landmarks[keypoint_indices][:, :3].copy()
    lm = transform_landmarks_to_image(lm, M)
    vis = 1.0 / (1.0 + np.exp(-landmarks[keypoint_indices][:, 3]))

    return lm, vis, float(pose_flag)


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
# Hands-arms matching & primary-body selection
# ---------------------------------------------------------------------------

def match_hands_to_arms(body_landmarks, hand_landmarks, threshold=100,
                        wrist_kps=None, shoulder_kps=None):
    """Match detected hands to arm wrists using optimal assignment.

    Uses the Hungarian algorithm (``linear_sum_assignment``) for
    globally optimal wrist-to-hand pairing, avoiding the greedy-order
    bias where early wrists could steal the only good match for a
    later wrist.

    A match is only accepted when the hand is closer to the wrist than
    to the shoulder midpoint, ensuring the hand is at the distal end of
    the arm rather than near the torso.

    *wrist_kps* and *shoulder_kps* are (left, right) index pairs into
    the body landmark array.  Defaults assume the 12-keypoint arm
    scheme.

    Returns list of (arm_idx, arm_wrist_kp, hand_idx) tuples.
    """
    if wrist_kps is None:
        wrist_kps = WRIST_KPS_12
    if shoulder_kps is None:
        shoulder_kps = SHOULDER_KPS_12

    if not body_landmarks or not hand_landmarks:
        return []

    # Build row index: each row is one (arm_idx, wrist_kp) pair
    rows = []
    wrist_positions = []
    shoulder_mids = []
    for arm_idx, arm_lm in enumerate(body_landmarks):
        shoulder_mid = (arm_lm[shoulder_kps[0], :2]
                        + arm_lm[shoulder_kps[1], :2]) / 2
        for wrist_kp in wrist_kps:
            rows.append((arm_idx, wrist_kp))
            wrist_positions.append(arm_lm[wrist_kp, :2])
            shoulder_mids.append(shoulder_mid)

    n_wrists = len(rows)
    n_hands = len(hand_landmarks)

    # Cost matrix: Euclidean distance between each wrist and hand wrist
    hand_wrists = np.array([h[0, :2] for h in hand_landmarks])
    wrist_arr = np.array(wrist_positions)
    cost = np.linalg.norm(
        wrist_arr[:, None, :] - hand_wrists[None, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= threshold:
            continue
        arm_idx, wrist_kp = rows[r]
        hand_wrist = hand_landmarks[c][0, :2]
        # Distality check: hand must be nearer to wrist than shoulder midpoint
        if cost[r, c] < np.linalg.norm(hand_wrist - shoulder_mids[r]):
            matches.append((arm_idx, wrist_kp, c))

    return matches


def select_primary_body(body_landmarks, body_visibilities, hand_landmarks, matches):
    """Keep only the body with the largest landmark bounding box.

    Hands matched to non-primary bodies are re-matched to the primary
    body if close enough; all other hands are preserved as-is so that
    the caller's upstream filters (age, spatial memory) are respected.

    Returns (body_landmarks, body_visibilities, hand_landmarks, matches)
    with exactly one body, all input hands, and re-indexed matches.
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

    # Re-index matches: keep only those that belonged to the primary body
    primary_matches = []
    for arm_idx, wrist_kp, hand_idx in matches:
        if arm_idx == best_idx:
            primary_matches.append((0, wrist_kp, hand_idx))

    # Hands are passed through unchanged; only the body list and
    # matches are filtered.
    return primary_body, primary_vis, hand_landmarks, primary_matches


# ---------------------------------------------------------------------------
# Main per-frame pipeline
# ---------------------------------------------------------------------------

def process_frame(frame, models, palm_anchors, pose_anchors,
                  prev_state=None, prev_hand_landmarks=None,
                  det_score_threshold=None, lm_score_threshold=None,
                  synthesise_hands=True, tracking=TRACKING_HANDS_ARMS):
    """Full pipeline: detect body poses and hand landmarks.

    *tracking* controls what is detected:

    - ``"hands"``: palm + hand landmarks only (no pose detection).
    - ``"hands-arms"``: 12 arm keypoints + hand landmarks (default).
    - ``"body"``: all 33 pose keypoints + hand landmarks.

    Detection bounding boxes and keypoints are smoothed against the
    previous frame (*prev_state*) before crop extraction so that the
    landmark model receives a stable input.

    *prev_hand_landmarks* (list of (21, 3) arrays in pixel coordinates)
    enables landmark-based re-cropping: when a tracked hand has no
    nearby palm detection, the previous landmarks supply a correctly-
    rotated crop so tracking survives through wrist rotations.

    When *synthesise_hands* is False, arm-guided synthetic palm
    detections are skipped.  Useful in multi-subject mode where
    synthetic detections from spurious body tracks would proliferate
    false hand tracks.

    Returns ``(body_landmarks, body_visibilities, hand_landmarks,
    hand_flags, state, frame_diagnostics)`` where *hand_flags* is a
    list of per-hand ``hand_flag`` confidence floats and
    *frame_diagnostics* is a :class:`metrics.FrameDiagnostics` instance.
    """
    if det_score_threshold is None:
        det_score_threshold = float(
            os.environ.get("POSE_BENCH_DET_SCORE_THRESH", "0.5"))
    if lm_score_threshold is None:
        lm_score_threshold = float(
            os.environ.get("POSE_BENCH_HAND_FLAG_THRESH", "0.65"))

    palm_det_model = models["palm_detection"]
    hand_lm_model = models["hand_landmark"]

    diag = FrameDiagnostics()

    body_landmarks = []
    body_visibilities = []
    kept_pose_dets = []

    # --- Body pose estimation (skipped in hands-only mode) -----------------
    if tracking != TRACKING_HANDS:
        kp_indices, _, _, arm_chains = tracking_pose_indices(tracking)

        pose_det_model = models["pose_detection"]
        pose_lm_model = models["pose_landmark"]

        pose_detections = run_detection(
            frame, pose_det_model, POSE_INPUT_SIZE, pose_anchors, 4)
        prev_pose = prev_state.get("pose_dets", []) if prev_state else []
        pose_detections = _smooth_detections(pose_detections, prev_pose,
                                             match_threshold=0.10)

        best_body_score = 0.0
        for det in pose_detections:
            if det["score"] < det_score_threshold:
                continue
            lm, vis, confidence = detect_pose_landmarks(
                frame, det, pose_lm_model, keypoint_indices=kp_indices)
            if lm is not None and confidence > lm_score_threshold:
                body_landmarks.append(lm)
                body_visibilities.append(vis)
                kept_pose_dets.append(det)
                best_body_score = max(best_body_score, det["score"])

        diag.body_detected = len(body_landmarks) > 0
        diag.body_det_score = best_body_score

    # --- Hand pose estimation ----------------------------------------------
    palm_detections = run_detection(
        frame, palm_det_model, PALM_INPUT_SIZE, palm_anchors, 7)
    prev_palm = prev_state.get("palm_dets", []) if prev_state else []
    palm_detections = _smooth_detections(palm_detections, prev_palm,
                                         match_threshold=0.15)

    # Snapshot real SSD detections before adding fallbacks; re-crop
    # overlap is checked against real detections only so that synthetic
    # entries (which use forearm direction, not true hand orientation)
    # cannot suppress the correctly-rotated re-crop.
    frame_h, frame_w = frame.shape[:2]
    real_palm_dets = list(palm_detections)
    diag.n_hands_real = len(real_palm_dets)

    # Synthesise palm detections from arm wrists not covered by real palms
    n_before_synth = len(palm_detections)
    if body_landmarks and synthesise_hands and tracking != TRACKING_HANDS:
        _, _, _, arm_chains = tracking_pose_indices(tracking)
        palm_detections.extend(_synthesise_hand_detections(
            body_landmarks, body_visibilities, palm_detections,
            frame_h, frame_w, arm_chains=arm_chains))
    diag.n_hands_synthetic = len(palm_detections) - n_before_synth

    # Re-crop from previous hand landmarks when palm detector misses
    n_before_recrop = len(palm_detections)
    if prev_hand_landmarks:
        palm_detections.extend(_recrop_from_landmarks(
            prev_hand_landmarks, real_palm_dets, frame_h, frame_w))
    diag.n_hands_recrop = len(palm_detections) - n_before_recrop

    hand_landmarks = []
    hand_flags = []
    kept_palm_dets = []
    # Per-detection diagnostic records
    hand_diag = []
    for det in palm_detections:
        if det["score"] < det_score_threshold:
            continue
        if det.get("recrop"):
            kind = "recrop"
        elif det.get("synthetic"):
            kind = "synthetic"
        else:
            kind = "real"
        lm, confidence = detect_hand_landmarks(frame, det, hand_lm_model)
        accepted = lm is not None and confidence > lm_score_threshold
        hand_diag.append({
            "kind": kind,
            "det_score": float(det["score"]),
            "hand_flag": round(float(confidence), 4),
            "accepted": accepted,
        })
        if accepted:
            hand_landmarks.append(lm)
            hand_flags.append(float(confidence))
            if kind == "real":
                kept_palm_dets.append(det)

    diag.hand_diag = hand_diag
    # Snapshot raw landmarks before smoothing
    diag.raw_body_landmarks = [lm.copy() for lm in body_landmarks]
    diag.raw_body_visibilities = [v.copy() for v in body_visibilities]
    diag.raw_hand_landmarks = [lm.copy() for lm in hand_landmarks]

    state = {"pose_dets": kept_pose_dets, "palm_dets": kept_palm_dets,
             "hand_diag": hand_diag}
    return body_landmarks, body_visibilities, hand_landmarks, hand_flags, state, diag
