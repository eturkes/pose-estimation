"""Crop extraction, landmark detection, and per-frame processing pipeline.

Detection-crop smoothing: instead of using raw SSD bounding boxes (which
jitter frame-to-frame), detections are matched across frames and their
keypoints / boxes are exponentially smoothed before crop extraction.  This
stabilises the input to the landmark model, eliminating the main source of
landmark flickering.
"""

import math
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import (
    HAND_INPUT_SIZE,
    PALM_INPUT_SIZE,
    POSE_INPUT_SIZE,
    POSE_LM_INPUT_SIZE,
    decode_detections,
)
from .metrics import FrameDiagnostics

# ---------------------------------------------------------------------------
# Tracking modes
# ---------------------------------------------------------------------------

TRACKING_HANDS = "hands"
TRACKING_HANDS_ARMS = "hands-arms"
TRACKING_BODY = "body"

# Pose landmark indices to extract per mode
ARM_KEYPOINT_INDICES = list(range(11, 23))  # 12 arm keypoints
BODY_KEYPOINT_INDICES = list(range(33))  # all 33 pose keypoints

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

# ---------------------------------------------------------------------------
# Pipeline tuning constants
# ---------------------------------------------------------------------------

# Detection-level smoothing
DEFAULT_DET_SMOOTH_ALPHA = 0.5  # EMA blend between previous and current
DET_MATCH_THRESHOLD_PALM = 0.15  # normalised image-space distance
DET_MATCH_THRESHOLD_POSE = 0.10
CARRIED_DET_SCORE_DECAY = 0.7  # confidence multiplier when carried one frame

# Synthetic palm detections derived from the forearm
SYNTHETIC_HAND_CENTRE_OFFSET = 0.4  # fraction of forearm length past wrist
SYNTHETIC_HAND_FINGER_OFFSET = 0.7  # fraction of forearm length to MCP guess
SYNTHETIC_HAND_BOX_HALF_FACTOR = 0.4  # half of 0.8 * forearm_len
DETECTION_OVERLAP_THRESHOLD = 0.1  # normalised distance to suppress fallback
RECROP_DET_SCORE = 0.9  # synthetic confidence assigned to re-crop entries

# Crop extraction
POSE_CROP_SCALE_FACTOR = 2.6
HAND_CROP_SCALE_FACTOR = 2.6
HAND_CROP_SHIFT_FACTOR = 0.05  # forward shift along the palm orientation
MIN_BONE_LENGTH_PX = 1.0  # below this we skip the chain entirely

# Palm-detection 7-keypoint layout (used by get_hand_crop)
PALM_KP_COUNT = 7
PALM_WRIST_KP_IDX = 0
PALM_FINGER_KP_IDX = 2

# Numerical guard
EPSILON = 1e-6


def tracking_pose_indices(tracking):
    """Return (keypoint_indices, wrist_kps, shoulder_kps, arm_chains)."""
    if tracking == TRACKING_BODY:
        return (BODY_KEYPOINT_INDICES, WRIST_KPS_33, SHOULDER_KPS_33, _ARM_CHAINS_33)
    return (ARM_KEYPOINT_INDICES, WRIST_KPS_12, SHOULDER_KPS_12, _ARM_CHAINS_12)


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


def _detection_centre(det):
    """Return the box centre of *det* in normalised image coordinates."""
    box = det["box"]
    return (box[:2] + box[2:]) / 2


def _detection_centres(dets):
    """Vectorised list of box centres for an iterable of detections."""
    return [_detection_centre(d) for d in dets]


def _detection_centres_array(dets):
    """Stack detection box centres into a single (n, 2) ndarray in one pass.

    Direct scalar fill beats ``np.array(list[(box[:2] + box[2:]) / 2 for ...])``
    at the typical 1-6 detection counts: a per-det numpy slice + ufunc
    pair costs ~1 µs of dispatch overhead, vs ~80 ns for the four scalar
    reads used here.
    """
    n = len(dets)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    out = np.empty((n, 2), dtype=np.float64)
    for i, d in enumerate(dets):
        box = d["box"]
        out[i, 0] = (float(box[0]) + float(box[2])) * 0.5
        out[i, 1] = (float(box[1]) + float(box[3])) * 0.5
    return out


def _make_palm_keypoints(centre_norm, wrist_norm, finger_norm):
    """Build a 7-keypoint palm-detection array.

    ``get_hand_crop`` reads the wrist (kp[0]) and middle-finger MCP
    (kp[2]); the remaining slots can be the box centre.
    """
    kps = np.empty((PALM_KP_COUNT, 2), dtype=np.float32)
    kps[:] = centre_norm  # broadcasts (2,) to (PALM_KP_COUNT, 2)
    kps[PALM_WRIST_KP_IDX] = wrist_norm
    kps[PALM_FINGER_KP_IDX] = finger_norm
    return kps


def _palm_centres_list(dets):
    """Return detection box centres as a list of ``(cx, cy)`` Python tuples.

    Both callers iterate the centres in a tight scalar overlap check; at
    the 0-6 palm-det sizes we see in production, Python iteration beats
    the per-step ufunc dispatch overhead of an ``(n, 2)`` ndarray.
    """
    if not dets:
        return None
    out = []
    for d in dets:
        box = d["box"]
        out.append(
            (
                (float(box[0]) + float(box[2])) * 0.5,
                (float(box[1]) + float(box[3])) * 0.5,
            )
        )
    return out


def _carry_detection(det):
    """Return a one-frame carry-forward copy of *det* with decayed score."""
    out = dict(det)
    out["score"] = out["score"] * CARRIED_DET_SCORE_DECAY
    out["_carried"] = True
    return out


def _smooth_detections(new_dets, prev_dets, match_threshold=DET_MATCH_THRESHOLD_PALM, alpha=None):
    """Smooth detection keypoints and boxes with an exponential moving average.

    Matches each new detection to the nearest previous detection (by box-
    centre distance in normalised [0, 1] coordinates).  Matched pairs have
    their keypoints and boxes blended; unmatched detections pass through
    as-is.

    Previous detections that have no match in *new_dets* are carried
    forward for one frame with decayed confidence
    (``score *= CARRIED_DET_SCORE_DECAY``).  A detection already marked
    ``_carried`` will not be carried again, limiting the grace period
    to a single frame.
    """
    if alpha is None:
        alpha = float(os.environ.get("POSE_BENCH_DET_SMOOTH_ALPHA", str(DEFAULT_DET_SMOOTH_ALPHA)))

    # Indices of prev_dets eligible for carry-forward (not already carried)
    carry_eligible = set()
    if prev_dets:
        carry_eligible = {i for i, d in enumerate(prev_dets) if not d.get("_carried")}

    if not prev_dets or not new_dets:
        # No new detections: carry forward eligible prev_dets for one frame
        if not new_dets and carry_eligible:
            return [_carry_detection(prev_dets[i]) for i in sorted(carry_eligible)]
        return list(new_dets) if new_dets else []

    new_centers = _detection_centres_array(new_dets)
    prev_centers = _detection_centres_array(prev_dets)

    # Optimal assignment via Hungarian algorithm.  ``np.hypot`` on explicit
    # dx/dy avoids the per-axis-norm path inside ``np.linalg.norm``, which
    # has measurable Python-side overhead at our tiny (≤4, ≤4) sizes.
    dx = new_centers[:, 0:1] - prev_centers[None, :, 0]
    dy = new_centers[:, 1:2] - prev_centers[None, :, 1]
    cost = np.hypot(dx, dy)
    row_ind, col_ind = linear_sum_assignment(cost)

    # new index -> prev index
    matched = {r: c for r, c in zip(row_ind, col_ind, strict=False) if cost[r, c] < match_threshold}

    smoothed = []
    for i, new_det in enumerate(new_dets):
        if i in matched:
            prev = prev_dets[matched[i]]
            smoothed.append(
                {
                    "keypoints": alpha * new_det["keypoints"] + (1 - alpha) * prev["keypoints"],
                    "box": alpha * new_det["box"] + (1 - alpha) * prev["box"],
                    "score": new_det["score"],
                }
            )
        else:
            smoothed.append(new_det)

    # Carry forward unmatched, non-carried prev_dets for one frame
    matched_prev = set(matched.values())
    smoothed.extend(_carry_detection(prev_dets[i]) for i in sorted(carry_eligible - matched_prev))

    return smoothed


# ---------------------------------------------------------------------------
# Arm-guided synthetic hand detections
# ---------------------------------------------------------------------------


def _synthesise_hand_detections(
    body_landmarks,
    body_visibilities,
    existing_palm_dets,
    frame_h,
    frame_w,
    arm_chains=None,
    overlap_threshold=DETECTION_OVERLAP_THRESHOLD,
):
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

    # Constants pulled out of the inner loop
    inv_fw = 1.0 / frame_w
    inv_fh = 1.0 / frame_h
    overlap_sq = overlap_threshold * overlap_threshold
    palm_centres = _palm_centres_list(existing_palm_dets)

    for body_lm, body_vis in zip(body_landmarks, body_visibilities, strict=False):
        for _shoulder_idx, elbow_idx, wrist_idx in arm_chains:
            wrist_x = float(body_lm[wrist_idx, 0])
            wrist_y = float(body_lm[wrist_idx, 1])
            elbow_x = float(body_lm[elbow_idx, 0])
            elbow_y = float(body_lm[elbow_idx, 1])

            wrist_nx = wrist_x * inv_fw
            wrist_ny = wrist_y * inv_fh

            # Skip if a real palm detection already covers this wrist
            # (squared-distance comparison avoids per-chain sqrt).  Scalar
            # loop beats numpy here at the small palm-centre counts.
            if palm_centres is not None:
                hit = False
                for cx, cy in palm_centres:
                    ddx = cx - wrist_nx
                    ddy = cy - wrist_ny
                    if ddx * ddx + ddy * ddy < overlap_sq:
                        hit = True
                        break
                if hit:
                    continue

            fdx = wrist_x - elbow_x
            fdy = wrist_y - elbow_y
            forearm_len = math.hypot(fdx, fdy)
            if forearm_len < MIN_BONE_LENGTH_PX:
                continue
            inv_fl = 1.0 / forearm_len
            dir_x = fdx * inv_fl
            dir_y = fdy * inv_fl

            centre_off = forearm_len * SYNTHETIC_HAND_CENTRE_OFFSET
            hand_cx = wrist_x + dir_x * centre_off
            hand_cy = wrist_y + dir_y * centre_off

            finger_off = forearm_len * SYNTHETIC_HAND_FINGER_OFFSET
            finger_px_x = wrist_x + dir_x * finger_off
            finger_px_y = wrist_y + dir_y * finger_off

            box_half = forearm_len * SYNTHETIC_HAND_BOX_HALF_FACTOR
            x1 = (hand_cx - box_half) * inv_fw
            y1 = (hand_cy - box_half) * inv_fh
            x2 = (hand_cx + box_half) * inv_fw
            y2 = (hand_cy + box_half) * inv_fh

            ccx = hand_cx * inv_fw
            ccy = hand_cy * inv_fh
            fnx = finger_px_x * inv_fw
            fny = finger_px_y * inv_fh

            keypoints = np.empty((PALM_KP_COUNT, 2), dtype=np.float32)
            keypoints[:, 0] = ccx
            keypoints[:, 1] = ccy
            keypoints[PALM_WRIST_KP_IDX, 0] = wrist_nx
            keypoints[PALM_WRIST_KP_IDX, 1] = wrist_ny
            keypoints[PALM_FINGER_KP_IDX, 0] = fnx
            keypoints[PALM_FINGER_KP_IDX, 1] = fny

            box = np.empty(4, dtype=np.float32)
            box[0] = x1
            box[1] = y1
            box[2] = x2
            box[3] = y2

            synthetic.append(
                {
                    "box": box,
                    "keypoints": keypoints,
                    "score": float(body_vis[wrist_idx]),
                    "synthetic": True,
                }
            )

    return synthetic


# ---------------------------------------------------------------------------
# Landmark-based re-crop detections
# ---------------------------------------------------------------------------


def _recrop_from_landmarks(
    prev_hand_landmarks,
    real_palm_dets,
    frame_h,
    frame_w,
    overlap_threshold=DETECTION_OVERLAP_THRESHOLD,
):
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

    inv_fw = 1.0 / frame_w
    inv_fh = 1.0 / frame_h
    overlap_sq = overlap_threshold * overlap_threshold
    palm_centres = _palm_centres_list(real_palm_dets)

    for hand_lm in prev_hand_landmarks:
        wrist_x = float(hand_lm[0, 0])
        wrist_y = float(hand_lm[0, 1])
        mcp_x = float(hand_lm[9, 0])
        mcp_y = float(hand_lm[9, 1])

        wrist_nx = wrist_x * inv_fw
        wrist_ny = wrist_y * inv_fh

        if palm_centres is not None:
            hit = False
            for cx, cy in palm_centres:
                ddx = cx - wrist_nx
                ddy = cy - wrist_ny
                if ddx * ddx + ddy * ddy < overlap_sq:
                    hit = True
                    break
            if hit:
                continue

        pdx = mcp_x - wrist_x
        pdy = mcp_y - wrist_y
        palm_len = math.hypot(pdx, pdy)
        if palm_len < MIN_BONE_LENGTH_PX:
            continue

        center_x = (wrist_x + mcp_x) * 0.5
        center_y = (wrist_y + mcp_y) * 0.5

        x1 = (center_x - palm_len) * inv_fw
        y1 = (center_y - palm_len) * inv_fh
        x2 = (center_x + palm_len) * inv_fw
        y2 = (center_y + palm_len) * inv_fh

        ccx = center_x * inv_fw
        ccy = center_y * inv_fh
        mcp_nx = mcp_x * inv_fw
        mcp_ny = mcp_y * inv_fh

        keypoints = np.empty((PALM_KP_COUNT, 2), dtype=np.float32)
        keypoints[:, 0] = ccx
        keypoints[:, 1] = ccy
        keypoints[PALM_WRIST_KP_IDX, 0] = wrist_nx
        keypoints[PALM_WRIST_KP_IDX, 1] = wrist_ny
        keypoints[PALM_FINGER_KP_IDX, 0] = mcp_nx
        keypoints[PALM_FINGER_KP_IDX, 1] = mcp_ny

        box = np.empty(4, dtype=np.float32)
        box[0] = x1
        box[1] = y1
        box[2] = x2
        box[3] = y2

        recrops.append(
            {
                "box": box,
                "keypoints": keypoints,
                "score": RECROP_DET_SCORE,
                "recrop": True,
            }
        )

    return recrops


# ---------------------------------------------------------------------------
# Affine crop helpers
# ---------------------------------------------------------------------------


def _affine_matrix(cx, cy, rotation, size, target_size):
    """Compute the 2x3 affine warp matrix for a rotation-aware crop.

    Returns *None* when any input is non-finite or the crop size is
    degenerate, so callers can bail out instead of producing a singular
    matrix.

    Inputs are Python / numpy scalars; ``math.*`` on scalars dispatches
    far faster than ``np.*`` (which goes through ufunc machinery), and
    np.empty + scalar writes beats ``np.array([[...]])`` for the small
    2x3 result.
    """
    if size < 1 or not (
        math.isfinite(cx) and math.isfinite(cy) and math.isfinite(rotation) and math.isfinite(size)
    ):
        return None

    neg_rot_rad = math.radians(-rotation)
    cos_r = math.cos(neg_rot_rad)
    sin_r = math.sin(neg_rot_rad)
    scale = target_size / size
    half = 0.5 * target_size
    cos_scale = cos_r * scale
    sin_scale = sin_r * scale
    tx = half - (cx * cos_r - cy * sin_r) * scale
    ty = half - (cx * sin_r + cy * cos_r) * scale
    # Translation is the only term that can blow up (scale or product of
    # finite values can overflow); rotation block is finite by trig.
    if not (math.isfinite(tx) and math.isfinite(ty) and math.isfinite(cos_scale)):
        return None

    M = np.empty((2, 3), dtype=np.float32)
    M[0, 0] = cos_scale
    M[0, 1] = -sin_scale
    M[0, 2] = tx
    M[1, 0] = sin_scale
    M[1, 1] = cos_scale
    M[1, 2] = ty
    return M


def get_pose_crop(
    img, detection, scale_factor=POSE_CROP_SCALE_FACTOR, target_size=POSE_LM_INPUT_SIZE
):
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


def get_hand_crop(img, detection, scale_factor=HAND_CROP_SCALE_FACTOR, target_size=HAND_INPUT_SIZE):
    """Extract a rotation-aware hand crop using palm detection keypoints."""
    img_h, img_w = img.shape[:2]
    kp_wrist = detection["keypoints"][PALM_WRIST_KP_IDX] * np.array([img_w, img_h])
    kp_middle = detection["keypoints"][PALM_FINGER_KP_IDX] * np.array([img_w, img_h])

    box = detection["box"] * np.array([img_w, img_h, img_w, img_h])
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    box_size = max(box[2] - box[0], box[3] - box[1]) * scale_factor

    rotation = np.degrees(np.arctan2(kp_middle[0] - kp_wrist[0], -(kp_middle[1] - kp_wrist[1])))
    shift = box_size * HAND_CROP_SHIFT_FACTOR
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
    """Transform landmarks from crop coordinates back to original image coordinates.

    M is a 2x3 affine ``[[a, b, c], [d, e, f]]``; its inverse is computed
    via the closed-form cofactor expansion (no ``np.linalg.inv`` / vstack
    / matmul / transpose ceremony), which dominated runtime at the
    keypoint counts seen here (12-33).
    """
    a = float(M[0, 0])
    b = float(M[0, 1])
    c = float(M[0, 2])
    d = float(M[1, 0])
    e = float(M[1, 1])
    f = float(M[1, 2])

    det = a * e - b * d
    if abs(det) < 1e-12:
        return landmarks.copy()

    inv_det = 1.0 / det
    a_inv = e * inv_det
    b_inv = -b * inv_det
    c_inv = (b * f - c * e) * inv_det
    d_inv = -d * inv_det
    e_inv = a * inv_det
    f_inv = (c * d - a * f) * inv_det

    result = landmarks.copy()
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    # Two fused multiply-add reductions over the keypoint axis; the
    # original (M_inv @ pts.T).T was the same math but paid 4 array allocs
    # plus matmul Python overhead.
    result[:, 0] = a_inv * x + b_inv * y + c_inv
    result[:, 1] = d_inv * x + e_inv * y + f_inv
    return result


# ---------------------------------------------------------------------------
# Landmark inference
# ---------------------------------------------------------------------------


def detect_pose_landmarks(frame, detection, pose_lm_compiled, keypoint_indices=None):
    """Run pose landmark model and extract the requested keypoints.

    *keypoint_indices* selects which of the 39 raw landmarks to return.
    Defaults to the 12 arm keypoints (indices 11-22) for backward
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
        landmarks = max(landmark_candidates, key=lambda lm: np.abs(lm[:, :2]).max())

    landmarks = transform_landmarks_to_image(landmarks, M)
    return landmarks, float(hand_flag)


# ---------------------------------------------------------------------------
# Hands-arms matching & primary-body selection
# ---------------------------------------------------------------------------


def match_hands_to_arms(
    body_landmarks, hand_landmarks, threshold=100, wrist_kps=None, shoulder_kps=None
):
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

    n_bodies = len(body_landmarks)
    n_wrists = len(wrist_kps)
    n_hands = len(hand_landmarks)

    # Stack body landmarks once → fancy-index wrists + shoulders in bulk.
    # The two original Python loops (per body / per wrist) collapse into
    # one numpy slice + reshape.  The ``[None]`` view trick avoids a
    # full ``np.stack`` copy for the common single-body case.
    if n_bodies == 1:
        bodies_stack = body_landmarks[0][None]
    else:
        bodies_stack = np.stack(body_landmarks)

    wrist_arr = bodies_stack[:, wrist_kps, :2].reshape(n_bodies * n_wrists, 2)
    shoulder_mid = (
        bodies_stack[:, shoulder_kps[0], :2] + bodies_stack[:, shoulder_kps[1], :2]
    ) * 0.5  # (n_bodies, 2)

    # Hand wrist [0] positions: (n_hands, 2).  np.empty + per-row copy
    # beats both ``np.stack(list)`` and ``np.array([h[0, :2] for ...])``
    # at the n ≤ 8 sizes seen in practice (no list-validation overhead).
    hand_wrists = np.empty((n_hands, 2), dtype=wrist_arr.dtype)
    for i, h in enumerate(hand_landmarks):
        hand_wrists[i] = h[0, :2]

    # Cost matrix via np.hypot (one ufunc) instead of subtract / square /
    # sum / sqrt that ``np.linalg.norm`` decomposes into.
    dx = wrist_arr[:, 0:1] - hand_wrists[None, :, 0]
    dy = wrist_arr[:, 1:2] - hand_wrists[None, :, 1]
    cost = np.hypot(dx, dy)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind, strict=False):
        c_rc = float(cost[r, c])
        if c_rc >= threshold:
            continue
        r_int = int(r)
        arm_idx = r_int // n_wrists
        wrist_kp = wrist_kps[r_int % n_wrists]
        # Distality check via scalar math.hypot — far cheaper than
        # np.linalg.norm at 2-element vector sizes.
        sx = float(shoulder_mid[arm_idx, 0])
        sy = float(shoulder_mid[arm_idx, 1])
        hx = float(hand_wrists[c, 0])
        hy = float(hand_wrists[c, 1])
        if c_rc < math.hypot(hx - sx, hy - sy):
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

    n_bodies = len(body_landmarks)
    if n_bodies == 1:
        best_idx = 0
    elif n_bodies == 2:
        # ``np.stack`` overhead exceeds 4 min/max ufuncs per body at this
        # size, so keep the per-body loop here.  Crossover measured at
        # ~3 bodies on the bench.
        best_idx = 0
        best_area = -1.0
        for i in range(n_bodies):
            lm = body_landmarks[i]
            xs = lm[:, 0]
            ys = lm[:, 1]
            area = (xs.max() - xs.min()) * (ys.max() - ys.min())
            if area > best_area:
                best_area = area
                best_idx = i
    else:
        # Stack once and reduce along the keypoint axis: 4 ufunc calls
        # total instead of 4·n_bodies, which dominated the per-body loop.
        stacked = np.stack(body_landmarks)
        xs = stacked[:, :, 0]
        ys = stacked[:, :, 1]
        areas = (xs.max(axis=1) - xs.min(axis=1)) * (ys.max(axis=1) - ys.min(axis=1))
        best_idx = int(np.argmax(areas))

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


def process_frame(
    frame,
    models,
    palm_anchors,
    pose_anchors,
    prev_state=None,
    prev_hand_landmarks=None,
    det_score_threshold=None,
    lm_score_threshold=None,
    synthesise_hands=True,
    tracking=TRACKING_HANDS_ARMS,
):
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
        det_score_threshold = float(os.environ.get("POSE_BENCH_DET_SCORE_THRESH", "0.5"))
    if lm_score_threshold is None:
        lm_score_threshold = float(os.environ.get("POSE_BENCH_HAND_FLAG_THRESH", "0.65"))

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

        pose_detections = run_detection(frame, pose_det_model, POSE_INPUT_SIZE, pose_anchors, 4)
        prev_pose = prev_state.get("pose_dets", []) if prev_state else []
        pose_detections = _smooth_detections(
            pose_detections, prev_pose, match_threshold=DET_MATCH_THRESHOLD_POSE
        )

        best_body_score = 0.0
        for det in pose_detections:
            if det["score"] < det_score_threshold:
                continue
            lm, vis, confidence = detect_pose_landmarks(
                frame, det, pose_lm_model, keypoint_indices=kp_indices
            )
            if lm is not None and confidence > lm_score_threshold:
                body_landmarks.append(lm)
                body_visibilities.append(vis)
                kept_pose_dets.append(det)
                best_body_score = max(best_body_score, det["score"])

        diag.body_detected = len(body_landmarks) > 0
        diag.body_det_score = best_body_score

    # --- Hand pose estimation ----------------------------------------------
    palm_detections = run_detection(frame, palm_det_model, PALM_INPUT_SIZE, palm_anchors, 7)
    prev_palm = prev_state.get("palm_dets", []) if prev_state else []
    palm_detections = _smooth_detections(
        palm_detections, prev_palm, match_threshold=DET_MATCH_THRESHOLD_PALM
    )

    # Snapshot real SSD detections before adding fallbacks; re-crop
    # overlap is checked against real detections only so that synthetic
    # entries (which use forearm direction, not true hand orientation)
    # cannot suppress the correctly-rotated re-crop.
    frame_h, frame_w = frame.shape[:2]
    real_palm_dets = list(palm_detections)
    diag.n_hands_real = len(real_palm_dets)

    n_before_synth = len(palm_detections)
    if body_landmarks and synthesise_hands and tracking != TRACKING_HANDS:
        palm_detections.extend(
            _synthesise_hand_detections(
                body_landmarks,
                body_visibilities,
                palm_detections,
                frame_h,
                frame_w,
                arm_chains=arm_chains,
            )
        )
    diag.n_hands_synthetic = len(palm_detections) - n_before_synth

    n_before_recrop = len(palm_detections)
    if prev_hand_landmarks:
        palm_detections.extend(
            _recrop_from_landmarks(prev_hand_landmarks, real_palm_dets, frame_h, frame_w)
        )
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
        hand_diag.append(
            {
                "kind": kind,
                "det_score": float(det["score"]),
                "hand_flag": round(float(confidence), 4),
                "accepted": accepted,
            }
        )
        if accepted:
            hand_landmarks.append(lm)
            hand_flags.append(float(confidence))
            if kind == "real":
                kept_palm_dets.append(det)

    diag.hand_diag = hand_diag
    # Raw (pre-smoothing) references; smoothers return new arrays without
    # mutating inputs, so sharing references here is safe.
    diag.raw_body_landmarks = body_landmarks
    diag.raw_body_visibilities = body_visibilities
    diag.raw_hand_landmarks = hand_landmarks

    state = {"pose_dets": kept_pose_dets, "palm_dets": kept_palm_dets, "hand_diag": hand_diag}
    return body_landmarks, body_visibilities, hand_landmarks, hand_flags, state, diag
