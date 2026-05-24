"""COCO-WholeBody → MediaPipe keypoint schema mapping.

Translates rtmlib output (n_persons, n_kps, 2) + scores into the
(body_landmarks, body_visibilities, hand_landmarks, matches) tuple
expected by export.frame_to_rows().
"""

import numpy as np

from .processing import (
    TRACKING_HANDS,
    TRACKING_HANDS_ARMS,
)

# ---------------------------------------------------------------------------
# COCO-WholeBody 133 index ranges
# ---------------------------------------------------------------------------
_COCO_BODY_END = 17
_COCO_FEET_START = 17
_COCO_FEET_END = 23
_COCO_FACE_START = 23
_COCO_LHAND_START = 91
_COCO_LHAND_END = 112
_COCO_RHAND_START = 112
_COCO_RHAND_END = 133

# Hand sub-array offsets for MCP (base) joints
_HAND_INDEX_MCP = 5
_HAND_MIDDLE_MCP = 9
_HAND_PINKY_MCP = 17
_HAND_THUMB_CMC = 1

# ---------------------------------------------------------------------------
# Mapping tables: COCO-WholeBody → MediaPipe ARM (12 keypoints)
# ---------------------------------------------------------------------------
# (mp_arm_idx, coco_133_idx)
_COCO_TO_ARM = [
    (0, 5),  # left_shoulder
    (1, 6),  # right_shoulder
    (2, 7),  # left_elbow
    (3, 8),  # right_elbow
    (4, 9),  # left_wrist
    (5, 10),  # right_wrist
    (6, 96),  # left_index_base  (left hand index MCP = 91+5)
    (7, 117),  # right_index_base (right hand index MCP = 112+5)
    (8, 100),  # left_middle_base (left hand middle MCP = 91+9)
    (9, 121),  # right_middle_base (right hand middle MCP = 112+9)
    (10, 108),  # left_pinky_base  (left hand pinky MCP = 91+17)
    (11, 129),  # right_pinky_base (right hand pinky MCP = 112+17)
]

# ---------------------------------------------------------------------------
# Mapping table: COCO-WholeBody → MediaPipe BODY (33 keypoints)
# ---------------------------------------------------------------------------
# (mp_body_idx, coco_133_idx_or_None)
# None means the keypoint needs special handling (face-derived or hand-derived)
_COCO_TO_BODY_DIRECT = [
    (0, 0),  # nose
    (2, 1),  # left_eye ← COCO left_eye
    (5, 2),  # right_eye ← COCO right_eye
    (7, 3),  # left_ear
    (8, 4),  # right_ear
    (11, 5),  # left_shoulder
    (12, 6),  # right_shoulder
    (13, 7),  # left_elbow
    (14, 8),  # right_elbow
    (15, 9),  # left_wrist
    (16, 10),  # right_wrist
    (23, 11),  # left_hip
    (24, 12),  # right_hip
    (25, 13),  # left_knee
    (26, 14),  # right_knee
    (27, 15),  # left_ankle
    (28, 16),  # right_ankle
    (29, 19),  # left_heel ← COCO feet index 19
    (30, 22),  # right_heel ← COCO feet index 22
    (31, 17),  # left_foot_index ← COCO left_big_toe (closest)
    (32, 20),  # right_foot_index ← COCO right_big_toe
]

# Face-derived MP keypoints: (mp_body_idx, coco_face_subindex)
# COCO face uses 68-point iBUG layout starting at index 23.
_FACE_OFFSET = 23
_COCO_TO_BODY_FACE = [
    (1, 36),  # left_eye_inner ← face inner left eye corner
    (3, 39),  # left_eye_outer ← face outer left eye corner
    (4, 42),  # right_eye_inner ← face inner right eye corner
    (6, 45),  # right_eye_outer ← face outer right eye corner
    (9, 48),  # mouth_left ← face left mouth corner
    (10, 54),  # mouth_right ← face right mouth corner
]

# Hand-derived MP wrist keypoints: (mp_body_idx, hand_base, hand_sub_offset)
_COCO_TO_BODY_HAND = [
    (17, _COCO_LHAND_START, _HAND_PINKY_MCP),  # left_pinky (wrist)
    (18, _COCO_RHAND_START, _HAND_PINKY_MCP),  # right_pinky (wrist)
    (19, _COCO_LHAND_START, _HAND_INDEX_MCP),  # left_index (wrist)
    (20, _COCO_RHAND_START, _HAND_INDEX_MCP),  # right_index (wrist)
    (21, _COCO_LHAND_START, _HAND_THUMB_CMC),  # left_thumb (wrist)
    (22, _COCO_RHAND_START, _HAND_THUMB_CMC),  # right_thumb (wrist)
]

# Wrist keypoint indices in output arrays (for constructing matches)
_ARM_WRIST_LEFT = 4
_ARM_WRIST_RIGHT = 5
_BODY_WRIST_LEFT = 15
_BODY_WRIST_RIGHT = 16


def coco_to_mediapipe(keypoints, scores, n_kps, tracking):
    """Map rtmlib COCO-WholeBody output to MediaPipe export schema.

    Parameters
    ----------
    keypoints : ndarray, shape (n_persons, n_kps, 2)
        Pixel-space (x, y) from rtmlib pose tracker.
    scores : ndarray, shape (n_persons, n_kps)
        Per-keypoint confidence scores.
    n_kps : int
        Number of keypoints per person (133 or 17).
    tracking : str
        One of "hands-arms", "body", "hands".

    Returns
    -------
    body_landmarks : list[ndarray (N_body, 3)]
        Per-person body landmarks in pixel-space (x, y, z=0).
    body_visibilities : list[ndarray (N_body,)]
        Per-person per-keypoint confidence.
    hand_landmarks : list[ndarray (21, 3)]
        Flat list of all hand landmark arrays.
    matches : list[tuple(person_idx, wrist_kp_idx, hand_list_idx)]
        Maps each hand in hand_landmarks back to its person + wrist slot.
    """
    if keypoints is None or keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return [], [], [], []

    n_persons = keypoints.shape[0]

    if n_kps == 133:
        return _map_133(keypoints, scores, n_persons, tracking)
    if n_kps == 17:
        return _map_17(keypoints, scores, n_persons, tracking)

    return [], [], [], []


def _xy_to_xyz(xy):
    """Expand (N, 2) → (N, 3) with z=0."""
    return np.column_stack([xy, np.zeros(xy.shape[0], dtype=xy.dtype)])


# ---------------------------------------------------------------------------
# 133-keypoint wholebody mapping
# ---------------------------------------------------------------------------


def _map_133(keypoints, scores, n_persons, tracking):
    if tracking == TRACKING_HANDS:
        return _map_133_hands(keypoints, scores, n_persons)
    if tracking == TRACKING_HANDS_ARMS:
        return _map_133_arms(keypoints, scores, n_persons)
    return _map_133_body(keypoints, scores, n_persons)


def _map_133_hands(keypoints, scores, n_persons):
    """Hands-only mode: no body, just hand landmarks assigned by x-position."""
    hand_landmarks = []
    for pi in range(n_persons):
        lh = keypoints[pi, _COCO_LHAND_START:_COCO_LHAND_END]
        rh = keypoints[pi, _COCO_RHAND_START:_COCO_RHAND_END]
        lh_score = scores[pi, _COCO_LHAND_START:_COCO_LHAND_END].mean()
        rh_score = scores[pi, _COCO_RHAND_START:_COCO_RHAND_END].mean()
        if lh_score > 0.1:
            hand_landmarks.append(_xy_to_xyz(lh))
        if rh_score > 0.1:
            hand_landmarks.append(_xy_to_xyz(rh))
    return [], [], hand_landmarks, []


def _map_133_arms(keypoints, scores, n_persons):
    """Hands-arms mode: 12 arm keypoints + matched hands."""
    body_landmarks = []
    body_visibilities = []
    hand_landmarks = []
    matches = []

    for pi in range(n_persons):
        kps = keypoints[pi]
        sc = scores[pi]

        arm = np.zeros((12, 3), dtype=np.float64)
        arm_vis = np.zeros(12, dtype=np.float64)

        for mp_idx, coco_idx in _COCO_TO_ARM:
            arm[mp_idx, :2] = kps[coco_idx]
            arm_vis[mp_idx] = sc[coco_idx]

        body_landmarks.append(arm)
        body_visibilities.append(arm_vis)

        # Extract hands and build matches
        lh = kps[_COCO_LHAND_START:_COCO_LHAND_END]
        rh = kps[_COCO_RHAND_START:_COCO_RHAND_END]
        lh_mean_score = sc[_COCO_LHAND_START:_COCO_LHAND_END].mean()
        rh_mean_score = sc[_COCO_RHAND_START:_COCO_RHAND_END].mean()

        if lh_mean_score > 0.1:
            hidx = len(hand_landmarks)
            hand_landmarks.append(_xy_to_xyz(lh))
            matches.append((pi, _ARM_WRIST_LEFT, hidx))

        if rh_mean_score > 0.1:
            hidx = len(hand_landmarks)
            hand_landmarks.append(_xy_to_xyz(rh))
            matches.append((pi, _ARM_WRIST_RIGHT, hidx))

    return body_landmarks, body_visibilities, hand_landmarks, matches


def _map_133_body(keypoints, scores, n_persons):
    """Body mode: 33 MediaPipe body keypoints + matched hands."""
    body_landmarks = []
    body_visibilities = []
    hand_landmarks = []
    matches = []

    for pi in range(n_persons):
        kps = keypoints[pi]
        sc = scores[pi]

        body = np.zeros((33, 3), dtype=np.float64)
        body_vis = np.zeros(33, dtype=np.float64)

        # Direct body mappings
        for mp_idx, coco_idx in _COCO_TO_BODY_DIRECT:
            body[mp_idx, :2] = kps[coco_idx]
            body_vis[mp_idx] = sc[coco_idx]

        # Face-derived keypoints
        for mp_idx, face_sub in _COCO_TO_BODY_FACE:
            coco_idx = _FACE_OFFSET + face_sub
            body[mp_idx, :2] = kps[coco_idx]
            body_vis[mp_idx] = sc[coco_idx]

        # Hand-derived wrist orientation keypoints
        for mp_idx, hand_base, hand_offset in _COCO_TO_BODY_HAND:
            coco_idx = hand_base + hand_offset
            body[mp_idx, :2] = kps[coco_idx]
            body_vis[mp_idx] = sc[coco_idx]

        body_landmarks.append(body)
        body_visibilities.append(body_vis)

        # Extract hands and build matches
        lh = kps[_COCO_LHAND_START:_COCO_LHAND_END]
        rh = kps[_COCO_RHAND_START:_COCO_RHAND_END]
        lh_mean_score = sc[_COCO_LHAND_START:_COCO_LHAND_END].mean()
        rh_mean_score = sc[_COCO_RHAND_START:_COCO_RHAND_END].mean()

        if lh_mean_score > 0.1:
            hidx = len(hand_landmarks)
            hand_landmarks.append(_xy_to_xyz(lh))
            matches.append((pi, _BODY_WRIST_LEFT, hidx))

        if rh_mean_score > 0.1:
            hidx = len(hand_landmarks)
            hand_landmarks.append(_xy_to_xyz(rh))
            matches.append((pi, _BODY_WRIST_RIGHT, hidx))

    return body_landmarks, body_visibilities, hand_landmarks, matches


# ---------------------------------------------------------------------------
# 17-keypoint body-only mapping (RTMPose-M)
# ---------------------------------------------------------------------------

# COCO 17-kp layout is identical to the first 17 of COCO-WholeBody 133.
# Maps to body mode only (no hands, no finger bases).

_COCO17_TO_BODY_DIRECT = [
    (0, 0),  # nose
    (2, 1),  # left_eye
    (5, 2),  # right_eye
    (7, 3),  # left_ear
    (8, 4),  # right_ear
    (11, 5),  # left_shoulder
    (12, 6),  # right_shoulder
    (13, 7),  # left_elbow
    (14, 8),  # right_elbow
    (15, 9),  # left_wrist
    (16, 10),  # right_wrist
    (23, 11),  # left_hip
    (24, 12),  # right_hip
    (25, 13),  # left_knee
    (26, 14),  # right_knee
    (27, 15),  # left_ankle
    (28, 16),  # right_ankle
]


def _map_17(keypoints, scores, n_persons, tracking):
    """17-keypoint body-only: maps to body mode, empty hands.

    Tracking mode is forced to body regardless of input — 17-kp models
    lack hand/finger keypoints needed for hands-arms or hands modes.
    """
    if tracking == TRACKING_HANDS:
        return [], [], [], []

    body_landmarks = []
    body_visibilities = []

    for pi in range(n_persons):
        kps = keypoints[pi]
        sc = scores[pi]

        if tracking == TRACKING_HANDS_ARMS:
            # Extract only the 6 arm keypoints available from 17-kp
            arm = np.zeros((12, 3), dtype=np.float64)
            arm_vis = np.zeros(12, dtype=np.float64)
            # shoulders(5,6)→(0,1), elbows(7,8)→(2,3), wrists(9,10)→(4,5)
            _17_to_arm = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]
            for mp_idx, coco_idx in _17_to_arm:
                arm[mp_idx, :2] = kps[coco_idx]
                arm_vis[mp_idx] = sc[coco_idx]
            body_landmarks.append(arm)
            body_visibilities.append(arm_vis)
        else:
            body = np.zeros((33, 3), dtype=np.float64)
            body_vis = np.zeros(33, dtype=np.float64)
            for mp_idx, coco_idx in _COCO17_TO_BODY_DIRECT:
                body[mp_idx, :2] = kps[coco_idx]
                body_vis[mp_idx] = sc[coco_idx]
            body_landmarks.append(body)
            body_visibilities.append(body_vis)

    return body_landmarks, body_visibilities, [], []


__all__ = ["coco_to_mediapipe"]
