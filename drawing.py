"""Skeleton definitions, color maps, Catmull-Rom spline rendering, and overlay drawing."""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

# Arm chains drawn as Catmull-Rom splines (indices into the 12 arm keypoints).
BODY_CHAINS = [
    ([0, 2, 4], "left_arm"),
    ([1, 3, 5], "right_arm"),
]
# Arm segments drawn as straight lines.
BODY_SEGMENTS = [
    ((0, 1), "shoulder"),
    ((4, 6), "left_arm"), ((4, 8), "left_arm"), ((4, 10), "left_arm"), ((6, 8), "left_arm"),
    ((5, 7), "right_arm"), ((5, 9), "right_arm"), ((5, 11), "right_arm"), ((7, 9), "right_arm"),
]

# Hand finger chains drawn as Catmull-Rom splines (indices into the 21 hand keypoints).
HAND_CHAINS = [
    ([0, 1, 2, 3, 4], "thumb"),
    ([0, 5, 6, 7, 8], "index"),
    ([0, 9, 10, 11, 12], "middle"),
    ([0, 13, 14, 15, 16], "ring"),
    ([0, 17, 18, 19, 20], "pinky"),
]
# Palm cross-connections drawn as straight lines.
HAND_SEGMENTS = [
    ((5, 9), "palm"),
    ((9, 13), "palm"),
    ((13, 17), "palm"),
    ((1, 5), "palm"),
    ((2, 5), "palm"),
]

# ---------------------------------------------------------------------------
# Color maps (BGR)
# ---------------------------------------------------------------------------

BODY_COLOR_MAP = {
    "shoulder": (0, 255, 255),
    "left_arm": (0, 200, 0),
    "right_arm": (0, 128, 255),
}
HAND_COLOR_MAP = {
    "thumb": (0, 0, 255),
    "index": (0, 128, 255),
    "middle": (0, 255, 0),
    "ring": (255, 128, 0),
    "pinky": (255, 0, 128),
    "palm": (180, 180, 180),
}
BRIDGE_COLOR = (0, 255, 255)

HAND_KEYPOINT_COLORS = (
    [(180, 180, 180)]
    + [(0, 0, 255)] * 4
    + [(0, 128, 255)] * 4
    + [(0, 255, 0)] * 4
    + [(255, 128, 0)] * 4
    + [(255, 0, 128)] * 4
)

# ---------------------------------------------------------------------------
# Catmull-Rom spline
# ---------------------------------------------------------------------------


def catmull_rom_spline(points, num_samples=20):
    """Generate a smooth curve through 2-D control points using Catmull-Rom interpolation.

    Phantom endpoints are reflected from the boundary for natural curve extension.
    """
    n = len(points)
    if n < 2:
        return points
    if n == 2:
        t = np.linspace(0, 1, num_samples).reshape(-1, 1)
        return (1 - t) * points[0] + t * points[1]

    p = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])

    result = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        is_last = i == len(p) - 3
        t = np.linspace(0, 1, num_samples, endpoint=is_last).reshape(-1, 1)
        t2, t3 = t * t, t * t * t
        pts = 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        result.append(pts)

    return np.vstack(result)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_body_landmarks(img, body_landmarks, body_visibilities, visibility_threshold=0.5):
    """Draw arm landmarks with curved splines for joint chains."""
    if not body_landmarks:
        return img

    img_limbs = np.copy(img)

    for landmarks, visibility in zip(body_landmarks, body_visibilities):
        points = landmarks[:, :2]

        for chain_indices, group in BODY_CHAINS:
            if all(visibility[i] > visibility_threshold for i in chain_indices):
                chain_pts = points[chain_indices]
                curve = catmull_rom_spline(chain_pts, num_samples=20).astype(np.int32)
                cv2.polylines(img_limbs, [curve], False, BODY_COLOR_MAP[group], 2, cv2.LINE_AA)

        for (i, j), group in BODY_SEGMENTS:
            if visibility[i] > visibility_threshold and visibility[j] > visibility_threshold:
                cv2.line(img_limbs, tuple(points[i].astype(np.int32)),
                         tuple(points[j].astype(np.int32)), BODY_COLOR_MAP[group], 2, cv2.LINE_AA)

        for idx, pt in enumerate(points):
            if visibility[idx] > visibility_threshold:
                pt_int = tuple(pt.astype(np.int32))
                cv2.circle(img, pt_int, 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt_int, 4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def draw_hand_landmarks(img, hand_landmarks):
    """Draw hand landmarks with curved splines for finger chains."""
    if not hand_landmarks:
        return img

    img_limbs = np.copy(img)

    for landmarks in hand_landmarks:
        points = landmarks[:, :2]

        for chain_indices, group in HAND_CHAINS:
            chain_pts = points[chain_indices]
            curve = catmull_rom_spline(chain_pts, num_samples=20).astype(np.int32)
            cv2.polylines(img_limbs, [curve], False, HAND_COLOR_MAP[group], 2, cv2.LINE_AA)

        for (i, j), group in HAND_SEGMENTS:
            cv2.line(img_limbs, tuple(points[i].astype(np.int32)),
                     tuple(points[j].astype(np.int32)), HAND_COLOR_MAP[group], 2, cv2.LINE_AA)

        for idx, pt in enumerate(points):
            pt_int = tuple(pt.astype(np.int32))
            cv2.circle(img, pt_int, 3, HAND_KEYPOINT_COLORS[idx], -1, cv2.LINE_AA)
            cv2.circle(img, pt_int, 3, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def draw_arm_hand_bridges(img, body_landmarks, hand_landmarks, matches):
    """Draw bridge lines connecting arm wrists to hand wrists."""
    for arm_idx, wrist_kp, hand_idx in matches:
        arm_wrist = body_landmarks[arm_idx][wrist_kp, :2].astype(np.int32)
        hand_wrist = hand_landmarks[hand_idx][0, :2].astype(np.int32)
        cv2.line(img, tuple(arm_wrist), tuple(hand_wrist), BRIDGE_COLOR, 2, cv2.LINE_AA)
    return img
