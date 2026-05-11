"""Skeleton definitions, color maps, Catmull-Rom spline rendering, and overlay drawing."""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Skeleton topology — 12-keypoint arm scheme
# ---------------------------------------------------------------------------

# Arm chains drawn as Catmull-Rom splines (indices into the 12 arm keypoints).
BODY_CHAINS = [
    ([0, 2, 4], "left_arm"),
    ([1, 3, 5], "right_arm"),
]
# Arm segments drawn as straight lines.
BODY_SEGMENTS = [
    ((0, 1), "shoulder"),
    ((4, 6), "left_arm"),
    ((4, 8), "left_arm"),
    ((4, 10), "left_arm"),
    ((6, 8), "left_arm"),
    ((5, 7), "right_arm"),
    ((5, 9), "right_arm"),
    ((5, 11), "right_arm"),
    ((7, 9), "right_arm"),
]

BODY_COLOR_MAP = {
    "shoulder": (0, 255, 255),
    "left_arm": (0, 200, 0),
    "right_arm": (0, 128, 255),
}

# ---------------------------------------------------------------------------
# Skeleton topology — 33-keypoint full body scheme
# ---------------------------------------------------------------------------

FULL_BODY_CHAINS = [
    ([11, 13, 15], "left_arm"),
    ([12, 14, 16], "right_arm"),
    ([23, 25, 27], "left_leg"),
    ([24, 26, 28], "right_leg"),
]
FULL_BODY_SEGMENTS = [
    # Torso
    ((11, 12), "torso"),
    ((11, 23), "torso"),
    ((12, 24), "torso"),
    ((23, 24), "torso"),
    # Wrist → finger connections
    ((15, 17), "left_arm"),
    ((15, 19), "left_arm"),
    ((15, 21), "left_arm"),
    ((17, 19), "left_arm"),
    ((16, 18), "right_arm"),
    ((16, 20), "right_arm"),
    ((16, 22), "right_arm"),
    ((18, 20), "right_arm"),
    # Feet
    ((27, 29), "left_leg"),
    ((29, 31), "left_leg"),
    ((27, 31), "left_leg"),
    ((28, 30), "right_leg"),
    ((30, 32), "right_leg"),
    ((28, 32), "right_leg"),
    # Face
    ((0, 1), "face"),
    ((1, 2), "face"),
    ((2, 3), "face"),
    ((3, 7), "face"),
    ((0, 4), "face"),
    ((4, 5), "face"),
    ((5, 6), "face"),
    ((6, 8), "face"),
    ((9, 10), "face"),
]

FULL_BODY_COLOR_MAP = {
    "torso": (0, 255, 255),
    "left_arm": (0, 200, 0),
    "right_arm": (0, 128, 255),
    "left_leg": (0, 200, 0),
    "right_leg": (0, 128, 255),
    "face": (180, 180, 180),
}

# ---------------------------------------------------------------------------
# Hand skeleton topology
# ---------------------------------------------------------------------------

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

# Cached (num_samples, endpoint) → (samples, 4) basis matrix.  Each row
# of the basis dotted with a (4, 2) control-point block produces one
# curve sample, so a whole segment is a single matmul.
_CATMULL_BASIS_CACHE: dict[tuple[int, bool], np.ndarray] = {}


def _catmull_basis(num_samples: int, endpoint: bool) -> np.ndarray:
    key = (num_samples, endpoint)
    cached = _CATMULL_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    # float32 throughout: landmark inputs are already float32 and the
    # caller casts the spline output to int32 for cv2 drawing, so float64
    # adds a copy with no precision benefit at sub-pixel scales.
    t = np.linspace(0.0, 1.0, num_samples, endpoint=endpoint, dtype=np.float32)
    t2 = t * t
    t3 = t2 * t
    # Catmull-Rom basis (tension=0.5):
    #   P(t) = 0.5 * [(2)P1 + (-P0 + P2) t + (2P0 - 5P1 + 4P2 - P3) t^2
    #                + (-P0 + 3P1 - 3P2 + P3) t^3]
    basis = 0.5 * np.column_stack(
        [
            -t + 2.0 * t2 - t3,  # P0 weight
            2.0 - 5.0 * t2 + 3.0 * t3,  # P1 weight
            t + 4.0 * t2 - 3.0 * t3,  # P2 weight
            -t2 + t3,  # P3 weight
        ]
    ).astype(np.float32, copy=False)
    _CATMULL_BASIS_CACHE[key] = basis
    return basis


def catmull_rom_spline(points, num_samples=20):
    """Generate a smooth curve through 2-D control points using Catmull-Rom interpolation.

    Phantom endpoints are reflected from the boundary for natural curve extension.
    """
    n = len(points)
    if n < 2:
        return points
    points = np.asarray(points, dtype=np.float32)
    if n == 2:
        # Linear interpolation between the two control points.
        t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)[:, None]
        return (1.0 - t) * points[0] + t * points[1]

    # Reflect phantom endpoints for natural curve extension.
    p = np.empty((n + 2, points.shape[1]), dtype=points.dtype)
    p[0] = 2.0 * points[0] - points[1]
    p[1:-1] = points
    p[-1] = 2.0 * points[-1] - points[-2]

    n_segments = n - 1
    # Interior segments use endpoint=False so adjacent segments don't repeat
    # the shared knot; the last segment includes its endpoint.
    interior = _catmull_basis(num_samples, endpoint=False)
    last = _catmull_basis(num_samples, endpoint=True) if n_segments >= 1 else interior

    out_len = (n_segments - 1) * num_samples + num_samples
    result = np.empty((out_len, points.shape[1]), dtype=points.dtype)
    pos = 0
    for i in range(n_segments):
        control = p[i : i + 4]  # (4, dim)
        basis = last if i == n_segments - 1 else interior
        seg = basis @ control
        seg_len = seg.shape[0]
        result[pos : pos + seg_len] = seg
        pos += seg_len
    return result[:pos]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _body_topology(n_keypoints):
    """Return (chains, segments, blended_color_map) for the given landmark count.

    The colour map is the pre-blended variant so callers can index it
    directly in the per-limb hot loop.
    """
    if n_keypoints == 33:
        return FULL_BODY_CHAINS, FULL_BODY_SEGMENTS, _FULL_BODY_BLENDED
    return BODY_CHAINS, BODY_SEGMENTS, _BODY_BLENDED


_LIMB_ALPHA = 0.6


def _blended_color(color):
    """Return ``color`` darkened so direct overlay on a mid-grey background
    matches the previous 60 %-alpha look.

    Used to build the module-level ``_*_BLENDED`` lookup tables; the per-
    frame drawing path indexes those tables directly instead of re-running
    the multiply/round per limb.
    """
    return tuple(int(round(c * _LIMB_ALPHA)) for c in color)


_BODY_BLENDED = {k: _blended_color(v) for k, v in BODY_COLOR_MAP.items()}
_FULL_BODY_BLENDED = {k: _blended_color(v) for k, v in FULL_BODY_COLOR_MAP.items()}
_HAND_BLENDED = {k: _blended_color(v) for k, v in HAND_COLOR_MAP.items()}


def draw_body_landmarks(img, body_landmarks, body_visibilities, visibility_threshold=0.5):
    """Draw body landmarks with curved splines for joint chains.

    Automatically selects 12-keypoint arm topology or 33-keypoint full
    body topology based on the shape of the first landmark array.

    Limbs are drawn directly on the frame at the pre-multiplied limb
    colour (formerly: copy + 60/40 addWeighted blend).  Drops the
    per-frame frame-sized buffer copy and the full-frame addWeighted
    pass, which together dominated the original cost.
    """
    if not body_landmarks:
        return img

    n_kp = body_landmarks[0].shape[0]
    chains, segments, color_map = _body_topology(n_kp)

    for landmarks, visibility in zip(body_landmarks, body_visibilities, strict=False):
        points = landmarks[:, :2]
        # Convert to Python lists once — the chain/segment/dot loops below
        # all read by index, and per-iteration numpy → Python int conversion
        # was a measurable cost at 33 keypoints × multiple call sites.
        pts_list = points.astype(np.int32, copy=False).tolist()
        vis_list = (visibility > visibility_threshold).tolist()

        for chain_indices, group in chains:
            if all(vis_list[i] for i in chain_indices):
                chain_pts = points[chain_indices]
                curve = catmull_rom_spline(chain_pts, num_samples=20).astype(np.int32)
                cv2.polylines(img, [curve], False, color_map[group], 2, cv2.LINE_AA)

        for (i, j), group in segments:
            if vis_list[i] and vis_list[j]:
                cv2.line(img, pts_list[i], pts_list[j], color_map[group], 2, cv2.LINE_AA)

        for idx, visible in enumerate(vis_list):
            if visible:
                pt = pts_list[idx]
                cv2.circle(img, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_hand_landmarks(img, hand_landmarks):
    """Draw hand landmarks with curved splines for finger chains."""
    if not hand_landmarks:
        return img

    for landmarks in hand_landmarks:
        points = landmarks[:, :2]
        # Single conversion shared across all three loops below; index lookups
        # into a Python list are far cheaper than per-element ``int(numpy_scalar)``.
        pts_list = points.astype(np.int32, copy=False).tolist()

        for chain_indices, group in HAND_CHAINS:
            chain_pts = points[chain_indices]
            curve = catmull_rom_spline(chain_pts, num_samples=20).astype(np.int32)
            cv2.polylines(img, [curve], False, _HAND_BLENDED[group], 2, cv2.LINE_AA)

        for (i, j), group in HAND_SEGMENTS:
            cv2.line(img, pts_list[i], pts_list[j], _HAND_BLENDED[group], 2, cv2.LINE_AA)

        for idx, pt in enumerate(pts_list):
            cv2.circle(img, pt, 3, HAND_KEYPOINT_COLORS[idx], -1, cv2.LINE_AA)
            cv2.circle(img, pt, 3, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_arm_hand_bridges(img, body_landmarks, hand_landmarks, matches):
    """Draw bridge lines connecting arm wrists to hand wrists."""
    for arm_idx, wrist_kp, hand_idx in matches:
        arm_wrist = body_landmarks[arm_idx][wrist_kp, :2]
        hand_wrist = hand_landmarks[hand_idx][0, :2]
        cv2.line(
            img,
            (int(arm_wrist[0]), int(arm_wrist[1])),
            (int(hand_wrist[0]), int(hand_wrist[1])),
            BRIDGE_COLOR,
            2,
            cv2.LINE_AA,
        )
    return img
