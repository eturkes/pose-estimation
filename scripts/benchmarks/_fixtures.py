"""Synthetic inputs shared by the benchmark suite.

All builders are deterministic (seeded NumPy RNG) so benchmark runs are
reproducible.  Shapes and value ranges mimic the pipeline's real
distributions — landmark coordinates in pixel space, keypoint counts
matching the MediaPipe schemes, detection keypoint counts matching
palm/pose SSDs.
"""

from __future__ import annotations

import numpy as np

FRAME_W = 1280
FRAME_H = 720

# Landmark counts
ARM_KP = 12
BODY_KP = 33
HAND_KP = 21
POSE_DET_KP = 4
PALM_DET_KP = 7

# Landmark model raw output counts
POSE_LM_RAW = 39  # before slice down to 12 or 33


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Keypoint arrays
# ---------------------------------------------------------------------------


def make_body_landmarks(n_kp: int = ARM_KP, seed: int = 1) -> np.ndarray:
    """One body: (n_kp, 3) in pixel coordinates."""
    r = rng(seed)
    xs = r.uniform(100, FRAME_W - 100, n_kp)
    ys = r.uniform(100, FRAME_H - 100, n_kp)
    zs = r.uniform(-50, 50, n_kp)
    return np.stack([xs, ys, zs], axis=1).astype(np.float64)


def make_hand_landmarks(seed: int = 2) -> np.ndarray:
    """One hand: (21, 3) in pixel coordinates, clustered tightly."""
    r = rng(seed)
    cx = r.uniform(200, FRAME_W - 200)
    cy = r.uniform(200, FRAME_H - 200)
    xs = cx + r.normal(0, 40, HAND_KP)
    ys = cy + r.normal(0, 40, HAND_KP)
    zs = r.normal(0, 20, HAND_KP)
    return np.stack([xs, ys, zs], axis=1).astype(np.float64)


def make_body_list(n_bodies: int = 1, n_kp: int = ARM_KP, seed: int = 3) -> list[np.ndarray]:
    return [make_body_landmarks(n_kp=n_kp, seed=seed + i) for i in range(n_bodies)]


def make_hand_list(n_hands: int = 2, seed: int = 4) -> list[np.ndarray]:
    return [make_hand_landmarks(seed=seed + i) for i in range(n_hands)]


def make_visibilities(n_kp: int, seed: int = 5) -> np.ndarray:
    r = rng(seed)
    return r.uniform(0.3, 1.0, n_kp).astype(np.float64)


def make_visibility_list(n_bodies: int, n_kp: int, seed: int = 6) -> list[np.ndarray]:
    return [make_visibilities(n_kp, seed + i) for i in range(n_bodies)]


def make_hand_flags(n_hands: int, seed: int = 7) -> list[float]:
    r = rng(seed)
    return r.uniform(0.6, 0.95, n_hands).tolist()


# ---------------------------------------------------------------------------
# Detection dicts
# ---------------------------------------------------------------------------


def make_detection(num_keypoints: int, seed: int = 10) -> dict:
    """One detection dict in normalised [0, 1] coords."""
    r = rng(seed)
    cx, cy = r.uniform(0.2, 0.8, 2)
    size = r.uniform(0.05, 0.2)
    box = np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], dtype=np.float32)
    keypoints = r.uniform(0.2, 0.8, (num_keypoints, 2)).astype(np.float32)
    return {"box": box, "keypoints": keypoints, "score": float(r.uniform(0.6, 0.95))}


def make_detections(n: int, num_keypoints: int, seed: int = 11) -> list[dict]:
    return [make_detection(num_keypoints, seed=seed + i) for i in range(n)]


def make_raw_ssd_outputs(n_anchors: int, num_keypoints: int, seed: int = 12):
    """Synthetic raw SSD outputs: (raw_boxes, raw_scores, anchors).

    Shapes match what ``decode_detections`` expects.  A fraction of
    anchors are given high scores so NMS actually has work to do.
    """
    r = rng(seed)
    values_per_anchor = 4 + num_keypoints * 2
    raw_boxes = r.uniform(-50, 50, (n_anchors, values_per_anchor)).astype(np.float32)
    raw_scores = r.uniform(-8, -2, (n_anchors, 1)).astype(np.float32)
    # Boost a handful to positive logits so the score threshold passes
    hot = r.choice(n_anchors, size=max(2, n_anchors // 40), replace=False)
    raw_scores[hot, 0] = r.uniform(1.0, 3.0, len(hot)).astype(np.float32)
    anchors = r.uniform(0.05, 0.95, (n_anchors, 2)).astype(np.float32)
    return raw_boxes, raw_scores, anchors


def make_nms_boxes(n: int, seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    r = rng(seed)
    # Create clusters so suppression has work to do
    centers = r.uniform(0.1, 0.9, (max(1, n // 5), 2))
    boxes = []
    for _ in range(n):
        c = centers[r.integers(0, len(centers))]
        half = r.uniform(0.02, 0.08)
        boxes.append([c[0] - half, c[1] - half, c[0] + half, c[1] + half])
    boxes_arr = np.array(boxes, dtype=np.float32)
    scores = r.uniform(0.3, 0.95, n).astype(np.float32)
    return boxes_arr, scores


# ---------------------------------------------------------------------------
# Frames for image-ops
# ---------------------------------------------------------------------------


def make_frame(h: int = FRAME_H, w: int = FRAME_W, seed: int = 20) -> np.ndarray:
    """Fake BGR frame; synthetic gradient so cv2 resize/convert do real work."""
    r = rng(seed)
    return r.integers(0, 255, (h, w, 3), dtype=np.uint8)
