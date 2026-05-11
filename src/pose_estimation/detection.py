"""Anchor generation, NMS, and detection decoding for SSD-based MediaPipe models."""

import numpy as np

PALM_INPUT_SIZE = 192
HAND_INPUT_SIZE = 224
POSE_INPUT_SIZE = 224
POSE_LM_INPUT_SIZE = 256


def generate_anchors(input_size, strides):
    """Generate SSD anchors for MediaPipe detection models.

    Layers with the same stride are grouped so that anchors at each grid cell
    are interleaved across sub-layers, matching the model's expected order.
    """
    anchors = []
    layer_id = 0
    while layer_id < len(strides):
        stride = strides[layer_id]
        same_stride_count = 0
        while (
            layer_id + same_stride_count < len(strides)
            and strides[layer_id + same_stride_count] == stride
        ):
            same_stride_count += 1
        grid_size = input_size // stride
        for y in range(grid_size):
            for x in range(grid_size):
                anchors.extend(
                    [[(x + 0.5) / grid_size, (y + 0.5) / grid_size]] * (same_stride_count * 2)
                )
        layer_id += same_stride_count
    return np.array(anchors, dtype=np.float32)


def nms(boxes, scores, iou_threshold=0.3):
    """Non-maximum suppression via a vectorised pairwise IoU matrix.

    The full (n, n) overlap matrix is computed once with broadcast
    minimum/maximum ops, after which the greedy keep/suppress sweep is
    just bool OR-reductions over rows.  Replaces the original
    per-iteration fancy-indexing + ``np.where`` filtering, which scaled
    poorly past a few dozen detections.
    """
    n = boxes.shape[0]
    if n == 0:
        return []

    order = scores.argsort()[::-1]
    sb = np.ascontiguousarray(boxes[order], dtype=np.float32)
    x1c = sb[:, 0:1]
    y1c = sb[:, 1:2]
    x2c = sb[:, 2:3]
    y2c = sb[:, 3:4]
    areas = ((sb[:, 2] - sb[:, 0]) * (sb[:, 3] - sb[:, 1]))[:, None]

    # Pairwise intersection width / height, clipped at zero
    inter_w = np.minimum(x2c, x2c.T)
    inter_w -= np.maximum(x1c, x1c.T)
    np.clip(inter_w, 0.0, None, out=inter_w)
    inter_h = np.minimum(y2c, y2c.T)
    inter_h -= np.maximum(y1c, y1c.T)
    np.clip(inter_h, 0.0, None, out=inter_h)
    inter_w *= inter_h  # reuse buffer
    intersection = inter_w
    union = areas + areas.T - intersection + 1e-6
    overlap = intersection > (union * float(iou_threshold))

    suppress = np.zeros(n, dtype=bool)
    keep_sorted = []
    for i in range(n):
        if suppress[i]:
            continue
        keep_sorted.append(i)
        suppress |= overlap[i]

    return [int(order[k]) for k in keep_sorted]


def decode_detections(
    raw_boxes,
    raw_scores,
    anchors,
    input_size,
    num_keypoints,
    score_threshold=0.5,
    iou_threshold=0.3,
):
    """Decode detection model outputs into detection results.

    Works for both pose detection (4 keypoints, 12 values) and palm detection
    (7 keypoints, 18 values). Each detection contains a bounding box, confidence
    score, and keypoints in normalized [0, 1] coordinates.

    Sigmoid is monotonic, so we apply the score threshold to raw logits and
    only run sigmoid on the surviving subset.  Keypoint decoding is a single
    broadcast add over an (n, k, 2) view rather than a Python loop.
    """
    values_per_anchor = 4 + num_keypoints * 2

    # Threshold on logits: sigmoid(x) >= t  <=>  x >= log(t / (1 - t))
    if 0.0 < score_threshold < 1.0:
        logit_thresh = float(np.log(score_threshold / (1.0 - score_threshold)))
    elif score_threshold <= 0.0:
        logit_thresh = -np.inf
    else:
        logit_thresh = np.inf

    raw_logits = raw_scores.reshape(-1)
    mask = raw_logits >= logit_thresh
    if not mask.any():
        return []

    filtered_logits = raw_logits[mask]
    # Sigmoid on the (small) surviving subset only
    filtered_scores = 1.0 / (1.0 + np.exp(-filtered_logits))

    filtered_boxes = raw_boxes.reshape(-1, values_per_anchor)[mask]
    filtered_anchors = anchors[mask]

    # In-place scale: all box + keypoint offsets share the same normalisation.
    # ``filtered_boxes`` is a fancy-indexed copy so this does not touch raw_boxes.
    filtered_boxes *= np.float32(1.0 / input_size)

    n = filtered_boxes.shape[0]
    cx = filtered_boxes[:, 0] + filtered_anchors[:, 0]
    cy = filtered_boxes[:, 1] + filtered_anchors[:, 1]
    w_half = filtered_boxes[:, 2] * 0.5
    h_half = filtered_boxes[:, 3] * 0.5

    boxes = np.empty((n, 4), dtype=filtered_boxes.dtype)
    np.subtract(cx, w_half, out=boxes[:, 0])
    np.subtract(cy, h_half, out=boxes[:, 1])
    np.add(cx, w_half, out=boxes[:, 2])
    np.add(cy, h_half, out=boxes[:, 3])

    # Single broadcast add over (n, k, 2) view; replaces the Python keypoint loop.
    keypoints = filtered_boxes[:, 4:].reshape(n, num_keypoints, 2) + filtered_anchors[:, None, :]

    indices = nms(boxes, filtered_scores, iou_threshold)

    return [
        {"box": boxes[i], "score": filtered_scores[i], "keypoints": keypoints[i]} for i in indices
    ]
