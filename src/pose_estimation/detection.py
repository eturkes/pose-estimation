"""Anchor generation, NMS, and detection decoding for SSD-based MediaPipe models."""

import math

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


def weighted_nms(boxes, scores, keypoints, iou_threshold=0.3):
    """Merge overlapping SSD detections using score-weighted coordinates.

    MediaPipe's pose and palm graphs use weighted non-maximum suppression:
    each highest-scoring detection gathers the remaining boxes that overlap
    it, then their boxes and keypoints are averaged with detection score as
    the weight.  The cluster retains its highest score.  Compared with hard
    suppression, this uses the redundant anchor predictions to stabilise the
    crop geometry passed to the landmark models.

    Returns ``(merged_boxes, merged_scores, merged_keypoints)`` in descending
    score order.
    """
    boxes = np.asarray(boxes)
    scores = np.asarray(scores).reshape(-1)
    keypoints = np.asarray(keypoints)
    n = boxes.shape[0]
    if n == 0:
        return (
            np.empty((0, 4), dtype=boxes.dtype),
            np.empty((0,), dtype=scores.dtype),
            np.empty((0, *keypoints.shape[1:]), dtype=keypoints.dtype),
        )
    if scores.shape[0] != n or keypoints.shape[0] != n:
        raise ValueError("boxes, scores, and keypoints must have the same first dimension")

    order = scores.argsort(kind="stable")[::-1]
    sorted_boxes = np.ascontiguousarray(boxes[order], dtype=np.float64)
    sorted_scores = scores[order].astype(np.float64, copy=False)
    sorted_keypoints = keypoints[order].astype(np.float64, copy=False)
    widths = np.maximum(0.0, sorted_boxes[:, 2] - sorted_boxes[:, 0])
    heights = np.maximum(0.0, sorted_boxes[:, 3] - sorted_boxes[:, 1])
    areas = widths * heights

    # MediaPipe compares the current highest-scoring box with all remaining
    # boxes, removes its overlap cluster, and repeats.  Computing just that
    # one IoU vector per iteration preserves those exact clustering semantics
    # while keeping peak memory O(n), rather than materialising several O(n²)
    # float matrices (hundreds of MiB at the 2,254-anchor pose-model limit).
    remaining = np.arange(n)
    merged_boxes = np.empty((n, 4), dtype=np.float64)
    merged_scores = np.empty(n, dtype=np.float64)
    merged_keypoints = np.empty((n, *keypoints.shape[1:]), dtype=np.float64)
    n_merged = 0
    while remaining.size:
        top = int(remaining[0])
        candidate_boxes = sorted_boxes[remaining]
        inter_w = np.minimum(candidate_boxes[:, 2], sorted_boxes[top, 2])
        inter_w -= np.maximum(candidate_boxes[:, 0], sorted_boxes[top, 0])
        np.maximum(inter_w, 0.0, out=inter_w)
        inter_h = np.minimum(candidate_boxes[:, 3], sorted_boxes[top, 3])
        inter_h -= np.maximum(candidate_boxes[:, 1], sorted_boxes[top, 1])
        np.maximum(inter_h, 0.0, out=inter_h)
        intersection = inter_w * inter_h
        union = areas[remaining] + areas[top] - intersection
        overlap = np.zeros(remaining.size, dtype=np.float64)
        np.divide(intersection, union, out=overlap, where=union > 0.0)
        cluster_mask = overlap > float(iou_threshold)
        # The top box can be degenerate (IoU 0 with itself), so retain it
        # explicitly to guarantee progress on malformed model output.
        cluster_mask[0] = True
        cluster = remaining[cluster_mask]
        remaining = remaining[~cluster_mask]

        weights = sorted_scores[cluster]
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            weights = np.ones(cluster.size, dtype=np.float64)
            total = float(weights.size)
        merged_boxes[n_merged] = np.average(sorted_boxes[cluster], axis=0, weights=weights)
        merged_keypoints[n_merged] = np.average(sorted_keypoints[cluster], axis=0, weights=weights)
        merged_scores[n_merged] = sorted_scores[top]
        n_merged += 1

    return (
        merged_boxes[:n_merged].astype(boxes.dtype, copy=False),
        merged_scores[:n_merged].astype(scores.dtype, copy=False),
        merged_keypoints[:n_merged].astype(keypoints.dtype, copy=False),
    )


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

    filtered_logits = raw_logits[mask].astype(np.float64, copy=False)
    # Sigmoid on the (small) surviving subset only
    filtered_scores = 1.0 / (1.0 + np.exp(-np.clip(filtered_logits, -100.0, 100.0)))

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

    boxes, filtered_scores, keypoints = weighted_nms(
        boxes, filtered_scores, keypoints, iou_threshold
    )

    return [
        {"box": box, "score": score, "keypoints": points}
        for box, score, points in zip(boxes, filtered_scores, keypoints, strict=True)
    ]
