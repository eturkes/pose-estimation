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
        while (layer_id + same_stride_count < len(strides)
               and strides[layer_id + same_stride_count] == stride):
            same_stride_count += 1
        grid_size = input_size // stride
        for y in range(grid_size):
            for x in range(grid_size):
                for _ in range(same_stride_count):
                    for _ in range(2):
                        anchors.append([(x + 0.5) / grid_size, (y + 0.5) / grid_size])
        layer_id += same_stride_count
    return np.array(anchors, dtype=np.float32)


def nms(boxes, scores, iou_threshold=0.3):
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def decode_detections(raw_boxes, raw_scores, anchors, input_size, num_keypoints,
                      score_threshold=0.5, iou_threshold=0.3):
    """Decode detection model outputs into detection results.

    Works for both pose detection (4 keypoints, 12 values) and palm detection
    (7 keypoints, 18 values). Each detection contains a bounding box, confidence
    score, and keypoints in normalized [0, 1] coordinates.
    """
    values_per_anchor = 4 + num_keypoints * 2
    scores = 1.0 / (1.0 + np.exp(-raw_scores.reshape(-1)))

    mask = scores >= score_threshold
    if not np.any(mask):
        return []

    filtered_scores = scores[mask]
    filtered_boxes = raw_boxes.reshape(-1, values_per_anchor)[mask]
    filtered_anchors = anchors[mask]

    cx = filtered_boxes[:, 0] / input_size + filtered_anchors[:, 0]
    cy = filtered_boxes[:, 1] / input_size + filtered_anchors[:, 1]
    w = filtered_boxes[:, 2] / input_size
    h = filtered_boxes[:, 3] / input_size

    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    keypoints = np.zeros((len(filtered_boxes), num_keypoints, 2))
    for k in range(num_keypoints):
        keypoints[:, k, 0] = filtered_boxes[:, 4 + 2 * k] / input_size + filtered_anchors[:, 0]
        keypoints[:, k, 1] = filtered_boxes[:, 4 + 2 * k + 1] / input_size + filtered_anchors[:, 1]

    indices = nms(boxes, filtered_scores, iou_threshold)

    detections = []
    for i in indices:
        detections.append({
            "box": boxes[i],
            "score": filtered_scores[i],
            "keypoints": keypoints[i],
        })
    return detections
