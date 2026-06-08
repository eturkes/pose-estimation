"""CSV export of per-frame landmark data for downstream feature selection."""

import csv
import pathlib

import numpy as np

from .processing import (
    TRACKING_BODY,
    TRACKING_HANDS,
    TRACKING_HANDS_ARMS,
    WRIST_KPS_12,
    WRIST_KPS_33,
)

ARM_KEYPOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index_base",
    "right_index_base",
    "left_middle_base",
    "right_middle_base",
    "left_pinky_base",
    "right_pinky_base",
]

BODY_KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

HAND_KEYPOINT_COUNT = 21


def _body_keypoint_names(tracking):
    """Return (prefix, names) for the body landmark columns."""
    if tracking == TRACKING_BODY:
        return "body", BODY_KEYPOINT_NAMES
    return "arm", ARM_KEYPOINT_NAMES


def wrist_to_side(tracking):
    """Return a dict mapping wrist keypoint index to 'left'/'right'."""
    if tracking == TRACKING_BODY:
        return {WRIST_KPS_33[0]: "left", WRIST_KPS_33[1]: "right"}
    return {WRIST_KPS_12[0]: "left", WRIST_KPS_12[1]: "right"}


def make_csv_header(tracking=TRACKING_HANDS_ARMS):
    """Return the full list of column names for the given tracking mode."""
    cols = ["video", "frame_idx", "timestamp_sec", "person_idx"]

    if tracking != TRACKING_HANDS:
        prefix, names = _body_keypoint_names(tracking)
        for name in names:
            cols.extend(
                [
                    f"{prefix}_{name}_x",
                    f"{prefix}_{name}_y",
                    f"{prefix}_{name}_z",
                    f"{prefix}_{name}_vis",
                ]
            )

    for side in ("left", "right"):
        for i in range(HAND_KEYPOINT_COUNT):
            cols.extend([f"{side}_hand_{i}_x", f"{side}_hand_{i}_y", f"{side}_hand_{i}_z"])

    return cols


def _blank_hand_side(row, side):
    """Fill one hand side with empty strings."""
    for i in range(HAND_KEYPOINT_COUNT):
        row[f"{side}_hand_{i}_x"] = ""
        row[f"{side}_hand_{i}_y"] = ""
        row[f"{side}_hand_{i}_z"] = ""


def _fill_hand_side(row, side, hlm, frame_h, frame_w):
    """Fill one hand side with normalised landmark values."""
    for i in range(HAND_KEYPOINT_COUNT):
        row[f"{side}_hand_{i}_x"] = round(hlm[i, 0] / frame_w, 6)
        row[f"{side}_hand_{i}_y"] = round(hlm[i, 1] / frame_h, 6)
        row[f"{side}_hand_{i}_z"] = round(hlm[i, 2] / frame_w, 6)


def _assign_hands_by_x(row, hand_landmarks, frame_h, frame_w):
    """Assign up to 2 hand landmark sets to left/right slots by wrist x."""
    sorted_hands = sorted(hand_landmarks[:2], key=lambda lm: lm[0, 0]) if hand_landmarks else []
    sides = ["left", "right"]
    for i, hlm in enumerate(sorted_hands):
        _fill_hand_side(row, sides[i], hlm, frame_h, frame_w)
    for side in sides[len(sorted_hands) :]:
        _blank_hand_side(row, side)


def frame_to_rows(
    video_name,
    frame_idx,
    timestamp_sec,
    frame_h,
    frame_w,
    body_landmarks,
    body_visibilities,
    hand_landmarks,
    matches,
    tracking=TRACKING_HANDS_ARMS,
    hand_only=False,
):
    """Convert one frame's landmark data into CSV rows (one per person).

    Coordinates are normalised to [0, 1] by dividing by frame dimensions.
    Missing hand data is filled with empty strings (written as blank in CSV).

    *tracking* determines the column layout:
    - ``"hands"``: hand columns only, no body columns.
    - ``"hands-arms"``: 12 arm keypoints + hands (default).
    - ``"body"``: 33 body keypoints + hands.

    When *hand_only* is True and no body was detected, a single row is
    emitted with blank body columns and hand landmarks assigned left/right
    by wrist x-coordinate.
    """
    rows = []

    prefix, kp_names = _body_keypoint_names(tracking)
    wrist_side = wrist_to_side(tracking)

    if tracking == TRACKING_HANDS:
        row = {
            "video": video_name,
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "person_idx": 0,
        }
        _assign_hands_by_x(row, hand_landmarks, frame_h, frame_w)
        rows.append(row)
        return rows

    # --- Modes with body landmarks (hands-arms / body) ---------------------
    if body_landmarks:
        # Build a lookup: arm_idx → {wrist_kp: hand_idx}
        hand_map = {}
        for arm_idx, wrist_kp, hand_idx in matches:
            hand_map.setdefault(arm_idx, {})[wrist_kp] = hand_idx

        for person_idx, (lm, vis) in enumerate(
            zip(body_landmarks, body_visibilities, strict=False)
        ):
            row = {
                "video": video_name,
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp_sec, 4),
                "person_idx": person_idx,
            }

            for kp_idx, name in enumerate(kp_names):
                row[f"{prefix}_{name}_x"] = round(lm[kp_idx, 0] / frame_w, 6)
                row[f"{prefix}_{name}_y"] = round(lm[kp_idx, 1] / frame_h, 6)
                row[f"{prefix}_{name}_z"] = round(lm[kp_idx, 2] / frame_w, 6)
                row[f"{prefix}_{name}_vis"] = round(vis[kp_idx], 4)

            matched_hands = hand_map.get(person_idx, {})
            for wrist_kp, side in sorted(wrist_side.items()):
                hand_idx = matched_hands.get(wrist_kp)
                if hand_idx is not None:
                    _fill_hand_side(row, side, hand_landmarks[hand_idx], frame_h, frame_w)
                else:
                    _blank_hand_side(row, side)

            rows.append(row)

    elif hand_only and hand_landmarks:
        # No body detected — emit hand-only row with blank body data.
        row = {
            "video": video_name,
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "person_idx": 0,
        }

        for name in kp_names:
            row[f"{prefix}_{name}_x"] = ""
            row[f"{prefix}_{name}_y"] = ""
            row[f"{prefix}_{name}_z"] = ""
            row[f"{prefix}_{name}_vis"] = ""

        _assign_hands_by_x(row, hand_landmarks, frame_h, frame_w)
        rows.append(row)

    return rows


def _keypoint_columns(tracking):
    """Return (names, specs) for one row's keypoint columns.

    ``names[i]`` is the bare keypoint name (e.g. ``"arm_left_wrist"``,
    ``"left_hand_4"``); ``specs[i]`` is ``(x_col, y_col, vis_col)``
    with ``vis_col=None`` for hand keypoints (which carry no
    visibility column).
    """
    names = []
    specs = []
    if tracking != TRACKING_HANDS:
        prefix, kp_names = _body_keypoint_names(tracking)
        for name in kp_names:
            names.append(f"{prefix}_{name}")
            specs.append((f"{prefix}_{name}_x", f"{prefix}_{name}_y", f"{prefix}_{name}_vis"))
    for side in ("left", "right"):
        for i in range(HAND_KEYPOINT_COUNT):
            names.append(f"{side}_hand_{i}")
            specs.append((f"{side}_hand_{i}_x", f"{side}_hand_{i}_y", None))
    return names, specs


def _cell_to_float(value):
    """Parse one CSV cell: blank/missing → NaN."""
    return float(value) if value not in ("", None) else float("nan")


def read_csv_keypoints(csv_path):
    """Read a per-camera keypoint CSV back into per-frame arrays.

    Inverse of the ``frame_to_rows`` schema, for 3D-fusion read-back.
    Tracking mode is inferred from the header.  Only ``person_idx == 0``
    rows are read — cross-camera person identity matching is not
    implemented, so multi-person fusion is out of scope.

    Returns ``(keypoint_names, frames)``:
    - ``keypoint_names``: bare names, one per keypoint (body/arm
      keypoints first, then ``{left,right}_hand_{0..20}``).
    - ``frames``: ``frame_idx → (kps, conf)`` where ``kps`` is
      ``(N, 2)`` *normalised* [0, 1] coordinates (NaN where blank) and
      ``conf`` is ``(N,)`` — the ``_vis`` column for body/arm
      keypoints, 1.0/0.0 presence for hand keypoints.

    Raises ``ValueError`` on a missing/foreign header.
    """
    csv_path = pathlib.Path(csv_path)
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = set(reader.fieldnames or [])
        if "body_nose_x" in header:
            tracking = TRACKING_BODY
        elif "arm_left_shoulder_x" in header:
            tracking = TRACKING_HANDS_ARMS
        else:
            tracking = TRACKING_HANDS
        names, specs = _keypoint_columns(tracking)
        required = {"frame_idx", "person_idx"} | {
            col for spec in specs for col in spec if col is not None
        }
        missing = sorted(required - header)
        if missing:
            raise ValueError(f"{csv_path}: not a keypoint CSV (missing columns: {missing[:4]})")

        frames = {}
        for row in reader:
            if row["person_idx"] not in ("0", ""):
                continue
            kps = np.empty((len(specs), 2), dtype=np.float64)
            conf = np.empty(len(specs), dtype=np.float64)
            for i, (x_col, y_col, vis_col) in enumerate(specs):
                x = _cell_to_float(row[x_col])
                y = _cell_to_float(row[y_col])
                kps[i] = (x, y)
                if vis_col is not None:
                    vis = _cell_to_float(row[vis_col])
                    conf[i] = vis if np.isfinite(vis) else 0.0
                else:
                    conf[i] = 1.0 if np.isfinite(x) and np.isfinite(y) else 0.0
            frames[int(row["frame_idx"])] = (kps, conf)
    return names, frames


def open_csv_writer(output_path, tracking=TRACKING_HANDS_ARMS):
    """Open a CSV file for writing and return (file_handle, csv.DictWriter).

    Caller owns the file handle and must close it (typically via try/finally).
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = make_csv_header(tracking)
    fh = output_path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=header)
    writer.writeheader()
    return fh, writer
