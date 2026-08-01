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
HANDEDNESS_MIN_SCORE = 0.6

WORLD3D_FILENAME = "world3d.csv"


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
            cols.extend(
                [
                    f"{side}_hand_{i}_x",
                    f"{side}_hand_{i}_y",
                    f"{side}_hand_{i}_z",
                    f"{side}_hand_{i}_conf",
                ]
            )

    return cols


def _blank_hand_side(row, side):
    """Fill one hand side with empty strings."""
    for i in range(HAND_KEYPOINT_COUNT):
        row[f"{side}_hand_{i}_x"] = ""
        row[f"{side}_hand_{i}_y"] = ""
        row[f"{side}_hand_{i}_z"] = ""
        row[f"{side}_hand_{i}_conf"] = 0.0


def _hand_confidence_vector(confidence):
    """Normalise a scalar/vector hand confidence to the 21-point schema."""
    if confidence is None:
        return np.ones(HAND_KEYPOINT_COUNT, dtype=np.float64)
    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim == 0:
        return np.full(HAND_KEYPOINT_COUNT, float(values), dtype=np.float64)
    if values.shape != (HAND_KEYPOINT_COUNT,):
        raise ValueError(
            f"hand confidence shape {values.shape} does not match {(HAND_KEYPOINT_COUNT,)}"
        )
    return values


def _fill_hand_side(row, side, hlm, frame_h, frame_w, confidence=None):
    """Fill one hand side with normalised coordinates and confidence."""
    confidence_values = _hand_confidence_vector(confidence)
    for i in range(HAND_KEYPOINT_COUNT):
        point_valid = np.isfinite(hlm[i, :3]).all()
        if point_valid:
            row[f"{side}_hand_{i}_x"] = round(hlm[i, 0] / frame_w, 6)
            row[f"{side}_hand_{i}_y"] = round(hlm[i, 1] / frame_h, 6)
            row[f"{side}_hand_{i}_z"] = round(hlm[i, 2] / frame_w, 6)
        else:
            row[f"{side}_hand_{i}_x"] = ""
            row[f"{side}_hand_{i}_y"] = ""
            row[f"{side}_hand_{i}_z"] = ""
        value = confidence_values[i]
        row[f"{side}_hand_{i}_conf"] = (
            round(float(np.clip(value, 0.0, 1.0)), 4) if point_valid and np.isfinite(value) else 0.0
        )


def _aligned_hand_confidences(hand_landmarks, hand_confidences):
    """Return optional confidence entries parallel to the first two hands."""
    n_hands = min(2, len(hand_landmarks))
    if hand_confidences is None:
        return [None] * n_hands
    confidences = list(hand_confidences)
    if len(confidences) < n_hands:
        raise ValueError("hand_confidences is shorter than hand_landmarks")
    return confidences[:n_hands]


def _assign_hands_by_x(row, hand_landmarks, frame_h, frame_w, hand_confidences=None):
    """Assign up to 2 hand landmark sets to left/right slots by wrist x."""
    hands = list(hand_landmarks[:2]) if hand_landmarks else []
    confidences = _aligned_hand_confidences(hands, hand_confidences)
    sorted_hands = sorted(zip(hands, confidences, strict=True), key=lambda pair: pair[0][0, 0])
    sides = ["left", "right"]
    for i, (hlm, confidence) in enumerate(sorted_hands):
        _fill_hand_side(row, sides[i], hlm, frame_h, frame_w, confidence)
    for side in sides[len(sorted_hands) :]:
        _blank_hand_side(row, side)


def _assign_hands(
    row,
    hand_landmarks,
    frame_h,
    frame_w,
    handedness=None,
    hand_confidences=None,
):
    """Assign hands by model handedness when unambiguous, else by wrist x."""
    hands = list(hand_landmarks[:2]) if hand_landmarks else []
    confidences = _aligned_hand_confidences(hands, hand_confidences)
    labels = handedness[: len(hands)] if handedness else []
    valid = (
        len(labels) == len(hands)
        and all(
            label is not None
            and label[0] in {"left", "right"}
            and np.isfinite(label[1])
            and label[1] >= HANDEDNESS_MIN_SCORE
            for label in labels
        )
        and len({label[0] for label in labels}) == len(labels)
    )
    if not valid:
        uncertain_single = (
            len(hands) == 1
            and len(labels) == 1
            and labels[0] is not None
            and labels[0][0] in {"left", "right"}
            and np.isfinite(labels[0][1])
            and labels[0][1] < HANDEDNESS_MIN_SCORE
        )
        if uncertain_single:
            # A supplied 0.5-ish classification contains no anatomical-side
            # information; assigning the sole hand to "left" by list position
            # fabricates a label. Missing metadata retains legacy x fallback.
            _blank_hand_side(row, "left")
            _blank_hand_side(row, "right")
            return
        _assign_hands_by_x(row, hands, frame_h, frame_w, confidences)
        return

    assigned = set()
    for landmarks, confidence, label in zip(hands, confidences, labels, strict=True):
        side = label[0]
        _fill_hand_side(row, side, landmarks, frame_h, frame_w, confidence)
        assigned.add(side)
    for side in {"left", "right"} - assigned:
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
    hand_handedness=None,
    hand_confidences=None,
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
    by model handedness with wrist x-coordinate as the missing/ambiguous-label
    fallback.
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
        _assign_hands(
            row,
            hand_landmarks,
            frame_h,
            frame_w,
            hand_handedness,
            hand_confidences,
        )
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
                    confidence = (
                        hand_confidences[hand_idx] if hand_confidences is not None else None
                    )
                    _fill_hand_side(
                        row,
                        side,
                        hand_landmarks[hand_idx],
                        frame_h,
                        frame_w,
                        confidence,
                    )
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

        _assign_hands(
            row,
            hand_landmarks,
            frame_h,
            frame_w,
            hand_handedness,
            hand_confidences,
        )
        rows.append(row)

    return rows


def _keypoint_columns(tracking, hand_confidence=False):
    """Return (names, specs) for one row's keypoint columns.

    ``names[i]`` is the bare keypoint name (e.g. ``"arm_left_wrist"``,
    ``"left_hand_4"``); ``specs[i]`` is ``(x_col, y_col, vis_col)``
    with ``vis_col=None`` only for legacy hand keypoints that predate
    explicit confidence columns.
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
            vis_col = f"{side}_hand_{i}_conf" if hand_confidence else None
            specs.append((f"{side}_hand_{i}_x", f"{side}_hand_{i}_y", vis_col))
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
    - ``frames``: ``frame_idx → (kps, conf, timestamp_sec)`` where
      ``kps`` is ``(N, 2)`` *normalised* [0, 1] coordinates (NaN where
      blank), ``conf`` is ``(N,)`` — the ``_vis`` column for body/arm
      keypoints and ``_conf`` for hands. Legacy hand CSVs without
      ``_conf`` retain 1.0/0.0 coordinate-presence semantics — and
      ``timestamp_sec`` is a float (NaN where blank).

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
        names, specs = _keypoint_columns(tracking, hand_confidence="left_hand_0_conf" in header)
        required = {"frame_idx", "person_idx", "timestamp_sec"} | {
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
            frames[int(row["frame_idx"])] = (kps, conf, _cell_to_float(row["timestamp_sec"]))
    return names, frames


def make_world3d_header(keypoint_names):
    """Return the column names for a ``world3d.csv`` file.

    Metadata columns mirror the 2D schema (``video`` holds the
    session id); each keypoint contributes nine columns::

        {name}_x_m, {name}_y_m, {name}_z_m       # world metres
        {name}_confidence                        # mean view confidence
        {name}_reproj_err_px                     # mean reprojection error
        {name}_candidate_n_views                  # valid views before consensus
        {name}_n_views                           # contributing views (int)
        {name}_cheirality_ok                     # 1/0 — in front of all views
        {name}_triangulation_angle_deg           # max acute consensus-ray angle

    Downstream consumers must gate on ``reproj_err_px``,
    ``cheirality_ok``, and ``triangulation_angle_deg``.
    """
    cols = ["video", "frame_idx", "timestamp_sec", "person_idx"]
    for name in keypoint_names:
        cols.extend(
            [
                f"{name}_x_m",
                f"{name}_y_m",
                f"{name}_z_m",
                f"{name}_confidence",
                f"{name}_reproj_err_px",
                f"{name}_candidate_n_views",
                f"{name}_n_views",
                f"{name}_cheirality_ok",
                f"{name}_triangulation_angle_deg",
            ]
        )
    return cols


def _fmt_float(value, ndigits):
    """Round for CSV output; non-finite → blank cell."""
    return round(float(value), ndigits) if np.isfinite(value) else ""


def write_world3d_csv(output_path, video_name, keypoint_names, frames):
    """Write triangulated 3D keypoints to ``world3d.csv``.

    *frames* is an iterable of ``(frame_idx, timestamp_sec, world,
    diag)`` — the ``SessionFusion.frames`` layout: ``world`` is
    ``(N, 3)`` metres (NaN where unfused) and ``diag`` is a
    ``FusionDiagnostics``.  ``video_name`` labels every row (the
    session id; ``person_idx`` is always 0 — fusion reads only
    person 0).  NaN values are written as blank cells, matching the
    2D schema convention.  Returns the output path.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = make_world3d_header(keypoint_names)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for frame_idx, timestamp_sec, world, diag in frames:
            angles = diag.get("triangulation_angle_deg")
            candidate_views = diag.get("candidate_n_views", diag["n_views"])
            row = {
                "video": video_name,
                "frame_idx": int(frame_idx),
                "timestamp_sec": _fmt_float(timestamp_sec, 4),
                "person_idx": 0,
            }
            for i, name in enumerate(keypoint_names):
                row[f"{name}_x_m"] = _fmt_float(world[i, 0], 6)
                row[f"{name}_y_m"] = _fmt_float(world[i, 1], 6)
                row[f"{name}_z_m"] = _fmt_float(world[i, 2], 6)
                row[f"{name}_confidence"] = _fmt_float(diag["confidence"][i], 4)
                row[f"{name}_reproj_err_px"] = _fmt_float(diag["reprojection_error_px"][i], 3)
                row[f"{name}_candidate_n_views"] = int(candidate_views[i])
                row[f"{name}_n_views"] = int(diag["n_views"][i])
                cheirality = diag["cheirality_ok"][i]
                row[f"{name}_cheirality_ok"] = (
                    int(bool(cheirality)) if np.isfinite(cheirality) else ""
                )
                row[f"{name}_triangulation_angle_deg"] = (
                    _fmt_float(angles[i], 3) if angles is not None else ""
                )
            writer.writerow(row)
    return output_path


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
