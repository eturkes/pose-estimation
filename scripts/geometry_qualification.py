"""Decode-sampled geometry qualification for the three-camera corpus."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import struct
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import av
import cv2
import numpy as np

SEED = 20260827
SAMPLE_INTERVAL_S = 0.5
ANALYSIS_MAX_DIM = 640
SIGNAL_WIDTH = 160
SIGNAL_HEIGHT = 90
VISUAL_BORDER_FRACTION = 0.15
MIN_TRACKS = 100
MIN_GRID_CELLS = 8
MIN_VALID_FRACTION = 0.80
DRIFT_MEDIAN_GATE_PX = 2.0
DRIFT_P95_GATE_PX = 4.0
RESIDUAL_P95_GATE_PX = 2.0


@dataclass(frozen=True)
class Asset:
    asset_id: str
    capture_id: str
    subject_ordinal: str
    view: str
    source_path: str
    content_sha256: str
    rotation_deg: int
    reported_width: int
    reported_height: int
    reported_frames: int


@dataclass(frozen=True)
class RigidityResult:
    asset_id: str
    capture_id: str
    subject_ordinal: str
    view: str
    device_config: str
    orientation_status: str
    orientation_values: str
    orientation_changes: int
    decode_status: str
    decoded_frames: int
    reported_frames: int
    sampled_frames: int
    valid_samples: int
    valid_fraction: float
    inliers_median: float
    grid_cells_median: float
    drift_median_px: float
    drift_p95_px: float
    residual_p95_px: float
    gate_status: str
    visual_static_component_p95_per_s: float
    visual_low_subject_border_median_per_s: float
    runtime_s: float


def load_assets(path: Path) -> list[Asset]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["disposition"] == "canonical"]
    return [
        Asset(
            asset_id=row["asset_id"],
            capture_id=row["capture_id"],
            subject_ordinal=row["subject_ordinal"],
            view=row["view"],
            source_path=row["source_path"],
            content_sha256=row["content_sha256"],
            rotation_deg=int(row["reported_rotation_deg"]),
            reported_width=int(row["reported_width"]),
            reported_height=int(row["reported_height"]),
            reported_frames=int(row["reported_frame_count"]),
        )
        for row in rows
    ]


def _atoms(stream: BinaryIO, end: int) -> list[tuple[bytes, int, int]]:
    atoms: list[tuple[bytes, int, int]] = []
    while stream.tell() + 8 <= end:
        start = stream.tell()
        header = stream.read(8)
        if len(header) < 8:
            break
        size, kind = struct.unpack(">I4s", header)
        body = start + 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                break
            (size,) = struct.unpack(">Q", extended)
            body = start + 16
        elif size == 0:
            size = end - start
        if size < body - start or start + size > end:
            break
        atoms.append((kind, body, start + size))
        stream.seek(start + size)
    return atoms


def _declared_keys(payload: bytes) -> list[str]:
    keys: list[str] = []
    offset = payload.find(b"keyd")
    while offset >= 0:
        if offset >= 4:
            (size,) = struct.unpack(">I", payload[offset - 4 : offset])
            if 12 <= size <= len(payload) - offset + 4:
                raw = payload[offset + 8 : offset - 4 + size]
                keys.append(raw.decode("utf-8", "replace"))
        offset = payload.find(b"keyd", offset + 4)
    return keys


def metadata_key_maps(path: Path) -> list[dict[int, str]]:
    containers = {b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta"}
    tracks: list[dict[int, str]] = []
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(0)
        top = _atoms(stream, end)
        moov = next(((start, stop) for kind, start, stop in top if kind == b"moov"), None)
        if moov is None:
            return tracks
        stream.seek(moov[0])
        for kind, start, stop in _atoms(stream, moov[1]):
            if kind != b"trak":
                continue
            handler: bytes | None = None
            keys: list[str] = []
            stack = [(start, stop)]
            while stack:
                child_start, child_stop = stack.pop()
                stream.seek(child_start)
                for child_kind, body, child_end in _atoms(stream, child_stop):
                    if child_kind in containers:
                        stack.append((body, child_end))
                    elif child_kind == b"hdlr":
                        stream.seek(body)
                        raw = stream.read(min(child_end - body, 24))
                        if len(raw) >= 12 and raw[8:12] != b"alis":
                            handler = raw[8:12]
                    elif child_kind == b"stsd":
                        stream.seek(body)
                        payload = stream.read(child_end - body)
                        keys.extend(_declared_keys(payload))
            if handler == b"meta":
                tracks.append({index + 1: key for index, key in enumerate(keys)})
    return tracks


def _sample_entries(payload: bytes) -> list[tuple[int, bytes]]:
    entries: list[tuple[int, bytes]] = []
    offset = 0
    while offset + 8 <= len(payload):
        size, key_id = struct.unpack(">II", payload[offset : offset + 8])
        if size < 8 or offset + size > len(payload):
            break
        entries.append((key_id, payload[offset + 8 : offset + size]))
        offset += size
    return entries


def orientation_sequence(path: Path) -> list[int]:
    maps = metadata_key_maps(path)
    values: list[int] = []
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type not in {"video", "audio"}]
        by_index = {
            stream.index: maps[index] if index < len(maps) else {}
            for index, stream in enumerate(streams)
        }
        for packet in container.demux(streams):
            if packet.size == 0:
                continue
            key_map = by_index.get(packet.stream.index, {})
            for key_id, value in _sample_entries(bytes(packet)):
                if key_map.get(key_id, "").endswith("video-orientation"):
                    values.append(int.from_bytes(value, "big"))
    return values


def _orientation_summary(values: list[int]) -> tuple[str, str, int]:
    distinct = sorted(set(values))
    transitions = sum(left != right for left, right in itertools.pairwise(values))
    if not values:
        status = "missing"
    elif transitions:
        status = "changed"
    else:
        status = "constant"
    return status, "|".join(map(str, distinct)), transitions


def _rotate_display(gray: np.ndarray, rotation_deg: int) -> np.ndarray:
    if rotation_deg == 90:
        return np.ascontiguousarray(np.rot90(gray, -1))
    if rotation_deg == 180:
        return np.ascontiguousarray(np.rot90(gray, 2))
    if rotation_deg == 270:
        return np.ascontiguousarray(np.rot90(gray, 1))
    if rotation_deg != 0:
        raise ValueError("unsupported header rotation")
    return np.ascontiguousarray(gray)


def _analysis_gray(frame: av.VideoFrame, rotation_deg: int) -> tuple[np.ndarray, float, float]:
    display = _rotate_display(frame.to_ndarray(format="gray"), rotation_deg)
    original_h, original_w = display.shape
    scale = min(1.0, ANALYSIS_MAX_DIM / max(original_w, original_h))
    width = max(32, round(original_w * scale))
    height = max(32, round(original_h * scale))
    resized = cv2.resize(display, (width, height), interpolation=cv2.INTER_AREA)
    return resized, original_w / width, original_h / height


def _signal_gray(frame: av.VideoFrame, rotation_deg: int) -> np.ndarray:
    if rotation_deg in {90, 270}:
        gray = frame.reformat(width=SIGNAL_HEIGHT, height=SIGNAL_WIDTH, format="gray").to_ndarray()
        gray = np.rot90(gray, -1 if rotation_deg == 90 else 1)
    else:
        gray = frame.reformat(width=SIGNAL_WIDTH, height=SIGNAL_HEIGHT, format="gray").to_ndarray()
        if rotation_deg == 180:
            gray = np.rot90(gray, 2)
    if gray.shape != (SIGNAL_HEIGHT, SIGNAL_WIDTH):
        raise ValueError("unexpected signal-frame shape")
    return cv2.GaussianBlur(np.ascontiguousarray(gray), (3, 3), 0)


def _border_mask() -> np.ndarray:
    y = max(1, round(SIGNAL_HEIGHT * VISUAL_BORDER_FRACTION))
    x = max(1, round(SIGNAL_WIDTH * VISUAL_BORDER_FRACTION))
    mask = np.ones((SIGNAL_HEIGHT, SIGNAL_WIDTH), dtype=bool)
    mask[y:-y, x:-x] = False
    return mask


VISUAL_BORDER_MASK = _border_mask()
VISUAL_CENTER_MASK = ~VISUAL_BORDER_MASK


def _visual_metrics(
    times: list[float], motion: list[float], border: list[float], centre: list[float]
) -> tuple[float, float]:
    time_array = np.asarray(times, dtype=np.float64)
    delta_t = np.diff(time_array, prepend=0)
    if len(time_array) < 2 or np.any(delta_t <= 0):
        raise ValueError("visual signal has invalid PTS support")
    border_array = np.asarray(border, dtype=np.float64) / delta_t
    centre_array = np.asarray(centre, dtype=np.float64) / delta_t
    static_component = np.minimum(border_array, centre_array)
    low_centre = centre_array <= np.quantile(centre_array, 0.20)
    return (
        float(np.quantile(static_component, 0.95)),
        float(np.median(border_array[low_centre])),
    )


def _tracking_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, 255, dtype=np.uint8)


def _grid_cell_count(points: np.ndarray, width: int, height: int) -> int:
    x = np.clip((points[:, 0] * 4 / width).astype(int), 0, 3)
    y = np.clip((points[:, 1] * 4 / height).astype(int), 0, 3)
    return len(set(zip(x.tolist(), y.tolist(), strict=True)))


def _match_sample(
    baseline: np.ndarray,
    current: np.ndarray,
    baseline_keypoints: Any,
    baseline_descriptors: np.ndarray,
    detector: Any,
    matcher: Any,
    scale_x: float,
    scale_y: float,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    current_keypoints, current_descriptors = detector.detectAndCompute(
        current, _tracking_mask(current.shape)
    )
    if current_descriptors is None or len(current_keypoints) < MIN_TRACKS:
        return None
    forward_knn = matcher.knnMatch(baseline_descriptors, current_descriptors, k=2)
    reverse_knn = matcher.knnMatch(current_descriptors, baseline_descriptors, k=2)
    forward = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in forward_knn
        if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance
    }
    reverse = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse_knn
        if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance
    }
    mutual = [
        (source, target) for source, target in forward.items() if reverse.get(target) == source
    ]
    if len(mutual) < MIN_TRACKS:
        return None
    source = np.asarray([baseline_keypoints[index].pt for index, _ in mutual], dtype=np.float64)
    target = np.asarray([current_keypoints[index].pt for _, index in mutual], dtype=np.float64)
    cv2.setRNGSeed(SEED + sample_index)
    threshold = DRIFT_P95_GATE_PX / max((scale_x + scale_y) / 2, 1.0)
    homography, mask = cv2.findHomography(
        source,
        target,
        cv2.USAC_MAGSAC,
        threshold,
        maxIters=10_000,
        confidence=0.999,
    )
    if homography is None or mask is None or not np.all(np.isfinite(homography)):
        return None
    inliers = mask.ravel().astype(bool)
    source_inliers = source[inliers]
    target_inliers = target[inliers]
    if len(source_inliers) < MIN_TRACKS:
        return None
    source_cells = _grid_cell_count(source_inliers, baseline.shape[1], baseline.shape[0])
    target_cells = _grid_cell_count(target_inliers, current.shape[1], current.shape[0])
    cells = min(source_cells, target_cells)
    xs = np.linspace(0.05 * baseline.shape[1], 0.95 * baseline.shape[1], 8)
    ys = np.linspace(0.05 * baseline.shape[0], 0.95 * baseline.shape[0], 8)
    grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    warped = cv2.perspectiveTransform(grid.reshape(1, -1, 2), homography).reshape(-1, 2)
    drift = np.hypot(
        (warped[:, 0] - grid[:, 0]) * scale_x,
        (warped[:, 1] - grid[:, 1]) * scale_y,
    )
    predicted = cv2.perspectiveTransform(source_inliers.reshape(1, -1, 2), homography).reshape(
        -1, 2
    )
    residual = np.hypot(
        (predicted[:, 0] - target_inliers[:, 0]) * scale_x,
        (predicted[:, 1] - target_inliers[:, 1]) * scale_y,
    )
    return drift, residual, len(source_inliers), cells


def _rigidity_metrics(
    frames: list[np.ndarray], scale_x: float, scale_y: float
) -> tuple[int, float, float, float, float, float, float, str]:
    if len(frames) < 3:
        return 0, 0.0, math.nan, math.nan, math.nan, math.nan, math.nan, "unmeasurable"
    baseline = frames[0]
    detector = cv2.SIFT.create(nfeatures=2500, contrastThreshold=0.01)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    baseline_keypoints, baseline_descriptors = detector.detectAndCompute(
        baseline, _tracking_mask(baseline.shape)
    )
    if baseline_descriptors is None or len(baseline_keypoints) < MIN_TRACKS:
        return 0, 0.0, math.nan, math.nan, math.nan, math.nan, math.nan, "unmeasurable"
    all_drift: list[np.ndarray] = []
    all_residual: list[np.ndarray] = []
    inliers: list[int] = []
    cells: list[int] = []
    for sample_index, current in enumerate(frames[1:], start=1):
        matched = _match_sample(
            baseline,
            current,
            baseline_keypoints,
            baseline_descriptors,
            detector,
            matcher,
            scale_x,
            scale_y,
            sample_index,
        )
        if matched is None:
            continue
        drift, residual, count, covered = matched
        all_drift.append(drift)
        all_residual.append(residual)
        inliers.append(count)
        cells.append(covered)
    valid_fraction = len(all_drift) / (len(frames) - 1)
    if not all_drift or valid_fraction < MIN_VALID_FRACTION:
        return (
            len(all_drift),
            valid_fraction,
            float(np.median(inliers)) if inliers else math.nan,
            float(np.median(cells)) if cells else math.nan,
            math.nan,
            math.nan,
            math.nan,
            "unmeasurable",
        )
    drift_values = np.concatenate(all_drift)
    residual_values = np.concatenate(all_residual)
    drift_median = float(np.median(drift_values))
    drift_p95 = float(np.quantile(drift_values, 0.95))
    residual_p95 = float(np.quantile(residual_values, 0.95))
    passed = drift_median <= DRIFT_MEDIAN_GATE_PX and drift_p95 <= DRIFT_P95_GATE_PX
    support_ok = float(np.median(cells)) >= MIN_GRID_CELLS
    return (
        len(all_drift),
        valid_fraction,
        float(np.median(inliers)),
        float(np.median(cells)),
        drift_median,
        drift_p95,
        residual_p95,
        "pass" if passed and support_ok else ("fail" if not passed else "unmeasurable"),
    )


def analyze_asset(asset: Asset, corpus_root: Path) -> RigidityResult:
    cv2.setNumThreads(1)
    started = time.perf_counter()
    path = corpus_root / asset.source_path
    orientation = orientation_sequence(path)
    orientation_status, orientation_values, orientation_changes = _orientation_summary(orientation)
    sampled: list[np.ndarray] = []
    scale_x = math.nan
    scale_y = math.nan
    signal_times: list[float] = []
    signal_motion: list[float] = []
    signal_border: list[float] = []
    signal_centre: list[float] = []
    previous_signal: np.ndarray | None = None
    previous_t = -math.inf
    first_t: float | None = None
    next_sample_t = 0.0
    decoded_frames = 0
    with av.open(str(path)) as container:
        model = container.metadata.get("com.apple.quicktime.model")
        software = container.metadata.get("com.apple.quicktime.software")
        if model is None or software is None:
            raise ValueError("missing device configuration")
        device_config = f"{model} / {software}"
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("video frame lacks PTS or time_base")
            frame_t = float(frame.pts * frame.time_base)
            if frame_t <= previous_t:
                raise ValueError("decoded PTS values are not strictly increasing")
            if first_t is None:
                first_t = frame_t
            relative_t = frame_t - first_t
            signal = _signal_gray(frame, asset.rotation_deg)
            decoded_frames += 1
            if previous_signal is not None:
                delta = signal.astype(np.float32) - previous_signal
                delta -= np.median(delta)
                magnitude = np.abs(delta)
                signal_times.append(frame_t)
                signal_motion.append(float(np.mean(magnitude)))
                signal_border.append(float(np.mean(magnitude[VISUAL_BORDER_MASK])))
                signal_centre.append(float(np.mean(magnitude[VISUAL_CENTER_MASK])))
            previous_signal = signal
            if relative_t + 1e-9 >= next_sample_t:
                gray, current_scale_x, current_scale_y = _analysis_gray(frame, asset.rotation_deg)
                if sampled and (current_scale_x != scale_x or current_scale_y != scale_y):
                    raise ValueError("display dimensions changed during decode")
                sampled.append(gray)
                scale_x, scale_y = current_scale_x, current_scale_y
                while next_sample_t <= relative_t + 1e-9:
                    next_sample_t += SAMPLE_INTERVAL_S
            previous_t = frame_t
    decode_status = "ok" if decoded_frames == asset.reported_frames else "frame_count_mismatch"
    visual_static, visual_border = _visual_metrics(
        signal_times,
        signal_motion,
        signal_border,
        signal_centre,
    )
    if orientation_status == "constant":
        metrics = _rigidity_metrics(sampled, scale_x, scale_y)
    else:
        metrics = (0, 0.0, math.nan, math.nan, math.nan, math.nan, math.nan, "excluded_orientation")
    (
        valid_samples,
        valid_fraction,
        inliers_median,
        cells_median,
        drift_median,
        drift_p95,
        residual_p95,
        gate_status,
    ) = metrics
    return RigidityResult(
        asset.asset_id,
        asset.capture_id,
        asset.subject_ordinal,
        asset.view,
        device_config,
        orientation_status,
        orientation_values,
        orientation_changes,
        decode_status,
        decoded_frames,
        asset.reported_frames,
        len(sampled),
        valid_samples,
        valid_fraction,
        inliers_median,
        cells_median,
        drift_median,
        drift_p95,
        residual_p95,
        gate_status,
        visual_static,
        visual_border,
        time.perf_counter() - started,
    )


def _worker(argument: tuple[Asset, str]) -> RigidityResult:
    asset, corpus_root = argument
    try:
        return analyze_asset(asset, Path(corpus_root))
    except Exception as error:
        return RigidityResult(
            asset.asset_id,
            asset.capture_id,
            asset.subject_ordinal,
            asset.view,
            "unknown",
            "error",
            "",
            0,
            type(error).__name__,
            0,
            asset.reported_frames,
            0,
            0,
            0.0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            "error",
            math.nan,
            math.nan,
            0.0,
        )


def quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return dict.fromkeys(("min", "p25", "median", "p75", "p95", "max"))
    points = np.quantile(finite, [0, 0.25, 0.5, 0.75, 0.95, 1])
    return dict(zip(("min", "p25", "median", "p75", "p95", "max"), map(float, points), strict=True))


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "gate_status": dict(Counter(str(row["gate_status"]) for row in rows)),
        "drift_median_px": quantiles([float(row["drift_median_px"]) for row in rows]),
        "drift_p95_px": quantiles([float(row["drift_p95_px"]) for row in rows]),
        "residual_p95_px": quantiles([float(row["residual_p95_px"]) for row in rows]),
    }


def _grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    labels = sorted({str(row[field]) for row in rows})
    return {label: _group_summary([row for row in rows if row[field] == label]) for label in labels}


def _visual_flags(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    fields = (
        "visual_static_component_p95_per_s",
        "visual_low_subject_border_median_per_s",
    )
    flags: list[np.ndarray] = []
    rules: dict[str, Any] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = median + 3.5 * 1.4826 * mad
        field_flags = values > threshold
        flags.append(field_flags)
        rules[field] = {
            "median": median,
            "mad": mad,
            "threshold": threshold,
            "count": int(np.sum(field_flags)),
        }
    return np.logical_or.reduce(flags), rules


def _reclassify_rigidity_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        orientation_status = str(row["orientation_status"])
        if orientation_status in {"changed", "missing"}:
            status = "excluded_orientation"
        elif orientation_status != "constant":
            status = "error"
        else:
            drift_median = float(row["drift_median_px"])
            drift_p95 = float(row["drift_p95_px"])
            support_ok = (
                math.isfinite(drift_median)
                and math.isfinite(drift_p95)
                and float(row["valid_fraction"]) >= MIN_VALID_FRACTION
                and float(row["grid_cells_median"]) >= MIN_GRID_CELLS
            )
            if not support_ok:
                status = "unmeasurable"
            elif drift_median <= DRIFT_MEDIAN_GATE_PX and drift_p95 <= DRIFT_P95_GATE_PX:
                status = "pass"
            else:
                status = "fail"
        row["gate_status"] = status


def summarize_rigidity(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    visual_flags, visual_rules = _visual_flags(rows)
    geometry_flags = np.asarray([row["gate_status"] == "fail" for row in rows])
    eligible = np.asarray([row["gate_status"] in {"pass", "fail"} for row in rows])
    return {
        "method": {
            "sampling_interval_s": SAMPLE_INTERVAL_S,
            "time_source": "PTS * time_base",
            "analysis_max_dimension_px": ANALYSIS_MAX_DIM,
            "background_model": "full-frame mutual SIFT; MAGSAC rejects independently moving subject features",
            "tracking": "mutual-ratio SIFT + MAGSAC homography to the first sampled frame",
            "grid": "8x8 in display coordinates",
            "support": {
                "minimum_inliers": MIN_TRACKS,
                "minimum_median_4x4_cells": MIN_GRID_CELLS,
                "minimum_valid_fraction": MIN_VALID_FRACTION,
            },
            "gate_px": {
                "median": DRIFT_MEDIAN_GATE_PX,
                "p95": DRIFT_P95_GATE_PX,
            },
            "diagnostic_px": {"homography_residual_p95": RESIDUAL_P95_GATE_PX},
            "seed": SEED,
        },
        "asset_count": len(rows),
        "decode_status": dict(Counter(str(row["decode_status"]) for row in rows)),
        "orientation_status": dict(Counter(str(row["orientation_status"]) for row in rows)),
        "gate_status": dict(Counter(str(row["gate_status"]) for row in rows)),
        "eligible_distribution": _group_summary(
            [row for row in rows if row["gate_status"] in {"pass", "fail"}]
        ),
        "by_view": _grouped(rows, "view"),
        "by_device_configuration": _grouped(rows, "device_config"),
        "visual_spike_comparison": {
            "visual_rules": visual_rules,
            "visual_flagged": int(np.sum(visual_flags)),
            "geometry_flagged": int(np.sum(geometry_flags)),
            "eligible_assets": int(np.sum(eligible)),
            "both_flagged": int(np.sum(visual_flags & geometry_flags)),
            "geometry_only": int(np.sum(~visual_flags & geometry_flags)),
            "visual_only": int(np.sum(visual_flags & ~geometry_flags)),
            "both_clear": int(np.sum(~visual_flags & ~geometry_flags)),
            "evaluable": {
                "count": int(np.sum(eligible)),
                "visual_flagged": int(np.sum(visual_flags & eligible)),
                "both_flagged": int(np.sum(visual_flags & geometry_flags & eligible)),
                "geometry_only": int(np.sum(~visual_flags & geometry_flags & eligible)),
                "visual_only": int(np.sum(visual_flags & ~geometry_flags & eligible)),
                "both_clear": int(np.sum(~visual_flags & ~geometry_flags & eligible)),
                "exact_agreement": float(
                    np.mean(visual_flags[eligible] == geometry_flags[eligible])
                ),
            },
        },
        "runtime": {
            "worker_cpu_s": float(sum(float(row["runtime_s"]) for row in rows)),
            "wall_s": wall_s,
            "decoded_frames": int(sum(int(row["decoded_frames"]) for row in rows)),
            "sampled_frames": int(sum(int(row["sampled_frames"]) for row in rows)),
        },
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_rigidity(args: argparse.Namespace) -> None:
    assets = load_assets(args.inventory)
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    started = time.perf_counter()
    arguments = [(asset, str(args.corpus_root)) for asset in assets]
    if args.workers == 1:
        results = [_worker(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_worker, argument) for argument in arguments]
            results = [future.result() for future in as_completed(futures)]
    rows = [asdict(result) for result in sorted(results, key=lambda result: result.asset_id)]
    _reclassify_rigidity_rows(rows)
    wall_s = time.perf_counter() - started
    _write_rows(args.output_dir / "rigidity_assets.csv", rows)
    summary = summarize_rigidity(rows, wall_s)
    _write_json(args.output_dir / "rigidity_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_summarize_rigidity(args: argparse.Namespace) -> None:
    rows_path = args.output_dir / "rigidity_assets.csv"
    with rows_path.open(newline="", encoding="utf-8") as stream:
        rows: list[dict[str, Any]] = list(csv.DictReader(stream))
    _reclassify_rigidity_rows(rows)
    summary_path = args.output_dir / "rigidity_summary.json"
    wall_s = 0.0
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        wall_s = float(prior.get("runtime", {}).get("wall_s", 0.0))
    _write_rows(rows_path, rows)
    summary = summarize_rigidity(rows, wall_s)
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--inventory", type=Path, default=root / "inventory" / "assets.csv")
    parser.add_argument("--corpus-root", type=Path, default=root / "videos" / "3-cam")
    parser.add_argument("--output-dir", type=Path, default=root / ".scratch" / "geometry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rigidity = subparsers.add_parser("rigidity")
    rigidity.add_argument("--workers", type=int, default=4)
    rigidity.set_defaults(function=command_rigidity)
    summarize = subparsers.add_parser("summarize-rigidity")
    summarize.set_defaults(function=command_summarize_rigidity)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
