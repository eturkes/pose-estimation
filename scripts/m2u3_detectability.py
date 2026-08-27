#!/usr/bin/env python3
"""Regenerate M2.3 detectability, image-scale, and person-count evidence."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import struct
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np

from pose_estimation.rtmlib_openvino import _patch_rtmlib_openvino
from pose_estimation.run import (
    _DET_INPUT_SIZE,
    _DET_URL,
    MODEL_REGISTRY,
    TRACKING_INDICES,
    SplitDeviceSolution,
)

MODEL_NAME = "rtmw-l"
SAMPLE_COUNT = 24
DETECTOR_THRESHOLD = 0.7
KEYPOINT_THRESHOLDS = (0.3, 0.4)
ORIENTATION_ROTATION = {1: 0, 6: 90, 3: 180, 8: 270}
_META_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("videos/3-cam"))
    parser.add_argument("--inventory", type=Path, default=Path("inventory"))
    parser.add_argument("--out", type=Path, default=Path("output/m2u3-detectability"))
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only N assets for a smoke run."
    )
    parser.add_argument("--resume", action="store_true", help="Keep completed checkpoint rows.")
    parser.add_argument(
        "--cpu-check-frames",
        type=int,
        default=32,
        help="Compare GPU and CPU detector outputs on N sampled frames.",
    )
    return parser.parse_args()


def _atoms(file: BinaryIO, end: int) -> list[tuple[bytes, int, int]]:
    out: list[tuple[bytes, int, int]] = []
    while file.tell() + 8 <= end:
        start = file.tell()
        header = file.read(8)
        if len(header) < 8:
            break
        size, atom_type = struct.unpack(">I4s", header)
        body = start + 8
        if size == 1:
            extended = file.read(8)
            if len(extended) < 8:
                break
            (size,) = struct.unpack(">Q", extended)
            body = start + 16
        elif size == 0:
            size = end - start
        if size < body - start or start + size > end:
            break
        out.append((atom_type, body, start + size))
        file.seek(start + size)
    return out


def _declared_keys(payload: bytes) -> list[str]:
    keys: list[str] = []
    offset = payload.find(b"keyd")
    while offset != -1:
        if offset >= 4:
            (size,) = struct.unpack(">I", payload[offset - 4 : offset])
            if 12 <= size <= len(payload):
                value = payload[offset + 8 : offset - 4 + size]
                keys.append(value.decode("utf-8", "replace"))
        offset = payload.find(b"keyd", offset + 4)
    return keys


def _metadata_key_maps(path: Path) -> list[dict[int, str]]:
    tracks: list[tuple[str | None, list[str]]] = []
    with path.open("rb") as file:
        end = path.stat().st_size
        top = _atoms(file, end)
        moov = next(((body, stop) for atom_type, body, stop in top if atom_type == b"moov"), None)
        if moov is None:
            return []
        file.seek(moov[0])
        for atom_type, body, stop in _atoms(file, moov[1]):
            if atom_type != b"trak":
                continue
            handler: str | None = None
            keys: list[str] = []
            stack = [(body, stop)]
            while stack:
                start, end = stack.pop()
                file.seek(start)
                for child_type, child_body, child_stop in _atoms(file, end):
                    if child_type in _META_CONTAINERS:
                        stack.append((child_body, child_stop))
                    elif child_type == b"hdlr":
                        file.seek(child_body)
                        raw = file.read(min(child_stop - child_body, 24))
                        if len(raw) >= 12 and raw[8:12] != b"alis":
                            handler = raw[8:12].decode("latin-1")
                    elif child_type == b"stsd":
                        file.seek(child_body)
                        payload = file.read(child_stop - child_body)
                        entry_offset = 8
                        while entry_offset + 8 <= len(payload):
                            (entry_size,) = struct.unpack(
                                ">I", payload[entry_offset : entry_offset + 4]
                            )
                            if entry_size < 8 or entry_offset + entry_size > len(payload):
                                break
                            keys.extend(
                                _declared_keys(payload[entry_offset : entry_offset + entry_size])
                            )
                            entry_offset += entry_size
            tracks.append((handler, keys))
    return [
        {index + 1: key for index, key in enumerate(keys)}
        for handler, keys in tracks
        if handler == "meta"
    ]


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


def _timed_metadata(path: Path) -> dict[str, object]:
    key_maps = _metadata_key_maps(path)
    orientations: list[tuple[float, int]] = []
    face_counts: list[int] = []
    with av.open(str(path)) as container:
        data_streams = [
            stream for stream in container.streams if stream.type not in ("video", "audio")
        ]
        by_index = {
            stream.index: key_maps[index] if index < len(key_maps) else {}
            for index, stream in enumerate(data_streams)
        }
        face_streams = {
            stream.index
            for stream in data_streams
            if any("detected-face" in key for key in by_index[stream.index].values())
        }
        for packet in container.demux(data_streams):
            if packet.size == 0:
                continue
            entries = _sample_entries(bytes(packet))
            if packet.stream.index in face_streams:
                face_counts.append(len(entries))
                continue
            if packet.pts is None:
                packet_time = math.nan
            else:
                packet_time = float(packet.pts * packet.time_base)
            key_map = by_index.get(packet.stream.index, {})
            for key_id, value in entries:
                key = key_map.get(key_id, "")
                if key.endswith("video-orientation"):
                    orientations.append((packet_time, int.from_bytes(value, "big")))
                elif key.endswith("detected-face"):
                    face_counts.append(1)
    orientations.sort(key=lambda item: (math.isnan(item[0]), item[0]))
    return {
        "orientations": orientations,
        "face_track_present": bool(face_counts),
        "face_max": max(face_counts, default=0),
        "face_samples": len(face_counts),
    }


def _rotation_at(
    pts_s: float, orientations: list[tuple[float, int]], header_rotation: int
) -> tuple[int, int | None, str]:
    finite = [(pts, code) for pts, code in orientations if math.isfinite(pts)]
    if finite:
        times = [pts for pts, _ in finite]
        index = max(0, bisect.bisect_right(times, pts_s) - 1)
        code = finite[index][1]
        if code in ORIENTATION_ROTATION:
            return ORIENTATION_ROTATION[code], code, "timed"
        return header_rotation, code, "unknown_timed_code"
    codes = [code for _, code in orientations]
    if codes and codes[0] in ORIENTATION_ROTATION:
        return ORIENTATION_ROTATION[codes[0]], codes[0], "untimed"
    return header_rotation, None, "header_fallback"


def _rotate(frame: np.ndarray, degrees_clockwise: int) -> np.ndarray:
    turns = (degrees_clockwise // 90) % 4
    return np.ascontiguousarray(np.rot90(frame, k=(-turns) % 4))


def _sample_frames(
    path: Path,
    sample_count: int,
    orientations: list[tuple[float, int]],
    header_rotation: int,
) -> tuple[list[dict[str, object]], str]:
    samples: list[dict[str, object]] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            return [], "no_video_stream"
        stream = container.streams.video[0]
        if stream.duration is None or stream.duration <= 0:
            return [], "no_pts_duration"
        start = stream.start_time or 0
        duration = int(stream.duration)
        targets = [
            start + ((2 * index + 1) * duration) // (2 * sample_count)
            for index in range(sample_count)
        ]
        for sample_index, target in enumerate(targets):
            container.seek(target, stream=stream, any_frame=False, backward=True)
            chosen = None
            prior = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if frame.pts >= target:
                    chosen = frame
                    break
                prior = frame
            if chosen is None:
                chosen = prior
            if chosen is None or chosen.pts is None:
                continue
            pts_s = float(chosen.pts * chosen.time_base)
            rotation, code, rotation_source = _rotation_at(pts_s, orientations, header_rotation)
            array = _rotate(chosen.to_ndarray(format="bgr24"), rotation)
            samples.append(
                {
                    "sample_index": sample_index,
                    "pts_s": pts_s,
                    "rotation": rotation,
                    "orientation_code": code,
                    "orientation_source": rotation_source,
                    "frame": array,
                }
            )
    status = "ok" if len(samples) == sample_count else "sample_shortfall"
    return samples, status


def _device_config(path: Path) -> str:
    """Return ``model/software`` from the container, matching qualify.py's reading.

    Deriving the label from codec + view instead reproduces the corpus-wide correlation
    while inventing the per-asset value, so any stratum that disagrees with the
    correlation is silently mislabelled rather than visible.
    """
    try:
        with av.open(str(path)) as container:
            metadata = container.metadata
    except Exception:
        return "unknown"
    model = metadata.get("com.apple.quicktime.model", "").strip()
    software = metadata.get("com.apple.quicktime.software", "").strip()
    if not model and not software:
        return "unknown"
    return f"{model}/{software}".strip("/")


def _primary_index(scores: np.ndarray | None) -> int | None:
    if scores is None or scores.ndim != 2 or scores.shape[0] == 0:
        return None
    finite = np.where(np.isfinite(scores), scores, 0.0)
    return int(np.argmax(finite.mean(axis=1)))


def _box_height(box: object, image_height: int) -> float:
    values = np.asarray(box, dtype=float).reshape(-1)
    if values.size < 4 or not np.isfinite(values[:4]).all():
        return math.nan
    return max(0.0, min(float(image_height), values[3]) - max(0.0, values[1]))


def _hand_span(keypoints: np.ndarray, scores: np.ndarray, threshold: float = 0.3) -> float:
    spans: list[float] = []
    for offset in (91, 112):
        first, second = offset + 5, offset + 17
        if (
            second < keypoints.shape[0]
            and scores[first] >= threshold
            and scores[second] >= threshold
            and np.isfinite(keypoints[[first, second]]).all()
        ):
            spans.append(float(np.linalg.norm(keypoints[first] - keypoints[second])))
    return float(np.median(spans)) if spans else math.nan


def _finite(values: list[float]) -> np.ndarray:
    return np.asarray([value for value in values if math.isfinite(value)], dtype=float)


def _quantile(values: list[float], probability: float) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, probability)) if finite.size else math.nan


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = _finite(values)
    if not finite.size:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": int(finite.size),
        "min": round(float(finite.min()), 6),
        "p25": round(float(np.quantile(finite, 0.25)), 6),
        "median": round(float(np.median(finite)), 6),
        "p75": round(float(np.quantile(finite, 0.75)), 6),
        "max": round(float(finite.max()), 6),
        "mean": round(float(finite.mean()), 6),
    }


def _execution_devices(tool: object) -> dict[str, object]:
    value = tool.compiled_model.get_property("EXECUTION_DEVICES")
    normalized = [value] if isinstance(value, str) else list(value)
    return {"raw_type": type(value).__name__, "devices": normalized}


def _make_models() -> tuple[object, object, dict[str, object]]:
    _patch_rtmlib_openvino()
    from rtmlib import YOLOX, PoseTracker

    model = MODEL_REGISTRY[MODEL_NAME]
    solution = partial(
        SplitDeviceSolution,
        det=_DET_URL,
        det_input_size=_DET_INPUT_SIZE,
        det_device="GPU",
        pose_class=model["pose_class"],
        pose=model["pose"],
        pose_input_size=model["pose_input_size"],
        pose_device="NPU",
    )
    tracker = PoseTracker(
        solution,
        mode="balanced",
        det_frequency=1,
        tracking=False,
        backend="openvino",
        to_openpose=False,
    )
    cpu_detector = YOLOX(
        _DET_URL,
        model_input_size=_DET_INPUT_SIZE,
        backend="openvino",
        device="CPU",
    )
    placement = {
        "detector": _execution_devices(tracker.det_model),
        "pose": _execution_devices(tracker.pose_model),
        "cpu_check_detector": _execution_devices(cpu_detector),
    }
    return tracker, cpu_detector, placement


def _box_iou(box_a: object, box_b: object) -> float:
    a = np.asarray(box_a, dtype=float).reshape(-1)[:4]
    b = np.asarray(box_b, dtype=float).reshape(-1)[:4]
    left_top = np.maximum(a[:2], b[:2])
    right_bottom = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(0.0, right_bottom - left_top)))
    area_a = float(np.prod(np.maximum(0.0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:] - b[:2])))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _compare_detectors(gpu_boxes: list[object], cpu_boxes: object) -> dict[str, object]:
    gpu = sorted((np.asarray(box) for box in gpu_boxes), key=lambda box: float(box[0]))
    cpu = sorted((np.asarray(box) for box in cpu_boxes), key=lambda box: float(box[0]))
    ious = [_box_iou(a, b) for a, b in zip(gpu, cpu, strict=False)]
    deviations = [
        float(np.max(np.abs(np.asarray(a)[:4] - np.asarray(b)[:4])))
        for a, b in zip(gpu, cpu, strict=False)
    ]
    return {
        "gpu_count": len(gpu),
        "cpu_count": len(cpu),
        "count_agrees": len(gpu) == len(cpu),
        "min_iou": min(ious, default=math.nan),
        "max_abs_box_delta_px": max(deviations, default=math.nan),
    }


def _asset_record(
    row: dict[str, str],
    path: Path,
    sample_count: int,
    tracker: object,
    cpu_detector: object,
    cpu_check_sample: int | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    metadata_status = "ok"
    try:
        metadata = _timed_metadata(path)
    except Exception as error:
        metadata = {
            "orientations": [],
            "face_track_present": False,
            "face_max": 0,
            "face_samples": 0,
        }
        metadata_status = f"error:{type(error).__name__}"
    orientations = metadata["orientations"]
    assert isinstance(orientations, list)
    samples, decode_status = _sample_frames(
        path,
        sample_count,
        orientations,
        int(float(row["reported_rotation_deg"] or 0)),
    )
    active = sorted(TRACKING_INDICES["hands-arms"])
    keypoint_scores: list[float] = []
    heights: list[float] = []
    hand_spans: list[float] = []
    person_counts: Counter[int] = Counter()
    orientation_sources: Counter[str] = Counter()
    inference_failures: Counter[str] = Counter()
    cpu_comparison = None
    for sample in samples:
        frame = sample.pop("frame")
        assert isinstance(frame, np.ndarray)
        # Detection and pose run statelessly, per frame. PoseTracker cannot serve here:
        # det_frequency=1 makes its `not tracking and det_frequency != 1` guard false, so
        # it takes the IoU-tracking branch despite tracking=False and matches each frame
        # against the previous one. Samples are seconds apart and one tracker spans all
        # 379 assets, so IoU is ~0, bboxes_last_frame empties, and detect_rate reads 0.
        boxes = list(tracker.det_model(frame))
        gpu_check_boxes = boxes if cpu_check_sample == sample["sample_index"] else None
        try:
            keypoints, scores = tracker.pose_model(frame, bboxes=boxes)
        except Exception as error:
            inference_failures[type(error).__name__] += 1
            continue
        person_counts[len(boxes)] += 1
        orientation_sources[str(sample["orientation_source"])] += 1
        primary = _primary_index(scores)
        if primary is not None and primary < len(boxes):
            primary_scores = np.asarray(scores[primary], dtype=float)
            primary_keypoints = np.asarray(keypoints[primary], dtype=float)
            selected = primary_scores[active]
            keypoint_scores.extend(selected[np.isfinite(selected)].tolist())
            heights.append(_box_height(boxes[primary], frame.shape[0]))
            hand_spans.append(_hand_span(primary_keypoints, primary_scores))
        if gpu_check_boxes is not None:
            cpu_boxes = cpu_detector(frame)
            cpu_comparison = _compare_detectors(gpu_check_boxes, cpu_boxes)
    completed = sum(person_counts.values())
    detected = completed - person_counts[0]
    orientation_values = sorted({int(code) for _, code in orientations})
    return (
        {
            "asset_id": row["asset_id"],
            "view": row["view"],
            "task": row["task"],
            "device_config": _device_config(path),
            "metadata_status": metadata_status,
            "decode_status": decode_status,
            "samples_requested": sample_count,
            "samples_inferred": completed,
            "inference_failures": dict(sorted(inference_failures.items())),
            "detect_rate": detected / completed if completed else math.nan,
            "zero_detection_fraction": person_counts[0] / completed if completed else math.nan,
            "keypoint_conf_median": _quantile(keypoint_scores, 0.5),
            "keypoint_conf_p25": _quantile(keypoint_scores, 0.25),
            "keypoint_coverage_ge_0_3": float(np.mean(np.asarray(keypoint_scores) >= 0.3))
            if keypoint_scores
            else math.nan,
            "keypoint_coverage_ge_0_4": float(np.mean(np.asarray(keypoint_scores) >= 0.4))
            if keypoint_scores
            else math.nan,
            "subject_px_height_median": _quantile(heights, 0.5),
            "hand_span_px_median": _quantile(hand_spans, 0.5),
            "person_count_hist": {str(key): value for key, value in sorted(person_counts.items())},
            "max_person_count": max(person_counts, default=0),
            "multi_person_frame_fraction": sum(
                value for key, value in person_counts.items() if key > 1
            )
            / completed
            if completed
            else math.nan,
            "orientation_values": orientation_values,
            "orientation_sources": dict(sorted(orientation_sources.items())),
            "face_track_present": bool(metadata["face_track_present"]),
            "face_max": int(metadata["face_max"]),
            "face_samples": int(metadata["face_samples"]),
        },
        cpu_comparison,
    )


def _grouped(records: list[dict[str, object]], key: str) -> dict[str, object]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    metrics = (
        "detect_rate",
        "zero_detection_fraction",
        "keypoint_conf_median",
        "keypoint_conf_p25",
        "keypoint_coverage_ge_0_3",
        "keypoint_coverage_ge_0_4",
    )
    return {
        name: {
            "assets": len(group),
            **{
                metric: _distribution([float(record[metric]) for record in group])
                for metric in metrics
            },
        }
        for name, group in sorted(groups.items())
    }


def _summary(
    records: list[dict[str, object]],
    placement: dict[str, object],
    comparisons: list[dict[str, object]],
    sample_count: int,
) -> dict[str, object]:
    metrics = (
        "detect_rate",
        "zero_detection_fraction",
        "keypoint_conf_median",
        "keypoint_conf_p25",
        "keypoint_coverage_ge_0_3",
        "keypoint_coverage_ge_0_4",
    )
    comparison_ious = [float(item["min_iou"]) for item in comparisons]
    comparison_deltas = [float(item["max_abs_box_delta_px"]) for item in comparisons]
    return {
        "schema": 1,
        "model": MODEL_NAME,
        "detector_model": "YOLOX-m HumanArt",
        "detector_score_threshold": DETECTOR_THRESHOLD,
        "tracking_scope": "hands-arms",
        "sample_rule": {
            "frames_per_asset": sample_count,
            "rule": "PTS midpoint of each equal-duration bin across the full video stream",
            "time_source": "decoded frame PTS multiplied by frame time_base",
        },
        "placements": placement,
        "assets": len(records),
        "sampled_frames": sum(int(record["samples_inferred"]) for record in records),
        "overall": {
            metric: _distribution([float(record[metric]) for record in records])
            for metric in metrics
        },
        "by_view": _grouped(records, "view"),
        "by_device_config": _grouped(records, "device_config"),
        "by_task": _grouped(records, "task"),
        "orientation": {
            "files_with_timed_track": sum(bool(record["orientation_values"]) for record in records),
            "files_with_multiple_values": sum(
                len(record["orientation_values"]) > 1 for record in records
            ),
            "files_without_timed_track": sum(
                not record["orientation_values"] for record in records
            ),
            "value_set_census": dict(
                sorted(Counter(str(record["orientation_values"]) for record in records).items())
            ),
        },
        "detector_gpu_cpu_check": {
            "frames": len(comparisons),
            "count_agreement": sum(bool(item["count_agrees"]) for item in comparisons),
            "min_iou": _distribution(comparison_ious),
            "max_abs_box_delta_px": _distribution(comparison_deltas),
        },
        "status": {
            "metadata": dict(
                sorted(Counter(str(record["metadata_status"]) for record in records).items())
            ),
            "decode": dict(
                sorted(Counter(str(record["decode_status"]) for record in records).items())
            ),
            "inference_failure_frames": sum(
                sum(record["inference_failures"].values()) for record in records
            ),
        },
    }


def _write_asset_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "asset_id",
        "view",
        "task",
        "device_config",
        "metadata_status",
        "decode_status",
        "samples_requested",
        "samples_inferred",
        "detect_rate",
        "zero_detection_fraction",
        "keypoint_conf_median",
        "keypoint_conf_p25",
        "keypoint_coverage_ge_0_3",
        "keypoint_coverage_ge_0_4",
        "subject_px_height_median",
        "hand_span_px_median",
        "max_person_count",
        "multi_person_frame_fraction",
        "person_count_hist",
        "orientation_values",
        "orientation_sources",
        "face_track_present",
        "face_max",
        "face_samples",
        "inference_failures",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda item: str(item["asset_id"])):
            writer.writerow(
                {
                    field: json.dumps(record[field], sort_keys=True, separators=(",", ":"))
                    if isinstance(record[field], (dict, list))
                    else record[field]
                    for field in fields
                }
            )


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.cpu_check_frames < 0 or args.limit < 0:
        raise SystemExit("samples must be positive; limits must be non-negative")
    with (args.inventory / "assets.csv").open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["disposition"] == "canonical"]
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "pose_assets.jsonl"
    prior: dict[str, dict[str, object]] = {}
    if args.resume and checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                prior[str(record["asset_id"])] = record
    elif checkpoint.exists():
        checkpoint.unlink()

    tracker, cpu_detector, placement = _make_models()
    check_positions = (
        set(
            np.linspace(0, len(rows) - 1, min(args.cpu_check_frames, len(rows)), dtype=int).tolist()
        )
        if rows and args.cpu_check_frames
        else set()
    )
    comparisons: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    start = time.perf_counter()
    mode = "a" if prior else "w"
    with checkpoint.open(mode, encoding="utf-8") as file:
        for index, row in enumerate(rows):
            asset_id = row["asset_id"]
            if asset_id in prior:
                records.append(prior[asset_id])
                continue
            path = args.corpus / row["source_path"]
            cpu_sample = index % args.samples if index in check_positions else None
            try:
                record, comparison = _asset_record(
                    row,
                    path,
                    args.samples,
                    tracker,
                    cpu_detector,
                    cpu_sample,
                )
            except Exception as error:
                record = {
                    "asset_id": asset_id,
                    "view": row["view"],
                    "task": row["task"],
                    "device_config": _device_config(path),
                    "metadata_status": "not_completed",
                    "decode_status": f"error:{type(error).__name__}",
                    "samples_requested": args.samples,
                    "samples_inferred": 0,
                    "inference_failures": {},
                    "detect_rate": math.nan,
                    "zero_detection_fraction": math.nan,
                    "keypoint_conf_median": math.nan,
                    "keypoint_conf_p25": math.nan,
                    "keypoint_coverage_ge_0_3": math.nan,
                    "keypoint_coverage_ge_0_4": math.nan,
                    "subject_px_height_median": math.nan,
                    "hand_span_px_median": math.nan,
                    "person_count_hist": {},
                    "max_person_count": 0,
                    "multi_person_frame_fraction": math.nan,
                    "orientation_values": [],
                    "orientation_sources": {},
                    "face_track_present": False,
                    "face_max": 0,
                    "face_samples": 0,
                }
                comparison = None
            records.append(record)
            file.write(
                json.dumps(record, allow_nan=True, sort_keys=True, separators=(",", ":")) + "\n"
            )
            file.flush()
            if comparison is not None:
                comparisons.append(comparison)
            print(f"asset {index + 1}/{len(rows)} status={record['decode_status']}", flush=True)
    elapsed = time.perf_counter() - start
    summary = _summary(records, placement, comparisons, args.samples)
    (args.out / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_asset_csv(args.out / "pose_assets.csv", records)
    runtime = {
        "wall_s": round(elapsed, 6),
        "assets_processed_this_pass": len(records) - len(prior),
        "assets_resumed": len(prior),
        "sampled_frames_total": summary["sampled_frames"],
        "placements": placement,
    }
    (args.out / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote deterministic summary for {summary['assets']} assets / "
        f"{summary['sampled_frames']} sampled frames in {elapsed:.3f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
