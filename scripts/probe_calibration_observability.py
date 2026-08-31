#!/usr/bin/env python3
"""Measure subject-keypoint calibration observability without publishing identifiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import av
import cv2
import numpy as np

from pose_estimation import inventory, qualify, sessions
from pose_estimation import measure as measurement
from pose_estimation.measure import detect
from pose_estimation.rtmlib_openvino import _patch_rtmlib_openvino

SEED = 20260831
CACHE_SCHEMA = 1
FRAMES_PER_EVENT = 8
STRATUM_EVENTS = 2
KEYPOINT_COUNT = 133
CONFIDENCE_THRESHOLDS = (0.3, 0.5, 0.7)
PRIMARY_CONFIDENCE = 0.5
CALIBRATION_INDICES = (*range(23), *range(91, 133))
KEYPOINT_SETS = {
    "calibration65": CALIBRATION_INDICES,
    "all133": tuple(range(KEYPOINT_COUNT)),
}
FX_PRIOR_1920 = {
    "iPad (5th generation)": 1873.3,
    "iPad Air 11-inch (M2)": 1553.2,
}
CROP_FACTOR_4_3_TO_16_9 = 1.08947
REFERENCE_LONG_DIM_PX = 1920.0
RANSAC_THRESHOLD_PX = 3.0
MIN_POSE_INLIERS = 30
MIN_PARALLAX_DEG = 1.0
MAX_POSE_SPREAD_DEG = 10.0
RANSAC_PROBABILITY = 0.999


@dataclass(frozen=True)
class Asset:
    asset_id: str
    path: Path
    view: str
    task: str
    rotation_deg: int
    duration_s: float
    fps: float
    device_config: str
    model: str


@dataclass(frozen=True)
class Camera:
    asset: Asset
    camera_name: str
    offset_s: float | None
    offset_status: str


@dataclass(frozen=True)
class Event:
    event_id: str
    task: str
    cameras: tuple[Camera, ...]
    overlap_start_s: float | None
    overlap_end_s: float | None
    eligibility: str


@dataclass(frozen=True)
class Inputs:
    events: tuple[Event, ...]
    fingerprint: str
    generations: dict[str, Any]


@dataclass(frozen=True)
class Selection:
    sample: tuple[Event, ...]
    pilot: tuple[Event, ...]
    strata: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Models:
    detector: Any
    pose: Any
    placements: dict[str, dict[str, object]]
    available_devices: tuple[str, ...]


@dataclass(frozen=True)
class EventData:
    meta: dict[str, Any]
    points: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    target_ref_s: np.ndarray
    actual_ref_s: np.ndarray
    widths: np.ndarray
    heights: np.ndarray


@dataclass(frozen=True)
class PoseEstimate:
    rotation: np.ndarray
    translation: np.ndarray
    essential: np.ndarray
    essential_inliers: int
    recover_inliers: int
    recover_mask: np.ndarray


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _model(device_config: str) -> str:
    return next((name for name in FX_PRIOR_1920 if device_config.startswith(name)), "")


def _float(value: str) -> float | None:
    if not value:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _hash_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).hexdigest()


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_inputs(
    corpus_root: Path,
    inventory_dir: Path,
    sessions_dir: Path,
    qualification_dir: Path,
    measurements_dir: Path,
) -> Inputs:
    inventory_generation = inventory.validate_generation(inventory_dir)
    sessions_generation = sessions.validate_generation(sessions_dir, inventory_dir=inventory_dir)
    sidecar = measurement.validate(measurements_dir, inventory_dir=inventory_dir)
    qualification_generation = qualify.validate_generation(
        qualification_dir,
        sessions_dir=sessions_dir,
        inventory_dir=inventory_dir,
        measurements_dir=measurements_dir,
    )

    qc_by_asset = {
        row["asset_id"]: row for row in _read_rows(qualification_dir / qualify.ASSETS_QC_FILENAME)
    }
    assets: dict[str, Asset] = {}
    for row in _read_rows(inventory_dir / inventory.ASSETS_FILENAME):
        if row["disposition"] != inventory.CANONICAL:
            continue
        qc = qc_by_asset[row["asset_id"]]
        config = qc["device_config"]
        assets[row["asset_id"]] = Asset(
            asset_id=row["asset_id"],
            path=sessions.resolve_source(corpus_root, row["source_path"]),
            view=row["view"],
            task=row["task"],
            rotation_deg=int(row["reported_rotation_deg"]),
            duration_s=float(row["nominal_duration_s"]),
            fps=float(row["reported_avg_fps"]),
            device_config=config,
            model=_model(config),
        )

    offset_rows = {
        (row["event_id"], row["asset_id"]): row
        for row in _read_rows(qualification_dir / qualify.CAMERAS_QC_FILENAME)
    }
    placed: dict[str, list[Camera]] = defaultdict(list)
    for row in _read_rows(sessions_dir / sessions.PLACEMENTS_FILENAME):
        if row["placement"] != sessions.PLACED:
            continue
        asset = assets[row["asset_id"]]
        offset_row = offset_rows[(row["event_id"], row["asset_id"])]
        placed[row["event_id"]].append(
            Camera(
                asset=asset,
                camera_name=row["camera_name"],
                offset_s=_float(offset_row["offset_s"]),
                offset_status=offset_row["offset_status"],
            )
        )

    event_ids = sorted(
        row["event_id"] for row in _read_rows(sessions_dir / sessions.EVENTS_FILENAME)
    )
    events: list[Event] = []
    for event_id in event_ids:
        cameras = tuple(sorted(placed[event_id], key=lambda camera: camera.camera_name))
        tasks = {camera.asset.task for camera in cameras}
        if len(tasks) != 1:
            raise RuntimeError("one recording event carries multiple task labels")
        eligibility = "eligible"
        overlap_start: float | None = None
        overlap_end: float | None = None
        if len(cameras) < 2:
            eligibility = "single_camera"
        elif any(
            camera.offset_s is None or camera.offset_status not in qualify.OFFSET_SOLVED_STATUSES
            for camera in cameras
        ):
            eligibility = "offset_incomplete"
        elif any(not camera.asset.model for camera in cameras):
            eligibility = "intrinsics_prior_unavailable"
        else:
            offsets = cast(list[float], [camera.offset_s for camera in cameras])
            overlap_start = max(-offset for offset in offsets)
            overlap_end = min(
                camera.asset.duration_s - offset
                for camera, offset in zip(cameras, offsets, strict=True)
            )
            slowest_period = max(1.0 / camera.asset.fps for camera in cameras)
            if overlap_end - overlap_start < FRAMES_PER_EVENT * slowest_period:
                eligibility = "overlap_below_frame_budget"
        events.append(
            Event(
                event_id=event_id,
                task=next(iter(tasks)),
                cameras=cameras,
                overlap_start_s=overlap_start,
                overlap_end_s=overlap_end,
                eligibility=eligibility,
            )
        )

    generations = {
        "inventory": inventory_generation["generation"],
        "sessions": sessions_generation,
        "measurements": sidecar.manifest["generation"]["manifest"],
        "qualification": qualification_generation,
    }
    fingerprint_payload = {
        "cache_schema": CACHE_SCHEMA,
        "source_sha256": _source_digest(),
        "generations": generations,
        "instrument": {
            "seed": SEED,
            "frames_per_event": FRAMES_PER_EVENT,
            "stratum_events": STRATUM_EVENTS,
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "primary_confidence": PRIMARY_CONFIDENCE,
            "calibration_indices": CALIBRATION_INDICES,
            "fx_prior_1920": FX_PRIOR_1920,
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Inputs(tuple(events), fingerprint, generations)


def select_events(inputs: Inputs) -> Selection:
    strata: dict[tuple[str, int], list[Event]] = defaultdict(list)
    for event in inputs.events:
        if event.eligibility == "eligible":
            strata[(event.task, len(event.cameras))].append(event)
    selected: list[Event] = []
    for stratum in sorted(strata):
        ranked = sorted(
            strata[stratum], key=lambda event: _hash_rank(f"sample:{stratum}", event.event_id)
        )
        selected.extend(ranked[:STRATUM_EVENTS])
    sample = tuple(sorted(selected, key=lambda event: _hash_rank("sample-order", event.event_id)))
    pilot_candidates = [event for event in sample if len(event.cameras) == 3]
    pilot = tuple(
        sorted(pilot_candidates, key=lambda event: _hash_rank("pilot", event.event_id))[:3]
    )
    if len(pilot) != 3:
        raise RuntimeError(
            "the predeclared sample does not contain three eligible three-camera events"
        )
    return Selection(sample, pilot, tuple(sorted(strata)))


def _distribution(values: Sequence[float], digits: int = 6) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    def rounded(value: float) -> float:
        return round(float(value), digits)

    return {
        "n": int(finite.size),
        "min": rounded(np.min(finite)),
        "p25": rounded(np.percentile(finite, 25)),
        "median": rounded(np.median(finite)),
        "mean": rounded(np.mean(finite)),
        "p75": rounded(np.percentile(finite, 75)),
        "p95": rounded(np.percentile(finite, 95)),
        "max": rounded(np.max(finite)),
    }


def plan_summary(inputs: Inputs, selection: Selection) -> dict[str, Any]:
    reasons = Counter(event.eligibility for event in inputs.events)
    eligible = [event for event in inputs.events if event.eligibility == "eligible"]
    sample_cameras = sum(len(event.cameras) for event in selection.sample)
    pilot_cameras = sum(len(event.cameras) for event in selection.pilot)
    return {
        "schema_version": CACHE_SCHEMA,
        "fingerprint": inputs.fingerprint,
        "instrument": {
            "seed": SEED,
            "model": detect.MODEL_NAME,
            "backend": detect.BACKEND,
            "detector_device": "CPU",
            "pose_device": "NPU",
            "keypoints_returned": KEYPOINT_COUNT,
            "calibration_keypoints": len(CALIBRATION_INDICES),
            "calibration_keypoint_rule": "COCO-WholeBody body/feet 0:23 plus hands 91:133; dense face excluded",
            "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
            "primary_confidence": PRIMARY_CONFIDENCE,
            "frames_per_event": FRAMES_PER_EVENT,
            "sampling": "midpoints of equal bins over every camera's common reference-time overlap",
            "frame_selection": "nearest decoded PTS around each target after a backward keyframe seek",
            "offset_application": "target_camera_s = target_reference_s + offset_s",
            "fx_prior_1920_px": FX_PRIOR_1920,
            "crop_factor_4_3_to_16_9": CROP_FACTOR_4_3_TO_16_9,
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
        },
        "population": {
            "recording_events": len(inputs.events),
            "placed_assets": sum(len(event.cameras) for event in inputs.events),
            "eligibility_reason_partition": dict(sorted(reasons.items())),
            "eligible_by_camera_count": dict(
                sorted(Counter(len(event.cameras) for event in eligible).items())
            ),
        },
        "sample": {
            "rule": "up to two fixed-hash-ranked eligible events per nonempty task x camera-count stratum",
            "nonempty_strata": len(selection.strata),
            "events": len(selection.sample),
            "events_by_camera_count": dict(
                sorted(Counter(len(event.cameras) for event in selection.sample).items())
            ),
            "cameras": sample_cameras,
            "view_frames": sample_cameras * FRAMES_PER_EVENT,
            "pilot_events": len(selection.pilot),
            "pilot_cameras": pilot_cameras,
            "pilot_view_frames": pilot_cameras * FRAMES_PER_EVENT,
        },
        "determinism": {
            "sorted_iteration": True,
            "fixed_seed": SEED,
            "wall_clock_in_plan_or_analysis": False,
            "cache_fingerprint_binds_script_and_four_validated_generations": True,
        },
    }


def _execution_devices(model: Any) -> dict[str, object]:
    value = model.compiled_model.get_property("EXECUTION_DEVICES")
    normalized = [value] if isinstance(value, str) else list(value)
    return {"raw_type": type(value).__name__, "devices": [str(item) for item in normalized]}


def _devices(placement: dict[str, object]) -> list[str]:
    values = placement["devices"]
    if not isinstance(values, list):
        raise RuntimeError("OpenVINO returned malformed execution-device metadata")
    return [str(value) for value in values]


def make_models() -> Models:
    from openvino import Core

    available = tuple(Core().available_devices)
    if set(available) != {"CPU", "GPU", "NPU"}:
        raise RuntimeError(f"expected CPU/GPU/NPU; found {available!r}")
    _patch_rtmlib_openvino()
    import rtmlib

    detector = rtmlib.YOLOX(
        detect._DET_URL,
        model_input_size=detect._DET_INPUT_SIZE,
        mode=detect.DETECTOR_MODE,
        nms_thr=detect.DETECTOR_NMS_THRESHOLD,
        score_thr=detect.DETECTOR_SCORE_THRESHOLD,
        backend=detect.BACKEND,
        device="CPU",
    )
    pose_class = getattr(rtmlib, detect.POSE_CLASS)
    pose = pose_class(
        detect.POSE_URL,
        model_input_size=detect.POSE_INPUT_SIZE,
        to_openpose=detect.POSE_TO_OPENPOSE,
        backend=detect.BACKEND,
        device="NPU",
    )
    placements = {"detector": _execution_devices(detector), "pose": _execution_devices(pose)}
    detector_devices = _devices(placements["detector"])
    pose_devices = _devices(placements["pose"])
    if not detector_devices or not all("CPU" in device for device in detector_devices):
        raise RuntimeError(f"detector did not execute exclusively on CPU: {detector_devices!r}")
    if not pose_devices or not all("NPU" in device for device in pose_devices):
        raise RuntimeError(f"pose model did not execute exclusively on NPU: {pose_devices!r}")
    return Models(detector, pose, placements, available)


def _stream_interval(path: Path) -> tuple[float, float]:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError("video stream absent")
        stream = container.streams.video[0]
        if stream.duration is None or stream.duration <= 0:
            raise RuntimeError("video stream duration absent")
        time_base = float(stream.time_base)
        start = float((stream.start_time or 0) * stream.time_base)
        return start, start + float(stream.duration) * time_base


def _nearest_frame(container: Any, stream: Any, target_s: float) -> Any | None:
    time_base = float(stream.time_base)
    target_tick = math.floor(target_s / time_base)
    container.seek(target_tick, stream=stream, any_frame=False, backward=True)
    prior = None
    after = None
    for frame in container.decode(stream):
        if frame.pts is None or frame.time_base is None:
            continue
        pts_s = float(frame.pts * frame.time_base)
        if pts_s >= target_s:
            after = frame
            break
        prior = frame
    candidates = [frame for frame in (prior, after) if frame is not None]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda frame: (
            abs(float(frame.pts * frame.time_base) - target_s),
            float(frame.pts * frame.time_base),
        ),
    )


def _infer_frame(
    frame: np.ndarray, models: Models
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    points = np.full((KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
    scores = np.full(KEYPOINT_COUNT, np.nan, dtype=np.float32)
    box = np.full(4, np.nan, dtype=np.float32)
    boxes = list(models.detector(frame))
    detected = int(bool(boxes))
    try:
        keypoints_raw, scores_raw = models.pose(frame, bboxes=boxes)
    except Exception:
        return points, scores, box, detected, 0, 1
    score_array = None if scores_raw is None else np.asarray(scores_raw, dtype=np.float32)
    point_array = None if keypoints_raw is None else np.asarray(keypoints_raw, dtype=np.float32)
    primary = detect._primary_index(score_array)
    if (
        primary is None
        or point_array is None
        or score_array is None
        or primary >= len(boxes)
        or primary >= len(point_array)
        or point_array[primary].shape != (KEYPOINT_COUNT, 2)
        or score_array[primary].shape != (KEYPOINT_COUNT,)
    ):
        return points, scores, box, detected, 0, 0
    points[:] = point_array[primary]
    scores[:] = score_array[primary]
    box_values = np.asarray(boxes[primary], dtype=np.float32).reshape(-1)
    if box_values.size >= 4:
        box[:] = box_values[:4]
    return points, scores, box, detected, 1, 0


def _sample_camera(
    camera: Camera,
    targets_ref_s: np.ndarray,
    models: Models,
) -> dict[str, np.ndarray | int]:
    frame_count = len(targets_ref_s)
    points = np.full((frame_count, KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
    scores = np.full((frame_count, KEYPOINT_COUNT), np.nan, dtype=np.float32)
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
    actual_ref = np.full(frame_count, np.nan, dtype=np.float64)
    widths = np.zeros(frame_count, dtype=np.int32)
    heights = np.zeros(frame_count, dtype=np.int32)
    detected_frames = 0
    pose_frames = 0
    inference_failures = 0
    decode_failures = 0
    try:
        orientations = detect._timed_orientations(camera.asset.path)
    except Exception:
        orientations = []
    if camera.offset_s is None:
        raise RuntimeError("eligible camera has no offset")
    with av.open(str(camera.asset.path)) as container:
        if not container.streams.video:
            raise RuntimeError("video stream absent")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, target_ref in enumerate(targets_ref_s):
            target_local = float(target_ref + camera.offset_s)
            chosen = _nearest_frame(container, stream, target_local)
            if chosen is None or chosen.pts is None or chosen.time_base is None:
                decode_failures += 1
                continue
            pts_s = float(chosen.pts * chosen.time_base)
            rotation = detect._rotation_at(pts_s, orientations, camera.asset.rotation_deg)
            frame = detect._rotate(chosen.to_ndarray(format="bgr24"), rotation)
            heights[index], widths[index] = frame.shape[:2]
            actual_ref[index] = pts_s - camera.offset_s
            (
                points[index],
                scores[index],
                boxes[index],
                detected,
                posed,
                failed,
            ) = _infer_frame(frame, models)
            detected_frames += detected
            pose_frames += posed
            inference_failures += failed
    return {
        "points": points,
        "scores": scores,
        "boxes": boxes,
        "actual_ref_s": actual_ref,
        "widths": widths,
        "heights": heights,
        "detected_frames": detected_frames,
        "pose_frames": pose_frames,
        "inference_failures": inference_failures,
        "decode_failures": decode_failures,
    }


def _empty_event(event: Event, fingerprint: str, error_type: str) -> EventData:
    camera_count = len(event.cameras)
    shape = (camera_count, FRAMES_PER_EVENT)
    meta = {
        "schema_version": CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "event_key": _event_key(event),
        "task": event.task,
        "views": [camera.asset.view for camera in event.cameras],
        "models": [camera.asset.model for camera in event.cameras],
        "camera_count": camera_count,
        "error_type": error_type,
        "detected_frames": 0,
        "pose_frames": 0,
        "inference_failures": 0,
        "decode_failures": camera_count * FRAMES_PER_EVENT,
    }
    return EventData(
        meta,
        np.full((*shape, KEYPOINT_COUNT, 2), np.nan, dtype=np.float32),
        np.full((*shape, KEYPOINT_COUNT), np.nan, dtype=np.float32),
        np.full((*shape, 4), np.nan, dtype=np.float32),
        np.full(FRAMES_PER_EVENT, np.nan, dtype=np.float64),
        np.full(shape, np.nan, dtype=np.float64),
        np.zeros(shape, dtype=np.int32),
        np.zeros(shape, dtype=np.int32),
    )


def collect_event(event: Event, inputs: Inputs, models: Models) -> EventData:
    try:
        intervals = [_stream_interval(camera.asset.path) for camera in event.cameras]
        offsets = cast(list[float], [camera.offset_s for camera in event.cameras])
        overlap_start = max(
            start - offset for (start, _), offset in zip(intervals, offsets, strict=True)
        )
        overlap_end = min(end - offset for (_, end), offset in zip(intervals, offsets, strict=True))
        if overlap_end <= overlap_start:
            raise RuntimeError("decoded streams have no common reference-time overlap")
        targets = np.asarray(
            [
                overlap_start
                + (2 * index + 1) * (overlap_end - overlap_start) / (2 * FRAMES_PER_EVENT)
                for index in range(FRAMES_PER_EVENT)
            ],
            dtype=np.float64,
        )
        sampled = [_sample_camera(camera, targets, models) for camera in event.cameras]
    except Exception as error:
        return _empty_event(event, inputs.fingerprint, type(error).__name__)

    meta = {
        "schema_version": CACHE_SCHEMA,
        "fingerprint": inputs.fingerprint,
        "event_key": _event_key(event),
        "task": event.task,
        "views": [camera.asset.view for camera in event.cameras],
        "models": [camera.asset.model for camera in event.cameras],
        "camera_count": len(event.cameras),
        "error_type": "",
        "detected_frames": sum(int(item["detected_frames"]) for item in sampled),
        "pose_frames": sum(int(item["pose_frames"]) for item in sampled),
        "inference_failures": sum(int(item["inference_failures"]) for item in sampled),
        "decode_failures": sum(int(item["decode_failures"]) for item in sampled),
        "overlap_s": overlap_end - overlap_start,
    }
    return EventData(
        meta=meta,
        points=np.stack([cast(np.ndarray, item["points"]) for item in sampled]),
        scores=np.stack([cast(np.ndarray, item["scores"]) for item in sampled]),
        boxes=np.stack([cast(np.ndarray, item["boxes"]) for item in sampled]),
        target_ref_s=targets,
        actual_ref_s=np.stack([cast(np.ndarray, item["actual_ref_s"]) for item in sampled]),
        widths=np.stack([cast(np.ndarray, item["widths"]) for item in sampled]),
        heights=np.stack([cast(np.ndarray, item["heights"]) for item in sampled]),
    )


def _event_key(event: Event) -> str:
    return hashlib.sha256(event.event_id.encode()).hexdigest()[:24]


def _cache_path(cache_dir: Path, event: Event) -> Path:
    return cache_dir / f"{_event_key(event)}.npz"


def _metadata_array(meta: dict[str, Any]) -> np.ndarray:
    payload = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    return np.frombuffer(payload, dtype=np.uint8)


def save_event(path: Path, data: EventData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        meta=_metadata_array(data.meta),
        points=data.points,
        scores=data.scores,
        boxes=data.boxes,
        target_ref_s=data.target_ref_s,
        actual_ref_s=data.actual_ref_s,
        widths=data.widths,
        heights=data.heights,
    )
    temporary.replace(path)


def load_event(path: Path, fingerprint: str, event: Event) -> EventData | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = {
                "meta",
                "points",
                "scores",
                "boxes",
                "target_ref_s",
                "actual_ref_s",
                "widths",
                "heights",
            }
            if set(archive.files) != expected:
                return None
            meta = json.loads(archive["meta"].tobytes().decode())
            if (
                meta.get("schema_version") != CACHE_SCHEMA
                or meta.get("fingerprint") != fingerprint
                or meta.get("event_key") != _event_key(event)
            ):
                return None
            return EventData(
                meta=meta,
                points=archive["points"].copy(),
                scores=archive["scores"].copy(),
                boxes=archive["boxes"].copy(),
                target_ref_s=archive["target_ref_s"].copy(),
                actual_ref_s=archive["actual_ref_s"].copy(),
                widths=archive["widths"].copy(),
                heights=archive["heights"].copy(),
            )
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def collect(
    inputs: Inputs,
    events: Sequence[Event],
    cache_dir: Path,
    *,
    force: bool,
    scope: str,
) -> dict[str, Any]:
    pending = [
        event
        for event in events
        if force or load_event(_cache_path(cache_dir, event), inputs.fingerprint, event) is None
    ]
    model_startup_s = 0.0
    models: Models | None = None
    if pending:
        started = time.perf_counter()
        models = make_models()
        model_startup_s = time.perf_counter() - started
    timings: list[dict[str, float | int | str]] = []
    for ordinal, event in enumerate(events, start=1):
        path = _cache_path(cache_dir, event)
        cached = None if force else load_event(path, inputs.fingerprint, event)
        if cached is not None:
            print(f"cache {ordinal}/{len(events)}")
            continue
        if models is None:
            raise RuntimeError("models were not initialized for an uncached event")
        print(f"collect {ordinal}/{len(events)}")
        started = time.perf_counter()
        data = collect_event(event, inputs, models)
        elapsed = time.perf_counter() - started
        save_event(path, data)
        timings.append(
            {
                "event_key": _event_key(event),
                "camera_count": len(event.cameras),
                "view_frames": len(event.cameras) * FRAMES_PER_EVENT,
                "elapsed_s": elapsed,
            }
        )
    timing = {
        "schema_version": CACHE_SCHEMA,
        "fingerprint": inputs.fingerprint,
        "scope": scope,
        "model_startup_s": model_startup_s,
        "events": timings,
    }
    _write_json(cache_dir / "timing.json", timing)
    placements = models.placements if models is not None else {}
    return {
        "scope": scope,
        "events_requested": len(events),
        "cache_hits": len(events) - len(timings),
        "events_collected": len(timings),
        "model_startup_s": round(model_startup_s, 3),
        "execution_devices": placements,
    }


def _valid(points: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & np.isfinite(scores) & (scores >= threshold)


def _normalized(points: np.ndarray, width: int, height: int, model: str) -> np.ndarray:
    fx = FX_PRIOR_1920[model] * max(width, height) / REFERENCE_LONG_DIM_PX
    normalized = np.empty_like(points, dtype=np.float64)
    normalized[:, 0] = (points[:, 0] - width / 2.0) / fx
    normalized[:, 1] = (points[:, 1] - height / 2.0) / fx
    return normalized


def _hartley(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    if mean_distance <= np.finfo(np.float64).eps:
        return centered
    return centered * (math.sqrt(2.0) / mean_distance)


def _design_statistics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if len(left) < 9:
        return {"rank8_ratio": math.nan, "null_ratio": math.nan, "condition8": math.nan}
    x1 = _hartley(left)
    x2 = _hartley(right)
    A = np.column_stack(
        [
            x2[:, 0] * x1[:, 0],
            x2[:, 0] * x1[:, 1],
            x2[:, 0],
            x2[:, 1] * x1[:, 0],
            x2[:, 1] * x1[:, 1],
            x2[:, 1],
            x1[:, 0],
            x1[:, 1],
            np.ones(len(x1)),
        ]
    )
    singular = np.linalg.svd(A, compute_uv=False)
    if len(singular) < 9 or singular[0] <= 0 or singular[7] <= 0:
        return {"rank8_ratio": math.nan, "null_ratio": math.nan, "condition8": math.nan}
    return {
        "rank8_ratio": float(singular[7] / singular[0]),
        "null_ratio": float(singular[8] / singular[7]),
        "condition8": float(singular[0] / singular[7]),
    }


def _estimate_pose(left: np.ndarray, right: np.ndarray, threshold: float) -> PoseEstimate | None:
    if len(left) < 8:
        return None
    cv2.setRNGSeed(SEED)
    try:
        essential, mask = cv2.findEssentialMat(
            left,
            right,
            np.eye(3),
            cv2.USAC_MAGSAC,
            RANSAC_PROBABILITY,
            threshold,
        )
    except cv2.error:
        return None
    if essential is None or mask is None:
        return None
    candidates: list[np.ndarray]
    if essential.shape == (3, 3):
        candidates = [essential]
    elif essential.ndim == 2 and essential.shape[1] == 3 and essential.shape[0] % 3 == 0:
        candidates = [essential[index : index + 3] for index in range(0, len(essential), 3)]
    else:
        return None
    best: PoseEstimate | None = None
    essential_inliers = int(np.count_nonzero(mask))
    for candidate in candidates:
        try:
            count, rotation, translation, recovered = cv2.recoverPose(
                candidate, left, right, np.eye(3), mask=mask.copy()
            )
        except cv2.error:
            continue
        estimate = PoseEstimate(
            rotation=np.asarray(rotation, dtype=np.float64),
            translation=np.asarray(translation, dtype=np.float64).reshape(3),
            essential=np.asarray(candidate, dtype=np.float64),
            essential_inliers=essential_inliers,
            recover_inliers=int(count),
            recover_mask=np.asarray(recovered).reshape(-1).astype(bool),
        )
        if best is None or estimate.recover_inliers > best.recover_inliers:
            best = estimate
    return best


def _homography_inliers(left: np.ndarray, right: np.ndarray, threshold: float) -> int:
    if len(left) < 4:
        return 0
    cv2.setRNGSeed(SEED)
    try:
        _, mask = cv2.findHomography(
            left,
            right,
            cv2.USAC_MAGSAC,
            threshold,
            maxIters=10_000,
            confidence=RANSAC_PROBABILITY,
        )
    except cv2.error:
        return 0
    return int(np.count_nonzero(mask)) if mask is not None else 0


def _rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    relative = left @ right.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_spread(rotations: Sequence[np.ndarray]) -> float:
    if len(rotations) < 2:
        return math.nan
    return max(
        _rotation_angle(rotations[left], rotations[right])
        for left, right in combinations(range(len(rotations)), 2)
    )


def _translation_axis_spread(translations: Sequence[np.ndarray]) -> float:
    if len(translations) < 2:
        return math.nan
    unit = [translation / np.linalg.norm(translation) for translation in translations]
    return max(
        float(np.degrees(np.arccos(np.clip(abs(np.dot(unit[left], unit[right])), -1.0, 1.0))))
        for left, right in combinations(range(len(unit)), 2)
    )


def _parallax(
    left: np.ndarray,
    right: np.ndarray,
    rotation: np.ndarray,
    keep: np.ndarray,
) -> float:
    if not np.count_nonzero(keep):
        return math.nan
    ray_left = np.column_stack([left[keep], np.ones(np.count_nonzero(keep))]).T
    ray_right = np.column_stack([right[keep], np.ones(np.count_nonzero(keep))]).T
    ray_left /= np.linalg.norm(ray_left, axis=0)
    ray_right = rotation.T @ (ray_right / np.linalg.norm(ray_right, axis=0))
    cosine = np.clip(np.sum(ray_left * ray_right, axis=0), -1.0, 1.0)
    return float(np.median(np.degrees(np.arccos(cosine))))


def _epipolar_inlier_ratio(
    estimate: PoseEstimate,
    left: np.ndarray,
    right: np.ndarray,
    threshold: float,
) -> float:
    if not len(left):
        return math.nan
    homogeneous_left = np.column_stack([left, np.ones(len(left))])
    homogeneous_right = np.column_stack([right, np.ones(len(right))])
    lines_right = (estimate.essential @ homogeneous_left.T).T
    lines_left = (estimate.essential.T @ homogeneous_right.T).T
    numerators = np.sum(homogeneous_right * lines_right, axis=1) ** 2
    denominators = 1.0 / np.maximum(
        lines_left[:, 0] ** 2 + lines_left[:, 1] ** 2,
        np.finfo(np.float64).eps,
    ) + 1.0 / np.maximum(
        lines_right[:, 0] ** 2 + lines_right[:, 1] ** 2,
        np.finfo(np.float64).eps,
    )
    symmetric_distance = np.sqrt(numerators * denominators)
    return float(np.mean(symmetric_distance <= threshold))


def _pair_geometry(
    data: EventData,
    left_camera: int,
    right_camera: int,
    indices: Sequence[int],
) -> dict[str, Any]:
    frame_sets: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    focal_values: list[float] = []
    models = cast(list[str], data.meta["models"])
    index_array = np.asarray(indices, dtype=int)
    for frame in range(FRAMES_PER_EVENT):
        points_left = data.points[left_camera, frame, index_array]
        points_right = data.points[right_camera, frame, index_array]
        scores_left = data.scores[left_camera, frame, index_array]
        scores_right = data.scores[right_camera, frame, index_array]
        shared = _valid(points_left, scores_left, PRIMARY_CONFIDENCE) & _valid(
            points_right, scores_right, PRIMARY_CONFIDENCE
        )
        if not np.any(shared):
            continue
        width_left = int(data.widths[left_camera, frame])
        height_left = int(data.heights[left_camera, frame])
        width_right = int(data.widths[right_camera, frame])
        height_right = int(data.heights[right_camera, frame])
        if min(width_left, height_left, width_right, height_right) <= 0:
            continue
        left = _normalized(points_left[shared], width_left, height_left, models[left_camera])
        right = _normalized(points_right[shared], width_right, height_right, models[right_camera])
        fx_left = FX_PRIOR_1920[models[left_camera]] * max(width_left, height_left) / 1920.0
        fx_right = FX_PRIOR_1920[models[right_camera]] * max(width_right, height_right) / 1920.0
        threshold = RANSAC_THRESHOLD_PX / ((fx_left + fx_right) / 2.0)
        frame_sets.append((frame, left, right, threshold))
        focal_values.append((fx_left + fx_right) / 2.0)

    empty = {
        "shared": 0,
        "rank8_ratio": math.nan,
        "null_ratio": math.nan,
        "condition8": math.nan,
        "homography_inliers": 0,
        "homography_ratio": math.nan,
        "essential_inliers": 0,
        "recover_inliers": 0,
        "recover_ratio": math.nan,
        "parallax_deg": math.nan,
        "baseline_rotation_deg": math.nan,
        "frame_poses": 0,
        "rotation_spread_deg": math.nan,
        "translation_axis_spread_deg": math.nan,
        "translation_sign_flips": 0,
        "split_pose_sets": 0,
        "split_rotation_spread_deg": math.nan,
        "split_translation_axis_spread_deg": math.nan,
        "split_translation_sign_flips": 0,
        "heldout_ratio": math.nan,
        "heldout_min_ratio": math.nan,
        "pose_out": False,
        "quality_pose": False,
        "_rotation": None,
        "_translation": None,
        "flags": ["insufficient_shared"],
    }
    if not frame_sets:
        return empty

    def grouped(frame_indices: set[int]) -> tuple[np.ndarray, np.ndarray, float] | None:
        selected = [item for item in frame_sets if item[0] in frame_indices]
        if not selected:
            return None
        return (
            np.concatenate([item[1] for item in selected]),
            np.concatenate([item[2] for item in selected]),
            float(np.median([item[3] for item in selected])),
        )

    left = np.concatenate([item[1] for item in frame_sets])
    right = np.concatenate([item[2] for item in frame_sets])
    threshold = RANSAC_THRESHOLD_PX / float(np.median(focal_values))
    design = _design_statistics(left, right)
    homography_inliers = _homography_inliers(left, right, threshold)
    pooled_estimate = _estimate_pose(left, right, threshold)
    frame_estimates = [
        frame_estimate
        for _, points_left, points_right, frame_threshold in frame_sets
        if (frame_estimate := _estimate_pose(points_left, points_right, frame_threshold))
        is not None
        and frame_estimate.recover_inliers >= MIN_POSE_INLIERS
    ]

    split_groups = (
        set(range(FRAMES_PER_EVENT // 2)),
        set(range(FRAMES_PER_EVENT // 2, FRAMES_PER_EVENT)),
        set(range(0, FRAMES_PER_EVENT, 2)),
        set(range(1, FRAMES_PER_EVENT, 2)),
    )
    split_estimates: list[PoseEstimate] = []
    for group in split_groups:
        subset = grouped(group)
        if subset is None:
            continue
        estimate = _estimate_pose(*subset)
        if estimate is not None and estimate.recover_inliers >= MIN_POSE_INLIERS:
            split_estimates.append(estimate)

    heldout_ratios: list[float] = []
    for train_indices, test_indices in (
        (split_groups[0], split_groups[1]),
        (split_groups[1], split_groups[0]),
        (split_groups[2], split_groups[3]),
        (split_groups[3], split_groups[2]),
    ):
        train = grouped(train_indices)
        test = grouped(test_indices)
        if train is None or test is None:
            continue
        train_estimate = _estimate_pose(*train)
        if train_estimate is None or train_estimate.recover_inliers < MIN_POSE_INLIERS:
            continue
        heldout_ratios.append(_epipolar_inlier_ratio(train_estimate, *test))

    rotations = [estimate.rotation for estimate in frame_estimates]
    translations = [estimate.translation for estimate in frame_estimates]
    split_rotations = [estimate.rotation for estimate in split_estimates]
    split_translations = [estimate.translation for estimate in split_estimates]

    def sign_flips(vectors: Sequence[np.ndarray]) -> int:
        if not vectors:
            return 0
        reference = vectors[0] / np.linalg.norm(vectors[0])
        return sum(np.dot(reference, vector / np.linalg.norm(vector)) < 0 for vector in vectors[1:])

    essential_inliers = pooled_estimate.essential_inliers if pooled_estimate is not None else 0
    recover_inliers = pooled_estimate.recover_inliers if pooled_estimate is not None else 0
    parallax = (
        _parallax(left, right, pooled_estimate.rotation, pooled_estimate.recover_mask)
        if pooled_estimate is not None
        else math.nan
    )
    rotation_spread = _rotation_spread(rotations)
    translation_spread = _translation_axis_spread(translations)
    split_rotation_spread = _rotation_spread(split_rotations)
    split_translation_spread = _translation_axis_spread(split_translations)
    frame_sign_flips = sign_flips(translations)
    split_sign_flips = sign_flips(split_translations)
    heldout_ratio = float(np.median(heldout_ratios)) if heldout_ratios else math.nan
    heldout_min = min(heldout_ratios, default=math.nan)

    flags: list[str] = []
    if len(left) < MIN_POSE_INLIERS:
        flags.append("insufficient_shared")
    if pooled_estimate is None:
        flags.append("essential_or_pose_absent")
    elif recover_inliers < MIN_POSE_INLIERS:
        flags.append("recover_below_30")
    if homography_inliers >= essential_inliers and homography_inliers >= MIN_POSE_INLIERS:
        flags.append("homography_dominant")
    if math.isfinite(parallax) and parallax < MIN_PARALLAX_DEG:
        flags.append("low_parallax")
    if len(frame_estimates) < 2:
        flags.append("fewer_than_two_frame_poses")
    if math.isfinite(rotation_spread) and rotation_spread > MAX_POSE_SPREAD_DEG:
        flags.append("rotation_inconsistent")
    if math.isfinite(translation_spread) and translation_spread > MAX_POSE_SPREAD_DEG:
        flags.append("translation_axis_inconsistent")
    if frame_sign_flips:
        flags.append("translation_sign_flip")
    if len(split_estimates) < 2:
        flags.append("fewer_than_two_split_poses")
    if math.isfinite(split_rotation_spread) and split_rotation_spread > MAX_POSE_SPREAD_DEG:
        flags.append("split_rotation_inconsistent")
    if math.isfinite(split_translation_spread) and split_translation_spread > MAX_POSE_SPREAD_DEG:
        flags.append("split_translation_axis_inconsistent")
    if split_sign_flips:
        flags.append("split_translation_sign_flip")
    if not heldout_ratios:
        flags.append("heldout_unmeasured")
    elif heldout_ratio < 0.5:
        flags.append("heldout_support_below_half")

    return {
        "shared": len(left),
        **design,
        "homography_inliers": homography_inliers,
        "homography_ratio": homography_inliers / len(left),
        "essential_inliers": essential_inliers,
        "essential_ratio": essential_inliers / len(left),
        "recover_inliers": recover_inliers,
        "recover_ratio": recover_inliers / len(left),
        "parallax_deg": parallax,
        "baseline_rotation_deg": (
            _rotation_angle(pooled_estimate.rotation, np.eye(3))
            if pooled_estimate is not None
            else math.nan
        ),
        "frame_poses": len(frame_estimates),
        "rotation_spread_deg": rotation_spread,
        "translation_axis_spread_deg": translation_spread,
        "translation_sign_flips": frame_sign_flips,
        "split_pose_sets": len(split_estimates),
        "split_rotation_spread_deg": split_rotation_spread,
        "split_translation_axis_spread_deg": split_translation_spread,
        "split_translation_sign_flips": split_sign_flips,
        "heldout_ratio": heldout_ratio,
        "heldout_min_ratio": heldout_min,
        "pose_out": pooled_estimate is not None and recover_inliers > 0,
        "quality_pose": pooled_estimate is not None and recover_inliers >= MIN_POSE_INLIERS,
        "_rotation": pooled_estimate.rotation if pooled_estimate is not None else None,
        "_translation": pooled_estimate.translation if pooled_estimate is not None else None,
        "flags": flags,
    }


def _motion(data: EventData, indices: Sequence[int]) -> dict[str, float]:
    raw_steps: list[float] = []
    centered_steps: list[float] = []
    spans: list[float] = []
    index_array = np.asarray(indices, dtype=int)
    for camera in range(data.points.shape[0]):
        tracks: dict[int, list[np.ndarray]] = defaultdict(list)
        for frame in range(FRAMES_PER_EVENT):
            width = int(data.widths[camera, frame])
            height = int(data.heights[camera, frame])
            if min(width, height) <= 0:
                continue
            points = data.points[camera, frame, index_array]
            scores = data.scores[camera, frame, index_array]
            valid = _valid(points, scores, PRIMARY_CONFIDENCE)
            normalized = points.astype(np.float64)
            normalized[:, 0] /= width
            normalized[:, 1] /= height
            box = data.boxes[camera, frame]
            if np.isfinite(box).all():
                center = np.asarray(
                    [(box[0] + box[2]) / (2.0 * width), (box[1] + box[3]) / (2.0 * height)]
                )
            else:
                center = np.nanmean(normalized[valid], axis=0) if np.any(valid) else np.zeros(2)
            for local_index in np.flatnonzero(valid):
                tracks[int(local_index)].append(
                    np.asarray(
                        [
                            frame,
                            normalized[local_index, 0],
                            normalized[local_index, 1],
                            normalized[local_index, 0] - center[0],
                            normalized[local_index, 1] - center[1],
                        ]
                    )
                )
        for observations in tracks.values():
            ordered = sorted(observations, key=lambda row: row[0])
            if len(ordered) < 2:
                continue
            raw = np.asarray([row[1:3] for row in ordered])
            centered = np.asarray([row[3:5] for row in ordered])
            raw_steps.extend(np.linalg.norm(np.diff(raw, axis=0), axis=1).tolist())
            centered_steps.extend(np.linalg.norm(np.diff(centered, axis=0), axis=1).tolist())
            spans.append(
                max(
                    float(np.linalg.norm(left - right)) for left, right in combinations(centered, 2)
                )
            )
    return {
        "raw_step_median": float(np.median(raw_steps)) if raw_steps else math.nan,
        "raw_step_p95": float(np.percentile(raw_steps, 95)) if raw_steps else math.nan,
        "centered_step_median": (float(np.median(centered_steps)) if centered_steps else math.nan),
        "centered_step_p95": (
            float(np.percentile(centered_steps, 95)) if centered_steps else math.nan
        ),
        "centered_span_median": float(np.median(spans)) if spans else math.nan,
        "centered_span_p95": float(np.percentile(spans, 95)) if spans else math.nan,
    }


def _connected(camera_count: int, pair_results: Sequence[tuple[int, int, dict[str, Any]]]) -> bool:
    reached = {0}
    changed = True
    while changed:
        changed = False
        for left, right, result in pair_results:
            if not result["quality_pose"]:
                continue
            if left in reached and right not in reached:
                reached.add(right)
                changed = True
            if right in reached and left not in reached:
                reached.add(left)
                changed = True
    return len(reached) == camera_count


def _scope_analysis(events: Sequence[Event], data_by_key: dict[str, EventData]) -> dict[str, Any]:
    confident_by_view: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    returned_by_view: dict[str, list[float]] = defaultdict(list)
    shared_by_threshold: dict[str, list[float]] = defaultdict(list)
    shared_by_view_pair: dict[str, list[float]] = defaultdict(list)
    calibration_shared: list[float] = []
    calibration_shared_by_view_pair: dict[str, list[float]] = defaultdict(list)
    calibration_index_array = np.asarray(CALIBRATION_INDICES, dtype=int)
    sync_residual_ms: list[float] = []
    errors = Counter()
    pose_frames = 0
    detected_frames = 0
    inference_failures = 0
    decode_failures = 0
    event_metrics_by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_metrics_by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flag_counts_by_set: dict[str, Counter[str]] = defaultdict(Counter)

    for event in events:
        data = data_by_key[_event_key(event)]
        if data.meta["error_type"]:
            errors[str(data.meta["error_type"])] += 1
        pose_frames += int(data.meta["pose_frames"])
        detected_frames += int(data.meta["detected_frames"])
        inference_failures += int(data.meta["inference_failures"])
        decode_failures += int(data.meta["decode_failures"])
        views = cast(list[str], data.meta["views"])
        for camera, view in enumerate(views):
            for frame in range(FRAMES_PER_EVENT):
                points = data.points[camera, frame]
                scores = data.scores[camera, frame]
                returned_by_view[view].append(
                    float(np.count_nonzero(np.isfinite(points).all(axis=-1) & np.isfinite(scores)))
                )
                for threshold in CONFIDENCE_THRESHOLDS:
                    confident_by_view[view][f"{threshold:.1f}"].append(
                        float(np.count_nonzero(_valid(points, scores, threshold)))
                    )
        for left, right in combinations(range(len(event.cameras)), 2):
            view_pair = "|".join(sorted((views[left], views[right])))
            for frame in range(FRAMES_PER_EVENT):
                if np.isfinite(data.actual_ref_s[[left, right], frame]).all():
                    sync_residual_ms.append(
                        abs(data.actual_ref_s[left, frame] - data.actual_ref_s[right, frame])
                        * 1000.0
                    )
                for threshold in CONFIDENCE_THRESHOLDS:
                    mask = _valid(
                        data.points[left, frame], data.scores[left, frame], threshold
                    ) & _valid(data.points[right, frame], data.scores[right, frame], threshold)
                    count = float(np.count_nonzero(mask))
                    shared_by_threshold[f"{threshold:.1f}"].append(count)
                    if threshold == PRIMARY_CONFIDENCE:
                        shared_by_view_pair[view_pair].append(count)
                        calibration_count = float(np.count_nonzero(mask[calibration_index_array]))
                        calibration_shared.append(calibration_count)
                        calibration_shared_by_view_pair[view_pair].append(calibration_count)

        motion = _motion(data, CALIBRATION_INDICES)
        for set_name, indices in KEYPOINT_SETS.items():
            pair_results: list[tuple[int, int, dict[str, Any]]] = []
            for left, right in combinations(range(len(event.cameras)), 2):
                result = _pair_geometry(data, left, right, indices)
                pair_results.append((left, right, result))
                pair_metrics_by_set[set_name].append(result)
                flag_counts_by_set[set_name].update(result["flags"])
            finite_pairs = [result for _, _, result in pair_results]
            rotation_cycle = math.nan
            quality_rotation_cycle = math.nan
            if len(event.cameras) == 3:
                pair_lookup = {(left, right): result for left, right, result in pair_results}
                if all(
                    pair_lookup[pair]["_rotation"] is not None for pair in ((0, 1), (0, 2), (1, 2))
                ):
                    rotation_01 = cast(np.ndarray, pair_lookup[(0, 1)]["_rotation"])
                    rotation_02 = cast(np.ndarray, pair_lookup[(0, 2)]["_rotation"])
                    rotation_12 = cast(np.ndarray, pair_lookup[(1, 2)]["_rotation"])
                    rotation_cycle = _rotation_angle(rotation_02, rotation_12 @ rotation_01)
                    if all(pair_lookup[pair]["quality_pose"] for pair in ((0, 1), (0, 2), (1, 2))):
                        quality_rotation_cycle = rotation_cycle
            event_metrics_by_set[set_name].append(
                {
                    "pose_graph_connected": _connected(len(event.cameras), pair_results),
                    "rotation_cycle_deg": rotation_cycle,
                    "quality_rotation_cycle_deg": quality_rotation_cycle,
                    "worst_rank8_ratio": min(
                        (
                            result["rank8_ratio"]
                            for result in finite_pairs
                            if math.isfinite(result["rank8_ratio"])
                        ),
                        default=math.nan,
                    ),
                    "worst_null_ratio": max(
                        (
                            result["null_ratio"]
                            for result in finite_pairs
                            if math.isfinite(result["null_ratio"])
                        ),
                        default=math.nan,
                    ),
                    "max_homography_ratio": max(
                        (
                            result.get("homography_ratio", math.nan)
                            for result in finite_pairs
                            if math.isfinite(result.get("homography_ratio", math.nan))
                        ),
                        default=math.nan,
                    ),
                    **motion,
                }
            )

    geometry: dict[str, Any] = {}
    pose: dict[str, Any] = {}
    for set_name in KEYPOINT_SETS:
        pairs = pair_metrics_by_set[set_name]
        event_metrics = event_metrics_by_set[set_name]
        geometry[set_name] = {
            "event_worst_rank8_ratio": _distribution(
                [item["worst_rank8_ratio"] for item in event_metrics], 9
            ),
            "event_worst_null_ratio": _distribution(
                [item["worst_null_ratio"] for item in event_metrics], 6
            ),
            "event_max_homography_ratio": _distribution(
                [item["max_homography_ratio"] for item in event_metrics], 6
            ),
            "pair_rank8_ratio": _distribution([item["rank8_ratio"] for item in pairs], 9),
            "pair_null_ratio": _distribution([item["null_ratio"] for item in pairs], 6),
            "pair_homography_ratio": _distribution(
                [item.get("homography_ratio", math.nan) for item in pairs], 6
            ),
        }
        pose[set_name] = {
            "camera_pairs": len(pairs),
            "pose_out_pairs": sum(bool(item["pose_out"]) for item in pairs),
            "quality_pose_pairs": sum(bool(item["quality_pose"]) for item in pairs),
            "pose_graph_connected_events": sum(
                bool(item["pose_graph_connected"]) for item in event_metrics
            ),
            "shared_correspondences": _distribution([float(item["shared"]) for item in pairs], 3),
            "homography_inliers": _distribution(
                [float(item["homography_inliers"]) for item in pairs], 3
            ),
            "essential_inliers": _distribution(
                [float(item["essential_inliers"]) for item in pairs], 3
            ),
            "recover_inliers": _distribution([float(item["recover_inliers"]) for item in pairs], 3),
            "recover_ratio": _distribution(
                [item.get("recover_ratio", math.nan) for item in pairs], 6
            ),
            "parallax_deg": _distribution([item["parallax_deg"] for item in pairs], 6),
            "baseline_rotation_deg": _distribution(
                [item["baseline_rotation_deg"] for item in pairs], 6
            ),
            "frame_poses": _distribution([float(item["frame_poses"]) for item in pairs], 3),
            "rotation_spread_deg": _distribution(
                [item["rotation_spread_deg"] for item in pairs], 6
            ),
            "translation_axis_spread_deg": _distribution(
                [item["translation_axis_spread_deg"] for item in pairs], 6
            ),
            "split_pose_sets": _distribution([float(item["split_pose_sets"]) for item in pairs], 3),
            "split_rotation_spread_deg": _distribution(
                [item["split_rotation_spread_deg"] for item in pairs], 6
            ),
            "split_translation_axis_spread_deg": _distribution(
                [item["split_translation_axis_spread_deg"] for item in pairs], 6
            ),
            "heldout_ratio": _distribution([item["heldout_ratio"] for item in pairs], 6),
            "heldout_min_ratio": _distribution([item["heldout_min_ratio"] for item in pairs], 6),
            "rotation_cycle_deg": _distribution(
                [item["rotation_cycle_deg"] for item in event_metrics], 6
            ),
            "quality_rotation_cycle_deg": _distribution(
                [item["quality_rotation_cycle_deg"] for item in event_metrics], 6
            ),
            "rotation_cycle_consistent_events": sum(
                math.isfinite(item["rotation_cycle_deg"])
                and item["rotation_cycle_deg"] <= MAX_POSE_SPREAD_DEG
                for item in event_metrics
            ),
            "degeneracy_flags": dict(sorted(flag_counts_by_set[set_name].items())),
        }

    calibration_events = event_metrics_by_set["calibration65"]
    motion_keys = (
        "raw_step_median",
        "raw_step_p95",
        "centered_step_median",
        "centered_step_p95",
        "centered_span_median",
        "centered_span_p95",
    )
    return {
        "events": len(events),
        "events_by_camera_count": dict(
            sorted(Counter(len(event.cameras) for event in events).items())
        ),
        "view_frames_budget": sum(len(event.cameras) for event in events) * FRAMES_PER_EVENT,
        "event_errors": dict(sorted(errors.items())),
        "detected_frames": detected_frames,
        "pose_frames": pose_frames,
        "inference_failures": inference_failures,
        "decode_failures": decode_failures,
        "returned_keypoints_per_view_frame": {
            view: _distribution(values, 3) for view, values in sorted(returned_by_view.items())
        },
        "confident_keypoints_per_view_frame": {
            view: {
                threshold: _distribution(values, 3)
                for threshold, values in sorted(thresholds.items())
            }
            for view, thresholds in sorted(confident_by_view.items())
        },
        "shared_keypoints_per_pair_frame": {
            threshold: _distribution(values, 3)
            for threshold, values in sorted(shared_by_threshold.items())
        },
        "shared_keypoints_at_0_5_by_view_pair": {
            pair: _distribution(values, 3) for pair, values in sorted(shared_by_view_pair.items())
        },
        "shared_calibration65_at_0_5_per_pair_frame": _distribution(calibration_shared, 3),
        "shared_calibration65_at_0_5_by_view_pair": {
            pair: _distribution(values, 3)
            for pair, values in sorted(calibration_shared_by_view_pair.items())
        },
        "sync_residual_ms": _distribution(sync_residual_ms, 6),
        "geometry": geometry,
        "motion_normalized_image_units": {
            key: _distribution([item[key] for item in calibration_events], 6) for key in motion_keys
        },
        "relative_pose": pose,
    }


def analyze(inputs: Inputs, selection: Selection, cache_dir: Path, *, scope: str) -> dict[str, Any]:
    requested = selection.pilot if scope == "pilot" else selection.sample
    data_by_key: dict[str, EventData] = {}
    missing = 0
    for event in requested:
        data = load_event(_cache_path(cache_dir, event), inputs.fingerprint, event)
        if data is None:
            missing += 1
        else:
            data_by_key[_event_key(event)] = data
    if missing:
        raise RuntimeError(f"{missing} selected event caches are absent or stale")
    output = {
        "schema_version": CACHE_SCHEMA,
        "fingerprint": inputs.fingerprint,
        "instrument": plan_summary(inputs, selection)["instrument"],
        "population": plan_summary(inputs, selection)["population"],
        "sample_design": plan_summary(inputs, selection)["sample"],
        "pilot": _scope_analysis(selection.pilot, data_by_key),
    }
    if scope == "sample":
        output["sample"] = _scope_analysis(selection.sample, data_by_key)
    return output


def timing_summary(inputs: Inputs, selection: Selection, cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "timing.json"
    try:
        timing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("timing cache is absent or invalid") from error
    if (
        timing.get("schema_version") != CACHE_SCHEMA
        or timing.get("fingerprint") != inputs.fingerprint
        or timing.get("scope") != "sample"
    ):
        raise RuntimeError("timing cache is stale or was not measured on the scaled sample")
    rows = timing.get("events")
    if not isinstance(rows, list) or len(rows) != len(selection.sample):
        raise RuntimeError("timing cache does not cover every scaled-sample event")
    by_camera_count: dict[int, list[float]] = defaultdict(list)
    all_seconds: list[float] = []
    for row in rows:
        camera_count = int(row["camera_count"])
        elapsed = float(row["elapsed_s"])
        by_camera_count[camera_count].append(elapsed)
        all_seconds.append(elapsed)
    eligible_counts = Counter(
        len(event.cameras) for event in inputs.events if event.eligibility == "eligible"
    )
    means = {count: float(np.mean(values)) for count, values in by_camera_count.items()}
    projected = float(timing["model_startup_s"]) + sum(
        means[count] * population for count, population in eligible_counts.items()
    )
    return {
        "hardware": "Intel Core Ultra 7 268V; detector CPU; pose NPU",
        "sample_events": len(rows),
        "sample_view_frames": sum(int(row["view_frames"]) for row in rows),
        "model_startup_s": round(float(timing["model_startup_s"]), 3),
        "event_wall_s": _distribution(all_seconds, 3),
        "event_wall_s_by_camera_count": {
            str(count): _distribution(values, 3)
            for count, values in sorted(by_camera_count.items())
        },
        "eligible_population_by_camera_count": dict(sorted(eligible_counts.items())),
        "projection_formula": "startup + mean_2cam*eligible_2cam + mean_3cam*eligible_3cam",
        "projected_eligible_population_s": round(projected, 3),
        "projected_eligible_population_min": round(projected / 60.0, 3),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))


def _safe_cache(root: Path, cache: Path) -> Path:
    scratch = (root / ".scratch").resolve()
    resolved = cache.resolve()
    if not resolved.is_relative_to(scratch):
        raise RuntimeError("raw keypoint caches must stay under the worktree .scratch directory")
    return resolved


def _parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=root / "videos" / "3-cam")
    parser.add_argument("--inventory", type=Path, default=root / "inventory")
    parser.add_argument("--sessions", type=Path, default=root / "sessions")
    parser.add_argument("--qualification", type=Path, default=root / "qualification")
    parser.add_argument("--measurements", type=Path, default=root / "measurements")
    parser.add_argument(
        "--cache", type=Path, default=root / ".scratch" / "calibration-observability"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="print the deterministic census and sample plan")
    collect_parser = subparsers.add_parser("collect", help="collect or resume keypoint caches")
    collect_parser.add_argument("--scope", choices=("pilot", "sample"), default="sample")
    collect_parser.add_argument("--force", action="store_true")
    analyze_parser = subparsers.add_parser(
        "analyze", help="print deterministic aggregate observability results"
    )
    analyze_parser.add_argument("--scope", choices=("pilot", "sample"), default="sample")
    subparsers.add_parser("timing", help="print the most recent scaled-sample timing model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    args = _parser(root).parse_args(argv)
    cache = _safe_cache(root, args.cache)
    inputs = load_inputs(
        args.corpus,
        args.inventory,
        args.sessions,
        args.qualification,
        args.measurements,
    )
    selection = select_events(inputs)
    if args.command == "plan":
        _print_json(plan_summary(inputs, selection))
    elif args.command == "collect":
        events = selection.pilot if args.scope == "pilot" else selection.sample
        _print_json(collect(inputs, events, cache, force=args.force, scope=args.scope))
    elif args.command == "analyze":
        _print_json(analyze(inputs, selection, cache, scope=args.scope))
    else:
        _print_json(timing_summary(inputs, selection, cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
