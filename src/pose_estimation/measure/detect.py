"""Measure pose-pipeline detectability for every canonical asset."""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.metadata
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import av
import numpy as np

from pose_estimation import inventory, sessions
from pose_estimation.measure import MeasureError, decimal, mebx, write_axis
from pose_estimation.rtmlib_openvino import _patch_rtmlib_openvino
from pose_estimation.run import _DET_INPUT_SIZE, _DET_URL, MODEL_REGISTRY, TRACKING_INDICES

AXIS = "detect"

MODEL_NAME = "rtmw-l"
SAMPLE_COUNT = 24
DETECTOR_DEVICE = "GPU"
POSE_DEVICE = "NPU"
BACKEND = "openvino"
DETECTOR_MODE = "human"
DETECTOR_NMS_THRESHOLD = 0.45
DETECTOR_SCORE_THRESHOLD = 0.7
POSE_TO_OPENPOSE = False
TRACKING_SCOPE = "hands-arms"
ACTIVE_KEYPOINT_INDICES = tuple(sorted(TRACKING_INDICES[TRACKING_SCOPE]))
ORIENTATION_ROTATION = {1: 0, 6: 90, 3: 180, 8: 270}


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


_MODEL = MODEL_REGISTRY[MODEL_NAME]
POSE_CLASS = str(_MODEL["pose_class"])
POSE_URL = str(_MODEL["pose"])
POSE_INPUT_SIZE = cast(tuple[int, int], _MODEL["pose_input_size"])
PROVENANCE: dict[str, Any] = {
    "software": {
        "PyAV": av.__version__,
        "NumPy": np.__version__,
        "OpenVINO": _version("openvino"),
        "rtmlib": _version("rtmlib"),
        "adapter": "pose_estimation.rtmlib_openvino._patch_rtmlib_openvino",
    },
    "sampling": {
        "frames_per_asset": SAMPLE_COUNT,
        "rule": "PTS midpoint of each equal-duration bin across the full video stream",
        "target_formula": "start + floor((2*i+1)*duration/(2*frames_per_asset))",
        "seek": "backward keyframe, then first decoded frame with PTS >= target; prior fallback",
        "time_source": "frame PTS * frame time_base",
        "orientation": "timed video-orientation at sampled PTS; registry header fallback",
        "orientation_code_to_clockwise_degrees": {
            str(code): degrees for code, degrees in ORIENTATION_ROTATION.items()
        },
    },
    "detector": {
        "model": "YOLOX-m HumanArt",
        "url": _DET_URL,
        "input_size": list(_DET_INPUT_SIZE),
        "mode": DETECTOR_MODE,
        "nms_threshold": DETECTOR_NMS_THRESHOLD,
        "score_threshold": DETECTOR_SCORE_THRESHOLD,
        "device": DETECTOR_DEVICE,
        "count_source": "direct detector return per sampled frame",
    },
    "pose": {
        "model": MODEL_NAME,
        "class": POSE_CLASS,
        "url": POSE_URL,
        "input_size": list(POSE_INPUT_SIZE),
        "to_openpose": POSE_TO_OPENPOSE,
        "device": POSE_DEVICE,
        "invocation": "direct pose model per frame with current detector boxes",
    },
    "statistics": {
        "detect_rate": "frames with >=1 detector box / pose-inference-success frames",
        "detect_conf_median": "median active-keypoint score of the highest-mean-score person",
        "primary_person": "maximum mean finite score over all pose keypoints",
        "active_keypoint_scope": TRACKING_SCOPE,
        "active_keypoint_indices": list(ACTIVE_KEYPOINT_INDICES),
        "subject_px_height_median": "median clipped detector-box height for the primary person",
        "pose_failure_denominator": "excluded and counted in inference_failure_frames",
    },
    "backend": {
        "name": BACKEND,
        "performance_hint": "LATENCY",
        "NPU_input_shape": "all dynamic dimensions reshaped to 1 before compile",
        "detector_dynamic_output": "GPU zero-padded; NPU forbidden",
    },
}


@dataclass(frozen=True)
class Asset:
    asset_id: str
    source_relative: str
    rotation_deg: int


@dataclass(frozen=True)
class Sample:
    sample_index: int
    frame: np.ndarray


@dataclass(frozen=True)
class DetectResult:
    asset_id: str
    sampled_frames: int
    inferred_frames: int
    inference_failure_frames: int
    detect_rate: float
    detect_conf_median: float
    subject_px_height_median: float
    error_type: str = ""


@dataclass(frozen=True)
class Models:
    detector: Any
    pose: Any
    placements: dict[str, dict[str, object]]
    available_devices: list[str]


def load_assets(inventory_dir: str | os.PathLike[str]) -> list[Asset]:
    path = Path(inventory_dir) / inventory.ASSETS_FILENAME
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["disposition"] == inventory.CANONICAL]
    return [
        Asset(
            asset_id=row["asset_id"],
            source_relative=row["source_path"],
            rotation_deg=int(row["reported_rotation_deg"]),
        )
        for row in rows
    ]


def _timed_orientations(path: Path) -> list[tuple[float, int]]:
    maps = mebx.key_maps(path)
    orientations: list[tuple[float, int]] = []
    with av.open(str(path)) as container:
        data_streams = [
            stream for stream in container.streams if stream.type not in {"video", "audio"}
        ]
        by_index = {
            stream.index: maps[index] if index < len(maps) else {}
            for index, stream in enumerate(data_streams)
        }
        for packet in container.demux(data_streams):
            if packet.size == 0:
                continue
            packet_time = (
                float(packet.pts * packet.time_base)
                if packet.pts is not None and packet.time_base is not None
                else math.nan
            )
            key_map = by_index.get(packet.stream.index, {})
            for key_id, value in mebx.sample_entries(bytes(packet)):
                if key_map.get(key_id, "").endswith("video-orientation"):
                    orientations.append((packet_time, int.from_bytes(value, "big")))
    orientations.sort(key=lambda item: (math.isnan(item[0]), item[0]))
    return orientations


def _rotation_at(
    pts_s: float,
    orientations: list[tuple[float, int]],
    header_rotation: int,
) -> int:
    finite = [(pts, code) for pts, code in orientations if math.isfinite(pts)]
    if finite:
        times = [pts for pts, _ in finite]
        code = finite[max(0, bisect.bisect_right(times, pts_s) - 1)][1]
        return ORIENTATION_ROTATION.get(code, header_rotation)
    codes = [code for _, code in orientations]
    return ORIENTATION_ROTATION.get(codes[0], header_rotation) if codes else header_rotation


def _rotate(frame: np.ndarray, degrees_clockwise: int) -> np.ndarray:
    turns = (degrees_clockwise // 90) % 4
    return np.ascontiguousarray(np.rot90(frame, k=(-turns) % 4))


def _sample_frames(
    path: Path,
    sample_count: int,
    orientations: list[tuple[float, int]],
    header_rotation: int,
) -> list[Sample]:
    samples: list[Sample] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            return []
        stream = container.streams.video[0]
        if stream.duration is None or stream.duration <= 0:
            return []
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
            if chosen is None or chosen.pts is None or chosen.time_base is None:
                continue
            pts_s = float(chosen.pts * chosen.time_base)
            rotation = _rotation_at(pts_s, orientations, header_rotation)
            samples.append(
                Sample(sample_index, _rotate(chosen.to_ndarray(format="bgr24"), rotation))
            )
    return samples


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


def _median(values: list[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    return float(np.median(finite)) if finite.size else math.nan


def _infer_samples(
    samples: list[Sample], detector: Any, pose: Any
) -> tuple[int, int, int, float, float, float]:
    completed = 0
    detected = 0
    failures = 0
    confidences: list[float] = []
    heights: list[float] = []
    for sample in samples:
        boxes = list(detector(sample.frame))
        try:
            keypoints, scores = pose(sample.frame, bboxes=boxes)
        except Exception:
            failures += 1
            continue
        completed += 1
        detected += bool(boxes)
        score_array = None if scores is None else np.asarray(scores, dtype=float)
        if score_array is None:
            continue
        primary = _primary_index(score_array)
        if primary is None or primary >= len(boxes):
            continue
        selected = score_array[primary, ACTIVE_KEYPOINT_INDICES]
        confidences.extend(selected[np.isfinite(selected)].tolist())
        heights.append(_box_height(boxes[primary], sample.frame.shape[0]))
        _ = keypoints
    rate = detected / completed if completed else math.nan
    return completed, detected, failures, rate, _median(confidences), _median(heights)


def analyze_asset(asset: Asset, path: Path, models: Models) -> DetectResult:
    try:
        orientations = _timed_orientations(path)
    except Exception:
        orientations = []
    samples = _sample_frames(path, SAMPLE_COUNT, orientations, asset.rotation_deg)
    completed, _, failures, rate, confidence, height = _infer_samples(
        samples, models.detector, models.pose
    )
    return DetectResult(
        asset.asset_id,
        len(samples),
        completed,
        failures,
        rate,
        confidence,
        height,
    )


def _execution_devices(model: Any) -> dict[str, object]:
    value = model.compiled_model.get_property("EXECUTION_DEVICES")
    normalized = [value] if isinstance(value, str) else list(value)
    return {"raw_type": type(value).__name__, "devices": [str(device) for device in normalized]}


def _devices(placement: dict[str, object]) -> list[str]:
    devices = placement["devices"]
    if not isinstance(devices, list):
        raise RuntimeError("OpenVINO returned a malformed EXECUTION_DEVICES property")
    return [str(device) for device in devices]


def make_models() -> Models:
    from openvino import Core

    available_devices = list(Core().available_devices)
    if available_devices != ["CPU", "GPU", "NPU"]:
        raise RuntimeError(f"expected CPU/GPU/NPU; found {available_devices!r}")
    _patch_rtmlib_openvino()
    import rtmlib

    detector = rtmlib.YOLOX(
        _DET_URL,
        model_input_size=_DET_INPUT_SIZE,
        mode=DETECTOR_MODE,
        nms_thr=DETECTOR_NMS_THRESHOLD,
        score_thr=DETECTOR_SCORE_THRESHOLD,
        backend=BACKEND,
        device=DETECTOR_DEVICE,
    )
    pose_class = getattr(rtmlib, POSE_CLASS)
    pose = pose_class(
        POSE_URL,
        model_input_size=POSE_INPUT_SIZE,
        to_openpose=POSE_TO_OPENPOSE,
        backend=BACKEND,
        device=POSE_DEVICE,
    )
    placements = {"detector": _execution_devices(detector), "pose": _execution_devices(pose)}
    detector_devices = _devices(placements["detector"])
    pose_devices = _devices(placements["pose"])
    if not detector_devices or not all("GPU" in device for device in detector_devices):
        raise RuntimeError(f"detector did not execute exclusively on GPU: {detector_devices!r}")
    if not pose_devices or not all("NPU" in device for device in pose_devices):
        raise RuntimeError(f"pose model did not execute exclusively on NPU: {pose_devices!r}")
    return Models(detector, pose, placements, available_devices)


def _error_result(asset: Asset, error: Exception) -> DetectResult:
    return DetectResult(
        asset.asset_id,
        0,
        0,
        0,
        math.nan,
        math.nan,
        math.nan,
        type(error).__name__,
    )


def _row(result: DetectResult) -> dict[str, str]:
    return {
        "asset_id": result.asset_id,
        "detect_rate": decimal(result.detect_rate),
        "detect_conf_median": decimal(result.detect_conf_median),
        "subject_px_height_median": decimal(result.subject_px_height_median),
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not finite.size:
        return {"n": 0, "median": None, "mean": None, "min": None}
    return {
        "n": int(finite.size),
        "median": round(float(np.median(finite)), 6),
        "mean": round(float(np.mean(finite)), 6),
        "min": round(float(np.min(finite)), 6),
    }


def summarize(results: list[DetectResult]) -> dict[str, Any]:
    return {
        "assets": len(results),
        "sampled_frames": sum(result.sampled_frames for result in results),
        "inferred_frames": sum(result.inferred_frames for result in results),
        "inference_failure_frames": sum(result.inference_failure_frames for result in results),
        "sample_shortfall_assets": sum(result.sampled_frames != SAMPLE_COUNT for result in results),
        "error_assets": sum(bool(result.error_type) for result in results),
        "detect_rate": _distribution([result.detect_rate for result in results]),
    }


def measure(
    inventory_dir: str | os.PathLike[str],
    corpus_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory_path = Path(inventory_dir)
    inventory.validate_generation(inventory_path)
    assets = sorted(load_assets(inventory_path), key=lambda asset: asset.asset_id)
    paths = [sessions.resolve_source(corpus_root, asset.source_relative) for asset in assets]
    models = make_models()
    started = time.perf_counter()
    results: list[DetectResult] = []
    for index, (asset, path) in enumerate(zip(assets, paths, strict=True), start=1):
        try:
            result = analyze_asset(asset, path, models)
        except Exception as error:
            result = _error_result(asset, error)
        results.append(result)
        if progress:
            print(f"asset {index}/{len(assets)}", flush=True)
    summary = summarize(results)
    summary["wall_seconds"] = round(time.perf_counter() - started)
    provenance = {
        **PROVENANCE,
        "execution": {
            "available_devices": models.available_devices,
            "placements": models.placements,
        },
        "assets": len(assets),
    }
    manifest = write_axis(
        out_dir,
        AXIS,
        [_row(result) for result in results],
        provenance,
        inventory_dir=inventory_path,
    )
    return manifest, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="Directory that holds assets.csv.")
    parser.add_argument("--corpus", required=True, help="Root directory of the recordings.")
    parser.add_argument("--out", required=True, help="Sidecar directory to record into.")
    arguments = parser.parse_args(argv)
    try:
        manifest, summary = measure(
            arguments.inventory,
            arguments.corpus,
            arguments.out,
            progress=True,
        )
    except (MeasureError, OSError, RuntimeError, ValueError, inventory.InventoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Axis {AXIS}: {manifest['axes'][AXIS]['rows']} rows")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
