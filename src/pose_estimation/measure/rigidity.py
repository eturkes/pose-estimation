"""Measure image-space background rigidity for every canonical asset."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from pose_estimation import inventory, sessions
from pose_estimation.measure import MeasureError, decimal, mebx, write_axis

AXIS = "rigidity"

SEED = 20260827
SAMPLE_INTERVAL_S = 0.5
ANALYSIS_MAX_DIM = 640
SIFT_FEATURES = 2500
SIFT_CONTRAST_THRESHOLD = 0.01
MATCH_RATIO = 0.78
MIN_TRACKS = 100
SUPPORT_GRID_SIZE = 4
MIN_GRID_CELLS = 8
MIN_VALID_FRACTION = 0.80
RANSAC_THRESHOLD_PX = 8.0
RANSAC_MAX_ITERATIONS = 10_000
RANSAC_CONFIDENCE = 0.999
DRIFT_GRID_SIZE = 8
DRIFT_GRID_MARGIN_FRACTION = 0.05
DRIFT_P95_GATE_PX = 20.0

PROVENANCE: dict[str, Any] = {
    "device": "CPU",
    "software": {
        "decoder": f"PyAV {av.__version__}",
        "feature_and_model": f"OpenCV {cv2.__version__}",
        "array": f"NumPy {np.__version__}",
    },
    "sampling": {
        "interval_s": SAMPLE_INTERVAL_S,
        "time_source": "PTS * time_base",
        "first_sample_s": 0.0,
        "reference": "first sampled frame",
        "analysis_max_dimension_px": ANALYSIS_MAX_DIM,
        "display_rotation": "inventory reported_rotation_deg",
        "orientation_eligibility": "constant timed video-orientation track",
    },
    "features": {
        "algorithm": "SIFT",
        "nfeatures": SIFT_FEATURES,
        "contrast_threshold": SIFT_CONTRAST_THRESHOLD,
        "mask": "full frame",
        "matching": "bidirectional 2-nearest-neighbour mutual ratio",
        "ratio": MATCH_RATIO,
        "minimum_tracks": MIN_TRACKS,
    },
    "model": {
        "algorithm": "USAC_MAGSAC homography",
        "threshold_native_px": RANSAC_THRESHOLD_PX,
        "threshold_analysis_conversion": "divide by mean native-to-analysis scale",
        "maximum_iterations": RANSAC_MAX_ITERATIONS,
        "confidence": RANSAC_CONFIDENCE,
        "rng_seed": SEED,
        "rng_per_sample": "seed + sample index",
        "opencv_threads_per_worker": 1,
    },
    "support": {
        "grid": f"{SUPPORT_GRID_SIZE}x{SUPPORT_GRID_SIZE}",
        "minimum_median_cells": MIN_GRID_CELLS,
        "minimum_valid_fraction": MIN_VALID_FRACTION,
    },
    "statistic": {
        "grid": f"{DRIFT_GRID_SIZE}x{DRIFT_GRID_SIZE}",
        "grid_margin_fraction": DRIFT_GRID_MARGIN_FRACTION,
        "units": "native pixels",
        "aggregates": ["median", "p95"],
    },
    "gate": {
        "rigid_when": "rigidity_drift_p95_px <= threshold",
        "threshold_px": DRIFT_P95_GATE_PX,
        "anchor": "3D pipeline reprojection tolerance",
    },
}


@dataclass(frozen=True)
class Asset:
    asset_id: str
    capture_id: str
    source_relative: str
    rotation_deg: int
    reported_frames: int


@dataclass(frozen=True)
class RigidityResult:
    asset_id: str
    capture_id: str
    orientation_status: str
    sampled_frames: int
    valid_samples: int
    valid_fraction: float
    inliers_median: float
    grid_cells_median: float
    drift_median_px: float
    drift_p95_px: float
    rigidity_flag: str
    error_type: str = ""


def load_assets(inventory_dir: str | os.PathLike[str]) -> list[Asset]:
    path = Path(inventory_dir) / inventory.ASSETS_FILENAME
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["disposition"] == inventory.CANONICAL]
    return [
        Asset(
            asset_id=row["asset_id"],
            capture_id=row["capture_id"],
            source_relative=row["source_path"],
            rotation_deg=int(row["reported_rotation_deg"]),
            reported_frames=int(row["reported_frame_count"]),
        )
        for row in rows
    ]


def orientation_sequence(path: Path) -> list[int]:
    maps = mebx.key_maps(path)
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
            for key_id, value in mebx.sample_entries(bytes(packet)):
                if key_map.get(key_id, "").endswith("video-orientation"):
                    values.append(int.from_bytes(value, "big"))
    return values


def _orientation_status(values: list[int]) -> str:
    if not values:
        return "missing"
    if any(left != right for left, right in itertools.pairwise(values)):
        return "changed"
    return "constant"


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


def _tracking_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, 255, dtype=np.uint8)


def _grid_cell_count(points: np.ndarray, width: int, height: int) -> int:
    x = np.clip((points[:, 0] * SUPPORT_GRID_SIZE / width).astype(int), 0, SUPPORT_GRID_SIZE - 1)
    y = np.clip((points[:, 1] * SUPPORT_GRID_SIZE / height).astype(int), 0, SUPPORT_GRID_SIZE - 1)
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
) -> tuple[np.ndarray, int, int] | None:
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
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
    }
    reverse = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse_knn
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
    }
    mutual = [
        (source, target) for source, target in forward.items() if reverse.get(target) == source
    ]
    if len(mutual) < MIN_TRACKS:
        return None
    source = np.asarray([baseline_keypoints[index].pt for index, _ in mutual], dtype=np.float64)
    target = np.asarray([current_keypoints[index].pt for _, index in mutual], dtype=np.float64)
    cv2.setRNGSeed(SEED + sample_index)
    threshold = RANSAC_THRESHOLD_PX / max((scale_x + scale_y) / 2, 1.0)
    homography, mask = cv2.findHomography(
        source,
        target,
        cv2.USAC_MAGSAC,
        threshold,
        maxIters=RANSAC_MAX_ITERATIONS,
        confidence=RANSAC_CONFIDENCE,
    )
    if homography is None or mask is None or not np.all(np.isfinite(homography)):
        return None
    inliers = mask.ravel().astype(bool)
    source_inliers = source[inliers]
    target_inliers = target[inliers]
    if len(source_inliers) < MIN_TRACKS:
        return None
    cells = min(
        _grid_cell_count(source_inliers, baseline.shape[1], baseline.shape[0]),
        _grid_cell_count(target_inliers, current.shape[1], current.shape[0]),
    )
    margin = DRIFT_GRID_MARGIN_FRACTION
    xs = np.linspace(margin * baseline.shape[1], (1 - margin) * baseline.shape[1], DRIFT_GRID_SIZE)
    ys = np.linspace(margin * baseline.shape[0], (1 - margin) * baseline.shape[0], DRIFT_GRID_SIZE)
    grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    warped = cv2.perspectiveTransform(grid.reshape(1, -1, 2), homography).reshape(-1, 2)
    drift = np.hypot(
        (warped[:, 0] - grid[:, 0]) * scale_x,
        (warped[:, 1] - grid[:, 1]) * scale_y,
    )
    return drift, len(source_inliers), cells


def _rigidity_metrics(
    frames: list[np.ndarray], scale_x: float, scale_y: float
) -> tuple[int, float, float, float, float, float]:
    if len(frames) < 3:
        return 0, 0.0, math.nan, math.nan, math.nan, math.nan
    baseline = frames[0]
    detector = cv2.SIFT.create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
    )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    baseline_keypoints, baseline_descriptors = detector.detectAndCompute(
        baseline, _tracking_mask(baseline.shape)
    )
    if baseline_descriptors is None or len(baseline_keypoints) < MIN_TRACKS:
        return 0, 0.0, math.nan, math.nan, math.nan, math.nan
    all_drift: list[np.ndarray] = []
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
        drift, count, covered = matched
        all_drift.append(drift)
        inliers.append(count)
        cells.append(covered)
    valid_fraction = len(all_drift) / (len(frames) - 1)
    inliers_median = float(np.median(inliers)) if inliers else math.nan
    cells_median = float(np.median(cells)) if cells else math.nan
    if not all_drift or valid_fraction < MIN_VALID_FRACTION:
        return len(all_drift), valid_fraction, inliers_median, cells_median, math.nan, math.nan
    drift_values = np.concatenate(all_drift)
    return (
        len(all_drift),
        valid_fraction,
        inliers_median,
        cells_median,
        float(np.median(drift_values)),
        float(np.quantile(drift_values, 0.95)),
    )


def rigidity_flag(
    *,
    orientation_status: str,
    drift_median_px: float,
    drift_p95_px: float,
    valid_fraction: float,
    grid_cells_median: float,
) -> str:
    if orientation_status in {"changed", "missing"}:
        return "excluded_orientation"
    if orientation_status != "constant":
        return "error"
    support_ok = (
        math.isfinite(drift_median_px)
        and math.isfinite(drift_p95_px)
        and valid_fraction >= MIN_VALID_FRACTION
        and grid_cells_median >= MIN_GRID_CELLS
    )
    if not support_ok:
        return "unmeasurable"
    return "rigid" if drift_p95_px <= DRIFT_P95_GATE_PX else "camera_motion"


def analyze_asset(asset: Asset, path: Path) -> RigidityResult:
    cv2.setNumThreads(1)
    orientation_status = _orientation_status(orientation_sequence(path))
    sampled: list[np.ndarray] = []
    scale_x = math.nan
    scale_y = math.nan
    previous_t = -math.inf
    first_t: float | None = None
    next_sample_t = 0.0
    decoded_frames = 0
    with av.open(str(path)) as container:
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
            decoded_frames += 1
            if relative_t + 1e-9 >= next_sample_t:
                gray, current_scale_x, current_scale_y = _analysis_gray(frame, asset.rotation_deg)
                if sampled and (current_scale_x != scale_x or current_scale_y != scale_y):
                    raise ValueError("display dimensions changed during decode")
                sampled.append(gray)
                scale_x, scale_y = current_scale_x, current_scale_y
                while next_sample_t <= relative_t + 1e-9:
                    next_sample_t += SAMPLE_INTERVAL_S
            previous_t = frame_t
    if decoded_frames != asset.reported_frames:
        raise ValueError("decoded frame count differs from the registry")
    if orientation_status == "constant":
        metrics = _rigidity_metrics(sampled, scale_x, scale_y)
    else:
        metrics = (0, 0.0, math.nan, math.nan, math.nan, math.nan)
    valid_samples, valid_fraction, inliers, cells, drift_median, drift_p95 = metrics
    flag = rigidity_flag(
        orientation_status=orientation_status,
        drift_median_px=drift_median,
        drift_p95_px=drift_p95,
        valid_fraction=valid_fraction,
        grid_cells_median=cells,
    )
    return RigidityResult(
        asset.asset_id,
        asset.capture_id,
        orientation_status,
        len(sampled),
        valid_samples,
        valid_fraction,
        inliers,
        cells,
        drift_median,
        drift_p95,
        flag,
    )


def _worker(argument: tuple[Asset, str]) -> RigidityResult:
    asset, path = argument
    try:
        return analyze_asset(asset, Path(path))
    except Exception as error:
        return RigidityResult(
            asset.asset_id,
            asset.capture_id,
            "error",
            0,
            0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            "error",
            type(error).__name__,
        )


def _row(result: RigidityResult) -> dict[str, str]:
    return {
        "asset_id": result.asset_id,
        "rigidity_drift_median_px": decimal(result.drift_median_px),
        "rigidity_drift_p95_px": decimal(result.drift_p95_px),
        "rigidity_valid_fraction": decimal(result.valid_fraction),
        "rigidity_flag": result.rigidity_flag,
    }


def summarize(results: list[RigidityResult]) -> dict[str, int]:
    flags = Counter(result.rigidity_flag for result in results)
    families: dict[str, list[RigidityResult]] = defaultdict(list)
    for result in results:
        families[result.capture_id].append(result)
    multi_asset = [members for members in families.values() if len(members) > 1]
    return {
        "assets": len(results),
        "rigid": flags["rigid"],
        "camera_motion": flags["camera_motion"],
        "eligible": flags["rigid"] + flags["camera_motion"],
        "unmeasurable": flags["unmeasurable"],
        "excluded_orientation": flags["excluded_orientation"],
        "error": flags["error"],
        "no_verdict": flags["unmeasurable"] + flags["excluded_orientation"] + flags["error"],
        "multi_asset_families": len(multi_asset),
        "all_members_rigid": sum(
            all(member.rigidity_flag == "rigid" for member in members) for members in multi_asset
        ),
    }


def measure(
    inventory_dir: str | os.PathLike[str],
    corpus_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    workers: int = 4,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    inventory_path = Path(inventory_dir)
    inventory.validate_generation(inventory_path)
    assets = sorted(load_assets(inventory_path), key=lambda asset: asset.asset_id)
    arguments = [
        (asset, str(sessions.resolve_source(corpus_root, asset.source_relative)))
        for asset in assets
    ]
    started = time.perf_counter()
    if workers == 1:
        results = [_worker(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_worker, arguments, chunksize=1))
    summary = summarize(results)
    summary["wall_seconds"] = round(time.perf_counter() - started)
    provenance = {**PROVENANCE, "assets": len(assets)}
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
    parser.add_argument("--workers", type=int, default=4, help="Decode and estimate workers.")
    arguments = parser.parse_args(argv)
    try:
        manifest, summary = measure(
            arguments.inventory,
            arguments.corpus,
            arguments.out,
            workers=arguments.workers,
        )
    except (MeasureError, OSError, ValueError, inventory.InventoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Axis {AXIS}: {manifest['axes'][AXIS]['rows']} rows")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
