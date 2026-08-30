#!/usr/bin/env python3
"""Generate redacted real-corpus evidence for the trajectory-grid contract."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pose_estimation.video_io import SourceTimestampClock, open_capture, safe_fps

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "scripts/probe_timebase_grid.py"
GENERATOR_VERSION = "v1"
WINDOW_SEC = 1.0
WINDOW_STEP_SEC = WINDOW_SEC / 2.0
GAP_INTERVAL_FACTOR = 1.5
P06_REL_ERR_MAX = 1e-4
GRID_SLOT_TOLERANCE = 0.25
MIN_CADENCE_SPAN_SEC = 1.0
RATE_ENDPOINT_TARGETS_HZ = (29.963, 29.987)


class ProbeError(RuntimeError):
    """A redacted probe failure."""


@dataclass(frozen=True)
class Asset:
    asset_id: str
    source_relative: Path
    device_config: str
    codec: str
    rotation: int
    inventory_fps: float
    nominal_duration_sec: float


@dataclass(frozen=True)
class GridResult:
    residual: float
    ok: bool


@dataclass(frozen=True)
class AssetResult:
    asset: Asset
    header_fps: float
    n_frames: int
    duration_sec: float
    terminal_frame_duration_sec: float
    nominal_fs_hz: float
    nominal_fs_rel_err: float
    median_diff_fs_hz: float
    median_diff_rel_err: float
    nominal_windows: tuple[GridResult, ...]
    median_diff_windows: tuple[GridResult, ...]


def _read_rows(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ProbeError(f"cannot read the {label} table") from exc


def _load_assets(inventory_path: Path, qualification_path: Path) -> list[Asset]:
    inventory_rows = _read_rows(inventory_path, "inventory")
    qualification_rows = _read_rows(qualification_path, "qualification")
    canonical = {
        row["asset_id"]: row for row in inventory_rows if row.get("disposition") == "canonical"
    }
    if len(canonical) != len(qualification_rows):
        raise ProbeError("inventory and qualification canonical populations differ")

    assets: list[Asset] = []
    seen: set[str] = set()
    for row in qualification_rows:
        asset_id = row.get("asset_id", "")
        source = canonical.get(asset_id)
        if not asset_id or source is None or asset_id in seen:
            raise ProbeError("qualification asset keys do not bijectively match the inventory")
        seen.add(asset_id)
        try:
            inventory_fps = safe_fps(float(source["reported_avg_fps"]))
            nominal_duration_sec = float(source["nominal_duration_s"])
            rotation_value = float(source["reported_rotation_deg"])
            rotation = int(rotation_value)
            source_relative = Path(source["source_path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProbeError("an input row has an invalid timebase field") from exc
        device_config = row.get("device_config", "").strip()
        codec = row.get("codec", "").strip()
        if (
            not device_config
            or not codec
            or not math.isfinite(nominal_duration_sec)
            or nominal_duration_sec <= 0
            or not math.isfinite(rotation_value)
            or rotation != rotation_value
            or source_relative.is_absolute()
        ):
            raise ProbeError("an input row has an invalid sample stratum")
        assets.append(
            Asset(
                asset_id=asset_id,
                source_relative=source_relative,
                device_config=device_config,
                codec=codec,
                rotation=rotation,
                inventory_fps=inventory_fps,
                nominal_duration_sec=nominal_duration_sec,
            )
        )
    return sorted(assets, key=lambda asset: asset.asset_id)


def _rank_key(seed: int, asset_id: str, purpose: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{purpose}\0{asset_id}".encode()).digest()


def _ranked(assets: list[Asset], seed: int, purpose: str) -> list[Asset]:
    return sorted(assets, key=lambda asset: _rank_key(seed, asset.asset_id, purpose))


def _base_quotas(counts: Counter[str], sample_size: int) -> dict[str, int]:
    strata = sorted(counts)
    if sample_size < len(strata):
        raise ProbeError("sample size must cover every device configuration")
    quotas = dict.fromkeys(strata, 1)
    remaining = sample_size - len(strata)
    total = sum(counts.values())
    raw = {stratum: remaining * counts[stratum] / total for stratum in strata}
    for stratum in strata:
        quotas[stratum] += math.floor(raw[stratum])
    leftover = sample_size - sum(quotas.values())
    order = sorted(strata, key=lambda stratum: (-(raw[stratum] % 1), stratum))
    for stratum in order[:leftover]:
        quotas[stratum] += 1
    return quotas


def _nearest_rate_asset(assets: list[Asset], target: float, seed: int) -> Asset:
    return min(
        assets,
        key=lambda asset: (
            abs(asset.inventory_fps - target),
            _rank_key(seed, asset.asset_id, f"rate:{target}"),
        ),
    )


def _mandatory_assets(assets: list[Asset], seed: int) -> dict[str, Asset]:
    mandatory: dict[str, Asset] = {}
    categories: tuple[tuple[str, Any], ...] = (
        ("device_config", lambda asset: asset.device_config),
        ("codec", lambda asset: asset.codec),
        ("rotation", lambda asset: asset.rotation),
    )
    for label, key in categories:
        values = sorted({key(asset) for asset in assets})
        for value in values:
            pool = [asset for asset in assets if key(asset) == value]
            chosen = _ranked(pool, seed, f"mandatory:{label}:{value}")[0]
            mandatory[chosen.asset_id] = chosen
    for target in RATE_ENDPOINT_TARGETS_HZ:
        chosen = _nearest_rate_asset(assets, target, seed)
        mandatory[chosen.asset_id] = chosen
    highest = min(
        assets,
        key=lambda asset: (
            -asset.inventory_fps,
            _rank_key(seed, asset.asset_id, "highest_fps"),
        ),
    )
    mandatory[highest.asset_id] = highest
    return mandatory


def _validate_coverage(population: list[Asset], selected: list[Asset], seed: int) -> None:
    for key in (
        lambda asset: asset.device_config,
        lambda asset: asset.codec,
        lambda asset: asset.rotation,
    ):
        if {key(asset) for asset in selected} != {key(asset) for asset in population}:
            raise ProbeError("the selected sample does not cover every required stratum")
    selected_ids = {asset.asset_id for asset in selected}
    required_ids = {
        _nearest_rate_asset(population, target, seed).asset_id
        for target in RATE_ENDPOINT_TARGETS_HZ
    }
    required_ids.add(max(population, key=lambda asset: asset.inventory_fps).asset_id)
    if not required_ids <= selected_ids:
        raise ProbeError("the selected sample omits a required cadence endpoint")


def _sample_assets(
    assets: list[Asset], sample_size: int, seed: int
) -> tuple[list[Asset], dict[str, Any]]:
    if sample_size <= 0 or sample_size > len(assets):
        raise ProbeError("sample size must be within the canonical population")
    mandatory = _mandatory_assets(assets, seed)
    if sample_size < len(mandatory):
        raise ProbeError("sample size cannot hold every mandatory special case")

    if sample_size == len(assets):
        selected = _ranked(assets, seed, "output")
        selection_rule = (
            "full canonical population; deterministic SHA-256(seed,purpose,asset_id) output order"
        )
    else:
        counts = Counter(asset.device_config for asset in assets)
        quotas = _base_quotas(counts, sample_size)
        mandatory_counts = Counter(asset.device_config for asset in mandatory.values())
        for stratum in sorted(counts):
            quotas[stratum] = max(quotas[stratum], mandatory_counts[stratum])
        while sum(quotas.values()) > sample_size:
            reducible = [
                stratum
                for stratum in sorted(counts)
                if quotas[stratum] > max(1, mandatory_counts[stratum])
            ]
            if not reducible:
                raise ProbeError("mandatory assets overfill the stratified sample")
            stratum = max(reducible, key=lambda key: (quotas[key] / counts[key], key))
            quotas[stratum] -= 1
        while sum(quotas.values()) < sample_size:
            expandable = [key for key in sorted(counts) if quotas[key] < counts[key]]
            stratum = min(expandable, key=lambda key: (quotas[key] / counts[key], key))
            quotas[stratum] += 1

        selected_by_id = dict(mandatory)
        for stratum in sorted(counts):
            pool = [asset for asset in assets if asset.device_config == stratum]
            have = sum(asset.device_config == stratum for asset in selected_by_id.values())
            for asset in _ranked(pool, seed, f"sample:{stratum}"):
                if have >= quotas[stratum]:
                    break
                if asset.asset_id not in selected_by_id:
                    selected_by_id[asset.asset_id] = asset
                    have += 1
        selected = _ranked(list(selected_by_id.values()), seed, "output")
        if len(selected) != sample_size:
            raise ProbeError("stratified selection did not reach the requested size")
        selection_rule = (
            "proportional device-config quotas; SHA-256(seed,purpose,asset_id) rank; "
            "mandatory coverage of every device_config/codec/rotation, nearest "
            "29.963/29.987 Hz assets, and the maximum-fps asset"
        )

    _validate_coverage(assets, selected, seed)
    return selected, {
        "n_assets": len(selected),
        "selection_rule": selection_rule,
        "seed": seed,
        "strata": {
            "codec": dict(sorted(Counter(asset.codec for asset in selected).items())),
            "device_config": dict(
                sorted(Counter(asset.device_config for asset in selected).items())
            ),
            "rotation": {
                str(value): count
                for value, count in sorted(Counter(asset.rotation for asset in selected).items())
            },
        },
    }


def _positive_diffs(timestamps: list[float]) -> list[float]:
    return [
        delta
        for left, right in itertools.pairwise(timestamps)
        if math.isfinite(delta := right - left) and delta > 0
    ]


def _nominal_fs(timestamps: list[float]) -> float:
    deltas = _positive_diffs(timestamps)
    if not deltas:
        raise ProbeError("rounded timestamps have no positive interval")
    median = statistics.median(deltas)
    kept = [delta for delta in deltas if delta <= GAP_INTERVAL_FACTOR * median]
    dt = statistics.mean(kept)
    if not math.isfinite(dt) or dt <= 0:
        raise ProbeError("nominal cadence is not finite and positive")
    return 1.0 / dt


def _median_diff_fs(timestamps: list[float]) -> float:
    deltas = [right - left for left, right in itertools.pairwise(timestamps)]
    dt = statistics.median(deltas)
    if not math.isfinite(dt) or dt <= 0:
        raise ProbeError("median-difference cadence is not finite and positive")
    return 1.0 / dt


def _grid_result(timestamps: list[float], fs: float) -> GridResult:
    raw = [(timestamp - timestamps[0]) * fs for timestamp in timestamps]
    slots = [round(value) for value in raw]
    residual = max(abs(value - slot) for value, slot in zip(raw, slots, strict=True))
    duplicate = len(slots) != len(set(slots))
    return GridResult(residual=residual, ok=residual <= GRID_SLOT_TOLERANCE and not duplicate)


def _window_series(timestamps: list[float]) -> list[list[float]]:
    duration = timestamps[-1] - timestamps[0]
    if duration < WINDOW_SEC:
        return []
    count = math.floor((duration - WINDOW_SEC) / WINDOW_STEP_SEC + 1e-12) + 1
    windows: list[list[float]] = []
    for index in range(count):
        start = timestamps[0] + index * WINDOW_STEP_SEC
        stop = start + WINDOW_SEC
        left = bisect.bisect_left(timestamps, start)
        right = bisect.bisect_left(timestamps, stop)
        if right - left >= 4:
            windows.append(timestamps[left:right])
    return windows


def _decode_asset(asset: Asset, corpus: Path) -> AssetResult:
    capture = open_capture(
        str(corpus / asset.source_relative), display=f"asset in {asset.device_config}"
    )
    if capture is None:
        raise ProbeError(f"OpenCV could not open an asset in {asset.device_config}")

    header_fps = asset.inventory_fps
    clock = SourceTimestampClock(capture, header_fps, live=False)
    timestamps: list[float] = []
    source_frame_idx = 0
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            timestamp = clock.timestamp(source_frame_idx)
            source_frame_idx += 1
            if frame is None or frame.size == 0:
                continue
            timestamps.append(round(timestamp, 4))
    finally:
        capture.release()

    if len(timestamps) < 2:
        raise ProbeError(f"an asset in {asset.device_config} decoded fewer than two usable frames")
    if any(not math.isfinite(value) for value in timestamps):
        raise ProbeError(f"an asset in {asset.device_config} produced a non-finite timestamp")

    duration_sec = timestamps[-1] - timestamps[0]
    terminal_frame_duration_sec = asset.nominal_duration_sec - duration_sec
    if not math.isfinite(terminal_frame_duration_sec) or terminal_frame_duration_sec <= 0:
        raise ProbeError(f"an asset in {asset.device_config} has no terminal frame duration")
    nominal_fs = _nominal_fs(timestamps)
    median_diff_fs = _median_diff_fs(timestamps)
    windows = _window_series(timestamps)
    return AssetResult(
        asset=asset,
        header_fps=header_fps,
        n_frames=len(timestamps),
        duration_sec=duration_sec,
        terminal_frame_duration_sec=terminal_frame_duration_sec,
        nominal_fs_hz=nominal_fs,
        nominal_fs_rel_err=abs(nominal_fs - header_fps) / header_fps,
        median_diff_fs_hz=median_diff_fs,
        median_diff_rel_err=abs(median_diff_fs - header_fps) / header_fps,
        nominal_windows=tuple(_grid_result(window, nominal_fs) for window in windows),
        median_diff_windows=tuple(_grid_result(window, median_diff_fs) for window in windows),
    )


def _clean(value: float) -> float:
    return float(f"{value:.12g}")


def _median(values: list[float]) -> float:
    if not values:
        raise ProbeError("an aggregate has no qualifying assets")
    return _clean(statistics.median(values))


def _asset_payload(index: int, result: AssetResult) -> dict[str, Any]:
    nominal = result.nominal_windows
    legacy = result.median_diff_windows
    return {
        "asset_key": f"a{index:02d}",
        "device_config": result.asset.device_config,
        "codec": result.asset.codec,
        "rotation": result.asset.rotation,
        "header_fps": _clean(result.header_fps),
        "n_frames": result.n_frames,
        "duration_sec": _clean(result.duration_sec),
        "terminal_frame_duration_sec": _clean(result.terminal_frame_duration_sec),
        "nominal_fs_hz": _clean(result.nominal_fs_hz),
        "nominal_fs_rel_err": _clean(result.nominal_fs_rel_err),
        "median_diff_fs_hz": _clean(result.median_diff_fs_hz),
        "median_diff_rel_err": _clean(result.median_diff_rel_err),
        "grid_residual_max_nominal": _clean(max((item.residual for item in nominal), default=0.0)),
        "grid_residual_max_median_diff": _clean(
            max((item.residual for item in legacy), default=0.0)
        ),
        "windows_total": len(nominal),
        "windows_on_grid_nominal": sum(item.ok for item in nominal),
        "windows_on_grid_median_diff": sum(item.ok for item in legacy),
    }


def _grid_residual_max(results: list[AssetResult], field: str) -> float:
    return _clean(
        max(
            (item.residual for result in results for item in getattr(result, field)),
            default=0.0,
        )
    )


def _aggregate_metrics(results: list[AssetResult]) -> dict[str, Any]:
    qualifying = [result for result in results if result.duration_sec >= MIN_CADENCE_SPAN_SEC]
    if not qualifying:
        raise ProbeError("an aggregate has no clip that reaches the P06 span floor")
    header_outliers = [
        result for result in qualifying if result.nominal_fs_rel_err > P06_REL_ERR_MAX
    ]
    return {
        "n_assets": len(results),
        "n_assets_span_ge_min": len(qualifying),
        "nominal_fs_rel_err_max": _clean(max(result.nominal_fs_rel_err for result in qualifying)),
        "median_diff_rel_err_max": _clean(max(result.median_diff_rel_err for result in qualifying)),
        "assets_within_p06_nominal": sum(
            result.nominal_fs_rel_err <= P06_REL_ERR_MAX for result in qualifying
        ),
        "assets_within_p06_median_diff": sum(
            result.median_diff_rel_err <= P06_REL_ERR_MAX for result in qualifying
        ),
        "grid_residual_max_nominal": _grid_residual_max(results, "nominal_windows"),
        "grid_residual_max_median_diff": _grid_residual_max(results, "median_diff_windows"),
        "windows_total": sum(len(result.nominal_windows) for result in results),
        "windows_on_grid_nominal": sum(
            item.ok for result in results for item in result.nominal_windows
        ),
        "windows_on_grid_median_diff": sum(
            item.ok for result in results for item in result.median_diff_windows
        ),
        "assets_header_outlier": len(header_outliers),
        "header_outlier_worst_rel_err": _clean(
            max((result.nominal_fs_rel_err for result in header_outliers), default=0.0)
        ),
        "assets_nominal_no_worse_than_legacy": sum(
            result.nominal_fs_rel_err <= result.median_diff_rel_err * (1 + 1e-12)
            for result in results
        ),
    }


def _stratum_aggregate(results: list[AssetResult]) -> dict[str, Any]:
    qualifying = [result for result in results if result.duration_sec >= MIN_CADENCE_SPAN_SEC]
    return {
        "n_assets": len(results),
        "nominal_fs_rel_err_median": _median([result.nominal_fs_rel_err for result in qualifying]),
        "median_diff_rel_err_median": _median(
            [result.median_diff_rel_err for result in qualifying]
        ),
        "windows_total": sum(len(result.nominal_windows) for result in results),
        "windows_on_grid_nominal": sum(
            item.ok for result in results for item in result.nominal_windows
        ),
        "windows_on_grid_median_diff": sum(
            item.ok for result in results for item in result.median_diff_windows
        ),
    }


def _aggregate(results: list[AssetResult]) -> dict[str, Any]:
    by_stratum: defaultdict[str, list[AssetResult]] = defaultdict(list)
    for result in results:
        by_stratum[result.asset.device_config].append(result)
    return {
        **_aggregate_metrics(results),
        "strata": {
            stratum: _stratum_aggregate(by_stratum[stratum]) for stratum in sorted(by_stratum)
        },
    }


def _grouped_metrics(results: list[AssetResult], field: str) -> dict[str, dict[str, Any]]:
    groups: defaultdict[str, list[AssetResult]] = defaultdict(list)
    for result in results:
        groups[str(getattr(result.asset, field))].append(result)
    return {key: _aggregate_metrics(groups[key]) for key in sorted(groups)}


def _source_sha256() -> dict[str, str]:
    return {GENERATOR: hashlib.sha256((ROOT / GENERATOR).read_bytes()).hexdigest()}


def _stale_source_mismatches(output: Path, current: dict[str, str]) -> list[str]:
    try:
        previous = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return sorted(current)
    recorded = previous.get("source_sha256") if isinstance(previous, dict) else None
    if not isinstance(recorded, dict):
        return sorted(current)
    names = set(current) | set(recorded)
    return sorted(name for name in names if recorded.get(name) != current.get(name))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "videos/3-cam")
    parser.add_argument("--inventory", type=Path, default=ROOT / "inventory/assets.csv")
    parser.add_argument("--qualification", type=Path, default=ROOT / "qualification/assets_qc.csv")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2404)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/timebase_grid_results.json")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source_sha256 = _source_sha256()
    if output.exists():
        mismatches = _stale_source_mismatches(output, source_sha256)
        if mismatches:
            print(
                "REFUSED: result file records different generator bytes; "
                f"remove it explicitly before regeneration. mismatched={','.join(mismatches)}",
                file=sys.stderr,
            )
            return 2

    population = _load_assets(args.inventory, args.qualification)
    selected, sample = _sample_assets(population, args.sample_size, args.seed)
    if args.aggregate_only and len(selected) != len(population):
        raise ProbeError("aggregate-only output requires the full canonical population")
    results: list[AssetResult] = []
    for index, asset in enumerate(selected, start=1):
        results.append(_decode_asset(asset, args.corpus))
        if args.progress and (index % 10 == 0 or index == len(selected)):
            print(f"processed={index}/{len(selected)}", file=sys.stderr, flush=True)

    payload = {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "source_sha256": source_sha256,
        "sample": sample,
        "bounds": {
            "p06_rel_err_max": P06_REL_ERR_MAX,
            "grid_slot_tolerance": GRID_SLOT_TOLERANCE,
            "min_cadence_span_sec": MIN_CADENCE_SPAN_SEC,
        },
        "aggregate": _aggregate(results),
    }
    if args.aggregate_only:
        payload["codecs"] = _grouped_metrics(results, "codec")
        payload["device_configs"] = _grouped_metrics(results, "device_config")
    else:
        payload["assets"] = [_asset_payload(index, result) for index, result in enumerate(results)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        print(f"probe_timebase_grid: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
