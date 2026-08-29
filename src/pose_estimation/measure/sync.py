"""The sync axis: both offset estimators over every within-family asset pair.

Publishes the two estimates unfused, each with its own status.  That is the
load-bearing choice.  Which pairs *qualify* is a policy — audio accepts 210 of
246 pairs and the visual corroborator 74, and requiring agreement collapses
graph connectivity by a factor of four — so baking a fused verdict into the
measurement would force a corpus-wide re-decode every time the policy is
re-ruled.  ``qualify`` applies the policy; this module only measures.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .. import inventory, sessions
from . import MeasureError, audio_offset, decimal, visual_offset, write_axis

AXIS = "sync"


@dataclasses.dataclass(frozen=True)
class Asset:
    asset_id: str
    capture_id: str
    source_relative: str
    content_sha256: str
    rotation_deg: int


@dataclasses.dataclass(frozen=True)
class PairKey:
    capture_id: str
    asset_a: str
    asset_b: str


def load_assets(inventory_dir: pathlib.Path) -> list[Asset]:
    """Return the canonical registry rows this axis measures, in registry order."""
    text = (inventory_dir / inventory.ASSETS_FILENAME).read_text(encoding="utf-8")
    assets = [
        Asset(
            asset_id=row["asset_id"],
            capture_id=row["capture_id"],
            source_relative=sessions.decode_source_path(row["source_path"]),
            content_sha256=row["content_sha256"],
            rotation_deg=int(row["reported_rotation_deg"] or 0),
        )
        for row in csv.DictReader(text.splitlines(), lineterminator="\n")
        if row["disposition"] == inventory.CANONICAL
    ]
    if len({asset.asset_id for asset in assets}) != len(assets):
        raise MeasureError("The registry publishes a duplicate canonical asset_id.")
    return assets


def enumerate_pairs(assets: list[Asset]) -> list[PairKey]:
    """Enumerate unordered within-family pairs in the order ``qualify`` uses.

    The two enumerations must agree exactly: a pair keyed the other way round
    is a key ``qualify`` never looks up, and this axis would read as having
    abstained on it rather than as having disagreed about its identity.
    """
    families: dict[str, list[str]] = {}
    for asset in assets:
        families.setdefault(asset.capture_id, []).append(asset.asset_id)
    pairs: list[PairKey] = []
    for capture_id in sorted(families):
        members = sorted(families[capture_id])
        for index, first in enumerate(members):
            pairs.extend(PairKey(capture_id, first, second) for second in members[index + 1 :])
    return pairs


def _cache_audio(arguments: tuple[str, str, str]) -> str:
    path, cache_dir, asset_id = arguments
    audio_offset.ensure_cached(pathlib.Path(path), pathlib.Path(cache_dir), asset_id)
    return asset_id


def _cache_visual(arguments: tuple[str, str, str, str, int]) -> str:
    path, cache_dir, asset_id, digest, rotation = arguments
    visual_offset.ensure_cached(
        pathlib.Path(path), pathlib.Path(cache_dir), asset_id, digest, rotation
    )
    return asset_id


def _estimate(arguments: tuple[PairKey, str, str]) -> dict[str, str]:
    """Estimate one pair with both estimators, independently.

    Each estimator's failure is its own: a visual trace that cannot be
    resampled must not erase an audio offset that was measured cleanly, because
    the pair's value to ``qualify`` is exactly the two verdicts side by side.
    """
    pair, audio_cache, visual_cache = arguments
    audio_dir = pathlib.Path(audio_cache)
    peak, drift, dur_a, dur_b = audio_offset.estimate(audio_dir, pair.asset_a, pair.asset_b)
    rate_a = audio_offset.source_rate(audio_dir, pair.asset_a)
    rate_b = audio_offset.source_rate(audio_dir, pair.asset_b)

    visual_dir = pathlib.Path(visual_cache)
    try:
        time_a, values_a = visual_offset.load_signal(visual_dir, pair.asset_a)
        time_b, values_b = visual_offset.load_signal(visual_dir, pair.asset_b)
        visual = visual_offset.estimate(time_a, values_a, time_b, values_b)
    except (OSError, ValueError, KeyError):
        visual = visual_offset.Offset(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), "signal_absent"
        )

    return {
        "capture_id": pair.capture_id,
        "asset_a": pair.asset_a,
        "asset_b": pair.asset_b,
        "offset_audio_s": decimal(peak.lag_s),
        "peak_rms_audio": decimal(peak.peak_rms),
        "peak_ratio_audio": decimal(peak.peak_ratio),
        "status_audio": peak.status,
        "drift_ppm": decimal(drift.ppm),
        "drift_se": decimal(drift.standard_error),
        "offset_visual_s": decimal(visual.offset_s),
        "conf_visual": decimal(visual.confidence),
        "peak_corr_visual": decimal(visual.peak_correlation),
        "status_visual": visual.status,
        "overlap_s": decimal(peak.overlap_s),
        "dur_a": decimal(dur_a),
        "dur_b": decimal(dur_b),
        # The exact rate each side decoded at, because P29 stratifies sync QC
        # by (model, OS, sample_rate) and a boolean names no stratum.  These
        # are the estimator's own rates, so ``qualify`` can check the rate it
        # read from the container header against the rate that actually
        # produced the offset instead of assuming the two agree.
        "audio_rate_a": "" if rate_a is None else str(rate_a),
        "audio_rate_b": "" if rate_b is None else str(rate_b),
        # Kept beside them so P28's priming cancellation stays falsifiable
        # against a named stratum, and now a pure function of the two cells.
        "same_audio_rate": "" if None in (rate_a, rate_b) else ("1" if rate_a == rate_b else "0"),
    }


def measure(
    inventory_dir: str | os.PathLike[str],
    corpus_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    *,
    workers: int = 4,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Measure the sync axis over the whole corpus and record it in the sidecar.

    Returns the manifest and the run's wall-clock timings separately.  Timings
    stay out of provenance on purpose: provenance holds what moves the numbers,
    and a clock reading there would make two runs producing identical tables
    carry different manifest digests, which ``qualify`` ingests.
    """
    inventory_path = pathlib.Path(inventory_dir)
    inventory.validate_generation(inventory_path)
    assets = load_assets(inventory_path)
    cache = pathlib.Path(cache_dir)
    audio_cache = cache / "audio"
    visual_cache = cache / "visual"

    started = time.perf_counter()
    paths = {
        asset.asset_id: str(sessions.resolve_source(corpus_root, asset.source_relative))
        for asset in assets
    }
    audio_work = [(paths[a.asset_id], str(audio_cache), a.asset_id) for a in assets]
    visual_work = [
        (paths[a.asset_id], str(visual_cache), a.asset_id, a.content_sha256, a.rotation_deg)
        for a in assets
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_cache_audio, audio_work, chunksize=1))
        list(executor.map(_cache_visual, visual_work, chunksize=1))
    cached_s = time.perf_counter() - started

    pairs = enumerate_pairs(assets)
    estimate_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # map, never as_completed: the row order is part of the published bytes.
        rows = list(
            executor.map(
                _estimate,
                [(pair, str(audio_cache), str(visual_cache)) for pair in pairs],
                chunksize=1,
            )
        )
    timings = {
        "cache_seconds": round(cached_s, 3),
        "estimate_seconds": round(time.perf_counter() - estimate_started, 3),
    }
    provenance = {
        "audio": audio_offset.PROVENANCE,
        "visual": visual_offset.PROVENANCE,
        "assets": len(assets),
        "pairs": len(pairs),
    }
    manifest = write_axis(out_dir, AXIS, rows, provenance, inventory_dir=inventory_path)
    return manifest, {**timings, **{f"count_{k}": v for k, v in summarize(rows).items()}}


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    """Return the redaction-safe acceptance tallies for the console."""
    audio = sum(row["status_audio"] == "ok" for row in rows)
    visual = sum(row["status_visual"] == "ok" for row in rows)
    both = sum(row["status_audio"] == "ok" and row["status_visual"] == "ok" for row in rows)
    return {"pairs": len(rows), "audio_ok": audio, "visual_ok": visual, "both_ok": both}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pose-estimation-measure",
        description="Measure one qualification axis into the measurement sidecar.",
    )
    parser.add_argument("--axis", default=AXIS, choices=[AXIS], help="Axis to measure.")
    parser.add_argument("--inventory", required=True, help="Directory that holds assets.csv.")
    parser.add_argument("--corpus", required=True, help="Root directory of the recordings.")
    parser.add_argument("--out", required=True, help="Sidecar directory to record into.")
    parser.add_argument("--cache", required=True, help="Directory for decoded signal caches.")
    parser.add_argument("--workers", type=int, default=4, help="Decode and estimate workers.")
    arguments = parser.parse_args(argv)
    try:
        manifest, timings = measure(
            arguments.inventory,
            arguments.corpus,
            arguments.out,
            arguments.cache,
            workers=arguments.workers,
        )
    except (MeasureError, OSError, ValueError, inventory.InventoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Axis {arguments.axis}: {manifest['axes'][arguments.axis]['rows']} rows")
    for key, value in timings.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
