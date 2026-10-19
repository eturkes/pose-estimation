"""Corpus census bundle, assembled from the redaction-safe publisher summaries.

Sources: `inventory/census.json`, `qualification/qualification.json`, the whole
`calibration_qc/` tree, `output/corpus-2d/run_report.json`, `cohort/cohort.json`.
Each is aggregates only by its own publisher's contract, so nothing here can name
a subject, a capture or a file.  The per-asset CSVs beside them are not read.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import Paths


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _headline(
    census: dict[str, Any] | None,
    qualification: dict[str, Any] | None,
    run: dict[str, Any] | None,
    cohort: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The numbers a reviewer checks first, each carrying its own source.

    Read off the publishers rather than restated, so a republished tree moves the
    tile instead of leaving a stale literal in the UI.
    """
    assets = (census or {}).get("assets", {})
    captures = (census or {}).get("captures", {})
    population = (cohort or {}).get("population", {})
    throughput = (run or {}).get("throughput", {})
    tiles: list[dict[str, Any]] = [
        {
            "key": "subjects",
            "value": (census or {}).get("subject_directories"),
            "source": "inventory",
        },
        {"key": "clips_discovered", "value": assets.get("discovered"), "source": "inventory"},
        {"key": "clips_canonical", "value": assets.get("canonical"), "source": "inventory"},
        {"key": "clips_quarantined", "value": assets.get("quarantined"), "source": "inventory"},
        {"key": "captures", "value": captures.get("total"), "source": "inventory"},
        {"key": "events", "value": population.get("events"), "source": "cohort"},
        {
            "key": "media_minutes",
            "value": assets.get("nominal_minutes_total"),
            "source": "inventory",
        },
        {
            "key": "frames_reported",
            "value": assets.get("reported_frames_total"),
            "source": "inventory",
        },
        {"key": "frames_decoded", "value": throughput.get("frames_decoded"), "source": "run"},
        {"key": "run_hours", "value": throughput.get("hours_total"), "source": "run"},
        {"key": "frame_rows", "value": population.get("frame_rows"), "source": "cohort"},
        {"key": "features", "value": population.get("features"), "source": "cohort"},
    ]
    if qualification:
        tiles.append(
            {
                "key": "decode_ok",
                "value": (qualification.get("assets", {}).get("decode_status", {}) or {}).get("ok"),
                "source": "qualification",
            }
        )
    return [tile for tile in tiles if tile["value"] is not None]


def _shapes(census: dict[str, Any] | None) -> list[dict[str, Any]]:
    """`1920x1080@29.985/h264/rot0` -> one parsed row per distinct capture shape."""
    parsed = []
    for label, count in sorted((census or {}).get("shapes", {}).items(), key=lambda kv: -kv[1]):
        geometry, _, tail = label.partition("/")
        resolution, _, fps = geometry.partition("@")
        codec, _, rotation = tail.partition("/")
        parsed.append(
            {
                "label": label,
                "resolution": resolution,
                "fps": fps,
                "codec": codec,
                "rotation": rotation.removeprefix("rot"),
                "count": count,
            }
        )
    return parsed


def bundle(paths: Paths) -> dict[str, Any]:
    census = _json(paths.inventory / "census.json")
    qualification = _json(paths.qualification / "qualification.json")
    calibration = _json(paths.calibration_qc / "calibration_qc.json")
    run = _json(paths.run / "run_report.json")
    cohort = _json(paths.cohort / "cohort.json")
    return {
        "available": paths.status(),
        "headline": _headline(census, qualification, run, cohort),
        "shapes": _shapes(census),
        "rotation_by_view": (census or {}).get("rotation_by_view", {}),
        "view_coverage": (census or {}).get("captures", {}).get("view_coverage", {}),
        "duration_s": (census or {}).get("assets", {}).get("nominal_duration_s", {}),
        "reason_codes": (census or {}).get("reason_codes", {}),
        "normalization": (census or {}).get("normalization", {}),
        "codec": (qualification or {}).get("assets", {}).get("codec", {}),
        "device_config": (qualification or {}).get("assets", {}).get("device_config", {}),
        "cameras_per_event": (qualification or {}).get("events", {}).get("n_cameras", {}),
        "sync_status": (qualification or {}).get("events", {}).get("sync_status", {}),
        "offset_status": (qualification or {}).get("cameras", {}).get("offset_status", {}),
        "pair_status": (qualification or {}).get("pairs", {}).get("status", {}),
        "qc_flags": (qualification or {}).get("qc_flags", {}),
        "measured_axes": (qualification or {}).get("measured_axes", []),
        "unmeasured_axes": (qualification or {}).get("unmeasured_axes", []),
        "ruling": (calibration or {}).get("corpus", {}).get("ruling", {}),
        "claims": (calibration or {}).get("claims", []),
        "evidence": _rows(paths.calibration_qc / "evidence_qc.csv"),
        "run_verdicts": (run or {}).get("verdicts", {}),
        "run_manifest_census": (run or {}).get("manifest", {}).get("census", {}),
        "run_configuration": (run or {}).get("configuration", {}),
        "provenance": {
            "inventory": (census or {}).get("tool_version"),
            "qualification": (qualification or {}).get("generation", {}).get("generator_version"),
            "calibration_qc": (calibration or {}).get("generation", {}).get("generator_version"),
            "run": (run or {}).get("generator_version"),
            "cohort": (cohort or {}).get("generation", {}).get("generator_version"),
            "opencv": (census or {}).get("opencv_version"),
        },
    }
