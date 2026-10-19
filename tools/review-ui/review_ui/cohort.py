"""Cohort explorer data: the published aggregate plus its bilingual descriptors.

`cohort/` is aggregates only by contract — 12 `(task, side)` cells, no subject row
and no identifier — so the whole tree ships to the browser as published.  The
descriptor join is by the name the publisher already writes,
`pose_<level>_<feature>`, so a renamed feature loses its label loudly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .config import Paths

NUMERIC_FIELDS = (
    "n_subjects",
    "n_events",
    "n_assets",
    "n_values",
    "n_events_multiview",
    "n_subjects_multiview",
    "n_frame_rows",
    "n_window_rows",
    "n_bout_rows",
)
# v2 agreement fields + v1's view_dispersion, so a retained v1 generation still renders.
STATISTIC_FIELDS = (
    "median",
    "q25",
    "q75",
    "mean",
    "sd",
    "view_abs_diff",
    "view_icc",
    "view_icc_range_lo",
    "view_icc_range_hi",
    "view_dispersion",
)


def _typed(row: dict[str, str]) -> dict[str, Any]:
    """Counts to int, statistics to float, an empty statistic to null.

    A cell below the subject floor publishes empty statistics with its counts
    intact, so the two field families need different empty handling: a missing
    count is a defect, a missing statistic is the suppression rule working.
    """
    out: dict[str, Any] = dict(row)
    for field in NUMERIC_FIELDS:
        if field in out:
            out[field] = int(out[field]) if out[field] not in ("", None) else None
    for field in STATISTIC_FIELDS:
        if field in out:
            out[field] = float(out[field]) if out[field] not in ("", None) else None
    return out


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [_typed(row) for row in csv.DictReader(stream)]


def _descriptors(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {column["raw"]: column for column in parsed.get("columns", [])}


def bundle(paths: Paths) -> dict[str, Any]:
    marker_path = paths.cohort / "cohort.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else {}
    rows = _rows(paths.cohort / "cohort_features.csv")
    descriptors = _descriptors(paths.cohort / "descriptors.yaml")

    features: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['level']}:{row['feature']}"
        if key in features:
            continue
        descriptor = descriptors.get(f"pose_{row['level']}_{row['feature']}", {})
        features[key] = {
            "key": key,
            "feature": row["feature"],
            "level": row["level"],
            "ja": descriptor.get("ja"),
            "en": descriptor.get("en"),
            "unit": descriptor.get("unit"),
            "range": descriptor.get("range"),
        }
    return {
        "available": paths.status()["cohort"],
        "population": marker.get("population", {}),
        "estimand": marker.get("estimand"),
        "columns_excluded": marker.get("columns", {}).get("excluded", []),
        "rows_below_subject_floor": marker.get("rows_below_subject_floor"),
        "descriptor_collision": marker.get("descriptor_collision", {}),
        "cells": _rows(paths.cohort / "cohort_cells.csv"),
        "features": sorted(features.values(), key=lambda item: (item["level"], item["feature"])),
        "rows": rows,
    }
