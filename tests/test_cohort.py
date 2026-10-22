"""Diff-blind acceptance suite for the cohort aggregate publisher.

Every fixture is synthetic. The publisher import stays inside each test so the
pre-implementation branch reports one predicate-specific red per case.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import shutil
import stat
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest
import yaml

from pose_estimation import corpus_run, inventory, sessions
from test_sessions import _Asset, _canonical, _write_registry

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_GOLDEN = _ROOT / "tests" / "goldens" / "r_clinical"
_REHAB_SCHEMA = _ROOT.parent / "rehab" / "schema" / "columns.yaml"
_METADATA = {
    "video",
    "frame_idx",
    "timestamp_sec",
    "person_idx",
    "window_start_sec",
    "window_end_sec",
    "side",
    "bout_idx",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "censored",
}
_LEVEL_SUFFIX = {
    "frame": "_clinical.csv",
    "window": "_clinical_windows.csv",
    "bout": "_clinical_bouts.csv",
}
_UNIT_CONSTANTS = {
    "UNIT_DEG": "deg_image_plane",
    "UNIT_FRAME_NORMALIZED": "frame_normalized",
    "UNIT_FRAME_NORMALIZED_PER_S": "frame_normalized_per_s",
    "UNIT_RATIO_SHOULDER_WIDTH": "ratio_shoulder_width",
    "UNIT_RATIO": "ratio",
    "UNIT_INDEX_SIGNED": "index_signed",
    "UNIT_DIMENSIONLESS": "dimensionless",
    "UNIT_SECONDS": "s",
    "UNIT_COUNT": "count",
}
_UNIT_VOCABULARY = frozenset(_UNIT_CONSTANTS.values())
_ANGLE_BASES = frozenset({"elbow_angle_deg", "wrist_deviation_deg", "finger_spread_deg"})
_DISTANCE_BASES = frozenset(
    {
        "reach_raw",
        "grasp_aperture_thumb_index",
        "grasp_aperture_thumb_pinky",
        "wrist_displacement",
        "fingertip_displacement",
    }
)
_DIMENSIONLESS_BASES = frozenset(
    {"wrist_sal", "wrist_normalized_jerk", "fingertip_normalized_jerk"}
)
_TRUNK_ANGLE_COLUMNS = frozenset(
    {
        "trunk_lean_deg",
        "trunk_lean_lateral_deg",
        "trunk_rotation_deg",
        "trunk_lean_mean",
        "trunk_lean_sd",
        "trunk_lean_range",
        "trunk_lean_lateral_mean",
        "trunk_lean_lateral_sd",
        "trunk_rotation_mean",
        "trunk_rotation_sd",
    }
)
_DISTRIBUTION_FIELDS = ("median", "q25", "q75", "mean", "sd")
_AGREEMENT_FIELDS = ("view_abs_diff", "view_icc", "view_icc_range_lo", "view_icc_range_hi")
# M4.5b: window SPARC + normalized jerk leave the export (one movement, never a window).
_CONSTRUCT_INVALID = frozenset(
    ("window", column)
    for base in ("wrist_sal", "wrist_normalized_jerk", "fingertip_normalized_jerk")
    for column in (
        f"left_{base}",
        f"right_{base}",
        f"{base}_symmetry_ratio",
        f"{base}_dominance_index",
        f"{base}_abs_diff",
    )
)
_EXCLUSION_REASONS = frozenset(
    {"structurally_absent", "construct_invalid", "no_eligible_bout_values"}
)
_BOUT_BASES = {
    "bout_duration": ("s", (0.0, None)),
    "bout_sparc": ("dimensionless", (None, 0.0)),
    "bout_normalized_jerk": ("dimensionless", (0.0, None)),
    "bout_peak_time_share": ("ratio", (0.0, 1.0)),
    "bout_speed_peaks": ("count", (1.0, None)),
    "bout_path_ratio": ("ratio", (1.0, None)),
    "bout_dwell": ("s", (0.0, None)),
}
_SUBJECT_FLOOR = 5
_CELL_HEADER = (
    "task",
    "side",
    "n_subjects",
    "n_events",
    "n_assets",
    "n_frame_rows",
    "n_window_rows",
    "n_bout_rows",
)
_FEATURE_HEADER = (
    "task",
    "side",
    "level",
    "feature",
    "n_subjects",
    "n_events",
    "n_assets",
    "n_values",
    "median",
    "q25",
    "q75",
    "mean",
    "sd",
    "view_abs_diff",
    "view_icc",
    "view_icc_range_lo",
    "view_icc_range_hi",
    "n_events_multiview",
    "n_subjects_multiview",
)
_MARKER_KEYS = frozenset(
    {
        "population",
        "columns",
        "estimand",
        "rows_zero_values",
        "agreement_reasons",
        "rows_below_subject_floor",
        "descriptor_collision",
        "measurement",
        "sensitivity",
        "generation",
    }
)
_GENERATION_KEYS = frozenset({"generator", "generator_version", "tree_digest", "input_digests"})
# A22 froze these four names as structural until the engine fixed them; it has, so the
# suite now pins the tokens a consumer reads by name.
_POPULATION_KEYS = frozenset(
    {
        "assets",
        "cells",
        "events",
        "feature_rows",
        "features",
        "frame_rows",
        "subjects",
        "window_rows",
        "bout_rows",
    }
)
_INPUT_DIGEST_KEYS = frozenset(
    {"inventory", "run_manifest", "sessions_generation", "sessions_tree"}
)
_ESTIMAND_DISTRIBUTION = (
    "asset median -> event median -> subject median -> cohort statistic over subjects"
)
_AGREEMENT_REASONS = frozenset(
    {
        "agreement_below_floor",
        "agreement_no_multiview",
        "agreement_zero_variance",
        "agreement_insufficient_resamples",
    }
)
_COLLISION_SOURCE = "../rehab/schema/columns.yaml"
_DESCRIPTOR_KEYS = frozenset({"raw", "ja", "en", "group", "role", "dtype", "unit", "range"})
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class _Inputs:
    inventory: pathlib.Path
    sessions: pathlib.Path
    run: pathlib.Path
    assets: tuple[_Asset, ...]


_InputFactory = Callable[[ModuleType], _Inputs]


def _cohort(predicate: str) -> ModuleType:
    try:
        return importlib.import_module("pose_estimation.cohort")
    except ModuleNotFoundError as error:
        if error.name == "pose_estimation.cohort":
            pytest.fail(f"{predicate}: src/pose_estimation/cohort.py is not implemented")
        raise


def _cohort_error(cohort: ModuleType) -> type[Exception]:
    error = getattr(cohort, "CohortError", None)
    assert isinstance(error, type), "consumer boundary requires public CohortError"
    assert issubclass(error, Exception), "CohortError must derive from Exception"
    return error


def _read_csv(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return tuple(reader.fieldnames), list(reader)


def _artifact_headers() -> dict[str, tuple[str, ...]]:
    names = {level: f"2d_idx{suffix}" for level, suffix in _LEVEL_SUFFIX.items()}
    headers: dict[str, tuple[str, ...]] = {}
    for level, name in names.items():
        header, _ = _read_csv(_GOLDEN / name)
        headers[level] = header
    return headers


def _source_columns() -> frozenset[tuple[str, str]]:
    return frozenset(
        (level, column)
        for level, header in _artifact_headers().items()
        for column in header
        if column not in _METADATA
    )


def _finite_value(cell: str) -> float | None:
    try:
        value = float(cell)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _feature_rows(cohort: ModuleType) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in cohort.FEATURES:
        if isinstance(entry, Mapping):
            row = dict(entry)
        elif hasattr(entry, "__dict__"):
            row = vars(entry)
        else:
            raise AssertionError("D07: FEATURES entries must expose named fields")
        assert set(row) == {"level", "column", "ja", "en", "unit", "range"}, (
            "A08: FEATURES must carry exactly level, column, ja, en, unit, range"
        )
        rows.append(row)
    return rows


def _unsided(column: str) -> str:
    return next(
        (
            column.removeprefix(prefix)
            for prefix in ("left_", "right_")
            if column.startswith(prefix)
        ),
        column,
    )


def _expected_unit(column: str) -> str:
    if _unsided(column) in _BOUT_BASES:
        return _BOUT_BASES[_unsided(column)][0]
    if column.endswith("_symmetry_ratio"):
        return _UNIT_CONSTANTS["UNIT_RATIO"]
    if column.endswith("_dominance_index"):
        return _UNIT_CONSTANTS["UNIT_INDEX_SIGNED"]
    if column in {"posture_symmetry", "posture_symmetry_mean", "posture_symmetry_sd"}:
        return _UNIT_CONSTANTS["UNIT_RATIO_SHOULDER_WIDTH"]
    if column == "compensatory_pattern_index":
        return _UNIT_CONSTANTS["UNIT_INDEX_SIGNED"]
    stem = _unsided(column)
    if stem.endswith("_abs_diff"):
        stem = stem.removesuffix("_abs_diff")
    if stem in _ANGLE_BASES or stem in _TRUNK_ANGLE_COLUMNS:
        return _UNIT_CONSTANTS["UNIT_DEG"]
    if stem in _DISTANCE_BASES:
        return _UNIT_CONSTANTS["UNIT_FRAME_NORMALIZED"]
    if stem == "reach_norm":
        return _UNIT_CONSTANTS["UNIT_RATIO_SHOULDER_WIDTH"]
    if stem in {"wrist_velocity_mean", "wrist_velocity_peak"}:
        return _UNIT_CONSTANTS["UNIT_FRAME_NORMALIZED_PER_S"]
    if stem in _DIMENSIONLESS_BASES:
        return _UNIT_CONSTANTS["UNIT_DIMENSIONLESS"]
    if stem == "wrist_movement_efficiency":
        return _UNIT_CONSTANTS["UNIT_RATIO"]
    raise AssertionError(f"P10: no independent unit rule for {column!r}")


def _expected_range(column: str) -> tuple[float | None, float | None]:
    if _unsided(column) in _BOUT_BASES:
        return _BOUT_BASES[_unsided(column)][1]
    if column.endswith("_symmetry_ratio"):
        return 0.0, 1.0
    if column.endswith("_dominance_index"):
        return -1.0, 1.0
    if column.endswith("_abs_diff"):
        low, high = _expected_range(column.removesuffix("_abs_diff"))
        return 0.0, None if low is None or high is None else high - low
    stem = _unsided(column)
    if stem in _ANGLE_BASES:
        return 0.0, 180.0
    if stem in _DISTANCE_BASES or stem in {
        "reach_norm",
        "wrist_velocity_mean",
        "wrist_velocity_peak",
    }:
        return 0.0, None
    if stem == "wrist_sal":
        return None, 0.0
    if stem in {"wrist_normalized_jerk", "fingertip_normalized_jerk"}:
        return 0.0, None
    if stem == "wrist_movement_efficiency":
        return 1.0, None
    if stem in {"trunk_lean_deg", "trunk_lean_mean"}:
        return 0.0, 90.0
    if stem == "trunk_lean_range":
        return 0.0, 90.0
    if stem in {
        "trunk_lean_lateral_deg",
        "trunk_lean_lateral_mean",
        "trunk_rotation_deg",
        "trunk_rotation_mean",
    }:
        return -180.0, 180.0
    if stem in {"trunk_lean_sd", "trunk_lean_lateral_sd", "trunk_rotation_sd"}:
        return 0.0, None
    if stem in {"posture_symmetry", "posture_symmetry_mean"}:
        return -1.0, 1.0
    if stem == "posture_symmetry_sd":
        return 0.0, None
    if stem == "compensatory_pattern_index":
        return -1.0, 1.0
    raise AssertionError(f"P09/A21: no independent range rule for {column!r}")


def _external_descriptor_raws(path: pathlib.Path) -> frozenset[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), "P09: external descriptor root is not a mapping"
    columns = document.get("columns")
    families = document.get("families")
    assert isinstance(columns, list), "P09: external descriptor schema lacks columns"
    assert isinstance(families, list), "P09: external descriptor schema lacks families"
    raws = [row["raw"] for row in columns]
    for family in families:
        template = family["template_raw"]
        raws.extend(
            template.format(side=side, level=level)
            for side in family["sides"]
            for level in family["levels"]
        )
    assert all(isinstance(raw, str) and raw for raw in raws), (
        "P09: external descriptor raw is not non-empty text"
    )
    assert len(raws) == len(set(raws)), "P09: external descriptor raws already collide"
    return frozenset(raws)


def _value(asset: _Asset) -> float:
    assert asset.subject_ordinal is not None
    if asset.subject_ordinal == 5:
        return 100.0
    if asset.subject_ordinal == 6:
        return 2.0
    if asset.subject_ordinal == 1 and asset.task == inventory.TASKS[0] and asset.side == "l":
        return 4.0 if asset.view == "left" else 2.0
    return float(asset.subject_ordinal)


def _write_artifact(
    path: pathlib.Path,
    header: tuple[str, ...],
    asset: _Asset,
    *,
    level: str,
    published: frozenset[tuple[str, str]],
    repeats: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for index in range(repeats):
            if level == "bout":
                for side in ("left", "right"):
                    row: dict[str, object] = dict.fromkeys(header, "NA")
                    row |= {"video": asset.source_path, "person_idx": 0, "side": side,
                            "bout_idx": index + 1, "start_frame": 10 * index,
                            "end_frame": 10 * index + 9, "start_sec": index / 3,
                            "end_sec": index / 3 + 0.3, "censored": 0}  # fmt: skip
                    for column in header:
                        if column.startswith(f"{side}_") and (level, column) in published:
                            row[column] = _value(asset)
                    writer.writerow(row)
                continue
            row: dict[str, object] = {}
            for column in header:
                if column == "video":
                    row[column] = asset.source_path
                elif column in {"frame_idx", "person_idx"}:
                    row[column] = index if column == "frame_idx" else 0
                elif column == "timestamp_sec":
                    row[column] = index / 30
                elif column == "window_start_sec":
                    row[column] = index
                elif column == "window_end_sec":
                    row[column] = index + 1
                else:
                    row[column] = _value(asset) if (level, column) in published else "NA"
            writer.writerow(row)


# contract-m4u7: every contributing event records the configuration + clinical producer it ran.
_SYNTHETIC_POSE_CONFIG: dict[str, object] = {
    "model": "rtmw-l-384",
    "tracking": "body",
    "tracker": "subject",
    "drop_lower_body": True,
    "drop_hips_camera": "above",
    "hand_gate": True,
    "camera_survey": True,
    "task_gap_s": 2.5,
    "det_device": "GPU",
    "pose_device": "GPU",
    "det_frequency": 1,
    "single_subject": True,
    "openvino_build": "synthetic",
    "calibration": "identity",
}
_SYNTHETIC_ANALYSIS_SHA256 = "a" * 64


def _write_event_metadata(event: pathlib.Path) -> None:
    assert set(_SYNTHETIC_POSE_CONFIG) == set(corpus_run.POSE_CONFIG_FIELDS)
    corpus_run.write_pose_config(event, _SYNTHETIC_POSE_CONFIG)
    corpus_run.write_marker(
        event, status=corpus_run.MARKER_COMPLETE, analysis_sha256=_SYNTHETIC_ANALYSIS_SHA256
    )


def _build_inputs(
    root: pathlib.Path,
    cohort: ModuleType,
    *,
    clone_subject: bool = False,
    all_single_view: bool = False,
    all_multiview: bool = False,
) -> _Inputs:
    subjects = range(1, 7 if clone_subject else 6)
    assets = [
        _canonical(
            subject,
            "above",
            task=task,
            side=side,
            directory=f"synthetic-{subject:02d}-{task}-{side}",
        )
        for task in inventory.TASKS
        for side in inventory.SIDES
        for subject in subjects
    ]
    if all_multiview:
        assets.extend(
            _canonical(
                subject,
                "left",
                task=task,
                side=side,
                directory=f"synthetic-{subject:02d}-{task}-{side}",
            )
            for task in inventory.TASKS
            for side in inventory.SIDES
            for subject in subjects
        )
    elif not all_single_view:
        assets.append(
            _canonical(
                1,
                "left",
                task=inventory.TASKS[0],
                side="l",
                directory="synthetic-01-cap-l",
            )
        )

    registry = _write_registry(root, assets)
    sessions.run(registry.root, registry.corpus, registry.out)
    _, placements = _read_csv(registry.out / sessions.PLACEMENTS_FILENAME)
    by_id = {asset.asset_id: asset for asset in assets}
    run = root / "run"
    run.mkdir()
    manifest: list[dict[str, str]] = []
    headers = _artifact_headers()
    published = frozenset((row["level"], row["column"]) for row in _feature_rows(cohort))
    for placement in placements:
        if placement["placement"] != sessions.PLACED:
            continue
        asset = by_id[placement["asset_id"]]
        repeats = 9 if asset.subject_ordinal == 5 else 1
        event = run / placement["event_id"]
        camera = placement["camera_name"]
        _write_artifact(
            event / f"{camera}_clinical.csv",
            headers["frame"],
            asset,
            level="frame",
            published=published,
            repeats=repeats,
        )
        _write_artifact(
            event / f"{camera}_clinical_windows.csv",
            headers["window"],
            asset,
            level="window",
            published=published | _CONSTRUCT_INVALID,
            repeats=repeats,
        )
        _write_artifact(
            event / f"{camera}_clinical_bouts.csv",
            headers["bout"],
            asset,
            level="bout",
            published=published,
            repeats=repeats,
        )
        _write_event_metadata(event)
        manifest.append(
            {
                "asset_id": asset.asset_id,
                "event_id": placement["event_id"],
                "camera_name": camera,
                "disposition": corpus_run.DISPOSITION_OK,
            }
        )
    corpus_run.write_manifest(run / corpus_run.MANIFEST_FILENAME, manifest)
    return _Inputs(registry.root, registry.out, run, tuple(assets))


@pytest.fixture
def input_factory(tmp_path_factory: pytest.TempPathFactory) -> _InputFactory:
    def build(cohort: ModuleType) -> _Inputs:
        return _build_inputs(tmp_path_factory.mktemp("cohort-inputs"), cohort)

    return build


def _rewrite_feature(
    inputs: _Inputs, level: str, feature: str, replacements: Mapping[int, str]
) -> None:
    suffix = _LEVEL_SUFFIX[level]
    for file_index, path in enumerate(sorted(inputs.run.glob(f"*/*{suffix}"))):
        header, rows = _read_csv(path)
        token = replacements.get(file_index)
        if token is None:
            continue
        for row in rows:
            row[feature] = token
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)


def _rewrite_asset_feature(
    inputs: _Inputs, level: str, feature: str, replacements: Mapping[str, str]
) -> None:
    _, placements = _read_csv(inputs.sessions / sessions.PLACEMENTS_FILENAME)
    suffix = _LEVEL_SUFFIX[level]
    for placement in placements:
        token = replacements.get(placement["asset_id"])
        if token is None or placement["placement"] != sessions.PLACED:
            continue
        path = inputs.run / placement["event_id"] / f"{placement['camera_name']}{suffix}"
        header, rows = _read_csv(path)
        for row in rows:
            row[feature] = token
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)


def _repeat_asset_rows(inputs: _Inputs, asset_id: str, factor: int) -> None:
    _, placements = _read_csv(inputs.sessions / sessions.PLACEMENTS_FILENAME)
    placement = next(row for row in placements if row["asset_id"] == asset_id)
    for suffix in _LEVEL_SUFFIX.values():
        path = inputs.run / placement["event_id"] / f"{placement['camera_name']}{suffix}"
        header, rows = _read_csv(path)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows * factor)


def _publish(cohort: ModuleType, inputs: _Inputs, out: pathlib.Path) -> None:
    cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    cohort.validate_generation(out)


def _marker(out: pathlib.Path) -> dict[str, Any]:
    payload = json.loads((out / "cohort.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _tree_state(root: pathlib.Path) -> dict[str, tuple[str, bytes]]:
    state: dict[str, tuple[str, bytes]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            state[relative] = ("link", os.fsencode(os.readlink(path)))  # noqa: PTH115
        elif path.is_file():
            state[relative] = ("file", path.read_bytes())
        else:
            state[relative] = ("dir", b"")
    return state


def _validated_input_witnesses(inputs: _Inputs) -> tuple[str, str]:
    inventory.validate_generation(inputs.inventory)
    sessions.validate_generation(inputs.sessions, inventory_dir=inputs.inventory)
    rows = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    corpus_run.validate_manifest(rows, [asset.asset_id for asset in inputs.assets])
    return sessions.tree_digest(inputs.sessions), sessions.generation_digest(inputs.sessions)


def _published_bytes(out: pathlib.Path) -> dict[str, bytes]:
    names = ["cohort_cells.csv", "cohort_features.csv", "descriptors.yaml", "cohort.json"]
    names += [name for name in ("cohort_sensitivity.csv",) if (out / name).exists()]
    return {name: (out / name).read_bytes() for name in names}


def _independent_cells(inputs: _Inputs) -> dict[tuple[str, str], dict[str, int]]:
    _, asset_rows = _read_csv(inputs.inventory / inventory.ASSETS_FILENAME)
    _, placements = _read_csv(inputs.sessions / sessions.PLACEMENTS_FILENAME)
    manifest = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    assets = {row["asset_id"]: row for row in asset_rows}
    placed = {row["asset_id"]: row for row in placements if row["placement"] == sessions.PLACED}
    good = [row for row in manifest if row["disposition"] == corpus_run.DISPOSITION_OK]
    members: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"subjects": set(), "events": set(), "assets": set()}
    )
    row_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: dict.fromkeys(_LEVEL_SUFFIX, 0)
    )
    for manifest_row in good:
        asset_id = manifest_row["asset_id"]
        asset = assets[asset_id]
        placement = placed[asset_id]
        cell = asset["task"], asset["side"]
        members[cell]["subjects"].add(asset["subject_ordinal"])
        members[cell]["events"].add(placement["event_id"])
        members[cell]["assets"].add(asset_id)
        for level, suffix in _LEVEL_SUFFIX.items():
            path = inputs.run / placement["event_id"] / f"{placement['camera_name']}{suffix}"
            _, rows = _read_csv(path)
            row_counts[cell][level] += len(rows)
    return {
        cell: {
            "n_subjects": len(populations["subjects"]),
            "n_events": len(populations["events"]),
            "n_assets": len(populations["assets"]),
            "n_frame_rows": row_counts[cell]["frame"],
            "n_window_rows": row_counts[cell]["window"],
            "n_bout_rows": row_counts[cell]["bout"],
        }
        for cell, populations in members.items()
    }


def _measured_partition(
    inputs: _Inputs,
) -> tuple[frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
    _, placements = _read_csv(inputs.sessions / sessions.PLACEMENTS_FILENAME)
    manifest = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    placed = {row["asset_id"]: row for row in placements if row["placement"] == sessions.PLACED}
    published: set[tuple[str, str]] = set()
    measured: set[tuple[str, str]] = set()
    for manifest_row in manifest:
        if manifest_row["disposition"] != corpus_run.DISPOSITION_OK:
            continue
        placement = placed[manifest_row["asset_id"]]
        for level, suffix in _LEVEL_SUFFIX.items():
            path = inputs.run / placement["event_id"] / f"{placement['camera_name']}{suffix}"
            header, rows = _read_csv(path)
            for column in header:
                if column in _METADATA:
                    continue
                key = level, column
                measured.add(key)
                if any(_finite_value(row[column]) is not None for row in rows):
                    published.add(key)
    # M4.5b: the retired window smoothness columns leave the export whatever their values.
    published -= _CONSTRUCT_INVALID
    return frozenset(published), frozenset(measured - published)


def _render_cell(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.9f}"


def _linear_quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _independent_stats(
    inputs: _Inputs, level: str, feature: str
) -> dict[tuple[str, str], dict[str, float | int | None]]:
    _, asset_rows = _read_csv(inputs.inventory / inventory.ASSETS_FILENAME)
    _, placements = _read_csv(inputs.sessions / sessions.PLACEMENTS_FILENAME)
    manifest = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    assets = {row["asset_id"]: row for row in asset_rows}
    placed = {row["asset_id"]: row for row in placements if row["placement"] == sessions.PLACED}
    good = {row["asset_id"] for row in manifest if row["disposition"] == corpus_run.DISPOSITION_OK}

    event_assets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    raw_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    asset_counts: dict[tuple[str, str], int] = defaultdict(int)
    for asset_id in good:
        asset = assets[asset_id]
        placement = placed[asset_id]
        suffix = _LEVEL_SUFFIX[level]
        path = inputs.run / placement["event_id"] / f"{placement['camera_name']}{suffix}"
        _, rows = _read_csv(path)
        if level == "bout":  # censored rows never contribute (contract-m4u5b §1)
            rows = [row for row in rows if row["censored"] == "0"]
        values = [value for row in rows if (value := _finite_value(row[feature])) is not None]
        if not values:
            continue
        cell = (asset["task"], asset["side"])
        raw_values[cell].extend(values)
        asset_counts[cell] += 1
        event_assets[
            (asset["task"], asset["side"], asset["subject_ordinal"], placement["event_id"])
        ].append(statistics.median(values))

    subject_events: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    event_counts: dict[tuple[str, str], int] = defaultdict(int)
    multiview: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    multiview_subjects: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (task, side, subject, _event), values in event_assets.items():
        cell = (task, side)
        event_counts[cell] += 1
        subject_events[(task, side, subject)].append(statistics.median(values))
        if len(values) >= 2:
            multiview[cell].append(values)
            multiview_subjects[cell].add(subject)

    subjects: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (task, side, _subject), values in subject_events.items():
        subjects[(task, side)].append(statistics.median(values))

    result: dict[tuple[str, str], dict[str, float | int | None]] = {}
    for cell, values in subjects.items():
        result[cell] = {
            "n_subjects": len(values),
            "n_events": event_counts[cell],
            "n_assets": asset_counts[cell],
            "n_values": len(raw_values[cell]),
            "median": statistics.median(values),
            "q25": _linear_quantile(values, 0.25),
            "q75": _linear_quantile(values, 0.75),
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) >= 2 else None,
            "n_events_multiview": len(multiview[cell]),
            "n_subjects_multiview": len(multiview_subjects[cell]),
            "pooled_median": statistics.median(raw_values[cell]),
        }
        enough = len(multiview_subjects[cell]) >= _SUBJECT_FLOOR
        result[cell]["view_abs_diff"] = (
            statistics.median(
                statistics.fmean(abs(a - b) for i, a in enumerate(v) for b in v[i + 1 :])
                for v in multiview[cell]
            )
            if enough
            else None
        )
        result[cell]["view_icc"] = _oracle_icc(multiview[cell]) if enough else None
    return result


def _oracle_icc(groups: list[list[float]]) -> float | None:
    """ICC(1,1), unbalanced one-way ANOVA (Shrout-Fleiss with the k0 view count)."""
    g = len(groups)
    n = sum(len(v) for v in groups)
    grand = sum(sum(v) for v in groups) / n
    msb = sum(len(v) * (statistics.fmean(v) - grand) ** 2 for v in groups) / (g - 1)
    msw = sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in groups) / (n - g)
    k0 = (n - sum(len(v) ** 2 for v in groups) / n) / (g - 1)
    denominator = msb + (k0 - 1) * msw
    return None if denominator <= 0 else (msb - msw) / denominator


def _feature_output(out: pathlib.Path, key: tuple[str, str, str, str]) -> dict[str, str]:
    _, rows = _read_csv(out / "cohort_features.csv")
    task, side, level, feature = key
    matches = [
        row
        for row in rows
        if (row["task"], row["side"], row["level"], row["feature"]) == (task, side, level, feature)
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_ascii_uint(cell: str, field: str) -> None:
    assert cell, f"P11: {field} is empty"
    assert cell.isascii(), f"P11: {field} uses non-ASCII digits"
    assert cell.isdecimal(), f"P11: {field} is not an unsigned integer"


def _assert_numeric_or_empty(cell: str, field: str) -> None:
    if not cell:
        return
    try:
        value = float(cell)
    except ValueError as error:
        raise AssertionError(f"P11: {field} is neither empty nor numeric") from error
    assert math.isfinite(value), f"P11: {field} is not finite"


def _assert_nonnegative_int(value: object, field: str) -> None:
    assert type(value) is int, f"P11: {field} is not an integer"
    assert value >= 0, f"P11: {field} is negative"


def _assert_typed_publication(cohort: ModuleType, out: pathlib.Path) -> None:
    cell_header, cell_rows = _read_csv(out / "cohort_cells.csv")
    assert cell_header == _CELL_HEADER, "P11: cell key set drifted"
    assert cell_rows, "P11: cell allowlist is vacuous"
    for row in cell_rows:
        assert row["task"] in inventory.TASKS, "P11: cell task is outside the registry domain"
        assert row["side"] in inventory.SIDES, "P11: cell side is outside the registry domain"
        for field in _CELL_HEADER[2:]:
            _assert_ascii_uint(row[field], field)

    feature_header, feature_rows = _read_csv(out / "cohort_features.csv")
    assert feature_header == _FEATURE_HEADER, "P11: feature key set drifted"
    assert feature_rows, "P11: feature allowlist is vacuous"
    features = {(row["level"], row["column"]): row for row in _feature_rows(cohort)}
    for row in feature_rows:
        assert row["task"] in inventory.TASKS, "P11: feature task is outside the registry domain"
        assert row["side"] in inventory.SIDES, "P11: feature side is outside the registry domain"
        key = row["level"], row["feature"]
        assert key in features, "P11: feature key is outside FEATURES"
        for field in (
            "n_subjects",
            "n_events",
            "n_assets",
            "n_values",
            "n_events_multiview",
            "n_subjects_multiview",
        ):
            _assert_ascii_uint(row[field], field)
        for field in (*_DISTRIBUTION_FIELDS, *_AGREEMENT_FIELDS):
            _assert_numeric_or_empty(row[field], field)
        if int(row["n_subjects_multiview"]) < 5:
            assert all(row[field] == "" for field in _AGREEMENT_FIELDS), (
                "P11: a below-floor multi-view population exposed an agreement field"
            )
        if int(row["n_subjects"]) < 5:
            assert all(row[field] == "" for field in _DISTRIBUTION_FIELDS), (
                "P11: a below-floor subject population exposed a distribution"
            )

    document = yaml.safe_load((out / "descriptors.yaml").read_text(encoding="utf-8"))
    assert isinstance(document, dict), "P11: descriptor root is not a mapping"
    assert set(document) == {"columns"}, "P11: descriptor root key set drifted"
    descriptors = document["columns"]
    assert isinstance(descriptors, list), "P11: descriptor rows are not a list"
    assert descriptors, "P11: descriptor allowlist is vacuous"
    feature_by_raw = {f"pose_{row['level']}_{row['column']}": row for row in _feature_rows(cohort)}
    for row in descriptors:
        assert isinstance(row, dict), "P11: descriptor row is not a mapping"
        assert set(row) == _DESCRIPTOR_KEYS, "P11: descriptor field set drifted"
        assert row["raw"] in feature_by_raw, "P11: descriptor raw is outside FEATURES"
        feature = feature_by_raw[row["raw"]]
        assert row["ja"] == feature["ja"], "P11: descriptor ja is outside FEATURES"
        assert row["en"] == feature["en"], "P11: descriptor en is outside FEATURES"
        assert row["group"] == "pose", "P11: descriptor group is outside its domain"
        assert row["role"] == "feature", "P11: descriptor role is outside its domain"
        assert row["dtype"] == "numeric", "P11: descriptor dtype is outside its domain"
        assert row["unit"] in _UNIT_VOCABULARY, "P11: descriptor unit is outside its domain"
        assert tuple(row["range"]) == tuple(feature["range"]), (
            "P11: descriptor range is outside FEATURES"
        )

    marker = _marker(out)
    assert set(marker) == _MARKER_KEYS, "P11: cohort.json top-level field set drifted"
    population = marker["population"]
    assert isinstance(population, dict), "P11/A22: population is not a mapping"
    assert set(population) == _POPULATION_KEYS, "P11/A22: population field set drifted"
    for key, value in population.items():
        _assert_nonnegative_int(value, f"population.{key}")
    assert marker["estimand"]["distribution"] == _ESTIMAND_DISTRIBUTION, (
        "P11/A22: estimand text drifted"
    )
    assert set(marker["estimand"]) == {"distribution", "agreement", "bout"}
    for field in ("rows_zero_values", "rows_below_subject_floor"):
        _assert_nonnegative_int(marker[field], field)
    assert set(marker["agreement_reasons"]) == _AGREEMENT_REASONS
    for field, value in marker["agreement_reasons"].items():
        _assert_nonnegative_int(value, field)

    columns = marker["columns"]
    assert isinstance(columns, dict), "P11: column census is not a mapping"
    assert set(columns) == {"published", "excluded"}, "P11: column census fields drifted"
    assert isinstance(columns["published"], list), "P11: published census is not a list"
    assert isinstance(columns["excluded"], list), "P11: excluded census is not a list"
    for row in columns["published"]:
        assert set(row) == {"level", "column"}, "P11: published census row fields drifted"
        assert (row["level"], row["column"]) in features, (
            "P11: published census value is outside FEATURES"
        )
    for row in columns["excluded"]:
        assert set(row) == {"level", "column", "reason"}, "P11: excluded census row fields drifted"
        assert (row["level"], row["column"]) in _source_columns(), (
            "P11: excluded census value is outside source headers"
        )
        assert row["reason"] in _EXCLUSION_REASONS, (
            "P11: exclusion reason is outside its frozen domain"
        )

    collision = marker["descriptor_collision"]
    assert isinstance(collision, dict), "P11: descriptor collision is not a mapping"
    assert set(collision) == {"checked", "source", "source_sha256", "n_external", "n_collisions"}, (
        "P11: descriptor collision fields drifted"
    )
    assert type(collision["checked"]) is bool, "P11: collision checked is not boolean"
    assert collision["source"] == _COLLISION_SOURCE, "P11/A22: collision source drifted"
    _assert_nonnegative_int(collision["n_external"], "descriptor_collision.n_external")
    _assert_nonnegative_int(collision["n_collisions"], "descriptor_collision.n_collisions")

    generation = marker["generation"]
    assert isinstance(generation, dict), "P11: generation is not a mapping"
    assert set(generation) == _GENERATION_KEYS, "P11: generation fields drifted"
    assert generation["generator"] == cohort.GENERATOR, "P11: generator token drifted"
    assert generation["generator_version"] == cohort.GENERATOR_VERSION, (
        "P11: generator version drifted"
    )
    assert _is_digest(generation["tree_digest"]), "P11: tree digest is outside its domain"
    input_digests = generation["input_digests"]
    assert isinstance(input_digests, dict), "P11/A22: input digests are not a mapping"
    assert set(input_digests) == _INPUT_DIGEST_KEYS, "P11/A22: input digest field set drifted"
    for key, value in input_digests.items():
        assert _is_digest(value), f"P11: input digest {key!r} is outside its domain"


# P01-P04: aggregate cardinality, census, and estimand.


def test_p01_cells_are_the_complete_registry_task_side_product(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P01")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    header, rows = _read_csv(out / "cohort_cells.csv")
    assert header == _CELL_HEADER, "P01: cohort_cells.csv header drifted"
    keys = [(row["task"], row["side"]) for row in rows]
    expected = {(task, side) for task in inventory.TASKS for side in inventory.SIDES}
    assert len(rows) == 12, "P01: the publisher did not emit exactly 12 cells"
    assert len(keys) == len(set(keys)), "P01/N1: duplicate cell key"
    assert set(keys) == expected, "P01/N1: a registry cell is missing or invented"
    oracle = _independent_cells(cohort_inputs)
    assert set(oracle) == expected, "P01: synthetic manifest does not cover every registry cell"
    for row in rows:
        cell = row["task"], row["side"]
        for field, value in oracle[cell].items():
            assert row[field] == str(value), f"P01: {field} does not count manifest-ok assets"
        assert int(row["n_subjects"]) >= 5, "P01: small cell published"


def test_p02_features_form_the_exact_cell_by_label_table_product(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P02")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    feature_rows = _feature_rows(cohort)
    header, rows = _read_csv(out / "cohort_features.csv")
    assert header == _FEATURE_HEADER, "P02: cohort_features.csv header drifted"
    feature_keys = [(row["level"], row["column"]) for row in feature_rows]
    assert len(feature_keys) == len(set(feature_keys)), "P02: FEATURES contains a duplicate key"
    cells = {(task, side) for task in inventory.TASKS for side in inventory.SIDES}
    expected = {
        (task, side, level, column) for task, side in cells for level, column in feature_keys
    }
    keys = [(row["task"], row["side"], row["level"], row["feature"]) for row in rows]
    assert len(rows) == len(cells) * len(feature_rows), (
        "P02: feature row count is not cells x len(FEATURES)"
    )
    assert len(expected) == len(cells) * len(feature_rows), (
        "P02: independently derived Cartesian count is inconsistent"
    )
    assert len(keys) == len(set(keys)), "P02/N2: duplicate feature key"
    assert set(keys) == expected, "P02/N2: feature key was dropped or invented"
    _, cell_rows = _read_csv(out / "cohort_cells.csv")
    cell_index = {(row["task"], row["side"]): row for row in cell_rows}
    for row in rows:
        cell = cell_index[row["task"], row["side"]]
        for field in ("n_subjects", "n_events", "n_assets"):
            assert int(row[field]) <= int(cell[field]), f"P02: {field} exceeds its cell population"
        row_field = "n_frame_rows" if row["level"] == "frame" else "n_window_rows"
        assert int(row["n_values"]) <= int(cell[row_field]), (
            "P02: finite leaf count exceeds its cell row population"
        )


def test_p03_column_census_is_the_independent_header_partition(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P03")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    marker = _marker(out)
    columns = marker.get("columns")
    assert isinstance(columns, dict), "P03: cohort.json has no column census"
    published_rows = columns.get("published")
    excluded_rows = columns.get("excluded")
    assert isinstance(published_rows, list), "P03: published census must be a row list"
    assert isinstance(excluded_rows, list), "P03: excluded census must be a row list"
    assert all(set(row) == {"level", "column"} for row in published_rows), (
        "P03: published census row shape drifted"
    )
    assert all(set(row) == {"level", "column", "reason"} for row in excluded_rows), (
        "P03: excluded census row shape drifted"
    )
    published = {(row["level"], row["column"]) for row in published_rows}
    excluded = {(row["level"], row["column"]) for row in excluded_rows}
    measured_published, measured_excluded = _measured_partition(cohort_inputs)
    assert measured_published | measured_excluded == _source_columns(), (
        "P03: synthetic measurement does not cover both source headers"
    )
    assert published == measured_published, "P03/N3: published is not exactly the finite set"
    assert excluded == measured_excluded, "P03: excluded is not exactly the zero-finite set"
    assert not published & excluded, "P03: census buckets overlap"
    for row in excluded_rows:
        key = row["level"], row["column"]
        expected = (
            "construct_invalid"
            if key in _CONSTRUCT_INVALID
            else "no_eligible_bout_values"
            if row["level"] == "bout"
            else "structurally_absent"
        )
        assert row["reason"] == expected, "P03: excluded column carries the wrong reason"
    assert excluded >= _CONSTRUCT_INVALID, "P03/M4.5b: a retired window column was published"
    table_columns = {(row["level"], row["column"]) for row in _feature_rows(cohort)}
    assert table_columns == measured_published, (
        "P03: FEATURES and the measured published set disagree"
    )


def test_p03_zero_finite_column_with_lagging_features_fails_by_name(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P03")
    root = tmp_path / "zero-finite-column"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    feature = _feature_rows(cohort)[0]
    level, column = feature["level"], feature["column"]
    suffix = _LEVEL_SUFFIX[level]
    files = sorted(inputs.run.glob(f"*/*{suffix}"))
    _rewrite_feature(inputs, level, column, dict.fromkeys(range(len(files)), "NA"))
    measured_published, measured_excluded = _measured_partition(inputs)
    assert (level, column) not in measured_published, (
        "P03: zero-finite seed remained measured-published"
    )
    assert (level, column) in measured_excluded, "P03: zero-finite seed is not discriminating"
    table_columns = {(row["level"], row["column"]) for row in _feature_rows(cohort)}
    assert (level, column) in table_columns, "P03: lagging-FEATURES seed is not discriminating"
    out = tmp_path / "cohort"
    with pytest.raises(_cohort_error(cohort)) as caught:
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    assert column in str(caught.value), "P03: feature-census mismatch did not name the column"
    assert not out.exists(), "P03: FEATURES-ahead mismatch still published a generation"


def test_p03_finite_column_with_lagging_features_fails_by_name(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P03")
    root = tmp_path / "lagging-features"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    feature_keys = {(row["level"], row["column"]) for row in _feature_rows(cohort)}
    candidates = sorted(_source_columns() - feature_keys)
    assert candidates, "P03: lagging-FEATURES seed needs one excluded source column"
    level, column = candidates[0]
    suffix = _LEVEL_SUFFIX[level]
    files = sorted(inputs.run.glob(f"*/*{suffix}"))
    _rewrite_feature(inputs, level, column, dict.fromkeys(range(len(files)), "3.5"))
    measured_published, _ = _measured_partition(inputs)
    assert (level, column) in measured_published, "P03/N3: finite seed stayed excluded"
    assert (level, column) not in feature_keys, "P03: lagging-FEATURES seed is not discriminating"
    out = tmp_path / "cohort"
    with pytest.raises(_cohort_error(cohort)) as caught:
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    assert column in str(caught.value), "P03/N3: finite-table mismatch did not name the column"
    assert not out.exists(), "P03/N3: lagging FEATURES still published a generation"


def test_p04_four_stage_subject_estimand_matches_an_independent_oracle(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P04")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    feature = next(row for row in _feature_rows(cohort) if row["level"] == "window")
    cell = (inventory.TASKS[0], "l")
    expected = _independent_stats(cohort_inputs, "window", feature["column"])[cell]
    actual = _feature_output(out, (*cell, "window", feature["column"]))
    for field in (
        "n_subjects",
        "n_events",
        "n_assets",
        "n_values",
        "n_events_multiview",
        "n_subjects_multiview",
    ):
        assert actual[field] == _render_cell(expected[field]), (
            f"P04: {field} does not use the four-stage population"
        )
    for field in ("median", "q25", "q75", "mean", "sd", "view_abs_diff", "view_icc"):
        assert actual[field] == _render_cell(expected[field]), (
            f"P04: {field} differs byte-for-byte from the four-stage oracle"
        )
    assert expected["pooled_median"] != expected["median"], (
        "P04/N4: synthetic seed is not discriminating"
    )
    assert actual["median"] != _render_cell(expected["pooled_median"]), (
        "P04/N4: publisher pooled source rows"
    )


def test_p04_singleton_subject_population_has_empty_sd(tmp_path: pathlib.Path) -> None:
    cohort = _cohort("P04")
    root = tmp_path / "singleton"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    feature = next(row for row in _feature_rows(cohort) if row["level"] == "window")
    target = inventory.TASKS[0], "l"
    replacements = {
        asset.asset_id: "NA"
        for asset in inputs.assets
        if (asset.task, asset.side) == target and asset.subject_ordinal != 1
    }
    _rewrite_asset_feature(inputs, "window", feature["column"], replacements)
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    expected = _independent_stats(inputs, "window", feature["column"])[target]
    actual = _feature_output(out, (*target, "window", feature["column"]))
    assert expected["n_subjects"] == 1, "P04: singleton seed is not discriminating"
    assert expected["sd"] is None, "P04: independent singleton SD must be absent"
    assert actual["sd"] == "", "P04: singleton sample SD must serialize empty, never zero"


# P05-P08: weighting, view population, non-vacuity, and finite filtering.


def test_p05_repeating_asset_rows_moves_counts_but_not_subject_estimates(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P05")
    base_root = tmp_path / "base"
    repeated_root = tmp_path / "repeated"
    base_root.mkdir()
    repeated_root.mkdir()
    base = _build_inputs(base_root, cohort)
    repeated = _build_inputs(repeated_root, cohort)
    target = inventory.TASKS[0], "l"
    repeated_asset = next(
        asset
        for asset in repeated.assets
        if (asset.task, asset.side, asset.subject_ordinal, asset.view) == (*target, 2, "above")
    )
    _repeat_asset_rows(repeated, repeated_asset.asset_id, 5)
    base_out = tmp_path / "base-cohort"
    repeated_out = tmp_path / "repeated-cohort"
    _publish(cohort, base, base_out)
    _publish(cohort, repeated, repeated_out)
    _, base_rows = _read_csv(base_out / "cohort_features.csv")
    _, repeated_rows = _read_csv(repeated_out / "cohort_features.csv")

    def feature_key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return row["task"], row["side"], row["level"], row["feature"]

    left = {feature_key(row): row for row in base_rows}
    right = {feature_key(row): row for row in repeated_rows}
    assert set(left) == set(right), "P05: replication moved the feature population"
    estimates = ("median", "q25", "q75", "mean", "sd", *_AGREEMENT_FIELDS)
    assert all(left[item][field] == right[item][field] for item in left for field in estimates), (
        "P05/N5: source-row multiplicity moved a subject-weighted estimate"
    )
    grains = ("n_subjects", "n_events", "n_assets")
    assert all(left[item][field] == right[item][field] for item in left for field in grains), (
        "P05: source-row replication moved a finite-contributor grain count"
    )
    for item in left:
        if item[:2] == target:
            assert int(right[item]["n_values"]) > int(left[item]["n_values"]), (
                "P05: n_values did not count the replicated finite rows"
            )
        else:
            assert right[item]["n_values"] == left[item]["n_values"], (
                "P05: row replication moved an unrelated feature count"
            )

    def cell_index(out: pathlib.Path) -> dict[tuple[str, str], dict[str, str]]:
        _, rows = _read_csv(out / "cohort_cells.csv")
        return {(row["task"], row["side"]): row for row in rows}

    base_cells = cell_index(base_out)
    repeated_cells = cell_index(repeated_out)
    assert set(base_cells) == set(repeated_cells), "P05: replication moved the cell population"
    assert all(
        base_cells[cell][field] == repeated_cells[cell][field]
        for cell in base_cells
        for field in grains
    ), "P05: source-row replication moved a cell grain count"
    for cell in base_cells:
        for field in ("n_frame_rows", "n_window_rows"):
            if cell == target:
                assert int(repeated_cells[cell][field]) > int(base_cells[cell][field]), (
                    f"P05: {field} did not count replicated rows"
                )
            else:
                assert repeated_cells[cell][field] == base_cells[cell][field], (
                    f"P05: {field} moved outside the replicated asset's cell"
                )


def test_p05_cloning_a_whole_subject_moves_every_population_count(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P05")
    base_root = tmp_path / "base"
    clone_root = tmp_path / "clone"
    base_root.mkdir()
    clone_root.mkdir()
    base = _build_inputs(base_root, cohort)
    clone = _build_inputs(clone_root, cohort, clone_subject=True)
    base_out = tmp_path / "base-cohort"
    clone_out = tmp_path / "clone-cohort"
    _publish(cohort, base, base_out)
    _publish(cohort, clone, clone_out)

    def features(out: pathlib.Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
        _, rows = _read_csv(out / "cohort_features.csv")
        return {(row["task"], row["side"], row["level"], row["feature"]): row for row in rows}

    left = features(base_out)
    right = features(clone_out)
    assert set(left) == set(right), "P05: subject clone moved the feature product"
    for feature in left:
        for field in ("n_subjects", "n_events", "n_assets", "n_values"):
            assert int(right[feature][field]) > int(left[feature][field]), (
                f"P05: whole-subject clone did not move feature {field}"
            )

    def cells(out: pathlib.Path) -> dict[tuple[str, str], dict[str, str]]:
        _, rows = _read_csv(out / "cohort_cells.csv")
        return {(row["task"], row["side"]): row for row in rows}

    base_cells = cells(base_out)
    clone_cells = cells(clone_out)
    assert set(base_cells) == set(clone_cells), "P05: subject clone moved the cell product"
    assert all(
        int(clone_cells[cell][field]) > int(base_cells[cell][field])
        for cell in base_cells
        for field in ("n_subjects", "n_events", "n_assets", "n_frame_rows", "n_window_rows")
    ), "P05: whole-subject clone did not move every cell population count"


def test_p06_view_agreement_matches_the_multiview_event_population(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P06")
    root = tmp_path / "all-multiview"
    root.mkdir()
    inputs = _build_inputs(root, cohort, all_multiview=True)
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    feature_rows = _feature_rows(cohort)
    representatives = [
        next(row for row in feature_rows if row["level"] == level)
        for level in ("frame", "window", "bout")
    ]
    for feature in representatives:
        expected = _independent_stats(inputs, feature["level"], feature["column"])
        assert all(
            int(row["n_subjects_multiview"] or 0) >= _SUBJECT_FLOOR for row in expected.values()
        ), f"P06: {feature['level']} agreement oracle is below the floor (vacuous)"
        for cell, oracle in expected.items():
            actual = _feature_output(out, (*cell, feature["level"], feature["column"]))
            for field in (
                "n_events_multiview",
                "n_subjects_multiview",
                "view_abs_diff",
                "view_icc",
            ):
                assert actual[field] == _render_cell(oracle[field]), (
                    f"P06: {field} differs from the independent ANOVA oracle"
                )
            assert bool(actual["view_icc_range_lo"]) == bool(actual["view_icc"]), (
                "P06: a range published without its point estimate (or the reverse)"
            )
    _, rows = _read_csv(out / "cohort_features.csv")
    assert all(0 <= int(row["n_events_multiview"]) <= int(row["n_events"]) for row in rows), (
        "P06: multiview population exceeds its finite-event population"
    )
    assert all(
        int(row["n_subjects_multiview"]) <= int(row["n_events_multiview"]) for row in rows
    ), "P06: multi-view subjects exceed multi-view events"


def test_p06_multiview_eligibility_is_per_feature_finite_population(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P06")
    root = tmp_path / "feature-finite"
    root.mkdir()
    inputs = _build_inputs(root, cohort, all_multiview=True)
    features = [row for row in _feature_rows(cohort) if row["level"] == "frame"][:2]
    assert len(features) == 2, "P06: finite-population seed needs two frame features"
    target = inventory.TASKS[0], "l"
    left_view = next(
        asset
        for asset in inputs.assets
        if (asset.task, asset.side, asset.subject_ordinal, asset.view) == (*target, 1, "left")
    )
    _rewrite_asset_feature(inputs, "frame", features[0]["column"], {left_view.asset_id: "NA"})
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    ineligible = _feature_output(out, (*target, "frame", features[0]["column"]))
    control = _feature_output(out, (*target, "frame", features[1]["column"]))
    assert ineligible["n_events_multiview"] == "4", (
        "P06: event with only one finite asset value remained eligible"
    )
    assert ineligible["n_subjects_multiview"] == "4"
    assert all(ineligible[field] == "" for field in _AGREEMENT_FIELDS), (
        "P06/D03: four multi-view subjects published agreement below the floor"
    )
    assert control["n_subjects_multiview"] == "5", (
        "P06: eligibility was stored as a per-cell constant rather than per feature"
    )
    assert control["view_abs_diff"], "P06: eligible control row has no agreement"


def test_p06_zero_mean_multiview_event_counts_toward_agreement(tmp_path: pathlib.Path) -> None:
    """The v1 CV had no scale at mean zero (A06); |difference| and the ICC do (M4.5b D01)."""
    cohort = _cohort("P06")
    root = tmp_path / "zero-mean"
    root.mkdir()
    inputs = _build_inputs(root, cohort, all_multiview=True)
    feature = next(row for row in _feature_rows(cohort) if row["column"] == "posture_symmetry")
    target = inventory.TASKS[0], "l"
    event_assets = [
        asset
        for asset in inputs.assets
        if (asset.task, asset.side, asset.subject_ordinal) == (*target, 1)
    ]
    assert {asset.view for asset in event_assets} == {"above", "left"}, (
        "P06: zero-mean seed does not contain exactly two target views"
    )
    replacements = {
        asset.asset_id: "-1" if asset.view == "above" else "1" for asset in event_assets
    }
    _rewrite_asset_feature(inputs, "frame", feature["column"], replacements)
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    expected = _independent_stats(inputs, "frame", feature["column"])[target]
    actual = _feature_output(out, (*target, "frame", feature["column"]))
    assert expected["n_events_multiview"] == 5, "P06: zero-mean event left the oracle"
    assert actual["n_events_multiview"] == "5", "P06: zero-mean event left n_events_multiview"
    assert actual["view_abs_diff"] == _render_cell(expected["view_abs_diff"])
    assert actual["view_icc"] == _render_cell(expected["view_icc"])


def test_p06_all_single_view_events_publish_empty_agreement(tmp_path: pathlib.Path) -> None:
    cohort = _cohort("P06")
    root = tmp_path / "single-view"
    root.mkdir()
    inputs = _build_inputs(root, cohort, all_single_view=True)
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    _, rows = _read_csv(out / "cohort_features.csv")
    assert rows, "P06: single-view seed published no feature rows"
    assert all(row["n_events_multiview"] == "0" for row in rows), (
        "P06: single-view event entered the multiview population"
    )
    assert all(row[field] == "" for row in rows for field in _AGREEMENT_FIELDS), (
        "P06/N6: absent agreement must serialize as an empty cell"
    )
    reasons = _marker(out)["agreement_reasons"]
    assert reasons["agreement_no_multiview"] == len(rows), (
        "P06: every single-view row must count as agreement_no_multiview"
    )


def test_p07_artifact_has_nonvacuous_cells_features_and_populations(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P07")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    _, cells = _read_csv(out / "cohort_cells.csv")
    _, rows = _read_csv(out / "cohort_features.csv")
    marker = _marker(out)
    columns = marker.get("columns")
    assert cells, "P07: published cell set is empty"
    assert rows, "P07: published feature table is empty"
    assert isinstance(columns, dict), "P07: marker has no column census"
    assert columns.get("published"), "P07: published source-feature set is empty"
    assert any(int(row["n_values"]) > 0 for row in rows), (
        "P07: no finite feature population was published"
    )
    assert any(int(row["n_events_multiview"]) > 0 for row in rows), (
        "P07: no multiview feature population was published"
    )
    assert marker.get("rows_zero_values") == sum(int(row["n_values"]) == 0 for row in rows), (
        "P07: rows_zero_values is not the measured feature-row census"
    )
    reasons = marker["agreement_reasons"]
    assert reasons["agreement_no_multiview"] == sum(
        int(row["n_events_multiview"]) == 0 for row in rows
    ), "P07: agreement_no_multiview is not the measured feature-row census"
    assert reasons["agreement_below_floor"] == sum(
        0 < int(row["n_subjects_multiview"]) < _SUBJECT_FLOOR for row in rows
    ), "P07: agreement_below_floor is not the measured feature-row census"


def test_p07_all_multiview_fixture_publishes_zero_empty_row_censuses(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P07")
    root = tmp_path / "all-multiview"
    root.mkdir()
    inputs = _build_inputs(root, cohort, all_multiview=True)
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    _, rows = _read_csv(out / "cohort_features.csv")
    assert rows, "P07: all-multiview seed published no feature rows"
    assert all(int(row["n_values"]) > 0 for row in rows), (
        "P07: all-multiview seed contains a zero-value feature row"
    )
    assert all(int(row["n_events_multiview"]) > 0 for row in rows), (
        "P07: all-multiview seed contains a feature row without an eligible event"
    )
    marker = _marker(out)
    assert marker.get("rows_zero_values") == 0, "P07: populated fixture has zero-value rows"
    assert marker.get("agreement_reasons", {}).get("agreement_no_multiview") == 0, (
        "P07: all-multiview fixture has rows without an eligible event"
    )
    assert all(row["view_abs_diff"] for row in rows), (
        "P07: a five-subject all-multiview row published no agreement"
    )


def test_p08_nonfinite_tokens_are_filtered_from_values_and_statistics(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P08")
    root = tmp_path / "mixed"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    feature = next(row for row in _feature_rows(cohort) if row["level"] == "window")
    _rewrite_feature(inputs, "window", feature["column"], {0: "NA", 1: "NaN", 2: "Inf", 3: "-Inf"})
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    expected = _independent_stats(inputs, "window", feature["column"])
    for cell, oracle in expected.items():
        actual = _feature_output(out, (*cell, "window", feature["column"]))
        for field in ("n_subjects", "n_events", "n_assets", "n_values", "n_events_multiview"):
            assert actual[field] == _render_cell(oracle[field]), (
                f"P08: nonfinite row changed finite-contributor {field}"
            )
        # A23: dropping subject 1's only finite value takes three cells to four subjects,
        # where A10's floor suppresses every distribution. The oracle models no floor, so
        # below it the published statement is emptiness — the floor outranks the oracle.
        subjects = oracle["n_subjects"]
        suppressed = subjects is None or subjects < _SUBJECT_FLOOR
        for field in _DISTRIBUTION_FIELDS:
            assert actual[field] == ("" if suppressed else _render_cell(oracle[field])), (
                f"P08/N7: nonfinite row entered {field}"
            )


def test_p08_cell_empty_feature_stays_in_the_run_level_product(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P08")
    root = tmp_path / "empty-cell"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    feature = next(row for row in _feature_rows(cohort) if row["level"] == "window")
    target = inventory.TASKS[0], inventory.SIDES[0]
    replacements = {
        asset.asset_id: "NA" for asset in inputs.assets if (asset.task, asset.side) == target
    }
    assert replacements, "P08/A18: empty-cell seed has no target assets"
    _rewrite_asset_feature(inputs, "window", feature["column"], replacements)
    measured_published, _ = _measured_partition(inputs)
    assert ("window", feature["column"]) in measured_published, (
        "P08/A18: feature became run-global empty instead of one-cell empty"
    )
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    _, rows = _read_csv(out / "cohort_features.csv")
    feature_rows = [
        row for row in rows if row["level"] == "window" and row["feature"] == feature["column"]
    ]
    expected_cells = {(task, side) for task in inventory.TASKS for side in inventory.SIDES}
    assert {(row["task"], row["side"]) for row in feature_rows} == expected_cells, (
        "P08/A18: cell-empty published feature was dropped from the product"
    )
    empty = _feature_output(out, (*target, "window", feature["column"]))
    assert all(
        empty[field] == "0"
        for field in ("n_subjects", "n_events", "n_assets", "n_values", "n_events_multiview")
    ), "P08: cell-empty feature retained a finite-contributor count"
    assert all(empty[field] == "" for field in _DISTRIBUTION_FIELDS), (
        "P08: cell-empty feature published a distribution statistic"
    )
    control = next(row for row in feature_rows if (row["task"], row["side"]) != target)
    assert int(control["n_values"]) > 0, "P08/A18: run-level presence control is empty"


# P09-P12: descriptors, units, redaction, and upstream immutability.


def test_p09_feature_ranges_are_measurement_domains_not_corpus_extremes() -> None:
    cohort = _cohort("P09")
    rows = _feature_rows(cohort)
    keys = {(row["level"], row["column"]) for row in rows}
    assert keys, "P09/A21: feature range oracle is vacuous"
    assert keys == set(cohort.FEATURE_KEYS), "P09: FEATURE_KEYS diverges from FEATURES"
    for row in rows:
        value_range = row["range"]
        assert isinstance(value_range, (list, tuple)), (
            f"P09: {row['column']} range is not a sequence"
        )
        assert len(value_range) == 2, f"P09: {row['column']} range is not two-bound"
        assert tuple(value_range) == _expected_range(row["column"]), (
            f"P09/A21: {row['column']} range is not its R-defined admissible domain"
        )


def test_p09_descriptor_fragment_is_the_bijective_feature_table_projection(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P09")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    document = yaml.safe_load((out / "descriptors.yaml").read_text(encoding="utf-8"))
    assert isinstance(document, dict), "P09: descriptor fragment is not a YAML mapping"
    assert set(document) == {"columns"}, "P09: descriptor root must contain only columns"
    descriptors = document["columns"]
    assert isinstance(descriptors, list), "P09: descriptor fragment has no columns list"
    feature_rows = _feature_rows(cohort)
    assert len(descriptors) == len(feature_rows), "P09: descriptor count differs from len(FEATURES)"
    required = {"raw", "ja", "en", "group", "role", "dtype", "unit", "range"}
    assert all(isinstance(row, dict) and set(row) == required for row in descriptors), (
        "P09: descriptor key set drifted"
    )
    expected = {
        f"pose_{row['level']}_{row['column']}": {
            "raw": f"pose_{row['level']}_{row['column']}",
            "ja": row["ja"],
            "en": row["en"],
            "group": "pose",
            "role": "feature",
            "dtype": "numeric",
            "unit": row["unit"],
            "range": list(row["range"]),
        }
        for row in feature_rows
    }
    actual = {row["raw"]: row for row in descriptors}
    assert len(actual) == len(descriptors), "P09/N8: descriptor raw names collide"
    assert actual == expected, "P09: descriptors were not derived exactly from FEATURES"
    assert all(row["ja"] and row["en"] for row in descriptors), "P09/N8: blank label published"
    assert all(any(ord(character) > 127 for character in row["ja"]) for row in descriptors), (
        "P09: Japanese label has no non-ASCII character"
    )

    collision = _marker(out).get("descriptor_collision")
    assert isinstance(collision, dict), "P09: descriptor collision outcome is absent"
    assert set(collision) == {"checked", "source", "source_sha256", "n_external", "n_collisions"}, (
        "P09/A22: descriptor collision field set drifted"
    )
    assert type(collision["checked"]) is bool, "P09: collision checked is not boolean"
    assert collision["source"] == _COLLISION_SOURCE, "P09/A22: collision source drifted"
    assert type(collision["n_external"]) is int, "P09: external count is not an integer"
    assert collision["n_external"] >= 0, "P09: external count is negative"
    assert type(collision["n_collisions"]) is int, "P09: collision count is not an integer"
    assert collision["n_collisions"] >= 0, "P09: collision count is negative"
    if _REHAB_SCHEMA.is_file():
        external = _external_descriptor_raws(_REHAB_SCHEMA)
        assert external, "P09: resolved external descriptor set is empty"
        assert not any(raw.startswith("pose_") for raw in external), (
            "P09: external control unexpectedly carries a pose_ raw"
        )
        assert collision["checked"] is True, "P09: resolved external schema was not checked"
        assert collision["n_external"] == len(external), (
            "P09: published external count differs from the expanded schema"
        )
        assert collision["n_collisions"] == 0, "P09: collision outcome is not clean"
        assert (
            collision["source_sha256"] == hashlib.sha256(_REHAB_SCHEMA.read_bytes()).hexdigest()
        ), "T4 R37: the census does not digest the schema it checked"
        assert not set(actual) & external, "P09/N8: descriptor raw collides with ../rehab"
    else:
        assert collision["checked"] is False, "P09: absent external schema reported checked"
        assert collision["source_sha256"] is None, "T4 R37: unchecked schema published a digest"
        assert collision["n_external"] == collision["n_collisions"] == 0, (
            "P09: unchecked external schema published nonzero counts"
        )


def test_p10_units_match_the_closed_measurement_vocabulary() -> None:
    cohort = _cohort("P10")
    public_constants = {name: getattr(cohort, name, None) for name in _UNIT_CONSTANTS}
    assert public_constants == _UNIT_CONSTANTS, "P10/A20: public unit token spelling drifted"
    assert cohort.UNIT_VOCABULARY == _UNIT_VOCABULARY, (
        "P10/A20: unit vocabulary is not the frozen seven-token set"
    )
    rows = _feature_rows(cohort)
    assert rows, "P10: unit predicate is vacuous"
    assert {row["unit"] for row in rows} == _UNIT_VOCABULARY, (
        "P10: published families do not exercise the closed vocabulary"
    )
    for row in rows:
        assert row["unit"] == _expected_unit(row["column"]), (
            f"P10/A20: {row['column']} carries the wrong measurement-family unit"
        )
    angle_rows = [
        row for row in rows if _expected_unit(row["column"]) == _UNIT_CONSTANTS["UNIT_DEG"]
    ]
    assert angle_rows, "P10: angle-unit predicate is vacuous"
    assert all(row["unit"] != "deg" for row in angle_rows), (
        "P10: image-plane angle carries unqualified deg"
    )


def test_p11_every_published_field_has_a_closed_typed_domain(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P11")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    _assert_typed_publication(cohort, out)


def test_p11_below_subject_floor_suppresses_only_distributions(
    tmp_path: pathlib.Path,
) -> None:
    cohort = _cohort("P11")
    root = tmp_path / "subject-floor"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    feature = next(row for row in _feature_rows(cohort) if row["level"] == "window")
    target = inventory.TASKS[0], inventory.SIDES[0]
    removed_subject = max(asset.subject_ordinal for asset in inputs.assets if asset.subject_ordinal)
    replacements = {
        asset.asset_id: "NA"
        for asset in inputs.assets
        if (asset.task, asset.side, asset.subject_ordinal) == (*target, removed_subject)
    }
    assert replacements, "P11: subject-floor seed removed no contributor"
    _rewrite_asset_feature(inputs, "window", feature["column"], replacements)
    out = tmp_path / "cohort"
    _publish(cohort, inputs, out)
    _assert_typed_publication(cohort, out)
    row = _feature_output(out, (*target, "window", feature["column"]))
    assert row["n_subjects"] == "4", "P11: subject-floor seed is not discriminating"
    assert all(int(row[field]) > 0 for field in ("n_events", "n_assets", "n_values")), (
        "P11: below-floor contributor counts were suppressed"
    )
    assert all(row[field] == "" for field in _DISTRIBUTION_FIELDS), (
        "P11: below-floor row exposed a subject distribution"
    )


def test_p11_no_input_identifier_or_path_reaches_any_published_byte(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P11")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    _, asset_rows = _read_csv(cohort_inputs.inventory / inventory.ASSETS_FILENAME)
    _, placement_rows = _read_csv(cohort_inputs.sessions / sessions.PLACEMENTS_FILENAME)
    needles = {
        *(row["asset_id"] for row in asset_rows),
        *(row["capture_id"] for row in asset_rows),
        *(row["source_path"] for row in asset_rows),
        *(pathlib.PurePosixPath(row["source_path"]).name for row in asset_rows),
        *(pathlib.PurePosixPath(row["source_path"]).suffix for row in asset_rows),
        *(row["event_id"] for row in placement_rows),
        *(row["camera_name"] for row in placement_rows),
    }
    needles.discard("")
    payload = b"\n".join(path.read_bytes() for path in sorted(out.iterdir()) if path.is_file())
    text = payload.decode("utf-8")
    leaked = sorted(needle for needle in needles if needle in text)
    assert not leaked, "P11/N9: published bytes carry an input identifier, filename, or path"


def test_p12_run_leaves_every_upstream_byte_and_link_unmoved(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P12")
    cohort_inputs = input_factory(cohort)
    before_validity = _validated_input_witnesses(cohort_inputs)
    before = {
        "inventory": _tree_state(cohort_inputs.inventory),
        "sessions": _tree_state(cohort_inputs.sessions),
        "run": _tree_state(cohort_inputs.run),
    }
    _publish(cohort, cohort_inputs, tmp_path / "cohort")
    after = {
        "inventory": _tree_state(cohort_inputs.inventory),
        "sessions": _tree_state(cohort_inputs.sessions),
        "run": _tree_state(cohort_inputs.run),
    }
    after_validity = _validated_input_witnesses(cohort_inputs)
    assert after == before, "P12: publisher changed an upstream byte, kind, or link text"
    assert after_validity == before_validity, (
        "P12: publisher moved a sessions tree or marker witness"
    )


def test_p12_n10_marker_byte_change_during_run_is_detected(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _cohort("P12")
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    tree_before, marker_before = _validated_input_witnesses(inputs)
    original = corpus_run.read_manifest
    mutated = False

    def mutate_then_read(path: pathlib.Path) -> list[dict[str, str]]:
        nonlocal mutated
        marker = inputs.sessions / sessions.GENERATION_FILENAME
        payload = json.loads(marker.read_text(encoding="utf-8"))
        marker.write_text(json.dumps(payload, sort_keys=True, indent=4) + "\n", encoding="utf-8")
        mutated = True
        return original(path)

    monkeypatch.setattr(corpus_run, "read_manifest", mutate_then_read)
    with pytest.raises(_cohort_error(cohort)):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")
    assert mutated, "P12/N10: mutation hook never reached the manifest read"
    sessions.validate_generation(inputs.sessions, inventory_dir=inputs.inventory)
    assert sessions.tree_digest(inputs.sessions) == tree_before, (
        "P12/N10: marker-only mutation unexpectedly moved the tree witness"
    )
    assert sessions.generation_digest(inputs.sessions) != marker_before, (
        "P12/N10: marker-byte witness did not see the reindent"
    )
    assert not (tmp_path / "cohort").exists(), "P12/N10: changed upstream still published"


# P13-P16: marker trust, ownership/atomicity, determinism, and idempotence.


def test_p13_marker_is_regular_and_duplicate_keys_are_rejected(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P13")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    marker = out / "cohort.json"
    assert stat.S_ISREG(marker.lstat().st_mode), "P13: marker is not a regular file"
    original = marker.read_text(encoding="utf-8")
    marker.write_text('{"population":null,' + original.lstrip()[1:], encoding="utf-8")
    with pytest.raises(_cohort_error(cohort)):
        cohort.validate_generation(out)


def test_p13_symlinked_marker_is_rejected(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P13")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    marker = out / "cohort.json"
    external = tmp_path / "external.json"
    marker.rename(external)
    marker.symlink_to(external)
    with pytest.raises(_cohort_error(cohort)):
        cohort.validate_generation(out)


def test_p13_generation_block_and_marker_rendering_are_canonical(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P13")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    marker_path = out / "cohort.json"
    marker = _marker(out)
    assert marker_path.read_bytes() == _canonical_json(marker), (
        "P13: marker is not canonical sorted two-space JSON with one terminal newline"
    )
    generation = marker.get("generation")
    assert isinstance(generation, dict), "P13: generation block is absent"
    assert set(generation) == _GENERATION_KEYS, "P13: generation field set drifted"
    assert generation["generator"] == cohort.GENERATOR, "P13: generator name drifted"
    assert generation["generator_version"] == cohort.GENERATOR_VERSION, (
        "P13: generator version drifted"
    )
    assert _is_digest(generation["tree_digest"]), "P13: tree digest is not lowercase SHA-256"
    input_digests = generation["input_digests"]
    assert isinstance(input_digests, dict), "P13/A22: input digests are not a mapping"
    assert set(input_digests) == _INPUT_DIGEST_KEYS, "P13/A22: input digest field set drifted"
    for key, value in input_digests.items():
        assert _is_digest(value), f"P13: input digest {key!r} is not lowercase SHA-256"


def test_p13_tree_digest_covers_generation_provenance(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P13")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    marker_path = out / "cohort.json"
    marker = _marker(out)
    generation = marker.get("generation")
    assert isinstance(generation, dict), "P13: generation block is absent"
    digests = generation.get("input_digests")
    assert isinstance(digests, dict), "P13: provenance digest set is not a mapping"
    assert digests, "P13: provenance digest set is vacuous"
    key = next(iter(digests))
    replacement = "0" * 64 if digests[key] != "0" * 64 else "1" * 64
    digests[key] = replacement
    marker_path.write_bytes(_canonical_json(marker))
    with pytest.raises(_cohort_error(cohort)):
        cohort.validate_generation(out)


@pytest.mark.parametrize("input_name", ["inventory", "sessions", "run"])
def test_p14_output_may_not_sit_inside_any_input(
    input_factory: _InputFactory, input_name: str
) -> None:
    cohort = _cohort("P14")
    cohort_inputs = input_factory(cohort)
    source = getattr(cohort_inputs, input_name)
    before = _tree_state(source)
    with pytest.raises(_cohort_error(cohort)):
        cohort.run(
            cohort_inputs.inventory,
            cohort_inputs.sessions,
            cohort_inputs.run,
            source / "nested-output",
        )
    assert _tree_state(source) == before, "P14/N12: overlap refusal changed its input"


@pytest.mark.parametrize("input_name", ["inventory", "sessions", "run"])
def test_p14_output_may_not_contain_any_input(
    input_factory: _InputFactory, tmp_path: pathlib.Path, input_name: str
) -> None:
    cohort = _cohort("P14")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "owned-container"
    _publish(cohort, cohort_inputs, out)
    embedded = out / "embedded-input"
    shutil.copytree(getattr(cohort_inputs, input_name), embedded, symlinks=True)
    paths = {
        "inventory": cohort_inputs.inventory,
        "sessions": cohort_inputs.sessions,
        "run": cohort_inputs.run,
    }
    paths[input_name] = embedded
    before = _tree_state(out)
    with pytest.raises(_cohort_error(cohort)):
        cohort.run(paths["inventory"], paths["sessions"], paths["run"], out)
    assert _tree_state(out) == before, "P14/N12: containing output changed or deleted its input"


def test_p14_unmarked_nonempty_destination_is_never_replaced(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P14")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "foreign"
    out.mkdir()
    (out / "foreign.txt").write_text("keep\n", encoding="utf-8")
    before = _tree_state(out)
    with pytest.raises(_cohort_error(cohort)):
        cohort.run(cohort_inputs.inventory, cohort_inputs.sessions, cohort_inputs.run, out)
    assert _tree_state(out) == before, "P14/N12: unmarked foreign tree was replaced"


@pytest.mark.parametrize(
    ("constant", "marker_field"),
    [("GENERATOR", "generator"), ("GENERATOR_VERSION", "generator_version")],
)
def test_p14_each_marker_identity_component_controls_ownership(
    input_factory: _InputFactory,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    marker_field: str,
) -> None:
    cohort = _cohort("P14")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / f"foreign-{marker_field}"
    owned_value = getattr(cohort, constant)
    foreign_value = f"{owned_value}-foreign"
    monkeypatch.setattr(cohort, constant, foreign_value)
    _publish(cohort, cohort_inputs, out)
    assert _marker(out)["generation"][marker_field] == foreign_value, (
        f"P14/A14: {marker_field} mismatch seed was not published validly"
    )
    monkeypatch.setattr(cohort, constant, owned_value)
    debris = out.with_name(f"{out.name}.staging.foreign")
    debris.mkdir()
    (debris / "sentinel").write_text("keep\n", encoding="utf-8")
    before_out = _tree_state(out)
    before_debris = _tree_state(debris)
    with pytest.raises(_cohort_error(cohort)):
        cohort.run(cohort_inputs.inventory, cohort_inputs.sessions, cohort_inputs.run, out)
    assert _tree_state(out) == before_out, f"P14/A14: wrong {marker_field} tree was replaced"
    assert _tree_state(debris) == before_debris, (
        f"P14/A14: wrong {marker_field} owner swept sibling debris"
    )


@pytest.mark.parametrize("marker_kind", ["directory", "symlink"])
def test_p14_ownership_marker_must_be_regular_and_nonsymlink(
    input_factory: _InputFactory, tmp_path: pathlib.Path, marker_kind: str
) -> None:
    cohort = _cohort("P14")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / f"foreign-{marker_kind}"
    out.mkdir()
    (out / "sentinel").write_text("keep\n", encoding="utf-8")
    marker = out / "cohort.json"
    if marker_kind == "directory":
        marker.mkdir()
    else:
        external = tmp_path / "external-owned-marker.json"
        external.write_text(
            json.dumps(
                {
                    "generation": {
                        "generator": cohort.GENERATOR,
                        "generator_version": cohort.GENERATOR_VERSION,
                    }
                }
            ),
            encoding="utf-8",
        )
        marker.symlink_to(external)
    before = _tree_state(out)
    with pytest.raises(_cohort_error(cohort)):
        cohort.run(cohort_inputs.inventory, cohort_inputs.sessions, cohort_inputs.run, out)
    assert _tree_state(out) == before, f"P14/A14: {marker_kind} ownership marker was followed"


def test_p14_failed_sibling_promotion_restores_the_previous_generation(
    input_factory: _InputFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _cohort("P14")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    before = _tree_state(out)
    original_rename = os.rename
    original_replace = os.replace
    promotion_attempted = False

    def fail_first_promotion(operation: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal promotion_attempted
            source_path = pathlib.Path(source)
            target_path = pathlib.Path(target)
            if (
                not promotion_attempted
                and source_path.parent == out.parent
                and source_path != out
                and target_path == out
            ):
                promotion_attempted = True
                raise OSError("injected sibling promotion failure")
            return operation(source, target, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(os, "rename", fail_first_promotion(original_rename))
    monkeypatch.setattr(os, "replace", fail_first_promotion(original_replace))
    with pytest.raises(OSError, match="injected sibling promotion failure"):
        cohort.run(cohort_inputs.inventory, cohort_inputs.sessions, cohort_inputs.run, out)
    assert promotion_attempted, "P14: publisher never promoted a staging sibling"
    assert _tree_state(out) == before, "P14: failed swap did not restore the old generation"


def test_p14_debris_is_not_swept_before_a_successful_swap(tmp_path: pathlib.Path) -> None:
    cohort = _cohort("P14")
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    out = tmp_path / "cohort"
    debris = [
        out.with_name(f"{out.name}.staging.stale"),
        out.with_name(f"{out.name}.retiring.stale"),
    ]
    for path in debris:
        path.mkdir()
        (path / "sentinel").write_text("keep\n", encoding="utf-8")
    before = {path: _tree_state(path) for path in debris}
    (inputs.run / corpus_run.MANIFEST_FILENAME).write_text(
        ",".join(corpus_run.MANIFEST_FIELDS) + "\n", encoding="utf-8"
    )
    with pytest.raises(_cohort_error(cohort)):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    assert all(_tree_state(path) == before[path] for path in debris), (
        "P14: crash debris was swept before a swap landed"
    )


def test_p14_successful_swap_preserves_every_sibling_it_does_not_own(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    """A30: a prefix-named sibling is somebody else's directory until a marker says otherwise.

    The retired sweep read the name alone, so a foreign tree and a complete generation
    published under a staging-shaped `--out` were both deleted by an unrelated run.
    """
    cohort = _cohort("P14")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    debris = [
        out.with_name(f"{out.name}.staging.stale"),
        out.with_name(f"{out.name}.retiring.stale"),
    ]
    for path in debris:
        path.mkdir()
        (path / "sentinel").write_text("keep\n", encoding="utf-8")
    before = {path: _tree_state(path) for path in debris}
    _publish(cohort, cohort_inputs, out)
    assert all(_tree_state(path) == before[path] for path in debris), (
        "P14/A30: a successful swap deleted a sibling this generator does not own"
    )
    assert not [
        entry
        for entry in out.parent.iterdir()
        if entry.name.startswith((f"{out.name}.staging.", f"{out.name}.retiring."))
        and entry not in debris
    ], "P14: the run left its own staging or retiring path behind"


def test_p15_tree_bytes_are_environment_and_output_name_independent(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P15")
    cohort_inputs = input_factory(cohort)
    script = """
import os, sys
from pose_estimation import cohort
os.umask(int(sys.argv[5], 8))
cohort.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
cohort.validate_generation(sys.argv[4])
"""
    sweeps = (
        ("1", "C", "UTC", "022", False),
        ("29", "C.UTF-8", "Asia/Tokyo", "077", False),
        ("101", "en_US.UTF-8", "UTC", "002", True),
        ("997", None, "Asia/Tokyo", "027", True),
    )
    generations: list[dict[str, tuple[str, bytes]]] = []
    for index, (seed, locale, timezone, mask, optimized) in enumerate(sweeps):
        out = tmp_path / f"variant-{index}"
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED": seed, "TZ": timezone})
        env.pop("LC_ALL", None)
        env.pop("LANG", None)
        if locale is not None:
            env["LC_ALL"] = locale
        command = [sys.executable]
        if optimized:
            command.append("-O")
        command.extend(
            [
                "-c",
                script,
                str(cohort_inputs.inventory),
                str(cohort_inputs.sessions),
                str(cohort_inputs.run),
                str(out),
                mask,
            ]
        )
        result = subprocess.run(
            command, cwd=_ROOT, env=env, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"P15: sweep {index} failed: {result.stderr}"
        generations.append(_tree_state(out))
    assert all(generation == generations[0] for generation in generations[1:]), (
        "P15: output depends on hash seed, locale, timezone, umask, -O, or --out"
    )


def test_p15_descriptor_order_is_level_then_column_not_label_order(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P15")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    document = yaml.safe_load((out / "descriptors.yaml").read_text(encoding="utf-8"))
    descriptors = document["columns"]
    feature_rows = _feature_rows(cohort)
    expected = [
        f"pose_{row['level']}_{row['column']}"
        for row in sorted(feature_rows, key=lambda row: (row["level"], row["column"]))
    ]
    label_orders = {
        tuple(
            f"pose_{row['level']}_{row['column']}"
            for row in sorted(
                feature_rows, key=lambda row: (row[label], row["level"], row["column"])
            )
        )
        for label in ("ja", "en")
    }
    assert all(tuple(expected) != order for order in label_orders), (
        "P15/N13: label-order controls do not differ from canonical order"
    )
    assert [row["raw"] for row in descriptors] == expected, (
        "P15/N13: descriptor order is not canonical (level, column)"
    )


def test_p16_republishing_owned_tree_is_exactly_byte_idempotent(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P16")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    before = _tree_state(out)
    _publish(cohort, cohort_inputs, out)
    assert _tree_state(out) == before, "P16: repeat publication moved a byte, kind, or path"


# P17-P18: consumer refusal matrix and durable registration.


@pytest.mark.parametrize(
    "name", ["cohort_cells.csv", "cohort_features.csv", "descriptors.yaml", "cohort.json"]
)
def test_p17_each_missing_publication_file_raises_exactly_cohort_error(
    input_factory: _InputFactory, tmp_path: pathlib.Path, name: str
) -> None:
    cohort = _cohort("P17")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    (out / name).unlink()
    error_type = _cohort_error(cohort)
    with pytest.raises(error_type) as caught:
        cohort.validate_generation(out)
    assert type(caught.value) is error_type, f"P17/A16: missing {name} raised a subclass"


@pytest.mark.parametrize("name", ["cohort_cells.csv", "cohort_features.csv", "descriptors.yaml"])
def test_p17_each_semantics_preserving_payload_edit_raises_exactly_cohort_error(
    input_factory: _InputFactory, tmp_path: pathlib.Path, name: str
) -> None:
    cohort = _cohort("P17")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    path = out / name
    path.write_bytes(path.read_bytes() + b"\n")
    error_type = _cohort_error(cohort)
    with pytest.raises(error_type) as caught:
        cohort.validate_generation(out)
    assert type(caught.value) is error_type, f"P17/A16: edited {name} raised a subclass"


@pytest.mark.parametrize(
    "delta", [pytest.param(-1, id="decrement"), pytest.param(1, id="increment")]
)
def test_p17_both_census_edit_directions_raise_exactly_cohort_error(
    input_factory: _InputFactory, tmp_path: pathlib.Path, delta: int
) -> None:
    cohort = _cohort("P17")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort"
    _publish(cohort, cohort_inputs, out)
    marker_path = out / "cohort.json"
    marker = _marker(out)
    population = marker.get("population")
    assert isinstance(population, dict), "P17/A22: population census is not a mapping"
    candidates = sorted(
        key
        for key, value in population.items()
        if isinstance(key, str) and type(value) is int and value > 0
    )
    assert candidates, "P17/A22: census edit has no positive unnamed population field"
    key = candidates[0]
    population[key] += delta
    assert population[key] >= 0, "P17: decrement control left the census domain"
    marker_path.write_bytes(_canonical_json(marker))
    error_type = _cohort_error(cohort)
    with pytest.raises(error_type) as caught:
        cohort.validate_generation(out)
    assert type(caught.value) is error_type, "P17/A16: edited census raised a subclass"


def test_p18_technical_document_states_all_four_claim_boundaries() -> None:
    cohort = _cohort("P18")
    path = _ROOT / "docs" / "technical" / "cohort.md"
    assert path.is_file(), "P18/A17: docs/technical/cohort.md is absent"
    lowered = " ".join(path.read_text(encoding="utf-8").lower().split())
    for phrase in (
        "asset median",
        "event median",
        "subject median",
        "view_abs_diff",
        "view_icc",
        "construct_invalid",
        "projection geometry",
        "directional per-limb",
        "structurally_absent",
        "sagittal",
        "out of plane",
        "one scalar",
        "anatomical angle",
    ):
        assert phrase in lowered, f"P18/A17: cohort.md omits {phrase!r}"
    excluded = sorted(_source_columns() - set(cohort.FEATURE_KEYS) - _CONSTRUCT_INVALID)
    assert excluded, "P18/A02: derived exclusion census is empty"
    assert re.search(rf"(?<!\d){len(excluded)}(?!\d)", lowered), (
        "P18/A17: cohort.md omits the derived exclusion count"
    )
    for _level, column in excluded:
        assert column in lowered, f"P18/A17: cohort.md omits excluded column {column!r}"


def test_p18_every_exhaustive_index_names_the_publisher() -> None:
    _cohort("P18")
    technical = _ROOT / "docs" / "technical"
    entrypoints = (technical / "entrypoints.md").read_text(encoding="utf-8")
    architecture = (technical / "architecture.md").read_text(encoding="utf-8")
    tests = (technical / "tests.md").read_text(encoding="utf-8")
    conventions = (technical / "conventions.md").read_text(encoding="utf-8")
    assert "Eleven console scripts" in entrypoints, "P18: entry-point count was not reconciled"
    assert "`pose-estimation-cohort`" in entrypoints, "P18: entry-point index omits the CLI"
    assert "`cohort.py`" in architecture, "P18: architecture module map omits cohort.py"
    assert "`tests/test_cohort.py`" in tests, "P18: test inventory omits the suite"
    assert "check_cohort_determinism.py" in conventions, (
        "P18/A17: campaign list omits the cohort determinism artifact"
    )


def test_p18_determinism_campaign_artifact_is_regular_and_registered() -> None:
    _cohort("P18")
    campaign = _ROOT / "scripts" / "check_cohort_determinism.py"
    assert campaign.exists(), "P18/A17: cohort determinism campaign is absent"
    assert stat.S_ISREG(campaign.lstat().st_mode), "P18/A17: campaign is not a regular file"
    assert not campaign.is_symlink(), "P18/A17: campaign registration resolves through a symlink"
    compile(campaign.read_text(encoding="utf-8"), str(campaign), "exec")


def test_p18_cli_smoke_accepts_the_frozen_four_flags(
    input_factory: _InputFactory, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort("P18")
    cohort_inputs = input_factory(cohort)
    out = tmp_path / "cohort-cli"
    main = getattr(cohort, "main", None)
    assert callable(main), "P18/A17: cohort.main is absent"
    result = main(
        [
            "--inventory",
            str(cohort_inputs.inventory),
            "--sessions",
            str(cohort_inputs.sessions),
            "--run",
            str(cohort_inputs.run),
            "--out",
            str(out),
        ]
    )
    assert result == 0, "P18/A17: CLI returned nonzero for the frozen flag set"
    cohort.validate_generation(out)


def test_p18_cli_and_generated_tree_are_registered() -> None:
    _cohort("P18")
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    table = re.search(r"(?ms)^\[project\.scripts\]\s*$\n(.*?)(?=^\[|\Z)", pyproject)
    assert table is not None, "P18/A17: project.scripts is absent"
    rows = [line for line in table.group(1).splitlines() if line.strip()]
    pairs = [line.split("=", 1) for line in rows]
    assert all(len(pair) == 2 for pair in pairs), "P18/A17: malformed console-script row"
    scripts = {key.strip(): json.loads(value.strip()) for key, value in pairs}
    assert len(scripts) == len(rows), "P18/A17: duplicate console-script key"
    assert len(scripts) == 11, "P18/A17: console-script count did not move from ten to eleven"
    assert scripts.get("pose-estimation-cohort") == "pose_estimation.cohort:main", (
        "P18/A17: packaged cohort command is absent or misdirected"
    )
    result = subprocess.run(
        ["git", "check-ignore", "cohort", "cohort.staging.1/", "cohort.retiring.1/"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, "P18: generated cohort trees are not all ignored"
    assert set(result.stdout.splitlines()) == {
        "cohort",
        "cohort.staging.1/",
        "cohort.retiring.1/",
    }, "P18: cohort ignore pattern is incomplete"
