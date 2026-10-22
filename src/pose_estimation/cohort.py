"""Cohort aggregate publisher — `(task, side)` statistics over the 2D corpus run.

Contract `.agent/archive/contract-m2u83.md`. This module carries the publisher's
contract-owned half: the domain exception, the label table and the rules that assign a
unit and a range to every published feature. The data-owned half — which columns are
published at all — is measured from the run at publish time (A02), and a disagreement
between the two is a refusal rather than a silent reconciliation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import re
import shutil
import stat
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Any

import yaml

from . import corpus_run, inventory, qualify, sessions, view_agreement

GENERATOR = "pose-estimation-cohort"
GENERATOR_VERSION = "v3"


class CohortError(Exception):
    """Every publisher and consumer refusal (A16). P17 asserts this class, not a supertype."""


# A09 + A20 + A25, + M4.5b `s` and `count`. Closed: P10 is a membership test, so growing the
# vocabulary weakens it. No token spells a bare `deg`: `export.COORD_NORMALIZATION` is a
# similarity map, so the published value IS the true image-plane angle, and it is still
# neither anatomical nor lens-distortion corrected.
UNIT_DEG = "deg_image_plane"
UNIT_FRAME_NORMALIZED = "frame_normalized"
UNIT_FRAME_NORMALIZED_PER_S = "frame_normalized_per_s"
UNIT_RATIO_SHOULDER_WIDTH = "ratio_shoulder_width"
UNIT_RATIO = "ratio"
UNIT_INDEX_SIGNED = "index_signed"
UNIT_DIMENSIONLESS = "dimensionless"
UNIT_SECONDS = "s"
UNIT_COUNT = "count"

UNIT_VOCABULARY = frozenset(
    {
        UNIT_DEG,
        UNIT_FRAME_NORMALIZED,
        UNIT_FRAME_NORMALIZED_PER_S,
        UNIT_RATIO_SHOULDER_WIDTH,
        UNIT_RATIO,
        UNIT_INDEX_SIGNED,
        UNIT_DIMENSIONLESS,
        UNIT_SECONDS,
        UNIT_COUNT,
    }
)

LEVELS = ("frame", "window", "bout")
EXCLUSION_REASONS = frozenset(
    {"structurally_absent", "construct_invalid", "no_eligible_bout_values"}
)

# Bounds are the measurement's admissible domain read from `analysis/clinical_features.R`
# (A21), never a corpus extreme: D05 refused min/max because at n=15-16 an extreme is one
# identifiable subject, and a descriptor is schema that must not move when the data does.
# `None` is an open end. Sources: symmetry ratio and dominance index :322-324, SAL :366,
# normalized jerk :402, movement efficiency :430, unsigned trunk lean :753, signed lateral
# lean :773, wrapped trunk rotation :795, posture symmetry :812-816.
_FAMILIES: dict[str, tuple[str, str, str, float | None, float | None]] = {
    "elbow_angle_deg": ("肘角度", "Elbow angle", UNIT_DEG, 0.0, 180.0),
    "wrist_deviation_deg": ("手関節偏位角度", "Wrist deviation angle", UNIT_DEG, 0.0, 180.0),
    "finger_spread_deg": ("手指開扇角度", "Finger spread angle", UNIT_DEG, 0.0, 180.0),
    "reach_raw": ("リーチ距離", "Reach distance", UNIT_FRAME_NORMALIZED, 0.0, None),
    "reach_norm": (
        "正規化リーチ距離",
        "Normalized reach distance",
        UNIT_RATIO_SHOULDER_WIDTH,
        0.0,
        None,
    ),
    "grasp_aperture_thumb_index": (
        "母指-示指把持間隔",
        "Thumb-index grasp aperture",
        UNIT_FRAME_NORMALIZED,
        0.0,
        None,
    ),
    "grasp_aperture_thumb_pinky": (
        "母指-小指把持間隔",
        "Thumb-little finger grasp aperture",
        UNIT_FRAME_NORMALIZED,
        0.0,
        None,
    ),
    "wrist_displacement": ("手関節変位", "Wrist displacement", UNIT_FRAME_NORMALIZED, 0.0, None),
    "fingertip_displacement": (
        "指尖変位",
        "Fingertip displacement",
        UNIT_FRAME_NORMALIZED,
        0.0,
        None,
    ),
    "wrist_sal": (
        "手関節スペクトルアーク長(SPARC)",
        "Wrist spectral arc length (SPARC)",
        UNIT_DIMENSIONLESS,
        None,
        0.0,
    ),
    "wrist_velocity_mean": (
        "手関節平均速度",
        "Wrist mean velocity",
        UNIT_FRAME_NORMALIZED_PER_S,
        0.0,
        None,
    ),
    "wrist_velocity_peak": (
        "手関節最大速度",
        "Wrist peak velocity",
        UNIT_FRAME_NORMALIZED_PER_S,
        0.0,
        None,
    ),
    "wrist_normalized_jerk": (
        "手関節正規化ジャーク",
        "Wrist normalized jerk",
        UNIT_DIMENSIONLESS,
        0.0,
        None,
    ),
    "wrist_movement_efficiency": (
        "手関節運動効率",
        "Wrist movement efficiency",
        UNIT_RATIO,
        1.0,
        None,
    ),
    "fingertip_normalized_jerk": (
        "指尖正規化ジャーク",
        "Fingertip normalized jerk",
        UNIT_DIMENSIONLESS,
        0.0,
        None,
    ),
    "trunk_lean": ("体幹傾斜角度", "Trunk lean angle", UNIT_DEG, 0.0, 90.0),
    "trunk_lean_lateral": (
        "体幹側方傾斜角度",
        "Trunk lateral lean angle",
        UNIT_DEG,
        -180.0,
        180.0,
    ),
    "trunk_rotation": ("体幹回旋角度", "Trunk rotation angle", UNIT_DEG, -180.0, 180.0),
    "posture_symmetry": ("姿勢対称性", "Posture symmetry", UNIT_RATIO_SHOULDER_WIDTH, -1.0, 1.0),
    "compensatory_pattern_index": (
        "代償パターン指数",
        "Compensatory pattern index",
        UNIT_INDEX_SIGNED,
        -1.0,
        1.0,
    ),
    # M4.5a bouts: ranges = contract-m4u5a §1 (finite values).
    "bout_duration": ("手関節運動区間の持続時間", "Wrist bout duration", UNIT_SECONDS, 0.0, None),
    "bout_sparc": (
        "手関節運動区間のスペクトルアーク長(SPARC)",
        "Wrist bout spectral arc length (SPARC)",
        UNIT_DIMENSIONLESS,
        None,
        0.0,
    ),
    "bout_normalized_jerk": (
        "手関節運動区間の正規化ジャーク",
        "Wrist bout normalized jerk",
        UNIT_DIMENSIONLESS,
        0.0,
        None,
    ),
    "bout_peak_time_share": (
        "手関節運動区間の最大速度到達時間比",
        "Wrist bout time-to-peak-speed share",
        UNIT_RATIO,
        0.0,
        1.0,
    ),
    "bout_speed_peaks": (
        "手関節運動区間の速度ピーク数",
        "Wrist bout speed peak count",
        UNIT_COUNT,
        1.0,
        None,
    ),
    "bout_path_ratio": ("手関節運動区間の経路比", "Wrist bout path ratio", UNIT_RATIO, 1.0, None),
    "bout_dwell": ("手関節運動区間間の停止時間", "Wrist inter-bout dwell", UNIT_SECONDS, 0.0, None),
}

# The R source spells the three trunk angles with a `_deg` suffix at frame level and
# without one in its window aggregates, so one family answers to both names.
_FAMILY_ALIASES = {
    "trunk_lean_deg": "trunk_lean",
    "trunk_lean_lateral_deg": "trunk_lean_lateral",
    "trunk_rotation_deg": "trunk_rotation",
}

_SIDES = {"left_": ("左", "Left"), "right_": ("右", "Right")}

# Suffix -> (ja template, en template). A bilateral ratio or index replaces its family's
# unit; every other derivation inherits it (A09).
_DERIVATIONS: dict[str, tuple[str, str, str | None]] = {
    "_symmetry_ratio": ("{}の左右対称比", "{} left-right symmetry ratio", UNIT_RATIO),
    "_dominance_index": ("{}の左右優位指数", "{} left-right dominance index", UNIT_INDEX_SIGNED),
    "_abs_diff": ("{}の左右差", "{} left-right absolute difference", None),
    "_mean": ("{}の平均", "{} mean", None),
    "_sd": ("{}の標準偏差", "{} standard deviation", None),
    "_range": ("{}の範囲", "{} range", None),
}


@dataclass(frozen=True)
class Feature:
    """One published feature. A08 deleted the separate id: `(level, column)` is the key."""

    level: str
    column: str
    ja: str
    en: str
    unit: str
    range: tuple[float | None, float | None]

    @property
    def raw(self) -> str:
        """`../rehab` descriptor key. D10's prefix is what keeps it off all 219 existing raws."""
        return f"pose_{self.level}_{self.column}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _decompose(column: str) -> tuple[str, str | None, str | None]:
    """`column` -> (family, side prefix, derivation suffix). Raises on an unknown column."""
    side = next((prefix for prefix in _SIDES if column.startswith(prefix)), None)
    stem = column[len(side) :] if side else column
    resolved = _FAMILY_ALIASES.get(stem, stem)
    if resolved in _FAMILIES:
        return resolved, side, None
    # Family match precedes suffix stripping, or `wrist_velocity_mean` decomposes as a
    # window aggregate of a `wrist_velocity` family that does not exist.
    for suffix in _DERIVATIONS:
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        base = _FAMILY_ALIASES.get(base, base)
        if base in _FAMILIES:
            return base, side, suffix
    raise CohortError(f"no label family for column {column!r}")


def describe(level: str, column: str) -> Feature:
    """Build one feature's labels, unit and range by rule (A09, A20, A21)."""
    if level not in LEVELS:
        raise CohortError(f"unknown level {level!r}")
    family, side, derivation = _decompose(column)
    ja, en, unit, low, high = _FAMILIES[family]
    if side:
        side_ja, side_en = _SIDES[side]
        ja, en = f"{side_ja}{ja}", f"{side_en} {_lower_first(en)}"
    if derivation:
        ja_template, en_template, override = _DERIVATIONS[derivation]
        ja, en = ja_template.format(ja), en_template.format(en)
        if override is not None:
            unit = override
            low, high = (0.0, 1.0) if override == UNIT_RATIO else (-1.0, 1.0)
        elif derivation in ("_abs_diff", "_range"):
            # A magnitude over the family's own span: zero to its width when bounded.
            low, high = 0.0, (None if low is None or high is None else high - low)
        elif derivation == "_sd":
            low, high = 0.0, None
    if unit not in UNIT_VOCABULARY:
        raise CohortError(f"unit {unit!r} for {level}/{column} is outside the vocabulary")
    return Feature(level=level, column=column, ja=ja, en=en, unit=unit, range=(low, high))


# The published schema: contract-owned and therefore frozen (A10), while the published
# PARTITION is measured from the run (A02). A column the run stops emitting, or newly
# emits, fails the cross-check by name rather than silently moving the product.
_FRAME_BILATERAL = (
    "elbow_angle_deg",
    "wrist_deviation_deg",
    "finger_spread_deg",
    "reach_raw",
    "reach_norm",
    "grasp_aperture_thumb_index",
    "grasp_aperture_thumb_pinky",
    "wrist_displacement",
    "fingertip_displacement",
)
_WINDOW_BILATERAL = (
    "wrist_velocity_mean",
    "wrist_velocity_peak",
    "wrist_movement_efficiency",
)
# M4.5b: SPARC + normalized jerk are defined on one discrete movement, never on a 1 s window
# mixing movement and rest (Balasubramanian 2015) — the R stage still emits them for legacy
# consumers, the export lists them as excluded `construct_invalid` (bout level replaces them).
_WINDOW_SMOOTHNESS = ("wrist_sal", "wrist_normalized_jerk", "fingertip_normalized_jerk")
_BOUT_FEATURES = (
    "bout_duration",
    "bout_sparc",
    "bout_normalized_jerk",
    "bout_peak_time_share",
    "bout_speed_peaks",
    "bout_path_ratio",
    "bout_dwell",
)
_BILATERAL_SUFFIXES = ("_symmetry_ratio", "_dominance_index", "_abs_diff")

FRAME_COLUMNS: tuple[str, ...] = (
    *(f"{side}{base}" for side in ("left_", "right_") for base in _FRAME_BILATERAL),
    *(f"{base}{suffix}" for base in _FRAME_BILATERAL for suffix in _BILATERAL_SUFFIXES),
    "trunk_lean_deg",
    "trunk_lean_lateral_deg",
    "trunk_rotation_deg",
    "posture_symmetry",
)
WINDOW_COLUMNS: tuple[str, ...] = (
    *(f"{side}{base}" for side in ("left_", "right_") for base in _WINDOW_BILATERAL),
    *(f"{base}{suffix}" for base in _WINDOW_BILATERAL for suffix in _BILATERAL_SUFFIXES),
    "compensatory_pattern_index",
    "trunk_lean_mean",
    "trunk_lean_sd",
    "trunk_lean_range",
    "trunk_lean_lateral_mean",
    "trunk_lean_lateral_sd",
    "trunk_rotation_mean",
    "trunk_rotation_sd",
    "posture_symmetry_mean",
    "posture_symmetry_sd",
)

# `trunk_lean_sagittal_deg` and its `_mean`/`_sd` aggregates are absent by measurement:
# `clinical_features.R:1038` assigns NA_real_ on the 2D branch because sagittal lean is
# out of plane, so they carry zero finite values over the whole corpus and A02 sends them
# to `excluded`. They are not listed here, and P03 re-derives that partition from the run.
BOUT_COLUMNS: tuple[str, ...] = tuple(
    f"{side}{base}" for side in ("left_", "right_") for base in _BOUT_FEATURES
)
CONSTRUCT_INVALID: frozenset[tuple[str, str]] = frozenset(
    ("window", column)
    for column in (
        *(f"{side}{base}" for side in ("left_", "right_") for base in _WINDOW_SMOOTHNESS),
        *(f"{base}{suffix}" for base in _WINDOW_SMOOTHNESS for suffix in _BILATERAL_SUFFIXES),
    )
)
#: Emitted with zero finite values on every 2D run (out-of-plane), excluded by name.
STRUCTURALLY_ABSENT: frozenset[tuple[str, str]] = frozenset(
    {
        ("frame", "trunk_lean_sagittal_deg"),
        ("window", "trunk_lean_sagittal_mean"),
        ("window", "trunk_lean_sagittal_sd"),
    }
)

FEATURES: tuple[Feature, ...] = tuple(
    describe(level, column)
    for level, columns in (
        ("frame", FRAME_COLUMNS),
        ("window", WINDOW_COLUMNS),
        ("bout", BOUT_COLUMNS),
    )
    for column in sorted(columns)
)

FEATURE_KEYS: frozenset[tuple[str, str]] = frozenset(
    (feature.level, feature.column) for feature in FEATURES
)


CELLS_FILENAME = "cohort_cells.csv"
FEATURES_FILENAME = "cohort_features.csv"
DESCRIPTORS_FILENAME = "descriptors.yaml"
COHORT_FILENAME = "cohort.json"
PUBLISHED_FILENAMES: tuple[str, ...] = (CELLS_FILENAME, FEATURES_FILENAME, DESCRIPTORS_FILENAME)

CELL_COLUMNS: tuple[str, ...] = (
    "task",
    "side",
    "n_subjects",
    "n_events",
    "n_assets",
    "n_frame_rows",
    "n_window_rows",
    "n_bout_rows",
)
FEATURE_COLUMNS: tuple[str, ...] = (
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
COUNT_FIELDS: tuple[str, ...] = (
    "n_subjects",
    "n_events",
    "n_assets",
    "n_values",
    "n_events_multiview",
    "n_subjects_multiview",
)
DISTRIBUTION_FIELDS: tuple[str, ...] = ("median", "q25", "q75", "mean", "sd")
AGREEMENT_FIELDS: tuple[str, ...] = (
    "view_abs_diff",
    "view_icc",
    "view_icc_range_lo",
    "view_icc_range_hi",
)
MARKER_KEYS: tuple[str, ...] = (
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
)
GENERATION_KEYS: tuple[str, ...] = (
    "generator",
    "generator_version",
    "tree_digest",
    "input_digests",
)
MEASUREMENT_KEYS: tuple[str, ...] = ("pose_config", "pose_config_sha256", "analysis_sha256")

# contract-m4u7 §3: one published file beside the features, present exactly when a baseline is.
SENSITIVITY_FILENAME = "cohort_sensitivity.csv"
SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "task",
    "side",
    "level",
    "feature",
    "status",
    "n_subjects",
    "baseline_n_subjects",
    "n_values",
    "baseline_n_values",
    "median",
    "baseline_median",
    "delta_median",
    "baseline_iqr",
    "delta_iqr_scaled",
)
SENSITIVITY_STATUSES: tuple[str, ...] = ("common", "new", "retired")
SENSITIVITY_ESTIMAND = (
    "difference between the published cohort medians of two generations, not paired-subject "
    "change and not an accuracy result; subjects and retained observations can differ; "
    "delta_median = current - baseline; delta_iqr_scaled = delta_median / baseline published IQR; "
    "feature definitions may differ across generations"
)
# v1 + v2 published the three required files under the digest `tree_digest` still computes (D03).
BASELINE_VERSIONS = frozenset({"v1", "v2", GENERATOR_VERSION})
BASELINE_COLUMNS: tuple[str, ...] = (
    "task",
    "side",
    "level",
    "feature",
    "n_subjects",
    "n_values",
    "median",
    "q25",
    "q75",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FEATURE_NAME = re.compile(r"[a-z0-9_]+")
_COUNT = re.compile(r"[0-9]+")
_DECIMAL = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")

# D05 refused min/max at n=15-16 because an extreme is one identifiable subject; a median
# over fewer than five is strictly worse, so below the floor the counts publish and every
# distribution cell stays empty (A10).
SUBJECT_FLOOR = 5
DECIMALS = 9
ESTIMAND = {
    "distribution": "asset median -> event median -> subject median -> cohort statistic over subjects",
    "agreement": (
        "descriptive agreement of same-event asset medians across views (view_abs_diff = median "
        "over events of the mean pairwise |difference|; view_icc = one-way ICC(1,1), events as "
        "targets; range = 2.5-97.5 percentile over 1000 subject resamples, coverage not claimed)"
    ),
    "bout": "asset value = median over the asset's uncensored bouts of that side",
}
EXTERNAL_DESCRIPTOR_SOURCE = "../rehab/schema/columns.yaml"

# Producer keys, never features (`analysis/utils.R` treats every other numeric column as one).
METADATA_COLUMNS = frozenset(
    {
        "video",
        "frame_idx",
        "timestamp_sec",
        "person_idx",
        "window_start_sec",
        "window_end_sec",
        # Bout keys (contract-m4u5a BOUT_SCHEMA).
        "side",
        "bout_idx",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "censored",
    }
)
LEVEL_SUFFIXES: dict[str, str] = {
    "frame": "_clinical.csv",
    "window": "_clinical_windows.csv",
    "bout": "_clinical_bouts.csv",
}
BOUT_SIDES = {"left": "left_", "right": "right_"}
#: contract-m4u5a `BOUT_SCHEMA` keys — required even on a header-only file (the producer's seam).
BOUT_KEYS = (
    "video",
    "person_idx",
    "side",
    "bout_idx",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "censored",
)


@dataclass(frozen=True)
class _Contributor:
    """One manifest-`ok` canonical asset, resolved to its cell and its artifacts."""

    asset_id: str
    task: str
    side: str
    subject: str
    event_id: str
    camera_name: str


def _cell(value: float | int | Decimal | None) -> str:
    """Serialize one published cell. A19 rules 9 decimals; a count renders as a count."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{DECIMALS}f}"


def _finite(text: str) -> float | None:
    """`NA`, `NaN` and both infinities are absences, not values (P08)."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _quantile(values: list[float], fraction: float) -> float:
    """Linear interpolation at position `(n-1)q` — numpy `linear`, R type 7 (A03)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _assert_unique_header(name: str, header: tuple[str, ...]) -> None:
    """A repeated column makes `DictReader` keep the last cell and the census count twice.

    So the same source column reaches the feature table once and the published census
    twice, and every set-valued check stays green while the counts disagree (A27).
    """
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise CohortError(f"{name} repeats the columns {duplicates}.")


def _read_table(path: pathlib.Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = tuple(reader.fieldnames or ())
        _assert_unique_header(path.name, header)
        missing = [column for column in required if column not in header]
        if missing:
            raise CohortError(f"{path.name} is missing the columns {missing}.")
        return list(reader)


def _read_artifact(
    path: pathlib.Path, level: str
) -> tuple[tuple[str, ...], int, dict[str, list[float]]]:
    """Return one artifact's header, row count and finite values per feature column.

    Bout rows (contract-m4u5b §1): `side` + `censored` validated, a censored row contributes no
    value, and a finite value on the other side's columns refuses — the writer leaves them NA,
    so one is a producer defect the median would otherwise absorb.
    """
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            header = tuple(reader.fieldnames or ())
            _assert_unique_header(path.name, header)
            if level == "bout" and (missing := [k for k in BOUT_KEYS if k not in header]):
                raise CohortError(f"{path.name} lacks the bout keys {missing}.")
            columns = [name for name in header if name not in METADATA_COLUMNS]
            values: dict[str, list[float]] = {name: [] for name in columns}
            rows = 0
            for row in reader:
                rows += 1
                eligible = columns
                if level == "bout":
                    eligible = _bout_columns(path.name, row, columns)
                for name in eligible:
                    finite = _finite(row[name] or "")
                    if finite is not None:
                        values[name].append(finite)
    except OSError as error:
        raise CohortError(f"The run tree does not carry a readable {level} artifact.") from error
    return header, rows, values


def _bout_columns(name: str, row: Mapping[str, str], columns: list[str]) -> list[str]:
    side = row.get("side")
    if side not in BOUT_SIDES or row.get("censored") not in ("0", "1"):
        raise CohortError(f"{name} carries a bout row with an invalid side or censored flag.")
    start, end = _finite(row["start_sec"] or ""), _finite(row["end_sec"] or "")
    frames = (row["start_frame"] or "", row["end_frame"] or "")
    if (
        start is None
        or end is None
        or start > end
        or not all(f.isascii() and f.isdigit() for f in frames)
        or int(frames[0]) > int(frames[1])
    ):
        raise CohortError(f"{name} carries a bout row with invalid bounds.")
    own = [c for c in columns if c.startswith(BOUT_SIDES[side])]
    if any(_finite(row[c] or "") is not None for c in columns if c not in own):
        raise CohortError(f"{name} carries a finite value on the other side's bout columns.")
    return [] if row["censored"] == "1" else own


@dataclass(frozen=True)
class _Aggregate:
    cells: list[dict[str, str]]
    features: list[dict[str, str]]
    published: list[tuple[str, str]]
    excluded: list[tuple[str, str]]
    population: dict[str, int]
    rows_zero_values: int
    agreement_reasons: dict[str, int]
    rows_below_subject_floor: int
    exclusion_reasons: dict[tuple[str, str], str]


def _aggregate(contributors: list[_Contributor], run_root: pathlib.Path) -> _Aggregate:
    """Four-stage subject estimand over the `(task, side)` product (D02, A01, A03, A06)."""
    cells = [(task, side) for task in inventory.TASKS for side in inventory.SIDES]
    headers: dict[str, tuple[str, ...]] = {}
    row_counts: dict[tuple[str, str], dict[str, int]] = {
        cell: dict.fromkeys(LEVELS, 0) for cell in cells
    }
    members: dict[tuple[str, str], dict[str, set[str]]] = {
        cell: {"subjects": set(), "events": set(), "assets": set()} for cell in cells
    }
    # (level, column) -> cell -> event -> [asset medians]; and the finite leaf census.
    medians: dict[tuple[str, str], dict[tuple[str, str], dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    subjects_of_event: dict[str, str] = {}
    leaves: dict[tuple[str, str], dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for contributor in contributors:
        cell = contributor.task, contributor.side
        if cell not in row_counts:
            raise CohortError(f"Asset cell {cell} is outside the registry task-side product.")
        members[cell]["subjects"].add(contributor.subject)
        members[cell]["events"].add(contributor.event_id)
        members[cell]["assets"].add(contributor.asset_id)
        subjects_of_event[contributor.event_id] = contributor.subject
        for level, suffix in LEVEL_SUFFIXES.items():
            path = run_root / contributor.event_id / f"{contributor.camera_name}{suffix}"
            header, rows, values = _read_artifact(path, level)
            previous = headers.setdefault(level, header)
            if header != previous:
                raise CohortError(
                    f"The run tree publishes two different {level} headers, so the source "
                    "column census is not a function of the run."
                )
            row_counts[cell][level] += rows
            for column, finite in values.items():
                if not finite:
                    continue
                key = (level, column)
                medians[key][cell][contributor.event_id].append(statistics.median(finite))
                leaves[key][cell] += len(finite)
    if set(headers) != set(LEVEL_SUFFIXES):
        raise CohortError("The run tree carries no artifact for every published level.")

    source_columns = sorted(
        (level, column)
        for level, header in headers.items()
        for column in header
        if column not in METADATA_COLUMNS
    )
    reasons = _partition(source_columns, {key for key in source_columns if medians.get(key)})
    published = sorted(key for key in source_columns if key not in reasons)
    excluded = sorted(reasons)

    cell_rows = [
        {
            "task": task,
            "side": side,
            "n_subjects": str(len(members[cell]["subjects"])),
            "n_events": str(len(members[cell]["events"])),
            "n_assets": str(len(members[cell]["assets"])),
            "n_frame_rows": str(row_counts[cell]["frame"]),
            "n_window_rows": str(row_counts[cell]["window"]),
            "n_bout_rows": str(row_counts[cell]["bout"]),
        }
        for cell in cells
        for task, side in (cell,)
    ]

    feature_rows: list[dict[str, str]] = []
    zero_values = below_floor = 0
    agreement_reasons = dict.fromkeys(view_agreement.REASONS, 0)
    for feature in FEATURES:
        key = (feature.level, feature.column)
        if key in reasons:
            continue
        by_cell = medians.get(key, {})
        for task, side in cells:
            events = by_cell.get((task, side), {})
            by_subject: dict[str, list[float]] = defaultdict(list)
            n_assets = 0
            for event_id, asset_values in events.items():
                n_assets += len(asset_values)
                by_subject[subjects_of_event[event_id]].append(statistics.median(asset_values))
            agree = view_agreement.agreement(
                events,
                subjects_of_event,
                f"{feature.level}/{feature.column}/{task}/{side}",
                SUBJECT_FLOOR,
            )
            for reason in agree.reasons:
                agreement_reasons[reason] += 1
            subject_values = [statistics.median(values) for values in by_subject.values()]
            n_subjects = len(subject_values)
            n_values = leaves.get(key, {}).get((task, side), 0)
            row = {
                "task": task,
                "side": side,
                "level": feature.level,
                "feature": feature.column,
                "n_subjects": str(n_subjects),
                "n_events": str(len(events)),
                "n_assets": str(n_assets),
                "n_values": str(n_values),
                "n_events_multiview": str(agree.n_events),
                "n_subjects_multiview": str(agree.n_subjects),
            }
            statistics_publishable = n_subjects >= SUBJECT_FLOOR
            row.update(
                {
                    "median": _cell(statistics.median(subject_values))
                    if statistics_publishable
                    else "",
                    "q25": _cell(_quantile(subject_values, 0.25)) if statistics_publishable else "",
                    "q75": _cell(_quantile(subject_values, 0.75)) if statistics_publishable else "",
                    "mean": _cell(statistics.fmean(subject_values))
                    if statistics_publishable
                    else "",
                    "sd": _cell(statistics.stdev(subject_values))
                    if statistics_publishable and n_subjects >= 2
                    else "",
                    "view_abs_diff": _cell(agree.abs_diff),
                    "view_icc": _cell(agree.icc),
                    "view_icc_range_lo": _cell(agree.range_lo),
                    "view_icc_range_hi": _cell(agree.range_hi),
                }
            )
            zero_values += n_values == 0
            below_floor += not statistics_publishable
            feature_rows.append(row)

    population = {
        "assets": sum(len(members[cell]["assets"]) for cell in cells),
        "cells": len(cell_rows),
        "events": sum(len(members[cell]["events"]) for cell in cells),
        "feature_rows": len(feature_rows),
        "features": len({(f.level, f.column) for f in FEATURES} - set(reasons)),
        "frame_rows": sum(row_counts[cell]["frame"] for cell in cells),
        "subjects": len({contributor.subject for contributor in contributors}),
        "window_rows": sum(row_counts[cell]["window"] for cell in cells),
        "bout_rows": sum(row_counts[cell]["bout"] for cell in cells),
    }
    return _Aggregate(
        cells=cell_rows,
        features=feature_rows,
        published=published,
        excluded=excluded,
        population=population,
        rows_zero_values=zero_values,
        agreement_reasons=agreement_reasons,
        rows_below_subject_floor=below_floor,
        exclusion_reasons=reasons,
    )


def _partition(
    source: Sequence[tuple[str, str]], finite: set[tuple[str, str]]
) -> dict[tuple[str, str], str]:
    """Excluded source columns -> reason; refuses what neither table names (A02, M4.5b P04).

    Precedence: `construct_invalid` (window smoothness, whatever its values), then the named
    `structurally_absent` set, then a declared bout column with no eligible value. A declared
    frame / window feature with no finite value, and any unknown column, refuse.
    """
    reasons: dict[tuple[str, str], str] = {}
    for key in source:
        level, column = key
        if key in CONSTRUCT_INVALID:
            reasons[key] = "construct_invalid"
        elif key in STRUCTURALLY_ABSENT and key not in finite:
            reasons[key] = "structurally_absent"
        elif key in FEATURE_KEYS and key not in finite:
            if level != "bout":
                raise CohortError(
                    f"cohort.FEATURES carries {level} column {column!r}, for which the run "
                    "publishes no finite value anywhere."
                )
            reasons[key] = "no_eligible_bout_values"
        elif key not in FEATURE_KEYS:
            raise CohortError(
                f"The run publishes {level} column {column!r}, which cohort.FEATURES does "
                "not carry."
            )
    missing = sorted(FEATURE_KEYS - set(source))
    if missing:
        level, column = missing[0]
        raise CohortError(f"The run tree carries no {level} column {column!r}.")
    return reasons


def _descriptor_rows() -> list[dict[str, object]]:
    """A08's named projection of `FEATURES`, in canonical `(level, column)` order (A15)."""
    return [
        {
            "raw": feature.raw,
            "ja": feature.ja,
            "en": feature.en,
            "group": "pose",
            "role": "feature",
            "dtype": "numeric",
            "unit": feature.unit,
            "range": list(feature.range),
        }
        for feature in sorted(FEATURES, key=lambda item: (item.level, item.column))
    ]


# YAML 1.1 breaks a line on NEL, LS and PS as well as LF, so a label carrying one of them
# survives JSON and comes back out of the loader as a space. JSON's own `\uXXXX` escape is
# what keeps the round trip total, and PyYAML reads it back as the codepoint.
_YAML_LINE_BREAKS = (0x85, 0x2028, 0x2029)


def _json_scalar(value: object) -> str:
    """Render one scalar as JSON, escaping the codepoints YAML would read as breaks."""
    text = json.dumps(value, ensure_ascii=False)
    for codepoint in _YAML_LINE_BREAKS:
        text = text.replace(chr(codepoint), f"\\u{codepoint:04x}")
    return text


def render_descriptors(rows: list[dict[str, object]]) -> str:
    """Render the fragment without a YAML emitter, so the bytes are ours to fix.

    Every scalar goes out as JSON, which is a YAML subset — that keeps the Japanese
    labels literal and keeps an emitter's line-width and quoting defaults out of a
    byte-pinned artifact.
    """
    lines = ["columns:"]
    for row in rows:
        prefix = "-"
        for key in ("raw", "ja", "en", "group", "role", "dtype", "unit", "range"):
            lines.append(f"{prefix} {key}: {_json_scalar(row[key])}")
            prefix = " "
    return "\n".join(lines) + "\n"


def external_raws(payload: bytes) -> frozenset[str] | None:
    """`../rehab` `columns.yaml` bytes → raw names, families expanded; `None` = no mapping."""
    document = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(document, dict):
        return None
    raws = [str(row["raw"]) for row in document.get("columns") or []]
    for family in document.get("families") or []:
        raws.extend(
            str(family["template_raw"]).format(side=side, level=level)
            for side in family["sides"]
            for level in family["levels"]
        )
    return frozenset(raws)


def _external_descriptors() -> tuple[frozenset[str], str] | None:
    """`../rehab`'s raw names + the digest of the bytes they came from; `None` = unresolved."""
    path = pathlib.Path(__file__).resolve().parents[2].parent / "rehab" / "schema" / "columns.yaml"
    if not path.is_file():
        return None
    # One read feeds both, so the recorded digest names the bytes the check ran on.
    payload = path.read_bytes()
    raws = external_raws(payload)
    return None if raws is None else (raws, _digest_bytes(payload))


def descriptor_collision(rows: list[dict[str, object]]) -> dict[str, object]:
    """A08's two collision layers. The external outcome publishes; it never silently skips."""
    raws = [str(row["raw"]) for row in rows]
    if len(raws) != len(set(raws)):
        raise CohortError("The descriptor fragment carries a duplicate raw name.")
    for raw in raws:
        if not raw.startswith(tuple(f"pose_{level}_" for level in LEVELS)):
            raise CohortError(f"Descriptor raw {raw!r} does not carry the namespacing prefix.")
    resolved = _external_descriptors()
    if resolved is None:
        return {
            "checked": False,
            "source": EXTERNAL_DESCRIPTOR_SOURCE,
            "source_sha256": None,
            "n_external": 0,
            "n_collisions": 0,
        }
    external, digest = resolved
    collisions = sorted(set(raws) & external)
    if collisions:
        raise CohortError(
            f"Descriptor raw {collisions[0]!r} collides with an existing consumer descriptor."
        )
    return {
        "checked": True,
        "source": EXTERNAL_DESCRIPTOR_SOURCE,
        "source_sha256": digest,
        "n_external": len(external),
        "n_collisions": 0,
    }


def render_marker(payload: Mapping[str, Any]) -> bytes:
    """A13's canonical rendering: the registry's own census bytes, so digests agree."""
    return inventory.render_json(dict(payload)).encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tree_digest(out_dir: str | os.PathLike[str], marker: Mapping[str, Any]) -> str:
    """Digest the set over its three files plus the marker minus this self-referential key.

    The marker is the one entry no digest inside the set can cover, so excluding its whole
    body would leave the census and the provenance uncovered — which is exactly the claim
    surface a consumer trusts most (A13).
    """
    out = pathlib.Path(out_dir)
    body = dict(marker)
    generation = {
        key: value
        for key, value in dict(body.get("generation") or {}).items()
        if key != "tree_digest"
    }
    body["generation"] = generation
    # The entry set joins the digest, or a fifth file sits inside the published generation
    # with nothing covering it and `validate_generation` still reads green (A26). The
    # marker is excluded so staging — which has no marker yet — digests to the same value.
    entries = sorted(entry.name for entry in out.iterdir() if entry.name != COHORT_FILENAME)
    lines = ["\t".join(["entries", *entries]) + "\n"]
    # The optional file digests like the required three wherever the entry set holds it; a
    # v1/v2 tree never does, so their recorded digests still recompute (contract-m4u7 D03).
    names = [*PUBLISHED_FILENAMES, *(n for n in (SENSITIVITY_FILENAME,) if n in entries)]
    lines += [f"{name}\t{_digest_bytes((out / name).read_bytes())}\n" for name in names]
    lines.append(render_marker(body).decode("utf-8"))
    return _digest_bytes("".join(lines).encode("utf-8"))


def _remove(path: pathlib.Path) -> None:
    """Remove one path of any kind. A dangling symlink is still ours to clear."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _is_within(candidate: pathlib.Path, ancestor: pathlib.Path) -> bool:
    resolved = os.path.realpath(candidate)
    root = os.path.realpath(ancestor)
    # `/` + os.sep is `//`, which no real path carries (`publishers.md`).
    return resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep)


def _assert_disjoint(out: pathlib.Path, source: pathlib.Path, name: str) -> None:
    """Refuse either containment direction before a single byte moves.

    The swap retires whatever sits at `out`, so an input underneath it is deleted by a
    successful run, and an output underneath an input is swept as debris by the next one.
    """
    if _is_within(out, source):
        raise CohortError(f"The output directory sits inside the {name} directory.")
    if _is_within(source, out):
        raise CohortError(f"The {name} directory sits inside the output directory.")


def _read_marker(path: pathlib.Path) -> dict[str, Any]:
    """Read one marker as a regular file. A symlink is refused, never followed."""
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise CohortError(f"{path.name} is not present.") from error
    if not stat.S_ISREG(mode):
        raise CohortError(f"{path.name} is not a regular file.")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=qualify._reject_duplicate_keys
        )
    except (OSError, ValueError) as error:
        raise CohortError(f"{path.name} is not readable canonical JSON.") from error
    if not isinstance(payload, dict):
        raise CohortError(f"{path.name} is not a JSON object.")
    return payload


def _is_ours(marker: Mapping[str, Any]) -> bool:
    """A14: ownership is the conjunction, so a foreign tool's `v1` is still foreign."""
    generation = marker.get("generation")
    if not isinstance(generation, dict):
        return False
    return (
        generation.get("generator") == GENERATOR
        and generation.get("generator_version") == GENERATOR_VERSION
    )


def _assert_owned(out: pathlib.Path) -> None:
    """Refuse to retire a directory this generator did not publish.

    Judged before the orphan sweep: a refusal must leave the caller's tree exactly as it
    found it, and sweeping first would delete siblings on the way to saying no.
    """
    if not out.exists():
        return
    if not out.is_dir():
        raise CohortError("The output path exists and is not a directory.")
    if any(out.iterdir()) and not _is_ours(_read_marker(out / COHORT_FILENAME)):
        raise CohortError(
            "The output directory is not empty and carries no marker from this generator."
        )


def _snapshot(root: pathlib.Path) -> dict[str, str]:
    """Recursive, non-following inventory of one input tree: path -> kind and content.

    Non-following is load-bearing — `videos`, `inventory` and `renv/library` are symlinks
    in a worktree, so following them digests a tree this publisher never read. `rglob` is
    what supplies it: pathlib never descends a symlinked directory, where `glob.glob`'s
    `**` does. Both see dotfiles. A semantic validator canonicalizes its input and
    therefore cannot witness bytes, which is why the read-only claim over all three inputs
    rests on this instead (A29).
    """
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        key = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries[key] = f"link:{os.readlink(path)}"  # noqa: PTH115
        elif path.is_dir():
            entries[key] = "dir"
        elif path.is_file():
            entries[key] = f"file:{_digest_bytes(path.read_bytes())}"
        else:
            entries[key] = "other"
    return entries


def _contributors(
    inventory_dir: pathlib.Path, sessions_dir: pathlib.Path, run_dir: pathlib.Path
) -> list[_Contributor]:
    """Join registry, placements and manifest into the run's `ok` canonical asset set."""
    assets = {
        row["asset_id"]: row
        for row in _read_table(inventory_dir / inventory.ASSETS_FILENAME, ("asset_id",))
    }
    placements = {
        row["asset_id"]: row
        for row in _read_table(sessions_dir / sessions.PLACEMENTS_FILENAME, ("asset_id",))
        if row["placement"] == sessions.PLACED
    }
    try:
        rows = corpus_run.read_manifest(run_dir / corpus_run.MANIFEST_FILENAME)
        corpus_run.validate_manifest(rows, sorted(placements))
    except corpus_run.ManifestError as error:
        raise CohortError(f"The run manifest is not a total partition: {error}") from error
    except OSError as error:
        raise CohortError(f"The run manifest is not readable: {error}") from error
    contributors = []
    for row in rows:
        if row["disposition"] != corpus_run.DISPOSITION_OK:
            continue
        asset = assets[row["asset_id"]]
        # The manifest is the run's own output and carries no digest of its own, while the
        # placement ledger is inside a validated generation. So the event grouping keys —
        # which decide the estimand's middle stage — are read from the ledger, and the
        # manifest has to agree with it rather than name the grouping (A28).
        placement = placements[row["asset_id"]]
        declared = (row["event_id"], row["camera_name"])
        placed = (placement["event_id"], placement["camera_name"])
        if declared != placed:
            raise CohortError(
                f"The run manifest places {row['asset_id']} at {declared}, "
                f"but the session tree places it at {placed}."
            )
        contributors.append(
            _Contributor(
                asset_id=row["asset_id"],
                task=asset["task"],
                side=asset["side"],
                subject=asset["subject_ordinal"],
                event_id=placement["event_id"],
                camera_name=placement["camera_name"],
            )
        )
    if not contributors:
        raise CohortError("The run manifest carries no successful asset.")
    return contributors


def _event_record(path: pathlib.Path, event: str) -> dict[str, Any]:
    try:
        payload = inventory.read_marker(path)
    except (OSError, ValueError) as error:
        raise CohortError(f"{event} carries no readable {path.name}.") from error
    if not isinstance(payload, dict):
        raise CohortError(f"{event}'s {path.name} is not a JSON object.")
    return payload


def _scalar(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (str, int))


def _measurement(run_root: pathlib.Path, contributors: list[_Contributor]) -> dict[str, Any]:
    """The one pose configuration + clinical method every contributing event ran (D01).

    Read per event because `run_report.json` is a driver's declaration while these two files
    are what resume and refresh key on; a tree mixing either describes no single instrument.
    """
    configs: dict[str, dict[str, Any]] = {}
    methods: set[str] = set()
    events = sorted({contributor.event_id for contributor in contributors})
    for index, event_id in enumerate(events, 1):
        # An event id is patient-adjacent (`data-boundary.md`): refusals name its position.
        event = f"Contributing event {index} of {len(events)}"
        config = _event_record(run_root / event_id / corpus_run.POSE_CONFIG_FILENAME, event)
        marker = _event_record(run_root / event_id / corpus_run.MARKER_FILENAME, event)
        if marker.get("status") != corpus_run.MARKER_COMPLETE:
            raise CohortError(f"{event}'s completion marker is not complete.")
        method = marker.get("analysis_sha256")
        if not isinstance(method, str) or not _HEX64.fullmatch(method):
            raise CohortError(f"{event} names no clinical producer digest.")
        if set(config) != set(corpus_run.POSE_CONFIG_FIELDS):
            raise CohortError(f"{event}'s pose configuration is not the frozen field set.")
        if not all(_scalar(value) for value in config.values()):
            raise CohortError(f"{event}'s pose configuration carries a non-scalar value.")
        # Canonical text, not `==`: Python equates True with 1 and 1.0, the records do not.
        configs[json.dumps(config, sort_keys=True)] = config
        methods.add(method)
    if len(configs) != 1:
        raise CohortError(f"The run tree carries {len(configs)} pose configurations, not one.")
    if len(methods) != 1:
        raise CohortError(f"The run tree carries {len(methods)} clinical producers, not one.")
    (config,) = configs.values()
    return {
        "pose_config": config,
        "pose_config_sha256": corpus_run.pose_config_digest(config),
        "analysis_sha256": methods.pop(),
    }


@dataclass(frozen=True)
class _Baseline:
    identity: dict[str, str]
    rows: dict[tuple[str, str, str, str], dict[str, str]]


def _baseline_row(row: dict[str, str]) -> tuple[str, str, str, str]:
    key = (row["task"], row["side"], row["level"], row["feature"])
    if (
        row["task"] not in inventory.TASKS
        or row["side"] not in inventory.SIDES
        or row["level"] not in LEVELS
        or not _FEATURE_NAME.fullmatch(row["feature"])
    ):
        raise CohortError(f"The baseline feature table names an unknown row {key}.")
    if not all(_COUNT.fullmatch(row[name]) for name in ("n_subjects", "n_values")):
        raise CohortError(f"The baseline row {key} carries a malformed count.")
    statistics_cells = [row[name] for name in DISTRIBUTION_FIELDS if name in row]
    if not all(cell == "" or _DECIMAL.fullmatch(cell) for cell in statistics_cells):
        raise CohortError(f"The baseline row {key} carries a malformed statistic.")
    if int(row["n_subjects"]) < SUBJECT_FLOOR and any(statistics_cells):
        raise CohortError(f"The baseline row {key} publishes a statistic below the subject floor.")
    return key


def _read_baseline(root: pathlib.Path) -> _Baseline:
    """A previous generation of this generator, validated by its own version's profile (D03).

    The digest is recomputed, not trusted, and the table is re-validated because a digest
    proves bytes, never shape (`publishers.md`).
    """
    marker = _read_marker(root / COHORT_FILENAME)
    generation = marker.get("generation")
    if not isinstance(generation, dict) or generation.get("generator") != GENERATOR:
        raise CohortError("The baseline was published by another generator.")
    version = generation.get("generator_version")
    if not isinstance(version, str) or version not in BASELINE_VERSIONS:
        raise CohortError(f"The baseline generator version {version!r} is not supported.")
    recorded = generation.get("tree_digest")
    if not isinstance(recorded, str) or not _HEX64.fullmatch(recorded):
        raise CohortError("The baseline marker carries no tree digest.")
    if version == GENERATOR_VERSION:
        validate_generation(root)
    else:
        for name in PUBLISHED_FILENAMES:
            path = root / name
            try:
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise CohortError(f"The baseline {name} is not a regular file.")
            except OSError as error:
                raise CohortError(f"The baseline is missing {name}.") from error
        try:
            recomputed = tree_digest(root, marker)
        except OSError as error:
            raise CohortError("The baseline set is not readable.") from error
        if recomputed != recorded:
            raise CohortError("The baseline tree digest does not match its bytes.")
    payload = (root / FEATURES_FILENAME).read_bytes()
    try:
        records = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise CohortError("The baseline feature table is not readable CSV.") from error
    if len(records) < 2:
        raise CohortError("The baseline feature table carries no row.")
    header = tuple(records[0])
    _assert_unique_header(f"baseline {FEATURES_FILENAME}", header)
    if missing := [name for name in BASELINE_COLUMNS if name not in header]:
        raise CohortError(f"The baseline feature table is missing the columns {missing}.")
    rows: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for record in records[1:]:
        if len(record) != len(header):
            raise CohortError("The baseline feature table carries a ragged row.")
        row = dict(zip(header, record, strict=True))
        key = _baseline_row(row)
        if key in rows:
            raise CohortError(f"The baseline feature table repeats the row {key}.")
        rows[key] = row
    identity = {
        "generator_version": version,
        "tree_digest": recorded,
        "features_sha256": _digest_bytes(payload),
    }
    return _Baseline(identity=identity, rows=rows)


_OPERANDS = ("median", "q25", "q75")


def _number(cell: str) -> Decimal | None:
    return Decimal(cell) if cell else None


def _sensitivity_rows(features: list[dict[str, str]], baseline: _Baseline) -> list[dict[str, str]]:
    """Union of both tables' keys; arithmetic on the serialized cells (contract-m4u7 P06-P07)."""
    current = {(r["task"], r["side"], r["level"], r["feature"]): r for r in features}
    widest = max(
        (len(row[n]) for row in (*features, *baseline.rows.values()) for n in _OPERANDS),
        default=1,
    )
    # P07 is arithmetic on the serialized cells: precision sized from the widest operand, so no
    # difference or 9-place quotient rounds before rendering; a fresh context, so the caller's
    # rounding mode or traps never reach a published digit.
    exact = Context(prec=2 * (widest + DECIMALS) + 4, rounding=ROUND_HALF_EVEN)
    with localcontext(exact):
        return [
            _sensitivity_row(key, current.get(key), baseline.rows.get(key))
            for key in sorted(current.keys() | baseline.rows.keys())
        ]


def _sensitivity_row(
    key: tuple[str, str, str, str], now: dict[str, str] | None, then: dict[str, str] | None
) -> dict[str, str]:
    row: dict[str, str] = dict(zip(("task", "side", "level", "feature"), key, strict=True))
    row["status"] = "common" if now and then else "new" if now else "retired"
    for name in ("n_subjects", "n_values"):
        row[name] = now[name] if now else ""
        row[f"baseline_{name}"] = then[name] if then else ""
    median = _number(now["median"]) if now else None
    base = _number(then["median"]) if then else None
    row["median"], row["baseline_median"] = _cell(median), _cell(base)
    q25, q75 = (_number(then[name]) if then else None for name in ("q25", "q75"))
    iqr = q75 - q25 if q25 is not None and q75 is not None else None
    delta = median - base if median is not None and base is not None else None
    row["baseline_iqr"] = _cell(iqr)
    row["delta_median"] = _cell(delta)
    scaled = delta / iqr if delta is not None and iqr is not None and iqr > 0 else None
    row["delta_iqr_scaled"] = _cell(scaled)
    return row


def _write(directory: pathlib.Path, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8", newline="")


def _publish(
    staging: pathlib.Path,
    aggregate: _Aggregate,
    input_digests: dict[str, str],
    measurement: dict[str, Any],
    baseline: _Baseline | None,
) -> None:
    """Write the set, digesting it last because the digest covers the marker."""
    staging.mkdir(parents=True)
    _write(staging, CELLS_FILENAME, inventory.render_csv(CELL_COLUMNS, aggregate.cells))
    _write(staging, FEATURES_FILENAME, inventory.render_csv(FEATURE_COLUMNS, aggregate.features))
    rows = _descriptor_rows()
    _write(staging, DESCRIPTORS_FILENAME, render_descriptors(rows))
    sensitivity = None
    if baseline is not None:
        changes = _sensitivity_rows(aggregate.features, baseline)
        _write(staging, SENSITIVITY_FILENAME, inventory.render_csv(SENSITIVITY_COLUMNS, changes))
        sensitivity = {
            "baseline": baseline.identity,
            "status_counts": {
                status: sum(row["status"] == status for row in changes)
                for status in SENSITIVITY_STATUSES
            },
            "estimand": SENSITIVITY_ESTIMAND,
        }
    marker: dict[str, Any] = {
        "population": aggregate.population,
        "columns": {
            "published": [
                {"level": level, "column": column} for level, column in aggregate.published
            ],
            "excluded": [
                {
                    "level": level,
                    "column": column,
                    "reason": aggregate.exclusion_reasons[(level, column)],
                }
                for level, column in aggregate.excluded
            ],
        },
        "estimand": ESTIMAND,
        "rows_zero_values": aggregate.rows_zero_values,
        "agreement_reasons": aggregate.agreement_reasons,
        "rows_below_subject_floor": aggregate.rows_below_subject_floor,
        "descriptor_collision": descriptor_collision(rows),
        "measurement": measurement,
        "sensitivity": sensitivity,
        "generation": {
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "input_digests": input_digests,
        },
    }
    marker["generation"]["tree_digest"] = tree_digest(staging, marker)
    (staging / COHORT_FILENAME).write_bytes(render_marker(marker))


def run(
    inventory_dir: str | os.PathLike[str],
    sessions_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    baseline_dir: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Publish the cohort set. Every refusal raises `CohortError` and publishes nothing.

    `baseline_dir` = a previous generation to publish `cohort_sensitivity.csv` against; it may
    be `out_dir` itself (read whole before the swap), never strictly inside or around it (D04).
    """
    registry = pathlib.Path(inventory_dir)
    tree = pathlib.Path(sessions_dir)
    run_root = pathlib.Path(run_dir)
    out = pathlib.Path(out_dir)
    for source, name in ((registry, "inventory"), (tree, "sessions"), (run_root, "run")):
        _assert_disjoint(out, source, name)
    baseline_root = None if baseline_dir is None else pathlib.Path(baseline_dir)
    if baseline_root is not None and os.path.realpath(baseline_root) != os.path.realpath(out):
        _assert_disjoint(out, baseline_root, "baseline")

    # Every upstream refusal reaches the caller as this module's own class (A16): a
    # consumer that has to name three foreign exception types to call one function cannot
    # tell a corrupt input from a missing one.
    try:
        inventory.validate_generation(registry)
        sessions.validate_generation(tree, inventory_dir=registry)
        sessions_tree = sessions.tree_digest(tree)
        sessions_generation = sessions.generation_digest(tree)
    except CohortError:
        raise
    except Exception as error:
        raise CohortError(f"The upstream generations are not readable: {error}") from error

    # An upstream tree that moves while it is being read makes every published count a
    # statement about a corpus that no longer exists. The snapshot covers all three inputs
    # and subsumes a digest re-comparison, which sees only the files its own contract names
    # (A26 is the same gap one publisher up).
    inputs = [("inventory", registry), ("sessions", tree), ("run", run_root)]
    if baseline_root is not None:
        inputs.append(("baseline", baseline_root))
    before = {name: _snapshot(path) for name, path in inputs}
    contributors = _contributors(registry, tree, run_root)
    aggregate = _aggregate(contributors, run_root)
    measurement = _measurement(run_root, contributors)
    baseline = None if baseline_root is None else _read_baseline(baseline_root)
    for name, path in inputs:
        if _snapshot(path) != before[name]:
            raise CohortError(f"The {name} tree changed while the cohort was being read.")
    input_digests = {
        "inventory": _digest_bytes((registry / inventory.CENSUS_FILENAME).read_bytes()),
        "run_manifest": _digest_bytes((run_root / corpus_run.MANIFEST_FILENAME).read_bytes()),
        "sessions_generation": sessions_generation,
        "sessions_tree": sessions_tree,
    }

    _assert_owned(out)
    # A pid-keyed sibling name is not this run's to delete: pid reuse and a foreign tool
    # both spell it, and the old pre-run clear destroyed whatever wore the name before the
    # swap could succeed. `mkdtemp` names a path nothing else holds, so cleanup touches
    # only what this invocation created and no orphan sweep is needed (A30).
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f"{out.name}.staging.", dir=out.parent))
    retiring = pathlib.Path(tempfile.mkdtemp(prefix=f"{out.name}.retiring.", dir=out.parent))
    published = False
    try:
        _remove(staging)
        _publish(staging, aggregate, input_digests, measurement, baseline)
        try:
            if out.exists():
                out.rename(retiring)
            staging.rename(out)
            published = True
        except OSError:
            if retiring.exists() and not out.exists():
                retiring.rename(out)
            raise
    finally:
        if not published:
            _remove(staging)
        _remove(retiring)
    return out


def validate_generation(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the marker of a published set, or raise when it is not this set's own.

    Recomputing the digest is what makes an edit to any published byte — census included —
    a refusal at the consumer rather than a silent read of a mixed generation.
    """
    out = pathlib.Path(out_dir)
    marker = _read_marker(out / COHORT_FILENAME)
    if tuple(sorted(marker)) != tuple(sorted(MARKER_KEYS)):
        raise CohortError("The cohort marker does not carry the frozen key set.")
    generation = marker["generation"]
    if not isinstance(generation, dict) or tuple(sorted(generation)) != tuple(
        sorted(GENERATION_KEYS)
    ):
        raise CohortError("The cohort generation block does not carry the frozen key set.")
    if not _is_ours(marker):
        raise CohortError("The cohort set was published by another generator.")
    for name in PUBLISHED_FILENAMES:
        path = out / name
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise CohortError(f"{name} is not a regular file.")
        except OSError as error:
            raise CohortError(f"The published set is missing {name}.") from error
    # A symlink in the optional slot digests its target's bytes, so membership needs `lstat`.
    optional = out / SENSITIVITY_FILENAME
    if (marker["sensitivity"] is None) == os.path.lexists(optional):
        raise CohortError("The sensitivity file and the marker's sensitivity block disagree.")
    if os.path.lexists(optional) and not stat.S_ISREG(optional.lstat().st_mode):
        raise CohortError(f"{SENSITIVITY_FILENAME} is not a regular file.")
    try:
        recomputed = tree_digest(out, marker)
    except OSError as error:
        raise CohortError("The published set is not readable.") from error
    if recomputed != generation["tree_digest"]:
        raise CohortError("The cohort tree digest does not match the published bytes.")
    return marker


def recheck_descriptor_collision(
    out_dir: str | os.PathLike[str], schema: str | os.PathLike[str]
) -> None:
    """Re-run a published set's collision check against the consumer schema it recorded.

    Raises when the set was published unchecked, *schema* is not the bytes its census
    digested, or a published raw collides — silence = the recorded outcome holds.
    """
    out = pathlib.Path(out_dir)
    collision = validate_generation(out)["descriptor_collision"]
    if collision.get("checked") is not True:
        raise CohortError("The cohort set was published without a descriptor collision check.")
    payload = pathlib.Path(schema).read_bytes()
    if _digest_bytes(payload) != collision.get("source_sha256"):
        raise CohortError("The consumer schema is not the one the collision check digested.")
    external = external_raws(payload)
    if external is None or len(external) != collision.get("n_external"):
        raise CohortError("The consumer schema does not yield the recorded descriptor count.")
    published = yaml.safe_load((out / DESCRIPTORS_FILENAME).read_text(encoding="utf-8"))
    collisions = sorted({str(row["raw"]) for row in published["columns"]} & external)
    if collisions:
        raise CohortError(
            f"Descriptor raw {collisions[0]!r} collides with an existing consumer descriptor."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=GENERATOR, description="Publish the cohort aggregate for the 2D corpus run."
    )
    parser.add_argument("--inventory", required=True, help="Asset registry directory.")
    parser.add_argument("--sessions", required=True, help="Session tree directory.")
    parser.add_argument("--run", required=True, help="Corpus run directory.")
    parser.add_argument("--out", required=True, help="Output directory to publish.")
    parser.add_argument(
        "--baseline",
        help="Previous cohort generation to compare against (writes cohort_sensitivity.csv).",
    )
    arguments = parser.parse_args(argv)
    out = run(
        arguments.inventory, arguments.sessions, arguments.run, arguments.out, arguments.baseline
    )
    marker = validate_generation(out)
    print(
        f"cohort: {marker['population']['cells']} cells, "
        f"{marker['population']['feature_rows']} feature rows, "
        f"{len(marker['columns']['published'])} published columns, "
        f"{len(marker['columns']['excluded'])} excluded"
    )
    if marker["sensitivity"] is not None:
        counts = marker["sensitivity"]["status_counts"]
        print("sensitivity: " + ", ".join(f"{counts[s]} {s}" for s in SENSITIVITY_STATUSES))
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script parity
    sys.exit(main())
