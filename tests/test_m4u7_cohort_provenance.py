"""M4.7 contract-only checks; synthetic fixtures, public publisher boundary."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import ModuleType
from typing import Any

import pytest

from pose_estimation import corpus_run, inventory
from test_cohort import _build_inputs, _Inputs, _marker, _publish, _read_csv

_CONFIG = dict.fromkeys(corpus_run.POSE_CONFIG_FIELDS, "synthetic") | {
    "model": "synthetic-pose",
    "det_frequency": 1,
    "single_subject": True,
    "task_gap_s": 2.5,
    "calibration": None,
}
_ANALYSIS = "a1" * 32
_FEATURES = "cohort_features.csv"
_SENSITIVITY = "cohort_sensitivity.csv"
_REQUIRED = ("cohort_cells.csv", _FEATURES, "descriptors.yaml")
_KEY = ("task", "side", "level", "feature")
_REQUIRED_FEATURES = (*_KEY, "n_subjects", "n_values", "median", "q25", "q75")
_DISTRIBUTION = ("median", "q25", "q75", "mean", "sd")
_SENSITIVITY_COLUMNS = (
    *_KEY,
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


@dataclass(frozen=True)
class _Seed:
    root: pathlib.Path
    inputs: _Inputs
    out: pathlib.Path


def _cohort() -> ModuleType:
    return importlib.import_module("pose_estimation.cohort")


def _json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _events(inputs: _Inputs) -> list[pathlib.Path]:
    return sorted(
        {
            inputs.run / row["event_id"]
            for row in corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
            if row["disposition"] == "ok"
        }
    )


def _metadata(inputs: _Inputs, config: Mapping[str, object] = _CONFIG) -> None:
    for event in _events(inputs):
        corpus_run.write_pose_config(event, config)
        corpus_run.write_marker(event, status=corpus_run.MARKER_COMPLETE, analysis_sha256=_ANALYSIS)


@pytest.fixture(scope="module")
def seed(tmp_path_factory: pytest.TempPathFactory) -> _Seed:
    root = tmp_path_factory.mktemp("m4u7")
    inputs_root = root / "inputs"
    inputs_root.mkdir()
    cohort = _cohort()
    inputs = _build_inputs(inputs_root, cohort)
    _metadata(inputs)
    out = root / "published"
    _publish(cohort, inputs, out)
    return _Seed(inputs_root, inputs, out)


def _copy_inputs(seed: _Seed, root: pathlib.Path) -> _Inputs:
    shutil.copytree(seed.root, root, symlinks=True)
    return _Inputs(
        root / seed.inputs.inventory.relative_to(seed.root),
        root / seed.inputs.sessions.relative_to(seed.root),
        root / seed.inputs.run.relative_to(seed.root),
        seed.inputs.assets,
    )


def _copy_output(seed: _Seed, out: pathlib.Path) -> pathlib.Path:
    shutil.copytree(seed.out, out, symlinks=True)
    return out


def _snapshot(root: pathlib.Path) -> dict[str, tuple[str, str]]:
    state: dict[str, tuple[str, str]] = {}

    def walk(path: pathlib.Path) -> None:
        key = path.relative_to(root).as_posix()
        if path.is_symlink():
            state[key] = "symlink", os.readlink(path)  # noqa: PTH115
        elif path.is_dir():
            state[key] = "directory", ""
            for child in sorted(path.iterdir()):
                walk(child)
        elif path.is_file():
            state[key] = "file", hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            state[key] = "other", ""

    if root.exists() or root.is_symlink():
        walk(root)
    return state


def _run(cohort: ModuleType, inputs: _Inputs, out: pathlib.Path, baseline: pathlib.Path) -> None:
    cohort.run(inputs.inventory, inputs.sessions, inputs.run, out, baseline_dir=baseline)


def _resign(cohort: ModuleType, root: pathlib.Path) -> None:
    marker = _marker(root)
    marker["generation"]["tree_digest"] = cohort.tree_digest(root, marker)
    _json(root / "cohort.json", marker)


def _legacy(cohort: ModuleType, root: pathlib.Path, version: str) -> None:
    marker = _marker(root)
    marker.pop("measurement", None)
    marker.pop("sensitivity", None)
    marker["generation"]["generator_version"] = version
    _json(root / "cohort.json", marker)
    _resign(cohort, root)


def _csv(path: pathlib.Path, header: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _raw_csv(path: pathlib.Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)


def _assert_refused(
    cohort: ModuleType, inputs: _Inputs, out: pathlib.Path, baseline: pathlib.Path | None = None
) -> None:
    before = _snapshot(out)
    kwargs = {} if baseline is None else {"baseline_dir": baseline}
    try:
        with pytest.raises(cohort.CohortError):
            cohort.run(inputs.inventory, inputs.sessions, inputs.run, out, **kwargs)
    finally:
        assert _snapshot(out) == before, "refusal changed the destination's bytes or membership"


def _key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in _KEY)


def _oracle(
    current: list[dict[str, str]], baseline: list[dict[str, str]], decimals: int
) -> list[dict[str, str]]:
    left = {_key(row): row for row in current}
    right = {_key(row): row for row in baseline}
    result = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        row = dict(zip(_KEY, key, strict=True))
        row["status"] = "common" if a is not None and b is not None else "new" if a else "retired"
        for prefix, source in (("", a), ("baseline_", b)):
            for field in ("n_subjects", "n_values"):
                row[prefix + field] = str(int(source[field])) if source else ""
            row[prefix + "median"] = (
                f"{Decimal(source['median']):.{decimals}f}" if source and source["median"] else ""
            )
        with localcontext() as ctx:
            ctx.prec = 80
            iqr = Decimal(b["q75"]) - Decimal(b["q25"]) if b and b["q75"] and b["q25"] else None
            delta = (
                Decimal(a["median"]) - Decimal(b["median"])
                if a and b and a["median"] and b["median"]
                else None
            )
            scaled = delta / iqr if delta is not None and iqr is not None and iqr > 0 else None
            for field, value in (
                ("baseline_iqr", iqr),
                ("delta_median", delta),
                ("delta_iqr_scaled", scaled),
            ):
                row[field] = "" if value is None else f"{value:.{decimals}f}"
        result.append(row)
    return result


def test_p01_uniform_measurement_and_recorded_method(seed: _Seed) -> None:
    expected = {
        "pose_config": _CONFIG,
        "pose_config_sha256": hashlib.sha256(
            json.dumps(_CONFIG, sort_keys=True).encode()
        ).hexdigest(),
        "analysis_sha256": _ANALYSIS,
    }
    assert _marker(seed.out).get("measurement") == expected


@pytest.mark.parametrize("value", [None, False, True, 0, 1, -3, 1.0, 0.125, "", "姿勢"])
def test_p01_scalar_config_round_trip(seed: _Seed, tmp_path: pathlib.Path, value: object) -> None:
    cohort = _cohort()
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    config = _CONFIG | {"model": value}
    _metadata(inputs, config)
    out = tmp_path / "out"
    _publish(cohort, inputs, out)
    block = _marker(out)["measurement"]
    assert json.dumps(block["pose_config"], sort_keys=True) == json.dumps(config, sort_keys=True)
    assert (
        block["pose_config_sha256"]
        == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    )


def test_p01_digest_is_canonical_over_generated_scalar_configs() -> None:
    rng = random.Random(47101)  # noqa: S311 — repeatable generated inputs
    scalars = [None, False, True, 0, 1, -1, 0.125, -2.5, "", "姿勢", '\\\n"']
    digest = getattr(corpus_run, "pose_config_digest", None)
    assert callable(digest), "P01: public pose_config_digest is absent"
    for _ in range(64):
        fields = list(corpus_run.POSE_CONFIG_FIELDS)
        rng.shuffle(fields)
        config = {name: rng.choice(scalars) for name in fields}
        expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        assert digest(config) == digest(dict(reversed(list(config.items())))) == expected
    assert digest(_CONFIG | {"single_subject": True}) != digest(_CONFIG | {"single_subject": 1})


_JSON_DAMAGE = (
    "absent",
    "symlink",
    "directory",
    "invalid-json",
    "invalid-utf8",
    "duplicate-key",
    "array",
    "null",
    "string",
)
_MEASUREMENT_CASES = [
    *(f"config:{kind}" for kind in _JSON_DAMAGE),
    *(f"marker:{kind}" for kind in _JSON_DAMAGE),
    "marker:status-absent",
    "marker:status-failed",
    "marker:status-null",
    "marker:status-true",
    "marker:analysis-absent",
    "marker:analysis-null",
    "marker:analysis-int",
    "marker:analysis-bool",
    "marker:analysis-list",
    "marker:analysis-object",
    "marker:analysis-empty",
    "marker:analysis-mixed",
    "marker:analysis-short",
    "marker:analysis-long",
    "marker:analysis-upper",
    "marker:analysis-nonhex",
    "marker:analysis-newline",
    "config:missing-key",
    "config:extra-key",
    "config:list",
    "config:object",
    "config:nan",
    "config:positive-inf",
    "config:negative-inf",
    "mixed:config",
    "mixed:boolean-int",
    "mixed:int-float",
    "mixed:analysis",
]


def _damage_measurement(inputs: _Inputs, case: str) -> None:
    owner, kind = case.split(":")
    events = _events(inputs)
    event = events[0]
    filename = corpus_run.MARKER_FILENAME if owner == "marker" else corpus_run.POSE_CONFIG_FILENAME
    path = event / filename
    if kind == "absent":
        path.unlink()
    elif kind == "symlink":
        target = event / "metadata-copy.json"
        path.rename(target)
        path.symlink_to(target)
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "invalid-json":
        path.write_text("{", encoding="utf-8")
    elif kind == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif kind == "duplicate-key":
        name = "analysis_sha256" if owner == "marker" else "model"
        text = path.read_text(encoding="utf-8")
        path.write_text('{"' + name + '":null,' + text.lstrip()[1:], encoding="utf-8")
    elif kind in {"array", "null", "string"}:
        _json(path, {"array": [], "null": None, "string": "metadata"}[kind])
    elif owner == "marker":
        marker = json.loads(path.read_text(encoding="utf-8"))
        field, variant = kind.split("-", 1)
        name = "analysis_sha256" if field == "analysis" else field
        if variant == "absent":
            marker.pop(name)
        else:
            marker[name] = {
                "failed": "failed",
                "null": None,
                "true": True,
                "int": 17,
                "bool": False,
                "list": [],
                "object": {},
                "empty": "",
                "mixed": "mixed",
                "short": "a" * 63,
                "long": "a" * 65,
                "upper": "A" * 64,
                "nonhex": "g" * 64,
                "newline": _ANALYSIS + "\n",
            }[variant]
        _json(path, marker)
    elif owner == "config":
        config: dict[str, object] = dict(_CONFIG)
        if kind == "missing-key":
            config.pop("model")
        elif kind == "extra-key":
            config["extra"] = 1
        else:
            config["model"] = {
                "list": [],
                "object": {},
                "nan": float("nan"),
                "positive-inf": float("inf"),
                "negative-inf": -float("inf"),
            }[kind]
        _json(path, config)
    else:
        assert len(events) > 1
        if kind == "analysis":
            corpus_run.write_marker(
                events[1], status=corpus_run.MARKER_COMPLETE, analysis_sha256="b2" * 32
            )
        else:
            field, value = {
                "config": ("model", "another-pose"),
                "boolean-int": ("single_subject", 1),
                "int-float": ("det_frequency", 1.0),
            }[kind]
            corpus_run.write_pose_config(events[1], _CONFIG | {field: value})


@pytest.mark.parametrize("case", _MEASUREMENT_CASES)
@pytest.mark.parametrize("existing", [False, True], ids=["new-output", "owned-output"])
def test_p02_measurement_refusals_preserve_destination(
    seed: _Seed, tmp_path: pathlib.Path, case: str, existing: bool
) -> None:
    cohort = _cohort()
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    out = tmp_path / "out"
    if existing:
        _copy_output(seed, out)
    _damage_measurement(inputs, case)
    _assert_refused(cohort, inputs, out)


@pytest.mark.parametrize("field", corpus_run.POSE_CONFIG_FIELDS)
def test_p02_scalar_rule_covers_every_config_field(
    seed: _Seed, tmp_path: pathlib.Path, field: str
) -> None:
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    corpus_run.write_pose_config(_events(inputs)[0], _CONFIG | {field: ["container"]})
    out = _copy_output(seed, tmp_path / "out")
    _assert_refused(_cohort(), inputs, out)


@pytest.mark.parametrize("filename", [corpus_run.POSE_CONFIG_FILENAME, corpus_run.MARKER_FILENAME])
def test_p02_nonregular_metadata_refuses_without_blocking(
    seed: _Seed, tmp_path: pathlib.Path, filename: str
) -> None:
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    path = _events(inputs)[0] / filename
    path.unlink()
    os.mkfifo(path)
    out = _copy_output(seed, tmp_path / "out")
    before = _snapshot(out)
    script = """
import sys
from pose_estimation import cohort
try:
    cohort.run(*sys.argv[1:])
except cohort.CohortError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            *(str(path) for path in (inputs.inventory, inputs.sessions, inputs.run, out)),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert _snapshot(out) == before
    assert result.returncode == 0, result.stderr


def test_p03_equivalent_config_serializations_publish_one_identity(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    for index, event in enumerate(_events(inputs)):
        config = dict(reversed(list(_CONFIG.items()))) if index % 2 else _CONFIG
        (event / corpus_run.POSE_CONFIG_FILENAME).write_text(
            json.dumps(config, indent=index % 3) + "\n", encoding="utf-8"
        )
    out = tmp_path / "out"
    _publish(_cohort(), inputs, out)
    assert (
        _marker(out)["measurement"]["pose_config_sha256"]
        == hashlib.sha256(json.dumps(_CONFIG, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize("kind", ["missing", "malformed", "foreign"])
def test_p03_noncontributing_events_do_not_enter_measurement(
    seed: _Seed, tmp_path: pathlib.Path, kind: str
) -> None:
    cohort = _cohort()
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    manifest_path = inputs.run / corpus_run.MANIFEST_FILENAME
    header, manifest = _read_csv(manifest_path)
    event_id = manifest[0]["event_id"]
    for row in manifest:
        if row["event_id"] == event_id:
            row["disposition"] = "run_failed"
    _csv(manifest_path, header, manifest)
    extra = inputs.run / "synthetic-extra-event"
    extra.mkdir()
    for event in (inputs.run / event_id, extra):
        for filename in (corpus_run.MARKER_FILENAME, corpus_run.POSE_CONFIG_FILENAME):
            path = event / filename
            path.unlink(missing_ok=True)
            if kind == "malformed":
                path.write_text("{", encoding="utf-8")
            elif kind == "foreign":
                _json(path, {"foreign": ["not contributing"]})
    out = tmp_path / "out"
    _publish(cohort, inputs, out)
    assert _marker(out)["measurement"]["pose_config"] == _CONFIG
    assert _marker(out)["measurement"]["analysis_sha256"] == _ANALYSIS


def test_p03_two_asset_event_has_one_measurement_identity(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    path = inputs.run / corpus_run.MANIFEST_FILENAME
    header, manifest = _read_csv(path)
    counts = Counter(row["event_id"] for row in manifest if row["disposition"] == "ok")
    event_id = next(event for event, count in counts.items() if count == 2)
    next(row for row in manifest if row["event_id"] == event_id)["disposition"] = "run_failed"
    assert sum(row["disposition"] == "ok" for row in manifest if row["event_id"] == event_id) == 1
    _csv(path, header, manifest)
    out = tmp_path / "one-asset-event"
    _publish(_cohort(), inputs, out)
    assert _marker(out)["measurement"] == _marker(seed.out)["measurement"]


@pytest.mark.parametrize("filename", [corpus_run.POSE_CONFIG_FILENAME, corpus_run.MARKER_FILENAME])
def test_p03_two_asset_event_bad_metadata_refuses(
    seed: _Seed, tmp_path: pathlib.Path, filename: str
) -> None:
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    manifest = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    counts = Counter(row["event_id"] for row in manifest if row["disposition"] == "ok")
    event_id = next(event for event, count in counts.items() if count == 2)
    (inputs.run / event_id / filename).write_text("{", encoding="utf-8")
    out = _copy_output(seed, tmp_path / "out")
    _assert_refused(_cohort(), inputs, out)


def _leaves(value: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, (*prefix, key))
    else:
        yield prefix, value


@pytest.mark.parametrize("block", ["measurement", "sensitivity"])
def test_p04_every_new_marker_leaf_is_digest_covered(
    seed: _Seed, tmp_path: pathlib.Path, block: str
) -> None:
    cohort = _cohort()
    out = tmp_path / "out"
    if block == "measurement":
        _copy_output(seed, out)
    else:
        _run(cohort, seed.inputs, out, seed.out)
    original = _marker(out)
    assert block in original
    assert isinstance(original[block], dict)
    leaves = list(_leaves(original[block], (block,)))
    assert leaves
    for keys, value in leaves:
        marker = json.loads(json.dumps(original))
        cursor = marker
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = (
            not value
            if isinstance(value, bool)
            else value + 1
            if isinstance(value, (int, float))
            else "altered"
            if value is None
            else str(value) + "changed"
        )
        _json(out / "cohort.json", marker)
        with pytest.raises(cohort.CohortError):
            cohort.validate_generation(out)
    _json(out / "cohort.json", original)
    cohort.validate_generation(out)


def test_p05_baseline_changes_only_sensitivity_and_digest(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort()
    out = tmp_path / "with-baseline"
    _run(cohort, seed.inputs, out, seed.out)
    assert all((out / name).read_bytes() == (seed.out / name).read_bytes() for name in _REQUIRED)
    plain, compared = _marker(seed.out), _marker(out)
    assert plain["measurement"] == compared["measurement"]
    assert plain["sensitivity"] is None
    assert not (seed.out / _SENSITIVITY).exists()
    assert compared["sensitivity"] is not None
    assert (out / _SENSITIVITY).is_file()
    assert plain["generation"]["tree_digest"] != compared["generation"]["tree_digest"]
    compared["sensitivity"] = None
    compared["generation"]["tree_digest"] = plain["generation"]["tree_digest"]
    assert compared == plain


def test_p05_replacement_without_baseline_removes_prior_sensitivity(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort()
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, seed.out)
    assert (out / _SENSITIVITY).is_file()
    _publish(cohort, seed.inputs, out)
    assert _marker(out)["sensitivity"] is None
    assert not (out / _SENSITIVITY).exists()
    assert _snapshot(out) == _snapshot(seed.out)


def _comparison_baseline(
    seed: _Seed, root: pathlib.Path
) -> tuple[pathlib.Path, list[dict[str, str]]]:
    cohort = _cohort()
    baseline = _copy_output(seed, root)
    _legacy(cohort, baseline, "v1")
    header, rows = _read_csv(baseline / _FEATURES)
    assert len(rows) >= 9
    rows[0] |= {"median": "-1.125000001", "q25": "0.125000000", "q75": "0.875000000"}
    rows[1] |= {"median": "1.000000000", "q25": "2.000000000", "q75": "2.000000000"}
    rows[2] |= {"median": "5.000000000", "q25": "7.000000000", "q75": "3.000000000"}
    rows[3] |= {"median": "", "q25": "1.000000000", "q75": "4.000000000"}
    rows[4] |= {"median": "1.000000000", "q25": "", "q75": "4.000000000"}
    rows[5] |= {"n_subjects": str(cohort.SUBJECT_FLOOR - 1), **dict.fromkeys(_DISTRIBUTION, "")}
    rows[6]["feature"] = "synthetic_retired_feature"
    rows[7] |= {"median": "1.000000000", "q25": "2.000000000", "q75": ""}
    _csv(baseline / _FEATURES, header, list(reversed(rows)))
    _resign(cohort, baseline)
    return baseline, rows


def test_p06_union_order_status_and_counts(seed: _Seed, tmp_path: pathlib.Path) -> None:
    cohort = _cohort()
    baseline, old_rows = _comparison_baseline(seed, tmp_path / "baseline")
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, baseline)
    header, rows = _read_csv(out / _SENSITIVITY)
    _, current = _read_csv(out / _FEATURES)
    assert {row["task"] for row in current} == set(inventory.TASKS)
    assert {row["side"] for row in current} == set(inventory.SIDES)
    assert {row["level"] for row in current} == set(cohort.LEVELS)
    left, right = {_key(row) for row in current}, {_key(row) for row in old_rows}
    assert header == _SENSITIVITY_COLUMNS == tuple(cohort.SENSITIVITY_COLUMNS)
    assert cohort.SENSITIVITY_FILENAME == _SENSITIVITY
    assert [_key(row) for row in rows] == sorted(left | right)
    for row in rows:
        key = _key(row)
        assert row["status"] == (
            "common" if key in left & right else "new" if key in left else "retired"
        )
    counts = dict(Counter(row["status"] for row in rows))
    assert set(counts) == {"common", "new", "retired"}
    assert _marker(out)["sensitivity"]["status_counts"] == counts


def test_p07_exact_serialized_arithmetic_and_blank_dependencies(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort()
    baseline, old_rows = _comparison_baseline(seed, tmp_path / "baseline")
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, baseline)
    _, actual = _read_csv(out / _SENSITIVITY)
    _, current = _read_csv(out / _FEATURES)
    assert actual == _oracle(current, old_rows, cohort.DECIMALS)
    common = [row for row in actual if row["status"] == "common"]
    assert any(row["delta_median"] and Decimal(row["delta_median"]) != 0 for row in common)
    assert any(row["baseline_iqr"] and Decimal(row["baseline_iqr"]) == 0 for row in common)
    assert any(row["baseline_iqr"] and Decimal(row["baseline_iqr"]) < 0 for row in common)
    assert any(int(row["baseline_n_subjects"]) < cohort.SUBJECT_FLOOR for row in common)


def test_p07_current_below_floor_keeps_baseline_iqr(seed: _Seed, tmp_path: pathlib.Path) -> None:
    cohort = _cohort()
    inputs = _copy_inputs(seed, tmp_path / "inputs")
    _, feature_rows = _read_csv(seed.out / _FEATURES)
    target = next(row for row in feature_rows if row["level"] == "window")
    assets = {
        asset.asset_id
        for asset in inputs.assets
        if (asset.task, asset.side, asset.subject_ordinal) == (target["task"], target["side"], 5)
    }
    manifest = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    assert assets
    for row in manifest:
        if row["asset_id"] in assets:
            path = inputs.run / row["event_id"] / f"{row['camera_name']}_clinical_windows.csv"
            header, values = _read_csv(path)
            for value in values:
                value[target["feature"]] = "NA"
            _csv(path, header, values)
    out = tmp_path / "out"
    _run(cohort, inputs, out, seed.out)
    _, rows = _read_csv(out / _SENSITIVITY)
    actual = next(row for row in rows if _key(row) == _key(target))
    assert actual["n_subjects"] == str(cohort.SUBJECT_FLOOR - 1)
    assert actual["median"] == actual["delta_median"] == actual["delta_iqr_scaled"] == ""
    assert actual["baseline_median"] == target["median"]
    assert (
        actual["baseline_iqr"]
        == f"{Decimal(target['q75']) - Decimal(target['q25']):.{cohort.DECIMALS}f}"
    )


def test_p07_generated_arithmetic_matches_decimal_oracle(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort()
    baseline = _copy_output(seed, tmp_path / "baseline")
    _legacy(cohort, baseline, "v2")
    header, old_rows = _read_csv(baseline / _FEATURES)
    rng = random.Random(47107)  # noqa: S311 — repeatable generated inputs
    for index, row in enumerate(old_rows):
        low = Decimal(rng.randint(-800, 800)) / 8
        width = Decimal(rng.choice((-8, -1, 0, 1, 2, 3, 4, 8, 16))) / 8
        row |= {
            "median": str(Decimal(rng.randint(-800, 800)) / 8),
            "q25": str(low),
            "q75": str(low + width),
        }
        if index % 11 == 0:
            row[rng.choice(("median", "q25", "q75"))] = ""
    _csv(baseline / _FEATURES, header, old_rows)
    _resign(cohort, baseline)
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, baseline)
    _, current = _read_csv(out / _FEATURES)
    _, actual = _read_csv(out / _SENSITIVITY)
    assert len(actual) == len(current) > 100
    assert actual == _oracle(current, old_rows, cohort.DECIMALS)


_BASELINE_CASES = [
    *(f"marker:{kind}" for kind in _JSON_DAMAGE),
    "marker:generator",
    "marker:version",
    "marker:digest-null",
    "marker:digest-short",
    "marker:digest-upper",
    "marker:digest-nonhex",
    "marker:digest-newline",
    "marker:digest-mismatch",
    "features:absent",
    "features:symlink",
    "features:directory",
    "features:header-only",
    "features:duplicate-header",
    "features:short-row",
    "features:long-row",
    "features:duplicate-key",
    *(f"features:missing-{name}" for name in _REQUIRED_FEATURES),
    *(f"features:empty-{name}" for name in _KEY),
    *(f"features:domain-{name}" for name in ("task", "side", "level")),
    *(f"features:feature-{kind}" for kind in ("upper", "dash", "unicode", "newline")),
    *(
        f"features:{name}-{kind}"
        for name in ("n_subjects", "n_values")
        for kind in ("empty", "negative", "decimal", "unicode", "space", "newline")
    ),
    *(
        f"features:{name}-{kind}"
        for name in ("median", "q25", "q75", "mean", "sd")
        for kind in ("nan", "inf", "nonnumeric")
    ),
    *(f"features:below-floor-{name}" for name in _DISTRIBUTION),
]


def _damage_baseline(cohort: ModuleType, baseline: pathlib.Path, case: str) -> None:
    owner, kind = case.split(":")
    path = baseline / ("cohort.json" if owner == "marker" else _FEATURES)
    if kind == "absent":
        path.unlink()
        return
    if kind == "symlink":
        external = baseline.parent / f"external-{path.name}"
        path.rename(external)
        path.symlink_to(external)
        return
    if kind == "directory":
        path.unlink()
        path.mkdir()
        return
    if owner == "marker":
        marker = _marker(baseline)
        if kind == "invalid-json":
            path.write_text("{", encoding="utf-8")
        elif kind == "invalid-utf8":
            path.write_bytes(b"\xff")
        elif kind == "duplicate-key":
            path.write_text(
                '{"generation":null,' + path.read_text(encoding="utf-8").lstrip()[1:],
                encoding="utf-8",
            )
        elif kind in {"array", "null", "string"}:
            _json(path, {"array": [], "null": None, "string": "marker"}[kind])
        elif kind in {"generator", "version"}:
            marker["generation"]["generator" if kind == "generator" else "generator_version"] = (
                "foreign"
            )
            _json(path, marker)
            _resign(cohort, baseline)
        elif kind == "digest-mismatch":
            feature = baseline / _FEATURES
            feature.write_bytes(feature.read_bytes() + b"\n")
        else:
            marker["generation"]["tree_digest"] = {
                "digest-null": None,
                "digest-short": "0" * 63,
                "digest-upper": "A" * 64,
                "digest-nonhex": "g" * 64,
                "digest-newline": "a" * 64 + "\n",
            }[kind]
            _json(path, marker)
        return
    header, rows = _read_csv(path)
    raw = [list(header), *[[row[name] for name in header] for row in rows]]
    if kind == "header-only":
        raw = raw[:1]
    elif kind == "duplicate-header":
        raw[0][-1] = raw[0][0]
    elif kind == "short-row":
        raw[1].pop()
    elif kind == "long-row":
        raw[1].append("extra")
    elif kind == "duplicate-key":
        raw.append(raw[1].copy())
    elif kind.startswith("missing-"):
        index = header.index(kind.removeprefix("missing-"))
        for row in raw:
            row.pop(index)
    elif kind.startswith("below-floor-"):
        field = kind.removeprefix("below-floor-")
        raw[1][header.index("n_subjects")] = str(cohort.SUBJECT_FLOOR - 1)
        for name in _DISTRIBUTION:
            raw[1][header.index(name)] = "1.000000000" if name == field else ""
    else:
        field, variant = kind.split("-", 1)
        if field == "empty":
            field, value = variant, ""
        elif field == "domain":
            field, value = variant, "foreign"
        elif field == "feature":
            value = {
                "upper": "Feature",
                "dash": "feature-x",
                "unicode": "特徴",
                "newline": "feature\n",
            }[variant]
        else:
            value = {
                "empty": "",
                "negative": "-1",
                "decimal": "1.0",
                "unicode": "\uff15",
                "space": " 5",
                "newline": "5\n",
                "nan": "NaN",
                "inf": "Infinity",
                "nonnumeric": "text",
            }[variant]
        raw[1][header.index(field)] = value
    _raw_csv(path, raw)
    _resign(cohort, baseline)


@pytest.mark.parametrize("case", _BASELINE_CASES)
@pytest.mark.parametrize("existing", [False, True], ids=["new-output", "owned-output"])
def test_p08_baseline_refusals_preserve_destination(
    seed: _Seed, tmp_path: pathlib.Path, case: str, existing: bool
) -> None:
    cohort = _cohort()
    baseline = _copy_output(seed, tmp_path / "baseline")
    _legacy(cohort, baseline, "v2")
    out = tmp_path / "out"
    if existing:
        _copy_output(seed, out)
    _damage_baseline(cohort, baseline, case)
    _assert_refused(cohort, seed.inputs, out, baseline)


@pytest.mark.parametrize("direction", ["baseline-contains-output", "output-contains-baseline"])
@pytest.mark.parametrize("alias", [False, True], ids=["direct", "symlink-alias"])
def test_p08_strict_containment_refuses_before_any_write(
    seed: _Seed, tmp_path: pathlib.Path, direction: str, alias: bool
) -> None:
    cohort = _cohort()
    outer = _copy_output(seed, tmp_path / "outer")
    inner = _copy_output(seed, outer / "inner")
    out, baseline = (inner, outer) if direction == "baseline-contains-output" else (outer, inner)
    if alias:
        link = tmp_path / "alias"
        link.symlink_to(baseline, target_is_directory=True)
        baseline = link
    before = _snapshot(outer)
    _assert_refused(cohort, seed.inputs, out, baseline)
    assert _snapshot(outer) == before


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_p08_supported_baseline_profiles_and_extra_legacy_columns(
    seed: _Seed, tmp_path: pathlib.Path, version: str
) -> None:
    cohort = _cohort()
    baseline = _copy_output(seed, tmp_path / "baseline")
    if version != "v3":
        _legacy(cohort, baseline, version)
        header, rows = _read_csv(baseline / _FEATURES)
        for row in rows:
            row["legacy_statistic"] = "1.250000000"
        _csv(baseline / _FEATURES, (*header, "legacy_statistic"), rows)
        _resign(cohort, baseline)
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, baseline)
    cohort.validate_generation(out)
    assert _marker(out)["sensitivity"]["baseline"]["generator_version"] == version


def test_p09_baseline_identity_names_exact_parsed_bytes(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort()
    baseline = _copy_output(seed, tmp_path / "baseline")
    _legacy(cohort, baseline, "v1")
    expected = {
        "generator_version": "v1",
        "tree_digest": _marker(baseline)["generation"]["tree_digest"],
        "features_sha256": hashlib.sha256((baseline / _FEATURES).read_bytes()).hexdigest(),
    }
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, baseline)
    block = _marker(out)["sensitivity"]
    assert set(block) == {"baseline", "status_counts", "estimand"}
    assert block["baseline"] == expected
    assert block["estimand"] == cohort.SENSITIVITY_ESTIMAND
    assert isinstance(block["estimand"], str)
    assert block["estimand"].strip()


@pytest.mark.parametrize("damage", ["edit", "append", "remove", "directory", "symlink"])
def test_p09_optional_file_tampering_is_detected(
    seed: _Seed, tmp_path: pathlib.Path, damage: str
) -> None:
    cohort = _cohort()
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, seed.out)
    cohort.validate_generation(out)
    path = out / _SENSITIVITY
    if damage == "edit":
        data = path.read_bytes()
        assert b"common" in data
        path.write_bytes(data.replace(b"common", b"retired", 1))
    elif damage == "append":
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "remove":
        path.unlink()
    elif damage == "directory":
        path.unlink()
        path.mkdir()
    else:
        external = tmp_path / "external.csv"
        path.rename(external)
        path.symlink_to(external)
    with pytest.raises(cohort.CohortError):
        cohort.validate_generation(out)


def test_p09_optional_file_added_under_null_is_rejected(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    cohort = _cohort()
    out = _copy_output(seed, tmp_path / "out")
    (out / _SENSITIVITY).write_text(",".join(_SENSITIVITY_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(cohort.CohortError):
        cohort.validate_generation(out)


def test_p09_optional_presence_is_in_tree_digest(seed: _Seed, tmp_path: pathlib.Path) -> None:
    cohort = _cohort()
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, seed.out)
    marker = _marker(out)
    before = cohort.tree_digest(out, marker)
    (out / _SENSITIVITY).unlink()
    try:
        after = cohort.tree_digest(out, marker)
    except cohort.CohortError:
        return
    assert after != before


def test_p09_v3_baseline_may_itself_carry_sensitivity(seed: _Seed, tmp_path: pathlib.Path) -> None:
    cohort = _cohort()
    baseline = tmp_path / "baseline"
    _run(cohort, seed.inputs, baseline, seed.out)
    out = tmp_path / "out"
    _run(cohort, seed.inputs, out, baseline)
    cohort.validate_generation(out)
    assert (
        _marker(out)["sensitivity"]["baseline"]["tree_digest"]
        == _marker(baseline)["generation"]["tree_digest"]
    )


def test_p10_baseline_is_recursively_byte_and_link_read_only(
    seed: _Seed, tmp_path: pathlib.Path
) -> None:
    baseline = _copy_output(seed, tmp_path / "baseline")
    before = _snapshot(baseline)
    assert set(before) == {".", "cohort.json", *_REQUIRED}
    assert all(kind == "file" for key, (kind, _) in before.items() if key != ".")
    _run(_cohort(), seed.inputs, tmp_path / "out", baseline)
    assert _snapshot(baseline) == before


def test_p10_mutation_during_baseline_read_refuses(
    seed: _Seed, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _cohort()
    baseline = _copy_output(seed, tmp_path / "baseline")
    target = baseline / "cohort.json"
    # Whitespace distinguishes this read from the byte-identical destination marker.
    target.write_text(
        json.dumps(_marker(baseline), sort_keys=True, indent=4) + "\n", encoding="utf-8"
    )
    target_bytes = target.read_bytes()
    out = _copy_output(seed, tmp_path / "out")
    before = _snapshot(out)
    original = json.loads
    mutated = False

    def loads(data: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal mutated
        payload = original(data, *args, **kwargs)
        raw = data.encode() if isinstance(data, str) else bytes(data)
        if not mutated and raw == target_bytes:
            target.write_bytes(target_bytes + b"\n")
            mutated = True
        return payload

    monkeypatch.setattr(json, "loads", loads)
    try:
        with pytest.raises(cohort.CohortError):
            _run(cohort, seed.inputs, out, baseline)
    finally:
        assert _snapshot(out) == before
    assert mutated, "mutation must occur during the baseline's JSON read"
    assert target.read_bytes() != target_bytes


@pytest.mark.parametrize("alias", [False, True], ids=["same-path", "symlink-alias"])
def test_p11_equal_root_reads_generation_being_replaced(
    seed: _Seed, tmp_path: pathlib.Path, alias: bool
) -> None:
    cohort = _cohort()
    out = _copy_output(seed, tmp_path / "out")
    previous = _marker(out)["generation"]["tree_digest"]
    features = (out / _FEATURES).read_bytes()
    baseline = tmp_path / "alias" if alias else out
    if alias:
        baseline.symlink_to(out, target_is_directory=True)
    _run(cohort, seed.inputs, out, baseline)
    cohort.validate_generation(out)
    assert (out / _FEATURES).read_bytes() == features
    block = _marker(out)["sensitivity"]
    assert block["baseline"]["tree_digest"] == previous
    assert block["baseline"]["features_sha256"] == hashlib.sha256(features).hexdigest()
    _, rows = _read_csv(out / _SENSITIVITY)
    assert rows
    assert block["status_counts"] == {"common": len(rows), "new": 0, "retired": 0}
    assert all(row["status"] == "common" for row in rows)
    assert all(not row["delta_median"] or Decimal(row["delta_median"]) == 0 for row in rows)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_p11_legacy_equal_root_refuses_unchanged(
    seed: _Seed, tmp_path: pathlib.Path, version: str
) -> None:
    cohort = _cohort()
    out = _copy_output(seed, tmp_path / "out")
    _legacy(cohort, out, version)
    _assert_refused(cohort, seed.inputs, out, out)


@pytest.mark.parametrize("phase", ["before-swap", "promotion"])
def test_p11_equal_root_swap_failure_preserves_previous_generation(
    seed: _Seed, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    cohort = _cohort()
    out = _copy_output(seed, tmp_path / "out")
    before = _snapshot(out)
    attempted = False

    def inject(operation: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal attempted
            src, dst = pathlib.Path(source), pathlib.Path(target)
            match = src == out if phase == "before-swap" else src != out and dst == out
            if not attempted and src.parent == out.parent and match:
                attempted = True
                raise OSError("M4.7 injected swap failure")
            return operation(source, target, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(os, "rename", inject(os.rename))
    monkeypatch.setattr(os, "replace", inject(os.replace))
    try:
        with pytest.raises((OSError, cohort.CohortError)):
            _run(cohort, seed.inputs, out, out)
    finally:
        assert _snapshot(out) == before
    assert attempted, "injected failure must reach the swap"
    cohort.validate_generation(out)


def test_p12_version_and_required_membership_are_v3() -> None:
    cohort = _cohort()
    assert cohort.GENERATOR_VERSION == "v3"
    assert set(cohort.PUBLISHED_FILENAMES) == set(_REQUIRED)
    assert {"measurement", "sensitivity"} <= set(cohort.MARKER_KEYS)


def test_p12_v2_destination_is_foreign(seed: _Seed, tmp_path: pathlib.Path) -> None:
    cohort = _cohort()
    out = _copy_output(seed, tmp_path / "out")
    _legacy(cohort, out, "v2")
    _assert_refused(cohort, seed.inputs, out)


def test_p12_empty_destination_accepts_v3(seed: _Seed, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _publish(_cohort(), seed.inputs, out)
    assert _marker(out)["generation"]["generator_version"] == "v3"


@pytest.mark.parametrize(
    "transform", ["moved-output", "relocated-baseline", "reordered-clinical", "reordered-manifest"]
)
def test_p13_reproducible_with_fixed_baseline(
    seed: _Seed, tmp_path: pathlib.Path, transform: str
) -> None:
    cohort = _cohort()
    baseline = _copy_output(seed, tmp_path / "baseline")
    reference = tmp_path / "reference"
    _run(cohort, seed.inputs, reference, baseline)
    inputs = seed.inputs
    if transform == "relocated-baseline":
        relocated = tmp_path / "relocated"
        shutil.copytree(baseline, relocated, symlinks=True)
        assert _snapshot(relocated) == _snapshot(baseline)
        baseline = relocated
    elif transform.startswith("reordered-"):
        inputs = _copy_inputs(seed, tmp_path / "inputs")
        paths = (
            [inputs.run / corpus_run.MANIFEST_FILENAME]
            if transform == "reordered-manifest"
            else list(inputs.run.glob("*/*_clinical*.csv"))
        )
        assert paths
        changed = 0
        for path in paths:
            header, rows = _read_csv(path)
            if rows != list(reversed(rows)):
                changed += 1
            _csv(path, header, list(reversed(rows)))
        assert changed > 0, "row-order transform must alter source bytes"
    out = tmp_path / "nested" / "moved-output"
    _run(cohort, inputs, out, baseline)
    cohort.validate_generation(out)
    if transform != "reordered-manifest":
        assert _snapshot(out) == _snapshot(reference)
        return
    assert set(_snapshot(out)) == set(_snapshot(reference))
    assert all(
        (out / name).read_bytes() == (reference / name).read_bytes()
        for name in (*_REQUIRED, _SENSITIVITY)
    )
    old, new = _marker(reference), _marker(out)
    digests = []
    trees = []
    for marker, run in ((old, seed.inputs.run), (new, inputs.run)):
        digest = marker["generation"]["input_digests"].pop("run_manifest")
        assert (
            digest == hashlib.sha256((run / corpus_run.MANIFEST_FILENAME).read_bytes()).hexdigest()
        )
        digests.append(digest)
        trees.append(marker["generation"].pop("tree_digest"))
    assert digests[0] != digests[1]
    assert trees[0] != trees[1]
    assert new == old


@pytest.mark.parametrize("builder", ["registry", "placements"])
def test_p13_reversed_input_builders_preserve_all_output_bytes(
    seed: _Seed, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    cohort = _cohort()
    fixtures = importlib.import_module("test_cohort")
    root = tmp_path / "inputs"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    _metadata(inputs)
    manifest_bytes = (inputs.run / corpus_run.MANIFEST_FILENAME).read_bytes()
    manifest_rows = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    reference = tmp_path / "reference"
    _run(cohort, inputs, reference, seed.out)
    shutil.rmtree(root)
    root.mkdir()
    reversed_rows = False
    registry = fixtures._write_registry
    read_csv = fixtures._read_csv

    def reverse_registry(path: pathlib.Path, assets: list[Any]) -> Any:
        nonlocal reversed_rows
        reordered = list(reversed(assets))
        reversed_rows = reordered != assets
        return registry(path, reordered)

    def reverse_placements(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
        nonlocal reversed_rows
        header, rows = read_csv(path)
        if path.name == "placements.csv":
            reordered = list(reversed(rows))
            reversed_rows = reordered != rows
            rows = reordered
        return header, rows

    if builder == "registry":
        monkeypatch.setattr(fixtures, "_write_registry", reverse_registry)
    else:
        monkeypatch.setattr(fixtures, "_read_csv", reverse_placements)
    reordered = _build_inputs(root, cohort)
    assert reversed_rows, "builder-order transform must actually reverse records"
    _metadata(reordered)
    manifest_path = reordered.run / corpus_run.MANIFEST_FILENAME
    assert sorted(
        corpus_run.read_manifest(manifest_path), key=lambda row: row["asset_id"]
    ) == sorted(manifest_rows, key=lambda row: row["asset_id"])
    # Manifest-order behavior has its own case; keep that byte witness fixed here.
    manifest_path.write_bytes(manifest_bytes)
    out = tmp_path / "out"
    _run(cohort, reordered, out, seed.out)
    assert _snapshot(out) == _snapshot(reference)


def test_p13_cli_baseline_reaches_the_publisher(seed: _Seed, tmp_path: pathlib.Path) -> None:
    cohort = _cohort()
    out = tmp_path / "cli-output"
    result = cohort.main(
        [
            "--inventory",
            str(seed.inputs.inventory),
            "--sessions",
            str(seed.inputs.sessions),
            "--run",
            str(seed.inputs.run),
            "--out",
            str(out),
            "--baseline",
            str(seed.out),
        ]
    )
    assert result == 0
    cohort.validate_generation(out)
    assert (
        _marker(out)["sensitivity"]["baseline"]["tree_digest"]
        == _marker(seed.out)["generation"]["tree_digest"]
    )
