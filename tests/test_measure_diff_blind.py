"""Diff-blind acceptance tests for the measurement sidecar contract."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import inspect
import itertools
import json
import pathlib
import wave
from typing import Any

import numpy as np
import pytest

from pose_estimation import inventory, measure, qualify
from pose_estimation.measure import audio_offset, sync, visual_offset
from test_qualify import _publish, _rows, _uniform, _write_media
from test_sessions import _canonical, _write_registry

AUDIO_STRENGTH_COLUMN = (
    "peak_rms_audio" if "peak_rms_audio" in measure.AXES["sync"].columns else "conf_audio"
)
AUDIO_REJECTED_STATUS = next(status for status in sorted(measure.AUDIO_STATUSES) if status != "ok")
VISUAL_REJECTED_STATUS = next(
    status for status in sorted(measure.VISUAL_STATUSES) if status != "ok"
)
SCALE_CONFIDENCE_CELL = (
    "none" if measure.CELL_ALPHABETS["scale_ref_conf"].fullmatch("none") else "0.000000000"
)


@dataclasses.dataclass(frozen=True)
class World:
    inventory_dir: pathlib.Path
    measurements: pathlib.Path
    family: tuple[dict[str, str], ...]
    outsider: dict[str, str]
    assets: tuple[dict[str, str], ...]


@dataclasses.dataclass(frozen=True)
class QualificationWorld:
    inventory_dir: pathlib.Path
    sessions_dir: pathlib.Path
    corpus: pathlib.Path
    out: pathlib.Path
    measurements: pathlib.Path
    family: tuple[dict[str, str], ...]
    outsider: dict[str, str]

    def measurement_world(self) -> World:
        return World(
            self.inventory_dir,
            self.measurements,
            self.family,
            self.outsider,
            (*self.family, self.outsider),
        )


@pytest.fixture
def world(tmp_path: pathlib.Path) -> World:
    registry = _write_registry(
        tmp_path,
        [
            _canonical(1, "above"),
            _canonical(1, "left"),
            _canonical(1, "right"),
            _canonical(2, "above"),
        ],
    )
    rows = tuple(
        row
        for row in csv.DictReader(
            (registry.root / inventory.ASSETS_FILENAME).read_text(encoding="utf-8").splitlines()
        )
        if row["disposition"] == inventory.CANONICAL
    )
    by_capture: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_capture.setdefault(row["capture_id"], []).append(row)
    family = tuple(sorted(max(by_capture.values(), key=len), key=lambda row: row["asset_id"]))
    outsider = next(row for row in rows if row["capture_id"] != family[0]["capture_id"])
    return World(registry.root, tmp_path / "measurements", family, outsider, rows)


@pytest.fixture
def qualification_world(tmp_path: pathlib.Path) -> QualificationWorld:
    assets = [
        _canonical(1, "above"),
        _canonical(1, "left"),
        _canonical(2, "above"),
    ]
    inventory_dir, sessions_dir, corpus, out = _publish(tmp_path, assets)
    for asset in assets:
        _write_media(corpus / asset.source_path, _uniform(30))
    rows = tuple(
        csv.DictReader(
            (inventory_dir / inventory.ASSETS_FILENAME).read_text(encoding="utf-8").splitlines()
        )
    )
    by_capture: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_capture.setdefault(row["capture_id"], []).append(row)
    family = tuple(sorted(max(by_capture.values(), key=len), key=lambda row: row["asset_id"]))
    outsider = next(row for row in rows if row["capture_id"] != family[0]["capture_id"])
    return QualificationWorld(
        inventory_dir,
        sessions_dir,
        corpus,
        out,
        tmp_path / "measurements",
        family,
        outsider,
    )


def _sync_row(
    world: World,
    pair: tuple[dict[str, str], dict[str, str]] | None = None,
    /,
    **updates: str,
) -> dict[str, str]:
    left, right = pair or (world.family[0], world.family[1])
    left, right = sorted((left, right), key=lambda row: row["asset_id"])
    row = {
        "capture_id": left["capture_id"],
        "asset_a": left["asset_id"],
        "asset_b": right["asset_id"],
        "offset_audio_s": "-0.125000000",
        AUDIO_STRENGTH_COLUMN: "5.000000000",
        "peak_ratio_audio": "2.500000000",
        "status_audio": "ok",
        "drift_ppm": "",
        "drift_se": "",
        "offset_visual_s": "-0.124000000",
        "conf_visual": "4.500000000",
        "peak_corr_visual": "0.800000000",
        "status_visual": "ok",
        "overlap_s": "4.000000000",
        "dur_a": "5.000000000",
        "dur_b": "5.000000000",
        "same_audio_rate": "1",
    }
    row.update(updates)
    return row


def _asset_axis_row(axis_name: str, asset_id: str) -> dict[str, str]:
    if axis_name == "rigidity":
        return {
            "asset_id": asset_id,
            "rigidity_drift_median_px": "1.000000000",
            "rigidity_drift_p95_px": "2.000000000",
            "rigidity_valid_fraction": "0.900000000",
            "rigidity_flag": "rigid",
        }
    if axis_name == "detect":
        return {
            "asset_id": asset_id,
            "detect_rate": "1.000000000",
            "detect_conf_median": "0.900000000",
            "subject_px_height_median": "100.000000000",
        }
    if axis_name == "scale":
        return {
            "asset_id": asset_id,
            "scale_ref_class": "none",
            "scale_ref_conf": SCALE_CONFIDENCE_CELL,
        }
    raise AssertionError(axis_name)


def _all_sync_rows(world: World) -> list[dict[str, str]]:
    return [_sync_row(world, pair) for pair in itertools.combinations(world.family, 2)]


def _write_sync(world: World, rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return measure.write_axis(
        world.measurements,
        "sync",
        [_sync_row(world)] if rows is None else rows,
        {"fixture": "synthetic", "analysis_resolution_s": "0.000000001"},
        inventory_dir=world.inventory_dir,
    )


def _manifest(root: pathlib.Path) -> dict[str, Any]:
    return json.loads((root / measure.MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _write_manifest(root: pathlib.Path, manifest: dict[str, Any]) -> None:
    manifest["generation"]["manifest"] = measure.manifest_digest(manifest)
    (root / measure.MANIFEST_FILENAME).write_text(
        inventory.render_json(manifest), encoding="utf-8", newline=""
    )


def _rewrite_axis_bytes(
    root: pathlib.Path,
    axis_name: str,
    body: bytes,
    *,
    rows: int,
) -> None:
    manifest = _manifest(root)
    axis = measure.AXES[axis_name]
    (root / axis.table).write_bytes(body)
    entry = manifest["axes"][axis_name]
    entry["sha256"] = hashlib.sha256(body).hexdigest()
    entry["rows"] = rows
    _write_manifest(root, manifest)


def _render_rows(columns: tuple[str, ...], rows: list[dict[str, str]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=columns,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _rewrite_axis_rows(
    root: pathlib.Path,
    axis_name: str,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] | None = None,
    declared_rows: int | None = None,
) -> None:
    axis = measure.AXES[axis_name]
    text = _render_rows(columns or axis.columns, rows)
    _rewrite_axis_bytes(
        root,
        axis_name,
        text.encode(),
        rows=len(rows) if declared_rows is None else declared_rows,
    )


def _load(world: World, axis_name: str = "sync") -> dict[tuple[str, ...], dict[str, str]]:
    sidecar = measure.validate(world.measurements, inventory_dir=world.inventory_dir)
    return measure.load_axis(sidecar, axis_name)


def _write_qualification_sync(world: QualificationWorld, **updates: str) -> dict[str, str]:
    measurement_world = world.measurement_world()
    row = _sync_row(measurement_world, **updates)
    measure.write_axis(
        world.measurements,
        "sync",
        [row],
        {"fixture": "qualification-policy"},
        inventory_dir=world.inventory_dir,
    )
    return row


def _qualify_arguments(world: QualificationWorld, out: pathlib.Path | None = None) -> list[str]:
    return [
        "--inventory",
        str(world.inventory_dir),
        "--sessions",
        str(world.sessions_dir),
        "--corpus",
        str(world.corpus),
        "--out",
        str(out or world.out),
        "--measurements",
        str(world.measurements),
    ]


def _encode_with_duplicate(value: Any, target: tuple[str, ...], path: tuple[str, ...] = ()) -> str:
    if isinstance(value, dict):
        fields: list[str] = []
        for key, child in value.items():
            child_path = (*path, key)
            encoded = f"{json.dumps(key)}:{_encode_with_duplicate(child, target, child_path)}"
            fields.append(encoded)
            if child_path == target:
                fields.append(encoded)
        return "{" + ",".join(fields) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_encode_with_duplicate(child, target, path) for child in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _foreign_asset_id(world: World) -> str:
    used = {row["asset_id"] for row in world.assets}
    return next(candidate for candidate in ("0" * 16, "f" * 16) if candidate not in used)


def _write_wav(path: pathlib.Path, frequency_hz: float, *, rate: int = 16000) -> None:
    sample_index = np.arange(rate * 2)
    samples = (np.sin(2 * np.pi * frequency_hz * sample_index / rate) * 10000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


def test_g2_load_axis_reads_the_bytes_that_validate_digested(world: World) -> None:
    original = _sync_row(world)
    _write_sync(world, [original])
    sidecar = measure.validate(world.measurements, inventory_dir=world.inventory_dir)
    changed = {**original, "offset_audio_s": "9.000000000"}
    _rewrite_axis_rows(world.measurements, "sync", [changed])

    loaded = measure.load_axis(sidecar, "sync")

    assert next(iter(loaded.values()))["offset_audio_s"] == original["offset_audio_s"]
    assert len(inspect.signature(measure.load_axis).parameters) == 2


def test_g3_manifest_symlink_is_refused(world: World) -> None:
    _write_sync(world)
    manifest = world.measurements / measure.MANIFEST_FILENAME
    target = world.measurements.parent / "manifest-target.json"
    manifest.rename(target)
    manifest.symlink_to(target)

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


def test_axis_table_symlink_is_refused(world: World) -> None:
    _write_sync(world)
    table = world.measurements / measure.AXES["sync"].table
    target = world.measurements.parent / "table-target.csv"
    table.rename(target)
    table.symlink_to(target)

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


@pytest.mark.parametrize(
    "target",
    [
        ("axes",),
        ("axes", "sync", "rows"),
        ("axes", "sync", "provenance", "fixture"),
        ("generation", "manifest"),
    ],
)
def test_g3_duplicate_json_key_at_any_depth_is_refused(
    world: World, target: tuple[str, ...]
) -> None:
    _write_sync(world)
    manifest = _manifest(world.measurements)
    raw = _encode_with_duplicate(manifest, target)
    (world.measurements / measure.MANIFEST_FILENAME).write_text(raw, encoding="utf-8")

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("offset_audio_s", "NaN"),
        ("offset_audio_s", "inf"),
        ("offset_audio_s", "-1e-3"),
        ("offset_audio_s", "+0.1"),
        ("offset_audio_s", ".1"),
        ("offset_audio_s", "1."),
        ("offset_audio_s", "1"),
        ("offset_audio_s", " 0.100000000"),
        ("offset_audio_s", f"{chr(0x660)}.{chr(0x661)}"),
        (AUDIO_STRENGTH_COLUMN, "-0.100000000"),
        ("peak_ratio_audio", "-0.100000000"),
        ("drift_se", "-0.100000000"),
        ("conf_visual", "-0.100000000"),
        ("peak_corr_visual", "1.100000000"),
        ("peak_corr_visual", "-1.100000000"),
        ("overlap_s", "-0.100000000"),
        ("dur_a", "-0.100000000"),
        ("dur_b", "-0.100000000"),
        ("same_audio_rate", "2"),
        ("same_audio_rate", "true"),
        ("same_audio_rate", "01"),
    ],
)
def test_g1_ingestion_rejects_invalid_numeric_cells(world: World, column: str, value: str) -> None:
    _write_sync(world)
    row = _sync_row(world, **{column: value})
    _rewrite_axis_rows(world.measurements, "sync", [row])

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize(
    ("axis_name", "column", "value"),
    [
        ("rigidity", "rigidity_drift_median_px", "NaN"),
        ("rigidity", "rigidity_flag", "BAD|flag"),
        ("detect", "detect_rate", "inf"),
        ("detect", "detect_conf_median", "-1e-3"),
        ("scale", "scale_ref_class", "BAD"),
        ("scale", "scale_ref_conf", "NaN"),
    ],
)
def test_g1_non_sync_axes_revalidate_their_cell_alphabets(
    world: World, axis_name: str, column: str, value: str
) -> None:
    row = _asset_axis_row(axis_name, world.family[0]["asset_id"])
    measure.write_axis(
        world.measurements,
        axis_name,
        [row],
        {"fixture": "synthetic"},
        inventory_dir=world.inventory_dir,
    )
    row[column] = value
    _rewrite_axis_rows(world.measurements, axis_name, [row])

    with pytest.raises(measure.MeasureError):
        _load(world, axis_name)


@pytest.mark.parametrize("axis_name", ["rigidity", "detect", "scale"])
def test_g1_non_sync_axes_require_the_exact_header(world: World, axis_name: str) -> None:
    row = _asset_axis_row(axis_name, world.family[0]["asset_id"])
    measure.write_axis(
        world.measurements,
        axis_name,
        [row],
        {"fixture": "synthetic"},
        inventory_dir=world.inventory_dir,
    )
    _rewrite_axis_rows(
        world.measurements,
        axis_name,
        [row],
        columns=measure.AXES[axis_name].columns[:-1],
    )

    with pytest.raises(measure.MeasureError):
        _load(world, axis_name)


@pytest.mark.parametrize(
    ("scale_class", "confidence"),
    [
        ("none", "none"),
        ("closure", "class_only"),
        ("coin", "class_only"),
        ("vessel", "class_only"),
        ("key", "class_only"),
        ("nut", "class_only"),
        ("peg", "class_only"),
        ("anthropometric", "class_only"),
        ("furniture", "class_only"),
        ("calibration_target", "class_only"),
        ("coin", "variant_verified"),
        ("coin", "dimension_verified"),
    ],
)
def test_r3_future_exhaustive_scale_axis_uses_the_ruled_token_alphabets(
    world: World, scale_class: str, confidence: str
) -> None:
    row = _asset_axis_row("scale", world.family[0]["asset_id"])
    measure.write_axis(
        world.measurements,
        "scale",
        [row],
        {"fixture": "synthetic"},
        inventory_dir=world.inventory_dir,
    )
    row.update(scale_ref_class=scale_class, scale_ref_conf=confidence)
    _rewrite_axis_rows(world.measurements, "scale", [row])

    assert list(_load(world, "scale").values()) == [row]


@pytest.mark.parametrize(
    ("scale_class", "confidence"),
    [
        ("none", "class_only"),
        ("coin", "none"),
        ("foreign", "class_only"),
        ("coin", "0.900000000"),
    ],
)
def test_r3_scale_class_and_confidence_pairing_is_closed(
    world: World, scale_class: str, confidence: str
) -> None:
    row = _asset_axis_row("scale", world.family[0]["asset_id"])
    measure.write_axis(
        world.measurements,
        "scale",
        [row],
        {"fixture": "synthetic"},
        inventory_dir=world.inventory_dir,
    )
    row.update(scale_ref_class=scale_class, scale_ref_conf=confidence)
    _rewrite_axis_rows(world.measurements, "scale", [row])

    with pytest.raises(measure.MeasureError):
        _load(world, "scale")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("offset_audio_s", "-0.500000000"),
        ("offset_audio_s", "-0.0"),
        ("offset_audio_s", "0.12345678901234567"),
        ("offset_visual_s", "-0.500000000"),
        ("drift_ppm", "-12.500000000"),
        ("drift_se", "0.000000000"),
    ],
)
def test_g1_ingestion_preserves_legal_signed_and_precise_decimals(
    world: World, column: str, value: str
) -> None:
    _write_sync(world)
    row = _sync_row(world, **{column: value})
    _rewrite_axis_rows(world.measurements, "sync", [row])

    loaded = _load(world)

    assert next(iter(loaded.values()))[column] == value


@pytest.mark.parametrize(
    ("status_column", "metric_column"),
    [
        ("status_audio", "offset_audio_s"),
        ("status_audio", AUDIO_STRENGTH_COLUMN),
        ("status_audio", "peak_ratio_audio"),
        ("status_visual", "offset_visual_s"),
        ("status_visual", "conf_visual"),
        ("status_visual", "peak_corr_visual"),
    ],
)
def test_g1_ok_status_requires_each_peak_statistic(
    world: World, status_column: str, metric_column: str
) -> None:
    _write_sync(world)
    row = _sync_row(world, **{status_column: "ok", metric_column: ""})
    _rewrite_axis_rows(world.measurements, "sync", [row])

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize("column", ["status_audio", "status_visual"])
def test_g1_both_estimator_status_cells_are_required(world: World, column: str) -> None:
    _write_sync(world)
    row = _sync_row(world, **{column: ""})
    _rewrite_axis_rows(world.measurements, "sync", [row])

    with pytest.raises(measure.MeasureError):
        _load(world)


def test_g1_drift_cells_are_the_only_unconditionally_optional_statistics(world: World) -> None:
    _write_sync(world)
    row = _sync_row(world, drift_ppm="", drift_se="")
    _rewrite_axis_rows(world.measurements, "sync", [row])

    assert next(iter(_load(world).values())) == row


def test_status_alphabets_are_shared_code_constants() -> None:
    assert isinstance(measure.AUDIO_STATUSES, frozenset)
    assert isinstance(measure.VISUAL_STATUSES, frozenset)
    assert isinstance(measure.DRIFT_STATUSES, frozenset)
    assert "ok" in measure.AUDIO_STATUSES
    assert "ok" in measure.VISUAL_STATUSES
    assert measure.AUDIO_STATUSES.isdisjoint(measure.DRIFT_STATUSES)
    assert measure.VISUAL_STATUSES.isdisjoint(measure.DRIFT_STATUSES)


def test_every_estimator_status_constant_ingests(world: World) -> None:
    _write_sync(world)
    for status in sorted(measure.AUDIO_STATUSES):
        row = _sync_row(world, status_audio=status)
        _rewrite_axis_rows(world.measurements, "sync", [row])
        assert next(iter(_load(world).values()))["status_audio"] == status
    for status in sorted(measure.VISUAL_STATUSES):
        row = _sync_row(world, status_visual=status)
        _rewrite_axis_rows(world.measurements, "sync", [row])
        assert next(iter(_load(world).values()))["status_visual"] == status


def test_drift_status_constants_never_enter_the_audio_status_column(world: World) -> None:
    _write_sync(world)
    for status in sorted(measure.DRIFT_STATUSES):
        _rewrite_axis_rows(
            world.measurements,
            "sync",
            [_sync_row(world, status_audio=status)],
        )
        with pytest.raises(measure.MeasureError):
            _load(world)


@pytest.mark.parametrize("status", ["OK", "ok|low_confidence", "future_status"])
def test_status_outside_the_code_constants_is_refused(world: World, status: str) -> None:
    _write_sync(world)
    _rewrite_axis_rows(
        world.measurements,
        "sync",
        [_sync_row(world, status_audio=status)],
    )

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize("estimator", ["audio", "visual"])
def test_non_ok_status_may_keep_or_omit_peak_statistics(world: World, estimator: str) -> None:
    _write_sync(world)
    statuses = measure.AUDIO_STATUSES if estimator == "audio" else measure.VISUAL_STATUSES
    status = next(value for value in sorted(statuses) if value != "ok")
    fields = (
        ("offset_audio_s", AUDIO_STRENGTH_COLUMN, "peak_ratio_audio")
        if estimator == "audio"
        else ("offset_visual_s", "conf_visual", "peak_corr_visual")
    )
    status_column = f"status_{estimator}"
    populated = _sync_row(world, **{status_column: status})
    _rewrite_axis_rows(world.measurements, "sync", [populated])
    assert next(iter(_load(world).values())) == populated

    empty = _sync_row(
        world,
        **{status_column: status, **dict.fromkeys(fields, "")},
    )
    _rewrite_axis_rows(world.measurements, "sync", [empty])
    assert next(iter(_load(world).values())) == empty


def test_r9_sync_schema_publishes_raw_peak_rms_not_gate_relative_confidence() -> None:
    assert "peak_rms_audio" in measure.AXES["sync"].columns
    assert "conf_audio" not in measure.AXES["sync"].columns


def test_p39_audio_gate_sweep_moves_only_status() -> None:
    rate = 2000
    rng = np.random.default_rng(7)
    signal_a = rng.normal(size=rate * 4)
    delay = 240
    signal_b = np.concatenate((np.zeros(delay), signal_a[:-delay]))
    low = audio_offset.gcc_phat_peak(
        signal_a,
        signal_b,
        rate,
        min_overlap_s=1.0,
        min_peak_rms=0.1,
        min_peak_ratio=0.1,
    )
    high = audio_offset.gcc_phat_peak(
        signal_a,
        signal_b,
        rate,
        min_overlap_s=1.0,
        min_peak_rms=1e9,
        min_peak_ratio=1e9,
    )
    low_raw = dataclasses.asdict(low)
    high_raw = dataclasses.asdict(high)
    low_status = low_raw.pop("status")
    high_status = high_raw.pop("status")

    assert low_status != high_status
    assert low_raw == high_raw
    assert "confidence" not in low_raw
    assert "peak_rms" in low_raw


def test_p37_audio_offset_sign_is_t_b_minus_t_a() -> None:
    rate = 2000
    rng = np.random.default_rng(17)
    signal_a = rng.normal(size=rate * 4)
    delay = 240
    signal_b = np.concatenate((np.zeros(delay), signal_a[:-delay]))

    forward = audio_offset.gcc_phat_peak(signal_a, signal_b, rate, min_overlap_s=1.0)
    reverse = audio_offset.gcc_phat_peak(signal_b, signal_a, rate, min_overlap_s=1.0)
    assert forward.lag_s > 0
    assert reverse.lag_s == pytest.approx(-forward.lag_s)


def test_p39_visual_gate_sweep_moves_only_status(monkeypatch: pytest.MonkeyPatch) -> None:
    grid_hz = 60.0
    rng = np.random.default_rng(9)
    time_s = np.arange(0.0, 8.0, 1.0 / grid_hz)
    signal_a = rng.normal(size=time_s.size)
    delay = 12
    signal_b = np.concatenate((np.zeros(delay), signal_a[:-delay]))

    def estimate(gates: tuple[float, float, float]) -> Any:
        monkeypatch.setattr(visual_offset, "MIN_PEAK_CORRELATION", gates[0])
        monkeypatch.setattr(visual_offset, "MIN_CONFIDENCE", gates[1])
        monkeypatch.setattr(visual_offset, "MIN_PEAK_RATIO", gates[2])
        return visual_offset.estimate(time_s, signal_a, time_s, signal_b, grid_hz=grid_hz)

    low = dataclasses.asdict(estimate((-1.0, 0.0, 0.0)))
    high = dataclasses.asdict(estimate((1.0, 1e9, 1e9)))
    low_status = low.pop("status")
    high_status = high.pop("status")

    assert low_status != high_status
    assert low == high


def test_p37_visual_offset_sign_is_t_b_minus_t_a(monkeypatch: pytest.MonkeyPatch) -> None:
    grid_hz = 60.0
    rng = np.random.default_rng(19)
    time_s = np.arange(0.0, 8.0, 1.0 / grid_hz)
    signal_a = rng.normal(size=time_s.size)
    delay = 12
    signal_b = np.concatenate((np.zeros(delay), signal_a[:-delay]))
    monkeypatch.setattr(visual_offset, "MIN_PEAK_CORRELATION", -1.0)
    monkeypatch.setattr(visual_offset, "MIN_CONFIDENCE", 0.0)
    monkeypatch.setattr(visual_offset, "MIN_PEAK_RATIO", 0.0)

    forward = visual_offset.estimate(time_s, signal_a, time_s, signal_b, grid_hz=grid_hz)
    reverse = visual_offset.estimate(time_s, signal_b, time_s, signal_a, grid_hz=grid_hz)
    assert forward.offset_s > 0
    assert reverse.offset_s == pytest.approx(-forward.offset_s)


@pytest.mark.parametrize(
    "columns",
    [
        measure.AXES["sync"].columns[:-1],
        (*measure.AXES["sync"].columns, "extra"),
        tuple(reversed(measure.AXES["sync"].columns)),
        (
            measure.AXES["sync"].columns[0],
            measure.AXES["sync"].columns[0],
            *measure.AXES["sync"].columns[2:],
        ),
    ],
)
def test_g1_ingestion_requires_the_exact_header(world: World, columns: tuple[str, ...]) -> None:
    _write_sync(world)
    _rewrite_axis_rows(world.measurements, "sync", [_sync_row(world)], columns=columns)

    with pytest.raises(measure.MeasureError):
        _load(world)


def test_g1_ingestion_rejects_an_extra_csv_cell(world: World) -> None:
    _write_sync(world)
    text = _render_rows(measure.AXES["sync"].columns, [_sync_row(world)])
    header, data = text.splitlines()
    _rewrite_axis_bytes(
        world.measurements,
        "sync",
        f"{header}\n{data},rogue\n".encode(),
        rows=1,
    )

    with pytest.raises(measure.MeasureError):
        _load(world)


def test_g1_ingestion_rejects_invalid_utf8(world: World) -> None:
    _write_sync(world)
    body = _render_rows(measure.AXES["sync"].columns, [_sync_row(world)]).encode() + b"\xff"
    _rewrite_axis_bytes(world.measurements, "sync", body, rows=1)

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize("declared_rows", [0, 2])
def test_g1_ingestion_rechecks_the_declared_row_count(world: World, declared_rows: int) -> None:
    _write_sync(world)
    _rewrite_axis_rows(
        world.measurements,
        "sync",
        [_sync_row(world)],
        declared_rows=declared_rows,
    )

    with pytest.raises(measure.MeasureError):
        _load(world)


def test_g1_ingestion_rejects_noncanonical_row_order(world: World) -> None:
    rows = _all_sync_rows(world)
    _write_sync(world, rows)
    _rewrite_axis_rows(world.measurements, "sync", list(reversed(rows)))

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize("conflicting", [False, True])
def test_g1_ingestion_rejects_duplicate_logical_keys(world: World, conflicting: bool) -> None:
    row = _sync_row(world)
    duplicate = {**row, "offset_audio_s": "8.000000000"} if conflicting else row.copy()
    _write_sync(world, [row])
    _rewrite_axis_rows(world.measurements, "sync", [row, duplicate])

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize("defect", ["reversed", "self"])
def test_g1_schema_ingestion_rejects_intrinsically_invalid_pair_keys(
    world: World, defect: str
) -> None:
    _write_sync(world)
    row = _sync_row(world)
    if defect == "reversed":
        row["asset_a"], row["asset_b"] = row["asset_b"], row["asset_a"]
    else:
        row["asset_b"] = row["asset_a"]
    _rewrite_axis_rows(world.measurements, "sync", [row])

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize("defect", ["cross_family", "wrong_capture", "foreign_asset"])
def test_p35_qualification_reconciliation_rejects_registry_relative_pair_keys(
    qualification_world: QualificationWorld, defect: str
) -> None:
    _write_qualification_sync(qualification_world)
    world = qualification_world.measurement_world()
    row = _sync_row(world)
    if defect == "cross_family":
        row["asset_a"], row["asset_b"] = sorted(
            (world.family[0]["asset_id"], world.outsider["asset_id"])
        )
    elif defect == "wrong_capture":
        row["capture_id"] = world.outsider["capture_id"]
    else:
        row["asset_a"], row["asset_b"] = sorted(
            (world.family[0]["asset_id"], _foreign_asset_id(world))
        )
    _rewrite_axis_rows(qualification_world.measurements, "sync", [row])

    assert qualify.main(_qualify_arguments(qualification_world)) == 2
    assert not qualification_world.out.exists()


def test_p35_qualification_reconciliation_rejects_a_foreign_per_asset_key(
    qualification_world: QualificationWorld,
) -> None:
    world = qualification_world.measurement_world()
    row = {
        "asset_id": world.family[0]["asset_id"],
        "detect_rate": "1.000000000",
        "detect_conf_median": "0.900000000",
        "subject_px_height_median": "100.000000000",
    }
    measure.write_axis(
        world.measurements,
        "detect",
        [row],
        {"fixture": "synthetic"},
        inventory_dir=world.inventory_dir,
    )
    row["asset_id"] = _foreign_asset_id(world)
    _rewrite_axis_rows(world.measurements, "detect", [row])

    assert qualify.main(_qualify_arguments(qualification_world)) == 2
    assert not qualification_world.out.exists()


def test_g4_present_zero_row_axis_is_measured_but_empty(world: World) -> None:
    measure.write_axis(
        world.measurements,
        "detect",
        [],
        {"fixture": "completed-empty"},
        inventory_dir=world.inventory_dir,
    )

    manifest = _manifest(world.measurements)
    assert "detect" in manifest["axes"]
    assert manifest["axes"]["detect"]["rows"] == 0
    assert _load(world, "detect") == {}


def test_absent_axis_loads_as_unmeasured(world: World) -> None:
    _write_sync(world)

    assert "detect" not in _manifest(world.measurements)["axes"]
    assert _load(world, "detect") == {}


def test_p35_omitted_registry_keys_ingest_as_unmeasured(world: World) -> None:
    row = {
        "asset_id": world.family[0]["asset_id"],
        "detect_rate": "1.000000000",
        "detect_conf_median": "0.900000000",
        "subject_px_height_median": "100.000000000",
    }
    measure.write_axis(
        world.measurements,
        "detect",
        [row],
        {"fixture": "sparse"},
        inventory_dir=world.inventory_dir,
    )

    assert list(_load(world, "detect").values()) == [row]


def test_g6_sparse_sync_ingests_but_generator_enumerates_every_pair(world: World) -> None:
    one = _sync_row(world)
    _write_sync(world, [one])

    assert list(_load(world).values()) == [one]
    assets = sync.load_assets(world.inventory_dir)
    generated = {(pair.asset_a, pair.asset_b) for pair in sync.enumerate_pairs(assets)}
    expected = {
        tuple(sorted((left["asset_id"], right["asset_id"])))
        for left, right in itertools.combinations(world.family, 2)
    }
    assert expected <= generated


def test_pair_enumeration_is_independent_of_registry_row_order(world: World) -> None:
    assets = sync.load_assets(world.inventory_dir)
    forward = sync.enumerate_pairs(assets)
    reverse = sync.enumerate_pairs(list(reversed(assets)))

    assert forward == reverse
    assert [(pair.asset_a, pair.asset_b) for pair in forward] == sorted(
        (pair.asset_a, pair.asset_b) for pair in forward
    )


def test_s19_table_paths_are_fixed_unique_basenames_not_inputs() -> None:
    tables = [axis.table for axis in measure.AXES.values()]

    assert set(measure.AXES) == {"sync", "rigidity", "detect", "scale"}
    assert len(tables) == len(set(tables))
    assert all(pathlib.PurePath(table).name == table for table in tables)
    assert "table" not in inspect.signature(measure.write_axis).parameters


def test_unknown_axis_is_refused_before_any_table_path_exists(world: World) -> None:
    with pytest.raises(measure.MeasureError):
        measure.write_axis(
            world.measurements,
            "../foreign",
            [],
            {},
            inventory_dir=world.inventory_dir,
        )

    assert not world.measurements.exists()


@pytest.mark.parametrize("location", ["top", "axis", "generation"])
def test_manifest_structural_key_sets_are_closed(world: World, location: str) -> None:
    _write_sync(world)
    manifest = _manifest(world.measurements)
    if location == "top":
        manifest["foreign"] = "value"
    elif location == "axis":
        manifest["axes"]["sync"]["foreign"] = "value"
    else:
        manifest["generation"]["foreign"] = "value"
    _write_manifest(world.measurements, manifest)

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


def test_provenance_payload_is_open_and_digest_bound(world: World) -> None:
    _write_sync(world)
    manifest = _manifest(world.measurements)
    manifest["axes"]["sync"]["provenance"]["future_instrument"] = {
        "nested": [1, "two", {"three": True}]
    }
    _write_manifest(world.measurements, manifest)

    sidecar = measure.validate(world.measurements, inventory_dir=world.inventory_dir)
    assert list(measure.load_axis(sidecar, "sync").values()) == [_sync_row(world)]


def test_manifest_semantic_digest_ignores_whitespace_and_key_order(world: World) -> None:
    _write_sync(world)
    manifest = _manifest(world.measurements)
    reordered = {key: manifest[key] for key in reversed(tuple(manifest))}
    path = world.measurements / measure.MANIFEST_FILENAME
    path.write_text(
        json.dumps(reordered, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    sidecar = measure.validate(world.measurements, inventory_dir=world.inventory_dir)
    assert list(measure.load_axis(sidecar, "sync").values()) == [_sync_row(world)]


def test_coherently_recomputed_sidecar_is_accepted_not_authenticated(world: World) -> None:
    _write_sync(world)
    changed = _sync_row(world, offset_audio_s="0.500000000")
    _rewrite_axis_rows(world.measurements, "sync", [changed])

    assert list(_load(world).values()) == [changed]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_closed_sidecar_directory_rejects_every_unmanifested_entry(world: World, kind: str) -> None:
    _write_sync(world)
    extra = world.measurements / "writer-debris"
    if kind == "file":
        extra.write_text("stale", encoding="utf-8")
    elif kind == "directory":
        extra.mkdir()
    else:
        target = world.measurements.parent / "debris-target"
        target.write_text("stale", encoding="utf-8")
        extra.symlink_to(target)

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


def test_axis_generator_version_is_checked_independently(world: World) -> None:
    _write_sync(world)
    manifest = _manifest(world.measurements)
    manifest["axes"]["sync"]["generator_version"] = "foreign-version"
    _write_manifest(world.measurements, manifest)

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


def test_one_axis_rerun_preserves_independently_produced_axes(world: World) -> None:
    detect = {
        "asset_id": world.family[0]["asset_id"],
        "detect_rate": "1.000000000",
        "detect_conf_median": "0.900000000",
        "subject_px_height_median": "100.000000000",
    }
    measure.write_axis(
        world.measurements,
        "detect",
        [detect],
        {"fixture": "first"},
        inventory_dir=world.inventory_dir,
    )
    _write_sync(world)
    _write_sync(world, [_sync_row(world, offset_audio_s="0.250000000")])

    assert list(_load(world, "detect").values()) == [detect]
    assert next(iter(_load(world).values()))["offset_audio_s"] == "0.250000000"


def test_flagless_qualification_generation_stays_byte_compatible(
    qualification_world: QualificationWorld,
) -> None:
    qualify.run(
        qualification_world.inventory_dir,
        qualification_world.sessions_dir,
        qualification_world.corpus,
        qualification_world.out,
    )
    census = json.loads(
        (qualification_world.out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8")
    )

    assert "measurements" not in census["generation"]
    assert set(census["generation"]) == set(qualify.GENERATION_KEYS)


def test_explicit_measurements_add_the_manifest_digest_as_a_third_upstream(
    qualification_world: QualificationWorld,
) -> None:
    _write_qualification_sync(qualification_world)

    assert qualify.main(_qualify_arguments(qualification_world)) == 0
    census = json.loads(
        (qualification_world.out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8")
    )
    manifest = _manifest(qualification_world.measurements)
    assert census["generation"]["measurements"] == manifest["generation"]["manifest"]


def test_explicit_missing_measurements_never_degrades_to_unmeasured(
    qualification_world: QualificationWorld,
) -> None:
    assert qualify.main(_qualify_arguments(qualification_world)) == 2
    assert not qualification_world.out.exists()


def test_measure_error_is_wrapped_in_the_qualification_error_domain(
    qualification_world: QualificationWorld,
) -> None:
    _write_qualification_sync(qualification_world)
    row = _sync_row(
        qualification_world.measurement_world(),
        status_audio="future_status",
    )
    _rewrite_axis_rows(qualification_world.measurements, "sync", [row])

    run: Any = qualify.run
    with pytest.raises(qualify.QualifyError):
        run(
            qualification_world.inventory_dir,
            qualification_world.sessions_dir,
            qualification_world.corpus,
            qualification_world.out,
            measurements_dir=qualification_world.measurements,
        )


def test_qualification_validation_rechecks_the_measurement_upstream(
    qualification_world: QualificationWorld,
) -> None:
    _write_qualification_sync(qualification_world)
    assert qualify.main(_qualify_arguments(qualification_world)) == 0
    changed = _sync_row(
        qualification_world.measurement_world(),
        offset_audio_s="0.500000000",
    )
    _rewrite_axis_rows(qualification_world.measurements, "sync", [changed])

    validate_generation: Any = qualify.validate_generation
    with pytest.raises(qualify.QualifyError):
        validate_generation(
            qualification_world.out,
            sessions_dir=qualification_world.sessions_dir,
            inventory_dir=qualification_world.inventory_dir,
            measurements_dir=qualification_world.measurements,
        )


def test_r3_sampled_scale_negative_stays_unmeasured_not_none(
    qualification_world: QualificationWorld,
) -> None:
    _write_qualification_sync(qualification_world)
    assert qualify.main(_qualify_arguments(qualification_world)) == 0
    assets = _rows(qualification_world.out / qualify.ASSETS_QC_FILENAME)

    assert all(row["scale_ref_class"] == row["scale_ref_conf"] == "" for row in assets)
    assert all("scale_unmeasured" in row["qc_flags"].split("|") for row in assets)


def test_present_empty_axis_counts_as_measured_in_qualification(
    qualification_world: QualificationWorld,
) -> None:
    measure.write_axis(
        qualification_world.measurements,
        "detect",
        [],
        {"fixture": "completed-empty"},
        inventory_dir=qualification_world.inventory_dir,
    )
    assert qualify.main(_qualify_arguments(qualification_world)) == 0
    census = json.loads(
        (qualification_world.out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8")
    )

    assert "detect" in census["measured_axes"]
    assert "detect" not in census["unmeasured_axes"]


@pytest.mark.parametrize(
    (
        "status_audio",
        "status_visual",
        "offset_audio_s",
        "offset_visual_s",
        "expected",
    ),
    [
        ("ok", "ok", "0.000000000", "0.010000000", "ok_corroborated"),
        ("ok", VISUAL_REJECTED_STATUS, "0.000000000", "0.010000000", "ok_uncorroborated"),
        ("ok", "ok", "0.000000000", "0.100000000", "contradicted"),
        (AUDIO_REJECTED_STATUS, "ok", "0.000000000", "0.010000000", "visual_only"),
        (
            AUDIO_REJECTED_STATUS,
            VISUAL_REJECTED_STATUS,
            "0.000000000",
            "0.010000000",
            "neither_accepted",
        ),
    ],
)
def test_r6_fusion_policy_stays_in_qualification(
    qualification_world: QualificationWorld,
    status_audio: str,
    status_visual: str,
    offset_audio_s: str,
    offset_visual_s: str,
    expected: str,
) -> None:
    _write_qualification_sync(
        qualification_world,
        status_audio=status_audio,
        status_visual=status_visual,
        offset_audio_s=offset_audio_s,
        offset_visual_s=offset_visual_s,
    )

    assert qualify.main(_qualify_arguments(qualification_world)) == 0
    pair = _rows(qualification_world.out / qualify.PAIRS_QC_FILENAME)[0]
    assert pair["status"] == expected


def test_p31_axis_entry_carries_exact_digest_count_version_and_open_provenance(
    world: World,
) -> None:
    row = _sync_row(world)
    manifest = _write_sync(world, [row])
    entry = manifest["axes"]["sync"]
    table = world.measurements / measure.AXES["sync"].table

    assert set(entry) == set(measure.AXIS_ENTRY_KEYS)
    assert entry["table"] == measure.AXES["sync"].table
    assert entry["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert entry["rows"] == 1
    assert entry["generator_version"] == measure.GENERATOR_VERSION
    assert entry["provenance"]["fixture"] == "synthetic"


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_manifest_row_count_is_a_nonnegative_json_integer(world: World, value: Any) -> None:
    _write_sync(world)
    manifest = _manifest(world.measurements)
    manifest["axes"]["sync"]["rows"] = value
    _write_manifest(world.measurements, manifest)

    with pytest.raises(measure.MeasureError):
        _load(world)


@pytest.mark.parametrize("defect", ["missing", "malformed", "non_object"])
def test_manifest_absence_or_malformed_shape_is_a_hard_error(world: World, defect: str) -> None:
    _write_sync(world)
    path = world.measurements / measure.MANIFEST_FILENAME
    if defect == "missing":
        path.unlink()
    elif defect == "malformed":
        path.write_text("{", encoding="utf-8")
    else:
        path.write_text("[]", encoding="utf-8")

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


@pytest.mark.parametrize("defect", ["missing", "edited", "directory"])
def test_named_axis_table_absence_or_mismatch_is_a_hard_error(world: World, defect: str) -> None:
    _write_sync(world)
    path = world.measurements / measure.AXES["sync"].table
    if defect == "missing":
        path.unlink()
    elif defect == "edited":
        path.write_text("edited", encoding="utf-8")
    else:
        path.unlink()
        path.mkdir()

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=world.inventory_dir)


def test_sidecar_stale_against_a_republished_registry_is_refused(world: World) -> None:
    _write_sync(world)
    other_root = world.measurements.parent / "other"
    other_root.mkdir()
    other = _write_registry(other_root, [_canonical(9, "above")])

    with pytest.raises(measure.MeasureError):
        measure.validate(world.measurements, inventory_dir=other.root)


def test_write_axis_bytes_ignore_output_name_and_mapping_insertion_order(world: World) -> None:
    rows = _all_sync_rows(world)
    first = world.measurements.parent / "first-sidecar"
    second = world.measurements.parent / "second-sidecar"
    measure.write_axis(
        first,
        "sync",
        rows,
        {"z": 1, "a": {"later": 2, "earlier": 1}},
        inventory_dir=world.inventory_dir,
    )
    measure.write_axis(
        second,
        "sync",
        rows,
        {"a": {"earlier": 1, "later": 2}, "z": 1},
        inventory_dir=world.inventory_dir,
    )

    first_bytes = {path.name: path.read_bytes() for path in first.iterdir()}
    second_bytes = {path.name: path.read_bytes() for path in second.iterdir()}
    assert first_bytes == second_bytes


def test_empty_sync_measurement_is_worker_count_and_path_deterministic(
    tmp_path: pathlib.Path,
) -> None:
    registry_root = tmp_path / "registry-fixture"
    registry_root.mkdir()
    registry = _write_registry(registry_root, [])
    first = tmp_path / "one-worker"
    second = tmp_path / "two-workers"
    sync.measure(
        registry.root,
        registry.corpus,
        first,
        tmp_path / "cache-one",
        workers=1,
    )
    sync.measure(
        registry.root,
        registry.corpus,
        second,
        tmp_path / "cache-two",
        workers=2,
    )

    first_bytes = {path.name: path.read_bytes() for path in first.iterdir()}
    second_bytes = {path.name: path.read_bytes() for path in second.iterdir()}
    assert first_bytes == second_bytes


def test_audio_cache_key_binds_source_content(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "signal.wav"
    cache = tmp_path / "cache"
    clean = tmp_path / "clean-cache"
    asset_id = "a" * 16
    _write_wav(source, 440.0)
    audio_offset.ensure_cached(source, cache, asset_id)
    before = {path.name: path.read_bytes() for path in audio_offset.cache_paths(cache, asset_id)}

    _write_wav(source, 880.0)
    audio_offset.ensure_cached(source, cache, asset_id)
    after = {path.name: path.read_bytes() for path in audio_offset.cache_paths(cache, asset_id)}
    audio_offset.ensure_cached(source, clean, asset_id)
    oracle = {path.name: path.read_bytes() for path in audio_offset.cache_paths(clean, asset_id)}

    assert after != before
    assert after == oracle


def test_partial_audio_cache_is_rebuilt_not_reused(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "signal.wav"
    cache = tmp_path / "cache"
    clean = tmp_path / "clean-cache"
    asset_id = "b" * 16
    _write_wav(source, 440.0)
    audio_offset.ensure_cached(source, cache, asset_id)
    audio_offset.cache_paths(cache, asset_id)[0].write_bytes(b"partial")

    audio_offset.ensure_cached(source, cache, asset_id)
    audio_offset.ensure_cached(source, clean, asset_id)
    repaired = {path.name: path.read_bytes() for path in audio_offset.cache_paths(cache, asset_id)}
    oracle = {path.name: path.read_bytes() for path in audio_offset.cache_paths(clean, asset_id)}
    assert repaired == oracle


def test_audio_provenance_names_every_public_numeric_constant() -> None:
    names = {
        "TARGET_RATE": "target_rate_hz",
        "COARSE_RATE": "coarse_rate_hz",
        "MIN_OVERLAP_S": "min_overlap_s",
        "PEAK_GUARD_S": "peak_guard_s",
        "MIN_PEAK_RMS": "min_peak_rms",
        "MIN_PEAK_RATIO": "min_peak_ratio",
        "EDGE_FADE_S": "edge_fade_s",
        "BAND_LOW_HZ": "band_low_hz",
        "BAND_HIGH_FRACTION": "band_high_fraction",
        "DRIFT_WINDOW_S": "drift_window_s",
        "DRIFT_MIN_WINDOW_S": "drift_min_window_s",
        "DRIFT_SEARCH_RADIUS_S": "drift_search_radius_s",
        "DRIFT_MAX_WINDOWS": "drift_max_windows",
        "DRIFT_MIN_WINDOWS": "drift_min_windows",
        "LOCAL_PEAK_GUARD_S": "local_peak_guard_s",
        "LOCAL_MIN_PEAK_RMS": "local_min_peak_rms",
        "LOCAL_MIN_PEAK_RATIO": "local_min_peak_ratio",
    }

    for constant, provenance_key in names.items():
        assert audio_offset.PROVENANCE[provenance_key] == getattr(audio_offset, constant)
    assert audio_offset.PROVENANCE["sign_convention"] == "t_b_minus_t_a"


def test_visual_provenance_names_every_public_instrument_constant() -> None:
    names = {
        "SIGNAL_VERSION": "signal_version",
        "SIGNAL_FIELD": "signal_field",
        "DISPLAY_WIDTH": "display_width",
        "DISPLAY_HEIGHT": "display_height",
        "BORDER_FRACTION": "border_fraction",
        "GRID_HZ": "grid_hz",
        "SMOOTH_S": "smooth_s",
        "PEAK_EXCLUSION_S": "peak_exclusion_s",
        "MIN_OVERLAP_S": "min_overlap_s",
        "MIN_PEAK_CORRELATION": "min_peak_correlation",
        "MIN_CONFIDENCE": "min_confidence",
        "MIN_PEAK_RATIO": "min_peak_ratio",
        "EDGE_GUARD_S": "edge_guard_s",
    }

    for constant, provenance_key in names.items():
        assert visual_offset.PROVENANCE[provenance_key] == getattr(visual_offset, constant)
    assert visual_offset.PROVENANCE["sign_convention"] == "t_b_minus_t_a"


def test_sidecar_rerun_keeps_the_directory_in_place_and_never_sweeps_siblings(
    world: World,
) -> None:
    _write_sync(world)
    inode = world.measurements.stat().st_ino
    sibling = world.measurements.parent / ".measurements.staging-999999"
    sibling.mkdir()
    (sibling / "testimony").write_text("keep", encoding="utf-8")

    _write_sync(world, [_sync_row(world, offset_audio_s="0.250000000")])

    assert world.measurements.stat().st_ino == inode
    assert (sibling / "testimony").read_text(encoding="utf-8") == "keep"


def test_concurrent_axis_writers_are_declared_unsupported() -> None:
    documentation = inspect.getdoc(measure.write_axis) or ""

    assert "concurrent" in documentation.lower()
    assert "unsupported" in documentation.lower()


def test_all_four_axis_modules_exist_and_name_their_axis() -> None:
    import importlib

    for axis_name in ("sync", "rigidity", "detect", "scale"):
        module = importlib.import_module(f"pose_estimation.measure.{axis_name}")
        assert axis_name == module.AXIS
