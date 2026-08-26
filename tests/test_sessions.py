from __future__ import annotations

import collections
import csv
import dataclasses
import hashlib
import inspect
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

from pose_estimation import inventory, multicam, sessions
from pose_estimation import run as run_cli


@dataclasses.dataclass(frozen=True)
class _Asset:
    source_path: str
    subject_ordinal: int | None = None
    view: str = ""
    task: str = ""
    side: str = ""
    repeat: int = 0
    disposition: str = inventory.CANONICAL
    reason_code: str = inventory.REASON_OK
    source_cell: str | None = None
    create_source: bool = True

    @property
    def asset_id(self) -> str:
        return inventory.asset_id_of(self.source_path)

    @property
    def capture_id(self) -> str:
        if self.disposition != inventory.CANONICAL or self.subject_ordinal is None:
            return ""
        return inventory.capture_id_of(self.subject_ordinal, self.task, self.side)

    @property
    def content(self) -> bytes:
        return f"synthetic-media:{self.asset_id}\n".encode()


@dataclasses.dataclass(frozen=True)
class _Registry:
    root: pathlib.Path
    corpus: pathlib.Path
    out: pathlib.Path
    assets: tuple[_Asset, ...]


def _canonical(
    subject_ordinal: int,
    view: str,
    *,
    task: str = "cap",
    side: str = "l",
    repeat: int = 0,
    extension: str = ".MOV",
    directory: str | None = None,
) -> _Asset:
    marker = f"_({repeat})" if repeat else ""
    parent = directory or f"synthetic-{subject_ordinal:02d}"
    return _Asset(
        source_path=(f"{parent}/{subject_ordinal}_{view}_{task}_{side}{marker}{extension}"),
        subject_ordinal=subject_ordinal,
        view=view,
        task=task,
        side=side,
        repeat=repeat,
    )


def _held_out(
    source_path: str,
    *,
    disposition: str,
    reason_code: str,
    source_cell: str | None = None,
) -> _Asset:
    return _Asset(
        source_path=source_path,
        disposition=disposition,
        reason_code=reason_code,
        source_cell=source_cell,
    )


def _asset_row(asset: _Asset, *, checksums: bool) -> dict[str, Any]:
    canonical = asset.disposition == inventory.CANONICAL
    opened = asset.disposition != inventory.EXCLUDED
    return {
        "asset_id": asset.asset_id,
        "capture_id": asset.capture_id,
        "disposition": asset.disposition,
        "reason_code": asset.reason_code,
        "source_path": asset.source_cell or asset.source_path,
        "subject_ordinal": asset.subject_ordinal if canonical else "",
        "view": asset.view if canonical else "",
        "task": asset.task if canonical else "",
        "side": asset.side if canonical else "",
        "repeat": asset.repeat if canonical else "",
        "normalizations": "",
        "size_bytes": len(asset.content),
        "content_sha256": hashlib.sha256(asset.content).hexdigest() if checksums else "",
        "reported_width": 1920 if opened else 0,
        "reported_height": 1080 if opened else 0,
        "reported_avg_fps": "30.0" if opened else "0.0",
        "reported_frame_count": 30 if opened else 0,
        "reported_rotation_deg": 0,
        "reported_fourcc": "avc1" if opened else "",
        "nominal_duration_s": "1.0" if opened else "",
        "fact_flags": "",
        "probe_status": "opened" if opened else "skipped",
        "grammar_version": inventory.GRAMMAR_VERSION,
        "tool_version": inventory.TOOL_VERSION,
    }


def _capture_rows(assets: tuple[_Asset, ...]) -> list[dict[str, Any]]:
    grouped: dict[str, list[_Asset]] = collections.defaultdict(list)
    for asset in assets:
        if asset.disposition == inventory.CANONICAL:
            grouped[asset.capture_id].append(asset)
    rows = []
    for capture_id in sorted(grouped):
        members = grouped[capture_id]
        views = sorted({asset.view for asset in members})
        first = members[0]
        rows.append(
            {
                "capture_id": capture_id,
                "subject_ordinal": first.subject_ordinal,
                "task": first.task,
                "side": first.side,
                "n_assets": len(members),
                "views": "|".join(views),
                "n_views": len(views),
                "view_conflict": int(len(members) != len(views)),
                "reported_frame_count_min": 30,
                "reported_frame_count_max": 30,
                "reported_fps_min": "30.0",
                "reported_fps_max": "30.0",
                "reported_fps_spread_hz": "0.0",
                "nominal_duration_min_s": "1.0",
                "nominal_duration_max_s": "1.0",
                "nominal_duration_spread_s": "0.0",
                "reported_resolution_agree": 1,
                "reported_rotation_agree": 1,
                "grammar_version": inventory.GRAMMAR_VERSION,
                "tool_version": inventory.TOOL_VERSION,
            }
        )
    return rows


def _write_registry(
    tmp_path: pathlib.Path, assets: list[_Asset], *, checksums: bool = True
) -> _Registry:
    registry = tmp_path / "inventory"
    corpus = tmp_path / "corpus"
    out = tmp_path / "sessions"
    registry.mkdir()
    corpus.mkdir()
    frozen_assets = tuple(assets)
    for asset in frozen_assets:
        if not asset.create_source:
            continue
        source = corpus / asset.source_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(asset.content)

    asset_rows = sorted(
        (_asset_row(asset, checksums=checksums) for asset in frozen_assets),
        key=lambda row: row["source_path"],
    )
    capture_rows = _capture_rows(frozen_assets)
    assets_text = inventory.render_csv(inventory.ASSET_COLUMNS, asset_rows)
    captures_text = inventory.render_csv(inventory.CAPTURE_COLUMNS, capture_rows)
    reason_counts = collections.Counter(asset.reason_code for asset in frozen_assets)
    census: dict[str, Any] = {
        "tool_version": inventory.TOOL_VERSION,
        "grammar_version": inventory.GRAMMAR_VERSION,
        "opencv_version": "synthetic",
        "backend_name": "synthetic",
        "orientation_auto": bool(frozen_assets),
        "checksums": checksums,
        "generation": {
            inventory.ASSETS_FILENAME: hashlib.sha256(assets_text.encode()).hexdigest(),
            inventory.CAPTURES_FILENAME: hashlib.sha256(captures_text.encode()).hexdigest(),
        },
        "assets": {
            "discovered": len(frozen_assets),
            "canonical": sum(a.disposition == inventory.CANONICAL for a in frozen_assets),
            "quarantined": sum(a.disposition == inventory.QUARANTINED for a in frozen_assets),
            "excluded": sum(a.disposition == inventory.EXCLUDED for a in frozen_assets),
        },
        "reason_codes": dict(sorted(reason_counts.items())),
        "captures": {
            "total": len(capture_rows),
            "with_view_conflict": sum(int(row["view_conflict"]) for row in capture_rows),
        },
    }
    census["generation"][inventory.CENSUS_FILENAME] = inventory.census_digest(census)
    (registry / inventory.ASSETS_FILENAME).write_text(assets_text, encoding="utf-8", newline="")
    (registry / inventory.CAPTURES_FILENAME).write_text(captures_text, encoding="utf-8", newline="")
    (registry / inventory.CENSUS_FILENAME).write_text(
        inventory.render_json(census), encoding="utf-8", newline=""
    )
    assert inventory.validate_generation(registry) == census
    return _Registry(root=registry, corpus=corpus, out=out, assets=frozen_assets)


def _sessions_module() -> ModuleType:
    return sessions


def _run_main(registry: _Registry, *, strict: bool = False, expected_status: int = 0) -> None:
    argv = [
        "--inventory",
        str(registry.root),
        "--corpus",
        str(registry.corpus),
        "--out",
        str(registry.out),
    ]
    if strict:
        argv.append("--strict")
    assert _sessions_module().main(argv) == expected_status


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _events(registry: _Registry) -> list[dict[str, str]]:
    return _read_csv(registry.out / "events.csv")


def _placements(registry: _Registry) -> list[dict[str, str]]:
    return _read_csv(registry.out / "placements.csv")


def _manifest(registry: _Registry, event_id: str) -> dict[str, Any]:
    return json.loads((registry.out / event_id / "session.json").read_text(encoding="utf-8"))


def test_p01_every_asset_row_has_exactly_one_placement(tmp_path: pathlib.Path) -> None:
    assets = [
        _canonical(1, "above"),
        _canonical(1, "left"),
        _held_out(
            "synthetic-hold/2_above_cap_.MOV",
            disposition=inventory.QUARANTINED,
            reason_code="side_missing",
        ),
        _held_out(
            "synthetic-hold/readme.bin",
            disposition=inventory.EXCLUDED,
            reason_code="unsupported_extension",
        ),
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    rows = _placements(registry)
    by_asset = collections.Counter(row["asset_id"] for row in rows)
    assert by_asset == collections.Counter(asset.asset_id for asset in assets)
    for asset in assets:
        row = next(row for row in rows if row["asset_id"] == asset.asset_id)
        assert row["disposition"] == asset.disposition
        assert row["placement_reason"]
        if asset.disposition == inventory.CANONICAL:
            assert row["placement"] == "placed"
            assert row["placement_reason"] == "ok"
            assert row["event_id"]
            assert row["camera_name"].startswith("cam-")
        else:
            assert row["placement"] == "held_out"
            assert row["placement_reason"] == (
                "quarantined_stem"
                if asset.disposition == inventory.QUARANTINED
                else "excluded_asset"
            )
            assert row["event_id"] == ""
            assert row["camera_name"] == ""


def test_p01_empty_registry_emits_complete_empty_generation(tmp_path: pathlib.Path) -> None:
    registry = _write_registry(tmp_path, [])
    _run_main(registry)

    assert _events(registry) == []
    assert _placements(registry) == []
    assert _sessions_module().validate_generation(registry.out)
    assert multicam.discover_sessions(registry.out) == []


def test_p01_strict_reports_held_out_after_publishing_ledger(tmp_path: pathlib.Path) -> None:
    registry = _write_registry(
        tmp_path,
        [
            _canonical(1, "above"),
            _held_out(
                "synthetic-hold/2_above_cap_.MOV",
                disposition=inventory.QUARANTINED,
                reason_code="side_missing",
            ),
        ],
    )
    _run_main(registry, strict=True, expected_status=1)

    placements = _placements(registry)
    assert len(placements) == 2
    assert {row["placement_reason"] for row in placements} == {"ok", "quarantined_stem"}
    assert _sessions_module().validate_generation(registry.out)


def test_p02_placed_assets_symlinks_and_event_camera_counts_conserve(
    tmp_path: pathlib.Path,
) -> None:
    assets = [
        *(_canonical(1, view) for view in ("above", "left", "right")),
        *(_canonical(2, view, task="coin", side="r") for view in ("above", "right")),
        _canonical(3, "left", task="glass"),
        _canonical(4, "above", task="key"),
        _canonical(4, "above", task="key", repeat=1),
        _held_out(
            "synthetic-hold/5_right_nut_.MOV",
            disposition=inventory.QUARANTINED,
            reason_code="side_missing",
        ),
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    events = _events(registry)
    placements = _placements(registry)
    placed = sum(bool(row["event_id"]) for row in placements)
    links = [
        entry
        for event in events
        for entry in (registry.out / event["event_id"]).iterdir()
        if entry.is_symlink()
    ]
    assert placed == len(links) == sum(int(event["n_cameras"]) for event in events)


def test_p03_event_ids_are_type_safe_and_round_trip(tmp_path: pathlib.Path) -> None:
    assets = [
        *(_canonical(7, view, task="peg", side="r") for view in ("above", "right")),
        _canonical(8, "left", task="nut"),
        _canonical(8, "left", task="nut", repeat=1),
        _canonical(8, "left", task="nut", repeat=2),
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    pattern = re.compile(r"^(?P<capture>s\d{2}-[a-z]+-[lr])_run-(?P<run>\d{2})$")
    capture_ids = {asset.capture_id for asset in assets}
    conflict_runs = []
    for event in _events(registry):
        match = pattern.fullmatch(event["event_id"])
        assert match is not None
        assert event["event_id"] not in capture_ids
        assert match["capture"] == event["capture_id"]
        assert int(match["run"]) == int(event["run_index"])
        assert event["event_id"] == (f"{event['capture_id']}_run-{int(event['run_index']):02d}")
        if event["capture_id"] == inventory.capture_id_of(8, "nut", "l"):
            conflict_runs.append(int(event["run_index"]))
    assert sorted(conflict_runs) == [1, 2, 3]


def test_p03_conflict_run_index_overflow_is_rejected(tmp_path: pathlib.Path) -> None:
    assets = [_canonical(9, "above", task="peg", repeat=repeat) for repeat in range(100)]
    registry = _write_registry(tmp_path, assets)
    module = _sessions_module()

    with pytest.raises(module.SessionsError):
        module.run(registry.root, registry.corpus, registry.out)
    assert not registry.out.exists()


def test_p04_conflict_assets_each_get_one_unresolved_single_camera_run(
    tmp_path: pathlib.Path,
) -> None:
    assets = [
        _canonical(5, "above", task="key"),
        _canonical(5, "above", task="key", repeat=1),
        _canonical(5, "above", task="key", repeat=2),
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    events = _events(registry)
    assert len(events) == len(assets)
    assert {event["take_resolution"] for event in events} == {"unresolved"}
    assert {event["n_cameras"] for event in events} == {"1"}
    assert {event["views"] for event in events} == {"above"}
    assert len({row["event_id"] for row in _placements(registry)}) == len(assets)
    for event in events:
        manifest = _manifest(registry, event["event_id"])
        assert manifest["take_resolution"] == "unresolved"
        assert manifest["n_cameras"] == 1
        assert len(manifest["cameras"]) == 1


def test_p04_non_conflict_family_stays_one_whole_run(tmp_path: pathlib.Path) -> None:
    assets = [_canonical(6, view, task="coin", side="r") for view in ("right", "above", "left")]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    events = _events(registry)
    assert len(events) == 1
    assert events[0]["take_resolution"] == "family"
    assert events[0]["n_cameras"] == "3"
    assert events[0]["views"] == "above|left|right"
    assert {row["event_id"] for row in _placements(registry)} == {events[0]["event_id"]}
    assert [camera["name"] for camera in _manifest(registry, events[0]["event_id"])["cameras"]] == [
        "cam-above",
        "cam-left",
        "cam-right",
    ]


def _tree_snapshot(root: pathlib.Path) -> dict[str, tuple[str, bytes]]:
    snapshot: dict[str, tuple[str, bytes]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.fsencode(path.readlink()))
        elif path.is_dir():
            snapshot[relative] = ("directory", b"")
        else:
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


def _subprocess_generate(registry: _Registry, out: pathlib.Path, env: dict[str, str]) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pose_estimation.sessions",
            "--inventory",
            str(registry.root),
            "--corpus",
            str(registry.corpus),
            "--out",
            str(out),
        ],
        cwd=pathlib.Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_p05_discover_sessions_matches_every_emitted_event(tmp_path: pathlib.Path) -> None:
    assets = [
        _canonical(1, "above"),
        *(_canonical(2, view, task="coin", side="r") for view in ("left", "right")),
        *(_canonical(3, view, task="peg") for view in ("above", "left", "right")),
        _canonical(4, "right", task="nut"),
        _canonical(4, "right", task="nut", repeat=1),
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    rows = {row["event_id"]: row for row in _events(registry)}
    discovered = {
        session.session_id: session for session in multicam.discover_sessions(registry.out)
    }
    assert discovered.keys() == rows.keys()
    for event_id, session in discovered.items():
        row = rows[event_id]
        assert session.session_id == event_id
        assert session.n_cameras == int(row["n_cameras"])
        assert session.camera_names() == [f"cam-{view}" for view in row["views"].split("|")]
        assert all(
            camera.file.is_file() and not camera.file.is_symlink() for camera in session.cameras
        )


def test_p05_no_checksum_registry_is_discoverable_without_reading_corpus_bytes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")], checksums=False)
    corpus = registry.corpus.resolve()
    original_open = pathlib.Path.open
    original_read_bytes = pathlib.Path.read_bytes

    def reject_corpus_open(path: pathlib.Path, *args: Any, **kwargs: Any):
        if path.resolve().is_relative_to(corpus):
            pytest.fail("Session materialization opened corpus bytes.")
        return original_open(path, *args, **kwargs)

    def reject_corpus_read_bytes(path: pathlib.Path):
        if path.resolve().is_relative_to(corpus):
            pytest.fail("Session materialization read corpus bytes.")
        return original_read_bytes(path)

    monkeypatch.setattr(pathlib.Path, "open", reject_corpus_open)
    monkeypatch.setattr(pathlib.Path, "read_bytes", reject_corpus_read_bytes)
    _run_main(registry)

    sessions_found = multicam.discover_sessions(registry.out)
    assert len(sessions_found) == 1
    manifest = _manifest(registry, _events(registry)[0]["event_id"])
    assert manifest["cameras"][0]["content_sha256"] == ""


_REAL_INVENTORY = pathlib.Path("inventory")
_REAL_CORPUS = pathlib.Path("videos") / "3-cam"


@pytest.mark.skipif(
    not (_REAL_INVENTORY.is_dir() and _REAL_CORPUS.is_dir()),
    reason="The committed registry or active corpus is unavailable.",
)
def test_p05_real_corpus_headline_counts(tmp_path: pathlib.Path) -> None:
    _sessions_module()
    registry = _Registry(
        root=_REAL_INVENTORY,
        corpus=_REAL_CORPUS,
        out=tmp_path / "sessions",
        assets=(),
    )
    _run_main(registry)

    events = _events(registry)
    placements = _placements(registry)
    assert len(events) == 193
    assert collections.Counter(int(row["n_cameras"]) for row in events) == {
        1: 58,
        2: 84,
        3: 51,
    }
    assert len(placements) == 382
    assert sum(bool(row["event_id"]) for row in placements) == 379
    assert sum(not row["event_id"] for row in placements) == 3
    assert len(multicam.discover_sessions(registry.out)) == 193


def test_p06_list_sessions_reports_the_generated_event_count(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _sessions_module()
    registry = _write_registry(
        tmp_path,
        [
            _canonical(1, "above"),
            _canonical(2, "above", task="coin"),
            _canonical(2, "right", task="coin"),
        ],
    )
    _run_main(registry)
    capsys.readouterr()

    # run.main keeps its SystemExit convention; only argv rides through it.
    with pytest.raises(SystemExit) as exit_info:
        run_cli.main(["--list-sessions", "--sessions-dir", str(registry.out)])
    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert f"Discovered sessions: {len(_events(registry))} session(s)" in output


def test_p06_list_sessions_defaults_to_sessions_root(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _sessions_module()
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()

    with pytest.raises(SystemExit) as exit_info:
        run_cli.main(["--list-sessions"])
    assert exit_info.value.code == 0
    assert "Discovered sessions: 1 session(s)" in capsys.readouterr().out


def test_p07_shuffled_iterdir_and_different_out_name_are_byte_identical(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sessions_module()
    assets = [
        *(_canonical(1, view) for view in ("above", "left", "right")),
        _canonical(2, "above", task="key"),
        _canonical(2, "above", task="key", repeat=1),
        _canonical(3, "right", task="glass", extension=".MP4"),
    ]
    registry = _write_registry(tmp_path, assets)
    first = dataclasses.replace(registry, out=tmp_path / "sessions-a")
    second = dataclasses.replace(registry, out=tmp_path / "sessions-b")
    _run_main(first)
    oracle = _tree_snapshot(first.out)

    original_iterdir = pathlib.Path.iterdir

    def reverse_iterdir(path: pathlib.Path):
        return iter(reversed(list(original_iterdir(path))))

    monkeypatch.setattr(pathlib.Path, "iterdir", reverse_iterdir)
    _run_main(second)
    assert _tree_snapshot(second.out) == oracle


def test_p07_hash_locale_timezone_and_out_matrix_is_byte_identical(
    tmp_path: pathlib.Path,
) -> None:
    _sessions_module()
    registry = _write_registry(
        tmp_path,
        [
            *(_canonical(1, view) for view in ("above", "left", "right")),
            _canonical(2, "above", task="nut"),
            _canonical(2, "above", task="nut", repeat=1),
        ],
    )
    locales = ("C", "C.utf8", "en_US.utf8", "POSIX")
    timezones = ("UTC", "Pacific/Kiritimati", "America/New_York", "Asia/Kathmandu")
    snapshots = []
    for index, (locale, timezone) in enumerate(zip(locales, timezones, strict=True)):
        out = tmp_path / f"sessions-{index}"
        env = os.environ.copy()
        env.update(
            {
                "LANG": locale,
                "LC_ALL": locale,
                "PYTHONHASHSEED": str(1000 + index),
                "TZ": timezone,
            }
        )
        _subprocess_generate(registry, out, env)
        snapshots.append(_tree_snapshot(out))
    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:])


def test_p08_regeneration_is_idempotent_and_removes_stale_entries(
    tmp_path: pathlib.Path,
) -> None:
    _sessions_module()
    registry = _write_registry(
        tmp_path,
        [
            _canonical(1, "above"),
            _canonical(1, "left"),
            _canonical(2, "right", task="coin"),
        ],
    )
    oracle_registry = dataclasses.replace(registry, out=tmp_path / "oracle")
    _run_main(oracle_registry)
    oracle = _tree_snapshot(oracle_registry.out)

    _run_main(registry)
    first = _tree_snapshot(registry.out)
    _run_main(registry)
    assert _tree_snapshot(registry.out) == first == oracle

    marker = json.loads((registry.out / "generation.json").read_text(encoding="utf-8"))
    assert isinstance(marker, dict)
    assert marker.get("generator_version")
    event_id = _events(registry)[0]["event_id"]
    (registry.out / "stale-root.txt").write_text("stale", encoding="utf-8")
    (registry.out / event_id / "stale-child.txt").write_text("stale", encoding="utf-8")
    _run_main(registry)
    assert _tree_snapshot(registry.out) == oracle


def _rewrite_inventory_generation(registry: _Registry) -> None:
    assets_path = registry.root / inventory.ASSETS_FILENAME
    rows = _read_csv(assets_path)
    rows[0]["reported_width"] = "1919"
    assets_text = inventory.render_csv(inventory.ASSET_COLUMNS, rows)
    census_path = registry.root / inventory.CENSUS_FILENAME
    census = json.loads(census_path.read_text(encoding="utf-8"))
    census["generation"][inventory.ASSETS_FILENAME] = hashlib.sha256(
        assets_text.encode()
    ).hexdigest()
    census["generation"][inventory.CENSUS_FILENAME] = inventory.census_digest(census)
    assets_path.write_text(assets_text, encoding="utf-8", newline="")
    census_path.write_text(inventory.render_json(census), encoding="utf-8", newline="")
    assert inventory.validate_generation(registry.root) == census


def _clone_tree(out: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    """Copy a published tree into its own parent, so mutations stay isolated."""
    clone = pathlib.Path(tempfile.mkdtemp(dir=tmp_path)) / "tree"
    shutil.copytree(out, clone, symlinks=True)
    return clone


def test_p09_valid_generation_returns_its_generation_block(tmp_path: pathlib.Path) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(1, "left")])
    _run_main(registry)

    # generation.json IS the block, flat: the registry nests its own inside
    # census.json only because that file also carries aggregates.
    payload = json.loads((registry.out / "generation.json").read_text(encoding="utf-8"))
    module = _sessions_module()
    assert set(payload) == set(module.GENERATION_KEYS)
    assert module.validate_generation(registry.out) == payload
    assert module.validate_generation(str(registry.out), str(registry.root)) == payload


def test_p09_generation_whitespace_and_key_order_are_not_tamper(
    tmp_path: pathlib.Path,
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    path = registry.out / "generation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    reordered = dict(reversed(tuple(payload.items())))
    path.write_text(json.dumps(reordered, separators=(",", ":")) + "\n", encoding="utf-8")

    assert _sessions_module().validate_generation(registry.out) == payload


def test_p09_tree_digest_covers_every_entry_except_the_marker(
    tmp_path: pathlib.Path,
) -> None:
    asset = _canonical(1, "above")
    registry = _write_registry(tmp_path, [asset])
    _run_main(registry)
    module = _sessions_module()
    expected = module.tree_digest(registry.out)

    # A link target's *contents* stay outside, so corpus bytes never enter.
    (registry.corpus / asset.source_path).write_bytes(b"changed target bytes")
    assert module.tree_digest(str(registry.out)) == expected

    # generation.json cannot digest itself.
    marker = registry.out / "generation.json"
    marker.write_text(marker.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert module.tree_digest(registry.out) == expected

    # Everything else is covered, kind included.
    for mutate in (
        lambda out: (out / "root-note.txt").write_text("unexplained", encoding="utf-8"),
        lambda out: (out / "empty-child").mkdir(),
    ):
        clone = _clone_tree(registry.out, tmp_path)
        mutate(clone)
        assert module.tree_digest(clone) != expected


def test_p09_tree_digest_distinguishes_link_text_and_entry_kind(
    tmp_path: pathlib.Path,
) -> None:
    """A directory test follows a link, so kind has to be digested explicitly."""
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    module = _sessions_module()
    expected = module.tree_digest(registry.out)

    # Path.readlink() returns a PurePath, whose constructor drops "./".
    clone = _clone_tree(registry.out, tmp_path)
    link = next(clone.glob("*/cam-*"))
    target = os.readlink(link)  # noqa: PTH115
    link.unlink()
    link.symlink_to(f"./{target}")
    assert module.tree_digest(clone) != expected

    # An event directory swapped for a link to a byte-identical outside copy.
    clone = _clone_tree(registry.out, tmp_path)
    victim = next(path for path in sorted(clone.iterdir()) if path.is_dir())
    outside = clone.parent / f"{clone.name}-outside"
    shutil.copytree(victim, outside, symlinks=True)
    shutil.rmtree(victim)
    victim.symlink_to(outside, target_is_directory=True)
    assert module.tree_digest(clone) != expected


@pytest.mark.parametrize(
    "tamper",
    [
        "events",
        "placements",
        "generation",
        "session_directory",
        "symlink",
        "upstream_inventory",
    ],
)
def test_p09_every_published_tamper_class_raises_sessions_error(
    tmp_path: pathlib.Path, tamper: str
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(1, "left")])
    _run_main(registry)
    event_dir = registry.out / _events(registry)[0]["event_id"]

    if tamper == "events":
        path = registry.out / "events.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif tamper == "placements":
        path = registry.out / "placements.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif tamper == "generation":
        path = registry.out / "generation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["tampered"] = True
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif tamper == "session_directory":
        shutil.rmtree(event_dir)
    elif tamper == "symlink":
        next(path for path in event_dir.iterdir() if path.is_symlink()).unlink()
    elif tamper == "upstream_inventory":
        _rewrite_inventory_generation(registry)
    else:
        raise AssertionError(f"Unknown tamper case: {tamper}")

    module = _sessions_module()
    if tamper == "upstream_inventory":
        assert module.validate_generation(registry.out)
        with pytest.raises(module.SessionsError):
            module.validate_generation(registry.out, registry.root)
    else:
        with pytest.raises(module.SessionsError):
            module.validate_generation(registry.out)


def test_p10_registry_validation_precedes_row_reads_and_error_propagates(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    module = _sessions_module()
    protected = {
        (registry.root / inventory.ASSETS_FILENAME).resolve(),
        (registry.root / inventory.CAPTURES_FILENAME).resolve(),
    }
    original_open = pathlib.Path.open
    row_reads: list[pathlib.Path] = []
    expected = inventory.InventoryError("synthetic registry rejection")
    validation_calls: list[str | pathlib.Path] = []

    def reject_generation(path: str | pathlib.Path):
        validation_calls.append(path)
        raise expected

    def guard_rows(path: pathlib.Path, *args: Any, **kwargs: Any):
        if path.resolve() in protected:
            row_reads.append(path)
            pytest.fail("The generator read a registry row before validation returned.")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(inventory, "validate_generation", reject_generation)
    monkeypatch.setattr(pathlib.Path, "open", guard_rows)
    with pytest.raises(inventory.InventoryError) as raised:
        module.run(registry.root, registry.corpus, registry.out)
    assert raised.value is expected
    assert validation_calls == [registry.root]
    assert row_reads == []
    assert not registry.out.exists()


def test_p10_public_library_surface_and_run_path_types(tmp_path: pathlib.Path) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(1, "left")])
    module = _sessions_module()
    public = {
        "run",
        "plan",
        "validate_generation",
        "decode_source_path",
        "resolve_source",
        "tree_digest",
        "render_manifest",
        "render_summary",
        "SessionsError",
        "Event",
        "Camera",
        "Placement",
        "main",
    }
    assert all(hasattr(module, name) for name in public)
    assert list(inspect.signature(module.run).parameters) == [
        "inventory_dir",
        "corpus_root",
        "out_dir",
    ]
    validation_parameters = inspect.signature(module.validate_generation).parameters
    assert list(validation_parameters) == ["out_dir", "inventory_dir"]
    assert validation_parameters["inventory_dir"].default is None

    path_out = tmp_path / "sessions-path"
    str_out = tmp_path / "sessions-str"
    path_events, path_placements = module.run(registry.root, registry.corpus, path_out)
    str_events, str_placements = module.run(str(registry.root), str(registry.corpus), str(str_out))
    assert all(isinstance(event, module.Event) for event in (*path_events, *str_events))
    assert all(
        isinstance(placement, module.Placement) for placement in (*path_placements, *str_placements)
    )
    assert [dataclasses.asdict(event) for event in path_events] == [
        dataclasses.asdict(event) for event in str_events
    ]
    assert [dataclasses.asdict(placement) for placement in path_placements] == [
        dataclasses.asdict(placement) for placement in str_placements
    ]


def test_p10_inventory_validate_generation_accepts_str_and_path_identically(
    tmp_path: pathlib.Path,
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    expected = inventory.validate_generation(registry.root)
    assert inventory.validate_generation(str(registry.root)) == expected


def _escaped_asset(subject: int, component: str, encoded_component: str) -> _Asset:
    filename = f"{subject}_above_cap_l.MOV"
    return _Asset(
        source_path=f"{component}/{filename}",
        source_cell=f"{encoded_component}/{filename}",
        subject_ordinal=subject,
        view="above",
        task="cap",
        side="l",
    )


def test_p11_escaped_source_paths_round_trip_to_four_distinct_targets(
    tmp_path: pathlib.Path,
) -> None:
    raw_byte_component = os.fsdecode(b"raw\x80component")
    assets = [
        _escaped_asset(1, r"slash\component", r"slash\\component"),
        _escaped_asset(2, "line\ncomponent", r"line\x0acomponent"),
        _escaped_asset(3, "unicode" + chr(0x80) + "component", r"unicode\xc2\x80component"),
        _escaped_asset(4, raw_byte_component, r"raw\x80component"),
    ]
    module = _sessions_module()
    decoded = [module.decode_source_path(asset.source_cell) for asset in assets]
    assert decoded == [asset.source_path for asset in assets]
    assert len({os.fsencode(path) for path in decoded}) == 4
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    rows = {row["asset_id"]: row for row in _placements(registry)}
    resolved_targets = set()
    for asset in assets:
        placement = rows[asset.asset_id]
        link = registry.out / placement["event_id"] / "cam-above.mov"
        assert link.is_symlink()
        target = link.resolve(strict=True)
        assert target == (registry.corpus / asset.source_path).resolve(strict=True)
        resolved_targets.add(target)
    assert len(resolved_targets) == 4


@pytest.mark.parametrize(
    "source_cell",
    [
        r"synthetic\q/1_above_cap_l.MOV",
        r"synthetic\x0/1_above_cap_l.MOV",
        r"synthetic\xgg/1_above_cap_l.MOV",
        r"synthetic\xAB/1_above_cap_l.MOV",
        "synthetic/1_above_cap_l.MOV\\",
    ],
)
def test_p11_malformed_source_path_escape_fails_the_run(
    tmp_path: pathlib.Path, source_cell: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cell this project's encoder never wrote means a corrupt registry.

    Holding the row out would drop a camera and publish the smaller event as
    if it were whole, so the run refuses instead.
    """
    module = _sessions_module()
    with pytest.raises(module.SessionsError) as raised:
        module.decode_source_path(source_cell)
    assert raised.value.reason == "source_path_unsafe"

    asset = dataclasses.replace(_canonical(1, "above"), source_cell=source_cell)
    registry = _write_registry(tmp_path, [asset])
    _run_main(registry, expected_status=2)
    assert not registry.out.exists()
    assert "escape" in capsys.readouterr().err


def test_p12_symlink_names_use_lowercase_discoverable_video_extensions(
    tmp_path: pathlib.Path,
) -> None:
    extensions = (".MOV", ".mP4", ".AVI", ".MKV", ".WEBM")
    assets = [
        _canonical(subject, "above", task="peg", extension=extension)
        for subject, extension in enumerate(extensions, 1)
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    rows = {row["asset_id"]: row for row in _placements(registry)}
    for asset, extension in zip(assets, extensions, strict=True):
        placement = rows[asset.asset_id]
        event_dir = registry.out / placement["event_id"]
        link = event_dir / f"cam-above{extension.lower()}"
        assert link.is_symlink()
        assert link.suffix == extension.lower()
        assert link.suffix in multicam.VIDEO_EXTENSIONS
        assert multicam._find_glob_for_name(event_dir, placement["camera_name"]) == link.resolve()
    assert len(multicam.discover_sessions(registry.out)) == len(assets)


def test_p12_inventory_only_extension_is_held_out_as_undiscoverable(
    tmp_path: pathlib.Path,
) -> None:
    asset = _canonical(1, "above", extension=".FLV")
    registry = _write_registry(tmp_path, [asset])
    _run_main(registry)

    assert _events(registry) == []
    placement = _placements(registry)[0]
    assert placement["asset_id"] == asset.asset_id
    assert placement["placement"] == "held_out"
    assert placement["placement_reason"] == "extension_not_discoverable"
    assert placement["event_id"] == ""
    assert placement["camera_name"] == ""


def _contains_token(value: Any, tokens: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_token(key, tokens) or _contains_token(item, tokens)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_token(item, tokens) for item in value)
    return isinstance(value, str) and any(token in value for token in tokens)


def test_p13_console_tables_manifests_and_tree_names_redact_source_names(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens = ("private-subject-token", "private-filename-token")
    assets = [
        _Asset(
            source_path=f"{tokens[0]}/{tokens[1]}.MOV",
            subject_ordinal=1,
            view="above",
            task="cap",
            side="l",
        ),
        _held_out(
            f"{tokens[0]}/held-{tokens[1]}.MOV",
            disposition=inventory.QUARANTINED,
            reason_code="side_missing",
        ),
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)
    console = capsys.readouterr()

    assert not any(token in console.out for token in tokens)
    assert not any(token in console.err for token in tokens)
    for name in ("events.csv", "placements.csv", "generation.json"):
        text = (registry.out / name).read_text(encoding="utf-8")
        assert not any(token in text for token in tokens)
    for event in _events(registry):
        event_id = event["event_id"]
        assert (registry.out / event_id).is_dir()
        assert event_id.startswith(event["capture_id"] + "_run-")
        manifest = _manifest(registry, event_id)
        assert not _contains_token(manifest, tokens)
        assert "calibration" not in manifest
        assert all("file" not in camera for camera in manifest["cameras"])
    assert not list(registry.out.rglob("calibration.json"))


def test_p13_missing_source_fails_the_run_and_redacts_the_listed_path(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    token = "private-missing-source-token"
    asset = dataclasses.replace(_canonical(1, "above", directory=token), create_source=False)
    registry = _write_registry(tmp_path, [asset])
    _run_main(registry, expected_status=2)

    captured = capsys.readouterr()
    assert token not in captured.out
    assert token not in captured.err
    # The asset_id is a pseudonym, so the error names it and stays path-free.
    assert asset.asset_id in captured.err
    assert not registry.out.exists()


def _absolute(path: str | os.PathLike[str]) -> pathlib.Path:
    return pathlib.Path(path).absolute()


def _capture_root_renames(
    monkeypatch: pytest.MonkeyPatch,
    out: pathlib.Path,
    observe: Callable[[], None],
) -> list[tuple[pathlib.Path, pathlib.Path]]:
    moves: list[tuple[pathlib.Path, pathlib.Path]] = []
    for name in ("rename", "replace"):
        original = getattr(os, name)

        def wrapped(src, dst, *args, _original=original, **kwargs):
            source = _absolute(src)
            destination = _absolute(dst)
            root_move = source == out or destination == out
            if root_move:
                observe()
            result = _original(src, dst, *args, **kwargs)
            if root_move:
                moves.append((source, destination))
                observe()
            return result

        monkeypatch.setattr(os, name, wrapped)
    return moves


def test_p14_initial_publication_uses_one_exact_sibling_staging_rename(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(1, "left")])
    out = registry.out.absolute()
    observations: list[frozenset[str] | None] = []

    def observe() -> None:
        observations.append(
            frozenset(session.session_id for session in multicam.discover_sessions(out))
            if out.is_dir()
            else None
        )

    moves = _capture_root_renames(monkeypatch, out, observe)
    _run_main(registry)

    expected_ids = frozenset(row["event_id"] for row in _events(registry))
    staging = out.with_name(f"{out.name}.staging.{os.getpid()}")
    assert moves == [(staging, out)]
    assert observations == [None, expected_ids]
    assert not staging.exists()


def test_p14_republication_builds_in_a_sibling_and_swaps_with_two_root_renames(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sessions_module()
    registry = _write_registry(
        tmp_path,
        [_canonical(1, "above"), _canonical(1, "left"), _canonical(2, "right")],
    )
    _run_main(registry)
    expected_ids = frozenset(row["event_id"] for row in _events(registry))
    observations: list[frozenset[str] | None | Exception] = []
    out = registry.out.absolute()

    def observe() -> None:
        if not out.is_dir():
            observations.append(None)
            return
        try:
            observations.append(
                frozenset(session.session_id for session in multicam.discover_sessions(out))
            )
        except Exception as exc:  # A partial child is the failure under test.
            observations.append(exc)

    moves = _capture_root_renames(monkeypatch, out, observe)
    _run_main(registry)

    assert len(moves) == 2
    old_source, backup = moves[0]
    staging, new_destination = moves[1]
    assert old_source == out
    assert new_destination == out
    assert backup == out.with_name(f"{out.name}.retiring.{os.getpid()}")
    assert staging == out.with_name(f"{out.name}.staging.{os.getpid()}")
    assert observations == [expected_ids, None, None, expected_ids]
    assert not backup.exists()
    assert not staging.exists()


def test_p15_generator_never_lists_the_corpus(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sessions_module()
    registry = _write_registry(
        tmp_path,
        [_canonical(1, "above"), _canonical(1, "left"), _canonical(2, "right")],
    )
    (registry.corpus / "unlisted-decoy.MOV").write_bytes(b"not in the registry")
    corpus = registry.corpus.resolve()
    original_iterdir = pathlib.Path.iterdir
    original_glob = pathlib.Path.glob
    original_rglob = pathlib.Path.rglob
    original_listdir = os.listdir
    original_scandir = os.scandir

    def reject_corpus(path: Any) -> None:
        if isinstance(path, int):
            return
        candidate = pathlib.Path(path).resolve()
        if candidate == corpus or candidate.is_relative_to(corpus):
            raise AssertionError(f"The generator listed the corpus through {candidate.name!r}.")

    def guarded_iterdir(path: pathlib.Path):
        reject_corpus(path)
        return original_iterdir(path)

    def guarded_glob(path: pathlib.Path, *args: Any, **kwargs: Any):
        reject_corpus(path)
        return original_glob(path, *args, **kwargs)

    def guarded_rglob(path: pathlib.Path, *args: Any, **kwargs: Any):
        reject_corpus(path)
        return original_rglob(path, *args, **kwargs)

    def guarded_listdir(path: Any = "."):
        reject_corpus(path)
        return original_listdir(path)

    def guarded_scandir(path: Any = "."):
        reject_corpus(path)
        return original_scandir(path)

    monkeypatch.setattr(pathlib.Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr(pathlib.Path, "glob", guarded_glob)
    monkeypatch.setattr(pathlib.Path, "rglob", guarded_rglob)
    monkeypatch.setattr(os, "listdir", guarded_listdir)
    monkeypatch.setattr(os, "scandir", guarded_scandir)
    _run_main(registry)

    assert len(_placements(registry)) == len(registry.assets)


def test_p16_every_symlink_target_is_relative_regular_and_corpus_contained(
    tmp_path: pathlib.Path,
) -> None:
    assets = [
        _canonical(1, "above", extension=".MOV"),
        _canonical(1, "left", extension=".MP4"),
        _canonical(2, "right", task="coin", extension=".AVI"),
    ]
    registry = _write_registry(tmp_path, assets)
    _run_main(registry)

    placements = {row["asset_id"]: row for row in _placements(registry)}
    corpus = registry.corpus.resolve()
    for asset in assets:
        row = placements[asset.asset_id]
        link = (
            registry.out
            / row["event_id"]
            / f"{row['camera_name']}{pathlib.Path(asset.source_path).suffix.lower()}"
        )
        target = link.readlink()
        resolved = (link.parent / target).resolve(strict=True)
        expected = os.path.relpath(registry.corpus / asset.source_path, link.parent)
        assert link.is_symlink()
        assert not target.is_absolute()
        assert os.fspath(target) == expected
        assert resolved.is_relative_to(corpus)
        assert stat.S_ISREG(resolved.stat().st_mode)


@pytest.mark.parametrize(
    "case",
    [
        "absolute",
        "empty_component",
        "dot_component",
        "parent_component",
        "nul",
        "missing",
        "directory",
        "outside_symlink",
    ],
)
def test_p16_listed_path_validation_rejects_unsafe_or_non_file_targets(
    tmp_path: pathlib.Path, case: str
) -> None:
    base = _canonical(1, "above")
    filename = pathlib.PurePosixPath(base.source_path).name
    cells = {
        "absolute": f"/synthetic/{filename}",
        "empty_component": f"synthetic//{filename}",
        "dot_component": f"synthetic/./{filename}",
        "parent_component": f"synthetic/../{filename}",
        "nul": rf"synthetic\x00/{filename}",
    }
    asset = dataclasses.replace(
        base,
        source_cell=cells.get(case),
        create_source=case != "missing",
    )
    registry = _write_registry(tmp_path, [asset])
    source = registry.corpus / asset.source_path
    if case == "directory":
        source.unlink()
        source.mkdir()
    elif case == "outside_symlink":
        outside = tmp_path / "outside-target.MOV"
        outside.write_bytes(b"outside")
        source.unlink()
        source.symlink_to(outside)

    _run_main(registry, expected_status=2)
    assert not registry.out.exists()


@pytest.mark.parametrize("marker", ["missing", "malformed", "array", "foreign"])
def test_p17_nonempty_unowned_output_is_refused_without_modification(
    tmp_path: pathlib.Path, marker: str
) -> None:
    """Ownership is the marker's shape: a foreign one must not license deletion."""
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    registry.out.mkdir()
    (registry.out / "owner-sentinel.txt").write_text("preserve me", encoding="utf-8")
    text = {
        "malformed": "{not-json\n",
        "array": "[]\n",
        "foreign": json.dumps({"schema": "someone-elses-tool"}) + "\n",
    }.get(marker)
    if text is not None:
        (registry.out / "generation.json").write_text(text, encoding="utf-8")
    before = _tree_snapshot(registry.out)

    _run_main(registry, expected_status=2)
    assert _tree_snapshot(registry.out) == before


def test_p17_stale_but_well_formed_marker_stays_adoptable(tmp_path: pathlib.Path) -> None:
    """Ownership never consults digests, or a stale tree could never regenerate."""
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    marker = registry.out / "generation.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["tree"] = "0" * 64
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(_sessions_module().SessionsError):
        _sessions_module().validate_generation(registry.out)

    _run_main(registry)
    assert _sessions_module().validate_generation(registry.out)


@pytest.mark.parametrize("tamper", ["added_key", "renamed_key", "wrong_version"])
def test_p09_generation_schema_is_closed(tmp_path: pathlib.Path, tamper: str) -> None:
    """No digest inside the document can catch a key the writer never wrote."""
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    marker = registry.out / "generation.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if tamper == "added_key":
        payload["extra"] = True
    elif tamper == "renamed_key":
        payload["tree_digest"] = payload.pop("tree")
    else:
        payload["generator_version"] = "v0"
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(_sessions_module().SessionsError):
        _sessions_module().validate_generation(registry.out)


def test_p03_run_index_beyond_the_two_digit_grammar_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A 100th run would print an id that EVENT_ID_PATTERN rejects."""
    module = _sessions_module()
    for count, expect_events in ((99, 99), (100, None)):
        assets = [_canonical(1, "above", repeat=index) for index in range(count)]
        root = tmp_path / f"n{count}"
        root.mkdir()
        registry = _write_registry(root, assets)
        if expect_events is None:
            _run_main(registry, expected_status=2)
            assert not registry.out.exists()
        else:
            _run_main(registry)
            events = _events(registry)
            assert len(events) == expect_events
            assert events[-1]["event_id"].endswith("_run-99")
            assert all(module.EVENT_ID_PATTERN.match(row["event_id"]) for row in events)


def test_p17_empty_unowned_output_is_safe_to_adopt(tmp_path: pathlib.Path) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    registry.out.mkdir()

    _run_main(registry)
    assert len(_events(registry)) == 1
    assert _sessions_module().validate_generation(registry.out)


def test_p18_manifest_and_glob_discovery_accept_generated_out_of_tree_symlink(
    tmp_path: pathlib.Path,
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    event_id = _events(registry)[0]["event_id"]
    event_dir = registry.out / event_id
    manifest = _manifest(registry, event_id)
    assert "file" not in manifest["cameras"][0]

    through_manifest = multicam.discover_session(event_dir)
    link = event_dir / "cam-above.mov"
    target = link.resolve(strict=True)
    assert link.is_symlink()
    assert not target.is_relative_to(event_dir.resolve())
    assert through_manifest.cameras[0].file == target

    (event_dir / multicam.SESSION_MANIFEST_FILENAME).unlink()
    through_glob = multicam.discover_session(event_dir)
    assert through_glob.session_id == through_manifest.session_id == event_id
    assert through_glob.camera_names() == through_manifest.camera_names() == ["cam-above"]
    assert through_glob.cameras[0].file == through_manifest.cameras[0].file == target
    assert target.is_file()


# P19 — publication deletes and replaces the whole output tree, so the output
# must overlap neither input, and the swap must not consume a symlinked --out.


def _mark_owned(directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _sessions_module().GENERATION_FILENAME).write_text(
        json.dumps({"generator_version": _sessions_module().GENERATOR_VERSION}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "shape", ["is_corpus", "is_registry", "encloses_both", "inside_corpus", "inside_registry"]
)
def test_p19_output_overlapping_an_input_is_refused_without_deleting_it(
    tmp_path: pathlib.Path, shape: str
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(1, "front")])
    census = inventory.validate_generation(registry.root)
    out = {
        "is_corpus": registry.corpus,
        "is_registry": registry.root,
        "encloses_both": tmp_path,
        "inside_corpus": registry.corpus / "nested",
        "inside_registry": registry.root / "nested",
    }[shape]
    # The marker keeps ownership from being what refuses: overlap alone must.
    _mark_owned(out)

    with pytest.raises(_sessions_module().SessionsError):
        _sessions_module().run(registry.root, registry.corpus, out)

    assert inventory.validate_generation(registry.root) == census
    for asset in registry.assets:
        assert (registry.corpus / asset.source_path).read_bytes() == asset.content


def test_p19_symlinked_output_publishes_through_the_link_across_republication(
    tmp_path: pathlib.Path,
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    target = tmp_path / "volume" / "sessions"
    target.mkdir(parents=True)
    registry.out.symlink_to(target, target_is_directory=True)

    for _ in range(2):
        _run_main(registry)
        assert registry.out.is_symlink()
        assert (target / "events.csv").is_file()
        assert len(_events(registry)) == 1
        assert _sessions_module().validate_generation(registry.out) == (
            _sessions_module().validate_generation(target)
        )
        for parent in (tmp_path, target.parent):
            debris = sorted(parent.glob("*.staging.*")) + sorted(parent.glob("*.retiring.*"))
            assert debris == []


def test_p19_dead_publication_siblings_are_swept_link_debris_included(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    out = registry.out
    dead_directory = out.with_name(f"{out.name}.staging.10101")
    dead_link = out.with_name(f"{out.name}.retiring.10102")
    live = out.with_name(f"{out.name}.staging.20202")
    unrelated = out.with_name(f"{out.name}.note")
    dead_directory.mkdir()
    (dead_directory / "leftover").write_bytes(b"debris")
    dead_link.symlink_to(registry.corpus, target_is_directory=True)
    live.mkdir()
    unrelated.mkdir()

    def liveness(pid: int, signal: int) -> None:
        assert signal == 0
        if pid != 20202:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", liveness)
    _run_main(registry)

    assert not dead_directory.exists()
    assert not dead_link.is_symlink()
    assert live.is_dir()
    assert unrelated.is_dir()
    for asset in registry.assets:
        assert (registry.corpus / asset.source_path).read_bytes() == asset.content


# P20 — upstream digests prove the registry's bytes, never its shape, so every
# field this module reads is checked rather than trusted.


def _tables(registry: _Registry) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return (
        _read_csv(registry.root / inventory.ASSETS_FILENAME),
        _read_csv(registry.root / inventory.CAPTURES_FILENAME),
    )


def _drop_column(rows: list[dict[str, str]], column: str) -> None:
    for row in rows:
        del row[column]


def _set_all(rows: list[dict[str, str]], **cells: str) -> None:
    for row in rows:
        row.update(cells)


_SHAPE_DEFECTS: dict[str, Callable[[list[dict[str, str]], list[dict[str, str]]], None]] = {
    "asset_column_missing": lambda assets, captures: _drop_column(assets, "source_path"),
    "capture_column_missing": lambda assets, captures: _drop_column(captures, "view_conflict"),
    "duplicate_asset_id": lambda assets, captures: assets.append(dict(assets[0])),
    "duplicate_capture_id": lambda assets, captures: captures.append(dict(captures[0])),
    "unknown_disposition": lambda assets, captures: assets[0].update(disposition="archived"),
    "orphan_canonical_asset": lambda assets, captures: captures.clear(),
    "non_numeric_subject": lambda assets, captures: assets[0].update(subject_ordinal="one"),
    # str.isdigit is true for both: the first raises ValueError from int(), the
    # second silently normalizes to a published ordinal its own cell never spells.
    "superscript_subject": lambda assets, captures: assets[0].update(subject_ordinal="²"),
    "arabic_indic_subject": lambda assets, captures: assets[0].update(subject_ordinal="٢"),
    "view_conflict_out_of_domain": lambda assets, captures: captures[0].update(view_conflict="2"),
    "view_conflict_overclaimed": lambda assets, captures: captures[0].update(view_conflict="1"),
    "view_conflict_underclaimed": lambda assets, captures: assets.append(
        {**assets[0], "asset_id": "duplicate-view", "source_path": "synthetic-01/copy.MOV"}
    ),
    "mixed_grammar_version": lambda assets, captures: assets[0].update(grammar_version="v2"),
    "formula_in_view": lambda assets, captures: assets[0].update(view="=cmd|' /c calc'!A1"),
    "formula_in_task": lambda assets, captures: assets[0].update(task="+2+3"),
    "formula_in_digest": lambda assets, captures: assets[0].update(content_sha256="=1+1"),
    "formula_in_grammar": lambda assets, captures: _set_all(
        assets + captures, grammar_version="=v1"
    ),
    "escape_in_asset_id": lambda assets, captures: assets[0].update(asset_id="a-\x1b[31mred"),
    "newline_in_capture_id": lambda assets, captures: _set_all(
        assets + captures, capture_id="s01-cap-l\nrm -rf /"
    ),
    # Python `$` matches before one trailing newline, so an anchored `match`
    # admits every cell below; only the whole-string form rejects them.
    "trailing_newline_in_asset_id": lambda assets, captures: assets[0].update(asset_id="a-safe\n"),
    "trailing_newline_in_capture_id": lambda assets, captures: _set_all(
        assets + captures, capture_id="s01-cap-l\n"
    ),
    "trailing_newline_in_digest": lambda assets, captures: assets[0].update(
        content_sha256="abcdef\n"
    ),
    "trailing_newline_in_grammar": lambda assets, captures: _set_all(
        assets + captures, grammar_version="v1\n"
    ),
    "trailing_newline_in_side": lambda assets, captures: assets[0].update(side="l\n"),
    "trailing_newline_in_task": lambda assets, captures: assets[0].update(task="cap\n"),
    "trailing_newline_in_view": lambda assets, captures: assets[0].update(view="above\n"),
}


@pytest.mark.parametrize("defect", sorted(_SHAPE_DEFECTS))
def test_p20_registry_shape_defects_raise_inside_the_error_domain(
    tmp_path: pathlib.Path, defect: str
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(1, "front")])
    assets, captures = _tables(registry)
    _SHAPE_DEFECTS[defect](assets, captures)

    with pytest.raises(_sessions_module().SessionsError):
        _sessions_module().plan(assets, captures, corpus_root=registry.corpus)


def _republish_table(registry: _Registry, filename: str, text: str) -> None:
    """Replace one table verbatim and re-digest, so only its shape is wrong."""
    census_path = registry.root / inventory.CENSUS_FILENAME
    census = json.loads(census_path.read_text(encoding="utf-8"))
    census["generation"][filename] = hashlib.sha256(text.encode()).hexdigest()
    census["generation"][inventory.CENSUS_FILENAME] = inventory.census_digest(census)
    (registry.root / filename).write_text(text, encoding="utf-8", newline="")
    census_path.write_text(inventory.render_json(census), encoding="utf-8", newline="")
    assert inventory.validate_generation(registry.root) == census


def _empty_registry(registry: _Registry, *, drop: dict[str, str]) -> None:
    for filename, columns in (
        (inventory.ASSETS_FILENAME, inventory.ASSET_COLUMNS),
        (inventory.CAPTURES_FILENAME, inventory.CAPTURE_COLUMNS),
    ):
        kept = tuple(column for column in columns if column != drop.get(filename))
        _republish_table(registry, filename, inventory.render_csv(kept, []))


def test_p20_an_empty_table_still_publishes_when_its_header_is_whole(
    tmp_path: pathlib.Path,
) -> None:
    """Control for the refusal below: emptiness alone is a legal registry."""
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _empty_registry(registry, drop={})

    _run_main(registry)
    assert _events(registry) == []


@pytest.mark.parametrize(
    "drop",
    [
        {inventory.ASSETS_FILENAME: "source_path"},
        {inventory.CAPTURES_FILENAME: "view_conflict"},
    ],
    ids=["assets", "captures"],
)
def test_p20_a_short_header_is_refused_even_with_no_rows(
    tmp_path: pathlib.Path, drop: dict[str, str]
) -> None:
    """Zero rows leave the per-row checks unreached, so the header carries the schema."""
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _empty_registry(registry, drop=drop)

    _run_main(registry, expected_status=2)
    assert not registry.out.exists()


def test_p20_the_published_grammar_rejects_a_trailing_newline() -> None:
    """The exported pattern is matched by consumers, so it must anchor at the end."""
    assert not _sessions_module().EVENT_ID_PATTERN.match("s02-cap-l_run-01\n")
    assert _sessions_module().EVENT_ID_PATTERN.match("s02-cap-l_run-01")


def test_p20_a_hostile_cell_never_reaches_the_error_that_rejects_it(
    tmp_path: pathlib.Path,
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    assets, captures = _tables(registry)
    hostile = "a-\x1b[31m\nrm -rf /"
    assets[0].update(asset_id=hostile)

    with pytest.raises(_sessions_module().SessionsError) as raised:
        _sessions_module().plan(assets, captures, corpus_root=registry.corpus)

    message = str(raised.value)
    assert "asset_id" in message
    assert hostile not in message
    assert "\x1b" not in message
    assert "\n" not in message


def test_p20_placements_publish_the_asset_grammar_version(tmp_path: pathlib.Path) -> None:
    registry = _write_registry(
        tmp_path,
        [
            _canonical(1, "above"),
            _held_out(
                "synthetic-02/2_unknown.MOV",
                disposition=inventory.QUARANTINED,
                reason_code="side_missing",
            ),
        ],
    )
    _run_main(registry, strict=True, expected_status=1)

    rows = _placements(registry)
    assert [row["placement"] for row in rows].count("held_out") == 1
    assert {row["grammar_version"] for row in rows} == {inventory.GRAMMAR_VERSION}
    header = (registry.out / "placements.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(_sessions_module().PLACEMENT_COLUMNS)


def test_p20_manifest_subject_ordinal_is_a_json_number(tmp_path: pathlib.Path) -> None:
    registry = _write_registry(tmp_path, [_canonical(7, "above")])
    _run_main(registry)
    event_id = _events(registry)[0]["event_id"]

    ordinal = _manifest(registry, event_id)["subject_ordinal"]
    assert ordinal == 7
    assert not isinstance(ordinal, str)
    assert _events(registry)[0]["subject_ordinal"] == "7"


def test_p20_source_removed_after_planning_fails_before_publication(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    module = _sessions_module()
    planned = module.plan

    def plan_then_remove(*args: Any, **kwargs: Any) -> Any:
        result = planned(*args, **kwargs)
        (registry.corpus / registry.assets[0].source_path).unlink()
        return result

    monkeypatch.setattr(module, "plan", plan_then_remove)
    with pytest.raises(module.SessionsError) as raised:
        module.run(registry.root, registry.corpus, registry.out)

    assert raised.value.reason == "source_missing"
    assert not registry.out.exists()
    assert sorted(tmp_path.glob(f"{registry.out.name}.*")) == []


def test_p20_failed_republication_leaves_the_published_tree_intact(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(2, "front")])
    _run_main(registry)
    published = _tree_snapshot(registry.out)
    module = _sessions_module()
    planned = module.plan

    def plan_then_remove(*args: Any, **kwargs: Any) -> Any:
        result = planned(*args, **kwargs)
        (registry.corpus / registry.assets[0].source_path).unlink()
        return result

    monkeypatch.setattr(module, "plan", plan_then_remove)
    with pytest.raises(module.SessionsError):
        module.run(registry.root, registry.corpus, registry.out)

    assert _tree_snapshot(registry.out) == published
    assert module.validate_generation(registry.out)
    assert sorted(tmp_path.glob(f"{registry.out.name}.*")) == []


_SWAP_REFUSED = "the swap is refused"


def _fail_the_swap(
    monkeypatch: pytest.MonkeyPatch, *, before: Callable[[], None] = lambda: None
) -> None:
    """Break `staging → out` alone, leaving the retirement rename working."""
    renamed = pathlib.Path.rename

    def rename(self: pathlib.Path, target: Any) -> pathlib.Path:
        if f".staging.{os.getpid()}" in self.name:
            before()
            raise OSError(_SWAP_REFUSED)
        return renamed(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", rename)


def test_p20_a_failed_swap_restores_the_previous_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(2, "front")])
    _run_main(registry)
    published = _tree_snapshot(registry.out)
    _fail_the_swap(monkeypatch)

    with pytest.raises(OSError, match=_SWAP_REFUSED):
        _sessions_module().run(registry.root, registry.corpus, registry.out)

    assert _tree_snapshot(registry.out) == published
    assert _sessions_module().validate_generation(registry.out)
    assert sorted(tmp_path.glob(f"{registry.out.name}.*")) == []


def test_p20_a_peer_owning_the_root_keeps_the_retired_tree_on_disk(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    published = _tree_snapshot(registry.out)

    def publish_as_a_peer() -> None:
        registry.out.mkdir()
        (registry.out / "peer.txt").write_bytes(b"a peer owns the root now")

    _fail_the_swap(monkeypatch, before=publish_as_a_peer)
    with pytest.raises(OSError, match=_SWAP_REFUSED):
        _sessions_module().run(registry.root, registry.corpus, registry.out)

    retired = registry.out.with_name(f"{registry.out.name}.retiring.{os.getpid()}")
    assert _tree_snapshot(retired) == published
    assert (registry.out / "peer.txt").is_file()


def test_p20_a_failed_build_keeps_a_dead_runs_retired_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    abandoned = registry.out.with_name(f"{registry.out.name}.retiring.10101")
    shutil.copytree(registry.out, abandoned, symlinks=True)
    module = _sessions_module()
    planned = module.plan

    def plan_then_remove(*args: Any, **kwargs: Any) -> Any:
        result = planned(*args, **kwargs)
        (registry.corpus / registry.assets[0].source_path).unlink()
        return result

    monkeypatch.setattr(module, "plan", plan_then_remove)
    monkeypatch.setattr(os, "kill", lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError))
    with pytest.raises(module.SessionsError):
        module.run(registry.root, registry.corpus, registry.out)

    assert module.validate_generation(abandoned)


def test_p20_a_failed_swap_into_an_absent_root_keeps_the_dead_generation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between the two renames leaves the only generation under a dead pid.

    Sweeping it before the swap and then failing the swap destroys it, and the
    empty-root rollback raises over the real error while doing so.
    """
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    _run_main(registry)
    abandoned = registry.out.with_name(f"{registry.out.name}.retiring.10101")
    registry.out.rename(abandoned)
    dead_staging = registry.out.with_name(f"{registry.out.name}.staging.10101")
    shutil.copytree(abandoned, dead_staging, symlinks=True)
    _fail_the_swap(monkeypatch)
    monkeypatch.setattr(os, "kill", lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError))

    with pytest.raises(OSError, match=_SWAP_REFUSED):
        _sessions_module().run(registry.root, registry.corpus, registry.out)

    assert _sessions_module().validate_generation(abandoned)


def test_p20_an_unrepresentable_pid_suffix_sweeps_as_dead_debris(tmp_path: pathlib.Path) -> None:
    """`os.kill` raises OverflowError, not ValueError, on an int wider than a C long."""
    registry = _write_registry(tmp_path, [_canonical(1, "above")])
    debris = registry.out.with_name(f"{registry.out.name}.staging.{'9' * 100}")
    debris.mkdir(parents=True)
    (debris / "leftover").write_bytes(b"debris")

    _run_main(registry)

    assert not debris.exists()
    assert _sessions_module().validate_generation(registry.out)


def test_p20_filesystem_root_corpus_contains_every_file_below_it(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "asset.mov"
    source.write_bytes(b"synthetic")
    relative = str(source.resolve().relative_to("/"))

    assert _sessions_module().resolve_source("/", relative) == source.resolve()
