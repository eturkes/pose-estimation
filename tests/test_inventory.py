"""Executable contract for the canonical corpus inventory."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import pytest

inventory = importlib.import_module("pose_estimation.inventory")
video_io = importlib.import_module("pose_estimation.video_io")

ASSET_COLUMNS = (
    "asset_id",
    "capture_id",
    "disposition",
    "reason_code",
    "source_path",
    "subject_ordinal",
    "view",
    "task",
    "side",
    "repeat",
    "normalizations",
    "size_bytes",
    "content_sha256",
    "reported_width",
    "reported_height",
    "reported_avg_fps",
    "reported_frame_count",
    "reported_rotation_deg",
    "reported_fourcc",
    "nominal_duration_s",
    "fact_flags",
    "probe_status",
    "grammar_version",
    "tool_version",
)
CAPTURE_COLUMNS = (
    "capture_id",
    "subject_ordinal",
    "task",
    "side",
    "n_assets",
    "views",
    "n_views",
    "view_conflict",
    "reported_frame_count_min",
    "reported_frame_count_max",
    "reported_fps_min",
    "reported_fps_max",
    "reported_fps_spread_hz",
    "nominal_duration_min_s",
    "nominal_duration_max_s",
    "nominal_duration_spread_s",
    "reported_resolution_agree",
    "reported_rotation_agree",
    "grammar_version",
    "tool_version",
)
ARTIFACT_NAMES = ("assets.csv", "captures.csv", "census.json")
VIEWS = ("above", "left", "right")
TASKS = ("cap", "coin", "glass", "key", "nut", "peg")
SIDES = ("l", "r")
REASONS_BY_DISPOSITION = {
    "canonical": {"ok"},
    "quarantined": {
        "token_count",
        "subject_token_nonnumeric",
        "view_unknown",
        "task_unknown",
        "side_missing",
        "side_unknown",
        "subject_token_conflict",
        "repeat_marker_unrecognized",
    },
    "excluded": {
        "broken_symlink",
        "control_character_in_path",
        "not_a_regular_file",
        "path_escapes_root",
        "path_not_utf8",
        "probe_unreadable",
        "read_error",
        "symlink_within_corpus",
        "unsupported_extension",
    },
}
NORMALIZATION_STEPS = (
    "case_folded",
    "leading_separator_stripped",
    "media_suffix_doubled",
    "outer_trimmed",
    "repeat_marker",
    "task_repaired",
    "underscore_collapsed",
    "whitespace_collapsed",
)
IDENTITY_COLUMNS = ("capture_id", "subject_ordinal", "view", "task", "side", "repeat")


@pytest.fixture(scope="session")
def mjpg_video_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Return a tiny real MJPG/AVI container; skip when the host codec is absent."""
    root = tmp_path_factory.mktemp("inventory_mjpg")
    path = root / "probe.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter.fourcc(*"MJPG"),
        15.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("MJPG/AVI codec unavailable on this host")
    try:
        for value in (0, 64, 128):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("MJPG/AVI codec unavailable on this host")
    return path.read_bytes()


def _put(corpus: pathlib.Path, relative: str, payload: bytes) -> pathlib.Path:
    path = corpus / pathlib.PurePosixPath(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _invoke(
    corpus: pathlib.Path,
    out: pathlib.Path | None = None,
    *,
    checksums: bool = True,
    strict: bool = False,
) -> int:
    argv = ["--corpus", os.fspath(corpus)]
    if out is not None:
        argv.extend(("--out", os.fspath(out)))
    if not checksums:
        argv.append("--no-checksums")
    if strict:
        argv.append("--strict")
    result = inventory.main(argv)
    assert isinstance(result, int), "inventory.main() must return an integer status"
    return result


def _read_csv(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return tuple(reader.fieldnames), list(reader)


def _read_outputs(
    out: pathlib.Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    assert {path.name for path in out.iterdir()} == set(ARTIFACT_NAMES)
    asset_header, assets = _read_csv(out / "assets.csv")
    capture_header, captures = _read_csv(out / "captures.csv")
    assert asset_header == ASSET_COLUMNS
    assert capture_header == CAPTURE_COLUMNS
    with (out / "census.json").open(encoding="utf-8") as handle:
        census = json.load(handle)
    assert isinstance(census, dict)
    return assets, captures, census


def _artifact_bytes(out: pathlib.Path) -> dict[str, bytes]:
    return {name: (out / name).read_bytes() for name in ARTIFACT_NAMES}


def _asset_id(relative_posix: str) -> str:
    digest = hashlib.blake2b(
        relative_posix.encode("utf-8", "surrogateescape"),
        digest_size=8,
        person=b"pose3cam-asset",
    ).hexdigest()
    return f"a-{digest}"


def _by_source(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {row["source_path"]: row for row in rows}
    assert len(result) == len(rows)
    return result


def _parse_views(value: str) -> set[str]:
    if value.startswith("["):
        parsed = json.loads(value)
        assert isinstance(parsed, list)
        return {str(item) for item in parsed}
    return {item for item in re.split(r"[|,; ]+", value) if item}


def _assert_blank_identity(row: Mapping[str, str]) -> None:
    assert {column: row[column] for column in IDENTITY_COLUMNS} == dict.fromkeys(
        IDENTITY_COLUMNS, ""
    )


def _assert_disposition_partition(
    rows: Sequence[dict[str, str]], discovered: Sequence[str]
) -> Counter[str]:
    source_counts = Counter(row["source_path"] for row in rows)
    assert source_counts == Counter(dict.fromkeys(discovered, 1))
    dispositions = Counter(row["disposition"] for row in rows)
    assert set(dispositions) <= set(REASONS_BY_DISPOSITION)
    assert sum(dispositions.values()) == len(rows) == len(discovered)
    assert len(rows) == sum(dispositions[name] for name in REASONS_BY_DISPOSITION)
    return dispositions


def _assert_row_case(
    row: Mapping[str, str],
    *,
    disposition: str,
    reason: str,
    subject: str = "",
    view: str = "",
    task: str = "",
    side: str = "",
    repeat: str | None = None,
) -> None:
    assert row["disposition"] == disposition
    assert row["reason_code"] == reason
    assert row["subject_ordinal"] == subject
    assert row["view"] == view
    assert row["task"] == task
    assert row["side"] == side
    if repeat is not None:
        assert row["repeat"] == repeat
    if disposition == "canonical":
        assert row["capture_id"] == f"s{int(subject):02d}-{task}-{side}"
    else:
        _assert_blank_identity(row)


def _walk_json(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from _walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = (*path, str(index))
            yield child_path, child
            yield from _walk_json(child, child_path)


@dataclasses.dataclass(frozen=True)
class _FactSpec:
    opened: bool = True
    width: float = 64.0
    height: float = 48.0
    fps: float = 15.0
    frame_count: float = 3.0
    rotation: float = 0.0
    fourcc: int = dataclasses.field(default_factory=lambda: cv2.VideoWriter.fourcc(*"MJPG"))
    orientation_auto: float = 1.0
    backend: str = "FAKE"


class _HeaderCapture:
    def __init__(self, spec: _FactSpec):
        self.spec = spec
        self.released = False
        self.set_calls: list[tuple[int, float]] = []
        self.read_calls = 0
        self.grab_calls = 0

    def isOpened(self) -> bool:
        return self.spec.opened and not self.released

    def getBackendName(self) -> str:
        return self.spec.backend

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: self.spec.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.spec.height,
            cv2.CAP_PROP_FPS: self.spec.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.spec.frame_count,
            cv2.CAP_PROP_ORIENTATION_META: self.spec.rotation,
            cv2.CAP_PROP_ORIENTATION_AUTO: self.spec.orientation_auto,
            cv2.CAP_PROP_FOURCC: float(self.spec.fourcc),
        }
        if prop not in values:
            raise AssertionError(f"unexpected header property: {prop}")
        return values[prop]

    def read(self) -> tuple[bool, None]:
        self.read_calls += 1
        raise AssertionError("P14 forbids VideoCapture.read() during inventory")

    def grab(self) -> bool:
        self.grab_calls += 1
        raise AssertionError("P14 forbids VideoCapture.grab() during inventory")

    def release(self) -> None:
        self.released = True


class _CaptureFactory:
    def __init__(self, specs: Mapping[str, _FactSpec], default: _FactSpec | None = None):
        self.specs = dict(specs)
        self.default = default or _FactSpec()
        self.sources: list[pathlib.Path] = []
        self.captures: list[_HeaderCapture] = []

    def __call__(self, source: str | os.PathLike[str]) -> _HeaderCapture:
        path = pathlib.Path(os.fspath(source))
        capture = _HeaderCapture(self.specs.get(path.name, self.default))
        self.sources.append(path)
        self.captures.append(capture)
        return capture


def _patch_capture(
    monkeypatch: pytest.MonkeyPatch,
    specs: Mapping[str, _FactSpec] | None = None,
    *,
    default: _FactSpec | None = None,
) -> _CaptureFactory:
    factory = _CaptureFactory(specs or {}, default)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    monkeypatch.setattr(inventory, "VideoCapture", factory, raising=False)
    monkeypatch.setattr(video_io, "VideoCapture", factory, raising=False)
    return factory


def test_exact_artifact_schema_identity_and_order(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P01/P03/P05/P07: schemas, path IDs, grouping, floats, and code-point order."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    names = (
        "subject/3_above_peg_l.MOV",
        "subject/3_ABOVE_PEG_L.mov",
        "subject/3_left_key_r.MoV",
    )
    for name in names:
        _put(corpus, name, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, captures, _ = _read_outputs(out)

    assert [row["source_path"] for row in assets] == sorted(names)
    assert [row["asset_id"] for row in assets] == [_asset_id(name) for name in sorted(names)]
    assert len({row["asset_id"] for row in assets}) == len(assets)
    assert len({row["content_sha256"] for row in assets}) == 1
    assert _by_source(assets)["subject/3_ABOVE_PEG_L.mov"]["normalizations"] == "case_folded"
    assert _by_source(assets)["subject/3_above_peg_l.MOV"]["normalizations"] == ""
    assert _by_source(assets)["subject/3_left_key_r.MoV"]["normalizations"] == ""
    expected_content_hash = hashlib.sha256(mjpg_video_bytes).hexdigest()
    for row in assets:
        assert row["content_sha256"] == expected_content_hash
        assert row["size_bytes"] == str(len(mjpg_video_bytes))
        assert row["disposition"] == "canonical"
        assert row["reason_code"] == "ok"
        assert row["reported_width"] == "64"
        assert row["reported_height"] == "48"
        assert float(row["reported_avg_fps"]) == 15.0
        assert row["reported_frame_count"] == "3"
        assert row["reported_rotation_deg"] == "0"
        assert row["reported_fourcc"].lower() == "mjpg"
        assert float(row["nominal_duration_s"]) == 0.2
        assert row["grammar_version"] == "v1"
        assert row["tool_version"] == "v1"

    assert [row["capture_id"] for row in captures] == ["s03-key-r", "s03-peg-l"]
    capture_by_id = {row["capture_id"]: row for row in captures}
    assert capture_by_id["s03-key-r"]["n_assets"] == "1"
    assert capture_by_id["s03-peg-l"]["n_assets"] == "2"
    assert capture_by_id["s03-peg-l"]["n_views"] == "1"
    assert capture_by_id["s03-peg-l"]["view_conflict"] == "1"
    assert capture_by_id["s03-peg-l"]["views"] == "above"
    assert float(capture_by_id["s03-peg-l"]["reported_fps_min"]) == 15.0
    assert float(capture_by_id["s03-peg-l"]["reported_fps_max"]) == 15.0
    assert float(capture_by_id["s03-peg-l"]["reported_fps_spread_hz"]) == 0.0
    assert float(capture_by_id["s03-peg-l"]["nominal_duration_spread_s"]) == 0.0
    assert {row["grammar_version"] for row in assets + captures} == {"v1"}
    assert {row["tool_version"] for row in assets + captures} == {"v1"}


def test_every_regular_file_gets_exactly_one_row(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P01/A11: recurse through nested, root-level, hidden, and unsupported regular files."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    expected = (
        ".hidden",
        "root_note",
        "group/subject/27_right_nut_r.MOV",
        "group/subject/notes.txt",
    )
    for index, name in enumerate(expected):
        _put(corpus, name, bytes([index + 1]))
    empty = corpus / "empty_subject"
    empty.mkdir()
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    _assert_disposition_partition(assets, sorted(expected))
    assert [row["source_path"] for row in assets] == sorted(expected)
    assert census["subject_directories"] == 1


def test_disposition_partition_and_closed_reason_vocabulary(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P02/P04/P05 + §10 cases 7-15: recompute the exhaustive disposition spine."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    files = {
        "subject/3_above_peg_l.MOV": mjpg_video_bytes,
        "subject/3_right_cap_.MOV": mjpg_video_bytes,
        "subject/3_above_coin_7.MOV": mjpg_video_bytes,
        "subject/3_above_coin_l_extra.MOV": mjpg_video_bytes,
        "subject/x_above_peg_l.MOV": mjpg_video_bytes,
        "subject/3_front_peg_l.MOV": mjpg_video_bytes,
        "subject/3_above_spoon_l.MOV": mjpg_video_bytes,
        "subject/3_above_peg_l copy.MOV": mjpg_video_bytes,
        "subject/3_above_peg_l (x).MOV": mjpg_video_bytes,
        "subject/notes.txt": b"synthetic notes",
        "subject/3_left_key_r.MOV": b"not a media container",
    }
    for name, payload in files.items():
        _put(corpus, name, payload)

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    by_source = _by_source(assets)
    dispositions = _assert_disposition_partition(assets, sorted(files))
    assert dispositions == Counter(canonical=1, quarantined=8, excluded=2)

    expected = {
        "subject/3_above_peg_l.MOV": ("canonical", "ok"),
        "subject/3_right_cap_.MOV": ("quarantined", "side_missing"),
        "subject/3_above_coin_7.MOV": ("quarantined", "side_unknown"),
        "subject/3_above_coin_l_extra.MOV": ("quarantined", "token_count"),
        "subject/x_above_peg_l.MOV": ("quarantined", "subject_token_nonnumeric"),
        "subject/3_front_peg_l.MOV": ("quarantined", "view_unknown"),
        "subject/3_above_spoon_l.MOV": ("quarantined", "task_unknown"),
        "subject/3_above_peg_l copy.MOV": ("quarantined", "token_count"),
        "subject/3_above_peg_l (x).MOV": (
            "quarantined",
            "repeat_marker_unrecognized",
        ),
        "subject/notes.txt": ("excluded", "unsupported_extension"),
        "subject/3_left_key_r.MOV": ("excluded", "probe_unreadable"),
    }
    for source, (disposition, reason) in expected.items():
        row = by_source[source]
        assert row["disposition"] == disposition
        assert row["reason_code"] == reason
        assert reason in REASONS_BY_DISPOSITION[disposition]
        assert (disposition == "canonical") is (reason == "ok")
        if disposition != "canonical":
            _assert_blank_identity(row)

    assert census["assets"]["canonical"] == dispositions["canonical"]
    assert census["assets"]["quarantined"] == dispositions["quarantined"]
    assert census["assets"]["excluded"] == dispositions["excluded"]


def test_probe_and_parse_run_independently_for_every_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A27: parse and probe admitted files independently; pre-parse exclusion wins."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    names = (
        "subject/nonsense.MOV",
        "subject/3_above_peg_l.txt",
        "subject/3_left_key_r.MOV",
    )
    for name in names:
        _put(corpus, name, b"synthetic")
    factory = _patch_capture(
        monkeypatch,
        {"3_left_key_r.MOV": _FactSpec(opened=False)},
    )

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    rows = _by_source(assets)
    assert Counter(path.name for path in factory.sources) == Counter(
        pathlib.PurePosixPath(name).name for name in (names[0], names[2])
    )
    _assert_row_case(
        rows["subject/nonsense.MOV"],
        disposition="quarantined",
        reason="token_count",
    )
    _assert_row_case(
        rows["subject/3_above_peg_l.txt"],
        disposition="excluded",
        reason="unsupported_extension",
    )
    _assert_row_case(
        rows["subject/3_left_key_r.MOV"],
        disposition="excluded",
        reason="probe_unreadable",
    )
    assert rows["subject/3_left_key_r.MOV"]["fact_flags"] == ""


@pytest.mark.parametrize(
    ("filename", "subject", "view", "task", "side", "repeat", "normalizations"),
    [
        pytest.param("3_above_peg_l.MOV", "3", "above", "peg", "l", "0", "", id="N8-case-1"),
        pytest.param(
            "3_above_peg_l.mov.MOV",
            "3",
            "above",
            "peg",
            "l",
            "0",
            "media_suffix_doubled",
            id="N1-case-2",
        ),
        pytest.param(
            "3_ABOVE_Peg_L.MOV",
            "3",
            "above",
            "peg",
            "l",
            "0",
            "case_folded",
            id="N2-case-3",
        ),
        pytest.param("9_right_cap_l .MOV", "9", "right", "cap", "l", "0", None, id="N3-A17"),
        pytest.param(
            "3 above peg l.MOV",
            "3",
            "above",
            "peg",
            "l",
            "0",
            "whitespace_collapsed",
            id="N4-case-4a",
        ),
        pytest.param(
            "3   above  peg   l.MOV",
            "3",
            "above",
            "peg",
            "l",
            "0",
            "whitespace_collapsed",
            id="N4-run",
        ),
        pytest.param(
            "3_above__peg_l.MOV",
            "3",
            "above",
            "peg",
            "l",
            "0",
            "underscore_collapsed",
            id="N5-case-4b",
        ),
        pytest.param(
            "_3_above_peg_l.MOV",
            "3",
            "above",
            "peg",
            "l",
            "0",
            "leading_separator_stripped",
            id="N6",
        ),
        pytest.param(
            "3_left_cap_l (2).MOV",
            "3",
            "left",
            "cap",
            "l",
            "2",
            "repeat_marker|whitespace_collapsed",
            id="N7-case-6",
        ),
        pytest.param(
            "3_above_grass_l.MOV",
            "3",
            "above",
            "glass",
            "l",
            "0",
            "task_repaired",
            id="N9-grass",
        ),
        pytest.param(
            "3_above_gcap_l.MOV", "3", "above", "cap", "l", "0", "task_repaired", id="N9-gcap"
        ),
        pytest.param(
            "3_above_gpeg_l.MOV", "3", "above", "peg", "l", "0", "task_repaired", id="N9-gpeg"
        ),
        pytest.param(
            "3_above_coini_l.MOV", "3", "above", "coin", "l", "0", "task_repaired", id="N9-coini"
        ),
    ],
)
def test_normalization_steps_are_individually_observable(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
    filename: str,
    subject: str,
    view: str,
    task: str,
    side: str,
    repeat: str,
    normalizations: str | None,
) -> None:
    """P13/N1-N9 + §10 cases 1-6: isolate each ordered grammar transform."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    source = f"subject/{filename}"
    _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    assert len(assets) == 1
    _assert_row_case(
        assets[0],
        disposition="canonical",
        reason="ok",
        subject=subject,
        view=view,
        task=task,
        side=side,
        repeat=repeat,
    )
    if normalizations is not None:
        assert assets[0]["normalizations"] == normalizations


def test_inventory_extension_tuple_includes_flv_without_multicam_import(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P13/N1/A1: video_io owns six census extensions, including a canonical `.flv`."""
    extensions = video_io.__dict__["VIDEO_EXTENSIONS"]
    assert extensions == (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")
    assert set(extensions) == video_io.VIDEO_EXTS
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.flv", b"synthetic")
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    _assert_row_case(
        assets[0],
        disposition="canonical",
        reason="ok",
        subject="3",
        view="above",
        task="peg",
        side="l",
        repeat="0",
    )
    assert assets[0]["normalizations"] == ""
    assert census["extension_case"] == {".flv": 1}


def test_normalization_steps_compose_in_the_fixed_order(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P13/N1-N9 combined: one stem requires every transform in normative order."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    filename = "  __ 003   ABOVE__grass  L  (007).mov.MOV"
    _put(corpus, f"subject/{filename}", mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    _assert_row_case(
        assets[0],
        disposition="canonical",
        reason="ok",
        subject="3",
        view="above",
        task="glass",
        side="l",
        repeat="7",
    )
    assert assets[0]["normalizations"] == "|".join(NORMALIZATION_STEPS)


def test_trailing_whitespace_and_trailing_separator_diverge(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P13/N3/N6/A2/A17: outer space is canonical; a trailing `_` is `side_missing`."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    canonical_source = "subject/9_right_cap_l .MOV"
    missing_source = "subject/9_right_cap_.MOV"
    for source in (canonical_source, missing_source):
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    rows = _by_source(assets)
    _assert_row_case(
        rows[canonical_source],
        disposition="canonical",
        reason="ok",
        subject="9",
        view="right",
        task="cap",
        side="l",
        repeat="0",
    )
    _assert_row_case(
        rows[missing_source],
        disposition="quarantined",
        reason="side_missing",
    )


def test_normalization_order_dependencies_are_load_bearing(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P13/N4→N5→N6→N7→N8→N9: adjacent separators and marker cleanup compose."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    expected = {
        "subject/3 _ above _ peg _ l.MOV": ("above", "peg", "l", "0"),
        "subject/_3_left_gcap_r_(2).MOV": ("left", "cap", "r", "2"),
        "subject/3 RIGHT coini L (007).mov.MOV": ("right", "coin", "l", "7"),
    }
    for source in expected:
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    rows = _by_source(assets)
    for source, (view, task, side, repeat) in expected.items():
        _assert_row_case(
            rows[source],
            disposition="canonical",
            reason="ok",
            subject="3",
            view=view,
            task=task,
            side=side,
            repeat=repeat,
        )


def test_task_repair_is_closed_and_only_targets_the_task_token(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P13/N9: reject edit-distance guesses and never repair view or side positions."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    expected = {
        "subject/3_above_gkey_l.MOV": "task_unknown",
        "subject/3_above_coiin_l.MOV": "task_unknown",
        "subject/3_above_glcass_l.MOV": "task_unknown",
        "subject/3_gcap_peg_l.MOV": "view_unknown",
        "subject/3_above_peg_gcap.MOV": "side_unknown",
    }
    for source in expected:
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    rows = _by_source(assets)
    for source, reason in expected.items():
        _assert_row_case(rows[source], disposition="quarantined", reason=reason)


def test_normalization_row_trace_and_census_aggregates_are_identical(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A27: only parser-ran rows contribute trace and normalization census counts."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    sources = (
        "subject/3 ABOVE__grass L (2).mov.MOV",
        "subject/3_left_gcap_7.MOV",
        "subject/3 ABOVE grass L.txt",
        "subject/3_right_coini_r.mov.MOV",
    )
    for source in sources:
        _put(corpus, source, b"synthetic")
    _patch_capture(
        monkeypatch,
        {pathlib.PurePosixPath(sources[3]).name: _FactSpec(opened=False)},
    )

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    rows = _by_source(assets)
    expected = {
        sources[0]: (
            "canonical",
            "ok",
            "case_folded|media_suffix_doubled|repeat_marker|task_repaired|"
            "underscore_collapsed|whitespace_collapsed",
        ),
        sources[1]: ("quarantined", "side_unknown", "task_repaired"),
        sources[2]: ("excluded", "unsupported_extension", ""),
        sources[3]: (
            "excluded",
            "probe_unreadable",
            "media_suffix_doubled|task_repaired",
        ),
    }
    for source, (disposition, reason, normalizations) in expected.items():
        assert rows[source]["disposition"] == disposition
        assert rows[source]["reason_code"] == reason
        assert rows[source]["normalizations"] == normalizations

    applied = Counter(
        token for row in assets for token in row["normalizations"].split("|") if token
    )
    assert census["normalization"]["applied"] == dict(sorted(applied.items()))
    assert census["normalization"]["task_repairs"] == {
        "coini": 1,
        "gcap": 1,
        "grass": 1,
    }
    repair_total = sum(census["normalization"]["task_repairs"].values())
    assert repair_total == census["normalization"]["applied"]["task_repaired"]
    assert repair_total == sum(
        "task_repaired" in row["normalizations"].split("|") for row in assets
    )
    assert rows[sources[1]]["disposition"] == "quarantined"


def test_parse_result_depends_only_on_filename_and_subject_crosscheck(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P13: directory spelling and file content cannot change the per-filename parse."""
    corpora = (tmp_path / "corpus_a", tmp_path / "corpus_b")
    outputs = (tmp_path / "out_a", tmp_path / "out_b")
    _put(corpora[0], "alpha/3_ABOVE_grass_L (007).MOV", b"first-content")
    _put(corpora[1], "beta/3_ABOVE_grass_L (007).MOV", b"different-content")
    _patch_capture(monkeypatch)

    compared = (
        "capture_id",
        "disposition",
        "reason_code",
        "subject_ordinal",
        "view",
        "task",
        "side",
        "repeat",
        "normalizations",
    )
    parsed_rows = []
    for corpus, out in zip(corpora, outputs, strict=True):
        assert _invoke(corpus, out) == 0
        assets, _, _ = _read_outputs(out)
        assert len(assets) == 1
        parsed_rows.append(assets[0])
    assert {column: parsed_rows[0][column] for column in compared} == {
        column: parsed_rows[1][column] for column in compared
    }
    _assert_row_case(
        parsed_rows[0],
        disposition="canonical",
        reason="ok",
        subject="3",
        view="above",
        task="glass",
        side="l",
        repeat="7",
    )


def test_repeat_marker_edge_cases_preserve_the_marker_integer(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """A13/A22: accept positive markers; blank every identity on quarantined rows."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    files = (
        "subject/3_left_cap_l (0).MOV",
        "subject/3_right_cap_r (007).MOV",
        "subject/(2).MOV",
        "subject/3_above_key_l (x).MOV",
        "subject/3_above_nut_r copy.MOV",
        "subject/3_above_coin_l (2)(3).MOV",
    )
    for source in files:
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    rows = _by_source(assets)
    zero = rows["subject/3_left_cap_l (0).MOV"]
    _assert_row_case(
        zero,
        disposition="quarantined",
        reason="repeat_marker_unrecognized",
    )
    assert zero["repeat"] == ""
    assert zero["normalizations"] == "whitespace_collapsed"
    assert "repeat_marker" not in zero["normalizations"]
    positive = rows["subject/3_right_cap_r (007).MOV"]
    _assert_row_case(
        positive,
        disposition="canonical",
        reason="ok",
        subject="3",
        view="right",
        task="cap",
        side="r",
        repeat="7",
    )
    assert positive["normalizations"] == "repeat_marker|whitespace_collapsed"
    marker_only = rows["subject/(2).MOV"]
    _assert_row_case(marker_only, disposition="quarantined", reason="token_count")
    assert marker_only["repeat"] == ""
    assert marker_only["normalizations"] == "repeat_marker"
    invalid = rows[files[3]]
    _assert_row_case(
        invalid,
        disposition="quarantined",
        reason="repeat_marker_unrecognized",
    )
    assert invalid["normalizations"] == "whitespace_collapsed"
    copied = rows[files[4]]
    _assert_row_case(copied, disposition="quarantined", reason="token_count")
    assert copied["normalizations"] == "whitespace_collapsed"
    stacked = rows[files[5]]
    _assert_row_case(stacked, disposition="quarantined", reason="token_count")
    assert stacked["repeat"] == ""
    assert stacked["normalizations"] == "repeat_marker|whitespace_collapsed"


def test_subject_directory_conflicts_quarantine_every_parsed_asset(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P04/P05 + §10 cases 16-17: directory↔ordinal conflicts fail closed globally."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    files = (
        "same_a/3_above_peg_l.MOV",
        "same_b/3_left_peg_l.MOV",
        "mixed/4_above_key_r.MOV",
        "mixed/5_left_key_r.MOV",
        "mixed/x_right_key_r.MOV",
        "clean/6_right_nut_l.MOV",
    )
    for source in files:
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, captures, _ = _read_outputs(out)
    rows = _by_source(assets)
    for source in files[:4]:
        _assert_row_case(
            rows[source],
            disposition="quarantined",
            reason="subject_token_conflict",
        )
    _assert_row_case(
        rows[files[4]],
        disposition="quarantined",
        reason="subject_token_nonnumeric",
    )
    _assert_row_case(
        rows[files[5]],
        disposition="canonical",
        reason="ok",
        subject="6",
        view="right",
        task="nut",
        side="l",
        repeat="0",
    )
    assert [row["capture_id"] for row in captures] == ["s06-nut-l"]


def test_leading_zero_and_large_ordinals_follow_integer_semantics(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P07/P13: 03==3 within one directory; 0 and ordinals beyond directory count remain valid."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    files = (
        "subject/03_above_peg_l.MOV",
        "subject/3_left_peg_l.MOV",
        "zero/0_right_cap_r.MOV",
        "large/123_above_nut_l.MOV",
    )
    for source in files:
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, captures, _ = _read_outputs(out)
    rows = _by_source(assets)
    assert rows[files[0]]["capture_id"] == "s03-peg-l"
    assert rows[files[1]]["capture_id"] == "s03-peg-l"
    assert rows[files[2]]["capture_id"] == "s00-cap-r"
    assert rows[files[3]]["capture_id"] == "s123-nut-l"
    assert {row["capture_id"] for row in captures} == {
        "s00-cap-r",
        "s03-peg-l",
        "s123-nut-l",
    }


def test_capture_rows_are_exactly_the_canonical_group_partition(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P06/P07 + §10 case 18: duplicate views stay assets and become one conflicted capture."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    files = (
        "subject/3_above_peg_l.MOV",
        "subject/3_above__peg_l.MOV",
        "subject/3_left_peg_l.MOV",
        "subject/3_right_cap_r.MOV",
        "subject/3_right_cap_.MOV",
    )
    for source in files:
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    assets, captures, _ = _read_outputs(out)
    canonical = [row for row in assets if row["disposition"] == "canonical"]
    capture_by_id = {row["capture_id"]: row for row in captures}
    assert set(capture_by_id) == {row["capture_id"] for row in canonical}
    assert sum(int(row["n_assets"]) for row in captures) == len(canonical) == 4
    for capture in captures:
        members = [row for row in canonical if row["capture_id"] == capture["capture_id"]]
        assert int(capture["n_assets"]) == len(members) >= 1
        assert capture["capture_id"] == (
            f"s{int(capture['subject_ordinal']):02d}-{capture['task']}-{capture['side']}"
        )
    peg = capture_by_id["s03-peg-l"]
    assert peg["n_assets"] == "3"
    assert peg["n_views"] == "2"
    assert peg["view_conflict"] == "1"
    assert _parse_views(peg["views"]) == {"above", "left"}


def test_empty_corpus_still_writes_all_zero_artifacts(tmp_path: pathlib.Path) -> None:
    """A23: empty input publishes total reason and quantile domains with status 0."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    corpus.mkdir()

    assert _invoke(corpus, out) == 0
    assets, captures, census = _read_outputs(out)
    assert assets == []
    assert captures == []
    assert census["assets"] == {
        "canonical": 0,
        "discovered": 0,
        "distinct_sha256": 0,
        "excluded": 0,
        "nominal_duration_s": {"count": 0},
        "nominal_minutes_total": 0.0,
        "quarantined": 0,
        "reported_frames_total": 0,
        "total_bytes": 0,
    }
    assert census["captures"]["total"] == 0
    assert census["captures"]["view_coverage"] == {}
    assert census["captures"]["duration_spread_s"] == {"count": 0}
    assert census["duration_spread_all_captures_s"] == {"count": 0}
    expected_reasons = sorted(
        reason for reasons in REASONS_BY_DISPOSITION.values() for reason in reasons
    )
    assert census["reason_codes"] == dict.fromkeys(expected_reasons, 0)
    assert census["normalization"] == {"applied": {}, "task_repairs": {}}


@pytest.mark.parametrize(
    ("spec", "field", "expected_cell", "expected_duration", "expected_flags"),
    [
        pytest.param(
            _FactSpec(width=0.0),
            "reported_width",
            "0",
            "0.2",
            "dimensions_invalid",
            id="dimensions-zero",
        ),
        pytest.param(
            _FactSpec(fps=0.0),
            "reported_avg_fps",
            "0",
            None,
            "fps_invalid",
            id="fps-zero",
        ),
        pytest.param(
            _FactSpec(frame_count=0.0),
            "reported_frame_count",
            "0",
            None,
            "frame_count_invalid",
            id="frame-count-zero",
        ),
        pytest.param(
            _FactSpec(fps=math.nan),
            "reported_avg_fps",
            "",
            None,
            "fps_invalid",
            id="fps-nonfinite",
        ),
        pytest.param(
            _FactSpec(rotation=45.0),
            "reported_rotation_deg",
            "45",
            "0.2",
            "rotation_unexpected",
            id="rotation-invalid",
        ),
    ],
)
def test_degenerate_container_facts_flag_without_changing_disposition(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: _FactSpec,
    field: str,
    expected_cell: str,
    expected_duration: str | None,
    expected_flags: str,
) -> None:
    """A19/A27: invalid facts stay raw/blank; duration requires positive fps and frames."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")
    _patch_capture(monkeypatch, default=spec)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    row = assets[0]
    _assert_row_case(
        row,
        disposition="canonical",
        reason="ok",
        subject="3",
        view="above",
        task="peg",
        side="l",
        repeat="0",
    )
    if expected_cell:
        assert float(row[field]) == float(expected_cell)
    else:
        assert row[field] == ""
    if expected_duration is None:
        assert row["nominal_duration_s"] == ""
    else:
        assert float(row["nominal_duration_s"]) == float(expected_duration)
    assert row["fact_flags"] == expected_flags


def test_fact_flags_are_closed_deduplicated_and_lexicographically_sorted(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P04/P14/A9: every simultaneous fact defect emits the exact ordered closed vocabulary."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")
    _patch_capture(
        monkeypatch,
        default=_FactSpec(
            width=0.0,
            height=-1.0,
            fps=math.nan,
            frame_count=0.0,
            rotation=45.0,
        ),
    )

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    assert assets[0]["disposition"] == "canonical"
    assert assets[0]["fact_flags"] == (
        "dimensions_invalid|fps_invalid|frame_count_invalid|rotation_unexpected"
    )


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_valid_rotation_metadata_is_preserved_without_a_fact_flag(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    rotation: int,
) -> None:
    """P04/P14 + §10 rotation 90/180/270: preserve metadata; disposition remains canonical."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_left_key_r.MOV", b"synthetic")
    _patch_capture(monkeypatch, default=_FactSpec(rotation=float(rotation)))

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    row = assets[0]
    assert row["disposition"] == "canonical"
    assert row["reported_rotation_deg"] == str(rotation)
    assert row["fact_flags"] == ""


def test_header_probe_sets_orientation_reads_raw_facts_and_never_decodes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P14: set/read orientation_auto; never use safe_fps, frame_count, read, or grab."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_glass_l.MOV", b"synthetic")
    spec = _FactSpec(
        width=320.0,
        height=240.0,
        fps=29.965,
        frame_count=11.0,
        rotation=90.0,
        orientation_auto=1.0,
    )
    factory = _patch_capture(monkeypatch, default=spec)

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    row = assets[0]
    assert row["reported_width"] == "320"
    assert row["reported_height"] == "240"
    assert row["reported_avg_fps"] == "29.965"
    assert row["reported_frame_count"] == "11"
    assert row["reported_rotation_deg"] == "90"
    assert row["reported_fourcc"].lower() == "mjpg"
    assert row["nominal_duration_s"] == str(round(11 / 29.965, 4))
    assert len(factory.captures) == 1
    capture = factory.captures[0]
    assert capture.read_calls == 0
    assert capture.grab_calls == 0
    orientation_sets = [
        value for prop, value in capture.set_calls if prop == cv2.CAP_PROP_ORIENTATION_AUTO
    ]
    assert orientation_sets
    assert all(bool(value) for value in orientation_sets)
    assert census["orientation_auto"] is True
    assert census["backend_name"] == "FAKE"


def test_census_uses_orientation_readback_from_the_first_opened_asset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P11/P14/A12: skip unreadable assets, then publish the first opened readback."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    names = (
        "3_above_cap_l.MOV",
        "3_left_cap_l.MOV",
        "3_right_cap_l.MOV",
    )
    for name in names:
        _put(corpus, f"subject/{name}", b"synthetic")
    _patch_capture(
        monkeypatch,
        {
            names[0]: _FactSpec(opened=False),
            names[1]: _FactSpec(orientation_auto=0.0),
            names[2]: _FactSpec(orientation_auto=0.0),
        },
    )

    assert _invoke(corpus, out) == 0
    _, _, census = _read_outputs(out)
    assert census["orientation_auto"] is False


def test_reruns_are_byte_identical_and_remove_stale_rows(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P09: repeated same-tree runs are byte-identical; regeneration drops stale rows."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    first = _put(corpus, "subject/3_above_peg_l.MOV", mjpg_video_bytes)
    _put(corpus, "subject/3_left_peg_l.MOV", mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    first_bytes = _artifact_bytes(out)
    assert _invoke(corpus, out) == 0
    assert _artifact_bytes(out) == first_bytes

    first.unlink()
    assert _invoke(corpus, out) == 0
    assets, captures, _ = _read_outputs(out)
    assert [row["source_path"] for row in assets] == ["subject/3_left_peg_l.MOV"]
    assert captures[0]["n_assets"] == "1"


def test_artifacts_ignore_filesystem_enumeration_order(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P10: opposite on-disk creation order yields byte-identical sorted artifacts."""
    names = [
        "3_right_nut_r.MOV",
        "3_above_cap_l.MOV",
        "3_left_key_r.MOV",
        "3_above_coin_r.MOV",
        "3_right_glass_l.MOV",
        "3_left_peg_l.MOV",
    ]
    payloads = {name: bytes([index + 1]) for index, name in enumerate(sorted(names))}
    roots = (tmp_path / "tree_a", tmp_path / "tree_b")
    for root, order in zip(roots, (names, list(reversed(names))), strict=True):
        for name in order:
            _put(root, f"subject/{name}", payloads[name])
    raw_a = [entry.name for entry in os.scandir(roots[0] / "subject")]
    raw_b = [entry.name for entry in os.scandir(roots[1] / "subject")]
    assert raw_a != raw_b, "fixture must expose different raw enumeration orders"
    _patch_capture(monkeypatch)

    outputs = (tmp_path / "out_a", tmp_path / "out_b")
    for corpus, out in zip(roots, outputs, strict=True):
        assert _invoke(corpus, out) == 0
    assert _artifact_bytes(outputs[0]) == _artifact_bytes(outputs[1])


def test_census_counts_recompute_from_the_two_csv_artifacts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A19/A23/A27/A29: rebuild the total census while skipping absent numeric facts."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    payloads = {
        "subject/3_above_peg_l.MOV": b"a",
        "subject/3_left_peg_l.MOV": b"bb",
        "subject/3_right_peg_l.MOV": b"ccc",
        "subject/3_right_cap_.MOV": b"dddd",
        "subject/notes.txt": b"eeeee",
    }
    for source, payload in payloads.items():
        _put(corpus, source, payload)
    specs = {
        "3_above_peg_l.MOV": _FactSpec(fps=10.0, frame_count=10.0),
        "3_left_peg_l.MOV": _FactSpec(
            fps=10.0,
            frame_count=11.0,
            fourcc=cv2.VideoWriter.fourcc(*"XVID"),
        ),
        "3_right_peg_l.MOV": _FactSpec(fps=10.0, frame_count=13.0),
        "3_right_cap_.MOV": _FactSpec(fps=10.0, frame_count=9.0),
        "notes.txt": _FactSpec(fps=10.0, frame_count=8.0),
    }
    _patch_capture(monkeypatch, specs)

    assert _invoke(corpus, out) == 0
    assets, captures, census = _read_outputs(out)

    def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        ordered = sorted(values)

        def _rank(fraction: float) -> float:
            return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 4)

        return {
            "count": len(ordered),
            "min": round(ordered[0], 4),
            "p25": _rank(0.25),
            "median": _rank(0.5),
            "p75": _rank(0.75),
            "p95": _rank(0.95),
            "max": round(ordered[-1], 4),
        }

    def _render_json(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, indent=2) + "\n"

    def _census_digest(payload: dict[str, Any]) -> str:
        body = dict(payload)
        generation = dict(payload["generation"])
        generation.pop("census.json", None)
        body["generation"] = generation
        round_tripped = json.loads(_render_json(body))
        return hashlib.sha256(_render_json(round_tripped).encode()).hexdigest()

    disposition_counts = Counter(row["disposition"] for row in assets)
    all_reasons = sorted(
        reason for reasons in REASONS_BY_DISPOSITION.values() for reason in reasons
    )
    reason_counts = Counter(dict.fromkeys(all_reasons, 0))
    reason_counts.update(row["reason_code"] for row in assets)
    extension_counts: Counter[str] = Counter()
    for row in assets:
        suffix = pathlib.PurePosixPath(row["source_path"]).suffix
        if not suffix:
            extension_counts["<none>"] += 1
        elif suffix.lower() in video_io.VIDEO_EXTS:
            extension_counts[suffix] += 1
        else:
            extension_counts["<unsupported>"] += 1
    normalizations = Counter(
        token for row in assets for token in row["normalizations"].split("|") if token
    )
    nominal_durations = [float(cell) for row in assets if (cell := row["nominal_duration_s"])]
    shape_counts: Counter[str] = Counter()
    for row in assets:
        if row["probe_status"] != "opened":
            continue
        numeric_cells = (
            row["reported_width"],
            row["reported_height"],
            row["reported_avg_fps"],
            row["reported_rotation_deg"],
        )
        if not all(numeric_cells):
            continue
        fps = format(round(float(row["reported_avg_fps"]), 3), "g")
        shape_counts[
            f"{int(float(row['reported_width']))}x{int(float(row['reported_height']))}"
            f"@{fps}/{row['reported_fourcc'] or '?'}/rot{int(float(row['reported_rotation_deg']))}"
        ] += 1
    rotation_by_view: dict[str, Counter[str]] = defaultdict(Counter)
    for row in assets:
        if row["disposition"] == "canonical":
            rotation_by_view[row["view"]][row["reported_rotation_deg"]] += 1

    directories = {
        pathlib.PurePosixPath(row["source_path"]).parent.as_posix()
        for row in assets
        if row["disposition"] != "excluded"
    }
    codecs_by_directory: dict[str, set[str]] = defaultdict(set)
    for row in assets:
        if row["probe_status"] != "opened":
            continue
        parent = pathlib.PurePosixPath(row["source_path"]).parent.as_posix()
        codecs_by_directory[parent].add(row["reported_fourcc"])

    multi_view = [row for row in captures if int(row["n_views"]) > 1]
    all_spreads = [float(cell) for row in captures if (cell := row["nominal_duration_spread_s"])]
    multi_spreads = [
        float(cell) for row in multi_view if (cell := row["nominal_duration_spread_s"])
    ]
    expected = {
        "assets": {
            "canonical": disposition_counts["canonical"],
            "discovered": len(assets),
            "distinct_sha256": len(
                {row["content_sha256"] for row in assets if row["content_sha256"]}
            ),
            "excluded": disposition_counts["excluded"],
            "nominal_duration_s": _quantiles(nominal_durations),
            "nominal_minutes_total": round(sum(nominal_durations) / 60.0, 4),
            "quarantined": disposition_counts["quarantined"],
            "reported_frames_total": sum(
                int(cell) for row in assets if (cell := row["reported_frame_count"])
            ),
            "total_bytes": sum(int(row["size_bytes"]) for row in assets),
        },
        "backend_name": "FAKE",
        "captures": {
            "duration_spread_s": _quantiles(multi_spreads),
            "frame_parity_within_20pct": sum(
                (int(row["reported_frame_count_max"]) - int(row["reported_frame_count_min"]))
                / int(row["reported_frame_count_max"])
                <= 0.20
                for row in multi_view
                if row["reported_frame_count_min"]
                and row["reported_frame_count_max"]
                and int(row["reported_frame_count_max"]) > 0
            ),
            "frame_parity_within_5pct": sum(
                (int(row["reported_frame_count_max"]) - int(row["reported_frame_count_min"]))
                / int(row["reported_frame_count_max"])
                <= 0.05
                for row in multi_view
                if row["reported_frame_count_min"]
                and row["reported_frame_count_max"]
                and int(row["reported_frame_count_max"]) > 0
            ),
            "multi_view": len(multi_view),
            "same_fps_3dp": sum(
                round(float(row["reported_fps_min"]), 3) == round(float(row["reported_fps_max"]), 3)
                for row in multi_view
                if row["reported_fps_min"] and row["reported_fps_max"]
            ),
            "same_resolution": sum(row["reported_resolution_agree"] == "1" for row in multi_view),
            "total": len(captures),
            "view_coverage": dict(sorted(Counter(row["n_views"] for row in captures).items())),
            "with_view_conflict": sum(row["view_conflict"] == "1" for row in captures),
        },
        "checksums": True,
        "directories_mixing_codecs": sum(
            len({codec for codec in codecs if codec}) > 1 for codecs in codecs_by_directory.values()
        ),
        "duration_spread_all_captures_s": _quantiles(all_spreads),
        "extension_case": dict(sorted(extension_counts.items())),
        "generation": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("assets.csv", "captures.csv")
        },
        "grammar_version": "v1",
        "normalization": {
            "applied": dict(sorted(normalizations.items())),
            "task_repairs": {},
        },
        "opencv_version": cv2.__version__,
        "orientation_auto": True,
        "reason_codes": dict(sorted(reason_counts.items())),
        "rotation_by_view": {
            view: dict(sorted(counts.items())) for view, counts in sorted(rotation_by_view.items())
        },
        "shapes": dict(sorted(shape_counts.items())),
        "subject_directories": len(directories),
        "tool_version": "v1",
    }
    expected["generation"]["census.json"] = _census_digest(expected)
    assert set(census) == {
        "assets",
        "backend_name",
        "captures",
        "checksums",
        "directories_mixing_codecs",
        "duration_spread_all_captures_s",
        "extension_case",
        "generation",
        "grammar_version",
        "normalization",
        "opencv_version",
        "orientation_auto",
        "reason_codes",
        "rotation_by_view",
        "shapes",
        "subject_directories",
        "tool_version",
    }
    assert census == expected
    assert (out / "census.json").read_bytes() == (
        json.dumps(expected, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def test_validate_generation_detects_every_artifact_tamper(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A29: the consumer boundary verifies both tables and the census digest."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    _, _, census = _read_outputs(out)
    generation = census["generation"]
    assert set(generation) == set(ARTIFACT_NAMES)
    for name in ("assets.csv", "captures.csv"):
        assert generation[name] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert inventory.validate_generation(out) == census
    assert census["checksums"] is True

    pristine = _artifact_bytes(out)
    for name in ARTIFACT_NAMES:
        path = out / name
        path.write_bytes(pristine[name] + b"tampered\n")
        with pytest.raises(inventory.InventoryError):
            inventory.validate_generation(out)
        path.write_bytes(pristine[name])
        assert inventory.validate_generation(out) == census

    captures_path = out / "captures.csv"
    captures_path.unlink()
    with pytest.raises(inventory.InventoryError) as missing_error:
        inventory.validate_generation(out)
    assert "captures.csv" in str(missing_error.value)
    assert "/" not in str(missing_error.value)
    assert "\\" not in str(missing_error.value)
    captures_path.write_bytes(pristine["captures.csv"])

    census_path = out / "census.json"
    census_path.write_bytes(
        pristine["census.json"].replace(b'"tool_version": "v1"', b'"tool_version": "v2"')
    )
    with pytest.raises(inventory.InventoryError):
        inventory.validate_generation(out)
    census_path.write_bytes(pristine["census.json"])

    census_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(inventory.InventoryError) as nonobject_error:
        inventory.validate_generation(out)
    assert "census.json" in str(nonobject_error.value)
    assert "/" not in str(nonobject_error.value)
    assert "\\" not in str(nonobject_error.value)
    census_path.write_bytes(pristine["census.json"])
    assert inventory.validate_generation(out) == census


def test_console_and_census_are_redacted_on_success(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P12: success console/census contain aggregates only, never paths, stems, or subject names."""
    corpus = tmp_path / "corpus_secret_root"
    out = tmp_path / "out"
    subject = "subject_secret_xyz"
    names = (
        f"{subject}/3_above_peg_l.MOV",
        f"{subject}/3_left_key_r.mov.MOV",
        f"{subject}/3_right_cap_.MOV",
    )
    for source in names:
        _put(corpus, source, mjpg_video_bytes)

    assert _invoke(corpus, out) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip()
    census_text = (out / "census.json").read_text(encoding="utf-8")
    forbidden = {
        os.fspath(corpus),
        corpus.name,
        subject,
        *(pathlib.PurePosixPath(name).stem for name in names),
    }
    for surface in (captured.out, captured.err):
        assert "/" not in surface
        assert "\\" not in surface
        assert not any(token in surface for token in forbidden)
    assert not any(token in census_text for token in forbidden)
    with (out / "census.json").open(encoding="utf-8") as handle:
        census = json.load(handle)
    for _, value in _walk_json(census):
        if isinstance(value, str):
            assert "/" not in value
            assert "\\" not in value
            assert not any(token in value for token in forbidden)


def test_console_errors_redact_every_operator_path(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P12/P16: domain errors use stderr ERROR: without echoing a path or directory name."""
    corpus = tmp_path / "missing_secret_corpus"
    out = tmp_path / "secret_out"

    assert _invoke(corpus, out) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ERROR:")
    assert "/" not in captured.err
    assert "\\" not in captured.err
    assert corpus.name not in captured.err
    assert out.name not in captured.err


def test_hostile_filesystem_entries_keep_explicit_exclusion_rows(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P01/P15/A4: escaping, broken, and non-regular entries keep safe relative rows."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    inside_source = "subject/3_above_peg_l.MOV"
    escape_source = "subject/escape_file.MOV"
    broken_source = "subject/broken_file.MOV"
    socket_source = "subject/socket_file.MOV"
    _put(corpus, inside_source, b"inside")
    outside_file = tmp_path / "outside.MOV"
    outside_file.write_bytes(b"outside")
    (corpus / escape_source).symlink_to(outside_file)
    (corpus / broken_source).symlink_to(tmp_path / "absent.MOV")
    unix_socket = socket.socket(socket.AF_UNIX)
    unix_socket.bind(os.fspath(corpus / socket_source))
    unix_socket.close()
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    rows = _by_source(assets)
    assert set(rows) == {inside_source, escape_source, broken_source, socket_source}
    assert rows[escape_source]["reason_code"] == "path_escapes_root"
    assert rows[broken_source]["reason_code"] == "broken_symlink"
    assert rows[socket_source]["reason_code"] == "not_a_regular_file"
    for source_name in (escape_source, broken_source, socket_source):
        assert rows[source_name]["disposition"] == "excluded"
        _assert_blank_identity(rows[source_name])
    for row in assets:
        source = pathlib.PurePosixPath(row["source_path"])
        assert not source.is_absolute()
        assert ".." not in source.parts
    assert census["reason_codes"]["broken_symlink"] == 1
    assert census["reason_codes"]["not_a_regular_file"] == 1
    assert census["reason_codes"]["path_escapes_root"] == 1


def test_output_inside_corpus_is_a_redacted_domain_error(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P08/P16 + §10 hostility: resolved in-corpus output fails before discovery/publication."""
    corpus = tmp_path / "corpus_secret"
    corpus.mkdir()
    out = corpus / "inventory_secret"

    assert _invoke(corpus, out) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ERROR:")
    assert corpus.name not in captured.err
    assert out.name not in captured.err
    assert not out.exists()


def test_unreadable_corpus_is_a_domain_error_not_an_empty_success(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P01/P16 + §10 hostility: unreadable discovery cannot masquerade as an empty corpus."""
    corpus = tmp_path / "unreadable_secret"
    out = tmp_path / "out"
    corpus.mkdir()
    corpus.chmod(0)
    try:
        result = _invoke(corpus, out)
    finally:
        corpus.chmod(0o700)

    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ERROR:")
    assert corpus.name not in captured.err
    assert not out.exists()


def test_control_character_path_is_escaped_and_excluded_without_aborting(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P01/P04/P12/A10: control exclusion outranks extension/probe and the run continues."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    raw_source = "subject/bad\x01.txt"
    escaped_source = "subject/bad\\x01.txt"
    _put(corpus, raw_source, b"synthetic")
    _patch_capture(monkeypatch, default=_FactSpec(opened=False))

    assert _invoke(corpus, out) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "\x01" not in captured.out
    assert raw_source not in captured.out
    assets, _, census = _read_outputs(out)
    assert len(assets) == 1
    row = assets[0]
    assert row["source_path"] == escaped_source
    assert row["asset_id"] == _asset_id(raw_source)
    assert row["disposition"] == "excluded"
    assert row["reason_code"] == "control_character_in_path"
    _assert_blank_identity(row)
    assert census["reason_codes"]["control_character_in_path"] == 1


def test_strict_status_depends_only_on_noncanonical_rows(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P16: non-strict complete=0, strict dirty=1, strict canonical-only=0; bytes unchanged."""
    corpus = tmp_path / "corpus"
    _put(corpus, "subject/3_above_peg_l.MOV", b"canonical")
    _put(corpus, "subject/3_above_spoon_l.MOV", b"quarantine")
    _patch_capture(monkeypatch)

    loose_out = tmp_path / "loose"
    strict_out = tmp_path / "strict"
    assert _invoke(corpus, loose_out) == 0
    assert _invoke(corpus, strict_out, strict=True) == 1
    assert _artifact_bytes(loose_out) == _artifact_bytes(strict_out)

    (corpus / "subject" / "3_above_spoon_l.MOV").unlink()
    clean_out = tmp_path / "clean"
    assert _invoke(corpus, clean_out, strict=True) == 0


def test_io_errors_return_two_and_preserve_the_error_channel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P16: an output I/O failure returns 2 and writes only a redacted ERROR: to stderr."""
    corpus = tmp_path / "corpus"
    _put(corpus, "subject/3_above_peg_l.MOV", b"canonical")
    out_file = tmp_path / "occupied_secret"
    out_file.write_bytes(b"not a directory")
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out_file) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ERROR:")
    assert out_file.name not in captured.err


def test_usage_errors_return_two_with_argparse_diagnostic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A18: argparse owns syntax errors; status and diagnostic stay path-free."""
    try:
        result = inventory.main(["--definitely-not-an-inventory-option"])
    except SystemExit as exc:
        result = exc.code
    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("usage:")
    assert ": error:" in captured.err
    assert "/" not in captured.err
    assert "\\" not in captured.err


def test_invariant_failure_preserves_all_previous_artifact_bytes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P03/P17: force an asset-ID collision; assertions run before any final replacement."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"one")
    _put(corpus, "subject/3_left_peg_l.MOV", b"two")
    out.mkdir()
    sentinels = {
        "assets.csv": b"previous-assets\n",
        "captures.csv": b"previous-captures\n",
        "census.json": b"previous-census\n",
    }
    for name, payload in sentinels.items():
        (out / name).write_bytes(payload)
    _patch_capture(monkeypatch)

    class _CollisionDigest:
        def hexdigest(self) -> str:
            return "0" * 16

        def digest(self) -> bytes:
            return b"\0" * 8

    def _collision_blake2b(*_args: Any, **_kwargs: Any) -> _CollisionDigest:
        return _CollisionDigest()

    monkeypatch.setattr(hashlib, "blake2b", _collision_blake2b)
    monkeypatch.setattr(inventory, "blake2b", _collision_blake2b, raising=False)

    try:
        result = _invoke(corpus, out)
    except AssertionError:
        result = -1
    assert result != 0
    assert _artifact_bytes(out) == sentinels
    assert {path.name for path in out.iterdir()} == set(ARTIFACT_NAMES)


def test_adversarial_filename_surfaces_follow_the_closed_grammar(
    tmp_path: pathlib.Path,
    mjpg_video_bytes: bytes,
) -> None:
    """P01/P03/P13: Unicode, media-token, mixed-case suffix, trailing dot, and case-only paths."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    expected = {
        "subject/3_above_peg_l.MoV": ("canonical", "ok"),
        "subject/3_ABOVE_PEG_L.mov": ("canonical", "ok"),
        "subject/3_above_mov_l.MOV": ("quarantined", "task_unknown"),
        "subject/3_above_peg_l..MOV": ("quarantined", "side_unknown"),
        "subject/3_above_peg_l_λ.MOV": ("quarantined", "token_count"),
        "subject/3_above_peg_l.bak.MOV": ("quarantined", "side_unknown"),
    }
    for source in expected:
        _put(corpus, source, mjpg_video_bytes)
    (corpus / "empty_subject").mkdir()

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    rows = _by_source(assets)
    assert [row["source_path"] for row in assets] == sorted(expected)
    assert len({row["asset_id"] for row in assets}) == len(expected)
    for source, (disposition, reason) in expected.items():
        assert rows[source]["disposition"] == disposition
        assert rows[source]["reason_code"] == reason


def test_default_output_is_dedicated_and_never_enters_a_rerun(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P08/P09: default inventory/ is a sibling, and rerunning never inventories its artifacts."""
    corpus = tmp_path / "corpus"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")
    _patch_capture(monkeypatch)
    monkeypatch.chdir(tmp_path)

    assert _invoke(corpus) == 0
    out = tmp_path / "inventory"
    first = _artifact_bytes(out)
    assert _invoke(corpus) == 0
    assets, _, _ = _read_outputs(out)
    assert [row["source_path"] for row in assets] == ["subject/3_above_peg_l.MOV"]
    assert _artifact_bytes(out) == first


def test_discover_paths_prunes_an_in_corpus_output_directory(tmp_path: pathlib.Path) -> None:
    """A7: library discovery prunes `out_dir` without imposing a sort contract."""
    corpus = tmp_path / "corpus"
    out = corpus / "generated"
    source = "subject/3_above_peg_l.MOV"
    generated = "generated/assets.csv"
    _put(corpus, source, b"asset")
    _put(corpus, generated, b"generated")

    assert Counter(inventory.discover_paths(corpus, out)) == Counter({source: 1})
    assert Counter(inventory.discover_paths(corpus, None)) == Counter({source: 1, generated: 1})


def test_symlinks_are_excluded_without_duplicate_fixity_or_membership(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A26: internal, escaping, and broken symlinks remain provenance-only rows."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    target_source = "subject/3_above_peg_l.MOV"
    internal_source = "subject/3_left_peg_l.MOV"
    escape_source = "subject/3_right_peg_l.MOV"
    broken_source = "subject/3_above_key_r.MOV"
    target = _put(corpus, target_source, b"target")
    (corpus / internal_source).symlink_to(target.name)
    outside = tmp_path / "outside.MOV"
    outside.write_bytes(b"outside")
    (corpus / escape_source).symlink_to(outside)
    (corpus / broken_source).symlink_to("absent.MOV")
    factory = _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    assets, captures, census = _read_outputs(out)
    rows = _by_source(assets)
    assert Counter(path.name for path in factory.sources) == Counter({target.name: 1})
    expected_reasons = {
        internal_source: "symlink_within_corpus",
        escape_source: "path_escapes_root",
        broken_source: "broken_symlink",
    }
    for source, reason in expected_reasons.items():
        _assert_row_case(rows[source], disposition="excluded", reason=reason)
        assert rows[source]["probe_status"] == "skipped"
        assert rows[source]["normalizations"] == ""
        assert rows[source]["content_sha256"] == ""
        assert rows[source]["size_bytes"] == ""
    target_hash = hashlib.sha256(b"target").hexdigest()
    assert Counter(row["content_sha256"] for row in assets if row["content_sha256"]) == Counter(
        {target_hash: 1}
    )
    assert len(captures) == 1
    assert captures[0]["capture_id"] == "s03-peg-l"
    assert captures[0]["n_assets"] == "1"
    assert rows[target_source]["size_bytes"] == str(len(b"target"))
    assert census["assets"]["total_bytes"] == len(b"target")
    assert census["reason_codes"]["symlink_within_corpus"] == 1
    assert census["reason_codes"]["path_escapes_root"] == 1
    assert census["reason_codes"]["broken_symlink"] == 1


def test_backend_name_is_the_sorted_observed_opened_set(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A27: census provenance joins distinct opened backends and blanks an empty set."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    names = (
        "3_above_cap_l.MOV",
        "3_left_coin_l.MOV",
        "3_right_key_l.MOV",
    )
    for name in names:
        _put(corpus, f"subject/{name}", b"synthetic")
    _patch_capture(
        monkeypatch,
        {
            names[0]: _FactSpec(backend="ZETA"),
            names[1]: _FactSpec(opened=False, backend="IGNORED"),
            names[2]: _FactSpec(backend="ALPHA"),
        },
    )

    assert _invoke(corpus, out) == 0
    _, _, census = _read_outputs(out)
    assert census["backend_name"] == "ALPHA|ZETA"

    empty = tmp_path / "empty"
    empty_out = tmp_path / "empty_out"
    empty.mkdir()
    assert _invoke(empty, empty_out) == 0
    _, _, empty_census = _read_outputs(empty_out)
    assert empty_census["backend_name"] == ""


def test_excluded_parses_cannot_change_canonical_identity_or_conflicts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A26: post-parse exclusions never enter subject or view/repeat conflict scans."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    canonical_source = "subject/3_above_peg_l.MOV"
    ordinal_conflict = "subject/4_left_key_r.MOV"
    view_repeat_conflict = "subject/3_above_peg_l (2).MOV"
    for source in (canonical_source, ordinal_conflict, view_repeat_conflict):
        _put(corpus, source, b"synthetic")
    _patch_capture(
        monkeypatch,
        {
            pathlib.PurePosixPath(ordinal_conflict).name: _FactSpec(opened=False),
            pathlib.PurePosixPath(view_repeat_conflict).name: _FactSpec(opened=False),
        },
    )

    assert _invoke(corpus, out) == 0
    assets, captures, _ = _read_outputs(out)
    rows = _by_source(assets)
    _assert_row_case(
        rows[canonical_source],
        disposition="canonical",
        reason="ok",
        subject="3",
        view="above",
        task="peg",
        side="l",
        repeat="0",
    )
    for source in (ordinal_conflict, view_repeat_conflict):
        _assert_row_case(rows[source], disposition="excluded", reason="probe_unreadable")
    assert rows[view_repeat_conflict]["normalizations"] == ("repeat_marker|whitespace_collapsed")
    assert len(captures) == 1
    assert captures[0]["capture_id"] == "s03-peg-l"
    assert captures[0]["n_assets"] == "1"
    assert captures[0]["view_conflict"] == "0"


def test_path_rendering_is_injective_and_non_utf8_names_are_excluded(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A28/A29: escaped controls, literal escapes, and surrogate bytes stay distinct."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    control_source = "subject/3_above_peg_l\n.MOV"
    literal_source = r"subject/3_above_peg_l\x0a.MOV"
    c1_source = "subject/3_above_peg_l_\x80.MOV"
    raw_c1_source = os.fsdecode(b"subject/3_above_peg_l_\x80.MOV")
    invalid_source = os.fsdecode(b"subject/3_above_peg_l_\xff.MOV")
    for source in (
        control_source,
        literal_source,
        c1_source,
        raw_c1_source,
        invalid_source,
    ):
        _put(corpus, source, b"synthetic")
    factory = _patch_capture(monkeypatch, default=_FactSpec(opened=False))

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    rows_by_id = {row["asset_id"]: row for row in assets}
    control = rows_by_id[_asset_id(control_source)]
    literal = rows_by_id[_asset_id(literal_source)]
    c1 = rows_by_id[_asset_id(c1_source)]
    raw_c1 = rows_by_id[_asset_id(raw_c1_source)]
    invalid = rows_by_id[_asset_id(invalid_source)]
    assert len({row["source_path"] for row in assets}) == 5
    assert control["source_path"] != literal["source_path"]
    assert c1["source_path"] != raw_c1["source_path"]
    assert "\n" not in control["source_path"]
    assert c1["source_path"].endswith(r"\xc2\x80.MOV")
    assert raw_c1["source_path"].endswith(r"\x80.MOV")
    assert invalid["source_path"].endswith(r"\xff.MOV")
    _assert_row_case(
        control,
        disposition="excluded",
        reason="control_character_in_path",
    )
    _assert_row_case(
        c1,
        disposition="excluded",
        reason="control_character_in_path",
    )
    _assert_row_case(raw_c1, disposition="excluded", reason="path_not_utf8")
    _assert_row_case(invalid, disposition="excluded", reason="path_not_utf8")
    for row in (control, c1, raw_c1, invalid):
        assert row["normalizations"] == ""
        assert row["probe_status"] == "skipped"
    assert Counter(path.name for path in factory.sources) == Counter(
        {pathlib.PurePosixPath(literal_source).name: 1}
    )
    assert census["reason_codes"]["control_character_in_path"] == 2
    assert census["reason_codes"]["path_not_utf8"] == 2


def test_read_error_is_post_parse_and_keeps_its_normalization_trace(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A29: a fixity read failure owns a row and cannot masquerade as no checksums."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    source = "subject/3_ABOVE_grass_L.MOV"
    asset = _put(corpus, source, b"synthetic")
    _patch_capture(monkeypatch)
    asset.chmod(0)
    try:
        assert _invoke(corpus, out) == 0
    finally:
        asset.chmod(0o600)

    assets, _, census = _read_outputs(out)
    row = assets[0]
    _assert_row_case(row, disposition="excluded", reason="read_error")
    assert row["normalizations"] == "case_folded|task_repaired"
    assert row["content_sha256"] == ""
    assert census["reason_codes"]["read_error"] == 1

    unchecked_out = tmp_path / "unchecked_out"
    assert _invoke(corpus, unchecked_out, checksums=False) == 0
    unchecked_assets, _, unchecked_census = _read_outputs(unchecked_out)
    _assert_row_case(
        unchecked_assets[0],
        disposition="canonical",
        reason="ok",
        subject="3",
        view="above",
        task="glass",
        side="l",
        repeat="0",
    )
    assert unchecked_assets[0]["content_sha256"] == ""
    assert unchecked_census["checksums"] is False
    assert unchecked_census["assets"]["distinct_sha256"] == 0


def test_probe_exceptions_become_open_failed_rows(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A28: an exceptional backend fails one probe instead of aborting discovery."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")

    def _raise_capture(_source: str | os.PathLike[str]) -> None:
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(cv2, "VideoCapture", _raise_capture)
    monkeypatch.setattr(inventory, "VideoCapture", _raise_capture, raising=False)
    monkeypatch.setattr(video_io, "VideoCapture", _raise_capture, raising=False)

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    _assert_row_case(assets[0], disposition="excluded", reason="probe_unreadable")
    assert assets[0]["probe_status"] == "open_failed"
    assert census["reason_codes"]["probe_unreadable"] == 1


def test_subject_ordinal_requires_ascii_digits(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A28: Unicode digit classes never alias or crash the integer identity grammar."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    sources = (
        "subject/٣_above_peg_l.MOV",
        "subject/³_left_peg_l.MOV",
        "subject/3_right_peg_l.MOV",
    )
    for source in sources:
        _put(corpus, source, b"synthetic")
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    assets, captures, _ = _read_outputs(out)
    rows = _by_source(assets)
    for source in sources[:2]:
        _assert_row_case(
            rows[source],
            disposition="quarantined",
            reason="subject_token_nonnumeric",
        )
    _assert_row_case(
        rows[sources[2]],
        disposition="canonical",
        reason="ok",
        subject="3",
        view="right",
        task="peg",
        side="l",
        repeat="0",
    )
    assert [row["capture_id"] for row in captures] == ["s03-peg-l"]


def test_mixed_orientation_readbacks_abort_before_publication(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A29: one census cannot mix coded and display-dimension header facts."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    names = ("3_above_peg_l.MOV", "3_left_key_r.MOV")
    for name in names:
        _put(corpus, f"subject/{name}", b"synthetic")
    _patch_capture(
        monkeypatch,
        {
            names[0]: _FactSpec(orientation_auto=1.0),
            names[1]: _FactSpec(orientation_auto=0.0),
        },
    )

    try:
        result = _invoke(corpus, out)
    except inventory.InventoryError:
        result = -1
    assert result != 0
    assert not out.exists() or not set(ARTIFACT_NAMES) & {path.name for path in out.iterdir()}


def test_subprocess_silences_native_backend_stderr(
    tmp_path: pathlib.Path,
) -> None:
    """A18/A29: raw subprocess descriptors stay path-free when FFmpeg rejects a file."""
    corpus = tmp_path / "corpus_native_secret"
    out = tmp_path / "out_native_secret"
    _put(corpus, "subject_secret/3_above_peg_l.MOV", b"not a container")
    environment = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "OPENCV_FFMPEG_LOGLEVEL", "PYTHONPATH"):
        environment.pop(name, None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pose_estimation.inventory",
            "--corpus",
            os.fspath(corpus),
            "--out",
            os.fspath(out),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert os.fspath(corpus) not in completed.stdout
    assert os.fspath(out) not in completed.stdout
    assert corpus.name not in completed.stdout
    assert out.name not in completed.stdout


def test_publication_uses_pid_staging_and_replaces_census_last(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A15/A18/A21/A29: publication order, staging names, and console stay bounded."""
    corpus = tmp_path / "corpus_publish_secret"
    out = tmp_path / "out_publish_secret"
    _put(corpus, "subject_secret/3_above_peg_l.MOV", b"synthetic")
    _patch_capture(monkeypatch)
    original_replace = pathlib.Path.replace
    replacements: list[tuple[str, str]] = []

    def _track_replace(
        source: pathlib.Path,
        target: str | os.PathLike[str],
    ) -> pathlib.Path:
        target_path = pathlib.Path(target)
        if target_path.parent == out:
            replacements.append((source.name, target_path.name))
        return original_replace(source, target)

    monkeypatch.setattr(pathlib.Path, "replace", _track_replace)

    assert _invoke(corpus, out) == 0
    captured = capsys.readouterr()
    expected_names = [
        (f"{name}.{os.getpid()}.tmp", name)
        for name in ("assets.csv", "captures.csv", "census.json")
    ]
    assert replacements == expected_names
    assert captured.err == ""
    assert "/" not in captured.out
    assert "\\" not in captured.out
    assert corpus.name not in captured.out
    assert out.name not in captured.out
    census = inventory.validate_generation(out)
    assert set(census["generation"]) == set(ARTIFACT_NAMES)


def test_publication_oserror_is_redacted_and_cleans_staging(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A18/A21: an OSError returns 2 without a traceback, path, or staging residue."""
    corpus = tmp_path / "corpus_failure_secret"
    out = tmp_path / "out_failure_secret"
    _put(corpus, "subject_secret/3_above_peg_l.MOV", b"synthetic")
    _patch_capture(monkeypatch)

    def _fail_replace(
        source: pathlib.Path,
        target: str | os.PathLike[str],
    ) -> pathlib.Path:
        raise OSError(f"cannot replace {source} with {target}")

    monkeypatch.setattr(pathlib.Path, "replace", _fail_replace)

    assert _invoke(corpus, out) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ERROR:")
    assert "Traceback" not in captured.err
    assert "/" not in captured.err
    assert "\\" not in captured.err
    assert corpus.name not in captured.err
    assert out.name not in captured.err
    assert not out.exists() or not any(path.name.endswith(".tmp") for path in out.iterdir())


def test_tiny_corpus_artifacts_match_exact_contract_bytes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7/A8/A23/A29: freeze all three artifact byte streams from the contract."""
    source = "subject/3_above_peg_l.MOV"
    payload = b"x"
    asset_id = _asset_id(source)
    content_sha256 = hashlib.sha256(payload).hexdigest()
    expected_assets = (
        "asset_id,capture_id,disposition,reason_code,source_path,subject_ordinal,"
        "view,task,side,repeat,normalizations,size_bytes,content_sha256,reported_width,"
        "reported_height,reported_avg_fps,reported_frame_count,reported_rotation_deg,"
        "reported_fourcc,nominal_duration_s,fact_flags,probe_status,grammar_version,"
        "tool_version\n"
        f"{asset_id},s03-peg-l,canonical,ok,{source},3,above,peg,l,0,,1,"
        f"{content_sha256},64,48,15.0,3,0,MJPG,0.2,,opened,v1,v1\n"
    )
    expected_captures = (
        "capture_id,subject_ordinal,task,side,n_assets,views,n_views,view_conflict,"
        "reported_frame_count_min,reported_frame_count_max,reported_fps_min,"
        "reported_fps_max,reported_fps_spread_hz,nominal_duration_min_s,"
        "nominal_duration_max_s,nominal_duration_spread_s,reported_resolution_agree,"
        "reported_rotation_agree,grammar_version,tool_version\n"
        "s03-peg-l,3,peg,l,1,above,1,0,3,3,15.0,15.0,0.0,0.2,0.2,0.0,1,1,v1,v1\n"
    )
    assets_digest = hashlib.sha256(expected_assets.encode()).hexdigest()
    captures_digest = hashlib.sha256(expected_captures.encode()).hexdigest()
    census_prefix = (
        "{\n"
        '  "assets": {\n'
        '    "canonical": 1,\n'
        '    "discovered": 1,\n'
        '    "distinct_sha256": 1,\n'
        '    "excluded": 0,\n'
        '    "nominal_duration_s": {\n'
        '      "count": 1,\n'
        '      "max": 0.2,\n'
        '      "median": 0.2,\n'
        '      "min": 0.2,\n'
        '      "p25": 0.2,\n'
        '      "p75": 0.2,\n'
        '      "p95": 0.2\n'
        "    },\n"
        '    "nominal_minutes_total": 0.0033,\n'
        '    "quarantined": 0,\n'
        '    "reported_frames_total": 3,\n'
        '    "total_bytes": 1\n'
        "  },\n"
        '  "backend_name": "FAKE",\n'
        '  "captures": {\n'
        '    "duration_spread_s": {\n'
        '      "count": 0\n'
        "    },\n"
        '    "frame_parity_within_20pct": 0,\n'
        '    "frame_parity_within_5pct": 0,\n'
        '    "multi_view": 0,\n'
        '    "same_fps_3dp": 0,\n'
        '    "same_resolution": 0,\n'
        '    "total": 1,\n'
        '    "view_coverage": {\n'
        '      "1": 1\n'
        "    },\n"
        '    "with_view_conflict": 0\n'
        "  },\n"
        '  "checksums": true,\n'
        '  "directories_mixing_codecs": 0,\n'
        '  "duration_spread_all_captures_s": {\n'
        '    "count": 1,\n'
        '    "max": 0.0,\n'
        '    "median": 0.0,\n'
        '    "min": 0.0,\n'
        '    "p25": 0.0,\n'
        '    "p75": 0.0,\n'
        '    "p95": 0.0\n'
        "  },\n"
        '  "extension_case": {\n'
        '    ".MOV": 1\n'
        "  },\n"
    )
    census_suffix = (
        '  "grammar_version": "v1",\n'
        '  "normalization": {\n'
        '    "applied": {},\n'
        '    "task_repairs": {}\n'
        "  },\n"
        f'  "opencv_version": "{cv2.__version__}",\n'
        '  "orientation_auto": true,\n'
        '  "reason_codes": {\n'
        '    "broken_symlink": 0,\n'
        '    "control_character_in_path": 0,\n'
        '    "not_a_regular_file": 0,\n'
        '    "ok": 1,\n'
        '    "path_escapes_root": 0,\n'
        '    "path_not_utf8": 0,\n'
        '    "probe_unreadable": 0,\n'
        '    "read_error": 0,\n'
        '    "repeat_marker_unrecognized": 0,\n'
        '    "side_missing": 0,\n'
        '    "side_unknown": 0,\n'
        '    "subject_token_conflict": 0,\n'
        '    "subject_token_nonnumeric": 0,\n'
        '    "symlink_within_corpus": 0,\n'
        '    "task_unknown": 0,\n'
        '    "token_count": 0,\n'
        '    "unsupported_extension": 0,\n'
        '    "view_unknown": 0\n'
        "  },\n"
        '  "rotation_by_view": {\n'
        '    "above": {\n'
        '      "0": 1\n'
        "    }\n"
        "  },\n"
        '  "shapes": {\n'
        '    "64x48@15/MJPG/rot0": 1\n'
        "  },\n"
        '  "subject_directories": 1,\n'
        '  "tool_version": "v1"\n'
        "}\n"
    )
    census_remaining = (
        census_prefix
        + '  "generation": {\n'
        + f'    "assets.csv": "{assets_digest}",\n'
        + f'    "captures.csv": "{captures_digest}"\n'
        + "  },\n"
        + census_suffix
    )
    census_digest = hashlib.sha256(census_remaining.encode()).hexdigest()
    expected_census = (
        census_prefix
        + '  "generation": {\n'
        + f'    "assets.csv": "{assets_digest}",\n'
        + f'    "captures.csv": "{captures_digest}",\n'
        + f'    "census.json": "{census_digest}"\n'
        + "  },\n"
        + census_suffix
    )
    expected = {
        "assets.csv": expected_assets.encode(),
        "captures.csv": expected_captures.encode(),
        "census.json": expected_census.encode(),
    }

    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, source, payload)
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    assert _artifact_bytes(out) == expected


def test_census_digest_round_trip_uses_string_key_order() -> None:
    """A30: digest canonicalization round-trips discriminating integer-key maps."""
    assets_digest = "a" * 64
    captures_digest = "b" * 64
    census = {
        "generation": {
            "assets.csv": assets_digest,
            "captures.csv": captures_digest,
            "census.json": "c" * 64,
        },
        "rotation_by_view": {"above": {0: 1, 90: 2, 180: 3, 270: 4}},
        "captures": {"view_coverage": {2: 2, 10: 1}},
    }
    expected_text = (
        "{\n"
        '  "captures": {\n'
        '    "view_coverage": {\n'
        '      "10": 1,\n'
        '      "2": 2\n'
        "    }\n"
        "  },\n"
        '  "generation": {\n'
        f'    "assets.csv": "{assets_digest}",\n'
        f'    "captures.csv": "{captures_digest}"\n'
        "  },\n"
        '  "rotation_by_view": {\n'
        '    "above": {\n'
        '      "0": 1,\n'
        '      "180": 3,\n'
        '      "270": 4,\n'
        '      "90": 2\n'
        "    }\n"
        "  }\n"
        "}\n"
    )
    expected_digest = hashlib.sha256(expected_text.encode()).hexdigest()

    assert inventory.census_digest(census) == expected_digest
    assert census["generation"]["census.json"] == "c" * 64


def test_exclusion_first_match_precedence_is_pinned(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A29: overlapping exclusion conditions resolve in the frozen first-match order."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    control_and_non_utf8 = os.fsdecode(b"subject/bad\x01\xff.MOV")
    non_utf8_and_broken = os.fsdecode(b"subject/broken_\xff.MOV")
    nonregular_and_unsupported = "subject/not_regular.txt"
    read_and_probe_error = "subject/3_ABOVE_grass_L.MOV"
    _put(corpus, control_and_non_utf8, b"control")
    (corpus / non_utf8_and_broken).symlink_to("absent.MOV")
    unix_socket = socket.socket(socket.AF_UNIX)
    unix_socket.bind(os.fspath(corpus / nonregular_and_unsupported))
    unix_socket.close()
    unreadable = _put(corpus, read_and_probe_error, b"unreadable")
    _patch_capture(monkeypatch, default=_FactSpec(opened=False))
    unreadable.chmod(0)
    try:
        assert _invoke(corpus, out) == 0
    finally:
        unreadable.chmod(0o600)

    assets, _, census = _read_outputs(out)
    rows_by_id = {row["asset_id"]: row for row in assets}
    _assert_row_case(
        rows_by_id[_asset_id(control_and_non_utf8)],
        disposition="excluded",
        reason="control_character_in_path",
    )
    _assert_row_case(
        rows_by_id[_asset_id(non_utf8_and_broken)],
        disposition="excluded",
        reason="path_not_utf8",
    )
    rows = _by_source(assets)
    _assert_row_case(
        rows[nonregular_and_unsupported],
        disposition="excluded",
        reason="not_a_regular_file",
    )
    _assert_row_case(
        rows[read_and_probe_error],
        disposition="excluded",
        reason="read_error",
    )
    assert rows[read_and_probe_error]["normalizations"] == "case_folded|task_repaired"
    assert census["reason_codes"]["control_character_in_path"] == 1
    assert census["reason_codes"]["path_not_utf8"] == 1
    assert census["reason_codes"]["not_a_regular_file"] == 1
    assert census["reason_codes"]["read_error"] == 1


def test_shape_histogram_counts_rendered_nan_keys(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A31: equal rendered NaN shapes aggregate before dictionary publication."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    names = ("3_above_peg_l.MOV", "3_left_key_r.MOV")
    for name in names:
        _put(corpus, f"subject/{name}", b"synthetic")
    _patch_capture(
        monkeypatch,
        {name: _FactSpec(fps=math.nan) for name in names},
    )

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    assert census["shapes"] == {"64x48@nan/MJPG/rot0": 2}
    assert all(row["reported_avg_fps"] == "" for row in assets)
    assert all(row["fact_flags"] == "fps_invalid" for row in assets)


def test_extension_case_redacts_unrecognized_suffixes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A31: only recognized extension case enters the redaction-safe census."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    secret_suffix = "subject_secret_xyz"
    _put(corpus, "subject/3_above_peg_l.MOV", b"video")
    _put(corpus, f"subject/notes.{secret_suffix}", b"unsupported")
    _put(corpus, "subject/notes_without_suffix", b"unsupported")
    _patch_capture(monkeypatch)

    assert _invoke(corpus, out) == 0
    _, _, census = _read_outputs(out)
    assert census["extension_case"] == {
        ".MOV": 1,
        "<none>": 1,
        "<unsupported>": 1,
    }
    census_text = (out / "census.json").read_text(encoding="utf-8")
    assert secret_suffix not in census_text


def test_reported_fourcc_preserves_a_trailing_space(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A31: a printable four-byte codec tag stays verbatim, including spaces."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")
    _patch_capture(
        monkeypatch,
        default=_FactSpec(fourcc=cv2.VideoWriter.fourcc(*"DIB ")),
    )

    assert _invoke(corpus, out) == 0
    assets, _, census = _read_outputs(out)
    assert assets[0]["reported_fourcc"] == "DIB "
    assert census["shapes"] == {"64x48@15/DIB /rot0": 1}


@pytest.mark.parametrize("stage", ["is_opened", "get"])
def test_probe_acquisition_exceptions_become_open_failed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """A28/A31: constructor-adjacent and property failures remain per-file outcomes."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")

    class _AcquisitionFails(_HeaderCapture):
        def isOpened(self) -> bool:
            if stage == "is_opened":
                raise RuntimeError("synthetic isOpened failure")
            return super().isOpened()

        def get(self, prop: int) -> float:
            if stage == "get":
                raise RuntimeError("synthetic property failure")
            return super().get(prop)

    def _factory(_source: str | os.PathLike[str]) -> _AcquisitionFails:
        return _AcquisitionFails(_FactSpec())

    monkeypatch.setattr(cv2, "VideoCapture", _factory)
    monkeypatch.setattr(inventory, "VideoCapture", _factory, raising=False)
    monkeypatch.setattr(video_io, "VideoCapture", _factory, raising=False)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    _assert_row_case(assets[0], disposition="excluded", reason="probe_unreadable")
    assert assets[0]["probe_status"] == "open_failed"


def test_probe_release_exception_keeps_opened_facts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A31: teardown failure does not rewrite a completed acquisition as open_failed."""
    corpus = tmp_path / "corpus"
    out = tmp_path / "out"
    _put(corpus, "subject/3_above_peg_l.MOV", b"synthetic")

    class _ReleaseFails(_HeaderCapture):
        def release(self) -> None:
            self.released = True
            raise RuntimeError("synthetic release failure")

    def _factory(_source: str | os.PathLike[str]) -> _ReleaseFails:
        return _ReleaseFails(_FactSpec())

    monkeypatch.setattr(cv2, "VideoCapture", _factory)
    monkeypatch.setattr(inventory, "VideoCapture", _factory, raising=False)
    monkeypatch.setattr(video_io, "VideoCapture", _factory, raising=False)

    assert _invoke(corpus, out) == 0
    assets, _, _ = _read_outputs(out)
    _assert_row_case(
        assets[0],
        disposition="canonical",
        reason="ok",
        subject="3",
        view="above",
        task="peg",
        side="l",
        repeat="0",
    )
    assert assets[0]["probe_status"] == "opened"


def test_utf8_filename_classification_is_locale_independent(
    tmp_path: pathlib.Path,
) -> None:
    """A31: a C-locale surrogateescape never reclassifies valid UTF-8 path bytes."""
    corpus = tmp_path / "corpus_locale"
    out = tmp_path / "out_locale"
    source = "subject/3_above_peg_l_é.MOV"
    _put(corpus, source, b"not a container")
    environment = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "OPENCV_FFMPEG_LOGLEVEL", "PYTHONPATH"):
        environment.pop(name, None)
    environment.update(
        {
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUTF8": "0",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8=0",
            "-m",
            "pose_estimation.inventory",
            "--corpus",
            os.fspath(corpus),
            "--out",
            os.fspath(out),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assets, _, census = _read_outputs(out)
    assert assets[0]["source_path"] == source
    assert assets[0]["reason_code"] == "probe_unreadable"
    assert census["reason_codes"]["path_not_utf8"] == 0


def test_duplicate_capture_identifiers_are_rejected_directly() -> None:
    """A29: the invariant rejects two capture rows carrying one identifier."""
    captures = [
        inventory.CaptureRecord("s03-peg-l", 3, "peg", "l", ()),
        inventory.CaptureRecord("s03-peg-l", 3, "peg", "l", ()),
    ]

    with pytest.raises(inventory.InventoryError, match=r"share one identifier"):
        inventory.check_invariants([], captures, [])
