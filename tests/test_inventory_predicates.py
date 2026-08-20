"""Predicate-level regressions for the M2.1 inventory contract."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from pose_estimation import inventory, video_io


def _run_ascii_locale(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        environment.pop(name, None)
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "0",
            "PYTHONPATH": str(pathlib.Path(inventory.__file__).parents[1]),
        }
    )
    return subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", script, *args],
        check=True,
        capture_output=True,
        encoding="ascii",
        env=environment,
        timeout=30,
    )


def test_printable_path_doubles_every_escape_introducer() -> None:
    """M005: every literal backslash remains distinguishable from an escape."""
    assert inventory._printable_path(r"entry\one\two.MOV") == r"entry\\one\\two.MOV"


def test_build_assets_parses_canonical_relative_text(tmp_path: pathlib.Path) -> None:
    """M007: parsing sees decoded UTF-8 whitespace under an ASCII filesystem locale."""
    corpus = os.fsencode(tmp_path / "corpus")
    os.mkdir(corpus)  # noqa: PTH102 - byte paths are the subject under test
    descriptor = os.open(
        corpus + b"/3\xc2\xa0above\xc2\xa0peg\xc2\xa0l.MOV",
        os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    os.close(descriptor)
    script = """
import pathlib
import sys
from pose_estimation import inventory
root = pathlib.Path(sys.argv[1])
asset = inventory.build_assets(root, root.parent / 'out', checksums=False)[0]
print(sys.getfilesystemencoding(), asset.parse.reason_code, '|'.join(asset.parse.applied))
"""

    completed = _run_ascii_locale(script, os.fsdecode(corpus))

    assert completed.stdout.strip() == "ascii ok whitespace_collapsed"


def test_discover_paths_returns_canonical_relative_text(tmp_path: pathlib.Path) -> None:
    """M009: discovery returns strict UTF-8 text under an ASCII filesystem locale."""
    corpus = os.fsencode(tmp_path / "corpus")
    os.mkdir(corpus)  # noqa: PTH102 - byte paths are the subject under test
    descriptor = os.open(corpus + b"/entry_\xc3\xa9.MOV", os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    script = """
import pathlib
import sys
from pose_estimation import inventory
root = pathlib.Path(sys.argv[1])
relative = inventory.discover_paths(root)[0]
print(sys.getfilesystemencoding(), relative.encode('utf-8').hex())
"""

    completed = _run_ascii_locale(script, os.fsdecode(corpus))

    assert completed.stdout.strip() == "ascii 656e7472795fc3a92e4d4f56"


def test_build_assets_classifies_canonical_relative_text(tmp_path: pathlib.Path) -> None:
    """M010: a valid UTF-8 C1 code point keeps control-character precedence."""
    corpus = os.fsencode(tmp_path / "corpus")
    os.mkdir(corpus)  # noqa: PTH102 - byte paths are the subject under test
    descriptor = os.open(corpus + b"/entry_\xc2\x80.MOV", os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    script = """
import pathlib
import sys
from pose_estimation import inventory
root = pathlib.Path(sys.argv[1])
asset = inventory.build_assets(root, root.parent / 'out', checksums=False)[0]
print(sys.getfilesystemencoding(), asset.reason_code)
"""

    completed = _run_ascii_locale(script, os.fsdecode(corpus))

    assert completed.stdout.strip() == "ascii control_character_in_path"


def _canonical_asset(
    asset_id: str,
    *,
    subject_ordinal: int = 1,
    view: str = "above",
    task: str = "peg",
    side: str = "l",
) -> inventory.AssetRecord:
    return inventory.AssetRecord(
        asset_id=asset_id,
        source_path=f"{asset_id}.MOV",
        disposition=inventory.CANONICAL,
        reason_code=inventory.REASON_OK,
        parse=inventory.StemParse(
            subject_ordinal=subject_ordinal,
            view=view,
            task=task,
            side=side,
        ),
        facts=inventory._skipped_facts(),
        size_bytes=1,
        content_sha256="",
    )


def test_validate_generation_wraps_invalid_utf8_census(tmp_path: pathlib.Path) -> None:
    """M024: invalid UTF-8 census bytes cross the boundary as InventoryError."""
    (tmp_path / inventory.CENSUS_FILENAME).write_bytes(b"\xff")

    with pytest.raises(inventory.InventoryError):
        inventory.validate_generation(tmp_path)


def test_validate_generation_wraps_missing_census(tmp_path: pathlib.Path) -> None:
    """M025: a missing census crosses the boundary as InventoryError."""
    with pytest.raises(inventory.InventoryError):
        inventory.validate_generation(tmp_path)


def test_validate_generation_wraps_table_read_oserror(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M026: every table-read OSError crosses the boundary as InventoryError."""
    (tmp_path / inventory.CENSUS_FILENAME).write_text(
        inventory.render_json({"generation": {}}),
        encoding="utf-8",
    )
    original_read_bytes = pathlib.Path.read_bytes

    def fail_assets_read(path: pathlib.Path) -> bytes:
        if path.name == inventory.ASSETS_FILENAME:
            raise OSError("synthetic table read failure")
        return original_read_bytes(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", fail_assets_read)

    with pytest.raises(inventory.InventoryError):
        inventory.validate_generation(tmp_path)


def test_capture_membership_rejects_an_extra_duplicate_member() -> None:
    """M030: capture membership preserves each asset identifier's multiplicity."""
    first = _canonical_asset("a-first")
    second = _canonical_asset("a-second", view="left")
    capture = inventory.CaptureRecord(
        capture_id=first.capture_id,
        subject_ordinal=1,
        task="peg",
        side="l",
        assets=(first, second, first),
    )

    with pytest.raises(inventory.InventoryError, match="membership"):
        inventory.check_invariants(
            [first, second],
            [capture],
            [first.source_path, second.source_path],
        )


def test_each_capture_contains_only_its_own_members() -> None:
    """M031: swapping members between captures violates their row identifiers."""
    first = _canonical_asset("a-first")
    second = _canonical_asset("a-second", task="key")
    captures = [
        inventory.CaptureRecord(first.capture_id, 1, "peg", "l", (second,)),
        inventory.CaptureRecord(second.capture_id, 1, "key", "l", (first,)),
    ]

    with pytest.raises(inventory.InventoryError, match="belongs to another capture"):
        inventory.check_invariants(
            [first, second],
            captures,
            [first.source_path, second.source_path],
        )


def test_capture_coverage_rejects_an_extra_empty_row() -> None:
    """M033: capture rows equal, rather than merely include, canonical family IDs."""
    asset = _canonical_asset("a-only")
    captures = [
        inventory.CaptureRecord(asset.capture_id, 1, "peg", "l", (asset,)),
        inventory.CaptureRecord("s02-key-r", 2, "key", "r", ()),
    ]

    with pytest.raises(inventory.InventoryError, match="cover"):
        inventory.check_invariants([asset], captures, [asset.source_path])


class _StatFails:
    def is_symlink(self) -> bool:
        return False

    def stat(self) -> os.stat_result:
        raise OSError("synthetic size lookup failure")


def test_size_lookup_oserror_does_not_escape() -> None:
    """M047: a failed size lookup remains a per-asset unknown fact."""
    inventory._size_of(_StatFails())


def test_failed_size_lookup_publishes_a_blank_cell_not_zero() -> None:
    """M048: an absent size fact stays distinct from a measured zero-byte file."""
    record = inventory.AssetRecord(
        asset_id="a-size-unknown",
        source_path="size-unknown.MOV",
        disposition=inventory.EXCLUDED,
        reason_code="read_error",
        parse=inventory.StemParse(reason_code="read_error"),
        facts=inventory._skipped_facts(),
        size_bytes=inventory._size_of(_StatFails()),
        content_sha256="",
    )

    assert inventory.asset_row(record)["size_bytes"] == ""


def test_fourcc_keeps_printable_spaces_but_rejects_controls() -> None:
    """M052: FOURCC text is verbatim only when all four bytes are printable."""

    def code(raw: bytes) -> float:
        return float(sum(byte << (8 * index) for index, byte in enumerate(raw)))

    assert video_io._fourcc_text(code(b"DIB ")) == "DIB "
    assert video_io._fourcc_text(code(b"DIB\x01")) == ""


def test_unreadable_probe_outranks_grammar_quarantine(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M063: an unreadable container owns the outcome even when its stem is invalid."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "invalid.MOV").write_bytes(b"synthetic")
    facts = video_io.ContainerFacts(
        probe_status=video_io.PROBE_OPEN_FAILED,
        backend_name="",
        reported_width=0,
        reported_height=0,
        reported_avg_fps=0.0,
        reported_frame_count=0,
        reported_rotation_deg=0,
        reported_fourcc="",
        orientation_auto=False,
    )
    monkeypatch.setattr(inventory, "probe_container", lambda _path: facts)

    asset = inventory.build_assets(corpus, tmp_path / "out", checksums=False)[0]

    assert asset.disposition == inventory.EXCLUDED
    assert asset.reason_code == "probe_unreadable"


def test_view_coverage_counts_distinct_views_not_assets() -> None:
    """M068: duplicate-view assets contribute one view to family coverage."""
    first = _canonical_asset("a-first")
    second = _canonical_asset("a-second")
    capture = inventory.CaptureRecord(
        capture_id=first.capture_id,
        subject_ordinal=1,
        task="peg",
        side="l",
        assets=(first, second),
    )

    census = inventory.build_census(
        [first, second],
        [capture],
        checksums=False,
        opencv_version="test",
        backend_name="",
    )

    assert census["captures"]["view_coverage"] == {1: 1}


def test_n2_lowercases_without_compatibility_folding() -> None:
    """M071: N2 preserves characters that lowercase does not compatibility-fold."""
    assert inventory.normalize_stem("ẞ_ﬁ") == "ß_ﬁ"


def test_capture_fps_is_unknown_when_any_member_is_nonfinite() -> None:
    """M018: one unknown member rate blanks every family FPS aggregate."""

    def member(asset_id: str, view: str, fps: float) -> inventory.AssetRecord:
        asset = _canonical_asset(asset_id, view=view)
        asset.facts = video_io.ContainerFacts(
            probe_status=video_io.PROBE_OPENED,
            backend_name="FAKE",
            reported_width=1,
            reported_height=1,
            reported_avg_fps=fps,
            reported_frame_count=30,
            reported_rotation_deg=0,
            reported_fourcc="TEST",
            orientation_auto=True,
        )
        return asset

    finite = member("a-finite", "above", 30.0)
    nonfinite = member("a-nonfinite", "left", float("nan"))
    capture = inventory.CaptureRecord("s01-peg-l", 1, "peg", "l", (finite, nonfinite))
    row = inventory.capture_row(capture)

    assert (
        row["reported_fps_min"],
        row["reported_fps_max"],
        row["reported_fps_spread_hz"],
    ) == ("", "", "")
