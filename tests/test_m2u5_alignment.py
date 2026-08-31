"""M2.5 phase-2 red suite: one test per phase-1 case, diff-blind expectations.

Rulings A01-A03 in `.agent/archive/contract-m2u5.md` fix every ambiguity these
cases raised, so an expectation here answers to a ruling and never to a guess.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import pathlib
import re
import runpy
import subprocess
import sys
from collections.abc import Iterable

import numpy as np
import pytest

from pose_estimation import inventory, qualify, sessions
from pose_estimation.measure import audio_offset
from test_qualify import _one_asset, _publish, _rows, _uniform, _write_media
from test_sessions import _canonical, _tree_snapshot

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALIGNMENT_DOCS = (
    "docs/capture_protocol.md",
    "docs/technical/multicam.md",
    "docs/technical/qualification.md",
    "docs/technical/sessions.md",
    "docs/technical/validation.md",
)


def _pair_row(
    asset_a: str,
    asset_b: str,
    offset_s: str,
    status: str = qualify.PAIR_OK_UNCORROBORATED,
    **extra: str,
) -> dict[str, str]:
    return {"asset_a": asset_a, "asset_b": asset_b, "offset_s": offset_s, "status": status, **extra}


def _directed(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], float]:
    return qualify._directed_edges(list(rows))


def _surface(
    members: list[str],
    views: dict[str, str],
    directed: dict[tuple[str, str], float],
    *,
    sync_measured: bool = True,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    event = {
        "event_id": "event-01",
        "capture_id": "capture-01",
        "n_cameras": str(len(members)),
        "views": "|".join(sorted({views[member] for member in members})),
    }
    member_map = {event["event_id"]: members}
    camera_names = {member: f"cam-{member}" for member in members}
    camera_rows = qualify._camera_rows(
        [event], member_map, camera_names, views, directed, sync_measured=sync_measured
    )
    event_row = qualify._event_rows(
        [event],
        member_map,
        camera_rows=camera_rows,
        directed=directed,
        sync_measured=sync_measured,
    )[0]
    return camera_rows, event_row


def _by_asset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["asset_id"]: row for row in rows}


def _assert_alignment_tables(
    camera_rows: list[dict[str, str]], event_rows: list[dict[str, str]]
) -> None:
    events = {row["event_id"]: row for row in event_rows}
    for event_id in sorted({row["event_id"] for row in camera_rows}):
        cameras = [row for row in camera_rows if row["event_id"] == event_id]
        references = [row for row in cameras if row["is_reference"] == "1"]
        assert len(references) == 1, f"reference count {len(references)}"
        reference = references[0]
        assert reference["offset_status"] == qualify.OFFSET_REFERENCE
        assert reference["offset_s"] == "0.000000000"
        assert {row["reference_camera"] for row in cameras} == {reference["camera_name"]}
        offsets = [
            float(row["offset_s"])
            for row in cameras
            if row["offset_status"] in qualify.OFFSET_SOLVED_STATUSES
        ]
        expected_span = f"{max(offsets) - min(offsets):.9f}" if len(offsets) > 1 else ""
        assert events[event_id]["offset_span_s"] == expected_span
        connected = not any(row["offset_status"] == qualify.OFFSET_UNREACHABLE for row in cameras)
        assert events[event_id]["graph_connected"] == ("1" if connected else "0")
        assert events[event_id]["sync_status"] == (
            qualify.SYNC_CONNECTED if connected else qualify.SYNC_UNCONNECTED
        )


def _asset_ref(asset_id: str, view: str) -> qualify.AssetRef:
    return qualify.AssetRef(asset_id, "capture", view, "task", "left", "1", f"{asset_id}.mov", 2)


def _decode_facts() -> qualify.DecodeFacts:
    return qualify.DecodeFacts(
        qualify.DECODE_OK, "h264", "model/os", 48_000, 2, 0.033, 0.033, 0.033, True
    )


def _sync_measurement(
    *,
    offset_s: str = "0.200000000",
    peak_rms: str = "5.000000000",
    peak_ratio: str = "2.500000000",
    drift_ppm: str = "",
    drift_se: str = "",
    status_audio: str = "ok",
    status_visual: str = "low_peak_correlation",
    offset_visual_s: str = "",
) -> dict[str, str]:
    return {
        "offset_audio_s": offset_s,
        "peak_rms_audio": peak_rms,
        "peak_ratio_audio": peak_ratio,
        "status_audio": status_audio,
        "offset_visual_s": offset_visual_s,
        "status_visual": status_visual,
        "drift_ppm": drift_ppm,
        "drift_se": drift_se,
        "overlap_s": "4.000000000",
        "dur_a": "5.000000000",
        "dur_b": "5.000000000",
        "audio_rate_a": "48000",
        "audio_rate_b": "48000",
    }


def _document(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _real_qualification() -> pathlib.Path:
    path = ROOT / "qualification" / qualify.CAMERAS_QC_FILENAME
    if not path.is_file():
        pytest.skip("v4 qualification/cameras_qc.csv is not published")
    return path.parent


def test_c01_carrier(tmp_path: pathlib.Path) -> None:
    """P01: the existing qualification root carries all four tables and its marker."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)

    assert {path.name for path in out.iterdir()} == {
        *qualify.CSV_FILENAMES,
        qualify.QUALIFICATION_FILENAME,
    }
    assert not any("alignment" in path.name for path in tmp_path.iterdir())


def test_c02_no_write_back(tmp_path: pathlib.Path) -> None:
    """P01: qualification publication cannot mutate the session tree."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    before = _tree_snapshot(sessions_dir)
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    assert _tree_snapshot(sessions_dir) == before


def test_c03_version_registration(tmp_path: pathlib.Path) -> None:
    """P02: v4 registers and digests cameras_qc as an ordinary payload."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    marker = qualify.run(inventory_dir, sessions_dir, corpus, out)
    camera_digest = hashlib.sha256((out / qualify.CAMERAS_QC_FILENAME).read_bytes()).hexdigest()

    assert qualify.GENERATOR_VERSION == "v4"
    assert qualify.CAMERAS_QC_FILENAME in qualify.CSV_FILENAMES
    assert qualify.CAMERAS_QC_FILENAME in qualify.GENERATION_KEYS
    assert marker["generation"][qualify.CAMERAS_QC_FILENAME] == camera_digest


def test_c04_payload_tamper(tmp_path: pathlib.Path) -> None:
    """P02: the camera payload digest detects a one-byte edit."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    camera_table = out / qualify.CAMERAS_QC_FILENAME
    camera_table.write_bytes(camera_table.read_bytes() + b"x")

    with pytest.raises(qualify.QualifyError, match=r"cameras_qc\.csv is a different generation"):
        qualify.validate_generation(out)


def test_c05_marker_shape(tmp_path: pathlib.Path) -> None:
    """P02: the v4 marker cannot omit the camera digest key."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    marker_path = out / qualify.QUALIFICATION_FILENAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    del marker["generation"][qualify.CAMERAS_QC_FILENAME]
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(qualify.QualifyError, match="not this generator's document"):
        qualify.validate_generation(out)


def _camera_cell_row(**updates: str) -> dict[str, str]:
    row = {
        "event_id": "event-01",
        "asset_id": "asset-a",
        "offset_s": "0.125000000",
        "offset_status": qualify.OFFSET_SOLVED,
        "is_reference": "0",
        "reference_camera": "cam-a",
    }
    row.update(updates)
    return row


def test_c06_exact_header(tmp_path: pathlib.Path) -> None:
    """P03: cameras_qc publishes the ruled eight-column header byte-for-byte."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    header = (out / qualify.CAMERAS_QC_FILENAME).read_bytes().splitlines()[0]

    expected = (
        b"event_id,asset_id,camera_name,view,offset_s,offset_status,is_reference,reference_camera"
    )
    assert tuple(expected.decode().split(",")) == qualify.CAMERAS_QC_COLUMNS
    assert header == expected


def test_c07_fullmatch_newline() -> None:
    """P03: a valid decimal prefix cannot hide a trailing newline."""
    row = _camera_cell_row(offset_s="0.125000000\n")
    with pytest.raises(qualify.QualifyError, match="offset_s cell"):
        qualify._assert_cell_alphabets(
            [row],
            qualify.CAMERA_CELL_ALPHABETS,
            qualify.CAMERAS_QC_FILENAME,
            ("event_id", "asset_id"),
        )


def test_c08_boolean_alphabet() -> None:
    """P03: is_reference uses the closed CSV boolean alphabet [01]."""
    row = _camera_cell_row(is_reference="yes")
    with pytest.raises(qualify.QualifyError, match="is_reference cell"):
        qualify._assert_cell_alphabets(
            [row],
            qualify.CAMERA_CELL_ALPHABETS,
            qualify.CAMERAS_QC_FILENAME,
            ("event_id", "asset_id"),
        )


def test_c09_status_alphabet() -> None:
    """P03/A03: offset_status is exactly the four-token total partition."""
    assert {
        "reference",
        "solved",
        "unreachable",
        "unmeasured",
    } == qualify.OFFSET_STATUSES
    for status in qualify.OFFSET_STATUSES:
        qualify._assert_cell_alphabets(
            [_camera_cell_row(offset_status=status)],
            qualify.CAMERA_CELL_ALPHABETS,
            qualify.CAMERAS_QC_FILENAME,
            ("event_id", "asset_id"),
        )
    with pytest.raises(qualify.QualifyError, match="offset_status cell"):
        qualify._assert_cell_alphabets(
            [_camera_cell_row(offset_status="partial")],
            qualify.CAMERA_CELL_ALPHABETS,
            qualify.CAMERAS_QC_FILENAME,
            ("event_id", "asset_id"),
        )


def test_c10_empty_population(tmp_path: pathlib.Path) -> None:
    """P04: an empty validated population publishes the exact header and no rows."""
    inventory_dir, sessions_dir, corpus, out = _publish(tmp_path, [])
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    table = out / qualify.CAMERAS_QC_FILENAME

    assert table.read_text(encoding="utf-8") == inventory.render_csv(qualify.CAMERAS_QC_COLUMNS, [])
    assert _rows(table) == []


def test_c11_placed_only_population(tmp_path: pathlib.Path) -> None:
    """P04/A03: rows cover placed assets only and sort by event then asset id."""
    canonical = [_canonical(1, "above"), _canonical(1, "left")]
    quarantined = dataclasses.replace(
        _canonical(1, "right"),
        subject_ordinal=None,
        view="",
        task="",
        side="",
        disposition=inventory.QUARANTINED,
        reason_code="token_count",
    )
    inventory_dir, sessions_dir, corpus, out = _publish(tmp_path, [*canonical, quarantined])
    for asset in canonical:
        _write_media(corpus / asset.source_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    cameras = _rows(out / qualify.CAMERAS_QC_FILENAME)
    events = _rows(out / qualify.EVENTS_QC_FILENAME)

    assert [row["asset_id"] for row in cameras] == sorted(asset.asset_id for asset in canonical)
    assert {row["offset_status"] for row in cameras} == {qualify.OFFSET_UNMEASURED}
    assert {row["is_reference"] for row in cameras} == {""}
    assert events[0]["sync_status"] == ""


def test_c12_explicit_missing_estimate() -> None:
    """P04: an unreachable placed camera remains an explicit row with no offset."""
    directed = _directed([_pair_row("a", "b", "0.200000000")])
    cameras, _ = _surface(["a", "b", "c"], {"a": "above", "b": "right", "c": "left"}, directed)
    rows = _by_asset(cameras)

    assert len(cameras) == 3
    assert rows["c"]["offset_s"] == ""
    assert rows["c"]["offset_status"] == qualify.OFFSET_UNREACHABLE


def test_c13_two_node_solve() -> None:
    """P05: one accepted edge is reproduced exactly with the reference gauge pinned."""
    directed = _directed([_pair_row("a", "b", "0.375000000")])
    solved = qualify._solve_offsets({"a", "b"}, directed, "a")
    cameras, _ = _surface(["a", "b"], {"a": "above", "b": "left"}, directed)
    rows = _by_asset(cameras)

    assert solved == pytest.approx({"a": 0.0, "b": 0.375})
    assert rows["a"]["offset_s"] == "0.000000000"
    assert rows["a"]["offset_status"] == qualify.OFFSET_REFERENCE
    assert rows["b"]["offset_s"] == "0.375000000"
    assert rows["b"]["offset_status"] == qualify.OFFSET_SOLVED


def test_c14_edge_reversal() -> None:
    """P05/P09: reversing an edge negates its reading without changing the solve."""
    forward = _directed([_pair_row("a", "b", "0.375000000")])
    reverse = _directed([_pair_row("b", "a", "-0.375000000")])
    expected = qualify._solve_offsets({"a", "b"}, forward, "a")

    assert reverse[("a", "b")] == 0.375
    assert qualify._solve_offsets({"a", "b"}, reverse, "a") == pytest.approx(expected)

    zero = _directed([_pair_row("a", "b", "-0.000000000")])
    cameras, _ = _surface(["a", "b"], {"a": "above", "b": "left"}, zero)
    assert _by_asset(cameras)["b"]["offset_s"] == "0.000000000"


def test_c15_inconsistent_triangle() -> None:
    """P05: unweighted least squares distributes an inconsistent closure residual."""
    directed = _directed(
        [
            _pair_row("a", "b", "1.000000000"),
            _pair_row("b", "c", "1.000000000"),
            _pair_row("a", "c", "2.300000000"),
        ]
    )
    solved = qualify._solve_offsets({"a", "b", "c"}, directed, "a")
    cameras, _ = _surface(["a", "b", "c"], {"a": "above", "b": "left", "c": "right"}, directed)

    assert solved == pytest.approx({"a": 0.0, "b": 1.1, "c": 2.2})
    assert {row["asset_id"]: row["offset_s"] for row in cameras} == {
        "a": "0.000000000",
        "b": "1.100000000",
        "c": "2.200000000",
    }


def test_c16_exact_open_path() -> None:
    """P05: a two-edge tree solves exactly rather than distributing nonexistent error."""
    directed = _directed(
        [
            _pair_row("a", "b", "-0.250000000"),
            _pair_row("b", "c", "0.750000000"),
        ]
    )
    solved = qualify._solve_offsets({"a", "b", "c"}, directed, "a")
    cameras, _ = _surface(["a", "b", "c"], {"a": "above", "b": "left", "c": "right"}, directed)

    assert solved == pytest.approx({"a": 0.0, "b": -0.25, "c": 0.5})
    assert {row["asset_id"]: row["offset_s"] for row in cameras} == {
        "a": "0.000000000",
        "b": "-0.250000000",
        "c": "0.500000000",
    }


def test_c17_qualified_status_pair() -> None:
    """P05/P06: both qualified statuses solve, while unusable accepted offsets abort."""
    for status in qualify.QUALIFIED_PAIR_STATUSES:
        directed = _directed([_pair_row("a", "b", "0.200000000", status)])
        cameras, _ = _surface(["a", "b"], {"a": "above", "b": "left"}, directed)
        rows = _by_asset(cameras)
        assert rows["a"]["offset_s"] == "0.000000000"
        assert rows["b"]["offset_s"] == "0.200000000"
        assert rows["b"]["offset_status"] == qualify.OFFSET_SOLVED

    for unusable in ("", "nan", "inf", "-inf"):
        with pytest.raises(qualify.QualifyError, match="not a finite number"):
            _directed([_pair_row("a", "b", unusable)])


def test_c18_refused_fused_statuses() -> None:
    """P06: no refused fused verdict contributes an incidence edge."""
    refused = {
        qualify.PAIR_CONTRADICTED,
        qualify.PAIR_VISUAL_ONLY,
        qualify.PAIR_NEITHER_ACCEPTED,
        qualify.PAIR_UNMEASURED,
    }
    assert refused.isdisjoint(qualify.QUALIFIED_PAIR_STATUSES)
    for status in refused:
        directed = _directed([_pair_row("a", "b", "0.200000000", status)])
        cameras, _ = _surface(["a", "b"], {"a": "above", "b": "left"}, directed)
        rows = _by_asset(cameras)
        assert directed == {}
        assert rows["a"]["offset_s"] == "0.000000000"
        assert rows["b"]["offset_s"] == ""
        assert rows["b"]["offset_status"] == qualify.OFFSET_UNREACHABLE


def test_c19_no_estimator_substitution() -> None:
    """P06: disagreement refuses both estimators; no value or average reaches the solve."""
    measured = _sync_measurement(
        offset_s="0.100000000",
        offset_visual_s="0.500000000",
        status_visual="ok",
    )
    status = qualify.fuse_pair(measured)
    directed = _directed([_pair_row("a", "b", measured["offset_audio_s"], status)])
    cameras, _ = _surface(["a", "b"], {"a": "above", "b": "left"}, directed)
    camera_b = _by_asset(cameras)["b"]

    assert status == qualify.PAIR_CONTRADICTED
    assert directed == {}
    assert camera_b["offset_s"] == ""
    assert camera_b["offset_status"] == qualify.OFFSET_UNREACHABLE


def test_c20_reference_inside_partial_pair() -> None:
    """P07: the reference component publishes while an isolated camera stays unreachable."""
    directed = _directed([_pair_row("a", "b", "0.200000000")])
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "right", "c": "left"}, directed)
    rows = _by_asset(cameras)

    assert (rows["a"]["offset_s"], rows["a"]["offset_status"]) == (
        "0.000000000",
        qualify.OFFSET_REFERENCE,
    )
    assert (rows["b"]["offset_s"], rows["b"]["offset_status"]) == (
        "0.200000000",
        qualify.OFFSET_SOLVED,
    )
    assert (rows["c"]["offset_s"], rows["c"]["offset_status"]) == (
        "",
        qualify.OFFSET_UNREACHABLE,
    )
    assert event["graph_connected"] == "0"
    assert event["sync_status"] == qualify.SYNC_UNCONNECTED


def test_c21_reference_isolated() -> None:
    """P07: a disconnected non-reference component never receives its own gauge."""
    directed = _directed([_pair_row("b", "c", "0.200000000")])
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "left", "c": "right"}, directed)
    rows = _by_asset(cameras)

    assert rows["a"]["offset_s"] == "0.000000000"
    assert rows["a"]["offset_status"] == qualify.OFFSET_REFERENCE
    assert {rows[name]["offset_status"] for name in ("b", "c")} == {qualify.OFFSET_UNREACHABLE}
    assert {rows[name]["offset_s"] for name in ("b", "c")} == {""}
    assert event["graph_connected"] == "0"
    assert event["sync_status"] == qualify.SYNC_UNCONNECTED


def test_c22_zero_edge_two_camera_event() -> None:
    """P07: zero accepted edges retain the semantic reference and flag the event."""
    cameras, event = _surface(["a", "b"], {"a": "left", "b": "right"}, {})
    rows = _by_asset(cameras)

    assert (rows["a"]["offset_s"], rows["a"]["offset_status"]) == (
        "0.000000000",
        qualify.OFFSET_REFERENCE,
    )
    assert (rows["b"]["offset_s"], rows["b"]["offset_status"]) == (
        "",
        qualify.OFFSET_UNREACHABLE,
    )
    assert event["graph_connected"] == "0"
    assert event["sync_status"] == qualify.SYNC_UNCONNECTED


def test_c23_hierarchy_beats_identifier() -> None:
    """P08: the above view outranks lexicographically lower asset identifiers."""
    cameras, _ = _surface(["z", "a", "b"], {"z": "above", "a": "left", "b": "right"}, {})
    rows = _by_asset(cameras)

    assert [row["asset_id"] for row in cameras if row["is_reference"] == "1"] == ["z"]
    assert {row["reference_camera"] for row in cameras} == {rows["z"]["camera_name"]}


def test_c24_hierarchy_fallback_to_left() -> None:
    """P08: left outranks right even when the right asset id sorts first."""
    cameras, _ = _surface(["z", "a"], {"z": "left", "a": "right"}, {})
    rows = _by_asset(cameras)

    assert qualify._view_reference(["a", "z"], {"z": "left", "a": "right"}) == "z"
    assert rows["z"]["is_reference"] == "1"
    assert rows["a"]["is_reference"] == "0"


def test_c25_hierarchy_fallback_to_right() -> None:
    """P08/P14: a right-only singleton is its exact-zero reference and is connected."""
    cameras, event = _surface(["a"], {"a": "right"}, {})
    row = cameras[0]

    assert row["asset_id"] == "a"
    assert row["is_reference"] == "1"
    assert row["offset_status"] == qualify.OFFSET_REFERENCE
    assert row["offset_s"] == "0.000000000"
    assert event["graph_connected"] == "1"
    assert event["sync_status"] == qualify.SYNC_CONNECTED


def test_c26_same_view_tie() -> None:
    """P08: the lowest asset id breaks a same-view tie independent of input order."""
    views = {"z": "above", "a": "above"}
    assert qualify._view_reference(["z", "a"], views) == "a"
    assert qualify._view_reference(["a", "z"], views) == "a"


def test_c27_positive_lead() -> None:
    """P09: the estimator and solve compose a constructed 375 ms camera lead."""
    rate = 2_000
    delay = round(0.375 * rate)
    signal_a = np.random.default_rng(17).normal(size=rate * 20)
    signal_b = np.concatenate((np.zeros(delay), signal_a[:-delay]))
    forward = audio_offset.gcc_phat_peak(signal_a, signal_b, rate, min_overlap_s=1.0)
    reverse = audio_offset.gcc_phat_peak(signal_b, signal_a, rate, min_overlap_s=1.0)

    assert forward.lag_s == pytest.approx(0.375, abs=0.0005)
    assert reverse.lag_s == pytest.approx(-forward.lag_s, abs=1e-12)
    directed = _directed([_pair_row("a", "b", f"{forward.lag_s:.9f}")])
    cameras, _ = _surface(["a", "b"], {"a": "above", "b": "left"}, directed)
    offset_b = float(_by_asset(cameras)["b"]["offset_s"])
    assert offset_b == pytest.approx(0.375, abs=0.0005)
    assert 5.375 - offset_b == pytest.approx(5.0, abs=0.0005)


def test_c28_negative_lead() -> None:
    """P09: the estimator and solve compose a constructed 250 ms camera lag."""
    rate = 2_000
    delay = round(0.250 * rate)
    signal_a = np.random.default_rng(23).normal(size=rate * 20)
    signal_b = np.concatenate((np.zeros(delay), signal_a[:-delay]))
    negative = audio_offset.gcc_phat_peak(signal_b, signal_a, rate, min_overlap_s=1.0)

    assert negative.lag_s == pytest.approx(-0.250, abs=0.0005)
    directed = _directed([_pair_row("a", "b", f"{negative.lag_s:.9f}")])
    cameras, _ = _surface(["a", "b"], {"a": "above", "b": "left"}, directed)
    offset_b = float(_by_asset(cameras)["b"]["offset_s"])
    assert offset_b == pytest.approx(-0.250, abs=0.0005)
    assert 4.750 - offset_b == pytest.approx(5.0, abs=0.0005)


def test_c29_confidence_extreme_triangle() -> None:
    """P10: peak statistics cannot weight the least-squares objective."""
    directed = _directed(
        [
            _pair_row("a", "b", "1.000000000", peak_rms="1e-30", peak_ratio="1e-30"),
            _pair_row("b", "c", "1.000000000", peak_rms="1e-30", peak_ratio="1e-30"),
            _pair_row("a", "c", "2.300000000", peak_rms="1e30", peak_ratio="1e30"),
        ]
    )
    assert qualify._solve_offsets({"a", "b", "c"}, directed, "a") == pytest.approx(
        {"a": 0.0, "b": 1.1, "c": 2.2}
    )


def test_c30_confidence_metamorphism() -> None:
    """P10: changing only peak statistics leaves camera bytes and event cells fixed."""

    def publish(peak_rms: str, peak_ratio: str) -> tuple[str, dict[str, str]]:
        directed = _directed(
            [
                _pair_row("a", "b", "1.000000000", peak_rms=peak_rms, peak_ratio=peak_ratio),
                _pair_row("b", "c", "1.000000000", peak_rms=peak_rms, peak_ratio=peak_ratio),
                _pair_row("a", "c", "2.300000000", peak_rms=peak_rms, peak_ratio=peak_ratio),
            ]
        )
        cameras, event = _surface(
            ["a", "b", "c"], {"a": "above", "b": "left", "c": "right"}, directed
        )
        return inventory.render_csv(qualify.CAMERAS_QC_COLUMNS, cameras), event

    assert publish("0.000000001", "0.000000001") == publish(
        "999999999.000000000", "777777777.000000000"
    )


def test_c31_rejection_rationale() -> None:
    """P10/A03: code and schema docs retain both measurements rejecting confidence weights."""
    surfaces = {
        "solver docstring": inspect.getdoc(qualify._solve_offsets) or "",
        "qualification schema": _document("docs/technical/qualification.md"),
    }
    required = ("unweighted", "+0.4141", "peak_rms", "+0.0659", "peak_ratio")
    missing = [
        f"{surface}: {token}"
        for surface, text in surfaces.items()
        for token in required
        if token.lower() not in text.lower()
    ]
    assert missing == []


def test_c32_schema_exclusion() -> None:
    """P11: cameras_qc carries no unsupported per-camera uncertainty field."""
    forbidden = (
        "standard_error",
        "variance",
        "confidence",
        "interval",
        "uncertainty",
        "weight",
        "residual",
    )
    header = "|".join(qualify.CAMERAS_QC_COLUMNS).lower()
    assert not any(token in header for token in forbidden)


def test_c33_closure_claim_boundary() -> None:
    """P11: closure is edge self-consistency and never per-camera uncertainty or accuracy."""
    directed = _directed(
        [
            _pair_row("a", "b", "0.100000000"),
            _pair_row("b", "c", "0.200000000"),
            _pair_row("a", "c", "0.310000000"),
        ]
    )
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "left", "c": "right"}, directed)
    schema = _document("docs/technical/qualification.md").lower()

    assert event["closure_residual_s"] == "0.010000000"
    assert not any(
        "uncert" in column or "residual" in column for column in qualify.CAMERAS_QC_COLUMNS
    )
    assert "closure_residual_s` is a self-consistency statistic" in schema
    assert "never read it as an accuracy statistic" in schema
    assert all("uncert" not in row for row in cameras)


def test_c34_camera_schema_has_no_rate() -> None:
    """P12/A03: applied camera alignment has no rate term; pair diagnostics remain."""
    camera_header = "|".join(qualify.CAMERAS_QC_COLUMNS).lower()
    forbidden = ("rate", "drift", "ppm", "slope", "clock_scale")
    assert not any(token in camera_header for token in forbidden)
    assert {"drift_ppm", "drift_se"} <= set(qualify.PAIRS_QC_COLUMNS)


def test_c35_drift_metamorphism() -> None:
    """P12: changing pair drift diagnostics cannot move applied alignment outputs."""
    assets = [_asset_ref("a", "above"), _asset_ref("b", "left")]
    facts = {asset.asset_id: _decode_facts() for asset in assets}

    def publish(drift_ppm: str, drift_se: str) -> tuple[list[dict[str, str]], str, dict[str, str]]:
        pairs = qualify._pair_rows(
            assets,
            facts,
            {("a", "b"): _sync_measurement(drift_ppm=drift_ppm, drift_se=drift_se)},
        )
        cameras, event = _surface(
            ["a", "b"], {"a": "above", "b": "left"}, qualify._directed_edges(pairs)
        )
        return pairs, inventory.render_csv(qualify.CAMERAS_QC_COLUMNS, cameras), event

    first = publish("-999999.000000000", "0.000000001")
    second = publish("999999.000000000", "777777.000000000")
    assert first[1:] == second[1:]
    assert (first[0][0]["drift_ppm"], first[0][0]["drift_se"]) == (
        "-999999.000000000",
        "0.000000001",
    )
    assert (second[0][0]["drift_ppm"], second[0][0]["drift_se"]) == (
        "999999.000000000",
        "777777.000000000",
    )


def test_c36_triangle_cross_table_arithmetic() -> None:
    """P13: span derives from solved camera rows while closure derives from original edges."""
    directed = _directed(
        [
            _pair_row("a", "b", "1.000000000"),
            _pair_row("b", "c", "1.000000000"),
            _pair_row("a", "c", "2.300000000"),
        ]
    )
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "left", "c": "right"}, directed)

    assert {row["asset_id"]: row["offset_s"] for row in cameras} == {
        "a": "0.000000000",
        "b": "1.100000000",
        "c": "2.200000000",
    }
    assert event["offset_span_s"] == "2.200000000"
    assert event["closure_residual_s"] == "0.300000000"


def test_c37_partial_component_span() -> None:
    """P13/A03: a partial reference component publishes its own computable span."""
    directed = _directed([_pair_row("a", "b", "0.200000000")])
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "right", "c": "left"}, directed)

    assert sum(bool(row["offset_s"]) for row in cameras) == 2
    assert _by_asset(cameras)["c"]["offset_s"] == ""
    assert event["offset_span_s"] == "0.200000000"
    assert event["sync_status"] == qualify.SYNC_UNCONNECTED


def test_c38_no_one_point_span() -> None:
    """P13: neither a singleton nor a reference-only component fabricates a zero span."""
    singleton_cameras, singleton = _surface(["a"], {"a": "above"}, {})
    isolated_cameras, isolated = _surface(["a", "b"], {"a": "above", "b": "left"}, {})

    assert len([row for row in singleton_cameras if row["offset_s"]]) == 1
    assert len([row for row in isolated_cameras if row["offset_s"]]) == 1
    assert singleton["offset_span_s"] == ""
    assert isolated["offset_span_s"] == ""


def test_c39_partial_is_not_connected() -> None:
    """P14: two solved rows do not make a three-camera event connected."""
    directed = _directed([_pair_row("a", "b", "0.200000000")])
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "right", "c": "left"}, directed)

    assert sum(bool(row["offset_s"]) for row in cameras) == 2
    assert event["graph_connected"] == "0"
    assert event["sync_status"] == qualify.SYNC_UNCONNECTED


def test_c40_singleton_is_connected() -> None:
    """P14: graph connectivity is vacuously true for one measured camera."""
    cameras, event = _surface(["a"], {"a": "above"}, {})

    assert cameras[0]["offset_s"] == "0.000000000"
    assert cameras[0]["offset_status"] == qualify.OFFSET_REFERENCE
    assert event["graph_connected"] == "1"
    assert event["sync_status"] == qualify.SYNC_CONNECTED


def test_c41_independent_event_rebuild(tmp_path: pathlib.Path) -> None:
    """P15/A03: an independent CSV parser rebuilds every cross-table invariant."""
    directed = _directed(
        [
            _pair_row("a", "b", "1.000000000"),
            _pair_row("b", "c", "1.000000000"),
            _pair_row("a", "c", "2.300000000"),
        ]
    )
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "left", "c": "right"}, directed)
    camera_path = tmp_path / "cameras.csv"
    event_path = tmp_path / "events.csv"
    camera_path.write_text(
        inventory.render_csv(qualify.CAMERAS_QC_COLUMNS, cameras), encoding="utf-8"
    )
    event_path.write_text(
        inventory.render_csv(qualify.EVENTS_QC_COLUMNS, [event]), encoding="utf-8"
    )

    _assert_alignment_tables(_rows(camera_path), _rows(event_path))


def test_c42_duplicate_reference_mutant() -> None:
    """P15: the independent oracle detects a second syntactically valid reference."""
    directed = _directed([_pair_row("a", "b", "0.200000000")])
    cameras, event = _surface(["a", "b", "c"], {"a": "above", "b": "right", "c": "left"}, directed)
    mutant = [dict(row) for row in cameras]
    mutant[1]["is_reference"] = "1"

    with pytest.raises(AssertionError, match="reference count 2"):
        _assert_alignment_tables(mutant, [event])


def test_c43_reference_name_mutant() -> None:
    """P15: the independent oracle detects rows that name different references."""
    directed = _directed([_pair_row("a", "b", "0.200000000")])
    cameras, event = _surface(["a", "b"], {"a": "above", "b": "left"}, directed)
    mutant = [dict(row) for row in cameras]
    mutant[1]["reference_camera"] = mutant[1]["camera_name"]

    with pytest.raises(AssertionError):
        _assert_alignment_tables(mutant, [event])


def test_c44_manifest_literal_zero() -> None:
    """P16: render_manifest emits an integer zero for every legacy frame-trim field."""
    cameras = tuple(
        sessions.Camera(
            name=f"cam-{view}",
            view=view,
            asset_id=f"asset-{index}",
            link_name=f"cam-{view}.mov",
            source_relative=f"source/{index}.mov",
            content_sha256=str(index) * 64,
        )
        for index, view in enumerate(("above", "left", "right"), start=1)
    )
    event = sessions.Event(
        "event-01", "capture-01", 1, "task", "left", 1, "family", 0, "v1", cameras
    )
    payload = json.loads(sessions.render_manifest(event))

    assert [camera["sync_offset"] for camera in payload["cameras"]] == [0, 0, 0]
    assert all(type(camera["sync_offset"]) is int for camera in payload["cameras"])


def test_c45_publisher_independence(tmp_path: pathlib.Path) -> None:
    """P16: a nearby time-domain table cannot influence session manifest frame trims."""
    inventory_dir, sessions_dir, corpus, _ = _publish(
        tmp_path, [_canonical(1, "above"), _canonical(1, "left")]
    )
    before = _tree_snapshot(sessions_dir)
    nearby = tmp_path / "qualification"
    nearby.mkdir()
    rows = [
        {
            "event_id": "event-01",
            "asset_id": f"asset-{index}",
            "camera_name": f"cam-{view}",
            "view": view,
            "offset_s": offset,
            "offset_status": qualify.OFFSET_SOLVED,
            "is_reference": "0",
            "reference_camera": "cam-above",
        }
        for index, (view, offset) in enumerate(
            (("above", "999999.000000000"), ("left", "-999999.000000000")), start=1
        )
    ]
    (nearby / qualify.CAMERAS_QC_FILENAME).write_text(
        inventory.render_csv(qualify.CAMERAS_QC_COLUMNS, rows), encoding="utf-8"
    )

    sessions.run(inventory_dir, corpus, sessions_dir)
    assert _tree_snapshot(sessions_dir) == before
    manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in sessions_dir.rglob("session.json")
    ]
    assert manifests
    assert all(
        camera["sync_offset"] == 0 for manifest in manifests for camera in manifest["cameras"]
    )


def test_c46_delivered_estimator_wording() -> None:
    """P17/A03: multicam and validation describe delivered estimation without claiming fusion."""
    for relative in ("docs/technical/multicam.md", "docs/technical/validation.md"):
        text = _document(relative).lower().replace("`", " ")
        compact = re.sub(r"\s+", " ", text)
        assert "cameras_qc.csv" in compact, relative
        assert not re.search(r"audio(?: cross-correlation|-xcorr).{0,60}future", compact), relative
        assert not re.search(r"future.{0,60}audio(?: cross-correlation|-xcorr)", compact), relative
        assert "fusion" in compact, relative
        # P17 fixes the fact, not its phrasing: a conforming doc states somewhere
        # that fusion leaves the published offsets alone, in whatever words.
        assert re.search(
            r"fusion[^.]{0,60}(?:does not|never)[^.]{0,30}(?:apply|consume)|unapplied", compact
        ), relative


def test_c47_quantity_separation_wording() -> None:
    """P17/P16: sessions documentation separates legacy integer trim from time-domain alignment."""
    text = re.sub(r"\s+", " ", _document("docs/technical/sessions.md").lower().replace("`", " "))
    # O09 closes on the positive statement, never on a milestone id: shipped docs
    # carry no roadmap provenance, so the field must read as staying unmeasured
    # and no future tense may promise it a value.
    assert re.search(r"sync_offset\s+stays\s+0\b", text)
    assert "unmeasured" in text
    assert not re.search(r"(?:will|going to)\s+(?:be\s+)?(?:fill|populat|carry)", text)
    assert "integer" in text
    assert "pre-roll" in text
    assert "sync_offset" in text
    assert "time-domain" in text
    assert "offset_s" in text


def test_c48_camera_schema_wording() -> None:
    """P17: qualification documents the complete camera schema and application transform."""
    text = re.sub(
        r"\s+", " ", _document("docs/technical/qualification.md").lower().replace("`", " ")
    ).replace("\N{MINUS SIGN}", "-")
    assert all(column in text for column in qualify.CAMERAS_QC_COLUMNS)
    assert re.search(r"offset_s\s*=\s*t_camera\s*-\s*t_reference", text)
    assert "positive" in text
    assert "started earlier" in text
    assert re.search(r"t_ref\s*=\s*t_camera\s*-\s*offset_s", text)


def test_c49_capture_guidance_correction() -> None:
    """P17: capture guidance labels trim and parity as coarse QA, not sub-frame proof."""
    text = "\n".join(
        _document(relative).lower().replace("`", " ")
        for relative in ("docs/capture_protocol.md", "docs/technical/validation.md")
    )
    compact = re.sub(r"\s+", " ", text)
    assert "cameras_qc.csv" in compact
    assert "sub-frame" in compact
    assert "coarse" in compact
    assert "capture qa" in compact or "quality" in compact
    # The required disclaimer is itself "does not prove temporal alignment", so
    # forbidding the co-occurrence would fail on the correction O24 asked for.
    # Every proof claim must instead be negated inside its own window.
    claims = re.finditer(
        r"(?:sync_offset|frame(?:-count)? parity).{0,80}?(?:prove|certif)", compact
    )
    for claim in claims:
        assert re.search(r"\b(?:not|never|no)\b", claim.group()), claim.group()


def test_c50_rolling_shutter_preservation() -> None:
    """P18/A03: shipped claims retain the full sweep, proxy boundary, and non-negligibility."""
    text = "\n".join(_document(relative) for relative in ALIGNMENT_DOCS)
    text += "\n" + inspect.getsource(qualify._solve_offsets)
    compact = re.sub(
        r"\s+",
        " ",
        text.lower().replace("\N{EN DASH}", " to ").replace("\N{EM DASH}", " to "),
    )

    assert re.search(r"0\s+(?:to|-)\s+33\.33\s*ms", compact)
    assert re.search(r"12\.4\s+(?:to|-)\s+30\.9\s*ms", compact)
    proxy_index = compact.index("12.4")
    assert any(
        token in compact[max(0, proxy_index - 240) : proxy_index + 240]
        for token in ("proxy", "other device", "other-device")
    )
    assert not re.search(r"rolling shutter.{0,180}negligible", compact)


def test_c51_aac_priming_preservation() -> None:
    """P18: 3.891 ms remains a cancelled prediction; the measured residual remains 0 ms."""
    text = "\n".join(_document(relative) for relative in ALIGNMENT_DOCS)
    compact = re.sub(r"\s+", " ", text.lower())

    assert re.search(r"measured\s+0\s*ms\s+residual", compact)
    occurrences = [match.start() for match in re.finditer(r"3\.891\s*ms", compact)]
    assert occurrences
    for index in occurrences:
        window = compact[max(0, index - 220) : index + 220]
        assert any(
            token in window for token in ("predicted", "prediction", "cancel", "raw untrimmed")
        )
        assert any(token in window for token in ("0 ms", "never quote", "cancel"))


def test_c52_closure_language_preservation() -> None:
    """P18: closure remains self-consistency evidence and never becomes timing accuracy."""
    text = "\n".join(_document(relative) for relative in ALIGNMENT_DOCS)
    text += "\n" + _document("src/pose_estimation/qualify.py")
    compact = re.sub(r"\s+", " ", text.lower())

    assert "closure_residual_s` is a self-consistency statistic" in compact
    assert re.search(r"self-consistency.{0,60}never accuracy", compact)
    for sentence in re.split(r"[.!?]", compact):
        if "closure" in sentence and "accuracy" in sentence:
            assert any(negation in sentence for negation in ("never", "not", "cannot", "does not"))


def test_c54_cross_predicate_corpus_oracle() -> None:
    """P19/A02: the real v4 corpus has 379 rows, 355 offsets, and 24 unreachable rows."""
    out = _real_qualification()
    qualify.validate_generation(out)
    cameras = _rows(out / qualify.CAMERAS_QC_FILENAME)
    events = _rows(out / qualify.EVENTS_QC_FILENAME)

    assert len(cameras) == 379
    assert len(events) == 193
    assert len({row["event_id"] for row in cameras}) == 193
    assert [(row["event_id"], row["asset_id"]) for row in cameras] == sorted(
        (row["event_id"], row["asset_id"]) for row in cameras
    )
    assert sum(row["is_reference"] == "1" for row in cameras) == 193
    assert sum(bool(row["offset_s"]) for row in cameras) == 355
    assert sum(not row["offset_s"] for row in cameras) == 24
    assert sum(row["offset_status"] == qualify.OFFSET_REFERENCE for row in cameras) == 193
    assert sum(row["offset_status"] == qualify.OFFSET_SOLVED for row in cameras) == 162
    assert sum(row["offset_status"] == qualify.OFFSET_UNREACHABLE for row in cameras) == 24
    assert all(
        row["offset_s"] == "0.000000000"
        for row in cameras
        if row["offset_status"] == qualify.OFFSET_REFERENCE
    )
    assert sum(row["graph_connected"] == "1" for row in events) == 173
    assert sum(row["sync_status"] == qualify.SYNC_UNCONNECTED for row in events) == 20
    _assert_alignment_tables(cameras, events)


def test_c55_committed_determinism_rerun(tmp_path: pathlib.Path) -> None:
    """P20: the committed determinism gate byte-compares cameras_qc in every sweep."""
    qualify.validate_generation(_real_qualification())
    script = ROOT / "scripts/check_qualify_determinism.py"
    output = tmp_path / "qualify_determinism_results.json"
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests")))
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    payload = json.loads(output.read_text(encoding="utf-8"))
    expected_artifacts = {*qualify.CSV_FILENAMES, qualify.QUALIFICATION_FILENAME}
    assert all(
        set(digests) == expected_artifacts for digests in payload["baseline_sha256"].values()
    )
    assert payload["sweeps"]
    assert all(row[f"verdict_{qualify.CAMERAS_QC_FILENAME}"] == "PASS" for row in payload["sweeps"])
    namespace = runpy.run_path(str(script))
    assert payload["source_sha256"] == namespace["source_digests"]()


def test_c56_determinism_reach(tmp_path: pathlib.Path) -> None:
    """P20: camera bytes and their shaper participate in artifact and stale-source reach."""
    _real_qualification()
    namespace = runpy.run_path(str(ROOT / "scripts/check_qualify_determinism.py"))
    artifacts = set(namespace["ARTIFACTS"])
    sources = set(namespace["SOURCE_FILES"])
    current = namespace["source_digests"]()
    result = tmp_path / "result.json"

    assert qualify.CAMERAS_QC_FILENAME in artifacts
    assert "src/pose_estimation/qualify.py" in sources
    result.write_text(json.dumps({"source_sha256": current}), encoding="utf-8")
    assert namespace["stale_source_mismatches"](result, current) == []

    recorded = dict(current)
    recorded["src/pose_estimation/qualify.py"] = "0" * 64
    result.write_text(json.dumps({"source_sha256": recorded}), encoding="utf-8")
    assert namespace["stale_source_mismatches"](result, current) == [
        "src/pose_estimation/qualify.py"
    ]
