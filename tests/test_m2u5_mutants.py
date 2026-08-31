"""Mutation credentials for M2.5 cross-view alignment."""

from __future__ import annotations

import json
import pathlib
import runpy
from collections import Counter

import pytest

from pose_estimation import qualify, sessions


def _pair(
    first: str,
    second: str,
    offset: float,
    *,
    status: str = qualify.PAIR_OK_UNCORROBORATED,
    visual: float | None = None,
    peak_rms: float = 5.0,
) -> dict[str, str]:
    return {
        "asset_a": first,
        "asset_b": second,
        "offset_s": f"{offset:.9f}",
        "offset_visual_s": "" if visual is None else f"{visual:.9f}",
        "peak_rms": f"{peak_rms:.9f}",
        "peak_ratio": "2.500000000",
        "status": status,
    }


def _event(event_id: str, members: list[str], views: dict[str, str]) -> dict[str, str]:
    return {
        "event_id": event_id,
        "capture_id": f"capture-{event_id}",
        "n_cameras": str(len(members)),
        "views": "|".join(sorted({views[member] for member in members})),
    }


def _partial_surface() -> tuple[
    list[dict[str, str]], list[dict[str, str]], dict[tuple[str, str], float]
]:
    event_id = "event-partial"
    members = ["asset-a", "asset-b", "asset-c"]
    views = {"asset-a": "above", "asset-b": "left", "asset-c": "right"}
    names = {member: f"cam-{views[member]}" for member in members}
    directed = qualify._directed_edges([_pair("asset-a", "asset-b", 0.1)])
    camera_rows = qualify._camera_rows(
        [_event(event_id, members, views)],
        {event_id: members},
        names,
        views,
        directed,
        sync_measured=True,
    )
    event_rows = qualify._event_rows(
        [_event(event_id, members, views)],
        {event_id: members},
        camera_rows=camera_rows,
        directed=directed,
        sync_measured=True,
    )
    return camera_rows, event_rows, directed


def test_m2u5_sign_oracle_recovers_constructed_lead() -> None:
    directed = qualify._directed_edges([_pair("asset-a", "asset-b", 0.375)])
    assert directed[("asset-a", "asset-b")] == 0.375
    assert directed[("asset-b", "asset-a")] == -0.375
    solved = qualify._solve_offsets({"asset-a", "asset-b"}, directed, "asset-a")
    assert solved == pytest.approx({"asset-a": 0.0, "asset-b": 0.375})
    assert 1.125 - solved["asset-b"] == pytest.approx(0.75)


def test_m2u5_reference_pin_removes_incidence_nullspace() -> None:
    directed = qualify._directed_edges([_pair("asset-a", "asset-b", 0.25)])
    solved = qualify._solve_offsets({"asset-a", "asset-b"}, directed, "asset-a")
    assert solved["asset-a"] == 0.0
    assert solved["asset-b"] == pytest.approx(0.25)


def test_m2u5_reference_view_precedence() -> None:
    members = ["asset-above", "asset-left", "asset-right"]
    views = {member: member.removeprefix("asset-") for member in members}
    assert qualify._view_reference(members, views) == "asset-above"
    assert qualify._view_reference(members[1:], views) == "asset-left"


def test_m2u5_reference_tie_breaks_on_lowest_asset_id() -> None:
    members = ["asset-z", "asset-a"]
    views = dict.fromkeys(members, "above")
    assert qualify._view_reference(members, views) == "asset-a"


def test_m2u5_reference_row_is_exact_zero() -> None:
    camera_rows, _, _ = _partial_surface()
    reference = next(row for row in camera_rows if row["is_reference"] == "1")
    assert reference["offset_s"] == "0.000000000"
    assert float(reference["offset_s"]) == 0.0


def test_m2u5_closed_triangle_distributes_closure_by_least_squares() -> None:
    directed = qualify._directed_edges(
        [
            _pair("asset-a", "asset-b", 1.0),
            _pair("asset-b", "asset-c", 1.0),
            _pair("asset-a", "asset-c", 3.0),
        ]
    )
    solved = qualify._solve_offsets(
        set("abc"),
        {(first[-1], second[-1]): value for (first, second), value in directed.items()},
        "a",
    )
    assert solved == pytest.approx({"a": 0.0, "b": 4.0 / 3.0, "c": 8.0 / 3.0})


def test_m2u5_visual_only_edge_cannot_make_a_camera_reachable() -> None:
    assert (
        qualify._directed_edges(
            [_pair("asset-a", "asset-b", 0.1, status=qualify.PAIR_VISUAL_ONLY, visual=0.1)]
        )
        == {}
    )


def test_m2u5_contradicted_edge_cannot_make_a_camera_reachable() -> None:
    assert (
        qualify._directed_edges(
            [_pair("asset-a", "asset-b", 0.1, status=qualify.PAIR_CONTRADICTED, visual=0.2)]
        )
        == {}
    )


def test_m2u5_solver_ignores_pair_confidence() -> None:
    low = qualify._directed_edges([_pair("asset-a", "asset-b", 0.2, peak_rms=0.1)])
    high = qualify._directed_edges([_pair("asset-a", "asset-b", 0.2, peak_rms=100.0)])
    assert low == high
    assert qualify._solve_offsets({"asset-a", "asset-b"}, low, "asset-a") == pytest.approx(
        qualify._solve_offsets({"asset-a", "asset-b"}, high, "asset-a")
    )


def test_m2u5_offset_span_recomputes_from_published_camera_offsets() -> None:
    event_id = "event-triangle"
    members = ["asset-a", "asset-b", "asset-c"]
    views = {"asset-a": "above", "asset-b": "left", "asset-c": "right"}
    names = {member: f"cam-{views[member]}" for member in members}
    directed = qualify._directed_edges(
        [
            _pair("asset-a", "asset-b", 0.1),
            _pair("asset-b", "asset-c", 0.2),
            _pair("asset-a", "asset-c", 0.31),
        ]
    )
    event = _event(event_id, members, views)
    cameras = qualify._camera_rows(
        [event], {event_id: members}, names, views, directed, sync_measured=True
    )
    rows = qualify._event_rows(
        [event],
        {event_id: members},
        camera_rows=cameras,
        directed=directed,
        sync_measured=True,
    )
    published = [float(row["offset_s"]) for row in cameras if row["offset_s"]]
    assert rows[0]["offset_span_s"] == f"{max(published) - min(published):.9f}"
    assert rows[0]["offset_span_s"] == "0.306666667"


def test_m2u5_unreachable_camera_is_present_not_absent() -> None:
    camera_rows, _, _ = _partial_surface()
    assert len(camera_rows) == 3
    assert {row["asset_id"] for row in camera_rows} == {"asset-a", "asset-b", "asset-c"}


def test_m2u5_unreachable_offset_is_empty_not_zero() -> None:
    camera_rows, _, _ = _partial_surface()
    unreachable = next(row for row in camera_rows if row["asset_id"] == "asset-c")
    assert unreachable["offset_status"] == qualify.OFFSET_UNREACHABLE
    assert unreachable["offset_s"] == ""


def test_m2u5_partial_reference_component_does_not_mean_graph_connected() -> None:
    _, event_rows, _ = _partial_surface()
    assert event_rows[0]["graph_connected"] == "0"
    assert event_rows[0]["sync_status"] == qualify.SYNC_UNCONNECTED


def test_m2u5_camera_schema_moves_generator_to_v4() -> None:
    assert qualify.GENERATOR_VERSION == "v4"


def test_m2u5_camera_table_joins_closed_filename_and_generation_sets() -> None:
    assert qualify.CSV_FILENAMES == (
        qualify.ASSETS_QC_FILENAME,
        qualify.PAIRS_QC_FILENAME,
        qualify.CAMERAS_QC_FILENAME,
        qualify.EVENTS_QC_FILENAME,
    )
    assert set(qualify.CSV_FILENAMES) <= set(qualify.GENERATION_KEYS)


def test_m2u5_camera_alphabets_reject_a_trailing_newline() -> None:
    row = {
        "event_id": "event-a",
        "asset_id": "asset-a",
        "offset_s": "0.000000000",
        "offset_status": qualify.OFFSET_REFERENCE,
        "is_reference": "1",
        "reference_camera": "cam-above\n",
    }
    with pytest.raises(qualify.QualifyError, match="does not match") as raised:
        qualify._assert_cell_alphabets(
            [row],
            qualify.CAMERA_CELL_ALPHABETS,
            qualify.CAMERAS_QC_FILENAME,
            ("event_id", "asset_id"),
        )
    assert raised.value.reason == "cell_alphabet"


def test_m2u5_render_manifest_can_never_write_nonzero_sync_offset() -> None:
    camera = sessions.Camera(
        name="cam-above",
        view="above",
        asset_id="asset-a",
        link_name="cam-above.mov",
        source_relative="source.mov",
        content_sha256="0" * 64,
    )
    event = sessions.Event(
        event_id="event-a",
        capture_id="capture-a",
        subject_ordinal=1,
        task="task",
        side="l",
        run_index=1,
        take_resolution=sessions.TAKE_FAMILY,
        view_conflict=0,
        grammar_version="v1",
        cameras=(camera,),
    )
    manifest = json.loads(sessions.render_manifest(event))
    assert [entry["sync_offset"] for entry in manifest["cameras"]] == [0]
    assert type(manifest["cameras"][0]["sync_offset"]) is int


def test_m2u5_unconnected_event_keeps_the_reference_component() -> None:
    camera_rows, _, _ = _partial_surface()
    by_asset = {row["asset_id"]: row for row in camera_rows}
    assert by_asset["asset-a"]["offset_status"] == qualify.OFFSET_REFERENCE
    assert by_asset["asset-a"]["offset_s"] == "0.000000000"
    assert by_asset["asset-b"]["offset_status"] == qualify.OFFSET_SOLVED
    assert by_asset["asset-b"]["offset_s"] == "0.100000000"
    assert by_asset["asset-c"]["offset_status"] == qualify.OFFSET_UNREACHABLE


def test_m2u5_nonreference_component_is_not_independently_gauge_fixed() -> None:
    camera_rows, _, _ = _partial_surface()
    outside = next(row for row in camera_rows if row["asset_id"] == "asset-c")
    assert outside["offset_status"] == qualify.OFFSET_UNREACHABLE
    assert outside["offset_s"] == ""


def test_m2u5_every_camera_row_names_the_same_event_reference() -> None:
    camera_rows, _, _ = _partial_surface()
    assert {row["reference_camera"] for row in camera_rows} == {"cam-above"}
    assert sum(row["is_reference"] == "1" for row in camera_rows) == 1


def test_m2u5_corroboration_never_moves_the_audio_offset() -> None:
    directed = qualify._directed_edges(
        [
            _pair(
                "asset-a",
                "asset-b",
                0.1,
                status=qualify.PAIR_OK_CORROBORATED,
                visual=0.12,
            )
        ]
    )
    assert directed[("asset-a", "asset-b")] == 0.1
    assert qualify._solve_offsets({"asset-a", "asset-b"}, directed, "asset-a")["asset-b"] == 0.1


def test_m2u5_closure_remains_nonzero_after_least_squares() -> None:
    event_id = "event-triangle"
    members = ["asset-a", "asset-b", "asset-c"]
    views = {"asset-a": "above", "asset-b": "left", "asset-c": "right"}
    names = {member: f"cam-{views[member]}" for member in members}
    directed = qualify._directed_edges(
        [
            _pair("asset-a", "asset-b", 0.1),
            _pair("asset-b", "asset-c", 0.2),
            _pair("asset-a", "asset-c", 0.31),
        ]
    )
    event = _event(event_id, members, views)
    cameras = qualify._camera_rows(
        [event], {event_id: members}, names, views, directed, sync_measured=True
    )
    rows = qualify._event_rows(
        [event],
        {event_id: members},
        camera_rows=cameras,
        directed=directed,
        sync_measured=True,
    )
    assert rows[0]["closure_residual_s"] == "0.010000000"


def test_m2u5_camera_name_and_view_remain_distinct_session_fields() -> None:
    event_id = "event-a"
    members = ["asset-a"]
    rows = qualify._camera_rows(
        [_event(event_id, members, {"asset-a": "above"})],
        {event_id: members},
        {"asset-a": "cam-optic"},
        {"asset-a": "above"},
        {},
        sync_measured=True,
    )
    assert rows[0]["camera_name"] == "cam-optic"
    assert rows[0]["view"] == "above"


def test_m2u5_determinism_harness_hashes_cameras_qc() -> None:
    namespace = runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "scripts/check_qualify_determinism.py")
    )
    assert namespace["ARTIFACTS"] == (
        "assets_qc.csv",
        "pairs_qc.csv",
        "events_qc.csv",
        "cameras_qc.csv",
        "qualification.json",
    )


def test_m2u5_census_is_379_355_24_with_173_connected() -> None:
    events: list[dict[str, str]] = []
    members_by_event: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    views: dict[str, str] = {}
    directed: dict[tuple[str, str], float] = {}

    def add_event(n_cameras: int, edges: tuple[tuple[int, int], ...]) -> None:
        event_id = f"event-{len(events):03d}"
        members = [f"asset-{len(events):03d}-{index}" for index in range(n_cameras)]
        event_views = ("above", "left", "right")[:n_cameras]
        for member, view in zip(members, event_views, strict=True):
            names[member] = f"cam-{view}"
            views[member] = view
        events.append(_event(event_id, members, views))
        members_by_event[event_id] = members
        for index, (first, second) in enumerate(edges, start=1):
            value = index / 10
            directed[(members[first], members[second])] = value
            directed[(members[second], members[first])] = -value

    for _ in range(58):
        add_event(1, ())
    for _ in range(74):
        add_event(2, ((0, 1),))
    for _ in range(41):
        add_event(3, ((0, 1), (1, 2)))
    for _ in range(10):
        add_event(2, ())
    for _ in range(6):
        add_event(3, ((0, 1),))
    for _ in range(4):
        add_event(3, ((1, 2),))

    cameras = qualify._camera_rows(
        events, members_by_event, names, views, directed, sync_measured=True
    )
    event_rows = qualify._event_rows(
        events,
        members_by_event,
        camera_rows=cameras,
        directed=directed,
        sync_measured=True,
    )
    statuses = Counter(row["offset_status"] for row in cameras)
    assert len(events) == 193
    assert len(cameras) == 379
    assert sum(bool(row["offset_s"]) for row in cameras) == 355
    assert statuses[qualify.OFFSET_UNREACHABLE] == 24
    assert sum(row["graph_connected"] == "1" for row in event_rows) == 173
