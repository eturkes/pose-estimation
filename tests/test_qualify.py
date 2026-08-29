"""Acceptance suite for the capture-qualification publisher.

Fixtures reuse ``test_sessions``' registry builder rather than reimplementing
it: that builder is already pinned by M2.2's own predicate tests, so a second
copy would be a second thing to keep true.

Media is generated here rather than sampled from the corpus.  Every decode
predicate needs a file whose timing is known in advance, and the corpus is
patient-adjacent, so a committed test may not read it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
from fractions import Fraction

import av
import numpy as np
import pytest

from pose_estimation import inventory, qualify, sessions
from test_sessions import _Asset, _canonical, _write_registry

FRAME_SIZE = (64, 48)
TICKS_PER_SECOND = 600


def _write_media(path: pathlib.Path, pts_ticks: list[int]) -> None:
    """Encode one clip whose presentation timestamps are exactly *pts_ticks*.

    The ticks are the whole point: a test that asserts on measured intervals
    needs a file whose intervals were chosen, not inherited from an encoder's
    default cadence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = FRAME_SIZE
        stream.pix_fmt = "yuv420p"
        # On the codec context, not the stream: the stream's own time base is
        # rewritten by the muxer, so a value set there never reaches the
        # encoder and every tick is read in the default 1/rate units instead.
        stream.codec_context.time_base = Fraction(1, TICKS_PER_SECOND)
        for index, ticks in enumerate(pts_ticks):
            pixels = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), index * 7 % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = ticks
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _uniform(count: int, step: int = 20) -> list[int]:
    return [index * step for index in range(count)]


def _publish(tmp_path: pathlib.Path, assets: list[_Asset]) -> tuple[pathlib.Path, ...]:
    """Build a registry and session tree, and return the four paths qualify needs."""
    registry = _write_registry(tmp_path, assets)
    sessions.run(registry.root, registry.corpus, registry.out)
    return registry.root, registry.out, registry.corpus, tmp_path / "qualification"


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    import csv

    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def _one_asset(tmp_path: pathlib.Path, pts_ticks: list[int]) -> tuple[pathlib.Path, ...]:
    asset = _canonical(1, "above")
    paths = _publish(tmp_path, [asset])
    _write_media(paths[2] / asset.source_path, pts_ticks)
    return paths


def test_p01_validate_rejects_a_registry_of_a_different_generation(
    tmp_path: pathlib.Path,
) -> None:
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    qualify.validate_generation(out, sessions_dir=sessions_dir, inventory_dir=inventory_dir)

    # Republishing the registry from a changed corpus makes it a different
    # generation while the qualification set still looks internally consistent.
    second = tmp_path / "second"
    second.mkdir()
    other = _write_registry(second, [_canonical(2, "left")])
    with pytest.raises(qualify.QualifyError, match="different generation"):
        qualify.validate_generation(out, inventory_dir=other.root)


def test_p02_an_edited_csv_and_an_edited_census_both_fail(tmp_path: pathlib.Path) -> None:
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)

    table = out / qualify.ASSETS_QC_FILENAME
    original = table.read_bytes()
    table.write_bytes(original + b"\n")
    with pytest.raises(qualify.QualifyError, match="different generation"):
        qualify.validate_generation(out)
    table.write_bytes(original)

    census = json.loads((out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8"))
    census["assets"]["rows"] = 999
    (out / qualify.QUALIFICATION_FILENAME).write_text(
        json.dumps(census, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline=""
    )
    with pytest.raises(qualify.QualifyError, match="edited after publication"):
        qualify.validate_generation(out)


def test_p02_a_file_added_to_the_set_fails(tmp_path: pathlib.Path) -> None:
    """The per-file digests cannot see an extra file; the tree digest must."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    (out / "stowaway.csv").write_text("x", encoding="utf-8")
    with pytest.raises(qualify.QualifyError, match="added, removed or changed"):
        qualify.validate_generation(out)


def test_p03_an_unregistered_corpus_file_never_enters_the_set(tmp_path: pathlib.Path) -> None:
    """Rows come from the registry, so a directory listing cannot add one."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    _write_media(corpus / "synthetic-01" / "1_left_cap_l.MOV", _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    rows = _rows(out / qualify.ASSETS_QC_FILENAME)
    assert len(rows) == 1


def test_p03_a_quarantined_asset_is_never_admitted(tmp_path: pathlib.Path) -> None:
    from pose_estimation import inventory as inventory_module

    quarantined = _Asset(
        source_path="synthetic-01/1_above_cap_l_extra.MOV",
        disposition=inventory_module.QUARANTINED,
        reason_code="unparsed_stem",
    )
    inventory_dir, sessions_dir, corpus, out = _publish(
        tmp_path, [_canonical(1, "above"), quarantined]
    )
    _write_media(corpus / "synthetic-01/1_above_cap_l.MOV", _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    rows = _rows(out / qualify.ASSETS_QC_FILENAME)
    assert [row["asset_id"] for row in rows] == [_canonical(1, "above").asset_id]


@pytest.mark.parametrize("label", ["corpus", "registry directory", "session tree"])
def test_p04_an_output_overlapping_an_input_is_refused(tmp_path: pathlib.Path, label: str) -> None:
    inventory_dir, sessions_dir, corpus, _ = _one_asset(tmp_path, _uniform(30))
    inside = {
        "corpus": corpus / "nested",
        "registry directory": inventory_dir / "nested",
        "session tree": sessions_dir / "nested",
    }[label]
    with pytest.raises(qualify.QualifyError, match="must sit outside"):
        qualify.run(inventory_dir, sessions_dir, corpus, inside)


def test_p04_a_symlinked_output_publishes_to_its_target(tmp_path: pathlib.Path) -> None:
    inventory_dir, sessions_dir, corpus, _ = _one_asset(tmp_path, _uniform(30))
    real = tmp_path / "real-qualification"
    real.mkdir()
    link = tmp_path / "linked-qualification"
    link.symlink_to(real)
    qualify.run(inventory_dir, sessions_dir, corpus, link)
    assert link.is_symlink()
    assert (real / qualify.ASSETS_QC_FILENAME).exists()


def test_p05_orphan_siblings_of_a_dead_publisher_are_swept(tmp_path: pathlib.Path) -> None:
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    # A pid no live process owns: the sweep must collect this, and must run
    # only after the swap, so a kill between the renames cannot lose the set.
    debris = out.with_name(f"{out.name}.staging.999999999")
    debris.mkdir()
    (debris / "half-written.csv").write_text("x", encoding="utf-8")
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    assert not debris.exists()
    qualify.validate_generation(out, sessions_dir=sessions_dir, inventory_dir=inventory_dir)


def test_p07_an_empty_registry_publishes_headers_and_still_validates(
    tmp_path: pathlib.Path,
) -> None:
    inventory_dir, sessions_dir, corpus, out = _publish(tmp_path, [])
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    for name, columns in (
        (qualify.ASSETS_QC_FILENAME, qualify.ASSETS_QC_COLUMNS),
        (qualify.PAIRS_QC_FILENAME, qualify.PAIRS_QC_COLUMNS),
        (qualify.EVENTS_QC_FILENAME, qualify.EVENTS_QC_COLUMNS),
    ):
        text = (out / name).read_text(encoding="utf-8")
        assert text.splitlines()[0] == ",".join(columns)
        assert _rows(out / name) == []
    qualify.validate_generation(out, sessions_dir=sessions_dir, inventory_dir=inventory_dir)


def test_p07_a_short_input_header_fails_rather_than_publishing(tmp_path: pathlib.Path) -> None:
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    table = inventory_dir / "assets.csv"
    table.write_text("asset_id,capture_id\n", encoding="utf-8", newline="")
    with pytest.raises(Exception, match=r"assets\.csv|census|generation"):
        qualify.run(inventory_dir, sessions_dir, corpus, out)
    assert not out.exists()


def test_p08_publication_is_byte_identical_under_a_changed_environment(
    tmp_path: pathlib.Path,
) -> None:
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    first = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())
    }

    second = tmp_path / "second-qualification"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "1",
            "LC_ALL": "C",
            "TZ": "Pacific/Kiritimati",
            "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1] / "src"),
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "pose_estimation.qualify",
            "--inventory",
            str(inventory_dir),
            "--sessions",
            str(sessions_dir),
            "--corpus",
            str(corpus),
            "--out",
            str(second),
        ],
        check=True,
        env=environment,
        capture_output=True,
    )
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(second.iterdir())
    } == first


def test_p09_the_census_carries_no_path_or_identifier(tmp_path: pathlib.Path) -> None:
    asset = _canonical(1, "above")
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    text = (out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8")
    for secret in (asset.source_path, asset.asset_id, asset.capture_id, "synthetic-01", ".MOV"):
        assert secret not in text


def test_p10_non_uniform_pts_yield_non_uniform_intervals(tmp_path: pathlib.Path) -> None:
    """The published interval must track the container, not a nominal rate.

    A clock that divided a frame ordinal by a nominal rate would report one
    constant interval for this file, which is exactly the substitution the
    published ``pts_source`` column exists to rule out.
    """
    ticks = [0, 20, 60, 80, 160, 180]
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, ticks)
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.ASSETS_QC_FILENAME)[0]
    assert row["decode_status"] == qualify.DECODE_OK
    assert row["pts_source"] == qualify.PTS_CONTAINER
    assert float(row["pts_dt_max_s"]) > float(row["pts_dt_median_s"])
    assert float(row["pts_dt_max_s"]) == pytest.approx(80 / TICKS_PER_SECOND)


def test_p10_uniform_pts_yield_one_interval(tmp_path: pathlib.Path) -> None:
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.ASSETS_QC_FILENAME)[0]
    assert float(row["pts_dt_median_s"]) == pytest.approx(20 / TICKS_PER_SECOND)
    assert float(row["pts_dt_max_s"]) == pytest.approx(20 / TICKS_PER_SECOND)


def test_p11_a_frame_count_mismatch_is_flagged_and_never_truncated(
    tmp_path: pathlib.Path,
) -> None:
    """The registry claims 30 frames; this file carries 12."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(12))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.ASSETS_QC_FILENAME)[0]
    assert row["frames_decoded"] == "12"
    assert row["frames_reported"] == "30"
    assert "frame_count_mismatch" in row["qc_flags"].split("|")


def test_p11_a_matching_frame_count_is_not_flagged(tmp_path: pathlib.Path) -> None:
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.ASSETS_QC_FILENAME)[0]
    assert row["frames_decoded"] == row["frames_reported"] == "30"
    assert "frame_count_mismatch" not in row["qc_flags"].split("|")


def test_an_unreadable_asset_is_published_as_a_failure_not_dropped(
    tmp_path: pathlib.Path,
) -> None:
    """The synthetic builder writes text, not media: decode must fail loudly."""
    asset = _canonical(1, "above")
    inventory_dir, sessions_dir, corpus, out = _publish(tmp_path, [asset])
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.ASSETS_QC_FILENAME)[0]
    assert row["decode_status"] == qualify.DECODE_OPEN_FAILED
    assert row["pts_source"] == ""
    assert row["frames_decoded"] == ""
    assert qualify.DECODE_OPEN_FAILED in row["qc_flags"].split("|")


def test_every_unmeasured_axis_publishes_an_empty_cell_and_a_named_flag(
    tmp_path: pathlib.Path,
) -> None:
    """An empty cell alone cannot distinguish "failed" from "not yet run"."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.ASSETS_QC_FILENAME)[0]
    for column in ("rigidity_drift_p95_px", "detect_rate", "scale_ref_class"):
        assert row[column] == ""
    flags = set(row["qc_flags"].split("|"))
    assert {"rigidity_unmeasured", "detect_unmeasured", "scale_unmeasured"} <= flags
    census = json.loads((out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8"))
    assert "sync" in census["unmeasured_axes"]
    # Orientation moved out of that set, so it must no longer claim to be pending.
    assert "orientation" not in census["unmeasured_axes"]
    assert "orientation" in census["measured_axes"]
    assert "orientation_unmeasured" not in flags


def test_pairs_enumerate_every_within_family_combination(tmp_path: pathlib.Path) -> None:
    """A pair absent from the table is a pair nothing was ever asked about."""
    assets = [_canonical(1, view) for view in ("above", "left", "right")]
    inventory_dir, sessions_dir, corpus, out = _publish(tmp_path, assets)
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    rows = _rows(out / qualify.PAIRS_QC_FILENAME)
    assert len(rows) == 3
    assert all(row["status"] == "unmeasured" for row in rows)
    assert all(row["asset_a"] < row["asset_b"] for row in rows)


def _atom(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _keyd(name: bytes) -> bytes:
    """One key declaration: size, 'keyd', a namespace, then the key name."""
    return struct.pack(">I", 12 + len(name)) + b"keyd" + b"mdta" + name


def _metadata_file(path: pathlib.Path, handler: bytes, names: list[bytes]) -> None:
    """Write a QuickTime skeleton carrying one track's key declarations."""
    hdlr = _atom(b"hdlr", b"\x00" * 8 + handler + b"\x00" * 12)
    stsd = _atom(b"stsd", b"\x00" * 8 + b"".join(_keyd(name) for name in names))
    mdia = _atom(b"mdia", hdlr + _atom(b"minf", _atom(b"stbl", stsd)))
    path.write_bytes(_atom(b"ftyp", b"qt  ") + _atom(b"moov", _atom(b"trak", mdia)))


ORIENTATION_KEY = b"com.apple.quicktime.video-orientation"


def test_orientation_key_declarations_are_read_positionally(tmp_path: pathlib.Path) -> None:
    """A sample names its key by declaration index, so order is the identity."""
    path = tmp_path / "meta.mov"
    _metadata_file(path, b"meta", [b"com.apple.quicktime.location", ORIENTATION_KEY])
    assert qualify._metadata_key_maps(path) == [
        {1: "com.apple.quicktime.location", 2: ORIENTATION_KEY.decode()}
    ]


def test_orientation_ignores_a_track_that_is_not_a_metadata_track(
    tmp_path: pathlib.Path,
) -> None:
    """Only a 'meta' handler declares orientation keys; a video track must not."""
    path = tmp_path / "video.mov"
    _metadata_file(path, b"vide", [ORIENTATION_KEY])
    assert qualify._metadata_key_maps(path) == []


def test_orientation_atom_walk_reads_a_64_bit_size(tmp_path: pathlib.Path) -> None:
    """Size 1 means the real size is the 64-bit value that follows the header."""
    body = b"payload"
    extended = struct.pack(">I4sQ", 1, b"free", 16 + len(body)) + body
    path = tmp_path / "big.mov"
    path.write_bytes(extended)
    with path.open("rb") as stream:
        atoms = qualify._atoms(stream, len(extended))
    assert atoms == [(b"free", 16, len(extended))]


def test_orientation_atom_walk_stops_at_an_overrunning_size(tmp_path: pathlib.Path) -> None:
    """A size past the parent stops the walk; it never seeks to a computed offset."""
    good = _atom(b"free", b"ok")
    truncated = struct.pack(">I4s", 4096, b"moov")
    path = tmp_path / "short.mov"
    path.write_bytes(good + truncated)
    with path.open("rb") as stream:
        atoms = qualify._atoms(stream, len(good) + len(truncated))
    assert [kind for kind, _, _ in atoms] == [b"free"]


def test_orientation_sample_entries_stop_at_an_impossible_size() -> None:
    """A size below its own header would not advance, so the split must stop."""
    good = struct.pack(">II", 10, 2) + b"\x00\x06"
    assert qualify._sample_entries(good) == [(2, b"\x00\x06")]
    assert qualify._sample_entries(good + struct.pack(">II", 4, 9)) == [(2, b"\x00\x06")]


def test_orientation_absent_track_is_flagged_and_publishes_empty_cells(
    tmp_path: pathlib.Path,
) -> None:
    """An encoded clip carries no orientation track, and says so."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.ASSETS_QC_FILENAME)[0]
    assert row["orientation_values"] == ""
    assert row["orientation_changes"] == ""
    assert "orientation_absent" in row["qc_flags"].split("|")


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (qualify.OrientationFacts(present=False, values=(), changes=None), "orientation_absent"),
        (qualify.OrientationFacts(present=True, values=(1, 6), changes=3), "orientation_changed"),
    ],
)
def test_orientation_flags_name_the_state(facts: qualify.OrientationFacts, expected: str) -> None:
    """A rotation constant applied to a whole asset is wrong for both states."""
    asset = qualify.AssetRef("a", "c", "above", "t", "s", "1", "x.mov", None)
    decode = qualify.DecodeFacts(qualify.DECODE_OK, "h264", "m/s", 1, None, None, None, True)
    assert expected in qualify._asset_flags(asset, decode, facts, frozenset())


def test_orientation_a_constant_track_is_not_flagged_as_changed() -> None:
    facts = qualify.OrientationFacts(present=True, values=(1,), changes=0)
    asset = qualify.AssetRef("a", "c", "above", "t", "s", "1", "x.mov", None)
    decode = qualify.DecodeFacts(qualify.DECODE_OK, "h264", "m/s", 1, None, None, None, True)
    flags = qualify._asset_flags(asset, decode, facts, frozenset())
    assert "orientation_changed" not in flags
    assert "orientation_absent" not in flags


def test_a_cell_that_breaks_its_alphabet_is_refused_rather_than_published() -> None:
    """fullmatch, not match: '^...$' would admit a trailing newline."""
    row = dict.fromkeys(qualify.ASSETS_QC_COLUMNS, "")
    row["asset_id"] = "a-1"
    row["orientation_values"] = "1|6"
    qualify._assert_cell_alphabets([row])
    row["orientation_values"] = "1|6\n"
    with pytest.raises(qualify.QualifyError) as raised:
        qualify._assert_cell_alphabets([row])
    assert raised.value.reason == "cell_alphabet"


def test_a_device_config_keeps_its_model_software_separator() -> None:
    """The '/' is part of the value, so the alphabet must admit exactly one."""
    row = dict.fromkeys(qualify.ASSETS_QC_COLUMNS, "")
    row["asset_id"] = "a-1"
    row["device_config"] = "iPad (5th generation)/16.6"
    qualify._assert_cell_alphabets([row])
    row["device_config"] = "iPad/16.6/extra"
    with pytest.raises(qualify.QualifyError):
        qualify._assert_cell_alphabets([row])


def _sync_sidecar(
    inventory_dir: pathlib.Path,
    measurements: pathlib.Path,
    offsets: dict[tuple[str, str], float],
    *,
    capture_id: str,
    accepted: set[tuple[str, str]] | None = None,
) -> pathlib.Path:
    """Write a sync axis carrying exactly *offsets*, ascending by pair key.

    Offsets are chosen rather than measured: every event predicate below asserts
    on a solved camera timeline, which needs edges whose arithmetic is known in
    advance.
    """
    from pose_estimation import measure

    rows = []
    for (asset_a, asset_b), offset in sorted(offsets.items()):
        ok = accepted is None or (asset_a, asset_b) in accepted
        rows.append(
            {
                "capture_id": capture_id,
                "asset_a": asset_a,
                "asset_b": asset_b,
                "offset_audio_s": f"{offset:.9f}",
                "peak_rms_audio": "5.000000000",
                "peak_ratio_audio": "2.500000000",
                "status_audio": "ok" if ok else "low_confidence",
                "drift_ppm": "",
                "drift_se": "",
                "offset_visual_s": "",
                "conf_visual": "",
                "peak_corr_visual": "",
                "status_visual": "low_peak_correlation",
                "overlap_s": "4.000000000",
                "dur_a": "5.000000000",
                "dur_b": "5.000000000",
                "same_audio_rate": "1",
            }
        )
    measure.write_axis(
        measurements,
        "sync",
        rows,
        {"fixture": "event-surface"},
        inventory_dir=inventory_dir,
    )
    return measurements


def _three_camera_world(
    tmp_path: pathlib.Path,
) -> tuple[tuple[pathlib.Path, ...], list[str], str]:
    """Publish one three-camera family; return its paths, member ids and key."""
    from pose_estimation import inventory

    assets = [_canonical(1, view) for view in ("above", "left", "right")]
    paths = _publish(tmp_path, assets)
    for asset in assets:
        _write_media(paths[2] / asset.source_path, _uniform(30))
    registry = [
        row for row in _rows(paths[0] / "assets.csv") if row["disposition"] == inventory.CANONICAL
    ]
    return paths, sorted(row["asset_id"] for row in registry), registry[0]["capture_id"]


def test_p19_an_event_is_sync_qualified_when_accepted_pairs_join_its_cameras(
    tmp_path: pathlib.Path,
) -> None:
    (inventory_dir, sessions_dir, corpus, out), members, capture_id = _three_camera_world(tmp_path)
    first, second, third = members
    _sync_sidecar(
        inventory_dir,
        tmp_path / "measurements",
        {(first, second): 0.1, (second, third): 0.2, (first, third): 0.31},
        capture_id=capture_id,
    )
    qualify.run(
        inventory_dir, sessions_dir, corpus, out, measurements_dir=tmp_path / "measurements"
    )
    row = _rows(out / qualify.EVENTS_QC_FILENAME)[0]
    assert row["graph_connected"] == "1"
    assert row["sync_qualified"] == "1"
    # Solved against the lowest id, taking the measured edge over an accumulated
    # path: 0, 0.1, 0.31.  The path would give 0.30, and the 10 ms between the
    # two is exactly what closure_residual_s publishes.
    assert row["offset_span_s"] == "0.310000000"
    # Geometry has not run, so the event is still not qualified overall.
    assert row["qualified"] == ""
    assert row["reason"] == "geom_unmeasured"


def test_p19_an_isolated_camera_leaves_its_event_unqualified(tmp_path: pathlib.Path) -> None:
    """Two of three cameras joined is not a joined event."""
    (inventory_dir, sessions_dir, corpus, out), members, capture_id = _three_camera_world(tmp_path)
    first, second, third = members
    _sync_sidecar(
        inventory_dir,
        tmp_path / "measurements",
        {(first, second): 0.1, (second, third): 0.2, (first, third): 0.31},
        capture_id=capture_id,
        accepted={(first, second)},
    )
    qualify.run(
        inventory_dir, sessions_dir, corpus, out, measurements_dir=tmp_path / "measurements"
    )
    row = _rows(out / qualify.EVENTS_QC_FILENAME)[0]
    assert row["graph_connected"] == "0"
    assert row["sync_qualified"] == "0"
    assert row["offset_span_s"] == ""
    assert row["reason"] == "sync_unqualified|geom_unmeasured"


def test_p16_closure_is_published_for_a_triangle_whose_three_edges_are_accepted(
    tmp_path: pathlib.Path,
) -> None:
    """0.1 + 0.2 - 0.31 is a 10 ms disagreement the artifact must state."""
    (inventory_dir, sessions_dir, corpus, out), members, capture_id = _three_camera_world(tmp_path)
    first, second, third = members
    _sync_sidecar(
        inventory_dir,
        tmp_path / "measurements",
        {(first, second): 0.1, (second, third): 0.2, (first, third): 0.31},
        capture_id=capture_id,
    )
    qualify.run(
        inventory_dir, sessions_dir, corpus, out, measurements_dir=tmp_path / "measurements"
    )
    row = _rows(out / qualify.EVENTS_QC_FILENAME)[0]
    assert row["closure_residual_s"] == "0.010000000"
    census = json.loads((out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8"))
    assert census["events"]["closure_residual_s"]["n"] == 1.0
    assert census["events"]["sync_qualified"] == {"1": 1}


def test_p16_an_unclosed_triangle_publishes_no_closure_rather_than_a_partial_one(
    tmp_path: pathlib.Path,
) -> None:
    """A path joins the event; it does not close it, and closure must say so."""
    (inventory_dir, sessions_dir, corpus, out), members, capture_id = _three_camera_world(tmp_path)
    first, second, third = members
    _sync_sidecar(
        inventory_dir,
        tmp_path / "measurements",
        {(first, second): 0.1, (second, third): 0.2, (first, third): 0.31},
        capture_id=capture_id,
        accepted={(first, second), (second, third)},
    )
    qualify.run(
        inventory_dir, sessions_dir, corpus, out, measurements_dir=tmp_path / "measurements"
    )
    row = _rows(out / qualify.EVENTS_QC_FILENAME)[0]
    assert row["graph_connected"] == "1"
    assert row["closure_residual_s"] == ""
    assert row["offset_span_s"] == "0.300000000"
    census = json.loads((out / qualify.QUALIFICATION_FILENAME).read_text(encoding="utf-8"))
    assert census["events"]["closure_residual_s"] is None


def test_an_events_views_cell_names_its_own_cameras_not_its_families(
    tmp_path: pathlib.Path,
) -> None:
    """A view-conflict family splits into single-camera events.

    Re-deriving ``views`` from the capture family credits each of those events
    with every camera the family holds, which is the one thing this column may
    never say.
    """
    assets = [
        _canonical(1, "above"),
        _canonical(1, "above", repeat=1),
        _canonical(1, "left"),
    ]
    inventory_dir, sessions_dir, corpus, out = _publish(tmp_path, assets)
    for asset in assets:
        _write_media(corpus / asset.source_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    rows = _rows(out / qualify.EVENTS_QC_FILENAME)
    assert len(rows) == 3
    for row in rows:
        assert row["n_cameras"] == "1"
        assert "|" not in row["views"]
    assert sorted(row["views"] for row in rows) == ["above", "above", "left"]


def test_a_flagless_run_leaves_every_event_axis_cell_unmeasured(tmp_path: pathlib.Path) -> None:
    """Without a sidecar there is no sync axis, and the row must not imply one."""
    (inventory_dir, sessions_dir, corpus, out), _, _ = _three_camera_world(tmp_path)
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    row = _rows(out / qualify.EVENTS_QC_FILENAME)[0]
    for column in ("graph_connected", "closure_residual_s", "offset_span_s", "sync_qualified"):
        assert row[column] == ""
    assert row["reason"] == "sync_unmeasured|geom_unmeasured"


# Recomputed only alongside a GENERATOR_VERSION bump.  Pinning the digest is what
# turns "did I change the published schema?" from a judgment into a check.
_SCHEMA_DIGEST = "3b566598b0ba00bc8b4c30727802666f162c80914a3dc6123d43527f4b4f145c"


def test_a_schema_change_must_move_the_generator_version() -> None:
    """A published set is self-describing only if the version tracks the schema.

    `validate_generation` refuses a document whose `generator_version` is not
    this generator's, which is the only mechanism stopping an older tree from
    being read under a newer schema.  That mechanism is silent when a column
    changes and the version does not, so the columns are digested here and the
    digest is recomputed only with the bump.
    """
    payload = "\n".join(
        "|".join(columns)
        for columns in (
            qualify.ASSETS_QC_COLUMNS,
            qualify.PAIRS_QC_COLUMNS,
            qualify.EVENTS_QC_COLUMNS,
        )
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == _SCHEMA_DIGEST, (
        "The published column set changed. Bump qualify.GENERATOR_VERSION, then "
        "recompute _SCHEMA_DIGEST, then republish the corpus and regenerate "
        "tests/qualify_determinism_results.json."
    )


def test_a_set_published_by_an_older_generator_is_refused(tmp_path: pathlib.Path) -> None:
    """The bump is only worth anything if the older document is actually rejected."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(30))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    marker = out / qualify.QUALIFICATION_FILENAME
    census = json.loads(marker.read_text(encoding="utf-8"))
    census["generation"]["generator_version"] = "v1"
    marker.write_text(inventory.render_json(census), encoding="utf-8", newline="")
    with pytest.raises(qualify.QualifyError):
        qualify.validate_generation(out)


def test_a_forged_measurements_mode_cannot_be_added_to_a_flagless_marker(
    tmp_path: pathlib.Path,
) -> None:
    """The marker declares its own mode, so the census digest has to cover it."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(3))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    marker = out / qualify.QUALIFICATION_FILENAME
    document = json.loads(marker.read_text(encoding="utf-8"))
    document["generation"]["measurements"] = "0" * 64
    marker.write_text(inventory.render_json(document), encoding="utf-8", newline="")
    with pytest.raises(qualify.QualifyError):
        qualify.validate_generation(out, sessions_dir, inventory_dir)


def test_the_marker_may_not_make_two_claims(tmp_path: pathlib.Path) -> None:
    """Last-key-wins hides the losing claim from every digest and every reader."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(3))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    marker = out / qualify.QUALIFICATION_FILENAME
    original = marker.read_text(encoding="utf-8")
    marker.write_text('{"assets":{"rows":999},' + original[1:], encoding="utf-8", newline="")
    with pytest.raises(qualify.QualifyError):
        qualify.validate_generation(out, sessions_dir, inventory_dir)


def test_the_marker_may_not_be_a_symlink(tmp_path: pathlib.Path) -> None:
    """tree_digest cannot cover the marker, so the marker's own identity is the check."""
    inventory_dir, sessions_dir, corpus, out = _one_asset(tmp_path, _uniform(3))
    qualify.run(inventory_dir, sessions_dir, corpus, out)
    marker = out / qualify.QUALIFICATION_FILENAME
    elsewhere = tmp_path / "elsewhere.json"
    marker.rename(elsewhere)
    marker.symlink_to(elsewhere)
    with pytest.raises(qualify.QualifyError):
        qualify.validate_generation(out, sessions_dir, inventory_dir)


@pytest.mark.parametrize("kind", ["symlink", "foreign"])
def test_a_marker_this_tool_did_not_write_cannot_license_deleting_a_tree(
    tmp_path: pathlib.Path, kind: str
) -> None:
    """Ownership decides a recursive delete, so it reads the strict predicate.

    A symlinked marker puts the licensing document outside the tree it condemns;
    a foreign one puts a different writer's shape inside it.  Neither is this
    tool's output, and the sentinel proves the refusal happened before the swap.
    """
    inventory_dir, sessions_dir, corpus, published = _one_asset(tmp_path, _uniform(3))
    qualify.run(inventory_dir, sessions_dir, corpus, published)
    target = tmp_path / "foreign-output"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("foreign", encoding="utf-8")
    marker = target / qualify.QUALIFICATION_FILENAME
    if kind == "symlink":
        external = tmp_path / "external-marker.json"
        external.write_bytes((published / qualify.QUALIFICATION_FILENAME).read_bytes())
        marker.symlink_to(external)
    else:
        marker.write_text(
            inventory.render_json({"generation": {"generator_version": "foreign-generator"}}),
            encoding="utf-8",
            newline="",
        )
    with pytest.raises(qualify.QualifyError):
        qualify.run(inventory_dir, sessions_dir, corpus, target)
    assert sentinel.read_text(encoding="utf-8") == "foreign"


@pytest.mark.parametrize(
    ("second_event", "case"),
    [("event-1", "same event"), ("event-2", "cross event")],
)
def test_placement_loader_refuses_a_duplicate_asset(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, second_event: str, case: str
) -> None:
    """P19: one asset is one camera in one event, so a repeat is an input defect.

    Silently keeping both rows inflates the event's camera count, and a graph
    that never connects the phantom camera reads as disconnected -- a false
    negative that looks exactly like a real sync failure.
    """
    rows = [
        {"event_id": "event-1", "asset_id": "asset-a", "placement": sessions.PLACED},
        {"event_id": second_event, "asset_id": "asset-a", "placement": sessions.PLACED},
    ]
    monkeypatch.setattr(qualify, "_read_table", lambda _path, _columns: rows)

    with pytest.raises(qualify.QualifyError, match="placed twice"):
        qualify.load_placements(tmp_path)


def test_placement_loader_admits_an_unplaced_repeat_of_a_placed_asset(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control: only PLACED rows claim a camera."""
    rows = [
        {"event_id": "event-1", "asset_id": "asset-a", "placement": sessions.PLACED},
        {"event_id": "event-2", "asset_id": "asset-a", "placement": "held"},
    ]
    monkeypatch.setattr(qualify, "_read_table", lambda _path, _columns: rows)

    assert qualify.load_placements(tmp_path) == {"event-1": ["asset-a"]}
