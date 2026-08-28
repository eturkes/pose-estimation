"""Publish the capture-qualification evidence set.

Third artifact-publishing tool, after ``inventory`` and ``sessions``, and it
inherits their publication contract whole: whole-tree swap, per-file digests, a
self-describing marker, and a ``validate_generation`` every consumer calls
before reading a row.

The registry and the session tree are read-only inputs.  This tool never walks
the corpus directory: every asset path comes from the registry's canonical
``source_path`` column, so an asset the registry does not know cannot enter the
evidence set through a directory listing.

Measurement provenance is a published column rather than an assumption.  The
decode clock in ``video_io`` substitutes ``frame_index / fps`` whenever cv2's
own time is absent, duplicate or regressing, and its callers cannot tell a
measured time from that surrogate.  This tool must not reproduce that
conflation, so every timing cell carries the source that produced it and an
unmeasured axis publishes an empty cell rather than a plausible number.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import os
import pathlib
import re
import shutil
import statistics
import struct
import sys
from fractions import Fraction
from typing import Any, BinaryIO

import av

from . import inventory, sessions

GENERATOR_VERSION = "v1"

ASSETS_QC_FILENAME = "assets_qc.csv"
PAIRS_QC_FILENAME = "pairs_qc.csv"
EVENTS_QC_FILENAME = "events_qc.csv"
QUALIFICATION_FILENAME = "qualification.json"

CSV_FILENAMES: tuple[str, ...] = (ASSETS_QC_FILENAME, PAIRS_QC_FILENAME, EVENTS_QC_FILENAME)

# The marker digests the census that carries it, so the key set is closed for
# the same reason sessions' is: an added or renamed key means a different
# writer, and no digest inside the document catches that.
GENERATION_KEYS: tuple[str, ...] = (
    ASSETS_QC_FILENAME,
    PAIRS_QC_FILENAME,
    EVENTS_QC_FILENAME,
    "census",
    "tree",
    "inventory",
    "sessions",
    "generator_version",
)

ASSETS_QC_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "capture_id",
    "view",
    "task",
    "side",
    "subject_ordinal",
    "device_config",
    "codec",
    "decode_status",
    "pts_source",
    "frames_source",
    "frames_decoded",
    "frames_reported",
    "pts_dt_median_s",
    "pts_dt_p95_s",
    "pts_dt_max_s",
    "pts_monotonic",
    "orientation_values",
    "orientation_changes",
    # Two statistics, never one: the sweep behind rulings R2 showed a single
    # residual figure tracking whatever threshold judged it, so the published
    # quantity is the drift itself, at two quantiles.
    "rigidity_drift_median_px",
    "rigidity_drift_p95_px",
    "rigidity_valid_fraction",
    "rigidity_flag",
    "detect_rate",
    "detect_conf_median",
    "subject_px_height_median",
    "scale_ref_class",
    "scale_ref_conf",
    "qc_flags",
)

PAIRS_QC_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "asset_a",
    "asset_b",
    "view_a",
    "view_b",
    "offset_s",
    "confidence",
    "peak_ratio",
    "status",
    "drift_ppm",
    "drift_se",
    "overlap_s",
    "dur_a",
    "dur_b",
    "same_device_config",
    "same_audio_rate",
)

EVENTS_QC_COLUMNS: tuple[str, ...] = (
    "event_id",
    "capture_id",
    "n_cameras",
    "views",
    "graph_connected",
    "closure_residual_s",
    "offset_span_s",
    "sync_qualified",
    "geom_qualified",
    "qualified",
    "reason",
)

# The registry's columns this tool reads, plus reported_frame_count: the header
# claim P11 compares a measurement against belongs to the registry generation,
# not to a second read of the same file.
ASSET_INPUT_COLUMNS: tuple[str, ...] = (
    *sessions.ASSET_INPUT_COLUMNS,
    "reported_frame_count",
)

EVENT_INPUT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "capture_id",
    "n_cameras",
    "run_index",
    "take_resolution",
)

# Populated cells only.  An unmeasured axis publishes "" precisely so that no
# alphabet has to admit a sentinel that could be mistaken for a measurement.
INTEGER_CELL = re.compile(r"[0-9]+")
DECIMAL_CELL = re.compile(r"-?[0-9]+\.[0-9]+")
FLAG_CELL = re.compile(r"[a-z0-9_]+(\|[a-z0-9_]+)*")
# The separator is part of the value: device_config is spelled "model/software".
DEVICE_CONFIG_CELL = re.compile(r"[A-Za-z0-9 ()._-]+(/[A-Za-z0-9 ()._-]+)?")
BOOLEAN_CELL = re.compile(r"[01]")
CODE_LIST_CELL = re.compile(r"[0-9]+(\|[0-9]+)*")

# Every generated cell whose spelling this tool owns.  Registry-derived identity
# columns are absent: the registry generation already validated them, and
# re-deciding their alphabet here would let this tool disagree with its input.
ASSET_CELL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "subject_ordinal": INTEGER_CELL,
    "device_config": DEVICE_CONFIG_CELL,
    "frames_decoded": INTEGER_CELL,
    "frames_reported": INTEGER_CELL,
    "pts_dt_median_s": DECIMAL_CELL,
    "pts_dt_p95_s": DECIMAL_CELL,
    "pts_dt_max_s": DECIMAL_CELL,
    "pts_monotonic": BOOLEAN_CELL,
    "orientation_values": CODE_LIST_CELL,
    "orientation_changes": INTEGER_CELL,
    "qc_flags": FLAG_CELL,
}

DECODE_OK = "ok"
DECODE_OPEN_FAILED = "open_failed"
DECODE_NO_VIDEO = "no_video_stream"
DECODE_NO_PTS = "no_pts"

PTS_CONTAINER = "container_pts_x_time_base"
FRAMES_DEMUXED = "demuxed_packet_count"

UNMEASURED = ""


class QualifyError(Exception):
    """A qualification set could not be published or could not be trusted."""

    def __init__(self, message: str, *, reason: str = "qualify_error") -> None:
        super().__init__(message)
        self.reason = reason


@dataclasses.dataclass(frozen=True)
class AssetRef:
    """One canonical registry row, with its corpus path still unresolved."""

    asset_id: str
    capture_id: str
    view: str
    task: str
    side: str
    subject_ordinal: str
    source_relative: str
    reported_frame_count: int | None


@dataclasses.dataclass(frozen=True)
class DecodeFacts:
    """Demux-derived timing testimony for one asset.

    Every field is measured through ``PTS x time_base``.  No field is ever
    synthesized from a frame ordinal and a nominal rate, which is the whole
    reason ``pts_source`` is published beside the numbers.

    The header's own frame count is deliberately absent: it belongs to the
    registry, which already published it, and re-reading it here would compare
    two fresh reads of the same file instead of comparing this measurement
    against the claim the registry generation actually carries.
    """

    status: str
    codec: str
    device_config: str
    frames_demuxed: int | None
    dt_median_s: float | None
    dt_p95_s: float | None
    dt_max_s: float | None
    monotonic: bool | None


@dataclasses.dataclass(frozen=True)
class OrientationFacts:
    """One asset's device-orientation track.

    ``present`` separates the two states an empty cell cannot: a track that was
    read and a track that does not exist.  ``values`` holds the distinct codes
    in ascending order, so a mid-clip rotation is visible as more than one.
    """

    present: bool
    values: tuple[int, ...]
    changes: int | None


def _count_mismatch(asset: AssetRef, facts: DecodeFacts) -> bool:
    """True when the registry's header count and the demuxed count disagree.

    A mismatch is published as a flag and never repaired: the header is the
    demuxer's claim, and silently trusting either number would erase the
    disagreement that makes the asset worth flagging.
    """
    return (
        facts.frames_demuxed is not None
        and asset.reported_frame_count is not None
        and asset.reported_frame_count > 0
        and facts.frames_demuxed != asset.reported_frame_count
    )


def _parse_int(cell: str) -> int | None:
    """Return a registry integer cell, or None when it carries no number."""
    try:
        return int(cell)
    except (TypeError, ValueError):
        return None


def _seconds(ticks: int, time_base: Fraction) -> float:
    return float(ticks * time_base)


def _decimal(value: float | None) -> str:
    """Format one seconds-valued cell, or leave it explicitly unmeasured."""
    if value is None:
        return UNMEASURED
    return f"{value:.9f}"


def _integer(value: int | None) -> str:
    if value is None:
        return UNMEASURED
    return str(value)


def _boolean(value: bool | None) -> str:
    if value is None:
        return UNMEASURED
    return "1" if value else "0"


def _read_table(path: pathlib.Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Read one published CSV, proving its header before any row is trusted.

    Deliberately a local copy of the reader in ``sessions``: promoting that one
    would widen a surface M2.2 froze behind its own predicate tests, and the
    duplicated cost is a dozen lines.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise QualifyError(f"The input table {path.name} is missing or cannot be read.") from error
    reader = csv.DictReader(text.splitlines(), lineterminator="\n")
    header = tuple(reader.fieldnames or ())
    missing = [name for name in columns if name not in header]
    if missing:
        raise QualifyError(
            f"The input table {path.name} is not this schema: it lacks {', '.join(missing)}."
        )
    return [dict(row) for row in reader]


def load_assets(inventory_dir: pathlib.Path) -> list[AssetRef]:
    """Return the canonical asset rows, in registry order.

    Held-out dispositions never enter the evidence set.  A quarantined stem is
    excluded here rather than filtered downstream, so no later axis can admit
    one by forgetting to ask.
    """
    rows = _read_table(inventory_dir / inventory.ASSETS_FILENAME, ASSET_INPUT_COLUMNS)
    assets: list[AssetRef] = []
    for row in rows:
        if row["disposition"] != inventory.CANONICAL:
            continue
        assets.append(
            AssetRef(
                asset_id=row["asset_id"],
                capture_id=row["capture_id"],
                view=row["view"],
                task=row["task"],
                reported_frame_count=_parse_int(row["reported_frame_count"]),
                side=row["side"],
                subject_ordinal=row["subject_ordinal"],
                source_relative=sessions.decode_source_path(row["source_path"]),
            )
        )
    return assets


def load_events(sessions_dir: pathlib.Path) -> list[dict[str, str]]:
    return _read_table(sessions_dir / sessions.EVENTS_FILENAME, EVENT_INPUT_COLUMNS)


def _device_config(container: av.container.InputContainer) -> str:
    """Return ``model/software``, the pair that identifies a capture era.

    The corpus was recorded by two iPad models across four (model, OS) pairs,
    and the tablets were swapped between positions partway through.  Neither
    field alone separates the eras, so both are published as one cell.
    """
    metadata = dict(container.metadata)
    model = metadata.get("com.apple.quicktime.model", "").strip()
    software = metadata.get("com.apple.quicktime.software", "").strip()
    if not model and not software:
        return UNMEASURED
    return f"{model}/{software}".strip("/")


def _atoms(stream: BinaryIO, end: int) -> list[tuple[bytes, int, int]]:
    """Return ``(kind, body_start, atom_end)`` for each atom in ``[tell, end)``.

    Size 1 means the real 64-bit size follows the header; size 0 means the atom
    runs to the end of its parent.  A size that would leave its own header or
    overrun the parent stops the walk rather than seeking to a computed offset,
    because a truncated file must yield fewer atoms and never a wild read.
    """
    atoms: list[tuple[bytes, int, int]] = []
    while stream.tell() + 8 <= end:
        start = stream.tell()
        header = stream.read(8)
        if len(header) < 8:
            break
        size, kind = struct.unpack(">I4s", header)
        body = start + 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                break
            (size,) = struct.unpack(">Q", extended)
            body = start + 16
        elif size == 0:
            size = end - start
        if size < body - start or start + size > end:
            break
        atoms.append((kind, body, start + size))
        stream.seek(start + size)
    return atoms


def _declared_keys(payload: bytes) -> list[str]:
    """Return the ``keyd`` key names declared in one ``stsd`` payload, in order.

    Position is the identity: a timed-metadata sample names its key by the
    1-based index of the declaration, never by the name itself.
    """
    keys: list[str] = []
    offset = payload.find(b"keyd")
    while offset >= 0:
        if offset >= 4:
            (size,) = struct.unpack(">I", payload[offset - 4 : offset])
            if 12 <= size <= len(payload) - offset + 4:
                keys.append(payload[offset + 8 : offset - 4 + size].decode("utf-8", "replace"))
        offset = payload.find(b"keyd", offset + 4)
    return keys


def _metadata_key_maps(path: pathlib.Path) -> list[dict[int, str]]:
    """Return one key-id to key-name map per timed-metadata track.

    PyAV exposes the packets of a ``mebx`` track but not its key declarations,
    so the ``moov`` atom is walked directly.  Only tracks whose handler is
    ``meta`` are returned, in track order, which is the order PyAV reports the
    corresponding non-audio non-video streams.
    """
    containers = {b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta"}
    tracks: list[dict[int, str]] = []
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(0)
        moov = next(((b, e) for kind, b, e in _atoms(stream, end) if kind == b"moov"), None)
        if moov is None:
            return tracks
        stream.seek(moov[0])
        for kind, start, stop in _atoms(stream, moov[1]):
            if kind != b"trak":
                continue
            handler: bytes | None = None
            keys: list[str] = []
            stack = [(start, stop)]
            while stack:
                child_start, child_stop = stack.pop()
                stream.seek(child_start)
                for child_kind, body, child_end in _atoms(stream, child_stop):
                    if child_kind in containers:
                        stack.append((body, child_end))
                    elif child_kind == b"hdlr":
                        stream.seek(body)
                        raw = stream.read(min(child_end - body, 24))
                        # 'alis' is the file-alias handler every track carries; the
                        # component subtype that names a metadata track sits at 8:12.
                        if len(raw) >= 12 and raw[8:12] != b"alis":
                            handler = raw[8:12]
                    elif child_kind == b"stsd":
                        stream.seek(body)
                        keys.extend(_declared_keys(stream.read(child_end - body)))
            if handler == b"meta":
                tracks.append({index + 1: key for index, key in enumerate(keys)})
    return tracks


def _sample_entries(payload: bytes) -> list[tuple[int, bytes]]:
    """Split one ``mebx`` packet into its ``(key_id, value)`` entries."""
    entries: list[tuple[int, bytes]] = []
    offset = 0
    while offset + 8 <= len(payload):
        size, key_id = struct.unpack(">II", payload[offset : offset + 8])
        if size < 8 or offset + size > len(payload):
            break
        entries.append((key_id, payload[offset + 8 : offset + size]))
        offset += size
    return entries


def probe_orientation(path: pathlib.Path) -> OrientationFacts:
    """Read the device-orientation track, in presentation order.

    The orientation an asset was shot at is a *track*, not a header constant:
    a tablet rotated mid-recording emits a new sample, and a single rotation
    applied to the whole clip is then wrong for part of it.  Publishing the
    distinct values and the transition count keeps that visible; an asset with
    no such track publishes an empty cell and keeps its unmeasured flag.
    """
    try:
        maps = _metadata_key_maps(path)
        values: list[int] = []
        with av.open(str(path)) as container:
            streams = [s for s in container.streams if s.type not in {"video", "audio"}]
            by_index = {
                stream.index: maps[index] if index < len(maps) else {}
                for index, stream in enumerate(streams)
            }
            if not by_index:
                return OrientationFacts(present=False, values=(), changes=None)
            for packet in container.demux(streams):
                if packet.size == 0:
                    continue
                key_map = by_index.get(packet.stream.index, {})
                for key_id, value in _sample_entries(bytes(packet)):
                    if key_map.get(key_id, "").endswith("video-orientation"):
                        values.append(int.from_bytes(value, "big"))
    except (av.FFmpegError, OSError, ValueError, struct.error):
        return OrientationFacts(present=False, values=(), changes=None)
    if not values:
        return OrientationFacts(present=False, values=(), changes=None)
    changes = sum(left != right for left, right in itertools.pairwise(values))
    return OrientationFacts(present=True, values=tuple(sorted(set(values))), changes=changes)


def probe_decode(path: pathlib.Path) -> DecodeFacts:
    """Measure one asset's presentation timebase by demuxing, never decoding.

    Demux is enough for every timing claim this tool makes and costs no pixel
    work.  Presentation timestamps are sorted before differencing because an
    h264 stream with B-frames demuxes out of presentation order; whether the
    stream needed that sort is itself published, as ``pts_monotonic``.
    """
    empty = DecodeFacts(
        status=DECODE_OPEN_FAILED,
        codec=UNMEASURED,
        device_config=UNMEASURED,
        frames_demuxed=None,
        dt_median_s=None,
        dt_p95_s=None,
        dt_max_s=None,
        monotonic=None,
    )
    try:
        with av.open(str(path)) as container:
            config = _device_config(container)
            if not container.streams.video:
                return dataclasses.replace(empty, status=DECODE_NO_VIDEO, device_config=config)
            stream = container.streams.video[0]
            codec = stream.codec_context.name or UNMEASURED
            time_base = stream.time_base
            if time_base is None:
                return dataclasses.replace(
                    empty, status=DECODE_NO_PTS, codec=codec, device_config=config
                )
            pts = [packet.pts for packet in container.demux(stream) if packet.pts is not None]
    except (av.FFmpegError, OSError, ValueError):
        return empty
    if not pts:
        return dataclasses.replace(
            empty,
            status=DECODE_NO_PTS,
            codec=codec,
            device_config=config,
            frames_demuxed=0,
        )
    monotonic = all(earlier <= later for earlier, later in itertools.pairwise(pts))
    ordered = sorted(pts)
    deltas = [
        _seconds(later - earlier, time_base) for earlier, later in itertools.pairwise(ordered)
    ]
    return DecodeFacts(
        status=DECODE_OK,
        codec=codec,
        device_config=config,
        frames_demuxed=len(pts),
        dt_median_s=statistics.median(deltas) if deltas else None,
        dt_p95_s=_percentile(deltas, 0.95) if deltas else None,
        dt_max_s=max(deltas) if deltas else None,
        monotonic=monotonic,
    )


def _percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile.

    Nearest-rank rather than an interpolating estimator: every published
    percentile is then a value the stream actually exhibited, which is what a
    reader checking one asset against its own timestamps will find.
    """
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def _asset_flags(asset: AssetRef, facts: DecodeFacts, orientation: OrientationFacts) -> list[str]:
    flags: list[str] = []
    if facts.status != DECODE_OK:
        flags.append(facts.status)
    if _count_mismatch(asset, facts):
        flags.append("frame_count_mismatch")
    if facts.monotonic is False:
        flags.append("pts_reordered")
    if not orientation.present:
        flags.append("orientation_absent")
    elif orientation.changes:
        # A per-asset rotation constant is wrong for part of such a clip, so
        # every downstream consumer that rotates whole assets must see this.
        flags.append("orientation_changed")
    # Named per axis, so a consumer can tell "this asset failed the check" from
    # "this check has not run yet".  An empty cell alone cannot say which.
    flags.extend(("rigidity_unmeasured", "detect_unmeasured", "scale_unmeasured"))
    return flags


def _asset_row(
    asset: AssetRef, facts: DecodeFacts, orientation: OrientationFacts
) -> dict[str, str]:
    measured = facts.status == DECODE_OK
    return {
        "asset_id": asset.asset_id,
        "capture_id": asset.capture_id,
        "view": asset.view,
        "task": asset.task,
        "side": asset.side,
        "subject_ordinal": asset.subject_ordinal,
        "device_config": facts.device_config,
        "codec": facts.codec,
        "decode_status": facts.status,
        "pts_source": PTS_CONTAINER if measured else UNMEASURED,
        "frames_source": FRAMES_DEMUXED if measured else UNMEASURED,
        "frames_decoded": _integer(facts.frames_demuxed),
        "frames_reported": _integer(asset.reported_frame_count),
        "pts_dt_median_s": _decimal(facts.dt_median_s),
        "pts_dt_p95_s": _decimal(facts.dt_p95_s),
        "pts_dt_max_s": _decimal(facts.dt_max_s),
        "pts_monotonic": _boolean(facts.monotonic),
        "orientation_values": "|".join(str(value) for value in orientation.values),
        "orientation_changes": _integer(orientation.changes),
        "rigidity_drift_median_px": UNMEASURED,
        "rigidity_drift_p95_px": UNMEASURED,
        "rigidity_valid_fraction": UNMEASURED,
        "rigidity_flag": UNMEASURED,
        "detect_rate": UNMEASURED,
        "detect_conf_median": UNMEASURED,
        "subject_px_height_median": UNMEASURED,
        "scale_ref_class": UNMEASURED,
        "scale_ref_conf": UNMEASURED,
        "qc_flags": "|".join(_asset_flags(asset, facts, orientation)),
    }


def _assert_cell_alphabets(rows: list[dict[str, str]]) -> None:
    """Refuse to publish an asset cell this tool cannot spell.

    ``fullmatch``, never ``match``: ``^...$`` would accept a trailing newline,
    which is exactly how a smuggled cell survives a pattern that looks strict.
    An empty cell always passes, because an unmeasured axis publishes one.
    """
    for row in rows:
        for column, pattern in ASSET_CELL_ALPHABETS.items():
            cell = row[column]
            if cell and not pattern.fullmatch(cell):
                raise QualifyError(
                    f"{ASSETS_QC_FILENAME}: {row['asset_id']}: {column} cell {cell!r} "
                    f"does not match {pattern.pattern}",
                    reason="cell_alphabet",
                )


def _pair_rows(assets: list[AssetRef], facts: dict[str, DecodeFacts]) -> list[dict[str, str]]:
    """Enumerate every unordered within-family asset pair.

    Enumeration is the deliverable here even though no offset is measured yet:
    a pair absent from this table is a pair no estimator was ever asked about,
    and that is a different claim from an estimator abstaining on it.
    """
    by_capture: dict[str, list[AssetRef]] = {}
    for asset in assets:
        by_capture.setdefault(asset.capture_id, []).append(asset)
    rows: list[dict[str, str]] = []
    for capture_id in sorted(by_capture):
        members = sorted(by_capture[capture_id], key=lambda item: item.asset_id)
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                left = facts.get(first.asset_id)
                right = facts.get(second.asset_id)
                same_config = (
                    _boolean(left.device_config == right.device_config)
                    if left is not None
                    and right is not None
                    and left.device_config
                    and right.device_config
                    else UNMEASURED
                )
                rows.append(
                    {
                        "capture_id": capture_id,
                        "asset_a": first.asset_id,
                        "asset_b": second.asset_id,
                        "view_a": first.view,
                        "view_b": second.view,
                        "offset_s": UNMEASURED,
                        "confidence": UNMEASURED,
                        "peak_ratio": UNMEASURED,
                        "status": "unmeasured",
                        "drift_ppm": UNMEASURED,
                        "drift_se": UNMEASURED,
                        "overlap_s": UNMEASURED,
                        "dur_a": UNMEASURED,
                        "dur_b": UNMEASURED,
                        "same_device_config": same_config,
                        "same_audio_rate": UNMEASURED,
                    }
                )
    return rows


def _event_rows(events: list[dict[str, str]], assets: list[AssetRef]) -> list[dict[str, str]]:
    views_by_capture: dict[str, list[str]] = {}
    for asset in assets:
        views_by_capture.setdefault(asset.capture_id, []).append(asset.view)
    rows: list[dict[str, str]] = []
    for event in sorted(events, key=lambda item: item["event_id"]):
        views = sorted(set(views_by_capture.get(event["capture_id"], ())))
        rows.append(
            {
                "event_id": event["event_id"],
                "capture_id": event["capture_id"],
                "n_cameras": event["n_cameras"],
                "views": "|".join(views),
                "graph_connected": UNMEASURED,
                "closure_residual_s": UNMEASURED,
                "offset_span_s": UNMEASURED,
                "sync_qualified": UNMEASURED,
                "geom_qualified": UNMEASURED,
                "qualified": UNMEASURED,
                "reason": "sync_unmeasured|geom_unmeasured",
            }
        )
    return rows


def build_census(
    asset_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Return the redaction-safe aggregate census.

    Counts, distributions and status tallies only.  No asset id, no capture id,
    no view label bound to either, no path and no timestamp: this is the one
    artifact whose numbers may be quoted outside the published tree, so it
    carries nothing that identifies a recording or a subject.
    """
    status_counts: dict[str, int] = {}
    codec_counts: dict[str, int] = {}
    config_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    reordered = 0
    mismatched = 0
    for row in asset_rows:
        status_counts[row["decode_status"]] = status_counts.get(row["decode_status"], 0) + 1
        if row["codec"]:
            codec_counts[row["codec"]] = codec_counts.get(row["codec"], 0) + 1
        if row["device_config"]:
            config_counts[row["device_config"]] = config_counts.get(row["device_config"], 0) + 1
        for flag in filter(None, row["qc_flags"].split("|")):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        if row["pts_monotonic"] == "0":
            reordered += 1
    mismatched = flag_counts.get("frame_count_mismatch", 0)
    medians = [float(row["pts_dt_median_s"]) for row in asset_rows if row["pts_dt_median_s"]]
    return {
        "assets": {
            "rows": len(asset_rows),
            "decode_status": dict(sorted(status_counts.items())),
            "codec": dict(sorted(codec_counts.items())),
            "device_config": dict(sorted(config_counts.items())),
            "pts_reordered": reordered,
            "frame_count_mismatch": mismatched,
            "pts_dt_median_s": _distribution(medians),
        },
        "pairs": {
            "rows": len(pair_rows),
            "status": _tally(row["status"] for row in pair_rows),
        },
        "events": {
            "rows": len(event_rows),
            "n_cameras": _tally(row["n_cameras"] for row in event_rows),
        },
        "qc_flags": dict(sorted(flag_counts.items())),
        "measured_axes": ["timebase", "orientation"],
        "unmeasured_axes": ["rigidity", "detectability", "scale", "sync"],
    }


def _tally(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": float(len(ordered)),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def tree_digest(out_dir: str | os.PathLike[str]) -> str:
    """Digest every published entry except the marker that will carry it."""
    lines: list[str] = []

    def visit(entry: pathlib.Path, label: str) -> None:
        if entry.is_symlink():
            lines.append(f"{label}\tlink\t{os.readlink(entry)}\n")  # noqa: PTH115
        elif entry.is_dir():
            lines.append(f"{label}\tdir\n")
            for child in sorted(entry.iterdir()):
                visit(child, f"{label}/{child.name}")
        else:
            lines.append(f"{label}\tfile\t{hashlib.sha256(entry.read_bytes()).hexdigest()}\n")

    for entry in sorted(pathlib.Path(out_dir).iterdir()):
        if entry.name != QUALIFICATION_FILENAME:
            visit(entry, entry.name)
    return hashlib.sha256("".join(lines).encode("utf-8", "surrogateescape")).hexdigest()


def _digest_bytes(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census_digest(census: dict[str, Any]) -> str:
    """Digest the census exactly as it is published, minus its own marker.

    The marker lives inside the document it certifies, so the digest has to be
    taken over the document without that key.  Rendering through the same
    serializer the file uses keeps the digest a function of published bytes
    rather than of an in-memory object.
    """
    body = {key: value for key, value in census.items() if key != "generation"}
    return hashlib.sha256(inventory.render_json(body).encode("utf-8")).hexdigest()


def _build(
    staging: pathlib.Path,
    asset_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    *,
    upstream_inventory: dict[str, Any],
    upstream_sessions: dict[str, Any],
) -> None:
    staging.mkdir(parents=True)
    tables = (
        (ASSETS_QC_FILENAME, ASSETS_QC_COLUMNS, asset_rows),
        (PAIRS_QC_FILENAME, PAIRS_QC_COLUMNS, pair_rows),
        (EVENTS_QC_FILENAME, EVENTS_QC_COLUMNS, event_rows),
    )
    for name, columns, rows in tables:
        (staging / name).write_text(
            inventory.render_csv(columns, rows), encoding="utf-8", newline=""
        )
    census = build_census(asset_rows, pair_rows, event_rows)
    census["generation"] = {
        ASSETS_QC_FILENAME: _digest_bytes(staging / ASSETS_QC_FILENAME),
        PAIRS_QC_FILENAME: _digest_bytes(staging / PAIRS_QC_FILENAME),
        EVENTS_QC_FILENAME: _digest_bytes(staging / EVENTS_QC_FILENAME),
        "census": census_digest(census),
        # Catches what the per-file digests cannot: a file added to the set.
        "tree": tree_digest(staging),
        "inventory": dict(upstream_inventory),
        "sessions": dict(upstream_sessions),
        "generator_version": GENERATOR_VERSION,
    }
    (staging / QUALIFICATION_FILENAME).write_text(
        inventory.render_json(census), encoding="utf-8", newline=""
    )


def _remove(path: pathlib.Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _is_within(child: str, parent: str) -> bool:
    """True when *child* is *parent* or sits under it.

    Compared on separator-terminated text rather than on prefixes, so a sibling
    named ``qualification-old`` is not read as living inside ``qualification``.
    """
    child = child.rstrip(os.sep)
    parent = parent.rstrip(os.sep)
    return child == parent or child.startswith(parent + os.sep)


def _assert_disjoint(out: pathlib.Path, other: str | os.PathLike[str], label: str) -> None:
    """Refuse an output that overlaps an input, in either direction.

    Publication replaces the whole output tree, so an output containing the
    corpus deletes the recordings, and one inside the registry deletes the rows
    it just read.
    """
    here = os.path.realpath(out)
    there = os.path.realpath(other)
    if _is_within(here, there) or _is_within(there, here):
        raise QualifyError(f"The output directory must sit outside the {label}.")


def _sweep_orphans(out: pathlib.Path) -> None:
    """Remove staging and retiring siblings that no live process owns."""
    for sibling in out.parent.glob(f"{out.name}.*"):
        stage, _, pid = sibling.name[len(out.name) + 1 :].rpartition(".")
        if stage not in ("staging", "retiring") or pid == str(os.getpid()):
            continue
        try:
            os.kill(int(pid), 0)
        except (ValueError, OverflowError, ProcessLookupError):
            _remove(sibling)
        except PermissionError:
            continue


def _assert_owned(out_dir: pathlib.Path) -> None:
    """Refuse a non-empty destination this tool did not publish."""
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise QualifyError("The output path exists and is not a directory.")
    if not any(out_dir.iterdir()):
        return
    refusal = QualifyError(
        "The output directory is not empty and carries no generation marker this tool wrote. "
        "Publishing would delete a directory this tool does not own."
    )
    try:
        marker = json.loads((out_dir / QUALIFICATION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise refusal from error
    if not isinstance(marker, dict) or "generation" not in marker:
        raise refusal
    if (
        not isinstance(marker["generation"], dict)
        or "generator_version" not in marker["generation"]
    ):
        raise refusal


def run(
    inventory_dir: str | os.PathLike[str],
    sessions_dir: str | os.PathLike[str],
    corpus_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Publish the qualification set, replacing any generation this tool owns."""
    inventory_path = pathlib.Path(inventory_dir)
    sessions_path = pathlib.Path(sessions_dir)
    out = pathlib.Path(os.path.realpath(out_dir))
    inventory_census = inventory.validate_generation(inventory_path)
    sessions_census = sessions.validate_generation(sessions_path, inventory_dir=inventory_path)
    for other, label in (
        (inventory_path, "registry directory"),
        (sessions_path, "session tree"),
        (corpus_root, "corpus"),
    ):
        _assert_disjoint(out, other, label)
    _assert_owned(out)

    assets = load_assets(inventory_path)
    events = load_events(sessions_path)
    paths = {
        asset.asset_id: sessions.resolve_source(corpus_root, asset.source_relative)
        for asset in assets
    }
    facts = {asset_id: probe_decode(path) for asset_id, path in paths.items()}
    orientations = {asset_id: probe_orientation(path) for asset_id, path in paths.items()}
    asset_rows = [
        _asset_row(asset, facts[asset.asset_id], orientations[asset.asset_id]) for asset in assets
    ]
    _assert_cell_alphabets(asset_rows)
    pair_rows = _pair_rows(assets, facts)
    event_rows = _event_rows(events, assets)

    staging = out.with_name(f"{out.name}.staging.{os.getpid()}")
    retiring = out.with_name(f"{out.name}.retiring.{os.getpid()}")
    _remove(staging)
    _remove(retiring)
    try:
        _build(
            staging,
            asset_rows,
            pair_rows,
            event_rows,
            upstream_inventory=inventory_census.get("generation", {}),
            upstream_sessions=sessions_census,
        )
        if out.exists():
            out.rename(retiring)
        try:
            staging.rename(out)
        except OSError:
            if retiring.exists() and not out.exists():
                retiring.rename(out)
            raise
        # Swept only once the swap has landed: after a kill between the two
        # renames the sole complete generation sits under a dead pid.
        _sweep_orphans(out)
        _remove(retiring)
    finally:
        _remove(staging)
    return json.loads((out / QUALIFICATION_FILENAME).read_text(encoding="utf-8"))


def validate_generation(
    out_dir: str | os.PathLike[str],
    sessions_dir: str | os.PathLike[str] | None = None,
    inventory_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the generation block of *out_dir*, or raise when it is stale.

    Every consumer calls this before reading a row.  With *sessions_dir* and
    *inventory_dir* it also proves the set was published from the upstream
    generations still on disk, which is the only check that catches an upstream
    rebuilt underneath a set that still looks internally consistent.
    """
    out = pathlib.Path(out_dir)
    try:
        census = json.loads((out / QUALIFICATION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualifyError(
            "The published set is unusable: qualification.json is missing or is not valid JSON."
        ) from error
    if not isinstance(census, dict) or not isinstance(census.get("generation"), dict):
        raise QualifyError("The published set is unusable: qualification.json has no generation.")
    generation = census["generation"]
    if set(generation) != set(GENERATION_KEYS) or generation["generator_version"] != (
        GENERATOR_VERSION
    ):
        raise QualifyError(
            "The published set is unusable: qualification.json is not this generator's document."
        )
    for name in CSV_FILENAMES:
        try:
            digest = _digest_bytes(out / name)
        except OSError as error:
            raise QualifyError(
                f"The published set is unusable: {name} is missing or cannot be read."
            ) from error
        if digest != generation.get(name):
            raise QualifyError(
                f"The published set is inconsistent: {name} is a different generation."
            )
    if census_digest(census) != generation.get("census"):
        raise QualifyError(
            "The published set is inconsistent: qualification.json was edited after publication."
        )
    try:
        current = tree_digest(out)
    except OSError as error:
        raise QualifyError("The published set is unusable: it cannot be walked.") from error
    if current != generation.get("tree"):
        raise QualifyError(
            "The published set is inconsistent: a file was added, removed or changed "
            "after publication."
        )
    if inventory_dir is not None:
        upstream = inventory.validate_generation(pathlib.Path(inventory_dir))
        if upstream.get("generation", {}) != generation.get("inventory"):
            raise QualifyError(
                "The published set is stale: the registry is a different generation."
            )
    if sessions_dir is not None:
        upstream_sessions = sessions.validate_generation(pathlib.Path(sessions_dir))
        if upstream_sessions != generation.get("sessions"):
            raise QualifyError(
                "The published set is stale: the session tree is a different generation."
            )
    return generation


def render_summary(census: dict[str, Any]) -> str:
    """Return the console summary.

    Counts only.  Every identifier this tool handles is patient-adjacent, so
    none of them reaches the console.
    """
    assets = census["assets"]
    lines = [
        f"Assets qualified: {assets['rows']}",
        f"  decode status: {_render_counts(assets['decode_status'])}",
        f"  codec: {_render_counts(assets['codec'])}",
        f"  PTS reordered: {assets['pts_reordered']}",
        f"  frame-count mismatch: {assets['frame_count_mismatch']}",
        f"Pairs enumerated: {census['pairs']['rows']}",
        f"Events: {census['events']['rows']}",
        f"Measured axes: {', '.join(census['measured_axes'])}",
        f"Unmeasured axes: {', '.join(census['unmeasured_axes'])}",
    ]
    return "\n".join(lines)


def _render_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pose-estimation-qualify",
        description="Publish the capture-qualification evidence set.",
    )
    parser.add_argument("--inventory", required=True, help="Directory that holds assets.csv.")
    parser.add_argument("--sessions", required=True, help="Directory that holds events.csv.")
    parser.add_argument("--corpus", required=True, help="Root directory of the recordings.")
    parser.add_argument("--out", required=True, help="Directory to publish the evidence set into.")
    arguments = parser.parse_args(argv)
    try:
        census = run(arguments.inventory, arguments.sessions, arguments.corpus, arguments.out)
    except (QualifyError, sessions.SessionsError, inventory.InventoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(render_summary(census))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
