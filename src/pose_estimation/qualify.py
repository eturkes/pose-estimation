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
import stat
import statistics
import struct
import sys
from fractions import Fraction
from typing import Any, BinaryIO

import av

from . import inventory, measure, sessions

# v3 publishes the exact audio sample rate and the (model, OS, sample_rate)
# stratum P29 requires; v2 renamed pairs_qc's `confidence` to `peak_rms` (R9),
# added the corroborator's three columns, and filled events_qc from the sync
# axis.  A published set is self-describing only if this moves with the schema:
# validate_generation refuses a document whose generator_version is not this
# one, and that refusal is the whole mechanism by which a v2 tree cannot be read
# as a v3 tree.
# `_SCHEMA_DIGEST` in the suite fails on any column change that skips this bump.
GENERATOR_VERSION = "v3"

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
    # The third component of P29's stratum, read from the container header, so
    # a stratum exists with or without a sidecar.  Exact Hz, never a class: the
    # 44 100/48 000 split tracks the two capture eras, and a boolean says only
    # that two assets differ without saying which population either sits in.
    "audio_rate_hz",
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
    # The audio estimate is the published offset: audio estimates and the
    # corroborator holds a veto, so these three are the accepted measurement.
    # ``peak_rms`` replaces the frozen ``confidence`` because ruling R9 found
    # that quantity divided by its own accept thresholds, which made a
    # published statistic move nine orders of magnitude when a gate alone was
    # re-ruled.  Both columns here are raw instrument readings.
    "offset_s",
    "peak_rms",
    "peak_ratio",
    "status_audio",
    # The corroborator's vote, published so ``status`` is a pure function of
    # columns this table carries: a reader can re-derive every verdict, and a
    # re-ruled fusion policy is checkable against the artifact it changed.
    "offset_visual_s",
    "status_visual",
    "status",
    "drift_ppm",
    "drift_se",
    "overlap_s",
    "dur_a",
    "dur_b",
    # P29's stratum, one cell per side, spelled `model/software/rate_hz`.  The
    # two booleans below stay -- P28 needs the rate stratum visible for the
    # priming cancellation to be falsifiable -- and both are now pure functions
    # of these two cells, so a reader re-derives them instead of trusting them.
    # An unmodelled per-device latency is a constant inside one stratum pair
    # and noise across the corpus, which is why the pair, not the corpus, is
    # the population every offset statistic is quoted over.
    "stratum_a",
    "stratum_b",
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
    "views",
)

# The session tree's asset-to-event ledger.  Membership is per event and the
# capture family cannot supply it: a view-conflict family resolves to several
# single-camera events, so any family-wide member list credits each of them
# with cameras it does not hold.
PLACEMENT_INPUT_COLUMNS: tuple[str, ...] = ("asset_id", "event_id", "placement")

# Populated cells only.  An unmeasured axis publishes "" precisely so that no
# alphabet has to admit a sentinel that could be mistaken for a measurement.
INTEGER_CELL = re.compile(r"[0-9]+")
# No zero and no leading zero: a rate is positive, and an alphabet that admits
# a spelling its producer can never emit checks less than it appears to.
POSITIVE_INTEGER_CELL = re.compile(r"[1-9][0-9]*")
DECIMAL_CELL = re.compile(r"-?[0-9]+\.[0-9]+")
FLAG_CELL = re.compile(r"[a-z0-9_]+(\|[a-z0-9_]+)*")
# The separator is part of the value: device_config is spelled "model/software".
# Interior spaces are real -- "iPad (5th generation)" -- but edge spaces are
# not: _device_config strips every component, so a cell carrying one was never
# written by this generator, and an alphabet laxer than its own producer
# forfeits detection of exactly the edits these patterns exist to catch.
_CONFIG_FIELD = r"[A-Za-z0-9()._-](?:[A-Za-z0-9 ()._-]*[A-Za-z0-9()._-])?"
DEVICE_CONFIG_CELL = re.compile(rf"{_CONFIG_FIELD}(/{_CONFIG_FIELD})?")
# One device_config, then the exact rate.  The field alphabet excludes "/", so
# the last field is the rate and no model string can borrow a separator to
# spell a stratum it does not belong to.  Separator count follows
# device_config's own: a capture era that published a model and no software
# string is one field, and its stratum must stay spellable.
STRATUM_CELL = re.compile(rf"{_CONFIG_FIELD}(/{_CONFIG_FIELD})?/[1-9][0-9]*")
BOOLEAN_CELL = re.compile(r"[01]")
CODE_LIST_CELL = re.compile(r"[0-9]+(\|[0-9]+)*")

# Every generated cell whose spelling this tool owns.  Registry-derived identity
# columns are absent: the registry generation already validated them, and
# re-deciding their alphabet here would let this tool disagree with its input.
ASSET_CELL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "subject_ordinal": INTEGER_CELL,
    "device_config": DEVICE_CONFIG_CELL,
    "audio_rate_hz": POSITIVE_INTEGER_CELL,
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

# The pair cells this tool spells itself.  Every other pair cell arrives from
# the sidecar, which validated it against its own alphabet; re-deciding those
# here would let this tool disagree with the record it ingested.
PAIR_CELL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "stratum_a": STRATUM_CELL,
    "stratum_b": STRATUM_CELL,
    "same_device_config": BOOLEAN_CELL,
    "same_audio_rate": BOOLEAN_CELL,
}

# The sidecar axes this tool ingests, named exactly as `measure.AXES` names them
# so a census axis name and a manifest key can never drift apart.
SIDECAR_ASSET_AXES: tuple[str, ...] = ("detect", "rigidity", "scale")
SIDECAR_AXES: tuple[str, ...] = (*SIDECAR_ASSET_AXES, "sync")
# Axes this tool measures itself, on every run, with no sidecar.
LOCAL_AXES: tuple[str, ...] = ("orientation", "timebase")

# R6's fusion alphabet, closed at five tokens over the pairs a sidecar carries,
# plus the token for a pair no sidecar measured.  Fusion lives here and never in
# the record (R8's G7): the sidecar publishes both estimators unfused, so
# re-ruling this policy re-reads bytes instead of re-decoding the corpus.
PAIR_OK_CORROBORATED = "ok_corroborated"
PAIR_OK_UNCORROBORATED = "ok_uncorroborated"
PAIR_CONTRADICTED = "contradicted"
PAIR_VISUAL_ONLY = "visual_only"
PAIR_NEITHER_ACCEPTED = "neither_accepted"
PAIR_UNMEASURED = "unmeasured"

PAIR_STATUSES: frozenset[str] = frozenset(
    {
        PAIR_OK_CORROBORATED,
        PAIR_OK_UNCORROBORATED,
        PAIR_CONTRADICTED,
        PAIR_VISUAL_ONLY,
        PAIR_NEITHER_ACCEPTED,
        PAIR_UNMEASURED,
    }
)

# A pair qualifies on audio acceptance that the corroborator did not veto.
QUALIFIED_PAIR_STATUSES: frozenset[str] = frozenset({PAIR_OK_CORROBORATED, PAIR_OK_UNCORROBORATED})

# One frame at the corpus's nominal cadence.  Two estimates further apart than
# this describe different events, which is a contradiction rather than noise;
# closer than this, no downstream consumer can act on the difference, because
# alignment is applied in whole frames.
AGREE_TOLERANCE_S = 1.0 / 29.97

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
    # Read from the audio stream's header, so it survives every decode failure
    # below it: a container that opens carries its stratum even when no video
    # timestamp does.
    audio_rate_hz: int | None
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


def load_placements(sessions_dir: pathlib.Path) -> dict[str, list[str]]:
    """Return each event's placed asset ids, ascending."""
    members: dict[str, list[str]] = {}
    # P19 counts an event's cameras by its member ids, so a repeated placement
    # inflates that cardinality and can read a connected event as disconnected.
    # One asset is one camera in one event, which makes both the repeat and the
    # cross-event claim input defects rather than something to deduplicate.
    placed_in: dict[str, str] = {}
    rows = _read_table(sessions_dir / sessions.PLACEMENTS_FILENAME, PLACEMENT_INPUT_COLUMNS)
    for row in rows:
        if row["placement"] != sessions.PLACED:
            continue
        asset_id, event_id = row["asset_id"], row["event_id"]
        if asset_id in placed_in:
            raise QualifyError(
                f"{sessions.PLACEMENTS_FILENAME}: {asset_id} is placed twice, "
                f"in {placed_in[asset_id]} and {event_id}."
            )
        placed_in[asset_id] = event_id
        members.setdefault(event_id, []).append(asset_id)
    return {event_id: sorted(ids) for event_id, ids in members.items()}


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


def _audio_rate(container: av.container.InputContainer) -> int | None:
    """Return the first audio stream's sample rate, the stratum's third field.

    ``streams.audio[0]`` deliberately, because that is the stream the sync
    axis decodes: a rate read from any other stream would label the offsets
    with a rate that never produced one.

    A rate that is absent, non-positive, fractional or outside the sidecar's
    own domain stays unmeasured rather than aborting the run: every other
    unreadable container fact on this row publishes a status and an empty cell,
    and one malformed file must not cost the corpus its evidence set.
    Truncating a fractional rate is the one outcome ruled out -- it would spell
    an exact stratum the file never had.

    The ceiling is read from ``measure.DOMAINS`` rather than restated, because
    both paths publish the same cell: a header rate this generator accepts but
    no sidecar could ever carry would make one stratum writable in one
    measurement mode and refused in the other.
    """
    if not container.streams.audio:
        return None
    rate = container.streams.audio[0].rate
    if rate is None or rate != int(rate) or int(rate) <= 0:
        return None
    low, high = measure.DOMAINS["audio_rate_a"]
    if not low <= rate <= high:
        return None
    return int(rate)


def _stratum(device_config: str, audio_rate_hz: int | None) -> str:
    """Return P29's `(model, OS, sample_rate)` cell, or leave it unmeasured.

    Partial is unmeasured: a stratum missing a component is not a coarser
    stratum, it is a row that cannot be compared with any other, and spelling
    it anyway would pool rows whose latency populations are unknown.
    """
    if not device_config or audio_rate_hz is None:
        return UNMEASURED
    return f"{device_config}/{audio_rate_hz}"


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
        audio_rate_hz=None,
        frames_demuxed=None,
        dt_median_s=None,
        dt_p95_s=None,
        dt_max_s=None,
        monotonic=None,
    )
    try:
        with av.open(str(path)) as container:
            config = _device_config(container)
            rate = _audio_rate(container)
            if not container.streams.video:
                return dataclasses.replace(
                    empty, status=DECODE_NO_VIDEO, device_config=config, audio_rate_hz=rate
                )
            stream = container.streams.video[0]
            codec = stream.codec_context.name or UNMEASURED
            time_base = stream.time_base
            if time_base is None:
                return dataclasses.replace(
                    empty,
                    status=DECODE_NO_PTS,
                    codec=codec,
                    device_config=config,
                    audio_rate_hz=rate,
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
            audio_rate_hz=rate,
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
        audio_rate_hz=rate,
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


def _asset_flags(
    asset: AssetRef,
    facts: DecodeFacts,
    orientation: OrientationFacts,
    measured_axes: frozenset[str],
) -> list[str]:
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
    # The flag tracks THIS asset's row, not the axis's presence: an axis can be
    # produced and still abstain on an asset, and that asset is unmeasured.
    flags.extend(f"{axis}_unmeasured" for axis in SIDECAR_ASSET_AXES if axis not in measured_axes)
    return flags


def _asset_row(
    asset: AssetRef,
    facts: DecodeFacts,
    orientation: OrientationFacts,
    axes: dict[str, dict[tuple[str, ...], dict[str, str]]],
) -> dict[str, str]:
    measured = facts.status == DECODE_OK
    ingested: dict[str, str] = {}
    covered: set[str] = set()
    for axis_name in SIDECAR_ASSET_AXES:
        row = axes.get(axis_name, {}).get((asset.asset_id,))
        if row is None:
            continue
        covered.add(axis_name)
        ingested.update({column: row[column] for column in measure.AXES[axis_name].columns[1:]})
    return {
        "asset_id": asset.asset_id,
        "capture_id": asset.capture_id,
        "view": asset.view,
        "task": asset.task,
        "side": asset.side,
        "subject_ordinal": asset.subject_ordinal,
        "device_config": facts.device_config,
        "audio_rate_hz": _integer(facts.audio_rate_hz),
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
        **ingested,
        "qc_flags": "|".join(_asset_flags(asset, facts, orientation, frozenset(covered))),
    }


def _assert_cell_alphabets(
    rows: list[dict[str, str]],
    alphabets: dict[str, re.Pattern[str]],
    filename: str,
    key: tuple[str, ...],
) -> None:
    """Refuse to publish a cell this tool cannot spell.

    ``fullmatch``, never ``match``: ``^...$`` would accept a trailing newline,
    which is exactly how a smuggled cell survives a pattern that looks strict.
    An empty cell always passes, because an unmeasured axis publishes one.
    """
    for row in rows:
        for column, pattern in alphabets.items():
            cell = row[column]
            if cell and not pattern.fullmatch(cell):
                raise QualifyError(
                    f"{filename}: {' '.join(row[name] for name in key)}: "
                    f"{column} cell {cell!r} does not match {pattern.pattern}",
                    reason="cell_alphabet",
                )


def enumerate_pairs(assets: list[AssetRef]) -> list[tuple[str, AssetRef, AssetRef]]:
    """Enumerate every unordered within-family asset pair, ascending.

    One enumeration serves the published table and the sidecar reconciliation.
    They must agree exactly — a pair keyed the other way round is a key this
    side never looks up, so the axis would read as having abstained on it
    rather than as having disagreed about its identity — and the only way two
    enumerations cannot drift is for there to be one.
    """
    by_capture: dict[str, list[AssetRef]] = {}
    for asset in assets:
        by_capture.setdefault(asset.capture_id, []).append(asset)
    pairs: list[tuple[str, AssetRef, AssetRef]] = []
    for capture_id in sorted(by_capture):
        members = sorted(by_capture[capture_id], key=lambda item: item.asset_id)
        for index, first in enumerate(members):
            pairs.extend((capture_id, first, second) for second in members[index + 1 :])
    return pairs


def fuse_pair(row: dict[str, str]) -> str:
    """Return R6's verdict for one measured pair.

    Audio estimates; the corroborator holds a veto where it spoke and no vote
    where it did not.  "Spoke" is the corroborator clearing **its own** gate,
    so a low-quality visual estimate carries no veto, and a pair both
    instruments accept while disagreeing by more than a frame is refused rather
    than resolved: two independent measurements disagree and neither is
    preferred.  The strict reading — qualify only on agreement — was priced and
    refused, because it leaves 111 of 137 families unrecoverable while the
    gross-error evidence behind it bounds the visual estimator alone.
    """
    audio_ok = row["status_audio"] == "ok"
    visual_ok = row["status_visual"] == "ok"
    if audio_ok and visual_ok:
        delta = abs(float(row["offset_audio_s"]) - float(row["offset_visual_s"]))
        return PAIR_OK_CORROBORATED if delta <= AGREE_TOLERANCE_S else PAIR_CONTRADICTED
    if audio_ok:
        return PAIR_OK_UNCORROBORATED
    if visual_ok:
        return PAIR_VISUAL_ONLY
    return PAIR_NEITHER_ACCEPTED


def _pair_rows(
    assets: list[AssetRef],
    facts: dict[str, DecodeFacts],
    sync: dict[tuple[str, ...], dict[str, str]],
) -> list[dict[str, str]]:
    """Enumerate every within-family pair, carrying the sync axis where it ran.

    Enumeration is the deliverable even where no offset was measured: a pair
    absent from this table is a pair no estimator was ever asked about, and
    that is a different claim from an estimator abstaining on it.
    """
    _assert_sidecar_rates(facts, sync)
    rows: list[dict[str, str]] = []
    for capture_id, first, second in enumerate_pairs(assets):
        left = facts.get(first.asset_id)
        right = facts.get(second.asset_id)
        same_config = (
            _boolean(left.device_config == right.device_config)
            if left is not None and right is not None and left.device_config and right.device_config
            else UNMEASURED
        )
        # Derived here rather than copied from the sidecar, so both booleans
        # come from the same measurement in both publication modes and stay a
        # pure function of the strata beside them.  The sidecar's own rates
        # still have to agree, which is what the assertion above proves.
        rate_left = left.audio_rate_hz if left is not None else None
        rate_right = right.audio_rate_hz if right is not None else None
        same_rate = (
            _boolean(rate_left == rate_right)
            if rate_left is not None and rate_right is not None
            else UNMEASURED
        )
        measured = sync.get((first.asset_id, second.asset_id))
        rows.append(
            {
                "capture_id": capture_id,
                "asset_a": first.asset_id,
                "asset_b": second.asset_id,
                "view_a": first.view,
                "view_b": second.view,
                "offset_s": UNMEASURED,
                "peak_rms": UNMEASURED,
                "peak_ratio": UNMEASURED,
                "status_audio": UNMEASURED,
                "offset_visual_s": UNMEASURED,
                "status_visual": UNMEASURED,
                "status": PAIR_UNMEASURED,
                "drift_ppm": UNMEASURED,
                "drift_se": UNMEASURED,
                "overlap_s": UNMEASURED,
                "dur_a": UNMEASURED,
                "dur_b": UNMEASURED,
                "stratum_a": _stratum(left.device_config if left else "", rate_left),
                "stratum_b": _stratum(right.device_config if right else "", rate_right),
                "same_device_config": same_config,
                "same_audio_rate": same_rate,
            }
        )
        if measured is None:
            continue
        rows[-1].update(
            {
                "offset_s": measured["offset_audio_s"],
                "peak_rms": measured["peak_rms_audio"],
                "peak_ratio": measured["peak_ratio_audio"],
                "status_audio": measured["status_audio"],
                "offset_visual_s": measured["offset_visual_s"],
                "status_visual": measured["status_visual"],
                "status": fuse_pair(measured),
                "drift_ppm": measured["drift_ppm"],
                "drift_se": measured["drift_se"],
                "overlap_s": measured["overlap_s"],
                "dur_a": measured["dur_a"],
                "dur_b": measured["dur_b"],
            }
        )
    return rows


def _assert_sidecar_rates(
    facts: dict[str, DecodeFacts], sync: dict[tuple[str, ...], dict[str, str]]
) -> None:
    """Refuse a sidecar whose decode rate contradicts this run's header read.

    The stratum published here is read from the container header; the offsets
    it labels were produced by the sidecar's own decode.  Both name
    ``streams.audio[0]``, so a disagreement means the two ran against different
    bytes, and the failure it would otherwise cause is silent and total: every
    offset in that stratum is filed under a rate that never produced one, which
    is exactly the structure P29 exists to expose.  Loud, because a relabelled
    population cannot be detected downstream from the artifact alone.

    The asymmetry is deliberate.  A sidecar cell naming a rate is a positive
    claim that the asset carries audio, so a header read that finds none
    contradicts it as squarely as a different number does -- both sides opened
    the same path and read ``streams.audio[0]``.  An empty sidecar cell is an
    abstention about that side's own cache, contradicts nothing, and relabels
    nothing: the stratum published is the header's either way.
    """
    for key, row in sync.items():
        for asset_id, cell in zip(key, (row["audio_rate_a"], row["audio_rate_b"]), strict=True):
            header = facts.get(asset_id)
            if not cell or header is None:
                continue
            if header.audio_rate_hz is None:
                raise QualifyError(
                    f"The sidecar decoded {asset_id} at {cell} Hz and this run's header "
                    f"read finds no usable audio rate.",
                    reason="audio_rate_disagreement",
                )
            if int(cell) != header.audio_rate_hz:
                raise QualifyError(
                    f"The sidecar decoded {asset_id} at {cell} Hz and this run's header "
                    f"read reports {header.audio_rate_hz} Hz.",
                    reason="audio_rate_disagreement",
                )


def _spanning_offsets(
    members: list[str], directed: dict[tuple[str, str], float]
) -> dict[str, float] | None:
    """Solve one offset per member against the lowest id, or None when unjoined.

    Breadth-first over accepted edges alone, so a camera reachable directly
    takes its own measured edge rather than an accumulated path.  Where a
    triangle does not close, which edges carry the solution changes it, so the
    traversal is fixed rather than incidental; ``closure_residual_s`` publishes
    exactly the disagreement between the two routes.
    """
    if not members:
        return None
    solved = {members[0]: 0.0}
    frontier = [members[0]]
    while frontier:
        current = frontier.pop(0)
        for other in members:
            if other in solved or (current, other) not in directed:
                continue
            solved[other] = solved[current] + directed[(current, other)]
            frontier.append(other)
    return solved if len(solved) == len(members) else None


def _event_rows(
    events: list[dict[str, str]],
    members_by_event: dict[str, list[str]],
    pair_rows: list[dict[str, str]],
    *,
    sync_measured: bool,
) -> list[dict[str, str]]:
    """One row per session event, carrying the sync axis where it ran.

    ``views`` is the session tree's own per-event cell rather than a re-derived
    one: the capture family of a view-conflict event holds views that event
    does not, and re-deriving published text is how the two spellings drift.

    Every cell below the identity block stays unmeasured without the sidecar,
    which is what keeps a flagless run byte-identical to every earlier one.
    """
    directed: dict[tuple[str, str], float] = {}
    for row in pair_rows:
        if row["status"] not in QUALIFIED_PAIR_STATUSES:
            continue
        offset = float(row["offset_s"])
        directed[(row["asset_a"], row["asset_b"])] = offset
        directed[(row["asset_b"], row["asset_a"])] = -offset
    rows: list[dict[str, str]] = []
    for event in sorted(events, key=lambda item: item["event_id"]):
        row = {
            "event_id": event["event_id"],
            "capture_id": event["capture_id"],
            "n_cameras": event["n_cameras"],
            "views": event["views"],
            "graph_connected": UNMEASURED,
            "closure_residual_s": UNMEASURED,
            "offset_span_s": UNMEASURED,
            "sync_qualified": UNMEASURED,
            "geom_qualified": UNMEASURED,
            "qualified": UNMEASURED,
            "reason": "sync_unmeasured|geom_unmeasured",
        }
        rows.append(row)
        if not sync_measured:
            continue
        members = members_by_event.get(event["event_id"], [])
        solved = _spanning_offsets(members, directed)
        # P19 quantifies over the event's own cameras, so a one-camera event is
        # connected: it carries no alignment to fail.  Geometry is what refuses
        # a single camera, and geometry answers on its own axis.
        row["graph_connected"] = _boolean(solved is not None)
        row["sync_qualified"] = row["graph_connected"]
        if solved is not None and len(members) > 1:
            row["offset_span_s"] = _decimal(max(solved.values()) - min(solved.values()))
        if len(members) == 3:
            first, second, third = members
            triangle = ((first, second), (second, third), (first, third))
            if all(edge in directed for edge in triangle):
                # A cocycle: acoustic propagation delay cancels identically
                # around the triangle, so this certifies self-consistency and
                # never accuracy (P16).
                row["closure_residual_s"] = _decimal(
                    abs(directed[triangle[0]] + directed[triangle[1]] - directed[triangle[2]])
                )
        blockers = (
            ["geom_unmeasured"] if solved is not None else ["sync_unqualified", "geom_unmeasured"]
        )
        row["reason"] = "|".join(blockers)
    return rows


def build_census(
    asset_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    measured_axes: frozenset[str] = frozenset(),
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
            "sync_strata": _sync_strata(pair_rows),
        },
        "events": {
            "rows": len(event_rows),
            "n_cameras": _tally(row["n_cameras"] for row in event_rows),
            "sync_qualified": _tally(row["sync_qualified"] for row in event_rows),
            # Self-consistency, never accuracy (P16).  It is quotable because it
            # is a distribution over triangles and names none of them.
            "closure_residual_s": _distribution(
                [
                    float(row["closure_residual_s"])
                    for row in event_rows
                    if row["closure_residual_s"]
                ]
            ),
        },
        "qc_flags": dict(sorted(flag_counts.items())),
        # Axis presence, not per-row coverage: an axis produced and empty is
        # measured, because its producer completed and found nothing to say.
        # Whether any single asset carries a value is the qc_flags question.
        "measured_axes": sorted({*LOCAL_AXES, *measured_axes}),
        "unmeasured_axes": sorted(set(SIDECAR_AXES) - measured_axes),
    }


def _sync_strata(pair_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Return the sync statistics grouped by P29's stratum pair.

    The key is the two strata sorted, never the pair's own ``a``/``b`` order:
    ``asset_a`` sorts by id, so the same two configurations would otherwise
    split across two keys and halve every population.  A pair missing either
    stratum is keyed ``unmeasured`` -- a real stratum always carries a ``/``,
    so the token cannot collide with one.

    Quotable outside the tree: counts and a distribution per configuration
    pair, naming no asset, no capture and no view.  This is the aggregate P29
    asks for, because an unmodelled per-device latency is a constant within one
    stratum pair and is invisible in a corpus-wide distribution.
    """
    strata: dict[str, dict[str, Any]] = {}
    offsets: dict[str, list[float]] = {}
    for row in pair_rows:
        left, right = row["stratum_a"], row["stratum_b"]
        key = "|".join(sorted((left, right))) if left and right else "unmeasured"
        entry = strata.setdefault(key, {"pairs": 0, "audio_ok": 0})
        entry["pairs"] += 1
        if row["status_audio"] == "ok" and row["offset_s"]:
            entry["audio_ok"] += 1
            offsets.setdefault(key, []).append(float(row["offset_s"]))
    for key, entry in strata.items():
        entry["offset_s"] = _distribution(offsets.get(key, []))
    return dict(sorted(strata.items()))


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

    The digest lives inside the document it certifies, so it is taken over the
    document without that one self-referential key -- and over everything else,
    the generation block included.  Excluding the whole block instead would
    leave the upstream provenance the consumers trust most as the only claim in
    the set nothing covers.  Rendering through the same serializer the file uses
    keeps the digest a function of published bytes rather than of an in-memory
    object.

    Detection, not authentication: a set carries no key, so an edit that also
    recomputes this digest is indistinguishable from a publication.  What the
    digest rules out is corruption and every edit that stops at the claim.
    """
    body = dict(census)
    if isinstance(body.get("generation"), dict):
        body["generation"] = {
            key: value for key, value in body["generation"].items() if key != "census"
        }
    return hashlib.sha256(inventory.render_json(body).encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError("qualification.json carries a duplicate key.")
    return dict(pairs)


def _is_own_generation(generation: Any) -> bool:
    """Whether this block is one of the two key sets this generator publishes.

    Shape and version only, never a digest: a set whose upstreams have since
    moved is stale but still this tool's to replace, and requiring freshness
    here would strand it behind a manual delete.
    """
    if not isinstance(generation, dict) or generation.get("generator_version") != GENERATOR_VERSION:
        return False
    base = set(GENERATION_KEYS)
    return set(generation) in (base, base | {"measurements"})


def _read_marker(out: pathlib.Path) -> dict[str, Any]:
    """Read the marker as the kind of file this tool would itself have written.

    The marker is the set's trust root and the one entry ``tree_digest`` cannot
    cover, so its own identity is all that stands behind it.  A symlink puts
    that root outside the set it certifies, and through ``_assert_owned`` lets a
    foreign directory license its own deletion.  A duplicate key puts two claims
    in one document, of which ``json.loads`` silently keeps the last.

    Raises ``OSError`` for a missing or non-regular path and ``ValueError`` for
    text that is not a single unambiguous JSON document; both callers render
    those as one refusal.
    """
    path = out / QUALIFICATION_FILENAME
    if not stat.S_ISREG(path.lstat().st_mode):
        raise OSError(f"{QUALIFICATION_FILENAME} is not a regular file.")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def _build(
    staging: pathlib.Path,
    asset_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    *,
    upstream_inventory: dict[str, Any],
    upstream_sessions: dict[str, Any],
    upstream_measurements: str | None = None,
    measured_axes: frozenset[str] = frozenset(),
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
    census = build_census(asset_rows, pair_rows, event_rows, measured_axes)
    census["generation"] = {
        ASSETS_QC_FILENAME: _digest_bytes(staging / ASSETS_QC_FILENAME),
        PAIRS_QC_FILENAME: _digest_bytes(staging / PAIRS_QC_FILENAME),
        EVENTS_QC_FILENAME: _digest_bytes(staging / EVENTS_QC_FILENAME),
        # Catches what the per-file digests cannot: a file added to the set.
        "tree": tree_digest(staging),
        "inventory": dict(upstream_inventory),
        "sessions": dict(upstream_sessions),
        "generator_version": GENERATOR_VERSION,
    }
    if upstream_measurements is not None:
        # Present only in the mode that has a third upstream.  An always-present
        # nullable key would change the published bytes for every consumer that
        # never asked for a sidecar, which is what P34 and P08 forbid.
        census["generation"]["measurements"] = upstream_measurements
    # Last, because it digests every other key including this block's own.
    census["generation"]["census"] = census_digest(census)
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
        marker = _read_marker(out_dir)
    except (OSError, ValueError) as error:
        raise refusal from error
    # This generator's shape and version, not merely some tool's marker: the
    # next statement after this one deletes the whole tree.
    if not isinstance(marker, dict) or not _is_own_generation(marker.get("generation")):
        raise refusal


def _ingest(
    measurements_dir: str | os.PathLike[str],
    inventory_path: pathlib.Path,
    assets: list[AssetRef],
) -> tuple[str, dict[str, dict[tuple[str, ...], dict[str, str]]]]:
    """Validate the sidecar and bind every axis key to this registry.

    Runs before one frame is decoded and before the output tree is touched, so
    a sidecar this run cannot trust costs an exit code rather than a discarded
    publication.  ``MeasureError`` is wrapped here rather than raised onward:
    callers of this tool face one error domain, and a sidecar defect is a
    qualification failure from where they stand.
    """
    expected_assets: dict[tuple[str, ...], dict[str, str]] = {
        (asset.asset_id,): {} for asset in assets
    }
    expected_pairs: dict[tuple[str, ...], dict[str, str]] = {
        (first.asset_id, second.asset_id): {"capture_id": capture_id}
        for capture_id, first, second in enumerate_pairs(assets)
    }
    try:
        sidecar = measure.validate(measurements_dir, inventory_dir=inventory_path)
        axes = {
            name: measure.load_axis(sidecar, name)
            for name in SIDECAR_AXES
            if name in sidecar.manifest["axes"]
        }
        for name, rows in axes.items():
            measure.reconcile(rows, expected_pairs if name == "sync" else expected_assets)
    except measure.MeasureError as error:
        raise QualifyError(str(error), reason=error.reason) from error
    return sidecar.manifest["generation"]["manifest"], axes


def run(
    inventory_dir: str | os.PathLike[str],
    sessions_dir: str | os.PathLike[str],
    corpus_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    measurements_dir: str | os.PathLike[str] | None = None,
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
    measurements_digest: str | None = None
    axes: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    if measurements_dir is not None:
        _assert_disjoint(out, measurements_dir, "measurement sidecar")
        measurements_digest, axes = _ingest(measurements_dir, inventory_path, assets)
    paths = {
        asset.asset_id: sessions.resolve_source(corpus_root, asset.source_relative)
        for asset in assets
    }
    facts = {asset_id: probe_decode(path) for asset_id, path in paths.items()}
    orientations = {asset_id: probe_orientation(path) for asset_id, path in paths.items()}
    asset_rows = [
        _asset_row(asset, facts[asset.asset_id], orientations[asset.asset_id], axes)
        for asset in assets
    ]
    _assert_cell_alphabets(asset_rows, ASSET_CELL_ALPHABETS, ASSETS_QC_FILENAME, ("asset_id",))
    pair_rows = _pair_rows(assets, facts, axes.get("sync", {}))
    _assert_cell_alphabets(
        pair_rows, PAIR_CELL_ALPHABETS, PAIRS_QC_FILENAME, ("asset_a", "asset_b")
    )
    event_rows = _event_rows(
        events,
        load_placements(sessions_path),
        pair_rows,
        sync_measured="sync" in axes,
    )

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
            upstream_measurements=measurements_digest,
            measured_axes=frozenset(axes),
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
    return _read_marker(out)


def validate_generation(
    out_dir: str | os.PathLike[str],
    sessions_dir: str | os.PathLike[str] | None = None,
    inventory_dir: str | os.PathLike[str] | None = None,
    measurements_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the generation block of *out_dir*, or raise when it is stale.

    Every consumer calls this before reading a row.  With *sessions_dir* and
    *inventory_dir* it also proves the set was published from the upstream
    generations still on disk, which is the only check that catches an upstream
    rebuilt underneath a set that still looks internally consistent.
    """
    out = pathlib.Path(out_dir)
    try:
        census = _read_marker(out)
    except (OSError, ValueError) as error:
        raise QualifyError(
            "The published set is unusable: qualification.json is missing, is not a regular "
            "file, or is not one unambiguous JSON document."
        ) from error
    if not isinstance(census, dict) or not isinstance(census.get("generation"), dict):
        raise QualifyError("The published set is unusable: qualification.json has no generation.")
    generation = census["generation"]
    # Closure is evaluated per mode: a set published with a sidecar carries one
    # more upstream than one published without.  Both key sets are closed, so a
    # key neither mode writes still fails.  The marker declaring its own mode is
    # sound only because `census_digest` now covers this block: adding the key
    # without recomputing that digest is caught below.
    expected_keys = set(GENERATION_KEYS)
    if "measurements" in generation:
        expected_keys.add("measurements")
    if set(generation) != expected_keys or generation["generator_version"] != (GENERATOR_VERSION):
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
    if measurements_dir is not None:
        if "measurements" not in generation:
            raise QualifyError(
                "The published set was published without a measurement sidecar, "
                "so it cannot be checked against one."
            )
        try:
            sidecar = measure.validate(measurements_dir, inventory_dir=inventory_dir)
        except measure.MeasureError as error:
            raise QualifyError(str(error), reason=error.reason) from error
        if sidecar.manifest["generation"]["manifest"] != generation["measurements"]:
            raise QualifyError(
                "The published set is stale: the measurement sidecar is a different generation."
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
        f"  sync qualified: {_render_counts(census['events']['sync_qualified'])}",
        f"Measured axes: {', '.join(census['measured_axes'])}",
        f"Unmeasured axes: {', '.join(census['unmeasured_axes'])}",
    ]
    return "\n".join(lines)


def _render_counts(counts: dict[str, int]) -> str:
    # A tally over cells that may be unmeasured keys the empty cell, which
    # renders as a nameless "=193" unless the console spells it out.
    return (
        ", ".join(f"{key or 'unmeasured'}={value}" for key, value in sorted(counts.items()))
        or "none"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pose-estimation-qualify",
        description="Publish the capture-qualification evidence set.",
    )
    parser.add_argument("--inventory", required=True, help="Directory that holds assets.csv.")
    parser.add_argument("--sessions", required=True, help="Directory that holds events.csv.")
    parser.add_argument("--corpus", required=True, help="Root directory of the recordings.")
    parser.add_argument("--out", required=True, help="Directory to publish the evidence set into.")
    parser.add_argument(
        "--measurements",
        help="Measurement sidecar directory to ingest. Omit it to publish the axes unmeasured.",
    )
    arguments = parser.parse_args(argv)
    try:
        census = run(
            arguments.inventory,
            arguments.sessions,
            arguments.corpus,
            arguments.out,
            measurements_dir=arguments.measurements,
        )
    except (QualifyError, sessions.SessionsError, inventory.InventoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(render_summary(census))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
