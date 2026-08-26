"""Corpus inventory: canonical capture records plus a header-only census.

One tool answers three questions about a directory of clips.  What is in it?
Which files carry a usable identity?  What did the container headers say?

Every regular file under the corpus root reaches exactly one recorded
disposition — ``canonical``, ``quarantined``, or ``excluded`` — so a file can
never leave the census by being silently skipped.  Absence is not an outcome
here; a row is.

Identity has two levels.  ``asset_id`` names one file and is derived from its
corpus-relative path, and the explicit collision check in ``check_invariants``
is what guarantees it, because a 64-bit digest of a unique path collides only
with negligible probability rather than never.  ``capture_id`` names a
task-side family —
one subject, one task, one side — and is the group key its views share.  It is
not a physical-take key: a family whose ``view_conflict`` is set holds more
than one take, so a consumer that needs a single recording event must reject
or resolve those families first.  The registry this module writes is the
project's single source of family identity, and the identity is the pair
``(grammar_version, capture_id)``, since a grammar change can move membership
while the readable key stays the same.

Nothing here decodes a pixel.  The container facts are the demuxer's claims,
which is why every one of them is named ``reported_*``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import re
import sys

import cv2

from .video_io import PROBE_OPENED, PROBE_SKIPPED, VIDEO_EXTENSIONS, ContainerFacts, probe_container

# FFmpeg writes its own diagnostics straight to the process stderr, outside
# every Python logging control, and some of them quote the source URL.  A
# corpus filename identifies a subject, so the native log is silenced before
# the first capture opens.  An operator who wants it back sets the variable.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

TOOL_VERSION = "v1"
GRAMMAR_VERSION = "v1"

VIEWS: tuple[str, ...] = ("above", "left", "right")
TASKS: tuple[str, ...] = ("cap", "coin", "glass", "key", "nut", "peg")
SIDES: tuple[str, ...] = ("l", "r")

# Closed repair table, one entry per misspelling observed in the corpus.  A
# generic edit-distance search is deliberately absent: it would resolve a tie
# silently, and a wrong task token moves a file into a different family.
TASK_SPELLING_REPAIRS: dict[str, str] = {
    "coini": "coin",
    "gcap": "cap",
    "gpeg": "peg",
    "grass": "glass",
}

CANONICAL = "canonical"
QUARANTINED = "quarantined"
EXCLUDED = "excluded"

REASON_OK = "ok"
EXCLUSION_REASONS: tuple[str, ...] = (
    "broken_symlink",
    "control_character_in_path",
    "not_a_regular_file",
    "path_escapes_root",
    "path_not_utf8",
    "probe_unreadable",
    "read_error",
    "symlink_within_corpus",
    "unsupported_extension",
)
QUARANTINE_REASONS: tuple[str, ...] = (
    "repeat_marker_unrecognized",
    "side_missing",
    "side_unknown",
    "subject_token_conflict",
    "subject_token_nonnumeric",
    "task_unknown",
    "token_count",
    "view_unknown",
)

FLAG_FPS = "fps_invalid"
FLAG_FRAME_COUNT = "frame_count_invalid"
FLAG_DIMENSIONS = "dimensions_invalid"
FLAG_ROTATION = "rotation_unexpected"
VALID_ROTATIONS = frozenset({0, 90, 180, 270})

ASSETS_FILENAME = "assets.csv"
CAPTURES_FILENAME = "captures.csv"
CENSUS_FILENAME = "census.json"

ASSET_COLUMNS: tuple[str, ...] = (
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

CAPTURE_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "subject_ordinal",
    "task",
    "side",
    "n_assets",
    "views",
    "n_views",
    "view_conflict",
    # Every derived column keeps the claim boundary of its source: the header
    # reported it, or the tool divided one header value by another.
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

FPS_DECIMALS = 6
SECOND_DECIMALS = 4
HASH_CHUNK_BYTES = 1024 * 1024

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_UNRENDERABLE = re.compile(r"[\x00-\x1f\x7f-\x9f\udc80-\udcff]")
_REPEAT_MARKER = re.compile(r"_?\((\d+)\)$")
_WHITESPACE_RUN = re.compile(r"\s+")
_UNDERSCORE_RUN = re.compile(r"_+")


class InventoryError(Exception):
    """Raised for a usage or corpus error the caller can correct."""


@dataclasses.dataclass(frozen=True)
class StemParse:
    """Outcome of applying the stem grammar to one filename."""

    subject_ordinal: int | None = None
    view: str = ""
    task: str = ""
    side: str = ""
    repeat: int = 0
    reason_code: str = REASON_OK
    applied: tuple[str, ...] = ()
    task_repair_from: str = ""

    @property
    def ok(self):
        return self.reason_code == REASON_OK


@dataclasses.dataclass
class AssetRecord:
    """One file, its disposition, its identity, and its container facts."""

    asset_id: str
    source_path: str
    disposition: str
    reason_code: str
    parse: StemParse
    facts: ContainerFacts
    size_bytes: int | None
    content_sha256: str

    @property
    def capture_id(self):
        if self.disposition != CANONICAL or self.parse.subject_ordinal is None:
            return ""
        return capture_id_of(self.parse.subject_ordinal, self.parse.task, self.parse.side)

    @property
    def nominal_duration_s(self):
        """Frames divided by the average rate.

        This is a sorting and reporting convenience.  It is not a measured
        duration and it may not be used to qualify synchrony or coverage.
        """
        fps = self.facts.reported_avg_fps
        frames = self.facts.reported_frame_count
        if math.isfinite(fps) and fps > 0 and frames > 0:
            return frames / fps
        return None

    @property
    def fact_flags(self):
        flags = []
        if self.facts.probe_status != PROBE_OPENED:
            return ()
        if not math.isfinite(self.facts.reported_avg_fps) or self.facts.reported_avg_fps <= 0:
            flags.append(FLAG_FPS)
        if self.facts.reported_frame_count <= 0:
            flags.append(FLAG_FRAME_COUNT)
        if self.facts.reported_width <= 0 or self.facts.reported_height <= 0:
            flags.append(FLAG_DIMENSIONS)
        if self.facts.reported_rotation_deg not in VALID_ROTATIONS:
            flags.append(FLAG_ROTATION)
        return tuple(sorted(flags))


@dataclasses.dataclass(frozen=True)
class CaptureRecord:
    """One task-side family: the canonical assets sharing a ``capture_id``.

    A family whose ``view_conflict`` is set holds more than one take, so it is
    not a physical trial and no consumer may bind a session to it.
    """

    capture_id: str
    subject_ordinal: int
    task: str
    side: str
    assets: tuple[AssetRecord, ...]

    @property
    def views(self):
        return tuple(sorted({a.parse.view for a in self.assets}))

    @property
    def view_conflict(self):
        return len(self.assets) > len(self.views)


def strip_media_suffixes(name):
    """N1: remove every trailing media extension, however many there are.

    Four corpus files carry a doubled suffix such as ``.mov.MOV``.  Stripping
    once leaves ``.mov`` occupying the side slot, which is the whole reason two
    earlier normalizers disagreed about the view coverage of this corpus.
    """
    return _strip_media_suffixes(name)[0]


def _strip_media_suffixes(name):
    """Return the stem and how many media suffixes came off it."""
    stem = name
    removed = 0
    while True:
        suffix = pathlib.PurePosixPath(stem).suffix
        if suffix and suffix.lower() in VIDEO_EXTENSIONS:
            stem = stem[: -len(suffix)]
            removed += 1
            continue
        return stem, removed


def normalize_stem(stem):
    """N2-N6: lower case, whitespace to underscore, collapse runs, trim the front.

    A leading separator carries nothing, so it goes.  A trailing separator is
    kept, because it marks a slot the operator left empty: ``9_right_cap_``
    must read as an absent side rather than as a three-token stem.
    """
    return _normalize_stem(stem)[0]


def _normalize_stem(stem):
    """Return the normalized stem and the steps that actually changed it.

    Each label comes from comparing the text across its own step, never from
    re-inspecting the raw name afterwards.  A guess reads ``a _ b`` as needing
    no underscore collapse, when substituting the spaces is exactly what
    creates the run that N5 then collapses.
    """
    applied = []
    text = stem.lower()
    if text != stem:
        applied.append("case_folded")
    trimmed = text.strip()
    if trimmed != text:
        applied.append("outer_trimmed")
    spaced = _WHITESPACE_RUN.sub("_", trimmed)
    if spaced != trimmed:
        applied.append("whitespace_collapsed")
    collapsed = _UNDERSCORE_RUN.sub("_", spaced)
    if collapsed != spaced:
        applied.append("underscore_collapsed")
    stripped = collapsed.lstrip("_")
    if stripped != collapsed:
        applied.append("leading_separator_stripped")
    return stripped, applied


def split_repeat(stem):
    """N7: pull a trailing ``(k)`` marker off the stem.

    The marker is what a file manager appends when a name already exists, so
    the marked file is a second recording rather than a copy.  Its integer is
    kept as ``repeat`` and never renumbered.
    """
    match = _REPEAT_MARKER.search(stem)
    if match is None:
        return stem, 0
    marker = int(match.group(1))
    if marker == 0:
        # No file manager numbers a copy from zero, so "(0)" is not a repeat
        # marker.  Leaving it on the stem routes it to the unrecognized branch
        # and keeps "repeat >= 1" equivalent to "a marker was consumed".
        return stem, 0
    return stem[: match.start()], marker


def parse_stem(name):
    """Apply the whole grammar to one filename and return the outcome.

    Every normalization that changed something is recorded in ``applied``, so
    the census can report how much repair the corpus needed instead of hiding
    it inside a parser.
    """
    raw, suffixes_removed = _strip_media_suffixes(name)
    stem, applied = _normalize_stem(raw)
    if suffixes_removed > 1:
        applied.append("media_suffix_doubled")
    stem, repeat = split_repeat(stem)
    if repeat:
        applied.append("repeat_marker")

    task_repair_from = ""

    def failed(reason):
        # A quarantined name keeps its repair provenance, so the census
        # histogram of repairs stays equal to the count of repaired rows.
        return StemParse(
            reason_code=reason, applied=tuple(applied), task_repair_from=task_repair_from
        )

    if repeat == 0 and stem.endswith(")"):
        # It looks like a copy marker and is not one, so the identity of this
        # file is undecidable rather than merely mis-spelled.
        return failed("repeat_marker_unrecognized")
    tokens = stem.split("_") if stem else []
    if len(tokens) != 4:
        return failed("token_count")
    ordinal_text, view, task, side = tokens
    repaired = TASK_SPELLING_REPAIRS.get(task, task)
    task_repair_from = task if repaired != task else ""
    if task_repair_from:
        applied.append("task_repaired")
    task = repaired
    # ASCII digits only.  ``str.isdigit`` accepts Arabic-Indic digits, which
    # ``int`` would silently fold onto the same ordinal as their ASCII twins,
    # and superscripts, which ``int`` rejects outright with a ValueError.
    if not (ordinal_text.isascii() and ordinal_text.isdigit()):
        return failed("subject_token_nonnumeric")
    if view not in VIEWS:
        return failed("view_unknown")
    if task not in TASKS:
        return failed("task_unknown")
    if not side:
        return failed("side_missing")
    if side not in SIDES:
        return failed("side_unknown")
    return StemParse(
        applied=tuple(applied),
        task_repair_from=task_repair_from,
        subject_ordinal=int(ordinal_text),
        view=view,
        task=task,
        side=side,
        repeat=repeat,
    )


def asset_id_of(relative_posix_path):
    """Name one file from its corpus-relative path.

    A 64-bit digest of a unique path collides only with negligible
    probability, which is a different claim from never.  The guarantee is the
    explicit collision check in ``check_invariants``, which refuses to publish
    rather than merging two rows.  Traversal order and file content never
    enter the identifier.
    """
    digest = hashlib.blake2b(
        relative_posix_path.encode("utf-8", "surrogateescape"),
        digest_size=8,
        person=b"pose3cam-asset",
    )
    return f"a-{digest.hexdigest()}"


def capture_id_of(subject_ordinal, task, side):
    """Name one task-side family from its components.

    The identifier is a low-entropy stable pseudonym.  It supports linkage and
    does not resist enumeration.
    """
    return f"s{subject_ordinal:02d}-{task}-{side}"


def sha256_of(path):
    """Return the file's SHA-256, or "" when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _iter_entries(root, skip):
    """Yield every filesystem entry under *root*, never descending into *skip*."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir())
        except OSError as exc:
            # No path in the text: a corpus directory name identifies a
            # subject, and this message reaches an agent's context.
            raise InventoryError(
                "A directory under the corpus cannot be read. Correct its permissions."
            ) from exc
        for entry in children:
            if entry.is_symlink():
                yield entry
            elif entry.is_dir():
                if entry.resolve() not in skip:
                    stack.append(entry)
            else:
                yield entry


def _relative_posix(entry, corpus_root):
    """Return the corpus-relative POSIX path as canonical UTF-8 text.

    Python decodes a filename with the filesystem encoding, so under a C
    locale every non-ASCII byte arrives as a surrogate.  Recovering the bytes
    and decoding them strictly once, here, is what makes classification,
    parsing, ordering and published text identical under every locale.
    Without it the same file publishes ``é`` on one machine and ``\\xc3\\xa9``
    on another, and a name separated by non-ASCII whitespace parses canonical
    under one locale and fails the token count under the other.

    A name whose bytes are not UTF-8 keeps its surrogate form, which is what
    ``_printable_path`` reverses to those bytes and what ``path_not_utf8``
    refuses.  ``asset_id`` is unaffected either way, because both forms encode
    back to the same bytes.
    """
    relative = entry.relative_to(corpus_root).as_posix()
    try:
        return os.fsencode(relative).decode("utf-8")
    except UnicodeDecodeError:
        return relative


def _exclusion_reason(entry, root, relative):
    """Return the exclusion code for *entry*, or None when it is a corpus file.

    *relative* arrives canonical from ``_relative_posix``, so every check here
    reads the same text on every machine.

    A symlink is never an asset. One resolving outside the corpus escapes the
    root; one resolving inside it is a second reference to bytes the walk
    already reached by their real path, and admitting it would add a phantom
    member to a capture.
    """
    if _CONTROL_CHARS.search(relative):
        return "control_character_in_path"
    try:
        relative.encode("utf-8")
    except UnicodeEncodeError:
        # Filename bytes that are not valid UTF-8.  The file may be perfectly
        # readable; its name is what cannot cross into a UTF-8 artifact, so it
        # is refused by name rather than misreported as an unreadable probe.
        return "path_not_utf8"
    if not entry.exists():
        return "broken_symlink"
    if not entry.resolve().is_relative_to(root):
        return "path_escapes_root"
    if entry.is_symlink():
        return "symlink_within_corpus"
    if not entry.is_file():
        return "not_a_regular_file"
    if pathlib.PurePosixPath(relative).suffix.lower() not in VIDEO_EXTENSIONS:
        return "unsupported_extension"
    return None


def discover_paths(corpus_root, out_dir=None):
    """Return every discovered entry as a corpus-relative POSIX path.

    ``run`` walks the corpus a second time through this function so the
    coverage invariant compares the rows against the filesystem rather than
    against the list the rows were built from.
    """
    skip = {out_dir.resolve()} if out_dir is not None else set()
    return [_relative_posix(entry, corpus_root) for entry in _iter_entries(corpus_root, skip)]


def build_assets(corpus_root, out_dir, *, checksums=True):
    """Return one AssetRecord for every entry under *corpus_root*."""
    root = corpus_root.resolve()
    skip = {out_dir.resolve()} if out_dir is not None else set()
    records = []
    for entry in _iter_entries(corpus_root, skip):
        relative = _relative_posix(entry, corpus_root)
        reason = _exclusion_reason(entry, root, relative)
        if reason is not None:
            records.append(
                AssetRecord(
                    asset_id=asset_id_of(relative),
                    source_path=relative,
                    disposition=EXCLUDED,
                    reason_code=reason,
                    parse=StemParse(reason_code=reason),
                    facts=_skipped_facts(),
                    size_bytes=_size_of(entry),
                    content_sha256="",
                )
            )
            continue
        facts = probe_container(entry)
        # Parse the canonical name, not ``entry.name``: under a C locale the
        # latter carries surrogates, so a name separated by non-ASCII
        # whitespace would fail the token count on one machine and parse
        # canonically on another.
        parse = parse_stem(relative.rsplit("/", 1)[-1])
        size_bytes = _size_of(entry)
        content_sha256 = sha256_of(entry) if checksums else ""
        if size_bytes is None or (checksums and not content_sha256):
            # Discovery reached the file and reading it failed.  Publishing a
            # canonical row with no fixity would be indistinguishable from a
            # deliberate --no-checksums run, so the failure gets its own row.
            disposition, reason_code = EXCLUDED, "read_error"
        elif facts.probe_status != PROBE_OPENED:
            disposition, reason_code = EXCLUDED, "probe_unreadable"
        elif parse.ok:
            disposition, reason_code = CANONICAL, REASON_OK
        else:
            disposition, reason_code = QUARANTINED, parse.reason_code
        records.append(
            AssetRecord(
                asset_id=asset_id_of(relative),
                source_path=relative,
                disposition=disposition,
                reason_code=reason_code,
                parse=parse,
                facts=facts,
                size_bytes=size_bytes,
                content_sha256=content_sha256,
            )
        )
    apply_subject_crosscheck(records)
    records.sort(key=lambda record: record.source_path)
    return records


def _skipped_facts():
    return ContainerFacts(
        probe_status=PROBE_SKIPPED,
        backend_name="",
        reported_width=0,
        reported_height=0,
        reported_avg_fps=0.0,
        reported_frame_count=0,
        reported_rotation_deg=0,
        reported_fourcc="",
        orientation_auto=False,
    )


def _size_of(entry):
    # A symlink's size is its target's.  Reading one imports bytes the corpus
    # does not own: an external target inflates the total, and an internal one
    # double-counts a file the walk already reached by its real path.
    try:
        if entry.is_symlink():
            return None
        return entry.stat().st_size
    except OSError:
        return None


def apply_subject_crosscheck(records):
    """Quarantine every asset whose subject ordinal contradicts its directory.

    The leading stem number is the subject ordinal, so it must be constant
    inside one subject directory and unique across the corpus.  A contradiction
    is unresolvable from the data, so both sides lose their canonical status
    rather than one side winning.
    """
    by_directory = collections.defaultdict(set)
    for record in records:
        if record.disposition == CANONICAL:
            directory = pathlib.PurePosixPath(record.source_path).parent.as_posix()
            by_directory[directory].add(record.parse.subject_ordinal)
    bad = {d for d, ordinals in by_directory.items() if len(ordinals) > 1}
    owners = collections.defaultdict(set)
    for directory, ordinals in by_directory.items():
        for ordinal in ordinals:
            owners[ordinal].add(directory)
    for directories in owners.values():
        if len(directories) > 1:
            bad.update(directories)
    if not bad:
        return
    for record in records:
        directory = pathlib.PurePosixPath(record.source_path).parent.as_posix()
        if record.disposition == CANONICAL and directory in bad:
            record.disposition = QUARANTINED
            record.reason_code = "subject_token_conflict"


def build_captures(assets):
    """Group canonical assets into one record per task-side family, by capture_id."""
    grouped = collections.defaultdict(list)
    for asset in assets:
        if asset.disposition == CANONICAL:
            grouped[asset.capture_id].append(asset)
    captures = []
    for capture_id in sorted(grouped):
        members = tuple(grouped[capture_id])
        first = members[0].parse
        captures.append(
            CaptureRecord(
                capture_id=capture_id,
                subject_ordinal=first.subject_ordinal,
                task=first.task,
                side=first.side,
                assets=members,
            )
        )
    return captures


def _round(value, decimals):
    """Render a float at a fixed precision, or "" when it does not exist.

    ``str`` of a rounded float is the shortest text that reads back as the same
    float, so the rendering is stable across platforms and does not lose a
    small within-capture rate difference the way a significant-digit format
    would.
    """
    if value is None or not math.isfinite(value):
        return ""
    return str(round(value, decimals))


def _printable_path(relative):
    """Escape control characters so no CSV cell can drive a terminal.

    Lone surrogates are escaped for a different reason: they are how Python
    carries a filename byte that is not valid UTF-8, and writing one to a
    UTF-8 artifact raises.

    The encoding is injective onto the path's bytes, so the cell reverses to
    them.  Two rules earn that.  The escape introducer is escaped first, which
    keeps a path holding a newline distinct from one holding the four
    characters ``\\x0a``.  A control code point then renders the ``\\xNN`` of
    each of its own UTF-8 bytes while a surrogate renders the single byte it
    carries, which keeps U+0080 (``\\xc2\\x80``) distinct from the raw byte
    ``0x80`` (``\\x80``).  Rendering both as ``\\x80`` conflated a file named
    with a C1 control and a file whose name is not UTF-8 at all.
    """

    def escape(match):
        code = ord(match.group())
        if 0xDC80 <= code <= 0xDCFF:
            return f"\\x{code - 0xDC00:02x}"
        return "".join(f"\\x{byte:02x}" for byte in chr(code).encode("utf-8"))

    return _UNRENDERABLE.sub(escape, relative.replace("\\", "\\\\"))


def asset_row(record):
    """Return the ordered CSV row for one asset."""
    parse = record.parse
    canonical = record.disposition == CANONICAL
    return {
        "asset_id": record.asset_id,
        "capture_id": record.capture_id,
        "disposition": record.disposition,
        "reason_code": record.reason_code,
        "source_path": _printable_path(record.source_path),
        "subject_ordinal": parse.subject_ordinal if canonical else "",
        "view": parse.view if canonical else "",
        "task": parse.task if canonical else "",
        "side": parse.side if canonical else "",
        "repeat": parse.repeat if canonical else "",
        # Every normalization that fired, on the row it fired on: the parse
        # outcome is exclusive, the name repairs behind it are not.
        "normalizations": "|".join(sorted(parse.applied)),
        "size_bytes": "" if record.size_bytes is None else record.size_bytes,
        "content_sha256": record.content_sha256,
        "reported_width": record.facts.reported_width,
        "reported_height": record.facts.reported_height,
        "reported_avg_fps": _round(record.facts.reported_avg_fps, FPS_DECIMALS),
        "reported_frame_count": record.facts.reported_frame_count,
        "reported_rotation_deg": record.facts.reported_rotation_deg,
        "reported_fourcc": record.facts.reported_fourcc,
        "nominal_duration_s": _round(record.nominal_duration_s, SECOND_DECIMALS),
        "fact_flags": "|".join(record.fact_flags),
        "probe_status": record.facts.probe_status,
        "grammar_version": GRAMMAR_VERSION,
        "tool_version": TOOL_VERSION,
    }


def _durations(capture):
    return [d for d in (a.nominal_duration_s for a in capture.assets) if d is not None]


def duration_spread_s(capture):
    """Return the widest nominal-duration difference inside one family."""
    durations = _durations(capture)
    return max(durations) - min(durations) if durations else None


def resolution_agree(capture):
    """Report whether every view of one family reports the same frame size."""
    return len({(a.facts.reported_width, a.facts.reported_height) for a in capture.assets}) == 1


def rotation_agree(capture):
    """Report whether every view of one family reports the same rotation."""
    return len({a.facts.reported_rotation_deg for a in capture.assets}) == 1


def capture_row(capture):
    """Return the ordered CSV row for one task-side family."""
    frames = [a.facts.reported_frame_count for a in capture.assets]
    rates = [a.facts.reported_avg_fps for a in capture.assets]
    # ``min``/``max`` with a NaN argument return whichever member came first,
    # so a family holding one unreported rate published a different row for a
    # different member order.  A family rate is unknown when any member's rate
    # is unknown, which is both deterministic and the honest reading: a
    # smallest, largest or spread taken over the members that happened to
    # report reads as a fact about the family it is not.
    known_rates = rates if all(math.isfinite(r) for r in rates) else []
    fps_min = min(known_rates) if known_rates else None
    fps_max = max(known_rates) if known_rates else None
    durations = _durations(capture)
    views = capture.views
    return {
        "capture_id": capture.capture_id,
        "subject_ordinal": capture.subject_ordinal,
        "task": capture.task,
        "side": capture.side,
        "n_assets": len(capture.assets),
        "views": "|".join(views),
        "n_views": len(views),
        "view_conflict": int(capture.view_conflict),
        "reported_frame_count_min": min(frames),
        "reported_frame_count_max": max(frames),
        "reported_fps_min": _round(fps_min, FPS_DECIMALS),
        "reported_fps_max": _round(fps_max, FPS_DECIMALS),
        "reported_fps_spread_hz": _round(
            None if fps_min is None else fps_max - fps_min, FPS_DECIMALS
        ),
        "nominal_duration_min_s": _round(min(durations) if durations else None, SECOND_DECIMALS),
        "nominal_duration_max_s": _round(max(durations) if durations else None, SECOND_DECIMALS),
        "nominal_duration_spread_s": _round(duration_spread_s(capture), SECOND_DECIMALS),
        "reported_resolution_agree": int(resolution_agree(capture)),
        "reported_rotation_agree": int(rotation_agree(capture)),
        "grammar_version": GRAMMAR_VERSION,
        "tool_version": TOOL_VERSION,
    }


def _quantiles(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    last = len(ordered) - 1

    def at(fraction):
        return round(ordered[min(last, int(fraction * len(ordered)))], SECOND_DECIMALS)

    return {
        "count": len(ordered),
        "min": round(ordered[0], SECOND_DECIMALS),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": round(ordered[-1], SECOND_DECIMALS),
    }


def build_census(assets, captures, *, checksums, opencv_version, backend_name, generation=None):
    """Return the aggregate census.

    Nothing in the returned mapping identifies a file, a directory, or a
    subject.  It is the only inventory artifact whose numbers may be quoted
    outside the local machine.
    """
    opened = [a for a in assets if a.facts.probe_status == PROBE_OPENED]
    canonical = [a for a in assets if a.disposition == CANONICAL]
    # Count the rendered key, never the raw tuple.  Two NaN rates are unequal
    # as mapping keys and identical once rendered, so counting tuples first
    # and rendering afterwards let one overwrite the other and lost an asset.
    # Rendering first also gives the sort a total order to work with.
    shapes = collections.Counter(
        _shape_key(
            (
                a.facts.reported_width,
                a.facts.reported_height,
                round(a.facts.reported_avg_fps, 3),
                a.facts.reported_fourcc,
                a.facts.reported_rotation_deg,
            )
        )
        for a in opened
    )
    rotation_by_view = collections.defaultdict(collections.Counter)
    for asset in canonical:
        rotation_by_view[asset.parse.view][asset.facts.reported_rotation_deg] += 1
    codecs_by_directory = collections.defaultdict(set)
    for asset in opened:
        directory = pathlib.PurePosixPath(asset.source_path).parent.as_posix()
        codecs_by_directory[directory].add(asset.facts.reported_fourcc)
    multi_view = [c for c in captures if len(c.views) > 1]
    parities = [_frame_parity(c) for c in multi_view]
    spreads = [s for s in (duration_spread_s(c) for c in captures) if s is not None]
    multi_spreads = [s for s in (duration_spread_s(c) for c in multi_view) if s is not None]
    return {
        "tool_version": TOOL_VERSION,
        "grammar_version": GRAMMAR_VERSION,
        "opencv_version": opencv_version,
        "backend_name": backend_name,
        "orientation_auto": bool(opened and opened[0].facts.orientation_auto),
        "checksums": checksums,
        "generation": dict(sorted((generation or {}).items())),
        "assets": {
            "discovered": len(assets),
            "canonical": len(canonical),
            "quarantined": sum(1 for a in assets if a.disposition == QUARANTINED),
            "excluded": sum(1 for a in assets if a.disposition == EXCLUDED),
            "total_bytes": sum(a.size_bytes or 0 for a in assets),
            "distinct_sha256": len({a.content_sha256 for a in assets if a.content_sha256}),
            "reported_frames_total": sum(max(a.facts.reported_frame_count, 0) for a in opened),
            "nominal_minutes_total": round(
                sum(a.nominal_duration_s or 0.0 for a in opened) / 60.0, SECOND_DECIMALS
            ),
            "nominal_duration_s": _quantiles(
                [d for d in (a.nominal_duration_s for a in opened) if d is not None]
            ),
        },
        # The whole closed vocabulary, zeros included: a consumer that looks up
        # a reason must never have to tell a missing key from a count of none.
        "reason_codes": {
            reason: sum(1 for a in assets if a.reason_code == reason)
            for reason in sorted((REASON_OK, *QUARANTINE_REASONS, *EXCLUSION_REASONS))
        },
        "normalization": {
            "applied": dict(
                sorted(
                    collections.Counter(step for a in assets for step in a.parse.applied).items()
                )
            ),
            "task_repairs": dict(
                sorted(
                    collections.Counter(
                        a.parse.task_repair_from for a in assets if a.parse.task_repair_from
                    ).items()
                )
            ),
        },
        "extension_case": dict(
            sorted(collections.Counter(_extension_of(a.source_path) for a in assets).items())
        ),
        "shapes": dict(sorted(shapes.items())),
        "rotation_by_view": {
            view: dict(sorted(counter.items()))
            for view, counter in sorted(rotation_by_view.items())
        },
        "directories_mixing_codecs": sum(1 for c in codecs_by_directory.values() if len(c) > 1),
        # Directories holding admitted assets, so a failed probe cannot make a
        # subject directory disappear from the count.  Codec mixing stays on
        # the opened set, because comparing codecs needs a codec.
        "subject_directories": len(
            {
                pathlib.PurePosixPath(a.source_path).parent.as_posix()
                for a in assets
                if a.disposition != EXCLUDED
            }
        ),
        "captures": {
            "total": len(captures),
            "view_coverage": dict(
                sorted(collections.Counter(len(c.views) for c in captures).items())
            ),
            "with_view_conflict": sum(1 for c in captures if c.view_conflict),
            "multi_view": len(multi_view),
            "same_resolution": sum(1 for c in multi_view if resolution_agree(c)),
            "same_fps_3dp": sum(1 for c in multi_view if _same_fps_3dp(c)),
            "frame_parity_within_5pct": sum(1 for p in parities if p <= 0.05),
            "frame_parity_within_20pct": sum(1 for p in parities if p <= 0.20),
            "duration_spread_s": _quantiles(multi_spreads),
        },
        "duration_spread_all_captures_s": _quantiles(spreads),
    }


def _extension_of(source_path):
    # The census is the one artifact whose contents may be quoted off this
    # machine, so only a recognized video extension crosses into it verbatim.
    # Any other suffix is free text from a filename -- `clip.<anything>` would
    # otherwise publish that text -- and the case survey it belongs to only
    # ever asked about video extensions.
    suffix = pathlib.PurePosixPath(source_path).suffix
    if not suffix:
        return "<none>"
    return suffix if suffix.lower() in VIDEO_EXTENSIONS else "<unsupported>"


def _shape_key(shape):
    width, height, fps, fourcc, rotation = shape
    return f"{width}x{height}@{fps:g}/{fourcc or '?'}/rot{rotation}"


def _same_fps_3dp(capture):
    return len({round(a.facts.reported_avg_fps, 3) for a in capture.assets}) == 1


def _frame_parity(capture):
    frames = [a.facts.reported_frame_count for a in capture.assets]
    top = max(frames)
    return (top - min(frames)) / top if top > 0 else 1.0


def check_invariants(assets, captures, discovered_paths):
    """Assert the partition before anything is published.

    A failure here means the census is wrong about its own corpus, so it must
    not replace a previous artifact.
    """
    counts = collections.Counter(a.source_path for a in assets)
    if counts != collections.Counter(discovered_paths):
        raise InventoryError("The asset rows do not match the discovered files exactly.")
    if len({a.asset_id for a in assets}) != len(assets):
        raise InventoryError("Two assets share one identifier.")
    partition = collections.Counter(a.disposition for a in assets)
    if set(partition) - {CANONICAL, QUARANTINED, EXCLUDED}:
        raise InventoryError("An asset carries a disposition outside the closed vocabulary.")
    if sum(partition.values()) != len(assets):
        raise InventoryError("The dispositions do not partition the assets.")
    for asset in assets:
        allowed = (
            (REASON_OK,)
            if asset.disposition == CANONICAL
            else QUARANTINE_REASONS
            if asset.disposition == QUARANTINED
            else EXCLUSION_REASONS
        )
        if asset.reason_code not in allowed:
            raise InventoryError(
                f"The reason code does not match the disposition: {asset.reason_code}."
            )
        if (asset.disposition == CANONICAL) != bool(asset.capture_id):
            raise InventoryError("A capture identifier exists only on canonical assets.")
    # One readback for the whole corpus, or the reported sizes mix display and
    # coded conventions and every resolution aggregate becomes meaningless.
    readbacks = {a.facts.orientation_auto for a in assets if a.facts.probe_status == PROBE_OPENED}
    if len(readbacks) > 1:
        raise InventoryError("The opened assets disagree about auto-rotation.")
    known = {c.capture_id for c in captures}
    if len(known) != len(captures):
        raise InventoryError("Two capture rows share one identifier.")
    canonical = [a for a in assets if a.disposition == CANONICAL]
    if {a.capture_id for a in canonical} != known:
        raise InventoryError("The capture rows do not cover the canonical assets exactly.")
    # Compare the membership multiset, never a total.  A count survives one
    # asset appearing twice while another never appears at all, which is the
    # exact shape of a faulty grouping.
    if collections.Counter(a.asset_id for c in captures for a in c.assets) != collections.Counter(
        a.asset_id for a in canonical
    ):
        raise InventoryError("The capture membership does not cover the canonical assets exactly.")
    for capture in captures:
        # A count alone survives a swap of two members between two captures.
        if any(a.capture_id != capture.capture_id for a in capture.assets):
            raise InventoryError("A capture holds an asset that belongs to another capture.")


def render_csv(columns, rows):
    """Return the deterministic CSV text for one table."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def render_json(payload):
    """Return the deterministic census text."""
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def write_text(path, text):
    """Write one artifact deterministically, then move it into place.

    The temporary name carries the process id, so two publishers writing the
    same directory cannot land on one another's staging file.  It never
    reaches an artifact, so it does not weaken determinism.
    """
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def census_digest(census):
    """Digest a census over every field except the one holding this digest.

    A document cannot carry a hash of itself, but it can carry a hash of
    everything else, which is what makes an edited census detectable from
    inside the published set.
    """
    body = dict(census)
    generation = census.get("generation") or {}
    body["generation"] = {k: v for k, v in generation.items() if k != CENSUS_FILENAME}
    # Digest the JSON round trip, not the live mapping.  Histogram keys are
    # integers in memory and strings once parsed, and string keys sort in a
    # different order, so only the normalized form digests the same on both
    # sides of publication.
    return _text_digest(render_json(json.loads(render_json(body))))


def validate_generation(out_dir):
    """Return the census of *out_dir*, or raise when the set is inconsistent.

    Every consumer calls this before reading a row.  Writing the census last
    detects nothing on its own: the proof only exists once something checks
    it, and a checksum each consumer reimplements is a checksum that drifts.
    """
    # Coerced because every other path-taking entry point in this package
    # accepts str | PathLike, and a boundary that raises TypeError on a plain
    # string is a boundary consumers route around.
    out_dir = pathlib.Path(out_dir)
    # Every failure leaves through InventoryError.  A consumer that must catch
    # JSONDecodeError to learn a set is unusable has no boundary at all, and a
    # truncated census is exactly the half-published case this call exists for.
    try:
        census = json.loads((out_dir / CENSUS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InventoryError(
            "The published set is unusable: census.json is missing or is not valid JSON."
        ) from error
    if not isinstance(census, dict):
        raise InventoryError("The published set is unusable: census.json is not an object.")
    generation = census.get("generation") or {}
    for name in (ASSETS_FILENAME, CAPTURES_FILENAME):
        # Hash the exact published bytes.  ``read_text`` translates CRLF to
        # LF, so a table rewritten with different line endings satisfied a
        # digest whose whole promise is that the bytes did not change.
        try:
            raw = (out_dir / name).read_bytes()
        except OSError as error:
            raise InventoryError(
                f"The published set is unusable: {name} is missing or cannot be read."
            ) from error
        if hashlib.sha256(raw).hexdigest() != generation.get(name):
            raise InventoryError(
                f"The published set is inconsistent: {name} is a different generation."
            )
    if census_digest(census) != generation.get(CENSUS_FILENAME):
        raise InventoryError(
            "The published set is inconsistent: census.json changed after publication."
        )
    return census


def _text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_summary(census):
    """Return the console summary.

    The census holds no path and no name, so this text cannot leak one.
    """
    assets = census["assets"]
    captures = census["captures"]
    lines = [
        f"Corpus:      {assets['discovered']} files, {assets['total_bytes']} bytes",
        f"Dispositions: canonical {assets['canonical']}, "
        f"quarantined {assets['quarantined']}, excluded {assets['excluded']}",
        f"Reasons:     { ({k: v for k, v in census['reason_codes'].items() if v}) }",
        f"Captures:    {captures['total']}, view coverage {captures['view_coverage']}, "
        f"view conflicts {captures['with_view_conflict']}",
        f"Reported:    {assets['reported_frames_total']} frames, "
        f"{assets['nominal_minutes_total']} nominal minutes",
        f"Probe:       OpenCV {census['opencv_version']} ({census['backend_name']}), "
        f"orientation_auto={census['orientation_auto']}, checksums={census['checksums']}",
    ]
    return "\n".join(lines)


def run(corpus_root, out_dir, *, checksums=True):
    """Census *corpus_root* into *out_dir* and return the census mapping."""
    if not corpus_root.is_dir():
        raise InventoryError("The corpus path is not a directory.")
    if out_dir.resolve() == corpus_root.resolve() or out_dir.resolve().is_relative_to(
        corpus_root.resolve()
    ):
        raise InventoryError("The output directory must sit outside the corpus.")
    assets = build_assets(corpus_root, out_dir, checksums=checksums)
    captures = build_captures(assets)
    check_invariants(assets, captures, discover_paths(corpus_root, out_dir))
    assets_text = render_csv(ASSET_COLUMNS, [asset_row(a) for a in assets])
    captures_text = render_csv(CAPTURE_COLUMNS, [capture_row(c) for c in captures])
    census = build_census(
        assets,
        captures,
        checksums=checksums,
        opencv_version=cv2.__version__,
        # Observed, never assumed: OpenCV picks a backend per file, so the
        # census reports the ones that actually opened this corpus.  More than
        # one is visible rather than hidden, and nothing opened reads empty.
        backend_name="|".join(
            sorted({a.facts.backend_name for a in assets if a.facts.backend_name})
        ),
        generation={
            ASSETS_FILENAME: _text_digest(assets_text),
            CAPTURES_FILENAME: _text_digest(captures_text),
        },
    )
    # The census names the digests of both tables and of its own remaining
    # fields, and it is written last.  ``validate_generation`` turns that into
    # a proof: a half-published set and an edited census both fail it.
    census["generation"][CENSUS_FILENAME] = census_digest(census)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / ASSETS_FILENAME, assets_text)
    write_text(out_dir / CAPTURES_FILENAME, captures_text)
    write_text(out_dir / CENSUS_FILENAME, render_json(census))
    return census


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="pose-estimation-inventory",
        description=(
            "Build the canonical capture registry and the container census for a video corpus. "
            "The tool reads container headers only. It does not decode video."
        ),
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=pathlib.Path,
        help="Read the video corpus from this directory. The tool searches every subdirectory.",
    )
    parser.add_argument(
        "--out",
        default=pathlib.Path("inventory"),
        type=pathlib.Path,
        help="Write the registry to this directory. The default is inventory.",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip the SHA-256 of each file. Use this option for a faster run.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return status 1 if one file or more is not canonical.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point for ``pose-estimation-inventory``."""
    args = _parse_args(argv)
    try:
        census = run(args.corpus, args.out, checksums=not args.no_checksums)
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError:
        # The path is deliberately absent: an operator path can name a subject
        # directory, and this text reaches an agent's context.
        print(
            "ERROR: The registry could not be written. Check the output directory.", file=sys.stderr
        )
        return 2
    print(render_summary(census))
    print(f"Wrote: {ASSETS_FILENAME}, {CAPTURES_FILENAME}, {CENSUS_FILENAME}")
    if args.strict and census["assets"]["canonical"] != census["assets"]["discovered"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
