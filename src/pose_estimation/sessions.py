"""Session materialization: the registry's families become discoverable recording events.

The corpus registry keys a *family* — one subject, one task, one side.  A
family is not a recording event: two of them provably hold more than one
physical take.  This module publishes the grain below it.

``event_id = f"{capture_id}_run-{run_index:02d}"`` and nothing else, so no
event key can equal a family key.  That is the point: calibration must never
bind to a family, and a rule the type system enforces cannot be forgotten.
``run-<index>`` is BIDS's entity for an otherwise-identical repeated
acquisition, which is exactly what a retake is.

A run groups only assets that can be asserted to come from one performance.
Where the registry proves more than one take and nothing says which view
belongs to which, each asset becomes its own single-camera run and says so
through ``take_resolution``.  No published pipeline infers same-take
membership from a filename, a duration or a frame count, and neither does
this one.  ``run_index`` is therefore an ordering, not a chronology.

Nothing here walks the corpus.  Every source is a row the registry already
published, and the tree is a function of those rows plus the corpus bytes
they name.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

from . import inventory
from .multicam import (
    SESSION_FORMAT_VERSION,
    SESSION_GENERATION_FILENAME,
    SESSION_MANIFEST_FILENAME,
)
from .multicam import VIDEO_EXTENSIONS as DISCOVERABLE_EXTENSIONS

GENERATOR_VERSION = "v1"

EVENTS_FILENAME = "events.csv"
PLACEMENTS_FILENAME = "placements.csv"
GENERATION_FILENAME = SESSION_GENERATION_FILENAME

GENERATION_KEYS: tuple[str, ...] = (
    EVENTS_FILENAME,
    PLACEMENTS_FILENAME,
    "tree",
    "inventory",
    "generator_version",
)

PLACED = "placed"
HELD_OUT = "held_out"

REASON_OK = "ok"
# A hold-out is a *qualification* verdict on an asset the registry described
# correctly.  A registry that disagrees with the corpus is not a qualification
# verdict, so it fails the run instead of shrinking the tree by one camera.
HOLD_OUT_REASONS: tuple[str, ...] = (
    "excluded_asset",
    "extension_not_discoverable",
    "quarantined_stem",
)

TAKE_FAMILY = "family"
TAKE_UNRESOLVED = "unresolved"

CAMERA_PREFIX = "cam-"

EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "capture_id",
    "subject_ordinal",
    "task",
    "side",
    "run_index",
    "take_resolution",
    "n_cameras",
    "views",
    "view_conflict",
    "grammar_version",
    "generator_version",
)

PLACEMENT_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "capture_id",
    "disposition",
    "placement",
    "placement_reason",
    "event_id",
    "camera_name",
    # A held-out row carries no event_id, so the ledger is the only place its
    # qualification can be read without reopening the registry.
    "grammar_version",
)

# The registry columns this module reads.  The first of each is its identity.
ASSET_INPUT_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "capture_id",
    "content_sha256",
    "disposition",
    "grammar_version",
    "side",
    "source_path",
    "subject_ordinal",
    "task",
    "view",
)

CAPTURE_INPUT_COLUMNS: tuple[str, ...] = ("capture_id", "grammar_version", "view_conflict")

# Every cell that becomes a filename, a published value, or console text is
# confined to a printable alphabet that cannot lead a spreadsheet formula or
# carry a terminal escape.  Each alphabet is wider than the live corpus needs
# and still excludes the classes above.  A refusal names the column and never
# the value, so a hostile cell cannot reach the console through its own error.
# These are applied with fullmatch, never match: `$` also matches before one
# trailing newline, which would admit a newline into every published cell.
_CELL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "asset_id": re.compile(r"[a-z0-9][a-z0-9._-]*"),
    "capture_id": re.compile(r"(?:[a-z0-9][a-z0-9._-]*)?"),
    "content_sha256": re.compile(r"[0-9a-f]*"),
    "grammar_version": re.compile(r"[a-z0-9][a-z0-9._-]*"),
}

_CANONICAL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "side": re.compile(r"[a-z]+"),
    # ASCII, not str.isdigit: that predicate is true for superscripts, which
    # then raise out of int(), and for other scripts' digits, which int()
    # silently normalizes into an ordinal the cell never spelled.
    "subject_ordinal": re.compile(r"[0-9]+"),
    "task": re.compile(r"[a-z0-9]+"),
    "view": re.compile(r"[a-z0-9][a-z0-9-]*"),
}

# `\Z`, not `$`: this pattern is exported, so a consumer's own `match` call has
# to reject a trailing newline too.
EVENT_ID_PATTERN = re.compile(r"^(?P<capture_id>s\d{2}-[a-z]+-[lr])_run-(?P<run_index>\d{2})\Z")

# Inverse of inventory._printable_path.  The introducer is matched before the
# escape so a path holding a backslash stays distinct from one holding the
# four characters of an escape.
_ESCAPE = re.compile(r"\\\\|\\x([0-9a-f]{2})")


class SessionsError(Exception):
    """Raised for a usage or registry error the caller can correct.

    ``reason`` carries the hold-out code when the error concerns one asset,
    so the caller records a vocabulary term rather than parsing a message.
    """

    def __init__(self, message, *, reason=""):
        super().__init__(message)
        self.reason = reason


@dataclasses.dataclass(frozen=True)
class Camera:
    """One view inside one recording event."""

    name: str
    view: str
    asset_id: str
    link_name: str
    source_relative: str
    content_sha256: str


@dataclasses.dataclass(frozen=True)
class Event:
    """One recording event: the assets assertable as a single performance."""

    event_id: str
    capture_id: str
    subject_ordinal: int
    task: str
    side: str
    run_index: int
    take_resolution: str
    view_conflict: int
    grammar_version: str
    cameras: tuple[Camera, ...]

    @property
    def views(self) -> str:
        return "|".join(sorted({c.view for c in self.cameras}))


@dataclasses.dataclass(frozen=True)
class Placement:
    """The outcome recorded for one discovered asset."""

    asset_id: str
    capture_id: str
    disposition: str
    placement: str
    placement_reason: str
    event_id: str
    camera_name: str
    grammar_version: str


def decode_source_path(cell: str) -> str:
    """Return the corpus-relative path a published ``source_path`` cell names.

    The registry's encoding is injective onto the path's bytes, so this is a
    total inverse rather than a best effort.  The live corpus contains no
    escaped cell at all, which means the corpus provides this function zero
    coverage and only synthetic paths can test it.
    """
    raw = bytearray()
    index = 0
    while index < len(cell):
        match = _ESCAPE.match(cell, index)
        if match:
            raw += b"\\" if match.group(1) is None else bytes([int(match.group(1), 16)])
            index = match.end()
        elif cell[index] == "\\":
            # A lone introducer cannot appear in a cell this encoder wrote, so
            # accepting it would decode a corrupt cell into a plausible path.
            raise SessionsError(
                "A source_path cell carries an unrecognized escape.",
                reason="source_path_unsafe",
            )
        else:
            raw += cell[index].encode("utf-8")
            index += 1
    return os.fsdecode(bytes(raw))


def _relative_parts(relative: str) -> tuple[str, ...]:
    """Return the path components, or raise when the path may not be joined."""
    if not relative or relative.startswith("/") or "\x00" in relative:
        raise SessionsError(
            "A source_path cell is absolute, empty, or holds a NUL.",
            reason="source_path_unsafe",
        )
    parts = tuple(relative.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise SessionsError(
            "A source_path cell holds an empty or traversing component.",
            reason="source_path_unsafe",
        )
    return parts


def absolute_lexical(path: str | os.PathLike[str]) -> pathlib.Path:
    """Make *path* absolute without resolving a symlink.

    ``Path.resolve`` is wrong everywhere this module places a path.  The
    corpus root is itself a symlink out of the checkout, and the container
    and the host reach the same tree through different absolute paths, so
    resolving either end would bake one machine's layout into a link target
    that is supposed to be portable.
    """
    return pathlib.Path(path).absolute()


def _is_within(child: str, parent: str) -> bool:
    """Return whether *child* is *parent* or sits below it. Both are real paths.

    The separator is appended to the *stripped* parent because a parent of
    ``/`` would otherwise form the prefix ``//``, which no real path carries,
    so every file below a filesystem-root corpus would read as an escape.
    """
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def resolve_source(corpus_root: str | os.PathLike[str], relative: str) -> pathlib.Path:
    """Return the absolute path of one listed asset, or raise.

    This validates a path the registry already published; it never discovers
    one.  Containment is checked against the *resolved* root, so a symlinked
    corpus root stays legal while a target escaping it does not.
    """
    _relative_parts(relative)
    root = absolute_lexical(corpus_root)
    target = root / relative
    real_root = os.path.realpath(root)
    real_target = os.path.realpath(target)
    if not _is_within(real_target, real_root):
        raise SessionsError(
            "A listed asset resolves outside the corpus root.", reason="source_path_unsafe"
        )
    if not pathlib.Path(real_target).is_file():
        raise SessionsError("A listed asset is not a regular file.", reason="source_missing")
    return target


def _read_table(path: pathlib.Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Read one registry table, proving the header carries every column read.

    A table with no rows carries its schema in the header alone, so the
    per-row checks below never reach it and an empty short table would
    otherwise publish an empty tree instead of failing.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in columns if column not in (reader.fieldnames or ())]
        if missing:
            raise SessionsError(f"{path.name} is missing {', '.join(missing)}.")
        return list(reader)


def _validate_tables(
    assets: list[dict[str, str]], captures: list[dict[str, str]]
) -> dict[str, tuple[bool, str]]:
    """Return each family's view conflict and grammar version, or raise.

    Upstream validation proves the registry's *bytes*, never its shape, so
    every field this module reads is checked here instead of trusted: an
    absent column would otherwise leak a ``KeyError`` past the documented
    error domain, and a malformed cell would publish a differently-grained
    tree that validates as authoritative.  The conflict flag is *derived*
    from the canonical rows by the registry's own definition, and the
    published cell has to agree with that derivation.
    """
    for rows, columns, table in (
        (assets, ASSET_INPUT_COLUMNS, "assets"),
        (captures, CAPTURE_INPUT_COLUMNS, "captures"),
    ):
        identities = set()
        for row in rows:
            missing = [column for column in columns if column not in row]
            if missing:
                raise SessionsError(f"A {table} row is missing {', '.join(missing)}.")
            for column, alphabet in _CELL_ALPHABETS.items():
                if column in row and not alphabet.fullmatch(row[column]):
                    raise SessionsError(f"A {table} row carries a {column} outside its alphabet.")
            identities.add(row[columns[0]])
        if len(identities) != len(rows):
            raise SessionsError(f"The {table} table repeats a {columns[0]}.")

    canonical: dict[str, list[dict[str, str]]] = {}
    dispositions = (inventory.CANONICAL, inventory.QUARANTINED, inventory.EXCLUDED)
    for row in assets:
        if row["disposition"] not in dispositions:
            raise SessionsError("An assets row carries a disposition the registry never writes.")
        if row["disposition"] != inventory.CANONICAL:
            continue
        for column, alphabet in _CANONICAL_ALPHABETS.items():
            if not alphabet.fullmatch(row[column]):
                raise SessionsError(
                    f"A canonical assets row carries a {column} outside its alphabet."
                )
        canonical.setdefault(row["capture_id"], []).append(row)

    families: dict[str, tuple[bool, str]] = {}
    for row in captures:
        capture_id, cell = row["capture_id"], row["view_conflict"]
        if cell not in ("0", "1"):
            raise SessionsError("A captures row carries a view_conflict outside 0 and 1.")
        members = canonical.get(capture_id, [])
        if (len(members) != len({member["view"] for member in members})) != (cell == "1"):
            raise SessionsError(f"Family {capture_id} contradicts its published view_conflict.")
        families[capture_id] = (cell == "1", row["grammar_version"])

    for capture_id, members in canonical.items():
        if capture_id not in families:
            raise SessionsError(f"Family {capture_id} holds canonical assets and no capture row.")
        if any(member["grammar_version"] != families[capture_id][1] for member in members):
            raise SessionsError(f"Family {capture_id} mixes grammar versions across the tables.")
    return families


def plan(
    assets: list[dict[str, str]],
    captures: list[dict[str, str]],
    *,
    corpus_root: str | os.PathLike[str],
) -> tuple[list[Event], list[Placement]]:
    """Return the events to publish and the outcome of every discovered asset.

    Pure over its inputs apart from the ``stat`` that proves a listed source
    exists, so the tree is a function of the registry's bytes.
    """
    families = _validate_tables(assets, captures)

    placements: list[Placement] = []
    members: dict[str, list[Camera]] = {}
    held: dict[str, str] = {}

    for row in assets:
        asset_id, capture_id = row["asset_id"], row["capture_id"]
        disposition = row["disposition"]
        if disposition != inventory.CANONICAL:
            held[asset_id] = (
                "quarantined_stem" if disposition == inventory.QUARANTINED else "excluded_asset"
            )
            continue
        try:
            relative = decode_source_path(row["source_path"])
            extension = pathlib.PurePosixPath(relative).suffix.lower()
            if extension not in DISCOVERABLE_EXTENSIONS:
                # inventory admits .flv; multicam's camera glob does not, so
                # such an asset would land in a tree that cannot discover it.
                # The registry is right about it, so this is a qualification
                # verdict and the ledger records it.
                held[asset_id] = "extension_not_discoverable"
                continue
            resolve_source(corpus_root, relative)
        except SessionsError as error:
            # A canonical row this module cannot decode or resolve means the
            # registry disagrees with the corpus.  Holding it out would drop a
            # camera from an event and publish the smaller event as if whole.
            raise SessionsError(
                f"{error} Asset {asset_id}. Publish the registry again.", reason=error.reason
            ) from error
        view = row["view"]
        members.setdefault(capture_id, []).append(
            Camera(
                name=f"{CAMERA_PREFIX}{view}",
                view=view,
                asset_id=asset_id,
                link_name=f"{CAMERA_PREFIX}{view}{extension}",
                source_relative=relative,
                content_sha256=row["content_sha256"],
            )
        )

    events: list[Event] = []
    assigned: dict[str, tuple[str, str]] = {}
    by_asset = {row["asset_id"]: row for row in assets}

    for capture_id in sorted(members):
        cameras = members[capture_id]
        conflict, grammar_version = families[capture_id]
        groups = [[c] for c in cameras] if conflict else [sorted(cameras, key=lambda c: c.view)]
        if len(groups) > 99:
            # The published grammar is exactly two digits, so a wider index
            # would emit an id the consumer's own pattern rejects.  Refusing
            # beats printing a key that fails the check this module advertises.
            raise SessionsError("A family needs more run indices than the two-digit grammar holds.")
        for run_index, group in enumerate(groups, start=1):
            row = by_asset[group[0].asset_id]
            event_id = f"{capture_id}_run-{run_index:02d}"
            if not EVENT_ID_PATTERN.match(event_id):
                # The registry's own grammar produces a conforming capture_id,
                # so this fires only for a foreign or corrupt registry.  Emitting
                # a key the module's published pattern rejects is worse than
                # refusing: every consumer parses that key back apart.
                raise SessionsError("A family yields an event id outside the published grammar.")
            events.append(
                Event(
                    event_id=event_id,
                    capture_id=capture_id,
                    subject_ordinal=int(row["subject_ordinal"]),
                    task=row["task"],
                    side=row["side"],
                    run_index=run_index,
                    take_resolution=TAKE_UNRESOLVED if conflict else TAKE_FAMILY,
                    view_conflict=int(conflict),
                    grammar_version=grammar_version,
                    cameras=tuple(group),
                )
            )
            for camera in group:
                assigned[camera.asset_id] = (event_id, camera.name)

    for row in assets:
        asset_id = row["asset_id"]
        event_id, camera_name = assigned.get(asset_id, ("", ""))
        placements.append(
            Placement(
                asset_id=asset_id,
                capture_id=row["capture_id"],
                disposition=row["disposition"],
                placement=PLACED if event_id else HELD_OUT,
                placement_reason=REASON_OK if event_id else held[asset_id],
                event_id=event_id,
                camera_name=camera_name,
                grammar_version=row["grammar_version"],
            )
        )
    events.sort(key=lambda e: e.event_id)
    placements.sort(key=lambda p: p.asset_id)
    return events, placements


def render_manifest(event: Event) -> str:
    """Return the deterministic ``session.json`` text for one event.

    No camera carries ``file``.  ``_safe_resolve`` resolves an explicit ref
    through the symlink and then rejects it for escaping the session
    directory, so naming the file breaks the one tree shape that works.
    ``calibration`` is absent for the same class of reason: a default
    reference would bind calibration before any evidence exists.
    """
    payload = {
        "format_version": SESSION_FORMAT_VERSION,
        "session_id": event.event_id,
        "capture_id": event.capture_id,
        "run_index": event.run_index,
        "subject_ordinal": event.subject_ordinal,
        "task": event.task,
        "side": event.side,
        "take_resolution": event.take_resolution,
        "n_cameras": len(event.cameras),
        "grammar_version": event.grammar_version,
        "generator_version": GENERATOR_VERSION,
        "cameras": [
            {
                "name": camera.name,
                # Always the literal zero, never a value derived from an
                # offset.  This is the legacy integer pre-roll trim in the
                # fusion reader's *frame* domain; the authoritative alignment is
                # `qualification/cameras_qc.csv`'s time-domain `offset_s`, which
                # the fusion reader does not apply.  Merging the two quantities
                # would round a sub-frame time onto a frame index and lose the
                # published evidence.  Removal was priced and refused: `multicam`
                # reads an absent field as 0, so dropping it moves the zero from
                # explicit to implicit and costs a 193-manifest republish.
                "sync_offset": 0,
                "view": camera.view,
                "asset_id": camera.asset_id,
                "content_sha256": camera.content_sha256,
            }
            for camera in event.cameras
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _event_rows(events: list[Event]) -> list[dict[str, str]]:
    return [
        {
            "event_id": e.event_id,
            "capture_id": e.capture_id,
            "subject_ordinal": str(e.subject_ordinal),
            "task": e.task,
            "side": e.side,
            "run_index": str(e.run_index),
            "take_resolution": e.take_resolution,
            "n_cameras": str(len(e.cameras)),
            "views": e.views,
            "view_conflict": str(e.view_conflict),
            "grammar_version": e.grammar_version,
            "generator_version": GENERATOR_VERSION,
        }
        for e in events
    ]


def _placement_rows(placements: list[Placement]) -> list[dict[str, str]]:
    return [dataclasses.asdict(p) for p in placements]


def tree_digest(out_dir: str | os.PathLike[str]) -> str:
    """Digest every entry under *out_dir* except the marker that will carry it.

    Covers each relative name, each entry's kind, each symbolic link's exact
    target text, and each regular file's bytes.  A link target's *contents*
    stay outside, so corpus bytes never enter this digest.

    Kind is load-bearing rather than decorative: ``is_dir()`` follows a link,
    so an event directory swapped for a link to an outside directory would
    otherwise digest as whatever it points at.  Inode, mtime and permissions
    stay outside, because they are not a function of the registry and would
    make a byte-identical regeneration read as a change.
    """
    lines: list[str] = []

    def visit(entry: pathlib.Path, label: str) -> None:
        if entry.is_symlink():
            # os.readlink, not Path.readlink: the latter returns a PurePath,
            # whose constructor drops a leading "./", so two different targets
            # digest the same and the exact-text claim quietly fails.
            lines.append(f"{label}\tlink\t{os.readlink(entry)}\n")  # noqa: PTH115
        elif entry.is_dir():
            lines.append(f"{label}\tdir\n")
            for child in sorted(entry.iterdir()):
                visit(child, f"{label}/{child.name}")
        else:
            lines.append(f"{label}\tfile\t{hashlib.sha256(entry.read_bytes()).hexdigest()}\n")

    for entry in sorted(pathlib.Path(out_dir).iterdir()):
        # A document cannot digest itself; everything else in the tree is fair
        # game, including a file nobody explained.
        if entry.name != GENERATION_FILENAME:
            visit(entry, entry.name)
    # surrogateescape, because a link target reproduces a source path that need
    # not be UTF-8: a plain encode raises on the lone surrogate carrying such a
    # byte, and the digest must cover the target's real bytes either way.
    return hashlib.sha256("".join(lines).encode("utf-8", "surrogateescape")).hexdigest()


def _digest_bytes(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(
    staging: pathlib.Path,
    events: list[Event],
    placements: list[Placement],
    *,
    corpus_root: str | os.PathLike[str],
    upstream: dict[str, str],
) -> None:
    staging.mkdir(parents=True)
    root = absolute_lexical(corpus_root)
    for event in events:
        directory = staging / event.event_id
        directory.mkdir()
        for camera in event.cameras:
            # Planning proved this target, and the corpus can change in
            # between.  A link published over a vanished file breaks discovery
            # while the tree still validates, so re-prove it here; the window
            # narrows rather than closes.
            try:
                resolve_source(corpus_root, camera.source_relative)
            except SessionsError as error:
                raise SessionsError(
                    f"{error} Asset {camera.asset_id}. Publish the registry again.",
                    reason=error.reason,
                ) from error
            # Relative, because the container and the host see this checkout
            # at different absolute paths and an absolute link would bake one
            # of them into the tree.
            target = os.path.relpath(root / camera.source_relative, absolute_lexical(directory))
            (directory / camera.link_name).symlink_to(target)
        (directory / SESSION_MANIFEST_FILENAME).write_text(
            render_manifest(event), encoding="utf-8", newline=""
        )
    (staging / EVENTS_FILENAME).write_text(
        inventory.render_csv(EVENT_COLUMNS, _event_rows(events)), encoding="utf-8", newline=""
    )
    (staging / PLACEMENTS_FILENAME).write_text(
        inventory.render_csv(PLACEMENT_COLUMNS, _placement_rows(placements)),
        encoding="utf-8",
        newline="",
    )
    generation = {
        EVENTS_FILENAME: _digest_bytes(staging / EVENTS_FILENAME),
        PLACEMENTS_FILENAME: _digest_bytes(staging / PLACEMENTS_FILENAME),
        "tree": tree_digest(staging),
        "inventory": dict(upstream),
        "generator_version": GENERATOR_VERSION,
    }
    (staging / GENERATION_FILENAME).write_text(
        json.dumps(generation, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline=""
    )


def _remove(path: pathlib.Path) -> None:
    """Remove a path whether it is a directory or a link to one.

    ``shutil.rmtree`` refuses a symbolic link, and ``ignore_errors`` swallows
    that refusal, so an output that was a link leaves its retiring twin behind.
    """
    if path.is_symlink():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _assert_disjoint(out: pathlib.Path, other: str | os.PathLike[str], label: str) -> None:
    """Refuse an output that overlaps an input, in either direction.

    Publication deletes and replaces the whole output tree, so an output that
    contains the corpus deletes the recordings it links to, and one that
    contains the registry deletes the rows it just read.  An output *inside*
    either is the mirror hazard: the next registry build would discover this
    tree's links as corpus assets.
    """
    here = os.path.realpath(out)
    there = os.path.realpath(other)
    if _is_within(here, there) or _is_within(there, here):
        raise SessionsError(f"The output directory must sit outside the {label}.")


def _sweep_orphans(out: pathlib.Path) -> None:
    """Remove staging and retiring siblings that no live process owns.

    A kill between the two renames leaves both siblings behind, and the
    pid suffix that keeps concurrent runs apart also stops a later run from
    recognizing them.  Sweeping by liveness collects the crash debris and
    still leaves a concurrent generator's directories alone.
    """
    for sibling in out.parent.glob(f"{out.name}.*"):
        stage, _, pid = sibling.name[len(out.name) + 1 :].rpartition(".")
        # This run's own two names are managed around the publication and are
        # live by construction, so liveness never has to decide them.
        if stage not in ("staging", "retiring") or pid == str(os.getpid()):
            continue
        try:
            os.kill(int(pid), 0)
        # OverflowError, not ValueError, is what a suffix wider than a C long
        # raises, and it is no more a live process than a non-numeric one.
        except (ValueError, OverflowError, ProcessLookupError):
            _remove(sibling)
        except PermissionError:
            # Live and owned by another user: not ours to remove.
            continue


def _assert_owned(out_dir: pathlib.Path) -> None:
    """Refuse a non-empty destination this tool did not publish.

    Ownership is the marker's *shape*, never its digests: a tree whose
    digests went stale must stay regenerable, while a foreign or corrupt
    marker must not license deleting someone else's directory.
    """
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise SessionsError("The output path exists and is not a directory.")
    if not any(out_dir.iterdir()):
        return
    refusal = SessionsError(
        "The output directory is not empty and carries no generation marker this tool wrote. "
        "Publishing would delete a directory this tool does not own."
    )
    try:
        marker = json.loads((out_dir / GENERATION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise refusal from error
    if not isinstance(marker, dict) or "generator_version" not in marker:
        raise refusal


def run(
    inventory_dir: str | os.PathLike[str],
    corpus_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
) -> tuple[list[Event], list[Placement]]:
    """Publish the session tree, replacing any generation this tool owns."""
    inventory_path = pathlib.Path(inventory_dir)
    # The swap below replaces the name it publishes to, so a symlinked --out
    # would become a real directory and orphan the tree it pointed at.
    # Publishing to the real path keeps the caller's link intact.
    out = pathlib.Path(os.path.realpath(out_dir))
    census = inventory.validate_generation(inventory_path)
    _assert_disjoint(out, inventory_path, "registry directory")
    _assert_disjoint(out, corpus_root, "corpus")
    _assert_owned(out)

    assets = _read_table(inventory_path / inventory.ASSETS_FILENAME, ASSET_INPUT_COLUMNS)
    captures = _read_table(inventory_path / inventory.CAPTURES_FILENAME, CAPTURE_INPUT_COLUMNS)
    events, placements = plan(assets, captures, corpus_root=corpus_root)

    # Staging and retiring are siblings, never children: discover_sessions
    # iterates the root's children, so a half-built session directory under
    # it would be a discoverable session for as long as it existed.
    staging = out.with_name(f"{out.name}.staging.{os.getpid()}")
    retiring = out.with_name(f"{out.name}.retiring.{os.getpid()}")
    # The sweep skips live pids, and this process is live, so our own two names
    # (a reused pid, or a crash that kept the number) still need clearing.
    _remove(staging)
    _remove(retiring)
    try:
        _build(
            staging,
            events,
            placements,
            corpus_root=corpus_root,
            upstream=census.get("generation", {}),
        )
        if out.exists():
            out.rename(retiring)
        try:
            staging.rename(out)
        except OSError:
            # The old tree is aside and the new one never landed.  A peer that
            # published in between owns the root now, so restore only into an
            # empty name; and an absent root leaves nothing retired, so a bare
            # rename here would raise over the failure that caused it.
            if retiring.exists() and not out.exists():
                retiring.rename(out)
            raise
        # Swept only once the swap has landed: after a kill between the two
        # renames the sole complete generation sits under a dead pid, so
        # sweeping any earlier destroys it whenever this run then fails.
        _sweep_orphans(out)
        _remove(retiring)
    finally:
        _remove(staging)
    return events, placements


def validate_generation(out_dir, inventory_dir=None):
    """Return the generation block of *out_dir*, or raise when it is stale.

    Every consumer calls this before reading a row or opening a camera.  With
    *inventory_dir* it also proves the tree was published from the registry
    generation still on disk, which is the only check that catches a registry
    rebuilt underneath a tree that still looks internally consistent.
    """
    out = pathlib.Path(out_dir)
    try:
        generation = json.loads((out / GENERATION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SessionsError(
            "The published tree is unusable: generation.json is missing or is not valid JSON."
        ) from error
    if not isinstance(generation, dict):
        raise SessionsError("The published tree is unusable: generation.json is not an object.")
    # A closed schema, because this document's whole job is proving the tree is
    # what it claims.  An added, missing or renamed key means a different writer
    # or an edit, and no digest inside the document can catch either.
    if set(generation) != set(GENERATION_KEYS) or generation["generator_version"] != (
        GENERATOR_VERSION
    ):
        raise SessionsError(
            "The published tree is unusable: generation.json is not this generator's document."
        )
    for name in (EVENTS_FILENAME, PLACEMENTS_FILENAME):
        try:
            digest = _digest_bytes(out / name)
        except OSError as error:
            raise SessionsError(
                f"The published tree is unusable: {name} is missing or cannot be read."
            ) from error
        if digest != generation.get(name):
            raise SessionsError(
                f"The published tree is inconsistent: {name} is a different generation."
            )
    try:
        current = tree_digest(out)
    except OSError as error:
        raise SessionsError("The published tree is unusable: it cannot be walked.") from error
    if current != generation.get("tree"):
        raise SessionsError(
            "The published tree is inconsistent: a session directory changed after publication."
        )
    if inventory_dir is not None:
        upstream = inventory.validate_generation(pathlib.Path(inventory_dir))
        if upstream.get("generation", {}) != generation.get("inventory"):
            raise SessionsError(
                "The published tree is stale: the registry is a different generation."
            )
    return generation


def render_summary(events: list[Event], placements: list[Placement]) -> str:
    """Return the console summary.

    Counts only.  Every name this tool handles is patient-adjacent, and a
    summary that cannot hold one cannot leak one.
    """
    cameras: dict[int, int] = {}
    for event in events:
        cameras[len(event.cameras)] = cameras.get(len(event.cameras), 0) + 1
    reasons = {}
    for placement in placements:
        if placement.placement == HELD_OUT:
            reasons[placement.placement_reason] = reasons.get(placement.placement_reason, 0) + 1
    placed = sum(1 for p in placements if p.placement == PLACED)
    unresolved = sum(1 for e in events if e.take_resolution == TAKE_UNRESOLVED)
    return "\n".join(
        [
            f"Events:      {len(events)}, camera counts {dict(sorted(cameras.items()))}",
            f"Takes:       {len(events) - unresolved} family, {unresolved} unresolved",
            f"Assets:      {len(placements)} discovered, {placed} placed, "
            f"{len(placements) - placed} held out",
            f"Hold-outs:   {dict(sorted(reasons.items()))}",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pose-estimation-sessions",
        description=(
            "Materialize the corpus registry into a discoverable tree of recording events."
        ),
    )
    parser.add_argument(
        "--inventory", default="inventory", help="Directory holding the published registry."
    )
    parser.add_argument(
        "--corpus",
        default="videos/3-cam",
        help="Corpus root the registry's relative paths resolve against.",
    )
    parser.add_argument("--out", default="sessions", help="Directory to publish the tree into.")
    parser.add_argument(
        "--strict", action="store_true", help="Return status 1 when any asset is held out."
    )
    args = parser.parse_args(argv)

    try:
        events, placements = run(args.inventory, args.corpus, args.out)
    except (SessionsError, inventory.InventoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("ERROR: the session tree could not be published.", file=sys.stderr)
        return 2
    print(render_summary(events, placements))
    if args.strict and any(p.placement == HELD_OUT for p in placements):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
