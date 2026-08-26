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
from .multicam import SESSION_FORMAT_VERSION, SESSION_MANIFEST_FILENAME
from .multicam import VIDEO_EXTENSIONS as DISCOVERABLE_EXTENSIONS

GENERATOR_VERSION = "v1"

EVENTS_FILENAME = "events.csv"
PLACEMENTS_FILENAME = "placements.csv"
GENERATION_FILENAME = "generation.json"

PLACED = "placed"
HELD_OUT = "held_out"

REASON_OK = "ok"
HOLD_OUT_REASONS: tuple[str, ...] = (
    "excluded_asset",
    "extension_not_discoverable",
    "quarantined_stem",
    "source_missing",
    "source_path_unsafe",
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
)

EVENT_ID_PATTERN = re.compile(r"^(?P<capture_id>s\d{2}-[a-z]+-[lr])_run-(?P<run_index>\d{2})$")

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
    subject_ordinal: str
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
    if real_target != real_root and not real_target.startswith(real_root + os.sep):
        raise SessionsError(
            "A listed asset resolves outside the corpus root.", reason="source_path_unsafe"
        )
    if not pathlib.Path(real_target).is_file():
        raise SessionsError("A listed asset is not a regular file.", reason="source_missing")
    return target


def _read_table(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    conflicted = {row["capture_id"] for row in captures if row["view_conflict"] == "1"}
    grammar = {row["capture_id"]: row["grammar_version"] for row in captures}

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
                raise SessionsError(
                    "A listed asset has no discoverable extension.",
                    reason="extension_not_discoverable",
                )
            resolve_source(corpus_root, relative)
        except SessionsError as error:
            held[asset_id] = error.reason or "source_missing"
            continue
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
        conflict = capture_id in conflicted
        if not conflict and len({c.view for c in cameras}) != len(cameras):
            raise SessionsError(
                "A family holds two assets in one view without a registry view conflict."
            )
        groups = [[c] for c in cameras] if conflict else [sorted(cameras, key=lambda c: c.view)]
        if len(groups) > 99:
            # The published grammar is exactly two digits, so a wider index
            # would emit an id the consumer's own pattern rejects.  Refusing
            # beats printing a key that fails the check this module advertises.
            raise SessionsError("A family needs more run indices than the two-digit grammar holds.")
        for run_index, group in enumerate(groups, start=1):
            row = by_asset[group[0].asset_id]
            event_id = f"{capture_id}_run-{run_index:02d}"
            events.append(
                Event(
                    event_id=event_id,
                    capture_id=capture_id,
                    subject_ordinal=row["subject_ordinal"],
                    task=row["task"],
                    side=row["side"],
                    run_index=run_index,
                    take_resolution=TAKE_UNRESOLVED if conflict else TAKE_FAMILY,
                    view_conflict=int(conflict),
                    grammar_version=grammar.get(capture_id, row["grammar_version"]),
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
                placement_reason=REASON_OK if event_id else held.get(asset_id, "source_missing"),
                event_id=event_id,
                camera_name=camera_name,
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
                # Unmeasured placeholder.  M2.5 owns alignment; a zero here
                # asserts nothing about starts or rates.
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
            "subject_ordinal": e.subject_ordinal,
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


def tree_digest(out_dir: pathlib.Path) -> str:
    """Digest the published tree by name, link text and file bytes.

    Inode, mtime and directory metadata are deliberately outside the digest:
    they are not a function of the registry, so including them would make a
    byte-identical regeneration read as a change.
    """
    lines: list[str] = []
    for directory in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        for entry in sorted(directory.iterdir()):
            label = f"{directory.name}/{entry.name}"
            if entry.is_symlink():
                lines.append(f"{label}\tlink\t{entry.readlink()}\n")
            else:
                lines.append(f"{label}\tfile\t{hashlib.sha256(entry.read_bytes()).hexdigest()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


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
    out = pathlib.Path(out_dir)
    census = inventory.validate_generation(inventory_path)
    if out.resolve() == inventory_path.resolve():
        raise SessionsError("The output directory must differ from the registry directory.")
    _assert_owned(out)

    assets = _read_table(inventory_path / inventory.ASSETS_FILENAME)
    captures = _read_table(inventory_path / inventory.CAPTURES_FILENAME)
    events, placements = plan(assets, captures, corpus_root=corpus_root)

    # Staging and retiring are siblings, never children: discover_sessions
    # iterates the root's children, so a half-built session directory under
    # it would be a discoverable session for as long as it existed.
    staging = out.with_name(f"{out.name}.staging.{os.getpid()}")
    retiring = out.with_name(f"{out.name}.retiring.{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(retiring, ignore_errors=True)
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
        staging.rename(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(retiring, ignore_errors=True)
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
