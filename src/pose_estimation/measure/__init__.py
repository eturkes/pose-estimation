"""Measurement sidecar: the expensive qualification axes, measured out of band.

``qualify`` publishes its evidence set in about thirty seconds by demuxing and
never decoding.  Rigidity decodes frames, detectability runs the pose pipeline
and the sync axis decodes audio corpus-wide, so those axes cannot live inside
that run.  They are measured here and ingested there.

This is a **record, not a publication**.  ``inventory``, ``sessions`` and
``qualify`` each replace a whole tree behind a staging swap because a torn
publication would be read as complete by a consumer that cannot tell.  A torn
sidecar cannot: its only consumer validates every digest before reading a row,
so the failure mode is a refusal and the repair is a rerun of one axis.  The
contract that buys is per-axis independence — an axis costing an hour of decode
is produced, and reproduced, without touching the axes beside it.

Staleness is the real hazard here, and three rules answer it.  An axis missing
from the manifest is unmeasured.  An axis named by the manifest whose table is
absent or altered is a hard error.  A table on disk that no manifest entry
names is also a hard error, so a table left behind by a schema change can never
be read as current.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import pathlib
import re
from typing import Any

from .. import inventory
from .statuses import AUDIO_STATUSES, DRIFT_STATUSES, REQUIRED_WHEN_OK, VISUAL_STATUSES

__all__ = ["AUDIO_STATUSES", "DRIFT_STATUSES", "REQUIRED_WHEN_OK", "VISUAL_STATUSES"]

GENERATOR_VERSION = "v2"

# Every version this build ingests. A03: an axis entry asserts the axis was
# produced, and its own version is what says by which generator.
SUPPORTED_VERSIONS: frozenset[str] = frozenset({GENERATOR_VERSION})

MANIFEST_FILENAME = "measurements.json"

# Closed for the same reason qualify's is: an added or renamed key means a
# different writer, and no digest inside the document catches that.
GENERATION_KEYS: tuple[str, ...] = ("manifest", "inventory", "generator_version")

AXIS_ENTRY_KEYS: tuple[str, ...] = (
    "table",
    "sha256",
    "rows",
    "generator_version",
    "provenance",
)

ASSET_KEYS: tuple[str, ...] = ("asset_id",)
PAIR_KEYS: tuple[str, ...] = ("asset_a", "asset_b")

SYNC_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "asset_a",
    "asset_b",
    "offset_audio_s",
    "peak_rms_audio",
    "peak_ratio_audio",
    "status_audio",
    "drift_ppm",
    "drift_se",
    "offset_visual_s",
    "conf_visual",
    "peak_corr_visual",
    "status_visual",
    "overlap_s",
    "dur_a",
    "dur_b",
    "same_audio_rate",
)

RIGIDITY_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "rigidity_drift_median_px",
    "rigidity_drift_p95_px",
    "rigidity_valid_fraction",
    "rigidity_flag",
)

DETECT_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "detect_rate",
    "detect_conf_median",
    "subject_px_height_median",
)

SCALE_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "scale_ref_class",
    "scale_ref_conf",
)


@dataclasses.dataclass(frozen=True)
class Axis:
    """One independently produced measurement table.

    ``keys`` is the identity a row carries into ``qualify``: an asset id for a
    per-asset axis, an ordered pair for a per-pair one.  The order is the
    registry's own, so a sidecar row and the pair ``qualify`` enumerates are
    the same key or the sidecar is wrong.
    """

    name: str
    table: str
    columns: tuple[str, ...]
    keys: tuple[str, ...]
    # Canonical row order. Not always the key: sync rows are enumerated family
    # by family, so capture_id leads and the ordered pair breaks ties.
    order: tuple[str, ...]
    enums: dict[str, frozenset[str]] = dataclasses.field(default_factory=dict)


AXES: dict[str, Axis] = {
    axis.name: axis
    for axis in (
        Axis(
            "sync",
            "sync_pairs.csv",
            SYNC_COLUMNS,
            PAIR_KEYS,
            ("capture_id", "asset_a", "asset_b"),
            {"status_audio": AUDIO_STATUSES, "status_visual": VISUAL_STATUSES},
        ),
        Axis("rigidity", "rigidity_assets.csv", RIGIDITY_COLUMNS, ASSET_KEYS, ASSET_KEYS),
        Axis("detect", "detect_assets.csv", DETECT_COLUMNS, ASSET_KEYS, ASSET_KEYS),
        Axis("scale", "scale_assets.csv", SCALE_COLUMNS, ASSET_KEYS, ASSET_KEYS),
    )
}

TABLE_NAMES: frozenset[str] = frozenset(axis.table for axis in AXES.values())


@dataclasses.dataclass(frozen=True)
class Sidecar:
    """A validated sidecar: the manifest, and the exact table text digested.

    Reading a row goes through this object rather than the directory, because a
    digest proves nothing about bytes fetched by a second ``open``.  ``validate``
    keeps the text it hashed and ``load_axis`` parses that, so the window
    between checking and reading does not exist.
    """

    manifest: dict[str, Any]
    tables: dict[str, str]


# Populated cells only; an axis that abstained on one row publishes "".
DECIMAL_CELL = re.compile(r"-?[0-9]+\.[0-9]+")
TOKEN_CELL = re.compile(r"[a-z0-9_]+")
BOOLEAN_CELL = re.compile(r"[01]")

CELL_ALPHABETS: dict[str, re.Pattern[str]] = {
    "offset_audio_s": DECIMAL_CELL,
    "peak_rms_audio": DECIMAL_CELL,
    "peak_ratio_audio": DECIMAL_CELL,
    "status_audio": TOKEN_CELL,
    "drift_ppm": DECIMAL_CELL,
    "drift_se": DECIMAL_CELL,
    "offset_visual_s": DECIMAL_CELL,
    "conf_visual": DECIMAL_CELL,
    "peak_corr_visual": DECIMAL_CELL,
    "status_visual": TOKEN_CELL,
    "overlap_s": DECIMAL_CELL,
    "dur_a": DECIMAL_CELL,
    "dur_b": DECIMAL_CELL,
    "same_audio_rate": BOOLEAN_CELL,
    "rigidity_drift_median_px": DECIMAL_CELL,
    "rigidity_drift_p95_px": DECIMAL_CELL,
    "rigidity_valid_fraction": DECIMAL_CELL,
    "rigidity_flag": TOKEN_CELL,
    "detect_rate": DECIMAL_CELL,
    "detect_conf_median": DECIMAL_CELL,
    "subject_px_height_median": DECIMAL_CELL,
    "scale_ref_class": TOKEN_CELL,
    "scale_ref_conf": DECIMAL_CELL,
}


class MeasureError(Exception):
    """A measurement sidecar could not be written or could not be trusted."""

    def __init__(self, message: str, *, reason: str = "measure_error") -> None:
        super().__init__(message)
        self.reason = reason


def decimal(value: float | None) -> str:
    """Format one measured number, or leave the cell explicitly unmeasured.

    Non-finite values never reach a cell.  An estimator that abstained says so
    through its status column; a ``nan`` in a numeric cell would be read as a
    measurement by every consumer that parses the column as a float.
    """
    if value is None or value != value or value in (float("inf"), float("-inf")):
        return ""
    # Fixed notation, always: an exponent would not match DECIMAL_CELL, and a
    # cell whose spelling depends on magnitude is a determinism hazard.
    return f"{value:.9f}"


def boolean(value: bool | None) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


def _digest_bytes(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Digest the manifest exactly as published, minus its own marker."""
    body = {key: value for key, value in manifest.items() if key != "generation"}
    return hashlib.sha256(inventory.render_json(body).encode("utf-8")).hexdigest()


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse a document that states one key twice.

    Last-key-wins would let the manifest validate on one claim while its bytes
    carry another, and the self-digest cannot see the difference: it digests
    what the parser returned, not what the file said.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise MeasureError(
                f"measurements.json states {key!r} twice.", reason="manifest_duplicate_key"
            )
        seen[key] = value
    return seen


def _read_manifest(out: pathlib.Path) -> dict[str, Any] | None:
    path = out / MANIFEST_FILENAME
    # Kind before existence: a dangling symlink reports absent, and treating it
    # as a fresh sidecar would write the next manifest through it.
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise MeasureError(
            "The sidecar's measurements.json is not a regular file.",
            reason="manifest_irregular",
        )
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasureError(
            "The sidecar is unusable: measurements.json is missing or is not valid JSON.",
            reason="manifest_unreadable",
        ) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("axes"), dict):
        raise MeasureError(
            "The sidecar is unusable: measurements.json declares no axes.",
            reason="manifest_shape",
        )
    return manifest


def _assert_cells(axis: Axis, rows: list[dict[str, str]]) -> None:
    """Refuse to record a cell this sidecar cannot spell.

    ``fullmatch``, never ``match``: ``^...$`` accepts a trailing newline, which
    is how a smuggled cell survives a pattern that reads as strict.
    """
    for row in rows:
        for column in axis.columns:
            cell = row.get(column, "")
            pattern = CELL_ALPHABETS.get(column)
            if cell and pattern is not None and not pattern.fullmatch(cell):
                raise MeasureError(
                    f"{axis.table}: {column} cell {cell!r} does not match {pattern.pattern}",
                    reason="cell_alphabet",
                )
            allowed = axis.enums.get(column)
            if allowed is None:
                continue
            if not cell:
                raise MeasureError(
                    f"{axis.table}: {column} is empty; every row carries both statuses.",
                    reason="status_empty",
                )
            if cell not in allowed:
                raise MeasureError(
                    f"{axis.table}: {column} carries {cell!r}, which it never publishes.",
                    reason="status_token",
                )
        for column, required in REQUIRED_WHEN_OK.items():
            if column not in axis.columns or row.get(column) != "ok":
                continue
            missing = [name for name in required if not row.get(name)]
            if missing:
                raise MeasureError(
                    f"{axis.table}: {column} accepted the row while {', '.join(missing)} "
                    "is empty; a gate rejects an estimate, it never erases one.",
                    reason="status_cells",
                )


def _assert_keys(axis: Axis, rows: list[dict[str, str]]) -> None:
    """Refuse a duplicated key, an empty key, or a mis-ordered pair.

    Pair order is not cosmetic.  ``qualify`` enumerates pairs in ascending
    asset order, so a row written the other way round is a key that side never
    looks up, and the axis would read as having abstained on it.
    """
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row.get(name, "") for name in axis.keys)
        if not all(key):
            raise MeasureError(f"{axis.table}: a row carries an empty key.", reason="empty_key")
        if key in seen:
            raise MeasureError(
                f"{axis.table}: the key {'/'.join(axis.keys)} is duplicated.",
                reason="duplicate_key",
            )
        seen.add(key)
        if axis.keys == PAIR_KEYS:
            if key[0] == key[1]:
                raise MeasureError(f"{axis.table}: a row pairs an asset with itself.")
            if key[0] > key[1]:
                raise MeasureError(
                    f"{axis.table}: a pair is written in descending asset order.",
                    reason="pair_order",
                )


def _assert_order(axis: Axis, rows: list[dict[str, str]]) -> None:
    """Refuse rows the axis would not have enumerated in this order.

    Canonical order is what makes two runs of one axis byte-identical, so it is
    a property of the record rather than a convenience for the reader.
    """
    keys = [tuple(row.get(name, "") for name in axis.order) for row in rows]
    if keys != sorted(keys):
        raise MeasureError(
            f"{axis.table}: rows are not in {'/'.join(axis.order)} order.", reason="row_order"
        )


def write_axis(
    out_dir: str | os.PathLike[str],
    axis_name: str,
    rows: list[dict[str, str]],
    provenance: dict[str, Any],
    *,
    inventory_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Record one axis, leaving every other axis's entry exactly as it stands.

    The other entries are copied verbatim rather than recomputed.  Recomputing
    would make any axis rerun launder an edit to a table it never touched,
    which is the one thing the digests exist to catch; copying keeps a rerun of
    one axis cheap and leaves ``validate`` the sole authority on the rest.

    Refusing a manifest measured against a different registry is the same
    argument at set level: two axes keyed to two generations of asset ids can
    agree on every digest and still describe no corpus that ever existed.
    """
    if axis_name not in AXES:
        raise MeasureError(f"There is no {axis_name!r} axis.", reason="unknown_axis")
    axis = AXES[axis_name]
    _assert_cells(axis, rows)
    _assert_keys(axis, rows)
    _assert_order(axis, rows)
    upstream = inventory.validate_generation(pathlib.Path(inventory_dir)).get("generation", {})

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(out) or {"axes": {}}
    recorded = manifest.get("generation", {}).get("inventory")
    if recorded is not None and recorded != upstream:
        raise MeasureError(
            "The sidecar was measured against a different registry generation. "
            "Remove it and remeasure, rather than mixing two generations.",
            reason="upstream_changed",
        )

    inventory.write_text(out / axis.table, inventory.render_csv(axis.columns, rows))
    manifest["axes"][axis_name] = {
        "table": axis.table,
        "sha256": _digest_bytes(out / axis.table),
        "rows": len(rows),
        "generator_version": GENERATOR_VERSION,
        "provenance": provenance,
    }
    manifest["axes"] = dict(sorted(manifest["axes"].items()))
    manifest["generation"] = {
        "manifest": manifest_digest(manifest),
        "inventory": dict(upstream),
        "generator_version": GENERATOR_VERSION,
    }
    inventory.write_text(out / MANIFEST_FILENAME, inventory.render_json(manifest))
    return manifest


def validate(
    out_dir: str | os.PathLike[str],
    inventory_dir: str | os.PathLike[str] | None = None,
) -> Sidecar:
    """Return the validated sidecar at *out_dir*, or raise when it is untrustworthy.

    Every consumer calls this before reading a row.  With *inventory_dir* it
    also proves the sidecar was measured against the registry still on disk,
    which is the only check that catches an upstream rebuilt underneath a
    sidecar that still looks internally consistent.
    """
    out = pathlib.Path(out_dir)
    manifest = _read_manifest(out)
    if manifest is None:
        raise MeasureError(
            "The sidecar is unusable: it holds no measurements.json.",
            reason="manifest_absent",
        )
    generation = manifest.get("generation")
    if not isinstance(generation, dict) or set(generation) != set(GENERATION_KEYS):
        raise MeasureError(
            "The sidecar is unusable: measurements.json is not this generator's document.",
            reason="manifest_shape",
        )
    if generation["generator_version"] != GENERATOR_VERSION:
        raise MeasureError(
            "The sidecar is unusable: it was written by a different generator version.",
            reason="generator_version",
        )
    if manifest_digest(manifest) != generation["manifest"]:
        raise MeasureError(
            "The sidecar is inconsistent: measurements.json was edited after it was written.",
            reason="manifest_edited",
        )

    named: set[str] = set()
    tables: dict[str, str] = {}
    for name, entry in manifest["axes"].items():
        if name not in AXES:
            raise MeasureError(
                f"The sidecar names an axis this tool does not measure: {name!r}.",
                reason="unknown_axis",
            )
        if not isinstance(entry, dict) or set(entry) != set(AXIS_ENTRY_KEYS):
            raise MeasureError(
                f"The sidecar's {name!r} entry is not this generator's shape.",
                reason="axis_shape",
            )
        if entry["generator_version"] not in SUPPORTED_VERSIONS:
            raise MeasureError(
                f"The sidecar's {name!r} axis was written by a generator this build "
                f"does not ingest: {entry['generator_version']!r}.",
                reason="generator_version",
            )
        axis = AXES[name]
        if entry["table"] != axis.table:
            raise MeasureError(
                f"The sidecar's {name!r} entry names the wrong table.", reason="axis_table"
            )
        table = out / axis.table
        if table.is_symlink() or not table.is_file():
            raise MeasureError(
                f"The sidecar's {name!r} table is missing or is not a regular file.",
                reason="table_absent",
            )
        try:
            data = table.read_bytes()
        except OSError as error:
            raise MeasureError(
                f"The sidecar's {name!r} table cannot be read.", reason="table_unreadable"
            ) from error
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise MeasureError(
                f"The sidecar is inconsistent: the {name!r} table is a different measurement.",
                reason="table_changed",
            )
        try:
            tables[name] = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MeasureError(
                f"The sidecar's {name!r} table is not UTF-8.", reason="table_encoding"
            ) from error
        named.add(axis.table)

    # A table the manifest does not name is the staleness case the digests
    # cannot see: it is internally whole, it is simply from another run.
    for entry in sorted(out.iterdir()):
        if entry.name == MANIFEST_FILENAME:
            continue
        if entry.name in TABLE_NAMES and entry.name not in named:
            raise MeasureError(
                f"The sidecar holds a {entry.name} that its manifest does not name; "
                "it is left over from an earlier measurement.",
                reason="table_unnamed",
            )
        if entry.name not in TABLE_NAMES:
            raise MeasureError(
                f"The sidecar holds an entry this tool never writes: {entry.name}.",
                reason="foreign_entry",
            )

    if inventory_dir is not None:
        upstream = inventory.validate_generation(pathlib.Path(inventory_dir)).get("generation", {})
        if upstream != generation["inventory"]:
            raise MeasureError(
                "The sidecar is stale: the registry is a different generation.",
                reason="upstream_changed",
            )
    return Sidecar(manifest, tables)


def load_axis(sidecar: Sidecar, axis_name: str) -> dict[tuple[str, ...], dict[str, str]]:
    """Return one validated axis, indexed by its declared key.

    The read path repeats every check the write path makes.  A table this tool
    wrote is the easy case; the sidecar's whole premise is that it is produced
    independently and re-validated at use, so trusting the writer here would
    leave a hand-edited or third-party table checked on its header alone.

    An absent axis returns an empty index rather than raising: absence is the
    unmeasured state every consumer already handles, and P33 has already made
    the difference between absent and altered a decision ``validate`` owns.
    """
    entry = sidecar.manifest.get("axes", {}).get(axis_name)
    if entry is None:
        return {}
    axis = AXES[axis_name]
    reader = csv.DictReader(sidecar.tables[axis_name].splitlines(), lineterminator="\n")
    if tuple(reader.fieldnames or ()) != axis.columns:
        raise MeasureError(
            f"The sidecar's {axis_name!r} table is not this schema.", reason="axis_schema"
        )
    rows = [dict(row) for row in reader]
    # DictReader pads a short row with None and collects a long one under the
    # None key, so both survive a header check that only compares field names.
    if any(None in row or None in row.values() for row in rows):
        raise MeasureError(
            f"The sidecar's {axis_name!r} table has a row that is not its width.",
            reason="row_shape",
        )
    if len(rows) != entry["rows"]:
        raise MeasureError(
            f"The sidecar's {axis_name!r} entry declares {entry['rows']} rows "
            f"and its table carries {len(rows)}.",
            reason="row_count",
        )
    _assert_cells(axis, rows)
    _assert_keys(axis, rows)
    _assert_order(axis, rows)
    return {tuple(row[name] for name in axis.keys): row for row in rows}
