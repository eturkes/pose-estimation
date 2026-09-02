"""Corpus-run manifest — the total per-asset disposition record (M2.8.2 D06).

A pass over the whole corpus fails partially by nature: a source can refuse to
decode, an event's clinical pass can exit non-zero, a registry row can name an
asset the session tree places nowhere.  Each of those is a *row* here and never
an absent one, because an asset silently missing from a denominator is the
defect this artifact exists to prevent.

The vocabulary is frozen in ``ASSET_DISPOSITIONS``, which the writer, the
validator and any downstream gate all read.  A set that lives only in a
contract is a stale number waiting to happen, so nothing restates it.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

DISPOSITION_OK = "ok"

#: Frozen manifest vocabulary (D06).  A new failure mode earns a code here
#: rather than an absent row, and the order is the published count order.
ASSET_DISPOSITIONS: tuple[str, ...] = (
    DISPOSITION_OK,
    "not_placed",
    "not_run",
    "run_failed",
    "clinical_failed",
    "no_landmarks",
)

MANIFEST_FIELDS: tuple[str, ...] = ("asset_id", "event_id", "camera_name", "disposition")
MANIFEST_FILENAME = "run_manifest.csv"

#: D05: an event is due exactly when this file is absent or records a failure.
#: Output presence is the forbidden policy — a killed run leaves a partial CSV
#: no row count can distinguish from a complete one, because the true count is
#: unknown until the source is fully decoded.
MARKER_FILENAME = "event_complete.json"
MARKER_COMPLETE = "complete"
MARKER_FAILED = "failed"

#: The pass a marker's failure came from, so one code never covers both.
STAGE_RUN = "run"
STAGE_CLINICAL = "clinical"


class ManifestError(RuntimeError):
    """A corpus-run manifest violates D06's totality or its vocabulary."""


def marker_path(event_out: Path) -> Path:
    return event_out / MARKER_FILENAME


def read_marker(event_out: Path) -> dict[str, str] | None:
    """Return the event's completion record, or ``None`` when it has none."""
    path = marker_path(event_out)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def is_complete(event_out: Path) -> bool:
    marker = read_marker(event_out)
    return bool(marker) and marker.get("status") == MARKER_COMPLETE


def write_marker(event_out: Path, **fields: object) -> None:
    """Record the attempt's outcome.  Written only after the outputs are final."""
    event_out.mkdir(parents=True, exist_ok=True)
    marker_path(event_out).write_text(
        json.dumps(fields, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def asset_disposition(event_out: Path, camera_name: str) -> str:
    """The D06 partition rule: one published code per asset, read from the marker.

    The event is the isolation grain because R is invoked once per event
    directory and its ``stop()`` ends that process, so one rejected asset takes
    the event's whole clinical pass with it.  Every asset of a failed event then
    lands on a published failure code rather than on nothing, which is what
    keeps the loss recorded instead of silent (P10).
    """
    marker = read_marker(event_out)
    if marker is None:
        # An unattempted event is not a failed one: a partial pass must still
        # publish a total manifest, and conflating the two hides real failures.
        return "not_run"
    if marker.get("status") != MARKER_COMPLETE:
        return "clinical_failed" if marker.get("stage") == STAGE_CLINICAL else "run_failed"
    if not (event_out / f"{camera_name}.csv").is_file():
        return "no_landmarks"
    return DISPOSITION_OK


def write_manifest(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    """Publish the manifest sorted by asset id, so its bytes are a function of its rows."""
    ordered = sorted(rows, key=lambda row: row["asset_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ManifestError("the manifest header is not the frozen field set")
        return list(reader)


def validate_manifest(
    rows: Sequence[Mapping[str, str]], canonical_asset_ids: Sequence[str]
) -> dict[str, int]:
    """Enforce P07 + P08 and return the per-disposition census.

    Row-set equality is not row-set identity: a manifest that duplicates one
    asset and drops another keeps both the row count and the key set, so
    uniqueness is a separate conjunct and it is the one carrying that defect
    (A05).  Emptiness satisfies every other clause, so it is refused first.
    """
    asset_ids = [row["asset_id"] for row in rows]
    canonical = set(canonical_asset_ids)
    if not rows or not canonical:
        raise ManifestError("an empty manifest is not a total partition")
    if len(asset_ids) != len(canonical_asset_ids):
        raise ManifestError("the manifest row count does not equal the canonical asset count")
    if len(asset_ids) != len(set(asset_ids)):
        raise ManifestError("the manifest repeats an asset")
    if set(asset_ids) != canonical:
        raise ManifestError("the manifest key set is not the canonical asset set")
    census = Counter(row["disposition"] for row in rows)
    unlisted = sorted(set(census) - set(ASSET_DISPOSITIONS))
    if unlisted:
        raise ManifestError("the manifest carries a disposition outside the frozen vocabulary")
    if sum(census.values()) != len(rows):
        raise ManifestError("the disposition counts do not sum to the row count")
    return {code: census.get(code, 0) for code in ASSET_DISPOSITIONS}
