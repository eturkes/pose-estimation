#!/usr/bin/env python3
"""Recount a published cameras_qc table independently of the tool that wrote it.

P19 requires the real-corpus census to be reproduced by an independent recount.
This reparses the published CSVs with ``csv.DictReader`` alone -- it never
imports ``qualify`` -- so every number answers to the bytes on disk rather than
to the producer.  ``--oracle`` additionally diffs the table row by row against
``orc_cameras_qc.py``'s independent implementation of the same contract.

The two published populations are counted side by side on purpose.  329 counts
cameras sitting inside a graph-connected event and 355 counts cameras carrying
an offset; quoting either as the other is the defect ruling A02 corrected, and
this project has conflated a pair of same-shaped counts three times.

Usage: ``python scripts/check_cameras_qc_census.py [--qualification DIR]
[--oracle CSV] [--expect-rows N]``.  Exits 0 when every check passes.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Half the last published decimal place: two conforming solvers may disagree
# below the precision the schema renders, and nowhere above it.
TOLERANCE_S = 5e-10
SOLVED_STATUSES = frozenset({"reference", "solved"})
# Ruling A02's census over the M2 corpus.  `offsets` = 193 event references at
# exactly 0 + 162 solved non-reference cameras.  `unreachable` = 10 two-camera
# unconnected + 6 three-camera with the reference inside the accepted pair + 8
# three-camera with the reference isolated.  `in_connected` is the *other*
# population -- cameras sitting inside a graph-connected event, 58 + 74*2 + 41*3
# -- and `recovered` is what the reference-component solve wins from events that
# fail as wholes, which is the difference between the two.
EXPECTED = {
    "rows": 379,
    "events": 193,
    "offsets": 355,
    "unreachable": 24,
    "connected": 173,
    "in_connected": 329,
    "recovered": 26,
}


def table(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def census(qualification: pathlib.Path, expect: dict[str, int]) -> list[tuple[str, object, object]]:
    cameras = table(qualification / "cameras_qc.csv")
    events = table(qualification / "events_qc.csv")
    marker = json.loads((qualification / "qualification.json").read_text(encoding="utf-8"))

    by_event: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in cameras:
        by_event[row["event_id"]].append(row)
    status = collections.Counter(row["offset_status"] for row in cameras)
    connected = collections.Counter(row["graph_connected"] for row in events)
    sync = collections.Counter(row["sync_status"] or "" for row in events)
    references = [row for row in cameras if row["is_reference"] == "1"]

    # P13: every event cell recomputes from the camera rows alone.
    spans_recomputed = 0
    for row in events:
        offsets = [float(r["offset_s"]) for r in by_event[row["event_id"]] if r["offset_s"]]
        expected = f"{max(offsets) - min(offsets):.9f}" if len(offsets) > 1 else ""
        spans_recomputed += row["offset_span_s"] == expected

    names_reference = 0
    for rows in by_event.values():
        pointed = {row["reference_camera"] for row in rows}
        marked = [row["camera_name"] for row in rows if row["is_reference"] == "1"]
        names_reference += len(pointed) == 1 and len(marked) == 1 and pointed == set(marked)

    in_connected = sum(
        len(by_event[row["event_id"]]) for row in events if row["graph_connected"] == "1"
    )
    nonempty = sum(1 for row in cameras if row["offset_s"])

    return [
        # Absolute pins first: these are ruling A02's census, and a structural
        # check alone would pass on a table that is self-consistent and wrong.
        ("camera rows", len(cameras), expect["rows"]),
        ("events", len(by_event), expect["events"]),
        ("rows carrying an offset", nonempty, expect["offsets"]),
        ("unreachable rows", status["unreachable"], expect["unreachable"]),
        ("graph-connected events", connected["1"], expect["connected"]),
        ("cameras inside a graph-connected event", in_connected, expect["in_connected"]),
        ("offsets recovered beyond that population", nonempty - in_connected, expect["recovered"]),
        # Structural checks: true of any conforming table, at any census.
        (
            "rows carrying an offset match their status",
            nonempty,
            status["reference"] + status["solved"],
        ),
        ("offset_status partitions every row", sum(status.values()), len(cameras)),
        ("one reference per event", len(references), len(events)),
        ("every row names its event reference", names_reference, len(events)),
        (
            "reference offset exactly 0.000000000",
            sum(1 for row in references if row["offset_s"] == "0.000000000"),
            len(events),
        ),
        (
            "solved rows carry a number",
            sum(
                1 for row in cameras if row["offset_status"] in SOLVED_STATUSES and row["offset_s"]
            ),
            nonempty,
        ),
        (
            "unreachable and unmeasured rows carry none",
            sum(
                1
                for row in cameras
                if row["offset_status"] not in SOLVED_STATUSES and row["offset_s"]
            ),
            0,
        ),
        ("graph_connected agrees with sync_status", connected["1"], sync["connected"]),
        ("unconnected agrees with sync_status", connected["0"], sync["unconnected"]),
        ("offset_span_s recomputes from camera rows", spans_recomputed, len(events)),
        (
            "row order is (event_id, asset_id)",
            cameras == sorted(cameras, key=lambda r: (r["event_id"], r["asset_id"])),
            True,
        ),
        ("generator_version", marker["generation"]["generator_version"], "v4"),
        ("cameras_qc digested by the marker", "cameras_qc.csv" in marker["generation"], True),
        ("census publishes a cameras block", "cameras" in marker, True),
    ]


def differential(qualification: pathlib.Path, oracle: pathlib.Path) -> list[str]:
    """Compare MAIN's table with the oracle's, semantic where the two declare a difference."""
    key = lambda row: (row["event_id"], row["asset_id"])  # noqa: E731
    main = {key(row): row for row in table(qualification / "cameras_qc.csv")}
    other = {key(row): row for row in table(oracle)}
    findings: list[str] = []
    if set(main) != set(other):
        findings.append(
            f"row keys differ: main-only {len(set(main) - set(other))}, "
            f"oracle-only {len(set(other) - set(main))}"
        )
    for row_key in sorted(set(main) & set(other)):
        mine, theirs = main[row_key], other[row_key]
        findings.extend(
            f"{row_key} {column}: {mine[column]!r} vs {theirs[column]!r}"
            for column in ("camera_name", "view", "is_reference", "reference_camera")
            if mine[column] != theirs[column]
        )
        if bool(mine["offset_s"]) != bool(theirs["offset_s"]):
            findings.append(f"{row_key} emptiness: {mine['offset_s']!r} vs {theirs['offset_s']!r}")
        elif mine["offset_s"]:
            delta = abs(float(mine["offset_s"]) - float(theirs["offset_s"]))
            if delta > TOLERANCE_S:
                findings.append(f"{row_key} offset delta {delta:.3e} s exceeds {TOLERANCE_S:.0e}")
        # The oracle emits one `solved` token where this schema partitions the
        # gauge pin from an estimate, which is a declared difference, not a
        # divergence.  Every other status must match outright.
        folded = "solved" if mine["offset_status"] in SOLVED_STATUSES else mine["offset_status"]
        if folded != theirs["offset_status"]:
            findings.append(
                f"{row_key} status: {mine['offset_status']!r} vs {theirs['offset_status']!r}"
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=pathlib.Path, default=ROOT / "qualification")
    parser.add_argument(
        "--oracle", type=pathlib.Path, help="cameras_qc.csv from orc_cameras_qc.py."
    )
    # Defaults are the M2 corpus's ruled census (A01, A02).  Override them to run
    # the structural half of this check against a synthetic tree.
    for name, default in EXPECTED.items():
        parser.add_argument(f"--expect-{name.replace('_', '-')}", type=int, default=default)
    arguments = parser.parse_args(argv)
    expect = {name: getattr(arguments, f"expect_{name}") for name in EXPECTED}

    failed = 0
    for label, got, want in census(arguments.qualification, expect):
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f" != {want!r}"))

    if arguments.oracle:
        findings = differential(arguments.qualification, arguments.oracle)
        failed += len(findings)
        print(f"{'PASS' if not findings else 'FAIL'} oracle differential: {len(findings)} findings")
        for line in findings[:20]:
            print("  " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
