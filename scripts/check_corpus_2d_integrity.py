#!/usr/bin/env python3
"""Corpus 2D run integrity: re-run totality (M2.8.4 P12) + the isotropic bound (P13 evidence).

A pure reader over one run tree; the corpus lives outside the repo, so run it by hand at each
corpus re-run:  scripts/gate.sh python scripts/check_corpus_2d_integrity.py [--tree DIR]

Totality: manifest asset ids unique and set-equal to the validated registry's canonical ids; every
disposition + census key inside `corpus_run.ASSET_DISPOSITIONS`, the census summing to the row
count; the report's verdict keys set-equal to the `verdicts = {...}` literal parsed out of
`scripts/corpus_run_2d.py` by AST at check time (a line span drifts), every one true.

Isotropic bound: coordinates normalised by one `max(w, h)` scalar put an in-frame landmark at
x <= w/max(w, h) and y <= h/max(w, h). Over every `ok` asset, landmarks with `_vis` > 0.8 that exceed
either bound stay under 0.1 % per display aspect the registry carries (a per-axis normalisation
fills [0, 1] and breaches ~30-40 %). Prints counts only; rc 1 names each failing predicate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pathlib
import sys

import numpy as np

from pose_estimation import corpus_run, inventory

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "corpus_run_2d.py"
VIS_MIN = 0.8
BREACH_MAX = 0.001


def driver_verdict_keys(source: pathlib.Path = DRIVER) -> set[str]:
    """The keys of the driver's `verdicts = {...}` literal."""
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == "verdicts" for target in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            return {str(key.value) for key in node.value.keys if isinstance(key, ast.Constant)}
    raise SystemExit(f"no `verdicts = {{...}}` literal in {source.name}")


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def totality(tree: pathlib.Path, registry: pathlib.Path) -> list[tuple[str, bool, str]]:
    inventory.validate_generation(registry)
    canonical = {
        row["asset_id"]
        for row in _rows(registry / inventory.ASSETS_FILENAME)
        if row["disposition"] == "canonical"
    }
    manifest = _rows(tree / corpus_run.MANIFEST_FILENAME)
    report = json.loads((tree / "run_report.json").read_text(encoding="utf-8"))
    ids = [row["asset_id"] for row in manifest]
    census = report["manifest"]["census"]
    frozen = set(corpus_run.ASSET_DISPOSITIONS)
    keys = driver_verdict_keys()
    verdicts = report["verdicts"]
    return [
        ("manifest_ids_unique", len(set(ids)) == len(ids), f"{len(set(ids))}/{len(ids)} unique"),
        (
            "manifest_ids_set_equal_registry",
            set(ids) == canonical,
            f"{len(set(ids) & canonical)}/{len(canonical)} canonical ids shared, "
            f"{len(set(ids) - canonical)} foreign",
        ),
        (
            "dispositions_frozen",
            {row["disposition"] for row in manifest} <= frozen and set(census) <= frozen,
            f"{len(frozen)} frozen codes",
        ),
        (
            "census_sums_to_rows",
            sum(census.values()) == len(ids),
            f"census {sum(census.values())} vs {len(ids)} rows",
        ),
        (
            "verdict_keys_set_equal_driver",
            set(verdicts) == keys,
            f"{len(set(verdicts) & keys)}/{len(keys)} driver keys, "
            f"{len(set(verdicts) - keys)} unknown",
        ),
        (
            "verdicts_all_true",
            bool(verdicts) and all(value is True for value in verdicts.values()),
            f"{sum(value is True for value in verdicts.values())}/{len(verdicts)} true",
        ),
    ]


def _breaches(path: pathlib.Path, width: int, height: int) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        names = [column[:-4] for column in header if column.endswith("_vis")]
        index = {column: i for i, column in enumerate(header)}
        wanted = [index[f"{name}_{axis}"] for name in names for axis in ("x", "y", "vis")]
        person = index.get("person_idx")
        values = [
            [float(row[i] or "nan") for i in wanted]
            for row in reader
            if person is None or row[person] == "0"
        ]
    if not values:
        return 0, 0
    cells = np.asarray(values, dtype=np.float64).reshape(len(values), len(names), 3)
    xs, ys, vis = cells[..., 0], cells[..., 1], cells[..., 2]
    seen = (vis > VIS_MIN) & np.isfinite(xs) & np.isfinite(ys)
    scale = float(max(width, height))
    over = (xs[seen] > width / scale) | (ys[seen] > height / scale)
    return int(seen.sum()), int(over.sum())


def isotropic_bound(tree: pathlib.Path, registry: pathlib.Path) -> list[tuple[str, bool, str]]:
    dims = {
        row["asset_id"]: (int(row["reported_width"]), int(row["reported_height"]))
        for row in _rows(registry / inventory.ASSETS_FILENAME)
    }
    tallies: dict[str, list[int]] = {}
    for row in _rows(tree / corpus_run.MANIFEST_FILENAME):
        if row["disposition"] != "ok":
            continue
        width, height = dims[row["asset_id"]]
        aspect = "portrait" if height > width else "landscape"
        seen, over = _breaches(tree / row["event_id"] / f"{row['camera_name']}.csv", width, height)
        tally = tallies.setdefault(aspect, [0, 0, 0])
        tally[0] += 1
        tally[1] += seen
        tally[2] += over
    results = []
    for aspect, (assets, seen, over) in sorted(tallies.items()):
        rate = over / seen if seen else float("nan")
        results.append(
            (
                f"isotropic_bound_{aspect}",
                bool(seen) and rate < BREACH_MAX,
                f"{over}/{seen} vis>{VIS_MIN} landmarks over the bound ({rate:.4%}) "
                f"across {assets} assets",
            )
        )
    if not results:
        results.append(("isotropic_bound", False, "no ok asset to measure"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=pathlib.Path, default=ROOT / "output" / "corpus-2d")
    parser.add_argument("--inventory", type=pathlib.Path, default=ROOT / "inventory")
    args = parser.parse_args(argv)
    results = totality(args.tree, args.inventory) + isotropic_bound(args.tree, args.inventory)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} predicates pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
