#!/usr/bin/env python3
"""Replay the fixed M2.5 alignment mutation catalogue."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUALIFY = "src/pose_estimation/qualify.py"
SESSIONS = "src/pose_estimation/sessions.py"
DETERMINISM = "scripts/check_qualify_determinism.py"
TEST_COMMAND = (sys.executable, "-m", "pytest", "-q", "tests/test_m2u5_mutants.py")


@dataclasses.dataclass(frozen=True)
class Patch:
    path: str
    old: str
    new: str


@dataclasses.dataclass(frozen=True)
class Mutant:
    id: str
    description: str
    patches: tuple[Patch, ...]


def patch(old: str, new: str, *, path: str = QUALIFY) -> Patch:
    return Patch(path, old, new)


def mutant(id_: str, description: str, *patches: Patch) -> Mutant:
    return Mutant(id_, description, tuple(patches))


MUTANTS = (
    mutant(
        "M01",
        "flip the accepted-edge RHS sign",
        patch(
            "        observed[index] = directed[(first, second)]",
            "        observed[index] = -directed[(first, second)]",
        ),
    ),
    mutant(
        "M02",
        "drop the reference gauge pin",
        patch(
            "    unknowns = sorted(asset_id for asset_id in component if asset_id != reference)",
            "    unknowns = sorted(component)",
        ),
    ),
    mutant(
        "M03",
        "put left above above in reference precedence",
        patch(
            'VIEW_HIERARCHY: tuple[str, ...] = ("above", "left", "right")',
            'VIEW_HIERARCHY: tuple[str, ...] = ("left", "above", "right")',
        ),
    ),
    mutant(
        "M04",
        "tie-break the reference on the highest asset id",
        patch("            return min(candidates)", "            return max(candidates)"),
    ),
    mutant(
        "M05",
        "publish a nonzero reference offset",
        patch(
            '                    "offset_s": _decimal(solved.get(asset_id)),',
            '                    "offset_s": _decimal(1e-9 if asset_id == reference else solved.get(asset_id)),',
        ),
    ),
    mutant(
        "M06",
        "replace all-edge least squares with breadth-first accumulation",
        patch(
            """    unknowns = sorted(asset_id for asset_id in component if asset_id != reference)
    if not unknowns:
        return {reference: 0.0}
    column = {asset_id: index for index, asset_id in enumerate(unknowns)}
    # Each undirected edge once, in a fixed order, so the design matrix is a
    # function of the component rather than of dict insertion order.
    edges = sorted(pair for pair in directed if pair[0] < pair[1] and set(pair) <= component)
    design = np.zeros((len(edges), len(unknowns)), dtype=np.float64)
    observed = np.empty(len(edges), dtype=np.float64)
    for index, (first, second) in enumerate(edges):
        if first != reference:
            design[index, column[first]] = -1.0
        if second != reference:
            design[index, column[second]] = 1.0
        observed[index] = directed[(first, second)]
    solution, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    if rank != len(unknowns):
        # An incidence system over a connected component is rank-deficient by
        # exactly 1, and pinning the reference removes it, so full column rank
        # here is a theorem.  A short rank means the component search and the
        # edge set disagree about connectivity, which is a defect in this file
        # rather than a property of the corpus.
        raise QualifyError(
            f"The alignment system for {reference} is rank {rank} against "
            f"{len(unknowns)} unknowns over a component this tool called connected.",
            reason="alignment_rank",
        )
    # `+ 0.0` normalises a negative zero the solver can return, which would
    # otherwise publish "-0.000000000" and make the bytes depend on rounding
    # rather than on the corpus.
    return {
        reference: 0.0,
        **{name: float(value) + 0.0 for name, value in zip(unknowns, solution, strict=True)},
    }""",
            """    solved = {reference: 0.0}
    frontier = [reference]
    while frontier:
        current = frontier.pop(0)
        for other in sorted(component):
            if other in solved or (current, other) not in directed:
                continue
            solved[other] = solved[current] + directed[(current, other)]
            frontier.append(other)
    return solved""",
        ),
    ),
    mutant(
        "M07",
        "accept visual-only edges",
        patch(
            "QUALIFIED_PAIR_STATUSES: frozenset[str] = frozenset({PAIR_OK_CORROBORATED, PAIR_OK_UNCORROBORATED})",
            "QUALIFIED_PAIR_STATUSES: frozenset[str] = frozenset({PAIR_OK_CORROBORATED, PAIR_OK_UNCORROBORATED, PAIR_VISUAL_ONLY})",
        ),
    ),
    mutant(
        "M08",
        "accept contradicted edges",
        patch(
            "QUALIFIED_PAIR_STATUSES: frozenset[str] = frozenset({PAIR_OK_CORROBORATED, PAIR_OK_UNCORROBORATED})",
            "QUALIFIED_PAIR_STATUSES: frozenset[str] = frozenset({PAIR_OK_CORROBORATED, PAIR_OK_UNCORROBORATED, PAIR_CONTRADICTED})",
        ),
    ),
    mutant(
        "M09",
        "let peak_rms weight the edge RHS",
        patch(
            '            offset = float(row["offset_s"])',
            '            offset = float(row["offset_s"]) * float(row["peak_rms"])',
        ),
    ),
    mutant(
        "M10",
        "derive event span from pair offsets",
        patch(
            '        solved_offsets = offsets_by_event.get(event["event_id"], [])',
            """        solved_offsets = [
            value
            for (first, second), value in directed.items()
            if first < second and first in members and second in members
        ]""",
        ),
    ),
    mutant(
        "M11",
        "omit unreachable camera rows",
        patch(
            """        for asset_id in members:
            if not sync_measured:""",
            """        for asset_id in solved:
            if not sync_measured:""",
        ),
    ),
    mutant(
        "M12",
        "spell an unreachable offset as zero",
        patch(
            '                    "offset_s": _decimal(solved.get(asset_id)),',
            '                    "offset_s": _decimal(solved.get(asset_id, 0.0)),',
        ),
    ),
    mutant(
        "M13",
        "call the reference component graph-connected",
        patch(
            '        connected = not unreachable_by_event.get(event["event_id"], 0)',
            '        connected = bool(offsets_by_event.get(event["event_id"]))',
        ),
    ),
    mutant(
        "M14",
        "skip the v3-to-v4 generator bump",
        patch('GENERATOR_VERSION = "v4"', 'GENERATOR_VERSION = "v3"'),
    ),
    mutant(
        "M15",
        "omit cameras_qc from CSV_FILENAMES",
        patch(
            """CSV_FILENAMES: tuple[str, ...] = (
    ASSETS_QC_FILENAME,
    PAIRS_QC_FILENAME,
    CAMERAS_QC_FILENAME,
    EVENTS_QC_FILENAME,
)""",
            """CSV_FILENAMES: tuple[str, ...] = (
    ASSETS_QC_FILENAME,
    PAIRS_QC_FILENAME,
    EVENTS_QC_FILENAME,
)""",
        ),
        patch(
            "        CAMERAS_QC_FILENAME: CAMERAS_QC_COLUMNS,\n",
            "",
        ),
    ),
    mutant(
        "M16",
        "omit cameras_qc from GENERATION_KEYS",
        patch(
            """GENERATION_KEYS: tuple[str, ...] = (
    *CSV_FILENAMES,""",
            """GENERATION_KEYS: tuple[str, ...] = (
    *(name for name in CSV_FILENAMES if name != CAMERAS_QC_FILENAME),""",
        ),
    ),
    mutant(
        "M17",
        "use prefix match for cell alphabets",
        patch(
            "            if cell and not pattern.fullmatch(cell):",
            "            if cell and not pattern.match(cell):",
        ),
    ),
    mutant(
        "M18",
        "write a nonzero legacy manifest trim",
        patch(
            '                "sync_offset": 0,', '                "sync_offset": 1,', path=SESSIONS
        ),
    ),
    mutant(
        "M19",
        "discard the reference component when the event is unconnected",
        patch(
            """            solved = _solve_offsets(_reached(members, directed, reference), directed, reference)""",
            '''            solved = _solve_offsets(_reached(members, directed, reference), directed, reference)
            if len(solved) != len(members):
                solved = {}
                reference = ""''',
        ),
    ),
    mutant(
        "M20",
        "solve every connected component under a separate gauge",
        patch(
            """            reference = _view_reference(members, views)
            solved = _solve_offsets(_reached(members, directed, reference), directed, reference)""",
            """            reference = _view_reference(members, views)
            remaining = set(members)
            while remaining:
                root = reference if reference in remaining else min(remaining)
                component = _reached(members, directed, root)
                solved.update(_solve_offsets(component, directed, root))
                remaining -= component""",
        ),
    ),
    mutant(
        "M21",
        "name each camera as its own reference",
        patch(
            '                    "reference_camera": camera_names.get(reference, "") if reference else "",',
            '                    "reference_camera": camera_names.get(asset_id, ""),',
        ),
    ),
    mutant(
        "M22",
        "average audio and visual corroborated offsets",
        patch(
            '            offset = float(row["offset_s"])',
            """            offset = (
                (float(row["offset_s"]) + float(row["offset_visual_s"])) / 2
                if row["status"] == PAIR_OK_CORROBORATED
                else float(row["offset_s"])
            )""",
        ),
    ),
    mutant(
        "M23",
        "derive closure from solved camera offsets",
        patch(
            """                    abs(directed[triangle[0]] + directed[triangle[1]] - directed[triangle[2]])""",
            """                    abs(
                        (solved_offsets[1] - solved_offsets[0])
                        + (solved_offsets[2] - solved_offsets[1])
                        - (solved_offsets[2] - solved_offsets[0])
                    )""",
        ),
    ),
    mutant(
        "M24",
        "publish the view token as camera_name",
        patch(
            '                    "camera_name": camera_names.get(asset_id, ""),',
            '                    "camera_name": views.get(asset_id, ""),',
        ),
    ),
    mutant(
        "M25",
        "omit cameras_qc from determinism hashes",
        patch(
            """ARTIFACTS = (
    "assets_qc.csv",
    "pairs_qc.csv",
    "events_qc.csv",
    "cameras_qc.csv",
    "qualification.json",
)""",
            """ARTIFACTS = (
    "assets_qc.csv",
    "pairs_qc.csv",
    "events_qc.csv",
    "qualification.json",
)""",
            path=DETERMINISM,
        ),
    ),
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_paths() -> tuple[str, ...]:
    return tuple(sorted({change.path for item in MUTANTS for change in item.patches}))


def baseline_files() -> dict[str, bytes]:
    return {path: (ROOT / path).read_bytes() for path in source_paths()}


def validate_catalog(originals: dict[str, bytes]) -> None:
    seen: set[str] = set()
    for item in MUTANTS:
        if item.id in seen:
            raise SystemExit(f"duplicate mutant id: {item.id}")
        seen.add(item.id)
        if not item.patches:
            raise SystemExit(f"mutant has no patches: {item.id}")
        texts = {path: raw.decode("utf-8") for path, raw in originals.items()}
        touched: set[str] = set()
        for change in item.patches:
            count = texts[change.path].count(change.old)
            if count != 1:
                raise SystemExit(
                    f"{item.id}: expected one occurrence in {change.path}, found {count}"
                )
            texts[change.path] = texts[change.path].replace(change.old, change.new, 1)
            touched.add(change.path)
        if all(texts[path].encode("utf-8") == originals[path] for path in touched):
            raise SystemExit(f"mutant does not change bytes: {item.id}")


def run_tests() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    return subprocess.run(
        TEST_COMMAND,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def failure_nodeids(completed: subprocess.CompletedProcess[str]) -> list[str]:
    output = completed.stdout + "\n" + completed.stderr
    found = re.findall(r"^(?:FAILED|ERROR) (tests/[^\s]+)", output, flags=re.MULTILINE)
    return list(dict.fromkeys(found))


def locate(item: Mutant, originals: dict[str, bytes]) -> str:
    first = item.patches[0]
    text = originals[first.path].decode("utf-8")
    line = text[: text.index(first.old)].count("\n") + 1
    return f"{first.path}:{line}"


def apply(item: Mutant, originals: dict[str, bytes]) -> None:
    texts = {path: raw.decode("utf-8") for path, raw in originals.items()}
    for change in item.patches:
        texts[change.path] = texts[change.path].replace(change.old, change.new, 1)
    for path, text in texts.items():
        (ROOT / path).write_text(text, encoding="utf-8", newline="")


def restore(originals: dict[str, bytes]) -> None:
    for path, raw in originals.items():
        (ROOT / path).write_bytes(raw)


def parse_only(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--check", action="store_true", help="validate the catalogue only")
    result.add_argument("--only", help="comma-separated mutant IDs")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    originals = baseline_files()
    validate_catalog(originals)
    if args.check:
        print(f"catalog_ok={len(MUTANTS)}")
        return 0

    selected_ids = parse_only(args.only)
    if selected_ids is not None:
        unknown = selected_ids - {item.id for item in MUTANTS}
        if unknown:
            raise SystemExit(f"unknown mutant ids: {','.join(sorted(unknown))}")
    selected = [item for item in MUTANTS if selected_ids is None or item.id in selected_ids]

    baseline = run_tests()
    if baseline.returncode:
        sys.stdout.write(baseline.stdout)
        sys.stderr.write(baseline.stderr)
        raise SystemExit("focused baseline is not green")

    target_hashes = {path: sha256(raw) for path, raw in originals.items()}
    survivors: list[str] = []
    try:
        for item in selected:
            apply(item, originals)
            completed = run_tests()
            nodeids = failure_nodeids(completed)
            verdict = "killed" if completed.returncode else "SURVIVED"
            if not completed.returncode:
                survivors.append(item.id)
            evidence = ",".join(nodeids) if nodeids else f"pytest-exit:{completed.returncode}"
            print(f"{item.id} {verdict} {locate(item, originals)} {evidence}", flush=True)
            restore(originals)
    finally:
        restore(originals)

    restored_hashes = {path: sha256((ROOT / path).read_bytes()) for path in originals}
    if restored_hashes != target_hashes:
        raise SystemExit("source restoration failed")
    final_baseline = run_tests()
    if final_baseline.returncode:
        sys.stdout.write(final_baseline.stdout)
        sys.stderr.write(final_baseline.stderr)
        raise SystemExit("focused baseline failed after restoration")
    print(
        f"mutants={len(selected)} killed={len(selected) - len(survivors)} survived={len(survivors)}"
    )
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
