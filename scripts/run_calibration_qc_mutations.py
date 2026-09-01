"""Mutation campaign for the `calibration_qc/` publisher (M2.7.1).

Seed state: every mutant carries an empty patch tuple, so the catalogue check
reports it UNENCODED and the run exits 1. Fill one `patches=` tuple at a time;
a mutant is killed iff the focused oracle returns non-zero against it.

Posture matches `run_m2u5_mutations.py`: stdout plus exit status are the whole
evidence, with no committed result file. Never run this beside another job in
the same tree -- it edits `src/` in place and restores it afterwards.

`qualify.py` is never mutated. F1a proves it invokes the upstream validator by
mutating that call inside `calibration_qc.py` instead, which keeps this
campaign's evidence independent of the qualifier's own.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CALIBRATION_QC = "src/pose_estimation/calibration_qc.py"
DETERMINISM = "scripts/check_calibration_qc_determinism.py"
TEST_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "tests/test_calibration_qc_mutants.py",
)


@dataclasses.dataclass(frozen=True)
class Patch:
    """One exact byte substitution. `old` must occur exactly once when applied."""

    old: str
    new: str
    path: str = CALIBRATION_QC


@dataclasses.dataclass(frozen=True)
class Mutant:
    id: str
    description: str
    patches: tuple[Patch, ...] = ()


# Every predicate-bearing branch F1a owns. A mutant that survives is either an
# unencoded predicate or a ruled equivalent; both are reported, never silently
# dropped.
MUTANTS: tuple[Mutant, ...] = (
    Mutant("M01", "ownership accepts a marker from another GENERATOR_VERSION"),
    Mutant("M02", "ownership accepts a non-empty root carrying no marker"),
    Mutant("M03", "marker probe uses stat instead of lstat, so a symlink licenses deletion"),
    Mutant("M04", "marker probe drops the S_ISREG check"),
    Mutant("M05", "duplicate-key rejection hook removed from the marker parse"),
    Mutant("M06", "_is_own_generation stops checking the generation key set"),
    Mutant("M07", "staging promoted before the live tree is moved to retiring"),
    Mutant("M08", "_sweep_orphans becomes a no-op"),
    Mutant("M09", "failed promotion no longer restores the retiring tree"),
    Mutant("M10", "sibling pid suffix dropped, so two runs share one staging name"),
    Mutant("M11", "retiring tree removed before the swap rather than after"),
    Mutant("M12", "_is_within compares raw prefixes without the separator guard"),
    Mutant("M13", "disjointness check skips the evidence directory"),
    Mutant("M14", "disjointness check skips the probe directory"),
    Mutant("M15", "output path no longer resolved before the disjointness checks"),
    Mutant("M16", "_assert_schema_is_redaction_safe becomes a no-op"),
    Mutant("M17", "FORBIDDEN_KEY_TOKENS loses the event token"),
    Mutant("M18", "IDENTIFIER_SHAPES loses the capture-id shape"),
    Mutant("M19", "_assert_cells_carry_no_identifier becomes a no-op"),
    Mutant("M20", "D04 schema check runs after the staging tree is written"),
    Mutant("M21", "corpus row cardinality relaxed from exactly one to at least one"),
    Mutant("M22", "empty evidence table accepted"),
    Mutant("M23", "header equality relaxed to set membership, so column order floats"),
    Mutant("M24", "cell alphabets use search instead of fullmatch"),
    Mutant("M25", "_token_alphabet drops its anchors"),
    Mutant("M26", "empty-cell allowance widened from shared_fraction to every column"),
    Mutant("M27", "INTEGER_CELL uses the non-ASCII digit class"),
    Mutant("M28", "_read_capture truncation guard dropped, so a cut record is skipped"),
    Mutant("M29", "_read_capture summary guard keys on a trailing brace again"),
    Mutant("M30", "probe digest mismatch downgraded from refusal to acceptance"),
    Mutant("M31", "_assert_cited_arms stops checking REQUIRED_ARM_PREFIXES"),
    Mutant("M32", "REQUIRED_ARMS loses the permutation null"),
    Mutant("M33", "a record missing a STATISTIC_KEYS entry publishes a short row"),
    Mutant("M34", "a missing probe script no longer refuses"),
    Mutant("M35", "_canonical stops sorting, so row order follows input order"),
    Mutant("M36", "marker included in its own tree_digest"),
    Mutant("M37", "census_digest stops excluding its own key"),
    Mutant("M38", "_assert_claim_conformance becomes a no-op"),
    Mutant("M39", "claim scan drops the case fold"),
    Mutant("M40", "claim scan drops the underscore-to-space fold"),
    Mutant("M41", "qualify.validate_generation call removed from run()"),
    Mutant("M42", "PROHIBITED_PARAPHRASES published into the marker"),
)


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_catalogue(mutants: tuple[Mutant, ...]) -> list[str]:
    """Return one problem line per invalid mutant. An empty list means runnable."""
    problems: list[str] = []
    seen: set[str] = set()
    for mutant in mutants:
        if mutant.id in seen:
            problems.append(f"{mutant.id} duplicate id")
        seen.add(mutant.id)
        if not mutant.patches:
            problems.append(f"{mutant.id} UNENCODED: {mutant.description}")
            continue
        sources: dict[str, str] = {}
        for patch in mutant.patches:
            text = sources.setdefault(patch.path, (ROOT / patch.path).read_text())
            if text.count(patch.old) != 1:
                problems.append(
                    f"{mutant.id} patch matches {text.count(patch.old)}x: {patch.old!r}"
                )
                continue
            sources[patch.path] = text.replace(patch.old, patch.new)
        for path, text in sources.items():
            if text == (ROOT / path).read_text():
                problems.append(f"{mutant.id} no-op against {path}")
    return problems


def _oracle() -> int:
    return subprocess.run(TEST_COMMAND, cwd=ROOT, check=False).returncode


def _score(mutant: Mutant, originals: dict[str, str]) -> str:
    try:
        for patch in mutant.patches:
            path = ROOT / patch.path
            path.write_text(path.read_text().replace(patch.old, patch.new, 1))
        return "KILLED" if _oracle() else "SURVIVED"
    finally:
        for path, text in originals.items():
            (ROOT / path).write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="Run one mutant id.")
    args = parser.parse_args(argv)

    mutants = MUTANTS
    if args.only:
        mutants = tuple(mutant for mutant in MUTANTS if mutant.id == args.only)
        if not mutants:
            print(f"unrecognized mutant id: {args.only}")
            return 2

    problems = validate_catalogue(mutants)
    if problems:
        print("\n".join(problems))
        print(f"catalogue invalid: {len(problems)} problem(s)")
        return 1

    paths = sorted({patch.path for mutant in mutants for patch in mutant.patches} | {DETERMINISM})
    originals = {path: (ROOT / path).read_text() for path in paths}
    before = {path: _digest(ROOT / path) for path in paths}

    if _oracle():
        print("baseline is red; fix it before scoring mutants")
        return 1

    survivors: list[str] = []
    try:
        for mutant in mutants:
            verdict = _score(mutant, originals)
            print(f"{mutant.id} {verdict:<9} {mutant.description}")
            if verdict == "SURVIVED":
                survivors.append(mutant.id)
    finally:
        for path, text in originals.items():
            (ROOT / path).write_text(text)

    after = {path: _digest(ROOT / path) for path in paths}
    if after != before:
        print("source restoration failed; the tree is dirty")
        return 1
    if _oracle():
        print("post-restoration baseline is red; the tree is dirty")
        return 1

    print(
        f"{len(mutants)} mutants · {len(mutants) - len(survivors)} killed · {len(survivors)} survived"
    )
    if survivors:
        print("survivors: " + " ".join(survivors))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
