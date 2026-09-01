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
import os
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
    Mutant(
        "M01",
        "ownership accepts a marker from another GENERATOR_VERSION",
        (
            Patch(
                '    if not isinstance(generation, dict) or generation.get("generator_version") != GENERATOR_VERSION:\n        return False',
                "    if not isinstance(generation, dict):\n        return False",
            ),
        ),
    ),
    Mutant(
        "M02",
        "ownership accepts a non-empty root carrying no marker",
        (
            Patch(
                "    if not any(out_dir.iterdir()):\n        return",
                "    if any(out_dir.iterdir()):\n        return",
            ),
        ),
    ),
    Mutant(
        "M03",
        "marker probe uses stat instead of lstat, so a symlink licenses deletion",
        (Patch("path.lstat().st_mode", "path.stat().st_mode"),),
    ),
    Mutant(
        "M04",
        "marker probe drops the S_ISREG check",
        (
            Patch(
                "    if not stat.S_ISREG(path.lstat().st_mode):",
                "    if False:",
            ),
        ),
    ),
    Mutant(
        "M05",
        "duplicate-key rejection hook removed from the marker parse",
        (
            Patch(
                '        path.read_text(encoding="utf-8"), object_pairs_hook=qualify._reject_duplicate_keys',
                '        path.read_text(encoding="utf-8")',
            ),
        ),
    ),
    Mutant(
        "M06",
        "_is_own_generation stops checking the generation key set",
        (Patch("    return set(generation) == set(GENERATION_KEYS)", "    return True"),),
    ),
    Mutant(
        "M07",
        "staging promoted before the live tree is moved to retiring",
        (
            Patch(
                "        if out.exists():\n            out.rename(retiring)",
                "        if out.exists():\n            staging.rename(out)",
            ),
            Patch(
                "        try:\n            staging.rename(out)\n        except OSError:",
                "        try:\n            out.rename(retiring)\n        except OSError:",
            ),
        ),
    ),
    Mutant(
        "M08",
        "_sweep_orphans becomes a no-op",
        (
            Patch(
                '    for sibling in out.parent.glob(f"{out.name}.*"):',
                "    for sibling in ():",
            ),
        ),
    ),
    Mutant(
        "M09",
        "failed promotion no longer restores the retiring tree",
        (Patch("            if retiring.exists() and not out.exists():", "            if False:"),),
    ),
    Mutant(
        "M10",
        "sibling pid suffix dropped, so two runs share one staging name",
        (
            Patch(
                '    staging = out.with_name(f"{out.name}.staging.{os.getpid()}")',
                '    staging = out.with_name(f"{out.name}.staging")',
            ),
            Patch(
                '    retiring = out.with_name(f"{out.name}.retiring.{os.getpid()}")',
                '    retiring = out.with_name(f"{out.name}.retiring")',
            ),
        ),
    ),
    Mutant(
        "M11",
        "retiring tree removed before the swap rather than after",
        (
            Patch(
                "        if out.exists():\n            out.rename(retiring)",
                "        if out.exists():\n            out.rename(retiring)\n            _remove(retiring)",
            ),
            Patch(
                "        _sweep_orphans(out)\n        _remove(retiring)",
                "        _sweep_orphans(out)",
            ),
        ),
    ),
    Mutant(
        "M12",
        "_is_within compares raw prefixes without the separator guard",
        (
            Patch(
                "    return child == parent or child.startswith(parent + os.sep)",
                "    return child == parent or child.startswith(parent)",
            ),
        ),
    ),
    Mutant(
        "M13",
        "disjointness check skips the evidence directory",
        (Patch('        (evidence_path, "evidence directory"),\n', ""),),
    ),
    Mutant(
        "M14",
        "disjointness check skips the probe directory",
        (Patch('        (probes_dir, "probe directory"),\n', ""),),
    ),
    Mutant(
        "M15",
        "output path no longer resolved before the disjointness checks",
        (
            Patch(
                "    out = pathlib.Path(os.path.realpath(out_dir))",
                "    out = pathlib.Path(out_dir)",
            ),
        ),
    ),
    Mutant(
        "M16",
        "_assert_schema_is_redaction_safe becomes a no-op",
        (
            Patch(
                "def _assert_schema_is_redaction_safe() -> None:\n    for table, columns in (",
                "def _assert_schema_is_redaction_safe() -> None:\n    return\n    for table, columns in (",
            ),
        ),
    ),
    Mutant(
        "M17",
        "FORBIDDEN_KEY_TOKENS loses the event token",
        (Patch('        "event",\n        "family",', '        "family",'),),
    ),
    Mutant(
        "M18",
        "IDENTIFIER_SHAPES loses the capture-id shape",
        (Patch('    re.compile(r"\\bs[0-9]{2}-[a-z]+-[lr]\\b"),\n', ""),),
    ),
    Mutant(
        "M19",
        "_assert_cells_carry_no_identifier becomes a no-op",
        (
            Patch(
                "def _assert_cells_carry_no_identifier(rows: list[dict[str, str]], filename: str) -> None:\n    for row in rows:",
                "def _assert_cells_carry_no_identifier(rows: list[dict[str, str]], filename: str) -> None:\n    return\n    for row in rows:",
            ),
        ),
    ),
    Mutant(
        "M20",
        "D04 schema check runs after the staging tree is written",
        (
            Patch(
                "\n_assert_schema_is_redaction_safe()\n\n\ndef _assert_cells_carry_no_identifier",
                "\n\n\ndef _assert_cells_carry_no_identifier",
            ),
            Patch(
                "            probes=digests,\n        )\n        if out.exists():",
                "            probes=digests,\n        )\n        _assert_schema_is_redaction_safe()\n        if out.exists():",
            ),
        ),
    ),
    Mutant(
        "M21",
        "corpus row cardinality relaxed from exactly one to at least one",
        (
            Patch(
                "    corpus_rows = [dict(RULING)]", "    corpus_rows = [dict(RULING), dict(RULING)]"
            ),
        ),
    ),
    Mutant(
        "M22",
        "empty evidence table accepted",
        (
            Patch("    if not records:", "    if False:"),
            Patch(
                'def _assert_cited_arms(arms: frozenset[str]) -> None:\n    """Refuse a capture that has lost an arm the ruling quotes by value."""\n    missing = sorted(REQUIRED_ARMS - arms)',
                'def _assert_cited_arms(arms: frozenset[str]) -> None:\n    """Refuse a capture that has lost an arm the ruling quotes by value."""\n    if not arms:\n        return\n    missing = sorted(REQUIRED_ARMS - arms)',
            ),
        ),
    ),
    Mutant(
        "M23",
        "header equality relaxed to set membership, so column order floats",
        (
            Patch(
                '            inventory.render_csv(columns, rows_by_table[name]), encoding="utf-8", newline=""',
                '            inventory.render_csv(tuple(sorted(columns)), rows_by_table[name]), encoding="utf-8", newline=""',
            ),
        ),
    ),
    Mutant(
        "M24",
        "cell alphabets use search instead of fullmatch",
        (
            Patch(
                "            if cell and not pattern.fullmatch(cell):",
                "            if cell and not pattern.search(cell):",
            ),
        ),
    ),
    Mutant(
        "M25",
        "_token_alphabet drops its anchors",
        (
            Patch(
                '    return re.compile("|".join(re.escape(token) for token in sorted(tokens)))',
                '    return re.compile(".*(?:" + "|".join(re.escape(token) for token in sorted(tokens)) + ").*")',
            ),
        ),
    ),
    Mutant(
        "M26",
        "empty-cell allowance widened from shared_fraction to every column",
        (
            Patch(
                "                    **{field: _cell(block.get(field)) for field in STATISTIC_FIELDS},",
                '                    **dict.fromkeys(STATISTIC_FIELDS, ""),',
            ),
        ),
    ),
    Mutant(
        "M27",
        "INTEGER_CELL uses the non-ASCII digit class",
        (Patch('INTEGER_CELL = re.compile(r"[0-9]+")', 'INTEGER_CELL = re.compile(r"\\d+")'),),
    ),
    Mutant(
        "M28",
        "_read_capture truncation guard dropped, so a cut record is skipped",
        (
            Patch(
                '            if line.startswith("{") and line.strip() != "{":',
                "            if False:",
            ),
        ),
    ),
    Mutant(
        "M29",
        "_read_capture summary guard keys on a trailing brace again",
        (
            Patch(
                "        try:\n            record = json.loads(line, object_pairs_hook=qualify._reject_duplicate_keys)",
                '        if not line.rstrip().endswith("}"):\n            continue\n        try:\n            record = json.loads(line, object_pairs_hook=qualify._reject_duplicate_keys)',
            ),
        ),
    ),
    Mutant(
        "M30",
        "probe digest mismatch downgraded from refusal to acceptance",
        (Patch("        if recorded != digests[probe]:", "        if False:"),),
    ),
    Mutant(
        "M31",
        "_assert_cited_arms stops checking REQUIRED_ARM_PREFIXES",
        (Patch("    for prefix in REQUIRED_ARM_PREFIXES:", "    for prefix in ():"),),
    ),
    Mutant(
        "M32",
        "REQUIRED_ARMS loses the permutation null",
        (Patch('        "REAL same view pair, keypoints permuted (null)",\n', ""),),
    ),
    Mutant(
        "M33",
        "a record missing a STATISTIC_KEYS entry publishes a short row",
        (
            Patch(
                "            if not isinstance(block, dict):",
                "            if not isinstance(block, dict):\n                continue\n            if False:",
            ),
        ),
    ),
    Mutant(
        "M34",
        "a missing probe script no longer refuses",
        (
            Patch(
                '        except OSError as error:\n            raise CalibrationQcError(\n                f"The cited probe {probe} is missing from the probe directory.",\n                reason="probe_missing",\n            ) from error',
                '        except OSError:\n            digests[probe] = ""',
            ),
        ),
    ),
    Mutant(
        "M35",
        "_canonical stops sorting, so row order follows input order",
        (
            Patch(
                "    return sorted(rows, key=lambda row: tuple(row[name] for name in key))",
                "    return rows",
            ),
        ),
    ),
    Mutant(
        "M36",
        "marker included in its own tree_digest",
        (Patch("        if entry.name != CALIBRATION_QC_FILENAME:", "        if True:"),),
    ),
    Mutant(
        "M37",
        "census_digest stops excluding its own key",
        (
            Patch(
                '            key: value for key, value in body["generation"].items() if key != "census"',
                '            key: value for key, value in body["generation"].items() if True',
            ),
        ),
    ),
    Mutant(
        "M38",
        "_assert_claim_conformance becomes a no-op",
        (
            Patch(
                "def _assert_claim_conformance(staging: pathlib.Path) -> None:\n",
                "def _assert_claim_conformance(staging: pathlib.Path) -> None:\n    return\n",
            ),
        ),
    ),
    Mutant(
        "M39",
        "claim scan drops the case fold",
        (
            Patch(
                '    folded = published.casefold().replace("_", " ")',
                '    folded = published.replace("_", " ")',
            ),
        ),
    ),
    Mutant(
        "M40",
        "claim scan drops the underscore-to-space fold",
        (
            Patch(
                '    folded = published.casefold().replace("_", " ")',
                "    folded = published.casefold()",
            ),
        ),
    ),
    Mutant(
        "M41",
        "qualify.validate_generation call removed from run()",
        (
            Patch(
                "    qualification_census = qualify.validate_generation(\n        qualification_path, sessions_dir=sessions_dir, inventory_dir=inventory_dir\n    )",
                "    qualification_census = {}",
            ),
        ),
    ),
    Mutant(
        "M42",
        "PROHIBITED_PARAPHRASES published into the marker",
        (
            Patch(
                "    _assert_claim_conformance(staging)",
                '    _assert_claim_conformance(staging)\n    census["prohibited_paraphrases"] = list(PROHIBITED_PARAPHRASES)\n    census["generation"]["census"] = census_digest(census)\n    (staging / CALIBRATION_QC_FILENAME).write_text(\n        inventory.render_json(census), encoding="utf-8", newline=""\n    )',
            ),
        ),
    ),
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


def _oracle(*, show_path: bool = False) -> int:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    if show_path:
        env["CQC_ORACLE_SHOW_PATH"] = "1"
        env["PYTEST_ADDOPTS"] = "--capture=tee-sys"
    return subprocess.run(TEST_COMMAND, cwd=ROOT, env=env, check=False).returncode


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
    parser.add_argument("--only", default=None, help="Run comma-separated mutant ids.")
    args = parser.parse_args(argv)

    mutants = MUTANTS
    if args.only:
        requested = {value.strip() for value in args.only.split(",") if value.strip()}
        known = {mutant.id for mutant in MUTANTS}
        if unknown := requested - known:
            print("unrecognized mutant id: " + ",".join(sorted(unknown)))
            return 2
        mutants = tuple(mutant for mutant in MUTANTS if mutant.id in requested)

    problems = validate_catalogue(mutants)
    if problems:
        print("\n".join(problems))
        print(f"catalogue invalid: {len(problems)} problem(s)")
        return 1

    paths = sorted({patch.path for mutant in mutants for patch in mutant.patches} | {DETERMINISM})
    originals = {path: (ROOT / path).read_text() for path in paths}
    before = {path: _digest(ROOT / path) for path in paths}

    if _oracle(show_path=True):
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
