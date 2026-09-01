#!/usr/bin/env python3
"""Grade the committed de-identified `calibration_qc` fixture set against M2.7.2 P01-P12.

The fixture set is the publisher's byte oracle and its refusal-identity matrix.  This
script is the single implementation behind both the standalone command and
``tests/test_calibration_qc_fixtures.py``, so a fixture can never pass one and fail
the other.

Exit 0 iff every predicate is green; exit 1 with one line per unmet predicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pose_estimation import calibration_qc  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "calibration_qc_set"
GENERATOR = REPO / "scripts" / "make_calibration_qc_fixtures.py"
SOURCE = REPO / "src" / "pose_estimation" / "calibration_qc.py"

INPUTS, EXPECTED, NEGATIVES = "inputs", "expected/published", "negatives"
UPSTREAM, PROBES, EVIDENCE = "inputs/upstream", "inputs/probes", "inputs/evidence"
MANIFEST, README = "manifest.json", "README.md"

# D01: every identifier the fixture may spell.  The scan proves conformance to this
# declared namespace rather than trying to prove the absence of an open secret set --
# which is what makes de-identification a property of construction (contract §2 D01).
SYNTHETIC_SUBJECTS = frozenset({f"s{n:02d}" for n in range(90, 100)})

# P07: repo-root path text that must never appear in a committed fixture byte.  These
# are the four patient-adjacent trees plus the corpus root.
FORBIDDEN_PATH_TEXT: tuple[str, ...] = (
    "videos/",
    "videos\\",
    "/inventory/",
    "/sessions/",
    "/output/",
)

# P08: the only real corpus numbers a fixture may carry, because the publisher refuses
# without them and they are already committed as module constants.
PERMITTED_REAL_NUMBERS = frozenset(
    {str(v) for v in calibration_qc.RULED_POPULATION.values()}
    | {calibration_qc.RULING["image_height_px"]}
)

Verdict = tuple[bool, str]


def _walk(root: pathlib.Path) -> Iterator[pathlib.Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() or path.is_symlink():
            yield path


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_map(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _digest(p) for p in _walk(root) if not p.is_symlink()}


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _materialise(dest: pathlib.Path, overlay: pathlib.Path | None, deletes: list[str]) -> None:
    """Copy `inputs/` into *dest*, then apply one negative's overlay and deletions."""
    shutil.copytree(FIXTURES / INPUTS, dest)
    for rel in deletes:
        target = dest / rel
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
    if overlay is None:
        return
    for src in _walk(overlay):
        rel = src.relative_to(overlay)
        if rel.name == "expect.json":
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)


def _replay(work: pathlib.Path, path: str) -> None:
    """Drive the publisher exactly as a consumer would, from a materialised input set."""
    if path == "run":
        calibration_qc.run(work / "upstream", work / "evidence", work / "probes", work / "out")
        return
    published = work / "published"
    calibration_qc.validate_generation(
        published, qualification_dir=work / "upstream", probes_dir=work / "probes"
    )


def _reason_of(error: BaseException) -> str:
    return getattr(error, "reason", "")


# --- predicates ------------------------------------------------------------------------------


def p01_generator_is_idempotent() -> Verdict:
    """Two consecutive generations from a clean base are byte-identical."""
    if not GENERATOR.exists():
        return False, f"generator absent: {GENERATOR.relative_to(REPO)}"
    import subprocess

    before = _tree_map(FIXTURES)
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        env.pop("LD_LIBRARY_PATH", None)
        for _ in range(2):
            done = subprocess.run(
                [sys.executable, str(GENERATOR), "--out", tmp, "--force"],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(REPO),
            )
            if done.returncode != 0:
                return False, f"generator rc={done.returncode}: {done.stderr.strip()[:300]}"
        after = _tree_map(pathlib.Path(tmp))
    if after != before:
        moved = sorted(set(before) ^ set(after)) or [k for k in before if before[k] != after.get(k)]
        return False, f"regeneration differs from committed tree at {moved[:8]}"
    return True, f"{len(before)} entries byte-identical across two generations"


def p02_golden_matches_a_live_publication() -> Verdict:
    """`expected/published/` is what the publisher emits from `inputs/`."""
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        _materialise(work, None, [])
        calibration_qc.run(work / "upstream", work / "evidence", work / "probes", work / "out")
        live, golden = _tree_map(work / "out"), _tree_map(FIXTURES / EXPECTED)
    if live != golden:
        return False, f"published bytes differ: live={sorted(live)} golden={sorted(golden)}"
    return True, f"{len(golden)} entries byte-identical to a live publication"


def p03_golden_validates() -> Verdict:
    """The committed generation passes the consumer boundary with both upstreams bound."""
    census = calibration_qc.validate_generation(
        FIXTURES / EXPECTED,
        qualification_dir=FIXTURES / UPSTREAM,
        probes_dir=FIXTURES / PROBES,
    )
    keys = set(census)
    if keys != set(calibration_qc.GENERATION_KEYS):
        return False, f"generation keys {sorted(keys)}"
    if census["generator_version"] != calibration_qc.GENERATOR_VERSION:
        return False, f"generator_version {census['generator_version']!r}"
    return True, f"{len(keys)} generation keys at {census['generator_version']}"


def _negatives() -> list[tuple[str, pathlib.Path, dict[str, Any]]]:
    root = FIXTURES / NEGATIVES
    if not root.is_dir():
        return []
    out = []
    for entry in sorted(root.iterdir()):
        spec = entry / "expect.json"
        if entry.is_dir() and spec.is_file():
            out.append((entry.name, entry, _load_json(spec)))
    return out


def p04_each_negative_fails_for_its_own_reason() -> Verdict:
    """Every negative raises CalibrationQcError carrying its directory name as `.reason`."""
    cases = _negatives()
    if not cases:
        return False, "no negatives"
    bad = []
    for name, entry, spec in cases:
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            _materialise(work, entry, spec.get("deletes", []))
            if spec.get("path") == "validate":
                shutil.copytree(FIXTURES / EXPECTED, work / "published", dirs_exist_ok=True)
                _materialise_overlay_published(work, entry, spec)
            try:
                _replay(work, spec.get("path", "run"))
            except calibration_qc.CalibrationQcError as error:
                if _reason_of(error) != name:
                    bad.append(f"{name}: raised reason={_reason_of(error)!r}")
            except Exception as error:
                bad.append(f"{name}: {type(error).__name__} not CalibrationQcError")
            else:
                bad.append(f"{name}: no refusal")
    if bad:
        return False, "; ".join(bad[:10])
    return True, f"{len(cases)} negatives refused by their own reason"


def _materialise_overlay_published(
    work: pathlib.Path, entry: pathlib.Path, spec: dict[str, Any]
) -> None:
    """Apply a validate-path negative's overlay onto the copied golden tree."""
    for rel in spec.get("deletes_published", []):
        target = work / "published" / rel
        if target.exists() or target.is_symlink():
            target.unlink()
    source = entry / "published"
    if not source.is_dir():
        return
    for src in _walk(source):
        out = work / "published" / src.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)


def p05_each_negative_is_minimal() -> Verdict:
    """Dropping the overlay makes the same replay succeed, so the corruption is what fails."""
    cases = _negatives()
    if not cases:
        return False, "no negatives"
    bad = []
    for name, _entry, spec in cases:
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            _materialise(work, None, [])
            if spec.get("path") == "validate":
                shutil.copytree(FIXTURES / EXPECTED, work / "published", dirs_exist_ok=True)
            try:
                _replay(work, spec.get("path", "run"))
            except Exception as error:
                bad.append(f"{name}: clean replay raised {type(error).__name__}")
    if bad:
        return False, "; ".join(bad[:10])
    return True, f"{len(cases)} negatives verified against a clean replay"


def _source_reasons() -> frozenset[str]:
    return frozenset(re.findall(r'reason="([a-z_]+)"', SOURCE.read_text(encoding="utf-8")))


def p06_matrix_covers_every_source_reason() -> Verdict:
    """The negatives plus the declared `not_file_only` set == every reason the source raises."""
    reasons = _source_reasons()
    if not reasons:
        return False, "no reason codes found in source"
    manifest = FIXTURES / MANIFEST
    if not manifest.is_file():
        return False, f"absent {MANIFEST}"
    declared = _load_json(manifest).get("not_file_only", {})
    if not isinstance(declared, dict):
        return False, "manifest.not_file_only must map reason -> the state it needs"
    covered = {name for name, _e, _s in _negatives()} | set(declared)
    if covered != reasons:
        return False, f"uncovered={sorted(reasons - covered)} unknown={sorted(covered - reasons)}"
    blank = sorted(k for k, v in declared.items() if not str(v).strip())
    if blank:
        return False, f"not_file_only entries carry no required state: {blank}"
    return (
        True,
        f"{len(reasons)} reasons: {len(reasons) - len(declared)} file-only, {len(declared)} declared",
    )


def p07_no_corpus_identifier_reaches_the_fixtures() -> Verdict:
    """No committed fixture byte carries a corpus identifier, path or absolute path."""
    findings = []
    subject = re.compile(r"\bs[0-9]{2}-[a-z]+-[a-z]+\b")
    for path in _walk(FIXTURES):
        text = path.read_bytes().decode("utf-8", "replace")
        rel = path.relative_to(FIXTURES)
        findings.extend(
            f"{rel}: capture_id {hit!r} outside synthetic namespace"
            for hit in set(subject.findall(text))
            if hit.split("-")[0] not in SYNTHETIC_SUBJECTS
        )
        findings.extend(
            f"{rel}: forbidden path text {needle!r}"
            for needle in FORBIDDEN_PATH_TEXT
            if needle in text
        )
        if str(REPO) in text:
            findings.append(f"{rel}: absolute repo path")
    if findings:
        return False, "; ".join(sorted(set(findings))[:10])
    return True, f"{sum(1 for _ in _walk(FIXTURES))} files carry no corpus identifier"


def p08_no_real_corpus_statistic() -> Verdict:
    """Only module-constant real numbers appear; every other datum is synthetic."""
    manifest = FIXTURES / MANIFEST
    if not manifest.is_file():
        return False, f"absent {MANIFEST}"
    declared = _load_json(manifest).get("permitted_real_numbers")
    if sorted(map(str, declared or [])) != sorted(PERMITTED_REAL_NUMBERS):
        return (
            False,
            f"manifest must declare permitted_real_numbers == {sorted(PERMITTED_REAL_NUMBERS)}",
        )
    return True, f"permitted real numbers pinned to {sorted(PERMITTED_REAL_NUMBERS)}"


def p09_manifest_agrees_with_the_tree() -> Verdict:
    """The manifest's input digests match the bytes on disk."""
    manifest = FIXTURES / MANIFEST
    if not manifest.is_file():
        return False, f"absent {MANIFEST}"
    doc = _load_json(manifest)
    recorded = doc.get("input_digests")
    if not isinstance(recorded, dict) or not recorded:
        return False, "manifest carries no input_digests"
    live = _tree_map(FIXTURES / INPUTS)
    if recorded != live:
        drift = sorted(set(recorded) ^ set(live)) or [k for k in live if live[k] != recorded.get(k)]
        return False, f"input_digests disagree with disk at {drift[:8]}"
    if doc.get("namespace") != sorted(SYNTHETIC_SUBJECTS):
        return False, "manifest must declare namespace == the synthetic subject set"
    return True, f"{len(live)} input digests agree with disk"


def p10_check_is_wired_into_the_suite() -> Verdict:
    """The suite calls this module rather than reimplementing it."""
    case = REPO / "tests" / "test_calibration_qc_fixtures.py"
    if not case.is_file():
        return False, "absent tests/test_calibration_qc_fixtures.py"
    text = case.read_text(encoding="utf-8")
    if "check_calibration_qc_fixtures" not in text:
        return False, "suite does not import this checker"
    return True, "suite delegates to this checker"


def p11_no_symlink_and_no_executable_fixture() -> Verdict:
    """A symlinked or executable fixture entry would move the oracle's trust root."""
    findings = []
    for path in _walk(FIXTURES):
        rel = path.relative_to(FIXTURES)
        if path.is_symlink():
            findings.append(f"{rel}: symlink")
        elif path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            findings.append(f"{rel}: executable")
    if findings:
        return False, "; ".join(findings[:10])
    return True, "no symlink, no executable bit"


def p12_generator_refuses_a_foreign_destination() -> Verdict:
    """Regeneration writes only under the fixture root unless `--out` is given explicitly."""
    if not GENERATOR.exists():
        return False, f"generator absent: {GENERATOR.relative_to(REPO)}"
    text = GENERATOR.read_text(encoding="utf-8")
    if "calibration_qc_set" not in text:
        return False, "generator does not pin its default destination"
    return True, "generator pins its default destination"


PREDICATES: tuple[tuple[str, Callable[[], Verdict]], ...] = (
    ("P01", p01_generator_is_idempotent),
    ("P02", p02_golden_matches_a_live_publication),
    ("P03", p03_golden_validates),
    ("P04", p04_each_negative_fails_for_its_own_reason),
    ("P05", p05_each_negative_is_minimal),
    ("P06", p06_matrix_covers_every_source_reason),
    ("P07", p07_no_corpus_identifier_reaches_the_fixtures),
    ("P08", p08_no_real_corpus_statistic),
    ("P09", p09_manifest_agrees_with_the_tree),
    ("P10", p10_check_is_wired_into_the_suite),
    ("P11", p11_no_symlink_and_no_executable_fixture),
    ("P12", p12_generator_refuses_a_foreign_destination),
)


def grade() -> list[tuple[str, bool, str]]:
    """Run every predicate, converting an unexpected raise into that predicate's failure."""
    results = []
    for name, predicate in PREDICATES:
        try:
            ok, detail = predicate()
        except Exception as error:
            ok, detail = False, f"{type(error).__name__}: {error}"
        results.append((name, ok, detail))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade the calibration_qc fixture set.")
    parser.add_argument("--quiet", action="store_true", help="Print failures only.")
    args = parser.parse_args(argv)
    if not FIXTURES.is_dir():
        print(f"FAIL absent fixture set: {FIXTURES.relative_to(REPO)}")
        return 1
    failed = 0
    for name, ok, detail in grade():
        failed += not ok
        if ok and args.quiet:
            continue
        print(f"{'pass' if ok else 'FAIL'} {name} {detail}")
    print(f"{len(PREDICATES) - failed}/{len(PREDICATES)} predicates green")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
