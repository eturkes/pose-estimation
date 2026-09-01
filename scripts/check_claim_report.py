#!/usr/bin/env python3
"""Grade the claim-bounded negative report against M2.7.3 P01-P09.

`docs/calibration_finding.md` is the human-facing home of the calibration ruling's
evidence and of the wording boundary that rides with it.  `calibration_qc` holds the
claim set as module constants and refuses a published set that drops one or
paraphrases one, but that scan covers published bytes alone -- a document restating a
claim is outside it.  This script closes that gap for the two documents that carry the
claim set in prose, so the constant stays the single source of truth.

It is the single implementation behind both the standalone command and
`tests/test_claim_report.py`, so a document can never pass one and fail the other.

Exit 0 iff every predicate is green; exit 1 with one line per unmet predicate.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Callable

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pose_estimation import calibration_qc  # noqa: E402

REPORT = REPO / "docs" / "calibration_finding.md"
TECHNICAL = REPO / "docs" / "technical" / "calibration_qc.md"
REGISTER = REPO / "docs" / "technical" / "conventions.md"
SUITE = REPO / "tests" / "test_claim_report.py"

# Every top-level directory a report path may name.  Anchoring on this set is what
# separates a repo-relative path from an arbitrary backticked identifier carrying a
# slash, so P07 tests paths and never tokens.
PATH_ROOTS = ("docs/", "scripts/", "src/", "tests/", "analysis/", "renv/")

# P08 needles.  The corpus is patient-adjacent, so a shipped document naming a capture
# family, a tree member or a media file leaks a subject identifier into a surface with
# no access control.  `videos/3-cam` is the repo's declared scope and stays sayable;
# anything below it is a subject directory.
CAPTURE_ID = re.compile(r"\bs\d{2}-(?:cap|coin|glass|key|nut|peg)-[lr]\b")
CORPUS_PATH = re.compile(r"videos/3-cam/\S")
MEDIA_FILE = re.compile(r"\.(?:mov|mp4|m4v)\b", re.IGNORECASE)


def _flatten(text: str) -> str:
    """Collapse Markdown line structure so a wrapped quotation still matches.

    A claim quoted across two lines, or behind a blockquote or list marker, is the
    same claim; a predicate that missed it would push the report toward one long
    unwrapped line for the checker's benefit rather than the reader's.
    """
    stripped = [re.sub(r"^\s*(?:>\s?|[-*+]\s|\d+\.\s)", "", line) for line in text.splitlines()]
    return re.sub(r"\s+", " ", " ".join(stripped))


def _claim_occurrences(path: pathlib.Path) -> list[tuple[int, int]]:
    flat = _flatten(path.read_text(encoding="utf-8"))
    return [
        (i, flat.count(re.sub(r"\s+", " ", claim)))
        for i, claim in enumerate(calibration_qc.CLAIMS, 1)
    ]


def _verbatim_claims(path: pathlib.Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"absent: {path.relative_to(REPO)}"
    wrong = [(i, n) for i, n in _claim_occurrences(path) if n != 1]
    if wrong:
        spelled = ", ".join(f"C{i:02d}x{n}" for i, n in wrong)
        return False, f"{path.relative_to(REPO)} claims not quoted verbatim exactly once: {spelled}"
    return (
        True,
        f"{path.relative_to(REPO)} carries all {len(calibration_qc.CLAIMS)} claims verbatim",
    )


def _p01_report_claims_verbatim() -> tuple[bool, str]:
    return _verbatim_claims(REPORT)


def _p02_technical_claims_verbatim() -> tuple[bool, str]:
    return _verbatim_claims(TECHNICAL)


def _p03_no_prohibited_paraphrase() -> tuple[bool, str]:
    """Total over both documents: no excluded span, no quoted-needle exemption.

    The report states each refused overreach by shape instead of spelling it, which is
    what lets this scan stay exemption-free.  Measured at freeze: no claim contains a
    needle, so quoting every claim and carrying no needle are compatible.
    """
    found: list[str] = []
    for path in (REPORT, TECHNICAL):
        if not path.is_file():
            return False, f"absent: {path.relative_to(REPO)}"
        folded = calibration_qc._fold(path.read_text(encoding="utf-8"))
        found += [
            f"{path.relative_to(REPO)}:{needle!r}"
            for needle in calibration_qc.PROHIBITED_PARAPHRASES
            if calibration_qc._fold(needle) in folded
        ]
    if found:
        return False, f"prohibited paraphrase present: {'; '.join(found)}"
    return (
        True,
        f"0 of {len(calibration_qc.PROHIBITED_PARAPHRASES)} prohibited paraphrases in 2 documents",
    )


def _p04_probe_scripts_named_and_present() -> tuple[bool, str]:
    if not REPORT.is_file():
        return False, f"absent: {REPORT.relative_to(REPO)}"
    text = REPORT.read_text(encoding="utf-8")
    missing = [s for s in calibration_qc.PROBE_SCRIPTS.values() if s not in text]
    absent = [
        s for s in calibration_qc.PROBE_SCRIPTS.values() if not (REPO / "scripts" / s).is_file()
    ]
    if missing or absent:
        return False, f"unnamed in report: {missing}; absent on disk: {absent}"
    return (
        True,
        f"both probe scripts named and present: {sorted(calibration_qc.PROBE_SCRIPTS.values())}",
    )


def _p05_ruling_identifiers_verbatim() -> tuple[bool, str]:
    """The two cells a human must not respell: the cause, and the arm with no outcome."""
    if not REPORT.is_file():
        return False, f"absent: {REPORT.relative_to(REPO)}"
    text = REPORT.read_text(encoding="utf-8")
    wanted = (calibration_qc.RULING["reason"], calibration_qc.RULING["unrun_arm"])
    missing = [cell for cell in wanted if cell not in text]
    if missing:
        return False, f"ruling cells not quoted verbatim: {missing}"
    return True, f"ruling cells quoted verbatim: {list(wanted)}"


def _p06_registered_in_human_register() -> tuple[bool, str]:
    """An artifact left off the inventory defaults to the agent register.

    Registration is therefore what puts the report under the rule it is written to,
    which makes its absence a silent register change rather than a missing line.
    """
    if not REGISTER.is_file():
        return False, f"absent: {REGISTER.relative_to(REPO)}"
    name = REPORT.relative_to(REPO).as_posix()
    if name not in REGISTER.read_text(encoding="utf-8"):
        return (
            False,
            f"{name} missing from the {REGISTER.relative_to(REPO)} text-register inventory",
        )
    return True, f"{name} registered in the human-facing inventory"


def _p07_named_paths_resolve() -> tuple[bool, str]:
    """Backticked repo paths and Markdown link targets both have to resolve.

    The empty set resolves vacuously, so the predicate also demands the report name at
    least one of each kind.  A checker that passes an empty document certifies nothing,
    which is the failure shape this repo has already shipped once in a row-wise check
    over a headers-only table.
    """
    if not REPORT.is_file():
        return False, f"absent: {REPORT.relative_to(REPO)}"
    text = REPORT.read_text(encoding="utf-8")
    quoted = {t for t in re.findall(r"`([^`\s]+)`", text) if t.startswith(PATH_ROOTS)}
    linked = {
        t
        for t in re.findall(r"\]\(([^)\s]+)\)", text)
        if not t.startswith(("http://", "https://", "#", "mailto:"))
    }
    if not quoted or not linked:
        return False, f"vacuous: {len(quoted)} quoted paths, {len(linked)} link targets"
    dangling = sorted(t for t in quoted if not (REPO / t).exists())
    dangling += sorted(t for t in linked if not (REPORT.parent / t).exists())
    if dangling:
        return False, f"named paths that do not resolve: {dangling}"
    return True, f"{len(quoted)} quoted paths + {len(linked)} link targets all resolve"


def _p08_no_corpus_identifier() -> tuple[bool, str]:
    if not REPORT.is_file():
        return False, f"absent: {REPORT.relative_to(REPO)}"
    text = REPORT.read_text(encoding="utf-8")
    hits = {
        "capture_id": CAPTURE_ID.findall(text),
        "corpus_path": CORPUS_PATH.findall(text),
        "media_file": MEDIA_FILE.findall(text),
    }
    named = {kind: found for kind, found in hits.items() if found}
    if named:
        return False, f"corpus identifier present: {sorted(named)}"
    return True, "no capture id, corpus path segment or media filename"


def _p09_checker_runs_from_the_suite() -> tuple[bool, str]:
    if not SUITE.is_file():
        return False, f"absent: {SUITE.relative_to(REPO)}"
    if pathlib.Path(__file__).name not in SUITE.read_text(encoding="utf-8"):
        return False, f"{SUITE.relative_to(REPO)} does not drive this script"
    return True, f"{SUITE.relative_to(REPO)} drives this script"


PREDICATES: dict[str, Callable[[], tuple[bool, str]]] = {
    "P01": _p01_report_claims_verbatim,
    "P02": _p02_technical_claims_verbatim,
    "P03": _p03_no_prohibited_paraphrase,
    "P04": _p04_probe_scripts_named_and_present,
    "P05": _p05_ruling_identifiers_verbatim,
    "P06": _p06_registered_in_human_register,
    "P07": _p07_named_paths_resolve,
    "P08": _p08_no_corpus_identifier,
    "P09": _p09_checker_runs_from_the_suite,
}


def grade() -> list[tuple[str, bool, str]]:
    results = []
    for name, predicate in PREDICATES.items():
        try:
            ok, detail = predicate()
        except Exception as error:  # a raising predicate is a failing predicate
            ok, detail = False, f"{type(error).__name__}: {error}"
        results.append((name, ok, detail))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade the claim-bounded negative report.")
    parser.add_argument("--quiet", action="store_true", help="Print failures only.")
    args = parser.parse_args(argv)
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
