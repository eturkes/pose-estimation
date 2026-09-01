#!/usr/bin/env python3
"""Grade the prospective capture specification against M2.7.4 P01-P13.

`docs/prospective_capture.md` specifies a future calibrated acquisition.  Nothing
computes it and no artifact carries it, so the document itself is the whole deliverable
and a defect in it is a false claim shipped to a human reader.  This script is the
consistency pass the `docs` tier owes, committed rather than performed once.

Three properties carry most of the value.  A section missing one of its five fields
cannot be operated, because a threshold with no failure action is advice.  A threshold
resting on a measured absence has to say so, because presenting a project choice as an
external standard is the defect no downstream consumer re-checks.  And a specification
for an unrun capture must not read as a record of one.

It is the single implementation behind both the standalone command and
`tests/test_prospective_capture.py`, so the document cannot pass one and fail the other.

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

SPEC = REPO / "docs" / "prospective_capture.md"
PROTOCOL = REPO / "docs" / "capture_protocol.md"
REGISTER = REPO / "docs" / "technical" / "conventions.md"
SUITE = REPO / "tests" / "test_prospective_capture.py"

# Contract section spine.  Ids and titles are frozen: a renamed section is a changed
# specification, and the checker is what makes that deliberate rather than incidental.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("S01", "Scope, estimands and claim boundary"),
    ("S02", "Document control and responsibilities"),
    ("S03", "Study design, population and sampling"),
    ("S04", "Ethics, consent and identifiable-video governance"),
    ("S05", "Task script, safety and trial schedule"),
    ("S06", "Room, lighting and scene controls"),
    ("S07", "Hardware and software inventory"),
    ("S08", "Camera layout and visibility proof"),
    ("S09", "Sensor mode and image-quality qualification"),
    ("S10", "Mechanical mounting, orientation and drift epochs"),
    ("S11", "Intrinsic calibration"),
    ("S12", "Extrinsic calibration and coordinate frame"),
    ("S13", "Metric-scale traceability"),
    ("S14", "Synchronization and rolling-shutter model"),
    ("S15", "Session, trial and provenance manifest"),
    ("S16", "Preflight, capture and postflight disposition"),
    ("S17", "Independent reference and validation acquisition"),
    ("S18", "Processing and model contract"),
    ("S19", "Acceptance statistics, uncertainty and exclusions"),
    ("S20", "Security, release, reproducibility and change control"),
)

REQUIRED_FIELDS = ("Owner", "Record", "Threshold", "Failure action", "Evidence")

# The five requirements a capture cannot trade away, each bound to the sections that
# carry it.  P03 checks the binding in both directions: the table names the sections,
# and every named section states an obligation.
NON_NEGOTIABLES: dict[str, tuple[str, ...]] = {
    "N1": ("S11", "S12"),
    "N2": ("S14",),
    "N3": ("S10",),
    "N4": ("S13",),
    "N5": ("S04", "S20"),
}

ABSENCES = ("L1", "L2", "L3", "L4", "L5")
LOCAL_DECISION = "**local decision**"

UNRUN_STATUS = "This specification defines a capture that nobody has performed."

# P11 needles.  A specification that reads as a record invites a reader to cite it as
# evidence, which is the one misreading this document cannot survive.
EXECUTION_CLAIMS = (
    "we ran",
    "we captured",
    "we measured",
    "we observed",
    "we recorded",
    "was run",
    "were run",
    "results show",
    "the capture produced",
    "data were collected",
)

MAX_SENTENCE_WORDS = 25

PATH_ROOTS = ("docs/", "scripts/", "src/", "tests/", "analysis/", "renv/")

# P08 needles, shared with the M2.7.3 report checker.  The corpus is patient-adjacent,
# so a shipped document naming a capture family or a media file leaks a subject
# identifier into a surface with no access control.
CAPTURE_ID = re.compile(r"\bs\d{2}-(?:cap|coin|glass|key|nut|peg)-[lr]\b")
CORPUS_PATH = re.compile(r"videos/3-cam/\S")
MEDIA_FILE = re.compile(r"\.(?:mov|mp4|m4v)\b", re.IGNORECASE)

SECTION_HEADING = re.compile(r"^## (S\d{2}) — (.+)$", re.MULTILINE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\[])")


def _flatten(text: str) -> str:
    """Collapse Markdown line structure so a wrapped sentence still matches.

    A sentence broken across two lines is the same sentence; a predicate that missed it
    would push the document toward one long unwrapped line for the checker's benefit
    rather than the reader's.
    """
    stripped = [re.sub(r"^\s*(?:>\s?|[-*+]\s|\d+\.\s)", "", line) for line in text.splitlines()]
    return re.sub(r"\s+", " ", " ".join(stripped))


def _spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def _sections() -> dict[str, str]:
    """Map section id to its body text, ending at the next heading of any level."""
    text = _spec_text()
    found = list(SECTION_HEADING.finditer(text))
    bodies: dict[str, str] = {}
    for i, match in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        body = text[match.end() : end]
        stop = re.search(r"^#{1,2} ", body, re.MULTILINE)
        bodies[match.group(1)] = body[: stop.start()] if stop else body
    return bodies


def _prose_sentences() -> list[str]:
    """Prose only: fences, tables, headings and the bibliography are not prose."""
    lines: list[str] = []
    in_fence = False
    for line in _spec_text().splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or line.lstrip().startswith(("|", "#")):
            continue
        if re.match(r"^\s*-\s*\[E\d+\]", line):
            continue
        lines.append(line)
    blocks = re.split(r"\n\s*\n", "\n".join(lines))
    sentences: list[str] = []
    for block in blocks:
        flat = _flatten(block).strip()
        if flat:
            sentences += [s for s in SENTENCE_SPLIT.split(flat) if s.strip()]
    return sentences


def _p01_sections_present_in_order() -> tuple[bool, str]:
    found = [(m.group(1), m.group(2).strip()) for m in SECTION_HEADING.finditer(_spec_text())]
    if found != list(SECTIONS):
        missing = [s for s in SECTIONS if s not in found]
        extra = [s for s in found if s not in SECTIONS]
        return False, f"spine mismatch; missing={missing[:3]} unexpected={extra[:3]}"
    return True, f"all {len(SECTIONS)} sections present, titles verbatim, ascending"


def _p02_every_section_carries_five_fields() -> tuple[bool, str]:
    bodies = _sections()
    broken: list[str] = []
    for sid, _ in SECTIONS:
        body = bodies.get(sid, "")
        for field in REQUIRED_FIELDS:
            match = re.search(rf"\*\*{re.escape(field)}:\*\*[ \t]*(\S[^\n]*)", body)
            if not match:
                broken.append(f"{sid}/{field}")
    if broken:
        return False, f"fields absent or empty: {broken}"
    return True, f"{len(SECTIONS)} sections x {len(REQUIRED_FIELDS)} fields all present, non-empty"


def _p03_non_negotiables_bound() -> tuple[bool, str]:
    text = _spec_text()
    bodies = _sections()
    broken: list[str] = []
    for nid, section_ids in NON_NEGOTIABLES.items():
        row = re.search(rf"^\|\s*{nid}\s*\|([^|]*)\|([^|]*)\|", text, re.MULTILINE)
        if not row:
            broken.append(f"{nid}/no-row")
            continue
        named = {t.strip() for t in re.findall(r"S\d{2}", row.group(2))}
        if named != set(section_ids):
            broken.append(f"{nid}/names={sorted(named)}!={sorted(section_ids)}")
        broken += [f"{sid}/no-MUST" for sid in section_ids if "MUST" not in bodies.get(sid, "")]
    if broken:
        return False, f"non-negotiable bindings broken: {broken}"
    bound = sum(len(v) for v in NON_NEGOTIABLES.values())
    return (
        True,
        f"{len(NON_NEGOTIABLES)} non-negotiables bound to {bound} sections, all stating MUST",
    )


def _p04_no_prohibited_paraphrase() -> tuple[bool, str]:
    """Total over the specification, no excluded span.

    Scoped to this document alone.  The needle list was written for the calibration
    ruling's claim surface, where `clinical validity` is an overreach; the same string
    appears in `capture_protocol.md` inside a sentence conceding a gap, which is the
    opposite of a claim.  Ranging a rule over a surface it was not written for buys a
    false positive, not coverage.
    """
    folded = calibration_qc._fold(_spec_text())
    found = [n for n in calibration_qc.PROHIBITED_PARAPHRASES if calibration_qc._fold(n) in folded]
    if found:
        return False, f"prohibited paraphrase present: {found}"
    total = len(calibration_qc.PROHIBITED_PARAPHRASES)
    return True, f"0 of {total} prohibited paraphrases in {SPEC.relative_to(REPO)}"


def _p05_absences_labelled_local() -> tuple[bool, str]:
    text = _spec_text()
    missing = [lid for lid in ABSENCES if lid not in text]
    labels = text.count(LOCAL_DECISION)
    if missing:
        return False, f"absence ids not stated: {missing}"
    if labels < len(ABSENCES):
        return False, f"{labels} local-decision labels for {len(ABSENCES)} absences"
    return True, f"{len(ABSENCES)} absences stated, {labels} local-decision labels"


def _p06_citations_resolvable() -> tuple[bool, str]:
    text = _spec_text()
    cited = {int(n) for n in re.findall(r"\[E(\d+)\]", text)}
    listed = {
        int(n): url
        for n, url in re.findall(r"^-\s*\[E(\d+)\][^\n]*?(https?://\S+)", text, re.MULTILINE)
    }
    if not cited or not listed:
        return False, f"vacuous: {len(cited)} citations, {len(listed)} reference entries"
    bare = sorted(cited - set(listed))
    unused = sorted(set(listed) - cited)
    if bare:
        return False, f"cited with no resolvable identifier: {[f'E{n}' for n in bare]}"
    if unused:
        return False, f"listed but never cited: {[f'E{n}' for n in unused]}"
    return True, f"{len(cited)} citations, every one carrying a URL or DOI"


def _p07_named_paths_resolve() -> tuple[bool, str]:
    """Backticked repo paths must resolve, and there must be at least one.

    The empty set resolves vacuously, so the floor is the predicate.  Prospective
    artifact names (`estimands.yaml`, `sync/<session>.json`) are deliberately outside
    `PATH_ROOTS`: they name what the future capture produces, and they must not resolve.
    """
    text = _spec_text()
    quoted = {t for t in re.findall(r"`([^`\s]+)`", text) if t.startswith(PATH_ROOTS)}
    if not quoted:
        return False, "vacuous: 0 backticked repo paths named"
    dangling = sorted(t for t in quoted if not (REPO / t).exists())
    if dangling:
        return False, f"named paths that do not resolve: {dangling}"
    return True, f"{len(quoted)} backticked repo paths all resolve"


def _p08_no_corpus_identifier() -> tuple[bool, str]:
    text = _spec_text()
    hits = {
        "capture_id": CAPTURE_ID.findall(text),
        "corpus_path": CORPUS_PATH.findall(text),
        "media_file": MEDIA_FILE.findall(text),
    }
    named = {kind: found for kind, found in hits.items() if found}
    if named:
        return False, f"corpus identifier present: {sorted(named)}"
    return True, "no capture id, corpus path segment or media filename"


def _p09_registered_in_human_register() -> tuple[bool, str]:
    """An artifact left off the inventory defaults to the agent register.

    Registration is what puts the specification under ASD-STE100, so its absence is a
    silent register change rather than a missing line.
    """
    name = SPEC.relative_to(REPO).as_posix()
    if name not in REGISTER.read_text(encoding="utf-8"):
        return False, f"{name} missing from the {REGISTER.relative_to(REPO)} inventory"
    return True, f"{name} registered in the human-facing inventory"


def _p10_scope_banner_both_ways() -> tuple[bool, str]:
    """Two overlapping normative capture documents drift when neither names the other."""
    banner = "Which document governs which capture."
    pairs = ((SPEC, PROTOCOL), (PROTOCOL, SPEC))
    broken: list[str] = []
    for doc, other in pairs:
        text = doc.read_text(encoding="utf-8")
        if banner not in text:
            broken.append(f"{doc.relative_to(REPO)}/no-banner")
        if other.relative_to(REPO).as_posix() not in text:
            broken.append(f"{doc.relative_to(REPO)}/does-not-name-{other.name}")
    if broken:
        return False, f"scope banner incomplete: {broken}"
    return True, "both documents carry the banner and name each other"


def _p11_no_execution_claim() -> tuple[bool, str]:
    flat = _flatten(_spec_text())
    folded = calibration_qc._fold(flat)
    found = [n for n in EXECUTION_CLAIMS if calibration_qc._fold(n) in folded]
    if found:
        return False, f"reads as a record of a run capture: {found}"
    stated = flat.count(UNRUN_STATUS)
    if stated != 1:
        return False, f"unrun status stated {stated} times, expected exactly 1"
    return True, f"0 of {len(EXECUTION_CLAIMS)} execution claims; unrun status stated once"


def _p12_checker_runs_from_the_suite() -> tuple[bool, str]:
    if not SUITE.is_file():
        return False, f"absent: {SUITE.relative_to(REPO)}"
    if pathlib.Path(__file__).name not in SUITE.read_text(encoding="utf-8"):
        return False, f"{SUITE.relative_to(REPO)} does not drive this script"
    return True, f"{SUITE.relative_to(REPO)} drives this script"


def _p13_sentences_within_register_bound() -> tuple[bool, str]:
    sentences = _prose_sentences()
    if not sentences:
        return False, "vacuous: 0 prose sentences found"
    long = [(len(s.split()), s[:60]) for s in sentences if len(s.split()) > MAX_SENTENCE_WORDS]
    if long:
        spelled = "; ".join(f"{n}w {t!r}" for n, t in sorted(long, reverse=True)[:5])
        return False, f"{len(long)} sentences over {MAX_SENTENCE_WORDS} words: {spelled}"
    return True, f"{len(sentences)} prose sentences, all within {MAX_SENTENCE_WORDS} words"


PREDICATES: dict[str, Callable[[], tuple[bool, str]]] = {
    "P01": _p01_sections_present_in_order,
    "P02": _p02_every_section_carries_five_fields,
    "P03": _p03_non_negotiables_bound,
    "P04": _p04_no_prohibited_paraphrase,
    "P05": _p05_absences_labelled_local,
    "P06": _p06_citations_resolvable,
    "P07": _p07_named_paths_resolve,
    "P08": _p08_no_corpus_identifier,
    "P09": _p09_registered_in_human_register,
    "P10": _p10_scope_banner_both_ways,
    "P11": _p11_no_execution_claim,
    "P12": _p12_checker_runs_from_the_suite,
    "P13": _p13_sentences_within_register_bound,
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
    parser = argparse.ArgumentParser(description="Grade the prospective capture specification.")
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
