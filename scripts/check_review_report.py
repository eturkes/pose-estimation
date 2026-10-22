#!/usr/bin/env python3
"""Grade a MILESTONE-REVIEW report against the Review contract's report shape.

Usage: scripts/gate.sh python scripts/check_review_report.py <report.md>
rc=0 <=> every predicate passes. Seed (all-`unknown`) grades nonzero by P03.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROW = re.compile(r"^\|\s*(?P<id>[A-Z]\d{2})\s*\|(?P<rest>.*)\|\s*$")
FILE_LINE = re.compile(r"[\w./-]+\.(?:py|R|md|json|yaml|toml|csv):\d+")
VERDICTS = {"pass", "fail", "unknown"}


def main(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    fails: list[str] = []

    def check(ok: bool, pid: str, detail: str) -> None:
        print(f"{'pass' if ok else 'FAIL'} {pid} — {detail}")
        if not ok:
            fails.append(pid)

    for sec in ("## Verdict table", "## Detail", "## Register"):
        check(sec in text, f"P01{sec[3:6]}", f"section {sec!r} present")

    rows: dict[str, list[str]] = {}
    dupes: list[str] = []
    for line in lines:
        m = ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group("rest").split("|")]
        if m.group("id") in rows:
            dupes.append(m.group("id"))
        rows[m.group("id")] = cells
    check(bool(rows), "P02", f"{len(rows)} verdict rows parsed")
    check(not dupes, "P02b", f"row ids unique ({sorted(set(dupes))} duplicated)")

    verdicts = {
        rid: next((c for c in cells if c.lower() in VERDICTS), "") for rid, cells in rows.items()
    }
    unshaped = sorted(r for r, v in verdicts.items() if not v)
    check(not unshaped, "P02c", f"every row carries a verdict cell ({unshaped} do not)")
    unknown = sorted(r for r, v in verdicts.items() if v.lower() == "unknown")
    check(not unknown, "P03", f"{len(unknown)} rows still `unknown` ({unknown[:8]})")

    empty_finding = sorted(
        rid
        for rid, cells in rows.items()
        if verdicts.get(rid, "").lower() == "pass"
        and not any(len(c) > 12 and c.lower() not in VERDICTS for c in cells)
    )
    check(
        not empty_finding,
        "P04",
        f"every `pass` row states what was checked ({empty_finding} do not)",
    )

    failed = sorted(r for r, v in verdicts.items() if v.lower() == "fail")
    missing_detail, thin_detail = [], []
    for rid in failed:
        m = re.search(rf"^###\s+{rid}\b(?P<body>.*?)(?=^###\s|\Z)", text, re.S | re.M)
        if not m:
            missing_detail.append(rid)
            continue
        body = m.group("body")
        needed = [
            bool(FILE_LINE.search(body)),
            "predicate" in body.lower(),
            "impact" in body.lower(),
            "acceptance" in body.lower(),
        ]
        if not all(needed):
            thin_detail.append(f"{rid}{''.join('.' if n else 'X' for n in needed)}")
    check(
        not missing_detail,
        "P05",
        f"every `fail` row has a `### <id>` detail section ({missing_detail} do not)",
    )
    check(
        not thin_detail,
        "P06",
        f"every `fail` detail carries file:line + breached predicate + impact + acceptance check ({thin_detail})",
    )

    reg = text.split("## Register", 1)[1] if "## Register" in text else ""
    reg_items = [ln for ln in reg.splitlines() if ln.strip().startswith(("- ", "* "))]
    bad_reg = [ln[:40] for ln in reg_items if "acceptance" not in ln.lower()]
    check(
        not bad_reg,
        "P07",
        f"{len(reg_items)} register entries each carry an acceptance check ({len(bad_reg)} do not)",
    )

    print(
        f"\n{'PASS' if not fails else 'FAIL'} — {len(rows)} rows, {len(unknown)} unknown, {len(failed)} fail; open predicates: {fails}"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
