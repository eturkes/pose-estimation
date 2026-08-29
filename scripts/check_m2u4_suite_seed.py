#!/usr/bin/env python3
"""Structural validator for M2.4's red suite.

Grades `tests/test_r_timebase_truth.py` against the frozen case set of
`.agent/archive/contract-m2u4.md` §7 + `test-m2u4` phase 1. Seed state (every
case unfilled) exits non-zero; a fully encoded suite exits zero. Encoding is
what it checks, never whether the cases pass — a red case is expected to fail
at baseline `6bbd50e`, so a pass/fail run cannot serve as the fill counter.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Case ids of the phase-1 table, by class. The table holds 82 rows; its own
# prose said 80, which is a miscount that this tuple settles.
CASE_COUNTS = {1: 5, 2: 15, 3: 6, 4: 10, 5: 20, 6: 9, 7: 9, 8: 8}
CASES = [f"C{k}.{i:02d}" for k, n in CASE_COUNTS.items() for i in range(1, n + 1)]

SUITE = Path("tests/test_r_timebase_truth.py")
UNFILLED = "unwritten"
KINDS = ("red", "control")


def fn_name(case_id: str) -> str:
    return "test_" + case_id.lower().replace(".", "_")


def main() -> int:
    if not SUITE.exists():
        print(f"FAIL missing {SUITE}")
        return 1
    text = SUITE.read_text(encoding="utf-8")

    missing = [c for c in CASES if f"def {fn_name(c)}(" not in text]
    # Each case declares its A26 kind so the baseline red/green expectation is
    # machine-readable rather than inferred from the case's prose.
    kinds = dict(re.findall(r"^#\s*kind:\s*(C\d\.\d\d)\s*=\s*(\w+)", text, re.M))
    unkinded = [c for c in CASES if kinds.get(c) not in KINDS]
    unfilled = [c for c in CASES if re.search(
        rf"def {fn_name(c)}\(.*?(?=\ndef |\Z)", text, re.S) and UNFILLED in (
        re.search(rf"def {fn_name(c)}\(.*?(?=\ndef |\Z)", text, re.S).group(0))]
    extra = sorted(set(re.findall(r"^def (test_c\d_\d\d)\(", text, re.M))
                   - {fn_name(c) for c in CASES})

    for label, rows in (("missing", missing), ("unfilled", unfilled),
                        ("unkinded", unkinded), ("unknown-id", extra)):
        if rows:
            print(f"FAIL {label} {len(rows)}: {' '.join(map(str, rows[:12]))}"
                  f"{' …' if len(rows) > 12 else ''}")
    total_bad = len(missing) + len(unfilled) + len(unkinded) + len(extra)
    print(f"{len(CASES) - len(missing) - len(unfilled)}/{len(CASES)} encoded, "
          f"{len([c for c in CASES if kinds.get(c) == 'red'])} red, "
          f"{len([c for c in CASES if kinds.get(c) == 'control'])} control")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
