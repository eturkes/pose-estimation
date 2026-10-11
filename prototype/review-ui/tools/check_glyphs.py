#!/usr/bin/env python3
"""Every character the UI can draw has a glyph in the vendored faces it falls back through.

Coverage per weight = the union of the `Plex Sans` + `Plex Sans JP` subsets, the order
`style.css --sans` resolves them in. Charset = `build_assets.charset()`, so the check reads
the same inputs the subset was built from and refuses where the builder refuses.

    uv run --directory prototype/review-ui python tools/check_glyphs.py [--extra TEXT]

`--extra` appends characters to the charset — the firing input (`--extra 鬱` → rc=1).
"""

from __future__ import annotations

import argparse
import sys

from build_assets import STATIC, charset
from fontTools.ttLib import TTFont

FACES = {
    400: ("plex-sans-400.woff2", "plex-sans-jp-400.woff2"),
    600: ("plex-sans-600.woff2", "plex-sans-jp-600.woff2"),
}


def missing(text: str, weight: int) -> list[str]:
    covered: set[int] = set()
    for name in FACES[weight]:
        covered |= set(TTFont(STATIC / "fonts" / name).getBestCmap())
    # Whitespace controls never draw a glyph.
    return sorted({c for c in text if ord(c) not in covered and c not in "\n\t\r"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extra", default="", help="characters appended to the charset")
    args = parser.parse_args()
    text = charset() + args.extra
    total = 0
    for weight in FACES:
        gaps = missing(text, weight)
        total += len(gaps)
        listed = " ".join(f"U+{ord(c):04X}" for c in gaps)
        print(
            f"weight {weight}: {len(set(text))} code points, {len(gaps)} missing {listed}".rstrip()
        )
    print("PASS" if total == 0 else f"FAIL {total} missing")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
