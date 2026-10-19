#!/usr/bin/env python3
"""Vendor the browser assets: IBM Plex subsets and the Plotly cartesian bundle.

The UI serves everything from disk, so it runs with no network and no CDN.  The
Japanese faces are subset to the characters this UI can actually render — the UI
strings plus every `ja` descriptor label the cohort export publishes — because a
complete JP face is tens of megabytes and a face subset to somebody else's
charset renders tofu.

Regenerate (prototype environment, dev group; needs network)::

    uv run --directory tools/review-ui python tools/build_assets.py
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

import yaml
from fontTools import subset
from fontTools.ttLib import TTFont

PROTOTYPE = Path(__file__).resolve().parents[1]
STATIC = PROTOTYPE / "review_ui" / "static"
CACHE = PROTOTYPE / ".cache"
DESCRIPTORS = PROTOTYPE.parent.parent / "cohort" / "descriptors.yaml"
REGISTRY = "https://registry.npmjs.org"

PACKAGES = {
    "@ibm/plex-sans": ("plex-sans", "1.1.0"),
    "@ibm/plex-mono": ("plex-mono", "1.1.0"),
    "@ibm/plex-sans-jp": ("plex-sans-jp", "3.0.0"),
    "plotly.js-cartesian-dist-min": ("plotly.js-cartesian-dist-min", "4.0.0"),
}
FACES = {
    "@ibm/plex-sans": {
        "plex-sans-400.woff2": "package/fonts/complete/woff2/IBMPlexSans-Regular.woff2",
        "plex-sans-600.woff2": "package/fonts/complete/woff2/IBMPlexSans-SemiBold.woff2",
    },
    "@ibm/plex-mono": {
        "plex-mono-400.woff2": "package/fonts/complete/woff2/IBMPlexMono-Regular.woff2",
    },
    "@ibm/plex-sans-jp": {
        "plex-sans-jp-400.woff2": "package/fonts/complete/woff2/hinted/IBMPlexSansJP-Regular.woff2",
        "plex-sans-jp-600.woff2": "package/fonts/complete/woff2/hinted/IBMPlexSansJP-SemiBold.woff2",
    },
}
#: Everything the UI can compose beyond its own strings: ASCII, the numeric and
#: unit punctuation the tables print, and Japanese punctuation.
BASE_CHARSET = (
    "".join(chr(code) for code in range(0x20, 0x7F)) + "×・…−–—°％±≥≤→←↔　、。「」（）【】〜"
)


def _tarball(name: str) -> tarfile.TarFile:
    base, version = PACKAGES[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{base}-{version}.tgz"
    if not path.is_file():
        url = f"{REGISTRY}/{name}/-/{base}-{version}.tgz"
        print(f"fetch {url}")
        with urllib.request.urlopen(url, timeout=600) as response:
            path.write_bytes(response.read())
    return tarfile.open(path, "r:gz")


def _member(archive: tarfile.TarFile, name: str) -> bytes:
    handle = archive.extractfile(name)
    if handle is None:
        raise SystemExit(f"missing {name} in archive")
    return handle.read()


def charset() -> str:
    """Every character the UI can put on screen, from its own data.

    Derived rather than typed: a new label in `strings.json` or a republished
    descriptor set changes the subset instead of silently reaching a glyph the
    face does not carry.
    """
    text = BASE_CHARSET
    strings = json.loads((STATIC / "strings.json").read_text(encoding="utf-8"))
    for entry in strings.values():
        text += entry["ja"] + entry["en"]
    # Gitignored + published by the cohort run: a subset built without it renders every
    # descriptor label as tofu, so its absence refuses rather than degrades.
    if not DESCRIPTORS.is_file():
        raise SystemExit(
            f"refused: {DESCRIPTORS.relative_to(PROTOTYPE.parent.parent)} absent — publish the "
            "cohort first (`.agent/spec.md` Artifacts); the JP subset needs its labels"
        )
    parsed = yaml.safe_load(DESCRIPTORS.read_text(encoding="utf-8")) or {}
    for column in parsed.get("columns", []):
        text += f"{column.get('ja', '')}{column.get('en', '')}{column.get('unit', '')}"
    return text


def build_fonts(text: str) -> None:
    out = STATIC / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    options = subset.Options(layout_features=["*"], flavor="woff2", desubroutinize=True)
    for package, faces in FACES.items():
        with _tarball(package) as archive:
            for filename, member in faces.items():
                font = TTFont(io.BytesIO(_member(archive, member)))
                subsetter = subset.Subsetter(options=options)
                subsetter.populate(text=text)
                subsetter.subset(font)
                font.flavor = "woff2"
                font.save(out / filename)
                print(f"  {filename} {(out / filename).stat().st_size} B")


def build_plotly() -> None:
    out = STATIC / "vendor"
    out.mkdir(parents=True, exist_ok=True)
    with _tarball("plotly.js-cartesian-dist-min") as archive:
        payload = _member(archive, "package/plotly-cartesian.min.js")
    (out / "plotly-cartesian.min.js").write_bytes(payload)
    print(f"  plotly-cartesian.min.js {len(payload)} B")


def main() -> int:
    text = charset()
    print(f"charset: {len(set(text))} distinct characters")
    build_fonts(text)
    build_plotly()
    (STATIC / "vendor" / "LICENSES.md").write_text(
        "# Vendored assets\n\n"
        "- IBM Plex Sans / Sans JP / Mono (`fonts/`) — SIL Open Font License 1.1,\n"
        "  IBM Corp. Subset from the `@ibm/plex-*` npm packages by `tools/build_assets.py`.\n"
        "- Plotly.js cartesian bundle (`vendor/plotly-cartesian.min.js`) — MIT License,\n"
        "  Plotly, Inc. Taken unmodified from the `plotly.js-cartesian-dist-min` npm package.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
