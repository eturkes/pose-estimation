#!/usr/bin/env python3
"""Replay the M2.1 inventory mutation campaign against its focused test gate."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
INVENTORY = "src/pose_estimation/inventory.py"
VIDEO_IO = "src/pose_estimation/video_io.py"
TEST_COMMAND = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "tests/test_inventory.py",
    "tests/test_inventory_review.py",
    "tests/test_inventory_predicates.py",
)


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


def patch(old: str, new: str, *, path: str = INVENTORY) -> Patch:
    return Patch(path, old, new)


def mutant(id_: str, description: str, *patches: Patch) -> Mutant:
    return Mutant(id_, description, tuple(patches))


MUTANTS = (
    mutant(
        "M001",
        "control escapes use code points instead of each UTF-8 byte",
        patch(
            'return "".join(f"\\\\x{byte:02x}" for byte in chr(code).encode("utf-8"))',
            'return f"\\\\x{code:02x}"',
        ),
    ),
    mutant(
        "M002",
        "escape-introducer backslashes are not doubled",
        patch(
            'return _UNRENDERABLE.sub(escape, relative.replace("\\\\", "\\\\\\\\"))',
            "return _UNRENDERABLE.sub(escape, relative)",
        ),
    ),
    mutant(
        "M003",
        "surrogate escape renders its code point instead of its carried byte",
        patch('return f"\\\\x{code - 0xDC00:02x}"', 'return f"\\\\x{code:02x}"'),
    ),
    mutant(
        "M004",
        "control-byte escapes use uppercase hexadecimal",
        patch(
            'return "".join(f"\\\\x{byte:02x}" for byte in chr(code).encode("utf-8"))',
            'return "".join(f"\\\\x{byte:02X}" for byte in chr(code).encode("utf-8"))',
        ),
    ),
    mutant(
        "M005",
        "only the first escape-introducer backslash is doubled",
        patch(
            'return _UNRENDERABLE.sub(escape, relative.replace("\\\\", "\\\\\\\\"))',
            'return _UNRENDERABLE.sub(escape, relative.replace("\\\\", "\\\\\\\\", 1))',
        ),
    ),
    mutant(
        "M006",
        "control bytes render as Unicode escapes instead of byte escapes",
        patch(
            'return "".join(f"\\\\x{byte:02x}" for byte in chr(code).encode("utf-8"))',
            'return "".join(f"\\\\u{byte:04x}" for byte in chr(code).encode("utf-8"))',
        ),
    ),
    mutant(
        "M007",
        "the parse path reads entry.name instead of canonical relative text",
        patch(
            'parse = parse_stem(relative.rsplit("/", 1)[-1])',
            "parse = parse_stem(entry.name)",
        ),
    ),
    mutant(
        "M008",
        "relative path canonicalization returns locale-decoded text unchanged",
        patch('return os.fsencode(relative).decode("utf-8")', "return relative"),
    ),
    mutant(
        "M009",
        "the independent discovery walk returns locale-decoded paths",
        patch(
            "return [_relative_posix(entry, corpus_root) for entry in _iter_entries(corpus_root, skip)]",
            "return [entry.relative_to(corpus_root).as_posix() for entry in _iter_entries(corpus_root, skip)]",
        ),
    ),
    mutant(
        "M010",
        "classification runs before canonical relative-path decoding",
        patch(
            "        relative = _relative_posix(entry, corpus_root)\n"
            "        reason = _exclusion_reason(entry, root, relative)",
            "        relative = entry.relative_to(corpus_root).as_posix()\n"
            "        reason = _exclusion_reason(entry, root, relative)\n"
            "        relative = _relative_posix(entry, corpus_root)",
        ),
    ),
    mutant(
        "M011",
        "classification strictly decodes canonical text a second time",
        patch(
            "    if _CONTROL_CHARS.search(relative):",
            '    relative = os.fsencode(relative).decode("utf-8")\n'
            "    if _CONTROL_CHARS.search(relative):",
        ),
    ),
    mutant(
        "M012",
        "extension_case publishes every raw suffix",
        patch(
            'return suffix if suffix.lower() in VIDEO_EXTENSIONS else "<unsupported>"',
            "return suffix",
        ),
    ),
    mutant(
        "M013",
        "extension_case maps an absent suffix to unsupported",
        patch('return "<none>"', 'return "<unsupported>"'),
    ),
    mutant(
        "M014",
        "extension_case folds recognized suffix case",
        patch(
            'return suffix if suffix.lower() in VIDEO_EXTENSIONS else "<unsupported>"',
            'return suffix.lower() if suffix.lower() in VIDEO_EXTENSIONS else "<unsupported>"',
        ),
    ),
    mutant(
        "M015",
        "extension_case merges unsupported and absent suffixes",
        patch(
            'return suffix if suffix.lower() in VIDEO_EXTENSIONS else "<unsupported>"',
            'return suffix if suffix.lower() in VIDEO_EXTENSIONS else "<none>"',
        ),
    ),
    mutant(
        "M016",
        "capture FPS min/max aggregate raw values including non-finite members",
        patch(
            "known_rates = rates if all(math.isfinite(r) for r in rates) else []",
            "known_rates = rates",
        ),
    ),
    mutant(
        "M017",
        "capture FPS is considered known when any member is finite",
        patch(
            "known_rates = rates if all(math.isfinite(r) for r in rates) else []",
            "known_rates = rates if any(math.isfinite(r) for r in rates) else []",
        ),
    ),
    mutant(
        "M018",
        "capture FPS silently drops non-finite members",
        patch(
            "known_rates = rates if all(math.isfinite(r) for r in rates) else []",
            "known_rates = [rate for rate in rates if math.isfinite(rate)]",
        ),
    ),
    mutant(
        "M019",
        "capture FPS minimum uses raw values while other fields blank",
        patch(
            "fps_min = min(known_rates) if known_rates else None",
            "fps_min = min(rates) if rates else None",
        ),
    ),
    mutant(
        "M020",
        "capture FPS maximum uses raw values while other fields blank",
        patch(
            "fps_max = max(known_rates) if known_rates else None",
            "fps_max = max(rates) if rates else None",
        ),
    ),
    mutant(
        "M021",
        "capture FPS spread uses raw values even when extrema are unknown",
        patch(
            "None if fps_min is None else fps_max - fps_min, FPS_DECIMALS",
            "max(rates) - min(rates) if rates else None, FPS_DECIMALS",
        ),
    ),
    mutant(
        "M022",
        "generation validation hashes newline-normalized table text",
        patch(
            "raw = (out_dir / name).read_bytes()",
            'raw = (out_dir / name).read_text(encoding="utf-8").encode("utf-8")',
        ),
    ),
    mutant(
        "M023",
        "generation validation leaks malformed-JSON exceptions",
        patch(
            "except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:",
            "except (OSError, UnicodeDecodeError) as error:",
        ),
    ),
    mutant(
        "M024",
        "generation validation leaks invalid-UTF-8 census exceptions",
        patch(
            "except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:",
            "except (OSError, json.JSONDecodeError) as error:",
        ),
    ),
    mutant(
        "M025",
        "generation validation leaks a missing-census OSError",
        patch(
            "except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:",
            "except (UnicodeDecodeError, json.JSONDecodeError) as error:",
        ),
    ),
    mutant(
        "M026",
        "table validation catches only FileNotFoundError",
        patch(
            "        except OSError as error:\n"
            "            raise InventoryError(\n"
            '                f"The published set is unusable: {name} is missing or cannot be read."',
            "        except FileNotFoundError as error:\n"
            "            raise InventoryError(\n"
            '                f"The published set is unusable: {name} is missing or cannot be read."',
        ),
    ),
    mutant(
        "M027",
        "generation validation accepts a non-object census",
        patch(
            "    if not isinstance(census, dict):\n"
            '        raise InventoryError("The published set is unusable: census.json is not an object.")\n',
            "",
        ),
    ),
    mutant(
        "M028",
        "generation validation checks captures before assets",
        patch(
            "for name in (ASSETS_FILENAME, CAPTURES_FILENAME):",
            "for name in (CAPTURES_FILENAME, ASSETS_FILENAME):",
        ),
    ),
    mutant(
        "M029",
        "capture membership compares total sizes instead of multisets",
        patch(
            "    if collections.Counter(a.asset_id for c in captures for a in c.assets) != collections.Counter(\n"
            "        a.asset_id for a in canonical\n"
            "    ):\n"
            '        raise InventoryError("The capture membership does not cover the canonical assets exactly.")',
            "    if sum(len(c.assets) for c in captures) != len(canonical):\n"
            '        raise InventoryError("The capture membership does not cover the canonical assets exactly.")',
        ),
    ),
    mutant(
        "M030",
        "capture membership compares sets instead of multisets",
        patch(
            "    if collections.Counter(a.asset_id for c in captures for a in c.assets) != collections.Counter(\n"
            "        a.asset_id for a in canonical\n"
            "    ):\n"
            '        raise InventoryError("The capture membership does not cover the canonical assets exactly.")',
            "    if {a.asset_id for c in captures for a in c.assets} != {\n"
            "        a.asset_id for a in canonical\n"
            "    }:\n"
            '        raise InventoryError("The capture membership does not cover the canonical assets exactly.")',
        ),
    ),
    mutant(
        "M031",
        "the per-capture member identifier check is removed",
        patch(
            "    for capture in captures:\n"
            "        # A count alone survives a swap of two members between two captures.\n"
            "        if any(a.capture_id != capture.capture_id for a in capture.assets):\n"
            '            raise InventoryError("A capture holds an asset that belongs to another capture.")',
            "",
        ),
    ),
    mutant(
        "M032",
        "duplicate capture identifiers are not rejected",
        patch(
            "    if len(known) != len(captures):\n"
            '        raise InventoryError("Two capture rows share one identifier.")\n',
            "",
        ),
    ),
    mutant(
        "M033",
        "capture coverage permits extra capture rows",
        patch(
            "if {a.capture_id for a in canonical} != known:",
            "if not {a.capture_id for a in canonical}.issubset(known):",
        ),
    ),
    mutant(
        "M034",
        "asset identifier uniqueness is not checked",
        patch(
            "    if len({a.asset_id for a in assets}) != len(assets):\n"
            '        raise InventoryError("Two assets share one identifier.")\n',
            "",
        ),
    ),
    mutant(
        "M035",
        "shape histogram counts raw tuples before rendering keys",
        patch(
            "    shapes = collections.Counter(\n"
            "        _shape_key(\n"
            "            (\n"
            "                a.facts.reported_width,\n"
            "                a.facts.reported_height,\n"
            "                round(a.facts.reported_avg_fps, 3),\n"
            "                a.facts.reported_fourcc,\n"
            "                a.facts.reported_rotation_deg,\n"
            "            )\n"
            "        )\n"
            "        for a in opened\n"
            "    )",
            "    shapes = collections.Counter(\n"
            "        (\n"
            "            a.facts.reported_width,\n"
            "            a.facts.reported_height,\n"
            "            round(a.facts.reported_avg_fps, 3),\n"
            "            a.facts.reported_fourcc,\n"
            "            a.facts.reported_rotation_deg,\n"
            "        )\n"
            "        for a in opened\n"
            "    )",
        ),
        patch(
            '        "shapes": dict(sorted(shapes.items())),',
            '        "shapes": dict(\n'
            "            sorted((_shape_key(shape), count) for shape, count in shapes.items())\n"
            "        ),",
        ),
    ),
    mutant(
        "M036",
        "shape histogram keys omit rotation",
        patch(
            "return f\"{width}x{height}@{fps:g}/{fourcc or '?'}/rot{rotation}\"",
            "return f\"{width}x{height}@{fps:g}/{fourcc or '?'}\"",
        ),
    ),
    mutant(
        "M037",
        "shape histogram keys omit FOURCC",
        patch(
            "return f\"{width}x{height}@{fps:g}/{fourcc or '?'}/rot{rotation}\"",
            'return f"{width}x{height}@{fps:g}/rot{rotation}"',
        ),
    ),
    mutant(
        "M038",
        "shape histogram FPS keys round to one decimal",
        patch(
            "return f\"{width}x{height}@{fps:g}/{fourcc or '?'}/rot{rotation}\"",
            "return f\"{width}x{height}@{fps:.1f}/{fourcc or '?'}/rot{rotation}\"",
        ),
    ),
    mutant(
        "M039",
        "shape histogram keys use raw tuple repr",
        patch(
            "return f\"{width}x{height}@{fps:g}/{fourcc or '?'}/rot{rotation}\"",
            "return repr(shape)",
        ),
    ),
    mutant(
        "M040",
        "probe constructor exceptions escape",
        patch(
            "    try:\n"
            "        cap = cv2.VideoCapture(text)\n"
            "    except Exception:\n"
            "        return _UNPROBED",
            "    try:\n        cap = cv2.VideoCapture(text)\n    except Exception:\n        raise",
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M041",
        "probe property and isOpened exceptions escape",
        patch(
            "        )\n    except Exception:\n        return _UNPROBED\n    finally:",
            "        )\n    except Exception:\n        raise\n    finally:",
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M042",
        "probe release exceptions escape",
        patch(
            "        with contextlib.suppress(Exception):\n            cap.release()",
            "        cap.release()",
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M043",
        "probe ignores a false isOpened result",
        patch(
            "        if not cap.isOpened():\n            return _UNPROBED",
            "        cap.isOpened()",
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M044",
        "probe constructor catches only OSError",
        patch(
            "    try:\n"
            "        cap = cv2.VideoCapture(text)\n"
            "    except Exception:\n"
            "        return _UNPROBED",
            "    try:\n"
            "        cap = cv2.VideoCapture(text)\n"
            "    except OSError:\n"
            "        return _UNPROBED",
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M045",
        "size accounting follows symlinks",
        patch("        if entry.is_symlink():\n            return None\n", ""),
    ),
    mutant(
        "M046",
        "size accounting publishes a symlink inode size",
        patch(
            "        if entry.is_symlink():\n            return None",
            "        if entry.is_symlink():\n            return entry.lstat().st_size",
        ),
    ),
    mutant(
        "M047",
        "size lookup OSError escapes",
        patch(
            "        return entry.stat().st_size\n    except OSError:\n        return None",
            "        return entry.stat().st_size\n    except OSError:\n        raise",
        ),
    ),
    mutant(
        "M048",
        "size lookup OSError becomes a zero byte fact",
        patch(
            "        return entry.stat().st_size\n    except OSError:\n        return None",
            "        return entry.stat().st_size\n    except OSError:\n        return 0",
        ),
    ),
    mutant(
        "M049",
        "FOURCC text strips both ends",
        patch(
            'return text if text.isprintable() else ""',
            'return text.strip() if text.isprintable() else ""',
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M050",
        "FOURCC text strips its right edge",
        patch(
            'return text if text.isprintable() else ""',
            'return text.rstrip() if text.isprintable() else ""',
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M051",
        "FOURCC bytes decode in reverse order",
        patch(
            'text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))',
            'text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(3, -1, -1))',
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M052",
        "FOURCC publishes unprintable bytes",
        patch('return text if text.isprintable() else ""', "return text", path=VIDEO_IO),
    ),
    mutant(
        "M053",
        "FOURCC truncates to three characters",
        patch(
            'return text if text.isprintable() else ""',
            'return text[:3] if text.isprintable() else ""',
            path=VIDEO_IO,
        ),
    ),
    mutant(
        "M054",
        "census self-digest uses one render instead of a JSON round trip",
        patch(
            "return _text_digest(render_json(json.loads(render_json(body))))",
            "return _text_digest(render_json(body))",
        ),
    ),
    mutant(
        "M055",
        "census self-digest empties generation in its body",
        patch(
            'body["generation"] = {k: v for k, v in generation.items() if k != CENSUS_FILENAME}',
            'body["generation"] = {}',
        ),
    ),
    mutant(
        "M056",
        "census self-digest omits generation from its body",
        patch(
            'body["generation"] = {k: v for k, v in generation.items() if k != CENSUS_FILENAME}',
            'body.pop("generation", None)',
        ),
    ),
    mutant(
        "M057",
        "census self-digest includes its own generation key",
        patch(
            'body["generation"] = {k: v for k, v in generation.items() if k != CENSUS_FILENAME}',
            'body["generation"] = dict(generation)',
        ),
    ),
    mutant(
        "M058",
        "JSON rendering drops key sorting",
        patch(
            'return json.dumps(payload, sort_keys=True, indent=2) + "\\n"',
            'return json.dumps(payload, indent=2) + "\\n"',
        ),
    ),
    mutant(
        "M059",
        "JSON rendering drops the trailing newline",
        patch(
            'return json.dumps(payload, sort_keys=True, indent=2) + "\\n"',
            "return json.dumps(payload, sort_keys=True, indent=2)",
        ),
    ),
    mutant(
        "M060",
        "publication omits the census self-digest key",
        patch('    census["generation"][CENSUS_FILENAME] = census_digest(census)\n', ""),
    ),
    mutant(
        "M061",
        "census self-digest hashes compact normalized JSON",
        patch(
            "return _text_digest(render_json(json.loads(render_json(body))))",
            "return _text_digest(json.dumps(json.loads(render_json(body))))",
        ),
    ),
    mutant(
        "M062",
        "mixed orientation-auto readbacks are accepted",
        patch(
            "    if len(readbacks) > 1:\n"
            '        raise InventoryError("The opened assets disagree about auto-rotation.")\n',
            "",
        ),
    ),
    mutant(
        "M063",
        "grammar quarantine outranks an unreadable probe",
        patch(
            "        elif facts.probe_status != PROBE_OPENED:\n"
            '            disposition, reason_code = EXCLUDED, "probe_unreadable"\n'
            "        elif parse.ok:\n"
            "            disposition, reason_code = CANONICAL, REASON_OK\n"
            "        else:\n"
            "            disposition, reason_code = QUARANTINED, parse.reason_code",
            "        elif not parse.ok:\n"
            "            disposition, reason_code = QUARANTINED, parse.reason_code\n"
            "        elif facts.probe_status != PROBE_OPENED:\n"
            '            disposition, reason_code = EXCLUDED, "probe_unreadable"\n'
            "        else:\n"
            "            disposition, reason_code = CANONICAL, REASON_OK",
        ),
    ),
    mutant(
        "M064",
        "an unreadable probe is classified as a read error",
        patch(
            'disposition, reason_code = EXCLUDED, "probe_unreadable"',
            'disposition, reason_code = EXCLUDED, "read_error"',
        ),
    ),
    mutant(
        "M065",
        "tokenization uses maxsplit and accepts an extra token as side text",
        patch(
            'tokens = stem.split("_") if stem else []',
            'tokens = stem.split("_", 3) if stem else []',
        ),
    ),
    mutant(
        "M066",
        "subject conflicts quarantine only one directory",
        patch("            bad.update(directories)", "            bad.add(max(directories))"),
    ),
    mutant(
        "M067",
        "every capture is marked as a view conflict",
        patch(
            "return len(self.assets) > len(self.views)",
            "return len(self.assets) >= len(self.views)",
        ),
    ),
    mutant(
        "M068",
        "view coverage counts assets instead of distinct views",
        patch(
            "sorted(collections.Counter(len(c.views) for c in captures).items())",
            "sorted(collections.Counter(len(c.assets) for c in captures).items())",
        ),
    ),
    mutant(
        "M069",
        "asset rows sort case-insensitively instead of by code point",
        patch(
            "records.sort(key=lambda record: record.source_path)",
            "records.sort(key=lambda record: record.source_path.lower())",
        ),
    ),
    mutant(
        "M070",
        "the observed grass-to-glass repair is removed",
        patch('    "grass": "glass",\n', ""),
    ),
    mutant(
        "M071",
        "normalization uses casefold instead of lower",
        patch("text = stem.lower()", "text = stem.casefold()"),
    ),
    mutant(
        "M072",
        "repeat marker zero is accepted as a consumed marker",
        patch(
            "    if marker == 0:\n"
            '        # No file manager numbers a copy from zero, so "(0)" is not a repeat\n'
            "        # marker.  Leaving it on the stem routes it to the unrecognized branch\n"
            '        # and keeps "repeat >= 1" equivalent to "a marker was consumed".\n'
            "        return stem, 0\n",
            "",
        ),
    ),
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def baseline_files() -> dict[str, bytes]:
    return {path: (ROOT / path).read_bytes() for path in (INVENTORY, VIDEO_IO)}


def validate_catalog(originals: dict[str, bytes]) -> None:
    seen: set[str] = set()
    for item in MUTANTS:
        if item.id in seen:
            raise SystemExit(f"duplicate mutant id: {item.id}")
        seen.add(item.id)
        if not item.patches:
            raise SystemExit(f"mutant has no patches: {item.id}")
        texts = {path: raw.decode("utf-8") for path, raw in originals.items()}
        for change in item.patches:
            text = texts[change.path]
            count = text.count(change.old)
            if count != 1:
                raise SystemExit(
                    f"{item.id}: expected one occurrence in {change.path}, found {count}"
                )
            texts[change.path] = text.replace(change.old, change.new, 1)
        if all(texts[path].encode("utf-8") == originals[path] for path in texts):
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


def write_result(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_only(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--check", action="store_true", help="validate the mutation catalogue only")
    result.add_argument("--only", help="comma-separated mutant IDs")
    result.add_argument(
        "--phase",
        choices=("initial", "closure"),
        default="initial",
        help="result section to update",
    )
    result.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "tests" / "inventory_mutation_results.json",
    )
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
    tested_head = git("rev-parse", "HEAD")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("target_sha256") != target_hashes:
            raise SystemExit("result file targets a different source baseline")
    else:
        payload = {
            "schema_version": 1,
            "tested_head": tested_head,
            "test_command": (
                "python -m pytest -q tests/test_inventory.py tests/test_inventory_review.py "
                "tests/test_inventory_predicates.py"
            ),
            "target_sha256": target_hashes,
            "initial": [],
            "closure": [],
        }

    rows = {row["id"]: row for row in payload[args.phase]}
    try:
        for item in selected:
            apply(item, originals)
            completed = run_tests()
            nodeids = failure_nodeids(completed)
            verdict = "killed" if completed.returncode else "SURVIVED"
            rows[item.id] = {
                "id": item.id,
                "file_line": locate(item, originals),
                "mutation": item.description,
                "verdict": verdict,
                "pytest_returncode": completed.returncode,
                "killing_tests": nodeids
                or ([f"<pytest-exit:{completed.returncode}>"] if completed.returncode else []),
            }
            restore(originals)
            payload[args.phase] = [rows[key] for key in sorted(rows)]
            write_result(output, payload)
            first = rows[item.id]["killing_tests"][:1]
            suffix = first[0] if first else "none"
            print(f"{item.id} {verdict} {suffix}")
    finally:
        restore(originals)

    if {path: sha256((ROOT / path).read_bytes()) for path in originals} != target_hashes:
        raise SystemExit("source restoration failed")
    final_baseline = run_tests()
    if final_baseline.returncode:
        sys.stdout.write(final_baseline.stdout)
        sys.stderr.write(final_baseline.stderr)
        raise SystemExit("focused baseline failed after restoration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
