#!/usr/bin/env python3
"""Replay the M2.3 measurement mutation campaign against its focused test gate.

Rerun:
    env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync \
        python scripts/run_measure_mutations.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The oracle sees exactly these files.  A suite that pins a mutated predicate
# and is not named here scores a silent no-op kill, so every test file covering
# `measure` or `qualify` joins this tuple in the commit that adds it.
TEST_COMMAND = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "tests/test_measure.py",
    "tests/test_measure_cache.py",
    "tests/test_measure_diff_blind.py",
    "tests/test_measure_mutants.py",
    "tests/test_qualify.py",
)
# `sys.executable` is an absolute interpreter path, so the committed result
# would bind to whoever ran it.  The record names the command, not the machine.
TEST_COMMAND_TEXT = " ".join(("python", *TEST_COMMAND[1:]))


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
    equivalent_reason: str | None = None


def patch(path: str, old: str, new: str) -> Patch:
    return Patch(path, old, new)


def mutant(
    identifier: str,
    description: str,
    *patches: Patch,
    equivalent_reason: str | None = None,
) -> Mutant:
    return Mutant(identifier, description, patches, equivalent_reason)


MEASURE = "src/pose_estimation/measure/__init__.py"
AUDIO = "src/pose_estimation/measure/audio_offset.py"
QUALIFY = "src/pose_estimation/qualify.py"

MUTANTS = (
    mutant(
        "M01-reconcile-foreign",
        "accept a key absent from the canonical registry",
        patch(MEASURE, "    if foreign:\n", "    if False and foreign:\n"),
    ),
    mutant(
        "M02-reconcile-omitted-return",
        "discard the omitted canonical-key return",
        patch(
            MEASURE,
            "    return frozenset(expected) - frozenset(rows)\n",
            "    return frozenset()\n",
        ),
        equivalent_reason=(
            "qualify._ingest discards the return; asset and pair omission semantics come from "
            "independent keyed lookups"
        ),
    ),
    mutant(
        "M03-scale-pairing",
        "accept an unmatched scale class and confidence",
        patch(
            MEASURE,
            '        if (row["scale_ref_class"] == NO_REFERENCE) != '
            '(row["scale_ref_conf"] == NO_REFERENCE):\n',
            '        if False and (row["scale_ref_class"] == NO_REFERENCE) != '
            '(row["scale_ref_conf"] == NO_REFERENCE):\n',
        ),
    ),
    mutant(
        "M04-row-count-bool",
        "accept a Boolean manifest row count",
        patch(
            MEASURE,
            "        if isinstance(declared, bool) or not isinstance(declared, int) "
            "or declared < 0:\n",
            "        if not isinstance(declared, int) or declared < 0:\n",
        ),
    ),
    mutant(
        "M05-fuse-corroborated",
        "refuse an agreeing pair",
        patch(
            QUALIFY,
            "        return PAIR_OK_CORROBORATED if delta <= AGREE_TOLERANCE_S else "
            "PAIR_CONTRADICTED\n",
            "        return PAIR_CONTRADICTED if delta <= AGREE_TOLERANCE_S else "
            "PAIR_CONTRADICTED\n",
        ),
    ),
    mutant(
        "M06-fuse-contradicted",
        "accept a contradicting pair",
        patch(
            QUALIFY,
            "        return PAIR_OK_CORROBORATED if delta <= AGREE_TOLERANCE_S else "
            "PAIR_CONTRADICTED\n",
            "        return PAIR_OK_CORROBORATED if delta <= AGREE_TOLERANCE_S else "
            "PAIR_OK_CORROBORATED\n",
        ),
    ),
    mutant(
        "M07-fuse-uncorroborated",
        "map audio-only acceptance to visual-only",
        patch(
            QUALIFY,
            "        return PAIR_OK_UNCORROBORATED\n",
            "        return PAIR_VISUAL_ONLY\n",
        ),
    ),
    mutant(
        "M08-fuse-visual-only",
        "map visual-only acceptance to audio-only",
        patch(
            QUALIFY,
            "        return PAIR_VISUAL_ONLY\n",
            "        return PAIR_OK_UNCORROBORATED\n",
        ),
    ),
    mutant(
        "M09-fuse-neither",
        "map dual rejection to visual-only",
        patch(
            QUALIFY,
            "    return PAIR_NEITHER_ACCEPTED\n",
            "    return PAIR_VISUAL_ONLY\n",
        ),
    ),
    mutant(
        "M10-fuse-boundary",
        "exclude exact agreement-tolerance equality",
        patch(
            QUALIFY,
            "delta <= AGREE_TOLERANCE_S",
            "delta < AGREE_TOLERANCE_S",
        ),
    ),
    mutant(
        "M11-spanning-connectivity",
        "return a partial spanning solution",
        patch(
            QUALIFY,
            "    return solved if len(solved) == len(members) else None\n",
            "    return solved\n",
        ),
    ),
    mutant(
        "M12-closure-three-edges",
        "compute triangle closure without all three edges",
        patch(
            QUALIFY,
            "            if all(edge in directed for edge in triangle):\n",
            "            if True:\n",
        ),
    ),
    mutant(
        "M13-span-multicamera",
        "publish a zero offset span for one camera",
        patch(
            QUALIFY,
            "        if solved is not None and len(members) > 1:\n",
            "        if solved is not None:\n",
        ),
    ),
    mutant(
        "M14-cache-full-digest",
        "ignore corruption of the full-rate audio array",
        patch(
            AUDIO,
            "        return all(artifacts.get(part.name) == _file_digest(part) "
            "for part in (full, coarse))\n",
            "        return all(artifacts.get(part.name) == _file_digest(part) "
            "for part in (coarse,))\n",
        ),
    ),
    mutant(
        "M15-cache-coarse-digest",
        "ignore corruption of the coarse audio array",
        patch(
            AUDIO,
            "        return all(artifacts.get(part.name) == _file_digest(part) "
            "for part in (full, coarse))\n",
            "        return all(artifacts.get(part.name) == _file_digest(part) "
            "for part in (full,))\n",
        ),
    ),
    mutant(
        "M16-cache-source-digest",
        "reuse audio arrays after source replacement",
        patch(
            AUDIO,
            '        if recorded.get("source_sha256") != source_sha256:\n',
            '        if False and recorded.get("source_sha256") != source_sha256:\n',
        ),
    ),
    mutant(
        "M17-placement-sort",
        "retain placement-table insertion order",
        patch(
            QUALIFY,
            "    return {event_id: sorted(ids) for event_id, ids in members.items()}\n",
            "    return dict(members)\n",
        ),
    ),
    mutant(
        "M18-closure-distribution-sort",
        "derive closure endpoints from encounter order",
        patch(
            QUALIFY,
            '    ordered = sorted(values)\n    return {\n        "n": float(len(ordered)),\n',
            '    ordered = list(values)\n    return {\n        "n": float(len(ordered)),\n',
        ),
    ),
)
TARGETS = tuple(sorted({change.path for item in MUTANTS for change in item.patches}))


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def baseline_files() -> dict[str, bytes]:
    return {path: (ROOT / path).read_bytes() for path in TARGETS}


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
    with tempfile.TemporaryDirectory(prefix="pose-measure-mutation-pyc-") as pycache:
        env["PYTHONPYCACHEPREFIX"] = pycache
        return subprocess.run(
            TEST_COMMAND,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )


def failure_nodeids(completed: subprocess.CompletedProcess[str]) -> list[str]:
    lines = (completed.stdout + completed.stderr).splitlines()
    return sorted({line.split()[1] for line in lines if line.startswith("FAILED ")})


def locate(item: Mutant, originals: dict[str, bytes]) -> str:
    change = item.patches[0]
    text = originals[change.path].decode("utf-8")
    index = text.index(change.old)
    return f"{change.path}:{text.count(chr(10), 0, index) + 1}"


def apply(item: Mutant, originals: dict[str, bytes]) -> None:
    texts = {path: raw.decode("utf-8") for path, raw in originals.items()}
    for change in item.patches:
        texts[change.path] = texts[change.path].replace(change.old, change.new, 1)
    for path in {change.path for change in item.patches}:
        (ROOT / path).write_text(texts[path], encoding="utf-8", newline="")


def restore(originals: dict[str, bytes]) -> None:
    for path, raw in originals.items():
        (ROOT / path).write_bytes(raw)


def write_result(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_only(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--check", action="store_true", help="Validate exact-one patch anchors only."
    )
    result.add_argument("--only", help="Run a comma-separated mutant-id subset.")
    result.add_argument(
        "--output",
        default=str(ROOT / "tests" / "measure_mutation_results.json"),
        help="Write an atomic JSON result, or use '-' to stream without a file.",
    )
    return result


def _payload(originals: dict[str, bytes], output: pathlib.Path | None) -> dict[str, Any]:
    target_hashes = {path: sha256(raw) for path, raw in originals.items()}
    if output is None or not output.exists():
        return {
            "schema_version": 1,
            "test_command": TEST_COMMAND_TEXT,
            "target_sha256": target_hashes,
            "results": [],
        }
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("target_sha256") != target_hashes:
        raise SystemExit("result file targets a different source baseline")
    return payload


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

    output = None if args.output == "-" else pathlib.Path(args.output)
    if output is not None and not output.is_absolute():
        output = ROOT / output
    payload = _payload(originals, output)
    rows = {row["id"]: row for row in payload["results"]}
    selected_rows: list[dict[str, Any]] = []
    try:
        for item in selected:
            apply(item, originals)
            completed = run_tests()
            nodeids = failure_nodeids(completed)
            if completed.returncode:
                verdict = "killed"
            elif item.equivalent_reason is not None:
                verdict = "equivalent"
            else:
                verdict = "SURVIVED"
            row = {
                "id": item.id,
                "file_line": locate(item, originals),
                "mutation": item.description,
                "verdict": verdict,
                "pytest_returncode": completed.returncode,
                "killing_tests": nodeids
                or ([f"<pytest-exit:{completed.returncode}>"] if completed.returncode else []),
                "equivalent_reason": item.equivalent_reason,
            }
            rows[item.id] = row
            selected_rows.append(row)
            restore(originals)
            payload["results"] = [rows[key] for key in sorted(rows)]
            if output is not None:
                write_result(output, payload)
            first = row["killing_tests"][:1]
            suffix = first[0] if first else "none"
            print(f"{item.id} {verdict} {suffix}", flush=True)
    finally:
        restore(originals)

    target_hashes = {path: sha256(raw) for path, raw in originals.items()}
    if {path: sha256((ROOT / path).read_bytes()) for path in originals} != target_hashes:
        raise SystemExit("source restoration failed")
    final_baseline = run_tests()
    if final_baseline.returncode:
        sys.stdout.write(final_baseline.stdout)
        sys.stderr.write(final_baseline.stderr)
        raise SystemExit("focused baseline failed after restoration")

    killed = sum(row["verdict"] == "killed" for row in selected_rows)
    equivalent = sum(row["verdict"] == "equivalent" for row in selected_rows)
    survived = sum(row["verdict"] == "SURVIVED" for row in selected_rows)
    print(
        f"mutants={len(selected_rows)} killed={killed} equivalent={equivalent} survived={survived}"
    )
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
