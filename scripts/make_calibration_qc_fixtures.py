#!/usr/bin/env python3
"""Generate the de-identified calibration_qc byte oracle and refusal fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tests" / "fixtures" / "calibration_qc_set"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from pose_estimation import calibration_qc, qualify  # noqa: E402
from test_qualify import _publish, _uniform, _write_media  # noqa: E402
from test_sessions import _canonical  # noqa: E402

FIXTURE_GENERATOR_VERSION = "v1"
GENERATOR_PATH = "scripts/make_calibration_qc_fixtures.py"
NAMESPACE = tuple(f"s{ordinal:02d}" for ordinal in range(90, 100))
PERMITTED_REAL_NUMBERS = sorted(
    {str(value) for value in calibration_qc.RULED_POPULATION.values()}
    | {calibration_qc.RULING["image_height_px"]}
)
REASONS = (
    "arm_duplicate",
    "arm_missing",
    "cell_alphabet",
    "census_digest",
    "claim_missing",
    "claim_prohibited",
    "corpus_cardinality",
    "digest_malformed",
    "digest_missing",
    "evidence_empty",
    "evidence_malformed",
    "evidence_missing",
    "evidence_schema",
    "forbidden_key",
    "forbidden_value",
    "generation_foreign",
    "marker_shape",
    "marker_unreadable",
    "not_owned",
    "output_not_directory",
    "output_overlap",
    "population_mismatch",
    "probe_digest",
    "probe_missing",
    "probe_stale",
    "qualification_stale",
    "table_digest",
    "table_missing",
    "tree_digest",
    "tree_unreadable",
)
VALIDATE_REASONS = frozenset(
    {
        "census_digest",
        "generation_foreign",
        "marker_shape",
        "marker_unreadable",
        "probe_stale",
        "qualification_stale",
        "table_digest",
        "table_missing",
        "tree_digest",
    }
)
NOT_FILE_ONLY = {
    "claim_missing": "Requires changing the module-constant claim set after the input files are read.",
    "corpus_cardinality": "Requires changing the module-constant one-row ruling before publication.",
    "output_overlap": "Requires an output path inside an input tree; the replay paths are fixed siblings.",
    "tree_unreadable": "Requires a filesystem read-permission failure that Git file modes cannot carry.",
}
README_TEXT = """# `calibration_qc` regression fixtures

This directory holds de-identified test data for the calibration ruling publisher
(`src/pose_estimation/calibration_qc.py`). No byte here comes from patient data.

## What is here

| path | content |
| ---- | ------- |
| `inputs/upstream/` | A synthetic published qualification generation. |
| `inputs/probes/` | Two synthetic probe scripts. The publisher digests these files. |
| `inputs/evidence/` | One synthetic probe capture and the digest it was taken under. |
| `expected/published/` | The generation the publisher must emit from `inputs/`. |
| `negatives/<reason>/` | One corrupted input set for each refusal the publisher can raise. |
| `manifest.json` | The synthetic namespace, the input digests and the refusal matrix. |

Directory names avoid the four tree names that `.gitignore` matches at any depth. A
fixture below one of those names is not committed. See the contract for the list.

## How to regenerate

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync \\
  python scripts/make_calibration_qc_fixtures.py --force
```

The generator is idempotent. Two runs from a clean base give identical bytes.

## How to check

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync \\
  python scripts/check_calibration_qc_fixtures.py
```

The command exits 0 when all 12 predicates pass. It exits 1 and prints each failure.
The test `tests/test_calibration_qc_fixtures.py` calls the same code.

## What the privacy scan forbids

The scan reads every file in this directory, this file included. It fails on any of
these:

- A capture identifier outside the synthetic subject namespace.
- Path text that names one of the patient-adjacent trees.
- An absolute path to this repository.
- A real corpus statistic. Only numbers that are already constants in
  `calibration_qc.py` are permitted.

The scan holds the needles it searches for. This file must not spell them, because a
document inside the scanned set carries every string it quotes. `check_calibration_qc_fixtures.py`
keeps that list, and the contract explains the rule.
"""


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _write_json(path: pathlib.Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {str(path.relative_to(root)): path for path in sorted(root.rglob("*")) if path.is_file()}


def _statistic(*, pixels: bool = False) -> dict[str, int | float | None]:
    if pixels:
        return {"n": 9, "median": 2.3456, "min": 1.2345, "max": 3.4567, "above_0p5": None}
    return {"n": 12, "median": 0.1234, "min": -0.5678, "max": 0.8765, "above_0p5": 3}


def _record(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "pairs": calibration_qc.RULED_POPULATION["pairs"],
        "events": calibration_qc.RULED_POPULATION["events"],
        "realizations": 2,
        "between_event_r": _statistic(),
        "between_event_r_abs": _statistic(),
        "within_event_r": _statistic(),
        "median_abs_px": _statistic(pixels=True),
        "shared_fraction": 0.25,
    }


def _arms() -> list[str]:
    return [
        *sorted(calibration_qc.REQUIRED_ARMS),
        "SYNTH shared image bias 4.0px",
        "SYNTH per-event bias 4.0px",
        "SYNTH noise sigma=4.0px",
    ]


def _records() -> list[dict[str, Any]]:
    return [_record(arm) for arm in _arms()]


def _write_capture(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    _write_text(path, "\n".join(lines) + "\n")


def _build_upstream(root: pathlib.Path, ordinal: int, view: str) -> pathlib.Path:
    root.mkdir(parents=True)
    asset = _canonical(ordinal, view)
    registry, sessions_dir, corpus, upstream = _publish(root, [asset])
    _write_media(corpus / asset.source_path, _uniform(6))
    qualify.run(registry, sessions_dir, corpus, upstream)
    return upstream


def _write_inputs(out: pathlib.Path, seed: pathlib.Path) -> None:
    upstream = _build_upstream(seed / "base", 90, "above")
    shutil.copytree(upstream, out / "inputs" / "upstream")
    probes = out / "inputs" / "probes"
    for name in calibration_qc.PROBE_SCRIPTS.values():
        _write_text(probes / name, f"# synthetic {name} fixture\n")
    evidence = out / "inputs" / "evidence"
    _write_capture(evidence / "bias_transfer.jsonl", _records())
    digest = _sha256(probes / calibration_qc.PROBE_SCRIPTS["bias_transfer"])
    _write_text(evidence / "bias_transfer.sha256", f"{digest}  probe_bias_transfer.py\n")


def _write_expected(out: pathlib.Path) -> None:
    calibration_qc.run(
        out / "inputs" / "upstream",
        out / "inputs" / "evidence",
        out / "inputs" / "probes",
        out / "expected" / "published",
    )


def _expect(entry: pathlib.Path, reason: str, **extra: Any) -> None:
    _write_json(
        entry / "expect.json",
        {"path": "validate" if reason in VALIDATE_REASONS else "run", "reason": reason, **extra},
    )


def _write_evidence_case(entry: pathlib.Path, reason: str) -> None:
    records = _records()
    if reason == "evidence_empty":
        _write_text(entry / "evidence" / "bias_transfer.jsonl", "{}\n")
    elif reason == "evidence_malformed":
        _write_text(entry / "evidence" / "bias_transfer.jsonl", '{"label":"REAL cut"}}\n')
    elif reason == "evidence_schema":
        records[0].pop("within_event_r")
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    elif reason == "forbidden_key":
        records[0]["capture_id"] = "redacted"
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    elif reason == "forbidden_value":
        records.append(_record("REAL s90-cap-l"))
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    elif reason == "arm_missing":
        records = [
            record for record in records if record["label"] != "REAL same view pair + same subject"
        ]
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    elif reason == "arm_duplicate":
        records.append(_record(str(records[0]["label"])))
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    elif reason == "cell_alphabet":
        records.append(_record("REAL invalid\n"))
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    elif reason == "population_mismatch":
        records[0]["pairs"] = 12
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    elif reason == "claim_prohibited":
        records.append(_record("no estimator could recover extrinsics"))
        _write_capture(entry / "evidence" / "bias_transfer.jsonl", records)
    else:
        raise ValueError(f"unknown evidence mutation: {reason}")


def _changed_upstream_overlay(out: pathlib.Path, entry: pathlib.Path, seed: pathlib.Path) -> None:
    alternate = _build_upstream(seed / "alternate", 91, "left")
    base_files = _files(out / "inputs" / "upstream")
    alternate_files = _files(alternate)
    if set(base_files) != set(alternate_files):
        raise RuntimeError("synthetic upstream generations have different entry sets")
    changed = 0
    for rel, source in alternate_files.items():
        if source.read_bytes() == base_files[rel].read_bytes():
            continue
        target = entry / "upstream" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        changed += 1
    if not changed:
        raise RuntimeError("alternate upstream generation did not move any byte")


def _marker(out: pathlib.Path) -> dict[str, Any]:
    return json.loads(
        (out / "expected" / "published" / calibration_qc.CALIBRATION_QC_FILENAME).read_text(
            encoding="utf-8"
        )
    )


def _write_case(out: pathlib.Path, seed: pathlib.Path, reason: str) -> None:
    entry = out / "negatives" / reason
    entry.mkdir(parents=True)
    if reason == "probe_missing":
        _expect(entry, reason, deletes=["probes/probe_bias_transfer.py"])
    elif reason == "evidence_missing":
        _expect(entry, reason, deletes=["evidence/bias_transfer.jsonl"])
    elif reason in {
        "arm_duplicate",
        "arm_missing",
        "cell_alphabet",
        "claim_prohibited",
        "evidence_empty",
        "evidence_malformed",
        "evidence_schema",
        "forbidden_key",
        "forbidden_value",
        "population_mismatch",
    }:
        _write_evidence_case(entry, reason)
        _expect(entry, reason)
    elif reason == "probe_digest":
        _write_text(entry / "evidence" / "bias_transfer.sha256", f"{'0' * 64}\n")
        _expect(entry, reason)
    elif reason == "digest_missing":
        _expect(entry, reason, deletes=["evidence/bias_transfer.sha256"])
    elif reason == "digest_malformed":
        _write_text(entry / "evidence" / "bias_transfer.sha256", "not-a-digest\n")
        _expect(entry, reason)
    elif reason == "not_owned":
        _write_text(entry / "out" / "foreign.txt", "foreign output\n")
        _expect(entry, reason)
    elif reason == "output_not_directory":
        _write_text(entry / "out", "regular file\n")
        _expect(entry, reason)
    elif reason == "marker_unreadable":
        _expect(entry, reason, deletes_published=[calibration_qc.CALIBRATION_QC_FILENAME])
    elif reason == "marker_shape":
        _write_json(entry / "published" / calibration_qc.CALIBRATION_QC_FILENAME, {})
        _expect(entry, reason)
    elif reason == "table_missing":
        _expect(entry, reason, deletes_published=[calibration_qc.CORPUS_QC_FILENAME])
    elif reason == "table_digest":
        table = out / "expected" / "published" / calibration_qc.CORPUS_QC_FILENAME
        _write_text(
            entry / "published" / calibration_qc.CORPUS_QC_FILENAME, table.read_text() + "x"
        )
        _expect(entry, reason)
    elif reason == "census_digest":
        marker = _marker(out)
        marker["schema_version"] = "v2"
        _write_json(entry / "published" / calibration_qc.CALIBRATION_QC_FILENAME, marker)
        _expect(entry, reason)
    elif reason == "tree_digest":
        _write_text(entry / "published" / "extra.txt", "extra entry\n")
        _expect(entry, reason)
    elif reason == "qualification_stale":
        _changed_upstream_overlay(out, entry, seed)
        _expect(entry, reason)
    elif reason == "probe_stale":
        name = calibration_qc.PROBE_SCRIPTS["calibration_bias"]
        _write_text(entry / "probes" / name, f"# changed synthetic {name} fixture\n")
        _expect(entry, reason)
    elif reason == "generation_foreign":
        marker = _marker(out)
        marker["generation"]["generator_version"] = "v0"
        _write_json(entry / "published" / calibration_qc.CALIBRATION_QC_FILENAME, marker)
        _expect(entry, reason)
    else:
        raise ValueError(f"unknown file-only reason: {reason}")


def _write_negatives(out: pathlib.Path, seed: pathlib.Path) -> None:
    for reason in REASONS:
        if reason not in NOT_FILE_ONLY:
            _write_case(out, seed, reason)


def _write_manifest(out: pathlib.Path) -> None:
    matrix = {
        reason: (
            "not_file_only"
            if reason in NOT_FILE_ONLY
            else f"negative:{'validate' if reason in VALIDATE_REASONS else 'run'}"
        )
        for reason in REASONS
    }
    inputs = out / "inputs"
    manifest = {
        "purpose": "De-identified regression fixtures for src/pose_estimation/calibration_qc.py. One valid generation (the byte oracle) plus one negative per file-only refusal reason.",
        "generator": GENERATOR_PATH,
        "validator": "scripts/check_calibration_qc_fixtures.py",
        "contract": ".agent/archive/contract-m2u72.md",
        "generator_version": FIXTURE_GENERATOR_VERSION,
        "namespace": list(NAMESPACE),
        "permitted_real_numbers": PERMITTED_REAL_NUMBERS,
        "input_digests": {rel: _sha256(path) for rel, path in _files(inputs).items()},
        "not_file_only": NOT_FILE_ONLY,
        "matrix": matrix,
    }
    _write_json(out / "manifest.json", manifest)


def _is_own_fixture_set(out: pathlib.Path) -> bool:
    """Judge a destination by the manifest this generator writes into it."""
    try:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and manifest.get("generator") == GENERATOR_PATH


def _reset(out: pathlib.Path, force: bool) -> None:
    if out.exists() or out.is_symlink():
        if not force:
            raise SystemExit(f"refusing existing destination without --force: {out}")
        # `--force` deletes a tree recursively, so it may name only this
        # generator's own output or an empty directory -- the ownership rule the
        # publishers apply before replacing a generation, for the same reason.
        if out.is_symlink() or not out.is_dir():
            out.unlink()
        elif any(out.iterdir()) and not _is_own_fixture_set(out):
            raise SystemExit(f"refusing to replace a destination this tool does not own: {out}")
        else:
            shutil.rmtree(out)
    out.mkdir(parents=True)


def generate(out: pathlib.Path, *, force: bool) -> None:
    out = out.resolve()
    if out == REPO or out == pathlib.Path(out.anchor):
        raise SystemExit(f"refusing unsafe destination: {out}")
    _reset(out, force)
    seed = out / ".seed"
    try:
        _write_text(out / "README.md", README_TEXT)
        _write_inputs(out, seed)
        _write_expected(out)
        _write_negatives(out, seed)
    finally:
        shutil.rmtree(seed, ignore_errors=True)
    _write_manifest(out)
    for path in _files(out).values():
        path.chmod(0o644)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    generate(args.out, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
