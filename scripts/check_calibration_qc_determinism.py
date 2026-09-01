"""Determinism + tamper campaign for the `calibration_qc/` publisher (M2.7.1).

Seed state: every sweep and tamper verdict reads `unknown` and the run exits 1.
Fill one `run_sweep` / `run_tamper` branch at a time and rerun; the campaign
exits 0 only when no row reads `unknown` and none reads `FAIL`.

Two-mode split is deliberately absent. The qualifier checker runs `flagless`
and `measured` because `--measurements` changes what it publishes; F1a has no
optional input, so one baseline covers it. The checker must never run an
estimator: it publishes from a fixed corpus row plus captured probe stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

ARTIFACTS: tuple[str, ...] = ("corpus_qc.csv", "evidence_qc.csv", "calibration_qc.json")
MARKER = "calibration_qc.json"
TAMPER_EDIT = "evidence_qc.csv"
TAMPER_DELETE = "corpus_qc.csv"

# Byte tripwire. Any file whose bytes can shape a published byte belongs here,
# because a source edit that changes output while the result JSON stays put is
# a silent false green. The qualifier fixture closure is included: this checker
# builds its upstream `qualification/` through `qualify.run`, so every module
# that call touches can move a published digest.
SOURCE_FILES: tuple[str, ...] = (
    "scripts/check_calibration_qc_determinism.py",
    "scripts/probe_bias_transfer.py",
    "scripts/probe_calibration_bias.py",
    "src/pose_estimation/__init__.py",
    "src/pose_estimation/calibration_qc.py",
    "src/pose_estimation/inventory.py",
    "src/pose_estimation/measure/__init__.py",
    "src/pose_estimation/measure/audio_offset.py",
    "src/pose_estimation/measure/detect.py",
    "src/pose_estimation/measure/mebx.py",
    "src/pose_estimation/measure/rigidity.py",
    "src/pose_estimation/measure/scale.py",
    "src/pose_estimation/measure/statuses.py",
    "src/pose_estimation/measure/sync.py",
    "src/pose_estimation/measure/visual_offset.py",
    "src/pose_estimation/multicam.py",
    "src/pose_estimation/qualify.py",
    "src/pose_estimation/sessions.py",
    "src/pose_estimation/video_io.py",
    "tests/test_calibration_qc.py",
    "tests/test_measure.py",
    "tests/test_qualify.py",
    "tests/test_sessions.py",
)

# Each sweep republishes under one perturbation and demands all three artifact
# digests equal the baseline. Q01-Q11 mirror the qualifier matrix; Q12 is F1a's
# own, because F1a is the first publisher taking repeatable evidence arguments.
SWEEPS: tuple[tuple[str, str], ...] = (
    ("Q01", "fresh process, identical arguments, immediate repeat"),
    ("Q02a", "PYTHONHASHSEED=0"),
    ("Q02b", "PYTHONHASHSEED=1"),
    ("Q02c", "PYTHONHASHSEED=4294967295"),
    ("Q02d", "PYTHONHASHSEED=random"),
    ("Q03a", "LC_ALL=C"),
    ("Q03b", "LC_ALL=C.UTF-8"),
    ("Q03c", "LC_ALL=en_US.UTF-8"),
    ("Q03d", "LC_ALL and LANG unset"),
    ("Q04", "shuffled Path.iterdir order over every input directory"),
    ("Q05a", "inputs reached through `..` detours"),
    ("Q05b", "inputs given relative to another cwd"),
    ("Q05c", "inputs reached through symlink aliases, links left intact"),
    ("Q06", "different output directory name"),
    ("Q07a", "TZ=UTC"),
    ("Q07b", "TZ=Pacific/Kiritimati"),
    ("Q08", "late second repeat, after the tamper phase warms the tree"),
    ("Q09", "umask 077"),
    ("Q10", "python -O"),
    ("Q11", "same-process republish over the live tree"),
    ("Q12", "repeated evidence arguments reversed, bytes unchanged"),
)

# Every tamper class the consumer boundary must refuse. `verdict` accepts only
# `calibration_qc.CalibrationQcError`; a different exception class is WRONG
# CLASS, and no message text decides a row. Input-validation negatives stay in
# pytest, because they must refuse BEFORE publication rather than after a
# published tree is edited.
TAMPERS: tuple[tuple[str, str], ...] = (
    ("T01", "clean tree accepted (control)"),
    ("T02", "corpus_qc.csv cell edited"),
    ("T03", "evidence_qc.csv cell edited"),
    ("T04", "marker census edited"),
    ("T05", "corpus_qc.csv deleted"),
    ("T06", "evidence_qc.csv deleted"),
    ("T07", "unexpected file added to the tree"),
    ("T08", "generation key removed"),
    ("T09", "generation key added"),
    ("T10", "generator_version rewritten"),
    ("T11", "upstream qualification/ regenerated after publication"),
    ("T12", "corpus_qc.csv truncated mid-row"),
    ("T13", "evidence_qc.csv rows reordered, bytes otherwise identical"),
    ("T14", "corpus_qc.csv replaced by a symlink to identical bytes"),
    ("T15", "marker replaced by non-JSON bytes"),
    ("T16", "marker rewritten with a duplicate JSON key"),
    ("T17", "marker replaced by a symlink to identical bytes"),
    ("T18", "cited probe script edited after publication (A04 staleness)"),
)

UNKNOWN = "unknown"


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digests() -> dict[str, str]:
    return {name: _digest(ROOT / name) for name in SOURCE_FILES}


def _refuse_stale(result_path: pathlib.Path, digests: dict[str, str]) -> None:
    """Exit 2 when an existing result was produced under different sources.

    Removal stays operator-explicit: a checker that silently overwrites its own
    evidence cannot be cited as evidence.
    """
    if not result_path.exists():
        return
    try:
        previous = json.loads(result_path.read_text())["source_digests"]
    except (OSError, ValueError, KeyError):
        previous = None
    if previous != digests:
        print(f"stale result at {result_path}: source digests moved; remove it and rerun")
        raise SystemExit(2)


def run_sweep(sweep_id: str, workdir: pathlib.Path) -> str:
    """Return PASS or FAIL for one sweep. Seed returns `unknown`."""
    return UNKNOWN


def run_tamper(tamper_id: str, workdir: pathlib.Path) -> str:
    """Return PASS, FAIL or WRONG CLASS for one tamper class. Seed returns `unknown`."""
    return UNKNOWN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--result",
        default=str(ROOT / "tests" / "calibration_qc_determinism_results.json"),
        help="Path to write the campaign result into.",
    )
    parser.add_argument("--workdir", default=None, help="Directory to build fixtures in.")
    args = parser.parse_args(argv)

    result_path = pathlib.Path(args.result)
    digests = source_digests()
    _refuse_stale(result_path, digests)

    workdir = (
        pathlib.Path(args.workdir) if args.workdir else result_path.parent / ".cqc-determinism"
    )
    workdir.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {"source_digests": digests, "sweeps": {}, "tampers": {}}
    sweeps: dict[str, str] = result["sweeps"]  # type: ignore[assignment]
    tampers: dict[str, str] = result["tampers"]  # type: ignore[assignment]

    for sweep_id, description in SWEEPS:
        sweeps[sweep_id] = run_sweep(sweep_id, workdir)
        print(f"{sweep_id} {sweeps[sweep_id]:<12} {description}")
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for tamper_id, description in TAMPERS:
        tampers[tamper_id] = run_tamper(tamper_id, workdir)
        print(f"{tamper_id} {tampers[tamper_id]:<12} {description}")
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    verdicts = list(sweeps.values()) + list(tampers.values())
    unknown = verdicts.count(UNKNOWN)
    failed = sum(1 for verdict in verdicts if verdict not in {"PASS", UNKNOWN})
    print(
        f"{len(SWEEPS)} sweeps / {len(TAMPERS)} tampers · "
        f"{len(verdicts) - unknown - failed} PASS · {failed} FAIL · {unknown} unknown"
    )
    return 1 if (unknown or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
