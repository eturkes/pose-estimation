#!/usr/bin/env python3
"""Determinism + tamper campaign for the `calibration_qc/` publisher (M2.7.1).

Every sweep republishes supplied evidence without running an estimator. The
fixture binds one synthetic, missing-media qualification generation plus fixed
captured probe stdout; no corpus recording is decoded or inferred over.

Results stream after every row. An existing result must match the exact source
tripwire before any write; removal stays operator-explicit.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

ARTIFACTS: tuple[str, ...] = ("corpus_qc.csv", "evidence_qc.csv", "calibration_qc.json")
MARKER = "calibration_qc.json"
TAMPER_EDIT = "evidence_qc.csv"
TAMPER_DELETE = "corpus_qc.csv"

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


@dataclasses.dataclass(frozen=True)
class Inputs:
    inventory: pathlib.Path
    sessions: pathlib.Path
    qualification: pathlib.Path
    evidence: pathlib.Path
    probes: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Campaign:
    inputs: Inputs
    baseline_dir: pathlib.Path
    baseline: dict[str, str]


_CAMPAIGNS: dict[pathlib.Path, Campaign] = {}


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests(out: pathlib.Path) -> dict[str, str]:
    return {name: _digest(out / name) for name in ARTIFACTS}


def source_digests() -> dict[str, str]:
    return {name: _digest(ROOT / name) for name in SOURCE_FILES}


def _refuse_stale(result_path: pathlib.Path, current: dict[str, str]) -> None:
    if not result_path.exists():
        return
    try:
        previous = json.loads(result_path.read_text(encoding="utf-8"))["source_digests"]
    except (OSError, UnicodeDecodeError, ValueError, KeyError):
        previous = None
    if previous != current:
        print(
            f"REFUSED: stale result at {result_path}; source digests moved. "
            "Remove it explicitly before regeneration.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "PYTHONHASHSEED", "PYTHONUNBUFFERED"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def build_fixture(root: pathlib.Path) -> Inputs:
    """Build one no-decode qualification plus captured probe evidence."""
    root.mkdir(parents=True)
    sys.path.insert(0, str(TESTS))
    from pose_estimation import qualify
    from test_calibration_qc import _write_evidence, _write_probes
    from test_qualify import _publish
    from test_sessions import _canonical

    inventory, sessions, corpus, qualification = _publish(root, [_canonical(1, "above")])
    # The source is deliberately absent. Qualify publishes its missing-media
    # status without opening a decoder, which is enough to bind a valid chain.
    qualify.run(inventory, sessions, corpus, qualification)
    probes = _write_probes(root / "probes")
    evidence = root / "evidence"
    _write_evidence(evidence, probes)
    return Inputs(inventory, sessions, qualification, evidence, probes)


def _worker_command(
    out: pathlib.Path,
    inputs: Inputs,
    *,
    qualification: str | os.PathLike[str] | None = None,
    evidence: Sequence[str | os.PathLike[str]] | None = None,
    probes: str | os.PathLike[str] | None = None,
    inventory: str | os.PathLike[str] | None = None,
    sessions: str | os.PathLike[str] | None = None,
    optimize: bool = False,
    shuffle_seed: int | None = None,
    republish: bool = False,
) -> list[str]:
    command = [sys.executable]
    if optimize:
        command.append("-O")
    command += [
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        "--qualification",
        os.fspath(qualification or inputs.qualification),
    ]
    for path in evidence or (inputs.evidence,):
        command += ["--evidence", os.fspath(path)]
    command += [
        "--probes",
        os.fspath(probes or inputs.probes),
        "--out",
        os.fspath(out),
        "--sessions",
        os.fspath(sessions or inputs.sessions),
        "--inventory",
        os.fspath(inventory or inputs.inventory),
    ]
    if shuffle_seed is not None:
        command += ["--shuffle-seed", str(shuffle_seed)]
    if republish:
        command.append("--republish")
    return command


def run_cli(
    out: pathlib.Path,
    inputs: Inputs,
    *,
    qualification: str | os.PathLike[str] | None = None,
    evidence: Sequence[str | os.PathLike[str]] | None = None,
    probes: str | os.PathLike[str] | None = None,
    inventory: str | os.PathLike[str] | None = None,
    sessions: str | os.PathLike[str] | None = None,
    env_overrides: dict[str, str | None] | None = None,
    cwd: pathlib.Path = ROOT,
    optimize: bool = False,
    umask: int | None = None,
    shuffle_seed: int | None = None,
    republish: bool = False,
) -> dict[str, str]:
    env = _base_env()
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    completed = subprocess.run(
        _worker_command(
            out,
            inputs,
            qualification=qualification,
            evidence=evidence,
            probes=probes,
            inventory=inventory,
            sessions=sessions,
            optimize=optimize,
            shuffle_seed=shuffle_seed,
            republish=republish,
        ),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=(lambda: os.umask(umask)) if umask is not None else None,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"publisher worker rc={completed.returncode}: {detail[-600:]}")
    return digests(out if out.is_absolute() else cwd / out)


def _prepare(workdir: pathlib.Path) -> Campaign:
    key = workdir.resolve()
    if key in _CAMPAIGNS:
        return _CAMPAIGNS[key]
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    inputs = build_fixture(workdir / "fixture")
    baseline_dir = workdir / "baseline"
    campaign = Campaign(inputs, baseline_dir, run_cli(baseline_dir, inputs))
    _CAMPAIGNS[key] = campaign
    return campaign


def _same(observed: dict[str, str], baseline: dict[str, str]) -> str:
    return "PASS" if observed == baseline else "FAIL"


def run_sweep(sweep_id: str, workdir: pathlib.Path) -> str:
    """Return PASS or FAIL for one encoded sweep; unfilled rows stay `unknown`."""
    campaign = _prepare(workdir)
    out = workdir / "sweep"
    try:
        if sweep_id == "Q01":
            return _same(run_cli(out, campaign.inputs), campaign.baseline)
        if sweep_id in {"Q02a", "Q02b", "Q02c", "Q02d"}:
            seed = {
                "Q02a": "0",
                "Q02b": "1",
                "Q02c": "4294967295",
                "Q02d": "random",
            }[sweep_id]
            observed = run_cli(out, campaign.inputs, env_overrides={"PYTHONHASHSEED": seed})
            return _same(observed, campaign.baseline)
        if sweep_id == "Q03a":
            observed = run_cli(out, campaign.inputs, env_overrides={"LC_ALL": "C", "LANG": "C"})
            return _same(observed, campaign.baseline)
        if sweep_id in {"Q03b", "Q03c"}:
            locale = {"Q03b": "C.UTF-8", "Q03c": "en_US.UTF-8"}[sweep_id]
            observed = run_cli(
                out, campaign.inputs, env_overrides={"LC_ALL": locale, "LANG": locale}
            )
            return _same(observed, campaign.baseline)
        if sweep_id == "Q03d":
            observed = run_cli(out, campaign.inputs, env_overrides={"LC_ALL": None, "LANG": None})
            return _same(observed, campaign.baseline)
        if sweep_id == "Q04":
            observed = run_cli(out, campaign.inputs, shuffle_seed=8117)
            return _same(observed, campaign.baseline)
        if sweep_id == "Q05a":
            detour = workdir / "detour"
            detour.mkdir(exist_ok=True)

            def dotdot(path: pathlib.Path) -> pathlib.Path:
                return detour / ".." / path.relative_to(workdir)

            observed = run_cli(
                detour / ".." / "sweep-dotdot",
                campaign.inputs,
                qualification=dotdot(campaign.inputs.qualification),
                evidence=(dotdot(campaign.inputs.evidence),),
                probes=dotdot(campaign.inputs.probes),
                inventory=dotdot(campaign.inputs.inventory),
                sessions=dotdot(campaign.inputs.sessions),
            )
            return _same(observed, campaign.baseline)
        if sweep_id == "Q05b":

            def relative(path: pathlib.Path) -> str:
                return os.path.relpath(path, workdir)

            observed = run_cli(
                pathlib.Path("sweep-relative"),
                campaign.inputs,
                qualification=relative(campaign.inputs.qualification),
                evidence=(relative(campaign.inputs.evidence),),
                probes=relative(campaign.inputs.probes),
                inventory=relative(campaign.inputs.inventory),
                sessions=relative(campaign.inputs.sessions),
                cwd=workdir,
            )
            return _same(observed, campaign.baseline)
        if sweep_id == "Q05c":
            aliases = workdir / "aliases"
            aliases.mkdir(exist_ok=True)
            for field in dataclasses.fields(Inputs):
                target = getattr(campaign.inputs, field.name).resolve()
                (aliases / field.name).symlink_to(target, target_is_directory=True)
            target_out = workdir / "symlink-output-target"
            target_out.mkdir()
            alias_out = aliases / "out"
            alias_out.symlink_to(target_out.resolve(), target_is_directory=True)
            observed = run_cli(
                alias_out,
                campaign.inputs,
                qualification=aliases / "qualification",
                evidence=(aliases / "evidence",),
                probes=aliases / "probes",
                inventory=aliases / "inventory",
                sessions=aliases / "sessions",
            )
            links_intact = alias_out.is_symlink() and all(
                (aliases / field.name).is_symlink() for field in dataclasses.fields(Inputs)
            )
            return _same(observed, campaign.baseline) if links_intact else "FAIL"
        if sweep_id == "Q06":
            observed = run_cli(workdir / "differently named output", campaign.inputs)
            return _same(observed, campaign.baseline)
        if sweep_id in {"Q07a", "Q07b"}:
            timezone = {"Q07a": "UTC", "Q07b": "Pacific/Kiritimati"}[sweep_id]
            observed = run_cli(out, campaign.inputs, env_overrides={"TZ": timezone})
            return _same(observed, campaign.baseline)
        if sweep_id == "Q08":
            from pose_estimation import calibration_qc

            warm = workdir / "q08-warm"
            shutil.copytree(campaign.baseline_dir, warm)
            (warm / "unexpected.bin").write_bytes(b"x")
            try:
                calibration_qc.validate_generation(
                    warm,
                    qualification_dir=campaign.inputs.qualification,
                    sessions_dir=campaign.inputs.sessions,
                    inventory_dir=campaign.inputs.inventory,
                    probes_dir=campaign.inputs.probes,
                )
            except calibration_qc.CalibrationQcError:
                pass
            else:
                return "FAIL"
            observed = run_cli(workdir / "late-repeat", campaign.inputs)
            return _same(observed, campaign.baseline)
        if sweep_id == "Q09":
            observed = run_cli(out, campaign.inputs, umask=0o077)
            return _same(observed, campaign.baseline)
        if sweep_id == "Q10":
            observed = run_cli(out, campaign.inputs, optimize=True)
            return _same(observed, campaign.baseline)
        if sweep_id == "Q11":
            observed = run_cli(out, campaign.inputs, republish=True)
            return _same(observed, campaign.baseline)
        if sweep_id == "Q12":
            first = workdir / "evidence-order-a"
            second = workdir / "evidence-order-b"
            shutil.copytree(campaign.inputs.evidence, first)
            shutil.copytree(campaign.inputs.evidence, second)
            before = {
                path.relative_to(workdir): path.read_bytes()
                for root in (first, second)
                for path in root.iterdir()
            }
            forward = run_cli(
                workdir / "evidence-order-forward",
                campaign.inputs,
                evidence=(first, second),
            )
            reverse = run_cli(
                workdir / "evidence-order-reverse",
                campaign.inputs,
                evidence=(second, first),
            )
            after = {
                path.relative_to(workdir): path.read_bytes()
                for root in (first, second)
                for path in root.iterdir()
            }
            return "PASS" if forward == reverse == campaign.baseline and before == after else "FAIL"
    except (OSError, RuntimeError, ValueError) as error:
        print(f"{sweep_id} detail: {error}", file=sys.stderr)
        return "FAIL"
    return UNKNOWN


def _validation_verdict(directory: pathlib.Path, inputs: Inputs, *, accept: bool) -> str:
    from pose_estimation import calibration_qc

    try:
        calibration_qc.validate_generation(
            directory,
            qualification_dir=inputs.qualification,
            sessions_dir=inputs.sessions,
            inventory_dir=inputs.inventory,
            probes_dir=inputs.probes,
        )
    except Exception as error:
        if type(error) is not calibration_qc.CalibrationQcError:
            return "WRONG CLASS"
        return "FAIL" if accept else "PASS"
    return "PASS" if accept else "FAIL"


def _tamper_tree(workdir: pathlib.Path, tamper_id: str, campaign: Campaign) -> pathlib.Path:
    directory = workdir / f"tamper-{tamper_id.lower()}"
    shutil.copytree(campaign.baseline_dir, directory)
    return directory


def _flip_first_data_byte(path: pathlib.Path) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    row = bytearray(lines[1])
    row[0] = ord("z") if row[0] != ord("z") else ord("y")
    lines[1] = bytes(row)
    path.write_bytes(b"".join(lines))


def _rewrite_marker(path: pathlib.Path, mutation: str) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    generation = document["generation"]
    if mutation == "census":
        document["corpus"]["rows"] += 1
    elif mutation == "generation_removed":
        generation.pop("tree")
    elif mutation == "generation_added":
        generation["unexpected"] = "value"
    elif mutation == "generator_version":
        generation["generator_version"] = "not-this-generator"
    else:
        raise AssertionError(f"unhandled marker mutation: {mutation}")
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_tamper(tamper_id: str, workdir: pathlib.Path) -> str:
    """Return PASS, FAIL or WRONG CLASS for one encoded tamper class."""
    campaign = _prepare(workdir)
    if tamper_id == "T01":
        return _validation_verdict(campaign.baseline_dir, campaign.inputs, accept=True)
    if tamper_id in {"T02", "T03"}:
        directory = _tamper_tree(workdir, tamper_id, campaign)
        filename = "corpus_qc.csv" if tamper_id == "T02" else "evidence_qc.csv"
        _flip_first_data_byte(directory / filename)
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T04":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        _rewrite_marker(directory / MARKER, "census")
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id in {"T05", "T06"}:
        directory = _tamper_tree(workdir, tamper_id, campaign)
        filename = "corpus_qc.csv" if tamper_id == "T05" else "evidence_qc.csv"
        (directory / filename).unlink()
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T07":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        (directory / "unexpected.bin").write_bytes(b"x")
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id in {"T08", "T09"}:
        directory = _tamper_tree(workdir, tamper_id, campaign)
        mutation = "generation_removed" if tamper_id == "T08" else "generation_added"
        _rewrite_marker(directory / MARKER, mutation)
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T10":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        _rewrite_marker(directory / MARKER, "generator_version")
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T11":
        sys.path.insert(0, str(TESTS))
        from pose_estimation import qualify
        from test_qualify import _publish
        from test_sessions import _canonical

        rebuilt_root = workdir / "rebuilt-qualification"
        rebuilt_root.mkdir()
        inventory, sessions, corpus, qualification = _publish(rebuilt_root, [_canonical(2, "left")])
        qualify.run(inventory, sessions, corpus, qualification)
        actual = dataclasses.replace(
            campaign.inputs,
            inventory=inventory,
            sessions=sessions,
            qualification=qualification,
        )
        return _validation_verdict(campaign.baseline_dir, actual, accept=False)
    if tamper_id == "T12":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        path = directory / "corpus_qc.csv"
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T13":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        path = directory / "evidence_qc.csv"
        lines = path.read_bytes().splitlines(keepends=True)
        lines[1], lines[2] = lines[2], lines[1]
        path.write_bytes(b"".join(lines))
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T14":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        path = directory / "corpus_qc.csv"
        external = workdir / "tamper-t14-target.csv"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external.resolve())
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T15":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        (directory / MARKER).write_bytes(b"not JSON\n")
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T16":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        marker = directory / MARKER
        text = marker.read_text(encoding="utf-8")
        marker.write_text(
            text.replace('  "generation": {', '  "generation": {},\n  "generation": {', 1),
            encoding="utf-8",
        )
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T17":
        directory = _tamper_tree(workdir, tamper_id, campaign)
        marker = directory / MARKER
        external = workdir / "tamper-t17-target.json"
        external.write_bytes(marker.read_bytes())
        marker.unlink()
        marker.symlink_to(external.resolve())
        return _validation_verdict(directory, campaign.inputs, accept=False)
    if tamper_id == "T18":
        probes = workdir / "tamper-t18-probes"
        shutil.copytree(campaign.inputs.probes, probes)
        script = probes / "probe_calibration_bias.py"
        script.write_text(script.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
        actual = dataclasses.replace(campaign.inputs, probes=probes)
        return _validation_verdict(campaign.baseline_dir, actual, accept=False)
    return UNKNOWN


def _write_result(path: pathlib.Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    current_sources = source_digests()
    _refuse_stale(result_path, current_sources)
    workdir = (
        pathlib.Path(args.workdir) if args.workdir else result_path.parent / ".cqc-determinism"
    )
    campaign = _prepare(workdir)
    result: dict[str, object] = {
        "schema_version": 1,
        "source_digests": current_sources,
        "baseline_sha256": campaign.baseline,
        "sweeps": {},
        "tampers": {},
    }
    sweeps: dict[str, str] = result["sweeps"]  # type: ignore[assignment]
    tampers: dict[str, str] = result["tampers"]  # type: ignore[assignment]
    _write_result(result_path, result)

    for sweep_id, description in SWEEPS:
        sweeps[sweep_id] = run_sweep(sweep_id, workdir)
        print(f"{sweep_id} {sweeps[sweep_id]:<12} {description}", flush=True)
        _write_result(result_path, result)
    for tamper_id, description in TAMPERS:
        tampers[tamper_id] = run_tamper(tamper_id, workdir)
        print(f"{tamper_id} {tampers[tamper_id]:<12} {description}", flush=True)
        _write_result(result_path, result)

    verdicts = list(sweeps.values()) + list(tampers.values())
    unknown = verdicts.count(UNKNOWN)
    failed = sum(1 for verdict in verdicts if verdict not in {"PASS", UNKNOWN})
    passed = len(verdicts) - unknown - failed
    print(
        f"{len(SWEEPS)} sweeps / {len(TAMPERS)} tampers · "
        f"{passed} PASS · {failed} FAIL · {unknown} unknown"
    )
    return 1 if unknown or failed else 0


def worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--evidence", required=True, action="append")
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--republish", action="store_true")
    args = parser.parse_args(argv)

    from pose_estimation import calibration_qc

    qc_args = ["--qualification", args.qualification]
    for evidence in args.evidence:
        qc_args += ["--evidence", evidence]
    qc_args += [
        "--probes",
        args.probes,
        "--out",
        args.out,
        "--sessions",
        args.sessions,
        "--inventory",
        args.inventory,
    ]

    def publish() -> int:
        code = calibration_qc.main(qc_args)
        if code == 0 and args.republish:
            code = calibration_qc.main(qc_args)
        return code

    if args.shuffle_seed is None:
        return publish()
    rng = random.Random(args.shuffle_seed)
    original = pathlib.Path.iterdir
    changed: list[str] = []

    def scrambled(path: pathlib.Path) -> Iterator[pathlib.Path]:
        entries = list(original(path))
        shuffled = list(entries)
        rng.shuffle(shuffled)
        if shuffled != entries:
            changed.append(path.name)
        return iter(shuffled)

    with mock.patch.object(pathlib.Path, "iterdir", scrambled):
        code = publish()
    if code == 0 and not changed:
        print("shuffle positive control failed: no directory order changed", file=sys.stderr)
        return 3
    return code


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--worker"]:
        raise SystemExit(worker_main(arguments[1:]))
    raise SystemExit(main(arguments))
