#!/usr/bin/env python3
"""Prove that qualification artifacts depend only on their declared inputs.

Each sweep runs the committed publisher in a subprocess. It compares all four
published files with one synthetic baseline by SHA-256. The fixture exercises
PyAV demux, the QuickTime atom walk, non-uniform PTS formatting, two upstreams,
and multi-key census tallies without reading patient-adjacent data. A child
process also proves that sidecar ingestion uses the bytes cached by validation.

Results stream to ``tests/qualify_determinism_results.json`` after every sweep.
An existing result must match every recorded source digest before any write.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
ARTIFACTS = ("assets_qc.csv", "pairs_qc.csv", "events_qc.csv", "qualification.json")
SOURCE_FILES = (
    "scripts/check_qualify_determinism.py",
    "scripts/probe_sync_policy.py",
    "src/pose_estimation/__init__.py",
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
    "tests/test_measure.py",
    "tests/test_qualify.py",
    "tests/test_sessions.py",
)


@dataclasses.dataclass(frozen=True)
class Inputs:
    inventory: pathlib.Path
    sessions: pathlib.Path
    corpus: pathlib.Path
    measurements: pathlib.Path


def digests(out: pathlib.Path) -> dict[str, str]:
    return {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ARTIFACTS}


def source_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def stale_source_mismatches(output: pathlib.Path, current: dict[str, str]) -> list[str]:
    try:
        previous = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return sorted(current)
    recorded = previous.get("source_sha256") if isinstance(previous, dict) else None
    if not isinstance(recorded, dict):
        return sorted(current)
    recorded_sources = {str(name): digest for name, digest in recorded.items()}
    names = set(current) | set(recorded_sources)
    return sorted(name for name in names if recorded_sources.get(name) != current.get(name))


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "PYTHONHASHSEED", "PYTHONUNBUFFERED"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def build_fixture(root: pathlib.Path) -> Inputs:
    """Reuse the acceptance-suite builders to make a decode-bearing corpus."""
    root.mkdir()
    sys.path.insert(0, str(TESTS))
    from pose_estimation import sessions
    from test_qualify import _write_media
    from test_sessions import _canonical, _write_registry

    assets = [
        _canonical(1, "above"),
        _canonical(1, "left"),
        _canonical(1, "right"),
        _canonical(2, "above"),
    ]
    registry = _write_registry(root, assets)
    sessions.run(registry.root, registry.corpus, registry.out)

    increments = (17, 23, 31)
    non_uniform = [0]
    for index in range(29):
        non_uniform.append(non_uniform[-1] + increments[index % len(increments)])
    _write_media(registry.corpus / assets[0].source_path, non_uniform)
    _write_media(registry.corpus / assets[1].source_path, [index * 20 for index in range(30)])
    _write_media(registry.corpus / assets[2].source_path, [index * 19 for index in range(30)])
    # The fourth source stays non-media, pinning a second decode-status tally.

    # Enumerated rather than transcribed: `_ingest` reconciles the sidecar
    # against `enumerate_pairs`, so building the rows from that same call is
    # what keeps the fixture from drifting into an unreconcilable sidecar.
    from pose_estimation import measure, qualify

    measurements = root / "measurements"
    rows = [
        {
            "capture_id": capture_id,
            "asset_a": first.asset_id,
            "asset_b": second.asset_id,
            "offset_audio_s": f"{(index + 1) * 0.125:.9f}",
            "peak_rms_audio": "5.000000000",
            "peak_ratio_audio": "2.500000000",
            "status_audio": "ok",
            "drift_ppm": "",
            "drift_se": "",
            "offset_visual_s": "",
            "conf_visual": "",
            "peak_corr_visual": "",
            "status_visual": "low_peak_correlation",
            "overlap_s": "4.000000000",
            "dur_a": "5.000000000",
            "dur_b": "5.000000000",
            "same_audio_rate": "1",
        }
        for index, (capture_id, first, second) in enumerate(
            qualify.enumerate_pairs(qualify.load_assets(registry.root))
        )
    ]
    measure.write_axis(
        measurements, "sync", rows, {"fixture": "determinism"}, inventory_dir=registry.root
    )
    return Inputs(registry.root, registry.out, registry.corpus, measurements)


def run_cli(
    out: pathlib.Path,
    inputs: Inputs,
    *,
    inventory: str | os.PathLike[str] | None = None,
    sessions: str | os.PathLike[str] | None = None,
    corpus: str | os.PathLike[str] | None = None,
    measurements: str | os.PathLike[str] | None = None,
    measured: bool = False,
    env_overrides: dict[str, str | None] | None = None,
    cwd: pathlib.Path = ROOT,
    optimize: bool = False,
    umask: int | None = None,
    shuffle_seed: int | None = None,
    republish: bool = False,
) -> dict[str, str]:
    """Run one hostile-but-legal variation and return artifact digests."""
    env = base_env()
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    command = [sys.executable]
    if optimize:
        command.append("-O")
    command += [
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        "--inventory",
        os.fspath(inventory or inputs.inventory),
        "--sessions",
        os.fspath(sessions or inputs.sessions),
        "--corpus",
        os.fspath(corpus or inputs.corpus),
        "--out",
        str(out),
    ]
    if measured:
        command += ["--measurements", os.fspath(measurements or inputs.measurements)]
    if shuffle_seed is not None:
        command += ["--shuffle-seed", str(shuffle_seed)]
    if republish:
        command.append("--republish")

    preexec = (lambda: os.umask(umask)) if umask is not None else None
    completed = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, preexec_fn=preexec
    )
    if completed.returncode:
        raise SystemExit(
            f"qualification run failed rc={completed.returncode}: {completed.stderr[-400:]}"
        )
    return digests(out)


def sweeps(
    work: pathlib.Path, inputs: Inputs, *, measured: bool
) -> Iterator[tuple[str, str, dict[str, str]]]:
    """Yield every subprocess variation and its four artifact digests.

    The mode is bound once rather than threaded through every variation: it is
    a property of the whole sweep set, and each set is compared against its own
    mode's baseline.
    """
    run = functools.partial(run_cli, measured=measured)
    out = work / "sweep"

    yield "Q01", "repeat in a fresh process", run(out, inputs)

    for seed in ("0", "1", "12345", "random"):
        yield (
            f"Q02:{seed}",
            f"PYTHONHASHSEED={seed}",
            run(out, inputs, env_overrides={"PYTHONHASHSEED": seed}),
        )

    for locale in ("C", "C.UTF-8", "en_US.UTF-8"):
        yield (
            f"Q03:{locale}",
            f"LC_ALL={locale}",
            run(out, inputs, env_overrides={"LC_ALL": locale, "LANG": locale}),
        )
    yield (
        "Q03:unset",
        "LANG and LC_ALL unset",
        run(out, inputs, env_overrides={"LC_ALL": None, "LANG": None}),
    )

    yield (
        "Q04",
        "shuffled Path.iterdir in a subprocess",
        run(out, inputs, shuffle_seed=8117),
    )

    detour = inputs.corpus / ".."
    yield (
        "Q05:dotdot",
        "all input paths with .. detours",
        run(
            out,
            inputs,
            inventory=detour / inputs.inventory.name,
            sessions=detour / inputs.sessions.name,
            corpus=inputs.inventory / ".." / inputs.corpus.name,
        ),
    )
    yield (
        "Q05:relative",
        "relative inputs from a different working directory",
        run(
            out,
            inputs,
            inventory=inputs.inventory.name,
            sessions=inputs.sessions.name,
            corpus=inputs.corpus.name,
            cwd=inputs.inventory.parent,
        ),
    )
    aliases = work / "aliases"
    aliases.mkdir(exist_ok=True)
    for name, target in dataclasses.asdict(inputs).items():
        link = aliases / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    yield (
        "Q05:symlinks",
        "equivalent symlink spellings for both upstreams and corpus",
        run(
            out,
            inputs,
            inventory=aliases / "inventory",
            sessions=aliases / "sessions",
            corpus=aliases / "corpus",
        ),
    )

    yield (
        "Q06",
        "different --out directory name",
        run(work / "differently named output", inputs),
    )

    for timezone in ("UTC", "Pacific/Kiritimati"):
        yield (
            f"Q07:{timezone}",
            f"TZ={timezone}",
            run(out, inputs, env_overrides={"TZ": timezone}),
        )

    yield "Q08", "second late repeat", run(work / "late", inputs)
    yield "Q09", "umask 077", run(out, inputs, umask=0o077)
    yield "Q10", "interpreter -O", run(out, inputs, optimize=True)
    yield (
        "Q11",
        "same-process PyAV demux and publication repeat",
        run(out, inputs, republish=True),
    )


def write_payload(output: pathlib.Path, payload: dict[str, object]) -> None:
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def ingestion_cache_verdict(inputs: Inputs, work: pathlib.Path) -> str:
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--ingestion-worker",
        "--inventory",
        str(inputs.inventory),
        "--work",
        str(work),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=base_env(), capture_output=True, text=True)
    if completed.returncode:
        detail = (
            completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no error"
        )
        return f"FAILED: {detail}"
    return "validated bytes preserved"


def _rewrite_marker(path: pathlib.Path, mutation: str) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    generation = document["generation"]
    if mutation == "census_value":
        document["assets"]["rows"] += 1
    elif mutation == "generation_removed":
        generation.pop("tree")
    elif mutation == "generation_added":
        generation["unexpected"] = "value"
    elif mutation == "generator_version":
        generation["generator_version"] = "not-this-generator"
    elif mutation == "measurements_added":
        generation["measurements"] = "0" * 64
    elif mutation == "measurements_removed":
        generation.pop("measurements")
    else:
        raise AssertionError(f"unhandled marker mutation: {mutation}")
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def tamper_verdicts(
    out: pathlib.Path, measured_out: pathlib.Path, inputs: Inputs, work: pathlib.Path
) -> dict[str, str]:
    """Put every tamper class through the public consumer boundary.

    Both published modes are covered because the mode is itself a claim in the
    marker: a flagless set that gains a `measurements` key and a measured set
    that loses one are each a forged provenance story, and neither shows up in
    a sweep of the flagless tree alone.
    """
    sys.path.insert(0, str(TESTS))
    from pose_estimation import qualify
    from pose_estimation import sessions as sessions_module
    from test_sessions import _canonical, _write_registry

    rebuilt_root = work / "rebuilt-upstreams"
    rebuilt_root.mkdir()
    rebuilt = _write_registry(rebuilt_root, [_canonical(9, "above")])
    sessions_module.run(rebuilt.root, rebuilt.corpus, rebuilt.out)

    def verdict(directory: pathlib.Path, actual_inputs: Inputs, expectation: str) -> str:
        try:
            qualify.validate_generation(
                directory,
                sessions_dir=actual_inputs.sessions,
                inventory_dir=actual_inputs.inventory,
            )
        except qualify.QualifyError:
            return "rejected" if expectation == "reject" else "REJECTED A VALID SET"
        except Exception as error:
            return f"WRONG CLASS: {type(error).__name__}"
        return "accepted" if expectation == "accept" else "ACCEPTED A TAMPERED SET"

    labels = (
        "edited CSV cell",
        "edited census value",
        "deleted CSV",
        "added file",
        "removed generation key",
        "added generation key",
        "wrong generator_version",
        "upstream inventory rebuilt",
        "upstream sessions rebuilt",
        "truncated CSV",
        "CSV reordered row",
        "symlink swapped into tree",
        "qualification.json non-JSON",
        "qualification.json duplicate key",
        "qualification.json symlinked",
    )
    verdicts = {
        "clean": verdict(out, inputs, "accept"),
        "clean measured": verdict(measured_out, inputs, "accept"),
    }
    for index, label in enumerate(labels):
        copy = work / f"tamper-{index:02d}"
        shutil.copytree(out, copy)
        actual_inputs = inputs
        assets = copy / ARTIFACTS[0]
        marker = copy / ARTIFACTS[-1]
        if label == "edited CSV cell":
            lines = assets.read_bytes().splitlines(keepends=True)
            row = bytearray(lines[1])
            row[0] = ord("z") if row[0] != ord("z") else ord("y")
            lines[1] = bytes(row)
            assets.write_bytes(b"".join(lines))
        elif label == "edited census value":
            _rewrite_marker(marker, "census_value")
        elif label == "deleted CSV":
            (copy / ARTIFACTS[1]).unlink()
        elif label == "added file":
            (copy / "unexpected.bin").write_bytes(b"x")
        elif label == "removed generation key":
            _rewrite_marker(marker, "generation_removed")
        elif label == "added generation key":
            _rewrite_marker(marker, "generation_added")
        elif label == "wrong generator_version":
            _rewrite_marker(marker, "generator_version")
        elif label == "upstream inventory rebuilt":
            actual_inputs = dataclasses.replace(inputs, inventory=rebuilt.root)
        elif label == "upstream sessions rebuilt":
            actual_inputs = dataclasses.replace(inputs, sessions=rebuilt.out)
        elif label == "truncated CSV":
            raw = assets.read_bytes()
            assets.write_bytes(raw[: len(raw) // 2])
        elif label == "CSV reordered row":
            lines = assets.read_bytes().splitlines(keepends=True)
            lines[1], lines[2] = lines[2], lines[1]
            assets.write_bytes(b"".join(lines))
        elif label == "symlink swapped into tree":
            external = work / f"symlink-target-{index}.csv"
            external.write_bytes(assets.read_bytes())
            assets.unlink()
            assets.symlink_to(external)
        elif label == "qualification.json non-JSON":
            marker.write_bytes(b"not JSON\n")
        elif label == "qualification.json duplicate key":
            text = marker.read_text(encoding="utf-8")
            marker.write_text('{"assets":{"rows":999},' + text[1:], encoding="utf-8")
        elif label == "qualification.json symlinked":
            external = work / f"marker-target-{index}.json"
            external.write_bytes(marker.read_bytes())
            marker.unlink()
            marker.symlink_to(external)
        else:
            raise AssertionError(f"unhandled tamper class: {label}")
        verdicts[label] = verdict(copy, actual_inputs, "reject")

    for label, source, mutation in (
        ("forged measurements provenance", out, "measurements_added"),
        ("removed measurements provenance", measured_out, "measurements_removed"),
    ):
        copy = work / f"tamper-mode-{mutation}"
        shutil.copytree(source, copy)
        _rewrite_marker(copy / ARTIFACTS[-1], mutation)
        verdicts[label] = verdict(copy, inputs, "reject")

    verdicts["validated sync cell changed after validation"] = ingestion_cache_verdict(
        inputs, work / "ingestion-cache"
    )
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "tests" / "qualify_determinism_results.json",
    )
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    current_sources = source_digests()
    if output.exists():
        mismatches = stale_source_mismatches(output, current_sources)
        if mismatches:
            print(
                "REFUSED: result file records different source digests; "
                f"remove it explicitly before regeneration. mismatched={','.join(mismatches)}",
                file=sys.stderr,
            )
            return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    tamper_results: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as raw:
        work = pathlib.Path(raw)
        inputs = build_fixture(work / "fixture")
        # No head SHA is recorded: the run that regenerates this file always
        # precedes the commit that carries it, so any SHA written here names the
        # parent state and can never be checked. `source_sha256` binds the
        # result to the bytes that produced it, which is the real dependency.
        baselines = {
            mode: run_cli(work / f"baseline-{mode}", inputs, measured=(mode == "measured"))
            for mode in ("flagless", "measured")
        }
        payload: dict[str, object] = {
            "schema_version": 3,
            "source_sha256": current_sources,
            "baseline_sha256": baselines,
            "sweeps": rows,
            "tamper_classes": tamper_results,
        }
        write_payload(output, payload)
        print(f"Q00 baselines {json.dumps(baselines, sort_keys=True)}", flush=True)

        for mode, baseline in baselines.items():
            for identifier, variation, observed in sweeps(
                work, inputs, measured=(mode == "measured")
            ):
                verdict = {
                    name: ("PASS" if observed[name] == baseline[name] else "FAIL")
                    for name in ARTIFACTS
                }
                rows.append(
                    {
                        "id": f"{identifier}:{mode}",
                        "variation": f"{variation} ({mode})",
                        **{f"verdict_{name}": verdict[name] for name in ARTIFACTS},
                        "observed_sha256": observed,
                    }
                )
                payload["sweeps"] = rows
                write_payload(output, payload)
                print(f"{identifier}:{mode} {' '.join(verdict.values())} {variation}", flush=True)

        tamper_results = tamper_verdicts(
            work / "baseline-flagless", work / "baseline-measured", inputs, work
        )
        payload["tamper_classes"] = tamper_results
        write_payload(output, payload)
        print(f"Q12 {json.dumps(tamper_results, sort_keys=True)}", flush=True)

    failures = [row for row in rows if any(value == "FAIL" for value in row.values())]
    tamper_failures = [
        value
        for value in tamper_results.values()
        if value.startswith(("ACCEPTED", "FAILED", "WRONG", "REJECTED"))
    ]
    print(
        f"\nsweeps={len(rows)} passed={len(rows) - len(failures)} failures={len(failures)} "
        f"tamper_classes={len(tamper_results) - 1} tamper_failures={len(tamper_failures)}"
    )
    return 1 if failures or tamper_failures else 0


def ingestion_worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(TESTS))
    from pose_estimation import inventory, measure
    from test_measure import _row

    out = pathlib.Path(args.work)
    original = _row()
    measure.write_axis(out, "sync", [original], {}, inventory_dir=args.inventory)
    sidecar = measure.validate(out, inventory_dir=args.inventory)
    changed = _row(peak_rms_audio="7.000000000")
    inventory.write_text(
        out / measure.AXES["sync"].table,
        inventory.render_csv(measure.SYNC_COLUMNS, [changed]),
    )
    key = tuple(original[name] for name in measure.AXES["sync"].keys)
    loaded = measure.load_axis(sidecar, "sync")
    if loaded.get(key, {}).get("peak_rms_audio") != original["peak_rms_audio"]:
        print("load_axis reopened the changed table", file=sys.stderr)
        return 1
    print("validated bytes preserved")
    return 0


def worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--measurements")
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--republish", action="store_true")
    args = parser.parse_args(argv)

    from pose_estimation import qualify

    qualify_args = [
        "--inventory",
        args.inventory,
        "--sessions",
        args.sessions,
        "--corpus",
        args.corpus,
        "--out",
        args.out,
    ]
    if args.measurements:
        qualify_args += ["--measurements", args.measurements]

    def publish() -> int:
        result = qualify.main(qualify_args)
        if result == 0 and args.republish:
            result = qualify.main(qualify_args)
        return result

    if args.shuffle_seed is None:
        return publish()
    rng = random.Random(args.shuffle_seed)
    original = pathlib.Path.iterdir

    def scrambled(path: pathlib.Path) -> Iterator[pathlib.Path]:
        entries = list(original(path))
        rng.shuffle(entries)
        return iter(entries)

    with mock.patch.object(pathlib.Path, "iterdir", scrambled):
        return publish()


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--ingestion-worker"]:
        raise SystemExit(ingestion_worker_main(arguments[1:]))
    if arguments[:1] == ["--worker"]:
        raise SystemExit(worker_main(arguments[1:]))
    raise SystemExit(main(arguments))
