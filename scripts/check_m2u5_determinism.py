#!/usr/bin/env python3
"""Run the fixed M2.5 row-order, output-name, and repeat sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import runpy
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable, Sequence
from typing import Any
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
determinism = types.SimpleNamespace(
    **runpy.run_path(str(ROOT / "scripts/check_qualify_determinism.py"))
)
SEED = 8117


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--measurements")
    return parser.parse_args(argv)


def _shuffled(
    values: Sequence[Any], rng: random.Random, changed: list[str], label: str
) -> list[Any]:
    original = list(values)
    result = list(original)
    rng.shuffle(result)
    if result != original:
        changed.append(label)
    return result


def shuffled_worker(argv: Sequence[str]) -> int:
    args = _arguments(argv)
    from pose_estimation import measure, qualify

    rng = random.Random(SEED)
    changed: list[str] = []
    load_events = qualify.load_events
    load_placements = qualify.load_placements
    load_axis = measure.load_axis

    # `load_assets` is deliberately not wrapped.  `assets_qc.csv` publishes in
    # registry order so a reader sees a capture's assets together, and D09 pins
    # that; permuting the loader's return asks the table to abandon a published
    # order rather than to become deterministic, and `asset_id` is a content
    # hash whose ordering carries no meaning to sort by.

    def events(*call_args: Any, **call_kwargs: Any) -> list[dict[str, str]]:
        return _shuffled(load_events(*call_args, **call_kwargs), rng, changed, "events")

    def placements(*call_args: Any, **call_kwargs: Any) -> dict[str, list[str]]:
        loaded = load_placements(*call_args, **call_kwargs)
        items = _shuffled(list(loaded.items()), rng, changed, "placement-events")
        result: dict[str, list[str]] = {}
        for event_id, members in items:
            result[event_id] = _shuffled(members, rng, changed, f"placements:{event_id}")
        return result

    def axis(*call_args: Any, **call_kwargs: Any) -> dict[tuple[str, ...], dict[str, str]]:
        loaded = load_axis(*call_args, **call_kwargs)
        return dict(_shuffled(list(loaded.items()), rng, changed, "sidecar-axis"))

    patches: tuple[tuple[object, str, Callable[..., object]], ...] = (
        (qualify, "load_events", events),
        (qualify, "load_placements", placements),
        (measure, "load_axis", axis),
    )
    with (
        mock.patch.object(patches[0][0], patches[0][1], patches[0][2]),
        mock.patch.object(patches[1][0], patches[1][1], patches[1][2]),
        mock.patch.object(patches[2][0], patches[2][1], patches[2][2]),
    ):
        qualify.run(
            args.inventory,
            args.sessions,
            args.corpus,
            args.out,
            measurements_dir=args.measurements,
        )
    if not changed:
        print("D06 positive control failed: no returned order changed", file=sys.stderr)
        return 3
    print("changed=" + ",".join(changed))
    return 0


def repeat_worker(argv: Sequence[str]) -> int:
    args = _arguments(argv)
    from pose_estimation import qualify

    for iteration in ("first", "second"):
        qualify.run(
            args.inventory,
            args.sessions,
            args.corpus,
            args.out,
            measurements_dir=args.measurements,
        )
        if iteration == "first":
            first = determinism.digests(pathlib.Path(args.out))
    second = determinism.digests(pathlib.Path(args.out))
    print(json.dumps({"first": first, "second": second}, sort_keys=True))
    return 0


def _worker_command(kind: str, out: pathlib.Path, inputs: Any, *, measured: bool) -> list[str]:
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        kind,
        "--inventory",
        str(inputs.inventory),
        "--sessions",
        str(inputs.sessions),
        "--corpus",
        str(inputs.corpus),
        "--out",
        str(out),
    ]
    if measured:
        command += ["--measurements", str(inputs.measurements)]
    return command


def _run_worker(kind: str, out: pathlib.Path, inputs: Any, *, measured: bool) -> str:
    completed = subprocess.run(
        _worker_command(kind, out, inputs, measured=measured),
        cwd=ROOT,
        env=determinism.base_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(
            f"{kind} worker failed rc={completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def shuffled_run(out: pathlib.Path, inputs: Any, *, measured: bool) -> tuple[dict[str, str], str]:
    control = _run_worker("--shuffle-worker", out, inputs, measured=measured)
    return determinism.digests(out), control


def repeat_run(
    out: pathlib.Path, inputs: Any, *, measured: bool
) -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(_run_worker("--repeat-worker", out, inputs, measured=measured))
    return payload["first"], payload["second"]


def _verdicts(observed: dict[str, str], baseline: dict[str, str]) -> dict[str, str]:
    return {
        name: "PASS" if observed[name] == baseline[name] else "FAIL"
        for name in determinism.ARTIFACTS
    }


def _render(identifier: str, verdicts: dict[str, str], suffix: str = "") -> None:
    print(
        f"{identifier} {' '.join(f'{name}={verdict}' for name, verdict in verdicts.items())}"
        f"{suffix}"
    )


def _row_order_only(
    baseline: pathlib.Path, observed: pathlib.Path, verdicts: dict[str, str]
) -> list[str]:
    names: list[str] = []
    for name, verdict in verdicts.items():
        if verdict != "FAIL" or not name.endswith(".csv"):
            continue
        with (baseline / name).open(encoding="utf-8", newline="") as stream:
            expected = list(csv.reader(stream))
        with (observed / name).open(encoding="utf-8", newline="") as stream:
            actual = list(csv.reader(stream))
        if expected[:1] == actual[:1] and sorted(expected[1:]) == sorted(actual[1:]):
            names.append(name)
    return names


def _registry_order(inventory_dir: pathlib.Path, out: pathlib.Path) -> bool:
    """D09: `assets_qc.csv` publishes the registry's own row order.

    The invariant D06 declines to sort away.  The order is inherited from a
    digest-validated input rather than chosen here, which is what makes it both
    deterministic and readable, and pinning it is what lets D06 leave
    `load_assets` unwrapped without dropping the property on the floor.
    """
    from pose_estimation import inventory

    with (inventory_dir / inventory.ASSETS_FILENAME).open(encoding="utf-8", newline="") as stream:
        registry = [row["asset_id"] for row in csv.DictReader(stream)]
    with (out / "assets_qc.csv").open(encoding="utf-8", newline="") as stream:
        published = [row["asset_id"] for row in csv.DictReader(stream)]
    return [asset for asset in registry if asset in set(published)] == published


def main() -> int:
    failures = {"D06": 0, "D07": 0, "D08": 0, "D09": 0}
    with tempfile.TemporaryDirectory() as raw:
        work = pathlib.Path(raw)
        inputs = determinism.build_fixture(work / "fixture")
        for mode in ("flagless", "measured"):
            measured = mode == "measured"
            baseline_out = work / f"baseline-{mode}"
            baseline = determinism.run_cli(baseline_out, inputs, measured=measured)

            shuffled_out = work / f"shuffled-{mode}"
            shuffled, control = shuffled_run(shuffled_out, inputs, measured=measured)
            d06 = _verdicts(shuffled, baseline)
            failures["D06"] += sum(verdict == "FAIL" for verdict in d06.values())
            order_only = _row_order_only(baseline_out, shuffled_out, d06)
            _render(
                f"D06:{mode}",
                d06,
                f" {control} row_order_only={','.join(order_only) or 'none'}",
            )

            renamed = determinism.run_cli(
                work / mode / "differently named output", inputs, measured=measured
            )
            d07 = _verdicts(renamed, baseline)
            failures["D07"] += sum(verdict == "FAIL" for verdict in d07.values())
            _render(f"D07:{mode}", d07)

            first, second = repeat_run(work / f"repeat-{mode}", inputs, measured=measured)
            d08 = {
                name: "PASS" if first[name] == second[name] == baseline[name] else "FAIL"
                for name in determinism.ARTIFACTS
            }
            failures["D08"] += sum(verdict == "FAIL" for verdict in d08.values())
            _render(f"D08:{mode}", d08)

            ordered = _registry_order(inputs.inventory, baseline_out)
            failures["D09"] += not ordered
            print(f"D09:{mode} assets_qc.csv={'PASS' if ordered else 'FAIL'} registry_row_order")

    total = sum(failures.values())
    print(
        f"D06_failures={failures['D06']} D07_failures={failures['D07']} "
        f"D08_failures={failures['D08']} D09_failures={failures['D09']} total_failures={total}"
    )
    return 1 if total else 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--shuffle-worker"]:
        raise SystemExit(shuffled_worker(arguments[1:]))
    if arguments[:1] == ["--repeat-worker"]:
        raise SystemExit(repeat_worker(arguments[1:]))
    raise SystemExit(main())
