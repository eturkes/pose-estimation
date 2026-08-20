#!/usr/bin/env python3
"""Prove the three inventory artifacts are a function of the corpus bytes alone.

Each sweep re-runs the committed CLI under one hostile-but-legal variation and
compares the three published files with the baseline by SHA-256.  A sweep fails
when any artifact digest moves, because nothing outside the corpus may reach the
published bytes: not the hash seed, not the filesystem locale, not the timezone,
not the directory iteration order, not the path spelling of the corpus itself.

The last sweep is the consumer boundary: ``validate_generation`` must accept a
freshly published set and must raise ``InventoryError`` for each tamper class.

Results stream to ``tests/inventory_determinism_results.json`` after every sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ("assets.csv", "captures.csv", "census.json")


def digests(out: pathlib.Path) -> dict[str, str]:
    return {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ARTIFACTS}


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "PYTHONHASHSEED", "PYTHONUNBUFFERED"):
        env.pop(key, None)
    return env


def run_cli(
    out: pathlib.Path,
    *,
    corpus: str = "videos/3-cam",
    env_overrides: dict[str, str | None] | None = None,
    cwd: pathlib.Path = ROOT,
    optimize: bool = False,
    umask: int | None = None,
) -> dict[str, str]:
    """Run the committed CLI once and return the three artifact digests."""
    env = base_env()
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    command = [sys.executable]
    if optimize:
        command.append("-O")
    command += ["-m", "pose_estimation.inventory", "--corpus", corpus, "--out", str(out)]

    preexec = (lambda: os.umask(umask)) if umask is not None else None
    completed = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, preexec_fn=preexec
    )
    if completed.returncode:
        raise SystemExit(
            f"inventory run failed rc={completed.returncode}: {completed.stderr[-400:]}"
        )
    return digests(out)


def shuffled_walk_digests(out: pathlib.Path, seed: int) -> dict[str, str]:
    """Run the pipeline in-process with a randomised directory iteration order.

    ``_iter_entries`` sorts each directory's children, so a shuffled ``iterdir``
    must not reach the output.  Patching the seam proves that rather than
    inferring it from the call site.
    """
    from pose_estimation import inventory

    rng = random.Random(seed)
    original = pathlib.Path.iterdir

    def scrambled(self):
        entries = list(original(self))
        rng.shuffle(entries)
        return iter(entries)

    pathlib.Path.iterdir = scrambled
    try:
        inventory.run(ROOT / "videos/3-cam", out, checksums=True)
    finally:
        pathlib.Path.iterdir = original
    return digests(out)


def _edited_census(raw: bytes, edit) -> bytes:
    census = json.loads(raw)
    edit(census)
    return json.dumps(census, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _bump_discovered(census) -> None:
    census["assets"]["discovered"] += 1


def _add_key(census) -> None:
    census["an_unexpected_key"] = 1


def _drop_key(census) -> None:
    census["assets"].pop("discovered")


# label, artifact, transform, expectation.  The two table digests cover exact
# published bytes, so a line-ending rewrite alone must fail.  The census digest
# covers content rather than bytes, because a document cannot carry a digest of
# itself, so insignificant whitespace is accepted by design and a moved value,
# an added key and a removed key are not.  ``None`` deletes the artifact.
TAMPERS = (
    ("assets.csv line endings", "assets.csv", lambda raw: raw.replace(b"\n", b"\r\n", 1), "reject"),
    (
        "assets.csv value",
        "assets.csv",
        lambda raw: raw.replace(b"canonical", b"quarantined", 1),
        "reject",
    ),
    ("assets.csv absent", "assets.csv", None, "reject"),
    (
        "captures.csv line endings",
        "captures.csv",
        lambda raw: raw.replace(b"\n", b"\r\n", 1),
        "reject",
    ),
    ("captures.csv absent", "captures.csv", None, "reject"),
    (
        "census.json value",
        "census.json",
        lambda raw: _edited_census(raw, _bump_discovered),
        "reject",
    ),
    ("census.json added key", "census.json", lambda raw: _edited_census(raw, _add_key), "reject"),
    (
        "census.json removed key",
        "census.json",
        lambda raw: _edited_census(raw, _drop_key),
        "reject",
    ),
    ("census.json truncated", "census.json", lambda raw: raw[: len(raw) // 2], "reject"),
    ("census.json not JSON", "census.json", lambda raw: b"not json at all\n", "reject"),
    ("census.json not an object", "census.json", lambda raw: b"[1, 2, 3]\n", "reject"),
    ("census.json invalid UTF-8", "census.json", lambda raw: b"\xff\xfe" + raw, "reject"),
    ("census.json absent", "census.json", None, "reject"),
    ("census.json trailing whitespace", "census.json", lambda raw: raw + b"  \n", "accept"),
)


def tamper_verdicts(out: pathlib.Path, work: pathlib.Path) -> dict[str, str]:
    """Accept a clean set; answer every tamper class through ``InventoryError``."""
    from pose_estimation import inventory

    def verdict(directory: pathlib.Path, expectation: str) -> str:
        try:
            inventory.validate_generation(directory)
        except inventory.InventoryError:
            return "rejected" if expectation == "reject" else "REJECTED A VALID SET"
        except Exception as exc:
            # Any class other than InventoryError is itself the defect.
            return f"WRONG CLASS: {type(exc).__name__}"
        return "accepted" if expectation == "accept" else "ACCEPTED A TAMPERED SET"

    verdicts = {"clean": verdict(out, "accept")}
    for index, (label, name, transform, expectation) in enumerate(TAMPERS):
        copy = work / f"tamper{index:02d}"
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(out, copy)
        target = copy / name
        if transform is None:
            target.unlink()
        else:
            target.write_bytes(transform(target.read_bytes()))
        verdicts[label] = verdict(copy, expectation)
    return verdicts


def sweeps(work: pathlib.Path):
    """Yield ``(id, variation, digests)`` for every environment-level sweep."""
    out = work / "sweep"

    yield "B01", "repeat, identical invocation", run_cli(out)

    for seed in ("0", "1", "12345", "random"):
        yield (
            f"B02:{seed}",
            f"PYTHONHASHSEED={seed}",
            run_cli(out, env_overrides={"PYTHONHASHSEED": seed}),
        )

    for locale in ("C", "C.UTF-8", "en_US.UTF-8"):
        yield (
            f"B03:{locale}",
            f"LC_ALL={locale}",
            run_cli(out, env_overrides={"LC_ALL": locale, "LANG": locale}),
        )
    yield (
        "B03:unset",
        "LANG and LC_ALL unset",
        run_cli(out, env_overrides={"LC_ALL": None, "LANG": None}),
    )

    absolute = str(ROOT / "videos/3-cam")
    yield "B05:absolute", "absolute corpus path", run_cli(out, corpus=absolute)
    yield (
        "B05:dotdot",
        "corpus path with a .. detour",
        run_cli(out, corpus=str(ROOT / "videos" / ".." / "videos" / "3-cam")),
    )
    yield (
        "B05:cwd",
        "run from a different working directory",
        run_cli(out, corpus=absolute, cwd=work),
    )

    renamed = work / "differently named output"
    yield "B06", "--out directory name", run_cli(renamed)

    for tz in ("UTC", "Pacific/Kiritimati"):
        yield f"B07:{tz}", f"TZ={tz}", run_cli(out, env_overrides={"TZ": tz})

    yield "B09", "umask 077", run_cli(out, umask=0o077)
    yield "B10", "interpreter -O", run_cli(out, optimize=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "tests" / "inventory_determinism_results.json",
    )
    args = parser.parse_args(argv)

    rows: list[dict[str, object]] = []
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as raw:
        work = pathlib.Path(raw)
        base_out = work / "baseline"
        baseline = run_cli(base_out)
        payload = {
            "schema_version": 1,
            "tested_head": subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "baseline_sha256": baseline,
            "sweeps": rows,
        }
        print(f"B00 baseline {json.dumps(baseline, sort_keys=True)}", flush=True)

        def record(identifier: str, variation: str, observed: dict[str, str]) -> None:
            verdict = {
                name: ("PASS" if observed[name] == baseline[name] else "FAIL") for name in ARTIFACTS
            }
            rows.append(
                {
                    "id": identifier,
                    "variation": variation,
                    **{f"verdict_{name}": verdict[name] for name in ARTIFACTS},
                    "observed_sha256": observed,
                }
            )
            payload["sweeps"] = rows
            output.write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            print(f"{identifier} {' '.join(verdict.values())} {variation}", flush=True)

        for identifier, variation, observed in sweeps(work):
            record(identifier, variation, observed)

        record(
            "B04",
            "shuffled directory iteration order",
            shuffled_walk_digests(work / "shuffled", 8117),
        )
        record("B08", "second run after every other sweep", run_cli(work / "late"))

        verdicts = tamper_verdicts(base_out, work)
        rows.append({"id": "B11", "variation": "validate_generation tamper classes", **verdicts})
        payload["sweeps"] = rows
        output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"B11 {json.dumps(verdicts, sort_keys=True)}", flush=True)

    failures = [
        row
        for row in rows
        if any(
            str(value).startswith(("FAIL", "ACCEPTED", "WRONG", "REJECTED"))
            for value in row.values()
        )
    ]
    print(f"\nsweeps={len(rows)} failures={len(failures)}")
    for row in failures:
        print(f"  {row['id']} {row['variation']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
