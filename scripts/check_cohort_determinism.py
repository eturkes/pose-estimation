"""Determinism + consumer-boundary campaign for the cohort aggregate publisher.

The published tree is gitignored (redaction-safe but uncommitted), so no golden byte
oracle exists for it. This campaign stands in for one: it re-publishes the same synthetic
corpus under moved cwd, moved output path, moved pid and reordered inputs, and requires
byte-identical artifacts every time; then it tampers with each published file and requires
`validate_generation` to refuse. Failures keep their scratch tree for inspection.

Run: `env -u LD_LIBRARY_PATH PYTHONPATH=src uv run --no-sync python \
scripts/check_cohort_determinism.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
SCRATCH = ROOT / ".cohort-determinism"
RESULTS = TESTS / "cohort_determinism_results.json"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TESTS))

from pose_estimation import cohort  # noqa: E402
from test_cohort import _build_inputs, _published_bytes  # noqa: E402

# Every source that shapes a published byte or the acceptance itself, so an edit anywhere
# under the campaign invalidates a recorded PASS rather than letting it certify bytes it
# never produced. The publisher reads five sibling modules and the fixture builds its
# corpus from three committed goldens, so a digest over `cohort.py` alone would certify a
# tree it did not see; the campaign script is in the tuple because it defines the sweeps.
SOURCES = (
    pathlib.Path(__file__).resolve(),
    ROOT / "src" / "pose_estimation" / "cohort.py",
    ROOT / "src" / "pose_estimation" / "corpus_run.py",
    ROOT / "src" / "pose_estimation" / "inventory.py",
    ROOT / "src" / "pose_estimation" / "qualify.py",
    ROOT / "src" / "pose_estimation" / "sessions.py",
    ROOT / "src" / "pose_estimation" / "view_agreement.py",
    TESTS / "test_cohort.py",
    TESTS / "test_sessions.py",
    TESTS / "goldens" / "r_clinical" / "2d_idx_clinical.csv",
    TESTS / "goldens" / "r_clinical" / "2d_idx_clinical_windows.csv",
    TESTS / "goldens" / "r_clinical" / "2d_idx_clinical_bouts.csv",
)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in SOURCES:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digests(out: pathlib.Path) -> dict[str, str]:
    return {
        name: hashlib.sha256(payload).hexdigest() for name, payload in _published_bytes(out).items()
    }


def _publish(inputs, out: pathlib.Path) -> dict[str, str]:
    cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    cohort.validate_generation(out)
    return _digests(out)


def _corpus(root: pathlib.Path, name: str):
    directory = root / name
    directory.mkdir(parents=True)
    return _build_inputs(directory, cohort)


def _sweeps(root: pathlib.Path) -> list[tuple[str, dict[str, str]]]:
    """Every sweep publishes the same corpus through a different incidental condition."""
    inputs = _corpus(root, "corpus")
    results = [("baseline", _publish(inputs, root / "out"))]
    results.append(("republish-over-self", _publish(inputs, root / "out")))
    results.append(("different-output-name", _publish(inputs, root / "elsewhere" / "cohort")))
    origin = pathlib.Path.cwd()
    try:
        os.chdir(root)
        results.append(("moved-cwd", _publish(inputs, root / "out-cwd")))
    finally:
        os.chdir(origin)
    # The staging and retiring siblings carry the pid, so a published byte that depended on
    # one would drift between processes rather than between runs.
    real_getpid = os.getpid
    os.getpid = lambda: real_getpid() + 1  # type: ignore[assignment]
    try:
        results.append(("moved-pid", _publish(inputs, root / "out-pid")))
    finally:
        os.getpid = real_getpid  # type: ignore[assignment]
    fresh = _corpus(root, "corpus-rebuilt")
    results.append(("rebuilt-corpus", _publish(fresh, root / "out-rebuilt")))
    return results


def _tampers(root: pathlib.Path) -> list[tuple[str, bool]]:
    """Every consumer-boundary class the marker claims to cover."""
    inputs = _corpus(root, "tamper-corpus")
    outcomes: list[tuple[str, bool]] = []

    def refuses(name: str, mutate) -> None:
        out = root / "tamper" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out, ignore_errors=True)
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
        mutate(out)
        try:
            cohort.validate_generation(out)
        except cohort.CohortError:
            outcomes.append((name, True))
        else:
            outcomes.append((name, False))

    def edit_marker(out: pathlib.Path, mutate) -> None:
        marker = json.loads((out / cohort.COHORT_FILENAME).read_text(encoding="utf-8"))
        mutate(marker)
        (out / cohort.COHORT_FILENAME).write_bytes(cohort.render_marker(marker))

    for name in cohort.PUBLISHED_FILENAMES:
        refuses(f"append-{name}", lambda out, name=name: (out / name).open("a").write("x\n"))
        refuses(f"remove-{name}", lambda out, name=name: (out / name).unlink())
    refuses("remove-marker", lambda out: (out / cohort.COHORT_FILENAME).unlink())
    refuses(
        "symlink-marker",
        lambda out: (
            (out / cohort.COHORT_FILENAME).unlink(),
            (out / cohort.COHORT_FILENAME).symlink_to(out / cohort.CELLS_FILENAME),
        ),
    )
    refuses(
        "foreign-generator",
        lambda out: edit_marker(
            out, lambda marker: marker["generation"].update(generator="someone-else")
        ),
    )
    refuses(
        "foreign-version",
        lambda out: edit_marker(
            out, lambda marker: marker["generation"].update(generator_version="v0")
        ),
    )
    refuses(
        "edited-census",
        lambda out: edit_marker(out, lambda marker: marker["population"].update(cells=99)),
    )
    refuses(
        "edited-estimand",
        lambda out: edit_marker(out, lambda marker: marker.update(estimand="something else")),
    )
    refuses(
        "dropped-marker-key",
        lambda out: edit_marker(out, lambda marker: marker.pop("descriptor_collision")),
    )
    refuses(
        "extra-marker-key",
        lambda out: edit_marker(out, lambda marker: marker.update(extra=1)),
    )
    refuses(
        "duplicate-marker-key",
        lambda out: (out / cohort.COHORT_FILENAME).write_text(
            (out / cohort.COHORT_FILENAME)
            .read_text(encoding="utf-8")
            .replace('"estimand"', '"estimand": "duplicate",\n  "estimand"', 1),
            encoding="utf-8",
        ),
    )
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="Keep the scratch tree after a passing run."
    )
    arguments = parser.parse_args(argv)
    digest = _source_digest()
    if RESULTS.is_file():
        recorded = json.loads(RESULTS.read_text(encoding="utf-8"))
        if recorded.get("source_digest") != digest:
            print(f"cohort determinism: source bytes moved; remove {RESULTS} to regenerate")
            return 1
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    started = time.monotonic()
    sweeps = _sweeps(SCRATCH)
    baseline = sweeps[0][1]
    drifted = [name for name, digests in sweeps if digests != baseline]
    tampers = _tampers(SCRATCH)
    accepted = [name for name, refused in tampers if not refused]
    elapsed = time.monotonic() - started
    result = {
        "source_digest": digest,
        "n_sweeps": len(sweeps),
        "n_tamper_classes": len(tampers),
        "drifted": drifted,
        "accepted_tampers": accepted,
        "elapsed_s": round(elapsed, 1),
        "verdict": "PASS" if not drifted and not accepted else "FAIL",
        "published_digests": baseline,
    }
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"cohort determinism: {result['verdict']} — {len(sweeps)} sweeps, "
        f"{len(tampers)} tamper classes in {result['elapsed_s']}s"
    )
    for name in drifted:
        print(f"  drifted: {name}")
    for name in accepted:
        print(f"  accepted tamper: {name}")
    if result["verdict"] == "PASS" and not arguments.keep:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
