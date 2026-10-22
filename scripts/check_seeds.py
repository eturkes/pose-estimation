#!/usr/bin/env python3
"""Replay each checker's negative input in a temporary tree; an inert seed exits 1."""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import pathlib
import re
import shutil
import sys
import tempfile
import types
from collections.abc import Callable
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SeedInertError(Exception):
    """The seed did not reach its intended refusal."""


def _load(stem: str) -> types.ModuleType:
    path = ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_seed_{stem}", path)
    if spec is None or spec.loader is None:
        raise SeedInertError(f"cannot load {stem}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture(call: Callable[[], int]) -> tuple[int, str]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            rc = call()
        except SystemExit as error:
            rc = error.code if isinstance(error.code, int) else 1
            if isinstance(error.code, str):
                print(error.code)
    return rc, stream.getvalue()


def _require_refusal(observed: tuple[int, str], rc: int, needle: str) -> str:
    actual, output = observed
    line = next((line.strip() for line in output.splitlines() if needle in line), "")
    if actual != rc or not line:
        raise SeedInertError(f"expected rc={rc} + {needle!r}; got rc={actual}: {output.strip()}")
    return f"rc={actual} {line}"


def _require_green(observed: tuple[int, str]) -> str:
    rc, output = observed
    if rc:
        raise SeedInertError(f"positive control rc={rc}: {output.strip()}")
    return output


def _literal(module: types.ModuleType, prefix: str) -> str:
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    matches = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(prefix)
    }
    if len(matches) != 1:
        raise SeedInertError(
            f"expected one source refusal starting {prefix!r}; found {len(matches)}"
        )
    return matches.pop()


def _predicate(module: types.ModuleType, function: str) -> str:
    return next(pid for pid, fn in dict(module.PREDICATES).items() if fn.__name__ == function)


def _json(path: pathlib.Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stop_campaign(*args: object, **kwargs: object) -> None:
    raise SeedInertError("stale-source barrier admitted the seed; campaign not run")


def _stale(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    stem = pathlib.Path(module.__file__).stem
    result = work / "result.json"
    if stem == "check_cohort_determinism":
        _json(result, {"source_digest": module._source_digest() + "-stale"})
        with (
            mock.patch.object(module, "SCRATCH", work / "campaign"),
            mock.patch.object(module, "_sweeps", _stop_campaign),
        ):
            observed = _capture(lambda: module.main(["--output", str(result)]))
        return _require_refusal(observed, 1, _literal(module, "cohort determinism: source bytes"))
    sources = module.source_digests()
    sources[next(iter(sources))] += "-stale"
    calibration = stem == "check_calibration_qc_determinism"
    key = "source_digests" if calibration else "source_sha256"
    _json(result, {key: sources})
    flag = "--result" if calibration else "--output"
    boundary = "_prepare" if calibration else "build_fixture"
    if stem == "check_inventory_determinism":
        boundary = "run_cli"
    with mock.patch.object(module, boundary, _stop_campaign):
        observed = _capture(lambda: module.main([flag, str(result)]))
    return _require_refusal(observed, 2, _literal(module, "REFUSED:"))


def _fixtures(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    _require_green(_capture(lambda: module.main([])))
    copied = work / "fixtures"
    shutil.copytree(module.FIXTURES, copied)
    manifest = copied / module.MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["input_digests"]
    _json(manifest, payload)
    pid = _predicate(module, "p09_manifest_agrees_with_the_tree")
    return _require_refusal(
        _capture(lambda: module.main(["--fixtures", str(copied)])), 1, f"FAIL {pid}"
    )


def _cameras(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    cameras = [
        {
            "event_id": "synthetic-event",
            "asset_id": "synthetic-asset",
            "camera_name": "view",
            "reference_camera": "view",
            "is_reference": "1",
            "offset_status": "reference",
            "offset_s": "0.000000000",
        }
    ]
    _csv(work / "cameras_qc.csv", cameras)
    _csv(
        work / "events_qc.csv",
        [
            {
                "event_id": "synthetic-event",
                "graph_connected": "1",
                "sync_status": "connected",
                "offset_span_s": "",
            }
        ],
    )
    _json(
        work / "qualification.json",
        {
            "generation": {"generator_version": "v4", "cameras_qc.csv": "synthetic"},
            "cameras": {},
        },
    )
    expected = {
        "rows": 1,
        "events": 1,
        "offsets": 1,
        "unreachable": 0,
        "connected": 1,
        "in_connected": 1,
        "recovered": 0,
    }
    args = ["--qualification", str(work)]
    for name, value in expected.items():
        args += [f"--expect-{name.replace('_', '-')}", str(value)]
    baseline = _require_green(_capture(lambda: module.main(args)))
    target = next(
        line.split(":", 1)[0] for line in baseline.splitlines() if "reference offset" in line
    )
    cameras[0]["offset_s"] = "1.000000000"
    _csv(work / "cameras_qc.csv", cameras)
    return _require_refusal(
        _capture(lambda: module.main(args)), 1, target.replace("PASS", "FAIL", 1)
    )


def _document_repo(work: pathlib.Path) -> pathlib.Path:
    repo = work / "repo"
    repo.mkdir()
    shutil.copytree(ROOT / "docs", repo / "docs")
    for name in ("src", "scripts", "tests", "analysis", "renv"):
        (repo / name).symlink_to(ROOT / name, target_is_directory=True)
    return repo


def _claims(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    repo = _document_repo(work)
    args = ["--repo", str(repo)]
    _require_green(_capture(lambda: module.main(args)))
    with module.REPORT.open("a", encoding="utf-8") as stream:
        stream.write("\n" + module.calibration_qc.PROHIBITED_PARAPHRASES[0] + "\n")
    pid = _predicate(module, "_p03_no_prohibited_paraphrase")
    return _require_refusal(_capture(lambda: module.main(args)), 1, f"FAIL {pid}")


def _isotropy(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    report = work / "run_report.json"
    _json(report, {"configuration": {"coord_normalization": module.export.COORD_NORMALIZATION}})
    module._require_isotropic_tree(work)
    _json(report, {"configuration": {"coord_normalization": "seed-invalid"}})
    needle = _literal(module, "; max(w, h) recovers")
    return _require_refusal(
        _capture(
            lambda: module.main(
                [
                    "--tree",
                    str(work),
                    "--inventory",
                    str(work / "unused"),
                    "--out",
                    str(work / "unused.json"),
                ]
            )
        ),
        1,
        needle,
    )


def _suite(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    path = work / "suite.py"
    text = "\n".join(
        f"# kind: {case} = {module.KINDS[0]}\ndef {module.fn_name(case)}():\n    pass\n"
        for case in module.CASES
    )
    path.write_text(text, encoding="utf-8")
    args = ["--suite", str(path)]
    _require_green(_capture(lambda: module.main(args)))
    path.write_text(text.replace("    pass", f"    {module.UNFILLED}", 1), encoding="utf-8")
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    label = next(
        node.elts[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and isinstance(node.elts[0], ast.Constant)
        and isinstance(node.elts[1], ast.Name)
        and node.elts[1].id == "unfilled"
    )
    return _require_refusal(_capture(lambda: module.main(args)), 1, f"FAIL {label}")


def _row_order(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    # Supply publisher outputs, not verdicts. Every comparison + main's rc stays shipped.
    registry = work / "registry"
    registry.mkdir()
    rows = [{"asset_id": "synthetic-a"}, {"asset_id": "synthetic-b"}]
    _csv(registry / "assets.csv", rows)
    inputs = types.SimpleNamespace(inventory=registry)
    reversed_rows = False

    def publish(out: pathlib.Path, *_args: object, **_kwargs: object) -> dict[str, str]:
        out.mkdir(parents=True, exist_ok=True)
        for name in module.determinism.ARTIFACTS:
            (out / name).write_text("synthetic\n", encoding="utf-8")
        _csv(out / "assets_qc.csv", list(reversed(rows)) if reversed_rows else rows)
        return module.determinism.digests(out)

    def shuffled(out: pathlib.Path, *args: object, **kwargs: object) -> tuple[dict[str, str], str]:
        return publish(out, *args, **kwargs), "synthetic publisher boundary"

    def repeated(
        out: pathlib.Path, *args: object, **kwargs: object
    ) -> tuple[dict[str, str], dict[str, str]]:
        return publish(out, *args, **kwargs), publish(out, *args, **kwargs)

    with (
        mock.patch.object(module.determinism, "build_fixture", return_value=inputs),
        mock.patch.object(module.determinism, "run_cli", publish),
        mock.patch.object(module, "shuffled_run", shuffled),
        mock.patch.object(module, "repeat_run", repeated),
    ):
        baseline = _require_green(_capture(module.main))
        target = next(line for line in baseline.splitlines() if "registry_row_order" in line)
        reversed_rows = True
        return _require_refusal(_capture(module.main), 1, target.replace("PASS", "FAIL", 1))


def _prospective_mutations(module: types.ModuleType, text: str) -> list[tuple[str, str, str]]:
    sections = list(module.SECTION_HEADING.finditer(text))
    bound = {sid for ids in module.NON_NEGOTIABLES.values() for sid in ids}
    bodies = module._sections()
    sid = next(sid for sid in sorted(bound, reverse=True) if bodies[sid].count("MUST") == 1)
    heading = next(match for match in sections if match.group(1) == sid)
    at = text.index("MUST", heading.end())
    citation = re.search(r"^-\s*\[E\d+\][^\n]*https?://\S+[^\n]*$", text, re.MULTILINE)
    path = next(
        token for token in re.findall(r"`([^`\s]+)`", text) if token.startswith(module.PATH_ROOTS)
    )
    if citation is None:
        raise SeedInertError("no citation with a URL")
    edits = (
        ("_p01_sections_present_in_order", text.replace(sections[6].group(), "", 1)),
        (
            "_p02_every_section_carries_five_fields",
            re.sub(
                rf"\*\*{re.escape(module.REQUIRED_FIELDS[3])}:\*\*[^\n]*",
                f"**{module.REQUIRED_FIELDS[3]}:**",
                text,
                count=1,
            ),
        ),
        ("_p03_non_negotiables_bound", text[:at] + "must" + text[at + len("MUST") :]),
        (
            "_p04_no_prohibited_paraphrase",
            text + "\n" + module.calibration_qc.PROHIBITED_PARAPHRASES[0],
        ),
        (
            "_p05_absences_labelled_local",
            re.sub(
                rf"{re.escape(module.LOCAL_DECISION)}\s+under {module.ABSENCES[2]}\b",
                f"local choice under {module.ABSENCES[2]}",
                text,
                count=1,
            ),
        ),
        (
            "_p06_citations_resolvable",
            text[: citation.start()]
            + re.sub(
                r"https?://\S+",
                "URL removed",
                citation.group(),
            )
            + text[citation.end() :],
        ),
        ("_p07_named_paths_resolve", text.replace(f"`{path}`", "`docs/seed-absent.md`", 1)),
        ("_p11_no_execution_claim", text + "\n" + module.EXECUTION_CLAIMS[0]),
    )
    return [
        (f"NC{i}", _predicate(module, function), changed)
        for i, (function, changed) in enumerate(edits, 1)
    ]


def _prospective(module: types.ModuleType, work: pathlib.Path, inert: str | None) -> str:
    source = ROOT / "docs" / "prospective_capture.md"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    repo = _document_repo(work)
    args = ["--repo", str(repo)]
    _require_green(_capture(lambda: module.main(args)))
    original = module.SPEC.read_text(encoding="utf-8")
    controls = _prospective_mutations(module, original)
    refused = []
    for name, pid, mutated in controls:
        if inert == name:
            mutated = original
        module.SPEC.write_text(mutated, encoding="utf-8")
        checker = _load("check_prospective_capture")
        observed = _capture(lambda checker=checker: checker.main(args))
        try:
            if mutated == original:
                raise SeedInertError("injection changed no byte")
            refused.append(_require_refusal(observed, 1, f"FAIL {pid}"))
            print(f"check_prospective_capture {name} REFUSED {refused[-1]}")
        except SeedInertError as error:
            print(f"check_prospective_capture {name} SEED-INERT {error}")
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"controls_firing {len(refused)} of {len(controls)}")
    print(f"prospective_capture sha256 before={before} after={after}")
    if len(refused) != len(controls) or before != after:
        raise SeedInertError("prospective controls did not all fire or original bytes moved")
    return "; ".join(refused)


def _review(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    report = work / "report.md"
    text = "## Verdict table\n\n| A01 | seed checks the report verdict | pass |\n\n## Detail\n\n## Register\n"
    report.write_text(text, encoding="utf-8")
    _require_green(_capture(lambda: module.main(report)))
    report.write_text(text.replace("| pass |", "| unknown |"), encoding="utf-8")
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    pid = next(
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check"
        and isinstance(node.args[0], ast.UnaryOp)
        and isinstance(node.args[0].operand, ast.Name)
        and node.args[0].operand.id == "unknown"
    )
    return _require_refusal(_capture(lambda: module.main(report)), 1, f"FAIL {pid}")


def _corpus_integrity(module: types.ModuleType, work: pathlib.Path, _: str | None) -> str:
    # The suite's synthetic registry + run tree builder (PyAV-free test_sessions registry).
    sys.path.insert(0, str(ROOT / "tests"))
    from test_corpus_2d_integrity import _run_tree

    tree, registry, _rows = _run_tree(work)
    argv = ["--tree", str(tree), "--inventory", str(registry)]
    _require_green(_capture(lambda: module.main(argv)))
    report = tree / "run_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    first = sorted(payload["verdicts"])[0]
    payload["verdicts"][f"{first}_seed"] = payload["verdicts"].pop(first)
    _json(report, payload)
    return _require_refusal(
        _capture(lambda: module.main(argv)), 1, "FAIL verdict_keys_set_equal_driver"
    )


SEEDS = {
    "check_calibration_qc_determinism": _stale,
    "check_calibration_qc_fixtures": _fixtures,
    "check_cameras_qc_census": _cameras,
    "check_claim_report": _claims,
    "check_cohort_determinism": _stale,
    "check_corpus_2d_integrity": _corpus_integrity,
    "check_inventory_determinism": _stale,
    "check_isotropy_angle_fidelity": _isotropy,
    "check_m2u4_suite_seed": _suite,
    "check_m2u5_determinism": _row_order,
    "check_prospective_capture": _prospective,
    "check_qualify_determinism": _stale,
    "check_review_report": _review,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(SEEDS), help="Run one checker's seeds.")
    parser.add_argument(
        "--inert",
        choices=tuple(f"check_prospective_capture:NC{i}" for i in range(1, 9)),
        help="Leave one prospective control unchanged.",
    )
    args = parser.parse_args(argv)
    present = {
        path.stem for path in (ROOT / "scripts").glob("check_*.py") if path.stem != "check_seeds"
    }
    if present != set(SEEDS):
        print(f"SEED-INERT checker census mismatch: {sorted(present ^ set(SEEDS))}")
        return 1
    selected = [args.only] if args.only else list(SEEDS)
    refused = 0
    with tempfile.TemporaryDirectory(prefix="check-seeds-") as raw:
        for stem in selected:
            work = pathlib.Path(raw) / stem
            work.mkdir()
            try:
                inert = (
                    args.inert.split(":", 1)[1]
                    if args.inert and args.inert.startswith(stem + ":")
                    else None
                )
                detail = SEEDS[stem](_load(stem), work, inert)
            except Exception as error:
                print(f"{stem} SEED-INERT {type(error).__name__}: {error}")
            else:
                refused += 1
                print(f"{stem} REFUSED {detail}")
    print(f"{refused}/{len(selected)} refused")
    return 0 if refused == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
