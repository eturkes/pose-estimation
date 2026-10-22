"""`scripts/check_corpus_2d_integrity.py` passes a consistent synthetic run and names each tamper."""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys

from pose_estimation import corpus_run, inventory
from test_sessions import _canonical, _write_registry

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_corpus_2d_integrity.py"


def _checker():
    spec = importlib.util.spec_from_file_location("_corpus_integrity", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _landmarks(path: pathlib.Path, n: int = 40) -> None:
    columns = ["video", "frame_idx", "person_idx"]
    columns += [f"body_left_wrist_{axis}" for axis in ("x", "y", "z", "vis")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for i in range(n):
            x, y = 0.1 + 0.8 * i / n, 0.1 + 0.4 * i / n
            writer.writerow(["clip", i, 0, f"{x:.6f}", f"{y:.6f}", "0.0", "0.95"])


def _run_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, list[dict[str, str]]]:
    registry = _write_registry(tmp_path, [_canonical(1, "above"), _canonical(1, "left")]).root
    with (registry / inventory.ASSETS_FILENAME).open(newline="", encoding="utf-8") as stream:
        ids = [row["asset_id"] for row in csv.DictReader(stream)]
    tree = tmp_path / "run"
    (tree / "e1").mkdir(parents=True)
    manifest = [
        {"asset_id": asset, "camera_name": f"cam-{i}", "disposition": "ok", "event_id": "e1"}
        for i, asset in enumerate(ids)
    ]
    for row in manifest:
        _landmarks(tree / "e1" / f"{row['camera_name']}.csv")
    _write_manifest(tree, manifest)
    census = dict.fromkeys(corpus_run.ASSET_DISPOSITIONS, 0) | {"ok": len(ids)}
    verdicts = dict.fromkeys(sorted(_checker().driver_verdict_keys()), True)
    report = {"manifest": {"census": census}, "verdicts": verdicts}
    (tree / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
    return tree, registry, manifest


def _write_manifest(tree: pathlib.Path, rows: list[dict[str, str]]) -> None:
    with (tree / corpus_run.MANIFEST_FILENAME).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _main(tree: pathlib.Path, registry: pathlib.Path, capsys) -> tuple[int, str]:
    rc = _checker().main(["--tree", str(tree), "--inventory", str(registry)])
    return rc, capsys.readouterr().out


def test_a_consistent_run_passes_every_predicate(tmp_path: pathlib.Path, capsys) -> None:
    tree, registry, _ = _run_tree(tmp_path)
    rc, out = _main(tree, registry, capsys)
    assert rc == 0, out
    assert "FAIL" not in out
    assert "PASS isotropic_bound_landscape: 0/80" in out


def test_a_duplicated_and_dropped_asset_fails_totality(tmp_path: pathlib.Path, capsys) -> None:
    tree, registry, manifest = _run_tree(tmp_path)
    _write_manifest(tree, [manifest[0], {**manifest[0], "camera_name": manifest[1]["camera_name"]}])
    rc, out = _main(tree, registry, capsys)
    assert rc == 1
    assert "FAIL manifest_ids_unique" in out
    assert "FAIL manifest_ids_set_equal_registry" in out


def test_a_renamed_verdict_key_fails(tmp_path: pathlib.Path, capsys) -> None:
    tree, registry, _ = _run_tree(tmp_path)
    report = json.loads((tree / "run_report.json").read_text(encoding="utf-8"))
    first = next(iter(report["verdicts"]))
    report["verdicts"][f"{first}_renamed"] = report["verdicts"].pop(first)
    (tree / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
    rc, out = _main(tree, registry, capsys)
    assert rc == 1
    assert "FAIL verdict_keys_set_equal_driver" in out
    assert "PASS verdicts_all_true" in out


def test_a_per_axis_rescaled_asset_breaches_the_isotropic_bound(
    tmp_path: pathlib.Path, capsys
) -> None:
    tree, registry, manifest = _run_tree(tmp_path)
    path = tree / "e1" / f"{manifest[0]['camera_name']}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Landscape 1920x1080: a per-axis y/h normalisation = the isotropic y times max(w, h)/h.
    for row in rows:
        row["body_left_wrist_y"] = f"{float(row['body_left_wrist_y']) * 1920 / 1080:.6f}"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    rc, out = _main(tree, registry, capsys)
    assert rc == 1
    assert "FAIL isotropic_bound_landscape" in out
