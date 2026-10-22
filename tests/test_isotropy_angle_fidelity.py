"""P16 evidence binds to the script, registry, run tree and configs that produced it."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

from pose_estimation import inventory

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_isotropy_angle_fidelity.py"
_EVIDENCE = _ROOT / "tests" / "isotropy_angle_fidelity_results.json"
_INVENTORY = _ROOT / "inventory"


def _load():
    spec = importlib.util.spec_from_file_location("_p16_fidelity", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _recorded() -> dict:
    return json.loads(_EVIDENCE.read_text(encoding="utf-8"))


def test_evidence_binds_the_shipped_script() -> None:
    recorded = _recorded()["inputs"]["script_sha256"]
    assert recorded == hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()


@pytest.mark.skipif(not _INVENTORY.is_dir(), reason="needs the published inventory")
def test_evidence_binds_the_validated_registry() -> None:
    fidelity = _load()
    recorded = _recorded()
    generation = inventory.validate_generation(_INVENTORY)["generation"]
    assert recorded["inputs"]["registry_generation"] == dict(generation)
    sample = fidelity.frozen_sample(fidelity._canonical_assets(_INVENTORY))
    assert recorded["sample_digest_sha256"] == fidelity.sample_digest(sample)


def test_evidence_carries_every_binding_and_passing_verdicts() -> None:
    recorded = _recorded()
    assert set(recorded["inputs"]) == {
        "script_sha256",
        "registry_generation",
        "run_manifest_sha256",
        "run_report_sha256",
        "sample_landmarks_sha256",
        "sample_pose_configs_sha256",
    }
    assert recorded["sample_size"] == len(recorded["per_asset"]) == 8
    assert all(recorded["verdicts"].values())


def test_stale_inputs_names_each_moved_binding() -> None:
    fidelity = _load()
    current = {"a": "1", "b": {"x": "2"}, "c": "3"}
    assert fidelity.stale_inputs(current, copy.deepcopy(current)) == []
    moved = {"a": "1", "b": {"x": "9"}, "d": "4"}
    assert fidelity.stale_inputs(moved, current) == ["b", "c", "d"]


def test_rerun_refuses_moved_inputs_and_fails_a_changed_result(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fidelity = _load()
    payload = _recorded()
    monkeypatch.setattr(fidelity, "measure", lambda tree, inventory_dir: copy.deepcopy(payload))
    out = tmp_path / "result.json"
    argv = ["--out", str(out)]

    assert fidelity.main(argv) == 0
    first = out.read_bytes()
    assert fidelity.main(argv) == 0
    assert out.read_bytes() == first

    moved = json.loads(first)
    moved["inputs"]["run_report_sha256"] = "0" * 64
    out.write_text(json.dumps(moved), encoding="utf-8")
    assert fidelity.main(argv) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8")) == moved

    changed = json.loads(first)
    changed["anisotropic_pooled_median_deg"] += 1.0
    out.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert fidelity.main(argv) == 1
    assert "FAIL" in capsys.readouterr().out
