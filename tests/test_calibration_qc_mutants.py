"""Focused oracle for the calibration-QC mutation catalogue."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import pathlib
import shutil

import pytest

from pose_estimation import calibration_qc, qualify
from test_calibration_qc import _arms, _record, _write_evidence, _write_probes
from test_qualify import _publish
from test_sessions import _canonical

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Inputs:
    inventory: pathlib.Path
    sessions: pathlib.Path
    qualification: pathlib.Path
    evidence: pathlib.Path
    probes: pathlib.Path


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    root = tmp_path_factory.mktemp("cqc-mutants")
    inventory, sessions, corpus, qualification = _publish(root, [_canonical(1, "above")])
    qualify.run(inventory, sessions, corpus, qualification)
    probes = _write_probes(root / "probes")
    evidence = root / "evidence"
    _write_evidence(evidence, probes)
    return Inputs(inventory, sessions, qualification, evidence, probes)


def _run(inputs: Inputs, out: pathlib.Path) -> dict[str, object]:
    return calibration_qc.run(
        inputs.qualification,
        inputs.evidence,
        inputs.probes,
        out,
        sessions_dir=inputs.sessions,
        inventory_dir=inputs.inventory,
    )


def _private_evidence(inputs: Inputs, tmp_path: pathlib.Path) -> Inputs:
    """A per-test copy of the module-scoped evidence, so a case may corrupt it."""
    evidence = tmp_path / "evidence"
    shutil.copytree(inputs.evidence, evidence)
    return dataclasses.replace(inputs, evidence=evidence)


def _tree_bytes(root: pathlib.Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def test_oracle_imports_the_worktree_source() -> None:
    path = pathlib.Path(calibration_qc.__file__).resolve()
    if os.environ.get("CQC_ORACLE_SHOW_PATH"):
        print(f"calibration_qc.__file__={path}")
    assert path == ROOT / "src/pose_estimation/calibration_qc.py"


def test_m01_another_generator_version_is_not_owned() -> None:
    generation = dict.fromkeys(calibration_qc.GENERATION_KEYS, "digest")
    generation["generator_version"] = "another-version"
    assert not calibration_qc._is_own_generation(generation)


def test_m02_nonempty_markerless_root_is_not_owned(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "foreign.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_owned(out)


def test_m03_symlink_marker_is_not_read(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    (out / calibration_qc.CALIBRATION_QC_FILENAME).symlink_to(target)
    with pytest.raises(OSError, match="not a regular file"):
        calibration_qc._read_marker(out)


def test_m04_nonregular_marker_is_rejected_before_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / calibration_qc.CALIBRATION_QC_FILENAME).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(calibration_qc.stat, "S_ISREG", lambda _mode: False)
    with pytest.raises(OSError, match="not a regular file"):
        calibration_qc._read_marker(out)


def test_m05_duplicate_marker_key_is_rejected(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / calibration_qc.CALIBRATION_QC_FILENAME).write_text(
        '{"generation": {}, "generation": {"generator_version": "v1"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        calibration_qc._read_marker(out)


def test_m06_generation_shape_is_closed() -> None:
    generation = dict.fromkeys(calibration_qc.GENERATION_KEYS, "digest")
    generation["generator_version"] = calibration_qc.GENERATOR_VERSION
    generation["unexpected"] = "value"
    assert not calibration_qc._is_own_generation(generation)


def test_m07_live_root_moves_before_staging_promotion(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "out"
    _run(inputs, out)
    before = _tree_bytes(out)
    _run(inputs, out)
    assert _tree_bytes(out) == before
    calibration_qc.validate_generation(out, qualification_dir=inputs.qualification)


def test_m08_dead_orphan_is_swept(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    dead = out.with_name(f"{out.name}.staging.999999999")
    dead.mkdir()
    calibration_qc._sweep_orphans(out)
    assert not dead.exists()


def test_m09_m11_failed_promotion_restores_the_live_tree(
    inputs: Inputs, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    _run(inputs, out)
    before = _tree_bytes(out)
    rename = pathlib.Path.rename

    def fail_promotion(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        if self.name.startswith(f"{out.name}.staging.") and pathlib.Path(target) == out:
            raise OSError("injected promotion failure")
        return rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", fail_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        _run(inputs, out)
    assert _tree_bytes(out) == before
    calibration_qc.validate_generation(out, qualification_dir=inputs.qualification)


def test_m10_staging_sibling_carries_the_process_id(
    inputs: Inputs, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    observed: list[str] = []

    def stop(staging: pathlib.Path, *_args: object, **_kwargs: object) -> None:
        observed.append(staging.name)
        raise RuntimeError("staging observed")

    monkeypatch.setattr(calibration_qc, "_build", stop)
    with pytest.raises(RuntimeError, match="staging observed"):
        _run(inputs, out)
    assert observed == [f"out.staging.{os.getpid()}"]


def test_m12_separator_guard_excludes_prefix_siblings(tmp_path: pathlib.Path) -> None:
    parent = tmp_path / "evidence"
    sibling = tmp_path / "evidence-old"
    assert calibration_qc._is_within(str(parent), str(parent))
    assert not calibration_qc._is_within(str(sibling), str(parent))


def test_m13_output_inside_evidence_is_refused(inputs: Inputs, tmp_path: pathlib.Path) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(inputs.evidence, evidence)
    local = dataclasses.replace(inputs, evidence=evidence)
    out = evidence / "nested-output"
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(local, out)
    assert not out.exists()


def test_m14_output_inside_probe_directory_is_refused(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    probes = tmp_path / "probes"
    shutil.copytree(inputs.probes, probes)
    local = dataclasses.replace(inputs, probes=probes)
    out = probes / "nested-output"
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(local, out)
    assert not out.exists()


def test_m15_symlinked_output_keeps_the_link(inputs: Inputs, tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "out"
    link.symlink_to(target, target_is_directory=True)
    _run(inputs, link)
    assert link.is_symlink()
    assert (target / calibration_qc.CALIBRATION_QC_FILENAME).is_file()


def test_m16_schema_guard_cannot_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calibration_qc, "EVIDENCE_COLUMNS", ("probe", "event_id"))
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_schema_is_redaction_safe()


def test_m17_event_token_remains_forbidden() -> None:
    assert "event" in calibration_qc.FORBIDDEN_KEY_TOKENS


def test_m18_capture_identifier_shape_remains_blocked() -> None:
    assert any(shape.search("s01-cap-l") for shape in calibration_qc.IDENTIFIER_SHAPES)


def test_m19_identifier_shaped_cell_is_rejected() -> None:
    rows = [{"arm": "s01-cap-l"}]
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_cells_carry_no_identifier(rows, "evidence_qc.csv")


def test_m20_schema_guard_runs_at_module_initialization() -> None:
    source = pathlib.Path(calibration_qc.__file__).read_text(encoding="utf-8")
    call = source.index("\n_assert_schema_is_redaction_safe()\n")
    cell_guard = source.index("\ndef _assert_cells_carry_no_identifier")
    assert call < cell_guard


def test_m21_publisher_emits_exactly_one_corpus_row(inputs: Inputs, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    _run(inputs, out)
    assert len(_rows(out / calibration_qc.CORPUS_QC_FILENAME)) == 1


def test_m22_empty_capture_cannot_publish(inputs: Inputs, tmp_path: pathlib.Path) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(inputs.evidence, evidence)
    probe = calibration_qc.INGESTED_PROBES[0]
    (evidence / f"{probe}.jsonl").write_text(
        json.dumps({"summary": []}, indent=1) + "\n", encoding="utf-8"
    )
    local = dataclasses.replace(inputs, evidence=evidence)
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(local, tmp_path / "out")


def test_m23_published_headers_keep_the_exact_column_order(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "out"
    _run(inputs, out)
    for name, columns in (
        (calibration_qc.CORPUS_QC_FILENAME, calibration_qc.CORPUS_COLUMNS),
        (calibration_qc.EVIDENCE_QC_FILENAME, calibration_qc.EVIDENCE_COLUMNS),
    ):
        header = next(csv.reader((out / name).read_text(encoding="utf-8").splitlines()))
        assert tuple(header) == columns


def test_m24_cell_alphabet_rejects_a_suffix() -> None:
    rows = [{**calibration_qc.RULING, "recovery_status": "unachievable\n"}]
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_cell_alphabets(
            rows, calibration_qc.CORPUS_CELL_ALPHABETS, calibration_qc.CORPUS_QC_FILENAME
        )


def test_m25_token_alphabet_rejects_a_wrapped_token() -> None:
    pattern = calibration_qc._token_alphabet(frozenset({"closed"}))
    assert pattern.fullmatch("closed")
    assert not pattern.fullmatch("xclosedx")


def test_m26_required_statistic_cells_stay_populated() -> None:
    rows = calibration_qc.evidence_rows("bias_transfer", "0" * 64, [_record("REAL same view pair")])
    row = next(item for item in rows if item["statistic"] == "between_event_r")
    assert all(row[field] for field in calibration_qc.STATISTIC_FIELDS)


def test_m27_integer_alphabet_rejects_non_ascii_digits() -> None:
    assert "٣".isdigit()
    assert not calibration_qc.INTEGER_CELL.fullmatch("٣")


def test_m28_m29_cut_record_is_not_silently_skipped(tmp_path: pathlib.Path) -> None:
    capture = tmp_path / "capture.jsonl"
    capture.write_text(
        json.dumps(_record("REAL same view pair")) + '\n{"label": "cut"',
        encoding="utf-8",
    )
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._read_capture(capture)


def test_m30_recorded_digest_mismatch_refuses_publication(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(inputs.evidence, evidence)
    probe = calibration_qc.INGESTED_PROBES[0]
    (evidence / f"{probe}.sha256").write_text("f" * 64 + "\n", encoding="utf-8")
    local = dataclasses.replace(inputs, evidence=evidence)
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(local, tmp_path / "out")


def test_m31_missing_reference_prefix_refuses_publication(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    prefix = calibration_qc.REQUIRED_ARM_PREFIXES[0]
    evidence = tmp_path / "evidence"
    _write_evidence(
        evidence, inputs.probes, arms=[arm for arm in _arms() if not arm.startswith(prefix)]
    )
    local = dataclasses.replace(inputs, evidence=evidence)
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(local, tmp_path / "out")


def test_m32_permutation_null_remains_required() -> None:
    assert "REAL same view pair, keypoints permuted (null)" in calibration_qc.REQUIRED_ARMS


def test_m33_missing_statistic_block_refuses_publication(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(inputs.evidence, evidence)
    probe = calibration_qc.INGESTED_PROBES[0]
    records = [
        {key: value for key, value in _record(arm).items() if key != "within_event_r"}
        for arm in _arms()
    ]
    (evidence / f"{probe}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    local = dataclasses.replace(inputs, evidence=evidence)
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(local, tmp_path / "out")


def test_m34_missing_uningested_probe_script_refuses_publication(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    probes = tmp_path / "probes"
    shutil.copytree(inputs.probes, probes)
    (probes / calibration_qc.PROBE_SCRIPTS["calibration_bias"]).unlink()
    local = dataclasses.replace(inputs, probes=probes)
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(local, tmp_path / "out")


def test_m35_canonical_order_is_independent_of_input_order() -> None:
    rows = [{"key": "b"}, {"key": "a"}]
    assert calibration_qc._canonical(rows, ("key",)) == [{"key": "a"}, {"key": "b"}]


def test_m36_clean_generation_validates(inputs: Inputs, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    _run(inputs, out)
    calibration_qc.validate_generation(out, qualification_dir=inputs.qualification)


def test_m37_census_digest_excludes_its_self_reference() -> None:
    first = {"generation": {"census": "first", "generator_version": "v1"}}
    second = {"generation": {"census": "second", "generator_version": "v1"}}
    assert calibration_qc.census_digest(first) == calibration_qc.census_digest(second)


def _claim_staging(tmp_path: pathlib.Path, injected: str) -> pathlib.Path:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "claims.txt").write_text(
        "\n".join((*calibration_qc.CLAIMS, injected)), encoding="utf-8"
    )
    return staging


def test_m38_claim_conformance_cannot_be_disabled(tmp_path: pathlib.Path) -> None:
    staging = _claim_staging(tmp_path, calibration_qc.PROHIBITED_PARAPHRASES[0])
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_claim_conformance(staging)


def test_m39_claim_scan_is_case_insensitive(tmp_path: pathlib.Path) -> None:
    staging = _claim_staging(tmp_path, calibration_qc.PROHIBITED_PARAPHRASES[0].upper())
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_claim_conformance(staging)


def test_m40_claim_scan_folds_snake_case(tmp_path: pathlib.Path) -> None:
    staging = _claim_staging(tmp_path, calibration_qc.PROHIBITED_PARAPHRASES[0].replace(" ", "_"))
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_claim_conformance(staging)


def test_m41_upstream_validation_precedes_publication(
    inputs: Inputs, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("upstream validation reached")

    monkeypatch.setattr(qualify, "validate_generation", stop)
    with pytest.raises(RuntimeError, match="upstream validation reached"):
        _run(inputs, tmp_path / "out")


def test_m42_prohibited_paraphrases_never_publish(inputs: Inputs, tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    _run(inputs, out)
    published = "\n".join(
        path.read_text(encoding="utf-8") for path in out.iterdir() if path.is_file()
    ).casefold()
    assert all(paraphrase not in published for paraphrase in calibration_qc.PROHIBITED_PARAPHRASES)
    calibration_qc.validate_generation(out, qualification_dir=inputs.qualification)


def test_m43_claim_scan_folds_hyphens(tmp_path: pathlib.Path) -> None:
    staging = _claim_staging(tmp_path, "per-event double-centered bias-and-pose failed")
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_claim_conformance(staging)


def test_m44_a_record_key_outside_the_closed_set_is_refused(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    private = _private_evidence(inputs, tmp_path)
    probe = calibration_qc.INGESTED_PROBES[0]
    lines = [json.dumps(_record(arm, subject_id="s01")) for arm in _arms()]
    (private.evidence / f"{probe}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(private, tmp_path / "out")
    assert error.value.reason == "forbidden_key"


def test_m45_a_statistic_missing_a_required_field_is_refused(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    private = _private_evidence(inputs, tmp_path)
    probe = calibration_qc.INGESTED_PROBES[0]
    lines = []
    for arm in _arms():
        record = json.loads(json.dumps(_record(arm)))
        del record["within_event_r"]["n"]
        lines.append(json.dumps(record))
    (private.evidence / f"{probe}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(private, tmp_path / "out")
    assert error.value.reason == "evidence_schema"


def test_m46_a_cited_arm_short_of_the_ruled_population_is_refused(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    private = _private_evidence(inputs, tmp_path)
    probe = calibration_qc.INGESTED_PROBES[0]
    lines = [json.dumps(_record(arm, pairs=1, events=1)) for arm in _arms()]
    (private.evidence / f"{probe}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(private, tmp_path / "out")
    assert error.value.reason == "population_mismatch"


def test_m47_a_duplicate_arm_label_is_refused(inputs: Inputs, tmp_path: pathlib.Path) -> None:
    private = _private_evidence(inputs, tmp_path)
    arms = _arms()
    _write_evidence(private.evidence, inputs.probes, arms=[*arms, arms[0]])

    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(private, tmp_path / "out")
    assert error.value.reason == "arm_duplicate"


def test_m48_the_corpus_table_is_held_to_exactly_one_row(tmp_path: pathlib.Path) -> None:
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        calibration_qc._build(tmp_path / "staging", [], [], upstream_qualification={}, probes={})
    assert error.value.reason == "corpus_cardinality"


def test_m49_a_capture_symlinked_into_the_output_is_refused(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    private = _private_evidence(inputs, tmp_path)
    out = tmp_path / "out"
    _run(private, out)
    capture = private.evidence / f"{calibration_qc.INGESTED_PROBES[0]}.jsonl"
    target = out / "publisher-input.jsonl"
    target.write_bytes(capture.read_bytes())
    capture.unlink()
    capture.symlink_to(target)

    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(private, out)
    assert error.value.reason == "output_overlap"


def test_m50_an_added_arm_cannot_give_the_unrun_arm_an_outcome(
    inputs: Inputs, tmp_path: pathlib.Path
) -> None:
    private = _private_evidence(inputs, tmp_path)
    _write_evidence(
        private.evidence,
        inputs.probes,
        arms=[*_arms(), "per-event double-centered bias-and-pose failed"],
    )

    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(private, tmp_path / "out")
    assert error.value.reason == "claim_prohibited"


def test_m51_evidence_is_validated_before_the_output_is_judged(
    inputs: Inputs, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = _private_evidence(inputs, tmp_path)
    probe = calibration_qc.INGESTED_PROBES[0]
    (private.evidence / f"{probe}.sha256").write_text("f" * 64 + "\n", encoding="utf-8")

    def judged(out: pathlib.Path) -> None:
        del out
        raise AssertionError("the output was judged before the inputs were validated")

    monkeypatch.setattr(calibration_qc, "_assert_owned", judged)
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(private, tmp_path / "out")
    assert error.value.reason == "probe_digest"
