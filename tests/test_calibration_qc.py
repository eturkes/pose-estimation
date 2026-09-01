"""Acceptance suite for the calibration-ruling publisher.

Fixtures reuse ``test_qualify``'s registry/session/media builders rather than
reimplementing them: those builders are already pinned by M2.1-M2.3's own
predicate tests, so a second copy would be a second thing to keep true.

The probe scripts are synthesised here rather than read from ``scripts/``.  The
publisher digests whatever script the operator names, so a hermetic directory
exercises the digest binding exactly; one separate case proves the real cited
scripts are on disk under the names the module spells.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import runpy
import subprocess
import sys
from typing import Any

import pytest

from pose_estimation import calibration_qc, qualify
from test_qualify import _publish, _uniform, _write_media
from test_sessions import _canonical

# One arm record shaped exactly as ``probe_bias_transfer.py`` emits it.
_STATS = {"n": 4341, "median": 0.0108, "min": -0.9, "max": 0.99, "above_0p5": 787}
_PX = {"n": 178, "median": 17.038, "min": 4.1, "max": 61.2}


def _record(label: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "label": label,
        "pairs": 178,
        "events": 103,
        "realizations": 1,
        "between_event_r": dict(_STATS),
        "between_event_r_abs": dict(_STATS),
        "within_event_r": dict(_STATS),
        "median_abs_px": dict(_PX),
        "shared_fraction": None,
    }
    record.update(overrides)
    return record


def _arms() -> list[str]:
    return [
        *sorted(calibration_qc.REQUIRED_ARMS),
        "REAL above|left",
        "SYNTH shared image bias 8.0px",
        "SYNTH shared image bias 8.0px, rig jitter 1.2m",
        "SYNTH per-event bias 8.0px",
        "SYNTH noise sigma=8.0px",
    ]


def _write_evidence(
    root: pathlib.Path, probes: pathlib.Path, *, arms: list[str] | None = None
) -> None:
    """Write one capture per ingested probe, plus the digest it was taken under."""
    root.mkdir(parents=True, exist_ok=True)
    for probe in calibration_qc.INGESTED_PROBES:
        lines = [json.dumps(_record(label)) for label in (arms if arms is not None else _arms())]
        # The probes close with a pretty-printed key list spanning several lines.
        lines.append(json.dumps({"summary": ["a", "b"]}, indent=1))
        (root / f"{probe}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        digest = hashlib.sha256(
            (probes / calibration_qc.PROBE_SCRIPTS[probe]).read_bytes()
        ).hexdigest()
        (root / f"{probe}.sha256").write_text(
            f"{digest}  scripts/{calibration_qc.PROBE_SCRIPTS[probe]}\n", encoding="utf-8"
        )


def _write_probes(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in calibration_qc.PROBE_SCRIPTS.values():
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    return root


@pytest.fixture
def published(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """A validated qualification tree plus the inputs the ruling publisher needs."""
    asset = _canonical(1, "above")
    registry, sessions_dir, corpus, qualification = _publish(tmp_path, [asset])
    _write_media(corpus / asset.source_path, _uniform(6))
    qualify.run(registry, sessions_dir, corpus, qualification)
    probes = _write_probes(tmp_path / "probes")
    evidence = tmp_path / "evidence"
    _write_evidence(evidence, probes)
    return {
        "registry": registry,
        "sessions": sessions_dir,
        "qualification": qualification,
        "probes": probes,
        "evidence": evidence,
        "out": tmp_path / "calibration_qc",
    }


def _run(paths: dict[str, pathlib.Path], **kwargs: pathlib.Path | None) -> dict[str, Any]:
    return calibration_qc.run(
        paths["qualification"], paths["evidence"], paths["probes"], paths["out"], **kwargs
    )


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


# --- publication ----------------------------------------------------------------------------------


def test_a_published_set_validates_and_carries_exactly_one_corpus_row(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    generation = calibration_qc.validate_generation(
        published["out"],
        qualification_dir=published["qualification"],
        probes_dir=published["probes"],
    )
    assert generation["generator_version"] == calibration_qc.GENERATOR_VERSION
    rows = _rows(published["out"] / calibration_qc.CORPUS_QC_FILENAME)
    assert len(rows) == 1
    assert rows[0] == calibration_qc.RULING


def test_the_ruling_is_a_constant_and_not_an_argument(published: dict[str, pathlib.Path]) -> None:
    """No CLI or call argument can spell a different verdict."""
    _run(published)
    row = _rows(published["out"] / calibration_qc.CORPUS_QC_FILENAME)[0]
    assert row["recovery_status"] == "unachievable"
    assert row["transfer_status"] == "absent"
    assert row["unrun_arm_status"] == "unrun"
    signature = set(
        calibration_qc.run.__code__.co_varnames[: calibration_qc.run.__code__.co_argcount]
    )
    assert not signature & set(calibration_qc.CORPUS_COLUMNS)


def test_publication_is_byte_identical_across_two_runs(published: dict[str, pathlib.Path]) -> None:
    _run(published)
    first = {
        name: (published["out"] / name).read_bytes()
        for name in (*calibration_qc.CSV_FILENAMES, calibration_qc.CALIBRATION_QC_FILENAME)
    }
    _run(published)
    for name, payload in first.items():
        assert (published["out"] / name).read_bytes() == payload


def test_evidence_rows_are_long_form_over_arm_and_statistic(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    rows = _rows(published["out"] / calibration_qc.EVIDENCE_QC_FILENAME)
    assert {row["statistic"] for row in rows} == set(calibration_qc.STATISTIC_KEYS)
    assert len(rows) == len(_arms()) * len(calibration_qc.STATISTIC_KEYS)
    assert {row["probe"] for row in rows} == set(calibration_qc.INGESTED_PROBES)
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["probe_sha256"]) for row in rows)


def test_published_row_order_is_a_function_of_the_rows(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    rows = _rows(published["out"] / calibration_qc.EVIDENCE_QC_FILENAME)
    keys = [(row["probe"], row["arm"], row["statistic"]) for row in rows]
    assert keys == sorted(keys)


def test_a_nullable_statistic_field_publishes_an_empty_cell(
    published: dict[str, pathlib.Path],
) -> None:
    """``median_abs_px`` carries no ``above_0p5``; an absent field is empty, never invented."""
    _run(published)
    rows = _rows(published["out"] / calibration_qc.EVIDENCE_QC_FILENAME)
    px = [row for row in rows if row["statistic"] == "median_abs_px"]
    assert px
    assert all(row["above_0p5"] == "" for row in px)
    assert all(row["median"] for row in px)


# --- ownership, atomicity, disjointness (probe seed classes 1-3) ------------------------------------


def test_a_non_empty_unowned_output_is_refused(published: dict[str, pathlib.Path]) -> None:
    published["out"].mkdir()
    (published["out"] / "someone_elses.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(published)
    assert (published["out"] / "someone_elses.txt").exists()


def test_an_output_path_that_is_not_a_directory_is_refused(
    published: dict[str, pathlib.Path],
) -> None:
    published["out"].write_text("file", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(published)


def test_a_marker_that_is_a_symlink_never_licenses_a_delete(
    published: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """The marker is the trust root; a symlinked one puts that root outside the set."""
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "payload.txt").write_text("not ours", encoding="utf-8")
    real = tmp_path / "elsewhere.json"
    real.write_text(json.dumps({"generation": {"generator_version": "v1"}}), encoding="utf-8")
    (foreign / calibration_qc.CALIBRATION_QC_FILENAME).symlink_to(real)
    published["out"] = foreign
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(published)
    assert (foreign / "payload.txt").exists()


def test_a_marker_carrying_a_duplicate_key_is_refused(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    marker = published["out"] / calibration_qc.CALIBRATION_QC_FILENAME
    marker.write_text(
        marker.read_text(encoding="utf-8").replace(
            '{\n  "claims"', '{\n  "claims": [],\n  "claims"', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"])


def test_a_set_published_by_another_generator_version_is_not_owned(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    marker = published["out"] / calibration_qc.CALIBRATION_QC_FILENAME
    census = json.loads(marker.read_text(encoding="utf-8"))
    census["generation"]["generator_version"] = "v0"
    marker.write_text(json.dumps(census), encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(published)
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"])


def test_a_symlinked_output_publishes_to_its_target(
    published: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    published["out"] = link
    _run(published)
    assert link.is_symlink()
    assert (target / calibration_qc.CALIBRATION_QC_FILENAME).is_file()


def test_orphan_siblings_of_a_dead_publisher_are_swept(
    published: dict[str, pathlib.Path],
) -> None:
    out = published["out"]
    dead = out.with_name(f"{out.name}.staging.999999999")
    dead.mkdir(parents=True)
    wide = out.with_name(f"{out.name}.retiring.{2**80}")
    wide.mkdir(parents=True)
    _run(published)
    assert not dead.exists()
    # An unrepresentable pid raises OverflowError from int(), never ValueError.
    assert not wide.exists()


def test_a_sibling_owned_by_a_live_process_survives_the_sweep(
    published: dict[str, pathlib.Path],
) -> None:
    """Pid 1 is always alive, and unreachable pids raise PermissionError, never a delete."""
    out = published["out"]
    alive = out.with_name(f"{out.name}.staging.1")
    alive.mkdir(parents=True)
    _run(published)
    assert alive.exists()


@pytest.mark.parametrize("key", ["qualification", "evidence", "probes"])
def test_an_output_overlapping_an_input_is_refused_in_both_directions(
    published: dict[str, pathlib.Path], key: str
) -> None:
    published["out"] = published[key] / "nested"
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(published)
    published["out"] = published[key].parent
    with pytest.raises(calibration_qc.CalibrationQcError):
        _run(published)


# --- integrity (probe seed class 4) -----------------------------------------------------------------


@pytest.mark.parametrize("name", calibration_qc.CSV_FILENAMES)
def test_an_edited_table_is_refused(published: dict[str, pathlib.Path], name: str) -> None:
    _run(published)
    path = published["out"] / name
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"])


def test_an_edited_marker_claim_is_refused(published: dict[str, pathlib.Path]) -> None:
    _run(published)
    marker = published["out"] / calibration_qc.CALIBRATION_QC_FILENAME
    census = json.loads(marker.read_text(encoding="utf-8"))
    census["corpus"]["ruling"]["recovery_status"] = "achievable"
    marker.write_text(json.dumps(census), encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"])


def test_a_file_added_to_the_set_is_refused(published: dict[str, pathlib.Path]) -> None:
    _run(published)
    (published["out"] / "extra.csv").write_text("smuggled\n", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"])


@pytest.mark.parametrize("name", calibration_qc.CSV_FILENAMES)
def test_a_file_removed_from_the_set_is_refused(
    published: dict[str, pathlib.Path], name: str
) -> None:
    _run(published)
    (published["out"] / name).unlink()
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"])


def test_a_stale_upstream_qualification_is_refused(
    published: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    _run(published)
    calibration_qc.validate_generation(
        published["out"], qualification_dir=published["qualification"]
    )
    asset = _canonical(2, "left")
    second = tmp_path / "second"
    second.mkdir()
    registry, sessions_dir, corpus, qualification = _publish(second, [asset])
    _write_media(corpus / asset.source_path, _uniform(6))
    qualify.run(registry, sessions_dir, corpus, qualification)
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"], qualification_dir=qualification)


def test_an_edited_probe_script_makes_the_published_set_stale(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    calibration_qc.validate_generation(published["out"], probes_dir=published["probes"])
    script = published["probes"] / calibration_qc.PROBE_SCRIPTS["calibration_bias"]
    script.write_text(script.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc.validate_generation(published["out"], probes_dir=published["probes"])


# --- alphabets (probe seed class 5) -----------------------------------------------------------------


def test_every_enumerated_alphabet_is_its_token_set_and_not_a_shape() -> None:
    """A character class accepts every token the partition excludes."""
    for tokens, column in (
        (calibration_qc.RECOVERY_STATUSES, "recovery_status"),
        (calibration_qc.TRANSFER_STATUSES, "transfer_status"),
        (calibration_qc.ARM_RUN_STATUSES, "unrun_arm_status"),
    ):
        pattern = calibration_qc.CORPUS_CELL_ALPHABETS[column]
        assert {token for token in tokens if pattern.fullmatch(token)} == set(tokens)
        for invented in ("achievable", "partial", "refused", "failed", "unknown"):
            if invented not in tokens:
                assert not pattern.fullmatch(invented)


def test_a_cell_alphabet_uses_fullmatch_so_a_trailing_newline_is_refused() -> None:
    rows = [{**calibration_qc.RULING, "recovery_status": "unachievable\n"}]
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        calibration_qc._assert_cell_alphabets(
            rows, calibration_qc.CORPUS_CELL_ALPHABETS, calibration_qc.CORPUS_QC_FILENAME
        )
    assert error.value.reason == "cell_alphabet"


def test_an_arm_label_the_probes_never_spell_is_refused() -> None:
    rows = [
        {
            "probe": "bias_transfer",
            "probe_sha256": "0" * 64,
            "arm": " leading space",
            "statistic": "between_event_r",
            "n": "1",
            "median": "0.1",
            "min": "0.0",
            "max": "0.2",
            "above_0p5": "0",
        }
    ]
    with pytest.raises(calibration_qc.CalibrationQcError):
        calibration_qc._assert_cell_alphabets(
            rows, calibration_qc.EVIDENCE_CELL_ALPHABETS, calibration_qc.EVIDENCE_QC_FILENAME
        )


def test_an_integer_cell_rejects_a_non_ascii_digit() -> None:
    """``str.isdigit()`` is true for superscripts and other scripts; the alphabet is ASCII."""
    assert "²".isdigit()
    assert not calibration_qc.INTEGER_CELL.fullmatch("²")
    assert not calibration_qc.INTEGER_CELL.fullmatch("٣")


def test_an_evidence_table_with_no_rows_is_refused(published: dict[str, pathlib.Path]) -> None:
    """A zero-row CSV carries its schema in the header alone, so it must never publish."""
    for probe in calibration_qc.INGESTED_PROBES:
        (published["evidence"] / f"{probe}.jsonl").write_text(
            json.dumps({"summary": []}, indent=1) + "\n", encoding="utf-8"
        )
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "evidence_empty"
    assert not published["out"].exists()


# --- D04 unrepresentability (probe seed class 6) -----------------------------------------------------


def test_no_published_column_can_key_a_row_to_a_recording_or_a_person() -> None:
    for columns in (calibration_qc.CORPUS_COLUMNS, calibration_qc.EVIDENCE_COLUMNS):
        for column in columns:
            assert not set(column.split("_")) & calibration_qc.FORBIDDEN_KEY_TOKENS


def test_the_schema_check_fires_on_a_forbidden_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calibration_qc, "EVIDENCE_COLUMNS", ("probe", "event_id"))
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        calibration_qc._assert_schema_is_redaction_safe()
    assert error.value.reason == "forbidden_key"


@pytest.mark.parametrize("smuggled", ["s01-cap-l", "0123456789abcdef"])
def test_an_identifier_shaped_cell_is_refused_even_where_the_alphabet_admits_it(
    published: dict[str, pathlib.Path], smuggled: str
) -> None:
    _write_evidence(published["evidence"], published["probes"], arms=[*_arms(), f"REAL {smuggled}"])
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "forbidden_value"
    assert not published["out"].exists()


def test_no_published_byte_carries_a_corpus_identifier(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    for path in published["out"].rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for shape in calibration_qc.IDENTIFIER_SHAPES:
                assert not shape.search(text)


# --- evidence validation (probe seed class 7) --------------------------------------------------------


def test_a_capture_truncated_mid_line_is_refused(published: dict[str, pathlib.Path]) -> None:
    probe = calibration_qc.INGESTED_PROBES[0]
    path = published["evidence"] / f"{probe}.jsonl"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text[: text.index("\n", len(text) // 2)] + '{"label": "REAL cut"}}\n', encoding="utf-8"
    )
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason in {"evidence_malformed", "arm_missing"}


def test_a_capture_missing_a_cited_arm_is_refused(published: dict[str, pathlib.Path]) -> None:
    kept = [arm for arm in _arms() if arm != "REAL same view pair + same subject"]
    _write_evidence(published["evidence"], published["probes"], arms=kept)
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "arm_missing"


def test_a_record_cut_before_its_closing_brace_is_refused(
    published: dict[str, pathlib.Path],
) -> None:
    """A killed run leaves a record with no closing brace; skipping it drops a cited arm."""
    probe = calibration_qc.INGESTED_PROBES[0]
    path = published["evidence"] / f"{probe}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0][: len(lines[0]) // 2]
    assert lines[0].startswith("{")
    assert not lines[0].endswith("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "evidence_malformed"
    assert not published["out"].exists()


def test_the_pretty_printed_closing_summary_is_not_read_as_a_record(
    published: dict[str, pathlib.Path],
) -> None:
    """The probes close with an indented document whose opening line is a bare brace."""
    records = calibration_qc._read_capture(
        published["evidence"] / f"{calibration_qc.INGESTED_PROBES[0]}.jsonl"
    )
    assert len(records) == len(_arms())
    assert all("summary" not in record for record in records)


@pytest.mark.parametrize("prefix", calibration_qc.REQUIRED_ARM_PREFIXES)
def test_a_capture_missing_a_reference_band_is_refused(
    published: dict[str, pathlib.Path], prefix: str
) -> None:
    """A corpus reading with no calibrated reference is not a placeable value."""
    kept = [arm for arm in _arms() if not arm.startswith(prefix)]
    _write_evidence(published["evidence"], published["probes"], arms=kept)
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "arm_missing"


def test_a_capture_omitting_a_required_statistic_is_refused(
    published: dict[str, pathlib.Path],
) -> None:
    probe = calibration_qc.INGESTED_PROBES[0]
    lines = [
        json.dumps({key: value for key, value in _record(label).items() if key != "within_event_r"})
        for label in _arms()
    ]
    (published["evidence"] / f"{probe}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "evidence_schema"


def test_a_capture_taken_under_a_different_script_is_refused(
    published: dict[str, pathlib.Path],
) -> None:
    probe = calibration_qc.INGESTED_PROBES[0]
    (published["evidence"] / f"{probe}.sha256").write_text("f" * 64 + "\n", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "probe_digest"
    assert not published["out"].exists()


def test_a_capture_with_no_recorded_digest_is_refused(
    published: dict[str, pathlib.Path],
) -> None:
    probe = calibration_qc.INGESTED_PROBES[0]
    (published["evidence"] / f"{probe}.sha256").unlink()
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "digest_missing"


def test_a_malformed_recorded_digest_is_refused(published: dict[str, pathlib.Path]) -> None:
    probe = calibration_qc.INGESTED_PROBES[0]
    (published["evidence"] / f"{probe}.sha256").write_text("not-a-digest\n", encoding="utf-8")
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "digest_malformed"


def test_a_missing_cited_probe_script_is_refused(published: dict[str, pathlib.Path]) -> None:
    (published["probes"] / calibration_qc.PROBE_SCRIPTS["calibration_bias"]).unlink()
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "probe_missing"


def test_an_uningested_probe_is_still_cited_and_digested(
    published: dict[str, pathlib.Path],
) -> None:
    """``calibration_bias`` backs the ruling by digest; its stdout is not ingested."""
    assert "calibration_bias" not in calibration_qc.INGESTED_PROBES
    census = _run(published)
    assert set(census["generation"]["probes"]) == set(calibration_qc.PROBE_SCRIPTS)
    row = _rows(published["out"] / calibration_qc.CORPUS_QC_FILENAME)[0]
    assert set(row["cited_probes"].split("|")) == set(calibration_qc.PROBE_SCRIPTS)


def test_the_cited_probe_scripts_exist_in_the_repository() -> None:
    scripts = pathlib.Path(__file__).resolve().parent.parent / "scripts"
    for name in calibration_qc.PROBE_SCRIPTS.values():
        assert (scripts / name).is_file()


# --- claim conformance (probe seed class 8) ----------------------------------------------------------


def test_every_required_claim_reaches_the_published_bytes(
    published: dict[str, pathlib.Path],
) -> None:
    _run(published)
    marker = (published["out"] / calibration_qc.CALIBRATION_QC_FILENAME).read_text(encoding="utf-8")
    census = json.loads(marker)
    assert census["claims"] == list(calibration_qc.CLAIMS)
    for claim in calibration_qc.CLAIMS:
        assert claim in json.dumps(census)


@pytest.mark.parametrize("paraphrase", calibration_qc.PROHIBITED_PARAPHRASES)
def test_no_prohibited_paraphrase_reaches_the_published_bytes(
    published: dict[str, pathlib.Path], paraphrase: str
) -> None:
    _run(published)
    for path in published["out"].rglob("*"):
        if path.is_file():
            assert paraphrase not in path.read_text(encoding="utf-8").casefold()


def test_the_claim_scan_refuses_a_set_that_overclaims(
    published: dict[str, pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: without this scan an overreach publishes cleanly."""
    monkeypatch.setattr(
        calibration_qc,
        "RULING",
        {**calibration_qc.RULING, "unrun_arm": "no_estimator_could_recover_extrinsics"},
    )
    monkeypatch.setattr(
        calibration_qc,
        "CORPUS_CELL_ALPHABETS",
        {
            key: (re.compile(r"[a-z_]+") if key == "unrun_arm" else value)
            for key, value in calibration_qc.CORPUS_CELL_ALPHABETS.items()
        },
    )
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "claim_prohibited"
    assert not published["out"].exists()


def test_the_claim_scan_refuses_a_set_that_dropped_a_claim(
    published: dict[str, pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = calibration_qc.build_census

    def _short(corpus_rows: list[dict[str, str]], evidence: list[dict[str, str]]) -> dict[str, Any]:
        census = original(corpus_rows, evidence)
        census["claims"] = list(calibration_qc.CLAIMS[:-1])
        return census

    monkeypatch.setattr(calibration_qc, "build_census", _short)
    with pytest.raises(calibration_qc.CalibrationQcError) as error:
        _run(published)
    assert error.value.reason == "claim_missing"


def test_the_unrun_arm_is_never_spelled_as_a_measured_outcome() -> None:
    assert frozenset({"unrun"}) == calibration_qc.ARM_RUN_STATUSES
    pattern = calibration_qc.CORPUS_CELL_ALPHABETS["unrun_arm_status"]
    for measured in ("failed", "refused", "impossible", "unachievable"):
        assert not pattern.fullmatch(measured)


def test_the_ruling_never_fills_a_qualification_sentinel(
    published: dict[str, pathlib.Path],
) -> None:
    """One corpus-level ruling holds while every per-event geometry cell stays unmeasured."""
    before = {
        path.name: path.read_bytes()
        for path in sorted(published["qualification"].iterdir())
        if path.is_file()
    }
    _run(published)
    after = {
        path.name: path.read_bytes()
        for path in sorted(published["qualification"].iterdir())
        if path.is_file()
    }
    assert before == after
    events = _rows(published["qualification"] / "events_qc.csv")
    assert events
    assert all(row["geom_qualified"] == "" for row in events)
    assert all("geom_unmeasured" in row["reason"] for row in events)


def test_committed_determinism_evidence_matches_its_exact_source_tripwire() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "tests/calibration_qc_determinism_results.json").read_text(encoding="utf-8")
    )
    checker = runpy.run_path(str(root / "scripts/check_calibration_qc_determinism.py"))
    assert result["source_digests"] == checker["source_digests"]()


# --- CLI --------------------------------------------------------------------------------------------


def test_the_cli_publishes_and_reports_counts_only(published: dict[str, pathlib.Path]) -> None:
    code = calibration_qc.main(
        [
            "--qualification",
            str(published["qualification"]),
            "--evidence",
            str(published["evidence"]),
            "--probes",
            str(published["probes"]),
            "--out",
            str(published["out"]),
        ]
    )
    assert code == 0
    calibration_qc.validate_generation(published["out"])


def test_the_cli_folds_a_refusal_to_status_two(
    published: dict[str, pathlib.Path], capsys: pytest.CaptureFixture[str]
) -> None:
    (published["probes"] / calibration_qc.PROBE_SCRIPTS["bias_transfer"]).unlink()
    code = calibration_qc.main(
        [
            "--qualification",
            str(published["qualification"]),
            "--evidence",
            str(published["evidence"]),
            "--probes",
            str(published["probes"]),
            "--out",
            str(published["out"]),
        ]
    )
    assert code == 2
    assert "Error:" in capsys.readouterr().err


def test_the_console_summary_carries_no_identifier(published: dict[str, pathlib.Path]) -> None:
    census = _run(published)
    summary = calibration_qc.render_summary(census if "claims" in census else _marker(published))
    for shape in calibration_qc.IDENTIFIER_SHAPES:
        assert not shape.search(summary)


def _marker(published: dict[str, pathlib.Path]) -> dict[str, object]:
    return json.loads(
        (published["out"] / calibration_qc.CALIBRATION_QC_FILENAME).read_text(encoding="utf-8")
    )


def test_the_packaged_command_is_registered() -> None:
    pyproject = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'pose-estimation-calibration-qc = "pose_estimation.calibration_qc:main"' in pyproject


def test_the_generated_tree_is_ignored_by_git() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.exit(0)"], capture_output=True, check=False
    )
    assert result.returncode == 0
    ignored = subprocess.run(
        ["git", "check-ignore", "calibration_qc", "calibration_qc.staging.1/"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 0
    assert {line.strip() for line in ignored.stdout.splitlines()} == {
        "calibration_qc",
        "calibration_qc.staging.1/",
    }


# --- crash states the pid suffix cannot separate ---------------------------------------------------


def test_a_reused_pid_restores_the_only_complete_generation(
    published: dict[str, pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between the two renames leaves the whole set under a dead pid.

    Pids are reused, so the next run can own that suffix.  Removing it before
    the replacement exists would destroy the sole complete generation -- the
    state the post-swap sweep is explicitly ordered to protect.
    """
    _run(published)
    out = published["out"]
    retiring = out.with_name(f"{out.name}.retiring.{os.getpid()}")
    out.rename(retiring)

    def fail_build(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected pre-promotion failure")

    monkeypatch.setattr(calibration_qc, "_build", fail_build)
    with pytest.raises(OSError, match="injected pre-promotion failure"):
        _run(published)

    assert not retiring.exists()
    calibration_qc.validate_generation(out, qualification_dir=published["qualification"])


def test_same_pid_debris_that_is_a_regular_file_is_swept(
    published: dict[str, pathlib.Path],
) -> None:
    """A regular file at either sibling path blocks the mkdir and then the swap."""
    _run(published)
    out = published["out"]
    for suffix in ("staging", "retiring"):
        out.with_name(f"{out.name}.{suffix}.{os.getpid()}").write_text("debris", encoding="utf-8")

    _run(published)

    for suffix in ("staging", "retiring"):
        assert not out.with_name(f"{out.name}.{suffix}.{os.getpid()}").exists()
    calibration_qc.validate_generation(out, qualification_dir=published["qualification"])


@pytest.mark.parametrize(
    ("upstream", "victim"), [("registry", "assets.csv"), ("sessions", "stray.txt")]
)
def test_an_upstream_refusal_reaches_the_operator_as_an_error_line(
    published: dict[str, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
    upstream: str,
    victim: str,
) -> None:
    """`run` validates the whole chain, so its refusals must not arrive as a traceback."""
    _run(published)
    target = published[upstream] / victim
    previous = target.read_text(encoding="utf-8") if target.exists() else ""
    target.write_text(previous + "tamper\n", encoding="utf-8")

    code = calibration_qc.main(
        [
            "--qualification",
            str(published["qualification"]),
            "--evidence",
            str(published["evidence"]),
            "--probes",
            str(published["probes"]),
            "--out",
            str(published["out"]),
            "--sessions",
            str(published["sessions"]),
            "--inventory",
            str(published["registry"]),
        ]
    )

    assert code == 2
    assert capsys.readouterr().err.startswith("Error: ")
