"""Adversarial regressions found by the M2 cohort milestone review."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import runpy

import pytest
import yaml

from pose_estimation import cohort, corpus_run, export, inventory, sessions
from test_cohort import _build_inputs

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_r05_angle_metadata_tracks_isotropic_coordinate_normalization() -> None:
    assert export.COORD_NORMALIZATION == "image-isotropic-maxdim"
    assert cohort.UNIT_DEG == "deg_image_plane"
    text = (_ROOT / "docs" / "technical" / "cohort.md").read_text(encoding="utf-8")
    assert "normalizes x and y by one scalar" in text
    assert "normalization is anisotropic" not in text


def test_r06_manifest_event_and_camera_must_match_sessions(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    manifest = corpus_run.read_manifest(inputs.run / corpus_run.MANIFEST_FILENAME)
    source, replacement = manifest[:2]
    assert (source["event_id"], source["camera_name"]) != (
        replacement["event_id"],
        replacement["camera_name"],
    )
    source["event_id"] = replacement["event_id"]
    source["camera_name"] = replacement["camera_name"]
    corpus_run.write_manifest(inputs.run / corpus_run.MANIFEST_FILENAME, manifest)

    with pytest.raises(cohort.CohortError):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")


def test_r07_duplicate_source_header_is_refused(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    feature = next(item.column for item in cohort.FEATURES if item.level == "frame")
    for path in inputs.run.glob("*/*_clinical.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        index = rows[0].index(feature)
        rows[0].append(feature)
        for row in rows[1:]:
            row.append(row[index])
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

    with pytest.raises(cohort.CohortError):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")


def test_r20_descriptor_renderer_round_trips_yaml_line_break_codepoints() -> None:
    label = "before" + chr(0x85) + "after"
    row: dict[str, object] = {
        "raw": "pose_frame_probe",
        "ja": label,
        "en": label,
        "group": "pose",
        "role": "feature",
        "dtype": "numeric",
        "unit": "ratio",
        "range": [None, 1.0],
    }
    loaded = yaml.safe_load(cohort.render_descriptors([row]))["columns"][0]
    assert loaded["ja"] == label
    assert loaded["en"] == label


def test_r25_extra_publication_entry_is_refused(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    out = tmp_path / "cohort"
    cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    (out / "unverified.txt").write_text("outside the digest\n", encoding="utf-8")

    with pytest.raises(cohort.CohortError):
        cohort.validate_generation(out)


def test_r29_pid_named_debris_survives_a_failed_preswap_attempt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    out = tmp_path / "cohort"
    debris = out.with_name(f"{out.name}.staging.{os.getpid()}")
    debris.mkdir()
    sentinel = debris / "foreign"
    sentinel.write_text("keep\n", encoding="utf-8")

    def fail_publish(*_args: object) -> None:
        raise OSError("injected pre-swap failure")

    monkeypatch.setattr(cohort, "_publish", fail_publish)
    with pytest.raises(OSError, match="injected pre-swap failure"):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_r30_complete_pid_reused_staging_survives_until_a_new_swap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    out = tmp_path / "cohort"
    staged = out.with_name(f"{out.name}.staging.{os.getpid()}")
    cohort.run(inputs.inventory, inputs.sessions, inputs.run, staged)
    before = {path.name: path.read_bytes() for path in staged.iterdir()}

    def fail_publish(*_args: object) -> None:
        raise OSError("injected pre-swap failure")

    monkeypatch.setattr(cohort, "_publish", fail_publish)
    with pytest.raises(OSError, match="injected pre-swap failure"):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    assert {path.name: path.read_bytes() for path in staged.iterdir()} == before


def test_r31_successful_swap_preserves_unowned_sibling_directory(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    out = tmp_path / "cohort"
    foreign = out.with_name(f"{out.name}.staging.foreign")
    foreign.mkdir()
    sentinel = foreign / "foreign"
    sentinel.write_text("keep\n", encoding="utf-8")

    cohort.run(inputs.inventory, inputs.sessions, inputs.run, out)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize("source", ["inventory", "sessions", "run"])
def test_r32_concurrent_upstream_mutation_is_detected(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    original = cohort._aggregate
    target = {
        "inventory": lambda: inputs.inventory / inventory.ASSETS_FILENAME,
        "sessions": lambda: inputs.sessions / sessions.PLACEMENTS_FILENAME,
        "run": lambda: next(inputs.run.glob("*/*_clinical.csv")),
    }[source]

    def aggregate_then_mutate(
        contributors: list[cohort._Contributor], run_root: pathlib.Path
    ) -> cohort._Aggregate:
        result = original(contributors, run_root)
        with target().open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(cohort, "_aggregate", aggregate_then_mutate)
    with pytest.raises(cohort.CohortError):
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")


def test_r35_committed_campaign_evidence_matches_its_declared_sources() -> None:
    """Re-derived from the campaign's own tuple: a transcribed list goes stale at R36."""
    campaign = _ROOT / "scripts" / "check_cohort_determinism.py"
    namespace = runpy.run_path(str(campaign), run_name="cohort_campaign_review")
    recorded = json.loads(
        (_ROOT / "tests" / "cohort_determinism_results.json").read_text(encoding="utf-8")
    )
    assert recorded["source_digest"] == namespace["_source_digest"]()


def test_r36_campaign_digest_covers_its_fixture_and_publisher_dependencies() -> None:
    campaign = _ROOT / "scripts" / "check_cohort_determinism.py"
    namespace = runpy.run_path(str(campaign), run_name="cohort_campaign_review")
    declared = {path.resolve() for path in namespace["SOURCES"]}
    required = {
        campaign.resolve(),
        *(
            _ROOT / "src" / "pose_estimation" / name
            for name in ("cohort.py", "inventory.py", "sessions.py", "corpus_run.py", "qualify.py")
        ),
        _ROOT / "tests" / "test_cohort.py",
        _ROOT / "tests" / "test_sessions.py",
        _ROOT / "tests" / "goldens" / "r_clinical" / "2d_idx_clinical.csv",
        _ROOT / "tests" / "goldens" / "r_clinical" / "2d_idx_clinical_windows.csv",
    }
    assert required <= declared


@pytest.mark.parametrize("invalid", ["inventory", "sessions", "manifest"])
def test_r41_invalid_inputs_raise_exactly_cohort_error(
    tmp_path: pathlib.Path, invalid: str
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    target = {
        "inventory": inputs.inventory / inventory.CENSUS_FILENAME,
        "sessions": inputs.sessions / sessions.GENERATION_FILENAME,
        "manifest": inputs.run / corpus_run.MANIFEST_FILENAME,
    }[invalid]
    target.unlink()

    with pytest.raises(cohort.CohortError) as caught:
        cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")
    assert type(caught.value) is cohort.CohortError


def _schema(path: pathlib.Path, raws: list[str]) -> pathlib.Path:
    path.write_text(yaml.safe_dump({"columns": [{"raw": raw} for raw in raws]}), encoding="utf-8")
    return path


def test_r37_census_digests_the_consumer_schema_so_the_collision_check_reruns(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = _schema(tmp_path / "columns.yaml", ["AGE", "ASIA_admission"])
    monkeypatch.setattr(
        cohort,
        "_external_descriptors",
        lambda: (
            cohort.external_raws(schema.read_bytes()),
            cohort._digest_bytes(schema.read_bytes()),
        ),
    )
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    out = cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")
    collision = cohort.validate_generation(out)["descriptor_collision"]
    assert collision["checked"] is True
    assert collision["n_external"] == 2
    assert collision["source_sha256"] == cohort._digest_bytes(schema.read_bytes())
    cohort.recheck_descriptor_collision(out, schema)

    published = yaml.safe_load((out / cohort.DESCRIPTORS_FILENAME).read_text(encoding="utf-8"))
    colliding = _schema(tmp_path / "moved.yaml", ["AGE", published["columns"][0]["raw"]])
    with pytest.raises(cohort.CohortError, match="not the one"):
        cohort.recheck_descriptor_collision(out, colliding)


def test_r37_an_unchecked_set_cannot_be_rechecked(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cohort, "_external_descriptors", lambda: None)
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    out = cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")
    collision = cohort.validate_generation(out)["descriptor_collision"]
    assert collision["checked"] is False
    assert collision["source_sha256"] is None
    with pytest.raises(cohort.CohortError, match="without a descriptor collision check"):
        cohort.recheck_descriptor_collision(out, _schema(tmp_path / "columns.yaml", ["AGE"]))


def test_r37_recheck_reds_a_census_whose_names_and_digest_disagree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    raw = cohort._descriptor_rows()[0]["raw"]
    schema = _schema(tmp_path / "columns.yaml", [str(raw), "AGE"])
    # A publisher that checked other names than the bytes it digested: 2 names, neither `raw`.
    monkeypatch.setattr(
        cohort,
        "_external_descriptors",
        lambda: (frozenset({"AGE", "SEX"}), cohort._digest_bytes(schema.read_bytes())),
    )
    out = cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")
    with pytest.raises(cohort.CohortError, match="collides"):
        cohort.recheck_descriptor_collision(out, schema)


def test_r37_recheck_reds_a_census_whose_count_and_digest_disagree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    inputs = _build_inputs(root, cohort)
    schema = _schema(tmp_path / "columns.yaml", ["AGE", "SEX"])
    monkeypatch.setattr(
        cohort,
        "_external_descriptors",
        lambda: (frozenset({"AGE"}), cohort._digest_bytes(schema.read_bytes())),
    )
    out = cohort.run(inputs.inventory, inputs.sessions, inputs.run, tmp_path / "cohort")
    with pytest.raises(cohort.CohortError, match="descriptor count"):
        cohort.recheck_descriptor_collision(out, schema)
