"""Ingestion-side validation of the measurement sidecar.

The sidecar is an *independently produced* record whose only consumer
re-validates it, so the read path carries the checks — not the write path.  A
table this project wrote is the easy case; a table edited by hand, produced by
another tool, or left behind by a half-finished run is the case these tests
pin.  Every one of them writes a sidecar directly rather than through
``write_axis``, because ``write_axis`` is exactly the path that must not be
trusted to have run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest

from pose_estimation import inventory, measure


def _row(**overrides: str) -> dict[str, str]:
    row = dict.fromkeys(measure.SYNC_COLUMNS, "")
    row.update(
        capture_id="c-1",
        asset_a="a-1",
        asset_b="a-2",
        offset_audio_s="0.100000000",
        peak_rms_audio="2.000000000",
        peak_ratio_audio="3.000000000",
        status_audio="ok",
        offset_visual_s="0.110000000",
        conf_visual="5.000000000",
        peak_corr_visual="0.900000000",
        status_visual="ok",
        overlap_s="10.000000000",
        dur_a="20.000000000",
        dur_b="21.000000000",
        audio_rate_a="48000",
        audio_rate_b="48000",
        same_audio_rate="1",
    )
    row.update(overrides)
    return row


def _sidecar(
    tmp_path: pathlib.Path,
    rows: list[dict[str, str]],
    *,
    declared_rows: int | None = None,
    version: object = measure.GENERATOR_VERSION,
    provenance: object | None = None,
) -> pathlib.Path:
    out = tmp_path / "measurements"
    out.mkdir(parents=True, exist_ok=True)
    table = out / "sync_pairs.csv"
    inventory.write_text(table, inventory.render_csv(measure.SYNC_COLUMNS, rows))
    manifest: dict = {
        "axes": {
            "sync": {
                "table": "sync_pairs.csv",
                "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                "rows": len(rows) if declared_rows is None else declared_rows,
                "generator_version": version,
                "provenance": {} if provenance is None else provenance,
            }
        }
    }
    manifest["generation"] = {
        "manifest": measure.manifest_digest(manifest),
        "inventory": {},
        "generator_version": measure.GENERATOR_VERSION,
    }
    inventory.write_text(out / measure.MANIFEST_FILENAME, inventory.render_json(manifest))
    return out


def _load(out: pathlib.Path) -> dict:
    return measure.load_axis(measure.validate(out), "sync")


def test_ingestion_reads_a_well_formed_sidecar(tmp_path: pathlib.Path) -> None:
    assert len(_load(_sidecar(tmp_path, [_row()]))) == 1


def test_ingestion_refuses_a_cell_the_alphabet_forbids(tmp_path: pathlib.Path) -> None:
    out = _sidecar(tmp_path, [_row(offset_audio_s="-1e-3")])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "cell_alphabet"


def test_ingestion_refuses_a_duplicated_logical_key(tmp_path: pathlib.Path) -> None:
    """A dict-built index silently keeps the last row; row_count would not notice."""
    out = _sidecar(tmp_path, [_row(), _row(peak_rms_audio="9.000000000")])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "duplicate_key"


def test_ingestion_refuses_a_descending_pair(tmp_path: pathlib.Path) -> None:
    out = _sidecar(tmp_path, [_row(asset_a="a-2", asset_b="a-1")])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "pair_order"


def test_ingestion_refuses_a_row_count_the_manifest_contradicts(tmp_path: pathlib.Path) -> None:
    """The digest covers the table and the self-digest covers the count, so a
    coherently rewritten manifest can still declare a count no one checks."""
    out = _sidecar(tmp_path, [_row()], declared_rows=2)
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "row_count"


def test_ingestion_refuses_rows_out_of_canonical_order(tmp_path: pathlib.Path) -> None:
    rows = [_row(asset_a="a-3", asset_b="a-4"), _row()]
    with pytest.raises(measure.MeasureError) as error:
        _load(_sidecar(tmp_path, rows))
    assert error.value.reason == "row_order"


def test_ingestion_refuses_an_unpublished_drift_status(tmp_path: pathlib.Path) -> None:
    """``short_overlap`` is a Drift status and reaches no published column."""
    out = _sidecar(tmp_path, [_row(status_audio="short_overlap")])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "status_token"


def test_an_accepted_row_publishes_its_statistics(tmp_path: pathlib.Path) -> None:
    """A gate rejects an estimate; it never erases one. ``ok`` implies numbers."""
    out = _sidecar(tmp_path, [_row(offset_audio_s="")])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "status_cells"


def test_a_refused_row_may_omit_its_statistics(tmp_path: pathlib.Path) -> None:
    rows = [_row(status_visual="signal_absent", offset_visual_s="", conf_visual="")]
    assert len(_load(_sidecar(tmp_path, rows))) == 1


def test_validate_refuses_a_symlinked_manifest(tmp_path: pathlib.Path) -> None:
    out = _sidecar(tmp_path, [_row()])
    manifest = out / measure.MANIFEST_FILENAME
    elsewhere = tmp_path / "elsewhere.json"
    manifest.rename(elsewhere)
    manifest.symlink_to(elsewhere)
    with pytest.raises(measure.MeasureError) as error:
        measure.validate(out)
    assert error.value.reason == "manifest_irregular"


def test_validate_refuses_a_manifest_that_makes_two_claims(tmp_path: pathlib.Path) -> None:
    """Last-key-wins would validate one claim while the bytes carry another."""
    out = _sidecar(tmp_path, [_row()])
    path = out / measure.MANIFEST_FILENAME
    text = path.read_text(encoding="utf-8")
    doubled = text.replace('"rows": 1', '"rows": 1, "rows": 2', 1)
    assert doubled != text
    path.write_text(doubled, encoding="utf-8")
    with pytest.raises(measure.MeasureError) as error:
        measure.validate(out)
    assert error.value.reason == "manifest_duplicate_key"


def test_validate_refuses_an_unsupported_axis_generator_version(tmp_path: pathlib.Path) -> None:
    out = _sidecar(tmp_path, [_row()], version="v0")
    with pytest.raises(measure.MeasureError) as error:
        measure.validate(out)
    assert error.value.reason == "generator_version"


def test_ingestion_reads_the_bytes_it_validated(tmp_path: pathlib.Path) -> None:
    """A digest proves nothing about bytes fetched through a second open."""
    out = _sidecar(tmp_path, [_row()])
    sidecar = measure.validate(out)
    swapped = [_row(asset_a="a-8", asset_b="a-9", peak_rms_audio="7.000000000")]
    inventory.write_text(
        out / "sync_pairs.csv", inventory.render_csv(measure.SYNC_COLUMNS, swapped)
    )
    assert list(measure.load_axis(sidecar, "sync")) == [("a-1", "a-2")]


def test_the_published_sync_statuses_are_the_estimators_own(tmp_path: pathlib.Path) -> None:
    """The alphabet is the estimator's, never a list transcribed beside it."""
    assert "short_overlap" in measure.DRIFT_STATUSES
    assert measure.DRIFT_STATUSES.isdisjoint(measure.AUDIO_STATUSES)
    assert {"ok", "low_confidence"} <= measure.AUDIO_STATUSES
    assert {"ok", "signal_absent", "insufficient_overlap"} <= measure.VISUAL_STATUSES


def test_manifest_json_stays_parseable_after_every_refusal(tmp_path: pathlib.Path) -> None:
    """Refusal is a read-side verdict: the record on disk is never rewritten."""
    out = _sidecar(tmp_path, [_row(offset_audio_s="-1e-3")])
    before = (out / measure.MANIFEST_FILENAME).read_bytes()
    with pytest.raises(measure.MeasureError):
        _load(out)
    assert (out / measure.MANIFEST_FILENAME).read_bytes() == before
    assert json.loads(before.decode("utf-8"))["axes"]["sync"]["rows"] == 1


@pytest.mark.parametrize("column", ["peak_rms_audio", "offset_audio_s", "drift_ppm"])
def test_ingestion_refuses_a_decimal_that_overflows_to_infinity(
    tmp_path: pathlib.Path, column: str
) -> None:
    """Spelling is not finiteness.

    The bounded columns admit `inf` because their upper bound is `inf`; the
    unbounded ones admit it because nothing compares them at all.  Both routes
    put a non-number in a table whose whole purpose is to carry instrument
    readings, so the refusal belongs to the alphabet, not to the domains.
    """
    out = _sidecar(tmp_path, [_row(**{column: "9" * 400 + ".0"})])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "cell_overflow"


def test_ingestion_refuses_an_axis_version_that_is_not_a_string(tmp_path: pathlib.Path) -> None:
    """The version check hashes its operand, so an unhashable one must not escape the domain."""
    out = _sidecar(tmp_path, [_row()], version=[])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "generator_version"


def test_ingestion_refuses_provenance_that_is_not_an_object(tmp_path: pathlib.Path) -> None:
    """A02 opens the provenance key set; it does not drop the object that holds the keys."""
    out = _sidecar(tmp_path, [_row()], provenance=[])
    with pytest.raises(measure.MeasureError) as error:
        _load(out)
    assert error.value.reason == "provenance_shape"


def test_ingestion_refuses_a_sidecar_directory_it_cannot_list(tmp_path: pathlib.Path) -> None:
    """Executable but unreadable: every named file still opens, the listing does not.

    The unnamed-table check is the one read that needs the directory itself, so
    it is the read that escapes the error domain when nothing wraps it.
    """
    out = _sidecar(tmp_path, [_row()])
    out.chmod(0o111)
    try:
        if os.access(out, os.R_OK):
            pytest.skip("this user can list a directory without read permission")
        with pytest.raises(measure.MeasureError) as error:
            _load(out)
        assert error.value.reason == "sidecar_unreadable"
    finally:
        out.chmod(0o755)


def test_every_measure_module_reaches_the_determinism_tripwire() -> None:
    """The staleness tripwire's reach is exactly its list, so the list must be complete.

    ``check_qualify_determinism.py`` refuses to overwrite a result measured
    against different source digests, and ``SOURCE_FILES`` is the whole set it
    digests.  A module this package adds is invisible to that refusal until it
    is listed, which turns a real staleness signal into a silent green.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "check_qualify_determinism.py").read_text(encoding="utf-8")
    listed = {
        line.strip().strip('",')
        for line in script.split("SOURCE_FILES = (", 1)[1].split(")", 1)[0].splitlines()
        if line.strip()
    }
    present = {
        f"src/pose_estimation/measure/{path.name}"
        for path in (root / "src" / "pose_estimation" / "measure").glob("*.py")
    }
    assert present <= listed, f"unlisted measure modules: {sorted(present - listed)}"
