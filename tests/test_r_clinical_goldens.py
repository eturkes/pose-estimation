"""Exact regression oracle for gap-free R clinical producer outputs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import pathlib
import shutil
import subprocess

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
_GENERATOR_PATH = _PROJECT_ROOT / "scripts" / "regenerate_r_clinical_goldens.py"
_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "goldens" / "r_clinical"
_DATASETS = {
    "2d_idx": ("2d_idx_clinical.csv", "2d_idx_clinical_windows.csv"),
    "2d_cumsum": ("2d_cumsum_clinical.csv", "2d_cumsum_clinical_windows.csv"),
    "2d_csv4dp": ("2d_csv4dp_clinical.csv", "2d_csv4dp_clinical_windows.csv"),
    "world3d": ("world3d_clinical_3d.csv", "world3d_clinical_3d_windows.csv"),
}
_OUTPUT_CASES = tuple(
    (dataset, filename, kind)
    for dataset, filenames in _DATASETS.items()
    for filename, kind in zip(filenames, ("frame", "window"), strict=True)
)
_TAG_COLUMNS = (
    "artifact_kind",
    "source_sha256",
    "coord_space",
    "distance_unit",
    "producer_version",
    "metric_method_version",
    "qc_policy_version",
    "metric_qualification",
    "provenance_class",
)
_METADATA_COLUMNS = {
    "video",
    "frame_idx",
    "timestamp_sec",
    "person_idx",
    "window_start_sec",
    "window_end_sec",
    *_TAG_COLUMNS,
}
_REQUIRED_WINDOW_METRICS = {
    f"{side}_{metric}"
    for side in ("left", "right")
    for metric in (
        "wrist_sal",
        "wrist_velocity_mean",
        "wrist_velocity_peak",
        "wrist_normalized_jerk",
        "wrist_movement_efficiency",
        "fingertip_normalized_jerk",
    )
}
_REQUIRED_WINDOW_METRICS.update(
    f"{metric}_{derivative}"
    for metric in (
        "wrist_sal",
        "wrist_velocity_mean",
        "wrist_velocity_peak",
        "wrist_normalized_jerk",
        "wrist_movement_efficiency",
        "fingertip_normalized_jerk",
    )
    for derivative in ("symmetry_ratio", "dominance_index", "abs_diff")
)


def _r_available() -> bool:
    """Return whether the producer's R runtime and packages are usable."""
    if not shutil.which("Rscript"):
        return False
    try:
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                'for (p in c("dplyr","tidyr","readr","stringr","purrr")) '
                "if (!requireNamespace(p, quietly=TRUE)) quit(status=1)",
            ],
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


requires_r = pytest.mark.skipif(not _r_available(), reason="R or required packages unavailable")


def _load_generator():
    spec = importlib.util.spec_from_file_location("r_clinical_golden_generator", _GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def regenerated_outputs(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("r-clinical-goldens")
    generator = _load_generator()
    generator.regenerate(output_dir)
    return output_dir


@requires_r
@pytest.mark.parametrize(("dataset", "filename", "kind"), _OUTPUT_CASES)
def test_clinical_golden_bytes(dataset, filename, kind, regenerated_outputs):
    """Complete-grid CSV bytes remain frozen across the whole producer surface."""
    del dataset, kind
    expected = _GOLDEN_DIR / filename
    actual = regenerated_outputs / filename
    assert actual.read_bytes() == expected.read_bytes(), (
        f"{filename} drifted: expected {_sha256(expected)}, got {_sha256(actual)}"
    )


@requires_r
@pytest.mark.parametrize(("dataset", "filename", "kind"), _OUTPUT_CASES)
def test_clinical_golden_numeric_values_exact(dataset, filename, kind, regenerated_outputs):
    """Metric cells compare by exact parsed float equality; no tolerances or rounding."""
    del dataset
    expected_fields, expected_rows = _read_csv(_GOLDEN_DIR / filename)
    columns = tuple(column for column in expected_fields if column not in _METADATA_COLUMNS)
    actual_fields, actual_rows = _read_csv(regenerated_outputs / filename)
    assert actual_fields == expected_fields
    assert len(actual_rows) == len(expected_rows)
    for column in columns:
        assert column in expected_fields
        numeric_count = 0
        for row_index, (expected, actual) in enumerate(
            zip(expected_rows, actual_rows, strict=True)
        ):
            expected_value = expected[column]
            actual_value = actual[column]
            if expected_value in {"", "NA"}:
                assert actual_value == expected_value, (
                    f"{filename}:{row_index}:{column} changed sentinel "
                    f"{expected_value!r} to {actual_value!r}"
                )
                continue
            numeric_count += 1
            assert actual_value not in {"", "NA"}, (
                f"{filename}:{row_index}:{column} lost numeric coverage"
            )
            assert float(actual_value) == float(expected_value), (
                f"{filename}:{row_index}:{column} changed from {expected_value} to {actual_value}"
            )
        if numeric_count == 0:
            assert all(row[column] == "NA" for row in expected_rows), (
                f"{filename}:{column} has no numeric coverage or stable NA schema"
            )


@requires_r
@pytest.mark.parametrize(("dataset", "filename", "kind"), _OUTPUT_CASES)
def test_clinical_golden_schema_exact(dataset, filename, kind, regenerated_outputs):
    """Rows, ordered columns, and every metric column remain structurally exact."""
    expected_fields, expected_rows = _read_csv(_GOLDEN_DIR / filename)
    actual_fields, actual_rows = _read_csv(regenerated_outputs / filename)
    assert actual_fields == expected_fields
    assert len(actual_rows) == len(expected_rows)
    # 3D outputs carry the artifact identity tags last; 2D outputs carry none
    # of them, which is what keeps metric-3D rows out of the 2D globs.
    is_3d = dataset == "world3d"
    if is_3d:
        assert tuple(actual_fields[-len(_TAG_COLUMNS) :]) == _TAG_COLUMNS
    else:
        assert not set(actual_fields) & set(_TAG_COLUMNS)
    base_width = 54 if kind == "frame" else 46
    assert len(actual_fields) == base_width + (len(_TAG_COLUMNS) if is_3d else 0)
    assert set(actual_fields) - _METADATA_COLUMNS
    if kind == "window":
        assert set(actual_fields) >= _REQUIRED_WINDOW_METRICS
    assert expected_rows
