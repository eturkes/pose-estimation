"""Producer-identity schema contract for 3D clinical artifacts."""

from __future__ import annotations

import csv
import hashlib
import pathlib
import shutil
import subprocess

import pytest

from pose_estimation.processing import TRACKING_BODY
from test_r_pipeline import _generate_csv, _write_world3d_fixture

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
_UTILS_R = _PROJECT_ROOT / "analysis" / "utils.R"

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
_KIND_VALUES = {
    "frame": ("clinical-frame-3d", "frame-instantaneous"),
    "window": ("clinical-window-3d", "gap-aware"),
    "phase": ("movement-phase-3d", "gap-unsafe"),
    "window_qc": ("window_qc", "gap-aware"),
}
_OUTPUT_SUFFIXES = {
    "frame": "_clinical_3d.csv",
    "window": "_clinical_3d_windows.csv",
    "phase": "_movement_phases_3d.csv",
    "window_qc": "_clinical_3d_window_qc.csv",
}
_SHARED_VALUES = {
    "coord_space": "world-metric-3d",
    "distance_unit": "m",
    "producer_version": "v2",
    "metric_method_version": "v1",
    "qc_policy_version": "v2",
    "provenance_class": "unverified",
}
_FRAME_BASE_COLUMNS = (
    "video",
    "frame_idx",
    "timestamp_sec",
    "person_idx",
    "left_elbow_angle_deg",
    "left_wrist_deviation_deg",
    "left_finger_spread_deg",
    "left_reach_raw",
    "left_reach_norm",
    "left_grasp_aperture_thumb_index",
    "left_grasp_aperture_thumb_pinky",
    "left_wrist_displacement",
    "left_fingertip_displacement",
    "right_elbow_angle_deg",
    "right_wrist_deviation_deg",
    "right_finger_spread_deg",
    "right_reach_raw",
    "right_reach_norm",
    "right_grasp_aperture_thumb_index",
    "right_grasp_aperture_thumb_pinky",
    "right_wrist_displacement",
    "right_fingertip_displacement",
    "elbow_angle_deg_symmetry_ratio",
    "elbow_angle_deg_dominance_index",
    "elbow_angle_deg_abs_diff",
    "wrist_deviation_deg_symmetry_ratio",
    "wrist_deviation_deg_dominance_index",
    "wrist_deviation_deg_abs_diff",
    "finger_spread_deg_symmetry_ratio",
    "finger_spread_deg_dominance_index",
    "finger_spread_deg_abs_diff",
    "reach_raw_symmetry_ratio",
    "reach_raw_dominance_index",
    "reach_raw_abs_diff",
    "reach_norm_symmetry_ratio",
    "reach_norm_dominance_index",
    "reach_norm_abs_diff",
    "grasp_aperture_thumb_index_symmetry_ratio",
    "grasp_aperture_thumb_index_dominance_index",
    "grasp_aperture_thumb_index_abs_diff",
    "grasp_aperture_thumb_pinky_symmetry_ratio",
    "grasp_aperture_thumb_pinky_dominance_index",
    "grasp_aperture_thumb_pinky_abs_diff",
    "wrist_displacement_symmetry_ratio",
    "wrist_displacement_dominance_index",
    "wrist_displacement_abs_diff",
    "fingertip_displacement_symmetry_ratio",
    "fingertip_displacement_dominance_index",
    "fingertip_displacement_abs_diff",
    "trunk_lean_lateral_deg",
    "trunk_lean_deg",
    "trunk_lean_sagittal_deg",
    "trunk_rotation_deg",
    "posture_symmetry",
)
_WINDOW_BASE_COLUMNS = (
    "video",
    "person_idx",
    "window_start_sec",
    "window_end_sec",
    "left_wrist_sal",
    "left_wrist_velocity_mean",
    "left_wrist_velocity_peak",
    "left_wrist_normalized_jerk",
    "left_wrist_movement_efficiency",
    "left_fingertip_normalized_jerk",
    "right_wrist_sal",
    "right_wrist_velocity_mean",
    "right_wrist_velocity_peak",
    "right_wrist_normalized_jerk",
    "right_wrist_movement_efficiency",
    "right_fingertip_normalized_jerk",
    "compensatory_pattern_index",
    "trunk_lean_mean",
    "trunk_lean_sd",
    "trunk_lean_range",
    "trunk_lean_sagittal_mean",
    "trunk_lean_sagittal_sd",
    "trunk_lean_lateral_mean",
    "trunk_lean_lateral_sd",
    "trunk_rotation_mean",
    "trunk_rotation_sd",
    "posture_symmetry_mean",
    "posture_symmetry_sd",
    "wrist_sal_symmetry_ratio",
    "wrist_sal_dominance_index",
    "wrist_sal_abs_diff",
    "wrist_velocity_mean_symmetry_ratio",
    "wrist_velocity_mean_dominance_index",
    "wrist_velocity_mean_abs_diff",
    "wrist_velocity_peak_symmetry_ratio",
    "wrist_velocity_peak_dominance_index",
    "wrist_velocity_peak_abs_diff",
    "wrist_normalized_jerk_symmetry_ratio",
    "wrist_normalized_jerk_dominance_index",
    "wrist_normalized_jerk_abs_diff",
    "wrist_movement_efficiency_symmetry_ratio",
    "wrist_movement_efficiency_dominance_index",
    "wrist_movement_efficiency_abs_diff",
    "fingertip_normalized_jerk_symmetry_ratio",
    "fingertip_normalized_jerk_dominance_index",
    "fingertip_normalized_jerk_abs_diff",
)
_PHASE_BASE_COLUMNS = (
    "video",
    "person_idx",
    "side",
    "movement_idx",
    "phase",
    "start_frame",
    "end_frame",
    "duration_sec",
    "peak_velocity",
    "mean_velocity",
    "path_length",
    "smoothness_nj",
    "smoothness_sal",
    "mean_reach_symmetry",
    "movement_duration_sec",
    "movement_n_phases",
    "movement_peak_velocity",
    "movement_path_length",
    "movement_efficiency",
)
_WINDOW_QC_BASE_COLUMNS = (
    "video",
    "person_idx",
    "window_start_sec",
    "window_end_sec",
    "metric_id",
    "source_group",
    "n_expected_frames",
    "n_valid_frames",
    "frame_coverage",
    "n_expected_intervals",
    "n_valid_intervals",
    "interval_coverage",
    "valid_duration_sec",
    "longest_gap_frames",
    "longest_gap_sec",
    "n_gaps",
    "required_keypoints",
    "n_required_keypoints_present",
    "min_coverage",
    "max_gap_sec",
    "qc_status",
    "qc_reason",
)
_BASE_COLUMNS = {
    "frame": _FRAME_BASE_COLUMNS,
    "window": _WINDOW_BASE_COLUMNS,
    "phase": _PHASE_BASE_COLUMNS,
    "window_qc": _WINDOW_QC_BASE_COLUMNS,
}


def _r_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    check = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status=as.integer(!all(vapply(c('dplyr','tidyr','readr','stringr','purrr'), requireNamespace, logical(1), quietly=TRUE))))",
        ],
        capture_output=True,
        timeout=30,
    )
    return check.returncode == 0


requires_r = pytest.mark.skipif(not _r_available(), reason="R or required packages unavailable")


def _run_producer(
    csv_path: pathlib.Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["Rscript", str(_CLINICAL_R), str(csv_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if check:
        assert result.returncode == 0, f"R script failed:\n{result.stderr}"
    return result


def _output_path(source: pathlib.Path, kind: str) -> pathlib.Path:
    return source.with_name(f"{source.stem}{_OUTPUT_SUFFIXES[kind]}")


def _read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    assert path.exists(), f"missing {path.name}"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _make_world3d(path: pathlib.Path, *, n_frames: int = 90) -> pathlib.Path:
    _write_world3d_fixture(path, n_frames=n_frames)
    return path


def _expected_header(kind: str) -> list[str]:
    return [*_BASE_COLUMNS[kind], *_TAG_COLUMNS]


def _r_character_vector(values: tuple[str, ...]) -> str:
    return "c(" + ", ".join(repr(value) for value in values) + ")"


def _assert_nonblank_tags(rows: list[dict[str, str]], source: str) -> None:
    for column in _TAG_COLUMNS:
        invalid = [
            index
            for index, row in enumerate(rows, start=1)
            if row[column] is None or row[column].strip() in {"", "NA"}
        ]
        assert not invalid, f"{source}: blank/NA {column} at rows {invalid}"


def _assert_tag_block(path: pathlib.Path, kind: str, source_hash: str) -> None:
    header, rows = _read_csv(path)
    assert header == _expected_header(kind)
    assert rows, f"{path.name} unexpectedly empty"
    _assert_nonblank_tags(rows, path.name)
    artifact_kind, qualification = _KIND_VALUES[kind]
    expected = {
        "artifact_kind": artifact_kind,
        "source_sha256": source_hash,
        **_SHARED_VALUES,
        "metric_qualification": qualification,
    }
    for column in _TAG_COLUMNS:
        values = [row[column] for row in rows]
        assert set(values) == {expected[column]}, f"{path.name}: wrong {column}"


@requires_r
def test_nonempty_artifacts_carry_exact_identity_schema(tmp_path: pathlib.Path) -> None:
    source = _make_world3d(tmp_path / "capture.csv")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    _run_producer(source)

    for kind in _OUTPUT_SUFFIXES:
        _assert_tag_block(_output_path(source, kind), kind, source_hash)


@requires_r
def test_identity_tags_are_nonblank_singletons_across_one_run(tmp_path: pathlib.Path) -> None:
    source = _make_world3d(tmp_path / "capture.csv")
    _run_producer(source)
    by_kind = {}
    for kind in _OUTPUT_SUFFIXES:
        header, rows = _read_csv(_output_path(source, kind))
        missing = set(_TAG_COLUMNS).difference(header)
        assert not missing, f"{kind}: missing tag columns: {sorted(missing)}"
        _assert_nonblank_tags(rows, kind)
        by_kind[kind] = rows

    for column in _TAG_COLUMNS:
        values = {kind: {row[column] for row in rows} for kind, rows in by_kind.items()}
        assert all(len(kind_values) == 1 for kind_values in values.values()), (
            f"non-singleton or blank {column}: {values}"
        )
        if column not in {"artifact_kind", "metric_qualification"}:
            assert len(set().union(*values.values())) == 1, f"cross-artifact drift: {column}"


@requires_r
def test_short_3d_writes_typed_empty_windows(tmp_path: pathlib.Path) -> None:
    source = _make_world3d(tmp_path / "short.csv", n_frames=3)
    _run_producer(source)

    header, rows = _read_csv(_output_path(source, "window"))
    assert header == _expected_header("window")
    assert rows == []


@requires_r
def test_static_3d_writes_typed_empty_phases(tmp_path: pathlib.Path) -> None:
    source = _make_world3d(tmp_path / "static.csv")
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        for coord in ("x_m", "y_m", "z_m"):
            row[f"body_left_wrist_{coord}"] = rows[0][f"body_left_wrist_{coord}"]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _run_producer(source)

    header, phase_rows = _read_csv(_output_path(source, "phase"))
    assert header == _expected_header("phase")
    assert phase_rows == []


@requires_r
def test_empty_and_nonempty_headers_match_for_each_derived_kind(tmp_path: pathlib.Path) -> None:
    nonempty = _make_world3d(tmp_path / "nonempty.csv")
    short = _make_world3d(tmp_path / "short.csv", n_frames=3)
    static = _make_world3d(tmp_path / "static.csv")
    with static.open(newline="") as handle:
        reader = csv.DictReader(handle)
        static_fields = list(reader.fieldnames or [])
        static_rows = list(reader)
    for row in static_rows:
        for coord in ("x_m", "y_m", "z_m"):
            row[f"body_left_wrist_{coord}"] = static_rows[0][f"body_left_wrist_{coord}"]
    with static.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=static_fields)
        writer.writeheader()
        writer.writerows(static_rows)

    for source in (nonempty, short, static):
        _run_producer(source)

    pairs = {
        "window": (nonempty, short),
        "phase": (nonempty, static),
        "window_qc": (nonempty, short),
    }
    for kind, (full_source, empty_source) in pairs.items():
        full_header, full_rows = _read_csv(_output_path(full_source, kind))
        empty_header, empty_rows = _read_csv(_output_path(empty_source, kind))
        assert full_rows
        assert empty_rows == []
        assert empty_header == full_header == _expected_header(kind)


@requires_r
def test_empty_rerun_overwrites_stale_derived_rows(tmp_path: pathlib.Path) -> None:
    short = _make_world3d(tmp_path / "short.csv", n_frames=3)
    static = _make_world3d(tmp_path / "static.csv")
    with static.open(newline="") as handle:
        reader = csv.DictReader(handle)
        static_fields = list(reader.fieldnames or [])
        static_rows = list(reader)
    for row in static_rows:
        for coord in ("x_m", "y_m", "z_m"):
            row[f"body_left_wrist_{coord}"] = static_rows[0][f"body_left_wrist_{coord}"]
    with static.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=static_fields)
        writer.writeheader()
        writer.writerows(static_rows)

    targets = {
        _output_path(short, "window"): "window",
        _output_path(static, "phase"): "phase",
        _output_path(short, "window_qc"): "window_qc",
    }
    for target, kind in targets.items():
        target.write_text("sentinel\nSURVIVED\n")
        _run_producer(short if kind in {"window", "window_qc"} else static)
        header, rows = _read_csv(target)
        assert header == _expected_header(kind)
        assert rows == [], f"stale rows survived in {target.name}"
        assert b"SURVIVED" not in target.read_bytes()


@requires_r
def test_identical_input_bytes_reproduce_identical_outputs(tmp_path: pathlib.Path) -> None:
    first = _make_world3d(tmp_path / "first.csv")
    second = tmp_path / "second.csv"
    second.write_bytes(first.read_bytes())
    _run_producer(first)
    _run_producer(second)

    for kind in _OUTPUT_SUFFIXES:
        assert _output_path(first, kind).read_bytes() == _output_path(second, kind).read_bytes()


@requires_r
def test_source_sha256_binds_bytes_not_filename_or_video(tmp_path: pathlib.Path) -> None:
    first = _make_world3d(tmp_path / "first.csv")
    renamed = tmp_path / "renamed.csv"
    changed = tmp_path / "changed.csv"
    renamed.write_bytes(first.read_bytes())
    changed.write_bytes(first.read_bytes() + b"\n")

    for source in (first, renamed, changed):
        _run_producer(source)

    hashes = {}
    for source in (first, renamed, changed):
        header, rows = _read_csv(_output_path(source, "frame"))
        assert "source_sha256" in header, "missing source_sha256"
        hashes[source.name] = {row["source_sha256"] for row in rows}
        assert hashes[source.name] == {hashlib.sha256(source.read_bytes()).hexdigest()}
    assert hashes[first.name] == hashes[renamed.name]
    assert hashes[first.name] != hashes[changed.name]


@requires_r
@pytest.mark.parametrize("bad_video", ["different-capture", "", "   ", "NA"])
def test_invalid_video_identity_fails_closed(tmp_path: pathlib.Path, bad_video: str) -> None:
    source = _make_world3d(tmp_path / "invalid_video.csv")
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if bad_video == "different-capture":
        rows[-1]["video"] = bad_video
    else:
        for row in rows:
            row["video"] = bad_video
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = _run_producer(source, check=False)
    assert result.returncode != 0
    assert "video" in result.stderr.lower()


@requires_r
def test_2d_outputs_keep_legacy_schema_and_empty_skip(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "short_2d.csv"
    _generate_csv(source, TRACKING_BODY, n_frames=3)
    _run_producer(source)

    frame = source.with_name("short_2d_clinical.csv")
    header, _ = _read_csv(frame)
    assert set(_TAG_COLUMNS).isdisjoint(header)
    assert not source.with_name("short_2d_clinical_windows.csv").exists()
    assert not source.with_name("short_2d_movement_phases.csv").exists()


@requires_r
def test_2d_named_world3d_routes_by_header_and_preserves_bytes(tmp_path: pathlib.Path) -> None:
    ordinary = tmp_path / "ordinary.csv"
    misleading = tmp_path / "world3d.csv"
    _generate_csv(ordinary, TRACKING_BODY)
    misleading.write_bytes(ordinary.read_bytes())
    input_bytes = misleading.read_bytes()

    _run_producer(ordinary)
    _run_producer(misleading)

    assert misleading.read_bytes() == input_bytes
    assert (
        misleading.with_name("world3d_clinical.csv").read_bytes()
        == ordinary.with_name("ordinary_clinical.csv").read_bytes()
    )
    assert not misleading.with_name("world3d_clinical_3d.csv").exists()


@requires_r
def test_identity_tags_never_become_aggregate_features(tmp_path: pathlib.Path) -> None:
    source = _make_world3d(tmp_path / "capture.csv")
    _run_producer(source)
    frame = _output_path(source, "frame")
    header, _ = _read_csv(frame)
    missing = set(_TAG_COLUMNS).difference(header)
    assert not missing, f"missing tag columns: {sorted(missing)}"
    tags = _r_character_vector(_TAG_COLUMNS)
    script = tmp_path / "aggregate.R"
    script.write_text(
        "\n".join(
            [
                f"source({str(_UTILS_R)!r})",
                f"x <- readr::read_csv({str(frame)!r}, show_col_types=FALSE)",
                f"stopifnot(all(vapply(x[{tags}], is.character, logical(1))))",
                "tagged <- names(aggregate_per_video(x, METADATA_COLS))",
                f"untagged <- names(aggregate_per_video(dplyr::select(x, -dplyr::all_of({tags})), METADATA_COLS))",
                "stopifnot(identical(tagged, untagged))",
                "hazard <- x",
                "hazard$producer_version <- 1",
                "polluted <- names(aggregate_per_video(hazard, METADATA_COLS))",
                "stopifnot(!identical(polluted, untagged))",
                "stopifnot(any(grepl('^producer_version__', polluted)))",
            ]
        )
    )
    result = subprocess.run(["Rscript", str(script)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"R aggregate oracle failed:\n{result.stderr}"


@requires_r
def test_phase_qualification_is_gap_unsafe_in_full_and_empty_schema(tmp_path: pathlib.Path) -> None:
    full = _make_world3d(tmp_path / "full.csv")
    static = _make_world3d(tmp_path / "static.csv")
    with static.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        for coord in ("x_m", "y_m", "z_m"):
            row[f"body_left_wrist_{coord}"] = rows[0][f"body_left_wrist_{coord}"]
    with static.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _run_producer(full)
    _run_producer(static)
    full_header, full_rows = _read_csv(_output_path(full, "phase"))
    empty_header, empty_rows = _read_csv(_output_path(static, "phase"))
    assert full_rows
    assert {row["metric_qualification"] for row in full_rows} == {"gap-unsafe"}
    assert empty_rows == []
    assert full_header == empty_header == _expected_header("phase")


@requires_r
def test_zero_row_3d_input_writes_all_canonical_artifacts(tmp_path: pathlib.Path) -> None:
    populated = _make_world3d(tmp_path / "empty.csv", n_frames=1)
    lines = populated.read_bytes().splitlines(keepends=True)
    populated.write_bytes(lines[0])
    _run_producer(populated)

    for kind in _OUTPUT_SUFFIXES:
        header, rows = _read_csv(_output_path(populated, kind))
        assert header == _expected_header(kind)
        assert rows == []


@requires_r
def test_identity_vocabulary_never_claims_session_or_trial(tmp_path: pathlib.Path) -> None:
    source = _make_world3d(tmp_path / "capture.csv")
    _run_producer(source)

    forbidden = ("session", "trial", "visit", "task")
    for kind in _OUTPUT_SUFFIXES:
        header, rows = _read_csv(_output_path(source, kind))
        missing = set(_TAG_COLUMNS).difference(header)
        assert not missing, f"{kind}: missing tag columns: {sorted(missing)}"
        identity_text = " ".join(
            [
                *(name for name in header if name in _TAG_COLUMNS),
                *(row[name] for row in rows for name in _TAG_COLUMNS),
            ]
        ).lower()
        assert not any(word in identity_text for word in forbidden)
