"""Acceptance oracle for the 3D per-metric window QC evidence artifact.

Trajectory source groups only; the derived and body groups arrive with the
metric registry that covers them.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import pathlib
import statistics
import subprocess
from dataclasses import dataclass

import pytest

from test_r_clinical_goldens import _load_generator
from test_r_pipeline import _r_available, _write_world3d_fixture
from test_r_trajectory_kernel import _run_r

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
_ANALYSIS_DOC = _PROJECT_ROOT / "docs" / "technical" / "analysis.md"

_QC_SUFFIX = "_clinical_3d_window_qc.csv"
_WINDOW_SUFFIX = "_clinical_3d_windows.csv"
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
_QC_COLUMNS = (
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
    *_TAG_COLUMNS,
)
_METRIC_IDS = tuple(
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
)
_SOURCE_KEYPOINTS = {
    "left_wrist": ("body_left_wrist",),
    "right_wrist": ("body_right_wrist",),
    "left_fingertip": ("left_hand_8",),
    "right_fingertip": ("right_hand_8",),
}
_EVIDENCE_FIELDS = (
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
    "n_required_keypoints_present",
)
_MIN_COVERAGE = 0.80
_MAX_GAP_SEC = 0.10
_POLICY_TOLERANCE = 1e-9

_BASELINE_2D_SHA256 = {
    "2d_csv4dp_clinical.csv": "4b6eb62c5833f45e20b0f9d7972014f131da9737dd6620547ae1e955e001e169",
    "2d_csv4dp_clinical_windows.csv": "7d3814f852211a7c0ef0815a8ce4fdc434a675f4cc9a76ac41f47732f995e340",
    "2d_cumsum_clinical.csv": "ee81990d04b803b0065837a1cef0c7907ded04fac0ca72004c09cc8c82743752",
    "2d_cumsum_clinical_windows.csv": "a0538b62d9442621ca421d85ef01b00dd79f050c5e81aad2dbcb6f8ac5b2b807",
    "2d_idx_clinical.csv": "e717bed3d5a3df4a75929c46a9ef49b86810d40306f6be82739afd5d5e733692",
    "2d_idx_clinical_windows.csv": "b138183069244fa227ef4898c5db82127e1a2a10118346493d2f45207bd9a8d7",
}
_BASELINE_3D_NORMALIZED_SHA256 = {
    "world3d_clinical_3d.csv": "24e1f44eb8036f6ca87b0001a828f9116b33b8aa456238f415a07fcc7be941c9",
    "world3d_clinical_3d_windows.csv": "d4c2ee605fe9993a9942965997ab7be467660d13fc2913860e649a2bef17b998",
}

pytestmark = pytest.mark.skipif(not _r_available(), reason="R or required R packages unavailable")


@dataclass(frozen=True)
class CorpusCase:
    name: str
    fps: int
    n_frames: int
    failed_keypoints: tuple[str, ...] = ()
    failed_indices: tuple[int, ...] = ()
    fail_right_side: bool = False


@dataclass(frozen=True)
class ProducerRun:
    source: pathlib.Path
    result: subprocess.CompletedProcess[str]

    @property
    def qc_path(self) -> pathlib.Path:
        return self.source.with_name(f"{self.source.stem}{_QC_SUFFIX}")

    @property
    def window_path(self) -> pathlib.Path:
        return self.source.with_name(f"{self.source.stem}{_WINDOW_SUFFIX}")


@dataclass(frozen=True)
class OracleEvidence:
    n_expected_frames: int
    n_valid_frames: int
    frame_coverage: float
    n_expected_intervals: int
    n_valid_intervals: int
    interval_coverage: float
    valid_duration_sec: float
    longest_gap_frames: int
    longest_gap_sec: float
    n_gaps: int
    n_required_keypoints_present: int
    qc_status: str
    qc_reason: str


def _spread_indices(n_slots: int, count: int) -> tuple[int, ...]:
    return tuple(int((index + 0.5) * n_slots / count) for index in range(count))


_CASES = (
    CorpusCase("complete91", 30, 91),
    CorpusCase("one_gap_91", 30, 91, ("body_left_wrist",), (45,)),
    CorpusCase("three_gap_91", 30, 91, ("body_left_wrist",), (44, 45, 46)),
    CorpusCase(
        "scattered15_91",
        30,
        91,
        ("body_left_wrist",),
        tuple(5 + 5 * index for index in range(15)),
    ),
    CorpusCase("right_missing_91", 30, 91, fail_right_side=True),
    CorpusCase(
        "fingertip_missing_91",
        30,
        91,
        ("left_hand_8",),
        tuple(range(91)),
    ),
    CorpusCase(
        "hips_missing_91",
        30,
        91,
        ("body_left_hip", "body_right_hip"),
        tuple(range(91)),
    ),
    CorpusCase(
        "coverage79_100",
        100,
        101,
        ("body_left_wrist",),
        _spread_indices(100, 21),
    ),
    CorpusCase(
        "coverage80_100",
        100,
        101,
        ("body_left_wrist",),
        _spread_indices(100, 20),
    ),
    CorpusCase(
        "coverage81_100",
        100,
        101,
        ("body_left_wrist",),
        _spread_indices(100, 19),
    ),
    CorpusCase("gap09_100", 100, 101, ("body_left_wrist",), tuple(range(40, 49))),
    CorpusCase("gap10_100", 100, 101, ("body_left_wrist",), tuple(range(40, 50))),
    CorpusCase("gap11_100", 100, 101, ("body_left_wrist",), tuple(range(40, 51))),
    CorpusCase("one_gap_24", 24, 25, ("body_left_wrist",), (12,)),
    CorpusCase("one_gap_60", 60, 61, ("body_left_wrist",), (30,)),
    CorpusCase("gap2_24", 24, 25, ("body_left_wrist",), (10, 11)),
    CorpusCase("gap3_24", 24, 25, ("body_left_wrist",), (10, 11, 12)),
    CorpusCase("gap6_60", 60, 61, ("body_left_wrist",), tuple(range(25, 31))),
    CorpusCase("gap7_60", 60, 61, ("body_left_wrist",), tuple(range(25, 32))),
    CorpusCase(
        "one_valid_30",
        30,
        31,
        ("body_left_wrist",),
        tuple(index for index in range(30) if index != 15),
    ),
    CorpusCase(
        "gap_and_coverage_60",
        60,
        61,
        ("body_left_wrist",),
        tuple(range(20, 40)),
    ),
    CorpusCase(
        "coverage_and_estimator_60",
        60,
        61,
        ("body_left_wrist",),
        _spread_indices(60, 13),
    ),
)
_CASE_NAMES = tuple(case.name for case in _CASES)
_CASE_PERSON_INDEX = {name: index for index, name in enumerate(_CASE_NAMES)}


def _source_group(metric_id: str) -> str:
    for group in _SOURCE_KEYPOINTS:
        if metric_id.startswith(f"{group}_"):
            return group
    raise AssertionError(f"metric id outside the trajectory groups: {metric_id}")


def _read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    assert path.exists(), f"M3.3 QC artifact missing: {path.name}"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _write_csv(path: pathlib.Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: str) -> float:
    if not value.strip() or value.strip().upper() == "NA":
        return math.nan
    return float(value)


def _as_int(value: str) -> int:
    return int(float(value))


def _finite(value: str) -> bool:
    return math.isfinite(_as_float(value))


def _run_producer(source: pathlib.Path) -> ProducerRun:
    result = subprocess.run(
        ["Rscript", str(_CLINICAL_R), str(source)],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    return ProducerRun(source, result)


def _assert_run_succeeded(run: ProducerRun) -> None:
    assert run.result.returncode == 0, run.result.stdout + run.result.stderr


def _minimum_jerk(progress: float) -> float:
    return 10 * progress**3 - 15 * progress**4 + 6 * progress**5


def _prepare_case_input(
    path: pathlib.Path, case: CorpusCase
) -> tuple[list[str], list[dict[str, str]]]:
    _write_world3d_fixture(path, n_frames=case.n_frames, fps=float(case.fps))
    header, rows = _read_csv(path)
    prefixes = tuple(name.removesuffix("_x_m") for name in header if name.endswith("_x_m"))

    for index, row in enumerate(rows):
        progress = index / max(case.n_frames - 1, 1)
        reach = _minimum_jerk(progress)
        row["video"] = case.name
        for prefix in prefixes:
            for suffix, value in (
                ("_confidence", "0.9"),
                ("_reproj_err_px", "0.5"),
                ("_candidate_n_views", "3"),
                ("_n_views", "3"),
                ("_cheirality_ok", "1"),
                ("_triangulation_angle_deg", "10"),
            ):
                column = f"{prefix}{suffix}"
                if column in row:
                    row[column] = value

        for side, sign in (("left", -1.0), ("right", 1.0)):
            wrist = f"body_{side}_wrist"
            fingertip = f"{side}_hand_8"
            x = sign * (0.30 + 0.40 * reach)
            y = -0.50 + 0.04 * math.sin(math.pi * progress)
            z = sign * 0.02
            for prefix, offset in ((wrist, 0.0), (fingertip, 0.03 * sign)):
                row[f"{prefix}_x_m"] = f"{x + offset:.12f}"
                row[f"{prefix}_y_m"] = f"{y:.12f}"
                row[f"{prefix}_z_m"] = f"{z:.12f}"

    failed_indices = set(case.failed_indices)
    failed_keypoints = set(case.failed_keypoints)
    if case.fail_right_side:
        failed_keypoints.update(
            prefix for prefix in prefixes if prefix.startswith(("body_right_", "right_hand_"))
        )
        failed_indices = set(range(case.n_frames))

    for index in failed_indices:
        for keypoint in failed_keypoints:
            rows[index][f"{keypoint}_reproj_err_px"] = "21"

    _write_csv(path, header, rows)
    return header, rows


def _point_valid(row: dict[str, str], keypoint: str) -> bool:
    coordinate_columns = tuple(f"{keypoint}_{axis}_m" for axis in "xyz")
    if any(column not in row for column in coordinate_columns):
        return False
    if not all(_finite(row[column]) for column in coordinate_columns):
        return False
    if not _finite(row[f"{keypoint}_reproj_err_px"]):
        return False
    if _as_float(row[f"{keypoint}_reproj_err_px"]) > 20:
        return False
    if not _finite(row[f"{keypoint}_cheirality_ok"]):
        return False
    if _as_float(row[f"{keypoint}_cheirality_ok"]) != 1:
        return False
    angle = f"{keypoint}_triangulation_angle_deg"
    return angle not in row or (_finite(row[angle]) and _as_float(row[angle]) >= 1)


def _mask_for_source(rows: list[dict[str, str]], source_group: str) -> list[bool]:
    if source_group == "cpi":
        trunk = _SOURCE_KEYPOINTS["cpi"]
        return [
            all(_point_valid(row, keypoint) for keypoint in trunk)
            and (_point_valid(row, "body_left_wrist") or _point_valid(row, "body_right_wrist"))
            for row in rows
        ]
    keypoints = _SOURCE_KEYPOINTS[source_group]
    return [all(_point_valid(row, keypoint) for keypoint in keypoints) for row in rows]


def _gap_summary(valid: list[bool]) -> tuple[int, int]:
    longest = 0
    count = 0
    run = 0
    for observed in valid:
        if observed:
            run = 0
            continue
        if run == 0:
            count += 1
        run += 1
        longest = max(longest, run)
    return longest, count


def _metric_estimate(window_rows: list[dict[str, str]], qc_row: dict[str, str]) -> float:
    for row in window_rows:
        if (
            row["video"] == qc_row["video"]
            and row["person_idx"] == qc_row["person_idx"]
            and _as_float(row["window_start_sec"])
            == pytest.approx(_as_float(qc_row["window_start_sec"]), abs=1e-12)
            and _as_float(row["window_end_sec"])
            == pytest.approx(_as_float(qc_row["window_end_sec"]), abs=1e-12)
        ):
            return _as_float(row[qc_row["metric_id"]])
    raise AssertionError(f"orphan QC row: {qc_row}")


def _oracle(
    input_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
    qc_row: dict[str, str],
) -> OracleEvidence:
    video = qc_row["video"]
    person_idx = qc_row["person_idx"]
    window_start = _as_float(qc_row["window_start_sec"])
    window_end = _as_float(qc_row["window_end_sec"])
    group_rows = [
        row for row in input_rows if row["video"] == video and row["person_idx"] == person_idx
    ]
    timestamps = [_as_float(row["timestamp_sec"]) for row in group_rows]
    deltas = [right - left for left, right in itertools.pairwise(timestamps)]
    fs = 1 / statistics.median(deltas)
    rows = [
        row for row in group_rows if window_start <= _as_float(row["timestamp_sec"]) < window_end
    ]
    source_group = qc_row["source_group"]
    valid = _mask_for_source(rows, source_group)
    n_expected_frames = len(valid)
    n_valid_frames = sum(valid)
    n_expected_intervals = max(n_expected_frames - 1, 0)
    n_valid_intervals = sum(left and right for left, right in itertools.pairwise(valid))
    frame_coverage = n_valid_frames / n_expected_frames
    interval_coverage = (
        n_valid_intervals / n_expected_intervals if n_expected_intervals else math.nan
    )
    longest_gap_frames, n_gaps = _gap_summary(valid)
    longest_gap_sec = longest_gap_frames / fs
    valid_duration_sec = n_valid_intervals / fs
    required = _SOURCE_KEYPOINTS[source_group]
    present = sum(any(_point_valid(row, keypoint) for row in rows) for keypoint in required)
    estimate = _metric_estimate(window_rows, qc_row)

    if n_valid_frames == 0:
        reason = "missing_required_keypoints"
    elif n_valid_frames < 2 or n_valid_intervals < 1:
        reason = "insufficient_observations"
    elif longest_gap_sec > _MAX_GAP_SEC * (1 + _POLICY_TOLERANCE):
        reason = "gap_too_long"
    elif frame_coverage < _MIN_COVERAGE * (1 - _POLICY_TOLERANCE):
        reason = "insufficient_coverage"
    elif not math.isfinite(estimate):
        reason = "estimator_undefined"
    else:
        reason = "none"

    return OracleEvidence(
        n_expected_frames=n_expected_frames,
        n_valid_frames=n_valid_frames,
        frame_coverage=frame_coverage,
        n_expected_intervals=n_expected_intervals,
        n_valid_intervals=n_valid_intervals,
        interval_coverage=interval_coverage,
        valid_duration_sec=valid_duration_sec,
        longest_gap_frames=longest_gap_frames,
        longest_gap_sec=longest_gap_sec,
        n_gaps=n_gaps,
        n_required_keypoints_present=present,
        qc_status="pass" if reason == "none" else "fail",
        qc_reason=reason,
    )


def _assert_numeric(actual: str, expected: float) -> None:
    value = _as_float(actual)
    if math.isnan(expected):
        assert math.isnan(value)
    else:
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)


def _assert_row_matches_oracle(row: dict[str, str], expected: OracleEvidence) -> None:
    for field in (
        "n_expected_frames",
        "n_valid_frames",
        "n_expected_intervals",
        "n_valid_intervals",
        "longest_gap_frames",
        "n_gaps",
        "n_required_keypoints_present",
    ):
        assert _as_int(row[field]) == getattr(expected, field)
    for field in (
        "frame_coverage",
        "interval_coverage",
        "valid_duration_sec",
        "longest_gap_sec",
    ):
        _assert_numeric(row[field], getattr(expected, field))
    assert row["qc_status"] == expected.qc_status
    assert row["qc_reason"] == expected.qc_reason


def _matches_case(row: dict[str, str], case_name: str) -> bool:
    if case_name in _CASE_PERSON_INDEX:
        return (
            row["video"] == "corpus" and _as_int(row["person_idx"]) == _CASE_PERSON_INDEX[case_name]
        )
    return row["video"] == case_name


def _rows_for_video(rows: list[dict[str, str]], video: str) -> list[dict[str, str]]:
    return [row for row in rows if _matches_case(row, video)]


def _row(
    rows: list[dict[str, str]],
    video: str,
    metric_id: str,
    *,
    window_start: float | None = None,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if _matches_case(row, video)
        and row["metric_id"] == metric_id
        and (
            window_start is None
            or _as_float(row["window_start_sec"]) == pytest.approx(window_start, abs=1e-12)
        )
    ]
    assert matches, f"missing {video}/{metric_id}/{window_start}"
    return matches[0]


def _qc_rows(run: ProducerRun) -> tuple[list[str], list[dict[str, str]]]:
    _assert_run_succeeded(run)
    return _read_csv(run.qc_path)


@pytest.fixture(scope="module")
def corpus_run(tmp_path_factory: pytest.TempPathFactory) -> ProducerRun:
    root = tmp_path_factory.mktemp("r-qc-corpus")
    source = root / "corpus.csv"
    header: list[str] | None = None
    all_rows: list[dict[str, str]] = []
    for case in reversed(_CASES):
        case_header, rows = _prepare_case_input(root / f"{case.name}.csv", case)
        if header is None:
            header = case_header
        else:
            assert case_header == header
        for row in rows:
            row["video"] = "corpus"
            row["person_idx"] = str(_CASE_PERSON_INDEX[case.name])
        all_rows.extend(rows)
    assert header is not None
    _write_csv(source, header, all_rows)
    return _run_producer(source)


@pytest.fixture(scope="module")
def invalid_run(tmp_path_factory: pytest.TempPathFactory) -> ProducerRun:
    root = tmp_path_factory.mktemp("r-qc-invalid")
    case = CorpusCase(
        "invalid_timebase",
        30,
        31,
        ("body_left_wrist",),
        tuple(range(31)),
    )
    source = root / "invalid.csv"
    header, rows = _prepare_case_input(source, case)
    rows[10]["timestamp_sec"] = rows[9]["timestamp_sec"]
    rows[20]["timestamp_sec"], rows[21]["timestamp_sec"] = (
        rows[21]["timestamp_sec"],
        rows[20]["timestamp_sec"],
    )
    _write_csv(source, header, rows)
    return _run_producer(source)


@pytest.fixture(scope="module")
def empty_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, ProducerRun]:
    root = tmp_path_factory.mktemp("r-qc-empty")
    short = root / "short.csv"
    _prepare_case_input(short, CorpusCase("short", 30, 3))

    zero = root / "zero.csv"
    header, _ = _prepare_case_input(zero, CorpusCase("zero", 30, 1))
    _write_csv(zero, header, [])
    return {"short": _run_producer(short), "zero": _run_producer(zero)}


@pytest.fixture(scope="module")
def regenerated_goldens(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    output_dir = tmp_path_factory.mktemp("r-qc-goldens")
    generator = _load_generator()
    generator.regenerate(output_dir)
    return output_dir


def test_qc_schema_inventory_identity_and_order(corpus_run: ProducerRun) -> None:
    header, rows = _qc_rows(corpus_run)
    assert header == list(_QC_COLUMNS)
    window_header, window_rows = _read_csv(corpus_run.window_path)
    expected_windows = {
        (
            row["video"],
            row["person_idx"],
            row["window_start_sec"],
            row["window_end_sec"],
        )
        for row in window_rows
    }
    assert len(rows) == len(expected_windows) * len(_METRIC_IDS)
    for key in expected_windows:
        metric_ids = [
            row["metric_id"]
            for row in rows
            if (
                row["video"],
                row["person_idx"],
                row["window_start_sec"],
                row["window_end_sec"],
            )
            == key
        ]
        assert metric_ids == list(_METRIC_IDS)
    assert set(_METRIC_IDS).issubset(window_header)


@pytest.mark.parametrize("case_name", _CASE_NAMES)
def test_corpus_group_matches_oracle(corpus_run: ProducerRun, case_name: str) -> None:
    _, qc_rows = _qc_rows(corpus_run)
    _, input_rows = _read_csv(corpus_run.source)
    _, window_rows = _read_csv(corpus_run.window_path)
    rows = _rows_for_video(qc_rows, case_name)
    assert rows, f"no QC rows for {case_name}"
    for row in rows:
        expected = _oracle(input_rows, window_rows, row)
        _assert_row_matches_oracle(row, expected)
        assert row["source_group"] == _source_group(row["metric_id"])
        assert row["required_keypoints"] == ",".join(_SOURCE_KEYPOINTS[row["source_group"]])
        _assert_numeric(row["min_coverage"], _MIN_COVERAGE)
        _assert_numeric(row["max_gap_sec"], _MAX_GAP_SEC)


def test_estimator_undefined_is_lowest_reason(corpus_run: ProducerRun) -> None:
    _, rows = _qc_rows(corpus_run)
    # A gap on the window's first slot leaves spectral arc length undefined,
    # because its estimand never extrapolates an edge, while coverage and gap
    # length both clear policy.  Every lower cause is therefore silent.
    sal = _row(rows, "one_gap_91", "left_wrist_sal", window_start=1.5)
    assert sal["qc_status"] == "fail"
    assert sal["qc_reason"] == "estimator_undefined"
    assert _as_float(sal["frame_coverage"]) >= _MIN_COVERAGE
    assert _as_float(sal["longest_gap_sec"]) <= _MAX_GAP_SEC
    # Its sibling over the identical trajectory keeps an unbroken observed
    # path, so the same evidence supports a pass.
    efficiency = _row(rows, "one_gap_91", "left_wrist_movement_efficiency", window_start=1.5)
    assert efficiency["qc_status"] == "pass"
    assert efficiency["qc_reason"] == "none"


def test_dependency_partitions_are_metric_specific(corpus_run: ProducerRun) -> None:
    _, rows = _qc_rows(corpus_run)
    assert _row(rows, "right_missing_91", "right_wrist_velocity_mean")["qc_reason"] == (
        "missing_required_keypoints"
    )
    assert _row(rows, "right_missing_91", "left_wrist_velocity_mean")["qc_reason"] == "none"
    assert (
        _row(rows, "fingertip_missing_91", "left_fingertip_normalized_jerk")["qc_reason"]
        == "missing_required_keypoints"
    )
    assert _row(rows, "fingertip_missing_91", "left_wrist_velocity_mean")["qc_reason"] == "none"
    # A keypoint no trajectory group depends on moves no evidence at all.
    for metric_id in _METRIC_IDS:
        assert _row(rows, "hips_missing_91", metric_id)["qc_reason"] == "none"


def test_divergent_metrics_share_source_evidence(corpus_run: ProducerRun) -> None:
    _, rows = _qc_rows(corpus_run)
    velocity = _row(
        rows,
        "one_gap_91",
        "left_wrist_velocity_mean",
        window_start=1.0,
    )
    efficiency = _row(
        rows,
        "one_gap_91",
        "left_wrist_movement_efficiency",
        window_start=1.0,
    )
    assert velocity["source_group"] == efficiency["source_group"] == "left_wrist"
    assert tuple(velocity[field] for field in _EVIDENCE_FIELDS) == tuple(
        efficiency[field] for field in _EVIDENCE_FIELDS
    )
    assert velocity["qc_reason"] == "none"
    assert efficiency["qc_reason"] == "estimator_undefined"


@pytest.mark.parametrize(
    ("higher", "lower", "video", "metric_id", "run_fixture"),
    [
        (
            "invalid_timebase",
            "missing_required_keypoints",
            "invalid_timebase",
            "left_wrist_velocity_mean",
            "invalid",
        ),
        (
            "missing_required_keypoints",
            "insufficient_observations",
            "right_missing_91",
            "right_wrist_velocity_mean",
            "corpus",
        ),
        (
            "insufficient_observations",
            "gap_too_long",
            "one_valid_30",
            "left_wrist_velocity_mean",
            "corpus",
        ),
        (
            "gap_too_long",
            "insufficient_coverage",
            "gap_and_coverage_60",
            "left_wrist_velocity_mean",
            "corpus",
        ),
        (
            "insufficient_coverage",
            "estimator_undefined",
            "coverage_and_estimator_60",
            "left_wrist_movement_efficiency",
            "corpus",
        ),
    ],
    ids=(
        "invalid_timebase-missing_required_keypoints",
        "missing_required_keypoints-insufficient_observations",
        "insufficient_observations-gap_too_long",
        "gap_too_long-insufficient_coverage",
        "insufficient_coverage-estimator_undefined",
    ),
)
def test_precedence(
    corpus_run: ProducerRun,
    invalid_run: ProducerRun,
    higher: str,
    lower: str,
    video: str,
    metric_id: str,
    run_fixture: str,
) -> None:
    del lower
    run = invalid_run if run_fixture == "invalid" else corpus_run
    _, rows = _qc_rows(run)
    assert _row(rows, video, metric_id)["qc_reason"] == higher


def test_pass_implies_finite_estimate(corpus_run: ProducerRun) -> None:
    _, qc_rows = _qc_rows(corpus_run)
    _, window_rows = _read_csv(corpus_run.window_path)
    for row in qc_rows:
        estimate = _metric_estimate(window_rows, row)
        if row["qc_status"] == "pass":
            assert math.isfinite(estimate)


def test_invalid_timebase_rows_retain_keys_and_na_evidence(
    invalid_run: ProducerRun,
) -> None:
    _, rows = _qc_rows(invalid_run)
    assert rows
    for row in rows:
        assert row["qc_reason"] == "invalid_timebase"
        assert row["qc_status"] == "fail"
        assert all(math.isnan(_as_float(row[field])) for field in _EVIDENCE_FIELDS)
        assert row["metric_id"] in _METRIC_IDS
        assert row["source_group"]
        assert row["required_keypoints"]
        _assert_numeric(row["min_coverage"], _MIN_COVERAGE)
        _assert_numeric(row["max_gap_sec"], _MAX_GAP_SEC)


def test_fully_reversed_timebase_keeps_the_window(tmp_path: pathlib.Path) -> None:
    """A descending clip is published as invalid_timebase, never dropped.

    Its timestamp extent still forms a window, so V21-V24 retain the row.
    Inferring the cadence from a signed difference would skip the whole clip
    before any window was keyed, and the artifact would report nothing at all
    for a defect it exists to report.
    """
    source = tmp_path / "reversed_timebase.csv"
    header, rows = _prepare_case_input(source, CorpusCase("reversed_timebase", 30, 31))
    for row, timestamp in zip(rows, reversed([row["timestamp_sec"] for row in rows]), strict=True):
        row["timestamp_sec"] = timestamp
    _write_csv(source, header, rows)

    _, qc_rows = _qc_rows(_run_producer(source))
    assert len(qc_rows) == len(_METRIC_IDS)
    for row in qc_rows:
        assert row["qc_status"] == "fail"
        assert row["qc_reason"] == "invalid_timebase"
        assert all(math.isnan(_as_float(row[field])) for field in _EVIDENCE_FIELDS)
        assert row["source_group"]
        assert row["required_keypoints"]
        assert all(row[column] for column in _TAG_COLUMNS)


@pytest.mark.parametrize("kind", ["short", "zero"])
def test_typed_empty_qc(empty_runs: dict[str, ProducerRun], kind: str) -> None:
    run = empty_runs[kind]
    header, rows = _qc_rows(run)
    assert header == list(_QC_COLUMNS)
    assert rows == []


def test_typed_empty_qc_clears_stale_rows(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "stale.csv"
    _prepare_case_input(source, CorpusCase("stale", 30, 3))
    qc_path = source.with_name(f"{source.stem}{_QC_SUFFIX}")
    qc_path.write_text("stale\nrow\n")
    run = _run_producer(source)
    header, rows = _qc_rows(run)
    assert header == list(_QC_COLUMNS)
    assert rows == []


def test_qc_identity_tags_and_versions(corpus_run: ProducerRun) -> None:
    _, rows = _qc_rows(corpus_run)
    source_hash = hashlib.sha256(corpus_run.source.read_bytes()).hexdigest()
    expected = {
        "artifact_kind": "window_qc",
        "source_sha256": source_hash,
        "coord_space": "world-metric-3d",
        "distance_unit": "m",
        "producer_version": "v2",
        "metric_method_version": "v1",
        "qc_policy_version": "v2",
        "metric_qualification": "gap-aware",
        "provenance_class": "unverified",
    }
    for row in rows:
        assert {column: row[column] for column in _TAG_COLUMNS} == expected


def test_qc_keys_are_unique_and_deterministically_sorted(corpus_run: ProducerRun) -> None:
    _, rows = _qc_rows(corpus_run)
    metric_rank = {metric_id: index for index, metric_id in enumerate(_METRIC_IDS)}
    keys = [
        (
            row["video"],
            _as_int(row["person_idx"]),
            _as_float(row["window_start_sec"]),
            _as_float(row["window_end_sec"]),
            row["metric_id"],
        )
        for row in rows
    ]
    assert len(keys) == len(set(keys))
    expected = sorted(
        keys,
        key=lambda key: (key[0], key[1], key[2], metric_rank[key[4]]),
    )
    assert keys == expected


def test_source_group_and_required_keypoint_vocabularies(corpus_run: ProducerRun) -> None:
    _, rows = _qc_rows(corpus_run)
    assert {row["source_group"] for row in rows} == set(_SOURCE_KEYPOINTS)
    for row in rows:
        source_group = _source_group(row["metric_id"])
        assert row["source_group"] == source_group
        assert row["required_keypoints"] == ",".join(_SOURCE_KEYPOINTS[source_group])


def test_source_group_siblings_have_identical_evidence(
    corpus_run: ProducerRun, invalid_run: ProducerRun
) -> None:
    for run in (corpus_run, invalid_run):
        _, rows = _qc_rows(run)
        groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (
                row["video"],
                row["person_idx"],
                row["window_start_sec"],
                row["window_end_sec"],
                row["source_group"],
            )
            groups.setdefault(key, []).append(row)
        for siblings in groups.values():
            expected = tuple(siblings[0][field] for field in _EVIDENCE_FIELDS)
            assert all(
                tuple(row[field] for field in _EVIDENCE_FIELDS) == expected for row in siblings[1:]
            )


def _normalized_3d_digest(path: pathlib.Path) -> str:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [
            field
            for field in (reader.fieldnames or ())
            if field not in {"producer_version", "qc_policy_version"}
        ]
        rows = [{field: row[field] for field in fields} for row in reader]
    payload = json.dumps([fields, rows], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_existing_3d_estimate_artifacts_are_unchanged(
    regenerated_goldens: pathlib.Path,
) -> None:
    assert (regenerated_goldens / f"world3d{_QC_SUFFIX}").exists()
    actual = {
        name: _normalized_3d_digest(regenerated_goldens / name)
        for name in _BASELINE_3D_NORMALIZED_SHA256
    }
    assert actual == _BASELINE_3D_NORMALIZED_SHA256


def test_2d_goldens_are_frozen_and_emit_no_qc(
    regenerated_goldens: pathlib.Path,
) -> None:
    assert (regenerated_goldens / f"world3d{_QC_SUFFIX}").exists()
    actual = {
        name: hashlib.sha256((regenerated_goldens / name).read_bytes()).hexdigest()
        for name in _BASELINE_2D_SHA256
    }
    assert actual == _BASELINE_2D_SHA256


def test_2d_inputs_write_no_qc_artifact(tmp_path: pathlib.Path) -> None:
    """The 2D partition is read where the producer writes, not where goldens land.

    regenerate() lifts a filename whitelist out of a staging directory it then
    deletes, so an unexpected 2D artifact could never reach the golden
    directory to be missed there. Run the producer in place instead and list
    the directory it actually wrote.
    """
    generator = _load_generator()
    for stem in ("2d_idx", "2d_cumsum", "2d_csv4dp"):
        source = tmp_path / f"{stem}.csv"
        generator._write_2d_input(source, stem.removeprefix("2d_"))
        generator._run_clinical(source)

    assert not list(tmp_path.glob(f"*{_QC_SUFFIX}"))
    assert not list(tmp_path.glob("*_window_qc.csv"))
    # Positive control: an empty glob has to mean the producer ran and declined.
    assert len(list(tmp_path.glob("*_clinical_windows.csv"))) == 3


def test_absent_rows_still_count_as_expected_frames(tmp_path: pathlib.Path) -> None:
    """Denominators come from the nominal grid, never from the observed rows.

    Every other fixture injects loss through the fusion diagnostics, which
    leaves one CSV row per nominal slot. Here the rows are physically absent,
    so a count taken from the window's row count would read full coverage.
    """
    source = tmp_path / "dropped_rows.csv"
    header, rows = _prepare_case_input(source, CorpusCase("dropped_rows", 30, 91))
    absent = {44, 45, 46}
    _write_csv(source, header, [row for index, row in enumerate(rows) if index not in absent])

    _, qc_rows = _qc_rows(_run_producer(source))
    row = _row(qc_rows, "dropped_rows", "left_wrist_velocity_mean", window_start=1.0)

    assert _as_int(row["n_expected_frames"]) == 30
    assert _as_int(row["n_valid_frames"]) == 27
    assert _as_float(row["frame_coverage"]) == pytest.approx(27 / 30)
    assert _as_int(row["n_expected_intervals"]) == 29
    assert _as_int(row["n_valid_intervals"]) == 25
    assert _as_int(row["longest_gap_frames"]) == 3
    assert _as_int(row["n_gaps"]) == 1
    assert row["qc_status"] == "pass"


@pytest.mark.parametrize(
    ("absent", "valid_frames", "valid_intervals", "longest_gap", "n_gaps", "reason"),
    [
        ({30}, 29, 28, 1, 1, "none"),
        ({59}, 29, 28, 1, 1, "none"),
        ({30, 59}, 28, 27, 1, 2, "none"),
        (set(range(30, 37)), 23, 22, 7, 1, "gap_too_long"),
    ],
    ids=("first-slot", "last-slot", "both-edge-slots", "seven-leading-slots"),
)
def test_window_edge_absence_keeps_the_nominal_slot_count(
    tmp_path: pathlib.Path,
    absent: set[int],
    valid_frames: int,
    valid_intervals: int,
    longest_gap: int,
    n_gaps: int,
    reason: str,
) -> None:
    """Loss at a window edge is measured, not anchored away.

    The kernel's grid starts at the first observed sample, so a row absent at
    an edge falls outside it and its slot would never be counted. Interior
    loss is exact either way, which is why only an edge case shows it.
    """
    source = tmp_path / "window_edge_loss.csv"
    header, rows = _prepare_case_input(source, CorpusCase("window_edge_loss", 30, 91))
    _write_csv(source, header, [row for index, row in enumerate(rows) if index not in absent])

    _, qc_rows = _qc_rows(_run_producer(source))
    row = _row(qc_rows, "window_edge_loss", "left_wrist_velocity_mean", window_start=1.0)

    assert _as_int(row["n_expected_frames"]) == 30
    assert _as_int(row["n_valid_frames"]) == valid_frames
    assert _as_float(row["frame_coverage"]) == pytest.approx(valid_frames / 30)
    assert _as_int(row["n_expected_intervals"]) == 29
    assert _as_int(row["n_valid_intervals"]) == valid_intervals
    assert _as_int(row["longest_gap_frames"]) == longest_gap
    assert _as_int(row["n_gaps"]) == n_gaps
    assert row["qc_reason"] == reason
    assert row["qc_status"] == ("pass" if reason == "none" else "fail")


def test_gap_threshold_tolerance_is_load_bearing(corpus_run: ProducerRun) -> None:
    """A gap on the 0.10 s boundary passes, and needs the relative tolerance.

    At 100 Hz a ten-frame hole divides out to 0.10000000000000009, which a
    bare comparison rejects. R-12's tolerance is what keeps a boundary case a
    boundary case rather than a representation artifact.
    """
    _, rows = _qc_rows(corpus_run)
    boundary = _row(rows, "gap10_100", "left_wrist_velocity_mean", window_start=0.0)
    measured = _as_float(boundary["longest_gap_sec"])
    assert measured > _MAX_GAP_SEC
    assert measured <= _MAX_GAP_SEC * (1 + _POLICY_TOLERANCE)
    assert boundary["qc_reason"] != "gap_too_long"
    beyond = _row(rows, "gap11_100", "left_wrist_velocity_mean", window_start=0.0)
    assert beyond["qc_reason"] == "gap_too_long"


def test_coverage_tolerance_is_pinned_at_the_policy_boundary() -> None:
    """No input can reach the coverage band, so the rule is probed directly.

    Coverage is k/n over the window's nominal slots, and the widest ratio
    below 0.80 that the tolerance still admits needs n >= 250 million slots,
    which is 96 days at 30 Hz. The band is real arithmetic all the same, and
    a bare comparison here would change what a caller is told.
    """
    result = _run_r(
        """
        result <- list(
          inside  = qc_reason_for(TRUE, 30L, 29L, 0, 0.80 * (1 - 5e-10), 1),
          outside = qc_reason_for(TRUE, 30L, 29L, 0, 0.80 * (1 - 5e-9), 1)
        )
        """
    )
    assert result["inside"] == "none"
    assert result["outside"] == "insufficient_coverage"


def test_qc_artifact_is_byte_identical_on_a_second_run(corpus_run: ProducerRun) -> None:
    """Rerunning over one unchanged input reproduces the artifact exactly.

    Numeric assertions read a parsed value, so they accept a moved decimal or
    a rewritten lexeme. Byte equality across two R processes is what covers
    serialization drift and an output that appends rather than replaces.
    """
    first = corpus_run.qc_path.read_bytes()
    _assert_run_succeeded(_run_producer(corpus_run.source))
    assert corpus_run.qc_path.read_bytes() == first


def test_docs_define_the_qc_contract_and_limitations() -> None:
    text = _ANALYSIS_DOC.read_text()
    for token in (
        "_clinical_3d_window_qc.csv",
        *_QC_COLUMNS,
        "invalid_timebase",
        "missing_required_keypoints",
        "insufficient_observations",
        "gap_too_long",
        "insufficient_coverage",
        "estimator_undefined",
        "engineering-provisional",
        "consecutive missing nominal slots",
        "1e-9",
        "reprojection",
        "cheirality",
        "confidence",
    ):
        assert token in text
    reason_positions = [
        text.index(reason)
        for reason in (
            "invalid_timebase",
            "missing_required_keypoints",
            "insufficient_observations",
            "gap_too_long",
            "insufficient_coverage",
            "estimator_undefined",
        )
    ]
    assert reason_positions == sorted(reason_positions)
