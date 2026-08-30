"""M2.4 red suite — timebase truth.

Diff-blind. Encodes the phase-1 verdict table of `.scratch/agents/test-m2u4.md`
under MAIN's rulings A01-A28 (`.agent/archive/contract-m2u4.md` §8). One test
per case id; the row's own `your reading` binds except where an amendment
overrides it.

A26 splits the suite: a `red` case fails at baseline `6bbd50e` and passes after
adoption; a `control` case passes at baseline and must keep passing. Each case
declares its kind in a `# kind:` line so `scripts/check_m2u4_suite_seed.py`
reads the expectation without running the suite.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import pytest

from test_r_clinical_goldens import _load_generator
from test_r_pipeline import _r_available
from test_r_qc_evidence import (
    CorpusCase,
    ProducerRun,
    _oracle,
    _prepare_case_input,
    _qc_rows,
    _read_csv,
    _run_producer,
    _write_csv,
)
from test_r_trajectory_kernel import _legacy_metrics_r, _run_r

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
_ANALYSIS_DOC = _PROJECT_ROOT / "docs" / "technical" / "analysis.md"
_QC_ORACLE = _PROJECT_ROOT / "tests" / "test_r_qc_evidence.py"
_GOLDEN_DIR = _PROJECT_ROOT / "tests" / "goldens" / "r_clinical"
_GRID_PROBE = _PROJECT_ROOT / "scripts" / "probe_timebase_grid.py"
_GRID_RESULTS = _PROJECT_ROOT / "tests" / "timebase_grid_results.json"
_BASELINE = "6bbd50e"
_RATES = (30.0, 29.97, 59.94, 60.0, 100.0, 119.88)
_VERSION_TAGS = ("producer_version", "metric_method_version", "qc_policy_version")
_FS_METRICS = tuple(
    [
        f"{side}_{metric}"
        for side in ("left", "right")
        for metric in (
            "wrist_sal",
            "wrist_velocity_mean",
            "wrist_velocity_peak",
            "wrist_normalized_jerk",
            "fingertip_normalized_jerk",
        )
    ]
    + [
        f"{metric}_{derivative}"
        for metric in (
            "wrist_sal",
            "wrist_velocity_mean",
            "wrist_velocity_peak",
            "wrist_normalized_jerk",
            "fingertip_normalized_jerk",
        )
        for derivative in ("symmetry_ratio", "dominance_index", "abs_diff")
    ]
)
_EXPECTED_GOLDENS = {
    "2d_idx_clinical.csv",
    "2d_idx_clinical_windows.csv",
    "2d_cumsum_clinical.csv",
    "2d_cumsum_clinical_windows.csv",
    "2d_csv4dp_clinical.csv",
    "2d_csv4dp_clinical_windows.csv",
    "world3d_clinical_3d.csv",
    "world3d_clinical_3d_windows.csv",
    "world3d_clinical_3d_window_qc.csv",
}

pytestmark = pytest.mark.skipif(not _r_available(), reason="R or required packages unavailable")


def _r_number(value: float) -> str:
    if math.isnan(value):
        return "NA_real_"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return repr(value)


def _r_fs(timestamps: list[float], *, magnitude: bool | None = None) -> float:
    values = ", ".join(_r_number(value) for value in timestamps)
    mode = "" if magnitude is None else f", magnitude={'TRUE' if magnitude else 'FALSE'}"
    result = _run_r(f"result <- list(fs=nominal_fs(c({values}){mode}))")
    value = result["fs"]
    return math.nan if value == "NA" else float(value)


def _rounded(fps: float, n: int, *, origin: float = 0.0) -> list[float]:
    return [round(origin + index / fps, 4) for index in range(n)]


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / expected


def _assert_cadence_accuracy(fps: float) -> None:
    timestamps = _rounded(fps, math.ceil(10 * fps) + 1)
    actual = _r_fs(timestamps)
    legacy = 1 / statistics.median(right - left for left, right in itertools.pairwise(timestamps))
    error = _relative_error(actual, fps)
    legacy_error = _relative_error(legacy, fps)
    assert error <= 1e-4
    assert error <= legacy_error * (1 + 1e-12)


def _assert_gapped_cadence(fps: float, omitted: set[int]) -> None:
    full = _rounded(fps, math.ceil(10 * fps) + 1)
    timestamps = [value for index, value in enumerate(full) if index not in omitted]
    actual = _r_fs(timestamps)
    legacy = 1 / statistics.median(right - left for left, right in itertools.pairwise(timestamps))
    error = _relative_error(actual, fps)
    assert error <= 1e-4
    assert error <= _relative_error(legacy, fps) * (1 + 1e-12)


def _assert_quantization_bound(timestamps: list[float], fps: float) -> None:
    actual = _r_fs(timestamps)
    span = abs(timestamps[-1] - timestamps[0])
    bound = 1e-4 / span
    assert _relative_error(actual, fps) <= math.nextafter(bound, math.inf)


def _oracle_rate(timestamps: list[float]) -> float:
    start = timestamps[0]
    end = timestamps[-1] + 1e-6
    input_rows = []
    for index, timestamp in enumerate(timestamps):
        input_rows.append(
            {
                "video": "oracle",
                "person_idx": "0",
                "timestamp_sec": repr(timestamp),
                "body_left_wrist_x_m": repr(index / 100),
                "body_left_wrist_y_m": "0",
                "body_left_wrist_z_m": "0",
                "body_left_wrist_reproj_err_px": "0.5",
                "body_left_wrist_cheirality_ok": "1",
                "body_left_wrist_triangulation_angle_deg": "10",
            }
        )
    qc_row = {
        "video": "oracle",
        "person_idx": "0",
        "window_start_sec": repr(start),
        "window_end_sec": repr(end),
        "source_group": "left_wrist",
        "metric_id": "left_wrist_velocity_mean",
    }
    window_rows = [
        {
            "video": "oracle",
            "person_idx": "0",
            "window_start_sec": repr(start),
            "window_end_sec": repr(end),
            "left_wrist_velocity_mean": "1",
        }
    ]
    evidence = _oracle(input_rows, window_rows, qc_row)
    return evidence.n_valid_intervals / evidence.valid_duration_sec


def _source_without_comments(path: pathlib.Path) -> str:
    return "\n".join(line.split("#", 1)[0] for line in path.read_text().splitlines())


def _baseline_bytes(filename: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{_BASELINE}:tests/goldens/r_clinical/{filename}"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _csv_bytes(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(payload.decode()))
    assert reader.fieldnames is not None
    return list(reader.fieldnames), list(reader)


def _changed_columns(
    before_rows: list[dict[str, str]],
    after_rows: list[dict[str, str]],
    columns: list[str] | tuple[str, ...],
) -> set[str]:
    assert len(after_rows) == len(before_rows)
    return {
        column
        for column in columns
        if any(
            before[column] != after[column]
            for before, after in zip(before_rows, after_rows, strict=True)
        )
    }


@pytest.fixture(scope="module")
def golden_outputs(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    output = tmp_path_factory.mktemp("m2u4-goldens")
    _load_generator().regenerate(output)
    return output


@dataclass(frozen=True)
class _ProducerCorpus:
    run: ProducerRun
    person_idx: dict[str, int]
    qc_header: list[str]
    qc_rows: list[dict[str, str]]


def _spread_indices(n_slots: int, count: int) -> tuple[int, ...]:
    return tuple(int((index + 0.5) * n_slots / count) for index in range(count))


@pytest.fixture(scope="module")
def producer_corpus(tmp_path_factory: pytest.TempPathFactory) -> _ProducerCorpus:
    root = tmp_path_factory.mktemp("m2u4-producer")
    specs: list[tuple[str, int, int, tuple[int, ...]]] = [
        ("gap10_100", 100, 101, tuple(range(40, 50))),
        ("gap11_100", 100, 101, tuple(range(40, 51))),
        *((f"n{n}", 30, n, (12, 13, 14)) for n in range(58, 64)),
        ("descending30", 30, 31, ()),
        ("reversal30", 30, 31, ()),
        ("duplicate30", 30, 31, ()),
        ("identical30", 30, 31, ()),
        ("nonfinite30", 30, 31, ()),
        ("short30", 30, 30, (12, 13, 14)),
        ("short60", 60, 31, tuple(range(12, 18))),
        ("gap4_30", 30, 61, tuple(range(12, 16))),
        ("coverage_low", 100, 101, _spread_indices(100, 21)),
        ("complete30", 30, 61, ()),
    ]
    person_idx = {name: index for index, (name, *_rest) in enumerate(specs)}
    header: list[str] | None = None
    all_rows: list[dict[str, str]] = []
    for name, fps, n_frames, failed_indices in specs:
        failed_keypoints = ("body_left_wrist",) if failed_indices else ()
        path = root / f"{name}.csv"
        case_header, rows = _prepare_case_input(
            path,
            CorpusCase(name, fps, n_frames, failed_keypoints, failed_indices),
        )
        if name == "descending30":
            values = [row["timestamp_sec"] for row in rows]
            for row, timestamp in zip(rows, reversed(values), strict=True):
                row["timestamp_sec"] = timestamp
        elif name == "reversal30":
            rows[10]["timestamp_sec"], rows[11]["timestamp_sec"] = (
                rows[11]["timestamp_sec"],
                rows[10]["timestamp_sec"],
            )
        elif name == "duplicate30":
            rows[10]["timestamp_sec"] = rows[9]["timestamp_sec"]
        elif name == "identical30":
            for row in rows:
                row["timestamp_sec"] = "0"
        elif name == "nonfinite30":
            rows[10]["timestamp_sec"] = "NA"
        if header is None:
            header = case_header
        else:
            assert case_header == header
        for row in rows:
            row["video"] = "suite"
            row["person_idx"] = str(person_idx[name])
        all_rows.extend(rows)
    assert header is not None
    source = root / "suite.csv"
    _write_csv(source, header, all_rows)
    run = _run_producer(source)
    qc_header, qc_rows = _qc_rows(run)
    return _ProducerCorpus(run, person_idx, qc_header, qc_rows)


def _case_rows(corpus: _ProducerCorpus, name: str) -> list[dict[str, str]]:
    person_idx = corpus.person_idx[name]
    return [row for row in corpus.qc_rows if int(float(row["person_idx"])) == person_idx]


def _case_row(
    corpus: _ProducerCorpus,
    name: str,
    *,
    metric_id: str = "left_wrist_velocity_mean",
    window_start: float = 0.0,
) -> dict[str, str]:
    rows = [
        row
        for row in _case_rows(corpus, name)
        if row["metric_id"] == metric_id
        and float(row["window_start_sec"]) == pytest.approx(window_start, abs=1e-12)
    ]
    assert rows, f"missing QC row for {name}/{metric_id}/{window_start}"
    return rows[0]


@pytest.fixture(scope="module")
def policy_results() -> dict[str, Any]:
    return _run_r(
        """
        reason <- function(fs, missing, coverage=1) {
          qc_reason_for(TRUE, 100L, 99L, missing / fs, coverage, 1)
        }
        fs30_n60 <- nominal_fs(round((0:59) / 30, 4))
        fs60_n59 <- nominal_fs(round((0:58) / 60, 4))
        result <- list(
          c5_01=reason(30, 2), c5_02=reason(fs30_n60, 3), c5_03=reason(30, 4),
          c5_04=reason(29.97, 2), c5_05=reason(29.97, 3),
          c5_06=reason(59.94, 5), c5_07=reason(59.94, 6),
          c5_08=reason(60, 5), c5_09=reason(fs60_n59, 6), c5_10=reason(60, 7),
          c5_11=reason(100, 9), c5_12=reason(100, 10), c5_13=reason(100, 11),
          c5_14=reason(119.88, 11), c5_15=reason(119.88, 12),
          equal=vapply(c(30,29.97,59.94,60,100,119.88), function(fs) {
            qc_reason_for(TRUE, 100L, 99L, QC_MAX_GAP_SEC, 1, 1)
          }, character(1)),
          policy_tolerance=QC_POLICY_TOLERANCE,
          margin=(1 / 120) / (QC_MAX_GAP_SEC * QC_POLICY_TOLERANCE),
          c5_20=qc_reason_for(TRUE, 100L, 99L, 0, 0.80 * (1 - 5e-5), 1)
        )
        """
    )


def _grid_data() -> dict[str, Any]:
    assert _GRID_RESULTS.is_file(), (
        "real-corpus evidence missing: run scripts/probe_timebase_grid.py"
    )
    return json.loads(_GRID_RESULTS.read_text())


def _assert_grid_asset(asset: dict[str, Any], bounds: dict[str, float]) -> None:
    # A31: only these two clauses bind on real assets.  `nominal_fs_rel_err` is
    # measured against the container header, and the header is not truth -- it
    # divides by a duration counting the terminal frame that the timestamp span
    # omits, so a VFR clip disagrees by `(terminal_dur - mean_interval) / span`
    # with no defect anywhere.  Header agreement publishes as an outlier count,
    # checked once over the whole artifact in C8.01, never per asset.
    assert asset["duration_sec"] >= bounds["min_cadence_span_sec"]
    assert asset["nominal_fs_rel_err"] <= asset["median_diff_rel_err"] * (1 + 1e-12)
    assert asset["grid_residual_max_nominal"] <= bounds["grid_slot_tolerance"]


# ---------------------------------------------------------------------------
# Class 1 — exactly-representable cadence (100 Hz) — the blind spot
# ---------------------------------------------------------------------------
# kind: C1.01 = red
def test_c1_01():
    result = _run_r(
        """
        t <- round((0:200) / 100, 4)
        result <- list(default=nominal_fs(t), magnitude=nominal_fs(t, magnitude=TRUE))
        """
    )
    assert float(result["default"]) == 100
    assert float(result["magnitude"]) == 100


# kind: C1.02 = control
def test_c1_02():
    _assert_cadence_accuracy(100)


# kind: C1.03 = control
def test_c1_03(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "gap10_100")
    assert row["qc_status"] == "pass"
    assert row["qc_reason"] == "none"


# kind: C1.04 = control
def test_c1_04(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "gap11_100")
    assert row["qc_status"] == "fail"
    assert row["qc_reason"] == "gap_too_long"


# kind: C1.05 = control
def test_c1_05():
    result = _run_r(
        """
        fs <- 100
        idx <- 0:200
        t <- idx / fs
        tau <- idx / 200
        s <- 10 * tau^3 - 15 * tau^4 + 6 * tau^5
        x <- 0.40 * s
        y <- 0.05 * sin(pi * tau)
        z <- rep(0, length(t))
        """
        + _legacy_metrics_r()
        + """
        actual <- trajectory_metrics(t, x, y, z, fs=fs)
        result <- list(
          identical=unname(vapply(names(legacy), function(name) {
            identical(actual[[name]], legacy[[name]])
          }, logical(1)))
        )
        """
    )
    assert result["identical"] == [True] * 5


# ---------------------------------------------------------------------------
# Class 2 — non-representable cadence (30, 29.97, 59.94, 60, 119.88 Hz)
# ---------------------------------------------------------------------------
# kind: C2.01 = control
def test_c2_01():
    _assert_cadence_accuracy(30)


# kind: C2.02 = control
def test_c2_02():
    _assert_cadence_accuracy(29.97)


# kind: C2.03 = control
def test_c2_03():
    _assert_cadence_accuracy(59.94)


# kind: C2.04 = control
def test_c2_04():
    _assert_cadence_accuracy(60)


# kind: C2.05 = control
def test_c2_05():
    _assert_cadence_accuracy(119.88)


# kind: C2.06 = red
def test_c2_06():
    source = _source_without_comments(_CLINICAL_R)
    assert re.search(
        r"fs\s*<-\s*nominal_fs\(\s*ts\s*,\s*magnitude\s*=\s*TRUE\s*\)",
        source,
    )
    assert not re.search(
        r"1\s*/\s*median\s*\(\s*(?:abs\s*\(\s*)?diff\s*\(",
        source,
    )


# kind: C2.07 = red
def test_c2_07():
    source = _source_without_comments(_CLINICAL_R)
    segmentation = source[source.index("segment_movements <- function") :]
    assert re.search(
        r"fs\s*<-\s*nominal_fs\(\s*ts\s*,\s*magnitude\s*=\s*FALSE\s*\)",
        segmentation,
    )


# kind: C2.08 = red
def test_c2_08():
    for fps in _RATES:
        timestamps = _rounded(fps, math.ceil(10 * fps) + 1)
        actual = _oracle_rate(timestamps)
        assert _relative_error(actual, fps) <= 1e-4


# kind: C2.09 = red
def test_c2_09():
    result = _run_r(
        """
        result <- list(
          producer=PRODUCER_VERSION,
          metric=METRIC_METHOD_VERSION,
          qc=QC_POLICY_VERSION
        )
        """
    )
    assert result == {"producer": "v3", "metric": "v2", "qc": "v3"}


# kind: C2.10 = control
def test_c2_10(golden_outputs: pathlib.Path):
    for filename in (
        "2d_idx_clinical.csv",
        "2d_cumsum_clinical.csv",
        "2d_csv4dp_clinical.csv",
    ):
        assert (golden_outputs / filename).read_bytes() == _baseline_bytes(filename)


# kind: C2.11 = red
def test_c2_11(golden_outputs: pathlib.Path):
    filename = "world3d_clinical_3d.csv"
    before_fields, before_rows = _csv_bytes(_baseline_bytes(filename))
    after_fields, after_rows = _read_csv(golden_outputs / filename)
    assert after_fields == before_fields
    stable = [field for field in before_fields if field not in _VERSION_TAGS]
    assert _changed_columns(before_rows, after_rows, stable) == set()
    assert len(after_rows) == len(before_rows)
    for row in after_rows:
        assert tuple(row[tag] for tag in _VERSION_TAGS) == ("v3", "v2", "v3")


# kind: C2.12 = red
def test_c2_12(golden_outputs: pathlib.Path):
    changed_union: set[str] = set()
    for filename in (
        "2d_idx_clinical_windows.csv",
        "2d_cumsum_clinical_windows.csv",
        "2d_csv4dp_clinical_windows.csv",
    ):
        before_fields, before_rows = _csv_bytes(_baseline_bytes(filename))
        after_fields, after_rows = _read_csv(golden_outputs / filename)
        assert after_fields == before_fields
        changed = _changed_columns(before_rows, after_rows, before_fields)
        assert changed <= set(_FS_METRICS)
        changed_union.update(changed)
    assert changed_union == set(_FS_METRICS)


# kind: C2.13 = red
def test_c2_13(golden_outputs: pathlib.Path):
    window_name = "world3d_clinical_3d_windows.csv"
    before_fields, before_rows = _csv_bytes(_baseline_bytes(window_name))
    after_fields, after_rows = _read_csv(golden_outputs / window_name)
    assert after_fields == before_fields
    window_changed = _changed_columns(before_rows, after_rows, before_fields)
    assert window_changed <= {*_FS_METRICS, *_VERSION_TAGS}
    assert set(_VERSION_TAGS) <= window_changed

    qc_name = "world3d_clinical_3d_window_qc.csv"
    before_fields, before_rows = _csv_bytes(_baseline_bytes(qc_name))
    after_fields, after_rows = _read_csv(golden_outputs / qc_name)
    added = {"qc_policy_tolerance", "qc_coverage_tolerance"}
    assert set(after_fields) - set(before_fields) == added
    assert set(before_fields) - set(after_fields) == set()
    policy_at = after_fields.index("min_coverage")
    assert after_fields[policy_at : policy_at + 4] == [
        "min_coverage",
        "max_gap_sec",
        "qc_policy_tolerance",
        "qc_coverage_tolerance",
    ]
    qc_changed = _changed_columns(before_rows, after_rows, before_fields)
    # A29: the fixture is a complete grid, so `longest_gap_frames` is 0 at any
    # cadence and `longest_gap_sec` moves in zero rows.  Requiring it to move
    # would demand a change the estimator swap cannot produce.
    assert qc_changed == {"valid_duration_sec", *_VERSION_TAGS}
    assert "longest_gap_sec" not in qc_changed
    for row in after_rows:
        assert float(row["qc_policy_tolerance"]) == 1e-4
        assert float(row["qc_coverage_tolerance"]) == 1e-9


# kind: C2.14 = control
def test_c2_14(tmp_path: pathlib.Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generator = _load_generator()
    generator.regenerate(first)
    generator.regenerate(second)
    assert {path.name for path in first.iterdir()} == _EXPECTED_GOLDENS
    assert {path.name for path in second.iterdir()} == _EXPECTED_GOLDENS
    for filename in _EXPECTED_GOLDENS:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


# kind: C2.15 = red
def test_c2_15():
    text = _ANALYSIS_DOC.read_text()
    assert "Known defect, not yet fixed" not in text
    for token in (
        "nominal_fs",
        "qc_policy_tolerance",
        "qc_coverage_tolerance",
        "1e-4",
        "1e-9",
    ):
        assert token in text


# ---------------------------------------------------------------------------
# Class 3 — clip-length sweep at 30 Hz over every residue of (n-1) mod 3
# ---------------------------------------------------------------------------
# kind: C3.01 = control
def test_c3_01(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "n58")
    assert row["qc_status"] == "pass"
    assert row["qc_reason"] == "none"


# kind: C3.02 = control
def test_c3_02(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "n59")
    assert row["qc_status"] == "pass"
    assert row["qc_reason"] == "none"


# kind: C3.03 = control
def test_c3_03(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "n60")
    assert row["qc_status"] == "pass"
    assert row["qc_reason"] == "none"


# kind: C3.04 = control
def test_c3_04(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "n61")
    assert row["qc_status"] == "pass"
    assert row["qc_reason"] == "none"


# kind: C3.05 = control
def test_c3_05(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "n62")
    assert row["qc_status"] == "pass"
    assert row["qc_reason"] == "none"


# kind: C3.06 = control
def test_c3_06(producer_corpus: _ProducerCorpus):
    row = _case_row(producer_corpus, "n63")
    assert row["qc_status"] == "pass"
    assert row["qc_reason"] == "none"


# ---------------------------------------------------------------------------
# Class 4 — descending, non-monotonic, duplicate and identical timestamps
# ---------------------------------------------------------------------------
# kind: C4.01 = control
def test_c4_01():
    actual = _r_fs(list(reversed(_rounded(30, 31))))
    assert math.isnan(actual)


# kind: C4.02 = red
def test_c4_02():
    actual = _r_fs(list(reversed(_rounded(30, 31))), magnitude=True)
    assert _relative_error(actual, 30) <= 1e-4


# kind: C4.03 = control
def test_c4_03(producer_corpus: _ProducerCorpus):
    rows = _case_rows(producer_corpus, "descending30")
    evidence = (
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
    assert rows
    for row in rows:
        assert row["qc_status"] == "fail"
        assert row["qc_reason"] == "invalid_timebase"
        assert all(row[field] == "NA" for field in evidence)
        assert all(
            row[field]
            for field in (
                "metric_id",
                "source_group",
                "required_keypoints",
                "producer_version",
                "metric_method_version",
                "qc_policy_version",
            )
        )


# kind: C4.04 = control
def test_c4_04(producer_corpus: _ProducerCorpus):
    rows = _case_rows(producer_corpus, "reversal30")
    assert rows
    assert {row["qc_reason"] for row in rows} == {"invalid_timebase"}
    assert {row["qc_status"] for row in rows} == {"fail"}


# kind: C4.05 = control
def test_c4_05(producer_corpus: _ProducerCorpus):
    rows = _case_rows(producer_corpus, "duplicate30")
    assert rows
    assert {row["qc_reason"] for row in rows} == {"invalid_timebase"}
    assert {row["qc_status"] for row in rows} == {"fail"}


# kind: C4.06 = control
def test_c4_06(producer_corpus: _ProducerCorpus):
    assert _case_rows(producer_corpus, "identical30") == []


# kind: C4.07 = red
def test_c4_07():
    assert math.isnan(_r_fs([0.0, 0.0333]))


# kind: C4.08 = control
def test_c4_08():
    actual = _r_fs([0.0, 0.0333, 0.0667])
    assert math.isfinite(actual)
    assert actual > 0


# kind: C4.09 = red
def test_c4_09(producer_corpus: _ProducerCorpus):
    # A33: interior NaN never reaches the timebase verdict.  Window enumeration
    # drops NA before the grid check, so the window keys cleanly and the failure
    # surfaces per metric, on whichever trajectory lost its own support.
    actual = _r_fs([0.0, 0.0333, math.nan, 0.1, 0.1333])
    assert math.isfinite(actual)
    rows = _case_rows(producer_corpus, "nonfinite30")
    assert rows
    reasons = {row["qc_reason"] for row in rows}
    assert reasons == {"none", "estimator_undefined"}
    assert "invalid_timebase" not in reasons
    assert all((row["qc_status"] == "pass") == (row["qc_reason"] == "none") for row in rows)


# kind: C4.10 = control
def test_c4_10():
    actual = _r_fs([0.0, 0.0333, 0.0333, 0.02, 0.0533, 0.0533])
    assert math.isfinite(actual)
    assert actual > 0


# ---------------------------------------------------------------------------
# Class 5 — gaps at, below and above the 0.10 s boundary at each cadence
# ---------------------------------------------------------------------------
# kind: C5.01 = control
def test_c5_01(policy_results: dict[str, Any]):
    assert policy_results["c5_01"] == "none"


# kind: C5.02 = red
def test_c5_02(policy_results: dict[str, Any]):
    assert policy_results["c5_02"] == "none"


# kind: C5.03 = control
def test_c5_03(policy_results: dict[str, Any]):
    assert policy_results["c5_03"] == "gap_too_long"


# kind: C5.04 = control
def test_c5_04(policy_results: dict[str, Any]):
    assert policy_results["c5_04"] == "none"


# kind: C5.05 = control
def test_c5_05(policy_results: dict[str, Any]):
    assert policy_results["c5_05"] == "gap_too_long"


# kind: C5.06 = control
def test_c5_06(policy_results: dict[str, Any]):
    assert policy_results["c5_06"] == "none"


# kind: C5.07 = control
def test_c5_07(policy_results: dict[str, Any]):
    assert policy_results["c5_07"] == "gap_too_long"


# kind: C5.08 = control
def test_c5_08(policy_results: dict[str, Any]):
    assert policy_results["c5_08"] == "none"


# kind: C5.09 = red
def test_c5_09(policy_results: dict[str, Any]):
    assert policy_results["c5_09"] == "none"


# kind: C5.10 = control
def test_c5_10(policy_results: dict[str, Any]):
    assert policy_results["c5_10"] == "gap_too_long"


# kind: C5.11 = control
def test_c5_11(policy_results: dict[str, Any]):
    assert policy_results["c5_11"] == "none"


# kind: C5.12 = control
def test_c5_12(policy_results: dict[str, Any]):
    assert policy_results["c5_12"] == "none"


# kind: C5.13 = control
def test_c5_13(policy_results: dict[str, Any]):
    assert policy_results["c5_13"] == "gap_too_long"


# kind: C5.14 = control
def test_c5_14(policy_results: dict[str, Any]):
    assert policy_results["c5_14"] == "none"


# kind: C5.15 = control
def test_c5_15(policy_results: dict[str, Any]):
    assert policy_results["c5_15"] == "gap_too_long"


# kind: C5.16 = control
def test_c5_16(policy_results: dict[str, Any]):
    assert policy_results["equal"] == ["none"] * len(_RATES)


# kind: C5.17 = red
def test_c5_17(producer_corpus: _ProducerCorpus):
    expected = {"qc_policy_tolerance": 1e-4, "qc_coverage_tolerance": 1e-9}
    assert set(expected) <= set(producer_corpus.qc_header)
    at = producer_corpus.qc_header.index("min_coverage")
    assert producer_corpus.qc_header[at : at + 4] == [
        "min_coverage",
        "max_gap_sec",
        "qc_policy_tolerance",
        "qc_coverage_tolerance",
    ]
    assert producer_corpus.qc_rows
    for row in producer_corpus.qc_rows:
        for column, value in expected.items():
            assert float(row[column]) == value


# kind: C5.18 = red
def test_c5_18(producer_corpus: _ProducerCorpus):
    for name, expected in (
        ("n58", "none"),
        ("gap4_30", "gap_too_long"),
        ("coverage_low", "insufficient_coverage"),
    ):
        row = _case_row(producer_corpus, name)
        gap_limit = float(row["max_gap_sec"]) * (1 + float(row["qc_policy_tolerance"]))
        coverage_limit = float(row["min_coverage"]) * (1 - float(row["qc_coverage_tolerance"]))
        if float(row["longest_gap_sec"]) > gap_limit:
            recomputed = "gap_too_long"
        elif float(row["frame_coverage"]) < coverage_limit:
            recomputed = "insufficient_coverage"
        else:
            recomputed = "none"
        assert recomputed == expected == row["qc_reason"]
        assert row["qc_status"] == ("pass" if expected == "none" else "fail")


# kind: C5.19 = red
def test_c5_19(policy_results: dict[str, Any]):
    assert float(policy_results["policy_tolerance"]) == 1e-4
    assert float(policy_results["margin"]) >= 100
    assert policy_results["c5_15"] == "gap_too_long"


# kind: C5.20 = control
def test_c5_20(policy_results: dict[str, Any]):
    assert policy_results["c5_20"] == "insufficient_coverage"


# ---------------------------------------------------------------------------
# Class 6 — gapped clips where the GAP_INTERVAL_FACTOR filter engages
# ---------------------------------------------------------------------------
# kind: C6.01 = control
def test_c6_01():
    _assert_gapped_cadence(30, {100})


# kind: C6.02 = control
def test_c6_02():
    _assert_gapped_cadence(30, set(range(100, 108)))


# kind: C6.03 = control
def test_c6_03():
    omitted = {*range(70, 73), *range(140, 148), 230}
    _assert_gapped_cadence(29.97, omitted)


# kind: C6.04 = control
def test_c6_04():
    actual = _r_fs([0.0, 0.0333, 0.0667, 0.3667])
    assert math.isfinite(actual)
    assert actual > 0


# kind: C6.05 = control
def test_c6_05():
    # A05 counts the two positive intervals before the large one is cut.
    actual = _r_fs([0.0, 0.0333, 1.0333])
    assert math.isfinite(actual)
    assert actual > 0


# kind: C6.06 = control
def test_c6_06():
    actual = _r_fs([0.0, 1.0, 2.0, 3.5])
    assert actual == pytest.approx(6 / 7, rel=1e-12, abs=1e-12)


# kind: C6.07 = red
def test_c6_07():
    timestamps = _rounded(29.97, 301)
    timestamps = [value for index, value in enumerate(timestamps) if index not in range(100, 109)]
    actual = _r_fs(list(reversed(timestamps)), magnitude=True)
    assert _relative_error(actual, 29.97) <= 1e-4


# kind: C6.08 = red
def test_c6_08():
    # A35: four of six intervals are 2-slot gaps, so the median is itself a gap,
    # the 1.5x filter retains every interval, and the mean blends to 18.0018 Hz.
    # `nominal_fs` does not fail closed here and is not the layer that should --
    # `trajectory_grid_status` adjudicates whether an estimate describes the
    # data, and rejects this blend at residual 0.401 against a 0.25 tolerance.
    # "Recover or fail closed" therefore binds on the pipeline, not the
    # estimator, and stays true if a future estimator recovers instead.
    full = _rounded(30, 11)
    timestamps = [full[index] for index in (0, 2, 4, 6, 8, 9, 10)]
    values = ", ".join(repr(value) for value in timestamps)
    result = _run_r(
        f"""
        t <- c({values})
        fs <- nominal_fs(t)
        status <- trajectory_grid_status(t, fs)
        result <- list(
            fs = fs, fault = if (is.null(status$fault)) "" else status$fault
        )
        """
    )
    fs = result["fs"]
    recovered = isinstance(fs, float) and math.isfinite(fs) and _relative_error(fs, 30) <= 1e-4
    assert recovered or result["fault"]


# kind: C6.09 = control
def test_c6_09():
    timestamps = _rounded(30, 301)
    timestamps[50] = math.nan
    timestamps[150] = math.inf
    timestamps = [value for index, value in enumerate(timestamps) if index not in range(200, 208)]
    actual = _r_fs(timestamps)
    assert _relative_error(actual, 30) <= 1e-4


# ---------------------------------------------------------------------------
# Class 7 — short clips (span < 1 s) where the estimator bound is loosest
# ---------------------------------------------------------------------------
# kind: C7.01 = control
def test_c7_01():
    result = _run_r(
        """
        rates <- c(30, 29.97, 59.94, 60, 100, 119.88)
        spans <- c(0, 0.25, 0.5, 0.999, 1, 2, 10)
        errors <- numeric(0)
        bounds <- numeric(0)
        for (fps in rates) {
          for (target in spans) {
            n <- if (target == 0) 3L else max(3L, floor(target * fps) + 1L)
            t <- round((0:(n - 1L)) / fps, 4)
            estimate <- nominal_fs(t)
            span <- abs(t[length(t)] - t[1])
            errors <- c(errors, abs(estimate - fps) / fps)
            bounds <- c(bounds, 1e-4 / span)
          }
        }
        result <- list(errors=unname(errors), bounds=unname(bounds))
        """
    )
    for error, bound in zip(result["errors"], result["bounds"], strict=True):
        assert float(error) <= math.nextafter(float(bound), math.inf)


# kind: C7.02 = control
def test_c7_02():
    timestamps = _rounded(30, 3)
    assert math.isfinite(_r_fs(timestamps))
    _assert_quantization_bound(timestamps, 30)


# kind: C7.03 = control
def test_c7_03():
    timestamps = _rounded(119.88, 3)
    assert math.isfinite(_r_fs(timestamps))
    _assert_quantization_bound(timestamps, 119.88)


# kind: C7.04 = red
def test_c7_04():
    result = _run_r(
        """
        t <- round((0:30) / 30, 4)
        t[31] <- 1.0001
        fs <- nominal_fs(t)
        result <- list(
          fs=fs,
          span=abs(t[length(t)] - t[1]),
          reason=qc_reason_for(TRUE, 30L, 29L, 3 / fs, 1, 1)
        )
        """
    )
    error = _relative_error(float(result["fs"]), 30)
    bound = 1e-4 / float(result["span"])
    assert 0.99 * bound <= error <= bound + math.ulp(1.0)
    assert result["reason"] == "none"


# kind: C7.05 = control
def test_c7_05():
    true_span = 0.99895
    fps = 30 / true_span
    timestamps = [round(index * true_span / 30, 4) for index in range(31)]
    assert abs((timestamps[-1] - timestamps[0]) - 0.999) <= 1e-12
    _assert_quantization_bound(timestamps, fps)


# kind: C7.06 = control
def test_c7_06(producer_corpus: _ProducerCorpus):
    assert _case_rows(producer_corpus, "short30") == []


# kind: C7.07 = control
def test_c7_07(producer_corpus: _ProducerCorpus):
    assert _case_rows(producer_corpus, "short60") == []


# kind: C7.08 = control
def test_c7_08():
    result = _run_r(
        """
        fps <- 119.88
        origins <- seq(0, 1e-4, length.out=11)
        errors <- numeric(0)
        bounds <- numeric(0)
        for (origin in origins) {
          t <- round(origin + (0:59) / fps, 4)
          estimate <- nominal_fs(t)
          span <- abs(t[length(t)] - t[1])
          errors <- c(errors, abs(estimate - fps) / fps)
          bounds <- c(bounds, 1e-4 / span)
        }
        result <- list(errors=unname(errors), bounds=unname(bounds))
        """
    )
    for error, bound in zip(result["errors"], result["bounds"], strict=True):
        assert float(error) <= math.nextafter(float(bound), math.inf)


# kind: C7.09 = red
def test_c7_09():
    result = _run_r(
        """
        rates <- c(30, 29.97, 59.94, 60, 100, 119.88)
        one <- vapply(rates, function(fps) nominal_fs(round((0:1) / fps, 4)), numeric(1))
        two <- vapply(rates, function(fps) nominal_fs(round((0:2) / fps, 4)), numeric(1))
        result <- list(one=unname(one), two=unname(two))
        """
    )
    assert result["one"] == ["NA"] * len(_RATES)
    assert all(math.isfinite(float(value)) for value in result["two"])


# ---------------------------------------------------------------------------
# Class 8 — real-corpus decode timestamps
# ---------------------------------------------------------------------------
# kind: C8.01 = red
def test_c8_01():
    data = _grid_data()
    assert data["generator"] == "scripts/probe_timebase_grid.py"
    assert data["generator_version"] == "v1"
    source_digests = data["source_sha256"]
    assert (
        source_digests["scripts/probe_timebase_grid.py"]
        == hashlib.sha256(_GRID_PROBE.read_bytes()).hexdigest()
    )
    for relative_path, expected in source_digests.items():
        path = _PROJECT_ROOT / relative_path
        assert path.is_file(), "probe source unavailable"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert data["assets"]
    for asset in data["assets"]:
        _assert_grid_asset(asset, data["bounds"])

    # A31: header agreement is a reported cross-check, so what binds is that the
    # artifact's own reported outlier count and worst case reconcile with its
    # asset rows.  Asserting the bound itself would reject truthful VFR data.
    aggregate = data["aggregate"]
    bound = data["bounds"]["p06_rel_err_max"]
    outliers = [asset for asset in data["assets"] if asset["nominal_fs_rel_err"] > bound]
    assert aggregate["assets_header_outlier"] == len(outliers)
    worst = max((asset["nominal_fs_rel_err"] for asset in outliers), default=0.0)
    assert aggregate["header_outlier_worst_rel_err"] == pytest.approx(worst)
    assert aggregate["assets_nominal_no_worse_than_legacy"] == len(data["assets"])


# kind: C8.02 = red
def test_c8_02():
    data = _grid_data()
    ordinary = [asset for asset in data["assets"] if asset["header_fps"] < 100]
    asset = min(ordinary, key=lambda row: row["header_fps"])
    _assert_grid_asset(asset, data["bounds"])
    assert asset["header_fps"] < 29.97
    assert abs(asset["nominal_fs_hz"] - asset["header_fps"]) < abs(29.97 - asset["header_fps"])


# kind: C8.03 = red
def test_c8_03():
    data = _grid_data()
    ordinary = [asset for asset in data["assets"] if asset["header_fps"] < 100]
    asset = max(ordinary, key=lambda row: row["header_fps"])
    _assert_grid_asset(asset, data["bounds"])
    assert asset["header_fps"] > 29.97
    assert abs(asset["nominal_fs_hz"] - asset["header_fps"]) < abs(29.97 - asset["header_fps"])


# kind: C8.04 = red
def test_c8_04():
    data = _grid_data()
    high_rate = [asset for asset in data["assets"] if asset["header_fps"] > 100]
    assert high_rate
    asset = max(high_rate, key=lambda row: row["header_fps"])
    _assert_grid_asset(asset, data["bounds"])
    assert asset["grid_residual_max_nominal"] <= data["bounds"]["grid_slot_tolerance"]


# kind: C8.05 = red
def test_c8_05():
    data = _grid_data()
    asset = max(data["assets"], key=lambda row: row["median_diff_rel_err"])
    _assert_grid_asset(asset, data["bounds"])
    assert asset["nominal_fs_rel_err"] < asset["median_diff_rel_err"]
    assert asset["windows_on_grid_nominal"] >= asset["windows_on_grid_median_diff"]


# kind: C8.06 = red
def test_c8_06():
    data = _grid_data()
    tolerance = data["bounds"]["grid_slot_tolerance"]
    assert data["assets"]
    for asset in data["assets"]:
        assert asset["grid_residual_max_nominal"] <= tolerance
        assert asset["windows_on_grid_nominal"] == asset["windows_total"]


# kind: C8.07 = red
def test_c8_07():
    data = _grid_data()
    sample = data["sample"]
    assets = data["assets"]
    assert sample["n_assets"] == len(assets)
    assert sample["selection_rule"]
    assert sample["seed"] == 2404
    assert sample["strata"]
    assert [asset["asset_key"] for asset in assets] == [
        f"a{index:02d}" for index in range(len(assets))
    ]
    expected_fields = {
        "asset_key",
        "device_config",
        "codec",
        "rotation",
        "header_fps",
        "n_frames",
        "duration_sec",
        "nominal_fs_hz",
        "nominal_fs_rel_err",
        "median_diff_fs_hz",
        "median_diff_rel_err",
        "grid_residual_max_nominal",
        "grid_residual_max_median_diff",
        "windows_total",
        "windows_on_grid_nominal",
        "windows_on_grid_median_diff",
        "terminal_frame_duration_sec",
    }
    assert all(set(asset) == expected_fields for asset in assets)
    assert {asset["codec"] for asset in assets} >= {"h264", "hevc"}
    assert {asset["rotation"] for asset in assets} == {0, 90, 180, 270}
    assert any(asset["header_fps"] > 100 for asset in assets)
    assert any(asset["header_fps"] < 100 for asset in assets)
    for asset in assets:
        _assert_grid_asset(asset, data["bounds"])


# kind: C8.08 = red
def test_c8_08():
    environment = os.environ.copy()
    environment.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "-k",
            "not test_c8_08",
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, "decisive gate failed"
    summary = result.stdout + result.stderr
    # A32: a bare floor lets a case vanish silently, so the gate reconciles
    # instead.  `-k` deselects this case alone, which is why the expected pass
    # count is the collected total minus one.
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only"],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    total = re.search(r"(\d+) tests? collected", collected.stdout)
    assert total, "collection count unavailable"
    passed = re.findall(r"(\d+) passed", summary)
    assert passed
    assert int(passed[-1]) == int(total.group(1)) - 1
    assert re.search(r"\b1 deselected\b", summary)
    for category in ("failed", "error", "errors", "skipped", "xfailed", "xpassed"):
        assert not re.search(rf"\b\d+ {category}\b", summary), category
