"""End-to-end tests for the ``pose_estimation.validation`` harness.

Synthetic, footage-independent input is built by
``synthetic_session`` and the ``rendered_session`` fixture (conftest): a
real ChArUco solve plus a symmetric 12-keypoint "arm" skeleton projected
into each calibrated camera, so fusion -> ``world3d.csv`` ->
self-consistency runs deterministically without live inference.

The harness exercises every branch: solve calibration, load calibration,
reuse existing CSVs (CLI path), and the no-calibration error.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
import shutil

import numpy as np
import pytest

from pose_estimation.calibration import save_calibration
from pose_estimation.multicam import Session, SessionCamera
from pose_estimation.validation import (
    QA_THRESHOLDS,
    REPORT_SCHEMA_VERSION,
    THRESHOLDS,
    AgreementSection,
    Band,
    CalibrationSection,
    CameraIntrinsics,
    CameraTracking,
    Fusion3DSection,
    QAReport,
    Thresholds,
    TimingSection,
    Tracking2DSection,
    ValidationError,
    ValidationReport,
    _aggregate_clinical,
    _bone_length_cv,
    _camera_tracking,
    _expected_fusion_frames,
    _measure_fusion,
    _qa_parity,
    _read_world3d,
    _temporal_jitter_mm,
    main,
    qa_check,
    run_validation,
)
from pose_estimation.video_io import frame_count
from synthetic_session import (
    HAS_R as _HAS_R,
)
from synthetic_session import (
    N_SUBJECT_FRAMES as _N_SUBJECT_FRAMES,
)
from synthetic_session import (
    full_skeleton_processor as _full_skeleton_processor,
)
from synthetic_session import (
    render_bad_capture as _render_bad_capture,
)
from synthetic_session import (
    skeleton_processor as _skeleton_processor,
)
from synthetic_session import (
    write_video as _write_video,
)


def _align_short_tracking_to_source_tail(session_dir: pathlib.Path) -> dict[str, int]:
    """Declare the synthetic 8-row tracking clip as the source-video tail."""
    offsets: dict[str, int] = {}
    cameras = []
    for video in sorted(session_dir.glob("cam*.avi")):
        offset = max(0, frame_count(video) - _N_SUBJECT_FRAMES)
        offsets[video.stem] = offset
        cameras.append({"name": video.stem, "file": video.name, "sync_offset": offset})
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "session_id": session_dir.name,
                "cameras": cameras,
            }
        )
    )
    return offsets


def _shift_csv_frame_indices(path: pathlib.Path, offset: int) -> None:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["frame_idx"] = str(int(row["frame_idx"]) + offset)
            writer.writerow(row)


def _fill_blank_hands_from_shoulders(path: pathlib.Path) -> None:
    """Make the synthetic hands-arms fixture complete for coverage grading."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for side in ("left", "right"):
                for index in range(21):
                    for axis in ("x", "y", "z"):
                        column = f"{side}_hand_{index}_{axis}"
                        if column in row and not row[column]:
                            row[column] = row[f"arm_{side}_shoulder_{axis}"]
                    confidence_column = f"{side}_hand_{index}_conf"
                    if confidence_column in row:
                        row[confidence_column] = "1.0"
            writer.writerow(row)


def _tail_aligned_processor(processor):
    """Wrap an 8-row synthetic processor so its raw indices match the manifest."""

    def _wrapped(**kwargs):
        processor(**kwargs)
        offset = max(0, frame_count(pathlib.Path(kwargs["source"])) - _N_SUBJECT_FRAMES)
        output_csv = pathlib.Path(kwargs["output_csv"])
        _shift_csv_frame_indices(output_csv, offset)
        _fill_blank_hands_from_shoulders(output_csv)

    return _wrapped


def _complete_hands_processor(processor):
    def _wrapped(**kwargs):
        processor(**kwargs)
        _fill_blank_hands_from_shoulders(pathlib.Path(kwargs["output_csv"]))

    return _wrapped


def _prewrite_tail_aligned(
    session_dir: pathlib.Path,
    output_dir: pathlib.Path,
    processor,
) -> None:
    session_out = output_dir / session_dir.name
    session_out.mkdir(parents=True, exist_ok=True)
    wrapped = _tail_aligned_processor(processor)
    for video in sorted(session_dir.glob("cam*.avi")):
        wrapped(
            source=str(video),
            output_csv=session_out / f"{video.stem}.csv",
            output_diag=session_out / f"{video.stem}_diag.csv",
            video_name=f"{session_dir.name}/{video.stem}",
        )


# ---------------------------------------------------------------------------
# End-to-end: solve calibration branch
# ---------------------------------------------------------------------------


def test_run_validation_end_to_end_solve(rendered_session, tmp_path: pathlib.Path):
    session_dir, solved = rendered_session
    work = tmp_path / "session"
    shutil.copytree(session_dir, work)  # isolate the solve side effect (calibration.json)
    out = tmp_path / "out"

    report = run_validation(
        work,
        calibration=work,  # a dir without calibration.json → harness solves it
        camera_processor=_complete_hands_processor(
            _skeleton_processor(solved, frames=tuple(range(60)))
        ),
        output_dir=out,
        run_clinical=False,
    )

    # Calibration: solved this run, three cameras, sane intrinsics.
    cal = report.calibration
    assert cal.solved is True
    assert cal.n_cameras == 3
    assert cal.world_frame == "cam1"
    assert 0.0 < cal.reprojection_error_px < 2.0
    assert {c.name for c in cal.cameras} == {"cam1", "cam2", "cam3"}
    assert all(c.fx > 0 and c.fy > 0 for c in cal.cameras)
    assert (work / "calibration.json").is_file()  # solve persisted it

    # 2D tracking: three cameras read back, no dropped frames, high confidence.
    trk = report.tracking_2d
    assert len(trk.cameras) == 3
    assert trk.reused_existing_csvs is False
    for cam in trk.cameras:
        assert cam.observed_frames == 60
        assert cam.expected_frames == 60
        assert 0.0 < cam.detection_rate <= 1.0
        assert cam.dropped_frames == 0
        assert cam.low_confidence_fraction == pytest.approx(0.0, abs=1e-9)

    # 3D fusion: arms plus the completed synthetic hands reconstruct in front.
    fus = report.fusion_3d
    assert fus.n_frames_fused == 60
    assert fus.expected_frames == 60
    assert fus.fused_frame_fraction == pytest.approx(1.0)
    assert fus.n_active_keypoints == 54
    assert fus.trusted_keypoint_fraction == pytest.approx(1.0)
    assert fus.triangulation_angle_available is True
    assert fus.triangulation_angle_deg_p05 >= fus.triangulation_angle_gate_deg
    assert fus.n_views_median == pytest.approx(3.0)
    assert fus.n_views_min == 3
    assert fus.cheirality_violation_rate == pytest.approx(0.0)
    assert fus.unfused_keypoint_fraction == pytest.approx(0.0)
    assert 0.0 <= fus.reproj_err_px_median < 2.0
    assert fus.reproj_err_px_max < 5.0

    # Self-consistency: rigid symmetric skeleton → ~0 CV / symmetry / jitter.
    agr = report.agreement
    assert agr.has_baseline is False
    assert np.isfinite(agr.mean_bone_length_cv)
    assert agr.mean_bone_length_cv < 0.05
    assert np.isfinite(agr.mean_symmetry_rel_diff)
    assert agr.mean_symmetry_rel_diff < 0.05
    assert np.isfinite(agr.temporal_jitter_mm)
    assert agr.temporal_jitter_mm < 5.0
    assert agr.per_metric_error is None

    # Timing present and finite.
    tim = report.timing
    assert tim.det_device == "CPU"
    assert tim.pose_device == "NPU"
    assert tim.total_sec >= 0.0
    assert np.isfinite(tim.throughput_fps)

    # JSON is CI-parseable and NaN-free.
    text = json.dumps(report.to_json())
    assert "NaN" not in text
    assert "## 3D fusion" in report.to_markdown()


# ---------------------------------------------------------------------------
# End-to-end: load calibration branch
# ---------------------------------------------------------------------------


def test_run_validation_load_calibration(rendered_session, tmp_path: pathlib.Path):
    session_dir, solved = rendered_session
    work = tmp_path / "session"
    shutil.copytree(session_dir, work)
    _align_short_tracking_to_source_tail(work)
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"

    report = run_validation(
        work,
        calibration=calib_json,
        camera_processor=_tail_aligned_processor(_skeleton_processor(solved)),
        output_dir=out,
        run_clinical=False,
    )

    assert report.calibration.solved is False  # loaded, not solved
    assert report.calibration.n_cameras == 3
    assert report.fusion_3d.n_active_keypoints == 54
    assert report.fusion_3d.reproj_err_px_median < 1.0  # calib matches projection exactly


# ---------------------------------------------------------------------------
# Clinical metrics (R) — skip-aware
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_R, reason="R or required R packages unavailable")
def test_run_validation_runs_clinical_pipeline(rendered_session, tmp_path: pathlib.Path):
    session_dir, solved = rendered_session
    work = tmp_path / "session"
    shutil.copytree(session_dir, work)
    _align_short_tracking_to_source_tail(work)
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"

    report = run_validation(
        work,
        calibration=calib_json,
        camera_processor=_tail_aligned_processor(_skeleton_processor(solved)),
        output_dir=out,
        run_clinical=True,
    )

    agr = report.agreement
    assert agr.clinical_csv_produced is True
    assert any("clinical_3d" in name for name in agr.clinical_outputs)
    # R writes its outputs next to world3d.csv.
    assert list((out / work.name).glob("*_clinical_3d*.csv"))


def test_clinical_aggregate_accepts_only_frame_artifacts(tmp_path: pathlib.Path) -> None:
    frame = tmp_path / "capture_clinical_3d.csv"
    frame.write_text("metric,label\n2,a\n4,b\n")
    qc = tmp_path / "capture_clinical_3d_window_qc.csv"
    qc.write_text("n_expected_frames,frame_coverage\n30,0.8\n")

    actual = _aggregate_clinical(tmp_path, [frame.name, qc.name])

    assert actual == {"metric": 3.0}


def test_self_consistency_runs_without_r(rendered_session, tmp_path: pathlib.Path):
    """Self-consistency surrogates do not depend on R being installed."""
    session_dir, solved = rendered_session
    work = tmp_path / "session"
    shutil.copytree(session_dir, work)
    _align_short_tracking_to_source_tail(work)
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"

    report = run_validation(
        work,
        calibration=calib_json,
        camera_processor=_tail_aligned_processor(_skeleton_processor(solved)),
        output_dir=out,
        run_clinical=False,
    )
    assert np.isfinite(report.agreement.mean_bone_length_cv)
    assert report.agreement.bone_length_cv  # at least one arm bone measured


# ---------------------------------------------------------------------------
# Coverage denominators + trusted 3D samples
# ---------------------------------------------------------------------------


def test_camera_tracking_counts_missing_source_frames_as_dropped():
    frames = {
        2: (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0.9, 0.2]),
            0.0,
        ),
        4: (
            np.array([[1.0, 2.0], [np.nan, np.nan]]),
            np.array([0.9, 0.0]),
            0.1,
        ),
        # Finite carry output with zero confidence is not a detector hit.
        5: (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0.0, 0.0]),
            0.2,
        ),
    }

    measured = _camera_tracking(
        "cam1",
        frames,
        confidence_floor=0.3,
        expected_frames=4,
        sync_offset=2,
        n_keypoints=2,
    )

    assert measured.observed_frames == 3
    assert measured.expected_frames == 4
    assert measured.n_frames == 3  # compatibility alias
    assert measured.detection_rate == pytest.approx(3 / 8)
    assert measured.low_confidence_fraction == pytest.approx(1 / 3)
    assert measured.dropped_frames == 2


def test_camera_tracking_accepts_one_based_rtmlib_indices():
    frames = {idx: (np.array([[1.0, 2.0]]), np.array([0.9]), idx / 30) for idx in range(1, 5)}

    measured = _camera_tracking(
        "cam1",
        frames,
        confidence_floor=0.3,
        expected_frames=4,
        n_keypoints=1,
    )

    assert measured.observed_frames == 4
    assert measured.detection_rate == pytest.approx(1.0)
    assert measured.dropped_frames == 0


def test_expected_fusion_frames_is_second_largest_camera_count():
    tracking = Tracking2DSection(
        confidence_floor=0.3,
        reused_existing_csvs=False,
        total_observed_frames=19,
        total_expected_frames=19,
        mean_detection_rate=1.0,
        cameras=[
            CameraTracking("cam1", 10, 10, 1.0, 0.0, 0),
            CameraTracking("cam2", 7, 7, 1.0, 0.0, 0),
            CameraTracking("cam3", 2, 2, 1.0, 0.0, 0),
        ],
    )

    assert _expected_fusion_frames(tracking, min_views=2) == 7


def _write_minimal_world3d(
    path: pathlib.Path,
    *,
    include_angle: bool = True,
    blank_angles: bool = False,
    include_candidate_views: bool = False,
) -> None:
    fields = [
        "video",
        "frame_idx",
        "timestamp_sec",
        "kp_x_m",
        "kp_y_m",
        "kp_z_m",
        "kp_confidence",
        "kp_reproj_err_px",
        "kp_n_views",
        "kp_cheirality_ok",
    ]
    if include_angle:
        fields.append("kp_triangulation_angle_deg")
    if include_candidate_views:
        fields.append("kp_candidate_n_views")
    rows = [
        (0, 5.0, 1, 5.0),  # trusted
        (1, 25.0, 1, 5.0),  # reprojection gate
        (2, 5.0, 0, 5.0),  # cheirality gate
        (3, 5.0, 1, 0.5),  # angle gate
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for frame_idx, reproj, cheirality, angle in rows:
            row = {
                "video": "synthetic",
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / 30,
                "kp_x_m": 0.1 * frame_idx,
                "kp_y_m": 0.0,
                "kp_z_m": 1.0,
                "kp_confidence": 0.9,
                "kp_reproj_err_px": reproj,
                "kp_n_views": 2,
                "kp_cheirality_ok": cheirality,
            }
            if include_angle:
                row["kp_triangulation_angle_deg"] = "" if blank_angles else angle
            if include_candidate_views:
                row["kp_candidate_n_views"] = 3
            writer.writerow(row)


def test_world3d_trust_gates_and_expected_coverage(tmp_path: pathlib.Path):
    path = tmp_path / "world3d.csv"
    _write_minimal_world3d(path)

    _names, frames = _read_world3d(path)
    assert [bool(fr["trusted"][0]) for fr in frames] == [True, False, False, False]

    fusion = _measure_fusion(path, expected_frames=5)
    assert fusion.n_frames_fused == 4
    assert fusion.fused_frame_fraction == pytest.approx(4 / 5)
    assert fusion.trusted_keypoint_fraction == pytest.approx(1 / 5)
    assert fusion.triangulation_angle_available is True
    assert fusion.triangulation_angle_deg_p05 < THRESHOLDS.min_triangulation_angle_p05_deg.warn


def test_world3d_legacy_without_angle_remains_trust_compatible(tmp_path: pathlib.Path):
    path = tmp_path / "legacy_world3d.csv"
    _write_minimal_world3d(path, include_angle=False)

    _names, frames = _read_world3d(path)
    # The fourth sample is trusted because legacy files cannot be angle-gated.
    assert [bool(fr["trusted"][0]) for fr in frames] == [True, False, False, True]
    fusion = _measure_fusion(path, expected_frames=4)
    assert fusion.triangulation_angle_available is False
    assert fusion.trusted_keypoint_fraction == pytest.approx(0.5)
    assert fusion.candidate_views_available is False
    assert math.isnan(fusion.view_rejection_fraction)


def test_world3d_all_blank_present_angle_column_fails_closed(tmp_path: pathlib.Path):
    path = tmp_path / "mock_world3d.csv"
    _write_minimal_world3d(path, blank_angles=True)

    _names, frames = _read_world3d(path)
    assert [bool(fr["trusted"][0]) for fr in frames] == [False, False, False, False]
    fusion = _measure_fusion(path, expected_frames=4)
    assert fusion.triangulation_angle_available is True
    assert fusion.trusted_keypoint_fraction == 0.0


def test_fusion_coverage_accepts_one_based_world_indices(tmp_path: pathlib.Path):
    path = tmp_path / "one_based_world3d.csv"
    _write_minimal_world3d(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["frame_idx"] = str(int(row["frame_idx"]) + 1)
            writer.writerow(row)

    fusion = _measure_fusion(path, expected_frames=4)
    assert fusion.n_frames_fused == 4
    assert fusion.fused_frame_fraction == pytest.approx(1.0)


def test_fusion_coverage_excludes_out_of_range_world_indices(tmp_path: pathlib.Path):
    path = tmp_path / "out_of_range_world3d.csv"
    _write_minimal_world3d(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows[-1]["frame_idx"] = "1000"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fusion = _measure_fusion(path, expected_frames=4)

    assert fusion.n_frames_fused == 3
    assert fusion.fused_frame_fraction == pytest.approx(3 / 4)


def test_candidate_view_rejection_is_reported(tmp_path: pathlib.Path):
    path = tmp_path / "consensus_world3d.csv"
    _write_minimal_world3d(path, include_candidate_views=True)

    fusion = _measure_fusion(path, expected_frames=4)
    assert fusion.candidate_views_available is True
    assert fusion.candidate_n_views_median == pytest.approx(3.0)
    assert fusion.view_rejection_fraction == pytest.approx(1 / 3)


def test_attempted_but_rejected_keypoint_remains_in_fusion_denominator(
    tmp_path: pathlib.Path,
):
    path = tmp_path / "rejected_world3d.csv"
    fields = [
        "video",
        "frame_idx",
        "timestamp_sec",
        "kp_x_m",
        "kp_y_m",
        "kp_z_m",
        "kp_confidence",
        "kp_reproj_err_px",
        "kp_n_views",
        "kp_cheirality_ok",
        "kp_triangulation_angle_deg",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for frame_idx in range(2):
            writer.writerow(
                {
                    "video": "synthetic",
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / 30,
                    "kp_x_m": "",
                    "kp_y_m": "",
                    "kp_z_m": "",
                    "kp_confidence": 0.0,
                    "kp_reproj_err_px": "",
                    "kp_n_views": 2,
                    "kp_cheirality_ok": 0,
                    "kp_triangulation_angle_deg": 0.5,
                }
            )

    fusion = _measure_fusion(path, expected_frames=2)
    assert fusion.n_active_keypoints == 1
    assert fusion.unfused_keypoint_fraction == pytest.approx(1.0)
    assert fusion.trusted_keypoint_fraction == pytest.approx(0.0)
    assert fusion.triangulation_angle_deg_p05 == pytest.approx(0.5)


def test_bone_cv_single_observation_is_unmeasured():
    cv = _bone_length_cv({"left_forearm": np.array([0.25, np.nan])})
    assert math.isnan(cv["left_forearm"])


def test_temporal_jitter_does_not_bridge_frame_gaps():
    positions = (0.0, 0.1, 0.0)
    frames = [
        {
            "frame_idx": np.asarray(frame_idx),
            "xyz": np.array([[x, 0.0, 1.0]]),
            "trusted_xyz": np.array([[x, 0.0, 1.0]]),
        }
        for frame_idx, x in zip((0, 2, 4), positions, strict=True)
    ]
    assert math.isnan(_temporal_jitter_mm(["kp"], frames))

    for i, frame in enumerate(frames):
        frame["frame_idx"] = np.asarray(i)
    assert _temporal_jitter_mm(["kp"], frames) == pytest.approx(200.0)


def test_qa_parity_includes_zero_frame_camera(monkeypatch, tmp_path: pathlib.Path):
    session = Session(
        session_id="synthetic",
        directory=tmp_path,
        cameras=[
            SessionCamera("cam1", pathlib.Path("cam1.avi")),
            SessionCamera("cam2", pathlib.Path("cam2.avi")),
            SessionCamera("cam3", pathlib.Path("cam3.avi")),
        ],
    )
    counts = {"cam1": 100, "cam2": 100, "cam3": 0}
    monkeypatch.setattr(
        "pose_estimation.validation.frame_count", lambda path: counts[pathlib.Path(path).stem]
    )

    parity = _qa_parity(session)
    assert parity.frame_counts == counts
    assert parity.disparity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI + reuse-existing-CSVs branch
# ---------------------------------------------------------------------------


def test_cli_reuses_existing_csvs_and_writes_reports(rendered_session, tmp_path: pathlib.Path):
    session_dir, solved = rendered_session
    work = tmp_path / "session"
    shutil.copytree(session_dir, work)
    _align_short_tracking_to_source_tail(work)
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"
    # Pre-write per-camera CSVs so the harness (no processor) reuses them.
    _prewrite_tail_aligned(work, out, _skeleton_processor(solved))

    report_json = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    rc = main(
        [
            "--session-dir",
            str(work),
            "--calibration",
            str(calib_json),
            "--output-dir",
            str(out),
            "--out",
            str(report_json),
            "--markdown",
            str(markdown),
            "--no-clinical",
        ]
    )

    assert rc == 0
    assert report_json.is_file()
    assert markdown.is_file()
    payload = json.loads(report_json.read_text())
    assert payload["tracking_2d"]["reused_existing_csvs"] is True
    assert payload["fusion_3d"]["n_active_keypoints"] == 54
    assert "NaN" not in report_json.read_text()
    assert markdown.read_text().startswith("# Validation report")


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_calibration_raises(tmp_path: pathlib.Path):
    session_dir = tmp_path / "session_nocal"
    session_dir.mkdir()
    # Three solid-colour videos so discovery succeeds, but no calibration.
    size = (64, 48)
    for name in ("cam1", "cam2", "cam3"):
        frames = [np.full((size[1], size[0]), 100, dtype=np.uint8) for _ in range(4)]
        if not _write_video(session_dir / f"{name}.avi", frames, size):
            pytest.skip("MJPG/AVI codec unavailable on this host")

    with pytest.raises(ValidationError, match="calibration"):
        run_validation(session_dir, output_dir=tmp_path / "out")


def test_cli_missing_calibration_exit_code(tmp_path: pathlib.Path):
    session_dir = tmp_path / "session_nocal"
    session_dir.mkdir()
    size = (64, 48)
    for name in ("cam1", "cam2", "cam3"):
        frames = [np.full((size[1], size[0]), 100, dtype=np.uint8) for _ in range(4)]
        if not _write_video(session_dir / f"{name}.avi", frames, size):
            pytest.skip("MJPG/AVI codec unavailable on this host")

    rc = main(["--session-dir", str(session_dir), "--output-dir", str(tmp_path / "out")])
    assert rc == 2


# ---------------------------------------------------------------------------
# Verdict grading — constructed reports, no harness run
# ---------------------------------------------------------------------------


def _good_report() -> ValidationReport:
    """A minimal all-PASS report; tests mutate single fields to grade them.

    Only the graded fields matter, so a single camera entry stands in for
    the rig (``n_views_median`` carries the cross-camera redundancy that
    the verdict actually grades).
    """
    return ValidationReport(
        session_id="synthetic",
        schema_version=REPORT_SCHEMA_VERSION,
        calibration=CalibrationSection(
            n_cameras=3,
            world_frame="cam1",
            reprojection_error_px=0.4,
            solved=True,
            cameras=[
                CameraIntrinsics(
                    name="cam1",
                    resolution=(1280, 720),
                    fx=900.0,
                    fy=900.0,
                    cx=640.0,
                    cy=360.0,
                    distortion_l2=0.01,
                )
            ],
        ),
        tracking_2d=Tracking2DSection(
            confidence_floor=THRESHOLDS.confidence_floor,
            reused_existing_csvs=False,
            total_observed_frames=24,
            total_expected_frames=24,
            mean_detection_rate=1.0,
            cameras=[
                CameraTracking(
                    name="cam1",
                    observed_frames=8,
                    expected_frames=8,
                    detection_rate=1.0,
                    low_confidence_fraction=0.0,
                    dropped_frames=0,
                )
            ],
        ),
        fusion_3d=Fusion3DSection(
            n_frames_fused=8,
            expected_frames=8,
            fused_frame_fraction=1.0,
            n_active_keypoints=12,
            trusted_keypoint_fraction=1.0,
            reproj_err_px_median=3.0,
            reproj_err_px_p95=10.0,
            reproj_err_px_max=12.0,
            n_views_median=3.0,
            n_views_min=3,
            cheirality_violation_rate=0.0,
            triangulation_angle_available=True,
            triangulation_angle_gate_deg=1.0,
            triangulation_angle_deg_p05=10.0,
            triangulation_angle_deg_median=15.0,
            candidate_views_available=True,
            candidate_n_views_median=3.0,
            view_rejection_fraction=0.0,
            unfused_keypoint_fraction=0.0,
        ),
        timing=TimingSection(
            det_device="CPU",
            pose_device="NPU",
            backend="onnxruntime",
            solve_sec=1.0,
            tracking_2d_sec=1.0,
            fusion_sec=0.1,
            clinical_sec=0.0,
            total_sec=2.1,
            throughput_fps=60.0,
            tracking_2d_per_camera={"cam1": 1.0},
        ),
        agreement=AgreementSection(
            has_baseline=False,
            clinical_csv_produced=False,
            clinical_outputs=[],
            mean_bone_length_cv=0.01,
            bone_length_cv={"left_forearm": 0.01},
            mean_symmetry_rel_diff=0.01,
            symmetry_rel_diff={"forearm": 0.01},
            temporal_jitter_mm=1.0,
            per_metric_error=None,
        ),
        notes=[],
    )


def test_verdict_good_report_passes():
    v = _good_report().verdict()
    assert v.grade == "PASS"
    assert v.passed is True
    assert v.thresholds_version == THRESHOLDS.version
    assert all(c.grade == "PASS" for c in v.checks if not c.informational)
    info = {c.name for c in v.checks if c.informational}
    assert info == {"timing.throughput_fps", "agreement.mean_symmetry_rel_diff"}


@pytest.mark.parametrize(
    ("mutate", "check_name"),
    [
        (
            lambda r: setattr(r.calibration, "reprojection_error_px", 5.0),
            "calibration.reprojection_error_px",
        ),
        (
            lambda r: setattr(r.fusion_3d, "reproj_err_px_median", 20.0),
            "fusion.reproj_err_px_median",
        ),
        (lambda r: setattr(r.fusion_3d, "reproj_err_px_p95", 30.0), "fusion.reproj_err_px_p95"),
        (
            lambda r: setattr(r.fusion_3d, "unfused_keypoint_fraction", 0.5),
            "fusion.unfused_keypoint_fraction",
        ),
        (
            lambda r: setattr(r.fusion_3d, "cheirality_violation_rate", 0.5),
            "fusion.cheirality_violation_rate",
        ),
        (
            lambda r: setattr(r.fusion_3d, "fused_frame_fraction", 0.5),
            "fusion.fused_frame_fraction",
        ),
        (
            lambda r: setattr(r.fusion_3d, "trusted_keypoint_fraction", 0.5),
            "fusion.trusted_keypoint_fraction",
        ),
        (
            lambda r: setattr(r.fusion_3d, "triangulation_angle_deg_p05", 0.5),
            "fusion.triangulation_angle_deg_p05",
        ),
        (
            lambda r: setattr(r.fusion_3d, "view_rejection_fraction", 0.3),
            "fusion.view_rejection_fraction",
        ),
        (lambda r: setattr(r.fusion_3d, "n_views_min", 1), "fusion.n_views_min"),
        (
            lambda r: setattr(r.agreement, "mean_bone_length_cv", 0.5),
            "agreement.mean_bone_length_cv",
        ),
        (
            lambda r: setattr(r.agreement, "temporal_jitter_mm", 50.0),
            "agreement.temporal_jitter_mm",
        ),
        (
            lambda r: setattr(r.tracking_2d.cameras[0], "low_confidence_fraction", 0.9),
            "tracking.worst_low_confidence_fraction",
        ),
        (
            lambda r: setattr(r.tracking_2d.cameras[0], "detection_rate", 0.4),
            "tracking.worst_detection_rate",
        ),
    ],
)
def test_verdict_fail_bands(mutate, check_name):
    r = _good_report()
    mutate(r)
    v = r.verdict()
    assert v.grade == "FAIL"
    assert v.passed is False
    assert check_name in {c.name for c in v.checks if c.grade == "FAIL"}


@pytest.mark.parametrize(
    ("mutate", "check_name"),
    [
        (
            lambda r: setattr(r.calibration, "reprojection_error_px", 1.5),
            "calibration.reprojection_error_px",
        ),
        (
            lambda r: setattr(r.fusion_3d, "reproj_err_px_median", 10.0),
            "fusion.reproj_err_px_median",
        ),
        (lambda r: setattr(r.fusion_3d, "n_views_median", 2.5), "fusion.n_views_median"),
        (
            lambda r: setattr(r.fusion_3d, "fused_frame_fraction", 0.9),
            "fusion.fused_frame_fraction",
        ),
        (
            lambda r: setattr(r.fusion_3d, "trusted_keypoint_fraction", 0.8),
            "fusion.trusted_keypoint_fraction",
        ),
        (
            lambda r: setattr(r.fusion_3d, "triangulation_angle_deg_p05", 1.5),
            "fusion.triangulation_angle_deg_p05",
        ),
        (
            lambda r: setattr(r.fusion_3d, "view_rejection_fraction", 0.1),
            "fusion.view_rejection_fraction",
        ),
        (
            lambda r: setattr(r.tracking_2d.cameras[0], "detection_rate", 0.7),
            "tracking.worst_detection_rate",
        ),
        (
            lambda r: setattr(r.agreement, "mean_bone_length_cv", 0.07),
            "agreement.mean_bone_length_cv",
        ),
    ],
)
def test_verdict_warn_bands(mutate, check_name):
    r = _good_report()
    mutate(r)
    v = r.verdict()
    assert v.grade == "WARN"
    assert v.passed is True  # WARN is not a hard failure
    assert check_name in {c.name for c in v.checks if c.grade == "WARN"}


def test_informational_checks_never_escalate_overall():
    r = _good_report()
    r.timing.throughput_fps = 0.1  # well below the fail band
    r.agreement.mean_symmetry_rel_diff = 0.9  # well below the fail band
    v = r.verdict()
    assert v.grade == "PASS"  # informational checks do not raise the overall
    failing_info = {c.name for c in v.checks if c.informational and c.grade == "FAIL"}
    assert failing_info == {"timing.throughput_fps", "agreement.mean_symmetry_rel_diff"}


def test_non_finite_metric_grades_warn():
    r = _good_report()
    r.fusion_3d.reproj_err_px_median = float("nan")
    v = r.verdict()
    assert v.grade == "WARN"
    check = next(c for c in v.checks if c.name == "fusion.reproj_err_px_median")
    assert check.grade == "WARN"


def test_legacy_report_without_angles_is_not_failed_for_missing_column():
    r = _good_report()
    r.fusion_3d.triangulation_angle_available = False
    r.fusion_3d.triangulation_angle_deg_p05 = float("nan")
    verdict = r.verdict()
    assert verdict.grade == "PASS"
    assert "fusion.triangulation_angle_deg_p05" not in {c.name for c in verdict.checks}
    assert any("legacy schema" in note for note in verdict.notes)


def test_legacy_report_without_candidate_views_is_not_assumed_healthy():
    r = _good_report()
    r.fusion_3d.candidate_views_available = False
    r.fusion_3d.candidate_n_views_median = float("nan")
    r.fusion_3d.view_rejection_fraction = float("nan")
    verdict = r.verdict()
    assert verdict.grade == "PASS"
    assert "fusion.view_rejection_fraction" not in {c.name for c in verdict.checks}
    assert any("candidate-view" in note for note in verdict.notes)


def test_verdict_no_baseline_notes_unvalidated():
    v = _good_report().verdict()
    assert any("UNVALIDATED" in note for note in v.notes)


def test_verdict_baseline_angle_agreement_grades_and_notes():
    r = _good_report()
    r.agreement.has_baseline = True
    r.agreement.per_metric_error = {"elbow_angle_deg": 12.0, "reach_raw": 0.05}
    v = r.verdict()
    names = {c.name for c in v.checks}
    assert v.grade == "FAIL"  # 12 deg exceeds the 10 deg fail tolerance
    assert "agreement.elbow_angle_deg" in names
    assert "agreement.reach_raw" not in names  # non-angle metric is not graded
    assert any("reach_raw" in note for note in v.notes)


def test_verdict_baseline_angle_within_tolerance_passes():
    r = _good_report()
    r.agreement.has_baseline = True
    r.agreement.per_metric_error = {"elbow_angle_deg": 3.0}
    v = r.verdict()
    assert v.grade == "PASS"
    angle = next(c for c in v.checks if c.name == "agreement.elbow_angle_deg")
    assert angle.grade == "PASS"


def test_verdict_accepts_custom_thresholds():
    strict = dataclasses.replace(
        THRESHOLDS, version=99, calib_reproj_rms_px=Band(warn=0.1, fail=0.2)
    )
    assert isinstance(strict, Thresholds)
    v = _good_report().verdict(strict)  # 0.4 px now exceeds the 0.2 fail band
    assert v.thresholds_version == 99
    assert v.grade == "FAIL"


def test_verdict_surfaced_in_json_and_markdown():
    r = _good_report()
    payload = r.to_json()
    assert payload["verdict"]["grade"] == "PASS"
    assert payload["verdict"]["thresholds_version"] == THRESHOLDS.version
    assert isinstance(payload["verdict"]["checks"], list)
    md = r.to_markdown()
    assert "## Verdict:" in md
    assert "PASS" in md
    assert f"thresholds v{THRESHOLDS.version}" in md


def test_verdict_json_is_nan_free():
    r = _good_report()
    r.fusion_3d.reproj_err_px_median = float("nan")
    text = json.dumps(r.to_json())
    assert "NaN" not in text
    payload = json.loads(text)
    check = next(
        c for c in payload["verdict"]["checks"] if c["name"] == "fusion.reproj_err_px_median"
    )
    assert check["value"] is None  # non-finite serialised to null
    assert check["grade"] == "WARN"


@pytest.mark.parametrize(
    ("grade", "strict", "expected_rc"),
    [
        ("PASS", False, 0),
        ("WARN", False, 0),
        ("FAIL", False, 1),
        ("PASS", True, 0),
        ("WARN", True, 1),  # --strict promotes WARN to a failure
        ("FAIL", True, 1),
    ],
)
def test_cli_exit_code_matches_verdict(
    monkeypatch, tmp_path: pathlib.Path, grade, strict, expected_rc
):
    report = _good_report()
    if grade == "WARN":
        report.calibration.reprojection_error_px = 1.5
    elif grade == "FAIL":
        report.calibration.reprojection_error_px = 5.0

    monkeypatch.setattr("pose_estimation.validation.run_validation", lambda *a, **k: report)

    argv = ["--session-dir", str(tmp_path), "--out", str(tmp_path / "report.json")]
    if strict:
        argv.append("--strict")
    assert main(argv) == expected_rc


# ---------------------------------------------------------------------------
# Pre-flight capture QA gate
# ---------------------------------------------------------------------------


def _checks_by_name(report: QAReport) -> dict[str, str]:
    """Map each QA check name → its grade for assertions."""
    return {c.name: c.grade for c in report.verdict().checks}


def test_qa_good_capture_passes(rendered_session, tmp_path: pathlib.Path):
    """A well-formed capture clears every QA sufficiency gate (no FAIL)."""
    session_dir, solved = rendered_session
    work = tmp_path / "good"
    shutil.copytree(session_dir, work)
    _align_short_tracking_to_source_tail(work)
    calib_capture = tmp_path / "calibration_capture"
    shutil.copytree(session_dir, calib_capture)
    save_calibration(solved, calib_capture / "calibration.json")  # load, not re-solve

    report = qa_check(
        work,
        calibration=calib_capture,
        camera_processor=_tail_aligned_processor(_full_skeleton_processor(solved)),
        output_dir=str(tmp_path / "out"),
    )
    grades = _checks_by_name(report)

    # Not a failure overall (board coverage may WARN on the oblique synthetic
    # cameras — an honest "sweep more" signal — but nothing FAILs).
    assert report.verdict().grade != "FAIL"
    # The sufficiency gates a good capture must clear outright:
    assert grades["calibration.reprojection_error_px"] == "PASS"
    assert grades["calibration.min_charuco_frames"] == "PASS"
    assert grades["calibration.worst_charuco_detection_rate"] == "PASS"
    assert grades["parity.frame_count_disparity"] == "PASS"
    assert grades["subject.worst_detection_rate"] == "PASS"
    # All three cameras' board coverage cleared the FAIL floor.
    cov_fail = QA_THRESHOLDS.min_board_coverage.fail
    assert all(c.coverage > cov_fail for c in report.calibration.cameras)
    assert report.parity.disparity == 0.0


def test_qa_bad_capture_is_flagged(tmp_path: pathlib.Path):
    """A sparse, desynced capture FAILs with the specific faults named."""
    bad = tmp_path / "bad"
    _render_bad_capture(bad)

    report = qa_check(bad, calibration=bad, output_dir=str(tmp_path / "out"))
    grades = _checks_by_name(report)

    assert report.verdict().grade == "FAIL"
    # Sparse centre-bound board → below the intrinsic floor + low coverage.
    assert grades["calibration.min_charuco_frames"] == "FAIL"
    assert grades["calibration.worst_board_coverage"] == "FAIL"
    # cam3 truncated to half its frames → parity (desync) violation.
    assert grades["parity.frame_count_disparity"] == "FAIL"
    assert report.parity.disparity >= 0.4
    # The failed solve is surfaced, not silently passed.
    assert any("RMS unassessed" in n for n in report.notes)


def test_qa_to_json_carries_verdict_and_is_nan_free(rendered_session, tmp_path: pathlib.Path):
    session_dir, solved = rendered_session
    work = tmp_path / "good"
    shutil.copytree(session_dir, work)
    _align_short_tracking_to_source_tail(work)
    calib_capture = tmp_path / "calibration_capture"
    shutil.copytree(session_dir, calib_capture)
    save_calibration(solved, calib_capture / "calibration.json")

    report = qa_check(
        work,
        calibration=calib_capture,
        camera_processor=_tail_aligned_processor(_full_skeleton_processor(solved)),
        output_dir=str(tmp_path / "out"),
    )
    payload = report.to_json()
    assert payload["verdict"]["grade"] in {"PASS", "WARN", "FAIL"}
    assert payload["schema_version"] == report.schema_version
    # Non-finite floats must serialise to null (CI-parseable).
    text = json.dumps(payload)
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "Capture QA" in report.to_markdown()


def test_qa_cli_exit_codes(rendered_session, tmp_path: pathlib.Path):
    """``--qa-only`` exits 0 on a good capture and 1 on a flagged one."""
    session_dir, solved = rendered_session
    good = tmp_path / "good"
    shutil.copytree(session_dir, good)
    _align_short_tracking_to_source_tail(good)
    calib_capture = tmp_path / "calibration_capture"
    shutil.copytree(session_dir, calib_capture)
    save_calibration(solved, calib_capture / "calibration.json")
    _prewrite_tail_aligned(good, tmp_path / "out_good", _full_skeleton_processor(solved))

    rc_good = main(
        [
            "--session-dir",
            str(good),
            "--calibration",
            str(calib_capture),
            "--qa-only",
            "--output-dir",
            str(tmp_path / "out_good"),
            "--out",
            str(tmp_path / "qa_good.json"),
        ]
    )
    assert rc_good == 0
    assert (tmp_path / "qa_good.json").is_file()

    bad = tmp_path / "bad"
    _render_bad_capture(bad)
    rc_bad = main(
        [
            "--session-dir",
            str(bad),
            "--calibration",
            str(bad),
            "--qa-only",
            "--output-dir",
            str(tmp_path / "out_bad"),
            "--out",
            str(tmp_path / "qa_bad.json"),
        ]
    )
    assert rc_bad == 1
