"""End-to-end tests for the ``pose_estimation.validation`` harness.

Synthetic, footage-independent input (roadmap Phase 1):

- **Calibration session** — a ChArUco board rendered into MJPG videos
  (3x supersample + INTER_AREA, mirroring ``test_charuco``) that
  ``solve_charuco`` resolves for real.
- **2D tracking output** — a symmetric 12-keypoint "arm" skeleton
  projected into each calibrated camera and written as per-camera CSVs
  (mirroring ``test_multicam``), so fusion → ``world3d.csv`` →
  self-consistency runs deterministically without live inference.

The harness exercises every branch: solve calibration, load calibration,
reuse existing CSVs (CLI path), and the no-calibration error.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
import pytest

from pose_estimation._types import CameraCalibration, SessionCalibration
from pose_estimation.calibration import save_calibration
from pose_estimation.charuco import make_charuco_board, render_charuco_board, solve_charuco
from pose_estimation.export import frame_to_rows, open_csv_writer
from pose_estimation.processing import TRACKING_HANDS_ARMS
from pose_estimation.triangulation import project_points
from pose_estimation.validation import (
    THRESHOLDS,
    AgreementSection,
    Band,
    CalibrationSection,
    CameraIntrinsics,
    CameraTracking,
    Fusion3DSection,
    Thresholds,
    TimingSection,
    Tracking2DSection,
    ValidationError,
    ValidationReport,
    main,
    run_validation,
)

# ---------------------------------------------------------------------------
# ChArUco calibration-session render (mirrors test_charuco)
# ---------------------------------------------------------------------------

_BOARD = make_charuco_board()
_SX, _SY = _BOARD.getChessboardSize()
_SQ = 0.04
_BOARD_W_M, _BOARD_H_M = _SX * _SQ, _SY * _SQ
_BOARD_IMG = render_charuco_board(_BOARD, px_per_square=100, margin_squares=0)
_BG_GRAY = 180

# Ground-truth rig: cam1 = world; cam2/cam3 yawed toward the volume.
_GT: dict[str, dict[str, Any]] = {
    "cam1": {
        "K": np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]),
        "size": (1280, 720),
        "rvec": np.zeros(3),
        "tvec": np.zeros(3),
    },
    "cam2": {
        "K": np.array([[880.0, 0, 650], [0, 880.0, 350], [0, 0, 1]]),
        "size": (1280, 720),
        "rvec": np.array([0.0, np.deg2rad(25.0), 0.0]),
        "tvec": np.array([-0.8, 0.0, 0.25]),
    },
    "cam3": {
        "K": np.array([[700.0, 0, 480], [0, 700.0, 270], [0, 0, 1]]),
        "size": (960, 540),
        "rvec": np.array([0.0, np.deg2rad(-20.0), 0.05]),
        "tvec": np.array([0.7, 0.0, 0.2]),
    },
}


def _board_poses(n: int, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    """Centre-parametrised board poses facing the rig with varied tilt."""
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.6)])
        tvec = np.array(
            [rng.uniform(-0.40, 0.25), rng.uniform(-0.15, 0.15), rng.uniform(0.70, 1.35)]
        )
        poses.append((rvec, tvec))
    return poses


def _render_view(
    K: np.ndarray,
    rvec_cam: np.ndarray,
    tvec_cam: np.ndarray,
    rvec_board: np.ndarray,
    tvec_board: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Render the board seen by a camera; plain background if off-view."""
    w, h = size
    blank = np.full((h, w), _BG_GRAY, dtype=np.uint8)
    corners_m = np.array(
        [[0, 0, 0], [_BOARD_W_M, 0, 0], [_BOARD_W_M, _BOARD_H_M, 0], [0, _BOARD_H_M, 0]]
    )
    centred = corners_m - np.array([_BOARD_W_M / 2, _BOARD_H_M / 2, 0.0])
    R_b = cv2.Rodrigues(np.asarray(rvec_board, dtype=np.float64))[0]
    R_c = cv2.Rodrigues(np.asarray(rvec_cam, dtype=np.float64))[0]
    pts_world = (R_b @ centred.T).T + tvec_board
    pts_cam = (R_c @ pts_world.T).T + tvec_cam
    if np.any(pts_cam[:, 2] <= 0.1):
        return blank
    proj = (K @ pts_cam.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    if proj[:, 0].min() < 10 or proj[:, 0].max() > w - 10:
        return blank
    if proj[:, 1].min() < 10 or proj[:, 1].max() > h - 10:
        return blank
    bh, bw = _BOARD_IMG.shape[:2]
    src = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
    ss = 3  # supersample: anti-alias marker interiors
    H = cv2.getPerspectiveTransform(src, proj.astype(np.float32) * ss)
    canvas = cv2.warpPerspective(
        _BOARD_IMG,
        H,
        (w * ss, h * ss),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_BG_GRAY,
    )
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def _write_video(path: pathlib.Path, frames: list[np.ndarray], size: tuple[int, int]) -> bool:
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, size)
    if not writer.isOpened():
        return False
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path.is_file() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Synthetic subject skeleton (mirrors test_multicam projected CSVs)
# ---------------------------------------------------------------------------

# 12 "arm" keypoints (hands-arms mode), left/right symmetric, in the rig's
# co-visible working volume (z ~ 1 m).  Order = export.ARM_KEYPOINT_NAMES.
_SKEL_BASE = np.array(
    [
        [-0.12, -0.10, 1.00],  # left_shoulder
        [0.12, -0.10, 1.00],  # right_shoulder
        [-0.16, 0.00, 1.02],  # left_elbow
        [0.16, 0.00, 1.02],  # right_elbow
        [-0.14, 0.12, 1.04],  # left_wrist
        [0.14, 0.12, 1.04],  # right_wrist
        [-0.13, 0.15, 1.04],  # left_index_base
        [0.13, 0.15, 1.04],  # right_index_base
        [-0.12, 0.16, 1.04],  # left_middle_base
        [0.12, 0.16, 1.04],  # right_middle_base
        [-0.11, 0.17, 1.04],  # left_pinky_base
        [0.11, 0.17, 1.04],  # right_pinky_base
    ]
)
_N_SUBJECT_FRAMES = 8


def _skel_world(frame_idx: int) -> np.ndarray:
    """Rigid skeleton translating linearly over time (constant bone lengths)."""
    return _SKEL_BASE + np.array([0.004, 0.002, 0.003]) * frame_idx


def _write_skeleton_csv(csv_path: pathlib.Path, camera: CameraCalibration) -> None:
    """Project the moving skeleton into *camera* and write its per-camera CSV."""
    width, height = camera["resolution"]
    fh, writer = open_csv_writer(csv_path, tracking=TRACKING_HANDS_ARMS)
    try:
        for f in range(_N_SUBJECT_FRAMES):
            px = project_points(_skel_world(f), camera)
            lm = np.concatenate([px, np.zeros((len(px), 1))], axis=1)
            vis = np.full(len(px), 0.95)
            for row in frame_to_rows(
                "v", f, f / 30.0, height, width, [lm], [vis], [], [], tracking=TRACKING_HANDS_ARMS
            ):
                writer.writerow(row)
    finally:
        fh.close()


def _skeleton_processor(calib: SessionCalibration):
    """camera_processor that writes a projected-skeleton CSV per camera."""

    def _proc(
        *,
        source: str,
        output_csv: pathlib.Path,
        output_diag: pathlib.Path,
        video_name: str,
        **_kw: Any,
    ) -> None:
        name = pathlib.Path(output_csv).stem
        _write_skeleton_csv(pathlib.Path(output_csv), calib["cameras"][name])

    return _proc


def _prewrite_csvs(session_out: pathlib.Path, calib: SessionCalibration) -> None:
    """Pre-write per-camera CSVs so the harness reuses them (CLI path)."""
    session_out.mkdir(parents=True, exist_ok=True)
    for name, camera in calib["cameras"].items():
        _write_skeleton_csv(session_out / f"{name}.csv", camera)


def _r_available() -> bool:
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
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


_HAS_R = _r_available()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered_session(tmp_path_factory: pytest.TempPathFactory):
    """A rendered 3-camera ChArUco session + its solved calibration.

    The calibration is solved once here (no ``calibration.json`` written
    into the directory, so the solve-path test can copy a clean tree).
    """
    session_dir = tmp_path_factory.mktemp("calib_session")
    poses = _board_poses(60)
    for name, gt in _GT.items():
        frames = [
            _render_view(gt["K"], gt["rvec"], gt["tvec"], rb, tb, gt["size"]) for rb, tb in poses
        ]
        if not _write_video(session_dir / f"{name}.avi", frames, gt["size"]):
            pytest.skip("MJPG/AVI codec unavailable on this host")
    solved = solve_charuco(session_dir)
    return session_dir, solved


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
        camera_processor=_skeleton_processor(solved),
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
        assert cam.n_frames == _N_SUBJECT_FRAMES
        assert 0.0 < cam.detection_rate <= 1.0
        assert cam.dropped_frames == 0
        assert cam.low_confidence_fraction == pytest.approx(0.0, abs=1e-9)

    # 3D fusion: the 12 arm keypoints reconstruct from 3 views, in front.
    fus = report.fusion_3d
    assert fus.n_frames_fused == _N_SUBJECT_FRAMES
    assert fus.n_active_keypoints == 12
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
    assert tim.device == "NPU"
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
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"

    report = run_validation(
        session_dir,
        calibration=calib_json,
        camera_processor=_skeleton_processor(solved),
        output_dir=out,
        run_clinical=False,
    )

    assert report.calibration.solved is False  # loaded, not solved
    assert report.calibration.n_cameras == 3
    assert report.fusion_3d.n_active_keypoints == 12
    assert report.fusion_3d.reproj_err_px_median < 1.0  # calib matches projection exactly


# ---------------------------------------------------------------------------
# Clinical metrics (R) — skip-aware
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_R, reason="R or required R packages unavailable")
def test_run_validation_runs_clinical_pipeline(rendered_session, tmp_path: pathlib.Path):
    session_dir, solved = rendered_session
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"

    report = run_validation(
        session_dir,
        calibration=calib_json,
        camera_processor=_skeleton_processor(solved),
        output_dir=out,
        run_clinical=True,
    )

    agr = report.agreement
    assert agr.clinical_csv_produced is True
    assert any("clinical_3d" in name for name in agr.clinical_outputs)
    # R writes its outputs next to world3d.csv.
    assert list((out / session_dir.name).glob("*_clinical_3d*.csv"))


def test_self_consistency_runs_without_r(rendered_session, tmp_path: pathlib.Path):
    """Self-consistency surrogates do not depend on R being installed."""
    session_dir, solved = rendered_session
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"

    report = run_validation(
        session_dir,
        calibration=calib_json,
        camera_processor=_skeleton_processor(solved),
        output_dir=out,
        run_clinical=False,
    )
    assert np.isfinite(report.agreement.mean_bone_length_cv)
    assert report.agreement.bone_length_cv  # at least one arm bone measured


# ---------------------------------------------------------------------------
# CLI + reuse-existing-CSVs branch
# ---------------------------------------------------------------------------


def test_cli_reuses_existing_csvs_and_writes_reports(rendered_session, tmp_path: pathlib.Path):
    session_dir, solved = rendered_session
    calib_json = tmp_path / "calibration.json"
    save_calibration(solved, calib_json)
    out = tmp_path / "out"
    # Pre-write per-camera CSVs so the harness (no processor) reuses them.
    _prewrite_csvs(out / session_dir.name, solved)

    report_json = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    rc = main(
        [
            "--session-dir",
            str(session_dir),
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
    assert payload["fusion_3d"]["n_active_keypoints"] == 12
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
# Verdict grading (Session 1B) — constructed reports, no harness run
# ---------------------------------------------------------------------------


def _good_report() -> ValidationReport:
    """A minimal all-PASS report; tests mutate single fields to grade them.

    Only the graded fields matter, so a single camera entry stands in for
    the rig (``n_views_median`` carries the cross-camera redundancy that
    the verdict actually grades).
    """
    return ValidationReport(
        session_id="synthetic",
        schema_version=2,
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
            total_frames=24,
            mean_detection_rate=1.0,
            cameras=[
                CameraTracking(
                    name="cam1",
                    n_frames=8,
                    detection_rate=1.0,
                    low_confidence_fraction=0.0,
                    dropped_frames=0,
                )
            ],
        ),
        fusion_3d=Fusion3DSection(
            n_frames_fused=8,
            n_active_keypoints=12,
            reproj_err_px_median=3.0,
            reproj_err_px_p95=10.0,
            reproj_err_px_max=12.0,
            n_views_median=3.0,
            n_views_min=3,
            cheirality_violation_rate=0.0,
            unfused_keypoint_fraction=0.0,
        ),
        timing=TimingSection(
            device="NPU",
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
