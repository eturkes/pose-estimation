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
from pose_estimation.validation import ValidationError, main, run_validation

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
