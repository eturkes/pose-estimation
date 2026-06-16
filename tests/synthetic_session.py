"""Shared synthetic 3-camera session builders for the validation suite.

Footage-independent fixtures (roadmap Phase 1) live here so both the
end-to-end harness tests (``test_validation.py``) and the failure-mode
suite (``test_validation_failuremodes.py``) grow from one source:

- **Calibration session** — a ChArUco board rendered into MJPG videos
  (3x supersample + INTER_AREA, mirroring ``test_charuco``) that
  ``solve_charuco`` resolves for real.
- **2D tracking output** — a symmetric 12-keypoint "arm" skeleton
  projected into each calibrated camera and written as per-camera CSVs
  (mirroring ``test_multicam``), so fusion -> ``world3d.csv`` ->
  self-consistency runs deterministically without live inference.

The expensive render+solve is exposed as the ``rendered_session``
fixture in ``conftest.py``; everything callable here is a plain helper
so individual tests can perturb the inputs (drop a camera, perturb
extrinsics, scale confidences, occlude a region) before re-running.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
import pytest

from pose_estimation._types import CameraCalibration, SessionCalibration
from pose_estimation.charuco import make_charuco_board, render_charuco_board
from pose_estimation.export import frame_to_rows, open_csv_writer, wrist_to_side
from pose_estimation.processing import TRACKING_HANDS_ARMS
from pose_estimation.triangulation import project_points

# ---------------------------------------------------------------------------
# ChArUco calibration-session render (mirrors test_charuco)
# ---------------------------------------------------------------------------

BOARD = make_charuco_board()
_SX, _SY = BOARD.getChessboardSize()
SQ = 0.04
BOARD_W_M, BOARD_H_M = _SX * SQ, _SY * SQ
BOARD_IMG = render_charuco_board(BOARD, px_per_square=100, margin_squares=0)
BG_GRAY = 180

# Ground-truth rig: cam1 = world; cam2/cam3 yawed toward the volume.
GT: dict[str, dict[str, Any]] = {
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


def board_poses(n: int, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
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


def render_view(
    K: np.ndarray,
    rvec_cam: np.ndarray,
    tvec_cam: np.ndarray,
    rvec_board: np.ndarray,
    tvec_board: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Render the board seen by a camera; plain background if off-view."""
    w, h = size
    blank = np.full((h, w), BG_GRAY, dtype=np.uint8)
    corners_m = np.array(
        [[0, 0, 0], [BOARD_W_M, 0, 0], [BOARD_W_M, BOARD_H_M, 0], [0, BOARD_H_M, 0]]
    )
    centred = corners_m - np.array([BOARD_W_M / 2, BOARD_H_M / 2, 0.0])
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
    bh, bw = BOARD_IMG.shape[:2]
    src = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
    ss = 3  # supersample: anti-alias marker interiors
    H = cv2.getPerspectiveTransform(src, proj.astype(np.float32) * ss)
    canvas = cv2.warpPerspective(
        BOARD_IMG,
        H,
        (w * ss, h * ss),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BG_GRAY,
    )
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def write_video(path: pathlib.Path, frames: list[np.ndarray], size: tuple[int, int]) -> bool:
    """Write grayscale *frames* as an MJPG/AVI video; True on success."""
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, size)
    if not writer.isOpened():
        return False
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def render_calibration_session(session_dir: pathlib.Path, *, n_poses: int = 60) -> None:
    """Render the full 3-camera ChArUco calibration session into *session_dir*.

    Skips the test via ``pytest.skip`` if the MJPG/AVI codec is missing.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    poses = board_poses(n_poses)
    for name, gt in GT.items():
        frames = [
            render_view(gt["K"], gt["rvec"], gt["tvec"], rb, tb, gt["size"]) for rb, tb in poses
        ]
        if not write_video(session_dir / f"{name}.avi", frames, gt["size"]):
            pytest.skip("MJPG/AVI codec unavailable on this host")


# ---------------------------------------------------------------------------
# Synthetic subject skeleton (mirrors test_multicam projected CSVs)
# ---------------------------------------------------------------------------

# 12 "arm" keypoints (hands-arms mode), left/right symmetric, in the rig's
# co-visible working volume (z ~ 1 m).  Order = export.ARM_KEYPOINT_NAMES.
SKEL_BASE = np.array(
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
N_SUBJECT_FRAMES = 8
DEFAULT_VELOCITY = np.array([0.004, 0.002, 0.003])  # metres/frame (slow, in-frame)


def skel_world(frame_idx: int, velocity: np.ndarray = DEFAULT_VELOCITY) -> np.ndarray:
    """Rigid skeleton translating linearly over time (constant bone lengths)."""
    return SKEL_BASE + np.asarray(velocity, dtype=np.float64) * frame_idx


def write_skeleton_csv(
    csv_path: pathlib.Path,
    camera: CameraCalibration,
    *,
    confidence: float = 0.95,
    occlude: tuple[int, ...] = (),
    occlude_frames: tuple[int, ...] | None = None,
    zero_conf: tuple[int, ...] = (),
    frames: tuple[int, ...] | None = None,
    velocity: np.ndarray = DEFAULT_VELOCITY,
    project_with: CameraCalibration | None = None,
    noise_px: float = 0.0,
    noise_seed: int = 0,
) -> None:
    """Project the moving skeleton into *camera* and write its per-camera CSV.

    Hooks for failure-mode injection (all default to the clean path):

    - ``confidence`` — per-keypoint visibility written into the CSV; values
      below the harness confidence floor drive ``low_confidence_fraction``.
    - ``occlude`` — keypoint indices to NaN out (a camera not seeing a body
      region), so fusion loses those views as *missing coordinates*.
    - ``occlude_frames`` — restrict ``occlude`` to these frame indices
      (``None`` = every frame); a region that drops out partway tips
      fused keypoints from active-in-some-frames to NaN-in-others.
    - ``zero_conf`` — keypoint indices whose *confidence* is forced to 0,
      exercising fusion's confidence-validity gate (distinct axis from
      ``occlude``'s coordinate gate) — both route to NaN, never garbage.
    - ``frames`` — emit only these raw frame indices (``None`` = all),
      modelling a camera that dropped a block of frames.
    - ``velocity`` — per-frame skeleton translation (metres); a faster
      subject makes a sync_offset desync reproject further off.
    - ``project_with`` — project through a *different* calibration than the
      one fusion later uses (miscalibration / degenerate-geometry probes);
      defaults to *camera* (self-consistent).
    - ``noise_px`` — std of Gaussian pixel noise added to the projection
      (amplified into 3D instability under degenerate geometry).
    """
    width, height = camera["resolution"]
    src_cam = project_with if project_with is not None else camera
    emit = range(N_SUBJECT_FRAMES) if frames is None else frames
    rng = np.random.default_rng(noise_seed)
    fh, writer = open_csv_writer(csv_path, tracking=TRACKING_HANDS_ARMS)
    try:
        for f in emit:
            px = project_points(skel_world(f, velocity), src_cam)
            if noise_px > 0.0:
                px = px + rng.normal(0.0, noise_px, size=px.shape)
            if occlude and (occlude_frames is None or f in occlude_frames):
                px[list(occlude)] = np.nan
            lm = np.concatenate([px, np.zeros((len(px), 1))], axis=1)
            vis = np.full(len(px), confidence)
            if zero_conf:
                vis[list(zero_conf)] = 0.0
            for row in frame_to_rows(
                "v", f, f / 30.0, height, width, [lm], [vis], [], [], tracking=TRACKING_HANDS_ARMS
            ):
                writer.writerow(row)
    finally:
        fh.close()


def skeleton_processor(calib: SessionCalibration, **csv_kwargs: Any):
    """camera_processor that writes a projected-skeleton CSV per camera.

    Extra keyword arguments are forwarded to :func:`write_skeleton_csv`
    (e.g. ``confidence=0.1`` for the low-confidence failure mode).  Per-
    camera overrides are supported via ``per_camera={name: {...}}``.
    """
    per_camera: dict[str, dict[str, Any]] = csv_kwargs.pop("per_camera", {})

    def _proc(
        *,
        source: str,
        output_csv: pathlib.Path,
        output_diag: pathlib.Path,
        video_name: str,
        **_kw: Any,
    ) -> None:
        name = pathlib.Path(output_csv).stem
        kwargs = {**csv_kwargs, **per_camera.get(name, {})}
        write_skeleton_csv(pathlib.Path(output_csv), calib["cameras"][name], **kwargs)

    return _proc


def prewrite_csvs(session_out: pathlib.Path, calib: SessionCalibration) -> None:
    """Pre-write per-camera CSVs so the harness reuses them (CLI path)."""
    session_out.mkdir(parents=True, exist_ok=True)
    for name, camera in calib["cameras"].items():
        write_skeleton_csv(session_out / f"{name}.csv", camera)


# QA needs a *fully detected* subject clip (arms AND hands) so its
# per-camera 2D detection rate reflects a real hands-arms capture (~1.0),
# unlike the arms-only fusion fixture above (hands left NaN by design).
WRIST_SIDE = wrist_to_side(TRACKING_HANDS_ARMS)


def write_full_skeleton_csv(csv_path: pathlib.Path, camera: CameraCalibration) -> None:
    """Project the arm skeleton AND finite hand keypoints into *camera*.

    Every keypoint in the hands-arms schema is finite, so the QA gate's
    detection-rate metric sees a fully-tracked clip.  Hand geometry is
    synthetic (a ramp off each wrist) — QA never fuses the subject, so
    only finiteness matters here.
    """
    width, height = camera["resolution"]
    fh, writer = open_csv_writer(csv_path, tracking=TRACKING_HANDS_ARMS)
    try:
        for f in range(N_SUBJECT_FRAMES):
            px = project_points(skel_world(f), camera)
            lm = np.concatenate([px, np.zeros((len(px), 1))], axis=1)
            vis = np.full(len(px), 0.95)
            hands, matches = [], []
            for hand_idx, (wrist_kp, _side) in enumerate(sorted(WRIST_SIDE.items())):
                base = px[min(wrist_kp, len(px) - 1)]
                ramp = np.arange(21, dtype=np.float64)
                hand = np.stack([base[0] + ramp, base[1] + ramp, np.zeros(21)], axis=1)
                hands.append(hand)
                matches.append((0, wrist_kp, hand_idx))
            for row in frame_to_rows(
                "v",
                f,
                f / 30.0,
                height,
                width,
                [lm],
                [vis],
                hands,
                matches,
                tracking=TRACKING_HANDS_ARMS,
            ):
                writer.writerow(row)
    finally:
        fh.close()


def full_skeleton_processor(calib: SessionCalibration):
    """camera_processor writing a fully-detected (arms+hands) CSV per camera."""

    def _proc(*, source: str, output_csv: pathlib.Path, **_kw: Any) -> None:
        name = pathlib.Path(output_csv).stem
        write_full_skeleton_csv(pathlib.Path(output_csv), calib["cameras"][name])

    return _proc


def prewrite_full_csvs(
    session_dir: pathlib.Path, calib: SessionCalibration, output_dir: pathlib.Path
) -> None:
    """Pre-write fully-detected per-camera CSVs so --qa-only reuses them."""
    session_out = output_dir / session_dir.name
    session_out.mkdir(parents=True, exist_ok=True)
    for name, camera in calib["cameras"].items():
        write_full_skeleton_csv(session_out / f"{name}.csv", camera)


# Six board poses clustered dead-centre — a deliberately bad calibration
# capture: few views (below the intrinsic floor) confined to a small image
# region (low FOV coverage).
BAD_BOARD_POSES: list[tuple[np.ndarray, np.ndarray]] = [
    (np.array([0.02, -0.03, 0.01]), np.array([0.00, 0.00, 1.00])),
    (np.array([-0.03, 0.02, -0.02]), np.array([0.02, -0.01, 1.02])),
    (np.array([0.01, 0.04, 0.03]), np.array([-0.02, 0.01, 0.98])),
    (np.array([0.03, -0.02, -0.01]), np.array([0.01, 0.02, 1.03])),
    (np.array([-0.02, -0.04, 0.02]), np.array([-0.01, -0.02, 0.99])),
    (np.array([0.04, 0.01, -0.03]), np.array([0.00, 0.00, 1.01])),
]


def render_bad_capture(session_dir: pathlib.Path) -> None:
    """Render a sparse, centre-bound ChArUco capture with a desynced camera.

    cam3 is truncated to half its frames -> a frame-count parity (desync)
    violation; all cameras see only 6 centre-clustered board views ->
    below the intrinsic floor and low coverage.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    for name, gt in GT.items():
        frames = [
            render_view(gt["K"], gt["rvec"], gt["tvec"], rb, tb, gt["size"])
            for rb, tb in BAD_BOARD_POSES
        ]
        n = 3 if name == "cam3" else len(frames)
        if not write_video(session_dir / f"{name}.avi", frames[:n], gt["size"]):
            pytest.skip("MJPG/AVI codec unavailable on this host")


def r_available() -> bool:
    """True when Rscript and the clinical pipeline's R packages are present."""
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


HAS_R = r_available()
