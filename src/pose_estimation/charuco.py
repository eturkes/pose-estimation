"""ChArUco-based multi-camera calibration solver.

Consumes a calibration *session* (same directory layout as recording
sessions — ``cam*.mp4`` + optional ``session.json``; see
``.claude/tech/multicam.md``) in which a printed ChArUco board was moved
through the working volume, and produces a ``SessionCalibration``:

1. **Detection** — ``detect_charuco_corners`` runs
   ``cv2.aruco.CharucoDetector`` over every frame of each camera's
   video.  Frame indices are *logical* (raw minus ``sync_offset``) so
   detections align across cameras exactly like 2D keypoint fusion.
2. **Intrinsics** — per camera, ``board.matchImagePoints`` +
   ``cv2.calibrateCamera`` over ≤ ``max_frames`` time-uniform frames.
   (The legacy ``cv2.aruco.calibrateCameraCharuco*`` API is absent from
   opencv-python-headless ≥ 4.7; this is the modern equivalent.)
3. **Extrinsics** — every other camera is solved *directly against the
   world-frame camera* via ``cv2.stereoCalibrate`` (intrinsics fixed)
   on frames where both see the board.  Chained topologies (A↔B↔C
   without A↔C overlap) are unsupported — record with pairwise overlap.
4. **Diagnostics** — a global reprojection RMS: per logical frame the
   board pose is anchored via the world-frame camera (fallback: any
   solved camera), reprojected into every detecting camera.

``calibration.py`` stays the cv2-free data layer; this module owns the
cv2.aruco solve path.  See ``.claude/tech/calibration.md``.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence  # used in cast() strings
from typing import cast

import cv2
import numpy as np

from ._types import CameraCalibration, SessionCalibration
from .calibration import CALIBRATION_FORMAT_VERSION, CalibrationError, utc_timestamp
from .multicam import discover_session

CHARUCO_SQUARES_DEFAULT: tuple[int, int] = (6, 9)
"""Default board layout: (squares_x, squares_y)."""

CHARUCO_SQUARE_SIZE_M_DEFAULT = 0.04
"""Default chessboard square side length in metres."""

CHARUCO_MARKER_SIZE_M_DEFAULT = 0.03
"""Default ArUco marker side length in metres (must be < square size)."""

CHARUCO_DICTIONARY_DEFAULT = cv2.aruco.DICT_4X4_250
"""Default ArUco dictionary id (6x9 board needs 27 ≤ 250 markers)."""

CHARUCO_SOLVER_TAG = "opencv-charuco"
"""``solver`` provenance string written into solved calibrations."""

MIN_CORNERS_PER_FRAME = 6
"""A frame contributes only when ≥ this many charuco corners detect."""

MIN_INTRINSIC_FRAMES = 8
"""Minimum usable frames per camera for an intrinsics solve."""

MIN_SHARED_FRAMES = 5
"""Minimum board-sharing frames per camera pair for an extrinsics solve."""

MIN_SHARED_CORNERS = 6
"""Minimum intersected corner ids for a frame to count as *shared*."""

MAX_SOLVE_FRAMES = 50
"""Time-uniform subsample cap fed to each calibrateCamera/stereoCalibrate."""


@dataclasses.dataclass(frozen=True)
class CharucoDetection:
    """Charuco corners detected in one video frame.

    ``frame_idx`` is the *logical* index (raw - ``sync_offset``), so
    equal indices across cameras refer to the same instant.
    """

    frame_idx: int
    corner_ids: np.ndarray  # (n,) int32
    corners: np.ndarray  # (n, 2) float32 pixel coords


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


def make_charuco_board(
    squares: tuple[int, int] = CHARUCO_SQUARES_DEFAULT,
    square_size_m: float = CHARUCO_SQUARE_SIZE_M_DEFAULT,
    marker_size_m: float = CHARUCO_MARKER_SIZE_M_DEFAULT,
    dictionary: int = CHARUCO_DICTIONARY_DEFAULT,
) -> cv2.aruco.CharucoBoard:
    """Build the ChArUco board object shared by solve/capture/print.

    The physical print must match these dimensions exactly — verify the
    printed square size with a ruler before recording.
    """
    if marker_size_m >= square_size_m:
        raise CalibrationError(
            f"marker_size_m={marker_size_m} must be smaller than square_size_m={square_size_m}"
        )
    dic = cv2.aruco.getPredefinedDictionary(dictionary)
    return cv2.aruco.CharucoBoard(squares, square_size_m, marker_size_m, dic)


def render_charuco_board(
    board: cv2.aruco.CharucoBoard,
    *,
    px_per_square: int = 240,
    margin_squares: float = 0.25,
) -> np.ndarray:
    """Render *board* to a grayscale image for printing or synthesis.

    ``margin_squares`` adds a white quiet zone (fraction of one square)
    around the pattern; use 0 for synthetic-test renders so pixel ↔
    metre mapping is exactly ``px_per_square / square_size``.
    """
    sx, sy = board.getChessboardSize()
    margin_px = round(px_per_square * margin_squares)
    size = (sx * px_per_square + 2 * margin_px, sy * px_per_square + 2 * margin_px)
    return board.generateImage(size, marginSize=margin_px, borderBits=1)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_charuco_corners(
    video_path: str | pathlib.Path,
    board: cv2.aruco.CharucoBoard,
    *,
    sync_offset: int = 0,
    min_corners: int = MIN_CORNERS_PER_FRAME,
) -> tuple[list[CharucoDetection], tuple[int, int]]:
    """Detect charuco corners in every frame of *video_path*.

    Returns ``(detections, (width, height))``.  Frames before
    ``sync_offset`` are skipped; surviving frames carry logical indices.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CalibrationError(f"cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = cv2.aruco.CharucoDetector(board)
    detections: list[CharucoDetection] = []
    raw_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            logical = raw_idx - sync_offset
            raw_idx += 1
            if logical < 0:
                continue
            corners, ids, _mk_corners, _mk_ids = detector.detectBoard(frame)
            if corners is None or ids is None or len(corners) < min_corners:
                continue
            detections.append(
                CharucoDetection(
                    frame_idx=logical,
                    corner_ids=np.asarray(ids, dtype=np.int32).reshape(-1),
                    corners=np.asarray(corners, dtype=np.float32).reshape(-1, 2),
                )
            )
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise CalibrationError(f"invalid frame size {width}x{height} for video: {video_path}")
    return detections, (width, height)


def _subsample(items: list, max_items: int) -> list:
    """Pick ≤ *max_items* entries spread uniformly across *items*."""
    if len(items) <= max_items:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, max_items).round().astype(int))
    return [items[i] for i in idx]


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def _solve_intrinsics(
    detections: list[CharucoDetection],
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-camera intrinsics: returns ``(K, distortion, rms_px)``."""
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    for det in detections:
        corners = np.asarray(det.corners, dtype=np.float32).reshape(-1, 1, 2)
        ids = np.asarray(det.corner_ids, dtype=np.int32).reshape(-1, 1)
        # cv2 stubs type detectedCorners as Sequence[MatLike]; runtime
        # accepts the single (n, 1, 2) array detectBoard produced.
        obj, img = board.matchImagePoints(cast("Sequence[np.ndarray]", corners), ids)
        if obj is None or len(obj) < MIN_CORNERS_PER_FRAME:
            continue
        obj_points.append(np.asarray(obj, dtype=np.float32))
        img_points.append(np.asarray(img, dtype=np.float32))
    if len(obj_points) < MIN_INTRINSIC_FRAMES:
        raise CalibrationError(
            f"only {len(obj_points)} usable frames for intrinsics "
            f"(need ≥ {MIN_INTRINSIC_FRAMES}) — record more board views"
        )
    # The K/dist arguments are pure output storage here: without
    # CALIB_USE_INTRINSIC_GUESS OpenCV re-initialises both internally.
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, np.zeros((3, 3)), np.zeros(5)
    )
    return np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1), rms


def _shared_correspondences(
    dets_a: dict[int, CharucoDetection],
    dets_b: dict[int, CharucoDetection],
    board: cv2.aruco.CharucoBoard,
    *,
    min_shared_corners: int = MIN_SHARED_CORNERS,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Intersect detections by logical frame and corner id.

    Returns per-frame ``(object_points, image_points_a, image_points_b)``
    lists shaped for ``cv2.stereoCalibrate``.
    """
    chessboard = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    obj_frames: list[np.ndarray] = []
    img_a_frames: list[np.ndarray] = []
    img_b_frames: list[np.ndarray] = []
    for frame_idx in sorted(set(dets_a) & set(dets_b)):
        da, db = dets_a[frame_idx], dets_b[frame_idx]
        shared = np.intersect1d(da.corner_ids, db.corner_ids)
        if len(shared) < min_shared_corners:
            continue
        pos_a = {int(cid): i for i, cid in enumerate(da.corner_ids)}
        pos_b = {int(cid): i for i, cid in enumerate(db.corner_ids)}
        obj_frames.append(chessboard[shared].reshape(-1, 1, 3))
        img_a_frames.append(da.corners[[pos_a[int(c)] for c in shared]].reshape(-1, 1, 2))
        img_b_frames.append(db.corners[[pos_b[int(c)] for c in shared]].reshape(-1, 1, 2))
    return obj_frames, img_a_frames, img_b_frames


def _solve_extrinsics(
    world_dets: dict[int, CharucoDetection],
    cam_dets: dict[int, CharucoDetection],
    board: cv2.aruco.CharucoBoard,
    world_K: np.ndarray,
    world_dist: np.ndarray,
    cam_K: np.ndarray,
    cam_dist: np.ndarray,
    image_size: tuple[int, int],
    *,
    min_shared_frames: int = MIN_SHARED_FRAMES,
    max_frames: int = MAX_SOLVE_FRAMES,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Camera-from-world pose of *cam* relative to the world camera.

    Returns ``(rvec, tvec, rms_px)``.  ``stereoCalibrate``'s (R, T) maps
    world-camera coordinates into *cam* coordinates; because the world
    camera *is* the world frame, that is directly camera-from-world.
    """
    obj_f, img_w, img_c = _shared_correspondences(world_dets, cam_dets, board)
    if len(obj_f) < min_shared_frames:
        raise CalibrationError(
            f"only {len(obj_f)} frames share ≥ {MIN_SHARED_CORNERS} corners with the "
            f"world-frame camera (need ≥ {min_shared_frames}) — record views where "
            "both cameras see the board simultaneously"
        )
    keep = _subsample(list(range(len(obj_f))), max_frames)
    rms, *_imgK, R, T, _E, _F = cv2.stereoCalibrate(
        [obj_f[i] for i in keep],
        [img_w[i] for i in keep],
        [img_c[i] for i in keep],
        world_K,
        world_dist,
        cam_K,
        cam_dist,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rvec = cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].reshape(-1)
    return rvec, np.asarray(T, dtype=np.float64).reshape(-1), rms


def _global_reprojection_rms(
    per_camera_dets: dict[str, dict[int, CharucoDetection]],
    cameras: dict[str, CameraCalibration],
    board: cv2.aruco.CharucoBoard,
    world_frame: str,
) -> float:
    """RMS reprojection error across all cameras and logical frames.

    Per frame the board pose is anchored by solvePnP in the world-frame
    camera when it sees the board (fallback: first detecting camera in
    session order), lifted to world coordinates, then reprojected into
    every detecting camera.
    """
    chessboard = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    rotations = {
        name: cv2.Rodrigues(np.asarray(cam["rvec"], dtype=np.float64))[0]
        for name, cam in cameras.items()
    }
    sq_errors: list[float] = []
    all_frames = sorted({f for dets in per_camera_dets.values() for f in dets})
    for frame_idx in all_frames:
        seeing = [n for n in cameras if frame_idx in per_camera_dets[n]]
        anchor = world_frame if world_frame in seeing else seeing[0]
        det = per_camera_dets[anchor][frame_idx]
        ok, rvec_b, tvec_b = cv2.solvePnP(
            chessboard[det.corner_ids],
            det.corners,
            np.asarray(cameras[anchor]["K"], dtype=np.float64),
            np.asarray(cameras[anchor]["distortion"], dtype=np.float64),
        )
        if not ok:
            continue
        # board→anchor lifted to board→world: x_w = R_a^T (x_a - t_a)
        R_a, t_a = rotations[anchor], np.asarray(cameras[anchor]["tvec"], dtype=np.float64)
        R_bw = R_a.T @ cv2.Rodrigues(rvec_b)[0]
        t_bw = R_a.T @ (tvec_b.reshape(-1) - t_a)
        for name in seeing:
            d = per_camera_dets[name][frame_idx]
            R_bc = rotations[name] @ R_bw
            t_bc = rotations[name] @ t_bw + np.asarray(cameras[name]["tvec"], dtype=np.float64)
            projected, _ = cv2.projectPoints(
                chessboard[d.corner_ids],
                cv2.Rodrigues(R_bc)[0],
                t_bc,
                np.asarray(cameras[name]["K"], dtype=np.float64),
                np.asarray(cameras[name]["distortion"], dtype=np.float64),
            )
            residual = projected.reshape(-1, 2) - d.corners
            sq_errors.extend(np.sum(residual.astype(np.float64) ** 2, axis=1).tolist())
    if not sq_errors:
        return float("nan")
    return float(np.sqrt(np.mean(sq_errors)))


# ---------------------------------------------------------------------------
# Top-level solve
# ---------------------------------------------------------------------------


def solve_charuco(
    session_dir: str | pathlib.Path,
    *,
    board: cv2.aruco.CharucoBoard | None = None,
    world_frame: str | None = None,
    max_frames: int = MAX_SOLVE_FRAMES,
    min_corners: int = MIN_CORNERS_PER_FRAME,
    min_shared_frames: int = MIN_SHARED_FRAMES,
) -> SessionCalibration:
    """Solve a full multi-camera calibration from a ChArUco session.

    ``session_dir`` uses the standard session layout (``cam*.mp4`` +
    optional ``session.json`` whose ``sync_offset`` values are honoured).
    ``world_frame`` defaults to the session's first camera.  Every other
    camera must share board views with the world-frame camera directly.

    Raises ``CalibrationError`` with recording guidance when coverage
    is insufficient.  Returns an in-memory ``SessionCalibration``; the
    caller persists it via ``calibration.save_calibration``.
    """
    session = discover_session(session_dir)
    board = board if board is not None else make_charuco_board()
    names = session.camera_names()
    world = names[0] if world_frame is None else world_frame
    if world not in names:
        raise CalibrationError(f"world_frame={world!r} is not a session camera (have {names})")

    detections: dict[str, dict[int, CharucoDetection]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for cam in session.cameras:
        try:
            dets, size = detect_charuco_corners(
                cam.file, board, sync_offset=cam.sync_offset, min_corners=min_corners
            )
            intrinsics[cam.name] = _solve_intrinsics(_subsample(dets, max_frames), board, size)
        except CalibrationError as exc:
            raise CalibrationError(f"camera {cam.name!r}: {exc}") from exc
        detections[cam.name] = {d.frame_idx: d for d in dets}
        sizes[cam.name] = size
        print(
            f"[calib] {cam.name}: {len(dets)} frames with ≥{min_corners} corners, "
            f"intrinsics rms={intrinsics[cam.name][2]:.3f} px"
        )

    cameras: dict[str, CameraCalibration] = {}
    for name in names:
        K, dist, _rms = intrinsics[name]
        if name == world:
            rvec, tvec = np.zeros(3), np.zeros(3)
        else:
            try:
                rvec, tvec, pair_rms = _solve_extrinsics(
                    detections[world],
                    detections[name],
                    board,
                    intrinsics[world][0],
                    intrinsics[world][1],
                    K,
                    dist,
                    sizes[world],
                    min_shared_frames=min_shared_frames,
                    max_frames=max_frames,
                )
            except CalibrationError as exc:
                raise CalibrationError(f"camera {name!r}: {exc}") from exc
            print(f"[calib] {world}↔{name}: stereo rms={pair_rms:.3f} px")
        cameras[name] = CameraCalibration(
            name=name, resolution=sizes[name], K=K, distortion=dist, rvec=rvec, tvec=tvec
        )

    global_rms = _global_reprojection_rms(detections, cameras, board, world)
    print(f"[calib] global reprojection rms={global_rms:.3f} px")
    return SessionCalibration(
        format_version=CALIBRATION_FORMAT_VERSION,
        session_id=session.session_id,
        world_frame=world,
        cameras=cameras,
        reprojection_error_px=global_rms,
        solver=CHARUCO_SOLVER_TAG,
        solved_at=utc_timestamp(),
    )


__all__ = [
    "CHARUCO_DICTIONARY_DEFAULT",
    "CHARUCO_MARKER_SIZE_M_DEFAULT",
    "CHARUCO_SOLVER_TAG",
    "CHARUCO_SQUARES_DEFAULT",
    "CHARUCO_SQUARE_SIZE_M_DEFAULT",
    "CharucoDetection",
    "detect_charuco_corners",
    "make_charuco_board",
    "render_charuco_board",
    "solve_charuco",
]
