#!/usr/bin/env python3
"""Command-line interface for camera-calibration management.

Subcommands:
    verify   — Load a calibration.json and print a summary.
    solve    — Solve calibration from a ChArUco recording session.
    board    — Render the ChArUco board to a PNG for printing.
    capture  — Live multi-camera capture: pygame grid with corner
               overlay; SPACE appends one synchronized frame per camera
               to per-camera videos (frame index = press index, so
               extrinsic pairing needs no sync offsets).

Standalone usage:
    pose-estimation-calibrate verify --calibration calib.json
    pose-estimation-calibrate board --output board.png
    pose-estimation-calibrate capture --session-dir videos/calib_01/ --devices 0,1,2
    pose-estimation-calibrate solve --session-dir videos/calib_01/ --output calib.json

The board defaults (6x9, 40 mm squares, 30 mm markers, DICT_4X4_250)
are shared by all subcommands; override consistently or the solve will
misinterpret the print.  See ``.claude/tech/calibration.md``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

from .calibration import (
    CALIBRATION_FILENAME,
    CalibrationError,
    load_calibration,
    save_calibration,
)
from .charuco import (
    CHARUCO_MARKER_SIZE_M_DEFAULT,
    CHARUCO_SQUARE_SIZE_M_DEFAULT,
    CHARUCO_SQUARES_DEFAULT,
    make_charuco_board,
    render_charuco_board,
    solve_charuco,
)
from .multicam import SessionError


def _print_summary(calib, source: str) -> None:
    print(f"calibration:  {source}")
    print(f"session_id:   {calib['session_id']}")
    print(f"world_frame:  {calib['world_frame']}")
    print(f"solver:       {calib.get('solver', '?')}")
    print(f"solved_at:    {calib.get('solved_at', '?')}")
    print(f"reproj_err:   {calib.get('reprojection_error_px', float('nan')):.4f} px")
    print(f"cameras ({len(calib['cameras'])}):")
    for name, cam in calib["cameras"].items():
        K = np.asarray(cam["K"], dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        tvec = np.asarray(cam["tvec"], dtype=np.float64)
        print(
            f"  {name:>8s}  res={cam['resolution'][0]}x{cam['resolution'][1]}  "
            f"f=({fx:.1f}, {fy:.1f})  c=({cx:.1f}, {cy:.1f})  "
            f"|t|={float(np.linalg.norm(tvec)):.3f} m"
        )


def _parse_squares(value: str) -> tuple[int, int]:
    """Parse a ``COLSxROWS`` board-layout string (e.g. ``6x9``)."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected COLSxROWS (e.g. 6x9), got {value!r}")
    try:
        sx, sy = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected COLSxROWS (e.g. 6x9), got {value!r}") from None
    if sx < 3 or sy < 3:
        raise argparse.ArgumentTypeError(f"board needs ≥ 3x3 squares, got {value!r}")
    return sx, sy


def _parse_devices(value: str) -> list[int]:
    """Parse a comma-separated camera-device index list (e.g. ``0,1,2``)."""
    try:
        devices = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers (e.g. 0,1,2), got {value!r}"
        ) from None
    if len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError(f"duplicate device index in {value!r}")
    return devices


def _board_from_args(args: argparse.Namespace):
    return make_charuco_board(
        squares=args.squares,
        square_size_m=args.square_size_m,
        marker_size_m=args.marker_size_m,
    )


def _add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--squares",
        type=_parse_squares,
        default=CHARUCO_SQUARES_DEFAULT,
        metavar="COLSxROWS",
        help="Board layout (default: %(default)s).",
    )
    parser.add_argument(
        "--square-size-m",
        type=float,
        default=CHARUCO_SQUARE_SIZE_M_DEFAULT,
        help="Chessboard square side in metres (default: %(default)s).",
    )
    parser.add_argument(
        "--marker-size-m",
        type=float,
        default=CHARUCO_MARKER_SIZE_M_DEFAULT,
        help="ArUco marker side in metres (default: %(default)s).",
    )


# ---------------------------------------------------------------------------
# verify / solve / board
# ---------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        calib = load_calibration(args.calibration)
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_summary(calib, args.calibration)
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    try:
        board = _board_from_args(args)
        calib = solve_charuco(
            args.session_dir,
            board=board,
            world_frame=args.world_frame,
            max_frames=args.max_frames,
        )
        save_calibration(calib, args.output)
    except (CalibrationError, SessionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote: {args.output}")
    _print_summary(calib, args.output)
    return 0


def _cmd_board(args: argparse.Namespace) -> int:
    try:
        board = _board_from_args(args)
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    img = render_charuco_board(board, px_per_square=args.px_per_square)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        print(f"ERROR: could not write {out}", file=sys.stderr)
        return 2
    sx, sy = args.squares
    w_mm, h_mm = sx * args.square_size_m * 1000, sy * args.square_size_m * 1000
    print(f"wrote: {out} ({img.shape[1]}x{img.shape[0]} px)")
    print(f"pattern size: {w_mm:.0f} x {h_mm:.0f} mm (plus white margin)")
    print(
        "print at 100% scale on rigid stock, then verify one square measures "
        f"{args.square_size_m * 1000:.0f} mm with a ruler."
    )
    return 0


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

_GRID_CELL_HEIGHT = 270
"""Per-camera preview height in the capture grid (px)."""


def _annotate_capture(frame: np.ndarray, corners, ids, name: str) -> np.ndarray:
    """Overlay detected charuco corners + camera label on a copy of *frame*."""
    out = frame.copy()
    n = 0 if corners is None else len(corners)
    if n:
        cv2.aruco.drawDetectedCornersCharuco(out, corners, ids, (0, 255, 0))
    cv2.putText(
        out,
        f"{name}: {n} corners",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if n else (0, 0, 255),
        2,
    )
    return out


def _compose_grid(frames: list[np.ndarray], cell_height: int = _GRID_CELL_HEIGHT) -> np.ndarray:
    """Scale frames to a common height and concatenate horizontally."""
    cells = []
    for frame in frames:
        h, w = frame.shape[:2]
        cells.append(cv2.resize(frame, (max(1, round(w * cell_height / h)), cell_height)))
    return np.hstack(cells)


def _cmd_capture(args: argparse.Namespace) -> int:
    devices = args.devices
    names = args.names.split(",") if args.names else [f"cam{i + 1}" for i in range(len(devices))]
    if len(names) != len(devices):
        print(f"ERROR: {len(names)} names for {len(devices)} devices", file=sys.stderr)
        return 2
    try:
        board = _board_from_args(args)
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    detector = cv2.aruco.CharucoDetector(board)
    session_dir = pathlib.Path(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    caps: list[cv2.VideoCapture] = []
    try:
        for dev, name in zip(devices, names, strict=True):
            cap = cv2.VideoCapture(dev)
            if args.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if args.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            if not cap.isOpened():
                print(f"ERROR: cannot open device {dev} ({name})", file=sys.stderr)
                return 2
            caps.append(cap)
        return _capture_loop(caps, names, detector, session_dir)
    finally:
        for cap in caps:
            cap.release()


def _capture_loop(
    caps: list,
    names: list[str],
    detector,
    session_dir: pathlib.Path,
) -> int:
    """Pygame preview loop; SPACE appends one frame per camera video."""
    import pygame

    pygame.init()
    screen: pygame.Surface | None = None
    writers: dict[str, cv2.VideoWriter] = {}
    n_saved = 0
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    clock = pygame.time.Clock()
    try:
        running = True
        while running:
            frames: list[np.ndarray] = []
            for cap, name in zip(caps, names, strict=True):
                ok, frame = cap.read()
                if not ok:
                    print(f"WARNING: frame grab failed on {name}", file=sys.stderr)
                    frame = np.zeros((_GRID_CELL_HEIGHT, _GRID_CELL_HEIGHT, 3), dtype=np.uint8)
                frames.append(frame)

            save_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        save_requested = True

            if save_requested:
                for frame, name in zip(frames, names, strict=True):
                    if name not in writers:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(session_dir / f"{name}.avi"), fourcc, 5.0, (w, h)
                        )
                        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
                        writers[name] = writer
                    writers[name].write(frame)
                n_saved += 1

            annotated = []
            for frame, name in zip(frames, names, strict=True):
                corners, ids, _mc, _mi = detector.detectBoard(frame)
                annotated.append(_annotate_capture(frame, corners, ids, name))
            grid = _compose_grid(annotated)
            rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            if screen is None or screen.get_size() != surface.get_size():
                screen = pygame.display.set_mode(surface.get_size())
            screen.blit(surface, (0, 0))
            pygame.display.set_caption(
                f"calibrate capture — saved {n_saved}  [SPACE] save  [Q] quit"
            )
            pygame.display.flip()
            clock.tick(30)
    finally:
        for writer in writers.values():
            writer.release()
        pygame.quit()
    if n_saved:
        print(f"captured {n_saved} synchronized frame(s) per camera into {session_dir}")
        print(f"next: pose-estimation-calibrate solve --session-dir {session_dir} --output ...")
    else:
        print("no frames captured", file=sys.stderr)
    return 0 if n_saved else 1


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose-estimation-calibrate",
        description="Camera-calibration management for multi-camera sessions.",
    )
    subs = parser.add_subparsers(dest="cmd", required=True, metavar="<cmd>")

    p_verify = subs.add_parser("verify", help="Load and summarise a calibration file.")
    p_verify.add_argument(
        "--calibration", required=True, help=f"Path to a {CALIBRATION_FILENAME}-format JSON file."
    )
    p_verify.set_defaults(func=_cmd_verify)

    p_solve = subs.add_parser(
        "solve",
        help="Solve calibration from a ChArUco recording session.",
    )
    p_solve.add_argument("--session-dir", required=True, help="Calibration recording session.")
    p_solve.add_argument(
        "--output",
        required=True,
        help=f"Destination path for {CALIBRATION_FILENAME}.",
    )
    p_solve.add_argument(
        "--world-frame",
        default=None,
        help="Camera name that defines the world origin (default: first camera).",
    )
    p_solve.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Max frames per calibrateCamera/stereoCalibrate call (default: %(default)s).",
    )
    _add_board_args(p_solve)
    p_solve.set_defaults(func=_cmd_solve)

    p_board = subs.add_parser(
        "board",
        help="Render the ChArUco board to an image for printing.",
    )
    p_board.add_argument("--output", required=True, help="Destination image path (PNG).")
    p_board.add_argument(
        "--px-per-square",
        type=int,
        default=240,
        help="Render resolution (default: %(default)s; 240 ≈ 150 dpi for 40 mm squares).",
    )
    _add_board_args(p_board)
    p_board.set_defaults(func=_cmd_board)

    p_capture = subs.add_parser(
        "capture",
        help="Live capture: SPACE saves one synchronized frame per camera.",
    )
    p_capture.add_argument(
        "--session-dir", required=True, help="Destination for the calibration recording."
    )
    p_capture.add_argument(
        "--devices",
        required=True,
        type=_parse_devices,
        help="Comma-separated cv2 device indices (e.g. 0,1,2).",
    )
    p_capture.add_argument(
        "--names",
        default=None,
        help="Comma-separated camera names matching --devices (default: cam1,cam2,...).",
    )
    p_capture.add_argument("--width", type=int, default=0, help="Requested capture width.")
    p_capture.add_argument("--height", type=int, default=0, help="Requested capture height.")
    _add_board_args(p_capture)
    p_capture.set_defaults(func=_cmd_capture)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
