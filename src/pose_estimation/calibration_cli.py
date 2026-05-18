#!/usr/bin/env python3
"""Command-line interface for camera-calibration management.

Subcommands:
    verify   — Load a calibration.json and print a summary (works now).
    solve    — Solve calibration from a ChArUco recording session (stub).
    capture  — Guided multi-camera calibration capture (stub).

Standalone usage:
    pose-estimation-calibrate verify --calibration calib.json
    pose-estimation-calibrate solve --session-dir videos/calib_session/ --output calib.json
    pose-estimation-calibrate capture --session-dir videos/calib_session/

See ``.claude/tech/calibration.md`` for the planned ChArUco workflow.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .calibration import (
    CALIBRATION_FILENAME,
    CalibrationError,
    load_calibration,
    solve_charuco,
)


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        calib = load_calibration(args.calibration)
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"calibration:  {args.calibration}")
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
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    try:
        solve_charuco(
            args.session_dir,
            square_size_m=args.square_size_m,
            world_frame=args.world_frame,
        )
    except NotImplementedError as exc:
        print(f"NOT YET WIRED: {exc}", file=sys.stderr)
        return 3
    print(f"wrote: {args.output}")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    # Capture is a future workflow; emit the design until wiring lands.
    print(
        "NOT YET WIRED: guided multi-camera calibration capture is a "
        "follow-up. The intended flow records N synchronized videos "
        "of a printed ChArUco board moved through the working volume. "
        f"Target session directory: {args.session_dir}",
        file=sys.stderr,
    )
    return 3


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
        help="Solve calibration from a ChArUco recording session (NOT YET WIRED).",
    )
    p_solve.add_argument("--session-dir", required=True, help="Calibration recording session.")
    p_solve.add_argument(
        "--output",
        required=True,
        help=f"Destination path for {CALIBRATION_FILENAME}.",
    )
    p_solve.add_argument(
        "--square-size-m",
        type=float,
        default=0.04,
        help="ChArUco square side length in metres (default: 0.04).",
    )
    p_solve.add_argument(
        "--world-frame",
        default="cam1",
        help="Camera name that defines the world origin (default: cam1).",
    )
    p_solve.set_defaults(func=_cmd_solve)

    p_capture = subs.add_parser(
        "capture",
        help="Guided multi-camera calibration capture (NOT YET WIRED).",
    )
    p_capture.add_argument(
        "--session-dir", required=True, help="Destination for the calibration recording."
    )
    p_capture.set_defaults(func=_cmd_capture)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
