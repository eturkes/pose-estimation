#!/usr/bin/env python3
"""Regenerate gap-free R clinical producer goldens from synthetic inputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import subprocess
import tempfile
from collections.abc import Sequence

import numpy as np

from pose_estimation.export import (
    BODY_KEYPOINT_NAMES,
    HAND_KEYPOINT_COUNT,
    make_csv_header,
    write_world3d_csv,
)
from pose_estimation.processing import TRACKING_BODY

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "tests" / "goldens" / "r_clinical"
_FPS = 30.0
_N_FRAMES = 91

# A group-disposition golden built only from well-formed input is header-only, so it
# pins no reason code, no row order and not the D05 partition. `2d_drop` carries one
# healthy group plus one group per short-input drop reason, which is what makes the
# artifact's own goldens non-vacuous.
_HEALTHY_GROUPS = ((0, _N_FRAMES),)
_MIXED_GROUPS = ((0, _N_FRAMES), (1, 3), (2, 20))
_2D_DATASETS = {
    "2d_idx": ("idx", _HEALTHY_GROUPS),
    "2d_cumsum": ("cumsum", _HEALTHY_GROUPS),
    "2d_csv4dp": ("csv4dp", _HEALTHY_GROUPS),
    "2d_drop": ("idx", _MIXED_GROUPS),
}
_DATASET_STEMS = (*_2D_DATASETS, "world3d")

_BODY_BASE = {
    "nose": (0.500, 0.125, -0.020),
    "left_eye_inner": (0.485, 0.115, -0.022),
    "left_eye": (0.475, 0.115, -0.022),
    "left_eye_outer": (0.465, 0.117, -0.021),
    "right_eye_inner": (0.515, 0.115, -0.022),
    "right_eye": (0.525, 0.115, -0.022),
    "right_eye_outer": (0.535, 0.117, -0.021),
    "left_ear": (0.445, 0.135, -0.010),
    "right_ear": (0.555, 0.135, -0.010),
    "mouth_left": (0.485, 0.155, -0.018),
    "mouth_right": (0.515, 0.155, -0.018),
    "left_shoulder": (0.385, 0.305, 0.000),
    "right_shoulder": (0.615, 0.305, 0.000),
    "left_elbow": (0.345, 0.405, -0.005),
    "right_elbow": (0.655, 0.405, 0.005),
    "left_wrist": (0.315, 0.505, -0.010),
    "right_wrist": (0.685, 0.505, 0.010),
    "left_pinky": (0.300, 0.490, -0.012),
    "right_pinky": (0.700, 0.490, 0.012),
    "left_index": (0.325, 0.475, -0.015),
    "right_index": (0.675, 0.475, 0.015),
    "left_thumb": (0.290, 0.495, -0.008),
    "right_thumb": (0.710, 0.495, 0.008),
    "left_hip": (0.430, 0.625, 0.010),
    "right_hip": (0.570, 0.625, 0.010),
    "left_knee": (0.435, 0.785, 0.015),
    "right_knee": (0.565, 0.785, 0.015),
    "left_ankle": (0.440, 0.935, 0.020),
    "right_ankle": (0.560, 0.935, 0.020),
    "left_heel": (0.435, 0.950, 0.015),
    "right_heel": (0.565, 0.950, 0.015),
    "left_foot_index": (0.420, 0.970, -0.005),
    "right_foot_index": (0.580, 0.970, -0.005),
}


def _minimum_jerk(u: float) -> float:
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _wrist_2d(side: str, u: float) -> tuple[float, float, float]:
    reach = _minimum_jerk(u)
    if side == "left":
        return (
            0.315 + 0.185 * reach,
            0.505 - 0.070 * math.sin(math.pi * u) + 0.018 * reach,
            -0.010 + 0.024 * math.sin(2 * math.pi * u),
        )
    return (
        0.685 - 0.142 * reach,
        0.505 - 0.052 * math.sin(math.pi * u + 0.18) + 0.012 * reach,
        0.010 + 0.017 * math.sin(2 * math.pi * u + 0.31),
    )


def _body_point_2d(name: str, u: float) -> tuple[float, float, float]:
    side = "left" if name.startswith("left_") else "right"
    trunk_shift = 0.006 * math.sin(2 * math.pi * u)
    if name in {"left_wrist", "right_wrist"}:
        return _wrist_2d(side, u)
    if name in {"left_elbow", "right_elbow"}:
        shoulder = _BODY_BASE[f"{side}_shoulder"]
        wrist = _wrist_2d(side, u)
        bend = -0.040 if side == "left" else 0.040
        return (
            shoulder[0] + 0.54 * (wrist[0] - shoulder[0]) + bend,
            shoulder[1] + 0.54 * (wrist[1] - shoulder[1]) + 0.018,
            shoulder[2] + 0.54 * (wrist[2] - shoulder[2]),
        )
    if name in {
        "left_index",
        "right_index",
        "left_pinky",
        "right_pinky",
        "left_thumb",
        "right_thumb",
    }:
        wrist = _wrist_2d(side, u)
        lateral = -1.0 if side == "left" else 1.0
        offsets = {
            "index": (0.014 * lateral, -0.033, -0.004 * lateral),
            "pinky": (-0.014 * lateral, -0.024, 0.003 * lateral),
            "thumb": (-0.025 * lateral, -0.005, 0.006 * lateral),
        }
        keypoint = name.split("_", 1)[1]
        dx, dy, dz = offsets[keypoint]
        return wrist[0] + dx, wrist[1] + dy, wrist[2] + dz
    x, y, z = _BODY_BASE[name]
    if "shoulder" in name or "hip" in name:
        return x + trunk_shift, y + 0.003 * math.sin(math.pi * u), z
    return x + 0.35 * trunk_shift, y, z


def _hand_offset(side: str, index: int, u: float, scale: float) -> tuple[float, float, float]:
    lateral = -1.0 if side == "left" else 1.0
    if index == 0:
        return 0.0, 0.010 * scale, 0.0
    if index <= 4:
        finger, joint = -1.35, index
    elif index <= 8:
        finger, joint = -0.55, index - 4
    elif index <= 12:
        finger, joint = 0.0, index - 8
    elif index <= 16:
        finger, joint = 0.55, index - 12
    else:
        finger, joint = 1.05, index - 16
    side_phase = 0.0 if side == "left" else 0.4
    flex = 1.0 - (0.10 + 0.03 * finger) * math.sin(math.pi * u + side_phase)
    dx = lateral * (0.008 * finger + 0.0022 * joint * finger) * scale
    dy = -(0.008 + 0.009 * joint * flex) * scale
    dz = (0.0015 * joint * math.sin(2 * math.pi * u + finger + side_phase)) * scale
    return dx, dy, dz


def _timestamps(mode: str) -> list[float]:
    if mode == "cumsum":
        current = 0.0
        values = []
        for _ in range(_N_FRAMES):
            current += 1 / _FPS
            values.append(current)
        return values
    values = [frame / _FPS for frame in range(_N_FRAMES)]
    if mode == "csv4dp":
        return [round(value, 4) for value in values]
    return values


def _format_timestamp(value: float, mode: str) -> str:
    if mode == "csv4dp":
        return f"{value:.4f}"
    return format(value, ".17g")


def _write_2d_input(
    path: pathlib.Path,
    mode: str,
    groups: Sequence[tuple[int, int]] = _HEALTHY_GROUPS,
) -> None:
    header = make_csv_header(TRACKING_BODY)
    timestamps = _timestamps(mode)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for person_idx, n_frames in groups:
            for frame, timestamp in enumerate(timestamps[:n_frames]):
                # Short groups truncate the healthy trajectory rather than rescaling it,
                # so the drop reason is the only thing separating them from group 0.
                u = frame / (_N_FRAMES - 1)
                row = dict.fromkeys(header, "")
                row.update(
                    video="clinical_golden_2d.avi",
                    frame_idx=str(frame),
                    timestamp_sec=_format_timestamp(timestamp, mode),
                    person_idx=str(person_idx),
                )
                for name in BODY_KEYPOINT_NAMES:
                    x, y, z = _body_point_2d(name, u)
                    prefix = f"body_{name}"
                    row[f"{prefix}_x"] = f"{x:.9f}"
                    row[f"{prefix}_y"] = f"{y:.9f}"
                    row[f"{prefix}_z"] = f"{z:.9f}"
                    row[f"{prefix}_vis"] = "0.97"
                for side in ("left", "right"):
                    wrist = _wrist_2d(side, u)
                    for index in range(HAND_KEYPOINT_COUNT):
                        dx, dy, dz = _hand_offset(side, index, u, 1.0)
                        prefix = f"{side}_hand_{index}"
                        row[f"{prefix}_x"] = f"{wrist[0] + dx:.9f}"
                        row[f"{prefix}_y"] = f"{wrist[1] + dy:.9f}"
                        row[f"{prefix}_z"] = f"{wrist[2] + dz:.9f}"
                        row[f"{prefix}_conf"] = "0.96"
                writer.writerow(row)


def _wrist_3d(side: str, u: float) -> tuple[float, float, float]:
    reach = _minimum_jerk(u)
    if side == "left":
        return (
            -0.48 + 0.36 * reach,
            0.58 - 0.10 * math.sin(math.pi * u) + 0.025 * reach,
            2.18 + 0.075 * math.sin(2 * math.pi * u),
        )
    return (
        0.48 - 0.28 * reach,
        0.58 - 0.075 * math.sin(math.pi * u + 0.2) + 0.018 * reach,
        2.18 + 0.052 * math.sin(2 * math.pi * u + 0.35),
    )


def _body_point_3d(name: str, u: float) -> tuple[float, float, float]:
    side = "left" if name.startswith("left_") else "right"
    lateral = -1.0 if side == "left" else 1.0
    fixed = {
        "left_shoulder": (-0.20, 0.40, 2.18),
        "right_shoulder": (0.20, 0.40, 2.18),
        "left_hip": (-0.15, 0.90, 2.00),
        "right_hip": (0.15, 0.90, 2.00),
    }
    if name in fixed:
        x, y, z = fixed[name]
        return x + 0.008 * math.sin(2 * math.pi * u), y, z
    if name in {"left_wrist", "right_wrist"}:
        return _wrist_3d(side, u)
    if name in {"left_elbow", "right_elbow"}:
        shoulder = fixed[f"{side}_shoulder"]
        wrist = _wrist_3d(side, u)
        return (
            shoulder[0] + 0.52 * (wrist[0] - shoulder[0]) + 0.08 * lateral,
            shoulder[1] + 0.52 * (wrist[1] - shoulder[1]) + 0.04,
            shoulder[2] + 0.52 * (wrist[2] - shoulder[2]),
        )
    if name in {
        "left_index",
        "right_index",
        "left_pinky",
        "right_pinky",
        "left_thumb",
        "right_thumb",
    }:
        wrist = _wrist_3d(side, u)
        offsets = {
            "index": (0.030 * lateral, -0.070, -0.010 * lateral),
            "pinky": (-0.030 * lateral, -0.052, 0.008 * lateral),
            "thumb": (-0.052 * lateral, -0.010, 0.012 * lateral),
        }
        dx, dy, dz = offsets[name.split("_", 1)[1]]
        return wrist[0] + dx, wrist[1] + dy, wrist[2] + dz
    index = BODY_KEYPOINT_NAMES.index(name)
    return (
        lateral * (0.03 + 0.006 * index),
        0.16 + 0.012 * index,
        2.08 + 0.002 * (index % 5),
    )


def _write_world3d_input(path: pathlib.Path) -> None:
    names = [f"body_{name}" for name in BODY_KEYPOINT_NAMES] + [
        f"{side}_hand_{index}" for side in ("left", "right") for index in range(HAND_KEYPOINT_COUNT)
    ]
    index_by_name = {name: index for index, name in enumerate(names)}
    frames = []
    for frame in range(_N_FRAMES):
        u = frame / (_N_FRAMES - 1)
        world = np.empty((len(names), 3), dtype=np.float64)
        for name in BODY_KEYPOINT_NAMES:
            world[index_by_name[f"body_{name}"]] = _body_point_3d(name, u)
        for side in ("left", "right"):
            wrist = _wrist_3d(side, u)
            for index in range(HAND_KEYPOINT_COUNT):
                dx, dy, dz = _hand_offset(side, index, u, 2.5)
                world[index_by_name[f"{side}_hand_{index}"]] = (
                    wrist[0] + dx,
                    wrist[1] + dy,
                    wrist[2] + dz,
                )
        diagnostics = {
            "confidence": np.full(len(names), 0.96),
            "reprojection_error_px": np.full(len(names), 0.4),
            "candidate_n_views": np.full(len(names), 3),
            "n_views": np.full(len(names), 3),
            "cheirality_ok": np.ones(len(names)),
            "triangulation_angle_deg": np.full(len(names), 12.0),
        }
        frames.append((frame, frame / _FPS, world, diagnostics))
    write_world3d_csv(path, "clinical_golden_world3d", names, frames)


def _run_clinical(input_path: pathlib.Path) -> None:
    environment = os.environ.copy()
    environment.update(LC_ALL="C", LANG="C", TZ="UTC")
    result = subprocess.run(
        ["Rscript", str(_CLINICAL_R), str(input_path)],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"clinical_features.R failed for {input_path.name}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _expected_outputs(stem: str) -> tuple[str, ...]:
    if stem == "world3d":
        return (
            "world3d_clinical_3d.csv",
            "world3d_clinical_3d_windows.csv",
            "world3d_clinical_3d_window_qc.csv",
            "world3d_clinical_3d_group_qc.csv",
        )
    return (
        f"{stem}_clinical.csv",
        f"{stem}_clinical_windows.csv",
        f"{stem}_clinical_group_qc.csv",
    )


def regenerate(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Regenerate every golden atomically into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[pathlib.Path] = []
    with tempfile.TemporaryDirectory(prefix="r-clinical-goldens-") as temporary:
        staging = pathlib.Path(temporary)
        for stem in _DATASET_STEMS:
            input_path = staging / f"{stem}.csv"
            if stem == "world3d":
                _write_world3d_input(input_path)
            else:
                mode, groups = _2D_DATASETS[stem]
                _write_2d_input(input_path, mode, groups)
            _run_clinical(input_path)
            for filename in _expected_outputs(stem):
                source = staging / filename
                destination = output_dir / filename
                pending = destination.with_suffix(f"{destination.suffix}.tmp")
                pending.write_bytes(source.read_bytes())
                pending.replace(destination)
                generated.append(destination)
    return generated


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Write files to this directory (default: committed golden directory).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for path in regenerate(args.output_dir.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
