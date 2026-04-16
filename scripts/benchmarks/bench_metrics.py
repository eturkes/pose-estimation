"""Benchmarks for metrics.MetricsCollector and postprocess.savgol_smooth_csv."""

from __future__ import annotations

import csv
import pathlib
import tempfile

import numpy as np

from pose_estimation.metrics import (
    ConstraintDiagnostics,
    FrameDiagnostics,
    MetricsCollector,
    SmoothingDiagnostics,
)
from pose_estimation.postprocess import savgol_smooth_csv

from ._fixtures import (
    ARM_KP,
    BODY_KP,
    HAND_KP,
    make_body_landmarks,
    make_hand_landmarks,
    make_visibilities,
)
from ._harness import run_group


def _metrics_record_case(n_frames: int, n_kp: int, detail: bool):
    """Write *n_frames* records to a MetricsCollector then flush.

    All synthetic inputs are precomputed outside the hot loop so the
    timer captures MetricsCollector.record itself (file I/O included)
    rather than fixture construction.
    """
    bodies = [make_body_landmarks(n_kp=n_kp, seed=6000 + i) for i in range(n_frames)]
    viss = [make_visibilities(n_kp, seed=6500 + i) for i in range(n_frames)]
    hands_L = [make_hand_landmarks(seed=7000 + i) for i in range(n_frames)]
    hands_R = [make_hand_landmarks(seed=7500 + i) for i in range(n_frames)]

    frame_diags = [
        FrameDiagnostics(
            body_detected=True,
            body_det_score=0.85,
            n_hands_real=2,
            raw_body_landmarks=[bodies[i]],
            raw_hand_landmarks=[hands_L[i], hands_R[i]],
            raw_body_visibilities=[viss[i]],
        )
        for i in range(n_frames)
    ]
    smooth_diag = SmoothingDiagnostics(
        body_smooth_delta_px=12.5,
        hand_smooth_deltas_px=[8.1, 9.4],
        body_carry=False,
        body_carry_frames=0,
        hand_carry_flags=[False, False],
    )
    constraint_diag = ConstraintDiagnostics(
        bone_correction_px=3.2,
        angle_corrections_n=1,
    )

    def _run():
        with tempfile.TemporaryDirectory() as tmp:
            collector = MetricsCollector(tmp, "synthetic.mp4", detail=detail)
            for i in range(n_frames):
                collector.record(
                    frame_idx=i,
                    timestamp_sec=i / 30.0,
                    person_idx=0,
                    body_lm_smooth=bodies[i],
                    body_vis=viss[i],
                    hand_L_smooth=hands_L[i],
                    hand_R_smooth=hands_R[i],
                    frame_diag=frame_diags[i],
                    smooth_diag=smooth_diag,
                    constraint_diag=constraint_diag,
                    hand_L_flag=0.9,
                    hand_R_flag=0.88,
                    match_dist_L=12.0,
                    match_dist_R=14.0,
                    inference_ms=18.0,
                )
            collector.flush()

    return _run


def _jitter_case(n_kp: int):
    prev_store: dict[int, np.ndarray] = {0: make_body_landmarks(n_kp=n_kp, seed=8000)}
    current = make_body_landmarks(n_kp=n_kp, seed=8100)

    def _run():
        MetricsCollector._jitter(current, prev_store, 0)

    return _run


def _savgol_csv_case(n_rows: int, n_cols: int, window: int):
    """Write a synthetic landmark CSV then run savgol_smooth_csv on it."""
    tmp = tempfile.mkdtemp(prefix="bench_savgol_")
    in_path = pathlib.Path(tmp) / "in.csv"
    out_path = pathlib.Path(tmp) / "out.csv"

    header = ["video", "frame_idx", "timestamp_sec", "person_idx"]
    coord_cols = []
    for kp in range(n_cols // 3):
        for suffix in ("_x", "_y", "_z"):
            coord_cols.append(f"body_kp{kp}{suffix}")
    header.extend(coord_cols)

    rng = np.random.default_rng(9000)
    with in_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        values = rng.normal(0.5, 0.05, (n_rows, len(coord_cols)))
        for i in range(n_rows):
            row = ["synthetic.mp4", i, round(i / 30.0, 4), 0, *[round(v, 6) for v in values[i]]]
            w.writerow(row)

    def _run():
        savgol_smooth_csv(in_path, out_path, window=window, polyorder=3)

    return _run


def build_cases():
    cases: list[tuple[str, object, dict]] = []

    # MetricsCollector.record end-to-end (includes file I/O).
    #
    # Only detail=False is exercised for body keypoints: passing
    # ``raw_body_landmarks`` (a list) through record → _write_detail triggers
    # a latent bug at metrics.py:332 ("raw_lm[kp, 0]" assumes ndarray, not
    # list).  Flagged in the benchmark report.
    for n_frames in (30, 120, 480):
        for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33")]:
            cases.append(
                (
                    "MetricsCollector.record (+flush)",
                    _metrics_record_case(n_frames, n_kp, detail=False),
                    {"frames": n_frames, "kp": label, "detail": False},
                )
            )

    # Jitter helper
    for n_kp, label in [(ARM_KP, "arm12"), (BODY_KP, "body33"), (HAND_KP, "hand21")]:
        cases.append(
            (
                "MetricsCollector._jitter",
                _jitter_case(n_kp),
                {"kp": label},
            )
        )

    # Savitzky-Golay post-processing at various CSV sizes
    for n_rows in (120, 600, 1800):
        for n_cols, label in [(36, "arm12_xyz"), (99, "body33_xyz")]:
            cases.append(
                (
                    "savgol_smooth_csv",
                    _savgol_csv_case(n_rows, n_cols, window=11),
                    {"rows": n_rows, "cols": label, "window": 11},
                )
            )

    return cases


def run(iters: int = 20, warmup: int = 2):
    # Low iteration count — these cases include file I/O and are expensive
    return run_group("metrics", build_cases(), iters=iters, warmup=warmup)
