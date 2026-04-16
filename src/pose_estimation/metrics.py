"""Per-frame metrics collection for optimization diagnostics.

Hooks into the pipeline to capture detection quality, smoothing
effectiveness, constraint activation, and tracking stability metrics.
Produces ``*_metrics.csv`` and optionally ``*_kp_detail.csv`` files
that the R analysis suite consumes.
"""

import csv
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Frame-level diagnostics returned by the instrumented pipeline
# ---------------------------------------------------------------------------

@dataclass
class FrameDiagnostics:
    """Per-frame diagnostic data collected during process_frame().

    Fields are populated incrementally by the pipeline stages; unset
    fields default to zero / empty so callers can safely ignore what
    they don't need.
    """
    # Detection counts
    body_detected: bool = False
    body_det_score: float = 0.0
    n_hands_real: int = 0
    n_hands_synthetic: int = 0
    n_hands_recrop: int = 0

    # Per-detection detail (list of dicts from process_frame hand_diag)
    hand_diag: list = field(default_factory=list)

    # Raw landmarks *before* smoothing (pixel coords)
    raw_body_landmarks: list = field(default_factory=list)
    raw_hand_landmarks: list = field(default_factory=list)
    raw_body_visibilities: list = field(default_factory=list)


@dataclass
class SmoothingDiagnostics:
    """Smoothing-stage diagnostics: raw-vs-smoothed deltas."""
    body_smooth_delta_px: float = 0.0
    hand_smooth_deltas_px: list = field(default_factory=list)
    body_carry: bool = False
    body_carry_frames: int = 0
    hand_carry_flags: list = field(default_factory=list)


@dataclass
class ConstraintDiagnostics:
    """Constraint-stage diagnostics: correction magnitudes."""
    bone_correction_px: float = 0.0
    angle_corrections_n: int = 0


# ---------------------------------------------------------------------------
# Metrics CSV schema
# ---------------------------------------------------------------------------

METRICS_FIELDS = [
    "video", "frame_idx", "timestamp_sec", "person_idx",
    # Detection
    "body_detected", "body_det_score",
    "n_hands_real", "n_hands_synthetic", "n_hands_recrop",
    # Confidence
    "body_vis_mean", "body_vis_min",
    "hand_L_flag", "hand_R_flag",
    # Jitter (frame-to-frame displacement in pixels, sum over keypoints)
    "body_jitter_px", "hand_L_jitter_px", "hand_R_jitter_px",
    # Smoothing delta (L2 between raw and smoothed, sum over keypoints)
    "body_smooth_delta_px", "hand_L_smooth_delta_px", "hand_R_smooth_delta_px",
    # Constraints
    "bone_correction_px", "angle_corrections_n",
    # Carry-forward
    "body_carry", "body_carry_frames",
    "hand_L_carry", "hand_R_carry",
    # Matching
    "hand_arm_match_dist_L", "hand_arm_match_dist_R",
    # FPS
    "inference_ms",
]

KP_DETAIL_FIELDS = [
    "frame_idx", "person_idx", "part", "kp_idx",
    "x_raw", "y_raw", "x_smooth", "y_smooth",
    "visibility", "jitter_px",
]


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulates per-frame metrics and writes CSV files at flush time.

    Instantiate once per video.  Call :meth:`record` each frame after
    all pipeline stages have run.  Call :meth:`flush` when the video
    ends to write the CSV(s).

    Parameters
    ----------
    output_dir : str or Path
        Directory for output files.
    video_name : str
        Name of the current video (used in the ``video`` column and
        as the file stem).
    detail : bool
        If True, also write a per-keypoint detail CSV.
    """

    def __init__(self, output_dir, video_name, detail=False):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_name = video_name
        self.detail = detail

        stem = pathlib.Path(video_name).stem

        # Main metrics CSV
        self._metrics_path = self.output_dir / f"{stem}_metrics.csv"
        self._metrics_fh = open(self._metrics_path, "w", newline="")
        self._metrics_w = csv.DictWriter(
            self._metrics_fh, fieldnames=METRICS_FIELDS)
        self._metrics_w.writeheader()

        # Per-keypoint detail CSV (optional)
        self._detail_w = None
        self._detail_fh = None
        if detail:
            self._detail_path = self.output_dir / f"{stem}_kp_detail.csv"
            self._detail_fh = open(self._detail_path, "w", newline="")
            self._detail_w = csv.DictWriter(
                self._detail_fh, fieldnames=KP_DETAIL_FIELDS)
            self._detail_w.writeheader()

        # Previous-frame state for jitter computation
        self._prev_body_lm = {}   # person_idx -> (n_kp, 3) array
        self._prev_hand_L = {}    # person_idx -> (21, 3) array
        self._prev_hand_R = {}    # person_idx -> (21, 3) array

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def record(self, frame_idx, timestamp_sec, person_idx,
               body_lm_smooth, body_vis,
               hand_L_smooth, hand_R_smooth,
               frame_diag, smooth_diag, constraint_diag,
               hand_L_flag=0.0, hand_R_flag=0.0,
               match_dist_L=None, match_dist_R=None,
               inference_ms=0.0):
        """Record one person-frame of metrics.

        All landmark arrays are in pixel coordinates.
        """
        row = {
            "video": self.video_name,
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "person_idx": person_idx,
        }

        # Detection
        row["body_detected"] = int(frame_diag.body_detected)
        row["body_det_score"] = round(frame_diag.body_det_score, 4)
        row["n_hands_real"] = frame_diag.n_hands_real
        row["n_hands_synthetic"] = frame_diag.n_hands_synthetic
        row["n_hands_recrop"] = frame_diag.n_hands_recrop

        # Confidence
        if body_vis is not None and len(body_vis) > 0:
            row["body_vis_mean"] = round(float(np.mean(body_vis)), 4)
            row["body_vis_min"] = round(float(np.min(body_vis)), 4)
        else:
            row["body_vis_mean"] = ""
            row["body_vis_min"] = ""
        row["hand_L_flag"] = round(hand_L_flag, 4) if hand_L_flag else ""
        row["hand_R_flag"] = round(hand_R_flag, 4) if hand_R_flag else ""

        # Jitter
        row["body_jitter_px"] = self._jitter(
            body_lm_smooth, self._prev_body_lm, person_idx)
        row["hand_L_jitter_px"] = self._jitter(
            hand_L_smooth, self._prev_hand_L, person_idx)
        row["hand_R_jitter_px"] = self._jitter(
            hand_R_smooth, self._prev_hand_R, person_idx)

        # Update previous-frame state
        if body_lm_smooth is not None:
            self._prev_body_lm[person_idx] = body_lm_smooth.copy()
        if hand_L_smooth is not None:
            self._prev_hand_L[person_idx] = hand_L_smooth.copy()
        if hand_R_smooth is not None:
            self._prev_hand_R[person_idx] = hand_R_smooth.copy()

        # Smoothing delta
        row["body_smooth_delta_px"] = round(
            smooth_diag.body_smooth_delta_px, 2)
        hand_sd = smooth_diag.hand_smooth_deltas_px
        row["hand_L_smooth_delta_px"] = round(hand_sd[0], 2) if len(hand_sd) > 0 else ""
        row["hand_R_smooth_delta_px"] = round(hand_sd[1], 2) if len(hand_sd) > 1 else ""

        # Constraints
        row["bone_correction_px"] = round(
            constraint_diag.bone_correction_px, 2)
        row["angle_corrections_n"] = constraint_diag.angle_corrections_n

        # Carry-forward
        row["body_carry"] = int(smooth_diag.body_carry)
        row["body_carry_frames"] = smooth_diag.body_carry_frames
        carry = smooth_diag.hand_carry_flags
        row["hand_L_carry"] = int(carry[0]) if len(carry) > 0 else ""
        row["hand_R_carry"] = int(carry[1]) if len(carry) > 1 else ""

        # Matching
        row["hand_arm_match_dist_L"] = (
            round(match_dist_L, 2) if match_dist_L is not None else "")
        row["hand_arm_match_dist_R"] = (
            round(match_dist_R, 2) if match_dist_R is not None else "")

        row["inference_ms"] = round(inference_ms, 2)

        self._metrics_w.writerow(row)

        # Per-keypoint detail
        if self._detail_w is not None:
            self._write_detail(
                frame_idx, person_idx,
                body_lm_smooth, frame_diag.raw_body_landmarks,
                body_vis, "body",
            )
            self._write_detail(
                frame_idx, person_idx,
                hand_L_smooth,
                frame_diag.raw_hand_landmarks[0]
                    if frame_diag.raw_hand_landmarks else None,
                None, "hand_L",
            )
            self._write_detail(
                frame_idx, person_idx,
                hand_R_smooth,
                frame_diag.raw_hand_landmarks[1]
                    if len(frame_diag.raw_hand_landmarks) > 1 else None,
                None, "hand_R",
            )

    def flush(self):
        """Close file handles.  Safe to call multiple times."""
        if self._metrics_fh is not None:
            self._metrics_fh.close()
            self._metrics_fh = None
        if self._detail_fh is not None:
            self._detail_fh.close()
            self._detail_fh = None

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _jitter(current, prev_store, person_idx):
        """Sum of per-keypoint Euclidean displacements from previous frame."""
        if current is None or person_idx not in prev_store:
            return ""
        prev = prev_store[person_idx]
        if prev.shape != current.shape:
            return ""
        deltas = np.linalg.norm(current[:, :2] - prev[:, :2], axis=1)
        return round(float(np.sum(deltas)), 2)

    def _write_detail(self, frame_idx, person_idx,
                      smooth_lm, raw_lm, vis, part):
        """Write per-keypoint rows to the detail CSV."""
        if smooth_lm is None:
            return
        n_kp = smooth_lm.shape[0]
        prev_key = {"body": self._prev_body_lm,
                     "hand_L": self._prev_hand_L,
                     "hand_R": self._prev_hand_R}[part]
        prev = prev_key.get(person_idx)

        for kp in range(n_kp):
            row = {
                "frame_idx": frame_idx,
                "person_idx": person_idx,
                "part": part,
                "kp_idx": kp,
                "x_raw": round(float(raw_lm[kp, 0]), 2) if raw_lm is not None else "",
                "y_raw": round(float(raw_lm[kp, 1]), 2) if raw_lm is not None else "",
                "x_smooth": round(float(smooth_lm[kp, 0]), 2),
                "y_smooth": round(float(smooth_lm[kp, 1]), 2),
                "visibility": round(float(vis[kp]), 4) if vis is not None else "",
                "jitter_px": "",
            }
            if prev is not None and prev.shape[0] > kp:
                row["jitter_px"] = round(float(np.linalg.norm(
                    smooth_lm[kp, :2] - prev[kp, :2])), 2)
            self._detail_w.writerow(row)
