"""Per-frame metrics collection for optimization diagnostics.

Hooks into the pipeline to capture detection quality, smoothing
effectiveness, constraint activation, and tracking stability metrics.
Produces ``*_metrics.csv`` and optionally ``*_kp_detail.csv`` files
that the R analysis suite consumes.
"""

import csv
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
    "video",
    "frame_idx",
    "timestamp_sec",
    "person_idx",
    # Detection
    "body_detected",
    "body_det_score",
    "n_hands_real",
    "n_hands_synthetic",
    "n_hands_recrop",
    # Confidence
    "body_vis_mean",
    "body_vis_min",
    "hand_L_flag",
    "hand_R_flag",
    # Jitter (frame-to-frame displacement in pixels, sum over keypoints)
    "body_jitter_px",
    "hand_L_jitter_px",
    "hand_R_jitter_px",
    # Smoothing delta (L2 between raw and smoothed, sum over keypoints)
    "body_smooth_delta_px",
    "hand_L_smooth_delta_px",
    "hand_R_smooth_delta_px",
    # Constraints
    "bone_correction_px",
    "angle_corrections_n",
    # Carry-forward
    "body_carry",
    "body_carry_frames",
    "hand_L_carry",
    "hand_R_carry",
    # Matching
    "hand_arm_match_dist_L",
    "hand_arm_match_dist_R",
    # FPS
    "inference_ms",
]

KP_DETAIL_FIELDS = [
    "frame_idx",
    "person_idx",
    "part",
    "kp_idx",
    "x_raw",
    "y_raw",
    "x_smooth",
    "y_smooth",
    "visibility",
    "jitter_px",
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

        # Main metrics CSV (persistent handle; closed in flush()).
        # csv.writer + positional lists is ~2x faster than DictWriter for
        # the per-frame hot path — no dict allocation, no per-field hash.
        self._metrics_path = self.output_dir / f"{stem}_metrics.csv"
        self._metrics_fh = self._metrics_path.open("w", newline="")
        self._metrics_w = csv.writer(self._metrics_fh)
        self._metrics_w.writerow(METRICS_FIELDS)

        # Per-keypoint detail CSV (optional)
        self._detail_w = None
        self._detail_fh = None
        if detail:
            self._detail_path = self.output_dir / f"{stem}_kp_detail.csv"
            self._detail_fh = self._detail_path.open("w", newline="")
            self._detail_w = csv.writer(self._detail_fh)
            self._detail_w.writerow(KP_DETAIL_FIELDS)

        # Previous-frame state for jitter computation
        self._prev_body_lm = {}  # person_idx -> (n_kp, 3) array
        self._prev_hand_L = {}  # person_idx -> (21, 3) array
        self._prev_hand_R = {}  # person_idx -> (21, 3) array

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def record(
        self,
        frame_idx,
        timestamp_sec,
        person_idx,
        body_lm_smooth,
        body_vis,
        hand_L_smooth,
        hand_R_smooth,
        frame_diag,
        smooth_diag,
        constraint_diag,
        hand_L_flag=0.0,
        hand_R_flag=0.0,
        match_dist_L=None,
        match_dist_R=None,
        inference_ms=0.0,
    ):
        """Record one person-frame of metrics.

        All landmark arrays are in pixel coordinates.
        """
        # Confidence — pre-format as strings; csv.writer skips str() on
        # string fields, which avoids the float→str conversion that
        # otherwise runs *inside* writerow for every numeric column.
        if body_vis is not None and len(body_vis) > 0:
            body_vis_mean = f"{float(body_vis.mean()):.4f}"
            body_vis_min = f"{float(body_vis.min()):.4f}"
        else:
            body_vis_mean = ""
            body_vis_min = ""

        # Jitter (computed before prev-state update) — inlined so the
        # prev-store lookup is shared with the prev-state write below
        # and the result is formatted once into its CSV-ready string.
        pid = person_idx
        prev_body = self._prev_body_lm.get(pid)
        if (
            body_lm_smooth is None
            or prev_body is None
            or prev_body.shape != body_lm_smooth.shape
        ):
            body_jitter = ""
        else:
            d = body_lm_smooth[:, :2] - prev_body[:, :2]
            body_jitter = f"{float(np.hypot(d[:, 0], d[:, 1]).sum()):.2f}"

        prev_hL = self._prev_hand_L.get(pid)
        if hand_L_smooth is None or prev_hL is None or prev_hL.shape != hand_L_smooth.shape:
            hand_L_jitter = ""
        else:
            d = hand_L_smooth[:, :2] - prev_hL[:, :2]
            hand_L_jitter = f"{float(np.hypot(d[:, 0], d[:, 1]).sum()):.2f}"

        prev_hR = self._prev_hand_R.get(pid)
        if hand_R_smooth is None or prev_hR is None or prev_hR.shape != hand_R_smooth.shape:
            hand_R_jitter = ""
        else:
            d = hand_R_smooth[:, :2] - prev_hR[:, :2]
            hand_R_jitter = f"{float(np.hypot(d[:, 0], d[:, 1]).sum()):.2f}"

        # Update previous-frame state
        if body_lm_smooth is not None:
            self._prev_body_lm[pid] = body_lm_smooth.copy()
        if hand_L_smooth is not None:
            self._prev_hand_L[pid] = hand_L_smooth.copy()
        if hand_R_smooth is not None:
            self._prev_hand_R[pid] = hand_R_smooth.copy()

        # Smoothing-delta / carry tuples (variable length)
        hand_sd = smooth_diag.hand_smooth_deltas_px
        n_hand_sd = len(hand_sd)
        carry = smooth_diag.hand_carry_flags
        n_carry = len(carry)

        # Build row positionally — order MUST match METRICS_FIELDS.
        self._metrics_w.writerow(
            (
                self.video_name,
                frame_idx,
                f"{timestamp_sec:.4f}",
                pid,
                # Detection
                "1" if frame_diag.body_detected else "0",
                f"{frame_diag.body_det_score:.4f}",
                frame_diag.n_hands_real,
                frame_diag.n_hands_synthetic,
                frame_diag.n_hands_recrop,
                # Confidence
                body_vis_mean,
                body_vis_min,
                f"{hand_L_flag:.4f}" if hand_L_flag else "",
                f"{hand_R_flag:.4f}" if hand_R_flag else "",
                # Jitter
                body_jitter,
                hand_L_jitter,
                hand_R_jitter,
                # Smoothing delta
                f"{smooth_diag.body_smooth_delta_px:.2f}",
                f"{hand_sd[0]:.2f}" if n_hand_sd > 0 else "",
                f"{hand_sd[1]:.2f}" if n_hand_sd > 1 else "",
                # Constraints
                f"{constraint_diag.bone_correction_px:.2f}",
                constraint_diag.angle_corrections_n,
                # Carry-forward
                "1" if smooth_diag.body_carry else "0",
                smooth_diag.body_carry_frames,
                ("1" if carry[0] else "0") if n_carry > 0 else "",
                ("1" if carry[1] else "0") if n_carry > 1 else "",
                # Matching
                f"{match_dist_L:.2f}" if match_dist_L is not None else "",
                f"{match_dist_R:.2f}" if match_dist_R is not None else "",
                # FPS
                f"{inference_ms:.2f}",
            )
        )

        # Per-keypoint detail
        if self._detail_w is not None:
            raw_hands = frame_diag.raw_hand_landmarks
            self._write_detail(
                frame_idx,
                person_idx,
                body_lm_smooth,
                frame_diag.raw_body_landmarks,
                body_vis,
                "body",
            )
            self._write_detail(
                frame_idx,
                person_idx,
                hand_L_smooth,
                raw_hands[0] if raw_hands else None,
                None,
                "hand_L",
            )
            self._write_detail(
                frame_idx,
                person_idx,
                hand_R_smooth,
                raw_hands[1] if len(raw_hands) > 1 else None,
                None,
                "hand_R",
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
        return _jitter(current, prev_store, person_idx)

    def _write_detail(self, frame_idx, person_idx, smooth_lm, raw_lm, vis, part):
        """Write per-keypoint rows to the detail CSV.  Caller guards on
        ``self._detail_w is not None``."""
        if smooth_lm is None or self._detail_w is None:
            return
        n_kp = smooth_lm.shape[0]
        prev_key = _PREV_STORE_BY_PART[part]
        prev_store = getattr(self, prev_key)
        prev = prev_store.get(person_idx)
        has_raw = raw_lm is not None
        has_vis = vis is not None
        has_prev = prev is not None
        prev_n = prev.shape[0] if has_prev else 0

        # Pre-extract per-row primitives to avoid repeated attribute lookups
        # inside the keypoint loop.
        smooth_xy = smooth_lm[:, :2]
        raw_xy = raw_lm[:, :2] if has_raw else None
        diff = (smooth_xy - prev[:, :2]) if (has_prev and prev_n >= n_kp) else None
        if diff is not None:
            jitter_per_kp = np.hypot(diff[:, 0], diff[:, 1])
        else:
            jitter_per_kp = None

        rows = [
            (
                frame_idx,
                person_idx,
                part,
                kp,
                f"{float(raw_xy[kp, 0]):.2f}" if has_raw else "",
                f"{float(raw_xy[kp, 1]):.2f}" if has_raw else "",
                f"{float(smooth_xy[kp, 0]):.2f}",
                f"{float(smooth_xy[kp, 1]):.2f}",
                f"{float(vis[kp]):.4f}" if has_vis else "",
                f"{float(jitter_per_kp[kp]):.2f}" if jitter_per_kp is not None else "",
            )
            for kp in range(n_kp)
        ]
        self._detail_w.writerows(rows)


# ---------------------------------------------------------------------------
# Module-level helpers (avoid staticmethod descriptor overhead in hot path)
# ---------------------------------------------------------------------------

_PREV_STORE_BY_PART = {
    "body": "_prev_body_lm",
    "hand_L": "_prev_hand_L",
    "hand_R": "_prev_hand_R",
}


def _jitter(current, prev_store, person_idx):
    if current is None:
        return ""
    prev = prev_store.get(person_idx)
    if prev is None or prev.shape != current.shape:
        return ""
    diff = current[:, :2] - prev[:, :2]
    return round(float(np.hypot(diff[:, 0], diff[:, 1]).sum()), 2)
