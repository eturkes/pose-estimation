"""End-to-end validation harness for the 3D clinical pipeline.

One command runs the full chain on a single session — calibration
(solve or load) → 2D tracking → ``world3d.csv`` fusion → clinical
metrics — and emits a structured :class:`ValidationReport` (JSON for
CI, Markdown for humans).  See ``docs/technical/validation.md``.

Design: this module *orchestrates* and *measures*; it reimplements no
pipeline maths.  It reuses :func:`charuco.solve_charuco` (calibration),
:func:`multicam.fuse_session_outputs` + :func:`export.write_world3d_csv`
(fusion), :func:`export.read_csv_keypoints` (2D read-back), and the R
``analysis/clinical_features.R`` script (clinical metrics).  The stages
are run individually (rather than via :func:`multicam.process_session`,
which bundles tracking and fusion) so each gets its own wall-clock.

The clinical-metric *agreement* leg is baseline-optional, by design:
no external ground truth exists yet.  With a
baseline → per-metric error vs the reference.  Without → internal
self-consistency surrogates (bone-length coefficient-of-variation,
left/right symmetry, temporal jitter) computed straight from the fused
``world3d.csv``.

Reports include versioned PASS/WARN/FAIL grading; the thresholds remain
provisional until calibrated against cleared real captures.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import enum
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time
import warnings
from typing import Any

import numpy as np

from ._types import SessionCalibration
from .calibration import (
    CALIBRATION_FILENAME,
    CalibrationError,
    load_calibration,
    save_calibration,
)
from .charuco import (
    MIN_INTRINSIC_FRAMES,
    detect_charuco_corners,
    make_charuco_board,
    solve_charuco,
)
from .export import WORLD3D_FILENAME, read_csv_keypoints, write_world3d_csv
from .multicam import (
    Session,
    SessionError,
    _resolve_session_output,
    discover_session,
    fuse_session_outputs,
)
from .video_io import frame_count

REPORT_SCHEMA_VERSION = 3
"""Bumped when the :class:`ValidationReport` JSON layout changes.

v3 adds observed/expected 2D frame counts, expected-frame 3D coverage,
trusted-keypoint coverage, triangulation-angle diagnostics, and
candidate-to-consensus view rejection.  Threshold *value* changes do **not**
bump this — they bump :data:`THRESHOLDS_VERSION` instead.
"""

REPROJ_GATE_PX = 20.0
"""Per-keypoint reprojection gate (px), mirrors the fusion-side
``triangulation.max_view_reproj_px`` and the R ``REPROJ_GATE_PX``.  At
exactly ``min_views`` an outlier view cannot be dropped, so a keypoint
above this gate is treated as untrustworthy by self-consistency.  The
fusion-reproj p95 FAIL band (:data:`THRESHOLDS`) is anchored to it: a
p95 at the gate means most keypoints sit at the edge of trust.
"""

TRIANGULATION_ANGLE_GATE_DEG = 1.0
"""Minimum acceptable acute viewing-ray angle for a trusted 3D sample.

This mirrors the fusion policy's current ``min_triangulation_angle_deg``
default.  Older ``world3d.csv`` files do not carry the angle column; their
samples remain assessable with the other trust gates and the missing angle is
reported explicitly.
"""

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"
_CLINICAL_TIMEOUT_S = 600

# Bones whose 3D length is expected rigid frame-to-frame.  Endpoint
# names match the world3d.csv schema (export.ARM_KEYPOINT_NAMES /
# BODY_KEYPOINT_NAMES, prefixed).  The arm-mode set uses "arm_" names,
# body-mode the "body_" names; the harness keeps only bones whose both
# endpoints exist in the fused keypoint set, so the active mode selects
# itself.  Index pairs are mirrored from constraints.BONE_SEGMENTS{,_BODY}.
_BONES: list[tuple[str, str, str]] = [
    ("left_upper_arm", "arm_left_shoulder", "arm_left_elbow"),
    ("right_upper_arm", "arm_right_shoulder", "arm_right_elbow"),
    ("left_forearm", "arm_left_elbow", "arm_left_wrist"),
    ("right_forearm", "arm_right_elbow", "arm_right_wrist"),
    ("shoulder_width", "arm_left_shoulder", "arm_right_shoulder"),
    ("body_left_upper_arm", "body_left_shoulder", "body_left_elbow"),
    ("body_right_upper_arm", "body_right_shoulder", "body_right_elbow"),
    ("body_left_forearm", "body_left_elbow", "body_left_wrist"),
    ("body_right_forearm", "body_right_elbow", "body_right_wrist"),
    ("body_left_thigh", "body_left_hip", "body_left_knee"),
    ("body_right_thigh", "body_right_hip", "body_right_knee"),
    ("body_left_shank", "body_left_knee", "body_left_ankle"),
    ("body_right_shank", "body_right_knee", "body_right_ankle"),
    ("body_shoulder_width", "body_left_shoulder", "body_right_shoulder"),
    ("body_hip_width", "body_left_hip", "body_right_hip"),
]

# (left_bone, right_bone) label pairs for the left/right symmetry surrogate.
_SYMMETRIC_BONES: list[tuple[str, str]] = [
    ("left_upper_arm", "right_upper_arm"),
    ("left_forearm", "right_forearm"),
    ("body_left_upper_arm", "body_right_upper_arm"),
    ("body_left_forearm", "body_right_forearm"),
    ("body_left_thigh", "body_right_thigh"),
    ("body_left_shank", "body_right_shank"),
]


# ---------------------------------------------------------------------------
# Acceptance thresholds + PASS/WARN/FAIL grading
# ---------------------------------------------------------------------------


class Grade(enum.IntEnum):
    """Ordered severity; the overall verdict is the worst (max) check."""

    PASS = 0
    WARN = 1
    FAIL = 2


@dataclasses.dataclass(frozen=True)
class Band:
    """A WARN/FAIL boundary pair with a comparison direction.

    ``direction="max"`` — lower is better (errors, fractions): a value
    ``<= warn`` PASSes, ``<= fail`` WARNs, else FAILs.
    ``direction="min"`` — higher is better (fps, view count): a value
    ``>= warn`` PASSes, ``>= fail`` WARNs, else FAILs.

    A non-finite value grades WARN: a metric the harness could not
    measure is surfaced for human review, never silently passed.
    """

    warn: float
    fail: float
    direction: str = "max"

    def grade(self, value: float | None) -> Grade:
        if value is None or not math.isfinite(value):
            return Grade.WARN
        if self.direction == "max":
            if value <= self.warn:
                return Grade.PASS
            return Grade.WARN if value <= self.fail else Grade.FAIL
        if value >= self.warn:
            return Grade.PASS
        return Grade.WARN if value >= self.fail else Grade.FAIL

    def describe(self, value: float | None) -> str:
        op = "<=" if self.direction == "max" else ">="
        return f"{_fmt(value)} (warn {op}{_fmt(self.warn)}, fail {op}{_fmt(self.fail)})"


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """Versioned, single-source-of-truth acceptance bands.

    Rationale + citations live in ``docs/technical/validation.md`` and
    inline at :data:`THRESHOLDS`.  Bump :data:`THRESHOLDS_VERSION` on any
    value change; the report's ``schema_version`` tracks JSON *layout*.
    """

    version: int
    confidence_floor: float
    calib_reproj_rms_px: Band
    fusion_reproj_median_px: Band
    fusion_reproj_p95_px: Band
    n_views_floor: int
    n_views_median: Band
    min_tracking_detection_rate: Band
    max_low_confidence_fraction: Band
    min_fused_frame_fraction: Band
    min_trusted_keypoint_fraction: Band
    min_triangulation_angle_p05_deg: Band
    max_view_rejection_fraction: Band
    max_unfused_fraction: Band
    max_cheirality_rate: Band
    min_throughput_fps: Band
    max_bone_length_cv: Band
    max_temporal_jitter_mm: Band
    max_symmetry_rel_diff: Band
    agreement_tolerance_deg: Band


THRESHOLDS_VERSION = 2
"""Bumped on any change to a :data:`THRESHOLDS` value. Recalibrate these
bands against cleared real captures."""

THRESHOLDS = Thresholds(
    version=THRESHOLDS_VERSION,
    # 2D keypoints below this confidence are "low-confidence". Provisional
    # rtmlib/RTMPose floor; re-tune against cleared real footage.
    confidence_floor=0.3,
    # Calibration reprojection RMS.  Photogrammetry/markerless standard:
    # < 1 px is the gold standard, ~2 px (~1 cm) still yields usable
    # kinematics downstream (Pose2Sim robustness, Pagnon et al. 2021).
    calib_reproj_rms_px=Band(warn=1.0, fail=2.0),
    # Per-keypoint fusion reprojection.  Pose2Sim triangulation cutoff
    # ~10 px; Anipose < 12 px in > 75 % of frames (Karashchuk et al. 2021).
    fusion_reproj_median_px=Band(warn=8.0, fail=12.0),
    # p95 anchored to the fusion gate: Anipose treats > 20 px as missing
    # (== REPROJ_GATE_PX); < 18 px in > 90 % of frames.
    fusion_reproj_p95_px=Band(warn=15.0, fail=REPROJ_GATE_PX),
    # DLT triangulation needs >= 2 views; below it is malformed fusion.
    n_views_floor=2,
    # Redundancy: with the 3-camera deployment a median < 3 means the
    # typical keypoint has no spare view to reject an outlier (multicam.md).
    n_views_median=Band(warn=3.0, fail=2.0, direction="min"),
    # Provisional engineering defaults; calibrate on cleared real data:
    min_tracking_detection_rate=Band(warn=0.80, fail=0.50, direction="min"),
    max_low_confidence_fraction=Band(warn=0.2, fail=0.4),
    min_fused_frame_fraction=Band(warn=0.95, fail=0.80, direction="min"),
    min_trusted_keypoint_fraction=Band(warn=0.90, fail=0.75, direction="min"),
    # The fusion hard-gates below 1 degree.  A fifth percentile under
    # 2 degrees warns that the accepted geometry still sits near that floor.
    min_triangulation_angle_p05_deg=Band(warn=2.0, fail=1.0, direction="min"),
    # Fraction of pre-consensus candidate views rejected as geometric
    # outliers. Persistent 3->2 rejection is 1/3 and therefore a hard fault.
    max_view_rejection_fraction=Band(warn=0.05, fail=0.20),
    max_unfused_fraction=Band(warn=0.1, fail=0.25),
    # Cheirality (point in front of camera) should essentially never fail.
    max_cheirality_rate=Band(warn=0.01, fail=0.05),
    # Whole-pipeline fused fps incl. one-time solve/R — a coarse perf
    # regression signal, graded but informational until a multi-device
    # performance study sets the real budget.
    min_throughput_fps=Band(warn=15.0, fail=5.0, direction="min"),
    # Self-consistency surrogates; no external baseline exists.
    # Rigid-bone length CoV: ~1 cm markerless keypoint noise on a ~0.25 m
    # forearm ~= 4-5 % (dual-camera OA RMSD ~11 mm, Ann. Biomed. Eng. 2025).
    max_bone_length_cv=Band(warn=0.05, fail=0.10),
    # Static-pose temporal noise via 2nd-difference magnitude (mm).
    max_temporal_jitter_mm=Band(warn=5.0, fail=15.0),
    # L/R symmetry is valid ONLY for symmetric-by-construction input;
    # graded informational (real anatomical asymmetry would confound it).
    max_symmetry_rel_diff=Band(warn=0.05, fail=0.10),
    # Joint-angle agreement vs a baseline: ~5 deg clinical precision
    # threshold; OpenCap ~6 deg flagged borderline (OpenCap validation 2024).
    # Used only when a baseline is supplied (none yet — UNVALIDATED).
    agreement_tolerance_deg=Band(warn=5.0, fail=10.0),
)
"""Current acceptance thresholds. See ``docs/technical/validation.md`` for the full
rationale table and the clinical-validity gap register."""


DEFAULT_CONFIDENCE_FLOOR = THRESHOLDS.confidence_floor
"""2D keypoints below this confidence count as low-confidence.  Sourced
from :data:`THRESHOLDS` so the floor has a single definition."""


COVERAGE_GRID = (8, 6)
"""(cols, rows) image grid for the ChArUco board-coverage QA metric: the
fraction of these cells the pooled board corners touch measures how much
of a camera's field of view the calibration sweep visited.  A board
confined to the centre lights up few cells (weak oblique-camera
intrinsics)."""


@dataclasses.dataclass(frozen=True)
class QAThresholds:
    """Versioned acceptance bands for the pre-flight capture QA gate.

    Distinct from :data:`THRESHOLDS`, which grades the *output* report:
    these grade a *raw capture* before its clinical metrics are trusted.
    Shared physical quantities (calibration RMS, the confidence floor) are
    read from :data:`THRESHOLDS` so they keep one definition; only
    capture-specific bands live here.  Bump :data:`QA_THRESHOLDS_VERSION`
    on any value change.
    """

    version: int
    min_charuco_frames: int
    min_charuco_detection_rate: Band
    min_board_coverage: Band
    max_frame_count_disparity: Band
    min_subject_detection_rate: Band


QA_THRESHOLDS_VERSION = 1
"""Bumped on any change to a :data:`QA_THRESHOLDS` value. Recalibrate these
provisional bands against cleared real captures."""

QA_THRESHOLDS = QAThresholds(
    version=QA_THRESHOLDS_VERSION,
    # Hard floor: the solver needs >= MIN_INTRINSIC_FRAMES usable board
    # views per camera for intrinsics (and >= MIN_SHARED_FRAMES shared with
    # the world camera); below this the solve cannot even run.
    min_charuco_frames=MIN_INTRINSIC_FRAMES,
    # Board-detection rate is capture-style dependent (a fast, varied sweep
    # detects in fewer frames yet constrains geometry better than a slow
    # static one), so these bands are lenient — the absolute frame floor
    # above is the real sufficiency gate. Provisional.
    min_charuco_detection_rate=Band(warn=0.30, fail=0.10, direction="min"),
    # Fraction of the COVERAGE_GRID cells the board swept.  A full-volume
    # translation+tilt sweep lights up most cells; a centre-bound board
    # weakly constrains oblique-camera intrinsics and couples fx error into
    # stereo translation. Provisional.
    min_board_coverage=Band(warn=0.60, fail=0.35, direction="min"),
    # Raw per-camera frame-count parity as a software-sync desync proxy.
    # Declared sync_offsets trim pre-roll, so a few frames of disparity is
    # normal; a large mismatch signals a dropped/desynced recording.
    max_frame_count_disparity=Band(warn=0.05, fail=0.20),
    # The subject should be tracked in most frames of a usable clip.
    # Provisional engineering default; calibrate on cleared real data.
    min_subject_detection_rate=Band(warn=0.80, fail=0.50, direction="min"),
)
"""Current capture-QA thresholds. See ``docs/technical/validation.md`` for the
rationale table; graded by :func:`qa_check` → :func:`_grade_qa`."""


@dataclasses.dataclass
class Check:
    """One graded metric within a :class:`Verdict`."""

    name: str
    value: float | None
    grade: str  # Grade.name: PASS / WARN / FAIL
    detail: str
    informational: bool = False  # surfaced but excluded from the overall grade


@dataclasses.dataclass
class Verdict:
    """Overall PASS/WARN/FAIL grade of a report against :class:`Thresholds`.

    ``grade`` is the worst *non-informational* check.  Informational
    checks (timing throughput, L/R symmetry) are reported but never
    escalate the overall grade — performance is not clinical validity,
    and symmetry is valid only on symmetric-by-construction input.
    """

    grade: str
    thresholds_version: int
    checks: list[Check]
    notes: list[str]

    @property
    def passed(self) -> bool:
        """True when the overall grade is not FAIL (the CI gate)."""
        return self.grade != Grade.FAIL.name


class ValidationError(RuntimeError):
    """Raised when the harness cannot run a session to completion."""


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CameraIntrinsics:
    """Per-camera intrinsic summary lifted from the calibration."""

    name: str
    resolution: tuple[int, int]
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_l2: float


@dataclasses.dataclass
class CalibrationSection:
    """Calibration provenance + quality."""

    n_cameras: int
    world_frame: str
    reprojection_error_px: float
    solved: bool  # True if solved this run, False if loaded from disk
    cameras: list[CameraIntrinsics]


@dataclasses.dataclass
class CameraTracking:
    """Per-camera 2D detection summary, read back from its CSV."""

    name: str
    observed_frames: int  # post-sync source frames represented in the CSV
    expected_frames: int  # source frame count after applying sync_offset
    detection_rate: float  # detected keypoints / expected frame-keypoint slots
    low_confidence_fraction: float  # of detected keypoints, fraction below floor
    dropped_frames: int  # expected frames with zero detected keypoints

    @property
    def n_frames(self) -> int:
        """Backward-compatible alias for the old observed-frame field."""
        return self.observed_frames


@dataclasses.dataclass
class Tracking2DSection:
    confidence_floor: float
    reused_existing_csvs: bool
    total_observed_frames: int
    total_expected_frames: int
    mean_detection_rate: float
    cameras: list[CameraTracking]

    @property
    def total_frames(self) -> int:
        """Backward-compatible alias for the old observed-frame total."""
        return self.total_observed_frames


@dataclasses.dataclass
class Fusion3DSection:
    """Raw and trust-gated diagnostics parsed from ``world3d.csv``."""

    n_frames_fused: int
    expected_frames: int
    fused_frame_fraction: float
    n_active_keypoints: int  # keypoints fused or attempted from >= 2 views
    trusted_keypoint_fraction: float  # trusted / expected active-keypoint slots
    reproj_err_px_median: float
    reproj_err_px_p95: float
    reproj_err_px_max: float
    n_views_median: float
    n_views_min: int
    cheirality_violation_rate: float  # of fused keypoints
    triangulation_angle_available: bool
    triangulation_angle_gate_deg: float
    triangulation_angle_deg_p05: float
    triangulation_angle_deg_median: float
    candidate_views_available: bool
    candidate_n_views_median: float
    view_rejection_fraction: float
    unfused_keypoint_fraction: float  # of active-keypoint frame slots


@dataclasses.dataclass
class TimingSection:
    det_device: str
    pose_device: str
    backend: str
    solve_sec: float
    tracking_2d_sec: float
    fusion_sec: float
    clinical_sec: float
    total_sec: float
    throughput_fps: float  # fused frames per total pipeline second
    tracking_2d_per_camera: dict[str, float]


@dataclasses.dataclass
class AgreementSection:
    """Baseline-optional clinical agreement / self-consistency.

    ``per_metric_error`` is populated only when a baseline is supplied;
    otherwise the self-consistency surrogates carry the evidence.
    """

    has_baseline: bool
    clinical_csv_produced: bool
    clinical_outputs: list[str]
    mean_bone_length_cv: float
    bone_length_cv: dict[str, float]
    mean_symmetry_rel_diff: float
    symmetry_rel_diff: dict[str, float]
    temporal_jitter_mm: float
    per_metric_error: dict[str, float] | None


@dataclasses.dataclass
class ValidationReport:
    """Full validation result for one session.

    ``to_json`` is CI-parseable (non-finite floats become ``null``);
    ``to_markdown`` is a human summary; ``verdict`` applies the versioned
    PASS/WARN/FAIL thresholds.
    """

    session_id: str
    schema_version: int
    calibration: CalibrationSection
    tracking_2d: Tracking2DSection
    fusion_3d: Fusion3DSection
    timing: TimingSection
    agreement: AgreementSection
    notes: list[str]

    def verdict(self, thresholds: Thresholds = THRESHOLDS) -> Verdict:
        """Grade this report's metrics against *thresholds* → PASS/WARN/FAIL."""
        return _grade_report(self, thresholds)

    def to_json(self) -> dict[str, Any]:
        payload = _native(dataclasses.asdict(self))
        payload["verdict"] = _native(dataclasses.asdict(self.verdict()))
        return payload

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_validation(
    session_dir: str | pathlib.Path,
    *,
    calibration: str | pathlib.Path | None = None,
    baseline: str | pathlib.Path | None = None,
    det_device: str = "CPU",
    pose_device: str = "NPU",
    backend: str = "onnxruntime",
    output_dir: str | pathlib.Path | None = None,
    camera_processor: Any = None,
    run_clinical: bool = True,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> ValidationReport:
    """Validate the full 3D clinical pipeline on one session.

    Stages, each independently timed:

    1. **Calibration.**  ``calibration`` may be a ``calibration.json``
       (loaded) or a ChArUco calibration-session directory (solved via
       :func:`charuco.solve_charuco` when it holds no ``calibration.json``,
       using the default board geometry; pre-solve with
       ``pose-estimation-calibrate solve`` for a custom board).  When
       ``None``, the session's own ``calibration.json`` is used.
    2. **2D tracking.**  If ``camera_processor`` is given it is invoked
       per camera (same callback contract as
       :func:`multicam.process_session`).  Else, existing per-camera CSVs
       under the output directory are reused.  Else, the default rtmlib
       backend is run as a subprocess (footage path; honours
       ``device`` / ``backend``).  Per-camera CSVs are then read back for
       detection metrics.
    3. **3D fusion.**  :func:`multicam.fuse_session_outputs` +
       :func:`export.write_world3d_csv` produce ``world3d.csv``.
    4. **Clinical metrics + agreement.**  ``analysis/clinical_features.R``
       consumes ``world3d.csv``; agreement vs ``baseline`` if supplied,
       else self-consistency surrogates from the fused coordinates.

    Returns a :class:`ValidationReport`.  Raises :class:`ValidationError`
    when no calibration is available (fusion impossible) or no per-camera
    CSVs can be obtained.
    """
    session_dir = pathlib.Path(session_dir)
    notes: list[str] = []

    session = discover_session(session_dir)

    # --- 1. Calibration -----------------------------------------------------
    t0 = time.perf_counter()
    if calibration is not None:
        calib, solved = _resolve_external_calibration(calibration)
        session.calibration = calib
    else:
        calib, solved = session.calibration, False
    solve_sec = time.perf_counter() - t0
    if session.calibration is None:
        raise ValidationError(
            f"session {session.session_id!r}: 3D validation needs a calibration "
            "(pass --calibration or place calibration.json in the session)"
        )
    calibration_section = _build_calibration_section(session.calibration, solved)

    session_out = _resolve_session_output(session, output_dir)

    # --- 2. 2D tracking -----------------------------------------------------
    per_camera_time, reused = _run_tracking(
        session, session_out, output_dir, camera_processor, det_device, pose_device, backend, notes
    )
    tracking_section = _measure_tracking(session, session_out, confidence_floor, reused, notes)

    # --- 3. 3D fusion -------------------------------------------------------
    t0 = time.perf_counter()
    fusion = fuse_session_outputs(session, output_dir)
    if not fusion.frames:
        raise ValidationError(
            f"session {session.session_id!r}: no logical frame is visible from >= 2 "
            "cameras; nothing to fuse"
        )
    world3d_path = write_world3d_csv(
        session_out / WORLD3D_FILENAME,
        session.session_id,
        fusion.keypoint_names,
        fusion.frames,
    )
    fusion_sec = time.perf_counter() - t0
    expected_fusion_frames = _expected_fusion_frames(tracking_section, min_views=2)
    fusion_section = _measure_fusion(world3d_path, expected_fusion_frames)

    # --- 4. Clinical metrics + agreement -----------------------------------
    t0 = time.perf_counter()
    clinical_outputs = _run_clinical(world3d_path, notes) if run_clinical else []
    clinical_sec = time.perf_counter() - t0
    if not run_clinical:
        notes.append("The pipeline skipped clinical metrics (run_clinical=False).")
    agreement_section = _build_agreement(
        world3d_path, session_out, baseline, clinical_outputs, notes
    )

    # --- Timing -------------------------------------------------------------
    tracking_sec = float(sum(per_camera_time.values()))
    total_sec = solve_sec + tracking_sec + fusion_sec + clinical_sec
    throughput = fusion_section.n_frames_fused / total_sec if total_sec > 0 else float("nan")
    timing_section = TimingSection(
        det_device=det_device,
        pose_device=pose_device,
        backend=backend,
        solve_sec=solve_sec,
        tracking_2d_sec=tracking_sec,
        fusion_sec=fusion_sec,
        clinical_sec=clinical_sec,
        total_sec=total_sec,
        throughput_fps=throughput,
        tracking_2d_per_camera=per_camera_time,
    )

    return ValidationReport(
        session_id=session.session_id,
        schema_version=REPORT_SCHEMA_VERSION,
        calibration=calibration_section,
        tracking_2d=tracking_section,
        fusion_3d=fusion_section,
        timing=timing_section,
        agreement=agreement_section,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Stage 1 — calibration
# ---------------------------------------------------------------------------


def _resolve_external_calibration(
    calibration: str | pathlib.Path,
) -> tuple[SessionCalibration, bool]:
    """Load a ``calibration.json`` or solve a calibration session.

    Returns ``(calibration, solved)`` where *solved* is True only when a
    ChArUco solve ran this call.  A freshly solved calibration is saved
    into the session directory so re-runs load it.
    """
    p = pathlib.Path(calibration)
    if p.is_file():
        return load_calibration(p), False
    if p.is_dir():
        inner = p / CALIBRATION_FILENAME
        if inner.is_file():
            return load_calibration(inner), False
        calib = solve_charuco(p)
        save_calibration(calib, inner)
        return calib, True
    raise ValidationError(f"calibration path not found: {p}")


def _build_calibration_section(calib: SessionCalibration, solved: bool) -> CalibrationSection:
    cameras = []
    for cam in calib["cameras"].values():
        K = np.asarray(cam["K"], dtype=np.float64)
        distortion = np.asarray(cam["distortion"], dtype=np.float64)
        res = cam["resolution"]
        cameras.append(
            CameraIntrinsics(
                name=cam["name"],
                resolution=(int(res[0]), int(res[1])),
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                distortion_l2=float(np.linalg.norm(distortion)),
            )
        )
    cameras.sort(key=lambda c: c.name)
    return CalibrationSection(
        n_cameras=len(cameras),
        world_frame=calib["world_frame"],
        reprojection_error_px=float(calib["reprojection_error_px"]),
        solved=solved,
        cameras=cameras,
    )


# ---------------------------------------------------------------------------
# Stage 2 — 2D tracking
# ---------------------------------------------------------------------------


def _run_tracking(
    session: Session,
    session_out: pathlib.Path,
    output_dir: str | pathlib.Path | None,
    camera_processor: Any,
    det_device: str,
    pose_device: str,
    backend: str,
    notes: list[str],
) -> tuple[dict[str, float], bool]:
    """Produce per-camera CSVs; return (per-camera seconds, reused?).

    Priority: injected ``camera_processor`` → reuse existing CSVs →
    default rtmlib subprocess backend.
    """
    if camera_processor is not None:
        session_out.mkdir(parents=True, exist_ok=True)
        per_camera_time: dict[str, float] = {}
        for cam in session.cameras:
            source = str((session.directory / cam.file).resolve())
            t0 = time.perf_counter()
            camera_processor(
                source=source,
                output_csv=session_out / f"{cam.name}.csv",
                output_diag=session_out / f"{cam.name}_diag.csv",
                video_name=f"{session.session_id}/{cam.name}",
            )
            per_camera_time[cam.name] = time.perf_counter() - t0
        return per_camera_time, False

    existing = [c for c in session.cameras if (session_out / f"{c.name}.csv").is_file()]
    if len(existing) >= 2:
        notes.append(
            f"The pipeline reused {len(existing)} existing per-camera CSV(s) under {session_out}."
        )
        return {c.name: 0.0 for c in existing}, True

    elapsed = _run_default_backend(session, output_dir, det_device, pose_device, backend, notes)
    return {"_backend_subprocess": elapsed}, False


def _run_default_backend(
    session: Session,
    output_dir: str | pathlib.Path | None,
    det_device: str,
    pose_device: str,
    backend: str,
    notes: list[str],
) -> float:
    """Run the real rtmlib backend via ``pose-estimation-run`` (footage path).

    Requires cleared footage. Reuses the existing entry point as a subprocess
    rather than re-importing its heavy model setup.
    """
    cmd = [
        sys.executable,
        "-m",
        "pose_estimation.run",
        "--headless",
        "--session-dir",
        str(session.directory),
        "--backend",
        backend,
        "--det-device",
        det_device,
        "--pose-device",
        pose_device,
    ]
    if output_dir is not None:
        cmd += ["--output-dir", str(output_dir)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise ValidationError(
            f"2D backend failed (exit {result.returncode}): {result.stderr.strip()[:300]}"
        )
    notes.append(
        f"The pipeline ran the default backend ({backend}, det={det_device}/pose={pose_device}) "
        "for 2D tracking."
    )
    return elapsed


def _measure_tracking(
    session: Session,
    session_out: pathlib.Path,
    confidence_floor: float,
    reused: bool,
    notes: list[str],
) -> Tracking2DSection:
    measured: dict[str, tuple[list[str], dict[int, tuple[np.ndarray, np.ndarray, float]]]] = {}
    keypoint_count = 0
    for cam in session.cameras:
        csv_path = session_out / f"{cam.name}.csv"
        if not csv_path.is_file():
            notes.append(f"Camera {cam.name!r} has no 2D CSV at {csv_path}.")
            continue
        names, frames = read_csv_keypoints(csv_path)
        measured[cam.name] = (names, frames)
        keypoint_count = max(keypoint_count, len(names))

    if not measured:
        raise ValidationError(
            f"session {session.session_id!r}: no per-camera CSVs under {session_out}"
        )

    cameras: list[CameraTracking] = []
    for cam in session.cameras:
        expected = max(0, frame_count(session.directory / cam.file) - cam.sync_offset)
        names, frames = measured.get(cam.name, ([], {}))
        cameras.append(
            _camera_tracking(
                cam.name,
                frames,
                confidence_floor,
                expected_frames=expected,
                sync_offset=cam.sync_offset,
                n_keypoints=len(names) or keypoint_count,
            )
        )

    total_observed = sum(c.observed_frames for c in cameras)
    total_expected = sum(c.expected_frames for c in cameras)
    rates = [c.detection_rate for c in cameras if math.isfinite(c.detection_rate)]
    mean_rate = float(np.mean(rates)) if rates else float("nan")
    return Tracking2DSection(
        confidence_floor=confidence_floor,
        reused_existing_csvs=reused,
        total_observed_frames=total_observed,
        total_expected_frames=total_expected,
        mean_detection_rate=mean_rate,
        cameras=cameras,
    )


def _camera_tracking(
    name: str,
    frames: dict[int, tuple[np.ndarray, np.ndarray, float]],
    confidence_floor: float,
    *,
    expected_frames: int,
    sync_offset: int = 0,
    n_keypoints: int | None = None,
) -> CameraTracking:
    """Measure one camera against its source-derived frame denominator.

    CSV frame indices are raw source indices.  Pre-roll rows below
    ``sync_offset`` are intentionally excluded, and absent post-offset rows
    contribute zero detected keypoints and one dropped frame.
    """
    if expected_frames < 0:
        raise ValueError(f"expected_frames must be non-negative (got {expected_frames})")
    assessed = _post_sync_frames(frames, expected_frames=expected_frames, sync_offset=sync_offset)
    if n_keypoints is None:
        n_keypoints = max((kps.shape[0] for kps, _conf, _ts in assessed.values()), default=0)

    detected_total = 0
    low_conf_total = 0
    frames_with_detection = 0
    for kps, conf, _ts in assessed.values():
        # Temporal carry/extrapolation rows deliberately use confidence 0;
        # finite coordinates alone therefore do not prove a detector hit.
        detected = np.isfinite(kps).all(axis=1) & np.isfinite(conf) & (conf > 0.0)
        n_det = int(detected.sum())
        if n_det:
            frames_with_detection += 1
        detected_total += n_det
        low_conf_total += int((conf[detected] < confidence_floor).sum())
    slots = expected_frames * n_keypoints
    return CameraTracking(
        name=name,
        observed_frames=len(assessed),
        expected_frames=expected_frames,
        detection_rate=(detected_total / slots) if slots else float("nan"),
        low_confidence_fraction=(low_conf_total / detected_total)
        if detected_total
        else float("nan"),
        dropped_frames=max(0, expected_frames - frames_with_detection),
    )


def _post_sync_frames(
    frames: dict[int, tuple[np.ndarray, np.ndarray, float]],
    *,
    expected_frames: int,
    sync_offset: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """Select the post-sync interval for zero- or one-based CSV indices.

    Current MediaPipe and rtmlib CSVs are zero-based; legacy rtmlib files may be
    one-based. Both candidate intervals are tested, and the one containing more
    observed rows wins. Ties use boundary evidence (index 0 proves zero-based,
    otherwise one-based).
    Missing rows remain absent and are charged to the expected denominator.
    """
    candidates: dict[int, dict[int, tuple[np.ndarray, np.ndarray, float]]] = {}
    for base in (0, 1):
        start = sync_offset + base
        stop = start + expected_frames
        candidates[base] = {idx: value for idx, value in frames.items() if start <= idx < stop}
    preferred_base = 0 if 0 in frames else 1
    base = max((0, 1), key=lambda item: (len(candidates[item]), item == preferred_base))
    return candidates[base]


def _expected_fusion_frames(tracking: Tracking2DSection, *, min_views: int) -> int:
    """Expected logical frames visible to at least ``min_views`` cameras."""
    counts = sorted((c.expected_frames for c in tracking.cameras), reverse=True)
    return counts[min_views - 1] if len(counts) >= min_views else 0


def _expected_world_frames(
    frames: list[dict[str, np.ndarray]], expected_frames: int
) -> list[dict[str, np.ndarray]]:
    """Return unique rows inside the inferred zero/one-based expected interval.

    Current world3d files use zero-based logical frame indices, while legacy
    files may be one-based.  Rows outside either plausible source-derived
    interval are never allowed to replace missing expected frames or inflate
    coverage.
    """
    unique = {int(frame["frame_idx"]): frame for frame in frames}
    if expected_frames <= 0:
        return [unique[index] for index in sorted(unique)]

    candidates = {
        base: {
            index: frame
            for index, frame in unique.items()
            if base <= index < base + expected_frames
        }
        for base in (0, 1)
    }
    preferred_base = 0 if 0 in unique else 1
    base = max((0, 1), key=lambda item: (len(candidates[item]), item == preferred_base))
    return [candidates[base][index] for index in sorted(candidates[base])]


# ---------------------------------------------------------------------------
# Stage 3 — fusion diagnostics (parsed from world3d.csv)
# ---------------------------------------------------------------------------


def _read_world3d(
    path: pathlib.Path,
) -> tuple[list[str], list[dict[str, np.ndarray]]]:
    """Parse ``world3d.csv`` into per-frame arrays.

    Returns ``(keypoint_names, frames)`` where each frame dict holds
    ``xyz`` (N, 3), ``reproj`` (N,), ``n_views`` (N, int),
    ``candidate_n_views`` / ``n_views`` (N,), ``cheirality`` (N, bool),
    ``conf`` (N,), ``angle`` (N,), a
    per-keypoint ``trusted`` mask, and the scalar ``frame_idx``.  A sample is
    trusted only when its XYZ and reprojection error are finite, cheirality is
    true, reprojection is within :data:`REPROJ_GATE_PX`, and — when the file
    carries angle columns — its viewing-ray angle is at least
    :data:`TRIANGULATION_ANGLE_GATE_DEG`.  Blank cells → NaN / 0.

    Angle and candidate-view columns were added after the original world3d
    schema; files without them remain readable and those diagnostics are
    reported unavailable rather than assumed healthy.
    """
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        names = [c[: -len("_x_m")] for c in header if c.endswith("_x_m")]
        angle_columns_present = any(f"{name}_triangulation_angle_deg" in header for name in names)
        candidate_columns_present = any(f"{name}_candidate_n_views" in header for name in names)
        frames: list[dict[str, np.ndarray]] = []
        for row in reader:
            n = len(names)
            xyz = np.full((n, 3), np.nan)
            reproj = np.full(n, np.nan)
            angle = np.full(n, np.nan)
            candidate_n_views = np.full(n, np.nan)
            n_views = np.zeros(n, dtype=int)
            cheirality = np.zeros(n, dtype=bool)
            conf = np.zeros(n)
            for i, name in enumerate(names):
                xyz[i] = (
                    _cell(row[f"{name}_x_m"]),
                    _cell(row[f"{name}_y_m"]),
                    _cell(row[f"{name}_z_m"]),
                )
                reproj[i] = _cell(row[f"{name}_reproj_err_px"])
                conf[i] = _cell(row[f"{name}_confidence"])
                n_views[i] = int(row[f"{name}_n_views"] or 0)
                cheirality[i] = (row[f"{name}_cheirality_ok"] or "0") == "1"
                angle_col = f"{name}_triangulation_angle_deg"
                if angle_col in row:
                    angle[i] = _cell(row[angle_col])
                candidate_col = f"{name}_candidate_n_views"
                if candidate_col in row:
                    candidate_n_views[i] = _cell(row[candidate_col])
            trusted = (
                np.isfinite(xyz).all(axis=1)
                & np.isfinite(reproj)
                & (reproj <= REPROJ_GATE_PX)
                & cheirality
            )
            frames.append(
                {
                    "frame_idx": np.asarray(int(row.get("frame_idx") or len(frames))),
                    "xyz": xyz,
                    "trusted": trusted,
                    "reproj": reproj,
                    "candidate_n_views": candidate_n_views,
                    "n_views": n_views,
                    "cheirality": cheirality,
                    "conf": conf,
                    "angle": angle,
                }
            )
    # Once the angle column exists it is part of the trust contract: a blank
    # value is missing evidence for that point, not permission to bypass the
    # geometry gate. Only truly legacy schemas without the column opt out.
    angle_available = angle_columns_present
    candidate_views_available = candidate_columns_present and any(
        np.isfinite(fr["candidate_n_views"]).any() for fr in frames
    )
    for fr in frames:
        if angle_available:
            fr["trusted"] &= np.isfinite(fr["angle"]) & (
                fr["angle"] >= TRIANGULATION_ANGLE_GATE_DEG
            )
        trusted_xyz = fr["xyz"].copy()
        trusted_xyz[~fr["trusted"]] = np.nan
        fr["trusted_xyz"] = trusted_xyz
        fr["angle_available"] = np.asarray(angle_available)
        fr["candidate_views_available"] = np.asarray(candidate_views_available)
    frames.sort(key=lambda fr: int(fr["frame_idx"]))
    return names, frames


def _measure_fusion(world3d_path: pathlib.Path, expected_frames: int) -> Fusion3DSection:
    names, frames = _read_world3d(world3d_path)
    frames = _expected_world_frames(frames, expected_frames)
    n = len(names)
    active_mask = np.zeros(n, dtype=bool)  # keypoints attempted or fused in >= 1 frame
    reproj_vals: list[float] = []
    views_fused: list[int] = []
    angle_vals: list[float] = []
    candidate_vals: list[float] = []
    rejected_candidate_views = 0.0
    candidate_view_total = 0.0
    cheirality_fail = 0
    cheirality_total = 0
    for fr in frames:
        finite = np.isfinite(fr["xyz"]).all(axis=1)
        candidate_finite = np.isfinite(fr["candidate_n_views"])
        attempted = np.where(candidate_finite, fr["candidate_n_views"], fr["n_views"])
        active_mask |= finite | (attempted >= 2)
        reproj_vals.extend(fr["reproj"][np.isfinite(fr["reproj"])].tolist())
        angle_vals.extend(fr["angle"][np.isfinite(fr["angle"])].tolist())
        eligible_candidates = candidate_finite & (fr["candidate_n_views"] >= 2)
        if eligible_candidates.any():
            candidates = fr["candidate_n_views"][eligible_candidates]
            consensus = fr["n_views"][eligible_candidates]
            candidate_vals.extend(candidates.tolist())
            candidate_view_total += float(candidates.sum())
            rejected_candidate_views += float(np.maximum(candidates - consensus, 0.0).sum())
        views_fused.extend(fr["n_views"][finite].tolist())
        cheirality_total += int(finite.sum())
        cheirality_fail += int((finite & ~fr["cheirality"]).sum())

    active = int(active_mask.sum())
    # Unfused fraction over *active* keypoints only.  "Active" includes a
    # keypoint that reached the two-view policy but was rejected by geometry;
    # otherwise an always-rejected keypoint would disappear from the denominator.
    # A never-observed keypoint (n_views=0 throughout) remains a 2D tracking gap.
    slots = active * len(frames)
    unfused = 0
    if active:
        idx = np.flatnonzero(active_mask)
        for fr in frames:
            finite = np.isfinite(fr["xyz"][idx]).all(axis=1)
            unfused += int((~finite).sum())

    # ``frames`` is unique and restricted to the inferred expected source
    # interval; out-of-range rows therefore cannot substitute for gaps.
    observed_frame_count = len(frames)
    trusted_count = sum(int(fr["trusted"][active_mask].sum()) for fr in frames) if active else 0
    expected_slots = active * expected_frames

    reproj_arr = np.asarray(reproj_vals, dtype=np.float64)
    views_arr = np.asarray(views_fused, dtype=np.float64)
    angle_arr = np.asarray(angle_vals, dtype=np.float64)
    angle_available = bool(frames and bool(frames[0]["angle_available"]))
    candidate_arr = np.asarray(candidate_vals, dtype=np.float64)
    candidate_views_available = bool(frames and bool(frames[0]["candidate_views_available"]))
    return Fusion3DSection(
        n_frames_fused=observed_frame_count,
        expected_frames=expected_frames,
        fused_frame_fraction=(observed_frame_count / expected_frames)
        if expected_frames
        else float("nan"),
        n_active_keypoints=active,
        trusted_keypoint_fraction=(trusted_count / expected_slots)
        if expected_slots
        else float("nan"),
        reproj_err_px_median=float(np.median(reproj_arr)) if reproj_arr.size else float("nan"),
        reproj_err_px_p95=float(np.percentile(reproj_arr, 95)) if reproj_arr.size else float("nan"),
        reproj_err_px_max=float(np.max(reproj_arr)) if reproj_arr.size else float("nan"),
        n_views_median=float(np.median(views_arr)) if views_arr.size else float("nan"),
        n_views_min=int(np.min(views_arr)) if views_arr.size else 0,
        cheirality_violation_rate=(cheirality_fail / cheirality_total)
        if cheirality_total
        else float("nan"),
        triangulation_angle_available=angle_available,
        triangulation_angle_gate_deg=TRIANGULATION_ANGLE_GATE_DEG,
        triangulation_angle_deg_p05=float(np.percentile(angle_arr, 5))
        if angle_arr.size
        else float("nan"),
        triangulation_angle_deg_median=float(np.median(angle_arr))
        if angle_arr.size
        else float("nan"),
        candidate_views_available=candidate_views_available,
        candidate_n_views_median=float(np.median(candidate_arr))
        if candidate_arr.size
        else float("nan"),
        view_rejection_fraction=(rejected_candidate_views / candidate_view_total)
        if candidate_view_total
        else float("nan"),
        unfused_keypoint_fraction=(unfused / slots) if slots else float("nan"),
    )


# ---------------------------------------------------------------------------
# Stage 4 — clinical metrics + agreement / self-consistency
# ---------------------------------------------------------------------------


def _run_clinical(world3d_path: pathlib.Path, notes: list[str]) -> list[str]:
    """Run ``clinical_features.R`` on ``world3d.csv``; return output names.

    Degrades gracefully: a missing Rscript or a non-zero exit is noted,
    not fatal — the self-consistency surrogates do not depend on R.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        notes.append("The pipeline skipped clinical metrics. PATH does not contain Rscript.")
        return []
    try:
        result = subprocess.run(
            [rscript, str(_CLINICAL_R), str(world3d_path)],
            capture_output=True,
            text=True,
            timeout=_CLINICAL_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        notes.append(f"clinical_features.R timed out after {_CLINICAL_TIMEOUT_S}s.")
        return []
    if result.returncode != 0:
        notes.append(
            f"clinical_features.R failed with exit code {result.returncode}: "
            f"{result.stderr.strip()[:200]}"
        )
        return []
    outputs = sorted(p.name for p in world3d_path.parent.glob("*_clinical_3d*.csv"))
    if not outputs:
        notes.append("clinical_features.R produced no *_clinical_3d.csv files.")
    return outputs


def _build_agreement(
    world3d_path: pathlib.Path,
    session_out: pathlib.Path,
    baseline: str | pathlib.Path | None,
    clinical_outputs: list[str],
    notes: list[str],
) -> AgreementSection:
    names, frames = _read_world3d(world3d_path)
    lengths = _bone_lengths(names, frames)
    bone_cv = _bone_length_cv(lengths)
    symmetry = _symmetry_rel_diff(lengths)
    jitter = _temporal_jitter_mm(names, frames)

    per_metric_error: dict[str, float] | None = None
    has_baseline = baseline is not None
    if has_baseline:
        per_metric_error = _baseline_agreement(baseline, session_out, clinical_outputs, notes)

    cv_vals = [v for v in bone_cv.values() if math.isfinite(v)]
    sym_vals = [v for v in symmetry.values() if math.isfinite(v)]
    return AgreementSection(
        has_baseline=has_baseline,
        clinical_csv_produced=bool(clinical_outputs),
        clinical_outputs=clinical_outputs,
        mean_bone_length_cv=float(np.mean(cv_vals)) if cv_vals else float("nan"),
        bone_length_cv=bone_cv,
        mean_symmetry_rel_diff=float(np.mean(sym_vals)) if sym_vals else float("nan"),
        symmetry_rel_diff=symmetry,
        temporal_jitter_mm=jitter,
        per_metric_error=per_metric_error,
    )


def _bone_lengths(names: list[str], frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per-bone trusted 3D lengths (NaN where either endpoint is untrusted)."""
    index = {name: i for i, name in enumerate(names)}
    lengths: dict[str, np.ndarray] = {}
    for label, a, b in _BONES:
        if a not in index or b not in index:
            continue
        ia, ib = index[a], index[b]
        vals = np.array(
            [
                float(
                    np.linalg.norm(
                        fr.get("trusted_xyz", fr["xyz"])[ia] - fr.get("trusted_xyz", fr["xyz"])[ib]
                    )
                )
                for fr in frames
            ]
        )
        if np.isfinite(vals).any():
            lengths[label] = vals
    return lengths


def _bone_length_cv(lengths: dict[str, np.ndarray]) -> dict[str, float]:
    """Coefficient of variation (std/mean) of each bone length over frames."""
    cv: dict[str, float] = {}
    for label, vals in lengths.items():
        finite = vals[np.isfinite(vals)]
        mean = float(np.mean(finite)) if finite.size else float("nan")
        # A single observation has zero sample variance by construction but
        # provides no evidence of temporal rigidity.  Surface it as unmeasured
        # (NaN -> WARN), never a false PASS.
        cv[label] = (
            float(np.std(finite) / mean) if finite.size >= 2 and mean > 1e-9 else float("nan")
        )
    return cv


def _symmetry_rel_diff(lengths: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean relative left/right difference |L-R| / mean(L,R) per symmetric pair."""
    out: dict[str, float] = {}
    for left, right in _SYMMETRIC_BONES:
        if left not in lengths or right not in lengths:
            continue
        lvals, rvals = lengths[left], lengths[right]
        both = np.isfinite(lvals) & np.isfinite(rvals)
        if not both.any():
            continue
        mean = (lvals[both] + rvals[both]) / 2.0
        rel = np.abs(lvals[both] - rvals[both]) / np.where(mean > 1e-9, mean, np.nan)
        label = left.replace("left_", "").replace("body_", "")
        out[label] = float(np.nanmean(rel))
    return out


def _temporal_jitter_mm(names: list[str], frames: list[dict[str, np.ndarray]]) -> float:
    """Median per-keypoint high-frequency jitter (mm).

    Uses the magnitude of the second temporal difference (acceleration),
    which isolates frame-to-frame noise from steady movement.  Returns
    NaN with fewer than three frames.
    """
    del names  # keypoint labels do not affect the temporal statistic
    if len(frames) < 3:
        return float("nan")
    stack = np.stack([fr.get("trusted_xyz", fr["xyz"]) for fr in frames])  # (T, N, 3)
    frame_idx = np.asarray(
        [int(fr.get("frame_idx", np.asarray(i))) for i, fr in enumerate(frames)],
        dtype=np.int64,
    )
    contiguous = (np.diff(frame_idx[:-1]) == 1) & (np.diff(frame_idx[1:]) == 1)
    if not contiguous.any():
        return float("nan")
    accel = stack[2:] - 2.0 * stack[1:-1] + stack[:-2]  # (T-2, N, 3)
    accel[~contiguous] = np.nan
    mag = np.linalg.norm(accel, axis=2)  # (T-2, N)
    # Untracked keypoints (present in the full skeleton header but never
    # fused) form all-NaN columns; their per-keypoint median is NaN by
    # design and is dropped by the finite filter below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_kp = np.nanmedian(mag, axis=0)  # (N,)
    finite = per_kp[np.isfinite(per_kp)]
    return float(np.median(finite) * 1000.0) if finite.size else float("nan")


def _baseline_agreement(
    baseline: str | pathlib.Path,
    session_out: pathlib.Path,
    clinical_outputs: list[str],
    notes: list[str],
) -> dict[str, float] | None:
    """Per-metric absolute error of clinical aggregates vs a baseline JSON.

    Baseline JSON: ``{metric_column_name: reference_value}``.  The
    harness aggregates each named column's mean from the produced
    ``*_clinical_3d.csv`` and reports ``|aggregate - reference|``.  A
    minimal contract to extend with Bland-Altman, ICC, and percent error
    once a real baseline exists.
    """
    bpath = pathlib.Path(baseline)
    if not bpath.is_file():
        notes.append(f"The baseline file does not exist: {bpath}.")
        return None
    reference = json.loads(bpath.read_text())
    if not clinical_outputs:
        notes.append("The report has a baseline but no clinical CSV for comparison.")
        return None
    aggregates = _aggregate_clinical(session_out, clinical_outputs)
    errors: dict[str, float] = {}
    for metric, ref in reference.items():
        if metric in aggregates and isinstance(ref, (int, float)):
            errors[metric] = abs(float(aggregates[metric]) - float(ref))
        else:
            notes.append(f"The clinical output does not contain baseline metric {metric!r}.")
    return errors


def _aggregate_clinical(session_out: pathlib.Path, clinical_outputs: list[str]) -> dict[str, float]:
    """Column means across the per-frame clinical CSV(s)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in clinical_outputs:
        if "windows" in name or "movement_phases" in name:
            continue
        with (session_out / name).open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                for col, cell in row.items():
                    val = _cell(cell)
                    if math.isfinite(val):
                        sums[col] = sums.get(col, 0.0) + val
                        counts[col] = counts.get(col, 0) + 1
    return {col: sums[col] / counts[col] for col in sums if counts[col]}


# ---------------------------------------------------------------------------
# Grading — report → PASS/WARN/FAIL verdict
# ---------------------------------------------------------------------------


def _grade_report(report: ValidationReport, thresholds: Thresholds) -> Verdict:
    """Grade every report metric against *thresholds*; worst wins.

    Timing throughput and L/R symmetry are graded but flagged
    ``informational`` — they are surfaced yet excluded from the overall
    grade (performance is not clinical validity; symmetry is valid only
    on symmetric-by-construction input).  The clinical-metric *agreement*
    leg contributes only when a baseline supplies angle (``_deg``)
    metrics — none exists yet (UNVALIDATED; see the gap register).
    """
    cal = report.calibration
    fus = report.fusion_3d
    trk = report.tracking_2d
    tim = report.timing
    agr = report.agreement
    checks: list[Check] = []
    notes: list[str] = []

    def _check(name: str, value: float | None, band: Band, *, info: bool = False) -> None:
        checks.append(
            Check(
                name=name,
                value=value,
                grade=band.grade(value).name,
                detail=band.describe(value),
                informational=info,
            )
        )

    _check(
        "calibration.reprojection_error_px",
        cal.reprojection_error_px,
        thresholds.calib_reproj_rms_px,
    )
    _check(
        "fusion.reproj_err_px_median", fus.reproj_err_px_median, thresholds.fusion_reproj_median_px
    )
    _check("fusion.reproj_err_px_p95", fus.reproj_err_px_p95, thresholds.fusion_reproj_p95_px)
    _check("fusion.n_views_median", fus.n_views_median, thresholds.n_views_median)
    _check(
        "fusion.fused_frame_fraction",
        fus.fused_frame_fraction,
        thresholds.min_fused_frame_fraction,
    )
    _check(
        "fusion.trusted_keypoint_fraction",
        fus.trusted_keypoint_fraction,
        thresholds.min_trusted_keypoint_fraction,
    )
    if fus.triangulation_angle_available:
        _check(
            "fusion.triangulation_angle_deg_p05",
            fus.triangulation_angle_deg_p05,
            thresholds.min_triangulation_angle_p05_deg,
        )
    else:
        notes.append(
            "world3d has no triangulation-angle columns (legacy schema). "
            "The report does not grade angle quality."
        )
    if fus.candidate_views_available:
        _check(
            "fusion.view_rejection_fraction",
            fus.view_rejection_fraction,
            thresholds.max_view_rejection_fraction,
        )
    else:
        notes.append(
            "world3d has no candidate-view diagnostics (legacy schema). "
            "The report does not grade consensus view rejection."
        )

    # Hard floor: a fused keypoint below min views means malformed /
    # degenerate fusion (DLT needs >= 2).  A discrete guard, not a Band.
    floor_ok = fus.n_views_min >= thresholds.n_views_floor
    checks.append(
        Check(
            name="fusion.n_views_min",
            value=fus.n_views_min,
            grade=(Grade.PASS if floor_ok else Grade.FAIL).name,
            detail=f"min {fus.n_views_min} (floor >= {thresholds.n_views_floor})",
        )
    )

    worst_low_conf = max(
        (
            c.low_confidence_fraction
            for c in trk.cameras
            if math.isfinite(c.low_confidence_fraction)
        ),
        default=float("nan"),
    )
    _check(
        "tracking.worst_low_confidence_fraction",
        worst_low_conf,
        thresholds.max_low_confidence_fraction,
    )
    worst_detection = min(
        (c.detection_rate for c in trk.cameras if math.isfinite(c.detection_rate)),
        default=float("nan"),
    )
    _check(
        "tracking.worst_detection_rate",
        worst_detection,
        thresholds.min_tracking_detection_rate,
    )
    _check(
        "fusion.unfused_keypoint_fraction",
        fus.unfused_keypoint_fraction,
        thresholds.max_unfused_fraction,
    )
    _check(
        "fusion.cheirality_violation_rate",
        fus.cheirality_violation_rate,
        thresholds.max_cheirality_rate,
    )

    # Self-consistency surrogates valid on ANY input (rigid bones, smooth
    # motion) — the substantive clinical evidence absent a baseline.
    _check("agreement.mean_bone_length_cv", agr.mean_bone_length_cv, thresholds.max_bone_length_cv)
    _check(
        "agreement.temporal_jitter_mm", agr.temporal_jitter_mm, thresholds.max_temporal_jitter_mm
    )

    # Informational: surfaced, never escalates the overall grade.
    _check("timing.throughput_fps", tim.throughput_fps, thresholds.min_throughput_fps, info=True)
    _check(
        "agreement.mean_symmetry_rel_diff",
        agr.mean_symmetry_rel_diff,
        thresholds.max_symmetry_rel_diff,
        info=True,
    )

    # Baseline agreement: angle metrics only. Add unit-aware grading,
    # Bland-Altman, and ICC once a real baseline exists.
    if agr.per_metric_error:
        graded = 0
        for metric, err in sorted(agr.per_metric_error.items()):
            if metric.endswith("_deg"):
                _check(f"agreement.{metric}", err, thresholds.agreement_tolerance_deg)
                graded += 1
            else:
                notes.append(
                    f"The report does not grade baseline metric {metric!r} because the metric "
                    "has no unit-aware tolerance."
                )
        if not graded:
            notes.append("The supplied baseline has no angle (_deg) metric to grade.")
    elif agr.has_baseline:
        notes.append("The baseline produced no per-metric error. See the report notes.")
    else:
        notes.append(
            "No baseline is available. Clinical-metric agreement remains UNVALIDATED. "
            "The verdict uses only self-consistency surrogates and reprojection "
            "(see the gap register)."
        )

    notes.append(
        "Timing throughput and L/R symmetry are informational. They do not affect the grade."
    )
    overall = max((Grade[c.grade] for c in checks if not c.informational), default=Grade.PASS)
    return Verdict(
        grade=overall.name,
        thresholds_version=thresholds.version,
        checks=checks,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Pre-flight capture QA gate
# ---------------------------------------------------------------------------

QA_REPORT_SCHEMA_VERSION = 1
"""Bumped when the :class:`QAReport` JSON layout changes."""


@dataclasses.dataclass
class CharucoCameraQA:
    """Per-camera ChArUco calibration-capture quality."""

    name: str
    n_frames: int  # total frames after the camera's sync_offset
    n_detected: int  # frames carrying a usable board detection
    detection_rate: float
    coverage: float  # fraction of COVERAGE_GRID cells the board swept


@dataclasses.dataclass
class CalibrationQA:
    """Calibration-capture quality: board sweep + solve RMS."""

    assessed: bool  # raw ChArUco session videos were available for coverage
    reprojection_error_px: float
    solved: bool  # RMS from a solve this run (vs a loaded calibration.json)
    cameras: list[CharucoCameraQA]


@dataclasses.dataclass
class ParityQA:
    """Frame-count parity across the subject cameras (desync proxy)."""

    frame_counts: dict[str, int]
    disparity: float  # (max - min) / max; 0.0 == perfect parity


@dataclasses.dataclass
class SubjectQA:
    """Subject-clip 2D tracking quality."""

    assessed: bool
    tracking: Tracking2DSection | None
    worst_detection_rate: float
    worst_low_confidence_fraction: float


@dataclasses.dataclass
class QAReport:
    """Pre-flight QA grade of a raw capture.

    Graded by :func:`qa_check` *before* clinical metrics are trusted, so a
    capture that fails QA (poor board coverage, desync, low subject
    detection) is caught here rather than yielding plausible-but-wrong 3D
    downstream.  ``to_json`` / ``to_markdown`` / ``verdict`` mirror
    :class:`ValidationReport`; the verdict grades against
    :data:`QA_THRESHOLDS` (plus the shared bands in :data:`THRESHOLDS`).
    """

    session_id: str
    schema_version: int
    qa_thresholds_version: int
    calibration: CalibrationQA
    parity: ParityQA
    subject: SubjectQA
    notes: list[str]

    def verdict(
        self,
        thresholds: Thresholds = THRESHOLDS,
        qa_thresholds: QAThresholds = QA_THRESHOLDS,
    ) -> Verdict:
        """Grade this capture → PASS/WARN/FAIL against the QA thresholds."""
        return _grade_qa(self, thresholds, qa_thresholds)

    def to_json(self) -> dict[str, Any]:
        payload = _native(dataclasses.asdict(self))
        payload["verdict"] = _native(dataclasses.asdict(self.verdict()))
        return payload

    def to_markdown(self) -> str:
        return _render_qa_markdown(self)


def qa_check(
    session_dir: str | pathlib.Path,
    *,
    calibration: str | pathlib.Path | None = None,
    output_dir: str | pathlib.Path | None = None,
    camera_processor: Any = None,
    det_device: str = "CPU",
    pose_device: str = "NPU",
    backend: str = "onnxruntime",
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    board: Any = None,
) -> QAReport:
    """Grade a raw 3-camera capture before its clinical metrics are trusted.

    The pre-flight gate assesses three failure
    surfaces without running the full fusion/clinical chain:

    1. **Calibration capture** — per-camera ChArUco detection rate and
       field-of-view coverage (does the board sweep the working volume?)
       plus the solved/loaded reprojection RMS.  ``calibration`` should be
       the raw ChArUco *session directory* for coverage; a
       ``calibration.json`` file (or ``None`` → the subject session's own)
       yields RMS only.
    2. **Synchronization** — raw frame-count parity across the subject
       cameras, a software-sync desync proxy.
    3. **Subject clip** — per-camera 2D detection rate (same three-way
       source as :func:`run_validation`: injected ``camera_processor`` →
       reuse existing CSVs → default backend; degrades to *unassessed*
       rather than raising when none is available).

    Returns a :class:`QAReport`; grade it with ``report.verdict()``.
    """
    session_dir = pathlib.Path(session_dir)
    notes: list[str] = []
    session = discover_session(session_dir)
    board = make_charuco_board() if board is None else board

    calibration_qa = _qa_calibration(session, calibration, board, notes)
    parity_qa = _qa_parity(session)
    subject_qa = _qa_subject(
        session,
        output_dir,
        camera_processor,
        det_device,
        pose_device,
        backend,
        confidence_floor,
        notes,
    )
    return QAReport(
        session_id=session.session_id,
        schema_version=QA_REPORT_SCHEMA_VERSION,
        qa_thresholds_version=QA_THRESHOLDS_VERSION,
        calibration=calibration_qa,
        parity=parity_qa,
        subject=subject_qa,
        notes=notes,
    )


def _qa_calibration(
    session: Session,
    calibration: str | pathlib.Path | None,
    board: Any,
    notes: list[str],
) -> CalibrationQA:
    """Reprojection RMS + per-camera board coverage / detection rate."""
    # Reprojection RMS (+ solved flag), reusing run_validation's resolver.
    reproj = float("nan")
    solved = False
    if calibration is not None:
        try:
            calib, solved = _resolve_external_calibration(calibration)
            reproj = float(calib["reprojection_error_px"])
        except (ValidationError, CalibrationError, SessionError) as exc:
            notes.append(f"The report leaves calibration RMS unassessed: {exc}")
    elif session.calibration is not None:
        reproj = float(session.calibration["reprojection_error_px"])
    else:
        notes.append(
            "The report leaves reprojection RMS unassessed because no calibration is available."
        )

    # Per-camera coverage/detection needs the raw ChArUco videos, so it runs
    # only when ``calibration`` resolves to a session directory.
    cameras: list[CharucoCameraQA] = []
    calib_dir = pathlib.Path(calibration) if calibration is not None else None
    if calib_dir is not None and calib_dir.is_dir():
        try:
            calib_session = discover_session(calib_dir)
        except SessionError as exc:
            notes.append(f"The report leaves calibration board coverage unassessed: {exc}")
        else:
            for cam in calib_session.cameras:
                video = calib_session.directory / cam.file
                cameras.append(_charuco_camera_qa(cam.name, video, cam.sync_offset, board))
    elif calib_dir is not None:
        notes.append(
            "The report cannot assess board coverage or detection from a calibration file. "
            "Only RMS is available."
        )
    else:
        notes.append(
            "The report cannot assess board coverage or detection because no ChArUco session "
            "was supplied."
        )

    return CalibrationQA(
        assessed=bool(cameras),
        reprojection_error_px=reproj,
        solved=solved,
        cameras=cameras,
    )


def _charuco_camera_qa(
    name: str, video: pathlib.Path, sync_offset: int, board: Any
) -> CharucoCameraQA:
    """Detect the board across *video* → detection rate + FOV coverage."""
    total = max(0, frame_count(video) - sync_offset)
    try:
        detections, size = detect_charuco_corners(video, board, sync_offset=sync_offset)
    except CalibrationError:
        detections, size = [], (0, 0)
    n_detected = len(detections)
    return CharucoCameraQA(
        name=name,
        n_frames=total,
        n_detected=n_detected,
        detection_rate=(n_detected / total) if total else float("nan"),
        coverage=_board_coverage(detections, size),
    )


def _board_coverage(detections: list, frame_size: tuple[int, int]) -> float:
    """Fraction of the COVERAGE_GRID image cells the detected board touched.

    Pools every detected ChArUco corner across all frames into a
    cols x rows grid over the frame; returns the fraction of cells holding
    at least one corner.  A full-volume sweep lights up most cells, a
    centre-bound board few.  NaN when there is nothing to measure.
    """
    cols, rows = COVERAGE_GRID
    w, h = frame_size
    if not detections or w <= 0 or h <= 0:
        return float("nan")
    occupied: set[tuple[int, int]] = set()
    for det in detections:
        cx = np.clip((det.corners[:, 0] / w * cols).astype(int), 0, cols - 1)
        cy = np.clip((det.corners[:, 1] / h * rows).astype(int), 0, rows - 1)
        occupied.update(zip(cx.tolist(), cy.tolist(), strict=True))
    return len(occupied) / (cols * rows)


def _qa_parity(session: Session) -> ParityQA:
    """Raw per-camera frame counts + their normalised disparity."""
    counts = {cam.name: frame_count(session.directory / cam.file) for cam in session.cameras}
    values = list(counts.values())
    maximum = max(values, default=0)
    disparity = (
        (maximum - min(values)) / maximum if len(values) >= 2 and maximum > 0 else float("nan")
    )
    return ParityQA(frame_counts=counts, disparity=disparity)


def _qa_subject(
    session: Session,
    output_dir: str | pathlib.Path | None,
    camera_processor: Any,
    det_device: str,
    pose_device: str,
    backend: str,
    confidence_floor: float,
    notes: list[str],
) -> SubjectQA:
    """Per-camera 2D detection on the subject clip (reuses run_validation)."""
    session_out = _resolve_session_output(session, output_dir)
    try:
        _per_camera_time, reused = _run_tracking(
            session,
            session_out,
            output_dir,
            camera_processor,
            det_device,
            pose_device,
            backend,
            notes,
        )
        tracking = _measure_tracking(session, session_out, confidence_floor, reused, notes)
    except (ValidationError, SessionError) as exc:
        notes.append(f"The report leaves subject 2D detection unassessed: {exc}")
        return SubjectQA(
            assessed=False,
            tracking=None,
            worst_detection_rate=float("nan"),
            worst_low_confidence_fraction=float("nan"),
        )
    rates = [c.detection_rate for c in tracking.cameras if math.isfinite(c.detection_rate)]
    lows = [
        c.low_confidence_fraction
        for c in tracking.cameras
        if math.isfinite(c.low_confidence_fraction)
    ]
    return SubjectQA(
        assessed=True,
        tracking=tracking,
        worst_detection_rate=min(rates) if rates else float("nan"),
        worst_low_confidence_fraction=max(lows) if lows else float("nan"),
    )


def _grade_qa(report: QAReport, thresholds: Thresholds, qa_thresholds: QAThresholds) -> Verdict:
    """Grade a :class:`QAReport` → PASS/WARN/FAIL; worst check wins."""
    cal = report.calibration
    par = report.parity
    sub = report.subject
    checks: list[Check] = []
    notes: list[str] = []

    def _check(name: str, value: float | None, band: Band) -> None:
        checks.append(
            Check(
                name=name,
                value=value,
                grade=band.grade(value).name,
                detail=band.describe(value),
            )
        )

    # Calibration RMS shares its band with the output-report verdict.
    _check(
        "calibration.reprojection_error_px",
        cal.reprojection_error_px,
        thresholds.calib_reproj_rms_px,
    )
    if cal.cameras:
        worst_cov = min(
            (c.coverage for c in cal.cameras if math.isfinite(c.coverage)), default=float("nan")
        )
        worst_rate = min(
            (c.detection_rate for c in cal.cameras if math.isfinite(c.detection_rate)),
            default=float("nan"),
        )
        min_frames = min((c.n_detected for c in cal.cameras), default=0)
        _check("calibration.worst_board_coverage", worst_cov, qa_thresholds.min_board_coverage)
        _check(
            "calibration.worst_charuco_detection_rate",
            worst_rate,
            qa_thresholds.min_charuco_detection_rate,
        )
        floor_ok = min_frames >= qa_thresholds.min_charuco_frames
        checks.append(
            Check(
                name="calibration.min_charuco_frames",
                value=min_frames,
                grade=(Grade.PASS if floor_ok else Grade.FAIL).name,
                detail=(
                    f"min {min_frames} usable board views "
                    f"(floor >= {qa_thresholds.min_charuco_frames})"
                ),
            )
        )
    else:
        notes.append(
            "The report leaves calibration board coverage unassessed because no raw ChArUco "
            "session videos are available."
        )

    # Synchronization: frame-count parity (desync proxy).
    _check("parity.frame_count_disparity", par.disparity, qa_thresholds.max_frame_count_disparity)

    # Subject clip 2D detection.
    if sub.assessed:
        _check(
            "subject.worst_detection_rate",
            sub.worst_detection_rate,
            qa_thresholds.min_subject_detection_rate,
        )
        _check(
            "subject.worst_low_confidence_fraction",
            sub.worst_low_confidence_fraction,
            thresholds.max_low_confidence_fraction,
        )
    else:
        notes.append(
            "The report leaves subject 2D detection unassessed because no per-camera CSVs or "
            "backend run are available."
        )

    overall = max((Grade[c.grade] for c in checks), default=Grade.PASS)
    return Verdict(
        grade=overall.name,
        thresholds_version=qa_thresholds.version,
        checks=checks,
        notes=notes,
    )


def _render_qa_markdown(report: QAReport) -> str:
    cal = report.calibration
    par = report.parity
    sub = report.subject
    v = report.verdict()
    lines = [
        f"# Capture QA — `{report.session_id}`",
        "",
        f"_QA schema v{report.schema_version} · QA thresholds v{report.qa_thresholds_version}._",
        "",
        f"## Verdict: {_GRADE_MARK.get(v.grade, v.grade)} **{v.grade}**",
        "",
        "| Check | Value | Grade | Details |",
        "|-------|-------|-------|---------|",
    ]
    for c in v.checks:
        mark = _GRADE_MARK.get(c.grade, c.grade)
        lines.append(f"| `{c.name}` | {_fmt(_cell(c.value))} | {mark} | {c.detail} |")
    lines += [
        "",
        "## Calibration capture",
        "",
        f"- Reprojection RMS: {_fmt(cal.reprojection_error_px)} px "
        f"({'solved' if cal.solved else 'loaded'})",
    ]
    if cal.cameras:
        lines += [
            "",
            "| Camera | Frames | Detected frames | Detection rate | Board coverage |",
            "|--------|--------|-----------------|----------------|----------------|",
        ]
        for c in cal.cameras:
            lines.append(
                f"| {c.name} | {c.n_frames} | {c.n_detected} | "
                f"{_fmt(c.detection_rate)} | {_fmt(c.coverage)} |"
            )
    else:
        lines.append(
            "- The report leaves board coverage unassessed because no raw ChArUco session "
            "was supplied."
        )
    counts = ", ".join(f"{k}={n}" for k, n in par.frame_counts.items())
    lines += [
        "",
        "## Frame-count parity (desynchronization proxy)",
        "",
        f"- Frame-count disparity: {_fmt(par.disparity)} ({counts})",
        "",
        "## Subject clip",
        "",
    ]
    if sub.assessed:
        lines += [
            f"- Worst per-camera detection rate: {_fmt(sub.worst_detection_rate)}",
            f"- Worst low-confidence fraction: {_fmt(sub.worst_low_confidence_fraction)}",
        ]
    else:
        lines.append("- The report leaves subject 2D detection unassessed.")
    allnotes = [*report.notes, *v.notes]
    if allnotes:
        lines += ["", "## Notes", ""]
        lines += [f"- {n}" for n in allnotes]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _cell(value: Any) -> float:
    """Parse a CSV cell: blank/None → NaN."""
    try:
        return float(value) if value not in ("", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _native(obj: Any) -> Any:
    """Coerce to JSON-native types; non-finite floats → ``None``."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _fmt(value: float | None, ndigits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{ndigits}f}"


_GRADE_MARK = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}


def _render_markdown(report: ValidationReport) -> str:
    cal = report.calibration
    trk = report.tracking_2d
    fus = report.fusion_3d
    tim = report.timing
    agr = report.agreement
    v = report.verdict()
    lines = [
        f"# Validation report — `{report.session_id}`",
        "",
        f"_Report schema v{report.schema_version} · thresholds v{v.thresholds_version}._",
        "",
        f"## Verdict: {_GRADE_MARK.get(v.grade, v.grade)} **{v.grade}**",
        "",
        "| Check | Value | Grade |",
        "|-------|-------|-------|",
    ]
    lines += [
        f"| {c.name}{' _(info)_' if c.informational else ''} | {c.detail} | "
        f"{_GRADE_MARK.get(c.grade, c.grade)} |"
        for c in v.checks
    ]
    if v.notes:
        lines += ["", *[f"> {n}" for n in v.notes]]
    lines += [
        "",
        "## Calibration",
        f"- Cameras: **{cal.n_cameras}**; world frame: `{cal.world_frame}`",
        f"- Reprojection RMS: **{_fmt(cal.reprojection_error_px)} px** "
        f"({'solved this run' if cal.solved else 'loaded'})",
        "",
        "| Camera | Resolution | fx | fy | cx | cy | ‖dist‖ |",
        "|--------|------------|----|----|----|----|--------|",
    ]
    lines += [
        f"| {c.name} | {c.resolution[0]}x{c.resolution[1]} | {_fmt(c.fx, 1)} | "
        f"{_fmt(c.fy, 1)} | {_fmt(c.cx, 1)} | {_fmt(c.cy, 1)} | {_fmt(c.distortion_l2)} |"
        for c in cal.cameras
    ]
    lines += [
        "",
        "## 2D tracking",
        f"- Confidence floor: {_fmt(trk.confidence_floor, 2)}",
        f"- Reused existing CSVs: {trk.reused_existing_csvs}",
        f"- Frames observed/expected: {trk.total_observed_frames}/"
        f"{trk.total_expected_frames}; mean detection rate: "
        f"**{_fmt(trk.mean_detection_rate)}**",
        "",
        "| Camera | Observed frames | Expected frames | Detection rate | Low-confidence fraction | Dropped frames |",
        "|--------|-----------------|-----------------|----------------|-------------------------|----------------|",
    ]
    lines += [
        f"| {c.name} | {c.observed_frames} | {c.expected_frames} | "
        f"{_fmt(c.detection_rate)} | "
        f"{_fmt(c.low_confidence_fraction)} | {c.dropped_frames} |"
        for c in trk.cameras
    ]
    lines += [
        "",
        "## 3D fusion",
        f"- Frames fused/expected: **{fus.n_frames_fused}/{fus.expected_frames}** "
        f"({_fmt(fus.fused_frame_fraction)}); active keypoints: {fus.n_active_keypoints}",
        f"- Trusted keypoint fraction (expected active slots): "
        f"**{_fmt(fus.trusted_keypoint_fraction)}**",
        f"- Reprojection error (px) — median {_fmt(fus.reproj_err_px_median)}, "
        f"p95 {_fmt(fus.reproj_err_px_p95)}, maximum {_fmt(fus.reproj_err_px_max)}",
        f"- Views per keypoint — median {_fmt(fus.n_views_median, 1)}, minimum {fus.n_views_min}",
        f"- Cheirality violation rate: {_fmt(fus.cheirality_violation_rate)}",
        f"- Triangulation angle (deg) — "
        f"{'p05 ' + _fmt(fus.triangulation_angle_deg_p05) + ', median ' + _fmt(fus.triangulation_angle_deg_median) if fus.triangulation_angle_available else 'unavailable (legacy world3d)'}; "
        f"trust gate >= {_fmt(fus.triangulation_angle_gate_deg)}",
        f"- Candidate views — "
        f"{'median ' + _fmt(fus.candidate_n_views_median, 1) + ', rejected fraction ' + _fmt(fus.view_rejection_fraction) if fus.candidate_views_available else 'unavailable (legacy world3d)'}",
        f"- Unfused-keypoint fraction (active): {_fmt(fus.unfused_keypoint_fraction)}",
        "",
        "## Timing",
        f"- Detector device: `{tim.det_device}`; pose device: `{tim.pose_device}`; "
        f"backend: `{tim.backend}`",
        f"- Solve: {_fmt(tim.solve_sec)} s · 2D tracking: {_fmt(tim.tracking_2d_sec)} s · "
        f"fusion: {_fmt(tim.fusion_sec)} s · clinical: {_fmt(tim.clinical_sec)} s · "
        f"total: {_fmt(tim.total_sec)} s",
        f"- Throughput: {_fmt(tim.throughput_fps, 1)} fused fps",
        "",
        "## Agreement and self-consistency",
        f"- Baseline supplied: {agr.has_baseline}; clinical CSV produced: "
        f"{agr.clinical_csv_produced}",
        f"- Clinical outputs: {', '.join(agr.clinical_outputs) or 'none'}",
        f"- Mean bone-length CV: **{_fmt(agr.mean_bone_length_cv)}**",
        f"- Mean left/right symmetry relative difference: **{_fmt(agr.mean_symmetry_rel_diff)}**",
        f"- Temporal jitter: {_fmt(agr.temporal_jitter_mm, 2)} mm",
    ]
    if agr.per_metric_error:
        lines.append("- Baseline per-metric absolute error:")
        lines += [f"  - {m}: {_fmt(e)}" for m, e in sorted(agr.per_metric_error.items())]
    if report.notes:
        lines += ["", "## Notes"] + [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Console entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pose-estimation-validate",
        description="Run the 3D clinical pipeline on one session. Report its quality.",
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Select a session directory (cam*.mp4 + ...).",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Select calibration.json or a ChArUco calibration-session directory to solve.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Read an optional reference-metrics JSON file for clinical agreement.",
    )
    parser.add_argument(
        "--det-device",
        default="CPU",
        help="Select the detector inference device: NPU, CPU, or GPU (default: CPU).",
    )
    parser.add_argument(
        "--pose-device",
        default="NPU",
        help="Select the pose-model inference device: NPU, CPU, or GPU (default: NPU).",
    )
    parser.add_argument(
        "--backend",
        default="onnxruntime",
        help="Select the rtmlib backend: onnxruntime or openvino (default: onnxruntime).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write per-camera CSVs and world3d to this directory.",
    )
    parser.add_argument(
        "--out",
        default="report.json",
        help="Write the report JSON to this path (default: report.json).",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Also write the Markdown summary to this path.",
    )
    parser.add_argument(
        "--no-clinical",
        action="store_true",
        help="Skip the R clinical-metrics stage.",
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Run only the pre-flight capture QA gate. Skip the fusion and clinical chain.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat a WARN verdict as a failure. Exit nonzero on WARN or FAIL.",
    )
    return parser.parse_args(argv)


def _emit(report: Any, out: str, markdown: str | None, strict: bool) -> int:
    """Write the report JSON (+ optional Markdown), print the verdict, return
    the CLI exit code.  Duck-typed over :class:`ValidationReport` and
    :class:`QAReport` (both expose ``to_json`` / ``to_markdown`` / ``verdict``).
    """
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_json(), indent=2))
    print(f"Wrote {out_path}")

    rendered = report.to_markdown()
    if markdown:
        md_path = pathlib.Path(markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(rendered)
        print(f"Wrote {md_path}")
    else:
        print()
        print(rendered)

    v = report.verdict()
    print(f"Verdict: {v.grade} (thresholds v{v.thresholds_version})")
    failed = v.grade == Grade.FAIL.name or (strict and v.grade == Grade.WARN.name)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.qa_only:
            report: Any = qa_check(
                args.session_dir,
                calibration=args.calibration,
                output_dir=args.output_dir,
                det_device=args.det_device,
                pose_device=args.pose_device,
                backend=args.backend,
            )
        else:
            report = run_validation(
                args.session_dir,
                calibration=args.calibration,
                baseline=args.baseline,
                det_device=args.det_device,
                pose_device=args.pose_device,
                backend=args.backend,
                output_dir=args.output_dir,
                run_clinical=not args.no_clinical,
            )
    except (ValidationError, SessionError, CalibrationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return _emit(report, args.out, args.markdown, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
