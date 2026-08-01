"""Failure-mode tests for the ``pose_estimation.validation`` harness.

Clinical safety rests on the harness *surfacing* a degraded capture
rather than silently emitting a plausible-but-wrong report.  Each test
injects one known degradation into the shared synthetic 3-camera session
(``rendered_session`` + ``synthetic_session``) and asserts the harness
**correctly identifies the fault**: the matching report field crosses its
threshold, the verdict degrades to WARN/FAIL, and bad data is routed to
NaN — never fabricated.

The tests account for robust minimal-set consensus: a miscalibrated or
desynchronised camera can be rejected while accepted-view reprojection stays
near zero, so candidate-to-consensus view rejection is asserted explicitly.
Injections are deterministic (fixed renders + solve, seeded noise).
"""

from __future__ import annotations

import copy
import csv
import json
import pathlib
import shutil

import numpy as np
import pytest

from pose_estimation._types import SessionCalibration
from pose_estimation.calibration import save_calibration
from pose_estimation.validation import run_validation
from pose_estimation.video_io import frame_count
from synthetic_session import DEFAULT_VELOCITY, N_SUBJECT_FRAMES, skeleton_processor

# Right-arm distal region (wrist + finger bases) and the full distal block.
_REGION = (5, 7, 9, 11)
_WIDE = (4, 5, 6, 7, 8, 9, 10, 11)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grades(report) -> dict[str, str]:
    """Map each verdict check name -> its grade."""
    return {c.name: c.grade for c in report.verdict().checks}


def _perturb_yaw(solved: SessionCalibration, camera: str, deg: float) -> SessionCalibration:
    """Deep-copy *solved*, adding *deg* of yaw to one non-world camera."""
    cal = copy.deepcopy(solved)
    cal["cameras"][camera]["rvec"] = cal["cameras"][camera]["rvec"] + np.array(
        [0.0, np.deg2rad(deg), 0.0]
    )
    return cal


def _degenerate_rig(solved: SessionCalibration, baseline_m: float) -> SessionCalibration:
    """A tiny-baseline, near-collinear rig: cam1 world, cam2/cam3 along +x.

    All cameras share intrinsics and look down +z from nearly the same
    point, so triangulation is ill-conditioned — small pixel noise blows
    up into large 3D displacement while 2D reprojection stays small.
    """
    cal = copy.deepcopy(solved)
    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]])
    for i, name in enumerate(("cam1", "cam2", "cam3")):
        cam = cal["cameras"][name]
        cam["K"] = K.copy()
        cam["distortion"] = np.zeros_like(cam["distortion"])
        cam["resolution"] = (1280, 720)
        cam["rvec"] = np.zeros(3)
        cam["tvec"] = np.array([baseline_m * i, 0.0, 0.0])
    return cal


def _save(cal: SessionCalibration, path: pathlib.Path) -> pathlib.Path:
    save_calibration(cal, path)
    return path


def _validate(
    rendered_session,
    tmp_path: pathlib.Path,
    *,
    calibration,
    processor,
    session_json: dict | None = None,
    tag: str = "work",
    complete_hands: bool = True,
):
    """Copy the clean session, optionally drop a manifest, run the harness."""
    session_dir, _solved = rendered_session
    work = tmp_path / tag
    shutil.copytree(session_dir, work)

    extra_offsets = {
        entry["name"]: int(entry.get("sync_offset", 0))
        for entry in (session_json or {}).get("cameras", [])
    }
    cameras = []
    base_offsets: dict[str, int] = {}
    for video in sorted(work.glob("cam*.avi")):
        base = max(0, frame_count(video) - N_SUBJECT_FRAMES)
        base_offsets[video.stem] = base
        cameras.append(
            {
                "name": video.stem,
                "file": video.name,
                "sync_offset": base + extra_offsets.get(video.stem, 0),
            }
        )
    (work / "session.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "session_id": (session_json or {}).get("session_id", tag),
                "cameras": cameras,
            }
        )
    )

    def aligned_processor(**kwargs):
        processor(**kwargs)
        csv_path = pathlib.Path(kwargs["output_csv"])
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row["frame_idx"] = str(int(row["frame_idx"]) + base_offsets[csv_path.stem])
                if complete_hands:
                    for side in ("left", "right"):
                        for index in range(21):
                            for axis in ("x", "y", "z"):
                                column = f"{side}_hand_{index}_{axis}"
                                if column in row and not row[column]:
                                    row[column] = row[f"arm_{side}_shoulder_{axis}"]
                            confidence_column = f"{side}_hand_{index}_conf"
                            if confidence_column in row:
                                row[confidence_column] = "1.0"
                writer.writerow(row)

    return run_validation(
        work,
        calibration=calibration,
        camera_processor=aligned_processor,
        output_dir=tmp_path / f"out_{tag}",
        run_clinical=False,
    )


def _manifest(offsets: dict[str, int]) -> dict:
    """A session.json declaring per-camera sync offsets (desync probe)."""
    return {
        "format_version": 1,
        "session_id": "fm",
        "cameras": [
            {"name": n, "file": f"{n}.avi", "sync_offset": offsets.get(n, 0)}
            for n in ("cam1", "cam2", "cam3")
        ],
    }


# ---------------------------------------------------------------------------
# 1. Camera dropout — one camera missing a block of frames
# ---------------------------------------------------------------------------


def test_camera_dropout_collapses_redundancy(rendered_session, tmp_path: pathlib.Path):
    """cam3 loses all but two frames -> the harness flags the lost view.

    A single 3->2 camera frame loss still fuses the full skeleton (two
    views suffice), so the honest signal is the redundancy collapse:
    ``n_views`` falls to the 2-view floor (no spare view to reject an
    outlier) and cam3's frame count is visibly short.  Pushing a keypoint
    truly *unfused* needs a second view to drop too — that is the
    occlusion test.
    """
    _session_dir, solved = rendered_session
    calib = _save(solved, tmp_path / "calib.json")
    report = _validate(
        rendered_session,
        tmp_path,
        calibration=calib,
        processor=skeleton_processor(solved, per_camera={"cam3": {"frames": (0, 1)}}),
        tag="dropout",
    )

    by_name = {c.name: c for c in report.tracking_2d.cameras}
    assert by_name["cam3"].observed_frames == 2  # dropped frames 2-7
    assert by_name["cam3"].expected_frames == 8
    assert by_name["cam3"].dropped_frames == 6
    assert by_name["cam3"].detection_rate == pytest.approx(0.25)
    assert by_name["cam1"].observed_frames == 8
    assert by_name["cam2"].observed_frames == 8

    fus = report.fusion_3d
    assert fus.n_views_median == pytest.approx(2.0)  # the third view is gone
    assert fus.n_views_min == 2
    # The surviving pair still fuses every keypoint (no false unfused).
    assert fus.n_active_keypoints == 54
    assert fus.unfused_keypoint_fraction == pytest.approx(0.0)
    assert fus.view_rejection_fraction == pytest.approx(0.0)

    v = report.verdict()
    assert v.grade == "FAIL"  # severe source-frame detection loss is a hard failure
    assert _grades(report)["fusion.n_views_median"] == "WARN"
    assert _grades(report)["tracking.worst_detection_rate"] == "FAIL"


# ---------------------------------------------------------------------------
# 2. Miscalibration — perturbed extrinsics on one camera
# ---------------------------------------------------------------------------


def test_miscalibration_inflates_fusion_reprojection(rendered_session, tmp_path: pathlib.Path):
    """A 5 deg yaw error is rejected and surfaces as lost redundancy.

    The skeleton is projected through the *true* solve but fused through
    the perturbed calibration, so that camera's rays no longer intersect the
    clean pair.  Minimal-set consensus rejects the inconsistent view: trusted
    reprojection remains low, while ``n_views`` honestly loses its spare view.
    """
    _session_dir, solved = rendered_session
    bad = _save(_perturb_yaw(solved, "cam2", 5.0), tmp_path / "miscal.json")
    report = _validate(
        rendered_session,
        tmp_path,
        calibration=bad,
        processor=skeleton_processor(solved),  # project through the TRUE rig
        tag="miscal",
    )

    fus = report.fusion_3d
    assert fus.reproj_err_px_median < 1.0
    assert fus.n_views_median == pytest.approx(2.0)
    assert fus.trusted_keypoint_fraction == pytest.approx(1.0)
    assert fus.view_rejection_fraction == pytest.approx(1 / 3)

    grades = _grades(report)
    assert grades["fusion.reproj_err_px_median"] == "PASS"
    assert grades["fusion.n_views_median"] == "WARN"
    assert grades["fusion.view_rejection_fraction"] == "FAIL"
    assert report.verdict().grade == "FAIL"
    # The solve itself was clean — the fault is geometric inconsistency at
    # fusion, not a bad calibration RMS.
    assert report.calibration.reprojection_error_px < 2.0


# ---------------------------------------------------------------------------
# 3. Desync — a wrong sync_offset on one camera
# ---------------------------------------------------------------------------


def test_desync_degrades_reprojection(rendered_session, tmp_path: pathlib.Path):
    """A 2-frame sync error on a fast subject collapses view redundancy.

    A control run (correct offsets, same fast motion) reconstructs near
    perfectly; declaring a wrong ``sync_offset`` on cam2 makes it
    contribute the wrong frame's pose.  Consensus rejects that view, keeping
    trusted reprojection low while surfacing the fault through ``n_views``.
    """
    _session_dir, solved = rendered_session
    calib = _save(solved, tmp_path / "calib.json")
    fast = skeleton_processor(solved, velocity=DEFAULT_VELOCITY * 5.0)

    control = _validate(
        rendered_session, tmp_path, calibration=calib, processor=fast, tag="sync_ok"
    )
    assert control.fusion_3d.reproj_err_px_median < 8.0  # aligned -> clean
    assert control.verdict().grade == "PASS"

    desynced = _validate(
        rendered_session,
        tmp_path,
        calibration=calib,
        processor=fast,
        session_json=_manifest({"cam2": 2}),
        tag="sync_bad",
    )
    fus = desynced.fusion_3d
    assert fus.reproj_err_px_median < 1.0
    assert control.fusion_3d.n_views_median == pytest.approx(3.0)
    assert fus.n_views_median == pytest.approx(2.0)
    assert _grades(desynced)["fusion.n_views_median"] == "WARN"
    assert fus.view_rejection_fraction > 0.20
    assert _grades(desynced)["fusion.view_rejection_fraction"] == "FAIL"
    assert desynced.verdict().grade == "FAIL"


# ---------------------------------------------------------------------------
# 4. Low confidence — detector scores below the floor
# ---------------------------------------------------------------------------


def test_pervasive_low_confidence_is_flagged(rendered_session, tmp_path: pathlib.Path):
    """Every keypoint below the confidence floor -> the harness flags it.

    Confidence below the floor (0.3) but above the fusion validity gate
    (0.0) is *flagged*, not silently dropped: fusion still reconstructs
    every keypoint cleanly, so the failure surfaces in the tracking metric
    rather than as fabricated or missing 3D.
    """
    _session_dir, solved = rendered_session
    calib = _save(solved, tmp_path / "calib.json")
    report = _validate(
        rendered_session,
        tmp_path,
        calibration=calib,
        processor=skeleton_processor(solved, confidence=0.1),
        tag="lowconf",
        complete_hands=False,
    )

    worst = max(c.low_confidence_fraction for c in report.tracking_2d.cameras)
    assert worst == pytest.approx(1.0)
    assert _grades(report)["tracking.worst_low_confidence_fraction"] == "FAIL"
    assert report.verdict().grade == "FAIL"
    # Low-but-positive confidence still fuses (flagged, not dropped).
    assert report.fusion_3d.n_active_keypoints == 12
    assert report.fusion_3d.reproj_err_px_median < 1.0


def test_subthreshold_confidence_gates_to_nan(rendered_session, tmp_path: pathlib.Path):
    """Zero-confidence views hit the validity gate and route to NaN.

    Forcing a region's confidence to 0 in two of three cameras leaves it
    with one valid view — below ``min_views`` — so those keypoints are
    gated out (never fused, hence dropped from the active set) rather than
    triangulated from a single ray into garbage.
    """
    _session_dir, solved = rendered_session
    calib = _save(solved, tmp_path / "calib.json")
    report = _validate(
        rendered_session,
        tmp_path,
        calibration=calib,
        processor=skeleton_processor(
            solved,
            per_camera={"cam2": {"zero_conf": _REGION}, "cam3": {"zero_conf": _REGION}},
        ),
        tag="zeroconf",
        complete_hands=False,
    )

    fus = report.fusion_3d
    # The 4 zero-confidence keypoints never reach two valid views -> excluded.
    assert fus.n_active_keypoints == 12 - len(_REGION)
    # What does fuse, fuses cleanly (the gate produced NaN, not garbage).
    assert fus.unfused_keypoint_fraction == pytest.approx(0.0)
    assert report.agreement.mean_bone_length_cv < 0.05


# ---------------------------------------------------------------------------
# 5. Occlusion — a body region missing in one or more cameras
# ---------------------------------------------------------------------------


def test_single_camera_occlusion_uses_remaining_views(rendered_session, tmp_path: pathlib.Path):
    """A region hidden from one camera reconstructs from the other two.

    With three cameras, occluding a region in a single view still leaves
    two valid views, so fusion recovers it — the harness does not flag a
    fault, and nothing is lost.
    """
    _session_dir, solved = rendered_session
    calib = _save(solved, tmp_path / "calib.json")
    report = _validate(
        rendered_session,
        tmp_path,
        calibration=calib,
        processor=skeleton_processor(solved, per_camera={"cam3": {"occlude": _REGION}}),
        tag="occ1",
    )

    fus = report.fusion_3d
    assert fus.n_active_keypoints == 54  # region + synthetic hands recovered
    assert fus.unfused_keypoint_fraction == pytest.approx(0.0)
    assert fus.view_rejection_fraction == pytest.approx(0.0)
    assert report.agreement.mean_bone_length_cv < 0.05  # recovered cleanly
    assert report.verdict().grade in {"PASS", "WARN"}


def test_two_camera_occlusion_reports_unfused_not_garbage(rendered_session, tmp_path: pathlib.Path):
    """A region hidden from two cameras part-way is reported NaN, not faked.

    cam3 never sees the distal block and cam2 loses it from frame 2 on, so
    those keypoints fall to a single view and cannot triangulate.  The
    harness marks them unfused (NaN) — the unfused fraction crosses its
    FAIL band — while the keypoints it *does* fuse stay geometrically
    consistent (low bone-length CoV), i.e. never garbage.
    """
    _session_dir, solved = rendered_session
    calib = _save(solved, tmp_path / "calib.json")
    report = _validate(
        rendered_session,
        tmp_path,
        calibration=calib,
        processor=skeleton_processor(
            solved,
            per_camera={
                "cam3": {"occlude": _WIDE},
                "cam2": {"occlude": _WIDE, "occlude_frames": (2, 3, 4, 5, 6, 7)},
            },
        ),
        tag="occ2",
        complete_hands=False,
    )

    fus = report.fusion_3d
    assert fus.unfused_keypoint_fraction > 0.25  # empirically ~0.5
    assert _grades(report)["fusion.unfused_keypoint_fraction"] == "FAIL"
    assert report.verdict().grade == "FAIL"
    # The surviving fusion is correct, not fabricated.
    assert report.agreement.mean_bone_length_cv < 0.05


# ---------------------------------------------------------------------------
# 6. Degenerate calibration — tiny baseline / near-collinear cameras
# ---------------------------------------------------------------------------


def test_degenerate_geometry_flags_instability(rendered_session, tmp_path: pathlib.Path):
    """A near-collinear, tiny-baseline rig is caught via 3D instability.

    With a 1 cm baseline the rays are nearly parallel, so 1 px of detector
    noise explodes into centimetres of depth error.  Reprojection stays
    *low* (the noisy 3D point still reprojects near its noisy 2D source),
    so the degeneracy is invisible to reprojection alone — it surfaces in
    the self-consistency surrogates: rigid bones vary wildly across frames
    and temporal jitter spikes.  This is exactly the "2D looks fine, 3D is
    garbage" failure the gap register warns about.
    """
    _session_dir, solved = rendered_session
    deg = _degenerate_rig(solved, baseline_m=0.01)
    deg_json = _save(deg, tmp_path / "degenerate.json")
    report = _validate(
        rendered_session,
        tmp_path,
        calibration=deg_json,
        processor=skeleton_processor(
            deg,
            noise_px=1.0,
            per_camera={
                "cam1": {"noise_seed": 1},
                "cam2": {"noise_seed": 2},
                "cam3": {"noise_seed": 3},
            },
        ),
        tag="degenerate",
    )

    fus, agr = report.fusion_3d, report.agreement
    assert fus.n_active_keypoints == 54  # fused, just unstable (no crash)

    grades = _grades(report)
    # Instability surrogates catch it...
    assert agr.mean_bone_length_cv > 0.10  # empirically ~0.31
    assert agr.temporal_jitter_mm > 15.0  # empirically ~155 mm
    assert grades["agreement.mean_bone_length_cv"] == "FAIL"
    assert grades["agreement.temporal_jitter_mm"] == "FAIL"
    # ...while reprojection alone would have passed it through.
    assert fus.reproj_err_px_p95 < 15.0
    assert grades["fusion.reproj_err_px_p95"] == "PASS"

    assert report.verdict().grade == "FAIL"
