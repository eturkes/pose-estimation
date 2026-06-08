"""Tests for ``pose_estimation.multicam`` discovery + iteration."""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
import pytest

from pose_estimation._types import CameraCalibration, SessionCalibration
from pose_estimation.calibration import (
    CALIBRATION_FILENAME,
    CalibrationError,
    save_calibration,
)
from pose_estimation.export import frame_to_rows, open_csv_writer, read_csv_keypoints
from pose_estimation.multicam import (
    CAMERA_GLOB,
    SESSION_MANIFEST_FILENAME,
    Session,
    SessionCamera,
    SessionError,
    discover_session,
    discover_sessions,
    fuse_session_outputs,
    iter_synchronized_frames,
    process_session,
)
from pose_estimation.processing import TRACKING_HANDS_ARMS
from pose_estimation.triangulation import project_points

# ---------------------------------------------------------------------------
# Video / calibration helpers
# ---------------------------------------------------------------------------


_TEST_FRAME_COUNT = 12
_TEST_FRAME_SIZE = (64, 48)  # (width, height) — small to keep the test cheap
_TEST_FPS = 10.0


def _write_synthetic_video(
    path: pathlib.Path,
    *,
    n_frames: int = _TEST_FRAME_COUNT,
    intensity: int = 128,
) -> bool:
    """Write a tiny MJPG/AVI video. Returns True on success.

    Each frame is a solid colour (so reads succeed) at low resolution.
    Skips when the codec is unavailable on the test host.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, _TEST_FPS, _TEST_FRAME_SIZE)
    if not writer.isOpened():
        return False
    try:
        for i in range(n_frames):
            frame = np.full(
                (_TEST_FRAME_SIZE[1], _TEST_FRAME_SIZE[0], 3),
                fill_value=(intensity + i) % 256,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def _ensure_video_codec_available(tmp_path: pathlib.Path) -> None:
    probe = tmp_path / "_probe.avi"
    if not _write_synthetic_video(probe, n_frames=2):
        pytest.skip("MJPG/AVI codec unavailable on this host")


def _write_calibration(path: pathlib.Path, *, cameras: list[str]) -> None:
    data = {
        "format_version": 1,
        "session_id": path.parent.name,
        "world_frame": cameras[0],
        "cameras": [
            {
                "name": name,
                "resolution": [_TEST_FRAME_SIZE[0], _TEST_FRAME_SIZE[1]],
                "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "rvec": [0.0, 0.0, 0.0] if i == 0 else [0.0, 0.3, 0.0],
                "tvec": [0.0, 0.0, 0.0] if i == 0 else [0.2, 0.0, 0.0],
            }
            for i, name in enumerate(cameras)
        ],
        "reprojection_error_px": 0.25,
        "solver": "test",
        "solved_at": "2026-05-18T12:00:00Z",
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# discover_session — glob fallback
# ---------------------------------------------------------------------------


def test_discover_session_glob_fallback(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_a"
    for name in ("cam1", "cam2", "cam3"):
        assert _write_synthetic_video(session_dir / f"{name}.avi"), "video write failed"

    session = discover_session(session_dir)
    assert session.session_id == "session_a"
    assert session.n_cameras == 3
    assert session.camera_names() == ["cam1", "cam2", "cam3"]
    assert all(c.sync_offset == 0 for c in session.cameras)
    assert session.calibration is None


def test_discover_session_empty_directory_raises(tmp_path: pathlib.Path):
    empty = tmp_path / "empty_session"
    empty.mkdir()
    with pytest.raises(SessionError, match=CAMERA_GLOB):
        discover_session(empty)


def test_discover_session_not_a_directory_raises(tmp_path: pathlib.Path):
    p = tmp_path / "not_a_dir.txt"
    p.write_text("x")
    with pytest.raises(SessionError, match="not a directory"):
        discover_session(p)


# ---------------------------------------------------------------------------
# discover_session — manifest
# ---------------------------------------------------------------------------


def test_discover_session_with_manifest(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_b"
    session_dir.mkdir()
    for name in ("alpha", "bravo"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    manifest = {
        "format_version": 1,
        "session_id": "custom_id",
        "cameras": [
            {"name": "alpha", "file": "alpha.avi", "sync_offset": 2},
            {"name": "bravo", "file": "bravo.avi", "sync_offset": 0},
        ],
    }
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))

    session = discover_session(session_dir)
    assert session.session_id == "custom_id"
    assert session.camera_names() == ["alpha", "bravo"]
    assert session.cameras[0].sync_offset == 2
    assert session.calibration is None


def test_discover_session_manifest_file_missing_raises(tmp_path: pathlib.Path):
    session_dir = tmp_path / "session_c"
    session_dir.mkdir()
    manifest = {
        "format_version": 1,
        "session_id": "c",
        "cameras": [{"name": "x", "file": "x.avi"}],
    }
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(SessionError, match="file not found"):
        discover_session(session_dir)


def test_discover_session_manifest_path_traversal_camera(tmp_path: pathlib.Path):
    session_dir = tmp_path / "session_pt"
    session_dir.mkdir()
    manifest = {
        "format_version": 1,
        "session_id": "pt",
        "cameras": [{"name": "evil", "file": "../../etc/passwd"}],
    }
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(SessionError, match="path traversal"):
        discover_session(session_dir)


def test_discover_session_manifest_path_traversal_calibration(tmp_path: pathlib.Path):
    session_dir = tmp_path / "session_pt2"
    _write_synthetic_video(session_dir / "cam1.mp4")
    manifest = {
        "format_version": 1,
        "session_id": "pt2",
        "cameras": [{"name": "cam1", "file": "cam1.mp4"}],
        "calibration": "../../etc/passwd",
    }
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(SessionError, match="path traversal"):
        discover_session(session_dir)


def test_discover_session_manifest_path_traversal_camera_name(
    tmp_path: pathlib.Path,
):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_pt_name"
    _write_synthetic_video(session_dir / "cam1.avi")
    manifest = {
        "format_version": 1,
        "session_id": "pt_name",
        "cameras": [{"name": "../../etc/passwd"}],
    }
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(SessionError, match="path separator"):
        discover_session(session_dir)


def test_session_camera_rejects_negative_sync_offset():
    with pytest.raises(SessionError, match="non-negative"):
        SessionCamera(name="x", file=pathlib.Path("/dev/null"), sync_offset=-1)


def test_discover_session_manifest_unknown_format_version(tmp_path: pathlib.Path):
    session_dir = tmp_path / "session_v"
    session_dir.mkdir()
    manifest = {"format_version": 99, "cameras": [{"name": "x", "file": "x.avi"}]}
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(SessionError, match="format_version"):
        discover_session(session_dir)


# ---------------------------------------------------------------------------
# Calibration resolution
# ---------------------------------------------------------------------------


def test_discover_session_auto_loads_calibration(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_cal"
    for name in ("cam1", "cam2"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    _write_calibration(session_dir / CALIBRATION_FILENAME, cameras=["cam1", "cam2"])

    session = discover_session(session_dir)
    assert session.calibration is not None
    assert session.calibration["world_frame"] == "cam1"
    assert set(session.calibration["cameras"].keys()) == {"cam1", "cam2"}


def test_discover_session_explicit_calibration_overrides_auto(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_cal_o"
    for name in ("cam1", "cam2"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    # Auto calibration in the session dir
    _write_calibration(session_dir / CALIBRATION_FILENAME, cameras=["cam1", "cam2"])
    # Override path (different session_id so we can distinguish)
    override = tmp_path / "alt_calibration.json"
    _write_calibration(override, cameras=["cam1", "cam2"])
    raw = json.loads(override.read_text())
    raw["session_id"] = "OVERRIDE"
    override.write_text(json.dumps(raw))

    session = discover_session(session_dir, calibration_path=override)
    assert session.calibration is not None
    assert session.calibration["session_id"] == "OVERRIDE"


def test_discover_session_explicit_calibration_missing_raises(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_cal_missing"
    for name in ("cam1", "cam2"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    with pytest.raises(CalibrationError, match="not found"):
        discover_session(session_dir, calibration_path=tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# discover_sessions
# ---------------------------------------------------------------------------


def test_discover_sessions_iterates_children(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    parent = tmp_path / "sessions"
    parent.mkdir()
    for sid in ("s1", "s2"):
        for cam in ("cam1", "cam2"):
            assert _write_synthetic_video(parent / sid / f"{cam}.avi")
    # A non-session subdir (no cameras, no manifest) — should be skipped silently.
    (parent / "_notes").mkdir()
    (parent / "_notes" / "README.txt").write_text("x")
    # A loose file — should be skipped silently.
    (parent / "top_level.txt").write_text("x")

    sessions = discover_sessions(parent)
    assert sorted(s.session_id for s in sessions) == ["s1", "s2"]


def test_discover_sessions_not_a_directory_raises(tmp_path: pathlib.Path):
    p = tmp_path / "missing"
    with pytest.raises(SessionError, match="not a directory"):
        discover_sessions(p)


# ---------------------------------------------------------------------------
# iter_synchronized_frames
# ---------------------------------------------------------------------------


def test_iter_synchronized_frames_yields_aligned_batches(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_iter"
    cams = ("cam1", "cam2", "cam3")
    for name in cams:
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    session = discover_session(session_dir)

    frames = list(iter_synchronized_frames(session))
    assert 0 < len(frames) <= _TEST_FRAME_COUNT
    for i, sf in enumerate(frames):
        assert sf["frame_index"] == i
        assert set(sf["frames"].keys()) == set(cams)
        for cam_name in cams:
            arr = sf["frames"][cam_name]
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (_TEST_FRAME_SIZE[1], _TEST_FRAME_SIZE[0], 3)
            assert arr.dtype == np.uint8


def test_iter_sync_offset_skips_leading_frames(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_offset"
    session_dir.mkdir()
    for name in ("cam1", "cam2"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    manifest = {
        "format_version": 1,
        "session_id": "offset",
        "cameras": [
            {"name": "cam1", "file": "cam1.avi", "sync_offset": 0},
            {"name": "cam2", "file": "cam2.avi", "sync_offset": 5},
        ],
    }
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    session = discover_session(session_dir)

    frames = list(iter_synchronized_frames(session))
    # cam2 starts 5 frames in, so at most (_TEST_FRAME_COUNT - 5) aligned frames.
    assert len(frames) <= _TEST_FRAME_COUNT - 5


def test_iter_sync_offset_exceeding_frames_raises(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_overflow"
    session_dir.mkdir()
    for name in ("cam1", "cam2"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    manifest = {
        "format_version": 1,
        "session_id": "overflow",
        "cameras": [
            {"name": "cam1", "file": "cam1.avi"},
            {"name": "cam2", "file": "cam2.avi", "sync_offset": _TEST_FRAME_COUNT + 5},
        ],
    }
    (session_dir / SESSION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    session = discover_session(session_dir)

    with pytest.raises(SessionError, match="exceeds available frames"):
        list(iter_synchronized_frames(session))


# ---------------------------------------------------------------------------
# process_session — per-camera orchestration
# ---------------------------------------------------------------------------


def test_process_session_calls_processor_per_camera(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_proc"
    cam_names = ("cam1", "cam2", "cam3")
    for name in cam_names:
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    session = discover_session(session_dir)

    calls: list[dict] = []

    def recorder(*, source, output_csv, output_diag, video_name):
        calls.append(
            {
                "source": source,
                "output_csv": output_csv,
                "output_diag": output_diag,
                "video_name": video_name,
            }
        )
        return f"ok-{video_name}"

    out = tmp_path / "out"
    results = process_session(session, camera_processor=recorder, output_dir=out)

    assert len(calls) == 3
    assert set(results.keys()) == set(cam_names)
    for name in cam_names:
        assert results[name] == f"ok-{session.session_id}/{name}"

    for call, name in zip(calls, cam_names, strict=True):
        assert call["source"].endswith(f"{name}.avi")
        assert call["output_csv"] == out / session.session_id / f"{name}.csv"
        assert call["output_diag"] == out / session.session_id / f"{name}_diag.csv"
        assert call["video_name"] == f"{session.session_id}/{name}"


def test_process_session_creates_output_directory(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_mkdir"
    assert _write_synthetic_video(session_dir / "cam1.avi")
    session = discover_session(session_dir)

    out = tmp_path / "deeply" / "nested" / "output"
    assert not out.exists()

    process_session(session, camera_processor=lambda **_kw: None, output_dir=out)
    assert (out / session.session_id).is_dir()


def test_process_session_default_output_dir(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    parent = tmp_path / "videos"
    session_dir = parent / "s1"
    assert _write_synthetic_video(session_dir / "cam1.avi")
    session = discover_session(session_dir)

    calls: list[dict] = []
    process_session(
        session,
        camera_processor=lambda **kw: calls.append(kw),
    )
    # Default output dir is <session_parent>/output/<session_id>/
    expected_base = parent / "output" / "s1"
    assert calls[0]["output_csv"] == expected_base / "cam1.csv"


def test_process_session_requires_camera_processor(tmp_path: pathlib.Path):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "session_no_proc"
    assert _write_synthetic_video(session_dir / "cam1.avi")
    session = discover_session(session_dir)
    with pytest.raises(TypeError, match="camera_processor"):
        process_session(session)  # ty: ignore[missing-argument]


# ---------------------------------------------------------------------------
# 3D fusion — CSV read-back + fuse_session_outputs + process_session wiring
# ---------------------------------------------------------------------------

# 12 synthetic "arm" keypoints (hands-arms mode) spread through the volume.
_ARM_BASE = np.array(
    [
        [-0.35, -0.25, 2.9],
        [0.35, -0.25, 2.9],
        [-0.45, 0.05, 3.0],
        [0.45, 0.05, 3.0],
        [-0.40, 0.30, 3.1],
        [0.40, 0.30, 3.1],
        [-0.38, 0.36, 3.1],
        [0.38, 0.36, 3.1],
        [-0.36, 0.38, 3.1],
        [0.36, 0.38, 3.1],
        [-0.34, 0.40, 3.1],
        [0.34, 0.40, 3.1],
    ]
)


def _arm_world(frame_idx: int) -> np.ndarray:
    """Ground-truth skeleton at a logical frame (translates over time)."""
    return _ARM_BASE + np.array([0.01, 0.004, 0.008]) * frame_idx


def _arc_calibration(
    session_id: str, names: tuple[str, ...] = ("cam1", "cam2", "cam3")
) -> SessionCalibration:
    """Cameras in a wide-baseline arc (world frame = first camera)."""
    poses = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, -0.3, 0.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.3, 0.0], [1.0, 0.0, 0.0]),
    ]
    cameras = {}
    for name, (rvec, tvec) in zip(names, poses, strict=False):
        cameras[name] = CameraCalibration(
            name=name,
            resolution=(1920, 1080),
            K=np.array(
                [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
            distortion=np.zeros(5, dtype=np.float64),
            rvec=np.asarray(rvec, dtype=np.float64),
            tvec=np.asarray(tvec, dtype=np.float64),
        )
    return SessionCalibration(
        format_version=1,
        session_id=session_id,
        world_frame=names[0],
        cameras=cameras,
        reprojection_error_px=0.0,
        solver="test",
        solved_at="2026-05-18T12:00:00Z",
    )


def _write_camera_csv(
    csv_path: pathlib.Path,
    camera: CameraCalibration,
    n_frames: int,
    *,
    offset: int = 0,
) -> None:
    """Project the ground-truth skeleton into *camera* and write its CSV.

    Rows carry *raw* per-camera frame indices (logical + offset),
    mirroring what per-camera 2D processing produces.
    """
    width, height = camera["resolution"]
    fh, writer = open_csv_writer(csv_path, tracking=TRACKING_HANDS_ARMS)
    try:
        for f in range(n_frames):
            px = project_points(_arm_world(f), camera)
            lm = np.concatenate([px, np.zeros((len(px), 1))], axis=1)
            vis = np.full(len(px), 0.9)
            for row in frame_to_rows(
                "v",
                f + offset,
                f / 30.0,
                height,
                width,
                [lm],
                [vis],
                [],
                [],
                tracking=TRACKING_HANDS_ARMS,
            ):
                writer.writerow(row)
    finally:
        fh.close()


def test_read_csv_keypoints_round_trip(tmp_path: pathlib.Path):
    csv_path = tmp_path / "cam1.csv"
    width, height = 1920, 1080
    lm = np.array([[100.0 + 10 * i, 50.0 + 5 * i, 7.0] for i in range(12)])
    vis = np.linspace(0.1, 1.0, 12)
    fh, writer = open_csv_writer(csv_path, tracking=TRACKING_HANDS_ARMS)
    try:
        for row in frame_to_rows(
            "v", 4, 0.13, height, width, [lm], [vis], [], [], tracking=TRACKING_HANDS_ARMS
        ):
            writer.writerow(row)
        # Frame 5 has two people; only person_idx 0 must be read back.
        for row in frame_to_rows(
            "v",
            5,
            0.17,
            height,
            width,
            [lm, lm + 50.0],
            [vis, vis],
            [],
            [],
            tracking=TRACKING_HANDS_ARMS,
        ):
            writer.writerow(row)
    finally:
        fh.close()

    names, frames = read_csv_keypoints(csv_path)

    assert len(names) == 12 + 42
    assert names[0] == "arm_left_shoulder"
    assert names[12] == "left_hand_0"
    assert set(frames) == {4, 5}
    for frame_idx in (4, 5):
        kps, conf = frames[frame_idx]
        np.testing.assert_allclose(kps[:12, 0], lm[:, 0] / width, atol=1e-6)
        np.testing.assert_allclose(kps[:12, 1], lm[:, 1] / height, atol=1e-6)
        np.testing.assert_allclose(conf[:12], vis, atol=1e-4)
        # Hands were never observed: NaN coordinates, zero confidence.
        assert np.all(np.isnan(kps[12:]))
        assert np.all(conf[12:] == 0.0)


def test_read_csv_keypoints_rejects_foreign_csv(tmp_path: pathlib.Path):
    p = tmp_path / "foreign.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="not a keypoint CSV"):
        read_csv_keypoints(p)


def test_fuse_session_outputs_reconstructs_skeleton(tmp_path: pathlib.Path):
    calib = _arc_calibration("s3d")
    out_base = tmp_path / "out"
    session_out = out_base / "s3d"
    session_out.mkdir(parents=True)
    offsets = {"cam1": 0, "cam2": 2, "cam3": 0}
    for name, off in offsets.items():
        _write_camera_csv(session_out / f"{name}.csv", calib["cameras"][name], 5, offset=off)
    session = Session(
        session_id="s3d",
        directory=tmp_path / "videos" / "s3d",
        cameras=[
            SessionCamera(name=n, file=pathlib.Path(f"{n}.mp4"), sync_offset=off)
            for n, off in offsets.items()
        ],
        calibration=calib,
    )

    fusion = fuse_session_outputs(session, out_base)

    assert fusion.keypoint_names[:2] == ["arm_left_shoulder", "arm_right_shoulder"]
    assert [f for f, _, _ in fusion.frames] == list(range(5))
    for frame_idx, world, diag in fusion.frames:
        np.testing.assert_allclose(world[:12], _arm_world(frame_idx), atol=1e-3)
        assert np.all(diag["n_views"][:12] == 3)
        assert np.all(diag["cheirality_ok"][:12])
        # Hand keypoints were never observed → NaN world, zero views.
        assert np.all(np.isnan(world[12:]))
        assert np.all(diag["n_views"][12:] == 0)


def test_fuse_session_outputs_requires_calibration(tmp_path: pathlib.Path):
    session = Session(
        session_id="s",
        directory=tmp_path,
        cameras=[SessionCamera(name="cam1", file=pathlib.Path("cam1.mp4"))],
    )
    with pytest.raises(SessionError, match="requires calibration"):
        fuse_session_outputs(session, tmp_path / "out")


def test_fuse_session_outputs_needs_min_views_csvs(tmp_path: pathlib.Path):
    calib = _arc_calibration("s1")
    out_base = tmp_path / "out"
    (out_base / "s1").mkdir(parents=True)
    _write_camera_csv(out_base / "s1" / "cam1.csv", calib["cameras"]["cam1"], 2)
    session = Session(
        session_id="s1",
        directory=tmp_path,
        cameras=[
            SessionCamera(name=n, file=pathlib.Path(f"{n}.mp4")) for n in ("cam1", "cam2", "cam3")
        ],
        calibration=calib,
    )
    with pytest.raises(SessionError, match="needs >= 2"):
        fuse_session_outputs(session, out_base)


def test_process_session_fuses_when_calibrated(tmp_path: pathlib.Path, capsys):
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "scal"
    for name in ("cam1", "cam2"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    calib = _arc_calibration("scal", names=("cam1", "cam2"))
    save_calibration(calib, session_dir / CALIBRATION_FILENAME)
    session = discover_session(session_dir)
    assert session.calibration is not None

    def processor(*, source, output_csv, output_diag, video_name):
        cam_name = video_name.split("/")[-1]
        _write_camera_csv(output_csv, calib["cameras"][cam_name], 3)
        return "ok"

    results = process_session(session, camera_processor=processor, output_dir=tmp_path / "out")

    captured = capsys.readouterr().out
    assert "3D fusion: 3 frame(s)" in captured
    assert "WARNING" not in captured
    assert set(results) == {"cam1", "cam2"}


def test_process_session_fusion_failure_warns(tmp_path: pathlib.Path, capsys):
    """A fusion failure (no CSVs written) must not lose per-camera results."""
    _ensure_video_codec_available(tmp_path)
    session_dir = tmp_path / "swarn"
    for name in ("cam1", "cam2"):
        assert _write_synthetic_video(session_dir / f"{name}.avi")
    calib = _arc_calibration("swarn", names=("cam1", "cam2"))
    save_calibration(calib, session_dir / CALIBRATION_FILENAME)
    session = discover_session(session_dir)

    results = process_session(
        session, camera_processor=lambda **_kw: "done", output_dir=tmp_path / "out"
    )

    captured = capsys.readouterr().out
    assert "WARNING: 3D fusion skipped" in captured
    assert results == {"cam1": "done", "cam2": "done"}
