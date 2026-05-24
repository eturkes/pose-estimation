"""Tests for ``pose_estimation.multicam`` discovery + iteration."""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
import pytest

from pose_estimation.calibration import CALIBRATION_FILENAME, CalibrationError
from pose_estimation.multicam import (
    CAMERA_GLOB,
    SESSION_MANIFEST_FILENAME,
    SessionCamera,
    SessionError,
    discover_session,
    discover_sessions,
    iter_synchronized_frames,
    process_session,
)

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
