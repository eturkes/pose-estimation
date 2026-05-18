"""Multi-camera session abstraction.

A *session* is a directory containing N per-camera video files plus an
optional ``session.json`` manifest and optional ``calibration.json``.
See ``.claude/tech/multicam.md`` for the directory layout and design
rationale.

This module owns three concerns:

1. **Discovery** — ``discover_session`` / ``discover_sessions`` parse a
   directory into a ``Session`` (with optional calibration attached).
2. **Iteration** — ``iter_synchronized_frames`` yields ``SessionFrame``
   dicts holding the per-camera BGR frames at each logical frame index,
   honouring per-camera ``sync_offset`` skip counts.
3. **Processing** — ``process_session`` is a ``NotImplementedError``
   stub.  Per-view pose estimation + ``triangulation.fuse_session_frame``
   wiring is a follow-up.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Iterator
from typing import Any, cast

import cv2
import numpy as np

from ._types import SessionCalibration, SessionFrame
from .calibration import (
    CalibrationError,
    load_calibration,
    load_session_calibration,
)

SESSION_MANIFEST_FILENAME = "session.json"
"""Conventional name for the optional per-session manifest."""

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
"""Recognised video extensions for camera discovery."""

CAMERA_GLOB = "cam*"
"""Glob prefix used when ``session.json`` is absent."""

SESSION_FORMAT_VERSION = 1
"""Current ``format_version`` recognised by ``discover_session``."""


class SessionError(ValueError):
    """Raised when a session directory cannot be parsed into a valid Session."""


@dataclasses.dataclass
class SessionCamera:
    """One camera within a multi-camera session."""

    name: str
    file: pathlib.Path
    sync_offset: int = 0

    def __post_init__(self) -> None:
        if self.sync_offset < 0:
            raise SessionError(
                f"camera {self.name!r}: sync_offset must be non-negative (got {self.sync_offset})"
            )


@dataclasses.dataclass
class Session:
    """A multi-camera recording session.

    ``cameras`` order is preserved from manifest order, or alphabetical
    when discovery falls back to glob.  ``calibration`` is ``None`` when
    no calibration is supplied (valid for per-view 2D workflows; required
    for triangulation).
    """

    session_id: str
    directory: pathlib.Path
    cameras: list[SessionCamera]
    calibration: SessionCalibration | None = None

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)

    def camera_names(self) -> list[str]:
        return [c.name for c in self.cameras]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_session(
    directory: str | pathlib.Path,
    *,
    calibration_path: str | pathlib.Path | None = None,
) -> Session:
    """Parse a single session directory into a ``Session``.

    Manifest path: prefers ``session.json`` if present.  Otherwise
    globs ``cam*.{ext}`` for video files and uses the file stem as the
    camera name.

    Calibration resolution order:
        1. ``calibration_path`` argument (if given) — must exist.
        2. Path declared in ``session.json:calibration`` (resolved
           relative to the session directory).
        3. ``<directory>/calibration.json`` if present.
        4. ``None``.

    Raises ``SessionError`` on structural problems (no cameras,
    missing files, duplicate names) and ``CalibrationError`` on
    calibration schema problems.
    """
    d = pathlib.Path(directory).resolve()
    if not d.is_dir():
        raise SessionError(f"not a directory: {d}")

    manifest_path = d / SESSION_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = _load_manifest(manifest_path)
        session_id = manifest.get("session_id") or d.name
        cameras = _cameras_from_manifest(manifest, directory=d)
        manifest_calibration_ref = manifest.get("calibration")
    else:
        session_id = d.name
        cameras = _cameras_from_glob(d)
        manifest_calibration_ref = None

    if not cameras:
        raise SessionError(
            f"{d}: no camera videos found (looked for {SESSION_MANIFEST_FILENAME} or "
            f"{CAMERA_GLOB}{{{','.join(VIDEO_EXTENSIONS)}}})"
        )
    _assert_unique_names(cameras, source=str(d))

    calibration = _resolve_calibration(
        directory=d,
        explicit_path=calibration_path,
        manifest_ref=manifest_calibration_ref,
    )
    return Session(
        session_id=session_id,
        directory=d,
        cameras=cameras,
        calibration=calibration,
    )


def discover_sessions(parent_dir: str | pathlib.Path) -> list[Session]:
    """Discover every session under ``parent_dir`` (one level deep).

    Skips subdirectories that contain no recognisable cameras (no
    manifest, no ``cam*.{ext}`` files).  Raises on any subdirectory
    that *does* look like a session but has structural problems —
    catching typos early beats silently dropping data.
    """
    p = pathlib.Path(parent_dir).resolve()
    if not p.is_dir():
        raise SessionError(f"not a directory: {p}")
    sessions: list[Session] = []
    for child in sorted(p.iterdir()):
        if not child.is_dir():
            continue
        if not _looks_like_session(child):
            continue
        sessions.append(discover_session(child))
    return sessions


def _looks_like_session(directory: pathlib.Path) -> bool:
    """Heuristic: directory contains a manifest or any cam*.{ext} file."""
    if (directory / SESSION_MANIFEST_FILENAME).is_file():
        return True
    return any(_iter_glob_videos(directory))


def _iter_glob_videos(directory: pathlib.Path) -> Iterator[pathlib.Path]:
    for ext in VIDEO_EXTENSIONS:
        yield from directory.glob(f"{CAMERA_GLOB}{ext}")


def _load_manifest(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SessionError(f"{path}: invalid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise SessionError(f"{path}: top-level value must be an object")
    version = data.get("format_version", SESSION_FORMAT_VERSION)
    if version != SESSION_FORMAT_VERSION:
        raise SessionError(
            f"{path}: unsupported format_version={version!r} "
            f"(this build accepts {SESSION_FORMAT_VERSION})"
        )
    if "cameras" not in data:
        raise SessionError(f"{path}: missing required field 'cameras'")
    return data


def _cameras_from_manifest(
    manifest: dict[str, Any], *, directory: pathlib.Path
) -> list[SessionCamera]:
    raw_cameras = manifest["cameras"]
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise SessionError(
            f"{directory / SESSION_MANIFEST_FILENAME}: cameras must be a non-empty array"
        )
    cameras: list[SessionCamera] = []
    for i, raw_entry in enumerate(raw_cameras):
        if not isinstance(raw_entry, dict):
            raise SessionError(f"{directory}: cameras[{i}] must be an object")
        entry = cast(dict[str, Any], raw_entry)
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise SessionError(f"{directory}: cameras[{i}].name must be a non-empty string")
        file_ref = entry.get("file")
        if file_ref is None:
            file_path = _find_glob_for_name(directory, name)
            if file_path is None:
                raise SessionError(
                    f"{directory}: cameras[{i}] ({name!r}) declares no file and no "
                    f"matching {name}.{{{','.join(e.lstrip('.') for e in VIDEO_EXTENSIONS)}}} exists"
                )
        else:
            file_path = (directory / str(file_ref)).resolve()
        if not file_path.is_file():
            raise SessionError(f"{directory}: cameras[{i}] ({name!r}) file not found: {file_path}")
        sync_offset = int(entry.get("sync_offset", 0))
        cameras.append(SessionCamera(name=name, file=file_path, sync_offset=sync_offset))
    return cameras


def _cameras_from_glob(directory: pathlib.Path) -> list[SessionCamera]:
    files = sorted(_iter_glob_videos(directory), key=lambda p: p.name)
    return [SessionCamera(name=p.stem, file=p.resolve(), sync_offset=0) for p in files]


def _find_glob_for_name(directory: pathlib.Path, name: str) -> pathlib.Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = directory / f"{name}{ext}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _assert_unique_names(cameras: list[SessionCamera], *, source: str) -> None:
    seen: set[str] = set()
    for cam in cameras:
        if cam.name in seen:
            raise SessionError(f"{source}: duplicate camera name {cam.name!r}")
        seen.add(cam.name)


def _resolve_calibration(
    *,
    directory: pathlib.Path,
    explicit_path: str | pathlib.Path | None,
    manifest_ref: str | None,
) -> SessionCalibration | None:
    if explicit_path is not None:
        p = pathlib.Path(explicit_path)
        if not p.is_file():
            raise CalibrationError(f"calibration file not found: {p}")
        return load_calibration(p)
    if manifest_ref is not None:
        p = (directory / str(manifest_ref)).resolve()
        if not p.is_file():
            raise CalibrationError(
                f"{directory / SESSION_MANIFEST_FILENAME}: calibration "
                f"{manifest_ref!r} not found at {p}"
            )
        return load_calibration(p)
    return load_session_calibration(directory)


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


def iter_synchronized_frames(session: Session) -> Iterator[SessionFrame]:
    """Yield one ``SessionFrame`` per logical (post-offset) frame index.

    Opens one ``cv2.VideoCapture`` per camera, discards each camera's
    leading ``sync_offset`` frames, then reads frame-by-frame from all
    cameras in lockstep.  Iteration ends as soon as any camera returns
    no frame (e.g. exhausted or read failure).

    Raises ``SessionError`` if any camera cannot be opened or its
    pre-roll cannot be discarded.  Releases all captures on exit.
    """
    captures = _open_session_captures(session)
    try:
        for cam, cap in zip(session.cameras, captures, strict=True):
            for _ in range(cam.sync_offset):
                ok, _frame = cap.read()
                if not ok:
                    raise SessionError(
                        f"camera {cam.name!r}: sync_offset={cam.sync_offset} exceeds "
                        f"available frames in {cam.file}"
                    )
        frame_index = 0
        while True:
            batch: dict[str, np.ndarray] = {}
            exhausted = False
            for cam, cap in zip(session.cameras, captures, strict=True):
                ok, frame = cap.read()
                if not ok:
                    exhausted = True
                    break
                batch[cam.name] = frame
            if exhausted:
                return
            yield SessionFrame(frame_index=frame_index, frames=batch)
            frame_index += 1
    finally:
        for cap in captures:
            cap.release()


def _open_session_captures(session: Session) -> list[cv2.VideoCapture]:
    captures: list[cv2.VideoCapture] = []
    try:
        for cam in session.cameras:
            cap = cv2.VideoCapture(str(cam.file))
            if not cap.isOpened():
                raise SessionError(f"camera {cam.name!r}: cannot open {cam.file}")
            captures.append(cap)
    except BaseException:
        for cap in captures:
            cap.release()
        raise
    return captures


# ---------------------------------------------------------------------------
# Processing (stub)
# ---------------------------------------------------------------------------


def process_session(session: Session, **kwargs: Any) -> Any:
    """Run the full multi-camera pipeline (per-view + triangulation).

    Not yet wired.  Follow-up tracks per-view pipeline reuse and
    ``triangulation.fuse_session_frame`` integration.  See
    ``.claude/tech/multicam.md`` for the planned data flow.
    """
    raise NotImplementedError(
        "process_session is not yet wired. The scaffolding (Session, "
        "iter_synchronized_frames, calibration IO, triangulation stubs) "
        "is in place; per-view processing and 3D fusion are tracked as a "
        f"follow-up. Called with session_id={session.session_id!r}, "
        f"n_cameras={session.n_cameras}, "
        f"calibration={'present' if session.calibration is not None else 'absent'}, "
        f"extra kwargs={sorted(kwargs)}."
    )


__all__ = [
    "CAMERA_GLOB",
    "SESSION_FORMAT_VERSION",
    "SESSION_MANIFEST_FILENAME",
    "VIDEO_EXTENSIONS",
    "Session",
    "SessionCamera",
    "SessionError",
    "discover_session",
    "discover_sessions",
    "iter_synchronized_frames",
    "process_session",
]
