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
3. **Processing** — ``process_session`` orchestrates per-camera video
   processing via a caller-supplied ``camera_processor`` callback,
   managing output directory layout and progress reporting.  When the
   session has calibration, it then fuses the per-camera CSVs into 3D
   via ``fuse_session_outputs`` (CSV read-back → ``fuse_session_frame``
   per logical frame) and writes ``world3d.csv``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable, Iterator
from typing import Any, cast

import cv2
import numpy as np

from ._types import FusionDiagnostics, SessionCalibration, SessionFrame
from .calibration import (
    CalibrationError,
    load_calibration,
    load_session_calibration,
)
from .export import WORLD3D_FILENAME, read_csv_keypoints, write_world3d_csv
from .triangulation import fuse_session_frame

SESSION_MANIFEST_FILENAME = "session.json"
"""Conventional name for the optional per-session manifest."""

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
"""Recognised video extensions for camera discovery."""

CAMERA_GLOB = "cam*"
"""Glob prefix used when ``session.json`` is absent."""

SESSION_FORMAT_VERSION = 1
"""Current ``format_version`` recognised by ``discover_session``."""


def _safe_resolve(base: pathlib.Path, ref: str) -> pathlib.Path:
    """Resolve *ref* relative to *base*, rejecting path traversal."""
    resolved = (base / ref).resolve()
    base_resolved = base.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise SessionError(f"path traversal detected: {ref!r} escapes {base}")
    return resolved


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
        raw_session_id = manifest.get("session_id")
        session_id = (
            _safe_name_component(str(raw_session_id), kind="session_id", source=str(manifest_path))
            if raw_session_id
            else d.name
        )
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


def resolve_cli_sessions(
    session_dir: str | pathlib.Path | None,
    sessions_dir: str | pathlib.Path | None,
    calibration_path: str | pathlib.Path | None = None,
    *,
    summary_label: str = "Multi-camera dispatch",
    redact_identifiers: bool = False,
) -> list[Session]:
    """Resolve the --session-dir / --sessions-dir CLI args into sessions.

    Shared by both entry points' session dispatch and the read-only
    ``--list-sessions`` discovery probe.  Exactly one of *session_dir* /
    *sessions_dir* must be given; raises ``SessionError`` on conflict or empty
    discovery.  Resolution reads filenames + session.json/calibration.json (no
    frame decoding — no video bytes are read); prints a summary headed by
    *summary_label*.  With *redact_identifiers* (the footage-gate probe), each
    per-session line shows only an ordinal + camera count + calibration presence,
    keeping the deny-listed tree's session ids / camera names out of context.
    """
    if session_dir and sessions_dir:
        raise SessionError("--session-dir and --sessions-dir are mutually exclusive")
    if session_dir:
        sessions = [discover_session(session_dir, calibration_path=calibration_path)]
    else:
        if calibration_path is not None:
            print(
                "WARNING: --calibration applies the same calibration to every "
                "discovered session; pass --session-dir for per-session overrides."
            )
        if sessions_dir is None:
            raise SessionError("one of --session-dir / --sessions-dir is required")
        sessions = discover_sessions(sessions_dir)
        if not sessions:
            raise SessionError(f"no sessions discovered under {sessions_dir}")
        if calibration_path is not None:
            sessions = [
                discover_session(s.directory, calibration_path=calibration_path) for s in sessions
            ]

    print(f"{summary_label}: {len(sessions)} session(s)")
    for i, s in enumerate(sessions, 1):
        cal = "present" if s.calibration is not None else "absent"
        if redact_identifiers:
            # --list-sessions footage gate: keep the deny-listed tree's identifiers
            # (session id, camera names) out of agent context — the gate needs only
            # the shape (camera count + calibration presence), keyed by an ordinal.
            print(f"  session #{i}: {s.n_cameras} cameras; calibration: {cal}")
        else:
            print(
                f"  {s.session_id}: {s.n_cameras} cameras "
                f"({', '.join(s.camera_names())}); calibration: {cal}"
            )
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


def _safe_name_component(value: str, *, kind: str, source: str) -> str:
    """Validate a manifest label that becomes a filesystem path component.

    ``session_id`` and camera ``name`` are used as directory / file names under
    the output tree (``<output>/<session_id>/<name>.csv``) and are echoed by the
    ``--list-sessions`` probe, so a manifest must not smuggle a path separator, a
    ``.``/``..`` traversal component, or a control character through them.
    Returns *value* unchanged when safe; raises ``SessionError`` otherwise.
    """
    if not value or value.isspace():
        raise SessionError(f"{source}: {kind} must be a non-empty string")
    if "/" in value or "\\" in value:
        raise SessionError(f"{source}: {kind} must not contain a path separator: {value!r}")
    if value in (".", ".."):
        raise SessionError(f"{source}: {kind} must not be a '.'/'..' path component: {value!r}")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise SessionError(f"{source}: {kind} must not contain control characters: {value!r}")
    return value


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
        if not isinstance(name, str):
            raise SessionError(f"{directory}: cameras[{i}].name must be a string")
        name = _safe_name_component(name, kind=f"cameras[{i}].name", source=str(directory))
        file_ref = entry.get("file")
        if file_ref is None:
            file_path = _find_glob_for_name(directory, name)
            if file_path is None:
                raise SessionError(
                    f"{directory}: cameras[{i}] ({name!r}) declares no file and no "
                    f"matching {name}.{{{','.join(e.lstrip('.') for e in VIDEO_EXTENSIONS)}}} exists"
                )
        else:
            file_path = _safe_resolve(directory, str(file_ref))
        if not file_path.is_file():
            raise SessionError(f"{directory}: cameras[{i}] ({name!r}) file not found: {file_path}")
        sync_offset = int(entry.get("sync_offset", 0))
        cameras.append(SessionCamera(name=name, file=file_path, sync_offset=sync_offset))
    return cameras


def _cameras_from_glob(directory: pathlib.Path) -> list[SessionCamera]:
    files = sorted(_iter_glob_videos(directory), key=lambda p: p.name)
    return [SessionCamera(name=p.stem, file=p.resolve(), sync_offset=0) for p in files]


def _find_glob_for_name(directory: pathlib.Path, name: str) -> pathlib.Path | None:
    if "/" in name or "\\" in name:
        raise SessionError(f"camera name contains path separator: {name!r}")
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
        p = _safe_resolve(directory, str(manifest_ref))
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
# Processing
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = "output"


def _resolve_session_output(
    session: Session, output_dir: str | pathlib.Path | None
) -> pathlib.Path:
    """Determine the output directory for a session's results.

    Layout: ``<output_dir>/<session_id>/``.  When *output_dir* is
    ``None``, defaults to ``output/`` alongside the session's parent.
    """
    if output_dir is not None:
        base = pathlib.Path(output_dir)
    else:
        base = session.directory.parent / _DEFAULT_OUTPUT_DIR
    return base / session.session_id


def process_session(
    session: Session,
    *,
    camera_processor: Callable[..., Any],
    output_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Run per-camera processing for every camera in *session*.

    ``camera_processor`` is called once per camera as::

        camera_processor(
            source=<str>,           # absolute path to the camera's video
            output_csv=<Path>,      # per-camera CSV path
            output_diag=<Path>,     # per-camera diagnostics CSV path
            video_name=<str>,       # display label: ``session_id/cam_name``
        )

    The callback encapsulates all backend-specific logic (model setup,
    inference, smoothing, CSV writing); ``process_session`` handles
    session-level orchestration: output directory creation, camera
    iteration, and progress reporting.  When the session carries
    calibration, the per-camera CSVs are then fused into 3D via
    ``fuse_session_outputs`` and written to ``world3d.csv`` (non-fatal
    on failure — the 2D results are already on disk).

    Returns a dict mapping camera name → the value returned by
    ``camera_processor`` for that camera.
    """
    session_out = _resolve_session_output(session, output_dir)
    session_out.mkdir(parents=True, exist_ok=True)

    print(
        f"\nSession {session.session_id!r}: processing {session.n_cameras} camera(s) "
        f"→ {session_out}"
    )

    results: dict[str, Any] = {}
    for i, cam in enumerate(session.cameras, 1):
        source = str((session.directory / cam.file).resolve())
        csv_path = session_out / f"{cam.name}.csv"
        diag_path = session_out / f"{cam.name}_diag.csv"
        video_name = f"{session.session_id}/{cam.name}"

        print(f"\n  [{i}/{session.n_cameras}] {cam.name} ({cam.file})")
        results[cam.name] = camera_processor(
            source=source,
            output_csv=csv_path,
            output_diag=diag_path,
            video_name=video_name,
        )
        print(f"    CSV:  {csv_path}")
        print(f"    Diag: {diag_path}")

    if session.calibration is not None:
        _fuse_and_report(session, output_dir)

    print(f"\nSession {session.session_id!r}: complete ({session.n_cameras} cameras)")
    return results


# ---------------------------------------------------------------------------
# 3D fusion (CSV read-back → fuse_session_frame per logical frame)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SessionFusion:
    """Triangulated 3D output for one session.

    ``frames`` holds one entry per fused logical frame index:
    ``(frame_index, timestamp_sec, world, diag)`` with ``world`` of
    shape ``(N, 3)`` in metres (NaN rows where fusion failed) and
    per-keypoint ``FusionDiagnostics``.  ``timestamp_sec`` comes from
    the world-frame camera's CSV when available (NaN otherwise).
    ``keypoint_names`` labels the ``N`` rows (body/arm keypoints
    first, then ``{left,right}_hand_{0..20}``).  This is the exact
    row layout consumed by ``export.write_world3d_csv``.
    """

    keypoint_names: list[str]
    frames: list[tuple[int, float, np.ndarray, FusionDiagnostics]]


def fuse_session_outputs(
    session: Session,
    output_dir: str | pathlib.Path | None = None,
    *,
    min_views: int = 2,
) -> SessionFusion:
    """Triangulate the per-camera CSVs of *session* into 3D keypoints.

    Reads back ``<output_dir>/<session_id>/<cam>.csv`` for every
    camera (skipping cameras whose CSV is absent), converts the
    normalised coordinates to pixels via each camera's calibrated
    resolution, aligns frame indices by subtracting ``sync_offset``
    (CSV rows hold *raw* per-camera indices), and fuses every logical
    frame observed by at least *min_views* cameras.

    Only ``person_idx == 0`` rows are fused — cross-camera person
    identity matching is not implemented.

    Raises ``SessionError`` when the session has no calibration, a
    camera lacks a calibration entry, cameras disagree on tracking
    mode, or fewer than *min_views* CSVs exist.
    """
    if session.calibration is None:
        raise SessionError(f"session {session.session_id!r}: 3D fusion requires calibration")
    if min_views < 2:
        raise ValueError(f"fuse_session_outputs: min_views must be >= 2 (got {min_views})")
    calibration = session.calibration
    session_out = _resolve_session_output(session, output_dir)

    keypoint_names: list[str] | None = None
    per_cam_frames: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]] = {}
    for cam in session.cameras:
        csv_path = session_out / f"{cam.name}.csv"
        if not csv_path.is_file():
            continue
        if cam.name not in calibration["cameras"]:
            raise SessionError(
                f"session {session.session_id!r}: camera {cam.name!r} has no calibration entry"
            )
        names, frames = read_csv_keypoints(csv_path)
        if keypoint_names is None:
            keypoint_names = names
        elif names != keypoint_names:
            raise SessionError(
                f"session {session.session_id!r}: camera {cam.name!r} CSV keypoint set "
                "differs from other cameras (mixed tracking modes?)"
            )
        # Normalised [0, 1] → pixels in the calibrated resolution.
        scale = np.asarray(calibration["cameras"][cam.name]["resolution"], dtype=np.float64)
        shifted: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for raw_idx, (kps, conf, ts) in frames.items():
            logical = raw_idx - cam.sync_offset
            if logical >= 0:
                shifted[logical] = (kps * scale, conf, ts)
        per_cam_frames[cam.name] = shifted

    if keypoint_names is None or len(per_cam_frames) < min_views:
        raise SessionError(
            f"session {session.session_id!r}: 3D fusion needs >= {min_views} per-camera "
            f"CSVs under {session_out} (found {len(per_cam_frames)})"
        )

    frame_counts: dict[int, int] = {}
    for frames_map in per_cam_frames.values():
        for idx in frames_map:
            frame_counts[idx] = frame_counts.get(idx, 0) + 1

    # Timestamp source preference: world-frame camera first (it defines
    # the session time base), then remaining cameras in session order.
    ts_priority = sorted(per_cam_frames, key=lambda name: (name != calibration["world_frame"],))

    fused: list[tuple[int, float, np.ndarray, FusionDiagnostics]] = []
    for idx in sorted(idx for idx, count in frame_counts.items() if count >= min_views):
        kp_arrays: dict[str, np.ndarray] = {}
        conf_arrays: dict[str, np.ndarray] = {}
        for name, frames_map in per_cam_frames.items():
            if idx in frames_map:
                kp_arrays[name], conf_arrays[name], _ = frames_map[idx]
        timestamp = float("nan")
        for name in ts_priority:
            ts = per_cam_frames[name].get(idx, (None, None, float("nan")))[2]
            if np.isfinite(ts):
                timestamp = ts
                break
        world, diag = fuse_session_frame(
            kp_arrays,
            calibration,
            confidences=conf_arrays,
            min_views=min_views,
        )
        fused.append((idx, timestamp, world, diag))
    return SessionFusion(keypoint_names=keypoint_names, frames=fused)


def _fuse_and_report(session: Session, output_dir: str | pathlib.Path | None) -> None:
    """Fuse freshly written per-camera CSVs into 3D and write world3d.csv.

    Failures are non-fatal: the per-camera CSVs are already on disk
    and fusion can be re-run later via ``fuse_session_outputs``.
    """
    try:
        fusion = fuse_session_outputs(session, output_dir)
        if not fusion.frames:
            print("  3D fusion: no logical frame is visible from >= 2 cameras; nothing fused")
            return
        out_path = write_world3d_csv(
            _resolve_session_output(session, output_dir) / WORLD3D_FILENAME,
            session.session_id,
            fusion.keypoint_names,
            fusion.frames,
        )
    except Exception as exc:
        print(f"  WARNING: 3D fusion skipped: {exc}")
        return
    errs = np.concatenate([d["reprojection_error_px"] for _, _, _, d in fusion.frames])
    views = np.concatenate([d["n_views"] for _, _, _, d in fusion.frames])
    finite = np.isfinite(errs)
    mean_err = float(np.mean(errs[finite])) if finite.any() else float("nan")
    print(
        f"  3D fusion: {len(fusion.frames)} frame(s), mean reprojection {mean_err:.2f}px, "
        f"mean views/keypoint {float(np.mean(views)):.2f}"
    )
    print(f"    3D:   {out_path}")


__all__ = [
    "CAMERA_GLOB",
    "SESSION_FORMAT_VERSION",
    "SESSION_MANIFEST_FILENAME",
    "VIDEO_EXTENSIONS",
    "Session",
    "SessionCamera",
    "SessionError",
    "SessionFusion",
    "discover_session",
    "discover_sessions",
    "fuse_session_outputs",
    "iter_synchronized_frames",
    "process_session",
]
