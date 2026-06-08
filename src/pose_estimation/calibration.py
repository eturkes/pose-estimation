"""Camera calibration data layer.

Loads, validates, and persists ``SessionCalibration`` JSON files used by
the multi-camera pipeline.  This module is deliberately cv2-free; the
ChArUco solve workflow lives in ``charuco.py``.

The on-disk format is JSON with nested lists for matrices/vectors.
``load_calibration`` materialises ``np.ndarray`` instances so downstream
code does not branch on type.  See ``.claude/tech/calibration.md`` for
the schema.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any

import numpy as np

from ._types import CameraCalibration, SessionCalibration

CALIBRATION_FORMAT_VERSION = 1
"""Current ``format_version`` written by ``save_calibration``."""

CALIBRATION_FILENAME = "calibration.json"
"""Conventional filename inside a session directory."""

_DISTORTION_LENGTHS: tuple[int, ...] = (4, 5, 8, 12, 14)
"""OpenCV-accepted distortion-coefficient vector lengths."""

_WORLD_FRAME_TOLERANCE = 1e-9
"""Numerical slack when asserting the world-frame camera has zero pose."""


class CalibrationError(ValueError):
    """Raised when a calibration file fails schema validation."""


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_calibration(path: str | pathlib.Path) -> SessionCalibration:
    """Load and validate a ``calibration.json`` file.

    Raises ``CalibrationError`` on any schema violation.  Returns a
    ``SessionCalibration`` with ``np.ndarray`` matrix/vector fields.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise CalibrationError(f"calibration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"{p}: invalid JSON ({exc.msg})") from exc
    return _from_jsonable(raw, source=str(p))


def save_calibration(calibration: SessionCalibration, path: str | pathlib.Path) -> None:
    """Write a ``SessionCalibration`` to JSON.

    Validates before writing so partial/corrupt files never land on
    disk.  Numpy arrays are converted to nested lists.
    """
    data = _to_jsonable(calibration)
    _validate_jsonable(data, source=str(path))
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_session_calibration(
    session_dir: str | pathlib.Path,
) -> SessionCalibration | None:
    """Auto-discover ``calibration.json`` inside a session directory.

    Returns ``None`` when the file is absent (no calibration is a
    valid state for sessions that won't be triangulated).  Validation
    errors still raise ``CalibrationError``.
    """
    candidate = pathlib.Path(session_dir) / CALIBRATION_FILENAME
    if not candidate.exists():
        return None
    return load_calibration(candidate)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_jsonable(data: Any, *, source: str) -> None:
    """Validate the JSON-shaped representation (lists, not arrays).

    Used by both ``load_calibration`` (after parsing) and
    ``save_calibration`` (before writing) so the contract is identical
    in both directions.
    """
    if not isinstance(data, dict):
        raise CalibrationError(f"{source}: top-level value must be an object")
    version = data.get("format_version")
    if version != CALIBRATION_FORMAT_VERSION:
        raise CalibrationError(
            f"{source}: unsupported format_version={version!r} "
            f"(this build accepts {CALIBRATION_FORMAT_VERSION})"
        )
    for required in ("session_id", "world_frame", "cameras"):
        if required not in data:
            raise CalibrationError(f"{source}: missing required field {required!r}")
    if not isinstance(data["session_id"], str) or not data["session_id"]:
        raise CalibrationError(f"{source}: session_id must be a non-empty string")
    if not isinstance(data["world_frame"], str) or not data["world_frame"]:
        raise CalibrationError(f"{source}: world_frame must be a non-empty string")
    cameras = data["cameras"]
    if not isinstance(cameras, list) or not cameras:
        raise CalibrationError(f"{source}: cameras must be a non-empty array")
    seen_names: set[str] = set()
    for i, cam in enumerate(cameras):
        _validate_camera(cam, index=i, source=source, seen_names=seen_names)
    world_frame = data["world_frame"]
    if world_frame not in seen_names:
        raise CalibrationError(
            f"{source}: world_frame={world_frame!r} does not match any camera name "
            f"(have {sorted(seen_names)})"
        )
    for cam in cameras:
        if cam["name"] == world_frame and (
            not _is_zero_vec(cam["rvec"]) or not _is_zero_vec(cam["tvec"])
        ):
            raise CalibrationError(
                f"{source}: world_frame camera {world_frame!r} must have zero rvec/tvec"
            )


def _validate_camera(cam: Any, *, index: int, source: str, seen_names: set[str]) -> None:
    if not isinstance(cam, dict):
        raise CalibrationError(f"{source}: cameras[{index}] must be an object")
    for required in ("name", "resolution", "K", "distortion", "rvec", "tvec"):
        if required not in cam:
            raise CalibrationError(f"{source}: cameras[{index}] missing field {required!r}")
    name = cam["name"]
    if not isinstance(name, str) or not name:
        raise CalibrationError(f"{source}: cameras[{index}].name must be a non-empty string")
    if name in seen_names:
        raise CalibrationError(f"{source}: duplicate camera name {name!r}")
    seen_names.add(name)

    res = cam["resolution"]
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(x, int) for x in res)):
        raise CalibrationError(f"{source}: {name}.resolution must be [width, height] of ints")
    if res[0] <= 0 or res[1] <= 0:
        raise CalibrationError(f"{source}: {name}.resolution must be positive")

    K = cam["K"]
    if not (isinstance(K, list) and len(K) == 3 and all(len(row) == 3 for row in K)):
        raise CalibrationError(f"{source}: {name}.K must be a 3x3 matrix")
    if not (
        _approx_equal(K[2][0], 0.0) and _approx_equal(K[2][1], 0.0) and _approx_equal(K[2][2], 1.0)
    ):
        raise CalibrationError(f"{source}: {name}.K last row must be [0, 0, 1]")

    distortion = cam["distortion"]
    if not isinstance(distortion, list) or len(distortion) not in _DISTORTION_LENGTHS:
        raise CalibrationError(
            f"{source}: {name}.distortion length must be one of {_DISTORTION_LENGTHS}, "
            f"got {len(distortion) if isinstance(distortion, list) else type(distortion).__name__}"
        )

    for field in ("rvec", "tvec"):
        v = cam[field]
        if not (isinstance(v, list) and len(v) == 3):
            raise CalibrationError(f"{source}: {name}.{field} must be a length-3 vector")


def _is_zero_vec(v: list[float]) -> bool:
    return all(abs(float(x)) <= _WORLD_FRAME_TOLERANCE for x in v)


def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# (de)serialization
# ---------------------------------------------------------------------------


def _from_jsonable(data: Any, *, source: str) -> SessionCalibration:
    _validate_jsonable(data, source=source)
    cameras: dict[str, CameraCalibration] = {}
    for cam in data["cameras"]:
        cameras[cam["name"]] = CameraCalibration(
            name=cam["name"],
            resolution=(int(cam["resolution"][0]), int(cam["resolution"][1])),
            K=np.asarray(cam["K"], dtype=np.float64),
            distortion=np.asarray(cam["distortion"], dtype=np.float64),
            rvec=np.asarray(cam["rvec"], dtype=np.float64),
            tvec=np.asarray(cam["tvec"], dtype=np.float64),
        )
    return SessionCalibration(
        format_version=int(data["format_version"]),
        session_id=str(data["session_id"]),
        world_frame=str(data["world_frame"]),
        cameras=cameras,
        reprojection_error_px=float(data.get("reprojection_error_px", 0.0)),
        solver=str(data.get("solver", "unknown")),
        solved_at=str(data.get("solved_at", "")),
    )


def _to_jsonable(calibration: SessionCalibration) -> dict[str, Any]:
    cameras_list: list[dict[str, Any]] = []
    for name, cam in calibration["cameras"].items():
        if cam["name"] != name:
            raise CalibrationError(
                f"cameras[{name!r}].name={cam['name']!r} does not match dict key"
            )
        cameras_list.append(
            {
                "name": cam["name"],
                "resolution": [int(cam["resolution"][0]), int(cam["resolution"][1])],
                "K": np.asarray(cam["K"], dtype=np.float64).tolist(),
                "distortion": np.asarray(cam["distortion"], dtype=np.float64).tolist(),
                "rvec": np.asarray(cam["rvec"], dtype=np.float64).tolist(),
                "tvec": np.asarray(cam["tvec"], dtype=np.float64).tolist(),
            }
        )
    return {
        "format_version": calibration["format_version"],
        "session_id": calibration["session_id"],
        "world_frame": calibration["world_frame"],
        "cameras": cameras_list,
        "reprojection_error_px": float(calibration.get("reprojection_error_px", 0.0)),
        "solver": calibration.get("solver", "unknown"),
        "solved_at": calibration.get("solved_at", ""),
    }


# ---------------------------------------------------------------------------
# Solver support — the ChArUco solve itself lives in charuco.py
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """ISO-8601 UTC ``solved_at`` value for fresh calibrations."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "CALIBRATION_FILENAME",
    "CALIBRATION_FORMAT_VERSION",
    "CalibrationError",
    "load_calibration",
    "load_session_calibration",
    "save_calibration",
    "utc_timestamp",
]
