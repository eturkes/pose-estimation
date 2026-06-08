"""Tests for ``pose_estimation.calibration`` IO + validation."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from pose_estimation.calibration import (
    CALIBRATION_FILENAME,
    CALIBRATION_FORMAT_VERSION,
    CalibrationError,
    load_calibration,
    load_session_calibration,
    save_calibration,
    utc_timestamp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_calibration_dict() -> dict:
    """A minimally valid calibration with 2 cameras (cam1 is the world frame)."""
    return {
        "format_version": CALIBRATION_FORMAT_VERSION,
        "session_id": "test_session",
        "world_frame": "cam1",
        "cameras": [
            {
                "name": "cam1",
                "resolution": [1920, 1080],
                "K": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "rvec": [0.0, 0.0, 0.0],
                "tvec": [0.0, 0.0, 0.0],
            },
            {
                "name": "cam2",
                "resolution": [1920, 1080],
                "K": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "rvec": [0.0, 0.5, 0.0],
                "tvec": [0.3, 0.0, 0.0],
            },
        ],
        "reprojection_error_px": 0.3,
        "solver": "test",
        "solved_at": "2026-05-18T12:00:00Z",
    }


@pytest.fixture
def calibration_path(tmp_path: pathlib.Path, minimal_calibration_dict: dict) -> pathlib.Path:
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(minimal_calibration_dict))
    return p


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_load_returns_typed_arrays(calibration_path: pathlib.Path):
    calib = load_calibration(calibration_path)
    assert calib["format_version"] == CALIBRATION_FORMAT_VERSION
    assert calib["world_frame"] == "cam1"
    assert set(calib["cameras"].keys()) == {"cam1", "cam2"}
    cam2 = calib["cameras"]["cam2"]
    assert isinstance(cam2["K"], np.ndarray)
    assert cam2["K"].shape == (3, 3)
    assert isinstance(cam2["distortion"], np.ndarray)
    assert cam2["distortion"].shape == (5,)
    assert isinstance(cam2["rvec"], np.ndarray)
    assert cam2["rvec"].shape == (3,)
    assert isinstance(cam2["tvec"], np.ndarray)
    assert cam2["tvec"].shape == (3,)
    assert cam2["resolution"] == (1920, 1080)


def test_save_then_load_roundtrips(tmp_path: pathlib.Path, calibration_path: pathlib.Path):
    calib = load_calibration(calibration_path)
    out = tmp_path / "out" / "calib_roundtrip.json"
    save_calibration(calib, out)
    assert out.is_file()
    reloaded = load_calibration(out)
    assert reloaded["session_id"] == calib["session_id"]
    assert reloaded["world_frame"] == calib["world_frame"]
    for name in calib["cameras"]:
        np.testing.assert_array_equal(reloaded["cameras"][name]["K"], calib["cameras"][name]["K"])
        np.testing.assert_array_equal(
            reloaded["cameras"][name]["distortion"],
            calib["cameras"][name]["distortion"],
        )


def test_load_session_calibration_discovers_file(
    tmp_path: pathlib.Path, minimal_calibration_dict: dict
):
    (tmp_path / CALIBRATION_FILENAME).write_text(json.dumps(minimal_calibration_dict))
    calib = load_session_calibration(tmp_path)
    assert calib is not None
    assert calib["world_frame"] == "cam1"


def test_load_session_calibration_returns_none_when_missing(tmp_path: pathlib.Path):
    assert load_session_calibration(tmp_path) is None


# ---------------------------------------------------------------------------
# Validation errors — schema
# ---------------------------------------------------------------------------


def _save_dict(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    return p


def test_unknown_format_version_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["format_version"] = 99
    with pytest.raises(CalibrationError, match="format_version"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_missing_world_frame_field_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    del minimal_calibration_dict["world_frame"]
    with pytest.raises(CalibrationError, match="world_frame"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_world_frame_must_match_a_camera(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["world_frame"] = "ghost"
    with pytest.raises(CalibrationError, match="does not match any camera"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_world_frame_camera_must_have_zero_pose(
    tmp_path: pathlib.Path, minimal_calibration_dict: dict
):
    minimal_calibration_dict["cameras"][0]["tvec"] = [1.0, 0.0, 0.0]
    with pytest.raises(CalibrationError, match="zero rvec/tvec"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_duplicate_camera_name_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["cameras"][1]["name"] = "cam1"
    with pytest.raises(CalibrationError, match="duplicate camera name"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_invalid_K_shape_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["cameras"][0]["K"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(CalibrationError, match="3x3"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_invalid_K_last_row_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["cameras"][0]["K"][2] = [0.0, 0.0, 2.0]
    with pytest.raises(CalibrationError, match=r"last row"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_invalid_distortion_length_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["cameras"][0]["distortion"] = [0.0, 0.0]
    with pytest.raises(CalibrationError, match="distortion length"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_invalid_rvec_length_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["cameras"][1]["rvec"] = [0.0, 0.0]
    with pytest.raises(CalibrationError, match=r"rvec.*length-3"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_negative_resolution_rejected(tmp_path: pathlib.Path, minimal_calibration_dict: dict):
    minimal_calibration_dict["cameras"][0]["resolution"] = [-1, 1080]
    with pytest.raises(CalibrationError, match="resolution must be positive"):
        load_calibration(_save_dict(tmp_path, minimal_calibration_dict))


def test_load_missing_file_raises(tmp_path: pathlib.Path):
    with pytest.raises(CalibrationError, match="not found"):
        load_calibration(tmp_path / "no_such_file.json")


def test_invalid_json_raises(tmp_path: pathlib.Path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    with pytest.raises(CalibrationError, match="invalid JSON"):
        load_calibration(p)


# ---------------------------------------------------------------------------
# Helpers (the ChArUco solver itself is covered by test_charuco.py)
# ---------------------------------------------------------------------------


def test_utc_timestamp_iso_format():
    ts = utc_timestamp()
    # Loose format check: 2026-05-18T12:34:56Z
    assert len(ts) == 20
    assert ts.endswith("Z")
    assert ts[10] == "T"
