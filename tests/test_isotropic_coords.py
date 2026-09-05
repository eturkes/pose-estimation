"""Diff-blind executable grading for M2.8.4 isotropic coordinates."""

from __future__ import annotations

import csv
import importlib
import json
import math
import pathlib
import runpy
import struct
from types import SimpleNamespace

import av
import cv2
import numpy as np
import pytest

import pose_estimation.export as export

corpus_run = importlib.import_module("pose_estimation.corpus_run")
golden_tests = importlib.import_module("test_r_clinical_goldens")
run_module = importlib.import_module("pose_estimation.run")
video_io = importlib.import_module("pose_estimation.video_io")

_ASPECTS = ((1080, 1920), (1920, 1080))


def _body_row(frame_h: int, frame_w: int, points: np.ndarray) -> dict[str, object]:
    landmarks = np.zeros((len(export.BODY_KEYPOINT_NAMES), 3), dtype=np.float64)
    landmarks[: len(points)] = points
    rows = export.frame_to_rows(
        video_name="synthetic",
        frame_idx=0,
        timestamp_sec=0.0,
        frame_h=frame_h,
        frame_w=frame_w,
        body_landmarks=[landmarks],
        body_visibilities=[np.ones(len(landmarks), dtype=np.float64)],
        hand_landmarks=[],
        matches=[],
        tracking=export.TRACKING_BODY,
    )
    assert len(rows) == 1
    return rows[0]


def _body_coords(row: dict[str, object], index: int) -> np.ndarray:
    prefix, names = export._body_keypoint_names(export.TRACKING_BODY)
    return np.array([row[f"{prefix}_{names[index]}_{axis}"] for axis in "xyz"], dtype=float)


def _geometry_metrics(coords: np.ndarray) -> tuple[float, float]:
    ba = coords[0] - coords[1]
    bc = coords[2] - coords[1]
    angle = math.degrees(
        math.acos(float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))))
    )
    ratio = float(np.linalg.norm(coords[0] - coords[1]) / np.linalg.norm(coords[2] - coords[3]))
    return angle, ratio


def test_p01_body_coordinates_use_one_max_dimension_scalar() -> None:
    """RED pre-fix: x/y/z must all use max(frame_w, frame_h)."""
    point = np.array([[960.0, 540.0, 270.0]])
    for frame_h, frame_w in _ASPECTS:
        actual = _body_coords(_body_row(frame_h, frame_w, point), 0)
        expected = point[0] / max(frame_w, frame_h)
        np.testing.assert_array_equal(actual, expected)


def test_p02_angles_and_distance_ratios_are_aspect_invariant() -> None:
    """RED pre-fix: transposing dimensions must preserve derived geometry."""
    points = np.array(
        [
            [100.0, 100.0, 0.0],
            [900.0, 250.0, 0.0],
            [300.0, 900.0, 0.0],
            [1000.0, 700.0, 0.0],
        ]
    )
    metrics = []
    for frame_h, frame_w in _ASPECTS:
        coords = np.vstack(
            [_body_coords(_body_row(frame_h, frame_w, points), index)[:2] for index in range(4)]
        )
        metrics.append(_geometry_metrics(coords))
    assert metrics[0] == metrics[1]


def test_p03_frame_corners_remain_inside_unit_range() -> None:
    """GREEN pre-fix: the isotropy repair must preserve the coordinate range."""
    for frame_h, frame_w in _ASPECTS:
        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [float(frame_w), 0.0, 0.0],
                [0.0, float(frame_h), 0.0],
                [float(frame_w), float(frame_h), 0.0],
            ]
        )
        row = _body_row(frame_h, frame_w, corners)
        exported = np.vstack([_body_coords(row, index)[:2] for index in range(4)])
        assert exported.shape == (4, 2)
        assert np.all((exported >= 0.0) & (exported <= 1.0))


def test_p04_coordinate_scale_survives_dimension_transpose() -> None:
    """RED pre-fix: a 90-degree dimension transpose must retain one scale."""
    point = np.array([[540.0, 360.0, 180.0]])
    exported = [_body_coords(_body_row(h, w, point), 0) for h, w in _ASPECTS]
    np.testing.assert_array_equal(exported[0], exported[1])


class _DimensionMismatchCapture:
    def __init__(self) -> None:
        self._read = False
        self.released = False

    def get(self, prop: int) -> float:
        values = {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 1.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }
        return values.get(prop, 0.0)

    def isOpened(self) -> bool:
        return not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._read:
            return False, None
        self._read = True
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _run_args() -> SimpleNamespace:
    return SimpleNamespace(
        tracking=export.TRACKING_BODY,
        headless=True,
        single_subject=False,
        max_frames=0,
    )


def test_p05_run_exports_using_decoded_frame_dimensions(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED pre-fix: capture properties must not size exported coordinates."""
    capture = _DimensionMismatchCapture()
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: capture)
    output = tmp_path / "coordinates.csv"
    keypoints = np.empty((1, 133, 2), dtype=np.float64)
    keypoints[..., 0] = 32.0
    keypoints[..., 1] = 24.0

    run_module.process_source(
        _run_args(),
        lambda _frame: (keypoints, np.ones((1, 133), dtype=np.float64)),
        "synthetic.mp4",
        draw_skeleton=None,
        output_csv=output,
    )

    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    prefix, names = export._body_keypoint_names(export.TRACKING_BODY)
    # Both axes divide by max(64, 48) = 64, so y is 24/64 and NOT 0.5; a y of 0.5
    # would mean per-axis division, which P01 refuses.  The properties would give
    # 32/640 and 24/640, so the pair still discriminates pixels from properties.
    assert float(row[f"{prefix}_{names[0]}_x"]) == 0.5
    assert float(row[f"{prefix}_{names[0]}_y"]) == 0.375
    assert capture.released


_FIXED_ONE = 1 << 16
_FIXED_W = 1 << 30
_TKHD_MATRIX_OFFSET = 48
_DISPLAY_MATRICES = {
    0: (_FIXED_ONE, 0, 0, 0, _FIXED_ONE, 0, 0, 0, _FIXED_W),
    90: (0, _FIXED_ONE, 0, -_FIXED_ONE, 0, 0, 0, 0, _FIXED_W),
    180: (-_FIXED_ONE, 0, 0, 0, -_FIXED_ONE, 0, 0, 0, _FIXED_W),
    270: (0, -_FIXED_ONE, 0, _FIXED_ONE, 0, 0, 0, 0, _FIXED_W),
}
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _iter_boxes(buffer: bytearray, start: int, end: int):
    position = start
    while position + 8 <= end:
        size = struct.unpack_from(">I", buffer, position)[0]
        kind = bytes(buffer[position + 4 : position + 8])
        if size == 0:
            size = end - position
        elif size == 1:
            size = struct.unpack_from(">Q", buffer, position + 8)[0]
        if size < 8 or position + size > end:
            return
        yield position, size, kind
        position += size


def _stamp_display_matrix(buffer: bytearray, start: int, end: int, matrix: tuple[int, ...]) -> bool:
    for position, size, kind in _iter_boxes(buffer, start, end):
        if kind == b"tkhd":
            struct.pack_into(">9i", buffer, position + _TKHD_MATRIX_OFFSET, *matrix)
            return True
        if kind in {b"moov", b"trak"} and _stamp_display_matrix(
            buffer, position + 8, position + size, matrix
        ):
            return True
    return False


def _write_rotated_clip(path: pathlib.Path, degrees: int) -> pathlib.Path:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        stream.options = {"crf": "0", "preset": "ultrafast", "tune": "stillimage"}
        for _ in range(2):
            pixels = np.zeros((32, 64, 3), dtype=np.uint8)
            pixels[:8, :16] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer = bytearray(path.read_bytes())
    assert _stamp_display_matrix(buffer, 0, len(buffer), _DISPLAY_MATRICES[degrees])
    path.write_bytes(buffer)
    return path


@pytest.fixture(scope="module")
def rotated_clips(tmp_path_factory: pytest.TempPathFactory) -> dict[int, pathlib.Path]:
    root = tmp_path_factory.mktemp("display-matrix")
    return {
        degrees: _write_rotated_clip(root / f"rotation-{degrees}.mp4", degrees)
        for degrees in _DISPLAY_MATRICES
    }


def _decode(
    path: pathlib.Path, *, orientation_auto: int | None = None
) -> tuple[np.ndarray, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    if orientation_auto is not None:
        assert capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, orientation_auto)
    reported = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert frame is not None
    return frame, reported


def _display_transform(frame: np.ndarray, degrees: int) -> np.ndarray:
    return frame if degrees == 0 else cv2.rotate(frame, _ROTATE_CODES[degrees])


@pytest.mark.parametrize("degrees", tuple(_DISPLAY_MATRICES))
def test_p06_backend_applies_declared_display_matrix(
    rotated_clips: dict[int, pathlib.Path], degrees: int
) -> None:
    """GREEN pre-fix: backend default rotation is the environmental premise."""
    displayed, reported = _decode(rotated_clips[degrees])
    coded, _ = _decode(rotated_clips[degrees], orientation_auto=0)
    np.testing.assert_array_equal(displayed, _display_transform(coded, degrees))
    assert reported == displayed.shape[1::-1]


@pytest.mark.parametrize("degrees", tuple(_DISPLAY_MATRICES))
def test_p07_corpus_pipeline_receives_display_oriented_frames(
    rotated_clips: dict[int, pathlib.Path], degrees: int
) -> None:
    """GREEN pre-fix: corpus decode must retain backend display orientation."""
    coded, _ = _decode(rotated_clips[degrees], orientation_auto=0)
    expected = _display_transform(coded, degrees)
    observed = []

    def record_frame(frame: np.ndarray) -> tuple[None, None]:
        observed.append(frame.copy())
        return None, None

    run_module.process_source(
        _run_args(),
        record_frame,
        str(rotated_clips[degrees]),
        draw_skeleton=None,
    )

    assert observed
    np.testing.assert_array_equal(observed[0], expected)
    capture = video_io.open_capture(str(rotated_clips[degrees]))
    assert capture is not None
    assert bool(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO))
    capture.release()


def test_p08_normalisation_identity_is_frozen_and_bound_to_its_behaviour() -> None:
    """A06 freezes all three slots; the binding conjunct is what stops token drift.

    A token is a claim about a scale, and a claim that nothing checks can outlive
    the behaviour it names — so the same case asserts the scale it denotes.
    """
    assert export.COORD_NORMALIZATION == "image-isotropic-maxdim"
    for frame_h, frame_w in _ASPECTS:
        assert export.coord_scale(frame_h, frame_w) == float(max(frame_w, frame_h))
    # The driver echoes the constant rather than restating the token, so the
    # report cannot carry an identity the export path does not implement.
    driver = runpy.run_path(str(_PROJECT_ROOT / "scripts" / "corpus_run_2d.py"), run_name="_p08")
    assert driver["COORD_NORMALIZATION"] == export.COORD_NORMALIZATION


def _hand_coords(row: dict[str, object], side: str, index: int = 0) -> np.ndarray:
    return np.array([row[f"{side}_hand_{index}_{axis}"] for axis in "xyz"], dtype=float)


def test_p09_body_matched_hand_and_hand_only_paths_are_isotropic() -> None:
    """RED pre-fix: all exported coordinate paths must share max-dimension semantics."""
    point = np.array([960.0, 540.0, 270.0])
    hand = np.zeros((export.HAND_KEYPOINT_COUNT, 3), dtype=np.float64)
    hand[0] = point
    body = np.zeros((len(export.BODY_KEYPOINT_NAMES), 3), dtype=np.float64)
    body[0] = point
    wrist_index, side = next(iter(export.wrist_to_side(export.TRACKING_BODY).items()))

    for frame_h, frame_w in _ASPECTS:
        expected = point / max(frame_h, frame_w)
        common = {
            "video_name": "synthetic",
            "frame_idx": 0,
            "timestamp_sec": 0.0,
            "frame_h": frame_h,
            "frame_w": frame_w,
        }
        matched = export.frame_to_rows(
            **common,
            body_landmarks=[body],
            body_visibilities=[np.ones(len(body), dtype=np.float64)],
            hand_landmarks=[hand],
            matches=[(0, wrist_index, 0)],
            tracking=export.TRACKING_BODY,
        )[0]
        fallback = export.frame_to_rows(
            **common,
            body_landmarks=[],
            body_visibilities=[],
            hand_landmarks=[hand],
            matches=[],
            tracking=export.TRACKING_BODY,
            hand_only=True,
            hand_handedness=[(side, 1.0)],
        )[0]
        hands_mode = export.frame_to_rows(
            **common,
            body_landmarks=[],
            body_visibilities=[],
            hand_landmarks=[hand],
            matches=[],
            tracking=export.TRACKING_HANDS,
            hand_handedness=[(side, 1.0)],
        )[0]

        np.testing.assert_array_equal(_body_coords(matched, 0), expected)
        np.testing.assert_array_equal(_hand_coords(matched, side), expected)
        np.testing.assert_array_equal(_hand_coords(fallback, side), expected)
        np.testing.assert_array_equal(_hand_coords(hands_mode, side), expected)


def test_p10_all_source_enumerated_2d_goldens_are_byte_identical(
    tmp_path: pathlib.Path,
) -> None:
    """GREEN pre-fix: all current 2D golden bytes remain fixed, not a stale count."""
    regenerated = tmp_path / "regenerated"
    golden_tests._load_generator().regenerate(regenerated)
    cases = [
        filename
        for dataset, entries in golden_tests._DATASETS.items()
        if dataset.startswith("2d_")
        for filename, _kind in entries
    ]
    assert cases
    assert len(cases) == len(set(cases))
    for filename in cases:
        assert (regenerated / filename).read_bytes() == (
            golden_tests._GOLDEN_DIR / filename
        ).read_bytes()


def test_p12_manifest_structure_uses_source_owned_vocabulary() -> None:
    """GREEN pre-fix: shipped validation pins the structural half of rerun totality."""
    canonical = tuple(f"asset-{index}" for index in range(len(corpus_run.ASSET_DISPOSITIONS)))
    rows = [
        {
            "asset_id": asset_id,
            "event_id": f"event-{index}",
            "camera_name": f"camera-{index}",
            "disposition": disposition,
        }
        for index, (asset_id, disposition) in enumerate(
            zip(canonical, corpus_run.ASSET_DISPOSITIONS, strict=True)
        )
    ]
    assert corpus_run.validate_manifest(rows, canonical) == dict.fromkeys(
        corpus_run.ASSET_DISPOSITIONS, 1
    )

    duplicate_drop = [*rows[:-1], {**rows[-1], "asset_id": rows[0]["asset_id"]}]
    with pytest.raises(corpus_run.ManifestError):
        corpus_run.validate_manifest(duplicate_drop, canonical)


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DETERMINISM_EVIDENCE = (
    (
        "scripts/check_qualify_determinism.py",
        "tests/qualify_determinism_results.json",
        "source_sha256",
    ),
    (
        "scripts/check_calibration_qc_determinism.py",
        "tests/calibration_qc_determinism_results.json",
        "source_digests",
    ),
)


@pytest.mark.parametrize(("checker_path", "result_path", "digest_key"), _DETERMINISM_EVIDENCE)
def test_p14_determinism_evidence_tracks_current_video_io_bytes(
    checker_path: str, result_path: str, digest_key: str
) -> None:
    """GREEN pre-fix, RED after source edit: evidence must be regenerated."""
    checker = runpy.run_path(str(_PROJECT_ROOT / checker_path))
    current = checker["source_digests"]()
    recorded = json.loads((_PROJECT_ROOT / result_path).read_text(encoding="utf-8"))[digest_key]
    video_io_path = "src/pose_estimation/video_io.py"
    assert video_io_path in current
    assert recorded == current
