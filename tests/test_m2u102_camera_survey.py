"""M2.10.2 P01-P09: contract-derived synthetic camera-survey witnesses."""

from __future__ import annotations

import csv
import importlib
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DRIVERS = ("corpus_run_2d.py", "pilot_corpus_run.py")
_CAMERA = ("cam_a", "cam_b", "cam_tx", "cam_ty")
_DIAGNOSTICS = (
    "task_start_frame",
    "task_end_frame",
    "frames_outside_task",
    "camera_reference_frame",
    "camera_steps_unmeasured",
    "camera_speed_p50",
    "camera_speed_p90",
)
_SETTINGS = ("camera_survey", "task_gap_s")
_REPORT = (
    "task_span",
    "frames_decoded",
    "frames_outside_task",
    "assets_trimmed",
    "assets_without_reference",
)


def _camera():
    return importlib.import_module("pose_estimation.camera_survey")


def _similarity(degrees=0.0, scale=1.0, tx=0.0, ty=0.0, centre=(0.0, 0.0)):
    theta = math.radians(degrees)
    a, b = scale * math.cos(theta), scale * math.sin(theta)
    cx, cy = centre
    return np.array(
        [[a, -b, tx + cx - a * cx + b * cy], [b, a, ty + cy - b * cx - a * cy], [0.0, 0.0, 1.0]]
    )


def _csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        return fields, list(reader)


def _driver(name):
    return runpy.run_path(str(_ROOT / "scripts" / name), run_name="_m2u102_driver")[
        "main"
    ].__globals__


def test_p09_run_defaults_and_explicit_flags():
    run = importlib.import_module("pose_estimation.run")
    args = run.parse_args([])
    assert getattr(args, "camera_survey", None) is False
    assert getattr(args, "task_gap_s", None) == 2.5
    args = run.parse_args(["--camera-survey", "--task-gap-s", "0.75"])
    assert args.camera_survey is True
    assert args.task_gap_s == 0.75


@pytest.mark.parametrize("enabled", [False, True])
def test_p09_run_main_forwards_through_session_dispatch(tmp_path, monkeypatch, enabled):
    from test_corpus_run_preconditions import _make_session
    from test_m2u91_subject_tracker import _solution_factory

    run = importlib.import_module("pose_estimation.run")
    session = _make_session(tmp_path / "sessions")
    calls = []
    monkeypatch.setattr(
        run, "SplitDeviceSolution", _solution_factory(lambda _index: np.empty((0, 4)))
    )
    monkeypatch.setattr(run, "process_source", lambda *_a, **kw: calls.append(kw) or [])
    monkeypatch.setattr(run, "resolve_cli_sessions", lambda *_a, **_k: [session])
    options = ["--camera-survey"] if enabled else []
    run.main(
        [
            "--session-dir",
            str(session.directory),
            "--output-dir",
            str(tmp_path / "out"),
            "--headless",
            "--backend",
            "onnxruntime",
            "--task-gap-s",
            "1.25",
            *options,
        ]
    )
    assert len(calls) == len(session.cameras) > 0
    assert all(call.get("camera_survey") is enabled for call in calls)
    assert all(call.get("task_gap_s") == 1.25 for call in calls)


def test_p09_pose_config_and_diagnostic_field_vocabulary():
    corpus = importlib.import_module("pose_estimation.corpus_run")
    run = importlib.import_module("pose_estimation.run")
    assert set(_SETTINGS) <= set(corpus.POSE_CONFIG_FIELDS)
    assert set(_DIAGNOSTICS) <= set(run.SOURCE_DIAGNOSTIC_FIELDS)


@pytest.mark.parametrize("name", _DRIVERS)
def test_p09_nc5_driver_defaults_and_generation_vocabulary(name, monkeypatch):
    driver = _driver(name)
    monkeypatch.setattr(sys, "argv", ["driver"])
    args = driver["_parse_args"]()
    assert getattr(args, "camera_survey", None) is True, "NC5: driver must opt into survey"
    assert getattr(args, "task_gap_s", None) == 2.5
    assert {*_SETTINGS, *_REPORT} <= set(driver["REPORT_FIELDS"])
    assert driver["GENERATOR_VERSION"] == ("v6" if name == "corpus_run_2d.py" else "v5")


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize("enabled", [False, True])
def test_p09_driver_flags_forward_effective_settings(name, enabled, tmp_path, monkeypatch):
    from test_m2u91_subject_tracker import _driver_args

    driver = _driver(name)
    monkeypatch.setattr(
        sys,
        "argv",
        ["driver", "--camera-survey" if enabled else "--no-camera-survey", "--task-gap-s", "0.75"],
    )
    parsed = driver["_parse_args"]()
    assert parsed.camera_survey is enabled
    assert parsed.task_gap_s == 0.75
    args = _driver_args(driver, monkeypatch, tmp_path)
    args.camera_survey, args.task_gap_s = enabled, 0.75
    commands = []
    monkeypatch.setattr(
        driver["subprocess"], "call", lambda command, **_kw: commands.append(command) or 0
    )
    if name == "corpus_run_2d.py":
        driver["_attempt_event"]("synthetic-event", args, tmp_path / "logs")
    else:
        (tmp_path / "logs").mkdir(exist_ok=True)
        driver["_run_event"](
            args.sessions / "synthetic-event", args.out, tmp_path / "logs" / "run.log", args
        )
    commands = [command for command in commands if "pose_estimation.run" in command]
    assert len(commands) == 1
    run = importlib.import_module("pose_estimation.run")
    forwarded = run.parse_args(commands[0][commands[0].index("pose_estimation.run") + 1 :])
    assert forwarded.camera_survey is enabled
    assert forwarded.task_gap_s == 0.75


def _generation(name, root, monkeypatch, *, enabled=True, gap=2.5, statistics=None):
    from test_m2u91_subject_tracker import _driver_args

    driver = _driver(name)
    args = _driver_args(driver, monkeypatch, root)
    args.camera_survey, args.task_gap_s = enabled, gap
    event = "synthetic-event"
    statistics = ((10, 4, 3), (6, 0, None), (8, 1, 0)) if statistics is None else statistics
    decoded, outside, reference = tuple(zip(*statistics, strict=True))
    cameras = tuple(f"cam-{chr(97 + i)}" for i in range(len(statistics)))
    _csv(
        args.inventory / "assets.csv",
        ("asset_id", "disposition", "reported_rotation_deg", "reported_frame_count"),
        [
            {
                "asset_id": f"asset-{i}",
                "disposition": "canonical",
                "reported_rotation_deg": "0",
                "reported_frame_count": str(n),
            }
            for i, n in enumerate(decoded)
        ],
    )
    _csv(
        args.qualification / "assets_qc.csv",
        ("asset_id", "codec", "device_config", "pts_monotonic"),
        [
            {
                "asset_id": f"asset-{i}",
                "codec": "h264",
                "device_config": "synthetic",
                "pts_monotonic": "1",
            }
            for i in range(len(decoded))
        ],
    )
    _csv(
        args.sessions / "placements.csv",
        ("asset_id", "event_id", "camera_name", "placement"),
        [
            {
                "asset_id": f"asset-{i}",
                "event_id": event,
                "camera_name": camera,
                "placement": "placed",
            }
            for i, camera in enumerate(cameras)
        ],
    )
    (args.sessions / "generation.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(driver, "_parse_args", lambda: args)
    monkeypatch.setitem(driver, "validate_generation", lambda *_a, **_k: {})
    if name == "pilot_corpus_run.py":
        args.min_assets = 1
        monkeypatch.setitem(driver, "_assert_sources_validated", lambda *_a: None)
        monkeypatch.setitem(
            driver, "_guard_verdicts", lambda *_a: {"events_probed": 1, "default_output_refused": 1}
        )
    pilot = driver["pilot"].__dict__ if name == "corpus_run_2d.py" else driver
    qc_header, _ = pilot["_r_constants"]()
    generated = []
    event_out = args.out / event
    run = importlib.import_module("pose_estimation.run")

    def subprocess_call(command, **_kwargs):
        if "pose_estimation.run" in command:
            generated.append(command.copy())
            event_out.mkdir(parents=True, exist_ok=True)
            for i, camera in enumerate(cameras):
                video = f"{event}/{camera}"
                _csv(
                    event_out / f"{camera}.csv",
                    ("video", "frame_idx", "person_idx"),
                    [{"video": video, "frame_idx": "0", "person_idx": "0"}],
                )
                diag = dict.fromkeys(run.SOURCE_DIAGNOSTIC_FIELDS, "0")
                diag.update(
                    video=video,
                    n_frames_decoded=str(decoded[i]),
                    pts_accepted=str(decoded[i]),
                    latency_ms_mean="1",
                    latency_ms_p95="1",
                )
                # New fields are explicit fixture data, never conditional on producer support.
                diag.update(
                    task_start_frame="0",
                    task_end_frame=str(decoded[i] - (outside[i] if args.camera_survey else 0)),
                    frames_outside_task=str(outside[i] if args.camera_survey else 0),
                    camera_reference_frame=str(reference[i])
                    if args.camera_survey and reference[i] is not None
                    else "",
                    camera_steps_unmeasured="0",
                    camera_speed_p50="0.010000" if args.camera_survey else "",
                    camera_speed_p90="0.020000" if args.camera_survey else "",
                )
                _csv(event_out / f"{camera}_diag.csv", tuple(diag), [diag])
        elif command[0] == "Rscript":
            for camera in cameras:
                _csv(event_out / f"{camera}_clinical_group_qc.csv", qc_header, [])
                _csv(
                    event_out / f"{camera}_clinical_windows.csv",
                    ("video", "person_idx"),
                    [{"video": f"{event}/{camera}", "person_idx": "0"}],
                )
        return 0

    monkeypatch.setattr(driver["subprocess"], "call", subprocess_call)
    assert driver["main"]() == 0
    assert len(generated) == 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["population"]["events"] == 1
    assert report["verdicts"]
    assert all(report["verdicts"].values())
    return driver, args, generated, event_out


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize("enabled", [False, True])
def test_p09_configuration_marker_and_task_span_report(name, enabled, tmp_path, monkeypatch):
    driver, args, _, event_out = _generation(name, tmp_path, monkeypatch, enabled=enabled, gap=0.75)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    marker = json.loads((event_out / "pose_config.json").read_text(encoding="utf-8"))
    for config in (report["configuration"], marker):
        assert config.get("camera_survey") is enabled
        assert config.get("task_gap_s") == 0.75
    assert report.get("task_span") == {
        "frames_decoded": 24,
        "frames_outside_task": 5 if enabled else 0,
        "assets_trimmed": 2 if enabled else 0,
        "assets_without_reference": 1 if enabled else 3,
    }
    assert {*_SETTINGS, *_REPORT} <= set(driver["REPORT_FIELDS"])


@pytest.mark.parametrize("name", _DRIVERS)
def test_p09_same_settings_reuse_complete_event(name, tmp_path, monkeypatch):
    driver, args, generated, event_out = _generation(name, tmp_path, monkeypatch)
    before = (event_out / "pose_config.json").read_bytes()
    assert json.loads(before).get("camera_survey") is True
    if name == "pilot_corpus_run.py":
        args.reuse_run = True
    assert driver["main"]() == 0
    assert len(generated) == 1
    assert (event_out / "pose_config.json").read_bytes() == before


_CHANGES = [
    ("camera_survey", True, False),
    ("camera_survey", False, True),
    ("task_gap_s", 2.5, 0.0),
    ("task_gap_s", 0.0, 3.25),
]


@pytest.mark.parametrize(("setting", "old", "new"), _CHANGES)
def test_p09_changed_setting_makes_event_due(setting, old, new, tmp_path, monkeypatch):
    driver, args, generated, event_out = _generation(
        "corpus_run_2d.py",
        tmp_path,
        monkeypatch,
        enabled=old if setting == "camera_survey" else True,
        gap=old if setting == "task_gap_s" else 2.5,
    )
    setattr(args, setting, new)
    assert driver["main"]() == 0
    assert len(generated) == 2, f"changed {setting} reused stale pose rows"
    assert json.loads((event_out / "pose_config.json").read_text(encoding="utf-8"))[setting] == new


@pytest.mark.parametrize(
    ("name", "mode"), [("corpus_run_2d.py", "analyse_only"), ("pilot_corpus_run.py", "reuse_run")]
)
@pytest.mark.parametrize(("setting", "old", "new"), _CHANGES)
def test_p09_changed_setting_refuses_analysis_or_reuse(
    name, mode, setting, old, new, tmp_path, monkeypatch, capsys
):
    driver, args, generated, event_out = _generation(
        name,
        tmp_path,
        monkeypatch,
        enabled=old if setting == "camera_survey" else True,
        gap=old if setting == "task_gap_s" else 2.5,
    )
    before = {p.name: p.read_bytes() for p in event_out.iterdir() if p.is_file()}
    assert {"cam-a.csv", "pose_config.json"} <= before.keys()
    setattr(args, setting, new)
    setattr(args, mode, True)
    refusal = driver.get("RunError", driver.get("PilotError"))
    assert refusal is not None
    capsys.readouterr()
    try:
        result = driver["main"]()
    except refusal as error:
        message = str(error)
    else:
        captured = capsys.readouterr()
        message = captured.out + captured.err
        assert result != 0, f"{mode} accepted mismatched {setting}"
    assert any(
        word in message.lower()
        for word in ("camera", "task", "config", "generation", "provenance", "mismatch")
    )
    assert len(generated) == 1
    for filename, content in before.items():
        assert (event_out / filename).read_bytes() == content


def _r(body):
    program = 'suppressWarnings(try(source("analysis/clinical_features.R"), silent=TRUE))\n' + body
    result = subprocess.run(
        ["Rscript", "-e", program], cwd=_ROOT, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "missing",
    [(), ("cam_a",), ("cam_a", "cam_b", "cam_tx", "cam_ty")],
    ids=["all-columns", "partial-columns", "no-columns"],
)
def test_p08_nc4_r_mapping_all_pairs_na_rows_and_z_untouched(tmp_path, missing):
    export = cast(Any, importlib.import_module("pose_estimation.export"))
    pairs = [field[:-2] for field in export.make_csv_header("body") if field.endswith("_x")]
    assert any(p.startswith("body_") for p in pairs)
    assert any(p.startswith("left_hand_") for p in pairs)
    assert any(p.startswith("right_hand_") for p in pairs)
    rng = np.random.default_rng(210208)
    rows = []
    for i in range(8):
        row: dict[str, Any] = {f"{p}_{axis}": float(rng.normal()) for p in pairs for axis in "xyz"}
        row.update(
            video="synthetic",
            frame_idx=i,
            person_idx=0,
            cam_a=0.4,
            cam_b=-0.8,
            cam_tx=0.123456,
            cam_ty=-0.234567,
        )
        if i < 4:
            row[_CAMERA[i]] = "NA"
        for field in missing:
            row.pop(field)
        rows.append(row)
    source, target = tmp_path / "input.csv", tmp_path / "mapped.csv"
    _csv(source, tuple(rows[0]), rows)
    _r(
        f'd <- read.csv({json.dumps(str(source))}, check.names=FALSE); out <- stabilize_camera(d); write.csv(out, {json.dumps(str(target))}, row.names=FALSE, na="NA")'
    )
    fields, actual = _read_csv(target)
    assert fields == list(rows[0])
    assert len(actual) == len(rows)
    for i, (before, after) in enumerate(zip(rows, actual, strict=True)):
        for pair in pairs:
            x, y = before[f"{pair}_x"], before[f"{pair}_y"]
            expected = (
                (x, y)
                if missing or i < 4
                else (0.4 * x + 0.8 * y + 0.123456, -0.8 * x + 0.4 * y - 0.234567)
            )
            assert float(after[f"{pair}_x"]) == pytest.approx(expected[0], abs=1e-13)
            assert float(after[f"{pair}_y"]) == pytest.approx(expected[1], abs=1e-13)
            assert float(after[f"{pair}_z"]) == pytest.approx(before[f"{pair}_z"], abs=1e-13)
        assert after["video"] == before["video"]
        assert int(after["frame_idx"]) == i


def test_p08_similarity_preserves_relative_joint_angles(tmp_path):
    from test_r_pipeline import _generate_csv

    path = tmp_path / "angles.csv"
    _generate_csv(path, "body", n_frames=12)
    _r(f"""
    d <- readr::read_csv({json.dumps(str(path))}, show_col_types=FALSE)
    d$cam_a <- 1.7*cos(0.37); d$cam_b <- 1.7*sin(0.37)
    d$cam_tx <- 0.27; d$cam_ty <- -0.13
    transformed <- stabilize_camera(d)
    original_angles <- compute_frame_features(d, "body")
    mapped_angles <- compute_frame_features(transformed, "body")
    columns <- grep("_(elbow_angle|wrist_deviation|finger_spread)_deg$", names(original_angles), value=TRUE)
    stopifnot(length(columns) >= 6)
    for (column in columns) {{
      x <- original_angles[[column]]; y <- mapped_angles[[column]]
      stopifnot(any(is.finite(x)), identical(is.na(x), is.na(y)))
      stopifnot(max(abs(x-y), na.rm=TRUE) < 1e-8)
    }}
    """)


def test_p08_cli_applies_stabilization_before_feature_computation(tmp_path):
    from test_r_pipeline import _generate_csv

    truth = tmp_path / "truth.csv"
    _generate_csv(truth, "body", n_frames=100)
    fields, rows = _read_csv(truth)
    pairs = [field[:-2] for field in fields if field.endswith("_x")]
    assert pairs
    assert rows
    shaken_rows = []
    for i, row in enumerate(rows):
        changed = row.copy()
        tx, ty = 0.15 * math.sin(i * 0.17), 0.08 * math.cos(i * 0.23)
        for pair in pairs:
            for axis, shift in (("x", tx), ("y", ty)):
                key = f"{pair}_{axis}"
                if row[key] not in ("", "NA"):
                    changed[key] = str(float(row[key]) - shift)
        changed.update(cam_a="1", cam_b="0", cam_tx=str(tx), cam_ty=str(ty))
        shaken_rows.append(changed)
    shaken = tmp_path / "shaken.csv"
    _csv(shaken, [*fields, *_CAMERA], shaken_rows)
    for path in (truth, shaken):
        result = subprocess.run(
            ["Rscript", str(_ROOT / "analysis" / "clinical_features.R"), str(path)],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    for suffix in ("_clinical.csv", "_clinical_windows.csv"):
        expected_fields, expected = _read_csv(truth.with_name(truth.stem + suffix))
        actual_fields, actual = _read_csv(shaken.with_name(shaken.stem + suffix))
        assert actual_fields == expected_fields
        assert len(actual) == len(expected) > 0
        compared = 0
        for before, after in zip(expected, actual, strict=True):
            for key in expected_fields:
                if key == "video":
                    continue
                try:
                    lhs, rhs = float(before[key]), float(after[key])
                except ValueError:
                    assert after[key] == before[key]
                else:
                    assert rhs == pytest.approx(lhs, rel=2e-6, abs=1e-8, nan_ok=True), key
                    compared += 1
        assert compared > 0


@pytest.mark.parametrize("dimensions", [(480, 640), (640, 480), (333, 777), (1, 1)])
def test_p05_export_normalized_similarity_matches_pixel_geometry(dimensions):
    export = cast(Any, importlib.import_module("pose_estimation.export"))
    height, width = dimensions
    transform = _similarity(degrees=17, scale=1.125, tx=-12.345678, ty=7.891011)
    actual = export.camera_values(transform, height, width)
    expected = [
        transform[0, 0],
        transform[1, 0],
        transform[0, 2] / export.coord_scale(height, width),
        transform[1, 2] / export.coord_scale(height, width),
    ]
    assert list(actual) == list(_CAMERA)
    assert list(actual.values()) == [f"{v:.6f}" for v in expected]
    a, b, tx, ty = (float(actual[key]) for key in _CAMERA)
    rng = np.random.default_rng(210205)
    for pixel in rng.uniform(0, 1, (30, 2)) * (width, height):
        x, y = pixel / export.coord_scale(height, width)
        projected = transform @ [*pixel, 1.0]
        mapped = np.array([a * x - b * y + tx, b * x + a * y + ty])
        np.testing.assert_allclose(
            mapped, projected[:2] / export.coord_scale(height, width), atol=1.5e-6, rtol=0
        )


def test_p06_header_and_writer_append_camera_columns_only_when_enabled(tmp_path):
    export = cast(Any, importlib.import_module("pose_estimation.export"))
    legacy = export.make_csv_header("body")
    assert len(legacy) == 304
    assert list(export.CAMERA_COLUMNS) == list(_CAMERA)
    assert export.make_csv_header("body", camera=False) == legacy
    assert export.make_csv_header("body", camera=True) == [*legacy, *_CAMERA]
    for enabled in (False, True):
        path = tmp_path / f"camera-{enabled}.csv"
        handle, writer = export.open_csv_writer(path, "body", camera=enabled)
        with handle:
            # A02: `open_csv_writer` returns a csv.DictWriter (its shipped contract).
            writer.writerow(dict.fromkeys([*legacy, *_CAMERA] if enabled else legacy, "0"))
        fields, rows = _read_csv(path)
        assert fields == ([*legacy, *_CAMERA] if enabled else legacy)
        assert len(rows) == 1


def test_p08_legacy_r_goldens_remain_byte_identical(tmp_path):
    from test_r_clinical_goldens import _DATASETS, _GOLDEN_DIR, _load_generator

    _load_generator().regenerate(tmp_path)
    cases = [filename for entries in _DATASETS.values() for filename, _ in entries]
    assert cases
    for filename in cases:
        assert (tmp_path / filename).read_bytes() == (_GOLDEN_DIR / filename).read_bytes(), filename


class _Capture:
    def __init__(self, frames, fps=10.0):
        self.frames, self.fps, self.index = frames, fps, 0
        self.released = False

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 120,
            cv2.CAP_PROP_POS_MSEC: max(0, self.index - 1) * 1000 / self.fps,
        }.get(prop, 0.0)

    def isOpened(self):
        return not self.released

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, None if frame is None else frame.copy()

    def release(self):
        self.released = True


class _Tracker:
    def __init__(self):
        self.posed, self.detected, self.resets = [], [], 0

    def reset(self):
        self.resets += 1

    def det_model(self, frame):
        self.detected.append(int(frame[0, 0, 0]))
        return np.array([[10.0, 10.0, 90.0, 90.0]])

    def __call__(self, frame):
        self.posed.append(int(frame[0, 0, 0]))
        return np.full((1, 133, 2), 20.0), np.full((1, 133), 0.9)


def _process(root, monkeypatch, *, frames=None, source=None, max_frames=0, **options):
    run = cast(Any, importlib.import_module("pose_estimation.run"))
    video_io = importlib.import_module("pose_estimation.video_io")
    frames = [np.full((120, 160, 3), i, np.uint8) for i in range(12)] if frames is None else frames
    captures = []
    tracker = _Tracker()

    def open_capture(*_args, **_kwargs):
        capture = _Capture(frames)
        captures.append(capture)
        return capture

    monkeypatch.setattr(run, "open_capture", open_capture)
    monkeypatch.setattr(video_io, "open_capture", open_capture)
    if options.get("camera_survey"):
        camera = _camera()
        if hasattr(camera, "open_capture"):
            monkeypatch.setattr(camera, "open_capture", open_capture)
    root.mkdir(parents=True, exist_ok=True)
    source = str(root / "synthetic.avi") if source is None else source
    if not str(source).isdigit():
        Path(source).touch()
    path, diag = root / "landmarks.csv", root / "diag.csv"
    run.process_source(
        run.parse_args(["--headless", "--tracking", "body", "--max-frames", str(max_frames)]),
        tracker,
        source,
        draw_skeleton=None,
        output_csv=str(path),
        output_diag=str(diag),
        video_name="synthetic",
        **options,
    )
    fields, rows = _read_csv(path)
    _, diagnostics = _read_csv(diag)
    assert len(diagnostics) == 1
    assert captures
    assert all(c.released for c in captures)
    return fields, rows, diagnostics[0], tracker, captures, path.read_bytes()


def test_p06_disabled_survey_preserves_legacy_csv_bytes(tmp_path, monkeypatch):
    legacy = _process(tmp_path / "legacy", monkeypatch)
    explicit = _process(tmp_path / "explicit", monkeypatch, camera_survey=False)
    assert explicit[-1] == legacy[-1]
    assert len(explicit[0]) == 304
    assert len(explicit[1]) == 12
    assert explicit[3].posed == list(range(12))
    assert len(explicit[4]) == 1
    assert explicit[2]["task_start_frame"] == "0"
    assert explicit[2]["task_end_frame"] == "12"
    for field in _DIAGNOSTICS[2:]:
        assert explicit[2][field] in ("", "0", "0.000000")


def test_p07_textureless_file_two_passes_no_reference_exports_identity(tmp_path, monkeypatch):
    fields, rows, diag, tracker, captures, _ = _process(tmp_path, monkeypatch, camera_survey=True)
    assert len(captures) == 2
    assert tracker.detected == [0, 5, 10]
    assert tracker.posed == list(range(12))
    assert tracker.resets >= 1
    assert fields[-4:] == list(_CAMERA)
    assert len(rows) == 12
    assert [int(row["frame_idx"]) for row in rows] == list(range(12))
    assert all(
        [row[c] for c in _CAMERA] == ["1.000000", "0.000000", "0.000000", "0.000000"]
        for row in rows
    )
    assert diag["task_start_frame"] == "0"
    assert diag["task_end_frame"] == "12"
    assert diag["frames_outside_task"] == "0"
    assert diag["camera_reference_frame"] == ""
    assert int(diag["camera_steps_unmeasured"]) == 11


@pytest.mark.parametrize("limit", [1, 5, 6, 11])
def test_p07_two_passes_obey_identical_max_frames_limits(tmp_path, monkeypatch, limit):
    _, rows, diag, tracker, captures, _ = _process(
        tmp_path, monkeypatch, camera_survey=True, max_frames=limit
    )
    assert len(captures) == 2
    assert [cap.index for cap in captures] == [limit, limit + 1]
    assert tracker.detected == list(range(0, limit, 5))
    assert tracker.posed == list(range(limit))
    assert len(rows) == limit
    assert int(diag["task_end_frame"]) == limit
    assert int(diag["camera_steps_unmeasured"]) == max(0, limit - 1)


def test_p07_malformed_read_consumes_index_but_not_max_frames_budget(tmp_path, monkeypatch):
    frames = [np.full((120, 160, 3), i, np.uint8) if i not in (1, 4) else None for i in range(12)]
    _, rows, diag, tracker, captures, _ = _process(
        tmp_path, monkeypatch, frames=frames, camera_survey=True, max_frames=6
    )
    assert len(captures) == 2
    assert [cap.index for cap in captures] == [8, 9]
    assert tracker.detected == [0, 5]
    assert tracker.posed == [0, 2, 3, 5, 6, 7]
    assert [int(row["frame_idx"]) for row in rows] == tracker.posed
    assert int(diag["task_end_frame"]) == 8


def test_p07_live_camera_is_never_surveyed(tmp_path, monkeypatch):
    _, rows, diag, tracker, captures, _ = _process(
        tmp_path, monkeypatch, source="0", camera_survey=True
    )
    assert len(captures) == 1
    assert tracker.detected == []
    assert tracker.posed == list(range(12))
    assert len(rows) == 12
    assert diag["camera_reference_frame"] == ""
    assert diag["frames_outside_task"] == "0"


def _steps(n):
    return np.repeat(np.eye(3)[None], n, axis=0)


def _complex_motion(transform):
    return complex(transform[0, 0], transform[1, 0]), complex(transform[0, 2], transform[1, 2])


def _motion_oracle(steps, measured, width, height, fps):
    result = []
    centre = complex(width / 2, height / 2)
    for t in range(len(steps) - 1):
        a, b = 1 + 0j, 0j
        valid = True
        for j in range(t + 1, min(len(steps), t + max(1, round(fps)) + 1)):
            c, d = _complex_motion(steps[j])
            a, b = c * a, c * b + d
            valid &= bool(measured[j])
        result.append(
            abs(a * centre + b - centre) / max(width, height)
            + abs(math.degrees(math.atan2(a.imag, a.real))) / 100
            + abs(math.log(abs(a)))
            if valid
            else math.inf
        )
    if len(steps):
        result.append(result[-1] if result else 0.0)
    return np.asarray(result)


def _transform_oracle(steps, reference):
    if reference is None:
        return _steps(len(steps))
    absolute = [(1 + 0j, 0j)]
    for step in steps[1:]:
        c, d = _complex_motion(step)
        a, b = absolute[-1]
        absolute.append((c * a, c * b + d))
    a_ref, b_ref = absolute[reference]
    result = []
    for a, b in absolute:
        c = a_ref / a
        d = b_ref - c * b
        result.append([[c.real, -c.imag, d.real], [c.imag, c.real, d.imag], [0, 0, 1]])
    return np.asarray(result)


def test_p01_frozen_camera_constants():
    camera = _camera()
    expected = {
        "GMC_WORK": 640,
        "ORB_WORK": 480,
        "SAMPLE_EVERY": 5,
        "PERSON_AREA": 0.01,
        "SETTLED": 0.05,
        "PERSON_SHARE": 0.6,
        "ANCHOR_INLIERS": 15,
        "T_MAX": 0.25,
        "R_MAX": 20,
        "S_MAX": 0.35,
        "TASK_GAP_S": 2.5,
    }
    for key, value in expected.items():
        assert getattr(camera, key) == value


@pytest.mark.parametrize("seed", [102, 2102, 202610])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_p01_similarity_recovery_with_independently_moving_foreground(seed, scale):
    camera = _camera()
    rng = np.random.default_rng(seed)
    height, width = 300, 400
    previous = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (3, 3), 0)
    transform = _similarity(
        degrees=1.0 if seed % 2 else -0.75,
        scale=1.004,
        tx=3.0,
        ty=-2.0,
        centre=(width / 2, height / 2),
    )
    current = cv2.warpAffine(
        previous,
        transform[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    # Independent foreground = 8% area; no person mask enters this estimator.
    patch = rng.integers(0, 256, (80, 120), dtype=np.uint8)
    previous[100:180, 160:280] = patch
    current[110:190, 172:292] = patch
    recovered = camera.estimate_step(previous, current, scale=scale)
    assert recovered is not None
    assert recovered.shape == (3, 3)
    expected = transform.copy()
    expected[:2, 2] *= scale
    corners = np.array(
        [
            [0, 0, 1],
            [(width - 1) * scale, 0, 1],
            [0, (height - 1) * scale, 1],
            [(width - 1) * scale, (height - 1) * scale, 1],
        ]
    )
    errors = np.linalg.norm((corners @ recovered.T - corners @ expected.T)[:, :2], axis=1)
    assert len(errors) == 4
    assert float(errors.max()) < 0.5


@pytest.mark.parametrize("values", [(0, 0), (127, 127), (0, 255)])
def test_p01_textureless_frames_are_unmeasured(values):
    previous, current = [np.full((160, 240), value, np.uint8) for value in values]
    assert _camera().estimate_step(previous, current) is None


@pytest.mark.parametrize("fps", [0.4, 1.0, 2.5, 3.5, 10.0])
def test_p02_generated_net_motion_matches_complex_similarity_oracle(fps):
    camera = _camera()
    rng = np.random.default_rng(210202)
    for n in (0, 1, 2, 7, 25):
        steps = _steps(n)
        measured = np.ones(n, dtype=bool)
        if n:
            measured[0] = False
        for t in range(1, n):
            steps[t] = _similarity(
                degrees=float(rng.uniform(-8, 8)),
                scale=float(np.exp(rng.uniform(-0.05, 0.05))),
                tx=float(rng.uniform(-25, 25)),
                ty=float(rng.uniform(-25, 25)),
            )
            measured[t] = rng.random() > 0.2
        before = steps.copy(), measured.copy()
        actual = camera.net_motion(steps, measured, 640, 480, fps)
        assert actual.shape == (n,)
        np.testing.assert_allclose(
            actual, _motion_oracle(steps, measured, 640, 480, fps), atol=1e-12, rtol=1e-12
        )
        np.testing.assert_array_equal(steps, before[0])
        np.testing.assert_array_equal(measured, before[1])


@pytest.mark.parametrize(
    ("kind", "expected"), [("translation", 0.125), ("rotation", 0.12), ("scale", 0.2)]
)
def test_p02_motion_component_weights(kind, expected):
    steps = _steps(2)
    if kind == "translation":
        steps[1] = _similarity(tx=80)
    elif kind == "rotation":
        steps[1] = _similarity(degrees=12, centre=(320, 240))
    else:
        steps[1] = _similarity(scale=math.exp(0.2), centre=(320, 240))
    result = _camera().net_motion(steps, np.array([False, True]), 640, 480, 1)
    np.testing.assert_allclose(result, [expected, expected], atol=1e-12, rtol=0)


def test_p02_horizon_order_unmeasured_and_tail():
    camera = _camera()
    steps = _steps(5)
    steps[1] = _similarity(tx=20)
    steps[2] = _similarity(degrees=90)
    steps[3] = _similarity(tx=800)
    measured = np.array([False, True, True, False, True])
    actual = camera.net_motion(steps, measured, 640, 480, 2)
    expected = _motion_oracle(steps, measured, 640, 480, 2)
    assert math.isfinite(actual[0])
    assert np.isinf(actual[1:3]).all()
    assert actual[3] == actual[4] == 0
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def _reference_oracle(motion, person, fps):
    placements = []
    for start in range(len(motion)):
        if motion[start] >= 0.05 or (start and motion[start - 1] < 0.05):
            continue
        end = start
        while end < len(motion) and motion[end] < 0.05:
            end += 1
        samples = [f for f in person if start <= f < end]
        if end - start >= fps and samples and sum(person[f] for f in samples) / len(samples) >= 0.6:
            placements.append((start, end, samples))
    if not placements:
        return None
    start, end, samples = min(placements, key=lambda r: (-(r[1] - r[0]), r[0]))
    return min(samples, key=lambda f: (abs(f - (start + end) / 2), f))


def test_p03_longest_qualifying_placement_and_earlier_ties():
    camera = _camera()
    motion = np.full(80, math.inf)
    motion[0:15] = motion[20:40] = motion[50:70] = 0
    person = dict.fromkeys(range(0, 80, 5), True)
    assert camera.choose_reference(motion, person, 10) == 30
    for i in (20, 25, 30):
        person[i] = False
    assert camera.choose_reference(motion, person, 10) == 60


@pytest.mark.parametrize(
    ("length", "motion_value", "present", "expected"),
    [
        (25, 0.0, 3, 10),
        (25, 0.0, 2, None),
        (25, 0.05, 5, None),
        (25, np.nextafter(0.05, 0), 5, 10),
        (9, 0.0, 5, None),
        (10, 0.0, 5, 5),
    ],
)
def test_p03_placement_duration_settled_and_person_share_boundaries(
    length, motion_value, present, expected
):
    person = {i: j < present for j, i in enumerate(range(0, length, 5))}
    assert _camera().choose_reference(np.full(length, motion_value), person, 10) == expected


def test_p03_midpoint_sample_tie_and_sampled_absent_person_reference():
    camera = _camera()
    assert camera.choose_reference(np.zeros(15), {0: True, 5: True, 10: True}, 5) == 5
    assert camera.choose_reference(np.zeros(11), {0: True, 5: False, 10: True}, 5) == 5
    assert camera.choose_reference(np.zeros(20), {}, 5) is None
    assert camera.choose_reference(np.full(20, math.inf), {0: True, 5: True}, 5) is None
    assert camera.choose_reference(np.zeros(0), {}, 5) is None


def test_p03_generated_placement_choice_matches_enumerated_oracle():
    camera = _camera()
    rng = np.random.default_rng(210203)
    for _ in range(80):
        n = int(rng.integers(1, 160))
        motion = np.where(rng.random(n) < 0.92, rng.uniform(0, 0.049, n), math.inf)
        person = {i: bool(rng.random() > 0.3) for i in range(0, n, 5)}
        fps = float(rng.choice([1, 5, 10, 29.97]))
        assert camera.choose_reference(motion, person, fps) == _reference_oracle(
            motion, person, fps
        )


@pytest.mark.parametrize("axis", ["translation", "rotation", "log-scale"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_p04_bounds_are_strict_on_each_motion_component(axis, sign):
    camera = _camera()
    for offset, expected in ((-1e-7, True), (0.0, False), (1e-7, False)):
        if axis == "translation":
            transform = _similarity(tx=sign * (0.25 + offset) * 640)
        elif axis == "rotation":
            transform = _similarity(degrees=sign * (20 + offset), centre=(320, 240))
        else:
            scale = math.exp(sign * (0.35 + offset))
            if offset == 0 and sign == 1:
                scale = np.nextafter(scale, math.inf)  # exp/log round-trip falls one ULP inside.
            transform = _similarity(scale=scale, centre=(320, 240))
        assert bool(camera.in_bounds(transform, 640, 480)) is expected, (axis, sign, offset)


def _bounds_oracle(transform, width, height):
    a, b = _complex_motion(transform)
    centre = complex(width / 2, height / 2)
    return (
        abs(a * centre + b - centre) / max(width, height) < 0.25
        and abs(math.degrees(math.atan2(a.imag, a.real))) < 20
        and abs(math.log(abs(a))) < 0.35
    )


def _view_oracle(steps, anchors, width, height):
    view = []
    for frame in range(len(steps)):
        candidates = []
        before = [f for f in anchors if f <= frame]
        after = [f for f in anchors if f >= frame]
        for anchor in ([max(before)] if before else []) + ([min(after)] if after else []):
            # Absolute frame transforms provide an independent reference to either nearest anchor.
            transform = anchors[anchor] @ _transform_oracle(steps, anchor)[frame]
            candidates.append(_bounds_oracle(transform, width, height))
        view.append(any(candidates))
    return np.array(view, dtype=bool)


def test_p04_anchor_reset_either_pass_and_missing_steps_as_identity():
    camera = _camera()
    steps = _steps(7)
    steps[1] = _similarity(tx=90)
    steps[2] = _similarity(tx=10)
    steps[3] = _similarity(tx=90)
    steps[4] = _similarity(tx=90)
    anchors = {0: np.eye(3), 2: np.eye(3), 6: np.eye(3)}
    actual = camera.in_view(steps, anchors, 100, 80)
    expected = _view_oracle(steps, anchors, 100, 80)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, expected)
    assert actual[1], "Backward pass alone recovers frame 1"
    assert not actual[3], "Both nearest anchors put frame 3 out of bounds"
    assert actual[4:7].all(), "Anchor resets backward carry across identity steps"
    np.testing.assert_array_equal(camera.in_view(steps, {}, 100, 80), np.zeros(7, dtype=bool))


def test_p04_generated_in_view_matches_independent_nearest_anchor_oracle():
    camera = _camera()
    rng = np.random.default_rng(210204)
    for n in (1, 2, 10, 35):
        steps = _steps(n)
        for i in range(1, n):
            if rng.random() < 0.2:
                continue  # An unmeasured step is stored as identity.
            steps[i] = _similarity(
                degrees=float(rng.uniform(-12, 12)),
                scale=float(np.exp(rng.uniform(-0.15, 0.15))),
                tx=float(rng.uniform(-50, 50)),
                ty=float(rng.uniform(-40, 40)),
            )
        anchors = {i: _similarity(tx=float(rng.uniform(-10, 10))) for i in range(0, n, 4)}
        actual = camera.in_view(steps, anchors, 160, 120)
        np.testing.assert_array_equal(actual, _view_oracle(steps, anchors, 160, 120))


@pytest.mark.parametrize(
    "mask",
    [[], [False], [True], [True, True, False, True], [False, True, False, True, True, False]],
)
def test_p04_runs_are_half_open_maximal_and_total(mask):
    actual = _camera().runs(np.array(mask, dtype=bool))
    covered = [i for start, end in actual for i in range(start, end)]
    assert covered == [i for i, value in enumerate(mask) if value]
    assert all(
        start < end and (start == 0 or not mask[start - 1]) and (end == len(mask) or not mask[end])
        for start, end in actual
    )
    if any(mask):
        assert actual


def _span_oracle(mask, reference, fps, gap_s):
    active = {i for i, value in enumerate(mask) if value}
    if reference not in active:
        return 0, len(mask)
    connected = {reference}
    radius = round(gap_s * fps) + 1
    while True:
        grown = connected | {i for i in active if any(abs(i - j) <= radius for j in connected)}
        if grown == connected:
            return min(grown), max(grown) + 1
        connected = grown


@pytest.mark.parametrize(("gap", "expected"), [(5, (0, 11)), (6, (0, 3))])
def test_p04_nc2_gap_at_rounded_limit_merges_one_more_does_not(gap, expected):
    mask = np.zeros(3 + gap + 3, dtype=bool)
    mask[:3] = mask[3 + gap :] = True
    assert _camera().task_span(mask, 1, 2, gap_s=2.5) == expected


def test_p04_repeated_merging_reaches_both_sides_without_concatenating_time():
    mask = np.zeros(27, dtype=bool)
    for a, b in ((1, 4), (6, 9), (11, 14), (16, 19), (22, 25)):
        mask[a:b] = True
    assert _camera().task_span(mask, 12, 4, gap_s=0.5) == (1, 19)


@pytest.mark.parametrize(("fps", "gap_s"), [(2, 1.25), (2, 1.75), (10, 0), (29.97, 0.1)])
def test_p04_generated_task_span_matches_gap_connectivity_oracle(fps, gap_s):
    camera = _camera()
    rng = np.random.default_rng(21020404)
    for _ in range(60):
        n = int(rng.integers(1, 80))
        mask = rng.random(n) > 0.6
        reference = int(rng.integers(n))
        assert camera.task_span(mask, reference, fps, gap_s=gap_s) == _span_oracle(
            mask, reference, fps, gap_s
        )
        assert camera.task_span(mask, None, fps, gap_s=gap_s) == (0, n)
    assert camera.task_span(np.array([], dtype=bool), None, fps, gap_s=gap_s) == (0, 0)


@pytest.mark.parametrize("reference", [None, 0, 7, 19])
def test_p05_generated_compensation_matches_absolute_trajectory_oracle(reference):
    camera = _camera()
    rng = np.random.default_rng(21020505)
    steps = _steps(20)
    for i in range(1, 20):
        if i % 6:
            steps[i] = _similarity(
                degrees=float(rng.uniform(-10, 10)),
                scale=float(np.exp(rng.uniform(-0.1, 0.1))),
                tx=float(rng.uniform(-40, 40)),
                ty=float(rng.uniform(-40, 40)),
            )
    before = steps.copy()
    actual = camera.to_reference(steps, reference)
    assert actual.shape == (20, 3, 3)
    np.testing.assert_allclose(actual, _transform_oracle(steps, reference), atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(steps, before)
    if reference is not None:
        np.testing.assert_allclose(actual[reference], np.eye(3), atol=1e-12)
        for t in range(1, len(steps)):
            np.testing.assert_allclose(actual[t] @ steps[t], actual[t - 1], atol=1e-10, rtol=1e-12)
    else:
        np.testing.assert_array_equal(actual, _steps(20))
    assert camera.to_reference(_steps(0), None).shape == (0, 3, 3)


def test_p05_nc3_registration_anchors_never_reset_exported_camera_transform(monkeypatch):
    camera = _camera()
    rng = np.random.default_rng(21020503)
    texture = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    frames = [texture.copy() for _ in range(30)]
    calls = []
    monkeypatch.setattr(camera, "estimate_step", lambda *_a, **_k: _similarity(tx=0.25))

    def register(*_args, **_kwargs):
        calls.append(True)
        return _similarity(tx=10), 30

    monkeypatch.setattr(camera, "register", register)
    survey = camera.survey_source(_Capture(frames), lambda _frame: np.array([[10, 10, 100, 100]]))
    assert survey is not None
    assert survey.n_frames == 30
    assert survey.reference == 15
    assert len(calls) >= 2, "NC3 requires genuine non-reference anchors"
    expected_steps = _steps(30)
    expected_steps[1:, 0, 2] = 0.25
    np.testing.assert_allclose(
        survey.to_reference, _transform_oracle(expected_steps, 15), atol=1e-12
    )
    assert not np.isclose(survey.to_reference[0, 0, 2], 10), (
        "Anchor and compensation must disagree in the seed"
    )


def test_p07_camera_survey_value_and_measured_speed_population():
    camera = _camera()
    steps = _steps(6)
    steps[1] = _similarity(tx=8, ty=6)
    steps[2] = _similarity(degrees=15, centre=(80, 60))
    steps[3] = _similarity(tx=100)
    steps[4] = _similarity(tx=-16)
    steps[5] = _similarity(scale=1.1, centre=(80, 60))
    measured = np.array([False, True, True, False, True, True])
    survey = camera.CameraSurvey(
        160, 120, 10.0, steps, measured, 2, (1, 6), camera.to_reference(steps, 2)
    )
    assert survey.n_frames == 6
    np.testing.assert_allclose(survey.speeds(0, 6), [0.625, 0, 1.0, 0], atol=1e-12)
    np.testing.assert_allclose(survey.speeds(2, 5), [0, 1.0], atol=1e-12)
    assert len(survey.speeds(0, 1)) == 0
    with pytest.raises(AttributeError):
        survey.reference = 1


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_p04_orb_registration_similarity_and_outlier_rejection(scale):
    camera = _camera()
    rng = np.random.default_rng(21020404)
    points = rng.uniform([0, 0], [480, 360], (40, 2)).astype(np.float32) * scale
    transform = _similarity(degrees=8, scale=1.05, tx=13 * scale, ty=-9 * scale)
    reference = (np.column_stack([points, np.ones(len(points))]) @ transform.T)[:, :2].astype(
        np.float32
    )
    reference[-10:] = rng.uniform([0, 0], [480, 360], (10, 2)) * scale
    descriptors = rng.integers(0, 256, (40, 32), dtype=np.uint8)
    actual, inliers = camera.register((points, descriptors), (reference, descriptors.copy()), scale)
    assert actual is not None
    assert inliers >= 30
    np.testing.assert_allclose(actual, transform, atol=5e-4, rtol=0)


@pytest.mark.parametrize("condition", ["missing", "nine-matches", "ambiguous-ratio"])
def test_p04_registration_refuses_missing_or_insufficient_unambiguous_matches(condition):
    camera = _camera()
    rng = np.random.default_rng(204)
    n = 9 if condition == "nine-matches" else 20
    points = rng.uniform(0, 400, (n, 2)).astype(np.float32)
    descriptors = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    other = descriptors.copy()
    if condition == "missing":
        other = None
    elif condition == "ambiguous-ratio":
        descriptors[:] = 0
        other[:] = 0
    transform, inliers = camera.register((points, descriptors), (points.copy(), other), 1.0)
    assert transform is None
    assert inliers < 15


@pytest.mark.parametrize(("area_fraction", "expected"), [(0.0099, None), (0.01, 15), (0.0101, 15)])
def test_p03_survey_person_area_boundary_and_sampling(monkeypatch, area_fraction, expected):
    camera = _camera()
    monkeypatch.setattr(camera, "estimate_step", lambda *_a, **_k: np.eye(3))
    frames = [np.full((120, 160, 3), i, np.uint8) for i in range(30)]
    sampled = []

    def detector(frame):
        sampled.append(int(frame[0, 0, 0]))
        return np.array([[0.0, 0.0, 160.0, 120.0 * area_fraction]])

    survey = camera.survey_source(_Capture(frames), detector)
    assert survey is not None
    assert survey.reference == expected
    assert sampled == [0, 5, 10, 15, 20, 25]
    assert survey.n_frames == 30
    np.testing.assert_array_equal(survey.steps[0], np.eye(3))
    assert not survey.measured[0]
    assert survey.measured[1:].all()


def test_p07_empty_survey_returns_none_without_detector_call():
    def detector(_frame):
        pytest.fail("Empty source must not reach a detector")

    assert _camera().survey_source(_Capture([]), detector) is None


class _Stateful:
    def __init__(self, kind):
        self.kind, self.resets, self.calls = kind, 0, 0
        self.hand_frames_present = self.hand_frames_gated = 0

    def reset(self):
        self.resets += 1

    def __call__(self, *args, **_kwargs):
        self.calls += 1
        return args[0] if self.kind == "gate" else args[:2]

    def output_track_keys(self):
        return [0]

    def live_track_keys(self):
        return [0]

    def prune(self, _keys):
        pass

    def update(self, _key, points, **_kwargs):
        self.calls += 1
        return points, 0.0


def test_p06_p07_trimmed_span_only_pose_rows_resets_and_exact_diagnostics(tmp_path, monkeypatch):
    camera = _camera()
    run = importlib.import_module("pose_estimation.run")
    steps = _steps(12)
    measured = np.ones(12, dtype=bool)
    measured[[0, 5]] = False
    for t in range(1, 12):
        if measured[t]:
            steps[t] = _similarity(tx=t)
    transforms = _transform_oracle(steps, 4)
    survey = camera.CameraSurvey(160, 120, 10.0, steps, measured, 4, (2, 8), transforms)
    calls = []

    def survey_source(capture, detector=None, *, max_frames=0, gap_s=2.5):
        calls.append((detector, max_frames, gap_s))
        while capture.read()[0]:
            pass
        return survey

    original = camera.survey_source
    monkeypatch.setattr(camera, "survey_source", survey_source)
    for name, value in vars(run).copy().items():
        if value is original:
            monkeypatch.setattr(run, name, survey_source)
    smoother, bones, gate = (_Stateful(kind) for kind in ("smoother", "bones", "gate"))
    fields, rows, diag, tracker, captures, _ = _process(
        tmp_path,
        monkeypatch,
        camera_survey=True,
        task_gap_s=0.75,
        smoother=smoother,
        bone_smoother=bones,
        hand_gate=gate,
    )
    assert len(calls) == 1
    assert calls[0] == (tracker.det_model, 0, 0.75)
    assert len(captures) == 2
    assert tracker.posed == list(range(2, 8))
    assert tracker.resets >= 1
    assert all(state.resets >= 1 for state in (smoother, bones, gate))
    assert all(state.calls == 6 for state in (smoother, bones, gate))
    assert fields[-4:] == list(_CAMERA)
    assert len(fields) == 308
    assert len(rows) == 6
    assert [int(row["frame_idx"]) for row in rows] == list(range(2, 8))
    for row, index in zip(rows, range(2, 8), strict=True):
        transform = transforms[index]
        expected = [transform[0, 0], transform[1, 0], transform[0, 2] / 160, transform[1, 2] / 160]
        assert [row[key] for key in _CAMERA] == [f"{value:.6f}" for value in expected]
    expected_diag = {
        "task_start_frame": "2",
        "task_end_frame": "8",
        "frames_outside_task": "6",
        "camera_reference_frame": "4",
        "camera_steps_unmeasured": "1",
        "camera_speed_p50": "0.250000",
        "camera_speed_p90": "0.412500",
    }
    assert {key: diag[key] for key in _DIAGNOSTICS} == expected_diag


@pytest.mark.parametrize("setting", _SETTINGS)
def test_p09_missing_setting_in_completed_marker_makes_event_due(setting, tmp_path, monkeypatch):
    driver, _, generated, event_out = _generation("corpus_run_2d.py", tmp_path, monkeypatch)
    marker = event_out / "pose_config.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload.pop(setting, None)
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert driver["main"]() == 0
    assert len(generated) == 2, f"Missing {setting} incorrectly reused the old generation"
    assert setting in json.loads(marker.read_text(encoding="utf-8"))


def test_p08_r_missing_coordinates_propagate_and_unpaired_columns_stay_raw():
    _r("""
    d <- data.frame(body_left_wrist_x=c(NA,1,2), body_left_wrist_y=c(4,NA,5),
                    body_left_wrist_z=c(11,12,13), unpaired_x=c(7,8,9),
                    cam_a=c(2,2,2),cam_b=c(3,3,3),cam_tx=c(4,4,NA),cam_ty=c(5,5,5))
    out <- stabilize_camera(d)
    stopifnot(all(is.na(out$body_left_wrist_x[1:2])), all(is.na(out$body_left_wrist_y[1:2])))
    stopifnot(identical(out[3,],d[3,]),identical(out$unpaired_x,d$unpaired_x),
              identical(out$body_left_wrist_z,d$body_left_wrist_z))
    """)


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize(
    "statistics", [((2, 0, None), (9, 5, 2)), ((4, 1, 0), (8, 3, 0), (5, 2, 1), (12, 8, 2))]
)
def test_p09_task_span_sums_change_with_diagnostic_population(
    name, statistics, tmp_path, monkeypatch
):
    _, args, _, _ = _generation(name, tmp_path, monkeypatch, statistics=statistics)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert statistics
    assert report.get("task_span") == {
        "frames_decoded": sum(row[0] for row in statistics),
        "frames_outside_task": sum(row[1] for row in statistics),
        "assets_trimmed": sum(row[1] > 0 for row in statistics),
        "assets_without_reference": sum(row[2] is None for row in statistics),
    }
