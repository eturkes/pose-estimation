"""Reviewer-2: boundary witnesses against M2.10.2 D01/D09/D10/D11."""

from __future__ import annotations

import csv
import importlib.util
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pose_estimation import camera_survey as camera
from pose_estimation import export, run, video_io

ROOT = Path(__file__).resolve().parents[1]


class Capture:
    def __init__(self, frames, fps=30.0):
        self.frames, self.fps, self.index = frames, fps, 0
        self.released = False

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 120,
            cv2.CAP_PROP_POS_MSEC: max(0, self.index - 1) * 1000 / 30,
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


class Tracker:
    def reset(self):
        pass

    def __call__(self, frame):
        return np.full((1, 133, 2), 20.0), np.full((1, 133), 0.9)

    def det_model(self, frame):
        return np.array([[10.0, 10.0, 90.0, 90.0]])


def _frames(n):
    return [np.full((120, 160, 3), i, np.uint8) for i in range(n)]


def test_b06_missing_previous_frame_cannot_produce_measured_step():
    """B06: D01/D09 require consecutive decoded frames, not last valid frame -> current."""
    rng = np.random.default_rng(210206)
    first = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    last = cv2.warpAffine(first, np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0]]), (640, 480))
    # Positive control: the same two frames, actually adjacent, have a measured step.
    adjacent = camera.survey_source(Capture([first, last]))
    assert adjacent is not None
    assert adjacent.measured[1]
    assert adjacent.steps[1, 0, 2] == pytest.approx(4.0, abs=0.1)
    survey = camera.survey_source(Capture([first, None, last]))
    assert survey is not None
    assert survey.n_frames == 3
    assert not survey.measured[1]
    assert not survey.measured[2], (
        "frame 1 is missing, but step[2] is marked measured from frame 0 -> 2; "
        f"tx={survey.steps[2, 0, 2]:.6f}"
    )
    np.testing.assert_array_equal(survey.steps[2], np.eye(3))


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), -1.0, 0.5, 500.0])
def test_b09_survey_and_pose_share_fps_sanitization(fps):
    """B09: both passes must use the shipped safe nominal FPS, including malformed readings."""
    survey = camera.survey_source(Capture(_frames(2), fps=fps))
    assert survey is not None
    assert survey.fps == video_io.safe_fps(fps), "survey and pose disagree on one-second/gap units"


def test_b07_task_report_counts_failed_assets_with_diagnostics(tmp_path):
    """B07: D11 sums over assets carrying diagnostics, including a failed clinical stage."""
    driver = runpy.run_path(str(ROOT / "scripts/corpus_run_2d.py"), run_name="_reviewer2")
    event = tmp_path / "synthetic-event"
    event.mkdir()
    (event / "cam-a.csv").write_text("video,frame_idx,person_idx\nsynthetic-event/cam-a,0,0\n")
    diag = dict.fromkeys(run.SOURCE_DIAGNOSTIC_FIELDS, "0")
    diag.update(
        video="synthetic-event/cam-a",
        n_frames_decoded="10",
        pts_accepted="10",
        task_start_frame="2",
        task_end_frame="8",
        frames_outside_task="4",
        camera_reference_frame="5",
    )
    with (event / "cam-a_diag.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run.SOURCE_DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerow(diag)
    row = {
        "asset_id": "synthetic-asset",
        "event_id": event.name,
        "camera_name": "cam-a",
        "disposition": "ok",
    }
    placed = {row["asset_id"]: SimpleNamespace(event_id=event.name, camera_name="cam-a")}
    expected = {
        "frames_decoded": 10,
        "frames_outside_task": 4,
        "assets_trimmed": 1,
        "assets_without_reference": 0,
    }
    # Positive control reaches the identical functions and identical diagnostic bytes.
    # A03: the fix keeps D11's population in its own `task_counters`, apart from the ok-only CFR one.
    before = driver["_artifacts"]([row], placed, tmp_path)
    assert driver["_task_span"](before["task_counters"]) == expected
    row["disposition"] = "clinical_failed"
    after = driver["_artifacts"]([row], placed, tmp_path)
    assert driver["_task_span"](after["task_counters"]) == expected


def test_b07_interrupted_pose_diagnostic_counts_decoded_prefix(tmp_path, monkeypatch):
    """B07: D10 outside count must describe the decoded prefix, not unseen second-pass frames."""
    steps = np.repeat(np.eye(3)[None], 20, axis=0)
    survey = camera.CameraSurvey(160, 120, 30, steps, np.zeros(20, bool), 10, (8, 12), steps.copy())
    monkeypatch.setattr(run, "survey_source", lambda *_a, **_kw: survey)
    captures = [Capture(_frames(20)), Capture(_frames(5))]
    monkeypatch.setattr(run, "open_capture", lambda *_a, **_kw: captures.pop(0))
    path = tmp_path / "diag.csv"
    run.process_source(
        run.parse_args(["--headless", "--tracking", "body"]),
        Tracker(),
        str(tmp_path / "synthetic.avi"),
        None,
        output_diag=path,
        camera_survey=True,
    )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert int(rows[0]["n_frames_decoded"]) == 5
    assert int(rows[0]["frames_outside_task"]) == 5, rows[0]


def test_b08_disabled_csv_matches_preunit_implementation(tmp_path, monkeypatch):
    """B08: byte comparison against actual e996025 run/export, not two calls to current code."""
    modules = {}
    for name in ("run", "export"):
        path = tmp_path / f"legacy_{name}.py"
        path.write_bytes(
            subprocess.check_output(
                ["git", "show", f"e996025:src/pose_estimation/{name}.py"], cwd=ROOT
            )
        )
        spec = importlib.util.spec_from_file_location(f"pose_estimation._reviewer2_{name}", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    old = modules["run"]
    monkeypatch.setattr(old, "frame_to_rows", modules["export"].frame_to_rows)
    monkeypatch.setattr(old, "open_csv_writer", modules["export"].open_csv_writer)
    outputs = []
    for label, module in (("old", old), ("new", run)):
        monkeypatch.setattr(module, "open_capture", lambda *_a, **_kw: Capture(_frames(7)))
        path = tmp_path / f"{label}.csv"
        module.process_source(
            module.parse_args(["--headless", "--tracking", "body"]),
            Tracker(),
            str(tmp_path / "synthetic.avi"),
            None,
            output_csv=path,
            video_name="synthetic",
        )
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
    assert len(export.make_csv_header("body")) == 304
