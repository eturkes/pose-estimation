"""M2.9.5 P01-P05: diff-blind body-drop oracle, pipeline and provenance witnesses."""

from __future__ import annotations

import csv
import importlib
import json
import runpy
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DRIVERS = ("corpus_run_2d.py", "pilot_corpus_run.py")
_FLAGS = tuple(product((False, True), repeat=2))
_FLAG_IDS = ("neither", "hips", "legs", "both")
_DROP_FIELDS = {"drop_lower_body", "drop_hips_camera"}
_BODY_PARTS = ("hip", "knee", "ankle", "heel", "foot_index")


def _hygiene():
    return importlib.import_module("pose_estimation.keypoint_hygiene")


def _drop_reference(scores, lower_body=False, hips=False):
    result = scores.copy()
    for person, row in enumerate(scores):
        for index in range(len(row)):
            if (lower_body and 13 <= index <= 22) or (hips and index in (11, 12)):
                result[person, index] = 0
    return result


def test_p01_public_index_sets_match_the_contract():
    hygiene = _hygiene()
    for name, expected in (("HIPS", [11, 12]), ("LOWER_BODY", list(range(13, 23)))):
        selection = getattr(hygiene, name)
        if not isinstance(selection, slice):
            selection = list(selection)
        np.testing.assert_array_equal(np.arange(133)[selection], expected)


@pytest.mark.parametrize(("lower_body", "hips"), _FLAGS, ids=_FLAG_IDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_p01_every_slice_boundary_empty_axis_and_readonly_stride(lower_body, hips, dtype):
    drop = _hygiene().drop_body_parts
    nonempty = affected = 0
    for people, count in product((0, 1, 3), (0, 1, 10, 11, 12, 13, 14, 16, 17, 22, 23, 24, 133)):
        backing = np.linspace(0.125, 0.875, people * count * 2, dtype=dtype).reshape(
            people, count * 2
        )
        scores = backing[:, ::2]
        before = scores.copy()
        scores.setflags(write=False)
        expected = _drop_reference(scores, lower_body, hips)
        actual = drop(scores, lower_body=lower_body, hips=hips)
        assert actual.shape == scores.shape
        np.testing.assert_array_equal(actual, expected)
        assert scores.tobytes() == before.tobytes()
        nonempty += int(scores.size > 0)
        affected += int(np.count_nonzero(before != expected))
    assert nonempty > 0
    assert affected > 0 if lower_body or hips else affected == 0


@pytest.mark.parametrize("seed", [95, 2095, 9533])
def test_p01_generated_scalar_oracle_and_projection_properties(seed):
    drop = _hygiene().drop_body_parts
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(100):
        people, count = int(rng.integers(1, 6)), int(rng.integers(1, 201))
        scores = rng.uniform(-1.0, 2.0, size=(people, count))
        scores.ravel()[::13] = np.nan
        scores.ravel()[::17] = np.inf
        scores.ravel()[::23] = -np.inf
        before = scores.copy()
        order = rng.permutation(people)
        for lower_body, hips in _FLAGS:
            actual = drop(scores, lower_body=lower_body, hips=hips)
            np.testing.assert_array_equal(actual, _drop_reference(scores, lower_body, hips))
            np.testing.assert_array_equal(drop(actual, lower_body=lower_body, hips=hips), actual)
            np.testing.assert_array_equal(
                drop(scores[order], lower_body=lower_body, hips=hips), actual[order]
            )
            assert scores.tobytes() == before.tobytes()
            checked += 1
        np.testing.assert_array_equal(
            drop(drop(scores, lower_body=True), hips=True),
            drop(scores, lower_body=True, hips=True),
        )
        np.testing.assert_array_equal(
            drop(drop(scores, hips=True), lower_body=True),
            drop(scores, lower_body=True, hips=True),
        )
    assert checked == 400


def test_p02_out_of_frame_drop_duplicate_order(monkeypatch):
    from test_m2u92_keypoint_hygiene import _pair, _reference, _zero_reference

    hygiene = _hygiene()
    drop = hygiene.drop_body_parts
    zero = hygiene.zero_out_of_frame
    duplicate = hygiene.suppress_duplicate_hand
    points, scores = _pair()
    points[0, 5] = [64.0, 20.0]
    before = points.copy(), scores.copy()
    calls = []

    def bounded(points, scores, width, height):
        calls.append(("out_of_frame", scores.copy()))
        return zero(points, scores, width, height)

    def dropped(scores, *, lower_body=False, hips=False):
        calls.append(("drop", scores.copy()))
        assert (lower_body, hips) == (True, True)
        return drop(scores, lower_body=lower_body, hips=hips)

    def suppressed(points, scores):
        calls.append(("duplicate", scores.copy()))
        return duplicate(points, scores)

    monkeypatch.setattr(hygiene, "zero_out_of_frame", bounded)
    monkeypatch.setattr(hygiene, "drop_body_parts", dropped)
    monkeypatch.setattr(hygiene, "suppress_duplicate_hand", suppressed)
    actual = hygiene.apply_hygiene(points, scores, 64, 48, drop_lower_body=True, drop_hips=True)
    assert [name for name, _scores in calls] == ["out_of_frame", "drop", "duplicate"]
    expected_bounded = _zero_reference(points, scores, 64, 48)
    np.testing.assert_array_equal(calls[1][1], expected_bounded)
    np.testing.assert_array_equal(calls[2][1], _drop_reference(expected_bounded, True, True))
    np.testing.assert_array_equal(
        actual, _drop_reference(_reference(points, scores, 64, 48), True, True)
    )
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(scores, before[1])


@pytest.mark.parametrize(("lower_body", "hips"), _FLAGS, ids=_FLAG_IDS)
def test_p02_generated_duplicate_verdict_coordinates_and_inputs_unchanged(lower_body, hips):
    from test_m2u92_keypoint_hygiene import _generated_pairs, _reference, _zero_reference

    hygiene = _hygiene()
    fired = kept = 0
    for points, scores in _generated_pairs(95, count=90):
        before = points.copy(), scores.copy()
        legacy = _reference(points, scores, 100, 100)
        bounded = _zero_reference(points, scores, 100, 100)
        changed = np.any(legacy[:, 91:] != bounded[:, 91:], axis=1)
        fired += int(np.count_nonzero(changed))
        kept += int(np.count_nonzero(~changed))
        actual = hygiene.apply_hygiene(
            points, scores, 100, 100, drop_lower_body=lower_body, drop_hips=hips
        )
        np.testing.assert_array_equal(actual, _drop_reference(legacy, lower_body, hips))
        np.testing.assert_array_equal(actual[:, 91:], legacy[:, 91:])
        np.testing.assert_array_equal(points, before[0])
        np.testing.assert_array_equal(scores, before[1])
    assert fired > 0
    assert kept > 0


@pytest.mark.parametrize("explicit", [False, True], ids=["omitted", "explicit-off"])
def test_p02_flags_off_equal_the_independent_m2u92_oracle(explicit):
    from test_m2u92_keypoint_hygiene import _generated_pairs, _reference

    options = {"drop_lower_body": False, "drop_hips": False} if explicit else {}
    seen = 0
    for points, scores in _generated_pairs(92, count=40):
        actual = _hygiene().apply_hygiene(points, scores, 100, 100, **options)
        np.testing.assert_array_equal(actual, _reference(points, scores, 100, 100))
        seen += len(points)
    assert seen > 0


@pytest.mark.parametrize(("lower_body", "hips"), _FLAGS, ids=_FLAG_IDS)
def test_p02_short_and_empty_rows_keep_shape_and_coordinates(lower_body, hips):
    for people, count in product((0, 2), (0, 11, 12, 13, 14, 17, 22, 23, 133)):
        points = np.full((people, count, 2), 20.0)
        scores = np.full((people, count), 0.875)
        before = points.copy(), scores.copy()
        actual = _hygiene().apply_hygiene(
            points, scores, 64, 48, drop_lower_body=lower_body, drop_hips=hips
        )
        assert actual.shape == scores.shape
        np.testing.assert_array_equal(actual, _drop_reference(scores, lower_body, hips))
        np.testing.assert_array_equal(points, before[0])
        np.testing.assert_array_equal(scores, before[1])


def _export(
    tmp_path,
    monkeypatch,
    outputs,
    *,
    source="synthetic.mp4",
    video_name=None,
    smoother=None,
    tracker_kind="stateful",
    **drop_options,
):
    from test_m2u92_keypoint_hygiene import _Capture, _Tracker

    run = importlib.import_module("pose_estimation.run")
    export = importlib.import_module("pose_estimation.export")
    capture = _Capture([(48, 64)] * len(outputs))
    tracker = _Tracker(outputs)
    before = [(points.copy(), scores.copy()) for points, scores in outputs]
    monkeypatch.setattr(run, "open_capture", lambda *_args, **_kwargs: capture)
    output = tmp_path / "synthetic-body-drop.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    run.process_source(
        SimpleNamespace(tracking="body", headless=True, single_subject=False, max_frames=0),
        tracker if tracker_kind == "stateful" else lambda frame: tracker(frame),
        source,
        draw_skeleton=None,
        smoother=smoother,
        output_csv=output,
        video_name=video_name,
        **drop_options,
    )
    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == export.make_csv_header("body")
        assert reader.fieldnames is not None
        assert len(reader.fieldnames) == 304
        rows = list(reader)
    assert len(rows) == sum(len(points) for points, _scores in outputs) > 0
    assert capture.released
    assert tracker.index == len(outputs)
    for (points, scores), (old_points, old_scores) in zip(outputs, before, strict=True):
        np.testing.assert_array_equal(points, old_points)
        np.testing.assert_array_equal(scores, old_scores)
    return rows


def _pose(count=133):
    return np.full((1, count, 2), 20.0), np.full((1, count), 0.875)


def _visibility_columns(*, lower_body, hips):
    return {
        f"body_{side}_{part}_vis"
        for side in ("left", "right")
        for part in _BODY_PARTS
        if (hips if part == "hip" else lower_body)
    }


_CAMERA_CASES = [
    ("event/cam-above", "synthetic.mp4", "above", True),
    ("event/cam-left", "synthetic.mp4", "above", False),
    ("above-event/cam-left", "synthetic.mp4", "above", False),
    ("nested/above-event/cam-right", "synthetic.mp4", "above", False),
    (None, "synthetic-cam-above.mp4", "above", True),
    (None, "above-directory/synthetic-cam-left.mp4", "above", False),
    ("cam-preabovepost.mp4", "synthetic.mp4", "bov", True),
    ("event/cam-above", "synthetic.mp4", None, False),
    ("event/cam-above", "synthetic.mp4", "", False),
    ("event/cam-ABOVE", "synthetic.mp4", "above", False),
    ("event/cam-上方", "synthetic.mp4", "上方", True),
    ("event/cam-above/", "synthetic.mp4", "above", False),
]


@pytest.mark.parametrize(
    ("label", "source", "token", "hips"),
    _CAMERA_CASES,
    ids=[
        "session-match",
        "side",
        "event-only",
        "nested-parent",
        "file-match",
        "file-parent",
        "substring",
        "no-token",
        "empty-token",
        "case-sensitive",
        "unicode",
        "trailing-slash",
    ],
)
def test_p03_camera_label_rule_and_exact_csv_mask(
    tmp_path, monkeypatch, label, source, token, hips
):
    outputs = [_pose()]
    baseline = _export(tmp_path / "baseline", monkeypatch, outputs, source=source, video_name=label)
    actual = _export(
        tmp_path / "dropped",
        monkeypatch,
        outputs,
        source=source,
        video_name=label,
        drop_lower_body=True,
        drop_hips_camera=token,
    )
    dropped = _visibility_columns(lower_body=True, hips=hips)
    assert len(dropped) == (10 if hips else 8)
    for before, after in zip(baseline, actual, strict=True):
        assert dropped <= before.keys()
        assert all(float(before[column]) > 0 for column in dropped)
        for column, value in before.items():
            assert float(after[column]) == 0 if column in dropped else after[column] == value


@pytest.mark.parametrize("tracker_kind", ["stateful", "callable"])
def test_p03_hip_only_keeps_legs_and_all_other_csv_values(tmp_path, monkeypatch, tracker_kind):
    outputs = [_pose()]
    baseline = _export(tmp_path / "baseline", monkeypatch, outputs, tracker_kind=tracker_kind)
    actual = _export(
        tmp_path / "dropped",
        monkeypatch,
        outputs,
        tracker_kind=tracker_kind,
        drop_lower_body=False,
        drop_hips_camera="synthetic",
    )
    dropped = _visibility_columns(lower_body=False, hips=True)
    assert len(dropped) == 2
    for column, value in baseline[0].items():
        assert float(actual[0][column]) == 0 if column in dropped else actual[0][column] == value


def test_p03_generated_camera_tokens_obey_only_last_component_membership(tmp_path, monkeypatch):
    rng = np.random.default_rng(95)
    matched = kept = 0
    for case in range(36):
        camera = "".join(rng.choice(list("abAB-_方"), size=12))
        token = (camera[3:7], "prefix-only", "", None, camera.upper(), "/")[case % 6]
        lower_body = bool(rng.integers(2))
        hips = bool(token) and token in camera
        rows = _export(
            tmp_path / str(case),
            monkeypatch,
            [_pose()],
            video_name=f"root/{token or 'above'}/{camera}",
            drop_lower_body=lower_body,
            drop_hips_camera=token,
        )
        dropped = _visibility_columns(lower_body=lower_body, hips=hips)
        for column in _visibility_columns(lower_body=True, hips=True):
            assert float(rows[0][column]) == (0.0 if column in dropped else 0.875)
        matched += int(hips)
        kept += int(not hips)
    assert matched > 0
    assert kept > 0


def test_p03_omitted_keyword_settings_keep_all_ten_body_visibilities(tmp_path, monkeypatch):
    rows = _export(tmp_path, monkeypatch, [_pose()], video_name="event/cam-above")
    columns = _visibility_columns(lower_body=True, hips=True)
    assert len(columns) == 10
    assert all(float(rows[0][column]) == 0.875 for column in columns)


def test_p03_drop_reaches_smoother_before_any_coordinate_filter(tmp_path, monkeypatch):
    from test_m2u92_keypoint_hygiene import _RecordingSmoother

    points, scores = _pose()
    smoother = _RecordingSmoother()
    _export(
        tmp_path,
        monkeypatch,
        [(points, scores)],
        smoother=smoother,
        video_name="event/cam-above",
        drop_lower_body=True,
        drop_hips_camera="above",
    )
    assert len(smoother.seen) == 1
    seen_points, seen_scores, _timestamp = smoother.seen[0]
    np.testing.assert_array_equal(seen_scores, _drop_reference(scores, True, True))
    np.testing.assert_array_equal(seen_points, points)


def test_p03_real_smoother_holds_dropped_positions_with_zero_csv_visibility(tmp_path, monkeypatch):
    from pose_estimation.rtmlib_smoothing import KeypointSmoother

    points, scores = _pose(17)
    moved = points.copy()
    moved[:, 11:17, 0] += 5.0

    class PrimedSmoother:
        def __init__(self):
            self.inner = KeypointSmoother(min_track_age=1)

        def reset(self):
            self.inner.reset()
            self.inner(points, scores, -1 / 30)

        def __call__(self, keypoints, confidence, timestamp, **kwargs):
            return self.inner(keypoints, confidence, timestamp, **kwargs)

    baseline = _export(
        tmp_path / "baseline",
        monkeypatch,
        [(moved, scores)],
        video_name="event/cam-above",
        smoother=PrimedSmoother(),
    )
    for part in ("hip", "knee", "ankle"):
        assert float(baseline[0][f"body_left_{part}_x"]) > 20 / 64
    actual = _export(
        tmp_path / "dropped",
        monkeypatch,
        [(moved, scores)],
        video_name="event/cam-above",
        smoother=PrimedSmoother(),
        drop_lower_body=True,
        drop_hips_camera="above",
    )
    for side, part in product(("left", "right"), ("hip", "knee", "ankle")):
        assert float(actual[0][f"body_{side}_{part}_x"]) == 20 / 64
        assert float(actual[0][f"body_{side}_{part}_vis"]) == 0
    assert float(actual[0]["body_left_shoulder_vis"]) == 0.875


@pytest.mark.parametrize(
    ("options", "lower_body", "token"),
    [
        ([], False, None),
        (["--drop-lower-body"], True, None),
        (["--drop-hips-camera", "above"], False, "above"),
        (["--drop-lower-body", "--drop-hips-camera", "cam-left"], True, "cam-left"),
        (["--drop-hips-camera", ""], False, ""),
    ],
    ids=["defaults", "legs", "hips", "both", "empty-token"],
)
def test_p04_main_forwards_through_real_session_dispatch(
    tmp_path, monkeypatch, options, lower_body, token
):
    from test_corpus_run_preconditions import _make_session
    from test_m2u91_subject_tracker import _solution_factory

    run = importlib.import_module("pose_estimation.run")
    session = _make_session(tmp_path, cameras=("cam-above", "cam-left"))
    calls = []
    monkeypatch.setattr(
        run, "SplitDeviceSolution", _solution_factory(lambda _index: np.empty((0, 4)))
    )
    monkeypatch.setattr(run, "resolve_cli_sessions", lambda *_args, **_kwargs: [session])

    def process(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(run, "process_source", process)
    run.main(
        [
            "--session-dir",
            str(session.directory),
            "--output-dir",
            str(tmp_path / "out"),
            "--headless",
            "--backend",
            "onnxruntime",
            *options,
        ]
    )
    assert len(calls) == 2
    assert {call["video_name"] for call in calls} == {
        f"{session.session_id}/{camera.name}" for camera in session.cameras
    }
    for call in calls:
        assert call["drop_lower_body"] is lower_body
        assert call["drop_hips_camera"] == token


def test_p04_parser_requires_a_camera_token(capsys):
    run = importlib.import_module("pose_estimation.run")
    with pytest.raises(SystemExit) as error:
        run.parse_args(["--drop-hips-camera"])
    assert error.value.code == 2
    assert "--drop-hips-camera: expected one argument" in capsys.readouterr().err


def _driver(name):
    return runpy.run_path(str(_ROOT / "scripts" / name), run_name="_m2u95_driver")[
        "main"
    ].__globals__


def _driver_args(driver, monkeypatch, root):
    from test_m2u91_subject_tracker import _driver_args as make_args

    return make_args(driver, monkeypatch, root)


def test_p05_pose_config_identity_includes_both_settings():
    from pose_estimation.corpus_run import POSE_CONFIG_FIELDS

    assert set(POSE_CONFIG_FIELDS) >= _DROP_FIELDS


@pytest.mark.parametrize("name", _DRIVERS)
def test_p05_driver_defaults_vocabulary_and_generator_version(name, monkeypatch):
    driver = _driver(name)
    monkeypatch.setattr(sys, "argv", [name])
    args = driver["_parse_args"]()
    assert (getattr(args, "drop_lower_body", None), getattr(args, "drop_hips_camera", None)) == (
        True,
        "above",
    )
    assert driver["REPORT_FIELDS"] >= _DROP_FIELDS
    # M2.9.5 introduced the drop fields at corpus v4 / pilot v3; later units bump past it.
    assert int(driver["GENERATOR_VERSION"].lstrip("v")) >= (4 if name == "corpus_run_2d.py" else 3)


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize(
    ("lower_body", "token"),
    [(True, "above"), (False, "cam-left"), (False, "")],
    ids=["enabled", "disabled-legs", "disabled-all"],
)
def test_p05_driver_parses_and_forwards_effective_settings(
    name, lower_body, token, tmp_path, monkeypatch
):
    run = importlib.import_module("pose_estimation.run")
    driver = _driver(name)
    args = _driver_args(driver, monkeypatch, tmp_path)
    args.drop_lower_body = lower_body
    args.drop_hips_camera = token
    commands = []
    monkeypatch.setattr(
        driver["subprocess"], "call", lambda command, **_kwargs: commands.append(command) or 0
    )
    if name == "corpus_run_2d.py":
        driver["_attempt_event"]("synthetic-event", args, tmp_path / "logs")
    else:
        driver["_run_event"](
            args.sessions / "synthetic-event", args.out, tmp_path / "run.log", args
        )
    pose_commands = [command for command in commands if "pose_estimation.run" in command]
    assert len(pose_commands) == 1
    command = pose_commands[0]
    parsed = run.parse_args(command[command.index("pose_estimation.run") + 1 :])
    assert parsed.drop_lower_body is lower_body
    assert (parsed.drop_hips_camera or "") == token
    monkeypatch.setattr(
        sys,
        "argv",
        [
            name,
            "--drop-lower-body" if lower_body else "--no-drop-lower-body",
            "--drop-hips-camera",
            token,
        ],
    )
    parsed = driver["_parse_args"]()
    assert (parsed.drop_lower_body, parsed.drop_hips_camera) == (lower_body, token)


def _csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _generation(name, tmp_path, monkeypatch, *, lower_body=False, token=""):
    driver = _driver(name)
    args = _driver_args(driver, monkeypatch, tmp_path)
    args.drop_lower_body, args.drop_hips_camera = lower_body, token
    asset_id, event_id, camera = "synthetic-asset", "synthetic-event", "cam-a"
    video = f"{event_id}/{camera}"
    _csv(
        args.inventory / "assets.csv",
        ("asset_id", "disposition", "reported_rotation_deg", "reported_frame_count"),
        [
            {
                "asset_id": asset_id,
                "disposition": "canonical",
                "reported_rotation_deg": "0",
                "reported_frame_count": "3",
            }
        ],
    )
    _csv(
        args.qualification / "assets_qc.csv",
        ("asset_id", "codec", "device_config", "pts_monotonic"),
        [
            {
                "asset_id": asset_id,
                "codec": "h264",
                "device_config": "synthetic",
                "pts_monotonic": "1",
            }
        ],
    )
    _csv(
        args.sessions / "placements.csv",
        ("asset_id", "event_id", "camera_name", "placement"),
        [
            {
                "asset_id": asset_id,
                "event_id": event_id,
                "camera_name": camera,
                "placement": "placed",
            }
        ],
    )
    (args.sessions / "generation.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(driver, "_parse_args", lambda: args)
    monkeypatch.setitem(driver, "validate_generation", lambda *_args, **_kwargs: {})
    if name == "pilot_corpus_run.py":
        args.min_assets = 1
        monkeypatch.setitem(driver, "_assert_sources_validated", lambda *_args: None)
        monkeypatch.setitem(
            driver,
            "_guard_verdicts",
            lambda *_args: {"events_probed": 1, "default_output_refused": 1},
        )
    pilot = driver["pilot"].__dict__ if name == "corpus_run_2d.py" else driver
    qc_header, _ = pilot["_r_constants"]()
    generated = []
    event_out = args.out / event_id

    def subprocess_call(command, **_kwargs):
        if "pose_estimation.run" in command:
            generated.append(command.copy())
            _csv(
                event_out / f"{camera}.csv",
                ("video", "person_idx", "frame_idx"),
                [{"video": video, "person_idx": "0", "frame_idx": "0"}],
            )
            diagnostic = {
                "video": video,
                "n_frames_decoded": "3",
                "pts_accepted": "3",
                "index_fallback": "0",
                "monotonic_forced": "0",
                "cfr_fallback_rate": "0",
                "latency_ms_mean": "1",
                "latency_ms_p95": "1",
            }
            _csv(event_out / f"{camera}_diag.csv", tuple(diagnostic), [diagnostic])
        else:
            assert command[0] == "Rscript"
            _csv(event_out / f"{camera}_clinical_group_qc.csv", qc_header, [])
            _csv(
                event_out / f"{camera}_clinical_windows.csv",
                ("video", "person_idx"),
                [{"video": video, "person_idx": "0"}],
            )
        return 0

    monkeypatch.setattr(driver["subprocess"], "call", subprocess_call)
    assert driver["main"]() == 0
    assert len(generated) == 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["population"]["events"] == 1
    assert all(report["verdicts"].values())
    return driver, args, generated, event_out


@pytest.mark.parametrize("name", _DRIVERS)
@pytest.mark.parametrize(
    ("lower_body", "token"),
    [(True, "above"), (False, "not-a-published-stratum"), (False, "")],
    ids=["defaults", "custom-token", "disabled-all"],
)
def test_p05_report_and_pose_marker_publish_the_drop_settings(
    name, lower_body, token, tmp_path, monkeypatch
):
    driver, args, _generated, event_out = _generation(
        name, tmp_path, monkeypatch, lower_body=lower_body, token=token
    )
    report = json.loads(args.report.read_text(encoding="utf-8"))
    marker = json.loads((event_out / "pose_config.json").read_text(encoding="utf-8"))
    expected = {"drop_lower_body": lower_body, "drop_hips_camera": token}
    assert {key: report["configuration"].get(key) for key in expected} == expected
    assert {key: marker.get(key) for key in expected} == expected
    assert driver["REPORT_FIELDS"] >= _DROP_FIELDS


@pytest.mark.parametrize("name", _DRIVERS)
def test_p05_same_settings_reuse_the_completed_generation(name, tmp_path, monkeypatch):
    driver, args, generated, event_out = _generation(name, tmp_path, monkeypatch)
    before = (event_out / "cam-a.csv").read_bytes()
    if name == "pilot_corpus_run.py":
        args.reuse_run = True
    assert driver["main"]() == 0
    assert len(generated) == 1
    assert (event_out / "cam-a.csv").read_bytes() == before


_SETTING_CHANGES = [
    ("drop_lower_body", False, True),
    ("drop_lower_body", True, False),
    ("drop_hips_camera", "", "above"),
    ("drop_hips_camera", "above", "cam-left"),
    ("drop_hips_camera", "above", ""),
]
_CHANGE_IDS = ["legs-on", "legs-off", "hips-on", "hips-other-camera", "hips-off"]


def _changed_generation(name, setting, old, new, tmp_path, monkeypatch):
    driver, args, generated, event_out = _generation(
        name,
        tmp_path,
        monkeypatch,
        lower_body=old if setting == "drop_lower_body" else False,
        token=old if setting == "drop_hips_camera" else "",
    )
    setattr(args, setting, new)
    return driver, args, generated, event_out


@pytest.mark.parametrize(("setting", "old", "new"), _SETTING_CHANGES, ids=_CHANGE_IDS)
def test_p05_changed_setting_makes_a_complete_event_due_on_resume(
    setting, old, new, tmp_path, monkeypatch
):
    driver, _args, generated, event_out = _changed_generation(
        "corpus_run_2d.py", setting, old, new, tmp_path, monkeypatch
    )
    assert driver["main"]() == 0
    assert len(generated) == 2, f"P05: complete event reused despite changed {setting}"
    marker = json.loads((event_out / "pose_config.json").read_text(encoding="utf-8"))
    assert marker[setting] == new


@pytest.mark.parametrize(
    ("name", "mode"), [("corpus_run_2d.py", "analyse_only"), ("pilot_corpus_run.py", "reuse_run")]
)
@pytest.mark.parametrize(("setting", "old", "new"), _SETTING_CHANGES, ids=_CHANGE_IDS)
def test_p05_changed_setting_refuses_analysis_or_reuse(
    name, mode, setting, old, new, tmp_path, monkeypatch, capsys
):
    driver, args, generated, event_out = _changed_generation(
        name, setting, old, new, tmp_path, monkeypatch
    )
    before = {path.name: path.read_bytes() for path in event_out.glob("*") if path.is_file()}
    assert "pose_config.json" in before
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
        assert result != 0, f"P05: {mode} accepted mismatched {setting}"
    assert any(
        term in message.lower()
        for term in ("drop", "config", "generation", "provenance", "mismatch")
    )
    assert len(generated) == 1
    for filename in ("cam-a.csv", "pose_config.json"):
        assert (event_out / filename).read_bytes() == before[filename]
