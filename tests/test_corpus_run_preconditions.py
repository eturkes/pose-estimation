"""M2.8.1 executable preconditions for the corpus run."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import textwrap
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest

from pose_estimation import run as run_module
from pose_estimation import sessions as sessions_module
from pose_estimation.multicam import Session, SessionCamera, SessionError, process_session
from pose_estimation.video_io import SourceTimestampClock


def _make_session(
    parent: pathlib.Path,
    *,
    session_id: str = "event-01",
    directory_name: str = "recording",
    cameras: tuple[str, ...] = ("cam-a", "cam-b"),
) -> Session:
    directory = parent / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    session_cameras = []
    for name in cameras:
        source = directory / f"{name}.avi"
        source.write_bytes(f"source-{name}".encode())
        session_cameras.append(SessionCamera(name=name, file=source))
    return Session(session_id=session_id, directory=directory.resolve(), cameras=session_cameras)


def _snapshot(root: pathlib.Path) -> tuple[tuple[str, str, bytes | str], ...]:
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append((relative, "link", str(path.readlink())))
        elif path.is_dir():
            entries.append((relative, "dir", ""))
        else:
            entries.append((relative, "file", path.read_bytes()))
    return tuple(entries)


def _write_generation(root: pathlib.Path) -> dict[str, object]:
    for name in (sessions_module.EVENTS_FILENAME, sessions_module.PLACEMENTS_FILENAME):
        (root / name).write_text(f"{name}\n", encoding="utf-8")
    generation: dict[str, object] = {
        sessions_module.EVENTS_FILENAME: hashlib.sha256(
            (root / sessions_module.EVENTS_FILENAME).read_bytes()
        ).hexdigest(),
        sessions_module.PLACEMENTS_FILENAME: hashlib.sha256(
            (root / sessions_module.PLACEMENTS_FILENAME).read_bytes()
        ).hexdigest(),
        "tree": sessions_module.tree_digest(root),
        "inventory": {},
        "generator_version": sessions_module.GENERATOR_VERSION,
    }
    (root / sessions_module.GENERATION_FILENAME).write_text(
        json.dumps(generation, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    assert sessions_module.validate_generation(root) == generation
    return generation


def _dispatch_with_writing_source(
    monkeypatch: pytest.MonkeyPatch,
    session: Session,
    output_dir: pathlib.Path,
) -> list[pathlib.Path]:
    written_csvs: list[pathlib.Path] = []

    def fake_resolve_sessions(*_args, **_kwargs):
        return [session]

    def fake_process_source(*_args, **kwargs):
        csv_path = pathlib.Path(kwargs["output_csv"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("camera-output\n", encoding="utf-8")
        written_csvs.append(csv_path)
        if output_diag := kwargs.get("output_diag"):
            pathlib.Path(output_diag).write_text("diagnostics\n", encoding="utf-8")
        return []

    monkeypatch.setattr(run_module, "resolve_cli_sessions", fake_resolve_sessions)
    monkeypatch.setattr(run_module, "process_source", fake_process_source)
    args = SimpleNamespace(
        session_dir=str(session.directory),
        sessions_dir=None,
        calibration=None,
        output_dir=str(output_dir),
    )
    run_module._dispatch_sessions(
        args,
        pose_tracker=object(),
        draw_skeleton=None,
        smoother=None,
        bone_smoother=None,
        screen=None,
    )
    return written_csvs


def test_p01_dispatch_routes_every_camera_to_explicit_output_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _make_session(
        tmp_path / "sessions",
        session_id="manifest-id",
        directory_name="directory-id",
    )
    output_dir = tmp_path / "external" / "fresh"

    written = _dispatch_with_writing_source(monkeypatch, session, output_dir)

    expected = [
        output_dir / session.session_id / f"{camera.name}.csv" for camera in session.cameras
    ]
    assert written == expected, "P01: --output-dir must reach every per-camera write"
    assert all(path.read_text(encoding="utf-8") == "camera-output\n" for path in expected)


def test_p02_published_tree_default_refuses_before_any_write(tmp_path: pathlib.Path) -> None:
    published = tmp_path / "published"
    session = _make_session(published)
    (published / sessions_module.GENERATION_FILENAME).write_text("{}\n", encoding="utf-8")
    before = _snapshot(published)
    calls = 0

    def recorder(**_kwargs):
        nonlocal calls
        calls += 1

    with pytest.raises(SessionError):
        process_session(session, camera_processor=recorder)

    assert calls == 0, "P02: containment refusal must precede camera callbacks"
    assert _snapshot(published) == before, "P02: containment refusal must precede mkdir/write"


def test_p03_external_dispatch_preserves_published_generation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "published"
    session = _make_session(published)
    generation = _write_generation(published)
    before = _snapshot(published)
    output_dir = tmp_path / "external"

    written = _dispatch_with_writing_source(monkeypatch, session, output_dir)

    assert written == [
        output_dir / session.session_id / f"{camera.name}.csv" for camera in session.cameras
    ]
    assert _snapshot(published) == before, "P03: an external run must not mutate published bytes"
    assert sessions_module.validate_generation(published) == generation


@pytest.mark.parametrize("relationship", ["inside", "equal", "ancestor", "symlink-inside"])
def test_p04_canonical_final_output_overlap_refuses_both_directions(
    tmp_path: pathlib.Path, relationship: str
) -> None:
    session_id = "event-01"
    if relationship == "ancestor":
        allowed_root = tmp_path / "allowed-published"
        allowed_session = _make_session(allowed_root, session_id=session_id, cameras=("cam-a",))
        (allowed_root / sessions_module.GENERATION_FILENAME).write_text("{}\n", encoding="utf-8")
        allowed_before = _snapshot(allowed_root)

        def writer(**kwargs):
            pathlib.Path(kwargs["output_csv"]).write_text("allowed\n", encoding="utf-8")

        process_session(allowed_session, camera_processor=writer, output_dir=tmp_path)
        assert (tmp_path / session_id / "cam-a.csv").is_file()
        assert _snapshot(allowed_root) == allowed_before

        output_dir = tmp_path / "outer"
        published = output_dir / session_id / "published"
    else:
        published = tmp_path / "published"
        if relationship == "inside":
            output_dir = published / "results"
        elif relationship == "equal":
            session_id = published.name
            output_dir = published.parent
        else:
            alias = tmp_path / "published-alias"
            alias.symlink_to(published, target_is_directory=True)
            output_dir = alias / "results"

    session = _make_session(published, session_id=session_id)
    (published / sessions_module.GENERATION_FILENAME).write_text("{}\n", encoding="utf-8")
    before = _snapshot(published)
    calls = 0

    def recorder(**kwargs):
        nonlocal calls
        calls += 1
        pathlib.Path(kwargs["output_csv"]).write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(SessionError):
        process_session(session, camera_processor=recorder, output_dir=output_dir)

    assert calls == 0, f"P04: {relationship} overlap must refuse before callbacks"
    assert _snapshot(published) == before


_SOURCE_DIAGNOSTIC_FIELDS = (
    "video",
    "n_frames_decoded",
    "pts_accepted",
    "index_fallback",
    "monotonic_forced",
    "cfr_fallback_rate",
    "fps_nominal",
    "latency_ms_mean",
    "latency_ms_p95",
)


class _EmptyCapture:
    def __init__(self) -> None:
        self.released = False

    def get(self, prop: int) -> float:
        values = {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 0.0,
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        }
        return values.get(prop, 0.0)

    def isOpened(self) -> bool:
        return not self.released

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        self.released = True


def _source_args() -> SimpleNamespace:
    return SimpleNamespace(
        headless=True,
        tracking="body",
        max_frames=0,
        single_subject=False,
    )


def _run_empty_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_diag: pathlib.Path | None,
    video_name: str = "source-label",
) -> list[float]:
    capture = _EmptyCapture()
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: capture)
    return cast(Any, run_module.process_source)(
        _source_args(),
        lambda _frame: (None, None),
        "synthetic.avi",
        lambda *_args, **_kwargs: None,
        output_diag=output_diag,
        video_name=video_name,
    )


def test_p05_successful_zero_frame_source_writes_one_exact_diagnostic_row(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_diag = tmp_path / "missing" / "source_diag.csv"

    latencies = _run_empty_source(monkeypatch, output_diag=output_diag)

    assert latencies == []
    with output_diag.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == _SOURCE_DIAGNOSTIC_FIELDS
        rows = list(reader)
    assert len(rows) == 1, "P05: one opened source must publish exactly one summary row"
    assert rows[0]["video"] == "source-label"
    assert tuple(rows[0][field] for field in _SOURCE_DIAGNOSTIC_FIELDS[1:5]) == (
        "0",
        "0",
        "0",
        "0",
    )
    assert rows[0]["cfr_fallback_rate"] == "0.000000"


def test_p05_capture_open_failure_does_not_publish_diagnostics(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_diag = tmp_path / "never-opened_diag.csv"
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: None)

    latencies = cast(Any, run_module.process_source)(
        _source_args(),
        lambda _frame: (None, None),
        "synthetic.avi",
        lambda *_args, **_kwargs: None,
        output_diag=output_diag,
    )

    assert latencies == []
    assert not output_diag.exists(), "P05: an unopened capture has no decoded-source summary"


def test_p06_dispatch_forwards_diagnostic_path_and_announces_existing_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = _make_session(tmp_path / "sessions", cameras=("cam-a",))
    output_dir = tmp_path / "external"
    received: list[pathlib.Path] = []

    monkeypatch.setattr(run_module, "resolve_cli_sessions", lambda *_args, **_kwargs: [session])

    def fake_process_source(*_args, **kwargs):
        csv_path = pathlib.Path(kwargs["output_csv"])
        diag_path = pathlib.Path(kwargs["output_diag"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("csv\n", encoding="utf-8")
        diag_path.write_text("diag\n", encoding="utf-8")
        received.append(diag_path)
        return []

    monkeypatch.setattr(run_module, "process_source", fake_process_source)
    args = SimpleNamespace(
        session_dir=str(session.directory),
        sessions_dir=None,
        calibration=None,
        output_dir=str(output_dir),
    )

    run_module._dispatch_sessions(
        args,
        pose_tracker=object(),
        draw_skeleton=None,
        smoother=None,
        bone_smoother=None,
        screen=None,
    )

    expected = output_dir / session.session_id / "cam-a_diag.csv"
    stdout = capsys.readouterr().out
    assert received == [expected], "P06: _camera_processor must forward the exact path"
    assert expected.read_text(encoding="utf-8") == "diag\n"
    assert f"Wrote diagnostics: {expected}" in stdout


def test_p06_session_suppresses_claims_for_files_the_callback_did_not_create(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = _make_session(tmp_path / "sessions", cameras=("cam-a",))

    process_session(
        session,
        camera_processor=lambda **_kwargs: None,
        output_dir=tmp_path / "external",
    )

    stdout = capsys.readouterr().out
    assert "Wrote CSV:" not in stdout
    assert "Wrote diagnostics:" not in stdout


def test_p07_explicitly_disabled_diagnostics_write_and_report_nothing(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = tuple(tmp_path.iterdir())

    latencies = _run_empty_source(monkeypatch, output_diag=None)

    stdout = capsys.readouterr().out.lower()
    assert latencies == []
    assert tuple(tmp_path.iterdir()) == before
    assert "diagnostic" not in stdout
    assert "fallback rate" not in stdout


class _TimestampCapture:
    def __init__(self, pos_msec: list[float]) -> None:
        self._pos_msec = iter(pos_msec)

    def get(self, _prop: int) -> float:
        return next(self._pos_msec)


def _clock_counts(clock: SourceTimestampClock) -> tuple[int, int, int]:
    state = vars(clock)
    return (
        cast(int, state["pts_accepted"]),
        cast(int, state["index_fallback"]),
        cast(int, state["monotonic_forced"]),
    )


def test_p08_clock_counters_form_an_exclusive_prefix_partition() -> None:
    file_clock = SourceTimestampClock(
        _TimestampCapture([0.0, -1.0, 10_000.0, -1.0]),
        10.0,
        live=False,
    )
    expected_file = [
        (1, 0, 0),
        (1, 1, 0),
        (2, 1, 0),
        (2, 1, 1),
    ]
    for index, expected in enumerate(expected_file):
        file_clock.timestamp(index)
        counts = _clock_counts(file_clock)
        assert counts == expected
        assert all(type(count) is int and count >= 0 for count in counts)
        assert sum(counts) == index + 1, "P08: every completed call has one disposition"

    ticks = iter([50.0, 50.25])
    live_clock = SourceTimestampClock(
        _TimestampCapture([]),
        10.0,
        live=True,
        monotonic=lambda: next(ticks),
    )
    assert _clock_counts(live_clock) == (0, 0, 0)
    for index in range(2):
        live_clock.timestamp(index)
        assert _clock_counts(live_clock) == (index + 1, 0, 0)


def test_p09_clock_counter_attribution_matches_returning_branch() -> None:
    cases = [
        (
            [0.0, 125.0, 300.0],
            10.0,
            [(1, 0, 0), (2, 0, 0), (3, 0, 0)],
        ),
        (
            [-1.0, -1.0, -1.0],
            10.0,
            [(0, 1, 0), (0, 2, 0), (0, 3, 0)],
        ),
        (
            [1_000.0, 1_000.0, 1_000.0],
            10.0,
            [(1, 0, 0), (1, 0, 1), (1, 0, 2)],
        ),
    ]

    for pos_msec, fps, expected_prefixes in cases:
        clock = SourceTimestampClock(_TimestampCapture(pos_msec), fps, live=False)
        for index, expected in enumerate(expected_prefixes):
            clock.timestamp(index)
            assert _clock_counts(clock) == expected
            assert sum(expected) == index + 1


def test_p10_timestamp_values_match_the_frozen_baseline_table() -> None:
    file_cases = [
        (
            [0.0, 100.0, float("nan"), 150.0, 400.0],
            10.0,
            [
                "0x0.0p+0",
                "0x1.999999999999ap-4",
                "0x1.999999999999ap-3",
                "0x1.3333333333333p-2",
                "0x1.999999999999ap-2",
            ],
        ),
        (
            [0.0, 175.0, 410.0],
            10.0,
            ["0x0.0p+0", "0x1.6666666666666p-3", "0x1.a3d70a3d70a3dp-2"],
        ),
        (
            [0.0, 0.0, 0.0],
            25.0,
            ["0x0.0p+0", "0x1.47ae147ae147bp-5", "0x1.47ae147ae147bp-4"],
        ),
    ]
    exercised: list[SourceTimestampClock] = []

    for pos_msec, fps, expected_hex in file_cases:
        clock = SourceTimestampClock(_TimestampCapture(pos_msec), fps, live=False)
        actual = [clock.timestamp(index).hex() for index in range(len(pos_msec))]
        assert actual == expected_hex
        exercised.append(clock)

    ticks = iter([50.0, 50.25, 50.20, 50.40])
    live_clock = SourceTimestampClock(
        _TimestampCapture([]),
        10.0,
        live=True,
        monotonic=lambda: next(ticks),
    )
    assert [live_clock.timestamp(index).hex() for index in range(4)] == [
        "0x0.0p+0",
        "0x1.0000000000000p-2",
        "0x1.6666666666666p-2",
        "0x1.9999999999980p-2",
    ]
    exercised.append(live_clock)

    counter_fields = {"pts_accepted", "index_fallback", "monotonic_forced"}
    assert all(counter_fields <= vars(clock).keys() for clock in exercised)


class _FrameCapture:
    def __init__(self, pos_msec: list[float], *, fps: float = 10.0) -> None:
        self._pos_msec = iter(pos_msec)
        self._frames_left = len(pos_msec)
        self._fps = fps
        self.released = False

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_MSEC:
            return next(self._pos_msec)
        values = {
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_COUNT: float(self._frames_left),
            cv2.CAP_PROP_FRAME_WIDTH: 8.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 6.0,
        }
        return values.get(prop, 0.0)

    def isOpened(self) -> bool:
        return not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frames_left == 0:
            return False, None
        self._frames_left -= 1
        return True, np.zeros((6, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_p11_fallback_rate_reaches_csv_and_stdout(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture = _FrameCapture([0.0, -1.0, 10_000.0, -1.0])
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: capture)
    ticks = iter(value for index in range(4) for value in (float(index), index + 0.001))
    monkeypatch.setattr(run_module.time, "perf_counter", lambda: next(ticks))
    output_diag = tmp_path / "source_diag.csv"

    latencies = cast(Any, run_module.process_source)(
        _source_args(),
        lambda _frame: (None, None),
        "synthetic.avi",
        lambda *_args, **_kwargs: None,
        output_diag=output_diag,
        video_name="pilot-source",
    )

    with output_diag.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == _SOURCE_DIAGNOSTIC_FIELDS
        rows = list(reader)
    assert latencies == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row[field] for field in _SOURCE_DIAGNOSTIC_FIELDS[:6]) == (
        "pilot-source",
        "4",
        "2",
        "1",
        "1",
        "0.500000",
    )
    rate_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "cfr" in line.lower() and "fallback" in line.lower()
    ]
    assert len(rate_lines) == 1, "P11: one requested diagnostic emits one rate summary"
    assert re.search(r"(?:0\.5(?:0+)?|50(?:\.0+)?%)", rate_lines[0])


def test_p11_interrupted_source_still_publishes_its_decoded_prefix(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _FrameCapture([0.0])
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: capture)
    output_diag = tmp_path / "interrupted_diag.csv"

    def interrupt(_frame):
        raise KeyboardInterrupt

    latencies = cast(Any, run_module.process_source)(
        _source_args(),
        interrupt,
        "synthetic.avi",
        lambda *_args, **_kwargs: None,
        output_diag=output_diag,
        video_name="interrupted-source",
    )

    with output_diag.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert latencies == []
    assert len(rows) == 1
    assert tuple(rows[0][field] for field in _SOURCE_DIAGNOSTIC_FIELDS[:6]) == (
        "interrupted-source",
        "1",
        "1",
        "0",
        "0",
        "0.000000",
    )
    assert capture.released


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CLINICAL_R = _PROJECT_ROOT / "analysis" / "clinical_features.R"


def _run_r_json(body: str) -> dict[str, Any]:
    source = f"suppressWarnings(try(source({json.dumps(str(_CLINICAL_R))}), silent=TRUE))\n"
    script = (
        source
        + textwrap.dedent(body)
        + "\njsonlite::write_json(result, stdout(), auto_unbox=TRUE, na='string', digits=17)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-"],
        input=script,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")),
        "",
    )
    assert payload, completed.stdout
    return cast(dict[str, Any], json.loads(payload))


def test_p12_every_group_drop_site_emits_its_frozen_reason_once() -> None:
    payload = _run_r_json(
        """
        mk <- function(video, timestamps) tibble(
          video=video, person_idx=0L, timestamp_sec=timestamps
        )
        df <- bind_rows(
          mk("too_few_frames", c(0, 0.5, 1.0)),
          mk("invalid_cadence", c(0, 0, 0, 0)),
          mk("no_finite_timestamps", c(0, 0.1, 0.2, Inf)),
          mk("shorter_than_window", c(0, 0.1, 0.2, 0.3)),
          mk("no_window_starts", c(10, 10.5, 11, 11.5)),
          mk("no_windows_emitted", c(20, 20.4, 20.8, 21.2))
        )
        base_seq <- base::seq
        seq <- function(from, to, by, ...) {
          if (identical(as.numeric(from), 10)) numeric(0) else base_seq(from, to, by, ...)
        }
        output <- compute_window_features(
          df, select(df, video, person_idx), "hands-arms", window_sec=1, is_3d=FALSE
        )
        rm(seq)
        result <- list(rows=output$group_qc)
        """
    )

    rows = cast(list[dict[str, Any]], payload["rows"])
    expected = {
        "too_few_frames": (3, "too_few_frames"),
        "invalid_cadence": (4, "invalid_cadence"),
        "no_finite_timestamps": (4, "no_finite_timestamps"),
        "shorter_than_window": (4, "shorter_than_window"),
        "no_window_starts": (4, "no_window_starts"),
        "no_windows_emitted": (4, "no_windows_emitted"),
    }
    actual = {row["video"]: (int(row["n_frames"]), row["drop_reason"]) for row in rows}
    assert len(rows) == len(expected), "P12: every dropped group has exactly one row"
    assert actual == expected


def test_p12_group_qc_row_refuses_an_unlisted_reason() -> None:
    payload = _run_r_json(
        """
        has_constructor <- exists("group_qc_row", mode="function")
        outcome <- tryCatch({
          group_qc_row("synthetic", 0L, 4L, "invented_reason")
          "accepted"
        }, error=function(condition) conditionMessage(condition))
        result <- list(has_constructor=has_constructor, outcome=outcome)
        """
    )

    assert payload["outcome"] != "accepted", "P12: unlisted reasons must fail closed"
    assert payload["has_constructor"] is True


def test_p13_group_outcomes_partition_video_person_groups_in_both_modes() -> None:
    payload = _run_r_json(
        """
        mk <- function(video, person_idx, timestamps) tibble(
          video=video,
          person_idx=as.integer(person_idx),
          timestamp_sec=timestamps,
          arm_left_wrist_x=seq_along(timestamps),
          arm_left_wrist_y=0,
          arm_left_wrist_z=0,
          arm_right_wrist_x=seq_along(timestamps) + 1,
          arm_right_wrist_y=0,
          arm_right_wrist_z=0,
          left_hand_8_x=seq_along(timestamps) + 2,
          left_hand_8_y=0,
          left_hand_8_z=0,
          right_hand_8_x=seq_along(timestamps) + 3,
          right_hand_8_y=0,
          right_hand_8_z=0
        )
        df <- bind_rows(
          mk("shared", 0L, seq(0, 1.5, by=0.1)),
          mk("shared", 1L, c(0, 0.5, 1.0)),
          mk("sparse", 0L, c(20, 20.4, 20.8, 21.2))
        )
        frame_features <- select(df, video, person_idx)
        summarize_partition <- function(output) {
          window_keys <- if (nrow(output$windows) == 0L) character(0) else unique(paste(
            output$windows$video, output$windows$person_idx, sep="|"
          ))
          group_keys <- if (is.null(output$group_qc) || nrow(output$group_qc) == 0L) {
            character(0)
          } else unique(paste(
            output$group_qc$video, output$group_qc$person_idx, sep="|"
          ))
          input_keys <- unique(paste(df$video, df$person_idx, sep="|"))
          list(
            n_input=length(input_keys),
            n_window_groups=length(window_keys),
            n_group_qc=length(group_keys),
            n_overlap=length(intersect(window_keys, group_keys)),
            n_union=length(union(window_keys, group_keys))
          )
        }
        result <- list(
          two_d=summarize_partition(compute_window_features(
            df, frame_features, "hands-arms", window_sec=1, is_3d=FALSE
          )),
          three_d=summarize_partition(compute_window_features(
            df, frame_features, "hands-arms", window_sec=1, is_3d=TRUE
          ))
        )
        """
    )

    expected = {
        "n_input": 3,
        "n_window_groups": 1,
        "n_group_qc": 2,
        "n_overlap": 0,
        "n_union": 3,
    }
    for mode in ("two_d", "three_d"):
        actual = {key: int(value) for key, value in payload[mode].items()}
        assert actual == expected, f"P13: {mode} outcomes must form a total disjoint partition"


_GROUP_QC_FIELDS = ("video", "person_idx", "n_frames", "drop_reason", "qc_status")
_GOLDEN_GENERATOR = _PROJECT_ROOT / "scripts" / "regenerate_r_clinical_goldens.py"
_GOLDEN_DIR = _PROJECT_ROOT / "tests" / "goldens" / "r_clinical"


def _load_golden_generator() -> Any:
    spec = importlib.util.spec_from_file_location("corpus_run_golden_generator", _GOLDEN_GENERATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv_rows(path: pathlib.Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def test_p14_group_qc_artifact_is_always_written_header_only_when_all_groups_emit_windows(
    tmp_path: pathlib.Path,
) -> None:
    generator = _load_golden_generator()
    input_2d = tmp_path / "valid_2d.csv"
    input_3d = tmp_path / "valid_3d.csv"
    generator._write_2d_input(input_2d, "idx")
    generator._write_world3d_input(input_3d)

    completed = subprocess.run(
        ["Rscript", str(_CLINICAL_R), str(tmp_path)],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    expected = (
        tmp_path / "valid_2d_clinical_group_qc.csv",
        tmp_path / "valid_3d_clinical_3d_group_qc.csv",
    )
    missing = [path.name for path in expected if not path.is_file()]
    assert not missing, f"P14: producer omitted always-written artifacts: {missing}"
    for path in expected:
        fields, rows = _read_csv_rows(path)
        assert fields == _GROUP_QC_FIELDS
        assert rows == [], f"P14: {path.name} must distinguish zero drops with a header-only file"


def test_p15_frozen_reason_set_is_rederived_from_the_r_source() -> None:
    source = _CLINICAL_R.read_text(encoding="utf-8")
    match = re.search(r"GROUP_QC_REASONS\s*<-\s*c\((.*?)\)", source, flags=re.DOTALL)
    assert match is not None, "P15: analysis source must publish GROUP_QC_REASONS"
    actual = set(re.findall(r"[\"']([^\"']+)[\"']", match.group(1)))
    expected = {
        "too_few_frames",
        "invalid_cadence",
        "no_finite_timestamps",
        "shorter_than_window",
        "no_window_starts",
        "no_windows_emitted",
    }
    assert actual == expected, (
        f"P15 reason names drifted: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )


def test_p16_committed_group_qc_goldens_carry_the_frozen_header_and_a_real_drop() -> None:
    names = (
        "2d_idx_clinical_group_qc.csv",
        "2d_cumsum_clinical_group_qc.csv",
        "2d_csv4dp_clinical_group_qc.csv",
        "2d_drop_clinical_group_qc.csv",
        "world3d_clinical_3d_group_qc.csv",
    )
    missing = [name for name in names if not (_GOLDEN_DIR / name).is_file()]
    assert not missing, f"P16 missing committed group-QC goldens: {missing}"
    headers = {name: _read_csv_rows(_GOLDEN_DIR / name)[0] for name in names}
    assert all(header == _GROUP_QC_FIELDS for header in headers.values()), headers
    # Four of the five are header-only by contract, so the populated one is what keeps
    # this predicate from ranging over an empty set (A07).
    fields, rows = _read_csv_rows(_GOLDEN_DIR / "2d_drop_clinical_group_qc.csv")
    del fields
    assert {row["drop_reason"] for row in rows} == {"too_few_frames", "shorter_than_window"}
