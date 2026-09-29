"""M2.9.1 reviewer witnesses; target implementation = 51e2094, synthetic only."""

from __future__ import annotations

import csv
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from pose_estimation.rtmlib_smoothing import KeypointSmoother

_ROOT = Path(__file__).resolve().parents[1]


def test_r07_empty_numpy_ids_are_a_valid_empty_identity_sequence() -> None:
    result = KeypointSmoother()(
        np.empty((0, 133, 2)), np.empty((0, 133)), 0.0, track_ids=np.array([], dtype=int)
    )
    assert result == (None, None)


def test_r07_zero_numpy_id_does_not_hide_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match=r"length.*does not match"):
        KeypointSmoother()(np.empty((0, 133, 2)), np.empty((0, 133)), 0.0, track_ids=np.array([0]))


def test_r14_unchanged_behavior_case_rejects_complete_smoothing_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_m2u91_subject_tracker import (
        test_p09_none_preserves_unnamed_smoothing_over_generated_sequences as unchanged_check,
    )

    points, scores = np.ones((1, 133, 2)), np.ones((1, 133))
    assert KeypointSmoother()(points, scores, 0.0)[0] is None

    def passthrough(_self, keypoints, scores, _t, track_ids=None):
        return keypoints, scores

    monkeypatch.setattr(KeypointSmoother, "__call__", passthrough)
    assert KeypointSmoother()(points, scores, 0.0)[0] is points
    with pytest.raises(AssertionError):
        unchanged_check()


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("corpus_run_2d.py", "resume"),
        ("corpus_run_2d.py", "analyse-only"),
        ("pilot_corpus_run.py", "reuse-run"),
    ],
)
def test_r09_report_tracker_names_the_generation_not_the_reanalysis_invocation(
    name: str,
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = runpy.run_path(str(_ROOT / "scripts" / name), run_name="_review_m2u91_driver")
    driver = driver["main"].__globals__
    monkeypatch.setattr(sys, "argv", [name])
    args = driver["_parse_args"]()
    for field in ("inventory", "qualification", "sessions", "out"):
        path = tmp_path / field
        path.mkdir()
        setattr(args, field, path)
    args.tracker = "rtmlib"
    args.report = tmp_path / "report.json"
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
    # Upstream publisher validity and the pilot's containment probe are orthogonal;
    # table joins, markers, resume/reuse, diagnostics and reporting stay real.
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
    generated: list[str] = []
    event_out = args.out / event_id

    def subprocess_call(command: list[str], **_kwargs: object) -> int:
        if "pose_estimation.run" in command:
            generated.append(command[command.index("--tracker") + 1])
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
    original = json.loads(args.report.read_text(encoding="utf-8"))
    assert generated == ["rtmlib"]
    assert original["configuration"]["tracker"] == "rtmlib"
    assert original["population"]["events"] == 1
    assert all(original["verdicts"].values())
    landmark_bytes = (event_out / f"{camera}.csv").read_bytes()

    args.tracker = "subject"
    if mode == "analyse-only":
        args.analyse_only = True
    elif mode == "reuse-run":
        args.reuse_run = True
    refusal = driver.get("RunError", driver.get("PilotError"))
    assert refusal is not None
    try:
        driver["main"]()
    except refusal as error:
        if not any(
            word in str(error).lower() for word in ("tracker", "config", "generation", "provenance")
        ):
            raise
        return
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if mode != "resume":
        assert generated == ["rtmlib"]
        assert (event_out / f"{camera}.csv").read_bytes() == landmark_bytes
    capsys.readouterr()
    assert payload["configuration"]["tracker"] == generated[-1], (
        f"{name} --{mode}: report tracker={payload['configuration']['tracker']}; "
        f"actual generation tracker={generated[-1]}; pose invocations={len(generated)}"
    )
