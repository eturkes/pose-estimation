from __future__ import annotations

import csv
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pose_estimation import run as run_module

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = _ROOT / "scripts" / name
    module_name = f"_review_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pilot_refuses_unvalidated_source_tables_before_publication(tmp_path, monkeypatch) -> None:
    pilot = _load_script("pilot_corpus_run.py")
    inventory = tmp_path / "inventory"
    qualification = tmp_path / "qualification"
    sessions = tmp_path / "sessions"
    out = tmp_path / "out"
    report = out / "pilot_report.json"
    _write_rows(
        inventory / "assets.csv",
        [
            {
                "asset_id": "asset-1",
                "disposition": "canonical",
                "reported_rotation_deg": "0",
                "reported_frame_count": "10",
            }
        ],
    )
    _write_rows(
        qualification / "assets_qc.csv",
        [
            {
                "asset_id": "asset-1",
                "pts_monotonic": "1",
                "codec": "hevc",
                "device_config": "subject90",
            }
        ],
    )
    _write_rows(
        sessions / "placements.csv",
        [
            {
                "asset_id": "asset-1",
                "event_id": "event-1",
                "camera_name": "cam-1",
                "placement": "placed",
            }
        ],
    )
    args = SimpleNamespace(
        inventory=inventory,
        qualification=qualification,
        sessions=sessions,
        out=out,
        report=report,
        min_assets=1,
        seed=0,
        model="rtmw-l",
        tracking="hands-arms",
        tracker="rtmlib",
        drop_lower_body=False,
        drop_hips_camera=None,
        hand_gate=False,
        camera_survey=False,
        task_gap_s=2.5,
        det_device="CPU",
        pose_device="NPU",
        det_frequency=7,
        single_subject=True,
        max_frames=0,
        reuse_run=True,
    )
    monkeypatch.setattr(pilot, "_parse_args", lambda: args)
    monkeypatch.setattr(pilot, "tree_digest", lambda _path: "stable")
    monkeypatch.setattr(pilot, "validate_generation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        pilot,
        "_guard_verdicts",
        lambda *_args, **_kwargs: {"events_probed": 1, "default_output_refused": 1},
    )

    def diagnostics(_event_out, assets):
        asset = next(iter(assets.values()))
        return [pilot.AssetRun(asset, 10, 10, 0, 0, 1.0, 1.0)]

    monkeypatch.setattr(pilot, "_diagnostics", diagnostics)
    monkeypatch.setattr(
        pilot,
        "_partition",
        lambda *_args, **_kwargs: pilot.Partition(n_input=1, n_windowed=1),
    )

    refused = False
    try:
        result = pilot.main()
    except Exception:
        refused = True
        result = 2

    assert refused or result != 0, (
        "unvalidated registry/qualification/session rows must not publish"
    )
    assert not report.exists(), "a stale or edited input generation must not produce a pilot report"


def test_pilot_rerun_clears_the_selected_event_before_launch(tmp_path, monkeypatch) -> None:
    pilot = _load_script("pilot_corpus_run.py")
    event_dir = tmp_path / "sessions" / "event-1"
    event_dir.mkdir(parents=True)
    out = tmp_path / "out"
    stale = out / event_dir.name / "cam-1_diag.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    log = out / "logs" / "run-00.log"
    log.parent.mkdir(parents=True)
    cleared_at_launch: list[bool] = []

    def launch(*_args, **_kwargs):
        cleared_at_launch.append(not stale.exists())
        return 0

    monkeypatch.setattr(pilot.subprocess, "call", launch)
    args = SimpleNamespace(
        model="rtmw-l",
        tracking="hands-arms",
        tracker="rtmlib",
        drop_lower_body=False,
        drop_hips_camera=None,
        hand_gate=False,
        camera_survey=False,
        task_gap_s=2.5,
        det_device="CPU",
        pose_device="NPU",
        det_frequency=7,
        single_subject=True,
        max_frames=0,
    )

    pilot._run_event(event_dir, out, log, args)

    assert cleared_at_launch == [True], (
        "a fresh pilot attempt must not credit diagnostics or clinical artifacts from an older run"
    )


def _snapshot(root: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_complete_resume_keeps_every_published_output_byte_identical(tmp_path, monkeypatch) -> None:
    driver = _load_script("corpus_run_2d.py")
    inventory = tmp_path / "inventory"
    qualification = tmp_path / "qualification"
    sessions = tmp_path / "sessions"
    out = tmp_path / "out"
    for path in (inventory, qualification, sessions):
        path.mkdir()
    args = SimpleNamespace(
        inventory=inventory,
        qualification=qualification,
        sessions=sessions,
        out=out,
        report=None,
        limit=0,
        model="rtmw-l",
        tracking="body",
        tracker="rtmlib",
        drop_lower_body=False,
        drop_hips_camera=None,
        hand_gate=False,
        camera_survey=False,
        task_gap_s=2.5,
        det_device="CPU",
        pose_device="NPU",
        det_frequency=7,
        single_subject=True,
        retry_failed=True,
        analyse_only=False,
    )
    asset = driver.pilot.Asset(
        asset_id="asset-1",
        event_id="event-1",
        camera_name="cam-1",
        codec="hevc",
        device_config="tablet",
        rotation_deg=0,
        pts_monotonic=1,
        reported_frames=10,
    )
    monkeypatch.setattr(driver, "_parse_args", lambda: args)
    monkeypatch.setattr(driver.pilot, "_load_assets", lambda *_args: [asset])
    monkeypatch.setattr(driver, "_canonical_asset_ids", lambda _path: [asset.asset_id])
    monkeypatch.setattr(driver, "tree_digest", lambda _path: "tree-digest")
    monkeypatch.setattr(driver, "generation_digest", lambda _path: "marker-digest")
    monkeypatch.setattr(driver, "validate_generation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        driver,
        "_partitions",
        lambda *_args, **_kwargs: (driver.pilot.Partition(n_input=1, n_windowed=1), 0),
    )
    attempts: list[str] = []

    def attempt(event_id, _args, _logs):
        attempts.append(event_id)
        event_out = out / event_id
        event_out.mkdir(parents=True, exist_ok=True)
        (event_out / f"{asset.camera_name}.csv").write_text("frame\n0\n", encoding="utf-8")
        _write_rows(
            event_out / f"{asset.camera_name}_diag.csv",
            [
                {
                    # R25's sibling acceptance check makes ownership part of a
                    # well-formed row, and `write_source_diagnostics` always
                    # emits it — a fixture omitting it grades `wrong_diag`, not
                    # the byte idempotence this case exists for.
                    "video": f"{asset.event_id}/{asset.camera_name}",
                    "n_frames_decoded": "10",
                    "pts_accepted": "10",
                    "index_fallback": "0",
                    "monotonic_forced": "0",
                    "cfr_fallback_rate": "0.000000",
                }
            ],
        )
        driver.write_marker(
            event_out,
            status=driver.MARKER_COMPLETE,
            stage=driver.STAGE_CLINICAL,
            exit_code=0,
            run_s=1.0,
            clinical_s=0.5,
        )
        return {
            "status": driver.MARKER_COMPLETE,
            "stage": driver.STAGE_CLINICAL,
            "run_s": 1.0,
            "clinical_s": 0.5,
        }

    monkeypatch.setattr(driver, "_attempt_event", attempt)

    assert driver.main() == 0
    first = _snapshot(out)
    assert driver.main() == 0
    second = _snapshot(out)

    assert attempts == [asset.event_id], "the completed event must spend inference only once"
    assert second == first, "P05 covers the manifest and report, not only per-event files"


def test_interrupted_source_propagates_after_writing_diagnostics(tmp_path, monkeypatch) -> None:
    class Capture:
        def __init__(self) -> None:
            self.open = True
            self.released = False

        def get(self, _prop):
            return 10.0

        def isOpened(self) -> bool:
            return self.open

        def read(self):
            self.open = False
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    capture = Capture()
    monkeypatch.setattr(run_module, "open_capture", lambda *_args, **_kwargs: capture)
    diagnostics = tmp_path / "source_diag.csv"
    args = SimpleNamespace(headless=True, tracking="body", max_frames=0, single_subject=True)

    def interrupt(_frame):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_module.process_source(
            args,
            interrupt,
            "synthetic.avi",
            lambda *_args, **_kwargs: None,
            output_diag=diagnostics,
            video_name="synthetic",
        )

    assert diagnostics.is_file(), "the finally arm must retain the decoded-prefix evidence"
    assert capture.released


def test_artifact_validator_rejects_a_diagnostic_owned_by_another_source(
    tmp_path,
) -> None:
    driver = _load_script("corpus_run_2d.py")
    asset = SimpleNamespace(asset_id="asset-1", event_id="event-1", camera_name="cam-1")
    event_out = tmp_path / asset.event_id
    event_out.mkdir()
    (event_out / f"{asset.camera_name}.csv").write_text("frame\n0\n", encoding="utf-8")
    _write_rows(
        event_out / f"{asset.camera_name}_diag.csv",
        [
            {
                "video": "other-event/other-camera",
                "n_frames_decoded": "10",
                "pts_accepted": "10",
                "index_fallback": "0",
                "monotonic_forced": "0",
                "cfr_fallback_rate": "0.000000",
                "fps_nominal": "10.000000",
                "latency_ms_mean": "1.000",
                "latency_ms_p95": "1.000",
            }
        ],
    )
    rows = [{"asset_id": asset.asset_id, "disposition": driver.DISPOSITION_OK}]

    verdict = driver._artifacts(rows, {asset.asset_id: asset}, tmp_path)

    assert verdict["wrong_diag"] == 1, "P09 ownership includes the diagnostic row's source key"
    assert verdict["counters"] == [], "foreign counters must not enter the corpus CFR population"


@pytest.mark.parametrize("sink", ["out", "report"])
def test_driver_refuses_every_sink_overlapping_the_published_sessions(
    tmp_path, monkeypatch, sink
) -> None:
    driver = _load_script("corpus_run_2d.py")
    inventory = tmp_path / "inventory"
    qualification = tmp_path / "qualification"
    sessions = tmp_path / "sessions"
    external = tmp_path / "external"
    for path in (inventory, qualification, sessions):
        path.mkdir()
    (sessions / "generation.json").write_text("{}\n", encoding="utf-8")
    before = _snapshot(sessions)
    args = SimpleNamespace(
        inventory=inventory,
        qualification=qualification,
        sessions=sessions,
        out=sessions if sink == "out" else external,
        report=sessions / "run_report.json" if sink == "report" else None,
        limit=0,
        model="rtmw-l",
        tracking="body",
        tracker="rtmlib",
        drop_lower_body=False,
        drop_hips_camera=None,
        hand_gate=False,
        camera_survey=False,
        task_gap_s=2.5,
        det_device="CPU",
        pose_device="NPU",
        det_frequency=7,
        single_subject=True,
        retry_failed=True,
        analyse_only=True,
    )
    monkeypatch.setattr(driver, "_parse_args", lambda: args)
    monkeypatch.setattr(driver.pilot, "_load_assets", lambda *_args: [])
    monkeypatch.setattr(driver, "_canonical_asset_ids", lambda _path: ["asset-1"])
    monkeypatch.setattr(driver, "tree_digest", lambda _path: "tree-digest")
    monkeypatch.setattr(driver, "generation_digest", lambda _path: "marker-digest")
    monkeypatch.setattr(driver, "validate_generation", lambda *_args, **_kwargs: {})

    refused = False
    try:
        result = driver.main()
    except Exception:
        refused = True
        result = 2

    assert refused or result != 0, f"the {sink} sink must be rejected before publication"
    assert _snapshot(sessions) == before, f"rejecting an overlapping {sink} must be write-free"


def test_shipped_redaction_guard_rejects_an_unknown_identifier_shaped_key() -> None:
    pilot = _load_script("pilot_corpus_run.py")

    with pytest.raises(pilot.PilotError):
        pilot._assert_redacted({"subject90": 1}, frozenset())


def test_cfr_derivation_rejects_a_nonfinite_stored_rate() -> None:
    driver = _load_script("corpus_run_2d.py")
    verdict = driver._cfr(
        [
            {
                "n_frames_decoded": 10,
                "pts_accepted": 10,
                "index_fallback": 0,
                "monotonic_forced": 0,
                "stored_rate": float("nan"),
            }
        ]
    )

    assert verdict["assets_rate_mismatch"] == 1, "NaN is not equal to any finite derivation"


def test_throughput_never_labels_mixed_frame_and_wall_populations_as_corpus(
    tmp_path, monkeypatch
) -> None:
    driver = _load_script("corpus_run_2d.py")
    inventory = tmp_path / "inventory"
    qualification = tmp_path / "qualification"
    sessions = tmp_path / "sessions"
    out = tmp_path / "out"
    for path in (inventory, qualification, sessions):
        path.mkdir()
    assets = [
        driver.pilot.Asset(
            asset_id=f"asset-{index}",
            event_id=f"event-{index}",
            camera_name="cam-1",
            codec="hevc",
            device_config="tablet",
            rotation_deg=0,
            pts_monotonic=1,
            reported_frames=frames,
        )
        for index, frames in ((1, 100), (2, 200))
    ]
    for asset, status in zip(assets, (driver.MARKER_COMPLETE, driver.MARKER_FAILED), strict=True):
        event_out = out / asset.event_id
        event_out.mkdir(parents=True)
        (event_out / f"{asset.camera_name}.csv").write_text("frame\n0\n", encoding="utf-8")
        _write_rows(
            event_out / f"{asset.camera_name}_diag.csv",
            [
                {
                    "video": f"{asset.event_id}/{asset.camera_name}",
                    "n_frames_decoded": str(asset.reported_frames),
                    "pts_accepted": str(asset.reported_frames),
                    "index_fallback": "0",
                    "monotonic_forced": "0",
                    "cfr_fallback_rate": "0.000000",
                }
            ],
        )
        driver.write_marker(
            event_out,
            status=status,
            stage=driver.STAGE_CLINICAL,
            exit_code=0 if status == driver.MARKER_COMPLETE else 1,
            run_s=float(asset.reported_frames),
            clinical_s=1.0,
        )
    report = out / "run_report.json"
    args = SimpleNamespace(
        inventory=inventory,
        qualification=qualification,
        sessions=sessions,
        out=out,
        report=report,
        limit=0,
        model="rtmw-l",
        tracking="body",
        tracker="rtmlib",
        drop_lower_body=False,
        drop_hips_camera=None,
        hand_gate=False,
        camera_survey=False,
        task_gap_s=2.5,
        det_device="CPU",
        pose_device="NPU",
        det_frequency=7,
        single_subject=True,
        retry_failed=False,
        analyse_only=True,
    )
    # M2.9.1 A03: a complete event names the pose configuration it was produced under.
    for asset in assets:
        driver.write_pose_config(out / asset.event_id, driver.pose_config(args))
    monkeypatch.setattr(driver, "_parse_args", lambda: args)
    monkeypatch.setattr(driver.pilot, "_load_assets", lambda *_args: assets)
    monkeypatch.setattr(driver, "_canonical_asset_ids", lambda _path: [a.asset_id for a in assets])
    monkeypatch.setattr(driver, "tree_digest", lambda _path: "tree-digest")
    monkeypatch.setattr(driver, "generation_digest", lambda _path: "marker-digest")
    monkeypatch.setattr(driver, "validate_generation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        driver,
        "_partitions",
        lambda *_args, **_kwargs: (driver.pilot.Partition(n_input=1, n_windowed=1), 0),
    )

    assert driver.main() == 1, "the clinical failure must keep the overall report red"
    payload = json.loads(report.read_text(encoding="utf-8"))
    throughput = payload["throughput"]

    assert not (
        throughput["sample"] == driver.THROUGHPUT_FULL
        and throughput["frames_decoded"] == 100
        and throughput["run_wall_s"] == 300.0
    ), "a corpus-labelled rate cannot divide one event's frames by two events' wall time"
