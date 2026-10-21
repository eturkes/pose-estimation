"""M4.3/M4.4 bench commands end to end on synthetic run trees (closing review A06/C06)."""

import copy
import csv
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

from pose_estimation import corpus_run
from pose_estimation.bench import confirm, labels

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path):
    records, jobs = [], []
    row = {"frame_idx": "0", "person_idx": "0"}
    sides = {}
    for side, word, x in (("L", "left", 20), ("R", "right", 70)):
        joints = {}
        for name, i in labels.HAND_JOINTS.items():
            xy = [x + i / 2, 40]
            joints[name] = {"state": "vis", "xy": xy}
            row.update(
                {
                    f"{word}_hand_{i}_x": str(xy[0] / 100),
                    f"{word}_hand_{i}_y": str(xy[1] / 100),
                    f"{word}_hand_{i}_conf": "0.9",
                }
            )
        body = {}
        for name, y in (("shoulder", 10), ("elbow", 25), ("wrist", 40)):
            body[name] = {"state": "vis", "xy": [x, y]}
            row.update(
                {
                    f"body_{word}_{name}_x": str(x / 100),
                    f"body_{word}_{name}_y": str(y / 100),
                    f"body_{word}_{name}_vis": "0.9",
                }
            )
        sides[side] = {"hand": "vis", "extent": [x, 39, x + 10, 41], "joints": joints, "body": body}
    for i in range(2):
        event = f"synthetic-event-{i}"
        key = f"{event}/left#0"
        record = {
            "frame": {"key": key, "index": 0, "width": 100, "height": 100},
            "labeller": "synthetic",
            "task": "task",
            "patient": [0, 0, 100, 100],
            "sides": copy.deepcopy(sides),
        }
        assert labels.validate(record) == []
        records.append(record)
        jobs.append(
            {
                "event_id": event,
                "camera_name": "left",
                "frame": 0,
                "subject": i,
                "stratum": "main",
                "view": "left",
            }
        )
    label_path, job_path = tmp_path / "labels.jsonl", tmp_path / "jobs.json"
    labels.write(label_path, records)
    job_path.write_text(json.dumps(jobs))
    trees = {}
    for arm in ("rtmw-l-384", "rtmw-m-256"):
        tree = tmp_path / f"PRIVATE_SOURCE_LOCATION_{arm}"
        trees[arm] = tree
        for job in jobs:
            target = tree / job["event_id"] / "left.csv"
            target.parent.mkdir(parents=True)
            with target.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            _complete(target.parent, arm)
    return label_path, job_path, records, row, trees


def _complete(event: Path, arm: str, model: str | None = None) -> None:
    """What a finished corpus event carries beside its CSVs: the marker + its generation."""
    corpus_run.write_marker(event, status=corpus_run.MARKER_COMPLETE, stage="clinical")
    corpus_run.write_pose_config(
        event, {"model": model or arm, "calibration": corpus_run.calibration_digest(arm)}
    )


def test_a06_confirm_command_scores_each_arm_on_the_same_records(tmp_path):
    label_path, job_path, records, row, trees = _fixture(tmp_path)
    # Positive control: the real per-frame scorer + bootstrap accept this exact synthetic table.
    scored = confirm.subject_means(
        [confirm.score_frame(record, row, str(i)) for i, record in enumerate(records)]
    )
    for endpoint in confirm.MARGINS:
        assert confirm.bootstrap(scored, scored, endpoint, n=10) == (0.0, 0.0, 0.0)
    contract = tmp_path / "contract.md"
    contract.write_text("## 8. Amendments\n\n- **A01 (candidates).** `rtmw-l-384`.\n")
    out = tmp_path / "confirmation.json"
    main = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    assert (
        main(
            [
                "confirm",
                "--contract",
                str(contract),
                "--labels",
                str(label_path),
                "--jobs",
                str(job_path),
                "--public-rank",
                "rtmw-l-384",
                "--out",
                str(out),
                *[arg for arm, tree in trees.items() for arg in ("--from-run", f"{arm}={tree}")],
            ]
        )
        == 0
    )
    report = json.loads(out.read_text())
    assert report["frames"] == 2
    assert report["subjects"] == 2
    assert report["states"] == {"rtmw-l-384": "non-inferior"}


def test_a06_presence_stdout_contains_no_source_tree_path(tmp_path, capsys):
    label_path, job_path, _, _, trees = _fixture(tmp_path)
    out = tmp_path / "results"
    (out / "rtmw-l-384").mkdir(parents=True)
    main = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    assert (
        main(
            [
                "presence",
                "--labels",
                str(label_path),
                str(label_path),
                "--sample",
                str(job_path),
                "--arm",
                "rtmw-l-384",
                "--from-run",
                f"rtmw-l-384={trees['rtmw-l-384']}",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert json.loads(printed)["subjects"] == 2
    assert "PRIVATE_SOURCE_LOCATION" not in printed


def test_a06_from_run_presence_creates_its_output_directory(tmp_path):
    label_path, job_path, _, _, trees = _fixture(tmp_path)
    out = tmp_path / "new-results"
    main = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    assert (
        main(
            [
                "presence",
                "--labels",
                str(label_path),
                str(label_path),
                "--sample",
                str(job_path),
                "--arm",
                "rtmw-l-384",
                "--from-run",
                f"rtmw-l-384={trees['rtmw-l-384']}",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "rtmw-l-384" / "presence.json").is_file()


def test_f05_two_candidates_and_baseline_reuse_all_records(tmp_path, monkeypatch):
    from pose_estimation.bench import confirm

    label_path, jobs, _, _, trees = _fixture(tmp_path)
    trees["rtmw-x-384"] = Path(shutil.copytree(trees["rtmw-l-384"], tmp_path / "x384"))
    for event in trees["rtmw-x-384"].iterdir():
        _complete(event, "rtmw-x-384")
    contract = tmp_path / "contract.md"
    contract.write_text(
        "## 8. Amendments\n- **A01 (candidates).** `rtmw-l-384` and `rtmw-x-384`.\n"
    )
    cli = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    calls = []
    original = confirm.score_frame

    def score(record, row, subject):
        calls.append((record["frame"]["key"], subject))
        return original(record, row, subject)

    monkeypatch.setattr(confirm, "score_frame", score)
    out = tmp_path / "new-parent" / "confirm.json"
    assert (
        cli(
            [
                "confirm",
                "--contract",
                str(contract),
                "--labels",
                str(label_path),
                "--jobs",
                str(jobs),
                "--public-rank",
                "rtmw-l-384",
                "rtmw-x-384",
                "--out",
                str(out),
                *[x for arm, tree in trees.items() for x in ("--from-run", f"{arm}={tree}")],
            ]
        )
        == 0
    )
    report = json.loads(out.read_text())
    assert report["states"] == {"rtmw-l-384": "non-inferior", "rtmw-x-384": "non-inferior"}
    assert report["frames"] == report["subjects"] == 2
    assert len(calls) == 6
    assert calls[:2] == calls[2:4] == calls[4:]


@pytest.mark.parametrize("command", ["presence", "pilot-eval"])
def test_f07_replay_commands_keep_runner_output_out_of_stdout(
    tmp_path, monkeypatch, capsys, command
):
    from pose_estimation import run, subject_tracker
    from test_m2u92_keypoint_hygiene import _Capture

    label_path, jobs_path, _, _, _ = _fixture(tmp_path)
    jobs = json.loads(jobs_path.read_text())
    run_root = tmp_path / "run"
    for job in jobs:
        diag = run_root / job["event_id"] / "left_diag.csv"
        diag.parent.mkdir(parents=True)
        diag.write_text("n_frames_decoded\n1\n")
    cli = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"].__globals__
    media = tmp_path / "PRIVATE_REPLAY_SOURCE_CANARY.avi"
    media.touch()
    monkeypatch.setitem(cli, "_media", lambda *args: media)
    monkeypatch.setattr(run, "open_capture", lambda *args, **kwargs: _Capture([(48, 64)]))
    # Device boundary only: keep the real CLI, replay wrapper, process_source, CSV writer,
    # source banner and final summary. One decoded synthetic frame with no detections.
    monkeypatch.setattr(run, "_patch_rtmlib_openvino", lambda: None)

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def __call__(self, frame):
            return np.empty((0, 133, 2)), np.empty((0, 133))

    monkeypatch.setattr(subject_tracker, "SubjectTracker", Tracker)
    assert (
        cli["main"](
            [
                command,
                "--labels",
                str(label_path),
                str(label_path),
                "--sample",
                str(jobs_path),
                "--run",
                str(run_root),
                "--sessions",
                str(tmp_path / "unused-sessions"),
                "--arm",
                "rtmw-l-384",
                "--out",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )
    emitted = capsys.readouterr().out
    assert '"subjects": 2' in emitted
    assert "PRIVATE_REPLAY_SOURCE_CANARY" not in emitted, (
        "replay source banner escaped aggregate summary"
    )


@pytest.mark.parametrize("command", ["presence", "pilot-eval", "confirm"])
def test_from_run_tree_missing_a_requested_clip_refuses(tmp_path, command):
    """A clip absent from a run tree is no export to score: a failed or unfinished event would
    read as the arm missing every point — a silent penalty, on one-use labels for `confirm`."""
    label_path, job_path, _, _, trees = _fixture(tmp_path)
    contract = tmp_path / "contract.md"
    contract.write_text("## 8. Amendments\n\n- **A01 (candidates).** `rtmw-l-384`.\n")
    runs = [x for arm, tree in trees.items() for x in ("--from-run", f"{arm}={tree}")]
    pair = ["--labels", str(label_path), str(label_path), "--sample", str(job_path)]
    arm = ["--arm", "rtmw-l-384"]

    def argv(out: Path) -> list[str]:
        return {
            "presence": ["presence", *pair, *arm, "--out", str(out)],
            "pilot-eval": ["pilot-eval", *pair, *arm, "--n", "10", "--out", str(out)],
            "confirm": ["confirm", "--contract", str(contract), "--labels", str(label_path),
                        "--jobs", str(job_path), "--public-rank", "rtmw-l-384",
                        "--out", str(out / "confirm.json")],
        }[command] + runs  # fmt: skip

    main = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    assert main(argv(tmp_path / "complete")) == 0  # positive control: complete trees score
    (trees["rtmw-l-384"] / "synthetic-event-1" / "left.csv").unlink()
    out = tmp_path / "incomplete"
    with pytest.raises(SystemExit) as refused:
        main(argv(out))
    message = str(refused.value)
    assert "rtmw-l-384" in message
    assert "1 of 2" in message
    assert "PRIVATE_SOURCE_LOCATION" not in message
    assert "synthetic-event" not in message
    assert not out.exists()


def test_replay_that_writes_no_landmark_csv_refuses(tmp_path, monkeypatch):
    """A crashed replay leaves no CSV; scoring its frames as empty exports penalises the arm."""
    from pose_estimation.bench import replay

    label_path, job_path, _, _, _ = _fixture(tmp_path)
    run_root = tmp_path / "run"
    for job in json.loads(job_path.read_text()):
        diag = run_root / job["event_id"] / "left_diag.csv"
        diag.parent.mkdir(parents=True)
        diag.write_text("n_frames_decoded\n1\n")
    cli = runpy.run_path(str(ROOT / "scripts/bench.py"))
    monkeypatch.setitem(cli["main"].__globals__, "_media", lambda *_: tmp_path / "clip.mp4")
    # A crashed runner: its window directory exists, its landmark CSV does not.
    monkeypatch.setattr(
        replay,
        "replay_window",
        lambda media, start, stop, out_dir, extra=(): out_dir.mkdir(parents=True),
    )
    out = tmp_path / "results"
    with pytest.raises(SystemExit) as refused:
        cli["main"](
            ["presence", "--labels", str(label_path), str(label_path), "--sample", str(job_path),
             "--arm", "rtmw-l-384", "--run", str(run_root), "--out", str(out)]
        )  # fmt: skip
    assert "rtmw-l-384" in str(refused.value)
    assert not (out / "rtmw-l-384" / "presence.json").exists()


@pytest.mark.parametrize("marker", [None, "failed"])
def test_from_run_refuses_an_unfinished_or_failed_event(tmp_path, marker):
    """A CSV prefix left by a killed or failed event is no whole-clip run: scored, its missing
    tail reads as the arm missing every later point."""
    label_path, job_path, _, _, trees = _fixture(tmp_path)
    event = trees["rtmw-l-384"] / "synthetic-event-1"
    corpus_run.marker_path(event).unlink()
    if marker:
        corpus_run.write_marker(event, status=marker, stage="run", exit_code=1)
    contract = tmp_path / "contract.md"
    contract.write_text("## 8. Amendments\n\n- **A01 (candidates).** `rtmw-l-384`.\n")
    main = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    with pytest.raises(SystemExit) as refused:
        main(
            ["confirm", "--contract", str(contract), "--labels", str(label_path),
             "--jobs", str(job_path), "--public-rank", "rtmw-l-384",
             "--out", str(tmp_path / "out.json"),
             *[x for arm, tree in trees.items() for x in ("--from-run", f"{arm}={tree}")]]
        )  # fmt: skip
    assert "1 of 2 events are not complete" in str(refused.value)
    assert "synthetic-event" not in str(refused.value)


def test_from_run_refuses_a_tree_of_another_generation(tmp_path):
    """Rows a run produced under another model or calibration are not this arm's rows."""
    label_path, job_path, _, _, trees = _fixture(tmp_path)
    _complete(trees["rtmw-m-256"] / "synthetic-event-0", "rtmw-m-256", model="rtmw-x-384")
    contract = tmp_path / "contract.md"
    contract.write_text("## 8. Amendments\n\n- **A01 (candidates).** `rtmw-l-384`.\n")
    main = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    with pytest.raises(SystemExit) as refused:
        main(
            ["confirm", "--contract", str(contract), "--labels", str(label_path),
             "--jobs", str(job_path), "--public-rank", "rtmw-l-384",
             "--out", str(tmp_path / "out.json"),
             *[x for arm, tree in trees.items() for x in ("--from-run", f"{arm}={tree}")]]
        )  # fmt: skip
    assert "rtmw-m-256" in str(refused.value)
    assert "1 of 2 were produced under another model or calibration" in str(refused.value)


def test_confirm_refuses_an_empty_candidate_amendment_before_reading_labels(tmp_path, monkeypatch):
    """M4.4 P04: an A01 heading naming nobody is no frozen candidate list."""
    label_path, job_path, _, _, trees = _fixture(tmp_path)
    contract = tmp_path / "contract.md"
    contract.write_text("## 8. Amendments\n- **A01 (candidates).**\n")

    def forbidden(_):
        pytest.fail("one-use labels read with no frozen candidate")

    monkeypatch.setattr(labels, "load", forbidden)
    main = runpy.run_path(str(ROOT / "scripts/bench.py"))["main"]
    with pytest.raises(SystemExit, match="names no candidates"):
        main(
            ["confirm", "--contract", str(contract), "--labels", str(label_path),
             "--jobs", str(job_path), "--public-rank", "rtmw-l-384",
             "--out", str(tmp_path / "out.json"),
             *[x for arm, tree in trees.items() for x in ("--from-run", f"{arm}={tree}")]]
        )  # fmt: skip


def test_pilot_cache_replays_after_a_hand_gate_refit(tmp_path, monkeypatch):
    """M4.3b refit → a new generation: the replay cache must not serve the old rows."""
    from pose_estimation import run
    from pose_estimation.bench import replay

    label_path, job_path, _, row, _ = _fixture(tmp_path)
    run_root = tmp_path / "run"
    for job in json.loads(job_path.read_text()):
        diag = run_root / job["event_id"] / "left_diag.csv"
        diag.parent.mkdir(parents=True)
        diag.write_text("n_frames_decoded\n1\n")
    cli = runpy.run_path(str(ROOT / "scripts/bench.py"))
    monkeypatch.setitem(cli["main"].__globals__, "_media", lambda *_: tmp_path / "left.mp4")
    calls = []

    def fake_replay(media, start, stop, out_dir, extra=()):
        calls.append(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "left.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    monkeypatch.setattr(replay, "replay_window", fake_replay)
    argv = ["presence", "--labels", str(label_path), str(label_path), "--sample", str(job_path),
            "--arm", "rtmw-m-256", "--run", str(run_root), "--out", str(tmp_path / "out")]  # fmt: skip
    assert cli["main"](argv) == 0
    assert len(calls) == 2  # one whole-clip replay per pilot clip
    assert cli["main"](argv) == 0
    assert len(calls) == 2  # same configuration → the finished cache is reused
    entry = run.MODEL_REGISTRY["rtmw-m-256"]
    monkeypatch.setitem(run.MODEL_REGISTRY, "rtmw-m-256", entry | {"hand_gate": (0.85, 0.8)})
    assert cli["main"](argv) == 0
    assert len(calls) == 4, "a hand-gate refit reused the pre-refit replay rows"
