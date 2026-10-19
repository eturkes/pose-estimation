"""M4.3/M4.4 bench commands end to end on synthetic run trees (closing review A06/C06)."""

import copy
import csv
import json
import runpy
from pathlib import Path

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
    return label_path, job_path, records, row, trees


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
