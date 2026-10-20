"""Seeded COCO-WholeBody person subsets: one draw per seed for every arm, scored against that
draw's GT alone, cached apart from the full set; `screen` refuses arms drawn differently."""

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pose_estimation.bench import arms

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench.py"


def _coco(tmp_path: Path, n: int) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    body = np.tile([100, 100, 2.0], (17, 1))
    body[5, 2] = 1
    image_rows, rows = [], []
    for i in range(n):
        cv2.imwrite(str(images / f"{i}.png"), np.zeros((240, 320, 3), np.uint8))
        image_rows.append({"id": i, "file_name": f"{i}.png"})
        rows.append(
            {
                "id": 100 + i,
                "image_id": i,
                "bbox": [20 + i, 30, 100, 150],
                "area": 15000,
                "keypoints": body.ravel().tolist(),
                "lefthand_valid": True,
                "lefthand_box": [90, 90, 20, 20],
                "lefthand_kpts": np.tile([100, 100, 2.0], 21).tolist(),
                "iscrowd": 0,
            }
        )
    # a degenerate box never enters the eligible population
    rows.append({**rows[0], "id": 999, "bbox": [0, 0, 1, 40]})
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({"images": image_rows, "annotations": rows, "categories": [{"id": 1}]})
    )
    return images, annotations


def _patch(monkeypatch, calls, scored, predicted):
    def predict(model, image, boxes):
        calls.append(len(boxes))
        return np.zeros((len(boxes), 133, 2)), np.ones((len(boxes), 133))

    def evaluate(args, **kwargs):
        gt = json.loads(Path(args[-2]).read_text())
        scored.append(sorted(r["id"] for r in gt["annotations"]))
        predicted.append(len(json.loads(Path(args[-1]).read_text())))
        assert {r["image_id"] for r in gt["annotations"]} == {r["id"] for r in gt["images"]}
        return SimpleNamespace(stdout="{}")

    monkeypatch.setattr(arms, "load", lambda *args: None)
    monkeypatch.setattr(arms, "predict", predict)
    monkeypatch.setattr("subprocess.run", evaluate)


def test_subset_draw_gt_and_cache(monkeypatch, tmp_path):
    cli = runpy.run_path(str(SCRIPT))
    images, annotations = _coco(tmp_path, 10)
    calls, scored, predicted = [], [], []
    _patch(monkeypatch, calls, scored, predicted)
    a = argparse.Namespace(
        coco_wb=annotations,
        images=images,
        arm="rtmw-l-384",
        out=tmp_path / "out",
        device="GPU",
        limit=4,
        seed=17,
    )
    assert cli["cmd_public_eval"](a) == 0
    want = sorted(100 + i for i in np.random.default_rng(17).permutation(10)[:4])
    assert scored[-1] == want
    assert predicted[-1] == 4
    assert calls == [1, 1, 1, 1]
    run = a.out / "rtmw-l-384" / "subset-4-seed-17"
    summary = json.loads((run / "coco_wb_summary.json").read_text())
    assert (summary["persons"], summary["eligible_persons"], summary["subset_seed"]) == (4, 10, 17)
    # cached predictions: a rerun scores again without inferring
    assert cli["cmd_public_eval"](a) == 0
    assert len(calls) == 4
    # another arm, same seed → the identical GT draw (paired)
    a.arm = "rtmw-m-256"
    assert cli["cmd_public_eval"](a) == 0
    other = a.out / "rtmw-m-256" / "subset-4-seed-17" / "coco_wb_gt.json"
    assert other.read_bytes() == (run / "coco_wb_gt.json").read_bytes()
    # another seed → another directory + draw; the full set keeps the top-level cache
    a.seed = 19
    assert cli["cmd_public_eval"](a) == 0
    assert scored[-1] == sorted(100 + i for i in np.random.default_rng(19).permutation(10)[:4])
    a.limit = 0
    assert cli["cmd_public_eval"](a) == 0
    assert (scored[-1], predicted[-1]) == ([*range(100, 110), 999], 10)
    assert (a.out / "rtmw-m-256" / "coco_wb_pred.npz").is_file()


def test_screen_refuses_unpaired_subsets(monkeypatch, tmp_path):
    cli = runpy.run_path(str(SCRIPT))
    images, annotations = _coco(tmp_path, 10)
    _patch(monkeypatch, [], [], [])
    out = tmp_path / "out"
    for arm, seed in (("rtmw-l-384", 17), ("rtmw-m-256", 19)):
        a = argparse.Namespace(
            coco_wb=annotations, images=images, arm=arm, out=out, device="GPU", limit=4, seed=seed
        )
        assert cli["cmd_public_eval"](a) == 0
    # the second arm's seed-17 directory carries the seed-19 draw
    src = out / "rtmw-m-256" / "subset-4-seed-19"
    src.rename(out / "rtmw-m-256" / "subset-4-seed-17")
    s = argparse.Namespace(
        coco_wb=annotations,
        public=out,
        arms=["rtmw-l-384", "rtmw-m-256"],
        coco_limit=4,
        coco_seed=17,
        dexycb=tmp_path,
        handsets=tmp_path,
        hand_seed=1,
        baseline="rtmw-m-256",
        n=10,
        seed=1,
        out=tmp_path / "screen.json",
    )
    with pytest.raises(SystemExit, match="not one paired draw"):
        cli["cmd_screen"](s)
