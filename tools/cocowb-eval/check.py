"""Self-check of score.py on synthetic COCO-WholeBody data (gate step `cocowb-eval`).

Fires when the scorer stops separating parts: a perfect result must score AP 1.0 on every part, a
result with one shifted right hand must lose right-hand AP alone, and a left/right hand swap must
lose hand AP. `bootstrap.accumulate_ap` must equal COCOeval on a mixed result over the full image
set, with one image dropped (COCOeval on the reduced set) and with one image drawn twice (COCOeval
on a set holding a renamed copy). rc 0 = all hold.
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import bootstrap
import numpy as np
import score as scorer
from xtcocotools.coco import COCO

rng = np.random.default_rng(7)
PARTS = {
    "keypoints": 17,
    "foot_kpts": 6,
    "face_kpts": 68,
    "lefthand_kpts": 21,
    "righthand_kpts": 21,
}


def person(image_id, ann_id, x0, y0):
    ann = {"id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
           "bbox": [x0, y0, 200, 300], "area": 200 * 300.0, "num_keypoints": 17}  # fmt: skip
    for field, n in PARTS.items():
        xy = np.column_stack([x0 + rng.uniform(10, 190, n), y0 + rng.uniform(10, 290, n)])
        ann[field] = np.column_stack([xy, np.full(n, 2.0)]).ravel().round(2).tolist()
    for side in ("lefthand", "righthand"):
        ann[f"{side}_valid"] = True
        ann[f"{side}_box"] = [x0, y0, 200, 300]
    ann["face_valid"] = ann["foot_valid"] = True
    ann["face_box"] = [x0, y0, 200, 300]
    return ann


def score(gt, results):
    with tempfile.TemporaryDirectory() as tmp:
        g, r = Path(tmp) / "gt.json", Path(tmp) / "res.json"
        g.write_text(json.dumps(gt))
        r.write_text(json.dumps(results))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            scorer.main(str(g), str(r))
        return {part: s["AP"] for part, s in json.loads(out.getvalue()).items()}


def bootstrap_matches(gt, results, counts):
    """(accumulate_ap at `counts`, COCOeval AP on the set those counts describe) per hand part."""
    images = sorted(img["id"] for img in gt["images"])
    drawn = {"images": [], "annotations": [], "categories": gt["categories"]}
    drawn_results = []
    for image_id, count in zip(images, counts, strict=True):
        for copy in range(count):
            # Copies adjacent in id order: COCOeval concatenates images by sorted id, which is
            # the block order `accumulate_ap` reproduces (ties break the same way).
            new_id = image_id * 100 + copy
            drawn["images"].append({"id": new_id, "file_name": f"{new_id}.jpg", "width": 640,
                                    "height": 480})  # fmt: skip
            for ann in gt["annotations"]:
                if ann["image_id"] == image_id:
                    drawn["annotations"].append(
                        ann | {"image_id": new_id, "id": ann["id"] + new_id * 100}
                    )
            drawn_results += [
                r | {"image_id": new_id} for r in results if r["image_id"] == image_id
            ]
    out = {}
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        g, r = Path(tmp) / "gt.json", Path(tmp) / "res.json"
        g.write_text(json.dumps(gt))
        r.write_text(json.dumps(results))
        coco = COCO(str(g))
        det = coco.loadRes(str(r))
        for part in ("lefthand", "righthand"):
            records, _, rec_thrs = bootstrap.per_image(coco, det, part)
            flat = bootstrap.flatten(records)
            out[part] = [bootstrap.accumulate_ap(flat, rec_thrs, np.asarray(counts))]
    reference = score(drawn, drawn_results)
    return {part: (v[0], reference[part]) for part, v in out.items()}


def main():
    anns = [person(i // 2 + 1, i + 1, 50 + 300 * (i % 2), 40) for i in range(6)]
    gt = {
        "images": [
            {"id": i, "file_name": f"{i}.jpg", "width": 640, "height": 480} for i in (1, 2, 3)
        ],
        "annotations": anns,
        "categories": [{"id": 1, "name": "person"}],
    }

    def result(a, shift=0.0, swap=False):
        row = {"image_id": a["image_id"], "category_id": 1, "score": 0.9}
        for field in PARTS:
            src = {"lefthand_kpts": "righthand_kpts", "righthand_kpts": "lefthand_kpts"}.get(
                field, field
            )
            k = np.asarray(a[src if swap else field], dtype=float).reshape(-1, 3)
            if field == "righthand_kpts":
                k[:, 0] += shift
            row[field] = np.column_stack([k[:, :2], np.ones(len(k))]).ravel().tolist()
        return row

    perfect = score(gt, [result(a) for a in anns])
    shifted = score(gt, [result(a, shift=25.0) for a in anns])
    swapped = score(gt, [result(a, swap=True) for a in anns])
    mixed = [
        result(a, shift=25.0 * (i % 3)) | {"score": 0.5 + 0.07 * i} for i, a in enumerate(anns)
    ]
    tied = [result(a, shift=25.0 * (i % 2)) for i, a in enumerate(anns)]  # every score 0.9
    agree = {
        label: bootstrap_matches(gt, rows, counts)
        for label, rows, counts in (
            ("full", mixed, [1, 1, 1]),
            ("dropped", mixed, [1, 0, 1]),
            ("doubled", mixed, [1, 1, 2]),
            ("tied doubled", tied, [2, 1, 1]),
            ("tied tripled", tied, [1, 3, 0]),
        )
    }
    checks = {
        "perfect = 1.0 on every part": all(abs(v - 1.0) < 1e-9 for v in perfect.values()),
        "shifted right hand loses righthand AP alone": shifted["righthand"] < 0.5
        and shifted["lefthand"] == 1.0
        and shifted["body"] == 1.0,
        "swapped hands lose hand AP": swapped["lefthand"] < 0.5 and swapped["righthand"] < 0.5,
        "mixed result is not trivially perfect": agree["full"]["righthand"][1] < 1.0,
    }
    for label, parts in agree.items():
        checks[f"bootstrap accumulation = COCOeval ({label})"] = all(
            abs(mine - ref) < 1e-12 for mine, ref in parts.values()
        )
    for name, ok in checks.items():
        print(f"cocowb-eval check: {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
