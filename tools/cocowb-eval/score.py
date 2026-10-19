"""COCO-WholeBody keypoint AP: mmpose's `CocoWholeBodyMetric._do_python_keypoint_eval`, six passes.

Usage: uv run --directory tools/cocowb-eval python score.py GT.json RESULTS.json → JSON stats.
RESULTS = mmpose `results2json` rows: image_id, category_id, keypoints (body 17 x 3), foot_kpts,
face_kpts, lefthand_kpts, righthand_kpts, score.
"""

import contextlib
import io
import json
import sys
import warnings

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

# mmpose configs/_base_/datasets/coco_wholebody.py, 133 sigmas.
SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068, 0.066, 0.066,
    0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031,
    0.025, 0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045,
    0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
    0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017,
    0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010,
    0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009,
    0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
    0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
    0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019,
    0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
    0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
    0.019, 0.022, 0.031,
])  # fmt: skip
CUTS = np.cumsum([0, 17, 6, 68, 21, 21])
PARTS = ("body", "foot", "face", "lefthand", "righthand")


def main(gt_path, res_path):
    # xtcocotools warns that per-part score keys are absent and falls back to `score` — the
    # mmpose result format carries one instance score, so the fallback is the protocol.
    warnings.filterwarnings("ignore", message=".*_score' not found")
    stats = {}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(gt_path)
        det = coco.loadRes(res_path)
        passes = [(p, SIGMAS[CUTS[i] : CUTS[i + 1]]) for i, p in enumerate(PARTS)]
        for part, sigmas in [*passes, ("wholebody", SIGMAS)]:
            ev = COCOeval(coco, det, f"keypoints_{part}", sigmas, use_area=True)
            ev.params.useSegm = None
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            stats[part] = {
                "AP": float(ev.stats[0]),
                "AP50": float(ev.stats[1]),
                "AR": float(ev.stats[5]),
            }
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
