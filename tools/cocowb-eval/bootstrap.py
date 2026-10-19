"""Image-bootstrap intervals for COCO-WholeBody part AP, paired across arms (contract M4.2 D03).

Usage: uv run --directory tools/cocowb-eval python bootstrap.py GT.json NAME=RESULTS.json ...
       [--parts lefthand righthand] [--n 1000] [--seed 20261019] → JSON on stdout.

COCOeval matches detections to GT once per image, and resampling images never changes a match,
so each arm-part pair runs `evaluate()` once and every resample re-runs only the accumulation over
the drawn images (a drawn image's detections repeat as blocks, copies adjacent). `accumulate_ap`
on the full image set must equal `COCOeval.stats[0]` — refused otherwise. Every arm sees the same
draws, so `pairs` holds the paired difference of the parts' mean AP.
"""

import argparse
import contextlib
import io
import itertools
import json
import warnings

import numpy as np
import score
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


def per_image(coco, det, part):
    """COCOeval's own matches for one part: area `all`, maxDets 20, one record per image."""
    sigmas = score.SIGMAS[
        score.CUTS[score.PARTS.index(part)] : score.CUTS[score.PARTS.index(part) + 1]
    ]
    ev = COCOeval(coco, det, f"keypoints_{part}", sigmas, use_area=True)
    ev.params.useSegm = None
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    p = ev.params
    a0 = [tuple(a) for a in p.areaRng].index((0**2, 1e5**2))
    n_img = len(p.imgIds)
    records = []
    for e in ev.evalImgs[a0 * n_img : (a0 + 1) * n_img]:
        if e is None:
            records.append(None)
            continue
        m = p.maxDets[-1]
        matched = e["dtMatches"][:, :m] >= 0
        ignored = e["dtIgnore"][:, :m].astype(bool)
        records.append(
            (
                np.asarray(e["dtScores"][:m], dtype=float),
                matched & ~ignored,
                ~matched & ~ignored,
                int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)),
            )
        )
    return records, float(ev.stats[0]), np.asarray(p.recThrs)


def flatten(records):
    keep = [(i, r) for i, r in enumerate(records) if r is not None]
    lengths = np.zeros(len(records), int)
    for i, r in keep:
        lengths[i] = len(r[0])
    return {
        "scores": np.concatenate([r[0] for _, r in keep]),
        "tp": np.concatenate([r[1] for _, r in keep], axis=1),
        "fp": np.concatenate([r[2] for _, r in keep], axis=1),
        "start": np.cumsum(lengths) - lengths,
        "length": lengths,
        "npig": np.bincount([i for i, _ in keep], [r[3] for _, r in keep], len(records)),
    }


def accumulate_ap(flat, rec_thrs, counts):
    """COCOeval.accumulate + stats[0] over images drawn `counts` times (all ones = the full set).

    A drawn image's detections repeat as whole blocks, copies adjacent in image order — exactly
    the concatenation COCOeval sees for a GT listing each image's copies consecutively — so the
    stable score sort breaks ties as COCOeval does (per-detection weights would not).
    """
    npig = float(np.dot(counts, flat["npig"]))
    if npig == 0:
        return -1.0
    images = np.repeat(np.arange(len(counts)), counts)
    lengths = flat["length"][images]
    offsets = np.cumsum(lengths) - lengths
    idx = np.repeat(flat["start"][images] - offsets, lengths) + np.arange(lengths.sum())
    order = idx[np.argsort(-flat["scores"][idx], kind="mergesort")]
    out = []
    for tp, fp in zip(flat["tp"][:, order], flat["fp"][:, order], strict=True):
        tp_sum = np.cumsum(tp, dtype=float)
        fp_sum = np.cumsum(fp, dtype=float)
        rc = tp_sum / npig
        pr = np.maximum.accumulate((tp_sum / (fp_sum + tp_sum + np.spacing(1)))[::-1])[::-1]
        q_idx = np.searchsorted(rc, rec_thrs, side="left")
        q = np.zeros(len(rec_thrs))
        inside = q_idx < len(pr)
        # COCOeval fills q until the first threshold past the curve's end, then stops.
        stop = len(q_idx) if inside.all() else int(np.argmin(inside))
        q[:stop] = pr[q_idx[:stop]]
        out.append(q)
    return float(np.mean(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("arms", nargs="+", help="NAME=RESULTS.json")
    ap.add_argument("--parts", nargs="+", default=["lefthand", "righthand"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*_score' not found")
    flats, full = {}, {}
    rec_thrs = None
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(a.gt)
        n_img = len(coco.getImgIds())
        for spec in a.arms:
            name, path = spec.split("=", 1)
            det = coco.loadRes(path)
            for part in a.parts:
                records, stat, rec_thrs = per_image(coco, det, part)
                if len(records) != n_img:
                    raise SystemExit(f"{name} {part}: {len(records)} records for {n_img} images")
                flats[name, part] = flatten(records)
                full[name, part] = accumulate_ap(flats[name, part], rec_thrs, np.ones(n_img, int))
                if abs(full[name, part] - stat) > 1e-12:
                    raise SystemExit(f"{name} {part}: accumulation {full[name, part]} != {stat}")
    names = [s.split("=", 1)[0] for s in a.arms]
    rng = np.random.default_rng(a.seed)
    boots = {name: [] for name in names}
    for _ in range(a.n):
        counts = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
        for name in names:
            boots[name].append(
                np.mean([accumulate_ap(flats[name, p], rec_thrs, counts) for p in a.parts])
            )

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    report = {
        "images": n_img,
        "resamples": a.n,
        "seed": a.seed,
        "parts": a.parts,
        "arms": {
            name: {
                "per_part": {p: full[name, p] for p in a.parts},
                "mean": float(np.mean([full[name, p] for p in a.parts])),
                "interval": interval(boots[name]),
            }
            for name in names
        },
        "pairs": {
            f"{x}-{y}": {
                "difference": float(
                    np.mean([full[x, p] for p in a.parts]) - np.mean([full[y, p] for p in a.parts])
                ),
                "interval": interval(np.subtract(boots[x], boots[y])),
            }
            for x, y in itertools.permutations(names, 2)
        },
    }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
