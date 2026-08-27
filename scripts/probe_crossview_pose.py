"""Can relative pose be recovered between two views of one capture family? (R2/R4, contract U3/U4)

Rigidity says a camera held still; it does not say two cameras share enough scene to calibrate.
M2.6 exists only if cross-view extrinsics are recoverable, and that was never measured.

Extrinsics come from the static background, so temporal alignment is irrelevant here: any frame of A
matches any frame of B. Two verdicts per pair, because the intrinsics prior is itself uncertain
(res1:U2 gives per-model fx with an unreported readout/stabilisation factor):

  F-inliers   intrinsics-free. Does a consistent epipolar geometry exist at all?
  E-pose      needs the prior. Recovered R across independent frame samples must agree, since a
              single essential-matrix fit always returns something and only cross-sample agreement
              separates real geometry from a noise fit.

Prints aggregates only. Decoded frames never leave the process.
"""

import csv
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import numpy as np

CORPUS = Path("videos/3-cam")
INVENTORY = Path("inventory/assets.csv")
OUT = Path(".scratch/m2u3/crossview_pairs.csv")

SAMPLE_FRACTIONS = (0.15, 0.40, 0.65, 0.85)
ANALYSIS_MAX_DIM = 960
SIFT_FEATURES = 4000
LOWE_RATIO = 0.78
# The algebraic minimum for a fundamental matrix, not a quality bar: the point is to record the
# match count that cross-view baselines actually produce, so a collapse reads as a number.
MIN_MATCHES = 8
F_THRESHOLD_PX = 3.0
E_THRESHOLD_PX = 3.0
MIN_F_INLIERS = 30
ROT_AGREE_DEG = 10.0
SEED = 20260827

# res1:U2 per-model focal priors, in native 1920-wide pixels. The readout/stabilisation factor is
# unreported, so E-pose carries prior uncertainty that F-inliers do not.
FX_PRIOR = {"iPad (5th generation)": 1873.3, "iPad Air 11-inch (M2)": 1553.2}


@dataclass
class PairResult:
    capture_id: str
    asset_a: str
    asset_b: str
    view_a: str
    view_b: str
    config_a: str
    config_b: str
    samples: int
    matches_median: float
    f_inliers_median: float
    f_inlier_ratio_median: float
    e_inliers_median: float
    rot_spread_deg: float
    baseline_angle_deg: float
    parallax_median_deg: float
    verdict: str


def _rotate(gray, rotation_deg):
    if rotation_deg == 90:
        return np.ascontiguousarray(np.rot90(gray, -1))
    if rotation_deg == 180:
        return np.ascontiguousarray(np.rot90(gray, 2))
    if rotation_deg == 270:
        return np.ascontiguousarray(np.rot90(gray, 1))
    return np.ascontiguousarray(gray)


def sample_frames(path: Path, rotation_deg: int, fractions=SAMPLE_FRACTIONS):
    """Seek to each fraction of the clip and take one frame. Returns [(gray, scale, model)]."""
    out = []
    with av.open(str(path)) as container:
        model = container.metadata.get("com.apple.quicktime.model") or ""
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if duration is None or duration <= 0:
            return [], model
        for fraction in fractions:
            target = duration * fraction
            try:
                container.seek(int(target / stream.time_base), stream=stream)
                frame = next(container.decode(stream))
            except (StopIteration, av.AVError, ValueError):
                continue
            display = _rotate(frame.to_ndarray(format="gray"), rotation_deg)
            h, w = display.shape
            scale = min(1.0, ANALYSIS_MAX_DIM / max(w, h))
            resized = cv2.resize(
                display,
                (max(32, round(w * scale)), max(32, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            # native-pixel focal must be expressed in the analysis frame it is used on
            out.append((resized, w / resized.shape[1]))
    return out, model


def _intrinsics(model: str, shape, native_over_analysis: float):
    fx_native = FX_PRIOR.get(model)
    if fx_native is None:
        return None
    fx = fx_native / native_over_analysis
    h, w = shape
    return np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _match(det, matcher, a, b):
    ka, da = det.detectAndCompute(a, None)
    kb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < MIN_MATCHES or len(kb) < MIN_MATCHES:
        return None
    fwd = {
        p[0].queryIdx: p[0].trainIdx
        for p in matcher.knnMatch(da, db, k=2)
        if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance
    }
    rev = {
        p[0].queryIdx: p[0].trainIdx
        for p in matcher.knnMatch(db, da, k=2)
        if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance
    }
    mutual = [(s, t) for s, t in fwd.items() if rev.get(t) == s]
    if len(mutual) < MIN_MATCHES:
        return None
    pa = np.asarray([ka[i].pt for i, _ in mutual], dtype=np.float64)
    pb = np.asarray([kb[j].pt for _, j in mutual], dtype=np.float64)
    return pa, pb


def _rotation_angle(r1, r2):
    delta = r1 @ r2.T
    cos = (np.trace(delta) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _parallax(pa, pb, ka, kb, rot, trans):
    """Median angle between back-projected rays, the quantity triangulation.py gates at 1 deg."""
    na = np.linalg.inv(ka) @ np.vstack([pa.T, np.ones(len(pa))])
    nb = np.linalg.inv(kb) @ np.vstack([pb.T, np.ones(len(pb))])
    na /= np.linalg.norm(na, axis=0)
    nb = rot.T @ (nb / np.linalg.norm(nb, axis=0))
    cos = np.clip(np.sum(na * nb, axis=0), -1.0, 1.0)
    return float(np.median(np.degrees(np.arccos(cos))))


def analyze_pair(task):
    (cap, a_id, a_view, a_path, a_rot, b_id, b_view, b_path, b_rot) = task
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SEED)
    det = cv2.SIFT.create(nfeatures=SIFT_FEATURES, contrastThreshold=0.01)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    frames_a, model_a = sample_frames(CORPUS / a_path, a_rot)
    frames_b, model_b = sample_frames(CORPUS / b_path, b_rot)
    n = min(len(frames_a), len(frames_b))
    matches, f_in, f_ratio, e_in, rots, parallax = [], [], [], [], [], []
    for i in range(n):
        (ga, sa), (gb, sb) = frames_a[i], frames_b[i]
        got = _match(det, matcher, ga, gb)
        if got is None:
            continue
        pa, pb = got
        matches.append(len(pa))
        fund, fmask = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, F_THRESHOLD_PX, 0.999, 10_000)
        if fund is None or fmask is None:
            continue
        inliers = int(fmask.sum())
        f_in.append(inliers)
        f_ratio.append(inliers / len(pa))
        ka = _intrinsics(model_a, ga.shape, sa)
        kb = _intrinsics(model_b, gb.shape, sb)
        if ka is None or kb is None or inliers < MIN_F_INLIERS:
            continue
        # Different intrinsics per camera: normalize each set with its own K, then E on unit focal.
        norm_a = (np.linalg.inv(ka) @ np.vstack([pa.T, np.ones(len(pa))]))[:2].T
        norm_b = (np.linalg.inv(kb) @ np.vstack([pb.T, np.ones(len(pb))]))[:2].T
        thr = E_THRESHOLD_PX / ((ka[0, 0] + kb[0, 0]) / 2.0)
        ess, emask = cv2.findEssentialMat(norm_a, norm_b, np.eye(3), cv2.USAC_MAGSAC, 0.999, thr)
        if ess is None or ess.shape != (3, 3) or emask is None:
            continue
        count, rot, trans, _ = cv2.recoverPose(ess, norm_a, norm_b, np.eye(3), mask=emask.copy())
        if count < MIN_F_INLIERS:
            continue
        e_in.append(count)
        rots.append(rot)
        keep = emask.ravel().astype(bool)
        parallax.append(_parallax(pa[keep], pb[keep], ka, kb, rot, trans))

    def med(v):
        return float(np.median(v)) if v else math.nan

    spread = math.nan
    if len(rots) >= 2:
        spread = max(
            _rotation_angle(rots[i], rots[j])
            for i in range(len(rots))
            for j in range(i + 1, len(rots))
        )
    baseline = math.nan
    if rots:
        baseline = _rotation_angle(rots[0], np.eye(3))
    if not f_in:
        verdict = "no_correspondence"
    elif med(f_in) < MIN_F_INLIERS:
        verdict = "weak_correspondence"
    elif not e_in:
        verdict = "no_pose"
    elif len(rots) < 2:
        verdict = "single_sample_pose"
    elif spread <= ROT_AGREE_DEG:
        verdict = "recoverable"
    else:
        verdict = "inconsistent_pose"
    return PairResult(
        cap,
        a_id,
        b_id,
        a_view,
        b_view,
        model_a,
        model_b,
        n,
        med(matches),
        med(f_in),
        med(f_ratio),
        med(e_in),
        spread,
        baseline,
        med(parallax),
        verdict,
    )


def main() -> None:
    with INVENTORY.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["disposition"] == "canonical"]
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r["capture_id"]].append(r)
    tasks = []
    for cap in sorted(by_cap):
        members = sorted(by_cap[cap], key=lambda r: r["asset_id"])
        tasks.extend(
            (
                cap,
                a["asset_id"],
                a["view"],
                a["source_path"],
                int(a["reported_rotation_deg"]),
                b["asset_id"],
                b["view"],
                b["source_path"],
                int(b["reported_rotation_deg"]),
            )
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        )
    print(
        f"pairs={len(tasks)} families={sum(1 for c in by_cap.values() if len(c) > 1)}", flush=True
    )
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, res in enumerate(pool.map(analyze_pair, tasks), 1):
            results.append(res)
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)} {time.perf_counter() - started:.0f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    verdicts = defaultdict(int)
    for r in results:
        verdicts[r.verdict] += 1
    print(f"\nwall={time.perf_counter() - started:.0f}s  pairs={len(results)}")
    print("verdicts:", json.dumps(dict(sorted(verdicts.items())), indent=None))
    rec = [r for r in results if r.verdict == "recoverable"]
    print(f"recoverable={len(rec)}/{len(results)}")
    for name in (
        "f_inliers_median",
        "f_inlier_ratio_median",
        "rot_spread_deg",
        "parallax_median_deg",
    ):
        v = sorted(getattr(r, name) for r in rec if math.isfinite(getattr(r, name)))
        if v:
            print(f"  {name:24} min={v[0]:8.3f} med={v[len(v) // 2]:8.3f} max={v[-1]:8.3f}")
    fams = defaultdict(set)
    for r in results:
        if r.verdict == "recoverable":
            fams[r.capture_id].add(r.asset_a)
            fams[r.capture_id].add(r.asset_b)
    print(f"families with >=1 recoverable pair: {len(fams)}")
    print(f"families with >=3 cameras joined:   {sum(1 for v in fams.values() if len(v) >= 3)}")
    # cross-view label structure feeds R1
    by_views = defaultdict(lambda: [0, 0])
    for r in results:
        key = "|".join(sorted((r.view_a, r.view_b)))
        by_views[key][0] += r.verdict == "recoverable"
        by_views[key][1] += 1
    print(
        "by view pair (recoverable/total):",
        {k: f"{a}/{b}" for k, (a, b) in sorted(by_views.items())},
    )


if __name__ == "__main__":
    sys.exit(main())
