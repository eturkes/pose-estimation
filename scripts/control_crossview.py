"""Positive control: does the matcher work at all, and how far does the baseline have to open?

Cross-view matching returned zero mutual matches on every smoke pair. A broken matcher and a real
wide-baseline null produce identical bytes, so ladder the baseline from same-frame upward:

  same frame          identity. Any failure here is a matcher bug.
  same asset, +2 frac same camera, seconds apart. Failure = decode/orientation bug.
  cross-view          the quantity under test.
"""

import csv
import importlib.util
import sys
from collections import defaultdict

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("cv2p", "scripts/probe_crossview_pose.py")
p = importlib.util.module_from_spec(spec)
sys.modules["cv2p"] = p
spec.loader.exec_module(p)

with p.INVENTORY.open(newline="", encoding="utf-8") as stream:
    rows = [r for r in csv.DictReader(stream) if r["disposition"] == "canonical"]
by_cap = defaultdict(list)
for r in rows:
    by_cap[r["capture_id"]].append(r)
cap = next(c for c in sorted(by_cap) if len({m["view"] for m in by_cap[c]}) == 3)
members = sorted(by_cap[cap], key=lambda r: r["asset_id"])
print(f"family={cap} views={[m['view'] for m in members]}")

det = cv2.SIFT.create(nfeatures=p.SIFT_FEATURES, contrastThreshold=0.01)
matcher = cv2.BFMatcher(cv2.NORM_L2)

frames = {}
for m in members:
    got, model = p.sample_frames(p.CORPUS / m["source_path"], int(m["reported_rotation_deg"]))
    frames[m["view"]] = got
    shapes = [g.shape for g, _ in got]
    means = [float(np.mean(g)) for g, _ in got]
    kp = [len(det.detectAndCompute(g, None)[0]) for g, _ in got]
    print(
        f"  {m['view']:6} model={model!r:28} n={len(got)} shapes={shapes} mean_px={[round(v, 1) for v in means]} sift_kp={kp}"
    )


def count(a, b, label):
    got = p._match(det, matcher, a, b)
    n = 0 if got is None else len(got[0])
    # _match returns None under MIN_MATCHES, so re-derive the raw count for the control
    _, da = det.detectAndCompute(a, None)
    _, db = det.detectAndCompute(b, None)
    raw = 0
    if da is not None and db is not None:
        fwd = {
            q[0].queryIdx: q[0].trainIdx
            for q in matcher.knnMatch(da, db, k=2)
            if len(q) == 2 and q[0].distance < p.LOWE_RATIO * q[1].distance
        }
        rev = {
            q[0].queryIdx: q[0].trainIdx
            for q in matcher.knnMatch(db, da, k=2)
            if len(q) == 2 and q[0].distance < p.LOWE_RATIO * q[1].distance
        }
        raw = sum(1 for s, t in fwd.items() if rev.get(t) == s)
    print(f"  {label:36} mutual={raw:5d}  passed_min({p.MIN_MATCHES})={n}")


views = list(frames)
print("\nladder")
v0 = views[0]
count(frames[v0][0][0], frames[v0][0][0], f"same frame ({v0}[0] vs itself)")
count(frames[v0][0][0], frames[v0][1][0], f"same asset ({v0}[0] vs {v0}[1])")
count(frames[v0][0][0], frames[v0][3][0], f"same asset far ({v0}[0] vs {v0}[3])")
for other in views[1:]:
    count(frames[v0][0][0], frames[other][0][0], f"cross-view ({v0}[0] vs {other}[0])")
    count(frames[v0][1][0], frames[other][1][0], f"cross-view ({v0}[1] vs {other}[1])")
