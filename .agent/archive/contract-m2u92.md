# Contract — M2.9.2 output hygiene: out-of-frame points, duplicate hands, face landmark sides

Tier `kernel`. Base = the commit carrying this file. Frozen at dispatch; amendments append to §8.
Catalog → `docs/technical/pose-weak-points.md` (C07, C10, the duplicate-hand subset of C05).

## 1. Artifact

- `src/pose_estimation/keypoint_hygiene.py` (new) — `zero_out_of_frame`,
  `suppress_duplicate_hand`, `apply_hygiene`.
- `src/pose_estimation/run.py` — `process_source` applies `apply_hygiene` to the tracker output.
- `src/pose_estimation/mapping.py` — `_COCO_TO_BODY_FACE`.
- `tests/test_mapping.py` — the two face-index pins move with the table (§2 D05).
- `docs/technical/tracking-modes.md` — hygiene + mapping reference.

## 2. Design decisions

**D01 — an out-of-frame keypoint is not an observation.** The top-down model returns every
keypoint inside its crop, and a crop at the frame edge extends past it, so points land outside
the image at visibility >= 0.3. Measured on the shipped tree (379 clips, visibility >= 0.3):
shoulders outside the frame on 42.3 % of above-view observations, elbows 33.8 %, face 67.2 % of
left-view observations. The R stage uses every score > 0 (`OBSERVATION_CONFIDENCE_GATE <- 0`), so
these extrapolations entered elbow flexion, reach and trunk metrics as measurements. A keypoint
whose `x` is outside `[0, width)` or `y` outside `[0, height)` of the decoded frame, or whose
coordinates are non-finite, gets score 0. Coordinates are left as returned.

**D02 — a duplicate hand is the occluded hand drawn on the visible one.** When one hand is hidden,
RTMW places the hidden hand's 21 points on the visible hand at lower confidence. Rule, per person,
over the COCO-WholeBody hands (left 91-111, right 112-132): *present* = finite and score > 0 after
D01. Fires iff >= 10 keypoint indices are present in both hands **and** `overlap < 0.3` **and**
`weaker < 0.5` **and** `weaker < 0.6 × stronger`, where `overlap` = median over the commonly
present indices of `|L_i - R_i|` divided by the larger of the two hands' extents (diagonal of the
bounding box of its present points), and `weaker`/`stronger` = the two hands' mean score over their
own present points. Firing zeroes the weaker hand's 21 scores; the stronger hand is untouched.
Ties (`weaker == stronger`) cannot fire. Measured on the shipped tree: fires on 7.5 % / 12.5 % /
6.9 % of frames with both hands present (above / left / right), concentrated in nut-task clips;
24 random firings sampled, 15 frames in 8 clips viewed: every one shows one visible hand carrying
both skeletons. Genuine bimanual grasps keep both hands above 0.5 and do not fire.

**D03 — order + scope.** `apply_hygiene(keypoints, scores, width, height)` = D01 then D02, per
person row, returning new scores (inputs unmodified). D02 applies only to 133-keypoint rows; D01
applies to any keypoint count. Empty input → empty output of the same shape.

**D04 — wiring.** `process_source` applies `apply_hygiene` to the tracker's output before the
smoother, with the decoded frame's own width and height, for both trackers. The tracker's internal
state (transport, association) sees the raw pose output. The smoother holds a zero-score point's
position and exports it at score 0, so the CSV row carries visibility / confidence 0 there.

**D05 — face landmarks take the subject's own side.** COCO-WholeBody's 68 face points follow the
iBUG-300W layout: 36-41 = the subject's right eye (36 outer, 39 inner), 42-47 = the left eye (42
inner, 45 outer), 48 = the right mouth corner, 54 = the left. `_COCO_TO_BODY_FACE` read them
mirrored. Measured: the face-derived eye points sat on the opposite side of the COCO body eyes in
98-99 % of 11 315 frames (97.6-97.7 % of 7 420 frontal frames), mouth corners 83-90 %. Corrected
table (MediaPipe index ← face sub-index, COCO index = 23 + sub): 1 ← 42, 3 ← 45, 4 ← 39, 6 ← 36,
9 ← 54, 10 ← 48. `tests/test_mapping.py::test_face_derived_mappings` pinned the mirrored indices
(59, 71); it moves to 65 and 77 in this unit, with this measurement as its evidence.

## 3. Predicates

Tester-owned (synthetic arrays + synthetic frames, no media):

- **P01** Out of frame: a keypoint at `x < 0`, `y < 0`, `x >= width` or `y >= height` scores 0;
  `x = 0`, `y = 0` and `x = width - 0.5` keep their score; non-finite coordinates score 0; every
  in-frame score is unchanged; the input arrays are not modified.
- **P02** Duplicate fires: hands overlapping at `overlap < 0.3`, weaker mean 0.3, stronger 0.8 →
  the weaker hand's 21 scores are 0, the stronger's unchanged; left and right each tested as the
  weaker.
- **P03** Duplicate does not fire: each single condition broken alone keeps all scores —
  `overlap >= 0.3` (boundary 0.3 exactly kept), `weaker >= 0.5`, `weaker >= 0.6 × stronger`,
  fewer than 10 commonly present indices, equal means.
- **P04** Order: hand points D01 zeroes are not present for D02 (an out-of-frame duplicate whose
  in-frame remainder has < 10 common indices does not fire).
- **P05** Scope: a 17-keypoint row gets D01 only; a multi-person input is handled per row; empty
  input returns an empty array of the same shape.
- **P06** Wiring: `process_source` hands the smoother the hygienic scores of whatever tracker
  object it is given (a fake tracker suffices), using the decoded frame's own size; an
  out-of-frame keypoint exports visibility 0 in the CSV row, a suppressed hand exports
  confidence 0 on its 21 columns.
- **P07** Face sides: for a synthetic face in the iBUG layout with the subject's left on the
  image's right, MediaPipe 1, 2, 3 and 9 lie right of the face midline and 4, 5, 6 and 10 left of
  it; the table equals D05's.

MAIN-owned, real corpus (the corpus rerun's own output):

- **P08** Face agreement: face-derived eye points on the same side as the COCO body eyes in
  >= 95 % of frames where both are confident.
- **P09** Zero out-of-frame observations at visibility >= 0.3 among the 10 feature keypoints.
- **P10** Duplicate-hand firing census by view, reported beside D02's shipped-tree figures.

## 4. Invariant surfaces

1. `KeypointSmoother`, trackers, CSV schema (304 columns), R stage: unchanged.
2. `tests/test_mapping.py` changes in exactly the two D05 assertions.
3. Published trees untouched in this unit.

## 5. Gate identity

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <ruff check|ruff format --check|ty check|pytest>
```

Close expects the base collection + the new cases, zero skipped. Red witness = targeted
`pytest tests/<suite>` on the base tree.

## 6. Negative controls

1. `x > width` instead of `x >= width` → P01 fires.
2. `overlap <= 0.3` → P03's boundary case fires.
3. Duplicate test before out-of-frame zeroing → P04 fires.
4. The mirrored face table → P07 fires.
5. Hygiene applied after the smoother instead of before → P06 fires on the held-position row.

## 7. Verdict table

Appended at close.

## 8. Amendments

Appended as ruled.
