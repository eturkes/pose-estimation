# 2D pose weak points — catalog + repair map

Population = the superseded rtmlib-f7 generation (rtmlib `PoseTracker`, `tracking=False`,
CPU detector every 7 frames, RTMW-L on NPU, `--single-subject`, `--tracking body`): 379 clips /
193 recording families. The M2.9 generation that replaced it → *M2.9 corpus, measured* below.
Two instruments:

- **Watch pass** — every clip through the review UI, overlay at keypoint threshold 0.3, 8 frames
  spread over each clip, 97 contact sheets. "Families noted" counts families where the class was
  seen at least once (approximate: one note line can name two families).
- **Triage** — per clip, person 0, image-isotropic coordinates (`.scratch/triage.py`, → `gates.md`):
  - `whole` = frames where >= 8 of the 10 feature keypoints jump > 0.10, per 1000 observed frames.
  - `iso` = frames where 1-2 of them jump > 0.10, per 1000.
  - `alt` = median |2nd difference| / mean |1st difference| over moving frames (0 smooth, 2 alternating).
  - `flick` = hand present↔absent transitions per 100 frames.
  - `oof` = share of feature-keypoint observations with visibility >= 0.3 outside the frame.
  - `gap` = decoded frames with no exported row.
  - `lowvis` = share of feature-keypoint observations with visibility < 0.3.

Views: above 155 clips, left 93, right 131.

## Classes

| id | symptom | mechanism | prevalence | repair |
| --- | --- | --- | --- | --- |
| C01 | overlay drawn early after the first frame with no row | player indexed the landmark series by playhead | every clip: first row >= frame 2 on 379/379 (exactly 2 on 352) | review UI `397cc8a` |
| C02 | skeleton jumps to a therapist, the person across the table or a passer-by | single-subject pick = argmax of mean score over 133 keypoints, 68 facial → a small fully visible bystander beats a large truncated patient | `whole > 10`: 144/379 (right 90/131, left 49/93, above 5/155); families noted 104/193 | M2.9.1 sticky largest-box subject |
| C03 | hand or arm untracked while visible, hands blinking | pose-derived crop between detector calls loses the limb; detector every 7th frame | `flick > 5`: 143/379 (above 117/155) | M2.9.1 detector every frame + box transport |
| C04 | limbs drawn along a therapist's arm or glove | two people inside one top-down crop | families noted 22/193 | M2.9.1 in part (crop = the subject's detector box); residual = model limit |
| C05 | hallucinated occluded parts: legs under the table, far arm + hand, vertical body lines from the frame edge (above), limbs stretched to frame corners, duplicate hand | a top-down model emits all 133 keypoints, many at moderate confidence | `iso > 10`: 106/379; families noted 107/193 | M2.9.2 drops the duplicate hand; M2.9.5 drops legs everywhere + hips under the overhead camera (user ruling); far arm/hand + stretched limbs remain a model limit |
| C06 | camera-setup footage tracked (floor, ceiling, operator) | no task segmentation | families noted 13/193 | deferred (`.agent/deferred.md`) |
| C07 | confident points outside the frame | extrapolation past the crop and frame edge at visibility >= 0.3 | `oof > 0.05`: 150/379 (above 106/155) | M2.9.2 hygiene |
| C08 | rows withheld at track birth + long gaps | smoother `min_track_age = 3` on every re-keyed track | first row >= frame 2 on 379/379; `gap > 0.05`: 28/379 | M2.9.1 identity-keyed smoothing |
| C09 | finger jitter and dropout | small hands inside a 256×192 whole-person crop | fingertip jitter + alternation (`.scratch/hand_metrics.py`) | M2.9.1 halves it; second-stage models evaluated, not adopted (below) |
| C10 | face landmarks on the wrong side | `_COCO_TO_BODY_FACE` reads the mirrored iBUG layout: face-derived eye points on the opposite side in 98-99 % of 11 315 frames, mouth 83-90 % | every clip | M2.9.2 mapping fix |
| C11 | arms-only subject untracked, background people tracked | same argmax as C02; the detector box on the arms is the largest (0.13-0.32 of the frame against 0.012-0.032, 3 probed clips) | families noted 19/193 | M2.9.1 sticky largest-box subject |

Shipped-corpus distributions over 379 clips: `whole` median 0 / mean 29.4 / p90 106.8; `iso` median
0 / mean 18.4 / p90 40.6; `alt` median 1.504; `flick` median 1.81 / mean 6.66 / p90 20.7; `oof`
median 0.025 / mean 0.077; `lowvis` median 0.072 / mean 0.144.

## M2.9.1 premise, measured

Watch set = 9 events / 22 clips / first 400 frames each, accelerator recipe. v0 = the shipped
chain; v1 = `SubjectTracker` + GPU f32 detector every frame + identity-keyed smoothing.

| metric (22 clips) | v0 | v1 |
| --- | --- | --- |
| `whole` mean | 46.1 | 0 |
| `iso` mean | 21.8 | 0 |
| `alt` median | 1.475 | 0.493 |
| `flick` mean | 8.95 | 0.17 |
| hand present, mean share | 0.94 | 0.997 |
| `gap` mean | 0.037 | 0.020 |
| `oof` mean | 0.024 | 0.026 |
| wall, 9 events | 733 s | 523 s |

`oof` and `lowvis` do not move: those classes belong to M2.9.2.

## M2.9.3 hand evaluation — not adopted

Same watch set. Finger metrics (`.scratch/hand_metrics.py`, median over 22 clips): `jitter` =
fingertip step / hand extent on quasi-static frames; `alt` = fingertip alternation on moving
frames; `collapse` = present hand frames with extent < 0.25 × forearm.

| arm | pose model | jitter | alt | hand conf | collapse (worst clip) | wall, 9 events |
| --- | --- | --- | --- | --- | --- | --- |
| v0 shipped chain | RTMW-L 256×192 NPU | 0.0198 | 1.376 | 0.57 | 3.3 % | 733 s |
| v1 `SubjectTracker` | RTMW-L 256×192 NPU | 0.0102 | 0.572 | 0.60 | 11.4 % | 523-666 s |
| v2x | RTMW-X 384×288 GPU f32 | 0.0177 | 1.186 | 1.00 | 22.6 % | 847 s |
| v3h | v1 + RTMPose-m hand 256×256 on hand crops | 0.0050 | 0.405 | 0.43 | 47.7 % | 737 s |

- RTMW-X: scores run on another scale — 92.9 % of body and 96.7 % of hand observations clip to
  1.0 — which disarms every confidence gate downstream (smoother, hygiene, R). Rejected.
- Hand-crop refinement: halves visible-hand jitter, but its confidence scale is lower (0.60 →
  0.43, moving every threshold calibrated on RTMW-L), and on occluded hands the crop holds no
  hand, so the output collapses (47.7 / 35.1 / 33.1 % of hand frames in the three worst clips).
  Not adopted this pass → `.agent/deferred.md`.
- Gated refinement (second attempt, user-requested): refine only hands the whole-body model
  scores >= 0.5, keep the whole-body confidences, fall back when the refined hand's extent leaves
  0.5-2× the whole-body hand's. Against the current pipeline (tracker + hygiene, same watch set):
  jitter 0.0098 → 0.0079 (above 0.0076 → 0.0051, right 0.0112 → 0.0122), alternation 0.600 →
  0.566, collapse rose on 0 of 22 clips, 59 % of hands refined / 36 % gated / 5 % fell back, wall
  523.8 → 741.1 s (+41 %). The user's adoption bar (jitter <= 0.006, wall <= +15 %) is missed on
  both → not adopted. Gating fixed the collapse; cost and side-view jitter remain.
- What shipped for C09 is M2.9.1's effect: jitter 0.0198 → 0.0102, alternation 1.376 → 0.572.
- `collapse` is not like-for-like across v0 and v1: v1 exports a hand on every frame (above-view
  presence 1.00 against 0.85), so its population includes the occluded frames v0 dropped. The
  median clip reads 0 on every arm; the worst clips are above views.

## M2.9 corpus, measured

Population = `output/corpus-2d/` M2.9 generation (`SubjectTracker`, GPU f32 detector every frame,
RTMW-L on NPU, hygiene, seated drops, SPARC v3), 379 clips; before = the rtmlib-f7 generation,
same 379 clips paired by label. Same triage instrument; it reproduces the catalog's own before
figures exactly except `flick > 5` (144 against the catalog's 143). `flick` = max of the two hands.

| metric | before | after |
| --- | --- | --- |
| `whole > 10` (C02, C11) | 144/379 | 0/379 — 144 fixed, 0 new; mean 29.4 → 0.018 |
| `iso > 10` (C05) | 106/379 | 1/379 — 106 fixed, 1 new; mean 18.4 → 0.097 |
| `alt` median | 1.504 | 0.525 (above 0.376, left 0.586, right 0.697) |
| `oof > 0.05` (C07) | 150/379 | 4/379; mean 0.077 → 0.003 |
| `gap > 0.05` (C08) | 28/379 | 11/379 (10 above, the C06 setup footage); mean 0.019 → 0.006 |
| `flick > 5` (C03) | 144/379 (above 117/155) | 167/379 (above 71, left 34, right 62) |
| `lowvis` median | 0.072 | 0.183 — above 0.509: hips score 0 there by ruling, so not like-for-like |

Corpus predicates of contract M2.9.2 (`.agent/archive/contract-m2u92.md`, frozen at `open`):

- **P08 pass** — face-derived eye points on the body eyes' side in 263 531/264 708 point-side
  comparisons, 4 per eligible frame over 66 177 frames (0.9956, bar >= 0.95; before
  33 754/732 788 over 183 197 frames = 0.0461). Mouth-corner comparisons 0.2850 → 0.7411, not
  graded by P08.
- **P09 fail** — 3648 of 2 555 251 feature-keypoint observations at visibility >= 0.3 lie outside
  the frame (0.143 %), on 84/379 clips (56 above). Overshoot median 36 px, p99 301 px, so not
  boundary rounding; elbows 1701, shoulders 1318, wrists 395; bottom edge 1709. Hygiene zeroes
  raw out-of-frame points before the smoother, so a later stage relocated these after it — the
  bone-length projection (`constraints.BoneLengthSmoother`, moves both endpoints) or the filter;
  which one is unmeasured. Seen on the overhead view: forearm lines run past the bottom edge.
- **P10 instrument vacuous** — its census counts only frames with both hands' coordinates
  present, and a hand whose mean score falls to <= 0.1 (`mapping._HAND_PRESENCE_THRESHOLD`)
  exports no coordinates, so a suppressed hand never enters it: 0 on both generations.

Hand blink-outs (present → absent on consecutive frames) per 100 rows, before → after: above
12.92 → 3.10, left 5.75 → 2.63, right 0.46 → 3.27; corpus 18 744 → 10 074. On the right view,
46 % of blink-outs follow a frame where the blinking hand sat on or near the other hand (median
point distance < 0.6 of hand extent); the attribution reads the frame before the blink, and no
counter records D02 firing (P10 above), so the cause is inferred, not measured. Watched on two
side-view clips: the blinker is the occluded far hand, hallucinated on or under the visible one,
present on some frames and absent on others; the superseded generation drew it on every frame,
beside stretched limbs and legs on the table. Inference: the right-view rise is residual C05
hallucination toggled by D02 rather than a tracking regression — a per-frame D02 counter decides
it (`.agent/deferred.md`). A far hand duplicating the near one can also survive the rule
outright (hand-to-wrist distance up to 9.8 forearm lengths on one clip).

## M2.10 messy footage, measured

The footage is hand-held and crowded. Over all 379 clips × 8 sampled positions (GPU detector,
`.scratch/messy_probe.py`), 186 clips show two or more people on at least half the positions
(above 21/155, left 58/93, right 107/131), and 142 clips have a second box at least half the size
of the largest. Labelled watch set = 81 clips / 36 families / 14 subjects, 12 frames per clip =
972 frames (MAIN + `general-purpose-1/-2/-3`); 20 clips tuned thresholds, 61 held them out.

| class | measured on the labels | M2.10 response |
| --- | --- | --- |
| hallucinated hand (C05) | 357 of 1 494 decided drawn hands sit where no hand is (24 %; 195 more marked unsure), mostly the occluded far hand under the overhead camera | M2.10.1 hand presence gate: held-out 191/212 removed, 28/754 real hands lost (3.7 %; per subject median 0, max 12.3 %) |
| setup / handling footage (C06) | 44 of 972 sampled frames; dense labels on 13 clips: 6 836 setup frames | M2.10.2 task span: setup excluded 0.967 (dense) / 41/44 (sparse); task lost 0.011 (dense) / 1/855 (sparse), + 4 frames the one-span rule drops by design |
| hand-held shake | camera motion at 0.5 s is 2.5 % of wrist motion at the median clip, ≥ 50 % on 10 overhead clips; per-frame it reaches wrist speed | M2.10.2 compensation: on known synthetic shake (14 clips) SPARC abs error median 0.3-1.9 uncompensated → 0.01-0.07 compensated |
| wrong person (C02, C11) | patient's box chosen on 459/473 visible labelled task frames (0.970); body on another person or mixed on 17 labelled task frames in 9 clips (3 more in setup footage); the two worst clips (4 and 3 frames) show limb lines drawn across to another arm and a therapist leaning over the patient | none this unit → `.agent/deferred.md` |
| other person's hand drawn as the patient's (C04) | 21 of 1 494 decided drawn hands | the gate removes 7/10 held-out; mask-conditioned pose → `.agent/deferred.md` |
| inconsistent angles | 2D image-plane elbow flexion disagrees across synced views by median 60° on 14 series, no better than time-shuffled pairing (ratio 1.00) | SAM 3D Body pilot (not adopted; user: pilot more first): 5.9° synced vs 12.0° shuffled (ratio 0.49) |

The label vocabulary, graders and their regeneration paths → `.claude/rules/gates.md`
§ *Scratch validators pending port*.
